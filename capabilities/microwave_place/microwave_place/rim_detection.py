"""Take-out grasp point from the wrist camera's whole tupperware mask.

Single SAM3 query per frame. Keep the existing mask-IoU agreement streak and
average 50 nearest locally consistent depths per frame, then the points across frames.
Read-only: this module never commands the arm. The filename is historical.
"""

import threading
import time
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

SAM3_SERVICE = "/perception/sam3/detect"
OVERLAY_TOPIC = "/microwave_place/sam3_overlay"  # RGB debug image for Foxglove
DEPTH_RANGE = (0.2, 2.0)  # m, the wrist depth used (grab_container DEPTH_MIN / DEPTH_MAX)
NEAREST_POINTS = 50  # the grasp point: the mean of this many nearest container depths
DEPTH_WINDOW = 5  # local masked neighborhood, pixels per side
DEPTH_MIN_NEIGHBORS = 5
DEPTH_OUTLIER_FLOOR_M = 0.01  # minimum allowed deviation from the local median
DEPTH_OUTLIER_MAD_SCALE = 3.0 * 1.4826  # three robust standard deviations
PAIR_MAX_NS = 100_000_000  # SAM3's frame and the buffered colour and depth: within 100 ms
KEEP = 180  # colour and depth frames buffered (12 s at 15 Hz)
SAM3_WAIT_S = 2.0


class NotDetected(Exception):
    pass


def _stamp(msg):
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def mask_iou(a, b):
    if a.shape != b.shape:
        return 0.0
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 0.0


def best_mask(response, prompt, labels):
    """The best-scoring visible object for `prompt`: (its mask, score), or None."""
    objs = [o for o in response.detections.objects if o.perception_class == prompt and o.state == 0 and o.mask_id]
    if not objs:
        return None
    best = max(objs, key=lambda o: o.score)
    mask = labels == best.mask_id
    return (mask, float(best.score)) if mask.any() else None


def select_depth_pixels(depth_m, container):
    """Nearest samples after a masked 5x5 median/MAD depth consistency check.

    Reject sparse neighborhoods and isolated depth spikes before ranking by depth.
    Real depth variation across the object is retained by using local statistics.
    """
    valid = container & np.isfinite(depth_m) & (depth_m > DEPTH_RANGE[0]) & (depth_m < DEPTH_RANGE[1])
    ys, xs = np.nonzero(valid)
    if not len(xs):
        return ys, xs, dict(valid_depth_px=0, depth_outliers_px=0)
    image = np.where(valid, depth_m, np.nan)
    pad = DEPTH_WINDOW // 2
    windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(image, pad, constant_values=np.nan), (DEPTH_WINDOW, DEPTH_WINDOW)
    )[ys, xs].reshape(len(xs), -1)
    median = np.nanmedian(windows, axis=1)
    mad = np.nanmedian(np.abs(windows - median[:, None]), axis=1)
    tolerance = np.maximum(DEPTH_OUTLIER_FLOOR_M, DEPTH_OUTLIER_MAD_SCALE * mad)
    keep = (np.isfinite(windows).sum(axis=1) >= DEPTH_MIN_NEIGHBORS) & (np.abs(depth_m[ys, xs] - median) <= tolerance)
    stats = dict(valid_depth_px=int(len(xs)), depth_outliers_px=int((~keep).sum()))
    ys, xs = ys[keep], xs[keep]
    pick = np.argsort(depth_m[ys, xs], kind="stable")[:NEAREST_POINTS]
    return ys[pick], xs[pick], stats


def container_point(depth_m, container, K, T):
    """Mean of 50 nearest locally consistent container depths, in arm_mount."""
    ys, xs, stats = select_depth_pixels(depth_m, container)
    if len(xs) < NEAREST_POINTS:
        return None, f"{len(xs)} filtered container pixels with depth (need {NEAREST_POINTS})"
    zs = depth_m[ys, xs].astype(float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    cam = np.column_stack(((xs - cx) * zs / fx, (ys - cy) * zs / fy, zs))
    pts = cam @ T[:3, :3].T + T[:3, 3]
    point = pts.mean(axis=0)
    return point, dict(
        depths_m=np.round(zs, 4).tolist(),
        pixels_uv=np.column_stack((xs, ys)).tolist(),
        spread_m=float(np.max(np.linalg.norm(pts - point, axis=1))),
        container_depth_px=stats["valid_depth_px"],
        depth_outliers_px=stats["depth_outliers_px"],
        container_px=int(container.sum()),
    )


def overlay_image(rgb, container, pixels, status):
    """RGB debug copy: cyan container mask and yellow nearest-depth samples."""
    import cv2

    out = rgb.copy()
    if container is not None:
        out[container] = (0.5 * out[container] + 0.5 * np.array([0, 200, 255])).astype(np.uint8)
    for u, v in pixels:
        cv2.circle(out, (int(u), int(v)), 2, (255, 255, 0), -1)
    out[:44] = (out[:44].astype(float) * 0.25).astype(np.uint8)
    cv2.putText(
        out,
        "cyan: container  yellow: nearest depth",
        (3, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(out, status, (3, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class WristCamera:
    """The wrist camera inside a running action of `engine`: buffered colour and depth, camera_info, TF, and the
    SAM3 client. Its subscriptions, publisher and client end with the action."""

    def __init__(self, engine, frame, topics):
        from cv_bridge import CvBridge
        from rammp_perception_interfaces.srv import Detect
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image

        self.frame, self.color_topic = frame, topics[0]
        self._Detect, self._bridge, self._lock = Detect, CvBridge(), threading.Lock()
        self._colors, self._depths, self._info = deque(maxlen=KEEP), deque(maxlen=KEEP), None

        def keep(buf):
            def cb(m):
                with self._lock:
                    buf.append(m)

            return cb

        def keep_info(m):
            self._info = m

        engine.create_action_subscription(Image, topics[0], keep(self._colors), qos_profile_sensor_data)
        engine.create_action_subscription(Image, topics[1], keep(self._depths), qos_profile_sensor_data)
        engine.create_action_subscription(CameraInfo, topics[2], keep_info, qos_profile_sensor_data)
        self._overlay = engine.sam3_overlay
        self._log = engine.get_logger()
        self._tf = engine.create_action_tf_buffer()
        self._sam3 = engine._action_scope().create_client(Detect, SAM3_SERVICE)
        if not self._sam3.wait_for_service(timeout_sec=SAM3_WAIT_S):
            raise NotDetected(f"SAM3 service {SAM3_SERVICE} not available within {SAM3_WAIT_S:.0f} s")

    def _call(self, prompt, image_topic, threshold, deadline):
        fut = self._sam3.call_async(
            self._Detect.Request(prompts=[prompt], image_topic=image_topic, score_threshold=threshold)
        )
        while not fut.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not fut.done():
            self._sam3.remove_pending_request(fut)
            return None
        return fut.result()

    def _pair(self, stamp):
        with self._lock:
            colors, depths = list(self._colors), list(self._depths)
        if not colors or not depths:
            return None
        color = min(colors, key=lambda m: abs(_stamp(m) - stamp))
        depth = min(depths, key=lambda m: abs(_stamp(m) - stamp))
        if max(abs(_stamp(color) - stamp), abs(_stamp(depth) - stamp)) > PAIR_MAX_NS:
            return None
        return color, depth

    def _pose(self, msg):
        from rclpy.duration import Duration
        from rclpy.time import Time

        t = self._tf.lookup_transform(
            self.frame, msg.header.frame_id, Time.from_msg(msg.header.stamp), timeout=Duration(seconds=0.5)
        ).transform
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]).as_matrix()
        T[:3, 3] = [t.translation.x, t.translation.y, t.translation.z]
        return T

    def _publish_overlay(self, color, container=None, pixels=(), status=""):
        """Diagnostics must not change detection or make a valid action fail."""
        try:
            rgb = self._bridge.imgmsg_to_cv2(color, "rgb8")
            image = overlay_image(rgb, container, pixels, status)
            message = self._bridge.cv2_to_imgmsg(image, encoding="rgb8")
            message.header = color.header
            self._overlay.publish(message)
        except Exception as e:  # noqa: BLE001 -- optional visualization cannot interrupt detection
            self._log.warning(f"SAM3 overlay unavailable: {e}")

    def shot(self, whole_prompt, threshold, deadline):
        """One frame: (dict(container, depth_m, depth_msg, score), None), or (None, why), or
        (None, None) when `deadline` passed first."""
        r1 = self._call(whole_prompt, self.color_topic, threshold, deadline)
        if r1 is None:
            return None, None
        if not r1.success:
            return None, f"SAM3: {r1.message}"
        stamp = _stamp(r1.detections)
        pair = self._pair(stamp)
        if pair is None:
            return None, "no buffered wrist colour and depth at SAM3's frame"
        color, depth = pair
        labels = self._bridge.imgmsg_to_cv2(r1.detections.label_image, "passthrough")
        best = best_mask(r1, whole_prompt, labels)
        if best is None:
            self._publish_overlay(color, status="No container mask")
            return None, f"SAM3 found no {whole_prompt!r}"
        container, score = best
        depth_m = self._bridge.imgmsg_to_cv2(depth, "passthrough").astype(np.float32)
        if depth.encoding in ("16UC1", "mono16"):
            depth_m /= 1000.0
        if depth_m.shape != container.shape:
            raise NotDetected(f"wrist depth {depth_m.shape[::-1]} px, SAM3's masks {container.shape[::-1]}")
        ys, xs, stats = select_depth_pixels(depth_m, container)
        pixels = np.column_stack((xs, ys)).tolist()
        self._publish_overlay(
            color, container, pixels, f"Samples {len(pixels)}/{NEAREST_POINTS}; rejected {stats['depth_outliers_px']}"
        )
        return dict(container=container, depth_m=depth_m, depth_msg=depth, score=score), None

    def K(self):
        if self._info is None:
            raise NotDetected("no wrist camera_info")
        return np.array(self._info.k, float).reshape(3, 3)


def detect_container(camera, whole_prompt, threshold, frames, mask_iou_min, timeout_s, log, checkpoint):
    """Require pairwise container-mask agreement across consecutive frames.

    Average 50 nearest locally consistent points per frame, transformed at each frame's
    timestamp, then average those frame points. No new point-distance gate.
    """
    deadline = time.monotonic() + timeout_s
    streak, rejected, attempts = [], {}, 0
    while time.monotonic() < deadline:
        checkpoint()
        attempts += 1
        shot, why = camera.shot(whole_prompt, threshold, deadline)
        if shot is None:
            if why is None:
                break
            rejected[why] = rejected.get(why, 0) + 1
            streak = []
            continue
        ious = [mask_iou(shot["container"], s["container"]) for s in streak]
        if any(iou < mask_iou_min for iou in ious):
            rejected["masks disagree with the streak"] = rejected.get("masks disagree with the streak", 0) + 1
            streak = []
        streak.append(shot)
        if len(streak) < frames:
            continue
        # the container point on every frame of the streak (each with TF at its own stamp), averaged: the nearest depths
        # come in 2-3 cm steps at ~0.5 m, so one frame's is that coarse
        points, infos, why = [], [], None
        for s_ in streak:
            try:
                T = camera._pose(s_["depth_msg"])
            except Exception as e:  # noqa: BLE001 -- tf2's lookup errors: the streak is not used
                why = f"no TF at a frame ({type(e).__name__})"
                break
            p_, info_ = container_point(s_["depth_m"], s_["container"], camera.K(), T)
            if p_ is None:
                why = info_
                break
            points.append(p_)
            infos.append(info_)
        if why is not None:
            rejected[why] = rejected.get(why, 0) + 1
            streak = []
            continue
        point = np.mean(points, axis=0)
        spread = float(np.max(np.linalg.norm(np.asarray(points) - point, axis=1)))
        info = dict(
            infos[-1],
            depths_m=[i["depths_m"] for i in infos],
            frame_points=np.round(points, 4).tolist(),
            frames_spread_m=spread,
        )
        iou_min = min(mask_iou(a["container"], b["container"]) for i, a in enumerate(streak) for b in streak[i + 1 :])
        log(
            f"container: {frames} frames agree (mask IoU >= {iou_min:.3f}) in {attempts} tries; "
            f"the {NEAREST_POINTS} nearest container "
            f"depths {info['depths_m']} m; per frame {info['frame_points']} -> mean {np.round(point, 3).tolist()} "
            f"(frames within {spread * 100:.1f} cm of it)"
        )
        return dict(point=point, info=info, iou_min=iou_min, attempts=attempts, rejected=rejected)
    raise NotDetected(
        f"the container not seen in {frames} agreeing frames within {timeout_s:.0f} s ({attempts} tries; rejected: "
        + ("; ".join(f"{k} x{n}" for k, n in rejected.items()) or "none")
        + ")"
    )
