"""Where the container's rim is, from the wrist camera: the grasp point of the take-out.

grab_container's wrist detection (feature/12-grab-container, Angela Dong: sam3.py select_lip_mask,
grasp.py five_closest_lip_points and the mask-agreement streak of detect_grasp), on this capability's node:

  1. SAM3 on the wrist colour image, prompt `whole_prompt`: the container with its sleeve, the best-scoring
     object's mask;
  2. SAM3 again on that same frame (republished on SNAPSHOT_TOPIC, since a combined call's label image holds one
     object per pixel and the rim lies inside the container), prompt `rim_prompt`: the rim, a wide, flat mask
     mostly inside the container's (select_rim_mask);
  3. until `frames` consecutive frames have both masks overlapping every earlier one of the streak by at least
     `mask_iou` (intersection over union), at most timeout_s;
  4. on each frame of that streak: the mean of the RIM_POINTS nearest depths inside rim AND container, in
     arm_mount (TF at the frame's stamp); their mean = the grasp point.

The wrist camera's stereo returns nothing nearer than ~0.22 m (it piles up at that floor): start the take-out
where the rim is ~0.35-0.5 m from it. Read-only: nothing here commands the arm.
"""

import threading
import time
from collections import deque

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

SAM3_SERVICE = "/perception/sam3/detect"
SNAPSHOT_TOPIC = "/microwave_place/wrist_query_image"  # the frozen frame the rim prompt runs on
DEPTH_RANGE = (0.2, 2.0)  # m, the wrist depth used (grab_container DEPTH_MIN / DEPTH_MAX)
RIM_POINTS = 5  # the grasp point: the mean of this many nearest rim depths
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


def select_rim_mask(response, prompt, labels, container):
    """grab_container's select_lip_mask: of the rim objects, a broad, shallow one mostly (85%) inside the container
    mask (span >= 60% of its width, height <= 65% of its height), its largest connected part; None if none."""
    cy, cx = np.nonzero(container)
    width, height_total = np.ptp(cx) + 1, np.ptp(cy) + 1
    choices = []
    for obj in response.detections.objects:
        if obj.perception_class != prompt or obj.state != 0 or not obj.mask_id:
            continue
        full = labels == obj.mask_id
        inside = full & container
        if inside.sum() < 9 or inside.sum() < 0.85 * full.sum():
            continue  # e.g. the turntable reaching outside the container
        parts, count = ndimage.label(full, structure=np.ones((3, 3)))
        if not count:
            continue
        sizes = np.bincount(parts.ravel())
        sizes[0] = 0
        part = parts == int(np.argmax(sizes))
        ys, xs = np.nonzero(part)
        span = (xs.max() - xs.min() + 1) / width
        height = (ys.max() - ys.min() + 1) / height_total
        if span < 0.6 or height > 0.65:
            continue  # a narrow neck or the whole body
        choices.append((span - 0.1 * height + 0.1 * float(obj.score), part))
    return max(choices, key=lambda c: c[0])[1] if choices else None


def rim_point(depth_m, rim, container, K, T):
    """grab_container's five_closest_lip_points: the mean of the RIM_POINTS smallest depths inside rim AND
    container, in arm_mount (T: optical -> arm_mount). (point, info) or the reason it can't."""
    valid = rim & container & np.isfinite(depth_m) & (depth_m > DEPTH_RANGE[0]) & (depth_m < DEPTH_RANGE[1])
    ys, xs = np.nonzero(valid)
    if len(xs) < RIM_POINTS:
        return None, f"{len(xs)} rim pixels with depth (need {RIM_POINTS})"
    zs = depth_m[ys, xs].astype(float)
    pick = np.argsort(zs, kind="stable")[:RIM_POINTS]
    xs, ys, zs = xs[pick], ys[pick], zs[pick]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    cam = np.column_stack(((xs - cx) * zs / fx, (ys - cy) * zs / fy, zs))
    pts = cam @ T[:3, :3].T + T[:3, 3]
    point = pts.mean(axis=0)
    return point, dict(
        depths_m=np.round(zs, 4).tolist(),
        pixels_uv=np.column_stack((xs, ys)).tolist(),
        spread_m=float(np.max(np.linalg.norm(pts - point, axis=1))),
        rim_depth_px=int(valid.sum()),
        rim_px=int(rim.sum()),
    )


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
        self._snapshot = engine.create_action_publisher(Image, SNAPSHOT_TOPIC, 1)
        self._tf = engine.create_action_tf_buffer()
        self._sam3 = engine._action_scope().create_client(Detect, SAM3_SERVICE)
        if not self._sam3.wait_for_service(timeout_sec=SAM3_WAIT_S):
            raise NotDetected(f"SAM3 service {SAM3_SERVICE} not available within {SAM3_WAIT_S:.0f} s")

    def _call(self, prompt, image_topic, threshold, deadline, republish=None):
        fut = self._sam3.call_async(
            self._Detect.Request(prompts=[prompt], image_topic=image_topic, score_threshold=threshold)
        )
        while not fut.done() and time.monotonic() < deadline:
            if republish is not None:
                self._snapshot.publish(republish)
            time.sleep(0.03 if republish is not None else 0.01)
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

    def shot(self, whole_prompt, rim_prompt, threshold, deadline):
        """One frame: (dict(container, rim, depth_m, color_msg, depth_msg, score), None), or (None, why), or
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
            return None, f"SAM3 found no {whole_prompt!r}"
        container, score = best
        r2 = self._call(rim_prompt, SNAPSHOT_TOPIC, threshold, deadline, republish=color)
        if r2 is None:
            return None, None
        if not r2.success or _stamp(r2.detections) != _stamp(color):
            return None, "SAM3's rim answer is not on the frozen frame"
        rim = select_rim_mask(
            r2, rim_prompt, self._bridge.imgmsg_to_cv2(r2.detections.label_image, "passthrough"), container
        )
        if rim is None:
            return None, "no rim on the container"
        depth_m = self._bridge.imgmsg_to_cv2(depth, "passthrough").astype(np.float32)
        if depth.encoding in ("16UC1", "mono16"):
            depth_m /= 1000.0
        if depth_m.shape != container.shape:
            raise NotDetected(f"wrist depth {depth_m.shape[::-1]} px, SAM3's masks {container.shape[::-1]}")
        return dict(container=container, rim=rim, depth_m=depth_m, depth_msg=depth, score=score), None

    def K(self):
        if self._info is None:
            raise NotDetected("no wrist camera_info")
        return np.array(self._info.k, float).reshape(3, 3)


def detect_rim(camera, whole_prompt, rim_prompt, threshold, frames, mask_iou_min, timeout_s, log, checkpoint):
    """detect_grasp's rule: SAM3 frames until `frames` consecutive ones have container and rim masks overlapping
    every earlier one of the streak by >= mask_iou_min; then the rim point on the last. Returns dict(point, info,
    frames, attempts, rejected). Raises NotDetected with every reason a frame was rejected."""
    deadline = time.monotonic() + timeout_s
    streak, rejected, attempts = [], {}, 0
    while time.monotonic() < deadline:
        checkpoint()
        attempts += 1
        shot, why = camera.shot(whole_prompt, rim_prompt, threshold, deadline)
        if shot is None:
            if why is None:
                break
            rejected[why] = rejected.get(why, 0) + 1
            streak = []
            continue
        ious = [(mask_iou(shot["container"], s["container"]), mask_iou(shot["rim"], s["rim"])) for s in streak]
        if any(min(p) < mask_iou_min for p in ious):
            rejected["masks disagree with the streak"] = rejected.get("masks disagree with the streak", 0) + 1
            streak = []
        streak.append(shot)
        if len(streak) < frames:
            continue
        # the rim point on every frame of the streak (each with TF at its own stamp), averaged: the nearest depths
        # come in 2-3 cm steps at ~0.5 m, so one frame's is that coarse
        points, infos, why = [], [], None
        for s_ in streak:
            try:
                T = camera._pose(s_["depth_msg"])
            except Exception as e:  # noqa: BLE001 -- tf2's lookup errors: the streak is not used
                why = f"no TF at a frame ({type(e).__name__})"
                break
            p_, info_ = rim_point(s_["depth_m"], s_["rim"], s_["container"], camera.K(), T)
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
        iou_min = min(
            min(mask_iou(a["container"], b["container"]), mask_iou(a["rim"], b["rim"]))
            for i, a in enumerate(streak)
            for b in streak[i + 1 :]
        )
        log(
            f"rim: {frames} frames agree (mask IoU >= {iou_min:.3f}) in {attempts} tries; the {RIM_POINTS} nearest rim "
            f"depths {info['depths_m']} m; per frame {info['frame_points']} -> mean {np.round(point, 3).tolist()} "
            f"(frames within {spread * 100:.1f} cm of it)"
        )
        return dict(point=point, info=info, iou_min=iou_min, attempts=attempts, rejected=rejected)
    raise NotDetected(
        f"the rim not seen in {frames} agreeing frames within {timeout_s:.0f} s ({attempts} tries; rejected: "
        + ("; ".join(f"{k} x{n}" for k, n in rejected.items()) or "none")
        + ")"
    )
