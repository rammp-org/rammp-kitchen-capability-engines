"""The wrist camera and SAM3 detection for tap-to-pick, on top of microwave_place's rim_detection.

microwave_place/rim_detection.py is used UNCHANGED: its constants, best_mask, mask_iou,
container_point, select_depth_pixels, overlay_image and NotDetected. What the fridge needs on top of
it lives here instead, so microwave_place stays exactly as it is:

  consistent_depth_pixels  rim_detection.select_depth_pixels' depth filter, every pixel it keeps
                           (select_depth_pixels then takes the NEAREST_POINTS nearest): the fridge's
                           grasp point needs the container's whole near face, not 50 points.
                           test_wrist_detection.py checks it against select_depth_pixels;
  WristCamera              as rim_detection.WristCamera (buffered colour and depth, camera_info,
                           TF, the SAM3 client), plus: a buffer length (teleop keeps it open for
                           minutes), non-blocking SAM3 requests for teleop, the newest colour frame
                           (the UI overlay), and `select` on shot(): which of SAM3's objects, given
                           all of them and the frame's depth (default: the best-scoring);
  SingleTarget             a `select` for a prompt that must name exactly one object: several in
                           view refuses at once (nothing is guessed); from then on the one
                           overlapping the first frame's most (it cannot switch objects);
  detect_container         as rim_detection.detect_container, plus `select` and `point(depth_m,
                           mask, K, T)` (the grasp point of one frame).

Read-only: nothing here commands the arm.
"""

import threading
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
from microwave_place import rim_detection as rd
from scipy.spatial.transform import Rotation

NotDetected = rd.NotDetected
NEAREST_POINTS = rd.NEAREST_POINTS
best_mask = rd.best_mask
mask_iou = rd.mask_iou
container_point = rd.container_point
stamp_ns = rd._stamp
STATE_VISIBLE = 0


def candidates(response, prompt, labels):
    """SAM3's visible `prompt` objects: [(mask, score)], best-scoring first."""
    out = []
    for o in response.detections.objects:
        if o.perception_class != prompt or o.state != STATE_VISIBLE or not o.mask_id:
            continue
        mask = labels == o.mask_id
        if mask.any():
            out.append((mask, float(o.score)))
    return sorted(out, key=lambda c: -c[1])


def best_of(cands, frame):
    """The default `select`: the best-scoring candidate (rim_detection.best_mask's choice)."""
    return max(cands, key=lambda c: c[1]) if cands else None


class SingleTarget:
    """A WristCamera.shot `select` for a `prompt` that must name exactly one object. On the first
    frame there must be exactly one candidate: several raises NotDetected at once (nothing is
    guessed: say which, e.g. a tracked id). From then on it takes the candidate overlapping the
    first frame's most (>= min_iou), else none: the streak waits for it rather than switching."""

    def __init__(self, prompt, min_iou):
        self.prompt, self.min_iou = prompt, float(min_iou)
        self.prev = None

    def __call__(self, cands, frame=None):
        if not cands:
            return None
        if self.prev is None:
            if len(cands) > 1:
                raise NotDetected(
                    f"{len(cands)} {self.prompt!r} in view: it must name exactly one "
                    f"(pick a tracked id from ~/in_view instead)"
                )
            self.prev = cands[0][0]
            return cands[0]
        ious = [mask_iou(m, self.prev) for m, _ in cands]
        k = int(np.argmax(ious))
        if ious[k] < self.min_iou:
            return None
        self.prev = cands[k][0]
        return cands[k]


def consistent_depth_pixels(depth_m, container):
    """Every container pixel passing rim_detection's masked 5x5 median/MAD depth consistency
    check (its constants): (ys, xs, stats). rim_detection.select_depth_pixels is this, then the
    NEAREST_POINTS nearest."""
    valid = (
        container
        & np.isfinite(depth_m)
        & (depth_m > rd.DEPTH_RANGE[0])
        & (depth_m < rd.DEPTH_RANGE[1])
    )
    ys, xs = np.nonzero(valid)
    if not len(xs):
        return ys, xs, dict(valid_depth_px=0, depth_outliers_px=0)
    image = np.where(valid, depth_m, np.nan)
    pad = rd.DEPTH_WINDOW // 2
    windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(image, pad, constant_values=np.nan), (rd.DEPTH_WINDOW, rd.DEPTH_WINDOW)
    )[ys, xs].reshape(len(xs), -1)
    median = np.nanmedian(windows, axis=1)
    mad = np.nanmedian(np.abs(windows - median[:, None]), axis=1)
    tolerance = np.maximum(rd.DEPTH_OUTLIER_FLOOR_M, rd.DEPTH_OUTLIER_MAD_SCALE * mad)
    keep = (np.isfinite(windows).sum(axis=1) >= rd.DEPTH_MIN_NEIGHBORS) & (
        np.abs(depth_m[ys, xs] - median) <= tolerance
    )
    stats = dict(valid_depth_px=int(len(xs)), depth_outliers_px=int((~keep).sum()))
    return ys[keep], xs[keep], stats


class WristCamera:
    """The wrist camera inside a running action of `engine` (rim_detection.WristCamera's, plus the
    above). Its subscriptions and SAM3 client end with the action. Debug images go to
    engine.sam3_overlay. NotDetected when SAM3 is not up within rim_detection.SAM3_WAIT_S."""

    def __init__(self, engine, frame, topics, keep_frames=rd.KEEP):
        from cv_bridge import CvBridge
        from rammp_perception_interfaces.srv import Detect
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image

        self.frame, self.color_topic = frame, topics[0]
        self._Detect, self._bridge, self._lock = Detect, CvBridge(), threading.Lock()
        self._colors, self._depths = deque(maxlen=keep_frames), deque(maxlen=keep_frames)
        self._info = None
        self.last_call_s = 0.0  # how long the last SAM3 call in shot() took

        def buffer_into(buf):
            def cb(m):
                with self._lock:
                    buf.append(m)

            return cb

        def keep_info(m):
            self._info = m

        sub = engine.create_action_subscription
        sub(Image, topics[0], buffer_into(self._colors), qos_profile_sensor_data)
        sub(Image, topics[1], buffer_into(self._depths), qos_profile_sensor_data)
        sub(CameraInfo, topics[2], keep_info, qos_profile_sensor_data)
        self._overlay = engine.sam3_overlay
        self._log = engine.get_logger()
        self._tf = engine.create_action_tf_buffer()
        # as rim_detection.WristCamera: the engine's action scope owns (and destroys) the client
        self._sam3 = engine._action_scope().create_client(Detect, rd.SAM3_SERVICE)
        if not self._sam3.wait_for_service(timeout_sec=rd.SAM3_WAIT_S):
            raise NotDetected(
                f"SAM3 service {rd.SAM3_SERVICE} not available within {rd.SAM3_WAIT_S:.0f} s"
            )

    # ------------------------------------------------------------------ for teleop (non-blocking)

    def detect_async(self, prompt, threshold):
        """One SAM3 request on the colour topic, not waited for: its future (poll .done(); drop a
        late one with cancel_request())."""
        return self._sam3.call_async(
            self._Detect.Request(
                prompts=[prompt], image_topic=self.color_topic, score_threshold=threshold
            )
        )

    def cancel_request(self, fut):
        self._sam3.remove_pending_request(fut)

    def latest_color(self):
        """The newest buffered colour frame, or None."""
        with self._lock:
            return self._colors[-1] if self._colors else None

    def labels(self, detections):
        """A detections message's label image as an array (compare with an object's mask_id)."""
        return self._bridge.imgmsg_to_cv2(detections.label_image, "passthrough")

    def buffered(self):
        """What the buffers hold: dict(colors, depths, oldest_ns, newest_ns) (stamps, ns)."""
        with self._lock:
            stamps = [stamp_ns(m) for m in (*self._colors, *self._depths)]
            n_color, n_depth = len(self._colors), len(self._depths)
        return dict(
            colors=n_color,
            depths=n_depth,
            oldest_ns=min(stamps, default=0),
            newest_ns=max(stamps, default=0),
        )

    def pair(self, stamp):
        """(colour, depth) buffered within rim_detection.PAIR_MAX_NS of `stamp` (ns), or None."""
        with self._lock:
            colors, depths = list(self._colors), list(self._depths)
        if not colors or not depths:
            return None
        color = min(colors, key=lambda m: abs(stamp_ns(m) - stamp))
        depth = min(depths, key=lambda m: abs(stamp_ns(m) - stamp))
        if max(abs(stamp_ns(color) - stamp), abs(stamp_ns(depth) - stamp)) > rd.PAIR_MAX_NS:
            return None
        return color, depth

    def depth_array(self, depth_msg) -> np.ndarray:
        """A depth image in metres (float32)."""
        depth_m = self._bridge.imgmsg_to_cv2(depth_msg, "passthrough").astype(np.float32)
        if depth_msg.encoding in ("16UC1", "mono16"):
            depth_m /= 1000.0
        return depth_m

    def frame_at(self, stamp):
        """The buffered depth at `stamp` (ns) as a `select` frame (depth_m, depth_msg, K(),
        pose(msg)), or None (e.g. for a tracked frame: its masks are SAM3's at that stamp)."""
        pair = self.pair(stamp)
        if pair is None:
            return None
        return SimpleNamespace(
            depth_m=self.depth_array(pair[1]), depth_msg=pair[1], K=self.K, pose=self.pose
        )

    def pose(self, msg):
        """T (4x4) from msg's frame to `frame` at msg's stamp. Raises tf2's lookup errors."""
        from rclpy.duration import Duration
        from rclpy.time import Time

        t = self._tf.lookup_transform(
            self.frame,
            msg.header.frame_id,
            Time.from_msg(msg.header.stamp),
            timeout=Duration(seconds=0.5),
        ).transform
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(
            [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]
        ).as_matrix()
        T[:3, 3] = [t.translation.x, t.translation.y, t.translation.z]
        return T

    def K(self):
        if self._info is None:
            raise NotDetected("no wrist camera_info")
        return np.array(self._info.k, float).reshape(3, 3)

    # ------------------------------------------------------------------ for the pick (blocking)

    def _why_unpaired(self, stamp):
        """Why SAM3's frame (stamp, ns) has no buffered colour + depth pair: a reject reason."""
        b = self.buffered()
        if not b["colors"] or not b["depths"]:
            return (
                f"no wrist frames received (colour {b['colors']}, depth {b['depths']}): check "
                f"wrist_topics"
            )
        if not stamp:
            return "SAM3's detections carry no stamp"
        if stamp < b["oldest_ns"]:
            return (
                f"SAM3's frame is older than the buffer (by "
                f"{(b['oldest_ns'] - stamp) / 1e9:.2f} s; it holds "
                f"{(b['newest_ns'] - b['oldest_ns']) / 1e9:.2f} s): raise wrist_buffer_frames"
            )
        if stamp > b["newest_ns"]:
            return (
                f"SAM3's frame is {(stamp - b['newest_ns']) / 1e9:.2f} s newer than any "
                f"buffered: clocks differ, or the wrist topics lag"
            )
        return (
            f"no colour AND depth within {rd.PAIR_MAX_NS / 1e6:.0f} ms of SAM3's frame (depth "
            f"and colour out of step?)"
        )

    def _call(self, prompt, threshold, deadline):
        fut = self.detect_async(prompt, threshold)
        while not fut.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not fut.done():
            self.cancel_request(fut)
            return None
        return fut.result()

    def _publish_overlay(self, color, container=None, pixels=(), status=""):
        """Diagnostics must not change detection or make a valid action fail."""
        try:
            rgb = self._bridge.imgmsg_to_cv2(color, "rgb8")
            message = self._bridge.cv2_to_imgmsg(
                rd.overlay_image(rgb, container, pixels, status), encoding="rgb8"
            )
            message.header = color.header
            self._overlay.publish(message)
        except Exception as e:  # noqa: BLE001 -- optional visualization
            self._log.warning(f"SAM3 overlay unavailable: {e}")

    def shot(self, prompt, threshold, deadline, select=None):
        """One frame: (dict(container, depth_m, depth_msg, score), None), or (None, why), or
        (None, None) when `deadline` passed first. `select(cands, frame)` -> (mask, score) or
        None: which of SAM3's objects (cands: candidates(); frame: depth_m, depth_msg, K(),
        pose(msg)); default best_of, the best-scoring."""
        t0 = time.monotonic()
        r = self._call(prompt, threshold, deadline)
        self.last_call_s = time.monotonic() - t0
        if r is None:
            return None, None
        if not r.success:
            return None, f"SAM3: {r.message}"
        stamp = stamp_ns(r.detections)
        pair = self.pair(stamp)
        if pair is None:
            return None, self._why_unpaired(stamp)
        color, depth = pair
        labels = self.labels(r.detections)
        cands = candidates(r, prompt, labels)
        if not cands:
            self._publish_overlay(color, status="No container mask")
            return None, f"SAM3 found no {prompt!r}"
        depth_m = self.depth_array(depth)
        if depth_m.shape != labels.shape:
            raise NotDetected(
                f"wrist depth {depth_m.shape[::-1]} px, SAM3's masks {labels.shape[::-1]}"
            )
        frame = SimpleNamespace(depth_m=depth_m, depth_msg=depth, K=self.K, pose=self.pose)
        best = (select or best_of)(cands, frame)
        if best is None:
            self._publish_overlay(color, status=f"{len(cands)} masks, none chosen")
            return None, f"none of the {len(cands)} {prompt!r} is the chosen one"
        container, score = best
        ys, xs, stats = rd.select_depth_pixels(depth_m, container)
        self._publish_overlay(
            color,
            container,
            np.column_stack((xs, ys)).tolist(),
            f"Samples {len(xs)}/{NEAREST_POINTS}; rejected {stats['depth_outliers_px']}",
        )
        return dict(container=container, depth_m=depth_m, depth_msg=depth, score=score), None


def detect_container(
    camera,
    prompt,
    threshold,
    frames,
    mask_iou_min,
    timeout_s,
    log,
    checkpoint,
    select=None,
    point=None,
    progress=None,
):
    """rim_detection.detect_container (`frames` consecutive masks agreeing pairwise at
    mask_iou_min, each frame's point with TF at its own stamp, averaged), with `select` (which of
    SAM3's objects, WristCamera.shot) and `point(depth_m, mask, K, T)` -> (point, info) or (None,
    why) (default rim_detection.container_point). Every try is logged as it happens, and
    `progress(message)` (the action's feedback) gets the latest."""
    point = point or container_point
    deadline = time.monotonic() + timeout_s
    t_start = time.monotonic()
    streak, rejected, attempts = [], {}, 0

    def say(message):
        took = getattr(camera, "last_call_s", 0.0)
        line = (
            f"detect {prompt!r} try {attempts} ({time.monotonic() - t_start:.1f}/"
            f"{timeout_s:.0f} s, SAM3 {took:.2f} s): {message}"
        )
        log(line)
        if progress is not None:
            progress(line)

    def reject(why):
        rejected[why] = rejected.get(why, 0) + 1
        say(f"rejected: {why}")

    while time.monotonic() < deadline:
        checkpoint()
        attempts += 1
        shot, why = camera.shot(prompt, threshold, deadline, select=select)
        if shot is None:
            if why is None:
                break
            reject(why)
            streak = []
            continue
        if any(mask_iou(shot["container"], s["container"]) < mask_iou_min for s in streak):
            reject("masks disagree with the streak")
            streak = []
        streak.append(shot)
        if len(streak) < frames:
            say(f"{len(streak)}/{frames} agreeing frames (score {shot['score']:.2f})")
            continue
        points, infos, why = [], [], None
        for s in streak:
            try:
                T = camera.pose(s["depth_msg"])
            except Exception as e:  # noqa: BLE001 -- tf2's lookup errors: the streak is not used
                why = f"no TF at a frame ({type(e).__name__})"
                break
            p, info = point(s["depth_m"], s["container"], camera.K(), T)
            if p is None:
                why = info
                break
            points.append(p)
            infos.append(info)
        if why is not None:
            reject(why)
            streak = []
            continue
        mean = np.mean(points, axis=0)
        spread = float(np.max(np.linalg.norm(np.asarray(points) - mean, axis=1)))
        info = dict(
            infos[-1],
            depths_m=[i["depths_m"] for i in infos],
            frame_points=np.round(points, 4).tolist(),
            frames_spread_m=spread,
        )
        iou_min = min(
            (
                mask_iou(a["container"], b["container"])
                for i, a in enumerate(streak)
                for b in streak[i + 1 :]
            ),
            default=1.0,
        )
        log(
            f"container: {frames} frames agree (mask IoU >= {iou_min:.3f}) in {attempts} tries; "
            f"per frame {info['frame_points']} -> mean {np.round(mean, 3).tolist()} (frames "
            f"within {spread * 100:.1f} cm of it)"
        )
        return dict(point=mean, info=info, iou_min=iou_min, attempts=attempts, rejected=rejected)
    raise NotDetected(
        f"the container not seen in {frames} agreeing frames within {timeout_s:.0f} s ({attempts} "
        f"tries; rejected: " + ("; ".join(f"{k} x{n}" for k, n in rejected.items()) or "none") + ")"
    )
