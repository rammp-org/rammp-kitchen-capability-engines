"""Coarse homing from the scene camera: where the fridge is, from its 2D box and depth.

The scene segmenter (/perception/object_detections, on the base-mounted scene camera) gives the
fridge a box (xyxy, pixels) and a mask (label_image, mono16, the box's mask_id). One point in the
box -- scene_box_point, by default 3/4 across (1/4 from the right border) and halfway down -- is
deprojected at the depth sampled around it: a window of scene_window_frac x the box's size, only
pixels of the fridge's own mask (so the arm in front of it is not sampled), within
scene_depth_range_m, at scene_depth_percentile. scene_frames detections are combined (median).

Depth must be aligned to colour (the same pixels as the detections). It may differ in resolution:
the box and K are scaled to it. Read-only: nothing here commands anything.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

POLL_S = 0.05  # how often estimate() looks for new detections (and honours pause and cancel)
SYNC_S = 0.1  # a depth frame within this of a detection's stamp is its depth
STATE_VISIBLE = 0  # rammp_perception_interfaces/DetectedObject.STATE_VISIBLE


def stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def box_point(box, frac):
    """The pixel (u, v) at fractions `frac` = (across, down) of an xyxy box."""
    x0, y0, x1, y1 = (float(v) for v in box)
    return x0 + float(frac[0]) * (x1 - x0), y0 + float(frac[1]) * (y1 - y0)


def image_array(msg) -> np.ndarray:
    """A sensor_msgs/Image as a 2D array (16UC1 / mono16 -> uint16, 32FC1 -> float32)."""
    enc = msg.encoding.lower()
    big = bool(msg.is_bigendian)
    if enc in ("16uc1", "mono16"):
        dt = np.dtype(">u2" if big else "<u2")
    elif enc == "32fc1":
        dt = np.dtype(">f4" if big else "<f4")
    else:
        raise ValueError(f"unsupported image encoding {msg.encoding!r} (want 16UC1, 32FC1, mono16)")
    a = np.frombuffer(bytes(msg.data), dtype=dt)
    return a.reshape(msg.height, msg.step // dt.itemsize)[:, : msg.width]


def depth_metres(msg) -> np.ndarray:
    """Depth image in metres (16UC1 is mm, 32FC1 is m); 0 / NaN stay invalid."""
    a = image_array(msg).astype(np.float64)
    return a if msg.encoding.lower() == "32fc1" else a * 1e-3


def sample_depth(depth, labels, mask_id, u, v, half_w, half_h, z_range, percentile):
    """(depth at `percentile` of the valid pixels in the window around (u, v), how many), or
    (None, 0). `labels` None: every pixel counts; else only those == mask_id."""
    h, w = depth.shape
    c0, c1 = max(0, int(round(u - half_w))), min(w, int(round(u + half_w)) + 1)
    r0, r1 = max(0, int(round(v - half_h))), min(h, int(round(v + half_h)) + 1)
    if c0 >= c1 or r0 >= r1:
        return None, 0
    z = depth[r0:r1, c0:c1]
    ok = np.isfinite(z) & (z >= z_range[0]) & (z <= z_range[1])
    if labels is not None:
        ok &= labels[r0:r1, c0:c1] == mask_id
    vals = z[ok]
    if vals.size == 0:
        return None, 0
    return float(np.percentile(vals, percentile)), int(vals.size)


def deproject(K, u, v, z) -> np.ndarray:
    """Pixel (u, v) at depth z (m) -> point in the optical frame (x right, y down, z forward)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([(u - cx) / fx * z, (v - cy) / fy * z, z])


class SceneFridge:
    """The scene camera's view of the fridge. Make it inside execute(): its subscriptions are the
    action's."""

    def __init__(self, engine, cfg):
        from rammp_perception_interfaces.msg import ObjectDetections
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image

        self.cls = cfg["scene_class"]
        self.min_score = float(cfg["scene_min_score"])
        self.frac = [float(v) for v in cfg["scene_box_point"]]
        self.window = float(cfg["scene_window_frac"])
        self.percentile = float(cfg["scene_depth_percentile"])
        self.z_range = [float(v) for v in cfg["scene_depth_range_m"]]
        self.min_samples = int(cfg["scene_min_samples"])
        self.topics = (
            cfg["scene_detections_topic"],
            cfg["scene_depth_topic"],
            cfg["scene_camera_info_topic"],
        )
        self._lock = threading.Lock()
        self._dets = deque(maxlen=10)
        self._depths = deque(maxlen=15)
        self._info = None
        self._warned = set()
        self._log = engine.get_logger()
        sub, qos = engine.create_action_subscription, qos_profile_sensor_data
        sub(ObjectDetections, self.topics[0], self._on_det, qos)
        sub(Image, self.topics[1], self._on_depth, qos)
        sub(CameraInfo, self.topics[2], self._on_info, qos)

    # on the executor: must never block
    def _on_det(self, msg) -> None:
        with self._lock:
            self._dets.append(msg)

    def _on_depth(self, msg) -> None:
        with self._lock:
            self._depths.append(msg)

    def _on_info(self, msg) -> None:
        with self._lock:
            self._info = msg

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            self._log.warning(message)

    def _best(self, det):
        objs = [
            o
            for o in det.objects
            if o.perception_class == self.cls
            and o.state == STATE_VISIBLE
            and o.score >= self.min_score
            and o.box[2] > o.box[0]
            and o.box[3] > o.box[1]
        ]
        return max(objs, key=lambda o: o.score) if objs else None

    def estimate_one(self, det):
        """((point in the camera's optical frame, score, samples), '') for one detection message,
        or (None, why)."""
        obj = self._best(det)
        if obj is None:
            return None, (
                f"no visible '{self.cls}' with score >= {self.min_score} on {self.topics[0]}"
            )
        t = stamp_ns(det.header.stamp)
        with self._lock:
            info = self._info
            depth_msg = min(self._depths, key=lambda m: abs(stamp_ns(m.header.stamp) - t),
                            default=None)
        if info is None:
            return None, f"no camera_info on {self.topics[2]}"
        if depth_msg is None:
            return None, f"no depth on {self.topics[1]}"
        dt = abs(stamp_ns(depth_msg.header.stamp) - t) * 1e-9
        if dt > SYNC_S:
            return None, f"no depth frame within {SYNC_S}s of the detection (nearest {dt:.2f}s)"
        depth = depth_metres(depth_msg)
        h, w = depth.shape
        # the detection's pixels are the colour image's: label_image's size, else camera_info's
        li = det.label_image
        det_w = li.width or info.width
        det_h = li.height or info.height
        sx, sy = w / float(det_w), h / float(det_h)
        K = np.asarray(info.k, dtype=float).reshape(3, 3).copy()
        K[0, :] *= w / float(info.width)
        K[1, :] *= h / float(info.height)
        if abs(sx - sy) > 0.02:
            self._warn_once(
                "aspect",
                f"depth {w}x{h} and detections {det_w}x{det_h} differ in aspect: is "
                f"{self.topics[1]} aligned to colour?",
            )
        labels = None
        if li.width and li.height:
            lab = image_array(li)
            if lab.shape == depth.shape:
                labels = lab
            else:
                self._warn_once(
                    "labels",
                    f"label_image {li.width}x{li.height} is not the depth's {w}x{h}: sampling "
                    f"the window without the fridge's mask",
                )
        box = (obj.box[0] * sx, obj.box[1] * sy, obj.box[2] * sx, obj.box[3] * sy)
        u, v = box_point(box, self.frac)
        half_w = self.window * (box[2] - box[0])
        half_h = self.window * (box[3] - box[1])
        z, n = sample_depth(depth, labels, obj.mask_id, u, v, half_w, half_h, self.z_range,
                            self.percentile)
        if z is None or n < self.min_samples:
            return None, (
                f"only {n} valid depth pixels of the fridge around ({u:.0f}, {v:.0f}) "
                f"(< scene_min_samples {self.min_samples}; depth range {self.z_range} m)"
            )
        return (deproject(K, u, v, z), float(obj.score), n), ""

    def estimate(self, ctx, wait_s: float, frames: int):
        """((median point, optical frame_id, newest stamp ns, summary), '') from up to `frames`
        detections within `wait_s`, or (None, why)."""
        deadline = time.monotonic() + wait_s
        seen, points, frame_id, newest, why = 0, [], "", 0, f"no message on {self.topics[0]}"
        scores, counts = [], []
        while len(points) < frames and time.monotonic() < deadline:
            with self._lock:
                fresh = [d for d in self._dets if stamp_ns(d.header.stamp) > seen]
            for det in fresh:
                seen = max(seen, stamp_ns(det.header.stamp))
                est, why = self.estimate_one(det)
                if est is not None:
                    p, score, n = est
                    points.append(p)
                    scores.append(score)
                    counts.append(n)
                    frame_id, newest = det.header.frame_id, seen
            ctx.checkpoint()
            time.sleep(POLL_S)
        if not points:
            return None, why
        pts = np.array(points)
        med = np.median(pts, axis=0)
        spread = float(np.max(np.linalg.norm(pts - med, axis=1)))
        summary = (
            f"{len(points)} detection(s), score {max(scores):.2f}, {int(np.median(counts))} depth "
            f"px each, spread {1000 * spread:.0f} mm, depth {med[2]:.2f} m"
        )
        return (med, frame_id, newest, summary), ""