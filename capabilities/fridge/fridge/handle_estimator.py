"""Always-on fridge handle estimator: the handle's position and the door's yaw kept up to date from the scene camera, so
they are ready before anyone asks. Read-only: it never commands the arm.

Per tracked frame:
- SAM 3's tracking session on the scene camera (StartTracking, prompts ['handle']) gives the handle's mask on
  /perception/sam3/object_detections (label_image). If another node already runs a session (detections are arriving),
  this node only listens; if none arrive for SESSION_WAIT_S it starts its own (and stops it on exit).
- The mask is paired with the depth frame of the same stamp and the camera's pose in the arm's frame at that stamp
  (TF). The handle instance is picked as the fridge's detector picks it (HandleDetector.pick_handle_mask).
- The handle point: as the detector computes it (_bar_front: per mask row, the bar's front; their median, at the
  middle of their height).
- The door's normal and yaw: door_normal.measure_door_normal (the printed handle's model registered to the mask's
  points), warm-started from the previous frame's fit; a cold fit when there is none or the warm one fails.
Over the last WINDOW_S: the median handle point and yaw and their spread (max - min). ok is false, with why, when there
are fewer than MIN_FRAMES fitted frames, the newest is older than MAX_AGE_S, or a spread is over its limit.

Published (latched, std_msgs/String JSON) on /fridge_side/handle_estimate at the tracking rate, and written to
OUT_JSON. When nothing else publishes /perception/sam3_overlay/compressed, the node publishes the overlay image there
too (the mask tinted, the estimate written on it).

Usage: python3 -m fridge.handle_estimator [--out PATH]
"""

import argparse
import collections
import json
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import String

from . import door_normal as dn

COLOR, DEPTH, INFO = (
    "/scene_camera/color/image_raw",
    "/scene_camera/depth/image_rect_raw",
    "/scene_camera/color/camera_info",
)
DETECTIONS = "/perception/sam3/object_detections"
OVERLAY = "/perception/sam3_overlay/compressed"
OUT_TOPIC = "/fridge_side/handle_estimate"
BASE = "arm_mount"
PROMPTS = ["handle"]
SCORE_THRESHOLD = 0.3  # as the fridge node's detector (sam3_threshold); SAM 3's default 0.5 dropped a handle seen at 0.32-0.38 (11-06)
WINDOW_S = 3.0  # the rolling estimate's window (SAM 3 tracks at ~3 Hz here: ~9 frames)
CONSEC_N = 2  # 11-06: ok when the newest CONSEC_N fits agree within CONSEC_YAW_DEG (their mean is the
CONSEC_YAW_DEG = 3.0  # estimate) -- replaces MIN_FRAMES / MAX_YAW_SPREAD_DEG over the window
MIN_FRAMES = 5  # fewer fitted frames in the window: not ok
MAX_AGE_S = 0.5  # the newest fitted frame older than this: not ok
MAX_YAW_SPREAD_DEG = 6.0  # door yaw spread in the window: its 10th to 90th percentile (one stray frame doesn't count)
MAX_HANDLE_SPREAD_M = 0.02  # each handle coordinate's 10th to 90th percentile in the window
SESSION_WAIT_S = 5.0  # no detections this long: start our own tracking session
FRAME_KEEP_S = 3.0  # depth / colour frames kept to pair with the detections
PAIR_TOL_S = 0.03  # a depth / colour frame pairs with a detection within this of its stamp (15 Hz: 67 ms apart)


def _stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def _nearest(buf, header, tol=PAIR_TOL_S):
    """The buffered message nearest `header`'s stamp, within tol; None if none."""
    t = _stamp(header)
    best = min(buf.values(), key=lambda m: abs(_stamp(m.header) - t), default=None)
    return best if best is not None and abs(_stamp(best.header) - t) <= tol else None


class Estimator:
    def __init__(self, node, out_path):
        from rammp_perception_interfaces.msg import ObjectDetections
        from rammp_perception_interfaces.srv import StartTracking, StopTracking

        from .perception import HandleDetector

        self.n, self.out_path, self.bridge = node, out_path, CvBridge()
        self.HandleDetector, self.StartTracking, self.StopTracking = HandleDetector, StartTracking, StopTracking
        self.lock = threading.Lock()
        self.depth, self.color = collections.OrderedDict(), collections.OrderedDict()
        self.K = None
        self.pending = None  # the newest detection message not processed yet
        self.last_det_wall = 0.0
        self.session = None
        self.prev_fit = None
        self.window = collections.deque()  # (stamp, handle (3,), yaw, normal (3,), ms)
        self.misses = collections.deque()  # (stamp, why)
        self.stats = {"frames": 0, "fitted": 0, "cold": 0}
        self.tf = __import__("tf2_ros").Buffer(cache_time=Duration(seconds=10))
        self._tfl = __import__("tf2_ros").TransformListener(self.tf, node)
        cb = ReentrantCallbackGroup()
        node.create_subscription(Image, DEPTH, self.on_depth, qos_profile_sensor_data, callback_group=cb)
        node.create_subscription(Image, COLOR, self.on_color, qos_profile_sensor_data, callback_group=cb)
        node.create_subscription(CameraInfo, INFO, self.on_info, qos_profile_sensor_data, callback_group=cb)
        node.create_subscription(ObjectDetections, DETECTIONS, self.on_det, 10, callback_group=cb)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = node.create_publisher(String, OUT_TOPIC, latched)
        self.overlay_pub = None
        self.start_cli = node.create_client(StartTracking, "/perception/sam3/start_tracking", callback_group=cb)
        self.stop_cli = node.create_client(StopTracking, "/perception/sam3/stop_tracking", callback_group=cb)
        node.create_timer(0.02, self.work, callback_group=MutuallyExclusiveCallbackGroup())
        node.create_timer(2.0, self.housekeeping, callback_group=MutuallyExclusiveCallbackGroup())

    # --- inputs ---------------------------------------------------------------------------------------------------
    def _keep(self, buf, msg):
        with self.lock:
            buf[(msg.header.stamp.sec, msg.header.stamp.nanosec)] = msg
            newest = _stamp(msg.header)
            while buf and newest - _stamp(next(iter(buf.values())).header) > FRAME_KEEP_S:
                buf.popitem(last=False)

    def on_depth(self, msg):
        self._keep(self.depth, msg)

    def on_color(self, msg):
        if self.overlay_pub is not None:
            self._keep(self.color, msg)

    def on_info(self, msg):
        self.K = np.asarray(msg.k, float).reshape(3, 3)

    def on_det(self, msg):
        self.last_det_wall = time.time()
        with self.lock:
            self.pending = msg  # older unprocessed ones are dropped: latency over completeness

    # --- the session and the overlay ------------------------------------------------------------------------------
    def housekeeping(self):
        if self.overlay_pub is None and self.n.count_publishers(OVERLAY) == 0:
            self.overlay_pub = self.n.create_publisher(CompressedImage, OVERLAY, 1)
            self.n.get_logger().info(f"nothing else publishes {OVERLAY}: publishing the overlay here")
        if time.time() - self.last_det_wall > SESSION_WAIT_S and self.session is None and self.start_cli.service_is_ready():
            self.session = "pending"
            req = self.StartTracking.Request(prompts=PROMPTS, image_topic=COLOR, score_threshold=SCORE_THRESHOLD, redetect_period_s=2.0)
            self.start_cli.call_async(req).add_done_callback(self._started)
        if self.session not in (None, "pending") and time.time() - self.last_det_wall > 4 * SESSION_WAIT_S:
            self.n.get_logger().warn(f"session {self.session}: no detections for {4 * SESSION_WAIT_S:g} s; restarting")
            self.session = None

    def _started(self, fut):
        r = fut.result()
        self.session = r.session_id if r is not None and r.accepted else None
        self.n.get_logger().info(f"start_tracking {PROMPTS}: {None if r is None else (r.accepted, r.message)}")

    def stop_session(self):
        if isinstance(self.session, int) and self.stop_cli.service_is_ready():
            fut = self.stop_cli.call_async(self.StopTracking.Request(session_id=self.session))
            t0 = time.time()
            while not fut.done() and time.time() - t0 < 2.0:
                time.sleep(0.05)

    # --- per frame ------------------------------------------------------------------------------------------------
    def work(self):
        with self.lock:
            det = self.pending
            if det is None:
                return
            depth, color = _nearest(self.depth, det.header), _nearest(self.color, det.header)
            if depth is None:
                if self.depth and _stamp(next(reversed(self.depth.values())).header) - _stamp(det.header) > 0.5:
                    self.pending = None  # its depth frame never came (or was dropped): skip it
                    self._miss(_stamp(det.header), "no depth frame with the detection's stamp")
                return
            self.pending = None
        t0 = time.perf_counter()
        why, fit, handle = self.process(det, depth)
        ms = (time.perf_counter() - t0) * 1000.0
        ts = _stamp(det.header)
        self.stats["frames"] += 1
        if fit is None:
            self._miss(ts, why)
        else:
            self.stats["fitted"] += 1
            self.window.append((ts, handle, fit.yaw_deg, fit.normal, ms, fit.rms_m, fit.inlier_fraction))
        est = self.estimate(ts)
        est["last_frame"] = {"stamp": ts, "compute_ms": round(ms, 1), "why": why or "",
                             "yaw_deg": None if fit is None else round(fit.yaw_deg, 2)}
        self.publish(est)
        if self.overlay_pub is not None and color is not None:
            self.overlay(color, det, est)

    def _miss(self, ts, why):
        self.misses.append((ts, why))
        while self.misses and ts - self.misses[0][0] > WINDOW_S:
            self.misses.popleft()

    def process(self, det, depth_msg):
        """(why, HandleFit, handle point) for one detection message; (why, None, None) on a miss."""
        if self.K is None:
            return "no camera_info yet", None, None
        objs = [o for o in det.objects if o.mask_id > 0]
        if not objs:
            return "SAM 3 tracks no handle in this frame", None, None
        li = det.label_image
        labels = np.frombuffer(bytes(li.data), dtype="<u2").reshape(li.height, li.step // 2)[:, : li.width]
        try:
            tr = self.tf.lookup_transform(BASE, depth_msg.header.frame_id, Time.from_msg(depth_msg.header.stamp),
                                          timeout=Duration(seconds=0.05))
        except Exception:  # noqa: BLE001 -- tf2's exceptions
            try:
                tr = self.tf.lookup_transform(BASE, depth_msg.header.frame_id, Time())
            except Exception as e:  # noqa: BLE001
                return f"no {BASE} <- {depth_msg.header.frame_id} transform: {e}", None, None
        q, p = tr.transform.rotation, tr.transform.translation
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T[:3, 3] = [p.x, p.y, p.z]
        depth = np.asarray(self.bridge.imgmsg_to_cv2(depth_msg, "passthrough"), np.float32)
        if labels.shape != depth.shape:
            return f"mask {labels.shape} and depth {depth.shape} differ in size", None, None
        inst = [(float(o.score), labels == o.mask_id, o.perception_class) for o in objs]
        detector = self.HandleDetector()
        mask = detector.pick_handle_mask(inst, {"depth_mm": depth, "K": self.K, "base_T_cam": T})
        if mask is None:
            return detector.last_miss or "no handle-like instance", None, None
        bar = detector._bar_front(mask, depth, self.K, T[2, :3])
        if bar is None:
            return "too little depth on the handle's front", None, None
        handle = T[:3, :3] @ bar[0] + T[:3, 3]
        fit = None
        if self.prev_fit is not None:
            try:
                fit = dn.measure_door_normal(depth, self.K, T, mask, prev=self.prev_fit)
            except dn.DoorNormalError:
                fit = None
        if fit is None:
            self.stats["cold"] += 1
            try:
                fit = dn.measure_door_normal(depth, self.K, T, mask)
            except dn.DoorNormalError as e:
                self.prev_fit = None
                return str(e), None, None
        self.prev_fit = fit
        return "", fit, handle

    def estimate(self, now):
        while self.window and now - self.window[0][0] > WINDOW_S:
            self.window.popleft()
        W = list(self.window)
        est = {"stamp": now, "wall_time": time.time(), "frame": BASE, "window_s": WINDOW_S, "n_frames": len(W),
               "n_misses": len(self.misses), "ok": False, "why": ""}
        if self.misses:
            est["last_miss"] = self.misses[-1][1]
        if not W:
            est["why"] = "no fitted frame in the last %.0f s" % WINDOW_S + (f" ({self.misses[-1][1]})" if self.misses else "")
            return est
        H = np.array([w[1] for w in W])
        Y = np.array([w[2] for w in W])
        L = W[-CONSEC_N:]  # the newest fits
        HL, YL = np.array([w[1] for w in L]), np.array([w[2] for w in L])
        yaw = float(np.mean(YL))
        n = dn.normal_from_yaw(yaw)
        est.update(
            handle=[round(float(v), 4) for v in HL.mean(axis=0)],
            handle_spread_m=[round(float(v), 4) for v in HL.max(axis=0) - HL.min(axis=0)],
            yaw_deg=round(yaw, 2),
            yaw_newest_deg=[round(float(v), 2) for v in YL],
            yaw_spread_deg=round(float(YL.max() - YL.min()), 2),
            yaw_window_spread_deg=round(float(np.percentile(Y, 90) - np.percentile(Y, 10)), 2),
            yaw_min_max_deg=[round(float(Y.min()), 2), round(float(Y.max()), 2)],
            normal=[round(float(v), 5) for v in n],
            age_s=round(now - W[-1][0], 3),
            compute_ms_median=round(float(np.median([w[4] for w in W])), 1),
            rms_mm_median=round(float(np.median([w[5] for w in W])) * 1000, 2),
            inlier_fraction_median=round(float(np.median([w[6] for w in W])), 3),
        )
        problems = []
        if len(W) < CONSEC_N:
            problems.append(f"{len(W)} fitted frames in {WINDOW_S:g} s (needs {CONSEC_N})")
        if est["age_s"] > MAX_AGE_S:
            problems.append(f"newest fit {est['age_s']:.2f} s old (max {MAX_AGE_S:g})")
        if est["yaw_spread_deg"] > CONSEC_YAW_DEG:
            problems.append(f"the newest {CONSEC_N} fits' yaws {est['yaw_newest_deg']} differ by "
                            f"{est['yaw_spread_deg']:.1f} deg (max {CONSEC_YAW_DEG:g})")
        if max(est["handle_spread_m"]) > MAX_HANDLE_SPREAD_M:
            problems.append(f"handle spread {max(est['handle_spread_m']) * 100:.1f} cm (max {MAX_HANDLE_SPREAD_M * 100:g})")
        est["ok"] = not problems
        est["why"] = "; ".join(problems)
        return est

    def publish(self, est):
        est["counts"] = dict(self.stats)
        s = json.dumps(est)
        self.pub.publish(String(data=s))
        if self.out_path:
            tmp = self.out_path + ".tmp"
            with open(tmp, "w") as f:
                f.write(s)
            os.replace(tmp, self.out_path)

    def overlay(self, color, det, est):
        img = self.bridge.imgmsg_to_cv2(color, "bgr8").copy()
        li = det.label_image
        if li.height == img.shape[0] and li.width == img.shape[1]:
            labels = np.frombuffer(bytes(li.data), dtype="<u2").reshape(li.height, li.step // 2)[:, : li.width]
            tint = img.copy()
            for o in det.objects:
                if o.mask_id:
                    tint[labels == o.mask_id] = (0, 200, 255)
            cv2.addWeighted(tint, 0.4, img, 0.6, 0, dst=img)
        if "yaw_deg" in est:
            text = (f"door yaw {est['yaw_deg']:+.1f} deg (spread {est['yaw_spread_deg']:.1f}), handle "
                    f"{np.round(est['handle'], 3).tolist()}, n {est['n_frames']}, {'OK' if est['ok'] else 'NOT OK'}")
        else:
            text = f"no estimate: {est['why'][:80]}"
        cv2.putText(img, text, (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        if not est.get("ok") and est.get("why"):
            cv2.putText(img, est["why"][:90], (6, img.shape[0] - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 255), 1,
                        cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img)
        self.overlay_pub.publish(CompressedImage(header=color.header, format="jpeg", data=buf.tobytes()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="", help="also write each estimate (JSON) to this file")
    args = ap.parse_args()
    import signal

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    rclpy.init()
    node = rclpy.create_node("fridge_handle_estimator")
    from door_common import common as dc

    dc.set_output(lambda *a, **k: None)  # the detector's per-frame notes, ~15 a second, are not logged
    est = Estimator(node, args.out)
    dn.load_handle_model()  # load the STL before the first frame
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        est.stop_session()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
