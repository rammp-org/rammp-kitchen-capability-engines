"""SAM3 always polling the wrist camera for food and drink (track_prompts), through peer and pick.

The SAM3 node (rammp-perception, object_detection/sam3/rammp_sam3/node.py) runs ONE tracking
session for the whole robot: StartTracking detects every instance of every prompt, then tracks
them at up to max_tracking_rate_hz on /perception/sam3/object_detections (an ObjectDetections
per frame: objects with a stable instance_id like "food_3", plus a mono16 label image), and every
redetect_period_s detects again (new items get ids, the ones still overlapping keep theirs). A new
StartTracking, from anyone, replaces the session; with lease_enabled false (its default) a
session runs until StopTracking.

  TrackingSession  engine-lifetime: wants a session while peer or pick runs, keeps it across
                   peer's handoff to pick (the instance ids stay valid), renews its lease every
                   RENEW_S (a no-op while SAM3's leases are off), restarts it when renewing says it
                   is gone (replaced by someone else), and stops it when the last action is done.
                   Never blocks a caller but want(timeout) at the start of an action.
  TrackedView      action-scoped: the last frames of /perception/sam3/object_detections, the
                   visible food and drink in them, their masks; ~/in_view (std_msgs/String JSON,
                   1 Hz) lists what is tracked now, with 3D points when the wrist depth is there.

Read-only: nothing here commands the arm.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque

import numpy as np

DETECTIONS_TOPIC = "/perception/sam3/object_detections"
START, RENEW, STOP = ("/perception/sam3/start_tracking", "/perception/sam3/renew_tracking",
                      "/perception/sam3/stop_tracking")
RENEW_S = 2.0  # renew the lease this often
HANDOVER_S = 20.0  # a session peer handed over, and no pick took within this: released
RESTART_S = 5.0  # a lost session is restarted at most this often
SERVICE_WAIT_S = 2.0
STATE_VISIBLE = 0
VIEW_KEEP = 8  # tracked frames kept
IN_VIEW_PERIOD_S = 1.0


def stamp_ns(msg) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def label_array(img) -> np.ndarray:
    """A mono16 / 16UC1 label image as an H x W uint16 array (pixel value = mask_id)."""
    if img.encoding not in ("mono16", "16UC1") or not img.height:
        raise ValueError(f"SAM3 label image: encoding {img.encoding!r}, {img.width}x{img.height}")
    dtype = ">u2" if img.is_bigendian else "<u2"
    a = np.frombuffer(bytes(img.data), dtype).reshape(img.height, img.step // 2)
    return a[:, : img.width].astype(np.uint16)


class TrackingSession:
    """The SAM3 tracking session for food and drink on the wrist camera. Make it in the engine's
    __init__ (its clients and renewal timer live with the engine)."""

    def __init__(self, engine, cfg, clients=None, types=None):
        self._log = engine.get_logger()
        self.prompts = [str(p) for p in cfg["track_prompts"]]
        self.topic = cfg["wrist_topics"][0]
        self.threshold = float(cfg["pick_sam3_threshold"])
        self.lease_s = float(cfg["track_lease_s"])
        self.redetect_s = float(cfg["track_redetect_s"])
        self.max_objects = int(cfg["track_max_objects"])
        self._lock = threading.Lock()
        self.session_id = 0  # ours, while SAM3 runs it; 0: none
        self._want = 0  # actions that want it (peer, pick)
        self._start_fut = self._renew_fut = None
        self._handover_until = None  # peer handed its claim to the coming pick (monotonic s)
        self._last_start = -math.inf
        self._next_renew = 0.0
        if clients is None:  # ROS: real clients and a 1 Hz timer
            from rammp_perception_interfaces.srv import RenewTracking, StartTracking, StopTracking
            from rclpy.callback_groups import ReentrantCallbackGroup

            group = ReentrantCallbackGroup()
            self._types = dict(start=StartTracking, renew=RenewTracking, stop=StopTracking)
            clients = {
                name: engine.create_client(t, srv, callback_group=group)
                for (name, t), srv in zip(self._types.items(), (START, RENEW, STOP))
            }
            engine.create_timer(1.0, self.tick, callback_group=group)
        else:  # tests: fake clients and service types, dict(start=, renew=, stop=)
            self._types = types
        self._clients = clients

    @property
    def active(self) -> bool:
        return self.session_id != 0

    # ------------------------------------------------------------------ for actions

    def want(self, timeout_s: float = 0.0) -> bool:
        """An action wants the session (pair with release()). Starts it if it is not running and
        waits up to timeout_s for SAM3 to accept. True when it runs."""
        with self._lock:
            if self._handover_until is not None:  # take peer's claim over: no new one
                self._handover_until = None
            else:
                self._want += 1
            running = self.active
        if running:
            return True
        start = self._clients["start"]
        if timeout_s > 0 and not start.wait_for_service(timeout_sec=min(timeout_s,
                                                                         SERVICE_WAIT_S)):
            self._log.warning(f"SAM3 tracking: {START} not available")
            return False
        self._send_start()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._poll_start():
                return self.active
            time.sleep(0.02)
        return self.active

    def hand_over(self) -> None:
        """peer is handing off to a pick goal: keep the session (and so its instance ids) for it.
        The pick's want() takes this claim; if none does within HANDOVER_S, it is released."""
        with self._lock:
            self._handover_until = time.monotonic() + HANDOVER_S

    def release(self) -> None:
        """An action is done with it. The last one stops the session."""
        with self._lock:
            self._want = max(0, self._want - 1)
            if self._want:
                return
            sid, self.session_id = self.session_id, 0
            # a start still in flight is kept: when SAM3 accepts it, _poll_start stops it again
            # (dropped, nobody would: with SAM3's leases off it would run forever)
            self._renew_fut = None
        if sid:
            req = self._types["stop"].Request()
            req.session_id = sid
            self._clients["stop"].call_async(req)  # not waited for
            self._log.info(f"SAM3 tracking: stopping session {sid}")

    # ------------------------------------------------------------------ the timer

    def tick(self) -> None:
        """1 Hz on the engine's executor: answers, renewals, restarts. Never blocks."""
        self._poll_start()
        now = time.monotonic()
        with self._lock:
            expired = self._handover_until is not None and now > self._handover_until
            if expired:
                self._handover_until = None
        if expired:
            self._log.warning("SAM3 tracking: no pick took the session peer handed over")
            self.release()
        with self._lock:
            want, sid, renew_fut = self._want, self.session_id, self._renew_fut
        if renew_fut is not None and renew_fut.done():
            self._renew_fut = None
            self._on_renew(sid, renew_fut)
            with self._lock:
                sid = self.session_id
        if not want:
            return
        if sid and self._renew_fut is None and now >= self._next_renew:
            self._next_renew = now + RENEW_S
            req = self._types["renew"].Request()
            req.session_id, req.lease_s = sid, self.lease_s
            self._renew_fut = self._clients["renew"].call_async(req)
        elif not sid and self._start_fut is None and now - self._last_start >= RESTART_S:
            self._log.warning("SAM3 tracking: no session; starting one")
            self._send_start()

    def _send_start(self) -> None:
        with self._lock:
            if self._start_fut is not None:
                return
            self._last_start = time.monotonic()
            req = self._types["start"].Request()
            req.prompts, req.image_topic = list(self.prompts), self.topic
            req.score_threshold, req.lease_s = self.threshold, self.lease_s
            req.redetect_period_s, req.max_objects = self.redetect_s, self.max_objects
            self._start_fut = self._clients["start"].call_async(req)

    def _poll_start(self) -> bool:
        """Handle a finished StartTracking. True when one was handled."""
        with self._lock:
            fut = self._start_fut
            if fut is None or not fut.done():
                return False
            self._start_fut = None
        try:
            r = fut.result()
        except Exception as exc:  # noqa: BLE001 -- a failed call: the timer retries
            self._log.warning(f"SAM3 tracking: start failed: {exc!r}")
            return True
        with self._lock:
            if r.accepted and self._want:
                self.session_id = int(r.session_id)
                self._next_renew = time.monotonic() + RENEW_S
            elif r.accepted:  # every action released it meanwhile: stop it again
                req = self._types["stop"].Request()
                req.session_id = int(r.session_id)
                self._clients["stop"].call_async(req)
        if r.accepted:
            self._log.info(f"SAM3 tracking: session {r.session_id} {self.prompts} on "
                           f"{self.topic}: {r.message}")
        else:
            self._log.warning(f"SAM3 tracking: start refused: {r.message}")
        return True

    def _on_renew(self, sid, fut) -> None:
        try:
            r = fut.result()
        except Exception as exc:  # noqa: BLE001 -- try again next time
            self._log.warning(f"SAM3 tracking: renew failed: {exc!r}")
            return
        if not r.renewed:
            with self._lock:
                if self.session_id == sid:
                    self.session_id = 0
            self._log.warning(f"SAM3 tracking: session {sid} is gone ({r.message}): replaced by "
                              f"another StartTracking, or stopped; restarting")

    def shutdown(self) -> None:
        """Engine teardown: stop our session (best effort; with SAM3's leases off nothing else
        would)."""
        with self._lock:
            self._want, self._handover_until = 1, None
        self.release()


def visible_objects(msg, prompts, label=None):
    """The visible objects of `msg` that are ours (a track_prompts class; `label`: only that
    one), best-scoring first."""
    out = [o for o in msg.objects
           if o.perception_class in prompts and (label is None or o.perception_class == label)
           and o.state == STATE_VISIBLE and o.mask_id]
    return sorted(out, key=lambda o: -o.score)


class TrackedView:
    """The tracked frames, inside a running action. `camera` (wrist_detection.WristCamera, or
    None): for the 3D points on ~/in_view."""

    def __init__(self, engine, prompts, camera=None, subscribe=True):
        self.prompts = list(prompts)
        self._camera = camera
        self._lock = threading.Lock()
        self._msgs = deque(maxlen=VIEW_KEEP)
        self._labels = {}  # id(msg) -> label array, for the kept frames
        self._next_in_view = 0.0
        self._log = engine.get_logger()
        self._pub = None
        if subscribe:
            from rammp_perception_interfaces.msg import ObjectDetections
            from rclpy.qos import QoSProfile
            from std_msgs.msg import String

            self._String = String
            engine.create_action_subscription(ObjectDetections, DETECTIONS_TOPIC, self.on_msg,
                                              QoSProfile(depth=2))
            self._pub = engine.create_action_publisher(String, "~/in_view", QoSProfile(depth=2))

    def on_msg(self, msg) -> None:  # on the executor
        with self._lock:
            self._msgs.append(msg)
            keep = {id(m) for m in self._msgs}
            self._labels = {k: v for k, v in self._labels.items() if k in keep}
        if self._pub is not None and time.monotonic() >= self._next_in_view:
            self._next_in_view = time.monotonic() + IN_VIEW_PERIOD_S
            try:
                self._pub.publish(self._String(data=json.dumps(self.describe(msg))))
            except Exception as exc:  # noqa: BLE001 -- a diagnostic: never fails the action
                self._log.warning(f"~/in_view: {exc!r}", throttle_duration_sec=10.0)

    def newest(self, newer_than_ns: int = 0):
        with self._lock:
            m = self._msgs[-1] if self._msgs else None
        return m if m is not None and stamp_ns(m) > newer_than_ns else None

    def wait_newer(self, newer_than_ns: int, timeout_s: float, checkpoint=None):
        """The first tracked frame stamped after newer_than_ns, or None after timeout_s."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if checkpoint is not None:
                checkpoint()
            m = self.newest(newer_than_ns)
            if m is not None:
                return m
            time.sleep(0.02)
        return None

    def visible(self, msg, label=None):
        return visible_objects(msg, self.prompts, label)

    def mask(self, msg, obj) -> np.ndarray:
        with self._lock:
            lab = self._labels.get(id(msg))
        if lab is None:
            lab = label_array(msg.label_image)
            with self._lock:
                self._labels[id(msg)] = lab
        return lab == obj.mask_id

    def find(self, instance_id, newer_than_ns: int = 0):
        """(msg, obj) of `instance_id` in the newest frame stamped after newer_than_ns where it
        is visible, or (None, None)."""
        with self._lock:
            msgs = list(self._msgs)
        for m in reversed(msgs):
            if stamp_ns(m) <= newer_than_ns:
                break
            for o in self.visible(m):
                if o.instance_id == instance_id:
                    return m, o
        return None, None

    def describe(self, msg) -> dict:
        """What ~/in_view says about one frame: the visible food and drink, with 3D points (m, in
        the arm frame) when the wrist depth and TF at that frame are there."""
        objs = self.visible(msg)
        points = [None] * len(objs)
        if self._camera is not None and objs:
            points = self._points(msg, objs)
        return {
            "stamp": round(stamp_ns(msg) / 1e9, 3),
            "age_s": round(time.time() - stamp_ns(msg) / 1e9, 2),
            "objects": [
                {"instance_id": o.instance_id, "class": o.perception_class,
                 "score": round(float(o.score), 3), "box": [round(float(v)) for v in o.box[:4]],
                 "point": p}
                for o, p in zip(objs, points)
            ],
        }

    def _points(self, msg, objs):
        from . import wrist_detection as wd

        frame = self._camera.frame_at(stamp_ns(msg))
        if frame is None:
            return [None] * len(objs)
        try:
            K, T = frame.K(), frame.pose(frame.depth_msg)
        except Exception:  # noqa: BLE001 -- no TF or camera_info yet: no points
            return [None] * len(objs)
        out = []
        for o in objs:
            p, _ = wd.container_point(frame.depth_m, self.mask(msg, o), K, T)
            out.append(None if p is None else np.round(p, 3).tolist())
        return out
