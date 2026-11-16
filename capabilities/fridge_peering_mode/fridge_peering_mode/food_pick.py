"""Tap-to-pick: take a food container out of the open fridge, from where teleop left the camera.

Two parts:

  FoodTargets  during `peer` teleop, never blocking the 50 Hz servo loop: while the arm is still (no
               motion for follow_settle_s), SAM3 runs on the wrist camera every
               pick_detect_period_s; its containers go to the UI as tappable objects over this
               capability's own overlay stream (the live wrist colour image, ui_overlay_hz). Once
               the arm moves the boxes are taken off the UI: a box is only tappable while it matches
               what the camera sees. A tap ends teleop.
               Without the UI (standalone, or for testing): ~/ui_panel (std_msgs/String, latched)
               mirrors, as JSON, the panel and objects the UI is sent, and ~/simulate_tap
               (std_msgs/String) takes what the UI bridge sends back for a tap: its JSON
               {"event_type": "object_tapped", "instance_id": ...}, a bare instance id, or "" for
               the best-scoring container in view.
  FoodPicker   the pick itself, from the pose teleop left the arm in (no fixed viewing pose, unlike
               the microwave's take_out, which first goes to take_home_joints). It reuses
               microwave_place's take-out (take_out.py, place_motion.py, unchanged) and
               wrist_detection.py (microwave_place's rim_detection, unchanged, plus what the
               fridge needs):
                 1. detect (wrist_detection.detect_container): pick_frames agreeing SAM3 masks
                    of the container, the tapped one chosen by mask IoU (the `pick` action: the
                    best-scoring); the grasp point is the top of the container's near face
                    (pick_point "top": the camera looks level into the fridge, so its nearest
                    depth is the face, not the rim) or the microwave's nearest depth (pick_point
                    "nearest");
                 2. plan (take_out.grasp_pose / plan_take_out): the hand's level heading into the
                    fridge is the camera's at the start; cuRobo to pre-grasp (pick_reach_in back
                    along it, the hand turned 90 deg so the fingers pinch the brim from above and
                    below), refused if the tool leaves a pick_approach_corridor_m tube around the
                    straight line there (the fridge is not in cuRobo's world); the straight line in,
                    the lift and the withdrawal on the arm model; then the leave line;
                 3. run (take_out.run_take_out): gripper open, approach, line in (its last
                    pick_impedance_in_m in joint impedance), close on the brim, lift pick_lift_m,
                    straight back to pre-grasp;
                 4. leave: straight and level back another pick_leave_back_m, out of the fridge,
                    then (pick_leave_to_joints) cuRobo to pick_leave_joints.
               A dry run (no execute) stops after 2. Nothing from the line in onward is
               collision-checked: someone stands at the e-stop.

A refusal while detecting or planning (PickRefused) has commanded nothing: `peer` goes back to
teleop. A failure once the pick moves is an EngineError: the arm holds where it is.
"""

from __future__ import annotations

import json
import math
import re
import time

import numpy as np

PICK_POINTS = ("top", "nearest")
INSTANCE = re.compile(r"^food_(\d+)_(\d+)$")
KEEP_BATCHES = 3  # detections a tap may still refer to
REQUEST_TIMEOUT_S = 5.0  # a SAM3 request not answered within this is dropped
STILL_LIN, STILL_ANG = 0.005, 0.03  # m/s, rad/s: below both the arm is still (as streaming.py)
STILL_HOLD_S = 0.3  # before detecting for the pick: still this long ...
STILL_WAIT_S = 3.0  # ... within this long
MIN_SPEED_SCALE = 0.01  # the driver refuses a speed_scale below this
# wrist frames buffered (each of colour and depth): a SAM3 answer (~0.4-0.9 s) must still find its
# frame. Not the microwave's 180: teleop keeps the camera open for minutes (~1 MB a colour frame)
CAMERA_KEEP = 45
STATE_VISIBLE = 0  # rammp_perception_interfaces/DetectedObject.STATE_VISIBLE

LOOK = "Move the stick to look around. Hold still to find food."
TAP = "Tap a food container to pick it up."
NONE_SEEN = "No food container in view. Move the stick to look around."


class PickRefused(Exception):
    """Detecting or planning the pick failed: nothing has been commanded."""


def stamp_ns(msg) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def level_heading(z_axis) -> np.ndarray:
    """The tool's z axis made level and unit. PickRefused when it points straight up or down."""
    h = np.array([float(z_axis[0]), float(z_axis[1]), 0.0])
    n = float(np.linalg.norm(h))
    if n < 0.1:
        raise PickRefused("the camera looks (almost) straight up or down: no heading in")
    return h / n


def instance_id(batch: int, k: int) -> str:
    return f"food_{batch}_{k}"


def sim_tap_id(data: str):
    """The instance id a ~/simulate_tap message taps: the UI bridge's feedback JSON
    ({"event_type": "object_tapped", "instance_id": ...}) or poll_ui_event()'s ({"type": ...}), a
    bare id, or "best" for an empty message. None for JSON that is no tap."""
    data = data.strip()
    if not data:
        return "best"
    if not data.startswith("{"):
        return data
    try:
        event = json.loads(data)
    except ValueError:
        return None
    if not isinstance(event, dict):
        return None
    if event.get("event_type", event.get("type")) not in ("object_tapped", None):
        return None
    return str(event.get("instance_id", "")).strip() or "best"


def jsonable(o):
    """o with numpy types made plain (the engine json.dumps the result)."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    return o


def segment_distance(points, a, b) -> np.ndarray:
    """Distance of each point (N x 3) from the segment a-b."""
    p = np.atleast_2d(np.asarray(points, dtype=float))
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ab = b - a
    L2 = float(ab @ ab)
    t = np.zeros(len(p)) if L2 < 1e-12 else np.clip((p - a) @ ab / L2, 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def face_top_point(
    depth_m, container, K, T, heading, consistent, near_points=50, band_m=0.02, top_pct=95.0
):
    """The grasp point of one frame for a container seen from the front: (point in T's target
    frame, info) or (None, why).

    Of the container's depth-consistent pixels (`consistent`:
    wrist_detection.consistent_depth_pixels), in T's frame: the near face is every point within
    band_m (along the level `heading`) of the mean of the `near_points` nearest; the point is at
    that mean's depth along the heading, the near face's median across it, and the near face's
    top_pct height percentile (its top edge: the rim).
    The microwave's container_point is the mean of the nearest points themselves: right looking
    down into the oven, where the nearest points are the rim; looking level into a fridge they are
    anywhere on the front face."""
    ys, xs, stats = consistent(depth_m, container)
    if len(xs) < near_points:
        return None, f"{len(xs)} filtered container pixels with depth (need {near_points})"
    zs = depth_m[ys, xs].astype(float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    cam = np.column_stack(((xs - cx) * zs / fx, (ys - cy) * zs / fy, zs))
    pts = cam @ T[:3, :3].T + T[:3, 3]
    h = np.array([float(heading[0]), float(heading[1])])
    h /= np.linalg.norm(h)
    left = np.array([-h[1], h[0]])
    along, across = pts[:, :2] @ h, pts[:, :2] @ left
    near = np.argsort(along, kind="stable")[:near_points]
    front = float(along[near].mean())
    face = along <= front + band_m
    top = float(np.percentile(pts[face, 2], top_pct))
    mid = float(np.median(across[face]))
    point = np.array([*(front * h + mid * left), top])
    return point, dict(
        depths_m=np.round(zs[near], 4).tolist(),
        pixels_uv=np.column_stack((xs[near], ys[near])).tolist(),
        spread_m=float(np.max(np.linalg.norm(pts[near] - pts[near].mean(axis=0), axis=1))),
        face_px=int(face.sum()),
        face_width_m=float(np.ptp(across[face])),
        face_height_m=float(np.ptp(pts[face, 2])),
        near_z=float(pts[near, 2].mean()),
        container_depth_px=stats["valid_depth_px"],
        depth_outliers_px=stats["depth_outliers_px"],
        container_px=int(container.sum()),
    )


def iou_selector(ref_mask, min_iou, mask_iou):
    """A WristCamera.shot `select`: of SAM3's visible `prompt` objects, the one whose mask overlaps
    `ref_mask` (the tapped one, seen from this same still pose) most, if by at least min_iou."""

    def select(response, prompt, labels):
        best = None
        for o in response.detections.objects:
            if o.perception_class != prompt or o.state != STATE_VISIBLE or not o.mask_id:
                continue
            mask = labels == o.mask_id
            iou = mask_iou(mask, ref_mask)
            if iou >= min_iou and (best is None or iou > best[0]):
                best = (iou, mask, float(o.score))
        return None if best is None else (best[1], best[2])

    return select


def take_cfg(cfg, speed: float = 1.0) -> dict:
    """The cfg keys microwave_place's take_out functions read, from this capability's pick_*."""
    scale = max(MIN_SPEED_SCALE, float(cfg["pick_speed_scale"]) * float(speed))
    return {
        "take_approach_speed_scale": scale,
        "take_exit_speed_scale": scale,
        "take_lift": float(cfg["pick_lift_m"]),
        "insert_speed": float(cfg["pick_line_speed"]),
        "impedance_speed": float(cfg["pick_impedance_speed"]),
        "impedance_in": float(cfg["pick_impedance_in_m"]),
        "grip_open": float(cfg["pick_grip_open"]),
        "settle_s": float(cfg["pick_settle_s"]),
        "take_gains": {
            "kq": [float(v) for v in cfg["pick_impedance_kq"]],
            "zeta": float(cfg["pick_impedance_zeta"]),
            "torque_limit": [float(v) for v in cfg["pick_impedance_torque_limit"]],
        },
    }


def open_camera(engine, cfg):
    """wrist_detection.WristCamera (colour + depth buffers, TF, the SAM3 client), this action's.
    PickRefused when SAM3 is not up."""
    import rammp_arm

    from . import wrist_detection as wd

    try:
        return wd.WristCamera(engine, rammp_arm.FRAME, list(cfg["wrist_topics"]),
                              keep_frames=CAMERA_KEEP)
    except wd.NotDetected as exc:
        raise PickRefused(str(exc)) from exc


class FoodTargets:
    """During teleop: the wrist camera's food containers on the UI, and the tap. Make it inside
    execute(); tick() it from the teleop loop."""

    def __init__(self, engine, ctx, camera, cfg, settled, now_ns):
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String

        self._String = String
        self._engine, self._ctx, self._camera = engine, ctx, camera
        self._log = engine.get_logger()
        self.prompt = cfg["pick_prompt"]
        self.threshold = float(cfg["pick_sam3_threshold"])
        self.period = float(cfg["pick_detect_period_s"])
        hz = float(cfg["ui_overlay_hz"])
        self.overlay_period = 1.0 / hz if hz > 0.0 else None
        self._settled = settled  # stamp ns -> measured with the arm still
        self._now_ns = now_ns
        self._batches = {}  # batch -> (detections, objects, stamp ns)
        self._batch = 0
        self._fut, self._fut_t = None, 0.0
        self._next_request = 0.0
        self._next_overlay = 0.0
        self._last_overlay = None
        self._shown = None  # stamp of the batch whose boxes are on the UI
        self._sim_tap = None
        self._bridge = None
        self._state = {"panel_json": self._panel(LOOK), "objects": []}  # what the UI is showing
        engine.create_action_subscription(String, "~/simulate_tap", self._on_sim_tap, 10)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._mirror = engine.create_action_publisher(String, "~/ui_panel", latched)
        self._ui = bool(ctx.update_user_panel(panel_json=self._state["panel_json"], objects=[]))
        if not self._ui:
            self._log.info(
                "no UI session (standalone?): the panel is mirrored on ~/ui_panel; tap with "
                "~/simulate_tap (an instance id, the bridge's object_tapped JSON, or '' for the "
                "best container)"
            )
        self._publish_mirror()

    @staticmethod
    def _panel(message: str) -> dict:
        return {"title": "Fridge", "message": message}

    def _on_sim_tap(self, msg) -> None:  # on the executor: never blocks
        self._sim_tap = sim_tap_id(msg.data)

    def say(self, message: str, objects=None) -> None:
        """Replace the panel's message (and, given, the tappable objects: [] clears them)."""
        if objects is not None and not objects:
            self._shown = None
        self._state["panel_json"] = self._panel(message)
        if objects is not None:
            self._state["objects"] = objects
        if self._ui:
            self._ctx.update_user_panel(panel_json=self._state["panel_json"], objects=objects)
        self._publish_mirror()

    def _publish_mirror(self) -> None:
        """~/ui_panel: the panel and objects the UI shows now, as UPDATE_USER_PANEL's JSON."""
        self._mirror.publish(self._String(data=json.dumps({"ui_session": self._ui, **self._state})))

    def tick(self):
        """Every teleop tick; never blocks. The tapped container (dict(instance_id, mask, score,
        box)) once the user picked one, else None."""
        tap = self._poll_tap()
        if tap is not None:
            return tap
        now = time.monotonic()
        self._overlay(now)
        self._detect(now)
        return None

    # ------------------------------------------------------------------ the tap

    def _poll_tap(self):
        ids = []
        if self._ui:
            event = self._ctx.poll_ui_event()
            if event is not None and event.get("type") == "object_tapped":
                ids.append(event.get("instance_id", ""))
        sim, self._sim_tap = self._sim_tap, None
        if sim:
            ids.append(sim)
        for iid in ids:
            target, why = self._lookup(iid)
            if target is not None:
                self._log.info(f"tapped {iid} (score {target['score']:.2f}): picking it up")
                return target
            self._log.warning(f"tap on {iid!r} ignored: {why}")
            if why.startswith("the camera moved"):
                self.say(f"{why}. Hold still and tap again.")
        return None

    def lookup(self, iid: str):
        """(the container `iid` names -- an instance id, or "best" --, '') or (None, why)."""
        return self._lookup(iid)

    def _lookup(self, iid: str):
        if iid == "best":
            if not self._batches:
                return None, "nothing detected yet"
            b = max(self._batches)
            objs = self._batches[b][1]
            if not objs:
                return None, "no food container in the latest detection"
            k = max(range(len(objs)), key=lambda i: objs[i].score)
        else:
            m = INSTANCE.match(iid)
            if m is None:
                return None, "not one of this action's food containers"
            b, k = int(m[1]), int(m[2])
        entry = self._batches.get(b)
        if entry is None or k >= len(entry[1]):
            return None, "that detection is too old"
        dets, objs, stamp = entry
        if not self._settled(stamp):
            return None, "the camera moved since that container was detected"
        o = objs[k]
        mask = self._camera.labels(dets) == o.mask_id
        if not mask.any():
            return None, "its mask is empty"
        return dict(
            instance_id=instance_id(b, k), mask=mask, score=float(o.score),
            box=[float(v) for v in o.box[:4]],
        ), ""

    # ------------------------------------------------------------------ detection

    def _detect(self, now: float) -> None:
        fut = self._fut
        if fut is not None:
            if fut.done():
                self._fut = None
                self._next_request = now + self.period
                try:
                    response = fut.result()
                except Exception as exc:  # noqa: BLE001 -- a failed request: try the next one
                    self._log.warning(f"SAM3: {exc!r}", throttle_duration_sec=5.0)
                    response = None
                if response is not None:
                    self._on_result(response)
            elif now - self._fut_t > REQUEST_TIMEOUT_S:
                self._camera.cancel_request(fut)
                self._fut = None
                self._next_request = now + self.period
                self._log.warning(f"SAM3 gave no answer in {REQUEST_TIMEOUT_S:.0f} s",
                                  throttle_duration_sec=5.0)
        if self._shown is not None and not self._settled(self._shown):
            self.say(LOOK, objects=[])  # the camera moved: the boxes no longer match the view
        if self._fut is None and now >= self._next_request and self._settled(self._now_ns()):
            self._fut, self._fut_t = self._camera.detect_async(self.prompt, self.threshold), now

    def _on_result(self, response) -> None:
        if not response.success:
            self._log.warning(f"SAM3: {response.message}", throttle_duration_sec=5.0)
            return
        dets = response.detections
        stamp = stamp_ns(dets)
        if not self._settled(stamp):
            return  # the camera moved while it ran (or since)
        objs = [
            o for o in dets.objects
            if o.perception_class == self.prompt and o.state == STATE_VISIBLE and o.mask_id
            and o.score >= self.threshold and o.box[2] > o.box[0] and o.box[3] > o.box[1]
        ]
        self._batch += 1
        b = self._batch
        self._batches[b] = (dets, objs, stamp)
        for old in sorted(self._batches)[:-KEEP_BATCHES]:
            del self._batches[old]
        objects = [
            {
                "instance_id": instance_id(b, k),
                "perception_class": self.prompt,
                "box": [float(v) for v in o.box[:4]],
                "source": "overlay",
            }
            for k, o in enumerate(objs)
        ]
        self.say(TAP if objs else NONE_SEEN, objects=objects)
        self._shown = stamp
        self._debug_overlay(dets, objs, b)

    # ------------------------------------------------------------------ images

    def _overlay(self, now: float) -> None:
        """The live wrist colour frame to the UI (the pixels the objects' boxes refer to)."""
        if not self._ui or self.overlay_period is None or now < self._next_overlay:
            return
        self._next_overlay = now + self.overlay_period
        msg = self._camera.latest_color()
        if msg is not None and msg is not self._last_overlay:
            self._last_overlay = msg
            self._ctx.publish_overlay_image(msg)

    def _debug_overlay(self, dets, objs, b: int) -> None:
        """The detection's frame with its boxes and instance ids on ~/sam3_overlay (Foxglove, and
        the ids for ~/simulate_tap). Must never fail the action."""
        try:
            import cv2
            from cv_bridge import CvBridge

            pair = self._camera.pair(stamp_ns(dets))
            if pair is None:
                return
            if self._bridge is None:
                self._bridge = CvBridge()
            img = self._bridge.imgmsg_to_cv2(pair[0], "rgb8").copy()
            for k, o in enumerate(objs):
                x0, y0, x1, y1 = (int(round(float(v))) for v in o.box[:4])
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 255), 2)
                cv2.putText(img, f"{instance_id(b, k)} {o.score:.2f}", (x0 + 2, max(12, y0 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
            out = self._bridge.cv2_to_imgmsg(img, encoding="rgb8")
            out.header = pair[0].header
            self._engine.sam3_overlay.publish(out)
        except Exception as exc:  # noqa: BLE001 -- visualization only
            self._log.warning(f"food overlay unavailable: {exc}", throttle_duration_sec=10.0)


class FoodPicker:
    """Detect, plan and run one pick from where the arm is. Make it inside execute()."""

    def __init__(self, engine, ctx, cfg, *, execute: bool, speed: float = 1.0, camera=None):
        from microwave_place import place_motion, take_out

        from . import wrist_detection

        self._pm, self._rd, self._to = place_motion, wrist_detection, take_out
        self._engine, self._ctx, self._arm = engine, ctx, engine.arm
        self._log = engine.get_logger()
        self.cfg = cfg
        self.execute = execute
        self.tcfg = take_cfg(cfg, speed)
        self.camera = camera if camera is not None else open_camera(engine, cfg)
        self._model = None

    @property
    def model(self):
        if self._model is None:  # the shared sim: built from the driver's model, inside execute()
            self._model = self._pm.Model()
        return self._model

    # ------------------------------------------------------------------ plan

    def plan(self, target=None) -> dict:
        """Detect the container (`target`: the tapped one; None: the best-scoring), plan the whole
        pick and check it. PickRefused (nothing commanded) when any of it fails."""
        pm, rd, to, c, ctx = self._pm, self._rd, self._to, self.cfg, self._ctx
        log = self._log.info
        try:
            ctx.progress("pick", 0.0, "waiting for the arm to be still")
            self._wait_still()
            world = ""
            if c["pick_world"]:
                world = to.set_approach_world(self._arm, c["pick_world"], log)
            else:
                self._log.warning("pick_world is empty: the approach is planned in whatever world "
                                  "cuRobo has loaded")
            q0 = np.asarray(self._arm.raw_joints(), dtype=float)
            model = self.model
            heading = level_heading(model.fk(q0)[:3, 2])
            ctx.progress("pick", 10.0, "detecting the container")
            select = None
            if target is not None:
                select = iou_selector(target["mask"], float(c["pick_select_iou"]), rd.mask_iou)
            point = None
            if c["pick_point"] == "top":
                band, pct = float(c["pick_face_band_m"]), float(c["pick_top_percentile"])

                def point(depth_m, mask, K, T):
                    return face_top_point(depth_m, mask, K, T, heading, rd.consistent_depth_pixels,
                                          rd.NEAREST_POINTS, band, pct)

            rim = rd.detect_container(
                self.camera, c["pick_prompt"], float(c["pick_sam3_threshold"]),
                int(c["pick_frames"]), float(c["pick_mask_iou"]), float(c["pick_timeout_s"]), log,
                ctx.checkpoint, select=select, point=point,
            )
            ctx.progress("pick", 40.0, "planning the pick")
            ctx.checkpoint()
            cand = to.grasp_pose(model, q0, rim["point"], float(c["pick_reach_in"]),
                                 float(c["pick_grasp_depth"]), float(c["pick_grasp_up"]))
            legs = to.plan_take_out(self._arm, model, q0, cand, self.tcfg)
            corridor = self._check_corridor(model, q0, legs["approach"], cand["pre"])
            leave = self._plan_leave(model, legs["out"][-1], cand["heading"])
        except (pm.Refused, rd.NotDetected) as exc:
            raise PickRefused(str(exc)) from exc
        T_pre = model.fk(legs["approach"][-1])
        summary = dict(
            approach_world=world,
            container_point=np.round(rim["point"], 3).tolist(),
            pick_point=c["pick_point"],
            container_frames_spread_cm=round(rim["info"]["frames_spread_m"] * 100, 1),
            container_mask_iou_min=round(rim["iou_min"], 3),
            tapped=None if target is None else target["instance_id"],
            start_tool=np.round(model.fk(q0)[:3, 3], 3).tolist(),
            heading=np.round(cand["heading"], 3).tolist(),
            pre_grasp=np.round(cand["pre"], 3).tolist(),
            grasp=np.round(cand["target"], 3).tolist(),
            approach_corridor_cm=round(corridor * 100, 1),
            approach_duration_s=round(float(legs["approach_t"][-1])
                                      / self.tcfg["take_approach_speed_scale"], 2),
            fingers_line_from_up_deg=round(
                float(np.degrees(np.arccos(np.clip(T_pre[2, 0], -1.0, 1.0)))), 1),
            lift_cm=round(self.tcfg["take_lift"] * 100, 1),
            withdrawal_end=np.round(model.fk(legs["out"][-1])[:3, 3], 3).tolist(),
            leave_end=None if leave is None else np.round(model.fk(leave[-1])[:3, 3], 3).tolist(),
            leave_to_joints=bool(c["pick_leave_to_joints"]),
            collision_checked=dict(approach=bool(world), line_in=False, lift=False,
                                   withdrawal=False, leave=False),
        )
        self._log.info(f"pick plan: {summary}")
        ctx.progress("pick", 50.0, "planned the pick")
        return dict(q0=q0, cand=cand, legs=legs, summary=jsonable(summary))

    def _wait_still(self) -> None:
        t0, since = time.monotonic(), None
        while time.monotonic() - t0 < STILL_WAIT_S:
            self._ctx.checkpoint()
            ee = self._arm.ee_state()
            still = (float(np.linalg.norm(ee.linear)) < STILL_LIN
                     and float(np.linalg.norm(ee.angular)) < STILL_ANG)
            since = (since or time.monotonic()) if still else None
            if since is not None and time.monotonic() - since >= STILL_HOLD_S:
                return
            time.sleep(0.02)
        raise PickRefused(f"the arm did not come to rest within {STILL_WAIT_S:.0f} s")

    def _check_corridor(self, model, q0, approach, pre) -> float:
        """The farthest the tool gets (m) from the straight line to pre-grasp along cuRobo's
        approach; PickRefused past pick_approach_corridor_m. cuRobo knows the robot's platform, not
        the fridge: a detour is a swing through shelves it cannot see."""
        qs = self._pm.dense([q0, *[np.asarray(q, dtype=float) for q in approach]],
                            self._pm.TRANSIT_STEP_DEG)
        pts = [model.fk(q)[:3, 3] for q in qs]
        worst = float(np.max(segment_distance(pts, pts[0], pre)))
        limit = float(self.cfg["pick_approach_corridor_m"])
        if worst > limit:
            raise self._pm.Refused(
                f"cuRobo's approach takes the tool {worst * 100:.1f} cm off the straight line to "
                f"pre-grasp (> pick_approach_corridor_m {limit * 100:.0f} cm): the fridge is not "
                f"in its world"
            )
        return worst

    def _plan_leave(self, model, q_from, heading):
        """The level straight line pick_leave_back_m back along -heading from q_from, or None."""
        d = float(self.cfg["pick_leave_back_m"])
        if d <= 0.0:
            return None
        pm = self._pm
        h = np.asarray(heading, dtype=float)
        qs = pm.plan_line(model, q_from, -d * h, "leave", step=pm.LINE_STEP)
        a = model.fk(np.asarray(q_from, dtype=float))[:3, 3]
        pm.check_line(model, qs, a, a - d * h, None, "leave")
        return qs

    # ------------------------------------------------------------------ run

    def run(self, plan) -> dict:
        """Run a plan from plan(): the take-out, then leave. EngineError once anything has moved."""
        from capability_engine import EngineError

        pm, to, ctx, c = self._pm, self._to, self._ctx, self.cfg
        log = self._log.info
        model, cand = self.model, plan["cand"]
        try:
            ctx.progress("pick", 55.0, "taking the container")
            out = to.run_take_out(self._arm, model, plan["q0"], cand, plan["legs"], self.tcfg,
                                  ctx, log)
            left_m = 0.0
            if float(c["pick_leave_back_m"]) > 0.0:
                ctx.checkpoint()
                ctx.progress("leave", 80.0, "backing out of the fridge")
                q = np.asarray(self._arm.raw_joints(), dtype=float)
                qs = self._plan_leave(model, q, cand["heading"])  # from where the arm really is
                log(f"-> straight, level back {c['pick_leave_back_m'] * 100:.0f} cm out of the "
                    f"fridge (position control, not collision-checked)")
                to.move_timed(self._arm, model, qs, "the leave", self.tcfg["insert_speed"],
                              "holding the container")
                left_m = float(c["pick_leave_back_m"])
            if c["pick_leave_to_joints"]:
                ctx.checkpoint()
                ctx.progress("leave", 90.0, "to pick_leave_joints")
                log("-> cuRobo to pick_leave_joints (the container's tilt is not held)")
                ok = self._arm.go_to_joint_config(
                    [float(v) for v in c["pick_leave_joints"]], "leave the fridge",
                    speed_scale=self.tcfg["take_approach_speed_scale"],
                )
                if not ok:
                    raise pm.Refused(f"STOPPED holding the container: to pick_leave_joints failed "
                                     f"({self._arm.last_error})")
        except pm.Refused as exc:
            raise EngineError("pick_failed", f"{exc} -- the arm holds where it is") from exc
        tool = model.fk(np.asarray(self._arm.raw_joints(), dtype=float))[:3, 3]
        log(f"pick done: holding the container, tool at {np.round(tool, 3).tolist()}")
        ctx.progress("leave", 100.0, "holding the container outside the fridge")
        return jsonable({**out, "executed": True, "left_m": left_m, "tool": np.round(tool, 3),
                         "plan": plan["summary"]})


def check_params(cfg) -> list[str]:
    """Everything wrong with the pick_* / ui_* parameters (node._problems)."""
    out = []
    for name in ("pick_detect_period_s", "pick_timeout_s", "pick_reach_in", "pick_line_speed",
                 "pick_impedance_speed", "pick_approach_corridor_m", "pick_impedance_zeta"):
        if not float(cfg[name]) > 0.0:
            out.append(f"{name} must be > 0, got {cfg[name]}")
    for name in ("ui_overlay_hz", "pick_lift_m", "pick_leave_back_m", "pick_face_band_m",
                 "pick_settle_s", "pick_impedance_in_m"):
        if float(cfg[name]) < 0.0:
            out.append(f"{name} must be >= 0, got {cfg[name]}")
    if not 0.0 < float(cfg["pick_speed_scale"]) <= 1.0:
        out.append("pick_speed_scale must be in (0, 1]")
    for name in ("pick_grip_open", "pick_sam3_threshold", "pick_mask_iou", "pick_select_iou"):
        if not 0.0 <= float(cfg[name]) <= 1.0:
            out.append(f"{name} must be in [0, 1]")
    if not float(cfg["pick_impedance_in_m"]) < float(cfg["pick_reach_in"]):
        out.append("pick_impedance_in_m must be shorter than pick_reach_in (the line in)")
    if int(cfg["pick_frames"]) < 1:
        out.append("pick_frames must be >= 1")
    if not 0.0 <= float(cfg["pick_top_percentile"]) <= 100.0:
        out.append("pick_top_percentile must be in [0, 100]")
    if cfg["pick_point"] not in PICK_POINTS:
        out.append(f"pick_point must be one of {PICK_POINTS}, got {cfg['pick_point']!r}")
    if len(cfg["wrist_topics"]) != 3:
        out.append("wrist_topics must be [colour, depth aligned to colour, camera_info]")
    for name in ("pick_impedance_kq", "pick_impedance_torque_limit", "pick_leave_joints"):
        v = list(cfg[name])
        if len(v) != 7 or not all(math.isfinite(float(x)) for x in v):
            out.append(f"{name} must be 7 finite numbers (J1..J7)")
    return out
