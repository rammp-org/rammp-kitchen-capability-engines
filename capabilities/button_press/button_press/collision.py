"""Collision checking: the microwave and the counter it stands on, as boxes, in two places.

1. cuRobo's world (PlannerWorld). The long moves (staging, back to start, bail) are planned by
   cuRobo in the driver, and its world used to have no microwave in it (2026-10-14 a staging move
   hit it). The planner takes a world as a scene YAML it reads itself (set_world with a path), so
   the run writes the boxes into `planner_world_dir`, a host directory the planner container mounts
   at the same path, and asks the planner to load it. A pushed scene replaces the robot's own
   world.yaml, so it lists that world's layers again (`base_layers`: flatbot's floor); the
   wheelchair and its user are the planner config's own layers and stay in every world. `restore`
   puts the robot's world back when the action ends. The same pattern as the fridge capability's
   world.py (feature/15-fridge).

2. The straight legs (Clearance). Every leg this capability sends as a joint trajectory (the
   approach, the straight lead-in to staging) is checked before anything moves: its joint
   solutions, interpolated to at most LEG_STEP_DEG per joint, on the driver's own model
   (capability_engine.sim, with the gripper's collision meshes and the fingers closed), at least
   CLEAR_MIN_M from every box. The press strokes and the way back from the button are not checked:
   they touch the panel on purpose. Legs straight out along the panel normal (retreats) are not
   either: they only move away from it.

The boxes come from the panel frame (the button's position and the panel normal) and the Comfee's
front as scene_finder measures it (FRONT_W_M x FRONT_H_M, the button FACE_CORNER_TO_START_MM from
its bottom-right corner). Every box is in arm_mount (rammp_arm.FRAME), upright, turned about up to
face the panel's horizontal normal. cuRobo pads every box by its own world_padding; Clearance does
not. Not modelled: the door handle, the open door, anything else on the counter.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from button_press import Abort, say
from button_press.scene_finder import BUTTON_FROM_DISPLAY_MM, FACE_CORNER_TO_START_MM, FRONT_H_M, FRONT_W_M

UP = np.array([0.0, 0.0, 1.0])
# ---- the microwave (the cream Comfee) and its counter
MICROWAVE_DEPTH_M = 0.40  # front face to back. NOT MEASURED (a Comfee 0.7 cu ft is ~0.34-0.37): measure it
FRONT_PAD_M = 0.01  # the front's width and height grown by this on every side (the finder's front is +-1 cm)
FEET_M = 0.013  # the front's bottom edge above the counter (scene_finder: the corner 1.3 cm above it)
COUNTER_OUT_M = 0.10  # the counter reaches this far out of the front toward the arm. NOT MEASURED
COUNTER_SIDE_M = 0.20  # ... and this far past the microwave on each side
COUNTER_THICK_M = 0.05
# ---- the straight legs
CLEAR_MIN_M = 0.005  # every arm and gripper link at least this far from every box. The pre-press spot puts
# the fingertip >= 1 cm out of the panel (check_fingertip), so this leaves the approach room to end there
LEG_STEP_DEG = 1.0  # a leg's joint solutions interpolated to at most this per joint for the check
MODEL_AGREE_M = 0.005  # the driver's model and the gating sim put the tool frame within this of each other
MODEL_AGREE_DEG = 1.0
SCENE_NAME = "button_press_world.yaml"


def level_axes(normal_out):
    """(out, right, up) in arm_mount: the panel normal made horizontal, the panel's right seen from
    the front, world up. Raises Abort if the normal is too close to vertical."""
    n = np.asarray(normal_out, dtype=float)
    out = np.array([n[0], n[1], 0.0])
    if np.linalg.norm(out) < 0.2:
        raise Abort(f"panel normal {np.round(n, 3)} is too close to vertical to model the microwave")
    out /= np.linalg.norm(out)
    return out, np.cross(UP, out), UP.copy()


def button_from_corner_m(target: str) -> np.ndarray:
    """`target`'s centre from the front's bottom-right corner, (right, up) in m."""
    if target not in BUTTON_FROM_DISPLAY_MM:
        raise Abort(f"no front model for target {target!r}")
    rel = np.asarray(BUTTON_FROM_DISPLAY_MM[target]) - np.asarray(BUTTON_FROM_DISPLAY_MM["start_30s"])
    return (np.asarray(FACE_CORNER_TO_START_MM, dtype=float) + rel) / 1000.0


def _box(name, centre, out, dims):
    """An upright box: centre (arm_mount), x along `out`, dims (along out, right, up)."""
    yaw = float(np.degrees(np.arctan2(out[1], out[0])))
    return {
        "name": name,
        "position": [round(float(v), 5) for v in centre],
        "rpy_deg": [0.0, 0.0, round(yaw, 4)],
        "dims": [round(float(d), 5) for d in dims],
    }


def microwave_boxes(origin, R, target: str) -> list[dict]:
    """The microwave's body and the counter under it, from the panel frame (`origin` the button,
    R[:, 2] the panel normal out) of the button `target`."""
    out, right, up = level_axes(np.asarray(R)[:, 2])
    dx, dy = button_from_corner_m(target)
    corner = np.asarray(origin, dtype=float) - dx * right - dy * up  # the front's bottom-right, on the face
    mid = corner - (FRONT_W_M / 2) * right  # the bottom edge's middle
    body = _box(
        "microwave",
        mid + (FRONT_H_M / 2) * up - (MICROWAVE_DEPTH_M / 2) * out,
        out,
        (MICROWAVE_DEPTH_M, FRONT_W_M + 2 * FRONT_PAD_M, FRONT_H_M + 2 * FRONT_PAD_M),
    )
    top = -FEET_M  # the counter's top, from the bottom edge
    counter = _box(
        "counter",
        mid + (top - COUNTER_THICK_M / 2) * up + ((COUNTER_OUT_M - MICROWAVE_DEPTH_M) / 2) * out,
        out,
        (MICROWAVE_DEPTH_M + COUNTER_OUT_M, FRONT_W_M + 2 * COUNTER_SIDE_M, COUNTER_THICK_M),
    )
    return [body, counter]


def describe(boxes) -> str:
    return "; ".join(
        f"{b['name']} {np.round(np.asarray(b['dims']) * 100, 0).tolist()} cm at {np.round(b['position'], 3).tolist()}"
        for b in boxes
    )


class PlannerWorld:
    """Pushes the boxes into the cuRobo planner (self.arm.set_world) and puts the robot's own world back.

    `write_dir`: where this node writes the scene; `planner_dir`: the same directory as the planner
    sees it ("" or None: the same path). `base_layers`: layer files (as the planner sees them) every
    pushed scene lists, standing in for the world.yaml it replaces. `restore_world`: the world put
    back at the end, resolved by the planner next to its config."""

    def __init__(self, arm, write_dir, planner_dir=None, base_layers=(), restore_world="world.yaml", frame="arm_mount"):
        self.arm = arm
        self.write_dir = Path(write_dir)
        self.planner_dir = Path(planner_dir or write_dir)
        self.base_layers = [x for x in base_layers if x]
        self.restore_world = restore_world
        self.frame = frame
        self.loaded = None  # the boxes the planner holds now (pushed by this action)

    def scene(self, boxes) -> dict:
        return {"base_frame": self.frame, "layers": list(self.base_layers), "obstacles": list(boxes)}

    def push(self, boxes, label):
        """Make `boxes` the planner's world. Raises Abort if it can't be written or the planner refuses it."""
        import yaml

        boxes = list(boxes)
        if boxes == self.loaded:
            return
        if not self.write_dir.is_absolute():
            raise Abort(f"planner_world_dir {self.write_dir} is not absolute: the planner reads the file by its path")
        try:
            self.write_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.write_dir / f".{SCENE_NAME}.tmp"
            tmp.write_text(f"# written by the button_press capability: {label}\n" + yaml.safe_dump(self.scene(boxes)))
            os.replace(tmp, self.write_dir / SCENE_NAME)
        except OSError as e:
            raise Abort(
                f"can't write cuRobo's world into {self.write_dir} ({e}) -- mount planner_world_dir in this "
                "container and the planner's"
            ) from e
        path = self.planner_dir / SCENE_NAME
        ok, msg = self.arm.set_world(str(path))
        if not ok:
            self.loaded = None
            raise Abort(f"the cuRobo planner refused the microwave's collision world {path} ({label}): {msg}")
        self.loaded = boxes
        say(f"  cuRobo world: {describe(boxes)} ({label})")

    def restore(self):
        """Put the robot's own world back, if this action pushed one (also after a cancel, where
        self.arm supports it). Says so if it can't."""
        if self.loaded is None:
            return
        self.loaded = None
        try:
            try:
                ok, msg = self.arm.set_world(self.restore_world, even_if_cancelled=True)
            except TypeError:  # a self.arm without even_if_cancelled (dev before feature/15-fridge)
                ok, msg = self.arm.set_world(self.restore_world)
        except Exception as e:  # noqa: BLE001 -- the action is ending either way
            ok, msg = False, f"{type(e).__name__}: {e}"
        if not ok:
            say(
                f"WARNING: could not put the planner's world back to {self.restore_world}: {msg} -- "
                "it still has the microwave in it"
            )


def _short_delta(a, b):
    d = (np.asarray(b, dtype=float) - np.asarray(a, dtype=float) + np.pi) % (2 * np.pi) - np.pi
    lim = [1, 3, 5]  # J2/J4/J6 can't wrap: their delta as it is
    d[lim] = np.asarray(b, dtype=float)[lim] - np.asarray(a, dtype=float)[lim]
    return d


def dense(qs, step_deg=LEG_STEP_DEG):
    """qs with points added so no joint moves more than step_deg between two (the continuous joints
    the short way round, as the driver interpolates)."""
    qs = [np.asarray(q, dtype=float) for q in qs]
    out = [qs[0]]
    for a, b in zip(qs[:-1], qs[1:], strict=True):
        d = _short_delta(a, b)
        k = max(1, int(np.ceil(np.max(np.abs(d)) / np.radians(step_deg))))
        out += [a + d * (i / k) for i in range(1, k + 1)]
    return out


class Clearance:
    """The boxes as bodies in the process's shared sim (capability_engine.sim: the driver's model,
    rooted at arm_mount), for clearance checks of the arm and the closed gripper. close() removes
    them; use it in a finally."""

    def __init__(self, boxes, sim_robot=None):
        import pybullet as pb
        from capability_engine import sim

        self.sim = sim
        self.scene, self.rb = sim_robot or sim.make_sim()
        self.bodies = []
        cid = self.rb.physics_client_id
        base = self.scene.robot_base_pose
        try:
            for b in boxes:
                shape = pb.createCollisionShape(
                    pb.GEOM_BOX, halfExtents=[d / 2 for d in b["dims"]], physicsClientId=cid
                )
                quat = Rotation.from_euler("xyz", b["rpy_deg"], degrees=True).as_quat()
                pos, orn = pb.multiplyTransforms(
                    list(base.position), list(base.orientation), list(b["position"]), list(quat), physicsClientId=cid
                )
                self.bodies.append(
                    (
                        pb.createMultiBody(0, shape, basePosition=pos, baseOrientation=orn, physicsClientId=cid),
                        b["name"],
                    )
                )
        except BaseException:
            self.close()
            raise

    def check_model(self, kin, q):
        """Refuse (Abort) if the driver's model and the gating sim (`kin`) disagree on the tool frame
        at joints q: then the boxes would be checked against an arm somewhere else."""
        p_sim, q_sim = self.sim.tool_pose(self.scene, self.rb, q)
        p_kin, q_kin = kin.fk(q)
        dp = float(np.linalg.norm(np.asarray(p_sim) - np.asarray(p_kin)))
        dr = float(np.degrees((Rotation.from_quat(q_sim) * Rotation.from_quat(q_kin).inv()).magnitude()))
        if dp > MODEL_AGREE_M or dr > MODEL_AGREE_DEG:
            raise Abort(
                f"collision check: the driver's model puts the tool {dp*1000:.1f} mm / {dr:.1f} deg from the gating "
                "sim's at the same joints (mount_rpy_deg wrong, or the model is not rooted at arm_mount?)"
            )

    def clearance(self, q):
        """(smallest distance m, "link vs box") from every arm and gripper link, fingers closed."""
        return self.sim.clearance(self.rb, q, self.bodies, closed=True)

    def check_leg(self, name, qs, clear_min=CLEAR_MIN_M):
        """Every pose of the leg through joint waypoints qs (densely interpolated) at least clear_min
        from every box. Returns (smallest clearance, where); raises Abort, naming the link and box."""
        best = (9.0, "")
        for q in dense(qs):
            best = min(best, self.clearance(q))
        if best[0] < clear_min:
            raise Abort(
                f"{name}: collision check -- {best[1]} {best[0]*100:.1f} cm (need >= {clear_min*100:.1f}) "
                "-- arm not commanded"
            )
        return best

    def close(self):
        import pybullet as pb

        cid = self.rb.physics_client_id
        for bid, _ in self.bodies:
            pb.removeBody(bid, physicsClientId=cid)
        self.bodies = []
