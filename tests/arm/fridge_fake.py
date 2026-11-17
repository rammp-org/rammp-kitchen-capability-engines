"""A CPU stand-in for self.arm's cuRobo planner and driver, for the fridge's planning tests (no GPU in this image).

FakeCurobo answers the calls the fridge's planning makes, on the flatbot model these tests run on: plan_to_pose solves
the goal with the node's own local IK (kinematics.Chain) and goes there in a straight joint-space line; blend joins the
legs end to end; set_world records the world it is given; execute_joint_trajectory jumps to the trajectory's end;
gripper records its call. It checks NO collisions -- that is cuRobo's, checked in data/curobo_sim's GPU harness: these
tests cover the control flow, the worlds pushed, the trajectories sent and the files written. `unreachable(pos, quat,
world)` lets a test say which goals cuRobo finds no arm configuration for. Nothing here can reach a real arm.
"""

import json
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import yaml
from door_common import common as dc
from door_common import open_door as shared
from fridge import constants, motion, planning, world
from fridge.kinematics import Chain
from fridge.planning import Planner
from fridge.world import PlannerWorld

URDF = Path("/fixtures/urdf/flatbot.urdf")  # the 30-deg flatbot arm (rammp-arm-test's fixture)
FRIDGE_START = np.radians((98.8, 10.2, 179.5, -94.7, 98.7, 100.2, 16.1))  # the web app's Fridge pose
HANDLE = np.array([0.85, -0.20, 0.30])  # in the placement that passed (MOUNTS[30]), the 11-01 handle height
NORMAL = np.array([-1.0, 0.0, 0.0])  # the door square to the arm
HANDLE_QUAT = np.array([-0.5, 0.5, 0.5, -0.5])  # door_common.perception.HANDLE_QUAT
IK_FAIL = "MotionGenStatus.IK_FAIL: no collision-free joint solution AT the goal"


@dataclass
class Plan:
    success: bool
    message: str
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    q: np.ndarray = field(default_factory=lambda: np.zeros((0, 7)))
    qd: np.ndarray = field(default_factory=lambda: np.zeros((0, 7)))
    planning_time: float = 0.0
    corner_m: np.ndarray = field(default_factory=lambda: np.zeros(0))


def door_is_open(obstacles):
    """True when `obstacles` (a pushed world) holds a door slab turned off the cabinet (open), wherever it is: also
    one shifted by a planned wheelchair move (door_where_it_stands tells them apart)."""
    by = {b["name"]: b for b in obstacles}
    if "door" not in by or "body" not in by:
        return False
    return abs(by["door"]["rpy_deg"][2] - by["body"]["rpy_deg"][2]) > 1.0


def door_where_it_stands(obstacles):
    """True when `obstacles` holds the open door where the door file puts it: not shifted by a wheelchair move. The
    close plans against the door shifted by a move under the same box names (the after-move check, 11-12), so a fake
    cuRobo that means "the door where it stands now is in the way" must check where the door is, not its name."""
    if not door_is_open(obstacles) or not dc.DOOR_FILE.exists():
        return False
    door = json.loads(dc.DOOR_FILE.read_text())
    want = world.FridgeModel(door).slab(float(door["door_open_deg"]))[0]["position"]
    got = next(b for b in obstacles if b["name"] == "door")["position"]
    return float(np.linalg.norm(np.subtract(got[:2], want[:2]))) < 1e-3


class FakeCurobo:
    def __init__(self, q0=FRIDGE_START):
        self.urdf = URDF.read_text()
        self.fk = Chain(self.urdf)
        self.q, self.gripper_pos, self.estopped = np.asarray(q0, float), 0.0, False
        self.worlds = []  # every world pushed: its obstacle dicts (or the name, for a named world)
        self.world_calls = []  # (world, even_if_cancelled)
        self.sent = []  # execute_joint_trajectory's keyword arguments, per trajectory
        self.grips = []  # (position, force, speed)
        self.blends = []  # (number of legs, radii, speed scales, holds)
        self.pose_goals = []  # (pos, the world it was planned in)
        self.unreachable = lambda pos, quat, obstacles: False
        self.start_blocked = False
        self.blend_fails = False

    # -- state --
    def get_state(self):
        p, quat = self.fk.pose(self.q)
        return {
            "position": list(self.q),
            "ee_pos": [*p, *quat],
            "gripper_pos": self.gripper_pos,
            "effort": [0.0] * 7,
            "velocity": [0.0] * 7,
        }

    def robot_description(self):
        return self.urdf

    def control_status(self):  # base_drive watches it for the arm's e-stop
        return types.SimpleNamespace(estopped=self.estopped)

    # -- the planner --
    def _to(self, start, q_goal):
        t, q, qd = planning.timed(np.asarray(start, float), np.asarray([q_goal], float))
        return Plan(True, "", t, q, qd)

    def plan_to_pose(self, pos, quat, start_joints, hold=0):
        obstacles = self.worlds[-1] if self.worlds else []
        self.pose_goals.append((np.asarray(pos, float), obstacles))
        if self.start_blocked:
            return Plan(False, "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION: the start state collides")
        if self.unreachable(np.asarray(pos, float), np.asarray(quat, float), obstacles):
            return Plan(False, IK_FAIL)
        if hold:  # as cuRobo keeps a hold (HOLD_LEVEL: the tilt) all the way: along a straight tool path, the hand
            # turning only from the start's orientation to the goal's (about vertical, for a level hold)
            p0, q0 = self.fk.pose(start_joints)
            path, _ = planning.path_poses(planning.Goal(pos, quat, "", [], path="line"), p0, q0)
            qs, q = [], np.asarray(start_joints, float)
            for pp, qq in path:
                q, _ = self.fk.ik(pp, qq, q)
                if q is None:
                    return Plan(False, IK_FAIL)
                qs.append(q)
            t, qq_, qd = planning.timed(np.asarray(start_joints, float), np.asarray(qs))
            return Plan(True, "", t, qq_, qd)
        q, _ = self.fk.ik(pos, quat, start_joints)
        return Plan(False, IK_FAIL) if q is None else self._to(start_joints, q)

    def plan_to_joints(self, q_goal, start_joints):
        # cuRobo checks a joint goal's start and the goal itself in its world as it does a pose goal's: the same hooks
        # (a test's unreachable / start_blocked, or its own plan_to_pose standing in for cuRobo) apply
        pos, quat = self.fk.pose(np.asarray(q_goal, float))
        if "plan_to_pose" in self.__dict__:  # a test's stand-in
            res = self.plan_to_pose(pos, quat, start_joints)
            if not res.success:
                return res
        obstacles = self.worlds[-1] if self.worlds else []
        if self.start_blocked:
            return Plan(False, "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION: the start state collides")
        if self.unreachable(np.asarray(pos, float), np.asarray(quat, float), obstacles):
            return Plan(False, IK_FAIL)
        return self._to(start_joints, q_goal)

    def blend(self, legs, blend_radius, speed_scale=None, hold=None, hold_orientation=None):
        legs = list(legs)
        self.blends.append((len(legs), list(blend_radius), speed_scale, hold))
        if self.blend_fails:
            return Plan(False, "BAD_LEG: the fake refuses")
        q = np.vstack([legs[0].q] + [leg.q[1:] for leg in legs[1:]])
        t = (np.arange(len(q)) + 1) * 0.02
        qd = np.gradient(q, t, axis=0)
        qd[0] = qd[-1] = 0.0
        return Plan(True, "", t, q, qd, corner_m=np.zeros(len(legs) - 1))

    def set_world(self, name, even_if_cancelled=False):
        self.world_calls.append((name, even_if_cancelled))
        path = Path(name)
        self.worlds.append(yaml.safe_load(path.read_text())["obstacles"] if path.is_file() else name)
        return True, ""

    # -- the driver --
    def execute_joint_trajectory(self, names, points, **kw):
        points = list(points)
        assert np.degrees(np.max(np.abs(np.asarray(points[0][1]) - self.q))) < 1.0, "starts away from the arm"
        self.sent.append(kw)
        self.q = np.asarray(points[-1][1], float)
        return types.SimpleNamespace(error_code=0, error_string="", final_error=np.zeros(7))

    def gripper(self, position, timeout_s=4.0, force=None, *, speed=None):
        self.grips.append((position, force, speed))
        self.gripper_pos = 0.8 if position > 0.5 else 0.0
        return True


def _events_now(events, on_error):
    """planning._timed_events for the tests: every event at once, in order (the fake arm takes no time)."""
    for _, fn in sorted(events, key=lambda e: e[0]):
        fn()
    return types.SimpleNamespace(join=lambda timeout=None: None, cancel=lambda: None)


@pytest.fixture
def fridge(tmp_path, monkeypatch):
    """The fridge configured for the 30-deg mount, its detection the handle at HANDLE on a door square to the arm, the
    fake planner and arm at the Fridge pose: (arm, planner, world). Every module constant restored afterwards."""
    saved = [dict(vars(m)) for m in (dc, shared, constants, world, motion)]
    from fridge import open_door

    open_door.configure(tmp_path, 30)
    shared.configure(HANDLE_SOURCE="depth")
    monkeypatch.setattr(shared, "detect_handle", lambda: (HANDLE.copy(), HANDLE_QUAT, NORMAL.copy(), None, None, None))
    monkeypatch.setattr(planning, "_timed_events", _events_now)
    dc.set_camera(type("NoFrames", (), {"grab": lambda self: None})())
    arm = FakeCurobo()
    planner_world = PlannerWorld(arm, tmp_path / "worlds", base_layers=["/config/robot/curobo/layer_mebot_floor.yaml"])
    yield arm, Planner(arm, planner_world, arm.fk), planner_world
    dc.set_camera(None)
    for m, v in zip((dc, shared, constants, world, motion), saved, strict=True):
        vars(m).update(v)
