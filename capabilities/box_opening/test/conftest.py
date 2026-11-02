"""Shared test doubles and fixtures.

FakeClient stands in for runtime.client.PlannerClient: plans are
two-point straight lines from the given start, executions teleport the
live joints to the goal unless scripted, worlds are recorded. FakeStore
stands in for worlds.WorldStore: records every push and returns the
world's name as its path. `leg` and `runner` build Runner inputs; the
`ctx` fixture is a mission context around the shipped container config
with a detected pose on the bench.
"""

import os
import sys
from pathlib import Path

import pytest

# box_opening from this checkout, and capability_engine beside it. Run from the repo root (the
# tests' paths are the root's), with ROS 2 Humble, the RAMMP interfaces and a build of this
# package sourced (its config is read from the install's share/):
#   python3 -m pytest -q capabilities/box_opening/test
# test_node.py needs capability_engine's PyBullet too; the capability's image has everything:
#   make build CAP=box_opening
#   docker run --rm --network host -e ROS_DOMAIN_ID=42 -v $PWD:/repo:ro -w /repo \
#     ghcr.io/rammp-org/rammp-kitchen-capability-engines/box_opening:dev \
#     bash -c "pip install -q pytest && python3 -m pytest -q -p no:cacheprovider capabilities/box_opening/test"
HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "capability_engine")]
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint  # noqa: E402

from box_opening.models.container import ContainerModel, ContainerPose  # noqa: E402
from box_opening.primitives.core import Ctx  # noqa: E402
from box_opening.runtime.legs import Kind, Leg  # noqa: E402
from box_opening.runtime.runner import Runner  # noqa: E402

# no test run records the lid-off's video of the cameras (perception/camera_video) unless it asks
os.environ.setdefault("BOX_OPENING_VIDEO", "0")

CFG = "capabilities/box_opening/config/containers/oxo_pop.yaml"

Q0 = [0.0] * 7
Q1 = [0.1] * 7
Q2 = [0.2] * 7


def traj(start, end, dt=1.0):
    t = JointTrajectory()
    t.joint_names = ["joint_%d" % i for i in range(1, 8)]
    for i, row in enumerate([start, end]):
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in row]
        p.velocities = [0.0] * 7
        p.accelerations = [0.0] * 7
        p.time_from_start.sec = int(i * dt)
        t.points.append(p)
    return t


class FakeClient:
    def __init__(self):
        self.live = list(Q0)
        self.efforts = True
        self.armed = True  # the client's motion latch (only --execute sets it)
        self.joint_names = ["joint_%d" % i for i in range(1, 8)]
        self.gripper_sent = []  # every gripper command that went out
        self.executed = []  # (n_points, speed)
        self.exec_starts = []  # first waypoint of each executed trajectory
        self.guards = []  # the guard object each execution carried (or None)
        self.worlds_pushed = []
        self.plans = []  # scripted plan_to_* responses (FIFO), else auto
        self.exec_script = []  # scripted execute outcomes (FIFO)
        self.approach_offsets = []  # per plan_to_pose call
        self.joint_starts = []  # per plan_to_joints call
        self.joint_targets = []  # the q7 each plan_to_joints was asked for
        self.contact_at = [0.45, 0.0, 0.085]  # fingertip TF at a guard trip
        self.primes = []  # the joints each prime_compliant was handed (None: the live ones)
        self.stills = []  # each wait_still: how many executions had gone out by then

    def joints(self):
        return list(self.live)

    def prime_compliant(self, q=None):
        self.primes.append(None if q is None else [float(v) for v in q])
        return True

    def wait_still(self):
        self.stills.append(len(self.executed))
        return 0.0, False  # standing still at once: nothing settles here

    def wrist_efforts(self):
        return [0.0] * 4 if self.efforts else None

    def efforts_present(self):
        return self.efforts

    def settled_efforts(self, duration_s=0.15, settle_s=0.15, timeout_s=2.0):
        return ([0.0] * 4, list(self.live)) if self.efforts else None

    def contact_xyz(self, timeout_s=0.25):
        return list(self.contact_at) if self.contact_at is not None else None

    def _plan(self, end, start):
        class R:
            success = True
            message = "ok"

        R.trajectory = traj(start if start else self.live, end)
        return R

    def plan_to_pose(self, xyz, quat_xyzw, start_joints, approach_offset_m=0.0):
        self.approach_offsets.append(float(approach_offset_m))
        if self.plans:
            return self.plans.pop(0)
        return self._plan(Q1, start_joints)

    def plan_to_joints(self, q7, start_joints):
        self.joint_starts.append(list(start_joints))
        self.joint_targets.append([float(v) for v in q7])
        if self.plans:
            return self.plans.pop(0)
        return self._plan(list(q7), start_joints)

    def execute(self, traj_, speed, guard=None, while_running=None, stop_when=None, gains=None):
        self.executed.append((len(traj_.points), speed))
        self.gains_sent = getattr(self, "gains_sent", []) + [gains]
        self.exec_starts.append(list(traj_.points[0].positions))
        self.guards.append(guard)
        info_extra = {}
        if stop_when is not None:
            stop_when(0.5)  # the real client polls it while the motion flies
        if while_running is not None and guard is None:
            try:
                info_extra["while_running"] = while_running()
            except Exception as exc:
                info_extra["while_running_error"] = str(exc)
        if self.exec_script:
            outcome, info = self.exec_script.pop(0)
        else:
            outcome, info = (
                "arrived",
                {
                    "message": "ok",
                    "progress": 1.0,
                    "torque_peak": 0.0,
                },
            )
        if outcome != "failed":
            self.live = list(traj_.points[-1].positions)
        info = dict(info)
        info.update(info_extra)
        return outcome, info

    def set_world(self, path_or_name):
        self.worlds_pushed.append(str(path_or_name))
        return True, "ok"

    def motion_enabled(self):
        return self.armed

    def gripper_send(self, position):
        self.gripper_sent.append(float(position))
        return ("handle", float(position))

    def gripper_join(self, handle):
        return True, float(handle[1]), False

    def gripper_cmd(self, position):
        return True, float(position if position is not None else 0.0), False


class FakeStore:
    def __init__(self):
        self.pushes = []  # (kind, kwargs) — world-shape assertions
        self.table_extent = None  # the surveyed bench: no measured table
        self.table_top_z = -0.027  # world_bench.yaml's surveyed table top
        self.base_frame = "base_link"  # the bench's planning frame (WorldStore.base_frame)

    def push_name(self, kind, **kw):
        self.pushes.append((kind, kw))
        tag = kw.get("tag", "")
        name = kind + (("_" + tag) if tag else "")
        return name, name + ".yaml"


def leg(
    name,
    start=Q0,
    end=Q1,
    chain=0,
    speed=0.25,
    world="full",
    guard=None,
    kind=Kind.MOTION,
    verify=None,
    cmd=None,
):
    return Leg(
        name=name,
        kind=kind,
        traj=traj(start, end) if kind is Kind.MOTION else None,
        speed=speed,
        guard=guard,
        world=world,
        chain=chain,
        target=("joints", end) if kind is Kind.MOTION else None,
        goal_joints=end if kind is Kind.MOTION else None,
        verify=verify,
        gripper_cmd=cmd,
    )


def runner(client, tmp_path):
    r = Runner(client, log_dir=tmp_path)
    return r


# where the lid goes in the `ctx` fixture: a run resolves its own before the
# press (reach_lid_off.resolve_lid_drop); nowhere is configured to fall back to
LID_SPOT = ContainerPose(xyz=(0.45, -0.25, -0.027), yaw=0.0)


@pytest.fixture
def ctx():
    """A mission context: the shipped container, detected on the bench,
    its lid's spot resolved (LID_SPOT)."""
    return Ctx(
        model=ContainerModel.load(CFG),
        cpose=ContainerPose(xyz=(0.45, 0.0, -0.07), yaw=0.0),
        client=FakeClient(),
        worlds=FakeStore(),
        config_path=CFG,
        lid_drop=LID_SPOT,
    )
