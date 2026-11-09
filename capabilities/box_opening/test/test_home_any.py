"""HOME from wherever the arm stands (primitives.home_any, EscapeBodyFree; 2026-11-09, the temporary HOME with
the arm 8 in forward on flatbot's chair): a start inside the chair and user's padding steps out of it first,
never deeper and the tool held; HOME is asked for again; a waypoint along the joint line is tried; and where
nothing plans, the refusal says so."""

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import FakeClient, flatbot_urdf

from box_opening.constants import HOME
from box_opening.primitives import core
from box_opening.runtime import approach, body
from box_opening.runtime.kinematics import mat_to_quat_xyzw
from box_opening.tasks import cli_common

PKG = Path(__file__).resolve().parents[1]
FLATBOT_WORLD = str(PKG / "config" / "world_flatbot.yaml")
CONTAINER = str(PKG / "config" / "containers" / "ankou_pink.yaml")
DOWN = mat_to_quat_xyzw(np.diag([1.0, -1.0, -1.0]))
START = [0.3, -0.9, -2.6, -1.8, -0.4, -0.7, 1.9]
ROBOT = """robot_cfg:
  kinematics:
    base_link: arm_mount
    collision_sphere_buffer: 0.005
    collision_spheres:
      arm_forearm_link:
      - {center: [0.0, 0.0, 0.0], radius: 0.06}
      arm_bracelet_link:
      - {center: [0.0, 0.0, -0.045], radius: 0.045}
      arm_robotiq_85_base_link:
      - {center: [0.0, 0.0, 0.04], radius: 0.05}
      - {center: [0.0, 0.0, 0.10], radius: 0.035}
"""


@pytest.fixture(autouse=True)
def chain_state(monkeypatch, tmp_path):
    monkeypatch.setenv("BOX_OPENING_STATE", str(tmp_path))
    saved = (approach._ROOT, approach._URDF_PATH, approach._CHAIN, approach._GRAVITY, approach._BODY)
    yield
    approach._ROOT, approach._URDF_PATH, approach._CHAIN, approach._GRAVITY, approach._BODY = saved


def _no_chain_ctx(monkeypatch):
    """A ctx with no arm chain: no body model, no joint limits to step off (the planner's word alone)."""
    monkeypatch.setattr(approach, "chain_ready", lambda: (False, "no URDF in this test"))
    client = SimpleNamespace(joints=lambda: list(START), joint_names=["joint_%d" % i for i in range(1, 8)])
    return SimpleNamespace(client=client)


class Planner:
    """plan_to's stand-in: refuses HOME `home_fails` times from the start, and from anywhere when `from_start_only`
    is False; plans every waypoint. Records (name, start joints, goal) per ask."""

    def __init__(self, home_fails=0, from_start_only=True, waypoints_fail=False):
        self.home_fails, self.from_start_only, self.waypoints_fail = home_fails, from_start_only, waypoints_fail
        self.asked = []

    def __call__(self, at, name, q):
        self.asked.append((name, list(at.joints), list(q)))
        if name == "home":
            from_start = np.allclose(at.joints, START)
            if from_start and self.home_fails > 0:
                self.home_fails -= 1
                raise RuntimeError("planning failed for home: MotionGenStatus.FINETUNE_TRAJOPT_FAIL")
            if not from_start and not self.from_start_only:
                raise RuntimeError("planning failed for home: MotionGenStatus.FINETUNE_TRAJOPT_FAIL")
        elif self.waypoints_fail:
            raise RuntimeError("planning failed for %s" % name)
        return SimpleNamespace(name=name, goal_joints=list(q))


def test_home_is_asked_for_again_before_anything_else(monkeypatch):
    ctx = _no_chain_ctx(monkeypatch)
    plan = Planner(home_fails=core.HOME_TRIES - 1)
    legs = core.home_any(ctx, core.PlanState(joints=list(START), chain=0), HOME, plan, say=lambda s: None)
    assert [leg.name for leg in legs] == ["home"]
    assert [a[0] for a in plan.asked] == ["home"] * core.HOME_TRIES


def test_where_home_never_plans_from_the_start_it_goes_through_a_waypoint_half_way_along_the_joint_line(monkeypatch):
    ctx = _no_chain_ctx(monkeypatch)
    plan = Planner(home_fails=99)
    said = []
    legs = core.home_any(ctx, core.PlanState(joints=list(START), chain=0), HOME, plan, say=said.append)
    assert [leg.name for leg in legs] == ["home-waypoint", "home"]
    waypoint = legs[0].goal_joints
    # half way, each continuous joint the short way round (joint_3: -2.6 to HOME's -2.9033, not a turn)
    assert waypoint == pytest.approx([(a + b) / 2 for a, b in zip(START, HOME, strict=True)], abs=1e-9)
    assert plan.asked[-1] == ("home", waypoint, list(HOME))  # HOME planned from the waypoint
    assert any("through a waypoint 50%" in s for s in said)


def test_where_nothing_plans_the_refusal_names_what_was_tried(monkeypatch):
    ctx = _no_chain_ctx(monkeypatch)
    plan = Planner(home_fails=99, from_start_only=False)
    with pytest.raises(
        RuntimeError,
        match=r"did not plan from here, 3 times directly nor through a waypoint "
        r"50%/25%/75% of the way .*FINETUNE_TRAJOPT_FAIL",
    ):
        core.home_any(ctx, core.PlanState(joints=list(START), chain=0), HOME, plan, say=lambda s: None)


def _body_ctx(monkeypatch, tmp_path, boxes):
    """flatbot's ctx (bench world, FakeClient) with a body model of `boxes` attached to its chain."""
    client = FakeClient()
    monkeypatch.setattr(cli_common, "init_runtime", lambda execute: (object(), client))
    monkeypatch.setattr(cli_common, "_arm_urdf_path", lambda node, frame: flatbot_urdf())
    ctx, _runner = cli_common.build_ctx(Namespace(container=CONTAINER, bench_world=FLATBOT_WORLD, execute=False))
    lines = ["layer: test", "obstacles:"]
    lines += ["  - {name: %s, position: %s, dims: %s}" % (n, list(c), list(d)) for n, c, d in boxes]
    (tmp_path / "layer_test.yaml").write_text("\n".join(lines) + "\n")
    (tmp_path / "robot.yaml").write_text(ROBOT)
    (tmp_path / "planner.yaml").write_text("robot: robot.yaml\nlayers: [layer_test.yaml]\n")
    approach.set_body(body.load(str(tmp_path / "planner.yaml")))
    approach.set_chain_root("arm_mount", flatbot_urdf())
    return ctx, approach._chain(), client


def test_a_start_inside_the_padding_steps_out_first_never_deeper_the_tool_held_and_home_plans_from_its_end(
    monkeypatch, tmp_path
):
    # a slab whose padding the hand stands in, 0.45 over a point 0.6 out in front of the arm
    ctx, chain, client = _body_ctx(monkeypatch, tmp_path, [("slab", (0.6, 0.0, 0.40), (0.30, 0.30, 0.06))])
    q = chain.nearest_solution([0.0, 0.3, 0.0, 1.8, 0.0, 1.0, 1.57], [0.6, 0.0, 0.45], DOWN)
    assert q is not None
    client.live = list(q)
    start_depth = float(chain.body.sphere_depths([q])[0].max())
    assert start_depth > 0.0  # inside the padding: the planner would refuse this start
    plan = Planner()
    said = []
    legs = core.home_any(ctx, core.PlanState(joints=list(q), chain=0), HOME, plan, say=said.append)
    assert [leg.name for leg in legs] == ["escape-body", "home"]
    pts = np.array([p.positions for p in legs[0].traj.points])
    depth = chain.body.sphere_depths(pts)[0].max(axis=1)
    assert np.all(np.diff(depth) <= 1e-6)  # never deeper on the way out
    assert depth[-1] <= -core.ESCAPE_CLEAR_M  # out by the margin
    z = np.array([chain.fk(p)[1][2] for p in pts])
    assert z.min() >= z[0] - core.ESCAPE_DROP_M - 1e-9  # the tool held
    assert legs[0].guard.trip == "obstruction"
    assert plan.asked == [("home", list(pts[-1]), list(HOME))]  # HOME planned from where it ends
    assert any("inside the chair and user's padding" in s for s in said)


def test_a_start_clear_of_the_padding_takes_no_step_out(monkeypatch, tmp_path):
    ctx, chain, client = _body_ctx(monkeypatch, tmp_path, [("far", (-0.8, 0.8, 0.2), (0.1, 0.1, 0.1))])
    q = chain.nearest_solution([0.0, 0.3, 0.0, 1.8, 0.0, 1.0, 1.57], [0.6, 0.0, 0.45], DOWN)
    client.live = list(q)
    assert not core.EscapeBodyFree().needed(q)
    legs = core.home_any(ctx, core.PlanState(joints=list(q), chain=0), HOME, Planner(), say=lambda s: None)
    assert [leg.name for leg in legs] == ["home"]
