"""The chair and its user, for the lines the planner never sees (runtime/body): every plan-free
line that enters a padded box is refused like an unsolvable one, the lid spot passes over a spot
whose carry or set-down would, and a body model that cannot be read stops the run before it moves."""

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import FakeClient, flatbot_urdf, traj

from box_opening.constants import HOME
from box_opening.models.container import ContainerPose
from box_opening.primitives import core
from box_opening.runtime import approach, body
from box_opening.runtime.kinematics import ArmChain, mat_to_quat_xyzw
from box_opening.tasks import cli_common, press_demo
from box_opening.tasks import reach_lid_off as rlo
from box_opening.tasks import reach_press as rp

PKG = Path(__file__).resolve().parents[1]
FLATBOT_WORLD = str(PKG / "config" / "world_flatbot.yaml")
CONTAINER = str(PKG / "config" / "containers" / "ankou_pink.yaml")
DOWN = mat_to_quat_xyzw(np.diag([1.0, -1.0, -1.0]))  # the hand straight down
ROBOT = """robot_cfg:
  kinematics:
    base_link: arm_mount
    collision_sphere_buffer: 0.005
    collision_spheres:
      arm_shoulder_link:
      - {center: [0.0, 0.0, -0.1], radius: 0.07}
      arm_forearm_link:
      - {center: [0.0, -0.1, 0.0], radius: 0.06}
      arm_robotiq_85_base_link:
      - {center: [0.0, 0.0, 0.05], radius: 0.04}
      arm_robotiq_85_left_finger_tip_link:
      - {center: [-0.02, 0.0, 0.03], radius: 0.02}
"""


@pytest.fixture(autouse=True)
def chain_state(monkeypatch, tmp_path):
    monkeypatch.setenv("BOX_OPENING_STATE", str(tmp_path))
    saved = (approach._ROOT, approach._URDF_PATH, approach._CHAIN, approach._GRAVITY, approach._BODY)
    yield
    approach._ROOT, approach._URDF_PATH, approach._CHAIN, approach._GRAVITY, approach._BODY = saved


def _model_files(tmp, boxes, robot=ROBOT, padding=None):
    """A planner config in `tmp` with one layer of `boxes` [(name, centre, dims)]: its path."""
    lines = ["layer: test", "obstacles:"]
    lines += ["  - {name: %s, position: %s, dims: %s}" % (n, list(c), list(d)) for n, c, d in boxes]
    (tmp / "layer_test.yaml").write_text("\n".join(lines) + "\n")
    (tmp / "robot.yaml").write_text(robot)
    pad = "" if padding is None else "world_padding: %s\n" % padding
    (tmp / "planner.yaml").write_text("robot: robot.yaml\nlayers: [layer_test.yaml]\n" + pad)
    return str(tmp / "planner.yaml")


def _chain(tmp, boxes, **kw):
    """flatbot's chain with a body model of `boxes` attached, as approach attaches it."""
    approach.set_body(body.load(_model_files(tmp, boxes, **kw)))
    approach.set_chain_root("arm_mount", flatbot_urdf())
    return approach._chain()


def _over_table(chain):
    """Joints with the hand straight down 0.45 over a point 0.6 out in front of the arm."""
    q = chain.nearest_solution(list(HOME), [0.6, 0.0, 0.45], DOWN)
    assert q is not None
    return q


def test_the_override_beats_the_param_and_empty_is_off():
    assert body.resolve(None, env={}) == body.DEFAULT_BODY_LAYERS == "/config/robot/curobo/planner.yaml"
    assert body.resolve("/x/planner.yaml", env={}) == "/x/planner.yaml"
    assert body.resolve("/x/planner.yaml", env={body.BODY_LAYERS_ENV: "/y/p.yaml"}) == "/y/p.yaml"
    assert body.resolve("/x/planner.yaml", env={body.BODY_LAYERS_ENV: ""}) == ""
    assert body.resolve("", env={}) == ""


def test_a_body_model_that_cannot_be_read_is_an_error_never_a_skip(tmp_path):
    with pytest.raises(body.BodyModelError, match="cannot read"):
        body.load(tmp_path / "missing.yaml")
    (tmp_path / "planner.yaml").write_text("robot: robot.yaml\nlayers: []\n")
    with pytest.raises(body.BodyModelError, match="names no layers"):
        body.load(tmp_path / "planner.yaml")
    with pytest.raises(body.BodyModelError, match="hold no obstacle"):
        body.load(_model_files(tmp_path, []))
    with pytest.raises(body.BodyModelError, match="no collision_spheres"):
        body.load(_model_files(tmp_path, [("b", (1, 1, 1), (0.1, 0.1, 0.1))], robot="robot_cfg: {kinematics: {}}\n"))
    (tmp_path / "planner.yaml").write_text("robot: robot.yaml\nlayers: [nowhere.yaml]\n")
    with pytest.raises(body.BodyModelError, match="cannot read .*nowhere.yaml"):
        body.load(tmp_path / "planner.yaml")


def test_the_boxes_are_padded_as_the_planner_pads_them_and_the_base_and_shoulder_are_exempt(tmp_path):
    m = body.load(_model_files(tmp_path, [("b", (1.0, 0.0, 0.5), (0.2, 0.4, 0.6))]))
    assert m.padding_m == body.WORLD_PADDING_M == 0.02 and m.buffer_m == 0.005
    assert m.lo[0].tolist() == pytest.approx([0.88, -0.22, 0.18]) and m.hi[0].tolist() == pytest.approx(
        [1.12, 0.22, 0.82]
    )
    assert "arm_shoulder_link" not in m.spheres and m.spheres["arm_forearm_link"][1].tolist() == pytest.approx([0.065])
    assert body.load(_model_files(tmp_path, [("b", (1, 0, 0), (0.1, 0.1, 0.1))], padding=0.03)).padding_m == 0.03


def test_a_box_in_another_frame_than_the_chain_or_a_link_the_urdf_lacks_stops_the_load(tmp_path):
    m = body.load(_model_files(tmp_path, [("b", (1, 0, 0), (0.1, 0.1, 0.1))]))
    chain = ArmChain(root="arm_mount", path=flatbot_urdf())
    with pytest.raises(body.BodyModelError, match="boxes are in arm_mount, the arm's chain in base_link"):
        m.check_for(chain, flatbot_urdf(), "base_link")
    m = body.load(
        _model_files(
            tmp_path,
            [("b", (1, 0, 0), (0.1, 0.1, 0.1))],
            robot=ROBOT + "      arm_camera_link:\n      - {center: [0, 0, 0], radius: 0.03}\n",
        )
    )
    with pytest.raises(body.BodyModelError, match="places no arm_camera_link"):
        m.check_for(chain, flatbot_urdf(), "arm_mount")
    path = _model_files(tmp_path, [("b", (1, 0, 0), (0.1, 0.1, 0.1))])
    layer = tmp_path / "layer_test.yaml"
    layer.write_text("meta: {frame: arm_mount}\n" + layer.read_text())
    assert body.load(path).names == ["test/b"]  # the generated layers' frame, the spheres' base_link
    layer.write_text(layer.read_text().replace("frame: arm_mount", "frame: arm_base_link"))
    with pytest.raises(
        body.BodyModelError, match="layer test is in arm_base_link, the robot file's spheres in arm_mount"
    ):
        body.load(path)


def test_a_straight_line_into_a_box_is_refused_naming_the_link_and_the_box(tmp_path):
    chain = _chain(tmp_path, [("far", (-1.0, -1.0, -1.0), (0.1, 0.1, 0.1))])
    q = _over_table(chain)
    pts, why = chain.straight_line(q, [0.6, 0.0, 0.30], DOWN)
    assert pts is not None and why is None  # nothing in the way
    chain = _chain(tmp_path, [("block", (0.6, 0.0, 0.30), (0.10, 0.10, 0.02))])
    pts, why = chain.straight_line(q, [0.6, 0.0, 0.30], DOWN)
    assert pts is None
    assert "the chair and user: arm_robotiq_85_" in why and "into test/block (padded 20 mm + 5 mm)" in why


def test_a_flown_walk_into_a_box_is_refused_and_reach_never_flown_is_not(tmp_path):
    chain = _chain(tmp_path, [("block", (0.6, 0.0, 0.30), (0.10, 0.10, 0.02))])
    q = _over_table(chain)
    pts, why = chain.walk(q, [0.6, 0.0, 0.30], DOWN, step_m=0.005)
    assert pts is None and "test/block" in why
    assert chain.reach(q, [0.6, 0.0, 0.30], DOWN) is not None  # the planner's to check


def test_a_line_from_inside_a_box_may_leave_it_but_not_go_deeper(tmp_path):
    chain = _chain(tmp_path, [("slab", (0.6, 0.0, 0.42), (0.30, 0.30, 0.06))])
    q = _over_table(chain)  # the hand 0.45 over the table: its fingertips in the slab
    assert chain.body.first_hit([q], start_exempt=False) is not None
    assert chain.body_refusal([q]) is None  # the arm already stands there
    up, why = chain.straight_line(q, [0.6, 0.0, 0.47], DOWN)
    assert up is not None, why  # out of it
    down, why = chain.straight_line(q, [0.6, 0.0, 0.42], DOWN)
    assert down is None and "test/slab" in why  # deeper in


def test_the_base_and_shoulder_in_a_box_refuse_nothing(tmp_path):
    chain = _chain(tmp_path, [("round_the_mount", (0.0, 0.0, 0.10), (0.30, 0.30, 0.30))])
    q = _over_table(chain)
    pts, why = chain.straight_line(q, [0.6, 0.0, 0.30], DOWN)
    assert pts is not None, why


def test_the_clear_margin_step_into_the_chair_is_refused(monkeypatch):
    stuck = [0.0617, -2.0574, -3.1385, -2.5233, -0.2102, -1.2174, 1.7257]
    client = FakeClient()
    client.live = list(stuck)
    monkeypatch.setattr(cli_common, "init_runtime", lambda execute: (object(), client))
    monkeypatch.setattr(cli_common, "_arm_urdf_path", lambda node, frame: flatbot_urdf())
    ctx, _runner = cli_common.build_ctx(Namespace(container=CONTAINER, bench_world=FLATBOT_WORLD, execute=False))
    monkeypatch.setattr(ArmChain, "body_refusal", lambda self, pts, start_exempt=True: "the chair and user: X")
    with pytest.raises(RuntimeError, match="clear-margin refused: the joint line toward the goal enters the chair"):
        list(core.ClearMarginFree(HOME).steps(ctx, core.PlanState(joints=None, chain=0)))


def _spot_ctx(ctx, chain):
    ctx.cpose = ContainerPose(xyz=(0.6, -0.15, 0.22), yaw=0.0)
    return ctx, ContainerPose(xyz=(0.6, 0.05, 0.22), yaw=0.0)


def test_the_lid_spot_passes_over_a_spot_whose_hover_or_set_down_enters_the_chair(ctx, tmp_path):
    chain = _chain(tmp_path, [("far", (-1.0, -1.0, -1.0), (0.1, 0.1, 0.1))])
    ctx, drop = _spot_ctx(ctx, chain)
    assert rlo.drop_reachable(ctx, drop, list(HOME)) is None
    _quat, contact, hover = rlo.setdown_poses(ctx, drop)
    chain = _chain(tmp_path, [("at_hover", tuple(hover), (0.05, 0.05, 0.05))])
    why = rlo.drop_reachable(ctx, drop, list(HOME))
    assert why.startswith("the hand at its carry height is in the way (the chair and user:") and "test/at_hover" in why
    low = (contact[0], contact[1], (contact[2] + hover[2]) / 2.0 - 0.03)
    chain = _chain(tmp_path, [("below", low, (0.03, 0.03, 0.01))])
    why = rlo.drop_reachable(ctx, drop, list(HOME))
    assert why.startswith("the set-down is not a straight line the arm can fly (the chair and user:"), why


def test_a_carry_into_the_chair_is_refused_and_the_next_spot_is_tried(ctx, tmp_path, monkeypatch):
    chain = _chain(tmp_path, [("far", (-1.0, -1.0, -1.0), (0.1, 0.1, 0.1))])
    ctx, drop = _spot_ctx(ctx, chain)
    cfg = press_demo.load_press_demo(CONTAINER)
    q_from = list(HOME)
    _quat, _contact, hover = rlo.setdown_poses(ctx, drop)
    q_hover = chain.nearest_solution(q_from, hover, _quat)

    def plan(ctx_, state, name, target, world, speed):
        return SimpleNamespace(traj=traj(q_from, q_hover), goal_joints=list(q_hover), gains=None), state

    monkeypatch.setattr(rp, "_plan_motion", plan)
    seen = []

    def refusal(self, pts, start_exempt=True):
        seen.append(len(pts))
        return "the chair and user: arm_forearm_link 30 mm into person/lap" if len(seen) == 1 else None

    monkeypatch.setattr(ArmChain, "body_refusal", refusal)
    carry, why = rlo.plan_carry(ctx, cfg, q_from, drop)
    assert carry is None and why == "the carry (the chair and user: arm_forearm_link 30 mm into person/lap)"
    assert seen[0] == 2  # the planner's own carry, checked before its set-down lines are made


def test_the_cli_stops_before_anything_moves_on_an_unreadable_body_model(monkeypatch, capsys):
    monkeypatch.delenv(body.BODY_LAYERS_ENV, raising=False)
    monkeypatch.setattr(cli_common, "init_runtime", lambda execute: (object(), FakeClient()))
    monkeypatch.setattr(cli_common, "_arm_urdf_path", lambda node, frame: flatbot_urdf())
    monkeypatch.setattr(cli_common, "Runner", lambda client: pytest.fail("a runner was made"))
    args = Namespace(container=CONTAINER, bench_world=FLATBOT_WORLD, execute=False, body_layers=None)
    with pytest.raises(SystemExit, match="not starting: body model: cannot read /config/robot/curobo/planner.yaml"):
        cli_common.build_ctx(args)
    monkeypatch.setenv(body.BODY_LAYERS_ENV, "/nowhere/planner.yaml")  # the override, unreadable too
    with pytest.raises(SystemExit, match="Nothing moved"):
        cli_common.build_ctx(Namespace(**{**vars(args), "body_layers": ""}))


def test_the_cli_loads_the_param_and_an_explicit_empty_value_turns_it_off(monkeypatch, tmp_path):
    monkeypatch.delenv(body.BODY_LAYERS_ENV, raising=False)
    monkeypatch.setattr(cli_common, "init_runtime", lambda execute: (object(), FakeClient()))
    monkeypatch.setattr(cli_common, "_arm_urdf_path", lambda node, frame: flatbot_urdf())
    path = _model_files(tmp_path, [("b", (1.0, 0.0, 0.5), (0.1, 0.1, 0.1))])
    cli_common.build_ctx(Namespace(container=CONTAINER, bench_world=FLATBOT_WORLD, execute=False, body_layers=path))
    assert approach._chain().body is not None and approach._chain().body.model.names == ["test/b"]
    cli_common.build_ctx(Namespace(container=CONTAINER, bench_world=FLATBOT_WORLD, execute=False, body_layers=""))
    assert approach._chain().body is None


def test_offline_the_env_var_alone_turns_it_on(monkeypatch, tmp_path):
    monkeypatch.setenv(body.BODY_LAYERS_ENV, _model_files(tmp_path, [("b", (1.0, 0.0, 0.5), (0.1, 0.1, 0.1))]))
    approach.set_body(None)  # never set: a library caller (the press study's harness)
    approach.set_chain_root("arm_mount", flatbot_urdf())
    assert approach._chain().body is not None
    monkeypatch.setenv(body.BODY_LAYERS_ENV, "")
    approach.set_body(None)
    approach.set_chain_root("arm_mount", flatbot_urdf())
    assert approach._chain().body is None
