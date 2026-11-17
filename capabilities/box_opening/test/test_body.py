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
from box_opening.models.container import ContainerModel, ContainerPose
from box_opening.primitives import core
from box_opening.runtime import approach, body
from box_opening.runtime.kinematics import ArmChain, mat_to_quat_xyzw
from box_opening.tasks import cli_common, press_demo
from box_opening.tasks import reach_lid_off as rlo
from box_opening.tasks import reach_press as rp

# The 2026-10-21 HOME (conftest.HOME_20261021): this file's recorded runs and poses were built on it.
pytestmark = pytest.mark.usefixtures("home_20261021")

PKG = Path(__file__).resolve().parents[1]
FLATBOT_WORLD = str(PKG / "config" / "world_flatbot.yaml")
CONTAINER = str(PKG / "config" / "containers" / "ankou_pink.yaml")
DOWN = mat_to_quat_xyzw(np.diag([1.0, -1.0, -1.0]))  # the hand straight down
# flatbot's robot.yaml (rammp-deployments dev 63e7563): the bracelet's and the stock 2F-85's spheres, the
# gripper locked open, as the planner and the chair study have them
STOCK_HAND = """robot_cfg:
  kinematics:
    base_link: arm_mount
    collision_sphere_buffer: 0.005
    collision_spheres:
      arm_forearm_link:
      - {center: [0.0, 0.0, 0.0], radius: 0.06}
      arm_bracelet_link:
      - {center: [0.0, 0.0, -0.045], radius: 0.045}
      - {center: [0.0, -0.05, -0.045], radius: 0.045}
      - {center: [0.045, 0.0, -0.05], radius: 0.036}
      arm_robotiq_85_base_link:
      - {center: [0.0, 0.0, 0.04], radius: 0.04}
      - {center: [0.0, 0.0, 0.055], radius: 0.042}
      - {center: [3.2e-05, 0.04, 0.06], radius: 0.032}
      - {center: [-3.2e-05, -0.04, 0.06], radius: 0.032}
      - {center: [-0.04, 3.2e-05, 0.06], radius: 0.032}
      - {center: [0.04, -3.2e-05, 0.06], radius: 0.032}
      - {center: [0.0, 0.0, 0.078], radius: 0.035}
      - {center: [3.6e-05, 0.045, 0.1], radius: 0.026}
      - {center: [-3.6e-05, -0.045, 0.1], radius: 0.026}
      - {center: [-0.045, 3.6e-05, 0.1], radius: 0.026}
      - {center: [0.045, -3.6e-05, 0.1], radius: 0.026}
      - {center: [3.6e-05, 0.045, 0.13], radius: 0.022}
      - {center: [-3.6e-05, -0.045, 0.13], radius: 0.022}
      - {center: [-0.045, 3.6e-05, 0.13], radius: 0.022}
      - {center: [0.045, -3.6e-05, 0.13], radius: 0.022}
      - {center: [3.6e-05, 0.045, 0.145], radius: 0.018}
      - {center: [-3.6e-05, -0.045, 0.145], radius: 0.018}
      - {center: [-0.045, 3.6e-05, 0.145], radius: 0.018}
      - {center: [0.045, -3.6e-05, 0.145], radius: 0.018}
      arm_robotiq_85_left_finger_link:
      - {center: [-0.010026, -4.3e-05, 0.019663], radius: 0.03}
      arm_robotiq_85_left_finger_tip_link:
      - {center: [-0.021578, -3.8e-05, 0.031998], radius: 0.02}
      arm_robotiq_85_right_finger_link:
      - {center: [0.010026, 4.1e-05, 0.019663], radius: 0.03}
      arm_robotiq_85_right_finger_tip_link:
      - {center: [0.021578, 3.7e-05, 0.031998], radius: 0.02}
"""
# the chair study's probe (sweep_after/selfcheck.py, results/coverage.json), in arm_end_effector_link: the 131 mm
# fingertips and the tool point, and the corners of presssim's wrist OAK box (90 x 30 x 30 mm at [0, 0.065, 0])
PROBE_TIPS = {"tip_left": (0.003, 0.0, 0.131), "tip_right": (-0.003, 0.0, 0.131), "tool_131": (0.0, 0.0, 0.131)}
PROBE_CAMERA = {
    "corner_%+d%+d%+d" % (sx, sy, sz): (0.045 * sx, 0.065 + 0.015 * sy, 0.015 * sz)
    for sx in (-1, 1)
    for sy in (-1, 1)
    for sz in (-1, 1)
}
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


def _model_files(tmp, boxes, robot=ROBOT, padding=None, layer="test"):
    """A planner config in `tmp` with one layer of `boxes` [(name, centre, dims)]: its path."""
    lines = ["layer: %s" % layer, "obstacles:"]
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
    path = _model_files(tmp_path, [("b", (1.0, 0.0, 0.5), (0.1, 0.1, 0.1))], layer="person")
    ctx, _r = cli_common.build_ctx(
        Namespace(container=CONTAINER, bench_world=FLATBOT_WORLD, execute=False, body_layers=path)
    )
    assert approach._chain().body is not None and approach._chain().body.model.names == ["person/b"]
    assert isinstance(ctx.model, ContainerModel)  # the container's, never the body model
    ctx, _r = cli_common.build_ctx(
        Namespace(container=CONTAINER, bench_world=FLATBOT_WORLD, execute=False, body_layers="")
    )
    assert approach._chain().body is None
    assert isinstance(ctx.model, ContainerModel)


def test_offline_the_env_var_alone_turns_it_on(monkeypatch, tmp_path):
    monkeypatch.setenv(body.BODY_LAYERS_ENV, _model_files(tmp_path, [("b", (1.0, 0.0, 0.5), (0.1, 0.1, 0.1))]))
    approach.set_body(None)  # never set: a library caller (the press study's harness)
    approach.set_chain_root("arm_mount", flatbot_urdf())
    assert approach._chain().body is not None
    monkeypatch.setenv(body.BODY_LAYERS_ENV, "")
    approach.set_body(None)
    approach.set_chain_root("arm_mount", flatbot_urdf())
    assert approach._chain().body is None


def _outside_mm(chain, check, points, qs):
    """{name: worst over `qs` of how far the hand point (arm_end_effector_link) lies outside every checked
    sphere, buffered, mm} — the chair study's coverage probe."""
    C, r, _links = check._centres(np.asarray(qs, float))
    out = {}
    for name, p in points.items():
        worst = -np.inf
        for i, q in enumerate(qs):
            R, t = chain.fk(q)
            ee = t - R @ np.array([0.0, 0.0, 0.120])  # tool_frame is 0.120 out of the flange
            P = ee + R @ np.asarray(p, float)
            worst = max(worst, float((np.linalg.norm(C[i] - P, axis=1) - r).min()))
        out[name] = 1000.0 * worst
    return out


def _probe_poses(chain):
    rng = np.random.default_rng(0)
    return [list(np.asarray(HOME) + rng.uniform(-1.0, 1.0, 7)) for _ in range(20)]


def test_the_hand_spheres_cover_the_131mm_fingertips_and_the_wrist_camera_the_robot_file_misses(tmp_path):
    path = _model_files(tmp_path, [("far", (-1.0, -1.0, -1.0), (0.1, 0.1, 0.1))], robot=STOCK_HAND)
    chain = ArmChain(root="arm_mount", path=flatbot_urdf())
    qs = _probe_poses(chain)
    stock = body.load(path, robot_body=False).check_for(chain, flatbot_urdf(), "arm_mount")
    assert stock.hand_added == [] and "arm_end_effector_link" not in stock.spheres
    tips, cam = _outside_mm(chain, stock, PROBE_TIPS, qs), _outside_mm(chain, stock, PROBE_CAMERA, qs)
    # the study's numbers (coverage.json): the tips 13.0-13.1 mm, the camera's outer corners 12.6 mm outside
    assert tips["tip_left"] == pytest.approx(13.1, abs=0.1) and tips["tool_131"] == pytest.approx(13.0, abs=0.1)
    assert max(cam.values()) == pytest.approx(12.6, abs=0.1)
    m = body.load(path)  # the shipped config/robot_body.yaml
    assert m.hand_source.endswith("config/robot_body.yaml") and [g.name for g in m.hand] == [
        "fingers_131mm",
        "wrist_camera",
    ]
    check = m.check_for(chain, flatbot_urdf(), "arm_mount")
    assert check.hand_added == ["fingers_131mm", "wrist_camera"] and check.hand_skipped == []
    assert (
        len(check.spheres["arm_robotiq_85_base_link"][1]) == 19 + 4
        and len(check.spheres["arm_end_effector_link"][1]) == 12
    )
    tips, cam = _outside_mm(chain, check, PROBE_TIPS, qs), _outside_mm(chain, check, PROBE_CAMERA, qs)
    # now inside: the tips 11.5 mm and the tool point 10 mm deep (5 of them the buffer), every camera corner >= 12 mm
    assert max(tips.values()) <= -9.9 and max(cam.values()) <= -12.0, (tips, cam)
    # and the bare spheres alone hold each group's boxes (the load checks it): none of them is the buffer's
    for g in m.hand:
        for _name, c, h in g.covers:
            assert body.outside_m(body.box_points(c, h), g.centres, g.radii).max() <= body.HAND_COVER_TOL_M


def test_a_hand_group_whose_spheres_miss_what_it_stands_for_stops_the_load(tmp_path):
    path = _model_files(tmp_path, [("b", (1, 0, 0), (0.1, 0.1, 0.1))])
    bad = tmp_path / "robot_body.yaml"
    bad.write_text(
        "hand_spheres:\n  tips:\n    link: arm_robotiq_85_base_link\n"
        "    covers: [{name: pads, centre: [0, 0, 0.12], half: [0.01, 0.01, 0.011]}]\n"
        "    spheres: [{center: [0, 0, 0.10], radius: 0.02}]\n"
    )
    with pytest.raises(body.BodyModelError, match="hand group tips leaves pads [0-9.]+ mm outside its spheres"):
        body.load(path, robot_body=bad)
    bad.write_text("hand_spheres:\n  tips: {link: arm_robotiq_85_base_link}\n")
    with pytest.raises(body.BodyModelError, match="hand group tips is malformed"):
        body.load(path, robot_body=bad)
    with pytest.raises(body.BodyModelError, match="cannot read"):
        body.load(path, robot_body=tmp_path / "nowhere.yaml")


def test_a_robot_file_that_carries_the_hand_spheres_is_not_doubled(tmp_path):
    import yaml

    hand = body.load_hand(body.robot_body_path())
    lines = []
    for g in hand:  # on flatbot's URDF the flange and the gripper base are one frame: all on the gripper base
        lines += [
            "      - {center: %s, radius: %s}" % (list(map(float, c)), float(r))
            for c, r in zip(g.centres, g.radii, strict=True)
        ]
    robot = STOCK_HAND.replace(
        "      arm_robotiq_85_left_finger_link:", "\n".join(lines) + "\n      arm_robotiq_85_left_finger_link:"
    )
    assert len(yaml.safe_load(robot)["robot_cfg"]["kinematics"]["collision_spheres"]["arm_robotiq_85_base_link"]) == 35
    path = _model_files(tmp_path, [("far", (-1.0, -1.0, -1.0), (0.1, 0.1, 0.1))], robot=robot)
    check = body.load(path).check_for(ArmChain(root="arm_mount", path=flatbot_urdf()), flatbot_urdf(), "arm_mount")
    assert check.hand_added == [] and check.hand_skipped == ["fingers_131mm", "wrist_camera"]
    assert len(check.spheres["arm_robotiq_85_base_link"][1]) == 35 and "arm_end_effector_link" not in check.spheres


def test_a_robot_file_whose_hand_spheres_fall_short_by_less_than_the_buffer_is_not_taken_for_them(tmp_path):
    # the same spheres 3 mm smaller: buffered they would still reach every box the groups stand for, but the
    # boxes would keep none of the buffer the groups add — judged bare, as the groups' own, both are added
    hand = body.load_hand(body.robot_body_path())
    lines = []
    for g in hand:
        lines += [
            "      - {center: %s, radius: %s}" % (list(map(float, c)), float(r) - 0.003)
            for c, r in zip(g.centres, g.radii, strict=True)
        ]
    robot = STOCK_HAND.replace(
        "      arm_robotiq_85_left_finger_link:", "\n".join(lines) + "\n      arm_robotiq_85_left_finger_link:"
    )
    path = _model_files(tmp_path, [("far", (-1.0, -1.0, -1.0), (0.1, 0.1, 0.1))], robot=robot)
    check = body.load(path).check_for(ArmChain(root="arm_mount", path=flatbot_urdf()), flatbot_urdf(), "arm_mount")
    assert check.hand_added == ["fingers_131mm", "wrist_camera"] and check.hand_skipped == []


def test_distance_is_the_signed_gap_to_the_unpadded_boxes_per_state_and_vectorised(tmp_path):
    chain = _chain(
        tmp_path, [("lap", (0.6, 0.0, 0.20), (0.20, 0.20, 0.10)), ("far", (-1.0, -1.0, -1.0), (0.1, 0.1, 0.1))]
    )
    q = _over_table(chain)  # the tool 0.45 over (0.6, 0): the box's top at 0.25
    body_ = chain.body
    one = body_.distance(q, layers=("test",))
    assert isinstance(one.m, float) and one.box == "test/lap" and one.link.startswith("arm_robotiq_85_")
    C, r, _links = body_.spheres_at(q)
    lowest = float((C[:, 2] - r).min())  # the box lies under the whole hand: the gap is the lowest sphere's
    assert one.m == pytest.approx(lowest - 0.25, abs=1e-9)
    down, _ = chain.straight_line(q, [0.6, 0.0, 0.40], DOWN)
    many = body_.distance(down, layers="test")
    assert many.m.shape == (len(down),) and many.m[-1] == pytest.approx(one.m - 0.05, abs=0.002)
    assert [body_.distance(p, layers="test").m for p in down] == pytest.approx(list(many.m))
    per, links = body_.sphere_distances(down, layers="test")
    assert per.shape == (len(down), len(links)) and per.min(axis=1) == pytest.approx(many.m)
    assert body_.distance(q).m == pytest.approx(one.m)  # all layers: the far box is further
    with pytest.raises(ValueError, match="no layer person"):
        body_.distance(q, layers=("person",))


def test_pose_clear_is_strict_where_a_line_from_inside_may_still_leave(tmp_path):
    chain = _chain(tmp_path, [("slab", (0.6, 0.0, 0.39), (0.30, 0.30, 0.06))])  # padded top 0.44
    q = _over_table(chain)
    assert chain.body_refusal([q]) is None  # a line from where the arm stands: allowed to start inside
    assert not chain.body.pose_clear(q) and "test/slab" in chain.body.pose_refusal(q)  # a goal: refused
    up, why = chain.straight_line(q, [0.6, 0.0, 0.48], DOWN)
    assert up is not None, why
    assert chain.body.pose_clear(up[-1]) and chain.body.pose_refusal(up[-1]) is None


def test_a_run_adds_its_own_boxes_and_lines_into_them_are_refused_naming_them(tmp_path):
    chain = _chain(tmp_path, [("far", (-1.0, -1.0, -1.0), (0.1, 0.1, 0.1))])
    q = _over_table(chain)
    base = chain.body
    # a counter edge turned 30 deg about z, its top 0.30: what the camera's table map would give
    c, s_ = np.cos(np.radians(30)), np.sin(np.radians(30))
    R = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    edge = {"name": "cabinet/counter", "centre": (0.6, 0.0, 0.25), "half": (0.10, 0.30, 0.05), "rotation": R}
    assert approach.set_obstacles([edge]) is chain.body and chain.body is not base and chain.body.base is base
    assert chain.body.layers == ["cabinet", "test"] and base.layers == ["test"]
    pts, why = chain.straight_line(q, [0.6, 0.0, 0.33], DOWN)
    assert (
        pts is None and why.startswith("the cabinet: arm_robotiq_85_") and "into cabinet/counter (padded 20 mm" in why
    )
    assert base.refusal(chain.straight_line(q, [0.6, 0.0, 0.47], DOWN)[0]) is None
    C, r, _links = chain.body.spheres_at(q)
    assert chain.body.distance(q, layers="cabinet").m == pytest.approx(float((C[:, 2] - r).min()) - 0.30, abs=1e-9)
    quat = mat_to_quat_xyzw(R)  # a quaternion does as well as the matrix
    same = base.with_obstacles([("cabinet/counter", (0.6, 0.0, 0.25), (0.10, 0.30, 0.05), quat)])
    assert same.distance(q, layers="cabinet").m == pytest.approx(chain.body.distance(q, layers="cabinet").m)
    loose = base.with_obstacles([("counter", (0.6, 0.0, 0.25), (0.10, 0.30, 0.05))], pad_m=0.0)
    assert loose.layers == ["obstacle", "test"] and loose._pad[-1] == 0.0
    with pytest.raises(ValueError, match="not a rotation"):
        base.with_obstacles([("x", (0, 0, 0), (0.1, 0.1, 0.1), np.diag([1.0, 2.0, 1.0]))])
    assert approach.set_obstacles([]) is not None and chain.body.layers == ["test"]  # cleared
    pts, why = chain.straight_line(q, [0.6, 0.0, 0.33], DOWN)
    assert pts is not None, why


def test_near_the_user_no_sphere_moves_faster_than_the_profile_allows(tmp_path):
    from box_opening.constants import JOINT_VMAX
    from box_opening.runtime.retime import RetimeParams, positions_to_traj, retime_group

    # a lap whose top is 0.24: a 0.3 m sweep of the hand 0.33 over the table brings the fingers ~55 mm over it
    path = _model_files(tmp_path, [("lap", (0.55, 0.0, 0.19), (0.30, 0.60, 0.10))], layer="person")
    approach.set_body(body.load(path))
    approach.set_chain_root("arm_mount", flatbot_urdf())
    chain = approach._chain()
    nu = chain.body.model.near_user
    assert nu.layers == ("person",) and (nu.far_m, nu.near_m, nu.far_mps, nu.near_fraction) == (0.15, 0.05, 0.5, 0.25)
    q = chain.nearest_solution(list(HOME), [0.55, -0.15, 0.33], DOWN)
    assert q is not None
    pts, why = chain.straight_line(q, [0.55, 0.15, 0.33], DOWN)
    assert pts is not None, why
    assert chain.body.distance(pts[len(pts) // 2], layers="person").m == pytest.approx(0.055, abs=0.01)
    names = ["joint_%d" % (i + 1) for i in range(7)]
    free, info0 = retime_group([positions_to_traj(names, pts)], [1.0], JOINT_VMAX, RetimeParams())
    slow, info = retime_group(
        [positions_to_traj(names, pts)], [1.0], JOINT_VMAX, RetimeParams(), speed_cap=chain.body.near_user_caps
    )
    assert info0["capped_samples"] == 0 and info["capped_samples"] > 0
    assert info["duration_s"] > 1.5 * info0["duration_s"]
    assert [list(p.positions) for p in slow.points] == [list(p.positions) for p in free.points]  # timing only

    def worst_over(traj):
        """The worst ratio of a sphere's speed to what the profile allows it, between samples."""
        Q = np.array([p.positions for p in traj.points])
        t = np.array([p.time_from_start.sec + 1e-9 * p.time_from_start.nanosec for p in traj.points])
        d, _links = chain.body.sphere_distances(Q, ("person",))
        C, _r, _l = chain.body._centres(Q)
        speed = np.linalg.norm(np.diff(C, axis=0), axis=2) / np.diff(t)[:, None]
        allowed = nu.speed_mps(np.minimum(d[:-1], d[1:]))
        return float(np.max(speed / allowed))

    assert worst_over(free) > 1.5  # at full speed the fingers sweep over the lap far faster
    assert worst_over(slow) <= 1.05  # held to it (5 %: the profile's samples against its intervals)
    high, why = chain.straight_line(_over_table(chain), [0.6, 0.0, 0.47], DOWN)  # the fingers 0.17 over the lap
    assert high is not None, why
    assert np.isinf(chain.body.near_user_caps(high)).all()


def test_the_runner_slows_near_the_user_only_with_a_body_model(tmp_path):
    from box_opening.runtime.runner import Runner

    runner = Runner(FakeClient(), log_dir=tmp_path / "runs")
    approach.set_body(False)
    assert runner.near_user_cap() is None  # the bench: no body model
    path = _model_files(tmp_path, [("lap", (0.6, 0.0, 0.10), (0.30, 0.30, 0.10))], layer="person")
    approach.set_body(body.load(path))
    approach.set_chain_root("arm_mount", flatbot_urdf())
    cap = runner.near_user_cap()
    assert cap is not None and cap.__self__ is approach._chain().body
    runner.slow_near_user = False
    assert runner.near_user_cap() is None


def test_a_body_model_without_the_users_layer_stops_the_run(monkeypatch, tmp_path):
    monkeypatch.delenv(body.BODY_LAYERS_ENV, raising=False)
    monkeypatch.setattr(cli_common, "init_runtime", lambda execute: (object(), FakeClient()))
    monkeypatch.setattr(cli_common, "_arm_urdf_path", lambda node, frame: flatbot_urdf())
    path = _model_files(tmp_path, [("b", (1.0, 0.0, 0.5), (0.1, 0.1, 0.1))])  # layer "test": no person
    args = Namespace(container=CONTAINER, bench_world=FLATBOT_WORLD, execute=False, body_layers=path)
    with pytest.raises(SystemExit, match="has no layer person, which .*robot_body.yaml's near_user slows"):
        cli_common.build_ctx(args)


def _goal_chain(tmp_path):
    """flatbot's chain with a person box under the hand 0.45 over (0.6, 0): the hand at 0.33 is inside it."""
    path = _model_files(tmp_path, [("lap", (0.6, 0.0, 0.20), (0.30, 0.30, 0.20))], layer="person")
    approach.set_body(body.load(path))
    approach.set_chain_root("arm_mount", flatbot_urdf())
    chain = approach._chain()
    return chain, _over_table(chain), chain.nearest_solution(list(HOME), [0.6, 0.0, 0.33], DOWN)


def test_a_goal_inside_the_body_model_is_refused_before_it_is_sent_but_never_the_rest(tmp_path):
    chain, clear, inside = _goal_chain(tmp_path)
    assert inside is not None and approach.goal_refusal(("joints", clear)) is None
    why = approach.goal_refusal(("joints", inside))
    assert why.startswith("the chair and user: arm_") and "into person/lap" in why
    # a pose goal is judged at the joints the client sends it as (its own IK); a plan's end at the end
    assert "person/lap" in approach.goal_refusal(("pose", [0.6, 0.0, 0.33], DOWN), list(clear))
    assert approach.goal_refusal(("pose", [0.6, 0.0, 0.45], DOWN), list(clear)) is None
    assert "person/lap" in approach.goal_refusal(("pose", [0.6, 0.0, 0.45], DOWN), list(clear), inside)
    # the rest pose: flatbot's HOME is inside the chair model's armrest today, and is never refused
    approach.set_rest_goals([inside])
    assert approach.goal_refusal(("joints", inside)) is None
    approach.set_rest_goals(None)
    approach.set_body(False)  # off: nothing refused
    assert approach.goal_refusal(("joints", inside)) is None


def test_plan_motion_refuses_a_goal_inside_unsent_and_a_plan_that_ends_inside(ctx, tmp_path):
    from box_opening.primitives.core import PlanState, _plan_motion

    _chain_, clear, inside = _goal_chain(tmp_path)
    world = ("full", "/w/full.yaml")
    with pytest.raises(RuntimeError, match="look refused before it was sent to the planner: its goal puts the arm"):
        _plan_motion(ctx, PlanState(joints=list(clear), chain=0), "look", ("joints", inside), world, 0.5)
    assert ctx.client.joint_targets == []  # never sent
    leg, _st = _plan_motion(ctx, PlanState(joints=list(clear), chain=0), "look", ("joints", clear), world, 0.5)
    assert leg.goal_joints == pytest.approx(list(clear))
    ctx.client.plans.append(
        SimpleNamespace(success=True, message="", trajectory=traj(clear, inside), planning_time=0.1)
    )
    with pytest.raises(RuntimeError, match="carry refused before it was flown: its goal puts the arm inside the chair"):
        _plan_motion(ctx, PlanState(joints=list(clear), chain=0), "carry", ("pose", [0.6, 0.0, 0.45], DOWN), world, 0.5)
    approach.set_rest_goals([inside])  # a home to the rest: planned
    try:
        leg, _st = _plan_motion(ctx, PlanState(joints=list(clear), chain=0), "home", ("joints", inside), world, 0.4)
        assert leg.goal_joints == pytest.approx(list(inside))
    finally:
        approach.set_rest_goals(None)


def test_a_lazy_leg_replanned_into_the_body_model_is_refused(tmp_path):
    from box_opening.runtime.runner import Runner

    _chain_, clear, inside = _goal_chain(tmp_path)
    client = FakeClient()
    client.live = list(clear)
    runner = Runner(client, log_dir=tmp_path / "runs")
    leg = SimpleNamespace(
        name="retreat", plan_free=False, world="full", world_path="/w/full.yaml", target=("joints", list(inside))
    )
    assert runner._replan_group([leg], 3) == (None, 3) and client.joint_targets == []


def test_with_the_body_model_on_only_a_planner_on_the_same_chair_passes():
    from box_opening.tasks import preflight

    src = "/config/robot/curobo/planner.yaml"
    assert preflight.planner_body_verdict("", src, src)[0] == "PASS"
    assert preflight.planner_body_verdict(None, "/config/robot/curobo/../curobo/planner.yaml", src)[0] == "PASS"
    level, text = preflight.planner_body_verdict("world_empty.yaml", src, src)
    assert level == "FAIL" and "plans in world world_empty.yaml" in text and 'with world ""' in text
    level, text = preflight.planner_body_verdict("", "/config/robot/curobo/planner_empty_world.yaml", src)
    assert level == "FAIL" and "config /config/robot/curobo/planner_empty_world.yaml" in text
    assert "did not say" in preflight.planner_body_verdict(None, None, src)[1]
    # through the planner-world line: the body on decides; off, as before (world_empty.yaml passes)
    assert preflight.planner_world_verdict("world_empty.yaml", "gen3_real.yaml", body_source=src)[0] == "FAIL"
    assert preflight.planner_world_verdict("world_empty.yaml", "gen3_real.yaml")[0] == "PASS"


def test_a_run_that_moves_refuses_an_empty_world_planner_with_the_body_model_on(monkeypatch, tmp_path):
    from box_opening.tasks import preflight

    monkeypatch.delenv(body.BODY_LAYERS_ENV, raising=False)
    monkeypatch.setattr(cli_common, "init_runtime", lambda execute: (object(), FakeClient()))
    monkeypatch.setattr(cli_common, "_arm_urdf_path", lambda node, frame: flatbot_urdf())
    path = _model_files(tmp_path, [("b", (1.0, 0.0, 0.5), (0.1, 0.1, 0.1))], layer="person")
    said = {"world": ("world_empty.yaml", "/config/robot/curobo/planner_empty_world.yaml")}
    monkeypatch.setattr(preflight, "planner_world", lambda node, with_config=False: said["world"])
    args = Namespace(container=CONTAINER, bench_world=FLATBOT_WORLD, execute=True, body_layers=path)
    with pytest.raises(SystemExit, match="not starting: the body model is on .* but the planner plans in world"):
        cli_common.build_ctx(args)
    cli_common.build_ctx(Namespace(**{**vars(args), "execute": False}))  # nothing moves: said, not refused
    said["world"] = ("", path)
    ctx_, _runner = cli_common.build_ctx(args)  # the same chair: on
    assert ctx_ is not None
    said["world"] = ("world_empty.yaml", None)
    cli_common.build_ctx(Namespace(**{**vars(args), "body_layers": ""}))  # the check off: no refusal
