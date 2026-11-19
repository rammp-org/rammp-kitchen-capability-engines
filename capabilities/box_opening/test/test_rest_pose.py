"""The arm's rest pose is the robot's own (runtime/rest): read from the arm driver's presets file, judged
before anything moves, and followed by everything that reads HOME; with none configured, HOME is as it was."""

import ast
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from conftest import CFG, HOME_20261021, FakeClient, flatbot_urdf

from box_opening import constants
from box_opening.primitives.look import look_joints
from box_opening.runtime import approach, body, branches, rest
from box_opening.tasks import cli_common, home_arm, preflight, press_demo
from box_opening.tasks import reach_probe as rpb

PKG = Path(__file__).resolve().parents[1]
FLATBOT_WORLD = str(PKG / "config" / "world_flatbot.yaml")
CONTAINER = str(PKG / "config" / "containers" / "ankou_pink.yaml")
# a pose of flatbot's arm that is no robot's HOME: the 2026-10-21 one with every joint moved a little
POSE = [0.167, -1.797, -3.037, -2.169, -0.111, -0.583, 1.825]
ROBOT = """robot_cfg:
  kinematics:
    base_link: arm_mount
    collision_sphere_buffer: 0.005
    collision_spheres:
      arm_forearm_link:
      - {center: [0.0, -0.1, 0.0], radius: 0.06}
      arm_robotiq_85_base_link:
      - {center: [0.0, 0.0, 0.05], radius: 0.04}
"""


@pytest.fixture(autouse=True)
def as_it_was(monkeypatch, tmp_path):
    """HOME, the chain and the body model put back after each test, and the environment naming no pose."""
    monkeypatch.setenv("BOX_OPENING_STATE", str(tmp_path))
    monkeypatch.delenv(rest.REST_POSE_ENV, raising=False)
    monkeypatch.delenv(body.BODY_LAYERS_ENV, raising=False)
    # the default file is the robot's, and no test reads this machine's: one that is not there
    monkeypatch.setattr(rest, "DEFAULT_REST_POSE", str(tmp_path / "config" / "robot" / "arm_presets.yaml"))
    home = list(constants.HOME)
    saved = (approach._ROOT, approach._URDF_PATH, approach._CHAIN, approach._GRAVITY, approach._BODY)
    goals = approach._REST_GOALS
    yield
    approach._ROOT, approach._URDF_PATH, approach._CHAIN, approach._GRAVITY, approach._BODY = saved
    approach._REST_GOALS = goals
    constants.HOME[:] = home


def _presets(tmp, home=POSE, name="arm_presets.yaml", node="/**", nested=False):
    """A presets file as the arm's driver is handed it (rammp-deployments config/robots/<robot>/): its path."""
    q = "[%s]" % ", ".join(repr(v) for v in home) if isinstance(home, list | tuple) else str(home)
    lines = ["%s:" % node, "  ros__parameters:"]
    lines += ["    presets:", "      home: %s" % q] if nested else ["    presets.home: %s" % q]
    (tmp / name).write_text("\n".join(lines) + "\n")
    return str(tmp / name)


def _planner(tmp, boxes, layer="seat"):
    """A planner config in `tmp` with one layer of `boxes` [(name, centre, dims)]: its path."""
    lines = ["layer: %s" % layer, "obstacles:"]
    lines += ["  - {name: %s, position: %s, dims: %s}" % (n, [float(v) for v in c], list(d)) for n, c, d in boxes]
    (tmp / "layer_seat.yaml").write_text("\n".join(lines) + "\n")
    (tmp / "robot.yaml").write_text(ROBOT)
    (tmp / "planner.yaml").write_text("robot: robot.yaml\nlayers: [layer_seat.yaml]\n")
    return str(tmp / "planner.yaml")


def _flatbot(tmp, boxes=None):
    """flatbot's chain, with a body model of `boxes` attached where given (the planner's spheres alone)."""
    approach.set_body(False if boxes is None else body.load(_planner(tmp, boxes), robot_body=False))
    approach.set_chain_root("arm_mount", flatbot_urdf())
    return approach._chain()


def _forearm(tmp, q):
    """Where the test robot file's forearm sphere stands at `q` (arm_mount)."""
    chain = _flatbot(tmp, [("far", (3.0, 0.0, 0.5), (0.1, 0.1, 0.1))])
    C, _r, links = chain.body.spheres_at(q)
    return np.array(C[list(links).index("arm_forearm_link")], float)


# -- where the pose comes from -------------------------------------------------------------
def test_the_override_beats_the_param_and_empty_is_the_built_in_pose():
    assert rest.resolve(None, env={}) == rest.DEFAULT_REST_POSE
    assert rest.resolve("/x/arm_presets.yaml", env={}) == "/x/arm_presets.yaml"
    assert rest.resolve("/x/arm_presets.yaml", env={rest.REST_POSE_ENV: "/y/p.yaml"}) == "/y/p.yaml"
    assert rest.resolve("/x/arm_presets.yaml", env={rest.REST_POSE_ENV: ""}) == ""
    assert rest.resolve("", env={}) == ""


def test_the_default_file_is_the_one_the_arms_driver_is_handed(monkeypatch):
    monkeypatch.undo()  # this test reads the module as it ships
    assert rest.DEFAULT_REST_POSE == "/config/robot/arm_presets.yaml" and rest.REST_POSE_KEY == "presets.home"
    assert rest.REST_POSE_ENV == "BOX_OPENING_REST_POSE"


def test_the_pose_is_the_drivers_presets_home_however_the_params_file_writes_it(tmp_path):
    assert rest.read(_presets(tmp_path)) == POSE  # rammp-deployments' own: /** and the dotted name
    assert rest.read(_presets(tmp_path, nested=True)) == POSE
    assert rest.read(_presets(tmp_path, node="kinova_gen3_node")) == POSE
    assert rest.read(_presets(tmp_path, home=[0, 1, 2, -2, 0, 1, 2])) == [0.0, 1.0, 2.0, -2.0, 0.0, 1.0, 2.0]
    # the team's own files read as written (flatbot's is the built-in pose)
    (tmp_path / "flatbot.yaml").write_text(
        "# flatbot's named arm poses\n/**:\n  ros__parameters:\n"
        "    presets.home: [0.4425, -1.5718, -2.9033, -2.1048, -0.7412, -0.9194, 2.1631]\n"
    )
    assert rest.read(tmp_path / "flatbot.yaml") == list(constants.HOME_BUILT_IN)


def test_a_file_that_is_not_there_or_names_no_home_gives_none_unless_it_was_named(tmp_path):
    assert rest.read(tmp_path / "nowhere.yaml") is None
    (tmp_path / "other.yaml").write_text(
        "/**:\n  ros__parameters:\n    presets.rest: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\n"
    )
    assert rest.read(tmp_path / "other.yaml") is None
    (tmp_path / "empty.yaml").write_text("")
    assert rest.read(tmp_path / "empty.yaml") is None
    with pytest.raises(rest.RestPoseError, match="cannot read .*nowhere.yaml"):
        rest.read(tmp_path / "nowhere.yaml", named=True)
    with pytest.raises(rest.RestPoseError, match="other.yaml names no presets.home"):
        rest.read(tmp_path / "other.yaml", named=True)


@pytest.mark.parametrize(
    "home",
    [
        POSE[:6],
        POSE + [0.0],
        "[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, .nan]",
        "[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, .inf]",
        "[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, true]",
        "[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 'x']",
        "0.5",
    ],
)
def test_a_home_that_is_not_seven_finite_angles_is_an_error_never_the_built_in_pose(tmp_path, home):
    with pytest.raises(rest.RestPoseError, match="is not seven joint angles"):
        rest.read(_presets(tmp_path, home=home))


def test_a_file_that_cannot_be_parsed_or_whose_nodes_disagree_is_an_error(tmp_path):
    (tmp_path / "bad.yaml").write_text("/**:\n  ros__parameters: [\n")
    with pytest.raises(rest.RestPoseError, match="cannot read"):
        rest.read(tmp_path / "bad.yaml")
    other = [v + 0.1 for v in POSE]
    (tmp_path / "two.yaml").write_text(
        "/**:\n  ros__parameters:\n    presets.home: %r\nkinova_gen3_node:\n  ros__parameters:\n    presets.home: %r\n"
        % (POSE, other)
    )
    with pytest.raises(rest.RestPoseError, match="more than one presets.home .*kinova_gen3_node.*neither is taken"):
        rest.read(tmp_path / "two.yaml")
    (tmp_path / "same.yaml").write_text(
        "/**:\n  ros__parameters:\n    presets.home: %r\nkinova_gen3_node:\n  ros__parameters:\n    presets.home: %r\n"
        % (POSE, POSE)
    )
    assert rest.read(tmp_path / "same.yaml") == POSE


# -- a configured pose is judged before anything moves ----------------------------------------
def test_a_pose_inside_the_chairs_padding_is_refused_in_plain_words(tmp_path):
    centre = _forearm(tmp_path, POSE)
    chain = _flatbot(tmp_path, [("armrest", centre, (0.1, 0.1, 0.1))])
    why = rest.refusal(POSE, chain, chain.body)
    assert "the arm would rest inside the chair and user's padding there: arm_forearm_link" in why
    assert "mm into seat/armrest (padded 20 mm, the arm's spheres 5 mm)" in why
    assert "the planner plans nothing from a start in it" in why
    with pytest.raises(rest.RestPoseError, match="is refused: the arm would rest inside the chair"):
        rest.configure(_presets(tmp_path))
    assert constants.HOME == list(constants.HOME_BUILT_IN)  # never put in force
    # the same pose with the box out of the way is taken, and says how it stands
    chain = _flatbot(tmp_path, [("armrest", centre + np.array([0.0, 0.0, 0.40]), (0.1, 0.1, 0.1))])
    assert rest.refusal(POSE, chain, chain.body) is None
    q, words = rest.configure(_presets(tmp_path))
    assert q == POSE == constants.HOME
    assert "presets.home of %s" % (tmp_path / "arm_presets.yaml") in words
    assert "mm outside the chair and user's padding at its nearest (arm_" in words


def test_a_pose_just_outside_the_padding_is_taken_and_one_just_inside_is_not(tmp_path):
    centre = _forearm(tmp_path, POSE)
    # the forearm's sphere (60 + 5 mm) over a box whose padded top (20 mm) stands 2 mm under it, then 2 mm in it
    for gap, refused in ((0.002, False), (-0.002, True)):
        top = centre[2] - 0.065 - 0.020 - gap
        chain = _flatbot(tmp_path, [("pad", (centre[0], centre[1], top - 0.05), (0.3, 0.3, 0.1))])
        why = rest.refusal(POSE, chain, chain.body)
        assert (why is not None) == refused, why


def test_a_pose_with_a_joint_at_or_past_its_limit_is_refused(tmp_path):
    chain = _flatbot(tmp_path)
    assert rest.refusal(POSE, chain) is None
    assert rest.limit_room_rad() == pytest.approx(0.18)  # what the plan-free step toward a goal asks of it
    near = list(POSE)
    near[3] = -2.57 + 0.17
    assert "arm_joint_4 at -2.400 rad stands 0.170 rad from its limit" in rest.refusal(near, chain)
    near[3] = -2.57 + 0.19
    assert rest.refusal(near, chain) is None
    past = list(POSE)
    past[1] = 2.30
    assert "arm_joint_2 at 2.300 rad is past its limit (-2.24 to 2.24)" in rest.refusal(past, chain)
    wound = list(POSE)
    wound[2] = 6.5
    assert "arm_joint_3 at 6.500 rad is past the planner's bound" in rest.refusal(wound, chain)
    assert "it has 6 joint angles" in rest.refusal(POSE[:6], chain)
    _flatbot(tmp_path)
    with pytest.raises(rest.RestPoseError, match="is refused: arm_joint_4 at -2.400 rad"):
        rest.configure(_presets(tmp_path, home=[POSE[0], POSE[1], POSE[2], -2.40, POSE[4], POSE[5], POSE[6]]))
    assert constants.HOME == list(constants.HOME_BUILT_IN)


def test_a_configured_pose_that_cannot_be_judged_is_refused(tmp_path):
    approach.set_body(False)
    approach.set_chain_root("arm_mount", None)  # no description heard: no limits to judge it by
    with pytest.raises(rest.RestPoseError, match="cannot be judged: no arm URDF rooted at arm_mount"):
        rest.configure(_presets(tmp_path))
    approach.set_chain_root("no_such_link", flatbot_urdf())  # a description with no chain from that root
    with pytest.raises(rest.RestPoseError, match="cannot be judged: URDF .* has no chain"):
        rest.configure(_presets(tmp_path))
    assert constants.HOME == list(constants.HOME_BUILT_IN)


# -- with none configured, HOME is exactly as it was -----------------------------------------
def test_with_no_pose_configured_home_is_not_touched(tmp_path, home_20261021):
    _flatbot(tmp_path)
    for param, why in (
        ("", "empty"),
        (None, "(no %s)" % rest.DEFAULT_REST_POSE),
    ):  # asked for; the default file not there
        q, words = rest.configure(param, env={})
        assert q is None and "the built-in one" in words and why in words
        assert constants.HOME == HOME_20261021  # as the test's fixture left it: not even reset
    Path(rest.DEFAULT_REST_POSE).parent.mkdir(parents=True)
    Path(rest.DEFAULT_REST_POSE).write_text(
        "/**:\n  ros__parameters:\n    presets.rest: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\n"
    )
    q, words = rest.configure(None, env={})  # the default file there, with no home in it: the built-in pose too
    assert q is None and "names no presets.home" in words and constants.HOME == HOME_20261021
    (tmp_path / "other.yaml").write_text(
        "/**:\n  ros__parameters:\n    presets.rest: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\n"
    )
    with pytest.raises(rest.RestPoseError, match="names no presets.home"):
        rest.configure(str(tmp_path / "other.yaml"), env={})  # named, and it gives none: no silent built-in
    q, _words = rest.configure("/elsewhere/arm_presets.yaml", env={rest.REST_POSE_ENV: ""})  # the override: off
    assert q is None and constants.HOME == HOME_20261021


def test_the_built_in_pose_is_flatbots_home_of_2026_11_09():
    assert list(constants.HOME_BUILT_IN) == [0.4425, -1.5718, -2.9033, -2.1048, -0.7412, -0.9194, 2.1631]
    assert constants.HOME == list(constants.HOME_BUILT_IN)


# -- everything that reads HOME follows the configured pose --------------------------------------
def test_a_configured_pose_is_followed_by_everything_that_reads_home(tmp_path):
    from box_opening.models.container import load_press_demo
    from box_opening.runtime.client import pose_goal_joints

    built_in = list(constants.HOME)
    cfg = load_press_demo(CONTAINER)
    chain = _flatbot(tmp_path)
    key_before = rpb.zone_key(flatbot_urdf(), CFG, root="arm_mount")
    q, _words = rest.configure(_presets(tmp_path))
    assert q == POSE and constants.HOME == POSE and constants.HOME is not q
    # the run's rest, its look and the sweeps off it
    assert press_demo.rest_joints(cfg) == POSE
    look = press_demo.search_targets(cfg)[0]
    assert look[0] == "look" and look[1] == look_joints(POSE) != look_joints(built_in)
    # the side of the wrap the planner is handed its joints on: the pose's own
    turned = list(POSE)
    turned[2] += 2 * np.pi
    assert branches.planner_branch(turned) == pytest.approx(POSE)
    # the goal the body check never refuses is this pose, no longer the built-in one
    approach.set_rest_goals(None)
    assert approach._is_rest_goal(POSE) and not approach._is_rest_goal(built_in)
    # the reach proof starts from it, and its grids are kept under another key
    dry = rpb.DryClient(chain)
    assert dry.live == POSE and dry.seeds[0] == POSE and dry.seeds[1] == look_joints(POSE)
    assert rpb.zone_key(flatbot_urdf(), CFG, root="arm_mount") != key_before
    # the client's IK for a pose goal is seeded from the look above it (no error with the new pose)
    R, t = chain.fk(look_joints(POSE))
    from box_opening.runtime.kinematics import mat_to_quat_xyzw

    assert pose_goal_joints(list(t), mat_to_quat_xyzw(R), list(POSE)) is not None


def test_the_wrists_front_views_turn_from_the_configured_pose_and_back_to_it(tmp_path, monkeypatch):
    from box_opening.models.container import ContainerModel
    from box_opening.tasks import survey as sv

    chain = _flatbot(tmp_path, [("far", (3.0, 0.0, 0.5), (0.1, 0.1, 0.1))])
    rest.configure(_presets(tmp_path))
    rests = []

    class Band:  # the views' band is made on the rest pose the views turn from
        def __init__(self, check, rest_joints, **_kw):
            rests.append(list(rest_joints))
            raise RuntimeError("seen")

    monkeypatch.setattr(sv, "FrontBand", Band)
    with pytest.raises(RuntimeError, match="seen"):
        sv.angled_views(chain, chain.body, list(POSE), sv.camera(), ContainerModel.load(CONTAINER))
    with pytest.raises(RuntimeError, match="seen"):
        sv.again_view(chain, chain.body, list(POSE), sv.camera(), ContainerModel.load(CONTAINER), (0.6, 0.0), 0.3)
    assert rests == [POSE, POSE]
    # and a line-up that ends without its Open turns the wrist back to it
    _flatbot(tmp_path)
    monkeypatch.setattr(sv, "fly_view", lambda ctx, runner, view, band, execute: list(view.joints))
    assert sv.rest_wrist(None, None, False) == POSE


def test_no_module_reads_home_when_it_is_imported():
    """A configured pose is written into the one list (rest.use): a module that copied HOME, or derived
    a pose from it, as it was imported would keep the built-in one. A function's defaults count: they
    are read when the module is."""

    def at_import(node, out):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            args = node.args
            for part in [*getattr(node, "decorator_list", []), *args.defaults, *[d for d in args.kw_defaults if d]]:
                at_import(part, out)
            return  # its body runs when it is called
        if isinstance(node, ast.Name) and node.id == "HOME" and isinstance(node.ctx, ast.Load):
            out.append(node.lineno)
        if isinstance(node, ast.Attribute) and node.attr == "HOME":
            out.append(node.lineno)
        for child in ast.iter_child_nodes(node):
            at_import(child, out)

    found = {}
    for path in sorted((PKG / "box_opening").rglob("*.py")):
        if path.name == "constants.py":
            continue  # where the list is made, and its built-in copy taken
        lines = []
        at_import(ast.parse(path.read_text()), lines)
        if lines:
            found[str(path.relative_to(PKG))] = lines
    assert found == {}


# -- the command lines and the node ---------------------------------------------------------
def test_every_command_line_takes_rest_pose_and_the_default_is_the_robots_file():
    args = cli_common.make_parser("x").parse_args([])
    assert args.rest_pose is None and rest.resolve(args.rest_pose, env={}) == rest.DEFAULT_REST_POSE
    assert cli_common.make_parser("x").parse_args(["--rest-pose", ""]).rest_pose == ""


def _cli(monkeypatch):
    monkeypatch.setattr(cli_common, "init_runtime", lambda execute: (object(), FakeClient()))
    monkeypatch.setattr(cli_common, "_arm_urdf_path", lambda node, frame: flatbot_urdf())


def test_the_cli_stops_before_anything_moves_on_a_rest_pose_inside_the_padding(monkeypatch, tmp_path):
    _cli(monkeypatch)
    monkeypatch.delenv(body.BODY_LAYERS_ENV, raising=False)
    monkeypatch.setattr(cli_common, "Runner", lambda client: pytest.fail("a runner was made"))
    centre = _forearm(tmp_path, POSE)
    # (the person's layer: the installed robot-body config slows the arm near it, and a model without it is refused)
    layers = _planner(tmp_path, [("lap", centre, (0.1, 0.1, 0.1))], layer="person")
    args = Namespace(
        container=CONTAINER, bench_world=FLATBOT_WORLD, execute=True, body_layers=layers, rest_pose=_presets(tmp_path)
    )
    with pytest.raises(SystemExit) as stop:
        cli_common.build_ctx(args)
    said = str(stop.value)
    assert said.startswith("[") and "not starting: rest pose: presets.home of " in said
    assert "is refused: the arm would rest inside the chair and user's padding there: arm_forearm_link" in said
    assert said.endswith("Nothing moved.")
    assert constants.HOME == list(constants.HOME_BUILT_IN)
    # a file named and not there stops it too; with none named, or "", the run goes on with the built-in pose
    with pytest.raises(SystemExit, match="not starting: rest pose: cannot read /nowhere/arm_presets.yaml"):
        cli_common.build_ctx(Namespace(**{**vars(args), "body_layers": "", "rest_pose": "/nowhere/arm_presets.yaml"}))
    monkeypatch.setattr(cli_common, "Runner", lambda client: "runner")
    for rest_pose in (None, ""):
        cli_common.build_ctx(Namespace(**{**vars(args), "execute": False, "body_layers": "", "rest_pose": rest_pose}))
        assert constants.HOME == list(constants.HOME_BUILT_IN)


def test_home_arm_goes_to_the_robots_own_rest_pose(monkeypatch, tmp_path):
    _cli(monkeypatch)
    args = Namespace(
        container=CONTAINER, bench_world=FLATBOT_WORLD, execute=False, body_layers="", rest_pose=_presets(tmp_path)
    )
    ctx, _runner = cli_common.build_ctx(args)
    assert constants.HOME == POSE
    goals = []
    monkeypatch.setattr(home_arm, "home_any", lambda ctx, state, goal, plan_to, **kw: goals.append(list(goal)) or [])
    home_arm.build_legs(ctx)
    assert goals == [POSE]
    # without the file it is the built-in pose, as it always was
    constants.HOME[:] = constants.HOME_BUILT_IN
    ctx, _runner = cli_common.build_ctx(Namespace(**{**vars(args), "rest_pose": None}))
    home_arm.build_legs(ctx)
    assert goals[-1] == list(constants.HOME_BUILT_IN)


def test_a_run_takes_the_robots_rest_pose_before_its_first_move_and_stops_on_one_it_refuses(monkeypatch, tmp_path):
    """press_demo's own start (the run open_box flies): the pose is in force before the rest goals are set
    and before home_first is asked for anything; a refused one ends the run with nothing moved."""
    _cli(monkeypatch)
    called = []
    monkeypatch.setattr(press_demo, "home_first", lambda *a, **k: called.append("home_first") or True)
    monkeypatch.setattr(press_demo.Runner, "run", lambda self, *a, **k: called.append("run") or [])

    def far_enough(client):
        raise RuntimeError("far enough: the pose is in force")

    monkeypatch.setattr(press_demo, "Runner", far_enough)
    argv = ["press_demo", "--container", CONTAINER, "--bench-world", FLATBOT_WORLD, "--body-layers", ""]
    monkeypatch.setattr(sys, "argv", [*argv, "--rest-pose", _presets(tmp_path)])
    with pytest.raises(RuntimeError, match="far enough"):
        press_demo.main()
    assert constants.HOME == POSE and approach._REST_GOALS == [POSE, POSE]
    constants.HOME[:] = constants.HOME_BUILT_IN
    at_limit = [POSE[0], POSE[1], POSE[2], -2.50, POSE[4], POSE[5], POSE[6]]
    monkeypatch.setattr(sys, "argv", [*argv, "--rest-pose", _presets(tmp_path, home=at_limit)])
    with pytest.raises(SystemExit) as stop:
        press_demo.main()
    said = str(stop.value)
    assert said.startswith("[press_demo] not starting: rest pose: presets.home of ")
    assert "is refused: arm_joint_4 at -2.500 rad" in said and said.endswith("Nothing moved.")
    assert called == [] and constants.HOME == list(constants.HOME_BUILT_IN)


def test_the_recorder_takes_the_robots_rest_pose_before_it_reads_home():
    import inspect

    from box_opening.tasks import record_detection_set as rds

    src = inspect.getsource(rds.main)
    assert 'cli_common.load_rest_or_exit(args, "[record]")' in src
    assert src.index("load_rest_or_exit(") < src.index("HOME")


def test_the_reach_proof_starts_from_the_robots_own_rest_pose_and_refuses_one_no_run_starts_at(tmp_path):
    ctx, _cfg = rpb.setup(flatbot_urdf(), "arm_mount", "", CFG, rest_pose=_presets(tmp_path))
    assert constants.HOME == POSE and ctx.client.live == POSE
    constants.HOME[:] = constants.HOME_BUILT_IN
    ctx, _cfg = rpb.setup(flatbot_urdf(), "arm_mount", "", CFG, rest_pose="")
    assert ctx.client.live == list(constants.HOME_BUILT_IN)
    at_limit = [POSE[0], POSE[1], POSE[2], -2.50, POSE[4], POSE[5], POSE[6]]
    with pytest.raises(rest.RestPoseError, match="is refused: arm_joint_4"):
        rpb.setup(flatbot_urdf(), "arm_mount", "", CFG, rest_pose=_presets(tmp_path, home=at_limit))


def test_preflight_says_which_rest_pose_a_run_would_take_and_fails_one_it_would_refuse(tmp_path):
    centre = _forearm(tmp_path, POSE)
    layers = _planner(tmp_path, [("armrest", centre + np.array([0.0, 0.0, 0.40]), (0.1, 0.1, 0.1))])
    approach.set_body(None)
    level, text = preflight.rest_pose_verdict(_presets(tmp_path), flatbot_urdf(), "arm_mount", layers)
    assert level == "PASS" and "rest pose: presets.home of" in text and "mm outside the chair" in text
    constants.HOME[:] = constants.HOME_BUILT_IN
    layers = _planner(tmp_path, [("armrest", centre, (0.1, 0.1, 0.1))])
    level, text = preflight.rest_pose_verdict(_presets(tmp_path), flatbot_urdf(), "arm_mount", layers)
    assert level == "FAIL" and "is refused: the arm would rest inside" in text
    assert text.endswith("a run stops there before it moves")
    level, text = preflight.rest_pose_verdict("", flatbot_urdf(), "arm_mount", "")
    assert level == "PASS" and "the built-in one" in text
    level, text = preflight.rest_pose_verdict(_presets(tmp_path), None, "arm_mount", "")  # no description heard
    assert level == "FAIL" and "cannot be judged" in text
