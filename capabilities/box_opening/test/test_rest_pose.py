"""The arm's rest pose is the robot's own (runtime/rest): read from a file in the arm driver's presets format,
judged before anything moves (the built-in one too, wherever a chair model is loaded), handed by the node to
every process it starts, and followed by everything that reads HOME."""

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
# the same arm with the planner's self-collision rule in its robot file: a sphere on the base too, the
# gripper's grown by its link's self_collision_buffer (%s), and the forearm exempt against both: the one pair
# compared is the base and the gripper
SELF = """      arm_base_link:
      - {center: [0.0, 0.0, 0.1], radius: 0.07}
    self_collision_buffer: {arm_base_link: 0.0, arm_robotiq_85_base_link: %s}
    self_collision_ignore:
      arm_forearm_link: [arm_robotiq_85_base_link, arm_base_link]
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
    rest._CONFIGURED = False


def _presets(tmp, home=POSE, name="arm_presets.yaml", node="/**", nested=False):
    """A presets file as the arm's driver is handed it (rammp-deployments config/robots/<robot>/): its path."""
    q = "[%s]" % ", ".join(repr(v) for v in home) if isinstance(home, list | tuple) else str(home)
    lines = ["%s:" % node, "  ros__parameters:"]
    lines += ["    presets:", "      home: %s" % q] if nested else ["    presets.home: %s" % q]
    (tmp / name).write_text("\n".join(lines) + "\n")
    return str(tmp / name)


def _planner(tmp, boxes, layer="seat", robot=ROBOT, more=""):
    """A planner config in `tmp` with one layer of `boxes` [(name, centre, dims)]: its path. `more`: further
    lines of the planner config itself."""
    lines = ["layer: %s" % layer, "obstacles:"]
    lines += ["  - {name: %s, position: %s, dims: %s}" % (n, [float(v) for v in c], list(d)) for n, c, d in boxes]
    (tmp / "layer_seat.yaml").write_text("\n".join(lines) + "\n")
    (tmp / "robot.yaml").write_text(robot)
    (tmp / "planner.yaml").write_text("robot: robot.yaml\nlayers: [layer_seat.yaml]\n" + more)
    return str(tmp / "planner.yaml")


def _flatbot(tmp, boxes=None, **kw):
    """flatbot's chain, with a body model of `boxes` attached where given (the planner's spheres alone)."""
    approach.set_body(False if boxes is None else body.load(_planner(tmp, boxes, **kw), robot_body=False))
    approach.set_chain_root("arm_mount", flatbot_urdf())
    return approach._chain()


FAR = [("far", (3.0, 0.0, 0.5), (0.1, 0.1, 0.1))]  # a chair model whose one box is nowhere near the arm


def _under(centre, gap_m):
    """A box under the forearm's sphere (60 + 5 mm) whose padded top (20 mm) stands `gap_m` under it."""
    top = centre[2] - 0.065 - 0.020 - gap_m
    return [("pad", (centre[0], centre[1], top - 0.05), (0.3, 0.3, 0.1))]


def _forearm(tmp, q):
    """Where the test robot file's forearm sphere stands at `q` (arm_mount)."""
    chain = _flatbot(tmp, [("far", (3.0, 0.0, 0.5), (0.1, 0.1, 0.1))])
    C, _r, links = chain.body.spheres_at(q)
    return np.array(C[list(links).index("arm_forearm_link")], float)


# -- where the pose comes from -------------------------------------------------------------
def test_the_environment_beats_the_param_and_the_flag_and_empty_is_the_built_in_pose():
    assert rest.resolve(None, env={}) == rest.DEFAULT_REST_POSE
    assert rest.resolve("/x/arm_presets.yaml", env={}) == "/x/arm_presets.yaml"
    assert rest.resolve("/x/arm_presets.yaml", env={rest.REST_POSE_ENV: "/y/p.yaml"}) == "/y/p.yaml"
    assert rest.resolve("/x/arm_presets.yaml", env={rest.REST_POSE_ENV: ""}) == ""
    assert rest.resolve("", env={}) == ""


def test_the_default_file_is_the_one_the_arms_driver_is_handed(monkeypatch):
    monkeypatch.undo()  # this test reads the module as it ships
    assert rest.DEFAULT_REST_POSE == "/config/robot/arm_presets.yaml" and rest.REST_POSE_KEY == "presets.home"
    assert rest.REST_POSE_ENV == "BOX_OPENING_REST_POSE" and rest.DRIVER_NODE == "kinova_gen3_node"


def test_the_pose_is_the_drivers_presets_home_however_the_params_file_writes_it(tmp_path):
    assert rest.read(_presets(tmp_path)) == POSE  # rammp-deployments' own: /** and the dotted name
    assert rest.read(_presets(tmp_path, nested=True)) == POSE
    for node in ("kinova_gen3_node", "/kinova_gen3_node", "/**/kinova_gen3_node", "/arm/kinova_gen3_node"):
        assert rest.read(_presets(tmp_path, node=node)) == POSE
    # the team's own files read as written (flatbot's is the built-in pose)
    (tmp_path / "flatbot.yaml").write_text(
        "# flatbot's named arm poses\n/**:\n  ros__parameters:\n"
        "    presets.home: [0.4425, -1.5718, -2.9033, -2.1048, -0.7412, -0.9194, 2.1631]\n"
    )
    assert rest.read(tmp_path / "flatbot.yaml") == list(constants.HOME_BUILT_IN)


def test_another_nodes_block_is_not_the_drivers_pose(tmp_path):
    """Only what the arm's driver would read counts: a block for some other node gives the driver nothing."""
    path = _presets(tmp_path, node="/arm/driver")
    assert rest.read(path) is None
    with pytest.raises(rest.RestPoseError, match="names no presets.home"):
        rest.read(path, named=True)
    (tmp_path / "both.yaml").write_text(
        "/camera:\n  ros__parameters:\n    presets.home: %r\n/**:\n  ros__parameters:\n    presets.home: %r\n"
        % ([v + 0.1 for v in POSE], POSE)
    )
    assert rest.read(tmp_path / "both.yaml") == POSE  # the other node's does not make it two


def test_a_file_that_is_not_there_or_names_no_home_gives_none_unless_it_was_named(tmp_path):
    assert rest.read(tmp_path / "nowhere.yaml") is None
    texts = {
        "other.yaml": "/**:\n  ros__parameters:\n    presets.rest: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\n",
        "empty.yaml": "",
        "typo.yaml": "/**:\n  ros__parameters:\n    presets.Home: %r\n" % POSE,
        "null.yaml": "/**:\n  ros__parameters:\n    presets.home:\n",
        "bare.yaml": "presets.home: %r\n" % POSE,
        "noparams.yaml": "/**:\n  presets.home: %r\n" % POSE,
    }
    for name, text in texts.items():
        (tmp_path / name).write_text(text)
        assert rest.read(tmp_path / name) is None, name
        with pytest.raises(rest.RestPoseError, match="%s names no presets.home" % name):
            rest.read(tmp_path / name, named=True)
    with pytest.raises(rest.RestPoseError, match="cannot read .*nowhere.yaml"):
        rest.read(tmp_path / "nowhere.yaml", named=True)


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


@pytest.mark.parametrize("home", ["[0, -1.1685, -2.8871, -1.8516, -0.856, -1, 2.2684]", "[0, 1, 2, -2, 0, 1, 2]"])
def test_a_home_with_a_whole_number_in_it_is_an_error_as_the_driver_does_not_start_on_it(tmp_path, home):
    with pytest.raises(rest.RestPoseError, match="has a whole number among its angles .*write 0.0, not 0"):
        rest.read(_presets(tmp_path, home=home))


def test_a_file_that_cannot_be_parsed_or_whose_blocks_disagree_is_an_error(tmp_path):
    (tmp_path / "bad.yaml").write_text("/**:\n  ros__parameters: [\n")
    with pytest.raises(rest.RestPoseError, match="cannot read"):
        rest.read(tmp_path / "bad.yaml")
    other = [v + 0.1 for v in POSE]
    two = (
        "/**:\n  ros__parameters:\n    presets.home: %r\nkinova_gen3_node:\n  ros__parameters:\n    presets.home: %r\n"
    )
    (tmp_path / "two.yaml").write_text(two % (POSE, other))
    with pytest.raises(rest.RestPoseError, match="more than one presets.home .*kinova_gen3_node.*neither is taken"):
        rest.read(tmp_path / "two.yaml")
    (tmp_path / "same.yaml").write_text(two % (POSE, POSE))
    assert rest.read(tmp_path / "same.yaml") == POSE


def test_a_process_says_first_which_numbers_it_rests_at_and_where_they_are_from(tmp_path):
    line = rest.said(_presets(tmp_path), env={})
    assert line.startswith("rest pose: [0.1670, -1.7970, -3.0370, -2.1690, -0.1110, -0.5830, 1.8250], presets.home of ")
    assert str(tmp_path / "arm_presets.yaml") in line and "(digest %s)" % rest.digest(POSE) in line
    line = rest.said("", env={})
    assert "the built-in one (--rest-pose / BOX_OPENING_REST_POSE empty)" in line
    assert "[0.4425, -1.5718, -2.9033, -2.1048, -0.7412, -0.9194, 2.1631]" in line
    assert rest.digest(constants.HOME_BUILT_IN) in line and rest.digest(POSE) != rest.digest(constants.HOME_BUILT_IN)
    # the digest is the numbers', to 0.1 mrad: the same pose read by two processes is one word
    assert rest.digest(POSE) == rest.digest([v + 4e-5 for v in POSE]) != rest.digest([v + 2e-4 for v in POSE])


# -- whatever pose a run rests at is judged before anything moves ------------------------------
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
    got = rest.configure(_presets(tmp_path))
    assert got.pose == POSE == constants.HOME == got.numbers and got.source == str(tmp_path / "arm_presets.yaml")
    assert "presets.home of %s" % (tmp_path / "arm_presets.yaml") in got.words
    assert "mm outside the chair and user's padding at its nearest (arm_" in got.words


def test_a_pose_too_near_the_padding_is_refused_and_one_inside_the_planners_standoff_is_warned_of(tmp_path):
    """A rest pose stands at least as far outside the padding as the distance at which every way home begins
    with a step clear (10 mm): nearer, each run would start by stepping away from its own rest pose. Between
    that and the planner's activation distance (30 mm) it is taken, with a warning."""
    centre = _forearm(tmp_path, POSE)
    assert rest.clear_m() == pytest.approx(0.010)
    for gap, refused, warned in (
        (0.035, None, False),
        (0.012, None, True),
        (0.008, "8 mm from", None),
        (-0.002, "inside", None),
    ):
        chain = _flatbot(tmp_path, _under(centre, gap))
        why = rest.refusal(POSE, chain, chain.body)
        if refused:
            assert why is not None and refused in why, (gap, why)
            if gap > 0:
                assert "nearer than the 10 mm inside which every way home begins with a step clear" in why
            with pytest.raises(rest.RestPoseError, match="is refused"):
                rest.configure(_presets(tmp_path))
            continue
        assert why is None, (gap, why)
        got = rest.configure(_presets(tmp_path))
        assert bool(got.warnings) == warned, (gap, got.warnings)
        if warned:
            assert "stands 12 mm outside the chair and user's padding" in got.warnings[0]
            assert "inside the 30 mm at which the planner starts pushing a path away" in got.warnings[0]


def test_a_pose_with_a_joint_near_or_past_its_limit_is_refused(tmp_path):
    chain = _flatbot(tmp_path)
    assert rest.refusal(POSE, chain) is None
    # what the plan-free step toward a goal asks of it (0.18) and what a hold may stand off its pose (0.05):
    # gen_1_5's own presets.home passed the first alone by 0.0006 rad (2026-11-19)
    assert rest.limit_room_rad() == pytest.approx(0.23)
    near = list(POSE)
    near[3] = -2.57 + 0.22
    why = rest.refusal(near, chain)
    assert "arm_joint_4 at -2.350 rad stands 0.220 rad from its limit (-2.57 to 2.57): a rest pose keeps 0.23" in why
    assert "the 0.18 the planner needs of a start and the 0.05 a hold may stand off its pose" in why
    near[3] = -2.57 + 0.24
    assert rest.refusal(near, chain) is None
    team = [0.2742, -1.6604, -2.7760, -2.3894, -0.6979, -0.5593, 2.3104]  # gen_1_5's presets.home of 2026-11-18
    assert "arm_joint_4 at -2.389 rad stands 0.181 rad from its limit" in rest.refusal(team, chain)
    past = list(POSE)
    past[1] = 2.30
    assert "arm_joint_2 at 2.300 rad is past its limit (-2.24 to 2.24)" in rest.refusal(past, chain)
    wound = list(POSE)
    wound[2] = 6.5
    assert "arm_joint_3 at 6.500 rad is past the planner's bound" in rest.refusal(wound, chain)
    assert "it has 6 joint angles" in rest.refusal(POSE[:6], chain)
    _flatbot(tmp_path)
    with pytest.raises(rest.RestPoseError, match="is refused: arm_joint_4 at -2.350 rad") as refused:
        rest.configure(_presets(tmp_path, home=[POSE[0], POSE[1], POSE[2], -2.35, POSE[4], POSE[5], POSE[6]]))
    # what to do about it, in words that hold for a joint limit as for the chair
    assert "Give the robot a rest pose it can start from there (or name another file" in str(refused.value)
    assert constants.HOME == list(constants.HOME_BUILT_IN)


def test_everything_wrong_with_a_pose_is_said_the_chair_first(tmp_path):
    """gen_1_5's own presets.home (2026-11-19) stands 58 mm inside the padding and 0.181 rad from a joint limit:
    the refusal said the limit alone, and the reader fixed the lesser thing."""
    low = [POSE[0], POSE[1], POSE[2], -2.39, POSE[4], POSE[5], POSE[6]]
    chain = _flatbot(tmp_path, [("armrest", _forearm(tmp_path, low), (0.1, 0.1, 0.1))])
    why = rest.refusal(low, chain, chain.body)
    assert why.startswith("the arm would rest inside the chair and user's padding there: arm_forearm_link")
    assert "; and arm_joint_4 at -2.390 rad stands 0.180 rad from its limit" in why
    with pytest.raises(rest.RestPoseError, match="is refused: the arm would rest inside .*; and arm_joint_4 at"):
        rest.configure(_presets(tmp_path, home=low))


def test_a_pose_that_touches_itself_by_the_planners_own_rule_is_refused(tmp_path):
    """cuRobo's self-collision rule, from the robot file (the reviewer, 2026-11-19: 19 of 1626 random poses the
    limits and the padding let through touched themselves, by up to 81 mm)."""
    pair = {"arm_base_link", "arm_robotiq_85_base_link"}
    chain = _flatbot(tmp_path, FAR, robot=ROBOT + SELF % 0.0)
    gap = chain.body.self_gap(POSE)
    assert gap.m > 0.05 and {gap.link, gap.box} == pair  # the forearm is exempt against both: not compared
    assert rest.refusal(POSE, chain, chain.body) is None
    got = rest.configure(_presets(tmp_path))
    assert "%.0f mm clear of itself (arm_" % (1000 * gap.m) in got.words
    # the same two spheres, the gripper's grown by its link's self_collision_buffer until they meet
    chain = _flatbot(tmp_path, FAR, robot=ROBOT + SELF % round(gap.m + 0.012, 4))
    assert chain.body.self_gap(POSE).m == pytest.approx(-0.012, abs=1e-4)
    why = rest.refusal(POSE, chain, chain.body)
    assert "the arm would touch itself there by the planner's own rule: arm_" in why
    assert "arm_base_link" in why and "arm_robotiq_85_base_link" in why
    assert "12 mm into each other (a rest pose keeps 10 mm)" in why
    assert "the planner plans nothing from a start that touches itself" in why
    with pytest.raises(rest.RestPoseError, match="is refused: the arm would touch itself there"):
        rest.configure(_presets(tmp_path))
    chain = _flatbot(tmp_path, FAR, robot=ROBOT + SELF % round(gap.m - 0.004, 4))  # 4 mm apart: under the 10 kept
    why = rest.refusal(POSE, chain, chain.body)
    assert "would stand too near itself" in why and "4 mm apart (a rest pose keeps 10 mm)" in why
    assert rest.REST_SELF_M == 0.010
    # a robot file that names no exemptions has no such check, and the words say so
    chain = _flatbot(tmp_path, FAR)
    assert chain.body.self_gap(POSE) is None
    assert "not judged against itself" in rest.configure(_presets(tmp_path)).words


def test_the_planners_padding_and_standoff_are_read_where_its_config_keeps_them(tmp_path):
    """The reviewer's low 6 (2026-11-19): body.load read world_padding at the top level only, and rammp-curobo's
    schema keeps it under `planner:` (a config that changed it there was padded 20 mm here all the same)."""
    plain = body.load(_planner(tmp_path, FAR), robot_body=False)
    assert plain.padding_m == 0.02 and plain.activation_m == 0.03  # the planner's own defaults
    under = "planner:\n  world_padding: 0.035\n  collision_activation_distance: 0.045\n"
    model = body.load(_planner(tmp_path, FAR, more=under), robot_body=False)
    assert model.padding_m == 0.035 and model.activation_m == 0.045
    model = body.load(_planner(tmp_path, FAR, more="world_padding: 0.03\n"), robot_body=False)
    assert model.padding_m == 0.03  # at the top level, as box opening's own study configs wrote it
    with pytest.raises(body.BodyModelError, match="world_padding is given more than once and differs"):
        body.load(
            _planner(tmp_path, FAR, more="world_padding: 0.03\nplanner:\n  world_padding: 0.02\n"), robot_body=False
        )
    # the rest pose's warning follows the planner's own standoff: 35 mm outside is inside a 45 mm one
    centre = _forearm(tmp_path, POSE)
    _flatbot(tmp_path, _under(centre, 0.035), more="planner:\n  collision_activation_distance: 0.045\n")
    got = rest.configure(_presets(tmp_path))
    assert len(got.warnings) == 1 and "inside the 45 mm at which the planner starts pushing" in got.warnings[0]


def test_with_no_chair_model_the_words_say_the_pose_was_not_judged_against_any(tmp_path):
    _flatbot(tmp_path)
    got = rest.configure(_presets(tmp_path))
    assert got.pose == POSE and "NOT judged against any chair, nor against itself" in got.words


def test_a_configured_pose_that_cannot_be_judged_is_refused(tmp_path):
    approach.set_body(False)
    approach.set_chain_root("arm_mount", None)  # no description heard: no limits to judge it by
    with pytest.raises(rest.RestPoseError, match="cannot be judged: no arm URDF rooted at arm_mount"):
        rest.configure(_presets(tmp_path))
    approach.set_chain_root("no_such_link", flatbot_urdf())  # a description with no chain from that root
    with pytest.raises(rest.RestPoseError, match=r"cannot be judged: (\w+: )?URDF .* has no chain"):
        rest.configure(_presets(tmp_path))
    assert constants.HOME == list(constants.HOME_BUILT_IN)
    # the built-in pose with nothing known of the arm is as it always was, and the words say it was not judged
    got = rest.configure("")
    assert got.pose is None and "NOT judged (no arm kinematics" in got.words


# -- the built-in pose: left as it stands, and judged wherever a chair model is loaded -----------
def test_with_no_pose_configured_home_is_not_touched(tmp_path, home_20261021):
    _flatbot(tmp_path)
    for param, why in (
        ("", "empty"),
        (None, "(no %s)" % rest.DEFAULT_REST_POSE),
    ):  # asked for; the default file not there
        got = rest.configure(param, env={})
        assert got.pose is None and got.source is None and "the built-in one" in got.words and why in got.words
        assert constants.HOME == HOME_20261021 == got.numbers  # as the test's fixture left it: not even reset
        assert got.warnings == []  # no robot's config is mounted here: nothing was passed over
    (tmp_path / "other.yaml").write_text(
        "/**:\n  ros__parameters:\n    presets.rest: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\n"
    )
    with pytest.raises(rest.RestPoseError, match="names no presets.home"):
        rest.configure(str(tmp_path / "other.yaml"), env={})  # named, and it gives none: no silent built-in
    got = rest.configure("/elsewhere/arm_presets.yaml", env={rest.REST_POSE_ENV: ""})  # the environment: off
    assert got.pose is None and constants.HOME == HOME_20261021


def test_the_built_in_pose_taken_where_a_robots_config_is_mounted_is_said_as_a_warning(tmp_path):
    """The reviewer's 4 (2026-11-19): a mounted robot config with no usable presets.home (no file, no node
    block, a typo in the key, a null) fell back to the built-in pose in silence."""
    _flatbot(tmp_path)
    Path(rest.DEFAULT_REST_POSE).parent.mkdir(parents=True)  # the robot's config folder, with no presets file
    got = rest.configure(None, env={})
    assert got.pose is None and len(got.warnings) == 1
    assert (
        "a robot's config is mounted at %s but gives no rest pose" % Path(rest.DEFAULT_REST_POSE).parent
        in got.warnings[0]
    )
    assert (
        "resting at box opening's BUILT-IN pose [0.4425, -1.5718," in got.warnings[0] and "flatbot" in got.warnings[0]
    )
    Path(rest.DEFAULT_REST_POSE).write_text("/**:\n  ros__parameters:\n    presets.Home: %r\n" % POSE)  # a typo
    got = rest.configure(None, env={})
    assert got.pose is None and "names no presets.home" in got.warnings[0]
    assert rest.configure("", env={}).warnings == []  # the built-in pose asked for by name: no warning


def test_the_built_in_pose_is_judged_too_wherever_a_chair_model_is_loaded(tmp_path):
    """On the Gen1.5 chair the built-in pose stands 33 mm inside the padding (2026-11-19): taken in silence it
    was a rest pose no run could leave. Now it is refused like any other."""
    built = list(constants.HOME)
    centre = _forearm(tmp_path, built)
    _flatbot(tmp_path, [("armrest", centre, (0.1, 0.1, 0.1))])
    for param in ("", None):
        with pytest.raises(rest.RestPoseError) as refused:
            rest.configure(param, env={})
        said = str(refused.value)
        assert said.startswith("rest pose: the built-in one (") and "[0.4425, -1.5718," in said
        assert "is refused: the arm would rest inside the chair and user's padding there: arm_forearm_link" in said
    _flatbot(tmp_path, FAR)
    got = rest.configure("", env={})
    assert got.pose is None and "mm outside the chair and user's padding" in got.words
    # with no chair model it is not judged at all, as on the bench it never was: not even its joints
    _flatbot(tmp_path)
    constants.HOME[3] = -2.50
    assert rest.configure("", env={}).pose is None


def test_asking_for_the_built_in_pose_after_a_configured_one_puts_the_built_in_numbers_back(tmp_path):
    _flatbot(tmp_path)
    assert rest.configure(_presets(tmp_path)).pose == POSE == constants.HOME
    got = rest.configure("")
    assert got.pose is None and got.numbers == list(constants.HOME_BUILT_IN) == constants.HOME
    assert "[0.4425, -1.5718, -2.9033, -2.1048, -0.7412, -0.9194, 2.1631]" in got.words


def test_the_built_in_pose_is_flatbots_home_of_2026_11_09():
    assert list(constants.HOME_BUILT_IN) == [0.4425, -1.5718, -2.9033, -2.1048, -0.7412, -0.9194, 2.1631]
    assert constants.HOME == list(constants.HOME_BUILT_IN)


# -- one rest pose for every process of a deployment -------------------------------------------
def test_a_process_that_reads_another_pose_than_the_node_did_stops(tmp_path):
    _flatbot(tmp_path)
    path = _presets(tmp_path)
    assert rest.configure(path, expect=rest.digest(POSE)).pose == POSE
    constants.HOME[:] = constants.HOME_BUILT_IN
    _presets(tmp_path, home=[v + 0.01 for v in POSE])  # the file changed after the node read it
    with pytest.raises(rest.RestPoseError) as stop:
        rest.configure(path, expect=rest.digest(POSE))
    said = str(stop.value)
    assert "this process reads presets.home of %s" % path in said and "the node that started it read another" in said
    assert rest.digest(POSE) in said and constants.HOME == list(constants.HOME_BUILT_IN)
    with pytest.raises(rest.RestPoseError, match="the node that started it read another"):
        rest.configure("", expect=rest.digest(POSE))  # the node read a file, this process the built-in pose
    assert rest.configure("", expect=rest.digest(constants.HOME_BUILT_IN)).pose is None


# -- everything that reads HOME follows the configured pose --------------------------------------
def test_a_configured_pose_is_followed_by_everything_that_reads_home(tmp_path):
    from box_opening.models.container import load_press_demo
    from box_opening.runtime.client import pose_goal_joints

    built_in = list(constants.HOME)
    cfg = load_press_demo(CONTAINER)
    chain = _flatbot(tmp_path)
    key_before = rpb.zone_key(flatbot_urdf(), CFG, root="arm_mount")
    got = rest.configure(_presets(tmp_path))
    assert got.pose == POSE and constants.HOME == POSE and constants.HOME is not got.pose
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

    chain = _flatbot(tmp_path, FAR)
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


# -- the command lines ---------------------------------------------------------------------
def test_every_command_line_takes_rest_pose_and_the_default_is_the_robots_file():
    args = cli_common.make_parser("x").parse_args([])
    assert args.rest_pose is None and rest.resolve(args.rest_pose, env={}) == rest.DEFAULT_REST_POSE
    assert args.rest_pose_digest is None
    args = cli_common.make_parser("x").parse_args(["--rest-pose", "", "--rest-pose-digest", "abc"])
    assert args.rest_pose == "" and args.rest_pose_digest == "abc"


def _cli(monkeypatch):
    monkeypatch.setattr(cli_common, "init_runtime", lambda execute: (object(), FakeClient()))
    monkeypatch.setattr(cli_common, "_arm_urdf_path", lambda node, frame: flatbot_urdf())


def test_the_cli_stops_before_anything_moves_on_a_rest_pose_inside_the_padding(monkeypatch, tmp_path):
    _cli(monkeypatch)
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
    # a file named and not there stops it too, at its first words; so does a pose the node read otherwise
    with pytest.raises(SystemExit, match="not starting: rest pose: cannot read /nowhere/arm_presets.yaml"):
        cli_common.build_ctx(Namespace(**{**vars(args), "body_layers": "", "rest_pose": "/nowhere/arm_presets.yaml"}))
    with pytest.raises(SystemExit, match="not starting: rest pose: this process reads .* the node that started it"):
        cli_common.build_ctx(Namespace(**{**vars(args), "body_layers": "", "rest_pose_digest": "0123456789ab"}))
    # with none named, or "", the run goes on with the built-in pose
    monkeypatch.setattr(cli_common, "Runner", lambda client: "runner")
    for rest_pose in (None, ""):
        cli_common.build_ctx(Namespace(**{**vars(args), "execute": False, "body_layers": "", "rest_pose": rest_pose}))
        assert constants.HOME == list(constants.HOME_BUILT_IN)


def test_a_command_lines_first_words_are_its_rest_pose_and_a_warning_is_said_as_one(monkeypatch, tmp_path, capsys):
    _cli(monkeypatch)
    monkeypatch.setattr(cli_common, "Runner", lambda client: "runner")
    centre = _forearm(tmp_path, POSE)
    layers = _planner(tmp_path, _under(centre, 0.012), layer="person")
    capsys.readouterr()
    cli_common.build_ctx(
        Namespace(
            container=CONTAINER,
            bench_world=FLATBOT_WORLD,
            execute=False,
            body_layers=layers,
            rest_pose=_presets(tmp_path),
        )
    )
    lines = capsys.readouterr().out.splitlines()
    assert "] rest pose: [0.1670, -1.7970, -3.0370, -2.1690, -0.1110, -0.5830, 1.8250], presets.home of " in lines[0]
    assert "(digest %s)" % rest.digest(POSE) in lines[0]
    warned = [ln for ln in lines if "WARNING" in ln]
    assert len(warned) == 1 and "inside the 30 mm at which the planner starts pushing a path away" in warned[0]


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
    ctx, _runner = cli_common.build_ctx(Namespace(**{**vars(args), "rest_pose": None}))
    home_arm.build_legs(ctx)
    assert goals[-1] == list(constants.HOME_BUILT_IN) == constants.HOME


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
    monkeypatch.setattr(
        sys, "argv", [*argv, "--rest-pose", _presets(tmp_path), "--rest-pose-digest", rest.digest(POSE)]
    )
    with pytest.raises(RuntimeError, match="far enough"):
        press_demo.main()
    assert constants.HOME == POSE and approach._REST_GOALS == [POSE, POSE]
    at_limit = [POSE[0], POSE[1], POSE[2], -2.50, POSE[4], POSE[5], POSE[6]]
    monkeypatch.setattr(sys, "argv", [*argv, "--rest-pose", _presets(tmp_path, home=at_limit)])
    with pytest.raises(SystemExit) as stop:
        press_demo.main()
    said = str(stop.value)
    assert said.startswith("[press_demo] not starting: rest pose: presets.home of ")
    assert "is refused: arm_joint_4 at -2.500 rad" in said and said.endswith("Nothing moved.")
    # the node read one pose and the run another (the file changed in between): the run stops
    monkeypatch.setattr(sys, "argv", [*argv, "--rest-pose", _presets(tmp_path), "--rest-pose-digest", "0123456789ab"])
    with pytest.raises(SystemExit, match="the node that started it read another"):
        press_demo.main()
    assert called == []


def test_the_recorder_sets_the_arm_and_the_chair_before_it_takes_the_rest_pose_and_runs_on(monkeypatch, tmp_path):
    """The reviewer's 5 (2026-11-19): the recorder took the rest pose with no chain root and no body model
    set, and any presets.home there ended it in an uncaught PackageNotFoundError. Run, not read."""
    from box_opening.tasks import record_detection_set as rds

    seen = {}

    def no_description(node, frame):
        seen["root"] = frame
        return None  # the bench: the installed description

    monkeypatch.setattr(cli_common, "init_runtime", lambda execute: (object(), FakeClient()))
    monkeypatch.setattr(cli_common, "_arm_urdf_path", no_description)

    def far_enough(client):
        raise RuntimeError("far enough: root %s, body %s, HOME %s" % (approach._ROOT, approach._BODY, constants.HOME))

    import box_opening.runtime.runner as runner_module

    monkeypatch.setattr(runner_module, "Runner", far_enough)
    bench = str(PKG / "config" / "world_bench.yaml")
    argv = [
        "record_detection_set",
        "--container",
        CONTAINER,
        "--bench-world",
        bench,
        "--body-layers",
        "",
        "--no-prompt",
    ]
    on_bench = [0.0, 0.262, 3.142, -2.269, 0.0, 0.96, 1.571]  # dojo's presets.home: the bench arm's own
    monkeypatch.setattr(sys, "argv", [*argv, "--rest-pose", _presets(tmp_path, home=on_bench)])
    with pytest.raises(RuntimeError, match="far enough: root base_link, body "):
        rds.main()
    assert seen["root"] == "base_link" and constants.HOME == on_bench
    # a pose the bench arm cannot rest at ends it in a line, not a traceback
    monkeypatch.setattr(
        sys, "argv", [*argv, "--rest-pose", _presets(tmp_path, home=[0.0, 0.262, 3.142, -2.50, 0.0, 0.96, 1.571])]
    )
    with pytest.raises(SystemExit, match=r"\[record\] not starting: rest pose: .* is refused: .*joint_4"):
        rds.main()
    # and a description that is not there (no package, no URDF): a line too
    monkeypatch.setattr(
        approach, "_chain", lambda: (_ for _ in ()).throw(LookupError("package 'kinova_gen3_description' not found"))
    )
    monkeypatch.setattr(sys, "argv", [*argv, "--rest-pose", _presets(tmp_path, home=on_bench)])
    with pytest.raises(SystemExit, match=r"\[record\] not starting: rest pose: .* cannot be judged: LookupError"):
        rds.main()


def test_the_reach_proof_starts_from_the_robots_own_rest_pose_and_refuses_one_no_run_starts_at(tmp_path):
    ctx, _cfg = rpb.setup(flatbot_urdf(), "arm_mount", "", CFG, rest_pose=_presets(tmp_path))
    assert constants.HOME == POSE and ctx.client.live == POSE
    ctx, _cfg = rpb.setup(flatbot_urdf(), "arm_mount", "", CFG, rest_pose="")
    assert ctx.client.live == list(constants.HOME_BUILT_IN)
    at_limit = [POSE[0], POSE[1], POSE[2], -2.50, POSE[4], POSE[5], POSE[6]]
    with pytest.raises(rest.RestPoseError, match="is refused: arm_joint_4"):
        rpb.setup(flatbot_urdf(), "arm_mount", "", CFG, rest_pose=_presets(tmp_path, home=at_limit))
    with pytest.raises(rest.RestPoseError, match="the node that started it read another"):
        rpb.setup(flatbot_urdf(), "arm_mount", "", CFG, rest_pose=_presets(tmp_path), rest_digest="0123456789ab")


def test_the_reach_probes_command_line_hands_its_rest_pose_on_and_a_refused_one_ends_it_in_a_line(
    monkeypatch, tmp_path, capsys
):
    """The reviewer's 6 and 7 (2026-11-19): nothing failed when main dropped --rest-pose, and a refused pose
    ended the probe in a traceback."""
    import os

    got = []

    def setup(*a):
        got.append(a)
        raise rest.RestPoseError("rest pose: refused for this test")

    monkeypatch.setattr(os, "nice", lambda _by: 0)  # the test's own process keeps its priority
    monkeypatch.setattr(rpb, "setup", setup)
    path = _presets(tmp_path)
    argv = ["--urdf", flatbot_urdf(), "--container", CFG, "--body-layers", "", "--serve"]
    capsys.readouterr()
    assert rpb.main([*argv, "--rest-pose", path, "--rest-pose-digest", rest.digest(POSE)]) == 2
    assert got[0][4:] == (path, rest.digest(POSE))
    said = capsys.readouterr()
    err = said.err.splitlines()
    # its first words, on stderr (its stdout is the node's): the pose it reads; then one line, no traceback
    assert err[0].startswith("reach_probe: rest pose: [0.1670, -1.7970,") and rest.digest(POSE) in err[0]
    assert err[-1] == "reach_probe not starting: rest pose: refused for this test. Nothing is proven."
    assert "Traceback" not in said.err and said.out == ""
    # a file named and not there ends it the same way
    assert rpb.main([*argv, "--rest-pose", str(tmp_path / "nowhere.yaml")]) == 2
    assert "reach_probe not starting: rest pose: cannot read" in capsys.readouterr().err
    # ... and for real: the proof of one place starts from the pose the file gives
    starts = []

    def proof(ctx, cfg, at, table_z=None):
        starts.append(list(ctx.client.live))
        return True, "", 30

    monkeypatch.undo()
    monkeypatch.setattr(os, "nice", lambda _by: 0)
    monkeypatch.setenv("BOX_OPENING_STATE", str(tmp_path))
    monkeypatch.delenv(rest.REST_POSE_ENV, raising=False)
    monkeypatch.delenv(body.BODY_LAYERS_ENV, raising=False)
    monkeypatch.setattr(rpb, "press_plans", proof)
    assert rpb.main([*argv[:-1], "--rest-pose", path, "--at", "0.6", "0.0", "0.3"]) == 0
    assert starts == [POSE]


# -- preflight: the rest pose, the fixed mounts, the planner's canary ------------------------------
def test_preflight_says_which_rest_pose_a_run_would_take_and_fails_one_it_would_refuse(tmp_path):
    centre = _forearm(tmp_path, POSE)
    layers = _planner(tmp_path, [("armrest", centre + np.array([0.0, 0.0, 0.40]), (0.1, 0.1, 0.1))])
    approach.set_body(None)
    rows = preflight.rest_pose_rows(_presets(tmp_path), flatbot_urdf(), "arm_mount", layers)
    assert [lvl for lvl, _t in rows] == ["PASS"]
    assert "rest pose: presets.home of" in rows[0][1] and "mm outside the chair" in rows[0][1]
    constants.HOME[:] = constants.HOME_BUILT_IN
    layers = _planner(tmp_path, [("armrest", centre, (0.1, 0.1, 0.1))])
    rows = preflight.rest_pose_rows(_presets(tmp_path), flatbot_urdf(), "arm_mount", layers)
    assert [lvl for lvl, _t in rows] == ["FAIL"] and "is refused: the arm would rest inside" in rows[0][1]
    assert rows[0][1].endswith("a run stops there before it moves")
    # a pose a run takes with a warning is a PASS row and a WARN row
    layers = _planner(tmp_path, _under(centre, 0.012))
    rows = preflight.rest_pose_rows(_presets(tmp_path), flatbot_urdf(), "arm_mount", layers)
    assert [lvl for lvl, _t in rows] == ["PASS", "WARN"] and "inside the 30 mm" in rows[1][1]
    rows = preflight.rest_pose_rows("", flatbot_urdf(), "arm_mount", "")
    assert rows[0][0] == "PASS" and "the built-in one" in rows[0][1] and "NOT judged against any chair" in rows[0][1]
    rows = preflight.rest_pose_rows(_presets(tmp_path), None, "arm_mount", "")  # no description heard
    assert rows[0][0] == "FAIL" and "cannot be judged" in rows[0][1]
    rows = preflight.rest_pose_rows(_presets(tmp_path), flatbot_urdf(), "arm_mount", "", expect="0123456789ab")
    assert rows[0][0] == "FAIL" and "the node that started it read another" in rows[0][1]


MOUNTS = """# camera mounts, in rammp-calibration's mounts.yaml format
mounts:
  ros__parameters:
    transforms: [scene_camera, wrist_camera]
    scene_camera:
      parent: arm_mount
      child: scene_camera
      xyz: [0.151329, -0.571954, 0.391385]
      quat_xyzw: [0.0137645, 0.0170993, 0.2148372, 0.9764032]
    wrist_camera:
      parent: arm_end_effector_link
      child: wrist_camera
      xyz: [-0.005351, 0.06953, 0.022506]
      quat_xyzw: [0.5062226, 0.506811, 0.4947239, -0.4920666]
"""
SCENE = ([0.151329, -0.571954, 0.391385], [0.0137645, 0.0170993, 0.2148372, 0.9764032])
WRIST = ([-0.005351, 0.06953, 0.022506], [0.5062226, 0.506811, 0.4947239, -0.4920666])


def test_preflight_fails_a_mount_in_tf_that_is_not_the_mounted_robot_configs(tmp_path):
    """Stand-ins only (a dict for TF): never run against the robot's TF tree."""
    path = tmp_path / "cameras.yaml"
    path.write_text(MOUNTS)
    assert preflight.config_mounts(path) == [
        ("arm_mount", "scene_camera", *SCENE),
        ("arm_end_effector_link", "wrist_camera", *WRIST),
    ]
    tf = {("arm_mount", "scene_camera"): SCENE, ("arm_end_effector_link", "wrist_camera"): WRIST}
    rows = preflight.mount_rows(lambda parent, child: tf.get((parent, child)), path)
    assert [lvl for lvl, _t in rows] == ["PASS", "PASS"]
    assert "TF arm_mount -> scene_camera is the mounted robot config's mount" in rows[0][1]
    # the same turn written with the other sign, and a mount 1 mm off: the same mount
    tf[("arm_end_effector_link", "wrist_camera")] = ([-0.005351, 0.06953, 0.023506], [-v for v in WRIST[1]])
    assert [lvl for lvl, _t in preflight.mount_rows(lambda p, c: tf.get((p, c)), path)] == ["PASS", "PASS"]
    # another robot's scene camera (flatbot's 2026-11-12 calibration: 159 mm and 26 deg from this file's)
    tf[("arm_mount", "scene_camera")] = ([0.3064, -0.5723, 0.4270], [-0.0067, 0.0607, -0.0113, 0.9981])
    rows = preflight.mount_rows(lambda p, c: tf.get((p, c)), path)
    assert [lvl for lvl, _t in rows] == ["FAIL", "PASS"]
    assert "TF arm_mount -> scene_camera is NOT the mounted robot config's mount: 159 mm and 2" in rows[0][1]
    assert "the stack publishing TF was started on another robot's config" in rows[0][1]
    # a turn alone fails it too (1 deg about z), and a frame TF does not have is a warning, as a file not there
    half = np.radians(1.0) / 2
    x, y, z, w = SCENE[1]
    turned = [x * np.cos(half) + y * np.sin(half), y * np.cos(half) - x * np.sin(half)]
    turned += [z * np.cos(half) + w * np.sin(half), w * np.cos(half) - z * np.sin(half)]
    tf[("arm_mount", "scene_camera")] = (SCENE[0], turned)
    rows = preflight.mount_rows(lambda p, c: tf.get((p, c)), path)
    assert rows[0][0] == "FAIL" and "0 mm and 1.0 deg" in rows[0][1]
    del tf[("arm_mount", "scene_camera")]
    rows = preflight.mount_rows(lambda p, c: tf.get((p, c)), path)
    assert rows[0][0] == "WARN" and "no TF arm_mount -> scene_camera to compare" in rows[0][1]
    rows = preflight.mount_rows(lambda p, c: tf.get((p, c)), tmp_path / "nowhere.yaml")
    assert [lvl for lvl, _t in rows] == ["WARN"] and "no mounts to compare TF with" in rows[0][1]
    assert preflight.MOUNT_FILES == ("cameras.yaml", "world.yaml")
    assert preflight.robot_config_dir("/config/robot/curobo/planner.yaml") == Path("/config/robot")


class Holds:
    """The planner's stand-in for preflight's canary: plan(goal, start) refuses, in cuRobo's words, a start
    that stands in the padding of the chair it holds (`check`, a Body; None: an empty world), and plans
    everything else. Never the real planner."""

    def __init__(self, check, words="MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION"):
        self.check, self.words, self.asked = check, words, []

    def __call__(self, goal, start):
        self.asked.append((list(goal), list(start)))
        inside = self.check is not None and not self.check.pose_clear(list(start))
        return Namespace(success=not inside, message=self.words if inside else "ok")


def _armrest(tmp_path, at=(0.6, 0.3, 0.0, 0.3)):
    """A chair model of one seat box (a 0.24 m cube) where the forearm stands with joints 1-4 moved by `at`
    from POSE: (chain, its Body)."""
    there = [a + b for a, b in zip(POSE, [*at, 0.0, 0.0, 0.0], strict=True)]
    centre = _forearm(tmp_path, there)
    chain = _flatbot(tmp_path, [("armrest", centre, (0.24, 0.24, 0.24))])
    return chain, chain.body


def test_the_canary_is_taken_from_the_mounted_chair_model_itself(tmp_path):
    chain, check = _armrest(tmp_path)
    starts = preflight.canary_starts(check, chain, POSE)
    assert 1 <= len(starts) <= preflight.CANARIES
    for q, box, depth in starts:
        assert box == "seat/armrest" and depth >= preflight.CANARY_IN_BOX_M
        near = check.distance(q, layers=("seat",))
        assert near.m == pytest.approx(-depth) and near.link == "arm_forearm_link"
        assert not check.pose_clear(q)  # deep in the padded box: no planner that holds it takes this start
        for v, (lo, hi) in zip(q, chain.limits, strict=True):
            assert lo is None or (v - lo >= preflight.CANARY_LIMIT_RAD and hi - v >= preflight.CANARY_LIMIT_RAD)
        assert q[4:] == POSE[4:]  # the wrist as it rests
    assert preflight.canary_starts(check, chain, POSE) == starts  # the same every time
    # a chair somewhere else gives other starts or none: nothing here is a constant
    _chain, moved = _armrest(tmp_path, at=(-0.6, 0.3, 0.0, 0.3))
    assert [s[0] for s in preflight.canary_starts(moved, chain, POSE)] != [s[0] for s in starts]
    # a model with the user's boxes alone has no canary (the planner may leave the user out)
    chain = _flatbot(tmp_path, FAR, layer="person")
    assert preflight.canary_starts(chain.body, chain, POSE) == []


def test_preflights_canary_fails_a_planner_that_does_not_hold_this_chair(tmp_path):
    """Stand-ins only (Holds): never run against the real planner. What the two plans prove: a planner that
    plans from a start the mounted chair model puts deep in the chair does not hold this chair; one that
    refuses it holds something there."""
    chain, check = _armrest(tmp_path)
    n = len(preflight.canary_starts(check, chain, POSE))
    # the planner holds this chair: the rest pose taken, every canary refused as a start in its world
    planner = Holds(check)
    rows = preflight.planner_canary_rows(planner, check, chain, POSE)
    assert [lvl for lvl, _t in rows] == ["PASS"] * (1 + n)
    assert "the planner takes the rest pose as a start" in rows[0][1]
    assert "the planner refuses a start the mounted chair model puts" in rows[1][1]
    assert "mm inside seat/armrest, as a start in its world" in rows[1][1]
    assert planner.asked[0][1] == POSE and planner.asked[0][0][:6] == POSE[:6]  # from the rest pose, a wrist turn
    assert all(goal == POSE for goal, _start in planner.asked[1:])  # from each canary, to the rest pose
    # an empty world, and another robot's chair (the box somewhere else): they plan from the canary
    _chain, elsewhere = _armrest(tmp_path, at=(-0.6, 0.3, 0.0, 0.3))
    for other in (None, elsewhere):
        rows = preflight.planner_canary_rows(Holds(other), check, chain, POSE)
        assert [lvl for lvl, _t in rows] == ["PASS"] + ["FAIL"] * n
        assert "the planner PLANS from a start the mounted chair model puts" in rows[1][1]
        assert "it does not hold this robot's chair" in rows[1][1] and "whatever its config's path says" in rows[1][1]
    # a planner that refuses the rest pose itself (a chair this pose stands in)
    stands_in = _flatbot(tmp_path, [("armrest", _forearm(tmp_path, POSE), (0.24, 0.24, 0.24))]).body
    rows = preflight.planner_canary_rows(Holds(stands_in), check, chain, POSE)
    assert rows[0][0] == "FAIL" and "the planner REFUSES a plan from the rest pose" in rows[0][1]
    assert "INVALID_START_STATE_WORLD_COLLISION" in rows[0][1]
    # refused, but not for its world (the arm touching itself, a joint limit): it proves nothing
    rows = preflight.planner_canary_rows(
        Holds(check, "MotionGenStatus.INVALID_START_STATE_SELF_COLLISION"), check, chain, POSE
    )
    assert [lvl for lvl, _t in rows] == ["PASS"] + ["WARN"] * n and "but not for its world" in rows[1][1]
    # no answer at all, and a chair model with no canary near the rest pose
    rows = preflight.planner_canary_rows(lambda goal, start: None, check, chain, POSE)
    assert [lvl for lvl, _t in rows] == ["FAIL"] and "did not answer" in rows[0][1]
    far = _flatbot(tmp_path, FAR)
    rows = preflight.planner_canary_rows(Holds(far.body), far.body, far, POSE)
    assert [lvl for lvl, _t in rows] == ["PASS", "WARN"] and "no canary start" in rows[1][1]
    # ... and the warning says how far the search for one got: draws, inside the limits, deep in a chair box
    assert (
        "(of 4000 draws " in rows[1][1]
        and " 0 of them with the upper arm or the forearm deep in a chair box" in rows[1][1]
    )


class Ready:
    """preflight's client with everything it asks ready; its planner holds nothing (it plans from any
    start). `tf`: {(parent, child): (xyz, quat)} for the mounts it is asked."""

    _tip_frames = ["arm_left_tip", "arm_right_tip"]
    tf = {}

    def __init__(self, node=None):
        self._base_frame = None

    def joints(self):
        return list(constants.HOME)

    def efforts_present(self):
        return True

    def driver_reachable(self, timeout_s=0.0):
        return True

    def driver_health(self, timeout_s=0.0):
        return None

    def planner_reachable(self, timeout_s=0.0):
        return True

    def plan_to_joints(self, q7, start_joints):
        return Namespace(success=True, message="ok")

    def tf_lookup(self, target, source, timeout_s=0.0):
        return self.tf.get((target, source))

    def contact_xyz(self, timeout_s=0.0):
        return [0.4, 0.0, 0.3]

    def tf_ready(self, target, source, timeout_s=0.0):
        return True

    def other_arm_clients(self):
        return []


def _preflight(monkeypatch, argv, world=("world_empty.yaml", None)):
    """preflight.main() with stand-ins for everything on the graph: its exit code."""
    monkeypatch.setattr(preflight, "rmw_refusal", lambda env: None)
    monkeypatch.setattr(preflight, "rclpy", Namespace(init=lambda: None, create_node=lambda name: object()))
    monkeypatch.setattr(preflight, "PlannerClient", Ready)
    monkeypatch.setattr(preflight, "planner_world", lambda node, timeout_s=3.0, with_config=False: world)
    monkeypatch.setattr(preflight, "_gripper_state", lambda node: Namespace(present=True, position=0.0))
    monkeypatch.setattr(preflight, "_listen", lambda node, topics: ({t: 9 for t, _r, _f in topics}, 9))
    monkeypatch.setattr(cli_common, "_arm_urdf_path", lambda node, frame: flatbot_urdf())
    monkeypatch.setattr(sys, "argv", ["preflight", "--bench-world", FLATBOT_WORLD, *argv])
    with pytest.raises(SystemExit) as stop:
        preflight.main()
    return stop.value.code


def test_preflight_ends_in_failure_on_a_rest_pose_a_mount_or_a_canary_that_fails(monkeypatch, tmp_path, capsys):
    """The reviewer's 6 (2026-11-19): nothing failed when preflight said the rest row's FAIL and did not
    count it. Its main, run on stand-ins for the graph."""
    good = _presets(tmp_path)
    capsys.readouterr()
    assert _preflight(monkeypatch, ["--body-layers", "", "--rest-pose", good]) == 0
    out = capsys.readouterr().out
    assert "rest pose: [0.1670, -1.7970," in out.splitlines()[0]  # its first words: the pose it reads
    assert "PASS  rest pose: presets.home of" in out and "FAIL" not in out
    constants.HOME[:] = constants.HOME_BUILT_IN
    at_limit = _presets(tmp_path, home=[POSE[0], POSE[1], POSE[2], -2.50, POSE[4], POSE[5], POSE[6]], name="bad.yaml")
    assert _preflight(monkeypatch, ["--body-layers", "", "--rest-pose", at_limit]) == 1
    out = capsys.readouterr().out
    assert "FAIL  rest pose: presets.home of" in out and "is refused: arm_joint_4 at -2.500 rad" in out
    assert out.count("FAIL") == 1  # that row alone ends it in failure
    # with a chair model on: the fixed mounts and the planner's canary are judged, and counted
    robot = tmp_path / "robot"
    (robot / "curobo").mkdir(parents=True)
    (robot / "transforms").mkdir()
    layers = _planner(robot / "curobo", FAR, layer="person")
    (robot / "transforms" / "cameras.yaml").write_text(MOUNTS)
    monkeypatch.setattr(
        Ready, "tf", {("arm_mount", "scene_camera"): SCENE, ("arm_end_effector_link", "wrist_camera"): WRIST}
    )
    argv = ["--body-layers", layers, "--rest-pose", good]
    assert _preflight(monkeypatch, argv, world=("", layers)) == 0
    out = capsys.readouterr().out
    assert "PASS  the planner takes the rest pose as a start" in out and "WARN  no canary start" in out
    assert out.count("is the mounted robot config's mount") == 2 and "WARN  no mounts to compare TF with" in out
    monkeypatch.setattr(Ready, "tf", {("arm_mount", "scene_camera"): ([0.3064, -0.5723, 0.4270], SCENE[1])})
    assert _preflight(monkeypatch, argv, world=("", layers)) == 1
    out = capsys.readouterr().out
    assert out.count("FAIL") == 1 and "FAIL  TF arm_mount -> scene_camera is NOT the mounted robot config's" in out
    monkeypatch.setattr(Ready, "tf", {})
    there = [a + b for a, b in zip(POSE, [0.6, 0.3, 0.0, 0.3, 0.0, 0.0, 0.0], strict=True)]
    monkeypatch.setattr(
        preflight, "canary_starts", lambda check, chain, rest_q, stats=None: [(there, "seat/armrest", 0.05)]
    )
    assert _preflight(monkeypatch, argv, world=("", layers)) == 1
    out = capsys.readouterr().out
    assert out.count("FAIL") == 1 and "FAIL  the planner PLANS from a start the mounted chair model puts 50 mm" in out
