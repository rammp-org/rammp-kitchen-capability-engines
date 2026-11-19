"""The wrist lens from the tool frame is the robot's own calibration, not a copy kept here.

config/camera_oak_wrist.yaml kept it by hand, [0, 0.065, -0.105]: the mount fitted from a depth-press
run on 2026-10-22, from a tool frame 0.120 m out of the flange. A copy for one mount is wrong on the
next (#105 wrote one in the wrong frame, #106 took it back; on gen_1_5 it stands 7.0 mm aside of the
2026-11-18 calibration). With the robot's config mounted (/config/robot) the lens and the camera's axes
are read from its transforms/cameras.yaml, the mount's parent placed from the flange by the arm's own
URDF. The figure kept by hand stays for what it is: the one measurement of this lens no calibration
session made, which a run reports its distance from and the depth press holds TF's mount against.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from conftest import arm_poses, both_chains, flange_pose, flatbot_urdf, puck_urdf

from box_opening.perception import d405
from box_opening.runtime import approach
from box_opening.tasks import cli_common
from box_opening.tasks import depth_press as dp
from box_opening.tasks import survey as sv

pytestmark = pytest.mark.usefixtures("approach_state")

COPY = [0.0, 0.065, -0.105]  # config/camera_oak_wrist.yaml's own figure: the 2026-10-22 fit
# the wrist camera's mount on the flange, as each robot config's transforms/cameras.yaml has had it: flatbot's
# fitted one (rammp-deployments #128), the hand-eye calibration of 2026-11-09 box opening's copy took on 11-10
# (every press on it came down ~1 cm to the right), and gen_1_5's calibration of 2026-11-18
MOUNT_FLATBOT = ([0.0, 0.065, 0.015], [-0.5, -0.5, -0.5, 0.5])
MOUNT_1109 = ([-0.010948, 0.061275, 0.022497], [0.5089592, 0.5007825, 0.498786, -0.4913144])
MOUNT_GEN15 = ([-0.005351, 0.06953, 0.022506], [0.5062226, 0.506811, 0.4947239, -0.4920666])
LENS_GEN15 = [-0.005351, 0.06953, -0.097494]  # its lens from the tool frame, 0.120 m out of the flange


def _robot_config(tmp_path, mount, robot="gen_1_5", parent="arm_end_effector_link", kinematics=None):
    """A robot config folder as rammp-deployments lays it out (mount.yaml, transforms/cameras.yaml and,
    `kinematics` given, curobo/arm_kinematics.json): the path of its cameras.yaml."""
    root = Path(tmp_path) / "robots" / ("%s_%s" % (robot, parent))
    (root / "transforms").mkdir(parents=True, exist_ok=True)
    (root / "mount.yaml").write_text("robot: %s\nrpy_deg: [0.0, 30.0, 0.0]\n" % robot)
    xyz, quat = mount
    (root / "transforms" / "cameras.yaml").write_text(
        "mounts:\n  ros__parameters:\n    transforms: [scene_camera, wrist_camera]\n"
        "    scene_camera:\n      parent: arm_mount\n      child: scene_camera\n"
        "      xyz: [0.151329, -0.571954, 0.391385]\n      quat_xyzw: [0.0, 0.0, 0.0, 1.0]\n"
        "    wrist_camera:\n      parent: %s\n      child: wrist_camera\n      xyz: %s\n"
        "      quat_xyzw: %s\n" % (parent, list(xyz), list(quat))
    )
    if kinematics is not None:
        (root / "curobo").mkdir(exist_ok=True)
        (root / "curobo" / "arm_kinematics.json").write_text(json.dumps(kinematics))
    return str(root / "transforms" / "cameras.yaml")


def _kinematics_of(chain):
    """The planner's kinematics reference for the description `chain` was built from, as arm_kinematics.json has it."""
    return {
        j[0]: {"parent": j[2], "child": child, "type": j[1], "xyz": [float(v) for v in j[3]]}
        for child, j in chain._by_child.items()
    }


# ---- the lens and the axes ----------------------------------------------------------------------------


def test_the_wrist_lens_is_the_robots_calibration_whichever_description_the_run_is_on(tmp_path):
    """The tool frame is 0.120 m out of the flange on every description, so a mount on the flange gives
    one lens: flatbot's the very figure that was kept by hand, gen_1_5's its 2026-11-18 calibration. It
    is where TF has the lens, and it says which file and which robot it came from."""
    cfg = d405.camera_config("camera_oak_wrist.yaml")
    assert cfg["robot_mount"] == "wrist_camera" and cfg["wrist_lens_in_hand_m"] == COPY
    flatbot = _robot_config(tmp_path, MOUNT_FLATBOT, robot="flatbot_fwd8")
    gen15 = _robot_config(tmp_path, MOUNT_GEN15)
    for chain in both_chains(tmp_path):
        lens, where = d405.wrist_lens_in_hand(cfg, chain=chain, path=flatbot)
        assert (
            lens == COPY
            and where == "wrist_camera in %s (robot flatbot_fwd8 by its mount.yaml), on arm_end_effector_link" % flatbot
        )
        lens, where = d405.wrist_lens_in_hand(cfg, chain=chain, path=gen15)
        assert (
            lens == LENS_GEN15
            and where == "wrist_camera in %s (robot gen_1_5 by its mount.yaml), on arm_end_effector_link" % gen15
        )
        q = arm_poses()[0]
        Rf, f = flange_pose(chain, q)
        R, t = chain.fk(q)
        assert np.allclose(np.asarray(R).T @ (f + Rf @ np.asarray(MOUNT_GEN15[0]) - t), lens, atol=1e-9)
    # the copy kept by hand, on gen_1_5: 7 mm sideways of it, over the 5 mm the depth press centred within
    aside = math.hypot(LENS_GEN15[0] - COPY[0], LENS_GEN15[1] - COPY[1])
    assert aside == pytest.approx(0.0070, abs=1e-4) and aside > dp.MOUNT_XY_TOL_M


def test_the_mounts_parent_is_placed_from_the_flange_by_the_arms_own_fixed_joints(tmp_path):
    """The robot may mount the camera on any link of the hand: the puck's link (the flange's own frame),
    the gripper's base or the description's tool frame (14 and 134 mm out of it with the puck). The
    same place on the arm gives the same lens; a parent that moves against the flange gives none, and
    the camera config's own figure stands in, saying why."""
    cfg = d405.camera_config("camera_oak_wrist.yaml")
    plain, puck = both_chains(tmp_path)
    xyz, quat = MOUNT_GEN15
    for parent, out in (("arm_sensor_puck_link", 0.0), ("arm_robotiq_85_base_link", 0.014), ("arm_tool_frame", 0.134)):
        path = _robot_config(tmp_path, ([xyz[0], xyz[1], xyz[2] - out], quat), parent=parent)
        lens, optical, where = d405.wrist_camera_in_hand(cfg, chain=puck, path=path)
        assert lens == pytest.approx(LENS_GEN15, abs=1e-9) and where.endswith(", on %s" % parent), parent
        assert np.allclose(optical, d405.quat_to_mat(*quat) @ d405.BODY_TO_OPTICAL, atol=1e-9)
    # on the description without the puck the gripper's base IS the flange, and there is no puck link
    path = _robot_config(tmp_path, MOUNT_GEN15, parent="arm_robotiq_85_base_link")
    assert d405.wrist_lens_in_hand(cfg, chain=plain, path=path)[0] == LENS_GEN15
    path = _robot_config(tmp_path, MOUNT_GEN15, parent="arm_sensor_puck_link")
    lens, where = d405.wrist_lens_in_hand(cfg, chain=plain, path=path)
    assert lens == COPY and "arm_sensor_puck_link, which the arm's URDF does not fix to its flange" in where
    path = _robot_config(tmp_path, MOUNT_GEN15, parent="arm_forearm_link")
    lens, where = d405.wrist_lens_in_hand(cfg, chain=puck, path=path)
    assert lens == COPY and "arm_forearm_link moves against arm_end_effector_link" in where


def test_the_cameras_own_figure_stands_in_where_no_calibration_can_be_read(tmp_path, capsys):
    cfg = d405.camera_config("camera_oak_wrist.yaml")
    plain, _puck = both_chains(tmp_path)
    gen15 = _robot_config(tmp_path, MOUNT_GEN15)
    # no robot config here (a test, an offline tool), or reading it turned off
    assert d405.wrist_camera_in_hand(cfg, chain=plain, path=str(tmp_path / "nowhere.yaml")) == (
        COPY,
        None,
        "the camera config (no robot calibration of wrist_camera here)",
    )
    assert d405.wrist_lens_in_hand(cfg, chain=plain, env={d405.ROBOT_CAMERAS_ENV: ""})[0] == COPY
    assert d405.wrist_lens_in_hand(cfg, chain=plain, env={d405.ROBOT_CAMERAS_ENV: gen15})[0] == LENS_GEN15
    # no chain to place the mount's parent by (no description heard)
    approach.set_chain_root("arm_mount", None)
    lens, where = d405.wrist_lens_in_hand(cfg, path=gen15)
    assert lens == COPY and "no arm chain" in where
    # a file that is no mounts file: the copy, never a stop, and said
    for text in (
        "mounts: {ros__parameters: {wrist_camera: {parent: arm_end_effector_link, xyz: [1, 2]}}}\n",
        "mounts: {ros__parameters: {wrist_camera: {parent: arm_end_effector_link, xyz: [0, 0, 0], "
        "quat_xyzw: [0, 0, 0, 0]}}}\n",
    ):
        junk = tmp_path / "junk.yaml"
        junk.write_text(text)
        capsys.readouterr()
        assert d405.wrist_lens_in_hand(cfg, chain=plain, path=str(junk))[0] == COPY
        assert "WARNING: the robot's camera mounts %s give no usable wrist_camera" % junk in capsys.readouterr().out
    # the bench's D405 names no robot mount: its own file holds its mount, and its figure stands
    bench = d405.camera_config("camera_d405_wrist.yaml")
    assert d405.wrist_lens_in_hand(bench, chain=plain, path=gen15) == ([0.0, 0.075, -0.125], "the camera config")


def test_a_run_takes_its_lens_and_its_cameras_axes_from_the_mounted_robot_config(tmp_path, monkeypatch):
    """The grabber's lens (what the depth press checks TF against and the lid's table map takes the hand
    out by) and the survey's camera follow the robot's calibration. gen_1_5's mount is 1.5 deg off
    square: the survey that judged a lid in view by a square camera was 10 px out at the frame's edge,
    against the 8 px its view keeps (VIEW_MARGIN_PX). flatbot's square mount is the square camera,
    exactly."""
    monkeypatch.setenv(d405.ROBOT_CAMERAS_ENV, _robot_config(tmp_path, MOUNT_GEN15))
    approach.set_body(False)
    approach.set_chain_root("arm_mount", puck_urdf(tmp_path))
    g = d405.D405Grabber.__new__(d405.D405Grabber)
    g._cfg = d405.camera_config("camera_oak_wrist.yaml")
    assert g.wrist_lens_in_hand_m == LENS_GEN15
    cam = sv.camera_of(None, "camera_oak_wrist.yaml")
    assert np.allclose(cam.lens, LENS_GEN15)
    off = math.degrees(math.acos(float(cam.optical[:, 2] @ sv.OPTICAL_IN_TOOL[:, 2])))
    assert off == pytest.approx(1.52, abs=0.02) and np.allclose(cam.optical.T @ cam.optical, np.eye(3), atol=1e-9)
    assert math.tan(math.radians(off)) * cam.k[0, 0] > sv.VIEW_MARGIN_PX  # what the square camera was out by
    monkeypatch.setenv(d405.ROBOT_CAMERAS_ENV, _robot_config(tmp_path, MOUNT_FLATBOT, robot="flatbot_fwd8"))
    cam = sv.camera_of(None, "camera_oak_wrist.yaml")
    assert np.array_equal(cam.optical, sv.OPTICAL_IN_TOOL) and cam.lens.tolist() == COPY
    # and with no robot config, as every test and offline tool before: the camera config's, the square mount
    monkeypatch.setenv(d405.ROBOT_CAMERAS_ENV, "")
    assert g.wrist_lens_in_hand_m == COPY
    cam = sv.camera_of(None, "camera_oak_wrist.yaml")
    assert np.allclose(cam.lens, sv.LENS_IN_TOOL_M) and np.array_equal(cam.optical, sv.OPTICAL_IN_TOOL)


# ---- what still checks the mount ------------------------------------------------------------------------


def _optical(mount):
    return d405.quat_to_mat(*mount[1]) @ d405.BODY_TO_OPTICAL


def _lens(mount):
    return [mount[0][0], mount[0][1], mount[0][2] - 0.120]


def test_a_mount_is_held_by_where_it_aims_against_the_mounts_presses_have_flown_on():
    """What a mount does to a press is where it puts the point the look aims at: its lens and its axes
    together, 0.30 m out. By the lens alone gen_1_5's calibration stands 7.0 mm from the fit and the
    2026-11-09 one 11.6; by the aimed point it is the other way round, 13.9 and 10.9 mm, and gen_1_5's
    aims 11.1 mm (4.4 across, 10.2 along the hand's y) from the mount the 11-12..17 flights flew on. An
    aim more than a centimetre from both flown mounts' is one no press has flown near."""
    cfg = d405.camera_config("camera_oak_wrist.yaml")
    (fit_what, fit_lens, fit_axes), (flown_what, flown_lens, flown_axes) = d405.flown_mounts(cfg)
    assert fit_what == "the 2026-10-22 fit" and fit_lens == COPY and np.array_equal(fit_axes, d405.SQUARE_OPTICAL)
    assert flown_what == "the mount flown 2026-11-12..17" and flown_lens == pytest.approx(_lens(MOUNT_1109))
    assert np.allclose(flown_axes, _optical(MOUNT_1109))
    assert np.allclose(d405.aim_point(COPY), [0.0, 0.065, 0.195]) and d405.LOOK_RANGE_M == 0.30
    off = {
        name: d405.aim_off_flown(cfg, _lens(m), _optical(m))
        for name, m in (("1109", MOUNT_1109), ("1118", MOUNT_GEN15))
    }
    assert [round(1000 * a, 1) for _w, a, _d in off["1109"]] == [10.9, 0.0]
    assert [round(1000 * a, 1) for _w, a, _d in off["1118"]] == [13.9, 11.1]
    assert np.round(1000 * np.asarray(off["1118"][0][2]), 1).tolist() == [-6.3, 12.4, 7.4]  # from the fit's
    assert np.round(1000 * np.asarray(off["1118"][1][2][:2]), 1).tolist() == [4.4, 10.2]  # from the flown mount's
    # the lens alone (its axes taken square) would have said 7.0 mm: the mount's turn is the other half
    assert d405.aim_off_flown(cfg, _lens(MOUNT_GEN15), None)[0][1] == pytest.approx(0.0070, abs=1e-4)
    # the two flown mounts stand; gen_1_5's is refused, saying both distances; a mount 20 mm along the axis too
    assert (
        d405.aim_refusal(cfg, COPY, None) is None
        and d405.aim_refusal(cfg, _lens(MOUNT_1109), _optical(MOUNT_1109)) is None
    )
    why = d405.aim_refusal(cfg, _lens(MOUNT_GEN15), _optical(MOUNT_GEN15))
    assert "13.9 mm from the 2026-10-22 fit's and 11.1 mm from the mount flown 2026-11-12..17's" in why
    assert d405.AIM_REF_TOL_M == 0.010 and d405.aim_refusal(cfg, [0.006, 0.071, -0.105], None) is None  # 8.5 mm
    assert d405.aim_refusal(cfg, [0.0, 0.065, -0.080], None) is not None  # 25 mm along the axis
    # a config that keeps no flown mount holds nothing; no lens, nothing
    assert d405.aim_refusal({}, COPY, None) is None and d405.aim_off_flown(cfg, None) == []


@pytest.mark.usefixtures("home_20261021")
def test_the_depth_press_does_not_centre_on_a_mount_that_aims_where_no_press_has_flown(ctx, monkeypatch, tmp_path):
    """TF is published from the robot's cameras.yaml, so TF against that file (the grabber's
    wrist_lens_in_hand_m) can only catch a TF tree and a config of two robots. What is independent is
    where TF's mount aims against the flown mounts': gen_1_5's, lens and axes, is not centred on, though
    its lens alone is 7 mm from the fit's; a mount 5 mm from the fit is."""
    from test_depth_press import _press, _seen

    cfg = d405.camera_config("camera_oak_wrist.yaml")
    real = dp.press_down_from_here

    def with_flown_mounts(node, ctx_, cfg_, runner, watcher, *a, **k):
        watcher.grab.aim_refusal = lambda lens, optical=None: d405.aim_refusal(cfg, lens, optical)
        return real(node, ctx_, cfg_, runner, watcher, *a, **k)

    monkeypatch.setattr(dp, "press_down_from_here", with_flown_mounts)
    tf, axes = tuple(LENS_GEN15), _optical(MOUNT_GEN15)
    result, events, arm, _l, _s = _press(
        ctx, monkeypatch, tmp_path, [_seen((0.0, 0.0), lens=tf, r_hc=axes)], lens=tf, r_hc=axes, measured=tf
    )
    assert result.free and "no press has flown near" in result.why and "13.9 mm from the 2026-10-22 fit's" in result.why
    assert events == ["depth:turn", "look:depth-look"] and arm.lines == [] and dp.exit_code(result) == 1
    tf = (0.004, COPY[1] + 0.003, COPY[2])
    seen = _seen((0.02, 0.03), lens=tf)
    result, events, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, [seen, seen, seen.after], lens=tf, measured=tf)
    assert dp.exit_code(result) == 0 and events[:3] == ["depth:turn", "look:depth-look", "depth:centre"]


# ---- what a run says of its hand ----------------------------------------------------------------------


def test_a_run_says_where_it_has_its_hand_before_it_moves(tmp_path, monkeypatch, capsys):
    """One line at the start: the tool frame and the point the heights go by (as flown, on either
    description), where the pads really end, the link the planner plans and the step to it, the lens
    with the file and the robot it came from, and where its aim lands against the two flown mounts'.
    On gen_1_5 that is more than a centimetre from both, and said as a warning; a guarded run, which
    centres its press on the mount, stops there with the arm at rest."""
    approach.set_body(False)
    gen15 = _robot_config(tmp_path, MOUNT_GEN15)
    monkeypatch.setenv(d405.ROBOT_CAMERAS_ENV, gen15)
    approach.set_chain_root("arm_mount", puck_urdf(tmp_path))
    capsys.readouterr()
    lines = cli_common.hand_or_exit("[press_demo]", "camera_oak_wrist.yaml")
    assert capsys.readouterr().out.splitlines() == lines and len(lines) == 2
    assert lines[0] == (
        "[press_demo] hand: tool frame 120 mm out of the flange, the point the heights go by 131 mm out, as "
        "flown, whatever the description; this hand's pads end 151 mm out, 20 mm past it; the planner plans "
        "arm_tool_frame (the arm's description), (+0.0, +0.0, +14.0) mm from the tool frame: a pose goes to it "
        "there; wrist lens (-5.4, +69.5, -97.5) mm from the tool frame (wrist_camera in %s (robot gen_1_5 by its "
        "mount.yaml), on arm_end_effector_link); its aim 300 mm out lands (-6.3, +12.4) mm from the 2026-10-22 "
        "fit's and (+4.4, +10.2) mm from the mount flown 2026-11-12..17's (across the hand: x, y)" % gen15
    )
    assert lines[1].startswith("[press_demo] WARNING: the wrist camera's mount is one no press has flown near: its aim")
    assert "13.9 mm from the 2026-10-22 fit's and 11.1 mm from the mount flown 2026-11-12..17's" in lines[1]
    with pytest.raises(SystemExit, match=r"\[press_demo\] not starting: a guarded run centres .*Nothing moved"):
        cli_common.hand_or_exit("[press_demo]", "camera_oak_wrist.yaml", guarded=True)
    assert "WARNING: the wrist camera's mount is one no press has flown near" in capsys.readouterr().out
    # the mount the flights flew on: said, no warning, and a guarded run starts
    monkeypatch.setenv(d405.ROBOT_CAMERAS_ENV, _robot_config(tmp_path, MOUNT_1109, robot="flatbot_fwd8"))
    lines = cli_common.hand_or_exit("[press_demo]", "camera_oak_wrist.yaml", guarded=True)
    assert len(lines) == 1 and lines[0].endswith(
        "its aim 300 mm out lands (-10.7, +2.2) mm from the 2026-10-22 fit's and (+0.0, +0.0) mm from the mount "
        "flown 2026-11-12..17's (across the hand: x, y)"
    )
    monkeypatch.setenv(d405.ROBOT_CAMERAS_ENV, "")
    approach.set_chain_root("arm_mount", flatbot_urdf())
    assert cli_common.hand_or_exit("[press_demo]", "camera_oak_wrist.yaml", guarded=True) == [
        "[press_demo] hand: tool frame 120 mm out of the flange, the point the heights go by 131 mm out, as flown, "
        "whatever the description; this hand's pads end 151 mm out, 20 mm past it; the planner plans arm_tool_frame "
        "(the arm's description), the same frame; wrist lens (+0.0, +65.0, -105.0) mm from the tool frame (the "
        "camera config (no robot calibration of wrist_camera here))"
    ]
    # no chain (no description heard): nothing is said, and nothing stops
    approach.set_chain_root("arm_mount", None)
    capsys.readouterr()
    assert cli_common.hand_or_exit("[press_demo]", "camera_oak_wrist.yaml") == [] and capsys.readouterr().out == ""


def test_a_run_warns_of_a_planner_frame_it_could_not_read_and_of_a_robot_config_that_is_not_its_arms(
    tmp_path, monkeypatch
):
    """Two more things a run cannot see for itself and says aloud: a description with no tool frame link
    by its usual name while the planner's own config was not read (the body check off: the planner's
    frame is only TAKEN for the tool frame), and a robot config mounted at /config/robot whose kinematics
    reference is another description's (flatbot's config without the puck on the arm described with it:
    its camera mounts, its chair and its user would be another robot's)."""
    import xml.etree.ElementTree as ET

    approach.set_body(False)
    plain, puck = both_chains(tmp_path)
    monkeypatch.setenv(d405.ROBOT_CAMERAS_ENV, "")
    tree = ET.parse(puck_urdf(tmp_path))
    for j in tree.getroot().findall("joint"):
        if j.get("name") == "arm_tool_frame_joint":
            j.find("child").set("link", "arm_tcp")
    renamed = tmp_path / "tcp.urdf"
    tree.write(renamed)
    approach.set_chain_root("arm_mount", str(renamed))
    lines = cli_common.hand_or_exit("[x]", "camera_oak_wrist.yaml")
    assert len(lines) == 2 and "the planner's frame taken for the tool frame" in lines[0]
    assert lines[1].startswith("[x] WARNING: the arm's description has no link tool_frame or arm_tool_frame")
    assert "its frame is TAKEN for box opening's tool frame" in lines[1]
    approach.set_chain_root("arm_mount", puck_urdf(tmp_path))
    # the right robot's config, a mount presses have flown on: one line
    monkeypatch.setenv(d405.ROBOT_CAMERAS_ENV, _robot_config(tmp_path, MOUNT_1109, kinematics=_kinematics_of(puck)))
    assert len(cli_common.hand_or_exit("[x]", "camera_oak_wrist.yaml")) == 1
    # another robot's config: the description without the puck's kinematics, on the arm described with it
    other = _robot_config(tmp_path, MOUNT_FLATBOT, robot="flatbot_fwd8", kinematics=_kinematics_of(plain))
    monkeypatch.setenv(d405.ROBOT_CAMERAS_ENV, other)
    lines = cli_common.hand_or_exit("[x]", "camera_oak_wrist.yaml")
    config_dir = Path(other).parent.parent
    assert len(lines) == 2 and "(robot flatbot_fwd8 by its mount.yaml)" in lines[0]
    assert (
        "WARNING: the robot config %s (robot flatbot_fwd8 by its mount.yaml) is NOT this arm's" % config_dir in lines[1]
    )
    assert "joint arm_robotiq_85_base_joint joins arm_sensor_puck_link -> arm_robotiq_85_base_link" in lines[1]
    assert cli_common.config_mismatch(puck, config_dir) is not None
    assert cli_common.config_mismatch(plain, config_dir) is None
    assert cli_common.config_mismatch(puck, tmp_path / "no_such_config") is None  # nothing to compare: nothing said


def test_a_mount_on_a_link_that_is_turned_on_the_flange_is_placed_with_its_turn(tmp_path):
    """The robot may mount the camera on a bracket link that is itself turned and set aside on the flange.
    The lens and the axes are taken through that link's own pose (its turn too), as the arm's URDF has it:
    the same place and the same view as the mount written straight on the flange."""
    import xml.etree.ElementTree as ET

    from box_opening.runtime.kinematics import ArmChain, _rpy

    cfg = d405.camera_config("camera_oak_wrist.yaml")
    tree = ET.parse(puck_urdf(tmp_path))
    root = tree.getroot()
    # a bracket turned on the flange, and the camera's plate turned on the bracket: two fixed joints, neither square
    poses = {
        "arm_bracket_link": ("arm_end_effector_link", (0.01, 0.02, 0.03), (0.3, -0.2, 0.5)),
        "arm_camera_plate_link": ("arm_bracket_link", (-0.02, 0.04, 0.01), (-0.4, 0.1, 0.25)),
    }
    for link, (parent, xyz, rpy) in poses.items():
        ET.SubElement(root, "link", name=link)
        j = ET.SubElement(root, "joint", name=link + "_joint", type="fixed")
        ET.SubElement(j, "origin", xyz="%g %g %g" % xyz, rpy="%g %g %g" % rpy)
        ET.SubElement(j, "parent", link=parent)
        ET.SubElement(j, "child", link=link)
    path = tmp_path / "bracket.urdf"
    tree.write(path)
    chain = ArmChain(root="arm_mount", path=str(path))
    # the plate's pose in the flange, by hand: the bracket's, then the plate's on it
    Rb, tb = _rpy(*poses["arm_bracket_link"][2]), np.array(poses["arm_bracket_link"][1])
    Rp, tp = _rpy(*poses["arm_camera_plate_link"][2]), np.array(poses["arm_camera_plate_link"][1])
    R_plate, t_plate = Rb @ Rp, tb + Rb @ tp
    (R, t), why = chain.fixed_in_flange("arm_camera_plate_link")
    assert why is None and np.allclose(R, R_plate, atol=1e-12) and np.allclose(t, t_plate, atol=1e-12)
    # gen_1_5's mount, written on the plate: the same lens and the same axes as written on the flange
    xyz, quat = MOUNT_GEN15
    R_mount = d405.quat_to_mat(*quat)
    on_plate = (
        [float(v) for v in R_plate.T @ (np.asarray(xyz) - t_plate)],
        [float(v) for v in d405.mat_to_quat_xyzw(R_plate.T @ R_mount)],
    )
    lens, optical, where = d405.wrist_camera_in_hand(
        cfg, chain=chain, path=_robot_config(tmp_path, on_plate, parent="arm_camera_plate_link")
    )
    assert where.endswith(", on arm_camera_plate_link") and lens == pytest.approx(LENS_GEN15, abs=1e-8)
    assert np.allclose(optical, R_mount @ d405.BODY_TO_OPTICAL, atol=1e-8)


def test_a_guarded_press_demo_does_not_leave_rest_on_a_mount_no_press_has_flown_near(tmp_path, monkeypatch, capfd):
    """press_demo --guarded centres its press on the wrist camera's mount. On gen_1_5's calibration it
    stopped once the arm stood over the box (after depth:turn and the look); it stops where it says its
    hand, before the first motion. The default, compliant run only warns and goes on."""
    import sys

    from box_opening.constants import HOME
    from box_opening.tasks import press_demo

    class Client:
        joint_names = ["joint_%d" % i for i in range(1, 8)]
        _base_frame = None

        def joints(self):
            return list(HOME)

    pkg = Path(__file__).resolve().parents[1]
    argv = ["press_demo", "--container", str(pkg / "config" / "containers" / "ankou_pink.yaml")]
    argv += ["--bench-world", str(pkg / "config" / "world_flatbot.yaml")]
    monkeypatch.setenv("BOX_OPENING_STATE", str(tmp_path))
    monkeypatch.setenv(d405.ROBOT_CAMERAS_ENV, _robot_config(tmp_path, MOUNT_GEN15))
    monkeypatch.setattr(cli_common, "init_runtime", lambda execute: (object(), Client()))
    monkeypatch.setattr(cli_common, "_arm_urdf_path", lambda node, frame: puck_urdf(tmp_path))
    called = []
    monkeypatch.setattr(press_demo, "home_first", lambda *a, **k: called.append("home_first") or True)
    monkeypatch.setattr(press_demo.Runner, "run", lambda self, *a, **k: called.append("run") or [])
    monkeypatch.setattr(
        cli_common, "load_body_or_exit", lambda *a, **k: sys.exit("the body's load: the test ends here")
    )
    monkeypatch.setattr(sys, "argv", argv + ["--guarded"])
    with pytest.raises(
        SystemExit, match=r"\[press_demo\] not starting: a guarded run centres its press .*Nothing moved"
    ):
        press_demo.main()
    assert called == [] and "WARNING: the wrist camera's mount is one no press has flown near" in capfd.readouterr().out
    monkeypatch.setattr(sys, "argv", argv)  # the default run: said, warned, and on to the body's load
    with pytest.raises(SystemExit, match="the body's load: the test ends here"):
        press_demo.main()
    out = capfd.readouterr().out
    assert "[press_demo] hand: tool frame 120 mm" in out and "WARNING: the wrist camera's mount is one no press" in out
