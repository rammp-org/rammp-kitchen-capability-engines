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
        assert lens == COPY and where == "wrist_camera in %s (robot flatbot_fwd8), on arm_end_effector_link" % flatbot
        lens, where = d405.wrist_lens_in_hand(cfg, chain=chain, path=gen15)
        assert lens == LENS_GEN15 and where == "wrist_camera in %s (robot gen_1_5), on arm_end_effector_link" % gen15
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
    assert g.wrist_lens_in_hand_m == LENS_GEN15 and g.wrist_lens_reference_m == COPY
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


def test_the_mounts_on_record_against_the_one_figure_no_calibration_made():
    """The 2026-10-22 fit is the only figure for this lens that did not come out of a calibration
    session. The calibration of 2026-11-09 stands 11.6 mm aside of it and pressed a centimetre beside
    the button; gen_1_5's of 2026-11-18 stands 7.0 mm aside; the two are 5.6 and 8.3 mm apart in x and
    y. The tolerance sits between: over what two calibrations differ by, under the one known bad."""
    cfg = d405.camera_config("camera_oak_wrist.yaml")
    off = {}
    for name, (xyz, _quat) in (("fit", MOUNT_FLATBOT), ("1109", MOUNT_1109), ("1118", MOUNT_GEN15)):
        off[name] = d405.lens_off_reference(cfg, [xyz[0], xyz[1], xyz[2] - 0.120])
    assert off["fit"] == pytest.approx((0.0, 0.0), abs=1e-12)
    assert off["1109"] == pytest.approx((0.0116, 0.0075), abs=2e-4) and off["1118"] == pytest.approx(
        (0.0070, 0.0075), abs=2e-4
    )
    assert off["1118"][0] < d405.LENS_REF_XY_TOL_M < off["1109"][0]
    assert max(abs(a - b) for a, b in zip(MOUNT_1109[0][:2], MOUNT_GEN15[0][:2], strict=True)) < d405.LENS_REF_XY_TOL_M
    assert off["1118"][1] < d405.LENS_REF_Z_TOL_M
    assert d405.lens_off_reference({}, COPY) is None and d405.lens_off_reference(cfg, None) is None


@pytest.mark.usefixtures("home_20261021")
def test_the_depth_press_does_not_centre_on_a_mount_a_centimetre_from_the_fit_though_tf_and_the_config_agree(
    ctx, monkeypatch, tmp_path
):
    """TF is published from the robot's cameras.yaml, so TF against that file (the grabber's
    wrist_lens_in_hand_m) can only catch a TF tree and a config of two robots. What is independent is
    the grabber's reference, the fit: the 11-09 calibration, 11.6 mm aside of it, is not centred on;
    gen_1_5's, 7.0 mm aside, is."""
    from test_depth_press import _press, _seen

    real = dp.press_down_from_here

    def with_reference(node, ctx_, cfg, runner, watcher, *a, **k):
        watcher.grab.wrist_lens_reference_m = list(COPY)
        return real(node, ctx_, cfg, runner, watcher, *a, **k)

    monkeypatch.setattr(dp, "press_down_from_here", with_reference)
    tf = (MOUNT_1109[0][0], MOUNT_1109[0][1], MOUNT_1109[0][2] - 0.120)
    result, events, arm, _l, _s = _press(ctx, monkeypatch, tmp_path, [_seen((0.0, 0.0), lens=tf)], lens=tf, measured=tf)
    assert result.free and "12 mm aside" in result.why and "2026-10-22 depth-press fit" in result.why
    assert events == ["depth:turn", "look:depth-look"] and arm.lines == [] and dp.exit_code(result) == 1
    tf = tuple(LENS_GEN15)
    seen = _seen((0.02, 0.03), lens=tf)
    result, events, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, [seen, seen, seen.after], lens=tf, measured=tf)
    assert dp.exit_code(result) == 0 and events[:3] == ["depth:turn", "look:depth-look", "depth:centre"]


# ---- what a run says of its hand ----------------------------------------------------------------------


def test_a_run_says_where_it_has_its_hand_before_it_moves(tmp_path, monkeypatch, capsys):
    """One line at the start: the tool frame and the point the heights go by (as flown, on either
    description), where the pads really end, the link the planner plans and the step to it, the lens
    with the file and the robot it came from and how far that is from the fit."""
    approach.set_body(False)
    gen15 = _robot_config(tmp_path, MOUNT_GEN15)
    monkeypatch.setenv(d405.ROBOT_CAMERAS_ENV, gen15)
    approach.set_chain_root("arm_mount", puck_urdf(tmp_path))
    capsys.readouterr()
    lines = cli_common.hand_or_exit("[press_demo]", "camera_oak_wrist.yaml")
    assert (
        capsys.readouterr().out.splitlines()
        == lines
        == [
            "[press_demo] hand: tool frame 120 mm out of the flange, the point the heights go by 131 mm out, as "
            "flown, whatever the description; this hand's pads end 151 mm out, 20 mm past it; the planner plans "
            "arm_tool_frame (the arm's description), (+0.0, +0.0, +14.0) mm from the tool frame: a pose goes to it "
            "there; wrist lens (-5.4, +69.5, -97.5) mm from the tool frame (wrist_camera in %s (robot gen_1_5), on "
            "arm_end_effector_link), 7.0 mm aside and 7.5 mm along the tool from the 2026-10-22 fit" % gen15
        ]
    )
    monkeypatch.setenv(d405.ROBOT_CAMERAS_ENV, "")
    approach.set_chain_root("arm_mount", flatbot_urdf())
    assert cli_common.hand_or_exit("[press_demo]", "camera_oak_wrist.yaml") == [
        "[press_demo] hand: tool frame 120 mm out of the flange, the point the heights go by 131 mm out, as flown, "
        "whatever the description; this hand's pads end 151 mm out, 20 mm past it; the planner plans arm_tool_frame "
        "(the arm's description), the same frame; wrist lens (+0.0, +65.0, -105.0) mm from the tool frame (the "
        "camera config (no robot calibration of wrist_camera here))"
    ]
    # no chain (no description heard): nothing is said, and nothing stops
    approach.set_chain_root("arm_mount", None)
    capsys.readouterr()
    assert cli_common.hand_or_exit("[press_demo]", "camera_oak_wrist.yaml") == [] and capsys.readouterr().out == ""


def test_a_run_warns_of_a_mount_far_from_the_fit_and_of_a_robot_config_that_is_not_its_arms(tmp_path, monkeypatch):
    """Two things a run cannot see for itself and says aloud: a calibrated mount further from the fit
    than the depth press centres on (the 11-09 one), and a robot config mounted at /config/robot whose
    kinematics reference is another description's (flatbot's config without the puck on the arm described
    with it: its camera mounts, its chair and its user would be another robot's)."""
    approach.set_body(False)
    plain, puck = both_chains(tmp_path)
    approach.set_chain_root("arm_mount", puck_urdf(tmp_path))
    # the right robot's config, a mount a centimetre from the fit
    monkeypatch.setenv(d405.ROBOT_CAMERAS_ENV, _robot_config(tmp_path, MOUNT_1109, kinematics=_kinematics_of(puck)))
    lines = cli_common.hand_or_exit("[x]", "camera_oak_wrist.yaml")
    assert len(lines) == 2 and "11.6 mm aside and 7.5 mm along the tool from the 2026-10-22 fit" in lines[0]
    assert lines[1].startswith("[x] WARNING: the wrist camera's mount stands further from the 2026-10-22 fit than 9 mm")
    # a good mount, the right robot's config: one line
    monkeypatch.setenv(d405.ROBOT_CAMERAS_ENV, _robot_config(tmp_path, MOUNT_GEN15, kinematics=_kinematics_of(puck)))
    assert len(cli_common.hand_or_exit("[x]", "camera_oak_wrist.yaml")) == 1
    # another robot's config: the description without the puck's kinematics, on the arm described with it
    other = _robot_config(tmp_path, MOUNT_FLATBOT, robot="flatbot_fwd8", kinematics=_kinematics_of(plain))
    monkeypatch.setenv(d405.ROBOT_CAMERAS_ENV, other)
    lines = cli_common.hand_or_exit("[x]", "camera_oak_wrist.yaml")
    assert len(lines) == 2 and "(robot flatbot_fwd8)" in lines[0]
    assert (
        "WARNING: the robot config %s (robot flatbot_fwd8) is NOT this arm's description" % Path(other).parent.parent
        in lines[1]
    )
    assert "joint arm_robotiq_85_base_joint joins arm_sensor_puck_link -> arm_robotiq_85_base_link" in lines[1]
    assert cli_common.config_mismatch(puck, Path(other).parent.parent) is not None
    assert cli_common.config_mismatch(plain, Path(other).parent.parent) is None
    assert cli_common.config_mismatch(puck, tmp_path / "no_such_config") is None  # nothing to compare: nothing said
