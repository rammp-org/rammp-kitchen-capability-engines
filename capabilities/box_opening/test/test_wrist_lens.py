"""The wrist lens from the tool frame is the robot's own calibration, not a copy kept here.

config/camera_oak_wrist.yaml kept it by hand, [0, 0.065, -0.105]: the team's mount of the camera on the
flange less a tool frame 0.120 out. A copy for one mount and one tool frame is wrong on the next of
either (#105 wrote one in the wrong frame, #106 took it back; on gen_1_5 the copy stood 7 mm sideways of
the 2026-11-18 calibration and 6.5 mm along the tool from it on the sensor puck's tool frame). With the
robot's config mounted (/config/robot) the lens is read from its transforms/cameras.yaml and placed on
the tool frame of the description the run is on.
"""

from pathlib import Path

import numpy as np
import pytest
from conftest import arm_poses, both_chains, flange_pose, flatbot_urdf, puck_urdf

from box_opening.perception import d405
from box_opening.runtime import approach

pytestmark = pytest.mark.usefixtures("approach_state")

# the wrist camera's mount on the flange, as each robot config's transforms/cameras.yaml has it: flatbot's
# fitted one (rammp-deployments #128, box opening's own robot config until 2026-11-17) and gen_1_5's
# calibration of 2026-11-18
MOUNT_FLATBOT = [0.0, 0.065, 0.015]
MOUNT_GEN15 = [-0.005351, 0.06953, 0.022506]


def _cameras_yaml(tmp_path, xyz, parent="arm_end_effector_link", name="cameras.yaml"):
    f = Path(tmp_path) / name
    f.write_text(
        "mounts:\n  ros__parameters:\n    transforms: [scene_camera, wrist_camera]\n"
        "    scene_camera:\n      parent: arm_mount\n      child: scene_camera\n"
        "      xyz: [0.151329, -0.571954, 0.391385]\n      quat_xyzw: [0.0, 0.0, 0.0, 1.0]\n"
        "    wrist_camera:\n      parent: %s\n      child: wrist_camera\n      xyz: %s\n"
        "      quat_xyzw: [0.5062226, 0.506811, 0.4947239, -0.4920666]\n" % (parent, list(xyz))
    )
    return str(f)


def test_the_wrist_lens_is_the_robots_calibration_on_the_runs_own_tool_frame(tmp_path):
    """camera_oak_wrist.yaml kept the lens from the tool frame by hand: [0, 0.065, -0.105], the team's
    mount less a tool frame 0.120 out. With the robot's config mounted the lens is read from its
    transforms/cameras.yaml and put on the tool frame of the description the run is on: the same numbers
    on flatbot's config as flown; on gen_1_5 its 2026-11-18 calibration on the puck's tool frame."""
    cfg = d405.camera_config("camera_oak_wrist.yaml")
    assert cfg["robot_mount"] == "wrist_camera" and cfg["wrist_lens_in_hand_m"] == [0.0, 0.065, -0.105]
    plain, puck = both_chains(tmp_path)
    flatbot = _cameras_yaml(tmp_path, MOUNT_FLATBOT, name="flatbot.yaml")
    gen15 = _cameras_yaml(tmp_path, MOUNT_GEN15, name="gen15.yaml")
    lens, where = d405.wrist_lens_in_hand(cfg, chain=plain, path=flatbot)
    assert lens == [0.0, 0.065, -0.105] == cfg["wrist_lens_in_hand_m"] and "robot's calibration" in where
    lens, where = d405.wrist_lens_in_hand(cfg, chain=puck, path=gen15)
    assert lens == [-0.005351, 0.06953, -0.111494] and "134 mm out of the flange" in where
    # it is where TF has the lens: the flange's pose, the mount, and the chain's own tool frame
    q = arm_poses()[0]
    Rf, f = flange_pose(puck, q)
    R, t = puck.fk(q)
    assert np.allclose(np.asarray(R).T @ (f + Rf @ np.asarray(MOUNT_GEN15) - t), lens, atol=1e-9)
    # the copy kept by hand, on gen_1_5: 7 mm sideways of it, over the 5 mm the depth press centres within
    from box_opening.tasks import depth_press as dp

    copy = cfg["wrist_lens_in_hand_m"]
    aside = float(np.hypot(lens[0] - copy[0], lens[1] - copy[1]))
    assert aside == pytest.approx(0.0070, abs=1e-4) and aside > dp.MOUNT_XY_TOL_M


def test_the_cameras_own_figure_stands_in_where_no_calibration_can_be_read(tmp_path, capsys):
    cfg = d405.camera_config("camera_oak_wrist.yaml")
    plain, _puck = both_chains(tmp_path)
    copy = [0.0, 0.065, -0.105]
    # no robot config here (a test, an offline tool), or reading it turned off
    assert d405.wrist_lens_in_hand(cfg, chain=plain, path=str(tmp_path / "nowhere.yaml")) == (
        copy,
        "the camera config (no robot calibration of wrist_camera here)",
    )
    gen15 = _cameras_yaml(tmp_path, MOUNT_GEN15)
    assert d405.wrist_lens_in_hand(cfg, chain=plain, env={d405.ROBOT_CAMERAS_ENV: ""})[0] == copy
    assert d405.wrist_lens_in_hand(cfg, chain=plain, env={d405.ROBOT_CAMERAS_ENV: gen15})[0] == [
        -0.005351,
        0.06953,
        -0.097494,
    ]
    # a mount that is not on the flange is no lens from the tool frame; nor is one with no chain to place it by
    other = _cameras_yaml(tmp_path, MOUNT_GEN15, parent="arm_bracelet_link", name="other.yaml")
    assert d405.wrist_lens_in_hand(cfg, chain=plain, path=other)[0] == copy
    approach.set_chain_root("arm_mount", None)  # no description heard: no chain
    lens, where = d405.wrist_lens_in_hand(cfg, path=gen15)
    assert lens == copy and "no arm chain" in where
    # a file that is no mounts file: the copy, never a stop, and said
    junk = tmp_path / "junk.yaml"
    junk.write_text("mounts: {ros__parameters: {wrist_camera: {parent: arm_end_effector_link, xyz: [1, 2]}}}\n")
    capsys.readouterr()
    assert d405.wrist_lens_in_hand(cfg, chain=plain, path=str(junk))[0] == copy
    assert "WARNING: the robot's camera mounts %s give no usable wrist_camera" % junk in capsys.readouterr().out
    # the bench's D405 names no robot mount: its own file holds its mount, and its figure stands
    bench = d405.camera_config("camera_d405_wrist.yaml")
    assert d405.wrist_lens_in_hand(bench, chain=plain, path=gen15) == ([0.0, 0.075, -0.125], "the camera config")


def test_a_run_on_gen_1_5_takes_its_lens_from_the_mounted_robot_config(tmp_path, monkeypatch):
    """The grabber's lens (what the depth press checks TF against and the lid's table map takes the hand
    out by) and the survey's camera follow the robot's calibration and the run's own chain."""
    from box_opening.tasks import survey as sv

    gen15 = _cameras_yaml(tmp_path, MOUNT_GEN15)
    monkeypatch.setenv(d405.ROBOT_CAMERAS_ENV, gen15)
    approach.set_body(False)
    approach.set_chain_root("arm_mount", puck_urdf(tmp_path))
    g = d405.D405Grabber.__new__(d405.D405Grabber)
    g._cfg = d405.camera_config("camera_oak_wrist.yaml")
    assert g.wrist_lens_in_hand_m == [-0.005351, 0.06953, -0.111494]
    assert np.allclose(sv.camera_of(None, "camera_oak_wrist.yaml").lens, [-0.005351, 0.06953, -0.111494])
    # the same robot config read on the description without the puck would be 14 mm along the tool from it
    approach.set_chain_root("arm_mount", flatbot_urdf())
    assert g.wrist_lens_in_hand_m == [-0.005351, 0.06953, -0.097494]
    # and with no robot config, as every test and offline tool before: the camera config's
    monkeypatch.setenv(d405.ROBOT_CAMERAS_ENV, "")
    assert g.wrist_lens_in_hand_m == [0.0, 0.065, -0.105]
    assert np.allclose(sv.camera_of(None, "camera_oak_wrist.yaml").lens, sv.LENS_IN_TOOL_M)


def test_a_run_says_where_it_has_its_hand_before_it_moves(tmp_path, monkeypatch, capsys):
    """One line at the start: the tool frame and where it was read, the pads the heights go by and this
    hand's real ones, the lens and where it is from. A run on a description or a robot config it was not
    flown on shows there before anything moves."""
    from box_opening.tasks import cli_common

    approach.set_body(False)
    monkeypatch.setenv(d405.ROBOT_CAMERAS_ENV, _cameras_yaml(tmp_path, MOUNT_GEN15))
    approach.set_chain_root("arm_mount", puck_urdf(tmp_path))
    capsys.readouterr()
    cli_common.log_hand("[press_demo]", "camera_oak_wrist.yaml")
    assert capsys.readouterr().out.strip() == (
        "[press_demo] hand: tool frame 134 mm out of the flange (the arm's description), the pads the heights go by "
        "145 mm out, this hand's 151 mm: 6 mm past them; wrist lens (-5.4, +69.5, -111.5) mm from the tool frame "
        "(the robot's calibration of wrist_camera, the tool frame 134 mm out of the flange)"
    )
    monkeypatch.setenv(d405.ROBOT_CAMERAS_ENV, "")
    approach.set_chain_root("arm_mount", flatbot_urdf())
    cli_common.log_hand("[press_demo]", "camera_oak_wrist.yaml")
    assert capsys.readouterr().out.strip() == (
        "[press_demo] hand: tool frame 120 mm out of the flange (the arm's description), the pads the heights go by "
        "131 mm out, this hand's 151 mm: 20 mm past them; wrist lens (+0.0, +65.0, -105.0) mm from the tool frame "
        "(the camera config (no robot calibration of wrist_camera here))"
    )
    # no chain (no description heard): nothing is said, and nothing stops
    approach.set_chain_root("arm_mount", None)
    cli_common.log_hand("[press_demo]", "camera_oak_wrist.yaml")
    assert capsys.readouterr().out == ""
