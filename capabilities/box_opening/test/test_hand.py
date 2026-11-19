"""The hand, stated once.

How far out of the flange the shut fingertips end is one number, constants.PADS_FROM_FLANGE_M. What is
about where the fingers really are follows from it: what they reach past the point the heights go by
(the press's contact band), the fingertips' spheres (config/robot_body.yaml, written back from the
pads' ends, from the flange) and how near the wrist's lens the hand is seen. One edit of the number,
when a ruler has been held to the hand, moves all of them; and on either description of the arm (the
gripper bolted on the flange, or Kinova's sensor puck under it: test_tool_frame) they stand at the
same place on the arm.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import SENSOR_PUCK_M as PUCK_M
from conftest import arm_poses, both_chains, flange_pose, flatbot_urdf, puck_urdf

from box_opening import constants
from box_opening.constants import FINGERS_PAST_MODEL_M, HEIGHTS_POINT_FROM_FLANGE_M, PADS_FROM_FLANGE_M
from box_opening.perception import clear_table, wrist_finder
from box_opening.runtime import approach, body
from box_opening.runtime.kinematics import ArmChain

pytestmark = pytest.mark.usefixtures("approach_state")

PINK = "capabilities/box_opening/config/containers/ankou_pink.yaml"
# how far out of the flange the wrist camera's lens is: flatbot's fitted mount (rammp-deployments #128, box
# opening's own robot config until 2026-11-17) and gen_1_5's calibration of 2026-11-18
LENS_OUT_FLATBOT, LENS_OUT_GEN15 = 0.015, 0.022506
# fingertips_151mm as it was written on 2026-11-17, in the gripper's base link, by its numbers
TIPS_1117 = [[-0.040, 0.0, 0.140], [-0.020, 0.0, 0.140], [0.0, 0.0, 0.140], [0.020, 0.0, 0.140], [0.040, 0.0, 0.140]]


def _with_pads_at(value):
    """constants.py run with PADS_FROM_FLANGE_M written as `value`: what one edit of that line gives."""
    src = Path(constants.__file__).read_text()
    line = "PADS_FROM_FLANGE_M = %r\n" % PADS_FROM_FLANGE_M
    assert src.count(line) == 1
    ns = {"__name__": "box_opening.constants_edited", "__file__": constants.__file__}
    exec(compile(src.replace(line, "PADS_FROM_FLANGE_M = %r\n" % value), constants.__file__, "exec"), ns)  # noqa: S102
    return SimpleNamespace(**ns)


def test_one_edit_of_the_pads_reach_moves_the_band_the_cutoff_and_the_fingertips_spheres(tmp_path, monkeypatch):
    """A ruler reading goes into constants.PADS_FROM_FLANGE_M and nowhere else. Today's 0.151 gives what
    2026-11-17 set by hand (20 mm in the press band, the hand's cutoff at 0.17 m, the spheres at z 0.140);
    7 mm more gives 7 mm more in each, the point the heights are flown by staying where it is."""
    assert PADS_FROM_FLANGE_M == 0.151 and HEIGHTS_POINT_FROM_FLANGE_M == 0.131
    assert FINGERS_PAST_MODEL_M == 0.020 and constants.WRIST_NEAREST_M == 0.17
    assert clear_table.WRIST_NEAREST_M is constants.WRIST_NEAREST_M is wrist_finder.WRIST_NEAREST_M
    longer = _with_pads_at(0.158)
    assert longer.FINGERS_PAST_MODEL_M == pytest.approx(0.027, abs=1e-12)
    assert longer.WRIST_NEAREST_M == pytest.approx(0.177, abs=1e-12)
    assert longer.HEIGHTS_POINT_FROM_FLANGE_M == 0.131 and longer.TCP_OFFSET_M == constants.TCP_OFFSET_M
    # the spheres: written back from the pads' ends (z_from: pads), read with the constant as it is when asked
    path = body.robot_body_path()
    tips = next(g for g in body.load_hand(path) if g.name == "fingertips_151mm")
    assert tips.frame == "arm_end_effector_link" and np.array_equal(tips.centres, np.asarray(TIPS_1117))
    _name, c, h = next(cv for cv in tips.covers if cv[0] == "longer_tips")
    assert c[2] + h[2] == pytest.approx(PADS_FROM_FLANGE_M, abs=1e-12) and c[2] == 0.1405
    monkeypatch.setattr(constants, "PADS_FROM_FLANGE_M", 0.158)
    moved = next(g for g in body.load_hand(path) if g.name == "fingertips_151mm")
    assert np.allclose(moved.centres, np.asarray(TIPS_1117) + [0.0, 0.0, 0.007], atol=1e-12)
    _name, c, h = next(cv for cv in moved.covers if cv[0] == "longer_tips")
    assert c[2] + h[2] == pytest.approx(0.158, abs=1e-12)
    stock = next(g for g in body.load_hand(path) if g.name == "fingers_131mm")
    assert stock.centres[:, 2].tolist() == [0.112] * 4  # the stock gripper's own: not the pads' to move
    # a word the load does not know stops it
    bad = tmp_path / "robot_body.yaml"
    bad.write_text(
        "hand_spheres:\n  tips:\n    link: arm_robotiq_85_base_link\n    z_from: flange_face\n"
        "    covers: [{name: pads, centre: [0, 0, 0.12], half: [0.01, 0.01, 0.011]}]\n"
        "    spheres: [{center: [0, 0, 0.12], radius: 0.03}]\n"
    )
    with pytest.raises(body.BodyModelError, match="hand group tips is malformed .*z_from"):
        body.load_hand(bad)


def test_the_presss_contact_band_is_the_same_on_both_descriptions_and_follows_the_pads_reach(
    ctx, tmp_path, monkeypatch
):
    """The band a press's contact may come in is counted from the point the heights go by, 131 mm out of
    the flange on either description, to where this hand's fingers end: the 20 mm it has had since
    2026-11-17 (the presses of 11-12..17 met their buttons 40-45 mm over the wrist's top by that point:
    inside it, as they must be on gen_1_5, where the arm and the hand are the same)."""
    from box_opening.models.container import ContainerModel
    from box_opening.tasks import depth_press as dp
    from box_opening.tasks import press_contact as pc
    from box_opening.tasks import reach_press as rp

    model = ContainerModel.load(PINK)

    def band(chain):
        monkeypatch.setattr(approach, "_chain", lambda: chain)
        stroke = SimpleNamespace(pad0=[0.8, -0.25, 0.40], end_z=0.28, quat=rp.reach_quat(model, (0.8, -0.25), 0.0))
        watch = dp.watch_press(ctx, [SimpleNamespace(stop_when=None)], stroke, 0.316)
        return watch.early_m, watch.late_m

    plain, puck = both_chains(tmp_path)
    assert band(plain) == band(puck)
    assert band(puck) == pytest.approx((pc.CONTACT_EARLY_M + 0.020, max(pc.CONTACT_LATE_M - 0.020, 0.0)))
    assert 0.045 < band(puck)[0]
    monkeypatch.setattr(dp, "FINGERS_PAST_MODEL_M", 0.027)  # the ruler says 158 mm
    assert band(puck) == pytest.approx((pc.CONTACT_EARLY_M + 0.027, max(pc.CONTACT_LATE_M - 0.027, 0.0)))


def test_the_fingertips_spheres_stay_on_the_tips_when_the_description_moves_the_gripper(tmp_path):
    """config/robot_body.yaml's fingertips_151mm is written from the flange (`frame`), where the tips'
    reach is known, and rides the gripper's base link. On the description without the puck the two are
    one frame: the spheres as written on 2026-11-17, to the last bit. With the puck the gripper's link
    is 14 mm out and the spheres stand 14 mm nearer it: on the tips, not 14 mm past them. The stock
    fingers' group is the gripper's own and goes out with it."""
    from test_body import STOCK_HAND, _model_files

    path = _model_files(tmp_path, [("far", (-1.0, -1.0, -1.0), (0.1, 0.1, 0.1))], robot=STOCK_HAND)
    m = body.load(path)  # the shipped config/robot_body.yaml
    tips = next(g for g in m.hand if g.name == "fingertips_151mm")
    stock = next(g for g in m.hand if g.name == "fingers_131mm")
    assert tips.link == "arm_robotiq_85_base_link" and tips.frame == "arm_end_effector_link" and stock.frame is None
    plain, puck = both_chains(tmp_path)
    on_plain = m.check_for(plain, flatbot_urdf(), "arm_mount")
    on_puck = m.check_for(puck, puck_urdf(tmp_path), "arm_mount")
    for check in (on_plain, on_puck):
        assert check.hand_added == ["fingers_131mm", "fingertips_151mm", "wrist_camera"]
        assert len(check.spheres["arm_robotiq_85_base_link"][1]) == 19 + 4 + 5
    cs0, cs1 = on_plain.spheres["arm_robotiq_85_base_link"][0], on_puck.spheres["arm_robotiq_85_base_link"][0]
    assert np.array_equal(cs0[-5:], np.asarray(TIPS_1117)) and np.array_equal(cs0[19:23], stock.centres)
    assert np.allclose(cs1[-5:], np.asarray(TIPS_1117) - [0.0, 0.0, PUCK_M], atol=1e-12)  # from the flange
    assert np.array_equal(cs1[19:23], stock.centres)  # the gripper's own: where the gripper is
    # the same check as 2026-11-17's group gave on the description the runs were flown on: every sphere, every link
    as_written = [
        body.HandGroup(g.name, g.link, np.asarray(TIPS_1117) if g is tips else g.centres, g.radii, g.covers)
        for g in m.hand
    ]
    was = body.BodyModel(**{**m.__dict__, "hand": as_written}).check_for(plain, flatbot_urdf(), "arm_mount")
    assert list(was.spheres) == list(on_plain.spheres) and list(was.paths) == list(on_plain.paths)
    for link in was.spheres:
        assert np.array_equal(was.spheres[link][0], on_plain.spheres[link][0])
        assert np.array_equal(was.spheres[link][1], on_plain.spheres[link][1])
    # and the tips' ends, PADS_FROM_FLANGE_M out of the flange and as wide as the wrist's frames have them,
    # are as deep inside the hand's spheres on both: by the flange's own pose
    ends = [(0.0, 0.0, PADS_FROM_FLANGE_M), (0.026, 0.0, PADS_FROM_FLANGE_M), (-0.022, 0.0, PADS_FROM_FLANGE_M)]
    for q in arm_poses()[:6]:
        depth = []
        for chain, check in ((plain, on_plain), (puck, on_puck)):
            Rf, f = flange_pose(chain, q)
            C, r, _links = check.spheres_at(q)
            depth.append([float((np.linalg.norm(C - (f + Rf @ np.asarray(p)), axis=1) - r).min()) for p in ends])
        assert max(depth[0]) <= -0.0099 and all(b <= a + 1e-9 for a, b in zip(depth[0], depth[1], strict=True))


def test_a_robot_file_that_carries_the_fingertips_spheres_where_the_puck_has_them_is_not_doubled(tmp_path):
    """Whether the planner's robot file already covers a group is judged in the frame the group is
    written in. On the description with the puck a robot file that holds the fingertips' spheres on the
    gripper's base, 14 mm nearer it than the group's numbers (the same place on the arm), covers them."""
    from test_body import STOCK_HAND, _model_files

    lines = ["      - {center: %s, radius: 0.0275}" % [c[0], c[1], round(c[2] - PUCK_M, 6)] for c in TIPS_1117]
    robot = STOCK_HAND.replace(
        "      arm_robotiq_85_left_finger_link:", "\n".join(lines) + "\n      arm_robotiq_85_left_finger_link:"
    )
    path = _model_files(tmp_path, [("far", (-1.0, -1.0, -1.0), (0.1, 0.1, 0.1))], robot=robot)
    check = body.load(path).check_for(
        ArmChain(root="arm_mount", path=puck_urdf(tmp_path)), puck_urdf(tmp_path), "arm_mount"
    )
    assert "fingertips_151mm" in check.hand_skipped and "fingertips_151mm" not in check.hand_added
    # the same robot file on the description without the puck holds them 14 mm short of the tips: added
    check = body.load(path).check_for(ArmChain(root="arm_mount", path=flatbot_urdf()), flatbot_urdf(), "arm_mount")
    assert "fingertips_151mm" in check.hand_added


def test_a_hand_group_given_in_a_frame_that_moves_against_its_link_stops_the_load(tmp_path):
    from test_body import STOCK_HAND, _model_files

    path = _model_files(tmp_path, [("b", (1, 0, 0), (0.1, 0.1, 0.1))], robot=STOCK_HAND)
    bad = tmp_path / "robot_body.yaml"
    bad.write_text(
        "hand_spheres:\n  tips:\n    link: arm_robotiq_85_base_link\n    frame: arm_forearm_link\n"
        "    covers: [{name: pads, centre: [0, 0, 0.12], half: [0.01, 0.01, 0.011]}]\n"
        "    spheres: [{center: [0, 0, 0.12], radius: 0.03}]\n"
    )
    chain = ArmChain(root="arm_mount", path=flatbot_urdf())
    with pytest.raises(body.BodyModelError, match="hand group tips is given in arm_forearm_link"):
        body.load(path, robot_body=bad).check_for(chain, flatbot_urdf(), "arm_mount")


def test_nearer_the_wrists_lens_than_its_cutoff_is_the_hand_at_either_lens_mount():
    """Depth nearer the lens than WRIST_NEAREST_M is taken for the hand itself. The fingertips end
    PADS_FROM_FLANGE_M out of the flange and the lens stands 15 mm (flatbot's fitted mount) or 22.5 mm
    (gen_1_5's calibration) out of it: 136 and 128.5 mm of depth. The cutoff keeps 34 mm over the nearer
    mount whatever the reach is set to, and 15 mm or more over fingers at the long end of what the
    recorded runs allow (161 mm) even before the constant says so."""
    for pads in (0.151, 0.158):
        k = _with_pads_at(pads)
        assert k.WRIST_NEAREST_M - (pads - LENS_OUT_FLATBOT) == pytest.approx(0.034, abs=1e-9)
        assert k.WRIST_NEAREST_M - (pads - LENS_OUT_GEN15) == pytest.approx(0.0415, abs=1e-4)
    for lens_out in (LENS_OUT_FLATBOT, LENS_OUT_GEN15):
        assert constants.WRIST_NEAREST_M - (0.161 - lens_out) >= 0.015
