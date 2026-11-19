"""The hand on both descriptions of the arm.

The same arm, the same hand, described two ways (test_tool_frame): the gripper bolted on the flange,
and with Kinova's sensor puck under it. The puck was on the arm all along. How far out of the flange
the shut fingertips end is stated once (constants.PADS_FROM_FLANGE_M); the description says where the
tool frame is; what follows from the two (the press's contact band, the fingertips' spheres, how near
the wrist's lens the hand is seen) stands at the same place on the arm either way, and on the
description the runs were flown on nothing moves.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import SENSOR_PUCK_M as PUCK_M
from conftest import arm_poses, both_chains, flange_pose, flatbot_urdf, puck_urdf

from box_opening import constants
from box_opening.constants import FINGERS_PAST_MODEL_M, PADS_FROM_FLANGE_M, TCP_OFFSET_M, TOOL_FROM_FLANGE_STOCK_M
from box_opening.perception import clear_table
from box_opening.runtime import approach, body, fingers
from box_opening.runtime.kinematics import ArmChain

pytestmark = pytest.mark.usefixtures("approach_state")

PINK = "capabilities/box_opening/config/containers/ankou_pink.yaml"
# how far out of the flange the wrist camera's lens is: flatbot's fitted mount (rammp-deployments #128, box
# opening's own robot config until 2026-11-17) and gen_1_5's calibration of 2026-11-18
LENS_OUT_FLATBOT, LENS_OUT_GEN15 = 0.015, 0.022506


def test_the_fingers_end_at_the_same_place_on_the_arm_whichever_description_the_run_is_on(tmp_path):
    """The shut pads end PADS_FROM_FLANGE_M out of the flange. The pads the arm's heights go by stand
    TCP_OFFSET_M past the tool frame, which the description places: 20 mm short of the real ones with
    the gripper on the flange (the figure of 2026-11-17, unchanged), 6 mm short with the puck."""
    plain, puck = both_chains(tmp_path)
    assert FINGERS_PAST_MODEL_M == 0.020 and PADS_FROM_FLANGE_M == 0.151
    assert fingers.past_model_m(plain) == FINGERS_PAST_MODEL_M  # exactly: nothing rounds on the old description
    assert fingers.past_model_m(puck) == pytest.approx(0.006, abs=1e-9)
    for chain in (plain, puck):
        assert chain.tool_from_flange_m + TCP_OFFSET_M + fingers.past_model_m(chain) == pytest.approx(
            PADS_FROM_FLANGE_M
        )
    # a stand-in chain that says nothing of its tool frame is taken for the gripper on the flange
    assert fingers.past_model_m(object()) == fingers.past_model_m(None) == FINGERS_PAST_MODEL_M
    assert fingers.past_model_m(puck, 0.0) == pytest.approx(-PUCK_M, abs=1e-9)


def test_the_presss_contact_band_stands_at_the_same_height_of_the_flange_on_both_descriptions(
    ctx, tmp_path, monkeypatch
):
    """The band a press's contact may come in is counted from the pads the watch measures. With the puck
    in the description those are 14 mm nearer the real fingertips, so the band's ends move down by as
    much: the flange stands as high over the cameras' top at the earliest and at the latest contact the
    press takes as it did on the description the 2026-11-12..17 presses set the band on."""
    from box_opening.models.container import ContainerModel
    from box_opening.tasks import depth_press as dp
    from box_opening.tasks import press_contact as pc
    from box_opening.tasks import reach_press as rp

    model = ContainerModel.load(PINK)
    got = {}
    for name, chain in zip(("plain", "puck"), both_chains(tmp_path), strict=True):
        monkeypatch.setattr(approach, "_chain", lambda chain=chain: chain)
        stroke = SimpleNamespace(pad0=[0.8, -0.25, 0.40], end_z=0.28, quat=rp.reach_quat(model, (0.8, -0.25), 0.0))
        watch = dp.watch_press(ctx, [SimpleNamespace(stop_when=None)], stroke, 0.316)
        got[name] = (watch.early_m, watch.late_m, chain.tool_from_flange_m + TCP_OFFSET_M)
    assert got["plain"][:2] == pytest.approx(
        (pc.CONTACT_EARLY_M + 0.020, max(pc.CONTACT_LATE_M - 0.020, 0.0))
    )  # as it was
    assert got["puck"][0] == pytest.approx(pc.CONTACT_EARLY_M + 0.006) and got["puck"][1] == pytest.approx(
        max(pc.CONTACT_LATE_M - 0.006, 0.0)
    )
    # the flange over the top at the earliest contact taken: the same on both
    assert got["plain"][2] + got["plain"][0] == pytest.approx(got["puck"][2] + got["puck"][0])
    assert got["plain"][2] + got["plain"][0] == pytest.approx(PADS_FROM_FLANGE_M + pc.CONTACT_EARLY_M)
    # 2026-11-12..17's presses met their buttons 40-45 mm over the wrist's top by the pads modelled 0.131 out
    # of the flange: 26-31 mm by pads modelled 0.145 out. Both are inside the band of their description
    assert 0.045 < got["plain"][0] and 0.031 < got["puck"][0] < got["plain"][0]


def test_the_fingertips_spheres_stay_on_the_tips_when_the_description_moves_the_gripper(tmp_path):
    """config/robot_body.yaml's fingertips_151mm is written from the flange (`frame`), where the tips'
    reach is known, and rides the gripper's base link. On the description without the puck the two are
    one frame: the spheres as written, to the last bit. With the puck the gripper's link is 14 mm out and
    the spheres stand 14 mm nearer it: on the tips, not 14 mm past them. The stock fingers' group is the
    gripper's own and goes out with it."""
    from test_body import STOCK_HAND, _model_files

    path = _model_files(tmp_path, [("far", (-1.0, -1.0, -1.0), (0.1, 0.1, 0.1))], robot=STOCK_HAND)
    m = body.load(path)  # the shipped config/robot_body.yaml
    tips = next(g for g in m.hand if g.name == "fingertips_151mm")
    stock = next(g for g in m.hand if g.name == "fingers_131mm")
    assert tips.link == "arm_robotiq_85_base_link" and tips.frame == "arm_end_effector_link" and stock.frame is None
    # the one statement of the reach: the far end of what the group stands for IS the constant
    _name, c, h = next(cv for cv in tips.covers if cv[0] == "longer_tips")
    assert c[2] + h[2] == pytest.approx(PADS_FROM_FLANGE_M, abs=1e-12)
    plain, puck = both_chains(tmp_path)
    on_plain = m.check_for(plain, flatbot_urdf(), "arm_mount")
    on_puck = m.check_for(puck, puck_urdf(tmp_path), "arm_mount")
    for check in (on_plain, on_puck):
        assert check.hand_added == ["fingers_131mm", "fingertips_151mm", "wrist_camera"]
        assert len(check.spheres["arm_robotiq_85_base_link"][1]) == 19 + 4 + 5
    cs0, cs1 = on_plain.spheres["arm_robotiq_85_base_link"][0], on_puck.spheres["arm_robotiq_85_base_link"][0]
    assert np.array_equal(cs0[-5:], tips.centres) and np.array_equal(cs0[19:23], stock.centres)  # as written
    assert np.allclose(cs1[-5:], tips.centres - [0.0, 0.0, PUCK_M], atol=1e-12)  # from the flange: 14 mm nearer
    assert np.array_equal(cs1[19:23], stock.centres)  # the gripper's own: where the gripper is
    # the same check as without `frame` on the description the runs were flown on: every sphere, every link
    unframed = [body.HandGroup(g.name, g.link, g.centres, g.radii, g.covers) for g in m.hand]
    was = body.BodyModel(**{**m.__dict__, "hand": unframed}).check_for(plain, flatbot_urdf(), "arm_mount")
    assert list(was.spheres) == list(on_plain.spheres) and list(was.paths) == list(on_plain.paths)
    for link in was.spheres:
        assert np.array_equal(was.spheres[link][0], on_plain.spheres[link][0])
        assert np.array_equal(was.spheres[link][1], on_plain.spheres[link][1])
    # and the tips' ends, PADS_FROM_FLANGE_M out of the flange and as wide as the wrist's frames have them,
    # are as deep inside the hand's spheres on both: by the flange's own pose, no tool frame in it
    ends = [(0.0, 0.0, PADS_FROM_FLANGE_M), (0.026, 0.0, PADS_FROM_FLANGE_M), (-0.022, 0.0, PADS_FROM_FLANGE_M)]
    for q in arm_poses()[:6]:
        depth = []
        for chain, check in ((plain, on_plain), (puck, on_puck)):
            Rf, f = flange_pose(chain, q)
            C, r, _links = check.spheres_at(q)
            depth.append([float((np.linalg.norm(C - (f + Rf @ np.asarray(p)), axis=1) - r).min()) for p in ends])
        assert max(depth[0]) <= -0.0099 and all(b <= a + 1e-9 for a, b in zip(depth[0], depth[1], strict=True))


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
    """Depth nearer the lens than clear_table.WRIST_NEAREST_M is taken for the hand itself. The fingertips
    end PADS_FROM_FLANGE_M out of the flange and the lens stands 15 mm (flatbot's fitted mount) or 22.5 mm
    (gen_1_5's calibration) out of it: 136 and 128.5 mm of depth, whatever the tool frame. The cutoff
    keeps 25 mm or more over both, and over fingers a centimetre longer than stated (the reach is known
    to 150-161 mm: constants)."""
    for lens_z in (LENS_OUT_FLATBOT, LENS_OUT_GEN15):
        ahead = PADS_FROM_FLANGE_M - lens_z
        assert 0.025 <= clear_table.WRIST_NEAREST_M - ahead <= 0.045
        assert clear_table.WRIST_NEAREST_M - (0.161 - lens_z) >= 0.015


def test_the_hands_constants_are_one_statement():
    """PADS_FROM_FLANGE_M is the fact; the figure for the gripper on the flange is worked out from it."""
    assert constants.FINGERS_PAST_MODEL_M == round(PADS_FROM_FLANGE_M - (TOOL_FROM_FLANGE_STOCK_M + TCP_OFFSET_M), 6)
