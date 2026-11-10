"""The side grasp's geometry and what cuRobo checks it against (side_grasp.py), on the fake planner
(fridge_fake.py): the hand's pose and tilts, the worlds of the slide, the hold and the swing, the
back-off. The whole open and close with the side grasp run in test_fridge_curobo.py (its fixture
runs every test with both grasps)."""

import numpy as np
import pytest
from door_common import common as dc
from door_common import open_door as shared
from fridge import constants as F
from fridge import open_door, push_open, side_grasp
from fridge.world import FridgeModel
from fridge_fake import HANDLE, HANDLE_QUAT, NORMAL, fridge  # noqa: F401 -- fridge: the fixture
from scipy.spatial.transform import Rotation as R


@pytest.fixture
def side(fridge):  # noqa: F811 -- the fixture above
    if F.GRASP != "side":
        pytest.skip("side grasp only")
    return fridge


def _geom():
    with dc.quiet():
        hinge, _, _, _ = shared.FIND_HINGE(HANDLE, HANDLE_QUAT, NORMAL, None)
    return dict(h_raw=HANDLE.copy(), orient=HANDLE_QUAT, n=NORMAL.copy(), hinge=hinge)


def _model(cand, geom):
    return FridgeModel(open_door.closed_door(geom, cand["h"], cand["ref"], cand))


def test_the_hand_slides_in_from_the_hinge_side_20_deg_off_the_door(side):
    geom = _geom()
    level = side_grasp.candidates(geom)[0]
    assert level["pitch"] == 0 and level["side"]
    rot = R.from_quat(level["quat"]).as_matrix()
    approach, fingers = rot[:, 2], rot[:, 0]
    toward_hinge = np.r_[(geom["hinge"] - HANDLE)[:2], 0.0]
    toward_hinge -= NORMAL * (toward_hinge @ NORMAL)
    toward_hinge /= np.linalg.norm(toward_hinge)
    # along the door away from the hinge, turned 20 deg into it, level
    off_normal = np.degrees(np.arccos(approach @ -NORMAL))
    assert off_normal == pytest.approx(F.SIDE_INSERT_DEG, abs=1e-6)
    assert approach @ toward_hinge < 0 and abs(approach[2]) < 1e-9
    # the fingers close across the door's normal (turned with the hand), level
    assert abs(fingers @ NORMAL) > np.cos(np.radians(90 - F.SIDE_INSERT_DEG)) - 1e-9
    assert abs(fingers[2]) < 1e-9
    # the pads where they were on the arm on 11-06: 1.1 cm behind the detected point, into the
    # door, then 5 cm on along the approach
    pads = level["pos"] + approach * shared.GRASP_PADS_PAST_TOOL
    expect = HANDLE - NORMAL * F.SIDE_PADS_INTO_M + approach * F.SIDE_PADS_PAST_BAR_M
    np.testing.assert_allclose(pads, expect, atol=1e-9)
    # 11-06's bar point: the detected point - into x (GRIP_EXT - X_ADJUST - pads - SIDE_DEEPER_M)
    assert F.SIDE_PADS_INTO_M == pytest.approx(-(0.126 - 0.040 - 0.042 - 0.055))


def test_every_tilt_from_40_up_to_40_down_keeps_the_pads_on_the_bar(side):
    cands = side_grasp.candidates(_geom())
    assert [c["pitch"] for c in cands] == list(F.SIDE_TILTS_DEG)
    assert min(F.SIDE_TILTS_DEG) == -40 and max(F.SIDE_TILTS_DEG) == 40
    assert F.SIDE_TILTS_DEG[0] == 0  # level first: what ran on the arm
    pads = [c["pos"] + R.from_quat(c["quat"]).as_matrix()[:, 2] * shared.GRASP_PADS_PAST_TOOL
            for c in cands]
    for c, p in zip(cands, pads, strict=True):
        np.testing.assert_allclose(p, pads[0], atol=1e-9)
        nose = R.from_quat(c["quat"]).as_matrix()[:, 2][2]  # + tilt: nose down
        assert np.degrees(np.arcsin(-nose)) == pytest.approx(c["pitch"], abs=1e-6)


def test_the_door_file_is_the_front_grasps_so_the_close_reads_it_the_same(side):
    geom = _geom()
    cand = side_grasp.candidates(geom)[3]
    door = open_door.closed_door(geom, cand["h"], cand["ref"], cand)
    assert door["grasp"] == "side"
    with dc.quiet():
        _, h, _, _ = shared.FIND_HINGE(HANDLE, HANDLE_QUAT, NORMAL, None)
        level = np.asarray(shared.grasp_pose(h, HANDLE_QUAT, NORMAL, 0)[1].position)
    front = open_door.closed_door(geom, h, level)
    for k in ("closed_grasp_pos", "closed_handle", "hinge", "face_behind_grasp", "door_t",
              "door_past_handle"):
        np.testing.assert_allclose(door[k], front[k], atol=1e-9)
    # the door's angle is read from where the hand holds the bar, however it turned about it
    pos, quat = side_grasp.swing_poses(cand, FridgeModel(door).df, 60.0, 25.0, [60.0])[0]
    angle = open_door.hand_angle(door, [*pos, *quat])
    assert angle == pytest.approx(60.0 * door["open_sign"], abs=1e-6)


def test_the_slide_and_the_hold_see_the_door_recessed_round_the_hand_and_no_handle(side):
    geom = _geom()
    cand = side_grasp.candidates(geom)[0]
    model = _model(cand, geom)
    world = side_grasp.hold_world(model, (0.0,))
    names = {b["name"] for b in world}
    assert not any(n.startswith("handle") for n in names)
    assert "door0_recessed" in names and "body" in names
    # the recessed slab's face SIDE_FACE_RECESS_M behind the door's; whole outside the window
    recessed = next(b for b in world if b["name"] == "door0_recessed")
    assert recessed["dims"][0] == pytest.approx(model.df.t - F.SIDE_FACE_RECESS_M, abs=1e-5)
    u0, u1, _, _ = side_grasp.window(model)
    assert u0 == pytest.approx(model.df.u_handle - F.SIDE_WINDOW_HINGE_M)
    for standoff in F.SIDE_SLIDE_M:  # every line-up, and so the whole slide, inside the window
        p = open_door.lineup_point(cand, standoff)
        assert u0 < float((p[:2] - model.df.hinge[:2]) @ model.df.e) < u1
    goals = open_door.approach_goals(model, cand, F.SIDE_SLIDE_M[0])
    assert [g.path for g in goals] == ["free", "line"] and goals[1].world == world


def test_the_swing_carries_the_bar_round_and_checks_the_arm_against_the_door(side):
    geom = _geom()
    cand = side_grasp.candidates(geom)[0]
    model = _model(cand, geom)
    df = model.df
    (goal,) = open_door.swing_goals(model, cand, 60.0, lag=20.0)
    degs = [60.0 * k / len(goal.through) for k in range(1, len(goal.through) + 1)]
    bar0 = np.asarray(cand["bar"], float)
    in_tool = R.from_quat(cand["quat"]).inv().apply(bar0 - cand["pos"])
    for (pos, quat), d in zip(goal.through, degs, strict=True):
        bar = pos + R.from_quat(quat).apply(in_tool)
        expect = df.hinge + R.from_euler("z", df.sign * d, degrees=True).apply(bar0 - df.hinge)
        np.testing.assert_allclose(bar[:2], expect[:2], atol=1e-9)  # the bar on the door's arc
        turn = np.degrees((R.from_quat(quat) * R.from_quat(cand["quat"]).inv()).magnitude())
        assert turn == pytest.approx(d - 20.0 * d / 60.0, abs=1e-6)  # 20 deg behind at the end
    # each stretch of the swing against the door where it was when the stretch started: never ahead
    # of the hand, at most SIDE_SWING_CHECK_DEG behind it; one world (one load, one check) per stretch
    behind = []
    for (label, _), d in zip(goal.segment_worlds, degs, strict=True):
        at = float(label.split("the door at ")[1].split(" deg")[0])
        assert at <= d + 1e-9
        behind.append(d - at)
    assert max(behind) <= F.SIDE_SWING_CHECK_DEG + 2.0 + 1e-6  # + one pose's step
    assert len({id(w) for _, w in goal.segment_worlds}) <= np.ceil(60.0 / F.SIDE_SWING_CHECK_DEG)


def test_the_side_back_off_slides_off_the_bar_then_out_then_to_the_front_grasps_off_pose(side):
    geom = _geom()
    cand = side_grasp.candidates(geom)[0]
    model = _model(cand, geom)
    pos, quat = side_grasp.swing_poses(cand, model.df, 60.0, 0.0, [60.0])[0]
    back, out, front = push_open.back_off_goals(model, 60.0, (pos, quat))
    approach = R.from_quat(quat).as_matrix()[:, 2]
    np.testing.assert_allclose(back.pos, pos - approach * F.SIDE_BACK_OFF_M, atol=1e-9)
    assert back.path == "line" and out.path == "free" and front.path == "free"
    n60 = np.r_[model.df._rot(60.0) @ model.df.n, 0.0]
    assert float((back.pos - pos) @ n60) > 0  # off the bar away from the door's face, never into it
    np.testing.assert_allclose(out.pos, back.pos + n60 * F.SIDE_BACK_OUT_M, atol=1e-9)
    # where the front grasp's back-off ends: its hold carried round with the door, BACK_OFF_M back
    p, q = push_open._swung(model.df, np.asarray(cand["front"][:3]), np.asarray(cand["front"][3:]),
                            60.0)
    np.testing.assert_allclose(front.pos, p - R.from_quat(q).as_matrix()[:, 2] * F.BACK_OFF_M,
                               atol=1e-9)
    assert any(b["name"].startswith("handle") for b in front.world)  # round the handle too


def test_the_side_grasp_has_its_own_placement_advice(side):
    assert F.PLACEMENT_CELLS == F.MOUNTS[30]["side_placement_cells"]
    side_box = (F.MOUNTS[30]["side_placement_x"], F.MOUNTS[30]["side_placement_y"])
    assert (F.PLACEMENT_X, F.PLACEMENT_Y) == side_box
    moves, (_, dy) = open_door.placement_moves(np.array([0.70, -0.70, 0.315]), NORMAL)
    assert dy > 0 and any("to its right" in m for m in moves)  # the handle too far right
    moves, _ = open_door.placement_moves(np.array([0.75, -0.30, 0.315]), NORMAL)
    assert moves == []
