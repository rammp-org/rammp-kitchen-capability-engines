"""The button sighted from over it (perception/button_sight): its centre from the wrist lens's vertical, by
the colour frame alone, on synthetic frames whose truth is known and on the recorded flatbot frames."""

import math

import numpy as np
import pytest
from conftest import fixture_path
from test_button_aim import BOX_H, BUTTON_R, BUTTON_XY, FLAT_TABLE_Z, OAK_HW, OAK_K, PINK, ROT_DOWN, SEAM_W, render

from box_opening.models.container import ContainerModel
from box_opening.perception import button_sight as bs


@pytest.fixture(scope="module")
def pink():
    """The pink canister, its button the size of the rendered seam: the model's 44.3 mm is the seam circle as
    the staging aim measured it (both of its groove's edges), and the render's groove runs from BUTTON_R to
    BUTTON_R + SEAM_W, 46 mm across at its middle. The offset is scaled by the circle's size against this."""
    import dataclasses

    return dataclasses.replace(ContainerModel.load(PINK), button_diameter_m=2.0 * BUTTON_R + SEAM_W)


@pytest.fixture(scope="module")
def recorded_pink():
    return ContainerModel.load(PINK)


def _turned(rot, axis, deg):
    """`rot` (the camera's axes in the base) turned `deg` about the base axis `axis`."""
    import cv2

    R, _ = cv2.Rodrigues(np.asarray(axis, float) / np.linalg.norm(axis) * math.radians(deg))
    return R @ rot


def _sighted(m, off, height, rot=ROT_DOWN, dist=None):
    """sight_button on a frame from a lens `height` over the lid, vertically over the aim at BUTTON_XY, the
    button truly `off` (base xy, m) from the aim: (ButtonSight, why)."""
    lid_z = FLAT_TABLE_Z + m.dims[2]
    true = (BUTTON_XY[0] + off[0], BUTTON_XY[1] + off[1])
    cam = np.r_[BUTTON_XY, lid_z + height]
    color, _depth = render(cam, rot, [(true, lid_z, 0.0)], lid_z - BOX_H, dist=dist)
    return bs.sight_button(color, OAK_K, dist, rot, m, (0.0, 0.0), height)


OFFS = [(0.0, 0.0)] + [
    (d * math.cos(math.radians(a)), d * math.sin(math.radians(a))) for d in (0.010, 0.025) for a in (0, 90, 180, 270)
]


@pytest.mark.parametrize("height", [0.15, 0.24])
@pytest.mark.parametrize("off", OFFS)
def test_the_button_is_placed_from_the_lens_straight_down(pink, off, height):
    """Case 1: the hand straight down, the lens over the aim, the button 0, 10 and 25 mm from it in four
    directions, 15 and 24 cm below: its offset within 1.5 mm."""
    s, why = _sighted(pink, off, height)
    assert s is not None, why
    assert math.hypot(s.off_xy[0] - off[0], s.off_xy[1] - off[1]) < 0.0015, (s, off)
    assert s.how in ("seam+rim", "seam")
    assert s.below_m == pytest.approx(height, rel=0.10)  # a record only: good to some 10 %


@pytest.mark.parametrize("axis", [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)])
@pytest.mark.parametrize("off", [(0.0, 0.0), (0.025, 0.0), (0.0, -0.025)])
def test_a_hand_5_deg_short_with_a_mount_1_5_deg_off_still_places_it(pink, off, axis):
    """Case 2: the hand 5 deg short of straight down and the camera's axis 1.5 deg off the tool's, the
    lens 20 cm over the lid: within 2 mm. The picture is turned to look straight down first; without the
    turn the lean alone would put the button some 0.2 x tan(6.5 deg) = 23 mm off."""
    rot = _turned(_turned(ROT_DOWN, axis, 5.0), (axis[1], -axis[0], 0.0), 1.5)
    s, why = _sighted(pink, off, 0.20, rot)
    assert s is not None, why
    assert math.hypot(s.off_xy[0] - off[0], s.off_xy[1] - off[1]) < 0.002, (s, off)


def test_the_lens_distortion_is_taken_out_with_the_turn(pink):
    """The OAK's own plumb_bob, its rays bent as the real lens bends them: the remap undistorts as it
    turns, and the button at 25 mm is placed as without it."""
    dist = [0.08, -0.12, 0.0, 0.0, 0.04]
    s, why = _sighted(pink, (0.025, 0.0), 0.20, dist=dist)
    assert s is not None, why
    assert math.hypot(s.off_xy[0] - 0.025, s.off_xy[1]) < 0.0015, s


def test_the_answer_is_the_picture_not_what_was_expected(pink):
    """expect_off and expect_below choose and gate the circle; they are never the answer. The same frame
    asked with the button expected 15 mm away and 15 % further down gives the same centre."""
    a, _ = _sighted(pink, (0.010, 0.0), 0.20)
    lid_z = FLAT_TABLE_Z + pink.dims[2]
    cam = np.r_[BUTTON_XY, lid_z + 0.20]
    color, _depth = render(cam, ROT_DOWN, [((BUTTON_XY[0] + 0.010, BUTTON_XY[1]), lid_z, 0.0)], lid_z - BOX_H)
    b, why = bs.sight_button(color, OAK_K, None, ROT_DOWN, pink, (-0.015, 0.0), 0.23)
    assert b is not None, why
    assert b.off_xy == pytest.approx(a.off_xy, abs=2e-4)


def test_a_frame_without_the_button_says_what_it_saw(pink):
    """Only table where the button is expected: None and why, never a guess."""
    lid_z = FLAT_TABLE_Z + pink.dims[2]
    cam = np.r_[BUTTON_XY[0], BUTTON_XY[1] + 0.30, lid_z + 0.20]
    color, _depth = render(cam, ROT_DOWN, [(BUTTON_XY, lid_z, 0.0)], lid_z - BOX_H)
    s, why = bs.sight_button(color, OAK_K, None, ROT_DOWN, pink, (0.0, 0.0), 0.20)
    assert s is None and "no circle" in why


def test_too_near_or_no_level_picture_is_refused(pink):
    color = np.zeros((*OAK_HW, 3), np.uint8)
    s, why = bs.sight_button(color, OAK_K, None, ROT_DOWN, pink, (0.0, 0.0), 0.05)
    assert s is None and "too near" in why
    sideways = np.column_stack([(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])  # its x along the world's z
    s, why = bs.sight_button(color, OAK_K, None, sideways, pink, (0.0, 0.0), 0.20)
    assert s is None and "near vertical" in why
    assert bs.sight_button(None, OAK_K, None, ROT_DOWN, pink, (0.0, 0.0), 0.20) == (None, "no frame")


def test_the_seam_is_told_from_its_family(pink):
    """which_seam: a seam-and-rim pair where the lid is round; else the innermost at the size expected;
    else nothing, saying what was seen. The groove's two edges are one circle."""
    want = pink.button_diameter_m / pink.lid_dims[0]
    seam, how = bs.which_seam([(0, 0, 40.0), (0, 0, 40.0 / want)], pink, 60.0)
    assert (seam[2], how) == (40.0, "seam+rim")
    seam, how = bs.which_seam([(0, 0, 41.0), (0, 0, 55.0)], pink, 40.0)
    assert (seam[2], how) == (41.0, "seam")
    seam, why = bs.which_seam([(0, 0, 60.0)], pink, 40.0)  # the 60 mm ring alone: 1.5 times the seam
    assert seam is None and "1.50 times" in why
    # a groove's darkening inner edge and brightening outer one are one circle; a second edge of one sign
    # beside one carried further round is that edge seen twice; a lone edge stands as it is
    fam = [(0, 0, 40.0, -1, 16), (1, 0, 42.0, 1, 12), (0, 0, 42.5, 1, 10), (0, 0, 60.0, -1, 14)]
    assert bs.grooves_merged(fam) == [(0.5, 0.0, 41.0), (0.0, 0.0, 60.0)]
    fams = bs.families([(100, 100, 40.0), (103, 101, 60.0), (200, 100, 40.0)])
    assert [len(f) for f in fams] == [2, 1]


def test_a_circle_the_frame_cuts_is_not_whole():
    inside = np.zeros((100, 100), bool)
    inside[10:90, 10:90] = True
    assert bs._whole((50, 50, 30), inside) and not bs._whole((50, 25, 30), inside)


def test_a_strong_distortion_off_the_lens_vertical_is_taken_out(pink):
    """A lens that bends its rays hard (k1 -0.30), the hand 10 deg short of straight down and the button
    25 mm off the aim: the button stands well off the picture's centre, where the bending is large. The
    remap undistorts as it turns: within 0.1 mm. The same frame read as if the lens bent nothing is more
    than 0.3 mm off (0.5 here), or not found: this case is one the undistortion decides. (The bend shrinks the
    circle's place and its size together, and the offset is their ratio: most of it cancels.)"""
    dist = [-0.30, 0.09, 0.0, 0.0, 0.0]
    rot = _turned(ROT_DOWN, (1.0, 1.0, 0.0), 10.0)
    off = (0.025, -0.010)
    s, why = _sighted(pink, off, 0.20, rot, dist=dist)
    assert s is not None, why
    assert math.hypot(s.off_xy[0] - off[0], s.off_xy[1] - off[1]) < 0.0001, (s, off)
    lid_z = FLAT_TABLE_Z + pink.dims[2]
    cam = np.r_[BUTTON_XY, lid_z + 0.20]
    true = (BUTTON_XY[0] + off[0], BUTTON_XY[1] + off[1])
    color, _depth = render(cam, rot, [(true, lid_z, 0.0)], lid_z - BOX_H, dist=dist)
    bent, _why = bs.sight_button(color, OAK_K, None, rot, pink, (0.0, 0.0), 0.20)
    assert bent is None or math.hypot(bent.off_xy[0] - off[0], bent.off_xy[1] - off[1]) > 0.0003, bent


def _ring_only(m, off, height, ring_r):
    """A frame from the lens `height` over the lid, straight down over the aim at BUTTON_XY: the button
    truly `off` from the aim, its seam and top painted over in the lid's colour, a dark ring `ring_r` (m)
    round it on the lid."""
    import cv2
    from test_button_aim import LID_BGR, _pixel

    lid_z = FLAT_TABLE_Z + m.dims[2]
    cam = np.r_[BUTTON_XY, lid_z + height]
    true = np.r_[BUTTON_XY[0] + off[0], BUTTON_XY[1] + off[1], lid_z]
    color, _depth = render(cam, ROT_DOWN, [(tuple(true[:2]), lid_z, 0.0)], lid_z - BOX_H)
    u, v = _pixel(true, cam, ROT_DOWN)
    px = math.hypot(*np.subtract(_pixel(true + np.r_[ring_r, 0.0, 0.0], cam, ROT_DOWN), (u, v)))
    seam_px = px * (BUTTON_R + SEAM_W + 0.002) / ring_r
    centre = (round(u * 16), round(v * 16))
    cv2.circle(color, centre, round(seam_px * 16), LID_BGR, -1, cv2.LINE_AA, 4)
    cv2.circle(color, centre, round(px * 16), (70, 55, 110), 3, cv2.LINE_AA, 4)
    return color


@pytest.mark.parametrize("more", [0.0, 0.03])
def test_a_seam_not_seen_beside_its_60_mm_ring_is_none_not_the_ring(recorded_pink, more):
    """A lid whose seam the picture does not show while its 60 mm moulded ring round the button does (the
    model's own 44.3 mm seam: the ring 1.35 times it), the lens over the button at the modelled height and
    with the real top 30 mm higher than modelled (the ring then looks larger still against the seam
    expected): None, or the centre within 3 mm. Never the ring taken for the seam, its scale making a 20 mm
    offset some 15 mm (SEAM_WINDOW)."""
    off, height = (0.020, 0.0), 0.19
    color = _ring_only(recorded_pink, off, height, 0.030)
    s, why = bs.sight_button(color, OAK_K, None, ROT_DOWN, recorded_pink, (0.0, 0.0), height + more)
    assert s is None or math.hypot(s.off_xy[0] - off[0], s.off_xy[1] - off[1]) < 0.003, (s, why)


# the flatbot OAK's colour intrinsics as its frames record them (640x400)
OAK_FLATBOT_K = np.array([[398.4, 0.0, 320.0], [0.0, 398.4, 200.0], [0.0, 0.0, 1.0]])
WEAK_RING, STRONG_RING = (150, 125, 215), (60, 45, 100)  # the moulded ring a shade off the lid; darker than the seam


def _rho(d, off, z):
    """How far from the button's centre (`off` from the lens's vertical) rays `d` (base axes) meet the level
    plane `z` below the lens."""
    t = -z / d[..., 2]
    return np.hypot(t * d[..., 0] - off[0], t * d[..., 1] - off[1])


def _lid_frame(m, off, below, ring_bgr=WEAK_RING, seed=0, ss=3, rot=ROT_DOWN, k=OAK_FLATBOT_K):
    """A frame from a camera with axes `rot` (default straight down) and intrinsics `k` (the flatbot OAK's),
    its lens `below` (m) over the real lid top and the button truly `off` (base xy) from the lens's
    vertical: the pink lid as it is (test_button_aim's render has no ring), its seam the model's
    button_diameter_m groove 2 mm wide, its 60 mm moulded ring 1.5 mm wide in `ring_bgr`, its rim, the box
    0.12 m tall on the table."""
    rng = np.random.default_rng(seed)
    h, w = OAK_HW
    seam_r, seam_w, ring_r, ring_w = m.button_diameter_m / 2.0, 0.002, 0.030, 0.0015
    lid_r, box_h = m.lid_dims[0] / 2.0, 0.12
    table, side, lid, seam, button = (95, 105, 110), (120, 95, 175), (175, 150, 235), (70, 55, 110), (125, 95, 205)
    vv, uu = np.mgrid[0:h, 0:w].astype(float)
    col = np.zeros((h, w, 3))
    for dv in (np.arange(ss) + 0.5) / ss - 0.5:
        for du in (np.arange(ss) + 0.5) / ss - 0.5:
            d = (
                np.stack([(uu + du - k[0, 2]) / k[0, 0], (vv + dv - k[1, 2]) / k[1, 1], np.ones_like(uu)], -1)
                @ np.asarray(rot, float).T
            )

            r_top, r_floor = _rho(d, off, below), _rho(d, off, below + box_h)
            c = np.where((r_floor < lid_r)[..., None], np.array(side, float), np.array(table, float))
            for mask, bgr in (
                (r_top < lid_r, lid),
                (np.abs(r_top - ring_r) < ring_w / 2.0, ring_bgr),
                (np.abs(r_top - seam_r) < seam_w / 2.0, seam),
                (r_top < seam_r - seam_w / 2.0, button),
            ):
                c = np.where(mask[..., None], np.array(bgr, float), c)
            col += c
    return np.clip(col / ss**2 + rng.normal(0.0, 2.0, col.shape), 0, 255).astype(np.uint8)


TOP_ERRS = [-0.020, -0.010, 0.0, 0.010, 0.020, 0.025, 0.028, 0.030, 0.032, 0.035, 0.040, 0.050]
SWEEP_DIRS = range(0, 360, 45)


def test_the_seam_in_view_is_the_truth_or_none_whatever_the_top_reads(recorded_pink):
    """The seam in plain view, the lens over the aim 196 mm over the top the model expects and the button
    20 mm off its vertical in eight directions, the real top from 20 mm lower to 50 mm higher than modelled
    (the recorded runs read it 25-32 mm low): the button's centre within 2 mm, or None; never another
    circle's. Nearly always found (some three in a hundred None here)."""
    m, expect = recorded_pink, 0.196
    wrong, none = [], 0
    for dz in TOP_ERRS:
        for a in SWEEP_DIRS:
            off = (0.020 * math.cos(math.radians(a)), 0.020 * math.sin(math.radians(a)))
            s, _why = bs.sight_button(_lid_frame(m, off, expect - dz), OAK_FLATBOT_K, None, ROT_DOWN, m, (0, 0), expect)
            if s is None:
                none += 1
            elif math.hypot(s.off_xy[0] - off[0], s.off_xy[1] - off[1]) >= 0.002:
                wrong.append((dz, a, s.how, round(s.r_px, 1), round(1000 * math.hypot(*np.subtract(s.off_xy, off)), 2)))
    assert not wrong, wrong
    assert none <= 0.05 * len(TOP_ERRS) * len(SWEEP_DIRS), none


def test_a_stray_circle_nearer_where_the_button_is_expected_does_not_hide_it(recorded_pink):
    """A dark ring of 70 px radius (1.57 times the seam's 45: no seam by its size, no rim to pair) drawn
    on the frame nearer where the button is expected than the button is (40 px from there; the button 81
    px): its family is asked first, holds no seam, and the button's family is asked next. The seam, within
    2 mm, by its rim."""
    import cv2

    m, below, off = recorded_pink, 0.196, (0.030, 0.0)
    color = _lid_frame(m, off, below)
    plain, why = bs.sight_button(color, OAK_FLATBOT_K, None, ROT_DOWN, m, (0.0, 0.0), below)
    assert plain is not None, why
    lens = OAK_FLATBOT_K[:2, 2]
    out = (np.asarray(plain.uv) - lens) / np.linalg.norm(np.asarray(plain.uv) - lens)
    stray = lens - 60.0 * out  # 60 px on the far side of the lens; the button expected 20 px that way
    cv2.circle(color, (int(round(stray[0])), int(round(stray[1]))), 70, (60, 45, 100), 3)
    s, why = bs.sight_button(color, OAK_FLATBOT_K, None, ROT_DOWN, m, (-off[0] / 3.0, -off[1] / 3.0), below)
    assert s is not None, why
    assert s.how == "seam+rim", s
    assert math.hypot(s.off_xy[0] - off[0], s.off_xy[1] - off[1]) < 0.002, s


# fixed scenes: the real top as modelled, 15 and 28 mm higher (the recorded runs read it 25-32 mm low); the
# button 20 mm off the lens's vertical in four directions
SCENE_DZ = (0.0, 0.015, 0.028)
SCENE_DIRS = (0, 90, 180, 270)


def _sweep(m, cases):
    """Over (frame, rot, expect_below, off, tag) `cases` of model `m`'s lid: (each wrong answer, how many
    None, how many)."""
    wrong, none, n = [], 0, 0
    for color, rot, expect, off, tag in cases:
        n += 1
        s, _why = bs.sight_button(color, OAK_FLATBOT_K, None, rot, m, (0.0, 0.0), expect)
        if s is None:
            none += 1
        elif math.hypot(s.off_xy[0] - off[0], s.off_xy[1] - off[1]) >= 0.002:
            wrong.append((tag, s.how, round(s.r_px, 1), round(1000 * math.hypot(*np.subtract(s.off_xy, off)), 2)))
    return wrong, none, n


def test_the_seam_is_the_truth_or_none_whatever_range_the_arm_expects(recorded_pink):
    """The lens 196 mm over the modelled top, straight down over the aim; each scene (SCENE_DZ, SCENE_DIRS,
    the ring weak and strong) fixed, and the range the arm expects stepped 1 mm at a time over +-15 mm: the
    button's centre within 2 mm, or None, and None in no more than 1 in 20. With Hough's radius choosing the
    circles (round 2) the seam was lost where a band's edge fell near it and a circle across its arc and the
    ring's taken, 4-6 mm off (the expected range 2-4 mm long over the top as modelled): 6 here, the frames
    sampled 3 x 3 a pixel (at 2 x 2 none fell on a band's edge)."""
    H = 0.196

    def cases():
        for ring in (WEAK_RING, STRONG_RING):
            for dz in SCENE_DZ:
                for a in SCENE_DIRS:
                    off = (0.020 * math.cos(math.radians(a)), 0.020 * math.sin(math.radians(a)))
                    color = _lid_frame(recorded_pink, off, H - dz, ring_bgr=ring)
                    for mm in range(-15, 16):
                        yield color, ROT_DOWN, H + mm / 1000.0, off, (ring, dz, a, mm)

    wrong, none, n = _sweep(recorded_pink, cases())
    assert not wrong, wrong
    assert none <= n / 20, (none, n)


def test_the_seam_is_the_truth_or_none_with_the_hand_short_of_straight_down(recorded_pink):
    """The same scenes with the hand 0, 1, 2, 5 and 8 deg short of straight down about each of the
    picture's axes, the lens over the aim 196 mm over the modelled top and that range expected: within 2 mm,
    or None in no more than 1 in 20. The ring weak in two directions, strong in the other two."""
    H = 0.196

    def cases():
        for axis in (0, 1):
            for deg in (0, 1, 2, 5, 8) if axis == 0 else (1, 2, 5, 8):
                rot = _turned(ROT_DOWN, ROT_DOWN[:, axis], deg)
                for dz in SCENE_DZ:
                    for i, a in enumerate(SCENE_DIRS):
                        off = (0.020 * math.cos(math.radians(a)), 0.020 * math.sin(math.radians(a)))
                        ring = (WEAK_RING, STRONG_RING)[i % 2]
                        color = _lid_frame(recorded_pink, off, H - dz, ring_bgr=ring, ss=2, rot=rot)
                        yield color, rot, H, off, (axis, deg, dz, a)

    wrong, none, n = _sweep(recorded_pink, cases())
    assert not wrong, wrong
    assert none <= n / 20, (none, n)


@pytest.mark.parametrize("dz", [0.020, 0.028])
def test_a_seam_beside_a_stronger_ring_is_still_the_seam(recorded_pink, dz):
    """The real top 20 and 28 mm higher than modelled, so the seam shows 1.11 and 1.17 times the size
    expected, and the 60 mm ring 1.35 times that, the stronger edge here (darker than the seam): when Hough's
    radius chose the circles (round 1, its bands 1.5 wide) the ring took the seam's place. Every direction:
    the seam, within 2 mm."""
    m, expect = recorded_pink, 0.196
    for a in SWEEP_DIRS:
        off = (0.020 * math.cos(math.radians(a)), 0.020 * math.sin(math.radians(a)))
        color = _lid_frame(m, off, expect - dz, ring_bgr=STRONG_RING)
        s, why = bs.sight_button(color, OAK_FLATBOT_K, None, ROT_DOWN, m, (0.0, 0.0), expect)
        assert s is not None, (a, why)
        assert math.hypot(s.off_xy[0] - off[0], s.off_xy[1] - off[1]) < 0.002, (a, s)


def test_a_circle_is_carried_by_one_edge_of_its_own_going_round_it():
    """carried: a dark ring's outer edge (brightening outward) and inner one (darkening) are each fitted
    alone, the ring's width apart, from a start some pixels off; a ring shown on half its round (an arc
    Hough closed into a circle) is None, and so is a picture with no edge at all, and an edge asked for with
    the other sign."""
    import cv2

    whole = np.full((200, 200), 200, np.uint8)
    cv2.circle(whole, (100, 100), 40, 60, 4)
    e = bs.edges_of(whole)
    out, inner = bs.carried(e, (102.0, 98.0, 40.0), 1), bs.carried(e, (102.0, 98.0, 40.0), -1)
    for fit in (out, inner):
        assert fit is not None and math.hypot(fit[0] - 100, fit[1] - 100) < 0.3 and fit[3] == bs.SUPPORT_SECTORS, fit
    assert 2.0 < out[2] - inner[2] < 6.0 and abs((out[2] + inner[2]) / 2.0 - 40.0) < 0.5, (out, inner)
    arc = np.full((200, 200), 200, np.uint8)
    cv2.ellipse(arc, (100, 100), (40, 40), 0, 0, 180, 60, 4)
    assert bs.carried(bs.edges_of(arc), (101.0, 99.0, 40.0), 1) is None
    assert bs.carried(bs.edges_of(np.full((200, 200), 200, np.uint8)), (101.0, 99.0, 40.0), 1) is None
    bright = np.full((200, 200), 60, np.uint8)
    cv2.circle(bright, (100, 100), 40, 200, -1)  # a bright disc: its edge darkens outward, and only that
    assert bs.carried(bs.edges_of(bright), (100.0, 100.0, 40.0), 1) is None
    assert bs.carried(bs.edges_of(bright), (100.0, 100.0, 40.0), -1) is not None


def test_edges_that_run_along_the_radius_do_not_carry_a_circle():
    """A dark ring on half its round (8 of 16 sectors: not carried) and, over the other half, dark stripes
    running out from its centre through its band, 40 to the turn (a texture, or straight edges crossing
    the ring's place): the stripes' edges lie at the ring's radius, but their gradient runs round the
    circle, not along the radius (RADIAL_COS), so they count for neither edge, and the ring is None either
    way. Counted, they carry both of its edges in 12 sectors."""
    import cv2

    yy, xx = np.mgrid[0:200, 0:200].astype(float)
    th, r = np.arctan2(yy - 100, xx - 100), np.hypot(xx - 100, yy - 100)
    img = np.full((200, 200), 200, np.uint8)
    cv2.ellipse(img, (100, 100), (40, 40), 0, 0, 180, 60, 4)
    img[(np.sin(40 * th) > 0) & (th < 0) & (r > 30) & (r < 50)] = 60
    e = bs.edges_of(img)
    assert bs.carried(e, (101.0, 99.0, 40.0), 1) is None
    assert bs.carried(e, (101.0, 99.0, 40.0), -1) is None


def test_a_seam_whose_edge_is_doubled_is_the_same_whatever_order_houghs_centres_come_in(recorded_pink):
    """Two brightening edges 1.2 px apart in radius and 2 px apart in centre (one edge seen twice: a shadow
    beside the seam's), each a whole ring of edge pixels, and Hough's two centres one on each: circles_about
    finds the same circles with the centres in either order, and the seam picked from them (grooves_merged,
    which_seam) is the same within 0.05 px. A fit stopped where it passes near a circle already found loses
    one of the two to whichever is fitted first, and the seam moves 1.5 px with the order."""
    rng = np.random.default_rng(212)
    parts = []
    for (cu, cv_), r in (((100.0, 100.0), 30.0), ((102.0, 100.0), 31.2)):
        a = np.linspace(0.0, 2.0 * math.pi, round(2.0 * math.pi * r), endpoint=False)
        x, y = cu + r * np.cos(a) + rng.normal(0, 0.15, len(a)), cv_ + r * np.sin(a) + rng.normal(0, 0.15, len(a))
        parts.append((x, y, 200.0 * np.cos(a), 200.0 * np.sin(a)))
    x, y, gx, gy = (np.concatenate(p) for p in zip(*parts, strict=True))
    e = bs.Edges(x, y, gx, gy, np.hypot(gx, gy))
    answers = []
    for centres in ([(102.0, 100.0), (100.0, 100.0)], [(100.0, 100.0), (102.0, 100.0)]):
        found = sorted(bs.circles_about(e, centres, 15.0, 60.0), key=lambda f: f[2])
        seam, how = bs.which_seam(bs.grooves_merged(found), recorded_pink, 30.0)
        assert seam is not None, how
        answers.append((found, seam))
    (found_a, seam_a), (found_b, seam_b) = answers
    assert len(found_a) == len(found_b) == 2, answers
    for f, g in zip(found_a, found_b, strict=True):
        assert f[3:] == g[3:] and max(abs(p - q) for p, q in zip(f[:3], g[:3], strict=True)) <= 0.05, answers
    assert max(abs(p - q) for p, q in zip(seam_a, seam_b, strict=True)) <= 0.05, answers


def test_a_fit_stopped_while_it_still_walks_is_not_taken(recorded_pink):
    """14:04: four candidates whose fits still walk after FIT_ROUNDS (0.04-0.35 px a round) toward circles
    of 74.7 and 90.6 px they reach by 60 rounds, and that a start on them holds at once. Stopped at 30 they
    stand 0.5-0.9 px short, where they began: each is None, or the circle a start on it holds (within 0.3
    px: another build's edges may let it settle in time), never where it stopped."""
    _view, e = _down_recorded("wrist_reach_look_flatbot_20261027_1404", recorded_pink)
    for own, starts in (
        ((203.7, 206.6, 74.7), [(201.5, 211.5, 70.6), (201.7, 205.8, 70.4), (202.5, 210.8, 71.5)]),
        ((199.4, 208.8, 90.5), [(201.7, 205.8, 87.8)]),
    ):
        held = bs.carried(e, own, -1)
        assert held is not None, own
        for start in starts:
            fit = bs.carried(e, start, -1)
            off = None if fit is None else max(abs(a - b) for a, b in zip(fit[:3], held[:3], strict=True))
            assert off is None or off <= 0.3, (start, fit, held)


def _lid_down(m, off, below, expect, **kw):
    """(Down, Edges, the circles each carried about Hough's centres) of a _lid_frame, the button expected
    `expect` below the lens and straight under it."""
    view, why = bs.look_down(_lid_frame(m, off, below, **kw), OAK_FLATBOT_K, None, ROT_DOWN, m, (0.0, 0.0), expect)
    assert view is not None, why
    e = bs.edges_of(view.gray)
    return view, e, bs.circles_about(e, bs._centres(view.gray, view.lo, view.hi), view.lo, view.hi)


def test_a_circle_across_the_seams_arc_and_the_rings_is_not_carried(recorded_pink):
    """A circle drawn across the seam's arc on one side and the ring's on the other (concentric, d px apart
    in radius: it stands d/2 off their centre), handed to carried as a candidate in eight directions and
    either sign: None. Hough's radius gave such circles in round 2, and a band wide enough to hold both arcs
    took them."""
    view, e, found = _lid_down(recorded_pink, (0.020, 0.0), 0.196, 0.196)
    seam, ring = bs.grooves_merged(found)[:2]
    d = ring[2] - seam[2]
    assert 12.0 < d < 20.0, (seam, ring)
    for a in range(0, 360, 45):
        u, v = seam[0] + d / 2.0 * math.cos(math.radians(a)), seam[1] + d / 2.0 * math.sin(math.radians(a))
        for sign in (1, -1):
            assert bs.carried(e, (u, v, (seam[2] + ring[2]) / 2.0), sign) is None, (a, sign)


STARTS = ((2.0, 0.0), (-2.0, 0.0), (0.0, 2.0), (0.0, -2.0))


def _one_answer(e, circle, sign):
    """carried begun 2 px off `circle` in four directions: the four fits, all within 0.3 px of each other."""
    fits = [bs.carried(e, (circle[0] + du, circle[1] + dv, circle[2]), sign) for du, dv in STARTS]
    assert all(f is not None for f in fits), (circle, fits)
    worst = max(max(abs(a[i] - b[i]) for i in range(3)) for a in fits for b in fits)
    assert worst <= 0.3, (circle, sign, fits)
    return fits


def test_the_fit_is_the_edges_own_wherever_it_begins(recorded_pink):
    """On a rendered lid, every circle the seam's family holds (the groove's two edges, the ring's two, the
    rim), fitted again from 2 px off in four directions: one answer, within 0.3 px."""
    _view, e, found = _lid_down(recorded_pink, (0.020, 0.0), 0.196, 0.196)
    assert len(found) >= 5, found
    for f in found:
        _one_answer(e, f, f[3])


def _down_recorded(name, pink):
    color, k, dist, rot, t, aim = _recorded(name, pink)
    view, why = bs.look_down(color, k, dist, rot, pink, (aim[0] - t[0], aim[1] - t[1]), t[2] - aim[2])
    assert view is not None, why
    return view, bs.edges_of(view.gray)


def test_the_fit_on_the_recorded_seam_is_its_edges_own_wherever_it_begins(recorded_pink):
    """16:23 (the seam's fit ran along the shadow and moved 1.4 mm with its start in round 2): the seam's
    two edges, each fitted from 2 px off in four directions, one answer within 0.3 px, their mean the
    groove's 36.1 px within 1 px. 10:59 (the fingers over 7 of its 16 sectors): from 2 px off where round 2's
    fit put the seam, in four directions, either edge: None every time, never a circle that moves with its
    start (on 9 sectors fits begun a pixel apart ended 6 px apart)."""
    view, e = _down_recorded("wrist_reach_aim_flatbot_20261027_1623", recorded_pink)
    s, why = _sight_recorded("wrist_reach_aim_flatbot_20261027_1623", recorded_pink)
    assert s is not None, why
    found = bs.circles_about(e, bs._centres(view.gray, view.lo, view.hi), view.lo, view.hi)
    edges = [f for f in found if abs(f[2] - s.r_px) < bs.GROOVE_FRAC * s.r_px and f[4] == bs.SUPPORT_SECTORS]
    assert sorted(f[3] for f in edges) == [-1, 1], edges
    fits = [_one_answer(e, f, f[3])[0] for f in edges]
    assert (fits[0][2] + fits[1][2]) / 2.0 == pytest.approx(36.1, abs=1.0)
    _view, e = _down_recorded("wrist_reach_regrip_flatbot_20261030_1059", recorded_pink)
    for sign in (1, -1):
        assert [bs.carried(e, (151.8 + du, 154.4 + dv, 30.9), sign) for du, dv in STARTS] == [None] * 4


# what the sight takes on 16:23: 717 fits and 843,906 measures. The bounds stand well clear of those, so
# that another build's edges pass, and well under what a fit measuring the whole picture takes
FITS, MEASURES = 1000, 1_200_000


def test_the_work_one_frame_takes_is_bounded(recorded_pink, monkeypatch):
    """16:23, the recorded frame that costs most: the sight's circle fits (each a least-squares circle
    through the pixels a fit holds) and pixel measures (each edge pixel tested against a fit) counted, not
    timed. At most FITS and MEASURES: with every pixel of the picture measured on every round of every
    candidate, the same 717 fits take 3.77 million measures."""
    counts = {"fits": 0, "measures": 0}
    fit, held = bs._Near.fit, bs._Near.held

    def counted_fit(self, on):
        counts["fits"] += 1
        return fit(self, on)

    def counted_held(self, f, band, sign):
        counts["measures"] += len(self.x)
        return held(self, f, band, sign)

    monkeypatch.setattr(bs._Near, "fit", counted_fit)
    monkeypatch.setattr(bs._Near, "held", counted_held)
    s, why = _sight_recorded("wrist_reach_aim_flatbot_20261027_1623", recorded_pink)
    assert s is not None, why
    assert counts["fits"] <= FITS and counts["measures"] <= MEASURES, counts


def test_a_seam_and_rim_in_any_family_comes_before_a_size_alone(pink):
    """which_seam by one way alone: 'rim' only a seam-and-rim pair, 'size' only the size window."""
    want = pink.button_diameter_m / pink.lid_dims[0]
    pair = [(0, 0, 40.0), (0, 0, 40.0 / want)]
    assert bs.which_seam(pair, pink, 40.0, by="rim") == ((0, 0, 40.0), "seam+rim")
    assert bs.which_seam([(0, 0, 41.0)], pink, 40.0, by="rim")[0] is None
    assert bs.which_seam(pair, pink, 40.0, by="size") == ((0, 0, 40.0), "seam")


# ---- the recorded frames (flatbot, the pink canister; .design aim-experiment E2) ----------------------------


def _recorded(name, pink):
    """(colour, k, dist, rot_cam, trans_cam, the aim in base xyz): the aim the run recorded, else the wrist
    finder's on the frame as its tests call it."""
    import cv2

    from box_opening.perception.button_aim import find_button

    z = np.load(fixture_path(name + ".npz"))
    color = cv2.imdecode(z["color_png"], cv2.IMREAD_COLOR)
    if "aimed" in z.files:
        aim = [float(v) for v in z["aimed"]]
    else:
        lid_z = float(z["lid_z"])
        table_z = float(z["table_z"]) if "table_z" in z.files else lid_z - pink.dims[2]
        depth = z["depth_mm"].astype(np.float32) / 1000.0
        prior = tuple(z["prior_xy"])
        s, why = find_button(color, depth, z["k"], z["rot_cam"], z["trans_cam"], table_z, pink, z["dist"], lid_z, prior)
        assert s is not None, why
        aim = list(s.xyz)
    return color, z["k"], z["dist"], z["rot_cam"], z["trans_cam"], aim


def _sight_recorded(name, pink):
    color, k, dist, rot, t, aim = _recorded(name, pink)
    return bs.sight_button(color, k, dist, rot, pink, (aim[0] - t[0], aim[1] - t[1]), t[2] - aim[2])


# frame: experiment E2's seam radius (px) and its xy by the picture (m)
E2 = {
    "wrist_reach_look_flatbot_20261030_1010": (29.0, (0.5913, -0.1119)),
    "wrist_reach_look_flatbot_20261027_1404": (43.6, (0.6436, -0.1733)),
    "wrist_reach_aim_flatbot_20261027_1623": (36.1, (0.7068, -0.2884)),
    "wrist_reach_aim_flatbot_20261027_1519": (33.7, (0.6483, -0.1918)),
    "wrist_reach_regrip_flatbot_20261030_1059": (30.5, (0.6124, -0.1010)),
}


# 10:10's groove shows both edges, 28.7 and 31.5 px: the seam is their mean, 30.1, and E2's 29.0 was Hough's
# radius on the inner one alone
R_TOL = {"wrist_reach_look_flatbot_20261030_1010": 1.5}
# 10:59's seam shows in 9 of 16 sectors, the fingers over the rest: too little of its round to hold a fit
# (carried), so the seam within 1 px, or None
UNDER_FINGERS = {"wrist_reach_regrip_flatbot_20261030_1059"}


@pytest.mark.parametrize("name", sorted(E2))
def test_the_seam_is_found_on_the_recorded_oblique_frames(recorded_pink, name):
    """Case 3: the five frames whose seam experiment E2 found from 25-30 deg oblique, rotated from their
    recorded lens pose, the button expected at the frame's own aim: the seam, its radius within 1 px of
    E2's (R_TOL; 10:59 the seam or None: UNDER_FINGERS). E2 did not undistort; this does, so the centres
    may differ by a few mm."""
    color, k, dist, rot, t, aim = _recorded(name, recorded_pink)
    s, why = bs.sight_button(color, k, dist, rot, recorded_pink, (aim[0] - t[0], aim[1] - t[1]), t[2] - aim[2])
    if name in UNDER_FINGERS and s is None:
        return
    assert s is not None, why
    r_e2, xy_e2 = E2[name]
    assert s.r_px == pytest.approx(r_e2, abs=R_TOL.get(name, 1.0))
    xy = (t[0] + s.off_xy[0], t[1] + s.off_xy[1])
    assert math.hypot(xy[0] - xy_e2[0], xy[1] - xy_e2[1]) < 0.006, xy


def test_a_seam_the_view_does_not_show_is_none_not_the_ring(recorded_pink):
    """Case 3: 12:59, 35 deg oblique, the seam not seen at any threshold (E2 took the 60 mm ring for it,
    143 mm out in height): None."""
    s, why = _sight_recorded("wrist_reach_look_flatbot_20261027_1259", recorded_pink)
    assert s is None, s
    assert why


def test_a_seam_under_the_fingers_is_none_not_the_rim(recorded_pink):
    """Case 3: the bench's D405 search frame of 2026-09-23, the fingers over the button: the lid's rim shows,
    the seam does not. The button expected where the frame's depth has the lid at the seam's place (458,
    409), read off the frame. None: the rim is not taken for the seam."""
    z = np.load(fixture_path("wrist_search_pink_fingers_20260923.npz"))
    k, rot, t = z["k"], z["rot_cam"], z["trans_cam"]
    u, v = 458, 409
    win = z["depth"][v - 5 : v + 6, u - 5 : u + 6].astype(float)
    depth = float(np.median(win[(win > 0.05) & (win < 2.0)]))
    p = t + rot @ (np.array([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], 1.0]) * depth)
    dist = z["dist"] if "dist" in z.files else None
    color = np.ascontiguousarray(z["color"])
    s, why = bs.sight_button(color, k, dist, rot, recorded_pink, (p[0] - t[0], p[1] - t[1]), t[2] - p[2])
    assert s is None, s


def test_the_button_cut_by_the_frames_edge_is_its_own_centre_or_none(recorded_pink):
    """Case 4: the 14:33 close-up, the seam cut by the frame's top edge. A cut arc fits more than one circle
    (Hough took a 51 px one 13 px off as well as the seam's own): only a circle wholly inside the frame
    counts, so the answer is the seam+rim at the seam's centre (308, 25) within 4 px, or None."""
    s, why = _sight_recorded("wrist_depth_look_flatbot_20261027_1433", recorded_pink)
    assert s is None or (s.how == "seam+rim" and math.hypot(s.uv[0] - 308, s.uv[1] - 25) <= 4.0), (s, why)
