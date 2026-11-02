"""The circle-only aim from staging, on the real frame that the plateau
gate refused (2026-09-17, the 4.1-inch OXO, wrist 25 cm above the lid)."""

from pathlib import Path

import numpy as np
import pytest

from box_opening.models.container import ContainerModel
from box_opening.perception.button_aim import (
    MIN_HITS,
    agreeing_median,
    circle_gates,
    find_button,
)

CFG = "capabilities/box_opening/config/containers/oxo_pop.yaml"
FRAME = Path(__file__).parent / "data" / "wrist_staging_big_box.npz"
TABLE_Z = -0.027


@pytest.fixture(scope="module")
def frame():
    z = np.load(FRAME)
    return {k: z[k] for k in ("color", "depth", "k", "rot_cam", "trans_cam")}


@pytest.fixture(scope="module")
def model():
    return ContainerModel.load(CFG)


def test_the_button_is_found_on_the_refused_staging_frame(frame, model):
    s, why = find_button(
        frame["color"], frame["depth"], frame["k"], frame["rot_cam"], frame["trans_cam"], TABLE_Z, model
    )
    assert why is None and s is not None
    # where the wrist's own search found it a minute later: [0.573, -0.022]
    assert s.xyz[0] == pytest.approx(0.574, abs=0.004)
    assert s.xyz[1] == pytest.approx(-0.026, abs=0.004)
    assert s.xyz[2] == pytest.approx(TABLE_Z + model.dims[2], abs=0.012)
    assert s.diameter_m == pytest.approx(model.button_diameter_m, rel=0.15)


def test_a_lid_corner_at_lid_height_is_not_the_button(frame, model):
    """Hough also returns the lid's rounded corners: flat, at lid height,
    and half their surround is table. The annulus gate refuses them."""
    lid_z = TABLE_Z + model.dims[2]
    why = circle_gates(frame["depth"], frame["k"], frame["rot_cam"], frame["trans_cam"], 508.0, 200.0, 33.5, lid_z)
    assert why is not None and "corner" in why
    assert (
        circle_gates(frame["depth"], frame["k"], frame["rot_cam"], frame["trans_cam"], 462.0, 274.0, 41.0, lid_z)
        is None
    )


def test_a_frame_without_the_lid_finds_nothing(frame, model):
    depth = frame["depth"].copy()
    depth[:] = 0.30  # a bare table at that range: nothing at lid height
    s, why = find_button(frame["color"], depth, frame["k"], frame["rot_cam"], frame["trans_cam"], TABLE_Z, model)
    assert s is None and "none the button" in why


def test_too_close_refuses_rather_than_aims(frame, model):
    t = np.array(frame["trans_cam"], float)
    t[2] = TABLE_Z + model.dims[2] + 0.05
    s, why = find_button(frame["color"], frame["depth"], frame["k"], frame["rot_cam"], t, TABLE_Z, model)
    assert s is None and "too close" in why


def test_hits_must_agree_before_they_aim():
    pts = [(0.5, 0.0, 0.08), (0.5, 0.01, 0.08), (0.5, 0.0, 0.08)]  # one 10 mm off
    assert agreeing_median(pts) is None
    pts = [(0.5, 0.0, 0.08)] * (MIN_HITS - 1)
    assert agreeing_median(pts) is None
    pts = [(0.500, 0.000, 0.080), (0.502, 0.001, 0.081), (0.501, -0.001, 0.080)]
    m = agreeing_median(pts)
    assert m is not None and m[0] == pytest.approx(0.501) and m[1] == pytest.approx(0.0)


def test_a_fix_is_three_in_a_row_or_a_tilted_views_newest_agreeing_pair():
    """agreeing_fix: n=3 is agreeing_median's rule, the last three. n=2 is
    the newest pair of the last AIM_WINDOW hits that agree, on the knob too
    (a hit without a knob reading agrees), else three of them that agree;
    a hit older than the window agrees with nothing."""
    from box_opening.perception.button_aim import AIM_WINDOW, agreeing_fix

    a, b = (0.5, -0.16, 0.29), (0.5, -0.15, 0.29)  # 10 mm apart
    nan = float("nan")
    assert agreeing_fix([b, a, a, a], None, 3) == (agreeing_median([b, a, a, a]), [1, 2, 3])
    assert agreeing_fix([a, a, b, a], [0.0] * 4, 3) is None  # three in a row, as ever
    assert agreeing_fix([a], [0.0], 2) is None and agreeing_fix([a, b], [0.0, 0.0], 2) is None
    assert agreeing_fix([a, a, a], [0.0, 0.0, 0.0], 2) == (a, [1, 2])  # the newest pair
    assert agreeing_fix([a, a], [nan, 0.004], 2) == (a, [0, 1])
    assert AIM_WINDOW == 4
    assert agreeing_fix([a, b, (0.6, 0.0, 0.29), (0.7, 0.0, 0.29), a], [0.0] * 5, 2) is None


def test_two_of_four_fixes_wherever_three_in_a_row_would_and_never_later():
    """The tilted aim's rule (agreeing_fix, n=2) on random streams of hits,
    three in ten up to 2 cm off, half the knob readings a shut box's and
    half anywhere from -2 to +12 mm: wherever the last three agree it has
    fixed, at that hit or sooner — two hits can only shorten an aim — its
    first fix holds the newest hit (the aim returns on it), and a pair it
    fixes on reads the knob alike."""
    import random

    from box_opening.perception.button_aim import KNOB_NOISE_M, agreeing_fix

    rng = random.Random(7)
    sooner = 0
    for _ in range(1000):
        pts, up, first = [], [], None
        for k in range(8):
            if rng.random() < 0.3:
                pts.append((0.5 + rng.uniform(-0.02, 0.02), -0.16 + rng.uniform(-0.02, 0.02), 0.29))
            else:
                pts.append((0.5 + rng.gauss(0.0, 0.0015), -0.16 + rng.gauss(0.0, 0.0015), 0.29))
            up.append(rng.gauss(0.002, 0.0015) if rng.random() < 0.5 else rng.uniform(-0.002, 0.012))
            got = agreeing_fix(pts, up, 2)
            if first is None and got is not None:
                first, idx = k, got[1]
                assert k in idx
                assert len(idx) == 3 or abs(up[idx[0]] - up[idx[1]]) <= KNOB_NOISE_M
            if agreeing_median(pts) is not None:
                assert first is not None and first <= k
                sooner += first < k
                break
    assert sooner > 100


MISSED = Path(__file__).parent / "data" / "wrist_aim_pressed_right_20260921.npz"


def test_the_press_point_is_the_centre_of_the_circle(model):
    """Bench 2026-09-21 13:35, the owner: "it's still pressing too far to
    the right of the button". This is that frame. Hough had the seam exactly
    (centre at pixel 454, 375), and the press went 13 mm up and to the right
    of it, onto the raised ring — the pads met a surface 10.8 mm above the
    button. A "dimple" detector (2026-09-17) had taken the lowest patch of
    depth inside the disc for the place a finger presses, and here that
    patch was stereo noise beside the seam. It fired on one real aim out of
    four, and that one missed; every press at the centre landed on the
    button. The owner had already said it: the gripper goes to the centre
    of the circle OpenCV segments. Nothing overrides that now."""
    z = np.load(MISSED)
    k, rot, trans = z["k"], z["rot_cam"], z["trans_cam"]

    def centre_of(s):
        pc = np.array([(s.uv[0] - k[0, 2]) / k[0, 0] * s.range_m, (s.uv[1] - k[1, 2]) / k[1, 1] * s.range_m, s.range_m])
        return tuple((rot @ pc + trans)[:2])

    s, why = find_button(z["color"], z["depth"], k, rot, trans, TABLE_Z, model)
    assert why is None and s is not None
    assert s.uv == pytest.approx((454.5, 375.5), abs=3.0)  # the seam, as logged that day (another of its 16 frames)
    assert s.xyz[:2] == pytest.approx(
        centre_of(s), abs=1e-4
    )  # the press point IS that pixel, lifted at the disc's range
    # that day's press point was [0.4463, -0.0542]: 11-13 mm from the centre
    assert np.hypot(s.xyz[0] - 0.4463, s.xyz[1] + 0.0542) > 0.010

    # The patch that fooled it came and went with the depth noise (it is not
    # in this frame of that run's 21). Put it back where it was — 4 mm low,
    # up and to the right — and the press point must not move.
    depth = z["depth"].copy()
    yy, xx = np.mgrid[0 : depth.shape[0], 0 : depth.shape[1]]
    depth[(xx - 470.7) ** 2 + (yy - 357.6) ** 2 < 9.0**2] += 0.004
    s2, why = find_button(z["color"], depth, k, rot, trans, TABLE_Z, model)
    assert why is None and s2.uv == pytest.approx(s.uv, abs=0.5)
    assert s2.xyz[:2] == pytest.approx(centre_of(s2), abs=1e-4)
    assert not hasattr(s2, "dimple")


def test_the_top_height_is_the_disc_median(frame, model):
    """The height reported for the button's top is the median over the
    disc, not the one pixel at its centre: what the closed pads meet, and
    the honest "top" for the calibration line."""
    s, why = find_button(
        frame["color"], frame["depth"], frame["k"], frame["rot_cam"], frame["trans_cam"], TABLE_Z, model
    )
    assert s.xyz[2] == pytest.approx(0.086, abs=0.003)


def test_the_aim_says_whether_the_button_already_stands_proud(model):
    """Two real aim frames of the same box, 2026-09-21. At 14:16 it was
    closed: the button's disc level with the lid ring around it. At 13:35 it
    had been left open by the run before, and the whole disc stood 8-10 mm
    above its own lid — a difference taken inside one frame, so neither the
    camera's mount nor the table's height enters it. That run pressed the
    open box shut; so did two more at 14:14. The touch cannot be trusted to
    notice (the raised knob is soft: strokes have ridden it all the way down
    under a 3 Nm guard), so the aim reports it and the mission does not
    press."""
    closed = np.load(Path(__file__).parent / "data" / "wrist_aim_closed_20260921.npz")
    s, why = find_button(
        closed["color"], closed["depth"], closed["k"], closed["rot_cam"], closed["trans_cam"], TABLE_Z, model
    )
    assert why is None and abs(s.above_lid_m) < 0.003
    opened = np.load(MISSED)
    s, why = find_button(
        opened["color"], opened["depth"], opened["k"], opened["rot_cam"], opened["trans_cam"], TABLE_Z, model
    )
    assert why is None and 0.006 < s.above_lid_m < 0.013


def test_a_knob_standing_high_is_still_the_button(model):
    """This knob has been met at +17.7 mm (2026-09-17), not only at ~10. The
    lid-height gate allowed the centre 12 mm either way — so a knob that
    high was "not the button", the aim found nothing, the mission fell back
    to the wrist search, and THAT path presses with no open-box check at
    all. Above the lid is where a popped button is: the gate lets the centre
    stand up to a knob's height above the lid, and still nothing below it."""
    from box_opening.perception.button_aim import KNOB_MAX_UP_M

    z = np.load(Path(__file__).parent / "data" / "wrist_aim_closed_20260921.npz")
    s0, _ = find_button(z["color"], z["depth"], z["k"], z["rot_cam"], z["trans_cam"], TABLE_Z, model)
    yy, xx = np.mgrid[0 : z["depth"].shape[0], 0 : z["depth"].shape[1]]
    disc = (xx - s0.uv[0]) ** 2 + (yy - s0.uv[1]) ** 2 < (0.97 * s0.r_px) ** 2
    for up in (0.017, KNOB_MAX_UP_M - 0.003):
        depth = z["depth"].copy()
        depth[disc] -= up  # the whole button that much nearer the camera
        s, why = find_button(z["color"], depth, z["k"], z["rot_cam"], z["trans_cam"], TABLE_Z, model)
        assert why is None, why
        assert s.above_lid_m == pytest.approx(up, abs=0.003)
    depth = z["depth"].copy()
    depth[disc] += 0.017  # a hole that deep is no button
    s, why = find_button(z["color"], depth, z["k"], z["rot_cam"], z["trans_cam"], TABLE_Z, model)
    assert s is None and "lid height" in why


def test_the_pink_canister_is_aimed_at_staging():
    """Bench 2026-09-23, the owner: "it isn't finding the box even though
    the box is right underneath it". This is that frame: the button in plain
    view, and the aim refused it — "ring around it is not lid (0% at lid
    height)" — because the model said the lid stood 95 mm above the table (a
    phone-photo estimate) and this frame measures 121. The aim looks for the
    lid within 12 mm of that height, so it was searching 26 mm too low; the
    wrist search's height band with it. Measured here: top 121 mm, lid
    ~105 mm across, button 44.3 mm, flush."""
    from box_opening.models.container import ContainerModel

    m = ContainerModel.load("capabilities/box_opening/config/containers/ankou_pink.yaml")
    z = np.load(Path(__file__).parent / "data" / "wrist_staging_pink_20260923.npz")
    s, why = find_button(z["color"], z["depth"], z["k"], z["rot_cam"], z["trans_cam"], TABLE_Z, m)
    assert why is None, why
    assert s.uv == pytest.approx((238.0, 390.0), abs=4.0)
    assert s.diameter_m == pytest.approx(0.044, abs=0.003)
    assert abs(s.above_lid_m) < 0.003  # closed: flush with its lid
    assert s.xyz[2] == pytest.approx(TABLE_Z + m.dims[2], abs=0.006)


# the live D405 colour stream's plumb_bob coefficients (camera_info, 2026-09-23)
D405_DIST = [-0.05412, 0.06031, 0.00018, 0.00016, -0.02035]


def test_the_press_point_is_corrected_for_lens_distortion():
    """The aim lifted the circle's RAW pixel through a pinhole K, but the
    colour stream is not rectified: near the image edge that is 1-2 mm of
    press error (1.4 mm at the pixel the 2026-09-23 miss was aimed from).
    The centre is undistorted before it becomes a ray; its depth is still
    read at the raw pixel, where the aligned depth lives."""
    import cv2

    from box_opening.models.container import ContainerModel

    m = ContainerModel.load("capabilities/box_opening/config/containers/ankou_pink.yaml")
    z = np.load(Path(__file__).parent / "data" / "wrist_staging_pink_20260923.npz")
    raw, _ = find_button(z["color"], z["depth"], z["k"], z["rot_cam"], z["trans_cam"], TABLE_Z, m)
    s, why = find_button(z["color"], z["depth"], z["k"], z["rot_cam"], z["trans_cam"], TABLE_Z, m, dist=D405_DIST)
    assert why is None and s.uv == raw.uv  # the same circle, the same pixel reported
    k = np.asarray(z["k"], float)
    u, v = cv2.undistortPoints(np.array([[raw.uv]], np.float64), k, np.array(D405_DIST), P=k)[0, 0]
    pc = np.array([(u - k[0, 2]) / k[0, 0] * s.range_m, (v - k[1, 2]) / k[1, 1] * s.range_m, s.range_m])
    want = z["rot_cam"] @ pc + z["trans_cam"]
    assert s.xyz[:2] == pytest.approx(tuple(want[:2]), abs=1e-5)
    moved = np.hypot(s.xyz[0] - raw.xyz[0], s.xyz[1] - raw.xyz[1])
    assert 0.0005 < moved < 0.003  # a millimetre or two, at this pixel


# --- the reach-out press: the button seen from a tilted hand ----------------
#
# On flatbot the box stands 0.80-0.89 m out and the hand cannot point straight
# down there: the reach-out press looks at the button from 29-39 deg off the
# lid's normal, the camera ~0.31 m above and 0.17-0.24 m behind it. These
# frames are synthetic: a round box (the 105 mm pink lid, its 44 mm button
# inside a dark seam ring) on a table 12 cm below the lid, ray-cast through an
# OAK-D Pro 640x400 pinhole for colour (3x3 supersampled, 2 grey levels of
# noise) and z-depth registered to it (2 mm noise, 1 mm steps, 1% holes).

OAK_K = np.array([[452.0, 0.0, 319.5], [0.0, 452.0, 199.5], [0.0, 0.0, 1.0]])
OAK_HW = (400, 640)
LID_R, BUTTON_R, SEAM_W, BOX_H = 0.0525, 0.022, 0.002, 0.12
FLAT_TABLE_Z = 0.30
BUTTON_XY = (0.84, 0.12)
PINK = "capabilities/box_opening/config/containers/ankou_pink.yaml"
TABLE, SIDE, LID, SEAM, BUTTON, KNOB_SIDE = range(6)
LID_BGR = (175, 150, 235)
DARK, BRIGHT = (125, 95, 205), (215, 205, 250)  # the button's top against the lid
ROT_DOWN = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])  # as the staging frames


@pytest.fixture(scope="module")
def pink():
    return ContainerModel.load(PINK)


def _h0(m):
    return FLAT_TABLE_Z + m.dims[2]


def look_at(cam, target):
    """base <- camera for a camera at `cam` looking at `target`, image x level."""
    z = np.asarray(target, float) - cam
    z /= np.linalg.norm(z)
    x = np.cross(z, (0.0, 0.0, 1.0))
    x /= np.linalg.norm(x)
    return np.c_[x, np.cross(z, x), z]


def reach_out_pose(button, incidence_deg, height, off_deg=5.0, phi_deg=30.0):
    """The camera `height` above the button and back toward the arm's base,
    the view ray `incidence_deg` off the lid's normal, the optical axis
    `off_deg` off the button."""
    b = np.asarray(button, float)
    back = -b[:2] / np.hypot(*b[:2])
    cam = b + np.r_[back * height * np.tan(np.radians(incidence_deg)), height]
    view = (b - cam) / np.linalg.norm(b - cam)
    lat = np.r_[-back[1], back[0], 0.0]
    t = np.linalg.norm(b - cam) * np.tan(np.radians(off_deg))
    phi = np.radians(phi_deg)
    return cam, look_at(cam, b + t * (np.cos(phi) * lat + np.sin(phi) * np.cross(view, lat)))


def _cylinder(o, d, c, r, z_lo, z_hi, inside=False):
    """Ray parameter of the entry into a vertical cylinder wall (inside: the
    exit, the wall of a hole seen from within), inf on a miss."""
    ox, oy = o[0] - c[0], o[1] - c[1]
    a = d[..., 0] ** 2 + d[..., 1] ** 2
    b = 2.0 * (ox * d[..., 0] + oy * d[..., 1])
    disc = b * b - 4.0 * a * (ox * ox + oy * oy - r * r)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (-b + (1.0 if inside else -1.0) * np.sqrt(disc)) / (2.0 * a)
    z = o[2] + t * d[..., 2]
    return np.where((disc > 0) & (t > 0) & (z >= z_lo) & (z <= z_hi), t, np.inf)


def _cast(cam, rot, xn, yn, boxes, table_z):
    """Camera z-depth and surface label along normalised rays. boxes:
    (xy, lid_z, up) — up is how far the button stands proud of its lid;
    below zero it is recessed, its top at the floor of a hole in the lid."""
    d = np.stack([xn, yn, np.ones_like(xn)], -1) @ rot.T  # camera z = 1: t IS the z-depth

    def plane(z, xy):
        t = (z - cam[2]) / d[..., 2]
        rho = np.hypot(cam[0] + t * d[..., 0] - xy[0], cam[1] + t * d[..., 1] - xy[1])
        return np.where(t > 0, t, np.inf), rho

    t, _ = plane(table_z, (0.0, 0.0))
    ts, labs = [t], [np.full(t.shape, TABLE)]
    for xy, lid_z, up in boxes:
        ts.append(_cylinder(cam, d, xy, LID_R, table_z, lid_z))
        labs.append(np.full(t.shape, SIDE))
        t, rho = plane(lid_z, xy)
        ts.append(np.where((rho < LID_R) & ((up >= 0) | (rho >= BUTTON_R)), t, np.inf))
        labs.append(np.where(rho < BUTTON_R, BUTTON, np.where(rho < BUTTON_R + SEAM_W, SEAM, LID)))
        if up != 0:
            t, rho = plane(lid_z + up, xy)
            ts.append(np.where(rho < BUTTON_R, t, np.inf))
            labs.append(np.full(t.shape, BUTTON))
            lo, hi = sorted((lid_z, lid_z + up))
            ts.append(_cylinder(cam, d, xy, BUTTON_R, lo, hi, inside=up < 0))
            labs.append(np.full(t.shape, KNOB_SIDE))
    ts, labs = np.stack(ts), np.stack(labs)
    i = np.argmin(ts, 0)[None]
    return np.take_along_axis(ts, i, 0)[0], np.take_along_axis(labs, i, 0)[0]


def render(cam, rot, boxes, table_z, button_bgr=DARK, dist=None, seed=0, ss=3):
    """(colour BGR uint8, depth float32 m) of the boxes seen from (cam, rot)."""
    import cv2

    rng = np.random.default_rng(seed)
    h, w = OAK_HW
    vv, uu = np.mgrid[0:h, 0:w].astype(float)
    lut = np.array([(95, 105, 110), (120, 95, 175), LID_BGR, (70, 55, 110), button_bgr, (90, 70, 150)], float)

    def rays(du, dv):
        if dist is None:
            return (uu + du - OAK_K[0, 2]) / OAK_K[0, 0], (vv + dv - OAK_K[1, 2]) / OAK_K[1, 1]
        pts = np.c_[(uu + du).ravel(), (vv + dv).ravel()].reshape(-1, 1, 2)
        n = cv2.undistortPoints(pts, OAK_K, np.asarray(dist, float)).reshape(h, w, 2)
        return n[..., 0], n[..., 1]

    col = np.zeros((h, w, 3))
    for dv in (np.arange(ss) + 0.5) / ss - 0.5:
        for du in (np.arange(ss) + 0.5) / ss - 0.5:
            col += lut[_cast(cam, rot, *rays(du, dv), boxes, table_z)[1]]
    color = np.clip(col / ss**2 + rng.normal(0.0, 2.0, col.shape), 0, 255).astype(np.uint8)
    t, _ = _cast(cam, rot, *rays(0.0, 0.0), boxes, table_z)
    depth = np.round((t + rng.normal(0.0, 0.002, t.shape)) * 1000.0) / 1000.0
    depth[rng.random(t.shape) < 0.01] = 0.0
    return color, depth.astype(np.float32)


_FRAMES = {}


def reach_out_frame(m, incidence_deg, lid_drop=0.0, up=0.0, button_bgr=DARK, dist=None, table_drop=None):
    """(colour, depth, cam, rot, true button top) for the pink box at
    BUTTON_XY, its lid `lid_drop` below the model's height (the prior), its
    table `table_drop` below FLAT_TABLE_Z (default: BOX_H under the lid)."""
    key = (incidence_deg, lid_drop, up, button_bgr, None if dist is None else tuple(dist), table_drop)
    if key not in _FRAMES:
        height = {29: 0.315, 39: 0.295}[incidence_deg]
        lid_z = _h0(m) - lid_drop
        table_z = lid_z - BOX_H if table_drop is None else FLAT_TABLE_Z - table_drop
        cam, rot = reach_out_pose((*BUTTON_XY, lid_z + up), incidence_deg, height)
        color, depth = render(cam, rot, [(BUTTON_XY, lid_z, up)], table_z, button_bgr, dist)
        _FRAMES[key] = (color, depth, cam, rot, np.r_[BUTTON_XY, lid_z + up])
    return _FRAMES[key]


def _pixel(p, cam, rot, dist=None):
    import cv2

    pc = rot.T @ (np.asarray(p, float) - cam)
    d = np.zeros(5) if dist is None else np.asarray(dist, float)
    return tuple(cv2.projectPoints(pc.reshape(1, 1, 3), np.zeros(3), np.zeros(3), OAK_K, d)[0].ravel())


def test_the_reach_out_poses_are_the_flatbot_ones(pink):
    """The synthetic camera stands where the reach-out press puts it."""
    for inc in (29, 39):
        _, _, cam, rot, b = reach_out_frame(pink, inc)
        ray = b - cam
        assert 0.29 < cam[2] - b[2] < 0.32
        assert 0.17 <= np.hypot(*ray[:2]) <= 0.24
        assert 0.355 < np.linalg.norm(ray) < 0.385
        assert np.degrees(np.arccos(-ray[2] / np.linalg.norm(ray))) == pytest.approx(inc, abs=0.5)
        assert np.degrees(np.arccos(ray @ rot[:, 2] / np.linalg.norm(ray))) <= 7.0
        assert np.hypot(*b[:2]) == pytest.approx(0.85, abs=0.05)


@pytest.mark.parametrize(
    "inc,button_bgr,with_prior",
    [(29, DARK, False), (39, DARK, False), (29, BRIGHT, True), (39, BRIGHT, True)],
)
def test_the_reach_out_aim_finds_the_button_centre(pink, inc, button_bgr, with_prior):
    """Tilted 29-39 deg the button is an ellipse and its disc is not flat in
    camera depth; rectified onto the lid's plane it is a circle again, and
    its centre is the press point within 2 mm, its height within 3."""
    color, depth, cam, rot, b = reach_out_frame(pink, inc, button_bgr=button_bgr)
    prior = tuple(b[:2]) if with_prior else None
    s, why = find_button(color, depth, OAK_K, rot, cam, FLAT_TABLE_Z, pink, prior_xy=prior)
    assert why is None, why
    assert s.oblique
    assert np.hypot(s.xyz[0] - b[0], s.xyz[1] - b[1]) < 0.002
    assert s.xyz[2] == pytest.approx(b[2], abs=0.003)
    assert s.lid_z_m == pytest.approx(b[2], abs=0.003)
    assert abs(s.above_lid_m) < 0.003
    assert s.incidence_deg == pytest.approx(inc, abs=1.0)
    assert s.range_m == pytest.approx(np.linalg.norm(b - cam), abs=0.003)
    assert 0.040 < s.diameter_m < 0.050  # the button's edge or the seam's outer one
    assert s.uv == pytest.approx(_pixel(b, cam, rot), abs=2.0)  # the RAW pixel
    assert s.r_px == pytest.approx(OAK_K[0, 0] * s.diameter_m / 2.0 / s.range_m)


@pytest.mark.parametrize("inc", [29, 39])
def test_the_reach_out_aim_measures_a_lid_below_its_prior(pink, inc):
    """The scene camera measured a lid top at ~0.399 m where table_z +
    dims.z says 0.421: 21 mm below the model. The oblique aim measures the
    lid in its own frame; the press point is not dragged to the prior."""
    color, depth, cam, rot, b = reach_out_frame(pink, inc, lid_drop=0.021)
    s, why = find_button(color, depth, OAK_K, rot, cam, FLAT_TABLE_Z, pink, prior_xy=tuple(b[:2]))
    assert why is None, why
    assert np.hypot(s.xyz[0] - b[0], s.xyz[1] - b[1]) < 0.002
    assert s.xyz[2] == pytest.approx(b[2], abs=0.003)
    assert s.lid_z_m == pytest.approx(b[2], abs=0.003)
    # the same frame with the true height as the prior: the same press point
    s2, why = find_button(color, depth, OAK_K, rot, cam, FLAT_TABLE_Z, pink, lid_z=b[2], prior_xy=tuple(b[:2]))
    assert why is None and np.hypot(s2.xyz[0] - s.xyz[0], s2.xyz[1] - s.xyz[1]) < 0.002


def _no_table_offset(*a):
    return float("nan"), float("nan")


@pytest.mark.parametrize("inc,table_drop", [(29, 0.028), (39, 0.018)])
def test_the_lid_is_judged_against_the_table_the_wrist_sees(pink, inc, table_drop, monkeypatch):
    """Flatbot 2026-10-27: since its repair the wrist reads every height
    ~20 mm low, its own table 18.1-21.6 mm below the scene camera's, and
    the live aims cleared the 35 mm prior-height gate by 2.7-13 mm. Here
    the wrist's table is 18-28 mm low and its lid 37 mm low, 9-19 mm lower
    still: judged against the prior alone it is refused, "lid ring -37 mm
    from its prior height". Against the table the frame sees, the lid is
    9-19 mm off and the button is aimed, its press point as measured. (28 mm
    seen at 39 deg is past the xy bound: test_an_offset_the_press_xy_...)"""
    from box_opening.perception import button_aim

    color, depth, cam, rot, b = reach_out_frame(pink, inc, lid_drop=0.037, table_drop=table_drop)
    s, why = find_button(color, depth, OAK_K, rot, cam, FLAT_TABLE_Z, pink, prior_xy=tuple(b[:2]))
    assert why is None, why
    assert np.hypot(s.xyz[0] - b[0], s.xyz[1] - b[1]) < 0.002
    assert s.table_offset_m == pytest.approx(-table_drop, abs=0.003) and s.table_offset_raw_m == s.table_offset_m
    assert s.xyz[2] == pytest.approx(b[2], abs=0.003) and s.lid_z_m == pytest.approx(b[2], abs=0.003)
    with monkeypatch.context() as mp:
        mp.setattr(button_aim, "_table_offset", _no_table_offset)  # as it was
        s0, why0 = find_button(color, depth, OAK_K, rot, cam, FLAT_TABLE_Z, pink, prior_xy=tuple(b[:2]))
    assert s0 is None and "lid ring -37 mm from its prior height" in why0, why0


@pytest.mark.parametrize("inc", [29, 39])
def test_a_lid_low_over_its_own_table_is_still_refused(pink, inc):
    """The lid alone 40 mm under its prior, its table where it was given: no
    common offset to take out, and that lid is no lid of this box."""
    color, depth, cam, rot, b = reach_out_frame(pink, inc, lid_drop=0.040, table_drop=0.0)
    s, why = find_button(color, depth, OAK_K, rot, cam, FLAT_TABLE_Z, pink, prior_xy=tuple(b[:2]))
    assert s is None and "lid ring -40 mm from its prior height" in why, why


def test_the_prior_cut_slides_a_circle_over_the_heights_the_gates_allow():
    """With a table offset the gates allow a lid around h0 moved by it, and
    the cut on the prior slides each circle over that band, not h0's."""
    from box_opening.perception.button_aim import LID_PRIOR_TOL_M, TOP_RESIDUAL_MAX_M, _prior_miss_m

    h0, cam = 0.42, np.array([0.60, 0.0, 0.70])
    s, c0, g = 1000.0, 100.0, np.array([0.80, 0.0, h0])  # the circle at the grid's centre, on h0 at g

    def at(h):  # that circle slid along the rays to h
        return (cam + (g - cam) * (cam[2] - h) / (cam[2] - h0))[:2]

    lowest = h0 - 0.030 - LID_PRIOR_TOL_M - TOP_RESIDUAL_MAX_M  # a lid the gates allow 30 mm under h0
    assert _prior_miss_m(c0, c0, s, c0, g, h0, cam, at(lowest), h0 - 0.030) == pytest.approx(0.0, abs=1e-9)
    assert _prior_miss_m(c0, c0, s, c0, g, h0, cam, at(lowest)) == pytest.approx(
        np.hypot(*(at(lowest) - at(h0 - LID_PRIOR_TOL_M - TOP_RESIDUAL_MAX_M))), abs=1e-9
    )


def test_the_table_offset_is_the_mode_of_the_heights_below_half_the_box():
    """_table_offset on a rectified grid the size of the live ones: the
    table is the mode of the heights below half the box — not the lid above
    that, its side spread over every height between, nor the floor past the
    table's edge. Read and taken out while its bounds hold, read but not
    taken out past them, and neither with too little table in view."""
    from box_opening.perception.button_aim import TABLE_MIN_CELLS, TABLE_SUBSAMPLE, _table_offset

    h0, table_z, n = 0.311, 0.214, 351
    rng = np.random.default_rng(3)

    def grid(table_at, keep=1.0, only=None):
        z = table_at + rng.normal(0.0, 0.002, (n, n))  # the table, 2 mm of noise
        z[:, :100] -= 0.30  # the floor past its edge, as noisy
        z[150:260, 150:260] = h0 - 0.022  # the lid, 22 mm low
        z[260:290, 150:260] = np.linspace(h0 - 0.022, table_at, 30)[:, None]  # its near side
        z[rng.random((n, n)) > keep] = np.nan  # holes
        if only is not None:  # nothing in view but an only x only patch of table
            z[:] = np.nan
            z[:only, 120 : 120 + only] = table_at
        lifted = np.full((n, n, 3), np.nan, np.float32)
        lifted[..., 2] = z
        return lifted

    off, raw = _table_offset(grid(table_z - 0.020), h0, table_z, 30.0)
    assert raw == pytest.approx(-0.020, abs=0.001) and off == raw
    off, raw = _table_offset(grid(table_z - 0.020, keep=0.4), h0, table_z, 30.0)  # holes are no height
    assert off == pytest.approx(-0.020, abs=0.001)
    off, raw = _table_offset(grid(table_z - 0.035), h0, table_z, 10.0)  # 6 mm in xy, but past TABLE_OFFSET_MAX_M
    assert np.isnan(off) and raw == pytest.approx(-0.035, abs=0.001)
    off, raw = _table_offset(grid(table_z - 0.025), h0, table_z, 45.0)  # 25 mm in xy at 45 deg
    assert np.isnan(off) and raw == pytest.approx(-0.025, abs=0.001)
    assert _table_offset(grid(table_z - 0.025), h0, table_z, 35.0)[0] == pytest.approx(-0.025, abs=0.001)  # 17.5
    side = int(np.ceil(np.sqrt(TABLE_MIN_CELLS))) * TABLE_SUBSAMPLE  # just enough table, subsampled
    assert _table_offset(grid(table_z + 0.010, only=side), h0, table_z, 30.0)[0] == pytest.approx(0.010, abs=1e-6)
    few = _table_offset(grid(table_z + 0.010, only=side - 2 * TABLE_SUBSAMPLE), h0, table_z, 30.0)
    assert np.isnan(few[0]) and np.isnan(few[1])


def test_an_offset_the_press_xy_cannot_carry_judges_the_lid_but_does_not_raise_the_fix(pink, monkeypatch):
    """What a common offset cannot say is whether the wrist's depth is off
    (the press xy, slid along the rays by the measured heights, then off by
    tan(incidence) x the offset) or its pose in TF (the xy right). Past
    TABLE_OFFSET_XY_MAX_M of that the offset is not taken out: the fix's z
    is not raised by it. The lid is still judged against the table the
    frame sees (owner 2026-10-30: the scene camera gives a general location,
    the wrist finds the lid): a depth that reads 45 mm long at 39 deg puts
    the table 29 mm low, and the button is found where the wrist sees it —
    29 mm off its centre here, past its 22 mm radius: the wrist's
    calibration keeps that right, not this gate. 10-28..10-30's wrist read
    the lid 12-25 mm low with its xy within 10 mm (the fitted mount shift,
    not its depth)."""
    color, depth, cam, rot, b = reach_out_frame(pink, 39)
    biased = np.where(depth > 0, depth + 0.045, 0.0).astype(np.float32)
    s, why = find_button(color, biased, OAK_K, rot, cam, FLAT_TABLE_Z, pink, prior_xy=tuple(b[:2]))
    assert why is None, why
    assert not np.isfinite(s.table_offset_m) and s.table_offset_raw_m == pytest.approx(-0.029, abs=0.003)
    assert np.hypot(s.xyz[0] - b[0], s.xyz[1] - b[1]) > BUTTON_R
    # the 29 deg case's 28 mm seen at 39 deg (42.8 at the prior, 37 mm above
    # the lid): 0.93 x 28 = 26 mm, not taken out, the lid judged against it
    color, depth, cam, rot, b = reach_out_frame(pink, 39, lid_drop=0.037, table_drop=0.028)
    s, why = find_button(color, depth, OAK_K, rot, cam, FLAT_TABLE_Z, pink, prior_xy=tuple(b[:2]))
    assert why is None, why
    assert not np.isfinite(s.table_offset_m) and s.table_offset_raw_m == pytest.approx(-0.028, abs=0.003)


@pytest.mark.parametrize(
    "inc,lid_drop,off",
    [
        (39, 0.0, (0.10, 0.0)),
        (39, 0.0, (-0.10, 0.0)),
        (29, 0.0, (0.0, 0.10)),
        (29, 0.0, (0.0, -0.10)),
        (39, 0.021, (0.0707, -0.0707)),
        (29, 0.021, (-0.085, 0.06)),
    ],
)
def test_the_reach_out_aim_survives_a_prior_10cm_off(pink, inc, lid_drop, off):
    """The scene camera's xy for the button may be 12 cm off: the rectified
    view around the prior still holds the button and its lid ring."""
    color, depth, cam, rot, b = reach_out_frame(pink, inc, lid_drop=lid_drop)
    prior = (b[0] + off[0], b[1] + off[1])
    s, why = find_button(color, depth, OAK_K, rot, cam, FLAT_TABLE_Z, pink, prior_xy=prior)
    assert why is None, why
    assert np.hypot(s.xyz[0] - b[0], s.xyz[1] - b[1]) < 0.002
    assert s.xyz[2] == pytest.approx(b[2], abs=0.003)


def test_a_circle_beyond_the_prior_range_is_not_the_button(pink):
    """17 cm off diagonally the button is still inside the rectified view,
    and farther from the prior than any scene error: refused."""
    from box_opening.perception.button_aim import OBLIQUE_HALF_M, PRIOR_MAX_M

    color, depth, cam, rot, b = reach_out_frame(pink, 39)
    off = 0.12
    assert np.hypot(off, off) > PRIOR_MAX_M and off + 1.5 * BUTTON_R < OBLIQUE_HALF_M
    s, why = find_button(color, depth, OAK_K, rot, cam, FLAT_TABLE_Z, pink, prior_xy=(b[0] - off, b[1] + off))
    assert s is None and "from the prior" in why
    # and 14 cm off, inside the range, it is found
    s, why = find_button(color, depth, OAK_K, rot, cam, FLAT_TABLE_Z, pink, prior_xy=(b[0] - 0.099, b[1] + 0.099))
    assert why is None and np.hypot(s.xyz[0] - b[0], s.xyz[1] - b[1]) < 0.002


@pytest.mark.parametrize("inc,up", [(29, 0.012), (39, 0.012), (29, 0.020), (39, 0.020)])
def test_a_popped_knob_reads_proud_from_the_reach_out_view(pink, inc, up):
    """Popped, the knob's top stands 12 mm above its lid (met at up to 17.7)
    and its side wall shows on the near side. The circle is its top; the aim
    must still see it, and say it stands proud, or the mission presses an
    open box shut. The knob hides the lid ring's far side from these rays;
    that is not a corner."""
    color, depth, cam, rot, b = reach_out_frame(pink, inc, up=up)
    s, why = find_button(color, depth, OAK_K, rot, cam, FLAT_TABLE_Z, pink, prior_xy=tuple(b[:2]))
    assert why is None, why
    assert s.above_lid_m == pytest.approx(up, abs=0.003)
    assert s.lid_z_m == pytest.approx(_h0(pink), abs=0.003)
    assert s.xyz[2] == pytest.approx(b[2], abs=0.003)
    assert np.hypot(s.xyz[0] - b[0], s.xyz[1] - b[1]) < 0.002


def test_a_proud_knob_is_centred_whichever_of_its_circles_hough_takes(pink, monkeypatch):
    """A proud knob shows two horizontal circles, its top's edge and its
    base on the lid, and Hough takes either. Each meets the rays at its own
    height: tan(incidence) x the knob's height apart in xy, 6-14 mm here.
    With the knob's top the lid's own colour its edge shows only against
    the side wall, the base has the seam too, and Hough takes the base; slid
    to the top's height that put the press point 5-8 mm off, or the lid ring
    onto the knob and the knob unseen. The centroid of the top's own points
    says which circle it is."""
    from box_opening.perception import button_aim

    h0 = _h0(pink)
    wrong = 0
    for inc, up in ((29, 0.012), (39, 0.010), (39, 0.0177)):
        height = {29: 0.315, 39: 0.295}[inc]
        cam, rot = reach_out_pose((*BUTTON_XY, h0 + up), inc, height)
        color, depth = render(cam, rot, [(BUTTON_XY, h0, up)], h0 - BOX_H, LID_BGR)
        s, why = find_button(color, depth, OAK_K, rot, cam, FLAT_TABLE_Z, pink, prior_xy=BUTTON_XY)
        assert why is None, why
        assert np.hypot(s.xyz[0] - BUTTON_XY[0], s.xyz[1] - BUTTON_XY[1]) < 0.002
        assert s.above_lid_m == pytest.approx(up, abs=0.003)
        with monkeypatch.context() as mp:
            mp.setattr(button_aim, "KNOB_SPLIT_M", 1.0)  # every circle taken for the top's edge
            mp.setattr(button_aim, "KNOB_NOISE_M", 1.0)
            s, why = find_button(color, depth, OAK_K, rot, cam, FLAT_TABLE_Z, pink, prior_xy=BUTTON_XY)
        wrong += s is None or np.hypot(s.xyz[0] - BUTTON_XY[0], s.xyz[1] - BUTTON_XY[1]) > 0.005
    assert wrong >= 2  # these views do take the base


def test_a_knob_at_its_full_height_is_still_seen_obliquely(pink):
    from box_opening.perception.button_aim import KNOB_MAX_UP_M

    up = KNOB_MAX_UP_M - 0.003
    color, depth, cam, rot, b = reach_out_frame(pink, 39, up=up)
    s, why = find_button(color, depth, OAK_K, rot, cam, FLAT_TABLE_Z, pink, prior_xy=tuple(b[:2]))
    assert why is None, why
    assert s.above_lid_m == pytest.approx(up, abs=0.003)
    assert np.hypot(s.xyz[0] - b[0], s.xyz[1] - b[1]) < 0.002


def test_the_reach_out_aim_takes_the_box_at_the_prior(pink):
    """Two boxes 13 cm apart, both buttons in view: the one the prior points
    at is the one aimed at, with the prior 4 cm off toward the other."""
    h0 = _h0(pink)
    a, b = np.r_[BUTTON_XY[0], BUTTON_XY[1] - 0.065, h0], np.r_[BUTTON_XY[0], BUTTON_XY[1] + 0.065, h0]
    cam, rot = reach_out_pose(np.r_[BUTTON_XY, h0], 35, 0.30, off_deg=0.0)
    color, depth = render(cam, rot, [(tuple(a[:2]), h0, 0.0), (tuple(b[:2]), h0, 0.0)], h0 - BOX_H)
    for want, other in ((a, b), (b, a)):
        prior = tuple(want[:2] + 0.3 * (other[:2] - want[:2]))
        s, why = find_button(color, depth, OAK_K, rot, cam, FLAT_TABLE_Z, pink, prior_xy=prior)
        assert why is None, why
        assert np.hypot(s.xyz[0] - want[0], s.xyz[1] - want[1]) < 0.002


def test_the_reach_out_aim_corrects_for_lens_distortion(pink):
    """A distorted stream (the D405's plumb_bob): the rectifying maps carry
    the distortion, each depth sample is lifted along its undistorted ray,
    and the press point's pixel is the distorted one."""
    color, depth, cam, rot, b = reach_out_frame(pink, 39, dist=tuple(D405_DIST))
    s, why = find_button(color, depth, OAK_K, rot, cam, FLAT_TABLE_Z, pink, dist=D405_DIST, prior_xy=tuple(b[:2]))
    assert why is None, why
    assert np.hypot(s.xyz[0] - b[0], s.xyz[1] - b[1]) < 0.002
    assert s.xyz[2] == pytest.approx(b[2], abs=0.003)
    assert s.uv == pytest.approx(_pixel(b, cam, rot, D405_DIST), abs=2.0)


def test_the_reach_out_aim_refuses_honestly(pink):
    color, depth, cam, rot, b = reach_out_frame(pink, 39)

    def aim(c=color, d=depth, r=rot, t=cam, **kw):
        return find_button(c, d, OAK_K, r, t, FLAT_TABLE_Z, pink, **kw)

    s, why = aim(prior_xy=(b[0] + 0.6, b[1]))  # beyond the top of the frame
    assert s is None and "outside the image" in why
    s, why = aim(prior_xy=(b[0] - 0.6, b[1]))  # back past the camera
    assert s is None and "outside the image" in why
    s, why = aim(d=np.zeros_like(depth), prior_xy=tuple(b[:2]))
    assert s is None and "no depth" in why
    holed = depth.copy()
    holed[np.random.default_rng(1).random(depth.shape) < 0.7] = 0.0
    s, why = aim(d=holed, prior_xy=tuple(b[:2]))
    assert s is None and "too little depth" in why
    far = np.where(depth > 0, depth + 0.08, 0.0).astype(np.float32)  # every surface 6 cm low
    # its table too: the lid is judged against it, but no offset that large
    # raises the fix (TABLE_OFFSET_MAX_M); a "table" past TABLE_GATE_MAX_M is
    # none, and the lid is judged against the one given
    s, why = aim(d=far, prior_xy=tuple(b[:2]))
    assert why is None and not np.isfinite(s.table_offset_m), why
    from box_opening.perception import button_aim

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(button_aim, "TABLE_GATE_MAX_M", 0.05)
        s, why = aim(d=far, prior_xy=tuple(b[:2]))
    assert s is None and "none the button" in why and "prior height" in why and "too far to be the table" in why
    # 55 deg at the button: judged there, after the circle is found
    cam55, rot55 = reach_out_pose(b, 55, 0.20, off_deg=0.0)
    c55, d55 = render(cam55, rot55, [(BUTTON_XY, b[2], 0.0)], b[2] - BOX_H)
    s, why = aim(c=c55, d=d55, r=rot55, t=cam55, prior_xy=tuple(b[:2]))
    assert s is None and "seen 55 deg off its normal — too oblique" in why
    # 70 at the prior: past any sense, refused before the rectification
    cam70, rot70 = reach_out_pose(b, 70, 0.12, off_deg=0.0)
    s, why = aim(r=rot70, t=cam70, prior_xy=tuple(b[:2]))
    assert s is None and "too oblique to aim" in why
    cam6, rot6 = reach_out_pose(b, 30, 0.06, off_deg=0.0)
    s, why = aim(r=rot6, t=cam6, prior_xy=tuple(b[:2]))
    assert s is None and "too close" in why
    color0, depth0 = render(cam, rot, [], b[2] - BOX_H)  # the bare table
    s, why = aim(c=color0, d=depth0, prior_xy=tuple(b[:2]))
    assert s is None and "no circle of" in why


def test_a_prior_outward_is_judged_at_the_button(pink):
    """A prior 10 cm out along the view is seen 51 deg off the lid's normal
    where the button, 21 mm under its prior height, is seen at 39. The
    50 deg limit is the button's: the prior's angle only bounds sanity."""
    from box_opening.perception.button_aim import OBLIQUE_MAX_INCIDENCE_DEG, _incidence_deg

    color, depth, cam, rot, b = reach_out_frame(pink, 39, lid_drop=0.021)
    out = (b[:2] - cam[:2]) / np.hypot(*(b[:2] - cam[:2]))
    prior = tuple(b[:2] + 0.10 * out)
    assert _incidence_deg(np.r_[prior, _h0(pink)], cam) > OBLIQUE_MAX_INCIDENCE_DEG
    s, why = find_button(color, depth, OAK_K, rot, cam, FLAT_TABLE_Z, pink, prior_xy=prior)
    assert why is None, why
    assert np.hypot(s.xyz[0] - b[0], s.xyz[1] - b[1]) < 0.002
    assert s.incidence_deg == pytest.approx(39, abs=1.0)


def test_the_prior_cut_allows_for_a_lid_above_its_prior(pink):
    """The lid 30 mm ABOVE its prior height and the button 14.5 cm from the
    prior, inside PRIOR_MAX_M. On h0 the rays meet the plane farther out than
    the lid, and there the circle is 16 cm from the prior; cut there, the
    button was never tried."""
    from box_opening.perception.button_aim import PRIOR_MAX_M

    color, depth, cam, rot, b = reach_out_frame(pink, 39, lid_drop=-0.030)
    a = np.radians(130.0)
    prior = b[:2] + 0.145 * np.r_[np.cos(a), np.sin(a)]
    h0 = _h0(pink)
    on_h0 = cam[:2] + (b[:2] - cam[:2]) * (cam[2] - h0) / (cam[2] - b[2])
    assert np.hypot(*(on_h0 - prior)) > PRIOR_MAX_M > np.hypot(*(b[:2] - prior))
    s, why = find_button(color, depth, OAK_K, rot, cam, FLAT_TABLE_Z, pink, prior_xy=tuple(prior))
    assert why is None, why
    assert np.hypot(s.xyz[0] - b[0], s.xyz[1] - b[1]) < 0.002
    assert s.lid_z_m == pytest.approx(b[2], abs=0.003)


@pytest.mark.parametrize("inc", [29, 39])
def test_a_circle_on_the_lids_far_rim_is_not_the_button(pink, inc):
    """The oblique twin of test_a_lid_corner_at_lid_height_is_not_the_button:
    the colour of the button's frame, the depth of one with the lid 38 mm
    nearer the camera, so the circle straddles the lid's FAR rim. From 29-39
    deg the box hides the table just past that rim, so every sample whose
    own xy is in the annulus is lid and that ring read 100% lid; the rays
    through the annulus' far part went past the lid to the table."""
    color, depth, cam, rot, b = reach_out_frame(pink, inc)
    toward = (cam[:2] - b[:2]) / np.hypot(*(cam[:2] - b[:2]))
    for off, refused in ((0.038, True), (0.032, True), (0.0, False)):
        lid = tuple(b[:2] + off * toward)
        _, moved = render(cam, rot, [(lid, b[2], 0.0)], b[2] - BOX_H)
        s, why = find_button(color, moved, OAK_K, rot, cam, FLAT_TABLE_Z, pink, prior_xy=tuple(b[:2]))
        if refused:
            assert s is None and "corner" in why, (off, why)
        else:
            assert why is None and np.hypot(s.xyz[0] - b[0], s.xyz[1] - b[1]) < 0.002


@pytest.mark.parametrize("inc", [29, 39])
def test_a_recessed_button_is_centred_at_its_rim(pink, inc):
    """Pressed below its lid the button sits at the floor of a hole. The
    circle is the hole's rim, on the lid; the near rim hides the floor's near
    part, so the floor's points are its far part, and centred at the floor's
    height the press point went tan(incidence) x the depth away from the
    camera: 6-8 mm at -10 mm. It is centred at the rim's height."""
    color, depth, cam, rot, b = reach_out_frame(pink, inc, up=-0.010)
    s, why = find_button(color, depth, OAK_K, rot, cam, FLAT_TABLE_Z, pink, prior_xy=tuple(b[:2]))
    assert why is None, why
    assert np.hypot(s.xyz[0] - b[0], s.xyz[1] - b[1]) < 0.002
    assert s.above_lid_m == pytest.approx(-0.010, abs=0.003)
    assert s.lid_z_m == pytest.approx(_h0(pink), abs=0.003)


@pytest.mark.parametrize("button_bgr", [DARK, LID_BGR])
def test_a_knob_a_few_mm_proud_is_centred_halfway(pink, button_bgr):
    """4 mm proud the knob's top and base are two circles 3 mm apart along
    the rays at 39 deg, and its top's points are not out of the depth noise:
    which one Hough took is unknown. Halfway between them is off by half
    that at most; either one alone was 2.6 mm off here."""
    color, depth, cam, rot, b = reach_out_frame(pink, 39, up=0.004, button_bgr=button_bgr)
    s, why = find_button(color, depth, OAK_K, rot, cam, FLAT_TABLE_Z, pink, prior_xy=tuple(b[:2]))
    assert why is None, why
    assert np.hypot(s.xyz[0] - b[0], s.xyz[1] - b[1]) < 0.002
    assert s.above_lid_m == pytest.approx(0.004, abs=0.003)


def _hits(depth, cam, rot):
    """Base points of every pixel's depth (pinhole), nan without depth."""
    vv, uu = np.mgrid[0 : depth.shape[0], 0 : depth.shape[1]].astype(float)
    d = np.where(depth > 0, depth, np.nan).astype(float)
    pc = np.stack([(uu - OAK_K[0, 2]) / OAK_K[0, 0] * d, (vv - OAK_K[1, 2]) / OAK_K[1, 1] * d, d], -1)
    return pc @ rot.T + cam


def _not_a_button(m, case):
    """(colour, depth, cam, rot, button) of a 39 deg frame (29 for "hole_29")
    that one oblique gate alone must refuse."""
    inc, up = {"hole_29": (29, -0.017), "hole_39": (39, -0.017), "knob_30mm": (39, 0.030)}.get(case, (39, 0.0))
    color, depth, cam, rot, b = reach_out_frame(m, inc, up=up)
    p = _hits(depth, cam, rot)
    with np.errstate(invalid="ignore"):
        rho = np.hypot(p[..., 0] - b[0], p[..., 1] - b[1])
        lid = np.abs(p[..., 2] - b[2]) < 0.005
    rnd = np.random.default_rng(2).random(depth.shape)
    depth = depth.copy()
    if case == "stepped":  # half the disc 15 mm up: a step, not a top
        depth[(rho < BUTTON_R - 0.002) & lid & (p[..., 1] > b[1])] -= 0.020
    elif case == "disc_holed":  # a tenth of the disc with depth
        depth[(rho < BUTTON_R - 0.002) & lid & (rnd < 0.90)] = 0.0
    elif case == "ring_holed":  # an eighth of the ring around it
        depth[(rho > 1.05 * BUTTON_R) & (rho < 1.65 * BUTTON_R) & lid & (rnd < 0.85)] = 0.0
    return color, depth, cam, rot, b


@pytest.mark.parametrize(
    "case,want",
    [
        ("hole_29", "from the lid around it"),  # a hole 17 mm deep is no button
        ("hole_39", "from the lid around it"),
        ("knob_30mm", "from the lid around it"),  # nor a knob 30 mm proud
        ("stepped", "not flat in base z"),
        ("disc_holed", "too little depth on the disc"),
        ("ring_holed", "too little depth around the circle"),
    ],
)
def test_each_reach_out_gate_refuses_what_is_not_a_button(pink, case, want):
    """Each gate of the oblique aim alone: a disc 12 mm below to 25 mm above
    its ring, flat in base z, and enough depth on the disc and on the ring."""
    color, depth, cam, rot, b = _not_a_button(pink, case)
    s, why = find_button(color, depth, OAK_K, rot, cam, FLAT_TABLE_Z, pink, prior_xy=tuple(b[:2]))
    assert s is None and want in why, why


def test_a_button_with_depth_on_a_quarter_of_it_is_still_the_button(pink):
    """Flatbot 2026-10-27 14:04: the glossy button seen at a slant held depth
    on 9-24 % of its disc, the lid ring around it on 39 %, and the aim
    refused it in all 31 frames. With the ring one lid at its height and the
    disc's own patches flat on it, a quarter of the disc is enough."""
    color, depth, cam, rot, b = reach_out_frame(pink, 39)
    p = _hits(depth, cam, rot)
    with np.errstate(invalid="ignore"):
        rho = np.hypot(p[..., 0] - b[0], p[..., 1] - b[1])
        lid = np.abs(p[..., 2] - b[2]) < 0.005
    depth = depth.copy()
    depth[(rho < BUTTON_R - 0.002) & lid & (np.random.default_rng(2).random(depth.shape) < 0.75)] = 0.0
    s, why = find_button(color, depth, OAK_K, rot, cam, FLAT_TABLE_Z, pink, prior_xy=tuple(b[:2]))
    assert s is not None, why
    assert np.hypot(s.xyz[0] - b[0], s.xyz[1] - b[1]) < 0.003 and abs(s.xyz[2] - b[2]) < 0.003
    assert abs(s.above_lid_m) < 0.003 and 0.15 < s.disc_fill < 0.4


REACH_1404 = Path(__file__).parent / "data" / "wrist_reach_look_flatbot_20261027_1404.npz"


def test_the_button_refused_at_1404_for_too_little_depth_is_aimed(pink):
    """Flatbot 2026-10-27 14:04, the reach-out look: tilted 30 deg, the
    camera 0.25 m off the lid, the scene camera placed by its calibration.
    The button's circle was found in 31 frames and refused in every one for
    depth on 9-24 % of its disc. This frame (23 %) aims the button's centre
    in the picture, flush with its lid: the box is shut."""
    import cv2

    f = np.load(REACH_1404)
    color = cv2.imdecode(f["color_png"], cv2.IMREAD_COLOR)
    depth = f["depth_mm"].astype(np.float32) / 1000.0
    lid_z, prior = float(f["lid_z"]), tuple(float(v) for v in f["prior_xy"])
    s, why = find_button(
        color,
        depth,
        f["k"],
        f["rot_cam"],
        f["trans_cam"],
        lid_z - pink.dims[2],
        pink,
        dist=f["dist"],
        lid_z=lid_z,
        prior_xy=prior,
    )
    assert s is not None and s.oblique, why
    assert s.uv == pytest.approx((262.7, 194.1), abs=3.0)
    assert abs(s.above_lid_m) < 0.003
    assert 0.15 <= s.disc_fill < 0.5
    assert np.hypot(s.xyz[0] - prior[0], s.xyz[1] - prior[1]) < 0.05  # the scene camera's box, 3 cm off


def _reach_frame(name):
    """A flatbot reach-out frame fixture: (colour, depth, k, rot, cam, dist,
    table_z, lid_z, prior_xy) — the table, lid and button the aim was given
    (the run's scene_fix and reach_plan notes)."""
    import cv2

    f = np.load(Path(__file__).parent / "data" / name)
    return (
        cv2.imdecode(f["color_png"], cv2.IMREAD_COLOR),
        f["depth_mm"].astype(np.float32) / 1000.0,
        f["k"],
        f["rot_cam"],
        f["trans_cam"],
        f["dist"],
        float(f["table_z"]),
        float(f["lid_z"]),
        tuple(float(v) for v in f["prior_xy"]),
    )


@pytest.mark.parametrize(
    "name,want",
    [
        ("wrist_reach_aim_flatbot_20261027_1519.npz", -0.0216),
        ("wrist_reach_aim_flatbot_20261027_1623.npz", -0.0181),
    ],
)
def test_the_live_reach_out_aims_read_their_table_20_mm_low(pink, name, want, monkeypatch):
    """Flatbot 2026-10-27 15:19 and 16:23, the aim frames from the look pose.
    The wrist read its own table 21.6 and 18.1 mm below the scene camera's
    and its lid rings 22 mm below the scene's lid, 12-13 mm inside the 35
    mm gate; judged against the table they see, 31-35 mm inside it. The
    press point stays as measured, the frame rectified again a box's height
    above that table moving it 2 mm at most. The aimer's fix stands it that much
    higher: 0.310, where the 15:19 touch met the lid at 0.3122 and the
    logged aims said 0.2881 and 0.2914."""
    from types import SimpleNamespace

    from box_opening.perception import button_aim, depth_source

    color, depth, k, rot, cam, dist, table_z, lid_z, prior = _reach_frame(name)
    s, why = find_button(color, depth, k, rot, cam, table_z, pink, dist=dist, lid_z=lid_z, prior_xy=prior)
    assert s is not None and s.oblique, why
    assert s.table_offset_m == pytest.approx(want, abs=0.003) and s.table_offset_raw_m == s.table_offset_m
    assert abs(s.lid_z_m - lid_z) > 0.020  # inside the gate by under 15 mm, as it was
    assert abs(s.lid_z_m - (lid_z + s.table_offset_m)) < 0.006  # by 29 mm or more now
    with monkeypatch.context() as mp:
        mp.setattr(button_aim, "_table_offset", _no_table_offset)
        s0, why0 = find_button(color, depth, k, rot, cam, table_z, pink, dist=dist, lid_z=lid_z, prior_xy=prior)
    # the gates moved; the press point only by the frame rectified again on that plane (TABLE_RECTIFY_M)
    assert s0 is not None and np.hypot(s0.xyz[0] - s.xyz[0], s0.xyz[1] - s.xyz[1]) < 0.003, why0
    # the aimer on MIN_HITS still frames of it: the fix raised by the offset they agree on
    monkeypatch.setattr(depth_source, "camera_pose_at", lambda g, offset_s=None, stamp=None: (rot, cam))
    g = SimpleNamespace(color=color, depth=depth, k=k, dist=dist, color_stamp=None)
    aimer = button_aim.ButtonAimer(g, pink, table_z, lid_z=lid_z, prior_xy=prior)
    for i in range(1, button_aim.MIN_HITS + 1):
        g.color_stamp = SimpleNamespace(sec=i, nanosec=0)
        fixed = aimer.tick()
    assert fixed, aimer.status()
    assert aimer.table_offset_m() == pytest.approx(s.table_offset_m, abs=1e-6)
    assert aimer.fix(measured=True) == pytest.approx(s.xyz, abs=1e-6)
    fx = aimer.fix()
    assert fx[:2] == pytest.approx(s.xyz[:2], abs=1e-6) and fx[2] == pytest.approx(0.310, abs=0.005)


def test_the_1259_look_with_the_scene_camera_9cm_high_finds_the_button_the_wrist_sees(pink):
    """Flatbot 2026-10-27 12:59, a reach-out look while the scene camera read
    9 cm high: its table at 0.303 and its lid at 0.406, the wrist's table
    106 mm below that, and the look was refused. The lid is judged against
    the table the wrist sees (owner 2026-10-30): the button is found, a
    box's height above that table; no offset that large raises the fix's z.
    A "table" past TABLE_GATE_MAX_M is still refused, and says so."""
    import re

    from box_opening.perception import button_aim

    color, depth, k, rot, cam, dist, table_z, lid_z, prior = _reach_frame("wrist_reach_look_flatbot_20261027_1259.npz")
    s, why = find_button(color, depth, k, rot, cam, table_z, pink, dist=dist, lid_z=lid_z, prior_xy=prior)
    assert why is None, why
    assert s.table_offset_raw_m == pytest.approx(-0.106, abs=0.003) and not np.isfinite(s.table_offset_m)
    assert s.xyz[2] == pytest.approx(lid_z - 0.106, abs=0.015)  # the lid as the wrist reads it, not raised
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(button_aim, "TABLE_GATE_MAX_M", 0.10)
        s, why = find_button(color, depth, k, rot, cam, table_z, pink, dist=dist, lid_z=lid_z, prior_xy=prior)
    assert s is None and "prior height" in why, why
    got = re.search(r"the table in view ([+-]\d+) mm from the one given, too far to be the table", why)
    assert got is not None, why
    assert int(got.group(1)) == pytest.approx(-106, abs=3)


def test_the_1010_look_with_the_wrist_55mm_below_the_scene_camera_finds_the_button(pink):
    """Flatbot 2026-10-30 10:10 (and 10:09): the wrist's table 55 mm below
    the scene camera's, past TABLE_OFFSET_MAX_M, and its two circles were
    refused — "disc -52 mm from the lid's prior height" — so nothing was
    pressed. Judged against the table the wrist sees, the button is found
    a box's height above it, flush with its lid; the fix's z is not
    raised."""
    color, depth, k, rot, cam, dist, table_z, lid_z, prior = _reach_frame("wrist_reach_look_flatbot_20261030_1010.npz")
    s, why = find_button(color, depth, k, rot, cam, table_z, pink, dist=dist, lid_z=lid_z, prior_xy=prior)
    assert why is None, why
    assert s.table_offset_raw_m == pytest.approx(-0.055, abs=0.003) and not np.isfinite(s.table_offset_m)
    assert abs(s.above_lid_m) < 0.003 and s.xyz[2] == pytest.approx(lid_z - 0.052, abs=0.006)
    assert np.hypot(s.xyz[0] - prior[0], s.xyz[1] - prior[1]) < 0.035  # near where the scene camera put it


def test_the_1059_regrip_rectified_on_the_wrists_own_lid_finds_the_knob_not_the_lids_ring(pink, monkeypatch):
    """Flatbot 2026-10-30 10:58: after a close on air, the re-aim from the
    look pose, the scene camera's lid 59 mm above the wrist's. Rectified
    on the scene camera's lid the 44 mm knob read ~17% small, near the
    radius window's floor, and the lid's ring (61 mm) was taken for it, 26
    mm off. Rectified again a box's height above the table the wrist sees
    (TABLE_RECTIFY_M), the circle is the knob's size, up, and ~19 mm from
    where the first grip went."""
    from box_opening.perception import button_aim

    color, depth, k, rot, cam, dist, table_z, lid_z, prior = _reach_frame(
        "wrist_reach_regrip_flatbot_20261030_1059.npz"
    )
    s, why = find_button(color, depth, k, rot, cam, table_z, pink, dist=dist, lid_z=lid_z, prior_xy=prior)
    assert why is None, why
    assert s.diameter_m == pytest.approx(0.049, abs=0.004) and s.above_lid_m > 0.004
    assert s.table_offset_raw_m == pytest.approx(-0.059, abs=0.003)
    assert np.hypot(s.xyz[0] - prior[0], s.xyz[1] - prior[1]) == pytest.approx(0.0205, abs=0.004)
    with monkeypatch.context() as mp:  # as it was: one rectification, on the scene camera's lid
        mp.setattr(button_aim, "TABLE_RECTIFY_M", 1.0)
        s0, why0 = find_button(color, depth, k, rot, cam, table_z, pink, dist=dist, lid_z=lid_z, prior_xy=prior)
    assert why0 is None and s0.diameter_m > 0.058, why0


def test_still_frames_fuse_their_depth_and_a_moved_camera_starts_again():
    """A tilted view's aim judges each frame on the depth of the still frames
    before it too: holes in one frame are filled from the others, per pixel,
    while the camera stands where the first was taken."""
    from box_opening.perception.button_aim import StillDepth

    pose = (np.eye(3), np.array([0.5, -0.1, 0.5]))
    a = np.full((4, 4), 0.30, np.float32)
    b = a.copy()
    a[:, :2] = 0.0  # each frame misses a different half
    b[:, 2:] = 0.0
    sd = StillDepth()
    assert np.array_equal(sd.add(pose, a), a)
    fused = sd.add(pose, b)
    assert (fused == pytest.approx(0.30)) and fused.min() > 0  # no hole left
    moved = (np.eye(3), np.array([0.5, -0.1, 0.55]))
    assert np.array_equal(sd.add(moved, b), b)  # from a new place, only its own frame
    assert len([sd.add(pose, a) for _ in range(9)]) and len(sd.depths) == sd.n


def test_the_window_changes_no_result(pink, monkeypatch):
    """Each candidate reads only the cells within OBLIQUE_WINDOW radii of
    its circle (5x fewer than the whole rectified grid); the sightings are
    the ones the whole grid gives."""
    from dataclasses import astuple

    from box_opening.perception import button_aim

    cases = [
        (29, 0.0, 0.0, None),
        (39, 0.021, 0.0, (0.10, 0.0)),
        (39, 0.0, 0.012, None),
        (29, 0.0, 0.020, None),
        (39, 0.0, 0.022, None),
        (39, 0.0, -0.010, None),
        (39, 0.0, 0.004, None),
    ]
    for inc, drop, up, off in cases:
        color, depth, cam, rot, b = reach_out_frame(pink, inc, lid_drop=drop, up=up)
        prior = tuple(b[:2] + (off or (0.0, 0.0)))
        s, why = find_button(color, depth, OAK_K, rot, cam, FLAT_TABLE_Z, pink, prior_xy=prior)
        with monkeypatch.context() as mp:
            mp.setattr(button_aim, "OBLIQUE_WINDOW", 1e3)
            s2, why2 = find_button(color, depth, OAK_K, rot, cam, FLAT_TABLE_Z, pink, prior_xy=prior)
        assert why is None and why2 is None, (why, why2)
        flat = [np.hstack([np.asarray(v, float).ravel() for v in astuple(x)]) for x in (s, s2)]
        np.testing.assert_allclose(flat[0], flat[1], atol=1e-9)


# find_button on the staging frames before the oblique path existed (59694a1):
# xyz, uv, r_px, range_m, diameter_m, above_lid_m
TOP_DOWN_BEFORE = [
    (
        "wrist_staging_big_box",
        CFG,
        False,
        (
            (0.5735663741667362, -0.026226748293923137, 0.08587350351853729),
            (461.5, 274.5),
            41.0,
            0.25,
            0.04713901020036241,
            -0.00100206971691226,
        ),
    ),
    (
        "wrist_aim_closed_20260921",
        CFG,
        False,
        (
            (0.44151526504018335, -0.049467505757993, 0.08494483804961558),
            (452.5, 372.5),
            40.29999923706055,
            0.24500000476837158,
            0.045407513898588485,
            -1.3551225596986072e-05,
        ),
    ),
    (
        "wrist_aim_pressed_right_20260921",
        CFG,
        False,
        (
            (0.4377643823969761, -0.04638347413522933, 0.09295999717706122),
            (454.5, 373.5),
            41.79999923706055,
            0.23499999940395355,
            0.04517526708020626,
            0.0070021378813037854,
        ),
    ),
    (
        "wrist_staging_pink_20260923",
        PINK,
        False,
        (
            (0.3880211639471889, -0.06134197568110687, 0.09396287421478833),
            (238.5, 389.5),
            43.79999923706055,
            0.2199999988079071,
            0.044315267504123436,
            1.0279345311059451e-07,
        ),
    ),
    (
        "wrist_staging_pink_20260923",
        PINK,
        True,
        (
            (0.3871300310789447, -0.06022768154514234, 0.09396287421478833),
            (238.5, 389.5),
            43.79999923706055,
            0.2199999988079071,
            0.044315267504123436,
            1.0279345311059451e-07,
        ),
    ),
]


@pytest.mark.parametrize("name,cfg,distorted,before", TOP_DOWN_BEFORE)
def test_a_top_down_frame_takes_the_old_path_unchanged(name, cfg, distorted, before):
    """Within OBLIQUE_MIN_DEG of straight down the aim is the staging aim it
    was: the same sighting to the last digit, and a prior or a lid height
    changes nothing."""
    from dataclasses import astuple

    m = ContainerModel.load(cfg)
    z = np.load(Path(__file__).parent / "data" / ("%s.npz" % name))
    d = D405_DIST if distorted else None
    s, why = find_button(z["color"], z["depth"], z["k"], z["rot_cam"], z["trans_cam"], TABLE_Z, m, dist=d)
    assert why is None and not s.oblique
    xyz, uv, r_px, range_m, diameter_m, above = before
    assert s.xyz == pytest.approx(xyz, abs=1e-9)
    assert s.uv == uv
    got = (s.r_px, s.range_m, s.diameter_m, s.above_lid_m)
    assert got == pytest.approx((r_px, range_m, diameter_m, above), abs=1e-9)
    assert np.isnan(s.lid_z_m) and s.incidence_deg == 0.0
    prior = (xyz[0] + 0.05, xyz[1])
    s2, _ = find_button(
        z["color"], z["depth"], z["k"], z["rot_cam"], z["trans_cam"], TABLE_Z, m, dist=d, prior_xy=prior
    )
    np.testing.assert_equal(astuple(s2), astuple(s))
    # ... until it says the box is elsewhere (DOWN_PRIOR_MAX_M)
    s4, why4 = find_button(
        z["color"],
        z["depth"],
        z["k"],
        z["rot_cam"],
        z["trans_cam"],
        TABLE_Z,
        m,
        dist=d,
        prior_xy=(xyz[0] + 0.1, xyz[1]),
    )
    assert s4 is None and "from where the box is" in why4
    # nor does a lid height: the staging aim's lid is table_z + dims.z, as it was
    for lid_z in (xyz[2] - 0.02, xyz[2] + 0.03):
        s3, _ = find_button(
            z["color"], z["depth"], z["k"], z["rot_cam"], z["trans_cam"], TABLE_Z, m, dist=d, lid_z=lid_z
        )
        np.testing.assert_equal(astuple(s3), astuple(s))


def test_a_synthetic_top_down_frame_takes_the_old_path(pink):
    """The synthetic box seen straight down from 13 cm, and from 8 deg off
    (inside OBLIQUE_MIN_DEG): the staging aim, not the oblique one."""
    from dataclasses import astuple

    h0 = _h0(pink)
    b = np.r_[BUTTON_XY, h0]
    cam = b + np.r_[0.01, -0.008, 0.13]
    color, depth = render(cam, ROT_DOWN, [(BUTTON_XY, h0, 0.0)], h0 - BOX_H)
    s, why = find_button(color, depth, OAK_K, ROT_DOWN, cam, FLAT_TABLE_Z, pink)
    assert why is None, why
    assert not s.oblique and np.isnan(s.lid_z_m)
    assert np.hypot(s.xyz[0] - b[0], s.xyz[1] - b[1]) < 0.002
    # where the box is expected does not move the sighting; a circle too far
    # from it is not the button (a cabinet's circle 28 cm off, 2026-10-23)
    s2, _ = find_button(color, depth, OAK_K, ROT_DOWN, cam, FLAT_TABLE_Z, pink, prior_xy=(b[0] + 0.05, b[1]))
    np.testing.assert_equal(astuple(s2), astuple(s))
    s3, why3 = find_button(color, depth, OAK_K, ROT_DOWN, cam, FLAT_TABLE_Z, pink, prior_xy=(b[0] + 0.1, b[1]))
    assert s3 is None and "100 mm from where the box is" in why3

    cam8, rot8 = reach_out_pose(b, 8, 0.13, off_deg=0.0)
    color, depth = render(cam8, rot8, [(BUTTON_XY, h0, 0.0)], h0 - BOX_H)
    s, why = find_button(color, depth, OAK_K, rot8, cam8, FLAT_TABLE_Z, pink)
    assert why is None, why
    assert not s.oblique


def test_a_tilt_past_the_threshold_takes_the_oblique_path(pink):
    from box_opening.perception.button_aim import OBLIQUE_MIN_DEG, view_tilt_deg

    h0 = _h0(pink)
    b = np.r_[BUTTON_XY, h0]
    cam, rot = reach_out_pose(b, OBLIQUE_MIN_DEG + 3.0, 0.14, off_deg=0.0)
    assert view_tilt_deg(rot) == pytest.approx(OBLIQUE_MIN_DEG + 3.0, abs=0.1)
    color, depth = render(cam, rot, [(BUTTON_XY, h0, 0.0)], h0 - BOX_H)
    s, why = find_button(color, depth, OAK_K, rot, cam, FLAT_TABLE_Z, pink)
    assert why is None, why
    assert s.oblique
    assert np.hypot(s.xyz[0] - b[0], s.xyz[1] - b[1]) < 0.002
    assert s.xyz[2] == pytest.approx(h0, abs=0.003)


def test_the_aimer_hands_its_priors_to_every_aim(pink, monkeypatch):
    """ButtonAimer passes lid_z and prior_xy to find_button, and an oblique
    frame's hits make a fix like any other."""
    from types import SimpleNamespace

    from box_opening.perception import button_aim, depth_source

    color, depth, cam, rot, b = reach_out_frame(pink, 39, lid_drop=0.021)
    monkeypatch.setattr(depth_source, "camera_pose_at", lambda g, offset_s=None, stamp=None: (rot, cam))
    seen = []
    real = button_aim.find_button

    def spy(*a, **kw):
        seen.append((kw.get("lid_z"), kw.get("prior_xy")))
        return real(*a, **kw)

    monkeypatch.setattr(button_aim, "find_button", spy)
    g = SimpleNamespace(color=color, depth=depth, k=OAK_K, dist=None, color_stamp=None)
    prior = (b[0] + 0.06, b[1] - 0.05)
    # the table this box stands on: the fix is then the one measured (a
    # table off it raises the fix: test_the_aimer_raises_its_fix_...)
    aimer = button_aim.ButtonAimer(g, pink, b[2] - BOX_H, lid_z=b[2] + 0.01, prior_xy=prior)
    fixed = False
    for i in range(1, button_aim.MIN_HITS + 1):  # parked: the first frame votes too (FIRST_LOOKBACK_S)
        g.color_stamp = SimpleNamespace(sec=i, nanosec=0)
        fixed = aimer.tick()
    assert fixed, aimer.status()
    assert len(seen) == button_aim.MIN_HITS
    assert all(lz == pytest.approx(b[2] + 0.01) and p == prior for lz, p in seen)
    fx = aimer.fix()
    assert np.hypot(fx[0] - b[0], fx[1] - b[1]) < 0.002 and fx[2] == pytest.approx(b[2], abs=0.003)
    assert aimer.last_sighting.oblique


def test_a_tilted_aim_on_two_hits_fixes_on_its_second_frame(pink, monkeypatch):
    """The reach-out's aimer (min_hits=2, tasks/reach_press) on the 39 deg
    view: no fix after the first still frame, a fix after the second, on
    those two hits — where the last-three rule takes a third."""
    from types import SimpleNamespace

    from box_opening.perception import button_aim, depth_source

    color, depth, cam, rot, b = reach_out_frame(pink, 39, lid_drop=0.021)
    monkeypatch.setattr(depth_source, "camera_pose_at", lambda g, offset_s=None, stamp=None: (rot, cam))
    g = SimpleNamespace(color=color, depth=depth, k=OAK_K, dist=None, color_stamp=None)
    aimer = button_aim.ButtonAimer(
        g, pink, b[2] - BOX_H, lid_z=b[2] + 0.01, prior_xy=(b[0] + 0.06, b[1] - 0.05), min_hits=2
    )
    got = []
    for i in range(1, 3):
        g.color_stamp = SimpleNamespace(sec=i, nanosec=0)
        got.append(aimer.tick())
    assert got == [False, True], aimer.status()
    assert aimer.last_sighting.oblique and aimer.fix_hits() == [0, 1]
    fx = aimer.fix()
    assert np.hypot(fx[0] - b[0], fx[1] - b[1]) < 0.002 and fx[2] == pytest.approx(b[2], abs=0.003)
    assert abs(aimer.button_above_lid_m()) < 0.003
    assert aimer.status().startswith("2 of 2 hits agree in 2 frames")


def _full_frame_d2(h, w, u, v, r_max):
    """The aim's masks as they were before _window_d2: the whole frame."""
    yy, xx = np.mgrid[0:h, 0:w]
    return 0, 0, (xx - u) ** 2 + (yy - v) ** 2


def test_the_window_holds_every_pixel_a_mask_can_pass():
    """_window_d2's window holds every pixel nearer than r_max — centres
    off the frame, on its border and between pixels — and never leaves the
    frame; its d2 is the full frame's over that window."""
    from box_opening.perception.button_aim import _window_d2

    h, w = OAK_HW
    rng = np.random.default_rng(7)
    cases = [
        (0.0, 0.0, 30.0),
        (w - 1.0, h - 1.0, 30.0),
        (319.5, 199.5, 80.0),
        (-20.0, 50.0, 25.0),
        (w + 10.5, h + 3.0, 40.0),
    ]
    cases += [(rng.uniform(-80, w + 80), rng.uniform(-80, h + 80), rng.uniform(1, 80)) for _ in range(300)]
    for u, v, r_max in cases:
        v0, u0, d2 = _window_d2(h, w, u, v, r_max)
        _, _, full = _full_frame_d2(h, w, u, v, r_max)
        assert 0 <= v0 and 0 <= u0 and v0 + d2.shape[0] <= h and u0 + d2.shape[1] <= w
        inside = np.zeros((h, w), bool)
        inside[v0 : v0 + d2.shape[0], u0 : u0 + d2.shape[1]] = True
        assert not (~inside & (full < r_max**2)).any(), (u, v, r_max)
        np.testing.assert_array_equal(d2, full[inside].reshape(d2.shape))


def test_the_aim_windows_change_no_result(frame, model, pink, monkeypatch):
    """circle_gates, disc_height and ring_height read only the window their
    disc or annulus can reach, not the whole frame (5-9 ms a call on the
    Jetson): every gate verdict and height is the full frame's, bit for bit
    — at the centre, clipped by each border, off the frame, between pixels."""
    from box_opening.perception import button_aim

    h0 = _h0(pink)
    cam = np.r_[BUTTON_XY, h0] + np.r_[0.01, -0.008, 0.13]
    color, depth = render(cam, ROT_DOWN, [(BUTTON_XY, h0, 0.0)], h0 - BOX_H)
    s, why = find_button(color, depth, OAK_K, ROT_DOWN, cam, FLAT_TABLE_Z, pink)
    assert why is None, why
    frames = [
        (
            frame["depth"],
            frame["k"],
            frame["rot_cam"],
            frame["trans_cam"],
            TABLE_Z + model.dims[2],
            (461.5, 274.5, 41.0),
        ),
        (depth, OAK_K, ROT_DOWN, cam, h0, (s.uv[0], s.uv[1], s.r_px)),
    ]
    reached = 0
    for d, k, rot, trans, lid_z, (bu, bv, br) in frames:
        hh, ww = d.shape
        circles = [
            (bu, bv, br),
            (bu + 0.3, bv - 0.7, br * 0.9),
            (508.0, 200.0, 33.5),
            (319.5, 199.5, 80.0),
            (3.0, bv, 40.0),
            (4.5, bv, br),
            (ww - 4.0, bv, 40.0),
            (ww - 5.5, bv, br),
            (bu, 3.0, 40.0),
            (bu, 4.5, br),
            (bu, hh - 4.0, 40.0),
            (bu, hh - 5.5, br),
            (3.0, 3.0, 50.0),
            (ww - 4.0, hh - 4.0, 50.0),
            (ww - 1.0, hh - 1.0, 30.0),
            (-60.0, bv, 30.0),
            (ww + 60.0, bv, 30.0),
            (bu, -60.0, 30.0),
            (bu, hh + 60.0, 30.0),
        ]
        for u, v, r in circles:
            got = (
                button_aim.circle_gates(d, k, rot, trans, u, v, r, lid_z),
                button_aim.disc_height(d, k, rot, trans, u, v, r),
                button_aim.ring_height(d, k, rot, trans, u, v, r),
            )
            with monkeypatch.context() as mp:
                mp.setattr(button_aim, "_window_d2", _full_frame_d2)
                want = (
                    button_aim.circle_gates(d, k, rot, trans, u, v, r, lid_z),
                    button_aim.disc_height(d, k, rot, trans, u, v, r),
                    button_aim.ring_height(d, k, rot, trans, u, v, r),
                )
            assert got[0] == want[0], (u, v, r)
            np.testing.assert_equal(got[1:], want[1:])
            if got[0] is None or ("border" not in got[0] and "centre" not in got[0]):
                reached += 1  # the gates got as far as the masks
            if u < -r or v < -r or u > ww + r or v > hh + r:
                assert np.isnan(got[1]) and np.isnan(got[2])
    assert reached >= 6


# --- one frame per tick: colour, depth and pose of the same moment ----------


def _st(t):
    """A header stamp at t seconds."""
    from types import SimpleNamespace

    sec = int(t)
    return SimpleNamespace(sec=sec, nanosec=int(round((t - sec) * 1e9)))


@pytest.fixture
def aimer_env(monkeypatch):
    """A ButtonAimer over a fake grabber, find_button and camera_pose_at
    scripted: poses["now"] at the frame's stamp, poses["past"] at any offset
    (the first frame's lookback); `calls` records (offset_s, stamp)."""
    from types import SimpleNamespace

    from box_opening.perception import button_aim, depth_source

    calls, found, poses = [], [], {}

    def pose_at(g, offset_s=None, stamp=None):
        calls.append((offset_s, stamp))
        return poses.get("past" if offset_s else "now")

    monkeypatch.setattr(depth_source, "camera_pose_at", pose_at)

    def fake_find(*a, **kw):
        found.append(1)
        return SimpleNamespace(
            xyz=(0.5, -0.16, 0.40), above_lid_m=0.0, table_offset_m=float("nan"), table_offset_raw_m=float("nan")
        ), None

    monkeypatch.setattr(button_aim, "find_button", fake_find)
    g = SimpleNamespace(
        color=np.zeros((4, 4, 3), np.uint8),
        depth=np.ones((4, 4), np.float32),
        k=np.eye(3),
        dist=None,
        color_stamp=None,
        depth_stamp=None,
        stamp_offset_s=0.0,
    )
    model = SimpleNamespace(dims=(0.105, 0.105, 0.121), button_diameter_m=0.044)
    rot, cam = np.diag([1.0, -1.0, -1.0]), np.array([0.5, -0.16, 0.7])
    poses["now"] = poses["past"] = (rot, cam)  # parked unless a test says otherwise
    return g, model, poses, calls, found


def _paired_frames(aimer, g, n, dt=0.08):
    """n frames dt apart, each depth stamped with its colour."""
    out = []
    for i in range(1, n + 1):
        g.color_stamp = g.depth_stamp = _st(i * dt)
        out.append(aimer.tick())
    return out


def test_a_frozen_depth_never_votes_again_while_colour_keeps_coming(aimer_env):
    """A depth stream that stalls while colour keeps arriving (the reach-out's
    after-press look: the knob popped in colour, the depth from before the
    press) must not let every new colour frame vote the same old depth."""
    from box_opening.perception import button_aim

    g, model, poses, calls, found = aimer_env
    aimer = button_aim.ButtonAimer(g, model, FLAT_TABLE_Z)
    assert not any(_paired_frames(aimer, g, button_aim.MIN_HITS - 1))  # one short of a fix
    hits, frames, frozen = aimer.hits, aimer.frames, g.depth_stamp
    for i in range(button_aim.MIN_HITS, button_aim.MIN_HITS + 10):  # colour advances, depth does not
        g.color_stamp = _st(i * 0.08)
        assert not aimer.tick()
    assert g.depth_stamp is frozen and (aimer.hits, aimer.frames) == (hits, frames)
    assert aimer.last_why == "depth not new"


def test_a_depth_that_lands_after_its_colour_is_not_lost(aimer_env):
    """Colour and depth are separate subscriptions: either may land first. A
    colour frame refused for want of its depth is tried again, not consumed,
    so a parked aim still fixes on MIN_HITS frames."""
    from box_opening.perception import button_aim

    g, model, poses, calls, found = aimer_env
    aimer = button_aim.ButtonAimer(g, model, FLAT_TABLE_Z)
    g.depth_stamp = _st(0.0)
    got = []
    for i in range(1, button_aim.MIN_HITS + 1):
        g.color_stamp = _st(i * 0.08)  # colour N, depth still N-1
        assert not aimer.tick()
        g.depth_stamp = _st(i * 0.08)  # ... then depth N lands
        got.append(aimer.tick())
    assert got[-1] and not any(got[:-1])
    assert aimer.frames == aimer.hits == button_aim.MIN_HITS


def test_a_depth_from_another_moment_than_its_colour_is_refused(aimer_env):
    from box_opening.perception import button_aim
    from box_opening.perception.depth_source import PAIR_TOL_S

    g, model, poses, calls, found = aimer_env
    aimer = button_aim.ButtonAimer(g, model, FLAT_TABLE_Z)
    for i in range(1, 6):
        g.color_stamp = _st(1.0 + i * 0.08)
        g.depth_stamp = _st(1.0 + i * 0.08 - PAIR_TOL_S - 0.01)
        assert not aimer.tick()
    assert aimer.frames == aimer.hits == 0 and not found
    assert aimer.last_why.startswith("depth not paired with colour")


def test_the_pose_is_looked_up_at_the_stamp_the_frame_was_deduplicated_by(monkeypatch):
    """The grabber's callbacks run on the perception thread: a frame landing
    between the aim's stamp read and the pose lookup used to place frame N
    with frame N+1's pose — and the next tick then compared N+1's pose with
    itself and called a camera moving 20 mm a frame still."""
    from types import SimpleNamespace

    from box_opening.perception import button_aim, depth_source

    rot = np.diag([1.0, -1.0, -1.0])

    class Grab:
        def __init__(self):
            self.color, self.depth = np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4), np.float32)
            self.k, self.dist, self.stamp_offset_s = np.eye(3), None, 0.0
            self.n, self.race = 0, False
            self.arrive()

        def arrive(self):  # both callbacks: frame n, 0.1 s apart
            self.n += 1
            self.color_stamp = self.depth_stamp = _st(0.1 * self.n)

    asked = []

    def pose_at(g, offset_s=None, stamp=None):
        if g.race:  # the callback thread wins the GIL inside the window
            g.arrive()
            g.race = False
        when = stamp if stamp is not None else g.color_stamp
        asked.append(when)
        t = when.sec + when.nanosec * 1e-9 + (offset_s or 0.0)
        return rot, np.array([0.5 + 0.2 * t, 0.0, 0.7])  # 20 mm a frame

    still = []
    monkeypatch.setattr(depth_source, "camera_pose_at", pose_at)
    monkeypatch.setattr(button_aim, "find_button", lambda *a, **kw: (still.append(a[4][0]), (None, "probe"))[1])
    g = Grab()
    aimer = button_aim.ButtonAimer(g, SimpleNamespace(dims=(0.1, 0.1, 0.1), button_diameter_m=0.044), 0.3)
    for i in range(12):
        g.arrive()
        if i in (3, 8):
            g.race = True
        before = len(asked)
        aimer.tick()
        assert all((s.sec, s.nanosec) == aimer._last_stamp for s in asked[before:])
        aimer.tick()  # the aim polls faster than frames land
    assert aimer.frames >= 12 and still == []


def test_a_parked_cameras_first_frame_votes(aimer_env):
    """The first frame's previous pose is the camera's own TF pose
    FIRST_LOOKBACK_S before its stamp: a parked arm fixes on MIN_HITS frames
    (the reach-out aims after a 0.46 s at-rest read, and lost a frame each)."""
    from box_opening.perception import button_aim

    g, model, poses, calls, found = aimer_env
    aimer = button_aim.ButtonAimer(g, model, FLAT_TABLE_Z)
    got = _paired_frames(aimer, g, button_aim.MIN_HITS)
    assert got[-1] and not any(got[:-1])
    assert aimer.frames == button_aim.MIN_HITS == aimer.hits
    assert calls[1][0] == pytest.approx(-button_aim.FIRST_LOOKBACK_S)
    assert calls[1][1] is calls[0][1]  # the lookback is from the same frame's stamp


def test_a_camera_that_moved_before_the_first_frame_is_refused(aimer_env):
    from box_opening.perception import button_aim

    g, model, poses, calls, found = aimer_env
    rot, cam = poses["now"]
    poses["past"] = (rot, cam + [0.0, 0.006, 0.0])  # 6 mm in the lookback
    aimer = button_aim.ButtonAimer(g, model, FLAT_TABLE_Z)
    got = _paired_frames(aimer, g, button_aim.MIN_HITS + 1)
    assert not got[0] and aimer.hits == button_aim.MIN_HITS and got[-1]


def test_without_tf_history_the_first_frame_still_never_votes(aimer_env):
    from box_opening.perception import button_aim

    g, model, poses, calls, found = aimer_env
    del poses["past"]  # the lookback finds nothing
    aimer = button_aim.ButtonAimer(g, model, FLAT_TABLE_Z)
    got = _paired_frames(aimer, g, button_aim.MIN_HITS + 1)
    assert not got[0] and got[-1] and len(found) == button_aim.MIN_HITS


def test_the_first_frames_lookback_honours_the_stamp_offset(aimer_env):
    from box_opening.perception import button_aim

    g, model, poses, calls, found = aimer_env
    g.stamp_offset_s = 0.03
    aimer = button_aim.ButtonAimer(g, model, FLAT_TABLE_Z)
    _paired_frames(aimer, g, 1)
    assert [c[0] for c in calls] == [None, pytest.approx(0.03 - button_aim.FIRST_LOOKBACK_S)]


NAN = float("nan")


@pytest.mark.parametrize(
    "offsets,raised,agreed,raw",
    [
        ([(-0.020, -0.020), (-0.021, -0.021), (-0.019, -0.019)], 0.020, -0.020, -0.020),  # all three agree
        ([(-0.020, -0.020), (NAN, -0.045), (-0.022, -0.022)], 0.021, -0.021, -0.022),  # two; one not taken out
        ([(-0.020, -0.020), (NAN, -0.045), (NAN, -0.046)], 0.0, NAN, -0.045),  # one taken out is not two
        ([(-0.010, -0.010), (-0.020, -0.020), (-0.030, -0.030)], 0.0, NAN, -0.020),  # none within 4 mm of another
        ([(0.015, 0.015), (0.016, 0.016), (0.015, 0.015)], 0.0, 0.015, 0.015),  # read high: never lowered
        ([(NAN, NAN)] * 3, 0.0, NAN, NAN),  # no table in view (and every straight-down sighting)
    ],
)
def test_the_aimer_raises_its_fix_by_an_agreed_table_offset_and_never_lowers_it(
    aimer_env, monkeypatch, offsets, raised, agreed, raw
):
    """The fix's z goes up by the wrist's table offset when two of its hits
    agree on one within TABLE_AGREE_M, and never down: the compliant press
    goes by the higher of it and the scene camera's lid, and a lowered aim
    would no longer bound the press from below. A hit whose offset was not
    taken out counts for nothing; its raw offset is still reported."""
    from box_opening.perception import button_aim
    from box_opening.perception.button_aim import ButtonSighting

    g, model, poses, calls, found = aimer_env
    script = list(offsets)

    def scripted(*a, **kw):
        off, off_raw = script.pop(0)
        return ButtonSighting(
            xyz=(0.5, -0.16, 0.290),
            uv=(320.0, 200.0),
            r_px=30.0,
            range_m=0.30,
            diameter_m=0.044,
            oblique=True,
            table_offset_m=off,
            table_offset_raw_m=off_raw,
        ), None

    monkeypatch.setattr(button_aim, "find_button", scripted)
    aimer = button_aim.ButtonAimer(g, model, FLAT_TABLE_Z)
    assert _paired_frames(aimer, g, button_aim.MIN_HITS)[-1] and not script
    assert aimer.fix() == pytest.approx((0.5, -0.16, 0.290 + raised), abs=1e-9)
    assert aimer.fix(measured=True) == pytest.approx((0.5, -0.16, 0.290), abs=1e-9)
    assert aimer.table_offset_m() == pytest.approx(agreed, abs=1e-9, nan_ok=True)
    assert aimer.table_offset_m(raw=True) == pytest.approx(raw, abs=1e-9, nan_ok=True)


def _scripted_hits(monkeypatch, script):
    """find_button returning the oblique sightings `script` lists, in turn:
    (xyz, above_lid_m, (accepted, raw) table offset)."""
    from box_opening.perception import button_aim
    from box_opening.perception.button_aim import ButtonSighting

    def scripted(*a, **kw):
        xyz, up, (off, raw) = script.pop(0)
        return ButtonSighting(
            xyz=xyz,
            uv=(320.0, 200.0),
            r_px=30.0,
            range_m=0.30,
            diameter_m=0.044,
            above_lid_m=up,
            oblique=True,
            table_offset_m=off,
            table_offset_raw_m=raw,
        ), None

    monkeypatch.setattr(button_aim, "find_button", scripted)


def test_a_tilted_aim_fixes_on_two_agreeing_hits_of_its_last_four(aimer_env, monkeypatch):
    """Flatbot 2026-10-27 15:19: the reach-out aim's first hit did not agree,
    and the last-three rule waited for three more. On two hits (min_hits=2)
    a tilted view's hits [A, A + 10 mm, A + 1 mm] fix on the first and the
    third, at their mean, and the knob, the table offset and the status are
    those two hits' — not the last two's or the last three's."""
    from box_opening.perception import button_aim

    g, model, poses, calls, found = aimer_env
    script = [
        ((0.500, -0.160, 0.290), 0.0010, (-0.020, -0.020)),
        ((0.500, -0.150, 0.290), 0.0090, (-0.030, -0.030)),  # 10 mm off
        ((0.501, -0.160, 0.292), 0.0020, (-0.022, -0.022)),
    ]
    _scripted_hits(monkeypatch, script)
    aimer = button_aim.ButtonAimer(g, model, FLAT_TABLE_Z, min_hits=2)
    assert _paired_frames(aimer, g, 3) == [False, False, True] and not script
    assert aimer.fix_hits() == [0, 2]
    assert aimer.fix(measured=True) == pytest.approx((0.5005, -0.160, 0.291), abs=1e-9)
    # the offset hits 1 and 3 agree on raises it; the last two (-30, -22 mm) agree on none
    assert aimer.table_offset_m() == pytest.approx(-0.021, abs=1e-9)
    assert aimer.fix() == pytest.approx((0.5005, -0.160, 0.312), abs=1e-9)
    assert aimer.table_offset_m(raw=True) == pytest.approx(-0.021, abs=1e-9)  # the last three's: -22 mm
    # the knob: 1.5 mm, where the last two read 5.5 mm and the last three 2.0
    assert aimer.button_above_lid_m() == pytest.approx(0.0015, abs=1e-9)
    assert aimer.status().startswith("2 of 3 hits agree in 3 frames")
    assert len(aimer.hit_s) == 3 and aimer.hit_s == sorted(aimer.hit_s) and aimer.hit_s[0] >= 0.0


def test_two_hits_that_read_the_knob_apart_wait_for_a_third(aimer_env, monkeypatch):
    """Whether the box is open (press_demo.refuse_open_box at 6 mm) must not
    rest on one outlying knob reading: two hits whose knobs read more than
    KNOB_NOISE_M apart are no fix, and a third that pairs with neither makes
    one of all three — their median, the knob's too."""
    from box_opening.perception import button_aim

    g, model, poses, calls, found = aimer_env
    nan2 = (float("nan"), float("nan"))
    script = [
        ((0.500, -0.160, 0.290), 0.0005, nan2),
        ((0.501, -0.160, 0.291), 0.0070, nan2),  # 6.5 mm apart: "open" on its own
        ((0.500, -0.159, 0.292), 0.0035, nan2),  # 3.0 and 3.5 mm from those: a pair with neither
    ]
    _scripted_hits(monkeypatch, script)
    aimer = button_aim.ButtonAimer(g, model, FLAT_TABLE_Z, min_hits=2)
    assert _paired_frames(aimer, g, 2) == [False, False]
    assert aimer.fix() is None and aimer.status().startswith("0 of 2 hits agree in 2 frames")
    g.color_stamp = g.depth_stamp = _st(3 * 0.08)
    assert aimer.tick() and not script
    assert aimer.fix_hits() == [0, 1, 2]
    assert aimer.fix() == pytest.approx((0.500, -0.160, 0.291), abs=1e-9)
    assert aimer.button_above_lid_m() == pytest.approx(0.0035, abs=1e-9)


def test_a_straight_down_aim_still_needs_three_hits_in_a_row(aimer_env, monkeypatch):
    """min_hits=2 is a tilted view's: straight-down sightings (no oblique
    flag) still fix on the last three agreeing, as every staging aim has."""
    from types import SimpleNamespace

    from box_opening.perception import button_aim

    g, model, poses, calls, found = aimer_env
    a, b = (0.5, -0.16, 0.40), (0.5, -0.15, 0.40)
    script = [a, a, b, a, a, a]

    def scripted(*args, **kw):
        return SimpleNamespace(
            xyz=script.pop(0), above_lid_m=0.0, table_offset_m=float("nan"), table_offset_raw_m=float("nan")
        ), None

    monkeypatch.setattr(button_aim, "find_button", scripted)
    aimer = button_aim.ButtonAimer(g, model, FLAT_TABLE_Z, min_hits=2)
    assert _paired_frames(aimer, g, 6) == [False] * 5 + [True] and not script
    assert aimer.fix_hits() == [3, 4, 5] and aimer.fix() == pytest.approx(a)


# lid_in_view: a camera straight down (its x along base +y, y along +x, as
# flatbot's wrist OAK at staging), 0.30 m above the lid
DOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
LID_K = np.array([[398.0, 0.0, 320.0], [0.0, 398.0, 200.0], [0.0, 0.0, 1.0]])


def _table_and_lid(model, lid_xy, cam=(0.45, -0.25, 0.71), table_z=0.27):
    """The depth a straight-down camera at `cam` reads: the table, and a
    round lid model.dims[2] above it at `lid_xy`."""
    vv, uu = np.mgrid[0:400, 0:640]
    lid_z = table_z + model.dims[2]
    d_lid = cam[2] - lid_z
    ray = np.stack([(uu - 320.0) / 398.0, (vv - 200.0) / 398.0, np.ones_like(uu, float)], -1)
    at_lid = (ray * d_lid) @ DOWN.T + np.array(cam)
    on = np.hypot(at_lid[..., 0] - lid_xy[0], at_lid[..., 1] - lid_xy[1]) < model.lid_dims[0] / 2
    return np.where(on, d_lid, cam[2] - table_z), np.array(cam)


def test_the_lid_in_view_is_the_patch_a_box_height_above_the_table():
    from box_opening.perception.button_aim import lid_in_view

    m = ContainerModel.load(PINK)
    depth, cam = _table_and_lid(m, (0.47, -0.22))
    lid, info = lid_in_view(depth, LID_K, DOWN, cam, m)
    assert lid[0] == pytest.approx(0.47, abs=0.003) and lid[1] == pytest.approx(-0.22, abs=0.003)
    assert not info["cut"] and info["frac"] == pytest.approx(1.0, abs=0.1)
    # judged against the table in view: a camera TF puts 5 cm too high finds the same lid
    lid2, _info = lid_in_view(depth, LID_K, DOWN, cam + [0.0, 0.0, 0.05], m)
    assert lid2[:2] == pytest.approx(lid[:2], abs=1e-6)
    bare, why = lid_in_view(np.full((400, 640), 0.44), LID_K, DOWN, cam, m)
    assert bare is None and "nothing stands" in why


def test_the_cut_lid_at_staging_is_found_where_the_box_was():
    """Flatbot 2026-10-23 13:33: staging over the scene camera's fix
    [0.475, -0.276]; the wrist frame shows half the lid at its edge. Its lid
    is found 7 cm from that fix, cut by the frame, where the wrist camera's
    own ring said the box was."""
    from box_opening.perception.button_aim import lid_in_view

    z = np.load(Path(__file__).parent / "data" / "wrist_staging_cut_20261023.npz")
    lid, info = lid_in_view(
        z["depth_mm"].astype(float) / 1000.0, z["k"], z["rot_cam"], z["trans_cam"], ContainerModel.load(PINK)
    )
    assert lid is not None, info
    assert info["cut"] and 0.15 < info["frac"] < 0.6
    assert lid[0] == pytest.approx(0.460, abs=0.01) and lid[1] == pytest.approx(-0.204, abs=0.01)
    assert 0.06 < np.hypot(lid[0] - 0.475, lid[1] + 0.276) < 0.10  # a re-staging, not a search


def test_the_button_is_judged_against_the_lid_this_frame_measures(pink):
    """Flatbot 2026-10-23: the scene camera's table swung 0.180-0.200 m with
    the box still, and against table + dims.z alone the real button read
    "12 mm from lid height" and was refused at 16:30. The lid ring around
    the circle, when within LID_PRIOR_TOL_M of that, is the lid it is
    judged against; farther, the prior stands and refuses."""
    from box_opening.perception.button_aim import LID_PRIOR_TOL_M

    h0 = _h0(pink)
    b = np.r_[BUTTON_XY, h0]
    cam = b + np.r_[0.01, -0.008, 0.13]
    color, depth = render(cam, ROT_DOWN, [(BUTTON_XY, h0, 0.0)], h0 - BOX_H)
    s, why = find_button(color, depth, OAK_K, ROT_DOWN, cam, FLAT_TABLE_Z, pink)
    assert why is None, why
    high, why_h = find_button(
        color, depth, OAK_K, ROT_DOWN, cam, FLAT_TABLE_Z + 0.015, pink
    )  # the table read 15 mm high
    assert why_h is None and high.xyz == pytest.approx(s.xyz)
    off, why_o = find_button(color, depth, OAK_K, ROT_DOWN, cam, FLAT_TABLE_Z + LID_PRIOR_TOL_M + 0.015, pink)
    assert off is None, why_o  # a lid that far from the box's is no lid of this box


def test_a_button_with_no_depth_at_its_very_centre_is_read_off_its_disc(pink):
    """16:34: the flat disc seen straight down read patchily, its centre's
    7x7 window empty in every frame: the disc's own median ranges it."""
    h0 = _h0(pink)
    b = np.r_[BUTTON_XY, h0]
    cam = b + np.r_[0.01, -0.008, 0.13]
    color, depth = render(cam, ROT_DOWN, [(BUTTON_XY, h0, 0.0)], h0 - BOX_H)
    s, why = find_button(color, depth, OAK_K, ROT_DOWN, cam, FLAT_TABLE_Z, pink)
    assert why is None, why
    u, v = (int(round(c)) for c in s.uv)
    holed = depth.copy()
    holed[v - 6 : v + 7, u - 6 : u + 7] = 0.0
    s2, why2 = find_button(color, holed, OAK_K, ROT_DOWN, cam, FLAT_TABLE_Z, pink)
    assert why2 is None, why2
    assert np.hypot(s2.xyz[0] - s.xyz[0], s2.xyz[1] - s.xyz[1]) < 0.002
