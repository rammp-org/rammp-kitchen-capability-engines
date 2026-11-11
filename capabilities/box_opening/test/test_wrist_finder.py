"""Finding the box from the wrist camera (perception/wrist_finder), on frames ray-cast here:
a counter, the canister standing on it (holder flange, canister, lid, button), and what else
may stand there — an open cup, a bottle, a second canister."""

import math
from dataclasses import replace

import numpy as np
import pytest

from box_opening.models.container import ContainerModel
from box_opening.perception import button_aim as ba
from box_opening.perception import clear_table as ct
from box_opening.perception import wrist_finder as wf

PINK = "capabilities/box_opening/config/containers/ankou_pink.yaml"
COUNTER_Z = 0.239  # the 0.76 m counter in arm_mount (chair study)
LID_H = 0.099  # the canister's top above its counter as the cameras measure it (not dims.z's 121)
W, H, F = 640, 400, 398.0  # the wrist OAK's colour stream

COLOURS = {  # BGR
    "counter": (200, 205, 210),
    "holder": (200, 120, 40),
    "canister": (215, 215, 220),
    "lid": (190, 175, 245),
    "button": (120, 60, 200),
    "cup": (240, 240, 240),
    "bottle": (60, 120, 60),
}


@pytest.fixture(scope="module")
def pink():
    return ContainerModel.load(PINK)


def canister(x, y, counter_z=COUNTER_Z, lid_h=LID_H):
    """The canister as cylinders (r, z0, z1, part): the holder's flange (152 mm, 45 mm under the
    lid top), the canister, the lid and its 44 mm button, flush (presssim's defaults)."""
    top = counter_z + lid_h
    return [
        ("cyl", x, y, 0.058, counter_z, top - 0.049, "holder"),
        ("cyl", x, y, 0.076, top - 0.049, top - 0.045, "holder"),
        ("cyl", x, y, 0.0525, top - 0.045, top - 0.012, "canister"),
        ("cyl", x, y, 0.0525, top - 0.012, top, "lid"),
        ("cyl", x, y, 0.022, top - 0.012, top + 0.0002, "button"),
    ]


def cup(x, y, r=0.045, height=0.095, wall=0.004, counter_z=COUNTER_Z):
    """An open cup: its wall, rim and inside bottom."""
    return [("cup", x, y, r, counter_z, counter_z + height, wall)]


def bottle(x, y, r=0.04, height=0.25, counter_z=COUNTER_Z):
    return [("cyl", x, y, r, counter_z, counter_z + height, "bottle")]


def _camera(cam, at):
    """The optical frame's axes (columns x right, y down, z ahead) looking from cam at `at`."""
    z = np.asarray(at, float) - np.asarray(cam, float)
    z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], 1)


def _cyl_hits(o, d, x, y, r, z0, z1, inside=False):
    """Ray depth (t, the camera's z-depth: d has z 1 in the camera) to a vertical cylinder's
    wall: the near root from outside, the far one from inside; inf where it misses."""
    ox, oy = o[0] - x, o[1] - y
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2 * (ox * d[:, 0] + oy * d[:, 1])
    c = ox * ox + oy * oy - r * r
    disc = b * b - 4 * a * c
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.sqrt(np.where(disc >= 0, disc, np.nan))
        t = (-b + s) / (2 * a) if inside else (-b - s) / (2 * a)
    z = o[2] + t * d[:, 2]
    ok = np.isfinite(t) & (t > 0) & (z >= z0) & (z <= z1)
    return np.where(ok, t, np.inf)


def _disc_hits(o, d, x, y, z, r_out, r_in=0.0):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (z - o[2]) / d[:, 2]
    px, py = o[0] + t * d[:, 0] - x, o[1] + t * d[:, 1] - y
    rr = np.hypot(px, py)
    return np.where((t > 0) & (rr <= r_out) & (rr >= r_in), t, np.inf)


def render(cam, at, things, counter=(-0.4, 1.2, -0.6, 1.2), counter_z=COUNTER_Z, seam=True):
    """A WristFrame of `things` on a counter (x0, x1, y0, y1 at counter_z), seen from cam looking
    at `at`: depth (camera z, metres; 0 where nothing is hit), BGR colour, the pose."""
    R = _camera(cam, at)
    o = np.asarray(cam, float)
    v, u = np.mgrid[0:H, 0:W]
    dc = np.stack([(u - W / 2.0) / F, (v - H / 2.0) / F, np.ones((H, W))], -1).reshape(-1, 3)
    d = dc @ R.T
    best = np.full(len(d), np.inf)
    label = np.full(len(d), "", object)

    def take(t, what):
        nearer = t < best
        best[nearer] = t[nearer]
        label[nearer] = what

    t = _disc_hits(o, d, 0.0, 0.0, counter_z, 10.0)
    with np.errstate(invalid="ignore"):
        px, py = o[0] + t * d[:, 0], o[1] + t * d[:, 1]
    on = (px >= counter[0]) & (px <= counter[1]) & (py >= counter[2]) & (py <= counter[3])
    take(np.where(on, t, np.inf), "counter")
    for kind, x, y, r, z0, z1, what in things:
        if kind == "cyl":
            take(_cyl_hits(o, d, x, y, r, z0, z1), what)
            take(_disc_hits(o, d, x, y, z1, r), what)
        else:  # an open cup: `what` is its wall's thickness
            take(_cyl_hits(o, d, x, y, r, z0, z1), "cup")
            take(_disc_hits(o, d, x, y, z1, r, r - what), "cup")
            take(_cyl_hits(o, d, x, y, r - what, z0 + what, z1, inside=True), "cup")
            take(_disc_hits(o, d, x, y, z0 + what, r - what), "cup")
    depth = np.where(np.isfinite(best), best, 0.0).reshape(H, W).astype(np.float32)
    color = np.zeros((H * W, 3), np.uint8)
    for what, bgr in COLOURS.items():
        color[label == what] = bgr
    color = color.reshape(H, W, 3)
    if seam:  # the button's seam: a dark ring where the button meets the lid
        import cv2

        for _kind, x, y, r, _z0, z1, what in things:
            if what == "button":  # its rim's image, sampled, drawn as a polygon
                ang = np.linspace(0, 2 * math.pi, 64)
                rim = np.stack([x + r * np.cos(ang), y + r * np.sin(ang), np.full_like(ang, z1)], 1)
                pc = (rim - o) @ R
                poly = np.stack([F * pc[:, 0] / pc[:, 2] + W / 2.0, F * pc[:, 1] / pc[:, 2] + H / 2.0], 1)
                cv2.polylines(color, [np.round(poly).astype(np.int32)], True, (40, 20, 60), 2)
    k = np.array([[F, 0, W / 2.0], [0, F, H / 2.0], [0, 0, 1.0]])
    return wf.WristFrame(color=color, depth=depth, k=k, rot_cam=R, trans_cam=o, dist=np.zeros(5))


def _down(x, y, above=0.30):
    """Straight down over (x, y), `above` the lid."""
    return render((x, y, COUNTER_Z + LID_H + above), (x + 1e-4, y, COUNTER_Z), canister(x, y))


def test_a_canister_seen_straight_down_is_found_where_it_stands(pink):
    found = wf.find_container([_down(0.45, 0.25)], pink)
    assert found.fix is not None, found.why
    assert found.fix.source == "wrist"
    assert math.hypot(found.fix.top_xyz[0] - 0.45, found.fix.top_xyz[1] - 0.25) < 0.004
    assert found.fix.top_xyz[2] == pytest.approx(COUNTER_Z + LID_H, abs=0.004)
    assert found.fix.table_z_scene == pytest.approx(COUNTER_Z, abs=0.003)
    assert found.fix.pose.xyz[2] == pytest.approx(COUNTER_Z, abs=0.003)  # the pose stands on the counter
    assert found.button_xyz is not None and found.confidence == pytest.approx(0.9)


def test_from_a_tilted_survey_pose_the_button_confirms_it(pink):
    """Side-on, from the arm's side of the counter (-y), the camera 0.35 m up looking down
    at the box 40 deg off straight down: the aim's oblique circle detector confirms it."""
    things = canister(0.40, 0.30)
    f = render((0.40, 0.02, COUNTER_Z + 0.36), (0.40, 0.30, COUNTER_Z + LID_H), things)
    assert ba.view_tilt_deg(f.rot_cam) > 30
    found = wf.find_container([f], pink)
    assert found.fix is not None, found.why
    assert math.hypot(found.fix.top_xyz[0] - 0.40, found.fix.top_xyz[1] - 0.30) < 0.006
    assert found.places[0].button == "seen"


def test_two_frames_of_one_box_are_one_place_and_raise_the_confidence(pink):
    things = canister(0.40, 0.30)
    a = render((0.40, 0.02, COUNTER_Z + 0.36), (0.40, 0.30, COUNTER_Z + LID_H), things)
    b = render((0.30, 0.05, COUNTER_Z + 0.40), (0.40, 0.30, COUNTER_Z + LID_H), things)
    found = wf.find_container([a, b], pink)
    assert found.fix is not None, found.why
    assert len(found.places) == 1 and found.fix.n_sightings == 2
    assert found.confidence == pytest.approx(1.0)
    assert len(found.clouds) == 2  # one per frame, each on its own counter plane


def test_an_open_cup_beside_it_is_no_container(pink):
    things = canister(0.40, 0.30) + cup(0.22, 0.30)
    f = render((0.31, 0.30, COUNTER_Z + LID_H + 0.35), (0.31 + 1e-4, 0.30, COUNTER_Z), things)
    found = wf.find_container([f], pink)
    assert found.fix is not None, found.why
    assert math.hypot(found.fix.top_xyz[0] - 0.40, found.fix.top_xyz[1] - 0.30) < 0.005
    rim = [p for p in found.patches if math.hypot(p.xy[0] - 0.22, p.xy[1] - 0.30) < 0.03]
    assert rim and all(p.why for p in rim)  # its rim is a ring, its inside lower


def test_what_stands_over_a_lid_height_top_makes_it_no_container(pink):
    """A bottle standing on the lid: the lid's ring shows around it at lid height, and the
    bottle over it is taller than any container (nor can a lid with a bottle on it come off)."""
    things = canister(0.40, 0.30) + bottle(0.40, 0.30, r=0.03, height=0.20, counter_z=COUNTER_Z + LID_H)
    f = render((0.40, 0.02, COUNTER_Z + 0.36), (0.40, 0.30, COUNTER_Z + LID_H), things)
    found = wf.find_container([f], pink)
    assert found.fix is None
    assert "taller than the container" in found.why


def test_two_canisters_are_two_places_and_no_fix(pink):
    things = canister(0.30, 0.30) + canister(0.52, 0.30)
    f = render((0.41, 0.30, COUNTER_Z + LID_H + 0.40), (0.41 + 1e-4, 0.30, COUNTER_Z), things)
    found = wf.find_container([f], pink)
    assert found.fix is None
    assert "two places pass" in found.why and len(found.places) == 2


def test_a_detector_box_on_one_of_two_breaks_the_tie(pink):
    things = canister(0.30, 0.30) + canister(0.52, 0.30)
    f = render((0.41, 0.30, COUNTER_Z + LID_H + 0.40), (0.41 + 1e-4, 0.30, COUNTER_Z), things)
    p = f.rot_cam.T @ (np.array([0.52, 0.30, COUNTER_Z + LID_H]) - f.trans_cam)
    u, v = F * p[0] / p[2] + W / 2.0, F * p[1] / p[2] + H / 2.0
    boxed = wf.WristFrame(**{**f.__dict__, "boxes": [(0.9, [u - 60, v - 60, u + 60, v + 60])]})
    found = wf.find_container([boxed], pink)
    assert found.fix is not None, found.why
    assert found.fix.top_xyz[0] == pytest.approx(0.52, abs=0.005)


def test_an_empty_counter_and_a_frame_without_depth_are_refused(pink):
    empty = render((0.40, 0.30, COUNTER_Z + 0.45), (0.40 + 1e-4, 0.30, COUNTER_Z), [])
    found = wf.find_container([empty], pink)
    assert found.fix is None and "nothing on the counter is the container" in found.why
    blind = wf.WristFrame(**{**empty.__dict__, "depth": np.zeros((H, W), np.float32)})
    found = wf.find_container([blind], pink)
    assert found.fix is None and "too little depth" in found.why


def test_a_lid_the_view_cuts_is_a_place_to_look_again_not_a_fix(pink):
    # the camera over the counter beside the box: its lid runs out of the image's top
    things = canister(0.40, 0.42)
    f = render((0.40, 0.21, COUNTER_Z + LID_H + 0.25), (0.40 + 1e-4, 0.21, COUNTER_Z), things)
    found = wf.find_container([f], pink)
    assert found.fix is None
    assert "cut off by the view" in found.why


def test_a_lid_without_its_button_where_the_view_shows_it_is_refused(pink):
    """A lid-sized solid top with no button circle (a closed tin, a jar) is no container
    where the view shows its top; with the circle detector off it is a geometry-only fix."""
    tin = [t for t in canister(0.45, 0.25) if t[-1] != "button"]
    f = render((0.45, 0.25, COUNTER_Z + LID_H + 0.30), (0.45 + 1e-4, 0.25, COUNTER_Z), tin, seam=False)
    found = wf.find_container([f], pink)
    assert found.fix is None and "no button circle" in found.why
    found = wf.find_container([f], pink, check_buttons=False)
    assert found.fix is not None and found.confidence == pytest.approx(0.6)


def test_a_top_no_view_could_check_the_button_on_is_a_place_to_look_again_not_a_fix(pink):
    """64 deg off straight down the aim cannot check the button (CHECK_MAX_INCIDENCE_DEG):
    geometry alone does not tell the canister from a closed tin of its size, so neither is a
    fix until a view nearer above shows the button; with the circle detector off it is a
    geometry-only fix."""
    far = ((0.40, -0.12, COUNTER_Z + 0.30), (0.40, 0.30, COUNTER_Z + LID_H))
    tin = [t for t in canister(0.40, 0.30) if t[-1] != "button"]
    for things, seam in ((tin, False), (canister(0.40, 0.30), True)):
        f = render(*far, things, seam=seam)
        found = wf.find_container([f], pink)
        assert found.fix is None and "look again nearer above it" in found.why, found.why
        assert len(found.places) == 1 and found.places[0].button == "unchecked"
    assert wf.find_container([f], pink, check_buttons=False).confidence == pytest.approx(0.6)
    near = render((0.40, 0.02, COUNTER_Z + 0.36), (0.40, 0.30, COUNTER_Z + LID_H), canister(0.40, 0.30))
    found = wf.find_container([f, near], pink)  # one view nearer above shows it
    assert found.fix is not None and found.confidence == pytest.approx(0.9), found.why


def test_one_frame_is_one_sighting_however_many_planes_judge_its_lid(pink):
    """A tray beside the box is a second counter-sized level plane: the one lid passes on both,
    and is still one frame's sighting (0.9), not two (1.0)."""
    things = canister(0.45, 0.25) + [("cyl", 0.45, 0.50, 0.16, COUNTER_Z, COUNTER_Z + 0.012, "holder")]
    f = render((0.45, 0.30, COUNTER_Z + LID_H + 0.35), (0.45 + 1e-4, 0.30, COUNTER_Z), things)
    found = wf.find_container([f], pink)
    assert found.fix is not None, found.why
    assert sum(not p.why for p in found.patches) == 2  # the lid, judged on the counter and on the tray
    assert found.confidence == pytest.approx(0.9)


def test_frames_whose_poses_err_apart_in_z_pool_onto_one_counter(pink):
    """Each wrist pose's z errs its own way (20-60 mm tilted): the fix's one cloud (room_map's
    single plane fit) holds every frame's counter at one height, or one frame's counter
    "stands" on the other's all over the map."""
    things = canister(0.45, 0.25) + cup(0.62, 0.25)
    a = render((0.50, 0.25, COUNTER_Z + LID_H + 0.40), (0.50 + 1e-4, 0.25, COUNTER_Z), things)
    b = render((0.42, 0.20, COUNTER_Z + LID_H + 0.35), (0.42 + 1e-4, 0.20, COUNTER_Z), things)
    b = wf.WristFrame(**{**b.__dict__, "trans_cam": b.trans_cam + np.array([0.0, 0.0, 0.03])})  # 30 mm off
    found = wf.find_container([a, b], pink)
    assert found.fix is not None, found.why
    zs = sorted(z for _pts, z in found.clouds)
    assert zs[1] - zs[0] == pytest.approx(0.03, abs=0.004)  # each frame's own cloud keeps its own counter
    tmap = ct.table_map([(found.fix.table_points, found.fix.table_z_scene)], found.fix.pose.xyz[:2])
    standing = tmap.standing()
    box = np.hypot(standing[:, 0] - 0.45, standing[:, 1] - 0.25) < 0.10
    cup_ = np.hypot(standing[:, 0] - 0.62, standing[:, 1] - 0.25) < 0.08
    assert box.any() and cup_.any() and not (~box & ~cup_).any()


def test_a_frame_without_depth_or_colour_is_no_fix(pink):
    f = _down(0.45, 0.25)
    assert "too little depth" in wf.find_container([wf.WristFrame(**{**f.__dict__, "depth": None})], pink).why
    found = wf.find_container([wf.WristFrame(**{**f.__dict__, "color": None})], pink)
    assert found.fix is None and "no colour image" in found.why


def test_the_lid_height_band_holds_the_measured_spread(pink):
    """The scene camera measured the canister's top 88-129 mm above its table; dims.z says 121."""
    for lid_h in (0.088, 0.129):
        x, y = 0.45, 0.25
        f = render((x, y, COUNTER_Z + lid_h + 0.30), (x + 1e-4, y, COUNTER_Z), canister(x, y, lid_h=lid_h))
        found = wf.find_container([f], pink)
        assert found.fix is not None, (lid_h, found.why)
        assert found.fix.top_xyz[2] - found.fix.table_z_scene == pytest.approx(lid_h, abs=0.004)


def test_its_clouds_make_the_lid_spot_map(pink):
    """What table_map (the lid's spot) is fed: the counter around the box, its own plane."""
    things = canister(0.45, 0.25) + cup(0.62, 0.25)
    f = render((0.50, 0.25, COUNTER_Z + LID_H + 0.40), (0.50 + 1e-4, 0.25, COUNTER_Z), things)
    found = wf.find_container([f], pink)
    assert found.fix is not None, found.why
    tmap = ct.table_map(list(found.clouds), found.fix.pose.xyz[:2])
    assert tmap is not None
    standing = tmap.standing()
    near_cup = standing[np.hypot(standing[:, 0] - 0.62, standing[:, 1] - 0.25) < 0.05]
    assert len(near_cup) and near_cup[:, 2].max() == pytest.approx(0.095, abs=0.01)
    # the SceneFix's own points do as well (press_demo.room_map reads them)
    assert ct.table_map([(found.fix.table_points, found.fix.table_z_scene)], found.fix.pose.xyz[:2]) is not None


def test_a_lid_whose_depth_runs_out_over_a_plain_table_is_placed_by_its_rim_in_the_colour_image():
    """Flatbot 2026-11-10 17:45, the reach-out's look at the pink canister on a white folding table: the stereo
    carries the lid's depth out over the bare table beside it, and the 105 mm lid reads a top 190 x 225 mm ("wider
    than the lid's", no fix). The colour image has its rim: one lid-sized circle on the plane at the top's height."""
    from conftest import fixture_path

    model = ContainerModel.load(PINK)
    frame = wf.frame_from_capture(str(fixture_path("wrist_reach_look_flatbot_20261110_1745.npz")))
    patches, plane, _pts = wf.frame_patches(frame, model)
    (lid,) = (p for p in patches if not p.why)
    assert lid.rim and lid.axes_m[1] > 1.25 * 0.105  # the depth's top too long; the rim placed it
    find = wf.find_container([frame], model, check_buttons=False)
    assert find.fix is not None, find.why
    # where a mask of the lid's own pink puts its middle, by the same depth and pose: [0.708, -0.347]
    assert np.hypot(find.fix.top_xyz[0] - 0.708, find.fix.top_xyz[1] + 0.347) < 0.015
    assert find.fix.top_xyz[2] - find.fix.table_z_scene == pytest.approx(0.09, abs=0.012)
    # a top that long with no circle on it stays refused: the same frame, its colour blank
    blank = replace(frame, color=np.full_like(frame.color, 128))
    assert wf.find_container([blank], model, check_buttons=False).fix is None


def test_only_a_top_too_long_is_looked_at_again_by_its_rim():
    model = ContainerModel.load(PINK)
    p = wf.Patch(
        0,
        (0.5, 0.0),
        0.34,
        0.24,
        0.10,
        (0.30, 0.34),
        0.9,
        1.0,
        0.0,
        900,
        False,
        "its top 300x340 mm, wider than the lid's 105",
    )
    assert wf.by_its_rim(None, None, p, model) is p  # three lids across its short side: a tray, not a smeared lid
    ring = replace(p, axes_m=(0.11, 0.18), why="its top a ring, 40% filled — an open cup's rim?")
    assert wf.by_its_rim(None, None, ring, model) is ring
    cut = replace(p, axes_m=(0.11, 0.18), cut=True)
    assert wf.by_its_rim(None, None, cut, model) is cut


@pytest.mark.parametrize(
    "name, button",
    [
        ("wrist_survey_flatbot_20261111_1913.npz", (0.731, -0.286)),
        ("wrist_survey_flatbot_20261111_1929.npz", (0.622, -0.202)),
    ],
)
def test_from_the_turned_wrist_the_lid_is_where_its_depth_tops_middle_is(name, button):
    """Flatbot 2026-11-11, the wrist turned where the arm rests, the pink canister 0.6 m off and 60-65 deg off
    straight down, at two places: the box where the reach-out's look then saw its button, within 15 mm — the first
    a top 72 x 158 mm ("wider than the lid's"), taken at its middle; no circle of the colour laid flat is asked."""
    from conftest import fixture_path

    model = ContainerModel.load(PINK)
    frame = wf.frame_from_capture(str(fixture_path(name)))
    find = wf.find_container([frame], model, check_buttons=False)
    assert find.fix is not None, find.why
    assert np.hypot(find.fix.top_xyz[0] - button[0], find.fix.top_xyz[1] - button[1]) < 0.015
    assert not any(q.rim for q in wf.frame_patches(frame, model)[0])
