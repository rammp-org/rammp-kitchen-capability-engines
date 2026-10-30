"""Where on the table there is room for the lid: what the scene camera (and
the wrist, at staging) see standing around the box, and the spots the lid
and the hand that carries it fit."""

import dataclasses
import math

import numpy as np
import pytest

from box_opening.geometry import ang_diff, yaw_about_world_z
from box_opening.models.container import ContainerModel
from box_opening.perception import clear_table as ct

PINK = "capabilities/box_opening/config/containers/ankou_pink.yaml"
DATA = "capabilities/box_opening/test/data/"
TABLE_Z = 0.187


@pytest.fixture(scope="module")
def pink():
    return ContainerModel.load(PINK)


def _flat(x0, x1, y0, y1, z, step=0.006):
    xs, ys = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step), indexing="ij")
    return np.c_[xs.ravel(), ys.ravel(), np.full(xs.size, float(z))]


def _block(x0, x1, y0, y1, top, step=0.006):
    """A thing standing on the table: its top, and its four sides."""
    pts = [_flat(x0, x1, y0, y1, top, step)]
    for z in np.arange(TABLE_Z + 0.02, top, step):
        pts += [
            _flat(x0, x1, y0, y0 + step, z, step),
            _flat(x0, x1, y1 - step, y1, z, step),
            _flat(x0, x0 + step, y0, y1, z, step),
            _flat(x1 - step, x1, y0, y1, z, step),
        ]
    return np.concatenate(pts)


def _table(x0=0.25, x1=0.75, y0=-0.70, y1=0.20, holes=(), step=0.006):
    """The table's points, minus where things stand on it."""
    t = _flat(x0, x1, y0, y1, TABLE_Z, step)
    for hx0, hx1, hy0, hy1 in holes:
        t = t[~((t[:, 0] > hx0) & (t[:, 0] < hx1) & (t[:, 1] > hy0) & (t[:, 1] < hy1))]
    return t


def _spots(pink, tmap, box_xy, **kw):
    lid_h = pink.lid_dims[2]
    zone = kw.pop("zone", (0.28, 0.65, box_xy[1] - 0.45, box_xy[1] + 0.45))
    return ct.clear_spots(
        tmap,
        box_xy,
        zone,
        kw.pop("min_from_box", 0.198),
        lambda xy: pink.press_quat((xy[0], xy[1], TABLE_Z)),
        0.5 * pink.lid_dims[0],
        lid_h,
        pink.dims[2] + lid_h + 0.04,
        lid_h,
        **kw,
    )


BOX = (0.39, -0.20)
BOX_AT = (BOX[0] - 0.06, BOX[0] + 0.06, BOX[1] - 0.06, BOX[1] + 0.06)


def test_what_stands_is_measured_from_the_table_the_points_show_not_the_base_frames_z():
    """Through TF's mount the flatbot scene camera's table leans 6 deg: its
    far side read 3-4 cm "below" the table, and a plain z band called the
    whole far half off its edge. Heights are measured from the plane the
    same points show, so a leaning table is all table and what stands on it
    still stands."""
    lean = math.radians(6.0)
    t = _table(holes=[BOX_AT])
    pts = np.concatenate([t, _block(*BOX_AT, TABLE_Z + 0.121)])
    pts[:, 2] += (pts[:, 0] - BOX[0]) * math.tan(lean)  # the far side lower, as through a tilted mount
    pts[:, 2] -= 2 * (pts[:, 0] - BOX[0]) * math.tan(lean)
    tmap = ct.table_map([(pts, TABLE_Z)], BOX)
    X, Y = tmap.centres()
    on_table = (X > 0.55) & (X < 0.70) & (Y > -0.6) & (Y < 0.1)
    assert not np.isfinite(tmap.tall[on_table]).any() and not tmap.off[on_table].any()
    at_box = (np.abs(X - BOX[0]) < 0.04) & (np.abs(Y - BOX[1]) < 0.04)
    assert np.nanmin(tmap.tall[at_box]) == pytest.approx(0.121, abs=0.02)


def test_speckle_is_not_a_thing_and_a_thing_is():
    """A few flying pixels in a cell of table are not something standing
    there (17:57: two "13 cm" cells off the microwave's dark door); a thing
    a few cells across is."""
    rng = np.random.default_rng(3)
    t = _table()
    speck = np.c_[rng.uniform(0.45, 0.47, 6), rng.uniform(-0.50, -0.48, 6), np.full(6, TABLE_Z + 0.13)]
    can = _block(0.55, 0.61, 0.0, 0.06, TABLE_Z + 0.10)
    tmap = ct.table_map([(np.concatenate([t, speck, can]), TABLE_Z)], BOX)
    st = tmap.standing()
    assert not ((np.abs(st[:, 0] - 0.46) < 0.03) & (np.abs(st[:, 1] + 0.49) < 0.03)).any()
    assert ((st[:, 0] > 0.54) & (st[:, 0] < 0.62) & (st[:, 1] > -0.01) & (st[:, 1] < 0.07)).sum() >= 4


def test_the_1757_spot_had_no_room_and_the_spot_it_finds_does(pink):
    """Flatbot 2026-10-23 17:57, the scene camera's and the wrist's points
    (test/data/table_room_flatbot_20261023_1757.npz): the box at [0.394,
    -0.206] with a microwave standing 14 cm in front of it and a bin beside
    it. The lid went to [0.56, -0.37], on the microwave's front edge, and
    the wrist camera came down on the microwave. The map says that spot has
    no room, and finds one on the open table beyond the box, the hand turned
    so the wrist camera faces the open table."""
    f = np.load(DATA + "table_room_flatbot_20261023_1757.npz")
    shift = np.r_[f["wrist_box"] - f["scene_box"], 0.0]
    scene = f["scene_mm"] / 1000.0 + shift
    wrist = f["wrist_mm"] / 1000.0
    box = tuple(f["wrist_box"])
    tmap = ct.table_map([(scene, float(f["table_z"])), (wrist, float(f["table_z"]))], box)
    q = pink.press_quat((0.56, -0.37, TABLE_Z))
    room, _what = ct.clearance(tmap, (0.56, -0.37), q, 0.5 * pink.lid_dims[0], pink.lid_dims[2])
    assert room < -0.03
    spots, why = _spots(pink, tmap, box)
    assert spots and why is None, why
    best = spots[0]
    assert best.room_m >= 0.0
    assert math.hypot(best.xy[0] - 0.56, best.xy[1] + 0.37) > 0.25  # nowhere near the microwave
    assert best.xy[1] > box[1]  # the open side of the box
    # the scene camera alone cannot see that side (the box is at its frame's
    # edge), and edge-on it cannot vouch for the narrow band before the
    # microwave: no spot, and it says why
    alone = ct.table_map([(scene, float(f["table_z"]))], box)
    spots, why = _spots(pink, alone, box)
    assert not spots and "short of the room it needs" in why


def test_the_hand_turns_so_the_wrist_camera_faces_the_room(pink):
    """A band of table between the near edge of what the camera saw and a
    20 cm wall standing across it: at the press attitude the wrist camera
    (65 mm out along the hand's y, world +x there) reaches over toward the
    wall, and turned half round it reaches past the near edge; turned a
    quarter it lies along the band, and the lid fits."""
    wall = (0.53, 0.60, -0.70, 0.20)
    t = _table(x0=0.30, holes=[BOX_AT, wall])
    pts = np.concatenate([t, _block(*BOX_AT, TABLE_Z + 0.121), _block(*wall, TABLE_Z + 0.20)])
    tmap = ct.table_map([(pts, TABLE_Z)], BOX)
    xy = (0.42, -0.45)

    def room(turn):
        q = yaw_about_world_z(pink.press_quat((xy[0], xy[1], TABLE_Z)), turn)
        return ct.clearance(tmap, xy, q, 0.5 * pink.lid_dims[0], pink.lid_dims[2])

    assert room(0.0)[0] < 0 and room(0.0)[1] == "the wrist camera"
    assert room(math.pi)[0] < 0
    assert room(math.pi / 2)[0] >= 0 and room(-math.pi / 2)[0] >= 0
    spots, why = _spots(pink, tmap, BOX)
    assert any(abs(s.turn) == pytest.approx(math.pi / 2) and s.xy[0] >= 0.41 for s in spots), (spots, why)


def test_with_room_either_way_the_spot_the_hand_turns_less_to_goes_first(pink):
    """A band of table across the box, between a low block (5 cm: the
    fingers and the lid must miss it, the wrist camera rides over it) and
    a 13 cm wall (the wrist camera must miss it; the lid rides over it on
    the carry). The roomiest spot has the hand turned half round, the wrist
    camera facing away from the wall, with 45 mm or more to spare; the best
    with the hand unturned has less, but ENOUGH_M. That is room enough: the
    unturned spot goes first. 2026-10-29 12:02 took a spot with 46 mm to
    spare that the hand turned 91 deg to; joint_7 turned 151 deg, 6.0 s of
    the 7.8 s carry. Where only the turned spot has room, it is still the
    one."""
    low, wall = (0.20, 0.30, -0.70, 0.20), (0.575, 0.645, -0.70, 0.20)
    pts = np.concatenate(
        [
            _table(x0=0.20, holes=[BOX_AT, low, wall]),
            _block(*BOX_AT, TABLE_Z + 0.121),
            _block(*low, TABLE_Z + 0.05),
            _block(*wall, TABLE_Z + 0.13),
        ]
    )
    tmap = ct.table_map([(pts, TABLE_Z)], BOX)

    def room(xy, turn):
        q = yaw_about_world_z(pink.press_quat((xy[0], xy[1], TABLE_Z)), turn)
        return ct.clearance(tmap, xy, q, 0.5 * pink.lid_dims[0], pink.lid_dims[2])[0]

    roomiest = (0.44, -0.45)
    assert room(roomiest, math.pi) >= 0.045 and room(roomiest, 0.0) < 0
    spots, why = _spots(pink, tmap, BOX)
    assert spots and all(s.turn == 0.0 for s in spots), (spots, why)
    assert ct.ENOUGH_M <= spots[0].room_m < room(roomiest, math.pi) - 0.02
    # where only half round has room, the spot is still found, turned
    spots, why = _spots(pink, tmap, BOX, zone=(0.48, 0.48, -0.65, -0.45))
    assert spots and all(abs(s.turn) == pytest.approx(math.pi) for s in spots), why
    assert room(spots[0].xy, 0.0) < 0 and room(spots[0].xy, math.pi / 2) < 0 and room(spots[0].xy, -math.pi / 2) < 0


def test_the_hands_rotation_is_counted_from_where_it_holds_the_lid(pink):
    """Each spot's `rotation` is how far the hand turns about the vertical
    from where it holds the lid to where it comes down there, and it is
    what ranks the spots with room enough, not the turn from the press
    attitude alone. The press attitude is fixed in the world, so from the
    hand facing as at the box it is the turn; from a hand facing 115 deg
    the other way (the arm's FK says where), the spot it turns least to
    goes first: a quarter turn back. With the attitude steered to each
    spot's bearing, the spot's bearing round the box counts too."""
    tmap = ct.table_map([(np.concatenate([_table(holes=[BOX_AT]), _block(*BOX_AT, TABLE_Z + 0.121)]), TABLE_Z)], BOX)

    def same(a, b):
        return abs(ang_diff(a, b)) < 1e-9

    judged = []
    spots, _why = _spots(pink, tmap, BOX, judged_out=judged)
    assert spots[0].turn == 0.0 and all(same(s.rotation, s.turn) for s in judged)
    facing = ct.hand_yaw(pink.press_quat((BOX[0], BOX[1], TABLE_Z))) - 2.0
    judged = []
    spots, _why = _spots(pink, tmap, BOX, judged_out=judged, hand_at=facing)
    for s in judged:
        down = ct.hand_yaw(yaw_about_world_z(pink.press_quat((s.xy[0], s.xy[1], TABLE_Z)), s.turn))
        assert same(s.rotation, down - facing)
    assert spots and all(s.turn == pytest.approx(-math.pi / 2) for s in spots)
    assert spots[0].rotation == pytest.approx(2.0 - math.pi / 2)
    # steered: the hand faces each spot's bearing from the arm, turned
    steered = dataclasses.replace(pink, press_yaw_steer=True)
    judged = []
    spots, _why = _spots(steered, tmap, BOX, judged_out=judged)
    for s in judged:
        assert same(s.rotation, math.atan2(s.xy[1], s.xy[0]) - math.atan2(BOX[1], BOX[0]) + s.turn)
    assert spots[0].turn == 0.0 and abs(spots[0].rotation) < ct.ROTATION_STEP_RAD / 2


def test_the_room_judged_once_is_ranked_again_for_another_hand_without_judging_it_again(pink, monkeypatch):
    """reach_lid_off.room_ahead judges the room while the press flies, the
    way the hand will face not known yet; the grip's lookahead ranks the
    spots from that cache for the hand as it faces: the spots judged
    afresh, and no spot's room judged twice."""
    tmap = ct.table_map([(np.concatenate([_table(holes=[BOX_AT]), _block(*BOX_AT, TABLE_Z + 0.121)]), TABLE_Z)], BOX)
    facing = ct.hand_yaw(pink.press_quat((BOX[0], BOX[1], TABLE_Z))) - 2.0
    fresh = _spots(pink, tmap, BOX, hand_at=facing)
    cache = {}
    _spots(pink, tmap, BOX, cache=cache)  # as the press flies: the hand as at the box
    calls = []
    real_c, real_cc = ct.clearance, ct.carry_clearance
    monkeypatch.setattr(ct, "clearance", lambda *a, **k: calls.append(1) or real_c(*a, **k))
    monkeypatch.setattr(ct, "carry_clearance", lambda *a, **k: calls.append(1) or real_cc(*a, **k))
    again = _spots(pink, tmap, BOX, hand_at=facing, cache=cache)
    assert again == fresh and fresh[0] and calls == []


def test_no_lid_goes_past_the_tables_edge_or_where_the_camera_could_not_see(pink):
    """Past the table's far edge the camera sees the floor: no lid goes
    down there, and none where it saw nothing at all."""
    t = _table(x1=0.50, holes=[BOX_AT])
    floor = _flat(0.50, 0.80, -0.70, 0.20, TABLE_Z - 0.70)
    pts = np.concatenate([t, floor, _block(*BOX_AT, TABLE_Z + 0.121)])
    tmap = ct.table_map([(pts, TABLE_Z)], BOX)
    spots, _why = _spots(pink, tmap, BOX)
    lid_r = 0.5 * pink.lid_dims[0]
    assert spots and all(s.xy[0] + lid_r <= 0.50 for s in spots)
    unseen = ct.table_map(
        [(np.concatenate([_table(y0=-0.70, y1=-0.10, holes=[BOX_AT]), _block(*BOX_AT, TABLE_Z + 0.121)]), TABLE_Z)], BOX
    )
    spots, _why = _spots(pink, unseen, BOX)
    assert spots and all(s.xy[1] + lid_r <= -0.10 for s in spots)


def test_the_carry_clears_what_stands_between_the_box_and_the_spot(pink):
    """The hand flies the lid over at the carry's height, the wrist camera
    ~29 cm up: a tall thing between the box and a spot that is clear itself
    rules that spot out; a short one does not."""
    lid_h = pink.lid_dims[2]
    carry_tips = pink.dims[2] + lid_h + 0.04
    post = (0.37, 0.41, -0.06, -0.02)
    t = _table(holes=[BOX_AT, post])
    for top, blocked in ((TABLE_Z + 0.40, True), (TABLE_Z + 0.08, False)):
        pts = np.concatenate([t, _block(*BOX_AT, TABLE_Z + 0.121), _block(*post, top)])
        tmap = ct.table_map([(pts, TABLE_Z)], BOX)
        room, what, _at = ct.carry_clearance(tmap, BOX, (0.39, 0.10), 0.5 * pink.lid_dims[0], carry_tips, lid_h)
        assert (room < 0) == blocked, (room, what)


def test_the_wrist_seeing_a_cell_bare_outvotes_the_scene_cameras_speckle_there():
    """Edge-on, the scene camera speckles a patch of table with points
    standing a few cm up; the wrist, looking straight down, saw it bare.
    Their points pool, and the bare table wins."""
    rng = np.random.default_rng(5)
    patch = np.c_[rng.uniform(0.44, 0.50, 60), rng.uniform(-0.46, -0.40, 60), TABLE_Z + rng.uniform(0.03, 0.08, 60)]
    scene = np.concatenate([_table(step=0.012), patch])
    wrist = _flat(0.30, 0.60, -0.55, -0.30, TABLE_Z, step=0.004)
    for clouds, stands in (([(scene, TABLE_Z)], True), ([(scene, TABLE_Z), (wrist, TABLE_Z)], False)):
        tmap = ct.table_map(clouds, BOX)
        X, Y = tmap.centres()
        at = (np.abs(X - 0.47) < 0.03) & (np.abs(Y + 0.43) < 0.03)
        assert np.isfinite(tmap.tall[at]).any() == stands


def test_the_wrist_frame_is_points_in_the_base_frame():
    """A frame straight down from 0.35 m over a table: every point lands on it."""
    k = np.array([[400.0, 0, 320], [0, 400.0, 200], [0, 0, 1]])
    depth = np.full((400, 640), 0.35, np.float32)
    rot = np.diag([1.0, -1.0, -1.0])  # looking down
    pts = ct.depth_points(depth, k, rot, [0.45, -0.2, TABLE_Z + 0.35])
    assert np.allclose(pts[:, 2], TABLE_Z, atol=1e-6)
    assert pts[:, 0].min() == pytest.approx(0.45 - 320 / 400 * 0.35, abs=0.01)


def _covered(boxes, x, y, z=None):
    """Is (x, y) — up to height z, when given — inside one of the cuboids?"""
    for b in boxes:
        (bx, by, bz), (dx, dy, dz) = b["position"], b["dims"]
        if abs(x - bx) <= dx / 2 + 1e-9 and abs(y - by) <= dy / 2 + 1e-9 and (z is None or z <= bz + dz / 2 + 1e-9):
            return True
    return False


def _1757_map():
    f = np.load(DATA + "table_room_flatbot_20261023_1757.npz")
    scene = f["scene_mm"] / 1000.0 + np.r_[f["wrist_box"] - f["scene_box"], 0.0]
    box, tz = tuple(f["wrist_box"]), float(f["table_z"])
    return ct.table_map([(scene, tz), (f["wrist_mm"] / 1000.0, tz)], box), box, tz


def test_what_stands_becomes_the_planners_cuboids_never_over_the_box_or_the_lid_spot(pink):
    """17:57's map as the planner's world (press_demo.adopt_seen): what
    stands is inside cuboids at least as tall — the microwave the wrist
    camera came down on among them — none covers the box (its own cuboid
    does) or free table where the hand comes down at the lid's spot, and
    they fit in the planner's cache."""
    tm, box, tz = _1757_map()
    spots, _why = _spots(pink, tm, box)
    boxes, note = ct.obstacle_boxes(tm, tz, box, 0.0525, keep_clear=[s.xy for s in spots])
    assert 0 < len(boxes) <= ct.PLANNER_BOXES and note["left_out"] == 0
    assert all(b["name"].startswith("seen_") for b in boxes)
    assert all(b["position"][2] - b["dims"][2] / 2 == pytest.approx(tz) for b in boxes)  # standing on the table
    X, Y = tm.centres()
    stands = np.isfinite(tm.tall)
    reach = 0.0525 + ct.BOX_CLEAR_M
    by_box = (np.abs(X - box[0]) - 0.01 < reach) & (np.abs(Y - box[1]) - 0.01 < reach)
    kept = [
        _covered(boxes, X[i, j], Y[i, j], tz + tm.tall[i, j])
        for i, j in zip(*np.nonzero(stands & ~by_box), strict=False)
    ]
    assert np.mean(kept) > 0.97  # all but the box's remnants
    assert not any(_covered(boxes, X[i, j], Y[i, j]) for i, j in zip(*np.nonzero(by_box), strict=False))
    for s in spots:
        gap = np.hypot(ct._to_square(X - s.xy[0], 0.01), ct._to_square(Y - s.xy[1], 0.01))
        near = ~stands & (gap <= ct.KEEP_CLEAR_M)
        assert not any(_covered(boxes, X[i, j], Y[i, j]) for i, j in zip(*np.nonzero(near), strict=False))
    assert _covered(boxes, 0.56, -0.37, tz + 0.13)  # where the lid went at 17:57, as high as the camera hit


def test_what_stands_against_the_arm_is_the_arm(pink):
    """A frame with the arm in it: what reaches within ARM_CLEAR_M of the
    arm's centre line is left out — in the planner's world it would put
    every start in collision — and nothing merged covers the arm's place."""
    post = (0.53, 0.57, -0.02, 0.02)
    pts = np.concatenate(
        [_table(holes=[BOX_AT, post]), _block(*BOX_AT, TABLE_Z + 0.121), _block(*post, TABLE_Z + 0.30)]
    )
    tm = ct.table_map([(pts, TABLE_Z)], BOX)
    low_arm = [(0.40, 0.0, TABLE_Z + 0.35), (0.70, 0.0, TABLE_Z + 0.35)]  # 5 cm over the post
    boxes, note = ct.obstacle_boxes(tm, TABLE_Z, BOX, 0.0525, arm_points=low_arm)
    assert note["under_arm"] > 0 and not _covered(boxes, 0.55, 0.0)
    high_arm = [(0.40, 0.0, TABLE_Z + 0.60), (0.70, 0.0, TABLE_Z + 0.60)]
    boxes, note = ct.obstacle_boxes(tm, TABLE_Z, BOX, 0.0525, arm_points=high_arm)
    assert note["under_arm"] == 0 and _covered(boxes, 0.55, 0.0, TABLE_Z + 0.29)


def test_more_things_than_the_planner_holds_merge_into_fewer_cuboids_around_the_lid_spot():
    """55 posts are more cuboids than the planner's cache takes: they
    merge, each still inside a cuboid as tall, none over the box or the
    free table by the lid's spot."""
    box, spot = (0.48, -0.20), (0.48, 0.06)
    posts = [
        (x - 0.02, x + 0.02, y - 0.02, y + 0.02)
        for x in np.arange(0.24, 0.73, 0.08)
        for y in np.arange(-0.56, 0.17, 0.08)
        if math.hypot(x - box[0], y - box[1]) >= 0.14 and math.hypot(x - spot[0], y - spot[1]) >= 0.12
    ]
    pts = np.concatenate(
        [_table(x0=0.20, x1=0.80, y0=-0.62, y1=0.22, holes=posts)]
        + [_block(*p, TABLE_Z + 0.05 + 0.002 * k) for k, p in enumerate(posts)]
    )
    tm = ct.table_map([(pts, TABLE_Z)], box)
    assert len(posts) > ct.PLANNER_BOXES
    boxes, note = ct.obstacle_boxes(tm, TABLE_Z, box, 0.0525, keep_clear=[spot])
    assert len(boxes) == ct.PLANNER_BOXES and note["left_out"] == 0 and note["free_blocked"] > 0
    for k, (x0, x1, y0, y1) in enumerate(posts):
        assert _covered(boxes, 0.5 * (x0 + x1), 0.5 * (y0 + y1), TABLE_Z + 0.05 + 0.002 * k - 0.01)
    X, Y = tm.centres()
    near = ~np.isfinite(tm.tall) & (
        np.hypot(ct._to_square(X - spot[0], 0.01), ct._to_square(Y - spot[1], 0.01)) <= ct.KEEP_CLEAR_M
    )
    assert near.any() and not any(_covered(boxes, X[i, j], Y[i, j]) for i, j in zip(*np.nonzero(near), strict=False))
    assert not _covered(boxes, *box)
