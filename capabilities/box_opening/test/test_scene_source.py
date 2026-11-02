"""Finding the box from the scene camera: depth inside the detector's box,
at lid height, in base_link."""

from types import SimpleNamespace

import numpy as np
import pytest

from box_opening.models.container import ContainerModel
from box_opening.perception.scene_source import (
    LID_BAND_M,
    MIN_LID_POINTS,
    box_from_scene_points,
    table_from_scene_points,
)

CFG = "capabilities/box_opening/config/containers/oxo_pop.yaml"
TABLE_Z = -0.027


@pytest.fixture(scope="module")
def model():
    return ContainerModel.load(CFG)


def _scene(model, box_xy=(0.45, -0.12), yaw=0.3, n_table=4000, n_top=400, rng=None):
    """Points in base_link with pixels: a table, a box top at lid height,
    a second container elsewhere, and speckle above the table. The pixel
    layout is a simple map of x -> u and y -> v so a bbox can be drawn."""
    rng = np.random.default_rng(0) if rng is None else rng
    lid_z = TABLE_Z + model.dims[2]
    hx, hy = model.dims[0] / 2, model.dims[1] / 2
    c, s = np.cos(yaw), np.sin(yaw)
    local = rng.uniform([-hx, -hy], [hx, hy], (n_top, 2))
    top = np.c_[
        box_xy[0] + c * local[:, 0] - s * local[:, 1],
        box_xy[1] + s * local[:, 0] + c * local[:, 1],
        lid_z + rng.normal(0, 0.004, n_top),
    ]
    # the near face, as an oblique scene camera sees it: a vertical wall of
    # points on the box's own -x side, running from the table up to the lid
    fl = np.c_[np.full(300, -hx) + rng.normal(0, 0.003, 300), rng.uniform(-hy, hy, 300)]
    face = np.c_[
        box_xy[0] + c * fl[:, 0] - s * fl[:, 1],
        box_xy[1] + s * fl[:, 0] + c * fl[:, 1],
        rng.uniform(TABLE_Z, lid_z, 300),
    ]
    table = np.c_[
        rng.uniform(0.2, 0.8, n_table), rng.uniform(-0.5, 0.5, n_table), TABLE_Z + rng.normal(0, 0.006, n_table)
    ]
    other = np.c_[rng.uniform(0.55, 0.65, 200), rng.uniform(0.2, 0.3, 200), lid_z + rng.normal(0, 0.004, 200)]
    speckle = np.c_[rng.uniform(0.2, 0.8, 300), rng.uniform(-0.5, 0.5, 300), rng.uniform(-0.05, 0.3, 300)]
    pts = np.vstack([table, top, face, other, speckle])
    uv = np.c_[(pts[:, 1] + 0.5) * 1000, (0.8 - pts[:, 0]) * 1000]  # y -> u, x -> v
    return pts, uv


def _bbox_around(pts, uv, box_xy, half=0.08):  # the 4.1-inch box is 0.052 to a side
    sel = (np.abs(pts[:, 0] - box_xy[0]) < half) & (np.abs(pts[:, 1] - box_xy[1]) < half)
    return uv[sel, 0].min(), uv[sel, 1].min(), uv[sel, 0].max(), uv[sel, 1].max()


def test_the_box_is_where_the_lid_points_inside_the_pixel_box_are(model):
    pts, uv = _scene(model)
    bbox = _bbox_around(pts, uv, (0.45, -0.12))
    pose, top, n = box_from_scene_points(pts, uv, bbox, TABLE_Z, model)
    assert pose is not None
    assert pose.xyz[0] == pytest.approx(0.45, abs=0.006)
    assert pose.xyz[1] == pytest.approx(-0.12, abs=0.006)
    assert pose.xyz[2] == pytest.approx(TABLE_Z)  # origin pinned to the table
    assert top[2] == pytest.approx(TABLE_Z + model.dims[2], abs=0.005)
    assert n >= MIN_LID_POINTS
    d = abs((pose.yaw - 0.3 + np.pi / 4) % (np.pi / 2) - np.pi / 4)
    assert d < np.radians(6)  # yaw mod 90 from the rect


def test_the_near_face_does_not_pull_the_box_toward_the_camera(model):
    """The face's upper part lies inside the lid band; only the top slab
    may vote, or the centre walks toward the camera."""
    pts, uv = _scene(model)
    bbox = _bbox_around(pts, uv, (0.45, -0.12))
    pose, top, _n = box_from_scene_points(pts, uv, bbox, TABLE_Z, model)
    assert pose.xyz[0] == pytest.approx(0.45, abs=0.006)  # not pulled to -x
    assert top[2] == pytest.approx(TABLE_Z + model.dims[2], abs=0.005)


def test_the_other_container_does_not_pull_the_box(model):
    """A second container in the scene is outside the OWL box, so it is not
    counted, however box-like it is."""
    pts, uv = _scene(model)
    bbox = _bbox_around(pts, uv, (0.45, -0.12))
    pose, _top, _n = box_from_scene_points(pts, uv, bbox, TABLE_Z, model)
    assert abs(pose.xyz[1] + 0.12) < 0.01


def test_a_pixel_box_with_nothing_at_lid_height_is_refused(model):
    pts, uv = _scene(model)
    bbox = _bbox_around(pts, uv, (0.3, 0.3), half=0.03)  # bare table
    pose, why, n = box_from_scene_points(pts, uv, bbox, TABLE_Z, model)
    assert pose is None and "lid height" in why and n < MIN_LID_POINTS


def test_a_taller_things_face_crossing_lid_height_is_not_a_box(model):
    """With the real top out of the band, only the near face's strip is left
    at lid height — a strip, and refused as one."""
    rng = np.random.default_rng(3)
    pts, uv = _scene(model, rng=rng)
    pts = pts.copy()
    top = np.abs(pts[:, 2] - (TABLE_Z + model.dims[2])) < 0.02
    pts[top, 2] += 0.08  # a taller container: its top is well above the band
    bbox = _bbox_around(pts, uv, (0.45, -0.12))
    pose, why, _n = box_from_scene_points(pts, uv, bbox, TABLE_Z, model)
    assert pose is None and "strip" in why


def test_the_lid_band_is_the_detectors_own(model):
    """A box whose top sits further from nominal than the band is not this
    box (or the table height is wrong) — refused, as the wrist path does."""
    rng = np.random.default_rng(3)
    pts, uv = _scene(model, rng=rng)
    pts = pts.copy()
    top = np.abs(pts[:, 2] - (TABLE_Z + model.dims[2])) < 0.02
    pts[top, 2] += LID_BAND_M + 0.02
    bbox = _bbox_around(pts, uv, (0.45, -0.12))
    pose, _why, _n = box_from_scene_points(pts, uv, bbox, TABLE_Z, model)
    assert pose is None


def test_the_table_height_is_read_off_the_scene(model):
    pts, uv = _scene(model)
    z = table_from_scene_points(pts, uv, image_h=1000, row_frac=0.0)
    assert z == pytest.approx(TABLE_Z, abs=0.004)


def test_another_larger_container_in_the_pixel_box_is_not_this_one(model):
    """A 19x9 cm tray at the same height as the OXO (2026-09-17: the OWL's
    best box flipped between the two): its top is the wrong size."""
    rng = np.random.default_rng(5)
    pts, uv = _scene(model, rng=rng)
    lid_z = TABLE_Z + model.dims[2]
    tray = np.c_[rng.uniform(0.25, 0.44, 1200), rng.uniform(0.25, 0.34, 1200), lid_z + rng.normal(0, 0.004, 1200)]
    tray_uv = np.c_[(tray[:, 1] + 0.5) * 1000, (0.8 - tray[:, 0]) * 1000]
    pts = np.vstack([pts, tray])
    uv = np.vstack([uv, tray_uv])
    bbox = (tray_uv[:, 0].min() - 5, tray_uv[:, 1].min() - 5, tray_uv[:, 0].max() + 5, tray_uv[:, 1].max() + 5)
    pose, why, _n = box_from_scene_points(pts, uv, bbox, TABLE_Z, model)
    assert pose is None and "another container" in why
    # the real box in its own pixel box still passes
    pose, _top, _n = box_from_scene_points(pts, uv, _bbox_around(pts, uv, (0.45, -0.12)), TABLE_Z, model)
    assert pose is not None


def test_the_scene_fix_is_the_first_owl_box_whose_top_is_this_container():
    """scene_fix_from_boxes: the loop the live locate ran inline, as one
    function an offline replay can call — the OWL's boxes best first, each
    lifted to the lid slab, the first that is this container's size wins."""
    import numpy as np

    from box_opening.models.container import ContainerModel
    from box_opening.perception.scene_source import scene_fix_from_boxes

    model = ContainerModel.load("capabilities/box_opening/config/containers/ankou_pink.yaml")
    table_z = -0.027
    top = table_z + model.dims[2]
    # a lid of the model's size at [0.45, 0.05], seen as points with pixels
    xs, ys = np.meshgrid(np.linspace(-0.05, 0.05, 30), np.linspace(-0.05, 0.05, 30))
    lid = np.stack([0.45 + xs.ravel(), 0.05 + ys.ravel(), np.full(xs.size, top)], 1)
    table = np.stack([0.20 + xs.ravel(), -0.20 + ys.ravel(), np.full(xs.size, table_z)], 1)
    pts = np.vstack([lid, table])
    uv = np.vstack(
        [
            np.stack([100 + 30 * (xs.ravel() + 0.05) / 0.1, 100 + 30 * (ys.ravel() + 0.05) / 0.1], 1),
            np.stack([400 + 30 * (xs.ravel() + 0.05) / 0.1, 400 + 30 * (ys.ravel() + 0.05) / 0.1], 1),
        ]
    )
    boxes = [(0.60, [390, 390, 440, 440]), (0.40, [90, 90, 140, 140])]  # the best box is the bare table
    got = scene_fix_from_boxes(boxes, pts, uv, table_z, model)
    assert got.pose is not None and got.box == boxes[1][1]
    assert abs(got.top[0] - 0.45) < 0.005 and abs(got.top[1] - 0.05) < 0.005
    assert len(got.tried) == 1  # the table box, refused and said why
    none = scene_fix_from_boxes(boxes[:1], pts, uv, table_z, model)
    assert none.pose is None and none.tried
    # every reason beside the box it refused, the candidate's own object (perception/viz draws it at that box):
    # one refused at the frame's edge, one by the top-face lift, then the side lift's refusals
    edge = [0.0, 200.0, 40.0, 240.0]
    got = scene_fix_from_boxes([(0.70, edge), *boxes], pts, uv, table_z, model, image_wh=(640, 480))
    assert got.pose is not None and got.box is boxes[1][1]
    assert len(got.tried) == len(got.tried_boxes) == 2
    assert got.tried[0].startswith("it runs off the frame at its left edge") and got.tried_boxes[0] is edge
    assert "(OWL 0.60)" in got.tried[1] and got.tried_boxes[1] is boxes[0][1]
    side = scene_fix_from_boxes(
        [(0.70, edge), boxes[0]], pts, uv, table_z, model, cam_xyz=(0.3, -0.2, 1.0), image_wh=(640, 480)
    )
    assert side.pose is None and len(side.tried) == len(side.tried_boxes) >= 3
    assert side.tried_boxes[0] is edge and all(b is boxes[0][1] for b in side.tried_boxes[1:])
    assert any(t.startswith("from the side") for t in side.tried[1:]), side.tried


def _table_low_wall_high(rng, n=3000, h=1000):
    """A table filling the view's lower rows, a wall its upper ones."""
    table = np.c_[rng.uniform(0.3, 0.9, n), rng.uniform(-0.4, 0.4, n), TABLE_Z + rng.normal(0, 0.003, n)]
    wall = np.c_[1.2 + rng.normal(0, 0.003, n), rng.uniform(-0.4, 0.4, n), rng.uniform(0.0, 0.6, n)]
    uv = np.c_[rng.uniform(0, 1000, 2 * n), np.r_[rng.uniform(0.65 * h, h, n), rng.uniform(0, 0.35 * h, n)]]
    return np.vstack([table, wall]), uv


def test_the_table_is_read_off_the_lower_rows_of_the_upright_view():
    """The frames arrive upright (the OAKs rotate on the device since
    rammp-deployments #133): the table in front of the camera is in the
    image's lower rows. A view still upside down has the wall there, and
    no table is read off it."""
    pts, uv = _table_low_wall_high(np.random.default_rng(1))
    assert table_from_scene_points(pts, uv, image_h=1000) == pytest.approx(TABLE_Z, abs=0.004)
    over = np.c_[uv[:, 0], 1000 - uv[:, 1]]  # the same view, the camera turned over
    assert np.isnan(table_from_scene_points(pts, over, image_h=1000))  # the wall is not a table


def test_which_way_up_the_scene_camera_hangs_is_read_off_its_pose():
    from box_opening.perception.scene_source import image_upside_down

    def pose(x, y, z):  # the colour optical frame's axes in the base frame
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2] = x, y, z
        return T

    assert image_upside_down(pose((0, 1, 0), (0, 0, 1), (1, 0, 0))) is True  # flatbot's OAK, live TF 2026-10-22
    assert image_upside_down(pose((0, -1, 0), (0, 0, -1), (1, 0, 0))) is False
    assert image_upside_down(pose((0, 1, 0), (1, 0, 0), (0, 0, -1))) is None  # straight down: no up to speak of


def test_a_robot_whose_tf_carries_the_scene_camera_is_placed_by_tf_not_the_calibration_file():
    """Flatbot 2026-10-22: the installed camera_scene.yaml calibrates the
    dojo bench's camera to base_link; flatbot's mount is on TF (sheppy's
    static_tf). A fix through the wrong one would be flown to."""
    from box_opening.perception.scene_source import SceneLocator

    T_tf, T_cal = np.eye(4), np.eye(4)
    T_tf[:3, 3], T_cal[:3, 3] = (0.2, -0.5, 0.4), (-0.07, 0.64, 0.35)
    asked = []

    class Grab:
        link_frame = "scene_camera_color_optical_frame"

        def lookup(self, parent, child):
            asked.append((parent, child))
            return T_tf

        def link_to_color(self):
            return np.eye(4)

    loc = SceneLocator.__new__(SceneLocator)
    loc.grab, loc.T_base_link, loc.base_frame = Grab(), None, "arm_mount"
    assert np.allclose(loc.base_to_color(), T_tf) and asked == [("arm_mount", "scene_camera_color_optical_frame")]
    asked.clear()
    loc.T_base_link = T_cal  # a calibration file in force: TF is not asked
    assert np.allclose(loc.base_to_color(), T_cal) and asked == []


# A camera level with the lid, as flatbot's scene OAK sits: 0.2 m behind the
# table's near edge, 0.1 m above the table, looking along +x.
SIDE_CAM = np.array([0.2, -0.5, 0.40])
SIDE_TABLE = 0.30
PINK = "capabilities/box_opening/config/containers/ankou_pink.yaml"


def _project(pts, cam=SIDE_CAM, f=284.0, cx=320.0, cy=200.0):
    """Pixels of base-frame points in a level camera at `cam` looking along +x."""
    rel = pts - cam
    return np.c_[cx + f * (-rel[:, 1]) / rel[:, 0], cy + f * (-rel[:, 2]) / rel[:, 0]]


def _cylinder_front(centre, r, z0, z1, n, rng, cam=SIDE_CAM):
    """The half of a standing cylinder's side the camera sees."""
    face = np.arctan2(cam[1] - centre[1], cam[0] - centre[0])
    a = face + rng.uniform(-1.3, 1.3, n)
    return np.c_[centre[0] + r * np.cos(a), centre[1] + r * np.sin(a), rng.uniform(z0, z1, n)] + rng.normal(
        0, 0.002, (n, 3)
    )


def _side_scene(rng, box=(0.70, -0.55), r=0.0525, height=0.121):
    """Table, the canister's visible side, the wall behind it, a taller thing
    beside it and the streaks a level camera sees under the table."""
    table = np.c_[rng.uniform(0.45, 1.2, 8000), rng.uniform(-0.95, -0.1, 8000), SIDE_TABLE + rng.normal(0, 0.003, 8000)]
    table = table[np.hypot(table[:, 0] - box[0], table[:, 1] - box[1]) > r + 0.01]
    canister = _cylinder_front(box, r, SIDE_TABLE, SIDE_TABLE + height, 1500, rng)
    tall = _cylinder_front((0.70, -0.85), 0.05, SIDE_TABLE, SIDE_TABLE + 0.26, 1500, rng)
    wall = np.c_[np.full(4000, 1.30), rng.uniform(-0.95, -0.1, 4000), rng.uniform(0.30, 0.90, 4000)]
    under = np.c_[rng.uniform(0.45, 0.9, 4000), rng.uniform(-0.95, -0.1, 4000), rng.uniform(0.0, 0.28, 4000)]
    pts = np.vstack([table, canister, tall, wall, under])
    return pts, _project(pts), canister, tall


def _pixel_box(uv_part, pad=(2, 6)):
    """The OWL box around some points' pixels, padded the way a lid's box
    takes in some wall above it."""
    lo, hi = uv_part.min(axis=0), uv_part.max(axis=0)
    return lo[0] - pad[0], lo[1] - pad[1], hi[0] + pad[0], hi[1] + pad[0]


def test_the_table_around_a_box_is_its_commonest_height_not_what_is_under_it():
    from box_opening.perception.scene_source import table_around

    pts, _uv, _c, _t = _side_scene(np.random.default_rng(3))
    assert table_around(pts, (0.70, -0.55), below_z=0.39) == pytest.approx(SIDE_TABLE, abs=0.004)
    # a TF 3 deg off in pitch (the table rises away from the camera), and the
    # table behind the box hidden by it: the visible ring is all in front, so
    # its median sits low; the plane is read AT the box
    tilted = pts[~((pts[:, 0] > 0.70) & (np.abs(pts[:, 2] - SIDE_TABLE) < 0.02))].copy()
    tilted[:, 2] += 0.05 * (tilted[:, 0] - 0.70)
    assert table_around(tilted, (0.70, -0.55), below_z=0.39) == pytest.approx(SIDE_TABLE, abs=0.004)


def test_a_level_camera_finds_the_canister_from_its_side():
    """Flatbot 2026-10-22: its scene OAK sits level with the lid and cannot
    move. The pixel box takes in the wall behind the lid; the side lift
    keeps the nearest object, finds the table around it, and puts the
    centre half the canister's width behind the face it sees."""
    from box_opening.perception.scene_source import box_from_side_points

    model = ContainerModel.load(PINK)
    pts, uv, canister, tall = _side_scene(np.random.default_rng(4))
    cap = canister[:, 2] > SIDE_TABLE + model.dims[2] - 0.025
    bbox = _pixel_box(_project(canister[cap]))
    pose, top, n, table_z = box_from_side_points(pts, uv, bbox, None, model, SIDE_CAM)
    assert pose is not None, top
    assert np.hypot(pose.xyz[0] - 0.70, pose.xyz[1] + 0.55) < 0.012
    assert table_z == pytest.approx(SIDE_TABLE, abs=0.004) and pose.xyz[2] == pytest.approx(table_z)
    # the thing beside it is too tall to be it, whatever the OWL says
    pose, why, _n, _tz = box_from_side_points(
        pts, uv, _pixel_box(_project(tall[tall[:, 2] > SIDE_TABLE + 0.23])), None, model, SIDE_CAM
    )
    assert pose is None and "from lid height" in why
    # and a canister twice as wide is another container
    pts2, uv2, wide, _t = _side_scene(np.random.default_rng(5), r=0.11)
    bbox = _pixel_box(_project(wide[wide[:, 2] > SIDE_TABLE + model.dims[2] - 0.025]))
    pose, why, _n, _tz = box_from_side_points(pts2, uv2, bbox, None, model, SIDE_CAM)
    assert pose is None and "another container" in why


def test_flatbot_scene_frame_finds_the_pink_canister_from_the_side():
    """The recorded flatbot frame (2026-10-22, camera level with the lid,
    the canister in a blue holder, a black speaker beside it): the top-face
    lift refuses the canister — the wall behind it falls in its box — and
    the side lift finds it, refusing the speaker on height."""
    from pathlib import Path

    from box_opening.perception.scene import color_cloud
    from box_opening.perception.scene_source import scene_fix_from_boxes

    z = np.load(Path(__file__).parent / "data" / "scene_side_flatbot_20261022.npz")
    T = z["T_base_color"]
    pc, uv = color_cloud(z["depth_mm"].astype(float) / 1000.0, z["kd"], z["k"], z["dist"], z["T_color_depth"], stride=2)
    pts = pc @ T[:3, :3].T + T[:3, 3]
    model = ContainerModel.load(PINK)
    boxes = [(float(b[4]), [float(v) for v in b[:4]]) for b in z["owl_boxes"]]
    lid = boxes[0][1]
    top_face, why, _n = box_from_scene_points(pts, uv, lid, 0.30, model)
    assert top_face is None and "another container" in why
    got = scene_fix_from_boxes(boxes, pts, uv, None, model, cam_xyz=T[:3, 3])
    assert got.box == lid and not got.tried
    # where this frame puts it under the hand-measured TF: a regression pin, not a tape truth
    assert got.pose.xyz[0] == pytest.approx(0.675, abs=0.01) and got.pose.xyz[1] == pytest.approx(-0.578, abs=0.01)
    assert got.table_z == pytest.approx(0.30, abs=0.01)
    speaker = scene_fix_from_boxes(boxes[2:], pts, uv, None, model, cam_xyz=T[:3, 3])
    assert speaker.pose is None and "from lid height" in speaker.tried[0]


def test_the_table_is_measured_where_the_camera_saw_it():
    from box_opening.perception.scene_source import table_extent

    pts, _uv, _c, _t = _side_scene(np.random.default_rng(6))
    ext = table_extent(pts, SIDE_TABLE, (0.70, -0.55))
    assert ext is not None
    x0, x1, y0, y1 = ext
    assert 0.44 <= x0 <= 0.50 and 1.10 <= x1 <= 1.21  # the table spans x 0.45-1.2 in the scene
    assert -0.96 <= y0 <= -0.90 and -0.20 <= y1 <= -0.10  # ... and only 0.6 m around the box is read
    assert table_extent(pts, SIDE_TABLE + 0.2, (0.70, -0.55)) is None  # nothing at a height with no table


def test_a_measured_table_height_always_comes_with_somewhere_to_plan_it():
    """Review 2026-10-22: a height with too few points to measure its
    extent reached the planner as no table at all. It falls back to the
    ring square the height was read from, and says it assumed it."""
    from box_opening.perception.scene_source import MIN_TABLE_POINTS, TABLE_RING_M, seen_table_extent, table_extent

    pts, _uv, _c, _t = _side_scene(np.random.default_rng(6))
    assert seen_table_extent(pts, SIDE_TABLE, (0.70, -0.55)) == (table_extent(pts, SIDE_TABLE, (0.70, -0.55)), False)
    rng = np.random.default_rng(7)
    a, d = rng.uniform(0, 2 * np.pi, MIN_TABLE_POINTS - 1), rng.uniform(*TABLE_RING_M, MIN_TABLE_POINTS - 1)
    sparse = np.c_[0.70 + d * np.cos(a), -0.55 + d * np.sin(a), np.full(len(a), SIDE_TABLE)]
    assert table_extent(sparse, SIDE_TABLE, (0.70, -0.55)) is None
    ext, assumed = seen_table_extent(sparse, SIDE_TABLE, (0.70, -0.55))
    assert assumed and ext == pytest.approx((0.40, 1.00, -0.85, -0.25))


def test_a_camera_that_sees_the_top_face_keeps_the_top_face_lift(model):
    """Which view holds depends on how high the table stands against the
    camera, so both are tried: the top-face lift first."""
    from box_opening.perception.scene_source import scene_fix_from_boxes

    pts, uv = _scene(model)
    boxes = [(0.5, list(_bbox_around(pts, uv, (0.45, -0.12))))]
    got = scene_fix_from_boxes(boxes, pts, uv, TABLE_Z, model, cam_xyz=np.array([0.0, 0.0, 0.6]))
    assert got.pose is not None and got.lift == "top"


def test_a_tray_ranked_first_does_not_beat_the_box_from_the_side(model):
    """Every candidate's top-face lift before any side lift: the side lift
    never measures how far a container runs along the view, so the 19x9 cm
    tray the top-face lift refuses would pass it ahead of the real box."""
    from box_opening.perception.scene_source import scene_fix_from_boxes

    rng = np.random.default_rng(5)
    pts, uv = _scene(model, rng=rng)
    lid_z = TABLE_Z + model.dims[2]
    tray = np.c_[rng.uniform(0.25, 0.44, 1200), rng.uniform(0.25, 0.34, 1200), lid_z + rng.normal(0, 0.004, 1200)]
    tray_uv = np.c_[(tray[:, 1] + 0.5) * 1000, (0.8 - tray[:, 0]) * 1000]
    pts = np.vstack([pts, tray])
    uv = np.vstack([uv, tray_uv])
    tray_box = [tray_uv[:, 0].min() - 5, tray_uv[:, 1].min() - 5, tray_uv[:, 0].max() + 5, tray_uv[:, 1].max() + 5]
    true_box = list(_bbox_around(pts, uv, (0.45, -0.12)))
    got = scene_fix_from_boxes(
        [(0.26, tray_box), (0.255, true_box)], pts, uv, TABLE_Z, model, cam_xyz=np.array([0.0, 0.0, 0.6])
    )
    assert got.box == true_box and got.lift == "top", got.tried
    assert got.pose.xyz[0] == pytest.approx(0.45, abs=0.01) and got.pose.xyz[1] == pytest.approx(-0.12, abs=0.01)
    assert "another container" in got.tried[0]


def test_a_camera_level_with_the_lid_takes_the_side_lift():
    """The recorded flatbot frame of the 20:04 run (2026-10-22, camera z
    0.400, the pink lid's top ~0.397, a table height read off the scene):
    the top-face 'slab' is the lid's side band seen edge-on, and whether
    it passed decided the fix on a few pixels of the OWL box — 13 mm in xy,
    7.5 mm in lid height between runs of the same placement."""
    import itertools
    from pathlib import Path

    from box_opening.perception.scene import color_cloud
    from box_opening.perception.scene_source import scene_fix_from_boxes

    z = np.load(Path(__file__).parent / "data" / "scene_level_flatbot_20261022_2004.npz")
    T = z["T_base_color"]
    pc, uv = color_cloud(z["depth_mm"].astype(float) / 1000.0, z["kd"], z["k"], z["dist"], z["T_color_depth"], stride=2)
    pts = pc @ T[:3, :3].T + T[:3, 3]
    # recorded before the OAKs rotated on the device: turned upright as they now
    # are (rammp-deployments #133), the points where they were
    h, w = z["color"].shape[:2]
    uv = np.c_[w - 1 - uv[:, 0], h - 1 - uv[:, 1]]
    model = ContainerModel.load(PINK)
    table_z = table_from_scene_points(pts, uv, h)  # what locate() does unsurveyed
    assert not np.isnan(table_z)
    bx0, by0, bx1, by1 = (float(v) for v in z["lid_box"])
    x0, y0, x1, y1 = w - 1 - bx1, h - 1 - by1, w - 1 - bx0, h - 1 - by0
    tops = []
    for dx, dy in itertools.product((0, 2, 4), (0, 3, 6)):  # an OWL box a few pixels looser
        got = scene_fix_from_boxes(
            [(0.6, [x0 - dx, y0 - dy, x1 + dx, y1 + dy])], pts, uv, table_z, model, cam_xyz=T[:3, 3]
        )
        assert got.pose is not None and got.lift == "side", (dx, dy, got.lift, got.tried)
        tops.append(got.top)
    tops = np.asarray(tops)
    assert np.all(tops.max(0) - tops.min(0) < 0.010)


def test_the_top_face_is_seen_only_from_above_the_lid():
    import itertools

    from box_opening.perception.scene_source import sees_top_face

    lid_z = 0.094
    xs, ys = np.meshgrid(np.linspace(-0.05, 0.05, 11), np.linspace(-0.05, 0.05, 11))

    def lid_at(x, y):
        pts = np.c_[x + xs.ravel(), y + ys.ravel(), np.full(xs.size, lid_z)]
        uv = np.c_[100 + 100 * xs.ravel(), 100 + 100 * ys.ravel()]
        return pts, uv

    bbox = (90, 90, 110, 110)
    pts, uv = lid_at(0.5, 0.0)
    assert not sees_top_face(pts, uv, bbox, lid_z, (0.0, 0.0, lid_z + 0.003))  # level with it, 0.5 m away
    bench_cam = (-0.06506, 0.64157, 0.35206)  # config/camera_scene.yaml
    for x, y in itertools.product((0.3, 0.75), (-0.45, 0.45)):
        pts, uv = lid_at(x, y)
        assert sees_top_face(pts, uv, bbox, lid_z, bench_cam), (x, y)
    assert sees_top_face(
        pts, uv, (500, 500, 510, 510), lid_z, (0.0, 0.0, lid_z)
    )  # nothing in the box: its lift says why


def test_a_box_the_frame_cuts_off_is_refused_not_lifted():
    """Flatbot 2026-10-23: the canister half off the scene image's edge was
    fixed from the half it saw, 7 cm short of where the wrist camera then
    found it, and the staging aimed past its button. A candidate reaching
    the frame's edge is now refused before either lift."""
    from box_opening.perception.scene_source import scene_fix_from_boxes

    model = ContainerModel.load(PINK)
    pts, uv, canister, _tall = _side_scene(np.random.default_rng(4))
    cap = canister[:, 2] > SIDE_TABLE + model.dims[2] - 0.025
    box = list(_pixel_box(_project(canister[cap])))
    inside = scene_fix_from_boxes([(0.4, box)], pts, uv, None, model, cam_xyz=SIDE_CAM, image_wh=(640, 400))
    assert inside.pose is not None and not inside.clipped
    w = int(np.ceil(box[2])) + 1  # the same box, with the frame ending at its right side
    cut = scene_fix_from_boxes([(0.4, box)], pts, uv, None, model, cam_xyz=SIDE_CAM, image_wh=(w, 400))
    assert cut.pose is None
    assert cut.clipped == ((0.4, box, ("right",)),)
    assert cut.tried == ("it runs off the frame at its right edge (OWL 0.40)",)


def test_the_way_off_the_frame_is_said_in_the_planning_frame():
    """Flatbot's scene OAK hangs upside down looking along +x: the image's
    right is arm_mount +y, which is what the operator can act on."""
    from box_opening.perception.scene_source import frame_edges, off_frame_why

    assert frame_edges([2, 50, 300, 120], (640, 400)) == ("left",)
    assert frame_edges([100, 50, 638, 399], (640, 400)) == ("right", "bottom")
    assert frame_edges([10, 10, 600, 380], (640, 400)) == ()
    flatbot = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # columns: image +u, +v, view
    why = off_frame_why(0.45, ("right",), flatbot)
    assert "toward arm_mount +y" in why and "move the box toward -y" in why


def test_the_recorded_flatbot_frame_still_finds_its_canister_inside_the_frame():
    from pathlib import Path

    from box_opening.perception.scene import color_cloud
    from box_opening.perception.scene_source import scene_fix_from_boxes

    z = np.load(Path(__file__).parent / "data" / "scene_side_flatbot_20261022.npz")
    T = z["T_base_color"]
    pc, uv = color_cloud(z["depth_mm"].astype(float) / 1000.0, z["kd"], z["k"], z["dist"], z["T_color_depth"], stride=2)
    pts = pc @ T[:3, :3].T + T[:3, 3]
    boxes = [(float(b[4]), [float(v) for v in b[:4]]) for b in z["owl_boxes"]]
    got = scene_fix_from_boxes(boxes, pts, uv, None, ContainerModel.load(PINK), cam_xyz=T[:3, 3], image_wh=(640, 400))
    assert got.box == boxes[0][1] and not got.clipped


# A level camera like SIDE_CAM: image +u along base -y, +v along -z (_project)
SIDE_R = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def _cut_side_scene(rng, cut_u):
    """_side_scene's canister with the frame ending at column `cut_u`: what
    lies past it is not in view."""
    pts, uv, canister, _tall = _side_scene(rng)
    keep = uv[:, 0] < cut_u
    cuv = _project(canister)
    model = ContainerModel.load(PINK)
    cap = (canister[:, 2] > SIDE_TABLE + model.dims[2] - 0.025) & (cuv[:, 0] < cut_u)
    x0, y0, x1, y1 = _pixel_box(cuv[cap])
    return pts[keep], uv[keep], [x0, y0, cut_u - 1.0, y1], model


def test_a_round_box_the_frame_cuts_at_its_side_is_finished_from_its_lid():
    from box_opening.perception.scene_source import scene_fix_from_boxes

    pts, uv, box, model = _cut_side_scene(np.random.default_rng(6), cut_u=350)  # the canister spans u ~318..378
    got = scene_fix_from_boxes([(0.45, box)], pts, uv, None, model, cam_xyz=SIDE_CAM, image_wh=(350, 400), cam_R=SIDE_R)
    assert got.pose is not None, got.tried
    assert got.partial == ("right",) and not got.clipped
    assert np.hypot(got.pose.xyz[0] - 0.70, got.pose.xyz[1] + 0.55) < 0.01
    # the side lift without it: what is in view is not centred on the box
    plain = scene_fix_from_boxes([(0.45, box)], pts, uv, None, model, cam_xyz=SIDE_CAM)
    assert plain.pose is None or np.hypot(plain.pose.xyz[0] - 0.70, plain.pose.xyz[1] + 0.55) > 0.015


def test_what_cannot_be_finished_from_part_of_it_is_still_refused():
    from dataclasses import replace

    from box_opening.perception.scene_source import scene_fix_from_boxes

    pts, uv, box, model = _cut_side_scene(np.random.default_rng(6), cut_u=350)
    kw = dict(cam_xyz=SIDE_CAM, image_wh=(350, 400))
    # a square container: no circle to finish it with
    square = scene_fix_from_boxes([(0.45, box)], pts, uv, None, replace(model, round=False), cam_R=SIDE_R, **kw)
    assert square.pose is None and square.clipped and "runs off the frame at its right edge" in square.tried[0]
    # cut at its top or bottom: the lid itself may be what is out of view
    hung = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])  # image +u along base z
    tall = scene_fix_from_boxes([(0.45, box)], pts, uv, None, model, cam_R=hung, **kw)
    assert tall.pose is None and tall.clipped
    # too little of the lid in view: no circle is fitted to a sliver
    pts2, uv2, box2, _m = _cut_side_scene(np.random.default_rng(6), cut_u=326)
    sliver = scene_fix_from_boxes(
        [(0.45, box2)], pts2, uv2, None, model, cam_xyz=SIDE_CAM, image_wh=(326, 400), cam_R=SIDE_R
    )
    assert sliver.pose is None and sliver.clipped and "cut off at the frame's right edge" in sliver.tried[0]


def test_the_cut_flatbot_frame_puts_the_canister_where_the_wrist_found_it():
    """Flatbot 2026-10-23 13:33: the canister half off the scene OAK's frame.
    The run's side lift put it at [0.475, -0.276]; at staging the wrist
    camera found its lid near [0.45, -0.21], through its own hand-measured
    mount. Finished from its lid's arc, the scene puts it within 3 cm of
    that: a regression pin, not a tape truth."""
    from pathlib import Path

    from box_opening.perception.scene import color_cloud
    from box_opening.perception.scene_source import scene_fix_from_boxes

    z = np.load(Path(__file__).parent / "data" / "scene_side_flatbot_cut_20261023.npz")
    T = z["T_base_color"]
    pc, uv = color_cloud(z["depth_mm"].astype(float) / 1000.0, z["kd"], z["k"], z["dist"], z["T_color_depth"], stride=2)
    pts = pc @ T[:3, :3].T + T[:3, 3]
    boxes = [(float(b[4]), [float(v) for v in b[:4]]) for b in z["owl_boxes"]]
    got = scene_fix_from_boxes(
        boxes, pts, uv, None, ContainerModel.load(PINK), cam_xyz=T[:3, 3], image_wh=(640, 400), cam_R=T[:3, :3]
    )
    assert got.pose is not None, got.tried
    assert got.partial == ("right",)
    assert got.pose.xyz[0] == pytest.approx(0.446, abs=0.01) and got.pose.xyz[1] == pytest.approx(-0.236, abs=0.01)
    assert np.hypot(got.pose.xyz[0] - 0.45, got.pose.xyz[1] + 0.21) < 0.03


def test_however_tight_the_owl_box_on_a_cut_lid_the_fix_does_not_slide():
    """Live 2026-10-23 14:40: the same canister, standing still, fixed at
    [0.497, -0.285] in most frames and [0.445, -0.235] in the rest. The OWL
    box on a cut lid is tight, the arc in it short, and a circle of known
    radius slides along a short arc; both press runs staged 7 cm off. Over
    boxes from a sliver to the whole canister, a fix is where the canister
    stood or none."""
    from pathlib import Path

    from box_opening.perception.scene import color_cloud
    from box_opening.perception.scene_source import scene_fix_from_boxes

    z = np.load(Path(__file__).parent / "data" / "scene_side_flatbot_cut_20261023.npz")
    T = z["T_base_color"]
    pc, uv = color_cloud(z["depth_mm"].astype(float) / 1000.0, z["kd"], z["k"], z["dist"], z["T_color_depth"], stride=2)
    pts = pc @ T[:3, :3].T + T[:3, 3]
    model = ContainerModel.load(PINK)
    fixed = []
    for x0 in (440, 500, 560, 600):
        for y0, y1 in ((20, 180), (40, 160), (100, 185), (120, 240)):
            got = scene_fix_from_boxes(
                [(0.43, [x0, y0, 639, y1])],
                pts,
                uv,
                None,
                model,
                cam_xyz=T[:3, 3],
                image_wh=(640, 400),
                cam_R=T[:3, :3],
            )
            if got.pose is not None:
                fixed.append(got.pose.xyz[:2])
    assert len(fixed) >= 8
    off = [np.hypot(x - 0.445, y + 0.24) for x, y in fixed]
    assert max(off) < 0.03, off


def test_a_short_arc_is_no_circle():
    """Thirty degrees of a lid's side: the circle through it could stand
    anywhere along it, so no fix is made of it."""
    from box_opening.perception.scene_source import lid_circle

    model = ContainerModel.load(PINK)
    r = model.lid_dims[0] / 2
    a = np.linspace(np.pi - 0.26, np.pi + 0.26, 200)  # facing a camera on -x
    band = np.c_[0.5 + r * np.cos(a), r * np.sin(a), np.full(200, 0.38)] + np.random.default_rng(1).normal(
        0, 0.002, (200, 3)
    )
    centre, why = lid_circle(band, model, (0.0, 0.0, 0.38))
    assert centre is None and ("two places" in why or "deg of the lid's circle" in why)
    wide = np.linspace(np.pi - 1.2, np.pi + 1.2, 200)
    band = np.c_[0.5 + r * np.cos(wide), r * np.sin(wide), np.full(200, 0.38)] + np.random.default_rng(1).normal(
        0, 0.002, (200, 3)
    )
    centre, why = lid_circle(band, model, (0.0, 0.0, 0.38))
    assert centre is not None and np.hypot(centre[0] - 0.5, centre[1]) < 0.003


def test_a_cut_lids_mirror_across_its_own_silhouette_is_no_second_place():
    """Half out of the frame, a lid shows its depth only as a strip by its
    silhouette on the side the frame does not cut, and the circle tangent to
    that strip from the other side holds it too: the lid's mirror across its
    silhouette, 105 mm away (detbench P4, 2026-10-27). Two places, as the arc
    alone says; but the box's uncut side is that silhouette, the lid touches
    it and the mirror runs 105 mm past it."""
    from box_opening.perception.scene_source import lid_circle, past_sides

    model = ContainerModel.load(PINK)
    r = model.lid_dims[0] / 2
    cam = np.array([0.0, 0.0, 0.40])
    lid = np.array([0.40, 0.0])
    a = np.arcsin(r / np.linalg.norm(lid))
    ray = np.array([np.cos(a), -np.sin(a)])  # the silhouette on the lid's uncut side
    tangent = np.sqrt(lid @ lid - r * r) * ray
    mirror = 2 * (lid @ ray) * ray - lid
    rng = np.random.default_rng(1)
    t_lid = np.arctan2(tangent[1] - lid[1], tangent[0] - lid[0]) - rng.uniform(-0.040, 0.030, 130) / r
    t_mir = np.arctan2(tangent[1] - mirror[1], tangent[0] - mirror[0]) + rng.uniform(-0.040, -0.015, 50) / r
    xy = np.r_[lid + r * np.c_[np.cos(t_lid), np.sin(t_lid)], mirror + r * np.c_[np.cos(t_mir), np.sin(t_mir)]]
    band = np.c_[xy + rng.normal(0, 0.0015, xy.shape), np.full(len(xy), 0.38)]  # the lid's strip, the smear past it
    across = np.array([-ray[1], ray[0]])
    side = np.r_[-np.sign(lid @ across) * across, 0.0]  # the plane of that ray, its normal away from the lid
    assert past_sides([lid, mirror], (side,), cam, 0.38, r) == pytest.approx([0.0, 0.105], abs=0.001)
    centre, why = lid_circle(band, model, cam)
    assert centre is None and "two places" in why
    centre, why = lid_circle(band, model, cam, sides=(side,))
    assert centre is not None, why
    assert np.hypot(centre[0] - lid[0], centre[1] - lid[1]) < 0.005


def test_a_boxs_uncut_side_is_a_plane_through_the_camera_read_off_the_cloud():
    """The side of a cut box the frame does not reach is a plane through the
    camera, found from the cloud's own pixels (no intrinsics passed in): the
    points past it are the ones whose pixels are past the box. A box cut at
    both its sides has no uncut side to bound its lid."""
    from box_opening.perception.scene_source import pixel_plane, uncut_sides

    pts, uv, _canister, _tall = _side_scene(np.random.default_rng(3))
    n = pixel_plane(0, 300.0, pts, uv, SIDE_CAM, SIDE_R)
    clear = np.abs(uv[:, 0] - 300.0) > 0.01
    assert np.all(np.sign((pts[clear] - SIDE_CAM) @ n) == np.sign(uv[clear, 0] - 300.0))
    (right,) = uncut_sides([0.0, 100.0, 300.0, 200.0], ("left",), pts, uv, SIDE_CAM, SIDE_R)
    (left,) = uncut_sides([300.0, 100.0, 639.0, 200.0], ("right",), pts, uv, SIDE_CAM, SIDE_R)
    assert right / np.linalg.norm(right) == pytest.approx(n / np.linalg.norm(n), abs=1e-9)
    assert left / np.linalg.norm(left) == pytest.approx(-n / np.linalg.norm(n), abs=1e-9)
    assert uncut_sides([0.0, 100.0, 639.0, 200.0], ("left", "right"), pts, uv, SIDE_CAM, SIDE_R) == ()


def test_a_scene_camera_whose_tf_still_has_its_frames_upside_down_is_refused():
    """Nothing turns frames for the OWL any more: a camera hung upside down
    rotates them on the device (rammp-deployments #133). One whose TF still
    has its image's down pointing up is refused before any box is looked for,
    and says how to fix it."""
    from types import SimpleNamespace

    from box_opening.perception import scene_source as ss

    loc = ss.SceneLocator.__new__(ss.SceneLocator)
    loc.spins = False
    calls = []
    loc.owl = SimpleNamespace(enable=lambda: calls.append("on"), disable=lambda: calls.append("off"), latest=None)
    loc.seg = None
    t = [0.0]
    loc._clock = lambda: t[0]
    loc._sleep = lambda s: t.__setitem__(0, t[0] + s)
    loc.grab = SimpleNamespace(color_stamp=None, color=np.zeros((400, 640, 3), np.uint8), missing=lambda: [])
    loc.ready = lambda: True
    hung = np.eye(4)
    hung[:3, 0], hung[:3, 1], hung[:3, 2] = (0, 1, 0), (0, 0, 1), (1, 0, 0)  # image down points up
    loc.base_to_color = lambda: hung
    assert loc.locate(0.5) is None
    assert "upside down" in loc.last_why and "i_rotation 180" in loc.last_why
    assert calls[-1] == "off"  # the OWL's latched enable never outlives the call


# --- the scene segmenter's boxes through the gate, on upright flatbot frames
# (test/data/scene_upright_*.npz: the median depth, intrinsics and upright mount
# of a detbench frame, the food-container boxes RF-DETR-S published on it, the
# verified holder box and the reference lid top)

_FIXTURES = {}


def _upright(name):
    """(pts_base, uv, table rows z or None, gate kwargs, segmenter objects, the GT holder box, the reference top),
    as locate() lifts the frame: the stride-2 cloud in the base frame."""
    if name not in _FIXTURES:
        from pathlib import Path

        from box_opening.perception.scene import color_cloud

        z = np.load(Path(__file__).parent / "data" / name)
        T = z["T_base_color"]
        pc, uv = color_cloud(
            z["depth_mm"].astype(float) / 1000.0, z["kd"], z["k"], z["dist"], z["T_color_depth"], stride=2
        )
        pts = pc @ T[:3, :3].T + T[:3, 3]
        tz = table_from_scene_points(pts, uv, 400)
        kw = dict(cam_xyz=T[:3, 3], side_table_z=None, image_wh=(640, 400), cam_R=T[:3, :3])
        objs = [("food container", float(o[0]), tuple(float(v) for v in o[1:])) for o in z["seg_objects"]]
        _FIXTURES[name] = (
            pts,
            uv,
            None if np.isnan(tz) else float(tz),
            kw,
            objs,
            tuple(float(v) for v in z["seg_holder_gt"]),
            np.asarray(z["ref_xyz"], float),
        )
    return _FIXTURES[name]


F_162417 = "scene_upright_flatbot_20261027_162417.npz"  # the 10-27 16:24 frame the OWL missed ("saw no box in 15 s")
F_144302 = "scene_upright_flatbot_20261022_144302.npz"  # P1: something nearer than the canister inside its holder box
F_133403 = "scene_upright_flatbot_20261023_133403.npz"  # P4: the holder cut by the frame's left edge
F_172707 = (
    "scene_upright_flatbot_20261023_172707.npz"  # P4: the lid half off the frame's left edge, its arc fit two places
)
# P3 (scored difficult, no reference: ref_xyz NaN): the canister ~90 % off the frame's left edge, only its holder's brim
# tip and a 5 px sliver of its lid in view; cap_gt is the GT box of the lid's cap
F_133217 = "scene_upright_flatbot_20261023_133217.npz"


def _ray_cast_cylinder(depth, k, T, centre_xy, rho, z0, z1):
    """`depth` (m, the upright colour-aligned image of intrinsics `k`, base <- colour `T`) with a standing cylinder
    rendered into it, nearest surface kept: what a second object beside or behind the canister would show."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    ray = np.stack([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones_like(u)], -1) @ np.asarray(T)[:3, :3].T
    o = np.asarray(T)[:3, 3]
    bx, by = o[0] - centre_xy[0], o[1] - centre_xy[1]
    a = ray[..., 0] ** 2 + ray[..., 1] ** 2
    b = 2 * (ray[..., 0] * bx + ray[..., 1] * by)
    disc = b * b - 4 * a * (bx * bx + by * by - rho * rho)
    t = np.where(disc >= 0, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a), np.inf)  # optical z, the rays' z being 1
    zs = o[2] + t * ray[..., 2]
    t = np.where((t > 0) & (zs >= z0) & (zs <= z1), t, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        tc = (z1 - o[2]) / ray[..., 2]
    on_top = (tc > 0) & (
        np.hypot(o[0] + tc * ray[..., 0] - centre_xy[0], o[1] + tc * ray[..., 1] - centre_xy[1]) <= rho
    )
    t = np.where(on_top & (tc < t), tc, t)
    hit = np.isfinite(t) & ((depth <= 0) | (t < depth))
    return np.where(hit, t, depth)


def _upright_at(name, ox=0, oy=0, lo=0, cylinders=()):
    """_upright's frame lifted at pixel parity (ox, oy) — the stride-2 cloud from another pixel of each 2 x 2 — and,
    with `lo`, with the frame's left edge moved `lo` px in (the depth runs up to the new edge, as it does at a
    right-edge cut); `cylinders` (centre_xy, radius, z0, z1) are rendered into its depth first
    (_ray_cast_cylinder). (pts_base, uv, table rows z or None, gate kwargs, segmenter objects, the npz)."""
    from pathlib import Path

    from box_opening.perception.scene import color_cloud

    z = np.load(Path(__file__).parent / "data" / name)
    T = z["T_base_color"]
    k, kd = np.asarray(z["k"], float).copy(), np.asarray(z["kd"], float).copy()
    full = z["depth_mm"].astype(float) / 1000.0
    for c in cylinders:
        full = _ray_cast_cylinder(full, z["k"], T, *c)
    k[0, 2] -= lo
    kd[0, 2] -= lo + ox
    kd[1, 2] -= oy
    depth = full[oy:, lo + ox :]
    pc, uv = color_cloud(depth, kd, k, z["dist"], z["T_color_depth"], stride=2)
    pts = pc @ T[:3, :3].T + T[:3, 3]
    tz = table_from_scene_points(pts, uv, 400)
    w = 640 - lo
    kw = dict(cam_xyz=T[:3, 3], side_table_z=None, image_wh=(w, 400), cam_R=T[:3, :3])
    objs = []
    for o in z["seg_objects"]:
        x0, y0, x1, y1 = (float(v) for v in o[1:])
        if x1 - lo > 1.0:
            objs.append(("food container", float(o[0]), (max(x0 - lo, 0.0), y0, x1 - lo, y1)))
    return pts, uv, None if np.isnan(tz) else float(tz), kw, objs, z


def _cylinder_by_the_lid(name, model, clear, rho, height, along):
    """A standing cylinder (centre_xy, radius, z0, z1) for _upright_at's `cylinders`: `height` m tall on the
    fixture's table (the lid's reference top less the container's height), `rho` across, `clear` m clear of the lid,
    image right of it ("beside", into the frame for a left cut), straight behind it from the camera ("behind") or
    between ("beside_behind")."""
    _pts, _uv, _tz, kw, _objs, z = _upright_at(name)
    ref, cam = np.asarray(z["ref_xyz"], float), kw["cam_xyz"]
    right = kw["cam_R"][:2, 0] / np.linalg.norm(kw["cam_R"][:2, 0])
    view = (ref[:2] - cam[:2]) / np.linalg.norm(ref[:2] - cam[:2])
    d = {"beside": right, "behind": view, "beside_behind": (right + view) / np.linalg.norm(right + view)}[along]
    table = ref[2] - model.dims[2]
    return ref[:2] + (model.lid_dims[0] / 2 + clear + rho) * d, rho, table, table + height


def test_the_band_above_the_segmenters_holder_box_finds_the_canister_the_owl_missed():
    """Flatbot 2026-10-27 16:24: the scene OWL saw no box in 15 s (the pale
    lid under its 0.12 floor). RF-DETR boxed the blue holder at 0.88; as it
    is, the gate reads its top 55 mm below lid height. The band above it
    is where the lid stands."""
    from box_opening.perception.scene_source import scene_fix_from_boxes
    from box_opening.perception.segmenter_source import segmenter_candidates

    model = ContainerModel.load(PINK)
    pts, uv, tz, kw, objs, _gt, ref = _upright(F_162417)
    raw = scene_fix_from_boxes([(objs[0][1], objs[0][2])], pts, uv, tz, model, **kw)
    assert raw.pose is None and "from lid height" in raw.tried[0] and "-55 mm" in raw.tried[0]
    cands = segmenter_candidates(objs, ("food container",), (640, 400))
    got = scene_fix_from_boxes(cands, pts, uv, tz, model, src="segmenter", rivals=True, **kw)
    assert got.pose is not None, got.tried
    assert got.lift == "side" and got.rivals == () and got.n == 299
    assert np.linalg.norm(np.subtract(got.top, ref)) < 0.015  # measured 8.2 mm
    assert got.top == pytest.approx((0.6613, -0.2654, 0.3103), abs=0.0005)


def test_a_band_that_looks_past_its_holder_is_refused():
    """A band above some other bowl lifts whatever stands behind it: the GT
    holder moved sideways, offered alone, as a phantom bowl. The plain band
    is accepted half a metre from the canister; anchored to its holder's
    front, the gate sees the nearest thing above it stand far behind it."""
    from box_opening.perception.scene_source import scene_fix_from_boxes
    from box_opening.perception.segmenter_source import segmenter_candidates

    model = ContainerModel.load(PINK)
    pts, uv, tz, kw, _objs, (x0, y0, x1, y1), ref = _upright(F_162417)
    for dx, away_mm, behind_mm in ((-120, 473, 317), (180, 439, 168)):
        (c,) = segmenter_candidates(
            [("food container", 0.5, (x0 + dx, y0, x1 + dx, y1))], ("food container",), (640, 400)
        )
        plain = scene_fix_from_boxes([(c.score, c.box)], pts, uv, tz, model, **kw)
        assert plain.pose is not None
        assert 1000 * np.hypot(plain.top[0] - ref[0], plain.top[1] - ref[1]) == pytest.approx(away_mm, abs=3)
        anchored = scene_fix_from_boxes([c], pts, uv, tz, model, src="segmenter", **kw)
        assert anchored.pose is None
        why = anchored.tried[-1]
        assert "looks past" in why and ("stands %d mm behind the holder's front" % behind_mm) in why, why


def test_the_looks_past_refusal_never_moves_a_fix():
    """Flatbot 2026-10-22 14:43 (P1): something nearer than the canister
    lies inside its holder box. Design 3 cut the side lift's depth slab at
    the holder's front and put this frame 29 mm off here, 46-52 mm at the
    other pixel parity; as a refusal only, the anchored band's fix is the
    plain band's to the millimetre."""
    from box_opening.perception.scene_source import scene_fix_from_boxes
    from box_opening.perception.segmenter_source import segmenter_candidates

    model = ContainerModel.load(PINK)
    pts, uv, tz, kw, objs, _gt, ref = _upright(F_144302)
    cands = segmenter_candidates(objs, ("food container",), (640, 400))
    assert cands[0].holder is not None
    anchored = scene_fix_from_boxes(cands, pts, uv, tz, model, src="segmenter", **kw)
    plain = scene_fix_from_boxes([(c.score, c.box) for c in cands], pts, uv, tz, model, **kw)
    assert anchored.pose is not None and plain.pose is not None
    assert anchored.top == plain.top
    assert anchored.top == pytest.approx((0.6734, -0.5822, 0.4009), abs=0.0005)
    assert np.linalg.norm(np.subtract(anchored.top, ref)) < 0.006  # measured 4.1 mm


def test_two_places_that_pass_as_the_container_are_rivals_not_a_fix():
    """A phantom bowl 180 px to the canister's right on the P1 frame passes
    the gate even anchored; ranked first, it wins its message — and the
    canister's own band, passing too 30 cm away, makes the message
    ambiguous."""
    from box_opening.perception.scene_source import scene_fix_from_boxes
    from box_opening.perception.segmenter_source import segmenter_candidates

    model = ContainerModel.load(PINK)
    pts, uv, tz, kw, objs, (x0, y0, x1, y1), ref = _upright(F_144302)
    (phantom,) = segmenter_candidates(
        [("food container", 0.9, (x0 + 180, y0, x1 + 180, y1))], ("food container",), (640, 400)
    )
    (canister,) = segmenter_candidates(objs, ("food container",), (640, 400))
    got = scene_fix_from_boxes([phantom, canister], pts, uv, tz, model, src="segmenter", rivals=True, **kw)
    assert got.pose is not None and tuple(got.box) == tuple(phantom.box)
    assert len(got.rivals) == 1 and np.linalg.norm(np.subtract(got.rivals[0], ref)) < 0.010
    alone = scene_fix_from_boxes([phantom, canister], pts, uv, tz, model, src="segmenter", **kw)
    assert alone.rivals == ()  # only asked for


def test_a_holder_the_frame_cuts_is_grown_and_finished_from_its_lid_in_view():
    """Flatbot 2026-10-23 13:34 (P4): the holder runs off the frame's left
    edge. Grown upward with its cut edge kept, it reaches the gate's
    partial lift, which finishes the lid from its arc in view."""
    from box_opening.perception.scene_source import scene_fix_from_boxes
    from box_opening.perception.segmenter_source import segmenter_candidates

    model = ContainerModel.load(PINK)
    pts, uv, tz, kw, objs, _gt, ref = _upright(F_133403)
    (c,) = segmenter_candidates(objs, ("food container",), (640, 400))
    assert c.kind == "grown" and c.holder is None and c.box[0] < 1.0
    got = scene_fix_from_boxes([c], pts, uv, tz, model, src="segmenter", rivals=True, **kw)
    assert got.pose is not None, got.tried
    assert got.partial == ("left",)
    assert np.linalg.norm(np.subtract(got.top, ref)) < 0.020  # measured 13.4 mm


def test_the_lid_half_off_the_frame_is_one_place_inside_its_boxs_uncut_side(monkeypatch):
    """Flatbot 2026-10-23 17:27 (detbench P4): the canister stands half off
    the scene frame's left edge, and the gate refused every sample of it:
    its lid's arc fit two places. The second is the lid's mirror across its
    own right silhouette, 103 mm off, and it holds more of the arc than the
    lid: only the rival rule kept it from being the fix. It runs past the
    grown holder box's right side, which the lid stays inside: one place,
    11 mm from the reference."""
    from box_opening.perception import scene_source as ss
    from box_opening.perception.segmenter_source import segmenter_candidates

    model = ContainerModel.load(PINK)
    pts, uv, tz, kw, objs, _gt, ref = _upright(F_172707)
    (c,) = segmenter_candidates(objs, ("food container",), (640, 400))
    assert c.kind == "grown" and c.box[0] < 1.0
    got = ss.scene_fix_from_boxes([c], pts, uv, tz, model, src="segmenter", rivals=True, **kw)
    assert got.pose is not None, got.tried
    assert got.partial == ("left",) and got.rivals == ()
    assert np.linalg.norm(np.subtract(got.top, ref)) < 0.015  # measured 11.2 mm
    assert got.top == pytest.approx((0.3966, -0.1725, 0.2829), abs=0.0005)
    # the bound alone picks the lid: no rival rule, the same fix
    monkeypatch.setattr(ss, "PARTIAL_RIVAL_FRAC", 2.0)
    alone = ss.scene_fix_from_boxes([c], pts, uv, tz, model, src="segmenter", **kw)
    assert alone.top == got.top
    # without the bound, the mirror: two places, or with no rival rule a fix 103 mm off
    monkeypatch.setattr(ss, "PARTIAL_BOX_SLACK_M", float("inf"))
    mirror = ss.scene_fix_from_boxes([c], pts, uv, tz, model, src="segmenter", **kw)
    assert mirror.pose is not None and np.hypot(mirror.top[0] - ref[0], mirror.top[1] - ref[1]) > 0.08
    monkeypatch.setattr(ss, "PARTIAL_RIVAL_FRAC", 0.85)
    two = ss.scene_fix_from_boxes([c], pts, uv, tz, model, src="segmenter", **kw)
    assert two.pose is None and "two places" in two.tried[-1]


def test_at_1727_the_half_off_canisters_box_side_moved_either_way_fixes_its_lid_or_nothing():
    """The uncut side the bound reads is the detector's: it may sit a little
    inside the lid's silhouette or well outside it (a holder's, or a loose
    box). Moved 40 px either way, the half-off canister at 17:27 is still
    fixed where it was; moved further, it is refused, never fixed somewhere
    else. One frame's measurement, not a rule: at 15:09, the OWL boxes'
    uncut side pulled 40 px into the lid fixes it 32 mm off (c693865:
    13.6 mm)."""
    from box_opening.perception.scene_source import scene_fix_from_boxes
    from box_opening.perception.segmenter_source import segmenter_candidates

    model = ContainerModel.load(PINK)
    pts, uv, tz, kw, objs, _gt, ref = _upright(F_172707)
    (c,) = segmenter_candidates(objs, ("food container",), (640, 400))
    x0, y0, x1, y1 = c.box
    for dx in (-60, -40, -20, 0, 20, 40, 60, 100):
        got = scene_fix_from_boxes([c._replace(box=(x0, y0, x1 + dx, y1))], pts, uv, tz, model, src="segmenter", **kw)
        if abs(dx) <= 40:
            assert got.pose is not None, (dx, got.tried)
        if got.pose is not None:
            assert np.linalg.norm(np.subtract(got.top, ref)) < 0.015, dx  # measured 10.6-11.3 mm


def test_a_cut_lid_inside_its_boxs_uncut_side_is_fixed_where_it_was(monkeypatch):
    """The bound only takes out circles that run past the box: where the
    best one stands inside it, the fix is the arc's own to the bit (detbench:
    no fix moved within its box, the 28 of 3,552 samples that moved changed
    winning box, the detector's first now passing)."""
    from box_opening.perception import scene_source as ss
    from box_opening.perception.segmenter_source import segmenter_candidates

    model = ContainerModel.load(PINK)
    pts, uv, tz, kw, objs, _gt, _ref = _upright(F_133403)
    cands = segmenter_candidates(objs, ("food container",), (640, 400))
    got = ss.scene_fix_from_boxes(cands, pts, uv, tz, model, src="segmenter", rivals=True, **kw)
    monkeypatch.setattr(ss, "PARTIAL_BOX_SLACK_M", float("inf"))
    plain = ss.scene_fix_from_boxes(cands, pts, uv, tz, model, src="segmenter", rivals=True, **kw)
    assert got.pose is not None and got.partial == ("left",)
    assert got.top == plain.top and got.n == plain.n


def test_a_cut_box_that_holds_only_the_holder_under_the_lid_is_refused():
    """A cut box around the holder alone (RF-DETR's holder box as it is, an
    OWL "a pink lid" that boxed the brim) lifts its top at the brim, under
    the lid's but inside the lid band, and the lid's circle fitted to the
    brim put the box 43-47 mm off on detbench P4 once the lid's mirror was no
    second place. The lid stands over that band: the box holds what is under
    the lid, and is refused. Grown over the lid, the same holder is the fix."""
    from box_opening.perception.scene_source import scene_fix_from_boxes
    from box_opening.perception.segmenter_source import segmenter_candidates

    model = ContainerModel.load(PINK)
    pts, uv, tz, kw, objs, _gt, ref = _upright(F_133403)
    as_is = scene_fix_from_boxes([(objs[0][1], list(objs[0][2]))], pts, uv, tz, model, **kw)
    assert as_is.pose is None and "under the lid" in as_is.tried[-1], as_is.tried
    (c,) = segmenter_candidates(objs, ("food container",), (640, 400))
    grown = scene_fix_from_boxes([c], pts, uv, tz, model, src="segmenter", **kw)
    assert grown.pose is not None and np.linalg.norm(np.subtract(grown.top, ref)) < 0.020  # measured 13.4 mm


def test_the_lid_all_but_out_of_view_settles_no_two_places(monkeypatch):
    """Flatbot 2026-10-23 13:32 (detbench, scored difficult): the canister ~90 %
    off the frame's left edge, only its holder's brim tip and a 5 px sliver of
    its lid in view. The band is the brim tip, and the circles through it are
    guesses: two places, as c693865 said. The box's side takes one out, and the
    one it keeps (28-45 points) stands 51 mm past the lid's own silhouette;
    b2fb875 fixed it, and the stream committed it in 300 of 300 runs. Too few
    points to settle two places on, at every pixel parity."""
    from box_opening.perception import scene_source as ss
    from box_opening.perception.segmenter_source import segmenter_candidates

    model = ContainerModel.load(PINK)
    r = model.lid_dims[0] / 2
    for ox, oy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        pts, uv, tz, kw, objs, z = _upright_at(F_133217, ox, oy)
        cands = segmenter_candidates(objs, ("food container",), (640, 400))
        assert [c.kind for c in cands] == ["grown"]
        got = ss.scene_fix_from_boxes(cands, pts, uv, tz, model, src="segmenter", rivals=True, **kw)
        assert got.pose is None and got.partial == () and "fits two places" in got.tried[-1], ((ox, oy), got.tried)
    assert "too few to settle it (need 100)" in got.tried[-1]
    # with no floor, (0, 1) and (1, 1) fix the brim tip, the circle a lid's radius past the lid's own silhouette (the
    # right edge of its cap's GT box)
    monkeypatch.setattr(ss, "PARTIAL_SETTLE_MIN_POINTS", 0)
    for ox, oy in ((0, 1), (1, 1)):
        pts, uv, tz, kw, objs, z = _upright_at(F_133217, ox, oy)
        cands = segmenter_candidates(objs, ("food container",), (640, 400))
        bad = ss.scene_fix_from_boxes(cands, pts, uv, tz, model, src="segmenter", rivals=True, **kw)
        assert bad.pose is not None, ((ox, oy), bad.tried)
        silhouette = ss.pixel_plane(0, float(z["cap_gt"][2]), pts, uv, kw["cam_xyz"], kw["cam_R"])
        past = float(ss.past_sides(np.asarray(bad.top[:2]), (silhouette,), kw["cam_xyz"], bad.top[2], r)[0])
        assert past > 0.045, (ox, oy)  # measured 50.9 and 51.2 mm


def test_a_taller_thing_beside_a_cut_box_does_not_refuse_it():
    """The lid stands on its holder, inside the box's own pixel columns: what
    stands over a cut box's top is counted there only. A bottle 80 mm taller
    than the lid 18 cm beside the half-off canister at 17:27 stood over the
    top inside the band's disc and refused it (b2fb875; at 14:40, with a
    lid-wide one 27 cm off, c693865 fixed it 11.6 mm from the reference);
    now it is the fix it is without the bottle. Something taller standing
    right behind the box, inside its columns, still refuses it."""
    from box_opening.perception.scene_source import scene_fix_from_boxes
    from box_opening.perception.segmenter_source import segmenter_candidates

    model = ContainerModel.load(PINK)
    pts, uv, tz, kw, objs, z = _upright_at(F_172707)
    ref = np.asarray(z["ref_xyz"], float)
    (c,) = segmenter_candidates(objs, ("food container",), (640, 400))
    alone = scene_fix_from_boxes([c], pts, uv, tz, model, src="segmenter", rivals=True, **kw)
    tall = model.dims[2] + 0.08
    bottle = _cylinder_by_the_lid(F_172707, model, 0.18 - model.lid_dims[0] / 2 - 0.035, 0.035, tall, "beside")
    pts, uv, tz, kw, objs, z = _upright_at(F_172707, cylinders=[bottle])
    got = scene_fix_from_boxes([c], pts, uv, tz, model, src="segmenter", rivals=True, **kw)
    assert got.pose is not None and got.partial == ("left",), got.tried
    assert np.linalg.norm(np.subtract(got.top, alone.top)) < 0.002  # measured 0.2 mm: the bottle's side in the band
    assert np.linalg.norm(np.subtract(got.top, ref)) < 0.015  # measured 11.2 mm
    behind = _cylinder_by_the_lid(F_172707, model, 0.01, 0.035, tall, "behind")
    pts, uv, tz, kw, objs, z = _upright_at(F_172707, cylinders=[behind])
    hid = scene_fix_from_boxes([c], pts, uv, tz, model, src="segmenter", rivals=True, **kw)
    assert hid.pose is None and "under the lid" in hid.tried[-1], hid.tried


def test_a_lids_circle_that_something_taller_stands_inside_is_refused(monkeypatch):
    """With the over-the-top count kept to the box's columns, a taller thing
    beside and behind a cut canister puts the side of it into the lid's band
    unrefused, and the circle slides toward it: a 20 cm cylinder 5 cm clear
    of the half-off canister at 13:34 (b2fb875 refused it on the whole disc's
    count, c693865 fixed it 151 mm off). The lid is the container's top:
    nothing stands 20-50 mm over it where the fitted circle is, and here the
    cylinder does."""
    from box_opening.perception import scene_source as ss
    from box_opening.perception.segmenter_source import segmenter_candidates

    model = ContainerModel.load(PINK)
    cyl = _cylinder_by_the_lid(F_133403, model, 0.05, 0.035, 0.20, "beside_behind")
    pts, uv, tz, kw, objs, z = _upright_at(F_133403, cylinders=[cyl])
    ref = np.asarray(z["ref_xyz"], float)
    cands = segmenter_candidates(objs, ("food container",), (640, 400))
    got = ss.scene_fix_from_boxes(cands, pts, uv, tz, model, src="segmenter", rivals=True, **kw)
    assert got.pose is None and "where the lid's circle is" in got.tried[-1], got.tried
    monkeypatch.setattr(ss, "PARTIAL_OVER_PAD_M", -1.0)  # no circle holds anything: the check is off
    slid = ss.scene_fix_from_boxes(cands, pts, uv, tz, model, src="segmenter", rivals=True, **kw)
    assert slid.pose is not None and np.linalg.norm(np.subtract(slid.top, ref)) > 0.04  # measured 55.8 mm


def test_known_limit_a_taller_thing_right_behind_a_cut_canister_passes_for_it():
    """A pinned limit, not a wish (review of b2fb875): a cylinder 16 cm tall
    1 cm behind the half-off canister at 13:34, with the lid box RF-DETR
    stacks on the holder there at 13:33 and 15:09. The lid box is refused:
    the cylinder stands over its top inside its columns. The grown holder
    band then lifts the cylinder's top, inside the lid band, and fixes the
    cylinder 118 mm off, with no rival left to flag it; c693865 passed the
    lid box too and called the message two places, for the OWL to settle.
    A second cue (free space) would be needed; until then, this fails
    unsafe."""
    from box_opening.perception.scene_source import scene_fix_from_boxes
    from box_opening.perception.segmenter_source import SceneCand, segmenter_candidates

    model = ContainerModel.load(PINK)
    pts, uv, tz, kw, objs, z = _upright_at(F_133403)
    ref = np.asarray(z["ref_xyz"], float)
    lid = SceneCand(0.29, (0.0, 214.0, 90.0, 277.0), None, (0.0, 214.0, 90.0, 277.0), "lid")
    cands = [lid] + segmenter_candidates(objs, ("food container",), (640, 400))
    clean = scene_fix_from_boxes(cands, pts, uv, tz, model, src="segmenter", rivals=True, **kw)
    assert clean.box == lid.box and np.linalg.norm(np.subtract(clean.top, ref)) < 0.015  # measured 13.3 mm
    cyl = _cylinder_by_the_lid(F_133403, model, 0.01, 0.04, 0.16, "behind")
    pts, uv, tz, kw, objs, z = _upright_at(F_133403, cylinders=[cyl])
    alone = scene_fix_from_boxes([lid], pts, uv, tz, model, src="segmenter", **kw)
    assert alone.pose is None and "under the lid" in alone.tried[-1], alone.tried
    got = scene_fix_from_boxes(cands, pts, uv, tz, model, src="segmenter", rivals=True, **kw)
    assert got.pose is not None and got.box != lid.box and got.rivals == ()
    assert np.linalg.norm(np.subtract(got.top, ref)) > 0.10  # measured 118 mm: the cylinder


def test_a_lid_boxs_circle_refined_past_its_side_falls_back_inside_it():
    """Where the depth runs to the frame's edge (a right-edge cut: the OAK's
    no-depth strip is at its left), the smear past the lid's silhouette pulls
    the refined circle past a lid box's side: the 16:24 frame with its left
    edge moved through the lid at 35 %, the lid's GT box cut there. b2fb875
    refused it (refined 20.3 mm past), and on detbench re-cuts like it the
    gate fell through to the grown holder band, up to 48 mm off. The RANSAC
    circle it was refined from stands inside the bound: the fix, 21 mm from
    the reference. A holder's side stands outside the lid's silhouette, and
    past a grown holder box the refined circle is still refused."""
    from box_opening.perception.scene_source import scene_fix_from_boxes
    from box_opening.perception.segmenter_source import SceneCand

    model = ContainerModel.load(PINK)
    pts, uv, tz, kw, _objs, z = _upright_at(F_162417, lo=201)
    ref = np.asarray(z["ref_xyz"], float)
    lid = (0.0, 201.0, 241.0 - 201, 221.0)  # detbench's GT lid box at 16:24, [179, 201, 241, 221], cut at u = 201
    got = scene_fix_from_boxes([(1.0, list(lid))], pts, uv, tz, model, **kw)
    assert got.pose is not None and got.partial == ("left",), got.tried
    assert np.linalg.norm(np.subtract(got.top, ref)) < 0.025  # measured 21.1 mm
    held = scene_fix_from_boxes([SceneCand(1.0, lid, None, lid, "grown")], pts, uv, tz, model, src="segmenter", **kw)
    assert held.pose is None and "refined, runs 20.3 mm past its box" in held.tried[-1], held.tried


def test_the_gates_reasons_name_the_detector_they_came_from():
    from box_opening.perception.scene_source import off_frame_why, scene_fix_from_boxes

    model = ContainerModel.load(PINK)
    pts, uv, canister, _tall = _side_scene(np.random.default_rng(4))
    cap = canister[:, 2] > SIDE_TABLE + model.dims[2] - 0.025
    box = list(_pixel_box(_project(canister[cap])))
    w = int(np.ceil(box[2])) + 1
    cut = scene_fix_from_boxes(
        [(0.88, box)], pts, uv, None, model, cam_xyz=SIDE_CAM, image_wh=(w, 400), src="segmenter"
    )
    assert cut.tried == ("it runs off the frame at its right edge (segmenter 0.88)",)
    flatbot = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert off_frame_why(0.88, ("right",), flatbot, src="segmenter").startswith("the segmenter's box (0.88) runs off")
    assert off_frame_why(0.45, ("right",), flatbot).startswith("the OWL's box (0.45) runs off")


# --- the vote ------------------------------------------------------------------


def _sighting(source, top, partial=()):
    from box_opening.perception.scene_source import Sighting

    return Sighting(
        source, 0.5, tuple(top), None, 300, tuple(partial), 0.21, "side", (0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)
    )


def test_a_scene_fix_needs_two_sightings_that_agree():
    from box_opening.perception.scene_source import commit_fix

    a = (0.660, -0.265, 0.310)
    near = (0.670, -0.265, 0.310)  # 10 mm away
    far = (0.690, -0.265, 0.310)  # 30 mm away
    assert commit_fix([_sighting("segmenter", a)], True) == (None, "too few")
    top, why = commit_fix([_sighting("segmenter", a), _sighting("segmenter", near)], True)
    assert why is None and top == pytest.approx((0.665, -0.265, 0.310))
    assert commit_fix([_sighting("segmenter", a), _sighting("segmenter", far)], True) == (None, "disagree")
    # only the newest two count: an old far sighting does not block
    top, why = commit_fix([_sighting("segmenter", far), _sighting("owl", a), _sighting("segmenter", near)], True)
    assert why is None and top == pytest.approx((0.665, -0.265, 0.310))


def test_the_owls_box_of_a_cut_lid_needs_the_segmenter_once_it_has_seen_the_box():
    """While the segmenter is live and has seen the container in this look,
    it must stand beside an OWL cut box; down or blind, the OWL's cut boxes
    commit on their own."""
    from box_opening.perception.scene_source import commit_fix

    a, b = (0.380, -0.170, 0.305), (0.384, -0.172, 0.305)  # 5 mm apart
    cut_pair = [_sighting("owl", a, ("left",)), _sighting("owl", b, ("left",))]
    assert commit_fix(cut_pair, True) == (None, "cut OWL box unconfirmed")
    assert commit_fix(cut_pair, False)[1] is None
    assert commit_fix([_sighting("owl", a, ("left",)), _sighting("segmenter", b)], True)[1] is None
    assert commit_fix([_sighting("owl", a), _sighting("owl", b)], True)[1] is None  # whole boxes need no one


def test_once_the_segmenter_saw_two_places_only_a_pair_with_the_owl_in_it_commits():
    from box_opening.perception.scene_source import commit_fix

    a, b = (0.661, -0.265, 0.310), (0.664, -0.262, 0.309)  # 4 mm apart
    segs = [_sighting("segmenter", a), _sighting("segmenter", b)]
    assert commit_fix(segs, True, owl_settles=True) == (None, "two places unsettled")
    assert commit_fix(segs, True)[1] is None  # no ambiguity in the look: the segmenter's pair is enough
    assert commit_fix([_sighting("segmenter", a), _sighting("owl", b)], True, owl_settles=True)[1] is None
    assert commit_fix([_sighting("owl", a), _sighting("owl", b)], True, owl_settles=True)[1] is None


def test_a_place_the_segmenter_passed_in_two_confirms_the_owls_cut_lid_there():
    """A segmenter message that passes in two places is no sighting: where
    every one does, the OWL's cut pair waited on a sighting that never came.
    One of those places within the vote's tolerance of each of the pair's
    sightings, as a third sighting would be, confirms the pair; a pair at
    neither place still waits, and the places never stand in for the OWL
    that settles which is the container."""
    from box_opening.perception.scene_source import commit_fix

    a, b = (0.380, -0.170, 0.305), (0.384, -0.172, 0.305)  # 5 mm apart
    cut_pair = [_sighting("owl", a, ("left",)), _sighting("owl", b, ("left",))]
    canister, bowl = (0.390, -0.171, 0.305), (0.380, 0.130, 0.305)  # 10 and 6 mm from the pair; 30 cm away
    for places in ([canister, bowl], [bowl, canister]):
        assert commit_fix(cut_pair, True, owl_settles=True, segmenter_places=places)[1] is None
    assert commit_fix(cut_pair, True, owl_settles=True) == (None, "cut OWL box unconfirmed")
    off = (0.412, -0.171, 0.305)  # 32 and 28 mm from the pair
    assert commit_fix(cut_pair, True, owl_settles=True, segmenter_places=[off, bowl]) == (
        None,
        "cut OWL box unconfirmed",
    )
    one = (0.401, -0.171, 0.305)  # 21 mm from a, 17 from b: not where the pair is
    assert commit_fix(cut_pair, True, owl_settles=True, segmenter_places=[one]) == (None, "cut OWL box unconfirmed")
    segs = [_sighting("segmenter", a), _sighting("segmenter", b)]
    assert commit_fix(segs, True, owl_settles=True, segmenter_places=[a, bowl]) == (None, "two places unsettled")


# --- locate(), driven by stubs on a test clock --------------------------------

CAM_T0 = 1000.0  # the camera's clock at the test's t = 0
SCENE_FRAME = "scene_camera_color_optical_frame"
UPRIGHT = np.array([[0.0, 0.0, 1.0, 0.2], [-1.0, 0.0, 0.0, -0.5], [0.0, -1.0, 0.0, 0.4], [0.0, 0.0, 0.0, 1.0]])
HOLDER = (157.75, 216.38, 255.62, 260.5)
SEG_TOP = (0.661, -0.265, 0.310)
OWL_BOX = (170.0, 195.0, 243.0, 222.0)
OWL_TOP = (0.664, -0.262, 0.309)


def _stamp(t):
    """The scene camera's newest frame at test time t (15 fps), as a ROS stamp."""
    ns = int(round((CAM_T0 + np.floor(t * 15 + 1e-9) / 15) * 1e9))
    return SimpleNamespace(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


class _Owl:
    """The scene OWL node and its rung: alive (a heartbeat) or not; while
    enabled, one scripted message per inference, the first 0.7 s after the
    enable edge, then every 1.3 s."""

    def __init__(self, world, script=(), alive=True):
        self.world = world
        self.script = [list(m) for m in script]  # each: [(score, box), ...]
        self.latest = [0.0, 0.0, 0.0, 0.0, -1.0, 0.0] if alive else None
        self.box = None
        self.enabled = False
        self.ons, self.offs = [], []
        self._next = None

    def enable(self):
        if not self.enabled:
            self._next = self.world.t + 0.7
        self.enabled = True
        self.ons.append(round(self.world.t, 3))

    def disable(self):
        self.enabled = False
        self.offs.append(round(self.world.t, 3))

    def fresh_box(self):
        return self.box

    def tick(self, t):
        if self.enabled and self.latest is not None and self.script and t >= self._next:
            m = []
            for score, b in self.script.pop(0):
                m += [*b, score, t]
            self.box = self.latest = m
            self._next += 1.3


class _World:
    """The test's clock, and what the scene camera, the segmenter and the OWL
    publish on it."""

    def __init__(
        self, seg_msgs=(), owl_script=(), owl_alive=True, seg_since=0.0, frame_id=SCENE_FRAME, publishers=None
    ):
        from box_opening.perception.segmenter_source import SegmenterRung

        self.t = 0.0
        self.seg_msgs = sorted(seg_msgs, key=lambda m: m[0])  # (arrival t, objects[, frame_id[, wh[, stamp t]]])
        self.owl = _Owl(self, owl_script, owl_alive)

        class Detected:
            size = None

        class Node:
            def create_subscription(self, *a):
                return None

            def count_publishers(self, topic):  # what the graph says; None: a node that cannot say
                if publishers is None:
                    raise AttributeError("count_publishers")
                return publishers

        self.rung = SegmenterRung(Node(), ("food container",), import_types=lambda: (object, Detected))
        self.rung.t_subscribed = -float(seg_since)  # subscribed this long before t = 0
        self.frame_id = frame_id
        self.grab = SimpleNamespace(
            color=np.zeros((400, 640, 3), np.uint8),
            missing=lambda: [],
            depth_median=lambda: None,
            cloud_in_color_frame=lambda depth=None, stride=1: (
                np.c_[np.zeros((50, 2)), np.ones(50)],
                np.zeros((50, 2)),
            ),
        )
        self.deliver()

    def clock(self):
        return self.t

    def sleep(self, s):
        self.t = round(self.t + s, 6)
        self.deliver()

    def deliver(self):
        from box_opening.perception.segmenter_source import SegFrame, stamp_ns

        self.grab.color_stamp, self.grab.color_frame_id = _stamp(self.t), self.frame_id
        while self.seg_msgs and self.seg_msgs[0][0] <= self.t:
            t, objects, *rest = self.seg_msgs.pop(0)
            frame_id = rest[0] if rest else SCENE_FRAME
            wh = rest[1] if len(rest) > 1 else (640, 400)
            shot = rest[2] if len(rest) > 2 else t - 0.075  # its frame was taken this long before it arrives
            self.rung.latest = SegFrame(t, stamp_ns(_stamp(shot)), frame_id, wh, tuple(objects), len(objects))
        self.owl.tick(self.t)


def _every(t0, t1, objects, *rest, period=0.2):
    """The segmenter's messages from t0 to t1, at 5 Hz (rest: frame_id, wh)."""
    return [(t, objects, *rest) for t in np.arange(t0, t1, period)]


def _band(holder):
    from box_opening.perception.segmenter_source import lid_band

    return lid_band(holder, (640, 400))[0]


def _locator(monkeypatch, world, outcomes, seg=True):
    """A SceneLocator on the world's stubs, its gate a stand-in: outcomes[box] is a top (the box passes), a dict
    {top, partial} (it passes, finished from part of it) or a reason (it is refused)."""
    from box_opening.models.container import ContainerPose
    from box_opening.perception import scene_source as ss

    loc = ss.SceneLocator.__new__(ss.SceneLocator)
    loc.spins, loc.model, loc.table_z, loc.base_frame, loc.tag_px = (
        False,
        ContainerModel.load(PINK),
        None,
        "arm_mount",
        None,
    )
    loc.last_why, loc.last_timing, loc.last_boxes, loc.last_seg_state, loc.moved_note = None, {}, {}, None, None
    loc.seg_box = "holder"
    loc.grab, loc.owl, loc.seg = world.grab, world.owl, world.rung if seg else None
    loc._clock, loc._sleep = world.clock, world.sleep
    loc.ready = lambda: True
    loc.base_to_color = lambda: UPRIGHT
    gated = []
    key = {ss.box_key(k): v for k, v in outcomes.items()}

    def gate(cands, pts, uv, table_z, model, src="OWL", rivals=False, **kw):
        gated.append((round(world.t, 3), src, len(cands)))
        tried, passing = [], []
        for c in cands:
            o = key[ss.box_key(c[1])]
            if isinstance(o, str):
                tried.append("%s (%s %.2f)" % (o, src, c[0]))
            else:
                passing.append((c, o if isinstance(o, dict) else {"top": o}))
        if not passing:
            return ss.SceneBoxFix(None, None, 0, None, tuple(tried))
        c, o = passing[0]
        top = tuple(o["top"])
        riv = tuple(
            tuple(p["top"])
            for _c, p in passing[1:]
            if np.hypot(p["top"][0] - top[0], p["top"][1] - top[1]) > ss.RIVAL_M
        )
        pose = ContainerPose(xyz=(top[0], top[1], top[2] - 0.121), yaw=0.0)
        return ss.SceneBoxFix(
            pose, top, 300, c[1], tuple(tried), 0.19, "side", (), tuple(o.get("partial", ())), riv if rivals else ()
        )

    monkeypatch.setattr(ss, "scene_fix_from_boxes", gate)
    return loc, gated


def _holder(dx=0.0):
    return (HOLDER[0] + dx, HOLDER[1], HOLDER[2] + dx, HOLDER[3])


def test_the_segmenters_frames_fix_the_box_with_the_owl_never_enabled(monkeypatch):
    # the frame at 0.1 s was taken before the look; the next two, 4 mm apart, commit their median
    before = [(0.1, [("food container", 0.88, _holder())])]
    after = [
        (t, [("food container", 0.88, _holder(1.0 if i % 2 else 0.0))]) for i, t in enumerate(np.arange(0.3, 3.0, 0.2))
    ]
    world = _World(seg_msgs=before + after)
    other = (SEG_TOP[0] + 0.004, SEG_TOP[1], SEG_TOP[2])
    loc, gated = _locator(monkeypatch, world, {_band(_holder()): SEG_TOP, _band(_holder(1.0)): other})
    fix = loc.locate(15.0)
    assert fix is not None, loc.last_why
    assert fix.source == "segmenter" and fix.n_sightings == 2 and fix.sources == ("segmenter", "segmenter")
    assert fix.top_xyz == pytest.approx((SEG_TOP[0] + 0.002, SEG_TOP[1], SEG_TOP[2]))
    assert fix.score == pytest.approx(0.88)
    assert world.owl.ons == [] and world.owl.offs  # never on the GPU; the latched gate shut all the same
    assert [g[1] for g in gated] == ["segmenter", "segmenter"]  # the frame from before the look never voted
    assert loc.last_timing["fix_s"] == pytest.approx(0.5, abs=0.05)
    assert loc.last_boxes["segmenter"] == [["food container", 0.88, *_holder(1.0)]]


def test_the_look_end_to_end_through_the_real_gate_on_the_frame_the_owl_missed():
    """locate() on the 10-27 16:24 frame with nothing stubbed between the
    segmenter's message and the SceneFix: its holder box in each 5 Hz
    message, the real band, gate, vote, pose and table extent. Two fresh
    frames commit; the OWL is never asked."""
    from pathlib import Path

    from box_opening.perception import scene_source as ss
    from box_opening.perception.scene import color_cloud

    z = np.load(Path(__file__).parent / "data" / F_162417)
    depth = z["depth_mm"].astype(float) / 1000.0
    _pts, _uv, _tz, _kw, objs, _gt, ref = _upright(F_162417)
    world = _World(seg_msgs=_every(0.1, 3.0, objs))
    world.grab.depth_median = lambda: depth
    world.grab.cloud_in_color_frame = lambda depth=None, stride=1: color_cloud(
        depth, z["kd"], z["k"], z["dist"], z["T_color_depth"], stride=stride
    )
    loc = ss.SceneLocator.__new__(ss.SceneLocator)
    loc.spins, loc.model, loc.table_z, loc.base_frame, loc.tag_px = (
        False,
        ContainerModel.load(PINK),
        None,
        "arm_mount",
        None,
    )
    loc.last_why, loc.last_timing, loc.last_boxes, loc.last_seg_state, loc.moved_note = None, {}, {}, None, None
    loc.seg_box, loc.grab, loc.owl, loc.seg = "holder", world.grab, world.owl, world.rung
    loc._clock, loc._sleep = world.clock, world.sleep
    loc.ready = lambda: True
    loc.base_to_color = lambda: z["T_base_color"]
    fix = loc.locate(5.0)
    assert fix is not None, loc.last_why
    assert fix.source == "segmenter" and fix.sources == ("segmenter", "segmenter") and world.owl.ons == []
    assert np.linalg.norm(np.subtract(fix.top_xyz, ref)) < 0.015  # the gate alone: 8.2 mm
    assert fix.score == pytest.approx(0.881, abs=0.001) and fix.n_points == 299
    # the pose stands on the table the side lift measured around it, under the pair's median top
    assert fix.pose.xyz[:2] == pytest.approx(fix.top_xyz[:2]) and fix.pose.xyz[2] == pytest.approx(fix.table_z_scene)
    assert abs(fix.top_xyz[2] - fix.pose.xyz[2] - loc.model.dims[2]) < ss.LID_BAND_M
    assert fix.table_extent is not None and len(fix.table_points) > 0  # no survey: the table it stands on, measured
    assert loc.last_timing["fix_s"] == pytest.approx(0.5, abs=0.05)  # the 0.1 s frame predates the look


def test_a_silent_segmenter_hands_the_look_to_the_owl_at_once(monkeypatch):
    world = _World(seg_since=5.0, owl_script=[[(0.3, OWL_BOX)], [(0.31, OWL_BOX)]])
    loc, _gated = _locator(monkeypatch, world, {OWL_BOX: OWL_TOP})
    fix = loc.locate(15.0)
    assert fix is not None, loc.last_why
    assert world.owl.ons[0] <= 0.03  # the first look at the segmenter found it silent
    assert fix.source == "owl" and fix.sources == ("owl", "owl") and fix.top_xyz == pytest.approx(OWL_TOP)
    assert loc.last_seg_state == "silent"


def test_a_live_segmenter_whose_boxes_are_all_refused_calls_the_owl_after_a_second(monkeypatch):
    from box_opening.perception.scene_source import OWL_AFTER_S

    world = _World(
        seg_msgs=_every(0.1, 10.0, [("food container", 0.88, _holder())]),
        owl_script=[[(0.3, OWL_BOX)], [(0.3, OWL_BOX)]],
    )
    loc, _gated = _locator(
        monkeypatch, world, {_band(_holder()): "from the side, its top is -55 mm from lid height", OWL_BOX: OWL_TOP}
    )
    fix = loc.locate(15.0)
    assert fix is not None and fix.source == "owl" and fix.sources == ("owl", "owl")
    assert OWL_AFTER_S == 1.0 and world.owl.ons[0] == pytest.approx(OWL_AFTER_S, abs=0.05)


def test_a_segmenter_frame_from_before_the_look_never_votes(monkeypatch):
    # its newest message is fresh, but of a frame no newer than the one the grabber held at the call
    world = _World(seg_msgs=[(0.0, [("food container", 0.88, _holder())], SCENE_FRAME, (640, 400), -0.05)])
    loc, gated = _locator(monkeypatch, world, {_band(_holder()): SEG_TOP})
    assert loc.locate(0.5) is None
    assert gated == []
    assert loc.last_why.startswith("scene segmenter: no frame newer than the look in 0.5 s")
    assert "scene OWL: not asked" in loc.last_why


def test_a_segmenter_watching_another_camera_is_down_and_says_which(monkeypatch):
    wrist = "wrist_camera_color_optical_frame"
    world = _World(seg_msgs=_every(0.0, 5.0, [("food container", 0.88, _holder())], wrist))
    loc, gated = _locator(monkeypatch, world, {_band(_holder()): SEG_TOP})
    assert loc.locate(3.0) is None
    assert world.owl.ons and world.owl.ons[0] <= 0.03  # its first message, there at the call, says so
    assert gated == []  # nothing it says is used
    assert ("it watches '%s', not the scene camera ('%s')" % (wrist, SCENE_FRAME)) in loc.last_why
    assert loc.last_why.endswith("scene OWL: the scene OWL saw no box in 3.0 s")


def test_two_places_in_one_segmenter_frame_are_no_sighting_and_call_the_owl(monkeypatch):
    other = (100.0, 216.0, 190.0, 260.0)  # a second bowl whose band passes 30 cm away
    world = _World(
        seg_msgs=_every(0.1, 10.0, [("food container", 0.88, _holder()), ("food container", 0.6, other)]),
        owl_script=[[(0.3, OWL_BOX)], [(0.3, OWL_BOX)]],
    )
    far = (SEG_TOP[0], SEG_TOP[1] + 0.3, SEG_TOP[2])
    loc, _gated = _locator(monkeypatch, world, {_band(_holder()): SEG_TOP, _band(other): far, OWL_BOX: OWL_TOP})
    fix = loc.locate(15.0)
    assert world.owl.ons[0] == pytest.approx(0.3, abs=0.05)  # at the first ambiguous frame, not a second later
    assert fix is not None and fix.sources == ("owl", "owl")


def test_after_two_places_the_segmenter_alone_never_picks_one(monkeypatch):
    """One frame passes the canister and a second bowl 30 cm away. In the
    frames after it the canister's band is refused and the bowl's passes:
    two agreeing segmenter sightings of the bowl are no evidence it is the
    container, and the OWL, boxing the canister's lid, never agrees."""
    other = (100.0, 216.0, 190.0, 260.0)
    far = (SEG_TOP[0], SEG_TOP[1] + 0.3, SEG_TOP[2])
    both = [(0.3, [("food container", 0.88, _holder()), ("food container", 0.6, other)])]
    outcomes = {_band(_holder()): SEG_TOP, _band(other): far, OWL_BOX: OWL_TOP}
    world = _World(
        seg_msgs=both + _every(0.5, 10.0, [("food container", 0.6, other)]),
        owl_script=[[(0.3, OWL_BOX)], [(0.3, OWL_BOX)]],
    )
    loc, _gated = _locator(monkeypatch, world, outcomes)
    assert loc.locate(4.0) is None  # before: the bowl, committed at 0.7 s
    assert world.owl.ons[0] == pytest.approx(0.3, abs=0.05)
    assert loc.last_why.startswith(
        "scene segmenter: two places pass as this container, [0.661, -0.265] and "
        "[0.661, 0.035], and the OWL did not settle which"
    )
    # the frames after it pass the canister alone: the OWL's first box settles it
    world = _World(
        seg_msgs=both + _every(0.5, 10.0, [("food container", 0.88, _holder())]),
        owl_script=[[(0.3, OWL_BOX)], [(0.3, OWL_BOX)]],
    )
    loc, _gated = _locator(monkeypatch, world, outcomes)
    fix = loc.locate(15.0)
    assert fix is not None and fix.sources == ("segmenter", "owl")
    assert fix.top_xyz == pytest.approx(tuple(0.5 * (s + o) for s, o in zip(SEG_TOP, OWL_TOP, strict=False)))
    assert loc.last_timing["fix_s"] == pytest.approx(1.0, abs=0.06)  # the OWL's box on its enable edge


def test_a_segmenter_nothing_publishes_hands_the_first_look_to_the_owl_at_prime(monkeypatch):
    """The box-opening profile without `scene_segmenter: rfdetr`: prime(),
    at the locator's construction, finds nothing on the graph and wakes the
    OWL there, as before the segmenter, so its first box is on the topic by
    the time the look starts. Waiting out the rung's first second instead
    (a node that cannot ask the graph) starts the OWL only in the look."""
    timings = {}
    for publishers in (0, None):
        world = _World(publishers=publishers, owl_script=[[(0.3, OWL_BOX)], [(0.3, OWL_BOX)]])
        loc, _gated = _locator(monkeypatch, world, {OWL_BOX: OWL_TOP})
        loc.prime()  # at construction: just subscribed
        world.sleep(0.8)  # the readiness checks, the arm to its rest pose
        fix = loc.locate(15.0)
        assert fix is not None and fix.sources == ("owl", "owl"), loc.last_why
        timings[publishers] = loc.last_timing["fix_s"]
        assert loc.last_seg_state == ("absent" if publishers == 0 else "silent")
    assert timings[0] == pytest.approx(1.2, abs=0.06)  # the second box: 1.3 s after the first, at 0.7 s
    assert timings[None] == pytest.approx(2.2, abs=0.06)  # the OWL woken 0.2 s into the look, then two boxes


def test_without_a_segmenter_for_the_container_the_owl_looks_as_it_always_did(monkeypatch):
    world = _World(owl_alive=False)
    loc, _gated = _locator(monkeypatch, world, {}, seg=False)
    assert loc.locate(2.0) is None
    assert world.owl.ons == [0.0]
    assert loc.last_why == (
        "the scene OWL is SILENT (no heartbeat in 2.0 s) — not started, still loading, or not "
        "receiving frames: check the launch terminal"
    )
    world = _World()
    loc, _gated = _locator(monkeypatch, world, {}, seg=False)
    assert loc.locate(2.0) is None and loc.last_why == "the scene OWL saw no box in 2.0 s"
    world = _World(owl_script=[[(0.3, OWL_BOX)]])
    loc, _gated = _locator(monkeypatch, world, {OWL_BOX: "only 26 depth points inside the box (need 40)"}, seg=False)
    assert loc.locate(2.0) is None
    assert (
        loc.last_why == "none of the OWL's 1 candidate(s) is this container: "
        "only 26 depth points inside the box (need 40) (OWL 0.30)"
    )
    world = _World(owl_script=[[(0.3, OWL_BOX)], [(0.3, OWL_BOX)]])
    loc, _gated = _locator(monkeypatch, world, {OWL_BOX: OWL_TOP}, seg=False)
    fix = loc.locate(5.0)
    assert fix is not None and fix.source == "owl" and fix.n_sightings == 2


def test_a_segmenter_this_install_cannot_read_leaves_the_owl_to_look_from_the_start(monkeypatch):
    world = _World(owl_script=[[(0.3, OWL_BOX)], [(0.3, OWL_BOX)]])
    world.rung.available = False
    world.rung.why = (
        "rammp_perception_interfaces does not import (no module): this install does not chain ~/ros2_ws/install"
    )
    loc, _gated = _locator(monkeypatch, world, {OWL_BOX: OWL_TOP})
    fix = loc.locate(15.0)
    assert world.owl.ons[0] == 0.0 and fix is not None and fix.sources == ("owl", "owl")
    world = _World(owl_alive=False)
    world.rung.available, world.rung.why = False, "its DetectedObject predates rammp-interfaces-ros2#16 (no size)"
    loc, _gated = _locator(monkeypatch, world, {})
    assert loc.locate(2.0) is None
    assert loc.last_why.startswith("scene segmenter: its DetectedObject predates rammp-interfaces-ros2#16")
    assert "scene OWL: the scene OWL is SILENT" in loc.last_why


def test_with_both_detectors_silent_the_look_says_so_for_each(monkeypatch):
    world = _World(seg_since=5.0, owl_alive=False)
    loc, _gated = _locator(monkeypatch, world, {})
    assert loc.locate(2.0) is None
    assert loc.last_why.startswith(
        "scene segmenter: SILENT: no /perception/object_detections in 1.0 s; select "
        "scene_segmenter: rfdetr in the sheppy profile; scene OWL: the scene OWL is SILENT"
    )


def test_when_neither_detector_finds_the_box_both_say_why(monkeypatch):
    world = _World(seg_msgs=_every(0.1, 10.0, [("food container", 0.88, _holder())]))
    loc, _gated = _locator(monkeypatch, world, {_band(_holder()): "from the side, its top is -55 mm from lid height"})
    assert loc.locate(3.0) is None
    why = loc.last_why
    assert why.startswith("scene segmenter: none of its ") and "-55 mm from lid height (segmenter 0.88)" in why
    assert why.endswith("; scene OWL: the scene OWL saw no box in 3.0 s")


def test_the_owls_cut_lid_waits_on_the_segmenter_only_once_it_has_seen_the_box(monkeypatch):
    """A live segmenter that sees nothing used to hold every cut OWL box back:
    with it blind, all 15 cut detbench frames got no fix. It confirms one only
    once it has seen the container in this look."""
    cut = {"top": OWL_TOP, "partial": ("left",)}
    refused = "only 26 depth points inside the box (need 40)"
    # blind: no food container at all, or one whose band is refused every frame; the OWL's two boxes commit
    for seg_objects, outcomes in (
        ([], {OWL_BOX: cut}),
        ([("food container", 0.88, _holder())], {_band(_holder()): refused, OWL_BOX: cut}),
    ):
        world = _World(seg_msgs=_every(0.1, 10.0, seg_objects), owl_script=[[(0.3, OWL_BOX)], [(0.3, OWL_BOX)]])
        loc, _gated = _locator(monkeypatch, world, outcomes)
        fix = loc.locate(4.0)
        assert fix is not None, loc.last_why
        assert fix.sources == ("owl", "owl") and fix.top_xyz == pytest.approx(OWL_TOP) and fix.partial
    # it saw the container once, 30 mm from where the OWL's boxes put it: the OWL's pair waits on it
    far = (OWL_TOP[0] + 0.030, OWL_TOP[1], OWL_TOP[2])
    once = [(0.3, [("food container", 0.88, _holder())])] + _every(0.5, 10.0, [("food container", 0.88, _holder(5.0))])
    world = _World(seg_msgs=once, owl_script=[[(0.3, OWL_BOX)], [(0.3, OWL_BOX)]])
    loc, _gated = _locator(monkeypatch, world, {_band(_holder()): far, _band(_holder(5.0)): refused, OWL_BOX: cut})
    assert loc.locate(4.0) is None
    assert "the OWL's box of the cut lid is not confirmed by the segmenter's" in loc.last_why
    assert loc.last_why.endswith("scene OWL: its boxes of the cut lid wait on the segmenter's")


def test_a_segmenter_whose_every_frame_passes_two_places_confirms_the_owls_cut_lid_at_one(monkeypatch):
    """The canister half off the frame (detbench P4, 15 of 37 frames) with a
    second bowl whose band passes too, 30 cm away: every segmenter frame
    passes in two places, so none is a sighting, and the OWL's cut boxes used
    to wait on one for all 15 s (NO BOX, nothing moved). The canister's place
    in those frames confirms them: owl+owl at the OWL's second box, as with
    the canister in view. Where neither place is the OWL's, they still wait."""
    other = (100.0, 216.0, 190.0, 260.0)
    far = (SEG_TOP[0], SEG_TOP[1] + 0.3, SEG_TOP[2])
    both = _every(0.1, 20.0, [("food container", 0.88, _holder()), ("food container", 0.6, other)])
    cut = {"top": OWL_TOP, "partial": ("left",)}
    world = _World(seg_msgs=both, owl_script=[[(0.3, OWL_BOX)]] * 12)
    loc, _gated = _locator(
        monkeypatch, world, {_band(_holder()): {"top": SEG_TOP, "partial": ("left",)}, _band(other): far, OWL_BOX: cut}
    )
    fix = loc.locate(15.0)
    assert fix is not None, loc.last_why
    assert fix.sources == ("owl", "owl") and fix.partial and fix.top_xyz == pytest.approx(OWL_TOP)
    assert np.hypot(fix.top_xyz[0] - SEG_TOP[0], fix.top_xyz[1] - SEG_TOP[1]) < 0.005  # on the canister, not the bowl
    assert loc.last_timing["fix_s"] == pytest.approx(2.3, abs=0.06)
    # the segmenter's canister 30 mm from where the OWL's boxes put it: they agree with neither place
    off = {"top": (OWL_TOP[0] + 0.030, OWL_TOP[1], OWL_TOP[2]), "partial": ("left",)}
    world = _World(seg_msgs=both, owl_script=[[(0.3, OWL_BOX)]] * 12)
    loc, _gated = _locator(monkeypatch, world, {_band(_holder()): off, _band(other): far, OWL_BOX: cut})
    assert loc.locate(15.0) is None
    assert loc.last_why.startswith("scene segmenter: two places pass as this container")
    assert loc.last_why.endswith("scene OWL: its boxes of the cut lid wait on the segmenter's")


def test_the_owl_is_shut_however_the_look_ends(monkeypatch):
    world = _World(seg_since=5.0)
    loc, _gated = _locator(monkeypatch, world, {})

    def boom(*a, **k):
        raise KeyboardInterrupt

    world.owl.script = [[(0.3, OWL_BOX)]]
    from box_opening.perception import scene_source as ss

    monkeypatch.setattr(ss, "scene_fix_from_boxes", boom)
    with pytest.raises(KeyboardInterrupt):
        loc.locate(5.0)
    assert world.owl.ons and world.owl.offs and world.owl.offs[-1] >= world.owl.ons[-1]


def test_prime_wakes_the_owl_only_where_the_segmenter_cannot_answer(monkeypatch):
    world = _World(seg_msgs=[(0.0, [])])
    loc, _gated = _locator(monkeypatch, world, {})
    loc.prime()
    assert world.owl.ons == [] and loc.last_seg_state == "live"
    world = _World(seg_since=0.5)  # just subscribed: no word yet is no failure yet
    loc, _gated = _locator(monkeypatch, world, {})
    loc.prime()
    assert world.owl.ons == [] and loc.last_seg_state == "waiting"
    world = _World(seg_since=5.0)
    loc, _gated = _locator(monkeypatch, world, {})
    loc.prime()
    assert world.owl.ons == [0.0]
    world = _World()
    loc, _gated = _locator(monkeypatch, world, {}, seg=False)
    loc.prime()
    assert world.owl.ons == [0.0]
