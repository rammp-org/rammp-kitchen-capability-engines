"""place_here's spot search (place_container.spots) on made-up scenes: pure numpy, no ROS."""

import numpy as np
import pytest

from place_container import spots

FOOT = spots.DEFAULTS["footprint_r"]
TABLE = spots.MOCK_TABLE


@pytest.fixture(scope="module")
def scene():
    return spots.synthetic_scene()


@pytest.fixture(scope="module")
def found(scene):
    return spots.find_spots(scene)


def box_distance(x, y, lo, hi):
    """xy distance from a point to an axis-aligned rectangle."""
    dx = max(lo[0] - x, 0.0, x - hi[0])
    dy = max(lo[1] - y, 0.0, y - hi[1])
    return float(np.hypot(dx, dy))


def test_the_table_comes_from_the_segmenter_mask_and_is_level(found):
    assert found.problem == "" and found.source == "segmenter"
    assert found.tilt_deg < 0.5
    assert spots.plane_z(found.plane, 0.9, 0.0) == pytest.approx(TABLE["z"], abs=0.002)
    assert len(found.spots) >= 10


def test_every_spot_is_on_a_grid_in_reach_with_its_ids_stable(scene, found):
    step = spots.DEFAULTS["step"]
    lo, hi = spots.DEFAULTS["reach"]
    for s in found.spots:
        assert lo <= np.hypot(s["x"], s["y"]) <= hi
        assert s["x"] / step == pytest.approx(round(s["x"] / step), abs=1e-3)
        assert s["y"] / step == pytest.approx(round(s["y"] / step), abs=1e-3)
        assert s["z"] == pytest.approx(TABLE["z"], abs=0.003)
    again = spots.find_spots(scene)
    assert [s["id"] for s in again.spots] == [s["id"] for s in found.spots]
    assert len({s["id"] for s in found.spots}) == len(found.spots)


def test_the_footprint_and_margin_stay_on_the_table(found):
    pad = FOOT + spots.DEFAULTS["edge_margin"] - spots.DEFAULTS["cell"]  # the grid's resolution
    for s in found.spots:
        assert TABLE["x"][0] + pad <= s["x"] <= TABLE["x"][1] - pad
        assert TABLE["y"][0] + pad <= s["y"] <= TABLE["y"][1] - pad


def test_nothing_on_the_table_is_within_the_clearance(found):
    keep = FOOT + spots.DEFAULTS["clearance"] - 1.5 * spots.DEFAULTS["cell"]
    for s in found.spots:
        for _, lo, hi, _ in spots.MOCK_BOXES:
            assert box_distance(s["x"], s["y"], lo, hi) >= keep, (s, lo, hi)


def test_without_masks_a_plane_fit_to_depth_finds_the_same_table(scene, found):
    plain = spots.find_spots(spots.synthetic_scene(labels=False))
    assert plain.problem == "" and plain.source == "depth plane"
    assert spots.plane_z(plain.plane, 0.9, 0.0) == pytest.approx(TABLE["z"], abs=0.002)
    # about the same free table (the masks only add flat things the depth can't see, below; spots right at the
    # clearance can go either way)
    a, b = {s["id"] for s in plain.spots}, {s["id"] for s in found.spots}
    assert len(a & b) >= 0.85 * max(len(a), len(b))


def test_a_flat_object_blocks_only_through_its_mask():
    flat = (("plate", (0.75, -0.25), (0.85, -0.15), 0.008),)
    with_masks = spots.find_spots(spots.synthetic_scene(boxes=flat))
    depth_only = spots.find_spots(spots.synthetic_scene(boxes=flat, labels=False))
    near = {s["id"] for s in depth_only.spots if box_distance(s["x"], s["y"], *flat[0][1:3]) < FOOT}
    assert near  # the depth alone doesn't see an 8 mm plate ...
    assert not near & {s["id"] for s in with_masks.spots}  # ... its mask does


def test_a_tilted_looking_table_means_the_camera_mount_is_off(scene):
    # the frame rendered from the true mount, read with a mount 10 deg off in pitch: as on flatbot 2026-11-12
    pitch = np.radians(10.0)
    rot = np.eye(4)
    rot[1:3, 1:3] = [[np.cos(pitch), -np.sin(pitch)], [np.sin(pitch), np.cos(pitch)]]
    wrong = spots.Scene(scene.depth_m, scene.K, scene.T @ rot, scene.labels, scene.objects)
    result = spots.find_spots(wrong)
    assert result.spots == [] and "tilted" in result.problem and "calibration" in result.problem
    assert result.tilt_deg == pytest.approx(10.0, abs=1.5)


def test_reach_and_exclusions(scene, found):
    near = spots.find_spots(scene, reach=(0.70, 0.80))
    assert 0 < len(near.spots) < len(found.spots)
    nothing = spots.find_spots(scene, reach=(2.0, 2.5))
    assert nothing.spots == [] and "reach" in nothing.problem
    # the cup, left out (as the hand holding the container is): spots next to it come back
    cup_lo, cup_hi = spots.MOCK_BOXES[0][1:3]
    freed = spots.find_spots(scene, exclude=[((cup_lo[0] + cup_hi[0]) / 2, (cup_lo[1] + cup_hi[1]) / 2, 0.1)])
    assert len(freed.spots) > len(found.spots)


def test_no_table_no_spots():
    empty = spots.synthetic_scene(table=dict(z=0.2, x=(5.0, 6.0), y=(5.0, 6.0)), boxes=(), labels=False)
    result = spots.find_spots(empty)
    assert result.spots == [] and result.problem
    blank = spots.Scene(np.zeros((40, 60)), np.eye(3), np.eye(4))
    assert "empty" in spots.find_spots(blank).problem


def test_projection_lands_where_the_depth_says(scene, found):
    """A spot projected into the image, and the depth at that pixel taken back into arm_mount: the same place."""
    uv, z = spots.project([[s["x"], s["y"], s["z"]] for s in found.spots], scene.K, scene.T)
    assert np.all(z > 0)
    for s, (u, v) in zip(found.spots, uv, strict=True):
        u, v = int(round(u)), int(round(v))
        depth = scene.depth_m[v, u]
        ray = np.linalg.inv(scene.K) @ [u, v, 1.0] * depth
        p = scene.T[:3, :3] @ ray + scene.T[:3, 3]
        assert np.hypot(p[0] - s["x"], p[1] - s["y"]) < 0.01 and abs(p[2] - s["z"]) < 0.005


def test_the_ui_objects_are_small_scene_boxes_around_each_spot(scene, found):
    objs = spots.ui_objects(found.spots, scene)
    assert len(objs) == len(found.spots)  # all of them are in view
    width, height = scene.size
    for o in objs:
        x0, y0, x1, y1 = o["box"]
        cu, cv = o["centroid"]
        assert o["source"] == "scene" and o["instance_id"].startswith("spot_")
        assert 0 <= x0 < cu < x1 < width and 0 <= y0 < cv < y1 < height
        assert x1 - x0 < width / 4 and y1 - y0 < height / 4
    behind = [dict(id="spot_behind", x=-1.0, y=-0.6, z=0.2)]  # behind the camera
    assert spots.ui_objects(behind, scene) == []


def test_the_table_height_at_a_spot_is_a_local_fit(found):
    s = found.spots[0]
    assert spots.local_table_z(found.inliers, s["x"], s["y"]) == pytest.approx(TABLE["z"], abs=0.002)
    with pytest.raises(ValueError, match="table points"):
        spots.local_table_z(found.inliers, 5.0, 5.0)
    rng = np.random.default_rng(1)
    rough = np.c_[rng.uniform(0.8, 1.0, (500, 2)), 0.2 + rng.normal(0, 0.03, 500)]
    with pytest.raises(ValueError, match="not flat"):
        spots.local_table_z(rough, 0.9, 0.9)


def test_fit_plane_ignores_walls_and_takes_the_largest_level_surface():
    rng = np.random.default_rng(2)
    table = np.c_[rng.uniform(0.5, 1.2, (3000, 2)), np.full(3000, 0.2)]
    wall = np.c_[np.full(2000, 1.3), rng.uniform(-0.5, 0.5, 2000), rng.uniform(0.2, 1.0, 2000)]
    plane = spots.fit_plane(np.vstack([table, wall]))
    assert spots.tilt_deg(plane) < 0.5 and plane[2] == pytest.approx(0.2, abs=3e-3)  # the wall's foot is in tol
    assert spots.fit_plane(wall, max_tilt_deg=20) is None
