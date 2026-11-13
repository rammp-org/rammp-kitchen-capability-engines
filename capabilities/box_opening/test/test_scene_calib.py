"""The scene camera's pose from the door and table planes plus the tag."""

import numpy as np
import pytest

from box_opening.perception.d405 import quat_to_mat
from box_opening.perception.planes import cloud_from_depth, fit_plane, ray_plane
from box_opening.perception.scene_calib import (
    PLANE_ANGLE_TOL_DEG,
    load_scene_yaml,
    optical_to_link,
    scene_pose_from_planes,
    transform,
    write_scene_yaml,
)


def _rot(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def _scene_truth():
    """A camera standing left of the arm, looking across the table and
    slightly down — roughly where the bench's scene camera is."""
    R = _rot([0, 0, 1], -40) @ _rot([1, 0, 0], 100)  # optical z forward-ish, y down-ish
    t = np.array([-0.02, 0.60, 0.35])
    return transform(R, t)


def test_the_pose_is_recovered_from_two_planes_and_a_point():
    T_true = _scene_truth()
    T_inv = np.linalg.inv(T_true)
    p_tag_base = np.array([0.79, 0.03, 0.49])
    n_door_base = np.array([-0.996, -0.08, 0.02])  # the door faces the arm
    c_door_base = p_tag_base
    up = np.array([0.0, 0.0, 1.0])
    c_table_base = np.array([0.5, -0.1, -0.027])

    def to_scene(p):
        return (T_inv @ np.append(p, 1))[:3]

    p_tag_scene = to_scene(p_tag_base)
    # a plane fit returns either sign: hand the solver the WRONG ones
    door_scene = (-(T_inv[:3, :3] @ n_door_base), to_scene(c_door_base))
    table_scene = (-(T_inv[:3, :3] @ up), to_scene(c_table_base))
    cal = scene_pose_from_planes(p_tag_base, (-n_door_base, c_door_base), p_tag_scene, door_scene, table_scene)
    assert cal.plane_angle_mismatch_deg < 1e-6
    assert np.allclose(cal.T_base_scene, T_true, atol=1e-9)


def test_inconsistent_planes_show_up_as_an_angle_mismatch():
    """A bent door (or a bad plane fit) cannot be hidden: the door/table angle
    disagrees between the frames, and the caller refuses on that."""
    T_true = _scene_truth()
    T_inv = np.linalg.inv(T_true)
    p_tag_base = np.array([0.79, 0.03, 0.49])
    n_door_base = np.array([-1.0, 0.0, 0.0])

    def to_scene(p):
        return (T_inv @ np.append(p, 1))[:3]

    p_tag_scene = to_scene(p_tag_base)
    door_scene = (T_inv[:3, :3] @ (_rot([0, 1, 0], 5) @ n_door_base), p_tag_scene)  # 5 deg off
    table_scene = (T_inv[:3, :3] @ np.array([0, 0, 1.0]), to_scene(np.array([0.5, 0, -0.027])))
    cal = scene_pose_from_planes(p_tag_base, (n_door_base, p_tag_base), p_tag_scene, door_scene, table_scene)
    assert cal.plane_angle_mismatch_deg > PLANE_ANGLE_TOL_DEG


def test_plane_fit_finds_a_table_under_clutter():
    """A tilted table with a box on it and sensor noise: the plane is the table."""
    rng = np.random.default_rng(1)
    k = np.array([[400.0, 0, 200.0], [0, 400.0, 150.0], [0, 0, 1.0]])
    n_true = np.array([0.0, 0.5, 1.0])
    n_true /= np.linalg.norm(n_true)
    d_true = 0.9  # the plane n.p = d, ray-cast per pixel
    v, u = np.mgrid[0:300, 0:400]
    rays = np.stack([(u - 200) / 400.0, (v - 150) / 400.0, np.ones_like(u, float)], -1)
    depth = d_true / (rays @ n_true)
    depth[100:160, 150:230] -= 0.10  # a box top, 10 cm nearer
    depth += rng.normal(0, 0.002, depth.shape)  # sensor noise
    pts, _u, _v = cloud_from_depth(depth, k)
    n, c, frac = fit_plane(pts, tol_m=0.006, rng=rng)
    assert frac > 0.85  # the box is 4 % of the pixels: the table wins
    assert abs(abs(n @ n_true) - 1.0) < 1e-3
    assert abs(abs(n @ c) - d_true) < 0.003


def test_ray_meets_the_plane_where_it_should():
    n = np.array([0.0, 0.0, -1.0])
    c = np.array([0.1, 0.2, 0.9])
    p = ray_plane([0.1, 0.0, 1.0], n, c)
    assert p[2] == pytest.approx(0.9) and p[0] == pytest.approx(0.09)


def test_the_yaml_round_trips_and_hangs_the_camera_off_its_link(tmp_path):
    T_base_opt = _scene_truth()
    # the driver's own link -> optical: the usual optical rotation plus a lens offset
    T_link_opt = transform(quat_to_mat(-0.5, 0.5, -0.5, 0.5), [0.0, -0.024, 0.0])
    T_base_link = optical_to_link(T_base_opt, T_link_opt)
    assert np.allclose(T_base_link @ T_link_opt, T_base_opt, atol=1e-12)
    f = tmp_path / "camera_scene.yaml"
    doc = write_scene_yaml(str(f), T_base_link, "scene_camera_link", note="test")
    assert doc["parent_frame"] == "base_link" and doc["child_frame"] == "scene_camera_link"
    back, T_back = load_scene_yaml(str(f))
    assert np.allclose(T_back, T_base_link, atol=1e-5)


def _tag_image(centre, side=60, shape=(720, 1280)):
    """A grey frame with the calibration tag drawn at `centre` (px)."""
    import cv2
    import numpy as np

    from box_opening.perception.scene_calib import TAG_DICT, TAG_ID

    img = np.full(shape + (3,), 180, np.uint8)
    marker = cv2.aruco.drawMarker(cv2.aruco.getPredefinedDictionary(TAG_DICT), TAG_ID, side)
    pad = 12  # the white quiet zone a detector needs around the marker
    tile = np.full((side + 2 * pad, side + 2 * pad), 255, np.uint8)
    tile[pad:-pad, pad:-pad] = marker
    x0, y0 = int(round(centre[0] - tile.shape[1] / 2)), int(round(centre[1] - tile.shape[0] / 2))
    img[y0 : y0 + tile.shape[0], x0 : x0 + tile.shape[1]] = tile[..., None]
    return img


def test_a_moved_scene_camera_is_caught_before_the_arm_moves():
    """Bench 2026-09-23: the scene camera had been re-aimed since its
    calibration; its fixes were 9-11 cm off and the arm staged beside the
    box, run after run — while the table read the same height every day (a
    pan keeps it) and nothing noticed. The calibration now records where
    the cabinet's tag appears in the scene image (711.5, 217.6 today,
    steady to 0.12 px), and each locate looks for it first: moved more
    than a few pixels, the scene fix is refused and says why."""
    from box_opening.perception.scene_calib import TAG_MOVED_PX, scene_camera_moved

    cal = (711.5, 217.6)
    moved, why = scene_camera_moved(_tag_image(cal), cal)
    assert moved is False and why is None
    moved, why = scene_camera_moved(_tag_image((711.5 + TAG_MOVED_PX + 4, 217.6)), cal)
    assert moved is True and "moved" in why.lower() and "calibrate_scene_camera.py" in why
    moved, why = scene_camera_moved(_tag_image((300, 400)), cal)
    assert moved is True
    import numpy as np

    moved, why = scene_camera_moved(np.full((720, 1280, 3), 180, np.uint8), cal)
    assert moved is None and "not in view" in why  # cannot tell: say so, do not refuse
    assert scene_camera_moved(_tag_image(cal), None) == (None, None)  # an old calibration without tag_px


def test_the_calibration_records_where_the_tag_appears(tmp_path):
    import numpy as np

    from box_opening.perception.scene_calib import load_scene_yaml, scene_tag_px, write_scene_yaml

    p = tmp_path / "camera_scene.yaml"
    write_scene_yaml(str(p), np.eye(4), "scene_camera_link", note="n", tag_px=(711.5, 217.6))
    doc, _T = load_scene_yaml(str(p))
    assert doc["tag_px"] == [711.5, 217.6]
    assert scene_tag_px(str(p)) == (711.5, 217.6)
    write_scene_yaml(str(p), np.eye(4), "scene_camera_link")
    assert scene_tag_px(str(p)) is None
    assert scene_tag_px(str(tmp_path / "missing.yaml")) is None
