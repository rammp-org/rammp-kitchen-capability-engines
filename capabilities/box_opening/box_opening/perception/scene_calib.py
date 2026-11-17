"""Where the scene camera stands, in the arm's frame.

The calibration bridges the two cameras through things they both see. The
wrist camera is already calibrated to the arm (config/camera_d405_wrist.yaml)
and sees the cabinet tag from HOME at 30 cm, so it can place the tag and
the door it is stuck to in base_link. The scene camera sees the same tag,
the same door and the table. From those:

  rotation     the door normal and the table normal, seen in both frames.
               Two big planes, thousands of points each — a reference the
               tag itself cannot give, being 32 pixels wide in the scene
               camera (its pose flips between mirror solutions, 34 deg of
               scatter measured 2026-09-16).
  translation  the tag centre: a pixel ray meeting the door plane, on both
               sides. No dependence on the printed tag size, which the
               label on the wall gets wrong (50 mm claimed, 57-58 mm by
               four depth readings).

Consistency is checked, not assumed: the angle between the door and the
table must agree between the two frames, or the calibration is refused.

Pure numpy here; the ROS grabbing lives in scripts/calibrate_scene_camera.py.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import yaml

from box_opening.perception.d405 import mat_to_quat_xyzw, quat_to_mat
from box_opening.perception.planes import angle_deg

# the door/table angle must match this closely between the two frames
PLANE_ANGLE_TOL_DEG = 2.0
SCENE_CALIB_FILE = "camera_scene.yaml"


def transform(rot, t):
    m = np.eye(4)
    m[:3, :3] = np.asarray(rot, float)
    m[:3, 3] = np.asarray(t, float).ravel()
    return m


def _frame(n1, n2):
    """An orthonormal frame with x along n1 and n2 in the xy plane."""
    e1 = np.asarray(n1, float)
    e1 = e1 / np.linalg.norm(e1)
    e3 = np.cross(e1, np.asarray(n2, float))
    e3 = e3 / np.linalg.norm(e3)
    e2 = np.cross(e3, e1)
    return np.stack([e1, e2, e3], axis=1)


@dataclass(frozen=True)
class PlaneCalibration:
    T_base_scene: np.ndarray  # base_link <- scene camera optical frame
    door_table_deg_base: float
    door_table_deg_scene: float

    @property
    def plane_angle_mismatch_deg(self):
        return abs(self.door_table_deg_base - self.door_table_deg_scene)


def orient_toward(normal, point_on_plane, viewer):
    """A fitted plane normal has an arbitrary sign; return it pointing to the
    side `viewer` is on. The camera is above the table and in front of the
    door, and base_link is likewise above the table and in front of the
    door, so orienting toward the observer makes the two frames agree."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    side = (np.asarray(viewer, float) - np.asarray(point_on_plane, float)) @ n
    return -n if side < 0 else n


def scene_pose_from_planes(
    p_tag_base,
    door_base,
    p_tag_scene,
    door_scene,
    table_scene,
    up_base=(0.0, 0.0, 1.0),
    table_point_base=(0.0, 0.0, -0.03),
):
    """Solve base_link <- scene optical frame.

    p_tag_*: the tag centre in each frame. door_* and table_scene: (normal,
    point-on-plane) as a plane fit returns them, sign unresolved. In base
    the table's normal is `up_base` through `table_point_base`. Returns a
    PlaneCalibration; the caller decides whether plane_angle_mismatch_deg
    is acceptable (PLANE_ANGLE_TOL_DEG)."""
    origin = np.zeros(3)
    n_db = orient_toward(door_base[0], door_base[1], origin)
    n_tb = orient_toward(up_base, table_point_base, origin)
    n_ds = orient_toward(door_scene[0], door_scene[1], origin)
    n_ts = orient_toward(table_scene[0], table_scene[1], origin)
    ang_b = angle_deg(n_db, n_tb)
    ang_s = angle_deg(n_ds, n_ts)
    R = _frame(n_db, n_tb) @ _frame(n_ds, n_ts).T
    t = np.asarray(p_tag_base, float) - R @ np.asarray(p_tag_scene, float)
    return PlaneCalibration(transform(R, t), ang_b, ang_s)


def optical_to_link(T_base_optical, T_link_optical):
    """The camera driver publishes its own tree from its link frame; we
    publish base_link -> link so the tree hangs off the arm's."""
    return np.asarray(T_base_optical) @ np.linalg.inv(np.asarray(T_link_optical))


# The calibration's tag: the ArUco on the cabinet door, seen by the wrist
# from HOME and by the scene camera (scripts/calibrate_scene_camera.py).
TAG_DICT = cv2.aruco.DICT_4X4_50
TAG_ID = 0
# The tag this far (px) from where the calibration saw it: the tag, or the
# camera, has moved. It sits steady to 0.12 px; 5 px at its 1.07 m is ~0.3
# deg of camera rotation — ~5 mm at the box. This used to REFUSE the scene
# camera, and it misfired: the tag rides the cabinet DOOR — at 35 px on
# 2026-09-24 12:30 and 5 px at 15:22, with the arm at the same pixel in both
# scene frames and the fixes' error unchanged — and a refused scene camera
# is a 3 s wrist search. It is a note now; what the scene camera is trusted
# on is what it measured (press_demo.scene_trust: how far its box was from
# the button the wrist found, on the last run).
TAG_MOVED_PX = 5.0


def detect_tag(img):
    """The calibration tag's four corners (px, float) in a BGR image, or None."""
    dictionary = cv2.aruco.getPredefinedDictionary(TAG_DICT)
    params = (
        cv2.aruco.DetectorParameters_create()
        if hasattr(cv2.aruco, "DetectorParameters_create")
        else cv2.aruco.DetectorParameters()
    )
    if hasattr(cv2.aruco, "ArucoDetector"):
        corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        )
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), dictionary, parameters=params)
    if ids is None or TAG_ID not in ids.ravel():
        return None
    return corners[list(ids.ravel()).index(TAG_ID)].reshape(4, 2).astype(np.float64)


def scene_camera_moved(color_bgr, tag_px_cal, tol_px=TAG_MOVED_PX):
    """Has the scene camera moved since its calibration? (moved, why):
    (False, None) the tag is where the calibration saw it; (True, why) it
    is not — the scene's fixes are off by an unknown amount (9-11 cm on
    2026-09-23, after a re-aim nothing noticed); (None, why) the tag is not
    in view, so it cannot be told; (None, None) the calibration predates
    recording the tag."""
    if tag_px_cal is None:
        return None, None
    corners = detect_tag(color_bgr)
    if corners is None:
        return None, "the calibration tag is not in view — cannot confirm the scene camera has not moved"
    c = corners.mean(axis=0)
    d = float(np.hypot(c[0] - tag_px_cal[0], c[1] - tag_px_cal[1]))
    if d > tol_px:
        return True, (
            "the calibration tag has moved %.0f px (at (%.0f, %.0f), calibrated at (%.0f, %.0f)): the cabinet door "
            "moved, or the camera did. Its fix is flown all the same — the close-up aim corrects it, and a camera "
            "that really moved shows as a large scene-vs-wrist error, which stops it being trusted. If it did move: "
            "python3 scripts/calibrate_scene_camera.py" % (d, c[0], c[1], tag_px_cal[0], tag_px_cal[1])
        )
    return False, None


def scene_tag_px(path):
    """Where the calibration in `path` saw the tag (px), or None."""
    try:
        doc, _T = load_scene_yaml(str(path))
    except (OSError, TypeError, KeyError):
        return None
    t = doc.get("tag_px")
    return None if t is None else (float(t[0]), float(t[1]))


def write_scene_yaml(path, T_base_link, child_frame, note="", tag_px=None, parent_frame="base_link"):
    q = mat_to_quat_xyzw(np.asarray(T_base_link)[:3, :3])
    doc = {
        "parent_frame": parent_frame,
        "child_frame": child_frame,
        "xyz": [round(float(v), 5) for v in np.asarray(T_base_link)[:3, 3]],
        "quat_xyzw": [round(float(v), 6) for v in q],
        "note": note,
    }
    if tag_px is not None:
        # where the scene camera saw the tag when this was solved: the check
        # that the camera has not moved since (scene_camera_moved)
        doc["tag_px"] = [round(float(tag_px[0]), 1), round(float(tag_px[1]), 1)]
    with open(path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    return doc


def load_scene_yaml(path):
    with open(path) as f:
        doc = yaml.safe_load(f)
    T = transform(quat_to_mat(*doc["quat_xyzw"]), doc["xyz"])
    return doc, T
