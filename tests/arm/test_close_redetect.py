"""close_redetect: the open door found again from the wrist camera, and the door model moved to it.

Synthetic looks (points sampled off the surfaces the wrist camera sees from the view pose: the
control panel, the frame 2 cm behind it, the cavity, the open door's top edge) and the real ones of
11-16 (fixtures/close_redetect_2026-11-16.npz: the view pose after the push-open, and the same pose
before the open with the door closed)."""

import json
from pathlib import Path

import numpy as np
import pytest
from door_common import close_redetect as cr

FIXTURE = Path(__file__).parent / "fixtures" / "close_redetect_2026-11-16.npz"

# the 11-16 door, roughly as the open wrote it: hinged on the arm's left (+y), swinging toward the
# arm (open_sign -1: clockwise seen from above)
N0 = np.array([-0.998, 0.063, 0.0]) / np.linalg.norm([-0.998, 0.063])
DOOR = {
    "closed_normal": N0.tolist(),
    "closed_grasp_pos": [0.6711, -0.4629, 0.3778],
    "closed_handle": [0.6911, -0.4642, 0.3778],
    # set below: the door's outer corner on the face (as the open saves it: its detected edge)
    "hinge": None,
    "door_mid": [0.72, -0.30, 0.40],
    "door_z": [0.302, 0.488],
    "door_open_deg": 90.0,
    "open_sign": -1.0,
}
DIMS = (0.0437, 0.04, 0.06, 0.40)  # face behind the grasp, door thickness, past the handle, body
CAM = np.array([0.205, -0.236, 0.612])  # the wrist camera at the view pose
_G = np.asarray(DOOR["closed_grasp_pos"])
DOOR["hinge"] = (_G - np.r_[N0[:2] * DIMS[0], 0.0] - np.cross([0, 0, 1.0], N0) * 0.33).tolist()


def _world(shift=(0.0, 0.0)):
    """The real microwave: the door file's geometry shifted -> (hinge, a closed-face point, t)."""
    hinge = np.asarray(DOOR["hinge"], float) + [*shift, 0.0]
    g = np.asarray(DOOR["closed_grasp_pos"], float) + [*shift, 0.0]
    face = g - np.r_[N0[:2] * DIMS[0], 0.0]
    t = -np.cross([0, 0, 1.0], N0)
    t *= 1.0 if t @ (hinge - g) > 0 else -1.0  # toward the hinge
    return hinge, face, t


def _look(shift=(0.0, 0.0), deg=75.0, door="top edge", noise=0.003, seed=0):
    """What the wrist camera sees from the view pose (points in the arm's frame)."""
    rng = np.random.default_rng(seed)
    hinge, face, t = _world(shift)

    def z(k, lo=0.34, hi=0.47):
        return rng.uniform(lo, hi, k)

    a_hinge = float((hinge[:2] - face[:2]) @ t[:2])
    parts = [
        # the control panel, beside the opening on the handle's side: in the closed door's plane
        np.c_[face[:2] + np.outer(rng.uniform(-0.18, -0.05, 2500), t[:2]), z(2500)],
        # the frame round the opening, 2 cm behind it (11-16), and the cavity's back wall
        np.c_[
            face[:2] - N0[:2] * 0.02 + np.outer(rng.uniform(-0.04, a_hinge, 4000), t[:2]), z(4000)
        ],
        np.c_[
            face[:2] - N0[:2] * 0.30 + np.outer(rng.uniform(0.0, a_hinge - 0.03, 2500), t[:2]),
            z(2500),
        ],
    ]
    if door is not None:
        # the open door, swung `deg` about the hinge: its top edge (its middle) or its inner face
        a = rng.uniform(a_hinge - 0.40, a_hinge - 0.02, 1200)
        off = DIMS[1] / 2 if door == "top edge" else DIMS[1]
        closed = face[:2] + np.outer(a, t[:2]) - off * N0[:2]
        opened = hinge[:2] + (closed - hinge[:2]) @ cr._rot(DOOR["open_sign"] * deg).T
        zz = z(len(a), 0.47, 0.49) if door == "top edge" else z(len(a))
        parts.append(np.c_[opened, zz])
    pts = np.concatenate(parts)
    return pts + rng.normal(0, noise, pts.shape)


def _expect(m, shift, deg, tol_m=0.012, tol_deg=2.5):
    hinge, _, _ = _world(shift)
    assert np.linalg.norm(np.asarray(m["hinge"]) - hinge[:2]) <= tol_m
    assert abs(m["deg"] - deg) <= tol_deg


def test_the_door_where_the_file_has_it_is_found_there():
    m = cr.measure(DOOR, _look(), CAM, DIMS)
    _expect(m, (0.0, 0.0), 75.0)
    assert m["door_face_seen"] == "top edge"
    assert m["yaw"] == 0.0  # the face's direction is the saved one


@pytest.mark.parametrize(
    "shift, deg", [((0.04, 0.0), 75.0), ((0.0, -0.05), 90.0), ((-0.03, 0.03), 62.0)]
)
def test_a_moved_microwave_and_its_door_angle_are_measured(shift, deg):
    _expect(cr.measure(DOOR, _look(shift, deg, seed=1), CAM, DIMS), shift, deg)


def test_the_face_is_the_panel_not_the_frame_behind_the_open_door():
    m = cr.measure(DOOR, _look(seed=2), CAM, DIMS)
    assert abs(float(np.asarray(m["shift"]) @ N0[:2])) <= 0.006  # not 2 cm into the frame's depth


def test_a_door_seen_by_its_inner_face_allows_for_its_thickness():
    cam_user_side = np.array([0.30, -0.55, 0.60])  # well off the door's plane, on its inner side
    m = cr.measure(DOOR, _look((0.02, 0.0), 80.0, door="inner", seed=3), cam_user_side, DIMS)
    assert m["door_face_seen"] == "inner"
    _expect(m, (0.02, 0.0), 80.0)


def test_a_door_seen_edge_on_is_taken_at_its_middle():
    m = cr.measure(
        DOOR, _look((0.02, 0.0), 80.0, door="inner", seed=3), CAM, DIMS
    )  # CAM nearly in its plane
    assert m["door_face_seen"] == "edge-on"
    _expect(m, (0.02, 0.0), 80.0, tol_m=DIMS[1] / 2 + 0.005)  # at most half the thickness off


def test_no_door_in_front_of_the_face_refuses():
    with pytest.raises(ValueError, match="is the door open"):
        cr.measure(DOOR, _look(door=None), CAM, DIMS)


def test_no_microwave_in_view_refuses():
    far = np.random.default_rng(0).uniform([2, 2, 0], [3, 3, 1], (3000, 3))
    with pytest.raises(ValueError, match="in the camera's view"):
        cr.measure(DOOR, far, CAM, DIMS)


def test_the_arms_points_are_taken_out():
    pts = np.array([[0.0, 0.0, 0.0], [0.05, 0, 0], [0.5, 0, 0]])
    assert cr.drop_near(pts, [[0.0, 0.0, 0.0]], 0.09).tolist() == [[0.5, 0, 0]]


def test_looks_that_agree_give_their_median_and_ones_that_dont_refuse():
    looks = [cr.measure(DOOR, _look(seed=s), CAM, DIMS, seed=s) for s in (4, 5, 6)]
    got = cr.combine(looks)
    assert got["looks"] == 3 and got["spread"]["hinge_m"] <= cr.AGREE_HINGE_M
    _expect(got, (0.0, 0.0), 75.0)
    moved = cr.move_door(DOOR, got)
    assert moved["door_open_deg"] == pytest.approx(75.0, abs=2.5)
    assert moved["closed_grasp_pos"][2] == DOOR["closed_grasp_pos"][2]  # heights are the open's
    bad = {**looks[0], "hinge": [looks[0]["hinge"][0] + 0.05, looks[0]["hinge"][1]]}
    with pytest.raises(ValueError, match="disagree"):
        cr.combine([bad, *looks[1:]])


def test_frame_points_lift_depth_into_the_arm_frame():
    K = np.array([[100.0, 0, 2], [0, 100.0, 2], [0, 0, 1]])
    T = np.eye(4)
    T[:3, 3] = [0.1, 0.2, 0.3]
    pts = cr.frame_points({"depth_mm": np.full((5, 5), 1000.0), "K": K, "base_T_cam": T}, stride=1)
    assert len(pts) == 25 and pts[:, 2] == pytest.approx(1.3)
    assert cr.frame_points({"depth_mm": np.zeros((5, 5)), "K": K, "base_T_cam": None}) is None


# --- the real 11-16 looks (snorlax, the wrist camera at the view pose; the real door.json) ---
@pytest.fixture(scope="module")
def real():
    f = np.load(FIXTURE)
    door = json.loads(str(f["door"]))
    dims = tuple(float(v) for v in f["dims"])

    def look(name):
        pts = cr.frame_points(
            {"depth_mm": f[f"{name}_depth"], "K": f[f"{name}_K"], "base_T_cam": f[f"{name}_T"]}
        )
        return cr.measure(door, pts, f[f"{name}_T"][:3, 3], dims)

    return look, door


def test_real_door_angle_is_measured_at_75_and_at_94_deg(real):
    look, _ = real
    assert look("open75")["deg"] == pytest.approx(
        75.3, abs=2.0
    )  # the push-open's "90" settled back
    assert look("open94")["deg"] == pytest.approx(94.0, abs=2.0)  # pushed further open by hand


def test_real_hinge_is_the_same_at_both_angles_and_near_the_opens_edge(real):
    look, door = real
    h75, h94 = np.asarray(look("open75")["hinge"]), np.asarray(look("open94")["hinge"])
    assert (
        np.linalg.norm(h75 - h94) <= 0.01
    )  # the microwave didn't move: 11-16 they agree to 0.4 cm
    assert (
        np.linalg.norm(h75 - np.asarray(door["hinge"][:2])) <= 0.025
    )  # the open's detected edge: 1.6-1.8 cm


def test_real_face_is_the_panel_at_both_angles(real):
    look, door = real
    n = np.asarray(door["closed_normal"])[:2]
    edge_depth = float(n @ np.asarray(door["hinge"][:2]))  # the open's detected door face
    for name in ("open75", "open94"):
        assert (
            abs(look(name)["face_depth"] - edge_depth) <= 0.015
        )  # not the frame 2 cm back, nor edge noise


def test_real_closed_door_refuses(real):
    look, _ = real
    with pytest.raises(ValueError, match="is the door open"):
        look("closed")
