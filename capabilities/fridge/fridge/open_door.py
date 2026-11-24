"""Open the fridge: detect the handle, grasp it, swing the door open and let go while still moving, then push it on (or
go round it).

The grasp (the node's `grasp`, constants.GRASP): "side" (side_grasp.py, the default) or "front" (fingers across the bar,
GRASP_PITCHES_DEG, each with the hand either way up). The side grasp differs only up to the hand being off the handle:
- its grasps: SIDE_TILTS_DEG (level first, then out to 40 deg nose-up / nose-down), in along the door from the hinge
  side, 20 deg off its face, one finger between the bar and the door, from a line-up SIDE_SLIDE_M back along the
  hand's level approach (cuRobo), then the slide (local IK) against side_grasp.hold_world;
- its swing: the bar carried round the hinge, the hand turning about it up to SIDE_HAND_LAGS_DEG less than the door (the
  first lag whose swing and back-off plan), checked against the door where it is at each step too;
- its back-off (push_open.back_off_goals): SIDE_BACK_OFF_M back along the hand's axis, off the bar; SIDE_BACK_OUT_M
  out from the door (cuRobo, the hand turned to unbend the wrist); then (cuRobo, round the door and handle) to where
  the front grasp's back-off would have left the hand. From there the push-open (its wheelchair move included), the
  full pull's way round the door, the placement advice and the close are the front grasp's code;
- if no grasp gives a whole open, each is tried again from other arm configurations (side_configs, SIDE_CONFIGS).

1. Detect: door_common's detection (two agreeing looks), with this fridge's values (`configure`): the handle from
   SAM 3's mask on the detection camera, the door's direction from the printed handle's model fit to the mask's depth
   (`door_normal: handle_model`, door_normal.py, the side grasp's method). The hinge is measured from the handle
   (hinge_along_door_m, ...). The grasp is at the middle of the bar, or `grasp_offset_m` above it (below if
   negative), within the measured bar (`set_grasp_offset`).
2. Plan everything before anything moves (planning.py) -- for each grasp in turn, the grasps that worked before first
   (proven.py: what planned and ran on the arm, then constants.PROVEN_SEEDS), then the rest (GRASP_PITCHES_DEG tilts,
   + nose-down and - nose-up, each with the hand either way up); the first whose approach, swing and what follows all
   plan is used:
   - the approach: cuRobo to a line-up point PREGRASP_STANDOFF_M straight out from the grasp (the closed door, its
     handle and the cabinet in its world), then straight in (local IK). If no grasp's swing plans from the arm
     configuration cuRobo chose, the line-up is planned again from others (ALT_CONFIG_SEEDS: elbow turned, wrist
     flipped) and reached by a joint-space plan;
   - the swing: ONE smooth leg round the hinge with the door to exactly `swing_deg`, the hand turning with it, ending
     at rest there (local IK, the spare joint keeping J2/J4/J6 off their limits);
   - what follows (push_open.PushOpen, ReleaseAround, ReleaseBackOff), from the swing's planned end: straight back off
     the handle first.
3. Execute, each motion one trajectory: the approach; the swing under the driver's stiffest joint impedance, the
   gripper closing as it starts; at the swing's end, the arm stopped and the door with it, the gripper opens (its
   reported position must say open); what follows, planned again from where that leaves the arm, as one more trajectory
   under impedance.

Refusals raise door_common.common.Refused: the arm holds where it is.
"""

import json
import threading
import time
from collections import Counter
from functools import partial
from itertools import chain

import numpy as np
from door_common import common as dc
from door_common import open_door as shared
from door_common.common import Refused, quiet, save_door_geometry, say
from scipy.spatial.transform import Rotation as R

from . import constants as F
from . import planning as pl
from . import push_open as P
from . import proven, relocalize
from . import side_grasp as S
from .motion import load_door, logged_run, opened
from .push_open import PushOpen, back_off_goals, push_after_chair_move
from .world import FridgeModel, configure_handle

GRASP_TRACK_ABORT_M = 0.03  # the approach must end the tool this close to the grasp, or the gripper stays open
REPLAN_DEG = 1.0  # after the grasp, the swing is planned again from the real joints if they are further than this off


DOOR_NORMALS = ("handle_model", "handle", "plane", "square")
# per method: two looks' door normals must agree within this (deg). handle_model: the side grasp's estimator calls two
# consecutive fits agreeing within 3 deg ok (11-06 still looks: within 0.7 deg typically, 2.9 at most); handle: on the
# RealSense frames, two looks within 5 deg put 90% of doors within 8.5 deg; plane: 11-04's looks were 31 deg apart
NORMAL_AGREE_DEG = {"handle_model": 3.0, "handle": 5.0, "plane": 10.0, "square": 10.0}


def _make_detector(handle_centre_to_face_m, door_normal="handle_model"):
    """The handle detector for one detect_handle. `door_normal` (DOOR_NORMALS): where each look's door normal comes
    from -- handle_model: the printed handle's model (RAMMP.Fridge.Handle5.STL) registered to SAM 3's handle mask on the
    depth (door_normal.measure_door_normal, the side grasp's estimator); handle: the bar's front and its plate as
    parallel planes; plane: a door-plane fit over the whole image; square: the door taken as square to the arm."""
    if door_normal == "handle_model":
        from .door_normal import make_detector  # numpy / scipy; the ROS side (open3d) only when called

        return make_detector(handle_centre_to_face_m=handle_centre_to_face_m)
    from .perception import HandleDetector  # here, not at import: open3d, which the planning tests don't have

    return HandleDetector(handle_centre_to_face_m, door_normal == "square", door_normal == "handle")


PLACEMENT_CELL_M = 0.05  # constants.PLACEMENT_CELLS' grid step: a handle within half of it of a cell is in that cell
PLACEMENT_SIM_PULL_DEG = 60  # the pull and handle height the cells come from (constants.MOUNTS) ...
PLACEMENT_SIM_Z_M = 0.315
PLACEMENT_Z_TOL_M = 0.05  # ... a handle further than this off that height gets a note that the placement may differ
PLACEMENT_INSET_M = 0.01  # the advice aims this far inside the region (the edge of a cell is where planning gets flaky)
PLACEMENT_DEADBAND_M = 0.02  # ... and never asks for a move smaller than this along an axis: two detections of the same
# placement differ by 1-3 cm (11-09: 2.8 cm between one run's looks), so a 1 cm move is noise -- the source of the
# "1 cm back", then "1 cm closer" advice of 11-09


BASE_MOVES = True  # the running open's base_moves (run sets it): False leaves CELLS_NEEDING_BASE_MOVE out of the advice


def _cells():
    """constants.PLACEMENT_CELLS, less the cells that need the push's wheelchair move when the open may not plan one."""
    cells = [tuple(c) for c in F.PLACEMENT_CELLS or ()]
    if BASE_MOVES:
        return cells
    drop = {(round(cx, 2), round(cy, 2)) for cx, cy in F.CELLS_NEEDING_BASE_MOVE}
    return [c for c in cells if (round(c[0], 2), round(c[1], 2)) not in drop]


def _placement_move(x, y):
    """The smallest handle displacement (dx, dy) (m, arm_mount) that puts the handle (x, y) PLACEMENT_INSET_M inside a
    cell whose four neighbours planned too (constants.PLACEMENT_CELLS: well inside, not on the edge; all cells if none
    is), each axis under PLACEMENT_DEADBAND_M dropped. (0, 0): inside, or close enough that detection noise is larger.
    The nearest POINT of the region, not the nearest cell's centre: a handle already right along one axis gets no move
    along it, and two detections a centimetre apart get the same advice."""
    cells = _cells()
    half = PLACEMENT_CELL_M / 2
    if any(abs(x - cx) <= half and abs(y - cy) <= half for cx, cy in cells):
        return 0.0, 0.0
    have = {(round(cx / PLACEMENT_CELL_M), round(cy / PLACEMENT_CELL_M)) for cx, cy in cells}

    def inside(c):
        i, j = round(c[0] / PLACEMENT_CELL_M), round(c[1] / PLACEMENT_CELL_M)
        return all(n in have for n in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)))

    pool = [c for c in cells if inside(c)] or cells
    reach = half - PLACEMENT_INSET_M
    moves = [
        (float(np.clip(x, cx - reach, cx + reach)) - x, float(np.clip(y, cy - reach, cy + reach)) - y)
        for cx, cy in pool
    ]
    dx, dy = min(moves, key=lambda m: m[0] ** 2 + m[1] ** 2)
    return (dx if abs(dx) >= PLACEMENT_DEADBAND_M else 0.0), (dy if abs(dy) >= PLACEMENT_DEADBAND_M else 0.0)


def chair_moves(dx, dy):
    """The wheelchair moves (words) that move the handle by (dx, dy) in arm_mount: it moves the other way."""

    def cm(d):
        return f"~{max(1, round(abs(d) * 100))} cm"

    out = []
    if dx < 0:
        out.append(f"move the wheelchair {cm(dx)} closer to the fridge")
    elif dx > 0:
        out.append(f"move the wheelchair {cm(dx)} back from the fridge")
    if dy > 0:
        out.append(f"move the wheelchair {cm(dy)} to its right")
    elif dy < 0:
        out.append(f"move the wheelchair {cm(dy)} to its left")
    return out


def placement_moves(handle, normal):
    """The wheelchair moves (words) that get the handle into the placement that passed in sim, and the handle's move
    (dx, dy); [] when it is inside already (or off by less than detection noise)."""
    x, y = float(handle[0]), float(handle[1])
    yaw = float(np.degrees(np.arctan2(-normal[1], -normal[0])))
    if F.PLACEMENT_CELLS:
        dx, dy = _placement_move(x, y)
    else:  # the box: the nearest point of it, the same deadband
        dx = float(np.clip(x, *F.PLACEMENT_X)) - x
        dy = float(np.clip(y, *F.PLACEMENT_Y)) - y
        dx, dy = (d if abs(d) >= PLACEMENT_DEADBAND_M else 0.0 for d in (dx, dy))
    moves = chair_moves(dx, dy)
    if yaw > F.PLACEMENT_YAW_DEG[1]:
        moves.append(f"turn the wheelchair ~{yaw - F.PLACEMENT_YAW_DEG[1]:.0f} deg so its front swings left")
    elif yaw < F.PLACEMENT_YAW_DEG[0]:
        moves.append(f"turn the wheelchair ~{F.PLACEMENT_YAW_DEG[0] - yaw:.0f} deg so its front swings right")
    return moves, (dx, dy)


def _where(handle, normal):
    yaw = round(float(np.degrees(np.arctan2(-normal[1], -normal[0])))) + 0.0  # + 0.0: no "-0"
    return (
        f"handle at x {float(handle[0]):.2f} y {float(handle[1]):.2f}, door yaw {yaw:+.0f} deg; the open planned in "
        f"sim with x {F.PLACEMENT_X[0]:.2f}-{F.PLACEMENT_X[1]:.2f}, y {F.PLACEMENT_Y[0]:.2f} to "
        f"{F.PLACEMENT_Y[1]:.2f}, yaw {F.PLACEMENT_YAW_DEG[0]:+.0f} to {F.PLACEMENT_YAW_DEG[1]:+.0f}"
    )


def placement_primary(handle, normal):
    """For a refusal that IS about feasibility (no grasp gives a whole open that plans): where to move the wheelchair,
    from the sim's placement; "" when the handle is inside it already."""
    moves, _ = placement_moves(handle, normal)
    if not moves:
        return ""
    return f"\n  Wheelchair placement ({_where(handle, normal)}): " + "; ".join(moves) + "."


def placement_secondary(handle, normal):
    """For a refusal about something else: the same advice, said to be secondary; "" when the handle is inside the
    placement already."""
    moves, _ = placement_moves(handle, normal)
    if not moves:
        return ""
    return (
        "\n  (Separately -- NOT the cause of this stop: the wheelchair is also outside the area where the open "
        f"planned in sim ({_where(handle, normal)}). Once the error above is fixed: " + "; ".join(moves) + ".)"
    )


def placement_advice(handle, normal, start_joints=None, level=True):
    """Where to move the wheelchair to get the handle into the placement that passed in sim (a guide: the open's own
    planning decides, and on a refusal search_placement asks it): logged after every detection."""
    x, y = float(handle[0]), float(handle[1])
    yaw = float(np.degrees(np.arctan2(-normal[1], -normal[0])))
    moves, (dx, dy) = placement_moves(handle, normal)
    lines = [
        f"placement: handle x {x:.3f} y {y:.3f}, yaw {yaw:+.1f} deg (passed in sim: x {F.PLACEMENT_X[0]:.2f}-"
        f"{F.PLACEMENT_X[1]:.2f}, y {F.PLACEMENT_Y[0]:.2f} to {F.PLACEMENT_Y[1]:.2f}, yaw "
        f"{F.PLACEMENT_YAW_DEG[0]:+.0f} to {F.PLACEMENT_YAW_DEG[1]:+.0f})"
    ]
    if dx or dy:
        lines[0] += (
            f"; into where the {PLACEMENT_SIM_PULL_DEG} deg open planned in sim: handle x {x + dx:.2f} y {y + dy:.2f}"
        )
    if F.PLACEMENT_CELLS and abs(float(handle[2]) - PLACEMENT_SIM_Z_M) > PLACEMENT_Z_TOL_M:
        lines.append(
            f"  (that sim had the handle {PLACEMENT_SIM_Z_M:.2f} m above the arm base; this one is "
            f"{float(handle[2]):.2f} m: the placement may differ)"
        )
    lines += [f"  -> {m}" for m in moves] or ["  -> inside that placement (a sim guide: the plan decides)"]
    if len(moves) > 1:
        lines.append("  (re-check after each move: turning the chair also shifts the handle)")
    if level and start_joints is not None and np.degrees(start_joints[0]) < 0:  # that pose is the level arm's
        lines.append(f"  -> start from the web app's Fridge start pose (J1 now {np.degrees(start_joints[0]):.0f} deg)")
    return "\n".join(lines)


_HANDLE_LENGTH = F.MEASURED["handle_length_m"]  # configure sets these two
_DOOR_Z_FROM_HANDLE = (-F.MEASURED["door_below_handle_m"], F.MEASURED["door_above_handle_m"])


def configure(data_dir, mount_tilt_deg=0, door_normal="handle_model", grasp="side", **measured):
    """Everything the detection and the door model need from the fridge: call once at startup. `mount_tilt_deg`: a key
    of constants.MOUNTS. `door_normal`: how each look measures the door's direction (_make_detector, DOOR_NORMALS;
    handle_model and handle need SAM 3's mask). `grasp`: constants.GRASPS. `measured`: constants.MEASURED's names."""
    if door_normal not in DOOR_NORMALS:
        raise ValueError(f"door_normal must be one of {', '.join(DOOR_NORMALS)}, not {door_normal!r}")
    global _HANDLE_LENGTH, _DOOR_Z_FROM_HANDLE
    m = {**F.MEASURED, **measured}
    _HANDLE_LENGTH = float(m["handle_length_m"])
    configure_handle(_HANDLE_LENGTH)
    _DOOR_Z_FROM_HANDLE = (-m["door_below_handle_m"], m["door_above_handle_m"])
    F.configure(data_dir, mount_tilt_deg, grasp)
    mount = F.MOUNTS[mount_tilt_deg]
    # the hinge and the door's face from the handle (where the fingers close), as they are measured. The offset in the
    # door's frame: into it, along it, up. A wrong one drags the door or the fridge during the swing
    hinge_offset = np.array([m["handle_centre_to_face_m"] + m["pin_behind_face_m"], m["hinge_along_door_m"], 0.0])
    shared.configure(
        MAKE_DETECTOR=partial(_make_detector, m["handle_centre_to_face_m"], door_normal),
        FIND_HINGE=shared.hinge_measured(hinge_offset, from_handle=True),
        GRIP_EXT=F.GRIP_EXT,
        VERTICAL_CORR=F.VERTICAL_CORR,
        GRASP_X_ADJUST=F.GRASP_X_ADJUST,
        GRASP_FORCE=F.GRASP_FORCE,
        # no gates on where the handle is or how high the grasp is: cuRobo's plans decide what can be done
        PLAUSIBLE_X=None,
        PLAUSIBLE_Y=None,
        PLAUSIBLE_Z=None,
        MIN_Z=None,
        MAX_Z=None,
        HINGE_RADIUS_RANGE=None,  # the measured hinge is always hinge_along_door_m from the grasp
        # no detected hinge to compare: the two looks' door normals directly
        NORMAL_AGREE_DEG=NORMAL_AGREE_DEG[door_normal],
        MAX_REACH=mount["max_reach"],  # a grasp further out refuses before planning, with the placement advice
        PLACEMENT_NOTE=partial(placement_advice, level=mount_tilt_deg == 0),
        DOOR_MODEL=dict(
            face_behind_grasp=m["handle_centre_to_face_m"],
            door_t=m["door_depth_m"],
            door_past_handle=m["door_width_m"] - m["hinge_along_door_m"],
            body_depth=m["body_depth_m"],
        ),
        DOOR_FACE_BEHIND_HANDLE=m["handle_centre_to_face_m"],
        DOOR_Z_FROM_HANDLE=_DOOR_Z_FROM_HANDLE,  # the camera sees only part of it
        KEEP_ARM_IN_FRAMES=True,
    )


def grasp_offset_limit():
    """How far above or below the middle of the bar the grasp may be (m): the finger pads stay GRASP_END_MARGIN_M in
    from either end of the measured bar."""
    return max(0.0, _HANDLE_LENGTH / 2 - F.GRASP_END_MARGIN_M)


def set_grasp_offset(offset):
    """Grasp `offset` (m) above the detected middle of the bar (below if negative) for this run; Refused beyond
    grasp_offset_limit. door_common's grasp height is the bar's middle + VERTICAL_CORR, and the door's top and bottom
    are placed from that grasp height (DOOR_Z_FROM_HANDLE), so both move with the offset."""
    limit = grasp_offset_limit()
    if abs(offset) > limit + 1e-9:
        raise Refused(
            f"grasp_offset_m {offset:+.3f} is off the handle: at most +-{limit:.3f} for a "
            f"{_HANDLE_LENGTH * 100:.1f} cm bar (handle_length_m), keeping the finger pads "
            f"{F.GRASP_END_MARGIN_M * 100:.0f} cm in from its ends."
        )
    lo, hi = _DOOR_Z_FROM_HANDLE
    shared.configure(VERTICAL_CORR=F.VERTICAL_CORR + offset, DOOR_Z_FROM_HANDLE=(lo - offset, hi - offset))
    if offset:
        say(f"grasping {abs(offset) * 100:.1f} cm {'above' if offset > 0 else 'below'} the middle of the bar")


# --- the door, the grasp, the swing -----------------------------------------------------------------------------------


def opening_sign(n, hinge, handle):
    """DoorFrame's sense for a door that opens toward the arm (its free edge moves out along the closed normal `n`)."""
    n2 = np.asarray(n, float)[:2] / np.linalg.norm(np.asarray(n, float)[:2])
    e = np.asarray(handle, float)[:2] - np.asarray(hinge, float)[:2]
    e -= n2 * (e @ n2)
    return 1.0 if (-e[1] * n2[0] + e[0] * n2[1]) > 0 else -1.0


def closed_door(geom, h, grasp_pos, cand=None):
    """The door dict of the closed door: the door file's fields, from the detection `geom`, the handle `h` and the
    grasp's tool point (`grasp_pos`: the door's frame is built on it). `cand`: the grasp (grasp_candidates'), if the
    hand holds the door anywhere but at grasp_pos (the side grasp: grasp_pos is then side_grasp.door_reference): where
    it holds the bar (closed_hold_pos, and hold_in_tool: that point in the tool frame), which the door's angle is read
    from (hand_angle), and where the front grasp would have held it (front_hold)."""
    hold, in_tool = np.asarray(grasp_pos, float), np.zeros(3)
    if cand is not None and cand.get("bar") is not None:
        hold = np.asarray(cand["bar"], float)
        in_tool = R.from_quat(cand["quat"]).inv().apply(hold - np.asarray(cand["pos"], float))
    n, hinge = np.asarray(geom["n"], float), np.asarray(geom["hinge"], float)
    door = dict(
        closed_normal=list(n),
        closed_grasp_pos=list(map(float, grasp_pos)),
        closed_handle=list(map(float, h)),
        handle_mid_z=float(geom["h_raw"][2] + F.VERTICAL_CORR),
        door_z=list(map(float, h[2] + np.asarray(shared.DOOR_Z_FROM_HANDLE))),
        hinge=list(hinge),
        door_open_deg=0.0,
        handle_length_m=_HANDLE_LENGTH,
        closed_hold_pos=list(map(float, hold)),
        hold_in_tool=list(map(float, in_tool)),
        grasp=F.GRASP,
        **({"front_hold": list(map(float, cand["front"]))} if cand is not None and cand.get("front") else {}),
        **shared.door_model(n, grasp_pos, h),
    )
    door["open_sign"] = opening_sign(n, hinge, h)
    return door


def swung(pos, quat, hinge, sign, deg):
    """Tool pose (`pos`, `quat`) carried round the vertical hinge axis through `hinge` with the door, `deg` open."""
    rot = R.from_euler("z", sign * deg, degrees=True)
    p = np.asarray(pos, float) - np.asarray(hinge, float)
    return np.asarray(hinge, float) + rot.apply(p), (rot * R.from_quat(quat)).as_quat()


def hand_angle(door, ee):
    """The door's opening (deg, signed as DoorFrame.angle_of) from the hand holding the handle at tool pose `ee` (x y
    z qx qy qz qw): how far the point it holds the bar by (hold_in_tool) has turned about the hinge from where it was
    on the closed door (closed_hold_pos). A door file without them: the tool point against closed_grasp_pos."""
    ref = np.asarray(door.get("closed_hold_pos") or door["closed_grasp_pos"], float)[:2]
    p = np.asarray(ee[:3], float)
    if door.get("hold_in_tool") is not None and len(ee) >= 7:
        p = p + R.from_quat(np.asarray(ee[3:7], float)).apply(np.asarray(door["hold_in_tool"], float))
    d = p[:2] - np.asarray(door["hinge"], float)[:2]
    g = ref - np.asarray(door["hinge"], float)[:2]
    return float(np.degrees(np.arctan2(g[0] * d[1] - g[1] * d[0], g @ d)))


class ArcTrace:
    """While it is open (`with`): the tool's position (x y, arm_mount) every 1/HINGE_FIT_HZ s, while the gripper is
    closed -- the hand on the handle, so the path is the door's arc."""

    def __init__(self, ai):
        self.ai, self.xy = ai, []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _sample(self):
        while not self._stop.wait(1.0 / F.HINGE_FIT_HZ):
            try:
                st = self.ai.get_state()
                if float(st["gripper_pos"]) > 0.5:
                    self.xy.append(np.asarray(st["ee_pos"][:2], float))
            except Exception:  # a missed sample only thins the fit
                pass


def fit_circle(xy):
    """The least-squares circle through points `xy` (N, 2): (centre, radius, RMS off it m, the arc it spans deg)."""
    p = np.asarray(xy, float)
    cx, cy, c = np.linalg.lstsq(np.c_[2 * p, np.ones(len(p))], (p**2).sum(axis=1), rcond=None)[0]
    centre = np.array([cx, cy])
    r = float(np.sqrt(c + cx**2 + cy**2))
    d = p - centre
    rms = float(np.sqrt(np.mean((np.linalg.norm(d, axis=1) - r) ** 2)))
    span = float(np.degrees(np.ptp(np.unwrap(np.arctan2(d[:, 1], d[:, 0])))))
    return centre, r, rms, span


def record_swing_hinge(xy):
    """The hinge from the swing's path `xy` (ArcTrace), into the door file (hinge and hinge_pivot) when the fit is
    good (HINGE_FIT_*); else the estimate stays. Also says it as the node's measured offset, from the handle. Returns
    True when the door file has the fitted hinge."""
    if len(xy) < 10:
        say(f"  hinge from the swing: {len(xy)} samples with the gripper closed -- not fitted, the estimate stays")
        return False
    door = load_door()
    old = np.asarray(door["hinge"], float)
    centre, r, rms, span = fit_circle(xy)
    shift = float(np.linalg.norm(centre - old[:2]))
    n = np.asarray(door["closed_normal"], float)[:2]
    n /= np.linalg.norm(n)
    d = centre - np.asarray(door["closed_handle"], float)[:2]
    into, along = float(d @ -n), float(d @ np.array([n[1], -n[0]]))  # as hinge_measured: -n, n x z
    msg = (
        f"hinge from the swing's arc: {np.round(centre, 4)}, radius {r:.3f} m, {rms * 1000:.1f} mm RMS over "
        f"{span:.0f} deg, {shift * 100:.1f} cm from the estimate (from the handle: {into:.3f} m into the door, "
        f"{along:.3f} m along it)"
    )
    if span < F.HINGE_FIT_MIN_DEG or rms > F.HINGE_FIT_MAX_RMS_M or shift > F.HINGE_FIT_MAX_SHIFT_M:
        say(
            f"  {msg} -- not used (needs {F.HINGE_FIT_MIN_DEG:.0f} deg, {F.HINGE_FIT_MAX_RMS_M * 1000:.0f} mm RMS, "
            f"{F.HINGE_FIT_MAX_SHIFT_M * 100:.0f} cm): the estimate stays"
        )
        return False
    hinge = [float(centre[0]), float(centre[1]), float(old[2])]
    save_door_geometry(hinge=hinge, hinge_pivot=hinge, hinge_estimate=old.tolist())
    say(f"  {msg} -- the door file's hinge now")
    return True


def record_hand_angle(ee):
    """After the swing, the hand still on the handle: the door's angle and sense from it, into the door file (as
    door_common's record_door_angle, read against where the hand held the closed door)."""
    door = load_door()
    a = hand_angle(door, ee)
    save_door_geometry(door_open_deg=abs(a), open_sign=float(np.sign(a) or 1.0))


def grasp_candidates(geom):
    """The grasps tried, in order: each tilt (+ nose-down, - nose-up), the hand up then upside down (fingers across the
    bar both ways). Dicts {pitch, flip, h, pos, quat}. A tilt whose grasp door_common refuses (beyond MAX_REACH) is
    skipped. The side grasp (F.GRASP "side"): side_grasp.candidates."""
    if F.GRASP == "side":
        return S.candidates(geom)
    out = []
    for pitch in F.GRASP_PITCHES_DEG:
        try:
            with quiet():
                h, g = shared.grasp_pose(geom["h_raw"], geom["orient"], geom["n"], pitch)
        except Refused as e:
            say(f"  tilt {pitch:2d} deg: {e}")
            continue
        for flip in (False, True):
            quat = np.asarray(g.orientation, float)
            if flip:
                quat = (R.from_quat(quat) * R.from_euler("z", 180, degrees=True)).as_quat()
            out.append(
                dict(pitch=pitch, flip=flip, h=np.asarray(h, float), pos=np.asarray(g.position, float), quat=quat)
            )
    return out


def lineup_point(cand, standoff):
    """`standoff` straight back from the grasp along its level approach (= out of the door face)."""
    approach = R.from_quat(cand["quat"]).as_matrix()[:, 2]
    approach[2] = 0.0
    return cand["pos"] - approach / np.linalg.norm(approach) * standoff


def approach_goals(model, cand, standoff, q_lineup=None):
    """To the line-up point (a free leg; or a joint-space one to `q_lineup`, cuRobo's joints for it), then straight in
    (the side grasp: the slide along the door, against side_grasp.hold_world)."""
    pre = lineup_point(cand, standoff)
    if q_lineup is None:
        first = pl.free(
            pre,
            cand["quat"],
            f"line up {standoff * 100:.0f} cm out",
            model.full(0.0),
            blend=0.03,
            world_label="the closed door",
        )
    else:
        first = pl.to_joints(
            q_lineup,
            pre,
            cand["quat"],
            f"line up {standoff * 100:.0f} cm out (other arm configuration)",
            model.full(0.0),
            blend=0.03,
            world_label="the closed door",
        )
    if cand.get("side"):
        return [
            first,
            pl.line(
                cand["pos"],
                cand["quat"],
                "slide in to the handle",
                hand_world(model, 0.0),
                world_label="the closed door, the side grasp's window (face recessed)",
            ),
        ]
    return [
        first,
        pl.line(
            cand["pos"],
            cand["quat"],
            "straight in to the handle",
            hand_world(model, 0.0),
            world_label="the closed door, hand window",
        ),
    ]


def standoffs(cand):
    """The line-up distances tried for grasp `cand`, in order: the one that worked before first (cand["proven"])."""
    return proven.first(
        (cand.get("proven") or {}).get("standoff"), F.SIDE_SLIDE_M if cand.get("side") else F.PREGRASP_STANDOFF_M
    )


def hand_lags(cand):
    """The side grasp's hand lags tried for grasp `cand`, in order: the one that worked before first; (0,) for the
    front grasp."""
    if not cand.get("side"):
        return [0.0]
    return proven.first((cand.get("proven") or {}).get("lag"), F.SIDE_HAND_LAGS_DEG)


def proven_file():
    """The record of the grasps that planned and ran (proven.py), next to the door file."""
    return dc.DOOR_FILE.parent / proven.FILE_NAME


def record_proven(plan, geom, source):
    """Record `plan`'s grasp as one that worked (`source`: "plan" or "arm"), for the next search to try first."""
    err = proven.record(proven_file(), proven.make_record(F.GRASP, plan, geom["h_raw"], geom["n"], source))
    if err:
        say(f"  (could not record the grasp that worked: {err})")


def hand_world(model, deg):
    """The door open `deg` with a window round the hand on the handle (and the handle) left out: onto or off it. The
    side grasp's: side_grasp.hold_world."""
    if F.GRASP == "side":
        return S.hold_world(model, (deg,))
    return model.holding(
        (deg,), model.df.u_handle, float(model.door["handle_mid_z"]), F.HAND_WINDOW_U_M, F.HAND_WINDOW_Z_M
    )


def swing_goals(model, cand, swing, lag=0.0):
    """Holding the handle: ONE leg round the hinge with the door to exactly `swing` deg, the hand turning with it,
    ending at rest there (the gripper opens only after: user, 11-09). Points every SWING_POINT_DEG along the arc (local
    IK through them). The side grasp: the hand turns about the bar `lag` deg less than the door
    (side_grasp.swing_poses), and each step is checked against the door where it is then, too (side_grasp.swing_worlds:
    its arm runs alongside the door). Returns [goal]."""
    df = model.df
    n = max(2, int(np.ceil(swing / SWING_POINT_DEG)))
    degs = [swing * k / n for k in range(1, n + 1)]
    if cand.get("side"):
        poses = S.swing_poses(cand, df, swing, lag, degs)
    else:
        poses = [swung(cand["pos"], cand["quat"], df.hinge, df.sign, d) for d in degs]
    # the door turns with the hand, so no one world can hold it: the swing is checked against the cabinet (with the
    # planner's chair layers and the arm itself), as the old planner's pull was
    return [
        pl.through(
            poses,
            f"swing to {swing:.0f} deg",
            model.cabinet(),
            world_label="the cabinet (the door turns with the hand)",
            segment_worlds=S.swing_worlds(model, degs) if cand.get("side") else None,
        )
    ]


SWING_POINT_DEG = 2.0  # the swing's path: a pose every this much door


def seeds(q0):
    """Start joints for planning the line-up from other arm configurations (ALT_CONFIG_SEEDS)."""
    out = []
    for kind, deg in F.ALT_CONFIG_SEEDS:
        q = np.array(q0, float)
        if kind == "J3":
            q[2] += np.radians(deg)
        else:  # the other wrist branch: the same hand
            q[4], q[5], q[6] = q[4] + np.pi, -q[5], q[6] + np.pi
        out.append((f"{kind} {deg:+.0f}" if kind == "J3" else "wrist flipped", q))
    return out


class Infeasible(Refused):
    """The live placement did not yield a complete motion within preflight."""

    def __init__(self, message, *, payload=None):
        super().__init__(message)
        self.payload = payload


class StagingRequired(Refused):
    def __init__(self, payload):
        self.payload = {"executed": False, **payload}
        super().__init__(payload["reason"] + " -- " + payload["staging_reason"])


OPEN_PREFLIGHT_S = 8.0


def base_rotation_hint(normal):
    yaw = float(np.degrees(np.arctan2(-normal[1], -normal[0])))
    correction = yaw - float(np.clip(yaw, *F.PLACEMENT_YAW_DEG))
    if correction:
        direction = "left" if correction > 0 else "right"
        return {"base_rotation_deg": correction, "suggestion": f"Rotate base by {abs(correction):.1f}° {direction}."}
    return {}


def plan_open(planner, geom, q0, swing, after, *, staging=None, adaptive_pull=False):
    """Plan the live open with a deadline shared by all of its candidates and planner calls."""
    previous_deadline = planner.deadline
    deadline = time.monotonic() + OPEN_PREFLIGHT_S
    planner.deadline = deadline if previous_deadline is None else min(deadline, previous_deadline)
    diagnostics = []
    try:
        result = _plan_open(planner, geom, np.asarray(q0), swing, after, diagnostics, staging, adaptive_pull)
        try:
            planner.join(result["approach"], result["model"].cabinet())
        except (pl.OverBudget, pl.PlannerBusy):
            raise
        except Refused as e:
            diagnostics.append({"stage": "approach join", "reason": str(e), "candidate": result["name"]})
            raise Infeasible(str(e)) from e
        return result
    except (Infeasible, pl.OverBudget) as e:
        counts = Counter("line-up" if item["stage"].startswith("line up") else item["stage"] for item in diagnostics)
        raise Infeasible(
            str(e),
            payload={
                "executed": False,
                "reason": str(e),
                "failure_counts": dict(counts.most_common(4)),
                "candidates": diagnostics,
                **base_rotation_hint(geom["n"]),
            },
        ) from e
    finally:
        planner.deadline = previous_deadline


def _plan_open(planner, geom, q0, swing, after, diagnostics, staging, adaptive_pull):
    """Plan the whole open from joints `q0` (see the module): every grasp with cuRobo's own arm configuration first,
    then every grasp from the other configurations (seeds). Returns the plan dict; raises Refused (with why)."""
    planner.search()
    cands = grasp_candidates(geom)
    if not cands:
        raise Infeasible("no grasp: every tilt's grasp is out of reach")
    # the door model is the same for every grasp (the face is placed from the handle); the file keeps the chosen one's
    model = FridgeModel(closed_door(geom, cands[0]["h"], cands[0].get("ref", cands[0]["pos"]), cands[0]))
    collision = planner.preflight_start(
        model.full(0.0), q0, np.radians(F.START_JOINTS_DEG) if staging is None else staging
    )
    if collision:
        raise StagingRequired(collision)
    # the grasps that worked before first (proven.py), each with its line-up and lag first; the rest in their order
    prefs = proven.preferred(proven.load(proven_file()) + list(F.PROVEN_SEEDS), F.GRASP, geom["h_raw"], geom["n"])
    cands = proven.order(cands, prefs)
    first_tried = [c["proven"] for c in cands if c.get("proven")]
    if first_tried:
        say(f"  tried first (worked before): {'; '.join(proven.describe(r) for r in first_tried)}")
    tried = []
    adaptive_pull = adaptive_pull and isinstance(after, PushOpen)
    pull_angles = P.pull_angles(swing) if adaptive_pull else (swing,)

    def fail(name, why, detail=None):
        item = dict(detail or {}, candidate=name)
        item.setdefault("stage", why.partition(":")[0])
        item.setdefault("reason", why)
        diagnostics.append(item)
        tried.append(f"{name}: {why}")
        say("    candidate rejected " + json.dumps(item))

    def side_pull_options(cand, insert_joints, name):
        for angle in pull_angles:
            ranked = []
            for lag in F.SIDE_HAND_LAGS_DEG:
                why, margin = side_sweep_score(planner, model, cand, angle, lag, insert_joints)
                if why:
                    fail(f"{name}, {angle:g}°, lag {lag:g}°", why, {"previous_limit_margin_deg": margin})
                else:
                    ranked.append((margin, lag))
            for _, lag in sorted(ranked, reverse=True):
                yield angle, lag

    def grasp_and_swing(cand, hint_name=None, seed=None, lineup=None):
        """The approach and the swing for grasp `cand` (from another arm configuration: `seed`; or `lineup`, (standoff,
        joints): the side grasp's configurations, side_configs), or None."""
        if cand.get("side"):
            name = S.name(cand)
        else:
            name = f"tilt {cand['pitch']} deg, hand {'upside down' if cand['flip'] else 'up'}"
        if hint_name:
            name += f", {hint_name}"
        for standoff in standoffs(cand) if lineup is None else (lineup[0],):
            q_lineup = None if lineup is None else lineup[1]
            if seed is not None:  # cuRobo's line-up joints, planned from another configuration
                goal = pl.free(
                    lineup_point(cand, standoff),
                    cand["quat"],
                    "line up (from another configuration)",
                    model.full(0.0),
                    world_label="the closed door",
                )
                leg, why = planner.leg(goal, seed)
                if leg is None or not len(leg.q):
                    fail(name, why or "line-up: planner returned no samples", planner.last_failure)
                    continue
                q_lineup = leg.q[-1]
            goals = approach_goals(model, cand, standoff, q_lineup)
            lineup_motion, why = planner.motion(goals[:1], q0, from_arm=True)
            if lineup_motion is None:
                fail(f"{name}, {standoff:.3f} m", why, planner.last_failure)
                continue
            line_goal = goals[0]
            line_pose = (line_goal.pos, line_goal.quat)
            options = iter(((swing, 0.0),))
            if cand.get("side"):
                insert_joints, why, margin = local_path(planner, goals[1:], lineup_motion.q_end, line_pose)
                if insert_joints is None:
                    fail(name, why, {"previous_limit_margin_deg": margin})
                    continue
                options = side_pull_options(cand, insert_joints, name)
            first = next(options, None)
            if first is None:
                continue
            insert, why = planner.motion(goals[1:], lineup_motion.q_end, start_pose=line_pose)
            if insert is None:
                fail(f"{name}, {standoff:.3f} m", why, planner.last_failure)
                if not cand.get("side") and pl.goal_unreachable(why):
                    break
                continue
            approach = pl.Motion(lineup_motion.goals + insert.goals, lineup_motion.legs + insert.legs, q0)
            for angle, lag in chain((first,), options):
                sw, why = planner.motion(
                    swing_goals(model, cand, angle, lag), approach.q_end, start_pose=(cand["pos"], cand["quat"])
                )
                if sw is not None and cand.get("side"):
                    end = (sw.goals[-1].pos, sw.goals[-1].quat)
                    off, why = planner.motion(back_off_goals(model, angle, end), sw.q_end, start_pose=end)
                    if off is None:
                        why, sw = f"swing OK but {why}", None
                if sw is None:
                    fail(name, why)
                    continue
                label = S.name(cand, lag) + (f", {hint_name}" if hint_name else "") if lag else name
                found = dict(
                    name=label,
                    cand=cand,
                    standoff=standoff,
                    approach=approach,
                    swing=sw,
                    hint=hint_name,
                    lag=lag,
                    swing_deg=angle,
                )
                if not adaptive_pull:
                    return found
                complete = after_swing(found)
                if complete is not None:
                    return complete
            if not adaptive_pull:
                break
        return None

    def after_swing(found):
        c = found["cand"]
        door = closed_door(geom, c["h"], c.get("ref", c["pos"]), c)
        sw = found["swing"]
        angle = found["swing_deg"]
        args = (planner, opened(door, angle), sw.q_end, (sw.goals[-1].pos, sw.goals[-1].quat))
        # Keep the established pull for a base move. The shorter handoff is planned at the measured dock.
        ok, msg = after.plan(*args, in_place=angle < swing) if adaptive_pull else after.plan(*args)
        if not ok:
            fail(found["name"], f"after swing: {msg}")
            return None
        say(f"  plan: {found['name']}, line-up {found['standoff'] * 100:.0f} cm out, swing {angle} deg; {msg}")
        return dict(found, door=door, model=model, after_msg=msg)

    def configs(cand):
        return side_configs(planner, model, cand, q0, pull_angles[0])

    try:
        return _search(cands, grasp_and_swing, None if adaptive_pull else after_swing, q0, tried, configs)
    except pl.OverBudget as e:
        raise Infeasible(f"{e}; {len(tried)} grasps failed before it: {why_not(tried)}") from e


def local_path(planner, goals, joints, pose):
    """Follow the same IK branch and step guard as the collision-checked planner legs."""
    q = np.asarray(joints, float)
    margin = planner.limit_margin(q)
    for goal in goals:
        poses, _ = pl.path_poses(goal, np.asarray(pose[0], float), np.asarray(pose[1], float))
        for k, (p, quat) in enumerate(poses):
            planner.check_budget()
            nq, why = planner.fk.ik(p, quat, q)
            if nq is None:
                return None, f"{goal.label}: point {k + 1}/{len(poses)}: {why}", margin
            if np.degrees(np.max(np.abs(nq - q))) > pl.MAX_STEP_DEG:
                return None, f"{goal.label}: point {k + 1}/{len(poses)}: a joint jumps", margin
            q = nq
            margin = min(margin, planner.limit_margin(q))
        pose = (goal.pos, goal.quat)
    return q, "", margin


def side_sweep_score(planner, model, cand, swing, lag, q_grasp):
    (goal,) = swing_goals(model, cand, swing, lag)
    back = back_off_goals(model, swing, (goal.pos, goal.quat))[0]
    _, why, margin = local_path(planner, (goal, back), q_grasp, (cand["pos"], cand["quat"]))
    return why, margin


def side_configs(planner, model, cand, q0, swing):
    """Rank line-up configurations by minimum joint clearance through insert, pull and back-off."""
    standoff = F.SIDE_SLIDE_M[0]
    pos = lineup_point(cand, standoff)
    found = []
    for name, seed in S.config_seeds(q0):
        planner.check_budget()
        q, _ = planner.fk.ik(pos, cand["quat"], seed)
        if q is None:
            continue
        for wrapped in planner.fk.wrist_equivalents(q):
            if any(np.max(np.abs(wrapped - item[2])) < np.radians(S.SAME_CONFIG_DEG) for item in found):
                continue
            goals = approach_goals(model, cand, standoff, wrapped)
            grasp, _, insert_margin = local_path(planner, goals[1:], wrapped, (pos, cand["quat"]))
            if grasp is None:
                continue
            for lag in F.SIDE_HAND_LAGS_DEG:
                why, margin = side_sweep_score(planner, model, cand, swing, lag, grasp)
                if not why:
                    label = name if np.array_equal(wrapped, q) else f"{name}, alternate wrist wrap"
                    found.append((min(insert_margin, margin), label, wrapped))
                    break
    found.sort(key=lambda item: (-item[0], np.max(np.abs(q0 - item[2]))))
    return [(name, (standoff, q)) for _, name, q in found[: F.SIDE_CONFIGS]]


def _search(cands, grasp_and_swing, after_swing, q0, tried, configs=None):
    """The first whole open that plans: for each grasp in `cands`' order (proven.order: what worked before first), its
    approach and swing, then at once what follows the swing -- the first grasp where all of it plans is used, the rest
    are never tried. 1. with cuRobo's own arm configuration; 2. only if no grasp's swing planned at all, from the other
    configurations (seeds), the first that swings per grasp; 3. the side grasp (`configs`: side_configs), if nothing
    gave a whole open, every grasp again from its own other configurations."""
    any_swung = False
    for cand in cands:
        found = grasp_and_swing(cand)
        if found:
            any_swung = True
            done = found if after_swing is None else after_swing(found)
            if done:
                return done
    if not any_swung:
        say(
            f"  no grasp's approach + swing plans with the arm configurations cuRobo chose ({len(tried)} tried); "
            "trying others"
        )
        for cand in cands:
            for hint_name, seed in seeds(q0):
                found = grasp_and_swing(cand, hint_name, seed)
                if found:
                    done = found if after_swing is None else after_swing(found)
                    if done:
                        return done
                    break
    if configs is not None and any(c.get("side") for c in cands):
        say(
            f"  no whole open with the arm configurations tried ({len(tried)} attempts); the side grasp's others "
            f"(up to {F.SIDE_CONFIGS} per grasp, the most wrist room first)"
        )
        for cand in cands:
            for hint_name, lineup in configs(cand):
                found = grasp_and_swing(cand, hint_name, lineup=lineup)
                done = found if after_swing is None else found and after_swing(found)
                if done:
                    return done
    raise Infeasible(f"no grasp gives a whole open that cuRobo plans ({len(tried)} tried): {why_not(tried)}")


PLACEMENT_SEARCH_MOVES_M = (0.05, 0.10, 0.15)  # on a refused open: the handle shifted this far (along each axis and
# both at once), as wheelchair moves would shift it -- after the sim table's own suggestion -- and the open planned
# again there: the advice the planner itself checked
PLACEMENT_SEARCH_S = 30.0  # shared budget for an explicitly requested dry-run placement search


def _shifted_geom(geom, d):
    """The detection `geom` with the handle and hinge moved by `d` (dx, dy) in arm_mount: the fridge as it would stand
    after a wheelchair move (no turn)."""
    out = dict(geom)
    for k in ("h_raw", "hinge"):
        if out.get(k) is not None:
            out[k] = np.asarray(out[k], float) + np.r_[d[0], d[1], 0.0]
    return out


def search_placement(planner, geom, q0, swing, make_after, hint=(0.0, 0.0), *, adaptive_pull=False):
    """After the open refused where the wheelchair stands: plan it again with the fridge shifted as wheelchair moves
    would shift it -- first the sim table's suggestion `hint` (dx, dy, the handle's move), then PLACEMENT_SEARCH_MOVES_M
    along each axis and diagonally, smallest first (among equal sizes, those along `hint` first) -- each try the whole
    open's search with its own budget, all within PLACEMENT_SEARCH_S. Returns the advice in words: the first move after
    which the whole open plans, or that none did (and which were not tried, if time ran out)."""
    if PLACEMENT_SEARCH_S <= 0:
        return ""
    hint = np.round(np.asarray(hint, float), 2)
    ring = set()
    for m in PLACEMENT_SEARCH_MOVES_M:
        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                if sx or sy:
                    ring.add((round(sx * m, 3), round(sy * m, 3)))
    order = sorted(ring, key=lambda d: (round(float(np.hypot(*d)), 3), -float(np.dot(d, hint)), d))
    if hint.any():
        h = (float(hint[0]), float(hint[1]))
        order = [h] + [d for d in order if d != h]
    t0 = time.monotonic()
    deadline = t0 + PLACEMENT_SEARCH_S
    tried = 0
    say(
        f"  searching for a wheelchair placement where the open plans: {len(order)} moves (up to "
        f"{max(PLACEMENT_SEARCH_MOVES_M) * 100:.0f} cm{', the sim advice first' if hint.any() else ''}), "
        f"each a full plan, stopping at the first that plans -- cancel the goal to stop it"
    )
    for d in order:
        if time.monotonic() - t0 > PLACEMENT_SEARCH_S:
            break
        tried += 1
        t1 = time.monotonic()
        try:
            with quiet():
                previous_deadline = planner.deadline
                planner.deadline = deadline
                try:
                    plan_open(planner, _shifted_geom(geom, d), q0, swing, make_after(), adaptive_pull=adaptive_pull)
                finally:
                    planner.deadline = previous_deadline
        except (
            Infeasible,
            StagingRequired,
            pl.StartNotClear,
        ) as e:  # doesn't plan there (the fridge moved onto the arm's start: neither)
            say(f"    handle moved {np.round(d, 2).tolist()} m: no ({_short_why(e)}, {time.monotonic() - t1:.0f} s)")
            continue
        except Refused as e:  # something that no placement changes (the planner, ...): stop searching
            return f"\n  (the placement search stopped: {_short_why(e)})"
        moves = " and ".join(chair_moves(*d))
        return (
            f"\n  -> checked by planning it: the open plans with the fridge shifted {np.round(d, 2).tolist()} m "
            f"-- {moves}, then dry-run it again ({tried} moves tried, {time.monotonic() - t0:.0f} s)"
        )
    if tried < len(order):
        return (
            f"\n  -> checked by planning it: none of the first {tried} wheelchair moves made the open plan; the "
            f"other {len(order) - tried} were not tried in {PLACEMENT_SEARCH_S:.0f} s"
        )
    return (
        f"\n  -> checked by planning it: no wheelchair move of up to {max(PLACEMENT_SEARCH_MOVES_M) * 100:.0f} cm "
        f"made the open plan ({tried} tried, {time.monotonic() - t0:.0f} s) -- within that, the refusal is not "
        "about where the wheelchair is"
    )


def _short_why(e):
    """A refusal's reason, short: what failed most."""
    why = str(e).split(" -- ")[0]
    return why if len(why) <= 110 else why[:107] + "..."


def why_not(tried):
    """The failed attempts ("<grasp>: <leg>: <why>") summed up by where and why they failed, most first."""
    import collections
    import re

    kinds = collections.Counter()
    for t in tried:
        parts = t.split(": ", 2)
        leg, why = (parts[1], parts[2]) if len(parts) == 3 else (t, "")
        leg = re.sub(r"\(.*", "", leg).strip()
        status = re.search(r"MotionGenStatus\.(\w+)", why)
        why = status.group(1) if status else re.sub(r"[-+]?\d+(\.\d+)?", "#", why)[:70]
        kinds[f"{leg}: {why}"] += 1
    return "; ".join(f"{n}x {k}" for k, n in kinds.most_common(4))


LAST_PLAN = None  # the last plan_open's plan (+ geom), e.g. to save a dry run's door


def detect(ai):
    """door_common's detection and the measured hinge: the geometry dict plan_open takes, and the placement advice."""
    st = ai.get_state()
    q0 = np.asarray(st["position"], dtype=float)
    shared.FRAME_ARM = ai
    h_raw, orient, n, door_z, door_mid, hinge_seen = shared.detect_handle()
    if shared.PLACEMENT_NOTE is not None:
        say(shared.PLACEMENT_NOTE(h_raw, n, q0))
    try:
        hinge, h, grasp, hinge_fields = shared.FIND_HINGE(h_raw, orient, n, hinge_seen)
    except Refused as e:
        raise Refused(f"{e}{placement_secondary(h_raw, n)}") from e
    geom = dict(
        h_raw=h_raw,
        orient=orient,
        n=n,
        hinge=hinge,
        hinge_seen=hinge_seen,
        door_z=door_z,
        door_mid=door_mid,
        hinge_fields=hinge_fields,
    )
    return geom, q0


def _close_gripper(ai):
    ai.gripper(1.0, force=F.GRASP_FORCE)


def execute_open(ai, planner, plan):
    """Approach; the swing under joint impedance, the gripper closing on the bar as it starts (GRIP_DURING_SWING), to a
    stop at exactly the angle asked for; then the gripper opens there, the hand still (push_open._release). What
    follows is planned again from where that leaves the arm. A swing the driver ran to its end that left the door short
    of its angle (at least SWING_REPLAN_MIN_DEG, the hinge fitted from its arc) lets go too, and what follows is then
    planned only against the door as measured (plan["after"].replan). Returns True when the hand let go; False when
    it stopped holding the handle; raises Refused."""
    cand, model = plan["cand"], plan["model"]
    st = ai.get_state()
    if float(st["gripper_pos"]) > 0.2:
        say("gripper is closed -- opening it first")
        ai.gripper(0.0)
        if float(ai.get_state()["gripper_pos"]) > 0.2:
            raise Refused("Gripper still closed after opening -- refusing to grasp.")
    if not planner.run(plan["approach"], "approach"):
        raise Refused("Did not reach the grasp -- ABORT before closing the gripper.")
    e = float(np.linalg.norm(np.asarray(ai.get_state()["ee_pos"][:3]) - cand["pos"]))
    say(f"grasp: tracking {e * 100:.1f} cm")
    if e > GRASP_TRACK_ABORT_M:
        raise Refused(f"Tracking {e * 100:.1f} cm at the grasp -- ABORT, gripper untouched.")
    # the closed door, for the close (door_mid written even when None, so a stale one can't linger)
    dc.DOOR_FILE.unlink(missing_ok=True)
    relocalize.ref_file().unlink(missing_ok=True)  # the close's reference was of the last door file
    save_door_geometry(**plan["door"], door_mid=plan.get("door_mid"))
    if not F.GRIP_DURING_SWING:
        _close_gripper(ai)
        g = float(ai.get_state()["gripper_pos"])
        say(f"gripper after close: {g:.4f} (can't confirm a grasp on this rig -- watch it)")
        if g < 0.2:
            raise Refused("Gripper is OPEN -- nothing grasped. Refusing to swing.")

    # the swing (and what follows it) from where the arm really is, if that is off the plan: a new search (the time the
    # detection and the approach took doesn't count against it)
    planner.search()
    st = ai.get_state()
    sw = plan["swing"]
    off = float(np.degrees(np.max(np.abs(pl._short_delta(sw.q0, st["position"])))))
    if off > REPLAN_DEG:
        say(f"  the grasp joints are {off:.1f} deg off the plan's -- planning the swing again from them")
        real = dict(cand, pos=np.asarray(st["ee_pos"][:3]), quat=np.asarray(st["ee_pos"][3:7]))
        sw, why = planner.motion(
            swing_goals(model, real, plan["swing_deg"], plan.get("lag", 0.0)),
            st["position"],
            start_pose=(real["pos"], real["quat"]),
        )
        if sw is None:
            raise Refused(f"swing from the real grasp: {why} -- stopping, the hand on the handle (door closed).")
        # the swing changed, so what follows it is checked again from the new swing's end
        after = plan["after"]
        deg_end = plan["swing_deg"]
        ok, msg = after.plan(planner, opened(plan["door"], deg_end), sw.q_end, (sw.goals[-1].pos, sw.goals[-1].quat))
        say(f"after-swing re-check from the real grasp: {msg}")
        if not ok:
            raise Refused(
                f"{msg} from the real grasp -- stopping BEFORE the swing, the hand on the handle (door closed)."
            )
    else:
        # the grasp is on the plan, so the swing is the planned one and what follows it was checked up front from the
        # same swing end: checking it again would only repeat that (it held the arm still ~6 s at the handle, 11-12)
        say(f"  the grasp is on the plan ({off:.1f} deg off): the swing and what follows it as planned up front")
    save_door_geometry(**plan["hinge_fields"])
    planner.join(sw, model.cabinet())
    events = [(0.0, partial(_close_gripper, ai))] if F.GRIP_DURING_SWING else []
    say(
        f"swing {plan['swing_deg']} deg"
        + (", the gripper closing as it starts" if F.GRIP_DURING_SWING else "")
        + f", to a stop there ({float(sw.joined.t[-1]):.1f} s); then the gripper opens"
    )
    # the door, not the plan, sets where the hand ends (F.SWING_END_TOL_M): judged by the angle it reached
    with ArcTrace(ai) as arc:
        ok = planner.run(
            sw,
            "swing",
            impedance={**F.PULL_GAINS, "path_tol": F.PULL_PATH_TOL_RAD},
            events=events,
            tol=F.SWING_END_TOL_M,
        )
    finished = planner.finished  # the driver ran the whole swing (no stop), wherever the hand ended
    final = ai.get_state()
    fitted = record_swing_hinge(arc.xy)  # the real hinge, for the angle below and all after (the push-open, the close)
    record_hand_angle(final["ee_pos"])  # the hand still holds the handle: it says where the door is
    deg = float(load_door()["door_open_deg"])
    if ok and deg < plan["swing_deg"] - F.SWING_ANGLE_TOL_DEG:
        say(f"  the door is ~{deg:.0f} deg open, over {F.SWING_ANGLE_TOL_DEG:.0f} short of {plan['swing_deg']}")
        ok = False
    # ran to its end but short: what follows was planned for a door at swing_deg -- let go and plan it again against
    # the door as measured (F.SWING_REPLAN_MIN_DEG), never run the old plan
    replan = not ok and finished and fitted and deg >= F.SWING_REPLAN_MIN_DEG and plan.get("after") is not None
    say(
        f"SWING {'DONE' if ok else 'SHORT' if replan else 'STOPPED SHORT'}: door ~{deg:.1f} deg, "
        f"EE {np.round(final['ee_pos'][:3], 4)}, gripper {float(final['gripper_pos']):.2f}"
    )
    if replan:
        say(
            f"  the swing ran to its end with the door at ~{deg:.0f} deg of {plan['swing_deg']}: letting go and "
            "planning what follows again, against the door where it is (the hinge from the swing's arc)"
        )
        plan["after"].replan = True
    elif not ok:
        return False
    P._release(ai)  # stopped at the angle: let go, the hand still (Refused if the gripper doesn't open)
    say(f"released: gripper {float(ai.get_state()['gripper_pos']):.2f}")
    return True


# --- what follows a full pull (push_open false) ---------------------------------------------------------------------


class _AfterPull:
    """What follows a full pull: plan from the hand on the handle, the gripper open (`plan`), run it re-planned from
    where the pull ended (`execute`)."""

    title = ""

    def __init__(self):
        self.motion = None

    def goals(self, model, deg, hold_pose, via):
        """The goals from `hold_pose` (on the handle); None: no such route (`via`)."""
        raise NotImplementedError

    def plan(self, planner, door, q_hold, hold_pose, from_arm=False):
        model = FridgeModel(door)
        deg = float(door["door_open_deg"])
        why = ""
        for via in (False, True):
            goals = self.goals(model, deg, hold_pose, via)
            if goals is None:
                break
            self.motion, why = planner.motion(goals, q_hold, start_pose=hold_pose, from_arm=from_arm)
            if self.motion is not None:
                return True, f"{self.title} OK{' (via a point past the free edge)' if via else ''}"
        return False, f"{self.title}: {why}"

    def dry_run(self, drive_chair):
        pass

    def execute(self, ai, planner, drive_chair=None):
        say(f"\n===== {self.title} =====")
        door = load_door()
        say(f"  {P.after_release(self, planner, ai, door)}")
        planner.join(self.motion, FridgeModel(door).cabinet())
        ok = planner.run(self.motion, self.title, impedance=P.AFTER_RELEASE)
        say(f"{self.title.upper()} {'DONE' if ok else 'STOPPED SHORT'}")
        return ok


class ReleaseAround(_AfterPull):
    """Back off the handle (the gripper open), then round the open door to its inner side, beside it, facing the
    fridge, upright. cuRobo routes round the free edge; only if it finds no way is it given one point past it."""

    title = "back off, round the door"

    def goals(self, model, deg, hold_pose, via):
        df = model.df
        out = back_off_goals(model, deg, hold_pose)
        pos, quat = out[-1].pos, out[-1].quat
        z = float(hold_pose[0][2])
        if via:
            d = df._rot(deg).T @ (pos[:2] - df.hinge[:2])
            v_now = float(d @ df.n) - df.mid
            out.append(
                pl.free(
                    df.pos(df.length + F.AROUND_EDGE_CLEAR_M[0], max(v_now, F.AROUND_OUT_M), deg, z),
                    quat,
                    "past the free edge",
                    model.full(deg),
                    blend=0.06,
                    world_label="the open door",
                )
            )
        out.append(
            pl.free(
                df.pos(F.AROUND_END_M[0], -F.AROUND_END_M[1], deg, z),
                P.facing_fridge(df, 1.0),
                "beside the door, inner side, upright",
                model.full(deg),
                blend=0.0,
                world_label="the open door",
            )
        )
        return out


class ReleaseBackOff(_AfterPull):
    """`around` False: back straight off the handle (the gripper open), nothing else (testing the grasp + pull
    alone)."""

    title = "back off the handle"

    def goals(self, model, deg, hold_pose, via):
        return None if via else back_off_goals(model, deg, hold_pose)


def release_back_off(ai, planner, execute):
    """release_only: let go and back straight off the handle from wherever the hand is (a run that stopped holding
    it), against the door where the hand says it is."""
    say("\n===== release, back off the handle =====")
    door = load_door()
    st = ai.get_state()
    door = opened(door, abs(hand_angle(door, st["ee_pos"])))
    model = FridgeModel(door)
    planner.search()
    pose = (np.asarray(st["ee_pos"][:3]), np.asarray(st["ee_pos"][3:7]))
    motion, why = planner.motion(
        back_off_goals(model, float(door["door_open_deg"]), pose), st["position"], start_pose=pose, from_arm=True
    )
    if motion is None:
        raise Refused(f"back-off: {why} -- NOT releasing, arm untouched.")
    if not execute:
        say("DRY RUN (release + back-off) -- nothing commanded.")
        return True
    planner.join(motion, model.cabinet())
    P._release(ai)
    return planner.run(motion, "back off the handle")


def save_planned_door(plan, push, after):
    """Write DOOR_FILE as the open just planned would leave it (a dry run's save_door): the closed door, the hinge, the
    door model, and the door open to PUSH_OPEN_TO_DEG (`push`; where it stands after the wheelchair move the push-open
    planned) or the pull's swing. Replaces the file."""
    door = opened(plan["door"], plan["swing_deg"])
    door = P.planned_door_after(door, push, after if push else None)
    door.update(plan["hinge_fields"])
    dc.DOOR_FILE.unlink(missing_ok=True)
    save_door_geometry(**door, saved_by="dry-run open (save_door): the door as the planned open would leave it")
    say(
        f"door file {dc.DOOR_FILE}: saved as the planned open would leave it -- door open {door['door_open_deg']:.0f} "
        f"deg (sense {int(door['open_sign']):+d}), hinge {np.round(door['hinge'], 3)}. Open the door to that by hand "
        "before testing the close."
    )


def run(
    ai,
    planner,
    execute,
    swing_deg=None,
    push=True,
    release_only=False,
    speed=1.0,
    log_args=None,
    grasp_offset=0.0,
    save_door=False,
    around=True,
    chair_moved=False,
    chair_back_dir=F.CHAIR_BACK_DIR,
    drive_chair=None,
    base_moves=True,
    search_alternate_placements=False,
    staging=None,
):
    """The open as the node runs it, logged to RUNS_DIR/<time>_open. `planner`: this action's planning.Planner.
    Pull to `swing_deg` when specified, or SWING_DEG without `push`. The default side-grasp push-open tries
    shorter pulls before PUSH_OPEN_PULL_DEG. `grasp_offset`:
    grasp this far (m) above the middle of the bar (set_grasp_offset). `release_only`: let go and back off the handle,
    nothing else (when a run stopped holding it). `save_door`: a dry run that plans OK also writes DOOR_FILE as the
    planned open would leave it (save_planned_door). `around` (the full-pull path only, `push` false): False tests the
    pull alone -- after the swing just let go and back off the handle. With `push`, the open may stop after the pull and
    the back-off, asking for a wheelchair move (push_open.NEXT_STEP); `chair_moved` then does the rest: the push, from
    where the hand is (push_after_chair_move). `chair_back_dir`: which way moving the wheelchair back moves the fridge
    in the arm's frame. `drive_chair` (the goal's drive_base: base_drive.move_chair): the push-open's wheelchair move is
    driven by the base within this goal, not asked for. `base_moves` False: the open plans no wheelchair move at all
    (the push from where the chair stands, or it refuses), and the placement advice leaves out the cells that need
    one (constants CELLS_NEEDING_BASE_MOVE). Returns True when it finished (or the dry run passed); raises Refused."""
    global BASE_MOVES
    P.NEXT_STEP, P.NEXT_MOVE = "", 0.0
    BASE_MOVES = bool(base_moves)

    def task():
        global LAST_PLAN
        if release_only:
            return release_back_off(ai, planner, execute)
        if chair_moved:
            if not BASE_MOVES:
                raise Refused("chair_moved is the push after a wheelchair move, and base_moves false plans none.")
            if not push:
                raise Refused("chair_moved is for the push-open's push, after the open asked for a wheelchair move.")
            return push_after_chair_move(ai, planner, execute, chair_back_dir)
        set_grasp_offset(grasp_offset)
        swing = swing_deg or (F.PUSH_OPEN_PULL_DEG if push else F.SWING_DEG)
        max_pull = F.PUSH_OPEN_TO_DEG - F.PUSH_OPEN_MIN_PUSH_DEG
        if push and swing > max_pull:  # push_open would refuse after the pull, holding the handle
            raise Refused(
                f"swing_deg {swing} leaves the push-open too little to push (it pushes on to {F.PUSH_OPEN_TO_DEG}): at "
                f"most {max_pull} with push_open, or push_open false for a full pull."
            )
        if execute and dc.DOOR_FILE.exists():
            # this open moves the door: a push an earlier open left waiting for a wheelchair move no longer fits it
            save_door_geometry(push_after_chair_move=None, close_chair_back=None)
        say("\n===== OPEN: detect, plan the whole open (cuRobo) =====")

        def make_after():
            if push:
                return PushOpen(chair_back_dir, base_moves=BASE_MOVES)
            return ReleaseAround() if around else ReleaseBackOff()

        after = make_after()
        LAST_PLAN = None
        geom, q0 = detect(ai)
        h, n = geom["h_raw"], geom["n"]
        start = f"Start joints {np.round(np.degrees(q0), 1)}."
        adaptive_pull = push and swing_deg is None and F.GRASP == "side"
        try:
            plan = plan_open(
                planner,
                geom,
                q0,
                swing,
                after,
                staging=staging,
                adaptive_pull=adaptive_pull,
            )
        except Infeasible as e:  # not plannable from here: where to put the wheelchair is the answer
            _, hint = placement_moves(h, n)
            found = ""
            if search_alternate_placements and not execute:
                found = search_placement(planner, geom, q0, swing, make_after, hint, adaptive_pull=adaptive_pull)
            nomove = (
                " (base_moves false: no wheelchair move was planned; from this placement the push needs one)"
                if push and not BASE_MOVES
                else ""
            )
            message = f"{e}{nomove} -- refusing before any motion. {start}{placement_primary(h, n)}{found}"
            payload = e.payload
            if found:
                payload = {**payload, "checked_placement_advice": found.strip()}
            raise Infeasible(message, payload=payload) from e
        except StagingRequired as e:
            e.args = (f"{e} -- refusing before any motion. {start}{placement_secondary(h, n)}",)
            raise
        except Refused as e:  # something else (the arm's start, the planner, ...): placement only as an aside
            raise Refused(f"{e} -- refusing before any motion. {start}{placement_secondary(h, n)}") from e
        plan.update(after=after, hinge_fields=geom["hinge_fields"], door_mid=geom["door_mid"])
        LAST_PLAN = dict(plan, geom=geom, q0=q0)
        say(f"  planning: {planner.summary()}")
        record_proven(plan, geom, "plan")  # the next search tries this grasp first (proven.py)
        if not execute:
            say(f"\nDRY RUN -- would grasp, then swing {plan['swing_deg']} deg. Nothing commanded.")
            after.dry_run(drive_chair)
            if save_door:
                save_planned_door(plan, push, after)
            return True
        if not execute_open(ai, planner, plan):
            holding = float(ai.get_state()["gripper_pos"]) > 0.2
            say(
                "swing did not finish -- stopping "
                + (
                    "(still holding the handle)."
                    if holding
                    else "(the gripper is OPEN: the hand is off or loose on the "
                    "handle; the door may move -- check it before release_only or a new open)."
                )
            )
            return False
        record_proven(plan, geom, "arm")  # the swing finished on the arm: tried before ones that only planned
        return after.execute(ai, planner, drive_chair)

    ok = logged_run("open", ai, log_args or {}, speed, task)
    say(f"\nOPEN {'OK' if ok else 'STOPPED'}{'' if execute else ' (dry run)'}")
    return ok
