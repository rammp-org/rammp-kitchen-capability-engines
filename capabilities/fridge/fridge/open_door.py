"""Open the fridge: detect the handle, grasp it, swing the door open and let go while still moving, then push it on (or
go round it).

The node's `grasp` selects "auto" (the default), "side" (side_grasp.py) or "front" (fingers across the bar,
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
3. Execute the approach, confirm that the gripper closed on the bar, then swing under joint impedance. An empty
   close gets one wrist re-look and retry. The gripper opens and is checked again after the door stops. The arm
   plans what follows from its measured state and executes it under impedance.

Refusals raise door_common.common.Refused: the arm holds where it is.
"""

import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
from functools import partial
from itertools import chain
from threading import Event

import numpy as np
import rammp_arm
from door_common import common as dc
from door_common import open_door as shared
from door_common import timing
from door_common.common import Refused, quiet, save_door_geometry, say
from scipy.spatial.transform import Rotation as R

from . import constants as F
from . import planning as pl
from . import proven, relocalize
from . import push_open as P
from . import side_grasp as S
from .door_sweep import SweepBlocked, check_open
from .motion import load_door, logged_run, opened
from .push_open import PushOpen, back_off_goals, push_after_chair_move
from .world import FridgeModel, configure_handle

GRASP_TRACK_ABORT_M = 0.03  # the approach must end the tool this close to the grasp, or the gripper stays open
HOOK_ANGLE_DEG = 94.0
HOOK_CLOSURE = 0.82
HOOK_FORCE = 0.15
HOOK_BACK_OFF_M = 0.05
HOOK_PREFLIGHT_S = 12.0
HOOK_DEFAULT_SPEED = 0.7
HOOK_EARLY_PULL_SCALE = 0.90
HOOK_EARLY_PULL_SPEEDUP = 0.90 / 0.595
HOOK_SLOW_FROM_DEG = 60.0
HOOK_FALLBACK_SETTLE_MARGIN_DEG = 3.0
HOOK_LOAD_WINDOW_DEG = (83.0, 92.0)
HOOK_MIN_HINGE_LOAD_NM = 0.08
HOOK_MIN_LOAD_SAMPLES = 3
HOOK_EXIT_LIMIT_MARGIN_DEG = 9.0
HOOK_LAGS_DEG = (55.0, 60.0, 45.0)
REPLAN_DEG = 1.0  # after the grasp, the swing is planned again from the real joints if they are further than this off


DOOR_NORMALS = ("handle_model", "handle", "plane", "square")
# per method: two looks' door normals must agree within this (deg). handle_model: the side grasp's estimator calls two
# consecutive fits agreeing within 3 deg ok (11-06 still looks: within 0.7 deg typically, 2.9 at most); handle: on the
# RealSense frames, two looks within 5 deg put 90% of doors within 8.5 deg; plane: 11-04's looks were 31 deg apart
NORMAL_AGREE_DEG = {"handle_model": 3.0, "handle": 5.0, "plane": 10.0, "square": 10.0}


def _make_detector(handle_centre_to_face_m, door_normal="handle_model", hinge_direction=1):
    """The handle detector for one detect_handle. `door_normal` (DOOR_NORMALS): where each look's door normal comes
    from -- handle_model: surface depth beside the SAM 3 mask sets yaw, checked against the printed handle model;
    handle: the bar's front and its plate as
    parallel planes; plane: a door-plane fit over the whole image; square: the door taken as square to the arm."""
    if door_normal == "handle_model":
        from .door_normal import make_detector  # numpy / scipy; the ROS side (open3d) only when called

        return make_detector(handle_centre_to_face_m=handle_centre_to_face_m, hinge_direction=hinge_direction)
    from .perception import HandleDetector  # here, not at import: open3d, which the planning tests don't have

    return HandleDetector(
        handle_centre_to_face_m, door_normal == "square", door_normal == "handle", hinge_direction=hinge_direction
    )


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


EARLY_EMPTY_ATTEMPTS = 3


def configure(
    data_dir,
    mount_tilt_deg=0,
    door_normal="handle_model",
    grasp="auto",
    *,
    hinge_side="left",
    early_empty_view_switch=False,
    **measured,
):
    """Everything the detection and the door model need from the fridge: call once at startup. `mount_tilt_deg`: a key
    of constants.MOUNTS. `door_normal`: how each look measures the door's direction (_make_detector, DOOR_NORMALS;
    handle_model and handle need SAM 3's mask). `grasp`: constants.GRASPS. `measured`: constants.MEASURED's names."""
    if door_normal not in DOOR_NORMALS:
        raise ValueError(f"door_normal must be one of {', '.join(DOOR_NORMALS)}, not {door_normal!r}")
    if hinge_side not in ("left", "right"):
        raise ValueError("hinge_side must be left or right when facing the fridge")
    hinge_direction = 1 if hinge_side == "left" else -1
    global _HANDLE_LENGTH, _DOOR_Z_FROM_HANDLE
    m = {**F.MEASURED, **measured}
    _HANDLE_LENGTH = float(m["handle_length_m"])
    configure_handle(_HANDLE_LENGTH)
    _DOOR_Z_FROM_HANDLE = (-m["door_below_handle_m"], m["door_above_handle_m"])
    F.configure(data_dir, mount_tilt_deg, grasp)
    mount = F.MOUNTS[mount_tilt_deg]
    # the hinge and the door's face from the handle (where the fingers close), as they are measured. The offset in the
    # door's frame: into it, along it, up. A wrong one drags the door or the fridge during the swing
    hinge_offset = np.array(
        [
            m["handle_centre_to_face_m"] + m["pin_behind_face_m"],
            hinge_direction * m["hinge_along_door_m"],
            0.0,
        ]
    )
    shared.configure(
        MAKE_DETECTOR=partial(_make_detector, m["handle_centre_to_face_m"], door_normal, hinge_direction),
        FIND_HINGE=shared.hinge_measured(hinge_offset, from_handle=True),
        GRIP_EXT=shared.GRASP_PADS_PAST_TOOL - F.HANDLE_BAR_CENTER_DEPTH_M,
        VERTICAL_CORR=F.VERTICAL_CORR,
        GRASP_X_ADJUST=0.0,
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
            body_width=m["body_width_m"],
            body_height=m["body_height_m"],
            support_height=m["support_height_m"],
            support_floor_z=m["support_floor_z_m"],
        ),
        DOOR_FACE_BEHIND_HANDLE=m["handle_centre_to_face_m"],
        DOOR_Z_FROM_HANDLE=_DOOR_Z_FROM_HANDLE,  # the camera sees only part of it
        KEEP_ARM_IN_FRAMES=True,
        EARLY_EMPTY_ATTEMPTS=(
            EARLY_EMPTY_ATTEMPTS if early_empty_view_switch and os.getenv("FRIDGE_LEGACY_LOOK_RETRIES") != "1" else None
        ),
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
        grasp=("side" if cand.get("side") else "front") if cand is not None else F.GRASP,
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


class HookLoadTrace(ArcTrace):
    """Measure the arm wrench's resisting moment about the hinge during the last moving part of the pull."""

    def __init__(self, ai, door):
        super().__init__(ai)
        self.door = door
        self.hinge = np.asarray(door["hinge"], float)
        self.sign = door["open_sign"]
        self.samples = []

    def sample(self):
        st = self.ai.get_state()
        pose = st["ee_pos"]
        angle = self.sign * hand_angle(self.door, pose)
        if not HOOK_LOAD_WINDOW_DEG[0] <= angle <= HOOK_LOAD_WINDOW_DEG[1]:
            return
        wrench = self.ai.ee_state()
        if wrench.frame_id != "arm_mount":
            return
        # The driver reports the environment's external wrench at the flange in arm_mount.
        radius = np.asarray(wrench.position) - self.hinge
        moment = self.sign * (np.cross(radius, wrench.force)[2] + wrench.torque[2])
        if np.isfinite(moment):
            self.samples.append(float(moment))

    def _sample(self):
        while not self._stop.wait(1.0 / F.HINGE_FIT_HZ):
            try:
                self.sample()
            except Exception:  # missing or stale feedback does not count as a loaded sample
                pass

    def retained(self):
        load = -float(np.median(self.samples)) if self.samples else float("nan")
        say(f"  hook late pull: {load:.3f} N m resisting hinge moment from {len(self.samples)} samples")
        return len(self.samples) >= HOOK_MIN_LOAD_SAMPLES and load >= HOOK_MIN_HINGE_LOAD_NM


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


def grasp_candidates(geom, grasp=None):
    """The grasps tried, in order: each tilt (+ nose-down, - nose-up), the hand up then upside down (fingers across the
    bar both ways). Dicts {pitch, flip, h, pos, quat}. A tilt whose grasp door_common refuses (beyond MAX_REACH) is
    skipped. The side grasp (F.GRASP "side"): side_grasp.candidates."""
    if (grasp or F.GRASP) in ("auto", "side"):
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
            pos = np.asarray(g.position, float)
            if geom.get("grasp_retry"):
                # Try a slightly shallower bite after the re-look, within the printed bar.
                pos = pos + F.GRASP_RETRY_OUT_M * np.asarray(geom["n"], float)
            out.append(dict(pitch=pitch, flip=flip, h=np.asarray(h, float), pos=pos, quat=quat))
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
    """List side hand lags with the previously successful lag first."""
    if not cand.get("side"):
        return [0.0]
    return proven.first((cand.get("proven") or {}).get("lag"), F.SIDE_HAND_LAGS_DEG)


def proven_file():
    """The record of the grasps that planned and ran (proven.py), next to the door file."""
    return dc.DOOR_FILE.parent / proven.FILE_NAME


@timing.timed("debug_proven")
def record_proven(plan, geom, source):
    """Record `plan`'s grasp as one that worked (`source`: "plan" or "arm"), for the next search to try first."""
    err = proven.record(
        proven_file(), proven.make_record(plan["door"]["grasp"], plan, geom["h_raw"], geom["n"], source)
    )
    if err:
        say(f"  (could not record the grasp that worked: {err})")


def hand_world(model, deg):
    """The door open `deg` with a window round the hand on the handle (and the handle) left out: onto or off it. The
    side grasp's: side_grasp.hold_world."""
    if model.door.get("grasp") == "side":
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
    if cand.get("side") or cand.get("hook"):
        poses = S.swing_poses(cand, df, swing, lag, degs)
    else:
        poses = [swung(cand["pos"], cand["quat"], df.hinge, df.sign, d) for d in degs]
    # the door turns with the hand, so no one world can hold it: the swing is checked against the cabinet (with the
    # planner's chair layers and the arm itself), as the old planner's pull was
    goal = pl.through(
        poses,
        f"swing to {swing:.0f} deg",
        model.cabinet(),
        world_label="the cabinet (the door turns with the hand)",
        segment_worlds=S.swing_worlds(model, degs) if cand.get("side") or cand.get("hook") else None,
    )
    if cand.get("hook"):
        # Speed up the middle of the stroke, keeping the original cap as the
        # door approaches the late-pull retention window.
        contact_scale = rammp_arm.JOINT_SPEED * rammp_arm.contact_speed_scale() / pl.LOCAL_JOINT_SPEED
        goal.speed = min(HOOK_EARLY_PULL_SCALE, contact_scale * HOOK_EARLY_PULL_SPEEDUP)
        goal.caps = [min(1.0, contact_scale / goal.speed) if deg >= HOOK_SLOW_FROM_DEG else 1.0 for deg in degs]
    return [goal]


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
OPEN_POLICY_S = 20.0
FRONT_SEARCH_S = 12.0
SIDE_FIXED_SEARCH_S = 7.0


def base_rotation_hint(normal):
    yaw = float(np.degrees(np.arctan2(-normal[1], -normal[0])))
    correction = yaw - float(np.clip(yaw, *F.PLACEMENT_YAW_DEG))
    if correction:
        direction = "left" if correction > 0 else "right"
        return {"base_rotation_deg": correction, "suggestion": f"Rotate base by {abs(correction):.1f}° {direction}."}
    return {}


def plan_open(planner, geom, q0, swing, after, *, staging=None, adaptive_pull=False, progress=None, prediction=None):
    """Plan the live open with one deadline shared by all strategies."""
    policy = F.GRASP == "auto" and adaptive_pull and isinstance(after, PushOpen)
    previous_deadline = planner.deadline
    start = time.monotonic()
    deadline = start + (OPEN_POLICY_S if policy else OPEN_PREFLIGHT_S)
    if previous_deadline is not None:
        deadline = min(deadline, previous_deadline)
    strategies = (
        (("front", True, FRONT_SEARCH_S), ("side", False, SIDE_FIXED_SEARCH_S), ("side", True, OPEN_POLICY_S))
        if policy
        else (("side" if F.GRASP == "auto" else F.GRASP, adaptive_pull, OPEN_PREFLIGHT_S),)
    )
    if prediction is not None:
        strategies = tuple(sorted(strategies, key=lambda strategy: strategy[0] != prediction["grasp"]))
    diagnostics = []
    failure = None
    sweep_failure = None
    try:
        for grasp, adaptive, limit in strategies:
            now = time.monotonic()
            if now >= deadline:
                break
            planner.deadline = min(now + limit, deadline)
            if progress is not None:
                detail = "adaptive pull" if adaptive else f"{swing:g}° pull"
                progress(min(0.99, (now - start) / (deadline - start)), f"Planning {grasp} grasp, {detail}")
            try:
                result = _plan_open(
                    planner,
                    geom,
                    np.asarray(q0),
                    swing,
                    after,
                    diagnostics,
                    staging,
                    adaptive,
                    grasp,
                    prediction=prediction,
                )
                planner.check_deadline()
                return result
            except SweepBlocked as e:
                sweep_failure = e
            except (Infeasible, pl.OverBudget) as e:
                failure = e
        if failure is not None:
            raise failure
        if sweep_failure is not None:
            raise sweep_failure
        raise pl.OverBudget("open preflight deadline reached")
    except SweepBlocked as e:
        raise Infeasible(str(e), payload=e.payload) from e
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


def _plan_open(planner, geom, q0, swing, after, diagnostics, staging, adaptive_pull, grasp, *, prediction=None):
    """Plan the whole open from joints `q0` (see the module): every grasp with cuRobo's own arm configuration first,
    then every grasp from the other configurations (seeds). Returns the plan dict; raises Refused (with why)."""
    planner.search()
    cands = grasp_candidates(geom, grasp)
    if not cands:
        raise Infeasible("no grasp: every tilt's grasp is out of reach")
    # the door model is the same for every grasp (the face is placed from the handle); the file keeps the chosen one's
    model = FridgeModel(closed_door(geom, cands[0]["h"], cands[0].get("ref", cands[0]["pos"]), cands[0]))
    push = after if isinstance(after, PushOpen) else None
    adaptive_pull = adaptive_pull and push is not None
    angles = (swing,)
    if adaptive_pull:
        # The gate uses the first angle to identify the full pull that permits a base move.
        angles = (
            P.pull_angles(swing)
            if grasp == "side"
            else (swing,) + tuple(angle for angle in (45, 50, 55) if angle < swing)
        )
    pull_angles = check_open(model, planner.world, angles, push)
    collision = planner.preflight_start(
        model.full(0.0),
        q0,
        np.radians(F.START_JOINTS_DEG) if staging is None else staging,
        grasp_retry=bool(geom.get("grasp_retry")),
    )
    if collision:
        raise StagingRequired(collision)
    # the grasps that worked before first (proven.py), each with its line-up and lag first; the rest in their order
    prefs = proven.preferred(proven.load(proven_file()) + list(F.PROVEN_SEEDS), grasp, geom["h_raw"], geom["n"])
    if prediction is not None and prediction["grasp"] == grasp:
        key = (round(prediction["pitch"]), prediction["flip"])
        prefs = [(key, prediction), *(item for item in prefs if item[0] != key)]
    cands = proven.order(cands, prefs)
    first_tried = [c["proven"] for c in cands if c.get("proven")]
    if first_tried:
        say(f"  tried first (worked before): {'; '.join(proven.describe(r) for r in first_tried)}")
    tried = []

    def fail(name, why, detail=None):
        item = dict(detail or {}, candidate=name)
        item.setdefault("stage", why.partition(":")[0])
        item.setdefault("reason", why)
        diagnostics.append(item)
        tried.append(f"{name}: {why}")
        say("    candidate rejected " + json.dumps(item))

    def side_pull_options(cand, insert_joints, name):
        preferred_lag = hand_lags(cand)[0] if (cand.get("proven") or {}).get("lag") is not None else None
        for angle in pull_angles:
            ranked = []
            for lag in F.SIDE_HAND_LAGS_DEG:
                why, margin = side_sweep_score(planner, model, cand, angle, lag, insert_joints)
                if why:
                    fail(f"{name}, {angle:g}°, lag {lag:g}°", why, {"previous_limit_margin_deg": margin})
                else:
                    ranked.append((lag == preferred_lag, margin, lag))
            for _, _, lag in sorted(ranked, reverse=True):
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
            options = iter((angle, 0.0) for angle in sorted(pull_angles))
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
        swung_door = opened(door, angle)
        args = (planner, swung_door, sw.q_end, (sw.goals[-1].pos, sw.goals[-1].quat))
        # The normal policy keeps the established pull for a base move. Experimental docking can
        # opt into checked translations after short pulls too.
        ok, msg = after.plan(*args, in_place=angle < swing) if adaptive_pull else after.plan(*args)
        if not ok:
            fail(found["name"], f"after swing: {msg}")
            return None
        phases = [("approach", found["approach"], model.cabinet())]
        if push is not None:
            phases.extend(
                (
                    ("pull", sw, model.cabinet()),
                    ("push" if not push.chair_move else "release", push.motion, FridgeModel(swung_door).cabinet()),
                )
            )
            if push.chair_move:
                moved = FridgeModel(P.shifted(swung_door, push.chair_move, push.chair_dir))
                phases.append(("push after chair move", push.push, moved.cabinet()))
        previous = q0
        for label, motion, obstacles in phases:
            try:
                samples = planner.join(motion, obstacles).joined.q
                if len(samples):
                    if np.max(np.abs(samples[0] - previous)) >= np.radians(pl.START_TOL_DEG):
                        raise Refused("joint discontinuity at the start of the joined trajectory")
                    previous = samples[-1]
            except (pl.OverBudget, pl.PlannerBusy):
                raise
            except Refused as e:
                fail(found["name"], f"{label} join: {e}")
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
                planner.deadline = min(deadline, t1 + OPEN_PREFLIGHT_S)
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


@timing.timed("detect")
def detect(ai, *, for_docking=False):
    """door_common's detection and the measured hinge: the geometry dict plan_open takes, and the placement advice."""
    st = ai.get_state()
    q0 = np.asarray(st["position"], dtype=float)
    shared.FRAME_ARM = ai
    h_raw, orient, n, door_z, door_mid, hinge_seen = shared.detect_handle()
    if shared.PLACEMENT_NOTE is not None:
        say(shared.PLACEMENT_NOTE(h_raw, n, q0))
    try:
        settings = {"check_reach": False} if for_docking else {}
        hinge, h, grasp, hinge_fields = shared.FIND_HINGE(h_raw, orient, n, hinge_seen, **settings)
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
    if for_docking:
        geom["closed_model"] = FridgeModel(closed_door(geom, h, grasp.position))
    return geom, q0


@timing.timed("wrist_relook")
def detect_with_wrist_relook(ai, planner, camera, scene_model):
    """Search for the vertical bar from two collision-checked viewpoints after an empty wrist look."""
    try:
        return detect(ai)
    except shared.NoHandleProposal as first:
        try:
            model = scene_model()
        except Refused as error:
            raise shared.NoHandleProposal(f"{first}; no scene geometry for a wrist re-look: {error}") from error
        frame = camera.grab()
        if frame is None or frame["base_T_cam"] is None:
            raise
        obstacles = model.full(0.0)
        normal = np.asarray(model.df.n)
        across = np.cross([0.0, 0.0, 1.0], normal)
        camera_pose = frame["base_T_cam"]
        original = ai.get_state()
        pos, quat = np.asarray(original["ee_pos"][:3]), original["ee_pos"][3:7]
        tool_rotation = R.from_quat(quat)
        # Translate 4 cm across the vertical bar and 3 cm outward, aiming back toward the
        # same point. The outward margin also gives the checked move more door clearance.
        distance = float(np.linalg.norm(model.door["closed_handle"] - camera_pose[:3, 3]))
        yaw = np.arctan2(0.04, distance + 0.03)
        for side in (1, -1):
            state = ai.get_state()
            target = pos + 0.03 * normal + side * 0.04 * across
            rotation = (R.from_rotvec(side * yaw * np.array([0.0, 0.0, 1.0])) * tool_rotation).as_quat()
            label = f"wrist handle re-look {side:+d}"
            goal = pl.free(target, rotation, label, obstacles, blend=0.0)
            planner.search()
            motion, why = planner.motion(
                [goal], state["position"], start_pose=(state["ee_pos"][:3], state["ee_pos"][3:7]), from_arm=True
            )
            if motion is None:
                say(f"  {label}: {why}")
                continue
            planner.join(motion, obstacles, label, speed=0.10)
            if not planner.run(motion, label):
                raise Refused(f"{label} did not reach the checked pose.") from first
            camera.require_frames_after_now()
            try:
                return detect(ai)
            except shared.NoHandleProposal:
                continue
        raise shared.NoHandleProposal(f"{first}; the checked wrist re-looks had no usable proposal.") from first


@timing.timed("raised_relook")
def detect_after_arrival_relook(ai, planner, camera, obstacles):
    """Take raised wrist views and return to the parked arm before any further drive."""
    original = ai.get_state()
    joints = np.asarray(original["position"], float)
    pos = np.asarray(original["ee_pos"][:3])
    quat = np.asarray(original["ee_pos"][3:7])
    moved = False
    try:
        for degrees in (12, -12):
            rotation = (R.from_euler("z", degrees, degrees=True) * R.from_quat(quat)).as_quat()
            label = f"raised wrist look {degrees:+d} degrees"
            target = pos + [-0.035, 0.0, 0.07]
            goal = pl.free(target, rotation, label, obstacles, blend=0.0)
            state = ai.get_state()
            planner.search()
            path, _ = planner.motion(
                [goal], state["position"], start_pose=(state["ee_pos"][:3], state["ee_pos"][3:7]), from_arm=True
            )
            if path is None:
                continue
            planner.join(path, obstacles, label, speed=0.10)
            if not planner.run(path, label):
                raise Refused(f"{label} did not reach its checked pose.")
            moved = True
            camera.require_frames_after_now()
            try:
                geom, _ = detect(ai, for_docking=True)
            except shared.NoHandleProposal:
                continue
            break
        else:
            raise shared.NoHandleProposal("The raised wrist views found no fridge handle.")
    except Refused:
        if moved:
            _park_after_arrival_relook(ai, planner, joints, pos, quat, obstacles)
        raise
    _park_after_arrival_relook(ai, planner, joints, pos, quat, obstacles)
    return geom, np.asarray(ai.joint_states().position, float)


def _park_after_arrival_relook(ai, planner, joints, pos, quat, obstacles):
    label = "return to the parked arm pose"
    state = ai.get_state()
    goal = pl.to_joints(joints, pos, quat, label, obstacles, blend=0.0)
    planner.search()
    path, why = planner.motion(
        [goal], state["position"], start_pose=(state["ee_pos"][:3], state["ee_pos"][3:7]), from_arm=True
    )
    if path is None:
        raise Refused(f"Could not {label}: {why}")
    planner.join(path, obstacles, label, speed=0.10)
    if not planner.run(path, label):
        raise Refused(f"Could not {label}.")


class GraspUnconfirmed(Refused):
    """The fingers did not stop at the expected handle width."""


@timing.timed("grasp_close")
def confirm_grasp(ai):
    if ai.gripper(1.0, force=F.GRASP_FORCE) is False:
        raise Refused("Gripper close did not complete. Check whether the hand is on the handle.")
    grip = ai.gripper_state() if hasattr(ai, "gripper_state") else None
    position = float(grip.position if grip is not None else ai.get_state()["gripper_pos"])
    gap = F.GRIPPER_STROKE_M * (1.0 - position)
    current = f", motor current {grip.current:.3f}" if grip is not None and np.isfinite(grip.current) else ""
    say(f"  gripper after close: {position:.3f}, estimated gap {gap * 1000:.1f} mm{current}")
    if grip is not None and not grip.present:
        raise Refused("Gripper feedback unavailable. Check whether the hand is on the handle.")
    if not F.GRASP_GAP_MIN_M <= gap <= F.GRASP_GAP_MAX_M:
        P._release(ai)
        if 0.0 <= gap < F.GRASP_GAP_MIN_M:
            raise GraspUnconfirmed(
                f"Grasp unconfirmed (empty close, estimated gap {gap * 1000:.1f} mm). Gripper opened."
            )
        raise Refused(
            f"Grasp unconfirmed (closure too wide or invalid, estimated gap {gap * 1000:.1f} mm). Gripper opened."
        )


@timing.timed("grasp_retry_view")
def retry_grasp_view(ai, planner, plan):
    """Retreat through the insertion corridor and return to the original viewing pose."""
    approach = plan["approach"]
    insertion = approach.goals[-1]
    lineup = approach.goals[-2]
    st = ai.get_state()
    planner.search()
    retreat_goal = pl.line(lineup.pos, lineup.quat, "back to the line-up", insertion.world, contact=False)
    view_pos, view_quat = planner.tool(approach.q0)
    view_goal = pl.to_joints(approach.q0, view_pos, view_quat, "back to the wrist look", lineup.world)
    retreat, why = planner.motion(
        [retreat_goal, view_goal],
        st["position"],
        start_pose=(st["ee_pos"][:3], st["ee_pos"][3:7]),
        from_arm=True,
    )
    if retreat is None:
        raise Refused(f"Grasp retry could not retreat: {why}")
    planner.join(retreat, plan["model"].cabinet())
    if not planner.run(retreat, "back to the wrist look"):
        raise Refused("Grasp retry could not reach the wrist look.")


def can_replan_swing(plan, measured_deg, end_error):
    """Allow the adaptive push to recover a small endpoint miss after a completed, fitted pull."""
    after = plan.get("after")
    if after is None:
        return False
    if measured_deg >= F.SWING_REPLAN_MIN_DEG:
        return True
    return (
        isinstance(after, PushOpen)
        and measured_deg >= plan["swing_deg"] - F.SWING_ANGLE_TOL_DEG
        and end_error <= F.SWING_RECOVERY_MAX_ERROR_M
    )


def execute_open(ai, planner, plan, *, fast=False):
    """Approach, confirm closure, swing under impedance, then release at rest."""
    cand, model = plan["cand"], plan["model"]
    st = ai.get_state()
    if float(st["gripper_pos"]) > 0.2:
        say("gripper is closed -- opening it first")
        P._release(ai)
    if not planner.run(plan["approach"], "approach"):
        raise Refused("Did not reach the grasp -- ABORT before closing the gripper.")
    e = float(np.linalg.norm(np.asarray(ai.get_state()["ee_pos"][:3]) - cand["pos"]))
    say(f"grasp: tracking {e * 100:.1f} cm")
    if e > GRASP_TRACK_ABORT_M:
        raise Refused(f"Tracking {e * 100:.1f} cm at the grasp -- ABORT, gripper untouched.")
    hook = cand.get("hook", False)
    if not hook:
        confirm_grasp(ai)
    dc.DOOR_FILE.unlink(missing_ok=True)
    relocalize.ref_file().unlink(missing_ok=True)
    save_door_geometry(**plan["door"], door_mid=plan.get("door_mid"))

    # the swing (and what follows it) from where the arm really is, if that is off the plan: a new search (the time the
    # detection and the approach took doesn't count against it)
    planner.search()
    st = ai.get_state()
    sw = plan["swing"]
    off = float(np.degrees(np.max(np.abs(pl._short_delta(sw.q0, st["position"])))))
    if off > REPLAN_DEG:
        if hook:
            raise GraspUnconfirmed("Hook pull start differs from preflight. The gripper is open.")
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
    # Reuse the checked preflight pull when the measured grasp agrees.
    if sw.joined is None:
        planner.join(sw, model.cabinet())
    if hook:
        if timing.timed("hook_close")(ai.gripper)(HOOK_CLOSURE, force=HOOK_FORCE) is False:
            raise Refused("Could not close to the hook position.")
        grip = ai.gripper_state() if hasattr(ai, "gripper_state") else None
        position = float(grip.position if grip is not None else ai.get_state()["gripper_pos"])
        hook_closure_at_grasp = position
        if (grip is not None and not grip.present) or not np.isfinite(position) or position <= 0.2:
            raise Refused("No valid closed gripper feedback after the hook command.")
    say(f"swing {plan['swing_deg']} deg, to a stop there ({float(sw.joined.t[-1]):.1f} s); then the gripper opens")
    # the door, not the plan, sets where the hand ends (F.SWING_END_TOL_M): judged by the angle it reached
    with HookLoadTrace(ai, plan["door"]) if hook else ArcTrace(ai) as arc:
        ok = planner.run(
            sw,
            "swing",
            impedance={**F.PULL_GAINS, "path_tol": F.PULL_PATH_TOL_RAD},
            tol=F.SWING_END_TOL_M,
            settle_for_release=True,
        )
    finished = planner.finished  # the driver ran the whole swing (no stop), wherever the hand ended
    final = ai.get_state()
    if hook:
        closure = float(final["gripper_pos"])
        say(f"  hook finger closure: {hook_closure_at_grasp:.3f} at grasp, {closure:.3f} after pull")
        if not np.isfinite(closure) or closure <= 0.2:
            say("  hook retention check: the fingers reopened during the pull")
            dc.DOOR_FILE.unlink(missing_ok=True)
            return False
        if not ok or not arc.retained():
            say("  hook pull could not verify retention; stopping before release")
            dc.DOOR_FILE.unlink(missing_ok=True)
            return False
        inferred = model.df.sign * hand_angle(plan["door"], final["ee_pos"])
        say(f"  hook angle from the tool and bar geometry: {inferred:.1f}°")
        if not 90.0 <= inferred <= HOOK_ANGLE_DEG + 2:
            dc.DOOR_FILE.unlink(missing_ok=True)
            return False
        save_door_geometry(door_open_deg=inferred, open_sign=model.df.sign)
        plan["after"].replan = abs(inferred - HOOK_ANGLE_DEG) > 1.0
        P._release(ai, early=fast)
        return True
    fitted = record_swing_hinge(arc.xy)  # the real hinge, for the angle below and all after (the push-open, the close)
    record_hand_angle(final["ee_pos"])  # the hand still holds the handle: it says where the door is
    deg = float(load_door()["door_open_deg"])
    if ok and deg < plan["swing_deg"] - F.SWING_ANGLE_TOL_DEG:
        say(f"  the door is ~{deg:.0f} deg open, over {F.SWING_ANGLE_TOL_DEG:.0f} short of {plan['swing_deg']}")
        ok = False
    # ran to its end but short: what follows was planned for a door at swing_deg -- let go and plan it again against
    # the door as measured (F.SWING_REPLAN_MIN_DEG), never run the old plan
    end_error = float(np.linalg.norm(np.asarray(final["ee_pos"][:3]) - sw.goals[-1].pos))
    replan = not ok and finished and fitted and can_replan_swing(plan, deg, end_error)
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
    P._release(ai, early=fast)  # held at the pull angle until the gripper clears the handle
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
        P.wait_until_still(ai)
        if isinstance(self, HookBackOff):
            margin = planner.limit_margin(self.motion.joined.q)
            if margin < HOOK_EXIT_LIMIT_MARGIN_DEG:
                raise Refused(f"Hook withdrawal: only {margin:.1f}° of joint clearance -- arm untouched.")
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


class HookBackOff(_AfterPull):
    """Clear the bar along the insertion axis on the exterior side."""

    title = "back off after hook pull"

    def goals(self, model, deg, hold_pose, via):
        if via:
            return None
        pos, quat = hold_pose
        pos = np.asarray(pos)
        hand = R.from_quat(quat)
        offset = hand.as_matrix()[:, 2] * HOOK_BACK_OFF_M
        end = pos - offset
        # Withdraw first, then ease J6 toward alignment with the arm reach.
        target = R.from_quat(S.unbend(end, quat))
        turn = (target * hand.inv()).as_rotvec()[2]
        orientation = (R.from_euler("z", np.clip(turn, *np.radians([-15, 15]))) * hand).as_quat()
        return [
            pl.through(
                [(pos - offset / 2, quat), (end, orientation)],
                self.title,
                S.hold_world(model, (max(0.0, deg - 2), deg + 2)),
            )
        ]


def _plan_hook(planner, plan, geom):
    """Try a stationary hook from the already checked approach."""
    model = plan["model"]
    original = plan["cand"]
    if original.get("side") or original["flip"]:
        return None
    cand = dict(original, bar=S.bar_point(original["h"], geom["n"]), hook=True)
    closed = closed_door(geom, cand["h"], cand["pos"], cand)
    try:
        check_open(model, planner.world, (HOOK_ANGLE_DEG + 2,), None)
        for lag in HOOK_LAGS_DEG:
            swing, why = planner.motion(
                swing_goals(model, cand, HOOK_ANGLE_DEG, lag),
                plan["approach"].q_end,
                start_pose=(cand["pos"], cand["quat"]),
            )
            if swing is None:
                say(f"  hook lag {lag:g}° declined: {why}")
                continue
            after = HookBackOff()
            door = opened(closed, HOOK_ANGLE_DEG)
            end = swing.goals[-1]
            ok, why = after.plan(planner, door, swing.q_end, (end.pos, end.quat))
            if not ok:
                say(f"  hook lag {lag:g}° declined: {why}")
                continue
            planner.join(after.motion, model.cabinet())
            margin = planner.limit_margin(after.motion.joined.q)
            if margin < HOOK_EXIT_LIMIT_MARGIN_DEG:
                say(f"  hook lag {lag:g}° declined: only {margin:.1f}° of joint clearance on withdrawal")
                continue
            planner.join(swing, model.cabinet())
            return dict(
                plan,
                door=closed,
                cand=cand,
                swing=swing,
                swing_deg=HOOK_ANGLE_DEG,
                lag=lag,
                name="stationary hook pull",
                after_msg="stationary hook pull and exterior release",
                after=after,
            )
    except Refused as e:
        say(f"  hook preflight declined: {e}")
    return None


def _hook_preflight(planner, geom, q0, staging, free_speed=None):
    """Plan the validated front approach before searching the push fallback."""
    if F.GRASP == "side":
        return None
    cand = next((g for g in grasp_candidates(geom, "front") if g["pitch"] == 25 and not g["flip"]), None)
    if cand is None:
        return None
    model = FridgeModel(closed_door(geom, cand["h"], cand["pos"], cand))
    planner.search()
    try:
        parked = np.radians(F.START_JOINTS_DEG) if staging is None else staging
        if planner.preflight_start(model.full(0.0), q0, parked):
            return None
        standoff = standoffs(cand)[0]
        approach, why = planner.motion(approach_goals(model, cand, standoff), q0, from_arm=True)
        if approach is None:
            say(f"  hook approach declined: {why}")
            return None
        planner.join(approach, model.cabinet())
        plan = _plan_hook(planner, dict(cand=cand, model=model, approach=approach), geom)
        if plan is not None and free_speed is not None:
            # Keep the preflight and fallback request order for refused hooks.
            # A declined retime leaves the original checked approach in place.
            try:
                planner.join(approach, model.cabinet(), speed=free_speed)
            except Refused as error:
                say(f"  hook approach retime declined: {error}")
        planner.check_deadline()
        return plan
    except Refused as e:
        say(f"  hook approach declined: {e}")
        return None


def _select_open_plan(planner, geom, q0, staging, hook, fallback, hook_free_speed=None):
    if not hook:
        return fallback()
    previous_deadline = planner.deadline
    deadline = time.monotonic() + HOOK_PREFLIGHT_S
    planner.deadline = deadline if previous_deadline is None else min(deadline, previous_deadline)
    try:
        plan = _hook_preflight(planner, geom, q0, staging, hook_free_speed)
    finally:
        planner.deadline = previous_deadline
    if plan is not None:
        say("  preflight selected the stationary hook pull")
        return plan
    say("  hook preflight declined; searching the fast push open")
    return fallback()


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
    door model, and the door at the planned push angle and dock (or the pull angle for a pull-only action).
    Replaces the file."""
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


def _predictive_dock(detect_at_dock, planner, make_after, swing, staging, adaptive_pull):
    """Use the scene to identify a candidate for wrist-confirmed replanning."""
    from capability_engine import ArmError

    from . import dock_ready

    stop = Event()
    future = None
    ready = Future()

    def checkpoint():
        if stop.is_set():
            raise Refused("The dock look is ready")
        planner.checkpoint()

    @timing.timed("speculative_preflight")
    def preview(geom, joints, delta, yaw, mount, route):
        inverse = R.from_rotvec([0, 0, -yaw])

        def point(value):
            return inverse.apply(np.asarray(value, float) + np.r_[mount - delta, 0]) - np.r_[mount, 0]

        h = point(geom["h_raw"])
        n = inverse.apply(geom["n"])
        orient = (inverse * R.from_quat(geom["orient"])).as_quat()
        mid = None if geom["door_mid"] is None else point(geom["door_mid"])
        seen = None if geom["hinge_seen"] is None else point(geom["hinge_seen"])
        try:
            hinge, _, _, fields = shared.FIND_HINGE(h, orient, n, seen)
            predicted = dict(
                geom, h_raw=h, n=n, orient=orient, door_mid=mid, hinge=hinge, hinge_seen=seen, hinge_fields=fields
            )
            speculative = pl.Planner(planner.arm, planner.world, planner.fk, checkpoint, planner.budget_s)
            if route and hasattr(planner.arm, "stream"):
                speculative.deadline = time.monotonic() + 1.5
                try:
                    motion = timing.timed("ready_arm_plan")(dock_ready.plan_ready)(
                        speculative, geom, predicted, joints, route, mount
                    )
                    if motion is not None:
                        ready.set_result(motion)
                except (Refused, ArmError, ValueError) as error:
                    say(f"  arm ready pose unavailable: {error}")
                finally:
                    speculative.deadline = None
            plan = plan_open(
                speculative, predicted, joints, swing, make_after(), staging=staging, adaptive_pull=adaptive_pull
            )
        except (Refused, ArmError, ValueError) as error:
            say(f"  scene preflight unavailable: {error}")
            return None
        return proven.make_record(plan["door"]["grasp"], plan, h, n, "plan")

    def on_route(*args):
        nonlocal future
        if future is None:
            future = pool.submit(copy_context().run, preview, *args)
            return lambda: ready.result() if ready.done() else None

    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            geom, joints = detect_at_dock(on_route)
        finally:
            stop.set()
    prediction = None if future is None else future.result()
    if prediction is not None and (
        np.linalg.norm(geom["h_raw"] - prediction["handle"]) > 0.03
        or np.dot(geom["n"], prediction["normal"]) < np.cos(np.radians(3))
    ):
        prediction = None
    return geom, joints, prediction


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
    progress=None,
    detect_at_dock=None,
    detect_at_start=None,
    detect_retry=None,
    allow_short_handoff=False,
    short_reentry=False,
    fast=False,
    predictive=False,
    hook=False,
):
    """The open as the node runs it, logged to RUNS_DIR/<time>_open. `planner`: this action's planning.Planner.
    Pull to `swing_deg` when specified, or SWING_DEG without `push`. The automatic push-open searches front then side
    grasps. `grasp_offset`:
    grasp this far (m) above the middle of the bar (set_grasp_offset). `release_only`: let go and back off the handle,
    nothing else (when a run stopped holding it). `save_door`: a dry run that plans OK also writes DOOR_FILE as the
    planned open would leave it (save_planned_door). `around` (the full-pull path only, `push` false): False tests the
    pull alone -- after the swing just let go and back off the handle. With `push`, the open may stop after the pull and
    the back-off, asking for a wheelchair move (push_open.NEXT_STEP); `chair_moved` then does the rest: the push, from
    where the hand is (push_after_chair_move). `chair_back_dir`: which way moving the wheelchair back moves the fridge
    in the arm's frame. `drive_chair` (the goal's drive_base: base_drive.move_chair): the push-open's wheelchair move is
    driven by the base within this goal, not asked for. `base_moves` False: the open plans no wheelchair move at all
    (the push from where the chair stands, or it refuses), and the placement advice leaves out the cells that need
    one (constants CELLS_NEEDING_BASE_MOVE). `hook` tries a stationary partial-closure pull when the default opening
    angle is requested, with the normal checked push as a pre-motion fallback. Its free-space approach uses full
    planner speed at the default hook speed. Returns True when it finished
    (or the dry run passed); raises Refused."""
    global BASE_MOVES
    P.NEXT_STEP, P.NEXT_MOVE, P.FINAL_ANGLE_DEG = "", 0.0, None
    BASE_MOVES = bool(base_moves)

    if fast:
        planner.feedback_settle = True

    @timing.timed("open_task")
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
                return PushOpen(
                    chair_back_dir,
                    base_moves=BASE_MOVES,
                    allow_short_handoff=allow_short_handoff,
                    short_reentry=short_reentry,
                    settle_margin_deg=HOOK_FALLBACK_SETTLE_MARGIN_DEG if hook and swing_deg is None else 0.0,
                )
            return ReleaseAround() if around else ReleaseBackOff()

        after = make_after()
        LAST_PLAN = None
        adaptive_pull = push and swing_deg is None and F.GRASP in ("auto", "side")
        prediction = None
        if predictive:
            geom, q0, prediction = _predictive_dock(detect_at_dock, planner, make_after, swing, staging, adaptive_pull)
        else:
            detect_initial = detect_at_dock or detect_at_start or partial(detect, ai)
            with timing.span("initial_observation"):
                geom, q0 = detect_initial()
        h, n = geom["h_raw"], geom["n"]
        start = f"Start joints {np.round(np.degrees(q0), 1)}."
        try:
            plan = timing.timed("final_preflight")(_select_open_plan)(
                planner,
                geom,
                q0,
                staging,
                hook and push and swing_deg is None,
                lambda: plan_open(
                    planner,
                    geom,
                    q0,
                    swing,
                    after,
                    staging=staging,
                    adaptive_pull=adaptive_pull,
                    progress=progress,
                    prediction=prediction,
                ),
                hook_free_speed=min(1.0, speed / HOOK_DEFAULT_SPEED),
            )
            if plan["cand"].get("hook"):
                after = plan["after"]
        except Infeasible as e:  # not plannable from here: where to put the wheelchair is the answer
            if e.payload.get("reason") == "door_sweep_blocked":
                raise
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
        if not plan["cand"].get("hook"):
            record_proven(plan, geom, "plan")  # the next search tries this grasp first (proven.py)
        if not execute:
            say(f"\nDRY RUN -- would grasp, then swing {plan['swing_deg']} deg. Nothing commanded.")
            after.dry_run(drive_chair)
            if save_door:
                save_planned_door(plan, push and not plan["cand"].get("hook"), after)
            return True
        try:
            swung = timing.timed("approach_and_pull")(execute_open)(ai, planner, plan, fast=fast)
        except GraspUnconfirmed as first:
            say(f"  {first} Taking one new wrist look before trying again.")
            retry_grasp_view(ai, planner, plan)
            try:
                geom, q0 = detect(ai) if detect_retry is None else detect_retry()
            except Refused as e:
                raise GraspUnconfirmed(f"{first} The retry wrist look failed: {e}") from e
            geom = dict(geom, grasp_retry=True)
            after = make_after()
            try:
                plan = timing.timed("retry_preflight")(plan_open)(
                    planner, geom, q0, swing, after, staging=staging, adaptive_pull=adaptive_pull
                )
            except Refused as error:
                raise Refused(f"{first} Could not plan a new grasp after retreating: {error}") from error
            plan.update(after=after, hinge_fields=geom["hinge_fields"], door_mid=geom["door_mid"])
            LAST_PLAN = dict(plan, geom=geom, q0=q0)
            swung = timing.timed("approach_and_pull")(execute_open)(ai, planner, plan, fast=fast)
        if not swung:
            if plan["cand"].get("hook"):
                say("Hook stopped in place. Check the door position and grip before commanding motion.")
                return False
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
        if not plan["cand"].get("hook"):
            record_proven(plan, geom, "arm")  # the swing finished on the arm: tried before ones that only planned
        ok = timing.timed("post_pull")(after.execute)(ai, planner, drive_chair)
        if ok and plan["cand"].get("hook"):
            P.FINAL_ANGLE_DEG = float(load_door()["door_open_deg"])
        return ok

    ok = logged_run("open", ai, log_args or {}, speed, task, fast=fast)
    say(f"\nOPEN {'OK' if ok else 'STOPPED'}{'' if execute else ' (dry run)'}")
    return ok
