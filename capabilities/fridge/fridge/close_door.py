"""Close the fridge: door_common's push-close, with three changes for a fridge.

1. The wheelchair is moved back ~CHAIR_BACK_M first; the recorded door is shifted by that much for the plan.
2. No going over the door: back out in front of the open door's free edge, left behind it, then forward to the swing
   start.
3. The swing stops with the door at `close_to_deg` (it swings shut by itself from there): no torque push.

The gripper is never touched (it may hold a container) and the hand never tilts. An action can't prompt, so an execute
goal must say the chair has been moved (`chair_moved`); until then it refuses, arm untouched. With the goal's
`drive_base` the base drives the chair back instead (base_drive), after checking the arm stays clear of the open door
all through the move, and the close is planned against the door shifted by the move the base really made.
"""

import json

import numpy as np
from door_common import close_door as shared
from door_common import common as dc
from door_common.common import Refused, make_sim, quiet, say

from . import constants as F
from .motion import logged_run

# The goal's settings, and their defaults: F.CLOSE_SETTINGS holds what a close of this fridge doesn't vary.
GOAL_SETTINGS = {
    **{k: shared.CLOSE_DEFAULTS[k] for k in ("door_deg", "side_past_hinge", "push_z", "retreat_dist")},
    **{k: F.CLOSE_SETTINGS[k] for k in ("push_radius", "push_radius_end", "hold_offset_x")},
    "chair_back": F.CHAIR_BACK_M,  # how far the chair is moved back (m); 0: not moved
    "close_to_deg": F.CLOSE_TO_DEG,  # swing the door shut until it is open this far (deg)
    "chair_moved": False,  # the chair has been moved back: required to execute with chair_back > 0
}


def _push_close(ai, execute, settings, shift, close_to_deg):
    args = shared.close_args(execute, **{**F.CLOSE_SETTINGS, **settings})
    return shared.push_close(
        ai, args, door_shift=shift, behind=True, stop_deg=close_to_deg, max_tilt_deg=F.CLOSE_MAX_TILT_DEG
    )


def close_height(ai, door, settings, shift, close_to_deg):
    """The swing height for the close: `settings`' push_z if given (refused outside the door), else the first height
    within the door (CLOSE_Z_MARGIN_M in from its top and bottom, every CLOSE_Z_STEP_M, nearest the hand first) whose
    whole push_close plan passes every gate (planned quietly, nothing moves). Raises Refused."""
    lo, hi = door["door_z"][0] + F.CLOSE_Z_MARGIN_M, door["door_z"][1] - F.CLOSE_Z_MARGIN_M
    say(f"door z {door['door_z'][0]:.3f}-{door['door_z'][1]:.3f}; the swing may run at z {lo:.3f}-{hi:.3f}")
    push_z = settings.get("push_z")
    if push_z is not None:
        if not lo <= push_z <= hi:
            raise Refused(f"push_z {push_z:.3f} is not within the door.")
        return push_z
    z_now = float(ai.get_state()["ee_pos"][2])
    for z in sorted(np.arange(lo, hi + 1e-9, F.CLOSE_Z_STEP_M), key=lambda z: abs(z - z_now)):
        with quiet():
            ok = _push_close(ai, False, {**settings, "push_z": float(z)}, shift, close_to_deg)
        say(f"  z {z:.3f}: {'passes' if ok else 'fails a gate'}")
        if ok:
            return float(z)
    raise Refused("no height within the door gives a plan that passes every gate.")


def move_chair(chair_back):
    """What the person does to the wheelchair (`chair_back` m: + back, - forward), as the close's and open's messages
    say it."""
    if chair_back < 0:
        return f"Move the WHEELCHAIR straight forward ~{-chair_back * 100:.0f} cm (toward the fridge, not turning it)"
    return f"Move the WHEELCHAIR straight back ~{chair_back * 100:.0f} cm (away from the fridge, without turning it)"


def clear_through_drive(ai, door, settings, chair_back, chair_back_dir):
    """Before the base drives the chair `chair_back` m: the arm, held where it is, stays CHAIR_MOVE_MIN_CLEAR_M off the
    open door model all through the move (the push-open's check for its own move). Raises Refused."""
    from .push_open import _clear_through_move  # here: push_open imports this module

    door = dict(door)
    if settings.get("door_deg") is not None:
        door["door_open_deg"] = float(settings["door_deg"])
    scene, rb = make_sim()
    if not _clear_through_move(scene, rb, door, ai.get_state()["position"], chair_back, chair_back_dir):
        raise Refused(
            f"drive_base: the arm, where it is, comes within {F.CHAIR_MOVE_MIN_CLEAR_M * 100:.0f} cm of the open door "
            f"while the base drives {chair_back * 100:.0f} cm back -- the base not driven, nothing moved."
        )


def run(
    ai,
    execute,
    chair_back=F.CHAIR_BACK_M,
    close_to_deg=F.CLOSE_TO_DEG,
    chair_moved=False,
    chair_back_dir=F.CHAIR_BACK_DIR,
    speed=1.0,
    log_args=None,
    drive_chair=None,
    **settings,
):
    """The close as the node runs it, logged to RUNS_DIR/<time>_close: against the door the open saved, shifted by the
    chair's move, routed in front of the free edge, swinging the door only to `close_to_deg`. `settings`: GOAL_SETTINGS'
    close_door ones. `drive_chair` (the goal's drive_base: base_drive.move_chair): an execute goal drives
    the chair back `chair_back` itself instead of needing `chair_moved`. Returns True when it finished (or the dry run
    passed); raises Refused."""

    def task():
        if not dc.DOOR_FILE.exists():
            raise Refused(f"need {dc.DOOR_FILE}: the open writes it.")
        door = json.loads(dc.DOOR_FILE.read_text())
        if "open_sign" not in door:
            raise Refused(f"{dc.DOOR_FILE} has no opened door: the swing writes it.")
        shift = np.asarray(chair_back_dir, float) * chair_back
        hold = settings.get("hold_offset_x", F.CLOSE_SETTINGS["hold_offset_x"])
        say(
            f"fridge close: door from {dc.DOOR_FILE}, planned {chair_back * 100:.0f} cm further back (the wheelchair), "
            f"swung to ~{close_to_deg:.0f} deg, holding a {F.CONTAINER_M * 100:.1f} cm container (hand kept "
            f"{-hold * 100:.1f} cm back, incl. {F.CONTAINER_CLEARANCE_M * 100:.1f} cm margin); handle protrudes "
            f"{F.HANDLE_PROTRUSION_M * 100:.1f} cm (modelled)"
        )
        z = close_height(ai, door, settings, shift, close_to_deg)
        say(f"swing height z {z:.3f}")
        driven = False
        if not execute:
            drives = drive_chair is not None and chair_back > 0
            say(
                "(plan only: planned as if the wheelchair has already moved back"
                + (f"; with execute true the base drives it {chair_back * 100:.0f} cm back first)" if drives else ")")
            )
        elif chair_back > 0 and not chair_moved:
            if drive_chair is None:
                raise Refused(
                    f"{move_chair(chair_back)}, then send this goal again with chair_moved true. If it is still back "
                    "from the last close, leave it."
                )
            clear_through_drive(ai, door, settings, chair_back, chair_back_dir)
            say(f"\n>>> drive_base: driving the wheelchair ~{chair_back * 100:.0f} cm back for the close")
            moved = drive_chair(chair_back)  # within the base_tolerance of chair_back, or it raises
            driven = True
            shift = np.asarray(chair_back_dir, float) * moved
            say(f"the close re-planned against the door shifted by the {moved * 100:.1f} cm the base moved")
            z = close_height(ai, door, settings, shift, close_to_deg)
            say(f"swing height z {z:.3f}")
        ok = _push_close(ai, execute, {**settings, "push_z": z}, shift, close_to_deg)
        if ok and execute:
            say(
                f"door left at ~{close_to_deg:.0f} deg to swing shut. "
                + ("The base stays where it drove: drive" if driven else "Move")
                + " the wheelchair forward again before the next open (it detects the door afresh)."
            )
        return ok

    ok = logged_run("close", ai, log_args or {}, speed, task)
    say(f"\nCLOSE {'OK' if ok else 'STOPPED'}{'' if execute else ' (dry run)'}")
    return ok
