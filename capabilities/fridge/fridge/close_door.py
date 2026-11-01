"""Close the fridge: door_common's push-close, with three changes for a fridge.

1. The wheelchair is moved back ~CHAIR_BACK_M first; the recorded door is shifted by that much for the plan.
2. No going over the door: back out in front of the open door's free edge, left behind it, then forward to the swing
   start.
3. The swing stops with the door at `close_to_deg` (it swings shut by itself from there): no torque push.

The gripper is never touched (it may hold a container) and the hand never tilts. An action can't prompt, so an execute
goal must say the chair has been moved (`chair_moved`); until then it refuses, arm untouched.
"""

import json

import numpy as np
from door_common import close_door as shared
from door_common import common as dc
from door_common.common import Refused, quiet, say

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


def run(
    ai,
    execute,
    chair_back=F.CHAIR_BACK_M,
    close_to_deg=F.CLOSE_TO_DEG,
    chair_moved=False,
    chair_back_dir=F.CHAIR_BACK_DIR,
    speed=1.0,
    log_args=None,
    **settings,
):
    """The close as the node runs it, logged to RUNS_DIR/<time>_close: against the door the open saved, shifted by the
    chair's move, routed in front of the free edge, swinging the door only to `close_to_deg`. `settings`: GOAL_SETTINGS'
    close_door ones. Returns True when it finished (or the dry run passed); raises Refused."""

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
            f"swung to ~{close_to_deg:.0f} deg, hand held back {-hold * 100:.1f} cm (container)"
        )
        z = close_height(ai, door, settings, shift, close_to_deg)
        say(f"swing height z {z:.3f}")
        if not execute:
            say("(dry run: planned as if the wheelchair has already moved back)")
        elif chair_back > 0 and not chair_moved:
            raise Refused(
                f"Move the WHEELCHAIR straight back ~{chair_back * 100:.0f} cm (away from the fridge, without turning "
                "it), then send this goal again with chair_moved true. If it is still back from the last close, leave "
                "it."
            )
        ok = _push_close(ai, execute, {**settings, "push_z": z}, shift, close_to_deg)
        if ok and execute:
            say(
                f"door left at ~{close_to_deg:.0f} deg to swing shut. Move the wheelchair forward again before the "
                "next open (it detects the door afresh)."
            )
        return ok

    ok = logged_run("close", ai, log_args or {}, speed, task)
    say(f"\nCLOSE {'OK' if ok else 'STOPPED'}{'' if execute else ' (dry run)'}")
    return ok
