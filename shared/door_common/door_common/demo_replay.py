"""Plan and run keypoints relative to a door: sim-gate every leg, then follow them as one smooth trajectory.

`plan` gates each leg (IK, joint jump, J4/J6, wrap, >= DEMO_MIN_CLEAR to the door model; legs from `trust_from` on
against the appliance's body only), `replay` checks the hand is at the first keypoint, plans, then (execute) does the
gripper action and sends it. Where the keypoints come from is the capability's: a recorded demo mapped onto today's
door, or computed for it.
"""

import time

import numpy as np
import rammp_arm
from scipy.spatial.transform import Rotation as R

from . import common as dc
from .common import (
    DoorFrame,
    drop,
    make_sim,
    plan_cartesian,
    run_cartesian_trajectory,
    say,
)

START_TOL = 0.06  # the hand must be within this of the demo's first keypoint
DEMO_MIN_CLEAR = 0.02  # to the door model, on legs before `trust_from`
# The last leg (the push) may stop short when all of it doesn't pass (10-30: snorlax's 30-deg mount folds the elbow
# past J4's guard as the push sweeps the hand back toward the arm): the longest of these fractions of it that passes.
# Below the smallest, the push is too short to be worth doing -- refuse.
SHORT_PUSH_FRACTIONS = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4)
LIFT_OFF_MIN_CLEAR = 0.005  # first leg of a demo that starts holding the handle: the fingers straddle the bar, i.e.
# start next to the door face (as the old door_push.RETREAT_OUT_MIN_CLEAR)


def plan(scene, rb, door, st, kps, trust_from, lift_off=False, short_last=False):
    """Sim-gated joint path through `kps` from the arm's state `st`: one list of joints per leg, or None.
    `short_last`: if the last leg fails, end it at the longest SHORT_PUSH_FRACTIONS of the way that passes -- `kps`'
    last keypoint is then replaced by where it ends (so the caller replays exactly what was planned)."""
    df = DoorFrame(door, door.get("open_sign", -1.0))
    q, legs = np.asarray(st["position"], float), []
    bodies = df.bodies(scene, rb, float(door.get("door_open_deg", 50.0)))
    try:
        pts = [(np.asarray(st["ee_pos"][:3]), np.asarray(st["ee_pos"][3:7]))] + kps
        for i in range(len(pts) - 1):
            trusted = trust_from is not None and i + 1 >= trust_from
            check = [b for b in bodies if b[1] != "door"] if trusted else bodies
            clear = LIFT_OFF_MIN_CLEAR if (i == 0 and lift_off) else DEMO_MIN_CLEAR
            leg = plan_cartesian(
                scene,
                rb,
                pts[i][0],
                pts[i][1],
                pts[i + 1][0],
                pts[i + 1][1],
                q,
                f"{'trusted' if trusted else 'leg'} {i + 1}",
                check,
                min_clear=clear,
            )
            if leg is None and short_last and i == len(pts) - 2:
                from scipy.spatial.transform import Slerp

                (p0, q0), (p1, q1) = pts[i], pts[i + 1]
                slerp = Slerp([0, 1], R.from_quat([q0, q1]))
                for f in SHORT_PUSH_FRACTIONS:
                    end = (np.asarray(p0) + (np.asarray(p1) - p0) * f, slerp([f]).as_quat()[0])
                    with dc.quiet():
                        leg = plan_cartesian(
                            scene, rb, p0, q0, *end, q, f"last leg to {f * 100:.0f}%", check, min_clear=clear
                        )
                    if leg is not None:
                        say(f"  last leg: all of it fails a gate -- pushing {f * 100:.0f}% of the way instead")
                        kps[-1] = end
                        break
            if leg is None:
                return None
            legs.append(leg)
            q = leg[-1]
    finally:
        drop(rb, bodies)
    return legs


def replay(pts, grippers, door, execute, trust_from, ai=None, impedance=None, speed_scale=1.0):
    """Replay keypoints: `pts` [(pos, quat)] (the first = where the hand must be now), `grippers` the
    gripper reading at each, `door` the door dict the legs are checked against. `ai`: the arm to use (the capability
    chains steps); else one is made here. `impedance`: run the legs compliant (self.arm's `impedance`); None: stiff
    position control. `speed_scale`: a further fraction of the contact speed. Returns True when it reached the last
    keypoint (or the dry run passed)."""
    own = ai is None
    ai = ai or rammp_arm.RammpArm()
    try:
        st = ai.get_state()
        off = float(np.linalg.norm(np.asarray(st["ee_pos"][:3]) - pts[0][0]))
        say(
            f"hand now {np.round(st['ee_pos'][:3], 3)}, {off * 100:.1f} cm from the demo's start; "
            f"gripper {st['gripper_pos']:.2f}"
        )
        if off > START_TOL:
            say(f"not at the demo's start (> {START_TOL * 100:.0f} cm) -- refusing.")
            return False
        scene, rb = make_sim()
        lift_off = grippers[0] > 0.2 and grippers[1] < 0.2
        kps = pts[1:]
        legs = plan(scene, rb, door, st, kps, trust_from, lift_off, short_last=True)
        if legs is None:
            say("a leg fails a gate -- refusing, arm untouched.")
            return False
        say(f"all {len(legs)} legs pass ({sum(len(leg) for leg in legs)} steps).")
        if not execute:
            say("\nDRY RUN -- nothing commanded.")
            return True
        if lift_off:  # the demo let go of the handle first
            say("opening the gripper ...")
            ai.gripper(0.0)
            time.sleep(0.5)
        # The push-further-open CAN'T HANDLE MORE SPEED: always run it at the contact speed
        # (rammp_arm.CONTACT_*_SPEED = 0.5 rad/s / 0.09 m/s), never the faster free-air speed. 10-24: at
        # 0.65/0.12 and 0.85/0.16 its last leg (keypoint 4 -> 5: the hand pushing into the door, which pushes
        # back, while turning 68 deg) aborted PATH_TOLERANCE_VIOLATED ~12.5 cm short both times; it ran clean
        # three times at 0.5 / 0.09.
        ok, err, _ = run_cartesian_trajectory(
            ai,
            [(list(p), list(q)) for p, q in kps],
            speed_scale=rammp_arm.contact_speed_scale() * speed_scale,
            impedance=impedance,
        )
        say(
            f"{'DONE' if ok else 'STOPPED SHORT'}: hand {np.round(ai.get_state()['ee_pos'][:3], 3)} "
            f"({err * 100:.1f} cm from the last keypoint)"
        )
        return bool(ok)
    finally:
        if own:
            ai.close()
