"""What the fridge's open, push-open and close share: run logged, plan what follows the swing, send checked joints."""

import time

import numpy as np
import rammp_arm
from door_common import open_door as shared
from door_common.common import DoorFrame, Refused, RunLog, say
from scipy.spatial.transform import Rotation as R

KEYPOINT_TOL_M = 0.01  # a path counts as done with the hand this close to its last keypoint


def logged_run(what, ai, log_args, speed, task):
    """`task()` -> ok, logged to RUNS_DIR/<time>_<what> (RunLog) with a camera frame before and after. `speed` (0..1]
    scales every arm motion for this run only. Returns ok; a refusal passes through."""
    saved = (rammp_arm.CART_SPEED, rammp_arm.JOINT_SPEED, rammp_arm.GO_TO_SPEED_SCALE)
    rammp_arm.CART_SPEED *= speed
    rammp_arm.JOINT_SPEED *= speed
    rammp_arm.GO_TO_SPEED_SCALE = max(0.01, rammp_arm.GO_TO_SPEED_SCALE * speed)
    log = None
    ok = False
    try:
        log = RunLog(what, log_args, ai)
        log.frame(f"{what}_start")
        ok = task()
        log.event(what, ok=ok)
        log.frame(f"{what}_end")
        return ok
    except BaseException as e:
        if log is not None:
            log.event("stopped", error=repr(e))
        raise
    finally:
        if log is not None:
            log.close(ok)
        rammp_arm.CART_SPEED, rammp_arm.JOINT_SPEED, rammp_arm.GO_TO_SPEED_SCALE = saved


def door_after(geom, cand):
    """The door dict as the open will have saved it once candidate `cand`'s swing ends (plan_open's `geom`, `cand`)."""
    h = cand["h"]
    door = dict(
        closed_normal=list(geom["n"]),
        closed_grasp_pos=list(cand["grasp"].position),
        closed_handle=list(h),
        door_z=list(h[2] + np.asarray(shared.DOOR_Z_FROM_HANDLE)),
        hinge=list(geom["hinge"]),
        door_open_deg=float(cand["swing"]),
        **shared.DOOR_MODEL,
    )
    door["open_sign"] = float(np.sign(DoorFrame(door, 1.0).angle_of(cand["wps"][-1][:3])) or 1.0)
    return door


class AfterSwing(shared.AfterSwing):
    """What follows the fridge's swing: `check` again from what the swing will really run, before it does."""

    def recheck(self, scene, rb, geom, cand, advice):
        ok, msg = self.check(scene, rb, geom, cand, True)
        say(f"after-swing re-check from the real grasp: {msg}")
        if not ok:
            raise Refused(
                f"{msg} from the real grasp -- stopping BEFORE the swing, still holding the handle (door closed). A "
                f"release_only goal lets go and backs off.{advice}"
            )


def send_legs(ai, pts, legs, speed_scale=1.0):
    """The joints the plan checked, as one smooth trajectory timed like the Cartesian path they follow. `pts`: the
    keypoints [(pos, quat)], `legs`: the joint list between each pair. Returns (reached the last keypoint, error m)."""
    lengths = []
    for (p0, q0), (p1, q1), leg in zip(pts[:-1], pts[1:], legs, strict=True):
        n = len(leg)
        ang = float((R.from_quat(q0).inv() * R.from_quat(q1)).magnitude())
        step = max(1e-4, float(np.linalg.norm(np.asarray(p1) - p0)) / n, ang / n * rammp_arm.ROT_EQUIV_M)
        lengths += [step] * n
    sent = ai.move_joints([q for leg in legs for q in leg], cart_lengths=lengths, speed_scale=speed_scale)
    time.sleep(0.3)
    err = float(np.linalg.norm(np.asarray(ai.get_state()["ee_pos"][:3]) - np.asarray(pts[-1][0])))
    say(f"hand {np.round(ai.get_state()['ee_pos'][:3], 3)}, {err * 100:.1f} cm from the last keypoint")
    return bool(sent) and err < KEYPOINT_TOL_M, err
