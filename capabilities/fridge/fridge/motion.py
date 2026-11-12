"""What the fridge's open, push-open and close share: each run logged (events, console, rosbag), the door files."""

import json
import shutil
import threading
import time

import numpy as np
import rammp_arm
from door_common import common as dc
from door_common.common import Refused, RunLog, say

_CONSOLE = None  # the running goal's console.log (logged_run sets it), or None between runs
_CONSOLE_LOCK = threading.Lock()  # the arm logs from its own threads too


def to_console(level, line):
    """Append one line the node logged to the running goal's console.log; nothing between runs (node.TeeLogger)."""
    with _CONSOLE_LOCK:
        if _CONSOLE is None:
            return
        try:
            with open(_CONSOLE, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {level} {line}\n")
        except OSError:
            pass  # a full disk must not stop a run


def _set_console(path):
    global _CONSOLE
    with _CONSOLE_LOCK:
        _CONSOLE = path


BAG = None  # (run_dir) -> a recorder with close() (bag.RunBag; the node sets it, record_bag); None records nothing
BAG_MIN_FREE_GB = 2.0  # no bag with less than this free on the data disk (~75 MB a minute a camera at 1 frame a
# second, ~30 more with the wrist camera's stereo pair)


def _start_bag(run_dir):
    """BAG's recorder into `run_dir`/bag, or None (says why). A recorder that fails must not stop the run."""
    if BAG is None:
        return None
    free_gb = shutil.disk_usage(run_dir).free / 1e9
    if free_gb < BAG_MIN_FREE_GB:
        say(f"  (no rosbag this run: {free_gb:.1f} GB free under {run_dir}, under {BAG_MIN_FREE_GB:g} GB)")
        return None
    try:
        rec = BAG(run_dir)
    except Exception as e:  # noqa: BLE001 -- never let the recording stop a run
        say(f"  (no rosbag this run: {e!r})")
        return None
    say(f"recording a rosbag to {run_dir / 'bag'}")
    return rec


def _stop_bag(rec):
    if rec is None:
        return
    try:
        frames = rec.close()
        pairs = getattr(rec, "stereo_pairs", None)
        if pairs is None:
            say(f"rosbag: {frames} camera frames")
        else:
            say(f"rosbag: {frames} camera frames, {pairs} stereo pairs")
            if pairs == 0:
                say(
                    "  (no stereo pair recorded: check the wrist_infra*_topic parameters name the topics the camera "
                    "publishes (ros2 topic list), that it publishes them (on the OAK: depth.i_left_rect_publish_topic "
                    "and depth.i_right_rect_publish_topic), and that both images carry the same stamp. "
                    "docs/TESTING.md, Set up)"
                )
    except Exception as e:  # noqa: BLE001 -- as _start_bag
        say(f"  (the rosbag did not close cleanly: {e!r})")


def logged_run(what, ai, log_args, speed, task):
    """`task()` -> ok, logged to RUNS_DIR/<time>_<what> (RunLog) with a camera frame before and after, everything the
    node logs meanwhile in its console.log, and a rosbag (BAG) in its bag/. `speed` (0..1] scales every arm
    motion for this run only. Returns ok; a refusal passes through."""
    saved = (rammp_arm.CART_SPEED, rammp_arm.JOINT_SPEED, rammp_arm.GO_TO_SPEED_SCALE)
    rammp_arm.CART_SPEED *= speed
    rammp_arm.JOINT_SPEED *= speed
    rammp_arm.GO_TO_SPEED_SCALE = max(0.01, rammp_arm.GO_TO_SPEED_SCALE * speed)
    log = bag = None
    ok = False
    try:
        log = RunLog(what, log_args, ai)
        _set_console(log.dir / "console.log")
        to_console("INFO", f"run log: {log.dir}")
        bag = _start_bag(log.dir)
        log.frame(f"{what}_start")
        ok = task()
        log.event(what, ok=ok)
        log.frame(f"{what}_end")
        return ok
    except BaseException as e:
        if log is not None:
            log.event("stopped", error=repr(e))
            to_console("ERROR", f"stopped: {e}")
        raise
    finally:
        if log is not None:
            log.close(ok)
        _stop_bag(bag)
        _set_console(None)
        rammp_arm.CART_SPEED, rammp_arm.JOINT_SPEED, rammp_arm.GO_TO_SPEED_SCALE = saved


def opened(door, deg):
    """A copy of door dict `door` (closed, or as saved) with the door open `deg`."""
    return {**door, "door_open_deg": float(deg)}


def shifted(door, move, chair_dir):
    """A copy of `door` as it stands in the arm's frame once the wheelchair has moved `move` (m, + back): its
    recorded positions moved by `chair_dir` x that (the node's chair_back_dir)."""
    dxy = np.asarray(chair_dir, float) * move
    return shifted_by(door, dxy)


def shifted_by(door, dxy):
    """A copy of `door` with its recorded positions moved by `dxy` (arm_mount x, y)."""
    out = dict(door)
    for k in ("hinge", "hinge_pivot", "closed_grasp_pos", "closed_handle", "door_mid"):
        if out.get(k) is not None:
            out[k] = [out[k][0] + dxy[0], out[k][1] + dxy[1], *out[k][2:]]
    return out


def load_door():
    """The door file (the open writes it), or Refused."""
    if not dc.DOOR_FILE.exists():
        raise Refused(f"need {dc.DOOR_FILE}: the open writes it.")
    return json.loads(dc.DOOR_FILE.read_text())
