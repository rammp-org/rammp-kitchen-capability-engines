"""Each run's console.log and rosbag, beside its RunLog (as the fridge's motion.py, feature/15-fridge).

`to_console`: every line the node logs (node.TeeLogger: say(), self.arm, the engine, errors) while a run is open goes
to that run's console.log. `BAG`: the node sets it to bag.RunBag when record_bag is on; `start_bag` / `stop_bag`
never let the recorder stop a run.
"""

import shutil
import threading
import time

from door_common.common import say

_CONSOLE = None  # the running goal's console.log (set_console), or None between runs
_CONSOLE_LOCK = threading.Lock()  # the arm logs from its own threads too

BAG = None  # (run_dir) -> a recorder with close() (bag.RunBag; the node sets it, record_bag); None records nothing
BAG_MIN_FREE_GB = 5.0  # no bag with less than this free on the data disk (a bag is at most the node's bag_max_mb)


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


def set_console(path):
    """Send what the node logs to `path` from now on (None: stop)."""
    global _CONSOLE
    with _CONSOLE_LOCK:
        _CONSOLE = path


def start_bag(run_dir):
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


def stop_bag(rec):
    """Close `rec` (start_bag's), saying what it holds."""
    if rec is None:
        return
    try:
        frames = rec.close()
        pairs = getattr(rec, "stereo_pairs", None)
        size = getattr(rec, "bytes", 0) / 1e6
        say(
            f"rosbag: {frames} wrist camera frames"
            + ("" if pairs is None else f", {pairs} stereo pairs")
            + f", {size:.0f} MB"
            + (" (hit bag_max_mb: no frames after that)" if getattr(rec, "full", False) else "")
        )
    except Exception as e:  # noqa: BLE001 -- as start_bag
        say(f"  (the rosbag did not close cleanly: {e!r})")
