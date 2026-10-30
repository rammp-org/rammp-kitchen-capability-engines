"""Where THIS gripper's knuckle reads when the fingers are shut on nothing.

The mission always commands a full close (GRIPPER_CMD_CLOSED, the driver's
setpoint 1.0); what the knuckle then reads is the gripper's business, and it
changed: 0.793 through 2026-09-25 11:50, 0.636 after the arm driver restarted
at 14:04 — the fingertips visibly pressed together in the wrist frames both
before and after, and RAMMP-generalized's runtime measured the same "the empty
gripper closes at 0.636 rad" on its own. A close that finds the fingers
already shut at 0.636 never moves and never reaches 0.8, so the mission
refused to press with its fingers closed, three runs running.

So the reading of an empty close is learned from the mission's own touch
(the fingers are shut in free air then; learn_closed) and kept in the state
dir, and "already closed" and the grip band follow it (scaled_band: the
band was measured against the 0.793 reading, and this change scaled every
reading alike). With nothing learned, the 0.793 it was measured at; the
bench's state dir holds 0.636 from the day it changed.
"""

import json
import time

from box_opening.constants import state_dir

NOMINAL_CLOSED = 0.793  # the empty-close reading the grip band was measured against
CLOSED_RANGE = (0.55, 0.82)  # an empty close reads in here; anything else is not learned
CLOSED_TOL = 0.02  # this close under the empty-close reading is closed


def _file():
    return state_dir() / "gripper.json"


def learned_closed():
    """The knuckle an empty close reads on this gripper: the last one the
    mission measured, else NOMINAL_CLOSED."""
    try:
        v = float(json.loads(_file().read_text())["closed_knuckle"])
    except (OSError, ValueError, KeyError, TypeError):
        return NOMINAL_CLOSED
    return v if CLOSED_RANGE[0] <= v <= CLOSED_RANGE[1] else NOMINAL_CLOSED


def reads_empty_close(pos):
    """Whether a knuckle reading `pos` is one an empty close can give
    (CLOSED_RANGE), whichever of them this gripper gives today: the fingers
    on the knob read ~0.4, on the box ~0.1, and neither may ever pass for shut."""
    return pos is not None and CLOSED_RANGE[0] <= float(pos) <= CLOSED_RANGE[1]


def learn_closed(pos):
    """Keep `pos` as the empty-close reading — only a reading an empty close
    can give (reads_empty_close). Returns whether it was kept."""
    if not reads_empty_close(pos):
        return False
    try:
        f = _file()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"closed_knuckle": round(float(pos), 4), "measured_at": round(time.time(), 1)}))
    except OSError:
        return False
    return True


def already_closed(pos, closed=None):
    """Whether a knuckle reading `pos` is the fingers shut on nothing."""
    closed = learned_closed() if closed is None else closed
    return pos is not None and float(pos) >= closed - CLOSED_TOL


AIR_MARGIN = 0.05  # the band's top stays this far under an empty close: closed on air must fail


def scaled_band(band, closed=None):
    """The grip band (lo, hi), measured when an empty close read
    NOMINAL_CLOSED, for the reading in force: the bottom follows a lower
    empty-close reading down (should every reading have shrunk with it), the
    top stays where it was measured (should the knob read as it always did)
    but always AIR_MARGIN under an empty close — whichever the gripper is
    doing, a grip on the knob passes and a close on air or on the box fails.

    It was scaled as a whole at first; the gripper went back to 0.793 the
    next run, the band computed from 0.636 topped out at 0.441, and the knob
    read its usual 0.419 (2026-09-28)."""
    closed = learned_closed() if closed is None else closed
    f = min(1.0, closed / NOMINAL_CLOSED)
    return (float(band[0]) * f, min(float(band[1]), closed - AIR_MARGIN))
