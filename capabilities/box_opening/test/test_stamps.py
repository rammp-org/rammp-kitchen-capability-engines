"""runtime/stamps.py: a trajectory point's time_from_start, read and written."""

import pytest
from builtin_interfaces.msg import Duration

from box_opening.runtime.stamps import secs, set_stamp


@pytest.mark.parametrize("t", [0.0, 0.02, 1.0, 1.5, 12.345678901])
def test_a_stamp_round_trips(t):
    d = Duration()
    set_stamp(d, t)
    assert 0 <= d.nanosec < 1000000000
    assert secs(d) == pytest.approx(t, abs=1e-9)


def test_a_fraction_that_rounds_to_a_whole_second_carries():
    """int(round(0.9999999996 * 1e9)) is 1000000000 — not a valid nanosec.
    Four of the six hand-rolled copies this module replaced wrote it as it
    was (retime, warp, the merge, the placeholder path)."""
    d = Duration()
    set_stamp(d, 1.9999999996)
    assert (d.sec, d.nanosec) == (2, 0)


def test_the_state_folder_can_be_pointed_elsewhere(tmp_path):
    """Runs, captures, generated worlds and calibration residuals live under
    ~/.ros/box_opening — unless BOX_OPENING_STATE says otherwise.
    The stub harnesses say otherwise: their synthetic runs once filled the
    real residuals file (2026-09-17), and the keep-newest-20 pruning of the
    every-run captures would let them push real bench frames out."""
    from pathlib import Path

    from box_opening.constants import state_dir

    assert state_dir({}) == Path.home() / ".ros" / "box_opening"
    assert state_dir({"BOX_OPENING_STATE": ""}) == Path.home() / ".ros" / "box_opening"
    assert state_dir({"BOX_OPENING_STATE": str(tmp_path)}) == tmp_path
