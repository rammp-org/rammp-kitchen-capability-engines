"""The gripper's empty-close reading, learned (runtime/fingers).

Bench 2026-09-25: after the arm driver restarted, a close on nothing read
0.636 instead of 0.793 — the fingertips visibly shut in the wrist frames
either way. The mission's close found them already shut, "never moved", and
it refused to press three runs running."""

import pytest

from box_opening.runtime import fingers


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setenv("BOX_OPENING_STATE", str(tmp_path))
    return tmp_path


def test_fingers_shut_at_the_new_reading_count_as_closed(state):
    assert fingers.learned_closed() == pytest.approx(0.793)  # nothing learned: the reading the band was measured at
    assert not fingers.already_closed(0.636)
    assert fingers.learn_closed(0.636)  # the 2026-09-25 bench
    assert fingers.already_closed(0.636) and fingers.already_closed(0.793)  # either reading of a shut hand
    assert not fingers.already_closed(0.41)  # on the knob
    assert not fingers.already_closed(0.05)  # open
    assert not fingers.already_closed(None)


def test_only_an_empty_close_is_ever_learned(state):
    assert fingers.learn_closed(0.79) and fingers.learned_closed() == pytest.approx(0.79)
    assert not fingers.already_closed(0.636)  # the gripper reads full again: 0.636 is not shut then
    assert not fingers.learn_closed(0.41)  # the knob in the fingers: never passes for shut
    assert not fingers.learn_closed(0.09)  # the box
    assert not fingers.learn_closed(None)
    assert fingers.learned_closed() == pytest.approx(0.79)


def test_an_empty_close_reads_shut_whichever_reading_was_learned(state):
    """A close sent to fingers already shut never moves, and fails: what
    the knuckle reads then says whether they are shut — any reading an
    empty close gives, whichever one was learned last."""
    for learned in (None, 0.636, 0.79):
        if learned is not None:
            assert fingers.learn_closed(learned)
        for pos in (0.636, 0.770, 0.793):
            assert fingers.reads_empty_close(pos), (learned, pos)
        for pos in (0.41, 0.09, 0.0, None):  # the knob, the box, open, no reading
            assert not fingers.reads_empty_close(pos), (learned, pos)


def test_the_grip_band_holds_the_knob_in_either_gripper_state(state):
    """The band [0.3, 0.55] was measured when an empty close read 0.793, a
    knob grip 0.41. With an empty close at 0.636, the knob may read 0.33 (if
    every reading shrank) or its usual 0.419 (2026-09-28): both must pass,
    and a close on air or on the box must not."""
    fingers.learn_closed(0.636)
    lo, hi = fingers.scaled_band((0.3, 0.55))
    for knob in (0.41 * 0.636 / 0.793, 0.419):
        assert lo < knob < hi
    assert hi <= 0.636 - fingers.AIR_MARGIN  # a close on air fails
    assert lo > 0.09  # a close on the box fails
    assert fingers.scaled_band((0.3, 0.55), closed=0.793) == pytest.approx((0.3, 0.55))
