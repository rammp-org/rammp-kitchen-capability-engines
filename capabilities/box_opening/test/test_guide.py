"""Lining the chair up with the box (box_opening/guide.py): what the user is told, and when they are asked."""

import pytest

from box_opening import guide as g

IN = g.Reach(inside=True, planned=True)
CHECKING = g.Reach(inside=True, planned=None)


def _drive(guide, track, t0=0.0, dt=0.2):
    """Feed `track` [(xy, Reach)] a tick every `dt` s; returns [(t, Say)]."""
    out = []
    for i, (xy, reach) in enumerate(track):
        t = t0 + i * dt
        out.append((t, guide.update(t, None if xy is None else g.Fix(t, xy), reach)))
    return out


def test_the_instruction_is_the_users_move_toward_the_boundary_and_names_the_larger_miss():
    """The box too far away is "closer", too near "back"; too far to the left "left" (toward it), to the right
    "right". Distances to the nearest 5 cm, a few centimetres "a little"."""
    assert g.instruction((-0.23, 0.02))[:2] == ("closer", 0.23)
    assert g.instruction((0.12, -0.05))[:2] == ("back", 0.12)
    assert g.instruction((0.02, -0.20))[:2] == ("left", 0.20)
    assert g.instruction((-0.02, 0.20))[:2] == ("right", 0.20)
    assert g.words("closer", 0.23) == "Move closer, about 25 cm"
    assert g.words("left", 0.07) == "Move left, about 5 cm"
    assert g.words("back", 0.02) == "Move back a little"
    # the axis named last is kept unless the other miss is AXIS_SWITCH_M larger: no flipping near the diagonal
    assert g.instruction((-0.10, 0.11))[0] == "right"
    assert g.instruction((-0.12, 0.11), axis="y")[0] == "right"
    assert g.instruction((-0.15, 0.11), axis="y")[0] == "closer"


def test_a_drive_up_is_told_closer_then_hold_still_then_asked_and_open_flies():
    """Outside the boundary the user is told which way; inside it nothing is asked until the box has stood
    still STEADY_S and the run's own plan for it has passed; then the question, with Open and Cancel; Open,
    pressed on a box seen within FRESH_S, is the yes."""
    guide = g.Guide()
    far = [((0.95 - 0.02 * i, -0.15), g.Reach(False, (-0.25 + 0.02 * i, 0.0))) for i in range(10)]
    said = _drive(guide, far)
    assert said[0][1].state == g.MOVE and said[0][1].message == "Move closer, about 25 cm"
    assert said[-1][1].message == "Move closer, about 5 cm" and said[-1][1].hint[0] == "closer"
    assert [s.buttons for _t, s in said] == [((g.CANCEL, "Cancel"),)] * 10  # nothing to say yes to yet
    assert said[-1][1].pct > said[0][1].pct  # the bar fills as the box comes in
    # inside, but still rolling: hold still
    rolling = [((0.75 - 0.02 * i, -0.15), IN) for i in range(4)]
    said = _drive(guide, rolling, t0=2.0)
    assert {s.state for _t, s in said} == {g.HOLD} and said[-1][1].message == "Hold still, checking"
    # standing still, the plan not made yet: still hold; made, and still for STEADY_S: the question
    said = _drive(guide, [((0.69, -0.15), CHECKING)] * 6, t0=2.8)
    assert {s.state for _t, s in said} == {g.HOLD}
    said = _drive(guide, [((0.69, -0.15), IN)] * 3, t0=4.0)
    ready = said[-1][1]
    assert ready.state == g.READY and ready.message == "Ready. Open the container?"
    assert ready.buttons == ((g.OPEN, "Open"), (g.CANCEL, "Cancel")) and ready.pct == 100.0
    assert ready.panel() == {
        "title": "Opening the container",
        "message": "Ready. Open the container?",
        "progress": {"phase": "ready", "pct": 100.0, "pausable": False},
        "buttons": [{"button_id": "open", "label": "Open"}, {"button_id": "cancel", "label": "Cancel"}],
    }
    assert guide.press(g.CANCEL, 4.5) is False and guide.state == g.READY  # Cancel is the caller's
    assert guide.press(g.OPEN, 4.5) is True and guide.state == g.GO


def test_open_is_no_yes_to_a_box_that_is_not_ready_or_was_last_seen_a_while_ago():
    guide = g.Guide()
    _drive(guide, [((0.90, 0.0), g.Reach(False, (-0.2, 0.0)))] * 3)
    assert guide.press(g.OPEN, 0.6) is False and guide.state == g.MOVE  # a press while it says "move closer"
    _drive(guide, [((0.69, 0.0), IN)] * 8, t0=1.0)
    assert guide.state == g.READY
    assert guide.press(g.OPEN, 2.4 + g.FRESH_S + 0.1) is False  # the camera has shown nothing since: no yes
    assert guide.state == g.READY
    # ... and with nothing seen for LOST_S the question goes: looking again
    say = guide.update(2.4 + g.LOST_S + 0.1, None, IN)
    assert say.state == g.LOOKING and say.message == "Looking for the container" and g.OPEN not in dict(say.buttons)


def test_ready_survives_a_flicker_and_is_withdrawn_when_the_box_really_leaves_or_moves():
    """One tick of the boundary or the plan saying no does not take the question away (LEAVE_S); the box
    out, or rolling, for that long does, and the user is told which way again."""
    guide = g.Guide()
    _drive(guide, [((0.69, 0.0), IN)] * 8)
    assert guide.state == g.READY
    out = g.Reach(False, (-0.06, 0.0))
    say = guide.update(1.6, g.Fix(1.6, (0.69, 0.0)), out)
    assert say.state == g.READY  # a flicker
    say = guide.update(1.8, g.Fix(1.8, (0.69, 0.0)), IN)
    assert say.state == g.READY
    said = _drive(guide, [((0.75, 0.0), out)] * 5, t0=2.0)
    assert said[1][1].state == g.READY and said[-1][1].state == g.MOVE
    assert said[-1][1].message == "Move closer, about 5 cm"
    # a plan that fails where the map said yes: told to move, by where the nearest good place is
    guide = g.Guide()
    said = _drive(guide, [((0.60, 0.20), g.Reach(True, (0.0, -0.08), planned=False))] * 8)
    assert said[-1][1].state == g.MOVE and said[-1][1].message == "Move left, about 10 cm"
    # ... or, with no such place known, that it cannot be opened from here
    guide = g.Guide()
    said = _drive(guide, [((1.30, 0.0), g.Reach(False, None))] * 2)
    assert said[-1][1].message == "The arm cannot open it from here"


def test_ready_survives_a_tick_the_boundary_cannot_judge_as_it_survives_one_that_says_no():
    """A tick with no Reach at all (one frame whose lid height falls in another boundary's bucket, a lid a
    view cut) is a flicker like any other: the question and its Open stay for LEAVE_S, a press in that time
    is still the user's yes, and only LEAVE_S of it is "Hold still, checking"."""
    guide = g.Guide()
    _drive(guide, [((0.69, 0.0), IN)] * 8)  # 0 .. 1.4 s
    assert guide.state == g.READY
    say = guide.update(1.6, g.Fix(1.6, (0.69, 0.0)), None)
    assert say.state == g.READY and guide.state == g.READY and g.OPEN in dict(say.buttons)
    say = guide.update(1.8, g.Fix(1.8, (0.69, 0.0)), IN)
    assert say.state == g.READY
    said = _drive(guide, [((0.69, 0.0), None)] * 5, t0=2.0)  # not judged from 2.0 s on
    assert [s.state for _t, s in said[:3]] == [g.READY] * 3  # 2.0, 2.2, 2.4: within LEAVE_S of the first
    assert said[-1][1].state == g.HOLD and said[-1][1].message == "Hold still, checking"
    assert g.OPEN not in dict(said[-1][1].buttons) and guide.press(g.OPEN, 2.85) is False
    said = _drive(guide, [((0.69, 0.0), IN)] * 2, t0=3.0)  # judged again, and it never moved: asked again
    assert said[-1][1].state == g.READY
    guide.update(3.4, g.Fix(3.4, (0.69, 0.0)), None)
    assert guide.press(g.OPEN, 3.45) is True and guide.state == g.GO  # Open in the flicker: taken


def test_an_instruction_stands_long_enough_to_read_and_the_line_up_has_its_time():
    """Another instruction replaces one only after MIN_SAY_S; a change of state is said at once. `budget_s`
    over, the line-up ends and says how to go on: nothing flies on a press after that."""
    guide = g.Guide()
    a = guide.update(0.0, g.Fix(0.0, (0.9, 0.3)), g.Reach(False, (-0.20, -0.10)))
    b = guide.update(0.2, g.Fix(0.2, (0.9, 0.3)), g.Reach(False, (-0.05, -0.20)))
    assert a.message == "Move closer, about 20 cm" and b is a  # "left" waits its turn
    c = guide.update(0.2 + g.MIN_SAY_S, g.Fix(0.9, (0.9, 0.3)), g.Reach(False, (-0.05, -0.20)))
    assert c.message == "Move left, about 20 cm"
    guide = g.Guide(budget_s=5.0)
    _drive(guide, [((0.9, 0.0), g.Reach(False, (-0.2, 0.0)))] * 3)
    say = guide.update(5.1, g.Fix(5.1, (0.69, 0.0)), IN)
    assert say.state == g.TIMED_OUT and say.message == "Press open again when you are in place" and say.buttons == ()
    assert guide.press(g.OPEN, 5.2) is False and guide.update(6.0, g.Fix(6.0, (0.69, 0.0)), IN) is say


@pytest.mark.parametrize("gap_s, steady", [(0.2, True), (0.6, True), (1.2, False)])
def test_steady_needs_the_box_seen_all_through_the_last_second(gap_s, steady):
    """A box that stood still but was last seen more than FRESH_S ago is not known to be still."""
    guide = g.Guide()
    _drive(guide, [((0.69, 0.0), IN)] * 8)  # 0 .. 1.4 s
    assert guide.steady(1.4 + gap_s) is steady
