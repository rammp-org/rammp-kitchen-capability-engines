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
    # inside, but still rolling: hold still, once it has read inside for HOLD_AFTER_MOVE_S (the instruction
    # stands until then: a box on the boundary's edge reads inside one frame and outside the next)
    rolling = [((0.75 - 0.02 * i, -0.15), IN) for i in range(4)]
    said = _drive(guide, rolling, t0=2.0)
    assert [s.state for _t, s in said] == [g.MOVE, g.MOVE, g.MOVE, g.HOLD]
    assert said[0][1].message == "Move closer, about 5 cm" and said[-1][1].message == "Hold still, checking"
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
        "buttons": [
            {"button_id": "open", "label": "Open", "style": "normal"},
            {"button_id": "cancel", "label": "Cancel", "style": "warning"},
        ],
    }
    assert guide.press(g.CANCEL, 4.5) is False and guide.state == g.READY  # Cancel is the caller's
    assert guide.press(g.OPEN, 4.5) is True and guide.state == g.GO


def test_a_box_on_the_boundarys_edge_is_told_to_move_not_hold_still_and_move_by_turns():
    """The review of 2026-11-17: a box on the boundary's edge reads inside it on one wrist frame and a
    centimetre or two outside on the next (five frames a second, the 3 cm margin either side). The guide
    had no hold between "Hold still, checking" and "Move closer a little", and said them by turns at frame
    rate: on the panel and, 51 lines in 10 s, on the console. A small move is still to make, and that is
    what stands. "Hold still" comes once the box has read inside for HOLD_AFTER_MOVE_S without a break,
    and the question is not held back by any of it."""
    guide = g.Guide()
    edge = [((0.80, 0.0), CHECKING if (i // 4) % 2 else g.Reach(False, (-0.02, 0.0))) for i in range(200)]
    said = _drive(guide, edge, dt=0.05)  # 10 s; each frame's word stands four ticks, 0.2 s
    assert {(s.state, s.message) for _t, s in said} == {(g.MOVE, "Move closer a little")}
    assert g.HOLD_AFTER_MOVE_S == 0.5  # two to three wrist frames
    # a nudge in, and it reads inside for good (its last frame on the edge, from 9.8 s, read inside too):
    # "hold still" half a second after it first did, and no sooner
    said = _drive(guide, [((0.78, 0.0), CHECKING)] * 20, t0=10.0, dt=0.05)
    held = [t for t, s in said if s.state == g.HOLD]
    assert held and held[0] == pytest.approx(9.8 + g.HOLD_AFTER_MOVE_S, abs=0.051)
    assert {s.state for t, s in said if t > held[0]} == {g.HOLD}
    # an instruction that takes "hold still"'s place is said at once: it is the way to go
    say = guide.update(11.0, g.Fix(11.0, (0.85, 0.0)), g.Reach(False, (-0.07, 0.0)))
    assert (say.state, say.message) == (g.MOVE, "Move closer, about 5 cm")
    # the question waits for none of it: straight from an instruction, the moment the box is inside,
    # steady and planned
    guide = g.Guide()
    _drive(guide, [((0.80, 0.0), g.Reach(False, (-0.02, 0.0)))] * 30, dt=0.05)  # 1.5 s standing just outside
    say = guide.update(1.5, g.Fix(1.5, (0.80, 0.0)), IN)
    assert say.state == g.READY and guide.press(g.OPEN, 1.55) is True
    # ... and a box that has just been found and is not judged yet is still "hold still" at once
    guide = g.Guide()
    assert guide.update(0.0, g.Fix(0.0, (0.70, 0.0)), None).state == g.HOLD


def test_open_is_no_yes_to_a_box_that_is_not_ready_or_was_last_seen_a_while_ago():
    guide = g.Guide()
    _drive(guide, [((0.90, 0.0), g.Reach(False, (-0.2, 0.0)))] * 3)
    assert guide.press(g.OPEN, 0.6) is False and guide.state == g.MOVE  # a press while it says "move closer"
    _drive(guide, [((0.69, 0.0), IN)] * 8, t0=1.0)
    assert guide.state == g.READY
    assert guide.press(g.OPEN, 2.4 + g.FRESH_S + 0.1) is False  # the camera has shown nothing since: no yes
    assert guide.state == g.READY
    # ... and with nothing seen for LOST_S the question goes: the arm no longer sees it, and says so
    say = guide.update(2.4 + g.LOST_S + 0.1, None, IN)
    assert say.state == g.LOOKING and say.message == g.UNSEEN and g.OPEN not in dict(say.buttons)


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


def test_out_of_time_it_says_its_own_words_and_keeps_the_newest_instruction_it_gave():
    """A goal with no UI is not told to "press open again": the engine hands the guide its last words
    (`again`). And the newest "Move ..." is kept (`instruction`): the engine's error carries it where the
    line-up ends without its yes, so the person knows what was still to do."""
    guide = g.Guide(budget_s=2.0, again="Out of time. Start open_box again when you are in place")
    assert guide.instruction is None and not guide.saw_box
    _drive(guide, [((0.95, 0.0), g.Reach(False, (-0.25, 0.0)))] * 3)
    assert guide.instruction == "Move closer, about 25 cm" and guide.saw_box
    _drive(guide, [((0.80, 0.0), g.Reach(False, (-0.10, 0.0)))] * 6, t0=0.6)  # (the first held MIN_SAY_S)
    assert guide.instruction == "Move closer, about 10 cm"
    said = _drive(guide, [((0.80, 0.0), g.Reach(False, (-0.10, 0.0)))], t0=2.5)
    assert said[-1][1].state == g.TIMED_OUT and said[-1][1].buttons == ()
    assert said[-1][1].message == "Out of time. Start open_box again when you are in place"
    assert guide.instruction == "Move closer, about 10 cm"  # what was still to do
    # the panel's own words where none are given, as they were
    late = g.Guide(budget_s=1.0)
    said = _drive(late, [((0.95, 0.0), g.Reach(False, (-0.25, 0.0)))] * 2, t0=0.0, dt=1.5)
    assert said[-1][1].message == g.AGAIN == "Press open again when you are in place" and late.saw_box
    assert g.ASK == "Ready. Open the container?"


LOOKED = "wrist camera: no counter in 1 frame (no level plane wider than a container)"
TWO = "wrist camera: two places pass as this container, [0.61, -0.10] and [0.66, 0.14]: refusing to guess"
UNSEEN, SEVERAL, NOT_CHOSEN, FAULT = g.K_UNSEEN, g.K_SEVERAL, g.K_NOT_CHOSEN, g.K_FAULT


def test_a_container_no_camera_sees_is_said_and_the_chair_is_what_is_to_move():
    """Owner 2026-11-19, asked whether a box no camera sees at all should become guidance ("move so the
    container is in front of the arm"): yes. Until then the panel stood on "Looking for the container" for the
    whole line-up, which tells the person nothing to do. Once what watches has looked and had none for as
    long as a lost container is given, the person is told the arm cannot see the container and to move the
    CHAIR. No direction and no distance: none is known. Nothing is asked, and Open is no yes."""
    guide = g.Guide()
    say = guide.update(0.0, None, None)
    assert (say.state, say.message) == (g.LOOKING, "Looking for the container")  # nothing has looked yet
    assert guide.update(0.2, None, None, none=UNSEEN).message == g.SEARCHING  # its first word: still looking
    say = guide.update(0.2 + g.LOST_S, None, None, none=UNSEEN)
    assert say.state == g.LOOKING and say.message == g.UNSEEN
    assert say.message == "The arm cannot see the container. Move the chair in front of it"
    assert say.buttons == ((g.CANCEL, "Cancel"),) and say.hint is None and say.pct == 0.0
    assert guide.press(g.OPEN, 2.0) is False and not guide.saw_box and guide.instruction is None
    # it stands for as long as nothing that could be the container is seen
    for i in range(3):
        assert guide.update(2.0 + 0.2 * i, None, None, none=UNSEEN).message == g.UNSEEN


def test_one_frame_without_the_container_does_not_tell_a_person_who_is_in_place_that_the_arm_cannot_see_it():
    """The review of 2026-11-19: LOST_S held the sentence back only after a first sighting. At the start of
    any line-up the watch's first frame alone, or a late TF, showed "The arm cannot see the container. Move
    the chair in front of it" to a person who was in place, and they were told to move. Until what watches
    has had none for LOST_S, as long as a lost container is given, it is still looking; and a watch whose
    word comes and goes starts that time again."""
    guide = g.Guide()
    assert guide.update(0.0, None, None).message == g.SEARCHING
    assert guide.update(0.1, None, None, none=UNSEEN).message == g.SEARCHING
    say = guide.update(0.3, g.Fix(0.3, (0.65, 0.0)), None)  # the next frame shows it: never "cannot see"
    assert (say.state, say.message) == (g.HOLD, "Hold still, checking")
    guide = g.Guide()
    for t, none, told in (
        (0.0, UNSEEN, g.SEARCHING),
        (1.0, UNSEEN, g.SEARCHING),
        (1.2, "", g.SEARCHING),  # (it has no word again: not yet said to have none)
        (1.4, UNSEEN, g.SEARCHING),
        (2.8, UNSEEN, g.SEARCHING),
        (1.4 + g.LOST_S, UNSEEN, g.UNSEEN),
    ):
        assert guide.update(t, None, None, none=none).message == told, t
    # a run whose own look found none HAD looked, for longer than that, before the line-up began: its word
    # is said from the first panel (the wrist's turn to its view is not seconds of "Looking")
    guide = g.Guide()
    assert guide.update(0.0, None, None, none=UNSEEN, looked=True).message == g.UNSEEN


def test_the_moment_a_camera_sees_the_container_the_way_and_the_distance_are_said_and_lost_it_is_unseen_again():
    """The ordinary guidance takes over at the first sighting, at once: which way and how far, where the
    boundary knows them. A container that goes out of view again is not "looked for": the arm cannot see
    it, and the person is told so again, whether or not what watches has yet said why."""
    guide = g.Guide()
    assert guide.update(0.0, None, None, none=UNSEEN, looked=True).message == g.UNSEEN
    say = guide.update(0.2, g.Fix(0.2, (1.05, 0.0), "scene"), g.Reach(False, (-0.30, 0.0)), none="")
    assert (say.state, say.message) == (g.MOVE, "Move closer, about 30 cm") and guide.saw_box
    # a sighting the boundary cannot judge yet says no distance it does not have
    say = guide.update(0.4, g.Fix(0.4, (1.05, 0.0), "scene"), g.Reach(False, None), none="")
    assert say.message == "The arm cannot open it from here" and say.hint is None
    lost = 0.4 + g.LOST_S + 0.1
    assert guide.update(lost - 0.2, None, None, none="").state == g.MOVE  # (not yet: one frame without it)
    say = guide.update(lost, None, None, none="")
    assert (say.state, say.message) == (g.LOOKING, g.UNSEEN)
    say = guide.update(lost + 0.2, g.Fix(lost + 0.2, (0.95, 0.0)), g.Reach(False, (-0.20, 0.0)), none=UNSEEN)
    assert say.message == "Move closer, about 20 cm"  # seen again: its sighting is newer than that word


def test_what_no_moving_of_the_chair_cures_is_never_said_as_a_container_the_arm_cannot_see():
    """Guidance is for what moving the chair can cure (review of 2026-11-19). A camera that delivers nothing,
    two things that could be the container, and a container nothing ties to the one the person chose all
    wore "The arm cannot see the container. Move the chair in front of it" (or "Move in front of only one",
    which no chair can do for two tops side by side). Each is said as what it is, with no word of moving."""
    assert g.NO_CURE == {
        SEVERAL: "The arm sees two things that could be the container",
        NOT_CHOSEN: "The arm cannot tell which container you chose",
        FAULT: "The arm's vision is not working",
    }
    # ("vision": what it is said of is as often no camera at all as a camera, the scene segmenter down, a
    # calibration missing, a container with no segmenter classes, a watch fallen silent: review of 2026-11-19)
    assert "camera" not in g.NO_CURE[FAULT]
    for kind, told in g.NO_CURE.items():
        assert g.says(kind) == told and "move" not in told.lower() and told != g.UNSEEN
        guide = g.Guide()
        assert guide.update(0.0, None, None, none=kind).message == g.SEARCHING  # (not on its first word)
        assert guide.update(g.LOST_S, None, None, none=kind).message == told
        assert guide.press(g.OPEN, g.LOST_S + 0.1) is False
    assert g.says(UNSEEN) == g.says("") == g.UNSEEN
    # "two" is two THINGS: the second may be any lid-sized top (a bowl, a roll of tape), not a container
    assert "containers" not in g.NO_CURE[SEVERAL]
    # ... a container that was seen and is lost to one of these: looked for until that has stood, then said.
    # Never, in between, that the arm cannot see the container
    guide = g.Guide()
    guide.update(0.0, g.Fix(0.0, (0.9, 0.0)), g.Reach(False, (-0.2, 0.0)))
    assert guide.update(g.LOST_S + 0.1, None, None, none=FAULT).message == g.SEARCHING
    assert guide.update(2 * g.LOST_S + 0.1, None, None, none=FAULT).message == g.NO_CURE[FAULT]
    # frame by frame the watch's word may change: one sentence stands long enough to be read before another
    guide = g.Guide()
    t0 = g.LOST_S
    assert guide.update(0.0, None, None, none=SEVERAL).message == g.SEARCHING
    assert guide.update(t0, None, None, none=SEVERAL).message == g.NO_CURE[SEVERAL]
    assert guide.update(t0 + 1.0, None, None, none=UNSEEN).message == g.NO_CURE[SEVERAL]
    assert guide.update(t0 + g.UNSEEN_STANDS_S + 0.1, None, None, none=UNSEEN).message == g.UNSEEN
    assert guide.update(t0 + g.UNSEEN_STANDS_S + 0.3, None, None, none=SEVERAL).message == g.UNSEEN
    # ... and a container seen takes any of them away at once
    at = t0 + g.UNSEEN_STANDS_S + 0.5
    assert guide.update(at, g.Fix(at, (1.0, 0.0)), g.Reach(False, (-0.2, 0.0))).state == g.MOVE


def test_the_time_running_out_on_what_no_chair_move_cures_ends_in_its_own_words():
    """A line-up may have fewer seconds than such a state must stand to end it (under the orchestrator's
    120 s). Its time then runs out with that state's sentence on the panel, and the last word is the same
    sentence: "Container not seen in time. Press open again in front of it" is not true of a camera that is
    not working. `uncured` tells the engine which state it was, for the end it reports."""
    assert set(g.ENDS) == {FAULT, NOT_CHOSEN}  # (two tops in a frame end no line-up: below)
    for kind in g.ENDS:
        told = g.NO_CURE[kind]
        guide = g.Guide(budget_s=4.0)
        guide.update(0.0, None, None, none=kind)
        assert guide.update(2.0, None, None, none=kind).message == told and guide.uncured == ""
        last = guide.update(4.1, None, None, none=kind)
        assert (last.state, last.message, last.buttons) == (g.TIMED_OUT, told, ()) and guide.uncured == kind
    # two tops in a frame as the time runs out: the time ran out, as at 21aa7cc, and that is what is said
    guide = g.Guide(budget_s=4.0)
    guide.update(0.0, None, None, none=SEVERAL)
    assert guide.update(2.0, None, None, none=SEVERAL).message == g.NO_CURE[SEVERAL]
    assert guide.update(4.1, None, None, none=SEVERAL).message == g.AGAIN_UNSEEN and guide.uncured == ""
    # a container that is simply not in view, and one that was seen and is out of reach: as they were
    guide = g.Guide(budget_s=4.0)
    guide.update(0.0, None, None, none=UNSEEN)
    assert guide.update(4.1, None, None, none=UNSEEN).message == g.AGAIN_UNSEEN and guide.uncured == ""
    guide = g.Guide(budget_s=4.0)
    guide.update(0.0, g.Fix(0.0, (0.9, 0.0)), g.Reach(False, (-0.2, 0.0)))
    assert guide.update(4.1, None, None).message == g.AGAIN and guide.uncured == ""
    # ... nor one whose word had not yet stood long enough to be said (the panel still said it was looking)
    guide = g.Guide(budget_s=4.0)
    guide.update(0.0, None, None)
    guide.update(3.0, None, None, none=UNSEEN)
    assert guide.update(3.9, None, None, none=FAULT).message == g.SEARCHING
    assert guide.update(4.1, None, None, none=FAULT).message == g.AGAIN_UNSEEN and guide.uncured == ""


def test_a_state_that_is_what_watches_own_is_said_at_once_whatever_was_seen_a_moment_ago():
    """Where the line-up watches with the wrist and the wrist's camera is at fault, the scene camera's
    sightings go on arriving; and a watch fallen silent leaves its last sighting on record. Neither hides
    the state: what watches says it stands (`stands`, counted by the source), and it is said now. As first
    written a sighting on record put off every such sentence by LOST_S and then by LOST_S again, and each
    new one started that anew: the fault was never said at all (review of 2026-11-19, the second)."""
    guide = g.Guide()
    assert guide.update(0.0, g.Fix(0.0, (0.9, 0.0)), g.Reach(False, (-0.2, 0.0))).state == g.MOVE
    # a sighting half a second old, and the fault stands: said, with Cancel alone
    said = guide.update(0.5, None, None, none=FAULT, stands=True)
    assert (said.state, said.message) == (g.LOOKING, g.NO_CURE[FAULT])
    assert [b for b, _label in said.buttons] == [g.CANCEL] and guide.press(g.OPEN, 0.6) is False
    # ... and it goes on being said while it stands, a newer sighting of the other camera or none
    assert guide.update(1.0, None, None, none=FAULT, stands=True).message == g.NO_CURE[FAULT]
    # without it the same tick is the guidance it was: the sighting is fresh
    plain = g.Guide()
    plain.update(0.0, g.Fix(0.0, (0.9, 0.0)), g.Reach(False, (-0.2, 0.0)))
    assert plain.update(0.5, None, None, none=FAULT).state == g.MOVE
    # `stands` is for what no chair move cures alone: a container that is not in view is never hurried
    plain = g.Guide()
    assert plain.update(0.0, None, None, none=UNSEEN, stands=True).message == g.SEARCHING


SAID_BY = {
    # (what each camera's word on having no container is of) -> the words, as the module that says them has them
    FAULT: {
        "tasks/survey.py": (
            "wrist camera: no wrist camera",
            "wrist camera: no new wrist frame",
            "no wrist frame at survey:angle-1 (no new wrist frame)",
            "wrist camera: no camera pose for the frame (TF)",
        ),
        "perception/depth_source.py": (
            "wrist camera: no depth stamp yet",
            "wrist camera: depth not new",
            "wrist camera: depth not paired with colour (dt 180 ms)",
        ),
        "perception/scene_source.py": (
            "scene camera streams missing: color, depth",
            "no TF between the scene camera's depth and colour frames",
            "no TF from arm_mount to the scene camera (oak_scene_link) — is sheppy's static_tf node up?",
            "no TF from the scene camera's link to its colour frame",
            "the scene segmenter cannot be read: no such topic",
            "no scene segmenter for this container (its config names no scene_detect.segmenter_classes)",
            "TF has the scene camera's frames upside down — its detectors need them upright",
            "no scene depth cloud",
            "scene segmenter: no frame newer than the look in 1.0 s",
        ),
        "perception/segmenter_source.py": (
            "scene segmenter: ABSENT: nothing publishes /perception/object_detections; select scene_segmenter",
            "scene segmenter: SILENT: no /perception/object_detections in 3.0 s",
            "scene segmenter: it watches 'wrist_color', not the scene camera ('scene_color')",
            "scene segmenter: its frames are 640x480, the scene camera's 1280x720",
            "scene segmenter: its newest frame is 4.2 s older than the scene camera's",
        ),
        "tasks/box_watch.py": ("no scene camera calibration, and the robot's TF does not carry its mount",),
    },
    SEVERAL: {
        "perception/wrist_finder.py": (TWO,),
        "perception/scene_source.py": (
            "scene segmenter: two places pass as this container, [0.80, -0.20] and [1.10, 0.30]: select which",
        ),
        "tasks/press_demo.py": (
            "a container at [0.650, 0.000], 12 mm to the side of the scene camera's line of sight to it, and "
            "another lid-sized top at [0.700, 0.100] in the same direction: not taken for the selected container x",
            "a container at [0.650, 0.000] and another lid-sized top at [0.700, 0.200]: two containers, and the "
            "selected one (fc_1) is in none of the scene camera's detections, so nothing says which it is",
        ),
    },
    NOT_CHOSEN: {
        # something that could be the container IS in view (the wrist has a lid), and nothing ties it to the
        # selected one: its id is in none of the scene camera's detections, or the lid stands off the scene
        # camera's line to it
        "tasks/press_demo.py": (
            "the selected container fc_1 is in none of the scene camera's detections: nothing says that what the "
            "wrist sees is the one selected, and the run would not take it",
            "a container at [0.620, 0.040], and the selected container fc_1 is in none of the scene camera's "
            "detections: nothing says it is the one selected",
            "a container at [0.650, 0.000], 212 mm to the side of the scene camera's line of sight to it: not "
            "taken for the selected container fc_1",
        ),
    },
    UNSEEN: {
        # a camera that DELIVERS and finds nothing that could be the container: not in view, a lid a view
        # cuts, a mount that is off, depth with holes. From what it says none of these can be told from
        # the others, and the chair is what the person is told to move for all of them
        "perception/wrist_finder.py": (
            LOOKED,
            "wrist camera: too little depth in 1 frame",
            "wrist camera: nothing on the counter is the container (2 things at lid height)",
            "wrist camera: the container may stand at [0.70, 0.30], cut off by the view (x): look again",
            "wrist camera: the lid-sized tops at [1.02, -0.20] and [1.04, 0.25] stand out of the arm's reach, and "
            "none is nearest the arm alone: refusing to guess",
        ),
        "perception/scene_source.py": (
            "scene segmenter: saw no food container in 14 frames",
            "scene segmenter: none of its 12 boxes is this container: the top is not at lid height",
            "scene segmenter: one of its boxes passed in 9 frames, and a fix needs two that agree",
            # a selected id that is merely ABSENT from the scene camera's frames: the container may simply be
            # out of its view, and nothing says that anything else is in it. Read as "not chosen" (a626985),
            # it ended a scene-only line-up after ten seconds with "Choose it again" for a container that
            # was not in view yet (review of 2026-11-19, the second)
            "scene segmenter: the selected container (fc_1) was in none of its 70 frames",
        ),
    },
}


def test_what_a_cameras_word_on_having_no_container_is_of_is_read_from_the_words_other_modules_say():
    """kind_of reads words that tasks/survey, perception/scene_source and the like say, which are not this
    module's to keep. Each mark is held to the source it is said to come from: reworded there, this fails
    and says where, in the place of a camera fault that would quietly read as "move the chair"."""
    from pathlib import Path

    import box_opening

    assert g.kind_of("") == g.kind_of(None) == ""
    for kind, modules in SAID_BY.items():
        for words in modules.values():
            for why in words:
                assert g.kind_of(why) == kind, why
    root = Path(box_opening.__file__).parent
    sources = {name: (root / name).read_text() for name in set().union(*SAID_BY.values())}
    said_in = {
        "no wrist camera": "tasks/survey.py",
        "no new wrist frame": "tasks/survey.py",
        "no wrist frame at": "tasks/survey.py",
        "no camera pose for the frame": "tasks/survey.py",
        "no depth stamp yet": "perception/depth_source.py",
        "depth not new": "perception/depth_source.py",
        "depth not paired with colour": "perception/depth_source.py",
        "scene camera streams missing": "perception/scene_source.py",
        "no TF between the scene camera": "perception/scene_source.py",
        "to the scene camera (": "perception/scene_source.py",
        "no TF from the scene camera": "perception/scene_source.py",
        "the scene segmenter cannot be read": "perception/scene_source.py",
        "no scene segmenter for this container": "perception/scene_source.py",
        "frames upside down": "perception/scene_source.py",
        "no scene depth cloud": "perception/scene_source.py",
        "no frame newer than the look": "perception/scene_source.py",
        "ABSENT: nothing publishes": "perception/segmenter_source.py",
        "SILENT: no ": "perception/segmenter_source.py",
        "not the scene camera (": "perception/segmenter_source.py",
        "its frames are ": "perception/segmenter_source.py",
        "older than the scene camera's": "perception/segmenter_source.py",
        "no scene camera calibration": "tasks/box_watch.py",
        "two places pass": "perception/wrist_finder.py",
        "another lid-sized top": "tasks/press_demo.py",
        "two containers": "tasks/press_demo.py",
        "in none of the scene camera's detections": "tasks/press_demo.py",
        "not taken for the selected container": "tasks/press_demo.py",
    }
    assert set(said_in) == set(g.FAULT_MARKS + g.SEVERAL_MARKS + g.NOT_CHOSEN_MARKS)
    for mark, name in said_in.items():
        assert mark in sources[name], "%r is no longer said in %s" % (mark, name)
    assert "two places pass" in sources["perception/scene_source.py"]  # (the scene locator's too)


def test_out_of_time_with_the_container_never_seen_the_last_words_say_that_was_what_was_missing():
    """A person who never brings the container into view gets the end a person who never gets into reach
    gets, and is told which it was: "when you are in place" is not what was missing."""
    never = g.Guide(budget_s=1.0)
    said = _drive(never, [(None, None)] * 2, t0=0.0, dt=1.5)
    assert said[-1][1].state == g.TIMED_OUT and said[-1][1].buttons == () and not never.saw_box
    assert said[-1][1].message == g.AGAIN_UNSEEN == "Container not seen in time. Press open again in front of it"
    console = g.Guide(budget_s=1.0, again="Out of time", again_unseen="Out of time: never seen")
    assert _drive(console, [(None, None)] * 2, t0=0.0, dt=1.5)[-1][1].message == "Out of time: never seen"
    # seen once and lost again by the end: it was seen, and what was missing is its place
    seen = g.Guide(budget_s=4.0)
    _drive(seen, [((0.95, 0.0), g.Reach(False, (-0.25, 0.0)))] * 3)
    assert seen.update(3.0, None, None, none=UNSEEN).message == g.UNSEEN
    assert seen.update(4.5, None, None, none=UNSEEN).message == g.AGAIN


def test_what_the_panel_is_given_to_show_is_one_short_message_and_never_tells_the_person_to_move_the_container():
    """The UI's panel shows one short message. Nothing said of a container no camera sees is longer than the
    longest the panel showed before it (the arm not at its rest pose: 63 characters). And the owner's rule
    (2026-11-11): the scene is never changed, only where the robot stands: the chair is what moves."""
    longest = len(g.unable(g.ARM_NOT_AT_REST).message)
    assert longest == 63
    for words in (g.UNSEEN, *g.NO_CURE.values(), g.AGAIN_UNSEEN, g.SEARCHING):
        assert len(words) <= longest, words
        assert g.unable(words).message == words  # (as the line-up's last panel says it: a sentence as it is)
        low = words.lower()
        assert not any(bad in low for bad in ("move the container", "move the box", "bring", "put the", "table"))
    assert "chair" in g.UNSEEN.lower()
