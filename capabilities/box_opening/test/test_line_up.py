"""The line-up before a run (line_up.LineUp): the panel's instruction, the question, and its three ends."""

import math

import pytest

from box_opening import guide, line_up
from box_opening.line_up import CANCELLED, GO, TIMED_OUT, BoxSource, LineUp, MockApproach
from box_opening.reach_zone import Zone

OPEN_PRESS = {"type": "button", "button_id": "open"}
CANCEL_PRESS = {"type": "button", "button_id": "cancel"}


class Clock:
    """Time that passes only as the loop sleeps."""

    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class Ctx:
    def __init__(self, clock, presses=()):
        self.clock = clock
        self.t0 = clock.t
        self.panels = []  # (seconds in, panel) of every panel sent
        self.presses = sorted(presses)  # (seconds in, event): what the user presses, and when
        self.cancel_at = None
        self.ui_sends = True

    def update_user_panel(self, panel_json=None):
        if self.ui_sends:
            self.panels.append((round(self.clock.t - self.t0, 2), panel_json))
        return self.ui_sends

    def poll_ui_event(self):
        if self.presses and self.clock.t - self.t0 >= self.presses[0][0]:
            return self.presses.pop(0)[1]
        return None

    def cancelled(self):
        return self.cancel_at is not None and self.clock.t - self.t0 >= self.cancel_at

    def messages(self):
        return list(dict.fromkeys(p["message"] for _t, p in self.panels))


def lined_up(source, presses=(), budget_s=40.0, live=True, cancel_at=None):
    clock = Clock()
    ctx = Ctx(clock, presses)
    ctx.cancel_at = cancel_at
    said = []
    lining = LineUp(ctx, lambda: live, source, budget_s, on_say=said.append, clock=clock, sleep=clock.sleep)
    return lining.run(), ctx, said, lining


class Still:
    """A box seen every tick where it stands; `reach` what the boundary says of it (a Reach, or None)."""

    def __init__(self, xy, reach):
        self.xy, self.reach = xy, reach

    def look(self, now):
        return guide.Fix(now, self.xy), self.reach


class Nothing:
    def look(self, now):
        return None, None


# -- the mock's approach, start to finish ------------------------------------------------------
def test_the_mock_box_comes_into_reach_and_open_starts_the_run():
    outcome, ctx, said, lining = lined_up(MockApproach(pace_s=12.0), presses=[(20.0, OPEN_PRESS)])
    assert outcome == GO and not lining.cancel_pressed
    words = ctx.messages()
    # 1.00 m out and 0.35 to the left; the nearest place inside is (0.75, 0.15): the larger miss is named first
    assert words[0] == "Move closer, about 25 cm"
    moves = [w.split(",")[0].replace(" a little", "") for w in words if w.startswith("Move")]
    turn = moves.index("Move left")
    assert turn >= 2 and set(moves[:turn]) == {"Move closer"} and set(moves[turn:]) == {"Move left"}
    assert words[-3:] == ["Hold still, checking", "Ready. Open the container?", "Opening the container"]
    assert [s.message for s in said] == words  # the node is told each new thing the panel says, once
    # the instruction's distance only ever shrinks as the box nears
    cm = [int(w.split("about ")[1].split(" ")[0]) for w in words if "about" in w and "closer" in w]
    assert cm == sorted(cm, reverse=True)


def test_the_question_has_open_and_cancel_and_every_other_panel_only_cancel():
    _outcome, ctx, _said, _lining = lined_up(MockApproach(pace_s=6.0), presses=[(15.0, OPEN_PRESS)])
    for _t, panel in ctx.panels:
        ids = [b["button_id"] for b in panel["buttons"]]
        assert ids == (["open", "cancel"] if panel["message"] == "Ready. Open the container?" else ["cancel"])
        assert panel["title"] == "Opening the container" and panel["progress"]["pausable"] is False
    ready = [t for t, p in ctx.panels if p["message"] == "Ready. Open the container?"]
    assert 6.0 + guide.STEADY_S * 0.8 <= ready[0] <= 6.0 + guide.STEADY_S + 0.5  # in place, then watched still


def test_open_is_no_yes_before_the_question_is_asked():
    """A press of Open that comes while the box is still on its way in (a stale tap) starts nothing."""
    outcome, ctx, _said, _lining = lined_up(MockApproach(pace_s=12.0), presses=[(3.0, OPEN_PRESS), (20.0, OPEN_PRESS)])
    assert outcome == GO
    went = [t for t, p in ctx.panels if p["message"] == "Opening the container"]
    assert went and went[0] >= 20.0


def test_a_box_in_place_at_once_is_still_watched_before_it_is_asked_of():
    outcome, ctx, _said, _lining = lined_up(MockApproach(pace_s=0.0), presses=[(5.0, OPEN_PRESS)])
    assert outcome == GO
    assert ctx.messages() == ["Hold still, checking", "Ready. Open the container?", "Opening the container"]


# -- its other ends ------------------------------------------------------------------------------
def test_cancel_on_the_panel_ends_it_and_is_said_to_be_the_panels():
    outcome, ctx, _said, lining = lined_up(MockApproach(pace_s=12.0), presses=[(4.0, CANCEL_PRESS)])
    assert outcome == CANCELLED and lining.cancel_pressed
    assert all(t <= 4.05 for t, _p in ctx.panels)


def test_the_orchestrators_cancel_ends_it_too():
    outcome, _ctx, _said, lining = lined_up(MockApproach(pace_s=12.0), cancel_at=2.0)
    assert outcome == CANCELLED and not lining.cancel_pressed


def test_out_of_time_it_says_to_press_open_again_and_holds_that_before_it_ends():
    outcome, ctx, _said, _lining = lined_up(Still((1.2, 0.0), guide.Reach(False, (-0.4, 0.0))), budget_s=10.0)
    assert outcome == TIMED_OUT
    assert ctx.messages() == ["Move closer, about 40 cm", "Press open again when you are in place"]
    last = [t for t, p in ctx.panels if p["message"].startswith("Press open again")]
    assert last[0] == pytest.approx(10.05, abs=0.06) and last[-1] - last[0] >= line_up.LINGER_S - line_up.REFRESH_S
    assert all(p["buttons"] == [] for _t, p in ctx.panels if p["message"].startswith("Press open again"))
    assert ctx.panels[-1][0] <= 10.0 + line_up.LINGER_S + 0.2


def test_open_pressed_once_the_time_is_out_starts_nothing():
    source = Still((0.65, 0.0), guide.Reach(True, None, True))
    outcome, _ctx, _said, _lining = lined_up(source, presses=[(10.5, OPEN_PRESS)], budget_s=10.0)
    assert outcome == TIMED_OUT


def test_with_no_box_in_sight_it_looks_and_never_asks():
    outcome, ctx, _said, _lining = lined_up(Nothing(), presses=[(3.0, OPEN_PRESS)], budget_s=8.0)
    assert outcome == TIMED_OUT
    # (a source that says nothing at all has not looked: its last words are that the container was not seen)
    assert ctx.messages() == ["Looking for the container", guide.AGAIN_UNSEEN]


class Unseen:
    """What watches has looked and has no container (`why_none`: its word on why), then from `seen_at` s in
    sees it at `xy`, outside the boundary by `to_zone`, until `lost_at`. As BoxSource does, it goes on
    saying what the boundary made of the place it last saw the container at, with no new sighting."""

    def __init__(self, why, seen_at=None, lost_at=None, xy=(1.05, 0.0), to_zone=(-0.30, 0.0)):
        self.why_none, self._why, self._reach = why, why, None
        self._seen_at, self._lost_at, self._xy, self._to_zone, self._t0 = seen_at, lost_at, xy, to_zone, None

    def look(self, now):
        self._t0 = now if self._t0 is None else self._t0
        t = now - self._t0
        if self._seen_at is None or t < self._seen_at or (self._lost_at is not None and t >= self._lost_at):
            self.why_none = self._why
            return None, self._reach
        self.why_none = ""  # seen: whatever it said before is no longer why (BoxSource)
        self._reach = guide.Reach(False, self._to_zone)
        return guide.Fix(now, self._xy, "scene"), self._reach


def test_a_container_no_camera_sees_is_told_to_the_person_with_the_chair_to_move_and_never_asked_of():
    """Owner 2026-11-19: a box no camera sees at all is guidance too, "move so the container is in front of
    the arm". The same loop, the same Cancel, the same time: what the watch says of seeing none becomes what
    the person is told, on the panel with Cancel alone, and a press of Open is no yes."""
    source = Unseen("wrist camera: no counter in 1 frame (no level plane wider than a container)")
    outcome, ctx, said, lining = lined_up(source, presses=[(3.0, OPEN_PRESS)], budget_s=8.0)
    assert outcome == TIMED_OUT and not lining.guide.saw_box
    # (it is still looking until its watch has had none for as long as a lost container is given)
    assert ctx.messages() == [guide.SEARCHING, guide.UNSEEN, guide.AGAIN_UNSEEN]
    assert ctx.messages()[1] == "The arm cannot see the container. Move the chair in front of it"
    unseen = [(t, p) for t, p in ctx.panels if p["message"] == guide.UNSEEN]
    assert unseen[0][0] == pytest.approx(guide.LOST_S, abs=0.06)
    assert all([b["button_id"] for b in p["buttons"]] == ["cancel"] for _t, p in unseen)
    assert all(p["progress"] == {"phase": "looking", "pct": 0.0, "pausable": False} for _t, p in unseen)
    # one line each, for the console too
    assert [s.message for s in said] == [guide.SEARCHING, guide.UNSEEN, guide.AGAIN_UNSEEN]
    # ... Cancel ends it as it ends any line-up, and so does the orchestrator's cancel
    outcome, _ctx, _said, lining = lined_up(source, presses=[(2.0, CANCEL_PRESS)])
    assert outcome == CANCELLED and lining.cancel_pressed
    outcome, _ctx, _said, _lining = lined_up(source, cancel_at=2.0)
    assert outcome == CANCELLED


def test_the_ordinary_guidance_takes_over_the_moment_a_camera_sees_it_and_gives_way_again_when_it_is_lost():
    source = Unseen("scene segmenter: saw no food container in 12 frames", seen_at=2.0, lost_at=4.0)
    outcome, ctx, _said, lining = lined_up(source, budget_s=8.0)
    assert outcome == TIMED_OUT and lining.guide.saw_box
    assert ctx.messages() == [guide.SEARCHING, guide.UNSEEN, "Move closer, about 30 cm", guide.AGAIN]
    first = {m: t for t, m in reversed([(t, p["message"]) for t, p in ctx.panels])}
    assert first["Move closer, about 30 cm"] == pytest.approx(2.0, abs=0.06)  # at once, at the first sighting
    # lost at 4 s: unseen again once it has not been seen for LOST_S (one frame without it is not that)
    back = [t for t, p in ctx.panels if p["message"] == guide.UNSEEN and t > 2.0]
    assert back and back[0] == pytest.approx(4.0 - line_up.POLL_S + guide.LOST_S, abs=0.11)


def test_with_no_ui_the_unseen_container_is_one_line_and_an_n_is_the_cancel():
    """The owner's goals are sent by hand: the same words on the console, a y to nothing asked is no yes, an
    n is the cancel, and out of time it says what was missing in the console's own words."""
    source = Unseen("wrist camera: nothing on the counter is the container (1 thing at lid height)")
    outcome, said, lining, unasked = answered(
        source, typed=[(1.0, "y")], budget_s=6.0, say_min_s=1.0, again="Out of time", again_unseen="Never seen"
    )
    assert outcome == TIMED_OUT and unasked == pytest.approx([1.0], abs=0.06) and not lining.said_yes
    assert [m for _t, m in said] == [guide.SEARCHING, guide.UNSEEN, "Never seen"]
    outcome, said, lining, _unasked = answered(source, typed=[(2.0, "n")], say_min_s=1.0)
    assert outcome == CANCELLED and lining.said_no and [m for _t, m in said] == [guide.SEARCHING, guide.UNSEEN]


def test_a_place_the_boundary_cannot_judge_yet_is_being_checked_not_refused():
    outcome, ctx, _said, _lining = lined_up(Still((0.6, 0.0), None), budget_s=5.0)
    assert outcome == TIMED_OUT and ctx.messages()[0] == "Hold still, checking"


# -- what goes to the UI ---------------------------------------------------------------------------
def test_an_unchanged_panel_is_sent_again_at_its_refresh_and_a_moving_bar_no_oftener_than_its_floor():
    _outcome, ctx, _said, _lining = lined_up(Still((0.65, 0.0), guide.Reach(True, None, True)), budget_s=6.0)
    ready = [t for t, p in ctx.panels if p["message"] == "Ready. Open the container?"]
    gaps = [round(b - a, 2) for a, b in zip(ready, ready[1:], strict=False)]
    assert gaps and all(g == pytest.approx(line_up.REFRESH_S, abs=0.06) for g in gaps)
    _outcome, ctx, _said, _lining = lined_up(MockApproach(pace_s=30.0), budget_s=8.0)  # the bar moves every tick
    sends = [(t, p["message"]) for t, p in ctx.panels if t < 8.0]
    bar_only = [b[0] - a[0] for a, b in zip(sends, sends[1:], strict=False) if a[1] == b[1]]
    assert bar_only and min(bar_only) >= line_up.SEND_MIN_S - 1e-6  # new words go at once, a moved bar waits
    assert len(sends) <= 8.0 / line_up.SEND_MIN_S + len({m for _t, m in sends})


def test_without_the_ui_nothing_is_sent():
    outcome, ctx, said, _lining = lined_up(MockApproach(pace_s=1.0), budget_s=4.0, live=False)
    assert outcome == TIMED_OUT and ctx.panels == [] and said  # the node still hears what it would say


def test_a_press_takes_the_bridges_panel_down_and_the_next_tick_sends_it_again():
    source = Still((1.2, 0.0), guide.Reach(False, (-0.4, 0.0)))
    _outcome, ctx, _said, _lining = lined_up(source, presses=[(2.52, OPEN_PRESS)], budget_s=4.0)
    times = [t for t, p in ctx.panels if p["message"].startswith("Move closer")]
    assert any(2.5 <= t <= 2.6 for t in times)  # off the 1 s refresh: sent again because of the press


# -- a goal with no UI: the same line-up, said on the console and answered y or n ------------------
class Typed(line_up.Answers):
    """Answers that come as the clock passes: (seconds in, what was typed)."""

    def __init__(self, clock, typed):
        super().__init__()
        self._clock, self._t0, self._typed = clock, clock.t, sorted(typed)

    def poll(self):
        while self._typed and self._clock.t - self._t0 >= self._typed[0][0]:
            self.put(self._typed.pop(0)[1])
        return super().poll()


def answered(source, typed=(), budget_s=40.0, say_min_s=0.0, **kw):
    """A line-up with no UI (no panel goes out), its answers `typed`: (outcome, what it said, the LineUp,
    how often a y came with nothing asked)."""
    clock = Clock()
    ctx = Ctx(clock)
    said, unasked = [], []
    lining = LineUp(
        ctx,
        lambda: False,
        source,
        budget_s,
        on_say=lambda say: said.append((round(clock.t - ctx.t0, 2), say.message)),
        clock=clock,
        sleep=clock.sleep,
        answers=Typed(clock, typed),
        say_min_s=say_min_s,
        on_unasked=lambda: unasked.append(round(clock.t - ctx.t0, 2)),
        **kw,
    )
    outcome = lining.run()
    assert ctx.panels == []  # nothing went to a UI that is not there
    return outcome, said, lining, unasked


def test_what_is_typed_is_an_answer_only_where_it_is_y_or_n():
    assert [line_up.read_answer(t) for t in ("y", " Y\n", "yes", "YES ")] == ["y"] * 4
    assert [line_up.read_answer(t) for t in ("n", "N", " no\n")] == ["n"] * 3
    assert [line_up.read_answer(t) for t in ("", "ok", "yy", "go", None, 1)] == [None] * 6  # never a yes
    answers = line_up.Answers()
    assert answers.put("maybe") is None and answers.poll() is None
    assert (answers.put("y"), answers.put("no")) == ("y", "n") and (answers.poll(), answers.poll()) == ("y", "n")
    answers.put("y")
    answers.clear()  # an answer is to the action that is running, never to the next
    assert answers.poll() is None


def test_with_no_ui_the_same_line_up_is_said_and_a_y_to_the_question_lets_the_run_go():
    """Owner 2026-11-17: "don't end, just tell you how far to move until it's close enough then asking if
    ready y/n." The mock's approach with no UI: every instruction the panel would show is said, then the
    question, and a y typed while it stands is Open's press."""
    outcome, said, lining, unasked = answered(MockApproach(pace_s=12.0), typed=[(20.0, "y")])
    assert outcome == GO and lining.said_yes and not lining.cancel_pressed and unasked == []
    words = [m for _t, m in said]
    assert words[0] == "Move closer, about 25 cm" and any(w.startswith("Move left") for w in words)
    assert words[-3:] == ["Hold still, checking", "Ready. Open the container?", "Opening the container"]
    assert said[-1][0] == pytest.approx(20.0, abs=0.06)
    # the same approach on the UI: the same words, and Open (no answer) is no said_yes
    outcome, ctx, ui_said, lining = lined_up(MockApproach(pace_s=12.0), presses=[(20.0, OPEN_PRESS)])
    assert outcome == GO and not lining.said_yes and [s.message for s in ui_said] == words


def test_a_y_typed_before_the_question_is_no_yes_and_is_said_to_be_none():
    """Nothing moves without the person's yes AFTER the guidance: a y typed ahead, while the box is still on
    its way in, starts nothing, and the person is told it was not taken."""
    outcome, said, _lining, unasked = answered(
        MockApproach(pace_s=12.0), typed=[(3.0, "y"), (5.0, "yes")], budget_s=30.0
    )
    assert outcome == TIMED_OUT and unasked == pytest.approx([3.0, 5.0], abs=0.06)
    assert "Opening the container" not in [m for _t, m in said]

    # ... nor is a y to a box that was Ready and has since gone from view (Guide.press: seen within FRESH_S)
    class Gone:
        def look(self, now):
            return (guide.Fix(now, (0.65, 0.0)) if now < 104.0 else None), guide.Reach(True, None, True)

    outcome, said, _lining, unasked = answered(Gone(), typed=[(6.0, "y")], budget_s=10.0)
    assert outcome == TIMED_OUT and unasked == pytest.approx([6.0], abs=0.06)
    assert "Ready. Open the container?" in [m for t, m in said if t < 4.0]  # it had been asked, and was gone


def test_an_n_is_cancel_at_every_moment_of_the_line_up():
    for at in (0.5, 6.0, 16.0):  # as it moves in, as it is told to hold still, while the question stands
        outcome, said, lining, _unasked = answered(MockApproach(pace_s=12.0), typed=[(at, "n")])
        assert outcome == CANCELLED and lining.cancel_pressed and lining.said_no, at
        assert all(t <= at + 0.06 for t, _m in said)
    # ... and while it says what no waiting cures (the arm not at its rest pose)
    stuck = Still((0.65, 0.0), None)
    stuck.unable = guide.ARM_NOT_AT_REST
    outcome, _said, lining, _unasked = answered(stuck, typed=[(1.0, "n")])
    assert outcome == CANCELLED and lining.said_no
    # an n on the UI's line-up is the same cancel
    clock = Clock()
    lining = LineUp(
        Ctx(clock), lambda: True, MockApproach(pace_s=12.0), 40.0, clock=clock, sleep=clock.sleep,
        answers=Typed(clock, [(2.0, "n")]),
    )  # fmt: skip
    assert lining.run() == CANCELLED and lining.said_no


def test_out_of_time_with_no_ui_it_says_the_words_it_was_given():
    outcome, said, lining, _unasked = answered(
        Still((1.2, 0.0), guide.Reach(False, (-0.4, 0.0))), budget_s=10.0, again="Out of time. Start it again"
    )
    assert outcome == TIMED_OUT and [m for _t, m in said] == ["Move closer, about 40 cm", "Out of time. Start it again"]
    assert lining.guide.instruction == "Move closer, about 40 cm" and lining.guide.saw_box


def test_a_distance_that_only_shrinks_is_said_once_a_second_and_everything_else_at_once():
    """The console is read by a person driving a chair: "one short line each time what it says changes,
    never five a second". A box driven in changes the distance every few ticks. With say_min_s the same
    instruction's new distance waits out the second (the newest is the one said), as does "hold still"
    where it takes an instruction's place; a new direction and the question are said the tick they come."""
    fast = dict(pace_s=3.0)  # 0.65 m in 3 s: a new 5 cm every quarter second
    _outcome, every, _lining, _unasked = answered(MockApproach(**fast), typed=[(8.0, "y")])
    outcome, said, _lining, _unasked = answered(MockApproach(**fast), typed=[(8.0, "y")], say_min_s=1.0)
    assert outcome == GO
    moves = [(t, m) for t, m in said if m.startswith("Move")]
    all_moves = [(t, m) for t, m in every if m.startswith("Move")]
    assert len(all_moves) >= 6 and len(moves) < len(all_moves)
    kind = lambda m: m.split(",")[0].replace(" a little", "")  # noqa: E731
    for (t0, m0), (t1, m1) in zip(moves, moves[1:], strict=False):
        assert kind(m0) != kind(m1) or t1 - t0 >= 1.0 - 1e-6, (t0, m0, t1, m1)
    # never five a second. Held: a distance, and "hold still" after an instruction (the second at most). The
    # question and what follows it come when they did without it
    assert all(sum(1 for t, _m in said if t0 <= t < t0 + 1.0) < 5 for t0, _m in said)
    others = lambda lines: [(t, m) for t, m in lines if not m.startswith(("Move", "Hold"))]  # noqa: E731
    assert others(said) == others(every) and others(said)[0][1] == "Ready. Open the container?"
    hold = lambda lines: next(t for t, m in lines if m.startswith("Hold"))  # noqa: E731
    assert hold(every) <= hold(said) <= hold(every) + 1.0
    first_of = lambda lines, word: next(t for t, m in lines if m.startswith(word))  # noqa: E731
    assert first_of(said, "Move left") == first_of(every, "Move left")  # a new direction is said at once
    # a chair that stops is still told the newest distance, once the second has passed
    still = Still((0.9, 0.0), guide.Reach(False, (-0.2, 0.0)))
    looks = iter([(-0.4, 0.0), (-0.3, 0.0)])

    def look(now, _look=still.look):
        fix, _reach = _look(now)
        return fix, guide.Reach(False, next(looks, (-0.2, 0.0)))

    still.look = look
    _outcome, said, _lining, _unasked = answered(still, budget_s=3.0, say_min_s=1.0)
    assert [m for _t, m in said][:2] == ["Move closer, about 40 cm", "Move closer, about 20 cm"]
    assert said[1][0] == pytest.approx(1.0, abs=0.06)


class Edge:
    """A box on the boundary's edge: each wrist frame (five a second) reads it the other side of the margin."""

    def look(self, now):
        inside = int(round((now - 100.0) / 0.2 - 0.5)) % 2 == 0
        reach = guide.Reach(True, None, None) if inside else guide.Reach(False, (-0.02, 0.0), None)
        return guide.Fix(now, (0.70, 0.0), "wrist"), reach


def test_a_box_on_the_boundarys_edge_is_one_line_on_the_console_not_five_a_second():
    """The review's replay, 2026-11-17: with the box reading inside and 2 cm outside on alternate frames
    the console got "Hold still, checking" / "Move closer a little" 51 times in 10 s. It is told the move
    that is still to make, once, and then that the time is up."""
    outcome, said, _lining, _unasked = answered(Edge(), budget_s=10.0, say_min_s=1.0, again="Out of time")
    assert outcome == TIMED_OUT
    assert [m for _t, m in said] in (
        ["Move closer a little", "Out of time"],
        ["Hold still, checking", "Move closer a little", "Out of time"],  # (its first frame read inside)
    )

    # the console's own hold, whatever the guide says: "hold still" and an instruction that take each other's
    # place every 0.6 s (longer than the guide's hold) are a line a second at the most, never two
    class Slow:
        def look(self, now):
            inside = int((now - 100.0) / 0.6) % 2 == 1
            reach = guide.Reach(True, None, None) if inside else guide.Reach(False, (-0.02, 0.0), None)
            return guide.Fix(now, (0.70, 0.0), "wrist"), reach

    _outcome, every, _lining, _unasked = answered(Slow(), budget_s=10.0)
    _outcome, said, _lining, _unasked = answered(Slow(), budget_s=10.0, say_min_s=1.0)
    told = [t for t, m in said if m.startswith(("Move", "Hold"))]  # (the time's end is said when it comes)
    assert len(every) > len(said) and len(told) >= 6
    assert all(b - a >= 1.0 - 1e-6 for a, b in zip(told, told[1:], strict=False))
    # the question is never held, nor its being taken away: a person must not answer what is no longer asked
    outcome, said, _lining, _unasked = answered(MockApproach(pace_s=0.0), typed=[(3.0, "y")], say_min_s=5.0)
    assert outcome == GO and [m for _t, m in said][:2] == ["Hold still, checking", "Ready. Open the container?"]
    assert said[1][0] < 2.0  # (five seconds were not waited out)


# -- the mock's box --------------------------------------------------------------------------------
def test_the_mock_box_walks_its_path_at_one_speed_and_stops_at_its_end():
    box = MockApproach(pace_s=10.0)
    assert box.at(0.0) == (1.00, 0.35) and box.at(10.0) == box.at(99.0) == (0.70, 0.0)
    assert box.at(-1.0) == (1.00, 0.35)
    # 0.30 m in, then 0.35 m across: the corner is 0.30 / 0.65 of the way
    assert box.at(10.0 * 0.30 / 0.65) == pytest.approx((0.70, 0.35))
    assert box.at(2.0) == pytest.approx((1.00 - 0.13, 0.35))
    assert MockApproach(pace_s=0.0).at(0.0) == (0.70, 0.0)
    zone = line_up.mock_zone()
    assert zone.done() and zone.inside((0.70, 0.0)) and not zone.inside((1.00, 0.35))


def test_the_deployments_bound_holds_the_mock_box_to_it_too():
    """line_up_far_m: what works further out than it is not in the boundary, so the user is brought nearer."""
    near = MockApproach(pace_s=0.0, path=((0.75, 0.0),), far_m=0.70)
    fix, reach = near.look(0.0)
    # within 0.70 m the middle row works out to 0.70 itself; its last cell that is inside is the one at 0.65
    assert fix.xy == (0.75, 0.0) and not reach.inside and reach.to_zone == pytest.approx((-0.10, 0.0))
    _fix, reach = MockApproach(pace_s=0.0, path=((0.75, 0.0),)).look(0.0)
    assert reach.inside


# -- a real run's source: the watch and the probe as children ------------------------------------------
class FakeChild:
    """line_up.Child without a process: what it "says" is queued by the test, what it is sent is kept."""

    made = []

    def __init__(self, argv, env, log=None, clock=None):
        self.argv, self.env, self.sent, self.stopped, self._got = list(argv), env, [], False, []
        self.lines = []  # its own log's last lines
        FakeChild.made.append(self)

    def says(self, t, msg):
        self._got.append((t, msg))

    def messages(self):
        got, self._got = self._got, []
        return got

    def send(self, msg):
        self.sent.append(msg)

    def stop(self, wait_s=None, quit_s=0.0):
        self.stopped, self.quit_s = True, quit_s

    def alive(self):
        return not self.stopped

    def last_lines(self, wait_s=None):
        return list(self.lines)


def _band_zone(lid_z=0.32):
    """Works 0.50 to 0.80 m out and 0.20 m to either side, for a lid at `lid_z`."""
    zone = Zone.blank((0.0, 1.0), (-0.5, 0.5), lid_z=lid_z, key="k")
    for i in range(zone.cells.shape[0]):
        for j in range(zone.cells.shape[1]):
            x, y = zone.centre(i, j)
            zone.set(i, j, 0.475 < x < 0.825 and abs(y) < 0.225)
    return zone


def _source(far_m=None):
    FakeChild.made = []
    source = BoxSource(
        ["box_watch"], lambda urdf, root: ["reach_probe", urdf, root], {"E": "1"}, far_m=far_m, spawn=FakeChild
    )
    source.watch.says(0.0, {"robot": {"urdf": "/state/arm.urdf", "root": "arm_mount"}})
    assert source.look(0.0) == (None, None)
    return source, source.watch, source.probe


def _fix(xyz, age_s=0.1, rough=False, table_z=0.22, by=None):
    """A sighting as the watch says it: the scene camera's, or with `by` "wrist" the wrist camera's."""
    fix = {"xyz": list(xyz), "table_z": table_z, "age_s": age_s, "rough": rough, "score": 0.9}
    return {"fix": dict(fix, by=by) if by else fix}


def test_the_probe_starts_once_the_watch_has_said_which_robot_this_is():
    FakeChild.made = []
    source = BoxSource(["box_watch", "--x"], lambda urdf, root: ["reach_probe", urdf, root], {}, spawn=FakeChild)
    assert source.probe is None and [c.argv for c in FakeChild.made] == [["box_watch", "--x"]]
    assert source.look(0.0) == (None, None)  # nothing said yet: looking
    source.watch.says(0.5, {"robot": {"urdf": "/state/arm.urdf", "root": "arm_mount"}})
    source.look(0.6)
    assert source.probe.argv == ["reach_probe", "/state/arm.urdf", "arm_mount"]
    said = []
    blind = BoxSource(["box_watch"], lambda urdf, root: ["reach_probe"], {}, log=said.append, spawn=FakeChild)
    blind.watch.says(0.5, {"robot": {"urdf": None, "root": "arm_mount"}})
    blind.look(0.6)
    assert blind.probe is None and "URDF was not heard" in said[0]  # no kinematics: no boundary, and it says so


def test_a_sighting_is_a_fix_as_old_as_it_says_and_goes_to_the_probe():
    source, watch, probe = _source()
    watch.says(10.0, _fix((0.95, 0.10, 0.315), age_s=0.2))
    fix, reach = source.look(10.05)
    assert fix == guide.Fix(pytest.approx(9.8), (0.95, 0.10), "scene") and reach is None  # no boundary yet
    assert probe.sent == [{"box": [0.95, 0.10, 0.315], "table_z": 0.22, "prove": False}]
    assert source.look(10.1) == (None, None) and len(probe.sent) == 1  # nothing new: nothing sent again
    watch.says(10.2, {"none": "scene segmenter: saw no food container in 5 frames"})
    source.look(10.25)
    assert source.why_none.startswith("scene segmenter")


def test_why_no_box_is_seen_is_the_watchs_newest_word_and_a_later_sighting_takes_it_back():
    """why_none goes into the error of a line-up that ran out of time: it must not say the container was
    not seen where it was seen since, only not brought into reach."""
    source, watch, _probe = _source()
    watch.says(1.0, {"none": "wrist camera: no lid-sized top on the table in view"})
    watch.says(1.2, {"none": "wrist camera: the lid is cut by the view's edge"})
    source.look(1.3)
    assert source.why_none == "wrist camera: the lid is cut by the view's edge"  # the newest
    watch.says(2.0, _fix((0.95, 0.0, 0.315), by="wrist"))
    source.look(2.1)
    assert source.why_none == ""  # seen since: that reason is no longer why
    watch.says(3.0, {"none": "wrist camera: no frame"})
    source.look(3.1)
    assert source.why_none == "wrist camera: no frame"


def test_a_line_up_that_begins_because_the_run_saw_no_container_says_so_before_its_watch_has_said_a_word():
    """A run whose own look found no container asks to be guided, and its watch then takes seconds to say
    anything (the wrist turns to its view first). The run has looked: the person is told at once that the
    arm cannot see the container, not "Looking for the container" through that turn. The run's word stands
    in for the watch's until the watch says its own, or sees the container. Nothing else of the source
    starts from a container having been seen: no fix, no place sent to the probe, nothing refused."""
    FakeChild.made = []
    why = "no camera sees the container (scene camera: scene segmenter: saw no food container in 14 frames)"
    source = BoxSource(["box_watch"], lambda urdf, root: ["reach_probe", urdf, root], {}, spawn=FakeChild, unseen=why)
    assert source.why_none == why and source.none_kind == guide.K_UNSEEN and source.looked
    assert source.look(0.0) == (None, None)
    source.watch.says(0.1, {"robot": {"urdf": "/state/arm.urdf", "root": "arm_mount"}})
    assert source.look(0.2) == (None, None) and source.why_none == why
    clock = Clock()
    ctx = Ctx(clock)
    lining = LineUp(ctx, lambda: True, source, 2.0, clock=clock, sleep=clock.sleep)
    assert lining.run() == TIMED_OUT
    assert ctx.panels[0][0] == 0.0 and ctx.panels[0][1]["message"] == guide.UNSEEN  # its first panel
    assert [b["button_id"] for b in ctx.panels[0][1]["buttons"]] == ["cancel"]
    assert ctx.messages() == [guide.UNSEEN, guide.AGAIN_UNSEEN]
    # the watch's own word takes its place, and a sighting takes that back: seen since, it is no longer why
    source.watch.says(3.0, {"none": "wrist camera: no counter in 1 frame"})
    source.look(3.1)
    assert source.why_none == "wrist camera: no counter in 1 frame" and source.none_kind == guide.K_UNSEEN
    source.watch.says(3.5, _fix((1.05, 0.0, 0.315), by="wrist"))
    fix, _reach = source.look(3.6)
    assert fix.xy == (1.05, 0.0) and source.why_none == "" and source.none_kind == "" and not source.looked
    assert source.probe.sent == [{"box": [1.05, 0.0, 0.315], "table_z": 0.22, "prove": False}]  # its first box
    # a source that begins for any other reason says nothing until its watch has
    plain = _source()[0]
    assert plain.why_none == "" and plain.none_kind == "" and not plain.looked


def test_reach_probe_is_started_at_the_first_sighting_where_the_line_up_began_with_no_container_seen():
    """reach_probe does nothing before a sighting: the lid's height names its grid. A line-up that begins
    because the run saw no container may never have one, and as first written its reach_probe was started at
    the watch's first word all the same: one that could not start, or died, ended the action
    line_up_unavailable, NOT recoverable, of a run that at 21aa7cc ended a recoverable not_found (review of
    2026-11-19). For such a line-up it is started at the first sighting. Every other line-up starts it at
    the watch's first word, as before: its start-up runs beside the wrist's turn."""
    robot = {"robot": {"urdf": "/state/arm.urdf", "root": "arm_mount"}}

    def spawn(argv, env, log=None, clock=None):
        if argv[0] == "reach_probe":
            raise OSError("reach_probe is not there to start")
        return FakeChild(argv, env, log, clock)

    probe = lambda urdf, root: ["reach_probe", urdf, root]  # noqa: E731
    unseen = BoxSource(["box_watch"], probe, {}, spawn=spawn, unseen="no camera sees the container (x)")
    unseen.watch.says(0.0, robot)
    for t in (0.1, 5.0, 60.0):  # the whole wait, with no container in view: nothing started, nothing gone
        unseen.watch.says(t, {"none": "wrist camera: no counter in 1 frame"})
        assert unseen.look(t + 0.05) == (None, None) and unseen.probe is None and not unseen.gone()
    unseen.watch.says(61.0, _fix((1.05, 0.0, 0.315), by="wrist"))
    with pytest.raises(OSError):  # the first sighting: only now is what judges its place needed, and missing
        unseen.look(61.1)
    plain = BoxSource(["box_watch"], probe, {}, spawn=spawn)
    plain.watch.says(0.0, robot)
    with pytest.raises(OSError):  # as before: at the watch's first word
        plain.look(0.1)
    # ... and one that starts is told the first sighting, as any probe is told every sighting
    FakeChild.made = []
    started = BoxSource(["box_watch"], probe, {}, spawn=FakeChild, unseen="no camera sees the container (x)")
    started.watch.says(0.0, robot)
    started.look(0.1)
    assert started.probe is None and len(FakeChild.made) == 1
    started.watch.says(2.0, _fix((1.05, 0.0, 0.315), by="wrist"))
    started.look(2.1)
    assert started.probe.argv == ["reach_probe", "/state/arm.urdf", "arm_mount"]
    assert started.probe.sent == [{"box": [1.05, 0.0, 0.315], "table_z": 0.22, "prove": False}]
    # ... the same of an unseen container's line-up after a yes of its goal (`looked` False: below)
    after = BoxSource(["box_watch"], probe, {}, spawn=spawn, unseen="no camera sees the container (x)", looked=False)
    after.watch.says(0.0, robot)
    assert after.look(0.1) == (None, None) and after.probe is None and not after.gone()


def test_after_a_yes_the_runs_look_that_found_no_container_is_not_said_before_the_watch_has_looked():
    """The first guidance of a goal for a container no camera saw says so from its first panel: the run has
    looked (`unseen`). After a yes of that goal it is otherwise (`looked` False, the engine's): the watch
    HAD the container a moment ago where the person now stands, and "The arm cannot see the container. Move
    the chair in front of it" from the first panel would tell a person who is in place to move. There the
    watch looks first, as at any line-up's start, and says so only once it has had none for guide.LOST_S."""
    why = "no camera sees the container (scene camera: x)"
    first = BoxSource(["box_watch"], lambda urdf, root: ["reach_probe"], {}, spawn=FakeChild, unseen=why)
    assert (first.looked, first.why_none, first.none_kind) == (True, why, guide.K_UNSEEN)
    after = BoxSource(["box_watch"], lambda urdf, root: ["reach_probe"], {}, spawn=FakeChild, unseen=why, looked=False)
    assert (after.looked, after.why_none, after.none_kind) == (False, "", "")
    for source, begins in ((first, guide.UNSEEN), (after, guide.SEARCHING)):
        clock = Clock()
        ctx = Ctx(clock)
        said = iter([{"none": "wrist camera: no counter in 1 frame"}] * 2 + [WRIST_FIX] * 200)

        def look(now, source=source, real=source.look, said=said):
            source.watch.says(now, next(said))
            return real(now)

        source.look = look
        LineUp(ctx, lambda: True, source, 3.0, clock=clock, sleep=clock.sleep).run()
        assert ctx.messages()[0] == begins and guide.UNSEEN not in ctx.messages()[1:]


ROBOT = {"robot": {"urdf": "/state/arm.urdf", "root": "arm_mount"}}
WRIST_FIX = {"fix": {"xyz": [0.65, 0.0, 0.315], "table_z": 0.216, "age_s": 0.05, "rough": False, "by": "wrist"}}


def watched(lines, budget_s=20.0, unseen="", step_s=0.2, then=None, far_m=None):
    """A line-up on the real BoxSource, Guide and LineUp whose watch says `lines`, one every `step_s`, and
    then goes on saying `then` at that pace (None: the last of `lines`; False: nothing more): (outcome, the
    panel's messages in order, the source, [(seconds in, message)] as each was first shown, the LineUp)."""
    FakeChild.made = []
    source = BoxSource(
        ["box_watch"], lambda urdf, root: ["reach_probe", urdf, root], {}, spawn=FakeChild, unseen=unseen, far_m=far_m
    )
    clock = Clock()
    ctx = Ctx(clock)
    pending = [(clock.t + step_s * i, ln) for i, ln in enumerate(lines)]
    again = [pending[-1][0], lines[-1] if then is None else then]
    real_look = source.look

    def look(now):
        while pending and pending[0][0] <= now:
            t, msg = pending.pop(0)
            source.watch.says(t, msg)
        if not pending and again[1] and now - again[0] >= step_s:
            again[0] = now
            source.watch.says(now, again[1])
        return real_look(now)

    source.look = look
    lining = LineUp(ctx, lambda: True, source, budget_s, clock=clock, sleep=clock.sleep)
    outcome = lining.run()
    first = {}
    for t, panel in ctx.panels:
        first.setdefault(panel["message"], t)
    return outcome, ctx.messages(), source, sorted((t, m) for m, t in first.items()), lining


@pytest.mark.parametrize(
    "none",
    [
        "wrist camera: no new wrist frame",
        "wrist camera: no camera pose for the frame (TF)",
        "scene camera streams missing: color, depth",
        "the scene segmenter cannot be read: no such topic",
        "scene segmenter: no frame newer than the look in 1.0 s",
    ],
)
def test_a_camera_that_delivers_nothing_is_said_as_that_and_ends_the_line_up_never_as_move_the_chair(none):
    """The review of 2026-11-19: every "none" a watch says, a fault among them, read "The arm cannot see the
    container. Move the chair in front of it" for the whole line-up, of every line-up (before a run,
    carried, in the middle of one): five minutes of a person driving a wheelchair at a camera that is not
    working. Now it is said as what it is, with no word of moving, and standing guide.NO_CURE_S it ends the
    line-up (UNABLE, as an arm that is not at rest ends it: its last LINGER_S with no button)."""
    outcome, messages, source, first, _lining = watched([ROBOT, {"none": none}], budget_s=60.0)
    assert outcome == line_up.UNABLE and source.uncured == guide.K_FAULT
    assert source.unable == guide.NO_CURE[guide.K_FAULT] == "The arm's vision is not working"
    assert messages == [guide.SEARCHING, "The arm's vision is not working"]  # never the cannot-see sentence
    (t_said,) = (t for t, m in first if m == "The arm's vision is not working")
    assert t_said == pytest.approx(0.2 + guide.LOST_S, abs=0.11)  # once it has stood, not on its first word
    ended = _lining._clock() - 100.0
    assert ended == pytest.approx(0.2 + guide.NO_CURE_S, abs=0.3) and ended < 15.0  # ten seconds, not the wait


NOT_IN_VIEW = [
    # what a watch says of a container that is merely not in view, in its own words or with their kind
    {"none": "wrist camera: no counter in 1 frame (no level plane wider than a container)"},
    {"none": "wrist camera: nothing on the counter is the container (1 thing at lid height)"},
    {"none": "wrist camera: the container may stand at [0.70, 0.30], cut off by the view (x): look again"},
    {"none": "scene segmenter: saw no food container in 14 frames"},
    {"none": "scene segmenter: saw no food container in 2 frames", "kind": "unseen"},
    # ... a selected id that is in none of the scene camera's frames among them: the container may simply
    # be out of its view (box_watch says the words alone, the run's own scene watch their kind with them)
    {"none": "scene segmenter: the selected container (fc_9) was in none of its 70 frames"},
    {"none": "scene segmenter: the selected container (fc_9) was in none of its 2 frames", "kind": "unseen"},
    # ... and a segmenter that is behind the look of a third of a second (press_demo's scene_kind)
    {"none": "scene segmenter: no frame newer than the look in 0.3 s", "kind": "unseen"},
]


@pytest.mark.parametrize("none", NOT_IN_VIEW, ids=[n["none"][:44] + "/" + n.get("kind", "") for n in NOT_IN_VIEW])
@pytest.mark.parametrize("seeded", [False, True])
def test_a_container_that_is_merely_not_in_view_never_ends_a_line_up_only_its_time_does(none, seeded):
    """The one state guidance is for. Whatever else ends a line-up, a container that is not in view never
    does: the person is driving the chair to bring it there, for the whole of the wait (five minutes on
    the UI). Nothing held that: with every kind of "none", this one included, made to end a line-up after
    ten seconds, every test of a626985 passed (review of 2026-11-19, the second). Its words do not end it,
    its kind does not, and where the selected id is merely absent from the scene camera's frames, as it is
    for a container that is not in its view yet, that is this state too, not "Choose it again"."""
    why = "no camera sees the container (scene camera: x)" if seeded else ""
    outcome, messages, source, _first, lining = watched([ROBOT, none], budget_s=300.0, step_s=1.0, unseen=why)
    assert outcome == TIMED_OUT and source.unable == "" and source.uncured == "" and source.standing == ""
    assert lining.guide.uncured == "" and lining._clock() - 100.0 > 300.0
    assert messages == ([] if seeded else [guide.SEARCHING]) + [guide.UNSEEN, guide.AGAIN_UNSEEN]


def test_a_line_up_whose_time_runs_out_first_ends_on_the_cameras_sentence_not_on_press_open_again():
    """A line-up with fewer seconds than guide.NO_CURE_S (the orchestrator's 120 s): its last panel is what
    stood on it, with no button, and its guide says which state that was (`uncured`) for the engine's end.
    One whose watch simply has no container ends in the words it always had."""
    outcome, messages, source, _first, lining = watched(
        [ROBOT, {"none": "wrist camera: no new wrist frame"}], budget_s=4.0
    )
    assert outcome == TIMED_OUT and lining.guide.uncured == guide.K_FAULT and source.uncured == ""
    assert messages == [guide.SEARCHING, "The arm's vision is not working"]
    outcome, messages, _source, _first, lining = watched(
        [ROBOT, {"none": "wrist camera: no counter in 1 frame"}], budget_s=4.0
    )
    assert outcome == TIMED_OUT and lining.guide.uncured == ""
    assert messages == [guide.SEARCHING, guide.UNSEEN, guide.AGAIN_UNSEEN]


def test_a_container_nothing_ties_to_the_selection_is_said_as_that_and_ends_the_line_up():
    """The review's worst case: a selected id that is in none of the scene camera's detections, the watch's
    wrist with a lid in every frame and taking none of them, and the panel on "The arm cannot see the
    container. Move the chair in front of it" until the time was out. No place of the chair cures it. The
    run's watch says what its "none" is of (kind), and the line-up says that and ends."""
    untied = {
        "none": "the selected container fc_9 is in none of the scene camera's detections: nothing says that what "
        "the wrist sees is the one selected, and the run would not take it",
        "kind": guide.K_NOT_CHOSEN,
    }
    outcome, messages, source, _first, lining = watched([ROBOT, untied], budget_s=300.0)
    assert outcome == line_up.UNABLE and source.uncured == guide.K_NOT_CHOSEN
    assert messages == [guide.SEARCHING, "The arm cannot tell which container you chose"]
    assert lining._clock() - 100.0 == pytest.approx(0.2 + guide.NO_CURE_S, abs=0.3)
    assert not lining.guide.saw_box and source.probe is not None and source.probe.sent == []
    # the same from the words alone, of a lid the wrist has that stands off the scene camera's line to the
    # selected container (press_demo.tie_refusal: with the scene camera's mount off, as on 2026-11-17, that
    # is the selected container itself, in every frame)
    off = {
        "none": "a container at [0.650, 0.000], 212 mm to the side of the scene camera's line of sight to it: not "
        "taken for the selected container fc_9"
    }
    outcome, messages, source, _first, _lining = watched([ROBOT, off], budget_s=300.0, step_s=1.0)
    assert outcome == line_up.UNABLE and source.uncured == guide.K_NOT_CHOSEN
    assert messages == [guide.SEARCHING, "The arm cannot tell which container you chose"]
    # ... while the scene camera's sightings of the selected container go on arriving, the line-up goes on
    # by them: the person is bringing it in front of the arm, past another that stands there
    scene = {"fix": {"xyz": [1.05, 0.0, 0.315], "table_z": 0.216, "age_s": 0.05, "rough": False, "score": 0.9}}
    lines = [ROBOT] + [scene if i % 5 == 0 else off for i in range(150)]
    outcome, messages, source, _first, lining = watched(lines, budget_s=20.0, then=scene, far_m=0.8)
    assert outcome == TIMED_OUT and source.uncured == "" and lining.guide.saw_box
    assert "Move closer, about 25 cm" in messages
    # ... and a run that asked to be guided for a container no camera saw, whose watch then has one it
    # cannot tie (the chair brought it into view; the track that was selected is gone): said, and ended
    seen_none = {"none": "wrist camera: no counter in 1 frame (no level plane wider than a container)"}
    outcome, messages, source, first, _lining = watched(
        [ROBOT, *[seen_none] * 20, untied], budget_s=300.0, unseen="no camera sees the container (x)"
    )
    assert outcome == line_up.UNABLE and source.uncured == guide.K_NOT_CHOSEN
    assert messages == [guide.UNSEEN, "The arm cannot tell which container you chose"]
    assert first[0] == (0.0, guide.UNSEEN) and first[1][0] == pytest.approx(4.2 + guide.LOST_S, abs=0.11)


def test_two_things_that_could_be_the_container_are_said_as_that_and_never_end_a_line_up():
    """Two lid-sized tops in a frame: said as two THINGS (the second may be no container at all), with no
    "move in front of only one". They end no line-up, as at 21aa7cc: as a626985 had it, seven seconds of a
    second top in the wrist's frame while the chair was driven ended any line-up, the one before a run and
    the one for a press that does not plan among them, and the container alone in view from then on was
    never read (the integrator, 2026-11-19: put it back). The sentence stands while they do, the line-up
    goes on reading sightings, and its own time bounds it."""
    two = {"none": "wrist camera: two places pass as this container, [0.61, -0.10] and [0.66, 0.14]: refusing to guess"}
    says = "The arm sees two things that could be the container"
    outcome, messages, source, _first, lining = watched([ROBOT, two], budget_s=120.0)
    assert outcome == TIMED_OUT and source.unable == "" and source.uncured == "" and lining.guide.uncured == ""
    assert messages == [guide.SEARCHING, says, guide.AGAIN_UNSEEN] and lining._clock() - 100.0 > 120.0
    # 7.6 s of two tops, and then the container alone in view: read, and guided by
    outcome, messages, source, _first, lining = watched(
        [ROBOT] + [two] * 38 + [WRIST_FIX] * 5, budget_s=20.0, then=WRIST_FIX
    )
    assert outcome == TIMED_OUT and source.unable == "" and lining.guide.saw_box
    assert messages[:3] == [guide.SEARCHING, says, "Hold still, checking"]
    # one frame of it in eight (a second lid passing through the view) is not even said
    none = {"none": "wrist camera: nothing on the counter is the container (1 thing at lid height)"}
    lines = [ROBOT] + [two if i % 8 == 0 else none for i in range(120)]
    outcome, messages, source, _first, _lining = watched(lines, budget_s=20.0, then=none)
    assert outcome == TIMED_OUT and source.uncured == "" and source.unable == ""
    assert messages == [guide.SEARCHING, guide.UNSEEN, guide.AGAIN_UNSEEN]


WRIST_FAULT = {"none": "wrist camera: no new wrist frame"}
VISION = "The arm's vision is not working"


def test_the_wrist_cameras_fault_is_its_own_state_which_the_scene_cameras_sightings_neither_hide_nor_restart():
    """Flatbot's own case (review of 2026-11-19, the second): the line-up watches with the wrist, the wrist
    camera is dead, and the scene camera goes on placing a lid in front of the arm, roughly (its frame cuts
    it), which is never asked on. Each sighting started the fault's ten seconds again: the fault was never
    said and never ended, the panel said "Hold still, checking" and "Looking for the container" by turns
    for the whole wait, and the next press began the same again. Only the wrist's frames bring a Ready
    there, so its camera's fault is the line-up's state whatever the other camera sees: said once it has
    stood guide.LOST_S, ended at guide.NO_CURE_S."""
    rough = {"fix": {"xyz": [0.65, 0.0, 0.315], "table_z": 0.216, "age_s": 0.05, "rough": True, "score": 0.9}}
    lines = [ROBOT] + [rough if i % 10 == 0 else WRIST_FAULT for i in range(100)]  # a sighting every 2 s
    outcome, messages, source, first, lining = watched(lines, budget_s=300.0, then=WRIST_FAULT)
    assert outcome == line_up.UNABLE and source.uncured == guide.K_FAULT and source.unable == VISION
    assert source.why_none == WRIST_FAULT["none"]  # (what it ended on, for the engine's message)
    (t_said,) = (t for t, m in first if m == VISION)
    assert t_said == pytest.approx(0.4 + guide.LOST_S, abs=0.11)  # (its first word came 0.4 s in)
    assert lining._clock() - 100.0 == pytest.approx(0.4 + guide.NO_CURE_S, abs=0.3)
    assert guide.UNSEEN not in messages and guide.ASK not in messages
    # ... a careful sighting of the scene camera's every second, the same: none of them is read while it stands
    scene = {"fix": {"xyz": [1.05, 0.0, 0.315], "table_z": 0.216, "age_s": 0.05, "rough": False, "score": 0.9}}
    lines = [ROBOT] + [scene if i % 5 == 0 else WRIST_FAULT for i in range(150)]
    outcome, messages, source, _first, lining = watched(lines, budget_s=300.0, then=WRIST_FAULT, far_m=0.8)
    assert outcome == line_up.UNABLE and source.uncured == guide.K_FAULT
    assert lining._clock() - 100.0 == pytest.approx(0.4 + guide.NO_CURE_S, abs=0.3)
    assert messages[-1] == VISION and "Move closer, about 25 cm" not in messages[messages.index(VISION) :]
    # what ends that state is the wrist's own word that its camera delivers: a sighting of its own ...
    lines = [ROBOT] + [WRIST_FAULT] * 20 + [WRIST_FIX] * 5  # 4 s of it, then the container in its frames
    outcome, messages, source, _first, lining = watched(lines, budget_s=20.0, then=WRIST_FIX)
    assert outcome == TIMED_OUT and source.uncured == "" and source.standing == "" and lining.guide.saw_box
    assert messages[:3] == [guide.SEARCHING, VISION, "Hold still, checking"]
    # ... or a frame it judged and found no container in: one in every 6 s of faults, for 30 s
    none = {"none": "wrist camera: no counter in 1 frame"}
    lines = [ROBOT] + [none if i % 30 == 29 else WRIST_FAULT for i in range(150)]
    outcome, _messages, source, _first, _lining = watched(lines, budget_s=25.0, then=WRIST_FAULT)
    assert outcome == TIMED_OUT and source.unable == ""
    # the SCENE camera's own fault, where it alone watches, is restarted by its own sightings as before:
    # a camera that has just placed a container is delivering
    down = {"none": "scene camera streams missing: color, depth"}
    lines = [ROBOT] + [scene if i % 5 == 0 else down for i in range(150)]
    outcome, messages, source, _first, lining = watched(lines, budget_s=20.0, then=scene, far_m=0.8)
    assert outcome == TIMED_OUT and source.uncured == "" and source.standing == "" and VISION not in messages


def test_a_watch_that_falls_silent_is_itself_what_watches_not_working_and_ends_the_line_up():
    """A watch that says nothing more leaves its last sentence on the panel with nothing behind it: five
    sightings and then silence, and "The arm cannot see the container. Move the chair in front of it" stood
    297 of a line-up's 300 s; a fault word and then silence, and the fault sentence stood until the time
    was out (review of 2026-11-19, the second). Silent for longer than it ever takes between two words
    (line_up.SILENT_S, from its own cadence), it is what watches that is not working: said, and ended
    guide.NO_CURE_S after."""
    assert line_up.SILENT_S == 5.0 and line_up.SILENT_S > line_up.NONE_FRESH_S
    outcome, messages, source, first, lining = watched([ROBOT] + [WRIST_FIX] * 5, budget_s=300.0, then=False)
    assert outcome == line_up.UNABLE and source.uncured == guide.K_FAULT and lining.guide.saw_box
    assert source.why_none.startswith("the watch has said nothing for ")
    said = dict((m, t) for t, m in first)
    assert said[VISION] == pytest.approx(1.0 + line_up.SILENT_S, abs=0.11)  # (its last word came 1.0 s in)
    assert said[VISION] - said[guide.UNSEEN] < line_up.SILENT_S  # the lost container's sentence: seconds
    assert lining._clock() - 100.0 == pytest.approx(1.0 + line_up.SILENT_S + guide.NO_CURE_S, abs=0.3)
    # a fault word and then nothing: its sentence stands, and the silence ends it
    outcome, messages, source, _first, lining = watched([ROBOT, WRIST_FAULT], budget_s=300.0, then=False)
    assert outcome == line_up.UNABLE and source.uncured == guide.K_FAULT
    assert messages == [guide.SEARCHING, VISION]
    assert lining._clock() - 100.0 == pytest.approx(0.2 + line_up.SILENT_S + guide.NO_CURE_S, abs=0.3)
    # "no container in view" and then nothing: the same
    none = {"none": "wrist camera: no counter in 1 frame"}
    outcome, messages, source, _first, _lining = watched([ROBOT, none, none], budget_s=300.0, then=False)
    assert outcome == line_up.UNABLE and source.uncured == guide.K_FAULT
    assert messages == [guide.SEARCHING, guide.UNSEEN, VISION]
    # a watch that has not yet said a camera's word is not silent: its wrist is turning to the view, for as
    # long as that takes
    outcome, messages, source, _first, _lining = watched([ROBOT], budget_s=30.0, then=False)
    assert outcome == TIMED_OUT and source.unable == "" and messages == [guide.SEARCHING, guide.AGAIN_UNSEEN]
    # ... and one that speaks again is watching again: nothing ended, and the guidance goes on
    FakeChild.made = []
    source = BoxSource(["box_watch"], lambda urdf, root: ["reach_probe", urdf, root], {}, spawn=FakeChild)
    clock = Clock()
    ctx = Ctx(clock)
    real = source.look

    def look(now, t0=clock.t):
        t = now - t0
        if t < 1.0 or 9.0 <= t:  # eight seconds of nothing between
            source.watch.says(now, WRIST_FIX)
        return real(now)

    source.look = look
    lining = LineUp(ctx, lambda: True, source, 20.0, clock=clock, sleep=clock.sleep)
    assert lining.run() == TIMED_OUT and source.unable == "" and source.standing == ""
    shown = [panel["message"] for _t, panel in ctx.panels]
    shown = [m for i, m in enumerate(shown) if i == 0 or shown[i - 1] != m]  # each change of the panel
    assert shown[shown.index(VISION) + 1] == "Hold still, checking"


def test_an_arm_that_is_not_at_rest_is_never_first_said_to_be_a_container_the_arm_cannot_see():
    """The watch's word for an arm off rest is a "none" too. The tick that heard it handed it to the guide
    before the loop read `unable`: one panel, one console line and one progress of "The arm cannot see the
    container. Move the chair in front of it" before "The arm is not at its rest pose" (review of
    2026-11-19). It is the arm's word, of no kind, and is said as what it is on the tick that hears it."""
    outcome, messages, source, _first, _lining = watched([ROBOT, {"none": guide.ARM_NOT_AT_REST}], budget_s=20.0)
    assert outcome == line_up.UNABLE and source.unable == guide.ARM_NOT_AT_REST and source.uncured == ""
    assert messages == [guide.SEARCHING, guide.unable(guide.ARM_NOT_AT_REST).message]
    assert source.none_kind == ""


def test_one_frame_without_the_container_is_not_shown_to_a_person_who_is_in_place():
    """A line-up of any kind whose watch's first frame shows no container (one frame alone, a late TF) and
    the next ones show it: the panel and the console went Looking -> cannot see, move the chair -> Hold
    still (review of 2026-11-19). The person was in place, and was told to move."""
    lines = [ROBOT, {"none": "wrist camera: no new wrist frame"}, {"none": "wrist camera: no counter in 1 frame"}]
    outcome, messages, _source, _first, lining = watched([*lines, WRIST_FIX], budget_s=3.0)
    assert lining.guide.saw_box and messages[:2] == [guide.SEARCHING, "Hold still, checking"]
    assert guide.UNSEEN not in messages and "The arm's vision is not working" not in messages


def test_the_joints_the_wrist_looks_from_go_to_the_probe_with_every_box_from_then_on():
    """{"view": {"joints": [...]}}: said once by the run, as its wrist stands at the line-up's view. The run
    starts its reach from there, so the probe is told ("q_from") with every box it is sent from then on."""
    source, watch, probe = _source()
    watch.says(1.0, _fix((0.95, 0.0, 0.315), by="wrist"))
    source.look(1.1)
    assert probe.sent[-1] == {"box": [0.95, 0.0, 0.315], "table_z": 0.22, "prove": False}  # none said yet
    view = [0.0, 0.26, 3.14, -2.27, 0.35, 0.96, 1.2]
    watch.says(2.0, {"view": {"joints": view}})
    watch.says(2.1, _fix((0.90, 0.0, 0.315), by="wrist"))
    source.look(2.2)
    assert probe.sent[-1] == {"box": [0.90, 0.0, 0.315], "table_z": 0.22, "prove": False, "q_from": view}
    said = []
    source._log = said.append
    watch.says(3.0, {"view": {"joints": [0.1, 0.2]}})  # not the arm's seven: not taken, and said
    watch.says(3.1, {"view": None})
    watch.says(3.2, _fix((0.85, 0.0, 0.315), by="wrist"))
    source.look(3.3)
    assert probe.sent[-1]["q_from"] == view and len(said) == 2 and "joints" in said[0]


def test_the_boundary_judges_the_box_and_its_own_place_is_proven_where_it_is_inside():
    source, watch, probe = _source()
    probe.says(1.0, {"zone": _band_zone().to_dict()})
    watch.says(1.0, _fix((0.95, 0.0, 0.315)))
    _fix_, reach = source.look(1.1)
    assert not reach.inside and reach.to_zone == pytest.approx((-0.20, 0.0)) and probe.sent[-1]["prove"] is False
    watch.says(2.0, _fix((0.65, 0.0, 0.315)))
    _fix_, reach = source.look(2.1)
    assert reach.inside and reach.planned is None and probe.sent[-1]["prove"] is True  # inside: prove it there
    probe.says(2.6, {"proof": {"xyz": [0.65, 0.0, 0.315], "ok": True, "why": "", "tilt_deg": 15, "s": 0.5}})
    watch.says(2.7, _fix((0.652, 0.001, 0.316)))
    _fix_, reach = source.look(2.8)
    assert reach.inside and reach.planned is True and probe.sent[-1]["prove"] is False  # proven: not asked again
    watch.says(3.0, _fix((0.70, 0.0, 0.315)))  # the chair moved 5 cm: that proof is not for here
    _fix_, reach = source.look(3.1)
    assert reach.inside and reach.planned is None and probe.sent[-1]["prove"] is True


def test_a_place_the_run_itself_refused_does_not_work_for_its_line_up_whatever_the_probe_proves():
    """A line-up in the middle of a run (run 2 of 2026-11-17: the box 0.79 m out, well inside the grid, and
    no press planned). The box still stands where the run found no press as the line-up begins. The grid
    has that cell working and the probe (no planner, no table's edges) may prove it: the user was then
    asked "Ready?" on the very place the run had just refused, said yes, and the run refused it again.
    The run's word is the last: that place is not inside and is planned False, the way out of it is said,
    and a place a chair's move away is judged as ever."""
    FakeChild.made = []
    source = BoxSource(
        ["box_watch"],
        lambda urdf, root: ["reach_probe", urdf, root],
        {},
        spawn=FakeChild,
        refused=[[0.712, -0.10, 0.315]],
    )
    watch = source.watch
    watch.says(0.0, {"robot": {"urdf": "/state/arm.urdf", "root": "arm_mount"}})
    source.look(0.0)
    probe = source.probe
    watch.says(1.0, _fix((0.712, -0.10, 0.315)))
    _f, reach = source.look(1.05)
    assert reach is None  # no grid yet: checking, not "ready"
    probe.says(1.1, {"zone": _band_zone().to_dict()})
    probe.says(1.2, {"proof": {"xyz": [0.712, -0.10, 0.315], "ok": True, "why": "", "tilt_deg": 30, "s": 0.5}})
    watch.says(1.3, _fix((0.713, -0.101, 0.316)))
    _f, reach = source.look(1.35)
    assert not reach.inside and reach.planned is False and reach.to_zone is not None
    assert probe.sent[-1]["prove"] is False  # nothing is spent proving what the run refused
    say = guide.Guide().update(1.35, guide.Fix(1.3, (0.713, -0.101)), reach)
    assert say.state == guide.MOVE and say.message.startswith("Move ")
    # the same grid without the run's word: inside, and proven
    plain, w2, p2 = _source()
    p2.says(1.1, {"zone": _band_zone().to_dict()})
    p2.says(1.2, {"proof": {"xyz": [0.712, -0.10, 0.315], "ok": True, "why": "", "tilt_deg": 30, "s": 0.5}})
    w2.says(1.3, _fix((0.713, -0.101, 0.316)))
    _f, reach = plain.look(1.35)
    assert reach.inside and reach.planned is True
    # the chair moved: 10 cm nearer is another place, judged by the grid and its own proof
    watch.says(3.0, _fix((0.60, -0.10, 0.315)))
    _f, reach = source.look(3.05)
    assert reach.inside and reach.planned is None and probe.sent[-1]["prove"] is True
    probe.says(3.5, {"proof": {"xyz": [0.60, -0.10, 0.315], "ok": True, "why": "", "tilt_deg": 30, "s": 0.5}})
    watch.says(3.6, _fix((0.60, -0.10, 0.315)))
    _f, reach = source.look(3.65)
    assert reach.inside and reach.planned is True
    # a grid that comes again (the probe says it as it fills) keeps the place refused
    probe.says(4.0, {"zone": _band_zone().to_dict()})
    watch.says(4.1, _fix((0.712, -0.10, 0.315)))
    _f, reach = source.look(4.15)
    assert not reach.inside and reach.planned is False


def test_a_refused_place_is_refused_at_whatever_height_this_line_ups_watch_reads_the_lid():
    """The review of 2026-11-17: the run's refused place counted only within 15 mm of the RUN's lid height,
    and its cell only in the grid of that height. The run's fix and the line-up's watch are two readings
    of one lid (the scene camera's against the wrist's: centimetres apart in height, another grid's
    bucket), so the box could stand on the very spot the run refused and be asked "Ready?". A place is
    refused by where it is on the table."""
    FakeChild.made = []
    source = BoxSource(
        ["box_watch"],
        lambda urdf, root: ["reach_probe", urdf, root],
        {},
        spawn=FakeChild,
        refused=[[0.712, -0.10, 0.2945]],  # as the run measured the lid: 40 mm under the watch's reading
    )
    watch = source.watch
    watch.says(0.0, {"robot": {"urdf": "/state/arm.urdf", "root": "arm_mount"}})
    source.look(0.0)
    probe = source.probe
    probe.says(1.1, {"zone": _band_zone(lid_z=0.34).to_dict()})  # the grid of the height the watch reads
    probe.says(1.2, {"proof": {"xyz": [0.712, -0.10, 0.335], "ok": True, "why": "", "tilt_deg": 30, "s": 0.5}})
    watch.says(1.3, _fix((0.713, -0.101, 0.335), by="wrist"))
    _f, reach = source.look(1.35)
    assert not reach.inside and reach.planned is False and reach.to_zone is not None
    assert probe.sent[-1]["prove"] is False
    # a place given without a height (x, y) is refused the same way
    source = BoxSource(["box_watch"], lambda u, r: ["reach_probe"], {}, spawn=FakeChild, refused=[(0.712, -0.10)])
    source.watch.says(0.0, {"robot": {"urdf": "/state/arm.urdf", "root": "arm_mount"}})
    source.look(0.0)
    source.probe.says(1.1, {"zone": _band_zone().to_dict()})
    source.watch.says(1.3, _fix((0.713, -0.101, 0.315)))
    _f, reach = source.look(1.35)
    assert not reach.inside and reach.planned is False


def test_every_place_a_run_refused_is_held_against_the_line_up_not_only_the_newest():
    """A second guidance of one run, or the line-up carried to the next open_box, is told every place the
    runs refused (node.Asking.refused). As first written the engine handed over only the place of the
    request in hand: the first place was forgotten at the second guidance, and the person could be led
    back onto it and asked "Ready?" there."""
    FakeChild.made = []
    places = [[0.712, -0.10, 0.315], [0.60, 0.10, 0.315]]
    source = BoxSource(["box_watch"], lambda u, r: ["reach_probe"], {}, spawn=FakeChild, refused=places)
    source.watch.says(0.0, {"robot": {"urdf": "/state/arm.urdf", "root": "arm_mount"}})
    source.look(0.0)
    source.probe.says(1.0, {"zone": _band_zone().to_dict()})
    for t, place in enumerate(places, start=2):
        source.probe.says(t, {"proof": {"xyz": place, "ok": True, "why": "", "tilt_deg": 30, "s": 0.5}})
        source.watch.says(t + 0.1, _fix(place))
        _f, reach = source.look(t + 0.2)
        assert not reach.inside and reach.planned is False, place
    # ... and a place neither run refused, a chair's move from both, is judged as ever
    source.watch.says(5.0, _fix((0.55, -0.10, 0.315)))
    _f, reach = source.look(5.1)
    assert reach.inside and reach.planned is None


def test_a_rough_sighting_says_which_way_to_move_and_is_never_asked_on():
    source, watch, probe = _source()
    probe.says(1.0, {"zone": _band_zone().to_dict()})
    watch.says(1.0, _fix((0.95, 0.0, 0.315), rough=True))
    _fix_, reach = source.look(1.1)
    assert not reach.inside and reach.to_zone == pytest.approx((-0.20, 0.0))
    probe.says(1.5, {"proof": {"xyz": [0.65, 0.0, 0.315], "ok": True, "why": "", "tilt_deg": 15, "s": 0.5}})
    watch.says(2.0, _fix((0.65, 0.0, 0.315), rough=True))
    fix, reach = source.look(2.1)
    assert fix is not None and reach is None and probe.sent[-1]["prove"] is False  # in place, roughly: checking


def test_while_the_wrist_sees_the_box_the_scene_cameras_sightings_are_left_out():
    """With the wrist look, a wrist frame that shows no box lets the scene camera's sighting be said in its
    place: centimetres from where the wrist has the box, often rough, at another height. While a wrist fix
    is fresh (guide.LOST_S) such a sighting neither replaces the fix, nor enters the guide's track (the
    box "moved"), nor goes to the probe (another grid); the question stays. With no fresh wrist fix the
    scene camera's are used as ever."""
    source, watch, probe = _source()
    probe.says(1.0, {"zone": _band_zone().to_dict()})
    probe.says(1.0, {"proof": {"xyz": [0.65, 0.0, 0.315], "ok": True, "why": "", "tilt_deg": 15, "s": 0.5}})
    asked = guide.Guide()
    for i in range(8):  # the wrist sees it stand still, 1.0 .. 2.4 s
        t = 1.0 + 0.2 * i
        watch.says(t, _fix((0.65, 0.0, 0.315), age_s=0.0, by="wrist"))
        fix, reach = source.look(t)
        say = asked.update(t, fix, reach)
    assert say.state == guide.READY and fix == guide.Fix(pytest.approx(2.4), (0.65, 0.0), "wrist")
    sent = len(probe.sent)
    watch.says(2.5, _fix((0.73, 0.02, 0.36), age_s=0.0, rough=True))  # the scene camera's, in between
    fix, reach = source.look(2.5)
    assert fix is None and reach.inside and reach.planned is True  # the wrist's fix stands, judged as it was
    assert len(probe.sent) == sent and asked.update(2.5, fix, reach).state == guide.READY
    watch.says(2.6, _fix((0.65, 0.0, 0.315), age_s=0.0, by="wrist"))
    fix, reach = source.look(2.6)
    assert asked.update(2.6, fix, reach).state == guide.READY and asked.steady(2.6)  # and it never "moved"
    late = 2.6 + guide.LOST_S + 0.1  # the wrist has shown nothing for LOST_S: the scene camera's word again
    watch.says(late, _fix((0.95, 0.0, 0.315), age_s=0.0))
    fix, reach = source.look(late)
    assert fix == guide.Fix(pytest.approx(late), (0.95, 0.0), "scene") and not reach.inside
    assert probe.sent[-1]["box"] == [0.95, 0.0, 0.315]


def test_a_place_the_grid_has_working_and_the_proof_refuses_is_led_away_from():
    source, watch, probe = _source()
    zone = _band_zone()
    probe.says(1.0, {"zone": zone.to_dict()})
    watch.says(1.0, _fix((0.51, 0.0, 0.315)))
    _fix_, reach = source.look(1.1)
    assert reach.inside and probe.sent[-1]["prove"] is True
    probe.says(1.6, {"proof": {"xyz": [0.51, 0.0, 0.315], "ok": False, "why": "the knees", "tilt_deg": None, "s": 1}})
    watch.says(1.7, _fix((0.51, 0.0, 0.315)))
    _fix_, reach = source.look(1.8)
    # the cell at 0.50 no longer works here: the nearest place still inside is a cell on, beside it
    assert not reach.inside and reach.planned is False
    assert reach.to_zone[0] == pytest.approx(0.04) and abs(reach.to_zone[1]) == pytest.approx(0.05)
    assert source._zone.inside((0.51 + reach.to_zone[0], reach.to_zone[1]))
    probe.says(2.0, {"zone": zone.to_dict()})  # the probe's grid, said again as it fills: still refused here
    watch.says(2.1, _fix((0.51, 0.0, 0.315)))
    _fix_, reach = source.look(2.2)
    assert not reach.inside and reach.to_zone is not None


def test_the_deployments_bound_and_the_lids_height_decide_which_boundary_speaks():
    source, watch, probe = _source(far_m=0.70)
    probe.says(1.0, {"zone": _band_zone().to_dict()})
    watch.says(1.0, _fix((0.78, 0.0, 0.315)))
    _fix_, reach = source.look(1.1)
    assert not reach.inside and reach.to_zone[0] < 0  # the proof plans there; the deployment does not trust it
    watch.says(2.0, _fix((0.65, 0.0, 0.36)))  # another table: this grid was asked for a lid at 0.32
    _fix_, reach = source.look(2.1)
    assert reach is None and probe.sent[-1] == {"box": [0.65, 0.0, 0.36], "table_z": 0.22, "prove": False}
    probe.says(2.5, {"zone": _band_zone(lid_z=0.36).to_dict()})
    watch.says(2.6, _fix((0.65, 0.0, 0.36)))
    _fix_, reach = source.look(2.7)
    assert reach.inside


def test_a_box_further_out_than_the_deployments_bound_is_told_closer_with_no_grid_to_judge_it():
    """line_up_far_m: nothing further out than it is in the boundary, whatever the grid comes to hold. So a
    box out there is not inside and the way in is toward the arm: said at once, not "Hold still, checking"
    for as long as a blank grid takes to fill."""
    source, watch, probe = _source(far_m=0.80)
    watch.says(1.0, _fix((0.90, 0.05, 0.315)))  # no grid said at all yet
    fix, reach = source.look(1.1)
    assert reach is not None and not reach.inside and reach.planned is None
    word, metres, _axis = guide.instruction(reach.to_zone)
    assert word == "closer" and metres == pytest.approx(0.10, abs=0.005)
    assert math.hypot(0.90 + reach.to_zone[0], 0.05 + reach.to_zone[1]) == pytest.approx(0.80)  # toward the base
    assert guide.Guide().update(1.1, fix, reach).message == "Move closer, about 10 cm"
    assert probe.sent[-1]["prove"] is False
    probe.says(1.5, {"zone": Zone.blank((0.0, 1.0), (-0.5, 0.5), lid_z=0.32, key="k").to_dict()})
    watch.says(1.6, _fix((0.90, 0.05, 0.315)))  # a grid with nothing asked yet: the same
    _fix_, reach = source.look(1.7)
    assert not reach.inside and guide.instruction(reach.to_zone)[0] == "closer"
    watch.says(2.0, _fix((0.70, 0.0, 0.315)))  # within the bound the grid alone can say, and cannot yet
    assert source.look(2.1)[1] is None
    probe.says(2.5, {"zone": _band_zone().to_dict()})  # ... and where the grid can say, the way in is its own
    watch.says(2.6, _fix((0.90, 0.05, 0.315)))
    _fix_, reach = source.look(2.7)
    assert not reach.inside and reach.to_zone == pytest.approx((-0.15, -0.05))
    source, watch, probe = _source()  # no bound: nothing is known without the grid
    watch.says(1.0, _fix((0.90, 0.05, 0.315)))
    assert source.look(1.1)[1] is None


def test_a_box_just_inside_the_bound_in_a_cell_whose_centre_is_beyond_it_is_told_closer_too():
    """The bound goes by a cell's centre (Zone.within, reach_probe --far-m): the cell that holds a box at
    0.796 m has its centre at 0.806 m, is never asked and never works. A user driving in from too far was
    told "closer" up to the bound, then "Hold still, checking" in that band, where holding still gets no
    answer: the box there is told closer as well, until it stands in a cell the proof is asked of."""
    source, watch, probe = _source(far_m=0.80)
    probe.says(0.5, {"zone": Zone.blank((0.0, 1.0), (-0.5, 0.5), lid_z=0.32, key="k").to_dict()})
    watch.says(1.0, _fix((0.79, 0.10, 0.315)))
    assert math.hypot(0.79, 0.10) < 0.80 < math.hypot(0.80, 0.10)  # the box within the bound, its cell's centre not
    _fix_, reach = source.look(1.1)
    assert reach is not None and not reach.inside
    word, metres, _axis = guide.instruction(reach.to_zone)
    assert word == "closer" and 0.0 < metres < 0.02
    watch.says(2.0, _fix((0.74, 0.10, 0.315)))  # a cell within the bound, not asked yet: the grid's to say
    assert source.look(2.1)[1] is None
    source, watch, probe = _source(far_m=0.80)  # with no grid said at all there is no cell to go by
    watch.says(1.0, _fix((0.79, 0.10, 0.315)))
    assert source.look(1.1)[1] is None


def test_the_probe_is_held_to_one_thread():
    """Its proofs are numpy's, beside the arm's driver on the robot's computer: its BLAS is not to take the
    cores. The watch's environment is as it was given."""
    source, watch, probe = _source()
    assert watch.env == {"E": "1"}
    assert probe.env == {"E": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def test_closing_ends_both_children_the_probe_told_to_quit_and_given_time_to_keep_its_grid():
    source, watch, probe = _source()
    source.close()
    assert watch.stopped and probe.stopped
    # reach_probe keeps what it asked only where it ends of itself: told to quit, then given a bounded time
    assert probe.sent[-1] == {"quit": True} and probe.quit_s == line_up.PROBE_QUIT_S
    assert 0.0 < line_up.PROBE_QUIT_S <= 3.0
    assert watch.sent == [] and watch.quit_s == 0.0  # the watch keeps nothing: ended as it always was
    FakeChild.made = []
    alone = BoxSource(["box_watch"], lambda urdf, root: [], {}, spawn=FakeChild)
    alone.close()  # the probe never started
    assert alone.watch.stopped


def test_a_line_up_on_a_real_source_from_too_far_to_open():
    """The loop on a BoxSource whose children are scripted: the box comes from 0.95 m to 0.65 m and stands;
    the boundary is there from the start, the proof comes once it is asked for."""
    source, watch, probe = _source()
    clock = Clock()
    ctx = Ctx(clock, presses=[(9.0, OPEN_PRESS)])
    probe.says(clock.t, {"zone": _band_zone().to_dict()})

    class Scripted:
        """Feeds the children's lines as time passes: a sighting every 0.2 s, the proof 0.6 s after it is asked."""

        def __init__(self):
            self.next_fix, self.proof_at = clock.t, None

        def look(self, now):
            t = now - ctx.t0
            if now >= self.next_fix:
                x = max(0.65, 0.95 - 0.1 * t)
                watch.says(now, _fix((x, 0.0, 0.315), age_s=0.1))
                self.next_fix = now + 0.2
            if self.proof_at is None and probe.sent and probe.sent[-1]["prove"] and probe.sent[-1]["box"][0] == 0.65:
                self.proof_at = now + 0.6
            if self.proof_at is not None and now >= self.proof_at:
                probe.says(now, {"proof": {"xyz": [0.65, 0.0, 0.315], "ok": True, "why": "", "tilt_deg": 15, "s": 0.6}})
                self.proof_at = float("inf")
            return source.look(now)

    lining = LineUp(ctx, lambda: True, Scripted(), 40.0, clock=clock, sleep=clock.sleep)
    assert lining.run() == GO
    words = ctx.messages()
    assert words[0] == "Move closer, about 20 cm" and "Hold still, checking" in words
    assert words[-2:] == ["Ready. Open the container?", "Opening the container"]
    ready_at = next(t for t, p in ctx.panels if p["message"] == "Ready. Open the container?")
    assert 3.0 + 0.6 <= ready_at <= 3.0 + 0.6 + guide.STEADY_S + 0.5  # in place at 3 s, proven, watched still


def test_a_child_is_a_process_whose_marked_lines_are_its_messages(tmp_path):
    import sys
    import time

    script = tmp_path / "child.py"
    script.write_text(
        "import json, sys\n"
        "print('a line of its own log', flush=True)\n"
        "print('@@ ' + json.dumps({'robot': {'urdf': None}}), flush=True)\n"
        "print('@@ not json', flush=True)\n"
        "for line in sys.stdin:\n"
        "    print('@@ ' + json.dumps({'heard': json.loads(line)}), flush=True)\n"
    )
    logged = []
    child = line_up.Child([sys.executable, str(script)], None, log=logged.append)
    try:
        child.send({"box": [1, 2, 3]})
        got, t0 = [], time.monotonic()
        while len(got) < 2 and time.monotonic() - t0 < 10.0:
            got += child.messages()
            time.sleep(0.02)
        assert [m for _t, m in got] == [{"robot": {"urdf": None}}, {"heard": {"box": [1, 2, 3]}}]
        assert all(t0 <= t <= time.monotonic() for t, _m in got)
        assert logged == ["a line of its own log"] and child.alive()
    finally:
        child.stop()
    assert not child.alive()
    child.send({"box": [0, 0, 0]})  # to a child that has ended: nothing, and no error


def test_a_watch_that_is_the_run_itself_is_not_this_sources_to_end():
    """The wrist look: the run, started with --line-up, says the watch's lines. close() ends the probe alone."""
    FakeChild.made = []
    run = FakeChild(["press_demo", "--line-up"], {})
    source = BoxSource(run, lambda urdf, root: ["reach_probe", urdf], {}, spawn=FakeChild)
    assert source.watch is run and len(FakeChild.made) == 1  # nothing spawned for the watch
    run.says(0.0, {"robot": {"urdf": "/state/arm.urdf", "root": "arm_mount"}})
    run.says(0.1, {"fix": {"xyz": [0.65, 0.0, 0.315], "table_z": 0.216, "age_s": 0.05, "rough": False, "by": "wrist"}})
    fix, _reach = source.look(0.2)
    assert fix.xy == (0.65, 0.0) and source.probe.argv == ["reach_probe", "/state/arm.urdf"]
    source.close()
    assert source.probe.stopped and not run.stopped


def test_a_line_up_ends_at_once_where_its_watch_has_gone():
    FakeChild.made = []
    source = BoxSource(["box_watch"], lambda urdf, root: [], {}, spawn=FakeChild)
    assert not source.gone()
    clock = Clock()
    ctx = Ctx(clock)
    lining = LineUp(ctx, lambda: True, source, 40.0, clock=clock, sleep=clock.sleep)
    source.watch.stopped = True  # it ended (not ready, its camera gone)
    assert lining.run() == line_up.GONE and clock.t - ctx.t0 < 0.1


def test_a_line_up_ends_at_once_where_its_probe_has_died_and_says_what_the_probe_said_last():
    """A reach_probe that died judges nothing: the panel would say "Hold still, checking" until the time ran
    out, and the user be told the container was not in place. The source is gone from then on, and says it
    was the probe, with the end of its log."""
    source, _watch, probe = _source()
    assert not source.gone()
    probe.lines = [
        '  File "reach_probe.py", line 446, in main',
        "    ctx, cfg = setup(args.urdf)",
        "ValueError: no chain",
    ]
    probe.stopped = True  # it ended of itself: a traceback at its start, an exception in a proof
    clock = Clock()
    ctx = Ctx(clock)
    lining = LineUp(ctx, lambda: True, source, 40.0, clock=clock, sleep=clock.sleep)
    assert lining.run() == line_up.GONE and clock.t - ctx.t0 < 0.1
    assert "reach_probe" in source.why_gone and "ValueError: no chain" in source.why_gone
    assert "setup(args.urdf)" in source.why_gone  # the last few lines, not the last alone


def test_an_arm_that_is_not_at_rest_ends_the_line_up_once_the_panel_has_said_so():
    """What no waiting cures: the run says the arm does not stand at its rest pose (guide.ARM_NOT_AT_REST, in
    the place of where the container is). The panel says it, with no button, for LINGER_S, and the line-up
    ends UNABLE: not a whole line-up of "Looking for the container". Any other reason for seeing no container
    is not that, and the line-up goes on."""
    source, watch, _probe = _source()
    watch.says(0.0, {"none": "wrist camera: no lid-sized top on the table in view"})
    source.look(0.1)
    assert source.unable == "" and source.why_none.startswith("wrist camera")
    watch.says(0.2, {"none": guide.ARM_NOT_AT_REST})
    outcome, ctx, said, lining = lined_up(source)
    assert outcome == line_up.UNABLE and source.unable == guide.ARM_NOT_AT_REST
    assert line_up.LINGER_S <= ctx.clock.t - ctx.t0 < line_up.LINGER_S + 3 * line_up.POLL_S
    shown = [
        p for _t, p in ctx.panels if p["message"] == "The arm is not at its rest pose: home it, then press open again"
    ]
    assert shown and all(p["buttons"] == [] and p["progress"]["phase"] == guide.UNABLE for p in shown)
    assert said[-1].state == guide.UNABLE and not lining.cancel_pressed


@pytest.mark.parametrize("by", ["the orchestrator", "the panel"])
def test_a_cancel_is_still_a_cancel_while_the_panel_says_the_arm_is_not_at_rest(by):
    """The orchestrator's cancel is read every tick of that panel's time too, and a press of Cancel made on
    the panel before it (it has no button of its own) is the user's cancel, not lost."""
    source, watch, _probe = _source()
    watch.says(0.0, {"none": guide.ARM_NOT_AT_REST})
    press = [(0.5, {"type": "button", "button_id": "cancel"})] if by == "the panel" else []
    outcome, ctx, _said, lining = lined_up(source, presses=press, cancel_at=0.5 if by == "the orchestrator" else None)
    assert outcome == CANCELLED and ctx.clock.t - ctx.t0 < 0.5 + 3 * line_up.POLL_S
    assert lining.cancel_pressed is (by == "the panel")


def test_the_source_knows_the_wrist_turned_once_the_run_says_its_view():
    source, watch, _probe = _source()
    assert source.viewed is False
    watch.says(0.5, {"view": {"joints": [0.0] * 7}})
    source.look(0.6)
    assert source.viewed is True


def test_closing_on_the_users_open_does_not_wait_on_the_probe():
    """The run's go comes after the source is closed: on an Open the probe is ended at once (its terminate
    keeps its grid too), not given PROBE_QUIT_S to finish the proof in flight."""
    source, _watch, probe = _source()
    source.close(quit_s=0.0)
    assert probe.sent[-1] == {"quit": True} and probe.stopped and probe.quit_s == 0.0
    source, _watch, probe = _source()
    source.close()
    assert probe.quit_s == line_up.PROBE_QUIT_S


def test_a_child_keeps_the_end_of_its_log_for_whoever_asks_why_it_ended(tmp_path):
    import sys
    import time

    script = tmp_path / "child.py"
    script.write_text(
        "import sys\n"
        "for i in range(40):\n"
        "    print('line %d' % i)\n"
        "print('@@ {\"zone\": {}}')\n"
        "sys.exit('ValueError: no chain from arm_mount')\n"
    )
    child = line_up.Child([sys.executable, str(script)], None)
    t0 = time.monotonic()
    while child.alive() and time.monotonic() - t0 < 10.0:
        time.sleep(0.02)
    said = child.last_lines()
    assert said[-2:] == ["line 39", "ValueError: no chain from arm_mount"]  # its log's lines, not its messages
    assert 2 < len(said) <= line_up.TAIL_LINES


PROBE = """
import json, sys, time
from pathlib import Path
print('@@ {"up": true}', flush=True)
for line in sys.stdin:
    if json.loads(line).get("quit") and sys.argv[2] == "saves":
        time.sleep(0.4)  # the proof in flight is finished first
        Path(sys.argv[1]).write_text("the grid")
        sys.exit(0)
"""


@pytest.mark.parametrize("probe", ["saves", "hangs"])
def test_a_probe_told_to_quit_keeps_its_grid_and_one_that_does_not_end_is_ended_all_the_same(tmp_path, probe):
    """close() against a real process standing in for reach_probe --serve, which writes its grid only as it
    ends of itself (on {"quit": true}): the grid is there afterwards. One that does not end is terminated
    once its time is up, so a close is bounded either way."""
    import sys
    import time

    script, kept = tmp_path / "probe.py", tmp_path / "grid.json"
    script.write_text(PROBE)
    FakeChild.made = []
    run = FakeChild(["press_demo", "--line-up"], {})
    source = BoxSource(
        run, lambda urdf, root: [sys.executable, str(script), str(kept), probe], None, spawn=line_up.Child
    )
    run.says(0.0, {"robot": {"urdf": "/state/arm.urdf", "root": "arm_mount"}})
    source.look(0.1)
    t0 = time.monotonic()
    while not source.probe.messages() and time.monotonic() - t0 < 10.0:  # it is up, and reading its stdin
        time.sleep(0.02)
    t0 = time.monotonic()
    source.close()
    took = time.monotonic() - t0
    assert not source.probe.alive() and not run.stopped
    if probe == "saves":
        assert kept.read_text() == "the grid" and 0.4 <= took < line_up.PROBE_QUIT_S
    else:
        assert not kept.exists() and line_up.PROBE_QUIT_S <= took < line_up.PROBE_QUIT_S + line_up.STOP_WAIT_S
