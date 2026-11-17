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
    assert ctx.messages() == ["Looking for the container", "Press open again when you are in place"]


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
    instruction's new distance waits out the second (the newest is the one said), and a new direction, hold
    still and the question are said the tick they come."""
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
    # never five a second, and nothing but distances was held: the other words come when they did without it
    assert all(sum(1 for t, _m in said if t0 <= t < t0 + 1.0) < 5 for t0, _m in said)
    others = lambda lines: [(t, m) for t, m in lines if not m.startswith("Move")]  # noqa: E731
    assert others(said) == others(every)
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
