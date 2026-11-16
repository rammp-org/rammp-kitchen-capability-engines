"""The line-up before a run (line_up.LineUp): the panel's instruction, the question, and its three ends."""

import pytest

from box_opening import guide, line_up
from box_opening.line_up import CANCELLED, GO, TIMED_OUT, LineUp, MockApproach

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
