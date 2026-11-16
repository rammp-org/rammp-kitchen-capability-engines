"""The line-up before a run: the loop that tells the user which way to move, and asks before the arm flies.

The base class's example (capabilities/example, count_with_panel) is its pattern: one loop inside
execute() that sends a panel (ctx.update_user_panel), reads the buttons (ctx.poll_ui_event) and honours a
cancel, none of them waited on. What the panel says is guide.Guide's; what the guide is told each tick,
where the box was last seen and what the boundary says of that place, is a `source`'s:

    source.look(now) -> (Fix or None, Reach or None)     the newest sighting (None: nothing new since the
                                                         last tick) and the boundary's word on the box

It ends one of three ways: GO (Open pressed on a fresh "Ready. Open the container?"), CANCELLED (the
panel's Cancel, or the orchestrator's), TIMED_OUT (the time for lining up ran out: the panel says to press
open again, and holds that for LINGER_S). Nothing here moves the arm or starts a run: the caller does,
on GO alone.

MockApproach is the mock's source: a box that comes into reach as a user driving up to a table brings it,
so the UI's side of a line-up can be rehearsed with no arm and no camera.
"""

import math
import time

from box_opening import guide
from box_opening.reach_zone import Zone

GO, CANCELLED, TIMED_OUT = "go", "cancelled", "timed_out"

POLL_S = 0.05
# a panel that changed is sent at once where its words or buttons did, and no oftener than this where only its
# bar moved (the bar follows the box as it nears); unchanged, it is sent again every REFRESH_S: a goal the
# bridge ended on its own is opened again by the next send (node.Panel's period)
SEND_MIN_S = 0.2
REFRESH_S = 1.0
LINGER_S = 3.0  # "Press open again when you are in place" stays up this long before the action ends
PHASE = "line_up"  # the phase the orchestrator's progress names while the user lines up


class LineUp:
    """One line-up. `live()`: whether this execution has the UI (node.Capability._ui_live). `on_say(say)`:
    called where what the panel says changes (the node's progress and log)."""

    def __init__(self, ctx, live, source, budget_s, on_say=None, clock=time.monotonic, sleep=time.sleep):
        self._ctx = ctx
        self._live = live
        self._source = source
        self._clock = clock
        self._sleep = sleep
        self._on_say = on_say
        self.guide = guide.Guide(budget_s=budget_s)
        self.cancel_pressed = False  # the panel's own Cancel ended it (the caller makes it the orchestrator's)
        self._sent = None  # the panel as last sent, None where the last send did not go out
        self._tried_at = None
        self._told = None  # (state, message) as last handed to on_say

    def _show(self, say, now):
        if (say.state, say.message) != self._told:
            self._told = (say.state, say.message)
            if self._on_say is not None:
                self._on_say(say)
        panel = say.panel()
        if self._tried_at is None or now - self._tried_at >= REFRESH_S:
            due = True
        elif self._sent is None or panel == self._sent:
            due = False  # a send that did not go out is tried again at the refresh, no oftener
        else:
            words = (panel["message"], panel["buttons"]) != (self._sent["message"], self._sent["buttons"])
            due = words or now - self._tried_at >= SEND_MIN_S
        if due:
            self._tried_at = now
            self._sent = panel if self._live() and self._ctx.update_user_panel(panel_json=panel) else None

    def run(self):
        """Until the user says yes, cancels, or the time runs out: GO, CANCELLED or TIMED_OUT."""
        timed_out_at = None
        while True:
            now = self._clock()
            if self._ctx.cancelled():
                return CANCELLED
            event = self._ctx.poll_ui_event()
            if event is not None and event.get("type") == "button":
                # any press ends the bridge's goal, and its panel with it: the next send opens it again
                self._sent = self._tried_at = None
                if event.get("button_id") == guide.CANCEL:
                    self.cancel_pressed = True
                    return CANCELLED
                if timed_out_at is None and self.guide.press(event.get("button_id"), now):
                    self._show(self.guide.update(now, None, None), now)  # "Opening the container"
                    return GO
            fix, reach = (None, None) if timed_out_at is not None else self._source.look(now)
            say = self.guide.update(now, fix, reach)
            self._show(say, now)
            if say.state == guide.TIMED_OUT:
                timed_out_at = now if timed_out_at is None else timed_out_at
                if now - timed_out_at >= LINGER_S:
                    return TIMED_OUT
            self._sleep(POLL_S)


# The mock's table, as the arm stands to it (m, the arm's frame). Stand-ins: the mock sees nothing and reaches
# nothing. The band is about where flatbot's arm opened boxes on 2026-11-12 (0.63 and 0.68 m out).
MOCK_ZONE_X = (0.50, 0.80)
MOCK_ZONE_Y = (-0.20, 0.20)
MOCK_PATH = ((1.00, 0.35), (0.70, 0.35), (0.70, 0.0))  # too far and to the left; closer; then left, into reach


def mock_zone():
    zone = Zone.blank((0.0, 1.0), (-0.5, 0.5), key="mock")
    nx, ny = zone.cells.shape
    for i in range(nx):
        for j in range(ny):
            x, y = zone.centre(i, j)
            zone.set(
                i,
                j,
                MOCK_ZONE_X[0] - 1e-9 <= x <= MOCK_ZONE_X[1] + 1e-9
                and MOCK_ZONE_Y[0] - 1e-9 <= y <= MOCK_ZONE_Y[1] + 1e-9,
            )
    return zone


class MockApproach:
    """A box that comes along MOCK_PATH in `pace_s` seconds from the first look and then stands still (0:
    in place at once), seen every tick, judged by mock_zone no further out than `far_m` (the node's
    line_up_far_m; None: no bound); the run "plans" wherever it is inside."""

    def __init__(self, pace_s=0.0, path=MOCK_PATH, far_m=None):
        self._pace_s = max(0.0, float(pace_s))
        self._path = [tuple(float(v) for v in p) for p in path]
        self._zone = mock_zone().within(far_m=far_m)  # the deployment's bound, as a real line-up takes it
        self._t0 = None

    def at(self, t):
        """Where the box is `t` seconds into the approach."""
        legs = [math.dist(a, b) for a, b in zip(self._path, self._path[1:], strict=False)]
        if self._pace_s <= 0.0 or t >= self._pace_s:
            return self._path[-1]
        gone = max(0.0, t) / self._pace_s * sum(legs)
        for (a, b), length in zip(zip(self._path, self._path[1:], strict=False), legs, strict=False):
            if gone <= length:
                f = gone / length if length else 1.0
                return a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])
            gone -= length
        return self._path[-1]

    def look(self, now):
        self._t0 = now if self._t0 is None else self._t0
        xy = self.at(now - self._t0)
        return guide.Fix(now, xy, "mock"), self._zone.reach(xy, planned=True)
