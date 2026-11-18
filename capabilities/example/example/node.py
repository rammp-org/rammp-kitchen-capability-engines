"""Reference capability: `count` counts up with progress; `count_with_panel`
does the same while also driving a live UI-bridge panel by calling `ctx`
directly; `count_with_runui_panel` does the same again but built on
`capability_ui.RunUI`'s reusable panel helpers instead.

Copy this directory with `make new CAP=<name>` and replace these with your
capability's actions. Everything orchestrator-facing (registration, the
activate/execute/control/hibernate servers, /example/status, SIGTERM) comes
from capability_engine.CapabilityEngine.

    ros2 launch example example.launch.py                          # with the orchestrator
    ros2 launch example example.launch.py use_orchestrator:=false  # standalone
    make action CAP=example ACTION=count PARAMS='{"to": 5}'        # send it a goal
    make action CAP=example ACTION=count_with_panel PARAMS='{"to": 5}'
    make action CAP=example ACTION=count_with_runui_panel PARAMS='{"to": 5}'
"""

import time

from capability_ui import RunUI, cancel_own_run
from sensor_msgs.msg import Image

from capability_engine import Action, CapabilityEngine, EngineError, run

_OVERLAY_WIDTH = 320
_OVERLAY_HEIGHT = 240


_STRIPE_BG = (20, 20, 20)
_STRIPE_FG = (0, 160, 255)
_STRIPE_WIDTH = 24


def _build_stripe_overlay_image() -> Image:
    """One rgb8 frame of diagonal stripes -- a placeholder for a real
    capability's own camera feed (see publish_overlay_image()'s docstring),
    generated once and republished unchanged every tick below. A fixed,
    recognizable pattern (same convention as static_overlay_node's own
    crosshair) makes it obvious on screen that a genuine frame reached UE,
    without the visual noise (and incompressibility) of random static --
    and the diagonal stripes are easily told apart from that mock's
    crosshair."""
    msg = Image()
    msg.height = _OVERLAY_HEIGHT
    msg.width = _OVERLAY_WIDTH
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = _OVERLAY_WIDTH * 3
    rows = []
    for y in range(_OVERLAY_HEIGHT):
        row = bytearray(_OVERLAY_WIDTH * 3)
        for x in range(_OVERLAY_WIDTH):
            is_fg = ((x + y) // _STRIPE_WIDTH) % 2 == 0
            r, g, b = _STRIPE_FG if is_fg else _STRIPE_BG
            row[x * 3 + 0] = r
            row[x * 3 + 1] = g
            row[x * 3 + 2] = b
        rows.append(bytes(row))
    msg.data = b"".join(rows)
    return msg


class Capability(CapabilityEngine):
    display_name = "Example"
    # Activated when the user selects one of these perception classes.
    perception_classes = ["counter"]
    # Resources the orchestrator locks while an action runs: "arm", "base",
    # "gripper". This example touches no hardware.
    required_resources = []
    _COUNT_PARAMS_SCHEMA = {
        "type": "object",
        "properties": {
            "to": {"type": "integer", "minimum": 1, "default": 3},
            "interval_s": {"type": "number", "minimum": 0, "default": 1.0},
        },
    }
    # _wait_for_stop()'s poll/checkpoint step while waiting out an interval.
    _UI_POLL_PERIOD_S = 0.1
    actions = [
        Action("count", "Count up", params_schema=_COUNT_PARAMS_SCHEMA),
        Action("count_with_panel", "Count up (live panel)", params_schema=_COUNT_PARAMS_SCHEMA),
        Action(
            "count_with_runui_panel", "Count up (RunUI panel)", params_schema=_COUNT_PARAMS_SCHEMA
        ),
    ]

    def __init__(self):
        super().__init__("example")

    def execute(self, action_id, params, ctx):
        to = int(params.get("to", 3))
        interval = float(params.get("interval_s", 1.0))
        if to < 1:
            raise EngineError("bad_params", "'to' must be at least 1", recoverable=True)

        if not self.mock:
            # self.arm talks to the arm driver; the first call opens it for this action.
            state = self.arm.get_state()
            self.get_logger().info(f"arm joints {state['position']}, tool at {state['ee_pos'][:3]}")

        if action_id == "count_with_panel":
            return self._count_with_panel(to, interval, ctx)
        if action_id == "count_with_runui_panel":
            return self._count_with_runui_panel(to, interval, ctx)
        return self._count(to, interval, ctx)

    def _count(self, to: int, interval: float, ctx) -> dict:
        for n in range(1, to + 1):
            ctx.checkpoint()  # honours pause and cancel between steps
            # A real capability would command hardware here, unless self.mock.
            time.sleep(interval)
            ctx.progress("counting", 100.0 * n / to, f"{n}/{to}")

        # A capability that changes the world reports it, e.g.
        # ctx.update_object(self.instance_id, opened=True) or
        # ctx.write_world_state(gripper={"holding": "cup_1"}).

        return {"counted_to": to}

    def _count_with_panel(self, to: int, interval: float, ctx) -> dict:
        """Same as `count`, but also drives a live, engine-owned panel on the
        UI bridge directly (independent of ctx.progress(), which only reports
        to the orchestrator and is never shown to the user): a progress bar
        plus a Stop button, refreshed every step, and polled for a tap.
        ctx.update_user_panel()/ctx.poll_ui_event() are both non-blocking --
        this loop keeps counting in between calls rather than waiting on the
        user. No session token (e.g. running standalone) makes every bridge
        call a harmless no-op, so this still behaves like plain `count`.

        A capability with its own camera (e.g. a wrist cam) would call
        ctx.publish_overlay_image(image) here, once per tick, to show the
        user a live view alongside the panel -- this example has no real
        camera, so it republishes one static diagonal-stripe test frame
        (_build_stripe_overlay_image(), built once per call) every
        tick instead, purely to demonstrate a frame actually reaches UE.
        ctx.set_overlay_visible() brackets the whole run below, to
        demonstrate the show/hide toggle (UE's `Open Camera Feed`/`Close
        Camera Feed` Remote Control functions): shown right before the loop
        starts, hidden in a `finally` so it's hidden on every exit path --
        normal completion, the user's Stop button, or ctx.checkpoint()
        raising Cancelled -- not just the happy path. The Orchestrator's own
        end-of-execution cleanup also unconditionally hides the overlay (see
        OrchestratorNode's execution-finish handling), so this is actually
        redundant by the time execute() returns -- it's here to show *this
        engine's own* call, for a capability that wants the feed hidden the
        instant its loop ends rather than waiting on that later cleanup.
        """
        self.get_logger().info("count_with_panel: showing overlay camera feed")
        shown = ctx.set_overlay_visible(True)
        self.get_logger().info(f"count_with_panel: set_overlay_visible(True) returned {shown}")
        overlay_image = _build_stripe_overlay_image()
        try:
            for n in range(1, to + 1):
                ctx.checkpoint()
                pct = 100.0 * n / to
                ctx.update_user_panel(
                    panel_json={
                        "title": "Counting",
                        "message": f"{n}/{to}",
                        "progress": {"phase": "counting", "pct": pct, "pausable": False},
                        "buttons": [{"button_id": "stop", "label": "Stop"}],
                    }
                )
                published = ctx.publish_overlay_image(overlay_image)
                self.get_logger().info(
                    f"count_with_panel: publish_overlay_image returned {published}"
                )
                # Poll for Stop throughout the interval, including on the last
                # iteration, instead of only once beforehand and then sleeping
                # through it uninterrupted -- otherwise a tap during any
                # interval is never consumed and the action reports normal
                # success.
                event = self._wait_for_stop(interval, ctx)
                if event is not None:
                    self.get_logger().info(f"count_with_panel: user pressed Stop at {n}/{to}")
                    return {"counted_to": n, "stopped_by_user": True}
                ctx.progress("counting", pct, f"{n}/{to}")

            return {"counted_to": to}
        finally:
            self.get_logger().info("count_with_panel: hiding overlay camera feed")
            hidden = ctx.set_overlay_visible(False)
            self.get_logger().info(
                f"count_with_panel: set_overlay_visible(False) returned {hidden}"
            )

    def _count_with_runui_panel(self, to: int, interval: float, ctx) -> dict:
        """Same as `count_with_panel` above, but built on `capability_ui.RunUI`'s reusable panel
        helpers instead of calling ctx.update_user_panel()/ctx.poll_ui_event()/
        ctx.publish_overlay_image()/ctx.set_overlay_visible() by hand -- and exercising most of
        RunUI's interactive surface, not just the progress bar, to serve as a worked reference for
        it:

        - `RunUI.ask()` with a plain button (`_ask_to_start`): waits for a tap (plus the automatic
          Stop) before anything starts.
        - `RunUI.ask()` again with `objects=`/`on_tap=` (`_ask_start_marker`): two markers overlaid
          on the camera feed (`"source": "overlay"`, this run's own `publish_overlay_image()`
          stream) the user can tap to choose where counting starts from, highlighting the tapped
          one via `selected_instance_id` -- the same mechanism a real capability uses to let the
          user pick between several detected objects (see `place_container`'s spot picker).
        - `RunUI.motion()` shows the progress-with-Stop panel and runs its own watcher thread that
          reads ctx.poll_ui_event() for the tap -- this loop never has to poll for it itself, unlike
          `_wait_for_stop()` above. Its `cancel` callback is wired to this engine's own cancel
          (`cancel_own_run`), the same cancel the orchestrator's own Stop/cancel goes through, so a
          Stop here shows a "Stopped" notice and re-raises `Cancelled` exactly as any other
          cancelled action would -- a different (and more featureful) Stop behaviour than
          `count_with_panel`'s own hand-rolled one above, which instead returns a normal result
          with `stopped_by_user: true`.
        - `RunUI.overlay_shown()` brackets the whole interactive run's overlay show/hide in one
          call, in place of `count_with_panel`'s own `try`/`finally` around
          `ctx.set_overlay_visible()` -- the markers are visible while being asked about, not just
          during the counting loop.
        - `RunUI.done()` ends the run with a final notice, waited on (bounded) as every notice is.
        """
        ui = RunUI(ctx, "Counting", cancel=lambda: cancel_own_run(self), logger=self.get_logger())
        overlay_image = _build_stripe_overlay_image()

        with ui.overlay_shown():
            ui.publish_overlay_image(overlay_image)
            self._ask_to_start(ui, to)
            start = self._ask_start_marker(ui, ctx)

            def do_count():
                for n in range(start, to + 1):
                    ctx.checkpoint()  # honours pause and cancel between steps
                    pct = 100.0 * n / to
                    ui.moving(f"{n}/{to}", pct, "counting")
                    ui.publish_overlay_image(overlay_image)
                    time.sleep(interval)
                    ctx.progress("counting", pct, f"{n}/{to}")
                return {"counted_to": to, "started_at": start}

            result = ui.motion(f"Counting to {to}.", do_count)

        ui.done(f"Counted to {to}.", mock=self.mock)
        return result

    def _ask_to_start(self, ui: RunUI, to: int) -> None:
        """A plain confirm button (plus the automatic Stop) before anything counts. Standalone (no
        UI session) starts immediately; connected but unanswered within the timeout refuses --
        nothing was counted -- rather than silently guessing. A Stop here cancels the action (via
        `RunUI.ask()`'s own handling) and is never seen as a plain `None` answer."""
        answer = ui.ask(f"Ready to count to {to}?", [{"button_id": "start", "label": "Start"}])
        if answer == "start" or ui.connected is False:
            return
        message = "No answer: nothing was counted."
        ui.notice(message)
        raise EngineError("no_answer", message, recoverable=True)

    def _ask_start_marker(self, ui: RunUI, ctx) -> int:
        """Tap one of two markers overlaid on the camera feed to choose where counting starts from
        (1, or 5 past "Start at 1"), or just tap Start at 1 -- demonstrates `RunUI.ask()`'s
        `objects=`/`on_tap=` for picking between objects shown on the overlay, including the
        selected-highlight pattern (`ctx.update_user_panel(selected_instance_id=...)`) from
        `docs/UI_JSON_SCHEMA.md`. Standalone, or with no tap, starts at 1."""
        markers = [
            {
                "instance_id": "marker_low",
                "perception_class": "marker",
                "box": [20.0, 20.0, 80.0, 80.0],
                "source": "overlay",
            },
            {
                "instance_id": "marker_high",
                "perception_class": "marker",
                "box": [_OVERLAY_WIDTH - 80.0, 20.0, _OVERLAY_WIDTH - 20.0, 80.0],
                "source": "overlay",
            },
        ]
        tapped = {"id": None}

        def on_tap(instance_id):
            tapped["id"] = instance_id
            ctx.update_user_panel(selected_instance_id=instance_id)  # the highlight follows the tap

        ui.ask(
            "Tap a marker to choose where counting starts from, or Start at 1.",
            [{"button_id": "start_at_1", "label": "Start at 1"}],
            objects=markers,
            on_tap=on_tap,
        )
        return 5 if tapped["id"] == "marker_high" else 1

    def _wait_for_stop(self, duration: float, ctx) -> dict | None:
        """Sleeps `duration` seconds in short steps, honouring cancel and
        polling for the Stop button throughout -- not just once beforehand --
        so a tap during any interval, including the action's last one, is
        never missed. Deliberately does NOT call ctx.checkpoint() to honour a
        pause: checkpoint() blocks for the whole pause, which would stop this
        loop from reaching poll_ui_event() and make the Stop button unable to
        end a paused action. Instead this keeps polling (and still raises
        Cancelled promptly) while paused, and only counts a paused tick
        towards `duration` once the pause lifts. Returns the Stop event as
        soon as it's seen, else None once the full duration has elapsed."""
        remaining = duration
        while True:
            if ctx.cancelled():
                ctx.checkpoint()  # raises Cancelled; never blocks since cancelled() is already true
            event = ctx.poll_ui_event()
            if (
                event is not None
                and event.get("type") == "button"
                and event.get("button_id") == "stop"
            ):
                return event
            if ctx.paused():
                time.sleep(self._UI_POLL_PERIOD_S)
                continue
            if remaining <= 0:
                return None
            step = min(self._UI_POLL_PERIOD_S, remaining)
            time.sleep(step)
            remaining -= step


def main():
    run(Capability)


if __name__ == "__main__":
    main()
