"""The touchscreen's one-press action: `look_in_fridge` runs `peer`, executing, under RunUI.

One progress panel with Stop (capability_ui.RunUI) is up from the press to the end of the run,
through scene homing, homing, the joystick teleop and a tap's pick. Its message follows the run's
steps (ui_text.phase_text, from ctx.progress); FoodTargets' own messages and its tappable food
boxes go onto the same panel, Stop kept. Stop cancels the run through the engine's own cancel: the
stream's next send raises Cancelled and closing the stream holds the arm, and self.arm stops a
planned move in flight. Then a "Stopped" notice says what the hand holds; a failure says what
happened in plain words; success says Done. Every final notice waits at most capability_ui's
NOTICE_TIMEOUT_S.

The bridge's events have one reader while the run moves, RunUI's Stop watcher (a second reader
would take taps meant for the other). _TapRelay keeps the object taps the watcher reads, and
RunCtx hands them to FoodTargets through its ctx.poll_ui_event(). The first wrist frame
FoodTargets publishes for the overlay also shows the overlay (ctx.set_overlay_visible); the
orchestrator hides it when the run ends.

The orchestrator cancels a run that sends no progress for 120 s (execute_timeout), and teleop
sends none after it starts: the button's teleop ends by itself after LOOK_S (a Done notice), and a
progress goes out before each final notice, so the run is never silent that long.

No ROS imported here at load time: the UI tests drive it with a fake ctx.
"""

import threading

from capability_ui import RunUI, cancel_own_run
from capability_ui.panels import STOP

from . import ui_text

# How long one press looks around (peer's max_teleop_s): under the orchestrator's 120 s without
# progress, with room for the pick that may follow a tap. Press again to keep looking.
LOOK_S = 90.0

# Why the button is greyed out with arm motion off (availability(), REVIEW_RULES O1: an action
# reported available must be runnable).
DISABLED_REASON = "arm motion is off: fridge_peering_mode was started with allow_execute false"

# The UI's buttons (the orchestrator shows the label): what each runs, with fixed params, and on
# which tapped class it is offered. `peer` and `pick` stay callable directly (dry runs by default,
# the CLI), not offered.
UI_ACTIONS = {
    "look_in_fridge": (
        "peer",
        "Look Inside Fridge",
        {"execute": True, "max_teleop_s": LOOK_S},
        ["fridge"],
    ),
}


class _TapRelay:
    """The ctx RunUI reads the bridge through: each event goes on to RunUI as it is, and an object
    tap is also kept (the latest only, as the bridge does) for FoodTargets."""

    def __init__(self, ctx):
        self._ctx = ctx
        self._tap = None
        self._lock = threading.Lock()

    def poll_ui_event(self):
        event = self._ctx.poll_ui_event()
        if event and event.get("type") == "object_tapped":
            with self._lock:
                self._tap = event
        return event

    def take_tap(self):
        with self._lock:
            tap, self._tap = self._tap, None
        return tap

    def __getattr__(self, name):
        return getattr(self._ctx, name)


class RunCtx:
    """The ctx `peer` runs with under the UI: the action's ctx, with its progress shown on the
    panel, its panels (FoodTargets') sent through RunUI with Stop, its taps from the relay, and the
    last progress phase kept (what the hand holds after a Stop)."""

    def __init__(self, ctx, ui, relay):
        self._ctx = ctx
        self._ui = ui
        self._relay = relay
        self.phase = ""
        self._message = ui_text.MOVING
        self._overlay_shown = False

    def progress(self, phase, pct=0.0, message="", *args, **kwargs):
        self.phase = phase
        text = ui_text.phase_text(phase, pct, message)
        if text and text != self._message:
            self._message = text
            self._ui.moving(text)
        return self._ctx.progress(phase, pct, message, *args, **kwargs)

    def update_user_panel(self, panel_json=None, objects=None, selected_instance_id=None):
        if panel_json is not None and panel_json.get("message"):
            self._message = panel_json["message"]
        return self._ui.send(self._message, [STOP], objects=objects, selected=selected_instance_id)

    def poll_ui_event(self):
        return self._relay.take_tap()

    def publish_overlay_image(self, image):
        if not self._overlay_shown:
            self._overlay_shown = True
            self._ctx.set_overlay_visible(True)
        return self._ctx.publish_overlay_image(image)

    def __getattr__(self, name):
        return getattr(self._ctx, name)


def run(engine, ui_id, ctx):
    """One press of `ui_id`: its action with its fixed params, under RunUI. Returns the action's
    result or raises what it raised, after the final notice."""
    from capability_engine import EngineError

    action_id, _, params, _ = UI_ACTIONS[ui_id]
    title = ui_text.TITLE
    log = engine.get_logger()
    relay = _TapRelay(ctx)
    ui = RunUI(relay, title, cancel=lambda: cancel_own_run(engine), logger=log)
    if not engine.mock and not engine.cfg["allow_execute"]:
        ui.refuse(ui_text.disabled(title))
        raise EngineError(
            "execution_disabled",
            f"{title} is disabled: launch with allow_execute:=true for an attended run.",
            recoverable=True,
        )
    run_ctx = RunCtx(ctx, ui, relay)
    try:
        result = ui.motion(
            ui_text.MOVING,
            lambda: engine.execute(action_id, dict(params), run_ctx),
            stopped_note=lambda: _stopped_note(ctx, run_ctx.phase),
        )
    except Exception as error:  # an EngineError, Cancelled, an arm error, a bug
        log.error(f"{ui_id}: {error!r}")
        # Stop showed its own notice; an orchestrator cancel has taken the UI back
        if not (ui.user_stopped or ctx.cancelled()):
            hand = ui_text.hand_note(run_ctx.phase)
            _notice_progress(ctx, "failed")
            if isinstance(error, EngineError):
                ui.error(ui_text.failure(str(error), hand, error.error_code))
            else:
                ui.error(ui_text.unexpected(error, hand))
        raise
    _notice_progress(ctx, "done")
    ui.done(ui_text.done(result), mock=bool(result.get("mock")))
    return result


def _stopped_note(ctx, phase):
    """What the hand holds, for RunUI's Stopped notice, which shows right after: a progress goes
    out first, as before the other final notices."""
    _notice_progress(ctx, "stopped")
    return ui_text.hand_note(phase)


def _notice_progress(ctx, outcome):
    """A progress before a final notice: the orchestrator's 120 s without progress then counts
    from here, not from the start of teleop."""
    ctx.progress("notice", 100.0, f"{outcome}: the final notice is up", pausable=False)
