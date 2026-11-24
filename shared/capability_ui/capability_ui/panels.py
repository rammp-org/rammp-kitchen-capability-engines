"""The panels a capability shows on the UI while one of its actions runs, with a Stop on every one.

The orchestrator hands the UI to the engine for the length of `execute()` and takes it back the moment `execute()`
returns, so everything here happens inside `execute()`: a final notice is shown and waited on (bounded) before the
action returns or raises.

Stop works through the engine's own cancel, the same one the orchestrator's cancel uses: `self.arm` checks
`ctx.cancelled()` every cycle of a motion and stops it, and `ctx.checkpoint()` raises `Cancelled` between steps. While
the arm moves (`RunUI.moving` / `RunUI.motion`) a watcher thread is the only reader of `ctx.poll_ui_event()`, so it
never takes a tap meant for one of the capability's own questions; it stops reading before any question is asked.

Typical use, in `execute()`:

    ui = RunUI(ctx, "Open microwave", cancel=lambda: cancel_own_run(self))
    try:
        result = ui.motion("Opening the door: the arm is moving.", lambda: self._open(ctx), stopped_note=...)
    except EngineError as error:
        ui.error(f"Couldn't open the door: {error}.")
        raise
    ui.done("Microwave door opened.")
    return result

A capability with its own camera (e.g. a wrist cam) can also show it alongside the panels above, via
`publish_overlay_image`/`show_overlay`/`hide_overlay` (thin wrappers over `ctx.publish_overlay_image`/
`ctx.set_overlay_visible` -- see `capability_engine.engine.Context` for the underlying contract), or the
`overlay_shown()` bracket below for the common show-then-hide-on-every-exit-path shape:

    with ui.overlay_shown():
        for frame in frames:
            ui.publish_overlay_image(frame)
            ...

Nothing here imports ROS: the tests drive it with a fake ctx.
"""

import contextlib
import textwrap
import threading
import time

STOP = {"button_id": "stop", "label": "Stop"}
DONE = {"button_id": "done", "label": "Done"}

# How long a final notice (Done) waits for its tap before execute() returns anyway: the action's outcome is already
# decided, so not tapping must never turn it into a failure.
NOTICE_TIMEOUT_S = 60.0
# How long a question waits for an answer (e.g. "move your wheelchair, then Continue"); no answer counts as Stop.
PROMPT_TIMEOUT_S = 600.0
# A panel the bridge dropped (rejected or aborted goal, bridge restart) is re-sent at most this often. An open panel
# is never re-sent: re-sending one every second redrew it on the UI, which flashed (microwave, 11-12).
REFRESH_S = 1.0
POLL_S = 0.1

# How long `motion()` waits, right after opening a fresh UI goal, for that goal's accept round-trip to finish before
# starting the caller's `fn()`. Opening a goal (e.g. the first panel of a new motion, after a prior question's goal
# ended on its button press) is asynchronous: a call to update_user_panel() made while it's still *opening* (not yet
# accepted) only merges its content locally and sends nothing over the network (capability_engine.Context's own
# "opening" branch) -- so a fast-moving loop (e.g. counting once a second) can silently overwrite step 1's content
# with step 2's, step 2's with step 3's, and so on, before the goal ever actually opens, and the first steps the user
# was meant to see on screen never reach it. Bounded so a bridge that's down or slow to accept never stalls the
# capability's real work indefinitely -- best effort only, same as every other wait in this module.
GOAL_OPEN_WAIT_S = 3.0

STOPPED_MESSAGE = "Stopped. The arm is holding still where it was."

# The UI (Unreal) doesn't wrap panel text: a long line runs off the screen (fridge, 11-12). Messages are broken into
# lines of at most this many characters before they are sent.
WRAP_CHARS = 40
# Unreal shows the message as ONE rich-text run, <body-sm-light>...</>, and a line break inside a run breaks it (the
# tags showed on screen, 11-12). So each line break closes the run and opens it again on the next line.
UI_TEXT_STYLE = "body-sm-light"


def wrap(text, width=None):
    """`text` broken into lines of at most `width` characters (paragraphs kept; long words not split), each line its
    own rich-text run for Unreal (UI_TEXT_STYLE)."""
    width = width or WRAP_CHARS
    lines = []
    for paragraph in str(text).split("\n"):
        lines += textwrap.wrap(paragraph, width, break_long_words=False, break_on_hyphens=False) or [""]
    return f"</>\n<{UI_TEXT_STYLE}>".join(lines)


def panel_open(ctx):
    """Whether the run's UI goal is still open (or opening), from the engine's own run record; True when it can't
    tell, so nothing is re-sent needlessly."""
    run = getattr(ctx, "_run", None)
    if run is None or not hasattr(run, "ui_goal_handle"):
        return True
    lock = getattr(run, "ui_lock", None)
    if lock is None:
        return bool(run.ui_goal_handle is not None or getattr(run, "ui_opening", False))
    with lock:
        return bool(run.ui_goal_handle is not None or getattr(run, "ui_opening", False))


def engine_stopping(ctx):
    """Whether the engine is shutting down (SIGTERM/SIGINT: CapabilityEngine.stop() has set its `_stopping`), from
    the engine's own state; False when it can't tell."""
    return getattr(getattr(ctx, "_engine", None), "_stopping", None) is not None


def cancel_own_run(engine):
    """Cancel the action this engine is running, exactly as a CANCEL on its `/<engine_id>/control` service does (the
    orchestrator's cancel). The motion in flight stops; `execute()` sees `Cancelled` at its next check."""
    from rammp_orchestrator_interfaces.srv import Control

    engine._on_control(Control.Request(command=Control.Request.CANCEL), Control.Response())


class RunUI:
    """One action's panels. `cancel` cancels the running action (normally `lambda: cancel_own_run(self)`)."""

    def __init__(self, ctx, title, cancel, clock=time.monotonic, sleep=time.sleep):
        self.ctx = ctx
        self.title = title
        self._cancel = cancel
        self._clock = clock
        self._sleep = sleep
        self.user_stopped = False  # the user pressed Stop (vs. an orchestrator cancel)
        self.connected = None  # None until the first panel is sent; False standalone (no UI session)
        self._panel = None
        self._lock = threading.Lock()
        self._watch = None
        self._watch_quit = threading.Event()

    # -- sending ---------------------------------------------------------------------------------------------------

    def send(self, message, buttons=(), progress=None, title=None, objects=None, selected=None, style=None):
        """Show a panel now. Returns whether there is a UI to show it on (False standalone). `style` (e.g. "error",
        "warning") is passed through as the panel's "style" for the UI to render; omitted when None."""
        panel = {"title": title or self.title, "message": wrap(message), "buttons": list(buttons)}
        if progress is not None:
            panel["progress"] = progress
        if style:
            panel["style"] = style
        with self._lock:
            self._panel = (panel, objects)
        kwargs = {"panel_json": panel}
        if objects is not None:
            kwargs["objects"] = objects
        if selected is not None:
            kwargs["selected_instance_id"] = selected
        ok = bool(self.ctx.update_user_panel(**kwargs))
        if self.connected is None or ok:
            self.connected = ok
        return ok

    def _resend_if_dropped(self):
        if panel_open(self.ctx):
            return
        with self._lock:
            panel, objects = self._panel if self._panel else (None, None)
        if panel is not None:
            kwargs = {"panel_json": panel}
            if objects is not None:
                kwargs["objects"] = objects
            self.ctx.update_user_panel(**kwargs)

    def _wait_while_opening(self, timeout_s):
        """Blocks (bounded by `timeout_s`) while this run's UI goal is mid-open (accepted or
        rejected yet, not either) -- see GOAL_OPEN_WAIT_S's own comment for why `motion()` calls
        this right after its first panel. A no-op wherever `ctx` has no such concept (e.g. the
        tests' simpler fakes, or no UI session at all)."""
        run = getattr(self.ctx, "_run", None)
        if run is None or not hasattr(run, "ui_opening"):
            return
        deadline = self._clock() + timeout_s
        while getattr(run, "ui_opening", False) and self._clock() < deadline:
            self._sleep(POLL_S)

    def select_object(self, instance_id, message=None, buttons=None):
        """Highlight `instance_id` (the selected-instance-id-follows-the-tap pattern) on the
        currently shown panel, optionally updating its message and/or `buttons` too (e.g. a confirm
        button's label changing to reflect what got selected) -- title/objects/style stay as last
        sent, and anything left as None (the default) is kept unchanged too. `buttons` is filtered
        and Stop appended exactly like `ask()` does, so pass just the non-Stop buttons. Call this
        from `on_tap` (see `ask()`) so the user sees their tap took effect, instead of a panel that
        looks unchanged after tapping an object."""
        with self._lock:
            panel, objects = self._panel if self._panel else (None, None)
        if panel is None:
            return bool(self.ctx.update_user_panel(selected_instance_id=instance_id))
        panel = dict(panel)
        if message is not None:
            panel["message"] = wrap(message)
        if buttons is not None:
            panel["buttons"] = [b for b in buttons if b["button_id"] != STOP["button_id"]] + [STOP]
        with self._lock:
            self._panel = (panel, objects)
        kwargs = {"panel_json": panel, "selected_instance_id": instance_id}
        if objects is not None:
            kwargs["objects"] = objects
        return bool(self.ctx.update_user_panel(**kwargs))

    # -- this run's own camera overlay (e.g. a wrist cam), alongside the panels above -----------------------------

    def publish_overlay_image(self, image) -> bool:
        """Publish one frame of this run's own camera (`ctx.publish_overlay_image`) to the UI bridge's
        shared overlay topic, stamped with this run's session token. `image` is a `sensor_msgs/Image`.
        Returns False (no-op, logged by the engine) if this run holds no session token -- same as
        `send()` above, there is nothing more to do standalone."""
        return bool(self.ctx.publish_overlay_image(image))

    def show_overlay(self) -> bool:
        """Show this run's overlay camera feed in UE (`ctx.set_overlay_visible(True)`) -- UE's `Open
        Camera Feed` Remote Control call. Not a gate on `publish_overlay_image()` above: keep
        publishing frames regardless of visibility, so the feed is already live the moment it's
        shown."""
        return bool(self.ctx.set_overlay_visible(True))

    def hide_overlay(self) -> bool:
        """Hide this run's overlay camera feed in UE (`ctx.set_overlay_visible(False)`) -- UE's `Close
        Camera Feed` Remote Control call. The orchestrator's own end-of-execution cleanup also
        unconditionally hides the overlay, so calling this explicitly is redundant by the time
        `execute()` returns -- use it (or `overlay_shown()` below) when the feed should hide the
        instant this run's own loop ends, rather than wait on that later cleanup."""
        return bool(self.ctx.set_overlay_visible(False))

    @contextlib.contextmanager
    def overlay_shown(self):
        """Brackets a block with `show_overlay()`/`hide_overlay()`, hiding on every exit path --
        normal completion, an exception, or Stop/Cancelled -- not just the happy path:

            with ui.overlay_shown():
                for frame in frames:
                    ui.publish_overlay_image(frame)
                    ...
        """
        self.show_overlay()
        try:
            yield
        finally:
            self.hide_overlay()

    # -- while the arm moves: a panel with Stop, watched from a thread -------------------------------------------

    def moving(self, message, pct=None, phase=""):
        """Show (or update) the progress panel with Stop. Starts the Stop watcher if it isn't running."""
        progress = None if pct is None else {"phase": phase, "pct": float(pct), "pausable": False}
        if not self.send(message, [STOP], progress):
            return
        if self._watch is None:
            self._watch_quit.clear()
            self._watch = threading.Thread(target=self._watch_loop, name="ui-stop-watch", daemon=True)
            self._watch.start()

    def quiet(self):
        """Stop the Stop watcher (before asking anything, and once the motion is over)."""
        watch, self._watch = self._watch, None
        if watch is not None:
            self._watch_quit.set()
            watch.join(timeout=2.0)

    def _watch_loop(self):
        refresh_at = self._clock() + REFRESH_S
        while not self._watch_quit.is_set():
            event = self.ctx.poll_ui_event()
            if event and event.get("type") == "button" and event.get("button_id") == STOP["button_id"]:
                self._stop()
                return
            if self._clock() >= refresh_at:
                self._resend_if_dropped()
                refresh_at = self._clock() + REFRESH_S
            self._watch_quit.wait(POLL_S)

    def _stop(self):
        if not self.user_stopped:
            self.user_stopped = True
            self._cancel()

    def motion(self, message, fn, stopped_note=None):
        """Run `fn()` (the moving part) under the Stop panel. If the user presses Stop, the cancel stops the motion
        and whatever `fn` raises is re-raised after a "Stopped" notice; `stopped_note()` can add what state the
        arm was left in (e.g. "The hand is still on the handle.").

        Waits (bounded, GOAL_OPEN_WAIT_S) for this panel's own UI goal-open to finish before running `fn()`, so a
        fast-moving `fn` (e.g. counting once a second) doesn't outrun the goal's accept round-trip and silently lose
        its first few updates -- see GOAL_OPEN_WAIT_S's own comment."""
        self.moving(message)
        self._wait_while_opening(GOAL_OPEN_WAIT_S)
        try:
            return fn()
        except BaseException:
            self.quiet()
            if self.user_stopped:
                note = ""
                if stopped_note is not None:
                    try:
                        note = stopped_note() or ""
                    except Exception:  # the notice must still show
                        note = ""
                self.notice(f"{STOPPED_MESSAGE} {note}".strip(), title=f"{self.title}: stopped", style="warning")
            raise
        finally:
            self.quiet()

    # -- waiting for a tap -------------------------------------------------------------------------------------------

    def _wait(self, wanted, timeout_s, on_tap=None, raise_on_cancel=True):
        """The id of the first tapped button in `wanted`, or None after `timeout_s`. Stop cancels the action and
        raises (via ctx.checkpoint()). `on_tap(instance_id)` handles object taps."""
        self.quiet()
        deadline = self._clock() + timeout_s
        refresh_at = self._clock() + REFRESH_S
        while True:
            # A cancel from elsewhere (the orchestrator) ends the wait; our own Stop doesn't, so its "Stopped"
            # notice stays up until Done. Unless the engine is shutting down (docker stop): then it ends too, or
            # execute() outlives the engine's stop wait and the container exits 1 after it.
            if self.ctx.cancelled() and not (
                self.user_stopped and not raise_on_cancel and not engine_stopping(self.ctx)
            ):
                if not raise_on_cancel:
                    return None
                self.ctx.checkpoint()  # raises Cancelled
            event = self.ctx.poll_ui_event()
            if event:
                if event.get("type") == "button":
                    button = event.get("button_id")
                    if button in wanted:
                        return button
                    if button == STOP["button_id"]:
                        self._stop()
                        if not raise_on_cancel:
                            return None
                        self.ctx.checkpoint()
                        raise RuntimeError("Stop pressed but the action wasn't cancelled")
                elif event.get("type") == "object_tapped" and on_tap is not None:
                    on_tap(event.get("instance_id"))
            now = self._clock()
            if now >= deadline:
                return None
            if now >= refresh_at:
                self._resend_if_dropped()
                refresh_at = now + REFRESH_S
            self._sleep(POLL_S)

    def ask(self, message, buttons, timeout_s=PROMPT_TIMEOUT_S, title=None, objects=None, on_tap=None, style=None):
        """A question with `buttons` plus Stop. Returns the tapped button's id, or None when there is no UI
        (standalone) or no answer within `timeout_s`. Stop cancels the action (raises `Cancelled`). With `objects`:
        `on_tap(instance_id)` runs on every object tap (the user may tap several before deciding) -- call
        `select_object()` from it so the panel visibly reflects the tap, not just a tap the UI bridge accepted."""
        buttons = [b for b in buttons if b["button_id"] != STOP["button_id"]] + [STOP]
        if not self.send(message, buttons, title=title, objects=objects, style=style):
            return None
        return self._wait({b["button_id"] for b in buttons} - {STOP["button_id"]}, timeout_s, on_tap)

    def retry(self, message, label, title=None, style="warning"):
        """'<message> Press <label> to try again.' with <label> and Stop. True to try again."""
        text = f"{message} Press {label} to try again."
        return self.ask(text, [{"button_id": "retry", "label": label}], title=title, style=style) == "retry"

    def notice(self, message, title=None, timeout_s=NOTICE_TIMEOUT_S, style=None):
        """A final message with Done, waited on for at most `timeout_s`. Never raises: by now the action's outcome
        is decided (an orchestrator cancel just ends the wait).

        Waits (bounded, GOAL_OPEN_WAIT_S) for this panel's own UI goal-open to finish, the same reason `motion()`
        does right after its own first panel (see GOAL_OPEN_WAIT_S's own comment): a notice sent right behind an
        earlier panel whose goal hasn't opened yet (e.g. a run that fails before its first real progress step, so
        there is next to no time between that first panel and this one) only merges its content locally, and
        without this wait nothing here notices or waits for the merge to actually reach the bridge -- the user
        would keep seeing the earlier (stale) panel until some unrelated event (a Stop tap) happens to end that
        goal."""
        self.quiet()
        if not self.send(message, [DONE], title=title, style=style):
            return
        self._wait_while_opening(GOAL_OPEN_WAIT_S)
        try:
            self._wait({DONE["button_id"]}, timeout_s, raise_on_cancel=False)
        except Exception:  # a notice must never change the action's result
            pass

    def done(self, message, mock=False):
        if mock:
            message = f"Mock: {message} No arm motion."
        self.notice(message)

    def error(self, message, title=None, timeout_s=NOTICE_TIMEOUT_S):
        self.notice(message, title=title or f"{self.title}: failed", timeout_s=timeout_s, style="error")

    def refuse(self, message):
        """A refusal before anything moved (e.g. execution disabled)."""
        self.notice(message, title=f"{self.title}: not started", style="warning")
