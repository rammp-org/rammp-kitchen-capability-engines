"""The run's panel on the user's screen: what step it is at, a progress bar, and a Stop button.

It goes to the UI bridge through the action's ctx (CapabilityEngine's update_user_panel /
poll_ui_event, docs/UI_JSON_SCHEMA.md in the repo). ctx.progress, by contrast, only reaches the
orchestrator and is never shown.

Nothing here waits on the UI. Standalone (no orchestrator, so no session token) the engine skips
every call, and on an engine built before the UI calls existed the ctx has none: then this does
nothing, and the run is the same as without it.
"""

import time

TITLE = "Microwave button press"
STOP = "stop"
# step (press_button's run.step names, and "done") -> what the user reads
TEXT = {
    "preflight": "Checking the arm and cameras",
    "detect": "Looking for the microwave",
    "stage": "Moving in front of the buttons",
    "redetect": "Finding the button",
    "plan": "Planning the press",
    "approach": "Reaching for the button",
    "press": "Pressing the button",
    "return": "Moving back",
    "done": "Done",
}
PAUSED_POLL_S = 0.1


class Panel:
    def __init__(self, ctx, target: str, dry_run: bool):
        self.ctx = ctx
        self.target = target.replace("_", " ")
        self.dry_run = dry_run
        self._update = getattr(ctx, "update_user_panel", None)
        self._poll = getattr(ctx, "poll_ui_event", None)
        self._paused = getattr(ctx, "paused", None)

    def show(self, step: str, fraction: float, pausable: bool = True, stop: bool = True) -> None:
        """Put the step on the panel. `fraction` is the run's 0..1; the panel's bar is 0..100."""
        if self._update is None:
            return
        message = f"{TEXT.get(step, step)} ({self.target})"
        if self.dry_run:
            message += " -- dry run, the arm does not move"
        panel = {
            "title": TITLE,
            "message": message,
            "progress": {"phase": step, "pct": 100.0 * float(fraction), "pausable": bool(pausable)},
        }
        if stop:
            panel["buttons"] = [{"button_id": STOP, "label": "Stop"}]
        self._update(panel_json=panel)

    def stop_pressed(self) -> bool:
        """Whether the user has tapped Stop since the last call (it consumes the tap)."""
        if self._poll is None:
            return False
        event = self._poll()
        return bool(event) and event.get("type") == "button" and event.get("button_id") == STOP

    def wait_out_pause(self) -> bool:
        """While the action is paused, keep watching for Stop: ctx.checkpoint() blocks for the whole
        pause and would leave a paused run unstoppable from the panel. True = Stop was tapped."""
        if self._paused is None:
            return self.stop_pressed()
        while True:
            if self.stop_pressed():
                return True
            if not self._paused() or self.ctx.cancelled():
                return False
            time.sleep(PAUSED_POLL_S)
