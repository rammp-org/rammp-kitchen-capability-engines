"""The user's panel: every step is shown with the bar on the UI's 0..100 scale, Stop is read back,
and a ctx without the UI calls (an engine built before them, the tests' fakes) changes nothing."""

from button_press.ui_panel import STOP, Panel


class UiCtx:
    def __init__(self, events=(), paused=()):
        self.panels, self.events, self.pauses = [], list(events), list(paused)

    def update_user_panel(self, panel_json=None, objects=None, selected_instance_id=None):
        self.panels.append(panel_json)
        return True

    def poll_ui_event(self):
        return self.events.pop(0) if self.events else None

    def paused(self):
        return self.pauses.pop(0) if self.pauses else False

    def cancelled(self):
        return False


def test_a_step_is_shown_with_its_bar_in_percent_and_a_stop_button():
    ctx = UiCtx()
    Panel(ctx, "start_30s", dry_run=False).show("press", 0.7, pausable=False)
    (panel,) = ctx.panels
    assert panel["message"] == "Pressing the button (start 30s)"
    assert panel["progress"] == {"phase": "press", "pct": 70.0, "pausable": False}
    assert panel["buttons"] == [{"button_id": STOP, "label": "Stop"}]


def test_a_dry_run_says_the_arm_does_not_move_and_done_has_no_stop_button():
    ctx = UiCtx()
    panel = Panel(ctx, "start_30s", dry_run=True)
    panel.show("stage", 0.25)
    panel.show("done", 1.0, pausable=False, stop=False)
    assert "dry run" in ctx.panels[0]["message"]
    assert "buttons" not in ctx.panels[1] and ctx.panels[1]["progress"]["pct"] == 100.0


def test_only_the_stop_button_counts_as_stop():
    events = [{"type": "object_tapped", "instance_id": "microwave_1"}, {"type": "button", "button_id": "other"}]
    panel = Panel(UiCtx(events + [{"type": "button", "button_id": STOP}]), "start_30s", dry_run=False)
    assert [panel.stop_pressed() for _ in range(4)] == [False, False, True, False]


def test_stop_is_seen_while_the_action_is_paused(monkeypatch):
    monkeypatch.setattr("button_press.ui_panel.time.sleep", lambda s: None)
    ctx = UiCtx(events=[None, None, {"type": "button", "button_id": STOP}], paused=[True, True, True])
    assert Panel(ctx, "start_30s", dry_run=False).wait_out_pause() is True
    ctx = UiCtx(paused=[True, False])
    assert Panel(ctx, "start_30s", dry_run=False).wait_out_pause() is False  # resumed, no Stop


def test_a_ctx_without_the_ui_calls_does_nothing():
    class OldCtx:
        pass

    panel = Panel(OldCtx(), "start_30s", dry_run=False)
    panel.show("press", 0.7)
    assert panel.stop_pressed() is False and panel.wait_out_pause() is False
