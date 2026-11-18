"""Offline UI flow tests for example's `count_with_runui_panel`: node.py's own methods (plus
`_build_stripe_overlay_image`), loaded with `ast` without ROS, run against the real
`capability_ui.RunUI` driven by a fake ctx that scripts button presses and object taps -- not robot
validation, this is the example capability's own RunUI reference action. Covers most of RunUI's
interactive surface: a plain confirm button (`_ask_to_start`), an object-overlay pick with the
selected-highlight pattern (`_ask_start_marker`), the progress-with-Stop motion panel, and Stop
cancelling the action mid-run."""

import ast
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "capabilities/example/example"
sys.path.insert(0, str(ROOT / "shared/capability_ui"))

from capability_ui import panels  # noqa: E402
from capability_ui.panels import RunUI  # noqa: E402


def classes(path, names):
    tree = ast.parse(path.read_text())
    body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    scope = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), scope)
    return scope


_ENGINE_PY = ROOT / "capability_engine/capability_engine/engine.py"
EngineError = classes(_ENGINE_PY, {"EngineError"})["EngineError"]


class Cancelled(Exception):
    pass


class Action:
    def __init__(self, action_id, label, params_schema=None):
        self.action_id, self.label = action_id, label


class Base:
    pass


class Image:
    """Stand-in for sensor_msgs.msg.Image: node.py only sets attributes and never inspects them."""

    def __init__(self):
        self.height = self.width = self.step = self.is_bigendian = 0
        self.encoding = ""
        self.data = b""


class FakeTime:
    """Replaces node.py's module-level `time` for the loaded Capability: sleep() is instant by
    default (a normal count never waits in real time); `on_sleep(n)` lets a test act (e.g. tap Stop)
    partway through, on the Nth call."""

    def __init__(self, on_sleep=None):
        self.n = 0
        self._on_sleep = on_sleep

    def sleep(self, _duration):
        self.n += 1
        if self._on_sleep is not None:
            self._on_sleep(self.n)


def load():
    """node.py's `_OVERLAY_*` constants, `_build_stripe_overlay_image`, and the `Capability` class
    with only the `count_with_runui_panel` path kept (base: `Base`, no hardware/ROS setup)."""
    tree = ast.parse((PKG / "node.py").read_text())
    consts = {"_OVERLAY_WIDTH", "_OVERLAY_HEIGHT", "_STRIPE_BG", "_STRIPE_FG", "_STRIPE_WIDTH"}

    def is_const(n):
        return isinstance(n, ast.Assign) and any(getattr(t, "id", "") in consts for t in n.targets)

    body = [n for n in tree.body if is_const(n)]
    overlay_fn = "_build_stripe_overlay_image"
    body += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == overlay_fn]
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Capability")
    cls.bases = [ast.Name(id="Base", ctx=ast.Load())]
    keep = {"execute", "_count_with_runui_panel", "_ask_to_start", "_ask_start_marker"}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in keep]
    body.append(cls)
    scope = {
        "Action": Action,
        "Base": Base,
        "Cancelled": Cancelled,
        "EngineError": EngineError,
        "RunUI": RunUI,
        "Image": Image,
        "time": FakeTime(),
        # the engine's own cancel (Control.CANCEL on itself): here, the fake ctx's
        "cancel_own_run": lambda engine: engine.test_ctx.cancel(),
    }
    ast.fix_missing_locations(cls)
    exec(compile(ast.Module(body=body, type_ignores=[]), "node.py", "exec"), scope)
    return scope


SCOPE = load()
Capability = SCOPE["Capability"]


def button(button_id):
    return {"type": "button", "button_id": button_id}


def tap_object(instance_id):
    return {"type": "object_tapped", "instance_id": instance_id}


class FakeCtx:
    """As the engine's ctx: an event queue (poll_ui_event pops one at a time), cancel() sets what
    checkpoint() raises on, update_user_panel() False without a UI session. `replies`: one entry per
    panel that asks something (has buttons beyond Stop, or objects to tap) -- each entry is the list
    of events (button taps / object taps) queued the moment that panel is shown, consumed in order
    by RunUI's own wait loop. A plain progress-with-Stop panel (buttons == ["stop"], no objects)
    never consumes a reply."""

    def __init__(self, replies=(), session=True):
        self.session = session
        self.replies = list(replies)
        self.sent = []
        self.overlay_images = []
        self.overlay_visible_calls = []
        self.progress_calls = []
        self._queue = []
        self._lock = threading.Lock()
        self._cancelled = threading.Event()

    def update_user_panel(self, panel_json=None, objects=None, selected_instance_id=None):
        selected = selected_instance_id
        self.sent.append({"panel": panel_json, "objects": objects, "selected": selected})
        # A call with `selected` set is a resend of the *same* question (RunUI.select_object(),
        # highlighting/confirming a tap -- e.g. the example's marker picker), not a new one: only
        # a fresh panel (no `selected`) consumes the next scripted reply, else the marker picker's
        # own select_object() resend would steal the reply meant for the final "Done" notice.
        is_question = (
            panel_json is not None
            and selected is None
            and ([b["button_id"] for b in panel_json.get("buttons", [])] != ["stop"] or objects)
        )
        if self.session and is_question and self.replies:
            with self._lock:
                self._queue.extend(self.replies.pop(0))
        return self.session

    def publish_overlay_image(self, image):
        self.overlay_images.append(image)
        return self.session

    def set_overlay_visible(self, visible):
        self.overlay_visible_calls.append(visible)
        return self.session

    def poll_ui_event(self):
        with self._lock:
            return self._queue.pop(0) if self._queue else None

    def tap(self, button_id):
        with self._lock:
            self._queue.append(button(button_id))

    def cancel(self):
        self._cancelled.set()

    def cancelled(self):
        return self._cancelled.is_set()

    def checkpoint(self):
        if self.cancelled():
            raise Cancelled()

    def progress(self, phase, pct, message="", pausable=True):
        self.progress_calls.append((phase, pct, message))


def wait_until(cond, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


@pytest.fixture(autouse=True)
def fast_watch(monkeypatch):
    monkeypatch.setattr(panels, "POLL_S", 0.002)


def make(replies=(), session=True, on_sleep=None):
    ctx = FakeCtx(replies, session)
    node = Capability()
    node.test_ctx = ctx
    node.mock = False
    node.get_logger = lambda: SimpleNamespace(info=_quiet, warning=_quiet, error=_quiet)
    SCOPE["time"] = FakeTime(on_sleep)
    return node, ctx


def _quiet(*_a, **_kw):
    pass


def make_ui(node):
    cancel = lambda: SCOPE["cancel_own_run"](node)  # noqa: E731
    return RunUI(node.test_ctx, "Counting", cancel=cancel, logger=node.get_logger())


# -- _ask_to_start: a plain confirm button --------------------------------------------------------


def test_ask_to_start_proceeds_when_start_is_tapped():
    node, ctx = make(replies=[[button("start")]])
    ui = make_ui(node)
    node._ask_to_start(ui, 3)  # does not raise
    assert "Ready to count to 3" in ctx.sent[0]["panel"]["message"]
    assert {b["button_id"] for b in ctx.sent[0]["panel"]["buttons"]} == {"start", "stop"}


def test_ask_to_start_standalone_proceeds_without_waiting():
    node, ctx = make(session=False)
    ui = make_ui(node)
    node._ask_to_start(ui, 3)  # does not raise, no tap needed
    assert ui.connected is False


def test_ask_to_start_raises_no_answer_when_connected_and_never_tapped(monkeypatch):
    # RunUI.ask()/.notice() bind their timeout defaults at def time (PROMPT_TIMEOUT_S/
    # NOTICE_TIMEOUT_S, 600s/60s), so patching the module constants has no effect on calls that
    # don't pass timeout_s explicitly -- patch each function's own bound default instead.
    ask_defaults = RunUI.ask.__defaults__
    monkeypatch.setattr(RunUI.ask, "__defaults__", (0.03,) + ask_defaults[1:])
    notice_defaults = RunUI.notice.__defaults__
    notice_patched = (notice_defaults[0], 0.03, notice_defaults[2])
    monkeypatch.setattr(RunUI.notice, "__defaults__", notice_patched)
    node, ctx = make(replies=[[]])  # the question is shown, but nothing is ever tapped
    ui = make_ui(node)
    with pytest.raises(EngineError) as excinfo:
        node._ask_to_start(ui, 3)
    assert excinfo.value.error_code == "no_answer"
    assert ctx.sent[-1]["panel"]["message"] == "No answer: nothing was counted."


def test_ask_to_start_stop_cancels():
    node, ctx = make(replies=[[button("stop")]])
    ui = make_ui(node)
    with pytest.raises(Cancelled):
        node._ask_to_start(ui, 3)
    assert ctx.cancelled()


# -- _ask_start_marker: tap one of two objects overlaid on the camera feed ------------------------


def test_ask_start_marker_tap_marker_high_starts_at_5():
    node, ctx = make(replies=[[tap_object("marker_high"), button("start_at_1")]])
    ui = make_ui(node)
    assert node._ask_start_marker(ui) == 5
    ask = ctx.sent[0]
    assert {o["instance_id"] for o in ask["objects"]} == {"marker_low", "marker_high"}
    assert all(o["source"] == "overlay" for o in ask["objects"])
    # the tap both highlighted the marker and updated the panel's own message to confirm it
    selected = next(s for s in ctx.sent if s["selected"] == "marker_high")
    assert "high (5)" in selected["panel"]["message"]


def test_ask_start_marker_tap_start_at_1_without_an_object_tap_starts_at_1():
    node, ctx = make(replies=[[button("start_at_1")]])
    ui = make_ui(node)
    assert node._ask_start_marker(ui) == 1


def test_ask_start_marker_standalone_starts_at_1():
    node, ctx = make(session=False)
    ui = make_ui(node)
    assert node._ask_start_marker(ui) == 1


# -- the whole action: start -> pick a marker -> count under the motion panel -> done -------------

# start tapped, then marker_high tapped (plus the always-present "start at 1" button), then the
# final "Counted to N." notice's Done -- the 3 questions a full run asks, in order.
_FULL_FLOW_REPLIES = [
    [button("start")],
    [tap_object("marker_high"), button("start_at_1")],
    [button("done")],
]


def test_full_flow_starts_at_the_tapped_markers_count_and_shows_progress():
    node, ctx = make(replies=_FULL_FLOW_REPLIES)
    result = node._count_with_runui_panel(6, 0.0, ctx)
    assert result == {"counted_to": 6, "started_at": 5}
    # overlay shown for the whole interactive run (ask + count), hidden once at the end
    assert ctx.overlay_visible_calls == [True, False]
    assert len(ctx.overlay_images) >= 1
    # only n=5,6 were counted (started_at=5), never 1-4
    assert ctx.progress_calls == [
        ("counting", pytest.approx(100.0 * 5 / 6), "5/6"),
        ("counting", 100.0, "6/6"),
    ]
    moving = [s["panel"] for s in ctx.sent if s["panel"] and s["panel"].get("progress")]
    assert [m["message"] for m in moving] == ["5/6", "6/6"]
    assert ctx.sent[-1]["panel"]["message"] == "Counted to 6."


def test_full_flow_standalone_counts_from_1_with_no_waiting():
    node, ctx = make(session=False)
    result = node._count_with_runui_panel(3, 0.0, ctx)
    assert result == {"counted_to": 3, "started_at": 1}
    assert ctx.overlay_visible_calls == [True, False]


def test_stop_mid_count_cancels_and_shows_a_stopped_notice():
    node, ctx = make(replies=_FULL_FLOW_REPLIES)

    def on_sleep(n):
        if n == 1:  # Stop while counting is in flight, right after the first tick's sleep
            ctx.tap("stop")
            assert wait_until(ctx.cancelled)

    SCOPE["time"] = FakeTime(on_sleep)
    with pytest.raises(Cancelled):
        node._count_with_runui_panel(6, 0.0, ctx)
    assert ctx.cancelled()
    # the overlay was still hidden on this exit path (overlay_shown's own finally)
    assert ctx.overlay_visible_calls == [True, False]
    stopped = ctx.sent[-1]["panel"]
    assert stopped["title"] == "Counting: stopped"
    assert stopped["message"].startswith("Stopped. The arm is holding still where")
    assert stopped["buttons"] == [panels.DONE]


# -- execute() dispatches to count_with_runui_panel -----------------------------------------------


def test_execute_dispatches_to_count_with_runui_panel():
    replies = [[button("start")], [button("start_at_1")], [button("done")]]
    node, ctx = make(replies=replies)
    node.mock = True  # skip execute()'s `if not self.mock: self.arm.get_state()` branch
    result = node.execute("count_with_runui_panel", {"to": 2, "interval_s": 0.0}, ctx)
    assert result == {"counted_to": 2, "started_at": 1}
