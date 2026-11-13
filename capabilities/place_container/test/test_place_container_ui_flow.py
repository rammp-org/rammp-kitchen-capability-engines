"""Offline UI flow tests for place_here's tap-to-place: the node's production methods run without ROS or hardware.

The methods are loaded through `ast` (node.py imports ROS at module level) with the shared capability_ui.RunUI, the
real spots/ui modules, and a fake ctx that behaves as the engine's: UI events delivered one at a time, a cancel flag
that checkpoint() raises on, update_user_panel() False with no UI session. cancel_own_run is stubbed to set that
flag, as the engine's own cancel does. The place itself (_act) is a stub: these test the flow, not the motion."""

import ast
import collections
import contextlib
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
NODE = ROOT / "capabilities/place_container/place_container/node.py"

sys.path.insert(0, str(ROOT / "shared/capability_ui"))

from capability_ui.panels import STOP, STOPPED_MESSAGE, RunUI  # noqa: E402

from place_container import spots, ui  # noqa: E402

METHODS = {
    "_place_here",
    "_place_here_run",
    "_pick_spot",
    "_find_spots",
    "_hand_exclusion",
    "_table_z",
    "_marked_spot",
}


def engine_classes():
    tree = ast.parse((ROOT / "capability_engine/capability_engine/engine.py").read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ("EngineError", "Cancelled")]
    scope = {}
    exec(compile(ast.Module(body=classes, type_ignores=[]), "engine.py", "exec"), scope)
    return scope["EngineError"], scope["Cancelled"]


EngineError, Cancelled = engine_classes()


class Refused(Exception):
    pass


PLAN = SimpleNamespace(BOWL_RADIUS=0.055, LIP_WIDTH=0.03175, Refused=Refused)


def load(**scope):
    """_PanelContext and the Capability class with only METHODS, no bases."""
    tree = ast.parse(NODE.read_text())
    body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_PanelContext"]
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Capability")
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
    body.append(cls)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(NODE), "exec"), scope)
    return scope


def tap(instance_id):
    return {"type": "object_tapped", "instance_id": instance_id}


def press(button_id):
    return {"type": "button", "button_id": button_id}


class FakeCtx:
    """The engine's ctx as the UI sees it. `script`: for each panel with a button other than Stop, the events the user
    makes on it (in order); once the script is used up, a panel with Done gets Done."""

    def __init__(self, script=(), session=True):
        self.script, self.session = list(script), session
        self.sent, self.progressed = [], []
        self.events = collections.deque()
        self._lock = threading.Lock()
        self._cancelled = threading.Event()

    def update_user_panel(self, panel_json=None, objects=None, selected_instance_id=None):
        self.sent.append(dict(panel=panel_json, objects=objects, selected=selected_instance_id))
        buttons = [b["button_id"] for b in (panel_json or {}).get("buttons", [])]
        if self.session and any(b != STOP["button_id"] for b in buttons):
            if self.script:
                events = self.script.pop(0)
            else:
                events = [press("done")] if "done" in buttons else []
            with self._lock:
                self.events.extend(events)
        return self.session

    def poll_ui_event(self):
        with self._lock:
            return self.events.popleft() if self.events else None

    def push(self, event):
        with self._lock:
            self.events.append(event)

    def progress(self, phase, pct, message="", pausable=True):
        self.progressed.append(phase)

    def cancel(self):
        self._cancelled.set()

    def cancelled(self):
        return self._cancelled.is_set()

    def checkpoint(self):
        if self.cancelled():
            raise Cancelled()

    def panels(self):
        return [s["panel"] for s in self.sent if s["panel"] is not None]


CFG = {
    "allow_execute": True,
    "spot_step": 0.09,
    "spot_edge_margin": 0.03,
    "spot_clearance": 0.03,
    "reach": [0.70, 1.10],
    "table_z_range": [-0.10, 0.50],
    "max_table_tilt_deg": 3.0,
    "table_check_tol": 0.03,
    "table_z_bias": 0.01,
    "scene_wait_s": 3.0,
    "scene_depth_topic": "/scene_camera/depth/image_rect_raw",
    "scene_info_topic": "/scene_camera/color/camera_info",
    "detections_topic": "/perception/object_detections",
    "max_spot_age_min": 30.0,
}


@pytest.fixture
def make():
    """make(script, mock=True, session=True, marked=None, act=None, camera=None) -> (node, ctx)."""

    def build(script=(), mock=True, session=True, marked=None, act=None, camera=None):
        ctx = FakeCtx(script, session)
        scope = load(
            EngineError=EngineError,
            Cancelled=Cancelled,
            RunUI=RunUI,
            cancel_own_run=lambda engine: engine.test_ctx.cancel(),
            spots=spots,
            ui=ui,
            plan=PLAN,
            time=time,
            rammp_arm=SimpleNamespace(FRAME="arm_mount"),
            SceneCamera=camera or Mock(side_effect=AssertionError("no camera in mock")),
        )
        node = scope["Capability"]()
        node.test_ctx = ctx
        node.mock, node.orchestrated = mock, session
        node.cfg = dict(CFG)
        node.get_logger = Mock(return_value=Mock())
        node._recording = lambda *args: contextlib.nullcontext(None)
        node._act = Mock(side_effect=act) if act else Mock(return_value={"executed": not mock, "mock": mock})
        node._load = Mock(return_value=marked) if marked else Mock(side_effect=Refused("no marked spot"))
        node.arm = Mock()
        node.arm.get_state.return_value = {"ee_pos": [0.2, -0.4, 0.6]}
        return node, ctx

    return build


@pytest.fixture(scope="module")
def mock_spots():
    found = spots.find_spots(spots.synthetic_scene(), footprint_r=PLAN.BOWL_RADIUS + PLAN.LIP_WIDTH)
    return [o["instance_id"] for o in spots.ui_objects(found.spots, spots.synthetic_scene())], found


def picker_panels(ctx):
    pickers = ("place", "look")
    return [s for s in ctx.sent if s["panel"] and any(b["button_id"] in pickers for b in s["panel"]["buttons"])]


def test_tap_then_place_here_places_at_that_spot(make, mock_spots):
    ids, found = mock_spots
    spot = found.by_id()[ids[3]]
    node, ctx = make([[tap(ids[3]), press("place")]])
    result = node._place_here({}, ctx)
    action, params, _ = node._act.call_args.args
    assert action == "place_here"
    assert params["x"] == spot["x"] and params["y"] == spot["y"]
    assert params["table_z"] == pytest.approx(0.20 + CFG["table_z_bias"], abs=0.003)  # biased up
    assert result["spot"] == params
    picker = picker_panels(ctx)[0]
    assert picker["panel"]["message"].startswith(ui.PICK)
    assert [b["label"] for b in picker["panel"]["buttons"]] == ["Place here", "Stop"]  # no marked spot saved
    assert {o["source"] for o in picker["objects"]} == {"scene"} and len(picker["objects"]) == len(ids)
    assert any(s["selected"] == ids[3] and s["panel"] is None for s in ctx.sent)  # the highlight follows the tap
    assert "Mock:" in ctx.panels()[-1]["message"] and ui.PLACED in ctx.panels()[-1]["message"]


def test_place_here_without_a_tap_asks_again(make, mock_spots):
    ids, _ = mock_spots
    node, ctx = make([[press("place")], [tap("not_a_spot"), tap(ids[0]), press("place")]])
    node._place_here({}, ctx)
    pickers = picker_panels(ctx)
    assert len(pickers) == 2 and pickers[1]["panel"]["message"].startswith(ui.TAP_FIRST)
    assert node._act.call_count == 1
    assert not any(s["selected"] == "not_a_spot" for s in ctx.sent)  # a tap on something else is not a choice


def test_the_marked_spot_is_offered_when_it_is_valid(make, mock_spots):
    marked = {"x": 0.9, "y": -0.2, "table_z": 0.2, "time": time.time() - 60}
    node, ctx = make([[press("marked")]], marked=marked)
    node._place_here({}, ctx)
    assert [b["label"] for b in picker_panels(ctx)[0]["panel"]["buttons"]] == ["Place here", "Use marked spot", "Stop"]
    assert node._act.call_args.args[1] == {}  # {} = the marked spot (_place loads and checks it again)
    old = dict(marked, time=time.time() - 3600)
    node, ctx = make([[tap(mock_spots[0][0]), press("place")]], marked=old)
    node._place_here({}, ctx)
    assert "Use marked spot" not in [b["label"] for b in picker_panels(ctx)[0]["panel"]["buttons"]]


def test_a_camera_that_disagrees_with_the_arm_offers_no_spots(make):
    marked = {"x": 0.9, "y": -0.2, "table_z": 0.10, "time": time.time() - 60}  # the arm felt the table 10 cm lower
    node, ctx = make([[press("marked")]], marked=marked)
    node._place_here({}, ctx)
    picker = picker_panels(ctx)[0]
    assert picker["objects"] == [] and "calibration" in picker["panel"]["message"]
    assert [b["label"] for b in picker["panel"]["buttons"]] == ["Look again", "Use marked spot", "Stop"]


def test_execution_disabled_is_refused_before_anything(make):
    node, ctx = make(mock=False)
    node.cfg["allow_execute"] = False
    with pytest.raises(EngineError, match="allow_execute"):
        node._place_here({}, ctx)
    node._act.assert_not_called()
    assert picker_panels(ctx) == []
    assert ctx.panels()[-1]["title"].endswith("not started") and ctx.panels()[-1]["buttons"][0]["label"] == "Done"


def test_the_place_shows_its_phases_with_stop(make, mock_spots):
    ids, _ = mock_spots

    def act(action, params, pctx):
        pctx.progress("carry", 0.0)
        pctx.progress("lower", 0.0)
        return {"executed": False, "mock": True}

    node, ctx = make([[tap(ids[0]), press("place")]], act=act)
    node._place_here({}, ctx)
    moving = [p for p in ctx.panels() if p.get("progress")]
    assert [p["progress"]["phase"] for p in moving] == ["carry", "lower"]
    assert all(p["buttons"] == [STOP] for p in moving)
    assert moving[1]["message"] == ui.PHASES["lower"][0]
    assert ctx.progressed == ["carry", "lower"]  # the orchestrator's feedback too


def test_stop_during_the_place_cancels_and_says_where_the_container_is(make, mock_spots):
    ids, _ = mock_spots

    def act(action, params, pctx):
        pctx.progress("lower", 0.0)
        pctx.push(press("stop"))  # the user presses Stop while the arm moves
        deadline = time.monotonic() + 5
        while not pctx.cancelled() and time.monotonic() < deadline:
            time.sleep(0.01)
        pctx.checkpoint()  # raises Cancelled, as self.arm does
        raise AssertionError("not cancelled")

    node, ctx = make([[tap(ids[0]), press("place")]], act=act)
    with pytest.raises(Cancelled):
        node._place_here({}, ctx)
    last = ctx.panels()[-1]
    assert last["title"].endswith("stopped") and last["message"].startswith(STOPPED_MESSAGE)
    assert "may still be in the hand" in last["message"]
    assert node._act.call_count == 1


def test_a_refusal_before_moving_goes_back_to_the_picker(make, mock_spots):
    ids, _ = mock_spots
    outcomes = [EngineError("refused", "adjust: no arm pose puts the tool there", True), {"executed": True}]

    def act(action, params, pctx):
        out = outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    node, ctx = make([[tap(ids[0]), press("place")], [tap(ids[1]), press("place")]], act=act)
    node._place_here({}, ctx)
    pickers = picker_panels(ctx)
    assert len(pickers) == 2 and "Couldn't place it there: adjust: no arm pose" in pickers[1]["panel"]["message"]
    assert node._act.call_count == 2


def test_a_failure_after_the_arm_moved_is_shown_and_ends_it(make, mock_spots):
    ids, _ = mock_spots
    error = EngineError("refused", "STOPPED: the hand stopped 4 cm above the table -- NOT letting go", True)
    node, ctx = make([[tap(ids[0]), press("place")]], act=lambda *a: (_ for _ in ()).throw(error))
    with pytest.raises(EngineError, match="STOPPED"):
        node._place_here({}, ctx)
    last = ctx.panels()[-1]
    assert last["title"].endswith("failed") and "NOT letting go" in last["message"]
    assert len(picker_panels(ctx)) == 1


def test_standalone_mock_places_at_the_first_free_spot(make):
    node, ctx = make(session=False)
    node._place_here({}, ctx)
    params = node._act.call_args.args[1]
    assert set(params) == {"x", "y", "table_z"}


def test_goal_params_skip_the_picker(make):
    node, ctx = make()
    node._place_here({"x": 0.9, "y": -0.1, "table_z": 0.2}, ctx)
    assert node._act.call_args.args[1] == {"x": 0.9, "y": -0.1, "table_z": 0.2}
    assert picker_panels(ctx) == []


def test_on_the_robot_the_scene_camera_is_read_and_a_tilted_view_refused(make):
    scene = spots.synthetic_scene()
    grabbed = Mock()
    grabbed.grab.return_value = scene
    camera = Mock(return_value=grabbed)
    node, ctx = make([[tap(picker_ids_of(scene)[0]), press("place")]], mock=False, camera=camera)
    node._place_here({}, ctx)
    assert camera.call_args.args[1:] == (
        "arm_mount",
        CFG["scene_depth_topic"],
        CFG["scene_info_topic"],
        CFG["detections_topic"],
    )
    assert node._act.call_args.args[1]["table_z"] == pytest.approx(0.21, abs=0.003)
    # the same frame read with a mount 10 deg off: no spots, the reason on the panel
    rot = np.eye(4)
    c, s = np.cos(np.radians(10)), np.sin(np.radians(10))
    rot[1:3, 1:3] = [[c, -s], [s, c]]
    grabbed.grab.return_value = spots.Scene(scene.depth_m, scene.K, scene.T @ rot, scene.labels, scene.objects)
    node, ctx = make([[press("stop")]], mock=False, camera=camera)
    with pytest.raises(Cancelled):
        node._place_here({}, ctx)
    picker = picker_panels(ctx)[0]
    assert picker["objects"] == [] and "tilted" in picker["panel"]["message"]


def test_a_camera_that_isnt_ready_says_so(make):
    grabbed = Mock()
    grabbed.grab.side_effect = RuntimeError("no depth or camera_info from the scene camera")
    node, ctx = make([[press("look")], [press("stop")]], mock=False, camera=Mock(return_value=grabbed))
    with pytest.raises(Cancelled):
        node._place_here({}, ctx)
    pickers = picker_panels(ctx)
    assert len(pickers) == 2 and "isn't ready" in pickers[0]["panel"]["message"]
    assert grabbed.grab.call_count == 2  # Look again looked again
    node._act.assert_not_called()


def picker_ids_of(scene):
    found = spots.find_spots(scene, footprint_r=PLAN.BOWL_RADIUS + PLAN.LIP_WIDTH)
    return [o["instance_id"] for o in spots.ui_objects(found.spots, scene)]


@pytest.fixture(autouse=True)
def _no_ui_wrap(monkeypatch):
    """Messages are compared whole here; capability_ui's own tests cover the UI's line wrapping."""
    from capability_ui import panels as ui_panels

    monkeypatch.setattr(ui_panels, "WRAP_CHARS", 10**6)
