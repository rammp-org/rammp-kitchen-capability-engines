"""The run's side of the line-up (press_demo --line-up): the wrist turned to look, gently, every frame judged
as the user drives up (survey.watch_front), and the engine's go or stop (press_demo.line_up_watch)."""

import io
import json
import time
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import FakeClient

from box_opening.constants import HOME
from box_opening.tasks import press_demo
from box_opening.tasks import survey as sv


def _view(name="survey:angle-1"):
    return sv.SurveyView(
        name, [0.1] * 7, (0.2, 0.2, 0.7), np.eye(3), np.ones(1, bool), np.ones(1, bool), 35.0, what=name
    )


def _fix(xyz=(0.65, -0.1, 0.315), table_z=0.216, partial=""):
    return SimpleNamespace(top_xyz=xyz, table_z_scene=table_z, partial=partial, rough=False)


def _said_fix(x):
    return {"fix": {"xyz": [x, 0.0, 0.315], "table_z": 0.216, "age_s": 0.0, "rough": False, "by": "wrist"}}


class Runner:
    def __init__(self):
        self.notes, self.stills, self.gripper = [], [], []

    def note(self, leg, **fields):
        self.notes.append((leg, fields))

    def wait_still(self, tag):
        self.stills.append(tag)

    def start_gripper(self, name, cmd, execute):
        self.gripper.append((name, cmd, execute))
        return True


@pytest.fixture
def front(monkeypatch):
    """The wrist look's parts, stood in for: a view to fly, frames that come at once, a finder scripted per
    frame (`finds`: a fix, or a reason), the body model on. Returns what was flown, said and asked."""
    from box_opening.perception import wrist_finder as wf
    from box_opening.runtime import approach

    w = SimpleNamespace(flown=[], said=[], finds=[], asked=[], also=0, fly=("flown", None), views=[_view()])
    monkeypatch.setattr(approach, "body_check", lambda: object())
    monkeypatch.setattr(approach, "_chain", lambda: object())
    monkeypatch.setattr(sv, "camera_of", lambda watcher, name=None: object())
    monkeypatch.setattr(sv, "FrontBand", lambda check, rest: "band")
    monkeypatch.setattr(sv, "angled_views", lambda *a, **k: (list(w.views), "no view" if not w.views else None, {}))
    monkeypatch.setattr(sv, "fly_view", lambda ctx, runner, view, band, execute: w.flown.append(view) or w.fly)

    def grab(node, watcher, runner, tag, n=1, timeout_s=1.0):
        w.asked.append((runner, n))
        return SimpleNamespace(tag=tag), SimpleNamespace(color_stamp=None), None, None

    monkeypatch.setattr(sv, "grab_frame", grab)

    def find(frames, model, check_buttons=True, reachable=None, seen_at=None, far_nearest=False):
        # each frame alone; a top out of the arm's reach no rival of one within it, and where none is within it
        # the nearest alone (the run's own word on reach: survey._within_reach)
        assert len(frames) == 1 and check_buttons is False and reachable is w.within and far_nearest is True
        got = w.finds.pop(0) if w.finds else "no lid-sized top"
        far = bool(getattr(got, "far", False))
        return SimpleNamespace(
            fix=None if isinstance(got, str) else got, why=got if isinstance(got, str) else "found", far=far
        )

    monkeypatch.setattr(wf, "find_container", find)
    w.within = lambda top: True
    monkeypatch.setattr(sv, "_within_reach", lambda model: w.within)
    return w


def _watch(w, n_frames, execute=True):
    ctx = SimpleNamespace(client=FakeClient(), model=object())
    runner = Runner()

    def also():
        w.also += 1

    state = sv.watch_front(
        None,
        ctx,
        runner,
        object(),
        execute,
        w.said.append,
        lambda: len(w.asked) >= n_frames,
        rest=list(HOME),
        also=also,
    )
    return state, runner


# -- the wrist's watch -----------------------------------------------------------------------------
def test_the_wrist_turns_once_to_the_view_that_shows_the_most_and_judges_every_frame(front):
    front.finds = [_fix((0.95, 0.0, 0.315)), _fix((0.90, 0.0, 0.315)), "no lid-sized top", _fix((0.65, 0.0, 0.315))]
    state, runner = _watch(front, 4)
    assert state == ("watched", None)
    (view,) = front.flown
    assert view.name == sv.LINE_UP_VIEW and view.joints == [0.1] * 7  # angled_views' first, flown once
    assert runner.stills == [sv.LINE_UP_VIEW] and all(r is None and n == 1 for r, n in front.asked)
    fixes = [m["fix"] for m in front.said if "fix" in m]
    assert [f["xyz"][0] for f in fixes] == [0.95, 0.9, 0.65]
    assert all(f["by"] == "wrist" and f["rough"] is False and f["table_z"] == 0.216 for f in fixes)
    assert [m for m in front.said if "none" in m] == [{"none": "wrist camera: no lid-sized top"}]
    assert front.also == 1  # the scene camera's own look, on the frame that showed no box
    assert runner.notes[-1] == ("line_up", {"stage": "watched", "frames": 4, "found": 3})


def test_a_lid_the_view_cuts_is_said_roughly_and_no_box_no_oftener_than_its_period(front, monkeypatch):
    front.finds = [_fix(partial="cut off by the frame")]
    _watch(front, 1)
    assert front.said[0]["fix"]["rough"] is True
    front.said.clear()
    front.asked.clear()
    _watch(front, 30)  # thirty frames in a moment, none showing a box
    assert len([m for m in front.said if "none" in m]) == 1 and front.also == 30


def test_a_container_out_of_the_arms_reach_is_said_roughly(front):
    """A far find (none within the arm's reach, this one nearest the arm alone: wrist_finder's far_nearest) says
    which way to move and is never asked on."""
    far = _fix((1.05, 0.0, 0.315))
    far.far = True
    front.finds = [far, _fix((0.65, 0.0, 0.315))]
    _watch(front, 2)
    assert [(m["fix"]["xyz"][0], m["fix"]["rough"]) for m in front.said] == [(1.05, True), (0.65, False)]


def test_a_second_lid_sized_top_beyond_the_arms_reach_does_not_blind_the_watch(monkeypatch):
    """The real finder on a frame of two lid-sized tops (2026-11-12's table: a second one 1.08 m out). The watch
    as first written judged each frame with no word on reach: "two places pass", and never a fix. One within
    reach is the container; none within reach, the one nearest the arm alone, roughly."""
    from test_wrist_finder import COUNTER_Z, LID_H, PINK, canister, render

    from box_opening.models.container import ContainerModel
    from box_opening.perception import wrist_finder as wf
    from box_opening.runtime import approach

    model = ContainerModel.load(PINK)
    frame = render(
        (0.41, 0.30, COUNTER_Z + LID_H + 0.40),
        (0.41 + 1e-4, 0.30, COUNTER_Z),
        canister(0.30, 0.30) + canister(0.52, 0.30),
    )
    assert "two places pass" in wf.find_container([frame], model, check_buttons=False).why
    grabbed = []
    monkeypatch.setattr(
        sv, "grab_frame", lambda *a, **k: grabbed.append(1) or (frame, SimpleNamespace(color_stamp=None), None, None)
    )
    monkeypatch.setattr(sv, "fly_view", lambda *a, **k: ("flown", None))
    monkeypatch.setattr(sv, "angled_views", lambda *a, **k: ([_view()], None, {}))
    monkeypatch.setattr(sv, "camera_of", lambda watcher, name=None: object())
    monkeypatch.setattr(sv, "FrontBand", lambda check, rest: "band")
    monkeypatch.setattr(approach, "body_check", lambda: object())
    monkeypatch.setattr(approach, "_chain", lambda: object())
    for within, rough in ((lambda top: top[0] < 0.40, False), (lambda top: False, True)):
        monkeypatch.setattr(sv, "_within_reach", lambda model, w=within: w)
        said = []
        grabbed.clear()
        ctx = SimpleNamespace(client=FakeClient(), model=model)
        sv.watch_front(None, ctx, Runner(), object(), True, said.append, lambda: len(grabbed) >= 1)
        (line,) = (m for m in said if "fix" in m or "none" in m)
        assert "fix" in line, line
        assert line["fix"]["xyz"][0] == pytest.approx(0.30, abs=0.005) and line["fix"]["rough"] is rough


@pytest.mark.parametrize(
    "fly, state", [(("refused", "its way there: the knees"), "refused"), (("stopped", "why"), "stopped")]
)
def test_a_turn_that_is_not_flown_watches_nothing(front, fly, state):
    front.fly = fly
    got, runner = _watch(front, 3)
    assert got == (state, fly[1]) and front.asked == [] and front.said == []
    assert runner.notes[-1][1]["flown"] is False


def test_a_fix_the_caller_judges_out_is_said_as_none_with_its_reason(front):
    """`judge` (the line-up's word on the orchestrator's selection): a find it refuses is no fix line, its own
    reason is said in the fix's place, no oftener than any other "none", and the scene camera looks."""
    refused = "the selected container x is in none of the scene camera's detections: the run would not look for it"
    front.finds = [_fix((0.65, 0.0, 0.315)), _fix((0.66, 0.0, 0.315)), _fix((0.70, 0.0, 0.315))]
    ctx = SimpleNamespace(client=FakeClient(), model=object())
    judged = []

    def judge(fix):
        judged.append(fix.top_xyz[0])
        return refused if fix.top_xyz[0] < 0.69 else None

    def also():
        front.also += 1

    state = sv.watch_front(
        None, ctx, Runner(), object(), True, front.said.append, lambda: len(front.asked) >= 3, also=also, judge=judge
    )
    assert state == ("watched", None) and judged == [0.65, 0.66, 0.70]
    assert front.said == [{"none": refused}, _said_fix(0.70)] and front.also == 2


def test_a_stop_heard_before_the_turn_turns_nothing(front):
    """The engine's stop (or its go) may come while the views are being chosen: the wrist then turned out to
    the view only to be turned straight back."""
    state, runner = _watch(front, 0)  # stopped from the start
    assert state == ("none", "stopped before the turn") and front.flown == [] and front.said == []
    assert runner.stills == [] and front.asked == []


def test_with_no_view_or_no_body_model_the_wrist_does_not_turn(front, monkeypatch):
    from box_opening.runtime import approach

    front.views = []
    assert _watch(front, 3)[0] == ("none", "no view") and front.flown == []
    monkeypatch.setattr(approach, "body_check", lambda: None)
    state, why = _watch(front, 3)[0]
    assert state == "none" and "body model" in why and front.flown == []


def test_a_preview_turns_nothing_and_watches_nothing(front):
    assert _watch(front, 3, execute=False)[0][0] == "none" and front.asked == []


def test_the_wrist_goes_back_to_rest_as_a_view_is_flown(front, monkeypatch):
    ctx, runner = SimpleNamespace(client=FakeClient()), Runner()
    assert sv.rest_wrist(ctx, runner, True, rest=list(HOME)) == ("flown", None)
    (back,) = front.flown
    assert back.name == sv.LINE_UP_REST and back.joints == list(HOME)  # fly_view flies it as a look


# -- the engine's go or stop -------------------------------------------------------------------------
class Ear:
    """The engine's answer, scripted: go or stop after `n` asks."""

    def __init__(self, n, go):
        self.n, self.go, self.stop, self.asks = n, False, False, 0
        self._go = go

    def heard(self):
        self.asks += 1
        if self.asks > self.n:
            self.go, self.stop = self._go, not self._go
        return self.go or self.stop


class Scene:
    """SceneLocator's part: a look tells one sighting where it has one."""

    def __init__(self, tops=()):
        self.tops, self.on_sighting, self.last_why, self.looks = list(tops), None, "scene segmenter: saw none", 0

    def locate(self, timeout_s):
        self.looks += 1
        if self.tops:
            top = self.tops.pop(0)
            self.on_sighting(SimpleNamespace(top=top, table_z=0.216, lift="side", score=0.9), 0.1)
        return None


@pytest.fixture
def lined(monkeypatch, tmp_path):
    """line_up_watch's surroundings: an arm at rest with closed fingers, a wrist watch that says one fix a
    time it is asked whether to stop, and what happened."""
    w = SimpleNamespace(watched=[], rested=[], watch=("watched", None), urdf=tmp_path / "arm.urdf")
    w.urdf.write_text("<robot/>")
    monkeypatch.setattr(press_demo.cli_common, "arm_urdf_file", lambda: w.urdf)

    def watch_front(
        node, ctx, runner, watcher, execute, say, stop, rest=None, camera_cfg_name=None, also=None, judge=None
    ):
        w.watched.append(rest)
        if w.watch[0] != "watched":
            return w.watch
        while not stop():
            why = None if judge is None else judge(_fix((0.65, 0.0, 0.315)))
            if why is not None:
                say({"none": why})
            else:
                say(
                    {"fix": {"xyz": [0.65, 0.0, 0.315], "table_z": 0.216, "age_s": 0.05, "rough": False, "by": "wrist"}}
                )
            also()
        return w.watch

    monkeypatch.setattr(sv, "watch_front", watch_front)
    monkeypatch.setattr(
        sv, "rest_wrist", lambda ctx, runner, execute, rest=None: w.rested.append(rest) or ("flown", None)
    )
    client = FakeClient()
    client.live = list(HOME)
    client.gripper_cmd = lambda position: (True, 0.79, 0.0)  # shut already
    w.ctx = SimpleNamespace(client=client, worlds=SimpleNamespace(base_frame="arm_mount", wrist_camera_config=None))
    w.cfg = SimpleNamespace(park_tool_down=False)
    monkeypatch.setattr(press_demo.fingers, "already_closed", lambda pos: pos is not None and pos > 0.7)
    return w


def _line_up(w, ear, wrist_first=True, execute=True, locator=None, instance_id=None, wrist_alone=False):
    out, runner = io.StringIO(), Runner()
    args = Namespace(wrist_first=wrist_first, execute=execute, instance_id=instance_id, wrist_alone=wrist_alone)
    got = press_demo.line_up_watch(None, w.ctx, w.cfg, runner, object(), args, locator, 0.05, ear=ear, out=out)
    lines = [json.loads(ln[3:]) for ln in out.getvalue().splitlines() if ln.startswith("@@ ")]
    return got, lines, runner


def test_at_rest_the_wrist_looks_and_a_go_lets_the_run_go_on_from_there(lined):
    (went, turned), lines, runner = _line_up(lined, Ear(3, go=True))
    assert (went, turned) == (True, True) and lined.watched == [list(HOME)] and lined.rested == []
    assert lines[0] == {"robot": {"urdf": str(lined.urdf), "root": "arm_mount"}}
    assert [ln["fix"]["by"] for ln in lines[1:]] == ["wrist"] * 3
    assert runner.gripper == []  # shut already: nothing sent
    assert runner.notes[-1] == ("line_up", {"stage": "go", "wrist": True})


def test_a_stop_turns_the_wrist_back_and_nothing_is_pressed(lined):
    (went, turned), _lines, runner = _line_up(lined, Ear(2, go=False))
    assert (went, turned) == (False, False) and lined.rested == [list(HOME)]
    assert runner.notes[-1] == ("line_up", {"stage": "stop", "wrist": True})


def test_open_fingers_close_first_as_at_every_runs_start(lined):
    lined.ctx.client.gripper_cmd = lambda position: (True, 0.0, 0.0)
    _got, _lines, runner = _line_up(lined, Ear(1, go=True))
    assert [g[0] for g in runner.gripper] == ["press:close"] and lined.watched


def test_an_arm_that_is_not_at_rest_is_not_moved_before_the_go(lined):
    """The wrist's turn is the one motion a line-up makes of its own: no home first, no wrist, the scene
    camera alone; and after the go the run homes as every run does (wrist_turned False)."""
    lined.ctx.client.live = [v + 0.3 for v in HOME]
    scene = Scene(tops=[(0.9, 0.0, 0.315)])
    (went, turned), lines, runner = _line_up(lined, Ear(3, go=True), locator=scene)
    assert (went, turned) == (True, False) and lined.watched == [] and runner.gripper == []
    assert lines[1] == {"fix": {"xyz": [0.9, 0.0, 0.315], "table_z": 0.216, "age_s": 0.1, "rough": False, "score": 0.9}}
    assert {"none": "scene segmenter: saw none"} in lines and scene.on_sighting is None  # unhooked as it ends
    # ... and the engine is told why the wrist does not look, as the last word on why there is no container
    nones = [ln for ln in lines if "none" in ln]
    assert nones[-1] == {"none": "the arm is not at its rest pose: home it, then press open again"}


def test_a_wrist_left_turned_looks_on_from_where_it_stands(lined):
    """After a Cancel while lining up, or a run that refused after Ready, the arm stands at rest but for its
    wrist, turned where it looked. The line-up never homes before the go, so as first written every later
    line-up lost its wrist look (the scene camera alone, which does not see a lid whole in front of the arm)
    until someone homed the arm another way. The wrist's own turn is what stands off rest: it looks on from
    there, goes on from there at the go, and turns back to rest at a stop."""
    turned_at = [v + (0.4 if j in sv.FRONT_JOINTS else 0.03) for j, v in enumerate(HOME)]
    lined.ctx.client.live = list(turned_at)
    (went, turned), lines, _runner = _line_up(lined, Ear(3, go=True))
    assert (went, turned) == (True, True) and lined.watched == [list(HOME)] and lined.rested == []
    assert [ln["fix"]["by"] for ln in lines[1:]] == ["wrist"] * 3 and not any("none" in ln for ln in lines)
    lined.watched.clear()
    (went, turned), _lines, _runner = _line_up(lined, Ear(2, go=False))
    assert (went, turned) == (False, False) and lined.watched == [list(HOME)] and lined.rested == [list(HOME)]


@pytest.mark.parametrize("joint, off", [(5, sv.FRONT_MAX_RAD + 0.1), (1, 0.1), (3, -0.1)])
def test_an_arm_off_rest_by_more_than_its_wrists_own_turn_is_not_looked_from(lined, joint, off):
    """A wrist joint further from rest than the front look ever turns it, or any joint the look does not turn off
    rest: not a line-up's leftover. Nothing turns, and the engine is told to have the arm homed."""
    lined.ctx.client.live = [v + (off if j == joint else 0.0) for j, v in enumerate(HOME)]
    (went, turned), lines, runner = _line_up(lined, Ear(3, go=False), locator=Scene())
    assert (went, turned) == (False, False) and lined.watched == [] and lined.rested == [] and runner.gripper == []
    assert lines[-1] == {"none": "the arm is not at its rest pose: home it, then press open again"}


def test_without_wrist_first_or_execute_only_the_scene_camera_looks(lined):
    for kw in ({"wrist_first": False}, {"execute": False}):
        lined.watched.clear()
        scene = Scene()
        (went, turned), _lines, _runner = _line_up(lined, Ear(2, go=False), locator=scene, **kw)
        assert (went, turned) == (False, False) and lined.watched == [] and lined.rested == [] and scene.looks >= 2


def test_a_wrist_turn_stopped_part_way_ends_the_line_up_the_arm_holding(lined):
    """The arm holds where the turn stopped, the wrist part turned: that is not "nothing moved, press open again"
    (EXIT_NOT_LINED_UP, the wrist back at rest), and the run's process ends with the exit that says so."""
    from box_opening.runtime.abort import EXIT_LINE_UP_HOLDS

    lined.watch = ("stopped", "the planner refused")
    scene = Scene()
    with pytest.raises(SystemExit) as e:
        _line_up(lined, Ear(5, go=True), locator=scene)
    assert e.value.code == EXIT_LINE_UP_HOLDS == 6
    assert lined.rested == [] and scene.on_sighting is None  # no further motion; unhooked as it ends


def test_a_wrist_that_does_not_turn_back_after_a_stop_ends_the_line_up_the_arm_holding(lined, monkeypatch):
    from box_opening.runtime.abort import EXIT_LINE_UP_HOLDS

    monkeypatch.setattr(sv, "rest_wrist", lambda ctx, runner, execute, rest=None: ("stopped", "failed (why)"))
    with pytest.raises(SystemExit) as e:
        _line_up(lined, Ear(2, go=False))
    assert e.value.code == EXIT_LINE_UP_HOLDS


def test_a_wrist_that_cannot_look_leaves_the_scene_camera_and_says_why(lined):
    lined.watch = ("refused", "its way there: the knees")
    scene = Scene()
    (went, turned), lines, _runner = _line_up(lined, Ear(3, go=True), locator=scene)
    assert (went, turned) == (True, False) and {"none": "wrist camera: its way there: the knees"} in lines
    assert scene.looks >= 1


def test_a_ctrl_c_asked_for_ends_the_scene_cameras_watch(lined):
    """The scene-only watch only sleeps and reads: a Ctrl+C that was asked for and did not arrive as an exception
    (it surfaced as another error under the wrist turn's planning, taken there for a refusal) is seen in the
    client's abort flag, and ends the line-up as a Ctrl+C does."""
    lined.watch = ("refused", "planning failed")
    lined.ctx.client._abort = SimpleNamespace(requested=True)
    scene = Scene()
    with pytest.raises(KeyboardInterrupt):
        _line_up(lined, Ear(50, go=True), locator=scene)
    assert scene.looks == 0 and scene.on_sighting is None


def _selecting(mine=None, held=None):
    """A scene locator that looks (Scene) and holds the orchestrator's selection as test_survey's does (_locator:
    `mine` the selected instance's box in the segmenter's newest frame, `held` how many frames of its last look
    held the selection)."""
    from test_survey import _locator

    scene = Scene()
    for name, value in vars(_locator(mine=mine, held=held)).items():
        if name != "last_why":
            setattr(scene, name, value)
    return scene


def test_with_a_selection_a_wrist_fix_is_said_only_where_the_run_would_look_for_that_container(lined):
    """--instance-id: after Open the run turns its wrist only where something ties what it finds to the selected
    container (wrist_look: the scene camera's box of it, or --wrist-alone for a selection the scene camera never
    held). The line-up said every lid its wrist found, so the panel reached "Ready" for a run that then stood,
    said "not looked for" and went home. It asks the run's own rule, and says why instead of the fix."""
    from test_survey import SELECTED

    lined.ctx.model = SimpleNamespace(lid_dims=(0.105, 0.105, 0.012))
    why = "the selected container %s is in none of the scene camera's detections: the run would not look for it"
    box = (380.0, 150.0, 440.0, 230.0)
    for kw, said in (
        ({"locator": _selecting(mine=None, held=0)}, False),  # the selection in none of its frames
        ({"locator": _selecting(mine=box, held=9)}, True),  # its box in the scene image ties a find to it
        ({"locator": _selecting(mine=None, held=0), "wrist_alone": True}, True),  # --wrist-alone: the one it sees
        ({"locator": _selecting(mine=None, held=None), "wrist_alone": True}, False),  # ... only once a look says so
        ({"locator": None}, False),  # no scene camera: nothing can tie it
    ):
        _got, lines, _runner = _line_up(lined, Ear(3, go=False), instance_id=SELECTED, **kw)
        fixes = [ln for ln in lines if ln.get("fix", {}).get("by") == "wrist"]
        assert bool(fixes) is said, kw
        assert ({"none": why % SELECTED} in lines) is not said, kw
    # no selection: every fix is said, as before
    _got, lines, _runner = _line_up(lined, Ear(3, go=False), locator=_selecting(mine=None, held=0))
    assert len([ln for ln in lines if "fix" in ln]) == 3


def test_the_ear_hears_go_or_stop_and_a_closed_pipe_is_a_stop():
    for text, go, stop in (
        ('{"go": true}\n', True, False),
        ('{"stop": true}\n', False, True),
        ("not json\n", False, True),  # then the pipe's end
        ("", False, True),  # the engine is gone
        ('{"go": "yes"}\n', False, True),  # not a go
    ):
        ear = press_demo.LineUpEar(io.StringIO(text))
        t0 = time.monotonic()
        while not ear.heard() and time.monotonic() - t0 < 2.0:
            time.sleep(0.01)
        assert (ear.go, ear.stop) == (go, stop), text
