"""The run's side of the line-up (press_demo --line-up): the wrist turned to look, gently, every frame judged
as the user drives up (survey.watch_front), and the engine's go or stop (press_demo.line_up_watch)."""

import io
import json
import math
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


def test_the_wrist_says_once_where_it_stands_at_the_line_ups_view(front):
    """{"view": {"joints": ...}}: the joints the arm stands at once its wrist has turned (as it stands, not as the
    view was asked for), said once before any frame is judged: where the run after the go starts from, for the
    engine's proof of the reach from there."""
    front.finds = [_fix(), "no lid-sized top"]
    _watch(front, 2)
    assert front.said[0] == {"view": {"joints": [0.0] * 7}}  # the fake arm stands where it stood (fly_view's stub)
    assert len([m for m in front.said if "view" in m]) == 1 and "fix" in front.said[1]
    for fly in (("refused", "why"), ("stopped", "why")):  # a turn that was not flown: the wrist is not at the view
        front.said.clear(), front.asked.clear()
        front.fly = fly
        _watch(front, 2)
        assert front.said == []
    front.fly = ("flown", None)
    _watch(front, 2, execute=False)  # nor in a preview, where nothing moved
    assert front.said == []


def test_a_lid_the_view_cuts_is_said_roughly_and_no_box_no_oftener_than_its_period(front, monkeypatch):
    front.finds = [_fix(partial="cut off by the frame")]
    _watch(front, 1)
    assert front.said[1]["fix"]["rough"] is True  # after its word on where the wrist stands
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
    assert [(m["fix"]["xyz"][0], m["fix"]["rough"]) for m in front.said[1:]] == [(1.05, True), (0.65, False)]


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

    def judge(fix, find):
        judged.append(fix.top_xyz[0])
        return refused if fix.top_xyz[0] < 0.69 else None

    def also():
        front.also += 1

    state = sv.watch_front(
        None, ctx, Runner(), object(), True, front.said.append, lambda: len(front.asked) >= 3, also=also, judge=judge
    )
    assert state == ("watched", None) and judged == [0.65, 0.66, 0.70]
    assert front.said[1:] == [{"none": refused}, _said_fix(0.70)] and front.also == 2


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
    """SceneLocator's part: a look tells one sighting where it has one. Its record of each look: `frames`,
    how many of the scene camera's frames the segmenter gave it (None: the record says nothing of the
    segmenter), and `state`, the segmenter's own at the look's end (SegmenterRung.state)."""

    def __init__(self, tops=(), frames=1, why="scene segmenter: saw none", state="live"):
        self.tops, self.on_sighting, self.last_why, self.looks = list(tops), None, why, 0
        self.frames, self.state = frames, state

    def locate(self, timeout_s):
        self.looks += 1
        if self.tops:
            top = self.tops.pop(0)
            self.on_sighting(SimpleNamespace(top=top, table_z=0.216, lift="side", score=0.9), 0.1)
        return None

    def look_record(self):
        if self.frames is None:
            return {"timing": {}}
        return {"timing": {}, "segmenter": {"frames": self.frames, "state": self.state}}


@pytest.fixture
def lined(monkeypatch, tmp_path):
    """line_up_watch's surroundings: an arm at rest with closed fingers, a wrist watch that says one fix a
    time it is asked whether to stop, and what happened."""
    w = SimpleNamespace(watched=[], rested=[], watch=("watched", None), urdf=tmp_path / "arm.urdf")
    w.at = (0.65, 0.0, 0.315)  # where its frames show the container
    w.find = SimpleNamespace(places=(), patches=())  # ... and all else they show (the finder's answer)
    w.stops_at = None  # the joints a turn that is not flown leaves the arm at (None: where it stood)
    w.urdf.write_text("<robot/>")
    monkeypatch.setattr(press_demo.cli_common, "arm_urdf_file", lambda: w.urdf)

    def watch_front(
        node, ctx, runner, watcher, execute, say, stop, rest=None, camera_cfg_name=None, also=None, judge=None
    ):
        w.watched.append(rest)
        if w.watch[0] != "watched":
            if w.stops_at is not None:
                ctx.client.live = list(w.stops_at)
            return w.watch
        while not stop():
            why = None if judge is None else judge(_fix(w.at), w.find)
            if why is not None:
                say({"none": why})
            else:
                say({"fix": {"xyz": list(w.at), "table_z": 0.216, "age_s": 0.05, "rough": False, "by": "wrist"}})
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
    monkeypatch.setattr(press_demo, "LINE_UP_LOOK_S", 0.01)  # the scene camera's looks here answer at once
    return w


def _line_up(w, ear, wrist_first=True, execute=True, locator=None, instance_id=None, wrist_alone=False):
    out, runner = io.StringIO(), Runner()
    args = Namespace(wrist_first=wrist_first, execute=execute, instance_id=instance_id, wrist_alone=wrist_alone)
    got = press_demo.line_up_watch(None, w.ctx, w.cfg, runner, object(), args, locator, 0.05, ear=ear, out=out)
    lines = [json.loads(ln[3:]) for ln in out.getvalue().splitlines() if ln.startswith("@@ ")]
    w.held = got[2]  # whether the wrist's own watch held the container as the go came
    return got[:2], lines, runner


def test_at_rest_the_wrist_looks_and_a_go_lets_the_run_go_on_from_there(lined):
    (went, turned), lines, runner = _line_up(lined, Ear(3, go=True))
    assert (went, turned) == (True, True) and lined.watched == [list(HOME)] and lined.rested == []
    assert lined.held is True  # its watch said a fix a moment before: the run's first scene look is the short one
    assert lines[0] == {"robot": {"urdf": str(lined.urdf), "root": "arm_mount"}}
    assert [ln["fix"]["by"] for ln in lines[1:]] == ["wrist"] * 3
    assert runner.gripper == []  # shut already: nothing sent
    assert runner.notes[-1] == ("line_up", {"stage": "go", "wrist": True, "held": True})


def test_one_frame_without_a_find_before_the_go_is_not_the_wrist_having_lost_the_container(lined, monkeypatch):
    """Between the user's Open and the go's arrival the watch goes on judging frames. One that shows no find
    (or one the selection refuses) made the run's first scene look the long one again: the arm stood 15 s
    after Open with its wrist turned. The wrist held the container where its last fix is a moment old."""

    def watch_front(node, ctx, runner, watcher, execute, say, stop, **_kw):
        say({"fix": {"xyz": [0.65, 0.0, 0.315], "table_z": 0.216, "age_s": 0.05, "rough": False, "by": "wrist"}})
        say({"none": "wrist camera: no lid-sized top on the table in view"})
        while not stop():
            pass
        return "watched", None

    monkeypatch.setattr(sv, "watch_front", watch_front)
    (went, turned), _lines, _runner = _line_up(lined, Ear(2, go=True))
    assert (went, turned) == (True, True) and lined.held is True
    # ... but a fix that is old is the wrist no longer holding it
    monkeypatch.setattr(press_demo, "LINE_UP_HELD_S", -1.0)
    (went, turned), _lines, _runner = _line_up(lined, Ear(2, go=True))
    assert (went, turned) == (True, True) and lined.held is False


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
    # (what its "none" is of goes with it: the scene camera delivers, and has nothing that could be it)
    assert {"none": "scene segmenter: saw none", "kind": "unseen"} in lines
    assert scene.on_sighting is None  # unhooked as it ends
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

    lined.watch = ("stopped", "aborted (the driver stopped the goal)")
    lined.stops_at = [q + (0.3 if j == 5 else 0.0) for j, q in enumerate(HOME)]  # part way to the view
    scene = Scene()
    with pytest.raises(SystemExit) as e:
        _line_up(lined, Ear(5, go=True), locator=scene)
    assert e.value.code == EXIT_LINE_UP_HOLDS == 6
    assert lined.rested == [] and scene.on_sighting is None  # no further motion; unhooked as it ends


def test_a_wrist_turn_that_was_never_sent_leaves_the_arm_where_it_stood_and_the_scene_camera_looks(lined):
    """A turn the client would not send (no fresh joint states, the driver's action not there, its own gate)
    comes back "stopped" as one cut short does, but the arm has not left where it stood: nothing holds
    part-turned, so it is no EXIT_LINE_UP_HOLDS (the engine: arm_holds, not recoverable) for what a second
    press of open mends. The line-up goes on by the scene camera, as for any look that did not start."""
    lined.watch = ("stopped", "failed (no fresh /joint_states)")
    scene = Scene()
    (went, turned), lines, _runner = _line_up(lined, Ear(3, go=True), locator=scene)
    assert (went, turned) == (True, False) and lined.held is False and lined.rested == []
    # (why the wrist does not look is no camera's word on a container: the scene camera's looks follow)
    assert {"none": "wrist camera: failed (no fresh /joint_states)", "kind": "unseen"} in lines and scene.looks >= 1


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
    assert (went, turned) == (True, False)
    assert {"none": "wrist camera: its way there: the knees", "kind": "unseen"} in lines
    assert scene.looks >= 1


def test_a_scene_look_that_returns_at_once_is_waited_out(lined, monkeypatch):
    """A scene locator with no segmenter to read answers at once: the scene-only watch then span a core flat out
    for the whole line-up, beside the arm's driver. Each pass lasts the look's period."""

    class Clock:
        """The engine's stop, a quarter of a second in."""

        go, stop = False, False

        def __init__(self):
            self.t0 = time.monotonic()  # each ear its own quarter second

        def heard(self):
            self.stop = time.monotonic() - self.t0 >= 0.25
            return self.stop

    monkeypatch.setattr(press_demo, "LINE_UP_LOOK_S", 0.1)
    for locator in (Scene(), None):
        ear = Clock()
        _got, lines, _runner = _line_up(lined, ear, wrist_first=False, locator=locator)
        assert len([ln for ln in lines if "none" in ln]) <= 1
        if locator is not None:
            assert 1 <= locator.looks <= 3  # 0.25 s of 0.1 s looks
    # a look that ends early with its fix is followed by the next at once: a wait would only age its sightings
    quick = Scene()
    quick.locate = lambda timeout_s: setattr(quick, "looks", quick.looks + 1) or object()
    monkeypatch.setattr(press_demo, "LINE_UP_LOOK_S", 5.0)
    t0 = time.monotonic()
    _line_up(lined, Ear(3, go=False), wrist_first=False, locator=quick)
    assert quick.looks == 3 and time.monotonic() - t0 < 1.0


def test_the_box_the_segmenter_followed_while_the_chair_moved_is_not_kept_for_the_run(lined):
    """A scene fix made while lining up leaves the segmenter rung following the selected container by where its
    box was in the image then (SegmenterRung.near). The scene camera rides the chair, and the chair has moved
    since: in the run that box could name another container. The run's first look starts as a fresh run's."""
    for ear, kw in ((Ear(3, go=True), {}), (Ear(2, go=False), {}), (Ear(3, go=True), {"wrist_first": False})):
        scene = Scene()
        scene.seg = SimpleNamespace(near=(380.0, 150.0, 440.0, 230.0))
        _line_up(lined, ear, locator=scene, **kw)
        assert scene.seg.near is None and scene.on_sighting is None, kw
    scene = Scene()  # a locator with no segmenter rung (its container names no classes): nothing to clear
    scene.seg = None
    assert _line_up(lined, Ear(1, go=True), locator=scene)[0] == (True, True)


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
    from test_survey import MODEL, SELECTED, _along

    lined.ctx.model = MODEL
    # ... and what that "none" is of: a find nothing can tie to the selection, which no moving of the chair
    # makes the selected one (the engine says that, and ends the line-up on it: line_up.BoxSource)
    untied = {"none": press_demo.UNTIED % SELECTED, "kind": "not_chosen"}
    assert "in none of the scene camera's detections" in untied["none"] and len(untied["none"]) <= 300
    box = (380.0, 150.0, 440.0, 230.0)
    for kw, said in (
        ({"locator": _selecting(mine=None, held=0)}, False),  # the selection in none of its frames
        ({"locator": _selecting(mine=box, held=9)}, True),  # its box in the scene image ties a find to it
        ({"locator": _selecting(mine=None, held=0), "wrist_alone": True}, True),  # --wrist-alone: the one it sees
        ({"locator": _selecting(mine=None, held=None), "wrist_alone": True}, False),  # ... only once a look says so
        ({"locator": None}, False),  # no scene camera: nothing can tie it
    ):
        if kw.get("locator") is not None and kw["locator"].seg.latest.mine is not None:
            # where the scene camera's line of sight to the selected container's box meets the table in front
            tie = press_demo.selection_tie(kw["locator"], None, lined.ctx.model, SELECTED)
            lined.at = (*_along(tie, 0.6), 0.315)
        _got, lines, _runner = _line_up(lined, Ear(3, go=False), instance_id=SELECTED, **kw)
        fixes = [ln for ln in lines if ln.get("fix", {}).get("by") == "wrist"]
        assert bool(fixes) is said, kw
        assert (untied in lines) is not said, kw
    # no selection: every fix is said, as before
    _got, lines, _runner = _line_up(lined, Ear(3, go=False), locator=_selecting(mine=None, held=0))
    assert len([ln for ln in lines if "fix" in ln]) == 3


def test_with_a_selection_a_wrist_fix_the_run_would_not_take_is_not_said_either(lined, monkeypatch):
    """The run does not take every find it looks for: one that does not stand where the scene camera has the
    selected container is refused (tie_refusal), and with --wrist-alone one that is not the only container
    within the arm's reach in the wrist's frames (alone_refusal). The line-up asks the run's own two tests of
    each find, so "Ready" is never for a container the run would look at and leave: why is said instead."""
    from test_survey import MODEL, SELECTED, _along

    lined.ctx.model = MODEL
    scene = _selecting(mine=(380.0, 150.0, 440.0, 230.0), held=9)
    tie = press_demo.selection_tie(scene, None, MODEL, SELECTED)
    lined.at = (*_along(tie, 0.6, side=0.5), 0.315)  # half a metre to the side of where the selected one is seen
    _got, lines, _runner = _line_up(lined, Ear(3, go=False), locator=scene, instance_id=SELECTED)
    assert not [ln for ln in lines if ln.get("fix", {}).get("by") == "wrist"]
    elsewhere = [ln for ln in lines if "not taken for the selected container %s" % SELECTED in ln.get("none", "")]
    # (the scene camera holds the selected container and the wrist's lid stands off its line to it: another
    # thing, and the chosen one is not in the wrist's view: "unseen", which moving the chair cures and which
    # never ends a guidance. For one round, 2026-11-19, this was "not_chosen", and a bowl nearer the arm ended
    # a healthy guided approach in 10 s with "Choose it again" (review of 2026-11-20). The price is the scene
    # mount that is off: the cannot-see sentence for the whole wait with a lid in every frame)
    assert elsewhere and {ln["kind"] for ln in elsewhere} == {"unseen"}
    # --wrist-alone, the selection in none of the scene camera's frames: a second lid-sized top in the frames
    lined.at = (0.65, 0.0, 0.315)
    tops = [SimpleNamespace(xy=xy, button="unchecked") for xy in ((0.65, 0.0), (1.08, -0.23))]
    lined.find = SimpleNamespace(places=tuple(tops), patches=())
    kw = {"locator": _selecting(mine=None, held=0), "instance_id": SELECTED, "wrist_alone": True}
    monkeypatch.setattr(sv, "_within_reach", lambda model: lambda top: True)  # both within the arm's reach
    _got, lines, _runner = _line_up(lined, Ear(3, go=False), **kw)
    assert not [ln for ln in lines if ln.get("fix", {}).get("by") == "wrist"]
    two = [ln for ln in lines if "two containers" in ln.get("none", "")]
    assert two and {ln["kind"] for ln in two} == {"several"}  # two that could be it: no chair move parts them
    # ... but a top the arm cannot reach is no second container to choose between (2026-11-12: one 1.08 m out
    # stood in the wrist's frame on both runs that opened the box)
    monkeypatch.setattr(sv, "_within_reach", lambda model: lambda top: math.hypot(top[0], top[1]) < 0.9)
    kw["locator"] = _selecting(mine=None, held=0)
    _got, lines, _runner = _line_up(lined, Ear(3, go=False), **kw)
    assert len([ln for ln in lines if ln.get("fix", {}).get("by") == "wrist"]) == 3


def test_the_scene_cameras_watch_says_what_its_none_is_of_a_camera_that_delivers_nothing_among_them(lined):
    """The review of 2026-11-19: a scene camera whose streams are missing, or whose segmenter is silent, read
    on the panel as "The arm cannot see the container. Move the chair in front of it". The run's own watch
    says what each "none" is of: its words' own kind (guide.kind_of), with two things it knows better.

    A look the segmenter gave no frame NEWER than its start, the segmenter delivering by its own state
    (press_demo.SEG_DELIVERS): that segmenter is behind, not down. The watch's looks are a third of a second
    and its first frame comes 0.04 to 0.24 s into one by the logs the second review read, so some looks
    miss, and with a segmenter 0.3 s behind every look does: as a626985 had it, a second of such looks was
    "a camera that is not working", and ten ended the line-up. And a selected id that is in none of its
    frames says only that the id is absent: the container may simply be out of view, which is "unseen"."""
    late = "scene segmenter: no frame newer than the look in 0.3 s"
    silent = "scene segmenter: SILENT: no /perception/object_detections in 1.0 s"
    stale = "scene segmenter: its newest frame is 4.2 s older than the scene camera's"
    two = "scene segmenter: two places pass as this container, [0.8, -0.2] and [1.1, 0.3]"
    gone = "scene segmenter: the selected container (fc_1) was in none of its 2 frames"
    for scene, kind in (
        (Scene(frames=0, why="scene camera streams missing: color, depth", state=None), "fault"),
        (Scene(frames=0, why=silent, state="silent"), "fault"),
        (Scene(frames=0, why=stale, state="stale"), "fault"),
        (Scene(frames=0, why=late, state="live"), "unseen"),  # behind, for look after look: no fault
        (Scene(frames=0, why=late, state="waiting"), "unseen"),  # (its first second)
        (Scene(frames=2, why="scene segmenter: saw no food container in 2 frames"), "unseen"),
        (Scene(frames=2, why=two), "several"),
        (Scene(frames=2, why=gone), "unseen"),
        # a record that says nothing of the segmenter: the words alone
        (Scene(frames=None, why=late), "fault"),
        (Scene(frames=None, why="scene segmenter: saw none"), "unseen"),
    ):
        _got, lines, _runner = _line_up(lined, Ear(3, go=False), wrist_first=False, locator=scene)
        nones = [ln for ln in lines if "none" in ln]
        assert nones and all(ln == {"none": scene.last_why, "kind": kind} for ln in nones), scene.last_why
    # the segmenter's own words for one that delivers are none of its words for one that is down
    from box_opening import guide
    from box_opening.perception import scene_source

    assert press_demo.SEG_DELIVERS == ("live", "waiting")
    assert not set(press_demo.SEG_DELIVERS) & set(scene_source.SEG_DOWN)
    assert press_demo.SEG_LATE in late and press_demo.SEG_LATE in guide.FAULT_MARKS


# -- in the middle of a run: no press plans from where the chair stands (--guide) -----------------------
class Pipe:
    """The engine's end of the run's stdin: lines as they are written, the pipe open until close()."""

    def __init__(self):
        import queue

        self._q = queue.Queue()

    def say(self, msg):
        self._q.put(json.dumps(msg) + "\n")

    def close(self):
        self._q.put(None)

    def __iter__(self):
        while True:
            line = self._q.get()
            if line is None:
                return
            yield line


def _until(what, s=2.0):
    t0 = time.monotonic()
    while not what() and time.monotonic() - t0 < s:
        time.sleep(0.005)
    return what()


def test_one_ear_hears_every_line_up_of_a_run_a_go_each_round():
    """A run may be lined up before its first move and again wherever no press plans: one listener for the
    run's life. As first written it ended at its first go, so a second line-up heard nothing and watched
    for ever. again() opens the next round with the go that let the run on forgotten."""
    pipe = Pipe()
    ear = press_demo.LineUpEar(pipe)
    assert not ear.heard()
    pipe.say({"go": True})
    assert _until(ear.heard) and ear.go and not ear.stop
    ear.again()  # the run flew on, found no press, and asks to be guided
    assert not ear.heard() and not ear.go
    pipe.say({"go": True})
    assert _until(ear.heard) and ear.go
    ear.again()
    pipe.say({"stop": True})
    assert _until(ear.heard) and ear.stop and not ear.go
    # an engine that is gone has not said go: the pipe closed after a go is the NEXT round's stop
    pipe = Pipe()
    ear = press_demo.LineUpEar(pipe)
    pipe.say({"go": True})
    assert _until(ear.heard) and ear.go
    pipe.close()
    time.sleep(0.05)
    assert ear.go and not ear.stop  # the run it let go flies on, as it did
    ear.again()
    assert ear.heard() and ear.stop and not ear.go


def _guiding(w, ear, place=(0.712, -0.331, 0.2945), why="no reach-out press plans (30 deg: the slide in)", **kw):
    """press_demo.guide_chair on `lined`'s arm: ((turned, held) or the exit code, its lines, the runner)."""
    out, runner = io.StringIO(), Runner()
    args = Namespace(wrist_first=kw.pop("wrist_first", True), execute=True, instance_id=None, wrist_alone=False)
    watcher = SimpleNamespace(table_z=0.2064, roi=(1, 2, 3, 4))
    w.watcher = watcher
    try:
        got = press_demo.guide_chair(
            None, w.ctx, w.cfg, runner, watcher, args, kw.pop("locator", None), 0.05, place, why, 1, ear, out=out, **kw
        )
    except SystemExit as e:
        got = e.code
    lines = [json.loads(ln[3:]) for ln in out.getvalue().splitlines() if ln.startswith("@@ ")]
    return got, lines, runner


@pytest.fixture
def guided(lined, monkeypatch):
    """`lined`, as a run stands where it found no press: the wrist where its own look left it, the scene as
    its cameras measured it from there (the table, the counter's edges kept out, the box's pose)."""
    from box_opening.runtime import approach

    lined.ctx.client.live = [v + (0.4 if j in sv.FRONT_JOINTS else 0.02) for j, v in enumerate(HOME)]
    lined.ctx.cpose, lined.ctx.counter_boxes = object(), {"look": [object()]}
    lined.ctx.survey_clouds, lined.ctx.wrist_far, lined.ctx.home_ahead = [object()], [(1.0, 0.0, 0.3)], [object()]
    lined.ctx.worlds.table_top_z, lined.ctx.worlds.table_extent = 0.2064, (0.53, 1.16, -0.83, -0.02)
    lined.obstacles = []
    monkeypatch.setattr(approach, "set_obstacles", lambda boxes: lined.obstacles.append(list(boxes)))
    lined.homes = []
    monkeypatch.setattr(press_demo, "try_home", lambda ctx, runner, execute, why: lined.homes.append(why) or True)
    return lined


class Round(Ear):
    """An Ear that is asked for the next round first, as guide_chair asks it."""

    def __init__(self, n, go):
        super().__init__(n, go)
        self.rounds = 0

    def again(self):
        self.rounds += 1
        self.go = self.stop = False
        self.asks = 0


def test_a_run_with_no_press_from_where_the_chair_stands_asks_to_be_guided_and_watches_until_the_go(guided):
    """Run 2 of 2026-11-17 ended here. Now: the engine is asked first, with where the box stood and why no
    press planned there; then the run watches exactly as a line-up's watch does (the wrist's fixes said), and
    at the go it goes on from where the wrist stands. Nothing but the wrist moved."""
    ear = Round(3, go=True)
    got, lines, runner = _guiding(guided, ear)
    assert got == (True, True) and ear.rounds == 1  # (wrist_turned, wrist_held): the look after it is short
    assert lines[0] == {
        "guide": {"n": 1, "box": [0.712, -0.331, 0.2945], "why": "no reach-out press plans (30 deg: the slide in)"}
    }
    assert lines[1] == {"robot": {"urdf": str(guided.urdf), "root": "arm_mount"}}
    assert [ln["fix"]["by"] for ln in lines[2:]] == ["wrist"] * 3
    assert guided.watched == [list(HOME)] and guided.rested == [] and guided.homes == [] and runner.gripper == []
    assert ("guide", {"n": 1, "box": [0.712, -0.331, 0.2945], "why": lines[0]["guide"]["why"]}) in runner.notes


NOT_SEEN = (
    "no camera sees the container (scene camera: scene segmenter: saw no food container in 14 frames; the wrist's "
    "look in front: wrist camera: no counter in 4 frames (no level plane wider than a container))"
)


def test_a_run_whose_cameras_saw_no_container_asks_to_be_guided_with_no_place_and_watches_the_same_way(guided):
    """Owner 2026-11-19: a box no camera sees at all is guidance too. Nothing of the guidance starts from a
    container having been seen, so the run that saw none is guided as one that found no press is: the engine
    is asked first (with why, what it is guided for, and no place: none was measured), what the cameras
    measured from where the chair stands is forgotten, and the run watches as every line-up's watch does
    until the go. The arm rests; nothing but the wrist moves."""
    guided.at = (1.05, 0.0, 0.315)  # where its frames then show the container: brought into view, still far
    ear = Round(3, go=True)
    got, lines, runner = _guiding(guided, ear, place=None, why=NOT_SEEN, what=press_demo.GUIDE_UNSEEN)
    assert got == (True, True) and ear.rounds == 1
    assert lines[0] == {"guide": {"n": 1, "box": None, "why": NOT_SEEN, "what": "unseen"}}
    assert lines[1] == {"robot": {"urdf": str(guided.urdf), "root": "arm_mount"}}
    assert [ln["fix"]["xyz"] for ln in lines[2:]] == [[1.05, 0.0, 0.315]] * 3  # whichever way it is seen, said
    assert guided.watched == [list(HOME)] and guided.rested == [] and guided.homes == [] and runner.gripper == []
    assert ("guide", {"n": 1, "box": None, "why": NOT_SEEN, "what": "unseen"}) in runner.notes
    ctx = guided.ctx
    assert ctx.cpose is None and ctx.worlds.table_top_z is None and ctx.worlds.table_extent is None
    assert ctx.wrist_seen is None and ctx.side_seen is None and guided.obstacles == [[]]
    # a stop ends it as it ends every guidance: the wrist back to rest, nothing pressed
    from box_opening.runtime.abort import EXIT_NOT_LINED_UP

    got, _lines, _runner = _guiding(guided, Round(2, go=False), place=None, why=NOT_SEEN, what="unseen")
    assert got == EXIT_NOT_LINED_UP and guided.rested == [list(HOME)]


def test_with_no_container_seen_the_wrist_turns_to_the_arms_own_front_view_not_toward_any_place_of_the_scenes(
    guided, monkeypatch
):
    """Where does the wrist look when nothing has been seen? Where it looks for every line-up: the front
    view that shows the most of where the arm itself presses (survey.watch_front with no hint, chosen on
    each run from the arm's kinematics and the chair's model). That is where the container has to be
    brought, and it stays true as the chair is driven. It is handed no place of the scene's: not a box's
    (none was seen), and not the table's as the cameras measured it before the move, which is forgotten
    because it is true of nowhere once the chair has moved."""
    asked = []

    def watch_front(node, ctx, runner, watcher, execute, say, stop, **kw):
        asked.append((kw, ctx.worlds.table_top_z, ctx.worlds.table_extent, ctx.cpose))
        while not stop():
            say({"none": "wrist camera: no counter in 1 frame (no level plane wider than a container)"})
        return "watched", None

    monkeypatch.setattr(sv, "watch_front", watch_front)
    got, lines, _runner = _guiding(guided, Round(2, go=False), place=None, why=NOT_SEEN, what="unseen")
    ((kw, table_z, extent, cpose),) = asked
    assert kw.get("hint") is None and kw["rest"] == list(HOME)
    assert (table_z, extent, cpose) == (None, None, None)  # nothing of the scene from before the move is kept
    assert [ln for ln in lines if "none" in ln][0] == {
        "none": "wrist camera: no counter in 1 frame (no level plane wider than a container)"
    }
    assert got == 5


def test_an_arm_that_does_not_get_home_before_an_unseen_container_is_guided_for_holds_and_says_what_it_was(
    guided, capsys
):
    """The side survey's pose: the arm homes before the chair is told to move, and one that does not get
    home holds. Its STOP says what the run was to be guided for; until 2026-11-19 it said "no press plans
    from where the chair stands" of a run that had tried no press."""
    guided.ctx.client.live = [v + 0.5 for v in HOME]
    press_demo_try_home = press_demo.try_home
    press_demo.try_home = lambda ctx, runner, execute, why: False
    try:
        got, lines, _runner = _guiding(guided, Round(2, go=True), place=None, why=NOT_SEEN, what="unseen")
        out = capsys.readouterr().out
        assert got == 1 and lines == [] and guided.watched == []  # never asked, never watched from there
        assert "STOP: no camera sees the container, and the arm did not get home for the chair to be moved" in out
        got, _lines, _runner = _guiding(guided, Round(2, go=True))
        assert (
            "STOP: no press plans from where the chair stands, and the arm did not get home" in capsys.readouterr().out
        )
    finally:
        press_demo.try_home = press_demo_try_home


# run 2 of 2026-11-17's own line (NOTES.md), 752 characters
RUN2_WHY = (
    "no reach-out press plans for the scene camera's box (30 deg: the way down (the slide in: trajectory sanity "
    "gate: arm_joint_1 excursion 2.012 rad > |Δ| + margin 1.632; arm_joint_3 excursion 1.525 rad > |Δ| + margin "
    "1.120); 35 deg: the way down (the slide in: trajectory sanity gate: arm_joint_1 excursion 2.188 rad > |Δ| + "
    "margin 2.049; arm_joint_3 excursion 1.668 rad > |Δ| + margin 1.567); 40 deg: the way down (the press stroke "
    "is not a straight line the arm can fly (the chair and user: arm_half_arm_1_link 0 mm into person/lap (padded "
    "20 mm + 5 mm), 46% of the way)); 45 deg: the way down (the press stroke is not a straight line the arm can "
    "fly (the chair and user: arm_half_arm_1_link 0 mm into person/lap (padded 20 mm + 5 mm), 92% of the way)))"
)


def test_the_engine_is_handed_the_whole_reason_not_its_first_300_characters(guided):
    """The review of 2026-11-17: the request carried `str(why)[:300]`, which for run 2's own line ends inside
    its second tilt. What the engine said from it (the progress a person at open_box_terminal reads, the
    not_lined_up message, the run log's guide row) gave the slide in's two tilts and dropped "the press
    stroke would pass too close to the user (40, 45 deg)": half the reason, for the person with no console."""
    from box_opening.flight_view import no_press_reason

    whole = (
        "no press plans from here: the slide in swings the arm wide (30, 35 deg); the press stroke would pass "
        "too close to the user (40, 45 deg)"
    )
    assert len(RUN2_WHY) > 700 and no_press_reason(RUN2_WHY) == whole
    assert no_press_reason(RUN2_WHY[:300]) == "no press plans from here: the slide in swings the arm wide (30, 35 deg)"
    _got, lines, runner = _guiding(guided, Round(1, go=True), why=RUN2_WHY)
    assert lines[0]["guide"]["why"] == RUN2_WHY and no_press_reason(lines[0]["guide"]["why"]) == whole
    (row,) = (fields for leg, fields in runner.notes if leg == "guide")
    assert row["why"] == RUN2_WHY
    # a line longer than a pipe writes whole (the request shares the run's log pipe) is shortened at the end
    # of a tilt's reason, never inside one: what is said of a tilt is its real reason or nothing
    gate = "the way down (the slide in: trajectory sanity gate: arm_joint_1 excursion 2.0 rad%s)"
    long = "no reach-out press plans (%s)" % "; ".join(
        "%d deg: %s" % (t, gate % (" x" * 450)) for t in (30, 35, 40, 45)
    )
    short = press_demo.reason_for_engine(long)
    assert len(long) > 3600 and len(short) <= press_demo.GUIDE_WHY_MAX + 4 and short.endswith(") ...")
    assert no_press_reason(short) == "no press plans from here: the slide in swings the arm wide (30, 35 deg)"
    _got, lines, _runner = _guiding(guided, Round(1, go=True), why=long)
    assert lines[0]["guide"]["why"] == short
    assert len(json.dumps({"guide": {"n": 1, "box": [0.712, -0.331, 0.2945], "why": short}})) < 4096 - 3


def test_what_the_cameras_measured_before_the_chair_moves_is_forgotten(guided):
    """The owner's rule: the table's height and extent, what stands on it, only from the cameras, each time.
    The run measured them from where the chair stood. After the move they are true of nowhere: the table's
    extent is in the arm's frame, and the counter's edges the arm is held out of would stand in the air."""
    _guiding(guided, Round(2, go=True))
    ctx = guided.ctx
    assert ctx.cpose is None and ctx.counter_boxes is None and ctx.survey_clouds is None
    assert ctx.wrist_far is None and ctx.home_ahead is None
    assert guided.obstacles == [[]]  # the body check's run-time boxes cleared
    assert ctx.worlds.table_top_z is None and ctx.worlds.table_extent is None
    assert guided.watcher.table_z is None and guided.watcher.roi is None


def test_a_table_the_run_measured_is_no_table_once_the_chair_moves_on_a_world_that_names_none():
    """flatbot's world file names no table: the run adopts the one its first fix measured
    (press_demo.adopt_scene_table), which adopts nothing while a table is known. Forgotten, the look after
    the move adopts its own; left in place, that look kept the height and extent measured from where the
    chair had stood."""
    from box_opening.worlds import WorldStore

    worlds = WorldStore("capabilities/box_opening/config/world_flatbot.yaml")
    assert worlds.table_top_z is None
    ctx = SimpleNamespace(mission_frames=None)
    watcher = SimpleNamespace(table_z=None, roi=None)
    fix = SimpleNamespace(table_z_scene=0.2064, table_extent=(0.53, 1.16, -0.83, -0.02))
    press_demo.adopt_scene_table(ctx, worlds, watcher, fix)
    assert worlds.table_top_z == 0.2064 and worlds.table_extent == (0.53, 1.16, -0.83, -0.02)
    moved = SimpleNamespace(table_z_scene=0.2110, table_extent=(0.40, 1.02, -0.70, 0.10))
    press_demo.adopt_scene_table(ctx, worlds, watcher, moved)
    assert worlds.table_top_z == 0.2064  # (kept: this is why it must be forgotten)
    from box_opening.runtime import approach

    real, approach.set_obstacles = approach.set_obstacles, lambda boxes: None
    try:
        press_demo.forget_scene(ctx, worlds, watcher)
    finally:
        approach.set_obstacles = real
    assert worlds.table_top_z is None and worlds.table_extent is None and watcher.table_z is None
    press_demo.adopt_scene_table(ctx, worlds, watcher, moved)
    assert worlds.table_top_z == 0.2110 and worlds.table_extent == (0.40, 1.02, -0.70, 0.10)
    assert watcher.table_z == 0.2110


def test_a_stop_while_it_is_guided_turns_the_wrist_back_and_ends_the_run_nothing_pressed(guided):
    from box_opening.runtime.abort import EXIT_NOT_LINED_UP

    got, _lines, runner = _guiding(guided, Round(2, go=False))
    assert got == EXIT_NOT_LINED_UP == 5 and guided.rested == [list(HOME)]
    assert runner.notes[-1] == ("line_up", {"stage": "stop", "wrist": True})


def test_an_engine_with_no_time_to_guide_still_has_the_wrist_the_look_turned_put_back(guided):
    """The stop comes before the watch's own turn (the orchestrator's 120 s left no time): the watch turns
    nothing, and the wrist the run's own look left turned is turned back to rest all the same. Without it the
    run ended as run 2 did: the arm where the look left it."""
    from box_opening.runtime.abort import EXIT_LINE_UP_HOLDS, EXIT_NOT_LINED_UP

    guided.watch = ("none", "stopped before the turn")
    got, _lines, _runner = _guiding(guided, Round(0, go=False))
    assert got == EXIT_NOT_LINED_UP and guided.rested == [list(HOME)]
    # ... and one that does not turn back is the arm holding, as after any line-up
    import box_opening.tasks.survey as survey

    real = survey.rest_wrist
    survey.rest_wrist = lambda ctx, runner, execute, rest=None: ("stopped", "aborted")
    try:
        got, _lines, _runner = _guiding(guided, Round(0, go=False))
    finally:
        survey.rest_wrist = real
    assert got == EXIT_LINE_UP_HOLDS
    # a line-up BEFORE the run never did this, and does not now: its arm is as it found it
    guided.rested.clear()
    (went, turned), _lines, _runner = _line_up(guided, Ear(0, go=False))
    assert (went, turned) == (False, False) and guided.rested == []


def test_an_arm_the_look_left_away_from_rest_goes_home_before_the_chair_is_told_to_move(guided):
    """The side survey's pose, or anywhere a look left the arm: the chair is moved only with the arm at
    rest, its wrist alone turned. It homes first, and only then asks to be guided; a home that fails ends
    the run, the arm holding, and the engine is never asked."""
    away = [v + 0.5 for v in HOME]
    guided.ctx.client.live = list(away)

    def home(ctx, runner, execute, why):
        guided.homes.append(why)
        ctx.client.live = list(HOME)
        return True

    press_demo_try_home = press_demo.try_home
    press_demo.try_home = home
    try:
        got, lines, _runner = _guiding(guided, Round(2, go=True))
    finally:
        press_demo.try_home = press_demo_try_home
    assert got == (True, True) and len(guided.homes) == 1 and "guide" in lines[0]
    # what that home is called is read by the flight view, which took "NO WAY IN: the arm goes to rest ..."
    # for a second refusal and printed a PRESS FAIL of a run that had tried no press (review of 2026-11-17):
    # as try_home logs it, the line is the arm's way to rest and nothing else
    from box_opening.flight_view import FlightView

    said = []
    view = FlightView(write=said.append)
    view.start()
    view.feed("[press_demo] OUT OF REACH: the wrist camera sees a lid-sized top at [1.02, -0.20]; nothing was pressed")
    view.feed("[press_demo] %s — returning home" % guided.homes[0])
    assert guided.homes[0].startswith("TO REST: ")
    assert said[-1] == "   home      ...   going to rest before the chair is guided"
    assert len([line for line in said if " FAIL " in line]) == 1 and not any("PRESS" in line for line in said)
    guided.ctx.client.live = list(away)
    watched = len(guided.watched)
    press_demo.try_home = lambda ctx, runner, execute, why: False
    try:
        got, lines, _runner = _guiding(guided, Round(2, go=True))
    finally:
        press_demo.try_home = press_demo_try_home
    assert got == 1 and lines == [] and len(guided.watched) == watched  # never asked, never watched from there


def test_the_tops_the_wrist_sees_out_of_reach_are_the_ones_the_chair_is_moved_for(monkeypatch):
    """far_tops: lid-sized tops in the wrist's frames that ALL stand out of the arm's reach (the finder then
    has no fix, and the run said NO BOX). With one within reach, or none seen, or a container found, there
    is nothing to move the chair for on this account."""
    top = lambda x, y, button="unchecked": SimpleNamespace(xy=(x, y), top_z=0.30, button=button)  # noqa: E731
    monkeypatch.setattr(sv, "_within_reach", lambda model: lambda t: math.hypot(t[0], t[1]) < 0.9)
    find = SimpleNamespace(fix=None, places=(top(1.10, 0.2), top(1.00, -0.1)))
    assert press_demo.far_tops(find, object()) == [(1.0, -0.1, 0.30), (1.10, 0.2, 0.30)]  # the nearest first
    assert press_demo.far_tops(SimpleNamespace(fix=None, places=(top(1.10, 0.2), top(0.70, 0.0))), object()) == []
    assert press_demo.far_tops(SimpleNamespace(fix=None, places=()), object()) == []
    assert press_demo.far_tops(SimpleNamespace(fix=object(), places=(top(1.10, 0.2),)), object()) == []
    assert press_demo.far_tops(SimpleNamespace(fix=None, places=(top(1.10, 0.2, button="none"),)), object()) == []
    assert press_demo.far_tops(None, object()) == []
    # a find it cannot read (a place with no height) is no reason to guide, and never the look's end
    assert press_demo.far_tops(SimpleNamespace(fix=None, places=(SimpleNamespace(xy=(1.1, 0.2)),)), object()) == []


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
