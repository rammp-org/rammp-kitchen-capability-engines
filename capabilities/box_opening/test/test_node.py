"""The engine node (box_opening.node): a run's rows as progress and result, its last lines as
the error code, a cancel as its own Ctrl+C, and the dry-run gate. The child is a stand-in
script (FAKE_RUN) that writes a run log and exits as the command line would."""

import json
import logging
import signal
import sys
import textwrap
import time
from pathlib import Path

import pytest
import rclpy
from capability_engine.engine import Cancelled, EngineError
from rclpy.parameter import Parameter

from box_opening import LOGGER, node

FRAGMENT_MOCK = Path(__file__).resolve().parents[1] / "rammp-alternative.mock.yaml"

# rows of a full run on flatbot (run-20261030-132845), shortened
FULL_RUN = [
    {"leg": "scene_fix", "kind": "note", "origin_xyz": [0.6018, -0.2302, 0.2091], "source": "segmenter"},
    {"leg": "reach:look", "kind": "motion", "ok": True, "outcome": "arrived"},
    {"leg": "press:close", "kind": "gripper", "ok": True, "outcome": "arrived"},
    {"leg": "reach:slide-in", "kind": "motion", "ok": True, "outcome": "arrived"},
    {"leg": "depth:press", "kind": "motion", "ok": True, "outcome": "arrived"},
    {"leg": "grip:open", "kind": "gripper", "ok": True, "outcome": "arrived"},
    {"leg": "lid_spot", "kind": "note", "xyz": [0.5965, -0.0706, 0.2091], "tier": "beside the box"},
    {"leg": "grip:close", "kind": "gripper", "ok": True, "outcome": "arrived"},
    {"leg": "lid:carry", "kind": "motion", "ok": True, "outcome": "arrived"},
    {"leg": "place:touch", "kind": "motion", "ok": True, "outcome": "arrived"},
    {"leg": "reach_grip_result", "kind": "note", "ok": True, "detail": "grip 0.472 vs band [0.3, 0.55]"},
    {"leg": "reach_setdown", "kind": "note", "knuckle": 0.79, "kept": False},
    {"leg": "place:retreat", "kind": "motion", "ok": True, "outcome": "arrived"},
    {"leg": "place:open", "kind": "gripper", "ok": True, "outcome": "arrived"},
    {"leg": "home", "kind": "motion", "ok": True, "outcome": "arrived"},
]

# the stand-in for press_demo: its run log's rows (ROWS), its console lines (SAY), then
# EXIT; with WAIT, it waits for Ctrl+C and exits 130 as the command line does
FAKE_RUN = """
import json, os, sys, time
from pathlib import Path
if "--line-up" in sys.argv:
    # press_demo --line-up: says where the box is (LINE_UP: "in place" | "far" | "dies") until a line on
    # stdin: go and the run goes on, stop (or the pipe closed) and it ends 5, nothing pressed, once its
    # wrist is back at rest (TURN_BACK_S: how long that turn takes; a Ctrl+C stops it there, 130)
    import select
    mode = os.environ.get("LINE_UP", "in place")
    if mode == "dies":
        sys.stdout.write("[press_demo] not starting: the arm driver is not up\\n")
        sys.stdout.flush()
        sys.exit(1)
    sys.stdout.write('@@ {"robot": {"urdf": null, "root": "arm_mount"}}\\n')
    sys.stdout.write("[press_demo] LINE-UP: the wrist turns to look at the table in front\\n")
    heard = None
    try:
        while heard is None:
            x = 0.65 if mode == "in place" else 1.2
            fix = {"xyz": [x, 0.0, 0.315], "table_z": 0.216, "age_s": 0.05, "rough": False, "by": "wrist"}
            sys.stdout.write("@@ " + json.dumps({"fix": fix}) + "\\n")
            sys.stdout.flush()
            if select.select([sys.stdin], [], [], 0.1)[0]:
                line = sys.stdin.readline()
                heard = json.loads(line) if line.strip() else {"stop": True}
    except KeyboardInterrupt:
        sys.stdout.write("[press_demo] stopped by Ctrl+C\\n")
        sys.exit(130)
    if not heard.get("go"):
        try:
            time.sleep(float(os.environ.get("TURN_BACK_S", "0")))
        except KeyboardInterrupt:
            sys.stdout.write("[press_demo] stopped by Ctrl+C\\n")
            sys.exit(130)
        sys.stdout.write("[press_demo] LINE-UP: ended without a go; nothing was pressed, the wrist back at rest\\n")
        sys.stdout.flush()
        sys.exit(5)
    sys.stdout.write("[press_demo] LINE-UP: go\\n")
runs = Path(os.environ["BOX_OPENING_STATE"]) / "runs"
runs.mkdir(parents=True, exist_ok=True)
with open(runs / "run-20261030-140000.jsonl", "a") as f:
    for row in json.loads(os.environ["ROWS"]):
        f.write(json.dumps(row) + "\\n")
for line in json.loads(os.environ["SAY"]):
    sys.stdout.write(line + "\\n")
sys.stdout.write("ARGV " + " ".join(sys.argv[1:]) + "\\n")
sys.stdout.flush()
if os.environ.get("WAIT"):
    try:
        time.sleep(30)
    except KeyboardInterrupt:
        sys.stdout.write("[press_demo] stopped by Ctrl+C\\n")
        sys.exit(130)
code = int(os.environ["EXIT"])
if code < 0:  # killed by that signal, as the node's kill ends a run that does not stop
    os.kill(os.getpid(), -code)
sys.exit(code)
"""


class FakeCtx:
    def __init__(self, accepts=True):
        self.said = []
        self.cancel = False
        self.objects = {}
        self.world = []  # every world-state write, in order
        self.accepts = accepts  # what the orchestrator answers a world-state write
        self.panels = []  # every panel sent to the UI, in order
        self.events = []  # what the UI reports, oldest first: read one at a time
        self.ui_sends = True  # what a send to the UI answers

    def progress(self, phase, pct, message="", pausable=True):
        self.said.append((phase, pct, message, pausable))

    def update_user_panel(self, panel_json=None, objects=None, selected_instance_id=None):
        if self.ui_sends:
            self.panels.append(panel_json)
        return self.ui_sends

    def poll_ui_event(self):
        return self.events.pop(0) if self.events else None

    def checkpoint(self):
        if self.cancel:
            raise Cancelled()

    def cancelled(self):
        return self.cancel

    def update_object(self, instance_id, **fields):
        if self.accepts:
            self.objects[instance_id] = fields
        return self.accepts

    def write_world_state(self, **updates):
        if self.accepts:
            self.world.append(updates)
        return self.accepts


class NowWriter:
    """node.WorldWriter, each write made at once on the caller's thread."""

    def put(self, what, write, done=None):
        ok = write()
        if done is not None:
            done(ok)


@pytest.fixture(scope="module")
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def engine(ros, tmp_path, monkeypatch):
    """A real engine node, standalone, its runs the FAKE_RUN script in `tmp_path`."""
    handlers = list(LOGGER.handlers)
    script = tmp_path / "fake_run.py"
    script.write_text(FAKE_RUN)
    overrides = {"use_orchestrator": False, "allow_execute": True, "state_dir": str(tmp_path)}
    cap = node.Capability(parameter_overrides=[Parameter(k, value=v) for k, v in overrides.items()])
    real_argv = cap._argv
    monkeypatch.setattr(cap, "_argv", lambda *a: [sys.executable, str(script)] + real_argv(*a)[1:])
    yield cap
    cap.destroy_node()
    LOGGER.handlers[:] = handlers  # the console again, for the other tests' capsys


def given(monkeypatch, rows=(), say=(), code=0, wait=False):
    monkeypatch.setenv("ROWS", json.dumps(list(rows)))
    monkeypatch.setenv("SAY", json.dumps(list(say)))
    monkeypatch.setenv("EXIT", str(code))
    if wait:
        monkeypatch.setenv("WAIT", "1")


def test_a_full_run_reports_its_phases_in_order_and_the_lid_off(engine, monkeypatch):
    video = "/state/captures/lid-video-20261030-140000.mp4"
    given(
        monkeypatch, FULL_RUN, ["[press_demo] VIDEO: what the wrist and scene cameras saw from the grip on: " + video]
    )
    ctx = FakeCtx()
    engine.instance_id = "food_container_1"
    out = engine.execute("open_box", {}, ctx)
    phases = [p for p, *_ in ctx.said]
    assert phases[0] == "locate" and phases[-1] == "home"
    order = ["locate", "look", "press", "grip", "carry", "release", "home"]
    assert [p for p in order if p in phases] == order
    pcts = [pct for _, pct, *_ in ctx.said]
    assert pcts == sorted(pcts) and 0.0 <= pcts[0] and pcts[-1] == 1.0  # REVIEW_RULES U2: 0 to 1, never backwards
    assert not any(pausable for *_, pausable in ctx.said)  # one motion: no pause once started
    # REVIEW_RULES U2: a person's words, never a leg's name ("depth:press: arrived")
    messages = [m for _, _, m, _ in ctx.said]
    assert "pressing the button" in messages and "setting the lid down" in messages and messages[-1] == "going home"
    assert not any(":" in m or "_" in m for m in messages)
    assert out["executed"] and out["lid_off"] and out["lid_xyz"] == [0.597, -0.071, 0.209]
    assert (out["presses"], out["grips"], out["video"]) == (1, 1, video)
    assert out["run_log"].endswith("run-20261030-140000.jsonl")
    # the object record's own field: the orchestrator refuses any other (REVIEW_RULES O6)
    assert ctx.objects == {"food_container_1": {"opened": True}} and out["world_state"] == "opened"
    # REVIEW_RULES C7: never left holding (the grip and the release came in one read here)
    assert not ctx.world or ctx.world[-1] == {"gripper": {"holding": None}}


def test_the_world_state_holds_the_lid_from_the_grip_that_held_it_to_its_release(engine):
    """REVIEW_RULES C7, row by row as a run writes them: holding from the grip's held result,
    empty again at the release (or a put-back's), each change written once."""
    ctx = FakeCtx()
    track = node.Tracker("open_box", "food_container_1")
    said = []
    for row in FULL_RUN:
        track.add(row)
        engine._sync_held(track, ctx, NowWriter())
        said.append(track.held)
    assert ctx.world == [{"gripper": {"holding": "food_container_1:lid"}}, {"gripper": {"holding": None}}]
    assert said.index(True) == 10 and said.index(False, 10) == 13  # reach_grip_result, then place:open
    track = node.Tracker("open_box")  # standalone: nothing to tell
    track.add(FULL_RUN[10])
    engine._sync_held(track, FakeCtx(), NowWriter())
    assert track.held and not track.held_said


def test_a_run_that_stops_with_the_lid_in_the_fingers_leaves_it_said(engine, monkeypatch):
    """REVIEW_RULES C7: the run stopped part way with the lid held (the driver stopped the
    carry): the world state already says the gripper holds it (written as the grip held, so a
    cancel needs no write either), and the error says so."""
    stop = "[press_demo] STOP: the lid's way to the table stopped during lid:carry (aborted) — the arm holds the lid"
    given(monkeypatch, FULL_RUN[:8] + [FULL_RUN[10]], [stop], code=1)  # the grip held, then the carry stopped
    ctx = FakeCtx()
    engine.instance_id = "food_container_1"
    engine.orchestrated = True  # the arm's system_status is the orchestrator's (standalone: nothing to tell)
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert e.value.error_code == "arm_holds" and not e.value.recoverable
    assert str(e.value).endswith("the lid is still in the fingers")
    assert ctx.world[0] == {"gripper": {"holding": "food_container_1:lid"}}
    # not recoverable: the arm's domain in error, which the orchestrator honours (must-fix, round 2)
    (status,) = (w["system_status"] for w in ctx.world[1:])
    assert status["arm"]["state"] == "error" and status["arm"]["error_code"] == "arm_holds"
    assert "the lid is still in the fingers" in status["arm"]["message"]
    # and open_box is not offered while the gripper holds it, whatever the container's record says
    world = {"gripper": {"holding": "food_container_1:lid"}, "objects": {"food_container_1": {"opened": False}}}
    (a,) = engine.availability("food_container_1", world)
    assert not a.available and "food_container_1:lid" in a.reason


def test_a_lid_put_back_on_the_box_leaves_the_gripper_empty_and_the_run_retryable(engine, monkeypatch):
    """No carry planned: the run put the lid back on the box and went home (reach_lid_off
    .put_lid_back). The lid is no longer held, and a retry can succeed."""
    rows = (
        FULL_RUN[:8]
        + [FULL_RUN[10]]
        + [
            {"leg": "lid:release", "kind": "gripper", "ok": True, "outcome": "arrived"},
            {"leg": "lid_put_back", "kind": "note", "ok": True, "why": "no carry to a spot the lid could go to plans"},
            {"leg": "home", "kind": "motion", "ok": True, "outcome": "arrived"},
        ]
    )
    stop = (
        "[press_demo] STOP: no carry to a spot the lid could go to plans — the lid is back on the box, "
        "and the arm is home"
    )
    given(monkeypatch, rows, [stop], code=1)
    ctx = FakeCtx()
    engine.instance_id = "food_container_1"
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert e.value.error_code == "lid_put_back" and e.value.recoverable and "still in the fingers" not in str(e.value)
    assert not ctx.world or ctx.world[-1] == {"gripper": {"holding": None}}  # not left holding
    assert not any("system_status" in w for w in ctx.world)  # recoverable: the arm is not in error


def test_the_selection_is_read_once_as_the_goal_comes(engine, monkeypatch):
    """REVIEW_RULES O3: a hibernate mid-run (the orchestrator's 120 s, or another selection)
    clears the engine's instance_id. The run's writes still name the container it was started
    for, never a later selection's, and never none."""
    given(monkeypatch, FULL_RUN)
    ctx = FakeCtx()
    engine.instance_id = "food_container_1"
    real = node.Capability._progress
    calls = []

    def progress(log, track, ctx_, panel):
        calls.append(1)
        engine.instance_id = "" if len(calls) == 1 else "food_container_2"  # hibernated, then another selected
        real(log, track, ctx_, panel)

    monkeypatch.setattr(node.Capability, "_progress", staticmethod(progress))
    out = engine.execute("open_box", {}, ctx)
    assert out["lid_off"] and ctx.objects == {"food_container_1": {"opened": True}}
    assert all(w["gripper"]["holding"] in (None, "food_container_1:lid") for w in ctx.world)


def test_a_slow_world_state_write_never_holds_up_the_cancel(engine, monkeypatch):
    """REVIEW_RULES R4: an orchestrator call can wait 10 s for its service and 10 s for its
    answer. The writes go out on a thread of their own: the loop that follows the run delivers
    a cancel as soon as it comes, whatever a write is waiting on."""
    import threading

    given(monkeypatch, FULL_RUN[:8] + [FULL_RUN[10]], wait=True)  # the grip held: `holding` is written
    ctx = FakeCtx()
    engine.instance_id = "food_container_1"
    real_write = ctx.write_world_state

    def slow_write(**updates):
        time.sleep(2.0)
        return real_write(**updates)

    ctx.write_world_state = slow_write
    sent = []
    real = node.Run.interrupt
    monkeypatch.setattr(node.Run, "interrupt", lambda run: (sent.append(time.monotonic()), real(run)))
    t_cancel = []
    threading.Timer(0.7, lambda: (t_cancel.append(time.monotonic()), setattr(ctx, "cancel", True))).start()
    with pytest.raises(Cancelled):
        engine.execute("open_box", {}, ctx)
    assert len(sent) == 1 and sent[0] - t_cancel[0] < 0.3  # the next poll, not after the write
    assert ctx.world == [{"gripper": {"holding": "food_container_1:lid"}}]  # and the write still went out


def test_the_container_is_opened_as_the_lid_is_let_go_whatever_follows(engine, monkeypatch):
    """REVIEW_RULES C6: `opened` is written as the release row comes, not at a clean exit: a
    run that set the lid down and let it go, then stopped on the way home, leaves the container
    open, and open_box is not offered on it again."""
    stop = "[press_demo] STOP: the way home stopped during home (aborted) — the arm holds"
    given(monkeypatch, FULL_RUN[:-1], [stop], code=1)  # everything but the way home
    ctx = FakeCtx()
    engine.instance_id = "food_container_1"
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert e.value.error_code == "arm_holds"
    assert ctx.objects == {"food_container_1": {"opened": True}}
    world = {"objects": {"food_container_1": {"opened": True}}}
    assert not engine.availability("food_container_1", world)[0].available
    # a run that never let go of the lid on the table (the carry stopped) does not
    given(
        monkeypatch, FULL_RUN[:9], ["[press_demo] STOP: the lid's way to the table stopped — the arm holds the lid"], 1
    )
    ctx = FakeCtx()
    with pytest.raises(EngineError):
        engine.execute("open_box", {}, ctx)
    assert ctx.objects == {}


def test_a_world_state_write_the_orchestrator_refuses_is_said_in_the_result(engine, monkeypatch):
    """REVIEW_RULES O6: a refused write would leave open_box offered on an open container."""
    given(monkeypatch, FULL_RUN)
    ctx = FakeCtx(accepts=False)
    engine.instance_id = "food_container_1"
    out = engine.execute("open_box", {}, ctx)
    assert out["lid_off"] and out["world_state"] == "not updated" and ctx.objects == {}


@pytest.mark.parametrize(
    "params",
    [{"execute": True}, {"execute": "false"}, {"exectue": True}, {"lift_first_m": 0.5}, ["execute"]],
)
def test_bad_params_fail_before_anything_runs(engine, monkeypatch, params):
    """REVIEW_RULES O4: the base class does not enforce params_schema. open_box takes no params:
    the dry run's `execute` is gone, and anything else (a typo, the old home action's lift) is
    refused before anything runs."""
    given(monkeypatch)
    started = []
    monkeypatch.setattr(node, "Run", lambda *a, **k: started.append(a))
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", params, FakeCtx())
    assert e.value.error_code == "bad_params" and not started


def test_a_run_that_stops_with_the_arm_holding_is_not_recoverable(engine, monkeypatch):
    given(monkeypatch, FULL_RUN[:5], ["[press_demo] STOP: the fingers did not open (timeout) — the arm holds"], code=1)
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, FakeCtx())
    assert e.value.error_code == "arm_holds" and not e.value.recoverable
    assert "the fingers did not open" in str(e.value) and "clear it" in str(e.value)


def test_a_run_the_driver_halted_is_arm_halted_and_not_recoverable(engine, monkeypatch):
    """REVIEW_RULES C5: the e-stop (or control lost) ends the run with nothing sent after it,
    press_demo's EXIT_HALTED: arm_halted, never retried."""
    from box_opening.runtime.abort import EXIT_HALTED

    say = ["[press_demo] STOP: HALTED: the driver ended the run's motion (HALTED: emergency stop) — nothing more"]
    given(monkeypatch, FULL_RUN[:5], say, code=EXIT_HALTED)
    ctx = FakeCtx()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert e.value.error_code == "arm_halted" and not ctx.world  # standalone: no world state to tell
    engine.orchestrated = True
    ctx = FakeCtx()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert e.value.error_code == "arm_halted" and not e.value.recoverable
    assert "emergency stop" in str(e.value)
    assert ctx.world == [
        {"system_status": {"arm": {"state": "error", "error_code": "arm_halted", "message": str(e.value)}}}
    ]


@pytest.mark.parametrize(
    ("say", "code", "error"),
    [
        (["[press_demo] not starting: /arm/robot_description not heard in 5 s. Nothing moved."], 1, "not_ready"),
        (["[press_demo] NO BOX — scene camera: no candidate"], 2, "not_found"),
        (["[press_demo] the arm is not at its rest pose"], 3, "not_home"),
        (["[press_demo] STOP: the press MOVED the box — check it is still standing on the table"], 1, "box_moved"),
        (["[press_demo] STOP: the stroke pressed nothing (3 N at its end)"], 1, "not_popped"),
        (["[press_demo] STOP: pressed, but the knob did not pop (2 mm up)"], 1, "not_popped"),
        (["[press_demo] STOP: no button from the look pose (no aim) — nothing was pressed"], 1, "no_button"),
        (["[press_demo] STOP: no reach-out press plans for the scene camera's box (IK); nothing moved"], 1, "no_press"),
        (["[press_demo] STOP: x — the lid is back on the box, and the arm is home"], 1, "lid_put_back"),
        (
            [
                "[press_demo] STOP: the press met something 60 mm above the button's top as the cameras saw it (more "
                "than 45): something else is in the way — the press was stopped there; nothing was pressed"
            ],
            1,
            "contact_early",
        ),
        (
            [
                "[press_demo] STOP: the pads met nothing down to 28 mm below the button's top as the cameras saw it "
                "(more than 25): the box is not where it was seen — the press was stopped there"
            ],
            1,
            "contact_late",
        ),
        (
            [
                "[press_demo] STOP: the press stroke ended with the pads meeting nothing: "
                "the box is not where it was seen"
            ],
            1,
            "contact_late",
        ),
    ],
)
def test_how_a_run_ended_is_its_error_code(engine, monkeypatch, say, code, error):
    """REVIEW_RULES U4: the run's STOP reasons as named codes, recoverable where the run says
    nothing moved, or that the arm got home with nothing held."""
    given(monkeypatch, say=say, code=code)
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, FakeCtx())
    assert e.value.error_code == error and e.value.recoverable


@pytest.mark.parametrize(
    ("say", "code", "error"),
    [
        (
            [
                "[press_demo] STOP: neither camera finds the box after the press — it may have left the table. "
                "The arm holds at the look pose; check the box"
            ],
            1,
            "arm_holds",
        ),
        (
            ["[press_demo] STOP: the arm is home, but the lid slipped out of the fingers on the way to [0.60, -0.07]"],
            1,
            "lid_dropped",
        ),
        (["[press_demo] STOP: something no rule names"], 1, "failed"),
        (
            ["[press_demo] STOP: the press's contact watch failed (ValueError: x) — the press was stopped there"],
            1,
            "watch_failed",
        ),
        (["Traceback (most recent call last):", "ValueError: a bug"], 1, "failed"),
        ([], 1, "failed"),
        (["[press_demo] STOP: the stroke pressed nothing"], -9, "stopped"),
        (["the run said nothing it could be judged by"], -9, "stopped"),
    ],
)
def test_an_unknown_end_is_not_recoverable(engine, monkeypatch, say, code, error):
    """REVIEW_RULES U4: a STOP no rule names, a traceback, a signal: nothing says where the arm
    is, so nothing is retried; the arm's domain is in error."""
    given(monkeypatch, say=say, code=code)
    ctx = FakeCtx()
    engine.orchestrated = True
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert e.value.error_code == error and not e.value.recoverable
    assert ctx.world[-1]["system_status"]["arm"]["error_code"] == error


def test_a_cancel_is_the_runs_own_ctrl_c_and_ends_the_action_cancelled(engine, monkeypatch):
    given(monkeypatch, FULL_RUN[:3], wait=True)
    ctx = FakeCtx()
    sent = []
    real = node.Run.interrupt

    def interrupt(run):
        sent.append(signal.SIGINT)
        real(run)

    monkeypatch.setattr(node.Run, "interrupt", interrupt)
    import threading

    threading.Timer(1.0, setattr, args=(ctx, "cancel", True)).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        engine.execute("open_box", {}, ctx)
    assert sent == [signal.SIGINT]  # once: a second is the escalation, after CANCEL_ESCALATE_S
    assert time.monotonic() - t0 < 5.0


CANCEL_PRESS = {"type": "button", "button_id": "cancel"}


def test_the_panel_says_what_the_run_does_with_a_bar_and_one_button_cancel():
    """REVIEW_RULES U3: valid panel_json that always includes `buttons`, and a cancel path. No Pause is
    offered (a run cannot pause); the bar is the UI's 0-100, the progress the orchestrator gets stays 0-1."""
    ctx, clock = FakeCtx(), [10.0]
    panel = node.Panel(ctx, lambda: True, clock=lambda: clock[0])
    assert panel.tick() is False
    assert ctx.panels == [
        {
            "title": "Opening the container",
            "message": "Starting",
            "progress": {"phase": "locate", "pct": 0.0, "message": "Starting", "pausable": False},
            "buttons": [{"button_id": "cancel", "label": "Cancel"}],
        }
    ]
    panel.say("press", 0.35, "pressing the button")
    panel.tick()  # a change is sent at once
    assert ctx.panels[-1]["message"] == "Pressing the button"
    assert ctx.panels[-1]["progress"] == {
        "phase": "press",
        "pct": 35.0,
        "message": "Pressing the button",
        "pausable": False,
    }
    for sent in ctx.panels:
        assert json.loads(json.dumps(sent)) == sent and [b["button_id"] for b in sent["buttons"]] == ["cancel"]
    # unchanged, it is sent again only as PANEL_REFRESH_S passes (a goal the bridge ended is opened again)
    for _ in range(5):
        panel.tick()
    assert len(ctx.panels) == 2
    clock[0] += node.PANEL_REFRESH_S
    panel.tick()
    assert len(ctx.panels) == 3 and ctx.panels[2] == ctx.panels[1]


def test_cancel_pressed_on_the_panel_is_said_once_and_leaves_no_button():
    ctx, clock = FakeCtx(), [10.0]
    panel = node.Panel(ctx, lambda: True, clock=lambda: clock[0])
    panel.say("press", 0.35, "pressing the button")
    panel.tick()
    ctx.events.append(CANCEL_PRESS)
    assert panel.tick() is True and panel.pressed
    # the press took the bridge's panel down: it is sent again at once, saying the arm is stopping, no button
    assert ctx.panels[-1]["message"] == node.PANEL_CANCELLED and ctx.panels[-1]["buttons"] == []
    assert ctx.panels[-1]["progress"]["pausable"] is False and ctx.panels[-1]["progress"]["pct"] == 35.0
    ctx.events.append(CANCEL_PRESS)  # a second one (a stale panel's): not a second cancel
    assert panel.tick() is False and panel.pressed
    # a press of anything else is no cancel
    other = node.Panel(FakeCtx(), lambda: True, clock=lambda: clock[0])
    other._ctx.events.append({"type": "button", "button_id": "pause"})
    assert other.tick() is False and not other.pressed
    other._ctx.events.append({"type": "object_tapped", "instance_id": "food_container_2"})
    assert other.tick() is False and not other.pressed


def test_the_panel_sends_nothing_without_the_ui_and_tries_a_failed_send_at_its_period():
    ctx, clock, asked = FakeCtx(), [10.0], []
    panel = node.Panel(ctx, lambda: asked.append(1) or False, clock=lambda: clock[0])  # standalone: no token
    for _ in range(40):  # the follow loop's ticks in 2 s
        panel.tick()
        clock[0] += node.POLL_S
    assert ctx.panels == [] and 2 <= len(asked) <= 3  # asked at the period, not at every tick
    ctx.ui_sends = False  # the UI is there, and the bridge's client is not ready
    panel = node.Panel(ctx, lambda: True, clock=lambda: clock[0])
    panel.tick()
    panel.say("look", 0.1, "looking for the button")
    panel.tick()
    ctx.ui_sends = True
    panel.tick()
    assert ctx.panels == []  # not before the period
    clock[0] += node.PANEL_REFRESH_S
    panel.tick()
    assert [p["message"] for p in ctx.panels] == ["Looking for the button"]


def test_a_run_from_the_ui_shows_its_panel_and_cancel_on_it_is_the_runs_own_ctrl_c(engine, monkeypatch):
    """Cancel on the panel ends the run as the orchestrator's cancel does: one Ctrl+C to the run (the motion
    in flight cancelled first), the action cancelled, nothing marked opened."""
    given(monkeypatch, FULL_RUN[:5], wait=True)  # up to the press, then it waits
    ctx = FakeCtx()
    engine.instance_id = "food_container_1"
    monkeypatch.setattr(engine, "_ui_live", lambda: True)
    sent, base = [], []
    real = node.Run.interrupt
    monkeypatch.setattr(node.Run, "interrupt", lambda run: (sent.append(signal.SIGINT), real(run)))
    monkeypatch.setattr(engine, "_cancel_from_ui", lambda: base.append(1))
    import threading

    threading.Timer(1.5, ctx.events.append, args=(CANCEL_PRESS,)).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        engine.execute("open_box", {}, ctx)
    assert sent == [signal.SIGINT] and base == [1] and time.monotonic() - t0 < 6.0
    messages = [p["message"] for p in ctx.panels]
    assert messages[0] == "Starting" and "Pressing the button" in messages
    assert messages[-1] == node.PANEL_CANCELLED and ctx.panels[-1]["buttons"] == []
    running = [p for p in ctx.panels if p["message"] != node.PANEL_CANCELLED]
    assert all([b["button_id"] for b in p["buttons"]] == ["cancel"] for p in running)
    pcts = [p["progress"]["pct"] for p in ctx.panels]
    assert pcts == sorted(pcts) and pcts[-1] == 35.0  # the bar's 0-100, never backwards
    assert ctx.objects == {} and ctx.world == []  # nothing opened, nothing held


def test_the_panels_cancel_is_the_base_classs_own(engine):
    """_cancel_from_ui cancels the goal running in the base class, as a Control.CANCEL does."""
    from capability_engine.engine import _Run

    engine._current = run = _Run("open_box", None)
    try:
        engine._cancel_from_ui()
        assert run.cancelled.is_set()
    finally:
        engine._current = None


def test_a_cancel_between_the_token_and_the_goal_is_kept_for_that_execution(engine, monkeypatch):
    """The orchestrator pushes its token for an execution, then (up to seconds later) sends the goal. A
    CANCEL in between had no run to cancel and was lost: the action ran to its end (uirig, 11-12)."""
    from rammp_orchestrator_interfaces.srv import Control

    given(monkeypatch, FULL_RUN)
    lines = []
    monkeypatch.setattr(LOGGER, "log", lambda level, msg, *a: lines.append(msg))
    cancel = Control.Request(command=Control.Request.CANCEL)
    # no token: no execution is coming, nothing is kept
    assert engine._on_control(cancel, Control.Response()).message == "ok (no action running)"
    engine.execute("open_box", {}, FakeCtx())
    assert sum(ln.startswith("ARGV ") for ln in lines) == 1
    # the token is here, the goal is not: the execution it is for ends cancelled, and nothing runs
    engine._assigned_token = bytes(range(1, 17))
    answer = engine._on_control(cancel, Control.Response())
    assert answer.accepted and "about to start" in answer.message
    ctx = FakeCtx()
    with pytest.raises(Cancelled):
        engine.execute("open_box", {}, ctx)
    assert ctx.said == [] and sum(ln.startswith("ARGV ") for ln in lines) == 1
    # once: the next goal under the same token runs
    engine.execute("open_box", {}, FakeCtx())
    assert sum(ln.startswith("ARGV ") for ln in lines) == 2
    # kept for ITS execution only: another token's runs
    engine._on_control(cancel, Control.Response())
    engine._assigned_token = bytes(range(2, 18))
    engine.execute("open_box", {}, FakeCtx())
    assert sum(ln.startswith("ARGV ") for ln in lines) == 3
    # a pause before the goal is nothing to keep
    engine._on_control(Control.Request(command=Control.Request.PAUSE), Control.Response())
    engine.execute("open_box", {}, FakeCtx())
    assert sum(ln.startswith("ARGV ") for ln in lines) == 4
    engine._assigned_token = bytes(16)


def test_a_run_is_stopped_when_following_it_raises(engine, monkeypatch):
    """REVIEW_RULES R3: an exception while following the run (here, reading its progress) must
    not leave the child running, and moving the arm, after the goal has ended."""
    given(monkeypatch, FULL_RUN[:3], wait=True)
    sent = []
    real = node.Run.interrupt

    runs = []

    def interrupt(run):
        sent.append(signal.SIGINT)
        runs.append(run)
        real(run)

    monkeypatch.setattr(node.Run, "interrupt", interrupt)
    calls = []

    def progress(log, track, ctx, panel):
        calls.append(1)
        if len(calls) == 20:  # ~1 s in: the run is waiting for its Ctrl+C
            raise RuntimeError("a row the tracker cannot read")

    monkeypatch.setattr(node.Capability, "_progress", staticmethod(progress))
    t0 = time.monotonic()
    with pytest.raises(RuntimeError):
        engine.execute("open_box", {}, FakeCtx())
    assert sent == [signal.SIGINT]  # its own stop, not a kill
    assert runs[0].proc.poll() == 130 and engine._run is None  # ended by its Ctrl+C
    assert time.monotonic() - t0 < 5.0  # the stand-in exits on its Ctrl+C at once


def test_a_run_past_its_budget_is_stopped_its_own_way_before_the_orchestrators_cancel(engine, monkeypatch):
    """REVIEW_RULES O3: the orchestrator cancels an action after 120 s (fixed there). A run still
    going at RUN_BUDGET_S is stopped first, by its own Ctrl+C, and fails with `timeout`."""
    # the worst of its own stop (a second Ctrl+C, then a kill) and the last writes, inside the 120 s
    assert node.RUN_BUDGET_S + node.CANCEL_ESCALATE_S + node.CANCEL_KILL_S + node.WORLD_WAIT_S < 120.0
    given(monkeypatch, FULL_RUN[:3], wait=True)
    monkeypatch.setattr(node, "RUN_BUDGET_S", 0.5)
    sent = []
    real = node.Run.interrupt
    monkeypatch.setattr(node.Run, "interrupt", lambda run: (sent.append(signal.SIGINT), real(run)))
    t0 = time.monotonic()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, FakeCtx())
    assert e.value.error_code == "timeout" and e.value.recoverable and sent == [signal.SIGINT]
    assert time.monotonic() - t0 < 5.0


def test_nothing_runs_unless_the_node_allows_it_and_it_is_not_offered(engine, monkeypatch):
    """REVIEW_RULES O1: no dry run. With allow_execute off (the default, and between attended
    runs) open_box is reported unavailable, with why, and a goal that comes anyway is refused
    before anything runs. On, it runs, and always with --execute."""
    given(monkeypatch)
    lines = []
    monkeypatch.setattr(LOGGER, "log", lambda level, msg, *a: lines.append(msg))
    engine.set_parameters([Parameter("allow_execute", value=False)])
    (a,) = engine.availability("c1", {})
    assert not a.available and a.reason.startswith("allow_execute is off")
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, FakeCtx())
    assert e.value.error_code == "not_allowed" and not any(ln.startswith("ARGV ") for ln in lines)
    engine.set_parameters([Parameter("allow_execute", value=True)])
    (a,) = engine.availability("c1", {})
    assert a.available and a.reason == ""
    engine.execute("open_box", {}, FakeCtx())
    argv = next(ln for ln in lines if ln.startswith("ARGV ")).split()
    assert argv[-1] == "--execute"


def test_the_one_action_is_open_box_and_runs_the_whole_run(engine):
    """Review on #23: exactly one action. Pressing, homing and the readiness check are steps of
    open_box's own run (press_demo checks readiness and homes first), not actions of their own."""
    assert [a.action_id for a in engine.actions] == ["open_box"] == list(node.EXECUTABLES) == list(node.MOCK_PHASES)
    real = node.Capability._argv.__get__(engine)
    argv = real("open_box")
    assert argv[0].endswith("lib/box_opening/press_demo") and "--press-only" not in argv
    assert argv[argv.index("--container") + 1].endswith("config/containers/ankou_pink.yaml")
    assert argv[argv.index("--bench-world") + 1].endswith("config/world_flatbot.yaml")
    assert argv[-1] == "--execute" and "--instance-id" not in argv  # standalone: no selection
    # the chair and user the run's plan-free lines are held out of: the planner's own files
    assert argv[argv.index("--body-layers") + 1] == "/config/robot/curobo/planner.yaml"
    engine.set_parameters([Parameter("body_layers", value="")])
    assert real("open_box")[real("open_box").index("--body-layers") + 1] == ""  # off: the bench
    engine.set_parameters([Parameter("body_layers", value="/config/robot/curobo/planner.yaml")])
    argv = real("open_box", "food_container_2")  # the selection as the goal came (execute)
    assert argv[argv.index("--instance-id") + 1] == "food_container_2"


def test_the_wrist_survey_param_asks_the_run_to_survey_and_its_looks_are_the_locate_phase(engine):
    """Off by default (the fragment declares it so): a run whose scene camera finds no box stops with NO BOX, as
    the owner had it. On (a chair parked side-on to a counter), the run surveys with the wrist camera
    (tasks/survey), and its looks report the locate phase."""
    import yaml

    real = node.Capability._argv.__get__(engine)
    assert "--wrist-survey" not in real("open_box") and node.PARAMS["wrist_survey"] is False
    engine.set_parameters([Parameter("wrist_survey", value=True)])
    assert "--wrist-survey" in real("open_box") and "--wrist-survey" in real("open_box", "food_container_2")
    engine.set_parameters([Parameter("wrist_survey", value=False)])
    fragment = yaml.safe_load(open(FRAGMENT_MOCK.with_name("rammp-alternative.yaml")))["params"]
    assert fragment["wrist_survey"] is False and set(node.PARAMS) <= set(fragment)
    # the wrist camera first (a table faced from the front): off by default, its own flag
    assert "--wrist-first" not in real("open_box") and node.PARAMS["wrist_first"] is False
    engine.set_parameters([Parameter("wrist_first", value=True)])
    assert "--wrist-first" in real("open_box")
    engine.set_parameters([Parameter("wrist_first", value=False)])
    assert fragment["wrist_first"] is False
    # the wrist alone for a selection no scene frame holds: the owner's to turn on, off in the code and the fragment
    assert "--wrist-alone" not in real("open_box", "food_container_2") and node.PARAMS["wrist_alone"] is False
    engine.set_parameters([Parameter("wrist_alone", value=True)])
    assert "--wrist-alone" in real("open_box", "food_container_2")
    engine.set_parameters([Parameter("wrist_alone", value=False)])
    assert fragment["wrist_alone"] is False
    track = node.Tracker("open_box")
    said = track.add({"leg": "survey:look-1", "kind": "motion", "ok": True, "outcome": "arrived"})
    assert said == ("locate", 0.03, "looking for the container with the wrist camera")
    assert track.add({"leg": "scene_fix", "kind": "note", "origin_xyz": [0.45, 0.4, 0.24], "source": "wrist"}) == (
        "locate",
        0.05,
        "found the container at [0.45, 0.40]",
    )


def test_each_goal_reads_the_params_as_they_are_then(engine):
    """Review round 2: allow_execute was read per goal, but speed_scale, video and the rest once
    at start-up, so a change from the sheppy TUI was accepted and ignored."""
    real = node.Capability._argv.__get__(engine)
    assert real("open_box")[real("open_box").index("--speed-scale") + 1] == "1.0"
    # off by default (review round 2, R6 ask): the video is for attended data collection
    assert engine._env()["BOX_OPENING_VIDEO"] == "0" and engine._env()["BOX_OPENING_MISSION_FRAMES"] == "0"
    engine.set_parameters(
        [
            Parameter("speed_scale", value=0.5),
            Parameter("video", value=True),
            Parameter("mission_frames", value=True),
            Parameter("container", value="/configs/other.yaml"),
        ]
    )
    argv = real("open_box")
    assert (
        argv[argv.index("--speed-scale") + 1] == "0.5" and argv[argv.index("--container") + 1] == "/configs/other.yaml"
    )
    assert engine._env()["BOX_OPENING_VIDEO"] == "1" and engine._env()["BOX_OPENING_MISSION_FRAMES"] == "1"


def test_a_run_is_told_on_stdout_one_short_line_a_step_beside_its_full_lines(engine, monkeypatch, capsys):
    """The flight view (flight_view.py; the owner 2026-11-12: "easier to read in flight terminal logs so I know
    if and what things are working or not working"): the run's lines, logged as ever through the node's logger
    (stderr), are read as they come and told on the node's stdout one short line a step, with the result — so
    `docker logs -f sheppy-box_opening 2>/dev/null` is the flight view alone. A run that stops says where."""
    p = "[press_demo] "
    given(
        monkeypatch,
        FULL_RUN,
        [
            p + "WRIST SURVEY: box at [0.645, -0.156] yaw 0.0 deg from 2982 lid points (WRIST 0.60) — reaching out",
            p + "REACH-OUT: hand tilted 30 deg; looking from 0.15 m back toward the arm",
            p + "AIMED from the look pose: button at [0.663, -0.154, 0.310], 18 mm from the scene camera's",
            p + "ONE MOTION from the look pose to below the button: reach:re-aim, reach:slide-in, depth:press",
            p + "PRESSED with ~21 N, 2 N sideways",
            p + "GRIP STRAIGHT FROM THE PRESS: up 45 mm, the fingers opening round the knob",
            p + "LID GRIPPED — grip 0.492 vs band [0.3, 0.55]",
            p + "RELEASE as the retreat starts (place:retreat), then HOME, slow, in MEDIUM impedance: home",
            p + "DONE — box open, lid set down at [0.65, 0.05] (clear), arm home",
        ],
    )
    said = []
    monkeypatch.setattr(LOGGER, "log", lambda level, msg, *a: said.append(msg))
    capsys.readouterr()
    engine.instance_id = "food_container_1"
    engine.execute("open_box", {}, FakeCtx())
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("BOX OPENING   ") and out[0].endswith("food_container_1")
    steps = [line[3:12].strip() for line in out if line[:2].strip().isdigit()]
    assert steps == ["FIND", "LOOK", "PRESS", "GRIP", "LIFT", "CARRY", "SET DOWN", "HOME"]
    assert all(" ok" in line for line in out if line[:2].strip().isdigit())
    assert out[-1].startswith("RESULT  DONE — box open, lid on the table, arm home")
    assert p + "PRESSED with ~21 N, 2 N sideways" in said  # ... and every full line still logged, as it came
    # a run that stops: the step it ended at, why, and where things were left
    given(
        monkeypatch,
        say=[
            p + "WRIST SURVEY: box at [0.792, -0.245] yaw 0.0 deg (WRIST 0.60) — reaching out",
            p + "STOP: no press reaches the aimed button (the way down (walk in: 0 deg short)) — nothing was pressed "
            "— returning home",
        ],
        code=1,
    )
    with pytest.raises(EngineError):
        engine.execute("open_box", {}, FakeCtx())
    out = capsys.readouterr().out.splitlines()
    assert " 2 PRESS     FAIL  the box is out of the arm's reach (0.83 m out)" in out
    assert out[-3].startswith("RESULT  FAILED at step 2 (PRESS)")
    assert out[-1] == "        box: not pressed    lid: on the box    arm: home"


def test_the_runs_own_ros_log_lines_keep_their_level(engine, monkeypatch):
    given(monkeypatch, say=["[ERROR] [1793385477.635189323] [box_opening_run]: no arm kinematics", "plain line"])
    said = []
    monkeypatch.setattr(LOGGER, "log", lambda level, msg, *a: said.append((level, msg)))
    engine.execute("open_box", {}, FakeCtx())
    assert (logging.ERROR, "[box_opening_run] no arm kinematics") in said
    assert (logging.INFO, "plain line") in said


def test_a_runs_first_home_is_not_the_way_home_and_no_part_of_the_run_done():
    """A run that does not start at rest homes first (press_demo.home_first), and its rows are home rows too:
    flatbot 2026-11-11 18:06 (clear-margin, home, NO BOX, home) reported 100 % "going home" as it began."""
    track = node.Tracker("open_box")
    assert track.up() == ("locate", 0.0, "looking for the container") and track.up() is None  # said once
    first = [
        {"leg": "clear-margin", "kind": "motion", "ok": True, "outcome": "arrived"},
        {"leg": "still", "kind": "note"},
        {"leg": "settle", "kind": "note"},
        {"leg": "home-waypoint", "kind": "motion", "ok": True, "outcome": "arrived"},
        {"leg": "home", "kind": "motion", "ok": True, "outcome": "arrived"},
    ]
    said = [s for s in map(track.add, first) if s is not None]
    assert said == [("locate", 0.0, "going to the start pose")] * 2
    assert track.add({"leg": "home", "kind": "motion", "ok": False}) == (
        "locate",
        0.0,
        "going to the start pose: stopped",
    )
    # past its start (it looked, and found none): the next home row is the way home
    assert track.add({"leg": "reach_refused", "kind": "note", "why": "NO BOX"})[2].startswith("stopping")
    assert track.add({"leg": "home", "kind": "motion", "ok": True}) == ("home", 1.0, "going home")
    # ... as after any leg of the run itself, with no note read between
    for leg in ("survey:angle-far", "reach:look", "place:open"):
        track = node.Tracker("open_box")
        track.add({"leg": "home", "kind": "motion", "ok": True})
        track.add({"leg": leg, "kind": "motion", "ok": True})
        assert track.add({"leg": "home", "kind": "motion", "ok": True}) == ("home", 1.0, "going home")


def test_the_class_has_a_name_without_a_space_for_a_list_typed_by_hand(engine):
    """interactive_perception_node's `add <instance_id> <perception_class>` is split on spaces: a "food
    container" cannot be typed into it. Both names are claimed, the segmenter's first."""
    classes = list(engine._manifest().perception_classes)
    assert classes == ["food container", "food_container"]
    typed = "add food_container_1 food_container".split()
    assert typed[2] in classes and len(typed) == 3 and not any(" " in c for c in classes[1:])


def test_open_box_is_offered_until_the_world_state_holds_the_container_opened(engine):
    def offered(record):
        world = {"objects": {"c1": record}} if record is not None else {}
        return {a.action_id: a.available for a in engine.availability("c1", world)}

    assert offered({"opened": False}) == offered({"opened": None}) == offered(None) == {"open_box": True}
    assert offered({"opened": True}) == {"open_box": False}


def test_open_box_is_not_offered_while_the_gripper_holds_anything(engine):
    """Must-fix, round 2: a run stopped with the lid in the fingers says so in gripper.holding
    (this engine writes it); the next run would start from wherever the arm stopped, holding it."""
    for holding, offered in ((None, True), ("c1:lid", False), ("spoon_2", False)):
        (a,) = engine.availability("c1", {"gripper": {"holding": holding}, "objects": {"c1": {"opened": False}}})
        assert a.available is offered and (offered or holding in a.reason)


def test_the_mock_holds_every_topic_the_real_run_publishes_and_the_real_node_none(ros):
    """scripts/smoke.py checks the mock container publishes every topic its fragment declares;
    the fragments declare what the real RUN publishes (REVIEW_RULES M3). The mock holds an idle
    publisher on each; the real node holds none — its run does, and press_demo refuses an arm
    another program publishes gripper commands to."""
    import yaml

    from box_opening.constants import GRIPPER_SETPOINT_TOPIC

    declared = set(yaml.safe_load(open(FRAGMENT_MOCK))["publishes"])
    handlers = list(LOGGER.handlers)
    mock = node.Capability(
        parameter_overrides=[Parameter("mock", value=True), Parameter("use_orchestrator", value=False)]
    )
    real = None
    try:
        held = {p.topic_name for p in mock._mock_pubs}
        assert held | {"/box_opening/status"} == declared
        mock.destroy_node()
        mock = None
        real = node.Capability(parameter_overrides=[Parameter("use_orchestrator", value=False)])
        assert real._mock_pubs == []
        assert GRIPPER_SETPOINT_TOPIC not in {p.topic_name for p in real.publishers}
    finally:
        for cap in (mock, real):
            if cap is not None:
                cap.destroy_node()
        LOGGER.handlers[:] = handlers


def test_the_mock_walks_the_runs_phases_and_moves_nothing(ros):
    handlers = list(LOGGER.handlers)
    cap = node.Capability(
        parameter_overrides=[Parameter("mock", value=True), Parameter("use_orchestrator", value=False)]
    )
    try:
        ctx = FakeCtx()
        (a,) = cap.availability("c1", {})  # allow_execute off, as by default: the mock is offered all the same
        assert a.available
        cap.instance_id = "c1"
        out = cap.execute("open_box", {}, ctx)
        assert [p for p, *_ in ctx.said] == [p for p, *_ in node.MOCK_PHASES["open_box"]]
        assert not any(pausable for *_, pausable in ctx.said)  # the real run cannot pause: neither can it
        # review round 2: a successful real run at the interface — its result, and its writes
        assert out["mock"] and out["executed"] and out["lid_off"] and out["lid_xyz"] == node.MOCK_LID_XYZ
        assert (out["presses"], out["grips"], out["world_state"]) == (1, 1, "opened")
        assert ctx.world == [{"gripper": {"holding": "c1:lid"}}, {"gripper": {"holding": None}}]
        assert ctx.objects == {"c1": {"opened": True}}
        world = {"gripper": {"holding": None}, "objects": {"c1": {"opened": True}}}
        assert not cap.availability("c1", world)[0].available  # flipped, as in an all-mock profile
        ctx = FakeCtx()
        ctx.cancel = True
        with pytest.raises(Cancelled):
            cap.execute("open_box", {}, ctx)
        assert ctx.said == [] and ctx.world == []
    finally:
        cap.destroy_node()
        LOGGER.handlers[:] = handlers


def test_the_paced_mock_rehearses_the_ui_its_panel_and_cancel(ros, monkeypatch):
    """mock_phase_s: each of the mock's phases lasts that long, its panel on the UI as the real run's, and
    Cancel on it ends it cancelled with nothing opened. 0 (the default): over at once, as it always was."""
    handlers = list(LOGGER.handlers)
    overrides = {"mock": True, "use_orchestrator": False, "mock_phase_s": 0.3}
    cap = node.Capability(parameter_overrides=[Parameter(k, value=v) for k, v in overrides.items()])
    try:
        monkeypatch.setattr(cap, "_ui_live", lambda: True)
        cap.instance_id = "c1"
        ctx = FakeCtx()
        t0 = time.monotonic()
        out = cap.execute("open_box", {}, ctx)
        phases = node.MOCK_PHASES["open_box"]
        assert len(phases) * 0.3 <= time.monotonic() - t0 < len(phases) * 0.3 + 2.0
        assert out["lid_off"] and ctx.objects == {"c1": {"opened": True}}
        shown = [p["message"] for p in ctx.panels]
        assert [m for m in dict.fromkeys(shown) if m != "Starting"] == [d[:1].upper() + d[1:] for _, _, d in phases]
        assert [p["progress"]["pct"] for p in ctx.panels] == sorted(p["progress"]["pct"] for p in ctx.panels)
        assert ctx.panels[-1]["progress"]["pct"] == 100.0
        assert all([b["button_id"] for b in p["buttons"]] == ["cancel"] for p in ctx.panels)
        # Cancel pressed while it "presses": cancelled, nothing held, nothing opened
        ctx = FakeCtx()
        import threading

        threading.Timer(0.75, ctx.events.append, args=(CANCEL_PRESS,)).start()
        with pytest.raises(Cancelled):
            cap.execute("open_box", {}, ctx)
        assert [p for p, *_ in ctx.said] == ["locate", "look", "press"]
        assert ctx.objects == {} and ctx.world == []
        assert ctx.panels[-1]["message"] == node.PANEL_CANCELLED and ctx.panels[-1]["buttons"] == []
    finally:
        cap.destroy_node()
        LOGGER.handlers[:] = handlers


def _mock_lining_up(monkeypatch, ui=True, **params):
    overrides = dict({"mock": True, "use_orchestrator": False, "line_up": True}, **params)
    cap = node.Capability(parameter_overrides=[Parameter(k, value=v) for k, v in overrides.items()])
    monkeypatch.setattr(cap, "_ui_live", lambda: ui)
    cap.instance_id = "c1"
    return cap


def test_the_mock_lines_up_first_and_runs_only_on_open(ros, monkeypatch):
    """line_up: before anything "moves" the panel tells the user which way to move, asks once the box is in
    reach and still, and the run starts on its Open. The mock's box comes in over 6 of its phases."""
    import threading

    handlers = list(LOGGER.handlers)
    cap = _mock_lining_up(monkeypatch, mock_phase_s=0.2)  # a 1.2 s approach, then a second watched still
    try:
        ctx = FakeCtx()
        threading.Timer(3.2, ctx.events.append, args=({"type": "button", "button_id": "open"},)).start()
        t0 = time.monotonic()
        out = cap.execute("open_box", {}, ctx)
        assert 3.2 <= time.monotonic() - t0 < 3.2 + 7 * 0.2 + 2.0
        assert out["lid_off"] and ctx.objects == {"c1": {"opened": True}}
        lining = [m for p, _pct, m, _pausable in ctx.said if p == "line_up"]
        assert lining[0].startswith("Move closer") and any(m.startswith("Move left") for m in lining)
        assert lining[-3:] == ["Hold still, checking", "Ready. Open the container?", "Opening the container"]
        phases = [p for p, *_ in ctx.said]
        assert phases[len(lining) :] == [p for p, *_ in node.MOCK_PHASES["open_box"]]  # the run, after its Open
        assert all(pct == 0.0 and not pausable for p, pct, _m, pausable in ctx.said if p == "line_up")
        asked = [p for p in ctx.panels if p["message"] == "Ready. Open the container?"]
        assert asked and [b["button_id"] for b in asked[0]["buttons"]] == ["open", "cancel"]
        first_run_panel = next(i for i, p in enumerate(ctx.panels) if p["progress"]["phase"] == "locate")
        assert all(
            p["progress"]["phase"] not in ("looking", "move", "hold", "ready") for p in ctx.panels[first_run_panel:]
        )
    finally:
        cap.destroy_node()
        LOGGER.handlers[:] = handlers


def test_cancel_while_lining_up_is_the_orchestrators_cancel_and_nothing_runs(ros, monkeypatch):
    import threading

    handlers = list(LOGGER.handlers)
    cap = _mock_lining_up(monkeypatch, mock_phase_s=1.0)
    try:
        base = []
        monkeypatch.setattr(cap, "_cancel_from_ui", lambda: base.append(1))
        ctx = FakeCtx()
        threading.Timer(0.6, ctx.events.append, args=(CANCEL_PRESS,)).start()
        with pytest.raises(Cancelled):
            cap.execute("open_box", {}, ctx)
        assert base == [1] and {p for p, *_ in ctx.said} == {"line_up"}
        assert ctx.objects == {} and ctx.world == []
        ctx = FakeCtx()  # the orchestrator's own cancel, mid line-up: the same end, and no press to pass on
        threading.Timer(0.6, setattr, args=(ctx, "cancel", True)).start()
        with pytest.raises(Cancelled):
            cap.execute("open_box", {}, ctx)
        assert base == [1] and ctx.world == []
    finally:
        cap.destroy_node()
        LOGGER.handlers[:] = handlers


def test_a_user_not_in_place_in_time_is_told_to_press_open_again_and_nothing_runs(ros, monkeypatch):
    handlers = list(LOGGER.handlers)
    cap = _mock_lining_up(monkeypatch, mock_phase_s=5.0)  # a 30 s approach: not in place in time
    try:
        monkeypatch.setattr(node, "LINE_UP_S", 0.6)
        monkeypatch.setattr(node.line_up, "LINGER_S", 0.3)
        ctx = FakeCtx()
        with pytest.raises(EngineError) as e:
            cap.execute("open_box", {}, ctx)
        assert e.value.error_code == "not_lined_up" and e.value.recoverable and "nothing moved" in str(e.value)
        assert ctx.panels[-1]["message"] == "Press open again when you are in place"
        assert {p for p, *_ in ctx.said} == {"line_up"} and ctx.objects == {} and ctx.world == []
    finally:
        cap.destroy_node()
        LOGGER.handlers[:] = handlers


def test_the_far_bound_is_the_deployments_and_none_by_default(ros, monkeypatch):
    handlers = list(LOGGER.handlers)
    cap = _mock_lining_up(monkeypatch)
    try:
        assert node.PARAMS["line_up_far_m"] == 0.0 and cap._far_m() is None  # as far as the proof plans
        cap.set_parameters([Parameter("line_up_far_m", value=0.6)])
        assert cap._far_m() == 0.6
        # the mock's box stops 0.70 m out: with the bound at 0.60 it is never asked of, only brought closer
        monkeypatch.setattr(node, "LINE_UP_S", 1.5)
        monkeypatch.setattr(node.line_up, "LINGER_S", 0.2)
        ctx = FakeCtx()
        with pytest.raises(EngineError) as e:
            cap.execute("open_box", {}, ctx)
        assert e.value.error_code == "not_lined_up"
        said = [m for _p, _pct, m, _pausable in ctx.said]
        assert said[0].startswith("Move closer") and "Ready. Open the container?" not in said
    finally:
        cap.destroy_node()
        LOGGER.handlers[:] = handlers


def test_with_line_up_on_a_goal_with_no_ui_is_refused_the_mocks_too(ros, monkeypatch):
    """line_up on: nothing "moves" without the user's Open, so a goal with no UI to ask on is refused, not
    run. Off, the mock runs with no UI as it always has."""
    handlers = list(LOGGER.handlers)
    cap = _mock_lining_up(monkeypatch, ui=False)
    try:
        ctx = FakeCtx()
        with pytest.raises(EngineError) as e:
            cap.execute("open_box", {}, ctx)
        assert e.value.error_code == "line_up_unavailable" and e.value.recoverable
        assert "line_up is on" in str(e.value) and "nothing moved" in str(e.value) and "line_up false" in str(e.value)
        assert ctx.said == [] and ctx.panels == [] and ctx.world == [] and ctx.objects == {}
        cap.set_parameters([Parameter("line_up", value=False)])
        out = cap.execute("open_box", {}, ctx)
        assert out["lid_off"] and [p for p, *_ in ctx.said] == [p for p, *_ in node.MOCK_PHASES["open_box"]]
        assert ctx.panels == []
    finally:
        cap.destroy_node()
        LOGGER.handlers[:] = handlers


@pytest.mark.parametrize("wrist_first", [False, True])
@pytest.mark.parametrize("token", [bytes(16), bytes(range(1, 17))], ids=["no token", "no bridge"])
def test_with_line_up_on_the_arm_never_flies_for_a_goal_with_no_ui(engine, monkeypatch, wrist_first, token):
    """The rule, on the real node: line_up on and no UI to ask on (a goal that came with no token: standalone,
    or sent to the node directly; or one with a token while the UI bridge is not there) starts no child at
    all. It is refused, recoverable, and says how to run without lining up."""
    given(monkeypatch, FULL_RUN)
    engine.set_parameters([Parameter("line_up", value=True), Parameter("wrist_first", value=wrist_first)])
    started, sources = [], []
    real_run = node.Run
    monkeypatch.setattr(node, "Run", lambda *a, **k: (started.append(a), real_run(*a, **k))[1])
    monkeypatch.setattr(engine, "_box_source", lambda *a, **k: sources.append(a))
    engine._assigned_token = token  # no bridge runs here: its request_user_input is not there either way
    try:
        ctx = FakeCtx()
        with pytest.raises(EngineError) as e:
            engine.execute("open_box", {}, ctx)
    finally:
        engine._assigned_token = bytes(16)
    assert e.value.error_code == "line_up_unavailable" and e.value.recoverable
    assert "no UI" in str(e.value) and "nothing moved" in str(e.value)
    assert started == [] and sources == [] and ctx.said == [] and ctx.world == []


def test_a_ui_that_goes_while_lining_up_is_no_open_and_the_run_is_never_told_to_go(engine, monkeypatch):
    """The UI is asked after once, as the goal comes. One that is gone a moment later (the bridge dropped)
    shows no panel and takes no press: the line-up runs out its time, and the run, started for its wrist
    look, is told to stop. It never goes without an Open."""
    given(monkeypatch, FULL_RUN)
    _wrist_lining_up(engine, monkeypatch)
    reads = []
    monkeypatch.setattr(engine, "_ui_live", lambda: (reads.append(1), len(reads) == 1)[1])
    monkeypatch.setattr(node, "LINE_UP_S", 1.0)
    monkeypatch.setattr(node.line_up, "LINGER_S", 0.2)
    runs, told, real_run = [], [], node.Run

    class Recorded(real_run):
        def __init__(self, argv, env, log, talk=False):
            runs.append(self)
            super().__init__(argv, env, log, talk=talk)

        def send(self, msg):
            told.append(msg)
            super().send(msg)

    monkeypatch.setattr(node, "Run", Recorded)
    ctx = FakeCtx()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert e.value.error_code == "not_lined_up"
    (run,) = runs
    assert told == [{"stop": True}] and not any("LINE-UP: go" in ln for ln in run.lines)
    assert ctx.panels == [] and ctx.objects == {}


class InPlace:
    """A real run's line-up source, stood in for: the box seen where it works, proven; closed as it ends."""

    def __init__(self):
        self.closed_at = None

    def look(self, now):
        from box_opening import guide

        return guide.Fix(now, (0.65, 0.0), "scene"), guide.Reach(True, None, True)

    def close(self):
        self.closed_at = time.monotonic()


def test_a_real_run_lines_up_first_and_starts_only_on_open_its_source_closed(engine, monkeypatch):
    """line_up on a real node: the run's command line starts after Open, never before, and what followed the
    box is closed first. The selection is the one the goal came with."""
    import threading

    given(monkeypatch, FULL_RUN)
    engine.set_parameters([Parameter("line_up", value=True)])
    monkeypatch.setattr(engine, "_ui_live", lambda: True)
    source, made, started = InPlace(), [], []
    monkeypatch.setattr(engine, "_box_source", lambda env, instance_id: (made.append(instance_id), source)[1])
    real_argv = engine._argv
    monkeypatch.setattr(engine, "_argv", lambda *a: (started.append((time.monotonic(), a)), real_argv(*a))[1])
    engine.instance_id = "food_container_1"
    ctx = FakeCtx()
    t0 = time.monotonic()
    threading.Timer(2.0, ctx.events.append, args=({"type": "button", "button_id": "open"},)).start()
    threading.Timer(1.0, setattr, args=(engine, "instance_id", "")).start()  # a hibernate mid line-up
    out = engine.execute("open_box", {}, ctx)
    assert out["lid_off"] and made == ["food_container_1"]
    ((run_at, args),) = started
    assert run_at - t0 >= 2.0 and source.closed_at is not None and source.closed_at <= run_at
    assert args == ("open_box", "food_container_1")  # the selection as the goal came, not as it is now
    phases = [p for p, *_ in ctx.said]
    assert phases[0] == "line_up" and "line_up" not in phases[phases.index("locate") :]
    assert ctx.objects == {"food_container_1": {"opened": True}}


def test_a_real_run_cancelled_or_out_of_time_while_lining_up_never_starts(engine, monkeypatch):
    import threading

    given(monkeypatch, FULL_RUN)
    engine.set_parameters([Parameter("line_up", value=True)])
    monkeypatch.setattr(engine, "_ui_live", lambda: True)
    source, started = InPlace(), []
    monkeypatch.setattr(engine, "_box_source", lambda env, instance_id: source)
    monkeypatch.setattr(engine, "_argv", lambda *a: started.append(a) or [])
    monkeypatch.setattr(engine, "_cancel_from_ui", lambda: None)
    ctx = FakeCtx()
    threading.Timer(0.5, ctx.events.append, args=(CANCEL_PRESS,)).start()
    with pytest.raises(Cancelled):
        engine.execute("open_box", {}, ctx)
    assert started == [] and source.closed_at is not None and ctx.world == []
    # out of time, with the watch's last word on why it saw no box in the error
    monkeypatch.setattr(node, "LINE_UP_S", 0.4)
    monkeypatch.setattr(node.line_up, "LINGER_S", 0.2)
    source.why_none = "scene segmenter: saw no food container in 5 frames"
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, FakeCtx())
    assert e.value.error_code == "not_lined_up" and e.value.recoverable and started == []
    assert "saw no food container" in str(e.value)


def test_a_line_up_that_cannot_start_starts_no_run(engine, monkeypatch):
    """line_up asked for and its source not there (its command line missing): no run without its Open."""
    given(monkeypatch, FULL_RUN)
    engine.set_parameters([Parameter("line_up", value=True)])
    monkeypatch.setattr(engine, "_ui_live", lambda: True)
    monkeypatch.setattr(node, "executable", lambda name: "/nonexistent/" + name)
    started = []
    monkeypatch.setattr(engine, "_argv", lambda *a: started.append(a) or [])
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, FakeCtx())
    assert e.value.error_code == "line_up_unavailable" and not e.value.recoverable and started == []
    assert "nothing moved" in str(e.value)


def test_the_line_ups_children_are_the_watch_and_the_probe_on_this_goals_params(engine, monkeypatch):
    monkeypatch.setattr(node, "executable", lambda name: "/lib/" + name)
    made = []

    class Source:
        def __init__(self, watch, probe, env, far_m=None, log=None):
            made.append((watch, probe, env, far_m))

    monkeypatch.setattr(node.line_up, "BoxSource", Source)
    engine.set_parameters([Parameter("line_up_far_m", value=0.8), Parameter("body_layers", value="/cfg/planner.yaml")])
    env = engine._env()
    engine._box_source(env, "food_container_2")
    ((watch, probe, env_, far_m),) = made
    assert watch[0] == "/lib/box_watch" and watch[-2:] == ["--instance-id", "food_container_2"]
    assert watch[1:5] == [
        "--container",
        node.config_path("ankou_pink.yaml", "containers"),
        "--bench-world",
        node.config_path("world_flatbot.yaml"),
    ]
    pink = node.config_path("ankou_pink.yaml", "containers")
    argv = probe("/state/arm.urdf", "arm_mount")
    assert argv[:6] == ["/lib/reach_probe", "--serve", "--urdf", "/state/arm.urdf", "--container", pink]
    assert argv[6:] == ["--root", "arm_mount", "--body-layers", "/cfg/planner.yaml"]
    assert env_ is env and far_m == 0.8
    engine._box_source(env, "")
    assert "--instance-id" not in made[-1][0]


# -- the line-up with the wrist look: the run's own process watches, and goes on at Open ---------------
class Proven:
    """reach_probe, stood in for: whatever the box, its place is inside the boundary and proven."""

    def __init__(self, source):
        from box_opening import guide

        self._look, self._guide = source.look, guide
        source.look = self.look

    def look(self, now):
        fix, _reach = self._look(now)
        return fix, self._guide.Reach(True, None, True)


def _wrist_lining_up(engine, monkeypatch, proven=True):
    """`engine` with line_up and wrist_first on and a UI; its line-up's source the real BoxSource on the run
    (FAKE_RUN --line-up), reach_probe never started. Returns the runs' argv, as they start."""
    engine.set_parameters([Parameter("line_up", value=True), Parameter("wrist_first", value=True)])
    monkeypatch.setattr(engine, "_ui_live", lambda: True)
    real_source, started = engine._box_source, []

    def source(env, instance_id, watch=None):
        made = real_source(env, instance_id, watch=watch)
        started.append(watch)
        return (Proven(made), made)[1] if proven else made

    monkeypatch.setattr(engine, "_box_source", source)
    return started


def test_with_the_wrist_look_the_run_itself_lines_up_and_goes_on_at_open(engine, monkeypatch):
    """line_up with wrist_first: ONE process, the run's, started with --line-up. It says where the box is;
    the panel asks; on Open it is told to go and the same process runs on to the end."""
    import threading

    given(monkeypatch, FULL_RUN)
    started = _wrist_lining_up(engine, monkeypatch)
    runs, real_run = [], node.Run

    class Recorded(real_run):
        def __init__(self, argv, env, log, talk=False):
            runs.append(self)
            self.argv = list(argv)
            super().__init__(argv, env, log, talk=talk)

    monkeypatch.setattr(node, "Run", Recorded)
    engine.instance_id = "food_container_1"
    ctx = FakeCtx()
    threading.Timer(2.5, ctx.events.append, args=({"type": "button", "button_id": "open"},)).start()
    out = engine.execute("open_box", {}, ctx)
    assert out["lid_off"] and ctx.objects == {"food_container_1": {"opened": True}}
    (watch,) = started
    assert isinstance(watch, node.Run)  # the line-up's eyes were the run itself: no second process for the arm
    said = [m for p, _pct, m, _pausable in ctx.said if p == "line_up"]
    assert said[-2:] == ["Ready. Open the container?", "Opening the container"]
    (run,) = runs  # one process from the line-up to the way home
    assert "--line-up" in run.argv and "--wrist-first" in run.argv and watch is run
    assert any("LINE-UP: go" in ln for ln in run.lines)
    assert not any(ln.startswith("@@") for ln in run.lines)  # its lines for the line-up are not its log's
    phases = [p for p, *_ in ctx.said]
    assert phases.index("locate") > phases.index("line_up") and phases[-1] == "home"


def test_a_run_that_has_its_go_is_stopped_when_what_comes_before_following_it_raises(engine, monkeypatch):
    """REVIEW_RULES R3, for a run that lined up: at Open it has its go and flies. Whatever raises between
    that and the loop that follows it (here the first progress report) must not leave it flying with no
    cancel, no budget and no one to stop it."""
    import threading

    given(monkeypatch, FULL_RUN[:3], wait=True)
    _wrist_lining_up(engine, monkeypatch)
    sent, runs = [], []
    real = node.Run.interrupt
    monkeypatch.setattr(node.Run, "interrupt", lambda run: (sent.append(signal.SIGINT), runs.append(run), real(run)))
    ctx = FakeCtx()
    line_up_says = ctx.progress

    def progress(phase, pct, message="", pausable=True):
        if phase != "line_up":
            raise RuntimeError("the goal's feedback could not be published")
        line_up_says(phase, pct, message, pausable)

    ctx.progress = progress
    threading.Timer(2.5, ctx.events.append, args=({"type": "button", "button_id": "open"},)).start()
    with pytest.raises(RuntimeError):
        engine.execute("open_box", {}, ctx)
    assert sent == [signal.SIGINT]  # its own stop, not a kill
    assert runs[0].proc.poll() in (130, -signal.SIGINT) and engine._run is None  # ended by its Ctrl+C


def test_out_of_time_the_run_is_told_so_turns_back_and_ends_and_nothing_was_pressed(engine, monkeypatch):
    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("LINE_UP", "far")
    _wrist_lining_up(engine, monkeypatch, proven=False)
    monkeypatch.setattr(node, "LINE_UP_S", 1.0)
    monkeypatch.setattr(node.line_up, "LINGER_S", 0.2)
    sent = []
    real = node.Run.interrupt
    monkeypatch.setattr(node.Run, "interrupt", lambda run: (sent.append(signal.SIGINT), real(run)))
    ctx = FakeCtx()
    t0 = time.monotonic()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert e.value.error_code == "not_lined_up" and e.value.recoverable and time.monotonic() - t0 < 8.0
    assert sent == []  # told on its stdin, and it ended of its own accord: no Ctrl+C
    assert engine._run is None and ctx.objects == {} and ctx.world == []
    assert {p for p, *_ in ctx.said} == {"line_up"}


@pytest.mark.parametrize("by", ["the orchestrator", "the panel"])
def test_a_cancel_while_the_wrist_turns_back_is_the_runs_own_ctrl_c(engine, monkeypatch, by):
    """Out of time, the run is told to stop and turns its wrist back to rest, beside the user. That turn is a
    motion as any other: the panel says so and keeps its Cancel, and a cancel (the orchestrator's, or that
    button) is the run's Ctrl+C at the next poll, the action cancelled. The run stays the node's to stop (a
    shutdown) until it has ended."""
    import threading

    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("LINE_UP", "far")
    monkeypatch.setenv("TURN_BACK_S", "20")
    _wrist_lining_up(engine, monkeypatch, proven=False)
    monkeypatch.setattr(node, "LINE_UP_S", 1.0)
    monkeypatch.setattr(node.line_up, "LINGER_S", 0.2)
    monkeypatch.setattr(node, "LINE_UP_END_S", 4.0)
    ctx = FakeCtx()
    sent, base, told = [], [], []
    real, real_send = node.Run.interrupt, node.Run.send
    monkeypatch.setattr(
        node.Run, "interrupt", lambda run: (sent.append((time.monotonic(), engine._run is run)), real(run))
    )
    monkeypatch.setattr(engine, "_cancel_from_ui", lambda: base.append(1))
    cancel = {
        "the orchestrator": lambda: setattr(ctx, "cancel", True),
        "the panel": lambda: ctx.events.append(CANCEL_PRESS),
    }

    def send(run, msg):
        if msg.get("stop"):  # the wrist starts its turn back: the cancel comes half a second into it
            told.append(time.monotonic())
            threading.Timer(0.5, cancel[by]).start()
        real_send(run, msg)

    monkeypatch.setattr(node.Run, "send", send)
    with pytest.raises(Cancelled):
        engine.execute("open_box", {}, ctx)
    ((at, still_the_nodes),) = sent  # one Ctrl+C, not the kill of a run that did not end in time
    assert 0.5 <= at - told[0] < 1.0 and still_the_nodes and engine._run is None
    assert base == ([1] if by == "the panel" else [])  # the button's press is made the orchestrator's cancel
    turning = [p for p in ctx.panels if p["message"] == node.PANEL_TURNING_BACK]
    assert turning and all([b["button_id"] for b in p["buttons"]] == ["cancel"] for p in turning)
    assert all(p["progress"]["pausable"] is False for p in turning)
    assert {p for p, *_ in ctx.said} == {"line_up"} and ctx.said[-1][2] == node.PANEL_TURNING_BACK
    assert ctx.objects == {} and ctx.world == []


def test_a_cancel_while_the_wrist_looks_is_the_runs_own_ctrl_c(engine, monkeypatch):
    import threading

    given(monkeypatch, FULL_RUN)
    _wrist_lining_up(engine, monkeypatch)
    monkeypatch.setattr(engine, "_cancel_from_ui", lambda: None)
    sent = []
    real = node.Run.interrupt
    monkeypatch.setattr(node.Run, "interrupt", lambda run: (sent.append(signal.SIGINT), real(run)))
    ctx = FakeCtx()
    threading.Timer(1.0, ctx.events.append, args=(CANCEL_PRESS,)).start()
    with pytest.raises(Cancelled):
        engine.execute("open_box", {}, ctx)
    assert sent == [signal.SIGINT] and engine._run is None and ctx.world == []


def test_a_run_that_ends_of_itself_while_lining_up_fails_with_what_it_said(engine, monkeypatch):
    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("LINE_UP", "dies")
    _wrist_lining_up(engine, monkeypatch, proven=False)
    ctx = FakeCtx()
    t0 = time.monotonic()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert e.value.error_code == "not_ready" and "the arm driver is not up" in str(e.value)
    assert time.monotonic() - t0 < 5.0  # seen at once, not after the line-up's 40 s


def test_without_the_wrist_look_the_scene_watch_lines_up_and_the_run_starts_after(engine, monkeypatch):
    """line_up with wrist_first off: box_watch (the scene camera alone) is the watch, a child of the line-up's
    own, and the run's command line has no --line-up."""
    engine.set_parameters([Parameter("line_up", value=True)])
    monkeypatch.setattr(engine, "_ui_live", lambda: True)
    monkeypatch.setattr(node, "executable", lambda name: "/lib/" + name)
    made = []

    class Source:
        def __init__(self, watch, probe, env, far_m=None, log=None):
            made.append(watch)

    monkeypatch.setattr(node.line_up, "BoxSource", Source)
    engine._box_source(engine._env(), "c1")
    assert made[0][0] == "/lib/box_watch"
    assert "--line-up" not in engine._argv("open_box", "c1")


def test_how_the_free_air_flies_is_the_deployments_and_as_flown_by_default(engine):
    from box_opening.constants import AIR_CONTROL_ENV

    assert node.PARAMS["air_control"] == "impedance" and engine._env()[AIR_CONTROL_ENV] == "impedance"
    engine.set_parameters([Parameter("air_control", value="position")])
    assert engine._env()[AIR_CONTROL_ENV] == "position"


def test_a_runs_budget_is_what_its_line_up_left(engine, monkeypatch):
    """The orchestrator's 120 s cover the line-up and the run: a run started late is stopped its own way
    that much sooner, and the longest run logged still fits after the longest line-up."""
    assert node.LINE_UP_S + node.RUN_LONGEST_S <= node.RUN_BUDGET_S
    given(monkeypatch, FULL_RUN[:3], wait=True)
    engine.set_parameters([Parameter("line_up", value=True)])
    monkeypatch.setattr(engine, "_ui_live", lambda: True)
    monkeypatch.setattr(engine, "_line_up", lambda ctx, source: node.RUN_BUDGET_S - 0.5)
    t0 = time.monotonic()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, FakeCtx())
    assert e.value.error_code == "timeout" and time.monotonic() - t0 < 5.0


def test_the_run_log_is_read_as_it_grows_a_row_at_a_time(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "run-20261030-130000.jsonl").write_text(json.dumps({"leg": "old"}) + "\n")
    log = node.RunLog(runs, node.RunLog.existing(runs))
    assert log.rows() == []  # the run before it is not this one's
    path = runs / "run-20261030-140000.jsonl"
    path.write_text(json.dumps({"leg": "a"}) + "\n" + '{"leg": "b"')  # b not yet finished
    assert [r["leg"] for r in log.rows()] == ["a"]
    with open(path, "a") as f:
        f.write(', "ok": true}\n')
    assert log.rows() == [{"leg": "b", "ok": True}]


def test_the_engine_says_its_lines_through_the_nodes_log(ros):
    handlers = list(LOGGER.handlers)
    cap = node.Capability(
        parameter_overrides=[Parameter("mock", value=True), Parameter("use_orchestrator", value=False)]
    )
    try:
        assert [type(h) for h in LOGGER.handlers] == [node.NodeLogHandler]
    finally:
        cap.destroy_node()
        LOGGER.handlers[:] = handlers
    assert [type(h).__name__ for h in LOGGER.handlers] == ["ConsoleHandler"]


def test_fake_run_script_is_python():
    compile(textwrap.dedent(FAKE_RUN), "fake_run.py", "exec")
