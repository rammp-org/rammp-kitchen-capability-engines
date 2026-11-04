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

    def progress(self, phase, pct, message="", pausable=True):
        self.said.append((phase, pct, message, pausable))

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

    def progress(log, track, ctx_):
        calls.append(1)
        engine.instance_id = "" if len(calls) == 1 else "food_container_2"  # hibernated, then another selected
        real(log, track, ctx_)

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

    def progress(log, track, ctx):
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
    argv = real("open_box", "food_container_2")  # the selection as the goal came (execute)
    assert argv[argv.index("--instance-id") + 1] == "food_container_2"


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


def test_the_runs_own_ros_log_lines_keep_their_level(engine, monkeypatch):
    given(monkeypatch, say=["[ERROR] [1793385477.635189323] [box_opening_run]: no arm kinematics", "plain line"])
    said = []
    monkeypatch.setattr(LOGGER, "log", lambda level, msg, *a: said.append((level, msg)))
    engine.execute("open_box", {}, FakeCtx())
    assert (logging.ERROR, "[box_opening_run] no arm kinematics") in said
    assert (logging.INFO, "plain line") in said


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
    finally:
        cap.destroy_node()
        LOGGER.handlers[:] = handlers


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
