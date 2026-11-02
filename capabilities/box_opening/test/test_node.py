"""The engine node (box_opening.node): a run's rows as progress and result, its last lines as
the error code, a cancel as its own Ctrl+C, and the dry-run gate. The child is a stand-in
script (FAKE_RUN) that writes a run log and exits as the command line would."""

import json
import logging
import signal
import sys
import textwrap
import time

import pytest
import rclpy
from capability_engine.engine import Cancelled, EngineError
from rclpy.parameter import Parameter

from box_opening import LOGGER, node

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
sys.exit(int(os.environ["EXIT"]))
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
    engine.instance_id = "food_container_1"
    track = node.Tracker("open_box")
    said = []
    for row in FULL_RUN:
        track.add(row)
        engine._sync_held(track, ctx)
        said.append(track.held)
    assert ctx.world == [{"gripper": {"holding": "food_container_1:lid"}}, {"gripper": {"holding": None}}]
    assert said.index(True) == 10 and said.index(False, 10) == 13  # reach_grip_result, then place:open
    engine.instance_id = ""  # standalone: nothing to tell
    track = node.Tracker("open_box")
    track.add(FULL_RUN[10])
    engine._sync_held(track, FakeCtx())
    assert track.held and not track.held_said


def test_a_run_that_stops_with_the_lid_in_the_fingers_leaves_it_said(engine, monkeypatch):
    """REVIEW_RULES C7: the run stopped part way with the lid held (the driver stopped the
    carry): the world state already says the gripper holds it (written as the grip held, so a
    cancel needs no write either), and the error says so."""
    stop = "[press_demo] STOP: the lid's way to the table stopped during lid:carry (aborted) — the arm holds the lid"
    given(monkeypatch, FULL_RUN[:8] + [FULL_RUN[10]], [stop], code=1)  # the grip held, then the carry stopped
    ctx = FakeCtx()
    engine.instance_id = "food_container_1"
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert e.value.error_code == "arm_holds" and not e.value.recoverable
    assert str(e.value).endswith("the lid is still in the fingers")
    assert ctx.world == [{"gripper": {"holding": "food_container_1:lid"}}]


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
    assert e.value.error_code == "failed" and e.value.recoverable and "still in the fingers" not in str(e.value)
    assert not ctx.world or ctx.world[-1] == {"gripper": {"holding": None}}  # not left holding


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


@pytest.mark.parametrize(
    ("say", "code", "error"),
    [
        (["[press_demo] not starting: /arm/robot_description not heard in 5 s. Nothing moved."], 1, "not_ready"),
        (["[press_demo] NO BOX — scene camera: no candidate"], 2, "not_found"),
        (["[press_demo] the arm is not at its rest pose"], 3, "not_home"),
        (["[press_demo] STOP: the press MOVED the box — check it is still standing on the table"], 1, "box_moved"),
        (["[press_demo] STOP: the stroke pressed nothing (3 N at its end)"], 1, "failed"),
    ],
)
def test_how_a_run_ended_is_its_error_code(engine, monkeypatch, say, code, error):
    given(monkeypatch, say=say, code=code)
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, FakeCtx())
    assert e.value.error_code == error and e.value.recoverable


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
    engine.instance_id = "food_container_2"
    argv = real("open_box")
    assert argv[argv.index("--instance-id") + 1] == "food_container_2"


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


def test_the_mock_walks_the_runs_phases_and_moves_nothing(ros):
    handlers = list(LOGGER.handlers)
    cap = node.Capability(
        parameter_overrides=[Parameter("mock", value=True), Parameter("use_orchestrator", value=False)]
    )
    try:
        ctx = FakeCtx()
        (a,) = cap.availability("c1", {})  # allow_execute off, as by default: the mock is offered all the same
        assert a.available
        out = cap.execute("open_box", {}, ctx)
        assert [p for p, *_ in ctx.said] == [p for p, _ in node.MOCK_PHASES["open_box"]]
        assert out["mock"] and not out["executed"] and not out["lid_off"]
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
