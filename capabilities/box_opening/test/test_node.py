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
    {"leg": "scene_fix", "kind": "note", "origin_xyz": [0.6018, -0.2302, 0.2091], "source": "owl"},
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
    def __init__(self):
        self.said = []
        self.cancel = False
        self.objects = {}

    def progress(self, phase, pct, message="", pausable=True):
        self.said.append((phase, pct, message, pausable))

    def checkpoint(self):
        if self.cancel:
            raise Cancelled()

    def cancelled(self):
        return self.cancel

    def update_object(self, instance_id, **fields):
        self.objects[instance_id] = fields
        return True


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
    overrides = {"use_orchestrator": False, "owl": False, "allow_execute": True, "state_dir": str(tmp_path)}
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
    out = engine.execute("open", {"execute": True}, ctx)
    phases = [p for p, *_ in ctx.said]
    assert phases[0] == "locate" and phases[-1] == "home"
    order = ["locate", "look", "press", "grip", "carry", "release", "home"]
    assert [p for p in order if p in phases] == order
    assert [pct for _, pct, *_ in ctx.said] == sorted(pct for _, pct, *_ in ctx.said)  # never backwards
    assert not any(pausable for *_, pausable in ctx.said)  # one motion: no pause once started
    assert out["executed"] and out["lid_off"] and out["lid_xyz"] == [0.597, -0.071, 0.209]
    assert (out["presses"], out["grips"], out["video"]) == (1, 1, video)
    assert out["run_log"].endswith("run-20261030-140000.jsonl")
    assert ctx.objects == {"food_container_1": {"lid": "off", "lid_xyz": [0.597, -0.071, 0.209]}}


def test_a_run_that_stops_with_the_arm_holding_is_not_recoverable(engine, monkeypatch):
    given(monkeypatch, FULL_RUN[:5], ["[press_demo] STOP: the fingers did not open (timeout) — the arm holds"], code=1)
    with pytest.raises(EngineError) as e:
        engine.execute("open", {"execute": True}, FakeCtx())
    assert e.value.error_code == "arm_holds" and not e.value.recoverable
    assert "the fingers did not open" in str(e.value) and "`home`" in str(e.value)


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
        engine.execute("open", {"execute": True}, FakeCtx())
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
        engine.execute("open", {"execute": True}, ctx)
    assert sent == [signal.SIGINT]  # once: a second is the escalation, after CANCEL_ESCALATE_S
    assert time.monotonic() - t0 < 5.0


def test_execute_moves_nothing_unless_the_node_allows_it(engine, monkeypatch):
    given(monkeypatch)
    lines = []
    monkeypatch.setattr(LOGGER, "log", lambda level, msg, *a: lines.append(msg))
    engine.cfg["allow_execute"] = False
    out = engine.execute("open", {"execute": True}, FakeCtx())
    argv = next(ln for ln in lines if ln.startswith("ARGV "))
    assert "--execute" not in argv.split() and not out["executed"] and "DRY RUN" in out["note"]
    engine.cfg["allow_execute"] = True
    lines.clear()
    engine.execute("press", {"execute": True}, FakeCtx())
    argv = next(ln for ln in lines if ln.startswith("ARGV ")).split()
    assert "--execute" in argv and "--press-only" in argv


def test_each_action_runs_its_own_command_line(engine):
    real = node.Capability._argv.__get__(engine)
    open_ = real("open", {"execute": True}, True)
    assert open_[0].endswith("lib/box_opening/press_demo") and "--press-only" not in open_
    assert open_[open_.index("--container") + 1].endswith("config/containers/ankou_pink.yaml")
    assert open_[open_.index("--bench-world") + 1].endswith("config/world_flatbot.yaml")
    home = real("home", {"execute": False, "lift_first_m": 0.08}, False)
    assert home[0].endswith("lib/box_opening/home_arm") and "--execute" not in home
    assert home[home.index("--lift-first") + 1] == "0.08"
    check = real("check", {}, False)
    assert check[0].endswith("lib/box_opening/preflight") and "--speed-scale" not in check


def test_the_runs_own_ros_log_lines_keep_their_level(engine, monkeypatch):
    given(monkeypatch, say=["[ERROR] [1793385477.635189323] [box_opening_run]: no arm kinematics", "plain line"])
    said = []
    monkeypatch.setattr(LOGGER, "log", lambda level, msg, *a: said.append((level, msg)))
    engine.execute("check", {}, FakeCtx())
    assert (logging.ERROR, "[box_opening_run] no arm kinematics") in said
    assert (logging.INFO, "plain line") in said


def test_the_lid_off_leaves_only_home_and_check_offered(engine):
    on = {a.action_id: a.available for a in engine.availability("c1", {"objects": {"c1": {"lid": "on"}}})}
    off = {a.action_id: a.available for a in engine.availability("c1", {"objects": {"c1": {"lid": "off"}}})}
    assert on == {"open": True, "press": True, "home": True, "check": True}
    assert off == {"open": False, "press": False, "home": True, "check": True}


def test_the_mock_walks_the_runs_phases_and_moves_nothing(ros):
    handlers = list(LOGGER.handlers)
    cap = node.Capability(
        parameter_overrides=[Parameter("mock", value=True), Parameter("use_orchestrator", value=False)]
    )
    try:
        ctx = FakeCtx()
        out = cap.execute("open", {"execute": True}, ctx)
        assert [p for p, *_ in ctx.said] == [p for p, _ in node.MOCK_PHASES["open"]]
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
