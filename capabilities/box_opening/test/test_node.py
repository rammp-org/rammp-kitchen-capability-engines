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
import select


# what a watch says of having no container, by its mode (lines_up): the wrist finder's words, and, of a find
# nothing ties to the selection, press_demo's own with what it is of (line_up_watch's judge)
NONES = {
    "unseen": {"none": "wrist camera: no lid-sized top on the table in view"},
    "untied": {
        "none": "the selected container fc_9 is in none of the scene camera's detections: nothing says that what "
        "the wrist sees is the one selected, and the run would not take it",
        "kind": "not_chosen",
    },
    "camera": {"none": "wrist camera: no new wrist frame"},
    "two": {"none": "wrist camera: two places pass as this container, [0.61, -0.10] and [0.66, 0.14]: refusing"},
}


def lines_up(mode):
    # press_demo's line-up watch (--line-up before the run; --guide in the middle of it): says where the box
    # is (`mode`: "in place" | "far" | "unseen", its frames show no container | "comes", they show none for
    # COMES_S and then the container in place | "untied", a lid in every frame that nothing ties to the
    # selection | "camera", its camera gives no frame | "two", two lid-sized tops side by side | "dies" |
    # "holds", its
    # wrist's turn to the view stopped part way: 6) until a line on stdin: go and the run goes on, stop (or
    # the pipe closed) and it ends 5, nothing pressed, once its wrist is back at rest (TURN_BACK_S: how long
    # that turn takes; a Ctrl+C stops it there, 130. TURN_BACK: "holds", that turn stopped part way: 6;
    # "halted", the e-stop came during it: 4)
    if mode == "dies":
        sys.stdout.write("[press_demo] not starting: the arm driver is not up\\n")
        sys.stdout.flush()
        sys.exit(1)
    robot = {"urdf": os.environ.get("URDF") or None, "root": "arm_mount"}  # URDF: reach_probe is started on it
    sys.stdout.write("@@ " + json.dumps({"robot": robot}) + "\\n")
    if mode == "off rest":  # the arm does not stand at rest: nothing turns, and the engine is told so
        heard = None
        try:
            while heard is None:
                why = "the arm is not at its rest pose: home it, then press open again"
                sys.stdout.write("@@ " + json.dumps({"none": why}) + "\\n")
                sys.stdout.flush()
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    heard = sys.stdin.readline()
        except KeyboardInterrupt:
            sys.exit(130)
        sys.stdout.write("[press_demo] LINE-UP: ended without a go; nothing was pressed\\n")
        sys.stdout.flush()
        sys.exit(5)
    sys.stdout.write("[press_demo] LINE-UP: the wrist turns to look at the table in front\\n")
    if mode == "holds":
        sys.stdout.write("[press_demo] STOP: the line-up's wrist turn: aborted - the arm holds; nothing was pressed\\n")
        sys.stdout.flush()
        sys.exit(6)
    if not os.environ.get("NO_VIEW"):  # NO_VIEW: stopped before its turn was done, the view never said
        sys.stdout.write("@@ " + json.dumps({"view": {"joints": [0.0] * 7}}) + "\\n")  # its wrist stands at the view
    heard = None
    began = time.monotonic()
    try:
        while heard is None:
            x = 0.65 if mode in ("in place", "comes") else 1.2
            fix = {"xyz": [x, 0.0, 0.315], "table_z": 0.216, "age_s": 0.05, "rough": False, "by": "wrist"}
            none = mode in NONES or (
                mode == "comes" and time.monotonic() - began < float(os.environ.get("COMES_S", "1.5"))
            )
            said = NONES.get(mode, NONES["unseen"]) if none else {"fix": fix}
            sys.stdout.write("@@ " + json.dumps(said) + "\\n")
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
        if os.environ.get("TURN_BACK") == "holds":
            sys.stdout.write("[press_demo] STOP: the line-up ended and the wrist did not turn back - the arm holds\\n")
            sys.stdout.flush()
            sys.exit(6)
        if os.environ.get("TURN_BACK") == "halted":
            sys.stdout.write("[press_demo] STOP: the arm was halted (e-stop) during line-up:rest\\n")
            sys.stdout.flush()
            sys.exit(4)
        sys.stdout.write("[press_demo] LINE-UP: ended without a go; nothing was pressed, the wrist back at rest\\n")
        sys.stdout.flush()
        try:
            time.sleep(float(os.environ.get("END_S", "0")))  # its own end (threads, log) taking this long
        except KeyboardInterrupt:
            sys.exit(130)
        sys.exit(5)
    sys.stdout.write("[press_demo] LINE-UP: go\\n")


if "--line-up" in sys.argv:
    lines_up(os.environ.get("LINE_UP", "in place"))
# press_demo --guide, where no press plans from where the chair stands (GUIDE: how many times it finds none
# before one plans; GUIDE_AFTER_S: how long its look and proof took before each): it says so (no STOP),
# asks the engine to guide, and watches as a line-up does (GUIDE_BOX: where its frames show the box) until
# the engine's go, from which it "finds the box again" (AGAIN_S: how long that takes, nothing written the
# while); a stop, and it ends 5 as a line-up does. GUIDE_ROWS: it writes its run log's rows as press_demo
# does, BEFORE it asks: the box found, the refusal (guided), and with GUIDE_HOME_S the home to rest that
# takes that long (the aimed button's case: reach_press.no_way_in), then its "guide" row.
# GUIDE_WHAT "unseen": what it finds is no container at all (press_demo's NOT SEEN): no box found, no place
# in its request, and by hand the NO BOX and exit 2 it always was
SCENE_NONE = "scene camera: scene segmenter: saw no food container in 14 frames"
NOT_SEEN = (
    "no camera sees the container (" + SCENE_NONE + "; the wrist's look in front: wrist camera: no counter in 4 "
    "frames (no level plane wider than a container))"
)
WHY = (
    "no reach-out press plans for the scene camera's box (30 deg: the way down (the slide in: trajectory sanity "
    "gate: arm_joint_1 excursion 2.012 rad > |d| + margin 1.632); 40 deg: the way down (the press stroke is not a "
    "straight line the arm can fly (the chair and user: arm_half_arm_1_link 0 mm into person/lap (padded 20 mm + 5 "
    "mm), 46% of the way)))"
)
for n in range(int(os.environ.get("GUIDE", "0"))):
    unseen = os.environ.get("GUIDE_WHAT") == "unseen"
    if "--guide" not in sys.argv:  # a run by hand stops there, as it always has
        if unseen:
            sys.stdout.write("[press_demo] NO BOX — " + SCENE_NONE + " — returning home\\n")
            sys.stdout.flush()
            sys.exit(2)
        sys.stdout.write("[press_demo] STOP: " + WHY + "; nothing moved\\n")
        sys.stdout.flush()
        sys.exit(1)
    try:
        time.sleep(float(os.environ.get("GUIDE_AFTER_S", "0")))
    except KeyboardInterrupt:
        sys.exit(130)
    homes = os.environ.get("GUIDE_HOME_S")
    if unseen:
        sys.stdout.write("[press_demo] NOT SEEN: " + NOT_SEEN + "; nothing was pressed\\n")
        if homes is not None:  # (a side survey left the arm over its side: press_demo.guide_chair's own home)
            sys.stdout.write("[press_demo] TO REST: the arm goes to rest before the chair is guided")
            sys.stdout.write(" — returning home\\n")
    else:
        sys.stdout.write("[press_demo] WRIST SURVEY: box at [0.712, -0.331] yaw 0.0 deg (WRIST 0.60) — reaching out\\n")
        ends = "; nothing moved" if homes is None else " — nothing was pressed — returning home"
        sys.stdout.write("[press_demo] NO WAY IN: " + WHY + ends + "\\n")
    sys.stdout.flush()
    if os.environ.get("GUIDE_ROWS"):
        def row(**fields):
            runs = Path(os.environ["BOX_OPENING_STATE"]) / "runs"
            runs.mkdir(parents=True, exist_ok=True)
            with open(runs / "run-20261030-140000.jsonl", "a") as f:
                f.write(json.dumps(fields) + "\\n")
        if unseen:
            row(leg="reach_refused", kind="note", why=NOT_SEEN, guided=True, what="unseen")
        else:
            row(leg="scene_fix", kind="note", origin_xyz=[0.712, -0.331, 0.2])
            row(leg="reach_refused", kind="note", why=WHY, guided=True)
        if homes is not None:
            try:
                time.sleep(float(homes))
            except KeyboardInterrupt:
                sys.exit(130)
            row(leg="home", kind="motion", ok=True, outcome="arrived")
        row(leg="guide", kind="note", n=n + 1)
    # (each time it stands somewhere else: 5 cm nearer the arm than the time before)
    box = [round(0.712 - 0.05 * n, 3), -0.331, 0.2945]
    asked = {"n": n + 1, "box": box, "why": WHY}
    if unseen:
        asked = {"n": n + 1, "box": None, "why": NOT_SEEN, "what": "unseen"}
    sys.stdout.write("@@ " + json.dumps({"guide": asked}) + "\\n")
    sys.stdout.flush()
    lines_up(os.environ.get("GUIDE_BOX", "in place"))
    try:
        time.sleep(float(os.environ.get("AGAIN_S", "0")))
    except KeyboardInterrupt:
        sys.exit(130)
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
        time.sleep(float(os.environ.get("WAIT_S", "30")))
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


@pytest.fixture(autouse=True)
def no_ten_minute_waits(monkeypatch):
    """A goal sent by hand waits ten minutes for its person (node.CONSOLE_LINE_UP_S). No test does: a
    line-up that nobody answers ends in seconds here, and the tests that are about that time set their own."""
    assert node.CONSOLE_LINE_UP_S == 600.0  # the limit as it is flown: stated in docs/interface.md
    monkeypatch.setattr(node, "CONSOLE_LINE_UP_S", 20.0)
    # ... and every test's orchestrator is the one that ends a goal 120 s after it came, whatever interfaces
    # the image under test has (the token in the goal or not: node.goal_brings_token), unless the test says
    # it is the one whose clock restarts at every progress (kept_alive below)
    monkeypatch.setattr(node, "goal_brings_token", lambda: False)


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
        # ... with the exit the run really ends it with (reach_press returns 2 there). Every exit 2 was
        # not_found before its words were read, and no_button could not be reached at all (review of
        # 2026-11-19); a NO BOX that speaks of a button circle a survey did not see is still no container found
        (
            ["[press_demo] STOP: no button from the look pose (0 of 3 aims) — nothing was pressed — returning home"],
            2,
            "no_button",
        ),  # fmt: skip
        (
            [
                "[press_demo] NO BOX — scene camera: none; wrist survey: wrist camera: a lid-sized top at [0.40, "
                "0.40], but no button circle where the view shows it (x) — returning home"
            ],
            2,
            "not_found",
        ),  # fmt: skip
        (["a run that said nothing of how it ended"], 2, "not_found"),
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
            "buttons": [{"button_id": "cancel", "label": "Cancel", "style": "warning"}],
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
    # no token and no goal for longer than CANCEL_EARLY_S: no execution was coming, and the next one runs
    monkeypatch.setattr(node, "CANCEL_EARLY_S", 0.2)
    assert engine._on_control(cancel, Control.Response()).accepted
    time.sleep(0.3)
    engine.execute("open_box", {}, FakeCtx())
    assert sum(ln.startswith("ARGV ") for ln in lines) == 1
    monkeypatch.setattr(node, "CANCEL_EARLY_S", 3.0)
    # the token is here, the goal is not: the execution it is for ends cancelled, and nothing runs
    engine._assigned_token = bytes(range(1, 17))
    answer = engine._on_control(cancel, Control.Response())
    assert answer.accepted and "starts now is cancelled" in answer.message
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


def test_a_cancel_that_overtakes_its_goal_is_kept_where_the_goal_itself_brings_the_token(engine, monkeypatch):
    """Since the engine update (2026-11-16) the execution's token comes in the goal: a CANCEL that arrives
    before the goal finds no token to be kept by, was answered "ok" and dropped, and the goal then ran (with
    line_up off, the whole run, after the user's cancel). It is kept by when it came: a goal that starts
    within CANCEL_EARLY_S of it does not run. Once: the press made after it does."""
    from rammp_orchestrator_interfaces.srv import Control

    given(monkeypatch, FULL_RUN)
    lines = []
    monkeypatch.setattr(LOGGER, "log", lambda level, msg, *a: lines.append(msg))
    assert engine._assigned_token == bytes(16)  # idle: the goal has not come, and nothing else brings a token
    answer = engine._on_control(Control.Request(command=Control.Request.CANCEL), Control.Response())
    assert answer.accepted and "starts now is cancelled" in answer.message
    ctx = FakeCtx()
    with pytest.raises(Cancelled):
        engine.execute("open_box", {}, ctx)  # the goal, a moment after its cancel
    assert ctx.said == [] and not any(ln.startswith("ARGV ") for ln in lines)  # nothing ran
    engine.execute("open_box", {}, FakeCtx())  # the next press runs
    assert sum(ln.startswith("ARGV ") for ln in lines) == 1


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
    # ... and --guide after it: where it finds no press from where the chair stands it asks, and does not end
    assert argv[-2:] == ["--execute", "--guide"]


def test_the_one_action_is_open_box_and_runs_the_whole_run(engine, monkeypatch):
    """Review on #23: exactly one action. Pressing, homing and the readiness check are steps of
    open_box's own run (press_demo checks readiness and homes first), not actions of their own."""
    monkeypatch.delenv("BOX_OPENING_REST_POSE", raising=False)  # (the tests' own: it beats the param)
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
    # the robot's own rest pose: the arm driver's presets file, from the same mount (runtime/rest), and the
    # digest of the numbers this node reads there now (no such file here: the built-in pose's)
    from box_opening import constants
    from box_opening.runtime import rest

    built_in = rest.digest(constants.HOME)
    i = argv.index("--rest-pose")
    assert argv[i : i + 4] == ["--rest-pose", "/config/robot/arm_presets.yaml", "--rest-pose-digest", built_in]
    engine.set_parameters([Parameter("rest_pose", value="")])
    i = real("open_box").index("--rest-pose")
    assert real("open_box")[i : i + 4] == ["--rest-pose", "", "--rest-pose-digest", built_in]  # the built-in pose
    engine.set_parameters([Parameter("rest_pose", value="/config/robot/arm_presets.yaml")])
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
            # the set-down felt the lid meet the table: the one case the run lets go in, and "on the table" is said
            p + "SET-DOWN CONTACT: the lid met the table +12 mm from where the plan put it; the touch stopped there",
            p + "RELEASE as the retreat starts (place:retreat), then HOME, slow, in MEDIUM impedance: home",
            p + "DONE — box open, lid on the table at [0.65, 0.05] (clear), let go where the arm felt it, arm home",
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
    assert out[-2].startswith("RESULT  DONE — box open, lid on the table, arm home")
    assert out[-1].startswith("        full lines: runs/console-")  # where the full lines are kept
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
    assert " 2 PRESS     FAIL  no press plans from here (0.83 m out): the way down does not plan" in out
    assert out[-4].startswith("RESULT  FAILED at step 2 (PRESS)")
    assert out[-2] == "        box: not pressed    lid: on the box    arm: home"


def test_a_goals_full_lines_are_kept_in_a_file_of_its_own_and_the_console_is_the_flight_views(
    engine, monkeypatch, tmp_path, capfd
):
    """Owner 2026-11-17, after three runs watched through `sheppy logs` (stdout and stderr as one): "i changed
    the way my box opening terminal outputs, no? why is it still the old outputs?" A run's full lines go to
    runs/console-<stamp>.log, an error to the console too; the result names the file; and the node's lines are
    the console's again once the goal has ended. With full_console on they are on the console as before."""
    p = "[press_demo] "
    given(monkeypatch, FULL_RUN, [p + "PRESSED with ~21 N, 2 N sideways", p + "DONE — box open, arm home"])
    capfd.readouterr()
    engine.execute("open_box", {}, FakeCtx())
    seen = capfd.readouterr()
    assert "PRESSED with ~21 N" not in seen.out + seen.err  # not on the console, in either stream
    last = seen.out.splitlines()[-1]
    assert last.startswith("        full lines: runs/console-") and last.endswith(".log")
    kept = (tmp_path / last.split(": ")[1]).read_text()
    assert p + "PRESSED with ~21 N, 2 N sideways" in kept and " INFO " in kept
    LOGGER.info("a line of the node's own, between goals")  # the console's again
    assert "between goals" in capfd.readouterr().err and "between goals" not in kept
    # an error is said on the console as well as kept
    given(monkeypatch, say=["[ERROR] [1793385477.635189323] [box_opening_run]: no arm kinematics"])
    engine.execute("open_box", {}, FakeCtx())
    assert "no arm kinematics" in capfd.readouterr().err
    # full_console: as before
    engine.set_parameters([Parameter("full_console", value=True)])
    given(monkeypatch, FULL_RUN, [p + "PRESSED with ~21 N, 2 N sideways", p + "DONE — box open, arm home"])
    engine.execute("open_box", {}, FakeCtx())
    seen = capfd.readouterr()
    assert "PRESSED with ~21 N" in seen.err and "full lines:" not in seen.out


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
        assert e.value.error_code == "not_lined_up" and e.value.recoverable
        # never "nothing moved": with the wrist look the wrist turned to look, and back
        assert "nothing was pressed" in str(e.value) and "nothing moved" not in str(e.value)
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


def test_with_no_ui_the_mock_lines_up_on_the_console_and_runs_only_on_a_y(ros, monkeypatch, capsys):
    """A. The owner, 2026-11-17, of a goal with no UI: "don't end, just tell you how far to move until it's
    close enough then asking if ready y/n." Until then such a goal was refused (line_up_unavailable). The same
    line-up, with no panel: each thing it says is one line of the flight view on the node's console, the
    question is asked there as y / n, and the run starts on a y heard on ~/answer. Nothing "moves" before it."""
    import threading

    handlers = list(LOGGER.handlers)
    cap = _mock_lining_up(monkeypatch, ui=False, mock_phase_s=0.2)  # a 1.2 s approach, then a second watched still
    try:
        ctx = FakeCtx()
        capsys.readouterr()
        threading.Timer(0.4, cap._answers.put, args=("y",)).start()  # typed ahead: no yes
        threading.Timer(3.2, cap._answers.put, args=("y",)).start()
        t0 = time.monotonic()
        out = cap.execute("open_box", {}, ctx)
        assert 3.2 <= time.monotonic() - t0 < 3.2 + 7 * 0.2 + 2.0
        assert out["lid_off"] and ctx.objects == {"c1": {"opened": True}}
        assert ctx.panels == []  # no UI: nothing was sent to one
        lining = [m for p, _pct, m, _pausable in ctx.said if p == "line_up"]
        assert lining[0].startswith("Move closer") and any(m.startswith("Move left") for m in lining)
        assert lining[-3:] == ["Hold still, checking", "Ready. Open the container?", "Opening the container"]
        phases = [p for p, *_ in ctx.said]
        assert phases[len(lining) :] == [p for p, *_ in node.MOCK_PHASES["open_box"]]  # the run, after its y
        view = capsys.readouterr().out.splitlines()
        assert view[0].startswith("BOX OPENING   ") and view[1].startswith("   line up   ...   lining the chair up")
        said = [ln[19:] for ln in view if ln.startswith("             say   ")]
        assert said[0].startswith("Move closer") and "that y was not to a question: answer once it asks" in said
        assert [m for m in said if m.startswith(("Move", "Hold"))] == lining[:-2]  # each line the panel would show
        asked = [ln for ln in view if ln.startswith("             ask   ")]
        assert asked == ["             ask   Ready. Open the container?   y / n"]
        assert view[view.index(asked[0]) + 1] == "             how   " + node.CONSOLE_HOW  # how to answer, once
        assert any(ln.startswith(" 1 LINE UP   ok    yes") for ln in view)
    finally:
        cap.destroy_node()
        LOGGER.handlers[:] = handlers


def test_with_no_ui_an_n_is_cancel_and_nobody_answering_ends_it_recoverable_and_nothing_runs(ros, monkeypatch, capsys):
    """Nothing moves without the person's yes. An n is the cancel at any moment. And a goal nobody answers
    does not wait for ever: after node.CONSOLE_LINE_UP_S (ten minutes as flown; 1.2 s here) it ends
    not_lined_up, recoverable, saying what was still to do."""
    import threading

    handlers = list(LOGGER.handlers)
    cap = _mock_lining_up(monkeypatch, ui=False, mock_phase_s=1.0)  # a 6 s approach
    try:
        base = []
        monkeypatch.setattr(cap, "_cancel_from_ui", lambda by="": base.append(by))
        ctx = FakeCtx()
        threading.Timer(0.6, cap._answers.put, args=("n",)).start()
        with pytest.raises(Cancelled):
            cap.execute("open_box", {}, ctx)
        assert base == ["n answered"] and {p for p, *_ in ctx.said} == {"line_up"}
        assert ctx.objects == {} and ctx.world == []
        view = capsys.readouterr().out
        assert " 1 LINE UP   FAIL  cancelled: n" in view
        # nobody answers
        monkeypatch.setattr(node, "CONSOLE_LINE_UP_S", 1.2)
        monkeypatch.setattr(node.line_up, "LINGER_S", 0.2)
        ctx = FakeCtx()
        with pytest.raises(EngineError) as e:
            cap.execute("open_box", {}, ctx)
        assert e.value.error_code == "not_lined_up" and e.value.recoverable
        assert "within 1 s" in str(e.value) and "last told: Move closer" in str(e.value)
        assert {p for p, *_ in ctx.said} == {"line_up"} and ctx.objects == {} and ctx.world == []
        view = capsys.readouterr().out
        assert (
            "say   " + node.CONSOLE_AGAIN in view and "LINE UP   FAIL  not in place within 1 s (last told: Move" in view
        )
        # line_up off: the mock runs with no UI as it always has, and writes no line-up
        cap.set_parameters([Parameter("line_up", value=False)])
        ctx = FakeCtx()
        out = cap.execute("open_box", {}, ctx)
        assert out["lid_off"] and [p for p, *_ in ctx.said] == [p for p, *_ in node.MOCK_PHASES["open_box"]]
        assert ctx.panels == [] and capsys.readouterr().out == ""
    finally:
        cap.destroy_node()
        LOGGER.handlers[:] = handlers


@pytest.mark.parametrize("wrist_first", [False, True])
@pytest.mark.parametrize("token", [bytes(16), bytes(range(1, 17))], ids=["no token", "no bridge"])
def test_with_line_up_on_the_arm_never_flies_for_a_goal_nobody_says_yes_to(engine, monkeypatch, wrist_first, token):
    """The rule, on the real node: with line_up on the arm never flies without the person's yes. A goal with
    no UI (it came with no token: sent by hand; or with one while the UI bridge is not there) is asked on
    the console, and where no y comes the run is never told to go: with the wrist look it is told to stop
    (its wrist back to rest), without it no run is started at all. It ends not_lined_up, recoverable. A goal
    with the orchestrator's token has what its 120 s leave, as a goal on the UI; one without waits
    CONSOLE_LINE_UP_S."""
    given(monkeypatch, FULL_RUN)
    engine.set_parameters([Parameter("line_up", value=True), Parameter("wrist_first", value=wrist_first)])
    monkeypatch.setattr(engine, "_box_source", lambda env, instance_id, watch=None, refused=(): InPlace())
    monkeypatch.setattr(node, "CONSOLE_LINE_UP_S", 1.5 if token == bytes(16) else 99.0)
    monkeypatch.setattr(node, "LINE_UP_S", 1.5 if token != bytes(16) else 99.0)
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
    engine._assigned_token = token  # no bridge runs here: its request_user_input is not there either way
    try:
        ctx = FakeCtx()
        t0 = time.monotonic()
        with pytest.raises(EngineError) as e:
            engine.execute("open_box", {}, ctx)
    finally:
        engine._assigned_token = bytes(16)
    assert e.value.error_code == "not_lined_up" and e.value.recoverable and time.monotonic() - t0 < 8.0
    assert "nothing was pressed" in str(e.value)
    asked = [m for p, _pct, m, _pausable in ctx.said if p == "line_up"]
    assert "Ready. Open the container?" in asked and "Opening the container" not in asked  # asked, never answered
    assert ctx.panels == [] and ctx.world == [] and ctx.objects == {}
    if wrist_first:
        (run,) = runs
        assert told == [{"stop": True}] and not any("LINE-UP: go" in ln for ln in run.lines)
    else:
        assert runs == [] and told == []


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

    def close(self, quit_s=None):
        self.closed_at = time.monotonic()


def test_a_real_run_lines_up_first_and_starts_only_on_open_its_source_closed(engine, monkeypatch):
    """line_up on a real node: the run's command line starts after Open, never before, and what followed the
    box is closed first. The selection is the one the goal came with."""
    import threading

    given(monkeypatch, FULL_RUN)
    engine.set_parameters([Parameter("line_up", value=True)])
    monkeypatch.setattr(engine, "_ui_live", lambda: True)
    source, made, started = InPlace(), [], []
    monkeypatch.setattr(engine, "_box_source", lambda env, instance_id, **kw: (made.append(instance_id), source)[1])
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
    monkeypatch.setattr(engine, "_box_source", lambda env, instance_id, **kw: source)
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


def test_a_reach_probe_that_died_ends_the_line_up_at_once_with_what_it_said(engine, monkeypatch):
    """line_up.BoxSource.gone / why_gone: the probe ended, so nothing judges the box's place. Not 40 s of
    "Hold still, checking" and then "the container was not where the arm opens it"."""
    given(monkeypatch, FULL_RUN)
    engine.set_parameters([Parameter("line_up", value=True)])
    monkeypatch.setattr(engine, "_ui_live", lambda: True)

    class ProbeDied(InPlace):
        why_gone = "reach_probe, which judges where the arm opens the container, ended (ValueError: no chain)"

        def gone(self):
            return True

    source, started = ProbeDied(), []
    monkeypatch.setattr(engine, "_box_source", lambda env, instance_id, **kw: source)
    monkeypatch.setattr(engine, "_argv", lambda *a: started.append(a) or [])
    t0 = time.monotonic()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, FakeCtx())
    assert e.value.error_code == "line_up_unavailable" and not e.value.recoverable and started == []
    assert "reach_probe" in str(e.value) and "ValueError: no chain" in str(e.value)
    assert time.monotonic() - t0 < 2.0 and source.closed_at is not None


@pytest.mark.parametrize("probe", ["missing", "dies"])
def test_with_the_wrist_look_a_reach_probe_that_cannot_start_or_dies_ends_the_line_up_as_unavailable(
    engine, monkeypatch, tmp_path, probe
):
    """With the wrist look the probe is started inside the line-up's loop, once the run has said which robot
    this is. One that cannot be started (its command line not there) or that dies as it starts ends the
    line-up at once, line_up_unavailable with why: not "exception", and not the line-up's whole time. The
    run is told to stop and never to go."""
    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("URDF", "/state/arm.urdf")
    _wrist_lining_up(engine, monkeypatch, proven=False)
    if probe == "dies":
        (tmp_path / "reach_probe").write_text("#!/bin/sh\necho 'ValueError: no chain from arm_mount' >&2\nexit 1\n")
        (tmp_path / "reach_probe").chmod(0o755)
    monkeypatch.setattr(node, "executable", lambda name: str(tmp_path / name))  # press_demo: the fixture's stand-in
    told, real_send = [], node.Run.send
    monkeypatch.setattr(node.Run, "send", lambda run, msg: (told.append(msg), real_send(run, msg)))
    ctx = FakeCtx()
    t0 = time.monotonic()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert e.value.error_code == "line_up_unavailable" and not e.value.recoverable
    assert time.monotonic() - t0 < 8.0 and told == [{"stop": True}] and engine._run is None
    assert ("No such file" if probe == "missing" else "ValueError: no chain from arm_mount") in str(e.value)
    assert "nothing was pressed" in str(e.value)


def test_the_line_ups_children_are_the_watch_and_the_probe_on_this_goals_params(engine, monkeypatch):
    monkeypatch.setattr(node, "executable", lambda name: "/lib/" + name)
    made = []

    class Source:
        def __init__(self, watch, probe, env, far_m=None, log=None, refused=()):
            made.append((watch, probe, env, far_m))

    monkeypatch.setattr(node.line_up, "BoxSource", Source)
    monkeypatch.delenv("BOX_OPENING_REST_POSE", raising=False)  # (the tests' own: it beats the param)
    engine.set_parameters([Parameter("line_up_far_m", value=0.8), Parameter("body_layers", value="/cfg/planner.yaml")])
    pose = [0.4238, -1.1685, -2.8871, -1.8516, -0.856, -1.0008, 2.2684]
    presets = engine.get_parameter("state_dir").value + "/arm_presets.yaml"
    with open(presets, "w") as f:
        f.write("/**:\n  ros__parameters:\n    presets.home: %r\n" % pose)
    engine.set_parameters([Parameter("rest_pose", value=presets)])
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
    # the deployment's bound goes to the probe too: it asks no cell further out than the line-up trusts
    # ... and the run's own chair model and rest pose: its proofs start where the run does (the file the node
    # resolves and the digest of the seven numbers the node itself reads there: a probe that reads others stops)
    from box_opening.runtime import rest

    assert argv[6:14] == [
        "--root",
        "arm_mount",
        "--body-layers",
        "/cfg/planner.yaml",
        "--rest-pose",
        presets,
        "--rest-pose-digest",
        rest.digest(pose),
    ]
    assert argv[14:] == ["--far-m", "0.8"]
    assert env_ is env and far_m == 0.8
    engine._box_source(env, "")
    assert "--instance-id" not in made[-1][0]
    engine.set_parameters([Parameter("line_up_far_m", value=0.0)])  # no bound: none said
    engine._box_source(env, "")
    assert made[-1][3] is None and "--far-m" not in made[-1][1]("/state/arm.urdf", "arm_mount")


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

    def source(env, instance_id, watch=None, refused=()):
        made = real_source(env, instance_id, watch=watch, refused=refused)
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
    assert "nothing was pressed" in str(e.value) and "nothing moved" not in str(e.value)  # the wrist did turn
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


def test_out_of_time_the_error_says_why_the_run_last_saw_no_container(engine, monkeypatch):
    """The run's own word on why its cameras showed no container ({"none": ...}) is in the not_lined_up
    error: "press open again in place" is not all the user is told where the box was never seen. Nor is it
    what they are told at all (2026-11-19): a container no camera saw was not in the wrong place, it was not
    in view. The same end, the same code, and the words for what was missing, on the panel while the
    line-up went as in the error once it is over."""
    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("LINE_UP", "unseen")
    _wrist_lining_up(engine, monkeypatch, proven=False)
    monkeypatch.setattr(node, "LINE_UP_S", 3.0)
    monkeypatch.setattr(node.line_up, "LINGER_S", 0.2)
    ctx = FakeCtx()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert e.value.error_code == "not_lined_up" and e.value.recoverable
    assert "nothing was pressed" in str(e.value)
    assert str(e.value).endswith("-- wrist camera: no lid-sized top on the table in view")
    assert str(e.value).startswith("no camera saw the container within 3 s: nothing was pressed (move the chair so")
    assert "not where the arm opens it" not in str(e.value)
    said = [m for _p, _pct, m, _pausable in ctx.said]
    assert "The arm cannot see the container. Move the chair in front of it" in said
    assert "Container not seen in time. Press open again in front of it" in said
    assert "Press open again when you are in place" not in said
    unseen = [p for p in ctx.panels if p["message"].startswith("The arm cannot see the container")]
    assert unseen and all([b["button_id"] for b in p["buttons"]] == ["cancel"] for p in unseen)


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


@pytest.mark.parametrize("turn", ["out", "back"])
def test_a_line_up_whose_wrist_turn_stopped_part_way_is_the_arm_holding_and_the_world_state_is_told(
    engine, monkeypatch, turn
):
    """The line-up's own wrist turn stopped part way, on its way to the view or, the line-up over, on its
    way back to rest: the run says so and exits abort.EXIT_LINE_UP_HOLDS. That is the arm holding where it
    stopped, as a run's own stop is: arm_holds, not recoverable, the arm's error in the world state. Never
    "nothing moved, press open again"."""
    given(monkeypatch, FULL_RUN)
    if turn == "out":
        monkeypatch.setenv("LINE_UP", "holds")
    else:
        monkeypatch.setenv("LINE_UP", "far")
        monkeypatch.setenv("TURN_BACK", "holds")
        monkeypatch.setattr(node, "LINE_UP_S", 1.0)
        monkeypatch.setattr(node.line_up, "LINGER_S", 0.2)
    _wrist_lining_up(engine, monkeypatch, proven=False)
    engine.orchestrated = True  # the arm's system_status is the orchestrator's (standalone: nothing to tell)
    ctx = FakeCtx()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert e.value.error_code == "arm_holds" and not e.value.recoverable
    assert "STOP: the line-up" in str(e.value) and "clear it, attended" in str(e.value)
    (status,) = (w["system_status"] for w in ctx.world)
    assert status["arm"]["state"] == "error" and status["arm"]["error_code"] == "arm_holds"
    assert engine._run is None and ctx.objects == {}


def test_a_run_that_had_to_be_stopped_after_its_line_up_is_the_arm_holding_too(engine, monkeypatch):
    """Told its line-up is over, the run did not end within LINE_UP_END_S and got its Ctrl+C: its wrist may
    stand part-turned. arm_holds, not recoverable; not "not lined up, press open again"."""
    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("LINE_UP", "far")
    monkeypatch.setenv("TURN_BACK_S", "20")
    _wrist_lining_up(engine, monkeypatch, proven=False)
    monkeypatch.setattr(node, "LINE_UP_S", 1.0)
    monkeypatch.setattr(node.line_up, "LINGER_S", 0.2)
    monkeypatch.setattr(node, "LINE_UP_END_S", 0.5)
    engine.orchestrated = True
    sent = []
    real = node.Run.interrupt
    monkeypatch.setattr(node.Run, "interrupt", lambda run: (sent.append(signal.SIGINT), real(run)))
    ctx = FakeCtx()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert e.value.error_code == "arm_holds" and not e.value.recoverable and sent == [signal.SIGINT]
    assert "part-turned" in str(e.value) and "clear it, attended" in str(e.value)
    assert [w["system_status"]["arm"]["error_code"] for w in ctx.world] == ["arm_holds"]


def test_an_arm_that_is_not_at_rest_ends_the_line_up_at_once_and_the_panel_says_to_home_it(engine, monkeypatch):
    """A line-up moves nothing but the wrist of an arm at rest. Where the arm stands anywhere else (after a
    cancel or a stop part way through a run) the run says so, and no waiting or driving the chair brings a
    Ready: the user was left "Looking for the container" for the whole line-up, told to press open again, and
    the same again at every press. The panel says to home the arm, and the action ends at once: not_home,
    recoverable. Nothing turned, so nothing is said to turn back."""
    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("LINE_UP", "off rest")
    _wrist_lining_up(engine, monkeypatch, proven=False)
    monkeypatch.setattr(node.line_up, "LINGER_S", 0.3)
    ctx = FakeCtx()
    t0 = time.monotonic()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert e.value.error_code == "not_home" and e.value.recoverable and time.monotonic() - t0 < 8.0
    assert "the arm is not at its rest pose: home it" in str(e.value) and "nothing moved" in str(e.value)
    said = [p for p in ctx.panels if p["message"].startswith("The arm is not at its rest pose")]
    assert said and all(p["buttons"] == [] and p["progress"]["phase"] == "unable" for p in said)
    assert not [p for p in ctx.panels if p["message"] == node.PANEL_TURNING_BACK]  # its wrist never turned
    ending = [p for p in ctx.panels if p["message"] == node.PANEL_LINE_UP_OVER]
    assert all([b["button_id"] for b in p["buttons"]] == ["cancel"] for p in ending)
    assert engine._run is None and ctx.objects == {} and ctx.world == []


def test_a_line_up_that_ends_while_the_wrist_is_still_turning_out_says_the_wrist_turns_back(engine, monkeypatch):
    """A line-up that ends in its first seconds (a reach_probe that died as it started) stops a run whose
    wrist is still on its way to the view: it has not said its view, and it will finish that turn and turn
    back, beside the user. The panel says the wrist is turning back from the run's own word that it turns."""
    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("LINE_UP", "far")
    monkeypatch.setenv("NO_VIEW", "1")
    monkeypatch.setenv("TURN_BACK_S", "1")
    _wrist_lining_up(engine, monkeypatch, proven=False)
    monkeypatch.setattr(node, "LINE_UP_S", 1.0)
    monkeypatch.setattr(node.line_up, "LINGER_S", 0.2)
    ctx = FakeCtx()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert e.value.error_code == "not_lined_up"
    assert [p for p in ctx.panels if p["message"] == node.PANEL_TURNING_BACK]
    assert not [p for p in ctx.panels if p["message"] == node.PANEL_LINE_UP_OVER]


def test_a_run_stopped_late_that_had_its_wrist_back_at_rest_is_not_the_arm_holding(engine, monkeypatch):
    """The stop for a run that outlasts LINE_UP_END_S is for a wrist that may stand part-turned. One that had
    already said its line-up was over, the wrist back at rest, and was only slow to end (its threads, its
    log) is the line-up's own end: not lined up, recoverable, and no arm error in the world state."""
    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("LINE_UP", "far")
    monkeypatch.setenv("END_S", "20")
    _wrist_lining_up(engine, monkeypatch, proven=False)
    monkeypatch.setattr(node, "LINE_UP_S", 1.0)
    monkeypatch.setattr(node.line_up, "LINGER_S", 0.2)
    monkeypatch.setattr(node, "LINE_UP_END_S", 0.5)
    engine.orchestrated = True
    sent = []
    real = node.Run.interrupt
    monkeypatch.setattr(node.Run, "interrupt", lambda run: (sent.append(signal.SIGINT), real(run)))
    ctx = FakeCtx()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert e.value.error_code == "not_lined_up" and e.value.recoverable and sent == [signal.SIGINT]
    assert ctx.world == [] and engine._run is None


def test_the_engine_failing_while_the_wrist_turns_back_stops_the_run(engine, monkeypatch):
    """REVIEW_RULES R3, for the wrist's turn back: it is a motion, beside the user, and the panel and the
    progress the engine reports all through it can raise as any call can. The run must not go on turning
    with the goal ended and nothing following it: its own stop first, and it stays the node's to stop
    (a shutdown) until it has ended."""
    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("LINE_UP", "far")
    monkeypatch.setenv("TURN_BACK_S", "20")
    _wrist_lining_up(engine, monkeypatch, proven=False)
    monkeypatch.setattr(node, "LINE_UP_S", 1.0)
    monkeypatch.setattr(node.line_up, "LINGER_S", 0.2)
    sent, runs = [], []
    real = node.Run.interrupt
    monkeypatch.setattr(
        node.Run, "interrupt", lambda run: (sent.append(engine._run is run), runs.append(run), real(run))
    )
    ctx = FakeCtx()
    line_up_says = ctx.progress

    def progress(phase, pct, message="", pausable=True):
        if message == node.PANEL_TURNING_BACK:
            raise RuntimeError("the goal's feedback could not be published")
        line_up_says(phase, pct, message, pausable)

    ctx.progress = progress
    t0 = time.monotonic()
    with pytest.raises(RuntimeError):
        engine.execute("open_box", {}, ctx)
    assert sent == [True]  # its own stop (one Ctrl+C), sent while it was still the node's to stop
    assert runs[0].proc.poll() in (130, -signal.SIGINT) and engine._run is None
    assert time.monotonic() - t0 < 10.0  # stopped at once, not left to its 20 s turn


@pytest.mark.parametrize("by", ["the orchestrator", "the panel"])
def test_a_cancel_that_comes_while_the_line_up_closes_its_source_is_read_before_the_go(engine, monkeypatch, by):
    """Between the user's Open and the run's go the line-up closes its source: reach_probe is given a moment
    to keep its grid (line_up.PROBE_QUIT_S), the panel already says the container is being opened and has
    its Cancel. A cancel in that moment is acted on before the arm is told to go, not just after: the run is
    never told to go, gets its Ctrl+C as any cancelled line-up does, and the action ends cancelled."""
    import threading

    given(monkeypatch, FULL_RUN[:3], wait=True)
    _wrist_lining_up(engine, monkeypatch)
    ctx = FakeCtx()
    told, sent, base, closes = [], [], [], []
    real_send, real_interrupt, real_close = node.Run.send, node.Run.interrupt, node.line_up.BoxSource.close
    cancel = {
        "the orchestrator": lambda: setattr(ctx, "cancel", True),
        "the panel": lambda: ctx.events.append(CANCEL_PRESS),
    }

    def close(source, **kw):
        closes.append(kw)
        cancel[by]()  # it comes as the source is closed, the Open already taken
        real_close(source, **kw)

    monkeypatch.setattr(node.line_up.BoxSource, "close", close)
    monkeypatch.setattr(node.Run, "send", lambda run, msg: (told.append(msg), real_send(run, msg)))
    monkeypatch.setattr(node.Run, "interrupt", lambda run: (sent.append(signal.SIGINT), real_interrupt(run)))
    monkeypatch.setattr(engine, "_cancel_from_ui", lambda: base.append(1))
    threading.Timer(2.5, ctx.events.append, args=({"type": "button", "button_id": "open"},)).start()
    with pytest.raises(Cancelled):
        engine.execute("open_box", {}, ctx)
    assert {"go": True} not in told and sent == [signal.SIGINT] and engine._run is None
    assert base == ([1] if by == "the panel" else [])  # the button's press is made the orchestrator's cancel
    assert ctx.objects == {} and ctx.world == []
    # (after the Open the source is closed at once, the probe not waited on: the go is what waits)
    assert closes == [{"quit_s": 0.0}]


def test_a_run_halted_while_its_wrist_turned_back_says_so_not_that_nothing_moved(engine, monkeypatch):
    """Whatever the line-up's own end was (here: out of time), a run that then ended any way but "nothing
    pressed, the wrist back at rest" is reported by what it said: the e-stop during the turn back is
    arm_halted, in the world state too."""
    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("LINE_UP", "far")
    monkeypatch.setenv("TURN_BACK", "halted")
    _wrist_lining_up(engine, monkeypatch, proven=False)
    monkeypatch.setattr(node, "LINE_UP_S", 1.0)
    monkeypatch.setattr(node.line_up, "LINGER_S", 0.2)
    engine.orchestrated = True
    ctx = FakeCtx()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert e.value.error_code == "arm_halted" and not e.value.recoverable
    assert [w["system_status"]["arm"]["error_code"] for w in ctx.world] == ["arm_halted"]


def test_the_line_ups_holds_exit_is_the_arm_holding_whatever_the_run_said_last():
    from box_opening.runtime.abort import EXIT_LINE_UP_HOLDS

    for lines in ([], ["[press_demo] LINE-UP: the wrist turns to look"], ["[press_demo] STOP: nothing was pressed"]):
        error = node.error_for("open_box", EXIT_LINE_UP_HOLDS, lines)
        assert error.error_code == "arm_holds" and not error.recoverable and "clear it, attended" in str(error)


def test_a_cancel_while_the_wrist_looks_stops_the_run_before_the_line_ups_source_is_closed(engine, monkeypatch):
    """Closing the line-up's source gives reach_probe a moment to keep its grid (line_up.PROBE_QUIT_S). The
    arm's stop does not wait on that: on a cancel the run gets its Ctrl+C first, the once."""
    import threading

    given(monkeypatch, FULL_RUN)
    _wrist_lining_up(engine, monkeypatch)
    monkeypatch.setattr(engine, "_cancel_from_ui", lambda: None)
    order = []
    real, real_close = node.Run.interrupt, node.line_up.BoxSource.close

    def close(source, **kw):
        order.append("the source closed")
        time.sleep(1.0)  # the probe finishing its proof and writing its grid
        real_close(source, **kw)

    monkeypatch.setattr(node.Run, "interrupt", lambda run: (order.append("Ctrl+C"), real(run)))
    monkeypatch.setattr(node.line_up.BoxSource, "close", close)
    ctx = FakeCtx()
    threading.Timer(1.0, ctx.events.append, args=(CANCEL_PRESS,)).start()
    with pytest.raises(Cancelled):
        engine.execute("open_box", {}, ctx)
    assert order == ["Ctrl+C", "the source closed"] and engine._run is None


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
        def __init__(self, watch, probe, env, far_m=None, log=None, refused=()):
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


class Lines:
    """The node's logger, stood in for: every line it is given, whatever its level."""

    def __init__(self):
        self.said = []

    def _say(self, msg, **_kw):
        self.said.append(msg)

    debug = info = warning = error = fatal = _say


def test_every_goal_says_which_settings_it_flies_on(engine, monkeypatch):
    """A `ros2 param set` lives only in the running node, and a restart puts the deployment's values back:
    the node's log says, at the start of every goal, which of the settings that decide how the arm flies
    were in force for it."""
    given(monkeypatch, FULL_RUN)
    log = Lines()
    monkeypatch.setattr(engine, "get_logger", lambda: log)
    engine.execute("open_box", {}, FakeCtx())
    (line,) = (ln for ln in log.said if ln.startswith("open_box goal:"))
    for setting in ("line_up=false", "ui=none", "line_up_far_m=0.0", "air_control=impedance"):
        assert setting in line
    assert "wrist_first=false" in line and "wrist_alone=false" in line
    log.said.clear()
    engine.set_parameters(
        [
            Parameter("line_up", value=True),
            Parameter("line_up_far_m", value=0.8),
            Parameter("air_control", value="position"),
            Parameter("wrist_first", value=True),
        ]
    )
    monkeypatch.setattr(node, "CONSOLE_LINE_UP_S", 0.5)
    monkeypatch.setattr(node.line_up, "LINGER_S", 0.1)
    with pytest.raises(EngineError):  # line_up on and nobody to say yes here: not lined up, and its line is said
        engine.execute("open_box", {}, FakeCtx())
    (line,) = (ln for ln in log.said if ln.startswith("open_box goal:"))
    for setting in ("line_up=true", "ui=none", "line_up_far_m=0.8", "air_control=position", "wrist_first=true"):
        assert setting in line
    assert "orchestrated=false" in line  # no token came with it: sent by hand, nothing cancels it at 120 s
    log.said.clear()
    monkeypatch.setattr(engine, "_ui_live", lambda: True)
    monkeypatch.setattr(engine, "_lined_up_run", lambda *a: (_ for _ in ()).throw(Cancelled()))
    with pytest.raises(Cancelled):
        engine.execute("open_box", {}, FakeCtx())
    line = next(ln for ln in log.said if ln.startswith("open_box goal:"))
    assert "ui=live" in line and "orchestrated=true" in line


def test_an_air_control_that_is_neither_impedance_nor_position_is_refused_before_anything_runs(engine, monkeypatch):
    """A misspelt value would fly as impedance without a word (the run reads anything but "position" so),
    while whoever set it believes the other."""
    given(monkeypatch, FULL_RUN)
    lines = []
    monkeypatch.setattr(LOGGER, "log", lambda level, msg, *a: lines.append(msg))
    engine.set_parameters([Parameter("air_control", value="postion")])
    ctx = FakeCtx()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert e.value.error_code == "bad_config" and not e.value.recoverable
    assert "'postion'" in str(e.value) and "impedance" in str(e.value) and "position" in str(e.value)
    assert ctx.said == [] and not any(ln.startswith("ARGV ") for ln in lines)  # no run was started
    for value in ("position", "impedance", ""):  # unset is impedance, as flown
        engine.set_parameters([Parameter("air_control", value=value)])
        assert engine.execute("open_box", {}, FakeCtx())["executed"]


def test_the_check_of_the_scene_camera_against_the_wrists_is_the_deployments_and_kept_by_default(engine, monkeypatch):
    """camera_check goes to every process the node starts, in its environment (the run, the line-up's watch:
    whatever builds a scene locator reads it there). A value that is none of the three is refused before
    anything runs: the run would read it as "keep" while whoever set it meant it off."""
    from box_opening.constants import CAMERA_CHECK_ENV
    from box_opening.perception import camera_check

    assert node.PARAMS["camera_check"] == "keep" and node.CAMERA_CHECKS == camera_check.MODES
    assert engine._env()[CAMERA_CHECK_ENV] == "keep"
    for value in ("run", "none"):
        engine.set_parameters([Parameter("camera_check", value=value)])
        assert engine._env()[CAMERA_CHECK_ENV] == value and camera_check.mode_of(engine._env()) == value
    engine.set_parameters([Parameter("camera_check", value="")])  # unset: kept
    assert engine._env()[CAMERA_CHECK_ENV] == "keep"
    given(monkeypatch, FULL_RUN)
    lines = []
    monkeypatch.setattr(LOGGER, "log", lambda level, msg, *a: lines.append(msg))
    engine.set_parameters([Parameter("camera_check", value="of")])
    ctx = FakeCtx()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert e.value.error_code == "bad_config" and not e.value.recoverable
    assert "'of'" in str(e.value) and '"keep", "run" or "none"' in str(e.value)
    assert ctx.said == [] and not any(ln.startswith("ARGV ") for ln in lines)  # no run was started
    engine.set_parameters([Parameter("camera_check", value="none")])
    assert engine.execute("open_box", {}, FakeCtx())["executed"]


def test_a_runs_budget_is_what_its_line_up_left(engine, monkeypatch):
    """The orchestrator's 120 s cover the line-up and the run: a run started late is stopped its own way
    that much sooner, and the longest run logged still fits after the longest line-up."""
    assert node.LINE_UP_S + node.RUN_LONGEST_S <= node.RUN_BUDGET_S
    given(monkeypatch, FULL_RUN[:3], wait=True)
    engine.set_parameters([Parameter("line_up", value=True)])
    monkeypatch.setattr(engine, "_ui_live", lambda: True)
    monkeypatch.setattr(engine, "_line_up", lambda ctx, source, ask: node.RUN_BUDGET_S - 0.5)
    t0 = time.monotonic()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, FakeCtx())
    assert e.value.error_code == "timeout" and time.monotonic() - t0 < 5.0


def test_the_run_is_told_the_time_it_has_as_it_is_started(engine, monkeypatch, tmp_path):
    """A run still going at its budget is stopped with a Ctrl+C wherever it is, and one cut with the lid in
    its fingers is the arm holding it. A lid that feels no table at its spot is taken on to another (owner
    2026-11-19: "it should keep trying spots"), so the run must know how long it has: each run is started
    with its budget in its environment (budget_env) — the seconds this node will give it, the time they
    count from, and whether they start again at each go (a goal sent by hand, whose budget _follow counts
    from its last go) — exactly as _follow counts it. The run's side reads the same name (RunClock)."""
    from box_opening.tasks import reach_lid_off as rlo

    assert node.RUN_BUDGET_ENV == rlo.RUN_BUDGET_ENV  # the engine's name for it and the run's are one
    real, envs = node.Run, []

    class Kept(real):
        def __init__(self, argv, env, log, talk=False):
            envs.append(dict(env))
            super().__init__(argv, env, log, talk=talk)

    monkeypatch.setattr(node, "Run", Kept)

    def run_once():
        _forget_run_logs(tmp_path)
        given(monkeypatch, FULL_RUN)
        t0 = time.monotonic()
        assert engine.execute("open_box", {}, FakeCtx())["lid_off"]
        (env,) = envs
        envs.clear()
        told = json.loads(env[node.RUN_BUDGET_ENV])
        # from the run's start, on the host's monotonic clock — the one Run.started and _follow count by, and
        # the run's RunClock: the wall clock (time.time(), as first written) can step under a run
        assert t0 - 0.01 <= told["from"] <= time.monotonic()
        return told

    # a goal sent by hand: the whole budget, again from each go
    told = run_once()
    assert (told["s"], told["go"]) == (node.RUN_BUDGET_S, True) and node.RUN_BUDGET_S == 90.0
    clock = rlo.RunClock(env={rlo.RUN_BUDGET_ENV: json.dumps(told)}, now=lambda: told["from"] + 30.0)
    assert clock.left_s() == pytest.approx(60.0)  # the run's own reading of it
    assert node.RUN_BUDGET_ENV not in engine._env()  # the run's alone: no other child of the goal is started on it
    # the orchestrator's goal (its token came with it): the budget never moves
    monkeypatch.setattr(engine, "_ui_live", lambda: True)
    told = run_once()
    assert (told["s"], told["go"]) == (node.RUN_BUDGET_S, False)
    # ... and a line-up made before the run was started took its 20 s of it
    engine.set_parameters([Parameter("line_up", value=True)])
    monkeypatch.setattr(engine, "_line_up", lambda ctx, source, ask: 20.0)
    told = run_once()
    assert (told["s"], told["go"]) == (node.RUN_BUDGET_S - 20.0, False)
    # with the wrist look the run is started before its line-up, which is inside its budget
    engine.set_parameters([Parameter("wrist_first", value=True)])
    monkeypatch.setattr(engine, "_run_lines_up", lambda ctx, env, run_, *a, **k: run_.send({"go": True}))
    told = run_once()
    assert (told["s"], told["go"]) == (node.RUN_BUDGET_S, False)
    # ... and by hand its budget starts at the line-up's go: the run reads that go in its own record
    monkeypatch.setattr(engine, "_ui_live", lambda: False)
    told = run_once()
    assert (told["s"], told["go"]) == (node.RUN_BUDGET_S, True)


# -- no press from where the chair stands: the run is guided, never just ended (2026-11-17, run 2) -----------
REASON = (
    "no press plans from here (0.79 m out): the slide in swings the arm wide (30 deg); the press stroke would "
    "pass too close to the user (40 deg)"
)


def _forget_run_logs(tmp_path):
    """FAKE_RUN writes every run's rows to one file name, and the engine follows the log a run CREATES: a
    test of a second goal on one engine starts it with no log there."""
    for f in (tmp_path / "runs").glob("run-*.jsonl"):
        f.unlink()


def _guided(engine, monkeypatch, proven=True):
    """`engine` whose runs may be guided in the middle (FAKE_RUN's GUIDE): the line-up's source the real
    BoxSource on the run, its place proven (reach_probe never started). Returns (what each source was made
    with, every line the run was sent, the runs as they started)."""
    real_source, made, told, runs, real_run = engine._box_source, [], [], [], node.Run

    def source(env, instance_id, watch=None, refused=(), **told_it):
        got = real_source(env, instance_id, watch=watch, refused=refused, **told_it)
        made.append({"watch": watch, "refused": [list(r) for r in refused], **told_it})
        return (Proven(got), got)[1] if proven else got

    class Recorded(real_run):
        def __init__(self, argv, env, log, talk=False):
            runs.append(self)
            self.argv, self.talk = list(argv), talk
            super().__init__(argv, env, log, talk=talk)

        def send(self, msg):
            told.append(msg)
            super().send(msg)

    monkeypatch.setattr(engine, "_box_source", source)
    monkeypatch.setattr(node, "Run", Recorded)
    return made, told, runs


def test_a_box_no_press_plans_for_does_not_end_the_run_the_chair_is_guided_on_the_console(engine, monkeypatch, capsys):
    """B, as the owner flew it on 2026-11-17 (line_up off, no UI, run 2): "the arm didn't do the fix where it
    says to move the arm/chair closer, it should NOT just fail EVER." The run found no press from where the
    chair stood and ended FAILED, "the box is out of the arm's reach (0.79 m out)". Now the SAME process asks
    to be guided, the console says the real reason and then each thing to do, asks y / n, and at the y the
    run is told to go, finds the box again and opens it. One process commands the arm throughout."""
    import threading

    given(
        monkeypatch,
        FULL_RUN,
        [
            # (the run that goes on is a run as any: "lid on the table" is said of a set-down it felt)
            "[press_demo] SET-DOWN CONTACT: the lid met the table +12 mm from where the plan put it",
            "[press_demo] DONE — box open, lid on the table at [0.60, -0.07] (clear), let go where the arm felt "
            "it, arm home",
        ],
    )
    monkeypatch.setenv("GUIDE", "1")
    made, told, runs = _guided(engine, monkeypatch)
    ctx = FakeCtx()
    capsys.readouterr()
    threading.Timer(2.5, engine._answers.put, args=("y",)).start()
    out = engine.execute("open_box", {}, ctx)
    assert out["lid_off"] and out["presses"] == 1
    (run,) = runs  # one process, from the first look to the way home
    assert run.argv[-2:] == ["--execute", "--guide"] and "--line-up" not in run.argv and run.talk
    assert told == [{"go": True}]
    # the run's own word on the place it refused goes to what judges the line-up
    assert made == [{"watch": run, "refused": [[0.712, -0.331, 0.2945]]}]
    said = [m for p, _pct, m, _pausable in ctx.said if p == "line_up"]
    # why, first, for whoever follows the action's progress (the terminal has no console to read it on)
    assert said[0] == "No press plans from here: the slide in swings the arm wide (30 deg); the press stroke " \
        "would pass too close to the user (40 deg). Move the chair as told"  # fmt: skip
    assert said[-2:] == ["Ready. Open the container?", "Opening the container"]
    phases = [p for p, *_ in ctx.said]
    assert phases[-1] == "home" and "line_up" not in phases[phases.index("look") :]  # the run, after its yes
    assert ctx.panels == []
    view = capsys.readouterr().out.splitlines()
    assert " 2 PRESS     FAIL  " + REASON in view  # E: why, not "out of the arm's reach"
    assert "             next  the run does not end here: wait while the arm comes to rest" in view
    assert "   line up   ...   " + node.GUIDE_DOING in view
    assert "             ask   Ready. Open the container?   y / n" in view
    assert any(ln.startswith(" 3 LINE UP   ok    yes") for ln in view)
    assert "   find      ...   looking for the box again" in view
    assert any(ln.startswith("RESULT  DONE — box open, lid on the table, arm home") for ln in view)
    assert not any("out of the arm's reach" in ln for ln in view)


def test_a_run_by_hand_at_the_same_place_would_have_stopped_there(engine, monkeypatch):
    """What the guidance replaces (and FAKE_RUN's own check): the same run without --guide ends at its STOP,
    as run 2 did, with the code it had then."""
    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("GUIDE", "1")
    real_run = node.Run
    monkeypatch.setattr(node, "Run", lambda argv, env, log, talk=False: real_run(argv[:-1], env, log, talk=talk))
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, FakeCtx())
    assert e.value.error_code == "no_press" and "no reach-out press plans" in str(e.value)


def test_with_the_ui_the_guidance_in_the_middle_of_a_run_is_on_its_panel_and_open_is_the_yes(
    engine, monkeypatch, capsys
):
    """line_up off means "guide only when needed": with the UI live the same guidance is the panel's, with
    Open and Cancel, and the run's own panel has the screen again once it goes on."""
    import threading

    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("GUIDE", "1")
    monkeypatch.setattr(engine, "_ui_live", lambda: True)
    _made, told, runs = _guided(engine, monkeypatch)
    ctx = FakeCtx()
    capsys.readouterr()
    threading.Timer(0.3, engine._answers.put, args=("y",)).start()  # (a y is read on the UI's goal too: not asked yet)
    threading.Timer(2.5, ctx.events.append, args=({"type": "button", "button_id": "open"},)).start()
    out = engine.execute("open_box", {}, ctx)
    assert out["lid_off"] and told == [{"go": True}] and len(runs) == 1
    asked = [i for i, p in enumerate(ctx.panels) if p["message"] == "Ready. Open the container?"]
    assert asked and [b["button_id"] for b in ctx.panels[asked[0]]["buttons"]] == ["open", "cancel"]
    after = ctx.panels[asked[-1] + 1 :]
    assert after[0]["message"] == "Opening the container"
    run_phases = {ph for ph, *_ in node.MOCK_PHASES["open_box"]}  # the run's own panel again, Cancel alone on it
    assert after[-1]["progress"]["phase"] in run_phases
    assert after[-1]["buttons"] == [{"button_id": "cancel", "label": "Cancel", "style": "warning"}]
    view = capsys.readouterr().out
    assert "             ask   Ready. Open the container?   (Open on the UI)" in view
    assert "LINE UP   ok    Open pressed" in view and node.CONSOLE_HOW not in view


def _panels_as_sent(ctx):
    """`ctx`, every panel it is sent kept with when: [(seconds, message, button ids)], each change once."""
    t0, seen, real = time.monotonic(), [], ctx.update_user_panel

    def update(panel_json=None, objects=None, selected_instance_id=None):
        now = (panel_json["message"], [b["button_id"] for b in panel_json["buttons"]])
        if not seen or seen[-1][1:] != now:
            seen.append((time.monotonic() - t0, *now))
        return real(panel_json=panel_json)

    ctx.update_user_panel = update
    return seen


def _tells_to_move(text):
    """Whether a line tells the person to move the chair, or that it is to be moved."""
    text = text.lower()
    return "move" in text or "chair is to" in text or "no way in" in text


def test_after_open_on_a_guidance_nothing_tells_the_person_to_move_while_the_run_goes_on(engine, monkeypatch, capsys):
    """The review of 2026-11-17, with the UI: the run writes its refusal's row before it asks to be guided,
    the panel said Ready, the person pressed Open, and the run's own panel came back with its last word
    still on it: "No way in from here: the chair is to move", beside Cancel, for as long as the run took to
    find the box again (2 s of scene look and the wrist's frames, up to 15 s more), while the arm was about
    to reach out. A person who drives on that word moves the chair under it. Once they have said yes,
    nothing tells them to move: the panel, the progress and the console say the run is looking again."""
    import threading

    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("GUIDE", "1")
    monkeypatch.setenv("GUIDE_ROWS", "1")  # its rows as press_demo writes them: the refusal before the request
    monkeypatch.setenv("AGAIN_S", "2.0")  # ... and after the go it finds the box again, no row the while
    monkeypatch.setattr(engine, "_ui_live", lambda: True)
    _made, told, _runs = _guided(engine, monkeypatch)
    ctx = FakeCtx()
    seen = _panels_as_sent(ctx)
    capsys.readouterr()
    threading.Timer(2.5, ctx.events.append, args=({"type": "button", "button_id": "open"},)).start()
    out = engine.execute("open_box", {}, ctx)
    assert out["lid_off"] and told == [{"go": True}]
    said = [message for _t, message, _buttons in seen]
    opened = said.index("Opening the container")
    assert said[opened - 1] == "Ready. Open the container?"
    after = seen[opened + 1 :]
    assert not [m for _t, m, _b in after if _tells_to_move(m)], after
    # what stands while the run finds the box again: that it is looking, with Cancel alone, for the whole of
    # that look (sent, and sent again at its period, until the run's next row)
    assert after[0][1:] == ("Looking for the container again", ["cancel"])
    assert len([p for p in ctx.panels if p["message"] == "Looking for the container again"]) >= 2
    # the progress, which a person at open_box_terminal reads, says the same from the yes on
    progress = [m for _p, _pct, m, _pausable in ctx.said]
    went = progress.index("Opening the container")
    assert progress[went + 1] == "looking for the container again"
    assert not [m for m in progress[went:] if _tells_to_move(m)]
    # ... and so does the console
    view = capsys.readouterr().out.splitlines()
    yes = next(i for i, line in enumerate(view) if "LINE UP   ok" in line)
    assert view[yes + 1] == "   find      ...   looking for the box again"
    assert not [line for line in view[yes:] if _tells_to_move(line)]


def test_a_row_of_before_the_yes_that_is_read_after_it_says_nothing():
    """The run writes its refusal's row (and the row of its home to rest) before it asks to be guided, and
    the engine reads a run's rows as it polls: one read only after the line-up, with the run flying on the
    person's yes, is still of before it. It says nothing then, and the run's next home is not taken for the
    arm going to rest. A row is known by when the run wrote it (its "t", this host's clock)."""
    track = node.Tracker("open_box")
    refused = {"leg": "reach_refused", "kind": "note", "why": "no reach-out press plans (x)", "guided": True}
    home = {"leg": "home", "kind": "motion", "ok": True}
    track.add(dict(FULL_RUN[0], t=1000.0))
    assert track.went(asked_at=1010.0) == ("locate", 0.05, "looking for the container again")
    assert track.add(dict(refused, t=1009.9)) is None and not track.to_rest  # read late
    assert track.add(dict(home, t=1009.95)) is None and track.pct == 0.05  # (nor is that home the way home)
    for row in FULL_RUN[:-1]:
        track.add(dict(row, t=1020.0))
    assert track.add(dict(home, t=1050.0)) == ("home", 1.0, "going home")
    # a refusal written after that yes is a new one, and is said; one with no time is taken as it comes
    track = node.Tracker("open_box")
    track.add(dict(refused, t=1009.9)), track.went(asked_at=1010.0)
    assert track.add(dict(refused, t=1031.0)) == ("locate", 0.0, node.NO_WAY_IN_WAIT) and track.to_rest
    assert track.went(asked_at=1032.0)[2] == "looking for the container again" and not track.to_rest
    assert track.add(refused) == ("locate", 0.0, node.NO_WAY_IN_WAIT)


def test_until_the_arm_is_at_rest_nothing_says_the_chair_is_to_move(engine, monkeypatch, capsys):
    """The review of 2026-11-17: where no press reaches the aimed button the run notes its refusal and THEN
    homes from the look pose (about 5 s on run 1). The engine reads the note within 50 ms, and the panel
    said "No way in from here: the chair is to move" for the whole home flight, then "Going to rest: the
    chair is to move"; the progress the same; the console `next  the chair is to move` before `home ...`.
    A person who starts to drive on that word moves the chair with the arm still out over the table. The
    first word that the chair is to move is the line-up's, and it comes only once the run has asked: its
    arm at rest."""
    import threading

    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("GUIDE", "1")
    monkeypatch.setenv("GUIDE_ROWS", "1")
    monkeypatch.setenv("GUIDE_HOME_S", "1.5")  # the home to rest, flown before it asks to be guided
    monkeypatch.setattr(engine, "_ui_live", lambda: True)
    _made, told, _runs = _guided(engine, monkeypatch)
    ctx = FakeCtx()
    seen = _panels_as_sent(ctx)
    capsys.readouterr()
    threading.Timer(4.5, ctx.events.append, args=({"type": "button", "button_id": "open"},)).start()
    out = engine.execute("open_box", {}, ctx)
    assert out["lid_off"] and told == [{"go": True}]
    # the progress: everything before the line-up's first word, in the order it was said
    phases = [p for p, *_ in ctx.said]
    before = [m for _p, _pct, m, _pausable in ctx.said[: phases.index("line_up")]]
    assert before[-2:] == ["no press plans from here: wait, the arm comes to rest first", "going to rest"]
    assert not [m for m in before if _tells_to_move(m)], before
    # ... and the first word of the line-up is the reason and that the chair is to move, the arm at rest
    assert ctx.said[phases.index("line_up")][2].endswith(". Move the chair as told")
    # the panel: for the whole home flight it says to wait
    said = [message for _t, message, _buttons in seen]
    wait = said.index("No press plans from here: wait, the arm comes to rest first")
    assert seen[wait + 1][0] - seen[wait][0] >= 1.2 and said[wait + 1] == "Going to rest"
    asked = said.index("Hold still, checking")  # the line-up's own first panel (its box stands in place)
    assert asked > wait + 1 and not [m for m in said[:asked] if _tells_to_move(m)]
    # the console: no word of moving before the line-up's own step, which follows the home's end
    view = capsys.readouterr().out.splitlines()
    step = next(i for i, line in enumerate(view) if line.startswith("   line up   ..."))
    assert view[step - 3 : step] == [
        "             next  the run does not end here: wait while the arm comes to rest",
        "   home      ...   going to rest before the chair is guided",
        next(line for line in view if " HOME      ok" in line),
    ]
    assert not [line for line in view[:step] if _tells_to_move(line)]


def test_a_line_ups_instruction_is_on_the_console_once_and_in_the_goals_file(engine, monkeypatch, capsys, tmp_path):
    """The review of 2026-11-17: every line-up line stood on the node's console twice, this node's own
    "[INFO] ... line-up: Move closer, about 25 cm" and under it the flight view's "say   Move closer, about
    25 cm". A goal's full lines go to a file of their own since that day (NodeLogHandler): where the flight
    view says a line-up's line, the node's own copy of it is that file's alone."""
    import threading

    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("GUIDE", "1")
    _guided(engine, monkeypatch)
    log = Lines()
    monkeypatch.setattr(engine, "get_logger", lambda: log)
    capsys.readouterr()
    threading.Timer(0.5, engine._answers.put, args=("y",)).start()  # typed ahead: no yes, and said once too
    threading.Timer(2.5, engine._answers.put, args=("y",)).start()
    out = engine.execute("open_box", {}, FakeCtx())
    assert out["lid_off"]
    view = capsys.readouterr().out.splitlines()
    assert len([line for line in view if line == "             ask   Ready. Open the container?   y / n"]) == 1
    assert len([line for line in view if "that y was not to a question" in line]) == 1
    # this node's own log (its console) has none of them, nor the reason the flight view's PRESS line gives
    mine = [line for line in log.said if "line-up:" in line or "a y came with nothing asked" in line]
    assert mine == [] and not [line for line in log.said if "the run is not ended" in line]
    assert [line for line in log.said if line.startswith("open_box goal:")]  # (its own few lines are its own)
    # ... and the goal's file has them all, with the run's full lines
    (kept,) = (tmp_path / "runs").glob("console-*.log")
    kept = kept.read_text()
    for line in (
        "INFO    line-up: Ready. Open the container?",
        "INFO    line-up: Opening the container",
        "INFO    line-up: the user's yes: the run goes on",
        "WARNING a y came with nothing asked",
        "WARNING no press plans from where the chair stands (no press plans from here: the slide in swings",
        "[press_demo] NO WAY IN: no reach-out press plans",
    ):
        assert line in kept, line


def test_under_the_orchestrators_120_s_a_guidance_that_does_not_fit_ends_recoverable_and_the_next_goal_lines_up_first(
    engine, monkeypatch, capsys, tmp_path
):
    """C. The orchestrator cancels any action at 120 s, and a run that goes on after a line-up needs its 55 s:
    a line-up may end no later than LINE_UP_S after the goal began, wherever in the run it comes. Where the
    look and the proof have left less than GUIDE_MIN_S of that, nothing is guided: the run is told to stop
    (its wrist back to rest), the action ends not_lined_up, recoverable, saying why no press planned, and
    "press open again" goes on from there: the next open_box lines up FIRST, with its whole time, though
    `line_up` is off. The one after that is as the deployment has it."""
    import threading

    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("GUIDE", "1")
    monkeypatch.setenv("GUIDE_AFTER_S", "1.0")  # its look and proof
    monkeypatch.setattr(engine, "_ui_live", lambda: True)
    monkeypatch.setattr(node, "LINE_UP_S", 5.0)  # of which 4 are left: under GUIDE_MIN_S
    assert node.GUIDE_MIN_S == 8.0 and node.LINE_UP_S + node.RUN_LONGEST_S <= node.RUN_BUDGET_S
    made, told, runs = _guided(engine, monkeypatch)
    sent = []
    real = node.Run.interrupt
    monkeypatch.setattr(node.Run, "interrupt", lambda run: (sent.append(signal.SIGINT), real(run)))
    ctx = FakeCtx()
    capsys.readouterr()
    t0 = time.monotonic()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert e.value.error_code == "not_lined_up" and e.value.recoverable and time.monotonic() - t0 < 4.0
    assert "no time left to guide the chair" in str(e.value) and "the slide in swings the arm wide" in str(e.value)
    assert "Press open again: it lines up first" in str(e.value) and "nothing was pressed" in str(e.value)
    assert told == [{"stop": True}] and sent == [] and engine._run is None  # told, and ended of itself: no Ctrl+C
    assert runs[0].proc.returncode == 5  # press_demo's EXIT_NOT_LINED_UP: the wrist back at rest
    assert ctx.world == [] and ctx.objects == {}
    view = capsys.readouterr().out
    assert "LINE UP   FAIL  no time left to guide the chair" in view
    assert "        box: not pressed    lid: on the box    arm: at rest" in view
    # press open again: this goal lines up first, though line_up is off, and goes on at Open
    monkeypatch.delenv("GUIDE")
    engine.set_parameters([Parameter("wrist_first", value=True)])
    monkeypatch.setattr(node, "LINE_UP_S", 35.0)
    ctx = FakeCtx()
    threading.Timer(2.0, ctx.events.append, args=({"type": "button", "button_id": "open"},)).start()
    out = engine.execute("open_box", {}, ctx)
    assert out["lid_off"] and "--line-up" in runs[1].argv and [p for p, *_ in ctx.said][0] == "line_up"
    # the place the run before it refused is held against the carried line-up too: as first written it was
    # built with none, and could ask "Ready?" on the very spot (review of 2026-11-17)
    assert made == [{"watch": runs[1], "refused": [[0.712, -0.331, 0.2945]]}]
    # ... once: the goal after it starts its run at once, and holds nothing against any place
    _forget_run_logs(tmp_path)
    ctx = FakeCtx()
    out = engine.execute("open_box", {}, ctx)
    assert out["lid_off"] and "--line-up" not in runs[2].argv and "line_up" not in [p for p, *_ in ctx.said]
    assert len(made) == 1 and engine._line_up_first is None


def test_a_guidance_that_runs_out_of_time_says_what_was_still_to_do_and_turns_the_wrist_back(
    engine, monkeypatch, capsys
):
    """The user did not get the box into reach in what the 120 s left: not_lined_up, recoverable, with the
    instruction they were last given and why no press had planned, the wrist turned back to rest by the run
    itself (told on its stdin: no Ctrl+C), and the next open_box lines up first."""
    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("GUIDE", "1")
    monkeypatch.setenv("GUIDE_BOX", "far")  # 1.2 m out, and it stays there
    engine.set_parameters([Parameter("line_up_far_m", value=0.8), Parameter("wrist_first", value=True)])
    monkeypatch.setattr(engine, "_ui_live", lambda: True)
    monkeypatch.setattr(node, "LINE_UP_S", 2.5)
    monkeypatch.setattr(node, "GUIDE_MIN_S", 0.5)
    monkeypatch.setattr(node.line_up, "LINGER_S", 0.2)
    _made, told, _runs = _guided(engine, monkeypatch, proven=False)
    sent = []
    real = node.Run.interrupt
    monkeypatch.setattr(node.Run, "interrupt", lambda run: (sent.append(signal.SIGINT), real(run)))
    ctx = FakeCtx()
    capsys.readouterr()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert e.value.error_code == "not_lined_up" and e.value.recoverable
    assert "last told: Move closer, about 40 cm" in str(e.value)
    assert "no press planned from where the chair stood: no press plans from here: the slide in swings" in str(e.value)
    assert told == [{"stop": True}] and sent == [] and engine._run is None
    assert engine._line_up_first is not None  # the next open_box lines up first
    assert ctx.panels[-1]["message"] in ("The wrist is turning back to rest", "Press open again when you are in place")
    view = capsys.readouterr().out
    assert "             say   Move closer, about 40 cm" in view
    assert "             say   The wrist is turning back to rest" in view
    assert "        box: not pressed    lid: on the box    arm: at rest, the wrist turned back" in view


def test_a_goal_sent_by_hand_waits_for_its_person_and_the_wait_is_no_part_of_the_runs_budget(
    engine, monkeypatch, tmp_path
):
    """C, with no orchestrator: nothing cancels such a goal at 120 s, and the owner's word is "don't end". A
    line-up may then take far longer than a run's own budget (ten minutes, node.CONSOLE_LINE_UP_S), and no
    budget may cut a run part way through a motion: the run's 90 s (3 s here) start again at each go. As
    first drafted the run that was let go after a 4 s line-up was stopped at once, 4 s being more than 3."""
    import threading

    monkeypatch.setattr(node, "RUN_BUDGET_S", 3.0)
    monkeypatch.setenv("WAIT_S", "1.0")  # a second of "flight" after the go

    def check(expect_line_up, **env):
        _forget_run_logs(tmp_path)
        given(monkeypatch, FULL_RUN, wait=True)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        ctx = FakeCtx()
        threading.Timer(4.0, engine._answers.put, args=("y",)).start()
        t0 = time.monotonic()
        out = engine.execute("open_box", {}, ctx)
        assert out["lid_off"] and time.monotonic() - t0 >= 4.0
        assert ("line_up" in [p for p, *_ in ctx.said]) is expect_line_up

    _guided(engine, monkeypatch)
    check(True, GUIDE="1")  # guided in the middle of the run
    monkeypatch.delenv("GUIDE")
    engine.set_parameters([Parameter("line_up", value=True), Parameter("wrist_first", value=True)])
    check(True)  # lined up before it, the run itself watching
    engine.set_parameters([Parameter("wrist_first", value=False)])
    monkeypatch.setattr(engine, "_box_source", lambda env, instance_id, watch=None, refused=(): InPlace())
    check(True)  # ... and with the scene camera's watch, the run started after
    # under the orchestrator's 120 s the budget is the goal's, the line-up inside it, as it was
    _forget_run_logs(tmp_path)
    monkeypatch.setattr(engine, "_ui_live", lambda: True)
    given(monkeypatch, FULL_RUN[:3], wait=True)
    monkeypatch.setenv("WAIT_S", "30")
    monkeypatch.setattr(node, "LINE_UP_S", 2.9)
    ctx = FakeCtx()
    threading.Timer(1.5, ctx.events.append, args=({"type": "button", "button_id": "open"},)).start()
    t0 = time.monotonic()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert e.value.error_code == "timeout" and time.monotonic() - t0 < 3.0 + 3.0


@pytest.mark.parametrize("by", ["the orchestrator", "n", "the panel"])
def test_a_cancel_while_the_chair_is_guided_is_the_runs_own_ctrl_c(engine, monkeypatch, by, capsys):
    """D. A cancel is read at every moment of the guidance and stops it: the orchestrator's, the person's n,
    Cancel on the panel. It is the run's Ctrl+C as every cancel is (a wrist turn in flight stops, the arm
    holds where it is), sent before the line-up's source is closed, and the run is never told to go."""
    import threading

    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("GUIDE", "1")
    monkeypatch.setenv("GUIDE_BOX", "far")
    monkeypatch.setattr(engine, "_ui_live", lambda: by == "the panel")
    _made, told, runs = _guided(engine, monkeypatch, proven=False)
    sent = []
    real = node.Run.interrupt
    monkeypatch.setattr(node.Run, "interrupt", lambda run: (sent.append(signal.SIGINT), real(run)))
    base = []
    monkeypatch.setattr(engine, "_cancel_from_ui", lambda by_="the panel": base.append(by_))
    ctx = FakeCtx()
    capsys.readouterr()
    if by == "the orchestrator":
        threading.Timer(1.5, setattr, args=(ctx, "cancel", True)).start()
    elif by == "n":
        threading.Timer(1.5, engine._answers.put, args=("n",)).start()
    else:
        threading.Timer(1.5, ctx.events.append, args=(CANCEL_PRESS,)).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        engine.execute("open_box", {}, ctx)
    assert time.monotonic() - t0 < 6.0 and sent == [signal.SIGINT] and told == [] and engine._run is None
    assert runs[0].proc.returncode in (130, -signal.SIGINT)
    assert base == {"the orchestrator": [], "n": ["n answered"], "the panel": ["the panel"]}[by]
    assert ctx.world == [] and ctx.objects == {}
    view = capsys.readouterr().out
    assert "RESULT  CANCELLED at step 3 (LINE UP)" in view
    assert "arm: stopped where it was (a wrist that looked stands turned)" in view


def test_an_n_while_the_run_flies_is_the_cancel_too_and_a_y_then_answers_nothing(engine, monkeypatch):
    """~/answer is heard for the whole action: an n at any moment is the person's cancel (the run's own
    Ctrl+C); a y where nothing is asked is nothing at all."""
    import threading

    given(monkeypatch, FULL_RUN[:3], wait=True)
    sent = []
    real = node.Run.interrupt
    monkeypatch.setattr(node.Run, "interrupt", lambda run: (sent.append(signal.SIGINT), real(run)))
    ctx = FakeCtx()
    threading.Timer(0.8, engine._answers.put, args=("y",)).start()
    threading.Timer(1.6, engine._answers.put, args=("n",)).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        engine.execute("open_box", {}, ctx)
    assert 1.6 <= time.monotonic() - t0 < 6.0 and sent == [signal.SIGINT]


def test_what_judges_the_place_dying_while_the_chair_is_guided_ends_it_safely_and_says_so(
    engine, monkeypatch, tmp_path
):
    """D, "the helper process dying": reach_probe ends in the middle of a guidance. The line-up ends at once
    with what it said, the run is told to stop and turns its wrist back (never a go, never a Ctrl+C), and
    the error says which helper it was."""
    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("GUIDE", "1")
    monkeypatch.setenv("URDF", "/state/arm.urdf")
    _made, told, runs = _guided(engine, monkeypatch, proven=False)
    (tmp_path / "reach_probe").write_text("#!/bin/sh\nsleep 1\necho 'MemoryError: the grid' >&2\nexit 1\n")
    (tmp_path / "reach_probe").chmod(0o755)
    monkeypatch.setattr(node, "executable", lambda name: str(tmp_path / name))  # press_demo: the fixture's stand-in
    ctx = FakeCtx()
    t0 = time.monotonic()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert e.value.error_code == "line_up_unavailable" and time.monotonic() - t0 < 8.0
    assert "reach_probe" in str(e.value) and "MemoryError: the grid" in str(e.value)
    assert told == [{"stop": True}] and engine._run is None and runs[0].proc.returncode == 5


def test_a_run_that_dies_while_the_chair_is_guided_fails_with_what_it_said(engine, monkeypatch):
    """... and the run itself (the watch) ending under the guidance: nothing is left to tell to go or stop,
    and the action fails with the run's own last word."""
    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("GUIDE", "1")
    monkeypatch.setenv("GUIDE_BOX", "holds")  # its wrist's turn to the view stopped part way: exit 6
    _made, told, _runs = _guided(engine, monkeypatch)
    ctx = FakeCtx()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert e.value.error_code == "arm_holds" and not e.value.recoverable and "the arm holds" in str(e.value)
    assert {"go": True} not in told and engine._run is None


def test_a_run_may_be_guided_more_than_once_each_time_on_the_persons_yes(engine, monkeypatch):
    """The place the person lined up on may still plan no press (the run has the planner and the table's
    edges; what judged the line-up had neither): the run asks again, and goes on only on another yes."""
    import threading

    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("GUIDE", "2")
    made, told, runs = _guided(engine, monkeypatch)
    ctx = FakeCtx()
    for at in (2.5, 5.5):
        threading.Timer(at, engine._answers.put, args=("y",)).start()
    out = engine.execute("open_box", {}, ctx)
    assert out["lid_off"] and told == [{"go": True}, {"go": True}] and len(runs) == 1 and len(made) == 2
    said = [m for p, _pct, m, _pausable in ctx.said if p == "line_up"]
    assert said.count("Opening the container") == 2
    # the second guidance holds BOTH places the run refused against its line-up, not only the newest: as
    # first written the first was forgotten, and the person could be led back onto it (review of 2026-11-17)
    assert made[0]["refused"] == [[0.712, -0.331, 0.2945]]
    assert made[1]["refused"] == [[0.712, -0.331, 0.2945], [0.662, -0.331, 0.2945]]
    # ... and after each yes the run is looking again, whatever it said before it
    progress = [m for _p, _pct, m, _pausable in ctx.said]
    assert progress.count("looking for the container again") == 2


def test_a_home_before_the_chair_is_guided_is_not_the_way_home_and_the_bar_keeps_its_room():
    """After the look and the aim the arm goes home before the chair is guided (reach_press.no_way_in), and
    the run then goes on from the find. Read as the way home, that home reported the run 100 % done and
    "going home", and everything after it (the press, the lid) at 100 %. The run says it is to be guided
    BEFORE it homes; from that row to its request the home is the arm going to rest."""
    track = node.Tracker("open_box")
    home = {"leg": "home", "kind": "motion", "ok": True}
    said = [
        track.add(row)
        for row in (
            FULL_RUN[0],  # found
            FULL_RUN[1],  # reach:look
            {"leg": "reach_refused", "kind": "note", "why": "no press reaches the aimed button (x)", "guided": True},
            home,
            {"leg": "guide", "kind": "note", "n": 1},
            FULL_RUN[0],  # found again, after the yes
            FULL_RUN[1],
            *FULL_RUN[2:-1],
            home,
        )
    ]
    # neither says the chair is to move: the arm is on its way to rest (test_until_the_arm_is_at_rest_...)
    assert said[2] == ("locate", 0.1, "no press plans from here: wait, the arm comes to rest first")
    assert said[3] == ("locate", 0.1, "going to rest")
    pcts = [pct for _phase, pct, _m in filter(None, said)]
    assert pcts == sorted(pcts) and pcts[-2] < 1.0 and said[-1] == ("home", 1.0, "going home")
    # a run that is NOT guided: its reach_refused row is no such word, and its home is the way home
    track = node.Tracker("open_box")
    for row in (FULL_RUN[0], {"leg": "reach_refused", "kind": "note", "why": "NO BOX"}):
        track.add(row)
    assert track.add(home) == ("home", 1.0, "going home")


# -- a container no camera sees: guided too, never just ended (owner 2026-11-19) -----------------------------
UNSEEN_SAYS = "The arm cannot see the container. Move the chair in front of it"
# FAKE_RUN's own two reasons (its NOT_SEEN and its WHY), as the engine hears them
NOT_SEEN_WHY = (
    "no camera sees the container (scene camera: scene segmenter: saw no food container in 14 frames; the wrist's "
    "look in front: wrist camera: no counter in 4 frames (no level plane wider than a container))"
)
NO_PRESS_WHY = (
    "no reach-out press plans for the scene camera's box (30 deg: the way down (the slide in: trajectory sanity "
    "gate: arm_joint_1 excursion 2.012 rad > |d| + margin 1.632); 40 deg: the way down (the press stroke is not a "
    "straight line the arm can fly (the chair and user: arm_half_arm_1_link 0 mm into person/lap (padded 20 mm + 5 "
    "mm), 46% of the way)))"
)


def test_a_container_no_camera_sees_does_not_end_the_run_the_chair_is_guided_on_the_console(
    engine, monkeypatch, capsys
):
    """The owner was asked: "A box no camera sees at all still ends as 'not found'. Should that become
    guidance too ('move so the container is in front of the arm')?" and answered yes (2026-11-19). The run
    whose cameras found no container ended FAILED at its find, not_found. Now the SAME process asks to be
    guided, as for a box out of reach: the person is told at once that the arm cannot see the container and
    to move the chair (no "Looking for the container" while the wrist turns: the run has looked), then,
    the moment the watch has it, what the ordinary guidance says, the question, and at the y the run finds
    the box afresh and opens it. No place is held against the line-up: none was measured."""
    import threading

    given(
        monkeypatch,
        FULL_RUN,
        [
            "[press_demo] SET-DOWN CONTACT: the lid met the table +12 mm from where the plan put it",
            "[press_demo] DONE — box open, lid on the table at [0.60, -0.07] (clear), let go where the arm felt "
            "it, arm home",
        ],
    )
    monkeypatch.setenv("GUIDE", "1")
    monkeypatch.setenv("GUIDE_WHAT", "unseen")
    monkeypatch.setenv("GUIDE_BOX", "comes")  # its watch sees none for 1.5 s, then the container in place
    made, told, runs = _guided(engine, monkeypatch)
    ctx = FakeCtx()
    capsys.readouterr()
    threading.Timer(5.0, engine._answers.put, args=("y",)).start()
    out = engine.execute("open_box", {}, ctx)
    assert out["lid_off"] and out["presses"] == 1
    (run,) = runs  # one process, from the first look to the way home
    assert run.argv[-2:] == ["--execute", "--guide"] and told == [{"go": True}]
    assert made == [{"watch": run, "refused": [], "unseen": NOT_SEEN_WHY}]  # no place refused: none was measured
    said = [m for p, _pct, m, _pausable in ctx.said if p == "line_up"]
    assert said[0] == "No camera sees the container. Move the chair as told"
    assert said[1] == UNSEEN_SAYS and "Looking for the container" not in said  # at once: the run has looked
    # ... then, the moment its watch has the container, the ordinary guidance, to the question and the yes
    assert said[2] == "Hold still, checking" and said[-2:] == ["Ready. Open the container?", "Opening the container"]
    assert UNSEEN_SAYS not in said[2:]
    phases = [p for p, *_ in ctx.said]
    assert phases[-1] == "home" and "line_up" not in phases[phases.index("look") :]  # the run, after its yes
    assert ctx.panels == []
    view = capsys.readouterr().out.splitlines()
    find = view.index("   find      ...   looking for the box")
    assert view[find + 1].startswith(" 1 FIND      FAIL  no camera sees the container")
    assert view[find + 2 : find + 6] == [
        "             next  the run asks for the chair to be guided",
        "   line up   ...   " + node.GUIDE_DOING,
        "             say   " + UNSEEN_SAYS,
        "             say   Hold still, checking",
    ]
    assert "             ask   Ready. Open the container?   y / n" in view
    assert any(ln.startswith(" 2 LINE UP   ok    yes") for ln in view)
    assert "   find      ...   looking for the box again" in view
    assert any(ln.startswith("RESULT  DONE — box open, lid on the table, arm home") for ln in view)
    assert not any("NO BOX" in ln or "no box found" in ln for ln in view)


def test_a_run_by_hand_whose_cameras_see_no_container_is_not_found_as_it_always_was(engine, monkeypatch, capsys):
    """What the guidance replaces (and FAKE_RUN's own check): without --guide the same run says NO BOX and
    ends with exit 2, and the action fails not_found, recoverable, with the words it had."""
    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("GUIDE", "1")
    monkeypatch.setenv("GUIDE_WHAT", "unseen")
    real_run = node.Run
    monkeypatch.setattr(node, "Run", lambda argv, env, log, talk=False: real_run(argv[:-1], env, log, talk=talk))
    capsys.readouterr()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, FakeCtx())
    assert e.value.error_code == "not_found" and e.value.recoverable
    assert str(e.value).startswith("[press_demo] NO BOX — scene camera: scene segmenter: saw no food container")
    view = capsys.readouterr().out
    assert " 1 FIND      FAIL  no box found" in view and "RESULT  FAILED at step 1 (FIND)" in view


def test_with_the_ui_an_unseen_container_is_said_on_the_panel_with_cancel_alone_until_a_camera_has_it(
    engine, monkeypatch, capsys
):
    """On the UI: until the arm is at rest and the run has asked, the panel says what was found and to wait
    (here the arm homes from a side survey's pose first), never that the chair is to move. Then the line-up's
    panel: that the arm cannot see the container and that the chair is to move, with Cancel alone; the
    ordinary guidance once the watch has it; the question with Open; and after Open the run's own panel."""
    import threading

    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("GUIDE", "1")
    monkeypatch.setenv("GUIDE_WHAT", "unseen")
    monkeypatch.setenv("GUIDE_BOX", "comes")
    monkeypatch.setenv("GUIDE_ROWS", "1")
    monkeypatch.setenv("GUIDE_HOME_S", "1.5")  # the home to rest, flown before it asks to be guided
    # (after the go it takes a moment to find the box again, as a run does: with none, whether the panel
    # that says so is sent before the run's next row is a race, lost once in a dozen runs on a busy machine)
    monkeypatch.setenv("AGAIN_S", "0.8")
    monkeypatch.setattr(engine, "_ui_live", lambda: True)
    _made, told, _runs = _guided(engine, monkeypatch)
    ctx = FakeCtx()
    seen = _panels_as_sent(ctx)
    capsys.readouterr()
    threading.Timer(6.5, ctx.events.append, args=({"type": "button", "button_id": "open"},)).start()
    out = engine.execute("open_box", {}, ctx)
    assert out["lid_off"] and told == [{"go": True}]
    # the progress before the line-up's first word: what was found and to wait, then the way to rest
    phases = [p for p, *_ in ctx.said]
    before = [m for _p, _pct, m, _pausable in ctx.said[: phases.index("line_up")]]
    assert before[-2:] == ["no container in view", "going to rest"]
    assert not [m for m in before if _tells_to_move(m)], before
    # the panel, in the order it was shown
    said = [message for _t, message, _buttons in seen]
    wait = said.index("No container in view")
    assert seen[wait + 1][0] - seen[wait][0] >= 1.2 and said[wait + 1] == "Going to rest"
    unseen = said.index(UNSEEN_SAYS)
    assert unseen > wait + 1 and seen[unseen][2] == ["cancel"]
    assert not [m for m in said[:unseen] if _tells_to_move(m)]
    assert "Looking for the container" not in said[wait:]  # (the run's own panel said it while it looked)
    # ... then, the moment its watch has the container, the ordinary guidance, to the question and Open
    assert said[unseen + 1] == "Hold still, checking" and UNSEEN_SAYS not in said[unseen + 1 :]
    opened = said.index("Opening the container")
    assert seen[opened - 1][1:] == ("Ready. Open the container?", ["open", "cancel"])
    assert seen[opened + 1][1:] == ("Looking for the container again", ["cancel"])  # the run's own, after Open
    view = capsys.readouterr().out.splitlines()
    step = next(i for i, line in enumerate(view) if line.startswith("   line up   ..."))
    assert view[step - 3 : step - 1] == [
        "             next  the run asks for the chair to be guided",
        "   home      ...   going to rest before the chair is guided",
    ]
    assert not [line for line in view[:step] if _tells_to_move(line)]
    assert "             ask   Ready. Open the container?   (Open on the UI)" in view


def test_a_container_that_never_comes_into_view_ends_as_one_that_never_comes_into_reach_and_says_which(
    engine, monkeypatch, capsys
):
    """The same end as a person who never gets into reach: out of time the run is told to stop on its stdin
    (no Ctrl+C), turns its wrist back to rest and ends, the action fails not_lined_up, recoverable, nothing
    pressed. And the words say what was missing: no camera saw the container, on the console's last line,
    in the flight view's result and in the error. The next open_box does not line up first on its account:
    a line-up that never saw the container has nothing to go on from."""
    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("GUIDE", "1")
    monkeypatch.setenv("GUIDE_WHAT", "unseen")
    monkeypatch.setenv("GUIDE_BOX", "unseen")  # its watch never has the container
    engine.set_parameters([Parameter("wrist_first", value=True)])
    monkeypatch.setattr(node, "CONSOLE_LINE_UP_S", 2.0)
    monkeypatch.setattr(node.line_up, "LINGER_S", 0.2)
    _made, told, runs = _guided(engine, monkeypatch, proven=False)
    sent = []
    real = node.Run.interrupt
    monkeypatch.setattr(node.Run, "interrupt", lambda run: (sent.append(signal.SIGINT), real(run)))
    ctx = FakeCtx()
    capsys.readouterr()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert e.value.error_code == "not_lined_up" and e.value.recoverable
    assert str(e.value).startswith("no camera saw the container within 2 s: nothing was pressed (move the chair so")
    assert "the chair was guided because the run's own look had found no container" in str(e.value)
    assert str(e.value).endswith("-- wrist camera: no lid-sized top on the table in view")
    assert "not where the arm opens it" not in str(e.value) and "no press planned" not in str(e.value)
    assert told == [{"stop": True}] and sent == [] and engine._run is None
    assert runs[0].proc.returncode == 5  # press_demo's EXIT_NOT_LINED_UP: the wrist back at rest
    assert engine._line_up_first is None and ctx.world == [] and ctx.objects == {}
    said = [m for p, _pct, m, _pausable in ctx.said if p == "line_up"]
    assert said[1] == UNSEEN_SAYS and node.CONSOLE_AGAIN_UNSEEN in said and node.CONSOLE_AGAIN not in said
    view = capsys.readouterr().out
    assert "             say   " + UNSEEN_SAYS in view
    assert "             say   Out of time: the container was never seen. Start open_box again in front of it" in view
    assert "LINE UP   FAIL  the container never came into view within 2 s" in view
    assert "        why: the container never came into view within 2 s" in view
    assert "        box: not pressed    lid: on the box    arm: at rest, the wrist turned back" in view


@pytest.mark.parametrize("by", ["the orchestrator", "n", "the panel"])
def test_a_cancel_while_the_arm_cannot_see_the_container_is_the_runs_own_ctrl_c(engine, monkeypatch, by):
    """Cancel, or n, at any moment of it, as of every line-up: the run's Ctrl+C, the action cancelled, the
    run never told to go."""
    import threading

    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("GUIDE", "1")
    monkeypatch.setenv("GUIDE_WHAT", "unseen")
    monkeypatch.setenv("GUIDE_BOX", "unseen")
    monkeypatch.setattr(engine, "_ui_live", lambda: by == "the panel")
    monkeypatch.setattr(engine, "_cancel_from_ui", lambda by="": None)
    _made, told, runs = _guided(engine, monkeypatch, proven=False)
    sent = []
    real = node.Run.interrupt
    monkeypatch.setattr(node.Run, "interrupt", lambda run: (sent.append(signal.SIGINT), real(run)))
    ctx = FakeCtx()
    cancel = {
        "the orchestrator": lambda: setattr(ctx, "cancel", True),
        "n": lambda: engine._answers.put("n"),
        "the panel": lambda: ctx.events.append(CANCEL_PRESS),
    }
    threading.Timer(1.5, cancel[by]).start()
    with pytest.raises(Cancelled):
        engine.execute("open_box", {}, ctx)
    assert sent == [signal.SIGINT] and told == [] and engine._run is None
    assert runs[0].proc.poll() in (130, -signal.SIGINT)
    assert UNSEEN_SAYS in [m for p, _pct, m, _pausable in ctx.said if p == "line_up"]
    assert ctx.world == [] and ctx.objects == {}


def test_under_the_orchestrators_120_s_an_unseen_container_with_no_time_left_ends_recoverable_and_says_so(
    engine, monkeypatch, capsys
):
    """The same limits as a box out of reach, untouched: a line-up may end no later than LINE_UP_S after the
    goal began, and with under GUIDE_MIN_S of that left nothing is guided. A find that sees no container has
    spent the scene camera's whole first look and the wrist's views by then, so with the orchestrator's
    token this is its usual end: not_lined_up at once, recoverable, saying what was missing, and the next
    open_box lines up first, with its whole time."""
    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("GUIDE", "1")
    monkeypatch.setenv("GUIDE_WHAT", "unseen")
    monkeypatch.setenv("GUIDE_AFTER_S", "1.0")  # its looks
    monkeypatch.setattr(engine, "_ui_live", lambda: True)
    monkeypatch.setattr(node, "LINE_UP_S", 5.0)  # of which 4 are left: under GUIDE_MIN_S
    _made, told, runs = _guided(engine, monkeypatch)
    ctx = FakeCtx()
    capsys.readouterr()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert e.value.error_code == "not_lined_up" and e.value.recoverable
    assert str(e.value).startswith("no camera sees the container, and this run had no time left to guide the chair")
    assert "Press open again: it lines up first" in str(e.value) and "nothing was pressed" in str(e.value)
    assert told == [{"stop": True}] and runs[0].proc.returncode == 5 and engine._line_up_first is not None
    assert engine._line_up_first[1] == []  # ... holding no place against it: none was measured
    view = capsys.readouterr().out
    assert " 1 FIND      FAIL  no camera sees the container" in view
    assert "LINE UP   FAIL  no time left to guide the chair" in view


def test_what_a_run_is_guided_for_is_said_as_what_it_is():
    """A run asks to be guided for a press that does not plan, for a container its wrist sees out of reach,
    or for one no camera sees. Until 2026-11-19 all three were said in the words of the first: the far top's
    row read "no press plans from here: wait ..." and its reason "no press plans from here: the way in does
    not plan", of a container no press had been planned for. The names are the run's own (press_demo)."""
    from box_opening.tasks import press_demo

    assert (node.FAR, node.UNSEEN) == (press_demo.GUIDE_FAR, press_demo.GUIDE_UNSEEN)
    far_why = "the wrist camera sees a lid-sized top at [1.02, -0.20], out of the arm's reach"
    assert node.guide_reason({"what": "unseen", "why": NOT_SEEN_WHY}) == "no camera sees the container"
    # (one thing only: what the scene camera made of two things further off is no part of it)
    two = NOT_SEEN_WHY.replace("saw no food container in 14 frames", "two places pass as this container, [0.8, 0.2]")
    assert node.guide_reason({"what": "unseen", "why": two}) == "no camera sees the container"
    assert node.guide_reason({"what": "far", "why": far_why}) == "what the wrist camera sees is out of the arm's reach"
    assert node.guide_reason({"why": NO_PRESS_WHY}) == REASON.replace(" (0.79 m out)", "")  # no way in: as it was
    assert node.guide_reason({}) == node.no_press_reason("no reason given")
    # the run's row, from the moment it is written until the run asks: never that the chair is to move
    # (nor, of these two, "wait, the arm comes to rest first": after a wrist look that found nothing the arm
    # stands where it stood, and under the 120 s rule the run may be about to end: review of 2026-11-19)
    for what, says in (
        ("unseen", "no container in view"),
        ("far", "out of reach from here"),
        (None, node.NO_WAY_IN_WAIT),
    ):
        track = node.Tracker("open_box")
        row = {"leg": "reach_refused", "kind": "note", "why": "x", "guided": True, **({"what": what} if what else {})}
        assert track.add(row) == ("locate", 0.0, says) and track.to_rest and not _tells_to_move(says)
        assert len(says) <= 63  # the run's panel shows it: one short message
        assert track.add({"leg": "home", "kind": "motion", "ok": True}) == ("locate", 0.0, "going to rest")
        track.add({"leg": "guide", "kind": "note", "n": 1})
        assert not track.to_rest
    # ... and its line is never how the run ended, whatever its cameras' reasons hold (a planner's "IK_FAIL"
    # in a view that was not flown): the run went on from it. Read as the last word, "nothing was pressed"
    # made a traceback after it a no_press that may be tried again
    line = "[press_demo] NOT SEEN: " + NOT_SEEN_WHY.replace("no counter in 4 frames", "not flown: refused (IK_FAIL)")
    lines = [line + "; nothing was pressed"]
    died = node.error_for("open_box", 1, lines + ["Traceback (most recent call last):", "ValueError: a bug"])
    assert died.error_code == "failed" and not died.recoverable and str(died) == "ValueError: a bug"
    stop = "[press_demo] STOP: the stroke pressed nothing (3 N at its end)"
    assert node.error_for("open_box", 1, lines + [stop]).error_code == "not_popped"
    ended = node.error_for("open_box", 5, lines + ["[press_demo] LINE-UP: ended without a go; nothing was pressed"])
    assert ended.error_code == "not_lined_up" and "NOT SEEN" not in str(ended)


# -- what no moving of the chair cures: said as what it is, and ended (review of 2026-11-19) ------------------
NO_CURE_CASES = [
    # (the watch's mode in FAKE_RUN, what it is of, the panel's sentence while it stands, its end's)
    (
        "untied",
        "not_chosen",
        "The arm cannot tell which container you chose",
        "The arm cannot tell which container you chose. Choose it again. You can try again",
    ),  # fmt: skip
    (
        "camera",
        "fault",
        "The arm's vision is not working",
        "The arm's vision is not working. Nothing was pressed. You can try again",
    ),  # fmt: skip
]


@pytest.mark.parametrize(("mode", "kind", "says", "ends"), NO_CURE_CASES)
def test_what_no_chair_move_cures_is_said_as_what_it_is_and_ends_the_guidance_in_seconds(
    engine, monkeypatch, capsys, mode, kind, says, ends
):
    """Guidance is for what moving the chair can cure. States it cannot cure were guided with "The arm
    cannot see the container. Move the chair in front of it" until the time was out, which on the UI is five
    minutes of a person driving a wheelchair with the arm's wrist turned: a lid in every frame of the watch
    that nothing ties to the selected container, a camera that gives no frame. Each is said as what it is,
    with Cancel and no word of moving; standing guide.NO_CURE_S (ten seconds; five here) it ends the
    guidance as the find that meets it ends, not_found and recoverable, with a sentence of its own for the
    panel's end; the run is told to stop on its stdin and turns its wrist back. (Two lid-sized tops in a
    frame are said too, and end nothing: the test after this one.)"""
    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("GUIDE", "1")
    monkeypatch.setenv("GUIDE_WHAT", "unseen")  # the run's own look saw nothing; its watch then does
    monkeypatch.setenv("GUIDE_BOX", mode)
    monkeypatch.setattr(engine, "_ui_live", lambda: True)
    monkeypatch.setattr(node, "LINE_UP_S", 60.0)
    monkeypatch.setattr(node.guide, "NO_CURE_S", 5.0)
    monkeypatch.setattr(node.line_up, "LINGER_S", 0.3)
    _made, told, runs = _guided(engine, monkeypatch, proven=False)
    sent = []
    real = node.Run.interrupt
    monkeypatch.setattr(node.Run, "interrupt", lambda run: (sent.append(signal.SIGINT), real(run)))
    ctx = FakeCtx()
    seen = _panels_as_sent(ctx)
    capsys.readouterr()
    t0 = time.monotonic()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert time.monotonic() - t0 < 12.0  # seconds, not the 60 its line-up had
    assert (e.value.error_code, e.value.recoverable) == ("not_found", True)
    assert node.end_says(error=e.value) == (ends, True) and len(ends) <= 100
    assert node.NO_CURE_WHY[kind] in str(e.value) and "nothing was pressed" in str(e.value)
    # ... and the watch's own last word on it, for whoever reads the result
    word = {"untied": "the run would not take it", "camera": "no new wrist frame"}[mode]
    assert " -- " in str(e.value) and str(e.value).endswith(word)
    assert told == [{"stop": True}] and sent == [] and runs[0].proc.returncode == 5 and engine._run is None
    assert engine._line_up_first is None and ctx.world == [] and ctx.objects == {}
    said = [message for _t, message, _buttons in seen]
    first = said.index(says)
    assert seen[first][2] == ["cancel"] and seen[first + 1][1:] == (says, [])  # then its last, with no button
    assert "Ready. Open the container?" not in said and not [m for m in said[first:] if _tells_to_move(m)]
    # from then on nothing says that the arm cannot see the container (before it, the run's own look had
    # found none in view, and that may have been said: this stand-in's watch speaks before the first panel)
    assert not [m for m in said[first:] if "cannot see" in m]
    view = capsys.readouterr().out
    assert "             say   " + says in view and "LINE UP   FAIL  " + node.NO_CURE_WHY[kind] in view
    assert "        box: not pressed    lid: on the box    arm: at rest, the wrist turned back" in view


def test_two_things_that_could_be_the_container_are_said_and_end_no_guidance_before_its_time(engine, monkeypatch):
    """Two lid-sized tops in the watch's frames: said as what the arm sees, and the guidance goes on to its
    own time, as it did at 21aa7cc (the integrator, 2026-11-19: as a626985 had it they ended every line-up
    after seven seconds). Here guide.NO_CURE_S is a second: anything that ended on it would have, long
    before the line-up's five."""
    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("GUIDE", "1")
    monkeypatch.setenv("GUIDE_WHAT", "unseen")
    monkeypatch.setenv("GUIDE_BOX", "two")
    monkeypatch.setattr(engine, "_ui_live", lambda: True)
    monkeypatch.setattr(node, "LINE_UP_S", 5.0)
    monkeypatch.setattr(node, "GUIDE_MIN_S", 0.5)
    monkeypatch.setattr(node.guide, "NO_CURE_S", 1.0)
    monkeypatch.setattr(node.line_up, "LINGER_S", 0.3)
    _made, told, _runs = _guided(engine, monkeypatch, proven=False)
    ctx = FakeCtx()
    seen = _panels_as_sent(ctx)
    t0 = time.monotonic()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert time.monotonic() - t0 > 4.0  # its whole time
    assert (e.value.error_code, e.value.recoverable) == ("not_lined_up", True)  # out of time, as ever
    assert str(e.value).startswith("no camera saw the container within") and told == [{"stop": True}]
    said = [message for _t, message, _buttons in seen]
    two = said.index("The arm sees two things that could be the container")
    assert seen[two][2] == ["cancel"] and not [m for m in said[two:] if "cannot see" in m]
    assert said[two + 1] == "Container not seen in time. Press open again in front of it"


def test_a_line_up_before_a_run_whose_camera_gives_no_frame_never_says_to_move_the_chair(engine, monkeypatch):
    """Of every line-up, not only the new one: before a run (`line_up` on), carried, or in the middle of one.
    A watch whose camera delivers nothing from the start says "Looking for the container" until that has
    stood, then what it is, and ends. The cannot-see sentence is never shown."""
    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("LINE_UP", "camera")
    _wrist_lining_up(engine, monkeypatch, proven=False)
    monkeypatch.setattr(node, "LINE_UP_S", 60.0)
    monkeypatch.setattr(node.guide, "NO_CURE_S", 3.0)
    monkeypatch.setattr(node.line_up, "LINGER_S", 0.3)
    ctx = FakeCtx()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert (e.value.error_code, e.value.recoverable, e.value.says) == ("not_found", True, node.END_NO_CURE["fault"])
    said = [m for _p, _pct, m, _pausable in ctx.said]
    assert said[:2] == ["Looking for the container", "The arm's vision is not working"]
    assert not [m for m in said if "cannot see" in m or _tells_to_move(m)]
    assert {p["message"] for p in ctx.panels} >= {"Looking for the container", "The arm's vision is not working"}
    assert not [p for p in ctx.panels if "cannot see" in p["message"]]


def test_a_line_up_whose_time_runs_out_on_a_camera_that_gives_no_frame_ends_as_that_and_not_as_out_of_time(
    engine, monkeypatch
):
    """Under the orchestrator's 120 s a line-up may have fewer seconds than guide.NO_CURE_S. Its time then
    runs out with "The arm's vision is not working" on the panel, and its end is that state's: the same
    sentence as its last, not_found with the camera's own end sentence. Not "Container not seen in time.
    Press open again in front of it" and "The arm never saw the container": no place of the chair is what
    was missing."""
    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("LINE_UP", "camera")
    _wrist_lining_up(engine, monkeypatch, proven=False)
    monkeypatch.setattr(node, "LINE_UP_S", 3.5)
    monkeypatch.setattr(node, "CONSOLE_LINE_UP_S", 3.5)
    monkeypatch.setattr(node.line_up, "LINGER_S", 0.3)
    assert node.guide.NO_CURE_S == 10.0  # as flown: it has not stood its time when the line-up's runs out
    ctx = FakeCtx()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert (e.value.error_code, e.value.recoverable, e.value.says) == ("not_found", True, node.END_NO_CURE["fault"])
    assert "as the line-up's" in str(e.value) and "ran out" in str(e.value) and engine._line_up_first is None
    said = [m for _p, _pct, m, _pausable in ctx.said]
    assert said[:2] == ["Looking for the container", "The arm's vision is not working"]
    assert not [m for m in said if "not seen" in m.lower() or "press open again" in m.lower() or "Out of time" in m]
    assert not [m for m in said if "cannot see" in m or _tells_to_move(m)]


@pytest.mark.parametrize("what", ["not_chosen", "several", "fault", None])
def test_a_find_that_ends_on_what_no_chair_move_cures_ends_as_it_always_did_with_its_own_sentence(
    engine, monkeypatch, what
):
    """At the find itself nothing is guided for these: the run says NO BOX, goes home and ends with exit 2,
    as at 21aa7cc, and the action fails not_found, recoverable. Its row says what the find was, and the
    panel's end says that, not "The arm could not find the container": of a container in front of the arm
    that nothing ties to the one chosen, the cure is to choose it again, and the person is told so."""
    why = "NO BOX — scene camera: scene segmenter: the selected container (fc_9) was in none of its 70 frames"
    row = {"leg": "reach_refused", "kind": "note", "why": why, **({"what": what} if what else {})}
    given(monkeypatch, [row, FULL_RUN[-1]], ["[press_demo] " + why + " — returning home"], code=2)
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, FakeCtx())
    assert (e.value.error_code, e.value.recoverable) == ("not_found", True) and "NO BOX" in str(e.value)
    said, failed = node.end_says(error=e.value)
    assert failed and len(said) <= 100
    assert (
        said
        == {
            "not_chosen": "The arm cannot tell which container you chose. Choose it again. You can try again",
            "several": "The arm cannot choose between two things that could be the container. You can try again",
            "fault": "The arm's vision is not working. Nothing was pressed. You can try again",
            None: "The arm could not find the container. Nothing was pressed. You can try again",
        }[what]
    )


def test_the_ends_on_what_no_chair_move_cures_are_one_short_line_each_and_true_of_every_case():
    """Every end sentence: one short line, 100 characters at most with what follows it, and saying nothing
    that is not known in every case that shows it."""
    from box_opening import guide

    assert set(node.END_NO_CURE) == set(node.NO_CURE_WHY) == set(guide.NO_CURE)
    for sentence in (*node.END_NO_CURE.values(), node.END_DISAGREE):
        for code, recoverable in (("not_found", True), ("not_found", False), ("not_lined_up", True)):
            error = node.saying(EngineError(code, "the log's words", recoverable=recoverable), sentence)
            said, failed = node.end_says(error=error)
            assert failed and said.startswith(sentence) and len(said) <= 100 and "\n" not in said, said
    # (a selected id the scene camera's detections do not hold: whether a camera sees a container at all is
    # not known in every case, and what one sees may be the very container the person chose)
    assert "sees a container" not in node.END_NO_CURE["not_chosen"]
    assert "not the one" not in node.END_NO_CURE["not_chosen"]
    # (two lid-sized tops: the second may be a bowl)
    assert "containers" not in node.END_NO_CURE["several"]
    # nothing tells the person to change what stands on the table
    for sentence in (*node.END_NO_CURE.values(), node.END_DISAGREE, *guide.NO_CURE.values()):
        assert not [bad for bad in ("move the container", "take away", "remove", "table") if bad in sentence.lower()]


def test_three_guidances_in_one_goal_and_the_next_request_ends_it_saying_the_two_did_not_agree(
    engine, monkeypatch, capsys
):
    """The review of 2026-11-19: Ready, yes, the run's own look finds no container (or no press) where its
    watch had one, guided again, with no end to the rounds: each line-up has its whole five minutes on the
    UI. Three (node.GUIDES_MAX) and the fourth request ends the action: the run is told to stop, its wrist
    turns back, not_lined_up and recoverable, and the panel's end says the arm could not go on each time.
    Nothing tells the person to move the chair for that fourth one."""
    import threading

    assert node.GUIDES_MAX == 3
    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("GUIDE", "4")  # it finds no press four times over, wherever the person stops
    made, told, runs = _guided(engine, monkeypatch)
    ctx = FakeCtx()
    capsys.readouterr()
    stop = threading.Event()

    def answer():  # a person who says yes to every "Ready" (a y to anything else is no yes)
        while not stop.wait(0.4):
            engine._answers.put("y")

    threading.Thread(target=answer, daemon=True).start()
    try:
        with pytest.raises(EngineError) as e:
            engine.execute("open_box", {}, ctx)
    finally:
        stop.set()
    assert (e.value.error_code, e.value.recoverable) == ("not_lined_up", True)
    assert e.value.says == node.END_DISAGREE
    assert node.end_says(error=e.value) == (
        "Ready three times, and the arm could not go on. Nothing was pressed. You can try again",
        True,
    )
    assert "a fourth time in one goal" in str(e.value) and "the slide in swings the arm wide" in str(e.value)
    assert told == [{"go": True}] * 3 + [{"stop": True}] and len(made) == 3 and len(runs) == 1
    assert runs[0].proc.returncode == 5 and engine._run is None and engine._line_up_first is None
    said = [m for p, _pct, m, _pausable in ctx.said if p == "line_up"]
    assert len([m for m in said if m.endswith(". Move the chair as told")]) == 3  # not for the fourth
    assert said.count("Opening the container") == 3
    view = capsys.readouterr().out
    assert "LINE UP   FAIL  guided 3 times already, and the run cannot go on" in view
    assert "        box: not pressed    lid: on the box    arm: at rest" in view


def test_after_a_yes_a_run_that_then_sees_no_container_is_looked_for_before_anyone_is_told_to_move(engine, monkeypatch):
    """The first guidance of a goal for a container no camera saw says so at once: the run has looked. After a
    yes it is otherwise: the watch HAD the container a moment ago where the person now stands, and "The arm
    cannot see the container. Move the chair in front of it" from the first panel would tell a person who is
    in place to move. There the watch looks first, as at any line-up's start."""
    import threading

    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("GUIDE", "2")
    monkeypatch.setenv("GUIDE_WHAT", "unseen")
    monkeypatch.setenv("GUIDE_BOX", "comes")
    monkeypatch.setenv("COMES_S", "0.6")
    made, told, _runs = _guided(engine, monkeypatch)
    ctx = FakeCtx()
    stop = threading.Event()

    def answer():
        while not stop.wait(0.4):
            engine._answers.put("y")

    threading.Thread(target=answer, daemon=True).start()
    try:
        out = engine.execute("open_box", {}, ctx)
    finally:
        stop.set()
    assert out["lid_off"] and told == [{"go": True}] * 2
    # only the first begins with the run's own word (both wait for a sighting before the boundary's judge)
    assert [(bool(m.get("unseen")), m.get("looked", True)) for m in made] == [(True, True), (True, False)]
    said = [m for p, _pct, m, _pausable in ctx.said if p == "line_up"]
    rounds = [i for i, m in enumerate(said) if m == "No camera sees the container. Move the chair as told"]
    assert len(rounds) == 2
    assert said[rounds[0] + 1] == UNSEEN_SAYS  # at once: the run had looked
    # ... the second: looking, and then (its container in view within the time a lost one is given) held and asked
    assert said[rounds[1] + 1] == "Looking for the container"
    assert UNSEEN_SAYS not in said[rounds[1] :]


def test_under_the_120_s_rule_an_unseen_containers_guidance_cut_short_carries_as_one_that_had_no_time_does(
    engine, monkeypatch, tmp_path
):
    """The review of 2026-11-19: under the orchestrator's 120 s a goal that found no container either had no
    time to guide (and the next open_box lined up first), or had a few seconds, timed out with the watch
    never having seen the container, and carried nothing: the next press began another whole look, which
    left that goal as little. The two ends are one now: where the 120 s rule cut the guidance of a run whose
    look had found none, the next open_box lines up first, with its whole time. Once: that carried line-up,
    if it sees nothing either, carries nothing. And a wait that was the person's own (a goal sent by hand:
    minutes) carries nothing for a container never seen, as before."""
    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("GUIDE", "1")
    monkeypatch.setenv("GUIDE_WHAT", "unseen")
    monkeypatch.setenv("GUIDE_BOX", "unseen")  # its watch never has the container
    monkeypatch.setenv("LINE_UP", "unseen")  # ... nor has the line-up carried to the goal after it
    engine.set_parameters([Parameter("wrist_first", value=True)])
    monkeypatch.setattr(engine, "_ui_live", lambda: True)  # the orchestrator's goal, its 120 s counted from it
    monkeypatch.setattr(node, "LINE_UP_S", 2.5)
    monkeypatch.setattr(node, "GUIDE_MIN_S", 0.5)
    monkeypatch.setattr(node.line_up, "LINGER_S", 0.2)
    _made, told, runs = _guided(engine, monkeypatch, proven=False)
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, FakeCtx())
    assert e.value.error_code == "not_lined_up" and str(e.value).startswith("no camera saw the container within")
    assert told == [{"stop": True}] and engine._line_up_first is not None  # the next one lines up first
    assert engine._line_up_first[1] == []  # (holding no place against it)
    # the goal after it: lined up first, sees nothing either, and carries nothing on
    _forget_run_logs(tmp_path)
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, FakeCtx())
    assert e.value.error_code == "not_lined_up" and "--line-up" in runs[1].argv and engine._line_up_first is None
    # a goal sent by hand: the wait was the person's, and nothing was cut short
    _forget_run_logs(tmp_path)
    monkeypatch.setattr(engine, "_ui_live", lambda: False)
    monkeypatch.setattr(node, "CONSOLE_LINE_UP_S", 1.5)
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, FakeCtx())
    assert e.value.error_code == "not_lined_up" and "--line-up" not in runs[2].argv
    assert engine._line_up_first is None


def test_the_answer_is_heard_on_the_nodes_own_topic_only_while_an_action_runs(engine):
    """~/answer (std_msgs/String): the subscription is the action's own (nothing is subscribed while the
    node is idle), what it hears is kept as y or n, and anything else is no answer."""
    from types import SimpleNamespace

    from capability_engine.action_scope import ActionScope
    from std_msgs.msg import String

    engine._hear_answers()  # no action runs: nothing to subscribe on (and nothing raised)
    scope = ActionScope(engine, engine._group)
    engine._current = SimpleNamespace(scope=scope)
    try:
        engine._answers.put("y")  # said to an action before this one
        engine._hear_answers()
        assert engine._answers.poll() is None
        ((_destroy, heard),) = scope._entities  # the action's own subscription, destroyed as it ends
        assert heard.topic_name == "/box_opening/answer" and heard.msg_type is String
        for data in ("y", " N\n", "maybe", ""):
            engine._on_answer(String(data=data))
        assert [engine._answers.poll() for _ in range(3)] == ["y", "n", None]
    finally:
        engine._current = None
        scope.close()
    assert scope._entities == []


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


# --- an orchestrator whose 120 s restart at every progress item (rammp-orchestrator e17c625, 2026-11-16) ---


def kept_alive(monkeypatch, **params):
    """The mock, lining up, for a goal of that orchestrator: its goal brings the token."""
    monkeypatch.setattr(node, "goal_brings_token", lambda: True)
    monkeypatch.setattr(node, "END_SHOW_S", 0.4)
    # ... and the deployment says so (`ui_line_up_s`: the package's fragment has the owner's 5 minutes)
    return _mock_lining_up(monkeypatch, **dict({"ui_line_up_s": 300.0}, **params))


OPEN_PRESS = {"type": "button", "button_id": "open"}


def test_the_uis_goal_waits_five_minutes_only_where_the_deployment_says_its_orchestrators_clock_restarts(
    ros, monkeypatch
):
    """`ui_line_up_s`: 0 in the node (it cannot ask the orchestrator which it is), the owner's "5 min"
    (2026-11-19) in the package's deployment fragment, and 0 whatever is set wherever something speaks
    against it."""
    import yaml

    assert node.PARAMS["ui_line_up_s"] == 0.0 and node.UI_LINE_UP_S == 300.0 and node.UI_LINE_UP_MAX_S == 600.0
    fragment = yaml.safe_load(open(FRAGMENT_MOCK.with_name("rammp-alternative.yaml")))["params"]
    assert fragment["ui_line_up_s"] == node.UI_LINE_UP_S  # as docs/interface.md states it
    handlers = list(LOGGER.handlers)
    cap = _mock_lining_up(monkeypatch)
    try:
        monkeypatch.setattr(node, "goal_brings_token", lambda: True)
        assert cap._ui_wait_s() == 0.0  # nothing set: the 120 s are the goal's
        for value, waits in (
            (300.0, 300.0),
            (300, 300.0),  # an integer, as a deployment may write it
            (0, 0.0),
            (-5.0, 0.0),
            (float("nan"), 0.0),
            (1e9, 600.0),
            (float("inf"), 600.0),
        ):
            cap.set_parameters([Parameter("ui_line_up_s", value=value)])
            assert cap._ui_wait_s() == waits, value
        cap.set_parameters([Parameter("ui_line_up_s", value=300.0)])
        monkeypatch.setattr(node, "goal_brings_token", lambda: False)
        assert cap._ui_wait_s() == 0.0  # no token in this build's goal: its orchestrator is the older one
        monkeypatch.setattr(node, "goal_brings_token", lambda: True)
        cap._clock_from_goal = True  # a goal of this node's was cut at 120 s
        assert cap._ui_wait_s() == 0.0
    finally:
        cap.destroy_node()
        LOGGER.handlers[:] = handlers
    # the node starts with the integer a deployment writes for "my orchestrator is the older one"
    cap = node.Capability(
        parameter_overrides=[Parameter("mock", value=True), Parameter("use_orchestrator", value=False)]
        + [Parameter("ui_line_up_s", value=0)]
    )
    cap.destroy_node()
    LOGGER.handlers[:] = handlers
    t0 = 100.0
    assert node.Asking(True, False, t0, wait_s=300.0).left_s(t0 + 200.0) == 300.0  # each line-up, whole
    assert node.Asking(True, True, t0).left_s(t0 + 10.0) == node.LINE_UP_S - 10.0  # the 120 s rule, as it was
    assert node.Asking(False, False, t0).left_s() == node.CONSOLE_LINE_UP_S  # a goal sent by hand, as it was


def test_under_such_an_orchestrator_a_line_up_outlasts_the_120_s_rule_and_the_end_is_said_on_the_panel(
    ros, monkeypatch
):
    """LINE_UP_S (what a goal's 120 s leave a line-up) does not bound it there: the person's Open, well after
    it, still starts the run. And how the action ended stays on the panel, with no button, before it ends."""
    import threading

    handlers = list(LOGGER.handlers)
    cap = kept_alive(monkeypatch, mock_phase_s=0.2)
    try:
        monkeypatch.setattr(node, "LINE_UP_S", 0.5)  # under the 120 s rule: over before the box is in reach
        ctx = FakeCtx()
        threading.Timer(3.2, ctx.events.append, args=(OPEN_PRESS,)).start()
        t0 = time.monotonic()
        out = cap.execute("open_box", {}, ctx)
        assert out["lid_off"] and ctx.objects == {"c1": {"opened": True}}
        assert time.monotonic() - t0 >= 3.2 + 0.4  # ... the end's 0.4 s on the panel included
        assert ctx.panels[-1] == {"title": node.PANEL_TITLE, "message": node.END_DONE, "buttons": []}
        before = [p for p in ctx.panels if p["message"] != node.END_DONE]
        assert before[-1]["message"] == "Going home" and ctx.panels.index(ctx.panels[-1]) > ctx.panels.index(before[-1])
        # the same goal under the 120 s rule: not in place in time, and no panel of the end's (no time for one)
        monkeypatch.setattr(node, "goal_brings_token", lambda: False)
        monkeypatch.setattr(node.line_up, "LINGER_S", 0.3)
        ctx = FakeCtx()
        with pytest.raises(EngineError) as e:
            cap.execute("open_box", {}, ctx)
        assert e.value.error_code == "not_lined_up"
        assert ctx.panels[-1]["message"] == "Press open again when you are in place"
    finally:
        cap.destroy_node()
        LOGGER.handlers[:] = handlers


def test_a_person_not_in_place_within_the_deployments_wait_is_told_so_and_nothing_runs(ros, monkeypatch):
    handlers = list(LOGGER.handlers)
    cap = kept_alive(monkeypatch, mock_phase_s=5.0, ui_line_up_s=0.8)  # a 30 s approach, a wait of 0.8 s
    try:
        monkeypatch.setattr(node.line_up, "LINGER_S", 0.3)
        ctx = FakeCtx()
        with pytest.raises(EngineError) as e:
            cap.execute("open_box", {}, ctx)
        assert e.value.error_code == "not_lined_up" and e.value.recoverable and "within 1 s" in str(e.value)
        assert ctx.panels[-1] == {
            "title": node.PANEL_TITLE,
            "message": "Not in place in time. Nothing was pressed. You can try again",
            "buttons": [],
            "style": "error",
        }
        assert {p for p, *_ in ctx.said} == {"line_up"} and ctx.objects == {} and ctx.world == []
    finally:
        cap.destroy_node()
        LOGGER.handlers[:] = handlers


def test_a_cancel_under_such_an_orchestrator_is_said_on_the_panel_as_the_action_ends(ros, monkeypatch):
    import threading

    handlers = list(LOGGER.handlers)
    cap = kept_alive(monkeypatch, mock_phase_s=1.0)
    try:
        monkeypatch.setattr(cap, "_cancel_from_ui", lambda *a: setattr(cap, "_person_cancelled", True))
        ctx = FakeCtx()
        threading.Timer(0.6, ctx.events.append, args=(CANCEL_PRESS,)).start()
        with pytest.raises(Cancelled):
            cap.execute("open_box", {}, ctx)
        assert ctx.panels[-1] == {"title": node.PANEL_TITLE, "message": "Cancelled", "buttons": []}
        assert ctx.objects == {} and ctx.world == []
        # the orchestrator's own cancel: the screen is its own again, and no end is sent to it
        ctx = FakeCtx()
        threading.Timer(0.6, setattr, args=(ctx, "cancel", True)).start()
        t0 = time.monotonic()
        with pytest.raises(Cancelled):
            cap.execute("open_box", {}, ctx)
        assert time.monotonic() - t0 < 0.6 + 0.35 and all(p["message"] != "Cancelled" for p in ctx.panels)
        assert cap._clock_from_goal is False  # ... and nowhere near 120 s after the goal
    finally:
        cap.destroy_node()
        LOGGER.handlers[:] = handlers


def test_while_the_person_takes_their_time_the_orchestrator_is_told_again_and_again(ros, monkeypatch):
    """The newest progress is sent again every KEEP_ALIVE_S with nothing new said (the orchestrator cancels
    after 120 s of silence); under the 120 s rule each thing is said once, as it always was."""
    import threading

    handlers = list(LOGGER.handlers)
    cap = kept_alive(monkeypatch, mock_phase_s=0.2)
    try:
        monkeypatch.setattr(node, "KEEP_ALIVE_S", 0.3)
        ctx = FakeCtx()
        threading.Timer(4.6, ctx.events.append, args=(OPEN_PRESS,)).start()  # asked at ~2.5 s: 2 s of waiting
        assert cap.execute("open_box", {}, ctx)["lid_off"]
        asked = [m for p, _pct, m, _pausable in ctx.said if p == "line_up" and m == "Ready. Open the container?"]
        assert len(asked) >= 4
        monkeypatch.setattr(node, "goal_brings_token", lambda: False)
        ctx = FakeCtx()
        threading.Timer(4.6, ctx.events.append, args=(OPEN_PRESS,)).start()
        assert cap.execute("open_box", {}, ctx)["lid_off"]
        asked = [m for p, _pct, m, _pausable in ctx.said if p == "line_up" and m == "Ready. Open the container?"]
        assert len(asked) == 1
    finally:
        cap.destroy_node()
        LOGGER.handlers[:] = handlers


def test_kept_alive_sends_the_newest_progress_again_only_after_its_period_of_silence():
    clock = [0.0]
    ctx = FakeCtx()
    kept = node.KeptAlive(ctx, period_s=30.0, clock=lambda: clock[0])
    kept.progress("line_up", 0.0, "Hold still, checking", pausable=False)
    clock[0] = 29.0
    assert kept.cancelled() is False and kept.poll_ui_event() is None and len(ctx.said) == 1
    clock[0] = 30.0
    assert kept.cancelled() is False
    assert ctx.said == [("line_up", 0.0, "Hold still, checking", False)] * 2
    clock[0] = 45.0  # counted from the one sent again
    kept.poll_ui_event()
    assert len(ctx.said) == 2
    clock[0] = 60.0
    kept.poll_ui_event()
    assert len(ctx.said) == 3
    kept.progress("press", 0.35, "Pressing the button", pausable=False)  # something new: the count starts again
    clock[0] = 89.0
    kept.cancelled()
    assert len(ctx.said) == 4
    clock[0] = 90.0
    kept.cancelled()
    assert ctx.said[-1] == ("press", 0.35, "Pressing the button", False) and len(ctx.said) == 5
    # everything else is the context's own
    ctx.events.append(CANCEL_PRESS)
    assert kept.poll_ui_event() == CANCEL_PRESS
    ctx.cancel = True
    assert kept.cancelled() is True
    assert kept.update_user_panel(panel_json={"a": 1}) and ctx.panels == [{"a": 1}]
    # ... and a goal that has said nothing yet still has something to send
    ctx = FakeCtx()
    kept = node.KeptAlive(ctx, period_s=30.0, clock=lambda: clock[0])
    clock[0] += 30.0
    kept.cancelled()
    assert len(ctx.said) == 1 and ctx.said[0][3] is False


def test_how_the_action_ended_is_one_short_line_in_the_persons_words():
    assert node.end_says(result={"lid_off": True}) == (node.END_DONE, False)
    assert node.end_says(result={"lid_off": False}) == (node.END_NOT_OFF, False)
    assert node.end_says(cancelled=True) == ("Cancelled", False)
    said, failed = node.end_says(error=EngineError("not_lined_up", "many words for the log", recoverable=True))
    assert failed and said == "Not in place in time. Nothing was pressed. You can try again"
    said, failed = node.end_says(error=EngineError("timeout", "...", recoverable=True))
    assert failed and said == "Taking too long. The run was stopped. You can try again"
    said, _failed = node.end_says(error=EngineError("a_code_of_tomorrow", "...", recoverable=False))
    assert said == "The container was not opened. Ask someone to check the arm"
    # an end with the lid in the fingers says that, whatever else (review of 2026-11-19: a cancel during the
    # carry said only "Cancelled", and open_box was then not offered with no word why)
    held = "The lid is still in the fingers. Ask someone to check the arm"
    assert node.end_says(cancelled=True, holding=True) == ("Cancelled. " + held, True)
    assert node.end_says(error=EngineError("timeout", "...", recoverable=False), holding=True) == (
        "Stopped. " + held,
        True,
    )
    # an error may carry its own words, where its code's are not true of it
    error = node.saying(EngineError("not_lined_up", "...", recoverable=True), node.END_NO_ANSWER)
    assert node.end_says(error=error) == ("No answer in time. Nothing was pressed. You can try again", True)
    # no line claims what its code does not always mean (the review's three)
    assert "stopped where it is" not in " ".join([node.END_CANCELLED, *node.END_SAYS.values()])
    assert "reach" not in node.END_SAYS["no_press"] and "othing moved" not in node.END_SAYS["not_home"]
    # every way a run's last words are read has a line of its own, bar the one that names a fault inside
    assert {code for _pattern, code, _recoverable in node.STOP_CODES} - set(node.END_SAYS) == {"watch_failed"}
    for code in node.END_SAYS:
        for recoverable in (True, False):
            said, failed = node.end_says(error=EngineError(code, "", recoverable=recoverable))
            assert failed and len(said) <= 100 and "\n" not in said
    # ... and so is every line an error carries for itself (saying)
    own = (node.END_NO_ANSWER, node.END_UI_GONE, node.END_NOT_SEEN, *node.END_NO_CURE.values(), node.END_DISAGREE)
    for sentence in own:
        for recoverable in (True, False):
            error = node.saying(EngineError("failed", "", recoverable=recoverable), sentence)
            said, failed = node.end_says(error=error)
            assert failed and said.startswith(sentence) and len(said) <= 100 and "\n" not in said


def test_the_end_is_not_held_on_the_panel_for_nobody(ros, monkeypatch):
    """Not while the node is being stopped (the goal's result must be out inside its stop's wait), and no
    longer than END_UNSHOWN_S where the bridge does not hold the panel (it refuses the goal's token)."""
    handlers = list(LOGGER.handlers)
    cap = kept_alive(monkeypatch)
    try:
        monkeypatch.setattr(node, "END_SHOW_S", 5.0)  # as flown
        assert node.END_UNSHOWN_S == 1.5
        monkeypatch.setattr(node, "END_UNSHOWN_S", 0.3)
        ctx = cap._kept = FakeCtx()
        ctx._run = _RunOf(held=False)
        t0 = time.monotonic()
        cap._show_end("Cancelled", False)
        assert 0.3 <= time.monotonic() - t0 < 1.0
        ctx = cap._kept = FakeCtx()
        cap._on_shutdown()
        t0 = time.monotonic()
        cap._show_end("Cancelled", False)
        assert time.monotonic() - t0 < 0.2 and ctx.panels == []
    finally:
        cap.destroy_node()
        LOGGER.handlers[:] = handlers


def test_shown_is_the_bridges_goal_and_its_own_word_on_its_app(ros, monkeypatch):
    handlers = list(LOGGER.handlers)
    cap = kept_alive(monkeypatch)
    try:
        ctx = FakeCtx()
        ctx._run = _RunOf(held=True)
        assert cap._shown(ctx)  # the bridge holds the panel, and its word has not been heard in this action
        cap._ue_heard = (time.monotonic(), True)
        assert cap._shown(ctx)
        cap._ue_heard = (time.monotonic(), False)  # the app is not connected: nobody sees the panel
        assert not cap._shown(ctx)
        assert node.UE_SILENT_S == 5.0
        cap._ue_heard = (time.monotonic() - 5.1, True)  # the bridge has said nothing for 5 s: hung, or gone
        assert not cap._shown(ctx)
        cap._ue_heard = (time.monotonic(), True)
        ctx._run.ui_goal_handle = None  # refused, or a press took it down
        assert not cap._shown(ctx)
    finally:
        cap.destroy_node()
        LOGGER.handlers[:] = handlers


def test_a_goal_cut_120_s_after_it_came_turns_the_long_wait_off_for_this_node(ros, monkeypatch):
    """The orchestrator that does not restart its clock: a cancel nobody here asked for, 120 s after the goal
    came, of a goal whose progress was being sent. The first cut cannot be undone; there is no second."""
    import threading

    handlers = list(LOGGER.handlers)
    cap = kept_alive(monkeypatch, mock_phase_s=5.0)
    try:
        assert node.CLOCK_FROM_GOAL_S == (117.0, 126.0)
        monkeypatch.setattr(node, "CLOCK_FROM_GOAL_S", (0.5, 1.5))  # 120 s, in a test's time
        ctx = FakeCtx()
        threading.Timer(0.8, setattr, args=(ctx, "cancel", True)).start()
        with pytest.raises(Cancelled):
            cap.execute("open_box", {}, ctx)
        assert cap._clock_from_goal is True and cap._ui_wait_s() == 0.0
        # the next goal is under the 120 s rule again: its line-up has what those leave, no end panel
        monkeypatch.setattr(node, "LINE_UP_S", 0.6)
        monkeypatch.setattr(node.line_up, "LINGER_S", 0.3)
        ctx = FakeCtx()
        with pytest.raises(EngineError) as e:
            cap.execute("open_box", {}, ctx)
        assert (
            e.value.error_code == "not_lined_up"
            and ctx.panels[-1]["message"] == "Press open again when you are in place"
        )
    finally:
        cap.destroy_node()
        LOGGER.handlers[:] = handlers


def test_the_orchestrators_goal_with_no_ui_as_it_comes_keeps_the_35_s_it_always_had(ros, monkeypatch):
    """The bridge's server not there (yet) as the goal comes: it is asked on the console of a node nobody
    sits at. At 21aa7cc that line-up ended 35 s after the goal; it does not wait `ui_line_up_s` there."""
    handlers = list(LOGGER.handlers)
    cap = kept_alive(monkeypatch, mock_phase_s=5.0)
    try:
        monkeypatch.setattr(cap, "_ui_live", lambda: False)
        cap._assigned_token = bytes([7]) * 16  # ... of a goal that brought the orchestrator's token
        monkeypatch.setattr(node, "LINE_UP_S", 0.6)
        monkeypatch.setattr(node.line_up, "LINGER_S", 0.2)
        ctx = FakeCtx()
        t0 = time.monotonic()
        with pytest.raises(EngineError) as e:
            cap.execute("open_box", {}, ctx)
        assert e.value.error_code == "not_lined_up" and "within 1 s" in str(e.value)
        assert time.monotonic() - t0 < 3.0 and ctx.panels == [] and cap._kept is None
    finally:
        cap.destroy_node()
        LOGGER.handlers[:] = handlers


def test_a_person_in_place_who_does_not_answer_is_not_told_they_were_not_in_place(ros, monkeypatch):
    handlers = list(LOGGER.handlers)
    cap = kept_alive(monkeypatch, mock_phase_s=0.2, ui_line_up_s=3.6)  # asked at about 2.5 s; nobody answers
    try:
        monkeypatch.setattr(node.line_up, "LINGER_S", 0.3)
        ctx = FakeCtx()
        with pytest.raises(EngineError) as e:
            cap.execute("open_box", {}, ctx)
        assert e.value.error_code == "not_lined_up" and e.value.says == node.END_NO_ANSWER
        assert ctx.panels[-1]["message"] == "No answer in time. Nothing was pressed. You can try again"
    finally:
        cap.destroy_node()
        LOGGER.handlers[:] = handlers


# --- ... on the REAL path (its run FAKE_RUN): the tests above are all the mock's (review of 2026-11-19) ---


def _kept_real(engine, monkeypatch, end_s=0.2):
    """The real node for a goal of the orchestrator whose clock restarts, with the UI, as its deployment says."""
    monkeypatch.setattr(node, "goal_brings_token", lambda: True)
    monkeypatch.setattr(node, "END_SHOW_S", end_s)
    monkeypatch.setattr(engine, "_ui_live", lambda: True)
    engine.set_parameters([Parameter("ui_line_up_s", value=300.0)])


def test_a_real_runs_budget_starts_at_the_persons_yes_under_such_an_orchestrator(engine, monkeypatch, tmp_path):
    """A yes that comes after more than the run's whole budget still gets a run with its whole budget: guided
    in the middle of the run, lined up before it with the run watching, and with the scene camera's watch.
    (With the 120 s rule's arithmetic left in place here every test of the mock's still passed.)"""
    import threading

    _kept_real(engine, monkeypatch)
    monkeypatch.setattr(node, "RUN_BUDGET_S", 3.0)
    monkeypatch.setenv("WAIT_S", "1.0")

    def check(**env):
        _forget_run_logs(tmp_path)
        given(monkeypatch, FULL_RUN, wait=True)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        ctx = FakeCtx()
        threading.Timer(4.0, ctx.events.append, args=(OPEN_PRESS,)).start()
        t0 = time.monotonic()
        out = engine.execute("open_box", {}, ctx)
        assert out["lid_off"] and time.monotonic() - t0 >= 4.0
        assert ctx.panels[-1]["message"] == node.END_DONE

    _guided(engine, monkeypatch)
    check(GUIDE="1")  # guided in the middle of the run
    monkeypatch.delenv("GUIDE")
    engine.set_parameters([Parameter("line_up", value=True), Parameter("wrist_first", value=True)])
    check()  # lined up before it, the run itself watching
    engine.set_parameters([Parameter("wrist_first", value=False)])
    monkeypatch.setattr(engine, "_box_source", lambda env, instance_id, watch=None, refused=(): InPlace())
    check()  # ... and with the scene camera's watch, the run started after


def test_the_orchestrator_is_told_while_a_real_run_flies_in_silence_and_while_its_wrist_turns_back(
    engine, monkeypatch, tmp_path
):
    _kept_real(engine, monkeypatch)
    monkeypatch.setattr(node, "KEEP_ALIVE_S", 0.3)
    given(monkeypatch, FULL_RUN[:3], wait=True)  # three rows, then silence
    monkeypatch.setenv("WAIT_S", "2.5")
    ctx = FakeCtx()
    out = engine.execute("open_box", {}, ctx)
    assert out["lid_off"] is False and ctx.panels[-1]["message"] == node.END_NOT_OFF
    last = ctx.said[-1]
    assert sum(1 for said in ctx.said if said == last) >= 5
    # ... and while the wrist turns back after a line-up the person did not get in place for
    _forget_run_logs(tmp_path)
    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("LINE_UP", "far")
    monkeypatch.setenv("TURN_BACK_S", "2.0")
    _wrist_lining_up(engine, monkeypatch, proven=False)
    engine.set_parameters([Parameter("ui_line_up_s", value=1.0)])
    monkeypatch.setattr(node.line_up, "LINGER_S", 0.2)
    ctx = FakeCtx()
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert e.value.error_code == "not_lined_up"
    assert len([m for _p, _pct, m, _pausable in ctx.said if m == node.PANEL_TURNING_BACK]) >= 4
    assert ctx.panels[-1]["message"].startswith("Not in place in time")


def test_a_real_guidance_whose_panel_the_ui_stopped_showing_turns_the_wrist_back_and_is_carried_over(
    engine, monkeypatch
):
    _kept_real(engine, monkeypatch)
    monkeypatch.setattr(node, "UI_GONE_S", 0.6)
    monkeypatch.setattr(node.line_up, "LINGER_S", 0.2)
    given(monkeypatch, FULL_RUN)
    monkeypatch.setenv("GUIDE", "1")
    monkeypatch.setenv("TURN_BACK_S", "0.5")
    engine.set_parameters([Parameter("wrist_first", value=True)])
    _made, told, runs = _guided(engine, monkeypatch)
    sent = []
    real = node.Run.interrupt
    monkeypatch.setattr(node.Run, "interrupt", lambda run: (sent.append(signal.SIGINT), real(run)))
    ctx = FakeCtx()
    ctx._run = _RunOf(held=False)  # every send refused
    with pytest.raises(EngineError) as e:
        engine.execute("open_box", {}, ctx)
    assert e.value.error_code == "not_lined_up" and "the UI stopped showing" in str(e.value)
    assert e.value.says == node.END_UI_GONE  # not "not in place": the person was
    # told its line-up is over, it turned its wrist back and ended its own way: no Ctrl+C
    assert told == [{"stop": True}] and sent == [] and runs[0].proc.returncode == 5 and engine._run is None
    assert any(m == node.PANEL_TURNING_BACK for _p, _pct, m, _pausable in ctx.said)
    assert engine._line_up_first is not None  # the next open_box goes on from the line-up, as after a time-out


def test_a_cancel_with_the_lid_in_the_fingers_says_so_on_the_panel(engine, monkeypatch):
    import threading

    _kept_real(engine, monkeypatch)
    given(monkeypatch, FULL_RUN[:11], wait=True)  # ... to the grip that held the lid, then it flies on
    engine.instance_id = "c1"
    ctx = FakeCtx()

    def cancel(*_by):
        engine._person_cancelled = True
        ctx.cancel = True

    monkeypatch.setattr(engine, "_cancel_from_ui", cancel)
    threading.Timer(1.5, ctx.events.append, args=(CANCEL_PRESS,)).start()
    with pytest.raises(Cancelled):
        engine.execute("open_box", {}, ctx)
    assert ctx.world[-1] == {"gripper": {"holding": "c1:lid"}}
    assert ctx.panels[-1] == {
        "title": node.PANEL_TITLE,
        "message": "Cancelled. The lid is still in the fingers. Ask someone to check the arm",
        "buttons": [],
        "style": "error",
    }


class _RunOf:
    """The base class's own run, as far as the engine reads it: the goal the bridge holds for the panel."""

    def __init__(self, held=True):
        self.ui_goal_handle = object() if held else None


def test_a_line_up_whose_panel_the_ui_stopped_showing_ends_in_seconds_not_after_its_five_minutes(ros, monkeypatch):
    """The rig's bridge_restart (2026-11-19): the bridge came back refusing the goal's token, and the line-up
    asked "Ready?" of no one for the rest of its time. Twenty seconds without the panel on the bridge, and it
    ends recoverable; a panel the bridge holds, or one a press took down for a moment, never ends it."""
    import threading

    handlers = list(LOGGER.handlers)
    cap = kept_alive(monkeypatch, mock_phase_s=0.2)
    try:
        assert node.UI_GONE_S == 20.0
        monkeypatch.setattr(node, "UI_GONE_S", 0.6)
        monkeypatch.setattr(node.line_up, "LINGER_S", 0.3)
        ctx = FakeCtx()
        ctx._run = _RunOf(held=False)  # every send refused: the bridge holds no goal of this run
        t0 = time.monotonic()
        with pytest.raises(EngineError) as e:
            cap.execute("open_box", {}, ctx)
        assert e.value.error_code == "not_lined_up" and e.value.recoverable
        assert "the UI stopped showing" in str(e.value) and "nothing was pressed" in str(e.value)
        assert 0.6 <= time.monotonic() - t0 < 0.6 + 0.3 + 0.4 + 2.0  # the wait, the linger, the end's panel
        assert {p for p, *_ in ctx.said} == {"line_up"} and ctx.objects == {} and ctx.world == []
        # held: the line-up goes on to its question and runs on Open, though the panel was down for a moment
        ctx = FakeCtx()
        ctx._run = _RunOf(held=True)
        threading.Timer(1.0, setattr, args=(ctx._run, "ui_goal_handle", None)).start()
        threading.Timer(1.3, setattr, args=(ctx._run, "ui_goal_handle", object())).start()
        threading.Timer(3.2, ctx.events.append, args=(OPEN_PRESS,)).start()
        assert cap.execute("open_box", {}, ctx)["lid_off"]
        # ... and the UI itself going (the bridge's server no longer there) is the same end
        live = [True]
        monkeypatch.setattr(cap, "_ui_live", lambda: live[0])
        ctx = FakeCtx()
        threading.Timer(0.5, live.__setitem__, args=(0, False)).start()
        with pytest.raises(EngineError) as e:
            cap.execute("open_box", {}, ctx)
        assert e.value.error_code == "not_lined_up" and "the UI stopped showing" in str(e.value)
    finally:
        cap.destroy_node()
        LOGGER.handlers[:] = handlers


def test_the_panel_watch_is_the_sources_own_but_for_a_ui_gone_too_long():
    class Source:
        unable = ""
        why_none = "no container seen"

        def look(self, now):
            return ("fix", "reach")

    clock, shown, source = [0.0], [True], Source()
    watch = node.PanelWatch(source, lambda: shown[0], gone_s=20.0, clock=lambda: clock[0])
    assert watch.unable == "" and watch.look(0.0) == ("fix", "reach") and watch.why_none == "no container seen"
    shown[0], clock[0] = False, 19.9
    assert watch.unable == "" and not watch.lost
    shown[0], clock[0] = True, 25.0  # back in time: counted afresh from here
    assert watch.unable == ""
    shown[0], clock[0] = False, 44.9
    assert watch.unable == "" and not watch.lost
    clock[0] = 45.0
    assert watch.unable == node.UI_GONE and watch.lost
    shown[0] = True  # once lost, lost: the line-up is ending
    assert watch.unable == node.UI_GONE
    # the source's own word comes first, and is not the UI's going
    watch = node.PanelWatch(source, lambda: True, gone_s=20.0, clock=lambda: clock[0])
    source.unable = "the arm is not at its rest pose"
    assert watch.unable == "the arm is not at its rest pose" and not watch.lost


def test_every_process_the_node_starts_is_handed_the_nodes_own_reading_of_the_rest_pose(engine, monkeypatch, tmp_path):
    """The reviewer's 3 (2026-11-19): the node's children read the rest pose's file for themselves, so a file
    that changed, or a child started with another, rested at another pose unseen. They are handed the file the
    node resolves AND the digest of the seven numbers the node reads there; BOX_OPENING_REST_POSE in the node's
    environment beats the param; a file named that cannot be read fails the goal before anything is started."""
    from box_opening.runtime import rest

    # with the environment naming a pose (as the tests' own does: ""), the param is not what is handed on
    engine.set_parameters([Parameter("rest_pose", value="/somewhere/arm_presets.yaml")])
    assert engine._rest_args()[:2] == ["--rest-pose", ""]
    monkeypatch.delenv(rest.REST_POSE_ENV)
    real = node.Capability._argv.__get__(engine)
    c = [0.4238, -1.1685, -2.8871, -1.8516, -0.856, -1.0008, 2.2684]
    path = tmp_path / "arm_presets_C.yaml"
    path.write_text("/**:\n  ros__parameters:\n    presets.home: %r\n" % c)
    engine.set_parameters([Parameter("rest_pose", value=str(path))])
    assert engine._rest_args() == ["--rest-pose", str(path), "--rest-pose-digest", rest.digest(c)]
    argv = real("open_box", "food_container_2")
    i = argv.index("--rest-pose")
    assert argv[i : i + 4] == engine._rest_args() and argv[-1] == "--execute"
    # the numbers are read as each run is started: a file that changed since is handed on as it now reads
    b = [0.5376, -1.2584, -2.7691, -1.9909, -1.0247, -1.2317, 2.4847]
    path.write_text("/**:\n  ros__parameters:\n    presets.home: %r\n" % b)
    assert engine._rest_args()[-1] == rest.digest(b) != rest.digest(c)
    # the environment beats the param: every box opening container of a deployment is given the one line
    other = tmp_path / "other.yaml"
    other.write_text("/**:\n  ros__parameters:\n    presets.home: %r\n" % c)
    monkeypatch.setenv(rest.REST_POSE_ENV, str(other))
    assert engine._rest_args() == ["--rest-pose", str(other), "--rest-pose-digest", rest.digest(c)]
    monkeypatch.delenv(rest.REST_POSE_ENV)
    # a file named and not there, or with a pose the driver would not start on: not_ready, nothing started
    engine.set_parameters([Parameter("rest_pose", value=str(tmp_path / "nowhere.yaml"))])
    with pytest.raises(EngineError) as e:
        real("open_box")
    assert e.value.error_code == "not_ready" and e.value.recoverable
    assert "rest pose: cannot read" in str(e.value) and "nothing ran" in str(e.value)
    monkeypatch.setattr(node, "Run", lambda *a, **k: pytest.fail("a run was started"))
    with pytest.raises(EngineError) as e:  # the goal itself: refused before a line-up or a run is started
        engine.execute("open_box", {}, FakeCtx())
    assert e.value.error_code == "not_ready" and "rest pose: cannot read" in str(e.value)
    path.write_text("/**:\n  ros__parameters:\n    presets.home: [0, 1, 2, -2, 0, 1, 2]\n")
    engine.set_parameters([Parameter("rest_pose", value=str(path))])
    with pytest.raises(EngineError, match="whole number among its angles"):
        engine._rest_args()


def test_the_legs_of_a_lid_taken_on_or_back_are_not_said_as_lifting_it_off():
    """The set-down that feels no table (2026-11-19: "it should keep trying spots") flies 'lid:' legs long
    after the lid is off; the panel's words for them say what the arm is doing then."""

    def says(leg):
        return next(doing for start, _phase, _pct, doing in node.LEG_PHASES if leg.startswith(start))

    assert says("lid:lift") == says("lid:carry") == "lifting the lid off"
    assert says("lid:back-up") == "no table felt there: taking the lid to another spot"
    assert says("lid:back-over") == "taking the lid back to the container"
    assert says("lid:release") == "letting go of the lid on the container"
    # ... and the fraction done never goes backwards from a set-down's touch to the next spot's carry
    pct = {start: p for start, _phase, p, _doing in node.LEG_PHASES}
    assert pct["place:down"] <= pct["lid:back-up"] <= pct["place:touch"] == pct["lid:back-"]
