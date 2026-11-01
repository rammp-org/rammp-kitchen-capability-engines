"""press_demo.main() driven end to end with every ROS, camera and planner
edge faked: no rclpy.init, nothing talks to DDS or the arm. run_main is the
seam; the reach-out (reach_press.run_reach_press) is scripted per test, and
everything the run writes lands in tmp_path."""

import json
import sys

import pytest
from conftest import FakeClient

PINK = "capabilities/box_opening/config/containers/ankou_pink.yaml"
BENCH = "capabilities/box_opening/config/world_bench.yaml"
BOX_XY = (0.45, -0.10)
TOUCH = ("touch", {"message": "contact", "progress": 0.97, "torque_peak": 3.2})


class _Node:
    def get_clock(self):
        class C:
            def now(self):
                class T:
                    nanoseconds = 0

                return T()

        return C()

    def get_logger(self):
        class L:
            def error(self, *a):
                pass

        return L()


class _Exec:
    def add_node(self, n):
        pass

    def spin(self):
        pass

    def shutdown(self):
        pass

    def remove_node(self, n):
        pass


class _Grab:
    color = None
    depth = None
    k = None
    color_stamp = None

    def missing(self):
        return []


class _Watcher:
    def __init__(self, node, cfg, model, table_z, base_frame=None, camera_cfg_name=None):
        self.model = model
        self.table_z = table_z
        self.grab = _Grab()
        self.roi = None
        self.active = False
        self.last_aim = None
        self.last_reject = None

    def to_container_pose(self, got):
        from box_opening.perception.depth_source import container_pose_from_top

        return container_pose_from_top(got[0], float(got[1]), self.model, self.table_z)

    def status(self):
        return "3/3 still frames found the button circle"

    def last_coarse(self):
        return None

    def coarse_fix(self, now=None):
        return None


LOCATES = []  # every timeout the scene locator was asked for, in order
HOME_FIRSTS = []  # the keywords of every home_first call, in order
REACH_OUTS = []  # the scene fixes handed to the reach-out


def run_main(monkeypatch, tmp_path, argv, approach, client=None, found=True):
    """press_demo.main() with `approach(ctx)` standing in for the reach-out:
    its result's `failed` decides the exit code; `found` False, the scene
    camera finds no box. Returns (exit code, client, run-log rows)."""
    import faulthandler

    import rclpy.executors

    from box_opening import worlds as worlds_mod
    from box_opening.models.container import ContainerModel
    from box_opening.perception import owl_source, scene, scene_calib, scene_source
    from box_opening.perception.depth_source import container_pose_from_top
    from box_opening.tasks import cli_common, press_demo

    client = client or FakeClient()
    model = ContainerModel.load(PINK)
    table_z = -0.057 + 0.03  # world_bench.yaml's table top
    top = (BOX_XY[0], BOX_XY[1], table_z + model.dims[2])

    class _Fix:
        pose = container_pose_from_top(top, 0.0, model, table_z)
        top_xyz = top
        n_points = 300
        score = 0.6
        table_z_scene = table_z
        table_extent = None

    class _Locator:
        last_why = None if found else "the OWL's 3 candidates refused"
        moved_note = None
        last_timing = {}
        grab = None

        def __init__(self, *a, **k):
            self.owl = type("O", (), {"enable": lambda s: None})()

        def prime(self):
            pass

        def locate(self, timeout):
            LOCATES.append(timeout)
            return _Fix if found else None

        def base_to_link(self):
            return None

    class _Rclpy:
        @staticmethod
        def create_node(name):
            return _Node()

        @staticmethod
        def spin_once(node, timeout_sec=0.0):
            return None

    monkeypatch.setattr(sys, "argv", ["press_demo", *argv, "--container", PINK, "--bench-world", BENCH])
    monkeypatch.setattr(cli_common, "init_runtime", lambda execute: (_Node(), client))
    monkeypatch.setattr(press_demo, "rclpy", _Rclpy)
    monkeypatch.setattr(faulthandler, "enable", lambda *a, **k: None)
    monkeypatch.setattr(rclpy.executors, "SingleThreadedExecutor", _Exec)
    monkeypatch.setattr(owl_source, "announce_container", lambda node, path: None)
    monkeypatch.setattr(
        owl_source, "OwlRung", lambda *a, **k: type("Owl", (), {"enable": lambda s: None, "disable": lambda s: None})()
    )
    monkeypatch.setattr(press_demo, "BoxTopWatcher", _Watcher)
    identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    monkeypatch.setattr(scene, "load_scene_calibration", lambda p=None: identity)
    monkeypatch.setattr(scene_source, "SceneLocator", _Locator)
    monkeypatch.setattr(scene_calib, "scene_tag_px", lambda p: None)
    monkeypatch.setattr(press_demo, "readiness_refusal", lambda *a, **k: None)
    monkeypatch.setattr(press_demo, "home_first", lambda *a, **k: HOME_FIRSTS.append(k) or True)
    monkeypatch.setattr(press_demo, "MissionFrames", lambda *a, **k: None)
    monkeypatch.setattr(press_demo, "CAPTURES_DIR", tmp_path / "captures")
    monkeypatch.setattr(press_demo, "residuals_path_for", lambda p: tmp_path / "residuals.jsonl")
    monkeypatch.setattr(press_demo, "dump_wrist_frame", lambda w, t: None)
    from box_opening.tasks import reach_press

    def fake_reach_out(node, ctx, cfg, runner, watcher, args, scene_fix, locator):
        # every scene-camera box goes to the reach-out (2026-10-28); what it
        # flies is test_reach_press's: here it ends as the scripted approach says
        ctx.cpose = scene_fix.pose
        REACH_OUTS.append(scene_fix)
        got = approach(ctx)
        return 1 if getattr(got, "failed", False) else 0

    monkeypatch.setattr(reach_press, "run_reach_press", fake_reach_out)
    real_init = worlds_mod.WorldStore.__init__
    monkeypatch.setattr(
        worlds_mod.WorldStore,
        "__init__",
        lambda self, path, out_dir=None: real_init(self, path, out_dir=tmp_path / "worlds"),
    )
    monkeypatch.setenv("BOX_OPENING_STATE", str(tmp_path))  # the Runner's log, the gripper file
    with pytest.raises(SystemExit) as e:
        press_demo.main()
    rows = []
    for f in sorted((tmp_path / "runs").glob("run-*.jsonl")):
        rows += [json.loads(line) for line in open(f)]
    return e.value.code, client, rows


def _at_staging(button_up_mm=0.0):
    """The scene approach as it ends at staging over the scene's box, with
    the first aim landing on the button (no pre-planned descent)."""

    def approach(ctx):
        from types import SimpleNamespace

        from box_opening.models.container import from_container

        b = from_container(ctx.cpose, ctx.model.button_offset)
        return SimpleNamespace(got=((b[0], b[1], b[2]), 0.0), preplanned=None, status="3/3", button_up_mm=button_up_mm)

    return approach


def test_the_first_look_for_the_box_keeps_at_it_longer_than_a_re_check(monkeypatch, tmp_path):
    """Flatbot 2026-10-27 15:34:01: the scene camera refused the OWL's three
    candidates in 5 s and the run went home; the next run had the box at
    once. The first look, before anything moves, gets SCENE_FIRST_LOCATE_S."""
    from box_opening.tasks import press_demo

    LOCATES.clear()
    client = FakeClient()
    client.exec_script = [TOUCH]
    REACH_OUTS.clear()
    run_main(monkeypatch, tmp_path, ["--execute", "--press-only"], _at_staging(), client=client)
    assert LOCATES[0] == press_demo.SCENE_FIRST_LOCATE_S > press_demo.SCENE_LOCATE_S
    assert len(REACH_OUTS) == 1  # a box within the straight-down staging reach goes to the reach-out too


def test_a_run_whose_scene_camera_finds_no_box_says_so_in_its_record(monkeypatch, tmp_path):
    """NO BOX is printed and the arm goes home: the record now says why the
    reach-out never started, before the home it flies (reach_refused)."""
    REACH_OUTS.clear()
    code, _c, rows = run_main(monkeypatch, tmp_path, ["--execute"], _at_staging(), found=False)
    assert code == 2 and not REACH_OUTS
    refused = [r for r in rows if r["leg"] == "reach_refused"]
    assert [r["why"] for r in refused] == ["NO BOX — scene camera: the OWL's 3 candidates refused"]
    assert rows.index(refused[0]) < len(rows) - 1 and rows[-1]["kind"] == "motion"  # then the home


class _Hears(FakeClient):
    """A client that keeps what it heard and when it sent, as PlannerClient
    does (nothing heard here): each execution says when it went out, so its
    arrival's settle is due."""

    def execute(self, traj_, speed, guard=None, while_running=None, stop_when=None, gains=None):
        import time

        outcome, info = super().execute(traj_, speed, guard, while_running, stop_when, gains)
        info.update(sent_at=time.monotonic(), sent_traj=traj_, feedback_q=[])
        return outcome, info

    def joint_history(self, since=None):
        return []

    def goals_sent(self):
        return []


def test_the_last_arrivals_settle_is_noted_as_the_run_ends(monkeypatch, tmp_path):
    """Nothing after a run's last motion goes out to note its settle: main's
    exit does (runner.flush), the NO BOX run's home's after its own row, cut
    where the run ended."""
    code, _c, rows = run_main(monkeypatch, tmp_path, ["--execute"], _at_staging(), client=_Hears(), found=False)
    assert code == 2 and rows[-2]["kind"] == "motion"
    assert rows[-1]["leg"] == "settle" and rows[-1]["cut"] == "run end" and rows[-1]["legs"][-1] == rows[-2]["leg"]


def _refused(ctx):
    """The scene approach refused: the run goes home and stops (exit 1),
    right after the start these tests are about."""
    from types import SimpleNamespace

    return SimpleNamespace(failed=True, status="the approach to staging was refused (IK_FAIL)")


def test_a_compliant_depth_press_run_starts_from_the_spring_hold_it_finds(monkeypatch, tmp_path):
    """Flatbot 2026-10-27 16:24: the run after a compliant run found the arm
    0.03 rad off HOME, the joint spring's hold, and flew a POSITION home
    before anything else. A default --depth-press run (compliant, the scene
    camera's reach-out) takes a start within REST_TOL_RAD as there."""
    from box_opening.constants import REST_TOL_RAD

    HOME_FIRSTS.clear()
    code, _c, _rows = run_main(monkeypatch, tmp_path, ["--execute", "--depth-press"], _refused)
    assert code == 1
    assert [k.get("tol") for k in HOME_FIRSTS] == [REST_TOL_RAD]


def test_every_other_run_still_starts_at_its_exact_rest_pose(monkeypatch, tmp_path):
    """--guarded reads its at-rest reference at the rest pose and checks
    gravity against it: it keeps AT_REST_RAD. A bare --execute is the
    compliant depth press since 2026-10-28, as --depth-press was: it takes
    REST_TOL_RAD."""
    from box_opening.constants import REST_TOL_RAD
    from box_opening.tasks import press_demo

    HOME_FIRSTS.clear()
    code, _c, _rows = run_main(monkeypatch, tmp_path / "bare", ["--execute"], _refused)
    assert code == 1 and [k.get("tol") for k in HOME_FIRSTS] == [REST_TOL_RAD]
    for i, argv in enumerate((["--execute", "--depth-press", "--guarded"],)):
        HOME_FIRSTS.clear()
        code, _c, _rows = run_main(monkeypatch, tmp_path / str(i), argv, _refused)
        assert code == 1, argv
        assert [k.get("tol") for k in HOME_FIRSTS] == [press_demo.AT_REST_RAD], argv


def test_the_flags_of_the_one_run(monkeypatch, tmp_path, capsys):
    """2026-10-28, the staging mode scrapped: --depth-press is accepted,
    hidden and without effect (older command lines, sheppy's tasks);
    --no-scene, whose press was the wrist search's, is gone; --detect-only
    and --wrist-detect are one or the other."""
    code, _c, _rows = run_main(monkeypatch, tmp_path / "help", ["--help"], _refused)
    shown = capsys.readouterr().out
    usage = shown[: shown.index("\n\n")]
    options = {ln.split()[0].rstrip(",") for ln in shown.splitlines() if ln.strip().startswith("--")}
    assert code == 0
    assert {"--press-only", "--guarded", "--detect-only", "--wrist-detect"} <= options
    assert "--depth-press" not in options and "--depth-press" not in usage and "--no-scene" not in options
    for i, argv in enumerate((["--no-scene"], ["--detect-only", "--wrist-detect"])):
        code, _c, _rows = run_main(monkeypatch, tmp_path / str(i), argv, _refused)
        assert code == 2, argv  # argparse refuses it: nothing started
        assert ("unrecognized arguments" if i == 0 else "not allowed with") in capsys.readouterr().err


def test_the_wrist_mount_observation_flies_the_look_pose_from_the_exact_rest_pose(monkeypatch, tmp_path, capsys):
    """--wrist-detect (was --detect-only --no-scene): the scene camera is not
    asked, the run starts at its exact rest pose (AT_REST_RAD), flies the
    look pose and reports the wrist's fixes there."""
    from box_opening.tasks import press_demo

    reports = []
    monkeypatch.setattr(press_demo, "detect_only_report", lambda *a, **k: reports.append(a[5]))
    LOCATES.clear()
    HOME_FIRSTS.clear()
    REACH_OUTS.clear()
    code, _c, _rows = run_main(monkeypatch, tmp_path, ["--wrist-detect"], _refused)
    assert code == 0 and reports == [False]  # a preview: not --execute
    assert not LOCATES and not REACH_OUTS
    assert [k.get("tol") for k in HOME_FIRSTS] == [press_demo.AT_REST_RAD]
    previewed = [ln.split()[0] for ln in capsys.readouterr().out.splitlines() if ln.split()[1:2] == ["motion"]]
    assert previewed == ["look"]


def test_the_perception_executor_shuts_down_whatever_the_viz_stop_does(monkeypatch, tmp_path):
    """The finally stops the viz first, then the perception executor: a
    second Ctrl+C in the viz's stop (KeyboardInterrupt regardless,
    runtime/abort) must not skip the executor's shutdown, or the
    interpreter tears rclpy down under a live spin. A run that unwinds from
    a Ctrl+C asks the viz for nothing more (last=False)."""
    from box_opening.tasks import press_demo

    calls = []
    monkeypatch.setattr(_Exec, "shutdown", lambda self: calls.append("shutdown"))

    def stop(last=True):
        calls.append(("stop", last))
        raise KeyboardInterrupt  # viz.stop lets none out (test_viz); the finally holds without that

    monkeypatch.setattr(press_demo, "stop_viz", stop)
    with pytest.raises(KeyboardInterrupt):
        run_main(monkeypatch, tmp_path, ["--press-only"], _at_staging())
    assert calls == [("stop", True), "shutdown"]
    calls.clear()
    monkeypatch.setattr(press_demo, "stop_viz", lambda last=True: calls.append(("stop", last)))

    def ctrl_c(ctx):
        raise KeyboardInterrupt

    code, _c, _rows = run_main(monkeypatch, tmp_path, ["--press-only"], ctrl_c)
    assert code == 130 and calls == [("stop", False), "shutdown"]


"""main()'s shape around the perception thread, read from its source (no
rclpy): once the thread spins, every exit — a failed startup included — must
pass through the finally that stops it, or the interpreter tears rclpy down
under a live executor and aborts ("terminate called without an active
exception", exit 134 instead of the run's own message or 130)."""

import ast  # noqa: E402
from pathlib import Path  # noqa: E402

from box_opening.tasks import press_demo  # noqa: E402


def _main_body():
    tree = ast.parse(Path(press_demo.__file__).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    return main.body


def _is_call(node, owner, attr):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == owner
    )


def _start_index(body):
    for i, stmt in enumerate(body):
        if any(_is_call(n, "pthread", "start") for n in ast.walk(stmt)):
            return i
    raise AssertionError("main() never starts the perception thread")


def _the_try(body):
    i = _start_index(body)
    assert i + 1 < len(body)
    return i, body[i + 1]


def test_the_try_opens_right_after_the_perception_thread_starts():
    i, nxt = _the_try(_main_body())
    assert isinstance(nxt, ast.Try), "line %d follows pthread.start(), not the try" % nxt.lineno
    final = ast.Module(body=nxt.finalbody, type_ignores=[])
    assert any(_is_call(n, "pexec", "shutdown") for n in ast.walk(final))
    # a startup Ctrl+C still exits 130
    assert any(isinstance(h.type, ast.Name) and h.type.id == "KeyboardInterrupt" for h in nxt.handlers)


def test_ctx_is_none_before_the_try():
    body = _main_body()
    i, _ = _the_try(body)
    assert any(
        isinstance(s, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "ctx" for t in s.targets)
        and isinstance(s.value, ast.Constant)
        and s.value.value is None
        for s in body[:i]
    )


def test_the_finally_reads_mission_frames_through_getattr():
    _, nxt = _the_try(_main_body())
    assert isinstance(nxt, ast.Try)
    final = ast.Module(body=nxt.finalbody, type_ignores=[])
    # no bare ctx.mission_frames: ctx may be None, startup having failed first
    assert not any(
        isinstance(n, ast.Attribute)
        and n.attr == "mission_frames"
        and isinstance(n.value, ast.Name)
        and n.value.id == "ctx"
        for n in ast.walk(final)
    )
    calls = [
        n
        for n in ast.walk(final)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "keep_mission_frames"
    ]
    assert len(calls) == 1
    arg = calls[0].args[0]
    assert isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "getattr"
    assert isinstance(arg.args[0], ast.Name) and arg.args[0].id == "ctx"
    assert arg.args[1].value == "mission_frames"
    assert len(arg.args) == 3 and arg.args[2].value is None


def test_no_frames_to_keep_is_a_no_op():
    assert press_demo.keep_mission_frames(None, SystemExit(2)) is None
    assert press_demo.keep_mission_frames(getattr(None, "mission_frames", None), None) is None
