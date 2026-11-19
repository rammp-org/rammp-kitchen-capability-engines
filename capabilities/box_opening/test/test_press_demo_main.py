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


# what the scene locator's last look is recorded as having seen (SceneLocator.look_record): the segmenter
# live, three of the scene camera's frames judged
LOOKED = {"timing": {"streams_s": 0.1}, "segmenter": {"state": "live", "frames": 3}}


def run_main(monkeypatch, tmp_path, argv, approach, client=None, found=True, record=None, why=None, home_first=None):
    """press_demo.main() with `approach(ctx)` standing in for the reach-out:
    its result's `failed` decides the exit code; `found` False, the scene
    camera finds no box (a list: what each look in turn finds, the last for
    every look after it); `record`: what its look is recorded as having seen
    (LOOKED), `why`: its word on having found none (the OWL's); `home_first` in
    place of the run's first home (the default: at rest already). Returns
    (exit code, client, run-log rows)."""
    import faulthandler

    import rclpy.executors

    from box_opening import worlds as worlds_mod
    from box_opening.models.container import ContainerModel
    from box_opening.perception import scene, scene_calib, scene_source
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

    finds = list(found) if isinstance(found, list) else [found]

    class _Locator:
        last_why = None if found is True else (why or "the OWL's 3 candidates refused")
        moved_note = None
        last_timing = {}
        grab = None

        def __init__(self, *a, **k):
            self.kwargs = k

        def prime(self):
            pass

        def locate(self, timeout):
            LOCATES.append(timeout)
            return _Fix if (finds.pop(0) if len(finds) > 1 else finds[0]) else None

        def base_to_link(self):
            return None

        def base_to_color(self):
            return None  # no TF here: nothing the scene camera saw ties a wrist find to a selection

        def look_record(self):
            return dict(LOOKED if record is None else record)

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
    monkeypatch.setattr(press_demo, "BoxTopWatcher", _Watcher)
    identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    monkeypatch.setattr(scene, "load_scene_calibration", lambda p=None: identity)
    monkeypatch.setattr(scene_source, "SceneLocator", _Locator)
    monkeypatch.setattr(scene_calib, "scene_tag_px", lambda p: None)
    monkeypatch.setattr(press_demo, "readiness_refusal", lambda *a, **k: None)
    monkeypatch.setattr(press_demo, "home_first", home_first or (lambda *a, **k: HOME_FIRSTS.append(k) or True))
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
        if getattr(got, "code", None) is not None:  # the reach-out's own answer, where the test scripts it
            return got.code
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


def rows_of(tmp_path):
    rows = []
    for f in sorted((tmp_path / "runs").glob("run-*.jsonl")):
        rows += [json.loads(line) for line in open(f)]
    return rows


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


def test_a_step_out_of_the_padding_that_stops_short_ends_the_run_as_an_arm_that_holds(monkeypatch, tmp_path):
    """RAMMP 1.5, 2026-11-19: the arm is parked inside the chair's padding, and a run's first motion is the step
    out of it, flown and judged where the arm stands (core.home_any's `fly`). Where that leaves it short of
    clear the arm HAS moved: the run ends with exit 1 (the node: arm_holds, from its STOP line), not with exit
    3, which says the start was refused and nothing moved (not_home, recoverable)."""
    from box_opening.primitives.core import StepOutStopped

    def stopped(*a, **k):
        raise StepOutStopped(
            "the arm stands 12 mm outside the chair and user's padding — the arm holds where it stands"
        )

    REACH_OUTS.clear()
    code, _c, _rows = run_main(monkeypatch, tmp_path, ["--execute"], _at_staging(), home_first=stopped)
    assert code == 1 and REACH_OUTS == []  # nothing of the run followed
    code, _c, _rows = run_main(monkeypatch, tmp_path, ["--execute"], _at_staging(), home_first=lambda *a, **k: False)
    assert code == 3  # a first home refused before anything moved: as before


def test_a_run_whose_scene_camera_finds_no_box_says_so_in_its_record(monkeypatch, tmp_path):
    """NO BOX is printed and the arm goes home: the record now says why the
    reach-out never started, before the home it flies (reach_refused)."""
    REACH_OUTS.clear()
    code, _c, rows = run_main(monkeypatch, tmp_path, ["--execute"], _at_staging(), found=False)
    assert code == 2 and not REACH_OUTS
    refused = [r for r in rows if r["leg"] == "reach_refused"]
    assert [r["why"] for r in refused] == ["NO BOX — scene camera: the OWL's 3 candidates refused"]
    assert rows.index(refused[0]) < len(rows) - 1 and rows[-1]["kind"] == "motion"  # then the home


def test_a_no_box_row_keeps_the_looks_seconds_and_the_segmenters_verdict(monkeypatch, tmp_path):
    """reach_refused on NO BOX keeps SceneLocator.look_record() too (sweep
    scene-g-05): which of the failures that wear the NO BOX face it was."""
    from box_opening.tasks import press_demo

    homes = []
    monkeypatch.setattr(press_demo, "try_home", lambda ctx, runner, execute, why: homes.append(why))
    code, _c, rows = run_main(monkeypatch, tmp_path, ["--execute"], _at_staging(), found=False)
    (refused,) = (r for r in rows if r["leg"] == "reach_refused")
    assert code == 2 and homes == [refused["why"]]
    assert refused["timing"] == {"streams_s": 0.1} and refused["segmenter"] == {"state": "live", "frames": 3}


def test_the_scene_fix_row_keeps_the_looks_seconds_and_the_segmenters_verdict(monkeypatch, tmp_path):
    """Sweep scene-g-05: 84 scene_fix rows carried none of the look's timing
    or the segmenter's verdict (printed only), so the freshness bounds could
    not be judged offline. The row now keeps SceneLocator.look_record()."""
    client = FakeClient()
    client.exec_script = [TOUCH]
    run_main(monkeypatch, tmp_path, ["--execute", "--press-only"], _at_staging(), client=client)
    (row,) = (r for r in rows_of(tmp_path) if r["leg"] == "scene_fix")
    assert row["timing"] == {"streams_s": 0.1} and row["segmenter"] == {"state": "live", "frames": 3}
    # ... and the box's height as this run measured it: the fix's top over the table it was read on
    assert row["box_height_mm"] == pytest.approx(1000.0 * (row["top_xyz"][2] - row["table_z_scene"]), abs=0.11)
    assert row["rough"] is False


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


def _lining_up(monkeypatch, answer):
    """press_demo.line_up_watch stood in for: it answers (went, wrist_turned, wrist_held); returns its calls."""
    from box_opening.tasks import press_demo

    calls = []

    def watch(node, ctx, cfg, runner, watcher, args, locator, rest_tol, ear=None, out=None, rest_any=False):
        calls.append((bool(args.wrist_first), rest_tol, len(HOME_FIRSTS), len(LOCATES)))
        return answer

    monkeypatch.setattr(press_demo, "line_up_watch", watch)
    return calls


def test_a_line_up_comes_before_anything_of_the_run_and_a_stop_ends_it_there(monkeypatch, tmp_path):
    """--line-up: the watch runs before the home, the fingers and the first look; ended without a go, the run
    exits EXIT_NOT_LINED_UP with nothing homed, looked for or pressed."""
    from box_opening.constants import REST_TOL_RAD
    from box_opening.runtime.abort import EXIT_NOT_LINED_UP

    HOME_FIRSTS.clear(), REACH_OUTS.clear(), LOCATES.clear()
    calls = _lining_up(monkeypatch, (False, False, False))
    code, client, _rows = run_main(monkeypatch, tmp_path, ["--execute", "--line-up", "--wrist-first"], _refused)
    assert code == EXIT_NOT_LINED_UP == 5
    assert calls == [(True, REST_TOL_RAD, 0, 0)]  # asked before any home or look
    assert HOME_FIRSTS == [] and REACH_OUTS == [] and LOCATES == [] and client.gripper_sent == []


def test_a_line_up_whose_wrist_turn_left_the_arm_holding_ends_the_run_with_its_own_exit(monkeypatch, tmp_path):
    """The line-up's wrist turn stopped part way, or the wrist did not turn back after a stop: line_up_watch
    ends the run EXIT_LINE_UP_HOLDS, and main lets that out as it is (the engine reads it: the arm holds, not
    "nothing moved"), with nothing homed, looked for or pressed after it."""
    from box_opening.runtime.abort import EXIT_LINE_UP_HOLDS, EXIT_NOT_LINED_UP
    from box_opening.tasks import press_demo

    def holds(*a, **k):
        sys.exit(EXIT_LINE_UP_HOLDS)

    HOME_FIRSTS.clear(), REACH_OUTS.clear(), LOCATES.clear()
    monkeypatch.setattr(press_demo, "line_up_watch", holds)
    code, client, _rows = run_main(monkeypatch, tmp_path, ["--execute", "--line-up", "--wrist-first"], _refused)
    assert code == EXIT_LINE_UP_HOLDS == 6 != EXIT_NOT_LINED_UP
    assert HOME_FIRSTS == [] and REACH_OUTS == [] and LOCATES == [] and client.gripper_sent == []


def test_after_a_go_the_run_goes_on_and_homes_first_unless_its_wrist_looked(monkeypatch, tmp_path):
    """A go from a line-up that turned the wrist: the arm stood at rest before it looked, and the run's own
    look goes on from that view, so no home first. From one that turned nothing, the run starts as always."""
    for i, (turned, homes) in enumerate(((True, 0), (False, 1))):
        HOME_FIRSTS.clear(), REACH_OUTS.clear()
        calls = _lining_up(monkeypatch, (True, turned, turned))
        code, _c, _rows = run_main(monkeypatch, tmp_path / str(i), ["--execute", "--line-up"], _refused)
        assert code == 1 and len(calls) == 1 and len(REACH_OUTS) == 1  # the run went on to its reach-out
        assert len(HOME_FIRSTS) == homes, turned


def test_after_a_go_from_a_line_up_whose_wrist_looked_the_first_scene_look_is_short(monkeypatch, tmp_path):
    """After Open the arm stood about 15 s before anything moved (both clean runs of 2026-11-12 spent the whole
    first look and the scene camera found nothing in front of the arm), which reads as a hang and comes out of
    the run's budget. Where the line-up's wrist held the container as the go came, that look only refreshes the
    selection's box and the hint. Every other start keeps the long first look: a run with no line-up, a
    line-up that turned nothing, and one whose wrist was turned and did not show the box (the scene camera's
    sightings brought the Ready, and the scene camera is then all the run has)."""
    from box_opening.tasks import press_demo

    cases = (((True, True, True), ["--line-up"]), ((True, True, False), ["--line-up"]),
             ((True, False, False), ["--line-up"]), ((True, False, False), []))  # fmt: skip
    for i, (answer, argv) in enumerate(cases):
        LOCATES.clear(), HOME_FIRSTS.clear(), REACH_OUTS.clear()
        _lining_up(monkeypatch, answer)
        run_main(monkeypatch, tmp_path / str(i), ["--execute", *argv], _refused)
        if answer[2]:
            assert LOCATES[0] <= 3.0, "the arm stands this long after Open"
            assert LOCATES[0] == press_demo.SCENE_LINED_UP_LOCATE_S
        else:
            assert LOCATES[0] == press_demo.SCENE_FIRST_LOCATE_S == 15.0, (answer, argv)


def _answers(codes):
    """The reach-out's answers in turn (reach_press.run_reach_press's: NO_WAY_IN, 0, 1), and what the run knew
    of the scene each time it was asked."""
    from types import SimpleNamespace

    seen = []

    def approach(ctx):
        seen.append((ctx.worlds.table_top_z, ctx.cpose is not None))
        return SimpleNamespace(code=codes.pop(0))

    return approach, seen


GUIDED_AT_MOST = 6  # more rounds than any test here means, and few enough to end at once


def _guided_by(monkeypatch, answer=(True, True)):
    """press_demo.guide_chair stood in for: it answers (wrist_turned, wrist_held), or exits with a code;
    returns its calls."""
    from box_opening.tasks import press_demo

    calls = []

    def guide(node, ctx, cfg, runner, watcher, args, locator, rest_tol, place, why, n, ear, out=None, what=None):
        at = None if place is None else tuple(place)
        calls.append({"place": at, "why": why, "n": n, "ear": ear, "homes": len(HOME_FIRSTS), "what": what})
        if isinstance(answer, int):
            sys.exit(answer)
        if len(calls) > GUIDED_AT_MOST:
            # a stand-in that always says "go" guides for ever where the run goes round again without end:
            # a test that expected none ran a full core for minutes and 2.3 GB of these calls before it was
            # killed (review of 2026-11-19). It ends loudly here
            sys.exit("guided %d times: the run goes round without end" % len(calls))
        return answer

    monkeypatch.setattr(press_demo, "guide_chair", guide)
    return calls


def test_a_box_no_press_plans_for_does_not_end_a_guided_run_it_looks_again_after_the_chair_moved(monkeypatch, tmp_path):
    """Run 2 of 2026-11-17 ended at "no reach-out press plans". With --guide (the engine's runs) the reach-out
    answers NO_WAY_IN instead: the chair is guided (guide_chair), and the run goes round again from the find,
    which is short where the wrist held the box as the go came. It does not home again (the arm rests but
    for its wrist), and the second answer is the run's."""
    from box_opening.tasks import press_demo
    from box_opening.tasks.reach_press import NO_WAY_IN

    LOCATES.clear(), HOME_FIRSTS.clear(), REACH_OUTS.clear()
    approach, _seen = _answers([NO_WAY_IN, NO_WAY_IN, 0])
    calls = _guided_by(monkeypatch)
    code, _c, rows = run_main(monkeypatch, tmp_path, ["--execute", "--guide"], approach)
    assert code == 0 and len(REACH_OUTS) == 3 and len(HOME_FIRSTS) == 1
    assert [c["n"] for c in calls] == [1, 2] and all(c["ear"] is not None for c in calls)
    assert calls[0]["place"] == tuple(REACH_OUTS[0].top_xyz)  # where the box stood: the engine holds it refused
    assert LOCATES == [press_demo.SCENE_FIRST_LOCATE_S, press_demo.SCENE_LINED_UP_LOCATE_S, 2.0]
    assert len([r for r in rows if r["leg"] == "scene_fix"]) == 3  # found again each time, from the cameras


def test_a_guided_run_whose_wrist_did_not_hold_the_box_looks_for_it_the_long_way(monkeypatch, tmp_path):
    from box_opening.tasks import press_demo
    from box_opening.tasks.reach_press import NO_WAY_IN

    LOCATES.clear(), HOME_FIRSTS.clear(), REACH_OUTS.clear()
    approach, _seen = _answers([NO_WAY_IN, 1])
    _guided_by(monkeypatch, (False, False))  # the scene camera alone watched (no wrist look)
    code, _c, _rows = run_main(monkeypatch, tmp_path, ["--execute", "--guide"], approach)
    assert code == 1 and LOCATES == [press_demo.SCENE_FIRST_LOCATE_S] * 2


def test_a_guided_run_that_is_told_to_stop_ends_not_lined_up_with_nothing_pressed(monkeypatch, tmp_path):
    from box_opening.runtime.abort import EXIT_NOT_LINED_UP
    from box_opening.tasks.reach_press import NO_WAY_IN

    REACH_OUTS.clear()
    approach, _seen = _answers([NO_WAY_IN, 0])
    calls = _guided_by(monkeypatch, EXIT_NOT_LINED_UP)
    code, _c, _rows = run_main(monkeypatch, tmp_path, ["--execute", "--guide"], approach)
    assert code == EXIT_NOT_LINED_UP and len(calls) == 1 and len(REACH_OUTS) == 1  # never pressed on


def test_the_chairs_move_forgets_what_the_cameras_measured_and_the_next_look_measures_it_again(monkeypatch, tmp_path):
    """The real guide_chair around a watch that answers go: the engine is asked on stdout before the watch
    says anything, and what the run measured of the scene is forgotten before the chair moves: the box's
    pose, the table's extent, the counter's edges kept out. (This bench world names its own table, which is
    what the run falls back to; a world with none, as flatbot's, has none again until the next look
    measures it: test_line_up_run.)"""
    from box_opening.constants import HOME
    from box_opening.runtime import approach as approach_mod
    from box_opening.tasks import press_demo
    from box_opening.tasks.reach_press import NO_WAY_IN

    REACH_OUTS.clear()
    approach, seen = _answers([NO_WAY_IN, 0])
    client = FakeClient()
    client.live = list(HOME)  # at rest (run_main's home_first is a stand-in)
    watched, cleared, forgot = [], [], []

    def watch(node, ctx, cfg, runner, watcher, args, locator, rest_tol, ear=None, out=None, rest_any=False):
        watched.append((rest_any, ear is not None))
        forgot.append((ctx.worlds.table_top_z, ctx.worlds.table_extent, ctx.cpose, watcher.table_z))
        return True, True, True

    monkeypatch.setattr(press_demo, "line_up_watch", watch)
    monkeypatch.setattr(approach_mod, "set_obstacles", lambda boxes: cleared.append(list(boxes)))
    printed = []
    real_write = sys.stdout.write
    monkeypatch.setattr(sys.stdout, "write", lambda text: (printed.append(text), real_write(text))[1])
    code, _c, rows = run_main(monkeypatch, tmp_path, ["--execute", "--guide"], approach, client=client)
    assert code == 0 and watched == [(True, True)] and cleared == [[]]
    asked = [json.loads(t[3:]) for t in printed if t.startswith("@@ ")]
    assert len(asked) == 1 and asked[0]["guide"]["n"] == 1
    assert asked[0]["guide"]["box"] == [round(float(v), 4) for v in REACH_OUTS[0].top_xyz]
    bench_table = -0.057 + 0.03  # world_bench.yaml's own
    # as the watch began: nothing of the scene before the move
    assert forgot == [(pytest.approx(bench_table), None, None, pytest.approx(bench_table))]
    assert seen[0][1] and seen[1] == seen[0]  # ... and the box placed again for the second reach-out
    (note,) = (r for r in rows if r["leg"] == "guide")
    assert note["n"] == 1 and note["box"] == asked[0]["guide"]["box"]


def _wrist_sees(monkeypatch, far):
    """press_demo.wrist_look stood in for: the wrist finds no container, and says which lid-sized tops it saw
    out of the arm's reach (as wrist_first does: ctx.wrist_far)."""
    from box_opening.tasks import press_demo

    def look(node, ctx, cfg, runner, watcher, args, locator, scene_fix, said):
        ctx.wrist_far, ctx.wrist_seen = list(far), "unseen"  # (its finder judged a frame: nothing in reach)
        said.append("wrist camera: the lid-sized top at [1.020, -0.200] stands out of the arm's reach")
        return None, None, None

    monkeypatch.setattr(press_demo, "wrist_look", look)


def test_a_container_the_wrist_sees_out_of_the_arms_reach_is_no_no_box_for_a_guided_run(monkeypatch, tmp_path, capsys):
    """The scene camera has none and the wrist sees a lid-sized top it calls out of the arm's reach: the run
    said NO BOX, went home and ended (not_found). Guided, the chair is moved for it: no NO BOX, no home, and
    the engine is asked with where that top stands. By hand it is the NO BOX it was. (Where the wrist saw
    nothing lid-sized at all, the guided run ended NO BOX too until 2026-11-19: the tests below.)"""
    from box_opening.runtime.abort import EXIT_NOT_LINED_UP
    from box_opening.tasks import press_demo

    homes = []
    monkeypatch.setattr(press_demo, "try_home", lambda ctx, runner, execute, why: homes.append(why))
    far = [(1.02, -0.20, 0.30), (1.15, 0.10, 0.30)]
    _wrist_sees(monkeypatch, far)
    REACH_OUTS.clear()
    calls = _guided_by(monkeypatch, EXIT_NOT_LINED_UP)
    argv = ["--execute", "--guide", "--wrist-first"]
    code, _c, rows = run_main(monkeypatch, tmp_path / "guided", argv, _refused, found=False)
    assert code == EXIT_NOT_LINED_UP and homes == [] and not REACH_OUTS
    assert [c["place"] for c in calls] == [far[0]] and calls[0]["n"] == 1  # the nearest the arm
    assert calls[0]["what"] == press_demo.GUIDE_FAR == "far"  # said with the request: not a press with no way in
    (refused,) = (r for r in rows if r["leg"] == "reach_refused")
    assert refused["guided"] is True and "out of the arm's reach" in refused["why"] and "NO BOX" not in refused["why"]
    assert refused["what"] == "far"
    out = capsys.readouterr().out
    assert "OUT OF REACH: the wrist camera sees a lid-sized top at [1.02, -0.20]" in out and "NO BOX" not in out
    # by hand (no --guide): NO BOX and home, as it was
    calls.clear()
    code, _c, rows = run_main(monkeypatch, tmp_path / "by hand", ["--execute", "--wrist-first"], _refused, found=False)
    assert code == 2 and calls == [] and len(homes) == 1 and homes[0].startswith("NO BOX")


SCENE_NONE = "scene camera: the OWL's 3 candidates refused"


def test_a_container_no_camera_sees_does_not_end_a_guided_run_the_chair_is_guided(monkeypatch, tmp_path, capsys):
    """Owner 2026-11-19, asked whether a box no camera sees at all should become guidance too ("move so the
    container is in front of the arm"): yes. Until then the guided run said NO BOX, went home and ended
    (not_found), as a run by hand does. Now it says what was missing, with no NO BOX and no home, notes it as
    the far top's is noted (guided, with what the look saw), and has the chair guided with no place to
    hand on. At the yes it looks again, the long way where the wrist did not hold the box, and goes on."""
    from box_opening.tasks import press_demo

    homes = []
    monkeypatch.setattr(press_demo, "try_home", lambda ctx, runner, execute, why: homes.append(why))
    LOCATES.clear(), HOME_FIRSTS.clear(), REACH_OUTS.clear()
    calls = _guided_by(monkeypatch, (False, False))
    code, _c, rows = run_main(monkeypatch, tmp_path, ["--execute", "--guide"], _at_staging(), found=[False, True])
    assert code == 0 and len(REACH_OUTS) == 1 and homes == []  # found after the chair moved, and reached out to
    (call,) = calls
    assert call["place"] is None and call["what"] == press_demo.GUIDE_UNSEEN == "unseen" and call["n"] == 1
    assert call["why"] == "no camera sees the container (%s)" % SCENE_NONE and call["ear"] is not None
    (refused,) = (r for r in rows if r["leg"] == "reach_refused")
    assert refused["why"] == call["why"] and refused["guided"] is True and refused["what"] == "unseen"
    assert refused["timing"] == {"streams_s": 0.1} and refused["segmenter"] == {"state": "live", "frames": 3}
    assert rows.index(refused) < [r["leg"] for r in rows].index("scene_fix")  # then found, from the cameras
    assert LOCATES == [press_demo.SCENE_FIRST_LOCATE_S] * 2
    out = capsys.readouterr().out
    assert "[press_demo] NOT SEEN: no camera sees the container (%s); nothing was pressed" % SCENE_NONE in out
    assert "NO BOX" not in out


def test_after_a_yes_a_fresh_find_that_sees_no_container_is_guided_again_not_ended(monkeypatch, tmp_path, capsys):
    """Every find of a guided run is the same find: the one after a line-up's yes (--line-up), and the one
    after a guidance's yes, where the box the watch held is not found by the run's own look. Each ended NO
    BOX. Each is guided again, and nothing moves toward a box but on another yes; a stop then ends the run
    as a line-up's stop does."""
    from box_opening.runtime.abort import EXIT_NOT_LINED_UP
    from box_opening.tasks import press_demo
    from box_opening.tasks.reach_press import NO_WAY_IN

    homes = []
    monkeypatch.setattr(press_demo, "try_home", lambda ctx, runner, execute, why: homes.append(why))
    # found, no press plans, guided; at the yes the fresh find has nothing; guided again; then found
    REACH_OUTS.clear()
    approach, _seen = _answers([NO_WAY_IN, 1])
    calls = _guided_by(monkeypatch, (True, True))
    code, _c, rows = run_main(
        monkeypatch, tmp_path / "again", ["--execute", "--guide"], approach, found=[True, False, True]
    )
    assert code == 1 and len(REACH_OUTS) == 2 and homes == []  # (the second reach-out's own end)
    assert [(c["n"], c["what"], c["place"] is None) for c in calls] == [(1, None, False), (2, "unseen", True)]
    assert [r.get("what") for r in rows if r["leg"] == "reach_refused"] == ["unseen"]  # (NO_WAY_IN's is its own)
    assert "NO BOX" not in capsys.readouterr().out
    # after a line-up before the run: the same
    calls.clear(), REACH_OUTS.clear()
    _lining_up(monkeypatch, (True, True, True))
    calls = _guided_by(monkeypatch, EXIT_NOT_LINED_UP)
    argv = ["--execute", "--line-up", "--guide"]
    code, _c, _rows = run_main(monkeypatch, tmp_path / "lined up", argv, _refused, found=False)
    assert code == EXIT_NOT_LINED_UP and not REACH_OUTS and homes == []
    assert [(c["n"], c["what"]) for c in calls] == [(1, "unseen")]


def test_a_run_by_hand_whose_cameras_see_no_container_ends_no_box_as_it_always_has(monkeypatch, tmp_path, capsys):
    """Without --guide nothing changed: NO BOX, the row, the home, exit 2 (the node's not_found). Nor for a
    preview with --guide, which moves nothing and is guided nowhere."""
    from box_opening.tasks import press_demo

    for i, argv in enumerate((["--execute"], ["--guide"])):
        homes = []
        monkeypatch.setattr(press_demo, "try_home", lambda ctx, runner, execute, why, _h=homes: _h.append(why))
        calls = _guided_by(monkeypatch)
        code, _c, rows = run_main(monkeypatch, tmp_path / str(i), argv, _refused, found=False)
        assert code == 2 and calls == [] and len(homes) == 1 and homes[0].startswith("NO BOX — scene camera:"), argv
        (refused,) = (r for r in rows if r["leg"] == "reach_refused")
        assert refused["why"] == homes[0] and "guided" not in refused and "what" not in refused
        out = capsys.readouterr().out
        assert "NOT SEEN" not in out


def _wrist_look_says(monkeypatch, survey):
    """tasks/survey.run_survey stood in for, the run's own wrist_look and wrist_first real: what the wrist's
    look in front comes back with. Returns the keywords of each look asked for."""
    from box_opening.tasks import survey as sv

    asked = []
    monkeypatch.setattr(sv, "run_survey", lambda *a, **k: asked.append(k) or survey)
    return asked


def _found_by_the_wrist(xy=(0.62, 0.04)):
    from types import SimpleNamespace

    from box_opening.tasks import survey as sv

    fix = SimpleNamespace(top_xyz=(xy[0], xy[1], 0.276), table_z_scene=0.185, partial="")
    return sv.Survey(fix=fix, find=SimpleNamespace(fix=fix, places=(), patches=()), clouds=["its points"])


NOTHING_THERE = "wrist camera: no counter in 4 frames (no level plane wider than a container)"
DOWN = {"timing": {}, "segmenter": {"state": "absent", "frames": 0}}  # the scene segmenter gave the look no frame


def test_a_camera_that_delivers_nothing_is_a_fault_not_a_container_out_of_view(monkeypatch, tmp_path, capsys):
    """NO BOX wears the face of several failures (SceneLocator.locate's own note): no container in view, but
    also streams missing, no TF, the scene segmenter down or silent, a wrist camera with no frame. No moving
    of the chair cures those, and "the arm cannot see the container, move the chair" is not true of them.
    A run is guided only where the camera its guidance would watch with DELIVERED and had nothing that could
    be the container; else it ends NO BOX with what was missing, as it always has, and its row says what
    that was. Which camera: the wrist's where the wrist look is on (review of 2026-11-19: as first written
    a wrist camera with no frame was "a camera that looked" because the scene segmenter had given one)."""
    from types import SimpleNamespace

    from box_opening.runtime.abort import EXIT_NOT_LINED_UP
    from box_opening.tasks import press_demo
    from box_opening.tasks import survey as sv

    homes = []
    monkeypatch.setattr(press_demo, "try_home", lambda ctx, runner, execute, why: homes.append(why))

    def ends(name, argv, **kw):
        homes.clear()
        calls = _guided_by(monkeypatch, EXIT_NOT_LINED_UP)
        code, _c, rows = run_main(
            monkeypatch, tmp_path / name, ["--execute", "--guide", *argv], _refused, found=False, **kw
        )
        out = capsys.readouterr().out
        refused = [r for r in rows if r["leg"] == "reach_refused"]
        if code == 2:  # NO BOX: said, homed, never guided, and what it was in its row
            assert calls == [] and len(homes) == 1 and homes[0].startswith("NO BOX") and "NOT SEEN" not in out
            return refused[-1].get("what")
        assert code == EXIT_NOT_LINED_UP and homes == [] and "NO BOX" not in out, name
        assert [c["what"] for c in calls] == ["unseen"] and refused[-1]["what"] == "unseen"
        return "guided"

    # the scene camera alone, and its segmenter gave the look no frame
    assert ends("scene down", [], record=DOWN) == "fault"
    assert ends("scene up", []) == "guided"
    # the wrist look on: it is the wrist's camera the guidance watches with
    no_frame = sv.Survey(fix=None, why="no wrist frame at survey:angle-1 (no new wrist frame)", find=None)
    _wrist_look_says(monkeypatch, no_frame)
    assert ends("wrist dead, scene up", ["--wrist-first"]) == "fault"  # however alive the scene segmenter is
    looked = sv.Survey(fix=None, why=NOTHING_THERE, find=SimpleNamespace(fix=None, places=(), patches=()))
    _wrist_look_says(monkeypatch, looked)
    assert ends("wrist up, scene down", ["--wrist-first"], record=DOWN) == "guided"  # ... or dead
    # a wrist that cannot look from any view (none to turn to) is no camera fault: the scene camera's word
    blind = sv.Survey(fix=None, why="no turn of the wrist shows the table in front (x)", find=None)
    _wrist_look_says(monkeypatch, blind)
    assert ends("wrist blind, scene up", ["--wrist-first"]) == "guided"
    assert ends("wrist blind, scene down", ["--wrist-first"], record=DOWN) == "fault"
    # by hand none of it is asked: NO BOX, and its row as it always was
    homes.clear()
    code, _c, rows = run_main(monkeypatch, tmp_path / "by hand", ["--execute"], _refused, found=False, record=DOWN)
    assert code == 2 and "what" not in [r for r in rows if r["leg"] == "reach_refused"][-1]


def test_what_a_find_with_no_container_is_of_decides_whether_the_chair_is_guided(monkeypatch):
    """no_container_kind: what each look saw, and which camera a guidance would depend on."""
    from argparse import Namespace
    from types import SimpleNamespace

    from box_opening import guide
    from box_opening.tasks import press_demo
    from box_opening.tasks import survey as sv

    def kind(wrist_first=True, wrist=None, side=None, record=LOOKED, why="scene segmenter: saw no food container"):
        ctx = SimpleNamespace(wrist_seen=wrist, side_seen=side)
        locator = SimpleNamespace(look_record=lambda: record, last_why=why)
        return press_demo.no_container_kind(ctx, Namespace(wrist_first=wrist_first), locator)

    two = "scene segmenter: two places pass as this container, [0.80, -0.20] and [1.10, 0.30]: select which"
    gone = "scene segmenter: the selected container (fc_1) was in none of its 70 frames"
    assert kind(wrist="unseen") == kind(wrist="unseen", record=DOWN) == kind(wrist="unseen", why=two) == "unseen"
    assert kind(wrist="unseen", why=gone) == "unseen"  # the wrist looked and has nothing: the chair cures that
    assert kind(wrist="fault") == "fault" and kind(wrist="several") == "several"
    assert kind(wrist="not_chosen") == kind(wrist="unseen", side="not_chosen") == "not_chosen"
    assert kind(wrist="unseen", side="several") == kind(wrist_first=False, side="several") == "several"
    # the scene camera alone (no wrist look, or a wrist that could not look from any view): its own word
    for wrist_first in (False, True):
        assert kind(wrist_first=wrist_first) == "unseen" and kind(wrist_first=wrist_first, record=DOWN) == "fault"
        assert kind(wrist_first=wrist_first, why=two) == "several"
        # a selected id that is merely absent from its frames: the container may simply be out of its view,
        # and that is a container the arm cannot see, which is guided for (review of 2026-11-19, the second)
        assert kind(wrist_first=wrist_first, why=gone) == "unseen"
        assert kind(wrist_first=wrist_first, record={"timing": {}}) == "fault"
    assert kind(wrist_first=False, wrist="fault") == "unseen"  # (a wrist the guidance would not watch with)
    assert press_demo.no_container_kind(SimpleNamespace(), Namespace(wrist_first=False), None) == "fault"
    # what a wrist look saw (looked_kind): by its finder's word where it judged a frame, a fault where its
    # camera gave none at a view it stood at, None where it never looked from a view at all
    find = SimpleNamespace(fix=None, places=(), patches=())
    for why, judged, saw in (
        (NOTHING_THERE, True, "unseen"),
        (
            "wrist camera: two places pass as this container, [0.61, -0.10] and [0.66, 0.14]: refusing to guess",
            True,
            "several",
        ),
        ("no wrist frame at survey:angle-1 (no new wrist frame)", False, "fault"),
        ("no wrist frame at survey:angle-2 (no camera pose for the frame (TF))", False, "fault"),
        ("no turn of the wrist shows the table in front (x)", False, None),
        ("the survey keeps its poses clear of the chair by the body model, and it is off (body_layers)", False, None),
        ("survey:angle-1 (the wrist turned to look at [0.60, 0.00]) not flown: refused (IK_FAIL)", False, None),
    ):
        assert press_demo.looked_kind(sv.Survey(fix=None, why=why, find=find if judged else None)) == saw, why
    assert guide.K_UNSEEN == press_demo.GUIDE_UNSEEN  # (one word for it, the run's and the guide's)


def test_a_selection_the_scene_camera_does_not_hold_is_guided_only_where_nothing_that_could_be_it_is_seen(
    monkeypatch, tmp_path, capsys
):
    """The review's worst case (2026-11-19). The UI's selection whose id is in none of the scene camera's
    frames (typed by hand, or a track perception dropped), wrist_alone off: nothing ties any find to it.
    As first guided, that run went into "The arm cannot see the container. Move the chair in front of it"
    with a lid in every frame of its watch, which took none of them: no place of the chair could bring its
    Ready, for a wait that is five minutes on the UI. Two things were one: a container in no camera's view,
    which the chair cures, and a container in front of the arm that nothing ties to the one chosen, which
    it does not. The scene camera cannot tell them apart ("was in none of its frames" either way), so a
    run the engine can guide turns its wrist to look, and takes nothing it finds:
      something that could be the container stands in front: NO BOX as at 21aa7cc, its row saying what;
      nothing does: the chair is guided, as for any container no camera sees."""
    from types import SimpleNamespace

    from box_opening.runtime.abort import EXIT_NOT_LINED_UP
    from box_opening.tasks import press_demo
    from box_opening.tasks import survey as sv

    homes = []
    monkeypatch.setattr(press_demo, "try_home", lambda ctx, runner, execute, why: homes.append(why))
    gone = "scene segmenter: the selected container (food_container_7) was in none of its 70 frames"
    argv = ["--execute", "--guide", "--wrist-first", "--instance-id", "food_container_7"]
    # a container in front of the arm, and the selected id in none of the scene camera's detections
    REACH_OUTS.clear()
    asked = _wrist_look_says(monkeypatch, _found_by_the_wrist())
    calls = _guided_by(monkeypatch, EXIT_NOT_LINED_UP)
    code, _c, rows = run_main(monkeypatch, tmp_path / "in front", argv, _refused, found=False, why=gone)
    assert code == 2 and calls == [] and not REACH_OUTS  # never guided, and never reached out to
    assert len(asked) == 1 and asked[0]["front"] is True and asked[0]["hint"] is None  # the wrist did look
    assert len(homes) == 1 and homes[0].startswith("NO BOX — scene camera: scene segmenter: the selected container")
    assert "a container at [0.620, 0.040], and the selected container food_container_7 is in none of" in homes[0]
    (refused,) = (r for r in rows if r["leg"] == "reach_refused")
    assert refused["what"] == "not_chosen" and refused["why"] == homes[0] and "guided" not in refused
    (noted,) = (r for r in rows if r["leg"] == "wrist_first")
    assert noted["found"] is True and noted["tied"] is False and noted["untied"] is True
    out = capsys.readouterr().out
    assert "the wrist looks only to tell whether anything that could be it stands in front of the arm" in out
    assert "NOT SEEN" not in out
    # ... two that could be it, the finder unable to choose: something is there, and nothing ties it either
    two = "wrist camera: two places pass as this container, [0.61, -0.10] and [0.66, 0.14]: refusing to guess"
    homes.clear()
    _wrist_look_says(monkeypatch, sv.Survey(fix=None, why=two, find=SimpleNamespace(fix=None, places=(), patches=())))
    code, _c, rows = run_main(monkeypatch, tmp_path / "two", argv, _refused, found=False, why=gone)
    assert code == 2 and calls == [] and [r["what"] for r in rows if r["leg"] == "reach_refused"] == ["not_chosen"]
    # ... a lid-sized top out of the arm's reach: something that could be it, and no place of the chair makes
    # it the one selected, so the chair is not guided to it either (as it is where there is no selection)
    homes.clear()
    far_why = "wrist camera: the lid-sized top at [1.020, -0.200] stands out of the arm's reach"
    monkeypatch.setattr(press_demo, "far_tops", lambda find, model: [(1.02, -0.20, 0.30)])
    _wrist_look_says(
        monkeypatch, sv.Survey(fix=None, why=far_why, find=SimpleNamespace(fix=None, places=(), patches=()))
    )
    code, _c, rows = run_main(monkeypatch, tmp_path / "far", argv, _refused, found=False, why=gone)
    assert code == 2 and calls == [] and [r["what"] for r in rows if r["leg"] == "reach_refused"] == ["not_chosen"]
    assert "a lid-sized top at [1.02, -0.20], out of the arm's reach, and the selected container" in homes[0]
    code, _c, rows = run_main(monkeypatch, tmp_path / "far, none chosen", argv[:3], _refused, found=False)
    assert code == EXIT_NOT_LINED_UP and [c["what"] for c in calls] == ["far"]  # (no selection: guided to it)
    calls.clear()
    monkeypatch.setattr(press_demo, "far_tops", lambda find, model: [])
    # nothing that could be the container in front of the arm: no camera sees it, and the chair is guided
    homes.clear()
    nothing = sv.Survey(fix=None, why=NOTHING_THERE, find=SimpleNamespace(fix=None, places=(), patches=()))
    _wrist_look_says(monkeypatch, nothing)
    code, _c, rows = run_main(monkeypatch, tmp_path / "nowhere", argv, _refused, found=False, why=gone)
    assert code == EXIT_NOT_LINED_UP and homes == [] and [c["what"] for c in calls] == ["unseen"]
    assert calls[0]["place"] is None and NOTHING_THERE in calls[0]["why"]
    assert "NO BOX" not in capsys.readouterr().out
    # the scene camera alone (no wrist look): all it knows is that the selected id is in none of its frames,
    # as it is for a container that is simply out of its view. That is a container the arm cannot see:
    # guided, never "Choose it again" (review of 2026-11-19, the second; a626985 ended it NO BOX, and the
    # same word ended a scene-only line-up after ten seconds)
    calls.clear(), homes.clear()
    code, _c, rows = run_main(monkeypatch, tmp_path / "scene", argv[:2] + argv[3:], _refused, found=False, why=gone)
    assert code == EXIT_NOT_LINED_UP and homes == [] and [c["what"] for c in calls] == ["unseen"]
    assert [r["what"] for r in rows if r["leg"] == "reach_refused"] == ["unseen"]
    # by hand the wrist is not turned for it and the run ends NO BOX, as it did, with nothing new in its row
    calls.clear(), asked.clear(), homes.clear()
    asked = _wrist_look_says(monkeypatch, _found_by_the_wrist())
    code, _c, rows = run_main(monkeypatch, tmp_path / "by hand", argv[:1] + argv[2:], _refused, found=False, why=gone)
    assert code == 2 and calls == [] and asked == [] and len(homes) == 1
    assert "not looked for: nothing the scene camera saw says where the selected container" in homes[0]
    assert "what" not in [r for r in rows if r["leg"] == "reach_refused"][-1]


def test_two_things_that_could_be_the_container_end_a_guided_run_as_they_always_ended_a_run(
    monkeypatch, tmp_path, capsys
):
    """ "Nothing found, or two places that could be it, is not_found" (docs/interface.md, since the side
    survey). Two lid-sized tops side by side within the arm's reach are not parted by where the chair
    stands, so a run is not guided for them: NO BOX, as ever, its row saying it was two. But what the SCENE
    camera makes of two things further off does not make a find "two" where the wrist looked in front of
    the arm and saw nothing: that is a container not in view, and guided (review of 2026-11-19: the words
    of the whole find were searched for "two", the scene camera's part included)."""
    from types import SimpleNamespace

    from box_opening.runtime.abort import EXIT_NOT_LINED_UP
    from box_opening.tasks import press_demo
    from box_opening.tasks import survey as sv

    homes = []
    monkeypatch.setattr(press_demo, "try_home", lambda ctx, runner, execute, why: homes.append(why))
    two = "wrist camera: two places pass as this container, [0.61, -0.10] and [0.66, 0.14]: refusing to guess"
    scene_two = "scene segmenter: two places pass as this container, [0.80, -0.20] and [1.10, 0.30]: select which"
    find = SimpleNamespace(fix=None, places=(), patches=())
    argv = ["--execute", "--guide", "--wrist-first"]
    _wrist_look_says(monkeypatch, sv.Survey(fix=None, why=two, find=find))
    calls = _guided_by(monkeypatch, EXIT_NOT_LINED_UP)
    code, _c, rows = run_main(monkeypatch, tmp_path / "wrist two", argv, _refused, found=False)
    assert code == 2 and calls == [] and len(homes) == 1 and homes[0].startswith("NO BOX")
    assert [r["what"] for r in rows if r["leg"] == "reach_refused"] == ["several"]
    homes.clear()
    _wrist_look_says(monkeypatch, sv.Survey(fix=None, why=NOTHING_THERE, find=find))
    code, _c, rows = run_main(monkeypatch, tmp_path / "scene two", argv, _refused, found=False, why=scene_two)
    assert code == EXIT_NOT_LINED_UP and homes == [] and [c["what"] for c in calls] == ["unseen"]
    # ... and the scene camera alone, with two it cannot choose between: its own word, and no guidance
    calls.clear()
    code, _c, rows = run_main(monkeypatch, tmp_path / "alone", argv[:2], _refused, found=False, why=scene_two)
    assert code == 2 and calls == [] and [r["what"] for r in rows if r["leg"] == "reach_refused"] == ["several"]


def _side_survey(monkeypatch, finds):
    """tasks/survey.run_survey stood in for (the wrist's survey over the arm's side, --wrist-survey):
    `finds` True, it finds the container beside the chair; False, none. Returns (its calls, its fix)."""
    from box_opening.models.container import ContainerModel
    from box_opening.perception.depth_source import container_pose_from_top
    from box_opening.tasks import survey as sv

    model = ContainerModel.load(PINK)
    table_z = -0.057 + 0.03
    top = (0.30, -0.45, table_z + model.dims[2])  # beside the chair, not in front of it

    class _SideFix:
        pose = container_pose_from_top(top, 0.0, model, table_z)
        top_xyz = top
        n_points = 900
        score = 0.6
        source = "wrist"
        table_z_scene = table_z
        table_extent = None

    calls = []

    def run_survey(node, ctx, runner, watcher, execute, **kw):
        calls.append(len(HOME_FIRSTS))
        if finds:
            return sv.Survey(fix=_SideFix, clouds=["the side survey's points"])
        return sv.Survey(fix=None, why="wrist camera: no lid-sized top on the counter beside the arm")

    monkeypatch.setattr(sv, "run_survey", run_survey)
    return calls, _SideFix


def test_with_the_wrist_survey_on_a_far_top_in_front_does_not_take_the_side_surveys_turn(monkeypatch, tmp_path, capsys):
    """wrist_first and wrist_survey both on, no selection: the container stands beside the chair and
    something lid-sized stands out of reach in front. Before there was any guidance the front look's none
    led to the side survey, which found the box. As the guidance was first written the far top came first:
    the user was guided toward it and the side survey was never tried (review of 2026-11-17).

    What is chosen: the side survey keeps its turn, before anything is made of the far top. Where it finds
    the container the run opens it and nobody is guided. Where it finds none either, the run still does
    not end NO BOX: the chair is guided to the top in front (guide_chair brings the arm the survey left
    over its side to rest first). With the orchestrator's selection there is no side survey (the wrist
    cannot tell which container was selected), and the far top is guided to at once, as without the flag."""
    from box_opening.runtime.abort import EXIT_NOT_LINED_UP
    from box_opening.tasks import press_demo

    homes = []
    monkeypatch.setattr(press_demo, "try_home", lambda ctx, runner, execute, why: homes.append(why))
    far = [(1.02, -0.20, 0.30)]
    _wrist_sees(monkeypatch, far)
    argv = ["--execute", "--guide", "--wrist-first", "--wrist-survey"]
    # the side survey finds the container: the run reaches out to IT, and nobody is guided anywhere
    REACH_OUTS.clear()
    surveys, side_fix = _side_survey(monkeypatch, finds=True)
    calls = _guided_by(monkeypatch, EXIT_NOT_LINED_UP)
    code, _c, rows = run_main(monkeypatch, tmp_path / "beside", argv, _refused, found=False)
    assert len(surveys) == 1 and calls == [] and REACH_OUTS == [side_fix] and code == 1  # (_refused's own end)
    assert not [r for r in rows if r["leg"] == "reach_refused"] and "OUT OF REACH" not in capsys.readouterr().out
    # it finds none either: no NO BOX and no end; the chair is guided to the top in front, after the survey
    REACH_OUTS.clear()
    surveys, _fix = _side_survey(monkeypatch, finds=False)
    code, _c, rows = run_main(monkeypatch, tmp_path / "nowhere", argv, _refused, found=False)
    assert len(surveys) == 1 and [c["place"] for c in calls] == [far[0]] and code == EXIT_NOT_LINED_UP
    assert not REACH_OUTS and homes == []
    out = capsys.readouterr().out
    assert out.index("WRIST SURVEY: no container beside the arm either") < out.index("OUT OF REACH: the wrist")
    assert "NO BOX" not in out
    (refused,) = (r for r in rows if r["leg"] == "reach_refused")
    assert refused["guided"] is True
    # a selection: no side survey for it (as ever), so the far top is what the chair is guided to
    calls.clear()
    surveys, _fix = _side_survey(monkeypatch, finds=True)
    code, _c, _rows = run_main(
        monkeypatch, tmp_path / "selected", [*argv, "--instance-id", "food_container_1"], _refused, found=False
    )
    assert surveys == [] and [c["place"] for c in calls] == [far[0]] and code == EXIT_NOT_LINED_UP
    # by hand (no --guide) nothing changed: the survey, then NO BOX and home where it finds none
    calls.clear()
    surveys, _fix = _side_survey(monkeypatch, finds=False)
    by_hand = ["--execute", "--wrist-first", "--wrist-survey"]
    code, _c, _rows = run_main(monkeypatch, tmp_path / "by hand", by_hand, _refused, found=False)
    assert len(surveys) == 1 and calls == [] and code == 2 and len(homes) == 1 and homes[0].startswith("NO BOX")


def test_the_wrist_look_says_which_tops_it_saw_out_of_the_arms_reach(monkeypatch):
    """wrist_first keeps the finder's answer where it found no container: the lid-sized tops that all stand
    out of the arm's reach (far_tops), for main's guidance; none where it found the container."""
    import math
    from argparse import Namespace
    from types import SimpleNamespace

    from box_opening.tasks import press_demo
    from box_opening.tasks import survey as sv

    top = lambda x, y: SimpleNamespace(xy=(x, y), top_z=0.30, button="unchecked")  # noqa: E731
    monkeypatch.setattr(sv, "_within_reach", lambda model: lambda t: math.hypot(t[0], t[1]) < 0.9)
    notes = []
    runner = SimpleNamespace(note=lambda leg, **f: notes.append((leg, f)))
    ctx = SimpleNamespace(model=object(), worlds=SimpleNamespace(wrist_camera_config=None), wrist_far=None)
    cfg = SimpleNamespace(park_tool_down=False)
    why = "wrist camera: the lid-sized top at [1.020, -0.200] stands out of the arm's reach"
    survey = sv.Survey(fix=None, why=why, find=SimpleNamespace(fix=None, places=(top(1.02, -0.20),)))
    monkeypatch.setattr(sv, "run_survey", lambda *a, **k: survey)
    said = []
    got = press_demo.wrist_first(None, ctx, cfg, runner, object(), Namespace(execute=True), None, said=said)
    assert got is None and ctx.wrist_far == [(1.02, -0.20, 0.30)] and said == [why]
    assert ctx.wrist_seen == "unseen"  # its finder judged a frame: what it did not find is not in view
    survey = sv.Survey(fix=None, why="wrist camera: no counter", find=SimpleNamespace(fix=None, places=()))
    monkeypatch.setattr(sv, "run_survey", lambda *a, **k: survey)
    assert press_demo.wrist_first(None, ctx, cfg, runner, object(), Namespace(execute=True), None) is None
    assert ctx.wrist_far == [] and ctx.wrist_seen == "unseen"
    # ... and a look that never stood at a view (none to turn to) has not looked at all
    survey = sv.Survey(fix=None, why="no turn of the wrist shows the table in front (x)", find=None)
    monkeypatch.setattr(sv, "run_survey", lambda *a, **k: survey)
    assert press_demo.wrist_first(None, ctx, cfg, runner, object(), Namespace(execute=True), None) is None
    assert ctx.wrist_far == [] and ctx.wrist_seen is None
    # a lid it HAS in view that the selection's tie does not take: off the scene camera's line to the selected
    # container, it is something in view that nothing ties to it, not a container the arm cannot see (with
    # the scene camera's mount off, as on 2026-11-17, that is the selected container itself); with a second
    # top in the selection's direction, two that could be it
    fix = SimpleNamespace(top_xyz=(0.65, 0.0, 0.30), table_z_scene=0.2, partial="")
    survey = sv.Survey(fix=fix, why="", find=SimpleNamespace(fix=fix, places=(), patches=()), clouds=["c"])
    monkeypatch.setattr(sv, "run_survey", lambda *a, **k: survey)
    off = "212 mm to the side of the scene camera's line of sight to it"
    for judged, rival, seen in (((False, off), None, "not_chosen"), ((True, "on its line"), (0.70, 0.10), "several")):
        tie = SimpleNamespace(instance_id="fc_1", judge=lambda xy, j=judged: j, rival=lambda find, xy, r=rival: r)
        said = []
        got = press_demo.wrist_first(
            None, ctx, cfg, runner, object(), Namespace(execute=True), None, said=said, tie=tie
        )
        assert got is None and ctx.wrist_seen == seen and "not taken for the selected container fc_1" in said[0]


def test_a_run_by_hand_is_never_guided(monkeypatch, tmp_path):
    """Without --guide nothing is asked of an engine that is not there: a preview (no --execute) neither."""
    from box_opening.tasks.reach_press import NO_WAY_IN

    for i, argv in enumerate((["--execute"], ["--guide"])):
        REACH_OUTS.clear()
        approach, _seen = _answers([NO_WAY_IN])
        calls = _guided_by(monkeypatch)
        code, _c, _rows = run_main(monkeypatch, tmp_path / str(i), argv, approach)
        assert code == 1 and calls == [] and len(REACH_OUTS) == 1, argv


def test_a_run_without_line_up_never_waits_on_anyone(monkeypatch, tmp_path):
    HOME_FIRSTS.clear(), REACH_OUTS.clear()
    calls = _lining_up(monkeypatch, (False, False, False))
    code, _c, _rows = run_main(monkeypatch, tmp_path, ["--execute"], _refused)
    assert code == 1 and calls == [] and len(HOME_FIRSTS) == 1 and len(REACH_OUTS) == 1


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


def test_a_cancelled_run_keeps_no_mission_frames(monkeypatch, tmp_path):
    """Review on #23 (REVIEW_RULES C1): a cancel (the engine's Ctrl+C, exit
    130) ends at once; the mission's frames are not written on the way out.
    A run that ends any other way still keeps them (when they are on)."""
    kept = []
    monkeypatch.setattr(press_demo, "keep_mission_frames", lambda frames, ending: kept.append(ending))

    def ctrl_c(ctx):
        raise KeyboardInterrupt

    code, _c, _rows = run_main(monkeypatch, tmp_path, ["--execute"], ctrl_c)
    assert code == 130 and kept == []
    code, _c, _rows = run_main(monkeypatch, tmp_path, ["--execute"], _at_staging())
    assert code == 0 and len(kept) == 1


def test_no_frames_to_keep_is_a_no_op():
    assert press_demo.keep_mission_frames(None, SystemExit(2)) is None
    assert press_demo.keep_mission_frames(getattr(None, "mission_frames", None), None) is None
