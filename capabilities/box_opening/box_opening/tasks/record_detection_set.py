"""Record a labelled detection set: the box at many placements, as both
cameras see it — the evaluation set for improving detection "anywhere"
(Onyx campaign 1, onyx/onyx.md).

    ros2 run box_opening record_detection_set --execute --set anywhere-01

For each point of a grid over the reachable table, the arm HOVERS 10 cm over
it, fingers down, and waits: slide the box under the gripper, centred,
hands clear, press Enter. Then, with nothing but the operator's Enter
between motions:

  1. HOME, and the scene camera's frames are saved — what the mission sees
     before the arm moves;
  2. the search poses (the look, then each sweep), stopping at each to save
     still wrist frames — what the wrist search sees;
  3. staging over the point, and the mission's own close-up aim at the
     button: its press point is the TRUTH (the aim is what the press lands
     on, to under a millimetre), saved with the staging frame;
  4. HOME, and the next point.

At the end, EMPTY scenes: take the box off the table, leave everything else
as it is — a detector that finds a box there is wrong.

Nothing is pressed. Every motion is planned (bench or full world, like the
mission's) and flown only with --execute; the operator runs it with the
e-stop in hand. Without --execute it plans the first hover and moves
nothing.

Output: <state>/detection_sets/<set>/p<NNN>/ (scene_<k>/, wrist_<pose>_<k>/,
staging/ — each a frame_000.npz — and truth.json), and manifest.json with a
sha256 for every file: the evaluator refuses a set that does not match it.
"""

import json
import math
import sys
import time
from pathlib import Path

from box_opening import LOGGER as log
from box_opening.constants import GRIPPER_CMD_CLOSED, HOME, HOME_START_TOL_RAD, TRANSIT_SPEED, state_dir
from box_opening.detection_set import grid_points, scene_frame, write_frame, write_manifest
from box_opening.models.container import ContainerModel, ContainerPose, load_press_demo
from box_opening.tasks import cli_common

HOVER_M = 0.10  # fingertips this far above the lid while the box is placed
SCENE_FRAMES = 3  # scene captures per placement (each a 5-frame depth median)
WRIST_FRAMES = 2  # still wrist captures per search pose
STILL_WAIT_S = 3.0  # to get still wrist frames at a pose
AT_HOME_RAD = 0.02  # this close to HOME is at HOME
# The aim's button this far from where the box was placed under the gripper:
# the operator centres it to ~1-2 cm, so the aim found something else — the
# truth is flagged, and the evaluator leaves it out
TRUTH_SUSPECT_M = 0.05


def table_refusal(worlds):
    """Why the recorder cannot run on this world, or None. Its hovers and
    the lid's height come from a surveyed table; flatbot's table is measured
    by the mission's scene fix, which the recorder does not take yet."""
    if worlds.table_top_z is not None:
        return None
    return (
        "[record] needs a surveyed table (a bench world with a 'table' obstacle); "
        "flatbot measures its table per run — not supported yet. Nothing moved."
    )


def save_scene_frame(scene, folder, T_base_link):
    """One scene capture — everything the scene locator uses, raw: colour
    (BGR), the median depth of the kept frames, both intrinsics, the depth
    -> colour extrinsic, link -> colour, and the calibration in force."""
    write_frame(folder, scene_frame(scene, T_base_link))


def main():
    ap = cli_common.make_parser(__doc__)
    ap.add_argument("--set", default=time.strftime("set-%Y%m%d-%H%M"), help="name of the set (its folder)")
    ap.add_argument("--start-at", type=int, default=0, help="resume: the first grid index to record")
    ap.add_argument("--points", type=int, default=None, help="record only this many grid points")
    ap.add_argument("--empty", type=int, default=3, help="empty scenes to record at the end")
    ap.add_argument("--scene-calib", default=None, help="scene camera calibration yaml (default: the installed one)")
    ap.add_argument(
        "--no-prompt",
        action="store_true",
        help="do not wait for Enter (the stub harness only: nobody is placing anything)",
    )
    args = ap.parse_args()

    import rclpy

    from box_opening.perception.depth_source import BoxTopWatcher, camera_is_still, camera_pose_at
    from box_opening.perception.scene import SceneGrabber, load_scene_calibration
    from box_opening.primitives.core import Ctx, _gripper_leg, _plan_motion
    from box_opening.runtime.runner import Runner
    from box_opening.tasks import press_demo as pd
    from box_opening.tasks.press_demo import _state
    from box_opening.worlds import WorldStore

    cfg_path = args.container or cli_common.default_container_yaml()
    bench = args.bench_world or cli_common.default_bench_yaml()
    model = ContainerModel.load(cfg_path)
    cfg = load_press_demo(cfg_path)
    cli_common.refuse_unmeasured(model, args.execute)
    worlds = WorldStore(bench)
    why = table_refusal(worlds)
    if why:
        sys.exit(why)  # before rclpy: nothing started, nothing moved
    node, client = cli_common.init_runtime(args.execute)
    client._base_frame = worlds.base_frame
    runner = Runner(client)
    cli_common.apply_speed_scale(runner, args)
    table_z = worlds.table_top_z
    lid_z = table_z + float(model.dims[2])

    pnode = rclpy.create_node("box_opening_recorder")
    import threading

    from rclpy.executors import SingleThreadedExecutor

    pexec = SingleThreadedExecutor()
    pexec.add_node(pnode)
    pthread = threading.Thread(target=pexec.spin, name="perception", daemon=True)
    pthread.start()
    watcher = BoxTopWatcher(
        pnode, cfg, model, table_z, base_frame=worlds.base_frame, camera_cfg_name=worlds.wrist_camera_config
    )
    scene = SceneGrabber(pnode, keep=5, need_depth=True)
    T_base_link = load_scene_calibration(args.scene_calib)
    ctx = Ctx(model=model, cpose=None, client=client, worlds=worlds, config_path=cfg_path)
    ctx.press_cfg = cfg
    out = state_dir() / "detection_sets" / args.set
    container = Path(cfg_path).name

    def fly(legs, what):
        res = runner.run(legs, execute=args.execute)
        if any(not r.ok for r in res):
            pd.try_home(ctx, runner, args.execute, "STOP: %s failed — recording ends here" % what)
            sys.exit(1)

    def home():
        if pd.rest_distance(client.joints(), HOME) < AT_HOME_RAD:
            return  # already there: a zero-length plan is refused by the driver's gate
        fly([pd.build_home_leg(ctx, client.joints())], "home")

    def wait_enter(msg):
        log.info("\n[record] >>> %s" % msg)
        if args.no_prompt:
            time.sleep(0.5)
            return "y"
        return input("[record] Enter when done (s = skip this point, q = finish): ").strip().lower()

    def capture_scene(folder):
        t0 = time.monotonic()
        while (
            scene.missing() or len(scene.depths) < 5 or scene.depth_to_color() is None or scene.link_to_color() is None
        ) and time.monotonic() - t0 < 8.0:
            time.sleep(0.05)
        if scene.missing() or T_base_link is None:
            log.info(
                "[record] (no scene frames: %s)" % (", ".join(scene.missing()) or "the scene camera is uncalibrated")
            )
            return 0
        n = 0
        for k in range(SCENE_FRAMES):
            save_scene_frame(scene, folder / ("scene_%d" % k), T_base_link)
            n += 1
            time.sleep(0.4)
        return n

    def capture_wrist(folder, tag):
        """WRIST_FRAMES still wrist frames here, each in its own folder."""
        n, last_cam, last_stamp = 0, None, None
        t0 = time.monotonic()
        g = watcher.grab
        while n < WRIST_FRAMES and time.monotonic() - t0 < STILL_WAIT_S:
            time.sleep(0.05)
            if g.color_stamp is None or g.depth is None:
                continue
            stamp = (g.color_stamp.sec, g.color_stamp.nanosec)
            if stamp == last_stamp:
                continue
            last_stamp = stamp
            cam = camera_pose_at(g)
            if cam is None:
                continue
            if camera_is_still(last_cam, cam):
                if pd._save_wrist_frame(g, folder / ("%s_%d" % (tag, n)), tag) is not None:
                    n += 1
            last_cam = cam
        return n

    def record(point_dir, nominal):
        """Scene, search and — with a box — staging truth, for one placement."""
        home()
        n_scene = capture_scene(point_dir)
        n_wrist = {}
        for step in pd.search_targets(cfg):
            fly([pd.build_search_leg(ctx, step, client.joints(), pd._NeverStops)], "search pose %s" % step[0])
            n_wrist[step[0]] = capture_wrist(point_dir, "wrist_" + step[0])
        truth = {
            "container": container if nominal else None,
            "nominal_xy": nominal,
            "table_z": table_z,
            "scene_frames": n_scene,
            "wrist_frames": n_wrist,
            "truth_xyz": None,
            "truth_source": None,
        }
        if nominal is not None:
            ctx.cpose = ContainerPose(xyz=(nominal[0], nominal[1], table_z), yaw=0.0)
            fly([pd.build_approach_leg(ctx, cfg)], "staging")
            seen = {}
            got, status = pd.aim_button_at_staging(node, watcher, 0.0, runner=runner, tag="aim", out=seen)
            pd._save_wrist_frame(watcher.grab, point_dir / "staging", "staging")
            if got is not None:
                truth.update(
                    truth_xyz=[round(float(v), 5) for v in got[0]],
                    truth_source="close-up aim",
                    button_above_lid_mm=seen.get("button_up_mm"),
                )
                off = math.hypot(got[0][0] - nominal[0], got[0][1] - nominal[1])
                truth["truth_suspect"] = off > TRUTH_SUSPECT_M
                log.info(
                    "[record] truth: button at [%.4f, %.4f] — %.0f mm from where it was placed%s"
                    % (
                        got[0][0],
                        got[0][1],
                        1000 * off,
                        "  !! SUSPECT: re-place the box and redo this point (--start-at)"
                        if truth["truth_suspect"]
                        else "",
                    )
                )
            else:
                truth.update(truth_xyz=[nominal[0], nominal[1], lid_z], truth_source="placed (aim failed: %s)" % status)
                log.info("[record] the close-up aim found no button (%s) — truth is the placement, to ~1-2 cm" % status)
        (point_dir / "truth.json").write_text(json.dumps(truth, indent=1))
        return truth

    try:
        why = pd.readiness_refusal(client, args.execute, camera_cfg_name=worlds.wrist_camera_config)
        if why:
            sys.exit("[record] not starting: %s" % why)
        live = client.joints()
        if pd.rest_distance(live, HOME) > HOME_START_TOL_RAD:
            sys.exit("[record] start the arm at HOME first: ros2 run box_opening home_arm --execute")
        points = grid_points()[args.start_at :]
        if args.points is not None:
            points = points[: args.points]
        out.mkdir(parents=True, exist_ok=True)
        log.info(
            "[record] set %s: %d placements, then %d empty scenes -> %s" % (args.set, len(points), args.empty, out)
        )
        # the fingers closed, as the mission holds them: they hang in every wrist frame
        fly(
            [_gripper_leg(ctx, _state(live), "close", GRIPPER_CMD_CLOSED, ctx.worlds.push_name("bench", model=model))],
            "closing the fingers",
        )
        if not args.execute:
            x, y = points[0]
            leg, _ = _plan_motion(
                ctx,
                _state(client.joints()),
                "hover",
                ("pose", [x, y, lid_z + HOVER_M], model.press_quat([x, y, lid_z])),
                ctx.worlds.push_name("bench", model=model),
                TRANSIT_SPEED,
            )
            runner.run([leg], execute=False)
            log.info("[record] dry run: the first hover planned; nothing moved. Add --execute to record.")
            return
        for i, (x, y) in enumerate(points, start=args.start_at):
            point_dir = out / ("p%03d" % i)
            log.info("\n[record] ===== placement %d/%d at [%.2f, %.2f]" % (i + 1, args.start_at + len(points), x, y))
            try:
                leg, _ = _plan_motion(
                    ctx,
                    _state(client.joints()),
                    "hover",
                    ("pose", [x, y, lid_z + HOVER_M], model.press_quat([x, y, lid_z])),
                    ctx.worlds.push_name("bench", model=model),
                    TRANSIT_SPEED,
                )
            except RuntimeError as e:
                log.info("[record] [%.2f, %.2f] is not reachable from here (%s) — skipped" % (x, y, e))
                continue
            fly([leg], "the hover")
            ans = wait_enter("slide the box under the gripper, centred on it, hands clear")
            if ans == "q":
                break
            if ans == "s":
                continue
            record(point_dir, [x, y])
            home()
            write_manifest(out)  # after every placement: a crash keeps what is done
        if pd.rest_distance(client.joints(), HOME) > 0.05:
            home()  # quit or skipped at a hover
        for j in range(args.empty):
            if (
                wait_enter(
                    "EMPTY scene %d/%d: take the box off the table; leave everything else as it is"
                    % (j + 1, args.empty)
                )
                == "q"
            ):
                break
            record(out / ("p%03d" % (900 + j)), None)
            home()
            write_manifest(out)
        doc = write_manifest(out)
        log.info("\n[record] DONE — %d placements in %s (manifest.json)" % (len(doc["placements"]), out))
    except KeyboardInterrupt:
        sys.exit(130)
    finally:
        runner.finish()
        pexec.shutdown()
        pthread.join(timeout=3.0)
        try:
            pexec.remove_node(pnode)
            pnode.destroy_node()
        except Exception:
            pass


if __name__ == "__main__":
    main()
