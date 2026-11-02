"""press_demo — the camera-driven autonomous box opening (owner design
2026-08-24; one run, the reach-out, since 2026-10-28).

    sheppy: the `arm` and `planner` nodes (rammp-deployments, december_2026)
    ros2 launch box_opening press_demo.launch.py   # TF, OWL nodes
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    ros2 run box_opening press_demo --execute

Every box is pressed by the reach-out, however near it stands (owner
2026-10-28: "make the arm always go to that 45 degree looking pose even if
it's close enough to go straight up and down"; the straight-down staging
mode is gone). States logged one line each:

  SCENE   before the arm moves toward the box, the fixed scene camera finds
          it — sheppy's scene segmenter, or the scene OWL where that cannot
          answer, boxes it; the depth inside the box at lid height places it
          through config/camera_scene.yaml, or through TF where the robot's
          TF carries the camera's mount (world external_static_tf:
          flatbot). What it sees standing on the table goes into the
          planner's worlds. The wrist never searches (owner, 2026-10-22):
          no scene fix -> home, exit 2. The fingers close as the run starts.
  SPOT    where the lid will go, among what the scene camera sees standing
          on the table, the first spot the hand reaches: resolved before
          anything moves toward the box; none -> exit 4
          (tasks/reach_lid_off.resolve_lid_drop).
  LOOK    the hand tilted 30-45 deg back toward the arm, above the box: the
          wrist camera aims at the button from there (the oblique aim).
          No button -> home, exit 2. A box already open is not pressed.
  PRESS   the way in as ONE execution — the re-aim, the slide in turning
          the hand down, the stroke — in the driver's joint impedance, the
          spring pushing through the button's bottom and holding there
          (tasks/reach_press, tasks/depth_press). Nothing the run reads
          stops the arm; the e-stop and Ctrl+C do.
  LID     back to the look pose, the knob judged there, the press once more
          where it is not up; the fingers open on the way down, close on the
          knob, and the lift, the carry and the set-down where the cameras
          saw room; home (tasks/reach_lid_off). An open box goes straight to
          this.

--press-only stops after the press: rise, home, the lid on. --guarded flies
the reach-out's depth press position-controlled, with its torque guards and
at-rest checks, as before 2026-10-27, and ends at the press. --detect-only
reports the scene camera's fixes for 15 s with nothing moving;
--wrist-detect is the wrist-mount calibration observation (the look pose,
15 s of the wrist camera's fixes, home). Without --execute the run is
previewed up to the look pose.

--execute alone arms it: NO typed confirmation (owner decision
2026-08-24 — autonomous once started, Ctrl+C stops everything via the
stub-proven cancel path). The run is attended, e-stop in hand, and the
measure_me worksheet gate still applies. Without --execute nothing is ever
sent to the driver: the client is not armed.
"""

import argparse
import math
import os
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import rclpy

from box_opening import LOGGER as log
from box_opening.constants import (
    COMPLIANT_DEFAULT,
    GRIPPER_CMD_CLOSED,
    HOME,
    REST_TOL_RAD,
    SEARCH_SPEED,
    TRANSIT_SPEED,
    state_dir,
    viz_enabled,
)
from box_opening.detection_set import MISSIONS_SET, MissionFrames, scene_frame, wrist_frame, write_frame
from box_opening.geometry import ang_diff, yaw_about_world_z
from box_opening.models.container import (
    ContainerModel,
    ContainerPose,
    from_container,
    load_press_demo,
)
from box_opening.perception import clear_table
from box_opening.perception.depth_source import BoxTopWatcher
from box_opening.primitives.core import (
    CARRY_CLEAR_M,
    CLEAR_MARGIN_MAX_STEP_RAD,
    ClearMarginFree,
    Ctx,
    PlanState,
    _full_world,
    _interaction_world,
    _plan_motion,
)
from box_opening.primitives.look import look_joints, sweep_targets
from box_opening.runtime import fingers
from box_opening.runtime.runner import Runner
from box_opening.tasks import cli_common
from box_opening.worlds import WorldStore


def _state(joints, chain=0):
    return PlanState(joints=list(joints), chain=int(chain))


def rest_joints(cfg):
    """Where a run starts and ends: factory HOME, or the LOOK pose when
    open_box.park_tool_down is set.

    The tool-down rest used to be a surveyed joint vector standing over a
    fixed bench point, which is exactly the kind of constant a wheelchair
    invalidates. It is now HOME's own wrist turned down: same saving (no
    2.4-2.9 rad wrist flip either side of the run), no bench in it."""
    return look_joints(HOME) if cfg.park_tool_down else list(HOME)


def rest_distance(live, joints):
    """Worst per-joint distance (wrap-aware) from a rest pose."""
    return max(abs(ang_diff(a, b)) for a, b in zip(live, joints, strict=False))


def search_targets(cfg):
    """The search, in order: look, then sweep each way.

    (name, joint target, speed) triples. The look is HOME's wrist turned
    down — one joint moving, 1.72 s with no planner call, against 3.89 s to
    fly to the old fixed scan pose (measured on the planner 2026-09-15).
    Nothing here is expressed in the bench frame, so the whole search holds
    wherever the chair is standing.

    The sweeps run slower than the look because they exist to SEE: the
    coarse detector reads frames while the arm pans, and the first sighting
    cancels the leg (build_search_leg)."""
    look = look_joints(HOME)
    steps = [("look", look, TRANSIT_SPEED)]
    for name, q in zip(("sweep:left", "sweep:right"), sweep_targets(look, cfg.search_arc_rad), strict=False):
        steps.append((name, q, SEARCH_SPEED))
    return steps


def build_search_leg(ctx, step, start_joints, watcher):
    """One search step as a leg that ends the moment the camera sees a box.

    Joint-space (no bench frame is involved before anything is detected)
    and planned in the bench world, whose unseen-container keep-out band
    keeps a pre-detection move above whatever is standing there.

    stop_when is polled while the leg flies: the first coarse fix cancels
    it, so the arm stops where it saw the box instead of finishing a pan it
    no longer needs."""
    name, target, speed = step
    world = ctx.worlds.push_name("bench", model=ctx.model)
    leg, _ = _plan_motion(
        ctx,
        _state(start_joints),
        name,
        ("joints", [float(v) for v in target]),
        world,
        speed,
    )
    leg.stop_when = lambda progress=None: watcher.coarse_fix() is not None
    return leg


class _NeverStops:
    """A watcher stand-in for a search leg that must fly to the end: the
    wrist-mount observation (--wrist-detect) parks at the look pose and
    reports from there, and record_detection_set stands at each pose."""

    @staticmethod
    def coarse_fix(now=None):
        return None


# How long the scene camera gets to find the box before the arm has moved
# at all: sheppy's scene segmenter fixes it in about 0.35 s (two agreeing
# frames at 5 Hz; scene_source.commit_fix), and the scene OWL fallback needs
# about 2.6 s for two boxes (~1.3 s an inference once enabled).
SCENE_LOCATE_S = 5.0  # was 3.0: the OWL's first box came at 2.2-2.7 s — too tight
# ... and the run's FIRST look, before anything moves, keeps at it longer:
# flatbot 2026-10-27 15:34:01 refused all three candidates the OWL offered
# in 5 s (0.12, 26-28 depth points in each) and the run went home; 25 s
# later the next run had the box 0.83 s in. Looking longer costs only time.
SCENE_FIRST_LOCATE_S = 15.0
# How long the straight-down aim gets (aim_button_at_staging's default:
# record_detection_set's, from its staging over a placed box; the reach-out
# gives its oblique aim its own). From 12 cm the lid fills a third of the
# frame and the hit rate is high; this is generous.
STAGING_FIX_S = 2.5
RESIDUALS_FILE = state_dir() / "scene_calib" / "residuals.jsonl"


def residuals_path_for(scene_calib_arg):
    """Where the scene-vs-wrist pairs for this run's calibration are kept:
    the real file, unless the run uses its own calibration file
    (--scene-calib), whose pairs sit next to that file. The old stub
    harness wrote ~90 synthetic pairs into the real file before this
    (2026-09-17)."""
    if scene_calib_arg:
        return Path(scene_calib_arg).resolve().with_name("residuals.jsonl")
    return RESIDUALS_FILE


def scene_source_label(fix):
    """'SEG 0.88, 2 sightings (segmenter+segmenter)': the detector behind a
    scene fix (the newest sighting's), its score, and the sightings that
    agreed on it (scene_source.commit_fix)."""
    source = getattr(fix, "source", "owl")
    n = int(getattr(fix, "n_sightings", 1))
    sources = tuple(getattr(fix, "sources", ()) or (source,))
    return "%s %.2f, %d sighting%s (%s)" % (
        "SEG" if source == "segmenter" else "OWL",
        float(fix.score),
        n,
        "" if n == 1 else "s",
        "+".join(sources),
    )


# The scene camera's box this far (xy) from the button the wrist found, on
# the last run under the calibration in force: scene_trust distrusts it.
# Since the wrist stopped searching (2026-10-22) a distrusted fix is still
# flown and the distrust only printed: the wrist's aim decides. Its fixes
# were 9-11 cm off after a re-aim (2026-09-23) and 12 cm off under a bad
# calibration (2026-09-24).
SCENE_TRUST_M = 0.05


def scene_trust(path, calib):
    """(trusted, why): the scene camera's fix is flown unless, on the last
    run under the calibration in force, its box was more than
    SCENE_TRUST_M from the button the wrist then found — what a re-aimed
    camera looks like. The cabinet-door tag used to decide this, and
    misfired (scene_calib.TAG_MOVED_PX). main flies a distrusted fix anyway
    and prints why: with no wrist search, the wrist's aim is the check. The
    pairs were recorded by the straight-down staging aim, one a run, until
    that mode was scrapped (2026-10-28): the reach-out records none, so
    this judges by the last pair recorded under the calibration in force."""
    from box_opening.perception.scene_refine import calibrated_at, load_pairs

    try:
        scene, wrist = load_pairs(path, since=calibrated_at(calib))
    except OSError:
        return True, None
    if not len(scene):
        return True, None
    off = math.hypot(float(wrist[-1][0] - scene[-1][0]), float(wrist[-1][1] - scene[-1][1]))
    if off > SCENE_TRUST_M:
        return False, "on the last run its box was %.0f mm from the button the wrist found" % (1000 * off)
    return True, None


def _table_offset_words(off, off_raw):
    """The AIM line's word on the wrist's table against the scene camera's
    (button_aim.ButtonAimer.table_offset_m; nan: none)."""
    if math.isfinite(off):
        return ", the wrist reads its table %+.0f mm from the scene's%s" % (
            1000 * off,
            ", the button's top raised %.0f mm" % (-1000 * off) if off < 0.0 else "",
        )
    if math.isfinite(off_raw):
        return ", the wrist reads its table %+.0f mm from the scene's (not taken out)" % (1000 * off_raw)
    return ""


# perception/viz, the run's perception drawn for Foxglove, is never loaded
# by the mission's thread: start_viz only starts the viz's own thread
# (_run_viz), and only with it on, and that thread loads it once the
# mission's thread waits. Loaded at a module level it was ~4 ms of every
# launch, on or off; in start_viz ~1.8 ms with it on (reviews 2026-10-28).
# A run with it off — the default (constants.viz_enabled) — never loads it,
# and every hook (viz_hook) is then a dict lookup
VIZ_MODULE = "box_opening.perception.viz"
# The viz's thread loads it at the first VIZ_BOOT_S in which the mission's
# thread ran under VIZ_BOOT_RUN_FRAC of the time (thread_runtime): the
# viz's own gate (viz.PERIOD_S, viz.IDLE_RUN_FRAC), which is not loaded yet
VIZ_BOOT_S = 0.25
VIZ_BOOT_RUN_FRAC = 0.2
_VIZ_STOPPED = None  # the started viz thread's stop (an Event): stop_viz sets it


def thread_runtime(tid):
    """A callable saying how long (ns) thread `tid` of this process has run
    (/proc/self/task/<tid>/schedstat, read with one pread), or None where
    the kernel does not keep it."""
    try:
        fd = os.open("/proc/self/task/%d/schedstat" % int(tid), os.O_RDONLY)
    except OSError:
        return None

    def ran():
        return int(os.pread(fd, 64, 0).split()[0])

    try:
        ran()
    except (OSError, ValueError, IndexError):
        os.close(fd)
        return None
    return ran


def start_viz(node, model, frame, env=None):
    """perception/viz for this run on `node`, when it is turned on
    ($BOX_OPENING_VIZ=1, constants.viz_enabled: off by default) and
    `node` can publish (else said): a thread of its own started (~0.3 ms)
    and nothing more — that thread loads the module and runs the viz
    (_run_viz). Its stop (an Event), else None."""
    global _VIZ_STOPPED
    if _VIZ_STOPPED is not None:
        stop_viz(False)  # one viz at a time
    if not viz_enabled(env):
        return None
    if not callable(getattr(node, "create_publisher", None)):
        log.info("[viz] (no Foxglove view of this run: %s publishes nothing)" % type(node).__name__)
        return None
    stopped = threading.Event()
    threading.Thread(
        target=_run_viz, args=(node, model, frame, threading.get_native_id(), stopped, env), name="viz", daemon=True
    ).start()
    _VIZ_STOPPED = stopped
    return stopped


def _run_viz(node, model, frame, mission, stopped, env):
    """The viz's thread: perception/viz loaded once the mission's thread
    (native id `mission`) waits — a VIZ_BOOT_S it ran under
    VIZ_BOOT_RUN_FRAC of — then the viz run on this thread until stop_viz
    (viz.run, the gate reading that thread's run time); nothing at all once
    `stopped`. Never raises."""
    try:
        ran = thread_runtime(mission)
        last = None if ran is None else (ran(), time.monotonic())
        while not stopped.wait(VIZ_BOOT_S):
            if ran is None:
                break  # no reading, no gate (viz's own)
            now = (ran(), time.monotonic())
            if now[0] - last[0] < VIZ_BOOT_RUN_FRAC * (now[1] - last[1]) * 1e9:
                break
            last = now
        if stopped.is_set():
            return
        from box_opening.perception import viz

        viz.run(node, model, frame, ran, stopped, env=env)
    except Exception as e:  # noqa: BLE001 -- no viz is no reason to stop a run
        log.info("[viz] (no Foxglove view of this run: %s)" % (str(e) or type(e).__name__))


def viz_hook(name, *args):
    """perception/viz's entry point `name` (scene, aim, press, grip, spot)
    called with `args` once the viz runs (viz._VIZ, set last: never a module
    still loading), else nothing. None of them raises or waits: each hands
    over what was computed anyway."""
    viz = sys.modules.get(VIZ_MODULE)
    if viz is not None and getattr(viz, "_VIZ", None) is not None:
        getattr(viz, name)(*args)


def stop_viz(last=True):
    """The viz stopped (viz.stop: `last`, after it published what the run
    left; not `last` — the run unwinds from a Ctrl+C — at once); its thread,
    should it not have started the viz yet, never does."""
    global _VIZ_STOPPED
    stopped, _VIZ_STOPPED = _VIZ_STOPPED, None
    if stopped is None:
        return
    stopped.set()
    stop = getattr(sys.modules.get(VIZ_MODULE), "stop", None)  # None: not loaded, or still loading
    if stop is not None:
        stop(last)


def aim_button_at_staging(
    node,
    watcher,
    yaw,
    timeout_s=STAGING_FIX_S,
    runner=None,
    tag="aim",
    out=None,
    lid_z=None,
    prior_xy=None,
    min_hits=None,
):
    """The wrist's press aim: the button's circle alone (owner's design
    2026-09-17 — no OWL and no box-footprint gate here; the scene camera
    already said which thing is the box), the press point its centre.
    Straight down from a staging over the box by default (record_detection_set);
    the reach-out's oblique look passes lid_z / prior_xy / min_hits. Returns (got, status) where got is
    ((x, y, z), yaw) like the watcher's fix, or None — z raised by the
    wrist's table offset where its hits agree on one (ButtonAimer.fix). The
    aimed frame is saved and the aim logged (`runner`) every time.

    lid_z / prior_xy: the lid height to expect and where the button should
    be — what the reach-out press's oblique look aims with (button_aim);
    min_hits: the agreeing hits a tilted view's fix takes (unset, MIN_HITS).
    `out` gets button_up_mm and measured_xyz, the press point as the wrist
    measured it: what the wrist's own later readings are judged against."""
    from box_opening.perception.button_aim import ButtonAimer

    # only what is set: unset, the call is the straight-down aim's own
    kw = {k: v for k, v in (("lid_z", lid_z), ("prior_xy", prior_xy), ("min_hits", min_hits)) if v is not None}
    aimer = ButtonAimer(watcher.grab, watcher.model, watcher.table_z, **kw)
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        _spin_detect(node)
        n0 = getattr(aimer, "frames", None)
        fixed = aimer.tick()
        viz_hook("aim", aimer, fixed, n0, tag)  # the frame it judged and what it found, drawn off this thread
        if fixed:
            xyz = aimer.fix()  # its z raised by the wrist's table offset, where that is agreed
            measured = aimer.fix(measured=True)
            agreeing = aimer.fix_hits()  # which hits the fix was made from
            s = aimer.last_sighting
            up_m = aimer.button_above_lid_m()
            up_mm = None if up_m is None else 1000.0 * up_m
            off, off_raw = aimer.table_offset_m(), aimer.table_offset_m(raw=True)
            if out is not None:
                out["button_up_mm"] = up_mm  # the caller decides (refuse_open_box)
                out["measured_xyz"] = tuple(float(v) for v in measured)
            # the agreeing hits apart from all of them: 15:19 2026-10-27 printed
            # "4/19 frames agree" where 3 of its 4 hits did
            log.info(
                "[press_demo] AIM: button circle at pixel (%.0f, %.0f) r %.0f px, %.0f mm across, %s above its own "
                "lid%s%s; pressing its centre; %d of %d hits agree (%d frames, %.1f s)"
                % (
                    s.uv[0],
                    s.uv[1],
                    s.r_px,
                    1000 * s.diameter_m,
                    "height unknown" if up_mm is None else "%+.1f mm" % up_mm,
                    "" if not s.oblique else ", depth on %.0f%% of it" % (100 * s.disc_fill),
                    _table_offset_words(off, off_raw),
                    len(agreeing),
                    aimer.hits,
                    aimer.frames,
                    time.monotonic() - t0,
                )
            )
            folder = save_aim_frame(watcher.grab, s, tag)
            # the press point, when the press goes on it (aim_source), and its frame
            watcher.last_aim, watcher.last_aim_capture = tuple(float(v) for v in xyz), folder
            if runner is not None:
                runner.note(
                    tag,
                    seam_uv=[round(v, 1) for v in s.uv],
                    r_px=round(s.r_px, 1),
                    diameter_mm=round(1000 * s.diameter_m, 1),
                    press_xyz=[round(v, 4) for v in xyz],
                    measured_z=round(measured[2], 4),
                    button_above_lid_mm=None if up_mm is None else round(up_mm, 1),
                    disc_fill=round(s.disc_fill, 2) if s.oblique else None,
                    # the wrist's table against the scene camera's: taken out (press_xyz's
                    # z raised by it, never lowered), and as read, taken out or not —
                    # drift shows here before it fails a run
                    table_offset_mm=round(1000 * off, 1) if math.isfinite(off) else None,
                    table_offset_raw_mm=round(1000 * off_raw, 1) if math.isfinite(off_raw) else None,
                    hits=aimer.hits,
                    frames=aimer.frames,
                    aim_s=round(time.monotonic() - t0, 2),
                    # the hits the fix was made from (indices), and each hit's time since
                    # the aimer was built and its knob reading: when the fix could have
                    # come, and whether the knob was read alike
                    agreeing=list(agreeing),
                    hit_s=[round(float(v), 2) for v in aimer.hit_s],
                    above_lid_mm=[round(1000 * float(v), 1) if math.isfinite(v) else None for v in aimer.above_lid],
                    # the wrist's depth-vs-colour stamp offsets: what PAIR_TOL_S rests on
                    pair_dt_ms=[round(v, 1) for v in aimer.pair_dt_ms] or None,
                    capture=None if folder is None else str(folder),
                )
            return (xyz, yaw), aimer.status()
        # A tick that took a frame goes straight on to the next: judging a
        # tilted one costs 45-75 ms (StillDepth and the oblique find), and
        # with a sleep after every tick the reach-out aims of 2026-10-27 took
        # 10.1-14.0 of the wrist OAK's 15 frames a second. Only a tick that
        # took no new frame (none landed, or its depth not yet) waits a beat.
        if getattr(aimer, "frames", None) == n0:
            time.sleep(0.02)
    watcher.last_reject = aimer.last_why  # for the frame dump's caption
    return None, aimer.status()


CAPTURES_DIR = state_dir() / "captures"
AIM_CAPTURES_KEPT = 20  # newest folders kept of each every-run capture (aim, pop); older ones are pruned


_PENDING_WRITES = []  # capture writes still in flight (background=True), joined by wait_capture_writes
_WRITE_LOCK = threading.Lock()  # one write at a time: two aims in one second share a folder


def wait_capture_writes(timeout_s=5.0):
    """Let every capture write still in flight finish (at most `timeout_s`
    in all): before anything copies a capture folder (MissionFrames.close)."""
    end = time.monotonic() + timeout_s
    for t in list(_PENDING_WRITES):
        t.join(max(0.0, end - time.monotonic()))
    _PENDING_WRITES[:] = [t for t in _PENDING_WRITES if t.is_alive()]


def _save_wrist_frame(grab, folder, caption, draw=None, background=False, after=None):
    """One wrist frame (colour, depth, K, camera pose, stamp) into `folder`
    in the recorder's format (replay: scripts/record_scan_frames.py
    --analyze), plus a JPEG with `caption` and whatever `draw(img)` adds,
    then `after()`. Returns the folder, or None when there is no frame or
    no pose for it. With `background` the frame is taken now and written on
    a thread (compressing it is ~45 ms the next leg need not wait for): the
    folder exists at once, its files a moment later (wait_capture_writes)."""
    import numpy as np

    from box_opening.perception.depth_source import camera_pose_at

    if grab is None or grab.color is None or grab.depth is None or grab.k is None or grab.color_stamp is None:
        return None
    pose = camera_pose_at(grab)
    if pose is None:
        return None
    # one snapshot: the grabber assigns new arrays per message, so these
    # stay this frame while a thread writes them, and the JPEG is the npz's
    arrays = wrist_frame(grab, pose)
    img = None
    try:
        import cv2

        img = np.ascontiguousarray(arrays["color"]).copy()
        if draw is not None:
            draw(img)
        cv2.putText(img, str(caption)[:90], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    except Exception:
        img = None

    def job():
        with _WRITE_LOCK:
            try:
                write_frame(folder, arrays)
            except Exception as e:  # a full disk: the run goes on without its evidence
                log.info("[press_demo] (wrist frame not written to %s: %s)" % (folder, e))
            if img is not None:
                try:
                    import cv2

                    cv2.imwrite(str(folder / "frame_000.jpg"), img)
                except Exception:
                    pass
            if after is not None:
                after()

    if not background:
        job()
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    t = threading.Thread(target=job, name="capture-write", daemon=False)  # not daemon: exit waits for it
    _PENDING_WRITES[:] = [p for p in _PENDING_WRITES if p.is_alive()] + [t]
    t.start()
    return folder


def dump_wrist_frame(watcher, tag):
    """Save the wrist camera's CURRENT frame and the detector's last reject
    reason, so "the wrist did not see the box" can be looked at afterwards
    instead of argued about (owner, 2026-09-16). Lands in
    ~/.ros/box_opening/captures/wrist-<tag>-<stamp>/. Returns the
    folder or None."""
    folder = CAPTURES_DIR / ("wrist-%s-%s" % (tag, time.strftime("%Y%m%d-%H%M%S")))
    out = _save_wrist_frame(getattr(watcher, "grab", None), folder, watcher.last_reject or "no reject recorded")
    if out is None:
        return None
    (out / "reason.txt").write_text("%s\n%s\n" % (watcher.status(), watcher.last_reject))
    log.info("[press_demo] wrist frame saved for inspection: %s" % out)
    return out


def save_aim_frame(grab, sighting, tag="aim"):
    """The frame the press was aimed on, with the seam circle (green) and
    the PRESS POINT — its centre (red +) — drawn, every run, so a miss is a
    ten-second look instead of a blind read of the run log (the 2026-09-17
    miss had no frame). Keeps the newest AIM_CAPTURES_KEPT folders."""
    folder = CAPTURES_DIR / ("wrist-%s-%s" % (tag, time.strftime("%Y%m%d-%H%M%S")))

    def draw(img):
        import cv2

        u, v = int(round(sighting.uv[0])), int(round(sighting.uv[1]))
        cv2.circle(img, (u, v), int(round(sighting.r_px)), (0, 255, 0), 2)
        cv2.drawMarker(img, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 26, 2)

    caption = "press point: the circle's centre, button %.0f mm, r %.0f px" % (
        1000 * sighting.diameter_m,
        sighting.r_px,
    )
    # written off the critical path: the next leg need not wait for zlib
    out = _save_wrist_frame(grab, folder, caption, draw, background=True, after=lambda: _prune_captures(tag))
    if out is None:
        return None
    log.info("[press_demo] aim frame saved: %s" % out)
    return out


def _prune_captures(tag):
    """Keep the newest AIM_CAPTURES_KEPT wrist-<tag>-* folders."""
    try:
        olds = sorted(CAPTURES_DIR.glob("wrist-%s-*" % tag))
        for d in olds[:-AIM_CAPTURES_KEPT]:
            for f in d.iterdir():
                f.unlink()
            d.rmdir()
    except Exception:
        pass


def build_home_leg(ctx, start_joints, joints=None, chain=0):
    """The no-box / recovery exit: back to the rest pose in the bench world,
    above the band where an unlocated container may be standing."""
    world = ctx.worlds.push_name("bench", model=ctx.model)
    leg, _ = _plan_motion(
        ctx,
        _state(start_joints, chain),
        "home",
        ("joints", list(HOME if joints is None else joints)),
        world,
        TRANSIT_SPEED,
    )
    return leg


def build_approach_leg(ctx, cfg, start_joints=None):
    """Staging directly above the button of ctx.cpose, cfg.staging_m up, in
    the full world, from `start_joints` (the live joints when not given):
    record_detection_set's close-up look over a placed box. press_demo no
    longer flies it (the reach-out presses every box, 2026-10-28)."""
    m = ctx.model
    button = from_container(ctx.cpose, m.button_offset)
    quat = m.press_quat(button)  # the same attitude the press holds
    staging = [button[0], button[1], button[2] + cfg.staging_m]
    leg, _ = _plan_motion(
        ctx,
        _state(ctx.client.joints() if start_joints is None else start_joints),
        "approach:staging",
        ("pose", staging, quat),
        _full_world(ctx),
        TRANSIT_SPEED,
    )
    return leg


# Is the knob UP? Two instruments, and everything the bench has produced:
#
#   the wrist CAMERA (the pop check from the hop; the aim's disc against
#   its own lid ring): down -0.8, 0.2, +1 mm; up 9.2, 10.1, 11.1, +10 mm.
#   the TOUCH (pad height above the predicted button top): pads on the
#   closed button -3.7 .. +6.2 mm (the button is slightly domed and the
#   pads' edges land on different parts of it); pads on the knob already up
#   +9.6, +10.8, +17.7 mm.
#
# This knob stands about 10 mm proud when up with the lid on the container.
# "Popped" began at 12 mm for four days, on the strength of the one +17.7
# touch — so a good pop read "NOT POPPED", the mission pressed again, and
# the second press shut the box (owner, 2026-09-21 14:16: "it pressed the
# button and opened it but then ... went down again and CLOSED the box").
# The touch is the weaker instrument: the raised knob is soft, and that
# second stroke rode it all the way down without its 3 Nm guard noticing —
# only the camera can be trusted to say the knob is up. NO band in between
# refuses a push: a 5-12 mm "rim" band (2026-09-17) refused a good +6.2 mm
# contact and homed the arm.
KNOB_UP_MIN_MM = 6.0  # the camera: at or above this the knob is up
POPPED_MIN_M = 0.008  # the touch: pads stopped this far above the button top — on the raised knob
POPPED_MAX_M = 0.035  # ... and above this it is not this button at all


def refuse_open_box(up_mm):
    """Why this box must not be pressed, or None: the aim found its button
    already standing `up_mm` above its own lid. Pressing an open OXO shuts
    it — three runs did exactly that on 2026-09-21 (13:35, 14:14, 14:14).
    The full run takes such a box's lid off unpressed (reach_press); this
    is what --press-only and --guarded stop on."""
    if not knob_is_up(up_mm):
        return None
    return (
        "the button already stands %.0f mm above its lid — this box is already OPEN (left up by the "
        "run before?). Pressing it would shut it. Push the button down flush and run again, or run without "
        "--press-only and --guarded: the full run takes an open box's lid off" % up_mm
    )


def knob_is_up(up_mm):
    """The pop check's verdict on a camera reading (mm above the lid; None: no reading)."""
    return up_mm is not None and up_mm >= KNOB_UP_MIN_MM


def touch_verdict(off_m):
    """ "press" | "popped" | "foreign" for a touch off_m above the predicted
    button top. Pure."""
    if off_m > POPPED_MAX_M:
        return "foreign"
    if off_m >= POPPED_MIN_M:
        return "popped"
    return "press"


DROP_X_M = (0.28, 0.65)  # set-down zone: inside the tool-down reach band
DROP_Y_M = (-0.42, 0.42)
DROP_EDGE_M = 0.03  # a lid set down on a measured table: this far inside its edge, past its own half-width


def drop_zone(m, worlds):
    """(x0, x1, y0, y1) the lid may be set down in: on the table the scene
    camera measured this run (inset so the whole lid lands on it — the
    planner then says what the arm reaches), or the surveyed bench's zone."""
    ext = worlds.table_extent
    if ext is None:
        return (*DROP_X_M, *DROP_Y_M)
    inset = 0.5 * max(float(m.lid_dims[0]), float(m.lid_dims[1])) + DROP_EDGE_M
    return (ext[0] + inset, ext[1] - inset, ext[2] + inset, ext[3] - inset)


LID_DIAG_EXTRA_M = 0.03  # a diagonal drop spot this much farther out than an axis one (lid_drop_candidates)


def lid_place_min_clear(m):
    """Smallest planar container-origin-to-lid_place distance that leaves
    the place hover IK-solvable: both footprint half-diagonals plus
    gripper-body room. Field 2026-08-26: IK_FAIL at 0.073 m separation,
    clean plan at 0.162 m."""
    return (math.hypot(m.dims[0], m.dims[1]) + math.hypot(m.lid_dims[0], m.lid_dims[1])) / 2 + 0.05


def lid_drop_candidates(m, cpose, table_z, prefer_xy=None, zone=None):
    """Every drop spot beside the DETECTED box, best first: a list of
    (xyz, direction), each lid_place_min_clear from the box and inside the
    set-down zone (`zone`: drop_zone's (x0, x1, y0, y1); the bench's when
    None).

    The four base-axis directions are offered first. The planner holds the
    (padded) container as a cuboid aligned to the BASE axes whatever the
    box's yaw, so a spot on a diagonal sits ~20 mm nearer that cuboid's
    corner than a spot at the same distance on an axis — and the carry pose
    above one 27 deg off the -y axis had no IK in any attitude, while the
    four axis spots around the same box all planned (field 2026-09-17).
    A preferred point (`prefer_xy`), when given, picks which axis side is tried
    first: the one nearest its direction from the box.

    The four diagonals come after them, LID_DIAG_EXTRA_M farther out (past
    that corner): a box at the edge of the table the scene camera saw has
    no axis spot inside it (flatbot 2026-10-23: the box at [0.39, -0.20],
    the zone x 0.43-0.63, y -0.63 to -0.23 — every axis spot out, a
    diagonal in). Without the scene camera's points these are the lid's
    spots (reach_lid_off.resolve_lid_drop), the first the hand reaches
    taken."""
    need = lid_place_min_clear(m)
    bx, by = float(cpose.xyz[0]), float(cpose.xyz[1])
    dirs = []
    if prefer_xy is not None:
        dx, dy = float(prefer_xy[0]) - bx, float(prefer_xy[1]) - by
        if math.hypot(dx, dy) > 1e-6:
            dirs.append((math.copysign(1.0, dx), 0.0) if abs(dx) >= abs(dy) else (0.0, math.copysign(1.0, dy)))
    for d in ((0.0, -1.0), (0.0, 1.0), (1.0, 0.0), (-1.0, 0.0)):
        if d not in dirs:
            dirs.append(d)
    x0, x1, y0, y1 = zone if zone is not None else (*DROP_X_M, *DROP_Y_M)
    out = []
    for ux, uy in dirs:
        x, y = bx + ux * need, by + uy * need
        if x0 <= x <= x1 and y0 <= y <= y1:
            out.append(([x, y, float(table_z)], (ux, uy)))
    diag = (need + LID_DIAG_EXTRA_M) / math.sqrt(2.0)
    for ux, uy in ((1.0, -1.0), (1.0, 1.0), (-1.0, -1.0), (-1.0, 1.0)):
        x, y = bx + ux * diag, by + uy * diag
        if x0 <= x <= x1 and y0 <= y <= y1:
            out.append(([x, y, float(table_z)], (ux / math.sqrt(2.0), uy / math.sqrt(2.0))))
    return out


CLEAR_SPOTS_Y_M = 0.45  # the map is searched for a drop spot this far either side of the box


def clear_lid_drops(m, cpose, tmap, table_z, judged_out=None, hand_at=None, cache=None):
    """(drops, spots, why): where the lid may go among what the cameras see
    standing on the table (perception/clear_table), best first: as
    ContainerPoses on the table at `table_z` — each one's yaw the turn the
    hand comes down with, the lid turning with it — and as the Spots that
    say how much room each has, and how far the hand turns to it; and, when
    there is none, why.

    The spot used to be derived from the box alone (lid_drop_candidates):
    flatbot 2026-10-23 17:57 put it on the front edge of the microwave
    standing 14 cm in front of the box, and the wrist camera came down on
    it. The search spans the reach band in x and CLEAR_SPOTS_Y_M either side
    of the box; the first the hand reaches is taken
    (reach_lid_off.resolve_lid_drop).

    `hand_at` is which way the hand holding the lid faces
    (clear_table.hand_yaw; reach_lid_off.spot_candidates measures it on the
    arm). None: as the reach-out grips the knob — the press attitude turned
    to the button's bearing (reach_press.reach_quat; depth_press.down_quat
    turns the hand down to it), where every spot's set-down (place_quat)
    is the press attitude at the spot, turned. `cache`: clear_table.clear_spots'
    (the room judged on this map before, kept)."""
    lid_h = float(m.lid_dims[2])
    bx, by = float(cpose.xyz[0]), float(cpose.xyz[1])
    if hand_at is None:
        from box_opening.tasks.reach_press import reach_quat

        hand_at = clear_table.hand_yaw(reach_quat(m, from_container(cpose, m.button_offset)[:2], 0.0))
    spots, why = clear_table.clear_spots(
        tmap,
        (bx, by),
        (DROP_X_M[0], DROP_X_M[1], by - CLEAR_SPOTS_Y_M, by + CLEAR_SPOTS_Y_M),
        lid_place_min_clear(m),
        lambda xy: m.press_quat((xy[0], xy[1], float(table_z))),
        0.5 * max(float(m.lid_dims[0]), float(m.lid_dims[1])),
        lid_h,  # the fingertips at the release: the lid's height up, at the least
        float(m.dims[2]) + lid_h + CARRY_CLEAR_M,  # ... and on the carry (Place.hover_for's floor)
        lid_h,
        judged_out=judged_out,
        hand_at=hand_at,
        cache=cache,
    )
    return [ContainerPose(xyz=(s.xy[0], s.xy[1], float(table_z)), yaw=s.turn) for s in spots], spots, why


def wrist_points(watcher):
    """The wrist camera's newest frame as points in the base frame, or None."""
    from box_opening.perception.depth_source import camera_pose_at

    g = getattr(watcher, "grab", None)
    if g is None or g.depth is None or g.k is None or g.color_stamp is None:
        return None
    pose = camera_pose_at(g)
    return None if pose is None else clear_table.depth_points(g.depth, g.k, *pose)


def room_map(watcher, scene_fix, cpose, at_staging):
    """The table map the lid's spot is chosen on (clear_lid_drops): the
    scene camera's points around the box, moved by its error at the box
    (the wrist measured where the box is), and, `at_staging`, the wrist's
    own frame looking straight down at the table around it — the scrapped
    staging run's look (until 2026-10-28; every caller now passes False,
    test_tasks replays 17:57's with it). None without the scene's points."""
    import numpy as np

    pts = getattr(scene_fix, "table_points", None)
    if pts is None or math.isnan(scene_fix.table_z_scene):
        return None
    shift = [float(cpose.xyz[0]) - scene_fix.pose.xyz[0], float(cpose.xyz[1]) - scene_fix.pose.xyz[1], 0.0]
    clouds = [(np.asarray(pts, float) + shift, scene_fix.table_z_scene)]
    if at_staging:
        clouds.append((wrist_points(watcher), scene_fix.table_z_scene))
    return clear_table.table_map(clouds, cpose.xyz[:2])


def container_half(m):
    """The half-width of the box's footprint about its centre, whatever its
    yaw: a round one's radius, a square one's half-diagonal."""
    return 0.5 * (max(m.dims[0], m.dims[1]) if m.round else math.hypot(m.dims[0], m.dims[1]))


def place_quat(m, lid):
    """The hand's attitude setting the lid down at `lid`: the press
    attitude, turned by the lid's yaw (clear_lid_drops)."""
    return yaw_about_world_z(m.press_quat(lid.xyz), float(lid.yaw))


def _rise_over_button(ctx, cfg, st, name):
    """A rise back to staging height above the BUTTON, from wherever the arm
    stands in the corridor it came down. Built from the button and nothing
    else: ctx.last_world / last_pose / lid_at belong to whatever was planned
    last, and after a press that is the LOOKAHEAD — the lid's set-down world
    and a pose above the drop spot, 20 cm away (the refused "rise" of bench
    2026-09-21). The name starts with "retreat": a rise out of the corridor
    just descended may fly at transit speed in an interaction world
    (runner._refusal), like the retreat and the lift."""
    m = ctx.model
    button = from_container(ctx.cpose, m.button_offset)
    ctx.lid_at = None  # nothing has been placed: no phantom lid in these worlds
    world = _interaction_world(ctx, button[2], cfg.travel_m, "button")
    ctx.last_world = world
    staging = [button[0], button[1], button[2] + cfg.staging_m]
    rise, st = _plan_motion(ctx, st, name, ("pose", staging, m.press_quat(button)), world, TRANSIT_SPEED)
    return world, rise, st


def _spin_detect(node):
    try:
        rclpy.spin_once(node, timeout_sec=0.1)
    except RuntimeError as e:
        # rclpy teardown artifact: our SIGINT handler raising inside a
        # subscription take surfaces as RuntimeError, not KeyboardInterrupt
        raise KeyboardInterrupt from e


# cuRobo takes 17-21 s to load after sheppy starts the planner (measured
# 2026-09-16 from the planner's own log), and it advertises nothing until it
# has. A run started inside that window is waited for, not failed.
PLANNER_WAIT_S = 30.0


def readiness_refusal(client, execute, planner_wait_s=PLANNER_WAIT_S, camera_cfg_name=None):
    """Why this run must not start, or None. Checked before ANYTHING is
    actuated — the gripper close included.

    Field 2026-09-16: the stack had just been restarted, the fingers closed,
    and only then did the run die on `set_world service unavailable`. The
    close was the first actuation and the planner was still loading."""
    if not client.planner_reachable(timeout_s=1.0):
        log.info("[press_demo] waiting up to %.0f s for the planner to finish " "loading" % planner_wait_s)
        if not client.planner_reachable(timeout_s=planner_wait_s):
            return (
                "the planner is not up after %.0f s — is sheppy's `planner` node "
                "running? (docker logs sheppy-planner should end in "
                "'rammp_curobo ready'). Nothing moved." % planner_wait_s
            )
    if execute and not client.driver_reachable(timeout_s=5.0):
        return (
            "the arm driver is not answering — is sheppy's `arm` node up? "
            "(its log is in ~/.sheppy/logs/arm/). Nothing moved."
        )
    if execute:
        others = client.other_arm_clients()
        if others:
            # bench 2026-09-25: ADL ran a cabinet task on this arm minutes
            # after a mission, and its move home ended guard-tripped, the
            # arm held wherever that was
            return (
                "another program that moves this arm is running (%s — the ADL runtime) — two controllers on "
                "one arm. `sheppy up box-opening` stops it and brings up what the mission needs. Nothing moved."
                % ", ".join(others)
            )
        from box_opening.perception.d405 import CAMERA_CONFIG, camera_config

        # the world's wrist camera, as the grabber uses: flatbot's OAK hangs off
        # arm_-prefixed frames, the D405 default's end_effector_link does not exist there
        parent = camera_config(camera_cfg_name or CAMERA_CONFIG)["parent_frame"]
        if not client.tf_ready(client._base_frame, parent, timeout_s=2.0):
            # bench 2026-09-25: box_opening stopped, no /tf at all; the run
            # flew to staging and then could not place one wrist frame
            # ("no camera pose for the frame", 0/60)
            return (
                "no TF from %s to %s — sheppy's `box_opening` node (the arm's TF, the scene camera's and "
                "the OWL detectors) is not running: `sheppy up box-opening`. Nothing moved."
                % (client._base_frame, parent)
            )
    return None


def _home_legs(ctx, cfg, rest, st, say=log.info):
    """Home to `rest` from the PlanState `st`: the bench world, else, with a
    located box, rising over it first. Raises the last refusal."""
    try:
        return [build_home_leg(ctx, st.joints, rest, chain=st.chain)]
    except RuntimeError as e:
        if cfg is None or getattr(ctx, "cpose", None) is None:
            raise
        say("[press_demo] home refused from here (%s) — rising over the box first" % e)
    _world, rise, after = _rise_over_button(ctx, cfg, st, "retreat:rise")
    home, _ = _plan_motion(
        ctx, after, "home", ("joints", list(HOME if rest is None else rest)), _full_world(ctx), TRANSIT_SPEED
    )
    return [rise, home]


def try_home(ctx, runner, execute, why):
    """Recovery home that cannot crash the exit path: a refused home plan
    leaves the arm holding with an honest line (2026-08-25 review). Goes
    to the mission's rest pose (ctx.press_cfg, when main set it).

    The home is planned in the blind bench world, whose whole placement
    band is blocked to container height — so a run that stops OVER the box
    (at the hop, at a touch) starts inside the band and is refused by
    construction: "home plan refused (... the start state collides with the
    world ...) — arm holds" (bench 2026-09-21). When the mission has located
    this box, it first rises out of the corridor it came down, back to
    staging height above the button (_rise_over_button: planned against the
    button's interaction world), and goes home from there.

    A joint too near its limit for the planner to start from gets home_arm's
    plan-free step along the joint line toward the rest pose first, the
    shortest one home plans from (primitives.ClearMarginFree): a staging at
    the reach limit left arm_joint_6 0.100 rad from it and both plans were
    refused (flatbot 2026-10-23).

    True when the home was planned and every leg of it flew; False — the
    arm holds — otherwise."""
    log.info("[press_demo] %s — returning home" % why)
    cfg = getattr(ctx, "press_cfg", None)
    rest = None if cfg is None else rest_joints(cfg)
    live = _state(ctx.client.joints())
    clear = ClearMarginFree(HOME if rest is None else rest)
    try:
        near = clear.near(ctx, live.joints)
        if not near:
            legs = _home_legs(ctx, cfg, rest, live)
        else:
            log.info(
                "[press_demo] %s: too near for the planner to start from — trying plan-free steps along the "
                "joint line toward the rest pose, shortest first"
                % ", ".join("%s is %.3f rad from its limit" % jr for jr in near)
            )
            legs, last = None, None
            for step, at, info in clear.steps(ctx, live):
                notes = []
                try:
                    legs = step + _home_legs(ctx, cfg, rest, at, say=notes.append)
                except RuntimeError as e:
                    last = e
                    continue
                for note in notes:
                    log.info(note)
                if step:
                    d = info["tool_move"]
                    log.info(
                        "[press_demo] a plan-free step %.0f%% of the way to the rest pose first (the largest joint "
                        "%.3f rad, the tool %+.0f mm up and %.0f mm across; obstruction-guarded)"
                        % (
                            100 * info["fraction"],
                            info["largest_rad"],
                            1000 * d[2],
                            1000 * (d[0] ** 2 + d[1] ** 2) ** 0.5,
                        )
                    )
                break
            if legs is None:
                raise RuntimeError(
                    "no plan-free step toward the rest pose of up to %.2f rad per joint lets the planner plan "
                    "home (last: %s) — jog the arm clear by hand" % (CLEAR_MARGIN_MAX_STEP_RAD, last)
                )
    except RuntimeError as e:
        log.info("[press_demo] home plan refused (%s) — arm holds" % e)
        return False
    res = runner.run(legs, execute=execute)
    return all(r.ok for r in res)


DETECT_REPORT_S = 15.0


def table_seen_over(extent):
    """' seen over x a..b, y c..d' for a measured table, '' for none."""
    if extent is None:
        return ""
    return " seen over x %.2f..%.2f, y %.2f..%.2f" % tuple(extent)


def scene_detect_report(locator, duration_s=DETECT_REPORT_S):
    """--detect-only: the scene camera's fixes for `duration_s` with nothing
    moving — where the mission would fly, and how steady that is. Returns
    how many fixes it made."""
    log.info("[press_demo] DETECT-ONLY: the scene camera's fixes for %.0f s — nothing moves" % duration_s)
    t0 = time.monotonic()
    tops = []
    said = None
    while time.monotonic() - t0 < duration_s:
        fix = locator.locate(min(SCENE_LOCATE_S, max(0.5, duration_s - (time.monotonic() - t0))))
        viz_hook("scene", locator, fix)  # what it judged, drawn off this thread (perception/viz)
        if fix is None:
            if locator.last_why != said:  # a refusal repeats every OWL tick: say it once
                log.info("[press_demo]   no fix: %s" % locator.last_why)
                said = locator.last_why
            time.sleep(0.5)  # an instant refusal must not spin the OWL's enable gate
            continue
        tops.append(fix.top_xyz)
        log.info(
            "[press_demo]   SCENE: box at [%.3f, %.3f] yaw %.1f deg, lid top z %.3f, table %.3f%s, %s, %d lid points%s"
            % (
                fix.pose.xyz[0],
                fix.pose.xyz[1],
                math.degrees(fix.pose.yaw),
                fix.top_xyz[2],
                fix.table_z_scene,
                table_seen_over(fix.table_extent),
                scene_source_label(fix),
                fix.n_points,
                "; %s, finished from its lid's arc in view" % fix.partial if getattr(fix, "partial", "") else "",
            )
        )
    if not tops:
        log.info("[press_demo] detect-only done: NO BOX — %s" % locator.last_why)
        return 0
    spread = [1000 * (max(t[i] for t in tops) - min(t[i] for t in tops)) for i in range(3)]
    log.info("[press_demo] detect-only done: %d fix(es), spread [%.0f, %.0f, %.0f] mm" % (len(tops), *spread))
    return len(tops)


def detect_only_report(node, watcher, ctx, cfg, runner, execute):
    """--wrist-detect, the wrist-mount calibration observation: hold at the
    look pose reporting every fresh fix (top-face centre, yaw, footprint),
    then park home.

    Place the container at a TAPE-MEASURED spot first; the printed base
    pose vs truth solves the wrist-mount error."""
    watcher.reset()
    watcher.active = True
    log.info("[press_demo] DETECT-ONLY: reporting fixes for 15 s")
    t0 = time.monotonic()
    last = None
    try:
        while time.monotonic() - t0 < 15.0:
            _spin_detect(node)
            got = watcher.fix()
            if got is None:
                continue
            pos, yaw = got
            key = tuple(round(float(v), 4) for v in pos)
            if key == last:
                continue
            last = key
            f = watcher.last_debug
            log.info(
                "fix: top [%.3f, %.3f, %.3f] yaw %.1f | footprint %.3fx%.3f m | %d px"
                % (
                    pos[0],
                    pos[1],
                    pos[2],
                    math.degrees(float(yaw)),
                    f.footprint[0],
                    f.footprint[1],
                    f.n_px,
                )
            )
    finally:
        watcher.active = False
    log.info("[press_demo] detect-only done (%s) — homing" % watcher.status())
    runner.run([build_home_leg(ctx, ctx.client.joints())], execute=execute)


# This close (every joint, wrap-aware) to the rest pose is AT it; anywhere
# else, the run's first motion goes there (home_first). A compliant run
# starts from anywhere within REST_TOL_RAD instead: after a
# compliant run's impedance home the joint spring holds the arm off HOME
# (flatbot 2026-10-27 16:24: 0.03 rad, and the run flew a POSITION home to
# take it out).
AT_REST_RAD = 0.01


def home_first(ctx, runner, execute, rest, tol=AT_REST_RAD):
    """The mission's first motion: the arm to its exact rest pose (`rest`,
    HOME) unless it already stands within `tol` of it, so every run starts
    from the same place (owner, 2026-09-25). Planned like home_arm's move:
    in the bench world, whose keep-out band keeps it above wherever a box
    may stand. Returns False — nothing moved — when that is refused, with
    the recovery named: an arm that far down (holding at the table after an
    abort) is home_arm's case, whose bare-world and lift-first fallbacks
    want the operator's eye on the bench, not an automatic start.

    main passes tol=REST_TOL_RAD for a compliant run, which relaxes "the
    same place" from 0.01 to 0.05 rad for that run alone. An impedance hold
    cannot stand at exact HOME: the joint spring holds the
    arm where its pull meets gravity, 0.03 rad off after the 16:23 run's
    impedance home (flatbot 2026-10-27), and the run's first compliant leg
    sags by as much anyway. Flying HOME from there cost the 16:24 run a
    motion and about 0.95 s: a 0.14 s plan, 0.55 s for a 0.30 s profile
    (0.25 s of it the switch out of impedance), and its first look would
    have paid the switch back (16:23's paid 0.26 s). Such a start is one
    line and a start_offset note, each joint's offset in it (the sag the
    impedance gains are judged by), and nothing after it plans from HOME:
    the reach-out plans from the live joints. A start
    beyond `tol` homes as before.

    It used to REFUSE any start more than 1.2 rad from a rest pose and fly
    from wherever it stood inside that; planning from a failure pose had
    swung the arm half upside down once (2026-09-01) — every joint goal
    now goes the short way round (runtime/branches), and the bench world
    and the trajectory sanity gate still stand between it and that."""
    live = ctx.client.joints()
    off = rest_distance(live, rest)
    if off <= AT_REST_RAD:
        return True
    if off <= tol:
        # three decimals: two would print STIFF's hold (about 0.015) as
        # "0.01", AT_REST_RAD's own value, on a start that is not at rest
        log.info(
            "[press_demo] the arm rests %.3f rad from its rest pose — the joint spring's hold after an "
            "impedance run; starting from there" % off
        )
        per_joint = [
            round(ang_diff(a, b), 4) for a, b in zip(live, rest, strict=False)
        ]  # live − rest, as the spring holds it
        runner.note("start_offset", rad=round(off, 4), per_joint=per_joint)
        return True
    log.info("[press_demo] HOME FIRST: the arm is %.2f rad from its rest pose — going there before anything else" % off)
    world = ctx.worlds.push_name("bench", model=ctx.model)
    try:
        leg, _ = _plan_motion(ctx, _state(live), "home", ("joints", list(rest)), world, TRANSIT_SPEED)
    except RuntimeError as e:
        log.info(
            "[press_demo] not starting: the move home was refused in the bench world (%s) — the arm is "
            "probably down at the table. Clear the bench and recover by hand:\n"
            "    scripts/sheppy_task.sh home_arm --execute    (add --lift-first 0.08 if that is refused too)" % e
        )
        return False
    res = runner.run([leg], execute=execute)
    return all(r.ok for r in res)


def adopt_scene_table(ctx, worlds, watcher, scene_fix):
    """No survey: this run's table, measured where the box stands, is the
    one every world from here on is planned around — and the one the
    mission's record keeps (it was built before the scene measured it, and
    kept no table on an unsurveyed rig: flatbot 2026-10-22)."""
    if worlds.table_top_z is not None or math.isnan(scene_fix.table_z_scene):
        return
    worlds.table_top_z = scene_fix.table_z_scene
    worlds.table_extent = scene_fix.table_extent
    watcher.table_z = scene_fix.table_z_scene
    if ctx.mission_frames is not None:
        ctx.mission_frames.note(table_z=float(scene_fix.table_z_scene))


def keep_mission_frames(frames, exc):
    """Write this run's detection record (ctx.mission_frames), with how the
    run ended (`exc`: the SystemExit in flight, or None), and say where it
    went. Never changes how the run ends: a failure here, or a Ctrl+C
    during the write, is a line in the log (a placement left without its
    truth.json is not part of the set: detection_set.write_manifest)."""
    if frames is None:
        return
    code = exc.code if isinstance(exc, SystemExit) else (0 if exc is None else type(exc).__name__)
    try:
        t0 = time.monotonic()
        wait_capture_writes()  # close() copies the aim's folder: it must be whole
        out = frames.close(exit_code=code)
    except (Exception, KeyboardInterrupt) as e:  # a full disk, a second Ctrl+C
        log.info("[press_demo] (this run's frames were not kept: %s)" % (str(e) or type(e).__name__))
        return
    if out is not None:
        log.info(
            "[press_demo] frames kept for detection work: %s (%s; %s; %.1f s)"
            % (out, ", ".join(frames.doc["frames"]), frames.doc["truth_note"], time.monotonic() - t0)
        )


def main():
    t_main = time.monotonic()  # "it takes too long to detect the box": timed
    # If the process ever dies on a fatal signal, say which thread was doing
    # what: a run ended "terminate called without an active exception" after
    # its honest STOP line (bench 2026-09-21 14:16) and left nothing to go on.
    import faulthandler

    faulthandler.enable()
    ap = cli_common.make_parser(__doc__)
    ap.add_argument(
        "--press-only",
        action="store_true",
        help="stop after the press: the reach-out presses, rises and goes home; the lid stays on",
    )
    # always on since 2026-10-28 (the reach-out's depth press IS the run):
    # accepted, and without effect, so older command lines and sheppy's
    # local tasks still parse
    ap.add_argument("--depth-press", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument(
        "--guarded",
        action="store_true",
        help="the reach-out's depth press with its torque guards and checks, position-controlled, as before "
        "2026-10-27; it ends at the press (no lid off). Without it the run is COMPLIANT "
        "(constants.COMPLIANT_DEFAULT): joint impedance, nothing it reads stops the arm",
    )
    detect = ap.add_mutually_exclusive_group()
    detect.add_argument(
        "--detect-only",
        action="store_true",
        help="report the scene camera's fixes for 15 s and exit; nothing moves",
    )
    detect.add_argument(
        "--wrist-detect",
        action="store_true",
        help="the wrist-mount calibration observation (was --detect-only --no-scene): the arm to the look pose, "
        "15 s of the wrist camera's fixes printed, home; the scene camera is not used",
    )
    ap.add_argument(
        "--scene-calib",
        default=None,
        help="scene camera calibration yaml (default: the installed config/camera_scene.yaml)",
    )
    args = ap.parse_args()
    # the reach-out reads it (reach_press.run_reach_press): its depth press
    # is the run (owner 2026-10-28), whatever the command line said
    args.depth_press = True
    args.compliant = COMPLIANT_DEFAULT and not args.guarded
    cfg_path = args.container or cli_common.default_container_yaml()
    bench = args.bench_world or cli_common.default_bench_yaml()
    model = ContainerModel.load(cfg_path)
    cfg = load_press_demo(cfg_path)
    # the grip band in this gripper's current readings (runtime/fingers: an
    # empty close read 0.793 when it was measured, 0.636 since 2026-09-25)
    cfg = replace(cfg, grip_band=tuple(round(v, 3) for v in fingers.scaled_band(cfg.grip_band)))
    cli_common.refuse_unmeasured(model, args.execute, measuring=args.detect_only or args.wrist_detect)
    node, client = cli_common.init_runtime(args.execute)
    worlds = WorldStore(bench)
    client._base_frame = worlds.base_frame
    from box_opening.runtime.approach import chain_ready, set_chain_root

    set_chain_root(worlds.base_frame, urdf_path=cli_common._arm_urdf_path(node, worlds.base_frame))
    ok, why = chain_ready()
    if not ok:
        # every aim, press and reach check needs the chain: without it the
        # run flew to staging and died there on a KeyError (review 2026-10-22)
        sys.exit(
            "[press_demo] not starting: /arm/robot_description not heard in 5 s (arm_state_publisher down?) — "
            "the arm kinematics for %s are unavailable (%s). Nothing moved." % (worlds.base_frame, why)
        )
    runner = Runner(client)
    cli_common.apply_speed_scale(runner, args)
    # PERCEPTION ON ITS OWN THREAD. The cameras and the scene camera's
    # detector rungs live on a second node spun by a background executor, so
    # the control loop's spin cadence (a 1 kHz joint-state stream, action
    # feedback, planning calls) never starves them: on the stub bench the
    # single executor gave the wrist's detector 1.6 frames a second
    # (2026-09-16). The persistent owl_detector nodes own the OWL models;
    # there is no in-process copy (a second OWLv2 beside cuRobo on one GPU,
    # field 2026-09-01).
    pnode = rclpy.create_node("box_opening_perception")
    from rclpy.executors import SingleThreadedExecutor

    pexec = SingleThreadedExecutor()
    pexec.add_node(pnode)
    import threading

    pthread = threading.Thread(target=pexec.spin, name="perception", daemon=True)
    ctx = None  # the finally below runs even if startup fails before Ctx
    pthread.start()
    try:
        # the owl_detector nodes were started by the launch with no container
        # of their own: tell them which one this run is for, so their prompts
        # and floor are this container's
        from box_opening.perception.owl_source import announce_container

        announce_container(pnode, cfg_path)
        # what the perception sees and decides, drawn for Foxglove on a thread
        # of its own (perception/viz), which loads it once this thread waits:
        # nothing here waits on it, and starting it is that thread's start.
        # Only with BOX_OPENING_VIZ=1: off (the default), the module is
        # not even loaded
        start_viz(pnode, model, worlds.base_frame)

        # the wrist camera's frames: the aim from the look pose
        # (aim_button_at_staging) and the wrist-mount observation
        watcher = BoxTopWatcher(
            pnode,
            cfg,
            model,
            worlds.table_top_z,
            base_frame=worlds.base_frame,
            camera_cfg_name=worlds.wrist_camera_config,
        )
        # The scene camera finds the box before the arm moves — the only thing
        # that does. It is placed by its calibration
        # (scripts/calibrate_scene_camera.py), or by TF where the robot's TF
        # carries its mount: flatbot's static_tf, whose camera is not the one
        # the installed camera_scene.yaml calibrates.
        from box_opening.perception.scene import load_scene_calibration, scene_calib_path

        scene_by_tf = worlds.external_static_tf
        locator = None
        if not args.wrist_detect:
            T_base_scene_link = None if scene_by_tf else load_scene_calibration(args.scene_calib)
            if scene_by_tf or T_base_scene_link is not None:
                from box_opening.perception.scene_calib import scene_tag_px
                from box_opening.perception.scene_source import SceneLocator

                locator = SceneLocator(
                    pnode,
                    cfg,
                    model,
                    worlds.table_top_z,
                    T_base_scene_link,
                    spins=False,
                    tag_px=None if scene_by_tf else scene_tag_px(args.scene_calib or scene_calib_path()),
                    scene_cfg=worlds.scene_camera_cfg,
                    base_frame=worlds.base_frame,
                )
                # the scene OWL NOW, where sheppy's scene segmenter cannot
                # answer (no classes for this container, or down): its first
                # bbox takes ~0.7 s (the node infers on the enable edge) and the
                # readiness checks below are about that long — by the time
                # locate() asks, the box is already on the topic. A live
                # segmenter keeps the OWL off, and off the GPU (prime)
                locator.prime()
        # camera on from here to exit
        ctx = Ctx(
            model=model,
            cpose=None,
            client=client,
            worlds=worlds,
            config_path=cfg_path,
        )

        ctx.press_cfg = cfg  # recovery homes go to the mission's rest pose
        if args.execute and not (args.detect_only or args.wrist_detect):
            # every run keeps what its cameras saw and, when the press lands,
            # where the button really was: a detection set that grows with the
            # bench's own runs (detection_set.MissionFrames; written at the end)
            ctx.mission_frames = MissionFrames(
                state_dir() / "detection_sets" / MISSIONS_SET, Path(cfg_path).name, worlds.table_top_z
            )
        why = readiness_refusal(client, args.execute, camera_cfg_name=worlds.wrist_camera_config)
        if why:
            sys.exit("[press_demo] not starting: %s" % why)
        if locator is None and not args.wrist_detect:
            sys.exit(
                "[press_demo] not starting: the scene camera finds the box, and it is not calibrated (no %s: "
                "scripts/calibrate_scene_camera.py). Nothing moved." % (args.scene_calib or scene_calib_path())
            )
        if args.detect_only:
            sys.exit(0 if scene_detect_report(locator) else 2)
        # THE FIRST MOTION: the arm to its exact rest pose, when it is not
        # already there (home_first) — every run starts from the same place.
        # A compliant run starts from the spring's hold it finds within
        # REST_TOL_RAD. client.compliant is still False here (only
        # run_reach_press sets it), so this is its condition read from args.
        # --guarded and the wrist-mount observation keep AT_REST_RAD: the
        # guarded reach-out reads an at-rest reference at the rest pose and
        # checks gravity against it.
        start_tol = REST_TOL_RAD if (args.compliant and not args.wrist_detect) else AT_REST_RAD
        if not home_first(ctx, runner, args.execute, rest_joints(cfg), tol=start_tol):
            sys.exit(3)
        if args.wrist_detect:
            # the calibration observation parks at the look pose and stays
            # there: a sighting must not cut its motion short
            live = client.joints()
            name, target, speed = search_targets(cfg)[0]
            if rest_distance(live, target) > REST_TOL_RAD:
                res = runner.run(
                    [build_search_leg(ctx, (name, target, speed), live, _NeverStops)],
                    execute=args.execute,
                )
                if any(not r.ok for r in res):
                    sys.exit(1)
            detect_only_report(node, watcher, ctx, cfg, runner, args.execute)
            sys.exit(0)
        # The fingers shut NOW and ride the look (they sit in the bottom rows
        # of the wrist camera's frame, the box in its middle — no occlusion,
        # capture 20260901-130610); the reach-out joins the close before the
        # arm moves toward the box (reach_press._reach_press). No closed
        # fingers = no press: an open aperture strikes the lid and can still
        # read "pressed", so a refused close ends the run here.
        _ok, finger_pos, _s = client.gripper_cmd(None)  # a query: nothing moves
        if fingers.already_closed(finger_pos):
            # already shut (the last run left them so): a close would not
            # move them, and "never moved" is a failed close (runtime/fingers)
            log.info("[press_demo] the fingers are already closed (%.3f)" % finger_pos)
        elif not runner.start_gripper("press:close", GRIPPER_CMD_CLOSED, args.execute):
            try_home(
                ctx,
                runner,
                args.execute,
                "press:close refused or failed — the fingers are not closed",
            )
            sys.exit(1)

        # THE SCENE CAMERA ALONE FINDS THE BOX. It sees the whole workspace
        # from where it stands, so the box is found before the arm has moved
        # toward it. The wrist never searches for it (owner, 2026-10-22): no
        # scene fix, and the run stops.
        t_scene = time.monotonic()
        scene_fix = locator.locate(SCENE_FIRST_LOCATE_S)
        viz_hook("scene", locator, scene_fix)  # what it judged, drawn off this thread (perception/viz)
        if ctx.mission_frames is not None:
            ctx.mission_frames.scene(
                scene_frame(locator.grab, locator.base_to_link()),
                fix_xyz=None if scene_fix is None else scene_fix.top_xyz,
                score=None if scene_fix is None else scene_fix.score,
                why=locator.last_why if scene_fix is None else None,
                source=None if scene_fix is None else getattr(scene_fix, "source", "owl"),
                boxes=getattr(locator, "last_boxes", None),
            )
        if locator.moved_note and scene_fix is not None:
            log.info("[press_demo] scene camera: %s" % locator.moved_note)  # e.g. the tag was not in view
        if scene_fix is None:
            runner.note("reach_refused", why="NO BOX — scene camera: %s" % locator.last_why)
            try_home(ctx, runner, args.execute, "NO BOX — scene camera: %s" % locator.last_why)
            sys.exit(2)
        if not scene_by_tf:
            trusted, distrust = scene_trust(
                residuals_path_for(args.scene_calib), args.scene_calib or scene_calib_path()
            )
            if not trusted:
                log.info(
                    "[press_demo] scene camera: %s — reaching out to its box anyway; the wrist's aim from the "
                    "look pose decides" % distrust
                )
        sp = scene_fix.pose
        adopt_scene_table(ctx, worlds, watcher, scene_fix)
        surveyed = watcher.table_z
        log.info(
            "[press_demo] SCENE: box at [%.3f, %.3f] yaw %.1f deg from %d lid points%s "
            "(%s; the scene reads the table at %.3f%s, surveyed %s; %.1f s to locate "
            "[%s], %.1f s since start) — reaching out"
            % (
                sp.xyz[0],
                sp.xyz[1],
                math.degrees(sp.yaw),
                scene_fix.n_points,
                " (%s, finished from its lid's arc in view)" % scene_fix.partial
                if getattr(scene_fix, "partial", "")
                else "",
                scene_source_label(scene_fix),
                scene_fix.table_z_scene,
                table_seen_over(scene_fix.table_extent),
                ("%.3f" % surveyed) if surveyed is not None else "n/a",
                time.monotonic() - t_scene,
                " ".join("%s %.2f" % (k[:-2], v) for k, v in locator.last_timing.items()),
                time.monotonic() - t_main,
            )
        )
        runner.note(
            "scene_fix",
            origin_xyz=[round(float(v), 4) for v in sp.xyz],
            yaw_deg=round(math.degrees(sp.yaw), 2),
            top_xyz=[round(float(v), 4) for v in scene_fix.top_xyz],
            n_points=scene_fix.n_points,
            owl_score=round(scene_fix.score, 3),  # the detector's score, whichever detector
            source=getattr(scene_fix, "source", "owl"),
            n_sightings=int(getattr(scene_fix, "n_sightings", 1)),
            table_z_scene=None
            if scene_fix.table_z_scene != scene_fix.table_z_scene
            else round(scene_fix.table_z_scene, 4),
        )
        # EVERY box the scene camera fixes is pressed by the reach-out: the
        # tilted look, the wrist's aim, the hand turned down onto the button,
        # and — the full run — the lid off and set down (tasks/reach_press,
        # tasks/reach_lid_off). Owner 2026-10-28: "make the arm always go to
        # that 45 degree looking pose even if it's close enough to go
        # straight up and down. completely scrap the home -> straight
        # vertical if close enough mode".
        from box_opening.tasks.reach_press import run_reach_press

        ctx.cpose = scene_fix.pose  # the box every world and plan from here is built around
        watcher.roi = None  # the scene bbox never applied to the wrist frame
        sys.exit(run_reach_press(node, ctx, cfg, runner, watcher, args, scene_fix, locator))
    except KeyboardInterrupt:
        sys.exit(130)  # the abort path already reported what was confirmed
    finally:
        ending = sys.exc_info()[1]  # how the run ended, for its record
        try:
            # the console as it was, before the node the viz's publishers
            # live on goes, and the viz's thread stopped (never left inside
            # OpenCV for the interpreter's teardown, a second Ctrl+C or
            # not); a run unwinding from a Ctrl+C (the 130 above) publishes
            # nothing more
            ctrl_c = isinstance(ending, KeyboardInterrupt) or isinstance(
                getattr(ending, "__context__", None), KeyboardInterrupt
            )
            stop_viz(not ctrl_c)
        finally:
            # Stop the perception thread BEFORE the interpreter tears rclpy
            # down: an executor still inside spin() on another thread at exit
            # ends in "terminate called without an active exception" and an
            # abort that eats the exit code (field 2026-09-16). Whatever the
            # viz's stop did — a second Ctrl+C in it included.
            pexec.shutdown()
            pthread.join(timeout=3.0)
            try:
                pexec.remove_node(pnode)
                pnode.destroy_node()
            except Exception:
                pass
            # the run is over and the arm stopped: now the last arrival's settle
            # is noted, from what was heard (runner.flush), and its frames written
            runner.flush()
            keep_mission_frames(getattr(ctx, "mission_frames", None), ending)


if __name__ == "__main__":
    main()
