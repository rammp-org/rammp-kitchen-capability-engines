"""press_demo — the camera-driven autonomous box opening (owner design
2026-08-24; one run, the reach-out, since 2026-10-28).

    sheppy: the `arm` and `planner` nodes (rammp-deployments, december_2026)
    ros2 launch box_opening press_demo.launch.py   # TF
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    ros2 run box_opening press_demo --execute

Every box is pressed by the reach-out, however near it stands (owner
2026-10-28: "make the arm always go to that 45 degree looking pose even if
it's close enough to go straight up and down"; the straight-down staging
mode is gone). States logged one line each:

  SCENE   before the arm moves toward the box, the fixed scene camera finds
          it — sheppy's scene segmenter boxes it (only the container the
          user selected, --instance-id, where one is given); the depth
          inside the box at lid height places it
          through config/camera_scene.yaml, or through TF where the robot's
          TF carries the camera's mount (world external_static_tf:
          flatbot). What it sees standing on the table goes into the
          planner's worlds. The wrist never searches (owner, 2026-10-22):
          no scene fix -> home, exit 2 (a run the engine guides, --guide,
          has the chair moved instead where a camera looked and saw no
          container: owner 2026-11-19). The fingers close as the run starts.
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
from dataclasses import dataclass, replace
from pathlib import Path

import rclpy

from box_opening import LOGGER as log
from box_opening import guide
from box_opening.constants import (
    COMPLIANT_DEFAULT,
    GRIPPER_CMD_CLOSED,
    HOME,
    HOME_SPEED,
    REST_TOL_RAD,
    SEARCH_SPEED,
    TRANSIT_SPEED,
    mission_frames_enabled,
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
    StepOutStopped,
    _full_world,
    _interaction_world,
    _plan_motion,
    as_home,
    flown_then_measured,
    home_any,
)
from box_opening.primitives.look import look_joints, sweep_targets
from box_opening.runtime import driver, fingers
from box_opening.runtime.abort import EXIT_HALTED, EXIT_LINE_UP_HOLDS, EXIT_NOT_LINED_UP, ArmHalted
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


# How long the scene camera gets to find the box (the looks after a press):
# sheppy's scene segmenter fixes it in about 0.35 s (two agreeing frames at
# 5 Hz; scene_source.commit_fix), so this leaves room for a slow segmenter.
SCENE_LOCATE_S = 5.0
# ... and the run's FIRST look, before anything moves, keeps at it longer:
# flatbot 2026-10-27 15:34:01 refused every candidate it was offered in 5 s
# (26-28 depth points in each) and the run went home; 25 s later the next
# run had the box 0.83 s in. Looking longer costs only time.
SCENE_FIRST_LOCATE_S = 15.0
# ... but short after the go of a line-up whose wrist held the container as the go came (--line-up; its
# watch's last fix a moment old at most: line_up_watch's wrist_held), and the scene camera had the whole
# line-up to see it.
# A wrist that was turned and never showed the box (the scene camera's sightings brought the Ready) leaves
# the scene camera all the run has, and that keeps the long look.
# This look only refreshes the selected container's box (selection_tie reads the segmenter's newest frame)
# and the hint for where the wrist looks first. At 15 s the arm stood with its wrist turned for 15 s after
# Open, which reads as a hang, and the time came out of a run budget the line-up had already used: both
# clean runs of 2026-11-12 spent the whole first look and the scene camera found nothing in front of the
# arm. 2 s is about ten of the segmenter's messages at its 5 Hz, several times the 0.35 s a fix takes it.
SCENE_LINED_UP_LOCATE_S = 2.0
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
    source = getattr(fix, "source", "segmenter")
    n = int(getattr(fix, "n_sightings", 1))
    sources = tuple(getattr(fix, "sources", ()) or (source,))
    return "%s %.2f, %d sighting%s (%s)" % (
        "SEG" if source == "segmenter" else source.upper(),
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


# The wrist reading its table this far from the scene camera's (the raw
# offset, taken out or not) is said as a warning: a camera has likely moved.
# 10-30 10:40-11:42, raw offsets of -57 mm and worse: 5 of 5 grips closed on
# air; at -43 mm 1 of 1 held, at -2 to -23 mm 16 of 18 (sweep aim-tablehealth).
# A warning only: whether to stop past ~40 mm is the owner's to decide
TABLE_OFFSET_WARN_M = 0.030


def table_offset_warning(off_raw):
    """The warning line for a raw table offset (m) past TABLE_OFFSET_WARN_M,
    else None."""
    if off_raw is None or not math.isfinite(off_raw) or abs(off_raw) <= TABLE_OFFSET_WARN_M:
        return None
    return (
        "[press_demo] WARNING: the wrist reads its table %+.0f mm from the scene camera's (more than %.0f mm): "
        "a camera may have moved — on 10-30 every grip at -57 mm or worse closed on air"
        % (1000 * off_raw, 1000 * TABLE_OFFSET_WARN_M)
    )


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
            warn = table_offset_warning(off_raw)
            if warn is not None:
                log.info(warn)
            # the frame the fix's newest hit was judged on, not the grabber's newest
            judged = aimer.judged_frame()
            folder = save_aim_frame(watcher.grab, s, tag, frame=judged)
            # ... and kept, with the table the wrist read in it: the look pose's view of the
            # table round the box, for the lid's room map (reach_lid_off.aim_cloud)
            watcher.last_aim_frame = (tag, judged, off_raw)
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
                    table_offset_warn=warn is not None,  # past TABLE_OFFSET_WARN_M: a camera may have moved
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
                    # why frames were turned down before the fix (button_aim.reject_kind):
                    # 3 of 9 tilted aims spent 0.5-1.2 s before their first hit, the
                    # cause (moving, no pose, a gate) recorded nowhere
                    rejects=dict(aimer.rejects.most_common(AIM_REJECTS_KEPT)),
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
    if runner is not None:
        # no fix: what each frame was turned down for, in the record, not only printed
        runner.note(
            "aim_missed",
            aim=tag,
            hits=aimer.hits,
            frames=aimer.frames,
            aim_s=round(time.monotonic() - t0, 2),
            hit_s=[round(float(v), 2) for v in aimer.hit_s],
            rejects=dict(aimer.rejects.most_common(AIM_REJECTS_KEPT)),
            last_reject=aimer.last_why,
        )
    return None, aimer.status()


AIM_REJECTS_KEPT = 8  # the most frequent of an aim's reject reasons its row keeps

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


POSE_WAIT_S = 0.05  # a frame's TF pose lands up to ~one /tf period (16 Hz) after the frame


def _pose_at(grab, stamp, wait_s=POSE_WAIT_S):
    """camera_pose_at(grab, stamp=stamp), asked again for up to `wait_s` while TF catches up
    with the frame: a wrist frame arrives ~36 ms after its stamp and /tf comes at 16 Hz, so a
    capture taken at a random moment found no pose 7.6 % of the time (none with 30 ms more)."""
    from box_opening.perception.depth_source import camera_pose_at

    end = time.monotonic() + wait_s
    while True:
        pose = camera_pose_at(grab, stamp=stamp)
        if pose is not None or time.monotonic() >= end:
            return pose
        time.sleep(0.005)


def _save_wrist_frame(grab, folder, caption, draw=None, background=False, after=None, frame=None):
    """One wrist frame (colour, depth, K, camera pose, stamp) into `folder`
    in the recorder's format (replay: scripts/record_scan_frames.py
    --analyze), plus a JPEG with `caption` and whatever `draw(img)` adds,
    then `after()`. Returns the folder, or None when there is no frame or
    no pose for it. `frame`: (snapshot, pose, extra arrays) to save instead
    of the grabber's newest (ButtonAimer.judged_frame: the frame an aim
    judged, a later one possibly in the grabber by now). With `background` the frame is taken now and written on
    a thread (compressing it is ~45 ms the next leg need not wait for): the
    folder exists at once, its files a moment later (wait_capture_writes).
    A pose TF does not hold yet is asked for again for up to POSE_WAIT_S (on
    the write thread with `background`, so the mission never waits)."""
    import numpy as np

    from box_opening.perception.d405 import frame_snapshot

    if grab is None and frame is None:
        return None
    extra = {}
    if frame is not None:
        f, pose, extra = frame  # its pose already found: nothing to wait for
    else:
        # one snapshot: the grabber assigns new arrays per message, so these
        # stay this frame while a thread writes them, and the JPEG is the npz's
        f = frame_snapshot(grab)
        if f.color is None or f.depth is None or f.k is None or f.color_stamp is None:
            return None
        pose = _pose_at(grab, f.color_stamp, wait_s=0.0 if background else POSE_WAIT_S)
        if pose is None and not background:
            return None
    img = None
    try:
        import cv2

        img = np.ascontiguousarray(f.color).copy()
        if draw is not None:
            draw(img)
        cv2.putText(img, str(caption)[:90], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    except Exception:
        img = None

    def job():
        got = pose if pose is not None else _pose_at(grab, f.color_stamp)
        if got is None:
            log.info("[press_demo] (wrist frame not written to %s: no camera pose for it in TF)" % folder)
            return
        arrays = {**wrist_frame(f, got), **extra}
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


def capture_stamp(t=None):
    """A capture folder's time, to the millisecond and fixed-width, so the
    names still sort in time order (_prune_captures, the replay scripts):
    YYYYmmdd-HHMMSS-mmm. Second-wide names put two captures of one second
    in one folder, the second's frame over the first's (10-28 16:16:08 and
    16:24:09, 10-29 12:03:00: two reach-after aims 0.18-0.43 s apart)."""
    t = time.time() if t is None else float(t)
    return "%s-%03d" % (time.strftime("%Y%m%d-%H%M%S", time.localtime(t)), min(999, int((t % 1.0) * 1000)))


def dump_wrist_frame(watcher, tag):
    """Save the wrist camera's CURRENT frame and the detector's last reject
    reason, so "the wrist did not see the box" can be looked at afterwards
    instead of argued about (owner, 2026-09-16). Lands in
    ~/.ros/box_opening/captures/wrist-<tag>-<stamp>/ (capture_stamp). Returns
    the folder or None."""
    folder = CAPTURES_DIR / ("wrist-%s-%s" % (tag, capture_stamp()))
    out = _save_wrist_frame(getattr(watcher, "grab", None), folder, watcher.last_reject or "no reject recorded")
    if out is None:
        return None
    (out / "reason.txt").write_text("%s\n%s\n" % (watcher.status(), watcher.last_reject))
    log.info("[press_demo] wrist frame saved for inspection: %s" % out)
    return out


def save_aim_frame(grab, sighting, tag="aim", frame=None):
    """The frame the press was aimed on, with the seam circle (green) and
    the PRESS POINT — its centre (red +) — drawn, every run, so a miss is a
    ten-second look instead of a blind read of the run log (the 2026-09-17
    miss had no frame). Keeps the newest AIM_CAPTURES_KEPT folders.
    `frame`: the frame the aim judged (ButtonAimer.judged_frame), with the
    fused depth it was judged on; None: the grabber's newest."""
    folder = CAPTURES_DIR / ("wrist-%s-%s" % (tag, capture_stamp()))

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
    out = _save_wrist_frame(
        grab, folder, caption, draw, background=True, after=lambda: _prune_captures(tag), frame=frame
    )
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
        HOME_SPEED,
    )
    return as_home(leg)


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
# The reach-out's TILTED look is less sure than the camera straight down.
# The wrist study of 2026-11-05 (I-Y2, M17, H5; 109 saved wrist captures
# and the run records) read popped knobs 5.0-9.7 mm on tilted frames, 5 of
# 18 under 6.0 (3 of the robot's 19 tilted fixes), and flush buttons up to
# 2.6 mm (4.4 in the records). One look at 6.0 read an open box shut, and a
# press on a popped knob latches it again (push-push; the physics study).
# So the camera's call before a press is a band (knob_call): below
# KNOB_SHUT_BELOW_MM the knob is flush and the box is pressed; at
# KNOB_UP_MIN_MM or more it is open and its lid comes off unpressed. In
# between the wrist looks once more, and still in between the box is taken
# for open: never pressed. On the study's readings that takes an open box
# pressed shut from 16 % of tilted looks to about none, for one more look
# on 3-10 % of shut boxes (5 of the 52 pre-press reads on record were 3.0
# mm or more). After a press the same holds: only a knob read below
# KNOB_SHUT_BELOW_MM is pressed again (reach_lid_off).
KNOB_SHUT_BELOW_MM = 3.0
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
        "run before?). Pressing it would shut it, so it is not pressed: --press-only and --guarded stop here. Run "
        "without them: the full run takes an open box's lid off" % up_mm
    )


def refuse_band_box(reads_mm):
    """Why this box must not be pressed: every look that read its knob
    (`reads_mm`, in order; None where one read no height) put it between
    KNOB_SHUT_BELOW_MM and KNOB_UP_MIN_MM, where a tilted look reads popped
    knobs too. The full run takes such a box's lid off unpressed
    (reach_press); this is what --press-only and --guarded stop on."""
    return (
        "the knob reads %s mm above its lid — neither flush (below %.0f) nor up (%.0f or more), and a popped "
        "knob reads there too. Pressing an open box would shut it, so it is not pressed: --press-only and "
        "--guarded stop here. Run without them: the full run takes the lid off a box it cannot call shut"
        % (knob_reads_text(reads_mm), KNOB_SHUT_BELOW_MM, KNOB_UP_MIN_MM)
    )


def knob_reads_text(reads_mm):
    """Knob readings in mm, in order, as the run says them: "4.2 then 1.1"; "?" for no reading."""
    return " then ".join("?" if r is None else "%.1f" % r for r in reads_mm)


def knob_is_up(up_mm):
    """The pop check's verdict on a camera reading (mm above the lid; None: no reading)."""
    return up_mm is not None and up_mm >= KNOB_UP_MIN_MM


def knob_call(up_mm):
    """The band's call on a camera reading (mm above the lid; None: no
    reading): "shut" below KNOB_SHUT_BELOW_MM, "open" at KNOB_UP_MIN_MM or
    more, "band" in between, "unread" with no reading. Pure."""
    if up_mm is None or not math.isfinite(up_mm):
        return "unread"
    if knob_is_up(up_mm):
        return "open"
    return "shut" if up_mm < KNOB_SHUT_BELOW_MM else "band"


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


def keep_out_counter(ctx, tmap, table_z, source, runner=None, key=None):
    """Hold the arm out of the cabinet below the counter's edges `tmap` shows: clear_table
    .counter_edges and counter_keep_out (the edge and the solid behind it, from the counter's top at
    `table_z` to the floor) become the body check's run-time boxes (approach.set_obstacles), so every
    line box_opening flies itself, every planner goal it sends and the lid's set-down are held out of
    them as out of the chair. The environment from the cameras, this run (the owner's rule): each
    `key` (default `source`, which map it is) keeps the edges its latest map showed, every key's in
    force at once; a map that saw no edge keeps nothing out, and the log says so. No map at all
    (None), or one that cannot be read, is no news of the counter: what the key's last map showed
    stays in force. Noted 'counter_edges'. Returns the edges `tmap` showed, or None where there is no
    body check to hold them (off)."""
    from box_opening.runtime import approach

    check = approach.body_check()
    if check is None:
        log.info("[press_demo] COUNTER (%s): the body check is off — the counter's edges are not kept out" % source)
        return None
    key = source if key is None else key
    kept = dict(getattr(ctx, "counter_boxes", None) or {})
    before = list(kept.get(key) or [])
    if tmap is None:
        edges, boxes = [], before
    else:
        try:
            edges = clear_table.counter_edges(tmap)
            boxes = clear_table.counter_keep_out(edges, table_z, pad_m=check.model.padding_m)
        except Exception as e:  # noqa: BLE001 - a map that cannot be read shows no edge: said, never the run's end
            log.info(
                "[press_demo] COUNTER (%s): its map could not be read for the counter's edges (%s) — nothing more "
                "is kept out (%d boxes its last map kept out stay in force)" % (source, e, len(before))
            )
            return []
    kept[key] = boxes
    ctx.counter_boxes = kept
    every = [b for bs in kept.values() for b in bs]
    approach.set_obstacles(every)
    if edges:
        log.info(
            "[press_demo] COUNTER (%s): %d edge%s seen — %s; the arm is held out of the cabinet below %s (%d boxes "
            "in force, padded %.0f mm)"
            % (
                source,
                len(edges),
                "" if len(edges) == 1 else "s",
                "; ".join(
                    "%.2f m of it through [%.2f, %.2f], the counter toward [%.2f, %.2f]"
                    % (e.length_m, e.point[0], e.point[1], -e.out[0], -e.out[1])
                    for e in edges
                ),
                "it" if len(edges) == 1 else "them",
                len(every),
                1000 * check.model.padding_m,
            )
        )
    elif tmap is None:
        log.info(
            "[press_demo] COUNTER (%s): no map of the table — nothing more is kept out (%d boxes its last map kept "
            "out stay in force; %d in all)" % (source, len(before), len(every))
        )
    else:
        log.info(
            "[press_demo] COUNTER (%s): no edge seen — nothing past the counter is kept out there (an edge the "
            "cameras did not see is not known; %d boxes in force from the other maps)" % (source, len(every))
        )
    if runner is not None and hasattr(runner, "note"):
        runner.note(
            "counter_edges",
            source=source,
            mapped=tmap is not None,
            table_z=round(float(table_z), 4),
            edges=[e.summary() for e in edges],
            boxes=[] if tmap is None else [b["name"] for b in boxes],
            kept_from_before=len(before) if tmap is None else 0,
            in_force=len(every),
        )
    return edges


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


DRIVER_HEALTH_WAIT_S = 2.5  # the driver's diagnostics come about once a second


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
    if execute and hasattr(client, "driver_health"):
        # up, but its link to the arm faulted, or e-stopped: every goal would be refused
        # (flatbot 2026-11-04 12:23, WRONG_SERVOING_MODE at the driver's start)
        down = driver.health_refusal(client.driver_health(timeout_s=DRIVER_HEALTH_WAIT_S))
        if down:
            return "%s. Nothing moved." % down
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
                "no TF from %s to %s — sheppy's TF nodes (the arm's and the scene camera's) are not "
                "running: `sheppy up box-opening`. Nothing moved." % (client._base_frame, parent)
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
        ctx, after, "home", ("joints", list(HOME if rest is None else rest)), _full_world(ctx), HOME_SPEED
    )
    as_home(home)
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
            if locator.last_why != said:  # a refusal repeats every segmenter frame: say it once
                log.info("[press_demo]   no fix: %s" % locator.last_why)
                said = locator.last_why
            time.sleep(0.5)  # an instant refusal (the segmenter absent) must not spin
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
                ("; %s" if getattr(fix, "rough", False) else "; %s, finished from its lid's arc in view") % fix.partial
                if getattr(fix, "partial", "")
                else "",
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


# --line-up: the run's own process is the line-up's eyes, so one process alone ever commands the arm.
# Its lines for the engine begin with LINE_UP_MARK (tasks/box_watch's lines; every other line is the run's
# log), and the engine answers on stdin with one JSON line.
LINE_UP_MARK = "@@ "
LINE_UP_LOOK_S = 0.3  # the scene camera's own look, each time round (its sightings said as they are made)
# what the watch says of a find where NOTHING ties any find to the selected container (the selected id is in
# none of the scene camera's detections, and --wrist-alone is off): it is not said, and it is of the kind no
# moving of the chair cures (line_up_watch's judge)
UNTIED = (
    "the selected container %s is in none of the scene camera's detections: nothing says that what the wrist "
    "sees is the one selected, and the run would not take it"
)
# what the engine is told where the arm stands off rest by more than its wrist's own turn: nothing of a line-up
# homes it (the wrist's turn is the one motion made before the go). The engine's line-up ends on these words
# at once and its panel says them (guide.ARM_NOT_AT_REST: the one sentence, both sides')
LINE_UP_NOT_AT_REST = guide.ARM_NOT_AT_REST
# the wrist's watch held the container as the go came where its last fix is no older than this: the guide's own
# "seen" (guide.LOST_S). One frame without a find between the user's Open and the go's arrival is not the
# wrist having lost the box
LINE_UP_HELD_S = guide.LOST_S


class LineUpEar:
    """What the engine says on stdin while the user lines up, heard on a thread of its own: `go`, or `stop`
    (said, or the pipe closed: an engine that is gone has not said go).

    One ear serves the whole run: a run may be lined up more than once (before its first move with
    --line-up, and again wherever no press plans from where the chair stands, --guide), and each round is
    answered by one line. again() opens the next round: the go that let the run on is forgotten, and an
    engine that has gone since (the pipe closed after its go) is a stop from the start."""

    def __init__(self, stream=None):
        import threading

        self.go = self.stop = False
        self._closed = False  # the pipe ended: nothing more will ever be heard
        self._stream = sys.stdin if stream is None else stream
        threading.Thread(target=self._listen, name="line-up-ear", daemon=True).start()

    def _listen(self):
        import json

        try:
            for line in self._stream:
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if isinstance(msg, dict) and msg.get("go") is True:
                    self.go = True
                elif isinstance(msg, dict) and msg.get("stop"):
                    self.stop = True
        except (OSError, ValueError):
            pass
        self._closed = True
        if not self.go:  # (after a go the run flies on, as it did: the pipe's end is the next round's stop)
            self.stop = True

    def again(self):
        """The next round: neither a go nor a stop heard yet, unless the engine is gone."""
        self.go, self.stop = False, self._closed

    def heard(self):
        return self.go or self.stop


def wrist_looks_from(live, rest, rest_tol):
    """Whether the line-up's wrist look starts from the joints `live`: every joint within `rest_tol` of the
    rest pose `rest`, but for the three the front look turns (survey.FRONT_JOINTS), which may stand as far
    from rest as that look ever turns them (survey.FRONT_MAX_RAD, and `rest_tol`). That is the arm as a
    line-up's own turn leaves it: after a Cancel while the wrist watched, or a run that refused after
    Ready, the wrist stands turned, and no line-up homes the arm before its go."""
    from box_opening.tasks import survey as sv

    return all(
        abs(ang_diff(a, b)) <= (sv.FRONT_MAX_RAD + rest_tol if j in sv.FRONT_JOINTS else rest_tol)
        for j, (a, b) in enumerate(zip(live, rest, strict=False))
    )


def line_up_watch(node, ctx, cfg, runner, watcher, args, locator, rest_tol, ear=None, out=None, rest_any=False):
    """--line-up: say where the container is until the engine answers, (went, wrist_turned, wrist_held:
    the wrist's own watch held the container as the go came, its last fix no older than LINE_UP_HELD_S).
    `rest_any` (a line-up in the middle of a run, guide_chair): a wrist that stands turned as the watch
    ends without a go is turned back to rest whoever turned it: this watch, or the run's own look before
    it, where the watch's own turn never began (a stop heard at once: the engine had no time to guide).

    The scene camera's sightings are said as they are made, as tasks/box_watch says them
    (SceneLocator.on_sighting). With --wrist-first, --execute and the arm at rest (within `rest_tol`) or at
    rest but for a wrist an earlier line-up left turned (wrist_looks_from), the fingers close as at every
    run's start and the wrist turns to look from where it stands (survey.watch_front, flown as a look), its
    frames' fixes said too. The wrist's turn is the only motion a line-up makes of its own: an arm that
    stands anywhere else is not brought home before the go, nothing turns, and the engine is told so
    (LINE_UP_NOT_AT_REST). With the orchestrator's selection (--instance-id) a wrist fix is said only where
    the run after the go would look for that container (selection_look, wrist_look's own rule); where it
    would not, why is said in the fix's place. On a go the run goes on from there, the wrist where it
    looked. On a stop the wrist is turned back to rest, as gently, and nothing was pressed (went False:
    main's EXIT_NOT_LINED_UP).
    A turn stopped part way, out or back: the arm holds where it stopped, the wrist turned, and the run ends
    here with the exit that says so (EXIT_LINE_UP_HOLDS), as a wrist look stopped part way ends a run."""
    import json

    from box_opening.tasks import box_watch
    from box_opening.tasks import survey as sv

    out = sys.stdout if out is None else out
    held_at = [None]  # when the wrist's own watch last said a fix: its camera held the container then
    refused = {}  # the newest find this watch would not say, as its "none" reads -> what that is of (judge)

    def say(msg):
        if msg.get("fix", {}).get("by") == "wrist":
            held_at[0] = time.monotonic()
        if "none" in msg and "kind" not in msg and msg["none"] in refused:
            # what the "none" is of, where this watch knows it better than its words tell (guide.K_*): the
            # engine says a state no moving of the chair cures as what it is, and ends on it (line_up.BoxSource)
            msg = dict(msg, kind=refused[msg["none"]])
        out.write(LINE_UP_MARK + json.dumps(msg) + "\n")
        out.flush()

    ear = LineUpEar() if ear is None else ear
    rest = rest_joints(cfg)
    # which robot this is, for the engine's reach_probe: the URDF this run's own kinematics were read from
    urdf = cli_common.arm_urdf_file()
    say({"robot": {"urdf": str(urdf) if urdf.is_file() else None, "root": ctx.worlds.base_frame}})

    def scene_look():
        """The scene camera's own look: (its fix or None, why it has none where it says)."""
        fix = None if locator is None else locator.locate(LINE_UP_LOOK_S)
        return fix, (str(locator.last_why) if locator is not None and fix is None and locator.last_why else None)

    def scene_frames():
        """How many of the scene camera's frames the segmenter gave the look that just ended (its record's
        own count), or None where the record does not say."""
        try:
            return int((locator.look_record() or {})["segmenter"]["frames"])
        except Exception:  # noqa: BLE001 - a record that cannot be read says nothing: the look's words do
            return None

    def scene_kind(why, frames):
        """What the scene camera's word on having no container is of (guide.K_*), by `frames`, how many of
        its frames the segmenter gave the looks since this was last said (a second of them; None: not
        known, and the word alone says). None at all, and the camera delivers nothing, whatever the newest
        look says (guide.K_FAULT). One or more, and it delivers: "no frame newer than the look" of ONE look
        of LINE_UP_LOOK_S, a third of a second, is a segmenter message that came a moment late, not a
        camera that is not working."""
        kind = guide.kind_of(why)
        if frames is None:
            return kind
        if not frames:
            return guide.K_FAULT
        return guide.K_UNSEEN if kind == guide.K_FAULT else kind

    if locator is not None:

        def told(sighting, age_s):
            line = box_watch.fix_line(sighting, age_s)
            if line is not None:
                say(line)

        locator.on_sighting = told
    selected = getattr(args, "instance_id", None) or None
    judge = None
    if selected:

        def judge(fix, find):
            """Why the run after the go would not take this find for the selected container, or None: nothing
            ties any find to it (selection_look: the id is in none of the scene camera's detections), or
            something does and what stands here is not it (tie_refusal, alone_refusal: wrist_first's own
            tests). What that is of goes with it (`refused`, for say): a find nothing can tie to the
            selection is one no moving of the chair makes the selected one (guide.K_NOT_CHOSEN), and so are
            two that could be it (K_SEVERAL); one that stands where the selected one does not is the
            selected one not in front of the arm (K_UNSEEN), which the chair cures."""
            tie, alone = selection_look(locator, None, ctx.model, selected, getattr(args, "wrist_alone", False))
            if tie is not None:
                why = tie_refusal(tie, find, fix.top_xyz)[0]
                kind = guide.K_SEVERAL if why is not None and guide.kind_of(why) == guide.K_SEVERAL else guide.K_UNSEEN
            elif alone:
                why, kind = alone_refusal(alone, find, fix.top_xyz, ctx.model), guide.K_SEVERAL
            else:
                why, kind = UNTIED % selected, guide.K_NOT_CHOSEN
            if why is not None:
                refused.clear()
                refused[str(why)[:300]] = kind  # (as watch_front cuts it for its "none")
            return why

    asked = bool(getattr(args, "wrist_first", False)) and bool(args.execute)
    wrist = asked and wrist_looks_from(ctx.client.joints(), rest, rest_tol)
    not_at_rest = asked and not wrist
    turned = False
    try:
        if wrist:
            _ok, finger_pos, _s = ctx.client.gripper_cmd(None)  # a query: nothing moves
            if not fingers.already_closed(finger_pos) and not runner.start_gripper(
                "press:close", GRIPPER_CMD_CLOSED, args.execute
            ):
                wrist = False
                log.info("[press_demo] LINE-UP: the fingers did not close: the wrist does not look")
        if wrist:
            log.info(
                "[press_demo] LINE-UP: the wrist turns to look at the table in front; the arm above it only "
                "settles onto its rest pose"
            )
            stood = list(ctx.client.joints())
            state, why = sv.watch_front(
                node,
                ctx,
                runner,
                watcher,
                args.execute,
                say,
                ear.heard,
                rest=rest,
                camera_cfg_name=ctx.worlds.wrist_camera_config,
                also=scene_look,
                judge=judge,
            )
            turned = state == "watched"
            if state == "stopped" and rest_distance(ctx.client.joints(), stood) <= rest_tol:
                # "stopped" is also the turn the client would not send at all (no fresh joint states, the
                # driver's action not there, its gate): the arm stands where it stood, nothing holds
                # part-turned, and the line-up goes on by the scene camera as for any look that did not start
                state = "not flown"
            if state == "stopped":
                log.info("[press_demo] STOP: the line-up's wrist turn: %s — the arm holds; nothing was pressed" % why)
                sys.exit(EXIT_LINE_UP_HOLDS)
            if state != "watched":
                log.info("[press_demo] LINE-UP: the wrist does not look (%s): the scene camera alone" % why)
                # (no camera's word on a container: the scene camera's own looks follow, and say what they see)
                say({"none": "wrist camera: %s" % str(why)[:260], "kind": guide.K_UNSEEN})
        elif not_at_rest:
            log.info("[press_demo] LINE-UP: the arm is not at rest: the wrist does not look before the go")
        last_none, frames = 0.0, 0
        while not ear.heard():  # the scene camera alone
            # this watch only sleeps and reads, so a Ctrl+C that did not arrive as an exception (it came out
            # as another error under the planner's wait, taken for a refusal) would never end it
            if getattr(getattr(ctx.client, "_abort", None), "requested", False):
                raise KeyboardInterrupt
            t_look = time.monotonic()
            fix, why = scene_look()
            got = None if locator is None else scene_frames()
            frames = None if got is None or frames is None else frames + got
            if (why is not None or not_at_rest) and time.monotonic() - last_none >= sv.LINE_UP_NONE_S:
                last_none = time.monotonic()
                if why is not None:
                    say({"none": why[:300], "kind": scene_kind(why, frames)})
                if not_at_rest:  # said last: the engine's end reason is the last word it heard
                    say({"none": LINE_UP_NOT_AT_REST})
                frames = 0
            # a pass with no fix lasts the look's period: with no scene camera, or one whose look answers at
            # once (no segmenter to read), the rest of it is slept out, and the loop never spins beside the
            # arm's driver. A look that ends early with its fix is followed by the next at once, as ever.
            left = LINE_UP_LOOK_S - (time.monotonic() - t_look)
            if fix is None and left > 0.0:
                time.sleep(left)
    finally:
        if locator is not None:
            locator.on_sighting = None
            # A scene fix made here left the segmenter rung following the selected container by where its
            # box was in the image then (SegmenterRung.follow). The scene camera rides the chair, which has
            # moved since: in the run that box may be another container's. Forgotten, so the run's first
            # look starts as a fresh run's does, the chair parked.
            seg = getattr(locator, "seg", None)
            if seg is not None:
                seg.near = None
    if ear.go:
        held = turned and held_at[0] is not None and time.monotonic() - held_at[0] <= LINE_UP_HELD_S
        log.info("[press_demo] LINE-UP: go — the run starts")
        runner.note("line_up", stage="go", wrist=turned, held=held)
        return True, turned, held
    runner.note("line_up", stage="stop", wrist=turned)
    back = turned
    if not turned and rest_any and args.execute:
        live = ctx.client.joints()
        back = rest_distance(live, rest) > rest_tol and wrist_looks_from(live, rest, rest_tol)
    if back:
        state, why = sv.rest_wrist(ctx, runner, args.execute, rest=rest)
        if state != "flown":
            log.info("[press_demo] STOP: the line-up ended and the wrist did not turn back (%s) — the arm holds" % why)
            sys.exit(EXIT_LINE_UP_HOLDS)
    log.info(
        "[press_demo] LINE-UP: ended without a go; nothing was pressed%s" % (", the wrist back at rest" if back else "")
    )
    return False, False, False


def far_tops(find, model):
    """The lid-sized tops a wrist look found that all stand out of the arm's reach: [(x, y, top z)], the
    nearest the arm first; [] where it found a container, found none, or has one within reach (which
    the finder then takes or refuses for its own reasons). The finder's own test of each place
    (survey._within_reach: reach_press.out_of_reach, the arm's kinematic bound), asked again of what
    it kept: a container the wrist can see and the arm cannot reach is one the chair is moved for
    (--guide), not a NO BOX."""
    from box_opening.tasks import survey as sv

    if find is None or getattr(find, "fix", None) is not None:
        return []
    try:
        tops = [
            (float(q.xy[0]), float(q.xy[1]), float(q.top_z))
            for q in getattr(find, "places", ()) or ()
            if getattr(q, "button", None) != "none"
        ]
        within = sv._within_reach(model)
        if not tops or any(within(top) for top in tops):
            return []
        return sorted(tops, key=lambda top: math.hypot(top[0], top[1]))
    except Exception:  # noqa: BLE001 - what it cannot read of a find is no reason to guide, and never the look's end
        return []


def looked_kind(survey):
    """What a wrist look that ended with no container saw (guide.K_*), or None where it did not look at
    all: no view to turn to, a view that was not flown, the body model off. Its camera delivering nothing
    at a view it stood at ("no wrist frame at ...": tasks/survey.run_survey) is the camera's fault,
    guide.K_FAULT; else, what its finder said of the frames it judged: two that could be the container
    (K_SEVERAL), or nothing that could (K_UNSEEN)."""
    kind = guide.kind_of(getattr(survey, "why", ""))
    if getattr(survey, "find", None) is None:
        return guide.K_FAULT if kind == guide.K_FAULT else None
    return guide.K_UNSEEN if kind == guide.K_FAULT else kind  # (its finder judged a frame: the camera delivers)


def no_container_kind(ctx, args, locator, scene_said=None):
    """What a find that ended with no container is OF (guide.K_*), for a run the engine can guide. Guidance
    is for what moving the chair cures: K_UNSEEN, nothing that could be the container in any view, and
    that alone is guided (main's NOT SEEN). The rest end the run as they did at 21aa7cc (NO BOX), each
    with what it was in its record for the engine's own words (review of 2026-11-19: all of them were
    guided with "The arm cannot see the container. Move the chair in front of it", for a wait that is five
    minutes on the UI):

    K_NOT_CHOSEN  a look saw something that could be the container and nothing ties it to the selection;
    K_SEVERAL     a look saw two that could be it, and nothing says which;
    K_FAULT       the camera the guidance would watch with delivered nothing.

    Which camera that is: the wrist's, where the wrist look is on (--wrist-first) and it looked from a
    view. It is then the wrist's word that counts, however alive or dead the scene segmenter is: a wrist
    camera with no frame is a fault whatever the scene camera gave, and a wrist that looked and saw
    nothing is a container not in view whatever the scene camera made of something further off. Else the
    scene camera's: a look the segmenter gave no frame of, a selected id in none of its frames (it cannot
    tell a container that is out of its view from an id no detection ever had: "was in none of its
    frames" is all it knows, and without the wrist nothing can say more), two places that pass."""
    wrist, side = getattr(ctx, "wrist_seen", None), getattr(ctx, "side_seen", None)
    for ends in (guide.K_NOT_CHOSEN, guide.K_SEVERAL):
        if ends in (wrist, side):
            return ends
    if getattr(args, "wrist_first", False) and wrist is not None:
        return wrist
    try:
        frames = ((locator.look_record() or {}).get("segmenter") or {}).get("frames")
    except Exception:  # noqa: BLE001 - a record that cannot be read is no frame known of
        frames = None
    if not frames:
        return guide.K_FAULT
    return guide.kind_of(scene_said or getattr(locator, "last_why", "")) or guide.K_UNSEEN


def forget_scene(ctx, worlds, watcher):
    """The chair is about to be moved: what this run's cameras measured of the scene was measured from
    where the chair stood, and is true of nowhere else. The table's height and extent (adopt_scene_table),
    the counter's edges the arm is held out of (keep_out_counter: the body check's run-time boxes), the
    box's pose and the wrist look's points are forgotten, so that the look after the move measures them
    again as a run's first look does (the owner's rule: the environment only from the cameras, each time)."""
    from box_opening.runtime import approach

    ctx.cpose = None
    ctx.counter_boxes = None
    approach.set_obstacles([])
    ctx.survey_clouds = None
    ctx.wrist_far = None
    ctx.wrist_seen = ctx.side_seen = None
    ctx.home_ahead = None
    worlds.table_top_z = None  # the world file's own again, where it names a table; else none until measured
    worlds.table_extent = None
    watcher.table_z = worlds.table_top_z
    watcher.roi = None


# The longest reason handed to the engine with a request to be guided (reason_for_engine). Run 2 of
# 2026-11-17's own line was 752 characters, four tilts with two joints' numbers each; a refusal that also
# names the shifted proofs runs to two or three times that. The request is one line on the pipe the run's log
# lines share, and a line of up to 4096 bytes is written whole (a pipe's atomic write): this keeps it under
# that with room for the box and the few characters JSON writes as six
GUIDE_WHY_MAX = 2500


def reason_for_engine(why, limit=GUIDE_WHY_MAX):
    """`why` (reach_press's line: "... (30 deg: <why>; 35 deg: <why>; ...)") as the engine is handed it:
    whole. As first written it was cut at 300 characters, which for run 2 of 2026-11-17's own line ended
    inside the second of its four tilts: everything the engine said from it (the progress line a person at
    open_box_terminal reads, the not_lined_up message) gave "the slide in swings the arm wide (30, 35 deg)"
    and dropped "the press stroke would pass too close to the user (40, 45 deg)" (review of 2026-11-17).
    Only a line longer than `limit` is shortened, and then at the end of a whole tilt's reason, never inside
    one: what is said of a tilt is its real reason or nothing."""
    import re

    why = str(why)
    if len(why) <= limit:
        return why
    ends = [m.start() for m in re.finditer(r"; \d+ deg: ", why) if m.start() <= limit]
    return (why[: ends[-1]] if ends else why[:limit]) + " ..."


# What a run asks to be guided for, said with its request and in its record (guide_chair's `what`) where it
# is not that no press plans for a container it found: one its wrist sees out of the arm's reach, or one no
# camera sees at all. The engine says each in its own words (node.guide_reason)
GUIDE_FAR, GUIDE_UNSEEN = "far", "unseen"
# ... and how the STOP reads where the arm does not get home for the chair to be moved
GUIDE_STOPS = {
    None: "no press plans from where the chair stands",
    GUIDE_FAR: "the container stands out of the arm's reach",
    GUIDE_UNSEEN: "no camera sees the container",
}


def guide_chair(node, ctx, cfg, runner, watcher, args, locator, rest_tol, place, why, n, ear, out=None, what=None):
    """--guide, where the run found no press from where the chair stands (reach_press.NO_WAY_IN), a
    container the wrist sees out of the arm's reach (`what` GUIDE_FAR), or no container at all (`what`
    GUIDE_UNSEEN, `place` None): the run does not end. Flatbot 2026-11-17 16:43 ended "FAILED ... arm:
    where the look left it"; the owner: "it should NOT just fail EVER ... don't end, just tell you how far
    to move until it's close enough then asking if ready y/n".

    The arm is brought to rest first where the run's look left it anywhere else (the side survey's pose):
    the chair is told to move only with the arm at rest, its wrist alone turned. Then the engine is asked
    to guide ({"guide": {"box", "why", "n"}}: where the box stood and why no press plans there, which the
    engine holds against that place; with `what`, that too), what the cameras measured from here is
    forgotten (forget_scene), and the run watches as a line-up's watch does (line_up_watch: the wrist
    turned to look, every frame's fix said) until the engine answers. On its go the run looks for the box
    again from where the wrist stands: (wrist_turned, wrist_held), as line_up_watch gives them. On a stop,
    or an engine that is gone, the wrist is turned back to rest and the run ends EXIT_NOT_LINED_UP, nothing
    pressed; a turn that stopped part way ends it EXIT_LINE_UP_HOLDS (line_up_watch).

    Nothing of this starts from a container having been seen, so a run that saw none is guided the same
    way. No place is handed on (`place` None: the engine holds nothing against any). This find placed no
    box and adopted no table, so forget_scene has only what an earlier round of the run measured to forget,
    which is as stale. And the wrist turns where it turns for every line-up: to the front view that shows
    the most of where the arm itself presses (survey.watch_front, with no hint), which is where the
    container has to be brought, the arm's own to know (its kinematics and the chair's model, solved on
    each run) and as true once the chair has moved. Not toward a table the cameras measured before the
    move: where that stood in the arm's frame is true of nowhere once the chair is driven, which is why
    it is forgotten."""
    import json

    out = sys.stdout if out is None else out
    rest = rest_joints(cfg)
    if not wrist_looks_from(ctx.client.joints(), rest, rest_tol):
        # ("TO REST": the flight view's word for this home. It is no refusal: as "NO WAY IN: ..." the view read
        # it as a second one, and printed a PRESS FAIL for a run that had tried no press: review of 2026-11-17)
        if not try_home(ctx, runner, args.execute, "TO REST: the arm goes to rest before the chair is guided"):
            log.info(
                "[press_demo] STOP: %s, and the arm did not get home for the chair to be moved — the arm holds; "
                "nothing was pressed" % GUIDE_STOPS.get(what, GUIDE_STOPS[None])
            )
            sys.exit(1)
    forget_scene(ctx, ctx.worlds, watcher)
    ear.again()
    box = None if place is None else [round(float(v), 4) for v in place[:3]]
    why = reason_for_engine(why)
    asked = {"n": n, "box": box, "why": why, **({"what": what} if what else {})}
    runner.note("guide", **asked)
    log.info(
        "[press_demo] GUIDE %d: the arm rests and the wrist looks while the chair is moved; the run goes on at "
        "the user's yes" % n
    )
    out.write(LINE_UP_MARK + json.dumps({"guide": asked}) + "\n")
    out.flush()
    went, turned, held = line_up_watch(
        node, ctx, cfg, runner, watcher, args, locator, rest_tol, ear=ear, out=out, rest_any=True
    )
    if not went:
        sys.exit(EXIT_NOT_LINED_UP)
    return turned, held


def home_first(ctx, runner, execute, rest, tol=AT_REST_RAD):
    """The mission's first motion: the arm to its exact rest pose (`rest`,
    HOME) unless it already stands within `tol` of it, so every run starts
    from the same place (owner, 2026-09-25). Planned like home_arm's move:
    in the bench world, whose keep-out band keeps it above wherever a box
    may stand. Returns False — nothing moved — when that is refused, with
    the recovery named: an arm that far down (holding at the table after an
    abort) is home_arm's case, whose bare-world and lift-first fallbacks
    want the operator's eye on the bench, not an automatic start. An arm
    parked inside the chair and user's padding steps out of it first, and
    with `execute` that step is flown here and judged where the arm then
    stands (core.home_any's `fly`); where it cannot be brought clear by the
    margin, StepOutStopped is raised after its STOP line: the arm has moved.

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
    # an arm parked inside the chair and user's padding (RAMMP 1.5, 2026-11-19: where the team's rest pose
    # leaves it, at the start of every run) steps out of it first, and that step is flown as soon as it is
    # planned: home is then planned from where the arm stands, by its measured joints (core.home_any's `fly`)
    fly = flown_then_measured(ctx, runner) if execute else None
    try:
        legs = _home_first_legs(ctx, live, rest, world, fly=fly)
    except StepOutStopped as e:
        # the arm has moved and is not clear by the margin: no "not starting", no "nothing moved". It holds,
        # and main ends the run as any stop with the arm holding (exit 1: the node's arm_holds), not as a
        # start that was refused (exit 3: not_home, which a caller may simply ask for again)
        log.info(
            "[press_demo] STOP: the step out of the chair and user's padding did not bring the arm clear — %s. "
            "No way to the rest pose was planned and nothing else moves in this run." % e
        )
        raise
    except RuntimeError as e:
        log.info(
            "[press_demo] not starting: the move home was refused in the bench world (%s) — the arm is "
            "probably down at the table, and this run does not start from there. Home the arm on its own first:\n"
            "    scripts/sheppy_task.sh home_arm --execute    (add --lift-first 0.08 if that is refused too)" % e
        )
        return False
    res = runner.run(legs, execute=execute)
    return all(r.ok for r in res)


def _home_first_legs(ctx, live, rest, world, fly=None):
    """home_first's move from the live joints `live`: HOME (`rest`) planned
    in `world` — after home_arm's plan-free step along the joint line toward
    it, the shortest one HOME plans from (primitives.ClearMarginFree, as
    try_home steps), where a joint stands too near its limit for the planner
    to start from. Planned directly, the press study's replay (11-06, E-S8)
    refused 3 recorded run starts (arm_joint_4 at its limit, 10-30 10:08 and
    11-02 09:45, or 0.070 rad from it, 10-30 13:26) and 1 run end
    (arm_joint_6 0.088 rad from it, 10-29 14:45) START_IN_LIMIT_MARGIN: exit
    3 before anything moved. Since 2026-11-09 through core.home_any, as home_arm: a start inside the chair
    and user's padding steps out of it first (EscapeBodyFree), HOME is asked for again, and a waypoint
    along the joint line is tried. Raises the last refusal."""

    def plan_to(at, name, q):
        return as_home(_plan_motion(ctx, at, name, ("joints", list(q)), world, HOME_SPEED)[0])

    return home_any(
        ctx, _state(live), rest, plan_to, who="[press_demo]", goal_words="the rest pose", plan_words="home", fly=fly
    )


# The orchestrator's selection (--instance-id) is one of the SCENE camera's detections, and the
# wrist's finder knows a lid by its geometry, not which container was selected. What the wrist finds
# is the selected container only where it stands in the direction the scene camera has that
# container (SelectionTie): no farther than SELECTED_ACROSS_M to either side of the scene camera's
# line of sight to it, and, where the scene camera placed it, within SELECTED_ALONG_M of that place
# along the line. A camera has an object's direction right and its range wrong: the one pair of
# fixes of one box by both cameras (2026-11-11 19:13, scene [0.787, -0.240], wrist [0.728, -0.295])
# differs 80 mm along the scene camera's line of sight and 8 mm across it, and the calibration
# study's offset for today's mount (+65, +50 mm, sd 30) lies along that line too for a box in front
# of the arm. Across: two of these containers cannot stand with their centres nearer than 152 mm
# (their holders' flanges touch), so a lid no more than 75 mm off the line is nearer to it than a
# neighbour standing beside the selected one can be. Along: the range error seen (80-82 mm, sd 30)
# and a rough fit's own (43 mm at worst). A second lid-sized top in the same direction, and nothing
# says which is the selected one: neither is taken (a wrong fix is worse than NO BOX).
SELECTED_ACROSS_M = 0.075
SELECTED_ALONG_M = 0.20


@dataclass(frozen=True)
class SelectionTie:
    """Where the scene camera has the selected container, as lines from its lens in the base
    frame's horizontal plane: `a`, `b` the unit directions to the container's two sides as it sees
    them — one line (a == b) through `at_xy` where it placed the lid (a fix: whole, cut or rough),
    else the two sides of the instance's own box in its image, `wider_m` more past a side its frame
    cuts the box at (the lid's width: the part out of view lies there)."""

    instance_id: str
    cam_xy: tuple
    a: tuple
    b: tuple
    at_xy: tuple = None
    wider_m: tuple = (0.0, 0.0)
    lid_m: float = 0.105
    by: str = "where the scene camera placed it"

    def judge(self, xy):
        """(tied, words): is (x, y) in the direction the scene camera has the selection, and how
        far off it lies, said."""
        rx, ry = float(xy[0]) - self.cam_xy[0], float(xy[1]) - self.cam_xy[1]
        ahead = rx * (self.a[0] + self.b[0]) + ry * (self.a[1] + self.b[1])
        if ahead <= 0.0:
            return False, "behind the scene camera's view of it"
        turn = self.a[0] * self.b[1] - self.a[1] * self.b[0]
        s = -1.0 if turn < 0.0 else 1.0
        out_a = -s * (self.a[0] * ry - self.a[1] * rx)  # > 0: outside, past the side `a`
        out_b = -s * (rx * self.b[1] - ry * self.b[0])
        across = max(out_a - self.wider_m[0], out_b - self.wider_m[1], 0.0)
        words = "%.0f mm to the side of the scene camera's line of sight to it" % (1000 * across)
        tied = across <= SELECTED_ACROSS_M
        if self.at_xy is not None:
            reach = math.hypot(self.at_xy[0] - self.cam_xy[0], self.at_xy[1] - self.cam_xy[1])
            along = rx * self.a[0] + ry * self.a[1] - reach
            words += " and %.0f mm %s along it than %s" % (
                1000 * abs(along),
                "nearer" if along < 0 else "farther",
                self.by,
            )
            tied = tied and abs(along) <= SELECTED_ALONG_M
        return tied, words

    def rival(self, find, xy):
        """(x, y) of another lid-sized top the wrist's frames hold in the selection's direction — a
        place that passes, or a top a view cut — at least half a lid from `xy`; or None."""
        tops = [q.xy for q in getattr(find, "places", ()) if getattr(q, "button", "") != "none"]
        tops += [p.xy for p in getattr(find, "patches", ()) if getattr(p, "why", "").endswith("cut off by the view")]
        for t in tops:
            if math.hypot(t[0] - xy[0], t[1] - xy[1]) >= 0.5 * self.lid_m and self.judge(t)[0]:
                return float(t[0]), float(t[1])
        return None


def selection_tie(locator, scene_fix, model, instance_id):
    """The SelectionTie for the selected container `instance_id` after the scene look: by the scene
    camera's own fix of it (`scene_fix`: whole, cut or rough) where it has one; else by the
    instance's own box in the scene image (the segmenter's newest frame that holds it:
    SegFrame.mine), its two sides as lines of sight from the lens; None where neither says where the
    selected container is (it was in none of the frames, or TF or the intrinsics are missing):
    nothing would tie what the wrist finds to it."""
    import numpy as np

    T = locator.base_to_color() if locator is not None else None
    if T is None:
        return None
    T = np.asarray(T, float)
    cam = (float(T[0, 3]), float(T[1, 3]))
    lid_m = float(max(model.lid_dims[0], model.lid_dims[1]))

    def unit(dx, dy):
        n = math.hypot(dx, dy)
        return None if n < 1e-6 else (dx / n, dy / n)

    if scene_fix is not None:
        at = (float(scene_fix.top_xyz[0]), float(scene_fix.top_xyz[1]))
        d = unit(at[0] - cam[0], at[1] - cam[1])
        if d is None:
            return None
        by = "where the scene camera placed it%s" % (" roughly" if getattr(scene_fix, "rough", False) else "")
        return SelectionTie(str(instance_id), cam, d, d, at_xy=at, lid_m=lid_m, by=by)
    frame = getattr(getattr(locator, "seg", None), "latest", None)
    box = getattr(frame, "mine", None)
    g = getattr(locator, "grab", None)
    k = getattr(g, "k", None)
    if box is None or k is None:
        return None
    k = np.asarray(k, float).reshape(3, 3)
    v = 0.5 * (float(box[1]) + float(box[3]))
    px = np.array([[float(box[0]), v], [float(box[2]), v]])
    dist = getattr(g, "dist", None)
    if dist is not None and np.any(np.asarray(dist, float)):
        import cv2

        n = cv2.undistortPoints(px.reshape(-1, 1, 2), k, np.asarray(dist, float)).reshape(-1, 2)
    else:
        n = np.stack([(px[:, 0] - k[0, 2]) / k[0, 0], (px[:, 1] - k[1, 2]) / k[1, 1]], 1)
    rays = np.concatenate([n, np.ones((2, 1))], 1) @ T[:3, :3].T  # the two sides' lines of sight, base frame
    a, b = unit(rays[0, 0], rays[0, 1]), unit(rays[1, 0], rays[1, 1])
    if a is None or b is None:
        return None
    from box_opening.perception.scene_source import frame_edges

    wh = getattr(frame, "wh", None) or (0, 0)
    color = getattr(g, "color", None)
    if not wh[0] and color is not None:
        wh = (int(color.shape[1]), int(color.shape[0]))
    cut = frame_edges(box, wh) if wh[0] else ()
    wider = (lid_m if "left" in cut else 0.0, lid_m if "right" in cut else 0.0)
    return SelectionTie(
        str(instance_id), cam, a, b, wider_m=wider, lid_m=lid_m, by="its box in the scene camera's image"
    )


def selection_unseen(locator):
    """Whether the scene look that just ended held the selected container in NONE of the segmenter's frames (its
    look record's `selected`: 0 of however many it read, a segmenter that cannot be read included). The
    selection is then no detection of the scene camera's at all: an object list typed by hand (the UI's on
    snorlax), or an id its tracker has since dropped. False where the record does not say."""
    record = locator.look_record() if locator is not None else None
    seg = (record or {}).get("segmenter")
    return seg is not None and seg.get("selected") == 0


def selection_look(locator, scene_fix, model, selected, wrist_alone):
    """Whether the wrist looks for the selected container `selected`, and by what its find is then taken:
    (tie, alone). `tie`: the SelectionTie (selection_tie), or None where nothing the scene camera saw says
    where the selected container is. `alone`: the selected id where there is no tie, the run has
    --wrist-alone (`wrist_alone`), the scene camera has no fix and its last look held the selection in none
    of its frames (selection_unseen); else None. Neither: the wrist is not turned for it. The run's own look
    (wrist_look) and the line-up's watch before it (line_up_watch) both ask this, so the panel never asks
    the user to open a container the run would then not look for."""
    tie = selection_tie(locator, scene_fix, model, selected)
    alone = tie is None and bool(wrist_alone) and scene_fix is None and selection_unseen(locator)
    return tie, (selected if alone else None)


def other_tops(find, xy, lid_m):
    """Every lid-sized top the wrist's frames hold besides the one at `xy` (a place that passes, or a top a view
    cut), at least half a lid from it and from each other: [(x, y), ...]."""
    tops = [q.xy for q in getattr(find, "places", ()) if getattr(q, "button", "") != "none"]
    tops += [p.xy for p in getattr(find, "patches", ()) if getattr(p, "why", "").endswith("cut off by the view")]
    out = []
    for t in tops:
        if all(math.hypot(t[0] - o[0], t[1] - o[1]) >= 0.5 * lid_m for o in [xy, *out]):
            out.append((float(t[0]), float(t[1])))
    return out


def tie_refusal(tie, find, at):
    """(why the wrist's find at `at` is not taken for the selected container `tie` ties, or None where it is
    taken; how the tie judged it): it does not stand in the direction the scene camera has the selected
    container, or a second lid-sized top of `find` stands in that direction too. wrist_first's rule, asked
    by the line-up's watch as well, so the panel asks the user to open only what the run would take."""
    tied, how = tie.judge(at[:2])
    rival = tie.rival(find, at[:2]) if tied else None
    if tied and rival is None:
        return None, how
    why = "a container at [%.3f, %.3f], %s" % (at[0], at[1], how)
    if rival is not None:
        why += ", and another lid-sized top at [%.3f, %.3f] in the same direction" % rival
    return why + ": not taken for the selected container %s" % tie.instance_id, how


def alone_refusal(alone, find, at, model):
    """Why the wrist's find at `at` is not taken for the selected container `alone` (--wrist-alone: in none
    of the scene camera's detections, so nothing ties a find to it), or None where it is: a second lid-sized
    top stands in the wrist's frames, and nothing says which of the two was selected. A top out of the
    arm's reach is no second container: the arm could not open it, so it is not what was selected to be
    opened, the rule the finder itself goes by (wrist_finder: "a place the arm cannot reach is no rival of
    one it can"; both runs that opened the box on 2026-11-12 had a lid-sized top 1.08 m out in the wrist's
    frame, which refused every find here). wrist_first's rule, asked by the line-up's watch as well."""
    from box_opening.tasks import survey as sv

    within = sv._within_reach(model)
    others = other_tops(find, at[:2], float(max(model.lid_dims[0], model.lid_dims[1])))
    others = [o for o in others if within((o[0], o[1], float(at[2])))]
    if not others:
        return None
    return (
        "a container at [%.3f, %.3f] and another lid-sized top at [%.3f, %.3f]: two containers, and the "
        "selected one (%s) is in none of the scene camera's detections, so nothing says which it is"
        % (at[0], at[1], *others[0], alone)
    )


def wrist_first(node, ctx, cfg, runner, watcher, args, scene_fix, said=None, tie=None, alone=None, untied=None):
    """--wrist-first: the wrist turned where the arm rests (tasks/survey.run_survey, front) and
    the container its camera finds in front of the arm: the fix (a SceneFix, source "wrist"), its
    frames' points kept on ctx — or None where it finds none, the run going on with the scene
    camera's `scene_fix` (None: its NO BOX). Where the scene camera has the box (a careful fix, or
    one placed roughly from a box its frame cut: SceneFix.rough) the wrist looks there first.
    A turn stopped part way: the arm holds (exit 1); a preview ends at the turned pose (exit 0).
    `said`: a list the look's own word is added to where it finds none (the run's NO BOX says it).
    `tie` (a SelectionTie: the orchestrator's selection): what the wrist finds is taken only where
    it stands in the direction the scene camera has the selected container, and no second lid-sized
    top does; else it is not taken (None), and said.
    `alone` (the selected id, --wrist-alone: a selection the scene camera's detections do not hold,
    so nothing ties a find to it): what the wrist finds is taken only where it is the ONE container
    its frames hold; a second lid-sized top anywhere in them, and it is not taken (None), and said.
    `untied` (the selected id, of a run the engine can guide, where nothing ties a find to it and
    --wrist-alone is off): the wrist looks only to tell whether anything that could be the container
    stands in front of the arm. What it finds is never taken (None), and said.
    What the look saw where it took no container is left on ctx.wrist_seen (looked_kind; of a find
    it would not take: guide.K_NOT_CHOSEN, K_SEVERAL, or K_UNSEEN where the selected one stands
    elsewhere), for main's word on what the find was (no_container_kind)."""
    from box_opening.tasks import survey as sv

    survey = sv.run_survey(
        node,
        ctx,
        runner,
        watcher,
        args.execute,
        rest=rest_joints(cfg),
        camera_cfg_name=ctx.worlds.wrist_camera_config,
        front=True,
        hint=None if scene_fix is None else tuple(float(v) for v in scene_fix.top_xyz[:2]),
    )
    if survey.stopped:
        log.info("[press_demo] STOP: the wrist look: %s — the arm holds; nothing was pressed" % survey.why)
        sys.exit(1)
    if survey.preview:
        log.info("[press_demo] preview ends at the wrist look's pose: its frames, the aim and the press need --execute")
        sys.exit(0)
    # lid-sized tops it saw that all stand out of the arm's reach: with --guide the chair is moved for them
    ctx.wrist_far = far_tops(survey.find, getattr(ctx, "model", None))
    # ... and what it saw where it finds no container, or none it takes: a camera that delivered nothing,
    # two that could be the container, or nothing that could (looked_kind)
    ctx.wrist_seen = None if survey.fix is not None else looked_kind(survey)
    if untied:
        # it looked only to tell what stands in front of the arm. Anything that could be the container
        # (a find, two the finder cannot choose between, a lid-sized top out of the arm's reach) is
        # something nothing ties to the selection: the chair is not guided to it, near or far, for no
        # place of the chair makes it the one selected
        far = ctx.wrist_far
        if survey.fix is not None or ctx.wrist_seen == guide.K_SEVERAL or far:
            if survey.fix is not None:
                seen = "a container at [%.3f, %.3f]" % tuple(survey.fix.top_xyz[:2])
            elif far:
                seen = "a lid-sized top at [%.2f, %.2f], out of the arm's reach" % tuple(far[0][:2])
            else:
                seen = "two places that could be a container"
            why = (
                "%s, and the selected container %s is in none of the scene camera's detections: nothing says "
                "it is the one selected" % (seen, untied)
            )
            ctx.wrist_seen, ctx.wrist_far = guide.K_NOT_CHOSEN, None
            log.info("[press_demo] WRIST LOOK: %s" % why)
            runner.note("wrist_first", found=survey.fix is not None, tied=False, untied=True, why=why, scene=False)
            if said is not None:
                said.append(why)
            return None
    if survey.fix is not None and tie is not None:
        at = survey.fix.top_xyz
        why, how = tie_refusal(tie, survey.find, at)
        if why is not None:
            # (a second top in the selection's direction: two that could be it; else the selected one
            # stands where the wrist does not see it, which moving the chair cures)
            ctx.wrist_seen = guide.K_SEVERAL if guide.kind_of(why) == guide.K_SEVERAL else guide.K_UNSEEN
            log.info("[press_demo] WRIST LOOK: %s" % why)
            runner.note("wrist_first", found=True, tied=False, why=why, scene=scene_fix is not None)
            if said is not None:
                said.append(why)
            return None
        log.info(
            "[press_demo] WRIST LOOK: the container it finds is the selected one (%s): %s" % (tie.instance_id, how)
        )
    if survey.fix is not None and alone:
        at = survey.fix.top_xyz
        why = alone_refusal(alone, survey.find, at, ctx.model)
        if why is not None:
            ctx.wrist_seen = guide.K_SEVERAL
            log.info("[press_demo] WRIST LOOK: %s" % why)
            runner.note("wrist_first", found=True, alone=True, taken=False, why=why, scene=False)
            if said is not None:
                said.append(why)
            return None
        log.info(
            "[press_demo] WRIST LOOK: the one container it finds, at [%.3f, %.3f], is taken for the selected %s, "
            "which is in none of the scene camera's detections (--wrist-alone)" % (at[0], at[1], alone)
        )
        runner.note("wrist_first", found=True, alone=True, taken=True, at=[round(float(v), 4) for v in at[:2]])
    if survey.fix is not None:
        ctx.survey_clouds = survey.clouds
        if scene_fix is not None:
            a, b = survey.fix.top_xyz, scene_fix.top_xyz
            log.info(
                "[press_demo] WRIST LOOK: the box at [%.3f, %.3f, %.3f]; the scene camera had it at [%.3f, %.3f, "
                "%.3f]%s, %.0f mm away — the wrist's is taken"
                % (
                    *a,
                    *b,
                    " (placed roughly)" if getattr(scene_fix, "rough", False) else "",
                    1000 * math.hypot(a[0] - b[0], a[1] - b[1]),
                )
            )
        return survey.fix
    log.info(
        "[press_demo] WRIST LOOK: no container (%s) — %s"
        % (
            survey.why,
            "the scene camera's box is taken%s" % (", placed roughly" if getattr(scene_fix, "rough", False) else "")
            if scene_fix is not None
            else "and the scene camera has none",
        )
    )
    runner.note("wrist_first", found=False, why=survey.why, scene=scene_fix is not None)
    if said is not None:
        said.append(survey.why)
    return None


def wrist_look(node, ctx, cfg, runner, watcher, args, locator, scene_fix, said):
    """--wrist-first after the scene look, with or without the orchestrator's selection: (the fix
    the run goes on with or None, the scene locator that checks the box again after the press or
    None, what the NO BOX says of the scene camera or None for its own word).

    No selection: the wrist's fix where it finds the box (wrist_first), else the scene camera's.
    A selection (--instance-id; the id the world state is written for is the caller's and never
    changes here): the wrist looks first where the scene camera has the selected container and
    its fix is taken where it is tied to the selection (SelectionTie: by the scene camera's fix of
    it — whole, cut or rough — else by the instance's own box in the scene image). Where the wrist
    finds none, or none tied, the run goes on with the scene camera's fix of the selected
    container, unless that fix is a rough one (a box its frame cut): that is no place to open a
    container on by itself, and the run ends NO BOX. Where nothing the scene camera saw says where
    the selected container is, a run by hand does not turn the wrist at all, and one the engine can
    guide (--guide) turns it only to tell whether anything that could be the container stands in
    front of the arm (wrist_first's `untied`: nothing it finds is taken): it could not tell which container
    it found.

    --wrist-alone (off unless a deployment sets it) is for a selection that is no detection of the
    scene camera's: the selected id was in NONE of its frames (selection_unseen: an object list
    typed by hand). The wrist then looks from its own views, tied to nothing, and its find is taken
    only where it is the one container its frames hold (wrist_first's `alone`); two, and NO BOX. A
    selection the scene camera did hold is tied as above, with or without it."""
    selected = getattr(args, "instance_id", None) or None
    tie, alone, untied = None, None, None
    if selected:
        tie, alone = selection_look(locator, scene_fix, ctx.model, selected, getattr(args, "wrist_alone", False))
        if alone:
            log.info(
                "[press_demo] WRIST LOOK: the selected container %s is in none of the scene camera's detections: the "
                "wrist looks from its own views, and what it finds is taken only where it is the one container it "
                "sees (--wrist-alone)" % selected
            )
        elif tie is None and getattr(args, "guide", False):
            # A run the engine can guide. Nothing the wrist finds can be taken for this selection, and the
            # run still has to know which of two things this is: a container that stands in no camera's
            # view, which moving the chair cures and the engine guides for (owner 2026-11-19), or a
            # container in front of the arm that nothing ties to the one selected, which no moving of the
            # chair cures. The scene camera cannot say (the id "was in none of its frames" either way), so
            # the wrist looks, as it looks for every other find, and takes nothing (wrist_first's `untied`).
            # As first guided (e16c35c) this find went straight into "The arm cannot see the container.
            # Move the chair in front of it" with a lid in every frame of the watch (review of 2026-11-19)
            untied = selected
            log.info(
                "[press_demo] WRIST LOOK: the selected container %s is in none of the scene camera's detections: "
                "the wrist looks only to tell whether anything that could be it stands in front of the arm (what "
                "it finds is not taken)" % selected
            )
        elif tie is None:
            said.append(
                "not looked for: nothing the scene camera saw says where the selected container %s is, and the "
                "wrist cannot tell which container was selected" % selected
            )
            log.info("[press_demo] WRIST LOOK: %s" % said[-1])
    if not selected or tie is not None or alone or untied:
        wrist_fix = wrist_first(
            node, ctx, cfg, runner, watcher, args, scene_fix, said=said, tie=tie, alone=alone, untied=untied
        )
        if wrist_fix is not None:
            return wrist_fix, None, None
    if selected and scene_fix is not None and getattr(scene_fix, "rough", False):
        t = scene_fix.top_xyz
        scene_said = (
            "the selected container %s is %s, at [%.3f, %.3f]: where the wrist does not find it there, that is no "
            "place to open it on" % (selected, scene_fix.partial or "placed roughly", t[0], t[1])
        )
        log.info("[press_demo] scene camera: %s" % scene_said)
        return None, None, scene_said
    return scene_fix, locator, None


def side_survey_has_a_turn(args):
    """Whether the wrist's survey over the arm's side is tried where nothing else finds the container
    (wrist_survey_or_exit): --wrist-survey, and no selection of the orchestrator's (the wrist cannot tell
    which container was selected)."""
    return bool(getattr(args, "wrist_survey", False)) and not getattr(args, "instance_id", None)


def wrist_survey_or_exit(
    node, ctx, cfg, runner, watcher, args, locator, wrist_look=None, scene_said=None, or_none=False, unseen=None
):
    """The scene camera found no box (locator.last_why): the wrist survey's fix (tasks/survey) with
    --wrist-survey, its frames' points kept on ctx (survey_clouds); else the run's end — NO BOX and
    home (exit 2), as without the survey; a survey motion stopped part way: the arm holds (exit 1);
    a preview: it ends at the survey pose (exit 0). A run given the orchestrator's selection
    (--instance-id) does not survey: the selection is one of the scene camera's detections, and the
    wrist cannot tell which container it is (docs/interface.md). `wrist_look`: what the wrist
    look in front (--wrist-first) said where it found none too, said with the NO BOX; `scene_said`:
    what to say of the scene camera where its own last word is not it (a rough fix not opened on).
    `or_none` (a guided run whose wrist saw a lid-sized top in front, out of the arm's reach): a survey
    that finds no container is not the run's end; None is returned, the survey's word said, and the caller
    has the chair guided to that top (main).
    `unseen` (a list, of a run the engine can guide: --guide): where nothing that could be the container
    is in any view (no_container_kind: guide.K_UNSEEN), that is not the run's end either (owner
    2026-11-19): why is added to it, None is returned with no NO BOX and no home, and the caller has the
    chair guided (main's NOT SEEN). What no moving of the chair cures (a camera that delivered nothing, a
    container nothing ties to the selection, two that could be it) is the NO BOX it was, in the same
    words, with what it was in its row (`what`) for the engine's own sentence."""
    from box_opening.tasks import survey as sv

    why = "scene camera: %s" % (scene_said or locator.last_why)
    if wrist_look:
        why += "; the wrist's look in front: %s" % wrist_look
    if getattr(args, "wrist_survey", False) and args.instance_id:
        why += (
            "; no wrist survey for the selected container %s: the selection is the scene camera's, and the wrist "
            "cannot tell which container the orchestrator selected" % args.instance_id
        )
    elif getattr(args, "wrist_survey", False):
        survey = sv.run_survey(
            node,
            ctx,
            runner,
            watcher,
            args.execute,
            rest=rest_joints(cfg),
            camera_cfg_name=ctx.worlds.wrist_camera_config,
        )
        if survey.stopped:
            log.info("[press_demo] STOP: the wrist survey: %s — the arm holds; nothing was pressed" % survey.why)
            sys.exit(1)
        if survey.preview:
            log.info(
                "[press_demo] preview ends at the wrist survey's pose: its frames, the aim and the press need --execute"
            )
            sys.exit(0)
        if survey.fix is not None:
            ctx.survey_clouds = survey.clouds
            return survey.fix
        why += "; wrist survey: %s" % survey.why
        ctx.side_seen = looked_kind(survey)  # what the look over the arm's side saw (no_container_kind)
        if or_none:
            log.info(
                "[press_demo] WRIST SURVEY: no container beside the arm either (%s): the chair is guided to the "
                "lid-sized top the wrist saw in front" % survey.why
            )
            return None
    what = {}
    if unseen is not None:
        kind = no_container_kind(ctx, args, locator, scene_said)
        if kind == guide.K_UNSEEN:
            unseen.append(why)
            return None
        what = {"what": kind}  # (a run by hand says none: its row is as it always was)
    # the look's seconds and the segmenter's verdict: what a NO BOX was
    runner.note("reach_refused", why="NO BOX — %s" % why, **what, **locator.look_record())
    try_home(ctx, runner, args.execute, "NO BOX — %s" % why)
    sys.exit(2)


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
    ap.add_argument(
        "--instance-id",
        default=None,
        help="the container to open: its instance_id on /perception/object_detections (the box_opening "
        "engine passes the orchestrator's selection). Without it, any box of the container's classes",
    )
    ap.add_argument(
        "--line-up",
        action="store_true",
        help="for the engine's line-up (line_up.BoxSource): before the run, say where the container is as the "
        'user drives up, a line each on stdout, and wait for a line on stdin: {"go": true} and the run goes '
        'on, {"stop": true} (or the pipe closed) and it ends with nothing pressed. With --wrist-first and the '
        "arm at rest the wrist turns to look, gently, and turns back on a stop; else only the scene camera looks "
        "and nothing moves before the go",
    )
    ap.add_argument(
        "--guide",
        action="store_true",
        help="for the engine (node.Capability._follow): where no press plans from where the chair stands, the "
        "wrist sees the container out of the arm's reach, or no camera that looked sees a container at all, the "
        "run does not end. It brings the arm to rest, says "
        '{"guide": ...} on stdout, watches as --line-up does while the engine tells the user how to move the '
        'chair, and on {"go": true} on stdin looks for the box again and goes on; {"stop": true} (or the pipe '
        "closed) and it turns the wrist back and ends with nothing pressed. Only with --execute",
    )
    ap.add_argument(
        "--wrist-first",
        action="store_true",
        help="the wrist camera finds the container: before anything else the wrist turns where the arm rests so "
        "its camera sees the table in front (up to four views, first where the scene camera has the box; a lid a "
        "view cuts is looked at again), and what it finds there is the box (a table faced from the front; "
        "tasks/survey); the scene camera's box only where the wrist finds none; with --instance-id the wrist's "
        "box is taken only where it stands in the direction the scene camera has the selected container",
    )
    ap.add_argument(
        "--wrist-alone",
        action="store_true",
        help="with --wrist-first and --instance-id: a selected container that is in NONE of the scene camera's "
        "detections (an object list typed by hand) is looked for by the wrist from its own views, tied to nothing, "
        "and what it finds is taken only where it is the one container the wrist sees (two or more: NO BOX). "
        "Without it such a run ends NO BOX and the wrist is not turned",
    )
    ap.add_argument(
        "--wrist-survey",
        action="store_true",
        help="where the scene camera finds no container, look for it with the wrist camera from one or two poses "
        "over the arm's side (a chair parked side-on to a counter; tasks/survey); not with --instance-id",
    )
    args = ap.parse_args()
    cli_common.say_rest_or_exit(args, "[press_demo]")  # first: which rest pose this run takes
    # the reach-out reads it (reach_press.run_reach_press): its depth press
    # is the run (owner 2026-10-28), whatever the command line said
    args.depth_press = True
    args.compliant = COMPLIANT_DEFAULT and not args.guarded
    args.guide = bool(args.guide and args.execute)  # a preview moves nothing and is guided nowhere
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
    from box_opening.runtime.approach import chain_ready, set_chain_root, set_rest_goals

    set_chain_root(worlds.base_frame, urdf_path=cli_common._arm_urdf_path(node, worlds.base_frame))
    ok, why = chain_ready()
    if not ok:
        # every aim, press and reach check needs the chain: without it the
        # run flew to staging and died there on a KeyError (review 2026-10-22)
        sys.exit(
            "[press_demo] not starting: /arm/robot_description not heard in 5 s (arm_state_publisher down?) — "
            "the arm kinematics for %s are unavailable (%s). Nothing moved." % (worlds.base_frame, why)
        )
    # the chair and its user, every plan-free line held out of them (runtime/body)
    body_model = cli_common.load_body_or_exit(args, "[press_demo]")
    # the robot's own rest pose, before anything reads HOME (runtime/rest: one inside the chair's padding
    # or at a joint limit ends the run here)
    cli_common.load_rest_or_exit(args, "[press_demo]")
    # the goals the body check lets through: the rest this run starts and ends at (the built-in HOME
    # stands inside some chair models' armrest; every goal the run chooses is held to it)
    set_rest_goals([list(HOME), rest_joints(cfg)])
    runner = Runner(client)
    cli_common.apply_speed_scale(runner, args)
    # PERCEPTION ON ITS OWN THREAD. The cameras and the scene camera's
    # detector rungs live on a second node spun by a background executor, so
    # the control loop's spin cadence (a 1 kHz joint-state stream, action
    # feedback, planning calls) never starves them: on the stub bench the
    # single executor gave the wrist's detector 1.6 frames a second
    # (2026-09-16). No detector runs here: the box comes from sheppy's scene
    # segmenter (/perception/object_detections).
    pnode = rclpy.create_node("box_opening_perception")
    from rclpy.executors import SingleThreadedExecutor

    pexec = SingleThreadedExecutor()
    pexec.add_node(pnode)
    import threading

    pthread = threading.Thread(target=pexec.spin, name="perception", daemon=True)
    ctx = None  # the finally below runs even if startup fails before Ctx
    pthread.start()
    try:
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
                    instance_id=args.instance_id,
                )
                # a second container the arm cannot reach is no rival of one it can (scene_source.within_reach_rivals)

                def _within_reach(top, _m=model):
                    from box_opening.tasks.reach_press import out_of_reach

                    return not out_of_reach(_m, top)

                locator.reachable = _within_reach
                locator.prime()  # the segmenter's state, for the log
        # camera on from here to exit
        ctx = Ctx(
            model=model,
            cpose=None,
            client=client,
            worlds=worlds,
            config_path=cfg_path,
        )

        ctx.press_cfg = cfg  # recovery homes go to the mission's rest pose
        if args.execute and not (args.detect_only or args.wrist_detect) and mission_frames_enabled():
            # a debug switch ($BOX_OPENING_MISSION_FRAMES, off by default): the run keeps
            # what its cameras saw and, when the press lands, where the button really
            # was, as a placement of the detection set (detection_set.MissionFrames;
            # written at the end, the newest MISSIONS_KEPT kept)
            ctx.mission_frames = MissionFrames(
                state_dir() / "detection_sets" / MISSIONS_SET, Path(cfg_path).name, worlds.table_top_z
            )
        why = readiness_refusal(client, args.execute, camera_cfg_name=worlds.wrist_camera_config)
        if why:
            sys.exit("[press_demo] not starting: %s" % why)
        # the planner up: with the body model on, a run that moves must plan around the same chair and
        # user (--detect-only moves nothing: said only)
        cli_common.planner_body_or_exit(node, body_model, args.execute and not args.detect_only, "[press_demo]")
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
        wrist_turned = wrist_held = False
        # the engine's answers on stdin, heard for the whole run: its go or stop to the line-up before the
        # run (--line-up), and to each one in the middle of it (--guide)
        ear = LineUpEar() if (getattr(args, "line_up", False) or args.guide) else None
        if getattr(args, "line_up", False):
            # THE LINE-UP (--line-up; owner 2026-11-16): the user is told which way to drive until the
            # container is where the arm opens it, and nothing of the run starts but on their Open
            # (a wrist turn of its own that stopped part way ends the run in there: EXIT_LINE_UP_HOLDS)
            went, wrist_turned, wrist_held = line_up_watch(
                node, ctx, cfg, runner, watcher, args, locator, start_tol, ear=ear
            )
            if not went:
                sys.exit(EXIT_NOT_LINED_UP)
        # (a line-up that turned the wrist started from rest, or from rest but for its wrist, and the run's own
        # look goes on from its view)
        if not wrist_turned:
            try:
                homed = home_first(ctx, runner, args.execute, rest_joints(cfg), tol=start_tol)
            except StepOutStopped:
                sys.exit(1)  # said by home_first (STOP ... the arm holds where it stands): it has moved
            if not homed:
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

        # THE SCENE CAMERA FINDS THE BOX. It sees the workspace in front of
        # the chair from where it stands, so the box is found before the arm
        # has moved toward it. The wrist never searches for it (owner,
        # 2026-10-22): no scene fix, and the run stops — unless the wrist
        # survey is on (--wrist-survey, the node's wrist_survey: a chair
        # parked side-on to a counter, which the scene camera cannot see;
        # tasks/survey), which looks for it with the wrist camera then.
        # A box the frame cuts at its side that no look could finish is
        # placed ROUGHLY from the part in view (scene_source.SCENE_ROUGH,
        # owner 2026-11-04): never pressed on as it stands, it says only
        # where the wrist looks, and the wrist's aim from there places the
        # button (SceneFix.rough).
        # FIND THE BOX AND PRESS IT, and again from the find wherever no press plans from where the chair
        # stands (--guide; owner 2026-11-17: "it should NOT just fail EVER"), or no camera sees the
        # container at all (owner 2026-11-19): the engine has the user move the chair (guide_chair), and
        # everything the cameras measured before the move is measured again
        guided = 0  # line-ups in the middle of this run
        while True:
            t_scene = time.monotonic()
            scene_fix = locator.locate(SCENE_LINED_UP_LOCATE_S if wrist_held else SCENE_FIRST_LOCATE_S)
            viz_hook("scene", locator, scene_fix)  # what it judged, drawn off this thread (perception/viz)
            if ctx.mission_frames is not None:
                ctx.mission_frames.scene(
                    scene_frame(locator.grab, locator.base_to_link()),
                    fix_xyz=None if scene_fix is None else scene_fix.top_xyz,
                    score=None if scene_fix is None else scene_fix.score,
                    why=locator.last_why if scene_fix is None else None,
                    source=None if scene_fix is None else getattr(scene_fix, "source", "segmenter"),
                    boxes=getattr(locator, "last_boxes", None),
                )
            if locator.moved_note and scene_fix is not None:
                log.info("[press_demo] scene camera: %s" % locator.moved_note)  # e.g. the tag was not in view
            # the scene camera that checks the box again after the press: none for the wrist's box,
            # which it never saw (a box it found elsewhere would be taken for this one moved)
            reach_locator = locator
            wrist_said, scene_said = [], None
            ctx.wrist_far = None  # (wrist_first says what it saw out of the arm's reach)
            ctx.wrist_seen = ctx.side_seen = None  # (... and each wrist look, what it saw where it found none)
            if getattr(args, "wrist_first", False):
                # THE WRIST CAMERA FINDS THE BOX (--wrist-first; owner, 2026-11-11: "having the wrist
                # cam just find the box ... the arm would just need to angle itself so it can see the
                # scene"): facing a table, the arm presses in front of itself, where the scene camera
                # sees no lid whole and puts what it sees 7-9 cm out (tasks/survey's front note). The
                # scene camera's box is kept for where the wrist finds none. With the orchestrator's
                # selection too (--instance-id, 2026-11-12): the wrist's box is taken where it is tied
                # to the selected container as the scene camera has it (wrist_look, SelectionTie).
                scene_fix, reach_locator, scene_said = wrist_look(
                    node, ctx, cfg, runner, watcher, args, locator, scene_fix, wrist_said
                )
            far = (getattr(ctx, "wrist_far", None) or []) if args.guide else []
            not_seen = [] if args.guide else None  # why no camera has it, where the run does not end on that
            if scene_fix is None and (not far or side_survey_has_a_turn(args)):
                # No camera has the container in front. With --wrist-survey the wrist looks over the arm's
                # side BEFORE anything else is made of it, as it did before there was any guidance: the
                # container may stand beside the chair, and a lid-sized top far off in front is then not
                # it. As first written the guidance below came first, and with both flags on the side
                # survey was never tried (review of 2026-11-17). By hand (no --guide), and where what the
                # find was is nothing the chair's move cures, this is the run's end where the survey finds
                # none (NO BOX); guided, its None comes back here
                scene_fix = wrist_survey_or_exit(
                    node,
                    ctx,
                    cfg,
                    runner,
                    watcher,
                    args,
                    locator,
                    wrist_look="; ".join(wrist_said),
                    scene_said=scene_said,
                    or_none=bool(far),
                    unseen=not_seen,
                )
                reach_locator = None
            if scene_fix is None and far:
                # THE WRIST SEES IT, OUT OF THE ARM'S REACH (--guide), and the side survey, where it has a
                # turn, found no container either: no NO BOX and no end. The chair is moved for it, as for a
                # box no press plans to (guide_chair, which first brings an arm the survey left over its
                # side to rest), and the run looks again
                why = "the wrist camera sees a lid-sized top at [%.2f, %.2f], out of the arm's reach%s" % (
                    far[0][0],
                    far[0][1],
                    "" if not scene_said else " (scene camera: %s)" % scene_said,
                )
                log.info("[press_demo] OUT OF REACH: %s; nothing was pressed" % why)
                runner.note("reach_refused", why=why, guided=True, what=GUIDE_FAR, **locator.look_record())
                guided += 1
                wrist_turned, wrist_held = guide_chair(
                    node, ctx, cfg, runner, watcher, args, locator, start_tol, far[0], why, guided, ear, what=GUIDE_FAR
                )
                continue
            if scene_fix is None:
                # NOTHING THAT COULD BE THE CONTAINER IS IN ANY CAMERA'S VIEW (--guide; no_container_kind): no
                # NO BOX and no end either (owner 2026-11-19, asked whether a box no camera sees at all
                # should become guidance too, "move so the container is in front of the arm": yes). Until
                # then such a run said NO BOX, went home and ended not_found. The chair is moved for it as
                # for a box out of reach: the arm at rest with only its wrist turned to look, the person
                # told that the arm cannot see the container until a camera does, and nothing moves toward
                # a box but on their yes. No place is handed to the engine: none was measured
                why = "no camera sees the container (%s)" % not_seen[-1]
                log.info("[press_demo] NOT SEEN: %s; nothing was pressed" % why)
                runner.note("reach_refused", why=why, guided=True, what=GUIDE_UNSEEN, **locator.look_record())
                guided += 1
                wrist_turned, wrist_held = guide_chair(
                    node, ctx, cfg, runner, watcher, args, locator, start_tol, None, why, guided, ear, what=GUIDE_UNSEEN
                )
                continue
            if not scene_by_tf and reach_locator is not None:
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
                "[press_demo] %s: box at [%.3f, %.3f] yaw %.1f deg from %d lid points%s "
                "(%s; the %s reads the table at %.3f%s, surveyed %s; %.1f s to locate "
                "[%s], %.1f s since start) — reaching out"
                % (
                    "SCENE" if reach_locator is not None else "WRIST SURVEY",
                    sp.xyz[0],
                    sp.xyz[1],
                    math.degrees(sp.yaw),
                    scene_fix.n_points,
                    (
                        " (%s from its lid's side in view — ROUGH: the wrist's aim from the look pose places it)"
                        % scene_fix.partial
                        if getattr(scene_fix, "rough", False)
                        else " (%s, finished from its lid's arc in view)" % scene_fix.partial
                    )
                    if getattr(scene_fix, "partial", "")
                    else "",
                    scene_source_label(scene_fix),
                    "scene" if reach_locator is not None else "wrist camera",
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
                owl_score=round(scene_fix.score, 3),  # the detector's score (the run log's field name since 2026-09)
                source=getattr(scene_fix, "source", "segmenter"),
                n_sightings=int(getattr(scene_fix, "n_sightings", 1)),
                rough=bool(getattr(scene_fix, "rough", False)),
                # the box's height as this run measured it: the lid's top over the table it was read on (the model's
                # dims.z only bounds it, scene_source.lid_band_off)
                box_height_mm=None
                if scene_fix.table_z_scene != scene_fix.table_z_scene
                else round(1000.0 * (float(scene_fix.top_xyz[2]) - float(scene_fix.table_z_scene)), 1),
                table_z_scene=None
                if scene_fix.table_z_scene != scene_fix.table_z_scene
                else round(scene_fix.table_z_scene, 4),
                **locator.look_record(),  # timing, segmenter: the look's seconds and what the segmenter gave it
            )
            # EVERY box the scene camera fixes is pressed by the reach-out: the
            # tilted look, the wrist's aim, the hand turned down onto the button,
            # and — the full run — the lid off and set down (tasks/reach_press,
            # tasks/reach_lid_off). Owner 2026-10-28: "make the arm always go to
            # that 45 degree looking pose even if it's close enough to go
            # straight up and down. completely scrap the home -> straight
            # vertical if close enough mode".
            from box_opening.tasks.reach_press import NO_WAY_IN, run_reach_press

            ctx.cpose = scene_fix.pose  # the box every world and plan from here is built around
            watcher.roi = None  # the scene bbox never applied to the wrist frame
            code = run_reach_press(node, ctx, cfg, runner, watcher, args, scene_fix, reach_locator)
            if code != NO_WAY_IN:
                sys.exit(code)
            if not args.guide:  # (the reach-out answers NO_WAY_IN only to a guided run: a failure, as it was)
                sys.exit(1)
            # NO PRESS PLANS FROM WHERE THE CHAIR STANDS (reach_press.no_way_in, only with --guide): nothing
            # was pressed, and the arm rests but for a wrist the look turned. The chair is moved, and the
            # run goes round again from the find
            guided += 1
            wrist_turned, wrist_held = guide_chair(
                node,
                ctx,
                cfg,
                runner,
                watcher,
                args,
                locator,
                start_tol,
                scene_fix.top_xyz,
                getattr(ctx, "no_way_in", "") or "no press plans from where the chair stands",
                guided,
                ear,
            )
    except KeyboardInterrupt:
        sys.exit(130)  # the abort path already reported what was confirmed
    except ArmHalted as e:
        # the driver halted a goal (the e-stop) or took control away: the run ended where it was,
        # nothing sent after it (REVIEW_RULES C5); the node reports it arm_halted, not recoverable
        log.info(
            "[press_demo] STOP: HALTED: the arm is not the run's to move (%s) — nothing more was sent; "
            "the arm stands where it stopped. Clear the e-stop and the arm, attended, before the next run" % e
        )
        sys.exit(EXIT_HALTED)
    finally:
        ending = sys.exc_info()[1]  # how the run ended, for its record
        # a run unwinding from a Ctrl+C (the engine's cancel: the 130 above)
        ctrl_c = isinstance(ending, KeyboardInterrupt) or isinstance(
            getattr(ending, "__context__", None), KeyboardInterrupt
        )
        try:
            # the console as it was, before the node the viz's publishers
            # live on goes, and the viz's thread stopped (never left inside
            # OpenCV for the interpreter's teardown, a second Ctrl+C or
            # not); a cancelled run publishes nothing more
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
            # is noted, from what was heard (runner.flush), and its frames written,
            # except after a cancel, which must end at once (no frames kept)
            runner.flush()
            runner.restore_contact_profile()  # the driver serves every program on the arm: as it was found
            if not ctrl_c:
                keep_mission_frames(getattr(ctx, "mission_frames", None), ending)


if __name__ == "__main__":
    main()
