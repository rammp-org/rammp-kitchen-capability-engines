"""Shared CLI plumbing: args, node/client/runner wiring, safety refusals.

Dry-run is the default for every CLI. --execute arms the client — the
only software gate on motion, since the kinova-gen3-ros2 driver executes
whatever it is sent — and requires a measured container config
(measure_me: false). There is no typed confirmation (owner, 2026-09-16):
the human on the physical e-stop is the gate. Every CLI refuses to start
off Cyclone DDS, the middleware sheppy's driver and planner containers speak.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import rclpy
from rclpy.signals import SignalHandlerOptions

from box_opening import LOGGER as log
from box_opening.constants import state_dir
from box_opening.models.container import ContainerModel
from box_opening.primitives.core import Ctx
from box_opening.runtime import approach
from box_opening.runtime.abort import AbortFlag, install_sigint
from box_opening.runtime.body import BODY_LAYERS_ENV, DEFAULT_BODY_LAYERS
from box_opening.runtime.client import PlannerClient
from box_opening.runtime.driver import rmw_refusal
from box_opening.runtime.runner import Runner
from box_opening.worlds import WorldStore


def _share_path(*parts):
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory("box_opening"), *parts)


def default_container_yaml():
    """The container every CLI falls back to:
    the round clear canister with the pink push-button lid (owner,
    2026-09-22: "this box is the new default"). The OXOs stay selectable
    with --container."""
    return str(_share_path("config", "containers", "ankou_pink.yaml"))


def default_bench_yaml():
    return str(_share_path("config", "world_bench.yaml"))


def make_parser(desc):
    # allow_abbrev=False: "--exec" must not arm a run that scripts/
    # sheppy_task.sh's drift gate took for a dry run (review 2026-10-22)
    ap = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="after previewing, execute: arms arm AND gripper motion on the "
        "kinova-gen3-ros2 driver (without it nothing is ever sent to it)",
    )
    ap.add_argument(
        "--container",
        default=None,
        help="container config YAML (default: installed ankou_pink.yaml, the pink canister; "
        "oxo_pop.yaml / oxo_pop_small.yaml are the OXOs)",
    )
    ap.add_argument(
        "--bench-world",
        default=None,
        help="bench world YAML (default: installed world_bench.yaml)",
    )
    ap.add_argument(
        "--body-layers",
        default=None,
        help="the cuRobo planner config whose layers (the chair and its user) and robot spheres every "
        "plan-free line is held out of (default %s; BOX_OPENING_BODY_LAYERS overrides; '' turns the "
        "check off: the bench). Unreadable: nothing moves" % DEFAULT_BODY_LAYERS,
    )
    ap.add_argument(
        "--speed-scale",
        type=float,
        default=1.0,
        help="whole-run slow mode: dilate every motion by 1/x (0 < x <= 1), "
        "guarded strokes included — a first attempt at a new placement",
    )
    return ap


def refuse_unmeasured(model, execute, measuring=False):
    """An unmeasured container (measure_me: true — geometry estimated, not
    taped) may not be flown at. `measuring` is the observation that
    measures it (press_demo --wrist-detect: the look pose and home, in the
    bench world above the keep-out band, never near the container), which
    the flag must not lock out (2026-09-22: it did)."""
    if execute and model.measure_me and not measuring:
        sys.exit(
            "container config still carries measure_me: true — run the "
            "measurement worksheet (docs/HARDWARE_BRINGUP.md) and flip it "
            "before any hardware execution (dry-run is fine)."
        )


def init_runtime(execute):
    """rclpy + SIGINT ownership + node + client, shared by every CLI.

    `execute` (the --execute flag) arms the client; nothing else does.
    A shell off Cyclone is refused before rclpy is touched: Fast DDS
    discovers the driver and planner containers and then loses their data,
    which reads as a stalled arm instead of a wiring error.

    Owning SIGINT means an in-flight stroke gets its cancel delivered on
    a live context before we exit (runtime/abort.py; proven by
    scripts/abort_e2e.py — rclpy's default handler makes it a race)."""
    why = rmw_refusal(os.environ)
    if why:
        sys.exit(why)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    abort = AbortFlag()
    install_sigint(abort)
    # not "box_opening": that is the engine node, which runs this command line as its child
    node = rclpy.create_node("box_opening_run")
    return node, PlannerClient(node, abort=abort, motion_enabled=bool(execute))


def arm_urdf_file():
    """Where the robot's own URDF is kept for this run: one fixed file,
    overwritten each start (a temp file per run left 44 in /tmp in a day)."""
    return state_dir() / "arm_robot_description.urdf"


def _arm_urdf_path(node, base_frame):
    """Return a path to the arm's URDF, or None for the bench default.

    On flatbot the nightly arm_state_publisher renders the URDF with arm_mount
    as root (incorporating the adapter plate tilt) and publishes it latched on
    /arm/robot_description. The FK chain must use THAT URDF so its root matches
    the planning frame. On the bench the installed kinova_gen3.urdf is correct.
    Off the bench None means there are no kinematics at all (the bench URDF
    has no arm_mount): runtime.approach refuses every chain, so a plan-free
    line or FK refuses instead of being solved in the wrong frame."""
    if base_frame == "base_link":
        return None
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    latched = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    got = []
    sub = node.create_subscription(String, "/arm/robot_description", got.append, latched)
    t0 = time.monotonic()
    while not got and time.monotonic() - t0 < 5.0:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    if not got:
        node.get_logger().error(
            "/arm/robot_description not received in 5 s — no arm kinematics rooted at %s: "
            "plan-free lines and FK are unavailable this run" % base_frame
        )
        return None
    out = arm_urdf_file()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(got[0].data)
    os.replace(tmp, out)
    return str(out)


def ap_prog():
    return os.path.basename(sys.argv[0] or "box_opening")


def load_body_or_exit(args, who):
    """The chair and user every plan-free line is held out of
    (approach.set_body_layers: --body-layers, BOX_OPENING_BODY_LAYERS over
    it), loaded before anything moves: a file that cannot be read or placed
    stops the run here, never a run without the check."""
    try:
        model = approach.set_body_layers(getattr(args, "body_layers", None))
    except RuntimeError as e:
        sys.exit(
            "%s not starting: %s — the plan-free lines cannot be checked against the chair and user. "
            "Nothing moved." % (who, e)
        )
    nu = getattr(model, "near_user", None)
    lacking = [] if nu is None else sorted(set(nu.layers) - {n.split("/", 1)[0] for n in model.names})
    if lacking:
        # the speed near the user would cap nothing: as unreadable as a missing file
        sys.exit(
            "%s not starting: body model: %s has no layer %s, which %s's near_user slows the arm near — "
            "the arm's speed near the user cannot be held. Nothing moved."
            % (who, model.source, ", ".join(lacking), model.hand_source)
        )
    if model is None:
        log.info(
            "%s body check OFF (--body-layers / %s empty): plan-free lines are not checked against "
            "the chair and user" % (who, BODY_LAYERS_ENV)
        )
    else:
        check = getattr(approach._CHAIN, "body", None)  # placed on this run's URDF (set_body_layers)
        spheres = check.spheres if check is not None else model.spheres
        log.info(
            "%s body check: %d boxes (%s), %d spheres on %d links, padded %.0f + %.0f mm; the hand's own "
            "(%s): %s"
            % (
                who,
                len(model.names),
                model.source,
                sum(len(r) for _c, r in spheres.values()),
                len(spheres),
                1000 * model.padding_m,
                1000 * model.buffer_m,
                model.hand_source or "none",
                (
                    "added %s, already in the robot file %s"
                    % (", ".join(check.hand_added) or "none", ", ".join(check.hand_skipped) or "none")
                    if check is not None
                    else "%d groups, placed with the chain" % len(model.hand or [])
                ),
            )
        )
        chain = approach._CHAIN
        if check is not None and getattr(chain, "ee_named_by", "").startswith("the planner's robot file"):
            from box_opening.runtime.kinematics import TOOL_FRAME_LINK

            if chain.ee_link not in (TOOL_FRAME_LINK, "arm_" + TOOL_FRAME_LINK):
                # said only where it is not the description's usual tool frame (a run's hand line has that one)
                log.info(
                    "%s the planner plans %s (its robot file), (%s) mm from box opening's tool frame: poses go to "
                    "it there" % (who, chain.ee_link, ", ".join("%+.1f" % (1000 * v) for v in chain.ee_in_tool[1]))
                )
        if nu is not None:
            log.info(
                "%s near the user (%s): a sphere within %.0f mm moves at most %.2f m/s there, falling to %.2f "
                "m/s at %.0f mm and closer"
                % (
                    who,
                    ", ".join(nu.layers),
                    1000 * nu.far_m,
                    nu.far_mps,
                    nu.far_mps * nu.near_fraction,
                    1000 * nu.near_m,
                )
            )
    return model


def config_mismatch(chain, config_dir):
    """Why the robot config in `config_dir` (rammp-deployments config/robots/<robot>, as mounted at
    /config/robot) is not the config of the arm `chain` was built from, or None: its
    curobo/arm_kinematics.json, the reference the planner itself holds the published URDF against, has a
    joint of the same name with another parent, child or origin (a config with the sensor puck on a
    description without it, or the other way round: the tool frame's joint 14 mm apart). None too where
    there is no such file to compare, or nothing the two share: then nothing is known, and nothing said."""
    import json

    try:
        ref = json.loads((Path(config_dir) / "curobo" / "arm_kinematics.json").read_text())
        mine = {j[0]: (j[2], child, j[3]) for child, j in chain._by_child.items()}  # name -> parent, child, xyz
    except Exception:  # noqa: BLE001 - no reference here, or a stand-in chain: nothing to compare
        return None
    for name in sorted(set(ref) & set(mine)):
        parent, child, xyz = mine[name]
        r = ref[name]
        if str(r.get("parent")) != parent or str(r.get("child")) != child:
            return "joint %s joins %s -> %s in the arm's description, %s -> %s in the config" % (
                name,
                parent,
                child,
                r.get("parent"),
                r.get("child"),
            )
        rx = [float(v) for v in (r.get("xyz") or (0.0, 0.0, 0.0))]
        if max(abs(a - b) for a, b in zip(xyz, rx, strict=True)) > 1e-6:
            return "joint %s is at (%s) in the arm's description, (%s) in the config" % (
                name,
                ", ".join("%g" % v for v in xyz),
                ", ".join("%g" % v for v in rx),
            )
    return None


def hand_or_exit(who, camera_cfg_name=None):
    """Where this run has its hand, said before anything moves, and the run stopped there where the arm's
    description gives it no hand to plan with.

    STOPS (SystemExit, "Nothing moved"): the description's tool frame, which the planner plans, is not
    fixed to the flange (kinematics.ToolFrameError): no pose could be handed to the planner nor a plan's
    end judged. With the body check off nothing else would have built the chain before the first motion.

    SAYS, one line: box opening's tool frame and the point its heights go by (the same on every
    description: where they were flown), where this hand's pads really end
    (constants.PADS_FROM_FLANGE_M), the link the planner plans and how far along the hand it is from the
    tool frame, the wrist lens with the file and the robot it was read from, and how far that stands
    from the one figure no calibration made. And WARNS, a line each: a lens further from that figure
    than the depth press centres on (d405.LENS_REF_XY_TOL_M), and a mounted robot config that is not
    this arm's description (config_mismatch). A run on a description or a robot config it was not flown
    on shows here. Returns the lines; with no chain to describe (no description heard) says nothing."""
    import math

    import numpy as np

    from box_opening.constants import FINGERS_PAST_MODEL_M, HEIGHTS_POINT_FROM_FLANGE_M, PADS_FROM_FLANGE_M
    from box_opening.runtime.kinematics import TOOL_FRAME_LINK, ToolFrameError

    if not approach.chain_ready()[0]:
        return []
    try:
        chain = approach._chain()
    except ToolFrameError as e:
        sys.exit("%s not starting: %s. Nothing moved." % (who, e))
    except Exception as e:  # noqa: BLE001 - a line of the log; whatever else is wrong stops where it always did
        log.info("%s hand: not described (%s)" % (who, e))
        return []
    lines = []
    try:
        if chain.ee_link is None:
            planner = "no link %s in the arm's description: the planner's frame is taken for the tool frame" % (
                TOOL_FRAME_LINK
            )
        elif chain.ee_is_tool:
            planner = "the planner plans %s (%s), the same frame" % (chain.ee_link, chain.ee_named_by)
        else:
            Re, te = chain.ee_in_tool
            turn = math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(Re)) - 1.0) / 2.0))))
            planner = "the planner plans %s (%s), (%s) mm from the tool frame%s: a pose goes to it there" % (
                chain.ee_link,
                chain.ee_named_by,
                ", ".join("%+.1f" % (1000 * v) for v in te),
                "" if turn < 1e-6 else " and turned %.1f deg" % turn,
            )
        line = (
            "%s hand: tool frame %.0f mm out of the flange, the point the heights go by %.0f mm out, as "
            "flown, whatever the description; this hand's pads end %.0f mm out, %.0f mm past it; %s"
            % (
                who,
                1000 * chain.tool_from_flange_m,
                1000 * HEIGHTS_POINT_FROM_FLANGE_M,
                1000 * PADS_FROM_FLANGE_M,
                1000 * FINGERS_PAST_MODEL_M,
                planner,
            )
        )
        warn = []
        if camera_cfg_name:
            from box_opening.perception import d405

            cfg = d405.camera_config(camera_cfg_name)
            lens, _optical, where = d405.wrist_camera_in_hand(cfg, chain=chain)
            if lens is not None:
                line += "; wrist lens (%s) mm from the tool frame (%s)" % (
                    ", ".join("%+.1f" % (1000 * v) for v in lens),
                    where,
                )
            mount = d405.robot_mount(cfg.get("robot_mount")) if cfg.get("robot_mount") else None
            off = d405.lens_off_reference(cfg, lens) if mount is not None else None
            if off is not None:
                line += ", %.1f mm aside and %.1f mm along the tool from the 2026-10-22 fit" % (
                    1000 * off[0],
                    1000 * off[1],
                )
                if off[0] > d405.LENS_REF_XY_TOL_M or off[1] > d405.LENS_REF_Z_TOL_M:
                    warn.append(
                        "%s WARNING: the wrist camera's mount stands further from the 2026-10-22 fit than %.0f mm "
                        "aside / %.0f mm along the tool: the calibration of 2026-11-09, that far from it, pressed "
                        "a centimetre beside the button. Watch where the first press lands"
                        % (who, 1000 * d405.LENS_REF_XY_TOL_M, 1000 * d405.LENS_REF_Z_TOL_M)
                    )
            if mount is not None:
                config_dir = Path(mount.path).parent.parent
                why = config_mismatch(chain, config_dir)
                if why is not None:
                    warn.append(
                        "%s WARNING: the robot config %s (robot %s) is NOT this arm's description: %s. Its camera "
                        "mounts, its chair and its user may be another robot's"
                        % (who, config_dir, mount.robot or "not named", why)
                    )
        lines = [line, *warn]
    except Exception as e:  # noqa: BLE001 - a line of the log, never a stop
        lines = ["%s hand: not described (%s)" % (who, e)]
    for text in lines:
        log.info(text)
    return lines


def planner_body_or_exit(node, model, execute, who):
    """With the body model on, a run that moves the arm starts only where the planner plans around
    the same chair and user (preflight.planner_body_verdict: its config the body model's, world
    ""). An empty-world planner would fly the look, the carry and home through them while the
    plan-free lines are held out: not starting, saying why. Without --execute, said only."""
    if model is None:
        return
    from box_opening.tasks.preflight import planner_body_verdict, planner_world

    try:
        world, config = planner_world(node, with_config=True)
    except Exception as e:  # noqa: BLE001 - no answer is an answer: refused below
        log.info("%s the planner's parameters could not be read (%s)" % (who, e))
        world, config = None, None
    level, text = planner_body_verdict(world, config, model.source)
    if level == "PASS":
        log.info("%s %s" % (who, text))
        return
    if execute:
        sys.exit("%s not starting: %s. Nothing moved." % (who, text))
    log.info("%s WARN %s (nothing moves this run)" % (who, text))


def build_ctx(args):
    """Ctx + Runner for a CLI that has not detected a container (cpose
    None): only pre-detection (bench-world) legs can be planned from it."""
    cfg = args.container or default_container_yaml()
    bench = args.bench_world or default_bench_yaml()
    model = ContainerModel.load(cfg)
    refuse_unmeasured(model, args.execute)
    node, client = init_runtime(args.execute)
    worlds = WorldStore(bench)
    client._base_frame = worlds.base_frame
    # the arm's chain in the frame it plans in, as press_demo — no exit
    # when the description is missing: HOME is a joint goal and needs no
    # chain; only a plan-free lift refuses (LiftFree)
    approach.set_chain_root(worlds.base_frame, urdf_path=_arm_urdf_path(node, worlds.base_frame))
    # the body model, never in `model` (the container's: the Ctx below plans with it)
    body_model = load_body_or_exit(args, "[%s]" % ap_prog())
    hand_or_exit("[%s]" % ap_prog(), worlds.wrist_camera_config)
    planner_body_or_exit(node, body_model, bool(getattr(args, "execute", False)), "[%s]" % ap_prog())
    ctx = Ctx(
        model=model,
        cpose=None,
        client=client,
        worlds=worlds,
        config_path=cfg,
    )
    return ctx, Runner(client)


def apply_speed_scale(runner, args):
    scale = float(getattr(args, "speed_scale", 1.0))
    if not 0.0 < scale <= 1.0:
        sys.exit("--speed-scale must be in (0, 1]")
    runner.time_scale = scale
    if scale < 1.0:
        log.info("[runner] SLOW MODE: every motion dilated by 1/%.2f" % scale)


def run_task(args, build_legs):
    ctx, runner = build_ctx(args)
    apply_speed_scale(runner, args)
    try:
        legs = build_legs(ctx)
    except RuntimeError as e:
        # a refused plan is an honest line, never a traceback — the arm
        # holds where it is
        sys.exit("plan refused — arm holds: %s" % e)
    try:
        results = runner.run(legs, execute=args.execute)
    except KeyboardInterrupt:
        sys.exit(130)  # the abort path already reported what was confirmed
    bad = [r for r in results if not r.ok]
    if bad:
        sys.exit(1)
