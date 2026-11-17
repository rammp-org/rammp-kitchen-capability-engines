"""Every-session preflight (spec §4): the checks a mission needs to pass.

FAIL (nonzero exit): this shell not on Cyclone DDS; /joint_states missing
or without effort fields; the driver's trajectory action or the planner's
actions unreachable; the planner unable to load a world this repo writes
(it runs in a container, so the worlds directory must be mounted into it
at the same path); the body model on (body_layers) and unreadable, or the
planner not planning around its same chair and user (its config and world
"": an empty-world planner would fly the look, the carry and home through
them); no TF for the finger tips, or for the wrist camera's
parent frame; another program that moves this arm on the graph (both of
which press_demo --execute refuses on); no gripper reported; a wrist
camera stream the mission reads silent (colour, camera_info, and the
ALIGNED depth, which only exists with align_depth.enable:=true).
WARN (exit unaffected): no TF for the scene camera where TF places it
(flatbot), a silent scene camera stream (the mission then finds no box and
stops before the arm moves toward one: the wrist no longer searches), or
sheppy's scene segmenter silent (/perception/object_detections: it is what
finds the box).
Report-only: the gripper's position.

Preflight never moves anything: its client is not armed.
"""

import os
import time

import rclpy

from box_opening import LOGGER as log
from box_opening.constants import GRIPPER_STATE_TOPIC
from box_opening.runtime import driver
from box_opening.runtime.client import PlannerClient
from box_opening.runtime.driver import rmw_refusal
from box_opening.tasks import cli_common
from box_opening.worlds import PLANNER_WORLD, WorldStore

STREAM_WINDOW_S = 3.0  # how long every camera and detector topic is listened to


def camera_topics(store=None):
    """(topic, required, fix) for every camera stream the mission reads:
    required ones FAIL preflight when silent, the rest WARN. Topic paths
    come from the camera configs named in the world YAML (OAK-D on flatbot,
    D405+Gemini on the bench)."""
    from box_opening.perception.d405 import camera_config
    from box_opening.perception.scene import SCENE_NS

    # wrist camera topics from the config file the grabber uses
    wcfg_name = store.wrist_camera_config if store is not None else None
    wcfg = camera_config(wcfg_name) if wcfg_name else camera_config()
    wrist_color = wcfg.get("color_topic")
    wrist_info = wcfg.get("color_info_topic")
    wrist_depth = wcfg.get("aligned_depth_topic")
    if wrist_color is None:
        # D405 fallback: derive from depth_topic namespace
        _ns = wcfg.get("depth_topic", "").rsplit("/depth/", 1)
        ns = _ns[0] if len(_ns) == 2 else "/wrist_camera"
        wrist_color = ns + "/color/image_raw"
        wrist_info = ns + "/color/camera_info"
        wrist_depth = ns + "/aligned_depth_to_color/image_raw"

    # scene camera topics from the scene config, or module defaults (Gemini)
    scfg = (store.scene_camera_cfg if store is not None else None) or {}
    sns = scfg.get("ns", SCENE_NS)
    scene_color = sns + scfg.get("color_topic_suffix", "/color/image_raw")
    scene_cinfo = sns + scfg.get("color_info_topic_suffix", "/color/camera_info")
    scene_depth = sns + scfg.get("depth_topic_suffix", "/depth/image_raw")
    scene_dinfo = sns + scfg.get("depth_info_topic_suffix", "/depth/camera_info")

    wrist_fix = "is sheppy's wrist_camera node up and the camera powered?"
    scene_fix = "without it the mission finds no box and stops — is sheppy's scene_camera node up?"
    depth_fix = (
        "the wrist camera must publish aligned depth (D405: align_depth.enable:=true; OAK-D: spatial alignment enabled)"
    )
    return [
        (wrist_color, True, wrist_fix),
        (wrist_info, True, wrist_fix),
        (wrist_depth, True, depth_fix),
        (scene_color, False, scene_fix),
        (scene_cinfo, False, scene_fix),
        (scene_depth, False, scene_fix),
        (scene_dinfo, False, scene_fix),
    ]


def tf_checks(store):
    """(target, source, required, fix) for every TF lookup the mission
    refuses on (required) or loses the scene camera without. The wrist
    frame is the one press_demo.readiness_refusal looks up."""
    from box_opening.perception.d405 import CAMERA_CONFIG, camera_config
    from box_opening.perception.scene import LINK_FRAME

    parent = camera_config(store.wrist_camera_config or CAMERA_CONFIG)["parent_frame"]
    out = [
        (
            store.base_frame,
            parent,
            True,
            "is sheppy's `box_opening` node up (`sheppy up box-opening`)? press_demo --execute refuses without it",
        )
    ]
    if store.external_static_tf:
        # flatbot: the scene camera is placed by TF, not by a calibration
        # (SceneLocator.base_to_link looks up the config's link_frame)
        scfg = store.scene_camera_cfg or {}
        out.append(
            (
                store.base_frame,
                scfg.get("link_frame", LINK_FRAME),
                False,
                "the mission cannot place the scene camera — is sheppy's static_tf node up?",
            )
        )
    return out


def tf_verdicts(checks, ok):
    """[("PASS" | "FAIL" | "WARN", text)] from tf_checks and whether each
    lookup succeeded (ok[i]). Pure."""
    out = []
    for (target, source, required, fix), got in zip(checks, ok, strict=False):
        if got:
            out.append(("PASS", "TF %s -> %s" % (target, source)))
        else:
            out.append(("FAIL" if required else "WARN", "no TF %s -> %s — %s" % (target, source, fix)))
    return out


DRIVER_HEALTH_WAIT_S = 3.0  # the driver's diagnostics come about once a second


def driver_health_verdict(health):
    """(level, text) for what the driver's diagnostics said of its link to the arm and its
    e-stop (driver.driver_health_from; None: nothing heard)."""
    if health is None or not health.get("link_state"):
        return "WARN", (
            "the arm driver's link state not heard on %s in %.0f s — a run will find out at its first goal"
            % (driver.DIAGNOSTICS_TOPIC, DRIVER_HEALTH_WAIT_S)
        )
    why = driver.health_refusal(health)
    if why:
        return "FAIL", why
    return "PASS", "the arm driver's link to the arm is running%s" % (
        "" if "estopped" not in health else ", its e-stop clear"
    )


def other_clients_verdict(others):
    """("PASS" | "FAIL", text) from client.other_arm_clients(). Pure."""
    if others:
        return "FAIL", (
            "another program that moves this arm is running (%s) — `sheppy up box-opening` stops it" % ", ".join(others)
        )
    return "PASS", "no other arm client on the graph"


def stream_verdicts(topics, counts, window_s=STREAM_WINDOW_S):
    """[(topic, "PASS" | "FAIL" | "WARN", text)] from how many messages each
    topic delivered in the window. Pure."""
    out = []
    for topic, required, fix in topics:
        n = int(counts.get(topic, 0))
        if n:
            out.append((topic, "PASS", "%s at %.1f Hz" % (topic, n / float(window_s))))
        else:
            out.append((topic, "FAIL" if required else "WARN", "%s silent for %.0f s — %s" % (topic, window_s, fix)))
    return out


def segmenter_verdict(n, window_s=STREAM_WINDOW_S):
    """("PASS" | "WARN", text) for sheppy's scene segmenter, from how many
    /perception/object_detections messages came in the window (None: this
    install cannot read them). It is what finds the box. Pure."""
    from box_opening.perception.segmenter_source import SEGMENTER_TOPIC

    if n is None:
        return "WARN", "%s cannot be read here (rammp_perception_interfaces does not import)" % SEGMENTER_TOPIC
    if n == 0:
        return "WARN", (
            "%s silent for %.0f s — sheppy's scene segmenter is not running, and nothing else finds the box: "
            "select scene_segmenter: rfdetr in the sheppy profile" % (SEGMENTER_TOPIC, window_s)
        )
    return "PASS", "%s at %.1f Hz (sheppy's scene segmenter)" % (SEGMENTER_TOPIC, n / float(window_s))


def _listen(node, topics, window_s=STREAM_WINDOW_S):
    """{topic: message count} and the scene segmenter's message count (None:
    its messages cannot be read here) over one window."""
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image

    from box_opening.perception.segmenter_source import SEGMENTER_TOPIC

    counts = {t: 0 for t, _r, _f in topics}
    seg = [0]
    subs = []

    def counter(topic):
        def cb(_msg):
            counts[topic] += 1

        return cb

    for topic, _r, _f in topics:
        kind = CameraInfo if topic.endswith("camera_info") else Image
        subs.append(node.create_subscription(kind, topic, counter(topic), qos_profile_sensor_data))
    try:
        from rammp_perception_interfaces.msg import ObjectDetections
    except ImportError:
        seg = None
    else:

        def seg_cb(_msg):
            seg[0] += 1

        subs.append(node.create_subscription(ObjectDetections, SEGMENTER_TOPIC, seg_cb, qos_profile_sensor_data))
    t0 = time.monotonic()
    while time.monotonic() - t0 < window_s:
        rclpy.spin_once(node, timeout_sec=0.05)
    for sub in subs:
        node.destroy_subscription(sub)
    return counts, None if seg is None else seg[0]


def _gripper_state(node, timeout_s=3.0):
    """One /gripper_state message, or None."""
    from rammp_arm_interfaces.msg import GripperState
    from rclpy.qos import qos_profile_sensor_data

    got = []
    sub = node.create_subscription(GripperState, GRIPPER_STATE_TOPIC, got.append, qos_profile_sensor_data)
    t0 = time.monotonic()
    while not got and time.monotonic() - t0 < timeout_s:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    return got[0] if got else None


def planner_world(node, timeout_s=3.0, with_config=False):
    """The planner node's `world` parameter (rammp_curobo's own name for the
    world it was started with), or None when it does not answer; with
    `with_config`, (world, config): its `config` too (a robot's own
    planner.yaml, rammp-deployments #146, names its world itself: `world`
    is then "")."""
    from rcl_interfaces.srv import GetParameters

    cli = node.create_client(GetParameters, "/rammp_curobo/get_parameters")
    got = (None, None)
    try:
        if cli.wait_for_service(timeout_sec=timeout_s):
            fut = cli.call_async(GetParameters.Request(names=["world", "config"]))
            t_end = time.monotonic() + timeout_s
            while not fut.done() and time.monotonic() < t_end:
                rclpy.spin_once(node, timeout_sec=0.05)
            if fut.done() and fut.result() is not None and fut.result().values:
                values = [v.string_value or None for v in fut.result().values]
                got = (values[0], values[1] if len(values) > 1 else None)
    finally:
        node.destroy_client(cli)
    return got if with_config else got[0]


def planner_body_verdict(world, config, body_source):
    """(level, text) for whether the planner plans around the same chair and user box_opening holds
    its own lines out of (its body model, read from `body_source`, a cuRobo planner config): PASS
    only where the planner was started on that same config with world "" (the config composes its
    layers into every world: flatbot's /config/robot/curobo/planner.yaml, both containers mounting
    config/robots/current at /config/robot). FAIL otherwise — an empty-world planner
    (world_empty.yaml, planner_empty_world.yaml) plans the look, the carry and home straight through
    the chair and the user while box_opening's own lines are held out of them, and a planner that
    does not say is not trusted to. Pure."""
    want = os.path.normpath(str(body_source))
    if world is None and config is None:
        return "FAIL", (
            "the planner did not say what it plans around (/rammp_curobo/get_parameters did not answer), and "
            "the body model is on (%s): its moves must avoid the same chair and user" % want
        )
    if not world and config and os.path.normpath(str(config)) == want:
        return "PASS", 'the planner plans around the body model\'s chair and user (%s, world "")' % config
    return "FAIL", (
        "the body model is on (%s) but the planner plans in %s: box_opening holds its own lines out of the "
        "chair and user, the planner's moves (the look, the carry, home) would not be — start the planner on "
        'config %s with world "" (profiles/box-opening.yaml), or turn the body check off (body_layers / '
        '--body-layers "")'
        % (want, ("world %s" % world) if world else ("config %s" % (config or "an unknown config")), want)
    )


def planner_world_verdict(world, config, body_source=None):
    """(level, text) for the world the planner was started with: box opening sends it none, so it
    plans in that one. With the body model on (`body_source`, its planner config), planner_body_verdict:
    only that same config with world "" passes. Off: PASS world_empty.yaml (the arm alone), or a
    robot's own planner config that names its world (world "": flatbot's, its MEBot chair, seat and
    occupant in every world); WARN otherwise: said, not failed, the deployment owns it."""
    if body_source:
        return planner_body_verdict(world, config, body_source)
    if world == PLANNER_WORLD:
        return "PASS", "planner started with %s" % world
    if not world and config and str(config).startswith("/config/robot/"):
        return "PASS", "planner started with its robot's own config (%s), the world it names" % config
    return "WARN", (
        "planner started with %s, not %s — box opening plans in it as it is (the deployment's "
        "planner `world` parameter sets it)" % (world or "an unknown world", PLANNER_WORLD)
    )


def body_source_verdict(body_layers):
    """(level, text, source): the body model (runtime/body.resolve of `body_layers`, the CLI's
    --body-layers) — PASS off (source None) or loaded (source its planner config), FAIL where it
    cannot be read (source None: a run would stop there before it moves)."""
    from box_opening.runtime import body

    path = body.resolve(body_layers)
    if not path:
        return (
            "PASS",
            "body check off (--body-layers / %s empty): nothing holds the arm out of a chair" % (body.BODY_LAYERS_ENV),
            None,
        )
    try:
        model = body.load(path)
    except body.BodyModelError as e:
        return "FAIL", "%s — a run stops there before it moves" % e, None
    layers = ", ".join(sorted({n.split("/", 1)[0] for n in model.names}))
    return "PASS", "body model %s: %d boxes (%s)" % (path, len(model.names), layers), path


def main():
    args = cli_common.make_parser(__doc__).parse_args()
    why = rmw_refusal(os.environ)
    if why:
        log.info("FAIL  " + why)
        raise SystemExit(1)

    rclpy.init()
    node = rclpy.create_node("box_opening_preflight")
    client = PlannerClient(node)  # not armed: nothing here can move
    failures = 0

    # 1. /joint_states fresh, with efforts (guarded primitives require them)
    t0 = time.monotonic()
    try:
        client.joints()
        fresh = time.monotonic() - t0
        efforts = client.efforts_present()
        log.info("PASS  /joint_states fresh (%.1f s) — efforts %s" % (fresh, "present" if efforts else "MISSING"))
        if not efforts:
            log.info("FAIL  effort fields absent — guarded primitives will refuse")
            failures += 1
    except SystemExit as exc:
        log.info("FAIL  %s" % exc)
        raise SystemExit(1) from exc

    # 2. the driver's trajectory action
    if client.driver_reachable(timeout_s=5.0):
        log.info("PASS  driver /execute_joint_trajectory reachable")
    else:
        log.info("FAIL  /execute_joint_trajectory unreachable — is sheppy's `arm` node up?")
        failures += 1

    # 2b. the driver's link to the arm, and its e-stop, as its diagnostics say (a driver up with
    #     its link faulted refuses every goal: flatbot 2026-11-04 12:23)
    level, text = driver_health_verdict(client.driver_health(timeout_s=DRIVER_HEALTH_WAIT_S))
    log.info("%-4s  %s" % (level, text))
    failures += level == "FAIL"

    # 3. the planner's actions
    if client.planner_reachable(timeout_s=5.0):
        log.info("PASS  planner actions reachable")
    else:
        log.info("FAIL  planner actions unreachable — is sheppy's `planner` node up?")
        failures += 1

    # 4. the world the planner plans in: the one its deployment started it
    #    with (box opening sends none, worlds.py). With the body model on (the
    #    chair and user box_opening's own lines are held out of), the planner
    #    must plan around the same ones (FAIL otherwise: an empty-world planner
    #    flies the look, the carry and home through them); off, world_empty.yaml
    #    or the robot's own config, said, not failed: the deployment owns it.
    store = WorldStore(args.bench_world or cli_common.default_bench_yaml())
    client._base_frame = store.base_frame
    level, text, source = body_source_verdict(getattr(args, "body_layers", None))
    log.info("%-4s  %s" % (level, text))
    failures += level == "FAIL"
    level, text = planner_world_verdict(*planner_world(node, with_config=True), body_source=source)
    log.info("%-4s  %s" % (level, text))
    failures += level == "FAIL"

    # 5. TF for the finger tips (robot_state_publisher from the launch)
    tips = client.contact_xyz(timeout_s=3.0)
    if tips is not None:
        log.info("PASS  TF %s -> %s" % (store.base_frame, ", ".join(client._tip_frames)))
    else:
        log.info(
            "FAIL  no TF for the finger tips — run press_demo.launch.py " "(tf:=true starts robot_state_publisher)"
        )
        failures += 1

    # 5b. TF for the wrist camera (press_demo --execute refuses without it)
    #     and, where TF places it, the scene camera
    checks = tf_checks(store)
    ok = [client.tf_ready(target, source, timeout_s=3.0) for target, source, _r, _f in checks]
    for level, text in tf_verdicts(checks, ok):
        log.info("%-4s  %s" % (level, text))
        failures += level == "FAIL"

    # 5c. no other program that moves this arm (press_demo --execute refuses)
    level, text = other_clients_verdict(client.other_arm_clients())
    log.info("%-4s  %s" % (level, text))
    failures += level == "FAIL"

    # 6. the gripper
    state = _gripper_state(node)
    if state is None:
        log.info("FAIL  no %s — is sheppy's `arm` node up?" % GRIPPER_STATE_TOPIC)
        failures += 1
    elif not state.present:
        log.info("FAIL  the driver reports no gripper")
        failures += 1
    else:
        log.info("PASS  gripper present, at %.3f (0 open .. 1 closed)" % state.position)

    # 7. the camera streams the mission reads, and the scene segmenter that
    #    finds the box (a mission used to discover a missing stream mid-run: 2026-09-22)
    topics = camera_topics(store)
    counts, n_seg = _listen(node, topics)
    for _topic, level, text in stream_verdicts(topics, counts):
        log.info("%-4s  %s" % (level, text))
        failures += level == "FAIL"
    level, text = segmenter_verdict(n_seg)
    log.info("%-4s  %s" % (level, text))

    raise SystemExit(1 if failures else 0)
