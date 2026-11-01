"""Every-session preflight (spec §4): the checks a mission needs to pass.

FAIL (nonzero exit): this shell not on Cyclone DDS; /joint_states missing
or without effort fields; the driver's trajectory action or the planner's
actions unreachable; the planner unable to load a world this repo writes
(it runs in a container, so the worlds directory must be mounted into it
at the same path); no TF for the finger tips, or for the wrist camera's
parent frame; another program that moves this arm on the graph (both of
which press_demo --execute refuses on); no gripper reported; a wrist
camera stream the mission reads silent (colour, camera_info, and the
ALIGNED depth, which only exists with align_depth.enable:=true).
WARN (exit unaffected): no TF for the scene camera where TF places it
(flatbot), a silent scene camera stream (the mission then finds no box and
stops before the arm moves toward one: the wrist no longer searches), an
owl_detector not running, or running but deaf (no frame reaching it).
Report-only: the gripper's position.

Preflight never moves anything: its client is not armed.
"""

import os
import time

import rclpy

from box_opening import LOGGER as log
from box_opening.constants import GRIPPER_STATE_TOPIC
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


def owl_verdict(scores, camera):
    """("PASS" | "WARN", text) for one owl_detector from the score slots of
    the messages it sent in the window: none means it is not running; only
    BLIND heartbeats mean it is up but no camera frame reaches it — the
    "deaf" state a restart has always cured (2026-09-16, -17, -22). Pure."""
    from box_opening.perception.owl_source import BLIND

    if not scores:
        return "WARN", (
            "owl_detector (%s camera) not running — just restarted? it takes ~9 s to load; run "
            "preflight again. Otherwise: is sheppy's box_opening node up? (the mission falls back "
            "to the cloud rung, then plain depth)" % camera
        )
    if all(s <= BLIND for s in scores):
        return "WARN", (
            "owl_detector (%s camera) is up but has had no %s frame — `sheppy restart box_opening` "
            "(the camera itself is checked above)" % (camera, camera)
        )
    return "PASS", "owl_detector (%s camera) up and receiving frames" % camera


def _listen(node, topics, window_s=STREAM_WINDOW_S):
    """{topic: message count} and {camera: [owl scores]} over one window."""
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import Float32MultiArray

    from box_opening.perception.owl_source import topics_for

    counts = {t: 0 for t, _r, _f in topics}
    owl = {"wrist": [], "scene": []}
    subs = []

    def counter(topic):
        def cb(_msg):
            counts[topic] += 1

        return cb

    for topic, _r, _f in topics:
        kind = CameraInfo if topic.endswith("camera_info") else Image
        subs.append(node.create_subscription(kind, topic, counter(topic), qos_profile_sensor_data))
    for cam in owl:
        subs.append(
            node.create_subscription(
                Float32MultiArray,
                topics_for(cam)[0],
                lambda m, cam=cam: owl[cam].append(float(m.data[4])) if len(m.data) >= 5 else None,
                1,
            )
        )
    t0 = time.monotonic()
    while time.monotonic() - t0 < window_s:
        rclpy.spin_once(node, timeout_sec=0.05)
    for sub in subs:
        node.destroy_subscription(sub)
    return counts, owl


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


def planner_world(node, timeout_s=3.0):
    """The planner node's `world` parameter (rammp_curobo's own name for the
    world it was started with), or None when it does not answer."""
    from rcl_interfaces.srv import GetParameters

    cli = node.create_client(GetParameters, "/rammp_curobo/get_parameters")
    try:
        if not cli.wait_for_service(timeout_sec=timeout_s):
            return None
        fut = cli.call_async(GetParameters.Request(names=["world"]))
        t_end = time.monotonic() + timeout_s
        while not fut.done() and time.monotonic() < t_end:
            rclpy.spin_once(node, timeout_sec=0.05)
        if not fut.done() or fut.result() is None or not fut.result().values:
            return None
        return fut.result().values[0].string_value or None
    finally:
        node.destroy_client(cli)


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

    # 3. the planner's actions
    if client.planner_reachable(timeout_s=5.0):
        log.info("PASS  planner actions reachable")
    else:
        log.info("FAIL  planner actions unreachable — is sheppy's `planner` node up?")
        failures += 1

    # 4. the world the planner plans in: the one its deployment started it
    #    with (box opening sends none, worlds.py) — world_empty.yaml, the arm
    #    and nothing else. Said, not failed: the deployment owns it.
    store = WorldStore(args.bench_world or cli_common.default_bench_yaml())
    client._base_frame = store.base_frame
    world = planner_world(node)
    if world == PLANNER_WORLD:
        log.info("PASS  planner started with %s" % world)
    else:
        log.info(
            "WARN  planner started with %s, not %s — box opening plans in it as it is (the deployment's "
            "planner `world` parameter sets it)" % (world or "an unknown world", PLANNER_WORLD)
        )

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

    # 7. the camera streams the mission reads, and the two OWL detectors
    #    (a mission used to discover a missing stream mid-run: 2026-09-22)
    topics = camera_topics(store)
    counts, owl = _listen(node, topics)
    for _topic, level, text in stream_verdicts(topics, counts):
        log.info("%-4s  %s" % (level, text))
        failures += level == "FAIL"
    for cam in ("wrist", "scene"):
        level, text = owl_verdict(owl[cam], cam)
        log.info("%-4s  %s" % (level, text))

    raise SystemExit(1 if failures else 0)
