"""Every-session preflight (spec §4): the checks a mission needs to pass.

FAIL (nonzero exit): this shell not on Cyclone DDS; /joint_states missing
or without effort fields; the driver's trajectory action or the planner's
actions unreachable; the planner unable to load a world this repo writes
(it runs in a container, so the worlds directory must be mounted into it
at the same path); the body model on (body_layers) and unreadable, or the
planner not planning around its same chair and user (its config and world
"": an empty-world planner would fly the look, the carry and home through
them); the robot's rest pose (rest_pose: the presets.home of the file it
names) unreadable, inside or too near the chair and user's padding,
touching itself or too near a joint limit; the planner planning from a start
the mounted chair model puts inside the chair, or refusing the rest pose
(its canary: plans only); a camera's or the arm's mount in TF that is not the
mounted robot config's; no TF for the finger tips, or for the wrist camera's
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


def rest_pose_rows(rest_pose, urdf_path, base_frame, body_layers, expect=None):
    """[(level, text)]: the rest pose a run would start and end at (runtime/rest.configure of `rest_pose`,
    the CLI's --rest-pose, BOX_OPENING_REST_POSE over it), judged as a run judges it: on this arm
    (`urdf_path`, its description rooted at `base_frame`; None on the bench: the installed one) and the
    body model of `body_layers` ('' off). PASS with where the pose comes from, its numbers and how it
    stands, and a WARN row for each thing a run says as a warning (a pose inside the planner's activation
    distance of the padding; the built-in pose taken where a robot's config is mounted); FAIL where a run
    would stop on it (a pose inside or too near the chair and user's padding, touching itself or too near
    a joint limit, a file that cannot be read)."""
    from box_opening.runtime import approach, rest

    approach.set_chain_root(base_frame, urdf_path=urdf_path)
    try:
        approach.set_body_layers(body_layers)
        got = rest.configure(rest_pose, expect=expect)
    except RuntimeError as e:  # rest.RestPoseError, body.BodyModelError: a run stops on either
        return [("FAIL", "%s — a run stops there before it moves" % e)]
    return [("PASS", got.words)] + [("WARN", text) for text in got.warnings]


# The TF tree's fixed mounts against the mounted robot config's own (transforms/cameras.yaml, the camera
# mounts, and transforms/world.yaml, where the arm's mount stands on the chair): a mount is the same where its
# place is within this and its turn within that. As the files stood on snorlax on 2026-11-19, flatbot's and
# gen_1_5's cameras.yaml held the SAME two mounts (the arm and its cameras moved to the new chair together),
# so the cameras alone do not tell those two robots apart; their base_link -> arm_mount differ by 87 mm, which
# is why world.yaml is compared too
MOUNT_M = 0.002
MOUNT_DEG = 0.5
MOUNT_FILES = ("cameras.yaml", "world.yaml")


def robot_config_dir(body_source=None):
    """The mounted robot config's folder: the one the body model's planner config sits in (its curobo/'s
    parent), else the rest pose's default folder (/config/robot)."""
    from pathlib import Path

    from box_opening.runtime import rest

    return Path(body_source).parent.parent if body_source else Path(rest.DEFAULT_REST_POSE).parent


def config_mounts(path):
    """[(parent, child, xyz, quat_xyzw)] of a robot config's transforms file (rammp-calibration's mounts
    format: `mounts: ros__parameters: transforms: [names]` and one mapping a name), or None where the file is
    not there or is not in that format."""
    import yaml

    try:
        with open(path) as f:
            params = (yaml.safe_load(f) or {})["mounts"]["ros__parameters"]
        out = []
        for name in params["transforms"]:
            m = params[name]
            out.append(
                (str(m["parent"]), str(m["child"]), [float(v) for v in m["xyz"]], [float(v) for v in m["quat_xyzw"]])
            )
        return out
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
        return None


def mount_rows(lookup, path):
    """[(level, text)]: each fixed mount of the mounted robot config (`path`: its transforms/cameras.yaml,
    its transforms/world.yaml) against the TF tree (`lookup(parent, child)` -> (xyz, quat_xyzw) or None:
    client.tf_lookup).

    What it proves: the TF tree carries the mounts THIS config names, so the stack that publishes TF was
    started on a config whose mounts are these. The planner's and box opening's own files are read through
    one mount (config/robots/current), but the static TF is published by another container, started when
    the stack was: one started for another robot (2026-11-19 11:50: a teammate's stack on flatbot's config
    ran beside gen_1_5's files) publishes that robot's mounts, and every place a camera gives is then off
    by the difference. FAIL where a mount in TF is not the file's (MOUNT_M, MOUNT_DEG); WARN where the
    file is not there or TF has no such frame (the frames themselves are checked apart).
    What it does not prove: that a mount is RIGHT (the file and TF may both be wrong alike), nor which
    robot the stack was started for where two robots' files hold the same mounts (see above)."""
    import numpy as np

    mounts = config_mounts(path)
    if not mounts:
        return [("WARN", "no mounts to compare TF with (%s is not there or names none)" % path)]
    rows = []
    for parent, child, xyz, quat in mounts:
        got = lookup(parent, child)
        if got is None:
            rows.append(("WARN", "no TF %s -> %s to compare with %s" % (parent, child, path)))
            continue
        off_m = float(np.linalg.norm(np.subtract(got[0], xyz)))
        a, b = np.asarray(got[1], float), np.asarray(quat, float)
        dot = abs(float(a @ b)) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)
        off_deg = float(np.degrees(2.0 * np.arccos(min(1.0, dot))))
        if off_m <= MOUNT_M and off_deg <= MOUNT_DEG:
            rows.append(("PASS", "TF %s -> %s is the mounted robot config's mount (%s)" % (parent, child, path)))
        else:
            rows.append(
                (
                    "FAIL",
                    "TF %s -> %s is NOT the mounted robot config's mount: %.0f mm and %.1f deg from %s — the "
                    "stack publishing TF was started on another robot's config (or the file changed since it "
                    "was): every place the cameras give is off by that"
                    % (parent, child, 1000.0 * off_m, off_deg, path),
                )
            )
    return rows


# The planner's canary. box opening knows the planner holds the chair only by the PATH of its config, and
# that path (/config/robot/curobo/planner.yaml) is the same string whatever robot `current` pointed at when
# the planner was started. 2026-11-19: the rest pose chosen for the Gen1.5 chair reads 9 mm clear on
# flatbot's world, so a planner still holding flatbot's chair would take it as a start and plan around the
# wrong armrest. So the planner is asked two plans, nothing flown: from a start the mounted chair model
# puts deep inside one of its chair boxes, which a planner holding THIS chair refuses as a start in its
# world; and from the rest pose, which it must take.
CANARY_IN_BOX_M = 0.020  # a sphere's surface this far inside the UNPADDED chair box: no padding argues it
CANARY_PERSON_M = 0.030  # ... its spheres this far from the user's bare boxes: outside their padding too
CANARY_SELF_M = 0.020  # ... of itself
CANARY_LIMIT_RAD = 0.25  # ... and every joint this far inside its limits: refused for the chair alone
CANARY_SPAN_RAD = 1.2  # looked for this far from the rest pose on joints 1-4, the wrist as it rests
CANARY_DRAWS = 4000
CANARY_SEED = 20261119
CANARIES = 2  # in different boxes where the model gives them
CANARY_TURN_RAD = 0.2  # the plan asked from the rest pose: its last joint turned this much
# cuRobo's word for a start that stands in its world, as rammp-curobo passes it on (its other refusals of a
# start, for the arm touching itself or a joint limit, begin alike and prove nothing about the chair)
CANARY_REFUSAL = "INVALID_START_STATE_WORLD_COLLISION"


def canary_starts(check, chain, rest_q, stats=None):
    """[(joints, box, depth_m)]: up to CANARIES joint states the mounted chair model (`check`, a Body on
    `chain`) puts deep inside a box of the CHAIR itself (not the user's: CANARY_IN_BOX_M inside the unpadded
    box) and nothing else wrong with them: clear of the user's boxes, of itself, inside the joint limits,
    no lower than the rest pose stands (so not in the floor, which every robot's world has). Taken from the
    model, never a constant: drawn (seeded: the same every run) around the rest pose `rest_q`, the nearest
    to it first, in different boxes. `stats` (a dict): filled with how many draws each rule left."""
    import numpy as np

    from box_opening.runtime.body import _bare

    nu = getattr(check.model, "near_user", None)
    user = set(nu.layers) if nu is not None else {"person"}

    rng = np.random.default_rng(CANARY_SEED)
    rest_q = np.asarray(rest_q, float)
    Q = np.tile(rest_q, (CANARY_DRAWS, 1))
    Q[:, :4] += rng.uniform(-CANARY_SPAN_RAD, CANARY_SPAN_RAD, size=(CANARY_DRAWS, 4))
    ok = np.ones(len(Q), bool)
    for j, (lo, hi) in enumerate(chain.limits):
        if lo is not None and hi is not None:
            ok &= (Q[:, j] - lo >= CANARY_LIMIT_RAD) & (hi - Q[:, j] >= CANARY_LIMIT_RAD)
    Q = Q[ok]
    stats = {} if stats is None else stats
    stats.update(draws=CANARY_DRAWS, in_limits=len(Q), in_chair=0, clear_of_user=0, clear_of_itself=0, high=0)
    chair = [
        lay
        for lay in check.layers
        if lay not in user and any(check.box_own[i] for i, x in enumerate(check.box_layers) if x == lay)
    ]
    if not len(Q) or not chair:
        return []
    near = check.distance(Q, layers=chair)
    ok = np.asarray(near.m) <= -CANARY_IN_BOX_M
    # the sphere that stands in the box is the robot file's own on the upper arm or the forearm (the planner
    # has it too: box opening's added hand spheres are not what refuses the start)
    ok &= np.array([_bare(str(lk)).startswith(("half_arm", "forearm")) for lk in near.link])
    stats["in_chair"] = int(ok.sum())
    theirs = [lay for lay in check.layers if lay in user]
    if theirs:
        ok &= np.asarray(check.distance(Q, layers=theirs).m) >= CANARY_PERSON_M
    stats["clear_of_user"] = int(ok.sum())
    gap = check.self_gap(Q) if hasattr(check, "self_gap") else None
    if gap is not None:
        ok &= np.asarray(gap.m) >= CANARY_SELF_M
    stats["clear_of_itself"] = int(ok.sum())
    C0, r0, _links = check.spheres_at(rest_q)
    floor = float((C0[:, 2] - r0).min())
    low = np.array([float((check.spheres_at(q)[0][:, 2] - r0).min()) for q in Q[ok]]) if ok.any() else np.zeros(0)
    idx = np.flatnonzero(ok)[low >= floor - 0.05]
    stats["high"] = len(idx)
    order = idx[np.argsort(np.abs(Q[idx] - rest_q).max(axis=1))]
    out, seen = [], set()
    for i in order:
        if near.box[i] in seen:
            continue
        seen.add(near.box[i])
        out.append(([float(v) for v in Q[i]], near.box[i], -float(near.m[i])))
        if len(out) >= CANARIES:
            break
    return out


def planner_canary_rows(plan, check, chain, rest_q):
    """[(level, text)]: whether the planner holds the chair the mounted model describes, asked by plans
    alone (`plan(goal_joints, start_joints)` -> a result with .success and .message, or None: the client's
    plan_to_joints; nothing is flown, the client is not armed).

    What it proves: (1) from the rest pose the planner plans (a turn of the last joint): the rest pose is a
    start it takes; (2) from each canary start (canary_starts: deep inside a box of the mounted chair,
    nothing else wrong) it refuses, as a start in its world: it holds SOMETHING there. A planner that plans
    from a canary does not hold this chair, whatever its config's path says: FAIL (an empty world, or
    another robot's chair with nothing at that place). Refused for another reason, or no canary in the
    model: WARN (it proves nothing either way).
    What it does not prove: that the planner holds this chair and no other (another chair with a box at
    the canary's place refuses it too: the test is one-way), nor that the chair's boxes are where the real
    chair is. NEVER RUN against the real planner as of 2026-11-19 (stand-ins only, test_rest_pose.py): its
    first attended preflight is the first time the planner's own words for these refusals are seen."""
    rows = []
    turned = [float(v) for v in rest_q]
    turned[-1] += CANARY_TURN_RAD
    res = plan(turned, [float(v) for v in rest_q])
    if res is None:
        return [("FAIL", "the planner did not answer a plan from the rest pose — the canary was not asked")]
    if getattr(res, "success", False):
        rows.append(("PASS", "the planner takes the rest pose as a start (a plan from it, not flown)"))
    else:
        rows.append(
            (
                "FAIL",
                "the planner REFUSES a plan from the rest pose (%s): no run starts there on this planner"
                % getattr(res, "message", "no reason given"),
            )
        )
    stats = {}
    starts = canary_starts(check, chain, rest_q, stats)
    if not starts:
        return rows + [
            (
                "WARN",
                "no canary start in the mounted chair model near the rest pose: not asked (of %d draws %d inside "
                "the joint limits, %d of them with the upper arm or the forearm deep in a chair box, %d of those "
                "clear of the user's boxes, %d clear of itself, %d no lower than the rest pose)"
                % tuple(
                    stats.get(k, 0)
                    for k in ("draws", "in_limits", "in_chair", "clear_of_user", "clear_of_itself", "high")
                ),
            )
        ]
    for q, box, depth in starts:
        what = "a start the mounted chair model puts %.0f mm inside %s" % (1000.0 * depth, box)
        res = plan([float(v) for v in rest_q], q)
        message = "no answer" if res is None else str(getattr(res, "message", ""))
        if res is not None and getattr(res, "success", False):
            rows.append(
                (
                    "FAIL",
                    "the planner PLANS from %s: it does not hold this robot's chair (another robot's world, or "
                    "none), whatever its config's path says. No run on this planner" % what,
                )
            )
        elif res is not None and CANARY_REFUSAL in message:
            rows.append(("PASS", "the planner refuses %s, as a start in its world: it holds a box there" % what))
        else:
            rows.append(
                ("WARN", "the planner refused %s, but not for its world (%s): it proves nothing" % (what, message))
            )
    return rows


def _tally(rows):
    """Say each (level, text) row and count the FAILs."""
    for level, text in rows:
        log.info("%-4s  %s" % (level, text))
    return sum(level == "FAIL" for level, _text in rows)


def main():
    args = cli_common.make_parser(__doc__).parse_args()
    try:  # first: which rest pose this process reads (a file it cannot read is a FAIL row further down)
        from box_opening.runtime import rest

        log.info("      %s" % rest.said(getattr(args, "rest_pose", None)))
    except RuntimeError:
        pass
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
    planner_up = client.planner_reachable(timeout_s=5.0)
    if planner_up:
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

    # 4b. the robot's rest pose, where a run starts and ends (runtime/rest: presets.home of the file
    #     rest_pose names): one a run would refuse ends every run before it moves, so it is said here first
    rows = rest_pose_rows(
        getattr(args, "rest_pose", None),
        cli_common._arm_urdf_path(node, store.base_frame),
        store.base_frame,
        source or "",
        expect=getattr(args, "rest_pose_digest", None),
    )
    failures += _tally(rows)
    rest_ok = rows[0][0] == "PASS"

    # 4c. the planner's canary (plans only, nothing flown): does the planner hold the chair the mounted
    #     model describes? Its config's path says so for any robot's chair alike (planner_canary_rows)
    from box_opening.runtime import approach

    check = approach.body_check()
    if check is not None and rest_ok and planner_up:
        from box_opening.constants import HOME

        failures += _tally(planner_canary_rows(client.plan_to_joints, check, approach._chain(), list(HOME)))

    # 4d. the TF tree's fixed mounts against the mounted robot config's own (mount_rows): a stack started
    #     on another robot's config publishes that robot's
    for name in MOUNT_FILES:
        failures += _tally(
            mount_rows(
                lambda parent, child: client.tf_lookup(parent, child, timeout_s=3.0),
                robot_config_dir(source) / "transforms" / name,
            )
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
