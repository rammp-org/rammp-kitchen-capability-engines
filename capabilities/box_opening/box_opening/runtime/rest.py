"""The arm's rest pose (HOME): the robot's own, from its config.

constants.HOME is one robot's: flatbot's, its arm 8 in forward on its chair (2026-11-09). The same arm
on the Gen1.5 chair (rammp-deployments gen_1_5, live since 2026-11-18) stands 33 mm inside that chair's
padded arm-side armrest there (seat/scan_05; offline 2026-11-19 on the live URDF and gen_1_5's layers),
and the planner plans nothing from a start inside its world. So the rest pose is robot config, as the
chair's boxes are (runtime/body): read at a run's start from a file in the format the arm's driver reads
its own named poses from, the robot's arm_presets.yaml (rammp-deployments #158; ROS params,
`presets.home`, 7 rad for arm_joint_1..7, what the driver's go_to_preset `home` flies to), which the
node's fragment already mounts with the rest of the robot's config at /config/robot.

`rest_pose` names the file (the node's param, each CLI's --rest-pose): by default
/config/robot/arm_presets.yaml. BOX_OPENING_REST_POSE in the environment BEATS the param and the flag,
even empty: one line in a deployment's containers gives every box opening process there (the node, its
runs, home_arm, preflight) the same file, whatever each was started with. '' means the built-in pose,
constants.HOME as it is written. The default file not there, or with no presets.home in it: the built-in
pose too, and where a robot's config is mounted that is said as a WARNING (resting at another robot's
pose is not what anyone asked for). A file NAMED and not there or without the pose, or one whose
presets.home is not seven real numbers, is an error: the run stops before it moves.

Whatever pose a run will rest at, the built-in one included, is JUDGED before anything moves wherever a
body model holds the arm out of a chair (refusal), and refused in plain words where it fails:
  - every revolute joint constants.REST_TOL_RAD further inside its limits than the plan-free step toward a
    goal asks of that goal (core.CLEAR_MARGIN_ROOM_RAD): a hold may stand that far off its rest pose, and
    must still be a start the planner takes;
  - every continuous joint inside the planner's bound;
  - no sphere nearer the chair and user's padding than core.ESCAPE_NEAR_M, the distance inside which every
    way home begins with a step clear (a pose nearer would start every run with one), and the arm
    REST_SELF_M clear of itself by the planner's own self-collision rule (Body.self_gap, the robot file's).
A pose that passes but stands inside the planner's collision_activation_distance of the padding (30 mm)
is said as a WARNING. With no body model a configured pose is judged on its joints alone, and the words
say that it was NOT judged against any chair; the built-in pose is then left as it always was. A
configured pose with no arm kinematics to judge it by (no description heard) is refused.

The pose in force is written INTO constants.HOME (use), the one list every module that imported HOME
holds. With no pose configured the list is left as it stands. A process the node starts is handed the
node's own reading too (digest, --rest-pose-digest): where its own differs, it stops, so two processes of
one deployment never rest at two poses unseen.
"""

import hashlib
import math
import os
from collections import namedtuple
from pathlib import Path

DEFAULT_REST_POSE = "/config/robot/arm_presets.yaml"
REST_POSE_ENV = "BOX_OPENING_REST_POSE"
REST_POSE_KEY = "presets.home"  # the arm driver's parameter (kinova_gen3_ros2 goto_preset_server.h)
# the node blocks of a ROS params file that reach the arm's driver: every node's, and the driver's own
# (rammp-deployments' manifest: ros_node_name kinova_gen3_node). Another node's block is not its pose
DRIVER_NODE = "kinova_gen3_node"
JOINTS = 7
# clear of itself by this much, by the planner's rule (the built-in pose and the Gen1.5 chair's pose C both stand
# 27.6 mm clear by gen_1_5's robot file, the wrist against the gripper's base)
REST_SELF_M = 0.010

# What configure answers: `pose` the configured joints (None: the built-in pose is in force), `numbers` the
# seven in force either way, `source` the file they were read from (None: built in), `words` for the run's
# log, `warnings` to say louder
Rest = namedtuple("Rest", "pose numbers source words warnings")
_CONFIGURED = False  # use() has written a configured pose into constants.HOME (and built_in() not taken it back)


class RestPoseError(RuntimeError):
    """The rest pose named cannot be read, cannot be judged, or is refused: nothing may move."""


def resolve(param=None, env=None):
    """The file to read the rest pose from: BOX_OPENING_REST_POSE where it is set (even empty), else
    `param` (the node's rest_pose, a CLI's --rest-pose), else DEFAULT_REST_POSE. '' means the built-in
    pose."""
    env = os.environ if env is None else env
    if REST_POSE_ENV in env:
        return str(env[REST_POSE_ENV]).strip()
    return DEFAULT_REST_POSE if param is None else str(param).strip()


def digest(q):
    """A short word for seven joint angles, the same in every process that reads the same numbers (to
    0.1 mrad): what the node hands the runs it starts, beside the file's path."""
    return hashlib.sha256(",".join("%.4f" % float(v) for v in q).encode()).hexdigest()[:12]


def _angles(value, path):
    """`value` as seven finite joint angles, or RestPoseError. Every one must be written as a real number:
    ROS refuses a list that mixes integers and reals (and reads an all-integer one as integers), so the
    arm's driver does not start on such a file, and a pose only box opening could read is not the robot's."""
    ok = isinstance(value, list | tuple) and len(value) == JOINTS
    ok = ok and all(isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v) for v in value)
    if not ok:
        raise RestPoseError(
            "rest pose: %s's %s is not seven joint angles in rad (it reads %r)" % (path, REST_POSE_KEY, value)
        )
    if any(isinstance(v, int) for v in value):
        raise RestPoseError(
            "rest pose: %s's %s has a whole number among its angles (%r): write 0.0, not 0 — ROS refuses a list "
            "that mixes the two, and the arm's driver does not start on it" % (path, REST_POSE_KEY, value)
        )
    return [float(v) for v in value]


def _reaches_driver(node):
    """Whether the params-file block `node` is one the arm's driver takes: every node's (`/**`), or one
    that names the driver's node (its last name, under any namespace or wildcard)."""
    node = str(node).strip()
    return node in ("/**", "**") or node.rstrip("/").rsplit("/", 1)[-1] == DRIVER_NODE


def read(path, named=False):
    """The seven joint angles (rad, arm_joint_1..7) `path` gives as presets.home, or None where the file
    is not there or gives none. The file is ROS params, as the arm's driver is handed it: each top-level
    key a node (`/**` for all), its `ros__parameters` the values, the pose written `presets.home: [...]`
    or nested under `presets:`. Only the blocks that reach the driver count (_reaches_driver).
    RestPoseError where the file cannot be parsed, where two such blocks give different poses (which one
    the driver flies to is not for this reader to guess: neither is taken), where the pose is not seven
    real numbers, and, with `named` (the file was asked for by name, not found at the default path), where
    the file is not there or gives no pose."""
    import yaml

    path = Path(path)
    if not path.is_file():
        if named:
            raise RestPoseError("rest pose: cannot read %s (no such file)" % path)
        return None
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, ValueError, yaml.YAMLError) as e:
        raise RestPoseError("rest pose: cannot read %s (%s)" % (path, e)) from e
    found = []
    for node, block in data.items() if isinstance(data, dict) else ():
        params = block.get("ros__parameters") if isinstance(block, dict) else None
        if not isinstance(params, dict) or not _reaches_driver(node):
            continue
        group, name = REST_POSE_KEY.split(".", 1)
        nested = params.get(group)
        for value in (params.get(REST_POSE_KEY), nested.get(name) if isinstance(nested, dict) else None):
            if value is not None:
                found.append((str(node), _angles(value, path)))
    if not found:
        if named:
            raise RestPoseError("rest pose: %s names no %s" % (path, REST_POSE_KEY))
        return None
    if any(q != found[0][1] for _node, q in found):
        raise RestPoseError(
            "rest pose: %s gives more than one %s (for %s): which one the arm's driver flies to is not for box "
            "opening to guess, so neither is taken" % (path, REST_POSE_KEY, ", ".join(node for node, _q in found))
        )
    return found[0][1]


def chosen(param=None, env=None):
    """What a process would rest at, read and not yet judged: (numbers, source, why). `source` the file
    the pose was read from, or None for the built-in one (constants.HOME as it stands, or as it is written
    where an earlier call had put a configured one in force), `why` then saying why it is the built-in
    one. RestPoseError where a file is named and unreadable, or its pose malformed."""
    from box_opening import constants

    path = resolve(param, env)
    built = list(constants.HOME_BUILT_IN) if _CONFIGURED else list(constants.HOME)
    if not path:
        return built, None, "--rest-pose / %s empty" % REST_POSE_ENV
    q = read(path, named=path != DEFAULT_REST_POSE)
    if q is None:
        why = "%s names no %s" % (path, REST_POSE_KEY) if Path(path).is_file() else "no %s" % path
        return built, None, why
    return q, path, None


def said(param=None, env=None):
    """One line for a process's first words: which numbers it rests at and which file they are from, read
    and not yet judged (chosen), with their digest: two box opening processes that disagree show it here."""
    numbers, source, why = chosen(param, env)
    return "rest pose: %s, %s (digest %s)" % (_fmt(numbers), _of(source, why), digest(numbers))


def limit_room_rad():
    """How far inside its limits each revolute joint of a rest pose must stand: what the plan-free step
    toward a goal asks of that goal (primitives.ClearMarginFree refuses one nearer: the planner plans no way
    to it from a start beside the limit), and constants.REST_TOL_RAD more, the distance a hold may stand off
    its rest pose and still be where a run starts. 2026-11-19: gen_1_5's own presets.home passed the first
    alone by 0.0006 rad."""
    from box_opening.constants import REST_TOL_RAD
    from box_opening.primitives.core import CLEAR_MARGIN_ROOM_RAD

    return CLEAR_MARGIN_ROOM_RAD + REST_TOL_RAD


def clear_m():
    """How far outside the chair and user's padding a rest pose must stand: core.ESCAPE_NEAR_M, the distance
    inside which every way home begins with a step clear of the padding (home_any). A rest pose nearer would
    be stepped away from at the start of every run and come back to at its end. flatbot's flown pose stands
    11 mm outside its own chair's."""
    from box_opening.primitives.core import ESCAPE_NEAR_M

    return ESCAPE_NEAR_M


def refusal(q, chain, check=None):
    """Why the joint state `q` may not be the rest pose, in plain words, or None: judged on the arm
    `chain` (its URDF's limits) and, where one is on, the body model `check` (runtime/body.Body: the
    chair and its user, padded as the planner pads them, and the planner's own self-collision rule).
    Everything that is wrong with it is said, the chair first (2026-11-19: gen_1_5's own presets.home
    stands 58 mm inside the padding AND 0.181 rad from a joint limit, and the limit alone was said)."""
    import numpy as np

    from box_opening.constants import REST_TOL_RAD
    from box_opening.runtime.branches import CONTINUOUS_JOINTS, PLANNER_CONTINUOUS_LIMIT_RAD

    q = [float(v) for v in q]
    room = limit_room_rad()
    names = list(getattr(chain, "joint_names", None) or ["joint_%d" % (j + 1) for j in range(len(q))])
    if len(chain.limits) != len(q):
        return "it has %d joint angles and this arm %d joints" % (len(q), len(chain.limits))
    why = []
    hit = check.first_hit([q], step_rad=0.0, start_exempt=False) if check is not None else None
    if hit is not None:
        why.append(
            "the arm would rest inside the chair and user's padding there: %s %.0f mm into %s (padded %.0f mm, "
            "the arm's spheres %.0f mm), and the planner plans nothing from a start in it"
            % (
                hit.link,
                1000.0 * hit.depth_m,
                hit.box,
                1000.0 * (check.model.padding_m if hit.padding_m is None else hit.padding_m),
                1000.0 * check.model.buffer_m,
            )
        )
    elif check is not None:
        depth, links = check.sphere_depths(np.atleast_2d(np.asarray(q, float)))
        i = int(np.argmax(depth[0]))
        if -float(depth[0][i]) < clear_m():
            why.append(
                "the arm would rest %.0f mm from the chair and user's padding there (%s): nearer than the %.0f mm "
                "inside which every way home begins with a step clear of it"
                % (-1000.0 * float(depth[0][i]), links[i], 1000.0 * clear_m())
            )
    gap = check.self_gap(q) if check is not None and hasattr(check, "self_gap") else None
    if gap is not None and gap.m < REST_SELF_M:
        why.append(
            "the arm would %s there by the planner's own rule: %s and %s %s (a rest pose keeps %.0f mm), and the "
            "planner plans nothing from a start that touches itself"
            % (
                "touch itself" if gap.m < 0.0 else "stand too near itself",
                gap.link,
                gap.box,
                "%.0f mm into each other" % (-1000.0 * gap.m) if gap.m < 0.0 else "%.0f mm apart" % (1000.0 * gap.m),
                1000.0 * REST_SELF_M,
            )
        )
    for j, (lo, hi) in enumerate(chain.limits):
        if lo is None or hi is None:
            continue
        if not lo <= q[j] <= hi:
            why.append("%s at %.3f rad is past its limit (%.2f to %.2f)" % (names[j], q[j], lo, hi))
        elif min(q[j] - lo, hi - q[j]) < room:
            why.append(
                "%s at %.3f rad stands %.3f rad from its limit (%.2f to %.2f): a rest pose keeps %.2f, the %.2f "
                "the planner needs of a start and the %.2f a hold may stand off its pose"
                % (names[j], q[j], min(q[j] - lo, hi - q[j]), lo, hi, room, room - REST_TOL_RAD, REST_TOL_RAD)
            )
    for j in CONTINUOUS_JOINTS:
        if j < len(q) and abs(q[j]) > PLANNER_CONTINUOUS_LIMIT_RAD:
            why.append(
                "%s at %.3f rad is past the planner's bound on a joint that turns without end (%.1f)"
                % (names[j], q[j], PLANNER_CONTINUOUS_LIMIT_RAD)
            )
    return "; and ".join(why) or None


def standing(q, chain, check=None):
    """(words, warnings) of how the rest pose `q` stands, for the run's log: its least room to a joint
    limit; with the body model on, how far its nearest sphere stands outside the chair and user's padding
    and how clear it is of itself, and a warning where that is inside the planner's activation distance;
    with none, that it was NOT judged against any chair."""
    import numpy as np

    rooms = [min(float(q[j]) - lo, hi - float(q[j])) for j, (lo, hi) in enumerate(chain.limits) if lo is not None]
    words, warnings = (["every joint %.2f rad or more inside its limits" % min(rooms)] if rooms else []), []
    if check is None:
        words.append("NOT judged against any chair, nor against itself (no body model: --body-layers empty)")
        return ", ".join(words), warnings
    depth, links = check.sphere_depths(np.atleast_2d(np.asarray(q, float)))
    i = int(np.argmax(depth[0]))
    out = -float(depth[0][i])
    words.append("%.0f mm outside the chair and user's padding at its nearest (%s)" % (1000.0 * out, links[i]))
    gap = check.self_gap(q) if hasattr(check, "self_gap") else None
    words.append(
        "not judged against itself (the robot file names no self-collision exemptions)"
        if gap is None
        else "%.0f mm clear of itself (%s, %s)" % (1000.0 * gap.m, gap.link, gap.box)
    )
    near = float(getattr(check.model, "activation_m", 0.0) or 0.0)
    if out < near:
        warnings.append(
            "the rest pose stands %.0f mm outside the chair and user's padding (%s): inside the %.0f mm at which "
            "the planner starts pushing a path away, so plans from it start under that push, and a hold that "
            "sags stands nearer still" % (1000.0 * out, links[i], 1000.0 * near)
        )
    return ", ".join(words), warnings


def use(q):
    """Put `q` in force as the rest pose: written into constants.HOME, the list every module holds."""
    global _CONFIGURED
    from box_opening import constants

    q = [float(v) for v in q]
    if len(q) != len(constants.HOME):
        raise RestPoseError("rest pose: %d joint angles, not %d" % (len(q), len(constants.HOME)))
    constants.HOME[:] = q
    _CONFIGURED = True
    return constants.HOME


def built_in():
    """Take a configured pose back out of force: constants.HOME as it is written (HOME_BUILT_IN). Only where
    use() put one in: a list nothing configured is left as it stands."""
    global _CONFIGURED
    from box_opening import constants

    if _CONFIGURED:
        constants.HOME[:] = constants.HOME_BUILT_IN
        _CONFIGURED = False
    return constants.HOME


def configure(param=None, env=None, expect=None):
    """The rest pose `param` names (resolve), read, judged on this run's arm and body model and put in
    force: a Rest. The arm's chain and the body model are runtime/approach's, so set_chain_root and
    set_body_layers come first. `expect`: the digest of the pose the process that started this one read
    (the node's, --rest-pose-digest): another here is refused. RestPoseError where the pose cannot be
    read, cannot be judged (a configured one, with no arm kinematics to read the limits from), is not the
    one expected, or is refused: the caller stops before anything moves."""
    from box_opening.runtime import approach

    path = resolve(param, env)
    numbers, source, why_built_in = chosen(param, env)
    what = "%s, %s" % (_of(source, why_built_in), _fmt(numbers))
    if expect and digest(numbers) != str(expect):
        raise RestPoseError(
            "rest pose: this process reads %s (digest %s) and the node that started it read another (digest %s): "
            "the file changed in between, or the two read different files" % (what, digest(numbers), expect)
        )
    warnings = []
    if source is None and path == DEFAULT_REST_POSE and Path(DEFAULT_REST_POSE).parent.is_dir():
        warnings.append(
            "a robot's config is mounted at %s but gives no rest pose (%s): resting at box opening's BUILT-IN "
            "pose %s, which was made for another robot (flatbot). Name this robot's own: rest_pose / --rest-pose "
            "/ %s" % (Path(DEFAULT_REST_POSE).parent, why_built_in, _fmt(numbers), REST_POSE_ENV)
        )
    ok, why = approach.chain_ready()
    chain = None
    if ok:
        try:
            chain = approach._chain()
        except Exception as e:  # noqa: BLE001 - a description with no chain, a description package not there
            why = "%s: %s" % (type(e).__name__, e)
    if chain is None:
        if source is not None:
            raise RestPoseError("rest pose: %s cannot be judged: %s" % (what, why))
        built_in()
        # the built-in pose where nothing is known of the arm: as it always was, and said
        return Rest(None, numbers, None, "rest pose: %s, NOT judged (no arm kinematics: %s)" % (what, why), warnings)
    check = getattr(chain, "body", None)
    # the built-in pose is judged wherever a body model holds the arm out of a chair; with none it is left
    # as it was (a bench: its joints were never refused)
    refused = refusal(numbers, chain, check) if (source is not None or check is not None) else None
    if refused is not None:
        raise RestPoseError(
            "rest pose: %s, is refused: %s. Give the robot a rest pose it can start from there (or name another "
            "file: rest_pose / --rest-pose / %s)" % (what, refused, REST_POSE_ENV)
        )
    if source is not None:
        use(numbers)
    else:
        built_in()
    words, more = standing(numbers, chain, check)
    pose = numbers if source is not None else None
    return Rest(pose, numbers, source, "rest pose: %s: %s" % (what, words), warnings + more)


def _of(source, why_built_in):
    return "the built-in one (%s)" % why_built_in if source is None else "%s of %s" % (REST_POSE_KEY, source)


def _fmt(q):
    return "[%s]" % ", ".join("%.4f" % float(v) for v in q)
