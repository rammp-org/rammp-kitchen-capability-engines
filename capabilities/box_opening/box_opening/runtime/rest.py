"""The arm's rest pose (HOME): the robot's own, from its config.

constants.HOME is one robot's: flatbot's, its arm 8 in forward on its chair (2026-11-09). The same arm
on the Gen1.5 chair (rammp-deployments gen_1_5, live since 2026-11-18) stands 33 mm inside that chair's
padded arm-side armrest there (seat/scan_05; offline 2026-11-19 on the live URDF and gen_1_5's layers),
and the planner plans nothing from a start inside its world. So the rest pose is robot config, as the
chair's boxes are (runtime/body): read at a run's start from the file the arm's driver reads its own
named poses from, the robot's arm_presets.yaml (rammp-deployments #158; ROS params, `presets.home`, 7 rad
for arm_joint_1..7, what the driver's go_to_preset `home` flies to), which the node's fragment already
mounts with the rest of the robot's config at /config/robot. One rest pose a robot: where everyone parks
the arm is where a run starts and ends.

`rest_pose` names that file (the node's param, each CLI's --rest-pose; BOX_OPENING_REST_POSE in the
environment overrides it, even empty): by default /config/robot/arm_presets.yaml. '' means the built-in
pose, constants.HOME as it is written. The default file not there, or with no presets.home in it (a
robot config from before #158, a bench with no config mounted): the built-in pose too, and the run says
so. A file NAMED and not there or without the pose, or one whose presets.home is not seven angles, is an
error: the run stops before it moves.

A configured pose is flown to at every run's start and end, so it is judged before anything moves
(refusal): every revolute joint REST_LIMIT_ROOM_RAD inside its limits, every continuous one inside the
planner's bound, and, with the body model on, no sphere inside the chair and user's padding. One that
fails is refused in plain words and nothing moves. The built-in pose is not judged: it is what it was.

The pose in force is written INTO constants.HOME (use), the one list every module that imported HOME
holds. With no pose configured the list is not touched at all.
"""

import math
import os
from pathlib import Path

DEFAULT_REST_POSE = "/config/robot/arm_presets.yaml"
REST_POSE_ENV = "BOX_OPENING_REST_POSE"
REST_POSE_KEY = "presets.home"  # the arm driver's parameter (kinova_gen3_ros2 goto_preset_server.h)
JOINTS = 7


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


def _angles(value, path):
    """`value` as seven finite joint angles, or RestPoseError."""
    ok = isinstance(value, list | tuple) and len(value) == JOINTS
    ok = ok and all(isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v) for v in value)
    if not ok:
        raise RestPoseError(
            "rest pose: %s's %s is not seven joint angles in rad (it reads %r)" % (path, REST_POSE_KEY, value)
        )
    return [float(v) for v in value]


def read(path, named=False):
    """The seven joint angles (rad, arm_joint_1..7) `path` gives as presets.home, or None where the file
    is not there or gives none. The file is ROS params, as the arm's driver is handed it: each top-level
    key a node (`/**` for all), its `ros__parameters` the values, the pose written `presets.home: [...]`
    or nested under `presets:`. RestPoseError where the file cannot be parsed, where two of its nodes give
    different poses (which one the driver flies to depends on its node's name: neither is taken), where
    the pose is not seven finite numbers, and, with `named` (the file was asked for by name, not found at
    the default path), where the file is not there or gives no pose."""
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
        if not isinstance(params, dict):
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
            "rest pose: %s gives more than one %s (for %s): which one the arm's driver flies to depends on its "
            "node's name, so neither is taken" % (path, REST_POSE_KEY, ", ".join(node for node, _q in found))
        )
    return found[0][1]


def limit_room_rad():
    """How far inside its limits each revolute joint of a rest pose must stand: what the plan-free step
    toward a goal asks of that goal (primitives.ClearMarginFree refuses one nearer: the planner plans
    no way to it from a start beside the limit)."""
    from box_opening.primitives.core import CLEAR_MARGIN_ROOM_RAD

    return CLEAR_MARGIN_ROOM_RAD


def refusal(q, chain, check=None):
    """Why the joint state `q` may not be the rest pose, in plain words, or None: judged on the arm
    `chain` (its URDF's limits) and, where one is on, the body model `check` (runtime/body.Body: the
    chair and its user, padded as the planner pads them)."""
    from box_opening.runtime.branches import CONTINUOUS_JOINTS, PLANNER_CONTINUOUS_LIMIT_RAD

    q = [float(v) for v in q]
    room = limit_room_rad()
    names = list(getattr(chain, "joint_names", None) or ["joint_%d" % (j + 1) for j in range(len(q))])
    if len(chain.limits) != len(q):
        return "it has %d joint angles and this arm %d joints" % (len(q), len(chain.limits))
    for j, (lo, hi) in enumerate(chain.limits):
        if lo is None or hi is None:
            continue
        if not lo <= q[j] <= hi:
            return "%s at %.3f rad is past its limit (%.2f to %.2f)" % (names[j], q[j], lo, hi)
        if min(q[j] - lo, hi - q[j]) < room:
            return (
                "%s at %.3f rad stands %.3f rad from its limit (%.2f to %.2f), and the planner plans no way "
                "back to a rest pose nearer than %.2f" % (names[j], q[j], min(q[j] - lo, hi - q[j]), lo, hi, room)
            )
    for j in CONTINUOUS_JOINTS:
        if j < len(q) and abs(q[j]) > PLANNER_CONTINUOUS_LIMIT_RAD:
            return "%s at %.3f rad is past the planner's bound on a joint that turns without end (%.1f)" % (
                names[j],
                q[j],
                PLANNER_CONTINUOUS_LIMIT_RAD,
            )
    hit = None if check is None else check.first_hit([q], step_rad=0.0, start_exempt=False)
    if hit is not None:
        return (
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
    return None


def standing(q, chain, check=None):
    """How the rest pose `q` stands, for the run's log: its least room to a joint limit and, with the
    body model on, how far its nearest sphere stands outside the chair and user's padding."""
    import numpy as np

    rooms = [min(float(q[j]) - lo, hi - float(q[j])) for j, (lo, hi) in enumerate(chain.limits) if lo is not None]
    said = ["every joint %.2f rad or more inside its limits" % min(rooms)] if rooms else []
    if check is not None:
        depth, links = check.sphere_depths(np.atleast_2d(np.asarray(q, float)))
        i = int(np.argmax(depth[0]))
        said.append(
            "%.0f mm outside the chair and user's padding at its nearest (%s)" % (-1000.0 * depth[0][i], links[i])
        )
    else:
        said.append("no body model to hold it out of a chair")
    return ", ".join(said)


def use(q):
    """Put `q` in force as the rest pose: written into constants.HOME, the list every module holds."""
    from box_opening import constants

    q = [float(v) for v in q]
    if len(q) != len(constants.HOME):
        raise RestPoseError("rest pose: %d joint angles, not %d" % (len(q), len(constants.HOME)))
    constants.HOME[:] = q
    return constants.HOME


def configure(param=None, env=None):
    """The rest pose `param` names (resolve), read, judged on this run's arm and body model and put in
    force: (the pose, or None for the built-in one; words for the run's log). The arm's chain and the
    body model are runtime/approach's, so set_chain_root and set_body_layers come first. RestPoseError
    where the pose cannot be read, cannot be judged (no arm kinematics to read the limits from) or is
    refused: the caller stops before anything moves."""
    from box_opening import constants
    from box_opening.runtime import approach

    path = resolve(param, env)
    if not path:
        return None, "rest pose: the built-in one (--rest-pose / %s empty), %s" % (REST_POSE_ENV, _fmt(constants.HOME))
    q = read(path, named=path != DEFAULT_REST_POSE)
    if q is None:
        why = "%s names no %s" % (path, REST_POSE_KEY) if Path(path).is_file() else "no %s" % path
        return None, "rest pose: the built-in one, %s (%s)" % (_fmt(constants.HOME), why)
    ok, why = approach.chain_ready()
    if not ok:
        raise RestPoseError("rest pose: %s of %s cannot be judged: %s" % (REST_POSE_KEY, path, why))
    try:
        chain = approach._chain()
    except RuntimeError as e:  # a description with no chain to the tool, a body model it does not place
        raise RestPoseError("rest pose: %s of %s cannot be judged: %s" % (REST_POSE_KEY, path, e)) from e
    check = getattr(chain, "body", None)
    why = refusal(q, chain, check)
    if why is not None:
        raise RestPoseError(
            "rest pose: %s of %s, %s, is refused: %s. Give the robot a rest pose clear of its chair there (or "
            "name another file: rest_pose / --rest-pose / %s)" % (REST_POSE_KEY, path, _fmt(q), why, REST_POSE_ENV)
        )
    use(q)
    return q, "rest pose: %s of %s, %s: %s" % (REST_POSE_KEY, path, _fmt(q), standing(q, chain, check))


def _fmt(q):
    return "[%s]" % ", ".join("%.4f" % float(v) for v in q)
