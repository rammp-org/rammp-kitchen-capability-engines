"""The run's own proof that a press plans, made away from the arm: the word behind every cell of the
line-up's boundary (reach_zone.Zone) and behind "the run plans for the box where it stands".

    reach_probe --urdf <arm URDF> --lid-z 0.315                      the boundary for a lid at that height
    reach_probe --urdf <arm URDF> --at 0.65 -0.10 0.315              one place: does a press plan there

Nothing here talks to the arm, the planner or the cameras, and nothing moves. press_plans() asks what a
run asks before it moves (tasks/reach_press._reach_press, the proof before the look): out_of_reach, then
plan_reach from HOME with the depth press's way down, then the way in from the look it planned
(solve_way_in, or plan_reach again from the look where its lines do not fly), on the robot's own URDF
and with the chair and user's model the run is held out of (--body-layers). The one thing a run has that
this has not is the planner: the look's plan is DryClient's stand-in, the arm's own IK and a straight
joint line, refused where a joint stands in the planner's limit margin. So a place this proves is one the
run's kinematics reach, not a promise the planner's path there exists; the run's own plan is the last word.

A proof takes about a second on the robot's computer (0.8 s where it refuses, 2 s where it plans: the
reach simulations of 2026-11-11, 1363 places), so the boundary is filled a cell at a time, the box's own
place first (Zone.order), and kept on disk for the next run at the same height (zone_path): what a grid
was asked of is its key (zone_key), and a changed arm mount, chair model, container or proof asks again.
"""

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from box_opening import reach_zone
from box_opening.constants import HOME, PLANNER_LIMIT_MARGIN_RAD, state_dir

log = logging.getLogger(__name__)

NAMES = ["joint_%d" % i for i in range(1, 8)]
GLOBAL_SEEDS = 8  # the stand-in's IK for a pose the arm's own finds none for: from this many postures
LINE_STEP_RAD = 0.02
# a lid's height is read to the centimetre at best (the wrist camera read the lid 40-44 mm under the arm's
# touch on 2026-11-12): one grid serves every lid within half this of the height it was asked at
LID_Z_BUCKET_M = 0.02


class _Plan:
    def __init__(self, ok, message, trajectory=None):
        self.success = ok
        self.message = message
        self.trajectory = trajectory
        self.planning_time = 0.0


def _joint_line(a, b, step_rad=LINE_STEP_RAD):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(1, int(math.ceil(float(np.abs(b - a).max()) / step_rad)))
    return [list(a + (b - a) * (k / n)) for k in range(n + 1)]


class DryClient:
    """The planner's stand-in, for a proof with no planner: a pose goal is planned to the joints the real
    client plans it to (client.pose_goal_joints, the arm's own IK), else to an IK from a few fixed postures;
    every plan is the straight joint line, refused as the planner refuses a start or a goal inside its
    limit margin or past its bound on a continuous joint. It holds no world: what a line may not enter is
    the chain's own body check, as in a run."""

    def __init__(self, chain):
        from box_opening.primitives.look import look_joints

        self.chain = chain
        self.live = list(HOME)
        rng = np.random.default_rng(1104)
        lo = np.array([(-math.pi if lim[0] is None else lim[0] + 0.12) for lim in chain.limits], float)
        hi = np.array([(math.pi if lim[1] is None else lim[1] - 0.12) for lim in chain.limits], float)
        self.seeds = [list(HOME), list(look_joints(HOME))]
        self.seeds += [list(rng.uniform(lo, hi)) for _ in range(GLOBAL_SEEDS - len(self.seeds))]

    def joints(self):
        return list(self.live)

    def set_world(self, _path_or_name):
        return True, ""

    def _refusal(self, start, goal):
        from box_opening.runtime.branches import CONTINUOUS_JOINTS, PLANNER_CONTINUOUS_LIMIT_RAD

        for what, q in (("START", start), ("GOAL", goal)):
            for j, (lo, hi) in enumerate(self.chain.limits):
                if lo is not None and hi is not None and min(q[j] - lo, hi - q[j]) < PLANNER_LIMIT_MARGIN_RAD:
                    return "%s_IN_LIMIT_MARGIN (joint_%d)" % (what, j + 1)
        for i in CONTINUOUS_JOINTS:
            if abs(goal[i]) > PLANNER_CONTINUOUS_LIMIT_RAD:
                return "GOAL past the continuous bound on joint_%d" % (i + 1)
        return None

    def _joint_plan(self, start, goal):
        from box_opening.runtime.retime import positions_to_traj

        why = self._refusal(start, goal)
        if why is not None:
            return _Plan(False, why)
        return _Plan(True, "ok (stand-in)", positions_to_traj(NAMES, _joint_line(start, goal)))

    def _global_ik(self, xyz, quat, start):
        from box_opening.perception.d405 import quat_to_mat
        from box_opening.runtime.branches import joint_time, nearest_branch

        R = np.asarray(quat_to_mat(*quat), float)
        best = None
        for seed in self.seeds:
            q = self.chain.reach(seed, xyz, quat)
            if q is None:
                q = self.chain.solve_pose(seed, xyz, R)
            if q is None:
                continue
            q = nearest_branch(q, start)
            if not self.chain.within_limits(q, PLANNER_LIMIT_MARGIN_RAD):
                continue
            t = joint_time(start, q)
            if best is None or t < best[0]:
                best = (t, q)
        return None if best is None else best[1]

    def plan_to_pose(self, xyz, quat_xyzw, start_joints, approach_offset_m=0.0):
        from box_opening.runtime.branches import planner_branch
        from box_opening.runtime.client import pose_goal_joints

        start = planner_branch(start_joints)
        q = pose_goal_joints(xyz, quat_xyzw, start_joints)
        if q is None:
            q = self._global_ik(xyz, quat_xyzw, start)
        if q is None:
            return _Plan(False, "IK_FAIL")
        return self._joint_plan(start, [float(v) for v in q])

    def plan_to_joints(self, q7, start_joints):
        from box_opening.runtime.branches import nearest_branch, planner_branch

        start = planner_branch(start_joints)
        return self._joint_plan(start, nearest_branch(q7, start))


class DryStore:
    """What a proof reads of the run's world store: the planning frame and the table's height. It writes
    no world file: the stand-in holds no world."""

    def __init__(self, base_frame, table_top_z=None):
        self.base_frame = base_frame
        self.table_top_z = table_top_z
        self.table_extent = None

    def push_name(self, kind, **kw):
        name = kind + (("_" + kw["tag"]) if kw.get("tag") else "")
        return name, name + ".yaml"


def setup(urdf, root, body_layers, container):
    """The proof's context for the robot whose arm is `urdf` (rooted at `root`) with the chair and user of
    `body_layers` ('' off; None: the run's default), opening `container` (its config): (ctx, cfg). Raises
    where any of them cannot be read: nothing is proven on a robot that is not the one named."""
    from box_opening.models.container import ContainerModel, load_press_demo
    from box_opening.primitives.core import Ctx
    from box_opening.runtime import approach

    approach.set_chain_root(root, urdf_path=urdf)
    ok, why = approach.chain_ready()
    if not ok:
        raise RuntimeError(why)
    approach.set_body_layers(body_layers)
    chain = approach._chain()
    ctx = Ctx(
        model=ContainerModel.load(container),
        cpose=None,
        client=DryClient(chain),
        worlds=DryStore(root),
        config_path=str(container),
    )
    return ctx, load_press_demo(container)


def press_plans(ctx, cfg, top_xyz, table_z=None, q_from=None):
    """Whether a run would plan its press for a lid whose button's top stands at `top_xyz` (the arm's
    frame), the arm at `q_from` (HOME): (True, "", tilt_deg) or (False, why, None). `table_z`: the table's
    top as the cameras have it (None: the box's own foot, as a run with no table reading takes it)."""
    from box_opening.runtime import approach
    from box_opening.tasks import reach_press as rp

    button = tuple(float(v) for v in top_xyz)
    start = [float(v) for v in (HOME if q_from is None else q_from)]
    approach.set_obstacles([])
    ctx.cpose = rp.cpose_for_button(button, 0.0, ctx.model.button_offset)
    ctx.worlds.table_top_z = None if table_z is None else float(table_z)
    ctx.contact_pad, ctx.lid_at, ctx.counter_boxes = 0.0, None, None
    ctx.client.live = list(start)
    if rp.out_of_reach(ctx.model, button):  # as the run's finder asks it of every candidate
        return False, "past everything the arm reaches", None
    shift = rp.PROOF_SHIFT_M > 0
    looks = [] if shift else None
    plan, why = rp.plan_reach(ctx, cfg, button, start, depth=True, looks=looks, shortest_walk=True)
    if plan is None and shift and looks:
        plan, _near, why_near = rp.plan_shifted_look(ctx, cfg, looks, button, start, rp.PROOF_SHIFT_M)
        why = "%s; shifted: %s" % (why, why_near)
    if plan is None:
        return False, "no reach-out press plans (%s)" % why, None
    q_look = [float(v) for v in plan.look_joints]
    ctx.client.live = list(q_look)
    way, why_way, kind = rp.solve_way_in(ctx, cfg, button, plan.tilt_deg, q_look)
    if way is None and kind == "lines":
        plan2, why2 = rp.plan_reach(ctx, cfg, button, q_look, depth=True)
        if plan2 is not None:
            return True, "", int(plan.tilt_deg)
        why_way = "as straight lines: %s; planned: %s" % (why_way, why2)
    if way is None:
        return False, "no press reaches the button from its look (%s)" % why_way, None
    return True, "", int(plan.tilt_deg)


# ---------------------------------------------------------------------------------------- the boundary
def zone_key(urdf, container):
    """What a grid is asked of, as one word: the arm's URDF, the chair model in force, the container's
    config, HOME, and the source of everything the proof runs. Any of them changed, the grid is asked
    again."""
    from box_opening.runtime import approach, body, kinematics
    from box_opening.tasks import depth_press, reach_press

    h = hashlib.sha256()
    h.update(Path(urdf).read_bytes())
    h.update(Path(container).read_bytes())
    model = approach.body_model()
    if model is None:
        h.update(b"no body model")
    else:
        h.update(repr((model.names, model.padding_m, model.buffer_m)).encode())
        h.update(np.round(np.asarray(model.lo, float), 4).tobytes())
        h.update(np.round(np.asarray(model.hi, float), 4).tobytes())
    h.update(repr([round(float(v), 4) for v in HOME]).encode())
    for module in (reach_press, depth_press, kinematics, body, approach, sys.modules[__name__]):
        h.update(Path(module.__file__).read_bytes())
    return h.hexdigest()[:16]


SPAN_FAN_DEG = range(-90, 91, 15)  # the bearings reach_span looks along
SPAN_CAP_M = 3.0  # where its search ends on kinematics that rule nothing out (no arm is this long)


def reach_span(model, lid_z):
    """The grid's ground for a lid at `lid_z`: everything in front of the arm out to the first ring that
    is past its reach on every bearing (reach_press.out_of_reach, the arm's own URDF's bound), to the
    grid's step. A box behind the arm's base is not one the user faces."""
    from box_opening.tasks import reach_press as rp

    step, r = reach_zone.STEP_M, reach_zone.STEP_M
    while r < SPAN_CAP_M:
        ring = [(r * math.cos(math.radians(a)), r * math.sin(math.radians(a)), float(lid_z)) for a in SPAN_FAN_DEG]
        if all(rp.out_of_reach(model, top) for top in ring):
            break
        r += step
    r = round(r, 4)
    return (0.0, r), (-r, r)


def lid_bucket(lid_z):
    """The height a grid for a lid at `lid_z` is asked at: the nearest LID_Z_BUCKET_M."""
    return round(round(float(lid_z) / LID_Z_BUCKET_M) * LID_Z_BUCKET_M, 4)


def zone_path(key, lid_z, where=None):
    where = Path(where) if where else state_dir() / "reach_zone"
    return where / ("zone_%s_z%+05d.json" % (key, round(1000 * lid_bucket(lid_z))))


def load_zone(key, lid_z, model, where=None):
    """The grid kept for `key` at this lid height, or a blank one over reach_span."""
    path = zone_path(key, lid_z, where)
    try:
        zone = reach_zone.Zone.from_json(path.read_text())
        if zone.key == key:
            return zone
    except (OSError, ValueError, KeyError):
        pass
    xs, ys = reach_span(model, lid_bucket(lid_z))
    return reach_zone.Zone.blank(xs, ys, lid_z=lid_bucket(lid_z), key=key)


def save_zone(zone, where=None):
    path = zone_path(zone.key, zone.lid_z, where)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(zone.to_json())
    os.replace(tmp, path)
    return path


def fill(zone, ctx, cfg, table_z=None, focus=None, budget_s=None, n=None, clock=time.monotonic):
    """Ask the cells Zone.order names, at most `n` of them or for `budget_s`: the number asked."""
    t0, asked = clock(), 0
    while (n is None or asked < n) and (budget_s is None or clock() - t0 < budget_s):
        todo = zone.order(focus)
        if not todo:
            break
        i, j = todo[0]
        x, y = zone.centre(i, j)
        ok, _why, _tilt = press_plans(ctx, cfg, (x, y, zone.lid_z), table_z=table_z)
        zone.set(i, j, ok)
        asked += 1
    return asked


def _say(line, to=None):
    """A line of the command's own answer: its stdout is what a caller reads."""
    to = sys.stdout if to is None else to
    to.write(line + "\n")
    to.flush()


def main(argv=None):
    from box_opening.tasks import cli_common

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--urdf", required=True, help="the arm's URDF (the robot's own, as /arm/robot_description has it)")
    ap.add_argument("--root", default="arm_mount", help="the URDF's root, the planning frame")
    ap.add_argument("--body-layers", default=None, help="the planner config with the chair and user; '' off")
    ap.add_argument("--container", default=None, help="the container's config")
    ap.add_argument("--lid-z", type=float, help="fill the boundary for a lid's top at this height (m)")
    ap.add_argument("--table-z", type=float, default=None, help="the table's top (m); default: the box's foot")
    ap.add_argument("--at", type=float, nargs=3, metavar=("X", "Y", "Z"), help="prove one place and stop")
    ap.add_argument("--focus", type=float, nargs=2, metavar=("X", "Y"), help="ask the cells nearest here first")
    ap.add_argument("--zone-dir", default=None, help="where grids are kept (default: the state dir's reach_zone)")
    ap.add_argument("--budget-s", type=float, default=None, help="stop asking after this long")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    container = args.container or cli_common.default_container_yaml()
    ctx, cfg = setup(args.urdf, args.root, args.body_layers, container)
    if args.at is not None:
        t0 = time.monotonic()
        ok, why, tilt = press_plans(ctx, cfg, args.at, table_z=args.table_z)
        took = round(time.monotonic() - t0, 2)
        _say(json.dumps({"at": list(args.at), "ok": ok, "why": why, "tilt_deg": tilt, "s": took}))
        return 0 if ok else 1
    if args.lid_z is None:
        ap.error("one of --lid-z and --at is needed")
    key = zone_key(args.urdf, container)
    zone = load_zone(key, args.lid_z, ctx.model, args.zone_dir)
    t0 = time.monotonic()
    while not zone.done() and (args.budget_s is None or time.monotonic() - t0 < args.budget_s):
        fill(zone, ctx, cfg, table_z=args.table_z, focus=args.focus, n=10)
        save_zone(zone, args.zone_dir)
        _say(json.dumps(dict(zone.counts(), s=round(time.monotonic() - t0, 1))), sys.stderr)
    path = save_zone(zone, args.zone_dir)
    _say(zone.picture())
    _say(json.dumps(dict(zone.counts(), done=zone.done(), lid_z=zone.lid_z, key=zone.key, path=str(path))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
