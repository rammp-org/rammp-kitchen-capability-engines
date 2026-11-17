"""The run's own proof that a press plans, made away from the arm: the word behind every cell of the
line-up's boundary (reach_zone.Zone) and behind "the run plans for the box where it stands".

    reach_probe --urdf <arm URDF> --lid-z 0.315                      the boundary for a lid at that height
    reach_probe --urdf <arm URDF> --at 0.65 -0.10 0.315              one place: does a press plan there
    reach_probe --urdf <arm URDF> --serve                            for the engine's line-up, while it runs
    reach_probe ... --far-m 0.8                                      ask no cell further out than this

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
import select
import signal
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
LID_Z_BUCKET_M, lid_bucket = reach_zone.LID_Z_BUCKET_M, reach_zone.lid_bucket


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
    config, HOME, and the source of everything the proof runs (the run's own modules, and this one's proof
    and stand-ins). Any of them changed, the grid is asked again."""
    import inspect

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
    for module in (reach_press, depth_press, kinematics, body, approach):
        h.update(Path(module.__file__).read_bytes())
    for part in (press_plans, DryClient, DryStore, _joint_line, reach_span):
        h.update(inspect.getsource(part).encode())
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


def zone_path(key, lid_z, where=None):
    where = Path(where) if where else state_dir() / "reach_zone"
    return where / ("zone_%s_z%+05d.json" % (key, round(1000 * lid_bucket(lid_z))))


def load_zone(key, lid_z, model, where=None):
    """The grid kept for `key` at this lid height, or a blank one over reach_span. A file that is not a
    whole grid of this key and height (cut short, not a grid's text at all, another height's under this
    name) is no grid kept: its cells are asked again and it is written over, and the probe does not end
    on it."""
    path = zone_path(key, lid_z, where)
    try:
        zone = reach_zone.Zone.from_json(path.read_text())
        whole = zone.cells.ndim == 2 and zone.cells.size > 0 and zone.step > 0
        whole = whole and all(math.isfinite(v) for v in (zone.x0, zone.y0, zone.step))
        if whole and zone.key == key and abs(float(zone.lid_z) - lid_bucket(lid_z)) < 1e-6:
            return zone
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    xs, ys = reach_span(model, lid_bucket(lid_z))
    return reach_zone.Zone.blank(xs, ys, lid_z=lid_bucket(lid_z), key=key)


def save_zone(zone, where=None):
    """Keep `zone` on disk: written whole beside its place, then moved into it, so the grid's own name
    never holds a half-written file (a probe ended as it writes leaves the earlier grid). The file written
    first carries this process's number: two probes may keep the same grid at once (the engine's and an
    offline fill), and under one shared name the second to move it in would find it gone."""
    path = zone_path(zone.key, zone.lid_z, where)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("%s.%d.tmp" % (path.name, os.getpid()))
    tmp.write_text(zone.to_json())
    os.replace(tmp, path)
    return path


def fill(zone, ctx, cfg, table_z=None, focus=None, budget_s=None, n=None, clock=time.monotonic, far_m=None):
    """Ask the cells Zone.order names, at most `n` of them or for `budget_s`: the number asked. `far_m`:
    the deployment's bound (--far-m; None: none), past which no cell is asked."""
    t0, asked = clock(), 0
    while (n is None or asked < n) and (budget_s is None or clock() - t0 < budget_s):
        todo = zone.order(focus, far_m)
        if not todo:
            break
        i, j = todo[0]
        x, y = zone.centre(i, j)
        ok, _why, _tilt = press_plans(ctx, cfg, (x, y, zone.lid_z), table_z=table_z)
        zone.set(i, j, ok)
        asked += 1
    return asked


# ------------------------------------------------------------------------------- for the engine's line-up
# --serve: the engine (line_up.BoxSource) writes a JSON line on stdin each time it sees the box,
#     {"box": [x, y, z], "table_z": z or null, "prove": bool}
# and reads the lines on stdout that begin with MARK (every other line there is the proof's own log):
#     @@ {"zone": {...}}                                       the boundary for that lid's height, as it fills
#     @@ {"proof": {"xyz": [...], "ok": bool, "why": "...", "tilt_deg": n, "s": seconds}}
# With "prove" the box's own place is proven first (the user is being asked to hold still for it); otherwise
# the boundary is filled, the cells nearest the box first. The newest box is the only one worked on.
# The engine ends it with {"quit": true} (heard after the proof in flight), by closing the pipe, or with a
# terminate (at once): each way the grid is kept first.
MARK = "@@ "
PROOF_NEAR_M = 0.015  # a box this near the place last proven is that place (guide.STEADY_M)
ZONE_SAY_S = 0.5  # a boundary that is filling is said no oftener than this
SAVE_EVERY = 20  # ... and kept on disk every this many cells, and as the server ends
IDLE_S = 0.2  # with nothing to ask, how long stdin is waited on


class Server:
    """One line-up's worth of proofs: hear() each line the engine writes, step() one proof at a time.
    `far_m`: the deployment's bound on the boundary (--far-m; None: none), past which no cell is asked."""

    def __init__(self, ctx, cfg, key, zone_dir=None, say=None, clock=time.monotonic, far_m=None):
        self.ctx, self.cfg, self.key, self.zone_dir = ctx, cfg, key, zone_dir
        self.far_m = far_m
        self._say = say if say is not None else (lambda msg: _say(MARK + json.dumps(msg)))
        self._clock = clock
        self.box = self.table_z = None
        self.prove = False
        self.quit = False
        self.zone = None
        self.proven = None  # (xyz, table_z) of the place last proven
        self._dirty = self._unsaved = 0
        self._said_at = None

    def hear(self, msg):
        if msg.get("quit"):
            self.quit = True
        if msg.get("box") is not None:
            self.box = tuple(float(v) for v in msg["box"])
            self.table_z = None if msg.get("table_z") is None else float(msg["table_z"])
            self.prove = bool(msg.get("prove"))

    def _zone_for(self, lid_z):
        if self.zone is None or self.zone.lid_z != lid_bucket(lid_z):
            self.close()
            self.zone = load_zone(self.key, lid_z, self.ctx.model, self.zone_dir)
            self._tell_zone()  # what is kept for this height, at once
        return self.zone

    def _tell_zone(self):
        self._say({"zone": self.zone.to_dict()})
        self._dirty, self._said_at = 0, self._clock()

    def _is_proven(self):
        if self.proven is None:
            return False
        (x, y, z), table_z = self.proven
        near = math.hypot(self.box[0] - x, self.box[1] - y) <= PROOF_NEAR_M and abs(self.box[2] - z) <= PROOF_NEAR_M
        return near and table_z == self.table_z

    def step(self):
        """One proof: the box's own place where it is asked for and not yet proven, else the boundary's next
        cell. False where there was nothing to ask."""
        if self.box is None:
            return False
        zone = self._zone_for(self.box[2])
        if self.prove and not self._is_proven():
            t0 = self._clock()
            ok, why, tilt = press_plans(self.ctx, self.cfg, self.box, table_z=self.table_z)
            self.proven = (self.box, self.table_z)
            took = round(self._clock() - t0, 2)
            self._say({"proof": {"xyz": list(self.box), "ok": ok, "why": why[:300], "tilt_deg": tilt, "s": took}})
            return True
        # a cell is asked with the box on its own foot (table_z None): one grid for every table at this lid height
        asked = fill(zone, self.ctx, self.cfg, focus=self.box[:2], n=1, far_m=self.far_m)
        self._dirty += asked
        self._unsaved += asked
        if self._unsaved >= SAVE_EVERY:
            self._keep()
        if self._dirty and self._clock() - self._said_at >= ZONE_SAY_S:
            self._tell_zone()
        return bool(asked)

    def _keep(self):
        """The grid onto the disk. Where it cannot be written (a folder that is not this user's, a full
        disk) that is said in the log and the grid goes on being served: a boundary that is not kept costs
        the next line-up its proofs, a probe that ends here costs this one its answers."""
        try:
            save_zone(self.zone, self.zone_dir)
        except OSError as e:
            log.warning("reach_probe: the boundary is not kept (%s)" % e)
        self._unsaved = 0

    def close(self):
        """What was asked is kept, and said."""
        if self.zone is not None and self._unsaved:
            self._keep()
        if self.zone is not None and self._dirty:
            self._tell_zone()


def serve(server, fd=0):
    """Until the engine says quit or closes the pipe: its lines heard between proofs, never during one.
    However the loop ends, what was asked is kept on the way out (Server.close): on quit and on the pipe's
    end, and where it is cut inside a proof, by the engine's terminate (end_on_terminate) or by an error."""
    buf = b""
    try:
        while not server.quit:
            busy = server.step()
            ready, _, _ = select.select([fd], [], [], 0.0 if busy else IDLE_S)
            if not ready:
                continue
            data = os.read(fd, 65536)
            if not data:
                break  # the engine is gone
            *lines, buf = (buf + data).split(b"\n")
            for line in lines:
                try:
                    server.hear(json.loads(line))
                except (ValueError, TypeError, AttributeError):
                    log.warning("reach_probe: not a line of the engine's: %r" % line[:120])
    finally:
        server.close()


def end_on_terminate(server):
    """SIGTERM ends serve(server) at once, with exit code 0, wherever it stands. The engine ends its children
    with a terminate and kills them 2 s later (line_up.Child.stop), and a proof in flight can take that
    long: waiting for it to hear quit would lose every cell asked since the last save. The handler does not
    save by itself: it may have cut a save in two. It leaves by SystemExit, and serve() keeps the grid as
    it unwinds. A second terminate, or one that comes when quit was already heard, changes nothing: the
    probe is on its way out, and the save it is making is not cut short."""

    def ended(_signum, _frame):
        if server.quit:
            return
        server.quit = True
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, ended)


def _say(line, to=None):
    """A line of the command's own answer: its stdout is what a caller reads."""
    to = sys.stdout if to is None else to
    to.write(line + "\n")
    to.flush()


# How far the probe lowers its own priority as it starts (os.nice's step). It computes one proof after
# another for as long as a line-up lasts, a core's worth, beside the arm's driver, and this computer has lost
# that driver under CPU load before. 10 is half the way to the lowest: everything at the ordinary priority
# (the driver, the planner, the run and its cameras) goes first, and what is niced further (the video
# encoder's 19) still gives way to a proof the user is waiting on.
NICE = 10


def _lower_priority():
    """The probe's own process niced by NICE. Where that is not this user's or this platform's to do, the
    probe works on at the priority it was started with: a proof late is worth more than none."""
    try:
        os.nice(NICE)
    except (AttributeError, OSError) as e:
        log.warning("reach_probe: not niced (%s): it runs at the priority it was started with" % e)


def main(argv=None):
    from box_opening.tasks import cli_common

    _lower_priority()  # before anything is computed: --serve, a fill and --at alike
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
    ap.add_argument(
        "--far-m",
        type=float,
        default=0.0,
        help="ask no cell further than this from the arm's base (m): the deployment's line_up_far_m; 0: no bound",
    )
    ap.add_argument("--serve", action="store_true", help="for the engine's line-up: boxes on stdin, answers on stdout")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    container = args.container or cli_common.default_container_yaml()
    # a grid is kept by what it was asked of, never by the bound it was filled under: its cells past a bound
    # are only unasked, and the same file serves a later run under another
    far_m = args.far_m if args.far_m > 0 else None
    ctx, cfg = setup(args.urdf, args.root, args.body_layers, container)
    if args.serve:
        server = Server(ctx, cfg, zone_key(args.urdf, container), args.zone_dir, far_m=far_m)
        end_on_terminate(server)
        serve(server)
        return 0
    if args.at is not None:
        t0 = time.monotonic()
        ok, why, tilt = press_plans(ctx, cfg, args.at, table_z=args.table_z)
        took = round(time.monotonic() - t0, 2)
        _say(json.dumps({"at": list(args.at), "ok": ok, "why": why, "tilt_deg": tilt, "s": took}))
        return 0 if ok else 1
    if args.lid_z is None:
        ap.error("one of --lid-z, --at and --serve is needed")
    key = zone_key(args.urdf, container)
    zone = load_zone(key, args.lid_z, ctx.model, args.zone_dir)
    t0 = time.monotonic()
    while not zone.done(far_m) and (args.budget_s is None or time.monotonic() - t0 < args.budget_s):
        fill(zone, ctx, cfg, table_z=args.table_z, focus=args.focus, n=10, far_m=far_m)
        save_zone(zone, args.zone_dir)
        _say(json.dumps(dict(zone.counts(), s=round(time.monotonic() - t0, 1))), sys.stderr)
    path = save_zone(zone, args.zone_dir)
    _say(zone.picture())
    _say(json.dumps(dict(zone.counts(), done=zone.done(far_m), lid_z=zone.lid_z, key=zone.key, path=str(path))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
