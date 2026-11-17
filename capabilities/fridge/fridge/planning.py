"""Every fridge motion: legs to tool poses, each in its own collision world, joined by cuRobo into one trajectory that
the driver runs.

A motion is a list of Goals, each planned from where the leg before it ends (the first from the arm's joints):
- a free-space goal ("free"): cuRobo's plan_to_pose (or plan_to_joints) against the world the goal names (FridgeModel's
  boxes, pushed with PlannerWorld, plus the planner's own chair layers): routing round the door, the handle, the
  cabinet and the wheelchair (its layers include the scene camera), the arm's configuration cuRobo's to choose;
- a goal on or along the door ("line", "arc", "through": straight in to the handle, the swing, the release, the push,
  the close's sweep): solved here, because cuRobo can't follow a line or an arc -- its pose and joint goals both left
  them by 1.5-9 cm on 11-07. Local IK along the exact path every PATH_STEP_M (kinematics.Chain.ik: one branch, the
  arm's spare joint pulling J2/J4/J6 back from their limits, as the side grasp's pull), timed rest to rest with the
  speed, acceleration and bend limits of the side grasp's fast_timing -- and then checked by cuRobo straight away (a
  one-leg blend: every state against the leg's world, the chair's layers and the arm itself, and the joint limits).
cuRobo's blend then joins the legs into ONE trajectory, cutting the corner at a goal where its `blend` radius allows
and stopping on it where it is 0, and checks every state again against the world it is given. A whole contact path is
one "through" leg (no junction, no stop). Planner.run sends it (execute_joint_trajectory), in position control or the
driver's joint impedance, with timed events (the gripper) while it runs.
"""

import re
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import rammp_arm
from door_common.common import Refused, say
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from .kinematics import JOINTS

# self.arm's orientation holds (capability_engine.arm_messages, PlanToPose's): free; keep the tilt, spin about
# vertical free in transit; keep the orientation entirely
HOLD_NONE, HOLD_LEVEL, HOLD_FIXED = 0, 1, 2
IMPEDANCE = 1  # execute_joint_trajectory's control_mode for joint impedance (capability_engine.arm.IMPEDANCE)

PATH_STEP_M, PATH_STEP_DEG = 0.01, 2.0  # a leg along the door: a point every this far / this much turn
MAX_STEP_DEG = 8.0  # ... and no joint more than this between two points
LOCAL_JOINT_SPEED = 1.0  # rad/s: such a leg's "full speed" (the blend scales it, as cuRobo's own legs) ...
LOCAL_ACCEL = 1.5  # ... rad/s^2 speeding up and slowing down (the side grasp's fast_timing) ...
LOCAL_SMOOTH_S = 0.15  # ... and its speed smoothed over this long
BLEND_MIN_M = 0.002  # the smallest corner a stop after a blended junction gets (join: works round a blend bug)
END_TOL_M = 0.01  # a planned leg must end this close to its goal (cuRobo's own tolerance is tighter)
START_TOL_DEG = 1.0  # a planned motion runs only if the arm is this close to the joints it was planned from
RAISED_RETRIES = 1  # a leg whose cuRobo plan raised (status EXCEPTION, not a planning answer) is asked again this often
KEYPOINT_TOL_M = 0.015  # a motion counts as done with the tool this close to its last goal
CONTACT_SPEED = 0.5  # a contact leg (`contact`) runs at this fraction of the free-space speed
SETTLE_S = 0.3  # after a motion, let the arm's state catch up


class StartNotClear(Refused):
    """cuRobo won't plan from where the arm is: every plan from there would fail the same way."""

    def blocking(self):
        """The obstacles the start collides with, as cuRobo names them ("door", "person/lap", ...); [] if it names
        none (a joint limit, or a message without the list)."""
        m = re.search(r"start collides with ([^\]]*)\]", str(self))
        return [] if m is None else [b.split(" (")[0].strip() for b in m.group(1).split(", ") if b.strip()]

    def only_fridge(self):
        """True when the start collides with the fridge's own boxes alone (FridgeModel's: they move with the
        wheelchair, so another wheelchair move may clear them), never the chair's layers."""
        boxes = self.blocking()
        return bool(boxes) and all(re.fullmatch(r"(m\d+_)?(body|door\w*|handle\w*)", b) for b in boxes)


class OverBudget(Refused):
    """The search has used up its planning time (Planner's budget_s)."""


GOAL_FAILURES = ("IK_FAIL", "INVALID_GOAL", "solution AT the goal")  # cuRobo statuses that mean the goal pose itself can't be reached: no
# route to it (a via point) can help


def goal_unreachable(why):
    """True when a leg failed because cuRobo found no collision-free arm configuration AT its goal (not on the way)."""
    return any(f in str(why) for f in GOAL_FAILURES)


@dataclass
class Goal:
    """One leg of a motion: the tool to (`pos`, `quat` xyzw) in arm_mount, planned against `world` (obstacle dicts),
    with orientation hold `hold` (HOLD_*). `blend`: how far (m) the arm may cut the corner at this goal into the next
    leg (0: it stops here). `contact`: the leg touches the handle or the door (slower). `path`: "free", "line" or
    "arc" (about `hinge`, the door's pivot xy) -- the path a leg along the door is solved on. `joints`: plan to these
    joints instead (cuRobo's joint-space plan; `pos`/`quat` are then where they put the tool)."""

    pos: np.ndarray
    quat: np.ndarray
    label: str
    world: list
    hold: int = HOLD_NONE
    blend: float = 0.0
    contact: bool = False
    path: str = "free"
    hinge: np.ndarray | None = None
    world_label: str = ""
    joints: np.ndarray | None = None
    speed: float | None = None  # this leg's speed, a fraction of cuRobo's full speed (None: free / contact default)
    through: list | None = None  # path "through": the tool poses [(pos, quat)] it passes, in order (the last = pos,
    # quat), solved here as ONE leg -- no junctions, no stops
    caps: list | None = None  # per pose of `through`: its speed there, a fraction of LOCAL_JOINT_SPEED
    segment_worlds: list | None = None  # per pose of `through`: (label, obstacles) the way to it is checked against
    # too, by cuRobo (a door that turns as the leg pushes it: each segment against the door where it is then)

    def __post_init__(self):
        self.pos = np.asarray(self.pos, float)[:3]
        self.quat = np.asarray(self.quat, float) / np.linalg.norm(self.quat)


@dataclass
class Motion:
    """A planned motion: its goals, each goal's leg (self.arm Plans), and the arm's joints it was planned from."""

    goals: list
    legs: list
    q0: np.ndarray
    joined: object = None  # the blended Plan, once joined
    notes: list = field(default_factory=list)

    @property
    def q_end(self):
        return np.asarray(self.legs[-1].q[-1], float)

    @property
    def duration(self):
        if self.joined is not None and len(self.joined.t):
            return float(self.joined.t[-1])
        return float(sum(leg.t[-1] for leg in self.legs if len(leg.t)))


class Planner:
    """The planner of one action: self.arm (`arm`), the world it pushes (`world`, PlannerWorld), the arm's forward
    kinematics (`fk`, kinematics.Chain) and `checkpoint` (the action's: a cancel stops a long search)."""

    def __init__(self, arm, world, fk, checkpoint=None, budget_s=None):
        self.arm, self.world, self.fk = arm, world, fk
        self.checkpoint = checkpoint or (lambda: None)
        self.budget_s = budget_s  # the time one search may take (s); None: no limit
        self.t0 = time.monotonic()
        self.search_t0 = self.t0  # when the current search started (search()): the budget's clock
        self.plans = 0
        self.cached = 0
        self.local = 0  # legs solved here (on or along the door)
        self.plan_s = 0.0
        self._cache = {}  # the same leg (goal, start, world) is planned once per action
        # a free goal cuRobo found no collision-free arm configuration AT (goal_unreachable): that doesn't depend on
        # where the leg starts, so it isn't asked again from another start (11-09 sim: the push-open's 16 hand
        # placements x 8 wheelchair moves, all IK_FAIL, asked again for every grasp, ran a search out of its budget)
        self._unreachable = {}

    def _key(self, goal, q_start, start_pose):
        r = lambda a: tuple(np.round(np.asarray(a, float).ravel(), 4))  # noqa: E731
        world = tuple((b["name"], r(b["position"]), r(b["rpy_deg"]), r(b["dims"])) for b in goal.world)
        sp = None if start_pose is None else (r(start_pose[0]), r(start_pose[1]))
        thr = None if goal.through is None else tuple((r(p), r(q)) for p, q in goal.through)
        caps = None if goal.caps is None else r(goal.caps)
        return (r(goal.pos), r(goal.quat), goal.hold, goal.path, None if goal.joints is None else r(goal.joints),
                None if goal.hinge is None else r(goal.hinge), r(q_start), sp, world, thr, caps)

    # -- planning ---------------------------------------------------------------------------------------------------

    def search(self):
        """Start a new search: budget_s counts from here. Called where planning starts afresh -- the whole open or
        close, and every re-plan after the arm or the base has moved -- so the time the arm spends moving (or the
        base driving, or the camera detecting) never counts against a later re-plan."""
        self.search_t0 = time.monotonic()

    def check_budget(self):
        """Raise OverBudget if the current search has used up budget_s (also for work done here, between plans)."""
        self.checkpoint()
        if self.budget_s is not None and time.monotonic() - self.search_t0 > self.budget_s:
            raise OverBudget(f"planning took longer than its {self.budget_s:.0f} s budget ({self.plans} plans)")

    def tool(self, q):
        """(position, quat) of the tool at joints `q`."""
        return self.fk.pose(q)

    def leg(self, goal, q_start, start_pose=None, from_arm=False):
        """cuRobo's leg to `goal` from joints `q_start`, checked (see the module). Returns (Plan, None) or (None,
        why). `from_arm`: `q_start` is where the arm is -- if cuRobo says that start is not clear (inside an obstacle of
        its world or the chair's layers, or past a limit) nothing can start from it, and it raises StartNotClear."""
        self.checkpoint()
        key = self._key(goal, q_start, start_pose)
        if key in self._cache:
            self.cached += 1
            return self._cache[key]
        goal_key = self._goal_key(goal)
        if goal_key in self._unreachable:
            self.cached += 1
            return self._unreachable[goal_key]
        if self.budget_s is not None and time.monotonic() - self.search_t0 > self.budget_s:
            raise OverBudget(f"planning took longer than its {self.budget_s:.0f} s budget ({self.plans} plans)")
        self._cache[key] = out = self._leg(goal, q_start, start_pose, from_arm)
        if goal_key is not None and out[0] is None and goal_unreachable(out[1]):
            self._unreachable[goal_key] = out
        return out

    def _goal_key(self, goal):
        """A free pose goal's key without its start (None for other legs: theirs depend on the start)."""
        if goal.path != "free" or goal.joints is not None:
            return None
        key = self._key(goal, np.zeros(7), None)
        return key[:6] + key[8:]  # without the start (joints, pose)

    def _leg(self, goal, q_start, start_pose, from_arm):
        self.world.push(goal.world, goal.world_label or goal.label)
        t0 = time.monotonic()
        if goal.path != "free":
            # on or along the door: cuRobo can't follow a line or an arc (11-07: its pose and joint goals both left them
            # by 1.5-9 cm), so the leg is local IK along it; cuRobo checks it when it blends the motion (join)
            self.local += 1
            return self._along(goal, q_start, start_pose)
        for _ in range(1 + RAISED_RETRIES):
            if goal.joints is not None:
                plan = self.arm.plan_to_joints(list(map(float, goal.joints)), list(map(float, q_start)))
            else:
                plan = self.arm.plan_to_pose(goal.pos, goal.quat, list(map(float, q_start)), hold=goal.hold)
            self.plans += 1
            # cuRobo raised (no planning answer: RAMMP-CuRobo 2.0.1 reports it as EXCEPTION; 11-08 a
            # "shape '[1, 1, 1, 3]' is invalid" seen in some runs and not others): ask again rather than take
            # the goal as unreachable
            if plan.success or not str(plan.message).startswith("EXCEPTION"):
                break
            say(f"    cuRobo raised planning {goal.label} ({_short(plan.message, 100)}); asking again")
        self.plan_s += time.monotonic() - t0
        if not plan.success:
            if from_arm and "START_STATE" in str(plan.message):
                raise StartNotClear(
                    f"the arm's start pose is not clear in cuRobo's world with {goal.world_label or 'the fridge'} in it: "
                    f"{' '.join(str(plan.message).split())} -- no plan can start from it. Move the arm out first (e.g. "
                    "the go_to_start action: the web app's Fridge pose)"
                )
            return None, f"{goal.label}: cuRobo found no plan ({_short(plan.message)})"
        if not len(plan.q):  # mock mode: nothing to check
            return plan, None
        pos, _ = self.fk.poses(plan.q)
        end_err = float(np.linalg.norm(pos[-1] - goal.pos))
        if end_err > END_TOL_M:
            return None, f"{goal.label}: the plan ends {end_err * 100:.1f} cm from its goal"
        return plan, None

    def _along(self, goal, q_start, start_pose):
        """A line or arc leg by local IK: the tool's path from its start to the goal every PATH_STEP_M / PATH_STEP_DEG
        (the line; or round the hinge, the radius and height going linearly), the hand turning evenly, each point solved
        from the last (one branch, the spare joint keeping J2/J4/J6 off their limits); timed to start and end at rest.
        Returns (LocalPlan, None) or (None, why)."""
        p0, q0 = start_pose if start_pose is not None else self.tool(q_start)
        poses, caps = path_poses(goal, np.asarray(p0, float), np.asarray(q0, float))
        qs, q = [], np.asarray(q_start, float)
        for k, (pos, quat) in enumerate(poses):
            nq, why = self.fk.ik(pos, quat, q)
            if nq is None:
                return None, f"{goal.label}: point {k + 1}/{len(poses)}: {why}"
            jump = float(np.degrees(np.max(np.abs(nq - q))))
            if jump > MAX_STEP_DEG:
                return None, f"{goal.label}: point {k + 1}/{len(poses)}: a joint jumps {jump:.0f} deg"
            qs.append(nq)
            q = nq
        t, qq, qd = timed(np.asarray(q_start, float), np.asarray(qs), caps)
        plan = LocalPlan(True, "local IK", t, qq, qd)
        # cuRobo checks it now, in this leg's own world: every state against the world, the chair's layers and the arm
        # itself, and the joint limits (a one-leg blend validates it), so a leg that clips something fails here, where
        # the search can still try another placement, not when the motion is joined
        self.world.push(goal.world, goal.world_label or goal.label)
        t0 = time.monotonic()
        res = self.arm.blend([plan], [], speed_scale=[1.0], hold=[goal.hold],
                             hold_orientation=[goal.quat if goal.hold != HOLD_NONE else None])
        self.plan_s += time.monotonic() - t0
        if not res.success:
            return None, f"{goal.label}: cuRobo rejects it ({_short(res.message)})"
        if goal.segment_worlds is not None:
            why = self._check_segments(goal, np.asarray(q_start, float), np.asarray(qs), segments(goal, p0, q0))
            if why:
                return None, f"{goal.label}: {why}"
        return plan, None

    def _check_segments(self, goal, q_start, qs, segs):
        """Each run of points of a through leg (`qs`, the pose `segs` each is on its way to) checked by cuRobo against
        its segment's world (goal.segment_worlds): the states timed rest to rest on their own (only where they are is
        checked; the leg's own timing was checked with it). Returns why it fails, or ''."""
        start = 0
        while start < len(qs):
            k = segs[start]
            end = start
            while end + 1 < len(qs) and goal.segment_worlds[segs[end + 1]] is goal.segment_worlds[k]:
                end += 1
            label, world = goal.segment_worlds[k]
            before = q_start if start == 0 else qs[start - 1]
            t, qq, qd = timed(before, qs[start : end + 1])
            self.world.push(world, label)
            t0 = time.monotonic()
            res = self.arm.blend([LocalPlan(True, "local IK", t, qq, qd)], [], speed_scale=[1.0], hold=[goal.hold],
                                 hold_orientation=[goal.quat if goal.hold != HOLD_NONE else None])
            self.plan_s += time.monotonic() - t0
            if not res.success:
                return f"against {label}, cuRobo rejects it ({_short(res.message)})"
            start = end + 1
        return ""

    def motion(self, goals, q0, start_pose=None, from_arm=False):
        """Plan every goal in turn, each from where the last ends. Returns (Motion, None) or (None, why). `from_arm`:
        `q0` is where the arm is (leg's)."""
        legs, done, q, pose = [], [], np.asarray(q0, float), start_pose
        for goal in goals:
            plan, why = self.leg(goal, q, pose, from_arm=from_arm and not done)
            if plan is None:
                return None, why
            legs.append(plan)
            done.append(goal)
            if len(plan.q):
                q = np.asarray(plan.q[-1], float)
            pose = (goal.pos, goal.quat)
        return Motion(done, legs, np.asarray(q0, float)), None

    def join(self, motion, world, world_label="blend", speed=None):
        """cuRobo's blend of `motion`'s legs into one trajectory, against `world` (obstacles every leg's own world
        has, so each leg is valid in it): free legs at GO_TO_SPEED_SCALE (x `speed`), contact legs at CONTACT_SPEED of
        that. If the corners can't be cut, it stops at every goal instead. Raises Refused if even that fails."""
        free = rammp_arm.GO_TO_SPEED_SCALE if speed is None else speed
        # a leg solved here: the old planner's contact joint speed (JOINT_SPEED x contact_speed_scale: both scaled by
        # the goal's speed) against its own LOCAL_JOINT_SPEED timing; cuRobo's: free / contact fractions of its own
        local = rammp_arm.JOINT_SPEED * rammp_arm.contact_speed_scale() / LOCAL_JOINT_SPEED
        scales = [
            max(0.01, min(1.0, g.speed if g.speed is not None
                          else local if g.path != "free"
                          else free * (CONTACT_SPEED if g.contact else 1.0)))
            for g in motion.goals
        ]
        holds = [g.hold for g in motion.goals]
        quats = [g.quat if g.hold != HOLD_NONE else None for g in motion.goals]
        radii = [g.blend for g in motion.goals[:-1]]
        # RAMMP-CuRobo 2.0.1's blend (blend.py) rejects a junction that stops (0) right after one that blends, when that
        # corner's overlap uses up the whole leg between them ("overlap 0 at junction k is outside 0..-1"; only a leg
        # short enough for the corner to swallow). The true stops are asked for first; only if cuRobo refuses them
        # does such a stop become a BLEND_MIN_M corner -- all but a stop (11-09: the push-open's stop at 90 deg follows
        # a cut corner and must stay a stop; its 30-deg leg can't be swallowed)
        bumped = list(radii)
        for k in range(1, len(bumped)):
            if bumped[k] == 0.0 and bumped[k - 1] > 0.0:
                bumped[k] = BLEND_MIN_M
        attempts = [radii] + ([bumped] if bumped != radii else []) + ([[0.0] * len(radii)] if any(radii) else [])
        self.world.push(world, world_label)
        for attempt in attempts:
            self.checkpoint()
            plan = self.arm.blend(motion.legs, attempt, speed_scale=scales, hold=holds, hold_orientation=quats)
            if plan.success:
                motion.joined = plan
                if attempt is bumped:
                    motion.notes.append(f"stops after a cut corner made {BLEND_MIN_M * 1000:.0f} mm corners (blend "
                                        "refused the stops)")
                elif not any(attempt) and any(radii):
                    motion.notes.append("corners not cut (blend refused them): stops at every goal")
                return motion
            what = "with true stops" if attempt is radii and any(radii) else (
                "with stops made small corners" if attempt is bumped else "without cut corners")
            say(f"  blend {what} failed: {_short(plan.message)}")
        raise Refused(f"cuRobo could not join the {len(motion.legs)} planned legs into one trajectory")

    # -- running ----------------------------------------------------------------------------------------------------

    def passes(self, motion, point):
        """When (s into the joined trajectory) the tool comes closest to `point` (a position, or a goal's index)."""
        if isinstance(point, int | np.integer):
            point = motion.goals[point].pos
        pos, _ = self.fk.poses(motion.joined.q)
        return float(motion.joined.t[int(np.argmin(np.linalg.norm(pos - np.asarray(point, float), axis=1)))])

    def run(self, motion, label, impedance=None, events=()):
        """Run the joined `motion` (join first). Refuses if the arm has moved since it was planned. `impedance`: run it
        compliant -- the driver's joint impedance with these gains ({kq, zeta, torque_limit}, + path_tol: the tracking
        limit, rad) instead of stiff position control. `events`: [(t s into the trajectory, fn)] called while it runs (a
        thread: the gripper closing or opening as the arm moves). Returns True when the tool ended within
        KEYPOINT_TOL_M of the last goal."""
        plan = motion.joined
        q_now = np.asarray(self.arm.get_state()["position"], float)
        off = np.degrees(np.max(np.abs(_short_delta(motion.q0, q_now))))
        if off > START_TOL_DEG:
            raise Refused(f"{label}: the arm is {off:.1f} deg from where it was planned from -- not sent")
        say(f"{label}: {len(motion.goals)} leg(s), {float(plan.t[-1]):.1f} s, one trajectory"
            + (" under joint impedance" if impedance else "") + " ...")
        kw = dict(path_tolerance=rammp_arm.PATH_TOL_RAD)
        if impedance:
            kw = dict(path_tolerance=impedance["path_tol"], control_mode=IMPEDANCE,
                      gains={k: impedance[k] for k in ("kq", "zeta", "torque_limit")})
        thread = _timed_events(events, self.say_error) if events else None
        res = None
        try:
            res = self.arm.execute_joint_trajectory(
                JOINTS, [(float(t), q, qd) for t, q, qd in zip(plan.t, plan.q, plan.qd, strict=True)], **kw
            )
        finally:
            if thread is not None:
                if res is None or res.error_code != 0:
                    # stopped short (an error, an e-stop, a cancel): what was timed for later in the motion -- the
                    # gripper opening at the pull's end -- must not happen with the arm stopped part way
                    thread.cancel()
                thread.join(timeout=10.0)
        time.sleep(SETTLE_S)
        st = self.arm.get_state()
        err = float(np.linalg.norm(np.asarray(st["ee_pos"][:3]) - motion.goals[-1].pos))
        if res.error_code != 0:
            say(f"{label}: STOPPED ({rammp_arm.ERRORS.get(res.error_code, res.error_code)} {res.error_string})")
            return False
        say(f"{label}: done, tool {err * 100:.1f} cm from the last goal")
        return err <= KEYPOINT_TOL_M

    @staticmethod
    def say_error(what, e):
        say(f"  {what} failed while the arm moved: {e!r}")

    def summary(self):
        return (f"{self.plans} cuRobo plans + {self.local} legs along the door ({self.cached} repeats skipped), "
                f"{self.plan_s:.1f} s planning, "
                f"{self.world.pushes} world pushes, {time.monotonic() - self.t0:.1f} s in all")


@dataclass
class LocalPlan:
    """A leg solved here (Planner._along), in self.arm's Plan's shape: cuRobo's blend takes it like its own."""

    success: bool
    message: str
    t: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    planning_time: float = 0.0


def segments(goal, p0, q0):
    """For each point of path_poses(goal, p0, q0) on a through leg: the index of the pose of goal.through it is on its
    way to."""
    out, prev = [], (p0, q0)
    for k, (pos, quat) in enumerate(goal.through):
        seg = Goal(np.asarray(pos, float), np.asarray(quat, float), "", [], path="line")
        out += [k] * len(path_poses(seg, np.asarray(prev[0], float), np.asarray(prev[1], float))[0])
        prev = (seg.pos, seg.quat)
    return out


def path_poses(goal, p0, q0):
    """The tool poses along a leg from (p0, q0) to the goal, every PATH_STEP_M / PATH_STEP_DEG -- a line, an arc, or
    through its poses (straight from one to the next) -- and each one's speed cap (fraction of LOCAL_JOINT_SPEED)."""
    if goal.path == "through":
        out, caps, prev = [], [], (p0, q0)
        for k, (pos, quat) in enumerate(goal.through):
            seg = Goal(np.asarray(pos, float), np.asarray(quat, float), "", [], path="line")
            pts, _ = path_poses(seg, np.asarray(prev[0], float), np.asarray(prev[1], float))
            out += pts
            caps += [1.0 if goal.caps is None else float(goal.caps[k])] * len(pts)
            prev = (seg.pos, seg.quat)
        return out, caps
    turn = float(np.degrees((R.from_quat(q0).inv() * R.from_quat(goal.quat)).magnitude()))
    if goal.path == "arc":
        h = np.asarray(goal.hinge, float)[:2]
        a0, a1 = (np.arctan2(*(p[:2] - h)[::-1]) for p in (p0, goal.pos))
        da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
        r0, r1 = (float(np.linalg.norm(p[:2] - h)) for p in (p0, goal.pos))
        length = max(abs(da) * max(r0, r1), float(np.linalg.norm(goal.pos - p0)))
    else:
        length = float(np.linalg.norm(goal.pos - p0))
    n = max(1, int(np.ceil(length / PATH_STEP_M)), int(np.ceil(turn / PATH_STEP_DEG)))
    slerp = Slerp([0.0, 1.0], R.from_quat([q0, goal.quat]))
    out = []
    for k in range(1, n + 1):
        s = k / n
        if goal.path == "arc":
            a, r = a0 + da * s, r0 + (r1 - r0) * s
            pos = np.r_[h + r * np.array([np.cos(a), np.sin(a)]), p0[2] + (goal.pos[2] - p0[2]) * s]
        else:
            pos = p0 + (goal.pos - p0) * s
        out.append((pos, slerp(s).as_quat()))
    out[-1] = (goal.pos, goal.quat)
    return out, [1.0] * len(out)


def timed(q0, qs, caps=None, dt=0.02):
    """Times for joint path [q0, *qs] (the side grasp's fast_timing, in short): speed along the path (its largest joint)
    at most LOCAL_JOINT_SPEED x the point's cap, speeding up and slowing down at most LOCAL_ACCEL, from rest to rest;
    the speed then smoothed in time (no jerk at the corners of the profile). Sampled every `dt` as cuRobo's plans are:
    the first sample IS q0, at rest at both ends, as cuRobo's blend needs its legs. Returns (t, q, qd), t from dt."""
    path = np.vstack([q0, qs])
    d = np.diff(path, axis=0)
    L = np.max(np.abs(d), axis=1)  # each step's length (largest joint, rad)
    cap = LOCAL_JOINT_SPEED * (np.ones(len(L)) if caps is None else np.asarray(caps, float))
    # the bend: at each inner point, how fast the joints' direction turns along the path (|q''| v^2 <= LOCAL_ACCEL) --
    # a corner of the path (onto the door, then round with it) is taken slowly enough, the path itself unchanged
    u = d / np.maximum(L, 1e-12)[:, None]
    bend = np.zeros(len(path))
    bend[1:-1] = np.max(np.abs(np.diff(u, axis=0)), axis=1) / np.maximum((L[:-1] + L[1:]) / 2, 1e-9)
    v_bend = np.sqrt(LOCAL_ACCEL / np.maximum(bend, 1e-9))
    v = np.zeros(len(path))  # speed at each point
    for i in range(1, len(path)):  # speeding up
        v[i] = min(cap[i - 1], v_bend[i], np.sqrt(v[i - 1] ** 2 + 2 * LOCAL_ACCEL * L[i - 1]))
    v[-1] = 0.0
    for i in range(len(path) - 2, -1, -1):  # slowing down into the end, and into every lower cap
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * LOCAL_ACCEL * L[i]))
    v[0] = 0.0
    seg_t = np.where(L > 0, 2 * L / np.maximum(v[:-1] + v[1:], 1e-9), 0.0)
    seg_t = np.where((L > 0) & (v[:-1] + v[1:] < 1e-6), np.sqrt(2 * L / LOCAL_ACCEL), seg_t)
    t_k = np.r_[0.0, np.cumsum(seg_t)]
    total = max(float(t_k[-1]), 2 * dt)
    n = max(2, int(np.ceil(total / dt)))
    t = np.linspace(0.0, total, n + 1)
    s = np.interp(t, t_k, np.arange(len(path), dtype=float))  # path index at each time
    # smooth the speed in time, kept at rest at both ends, and rescaled to end on the path's end
    sd = np.gradient(s, t)
    w = max(1, int(round(LOCAL_SMOOTH_S / (t[1] - t[0]))))
    k = np.ones(2 * w + 1) / (2 * w + 1)
    sd = np.convolve(np.r_[np.zeros(w), sd, np.zeros(w)], k, mode="same")[w:-w]
    sd[0] = sd[-1] = 0.0
    s = np.r_[0.0, np.cumsum((sd[1:] + sd[:-1]) / 2 * np.diff(t))]
    s *= (len(path) - 1) / max(s[-1], 1e-12)
    idx = np.arange(len(path), dtype=float)
    q = np.column_stack([np.interp(s, idx, path[:, j]) for j in range(path.shape[1])])
    tt = (np.arange(len(t)) + 1) * dt
    qd = np.gradient(q, tt, axis=0)
    qd[0] = qd[-1] = 0.0
    return tt, q, qd


class _Events(threading.Thread):
    """_timed_events' thread: join() it; cancel() drops every event not yet called (and wakes it to end)."""

    def __init__(self, events, on_error):
        super().__init__(daemon=True)
        self.events, self.on_error = sorted(events, key=lambda e: e[0]), on_error
        self.t0 = time.monotonic()
        self._cancelled = threading.Event()
        self.called = []  # the events' times, as each was called

    def cancel(self):
        self._cancelled.set()

    def run(self):
        for t, fn in self.events:
            if self._cancelled.wait(max(0.0, self.t0 + t - time.monotonic())):
                return
            try:
                self.called.append(t)
                fn()
            except Exception as e:  # noqa: BLE001 -- reported; the motion goes on (its own checks judge it)
                self.on_error(getattr(fn, "__name__", "event"), e)


def _timed_events(events, on_error):
    """Call each (t, fn) of `events` t seconds from now, in order, in a thread (started; join it, or cancel() it to
    drop what has not been called yet)."""
    th = _Events(events, on_error)
    th.start()
    return th


def _short_delta(a, b):
    d = np.asarray(b, float) - np.asarray(a, float)
    for j in (0, 2, 4, 6):  # the free joints, the short way round
        d[j] = (d[j] + np.pi) % (2 * np.pi) - np.pi
    return d


def _short(message, n=160):
    message = " ".join(str(message).split())
    return message if len(message) <= n else message[: n - 3] + "..."


def free(pos, quat, label, world, blend=0.03, hold=HOLD_NONE, world_label=""):
    """A free-space goal: cuRobo routes round everything in `world`."""
    return Goal(pos, quat, label, world, hold=hold, blend=blend, world_label=world_label)


def to_joints(joints, pos, quat, label, world, blend=0.03, world_label=""):
    """A free-space goal given as joints (the tool then at `pos`, `quat`): cuRobo's joint-space plan."""
    return Goal(pos, quat, label, world, blend=blend, world_label=world_label, joints=np.asarray(joints, float))


def line(pos, quat, label, world, contact=True, blend=0.0, hold=HOLD_FIXED, world_label=""):
    """A straight leg (onto or off the handle or door)."""
    return Goal(pos, quat, label, world, hold=hold, blend=blend, contact=contact, path="line", world_label=world_label)


def through(poses, label, world, caps=None, blend=0.0, hold=HOLD_NONE, world_label="", segment_worlds=None):
    """ONE contact leg through the tool poses [(pos, quat)] in order (straight from one to the next: give them densely
    along an arc), each with its speed cap (`caps`, a fraction of LOCAL_JOINT_SPEED): no junction, no stop. Checked
    against `world` whole, and the way to each pose against `segment_worlds` [(label, obstacles)] too, if given."""
    pos, quat = poses[-1]
    return Goal(pos, quat, label, world, hold=hold, blend=blend, contact=True, path="through",
                through=[(np.asarray(p, float), np.asarray(q, float)) for p, q in poses],
                caps=None if caps is None else list(caps), world_label=world_label,
                segment_worlds=None if segment_worlds is None else list(segment_worlds))


def arc(pos, quat, label, world, hinge, blend=0.01, hold=HOLD_NONE, world_label=""):
    """A leg along an arc about the hinge (the swing, the pushes): a contact leg."""
    return Goal(pos, quat, label, world, hold=hold, blend=blend, contact=True, path="arc", hinge=hinge,
                world_label=world_label)
