"""home_arm: one action, `go_home`, that takes the arm to its home pose around the wheelchair and
its user, from wherever it stands.

The move is planned by cuRobo in its world, which on a wheelchair-mounted arm holds the chair, the
seat and the person in it, so the path keeps clear of them. Four things stop a plain plan:

* a start inside the padding round those boxes (clear of the chair itself, but within the planner's
  world_padding) is refused outright. Where the engine's copy of the planner's boxes
  (home_arm.body) puts the start in the padding and the planner refuses it too, the engine first
  takes a short plan-free step out of the padding, never deeper, the tool held within 5 mm of its
  height, and plans from its end;
* a joint too near its limit is refused too (START_IN_LIMIT_MARGIN): the engine takes the shortest
  plan-free step along the joint line toward the home that clears it and that the home plans from
  (homing.limit_steps), the tool held and never into the padding;
* the planner is stochastic and sometimes fails, or answers with another arm configuration at the
  same tool pose: a plan that does not end at the home counts as failed;
* the planner sometimes answers with a plan that ends at the home but swings a joint far wider than
  the straight way there (flatbot 2026-11-10: joint_1 out 0.9 rad past its start and back, the hand
  toward the user). Such a plan counts as failed too (homing.swing), and asking again gets the
  same plan, so the engine goes on to the waypoints.

The home is asked for up to `plan_tries` times, then through a waypoint along the joint line toward
it (home_arm.homing), the second leg planned from where the first one ends.

Everything is planned before anything moves, then the arm's joints are read again: an arm that
moved meanwhile is planned for again, once. Then the plan-free step (if any) and the plan(s) are
each sent to the driver once (`execute_joint_trajectory`), the plans at `speed_scale` (the go_to
goals' default, 0.3): a goal the driver aborts or rejects fails the action, never retried, and a
cancel stops the arm where it is (self.arm cancels the goal first).

    ros2 launch home_arm home_arm.launch.py use_orchestrator:=false
make action CAP=home_arm ACTION=go_home PARAMS='{"dry_run": true}'  # plans, sends nothing"""

import importlib
import time
from dataclasses import dataclass, field

import numpy as np
from rammp_arm import GO_TO_SPEED_SCALE, JOINTS

from capability_engine import Action, CapabilityEngine, EngineError, run
from home_arm import body, homing

# flatbot (2026-11-09, the arm 8 in forward on the chair): the hand where the arm was left that day
# (x 0.330, y 0.077, z 0.560 in arm_mount) with the elbow swung out away from the armrest. Every
# planner sphere 11 mm outside the padded chair and user boxes; joint_3 written in (-pi, pi], as the
# arm reports it.
DEFAULT_HOME = [0.4425, -1.5718, -2.9033, -2.1048, -0.7412, -0.9194, 2.1631]
DEFAULT_BODY_LAYERS = "/config/robot/curobo/planner.yaml"  # the planner's own config, the robot's
AT_HOME_RAD = 0.02  # every joint this close to the home already: nothing to do
# All the planning of one go_home ends by here: a plan is only asked for while a whole
# arm.plan_timeout_s still fits before it.
PLAN_BUDGET_S = 60.0
# ... and the whole action, the motion at its speed_scale included, by here: inside the
# orchestrator's 120 s execute_timeout, with room for the driver's last result
ACTION_BUDGET_S = 100.0
# The joints read again before anything is sent: further than this from where the plans start
# (any joint) is an arm that moved while they were made. It is planned for again, once.
START_MOVED_RAD = 0.02
SUCCESSFUL = 0  # a driver goal's error_code when it succeeded
SWUNG = "the plan swings"  # how _judge's refusal of a wide plan begins: asking again gets the same


class _OutOfTime(EngineError):
    """The planning budget is spent: no step or waypoint is tried after it."""


@dataclass
class _Route:
    """Everything one go_home sends, planned: the plan-free step (k, J) or None, the plans after
    it, how they were found, and what was done to get there."""

    free: np.ndarray | None
    plans: list
    how: str
    notes: list = field(default_factory=list)


def _plan_timeout_s():
    """self.arm's wait for one plan (arm.plan_timeout_s), as set at startup."""
    return float(getattr(importlib.import_module("capability_engine.arm"), "PLAN_TIMEOUT_S", 30.0))


class Capability(CapabilityEngine):
    display_name = "Home arm"
    # Not tied to an object: offered standalone, or by an orchestrator that offers it on any
    # selection.
    perception_classes = []
    required_resources = ["arm"]
    actions = [
        Action(
            "go_home",
            "Go home",
            params_schema={
                "type": "object",
                "properties": {"dry_run": {"type": "boolean", "default": False}},
                "additionalProperties": False,
            },
        )
    ]

    def __init__(self):
        super().__init__("home_arm")
        self.declare_parameter("home_joints", DEFAULT_HOME)
        self.declare_parameter("body_layers", DEFAULT_BODY_LAYERS)
        self.declare_parameter("plan_tries", 3)
        self.declare_parameter("tool_link", "arm_tool_frame")
        self.declare_parameter("speed_scale", GO_TO_SPEED_SCALE)

    # -- the action -------------------------------------------------------------------------------

    def execute(self, action_id, params, ctx):
        began = time.monotonic()
        unknown = sorted(set(params) - {"dry_run"})
        if unknown:
            raise EngineError("bad_params", f"unknown parameters {unknown}", recoverable=True)
        dry_run = params.get("dry_run", False)
        if not isinstance(dry_run, bool):
            raise EngineError("bad_params", "'dry_run' must be true or false", recoverable=True)
        home = np.asarray(self.get_parameter("home_joints").value, float)
        if home.shape != (len(JOINTS),):
            raise EngineError(
                "bad_config", f"home_joints has {home.size} values, not {len(JOINTS)}"
            )
        speed = float(self.get_parameter("speed_scale").value)
        if not 0.0 < speed <= 1.0:
            raise EngineError("bad_config", f"speed_scale {speed} is not in (0, 1]")
        tries = max(1, int(self.get_parameter("plan_tries").value))

        ctx.progress("checking", 0.0, "reading where the arm is")
        if self.mock:
            ctx.checkpoint()
            self.arm.go_to_joint_config(home, "home (mock)")
            ctx.progress("done", 1.0, "home (mock)")
            return {"home": True, "moved": False, "mock": True}
        q0 = self._joints()
        off, _ = homing.end_error(q0, home)
        if off <= AT_HOME_RAD:
            ctx.progress("done", 1.0, "already home")
            return {"home": True, "moved": False, "off_rad": round(off, 4)}

        chain, model, no_model = self._kinematics()
        self._plan_until = began + PLAN_BUDGET_S
        ctx.progress("planning", 0.2, "planning the way home around the chair")
        route = self._route(q0, home, tries, ctx, chain, model, no_model)
        ctx.checkpoint()
        if not dry_run:
            now = self._joints()
            moved, j = homing.end_error(now, q0)
            if moved > START_MOVED_RAD:
                note = f"the arm moved {moved:.3f} rad (joint {j + 1}) while it was planned for"
                self.get_logger().warning(f"go_home: {note}: planning again from where it stands")
                q0 = now
                route = self._route(q0, home, tries, ctx, chain, model, no_model)
                route.notes.insert(0, note + ", and was planned for again")
                ctx.checkpoint()
                moved, j = homing.end_error(self._joints(), q0)
                if moved > START_MOVED_RAD:
                    raise EngineError(
                        "arm_moving",
                        f"the arm moved {moved:.3f} rad (joint {j + 1}) while it was planned for, "
                        "twice: it is being moved; nothing sent",
                        recoverable=True,
                    )

        step = homing.stepped(route.free) if route.free is not None else []
        try:
            way = homing.joined(route.plans)
        except ValueError as e:  # each plan is planned from the last one's end: a bug if not
            raise EngineError("no_way_home", f"the plans home do not join ({e})") from e
        step_s = step[-1][0] if step else 0.0
        home_s = way[-1][0] / speed if way else 0.0
        for note in route.notes:
            self.get_logger().info(f"go_home: {note}")
        result = {
            "home": not dry_run,
            "moved": False,
            "stepped_out": bool(step),
            "planned": route.how,
            "notes": route.notes,
            "duration_s": round(step_s + home_s, 2),
            "speed_scale": speed,
        }
        over = time.monotonic() - began + step_s + home_s
        if over > ACTION_BUDGET_S:
            raise EngineError(
                "too_slow",
                f"the way home takes {step_s + home_s:.0f} s at speed_scale {speed:g}, past the "
                f"action's {ACTION_BUDGET_S:.0f} s with the planning: raise speed_scale, or jog "
                "the arm nearer home; nothing sent",
            )
        if dry_run:
            ctx.progress("done", 1.0, f"dry run: home planned {route.how}, nothing sent")
            return dict(result, dry_run=True)

        if step:
            ctx.progress("stepping clear", 0.4, "a short step clear first", pausable=False)
            self._fly(step, "the plan-free step clear", 1.0)
            result["moved"] = True
            stopped, j = homing.end_error(self._joints(), route.free[-1])
            if stopped > START_MOVED_RAD:
                raise EngineError(
                    "motion_failed",
                    f"the plan-free step clear ended {stopped:.3f} rad (joint {j + 1}) short of "
                    "its end, where the way home starts: the arm holds where it stopped",
                )
        ctx.progress("going home", 0.6, "going home", pausable=False)
        self._fly(way, "the way home", speed)
        ctx.progress("done", 1.0, "home")
        return dict(result, moved=True)

    # -- its parts --------------------------------------------------------------------------------

    def _joints(self):
        """The arm's joints now, as the driver reports them (/joint_states, at most
        arm.state_stale_s old: self.arm refuses an older one)."""
        return np.asarray(self.arm.joint_states().position, float)

    def _kinematics(self):
        """(chain, model, why there is no model): the driver's URDF as a kinematic chain (joint
        limits, the tool's height; body_model where it cannot be read, nothing can be checked
        without it), and the planner's chair and user boxes placed on it, None where body_layers
        is "" or they cannot be read (then why, and a start the planner refuses as in collision is
        not stepped out of)."""
        source = str(self.get_parameter("body_layers").value)
        tool = str(self.get_parameter("tool_link").value)
        model, why = None, ""
        if source:
            try:
                model = body.load(source)
            except body.BodyModelError as e:
                why = str(e)
        else:
            why = "body_layers is empty"
        try:
            root = (model.base_link if model is not None else "") or "arm_mount"
            chain = body.Chain(self.arm.robot_description(), root, JOINTS)
            if not chain.has(tool):
                raise body.BodyModelError(f"body model: the URDF has no tool link {tool}")
        except body.BodyModelError as e:
            raise EngineError("body_model", str(e)) from e
        if model is not None:
            try:
                model = model.place(chain, tool)
            except body.BodyModelError as e:
                model, why = None, str(e)
        if model is None:
            self.get_logger().warning(
                f"go_home: no body model ({why}): a start the planner refuses as in the chair's "
                "padding is not stepped out of"
            )
        return chain, model, why

    def _route(self, q0, home, tries, ctx, chain, model, no_model):
        """The _Route home from `q0`. EngineError (start_in_padding, near_limit, no_way_home,
        body_model, bad_config) where there is none; nothing has moved."""
        start, free, notes = q0, None, []
        if model is not None and float(model.depth(q0)[0]) > 0.0:
            # in the padding by the engine's copy of the planner's boxes: the planner, the
            # authority on its own world (another profile's, a pushed one), is asked first
            goal = homing.on_branch(home, q0)
            plan = self._plan(goal, q0)
            if not self._judge(plan, goal) and not homing.near_limits(q0, chain.limits):
                return _Route(None, [plan], "directly", notes)
            depth = float(model.depth(q0)[0])
            path, why = model.escape(q0)
            if path is None:
                raise EngineError("start_in_padding", f"{why}: jog the arm clear by hand")
            free, start = path, path[-1]
            notes.append(
                f"stepped {1000 * depth:.0f} mm out of the padding round the chair and user "
                f"(the largest joint {float(np.abs(path[-1] - q0).max()):.3f} rad)"
            )
        ctx.checkpoint()
        try:
            near = homing.near_limits(start, chain.limits)
            if near:
                z_from = float(chain.tool_z(q0, str(self.get_parameter("tool_link").value))[0])
                step, plans, how = self._off_the_limit(
                    start, home, near, chain, model, ctx, tries, z_from
                )
                free = step if free is None else np.vstack([free, step[1:]])
                notes.append(how)
                how = "from the end of the step off the limit"
            else:
                plans, how = self._plan_home(start, home, tries, ctx, model)
        except EngineError as e:
            if model is None and no_model and "INVALID_START" in str(e).upper():
                raise EngineError(
                    "body_model",
                    f"the planner refuses the start as in collision, and there is no body model "
                    f"to step out of the chair's padding with ({no_model}): jog the arm clear by "
                    "hand",
                ) from e
            raise
        return _Route(free, plans, how, notes)

    def _off_the_limit(self, start, home, near, chain, model, ctx, tries, z_from):
        """(step path, [plans], how): the shortest plan-free step along the joint line to the home
        that the home plans from (_plan_home: tried, then through a waypoint), the tool no lower
        than ESCAPE_DROP_M under its height `z_from` (at the action's start) and no deeper in the
        padding. EngineError("near_limit")."""
        tool = str(self.get_parameter("tool_link").value)
        named = ", ".join(f"joint {j + 1} is {room:.3f} rad from its limit" for j, room in near)
        try:
            steps = homing.limit_steps(start, home, chain.limits)
        except ValueError as e:
            raise EngineError("bad_config", str(e)) from e
        last = "no step up to the cap"
        for f, q1 in steps:
            ctx.checkpoint()
            path = homing.line(start, q1)
            z = chain.tool_z(path, tool)
            if z.min() < z_from - body.ESCAPE_DROP_M:
                drop = 1000 * (z_from - float(z.min()))
                last = f"the line toward the home lowers the tool {drop:.0f} mm"
                break  # a longer step runs the same line: it drops as much or more
            why = model.path_refusal(path) if model is not None else ""
            if why:
                last = f"the line toward the home: {why}"
                break  # a longer step runs the same line on: as deep or deeper
            try:
                plans, how_home = self._plan_home(q1, home, tries, ctx, model)
            except _OutOfTime:
                raise
            except EngineError as e:
                last = str(e)
                continue
            how = (
                f"{named}: a plan-free step {100 * f:.0f}% of the way along the joint line "
                f"first (the largest joint {float(np.abs(q1 - start).max()):.3f} rad), then home "
                f"{how_home}"
            )
            return path, plans, how
        raise EngineError(
            "near_limit",
            f"{named}, too near for the planner to start from, and no plan-free step toward the "
            f"home of up to {homing.LIMIT_MAX_STEP_RAD:.2f} rad lets it plan the home (last: "
            f"{last}): jog the arm clear by hand",
        )

    def _plan_home(self, start, home, tries, ctx, model):
        """([plans], how) to the home from `start`: up to `tries` times directly (stopping at the
        first plan that swings wide: the planner answers alike), then through each waypoint along
        the joint line clear of the padding, its second leg planned from where its first one ends
        (up to homing.WAYPOINT_END_RAD from the waypoint, clear of the padding too), then along
        the joint line in homing.LINE_LEGS equal legs. EngineError("no_way_home") otherwise."""
        last = ""
        goal = homing.on_branch(home, start)
        for k in range(tries):
            ctx.checkpoint()
            plan = self._plan(goal, start)
            last = self._judge(plan, goal)
            if not last:
                return [plan], "directly" if k == 0 else f"directly, at try {k + 1}"
            if last.startswith(SWUNG):
                break
        for f, w in zip(homing.WAYPOINTS, homing.waypoints(start, home), strict=True):
            if model is not None and float(model.depth(w)[0]) > 0.0:
                continue
            ctx.checkpoint()
            first, why = self._to_waypoint(w, start, model)
            if why:
                last = why
                continue
            reached = np.asarray(first.q[-1], float)
            to_home = homing.on_branch(home, reached)
            second = self._plan(to_home, reached)
            last = self._judge(second, to_home)
            if not last:
                return [first, second], f"through a waypoint {100 * f:.0f}% of the way"
        for n in homing.LINE_LEGS:
            ws = homing.waypoints(start, home, [k / n for k in range(1, n)])
            if model is not None and any(float(model.depth(w)[0]) > 0.0 for w in ws):
                continue
            plans, at, why = [], np.asarray(start, float), ""
            for w in ws:
                ctx.checkpoint()
                plan, why = self._to_waypoint(w, at, model)
                if why:
                    break
                plans.append(plan)
                at = np.asarray(plan.q[-1], float)
            if not why:
                to_home = homing.on_branch(home, at)
                plans.append(self._plan(to_home, at))
                why = self._judge(plans[-1], to_home)
            if not why:
                return plans, f"in {n} legs along the joint line (no single waypoint planned)"
            last = why
        legs = "/".join(str(n) for n in homing.LINE_LEGS)
        raise EngineError(
            "no_way_home",
            f"the planner found no way home around the chair, {tries} times directly nor through "
            f"a waypoint along the joint line nor along it in {legs} legs (last: {last}): jog the "
            "arm clear by hand",
        )

    def _to_waypoint(self, w, start, model):
        """(plan, why not) to the waypoint `w` from `start`: the plan may end up to
        homing.WAYPOINT_END_RAD from `w` (the next leg is planned from where it ends), outside
        the padding."""
        plan = self._plan(w, start)
        why = self._judge(plan, w, homing.WAYPOINT_END_RAD)
        if not why and model is not None and float(model.depth(plan.q[-1])[0]) > 0.0:
            why = "the plan to the waypoint ends inside the padding round the chair and user"
        return plan, why

    def _plan(self, goal, start):
        """The planner's plan to `goal` from `start`, while the planning budget still holds a whole
        arm.plan_timeout_s."""
        wait = _plan_timeout_s()
        if time.monotonic() + wait > self._plan_until:
            raise _OutOfTime(
                "no_way_home",
                f"no plan home within {PLAN_BUDGET_S:.0f} s of planning (each plan may wait "
                f"{wait:.0f} s): jog the arm clear by hand",
            )
        return self.arm.plan_to_joints(goal, start)

    @staticmethod
    def _judge(plan, goal, end_rad=homing.PLAN_END_RAD):
        """'' for a plan that succeeded, ends within `end_rad` of `goal` (any joint) and swings no
        joint more than SWING_MARGIN_RAD wider than the straight way there, else why not."""
        if not plan.success:
            return plan.message or "the plan failed"
        if len(plan.q) == 0:
            return "the plan has no samples"
        rad, j = homing.end_error(plan.q[-1], goal)
        if rad > end_rad:
            return (
                f"the plan ends {rad:.3f} rad from its goal (joint {j + 1}): "
                "another arm configuration"
            )
        extra, j = homing.swing(plan.q, goal)
        if extra > homing.SWING_MARGIN_RAD:
            return (
                f"{SWUNG} joint {j + 1} {extra:.2f} rad wider than the straight way there "
                f"(at most {homing.SWING_MARGIN_RAD:.2f})"
            )
        return ""

    def _fly(self, points, what, speed_scale):
        """Send `points` to the driver once, at `speed_scale`; a goal it does not complete fails
        the action."""
        res = self.arm.execute_joint_trajectory(JOINTS, points, speed_scale=speed_scale)
        if res.error_code != SUCCESSFUL:
            raise EngineError(
                "motion_failed",
                f"{what} did not complete ({res.error_code}: {res.error_string}): "
                "the arm holds where it stopped",
            )


def main():
    run(Capability)


if __name__ == "__main__":
    main()
