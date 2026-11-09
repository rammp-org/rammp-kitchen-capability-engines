"""home_arm: one action, `go_home`, that takes the arm to its home pose around the wheelchair and
its user, from wherever it stands.

The move is planned by cuRobo in its world, which on a wheelchair-mounted arm holds the chair, the
seat and the person in it, so the path keeps clear of them. Three things stop a plain plan:

* a start inside the padding round those boxes (clear of the chair itself, but within the planner's
  world_padding) is refused outright. The engine first takes a short plan-free step out of the
  padding, never deeper, the tool held within 5 mm of its height (home_arm.body), and plans from its
  end;
* a joint too near its limit is refused too (START_IN_LIMIT_MARGIN): the engine takes the shortest
  plan-free step along the joint line toward the home that clears it and that the home plans from
  (homing.limit_steps), the tool held and never into the padding;
* the planner is stochastic and sometimes fails, or answers with another arm configuration at the
  same tool pose. The home is asked for up to `plan_tries` times (a plan that does not end at the
  home counts as failed), then through a waypoint along the joint line toward it (home_arm.homing).

Everything is planned before anything moves. Then the plan-free step (if any) and the plan(s) are
each sent to the driver once (`execute_joint_trajectory`): a goal the driver aborts or rejects fails
the action, never retried, and a cancel stops the arm where it is (self.arm cancels the goal first).

    ros2 launch home_arm home_arm.launch.py use_orchestrator:=false
make action CAP=home_arm ACTION=go_home PARAMS='{"dry_run": true}'  # plans, sends nothing"""

import time

import numpy as np
from rammp_arm import JOINTS

from capability_engine import Action, CapabilityEngine, EngineError, run
from home_arm import body, homing

# flatbot (2026-11-09, the arm 8 in forward on the chair): the hand where the arm was left that day
# (x 0.330, y 0.077, z 0.560 in arm_mount) with the elbow swung out away from the armrest. Every
# planner sphere 11 mm outside the padded chair and user boxes; joint_3 written in (-pi, pi], as the
# arm reports it.
DEFAULT_HOME = [0.4425, -1.5718, -2.9033, -2.1048, -0.7412, -0.9194, 2.1631]
DEFAULT_BODY_LAYERS = "/config/robot/curobo/planner.yaml"  # the planner's own config, the robot's
AT_HOME_RAD = 0.02  # every joint this close to the home already: nothing to do
# all the planning of one go_home stops here (each plan may wait arm.plan_timeout_s): the action
# stays well inside the orchestrator's 120 s execute_timeout
PLAN_BUDGET_S = 60.0
SUCCESSFUL = 0  # a driver goal's error_code when it succeeded


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

    # -- the action -------------------------------------------------------------------------------

    def execute(self, action_id, params, ctx):
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
        tries = int(self.get_parameter("plan_tries").value)

        ctx.progress("checking", 0.0, "reading where the arm is")
        if self.mock:
            ctx.checkpoint()
            self.arm.go_to_joint_config(home, "home (mock)")
            ctx.progress("done", 1.0, "home (mock)")
            return {"home": True, "moved": False, "mock": True}
        q0 = np.asarray(self.arm.raw_joints(), float)
        off, _ = homing.end_error(q0, home)
        if off <= AT_HOME_RAD:
            ctx.progress("done", 1.0, "already home")
            return {"home": True, "moved": False, "off_rad": round(off, 4)}

        chain, model = self._kinematics()
        start, free, notes = q0, None, []
        if model is not None:
            depth = float(model.depth(q0)[0])
            if depth > 0.0:
                path, why = model.escape(q0)
                if path is None:
                    raise EngineError("start_in_padding", f"{why}: jog the arm clear by hand")
                free, start = path, path[-1]
                notes.append(
                    f"stepped {1000 * depth:.0f} mm out of the padding round the chair and user "
                    f"(the largest joint {float(np.abs(path[-1] - q0).max()):.3f} rad)"
                )
        ctx.checkpoint()

        ctx.progress("planning", 0.2, "planning the way home around the chair")
        self._deadline = time.monotonic() + PLAN_BUDGET_S
        near = homing.near_limits(start, chain.limits)
        if near:
            step, plans, how = self._off_the_limit(start, home, near, chain, model, ctx)
            free = step if free is None else np.vstack([free, step[1:]])
            notes.append(how)
            how = "from the end of the step off the limit"
        else:
            plans, how = self._plan_home(start, home, tries, ctx, model)
        for note in notes:
            self.get_logger().info(f"go_home: {note}")
        result = {
            "home": not dry_run,
            "moved": False,
            "stepped_out": free is not None,
            "planned": how,
            "notes": notes,
            "duration_s": round(float(sum(p.t[-1] for p in plans)), 2),
        }
        ctx.checkpoint()
        if dry_run:
            ctx.progress("done", 1.0, f"dry run: home planned {how}, nothing sent")
            return dict(result, dry_run=True)

        if free is not None:
            ctx.progress("stepping clear", 0.4, "a short step clear first", pausable=False)
            self._fly(homing.timed_path(free), "the plan-free step clear")
            result["moved"] = True
        ctx.progress("going home", 0.6, "going home", pausable=False)
        self._fly(homing.joined(plans), "the way home")
        ctx.progress("done", 1.0, "home")
        return dict(result, moved=True)

    # -- its parts --------------------------------------------------------------------------------

    def _kinematics(self):
        """(chain, model): the driver's URDF as a kinematic chain (joint limits, the tool's height)
        and the planner's chair and user boxes placed on it, None where body_layers is ""."""
        source = str(self.get_parameter("body_layers").value)
        tool = str(self.get_parameter("tool_link").value)
        try:
            model = body.load(source) if source else None
            root = (model.base_link if model is not None else "") or "arm_mount"
            chain = body.Chain(self.arm.robot_description(), root, JOINTS)
            if not chain.has(tool):
                raise body.BodyModelError(f"body model: the URDF has no tool link {tool}")
            if model is None:
                self.get_logger().warning(
                    "body_layers is empty: a start inside the chair's padding is not stepped out of"
                )
                return chain, None
            return chain, model.place(chain, tool)
        except body.BodyModelError as e:
            raise EngineError("body_model", str(e)) from e

    def _off_the_limit(self, start, home, near, chain, model, ctx):
        """(step path, [plan], how): the shortest plan-free step along the joint line to the home
        that the home plans from (homing.limit_steps), the tool no lower than ESCAPE_DROP_M under
        its start on the way and no deeper in the padding. EngineError("near_limit")."""
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
            if z.min() < z[0] - body.ESCAPE_DROP_M:
                last = f"the line toward the home lowers the tool {1000 * (z[0] - z.min()):.0f} mm"
                break  # a longer step runs the same line: it drops as much or more
            why = model.path_refusal(path) if model is not None else ""
            if why:
                last = f"the line toward the home: {why}"
                break  # a longer step runs the same line on: as deep or deeper
            goal = homing.on_branch(home, q1)
            plan = self._plan(goal, q1)
            last = self._judge(plan, goal)
            if not last:
                how = (
                    f"{named}: a plan-free step {100 * f:.0f}% of the way along the joint line "
                    f"first (the largest joint {float(np.abs(q1 - start).max()):.3f} rad)"
                )
                return path, [plan], how
        raise EngineError(
            "near_limit",
            f"{named}, too near for the planner to start from, and no plan-free step toward the "
            f"home of up to {homing.LIMIT_MAX_STEP_RAD:.2f} rad lets it plan the home (last: "
            f"{last}): jog the arm clear by hand",
        )

    def _plan_home(self, start, home, tries, ctx, model):
        """([plans], how) to the home from `start`: up to `tries` times directly, then through each
        waypoint along the joint line clear of the padding. EngineError("no_way_home") otherwise."""
        last = ""
        goal = homing.on_branch(home, start)
        for k in range(tries):
            ctx.checkpoint()
            plan = self._plan(goal, start)
            last = self._judge(plan, goal)
            if not last:
                return [plan], "directly" if k == 0 else f"directly, at try {k + 1}"
        for f, w in zip(homing.WAYPOINTS, homing.waypoints(start, home), strict=True):
            if model is not None and float(model.depth(w)[0]) > 0.0:
                continue
            ctx.checkpoint()
            first = self._plan(w, start)
            last = self._judge(first, w)
            if last:
                continue
            to_home = homing.on_branch(home, w)
            second = self._plan(to_home, w)
            last = self._judge(second, to_home)
            if not last:
                return [first, second], f"through a waypoint {100 * f:.0f}% of the way"
        raise EngineError(
            "no_way_home",
            f"the planner found no way home around the chair, {tries} times directly nor through "
            f"a waypoint along the joint line (last: {last}): jog the arm clear by hand",
        )

    def _plan(self, goal, start):
        """The planner's plan to `goal` from `start`, while the planning budget lasts."""
        if time.monotonic() > self._deadline:
            raise EngineError(
                "no_way_home",
                f"no plan home within {PLAN_BUDGET_S:.0f} s of planning: jog the arm clear by hand",
            )
        return self.arm.plan_to_joints(goal, start)

    @staticmethod
    def _judge(plan, goal):
        """'' for a plan that succeeded and ends at `goal`, else why not."""
        if not plan.success:
            return plan.message or "the plan failed"
        if len(plan.q) == 0:
            return "the plan has no samples"
        rad, j = homing.end_error(plan.q[-1], goal)
        if rad > homing.PLAN_END_RAD:
            return (
                f"the plan ends {rad:.3f} rad from its goal (joint {j + 1}): "
                "another arm configuration"
            )
        return ""

    def _fly(self, points, what):
        """Send `points` to the driver once; a goal it does not complete fails the action."""
        res = self.arm.execute_joint_trajectory(JOINTS, points)
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
