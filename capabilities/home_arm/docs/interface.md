# home_arm Capability — Interface

**Rung:** ad-hoc (declared by this module, not standardized)
**Owner:** @ChrissCox
**Tier:** experimental

## What it does

One action, `go_home`: the arm to its home pose (`home_joints`) from wherever it
stands, around the wheelchair and the person in it. cuRobo plans the move in its
world, which on a wheelchair-mounted arm holds the chair, the seat and the user,
so the path keeps clear of them; the driver executes it. Four things stop a
plain plan, and the action handles each before anything moves:

1. **A start inside the padding round the chair and user.** The planner pads
   every box by its `world_padding` and refuses a start inside the padding, even
   when the arm is clear of the chair itself (left near the user by a stop or a
   jog). The engine reads the planner's own boxes and spheres (`body_layers`, the
   robot's planner config, padded by its `planner.world_padding`). Where they put
   the start in the padding, the planner is asked first (it is the authority on
   its own world, which another profile or a pushed world may change); where it
   refuses the start itself (a plan that fails past the start, swings or ends
   elsewhere is a planner that takes it, and the home is then asked for from
   there as from any start), the engine finds a short plan-free step out of the padding in joint
   space (never deeper, the tool never more than 5 mm lower than it started, no
   joint more than 0.6 rad), and plans from its end. Where there is none, the action fails
   `start_in_padding` before anything moves. Where the files cannot be read, the
   action goes on without them, and fails `body_model` only if the planner then
   refuses the start as in collision, or a joint stands too near its limit (the
   plan-free step off it could not be checked against the chair and user). With
   `body_layers` empty there is no chair to check against: that step is held to
   the tool's height alone.
1. **A joint too near its limit.** The planner refuses a start inside its
   joint-limit margin (`START_IN_LIMIT_MARGIN`, 0.10 rad on flatbot) and often
   fails just outside it. A start with a joint nearer than 0.18 rad takes the
   shortest plan-free step along the joint line toward the home that clears
   0.12 rad and that the home plans from (0.05 rad longer each try, at most 0.35
   rad, the home asked for from each as below), the tool never lower than 5 mm
   under its height at the start and never into the padding on the way. Where
   none plans, the action fails `near_limit`.
1. **A plan that fails or ends elsewhere.** The home is asked for up to
   `plan_tries` times (a plan that does not end at the home, the planner's other
   arm configuration at the same tool pose, counts as failed), then through a
   waypoint 50, 25 or 75 % of the way along the joint line toward it, clear of the
   padding, its second leg planned from where its first one ends (the plan to a
   waypoint may end up to 0.35 rad from it, outside the padding: the planner's
   pose fallback lands a little along the arm's self-motion), then along that
   joint line in 4, then 6, equal legs: an arm whose elbow must straighten and
   fold the other way swings wide on long legs and follows the line on short
   ones. Where none plans, the action fails `no_way_home`.
1. **A plan that swings wide.** A plan that ends at the home can still take a
   joint far further than the straight way there: on flatbot (2026-11-10) every
   plan home from one start took joint 1 0.9 rad out past it and back, the hand
   toward the user. A plan that takes any joint more than 0.35 rad wider than the
   straight way (box_opening's trajectory sanity gate) counts as failed. Asking
   again gets the same plan, so the engine goes on to the waypoints.

A plan is asked for only while a whole `arm.plan_timeout_s` still fits in 60 s of
planning, and a way home that would take the action past 100 s at `speed_scale`
fails `too_slow`: well inside the orchestrator's 120 s timeout. Time the action
spends paused is no part of the 60 s of planning; where a pause leaves the way
home no room in the 100 s, the action fails `paused_too_long` before anything is
sent (whatever cancels a long action counts its pauses too). A planner that does not answer (no action
server, no result in `arm.plan_timeout_s`) fails `planner_unavailable` at the
first plan, not `no_way_home` after every waypoint.

Before anything is sent the arm's joints are read again (`/joint_states`, at
most `arm.state_stale_s` old): an arm that moved more than 0.02 rad while it was
planned for is planned for again from where it stands, once, and one still moving
fails `arm_moving`. Then the plan-free step (if any) and the plan(s) are each sent
to the driver once (`/execute_joint_trajectory`, position control), the step on
its own slow minimum-jerk timing (at most 0.15 rad/s), the plans at
`speed_scale`, each goal under the driver's divergence guard (`arm.path_tol_rad`
per joint: a goal that names no tolerance runs with the guard off). The arm is
read once more when the way home ends, and the action succeeds only where every
joint is within `arm.go_to_track_deg` of the home: the driver ends a goal when
its clock runs out, wherever the arm is. A step that ends short of where the way home starts, or a goal the
driver aborts or rejects, fails the action (`motion_failed`), never retried; a
cancel stops the arm where it is. `dry_run: true` plans and reports, and sends
nothing.

A pause never stops a goal half way (the flying steps are reported not pausable).
One asked for before a goal is sent is taken there, and one asked for while the
step clear flies is taken at its end, before the way home is sent. After either
the arm is read again: moved more than 0.02 rad from where the next goal starts,
the action fails `arm_moving` and that goal is not sent; and the time the way
home still needs is checked against the 100 s again (`paused_too_long`).

Without it, an arm left anywhere but home is recovered by hand: the planner
refuses starts in the chair's padding, and a home planned without the chair can
go through the user.

## Publishes

| Topic              | Type              | Rate | Meaning                                                                                                   |
| ------------------ | ----------------- | ---- | --------------------------------------------------------------------------------------------------------- |
| `/home_arm/status` | `std_msgs/String` | 1 Hz | Health topic: JSON `state` (idle/executing/paused), `action`, `instance_id`, `mode`, `registered`, `mock` |

## Subscribes

| Topic                    | Type                                    | Required | Meaning                                                                      |
| ------------------------ | --------------------------------------- | -------- | ---------------------------------------------------------------------------- |
| `/joint_states`          | `sensor_msgs/JointState`                | yes      | Where the arm starts (through `self.arm`, during the action only)            |
| `/ee_state`              | `rammp_arm_interfaces/EeState`          | yes      | `self.arm` waits for it before a state read or a command; not otherwise read |
| `/gripper_state`         | `rammp_arm_interfaces/GripperState`     | yes      | The same                                                                     |
| `/control_status`        | `rammp_common_interfaces/ControlStatus` | yes      | The e-stop and who holds the arm, read by `self.arm` before a goal is sent   |
| `/arm/robot_description` | `std_msgs/String`                       | yes      | The driver's URDF: places the planner's spheres (one-shot, latched)          |

All of them through `self.arm`, opened when the action first needs them and closed
as it ends. It calls the cuRobo planner's `/rammp_curobo/plan_to_joints` action
(plans only) and the driver's `/execute_joint_trajectory` action, both through
`self.arm`, and reads the planner's config files from `/config/robot` (the robot
config, mounted read-only).

## Orchestrator interface

Served under `/home_arm/`, types from `rammp_orchestrator_interfaces`:

| Name                       | Type                   | Meaning                                       |
| -------------------------- | ---------------------- | --------------------------------------------- |
| `/home_arm/activate`       | `srv/Activate`         | Which actions are available for the selection |
| `/home_arm/execute_action` | `action/ExecuteAction` | Run an action; progress as feedback           |
| `/home_arm/control`        | `srv/Control`          | Pause, resume, cancel the running action      |
| `/home_arm/hibernate`      | `srv/Hibernate`        | Drop the selection                            |

| Action    | Params                                                | Result payload                                                                                                                                                                                                                                                       |
| --------- | ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `go_home` | `dry_run` (bool, default false: plan only, send none) | `{"home", "moved", "stepped_out", "planned", "notes", "duration_s", "speed_scale"}`, with `"off_rad"` (how near the home the arm was read) once flown, or `{"home": true, "moved": false}` when already home; the mock answers with the same keys and `"mock": true` |

| Error code            | Recoverable | Meaning                                                                                                                                                                                                                                                     |
| --------------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `standalone_only`     | no          | The engine runs under an orchestrator (`use_orchestrator` true), which cannot dispatch `go_home`: the goal was sent to it directly, outside the orchestrator's `arm` lock; nothing planned or sent                                                          |
| `bad_params`          | yes         | Params that are not a JSON object, a parameter other than `dry_run`, or `dry_run` not a boolean                                                                                                                                                             |
| `arm_state`           | yes         | The driver's `/joint_states` or URDF did not come, or is stale; nothing planned or sent                                                                                                                                                                     |
| `bad_config`          | no          | `home_joints` is not 7 values, or `speed_scale` not in \[0.01, 1\] (the driver rejects a goal sent at less than 0.01)                                                                                                                                       |
| `body_model`          | no          | The driver's URDF cannot be read; or the planner's config cannot be read and it is needed: the planner refuses the start as in collision, or a joint is too near its limit (the step off it could not be checked against the chair and user); nothing moved |
| `start_in_padding`    | no          | No step out of the chair's padding keeps the tool up; nothing moved: jog it clear                                                                                                                                                                           |
| `near_limit`          | no          | No step off a joint limit lets the home plan; nothing moved: jog it clear                                                                                                                                                                                   |
| `no_way_home`         | no          | No plan home, directly or through a waypoint; nothing moved: jog it clear                                                                                                                                                                                   |
| `planner_unavailable` | yes         | The planner did not answer a plan request (not running, or no result in time); nothing moved                                                                                                                                                                |
| `too_slow`            | no          | The way home at `speed_scale` would take the action past 100 s; nothing moved, or (found at a pause after the step clear) the way home not sent                                                                                                             |
| `paused_too_long`     | yes         | A pause left the way home no room in the action's 100 s; nothing sent, or after the step clear the way home not sent: run it again                                                                                                                          |
| `arm_moving`          | yes         | The arm moved while it was planned for, twice, or during a pause before a goal was sent; nothing sent, or after the step clear the way home not sent                                                                                                        |
| `motion_failed`       | no          | The driver refused or did not complete a goal (an e-stop, control taken and the divergence guard included), the arm ended away from the home, or its joints could not be read after a goal flew; it holds where it stopped                                  |

It names no perception classes, because `go_home` is not about an object; an
object's selection activates the engines of its class, so no selection offers
`go_home` and the orchestrator never dispatches it.

That is a deliberate, stated limit: `go_home` is standalone only, until the
orchestrator can dispatch an action that belongs to no object. Both fragments and
the launch file run the engine with `use_orchestrator:=false` (it registers with
nothing), the goal sent with `make action` or `ros2 action send_goal`, on a robot
where nothing else is commanding the arm. With `use_orchestrator:=true` it
registers with `/rammp_orchestrator/register_engine` (retrying until the
orchestrator is up) and refuses `go_home` (`standalone_only`, a dry run too),
and `activate` reports it unavailable with that reason: any goal an orchestrated
engine gets was sent to it directly, outside the orchestrator's `arm` lock, so
the orchestrator could dispatch another engine's arm action while the arm goes
home.

## Parameters

ROS parameters, declared with defaults under `params:` in the real fragment. A
deployment sets one per robot with an `overrides:` entry in its profile; sheppy
passes them as a params file.

| Name               | Default                                                         | Meaning                                                                                                                                                    |
| ------------------ | --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `use_orchestrator` | `true` (the fragments and the launch file set `false`)          | `false` runs standalone, the only mode `go_home` runs in: no registration, drive it by CLI; `true` registers, and `go_home` is refused (`standalone_only`) |
| `mock`             | `false`                                                         | `true` skips hardware (the mock alternative sets it)                                                                                                       |
| `home_joints`      | `[0.4425, -1.5718, -2.9033, -2.1048, -0.7412, -0.9194, 2.1631]` | The home (rad, J1..J7, continuous joints in (-pi, pi\]): flatbot's, the arm 8 in forward on the chair (2026-11-09)                                         |
| `body_layers`      | `"/config/robot/curobo/planner.yaml"`                           | The planner config whose boxes and spheres the step out uses; `""` turns the step out off                                                                  |
| `plan_tries`       | `3`                                                             | How many times the home is asked for directly before the waypoints                                                                                         |
| `tool_link`        | `"arm_tool_frame"`                                              | The link whose height the step out holds                                                                                                                   |
| `speed_scale`      | `0.3`                                                           | The driver's speed_scale for the plans home (0.01..1, the go_to goals' default); the step clear has its own slow timing                                    |

`self.arm` reads these `arm.*` parameters once at startup and refuses changes
afterwards. They are set per robot like any other; `arm.mock_urdf` names the
URDF `make_sim()` loads in mock runs (the real driver supplies its own).
`arm.sender_id` is the one left out of the real fragment: its default is the
engine id, so each capability sends its goals under its own name.

| Name                           | Default                                            | Meaning                                                    |
| ------------------------------ | -------------------------------------------------- | ---------------------------------------------------------- |
| `arm.tool_z`                   | `0.12`                                             | Tool frame offset along the flange z (m)                   |
| `arm.dt`                       | `0.02`                                             | Trajectory sample spacing (s)                              |
| `arm.joint_speed`              | `0.85`                                             | Peak joint speed of joint moves (rad/s)                    |
| `arm.cart_speed`               | `0.16`                                             | Peak tool speed of Cartesian paths (m/s)                   |
| `arm.contact_joint_speed`      | `0.5`                                              | Peak joint speed of contact moves (rad/s)                  |
| `arm.contact_cart_speed`       | `0.09`                                             | Peak tool speed of contact moves (m/s)                     |
| `arm.min_duration_s`           | `1.0`                                              | Shortest trajectory duration (s)                           |
| `arm.cart_step_m`              | `0.02`                                             | Longest Cartesian step before IK (m)                       |
| `arm.cart_step_deg`            | `5.0`                                              | Largest Cartesian turn per step before IK (deg)            |
| `arm.rot_equiv_m`              | `0.15`                                             | Travel a 1 rad turn counts as when timing a path (m)       |
| `arm.max_knot_jump_deg`        | `20.0`                                             | Largest joint change between Cartesian knots (deg)         |
| `arm.path_tol_rad`             | `0.15`                                             | Per-joint tracking tolerance of trajectories (rad)         |
| `arm.gripper_force`            | `0.5`                                              | Default gripper force (0..1)                               |
| `arm.gripper_speed`            | `0.5`                                              | Default gripper speed (0..1)                               |
| `arm.frame`                    | `"arm_mount"`                                      | Frame of go_to_ee_pose goals and /ee_state                 |
| `arm.sender_id`                | the engine id (RammpArm standalone: `"microwave"`) | sender_id on goals (the engine id under the engine)        |
| `arm.go_to_speed_scale`        | `0.3`                                              | speed_scale of go_to goals (0.01..1)                       |
| `arm.go_to_timeout_s`          | `120.0`                                            | go_to planning plus execution limit (s)                    |
| `arm.go_to_track_m`            | `0.01`                                             | go_to_ee_pose: largest tool error at the end (m)           |
| `arm.go_to_track_deg`          | `2.0`                                              | go_to_joint_config: largest end joint error (deg)          |
| `arm.max_ik_err`               | `0.02`                                             | IK position error above which a pose is unreachable (m)    |
| `arm.max_step_jump_deg`        | `20.0`                                             | Largest joint change of a planned step (deg)               |
| `arm.j6_guard_deg`             | `115.0`                                            | Largest absolute J6 the sim accepts (deg)                  |
| `arm.j4_guard_deg`             | `144.0`                                            | Largest absolute J4 the sim accepts (deg)                  |
| `arm.max_continuous_delta_deg` | `170.0`                                            | Largest short-way step of a free joint (deg)               |
| `arm.keep_out_err`             | `9.0`                                              | IK error reported for a keep-out solution (m)              |
| `arm.ik_max_iterations`        | `200`                                              | PyBullet IK iterations per solve                           |
| `arm.clearance_query_m`        | `0.5`                                              | clearance() ignores bodies farther than this (m)           |
| `arm.fingertip_open_rad`       | `0.0`                                              | Sim finger joint angle with the gripper open (rad)         |
| `arm.finger_closed_rad`        | `0.7`                                              | Sim finger joint angle with the gripper closed (rad)       |
| `arm.state_wait_s`             | `10.0`                                             | First arm call: wait for the driver's state (s)            |
| `arm.state_stale_s`            | `0.5`                                              | Motion state older than this is an error (s)               |
| `arm.diagnostics_stale_s`      | `3.0`                                              | /diagnostics older than this is an error (s)               |
| `arm.goal_accept_s`            | `5.0`                                              | Wait for the driver to accept or reject a goal (s)         |
| `arm.result_margin_s`          | `30.0`                                             | A trajectory's result may come this long after its end (s) |
| `arm.poll_s`                   | `0.01`                                             | Poll period of a running motion (s)                        |
| `arm.cancel_settle_s`          | `0.4`                                              | Wait for a cancelled goal to settle (s)                    |
| `arm.halt_status_wait_s`       | `0.2`                                              | After a HALTED goal, wait for an e-stop status (s)         |
| `arm.planner_wait_s`           | `5.0`                                              | Wait for the cuRobo planner's action server (s)            |
| `arm.go_to_settle_s`           | `0.3`                                              | After a go_to goal, let the state catch up (s)             |
| `arm.gripper_period_s`         | `0.02`                                             | gripper() setpoint period (s)                              |
| `arm.gripper_moved`            | `0.002`                                            | gripper(): position change that counts as moving (0..1)    |
| `arm.gripper_judge_after_s`    | `0.5`                                              | gripper(): judge the fingers still only after (s)          |
| `arm.gripper_still_s`          | `0.3`                                              | gripper(): still this long means settled (s)               |
| `arm.plan_timeout_s`           | `30.0`                                             | Wait for a cuRobo plan (s)                                 |
| `arm.service_wait_s`           | `5.0`                                              | Wait for a service server to come up (s)                   |
| `arm.service_call_s`           | `10.0`                                             | Wait for a service's answer (s)                            |
| `arm.stream_timeout_s`         | `0.1`                                              | stream(): default setpoint timeout of a session (s)        |
| `arm.publisher_match_s`        | `1.0`                                              | Wait for a new publisher to reach the driver (s)           |
| `arm.close_stream_wait_s`      | `1.0`                                              | Wait for /close_stream's answer (s)                        |
| `arm.mock_joints`              | `[0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57]`        | Mock mode: the joints it starts at (rad, J1..J7)           |
| `arm.mock_urdf`                | `""`                                               | Mock mode: the URDF file make_sim() loads (empty: none)    |

## Hardware

The arm, through the driver and the planner (`self.arm`). The robot config is
mounted read-only at `/config/robot` (`rammp-alternative.yaml`), as the planner
mounts it, for the planner's own boxes and spheres.

## Alternatives

| id              | What it is                                                                         |
| --------------- | ---------------------------------------------------------------------------------- |
| `home_arm_real` | The module proper                                                                  |
| `home_arm_mock` | Same interface, no hardware — for developing against this module without owning it |
