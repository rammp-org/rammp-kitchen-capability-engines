# rammp-kitchen-capability-engines

Shared codebase for all kitchen tasks implemented on RAMMP, for the December
2026 demos.

Each capability (button press, door opening, ...) is its own RAMMP module: one
directory under `capabilities/`, one image at
`ghcr.io/rammp-org/rammp-kitchen-capability-engines/<capability>`, one real and
one mock sheppy alternative. This repo enforces the
[RAMMP module standards](https://github.com/rammp-org/rammp-module-template)
for every capability, automatically, on every PR.

## Quickstart

Prerequisites: [`uv`](https://docs.astral.sh/uv/), `pre-commit`, Docker. Images
are arm64 (`rammp-base` is Jetson-only), so build on the Jetson or let CI do it.

```bash
pre-commit install            # once per clone; see CONTRIBUTING.md
make new CAP=button_press     # start a capability from the example
make check                    # repo rules + sheppy's own loader, all capabilities
make build CAP=button_press   # docker build (arm64 host)
make smoke CAP=button_press   # publishes what it declares, exits on SIGTERM
make lint                     # pre-commit run --all-files
```

## Standalone or with the orchestrator

Every capability is a `capability_engine.CapabilityEngine`: it implements the
[RAMMP orchestrator](https://github.com/rammp-org/rammp-orchestrator)'s engine
contract (register, activate, execute_action, control, hibernate) and only
writes its own `execute()`. Every capability, and `capability_engine` itself,
is an `ament_python` package, so it builds natively with colcon or in its image,
and both run the same launch file with the same two switches:
`use_orchestrator:=false` (standalone) and `mock:=true` (no hardware).

### Natively with colcon (the RAMMP hybrid workflow)

Clone into the workspace from [developer setup](https://rammp-org.github.io/setup/),
next to `rammp-interfaces-ros2` (at a version that carries
`rammp_orchestrator_interfaces`, currently its `dev` branch):

```bash
cd ~/ros2_ws/src
git clone git@github.com:rammp-org/rammp-kitchen-capability-engines.git
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -y
colcon build --packages-up-to example
source install/setup.bash

ros2 launch example example.launch.py                          # registers with the orchestrator; waits for it if absent
ros2 launch example example.launch.py use_orchestrator:=false  # standalone; add mock:=true to skip hardware
```

Edit, `colcon build --packages-select example`, relaunch. No image involved.
To have sheppy start your build, add a local `kind: launch_file` alternative as in
[hybrid workflow](https://rammp-org.github.io/development/hybrid/) (an uncommitted
edit to your manifest).

### In the image

```bash
make build CAP=example                # arm64 host
make run CAP=example                  # with the orchestrator
make run CAP=example STANDALONE=1     # standalone; MOCK=1 skips hardware
```

### Sending it an action

Either way, drive a running capability as the orchestrator would:

```bash
make action CAP=example ACTION=count PARAMS='{"to": 5}'
# or natively:
ros2 action send_goal --feedback /example/execute_action rammp_orchestrator_interfaces/action/ExecuteAction \
  '{engine_id: example, action_id: count, params_json: "{\"to\": 5}"}'
ros2 service call /example/control rammp_orchestrator_interfaces/srv/Control '{command: 0}'  # 0 pause, 1 resume, 2 cancel
ros2 topic echo /example/status
```

Standalone, the orchestrator calls a capability makes (`ctx.request_intent`,
`ctx.write_world_state`, `ctx.update_object`) are logged and skipped. Arm control
arbitration (`AcquireControl`) is not wired in yet: the orchestrator does not
call it today.

To run it on the robot, transcribe the capability's two fragments into
[`rammp-deployments`](https://github.com/rammp-org/rammp-deployments)'s
`december_2026/sheppy-manifest.yaml` as a node with both alternatives.

## API

A capability subclasses `CapabilityEngine`, implements `execute()`, and controls
the arm through `self.arm`. `capabilities/example` is a working one. The rules
for using it (cancel, subscriptions, the sim, parameters) are in
[Using `self.arm`](CONTRIBUTING.md#using-selfarm). Everything below is importable
as `from capability_engine import ...` unless a module is named.

### `CapabilityEngine`

Class attributes, read at registration:

| Attribute            | Meaning                                                                          |
| -------------------- | -------------------------------------------------------------------------------- |
| `display_name`       | Shown to the user                                                                |
| `perception_classes` | The engine is activated when the user selects an object of one of these          |
| `required_resources` | What the orchestrator locks while an action runs: `"arm"`, `"base"`, `"gripper"` |
| `state_reads`        | World-state keys the engine reads                                                |
| `state_writes`       | World-state keys the engine writes                                               |
| `intents_handled`    | Intents (e.g. `MOVE_BASE`) other engines may delegate to this one                |
| `actions`            | `list[Action]`                                                                   |

`Action(action_id, label, params_schema={}, sticky_followup=False, perception_classes=[])`:
one action the engine offers; `params_schema` is JSON Schema.

| Method / attribute                                           | Meaning                                                                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| `__init__(engine_id, **node_kwargs)`                         | Call `super().__init__("<engine_id>")`; the node name is the engine id                                  |
| `execute(action_id, params, ctx) -> dict`                    | **Override.** Run one action and return its result payload. Raise `EngineError` to fail it              |
| `availability(instance_id, world_state)`                     | Override to say which actions can run on this object now. Default: all                                  |
| `hibernate()`                                                | Override to drop per-selection state. Default: nothing                                                  |
| `create_action_subscription(msg_type, topic, callback, qos)` | `Node.create_subscription` for the running action only; destroyed when it ends. Call inside `execute()` |
| `create_action_publisher(msg_type, topic, qos)`              | The same, for a publisher                                                                               |
| `create_action_tf_buffer()`                                  | A `tf2_ros.Buffer` kept current for the running action only. Use it in place of a `TransformListener`   |
| `self.arm`                                                   | The arm: see below                                                                                      |
| `self.mock`                                                  | The `mock` parameter: no hardware                                                                       |
| `self.instance_id`                                           | The object the engine was activated on                                                                  |
| `stop()`                                                     | Stop accepting goals and cancel the running one. `run()` calls it on SIGTERM / SIGINT                   |

`run(EngineClass)` is a capability's entry point: it spins until SIGTERM or
Ctrl-C. `EngineError(error_code, message="", recoverable=False)` fails the action
with that code.

### `ctx` (`Context`)

| Method                                               | Meaning                                                                                  |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `checkpoint()`                                       | Call between steps: blocks while paused, raises `Cancelled` once the action is cancelled |
| `cancelled() -> bool`                                | Whether the action has been cancelled. Never blocks                                      |
| `progress(phase, pct, message="", pausable=True)`    | Report progress to the orchestrator                                                      |
| `request_intent(intent_type, params=None) -> Result` | Ask the orchestrator to delegate an intent (e.g. `MOVE_BASE`) and wait for its result    |
| `write_world_state(**updates) -> bool`               | Merge into the world state (keys `gripper`, `doors`, `objects`, `arm`, `base_pose`)      |
| `update_object(instance_id, **fields) -> bool`       | Update one object's fields in the world state                                            |

### `self.arm` (`ArmInterface`)

Usable only inside `execute()`; a call outside raises `NotInAction`. It has two
layers. They share the same connection to the arm driver, but report failure
differently.

**The students' `RammpArm` API.** Motions return `True`, or `False` when refused
or failed (an e-stop included), with the reason in `self.arm.last_error`.

| Method                                                                                                                                   | Meaning                                                                                                                    |
| ---------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `get_state() -> dict`                                                                                                                    | `position` (joints wrapped to ±π), `velocity`, `effort`, `ee_pos` (tool frame, `[x, y, z, qx, qy, qz, qw]`), `gripper_pos` |
| `raw_joints()`                                                                                                                           | Measured joints, not wrapped: where a cuRobo plan must start                                                               |
| `move_joints(knots, cart_lengths=None, speed_scale=1.0, impedance=None, monitor=None)`                                                   | Through joint `knots`. `monitor()` returning True stops the move and sets `monitor_stopped`                                |
| `move_cartesian(traj, impedance=None, speed_scale=1.0)`                                                                                  | Straight-line segments through `[(pos, quat_xyzw), ...]` in the base frame                                                 |
| `go_to_ee_pose(pos, quat, label=..., speed_scale=None, *, orientation_hold=HOLD_NONE, control_mode=POSITION, gains=None, feedback=None)` | cuRobo plans the tool to a pose, the driver executes it                                                                    |
| `go_to_joint_config(q, label=..., speed_scale=None, *, control_mode=POSITION, gains=None, feedback=None)`                                | The same, to joints `q`                                                                                                    |
| `go_to_preset(name, speed_scale=None, *, control_mode=POSITION, gains=None, feedback=None)`                                              | The same, to one of the driver's named presets                                                                             |
| `gripper(position, timeout_s=4.0, force=None, *, speed=None)`                                                                            | `position` 0 (open) to 1 (closed); returns when the fingers stop                                                           |
| `execute_command(cmd)`                                                                                                                   | A `cornell_commands` command (`JointCommand`, `CartesianTrajectoryCommand`, `Open/CloseGripperCommand`)                    |
| `last_error`, `last_path`, `monitor_stopped`                                                                                             | The last refusal, the last move's measured tool path `[(t, x, y, z)]`, whether its monitor stopped it                      |

**The driver API.** Reads and motions raise `ArmError` subclasses instead of
returning `False`.

| Method                                                                                                                                                                                                                   | Meaning                                                                                                     |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| `joint_states()`, `ee_state()`, `gripper_state()`                                                                                                                                                                        | The latest message as a dataclass with `age_s`; `StateTimeout` when none arrives or it is stale             |
| `control_status()`, `stream_status()`, `diagnostics()`                                                                                                                                                                   | The same for `/control_status`, `/stream_status`, `/diagnostics`                                            |
| `not_ready() -> str \| None`                                                                                                                                                                                             | Why the arm must not be commanded now (e.g. e-stopped), or None                                             |
| `robot_description() -> str`                                                                                                                                                                                             | The driver's URDF                                                                                           |
| `execute_joint_trajectory(names, points, *, path_tolerance=None, goal_tolerance=None, goal_time_tolerance=0.0, control_mode=POSITION, preemption=QUEUE, gains=None, speed_scale=1.0, feedback=None) -> TrajectoryResult` | The driver's trajectory action, every field; `points` are `[(t, positions, velocities), ...]` sent as given |
| `with stream(controller, gains=None, timeout_s=None) as s: s.send(setpoint)`                                                                                                                                             | A streaming session; `send()` at least every `timeout_s`. Leaving the block holds and closes it             |
| `list_controllers() -> list[Controller]`                                                                                                                                                                                 | The driver's streaming controllers and whether each can be opened now                                       |
| `set_gains(gains) -> (accepted, message)`                                                                                                                                                                                | The session's default impedance gains                                                                       |
| `plan_to_pose(pos, quat, start_joints, hold=HOLD_NONE) -> Plan`                                                                                                                                                          | A cuRobo plan, nothing moved                                                                                |
| `plan_to_joints(q_goal, start_joints) -> Plan`                                                                                                                                                                           | The same, to joints                                                                                         |
| `set_world(world) -> (success, message)`                                                                                                                                                                                 | Swap cuRobo's collision world for a packaged one                                                            |

The constants (`HOLD_NONE` / `HOLD_LEVEL` / `HOLD_FIXED`, `POSITION` /
`IMPEDANCE`, `QUEUE` / `LATEST_WINS`) and the dataclasses are in
`capability_engine.arm`. `RammpArm`'s tunables and path helpers
(`cartesian_knots`, `joint_trajectory`, `FRAME`, ...) are in `rammp_arm`; the
shared sim (`make_sim`, `solve_ik`, `tool_pose`, `set_keep_out`) is in
`capability_engine.sim`. Every tunable is an `arm.*` ROS parameter.

### Exceptions

| Exception      | Raised when                                                                         |
| -------------- | ----------------------------------------------------------------------------------- |
| `Cancelled`    | The orchestrator cancelled the action (by `ctx.checkpoint()` and by any arm motion) |
| `ArmError`     | Base of the arm's errors below; also a refused or failed driver-API call            |
| `NotInAction`  | An arm call outside `execute()`, or `send()` outside its `with` block               |
| `ControlTaken` | Another sender took control of the arm mid-action                                   |
| `EStopped`     | A driver-API command while the arm is e-stopped                                     |
| `StreamClosed` | The driver closed your stream                                                       |
| `ModeConflict` | A stream or goal while another one runs                                             |
| `StateTimeout` | A state read got no fresh message                                                   |

## A capability

```
capability_engine/               shared ament_python package: the orchestrator engine contract
capabilities/<name>/             one ament_python package per capability (`make new CAP=<name>`)
  package.xml, setup.py, setup.cfg
  <name>/node.py                the capability: a CapabilityEngine subclass
  launch/<name>.launch.py       what both the native build and the image run
  Dockerfile                    FROM a pinned rammp-base / rammp-cuda tag, built from the repo root
  rammp-alternative.yaml        the real alternative: sheppy's schema plus `tier`
  rammp-alternative.mock.yaml   same topics, no hardware — CI smoke-tests this one
  docs/interface.md             the page rammp-docs pulls; keep its headings
```

plus an owner line in `.github/CODEOWNERS`.

## What CI enforces

| Check                      | Rule                                                                                                                                          |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `pre-commit`               | Ruff (lint + format, no `print`), clang-format, hadolint, mdformat, actionlint, file hygiene, no large files or keys                          |
| `checks`                   | the files above exist; fragments are schema-valid; mock topics equal real topics; pinned base; owner in CODEOWNERS; **sheppy loads the node** |
| `smoke (<capability>)`     | on arm64, the mock stays up, every declared `publishes` topic appears, and it exits on SIGTERM within 5 s                                     |
| `tier: demo-ready` (extra) | image pinned to a release tag or digest (never `:dev`/`:main`/`:latest`), and a `/…/status` health topic                                      |

Not machine-checkable, so it is on the PR author and the owner: a
hardware-in-the-loop pass (attended, human on the e-stop) before a capability
is marked `demo-ready`, and an on-call owner for the demo.

Merges to `dev` and `main` publish images tagged with the branch name and commit
SHA. A `v*` tag publishes semver tags for demo profiles to pin.
