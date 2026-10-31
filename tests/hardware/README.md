# `self.arm` on the real arm

These tests check that `self.arm` does on the cell's arm what the README and
CONTRIBUTING tell capability authors it does. Each check runs inside a real
action of a standalone `CapabilityEngine`, the way a capability's `execute()`
runs. An observer reads the driver's topics on its own, never through
`self.arm`, and judges the result.

## Before you start

- **Run it on the cell (abra).** The image is arm64 and built from the same
  `rammp-base` as the capabilities. It needs the cell's sheppy stack up: `arm`,
  `arm_state_publisher` and `planner`.
- **Use the driver and planner images the students will run.** `go_to_ee_poses`
  and `go_to_joint_configs` need kinova-gen3-ros2 #65 and RAMMP-CuRobo #48, so
  nightly 2026-10-30 or later. If the driver does not serve `/go_to_ee_poses`,
  those tests skip.
- **Know the planner's world.** A `--motion` session first loads `--world`
  (default `world_real_bench.yaml`, dojo's) and stops if it can't. Say aloud
  what obstacles that world holds before anything moves.
- **Keep a person at the e-stop** for every `--motion` run. Every move starts
  from the driver's `home` preset and stays within about 10 cm of its tool pose,
  apart from wrist turns of up to 3 rad. Clear the bench around it. As a
  backstop, the test e-stops the arm itself and ends the session if the tool
  leaves 15 cm of its home position or J1–J6 leave 25° of home. That e-stop
  stays engaged until a person clears it.

## Running it

From a checkout on the cell:

```bash
make test-hw                                                      # stage 0: nothing moves
make test-hw HW_ARGS='--motion tests/hardware/test_10_joints.py'  # one stage
make test-hw HW_ARGS='--motion -x'                                # all automatic stages
make test-hw HW_ARGS='--motion --manual tests/hardware/test_80_manual.py'
```

Run the stages in order and read each one's failures before you start the next.
`-x` stops at the first failure, with the arm at rest. The results go to
`tests/hardware/results.xml`. To test another branch, such as #28, check it out
with this directory in place. The code under test is the mounted checkout.

| Stage               | Moves         | What it checks                                                                                                                                                                                                     |
| ------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `test_00_readonly`  | no¹           | The state reads are fresh, `get_state()` is the students' dict, the sim's tool frame matches the driver's, the planner's goal frame is the tool frame, `NotInAction` outside an action, and `set_world`¹           |
| `test_10_joints`    | yes           | `move_joints` (end, timing, `speed_scale`, knots, wrapped joints, `monitor`, impedance), `go_to_joint_config`, `go_to_preset`, `JointCommand`, `execute_joint_trajectory` (result, feedback, refusal raises)       |
| `test_20_cartesian` | yes           | `move_cartesian` goes straight, keeps orientation, turns in place, is refused before moving when unreachable, and stays under `CART_SPEED`                                                                         |
| `test_30_goto`      | yes           | `go_to_ee_pose` (end, `HOLD_FIXED`, `HOLD_LEVEL`, `speed_scale`, impedance, feedback, unreachable gives False), and `go_to_ee_poses` / `go_to_joint_configs` (blending cuts corners and is faster)                 |
| `test_40_gripper`   | gripper       | `gripper()` returns once the fingers stop: open, closed, half-closed, `speed`, `force`, and the open/close commands                                                                                                |
| `test_50_stream`    | yes           | `stream()` on joint velocity, twist, joint position and pose, each position and impedance. Also the hold on exit, the driver closing an idle session (`StreamClosed`), a bad setpoint, `ModeConflict`, `set_gains` |
| `test_60_interrupt` | yes           | Every kind of motion cut short by a cancel, `stop()`, pause, SIGINT or SIGTERM: the arm is still within 1 s, short of its target, no stream is left open, and the next action moves. Pause lets a motion finish    |
| `test_70_estop`     | yes           | `/estop`, published by the test as the operator would: every motion returns False with the reason, the driver API raises `EStopped`, an e-stop mid-motion stops the arm, and the arm moves after the clear         |
| `test_80_manual`    | yes, a person | The red e-stop, and a push that trips contact detection, mid-motion. The console says what to do and when                                                                                                          |

¹ `test_set_world` is marked `motion`: it reloads the planner's world and puts
the cell's own back.

The tolerances are constants at the top of each file. If a test fails by a
small margin, look at the measured value in the assertion before you widen
one.

## What it does not cover

- **`ControlTaken`.** The cell runs `arbitration_mode: disabled`, and the
  orchestrator does not acquire control yet.
- **`joint_torque` streaming.** The test never sends raw torques to the arm.
- **The keep-out zone.** It is only used in the fridge runs.
- **Mock mode and the orchestrator's calls** (`request_intent`,
  `write_world_state`). Neither touches the arm, and both are covered off-arm by
  `make test-arm`.
- **`docker stop` of a capability's container.** The signal tests send the
  same SIGINT to a capability process.
