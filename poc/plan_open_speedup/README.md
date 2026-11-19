# plan_open speed-up: proof of concepts

Two ways to cut the microwave open's planning search (`door_common.open_door.plan_open`),
measured on abra (Jetson AGX Orin, no arm) on 2026-11-18. Not for merge as is.

## The case

Flatbot's arm model (`/robot_state_publisher` on snorlax), door_common configured for the
microwave, the door and start joints of `tests/arm/test_door_planning.py::test_microwave_open_plans`,
swing 50 then 45 deg, `PushOpen(None)` (no planner attached, so no cuRobo calls in the serial
baseline). `setup_case.py` holds it.

## Results

| | plan_open | candidates | chosen plan |
| --- | --- | --- | --- |
| serial PyBullet (today) | 7.9–8.0 s | 122 | tilt 17, swing 50 |
| PoC 1: fork pool, 8 workers | 1.26–1.30 s | 122 | identical, bit for bit |
| PoC 1: fork pool, 15 workers | 0.96–1.04 s | 122 | identical, bit for bit |
| PoC 2: batched cuRobo IK | 1.22–1.28 s | 96 | tilt 25, swing 50 |

PoC 2 breakdown: 31 IK calls, 7,622 problems, 0.64 s on the GPU (about 20 ms a call), the rest
CPU gates (keep-out, limits, trajectory keep-out) and the socket.

The chosen plan's path, measured the same way for both (sim FK for pose error, cuRobo's spheres
for self-collision; `compare.py`):

| | serial | batched cuRobo |
| --- | --- | --- |
| tool position error, max / mean | 3.39 / 0.44 mm | 0.15 / 0.02 mm |
| tool orientation error, max | 0.09 deg | 0.03 deg |
| largest joint step on the swing (2 cm waypoints) | 17.8 deg | 10.0 deg |
| joint travel over the swing (all joints) | 459 deg | 447 deg |
| margin to the J4/J6 guards | 8.2 deg | 14.3 deg |
| self-collision | none | none |

(side, tilt, swing) combinations found: 24 by both, 1 by serial only, 2 by batched only.

## PoC 1: fork pool (`open_door.PLAN_WORKERS`)

`plan_open`'s per-tilt search (3 side angles x 5 tilts) moved into `_tilt_candidates`;
`_search_tilts` runs the tilts in a `fork` pool when `PLAN_WORKERS > 1`. Each worker uses its
forked copy of the PyBullet sim, so nothing is rebuilt. Results come back in tilt order, so the
pick is unchanged. A tilt that finds no configuration in the sim is searched again in the parent
when a planner is attached (cuRobo's grasp-shape fallback can't run in a forked child). In
parallel, the one-time detailed printout of the first failing approach is skipped. Default 1 (off).

- `tests/arm/test_door_planning.py::test_microwave_open_plans_the_same_in_parallel` pins the
  parallel plan to the serial one.
- `bench_pool_rclpy.py`: 10 runs with 12 workers forked while an rclpy node spins a 4-thread
  `MultiThreadedExecutor` with two busy timers: no hang, the same plan every time, the executor
  kept running.

## PoC 2: batched cuRobo IK (`plan_open_batched.py`)

The same candidates, gates and pick as `plan_open`, but the IK comes from cuRobo's
`IKSolver.solve_batch` (`curobo_ik_server.py`, flatbot's own `config/robots/flatbot/curobo/robot.yaml`,
URDF sanitised as the planner node does) over a Unix socket, standing in for an IK service on the
planner:

- A: every tilt x every configuration seed, one call.
- L: every configuration's line-up at every standoff, one call; then the straight-ins, one call per
  2 cm step, each seeded from the step before.
- S: every swing, one call per waypoint, each seeded from the waypoint before.

The chaining matters. Seeding every waypoint of a path from the grasp joints in one big call
(11.7k problems, 2 calls) lost most swings to the jump gate: 14 candidates instead of 122.

Settings that matter (`run_curobo.sh <seeds> <regularization> <particle> <grad_iters> <batch>`;
`sweep.sh` runs a list):

| setting | plan_open |
| --- | --- |
| no CUDA graph, default iterations | 7.4 s (31 calls x ~230 ms fixed cost) |
| CUDA graph (every call padded to 512), default iterations | 1.9–2.0 s |
| CUDA graph, 512, `grad_iters` 12 | 1.2–1.3 s |

cuRobo v0.7.8 quirks hit: its kernels assert on non-C-contiguous tensors (`np.ascontiguousarray`),
and `use_particle_opt=False` crashed in the line search before that fix (not retried since).

## Running

On abra, from `~/flatbot-bot-poc/repo/poc/plan_open_speedup` (`/data` = `~/flatbot-bot-poc/data`):

```bash
./run_engine.sh bench_serial.py /data/flatbot.urdf /data/serial.json 3
./run_engine.sh bench_pool.py /data/flatbot.urdf /data/serial.json /data/pool8.json 8 3
./run_curobo.sh 1 1 1 12 512 &      # the IK server, then:
./run_engine.sh bench_batched.py /data/flatbot.urdf /data/batched.json 3
./run_engine.sh compare.py /data/flatbot.urdf /data/compare.json
```

`run_engine.sh` uses the `microwave:dev` image with this checkout mounted over its packages;
`run_curobo.sh` the `rammp-curobo:nightly` image with RAMMP-CuRobo's current `core/` mounted.

## Not covered

- No arm: these are planned joint paths, not executed ones. The swing still runs on its checked
  joints (`SWING_EXACT`), so the step sizes above are what the arm would be sent.
- The cuRobo calls a real open makes (`_CuroboFirst`, the push-open checks) aren't in either
  timing.
- PoC 2 needs a batched IK service on the planner (RAMMP-CuRobo has none today) before an engine
  can use it; here it's a socket to a standalone process.
- `test_microwave_close_plans` fails with flatbot's URDF standing in for dojo's, on unmodified
  `dev` too; the other door tests pass.
