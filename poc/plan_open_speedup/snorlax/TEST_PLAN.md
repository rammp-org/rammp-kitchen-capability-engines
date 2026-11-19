# Side-by-side planner test on flatbot (snorlax)

Compares the microwave open's three planners on the real system:

- `serial`: today's PyBullet search (~8 s).
- `pool`: the same search in 8 processes. Same IK, so the same plan, ~1.3 s.
- `curobo`: the same search with cuRobo's batched IK (PoC server), ~1.25 s. It picks its own plan
  (often a different tilt), so this is the one whose swing needs to be compared on the arm.

Everything runs from `~/flatbot-bot-poc/repo/poc/plan_open_speedup/snorlax` on snorlax. The
microwave engine runs standalone in a container (`flatbot-poc-microwave`) with this branch's code
mounted. It is not sheppy's microwave node, and nothing in `~/rammp-deployments` changes.

## Before you start

0. Staged: `poc/plan_open_speedup/snorlax/stage.sh` from a machine with this branch and RAMMP-CuRobo
   checked out copies both to `~/flatbot-bot-poc` on snorlax and runs `setup.sh` (step 2) there.
1. Sheppy up with the daily profile's nodes the microwave needs (arm, arm_state_publisher,
   planner, static_tf, both cameras, scene_segmenter), but **stop sheppy's microwave node** in
   the TUI. Two microwave nodes would both take every goal; `run_microwave.sh` refuses if it sees
   another one.
2. `./setup.sh`: pulls the microwave image, saves the arm model, checks the branch imports.
   Nothing moves. Rerun it if the arm's mount changed.
3. `./run_ik_server.sh`: starts the cuRobo IK server (`flatbot-poc-ik`, GPU only, about 2 min
   warm-up). It computes joint angles and can't command anything.

## Part 1: dry runs (nothing moves)

```bash
./run_microwave.sh          # allow_execute false: every goal is a dry run
./send.sh open '{"execute": false, "compare_planners": true}'
```

Each goal detects the door once, then plans with all three planners on that same detection (run
log `plan_compare`), then runs the usual dry run. Do 5 placements: move or turn the microwave a
little between goals (a few cm, ~10 deg) so the comparison covers more than one pose.

## Part 2: executed swings (attended)

Someone at flatbot with a hand on the e-stop and the area clear before this step.

```bash
docker rm -f flatbot-poc-microwave
./run_microwave.sh --allow-execute
./send.sh open '{"execute": true, "push_open": true, "planner": "pool", "speed": 0.5}'
./send.sh open '{"execute": true, "push_open": true, "planner": "curobo", "speed": 0.5}'
```

- The first run of each planner at `"speed": 0.5`. Watch the cuRobo one: it grasps at its own
  tilt.
- Then 5 runs each at full speed, alternating pool, curobo, pool, curobo, with the microwave in
  the same place.
- Between runs, close the door: `./send.sh close_door_ui '{}'`, or by hand once the arm has
  stopped.
- `serial` doesn't need executed runs: its plan is the pool's, bit for bit (part 1 checks this on
  every detection).

## Results

```bash
./analyze.sh --since 20261119_0900   # runs from then on (run folders are named by start time)
```

- **Planning:** each planner's time on every dry-run detection, and whether the pool's plan was the
  serial one ("same as serial": yes).
- **Per executed run, then per planner (mean ± std):**
  - planning time
  - the planned largest step between swing waypoints
  - swing duration
  - peak joint speed and acceleration
  - peak joint torque, i.e. how hard the arm pulled on the door
  - how far the hand left the commanded arc (cm)
  - the circle fit's rms
  - end error
  - whether the swing and the run finished

Run logs (`run.json`, `arm.npz`, `console.log`, bag) are in `~/flatbot-bot-poc/data/microwave/runs/`.

## When done

```bash
docker rm -f flatbot-poc-microwave flatbot-poc-ik
```

Then restart sheppy's microwave node in the TUI if the demo needs it.

## If something's off

- `docker logs -f flatbot-poc-microwave`: the engine's log (it prints `planned with <planner> in
  N s` on every run).
- `docker logs flatbot-poc-ik`: the IK server's. With `"planner": "curobo"` and no server, the run
  refuses before any motion: "no cuRobo IK server at /poc/ik.sock".
- After the grasp, if the arm is more than 3 deg off the plan, the swing is re-planned from where
  it is, with the same planner (run log `swing_replanned`).
