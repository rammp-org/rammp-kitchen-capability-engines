# Gen1.5 demo session

Work down the checklist. Save the failed run before changing anything. One operator can do the survey, dry runs and
empty-chair trials from the stop station. For occupied trials, have the rider and a separate operator on the stop.

## 1. Set up

- [ ] Pin the fridge image built from this checkout and record its ID, the deployment revision, robot profile and
  applied ROS parameters. Build from the repo root with
  `docker build --network host -f capabilities/fridge/Dockerfile -t fridge:ws11-hw .`. In `rammp-deployments/december_2026/sheppy-manifest.yaml`, use the current
  [fridge fragment](../rammp-alternative.yaml) for the real setup. Check
  `./config/robots/current:/config/robot:ro` and `mount_tilt_deg: 30`. Verify the floor, platform, seat and person layers
  resolve in both containers. Check that `planner_world_dir` and `planner_world_dir_in_planner` are the two
  views of the same shared world directory. Confirm the planner's `collision_cache_obb: 60` and retain the safety
  layers. Save the deployed planner YAML.
- [ ] Start Sheppy with the Gen1.5 robot config and the session profile. Check arm, planner, TF, wrist and scene RGB-D,
  and SAM 3 (including its model mount). Use the same ROS domain and CycloneDDS as the deployment. On the Gen1.5
  setup with separate interfaces, `eno1` is the MIB sensor network. Check there is one `/fridge/execute_action` server.
  Start with `allow_execute: false` and docking, guided docking, scene fallback and translated wrist re-look off.
  Before UI docking, check the deployed orchestrator's `execute_timeout_s`. Disable that deadline for attended
  commissioning if supported, or raise it to cover the parking prompt, checked dock, planning and motion. Check
  Stop still cancels. Record the value and set the demo timeout from measured full runs.
- [ ] Ready the stop terminal. Test engage and release at rest and verify `/control_status` changes. Once the arm is
  clear, verify the software stop during the first empty-chair, low-speed arm-only move before any contact. Check
  the release and `/clear_faults` before resuming. Keep the physical stop accessible according to the site's procedure
  for an arm that can drop on power loss.

Run these from the robot host. Use the arm container name deployed by Sheppy.

```bash
# Engage, repeated until stopped. Ctrl-C in this terminal to end the publisher.
docker exec -it sheppy-arm bash -lc 'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash &&
  ros2 topic pub -r 10 /estop rammp_common_interfaces/msg/EStop "{state: 1, source: operator, reason: stop}"'
# Clear once the area is clear and the stop publisher has ended.
docker exec sheppy-arm bash -lc 'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash &&
  ros2 topic pub -t 10 -r 10 /estop rammp_common_interfaces/msg/EStop "{state: 0, source: operator, reason: clear}"'
```

In a shell with the deployment's ROS interfaces and DDS environment sourced, use:

```bash
fridge_goal() {
  ros2 action send_goal --feedback /fridge/execute_action \
    rammp_orchestrator_interfaces/action/ExecuteAction \
    "{engine_id: fridge, action_id: $1, params_json: '$2'}"
}
fridge_goal go_to_start '{"execute":false,"speed":0.2}'
# After the area is clear and the e-stop is released:
ros2 service call /clear_faults std_srvs/srv/Trigger
```

## 2. Survey and mark the site

- [ ] Measure the handle anchor height, straight bar length and cross-section, door top and bottom heights, width
  and depth including bins, fridge body dimensions and cabinet support height. Measure the hinge side facing the
  fridge and pin offsets from the handle anchor. Fixture A is left hinged. The model puts the pin 49 mm inward
  and 374 mm along the door from the fitted outer anchor. Record those datums and the measurements. Update the
  [measured geometry parameters](interface.md) and check that the model covers the surveyed solid surfaces.
- [ ] With the robot clear, measure breakaway force and moving force at about 30° and 80° using a pull scale at the
  bar, tangential to the swing. Record N, angle and direction, and any closing drift or cabinet motion.
  Repeat dry checks after changes to the fixture. Stop if the cabinet shifts or the swing corridor is obstructed.
- [ ] Lay down a taped axle-midpoint mark and a chair heading line. Position the closed-door handle anchor at
  `(x, y) = (0.825, -0.235) m` in the cuRobo `arm_mount` frame, with the inward normal at −14° relative to chair
  forward. For the stock measured Gen1.5, the mount is `(+0.116, +0.284) m` from the axle in chair axes, so the
  handle anchor projects to `(+0.941, +0.049) m` from the axle. For fixture A the nominal floor mark is 0.97 m
  out from the closed front and 0.65 m along the front from the hinge toward the handle. Point the chair 14°
  toward the hinge from square. Use the verified transform if it differs. Photograph
  the mark with the axle reference, heading line and measured clearance. Verify sight lines for both cameras
  and clear the full chair, arm and door sweep before checking the mark under motion.

## 3. Cameras and dry-run ladder

- [ ] Mock the UI first if it will be used. Check Go to Fridge, Open, Done and Stop. The parking prompt is checked in the
  attended UI trial below. Run `go_to_start` with `execute: false` on the real stack. Inspect the planned path
  in the clear area. After the stop test, stage at `speed: 0.2` with the empty chair and room away from the fridge,
  then park at the mark. Return `allow_execute` to false for the dry runs. Log start joints and clearances.
- [ ] At the same stationary pose with the door shut, dry-run Open once per camera with `detect_camera:=wrist` and
  `detect_camera:=scene`. Repeat at the intended lighting and selected small offsets around the mark. Log exposure,
  gain, RGB-D stamps, masks, depth holes, TF revisions and each settled handle XYZ and outward unit normal in
  `arm_mount`. Check the matched RGB/depth overlay and avoid clipped highlights over the bar and adjacent door.
  Adjust exposure on the actual camera driver and rerun. The survey pass is ≤30 mm Euclidean separation and ≤5°
  between normals. Require ≤3° before fine-dock as well, since `confirm_wrist` uses 30 mm / 3° at this revision.
  A failed detection or excess disagreement stops execution here. Restore wrist as the primary camera.
- [ ] From the mark, dry-run Standard, then Fast and Hook before their first respective executing trials. For
  Standard, repeat at representative ±3 cm / ±3° offsets with verified clearance. Inspect the complete approach,
  pull, inner-face push and retreat. Expect a front-grasp family at the nominal mark and a checked route to 90°.
  Log stage and refusal reason, selected grasp and handoff, world box count, planner calls and elapsed seconds.
  On the Jetson, time cold and warm searches with normal camera, SAM 3 and bag load. Repeated full plans must fit
  the session latency budget and the 240 s per-search ceiling (`PLAN_BUDGET_S` in `constants.py`). Investigate
  overruns or box-budget refusals before changing either budget. If testing scene fallback, change only
  `open_scene_fallback_enabled` after camera agreement. Check which source supplied the successful plan in the log,
  and keep translated wrist re-look off until its approach has been verified separately.

```bash
fridge_goal open '{"execute":false,"drive_base":false,"base_moves":false,"profile":"standard","speed":0.2}'
```

## 4. Grasp confirmation and Standard Open, base parked

- [ ] Calibrate the gripper before the first pull. Using the supported gripper controls, perform repeated attended
  closes in free space and on the straight section of a secured matching bar, at the planned bite orientation and
  `GRASP_FORCE` (0.5). After each close settles, keep hands clear and record one window with the corresponding label.
  Run the helper with the deployment's ROS setup sourced. Use a fresh output filename per close.

```bash
python3 scripts/fridge_calibrate_grasp_aperture.py empty empty-01.csv
python3 scripts/fridge_calibrate_grasp_aperture.py bar bar-01.csv
```

- [ ] Compare all settled gap ranges with the 10–42 mm gate in `constants.py` and the independently measured bar
  thickness. Pass if empty closes stay below the lower bound and bar closes stay inside the gate with margin.
  Save CSVs and measured gaps. If they overlap or the width is wrong, check feedback, force, finger alignment and
  the 85 mm stroke estimate before adjusting `GRASP_GAP_MIN_M` / `GRASP_GAP_MAX_M` and repeating the fixture test.
- [ ] Enable execution for the attended run. At the mark, resend the verified Standard goal with `execute: true`.
  Watch finger insertion, confirmed gap, seal break, pull handoff, release/back-off, first push on the **inner**
  face, and retreat. Log angles independently at the end of pull, after release, at push contact, at first 90° and
  after retreat. Pass when the action completes, the stationary door remains at least 90°, and the chair has stayed
  still throughout. Check release drift and contact entry before repeating a failed push. Stop at unexpected contact
  or slip. If the hand may still be on the bar, check the actual door and arm pose before dry-running an attended
  `release_only` goal. That goal needs valid saved door geometry. If it is missing or invalid, use the site
  recovery procedure. Reset the door and chair only after the arm is clear.

## 5. Hook, base parked

- [ ] Use the dry-run ladder with `profile: hook` and check the planned 94° pull and withdrawal. Check that an
  unavailable hook plan reports the checked Fast fallback. A fallback counts as a fallback in the log. Verify the
  driver has fresh `/ee_state` in `arm_mount`. For J1–J7 diagnostics, configure the arm's
  `contact_debug_topic: true` with contact detection enabled and check `/contact_monitor/residual` on the deployed
  driver. Start the recorder before the reviewed low-speed pulls. Keep a CSV and console log per trial.

```bash
python3 scripts/fridge_record_hook_wrench.py record hook-trial-01.csv
```

- [ ] Use the production `hook late pull` line as the score. The gate uses the median **resisting hinge moment**
  from the flange wrench during 83–92° of tool-inferred travel, at least three samples. The provisional
  `HOOK_MIN_HINGE_LOAD_NM` is 0.08 N·m in `open_door.py`. Independently label retention from door observation
  and record the score, sample count, speeds and final door angle. Compare repeated retained pulls to supervised,
  independently labelled unloaded reference pulls on an approved clear bench setup with matching trajectory and
  angle window. Keep Hook out of the demo if a comparable unloaded reference cannot be collected or the scores
  overlap. Qualify Hook only after fresh held-out trials separate with margin, the selected hook completes
  at ≥90° after retreat, and its Fast fallback passes step 7. Never induce a slip on the
  installed fridge. An unverified pull stops before release and invalidates the saved door estimate. Use the site recovery procedure before returning to Standard.

```bash
python3 scripts/fridge_record_hook_wrench.py propose --retained retained-*.log --unloaded unloaded-*.log
```

## 6. Base checks and guided parking

- [ ] With the arm safely parked, chair empty and floor clear, check live `/rammp/mib/status` and fresh
  `/base/odom` with volatile, best-effort QoS (expect about 20 Hz even at rest). Measure the drive-wheel midpoint,
  then compare tape and odometry on a short straight and turns in both directions. The stock
  `docking_odom_to_axle_xy` assumes the odom child origin is the axle. Verify signs and offsets against the
  measured `docking_axle_to_mount_xy` above. Log residual axle translation on a turn and straight distance
  error. Resolve mismatches before docking. Compare against the 8 mm / 0.7° leg tolerances.
- [ ] In the approved base commissioning mode, start at 0.02 m/s and repeat at 0.03 m/s if stopping margins allow.
  Verify commanded zero, cancel, joystick takeover, software e-stop and loss-of-command watchdog (300 ms from the
  last fresh command). Record command/odom timestamps, MIB mode, stop latency and physical braking distance.
  Pass only when every stop behaves as expected and fits the measured clearance. Commission yaw separately from
  the lowest stable turning speed. Set `docking_max_linear_speed` and `docking_max_yaw_speed` to the verified
  caps. The shipping caps are 0.05 m/s and 0.12 rad/s. UI requests 0.03 m/s before capping.
- [ ] With the checked arm start pose, set `docking_enabled: true`, `guided_docking_enabled: true` and
  `guided_parking_instruction` to name the taped axle and heading marks. Bring up the orchestrator/UI with
  `use_orchestrator: true`. Press **Open Fridge Door (wheelchair drives)**, park, stop, and press **Continue**.
  This hands off to checked autonomous fine-dock. The UI goal uses Standard at arm speed 0.7, so qualify that
  speed first with the base parked. Use direct Standard drive goals with explicit lower speeds for commissioning
  the fine-dock path before the UI trial. For example, once the 0.02 m/s check has passed, use
  `fridge_goal open '{"execute":true,"profile":"standard","speed":0.2,"drive_base":true,"base_speed":0.02,"base_moves":true}'`.
- [ ] Start the empty chair at the mark with zero correction, then try small known offsets (build up to about
  10 cm / 5° if clear). Require a stopped final pose within 2.5 cm and 2° of target, camera confirmation and
  a complete open to at least 90° after retreat. Check any separately planned straight pull-to-push base handoff
  in the same clearance envelope. Log prompts, arrival pose, route, correction count, minimum
  clearances, final pose, camera agreement and button-to-90° time. On refusal, clear the cause and repark.
  After repeatable empty-chair passes and verified occupant envelopes, repeat the no-correction case with a rider
  and stop operator, then one small correction. Stop for any clearance, feedback or braking failure.

## 7. Fast profile and save the session

- [ ] Dry-run `profile: fast` from the verified mark. Then use the same attended fixed-base ladder as Standard.
  Increase arm speed only through verified steps. If qualified, repeat with the checked dock using an explicit
  goal (`drive_base: true, base_moves: true`). Log the release, first push and post-retreat angle. Require repeated
  physically measured ≥90° after retreat and clean clearances before comparing paired Standard/Fast timings at
  the same settings. Revert to the verified Standard setup if Fast alters the result. UI buttons at this revision
  select Standard. Use explicit `profile` goals for timed Fast and Hook tests.
- [ ] Save the config, survey, angle video and per-run result, first failed stage and timing. Copy
  `/tmp/fridge_data/` (especially `runs/*/console.log`, bags and detections) before reboot. Record extra telemetry
  on the same clock, including `/base/odom`, `/base/setpoint/twist`, `/rammp/mib/status`, `/control_status`,
  `/gripper_state`, `/ee_state` and `/contact_monitor/residual` when enabled. Record goal receipt, prompt
  acknowledgement, first 90°, retreat completion and final physical angle separately.
