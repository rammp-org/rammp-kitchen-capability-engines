# `button_press` — the microwave button press

The autonomous button press from `feeding_deployment/button_press/` (branch `button-task`; read
its `README.md` for how detection and the panel frame work), ported from the Cornell arm RPC
(`ArmInterfaceClient` / `arm_server.py`) to RAMMP's arm driver, as a capability engine: one
action, `press_button`, run by `capability_engine.CapabilityEngine` and moving the arm through
`self.arm`. Interface (action, parameters, topics): [`docs/interface.md`](docs/interface.md).

**Status (2026-11-01):** ported to `CapabilityEngine`; the offline tests pass, including
`PressArm` on the real `self.arm` beside `capability_engine`'s fake driver. **Not yet run on the
arm as an engine.** Before the port, as a script on the HERL bench: live detection and dry runs
passed, and an attended 1.0 cm push along the normal from the hand-placed pre-press spot clicked
start_30s; the full sequence had not run.

## Running it

```bash
ros2 launch button_press button_press.launch.py use_orchestrator:=false                  # dry runs only
ros2 launch button_press button_press.launch.py use_orchestrator:=false mock:=true       # no hardware
make action CAP=button_press ACTION=press_button PARAMS='{"target": "start_30s"}'
```

**The arm moves only with the `allow_execute` parameter true** (the fragment's default is
false); otherwise every action is a dry run: detect, plan and gate every leg, move nothing. A dry
run that needs staging stops after gating the staging goal, because the press is planned from the
re-detect there. Someone stands at the e-stop whenever `allow_execute` is on. The first motion:
`{"presses": 0, "no_return": true}`, attended, before any press.

How a run ends: a check that refuses a leg before it is sent backs off to the last safe pose (the
start, or the pre-press spot after a press its own checks stopped). A motion the driver fails (an
e-stop, control taken, a tracking abort) or a cancel leaves the arm holding where it stopped. The
approach, press and return report `pausable: false`.

## The pre-press spot (per rig)

`PREPRESS_EE_OFFSET_M` / `PREPRESS_EE_QUAT_PANEL` in `press_button.py` were measured on
rchi-cpu-5, in that rig's *detected* panel frame, so they absorb its hand-eye bias. With this
URDF's fingertip (tool frame + 5.96 cm) they put the fingertip **7 cm into the panel**. The
10-07 microwave port saw the same rig's grasp constants land 9.8 cm too deep on this Jetson.
`check_fingertip` refuses any pre-press spot whose fingertip is less than 1 cm in front of the
fitted plane. To re-measure, the
`measure_prepress_offset` script (removed with the port, in this capability's history at
`d5ed214`) detected from ~25 cm, then had you stop the driver, hand-guide the LEFT fingertip 2 cm
in front of the button, bring the driver back, and printed the two constants to paste. It needs
porting before the next rig.

## Scene camera far stage

The first look (`far: scene`, the default) uses the fixed **scene camera** (Orbbec Gemini 336L,
`/scene_camera`), not the wrist camera, so the arm may start anywhere: facing away, at HOME, wherever
the last task left it. `scene_finder.py` finds the cream Comfee's **front** by geometry (a vertical
plane of the front's size, standing on a surface, facing the arm, with the dark door window on the
left and the control strip on the right; cream as a confirmation) (narrowed to the scene segmenter's `microwave` box on `/perception/object_detections` when it has one) and places the button by a stored
offset from the front's bottom-right corner. From ~1.2 m it is good to a few cm: **staging only**.
cuRobo takes the arm straight to the pre-press pose + `SCENE_STAGE_OUT_M` (15 cm), the wrist-turn
limit lifted for that one move, and the wrist camera's dome detection there plans the press, exactly
as before.

- **Needs** the `scene_camera` node and its tf (`base_link -> scene_camera_*`), which box-opening's
  launch publishes from its `config/camera_scene.yaml`. This package reads that tf as published and
  changes nothing of box-opening's. This package's own sheppy manifest has no scene camera: run the
  bench profile that has it.
- **Falls back** to the wrist camera (domes, then the display) whenever the scene camera finds
  nothing usable: no frames, no tf, fewer than 4 of 5 frames agreeing within 2 cm / 5 deg, two
  microwave-like fronts, or no front passing the gates (a partly hidden one fails the size gate;
  "arm in the way?" when the tool stands in front of it). Every reason is printed and logged.
  `far: wrist` is the behaviour before 2026-10-14.
- **Constants** (`scene_finder.py`): `FRONT_W_M` / `FRONT_H_M` are what the finder measured on the
  HERL microwave (tape-measure to confirm). `FACE_CORNER_TO_START_MM` started from the button's
  pixel in the scene image; after a few staged runs, `python3 tools/scene_offset_from_logs.py`
  prints the median of what the wrist camera actually found.
- **Logs**: `panel_frame_scene` (the front, the frames used, rejected frames' reasons),
  `scene_fallback`, `staging_redetect` (now with `source` and `moved_cm`: the scene-vs-wrist distance,
  which is how calibration drift shows up), and `abort` (every abort's reason, for all runs).
- If the wrist camera finds no domes at the staging pose, the run holds there and names the scene
  calibration or the corner offset as the likely cause.
- **Not covered**: a front turned away from the scene camera, a front hidden by the fridge door
  (recorded as `tools/frames/scene/neg_fridge_door_open_*`), other microwave models, the door
  handle (cuRobo's world has the microwave and counter as boxes from the scene estimate, good to a
  few cm: still watch the staging move).
- **Tests**: `test/test_scene_finder.py` ray-casts a box model of the microwave (moved, turned,
  wrong size, no window, grey, two of them, hidden, floating, facing away) and runs the recorded
  scene frames in `tools/frames/scene/` (`record_scene_frames.py LABEL`; `neg_*` = must find nothing).

## What changed from `button-task`

| Old (`ArmInterfaceClient`)                       | Now (`PressArm` on `self.arm`, `press_arm.py`)                                                                                                      |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `get_state()["position"]`                        | `/joint_states` (best effort; `joint_1..7`, wrapped to ±π)                                                                                          |
| `get_state()["ee_pos"]` (Kortex tool pose)       | `/ee_state` + 0.120 m along the flange z (`/ee_state` is the flange, `end_effector_link`)                                                           |
| `get_state()["gripper_pos"]`                     | `/gripper_state.position` (0 open … 1 closed)                                                                                                       |
| `get_arm_state()` (`SERVOING_READY`)             | `/control_status` (e-stop, arbitration, owner) + `/diagnostics` (`Arm` task)                                                                        |
| `get_speed()` low/medium                         | none. Each leg is timed here: min-jerk, peak 6 cm/s tool, 0.35 rad/s per joint                                                                      |
| `CartesianTrajectoryCommand` (Kortex's own IK)   | `execute_joint_trajectory`: **the exact joint solutions the sim gates checked**, 20 ms samples, driver-side `path_tolerance` 0.15 rad               |
| full `FeedingDeploymentPyBulletSimulator` for IK | `urdf/gen3_tool_frame_kinematic.urdf` alone (same chain, same `tool_frame`; FK/IK identical to the full sim, tested); loads in 10 ms instead of 2 s |
| camera `/camera/...`, frame `arm_base_link`      | `/wrist_camera/...` (`camera_ns`), frame `arm_mount`                                                                                                |
| `feeding_deployment.ros2.calibration_tf`         | the deployment's camera mounts in tf (rammp-deployments `static_tf`)                                                                                |

Unchanged: `dome_pattern.py`, `panel_frame.py`, `geometry.py` are copies (only the import path
differs). The detection, panel
frame, staging and planning in `press_button.py`, and every existing gate: IK error ≤ 5 mm,
orientation ≤ 1°, joint step ≤ 10°, reach ≤ 1.2 m, tool height 0–1.2 m above the arm base (a table guard; widened 2026-10-13), wrist turn ≤ 45° (the 40 cm move cap was removed 2026-10-12),
press ≤ 3 cm, final position within 1 cm (approach and press; staging, bail and back-to-start within 5 cm since 2026-10-12).

**New refusals** (none of them loosen anything):

- **Joint limits:** J2/J4/J6 must stay 2° inside the driver URDF's limits. Kortex used to do
  the IK; now our solutions go to the driver as they are. Raw PyBullet IK does return
  out-of-range J4.
- **Fingertip clearance:** see above.
- **Tool-frame check at preflight:** sim FK at the live joints must match `/ee_state` + 0.12 m
  to within 5 mm. If the driver ever reports a different frame, the pre-press offset would be
  silently wrong, so the run refuses.
- **Start drift:** a leg is refused if the arm moved more than 0.5° since it was planned.
- **Arbitration:** none here. The engine never calls `/acquire_control` or `/release_control`;
  the orchestrator owns control, and `self.arm` refuses to command an arm that must not be.

**Long moves (staging, back to start, bail)** always go through the driver's `go_to_ee_pose`, but never starting at the panel: back to start and a bail from the approach first back straight out `RETREAT_OUT_M` (15 cm) along the panel normal, and an aborted press stroke goes straight back to the pre-press spot (2026-10-14: cuRobo's return from 2 cm in front of the button swept the gripper into the microwave). Then
cuRobo plans from the measured joints and the driver
executes. The target is `tool_frame` in `base_link`, the same frame as `ee_pose()`. What that gives up:
the path can't be previewed or gated before it runs (only the goal's reach and z band are checked,
and a dry run checks only the goal). These moves run at `self.arm`'s
`arm.go_to_speed_scale` (0.3 by default; 1.0 before the port): on
2026-10-14 a 51 cm staging move from HOME took 1.2 s at full speed and hit the microwave. Since then
cuRobo's world has the microwave and its counter (`collision.py`, from each panel estimate; the
wheelchair and user are the planner config's own layers on flatbot), and the approach and the straight
lead-in are clearance-checked against the same boxes (docs/interface.md, Collision checking). The approach and the press
never use `go_to_ee_pose` (its path is not a straight line). The only gripper command is preflight's close on an execute run (never in a dry run, never opened).

## Still to check on hardware

- The camera on the wrist: the local manifest says D435i (serial 140122079014, 2026-10-07),
  RAMMP's December manifest says D405. The detector's ~22 cm minimum range and depth behaviour
  were tuned on rchi-cpu-5's camera.
- The calibration on this box is Capybara's (copied 2026-10-07). Touch-checked to ~2 cm for
  the door handle, never for buttons.
- The first motion: `--presses 0 --no-return` from 23 cm, attended, before any press.
