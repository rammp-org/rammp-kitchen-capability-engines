# microwave_place

Put the tupperware the gripper is already holding (after `grab_container`) into the open microwave
(`place`), and take it out again (`take_out`).

- `microwave_place/microwave_detection.py`: where the microwave is and where the container goes, from the
  **scene camera** (not the wrist camera: the held container blocks its view).
- `microwave_place/place_motion.py`: planning, checking and running the arm: Walter Hsieh's
  `place_container` motion (arm model, impedance lowering), with the transit, the straight lines and the
  release the microwave needs.
- `microwave_place/rim_detection.py`: `take_out`: where the container's rim is, from the **wrist camera**
  (`grab_container`'s SAM3 rim detection).
- `microwave_place/take_out.py`: `take_out`: planning and running the arm.
- `microwave_place/microwave_place_node.py`: the `CapabilityEngine` action server: parameters, the actions, the plan and
  state files.

## Finding the spot: `microwave_detection.py`

`fit_interior(depth_m, K, world_T_cam, cavity_mask)` places this microwave's opening, a rectangle of
known size standing upright, from one scene-camera frame. It assumes nothing about where the oven
stands or which way it faces; only that it is upright and has the tape-measured dimensions in the
constants block at the top of `microwave_detection.py`. It returns, in `arm_mount` (gravity-aligned, metres):

- `point`: where the container goes: centred in the opening, half-way from the frame around the
  opening to the back wall, on the turntable glass;
- `insertion_axis` (horizontal, into the oven) and `lateral_axis_left`;
- `opening_corners`: the opening rectangle, in the plane of the frame around it;
- `floor_z`; `pins` and `fit` (the overlap and the depth checks' numbers); `edges` (empty).

How it works:

1. **The cavity mask.** SAM3, prompted with "inside of microwave", masks everything seen through the
   opening: every piece of the mask is kept and its holes are filled (at least 400 px).
1. **The rectangle, by best overlap.** The opening's four unknowns (where the middle of its bottom
   edge is, and its turn about vertical) are fitted so the projected 30.48 x 19.05 cm upright
   rectangle overlaps the mask best (intersection over union), from several starting turns (-60 to
   60 deg in 15 deg steps about facing the camera, the rectangle on the ray through the mask's middle
   at the distance its image height gives). The best overlap must be at least 0.7. Nothing picks edges
   or sides: whatever stands in front of part of the opening only takes that part out of the mask.
1. **Depth checks the fit.** The camera must be in front of the opening; at least 90% of what is
   seen through it (the mask 1.5 cm in from its edge, at least 50 depth points) must lie between 1 cm
   and the oven's depth plus 3 cm behind the fitted plane; and when the frame around the opening has
   at least 50 depth points (a ring just outside the mask, leaving out points more than 5 cm in front
   of the plane, such as the arm), their median must be within 1.5 cm of the plane.
1. **The point** follows from the dimensions: centred sideways, `DEPTH / 2` behind the frame plane,
   `FLOOR` above the opening's bottom edge.

Any check that fails raises `NotDetected` with the reason. Nothing is filled in.

`locate(...)` runs it on the live camera inside an action (`SceneCamera`). It first gets one frame of
SAM3's "microwave" mask (`microwave_prompt`), with its depth and camera pose, in a call of its own
(SAM3's label image holds one object per pixel, and the cavity lies inside the oven); that frame is
used for the obstacle. Then it detects continuously: each round it asks SAM3
(`/perception/sam3/detect`) for the cavity on the next color frame, pairs the mask with the depth
frame of the same stamp, and fits it with the camera pose from TF at that stamp; the next call is sent
before the current frame is fitted. Frames that are not detected are skipped. It stops at the first
`frames` consecutive detected frames whose points all lie within `agree_m` of their median and whose
insertion axes all lie within `agree_deg` of their median heading, and uses the fit of the one nearest
the median point. None within `timeout_s`: it raises `NotLocated` with the reasons.

| Input                             | Where it comes from                                                                           |
| --------------------------------- | --------------------------------------------------------------------------------------------- |
| `cavity_mask`, "microwave" mask   | rammp-perception's `sam3` node, `/perception/sam3/detect`, on `/scene_camera/color/image_raw` |
| `depth_m` (aligned to color), `K` | the scene camera: `/scene_camera/depth/image_rect_raw`, `/scene_camera/color/camera_info`     |
| `world_T_cam`                     | TF: the scene camera's optical frame in `arm_mount`, at the frame's stamp                     |

The dimensions are tape-measured: opening 30.48 x 19.05 cm (12 x 7.5 in), frame to back wall 30.48
cm, side walls flush with the opening, turntable glass 1.59 cm (5/8 in) above the opening's bottom
edge and 17.46 cm below the ceiling, frame 2.86 cm behind the control panel's face.

## The place: `place_motion.py`

`place_motion.py` keeps Walter Hsieh's `place_container` motion (`plan.py` and `lower.py`, from
WalterHsieh0713/rammp-kitchen-capability-engines `feature/7-place-container`): his arm model and the
impedance lowering with the contact stop and hold. The model's IK uses PyBullet's analytic Jacobian of
the tool frame. It adds the microwave: the obstacle, the side-shift and heading search, the checks on
the arm model, the release, and running the checked plan (`run_place`).

The held tupperware (17 cm across) is planned as an 18 x 18 x 10 cm box, level, its centre
`CONTAINER_REACH` (about 10.2 cm) ahead of the tool frame along the hand's level heading, at the tool's
height. It is used only to place the target (the box's centre over the spot: `closer` back from the
turntable's middle toward the opening, along the oven's axis) and the pre-insert pose (the whole box 3
cm out of the oven's front; it moves with the spot); it is not collision-checked. The hand keeps the
tilt it has at the start until the release.

1. **Locate** the opening (above). Nothing is checked before it: not the arm's state, the gripper or
   the tool frame.
1. **Obstacle**: the microwave's body as one box, aligned with the oven, from the "microwave" mask's
   depth: from its outer front (the control panel's face) back, spanning the 2nd to 98th percentile of
   the mask's depth points at or behind that front and near the located opening (within 50 cm across
   its middle, 70 cm behind its plane, 15 cm below and 50 cm above its bottom edge; at least 200
   points). The open door, in front of the front, is not in the box. cuRobo gets the same box (written
   to `worlds_dir`, loaded with `set_world`). Only the side shift and cuRobo's transit to the
   pre-insert pose are checked against it; from the line in on, nothing is collision-checked.
1. **Search**: each of `side_shifts_cm` in turn (3, -3, 6, -6, 9, -9 cm), each with every offset of
   `heading_offsets_deg` (default only 0). The side shift moves the hand straight and level to its
   left (- = right), orientation kept. The hand's heading is the oven's axis turned by the offset (+ to
   the left). The target puts the box's centre over the spot, the tool `above` over the glass; the
   pre-insert pose is back along the heading until the whole box is 3 cm outside the outer front. The
   first shift and heading that pass every stage win: the side shift (`clear_min` from the box, tilt),
   IK at the target and at the pre-insert pose, the straight line in, cuRobo's level transit
   (`plan_to_pose`, `HOLD_LEVEL`) from the shifted pose to the pre-insert pose (refused if it swings
   any joint more than 120 deg in total: cuRobo's pose goal can end in another arm posture), and the
   check below. Every attempt is logged and returned with the stage and reason it failed.
1. **Check** (on the arm model; joint steps of at most 1 deg on the transit and 0.5 deg on the lines,
   the short way round): the side shift and the transit keep every arm and gripper link at least
   `clear_min` from the microwave box (clearance is computed at the first and last sample and wherever
   the joint motion since the last computed one could have moved some point more than that clearance's
   margin over `clear_min`, from each joint's radius; the samples skipped are clear by that bound); the
   tilt stays within `tilt_max_deg` of the start on the side shift, the transit, the line in and the
   line down; cuRobo's end is on the pre-insert pose and the line in stays on its line and ends on the
   target (1 cm, 2 deg); the line down must reach 1 cm below where the box's bottom meets the glass.
   From where the box's bottom meets the glass on the line down (where the model has the lowering
   stop), the release is planned: the hand pitched `RELEASE_TILT_DEG` (10 deg) nose-down about the
   tool frame, and from there the level line back out along the heading to above the pre-insert pose's
   x, y at that height, the hand's orientation kept (only that it stays on its line is checked); none
   of it is collision-checked. A **dry run** stops here and returns the plan (target, pre-insert, side
   shift, heading, the transit's clearance, tilt, joint changes and travel, the approach's duration,
   the line down, attempts); it is written to `state_dir/plans/` and `state_dir/last_plan.json`.
1. **Execute** (only with `execute`): the arm must be ready to command (`not_ready()`) and within 1 deg
   of the plan's start. The side shift, cuRobo's transit and the line in run as planned and checked in
   **one** position-control motion (`approach_knots`), with no stop or check between them. It is timed
   by `pose_lengths`: each segment counts as the tool's travel, the hand's turn x `ROT_LENGTH`, or the
   largest joint change x `ROT_LENGTH` (0.25 m per rad), whichever is longest, so at the default
   `insert_speed` peak nothing turns faster than about 37 deg/s. Only its end is checked: within 1 cm /
   2 deg of the target. The line down is the checked one when the arm is within `REUSE_TOL_DEG` (0.3
   deg) of it, otherwise planned again from where the arm is (its tilt checked). Then Walter's
   lowering: settle in joint impedance for `settle_s`, down until the glass stops it (the lower finger
   touches about 1.3 cm before the box does), hold `hold_press` below the contact. The descent passes
   where the glass should be at about `lower_speed`, and contact counts only while the commanded speed
   is at least 0.75 x the smaller of `lower_speed` and the descent's own peak speed; the descent is
   recorded to `state_dir/descents/`. Then the release: the gripper opened to `RELEASE_OPEN` (0.5) in
   place, still in impedance; the hand pitched `RELEASE_TILT_DEG` nose-down about the tool frame in
   impedance (planned from the measured joints, at `TILT_SPEED_SCALE` of the arm's joint speed) so the
   container slides off the fingers; a stop; out of impedance; and, in position control, the level line
   back out with the hand kept tilted, to above the pre-insert pose's x, y at that height, outside the
   opening (its end checked within 1 cm).

**Why the gripper opens only to 0.5.** The fingers stand vertically. Fully open they span about 15.5 cm
of the 17.5 cm between the glass and the ceiling: 1.0 cm each side at best. Opened to 0.5 (0.4 rad)
they leave about 2 cm free on each side of the container's wall.

A faulted arm makes the driver reject the approach: the place stops there, nothing moved, and says so;
clear the fault and re-run.

## The take-out: `rim_detection.py`, `take_out.py`

`take_out` starts from the take-out start pose: the arm in front of the open oven, the hand pointing into
it, the wrist camera looking at the tupperware from about 0.4-0.5 m, the gripper open. It uses neither the
scene camera nor a model of the oven: the grasp comes from the whole-container mask and filtered wrist depth, the line in
from the hand's own heading.

1. **Detect** (`rim_detection.detect_container`; the filename is historical). SAM3 queries
   `container_prompt` once on the wrist color image and selects the highest-scoring container mask.
   There is no rim query or rim-shape filter. The existing `rim_frames` (default 3), `rim_mask_iou`
   (0.85) and `rim_timeout_s` (60 s) parameter names are retained for deployment compatibility.
   Every container mask must agree with every earlier mask in the consecutive-frame streak.
   Disagreement or a failed detection resets that streak. Within each mask, average the 50
   nearest locally consistent valid depth pixels (finite and strictly between 0.2 and 2 m), transformed into
   `arm_mount` at the depth timestamp. Then average those per-frame points over the three frames.
   This preserves the previous two-stage averaging, without adding a point-distance agreement gate.
   Defaults use zero heading offset and a 3.5 cm upward tool offset from that averaged point.
1. **Grasp pose** (`take_out.grasp_pose`). The hand's level heading at the start is the line in's
   direction. The hand is turned 90 deg about its own axis, the way that points the line between the
   fingers (tool x) straight up, so the fingers come in above and below the brim, and levelled. The
   tool frame's target is `grasp_depth` along the heading and `grasp_up` above the averaged container point; the
   pre-grasp pose is `reach_in` back from it along the heading (30 cm by default).
   The tool frame is about 6 cm behind the fingertips; offsets are tuned for this container.
1. **Plan** (`plan_take_out`). Load `take_approach_world` (default
   `/config/robot/curobo/world.yaml`, a path inside the planner container). Require `set_world` to
   acknowledge the persistent platform, seat and person layers and the floor scene layer. cuRobo
   plans from the measured start joints to the pre-grasp pose, including the hand's turn, with
   orientation free in transit. The approach can curve; only its endpoint must match the pre-grasp.
   From its endpoint, local IK plans the straight insertion and lift. cuRobo then preplans
   the exit from that nominal lifted configuration before any motion is allowed.
   A **dry run** stops here and saves the cuRobo samples, timing and velocities alongside those
   Cartesian paths in `state_dir/plans/take_out_*.json`. A planner failure refuses without a local
   IK fallback.
1. **Execute**. Set the gripper to `grip_open`, execute the exact cuRobo trajectory through the
   driver's `execute_joint_trajectory` in position control, at `take_approach_speed_scale` (default
   0.3 of planned speed), and stop at pre-grasp. Check the measured pre-grasp position and full
   orientation (1 cm / 2 deg), then rebuild the Cartesian paths from the measured joints. The
   insertion up to `impedance_in` short of grasp uses the existing local IK and position control,
   timed at `insert_speed`; the arm then settles in joint impedance (`take_impedance_kq`,
   `take_impedance_zeta`, `impedance_torque_limit`) for `settle_s` and does the rest of the line in
   at `impedance_speed` peak.
   It must end within 3 cm of the target. Impedance is switched off (`exit_impedance`: the position
   setpoint on the measured joints, so nothing jumps) before the gripper closes, whatever happened.
   Then, in position control: the gripper closes with `grab_container`'s force and is checked (the
   fingers stopped, closed past 0.6, not to 1.0 = on nothing; otherwise it stops in the oven without
   lifting), and the hand goes straight up `take_lift`. The exit is replanned with cuRobo
   from the measured lifted joints, then executed with its original samples and timing.

Compliance is used only where the open fingers go over the brim; everything carrying the container is
stiff (the place's soft gains let the arm sag ~5 cm under it). The library runs one trajectory goal at a
time, so the switch into impedance is a short stop. The cuRobo approach and exit are
collision-checked; local insertion and lift are not collision-checked. The approach
world contains the chair, person and floor, not a microwave model or the container. Run the planner
with the current flatbot `planner.yaml` from rammp-deployments, its layers mounted at
`/config/robot/curobo`, and `world:=""`; an empty-world or older planner without the required layer
acknowledgement is refused. The capability does not install or update that deployment configuration.

The wrist camera's stereo returns nothing nearer than about 0.22 m. Start with a clear
view of the container and enough locally consistent depth samples inside its mask.

## Actions

The UI is offered three one-press actions, each running a whole action with fixed params (details in
[docs/interface.md](docs/interface.md), "Orchestrator interface"):

- `retrieve` (on a microwave tap): `take_out`, executing;
- `place_in_microwave` (on a microwave tap): `place`, executing;
- `retrieve_from_fridge` (on a fridge tap): `take_out` with `"site": "fridge"` (the fridge's viewing pose,
  approach world, grasp height and line-in length), executing. Refused unless
  `fridge_accept_unmodeled_door:=true`: nothing it plans knows the fridge, its shelves or its open door.

Each shows one progress panel with Stop (`capability_ui.RunUI`); Stop cancels the run and the motion in flight, and
the "Stopped" notice says what the hand holds. A recoverable refusal before the arm started on the place or take-out
proper offers the action again; a failure once it started (`STOPPED ...`) is shown with Done only. The node must
still run with `allow_execute:=true` for a hardware run; `mock:=true` exercises the panels without arm motion.
`place` and `take_out` remain available for direct execution and dry runs.

This uses `ctx.update_user_panel()` / `ctx.poll_ui_event()` from the
`feat/engine_update` counter example. Run with the orchestrator and UI bridge,
which must grant the same execution token to the engine, arm and bridge. The UI
interface package is pinned in the Dockerfile; native builds need
`rammp_ui_bridge_interfaces` and `capability_ui` (`shared/capability_ui`) in their workspace.
`ui_bridge_node_name` defaults to `ui_bridge_node`. Without a UI session nothing waits for a tap: a refusal is
returned as the recoverable error it is.

```bash
make action CAP=microwave_place ACTION=retrieve
```

For the four-terminal Unreal panel demo, keep the UI bridge and orchestrator
commands. In interactive perception enter `add microwave_1 microwave`, and
replace the counter launch with:

```bash
ros2 launch microwave_place microwave_place.launch.py mock:=true state_dir:="$HOME/.microwave_place"
```

Select **microwave_1 → Retrieve Tupperware from Microwave** in Unreal. Mock mode
shows completion without planning or moving the arm; it does not simulate HOME
planning failure. Real retrieval requires the arm driver, cuRobo, wrist camera,
TF and SAM3 stack in addition to these four demo terminals.

The launch file and deployment fragments set `arm.service_call_s` to 30 seconds:
Snorlax's cuRobo world reload completed after the previous 10-second timeout.
Override this launch setting with `service_call_s:=30.0`. Cancellation remains
available during the wait; the collision-world and HOME checks still apply.

| Action     | Params            | What it does                                                                                                      |
| ---------- | ----------------- | ----------------------------------------------------------------------------------------------------------------- |
| `place`    | `execute` (false) | Locates the oven, plans and checks the whole place and returns it; with `execute`, runs it                        |
| `take_out` | `execute` (false) | Finds the container grasp point from the wrist camera, plans the take-out and returns it; with `execute`, runs it |

A refusal fails the goal with error code `refused` (recoverable).

## Running it standalone

The arm moves only if the node runs with `allow_execute:=true` **and** the goal sets `execute`.
Otherwise every goal is a dry run: it detects, plans and checks, and returns the plan. It needs
SAM3 (rammp-perception) and the cuRobo planner up; the planner must read `worlds_dir` at the same path
(mounted in both containers).

```bash
# natively (colcon), or in the image with `make run CAP=microwave_place STANDALONE=1`
ros2 launch microwave_place microwave_place.launch.py use_orchestrator:=false
make action CAP=microwave_place ACTION=place                          # dry run
```

On the arm, attended, someone on the e-stop:

```bash
ros2 launch microwave_place microwave_place.launch.py use_orchestrator:=false allow_execute:=true \
  state_dir:=$HOME/.microwave_place
make action CAP=microwave_place ACTION=place PARAMS='{"execute": true}'
make action CAP=microwave_place ACTION=take_out PARAMS='{"execute": true}'   # from the take-out start pose
```

## Parameters

All are ROS parameters, with their defaults in `rammp-alternative.yaml`:

| Parameter                                                  | Default                                     | Meaning                                                                                                      |
| ---------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `allow_execute`                                            | false                                       | true lets `execute` goals move the arm (attended runs only)                                                  |
| `frames`                                                   | 3                                           | locate: this many consecutive detected frames must agree                                                     |
| `agree_m`, `agree_deg`                                     | 0.01, 2.0                                   | ... within this of their median                                                                              |
| `timeout_s`                                                | 20.0                                        | locate (with the "microwave" frame): give up after this                                                      |
| `microwave_prompt`                                         | "microwave"                                 | SAM3's prompt for the oven's outside                                                                         |
| `depth_topic`, `camera_info_topic`, `detections_topic`     | the scene camera's                          | locate's topics (the 3D detections are not used)                                                             |
| `above`                                                    | 0.055                                       | m, the tool over the glass at the end of the line in                                                         |
| `closer`                                                   | 0.02                                        | m, the spot this far back from the turntable's middle toward the opening (the pre-insert pose moves with it) |
| `side_shifts_cm`                                           | 3, -3, 6, -6, 9, -9                         | the hand moved this far level to its left before the transit, tried in order                                 |
| `heading_offsets_deg`                                      | 0                                           | the hand's headings off the oven's axis (+ = left), tried with each side shift                               |
| `clear_min`                                                | 0.01                                        | m, every arm and gripper link from the microwave box (side shift, transit)                                   |
| `tilt_max_deg`                                             | 2.0                                         | the hand's tilt vs the start (side shift, transit, line in, line down)                                       |
| `worlds_dir`                                               | /home/snorlax/.ros/rammp_box_opening/worlds | cuRobo's world files; same path in the planner's container                                                   |
| `insert_speed`                                             | 0.16                                        | m/s, the peak speed of the approach (side shift + transit + line in, one motion) and the back-out            |
| `lower_speed`                                              | 0.12                                        | m/s, the lowering where the glass should be (contact tuned at it)                                            |
| `vz_stop`, `hold_press`, `settle_s`, `early_stop`          | -0.01, 0.003, 0.2, 0.03                     | the lowering's contact test and hold                                                                         |
| `impedance_kq`, `impedance_zeta`, `impedance_torque_limit` | the driver's `rammp_arm.IMPEDANCE`          | joint impedance for the lowering, the gripper opening and the release tilt                                   |
| `flange_to_tip`                                            | 0.17955                                     | flange face to the closed fingertips (m)                                                                     |
| `state_dir`                                                | /config/microwave_place                     | plans and the descent recordings                                                                             |
| `take_approach_world`                                      | `/config/robot/curobo/world.yaml`           | take_out: scene loaded by the planner; requires platform, seat, person and floor layers                      |
| `take_approach_speed_scale`                                | `0.3`                                       | take_out: fraction of cuRobo approach speed, finite and in (0, 1\]                                           |
| `wrist_topics`                                             | the wrist camera's colour, depth, info      | take_out: the camera the rim is found on                                                                     |
| `container_prompt`, `rim_prompt`                           | the container with its blue sleeve; its rim | take_out: SAM3's prompts                                                                                     |
| `sam3_threshold`                                           | 0.5                                         | take_out: SAM3's score threshold                                                                             |
| `rim_frames`, `rim_mask_iou`, `rim_timeout_s`              | 3, 0.85, 60.0                               | take_out: consecutive frames whose masks overlap this much, at most this long                                |
| `grasp_depth`, `grasp_up`                                  | -0.05, 0.035                                | take_out: the tool frame this far (m) along the heading (- = short of it) and above the rim point            |
| `reach_in`                                                 | 0.25                                        | take_out: the pre-grasp pose this far back (m): the line in's length                                         |
| `grip_open`                                                | 0.0                                         | take_out: the gripper's opening for the reach in (0 = fully open)                                            |
| `impedance_in`, `impedance_speed`                          | 0.10, 0.08                                  | take_out: the last this far (m) of the line in in joint impedance, at this peak speed (m/s)                  |
| `take_impedance_kq`, `take_impedance_zeta`                 | 400 x 4, 200 x 3; 0.7                       | take_out: that joint impedance (the driver caps kq at 400)                                                   |
| `take_lift`                                                | 0.03                                        | take_out: straight up this far (m) after the close                                                           |

## SAM3 debug view

In Foxglove, add an Image panel on `/microwave_place/sam3_overlay` during a planning-only
`take_out` goal (`execute: false`, `allow_execute: false`). Cyan shows the whole container mask;
yellow dots mark its 50 nearest locally consistent depth samples. The banner shows the score and sample count.
The image preserves the source color header. The topic stays advertised while idle, retaining
the latest image with transient-local durability. In Foxglove use automatic or transient-local
durability to receive the retained image after detection finishes. No rim query or frozen-query topic is used.

The take-out grasp target is 5 cm deeper along the approach heading than before
(`grasp_depth: 0.0`). `reach_in: 0.30` keeps the pre-grasp pose unchanged and
extends the straight insertion by 5 cm (30 cm total); the last 10 cm remains in impedance.

Depth filtering uses masked 5x5 neighborhoods: require at least five valid neighbors and
reject samples farther from the local median than max(1 cm, 3 x 1.4826 x MAD).
Then average the nearest 50 surviving samples per frame, refusing fewer than 50.
The existing three-frame pairwise mask-IoU agreement and frame-point averaging remain unchanged.
The overlay displays exactly the filtered samples used by the estimator.

## Take-out starting home

`take_home_joints` stores seven joint angles in radians in arm_joint_1..7 order,
captured from the viewing pose on 2026-11-11. Executing take-out first uses cuRobo
to reach this configuration before starting wrist detection (unless already within
1 degree per joint). It uses `take_approach_speed_scale` and the confirmed flatbot
collision world. This home step leaves the gripper unchanged. A dry run refuses
when off-home because it cannot observe the future camera view without moving.
The explicit `allow_execute` and goal `execute` gates apply to this move too.

## Per-run recording

`record_bag` defaults to `true` for place, take-out, dry runs and mock runs. Each
accepted action creates `state_dir/runs/<action>_<UTC timestamp>_<unique id>/bag`
(sqlite3 ROS 2 bag) and a sibling `run.json`. The existing `state_dir` volume keeps
these across container restarts. A successful result includes `bag_path` and
`recording_errors`; failed runs log their bag path before behavior begins.

| Parameter      | Default | Meaning                                                       |
| -------------- | ------- | ------------------------------------------------------------- |
| `record_bag`   | `true`  | Record each action; startup failure refuses before motion     |
| `bag_frame_hz` | `1.0`   | Maximum synchronized camera sets per second                   |
| `bag_state_hz` | `25.0`  | Maximum joint, tool and gripper messages per second per topic |

Take-out records its configured wrist color image, color calibration and aligned
depth together (timestamps within 50 ms), plus the SAM3 overlay at `bag_frame_hz`.
Place records the scene color, calibration and depth it uses. Images retain their
original header timestamps and encodings, including raw depth units. These are
sampled observations, not every exact frame consumed by SAM3.

The bag also subscribes to `/joint_states`, `/ee_state`, `/gripper_state`, `/tf`
(up to 10 Hz per transform group), `/tf_static`, `/arm/robot_description`,
`/diagnostics` (1 Hz; errors unthrottled), and `/rosout` (this node's logs and all
nodes' warnings/errors, unthrottled). Missing publishers produce empty topics,
without blocking the run. Static-topic QoS is stored for replay. Recorder
subscriptions belong to the action and do not remain active while idle.

The bag-only `/microwave_place/run_event` topic stores start settings and the final
result or exception. `run.json` also stores message counts and any recording I/O
errors. Failures include a synchronous error log and traceback; cancellation and
exceptions close the bag just like success. Startup recording errors refuse the
action; later recording errors are reported without interrupting arm cleanup.
SIGKILL, power loss, or disk failure can still prevent a clean final bag.

Inspect using `ros2 bag info <bag_path>` or open the bag directory in Foxglove.

## Take-out withdrawal

After the 3 cm vertical lift, local IK generates a fixed-heading, level straight
retreat to pre-grasp x/y at the measured lifted height, then stops. The local
path is previewed before approach, insertion and lift, and regenerated from
measured joints before retreat. This preserves the tested insertion/lift/retreat
geometry. These local paths are not collision-checked against the cuRobo world;
HOME and approach still use cuRobo with the configured collision layers.

A fully closed gripper reading does not establish an empty grasp and does not
block lifting or retreat. Finger motion and insufficient closure still refuse.
Run results report `exit_collision_checked: false` and record the actual arm
states. The experimental cuRobo exit helpers and `take_exit_speed_scale` /
`take_pullback_m` settings are retained but are not used for removal.
