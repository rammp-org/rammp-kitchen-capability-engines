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
scene camera nor a model of the oven: the grasp comes from the rim the wrist camera finds, the line in
from the hand's own heading.

1. **Detect** (`rim_detection.detect_rim`, `grab_container`'s wrist detection on this node). SAM3
   (`/perception/sam3/detect`) masks `container_prompt` (the container with its sleeve) on the wrist
   colour frame; the best-scoring object is the container. The same frame is republished on
   `/microwave_place/wrist_query_image` and SAM3 masks `rim_prompt` on it (a combined call's label image
   holds one object per pixel, and the rim lies inside the container); the rim is a wide, flat object
   mostly (85%) inside the container mask. It waits for `rim_frames` consecutive frames whose container
   and rim masks overlap every earlier one of the streak by at least `rim_mask_iou` (intersection over
   union). On each of them the 5 nearest depths inside both masks (the wrist depth aligned to colour,
   0.2-2 m), back-projected and put in `arm_mount` with TF at the frame's stamp, are averaged; the rim
   point is the mean over the frames. Nothing within `rim_timeout_s`: it refuses with every reason a
   frame was rejected.
1. **Grasp pose** (`take_out.grasp_pose`). The hand's level heading at the start is the line in's
   direction. The hand is turned 90 deg about its own axis, the way that points the line between the
   fingers (tool x) straight up, so the fingers come in above and below the brim, and levelled. The
   tool frame's target is `grasp_depth` along the heading and `grasp_up` above the rim point; the
   pre-grasp pose is `reach_in` back from it along the heading. The offsets are measured on the arm:
   the nearest rim depths read the bowl wall behind the brim's thin edge (the wrist stereo is 320 x 200),
   and the tool frame is about 6 cm behind the fingertips.
1. **Plan** (`plan_take_out`, on the arm model, joint waypoints by IK): from the start, a straight move to
   the pre-grasp pose while the hand turns (slerped; refused if a step moves any joint more than 15 deg),
   the straight level line in (every pose on its line and heading, ending within 1 cm / 2 deg of the
   target), the raise and the line out. A **dry run** stops here and returns the plan (also written to
   `state_dir/plans/take_out_*.json`).
1. **Execute**. The gripper set to `grip_open`. The turn and the line in up to `impedance_in` short of
   the grasp run as **one** position-control motion, timed by `pose_lengths` at `insert_speed` peak; a
   short stop there; the arm settles in joint impedance (`take_impedance_kq`, `take_impedance_zeta`,
   `impedance_torque_limit`) for `settle_s` and does the rest of the line in at `impedance_speed` peak.
   It must end within 3 cm of the target. Impedance is switched off (`exit_impedance`: the position
   setpoint on the measured joints, so nothing jumps) before the gripper closes, whatever happened.
   Then, in position control: the gripper closes with `grab_container`'s force and is checked (the
   fingers stopped, closed past 0.6, not to 1.0 = on nothing; otherwise it stops in the oven without
   lifting), the hand goes straight up `take_lift`, and the level line out to above the pre-grasp pose.

Compliance is used only where the open fingers go over the brim; everything carrying the container is
stiff (the place's soft gains let the arm sag ~5 cm under it). The library runs one trajectory goal at a
time, so the switch into impedance is a short stop. Nothing is collision-checked: the start faces the
opening, and the line in runs straight along the hand's heading.

The wrist camera's stereo returns nothing nearer than about 0.22 m; from 0.4-0.5 m the rim has 90-100%
depth. At about 0.55 m SAM3's rim mask drops out on many frames.

## Actions

| Action     | Params            | What it does                                                                                    |
| ---------- | ----------------- | ----------------------------------------------------------------------------------------------- |
| `place`    | `execute` (false) | Locates the oven, plans and checks the whole place and returns it; with `execute`, runs it      |
| `take_out` | `execute` (false) | Finds the rim from the wrist camera, plans the take-out and returns it; with `execute`, runs it |

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
