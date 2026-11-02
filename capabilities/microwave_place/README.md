# microwave_place

Put the tupperware the gripper is already holding (after `grab_container`) into the open microwave.

- `microwave_place/microwave_detection.py`: where the microwave is and where the container goes, from the
  **scene camera** (not the wrist camera: the held container blocks its view).
- `microwave_place/place_motion.py`: planning, checking and running the arm: Walter Hsieh's
  `place_container` motion (arm model, impedance lowering), with the transit, the straight lines and the
  release the microwave needs.
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
- `floor_z`, `pins` (what fixed each direction), `fit` and `edges` (for drawing).

How it works:

1. **The opening's edges, from the image.** SAM3, prompted with "inside of microwave", masks the
   cavity. The mask's outline is the inner edge of the frame around the opening. Its straight sides
   are found and each is called vertical or horizontal from where world-up points in the image
   (camera pose from TF). Sides along the image border or the open door are not edges. A lone side
   is named by where the cavity lies: below a top edge, above a bottom edge, right of a left edge
   (facing the oven, its left is on the image left), left of a right edge.
1. **The rectangle, by perspective.** The opening's four unknowns (where the middle of its bottom
   edge is, and its turn about vertical) are fitted so the projected 30.48 x 19.05 cm rectangle lies
   on the sides found. One vertical and one horizontal side are enough; no depth is needed on the
   frame, so it works from steep angles and up close.
1. **Depth checks the result.** What is seen through the opening must lie inside the oven's depth
   behind it (this also decides which way the oven faces), and the frame strips' depth must lie on
   the fitted plane. Two different poses that pass mean "not detected".
1. **Depth refines the distance.** The frame strips' depth points are added to the fit: the edges
   fix the turn and the shape, the depth the distance, which the edges alone pin only loosely at
   steep angles.
1. **The point** follows from the dimensions: centred sideways, `DEPTH / 2` behind the frame plane,
   `FLOOR` above the opening's bottom edge.

Any check that fails raises `NotDetected` with the reason (`estimate_interior` returns `None`).
Nothing is filled in.

`locate(...)` runs it on the live camera inside an action (`SceneCamera`): each round it asks SAM3
(`/perception/sam3/detect`) for the cavity on the next color frame, pairs the mask with the depth
frame of the same stamp, and fits it with the camera pose from TF at that stamp. The place asks for up
to `max_frames` detected frames and needs at least `frames` of them within `agree_m` and `agree_deg`
of their median; it uses the fit of the agreeing frame nearest the median. Otherwise it raises
`NotLocated` with the reasons. `scene_frame(...)` gets one more frame, SAM3's "microwave" mask with its
depth and camera pose, for the obstacles.

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
WalterHsieh0713/rammp-kitchen-capability-engines `feature/7-place-container`) unchanged: his arm model and
the impedance lowering with the contact stop and hold. It adds the microwave: the obstacles, the heading
search, the checks on the arm model, the release, and running the checked plan (`run_place`).

The held tupperware is modelled as a 17 x 17 x 10 cm box centred on the hand's axis, its centre
`model.reach` ahead of the tool frame. The hand keeps the tilt it has at the start throughout.

1. **Checks**: the arm may be commanded (`not_ready()`), the gripper is closed (unless
   `allow_empty_gripper`), the arm's tool frame matches the model's.
1. **Locate** the opening (above).
1. **Obstacles**: the oven as 5 boxes around its free cavity and opening, out to its outside (the
   "microwave" mask's depth points at or behind the outer front); everything the camera sees within
   50 cm in front of the outer front and 45 cm to its sides (the open door) as depth columns, from 3 cm
   cells up until at most 53 fit (cuRobo keeps 60 boxes). Columns at the robot or the held box at the
   start are dropped. cuRobo pads every box 2 cm and the gaps inside the oven are 1-2 cm, so its copy
   (written to `worlds_dir`, loaded with `set_world`) has the shell's inner faces pulled back 2 cm. Our
   own checks use the true boxes.
1. **Heading**: each offset of `heading_offsets_deg` in turn (the hand's heading = the oven's axis
   turned that far, + to the left). The target puts the box's centre over the placement point, the tool
   `above` over the glass; the pre-insert pose is back along the heading until the whole box is 3 cm
   outside the outer front. The first offset that passes every stage wins: IK at the target with
   `clear_min` clearance, IK at the pre-insert pose, the straight line in clear by `clear_min`, cuRobo's
   level transit (`plan_to_pose`, `HOLD_LEVEL`) to the pre-insert pose (refused if it swings any joint
   more than 120 deg in total: cuRobo's pose goal can end in another arm posture), and the check below.
   Every attempt is logged and returned with the stage and reason it failed.
1. **Check** (dense, joint steps of at most 1 deg on the transit and 0.5 deg on the lines, the short way
   round): every arm and gripper link at least `clear_min` from every true box (the held box too with
   `check_container`), tilt within `tilt_max_deg` of the start, cuRobo's end on the pre-insert pose and
   the line in's end on the target (1 cm, 2 deg); the line down must reach 1 cm below where the box
   meets the glass, and it (down to that point) is checked against every box but the floor under the
   glass, on which the lower finger lands. From where the model has the lowering stop: the **release
   lift** (the smallest straight lift after which the fingers, opened to `RELEASE_OPEN`, are `clear_min`
   from the glass and the ceiling) and the **back-out** (back onto the line in's end, then the line in's
   joints reversed), both with the fingers at that opening. A **dry run** stops here and returns the plan
   (target, pre-insert, heading, clearances per leg, tilt, joint changes and travel, durations, attempts);
   it is written to `state_dir/plans/` and `state_dir/last_plan.json`.
1. **Execute** (only with `execute`): the arm must be within 1 deg of the plan's start. The transit
   runs exactly as checked (`execute_joint_trajectory`, starting from the measured joints, cuRobo's
   timing slowed to `transit_speed`); it must end within 1 cm / 2 deg of the pre-insert pose. The line
   in is planned and checked again from where the arm is and run at `insert_speed`; it must end within
   1 cm / 2 deg of the target. Then Walter's lowering: the line down planned and checked from there,
   settle in impedance, down until the glass stops it (the lower finger touches about 1.6 cm before the
   box does), hold `hold_press` below the contact. Then position control (impedance moves lag too far
   for the gaps inside), each step planned from the measured joints, checked, and its end measured: the
   release lift, the gripper opened to `RELEASE_OPEN`, and the back-out to the pre-insert pose, outside
   the opening. The descent is recorded to `state_dir/descents/`.

**Why the gripper opens only to 0.5.** The fingers stand vertically. Fully open they span about 15.5 cm
of the 17.5 cm between the glass and the ceiling: 1.0 cm each side at best. Opened to 0.5 (0.4 rad)
they leave about 2 cm free on each side of the container's wall, and after a ~1.5 cm lift they are 1.2 cm
from the glass and over 2 cm from the ceiling.

A faulted arm makes the driver reject the transit: the place stops there and says so; clear the fault
and re-run.

## Actions

| Action    | Params                                   | What it does                                                                               |
| --------- | ---------------------------------------- | ------------------------------------------------------------------------------------------ |
| `place`   | `execute` (false), `allow_empty_gripper` | Locates the oven, plans and checks the whole place and returns it; with `execute`, runs it |
| `retreat` | `execute`                                | The release lift and back-out alone (gripper open), retracing the last executed line in    |
| `reset`   | `execute`, `save`, `empty`               | Back to the start position (straight up, then a joint move), checked against the last plan |

A refusal fails the goal with error code `refused` (recoverable).

## Running it standalone

The arm moves only if the node runs with `allow_execute:=true` **and** the goal sets `execute`.
Otherwise every `place` is a dry run: it locates, plans and checks, and returns the plan. It needs
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
```

## Parameters

All are ROS parameters, with their defaults in `rammp-alternative.yaml`. The place's own:

| Parameter              | Default                                     | Meaning                                                              |
| ---------------------- | ------------------------------------------- | -------------------------------------------------------------------- |
| `frames`, `max_frames` | 3, 5                                        | locate: need `frames` of up to `max_frames` detected frames to agree |
| `agree_m`, `agree_deg` | 0.01, 2.0                                   | ... within this of their median                                      |
| `timeout_s`            | 20.0                                        | locate (and the "microwave" frame): give up after this               |
| `microwave_prompt`     | "microwave"                                 | SAM3's prompt for the oven's outside                                 |
| `above`                | 0.08                                        | m, the tool over the glass at the end of the line in                 |
| `heading_offsets_deg`  | 0, 5, -5, ... 40, -40                       | the headings tried, in order                                         |
| `clear_min`            | 0.01                                        | m, every link (and the box) from every true obstacle box             |
| `tilt_max_deg`         | 2.0                                         | the hand's tilt vs the start                                         |
| `check_container`      | false                                       | also keep the held box `clear_min` from the obstacles                |
| `worlds_dir`           | /home/snorlax/.ros/rammp_box_opening/worlds | cuRobo's world files; same path in the planner's container           |
| `transit_speed`        | 1.0                                         | the transit at this fraction of cuRobo's timing                      |
| `insert_speed`         | 0.16                                        | m/s, the line in, the release lift and the back-out                  |
| `state_dir`            | /config/microwave_place                     | plans, start position, descent recordings                            |

Walter's lowering keeps its own: `lower_speed`, `vz_stop`, `hold_press`, `settle_s`, `early_stop`,
`impedance_*`.
