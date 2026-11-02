# microwave_place

Put the tupperware the gripper is already holding (after `grab_container`) into the open microwave.

- `microwave_place/interior.py`: where the container goes, from the **scene camera** (not the wrist
  camera: the held container blocks its view).
- `microwave_place/place.py`: the motion, planned in full before anything moves.
- `microwave_place/node.py`: the `CapabilityEngine` that runs them.

## Finding the spot: `interior.py`

`fit_interior(depth_m, K, world_T_cam, cavity_mask)` places this microwave's opening, a rectangle of
known size standing upright, from one scene-camera frame. It assumes nothing about where the oven
stands or which way it faces; only that it is upright and has the tape-measured dimensions in the
constants block at the top of `interior.py`. It returns, in `arm_mount` (gravity-aligned, metres):

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

`locate(...)` runs it on the live camera inside an action: each round it asks SAM3
(`/perception/sam3/detect`) for the cavity on the next color frame, pairs the mask with the depth
frame of the same stamp, and fits it with the camera pose from TF at that stamp, until `frames` are
detected. It returns their median `point`, `insertion_axis` and `floor_z` only if they agree to
within `agree_m` and `agree_deg`; otherwise it raises `NotLocated` with the last reasons.

| Input | Where it comes from | | --------------------------------- |
\-----------------------------------------------------------------------------------------------------------------------------
| | `cavity_mask` | rammp-perception's `sam3` node, `/perception/sam3/detect`, prompt "inside of
microwave", on `/scene_camera/color/image_raw` | | `depth_m` (aligned to color), `K` | the scene
camera (sheppy `scene_camera`): `/scene_camera/depth/image_rect_raw`,
`/scene_camera/color/camera_info` | | `world_T_cam` | TF: the scene camera's optical frame in
`arm_mount`, at the frame's stamp |

The dimensions are tape-measured: opening 30.48 x 19.05 cm (12 x 7.5 in), frame to back wall 30.48
cm, side walls flush with the opening, turntable glass 1.59 cm (5/8 in) above the opening's bottom
edge and 17.46 cm below the ceiling, frame 2.86 cm behind the control panel's face.

## The place: `place.py`

The hold model, the arm model, the impedance lowering with the contact stop, the hold and the
release come from `place_container`. The hand holds the container by its lip, level, pointed
`pitch_deg` down. `plan_microwave` plans and checks every leg before anything moves; `run` executes
it, planning and checking each leg again from the measured joints just before it runs:

1. **adjust**: square the hand in place to the nearest level hold (`move_joints`).
1. **lift**: straight up to the height it reaches in at, if lower (`move_joints`).
1. **transit**: to pre-insert, in front of the oven, heading along its insertion axis, the
   container's far edge `standoff` in front of the control panel's face. cuRobo plans it
   (`plan_to_pose`, level hold); the plan is checked and then run exactly as planned, its own
   samples (`execute_joint_trajectory`, `go_to_speed_scale`).
1. **insert**: straight in along the axis until the lip's centre is over the spot (`move_joints`,
   stiff, `insert_speed`).
1. **lower**: straight down in impedance until the floor stops it, then hold `hold_press` below the
   contact (`move_joints`, impedance).
1. **open** the gripper.
1. **back out**: straight back along the axis the whole way to pre-insert, gripper open, the first
   quarter lifting the hold's press (`move_joints`, impedance).

The arm is left in impedance, held where the back-out ended.

**The height it reaches in at** (`insert_height`): the container's lowest point at least
`insert_clearance` above the turntable glass, and every link and the container at least
`clear_inside` below the ceiling (the opening's top is flush with it), measured on the arm's own
model in the sim at the end of the way in. The room between goes half to each; no room refuses.

**Obstacles.** cuRobo's collision world is empty, so the check is done here, in the arm's PyBullet
copy: every configuration of every leg, sampled at most `check_step_m` apart (and 1 deg per joint),
including between waypoints. Every moving link of the arm and gripper (fingers open on the way out)
and the held container (a cylinder of the lip's radius, `BOWL_DEPTH` tall, hanging from the lip at
the tool frame) are measured against:

- the **oven model** (`Oven`): five boxes around the free interior from the located pose and
  `interior.py`'s tape-measured constants: the side walls, the glass, the ceiling, the back wall,
  and together the front, a plate with the opening in it. The body beyond the measured cavity is
  taken to reach `OVEN_BEYOND` past it (left 6 cm, right 16 cm for the control panel, above 10,
  below 10, behind 10). The opening's bottom is taken at the glass and the frame at the panel's
  face, both more than the real oven;
- the **scene voxels**: `obstacle_frames` new scene-camera depth frames, taken after `locate` with
  the arm still, as `voxel_m` cubes, within `region_m` of the arm's base, except points within
  `surface_band` of the oven model (its own surfaces) or within `self_band` of the arm and the
  container (the robot itself). They are the open door, the table and anything else.

Clearances: every link `clear_link`, the container `clear_container` from everything; on insert,
lower and back out, `clear_inside` from the oven model; while it is lowered the container may reach
the glass. Any configuration that fails refuses the whole place before anything moves, naming the
closest pair; a check that fails at execution stops before that leg.

**Frames.** `/ee_state` (`self.arm.ee_state()`) is the flange, `arm_end_effector_link`.
`get_state()["ee_pos"]` is the tool frame, `arm_tool_frame`, `TOOL_Z` (12 cm) further along the
flange's z, which is what `go_to_ee_pose`, cuRobo and `Model.fk()` use. `place` first compares both
with the model and refuses a mismatch (reading one as the other is 12 cm off); the lowering compares
`/ee_state` only with the model's flange.

**Show the plan.** Every `place` publishes what it planned on `/microwave_place/plan`
(`visualization_msgs/MarkerArray`, frame `arm_mount`, latched, so it stays after the action):

| namespace | what | | ----------- |
------------------------------------------------------------------------------ | | `oven` | the oven
model's five boxes (the glass brown) | | `voxels` | the scene voxels (orange cubes) | | `target` |
the spot (green sphere) and the insertion axis (arrow) | | `path` | the tool's path on each leg, one
colour per leg; red if it failed | | `clearance` | per leg: its tightest pair and the clearance it
needs, or why it failed | | `closest` | a line between the tightest pair's two points | |
`container` | the container at pre-insert, inserted and resting |

The result returns the same per-leg clearances (`clearance`).

## Actions

| Action | Params | What it does | | -------- | ------------------------------------------------ |
\-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| | `locate` | none | No motion. Returns the agreed `point`, `insertion_axis`, `floor_z` in
`arm_mount`, with `frames`, `spread_m`, `spread_deg`. | | `place` | `execute` (false),
`allow_empty_gripper` (false) | Checks the arm's flange and tool frame against the model and that
the gripper is not fully open, locates, captures the obstacles, plans and checks the whole place,
publishes it on `/microwave_place/plan` and returns it with the per-leg clearances. With `execute`
(and `allow_execute:=true`), runs it, checking each leg again, and records `/ee_state` (the flange)
of the lowering and back-out to `state_dir/descents/`. |

A refusal fails the goal with error code `refused` (recoverable).

## Running it standalone

The arm moves only if the node runs with `allow_execute:=true` **and** the goal sets `execute`.
Otherwise every `place` is a dry run: it locates, plans and checks, and returns the plan.

```bash
# natively (colcon), or in the image with `make run CAP=microwave_place STANDALONE=1`
ros2 launch microwave_place microwave_place.launch.py use_orchestrator:=false

make action CAP=microwave_place ACTION=locate
make action CAP=microwave_place ACTION=place                          # dry run
```

On the arm, attended, someone on the e-stop:

```bash
ros2 launch microwave_place microwave_place.launch.py use_orchestrator:=false allow_execute:=true \
  state_dir:=$HOME/.microwave_place
make action CAP=microwave_place ACTION=place PARAMS='{"execute": true}'
```

Tunables (speeds, impedance gains, clearances, topics, `frames`/`agree_*`) are ROS parameters,
listed with their defaults in `rammp-alternative.yaml`.
