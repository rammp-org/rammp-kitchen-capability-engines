# Microwave Place — Interface

**Rung:** ad-hoc (declared by this module, not standardized)
**Owner:** @seanliyucheng
**Tier:** experimental

## What it does

Puts the container the gripper is already holding (after `grab_container`) into the open microwave. It
finds the microwave's opening with the scene camera (the known-size opening fitted to SAM3's cavity mask,
checked with depth), models the microwave's body as one obstacle box, tries side shifts and hand headings
until one the arm can reach passes, plans a side shift and cuRobo's level transit to in front of the
opening (both checked against the box on the arm model) and a straight line in, and runs the three as one
position-control motion. It lowers the container onto the turntable glass in impedance until the glass
stops it, opens the gripper part way in place, pitches the hand 10 deg nose-down so the container slides
off the fingers, and backs out level with the hand tilted. Nothing from the line in on is
collision-checked. Without it the microwave demo has no way to put the food in.

It also takes the container out (`take_out`). SAM3 finds the whole tupperware mask on the wrist
image, with no rim query. It averages the 50 nearest locally consistent masked depths per frame, then the
points from three consecutive frames with pairwise container-mask IoU at least 0.85. Each point
uses TF at its depth timestamp. A failed detection or mask disagreement resets the streak.
The grasp offsets and local insertion/lift are retained. Both approach and exit use cuRobo.

## Publishes

| Topic                     | Type              | Rate | Meaning                                                                                                   |
| ------------------------- | ----------------- | ---- | --------------------------------------------------------------------------------------------------------- |
| `/microwave_place/status` | `std_msgs/String` | 1 Hz | Health topic: JSON `state` (idle/executing/paused), `action`, `instance_id`, `mode`, `registered`, `mock` |

The `/microwave_place/sam3_overlay` debug topic (`sensor_msgs/Image`, RGB8) is advertised while the capability runs and updates during
take-out detection. Cyan shows the container mask, yellow dots show its 50 nearest filtered
valid depth samples, and a banner shows the score and sample count. The original color header
is preserved. No rim masks or rim query are used.

## Subscribes

| Topic                                | Type                                           | Required | Meaning                                          |
| ------------------------------------ | ---------------------------------------------- | -------- | ------------------------------------------------ |
| `/tf`, `/tf_static`                  | `tf2_msgs/TFMessage`                           | yes      | The scene camera's pose in `arm_mount`           |
| `/scene_camera/depth/image_rect_raw` | `sensor_msgs/Image`                            | yes      | Depth aligned to color, paired with SAM3's frame |
| `/scene_camera/color/camera_info`    | `sensor_msgs/CameraInfo`                       | yes      | Intrinsics                                       |
| `/perception/object_detections_3d`   | `rammp_perception_interfaces/ObjectDetections` | no       | Declared for the locate topics; not used         |
| `/wrist_camera/color/image_raw`      | `sensor_msgs/Image`                            | take_out | The frame used for whole-container detection     |
| `/wrist_camera/depth/image_rect_raw` | `sensor_msgs/Image`                            | take_out | Depth aligned to color                           |
| `/wrist_camera/color/camera_info`    | `sensor_msgs/CameraInfo`                       | take_out | Intrinsics                                       |

It also calls `/perception/sam3/detect` (`rammp_perception_interfaces/srv/Detect`, prompts "inside of
microwave" and "microwave" on `/scene_camera/color/image_raw`; for take_out, `container_prompt` on
`/wrist_camera/color/image_raw`). Through `self.arm` it also uses the arm driver and
the cuRobo planner (`plan_to_pose` with `HOLD_LEVEL` for place, no hold for take-out approach and `HOLD_FIXED` for exit,
and `set_world`). Take-out loads `take_approach_world` and requires the planner to confirm the platform,
seat and person config layers and the floor scene layer. It executes the returned approach trajectory
with the driver's `execute_joint_trajectory`; insertion/lift keep local IK and insertion impedance.
Exit is preplanned before approach and replanned from the measured lifted joints, with no local fallback.
The planner must run the current flatbot configuration from rammp-deployments; this change does not
update or restart the running planner.

## Orchestrator interface

The UI is offered three one-press actions (`availability()` returns only these; the orchestrator labels each
button with its id, and sends no params). Each runs a whole action with fixed params under
`capability_ui.RunUI` (`shared/capability_ui`):

| Action id              | Offered on a tap of | Runs                                             |
| ---------------------- | ------------------- | ------------------------------------------------ |
| `retrieve`             | `microwave`         | `take_out` `{"execute": true}`                   |
| `place_in_microwave`   | `microwave`         | `place` `{"execute": true}`                      |
| `retrieve_from_fridge` | `fridge`            | `take_out` `{"execute": true, "site": "fridge"}` |

The capability's `perception_classes` is `["microwave", "fridge"]` so a fridge tap activates it; each UI action's
own `perception_classes` picks where it is offered. `place` and `take_out` stay callable directly (dry runs,
CLI) but are not offered to the UI.

Every UI action, in `execute()`:

- `allow_execute` false (not mock): a "not started" notice with Done, then `execution_disabled` (recoverable).
  Nothing moves.
- `retrieve_from_fridge` with `fridge_accept_unmodeled_door` false (the default, not mock): a "not started"
  notice explaining the collision gap below, then `fridge_door_unmodeled` (recoverable). Nothing moves.
- While it runs: one progress panel with **Stop** (sent once; re-sent only if the bridge dropped it, so it no
  longer flashes). Stop cancels the engine's own run, as the orchestrator's cancel does: `self.arm` stops the
  motion in flight and the next `ctx.checkpoint()` raises `Cancelled`. A "Stopped" notice then says what the
  hand was left holding (from the run's last progress phase, e.g. "The hand is holding the container in the
  microwave.").
- A recoverable refusal (`refused`, `home_failed`, `recording`) whose message does **not** start with
  `STOPPED`: the reason, with the same action again (button `retry`, labelled Retrieve / Place) and Stop. A
  tap re-runs it from the start (fresh run bag, detection and plan). These are checks that refused before the
  arm started on the take-out or place proper (HOME may have moved).
- A refusal starting `STOPPED` (anything once the arm has started on the place or the take-out proper: the
  node prefixes any refusal from `run_place` / `run_take_out` that doesn't already), any other error, or an
  exception (e-stop, control taken): a "failed" notice with the reason and what the hand holds, Done only,
  never retried.
- Success: a Done notice. Mock: "Mock: ... No arm motion."
- Every final notice is waited on for at most 60 s before `execute()` returns or raises. An orchestrator
  cancel shows nothing more (the orchestrator has taken the UI back). Without a UI session nothing waits: the
  error is returned.

The engine uses the direct UI bridge API (`ctx.update_user_panel()` / `ctx.poll_ui_event()`).
`ui_bridge_node_name` (default `ui_bridge_node`) selects `/ui_bridge_node/request_user_input` and its `/update`
service. JSON shapes are documented in [UI_JSON_SCHEMA.md](../../../docs/UI_JSON_SCHEMA.md).

### Retrieve from the fridge: the collision gap

`take_out` with `"site": "fridge"` is the microwave take-out with the fridge's settings:
`fridge_take_home_joints` (viewing pose), `fridge_take_approach_world`, `fridge_grasp_up`, `fridge_reach_in`;
its messages name the fridge. Its only collision-checked leg is cuRobo's approach, and its world
(`/config/robot/curobo/world.yaml`: chair, user, floor) has **no fridge cabinet, shelves or open door**; the
reach in, lift and back-out are not checked against anything. The fridge capability models the door
(`fridge/world.py`, from the door file `door.json` its open saves in its own `data_dir`), but that is not
available here: the door file lives in the fridge container's `/fridge_data` (not mounted in this one), its
positions are in `arm_mount` as of the open and go stale if the wheelchair moves after it (the fridge's close
corrects for that with logic of its own), and the model's code is in the fridge package, which this image
doesn't have. Its cabinet box would also contain the container being grasped, so only the door slab and handle
could be added. Until that is done and tried, an **executing** fridge take-out (UI or direct) is refused unless
`fridge_accept_unmodeled_door:=true`, for attended runs only with a clear straight path. Dry runs are not gated.
`fridge_take_home_joints` defaults to the fridge capability's start pose (`fridge/constants.py`
`START_JOINTS_DEG`, in radians), the only recorded fridge-facing pose; it has not been tried as a take-out
viewing pose.

`arm.service_call_s` is 30 seconds in the microwave launch file and deployment
fragments, allowing cuRobo's collision-world reload to finish on Snorlax. The
launch argument `service_call_s` overrides this value. The shared engine default
remains 10 seconds when running the executable directly without a parameter file.

Served under `/microwave_place/`, types from `rammp_orchestrator_interfaces`:

| Name                                                       | Default                                       | Meaning                                                                                                                             |
| ---------------------------------------------------------- | --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `allow_execute`                                            | `false`                                       | `true` lets `execute` goals move the arm (attended runs only)                                                                       |
| `state_dir`                                                | `/config/microwave_place`                     | Plans and the descent recordings                                                                                                    |
| `worlds_dir`                                               | `/home/snorlax/.ros/rammp_box_opening/worlds` | cuRobo's world files; the planner's container sees it at the same path                                                              |
| `frames`                                                   | `3`                                           | Locate: this many consecutive detected frames must agree                                                                            |
| `agree_m`, `agree_deg`                                     | `0.01`, `2.0`                                 | ... within this of their median                                                                                                     |
| `timeout_s`                                                | `20.0`                                        | Locate (with the "microwave" frame): give up after this (s)                                                                         |
| `microwave_prompt`                                         | `"microwave"`                                 | SAM3's prompt for the oven's outside (the obstacle)                                                                                 |
| `above`                                                    | `0.055`                                       | The tool over the glass at the end of the line in (m)                                                                               |
| `closer`                                                   | `0.02`                                        | The spot this far back from the turntable's middle toward the opening, along the oven's axis (m); the pre-insert pose moves with it |
| `side_shifts_cm`                                           | `[3, -3, 6, -6, 9, -9]`                       | The hand moved this far level to its left (- = right) before the transit, tried in order                                            |
| `heading_offsets_deg`                                      | `[0.0]`                                       | The hand's headings off the oven's axis (+ = left), tried in order with each side shift                                             |
| `clear_min`                                                | `0.01`                                        | The side shift and the transit: every arm and gripper link from the microwave box (m)                                               |
| `tilt_max_deg`                                             | `2.0`                                         | The hand's tilt vs the start, on the transit and the lines                                                                          |
| `insert_speed`                                             | `0.16`                                        | Peak speed of the approach (side shift + transit + line in, one motion) and the back-out (m/s)                                      |
| `lower_speed`                                              | `0.12`                                        | The lowering's speed where the glass should be (m/s; contact tuned at it)                                                           |
| `vz_stop`, `hold_press`, `settle_s`, `early_stop`          | `-0.01`, `0.003`, `0.2`, `0.03`               | The lowering's contact test and hold                                                                                                |
| `impedance_kq`, `impedance_zeta`, `impedance_torque_limit` | the driver's `rammp_arm.IMPEDANCE`            | Joint impedance for the lowering, the gripper opening and the release tilt                                                          |
| `flange_to_tip`                                            | `0.17955`                                     | Flange face to the closed fingertips (m)                                                                                            |
| `depth_topic`, `camera_info_topic`, `detections_topic`     | the scene camera's                            | Locate's topics (the 3D detections are not used)                                                                                    |
| `take_approach_world`                                      | `/config/robot/curobo/world.yaml`             | take_out: world path inside the planner container; must confirm flatbot layers and floor                                            |
| `take_approach_speed_scale`                                | `0.3`                                         | take_out: fraction of planned approach speed, finite and in (0, 1\]                                                                 |
| `fridge_take_home_joints`                                  | the fridge start pose (rad)                   | take_out site fridge: the viewing pose, arm_joint_1..7 (rad). Default: fridge `START_JOINTS_DEG`; untried as a take-out pose        |
| `fridge_take_approach_world`                               | `/config/robot/curobo/world.yaml`             | take_out site fridge: the approach world (planner path); same layer checks. The default has no fridge in it                         |
| `fridge_grasp_up`, `fridge_reach_in`                       | `0.035`, `0.30`                               | take_out site fridge: `grasp_up` and `reach_in` (m); the microwave's values, untuned for the fridge                                 |
| `fridge_accept_unmodeled_door`                             | `false`                                       | Lets an executing fridge take-out run although nothing it plans knows the fridge or its open door (attended runs only)              |
| `wrist_topics`                                             | the wrist camera's colour, depth, camera_info | take_out: the camera the rim is found on                                                                                            |
| `container_prompt`, `sam3_threshold`                       | the container with its blue sleeve; `0.5`     | take_out: SAM3's prompts and score threshold                                                                                        |
| `rim_frames`, `rim_mask_iou`, `rim_timeout_s`              | `3`, `0.85`, `60.0`                           | take_out: historical parameter names; consecutive whole-container masks must overlap this much (IoU), at most this long (s)         |
| `grasp_depth`, `grasp_up`                                  | `0.0`, `0.035`                                | take_out: the tool frame this far along the hand's heading (- = short of it) and above the rim point (m)                            |
| `reach_in`                                                 | `0.30`                                        | take_out: the pre-grasp pose this far back from the grasp (m)                                                                       |
| `grip_open`                                                | `0.0`                                         | take_out: the gripper's opening for the reach in (0 = fully open)                                                                   |
| `impedance_in`, `impedance_speed`                          | `0.10`, `0.08`                                | take_out: the last this far (m) of the line in in joint impedance, at this peak speed (m/s)                                         |
| `take_impedance_kq`, `take_impedance_zeta`                 | `[400]*4 + [200]*3`, `0.7`                    | take_out: that joint impedance (torque limits: `impedance_torque_limit`)                                                            |
| `take_lift`                                                | `0.03`                                        | take_out: straight up this far after the close (m)                                                                                  |

The capability engine itself also reads `use_orchestrator` (default `true`; `false` runs standalone: no
registration, drive it by CLI) and `mock` (default `false`; `true` skips hardware, the mock alternative
sets it).

`self.arm` reads the `arm.*` parameters once at startup; they are listed with their defaults in the real
fragment and described in the example capability's interface page.

## Hardware

The Kinova Gen3 arm with the Robotiq 2F-85 gripper, the scene camera (place) and the wrist camera
(take_out), through the arm driver, the
cuRobo planner and rammp-perception's SAM3 node. No devices of its own.

## Alternatives

| id                     | What it is                                                                         |
| ---------------------- | ---------------------------------------------------------------------------------- |
| `microwave_place_real` | The module proper                                                                  |
| `microwave_place_mock` | Same interface, no hardware — for developing against this module without owning it |

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
