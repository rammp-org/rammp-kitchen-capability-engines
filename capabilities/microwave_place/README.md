# microwave_place

Put the tupperware the gripper is already holding (after `grab_container`) into the open microwave.

1. **Find the spot:** with the container in hand, use the **scene camera** (not the wrist camera: the
   held container blocks its view) to find the open microwave and the free floor inside it, and
   compute where the container goes.
1. **Place:** carry the container there level, set it down on the floor, release, and back out.

## Finding the spot: `microwave_place/interior.py`

`estimate_interior(depth_m, K, world_T_cam, objects_3d)` fits this microwave's known interior
(`DEPTH`, `HEIGHT`, `OPENING`, `LEFT_WALL`, `RECESS` in the code) to the inside surfaces the scene
camera sees in one frame. It returns, in `world` (= `arm_mount`, gravity-aligned, metres):

- `point`: where the container goes: the opening's centre sideways, half-way from the opening to the
  back wall, on the floor (turntable top);
- `insertion_axis` (horizontal, into the oven) and `lateral_axis_left`;
- `opening_corners`: the opening rectangle, in the plane of the frame around it;
- `floor_z`.

It returns `None` (not detected) unless the frame pins every direction with measured surfaces: the
back wall (or the frame and the outer front together) for in/out, the left wall for sideways, the
floor or ceiling for height, all seen through the opening, and exactly one way the oven can face
fits. Nothing is filled in; take the target from the frames that are detected. Pure numpy/scipy,
no ROS; the node that runs it on the live camera comes next.

| Input                             | Where it comes from                                                                                                                                                              |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `depth_m` (aligned to color), `K` | the scene camera (sheppy `scene_camera`, OAK-D Pro W): `/scene_camera/depth/image_rect_raw`, `/scene_camera/color/camera_info`                                                   |
| `world_T_cam`                     | TF: the scene camera's optical frame in `world`                                                                                                                                  |
| `objects_3d`                      | `/perception/object_detections_3d` (rammp-perception): `scene_segmenter` (RF-DETR) finds the `microwave` in the color image, `object_lifter` lifts it to a 3D box with the depth |

The 3D box only says where to look. Everything else comes from the depth's point cloud. The
interior dimensions are this microwave's, measured with it at rest; the right wall is never in the
scene camera's view, so the interior width is not used.
