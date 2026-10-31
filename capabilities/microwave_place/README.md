# microwave_place

Put the tupperware the gripper is already holding (after `grab_container`) into the open microwave.

1. **Find the spot:** with the container in hand, use the **scene camera** (not the wrist camera: the
   held container blocks its view) to find the open microwave and the free floor inside it, and
   compute where the container goes.
1. **Place:** carry the container there level, set it down on the floor, release, and back out.

## Finding the spot: `microwave_place/interior.py`

`estimate_interior(color_bgr, depth_m, K, world_T_cam, objects_3d)` fits the open microwave's
interior box to one scene-camera frame. It returns:

- `bottom_middle`: the middle of the interior floor, where the container goes, in `world` (=
  `arm_mount`, gravity-aligned, metres);
- `insertion_axis`: horizontal, into the oven;
- `interior_box`: centre, yaw, size and 8 corners; also the opening, the floor, ceiling and table
  heights;
- `measured` / `assumed`: which surfaces were seen in the depth and which were taken from the
  opening's edges or a prior;
- `error` instead of the geometry when there is no open microwave to fit.

Pure numpy/scipy, no ROS; the node that runs it on the live camera comes next.

| Input                                          | Where it comes from                                                                                                                                                              |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `color_bgr`, `depth_m` (aligned to color), `K` | the scene camera (sheppy `scene_camera`, OAK-D Pro W): `/scene_camera/color/image_raw`, `/scene_camera/depth/image_rect_raw`, `/scene_camera/color/camera_info`                  |
| `world_T_cam`                                  | TF: the scene camera's optical frame in `world`                                                                                                                                  |
| `objects_3d`                                   | `/perception/object_detections_3d` (rammp-perception): `scene_segmenter` (RF-DETR) finds the `microwave` in the color image, `object_lifter` lifts it to a 3D box with the depth |

The detection only seeds the fit (the region to look in and a yaw guess): with the door open its
box covers the door too. The fit itself works on the depth: the oven's yaw from its vertical faces,
the front plane and the opening in it, then the inside walls, floor and ceiling where the camera
sees them. The scene camera sits about level with the oven floor, so it sees the floor nearly
edge-on, and the wall on its own side of the opening is hidden; that wall is taken from the
opening's edge.
