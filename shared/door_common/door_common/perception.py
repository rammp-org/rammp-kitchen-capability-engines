"""The wrist camera, and the handle detector that reads its frames.

WristCamera: the latest RGB-D frame and where the camera was, in the arm's base frame.

Replaces feeding_deployment's RealSenseROS2Interface (colour + aligned depth + camera_info,
synchronised) and TFInterface (tf2 lookup base <- camera) for this capability: same frame contents
(colour bgr8, depth float32 mm aligned to colour, K), on the capability's own node. Read-only:
nothing here commands anything.

The base frame is the arm's FRAME (`arm.frame`, arm_mount on the nightly driver): the frame
/ee_state reports and every goal is in, so detections come out in the same frame as the arm's
poses. (The Jetson scripts looked up `arm_base_link`, an alias of the 1.1.0 driver's base_link.)

HandleDetector: a door's handle + face from one RGB-D frame, in the arm's base frame.

The handle-finding half of `AppliancePerception.detect_handle_and_placement`, copied as-is
(same thresholds, same order of operations) so the grasp constants tuned against it still hold:

1. The door's region of the image (`door_region`, the capability's).
2. Every region pixel with valid depth -> 3D point (camera frame); `crop_points` may narrow them.
3. RANSAC plane = the door face; reject it if its normal isn't horizontal (table/floor).
4. Points 0-7 cm in front of the plane -> DBSCAN -> the most vertical, elongated cluster.
5. Handle = the cluster's median (the middle of the bar).
6. Hinge = the door-face point on the edge farthest from the handle, nearest the handle's height.
   Reject if that edge is not clearly on the far side (the plane pulled in background).
7. Into the arm's base frame with the frame's base_T_cam (perception.WristCamera, tf2).

Dropped vs the original: the hinge/placement/top-of-appliance poses, overlays, RViz markers,
image logging, the camera flip (this rig's camera is upright) and HANDLE_DEPTH_CORR (0 here).
"""

import threading
import time

import message_filters
import numpy as np
import open3d as o3d
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R

from .common import say

TF_TIMEOUT_S = 2.0  # wait this long for base <- camera at the frame's stamp, then use the latest
SYNC_SLOP_S = 0.05  # colour, depth and camera_info within this of each other count as one frame
BOX_MAX_AGE_S = 5.0  # a boxer box older than this is not used
SAM3_TIMEOUT_S = 10.0  # one SAM 3 detect call (~0.5 s on the Orin, plus waiting for the next frame)


class _Input(message_filters.SimpleFilter):
    """A message_filters input fed by `subscribe` (create_action_subscription, or create_subscription on a plain
    node): message_filters.Subscriber always makes a plain subscription."""

    def __init__(self, subscribe, msg_type, topic, qos):
        super().__init__()
        subscribe(msg_type, topic, self.signalMessage, qos)


class WristCamera:
    def __init__(
        self,
        node,
        color_topic,
        info_topic,
        depth_topic,
        base_frame,
        camera_frame="",
        box_topic="",
        box_class="",
        sam3=None,
        sam3_prompts=("handle",),
        sam3_threshold=0.3,
        box2d_topic="",
    ):
        """`camera_frame` "" = the colour image's own frame_id. Depth must be aligned to colour.
        `box_topic`: boxer's 3D boxes (rammp_perception_interfaces/ObjectDetections, from the scene camera); each
        frame then carries the latest `box_class` box in the base frame, for the capability's detector. "" = no boxer.
        `sam3`: a client of SAM 3's Detect service (made once, for the node's
        lifetime: a client gets no messages) -- `sam3_handle_mask` asks it for `sam3_prompts` on this camera's colour
        image. None = none.
        `box2d_topic`: a 2D detector's ObjectDetections on THIS camera's image (the scene segmenter, e.g. RF-DETR on
        /perception/object_detections); each frame then carries the latest visible `box_class` box, `box2d`. "" = none.
        `fixed_box`: a box set by the caller (e.g. placed from another camera), carried as each frame's `box` when
        there is no box_topic.

        On a CapabilityEngine node the subscriptions and the tf buffer are the running action's
        (create_action_subscription, create_action_tf_buffer): make the camera inside execute(); the engine destroys
        them when the action ends. On a plain node (the bench scripts) they are plain ones that live as long as the
        node."""
        import tf2_ros
        from cv_bridge import CvBridge
        from sensor_msgs.msg import CameraInfo, Image

        self.node, self.base_frame, self.camera_frame = node, base_frame, camera_frame
        self.box_topic, self.box_class = box_topic, box_class
        self.color_topic, self.sam3_prompts, self.sam3_threshold = color_topic, list(sam3_prompts), sam3_threshold
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest = None
        self._box = None  # (header, DetectedObject): the latest box_class object with a pose
        self._box2d = None  # (header, DetectedObject): the latest visible box_class object on box2d_topic
        self.box2d_topic, self.fixed_box = box2d_topic, None
        if hasattr(node, "create_action_subscription"):
            subscribe, self.tf_buffer = node.create_action_subscription, node.create_action_tf_buffer()
        else:
            subscribe, self.tf_buffer = node.create_subscription, tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self.tf_buffer, node)
        qos = qos_profile_sensor_data
        self._inputs = [
            _Input(subscribe, Image, color_topic, qos),
            _Input(subscribe, CameraInfo, info_topic, qos),
            _Input(subscribe, Image, depth_topic, qos),
        ]
        self._sync = message_filters.ApproximateTimeSynchronizer(self._inputs, 10, SYNC_SLOP_S)
        self._sync.registerCallback(self._on_frame)
        if box_topic:
            from rammp_perception_interfaces.msg import ObjectDetections

            subscribe(ObjectDetections, box_topic, self._on_boxes, qos)
        if box2d_topic:
            from rammp_perception_interfaces.msg import ObjectDetections

            subscribe(ObjectDetections, box2d_topic, self._on_boxes_2d, qos)
        self._sam3 = sam3
        self.topics = (color_topic, info_topic, depth_topic)

    def _on_frame(self, color, info, depth):
        with self._lock:
            self._latest = (color, info, depth)

    def _on_boxes(self, msg):
        objs = [o for o in msg.objects if o.perception_class == self.box_class and o.has_pose]
        if objs:
            with self._lock:
                self._box = (msg.header, max(objs, key=lambda o: o.score))

    def _on_boxes_2d(self, msg):
        objs = [
            o
            for o in msg.objects
            if o.perception_class == self.box_class and o.state == o.STATE_VISIBLE and o.box[2] > o.box[0]
        ]
        if objs:
            with self._lock:
                self._box2d = (msg.header, max(objs, key=lambda o: o.score))

    def appliance_box_2d(self):
        """The latest visible `box_class` box on box2d_topic: {xyxy (ints, this camera's pixels), score}, or None (says
        why)."""
        with self._lock:
            latest = self._box2d
        if latest is None:
            say(f"  no visible '{self.box_class}' on {self.box2d_topic} yet (scene segmenter up?)")
            return None
        header, obj = latest
        age = (self.node.get_clock().now() - Time.from_msg(header.stamp)).nanoseconds / 1e9
        if age > BOX_MAX_AGE_S:
            say(f"  the last '{self.box_class}' 2D box is {age:.1f} s old (max {BOX_MAX_AGE_S:g})")
            return None
        return {"xyxy": [int(round(v)) for v in obj.box], "score": float(obj.score)}

    def sam3_handle_mask(self):
        """SAM 3's mask of the handle on this camera's next colour frame (bool, image-sized): the best-scoring instance
        of any of `sam3_prompts`. None (says why) without a service, a reply or an instance."""
        if self._sam3 is None:
            return None
        if not self._sam3.wait_for_service(timeout_sec=2.0):
            say(f"  no SAM 3 service {self._sam3.srv_name} (is the sam3 node up?)")
            return None
        req = self._sam3.srv_type.Request(
            prompts=self.sam3_prompts, image_topic=self.color_topic, score_threshold=float(self.sam3_threshold)
        )
        done, fut = threading.Event(), self._sam3.call_async(req)
        fut.add_done_callback(lambda _: done.set())
        if not done.wait(SAM3_TIMEOUT_S):
            say(f"  SAM 3 did not answer in {SAM3_TIMEOUT_S:g} s")
            return None
        resp = fut.result()
        if not resp.success:
            say(f"  SAM 3: {resp.message}")
            return None
        objs = [o for o in resp.detections.objects if o.mask_id > 0]
        if not objs:
            say(f"  SAM 3 found no {' / '.join(self.sam3_prompts)} ({resp.message})")
            return None
        best = max(objs, key=lambda o: o.score)
        li = resp.detections.label_image
        labels = np.frombuffer(bytes(li.data), dtype="<u2").reshape(li.height, li.step // 2)[:, : li.width]
        say(f"  SAM 3: '{best.perception_class}' {best.score:.2f} (best of {len(objs)}; {resp.message})")
        return labels == best.mask_id

    def appliance_box(self):
        """The latest boxer box of `box_class` in the base frame: {center, R (3x3, columns = box axes), size, score},
        or None (says why)."""
        with self._lock:
            latest = self._box
        if latest is None:
            say(f"  no '{self.box_class}' box with a pose on {self.box_topic} yet (boxer + scene segmenter up?)")
            return None
        header, obj = latest
        age = (self.node.get_clock().now() - Time.from_msg(header.stamp)).nanoseconds / 1e9
        if age > BOX_MAX_AGE_S:
            say(f"  the last '{self.box_class}' box is {age:.1f} s old (max {BOX_MAX_AGE_S:g})")
            return None
        T = self._base_T(header.frame_id, header.stamp)
        if T is None:
            say(f"  no arm base <- {header.frame_id} transform for the box")
            return None
        q, c = obj.pose.orientation, obj.pose.position
        Rb = T[:3, :3] @ R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        return {
            "center": T[:3, :3] @ np.array([c.x, c.y, c.z]) + T[:3, 3],
            "R": Rb,
            "size": np.array([obj.size.x, obj.size.y, obj.size.z]),
            "score": float(obj.score),
        }

    def wait_for_frames(self, timeout_s=10.0):
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            with self._lock:
                if self._latest is not None:
                    return True
            time.sleep(0.05)
        return False

    def base_T_cam(self, frame_id, stamp):
        """4x4 base <- camera at `stamp` (else the latest available), or None."""
        return self._base_T(self.camera_frame or frame_id, stamp)

    def _base_T(self, target, stamp):
        """4x4 base <- `target` at `stamp` (else the latest available), or None."""
        for when in (Time.from_msg(stamp), Time()):
            try:
                t = self.tf_buffer.lookup_transform(
                    self.base_frame, target, when, timeout=Duration(seconds=TF_TIMEOUT_S)
                ).transform
            except Exception:  # noqa: BLE001 -- tf2's exceptions; try the latest, then give up
                continue
            m = np.eye(4)
            m[:3, :3] = R.from_quat([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]).as_matrix()
            m[:3, 3] = [t.translation.x, t.translation.y, t.translation.z]
            return m
        return None

    def grab(self, timeout_s=10.0):
        """The latest frame: {rgb (bgr8), depth_mm (float32), K (3x3), base_T_cam (4x4 or None), time,
        frame_id, box (appliance_box(), or None without a box_topic)}. Raises RuntimeError if no frame has come in
        within `timeout_s`."""
        if not self.wait_for_frames(timeout_s):
            raise RuntimeError(f"no RGB-D frames on {', '.join(self.topics)} (depth aligned to colour?)")
        with self._lock:
            color, info, depth = self._latest
        rgb = self._bridge.imgmsg_to_cv2(color, "bgr8")
        depth_mm = np.asarray(self._bridge.imgmsg_to_cv2(depth, "passthrough"), dtype=np.float32)
        return {
            "rgb": rgb,
            "depth_mm": depth_mm,
            "K": np.asarray(info.k, dtype=float).reshape(3, 3),
            "base_T_cam": self.base_T_cam(color.header.frame_id, color.header.stamp),
            "time": time.time(),
            "frame_id": self.camera_frame or color.header.frame_id,
            "box": self.appliance_box() if self.box_topic else self.fixed_box,
            "box2d": self.appliance_box_2d() if self.box2d_topic else None,
        }


# ---------------------------------------------------------------------------
# Handle detector
# ---------------------------------------------------------------------------
DEPTH_RANGE_M = (0.05, 2.0)
PLANE_DIST_M = 0.02
HANDLE_PROTRUSION_MAX_M = 0.07  # how far in front of the door plane a handle can sit
DBSCAN_EPS_M, DBSCAN_MIN_SAMPLES = 0.02, 50
HANDLE_MIN_CLUSTER_FRACTION = 0.25  # clusters below this fraction of the biggest are noise
HANDLE_ELONGATION_CAP = 10.0  # so a very thin sliver can't dominate the score
HANDLE_MIN_LENGTH_M = 0.06  # and at least this tall (5th-95th pct up it): a 2.3 cm chip of the bar set the grasp
# height on 10-30; a fragment is a miss (retried), not the handle
HANDLE_MIN_VERTICAL = 0.8  # the bar's long axis within ~37 deg of vertical (|cos| to up); checked FIRST (10-30: the
# size filter used to run first and could drop the bar for the door's bottom lip / the table edge)
MAX_DOOR_NORMAL_VERTICAL = 0.5  # door is vertical, so its normal must be ~horizontal
# SAM 3 handle (handle_source "sam3")
HANDLE_MASK_MIN_PIXELS = 50
HANDLE_PROTRUSION_DEFAULT_M = 0.03  # how far the bar stands out when its own depth doesn't say (too thin to see)
# fixed handle orientation the detector has always returned (the grasp yaws it to the face)
HANDLE_QUAT = np.array([-0.5, 0.5, 0.5, -0.5])


class HandleDetector:
    """Everything but which part of the image the door is in: a capability subclasses this with `door_region`."""

    def door_region(self, rgb, box, debug):
        """The door's region of the image (bool, image-sized), or None (says why) -- see `detect` for `box`, `debug`."""
        raise NotImplementedError

    def crop_points(self, pts, box, base_R, base_t):
        """The camera-frame points `pts` to fit, narrowed if the capability wants (None: too few, says why). `base_R`,
        `base_t`: arm base <- camera."""
        return pts

    def _handle_from_depth(self, off_plane, normal, offset, up_cam, z0, debug=None):
        """The handle (camera frame) from depth alone: what sticks out of the door plane, the vertical bar among it
        (handle_source \"depth\"). None if there is none (says why)."""
        # protruding points -> clusters -> the vertical, elongated one
        signed = off_plane @ normal + offset
        handle_pts = off_plane[(signed > 0.0) & (signed < HANDLE_PROTRUSION_MAX_M)]
        if debug is not None:
            debug["protruding"] = handle_pts
        if len(handle_pts) == 0:
            say("  nothing protruding in front of the door plane")
            return None
        # open3d's DBSCAN (was sklearn's; same clusters, ~3x faster, one dependency fewer -- 10-07)
        labels = np.asarray(
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(handle_pts)).cluster_dbscan(
                eps=DBSCAN_EPS_M, min_points=DBSCAN_MIN_SAMPLES, print_progress=False
            )
        )
        if not np.any(labels >= 0):
            say("  DBSCAN found no clusters")
            return None
        # the vertical bar first: only clusters whose long axis runs up and down count; the size filter and the
        # score then pick among those (a bottom lip or table edge can out-number the bar's points, but isn't vertical)
        vertical = []  # (label, points, score)
        for label, count in zip(*np.unique(labels[labels >= 0], return_counts=True), strict=True):
            cluster = handle_pts[labels == label]
            _, s, vt = np.linalg.svd(cluster - cluster.mean(axis=0), full_matrices=False)
            v = abs(float(vt[0] @ up_cam))
            up = cluster @ up_cam
            if v >= HANDLE_MIN_VERTICAL and np.percentile(up, 95) - np.percentile(up, 5) >= HANDLE_MIN_LENGTH_M:
                vertical.append((label, count, v * min(float(s[0] / max(s[1], 1e-9)), HANDLE_ELONGATION_CAP)))
        if not vertical:
            say(f"  no vertical bar >= {HANDLE_MIN_LENGTH_M * 100:.0f} cm tall among the protruding clusters")
            return None
        biggest = max(c for _, c, _ in vertical)
        best_label = max((t for t in vertical if t[1] >= HANDLE_MIN_CLUSTER_FRACTION * biggest), key=lambda t: t[2])[0]
        cluster = handle_pts[labels == best_label]
        if debug is not None:
            debug["bar"] = cluster
        handle_cam = np.median(cluster, axis=0)
        # 10-07: no longer "4 cm in from the cluster's max camera y" -- on the original upside-down camera that was
        # the TOP of the bar, on this upright one it is the BOTTOM end where the bar curves into the door.
        # The middle of the bar is right on either.
        # 10-30: its height is the middle of the bar's EXTENT (ends at the 5th / 95th percentile up the bar), not the
        # median point: the median followed wherever the depth points bunched on the bar (0.33 / 0.39 / 0.35 for one
        # microwave on one table)
        up_proj = cluster @ up_cam
        bar_lo, bar_hi = float(np.percentile(up_proj, 5)), float(np.percentile(up_proj, 95))
        handle_cam = handle_cam + up_cam * ((bar_lo + bar_hi) / 2 - float(handle_cam @ up_cam))
        say(
            f"  handle bar: {len(cluster)} pts, z {bar_lo + z0:.3f} to {bar_hi + z0:.3f} "
            f"({(bar_hi - bar_lo) * 100:.1f} cm) -> middle {(bar_lo + bar_hi) / 2 + z0:.3f}"
        )
        return handle_cam

    def _handle_from_mask(self, mask, depth, K, normal, offset, up_cam, z0):
        """The handle (camera frame) from SAM 3's mask of it (handle_source "sam3"): the bar stands out from the door
        by what the mask's own depth says (else HANDLE_PROTRUSION_DEFAULT_M); the mask's pixels on that plane give
        where it is, its height the middle of their vertical extent. The 2D mask doesn't flicker the way a thin bar's
        depth does. None if the mask is too small (says why)."""
        vs, us = np.nonzero(mask)
        if len(us) < HANDLE_MASK_MIN_PIXELS:
            say(f"  SAM 3 handle mask has {len(us)} pixels (min {HANDLE_MASK_MIN_PIXELS})")
            return None
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        rays = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones(len(us))], axis=1)
        d = depth[vs, us] / 1000.0
        ok = ~np.isnan(d) & (d >= DEPTH_RANGE_M[0]) & (d <= DEPTH_RANGE_M[1])
        signed = (rays[ok] * d[ok][:, None]) @ normal + offset
        signed = signed[(signed > 0.005) & (signed < HANDLE_PROTRUSION_MAX_M)]
        measured = len(signed) >= 20
        out = float(np.median(signed)) if measured else HANDLE_PROTRUSION_DEFAULT_M
        # each mask ray meets the bar's own plane -- the door plane moved `out` toward the camera -- not the door
        # behind it (that put an off-centre bar ~2 cm too far out sideways)
        on_bar = rays * ((out - offset) / (rays @ normal))[:, None]
        up_proj = on_bar @ up_cam
        lo, hi = float(np.percentile(up_proj, 5)), float(np.percentile(up_proj, 95))
        handle = np.median(on_bar, axis=0)
        handle += up_cam * ((lo + hi) / 2 - float(handle @ up_cam))
        say(
            f"  SAM 3 handle: {len(us)} px, z {lo + z0:.3f} to {hi + z0:.3f} -> middle {(lo + hi) / 2 + z0:.3f}; "
            f"{out * 100:.1f} cm out from the door ({f'measured, {len(signed)} pts' if measured else 'default'})"
        )
        return handle

    def _door_plane(self, rgb, K, depth, base_T_cam, box, debug):
        """The door plane in the camera frame: (plane points, the other points, normal out of the door toward the
        camera, offset, base +z in the camera frame, base <- camera rotation, translation), or None (says why)."""
        if base_T_cam is None:
            say("no arm base <- camera transform")
            return None
        up_cam = base_T_cam[2, :3]  # base +z in the camera frame
        Rb, tb = base_T_cam[:3, :3], base_T_cam[:3, 3]

        # pixels -> camera-frame points (row-major, like the original's np.where on the mask)
        mask = self.door_region(rgb, box, debug)
        if mask is None:
            return None
        vs, us = np.where(mask)
        d = depth[vs, us] / 1000.0
        ok = ~np.isnan(d) & (d >= DEPTH_RANGE_M[0]) & (d <= DEPTH_RANGE_M[1])
        us, vs, d = us[ok], vs[ok], d[ok]
        if len(d) == 0:
            say("  no valid depth in the box")
            return None
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        pts = np.stack([d / fx * (us - cx), d / fy * (vs - cy), d], axis=1)
        pts = self.crop_points(pts, box, Rb, tb)
        if pts is None:
            return None

        # door plane
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        plane_model, inliers = pcd.segment_plane(distance_threshold=PLANE_DIST_M, ransac_n=3, num_iterations=500)
        plane_pts = pts[inliers]
        if debug is not None:
            debug["door"] = plane_pts
        off_plane = pts[np.setdiff1d(np.arange(len(pts)), inliers)]
        normal, offset = np.asarray(plane_model[:3], float), float(plane_model[3])
        nn = np.linalg.norm(normal)
        normal, offset = normal / nn, offset / nn
        if np.dot(normal, plane_pts.mean(axis=0)) > 0:  # point toward the camera: "in front" is +
            normal, offset = -normal, -offset
        if abs(float(np.dot(normal, up_cam))) > MAX_DOOR_NORMAL_VERTICAL:
            say("  plane is not a door (normal not horizontal -- table/floor?)")
            return None
        return plane_pts, off_plane, normal, offset, up_cam, Rb, tb

    def door_face(self, rgb, K, depth, base_T_cam, box=None, debug=None):
        """The door face alone -- no handle -- from any RGB-D camera (e.g. a scene camera too far out to see the
        handle): dict(normal, door_z, door_mid) in the arm's base frame as `detect` returns them, and the face's width
        (m), or None."""
        plane = self._door_plane(rgb, K, depth, base_T_cam, box, debug)
        if plane is None:
            return None
        plane_pts, _, normal, _, up_cam, Rb, tb = plane
        axis = np.cross(normal, up_cam)
        axis /= np.linalg.norm(axis)
        proj = plane_pts @ axis
        lo, hi = float(np.percentile(proj, 1)), float(np.percentile(proj, 99))
        n_base = Rb @ normal
        n_base[2] = 0.0
        plane_base = plane_pts @ Rb.T + tb
        return {
            "normal": n_base / np.linalg.norm(n_base),
            "door_z": (float(np.percentile(plane_base[:, 2], 1)), float(np.percentile(plane_base[:, 2], 99))),
            "door_mid": plane_base[int(np.argmin(np.abs(proj - 0.5 * (lo + hi))))].copy(),
            "width": hi - lo,
        }

    def detect(self, rgb, K, depth, base_T_cam, box=None, handle_mask=None, debug=None):
        """`K` 3x3, `base_T_cam` 4x4 (arm base <- camera, or None). `box`: boxer's box in the base frame
        (WristCamera.appliance_box), or None -- `door_region` and `crop_points` decide what it means.
        `handle_mask`: SAM 3's mask of the handle (bool, image-sized) -- the handle from it instead of
        from what sticks out (handle_source). `debug`: a dict filled with the intermediate camera-frame points (door,
        protruding, bar) and the 2D box, for detection_debug. Returns dict(handle, quat, normal, hinge, door_z,
        door_mid) in the arm's base frame, or None.
        normal: horizontal door-face normal, out of the door toward the camera.
        hinge: door-face point on the edge farthest from the handle, nearest the handle's height
          (the swing pivots about the vertical line through it).
        door_z: (low, high) of the door face (1st/99th percentile height).
        door_mid: door-face point at the middle of its horizontal span."""
        plane = self._door_plane(rgb, K, depth, base_T_cam, box, debug)
        if plane is None:
            return None
        plane_pts, off_plane, normal, offset, up_cam, Rb, tb = plane

        z0 = float(base_T_cam[2, 3])  # base z = camera-frame point @ up_cam + the camera's height
        if handle_mask is not None:
            handle_cam = self._handle_from_mask(handle_mask, depth, K, normal, offset, up_cam, z0)
        else:
            handle_cam = self._handle_from_depth(off_plane, normal, offset, up_cam, z0, debug)
        if handle_cam is None:
            return None

        # door edges along the plane's own horizontal axis (percentiles: RANSAC's edge points drift)
        axis = np.cross(normal, up_cam)
        axis /= np.linalg.norm(axis)
        proj = plane_pts @ axis
        h_proj = float(handle_cam @ axis)
        lo, hi = float(np.percentile(proj, 1)), float(np.percentile(proj, 99))
        edge = lo if abs(h_proj - lo) > abs(h_proj - hi) else hi
        strip = plane_pts[np.abs(proj - edge) < 0.02]
        hinge_cam = strip[int(np.argmin(np.linalg.norm(strip - handle_cam, axis=1)))]
        if abs(float(hinge_cam @ axis) - h_proj) < 0.5 * max(abs(h_proj - lo), abs(h_proj - hi)):
            say("  far door edge is on the handle's side -- plane includes background")
            return None

        # to the arm's base frame
        n_base = Rb @ normal
        n_base[2] = 0.0
        plane_base = plane_pts @ Rb.T + tb
        return {
            "handle": Rb @ handle_cam + tb,
            "quat": HANDLE_QUAT,
            "normal": n_base / np.linalg.norm(n_base),
            "hinge": Rb @ hinge_cam + tb,
            "door_z": (float(np.percentile(plane_base[:, 2], 1)), float(np.percentile(plane_base[:, 2], 99))),
            "door_mid": plane_base[int(np.argmin(np.abs(proj - 0.5 * (lo + hi))))].copy(),
        }
