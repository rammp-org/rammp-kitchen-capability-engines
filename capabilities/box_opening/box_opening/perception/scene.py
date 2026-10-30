"""The scene camera: the Orbbec Gemini on the fixed mount, seeing the whole
workspace. Colour at 1280x720, depth at 848x480 with its own intrinsics and
its own optical frame; the driver publishes the depth<-colour extrinsic in
its TF tree, and `cloud_in_color_frame` uses it so every depth point can be
placed in the colour image and, through config/camera_scene.yaml, in
base_link.

Stores only what arrives; callers spin their own node.
"""

from pathlib import Path

import numpy as np

from box_opening.perception.d405 import quat_to_mat
from box_opening.perception.planes import cloud_from_depth
from box_opening.perception.scene_calib import SCENE_CALIB_FILE, load_scene_yaml, transform

SCENE_NS = "/scene_camera"
COLOR_TOPIC = SCENE_NS + "/color/image_raw"
COLOR_INFO = SCENE_NS + "/color/camera_info"
DEPTH_TOPIC = SCENE_NS + "/depth/image_raw"
DEPTH_INFO = SCENE_NS + "/depth/camera_info"
LINK_FRAME = "scene_camera_link"
COLOR_FRAME = "scene_camera_color_optical_frame"


def scene_calib_path():
    """The installed calibration, or None when the camera is uncalibrated."""
    from ament_index_python.packages import get_package_share_directory

    p = Path(get_package_share_directory("box_opening"), "config", SCENE_CALIB_FILE)
    return p if p.exists() else None


class SceneGrabber:
    def __init__(self, node, keep=1, need_depth=True, scene_cfg=None):
        from tf2_ros import Buffer, TransformListener

        self.node = node
        self.keep = int(keep)
        self.need_depth = bool(need_depth)
        self.colors = []  # newest last, up to `keep`
        self.depths = []
        self.color_stamp = None
        # the colour frames' own frame_id: what the scene segmenter's messages,
        # which copy their camera frame's header, must name too
        self.color_frame_id = None
        self.k = self.dist = None  # colour intrinsics
        self.kd = None  # depth intrinsics
        self.depth_frame = None
        # scene_cfg overrides topic/frame names for non-Gemini cameras (OAK-D).
        # None = Gemini defaults (bench backward compat).
        cfg = scene_cfg or {}
        ns = cfg.get("ns", SCENE_NS)
        self._color_topic = ns + cfg.get("color_topic_suffix", "/color/image_raw")
        self._color_info = ns + cfg.get("color_info_topic_suffix", "/color/camera_info")
        self._depth_topic = ns + cfg.get("depth_topic_suffix", "/depth/image_raw")
        self._depth_info = ns + cfg.get("depth_info_topic_suffix", "/depth/camera_info")
        self._link_frame = cfg.get("link_frame", LINK_FRAME)
        self._color_frame = cfg.get("color_frame", COLOR_FRAME)
        self.tf_buffer = Buffer()
        # spin_thread=False: the listener's own executor thread would spin
        # this node alongside the one that already spins it (the OWL node's
        # rclpy.spin, the mission's perception executor) — two executors
        # on one node, which rclpy does not guard. The OWL scene instance
        # went deaf after its first inferences three times on 2026-09-17
        # (ticks and params alive, image callbacks never delivered again,
        # re-subscribing did nothing). Whoever spins the node feeds TF.
        self.tf_listener = TransformListener(self.tf_buffer, node, spin_thread=False)
        self._subs = []
        self.resubscribes = 0
        self._subscribe()

    def _subscribe(self):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image

        node = self.node
        self._subs = [
            node.create_subscription(Image, self._color_topic, self._color_cb, qos_profile_sensor_data),
            node.create_subscription(CameraInfo, self._color_info, self._info_cb, qos_profile_sensor_data),
        ]
        if self.need_depth:
            self._subs += [
                node.create_subscription(Image, self._depth_topic, self._depth_cb, qos_profile_sensor_data),
                node.create_subscription(CameraInfo, self._depth_info, self._dinfo_cb, qos_profile_sensor_data),
            ]

    def resubscribe(self):
        """Tear the image subscriptions down and create them afresh. A
        long-lived OWL instance stopped receiving scene frames minutes
        after start while a fresh subscriber in another process got them
        at 30 Hz (2026-09-16, twice); a fresh subscription is the repair
        that needs no launch restart."""
        for sub in self._subs:
            try:
                self.node.destroy_subscription(sub)
            except Exception:
                pass
        self._subs = []
        self.resubscribes += 1
        self._subscribe()

    # -- callbacks ---------------------------------------------------------
    def _push(self, lst, item):
        lst.append(item)
        del lst[: -self.keep]

    def _color_cb(self, msg):
        if msg.encoding not in ("rgb8", "bgr8"):
            return
        a = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        self._push(self.colors, a[:, :, ::-1].copy() if msg.encoding == "rgb8" else a.copy())
        self.color_stamp = msg.header.stamp
        self.color_frame_id = msg.header.frame_id

    def _depth_cb(self, msg):
        if msg.encoding != "16UC1":
            return
        d = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        self._push(self.depths, d.astype(np.float32) / 1000.0)

    def _info_cb(self, msg):
        self.k = np.array(msg.k, dtype=float).reshape(3, 3)
        self.dist = np.array(msg.d, dtype=float)

    def _dinfo_cb(self, msg):
        self.kd = np.array(msg.k, dtype=float).reshape(3, 3)
        self.depth_frame = msg.header.frame_id

    # -- state -------------------------------------------------------------
    @property
    def color(self):
        return self.colors[-1] if self.colors else None

    @property
    def depth(self):
        return self.depths[-1] if self.depths else None

    def depth_median(self):
        """The median of the kept depth frames (median_depth)."""
        return median_depth(self.depths)

    def missing(self):
        out = []
        if not self.colors:
            out.append("colour")
        if self.k is None:
            out.append("colour camera_info")
        if self.need_depth and not self.depths:
            out.append("depth")
        if self.need_depth and self.kd is None:
            out.append("depth camera_info")
        return out

    def lookup(self, parent, child):
        """base_link-style 4x4 from TF, or None."""
        import rclpy.time as rt

        try:
            tf = self.tf_buffer.lookup_transform(parent, child, rt.Time())
        except Exception:
            return None
        t, r = tf.transform.translation, tf.transform.rotation
        return transform(quat_to_mat(r.x, r.y, r.z, r.w), [t.x, t.y, t.z])

    def depth_to_color(self):
        """colour optical <- depth optical, from the driver's own tree."""
        if self.depth_frame is None:
            return None
        return self.lookup(self._color_frame, self.depth_frame)

    def link_to_color(self):
        return self.lookup(self._link_frame, self._color_frame)

    @property
    def link_frame(self):
        return self._link_frame

    def cloud_in_color_frame(self, depth=None, stride=1):
        """(points (N,3) in the colour optical frame, their colour pixels
        (N,2)) for the valid depth pixels. None until depth, intrinsics and
        the depth<-colour extrinsic have all arrived."""
        d = self.depth if depth is None else depth
        T_c_d = self.depth_to_color()
        if d is None or self.kd is None or self.k is None or T_c_d is None:
            return None
        return color_cloud(d, self.kd, self.k, self.dist, T_c_d, stride=stride)


def median_depth(depths):
    """The median of depth frames (m), zeros ignored — a steadier surface
    than any single frame. None for no frames."""
    if not depths:
        return None
    stack = np.stack(depths)
    dt = stack.dtype if np.issubdtype(stack.dtype, np.floating) else np.dtype(np.float64)
    stack = stack.astype(dt, copy=False)
    valid = stack > 0
    n = valid.sum(axis=0)
    # holes sort to the end as +inf, so each pixel's n valid depths come
    # first and its median is the middle one (or two) of those — exactly
    # nanmedian's answer at a fifth of its cost (82 ms on 3 OAK frames)
    s = np.where(valid, stack, np.inf)
    s.sort(axis=0)
    lo = np.take_along_axis(s, np.maximum((n - 1) // 2, 0)[None], 0)[0]
    hi = np.take_along_axis(s, np.minimum(n // 2, len(depths) - 1)[None], 0)[0]
    med = (lo + hi) * dt.type(0.5)
    # pixels with no depth in any frame are simply holes
    return np.where(n > 0, med, dt.type(0)).astype(dt, copy=False)


def color_cloud(depth, kd, k, dist, T_color_depth, stride=1):
    """(points (N,3) in the colour optical frame, their colour pixels (N,2))
    for the valid pixels of a depth image with intrinsics `kd`, moved into
    the colour camera by `T_color_depth` and projected with its `k` and
    `dist`. The live grabber and an offline replay of recorded arrays both
    compute it here."""
    pts, _u, _v = cloud_from_depth(depth, kd, stride=stride)
    T = np.asarray(T_color_depth, float)
    pc = pts @ T[:3, :3].T + T[:3, 3]
    d = None if dist is None else np.asarray(dist, float).ravel()
    if d is not None and d.size and np.any(d != 0):
        import cv2

        uv, _ = cv2.projectPoints(
            pc.reshape(-1, 1, 3).astype(np.float64),
            np.zeros(3),
            np.zeros(3),
            np.asarray(k, float),
            d,
        )
        return pc, uv.reshape(-1, 2)
    # projectPoints with zero distortion is this pinhole, and costs 44 ms on the OAK
    K = np.asarray(k, float)
    uv = np.stack([K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]], axis=1)
    return pc, uv


def load_scene_calibration(path=None):
    """base_link <- scene_camera_link (4x4) from the calibration yaml — the
    installed one unless `path` names another; None when uncalibrated."""
    p = scene_calib_path() if path is None else Path(path)
    if p is None or not p.exists():
        return None
    _doc, T_base_link = load_scene_yaml(str(p))
    return T_base_link
