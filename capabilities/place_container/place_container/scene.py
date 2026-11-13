"""The scene camera's latest frame for place_here's tap-to-place: depth (aligned to colour), camera_info, the scene
segmenter's detections and tf, all for the running action only (create_action_subscription /
create_action_tf_buffer: the engine destroys them when the action ends). Read-only: nothing here moves anything.

`grab()` returns a spots.Scene in the arm's frame. The conversions (`depth_metres`, `label_array`, `matrix`) are
plain functions, tested without ROS.
"""

import threading
import time

import numpy as np

from . import spots

DETECTIONS_MAX_AGE_S = 1.0  # detections this far from the depth frame's stamp are not used with it
DEPTH_SCALE = {"16UC1": 0.001, "mono16": 0.001, "32FC1": 1.0}  # encoding -> metres per unit


def depth_metres(msg):
    """A sensor_msgs/Image of depth -> HxW float32 metres (0 = none)."""
    scale = DEPTH_SCALE.get(msg.encoding)
    if scale is None:
        raise ValueError(f"depth encoding {msg.encoding!r} (expected one of {sorted(DEPTH_SCALE)})")
    dtype = np.dtype(np.uint16 if scale != 1.0 else np.float32).newbyteorder(">" if msg.is_bigendian else "<")
    row = msg.step // dtype.itemsize
    depth = np.frombuffer(bytes(msg.data), dtype=dtype).reshape(msg.height, row)[:, : msg.width]
    out = depth.astype(np.float32) * scale
    out[~np.isfinite(out)] = 0.0
    return out


def label_array(msg):
    """ObjectDetections.label_image (mono16) -> HxW uint16, or None when it is empty."""
    if msg.width == 0 or msg.height == 0:
        return None
    dtype = np.dtype(np.uint16).newbyteorder(">" if msg.is_bigendian else "<")
    row = msg.step // 2
    return np.frombuffer(bytes(msg.data), dtype=dtype).reshape(msg.height, row)[:, : msg.width].astype(np.uint16)


def matrix(transform):
    """geometry_msgs/Transform -> 4x4."""
    q, t = transform.rotation, transform.translation
    x, y, z, w = np.array([q.x, q.y, q.z, q.w]) / np.linalg.norm([q.x, q.y, q.z, q.w])
    T = np.eye(4)
    T[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def stamp_s(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


class SceneCamera:
    """Subscribes for the running action: make it inside execute()."""

    def __init__(self, node, base_frame, depth_topic, info_topic, detections_topic=""):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image

        self.node, self.base_frame = node, base_frame
        self.topics = (depth_topic, info_topic, detections_topic)
        self._lock = threading.Lock()
        self._depth = self._info = self._detections = None
        self.tf = node.create_action_tf_buffer()
        qos = qos_profile_sensor_data
        node.create_action_subscription(Image, depth_topic, self._set("_depth"), qos)
        node.create_action_subscription(CameraInfo, info_topic, self._set("_info"), qos)
        self.detections = False
        if detections_topic:
            try:
                from rammp_perception_interfaces.msg import ObjectDetections
            except ImportError:  # the image without the perception interfaces: the depth plane only
                node.get_logger().warning("rammp_perception_interfaces missing: no table masks, the depth plane only")
            else:
                node.create_action_subscription(ObjectDetections, detections_topic, self._set("_detections"), qos)
                self.detections = True

    def _set(self, name):
        def store(msg):
            with self._lock:
                setattr(self, name, msg)

        return store

    def grab(self, timeout_s, checkpoint=lambda: None):
        """The newest frame as a spots.Scene, waiting up to `timeout_s` for depth, camera_info and tf. Raises
        RuntimeError (why) when something is missing."""
        from rclpy.time import Time

        deadline = time.monotonic() + timeout_s
        why = ""
        while True:
            checkpoint()
            with self._lock:
                depth, info, dets = self._depth, self._info, self._detections
            if depth is None or info is None:
                why = f"no depth or camera_info from the scene camera ({self.topics[0]}, {self.topics[1]})"
            else:
                frame = depth.header.frame_id.lstrip("/") or info.header.frame_id.lstrip("/")
                try:
                    tf = self.tf.lookup_transform(self.base_frame, frame, Time())
                except Exception as e:  # tf2's own exceptions: not there yet
                    why = f"no tf {self.base_frame} <- {frame} ({type(e).__name__})"
                else:
                    break
            if time.monotonic() > deadline:
                raise RuntimeError(why)
            time.sleep(0.05)
        d = depth_metres(depth)
        if (info.width, info.height) != (d.shape[1], d.shape[0]):
            raise RuntimeError(
                f"the depth image is {d.shape[1]}x{d.shape[0]} but camera_info {info.width}x{info.height}: "
                "depth must be aligned to colour"
            )
        labels, objects = None, ()
        if dets is not None and abs(stamp_s(dets.header) - stamp_s(depth.header)) <= DETECTIONS_MAX_AGE_S:
            labels = label_array(dets.label_image)
            objects = [(o.perception_class, int(o.mask_id)) for o in dets.objects if o.state == o.STATE_VISIBLE]
        return spots.Scene(
            d, np.asarray(info.k).reshape(3, 3), matrix(tf.transform), labels, objects, stamp_s(depth.header), "camera"
        )
