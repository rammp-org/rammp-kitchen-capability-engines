"""Videos of what the open sees, from a bag: the scene camera with boxer's microwave box, and the wrist camera with
that box (the detector's crop) and the handle/hinge detections the run made. Read-only. Run it in the microwave
container (it has the message types):

    ros2 bag record -o /microwave_data/bags/<name> /scene_camera/color/image_raw /scene_camera/color/camera_info \\
        /perception/object_detections_3d /wrist_camera/color/image_raw /wrist_camera/color/camera_info /tf /tf_static
    python3 -m microwave.detection_video /microwave_data/bags/<name>     # -> <bag>/scene.mp4, <bag>/wrist.mp4

The detections come from the run's saved detection frames (data_dir/detections/*_result.json, by their time).
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from door_common.common import say
from scipy.spatial.transform import Rotation as R

SCENE_TOPICS = ("/scene_camera/color/image_raw", "/scene_camera/color/camera_info")
WRIST_TOPICS = ("/wrist_camera/color/image_raw", "/wrist_camera/color/camera_info")
BOX_TOPIC = "/perception/object_detections_3d"
# box corners by sign pattern (x, y, z) -> the 12 edges between them
SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.sum(SIGNS[i] != SIGNS[j]) == 1]
GREEN, RED, BLUE, YELLOW = (0, 200, 0), (0, 0, 255), (255, 0, 0), (0, 220, 220)


def _matrix(rotation, translation):
    m = np.eye(4)
    m[:3, :3] = R.from_quat([rotation.x, rotation.y, rotation.z, rotation.w]).as_matrix()
    m[:3, 3] = [translation.x, translation.y, translation.z]
    return m


def _stamp_s(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def _messages(bag, topics=None):
    """(topic, message, bag time ns) from a rosbag2 directory, optionally only `topics`."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    storage = "mcap" if any(Path(bag).glob("*.mcap")) else "sqlite3"
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=storage), rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    if topics:
        reader.set_filter(rosbag2_py.StorageFilter(topics=[t for t in topics if t in types]))
    while reader.has_next():
        topic, data, t = reader.read_next()
        yield topic, deserialize_message(data, types[topic]), t


def _project(K, pts):
    """Camera-frame points -> pixel coordinates (int), None for points behind the camera."""
    out = []
    for p in pts:
        out.append(
            None if p[2] <= 0.01 else (int(K[0, 0] * p[0] / p[2] + K[0, 2]), int(K[1, 1] * p[1] / p[2] + K[1, 2]))
        )
    return out


def _draw_box(img, K, corners_cam, color):
    px = _project(K, corners_cam)
    for i, j in EDGES:
        if px[i] is not None and px[j] is not None:
            cv2.line(img, px[i], px[j], color, 2)


def _corners(obj):
    """The box's 8 corners in its message frame."""
    pose_m = _matrix(obj.pose.orientation, obj.pose.position)
    local = SIGNS * np.array([obj.size.x, obj.size.y, obj.size.z]) / 2
    return local @ pose_m[:3, :3].T + pose_m[:3, 3]


def _label(img, text, row, color=YELLOW):
    cv2.putText(img, text, (8, 20 + 20 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def _results(detections_dir, t0, t1):
    """The run's handle detections in [t0, t1] (wall time s): [(time, tag, result dict)], by time."""
    out = []
    for js in Path(detections_dir).glob("*_result.json"):
        parts = js.name.split("_")
        try:
            t = time.mktime(time.strptime(f"{parts[0]}_{parts[1]}", "%Y%m%d_%H%M%S"))
        except (ValueError, IndexError):
            continue
        if t0 - 1 <= t <= t1 + 1:
            out.append((t, "_".join(parts[2:4]), json.loads(js.read_text())))
    return sorted(out, key=lambda r: r[0])


class _Writer:
    def __init__(self, path):
        self.path, self.frames = path, []

    def add(self, stamp_s, img):
        self.frames.append((stamp_s, img))

    def close(self):
        if len(self.frames) < 2:
            say(f"  {self.path.name}: {len(self.frames)} frame(s) -- nothing written")
            return
        fps = max(1.0, (len(self.frames) - 1) / max(self.frames[-1][0] - self.frames[0][0], 1e-3))
        h, w = self.frames[0][1].shape[:2]
        vw = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for _, img in self.frames:
            vw.write(img)
        vw.release()
        say(f"  wrote {self.path} ({len(self.frames)} frames, {fps:.1f} fps)")


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("bag")
    a.add_argument("--base", default="arm_mount", help="the arm's base frame (arm.frame)")
    a.add_argument("--box-class", default="microwave")
    a.add_argument("--detections", default="/microwave_data/detections", help="the run's detection frames")
    a.add_argument("--every", type=int, default=1, help="keep every Nth image")
    args = a.parse_args()

    import tf2_ros
    from cv_bridge import CvBridge
    from rclpy.duration import Duration
    from rclpy.time import Time

    bridge, tf = CvBridge(), tf2_ros.Buffer(cache_time=Duration(seconds=3600))
    infos, boxes = {}, []  # boxes: (stamp s, header, DetectedObject)
    say(f"reading {args.bag} ...")
    for topic, m, _ in _messages(args.bag, ["/tf", "/tf_static", BOX_TOPIC, SCENE_TOPICS[1], WRIST_TOPICS[1]]):
        if topic in ("/tf", "/tf_static"):
            for t in m.transforms:
                (tf.set_transform_static if topic == "/tf_static" else tf.set_transform)(t, "bag")
        elif topic == BOX_TOPIC:
            objs = [o for o in m.objects if o.perception_class == args.box_class]
            if objs:
                boxes.append((_stamp_s(m.header.stamp), m.header, max(objs, key=lambda o: o.score)))
        else:
            infos[topic] = np.asarray(m.k, float).reshape(3, 3)
    say(f"  {len(boxes)} '{args.box_class}' box messages; camera info for {sorted(infos)}")

    def base_T(frame, stamp):
        try:
            t = tf.lookup_transform(args.base, frame, Time.from_msg(stamp)).transform
        except Exception:  # noqa: BLE001 -- tf2's exceptions
            return None
        return _matrix(t.rotation, t.translation)

    def box_at(s):
        """The latest box message at or before `s`, and how old it is."""
        before = [b for b in boxes if b[0] <= s + 0.05]
        return (before[-1], s - before[-1][0]) if before else (None, None)

    out = Path(args.bag)
    scene, wrist = _Writer(out / "scene.mp4"), _Writer(out / "wrist.mp4")
    span, results, counts = [None, None], None, {}
    for topic, m, _ in _messages(args.bag, [SCENE_TOPICS[0], WRIST_TOPICS[0]]):
        counts[topic] = counts.get(topic, 0) + 1
        if (counts[topic] - 1) % args.every:
            continue
        s = _stamp_s(m.header.stamp)
        span = [s if span[0] is None else span[0], s]
        img = bridge.imgmsg_to_cv2(m, "bgr8").copy()
        box, age = box_at(s)
        if topic == SCENE_TOPICS[0]:
            K = infos.get(SCENE_TOPICS[1])
            if box is None:
                _label(img, "no boxer box yet", 0, RED)
            else:
                _, header, obj = box
                x1, y1, x2, y2 = (int(v) for v in obj.box)
                cv2.rectangle(img, (x1, y1), (x2, y2), YELLOW, 1)
                if obj.has_pose and K is not None and header.frame_id == m.header.frame_id:
                    _draw_box(img, K, _corners(obj), GREEN)
                _label(img, f"boxer {obj.perception_class} {obj.score:.2f}  pose {obj.has_pose}  age {age:.2f}s", 0)
            scene.add(s, img)
            continue
        K, cam = infos.get(WRIST_TOPICS[1]), base_T(m.header.frame_id, m.header.stamp)
        if K is None or cam is None:
            _label(img, f"no {'camera info' if K is None else args.base + ' <- wrist camera transform'}", 0, RED)
            wrist.add(s, img)
            continue
        cam_T_base = np.linalg.inv(cam)
        if box is not None and box[2].has_pose:
            _, header, obj = box
            scene_T = base_T(header.frame_id, header.stamp)
            if scene_T is not None:
                c = _corners(obj) @ scene_T[:3, :3].T + scene_T[:3, 3]  # -> base
                _draw_box(img, K, c @ cam_T_base[:3, :3].T + cam_T_base[:3, 3], GREEN)
        if results is None:  # the bag's time span is known once images flow; load once
            results = _results(args.detections, span[0], span[0] + 3600)
        done = [r for r in results if r[0] <= s]
        if done:
            t, tag, res = done[-1]
            for key, color in (("handle", RED), ("hinge", BLUE)):
                if res.get(key) is not None:
                    p = cam_T_base[:3, :3] @ np.asarray(res[key], float) + cam_T_base[:3, 3]
                    (px,) = _project(K, [p])
                    if px is not None:
                        cv2.circle(img, px, 8, color, 2)
                        cv2.putText(img, f"{key} z={res[key][2]:.3f}", (px[0] + 10, px[1]), 0, 0.5, color, 1)
            _label(img, f"detection {tag} ({s - t:.1f}s ago)", 0)
        else:
            _label(img, "no handle detection yet", 0)
        wrist.add(s, img)
    scene.close()
    wrist.close()


if __name__ == "__main__":
    main()
