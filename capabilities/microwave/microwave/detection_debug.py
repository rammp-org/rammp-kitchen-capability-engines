"""What the handle detector saw, per saved detection frame: the colour image with every stage drawn on it (YOLO box,
door-plane points, what sticks out of the door, the cluster taken as the bar, the handle and hinge), the depth
image in colour, and that depth blended over the colour image. Re-runs the detector on the frames the open saved
(data_dir/detections: colour, depth, K, camera pose); read-only. Run it in the microwave container:

    python3 -m microwave.detection_debug                       # the last 4 frames -> <frame>_debug.jpg beside them
    python3 -m microwave.detection_debug --live                # one frame from the wrist camera NOW (arm anywhere)
    python3 -m microwave.detection_debug --last 10 --model /microwave_data/yolo26s.pt
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from .microwave_common import save_frame, say
from .perception import YOLO_MODEL, HandleDetector

DEPTH_RANGE_VIEW_M = (0.15, 1.2)  # the depth colour scale
OVERLAY_ALPHA = 0.5  # depth colour's weight in the overlay panel
COLOURS = {"door": (0, 200, 0), "protruding": (0, 220, 220), "bar": (0, 0, 255)}  # BGR


def _pixels(K, pts):
    pts = pts[pts[:, 2] > 0.01]
    return np.stack([K[0, 0] * pts[:, 0] / pts[:, 2] + K[0, 2], K[1, 1] * pts[:, 1] / pts[:, 2] + K[1, 2]], 1).astype(
        int
    )


def _dots(img, K, pts, colour, step=1):
    h, w = img.shape[:2]
    for u, v in _pixels(K, pts[::step]):
        if 0 <= u < w and 0 <= v < h:
            img[v, u] = colour


def render(stem, detector):
    """<stem>_debug.jpg from <stem>_rgb.jpg + <stem>.npz. Returns its path, or None."""
    rgb, z = cv2.imread(f"{stem}_rgb.jpg"), np.load(f"{stem}.npz")
    if rgb is None or "base_T_cam" not in z:
        return None
    depth, K, base_T_cam = z["depth_mm"].astype(np.float32), z["K"], z["base_T_cam"]
    depth[depth == 0] = np.nan
    debug = {}
    det = detector.detect(rgb, K, depth, base_T_cam, debug=debug)
    left = rgb.copy()
    for key in ("door", "protruding", "bar"):  # bar last: on top
        if key in debug:
            _dots(left, K, debug[key], COLOURS[key], step=1 if key != "door" else 3)
    lo, hi = DEPTH_RANGE_VIEW_M
    scaled = np.clip((np.nan_to_num(depth, nan=hi * 1000) / 1000 - lo) / (hi - lo), 0, 1)
    right = cv2.applyColorMap((255 - scaled * 255).astype(np.uint8), cv2.COLORMAP_JET)  # near = red, far = blue
    right[np.isnan(depth)] = 0  # no depth = black
    # depth blended over the colour image, where there is depth (no depth: the plain image)
    overlay = rgb.copy()
    valid = ~np.isnan(depth)
    overlay[valid] = cv2.addWeighted(rgb, 1 - OVERLAY_ALPHA, right, OVERLAY_ALPHA, 0)[valid]
    if "box2d" in debug:
        x1, y1, x2, y2 = debug["box2d"]
        for img in (left, right, overlay):
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
    if det is not None:
        cam_T_base = np.linalg.inv(base_T_cam)
        for key, colour in (("handle", (0, 0, 255)), ("hinge", (255, 0, 0))):
            p = cam_T_base[:3, :3] @ np.asarray(det[key], float) + cam_T_base[:3, 3]
            (u, v), *_ = _pixels(K, p[None])
            for img in (left, right, overlay):
                cv2.circle(img, (int(u), int(v)), 9, colour, 2)
                cv2.putText(img, f"{key} z={det[key][2]:.3f}", (int(u) + 11, int(v)), 0, 0.5, colour, 1)
    counts = ", ".join(f"{k} {len(debug[k])}" for k in ("door", "protruding", "bar") if k in debug)
    cv2.putText(
        left,
        f"{Path(stem).name}  {'DETECTED' if det is not None else 'no detection'}",
        (8, 18),
        0,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(left, counts or "no points", (8, 38), 0, 0.5, (255, 255, 255), 1)
    cv2.putText(right, f"depth {lo:g}-{hi:g} m (near red, far blue, none black)", (8, 18), 0, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, "depth over colour", (8, 18), 0, 0.5, (255, 255, 255), 1)
    out = f"{stem}_debug.jpg"
    cv2.imwrite(out, np.hstack([left, right, overlay]))
    return out


def live(args, detector):
    """Grab one frame from the wrist camera now (topics and TF on this machine), save it next to the open's frames
    and render it."""
    import threading

    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    from .perception import WristCamera

    rclpy.init()
    node = rclpy.create_node("microwave_detection_debug")
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    cam = WristCamera(node, args.color, args.info, args.depth, args.base)
    try:
        f = cam.grab()
        if f["base_T_cam"] is None:
            say(f"no {args.base} <- {f['frame_id']} transform -- is the arm up?")
            return
        stem = save_frame(f, "live", args.dir)
        say(f"  -> {render(str(stem), detector)}")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--dir", default="/microwave_data/detections")
    a.add_argument("--last", type=int, default=4, help="how many of the newest frames")
    a.add_argument("--model", default=YOLO_MODEL, help="YOLO weights (the node's yolo_model)")
    a.add_argument("--live", action="store_true", help="grab one frame from the wrist camera now instead")
    a.add_argument("--color", default="/wrist_camera/color/image_raw")
    a.add_argument("--info", default="/wrist_camera/color/camera_info")
    a.add_argument("--depth", default="/wrist_camera/depth/image_rect_raw", help="depth aligned to colour (OAK)")
    a.add_argument("--base", default="arm_mount", help="the arm's base frame (arm.frame)")
    args = a.parse_args()
    if args.live:
        live(args, HandleDetector(args.model))
        return
    stems = sorted(str(p)[: -len(".npz")] for p in Path(args.dir).glob("*.npz"))[-args.last :]
    if not stems:
        say(f"no saved frames in {args.dir}")
        return
    detector = HandleDetector(args.model)
    for stem in stems:
        say(f"{Path(stem).name}:")
        out = render(stem, detector)
        say(f"  -> {out}" if out else "  (no colour image or camera pose saved -- skipped)")


if __name__ == "__main__":
    main()
