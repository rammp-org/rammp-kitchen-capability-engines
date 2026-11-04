"""Where the container goes in the open microwave, from one scene-camera frame and its cavity mask.

Entry points:
    fit_interior(depth_m, K, world_T_cam, cavity_mask) -> dict; raises NotDetected with the reason
        one frame; pure numpy/opencv
    locate(engine, ...) -> dict
        the live scene camera inside a running action: SAM3's cavity mask on frame after frame until
        `frames` consecutive detected frames agree; their median, and the frame nearest it; raises
        NotLocated otherwise
    SceneCamera(engine, ...).request(prompts) / .collect(future, prompts, deadline)
        one live frame: SAM3's masks for prompts, the depth frame of its stamp, the camera pose at it
    cavity_mask_from_detect(response) -> HxW bool or None
        the cavity mask from a rammp_perception_interfaces/srv/Detect response

This is for one microwave, whose dimensions are known (the constants below). It is measured from the
frame face around its opening: the flat face, in one vertical plane, where the closed door's seal
rests. The only other assumption is that the oven stands upright.

Oven frame (horizontal axes in the gravity-aligned world xy plane):
    n : horizontal, into the oven (= insertion axis)
    w : z x n, horizontal, the oven's LEFT as seen from the front
    A : coordinate along w,  S : coordinate along n,  Z : world z

Steps
  1. The cavity mask (SAM3 "inside of microwave"): everything seen through the opening, every piece of
     it, holes filled.
  2. The opening, a rectangle of its known size standing upright, posed (where its bottom edge's middle
     is, its turn about vertical) to overlap the mask best (intersection over union, from several
     starting turns). Nothing picks edges or sides: whatever stands in front of part of the opening only
     takes that part out of the mask.
  3. Depth checks that one fit: the frame around the opening lies on its plane, and what is seen
     through it lies within the oven's depth behind it.
  4. The point: centred sideways, half-way from the frame plane to the back wall, on the floor
     (the turntable glass).
Any requirement that fails: not detected. Nothing is guessed.
"""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as _Rot

# ---------------------------------------------------------------------------------------------
# This microwave: Comfee retro CMO-MMP02AA (0.9 cu ft). Metres, tape-measured by hand (2026-11-02).
# ---------------------------------------------------------------------------------------------
OPENING = (0.3048, 0.1905)  # opening width, height in the frame plane [12 x 7.5 in]
RECESS = 0.0286  # control panel face -> frame plane around the opening
DEPTH = 0.3048  # frame plane -> back wall [12 in]; the side walls are flush with the opening's edges
FLOOR = 0.0159  # floor (turntable glass top) above the opening's bottom edge [5/8 in]

# the depth checks of the fit
RING = (3, 0.04)  # ring outside the mask: from this many px (the frame's stereo depth spreads ~1 cm into
# the opening; the ring needs only the frame itself) to this far (m) out
RING_MIN = 50  # with at least this many depth points on the frame around the opening ...
RING_TOL = 0.015  # ... their median must be within this of the fitted plane
IN_FRONT = 0.05  # depth points this far in front of the fitted plane are something in front of the oven (the
# arm reaching toward it), not its frame: the frame check leaves them out
BEHIND = (0.01, DEPTH + 0.03)  # what is seen through the opening lies this far behind the plane ...
BEHIND_EDGE = 0.015  # m: what is seen through the opening is taken this far in from the mask's edge
BEHIND_SHARE = 0.9  # ... in this share of the mask's depth points (eroded by RING[0]) ...
BEHIND_MIN = 50  # ... of which at least this many are needed


class NotDetected(Exception):
    """This frame does not show the interior; the message says which requirement failed."""


def _backproject(depth, K):
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    z = depth.astype(np.float32)
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], -1)


def _normals(Pw, valid, k=3, max_jump=0.02):
    """Organized-cloud normals by central differences over k px.
    Returns unit normals (sign arbitrary), ok mask."""
    n = np.zeros_like(Pw)
    ok = np.zeros(valid.shape, bool)
    dx = Pw[k:-k, 2 * k :] - Pw[k:-k, : -2 * k]
    dy = Pw[2 * k :, k:-k] - Pw[: -2 * k, k:-k]
    cr = np.cross(dx, dy)
    nn = np.linalg.norm(cr, axis=-1, keepdims=True)
    v = (
        valid[k:-k, 2 * k :]
        & valid[k:-k, : -2 * k]
        & valid[2 * k :, k:-k]
        & valid[: -2 * k, k:-k]
        & valid[k:-k, k:-k]
        & (np.linalg.norm(dx, axis=-1) < max_jump * 2 * k)
        & (np.linalg.norm(dy, axis=-1) < max_jump * 2 * k)
        & (nn[..., 0] > 1e-12)
    )
    n[k:-k, k:-k] = cr / np.maximum(nn, 1e-12)
    ok[k:-k, k:-k] = v
    return n, ok


# ---------------------------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------------------------


def _upright_rect(yaw, cx, cy, cz):
    """The opening in world for a pose: (corners {bl, br, tl, tr}, insertion axis n, the oven's left w);
    (cx, cy, cz) is the middle of its bottom edge, n points into the oven."""
    n = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    w = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    b = np.array([cx, cy, cz])
    up = np.array([0.0, 0.0, OPENING[1]])
    half = w * OPENING[0] / 2
    return dict(bl=b + half, br=b - half, tl=b + half + up, tr=b - half + up), n, w


AREA_STARTS_DEG = tuple(range(-60, 61, 15))  # the fit's starting turns, about facing the camera
AREA_MIN_OVERLAP = 0.7  # the best fit must overlap the cavity mask at least this much (intersection / union)
AREA_PAD_PX = 40  # the overlap is computed in the mask's box grown by this


def _filled(mask):
    """The mask, every piece of it, with its holes filled."""
    m = np.asarray(mask, bool).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(m)
    cv2.drawContours(out, cs, -1, 1, -1)
    return out > 0


def fit_interior(depth_m, K, world_T_cam, cavity_mask):
    """depth_m: (H,W) metres aligned to the color image, 0 = invalid. K: 3x3. world_T_cam: 4x4
    camera-optical -> world (z up). cavity_mask: HxW bool, SAM3 "inside of microwave" on the same image.

    The opening is a rectangle of known size (OPENING) standing upright. Its pose (the middle of its bottom
    edge, and its turn about vertical) is the one whose projection overlaps the cavity mask best
    (intersection over union, from each of AREA_STARTS_DEG); nothing else picks it. Depth then checks that
    one fit: the frame around the opening must lie on its plane, and what is seen through it must lie inside
    the oven's depth behind it.

    Returns, in world: point (where the container goes), insertion_axis, lateral_axis_left, opening_corners
    (bottom-right, bottom-left, top-left, top-right in the frame plane, as seen from the front), floor_z,
    pins, fit (overlap), edges ([]). Raises NotDetected with the reason."""
    from scipy.optimize import minimize

    if cavity_mask is None:
        raise NotDetected("no cavity mask")
    K = np.asarray(K, float).reshape(3, 3)
    T = np.asarray(world_T_cam, float)
    R, cam = T[:3, :3], T[:3, 3]
    cTw = np.linalg.inv(T)
    depth = np.asarray(depth_m, np.float32)
    H, W = depth.shape
    mask = _filled(cavity_mask)
    if mask.sum() < 400:
        raise NotDetected(f"cavity mask too small ({int(mask.sum())} px)")
    valid = np.isfinite(depth) & (depth > 0.15) & (depth < 3.0)
    Pw = _backproject(np.where(valid, depth, 0), K) @ R.T.astype(np.float32) + cam.astype(np.float32)
    Nw, nok = _normals(Pw, valid)
    out_px = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    # what is seen through the opening: the mask without the band where the frame's stereo depth spreads into
    # it (~1 cm at the frame's distance)
    d_in = float(np.median(depth[mask & valid])) if (mask & valid).any() else 1.0
    in_px = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    seen = Pw[(in_px > max(RING[0], BEHIND_EDGE * K[0, 0] / d_in)) & valid].astype(float)
    if len(seen) < BEHIND_MIN:
        raise NotDetected(f"{len(seen)} depth points seen through the opening (need {BEHIND_MIN})")

    # 1. the pose whose projection overlaps the mask best
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    X0, Y0 = max(x0 - AREA_PAD_PX, 0), max(y0 - AREA_PAD_PX, 0)
    X1, Y1 = min(x1 + AREA_PAD_PX + 1, W), min(y1 + AREA_PAD_PX + 1, H)
    m = mask[Y0:Y1, X0:X1].astype(np.float32)
    m_sum = float(m.sum())
    shift = 4  # sub-pixel bits for the polygon's corners

    def overlap(p):
        C = _upright_rect(*p)[0]
        X = np.array([C["bl"], C["br"], C["tr"], C["tl"]])
        pc = X @ cTw[:3, :3].T + cTw[:3, 3]
        if np.any(pc[:, 2] < 0.05):
            return 0.0
        uv = (pc @ K.T)[:, :2] / pc[:, 2:]
        poly = np.zeros_like(m)
        cv2.fillPoly(poly, [np.round((uv - [X0, Y0]) * (1 << shift)).astype(np.int32)], 1.0, cv2.LINE_AA, shift)
        inter = float(np.minimum(poly, m).sum())
        return inter / max(float(poly.sum()) + m_sum - inter, 1e-9)

    # starts: the opening's middle on the ray through the mask's middle, as far as its height in the image says
    ray = R @ (np.linalg.inv(K) @ np.r_[(x0 + x1) / 2, (y0 + y1) / 2, 1.0])
    mid = cam + ray * (K[1, 1] * OPENING[1] / max(y1 - y0, 1))
    facing = np.arctan2(mid[1] - cam[1], mid[0] - cam[0])
    best = None
    for s in AREA_STARTS_DEG:
        p0 = np.r_[facing + np.radians(s), mid[0], mid[1], mid[2] - OPENING[1] / 2]
        simplex = np.array([p0] + [p0 + e for e in np.diag([np.radians(8.0), 0.02, 0.02, 0.02])])
        r = minimize(
            lambda q: 1.0 - overlap(q),
            p0,
            method="Nelder-Mead",
            options=dict(initial_simplex=simplex, xatol=1e-4, fatol=1e-5, maxfev=600),
        )
        if best is None or r.fun < best[0]:
            best = (float(r.fun), r.x)
    p, best_overlap = best[1], 1.0 - best[0]
    if best_overlap < AREA_MIN_OVERLAP:
        raise NotDetected(
            f"the known-size opening overlaps the cavity mask {best_overlap:.2f} at best (need {AREA_MIN_OVERLAP})"
        )

    # 2. depth checks that fit
    C, n, w = _upright_rect(*p)
    b = np.asarray(p[1:])
    if n @ (b - cam) <= 0:
        raise NotDetected("the best fit has the camera behind the opening")
    behind = (seen - b) @ n
    share = float(np.mean((behind > BEHIND[0]) & (behind < BEHIND[1])))
    if share < BEHIND_SHARE:
        raise NotDetected(
            f"the best fit ({best_overlap:.2f} overlap): only {share * 100:.0f}% of what is seen through it lies "
            "inside the oven's depth behind it"
        )
    ring = valid & nok & (out_px >= RING[0]) & (out_px <= RING[0] + 15)
    Pr = Pw[ring].astype(float)
    a_r, z_r = (Pr - b) @ w, Pr[:, 2] - b[2]
    on = (np.abs(a_r) < OPENING[0] / 2 + 0.04) & (z_r > -0.04) & (z_r < OPENING[1] + 0.04)
    on &= (Pr - b) @ n > -IN_FRONT
    frame_off = None
    if on.sum() >= RING_MIN:
        frame_off = float(np.median((Pr[on] - b) @ n))
        if abs(frame_off) > RING_TOL:
            raise NotDetected(
                f"the best fit ({best_overlap:.2f} overlap): the frame's depth is {frame_off * 100:+.1f} cm "
                "off its plane"
            )

    # 3. the point
    return dict(
        point=(b + n * DEPTH / 2 + np.array([0.0, 0.0, FLOOR])).tolist(),
        insertion_axis=n.tolist(),
        lateral_axis_left=w.tolist(),
        opening_corners=[C[k].tolist() for k in ("br", "bl", "tl", "tr")],
        floor_z=float(b[2] + FLOOR),
        pins=dict(
            fit=[f"the {OPENING[0] * 100:.2f} x {OPENING[1] * 100:.2f} cm opening, best overlap with the cavity mask"]
        ),
        fit=dict(
            overlap=best_overlap,
            behind_share=share,
            frame_off_cm=None if frame_off is None else round(frame_off * 100, 2),
            frame_depth_points=int(on.sum()),
        ),
        edges=[],
    )


# ---------------------------------------------------------------------------------------------
# Live: SAM3's cavity mask on the scene camera, several detected frames that agree (inside a running action)
# ---------------------------------------------------------------------------------------------

DEPTH_SCALE = {"16UC1": 0.001, "mono16": 0.001, "32FC1": 1.0}  # depth encoding -> metres per unit
SAM3_SERVICE = "/perception/sam3/detect"  # rammp_perception_interfaces/srv/Detect
SAM3_PROMPT = "inside of microwave"
SAM3_IMAGE_TOPIC = "/scene_camera/color/image_raw"
SAM3_THRESHOLD = 0.5  # SAM3 keeps objects scoring at least this (the cavity scored 0.82-0.95 in the recordings)
SAM3_WAIT_S = 2.0  # the SAM3 service must be up within this
PAIR_MAX_S = 0.01  # the depth frame paired with SAM3's color frame: stamps this close (the camera stamps both alike)
DEPTH_KEEP = 60  # depth frames kept for pairing (~2 s at 30 Hz; a SAM3 call takes a few hundred ms)


class NotLocated(Exception):
    """No agreed interior; the message says what was seen."""


def _stamp(msg):
    return msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec


def cavity_mask_from_detect(response, prompt=None):
    """HxW bool: the highest-scoring object's mask in a rammp_perception_interfaces/srv/Detect response
    (with `prompt`: among the objects SAM3 found for that prompt, its perception_class); None when SAM3
    failed or found nothing. Raises ValueError when objects come without a usable label image."""
    objects = [o for o in response.detections.objects if prompt is None or o.perception_class == prompt]
    if not response.success or not objects:
        return None
    img = response.detections.label_image
    if img.encoding not in ("mono16", "16UC1") or not img.height:
        raise ValueError(f"SAM3 label image: encoding {img.encoding!r}, {img.width}x{img.height} (expected mono16)")
    labels = np.frombuffer(bytes(img.data), np.dtype(">u2" if img.is_bigendian else "<u2"))
    labels = labels.reshape(img.height, img.step // 2)[:, : img.width]
    top = max(objects, key=lambda o: o.score)
    return labels == top.mask_id


class SceneCamera:
    """The live scene camera inside a running action of `engine`: SAM3 (sam3_service) masks on its color
    frames, each paired with the depth frame of the same stamp and with the camera pose in `frame` from TF
    at that stamp. topics: (depth, camera_info, ...). Its subscriptions and client end with the action.
    Raises NotLocated when the SAM3 service is not up."""

    def __init__(
        self,
        engine,
        frame,
        topics,
        sam3_service=SAM3_SERVICE,
        sam3_threshold=SAM3_THRESHOLD,
        sam3_image_topic=SAM3_IMAGE_TOPIC,
    ):
        import threading
        from collections import deque

        from cv_bridge import CvBridge
        from rammp_perception_interfaces.srv import Detect
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image

        self.frame, self._Detect = frame, Detect
        self._service, self._threshold, self._image_topic = sam3_service, sam3_threshold, sam3_image_topic
        self._depths, self._latest, self._lock = deque(maxlen=DEPTH_KEEP), {}, threading.Lock()
        self._bridge = CvBridge()

        def keep_depth(m):
            with self._lock:
                self._depths.append(m)

        self._tf = engine.create_action_tf_buffer()
        engine.create_action_subscription(Image, topics[0], keep_depth, qos_profile_sensor_data)
        engine.create_action_subscription(
            CameraInfo, topics[1], lambda m: self._latest.update(info=m), qos_profile_sensor_data
        )
        # The engine has no public action-scoped service client; its action scope makes one (as for the
        # subscriptions above) and destroys it safely when the action ends.
        self._sam3 = engine._action_scope().create_client(Detect, sam3_service)
        if not self._sam3.wait_for_service(timeout_sec=SAM3_WAIT_S):
            raise NotLocated(f"SAM3 service {sam3_service} not available within {SAM3_WAIT_S:.0f} s")

    def request(self, prompts):
        """Ask SAM3 for `prompts` (a list) on the next color frame; the pending call, for collect()."""
        return self._sam3.call_async(
            self._Detect.Request(prompts=list(prompts), image_topic=self._image_topic, score_threshold=self._threshold)
        )

    def cancel(self, future):
        if not future.done():
            self._sam3.remove_pending_request(future)

    def collect(self, future, prompts, deadline):
        """The frame of a request() for `prompts`. (shot, None) with shot = dict(depth_m (HxW m), K (3x3),
        world_T_cam (4x4, optical -> `frame`), mask (HxW bool, SAM3's best object for prompts[0]), masks
        ({prompt: mask} for every prompt), stamp); (None, why) when SAM3 found nothing for one of the prompts
        or there is no depth frame, camera_info or TF at its frame; (None, None) when time.monotonic() passed
        `deadline` first. Raises NotLocated on a depth encoding or image size it cannot use."""
        import time

        from rclpy.duration import Duration
        from rclpy.time import Time

        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            self.cancel(future)
            return None, None
        response = future.result()
        # SAM3's label image holds one object per pixel: prompts whose objects overlap (the cavity lies inside
        # the oven) can't share a call
        masks = {q: cavity_mask_from_detect(response, q if len(prompts) > 1 else None) for q in prompts}
        missing = [q for q, m in masks.items() if m is None]
        if missing:
            return None, f"SAM3 found no {missing[0]!r}" + (f" ({response.message})" if response.message else "")
        mask = masks[prompts[0]]
        stamp = _stamp(response.detections)
        with self._lock:
            depth = min(self._depths, key=lambda m: abs(_stamp(m) - stamp), default=None)
        info = self._latest.get("info")
        if depth is None or info is None or abs(_stamp(depth) - stamp) > PAIR_MAX_S:
            return None, "no depth frame or camera_info at SAM3's frame"
        if depth.encoding not in DEPTH_SCALE:
            raise NotLocated(f"scene depth encoding {depth.encoding!r} (expected one of {sorted(DEPTH_SCALE)})")
        try:
            t = self._tf.lookup_transform(
                self.frame, depth.header.frame_id, Time.from_msg(depth.header.stamp), timeout=Duration(seconds=0.5)
            ).transform
        except Exception as e:  # noqa: BLE001 -- tf2's lookup errors: this frame is not used
            return None, f"no TF {self.frame} <- {depth.header.frame_id} at the frame's stamp ({type(e).__name__})"
        T = np.eye(4)
        T[:3, :3] = _Rot.from_quat([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]).as_matrix()
        T[:3, 3] = [t.translation.x, t.translation.y, t.translation.z]
        depth_m = self._bridge.imgmsg_to_cv2(depth).astype(np.float32) * DEPTH_SCALE[depth.encoding]
        if depth_m.shape != mask.shape:
            raise NotLocated(f"SAM3's mask is {mask.shape[::-1]} px, the depth image {depth_m.shape[::-1]}")
        K = np.array(info.k, float).reshape(3, 3)
        return dict(depth_m=depth_m, K=K, world_T_cam=T, mask=mask, masks=masks, stamp=_stamp(depth)), None


def locate(
    engine,
    frame,
    topics,
    frames,
    timeout_s,
    agree_m,
    agree_deg,
    log,
    sam3_service=SAM3_SERVICE,
    sam3_prompt=SAM3_PROMPT,
    sam3_threshold=SAM3_THRESHOLD,
    sam3_image_topic=SAM3_IMAGE_TOPIC,
    camera=None,
    extra_prompt=None,
):
    """The interior in `frame` (gravity-aligned, z up: arm_mount) from the scene camera, inside a
    running action of `engine`. topics: (depth, camera_info, object_detections_3d); the 3D detections
    are not used. Detects continuously: each round asks SAM3 (sam3_service) for sam3_prompt on the next
    color frame, pairs its mask with the depth frame of the same stamp, and fits that with the camera pose
    from TF at the stamp (fit_interior); the next frame's call is sent before the current one is fitted. It
    stops at the first `frames` consecutive detected frames that agree: every point within agree_m of
    their median, every insertion axis within agree_deg of their median heading. Frames that are not
    detected are skipped. None such within timeout_s: NotLocated. With extra_prompt (the oven's body,
    "microwave", for the obstacles): one frame of it first (SAM3 can't return it in the same call: the
    cavity lies inside it), returned as "body". camera: a SceneCamera to use (None: one is made).

    Returns {point, insertion_axis, floor_z (medians of the frames used), frames (used), detected,
    spread_m, spread_deg, fit (the used frame's fit_interior result nearest the median point), scene (that
    frame: depth_m, K, world_T_cam, mask, stamp), body}; raises NotLocated."""
    import time

    def medians(items):
        pts = np.array([r["point"] for r, _ in items])
        yaws = np.array([np.arctan2(r["insertion_axis"][1], r["insertion_axis"][0]) for r, _ in items])
        yaw = yaws[0] + float(np.median((yaws - yaws[0] + np.pi) % (2 * np.pi) - np.pi))
        off_m = np.linalg.norm(pts - np.median(pts, axis=0), axis=1)
        off_deg = np.degrees(np.abs((yaws - yaw + np.pi) % (2 * np.pi) - np.pi))
        return np.median(pts, axis=0), yaw, off_m, off_deg

    if camera is None:
        camera = SceneCamera(engine, frame, topics, sam3_service, sam3_threshold, sam3_image_topic)
    prompts = [sam3_prompt]
    found, misses, asked = [], [], 0
    deadline = time.monotonic() + timeout_s
    body_shot = None
    if extra_prompt:
        body_pending = camera.request([extra_prompt])
        pending = camera.request(prompts)
        while body_shot is None:
            body_shot, why = camera.collect(body_pending, [extra_prompt], deadline)
            if body_shot is None:
                if why is None:
                    camera.cancel(pending)
                    raise NotLocated(f"no SAM3 {extra_prompt!r} frame in {timeout_s:.0f} s")
                misses.append(why)
                body_pending = camera.request([extra_prompt])
    else:
        pending = camera.request(prompts)
    window = None
    while time.monotonic() < deadline:
        shot, why = camera.collect(pending, prompts, deadline)
        pending = None
        if shot is None and why is None:
            break
        asked += 1
        pending = camera.request(prompts)  # SAM3 works on the next frame while this one is fitted
        if shot is None:
            misses.append(why)
            continue
        try:
            found.append((fit_interior(shot["depth_m"], shot["K"], shot["world_T_cam"], shot["mask"]), shot))
        except NotDetected as e:
            misses.append(str(e))
            log(f"frame not detected: {e}")
            continue
        if len(found) >= frames:
            last = found[-frames:]
            _, _, off_m, off_deg = medians(last)
            if np.all(off_m <= agree_m) and np.all(off_deg <= agree_deg):
                window = last
                break
    if pending is not None:
        camera.cancel(pending)
    if window is None:
        why = "; ".join(dict.fromkeys(misses[-3:])) or "-"
        raise NotLocated(
            f"no {frames} consecutive detected frames within {agree_m * 100:.1f} cm and {agree_deg:.1f} deg of "
            f"their median in {timeout_s:.0f} s ({len(found)} of {asked} SAM3 frames detected; last reasons: {why})"
        )
    mid, yaw, off_m, off_deg = medians(window)
    log(
        f"located: {frames} consecutive frames agree (detected {len(found)} of {asked}), cm from their median: "
        f"{np.round(off_m * 100, 1).tolist()}, deg: {np.round(off_deg, 1).tolist()}"
    )
    fit, shot = window[int(np.argmin(off_m))]
    return dict(
        point=mid.tolist(),
        insertion_axis=[float(np.cos(yaw)), float(np.sin(yaw)), 0.0],
        floor_z=float(np.median([r["floor_z"] for r, _ in window])),
        frames=len(window),
        detected=len(found),
        spread_m=float(np.max(off_m)),
        spread_deg=float(np.max(off_deg)),
        fit=fit,
        scene=shot,
        body=body_shot,
    )
