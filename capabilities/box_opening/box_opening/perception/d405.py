"""The wrist D405: its streams, where it sits on the arm, and the fix gate.

Box-opening owns these since RAMMP-CuRobo v1.0.0 became a pure planner and
deleted its perception (seek_core, cameras, perception.py and the D405
mount YAML) — they live nowhere else now, so there is no upstream copy to
drift from. Only what this repo uses came along: the grabber's three
streams and its mount, the n-agreeing-sightings commit gate, and the
quaternion math the mount composition needs.
"""

import os
from types import SimpleNamespace

import numpy as np

from box_opening.constants import load_yaml

CAMERA_CONFIG = "camera_d405_wrist.yaml"


def camera_config(name=CAMERA_CONFIG):
    """config/<name> from the installed package, else from the source tree."""
    candidates = []
    try:
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("box_opening")
        candidates.append(os.path.join(share, "config", name))
    except Exception:
        pass
    package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates.append(os.path.join(package_root, "config", name))
    for path in candidates:
        if os.path.isfile(path):
            with open(path) as f:
                return load_yaml(f)
    raise FileNotFoundError("camera config %r not found (tried %s)" % (name, candidates))


# The active robot's own camera mounts: rammp-deployments config/robots/<robot>/transforms/cameras.yaml, the
# file its static_tf publishes and a calibration replaces, in the robot config the node's container mounts at
# /config/robot (as the planner's own files are read there: runtime/body). BOX_OPENING_ROBOT_CAMERAS names
# another file (an offline tool); set empty, none is read.
ROBOT_CAMERAS = "/config/robot/transforms/cameras.yaml"
ROBOT_CAMERAS_ENV = "BOX_OPENING_ROBOT_CAMERAS"
# The OAK driver's optical frame in its root frame (the mount's child, body axes: x out of the lens, y left,
# z up): optical x = -y, y = -z, z = +x. With the mount's own turn and the flange's pose it gives TF's optical
# frame to 1e-4 on a 2026-11-17 capture (wrist-survey-20261117-164104, the lens to 0.04 mm).
BODY_TO_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
# WHAT STILL CHECKS THE WRIST MOUNT. TF is published from the robot's cameras.yaml, so TF against that file
# says nothing of the mount itself. What a mount does to a press is where it puts the point the look aims at:
# its lens AND its axes together, at the range the look aims from (the reach-out's look pose stands the lens
# ~0.30 m from the button: REACH_LOOK_RISE_M's note; 0.27-0.31 m on the recorded aims). A degree of turn is
# 5 mm there, as much as the lens's place. So a mount is held by its AIMED POINT (aim_point), against the two
# mounts presses have been flown on, both kept in the camera config for this and nothing else:
#  - wrist_lens_in_hand_m, the mount fitted from the 2026-10-22 depth-press run (rammp-deployments #128), axes
#    square: TF's from 10-22 to 11-09, the one figure no calibration session made. Its aims fell 6-11 mm short
#    along the bearing (2026-10-29's looks after the press), which reach_press.AIM_FORWARD_M (10 mm) makes up;
#  - wrist_mount_flown, the hand-eye calibration of 2026-11-09: TF's on the flights of 2026-11-12..17, the
#    last clean runs. It aims 10.9 mm from the fit's (10.7 across the hand, 2.2 along its y) and every press
#    on it came down ~1 cm to the right (owner, 2026-11-17); the box opened on five of them.
# AIM_REF_TOL_M: an aim further than this, across the hand, from BOTH of those is an aim no press has flown
# near: 10 mm, the centimetre a press is known to land beside the button's middle and still open it (the
# button is 44 mm across; 2026-10-22's 35-45 mm off was the lid's edge). Along the axis 20 mm, as the depth
# press always took. gen_1_5's calibration of 2026-11-18 aims 13.9 mm from the fit's (6.3 across, 12.4 along
# y) and 11.1 mm from the flown mount's (4.4, 10.2): outside both, mostly along the bearing, where it adds to
# AIM_FORWARD_M what the fit lacked. A run says the two distances as it starts and warns past the tolerance;
# the guarded depth press, which centres on the mount, does not start.
# WHAT IT CAN CATCH: a mount whose aim is a centimetre or more from every aim a press has flown on, a wrong
# sign, the hand measurement of 2026-10-19 (35 mm). WHAT IT CANNOT: say that a mount inside it is right (the
# flown mount is inside it and pressed a centimetre aside); tell two mounts under a centimetre apart; see an
# error both flown mounts share.
LOOK_RANGE_M = 0.30
AIM_REF_TOL_M = 0.010
AIM_REF_ALONG_TOL_M = 0.020
SQUARE_OPTICAL = np.diag([-1.0, -1.0, 1.0])  # the optical axes in the tool frame with the mount square on the hand


def aim_point(lens, optical=None, range_m=LOOK_RANGE_M):
    """Where a mount (the lens from the tool frame, its optical axes there; None: square) puts the point
    `range_m` down its optical axis, in the tool frame (m): what the look's aim goes by."""
    axis = SQUARE_OPTICAL[:, 2] if optical is None else np.asarray(optical, float)[:, 2]
    return np.asarray(lens, float)[:3] + float(range_m) * axis


def flown_mounts(cfg):
    """[(what it is, lens from the tool frame, optical axes)] the mounts presses have been flown on, as
    camera config `cfg` keeps them: its own wrist_lens_in_hand_m (the 2026-10-22 fit, square) and its
    wrist_mount_flown (on the flange: the tool frame is constants.TOOL_FROM_FLANGE_STOCK_M out of it)."""
    from box_opening.constants import TOOL_FROM_FLANGE_STOCK_M

    out = []
    fit = cfg.get("wrist_lens_in_hand_m")
    if fit is not None:
        out.append(("the 2026-10-22 fit", [float(v) for v in fit], SQUARE_OPTICAL))
    flown = cfg.get("wrist_mount_flown")
    if flown:
        xyz = [float(v) for v in flown["xyz"]]
        out.append(
            (
                "the mount flown %s" % flown.get("when", ""),
                [xyz[0], xyz[1], xyz[2] - TOOL_FROM_FLANGE_STOCK_M],
                quat_to_mat(*(float(v) for v in flown["quat_xyzw"])) @ BODY_TO_OPTICAL,
            )
        )
    return out


def aim_off_flown(cfg, lens, optical=None):
    """[(what, across the hand (m), [dx, dy, dz])] how far the aimed point of the mount (`lens`, `optical`)
    stands from each flown mount's (flown_mounts), in the tool frame; [] where the config keeps none or
    there is no lens."""
    if lens is None:
        return []
    mine = aim_point(lens, optical)
    out = []
    for what, ref_lens, ref_optical in flown_mounts(cfg):
        d = mine - aim_point(ref_lens, ref_optical)
        out.append((what, float(np.hypot(d[0], d[1])), [float(v) for v in d]))
    return out


def aim_refusal(cfg, lens, optical=None):
    """Why a press may not be centred on the mount (`lens`, `optical`), or None: its aimed point is further
    than AIM_REF_TOL_M across the hand, or AIM_REF_ALONG_TOL_M along the axis, from the aimed point of
    every mount a press has flown on (the comment above). None too where the config keeps no such mount."""
    off = aim_off_flown(cfg, lens, optical)
    if not off or any(aside <= AIM_REF_TOL_M and abs(d[2]) <= AIM_REF_ALONG_TOL_M for _w, aside, d in off):
        return None
    return (
        "its aim %.0f mm out is %s: further than %.0f mm across the hand (or %.0f along the axis) from every aim a "
        "press has flown on"
        % (
            1000 * LOOK_RANGE_M,
            " and ".join("%.1f mm from %s's" % (1000 * aside, what) for what, aside, _d in off),
            1000 * AIM_REF_TOL_M,
            1000 * AIM_REF_ALONG_TOL_M,
        )
    )


def robot_mount(name, path=None, env=None):
    """The robot's own mount of camera frame `name` (ROBOT_CAMERAS, or `path`): a namespace of its
    `parent` link, `xyz`, `quat` (xyzw, or None where the file gives none), the `path` it was read
    from and the `robot` that config says it is (the `robot:` of the mount.yaml beside transforms/,
    None where there is none). None: no such file here (the bench, a test, an offline tool without
    the robot's config), or no such mount in it. A file that is there and cannot be read as mounts
    is None too, said in the log: the caller falls back to its own figure."""
    env = os.environ if env is None else env
    if path is None:
        path = env[ROBOT_CAMERAS_ENV].strip() if ROBOT_CAMERAS_ENV in env else ROBOT_CAMERAS
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            params = (load_yaml(f).get("mounts") or {}).get("ros__parameters") or {}
        m = params.get(str(name))
        if not isinstance(m, dict):
            return None
        xyz = [float(v) for v in m["xyz"]]
        if len(xyz) != 3 or not all(np.isfinite(xyz)):
            raise ValueError("xyz %r" % (m["xyz"],))
        quat = m.get("quat_xyzw")
        quat = None if quat is None else [float(v) for v in quat]
        if quat is not None and (len(quat) != 4 or not all(np.isfinite(quat)) or not any(quat)):
            raise ValueError("quat_xyzw %r" % (m["quat_xyzw"],))
        return SimpleNamespace(parent=str(m["parent"]), xyz=xyz, quat=quat, path=str(path), robot=_robot_of(path))
    except Exception as e:  # noqa: BLE001 - an unreadable calibration is not a reason to stop: the copy stands in
        from box_opening import LOGGER as log

        log.info("[camera] WARNING: the robot's camera mounts %s give no usable %s (%s)" % (path, name, e))
        return None


def _robot_of(cameras_path):
    """The robot a config's transforms/cameras.yaml belongs to: the `robot:` of the mount.yaml one
    folder up (rammp-deployments config/robots/<robot>/), or None."""
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(cameras_path))), "mount.yaml")) as f:
            robot = load_yaml(f).get("robot")
        return None if robot is None else str(robot)
    except Exception:  # noqa: BLE001 - no mount.yaml beside it: not said
        return None


def wrist_camera_in_hand(cfg, chain=None, path=None, env=None):
    """(lens [x, y, z] or None, optical axes (3, 3) or None, where it is from): the wrist camera from
    box opening's tool frame, in the hand's axes (m; the optical frame's axes as columns).

    Where the robot's own calibration can be read it is that, not a copy kept here: camera config
    `cfg` names the mount whose child frame the driver puts AT the colour sensor (`robot_mount`:
    wrist_camera, the root of the OAK's tree). Its parent is whatever link of the hand the robot
    mounts it on (the flange, the sensor puck's link, the gripper's base, the description's tool
    frame): `chain` places that link from the flange by its URDF's fixed joints, and the tool frame
    stands `chain`.tool_from_flange_m out of the flange with the flange's axes, on every description.
    To the nanometre: the team's [0, 0.065, 0.015] on the flange is the [0, 0.065, -0.105] this was
    kept as. A copy written for one mount was wrong on the next (and once written in the wrong frame:
    #105, reverted by #106; on gen_1_5 it stands 7.0 mm aside of the 2026-11-18 calibration).

    Else `cfg`'s own wrist_lens_in_hand_m and no axes: a config that names no mount (the bench's
    D405, whose mount is in that same file), no robot config here, a parent this arm's URDF does not
    fix to its flange, or no chain to place it by. `chain`: None is the run's own (runtime/approach),
    where it has one."""
    lens = cfg.get("wrist_lens_in_hand_m")
    copy = None if lens is None else [float(v) for v in lens]
    name = cfg.get("robot_mount")
    if not name:
        return copy, None, "the camera config"
    mount = robot_mount(name, path=path, env=env)
    if mount is None:
        return copy, None, "the camera config (no robot calibration of %s here)" % name
    if chain is None:
        try:
            from box_opening.runtime import approach

            chain = approach._chain() if approach.chain_ready()[0] else None
        except Exception:  # noqa: BLE001 - no chain (no URDF yet): the copy
            chain = None
    find, tool = getattr(chain, "fixed_in_flange", None), getattr(chain, "tool_from_flange_m", None)
    if find is None or tool is None:
        return copy, None, "the camera config (no arm chain yet to place %s's mount by)" % name
    pose, why = find(mount.parent)
    if pose is None:
        return (
            copy,
            None,
            "the camera config (the robot mounts %s on %s, which the arm's URDF does not fix to its flange: %s)"
            % (
                name,
                mount.parent,
                why,
            ),
        )
    R, t = pose
    at = R @ np.asarray(mount.xyz, float) + t - np.array([0.0, 0.0, float(tool)])
    # (+ 0.0: no minus zero in what a square mount turns out; rounded, its axes are exactly square)
    optical = None if mount.quat is None else np.round(R @ quat_to_mat(*mount.quat) @ BODY_TO_OPTICAL, 12) + 0.0
    robot = "robot %s by its mount.yaml" % mount.robot if mount.robot else "no mount.yaml beside it to name the robot"
    where = "%s in %s (%s), on %s" % (name, mount.path, robot, mount.parent)
    return [round(float(v), 9) + 0.0 for v in at], optical, where


def wrist_lens_in_hand(cfg, chain=None, path=None, env=None):
    """(lens [x, y, z] or None, where it is from): wrist_camera_in_hand's lens."""
    lens, _optical, where = wrist_camera_in_hand(cfg, chain=chain, path=path, env=env)
    return lens, where


def quat_to_mat(x, y, z, w):
    """xyzw quaternion (normalized here) -> 3x3 rotation matrix."""
    n = (x * x + y * y + z * z + w * w) ** 0.5 or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_to_quat_xyzw(m):
    """3x3 rotation -> xyzw quaternion (the inverse of quat_to_mat)."""
    m = np.asarray(m, dtype=float)
    t = float(np.trace(m))
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        return np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2.0
    q = np.zeros(4)
    q[i] = 0.25 * s
    q[j] = (m[j, i] + m[i, j]) / s
    q[k] = (m[k, i] + m[i, k]) / s
    q[3] = (m[k, j] - m[j, k]) / s
    return q / np.linalg.norm(q)


# frame_snapshot's depth_stamp for a grabber that keeps none (a test's fake):
# the pairing check has nothing to judge. A real grabber's None means no
# depth yet, and the frame waits.
MISSING = object()


def frame_snapshot(g):
    """g.snapshot() when the grabber has one; otherwise the same namespace
    read from its attributes (depth_stamp MISSING when it keeps none)."""
    snap = getattr(g, "snapshot", None)
    if snap is not None:
        return snap()
    return SimpleNamespace(
        color=g.color,
        color_stamp=g.color_stamp,
        depth=g.depth,
        depth_stamp=getattr(g, "depth_stamp", MISSING),
        k=g.k,
        dist=getattr(g, "dist", None),
    )


def stable_fix(samples, tol, n):
    """Median position once the newest `n` sightings agree pairwise within
    `tol` metres, else None. `samples` = [(position, t), ...], newest last.

    The acquisition gate: one frame never moves the arm — n agreeing frames
    from a still camera do."""
    if len(samples) < n:
        return None
    pts = np.array([np.asarray(p, dtype=float) for p, _t in samples[-n:]])
    spread = np.linalg.norm(pts[:, None] - pts[None, :], axis=2)
    if float(spread.max()) > tol:
        return None
    return np.median(pts, axis=0)


class D405Grabber:
    """The newest colour frame (BGR), aligned depth (metres) and intrinsics
    of the wrist camera, with what composing the camera's pose needs: a TF
    buffer, the parent frame and the mount (config/camera_d405_wrist.yaml or
    config/camera_oak_wrist.yaml).  It only stores what arrives; callers spin
    their own node.

    need_depth=False (the OWL node): colour and intrinsics only.
    cfg_name: override the config file (None = CAMERA_CONFIG default)."""

    def __init__(self, node, need_depth=True, cfg_name=None):
        from tf2_ros import Buffer, TransformListener

        self.node = node
        cfg = camera_config(cfg_name or CAMERA_CONFIG)
        self.parent = cfg["parent_frame"]
        self.mount_xyz = np.asarray(cfg["mount_xyz"], dtype=float)
        self.mount_quat = list(cfg["mount_quat_xyzw"])
        # the stamp lag against the arm; camera_pose_at looks the pose up at
        # stamp + this
        self.stamp_offset_s = float(cfg.get("stamp_offset_s", 0.0) or 0.0)
        # the lens from tool_frame the depth press checks TF against (tasks/depth_press) and the lid's
        # table map takes the hand out by: the robot's own calibration where it can be read, else this
        # config's (wrist_lens_in_hand_m below, worked out when asked: the arm's chain is set by then)
        self._cfg = cfg
        # Topic paths: explicit keys take precedence over namespace-derived
        # defaults (OAK-D uses /rgb and /stereo instead of /color and
        # /aligned_depth_to_color).
        ns = cfg.get("depth_topic", "").rsplit("/depth/", 1)
        _ns = ns[0] if len(ns) == 2 else None
        self._color_topic = cfg.get("color_topic") or (_ns + "/color/image_raw" if _ns else None)
        self._info_topic = cfg.get("color_info_topic") or (_ns + "/color/camera_info" if _ns else None)
        self._depth_topic = cfg.get("aligned_depth_topic") or (
            _ns + "/aligned_depth_to_color/image_raw" if _ns else None
        )
        self.tf_buffer = Buffer()
        # spin_thread=False: the listener's own executor thread would spin
        # this node alongside the one that already spins it (the OWL node's
        # rclpy.spin, the mission's perception executor) — two executors
        # on one node, which rclpy does not guard. The OWL scene instance
        # went deaf after its first inferences three times on 2026-09-17
        # (ticks and params alive, image callbacks never delivered again,
        # re-subscribing did nothing). Whoever spins the node feeds TF.
        self.tf_listener = TransformListener(self.tf_buffer, node, spin_thread=False)
        self.need_depth = bool(need_depth)
        # each image is stored with its own stamp as ONE tuple, so a reader on
        # another thread never sees a frame with another frame's stamp, and
        # a depth is never paired with a colour it was not shot with
        self._color = self._depth = (None, None)
        self.info = None
        self.k = self.dist = None
        self._subs = []
        self.resubscribes = 0
        self._subscribe()

    @property
    def wrist_lens_in_hand_m(self):
        """The lens from tool_frame, in the hand's axes (wrist_lens_in_hand: the robot's own calibration on
        this run's tool frame, else the camera config's figure); None: unmeasured, the depth press does
        not centre."""
        return wrist_lens_in_hand(self._cfg)[0]

    def aim_refusal(self, lens, optical=None):
        """Why the depth press may not centre on TF's mount (`lens` from the tool frame, its axes), or None:
        d405.aim_refusal against the mounts this camera's config says presses have flown on."""
        return aim_refusal(self._cfg, lens, optical)

    def _subscribe(self):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image

        node = self.node
        self._subs = [
            node.create_subscription(Image, self._color_topic, self._color_cb, qos_profile_sensor_data),
            node.create_subscription(CameraInfo, self._info_topic, self._info_cb, qos_profile_sensor_data),
        ]
        if self.need_depth:
            self._subs.append(
                node.create_subscription(
                    Image,
                    self._depth_topic,
                    self._depth_cb,
                    qos_profile_sensor_data,
                )
            )

    def resubscribe(self):
        """Fresh image subscriptions (see SceneGrabber.resubscribe)."""
        for sub in self._subs:
            try:
                self.node.destroy_subscription(sub)
            except Exception:
                pass
        self._subs = []
        self.resubscribes += 1
        self._subscribe()

    def _color_cb(self, msg):
        if msg.encoding in ("rgb8", "bgr8"):
            a = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            arr = a[:, :, ::-1].copy() if msg.encoding == "rgb8" else a.copy()
            self._color = (arr, msg.header.stamp)

    def _depth_cb(self, msg):
        if msg.encoding == "16UC1":
            arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).astype(np.float32) / 1000.0
            self._depth = (arr, msg.header.stamp)

    @property
    def color(self):
        return self._color[0]

    @property
    def color_stamp(self):
        return self._color[1]

    @property
    def depth(self):
        return self._depth[0]

    @property
    def depth_stamp(self):
        return self._depth[1]

    def snapshot(self):
        """The newest colour and depth, each with its own stamp, and the
        intrinsics — each read ONCE, so the pixels, the stamp the pose is
        looked up at and the dedup key all belong to the same frames."""
        (color, color_stamp), (depth, depth_stamp) = self._color, self._depth
        return SimpleNamespace(
            color=color, color_stamp=color_stamp, depth=depth, depth_stamp=depth_stamp, k=self.k, dist=self.dist
        )

    def _info_cb(self, msg):
        k = np.array(msg.k).reshape(3, 3)
        self.k = k
        self.dist = np.array(msg.d, dtype=float).ravel()
        self.info = dict(fx=k[0, 0], fy=k[1, 1], cx=k[0, 2], cy=k[1, 2])

    def missing(self):
        """Which streams have not arrived yet (diagnosis)."""
        out = []
        if self.color is None:
            out.append("color (%s)" % self._color_topic)
        if self.need_depth and self.depth is None:
            out.append("depth (%s)" % self._depth_topic)
        if self.info is None:
            out.append("camera_info (%s)" % self._info_topic)
        return out
