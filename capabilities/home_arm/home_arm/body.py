"""The chair and its user as the planner sees them, and the way out of them.

cuRobo refuses to plan from a start inside its world (INVALID_START_STATE_WORLD_COLLISION), and on a
wheelchair-mounted arm that world holds the chair, the seat and the person in it, each box padded by
the planner's world_padding. An arm left within that padding (a stop, a jog, an e-stop near the
user) is clear of the chair itself but cannot be planned from. This module reads the planner's own
files and finds a short joint-space step out of the padding, so the planner can take over from
there.

The files (robot config, never environment): `planner_yaml` (the planner's config, mounted at
/config/robot/curobo/planner.yaml on the robot) names its `layers`, each obstacle an axis-aligned
box {name, position, dims} in the robot's base frame, UNPADDED and padded here per side by its
planner.world_padding (rammp-curobo's default 0.02 m where it names none; every layer box is padded,
no_pad_names exempting scene obstacles only), and its `robot` file, whose
collision_spheres per link are grown by its collision_sphere_buffer. Those are exactly the planner's
boxes and spheres. The links are placed by forward kinematics of the driver's URDF, every joint
other than the arm's at 0 (cuRobo locks the gripper open). The base and the shoulder are left out:
they stand where the arm is mounted at every joint state, and the layers are generated clear of
them.

Pure numpy and yaml: no ROS, so it is tested on its own."""

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

WORLD_PADDING_M = 0.02  # cuRobo's default world_padding, where the planner config names none
EXEMPT_LINKS = ("base_link", "shoulder_link")  # with or without the arm_ prefix

# The way out of the padding (escape): down the deepest sphere's depth in joint space,
# ESCAPE_STEP_RAD a step (the joint that moves most), never deeper, the tool never lower than
# ESCAPE_DROP_M under where it started (the table and what stands on it are below the hand), each
# joint with limits kept LIMIT_ROOM_RAD inside them (or no nearer than it stands), until every
# sphere is ESCAPE_CLEAR_M outside the padding. Where no step gets further and the arm is already
# ESCAPE_OUT_M out, the way out ends there. Measured on flatbot (2026-11-09, the arm 8 in forward on
# the chair, cuRobo planning the home from each end): of 101 physically clear starts inside the
# padding, 88 got out and planned home.
# Only from a start IN THE PADDING, never from one where a sphere stands in a box itself (deeper
# than the padding): "never deeper" holds the deepest sphere alone, so from there another link
# could go as deep, into the box. Such a start is refused (in_a_box), here and for any other
# plan-free path (path_refusal): a deliberate limit, the arm is jogged clear by hand.
ESCAPE_CLEAR_M = 0.010
ESCAPE_OUT_M = 0.001
ESCAPE_STEP_RAD = 0.02
ESCAPE_MAX_RAD = 0.6  # no joint moves further than this from the start: it is a step, not a motion
ESCAPE_DROP_M = 0.005
ESCAPE_MAX_STEPS = 200
# rammp-curobo refuses a start inside its joint-limit margin (joint_limit_margin_rad 0.10 on
# flatbot), and its finetune failed from up to 0.154 rad of room (box_opening, 2026-10-23): the step
# out keeps each joint this far inside its limits, or no nearer than it stands
LIMIT_ROOM_RAD = 0.18
FD_RAD = 1e-4  # central differences for the depth's gradient


class BodyModelError(RuntimeError):
    """The planner's files cannot be read, or place nothing."""


def _yaml(path):
    try:
        data = yaml.safe_load(Path(path).read_text())
    except (OSError, ValueError, yaml.YAMLError) as e:
        raise BodyModelError(f"body model: cannot read {path} ({e})") from e
    if not isinstance(data, dict):
        raise BodyModelError(f"body model: {path} is not a mapping")
    return data


def _bare(link):
    return link[4:] if link.startswith("arm_") else link


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = (
        math.cos(r),
        math.sin(r),
        math.cos(p),
        math.sin(p),
        math.cos(y),
        math.sin(y),
    )
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _axis_rot(axis, theta):
    """(n, 3, 3) rotations by angles `theta` (n,) about the unit `axis`."""
    x, y, z = axis
    c, s = np.cos(theta), np.sin(theta)
    t = 1.0 - c
    return np.stack(
        [
            np.stack([t * x * x + c, t * x * y - s * z, t * x * z + s * y], -1),
            np.stack([t * x * y + s * z, t * y * y + c, t * y * z - s * x], -1),
            np.stack([t * x * z - s * y, t * y * z + s * x, t * z * z + c], -1),
        ],
        -2,
    )


class Chain:
    """Forward kinematics of a URDF from `root`, for the arm joints `joints` (the rest at 0)."""

    def __init__(self, urdf_text, root, joints):
        self.joints = list(joints)
        index = {n: k for k, n in enumerate(self.joints)}
        self._by_child = {}
        self.limits = [(None, None)] * len(self.joints)
        found = set()
        try:
            self._read(ET.fromstring(urdf_text), index, found)
        except (ET.ParseError, AttributeError, TypeError, ValueError) as e:
            raise BodyModelError(f"body model: the URDF cannot be read ({e!r})") from e
        self.root = root
        missing = [n for n in self.joints if n not in found]
        if missing:
            raise BodyModelError(
                f"body model: the URDF has no revolute or continuous joints {missing}"
            )

    def _read(self, robot, index, found):
        """The joints of the URDF element `robot` into _by_child and limits."""
        for j in robot.findall("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            ax = j.find("axis")
            axis = np.array(
                [float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()], float
            )
            axis = axis / (np.linalg.norm(axis) or 1.0)
            name, kind = j.get("name"), j.get("type")
            moving = kind in ("revolute", "continuous") and name in index
            if moving:
                found.add(name)
            self._by_child[j.find("child").get("link")] = (
                j.find("parent").get("link"),
                _rpy(*rpy),
                np.array(xyz, float),
                axis if moving else None,
                index.get(name) if moving else None,
            )
            lim = j.find("limit")
            if name in index and kind == "revolute" and lim is not None:
                self.limits[index[name]] = (float(lim.get("lower")), float(lim.get("upper")))

    def tool_z(self, Q, link):
        """(n,) the height of `link` in the root frame at the joint states Q."""
        return self.frames(Q, [link])[link][:, 2, 3]

    def has(self, link):
        at = link
        while at != self.root:
            if at not in self._by_child:
                return False
            at = self._by_child[at][0]
        return True

    def frames(self, Q, links):
        """{link: (n, 4, 4)} of each of `links` at the joint states Q (n, len(joints))."""
        Q = np.atleast_2d(np.asarray(Q, float))
        n = len(Q)
        cache = {self.root: np.broadcast_to(np.eye(4), (n, 4, 4)).copy()}

        def get(link):
            if link in cache:
                return cache[link]
            parent, R, t, axis, k = self._by_child[link]
            T = get(parent).copy()
            local = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
            local[:, :3, :3] = R
            local[:, :3, 3] = t
            if axis is not None:
                rot = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
                rot[:, :3, :3] = _axis_rot(axis, Q[:, k])
                local = local @ rot
            cache[link] = T @ local
            return cache[link]

        return {link: get(link) for link in links}


@dataclass
class BodyModel:
    """The planner's padded boxes (lo, hi: (B, 3)) and the arm's spheres per link (centre in the
    link frame (k, 3), radius with the buffer (k,)), placed by `chain`."""

    source: str
    names: list
    lo: np.ndarray
    hi: np.ndarray
    spheres: dict
    padding_m: float
    base_link: str = ""
    chain: Chain = None
    tool_link: str = ""

    def place(self, chain, tool_link):
        """This model on the arm `chain`; `tool_link` is the link whose height the escape holds."""
        missing = [link for link in self.spheres if not chain.has(link)]
        if missing:
            raise BodyModelError(f"body model: the URDF has no path from {chain.root} to {missing}")
        if not chain.has(tool_link):
            raise BodyModelError(f"body model: the URDF has no tool link {tool_link}")
        self.chain, self.tool_link = chain, tool_link
        self._links = list(self.spheres)
        self._centres = [self.spheres[k][0] for k in self._links]
        self._radii = np.concatenate([self.spheres[k][1] for k in self._links])
        return self

    def _sphere_centres(self, Q):
        F = self.chain.frames(Q, self._links)
        out = []
        for link, c in zip(self._links, self._centres, strict=True):
            T = F[link]
            out.append(np.einsum("nij,kj->nki", T[:, :3, :3], c) + T[:, None, :3, 3])
        return np.concatenate(out, axis=1)  # (n, S, 3)

    def depth(self, Q):
        """(n,) how deep the deepest sphere's surface stands inside a padded box at each joint state
        (m; > 0 inside the padding, < 0 how far outside every box)."""
        C = self._sphere_centres(Q)  # (n, S, 3)
        centre = (self.lo + self.hi) / 2.0
        half = (self.hi - self.lo) / 2.0
        e = np.abs(C[:, :, None, :] - centre) - half  # (n, S, B, 3)
        m = e.max(axis=-1)
        signed = np.where(m < 0.0, m, np.sqrt(np.square(np.maximum(e, 0.0)).sum(axis=-1)))
        return (self._radii[None, :, None] - signed).max(axis=(1, 2))

    def tool_z(self, Q):
        """(n,) the tool link's height in the base frame."""
        return self.chain.tool_z(Q, self.tool_link)

    def in_a_box(self, depth_m):
        """Why no plan-free step is taken from a start whose deepest sphere stands `depth_m` in the
        padded boxes, or '': '' in the padding or out of it, the reason where the sphere is in a
        box itself, past the padding. Every sphere stands at most as deep as the deepest, so from a
        start in the padding a path that goes no deeper keeps every link out of the boxes."""
        inside = float(depth_m) - self.padding_m
        if inside <= 1e-9:
            return ""
        return (
            f"the arm stands {1000 * inside:.0f} mm inside a chair or user box itself, past its "
            f"{1000 * self.padding_m:.0f} mm of padding, and a plan-free step from there is not "
            "held out of the box"
        )

    def path_refusal(self, path):
        """Why the joint path `path` (k, J; its points close together) would take the arm into the
        padding, or '': from a start outside it, no point enters; from a start inside it, no point
        is deeper than the start; a start in a box itself is refused (in_a_box)."""
        depth = self.depth(path)
        boxed = self.in_a_box(depth[0])
        if boxed:
            return boxed
        allowed = max(float(depth[0]), 0.0) + 1e-6
        k = int(np.argmax(depth))
        if depth[k] > allowed:
            return (
                f"it goes {1000 * depth[k]:.0f} mm into the padding round the chair and user "
                f"({k} of {len(depth) - 1} along)"
            )
        return ""

    def escape(self, q0):
        """The way out of the padding from joint state `q0`: (path (k, J) from q0, why). `path` None
        with `why` where there is none within ESCAPE_MAX_RAD a joint, or where q0 stands in a box
        itself (in_a_box); a path of q0 alone where q0 is already out of the padding."""
        q0 = np.asarray(q0, float)
        J = len(q0)
        lo = np.full(J, -np.inf)
        hi = np.full(J, np.inf)
        for j, (a, b) in enumerate(self.chain.limits):
            if a is not None:
                lo[j] = min(a + LIMIT_ROOM_RAD, q0[j])
                hi[j] = max(b - LIMIT_ROOM_RAD, q0[j])
        z0 = float(self.tool_z(q0)[0])
        q, d = q0.copy(), float(self.depth(q0)[0])
        if d <= 0.0:
            return np.array([q0]), ""
        boxed = self.in_a_box(d)
        if boxed:
            return None, boxed
        start_mm = 1000.0 * d
        path = [q0.copy()]
        eye = np.eye(J) * FD_RAD
        for _ in range(ESCAPE_MAX_STEPS):
            if d <= -ESCAPE_CLEAR_M:
                return np.array(path), ""
            if float(np.max(np.abs(q - q0))) > ESCAPE_MAX_RAD:
                break
            both = np.vstack([q + eye, q - eye])
            D, Z = self.depth(both), self.tool_z(both)
            g, zg = (D[:J] - D[J:]) / (2 * FD_RAD), (Z[:J] - Z[J:]) / (2 * FD_RAD)
            way = -g
            if (
                float(self.tool_z(q)[0]) - z0 <= -ESCAPE_DROP_M + 0.001
                and way @ zg < 0.0
                and zg @ zg > 0.0
            ):
                way = (
                    way - (way @ zg) / (zg @ zg) * zg
                )  # at the floor of the drop: nothing downward
            if float(np.max(np.abs(way))) < 1e-9:
                break
            dq = way / float(np.max(np.abs(way))) * ESCAPE_STEP_RAD
            for _half in range(4):
                qn = np.clip(q + dq, lo, hi)
                dn = float(self.depth(qn)[0])
                if dn <= d + 1e-6 and float(self.tool_z(qn)[0]) >= z0 - ESCAPE_DROP_M:
                    break
                dq = dq / 2.0
            else:
                break
            q, d = qn, dn
            path.append(q.copy())
        if d <= -ESCAPE_OUT_M:
            return np.array(path), ""
        return None, (
            f"the arm is {start_mm:.0f} mm inside the padding round the chair and user, and "
            f"every way out within {ESCAPE_MAX_RAD:.1f} rad a joint goes deeper, lowers the tool "
            f"more than {1000 * ESCAPE_DROP_M:.0f} mm or meets a joint limit"
        )


def padding(cfg):
    """The world_padding of planner config `cfg` (m), where rammp-curobo reads it: its `planner`
    block, merged over the default (WORLD_PADDING_M). A top-level `world_padding` (this module's
    reading before 2026-11-10, which the planner ignores) is refused rather than guessed at."""
    block = cfg.get("planner") or {}
    if not isinstance(block, dict):
        raise BodyModelError("body model: the planner config's `planner` entry is not a mapping")
    if "world_padding" in cfg and "world_padding" not in block:
        raise BodyModelError(
            "body model: world_padding is at the planner config's top level, where the planner "
            "does not read it (it reads planner.world_padding)"
        )
    try:
        pad = float(block.get("world_padding", WORLD_PADDING_M))
    except (TypeError, ValueError) as e:
        raise BodyModelError(f"body model: planner.world_padding is not a number ({e})") from e
    if not (math.isfinite(pad) and pad >= 0.0):
        raise BodyModelError(f"body model: planner.world_padding {pad} is not >= 0")
    return pad


def load(planner_yaml):
    """The BodyModel of cuRobo planner config `planner_yaml`, or BodyModelError: a file that cannot
    be read, and one whose entries are not what the planner's are (a sphere without a radius, a
    buffer that is no number, a robot_cfg that is no mapping), never another exception."""
    try:
        return _load(planner_yaml)
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        raise BodyModelError(
            f"body model: {planner_yaml} or a file it names is malformed ({e!r})"
        ) from e


def _load(planner_yaml):
    planner_yaml = Path(planner_yaml)
    cfg = _yaml(planner_yaml)
    here = planner_yaml.parent
    layers = cfg.get("layers")
    if not layers:
        raise BodyModelError(f"body model: {planner_yaml} names no layers")
    pad = padding(cfg)
    names, lo, hi = [], [], []
    for name in layers:
        data = _yaml(here / str(name))
        layer = str(data.get("layer") or Path(str(name)).stem)
        for o in data.get("obstacles") or []:
            try:
                c = np.array([float(v) for v in o["position"]], float)
                d = np.array([float(v) for v in o["dims"]], float)
            except (KeyError, TypeError, ValueError) as e:
                raise BodyModelError(
                    f"body model: {name} has an obstacle without position/dims ({e})"
                ) from e
            turned = any(float(v) != 0.0 for v in o.get("rpy_deg") or [])
            if c.shape != (3,) or d.shape != (3,) or "pose" in o or "quat" in o or turned:
                raise BodyModelError(
                    f"body model: {layer}/{o.get('name')} is not an axis-aligned box"
                )
            names.append(f"{layer}/{o.get('name', len(names))}")
            lo.append(c - d / 2.0 - pad)
            hi.append(c + d / 2.0 + pad)
    if not names:
        raise BodyModelError(f"body model: the layers of {planner_yaml} hold no obstacle")
    robot = cfg.get("robot")
    if not robot:
        raise BodyModelError(f"body model: {planner_yaml} names no robot file")
    kin = ((_yaml(here / str(robot)).get("robot_cfg") or {}).get("kinematics")) or {}
    raw = kin.get("collision_spheres")
    if not isinstance(raw, dict) or not raw:
        raise BodyModelError(f"body model: {here / str(robot)} has no collision_spheres")
    buf = float(kin.get("collision_sphere_buffer", 0.0) or 0.0)
    spheres = {}
    for link, ss in raw.items():
        if _bare(str(link)) in EXEMPT_LINKS:
            continue
        cs = np.array([[float(v) for v in s["center"]] for s in ss], float).reshape(-1, 3)
        rs = np.array([float(s["radius"]) + buf for s in ss], float)
        spheres[str(link)] = (cs, rs)
    if not spheres:
        raise BodyModelError(f"body model: {robot} has no spheres off the base and the shoulder")
    return BodyModel(
        str(planner_yaml),
        names,
        np.array(lo),
        np.array(hi),
        spheres,
        pad,
        str(kin.get("base_link") or ""),
    )
