"""The chair and its user, for the lines the planner never sees.

Every plan-free line (kinematics.ArmChain.straight_line / walk: the re-aim,
the slide, the turn, the stroke, the rise, the grip from the press, the
lift, the set-down, the way out) is flown without the planner, and the
planner itself may run with no world at all (flatbot's starts on
world_empty.yaml). So box_opening holds those lines to the boxes the
planner would: the robot's own body (the chair's platform, seat and the
person in it) from the planner's own files — robot-body config, which the
owner's rule allows; the cabinet, the table and what is on them stay the
cameras' each run.

The files: `body_layers` names a cuRobo planner config (default
/config/robot/curobo/planner.yaml, the planner's own; the node param
body_layers, each CLI's --body-layers, overridden by BOX_OPENING_BODY_LAYERS
when that is set — offline runs and the press study). From it: its `layers`
(each obstacle {name, position, dims} in the robot base frame, UNPADDED),
padded per side by its world_padding (cuRobo's default 0.02 when it names
none); its `robot` file's collision_spheres per link, each grown by its
collision_sphere_buffer. The base and the shoulder are exempt, as KCE's
enable_chair_collision: they stand where the arm is mounted at every joint
state, and the layers were carved around cuRobo's spheres of them. Every
other link with spheres is placed by the run's own URDF (the gripper's
fingers at 0, as cuRobo locks the knuckle open); robot.yaml's spheres are
the planner's, and box_opening adds what they miss of the real hand
(config/robot_body.yaml `hand_spheres`, robot-body config): the 131 mm
fingers closed on the tool axis (the stock 2F-85 spheres are locked open:
the tips were 13 mm outside them) and the wrist OAK and its bracket (12.6
mm outside them). Each group carries the boxes it stands for and its
spheres must cover them, or the load stops; a group the robot file's own
spheres already cover is skipped, never doubled (BodyCheck.hand_added /
hand_skipped).

A file that cannot be read, or that places nothing, is an error: the run
stops before any motion (never a silent skip). An explicitly empty value
turns the check off (the bench: no chair)."""

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_BODY_LAYERS = "/config/robot/curobo/planner.yaml"
BODY_LAYERS_ENV = "BOX_OPENING_BODY_LAYERS"
# the robot-body config beside it: the hand's spheres the planner's robot file lacks (and, from
# box_opening's own config/, the speed near the user)
ROBOT_BODY_FILE = "robot_body.yaml"
# a hand group's covers are sampled this finely, and a sample may lie this far outside its spheres
HAND_COVER_STEP_M = 0.0025
HAND_COVER_TOL_M = 0.001
WORLD_PADDING_M = 0.02  # cuRobo's default world_padding, where the planner config names none
# the arm's first two links: where it is mounted, at every joint state
EXEMPT_LINKS = ("base_link", "shoulder_link")
# a joint path is sampled at least this finely for the check (rad of the joint that moves most)
CHECK_STEP_RAD = 0.02
# A line that STARTS in a padded box may move in it or out of it, never this much deeper, and
# never into another: the arm is already there (a planner on an empty world put it there, or
# the layers' best guesses do — flatbot's HOME and its look poses beside the chair read 60-100 mm
# into seat/armrest_arm, measured 11-06 on the 10-21 URDF). What a line ENTERS is refused.
DEEPER_M = 0.005
_CHUNK = 256  # states checked at once (memory: chunk x spheres x boxes x 3)


class BodyModelError(RuntimeError):
    """The body model named could not be loaded: nothing may move."""


def resolve(param=None, env=None):
    """The planner config to read: BOX_OPENING_BODY_LAYERS where it is set
    (even empty), else `param` (the node's body_layers, a CLI's
    --body-layers), else DEFAULT_BODY_LAYERS. '' means off."""
    env = os.environ if env is None else env
    if BODY_LAYERS_ENV in env:
        return str(env[BODY_LAYERS_ENV]).strip()
    return DEFAULT_BODY_LAYERS if param is None else str(param).strip()


def _yaml(path):
    import yaml

    try:
        data = yaml.safe_load(Path(path).read_text())
    except (OSError, ValueError, yaml.YAMLError) as e:
        raise BodyModelError("body model: cannot read %s (%s)" % (path, e)) from e
    if not isinstance(data, dict):
        raise BodyModelError("body model: %s is not a mapping" % path)
    return data


def _bare(link):
    return link[4:] if link.startswith("arm_") else link


def robot_body_path():
    """config/robot_body.yaml: the installed package's, else this source tree's (an offline
    copy of the package), or None."""
    candidates = []
    try:
        from ament_index_python.packages import get_package_share_directory

        candidates.append(Path(get_package_share_directory("box_opening")) / "config" / ROBOT_BODY_FILE)
    except Exception:  # noqa: BLE001 - no ament index offline: the source tree's
        pass
    candidates.append(Path(__file__).resolve().parents[2] / "config" / ROBOT_BODY_FILE)
    return next((p for p in candidates if p.is_file()), None)


def box_points(centre, half, step=HAND_COVER_STEP_M):
    """(m, 3) points filling the box (centre, half extents) every `step` or finer, its faces and corners on."""
    c, h = np.asarray(centre, float), np.asarray(half, float)
    axes = [np.linspace(c[i] - h[i], c[i] + h[i], max(2, int(math.ceil(2.0 * h[i] / step)) + 1)) for i in range(3)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def outside_m(points, centres, radii):
    """(m,) how far each point lies outside the union of the spheres (<= 0: inside one)."""
    P, C, r = np.asarray(points, float), np.asarray(centres, float), np.asarray(radii, float)
    if not len(C):
        return np.full(len(P), np.inf)
    return (np.linalg.norm(P[:, None, :] - C[None], axis=-1) - r[None]).min(axis=1)


@dataclass
class HandGroup:
    """Spheres box_opening adds to the robot file's (config/robot_body.yaml `hand_spheres`)."""

    name: str
    link: str
    centres: np.ndarray  # (k, 3) in `link`
    radii: np.ndarray  # (k,) bare: the robot file's buffer is added where they are placed
    covers: list  # [(name, centre (3,), half (3,))] in `link`: what the spheres stand for


def load_hand(path):
    """The hand groups of robot-body config `path`, each checked to cover its boxes, or BodyModelError."""
    data = _yaml(path).get("hand_spheres") or {}
    if not isinstance(data, dict):
        raise BodyModelError("body model: %s's hand_spheres is not a mapping" % path)
    out = []
    for name, g in data.items():
        try:
            link = str(g["link"])
            cs = np.array([[float(v) for v in s["center"]] for s in g["spheres"]], float).reshape(-1, 3)
            rs = np.array([float(s["radius"]) for s in g["spheres"]], float)
            covers = [
                (str(c.get("name", k)), np.array(c["centre"], float).reshape(3), np.array(c["half"], float).reshape(3))
                for k, c in enumerate(g.get("covers") or [])
            ]
        except (KeyError, TypeError, ValueError) as e:
            raise BodyModelError("body model: %s's hand group %s is malformed (%s)" % (path, name, e)) from e
        if not len(rs) or not covers:
            raise BodyModelError("body model: %s's hand group %s has no spheres or nothing it covers" % (path, name))
        for cname, c, h in covers:
            worst = float(outside_m(box_points(c, h), cs, rs).max())
            if worst > HAND_COVER_TOL_M:
                raise BodyModelError(
                    "body model: %s's hand group %s leaves %s %.1f mm outside its spheres"
                    % (path, name, cname, 1000.0 * worst)
                )
        out.append(HandGroup(name=str(name), link=link, centres=cs, radii=rs, covers=covers))
    return out


@dataclass
class BodyModel:
    """The padded boxes and the arm's buffered spheres, as the planner pads them."""

    source: str
    base_link: str
    names: list  # "layer/obstacle"
    lo: np.ndarray  # (B, 3) padded
    hi: np.ndarray  # (B, 3)
    spheres: dict  # link -> (centres (k, 3), radii (k,)), buffer included, exempt links left out
    padding_m: float
    buffer_m: float
    hand: list = None  # [HandGroup] from robot-body config, placed (or skipped) by BodyCheck
    hand_source: str = None

    def check_for(self, chain, urdf_path, root):
        """A BodyCheck of this model on `chain`'s URDF, rooted at `root`."""
        return BodyCheck(self, chain, urdf_path, root)


def load(planner_yaml, robot_body=None):
    """The BodyModel from cuRobo planner config `planner_yaml`, or BodyModelError. `robot_body`:
    the robot-body config whose hand_spheres join the robot file's (None: robot_body_path(), which
    must exist; False: none, the planner's spheres alone)."""
    planner_yaml = Path(planner_yaml)
    cfg = _yaml(planner_yaml)
    here = planner_yaml.parent
    layers = cfg.get("layers")
    if not layers:
        raise BodyModelError("body model: %s names no layers" % planner_yaml)
    pad = float(cfg.get("world_padding", WORLD_PADDING_M))
    names, lo, hi, frames = [], [], [], {}
    for name in layers:
        data = _yaml(here / str(name))
        layer = str(data.get("layer") or Path(str(name)).stem)
        meta = data.get("meta")
        frame = meta.get("frame") if isinstance(meta, dict) else None
        if frame:  # the generated layers name it (arm_mount); it must be the spheres' base_link
            frames[layer] = str(frame)
        for o in data.get("obstacles") or []:
            try:
                c = np.array([float(v) for v in o["position"]], float)
                d = np.array([float(v) for v in o["dims"]], float)
            except (KeyError, TypeError, ValueError) as e:
                raise BodyModelError("body model: %s has an obstacle without position/dims (%s)" % (name, e)) from e
            if c.shape != (3,) or d.shape != (3,) or "pose" in o or "quat" in o:
                raise BodyModelError("body model: %s/%s is not an axis-aligned box" % (layer, o.get("name")))
            names.append("%s/%s" % (layer, o.get("name", len(names))))
            lo.append(c - d / 2.0 - pad)
            hi.append(c + d / 2.0 + pad)
    if not names:
        raise BodyModelError("body model: the layers of %s hold no obstacle" % planner_yaml)
    robot = cfg.get("robot")
    if not robot:
        raise BodyModelError("body model: %s names no robot file" % planner_yaml)
    kin = ((_yaml(here / str(robot)).get("robot_cfg") or {}).get("kinematics")) or {}
    raw = kin.get("collision_spheres")
    if not isinstance(raw, dict) or not raw:
        raise BodyModelError("body model: %s has no collision_spheres" % (here / str(robot)))
    base = str(kin.get("base_link") or "")
    wrong = {k: f for k, f in frames.items() if base and f != base}
    if wrong:
        raise BodyModelError(
            "body model: layer %s is in %s, the robot file's spheres in %s"
            % (", ".join(sorted(wrong)), ", ".join(sorted(set(wrong.values()))), base)
        )
    buf = float(kin.get("collision_sphere_buffer", 0.0) or 0.0)
    spheres = {}
    for link, ss in raw.items():
        if _bare(str(link)) in EXEMPT_LINKS:
            continue
        cs = np.array([[float(v) for v in s["center"]] for s in ss], float).reshape(-1, 3)
        rs = np.array([float(s["radius"]) + buf for s in ss], float)
        spheres[str(link)] = (cs, rs)
    if not spheres:
        raise BodyModelError("body model: %s has no spheres off the base and the shoulder" % robot)
    hand, hand_source = [], None
    if robot_body is not False:
        hand_source = robot_body if robot_body is not None else robot_body_path()
        if hand_source is None:
            raise BodyModelError(
                "body model: no %s (box_opening's config) to add the hand's spheres from" % ROBOT_BODY_FILE
            )
        hand = load_hand(hand_source)
    return BodyModel(
        source=str(planner_yaml),
        base_link=base,
        names=names,
        lo=np.array(lo),
        hi=np.array(hi),
        spheres=spheres,
        padding_m=pad,
        buffer_m=buf,
        hand=hand,
        hand_source=None if hand_source is None else str(hand_source),
    )


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


@dataclass
class Hit:
    """The first state of a path inside a padded box."""

    index: int  # into the path's states
    fraction: float  # of the way along the path
    link: str
    box: str
    depth_m: float  # how far the sphere's surface is inside the padded box

    def why(self, model):
        return "the chair and user: %s %.0f mm into %s (padded %.0f mm + %.0f mm), %.0f%% of the way" % (
            self.link,
            1000.0 * self.depth_m,
            self.box,
            1000.0 * model.padding_m,
            1000.0 * model.buffer_m,
            100.0 * self.fraction,
        )


class BodyCheck:
    """A BodyModel placed on the arm by its URDF: which joint states put a
    sphere of a link inside a padded box."""

    def __init__(self, model, chain, urdf_path, root):
        self.model = model
        if model.base_link and model.base_link != root:
            raise BodyModelError(
                "body model: %s's boxes are in %s, the arm's chain in %s" % (model.source, model.base_link, root)
            )
        joints = {}
        for j in ET.parse(urdf_path).getroot().findall("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            ax = j.find("axis")
            axis = np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()], float)
            axis = axis / (np.linalg.norm(axis) or 1.0)
            joints[j.find("child").get("link")] = (
                j.get("name"),
                j.get("type"),
                j.find("parent").get("link"),
                xyz,
                rpy,
                axis,
            )
        index = self.index = {n: k for k, n in enumerate(chain.joint_names)}
        # each sphere link's path from the root: (origin R, origin t, axis or None, arm joint index or None)
        self.paths = {}
        missing = []
        hand = list(model.hand or [])
        for link in list(model.spheres) + [g.link for g in hand if g.link not in model.spheres]:
            path, at = [], link
            while at != root:
                if at not in joints:
                    break
                name, typ, parent, xyz, rpy, axis = joints[at]
                moving = typ in ("revolute", "continuous") and name in index
                path.append((_rpy(*rpy), np.array(xyz, float), axis if moving else None, index.get(name)))
                at = parent
            if at != root:
                missing.append(link)
                continue
            self.paths[link] = path[::-1]
        if missing:
            raise BodyModelError(
                "body model: the arm's URDF %s places no %s from %s" % (urdf_path, ", ".join(missing), root)
            )
        # the spheres checked: the robot file's, and each hand group its own spheres do not already cover
        self.spheres = dict(model.spheres)
        self.hand_added, self.hand_skipped = [], []
        for g in hand:
            if self._robot_file_covers(g, joints):
                self.hand_skipped.append(g.name)
                continue
            cs, rs = self.spheres.get(g.link, (np.zeros((0, 3)), np.zeros(0)))
            self.spheres[g.link] = (np.vstack([cs, g.centres]), np.concatenate([rs, g.radii + model.buffer_m]))
            self.hand_added.append(g.name)
        self.paths = {link: p for link, p in self.paths.items() if link in self.spheres}

    @staticmethod
    def _rigid_root(link, joints):
        """The link `link` is fixed to: up its fixed joints to the first that moves."""
        while link in joints and joints[link][1] == "fixed":
            link = joints[link][2]
        return link

    def _link_frames(self, Q, links):
        """{link: (R (n, 3, 3), t (n, 3))} in the root at each row of Q."""
        n = len(Q)
        out = {}
        for link in links:
            R = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
            t = np.zeros((n, 3))
            for Ro, to, axis, k in self.paths[link]:
                t = t + R @ to
                R = R @ Ro
                if axis is not None:
                    th = Q[:, k]
                    K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
                    Rj = np.eye(3) + np.sin(th)[:, None, None] * K + (1.0 - np.cos(th))[:, None, None] * (K @ K)
                    R = R @ Rj
            out[link] = (R, t)
        return out

    def _robot_file_covers(self, group, joints):
        """Whether the robot file's own spheres, on the links fixed to `group`'s, already cover every
        box the group stands for (HAND_COVER_TOL_M): a robot file with the hand's spheres. Judged BARE,
        as load_hand judges the group's own: a robot file whose spheres fall short by less than the
        buffer is not taken for them (the box would keep none of the buffer the group adds)."""
        rigid = self._rigid_root(group.link, joints)
        mates = [link for link in self.model.spheres if self._rigid_root(link, joints) == rigid]
        if not mates:
            return False
        q0 = np.zeros((1, len(self.index)))
        frames = self._link_frames(q0, mates + [group.link])
        Rg, tg = frames[group.link]
        C, r = [], []
        for link in mates:
            R, t = frames[link]
            cs, rs = self.model.spheres[link]
            C.append(t[0] + cs @ R[0].T)
            r.append(rs - self.model.buffer_m)  # model.spheres carry the buffer: bare, as the group's
        C, r = np.concatenate(C), np.concatenate(r)
        for _name, c, h in group.covers:
            P = tg[0] + box_points(c, h) @ Rg[0].T
            if float(outside_m(P, C, r).max()) > HAND_COVER_TOL_M:
                return False
        return True

    def _centres(self, Q):
        """(n, S, 3) every checked sphere's centre at each row of Q, (S,) radii, [link per sphere]."""
        out, radii, links = [], [], []
        for link, (R, t) in self._link_frames(Q, list(self.paths)).items():
            cs, rs = self.spheres[link]
            out.append(t[:, None, :] + np.einsum("nij,kj->nki", R, cs))
            radii.append(rs)
            links += [link] * len(rs)
        return np.concatenate(out, axis=1), np.concatenate(radii), links

    def first_hit(self, pts, step_rad=CHECK_STEP_RAD, start_exempt=True):
        """The first state along joint path `pts` (its waypoints, joined by
        straight joint lines sampled every `step_rad`) that puts a sphere
        inside a padded box, as a Hit, or None. `start_exempt`: a sphere
        already in a box at the first state counts only where it goes
        DEEPER_M deeper (a line out of where the arm stands); False: every
        state counts, the first too (a pose the arm is yet to be flown to)."""
        P = np.asarray(pts, float)
        if P.ndim != 2 or len(P) == 0:
            return None
        if len(P) > 1 and step_rad:
            rows = [P[:1]]
            for a, b in zip(P[:-1], P[1:], strict=False):
                m = max(1, int(math.ceil(float(np.abs(b - a).max()) / step_rad)))
                f = (np.arange(1, m + 1) / m)[:, None]
                rows.append(a + f * (b - a))
            Q = np.concatenate(rows)
        else:
            Q = P
        allowed = None  # (S, B): how deep each sphere may be in each box
        for s in range(0, len(Q), _CHUNK):
            depth, links = self._depths(Q[s : s + _CHUNK])
            if allowed is None:
                allowed = np.where(depth[0] > 0.0, depth[0] + DEEPER_M, 0.0) if start_exempt else 0.0
            hit = depth > allowed
            if hit.any():
                i = int(np.argwhere(hit.any(axis=(1, 2)))[0][0])
                si, bi = np.unravel_index(int(np.argmax(np.where(hit[i], depth[i], -np.inf))), depth[i].shape)
                k = s + i
                return Hit(
                    index=k,
                    fraction=k / max(1, len(Q) - 1),
                    link=links[si],
                    box=self.model.names[bi],
                    depth_m=float(depth[i, si, bi]),
                )
        return None

    def _depths(self, Q):
        """(n, S, B) how far each sphere's surface is inside each padded box
        (> 0 inside) at each row of Q, and the link of each sphere. A box no
        sphere comes within its radius of over all of Q reads -inf (never
        inside): on the Jetson the full (n, S, B, 3) sum cost ~20 ms a line."""
        lo, hi = self.model.lo, self.model.hi
        C, r, links = self._centres(Q)
        n, S = C.shape[:2]
        out = np.full((n, S, len(lo)), -np.inf)
        span_lo = C.min(axis=0) - r[:, None]  # (S, 3): where each sphere reaches over Q
        span_hi = C.max(axis=0) + r[:, None]
        near = np.flatnonzero(np.any(np.all((span_lo[:, None, :] < hi) & (span_hi[:, None, :] > lo), axis=-1), axis=0))
        if near.size:
            Cb = C[:, :, None, :]
            d = np.maximum(lo[near] - Cb, Cb - hi[near])  # per axis: > 0 outside the box that side
            m = d.max(axis=-1)  # < 0: inside, -m from the nearest face
            dist = np.sqrt(np.square(np.maximum(d, 0.0)).sum(axis=-1))  # outside: to the box's nearest point
            out[:, :, near] = r[None, :, None] - np.where(m < 0.0, m, dist)
        return out, links

    def refusal(self, pts, step_rad=CHECK_STEP_RAD, start_exempt=True):
        """Why joint path `pts` enters the chair or its user, or None."""
        hit = self.first_hit(pts, step_rad, start_exempt)
        return None if hit is None else hit.why(self.model)
