"""The fridge in cuRobo's collision world: the cabinet, the door at any angle and its handle, as boxes.

cuRobo plans every fridge motion, so it has to see the fridge. The planner takes a world as a scene YAML it reads
itself (/rammp_curobo/set_world with a path), so the node writes the scene into `planner_world_dir`, a directory the
planner container mounts at the same path, and asks the planner to load it. The planner composes its config's layers
into every world it loads (on flatbot the wheelchair's drive base, seat and occupant), so the chair stays in every
plan; a pushed scene replaces the robot's own world.yaml, so it lists that world's layers again (`base_layers`:
flatbot's floor). `restore` puts the robot's world back when the action ends.

Every box is in arm_mount and unpadded: the planner adds its world_padding (2 cm) on every side. The door model is the
door file's (door_common.common.DoorFrame): a slab `door_t` thick (the door with its bins) from the hinge to the free
edge over the door's height, turning about the hinge; the cabinet behind it, `body_depth` deep; the handle a bar
standing out of the door's face. What a motion may touch decides which of them its world has (FridgeScene).
"""

import os
from pathlib import Path

import numpy as np
import yaml
from door_common.common import DoorFrame, Refused, door_dims, say
from scipy.spatial.transform import Rotation as R

# the handle: a bar this far out of the door's face (measured 11-06), this wide along the door, and this much taller
# than its grippable straight length (its mounts)
HANDLE_PROTRUSION_M = 3.5 * 0.0254
HANDLE_WIDTH_M = 0.05
HANDLE_MOUNTS_M = 0.05
HANDLE_LENGTH_M = 8 * 0.0254  # for a door file without handle_length_m: the node's (configure_handle sets it)
MIN_DIM_M = 0.002  # a piece of a cut slab thinner than this is left out
SCENE_NAME = "fridge_world.yaml"


def _yaw_deg(xy):
    return float(np.degrees(np.arctan2(xy[1], xy[0])))


def _door_z(door):
    lo, hi = door["door_z"]
    return float(lo), float(hi)


class FridgeModel:
    """The door file's door as boxes. `door`: the door dict (door_common's door file, or the open's plan of it).
    `prefix`: put before every box's name (several copies of the fridge in one world: the door during a wheelchair
    move)."""

    def __init__(self, door, handle_length_m=None, prefix=""):
        self.door = door
        self.df = DoorFrame(door, float(door.get("open_sign", 1.0)))
        if handle_length_m is None:  # the door file's, else the node's handle_length_m
            handle_length_m = door.get("handle_length_m", HANDLE_LENGTH_M)
        self.handle_length = float(handle_length_m)
        self.prefix = prefix

    def _box(self, name, u, v, deg, z, du, dv, dz):
        """A box centred at door-frame (u, v) at opening `deg` and height z, `du` along the door, `dv` along its
        normal, `dz` tall."""
        df = self.df
        centre = df.pos(u, v, deg, z)
        n_now = df._rot(deg) @ df.n
        return {
            "name": self.prefix + name,
            "position": [round(float(x), 5) for x in centre],
            "rpy_deg": [0.0, 0.0, round(_yaw_deg(n_now), 4)],
            "dims": [round(float(dv), 5), round(float(du), 5), round(float(dz), 5)],
        }

    def body(self):
        """The cabinet behind the closed door (it never moves)."""
        df = self.df
        _, _, _, body_depth = door_dims(self.door)
        lo, hi = _door_z(self.door)
        return [self._box("body", df.length / 2, -df.t / 2 - body_depth / 2, 0.0, (lo + hi) / 2, df.length,
                          body_depth, hi - lo)]

    def slab(self, deg, tag="door", window=None, v=None):
        """The door slab open `deg`. `window` (u0, u1, z0, z1): leave this part of it out, through its thickness --
        where the hand holds or pushes it. `v` (v0, v1): only this layer of it (off its mid-plane, + out of the door's
        front), default all of it."""
        df = self.df
        lo, hi = _door_z(self.door)
        v0, v1 = (-df.t / 2, df.t / 2) if v is None else v
        vc, dv = (v0 + v1) / 2, v1 - v0
        if window is None:
            return [self._box(tag, df.length / 2, vc, deg, (lo + hi) / 2, df.length, dv, hi - lo)]
        u0, u1 = max(0.0, window[0]), min(df.length, window[1])
        z0, z1 = max(lo, window[2]), min(hi, window[3])
        pieces = [
            ("hinge_side", 0.0, u0, lo, hi),
            ("free_side", u1, df.length, lo, hi),
            ("below", u0, u1, lo, z0),
            ("above", u0, u1, z1, hi),
        ]
        out = []
        for name, ua, ub, za, zb in pieces:
            if ub - ua > MIN_DIM_M and zb - za > MIN_DIM_M:
                out.append(self._box(f"{tag}_{name}", (ua + ub) / 2, vc, deg, (za + zb) / 2, ub - ua, dv, zb - za))
        return out

    def pushed(self, deg, u, z, toward_hinge, toward_edge, half_z, panel):
        """The door open `deg` as a hand pushing its inner face at (door-frame `u`, height `z`) may touch it: its outer
        `panel` (the front face, and the outer part of every edge) whole; its inner part (the bins) with a window round
        the hand left out (`toward_hinge` and `toward_edge` along the door, `half_z` up and down); the handle; the
        cabinet. The hand may press into the inner face, never through to the front or round the free edge."""
        df = self.df
        front = df.t / 2
        window = (u - toward_hinge, u + toward_edge, z - half_z, z + half_z)
        return (self.body() + self.slab(deg, "door_panel", v=(front - panel, front))
                + self.slab(deg, "door_inner", window=window, v=(-front, front - panel)) + self.handle(deg))

    def handle(self, deg, tag="handle"):
        """The handle bar open `deg`."""
        df = self.df
        z = float(self.door.get("handle_mid_z", self.door["closed_handle"][2]))
        return [self._box(tag, df.u_handle, df.t / 2 + HANDLE_PROTRUSION_M / 2, deg, z, HANDLE_WIDTH_M,
                          HANDLE_PROTRUSION_M, self.handle_length + HANDLE_MOUNTS_M)]

    # -- the worlds the motions plan in --

    def full(self, deg):
        """Everything: free-space moves round the fridge with the door open `deg`."""
        return self.body() + self.slab(deg) + self.handle(deg)

    def no_handle(self, deg):
        """The handle left out: onto or off it."""
        return self.body() + self.slab(deg)

    def cabinet(self):
        """The cabinet only: a move that pushes the door (it may touch it), and what every leg of a motion has."""
        return self.body()

    def holding(self, degs, u, z, half_u, half_z):
        """The hand holds the handle while the door goes from degs[0] to degs[-1]: the slab at each of `degs` with a
        window round the hand (door-frame `u`, height `z`, +-`half_u` / `half_z`), the handle left out."""
        window = (u - half_u, u + half_u, z - half_z, z + half_z)
        out = self.body()
        for k, deg in enumerate(degs):
            out += self.slab(deg, tag=f"door{k}", window=window)
        return out


def configure_handle(length_m):
    """The handle's grippable length (the node's handle_length_m), for door files that don't carry their own."""
    global HANDLE_LENGTH_M
    HANDLE_LENGTH_M = float(length_m)


class PlannerWorld:
    """Pushes scenes into the cuRobo planner (self.arm.set_world) and puts the robot's own world back.

    `write_dir`: where this node writes the scene; `planner_dir`: the same directory as the planner sees it (the
    planner container mounts it; the same path in both by default). `base_layers`: layer files (as the planner sees
    them) every pushed scene lists, standing in for the robot's world.yaml it replaces. `restore_world`: the world to
    put back, resolved by the planner next to its config. `extra`: obstacle dicts every scene also has (the level
    mount's stand-in for the chair)."""

    def __init__(self, arm, write_dir, planner_dir=None, base_layers=(), restore_world="world.yaml", extra=()):
        self.arm = arm
        self.write_dir = Path(write_dir)
        self.planner_dir = Path(planner_dir or write_dir)
        self.base_layers = list(base_layers)
        self.restore_world = restore_world
        self.extra = list(extra)
        self.loaded = None  # the obstacles the planner holds now (pushed by this action), None: not ours
        self.pushes = 0

    def push(self, obstacles, label):
        """Make `obstacles` (+ `extra`) the planner's world. Nothing is sent if it holds exactly these already.
        Raises Refused if the planner refuses it."""
        obstacles = self.extra + list(obstacles)
        if obstacles == self.loaded:
            return
        scene = {"base_frame": "arm_mount", "layers": self.base_layers, "obstacles": obstacles}
        self.write_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.write_dir / f".{SCENE_NAME}.tmp"
        tmp.write_text(f"# written by the fridge capability: {label}\n" + yaml.safe_dump(scene, sort_keys=False))
        os.replace(tmp, self.write_dir / SCENE_NAME)
        ok, msg = self.arm.set_world(str(self.planner_dir / SCENE_NAME))
        if not ok:
            self.loaded = None
            raise Refused(f"the cuRobo planner refused the fridge's collision world ({label}): {msg}")
        self.loaded = obstacles
        self.pushes += 1

    def restore(self):
        """Put the robot's own world back (also after a cancel). Says so if the planner refuses it."""
        if self.loaded is None and not self.pushes:
            return
        try:
            ok, msg = self.arm.set_world(self.restore_world, even_if_cancelled=True)
        except Exception as e:  # noqa: BLE001 -- the action is ending either way
            ok, msg = False, repr(e)
        self.loaded, self.pushes = None, 0
        if not ok:
            say(f"WARNING: could not put the planner's world back to {self.restore_world}: {msg}")


def hand_corners(pos, quat, box):
    """The 8 corners of the hand's box `box` (((x0, x1), (y0, y1), (z0, z1)) in the tool frame: constants
    HAND_BOX_TOOL_M) with the tool at (`pos`, `quat`), in arm_mount."""
    corners = np.array([[x, y, z] for x in box[0] for y in box[1] for z in box[2]], float)
    return np.asarray(pos, float) + R.from_quat(quat).apply(corners)


def hand_reach(quat, direction, box):
    """How far the hand's box reaches past the tool point along `direction` (arm_mount), the hand turned `quat`."""
    d = np.asarray(direction, float)
    return float(np.max((hand_corners(np.zeros(3), quat, box)) @ (d / np.linalg.norm(d))))


def chair_stand_in(behind_m):
    """The level mount's stand-in for the chair (no chair layers there): everything more than `behind_m` behind the
    arm base, on its right."""
    return {
        "name": "chair_stand_in",
        "position": [-behind_m - 0.5, -0.75, 0.0],
        "rpy_deg": [0.0, 0.0, 0.0],
        "dims": [1.0, 1.5, 2.0],
    }
