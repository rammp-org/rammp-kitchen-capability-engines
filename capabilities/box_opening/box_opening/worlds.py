"""The planner's world: the one its deployment starts it with, never one
box opening makes.

Box opening sends the planner no world (owner, 2026-10-29: "we need to be
using the standard world empty yaml that everyone is using. we need to not
dynamically populate objects into the yaml and we need to move away from
statically adding a table"). The planner plans in the world it was started
with, rammp_curobo's world_empty.yaml on flatbot (the arm and nothing else),
and it refuses an empty scene sent at runtime (core/rammp_curobo/world.py:
compose_cuboids allows one only when the planner is built), so there is
nothing to send: PlannerClient.set_world sends nothing. What stands on the
table — the table itself, the box, what the cameras see — is the mission's
own, measured each run, and keeps no place in any planner world.

WorldStore keeps the run's measured table (table_top_z, table_extent) and
the bench file's robot settings; push_name names a leg's world (callers
gate on the name's prefix, "full"/"bench"/"interaction") and gives the
planner's: PLANNER_WORLD.

One exception, the owner's (2026-11-12): while a way HOME is planned the
planner is handed the table the run measured, and nothing else. The planner
knowing no table planned HOME through it (24 of 344 poses over a 29 in
table, and from both poses the arm was left in on 2026-11-10), and checking
its plans afterwards (core.under_table) refused what it could have planned
around: with the table in its world every one of those poses homed, plans
only, on the robot's planner. It is done the way rammp-feeding hands the
planner its world: a file in a folder the planner's container mounts
(PlannerTable; SetWorld by path), only where the deployment shares one
(PLANNER_WORLDS_AS_ENV; off without it, and everything above holds as it
did), and the planner's own world put back when the plans are made. The
file is the planner's start-up scene and one block more: nothing but the
measured table, and nothing of it kept from one run to the next.
"""

import hashlib
import os
from contextlib import contextmanager
from pathlib import Path

from box_opening import LOGGER as log
from box_opening.constants import load_yaml

# The world the planner holds, by name: rammp_curobo's own, the deployment's
# (configs/world_empty.yaml on flatbot)
PLANNER_WORLD = "world_empty.yaml"
# The margins box opening's own clearances over the box stand on (they were
# the container cuboid's err-tall top and the planner's per-side padding
# while box opening wrote the box into the planner's world; reach_press's
# clears keep the hand above them still)
ERR_TALL_M = 0.02
PLANNER_PADDING_M = 0.02
# What the cameras saw standing on the table, as clear_table.obstacle_boxes
# names its cuboids (perception/lid_room_view draws them)
SEEN_PREFIX = "seen_"


# The folder the planner is handed worlds from: where box opening writes them
# (default: planner_worlds under the state folder) and the same folder as the
# planner's container reads it (unset or empty: box opening sends no world)
PLANNER_WORLDS_DIR_ENV = "BOX_OPENING_PLANNER_WORLDS_DIR"
PLANNER_WORLDS_AS_ENV = "BOX_OPENING_PLANNER_WORLDS_AS"
# The planner's table, as it stands padded: its top this far under the top
# the cameras measured. A hand resting near the table has buffered spheres
# reaching the measured top, and at the exact height the planner refuses
# such a start (2026-11-12, plans only: 21 of 28 poses homed at 0, 28 of 28
# at 20 mm); core.under_table still judges every plan at the measured top
HOME_TABLE_DROP_M = 0.02
HOME_TABLE_DEPTH_M = 1.0  # a block, not a slab: nothing is planned under the top
TABLE_BOX = "box_opening_table"
WORLDS_KEPT = 8  # written world files kept in the folder, the newest


def planned_top_z(model, cpose):
    """The height box opening's clears over the box keep the hand above:
    the box's top, ERR_TALL_M taller, PLANNER_PADDING_M more — what the
    planner padded it to while the box was in its world (a start whose
    gripper spheres reached below this was refused, 2026-10-27/28)."""
    return float(cpose.xyz[2]) + float(model.dims[2]) + ERR_TALL_M + PLANNER_PADDING_M


def _table_top_z(bench):
    t = next((o for o in bench.get("obstacles", ()) if o["name"] == "table"), None)
    if t is None:
        return None
    return t["position"][2] + t["dims"][2] / 2


class PlannerTable:
    """The folder a deployment shares with the planner, and the planner's own start-up scene: what a table world
    is written from. `write_dir` is the folder as box opening sees it, `as_dir` as the planner's container does;
    `config_path` the planner's config (the one the body check reads: runtime/body.resolve). `boot_world` is what
    that config names as its world, the name SetWorld takes to put it back."""

    def __init__(self, write_dir, as_dir, config_path):
        self.write_dir, self.as_dir = Path(write_dir), str(as_dir).rstrip("/")
        self.config_path = Path(config_path)
        with open(self.config_path) as f:
            cfg = load_yaml(f)
        if not isinstance(cfg, dict) or not cfg.get("world"):
            raise ValueError("%s names no world" % self.config_path)
        self.boot_world = str(cfg["world"])
        self.padding = float((cfg.get("planner") or {}).get("world_padding", PLANNER_PADDING_M))
        self._scene_path = (self.config_path.parent / self.boot_world).resolve()
        with open(self._scene_path) as f:
            self._scene = load_yaml(f)
        if not isinstance(self._scene, dict) or "base_frame" not in self._scene:
            raise ValueError("%s is no scene" % self._scene_path)
        self._config_layers = {
            (self.config_path.parent / str(ref)).resolve()
            for ref in (cfg.get("layers") or [])
            if not str(ref).startswith("example:")
        }
        if not (cfg.get("layers") or self._scene.get("layers") or self._scene.get("obstacles")):
            # SetWorld refuses an empty scene, so the planner's own could not be put back
            raise ValueError("the planner's own world (%s) is empty: it could not be put back" % self.boot_world)

    @classmethod
    def from_env(cls, env=None):
        """The PlannerTable the environment names, or None: PLANNER_WORLDS_AS_ENV unset or empty (off), the body
        check's planner config off, or anything of it unreadable (said once; box opening then sends no world)."""
        env = os.environ if env is None else env
        as_dir = str(env.get(PLANNER_WORLDS_AS_ENV, "")).strip()
        if not as_dir:
            return None
        from box_opening.constants import state_dir
        from box_opening.runtime import approach

        try:
            # the planner's config is the one the body check reads (--body-layers, BOX_OPENING_BODY_LAYERS)
            config = getattr(approach.body_model(), "source", None)
        except RuntimeError as e:
            config = None
            log.info("[worlds] the planner's config is unreadable (%s)" % e)
        if not config:
            log.info("[worlds] %s is set but the planner's config is off: no world is sent" % PLANNER_WORLDS_AS_ENV)
            return None
        write_dir = str(env.get(PLANNER_WORLDS_DIR_ENV, "")).strip() or str(state_dir() / "planner_worlds")
        try:
            return cls(write_dir, as_dir, config)
        except (OSError, ValueError, KeyError, TypeError) as e:
            log.info("[worlds] no world is sent to the planner: %s" % e)
            return None

    def ours(self, path):
        """Whether `path` names a world this folder holds, as the planner reads it."""
        return str(path).startswith(self.as_dir + "/")

    def block(self, table_z, extent):
        """(x0, x1, y0, y1, top): the table as the planner holds it, padded: the extent the cameras saw, its top
        HOME_TABLE_DROP_M under the measured one. None for an extent too small to be a block."""
        x0, x1, y0, y1 = (float(v) for v in extent)
        if x1 - x0 <= 2 * self.padding or y1 - y0 <= 2 * self.padding:
            return None
        return (x0, x1, y0, y1, float(table_z) - HOME_TABLE_DROP_M)

    def write(self, table_z, extent):
        """Write the table world (the planner's start-up scene and the table's block) and return its path as the
        planner reads it; None where there is no block. Layer files the scene lists are copied beside it (a
        scene's layers are read from its own folder); a layer the planner's config lists is in every world
        already and is left out."""
        import yaml

        blk = self.block(table_z, extent)
        if blk is None:
            return None
        x0, x1, y0, y1, top = blk
        pad = self.padding
        self.write_dir.mkdir(parents=True, exist_ok=True)
        layers = []
        for i, ref in enumerate(self._scene.get("layers") or []):
            ref = str(ref)
            if ref.startswith("example:"):
                layers.append(ref)
                continue
            src = (self._scene_path.parent / ref).resolve()
            if src in self._config_layers:
                continue
            name = "layer%d_%s" % (i, src.name)
            _write_atomic(self.write_dir / name, src.read_bytes())
            layers.append(name)
        table = {
            "name": TABLE_BOX,
            "position": [
                round((x0 + x1) / 2, 4),
                round((y0 + y1) / 2, 4),
                round(top - pad - HOME_TABLE_DEPTH_M / 2, 4),
            ],
            "dims": [round(x1 - x0 - 2 * pad, 4), round(y1 - y0 - 2 * pad, 4), HOME_TABLE_DEPTH_M],
        }
        scene = {"base_frame": self._scene["base_frame"], "layers": layers}
        scene["obstacles"] = list(self._scene.get("obstacles") or []) + [table]
        for key in ("objects", "targets"):
            if self._scene.get(key):
                scene[key] = self._scene[key]
        body = (
            "# written by box opening (worlds.PlannerTable): the planner's start-up scene (%s) and the table this run\n"
            "# measured, held by the planner only while a way HOME is planned\n" % self.boot_world
        ) + yaml.safe_dump(scene, default_flow_style=None, sort_keys=False)
        name = "home_table_%s.yaml" % hashlib.sha1(body.encode()).hexdigest()[:10]
        _write_atomic(self.write_dir / name, body.encode())
        for old in sorted(self.write_dir.glob("home_table_*.yaml"), key=lambda p: p.stat().st_mtime)[:-WORLDS_KEPT]:
            if old.name != name:
                old.unlink(missing_ok=True)
        return "%s/%s" % (self.as_dir, name)


def _write_atomic(path, data):
    """`data` at `path`, whole or not at all: the planner never reads half a file. Unchanged content is left."""
    path = Path(path)
    if path.exists() and path.read_bytes() == data:
        os.utime(path)
        return
    tmp = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    tmp.write_bytes(data)
    os.replace(tmp, path)


_PLANNER_TABLE = [False]  # the environment's PlannerTable, read once (False: not read yet)


def planner_table():
    """The PlannerTable this process's environment names, or None (PlannerTable.from_env, read once)."""
    if _PLANNER_TABLE[0] is False:
        _PLANNER_TABLE[0] = PlannerTable.from_env()
    return _PLANNER_TABLE[0]


class WorldStore:
    def __init__(self, bench_yaml_path, out_dir=None):
        with open(bench_yaml_path) as f:
            self._bench = load_yaml(f)
        self._surface_z = None  # runtime-measured surface; overrides YAML when set
        self._surface_xy = None  # (x0, x1, y0, y1): where the scene camera saw that surface
        self._home_world = []  # home_plans in force: the table world's path as the planner reads it, or None

    @property
    def table_top_z(self):
        """Measured table-top height in the planning frame.

        Read from the bench YAML's 'table' obstacle when present; otherwise
        None until the scene camera measures the surface (set by press_demo
        after the first SceneFix)."""
        if self._surface_z is not None:
            return self._surface_z
        return _table_top_z(self._bench)

    @table_top_z.setter
    def table_top_z(self, value):
        self._surface_z = float(value) if value is not None else None

    @property
    def table_extent(self):
        """(x0, x1, y0, y1) where the scene camera saw the table, or None."""
        return self._surface_xy

    @table_extent.setter
    def table_extent(self, value):
        self._surface_xy = None if value is None else tuple(float(v) for v in value)

    @property
    def base_frame(self):
        """The planning root frame for this robot config ('base_link' on the
        bench, 'arm_mount' on flatbot)."""
        return self._bench.get("base_frame", "base_link")

    @property
    def external_static_tf(self):
        """True when an external node (e.g. sheppy's static_tf) already
        publishes /tf_static — the launch file must not spawn its own."""
        return bool(self._bench.get("external_static_tf", False))

    @property
    def wrist_camera_config(self):
        """Camera config name for the wrist camera, or None (use default)."""
        return self._bench.get("wrist_camera_config")

    @property
    def scene_camera_cfg(self):
        """Scene camera config dict for SceneGrabber, or None (use Gemini defaults)."""
        name = self._bench.get("scene_camera_config")
        if name is None:
            return None
        from box_opening.perception.d405 import camera_config

        return camera_config(name)

    @property
    def table_normal(self):
        """The table's normal in base_link as measured (see the yaml), or
        exactly +z when the bench file does not say."""
        n = self._bench.get("table_normal_base", [0.0, 0.0, 1.0])
        n = [float(v) for v in n]
        s = sum(v * v for v in n) ** 0.5 or 1.0
        return [v / s for v in n]

    def push_name(
        self, kind, model=None, cpose=None, contact_z=None, depth_max=None, lid_at=None, tag="", container_pad_xy=0.0
    ):
        """(the leg's world name, PLANNER_WORLD): `kind` ("bench", "full",
        "interaction") and `tag` name it as before — the runner's gates read
        the prefix — and nothing is built or written: what the arguments
        described (the box, its pad, the lid, the contact's plane) is no
        longer the planner's."""
        if kind not in ("bench", "full", "interaction"):
            raise ValueError("unknown world kind %r" % kind)
        held = self._home_world[-1] if self._home_world else None
        return kind + (("_" + tag) if tag else ""), held or PLANNER_WORLD

    @property
    def home_table_held(self):
        """Whether the planner is being handed the run's table just now (inside home_plans, with one)."""
        return bool(self._home_world and self._home_world[-1])

    def home_table_block(self):
        """(x0, x1, y0, y1, top): the table as the planner would hold it while a way HOME is planned (padded: the
        extent the scene camera saw, PlannerTable.block), or None: no folder shared with the planner, or no
        table measured this run (a bench file's own table is not a measurement: it has no extent)."""
        table = planner_table()
        if table is None or self._surface_z is None or self._surface_xy is None:
            return None
        return table.block(self._surface_z, self._surface_xy)

    @contextmanager
    def home_plans(self, table=True):
        """While this holds, every leg's world is the run's measured table (push_name gives the written file's
        path, as the planner reads it), where home_table_block gives one and `table` allows it; otherwise the
        planner's own, as outside it. Yields whether the table world is in force. The caller puts the planner's
        own back when it leaves (core.home_any)."""
        path = None
        if table and self.home_table_block() is not None:
            try:
                path = planner_table().write(self._surface_z, self._surface_xy)
            except OSError as e:
                log.info("[worlds] the table world could not be written (%s): HOME is planned in the planner's own" % e)
        self._home_world.append(path)
        try:
            yield path is not None
        finally:
            self._home_world.pop()
