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
"""

import yaml

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


class WorldStore:
    def __init__(self, bench_yaml_path, out_dir=None):
        with open(bench_yaml_path) as f:
            self._bench = yaml.safe_load(f)
        self._surface_z = None  # runtime-measured surface; overrides YAML when set
        self._surface_xy = None  # (x0, x1, y0, y1): where the scene camera saw that surface

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
        return kind + (("_" + tag) if tag else ""), PLANNER_WORLD
