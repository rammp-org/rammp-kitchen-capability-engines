"""The planner's world is the one its deployment starts it with
(rammp_curobo's world_empty.yaml): box opening builds, writes and sends
none (owner 2026-10-29). WorldStore keeps the run's measured table and the
bench file's robot settings for the mission's own geometry."""

from types import SimpleNamespace

import pytest
import yaml

from box_opening.models.container import ContainerModel, ContainerPose
from box_opening.worlds import PLANNER_WORLD, WorldStore

CFG = "capabilities/box_opening/config/containers/oxo_pop.yaml"
BENCH = "capabilities/box_opening/config/world_bench.yaml"
FLATBOT = "capabilities/box_opening/config/world_flatbot.yaml"


def test_every_world_a_leg_names_is_the_planners_own_and_nothing_is_written(tmp_path, monkeypatch):
    """Owner 2026-10-29: "we need to be using the standard world empty yaml
    that everyone is using. we need to not dynamically populate objects into
    the yaml and we need to move away from statically adding a table". Every
    kind of world a leg plans in — whatever box, pad, lid or contact it once
    described — is rammp_curobo's world_empty.yaml by name, and no file is
    written; the name keeps its kind (the runner's gates read the prefix)."""
    monkeypatch.setenv("BOX_OPENING_STATE", str(tmp_path))
    m = ContainerModel.load(CFG)
    cp = ContainerPose(xyz=(0.55, -0.40, 0.30), yaw=0.0)
    for bench in (BENCH, FLATBOT):
        store = WorldStore(bench)
        store.table_top_z, store.table_extent = 0.30, (0.45, 1.10, -0.90, -0.10)
        assert store.push_name("bench", model=m) == ("bench", PLANNER_WORLD)
        assert store.push_name("bench", model=None, tag="bare") == ("bench_bare", PLANNER_WORLD)
        assert store.push_name("full", model=m, cpose=cp, container_pad_xy=0.03) == ("full", PLANNER_WORLD)
        assert store.push_name("full", model=m, cpose=cp, lid_at=cp, tag="lid") == ("full_lid", PLANNER_WORLD)
        assert store.push_name("interaction", model=m, cpose=cp, contact_z=0.31, depth_max=0.02, tag="button") == (
            "interaction_button",
            PLANNER_WORLD,
        )
    assert PLANNER_WORLD == "world_empty.yaml"
    assert not list(tmp_path.rglob("*.yaml"))  # no world file anywhere under the state dir
    with pytest.raises(ValueError):
        WorldStore(FLATBOT).push_name("kitchen")


def test_the_client_sends_the_planner_no_world():
    """rammp_curobo refuses an empty scene sent at runtime (compose_cuboids
    allows one only when the planner is built), and box opening sends none:
    PlannerClient.set_world records the name and calls no service."""
    from box_opening.runtime.client import PlannerClient

    client = SimpleNamespace(world_held=None)  # no _set_world: a service call would raise
    ok, msg = PlannerClient.set_world(client, PLANNER_WORLD)
    assert ok and client.world_held == PLANNER_WORLD and "sends no world" in msg


def test_the_measured_table_is_the_missions_and_overrides_the_bench_files():
    """The table the scene camera measures each run is the mission's own
    (the lid's spot, the set-down's height) — never the planner's. The lab
    bench's file states one; flatbot's states none (the chair goes to a
    different table every time) until the scene camera measures it."""
    bench = WorldStore(BENCH)
    with open(BENCH) as f:
        t = next(o for o in yaml.safe_load(f)["obstacles"] if o["name"] == "table")
    assert bench.table_top_z == pytest.approx(t["position"][2] + t["dims"][2] / 2)
    flatbot = WorldStore(FLATBOT)
    assert flatbot.table_top_z is None and flatbot.table_extent is None
    flatbot.table_top_z, flatbot.table_extent = 0.2933, (0.45, 1.10, -0.90, -0.10)
    assert flatbot.table_top_z == pytest.approx(0.2933)
    assert flatbot.table_extent == (0.45, 1.10, -0.90, -0.10)
    assert flatbot.base_frame == "arm_mount"


def test_the_flatbot_world_file_carries_only_keys_worldstore_reads():
    """world_flatbot.yaml once set depth_collision: true, which nothing read,
    under a comment claiming OAK depth streamed into the planner's collision
    world. It holds the robot's settings, no table."""
    with open(FLATBOT) as f:
        w = yaml.safe_load(f)
    assert set(w) <= {
        "base_frame",
        "table_normal_base",
        "wrist_camera_config",
        "scene_camera_config",
        "external_static_tf",
        "obstacles",
    }
    assert "table" not in {o["name"] for o in w["obstacles"]}
