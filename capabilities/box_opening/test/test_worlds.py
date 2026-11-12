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
    # nor a 'floor' that reads as a keep-out: nothing here reaches the planner (worlds.py)
    assert w["obstacles"] == []


# --- the one world box opening sends: the measured table, while a way HOME is planned (owner 2026-11-12) ---


def _planner_config(
    tmp_path, world="world.yaml", config_layers=("layer_chair.yaml",), scene_layers=("layer_floor.yaml",)
):
    """A planner config folder as flatbot's: planner.yaml naming its world and layers, the world a floor layer."""
    cfg = tmp_path / "curobo"
    cfg.mkdir()
    for name in set(config_layers) | set(scene_layers):
        (cfg / name).write_text(
            "layer: %s\nobstacles:\n  - {name: a, position: [0, 0, -1], dims: [1, 1, 0.1]}\n" % name[6:-5]
        )
    (cfg / "planner.yaml").write_text(
        "robot: robot.yaml\nworld: %s\nlayers: [%s]\n" % (world, ", ".join(config_layers))
    )
    (cfg / world).write_text("base_frame: arm_mount\nlayers: [%s]\nobstacles: []\n" % ", ".join(scene_layers))
    return cfg / "planner.yaml"


@pytest.fixture
def shared(tmp_path, monkeypatch):
    """A folder shared with the planner (written at tmp_path/out, read by the planner as /box_opening)."""
    from box_opening import worlds

    table = worlds.PlannerTable(tmp_path / "out", "/box_opening/", _planner_config(tmp_path))
    monkeypatch.setattr(worlds, "_PLANNER_TABLE", [table])
    return table


def test_the_table_world_is_the_planners_own_scene_and_the_measured_table_and_nothing_else(tmp_path, shared):
    from box_opening import worlds

    path = shared.write(0.2159, (0.35, 1.10, -1.0, 0.6))
    assert path.startswith("/box_opening/home_table_") and shared.ours(path) and not shared.ours(PLANNER_WORLD)
    with open(tmp_path / "out" / path.rsplit("/", 1)[1]) as f:
        scene = yaml.safe_load(f)
    assert set(scene) == {"base_frame", "layers", "obstacles"} and scene["base_frame"] == "arm_mount"
    # the scene's own layer, copied beside the file (a scene's layers are read from its folder); the chair's
    # layer is the planner config's, in every world already
    assert scene["layers"] == ["layer0_layer_floor.yaml"]
    assert (tmp_path / "out" / "layer0_layer_floor.yaml").read_text() == (
        tmp_path / "curobo" / "layer_floor.yaml"
    ).read_text()
    (box,) = scene["obstacles"]
    assert box["name"] == worlds.TABLE_BOX
    pad, top = 0.02, 0.2159 - worlds.HOME_TABLE_DROP_M
    # padded by the planner, the block is the seen extent with its top HOME_TABLE_DROP_M under the measured one
    assert box["dims"] == pytest.approx([0.75 - 2 * pad, 1.6 - 2 * pad, worlds.HOME_TABLE_DEPTH_M])
    assert box["position"][:2] == pytest.approx([0.725, -0.2])
    assert box["position"][2] + box["dims"][2] / 2 + pad == pytest.approx(top, abs=1e-4)
    assert shared.block(0.2159, (0.35, 1.10, -1.0, 0.6)) == pytest.approx((0.35, 1.10, -1.0, 0.6, top))
    assert shared.write(0.2159, (0.35, 1.10, -1.0, 0.6)) == path  # the same table is the same file
    assert shared.write(0.2656, (0.35, 1.10, -1.0, 0.6)) != path
    assert shared.block(0.2159, (0.50, 0.53, -1.0, 0.6)) is None  # an extent thinner than the padding: no block
    for k in range(worlds.WORLDS_KEPT + 4):  # the folder keeps the newest few
        shared.write(0.30 + 0.001 * k, (0.35, 1.10, -1.0, 0.6))
    assert len(list((tmp_path / "out").glob("home_table_*.yaml"))) == worlds.WORLDS_KEPT


def test_a_planner_whose_own_world_could_not_be_put_back_is_handed_none(tmp_path):
    from box_opening import worlds

    with pytest.raises(ValueError, match="could not be put back"):
        worlds.PlannerTable(
            tmp_path / "out", "/box_opening", _planner_config(tmp_path, config_layers=(), scene_layers=())
        )


def test_only_inside_home_plans_and_only_with_a_measured_table_does_a_leg_name_the_table_world(shared):
    store = WorldStore(FLATBOT)
    assert store.home_table_block() is None  # nothing measured
    with store.home_plans() as held:
        assert not held and store.push_name("bench", tag="bare") == ("bench_bare", PLANNER_WORLD)
    store.table_top_z = 0.2159
    assert store.home_table_block() is None  # a height with no extent is no block
    store.table_extent = (0.35, 1.10, -1.0, 0.6)
    assert store.push_name("bench", tag="bare") == ("bench_bare", PLANNER_WORLD)  # outside: the planner's own
    with store.home_plans() as held:
        name, path = store.push_name("full", model=None)
        assert held and store.home_table_held and name == "full" and shared.ours(path)
        with store.home_plans(table=False) as inner:
            assert not inner and store.push_name("bench")[1] == PLANNER_WORLD
        assert store.push_name("bench")[1] == path
    assert not store.home_table_held and store.push_name("full", model=None) == ("full", PLANNER_WORLD)


def test_with_no_folder_shared_nothing_is_written_or_named(tmp_path, monkeypatch):
    from box_opening import worlds

    monkeypatch.delenv(worlds.PLANNER_WORLDS_AS_ENV, raising=False)
    monkeypatch.setattr(worlds, "_PLANNER_TABLE", [False])
    monkeypatch.setenv("BOX_OPENING_STATE", str(tmp_path))
    store = WorldStore(FLATBOT)
    store.table_top_z, store.table_extent = 0.2159, (0.35, 1.10, -1.0, 0.6)
    assert worlds.planner_table() is None and store.home_table_block() is None
    with store.home_plans() as held:
        assert not held and store.push_name("bench") == ("bench", PLANNER_WORLD)
    assert not list(tmp_path.rglob("*.yaml"))


def _client(answers):
    """A PlannerClient's set_world on a stand-in whose SetWorld answers from `answers` (world -> (ok, message),
    default ok); .sent lists what was sent."""
    from box_opening.runtime.client import PlannerClient

    sent = []

    def send(world, timeout_s=10.0):
        sent.append(world)
        return answers.get(world, (True, "world set to %s" % world))

    c = SimpleNamespace(world_held=None, _send_world=send, sent=sent)
    c.set_world = lambda w: PlannerClient.set_world(c, w)
    c.planner_own_world = lambda: PlannerClient.planner_own_world(c)
    return c


def test_the_client_sends_the_table_world_and_puts_the_planners_own_back(shared):
    table = shared.write(0.2159, (0.35, 1.10, -1.0, 0.6))
    c = _client({})
    assert c.set_world(table)[0] and c.sent == ["world.yaml", table]  # the planner's own first: what it held is unknown
    assert c.set_world(table) == (True, "held") and len(c.sent) == 2
    assert c.set_world(PLANNER_WORLD)[0] and c.sent[-1] == "world.yaml" and c.world_held == PLANNER_WORLD
    assert c.set_world(PLANNER_WORLD) == (True, "held") and len(c.sent) == 3
    c.planner_own_world()  # nothing of ours is held: nothing sent
    assert len(c.sent) == 3
    c.set_world(table)
    c.planner_own_world()
    assert c.sent[-2:] == [table, "world.yaml"] and c.world_held == PLANNER_WORLD
    fresh = _client({})  # a run's first plan, not a home: the planner's own is put back whatever it held
    assert fresh.set_world(PLANNER_WORLD)[0] and fresh.sent == ["world.yaml"]
    fresh.planner_own_world()
    assert fresh.sent == ["world.yaml"]


def test_a_table_world_the_planner_refuses_leaves_it_its_own_and_the_plan_goes_on(shared):
    table = shared.write(0.2159, (0.35, 1.10, -1.0, 0.6))
    c = _client({table: (False, "world file not found")})
    ok, msg = c.set_world(table)
    assert ok and "not found" in msg  # the plan goes on, in the planner's own world
    assert c.world_held == PLANNER_WORLD and c.sent == ["world.yaml", table]
    assert c.set_world(table)[0] and c.world_held == PLANNER_WORLD  # asked again, refused again, still its own
