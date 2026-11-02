"""Every CLI's arm chain is rooted in the frame it plans in.

On flatbot base_link leans 30 deg from the gravity-aligned arm_mount the
planner and the world use. home_arm never re-rooted the chain (only
press_demo did), so its "straight up" --lift-first moved the fingers 40 mm
toward the box for 69 mm up. And a missing /arm/robot_description left
press_demo with a chain rooted where the bench URDF has no link: it flew to
staging and died there on a KeyError (review 2026-10-22).

rclpy is never initialised: the node, the subscription and the client are
stubs."""

import glob
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from box_opening.constants import HOME
from box_opening.primitives.core import LiftFree, PlanState
from box_opening.runtime import approach
from box_opening.runtime.kinematics import ArmChain
from box_opening.tasks import cli_common, home_arm, press_demo

PKG = Path(__file__).resolve().parents[1]
FLATBOT_URDF = str(PKG / "test" / "data" / "arm_flatbot_20261021.urdf")
FLATBOT_WORLD = str(PKG / "config" / "world_flatbot.yaml")
BENCH_WORLD = str(PKG / "config" / "world_bench.yaml")
CONTAINER = str(PKG / "config" / "containers" / "ankou_pink.yaml")


@pytest.fixture(autouse=True)
def chain_state(monkeypatch, tmp_path):
    monkeypatch.setenv("BOX_OPENING_STATE", str(tmp_path))
    saved = (approach._ROOT, approach._URDF_PATH, approach._CHAIN, approach._GRAVITY)
    yield
    approach._ROOT, approach._URDF_PATH, approach._CHAIN, approach._GRAVITY = saved


class _Client:
    joint_names = ["joint_%d" % i for i in range(1, 8)]

    def __init__(self):
        self._base_frame = None

    def joints(self):
        return list(HOME)


class _NoSubNode:
    """A node that must never be asked for the description."""

    def create_subscription(self, *a, **k):
        raise AssertionError("the bench subscribed to /arm/robot_description")


def _args(world):
    return Namespace(container=CONTAINER, bench_world=world, execute=False)


def _stub_runtime(monkeypatch, node=None):
    monkeypatch.setattr(cli_common, "init_runtime", lambda execute: (node or object(), _Client()))


def test_build_ctx_on_flatbot_roots_the_chain_at_arm_mount_and_the_lift_is_vertical(monkeypatch):
    _stub_runtime(monkeypatch)
    monkeypatch.setattr(cli_common, "_arm_urdf_path", lambda node, frame: FLATBOT_URDF)
    ctx, _runner = cli_common.build_ctx(_args(FLATBOT_WORLD))
    assert approach._ROOT == "arm_mount"
    assert approach._URDF_PATH == FLATBOT_URDF
    legs, _ = LiftFree(0.08).plan(ctx, PlanState(joints=list(HOME), chain=0))
    pts = [p.positions for p in legs[0].traj.points]
    fb = ArmChain(root="arm_mount", path=FLATBOT_URDF)
    d = (fb.fk(pts[-1])[1] - fb.fk(pts[0])[1]) * 1000
    # it was [+40, 0, +69.3] mm: the tilted base's z
    assert np.allclose(d, [0.0, 0.0, 80.0], atol=1.0), d


def test_build_ctx_on_the_bench_keeps_the_default_chain_and_never_subscribes(monkeypatch):
    _stub_runtime(monkeypatch, node=_NoSubNode())
    cli_common.build_ctx(_args(BENCH_WORLD))
    assert approach._ROOT == "base_link"
    assert approach._URDF_PATH is None


def test_a_root_without_its_urdf_refuses_every_chain_with_a_runtime_error():
    approach.set_chain_root("arm_mount", None)
    with pytest.raises(RuntimeError, match="arm_mount"):
        approach._chain()
    with pytest.raises(RuntimeError, match="arm_mount"):
        approach._gravity()
    assert approach.chain_ready()[0] is False


def test_the_bench_urdf_has_no_arm_mount_and_says_so():
    with pytest.raises(RuntimeError, match="root arm_mount"):
        ArmChain(root="arm_mount", path=None)


def test_lift_free_refuses_a_chain_rooted_off_the_planning_frame(monkeypatch):
    _stub_runtime(monkeypatch)
    monkeypatch.setattr(cli_common, "_arm_urdf_path", lambda node, frame: FLATBOT_URDF)
    ctx, _runner = cli_common.build_ctx(_args(FLATBOT_WORLD))
    approach.set_chain_root("base_link", None)  # what home_arm had before
    with pytest.raises(RuntimeError, match="chain root base_link != planning frame arm_mount"):
        LiftFree(0.08).plan(ctx, PlanState(joints=list(HOME), chain=0))


def test_home_arm_lift_without_a_description_is_refused_and_nothing_runs(monkeypatch):
    _stub_runtime(monkeypatch)
    monkeypatch.setattr(cli_common, "_arm_urdf_path", lambda node, frame: None)
    ran = []

    class _Runner:
        def __init__(self, client):
            self.time_scale = 1.0

        def run(self, legs, execute=False):
            ran.append(legs)
            return []

    monkeypatch.setattr(cli_common, "Runner", _Runner)
    args = _args(FLATBOT_WORLD)
    args.speed_scale = 1.0
    with pytest.raises(SystemExit) as e:
        cli_common.run_task(args, lambda ctx: home_arm.build_legs(ctx, 0.08))
    assert "plan refused — arm holds" in str(e.value)
    assert "arm_mount" in str(e.value)
    assert ran == []


def test_press_demo_without_the_description_exits_before_anything_moves(monkeypatch):
    _stub_runtime(monkeypatch)
    monkeypatch.setattr(cli_common, "_arm_urdf_path", lambda node, frame: None)
    called = []
    monkeypatch.setattr(press_demo, "home_first", lambda *a, **k: called.append("home_first") or True)
    monkeypatch.setattr(press_demo.Runner, "run", lambda self, *a, **k: called.append("run") or [])
    monkeypatch.setattr(sys, "argv", ["press_demo", "--container", CONTAINER, "--bench-world", FLATBOT_WORLD])
    with pytest.raises(SystemExit) as e:
        press_demo.main()
    assert "Nothing moved" in str(e.value)
    assert called == []


class _DescNode:
    def __init__(self, xml):
        self.xml = xml

    def create_subscription(self, _type, topic, cb, _qos):
        assert topic == "/arm/robot_description"
        cb(Namespace(data=self.xml))
        return object()

    def destroy_subscription(self, sub):
        pass


def test_the_description_goes_to_one_fixed_file_not_a_temp_file_per_run(tmp_path):
    xml = Path(FLATBOT_URDF).read_text()
    before = set(glob.glob(str(Path(tempfile.gettempdir()) / "tmp*.urdf")))
    a = cli_common._arm_urdf_path(_DescNode(xml), "arm_mount")
    b = cli_common._arm_urdf_path(_DescNode(xml), "arm_mount")
    assert a == b == str(tmp_path / "arm_robot_description.urdf")
    assert Path(a).read_text() == xml
    assert set(glob.glob(str(Path(tempfile.gettempdir()) / "tmp*.urdf"))) == before
    assert ArmChain(root="arm_mount", path=a).joint_names[0] == "arm_joint_1"
