"""The CLIs' runtime wiring: nothing reaches ROS off Cyclone, and only
--execute arms the client (the driver has no dry-run gate of its own).

rclpy is replaced because these assertions are about what happens before
and around it — a real init would join a live graph."""

import pytest

from box_opening.tasks import cli_common


class _Rclpy:
    def __init__(self):
        self.inited = False

    def init(self, **_kw):
        self.inited = True

    def create_node(self, name):
        return object()


class _Client:
    def __init__(self, node, abort=None, motion_enabled=False):
        self.armed = motion_enabled


@pytest.fixture
def ros(monkeypatch):
    fake = _Rclpy()
    monkeypatch.setattr(cli_common, "rclpy", fake)
    monkeypatch.setattr(cli_common, "install_sigint", lambda flag: None)
    monkeypatch.setattr(cli_common, "PlannerClient", _Client)
    return fake


def test_only_execute_arms_the_client(ros, monkeypatch):
    monkeypatch.setenv("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
    _node, dry = cli_common.init_runtime(execute=False)
    _node, live = cli_common.init_runtime(execute=True)
    assert dry.armed is False
    assert live.armed is True


def test_a_shell_not_on_cyclone_never_reaches_ros(ros, monkeypatch):
    monkeypatch.setenv("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    with pytest.raises(SystemExit):
        cli_common.init_runtime(execute=True)
    assert not ros.inited
