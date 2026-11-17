"""The engine's direct UI-bridge panel (ctx.update_user_panel): the goal it opens, coalesces and
closes, on an engine with stubbed ROS clients. No ROS graph: the clients are mocks."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from capability_engine.engine import CapabilityEngine, _Run

TOKEN = bytes([1] * 16)


@pytest.fixture
def engine():
    """A CapabilityEngine without __init__: just the attributes the UI methods read."""
    node = CapabilityEngine.__new__(CapabilityEngine)
    node._assigned_token = TOKEN
    node.get_logger = Mock(return_value=Mock())
    node.get_name = Mock(return_value="test_engine")
    node._ui_request_client = Mock()
    node._ui_update_client = Mock()
    return node


def _update(node, run, message):
    ctx = SimpleNamespace(_run=run)
    return node._ui_update_panel(
        ctx, panel_json={"message": message}, objects=None, selected_instance_id=None
    )


def _accept(node, run):
    """The bridge accepts the goal the last send_goal_async sent: runs _ui_on_goal_response."""
    goal_handle = Mock(accepted=True)
    send_future = Mock()
    send_future.result.return_value = goal_handle
    node._ui_request_client.send_goal_async.return_value.add_done_callback.call_args[0][0](
        send_future
    )
    return goal_handle


def test_the_panel_opens_once_the_bridge_is_back(engine):
    run = _Run("act", None)
    engine._ui_request_client.server_is_ready.return_value = False
    assert not _update(engine, run, "first")
    engine._ui_request_client.server_is_ready.return_value = True
    assert _update(engine, run, "retry")
    engine._ui_request_client.send_goal_async.assert_called_once()


def test_a_closed_run_opens_no_panel(engine):
    run = _Run("act", None)
    engine._ui_request_client.server_is_ready.return_value = True
    engine._ui_close(run)
    assert not _update(engine, run, "too late")
    assert not engine._ui_open_goal(run, TOKEN)
    engine._ui_request_client.send_goal_async.assert_not_called()


def test_an_update_made_while_the_goal_is_accepted_reaches_the_bridge(engine):
    run = _Run("act", None)
    engine._ui_request_client.server_is_ready.return_value = True
    engine._ui_update_client.service_is_ready.return_value = True
    assert _update(engine, run, "first")
    assert _update(engine, run, "second")  # the open is in flight: merged, not sent
    engine._ui_update_client.call_async.assert_not_called()
    _accept(engine, run)
    engine._ui_update_client.call_async.assert_called_once()
    sent = engine._ui_update_client.call_async.call_args[0][0]
    assert json.loads(json.loads(sent.request_json)["panel_json"]) == {"message": "second"}
