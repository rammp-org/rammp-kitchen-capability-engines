"""The engine's direct UI-bridge panel (ctx.update_user_panel): the goal it opens, coalesces and
closes, on an engine with stubbed ROS clients. No ROS graph: the clients are mocks."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from capability_engine import engine as engine_module
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


def _answer_update(node, call, accepted, message=""):
    """The bridge answers update call `call` (an index into call_async's calls)."""
    future = Mock()
    future.result.return_value = SimpleNamespace(accepted=accepted, message=message)
    node._ui_update_client.call_async.return_value.add_done_callback.call_args_list[call][0][0](
        future
    )


@pytest.fixture
def opened(engine):
    """A run whose panel goal the bridge has accepted; retries run when `later` is drained."""
    later = []
    engine._ui_later = lambda delay_s, callback: later.append(callback)
    engine._ui_request_client.server_is_ready.return_value = True
    engine._ui_update_client.service_is_ready.return_value = True
    run = _Run("act", None)
    assert _update(engine, run, "running")
    goal = _accept(engine, run)  # sends the flush update (call 0)
    return engine, run, goal, later


def _drain(later):
    while later:
        later.pop(0)()


def test_an_update_refused_just_after_acceptance_is_resent_to_the_same_goal(opened):
    # The bridge accepts a goal before recording it as active, so the flush right after
    # acceptance is refused although the goal is open (seen on snorlax 2026-11-20).
    engine, run, goal, later = opened
    _answer_update(engine, 0, accepted=False, message="no active RequestUserInput goal")
    _drain(later)
    assert run.ui_goal_handle is goal
    engine._ui_request_client.send_goal_async.assert_called_once()  # no second goal
    assert engine._ui_update_client.call_async.call_count == 2
    _answer_update(engine, 1, accepted=True)
    assert _update(engine, run, "done")  # the next panel goes to the same open goal
    assert engine._ui_update_client.call_async.call_count == 3
    sent = engine._ui_update_client.call_async.call_args[0][0]
    assert json.loads(json.loads(sent.request_json)["panel_json"]) == {"message": "done"}
    engine._ui_request_client.send_goal_async.assert_called_once()
    goal.cancel_goal_async.assert_not_called()


def test_a_refusal_because_the_goal_ended_opens_a_new_goal(opened):
    engine, run, goal, later = opened
    _answer_update(engine, 0, accepted=False, message="no active RequestUserInput goal")
    # the goal's result (a button press) arrives before the retry runs
    result = Mock()
    result.result.return_value.result.result_json = json.dumps(
        {"status": "button", "button_id": "done"}
    )
    goal.get_result_async.return_value.add_done_callback.call_args[0][0](result)
    _drain(later)
    assert engine._ui_request_client.send_goal_async.call_count == 2
    sent = engine._ui_request_client.send_goal_async.call_args[0][0]
    assert json.loads(json.loads(sent.request_json)["panel_json"]) == {"message": "running"}
    assert engine._ui_update_client.call_async.call_count == 1  # nothing resent to the ended goal


def test_a_refusal_that_lasts_cancels_the_goal_and_opens_a_new_one(opened):
    engine, run, goal, later = opened
    for call in range(engine_module.UI_UPDATE_RETRIES):
        _answer_update(engine, call, accepted=False, message="no active RequestUserInput goal")
        _drain(later)
    goal.cancel_goal_async.assert_called_once()
    assert engine._ui_request_client.send_goal_async.call_count == 2
    assert engine._ui_update_client.call_async.call_count == engine_module.UI_UPDATE_RETRIES
    engine.get_logger.return_value.error.assert_called_once()


def test_no_retry_after_the_run_closes(opened):
    engine, run, goal, later = opened
    _answer_update(engine, 0, accepted=False, message="no active RequestUserInput goal")
    engine._ui_close(run)
    _drain(later)
    assert engine._ui_update_client.call_async.call_count == 1
    engine._ui_request_client.send_goal_async.assert_called_once()
