"""The microwave's world-state preconditions (microwave.world): which of open / close the
world state lets the UI offer."""

import pytest
from microwave import world

MW = "microwave_1"


def _state(opened=None, doors=None, holding=None):
    objects = {} if opened is None else {MW: {"perception_class": "microwave", "opened": opened}}
    return {"objects": objects, "doors": doors or {}, "gripper": {"holding": holding}}


@pytest.mark.parametrize(
    ("state", "open_ok", "close_ok"),
    [
        (_state(), True, True),  # nobody knows: both offered, each finds the door itself
        (_state(opened=False), True, False),
        (_state(opened=True), False, True),
        (_state(doors={"microwave": "closed"}), True, False),
        (_state(doors={"microwave": "open"}), False, True),
        # the object's own record wins over the doors entry
        (_state(opened=True, doors={"microwave": "closed"}), False, True),
        # holding something: no grasp of the handle; the close may carry a container
        (_state(opened=False, holding="container_2"), False, False),
        (_state(opened=True, holding="container_2"), False, True),
    ],
)
def test_preconditions(state, open_ok, close_ok):
    assert (world.why_not("open", MW, state) == "") is open_ok
    assert (world.why_not("close", MW, state) == "") is close_ok


def test_a_refusal_says_why():
    assert "already open" in world.why_not("open", MW, _state(opened=True))
    assert "already closed" in world.why_not("close", MW, _state(opened=False))
    assert "container_2" in world.why_not("open", MW, _state(holding="container_2"))


def test_an_empty_world_state_refuses_nothing():
    assert world.why_not("open", MW, {}) == world.why_not("close", MW, {}) == ""
