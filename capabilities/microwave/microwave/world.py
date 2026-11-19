"""The microwave's world-state preconditions: what the orchestrator's world state (`objects`, `doors`, `gripper`;
rammp-orchestrator core/world_state.py) must say for an action to be offered. Checks only: nothing here writes it.

- open: the door not known open, and the gripper holding nothing (the open grasps the handle);
- close: the door not known closed (the gripper may hold something: the close pushes with the side of the hand, and
  may carry a container).

A door whose state nobody knows (no record, `opened` None) is not a reason to refuse: the open and close find the door
themselves. Used by the engine's availability (REVIEW_RULES O1: fast, no side effects).
"""

DOOR_KEY = "microwave"  # the world state's doors entry, e.g. {"microwave": "closed"}


def door_open(instance_id, world_state):
    """True / False when the world state says the door is open / closed, None when it doesn't know: the object's
    `opened` first (this microwave), else `doors.microwave`."""
    opened = ((world_state.get("objects") or {}).get(instance_id) or {}).get("opened")
    if opened is not None:
        return bool(opened)
    return {"open": True, "closed": False}.get((world_state.get("doors") or {}).get(DOOR_KEY))


def why_not(action_id, instance_id, world_state):
    """Why `action_id` ("open" / "close") can't run on `instance_id` in `world_state`, or ""."""
    holding = (world_state.get("gripper") or {}).get("holding")
    door = door_open(instance_id, world_state)
    if action_id == "open":
        if holding:
            return f"the gripper holds {holding}: empty it before opening the door"
        if door is True:
            return "the microwave door is already open"
    elif action_id == "close" and door is False:
        return "the microwave door is already closed"
    return ""
