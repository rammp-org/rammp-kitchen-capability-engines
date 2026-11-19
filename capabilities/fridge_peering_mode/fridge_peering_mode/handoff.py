"""A `pick` goal sent while `peer` teleops: end peer, then let the pick goal in.

The engine runs one goal at a time: CapabilityEngine._on_goal rejects a goal while another runs.
So that a second `ros2 action send_goal ... action_id: pick` can end peering and pick (no UI, no
orchestrator), node.Capability._on_goal first calls hand_off(), which holds the pick goal's
acceptance while the running peer ends:

  peer teleoperating   the handoff event is set; peer's teleop loop sees it on its next tick,
                       leaves the stream (the arm holds) and succeeds with ended "handed_off"
                       (SAM3's tracking session carries over: the pick's `label` ids stay valid);
  peer not teleoping   (scene homing, homing, waiting for the fridge TF) it is cancelled: a motion
                       under way stops where it is;
  peer picking         (a tap's pick, planning or moving) the pick goal is refused: the running
                       pick is never interrupted by another.

Then the engine's own _on_goal accepts the pick goal, which runs from wherever the arm is. No ROS
here: the engine's lifecycle condition and current run are passed in.
"""

from __future__ import annotations

import json
import time

HANDOFF_WAIT_S = 15.0  # a peer not ended within this: the pick goal is refused (busy)


def requested_label(params_json: str) -> str:
    """The `label` a pick goal's params_json names, or "" (for the log)."""
    try:
        params = json.loads(params_json or "{}")
    except ValueError:
        return ""
    if not isinstance(params, dict):
        return ""
    return str(params.get("label") or "").strip()


def hand_off(lifecycle, current, session, params_json, event, log, wait_s=HANDOFF_WAIT_S) -> bool:
    """End a running peer for a pick goal. `lifecycle`: the engine's threading.Condition (notified
    when a run ends); `current()`: the engine's running run (None: idle); `session`: the running
    peer's PeerSession (or None). True when nothing runs any more (the pick goal can be accepted),
    False when something still does (the engine then rejects it as busy)."""
    with lifecycle:
        run = current()
    if run is None:
        return True
    if run.action_id != "peer":
        log.warning(f"pick goal: {run.action_id} is running; refused")
        return False
    if session is not None and session.picking:
        log.warning("pick goal: peer is picking a tapped container; refused")
        return False
    iid = requested_label(params_json)
    if session is not None:
        session.handoff_request = iid
    event.set()
    if session is None or not session.in_teleop:
        log.warning(
            "pick goal: peer is not teleoperating (homing or waiting for the fridge TF): "
            "cancelling it, the arm stops where it is"
        )
        run.cancelled.set()
    else:
        log.info(f"pick goal ({iid or 'the only food or drink in view'}): ending peer's " f"teleop")
    deadline = time.monotonic() + wait_s
    with lifecycle:
        while current() is run and time.monotonic() < deadline:
            lifecycle.wait(0.05)
        ended = current() is not run
    if not ended:
        log.error(f"pick goal: peer did not end within {wait_s:.0f} s; refused")
    return ended
