"""A deferred gripper leg sent under the motion before it once its gate
says so (Leg.send_when; Runner._gated): the grip straight from the press
opens the fingers on the way up once the pads have left the knob
(press_contact.PadsAbove), not as the rise begins — owner 2026-11-02: "it
opens the grippers too fast and makes the fingers slide along the lid"."""

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import Q0, Q1, FakeClient
from conftest import leg as make_leg
from conftest import runner as make_runner

from box_opening.runtime.legs import Kind
from box_opening.tasks import press_contact as pc


def _open(gate):
    lg = make_leg("grip:open", kind=Kind.GRIPPER, cmd=0.2)
    lg.defer_join, lg.send_with_previous_motion, lg.send_when = True, True, gate
    return lg


def _watched(client):
    """The order the client was asked to execute and to move the fingers."""
    said, execute, send = [], client.execute, client.gripper_send

    def ex(*a, **k):
        said.append("rise sent")
        out = execute(*a, **k)
        said.append("rise ended")
        return out

    def gs(cmd):
        said.append("open %.1f" % cmd)
        return send(cmd)

    client.execute, client.gripper_send = ex, gs
    return said


def test_a_gated_open_goes_out_under_the_rise_once_its_gate_says_so(tmp_path):
    client = FakeClient()
    said = _watched(client)
    asked = []
    rise = make_leg("grip:rise", start=Q0, end=Q1)
    runner = make_runner(client, tmp_path)
    res = runner.run([rise, _open(lambda progress: asked.append(progress) or True)], execute=True)
    # sent under the rise, joined once it ended (the run's next leg), before anything after it
    assert [(r.leg_name, r.ok) for r in res] == [("grip:rise", True), ("grip:open", True)]
    assert said == ["rise sent", "open 0.2", "rise ended"] and asked == [0.5]
    assert rise.stop_when is None  # the rise's own stop_when put back
    assert runner.join_pending() is None


def test_a_gate_that_never_says_so_sends_the_open_after_the_rise(tmp_path):
    client = FakeClient()
    said = _watched(client)
    runner = make_runner(client, tmp_path)
    runner.run([make_leg("grip:rise", start=Q0, end=Q1), _open(lambda progress: False)], execute=True)
    assert said == ["rise sent", "rise ended", "open 0.2"]
    assert runner.join_pending().ok


def test_a_gate_that_fails_sends_the_open_after_the_rise_and_is_not_asked_again(tmp_path, capsys):
    client = FakeClient()
    said = _watched(client)
    asked = []

    def broken(progress):
        asked.append(progress)
        raise ValueError("no chain")

    real = client.execute

    def twice(traj_, speed, stop_when=None, **kw):  # the real client polls it at every spin
        stop_when(0.3)
        return real(traj_, speed, stop_when=stop_when, **kw)

    client.execute = twice
    runner = make_runner(client, tmp_path)
    runner.run([make_leg("grip:rise", start=Q0, end=Q1), _open(broken)], execute=True)
    assert said[-1] == "open 0.2" and asked == [0.3]
    assert "grip:open: its gate failed (no chain)" in capsys.readouterr().out


def test_a_gate_rides_with_the_rises_own_stop_when(tmp_path):
    """The rise's own stop_when is still asked at every poll, after the
    gate, and still stops it."""
    client = FakeClient()
    _watched(client)
    own = []
    rise = make_leg("grip:rise", start=Q0, end=Q1)
    rise.stop_when = lambda progress: own.append(progress) or False
    make_runner(client, tmp_path).run([rise, _open(lambda progress: True)], execute=True)
    assert own == [0.5] and rise.stop_when is not None and rise.stop_when.__name__ == "<lambda>"


class _Chain:
    """Joint 1 is the tool frame's height; nothing else moves."""

    def fk(self, q):
        return np.eye(3), np.array([0.0, 0.0, float(q[0])])


def test_pads_above_says_so_once_the_measured_pads_stand_at_its_height():
    pad = pc.pads_z(_Chain(), [0.0] * 7)  # the pads below the tool frame
    info = {"sent_at": 10.0, "sent_traj": object(), "feedback_q": []}
    client = SimpleNamespace(last_execution=info)
    gate = pc.PadsAbove(_Chain(), client, 0.312)
    assert gate(0.0) is False  # nothing heard yet
    for t, z in ((10.1, 0.300), (10.2, 0.306), (10.3, 0.311)):
        info["feedback_q"].append((t, 0.1, [z - pad] + [0.0] * 6, None))
        assert gate(0.1) is False
    assert gate(0.1) is False  # no new sample: no kinematics spent
    info["feedback_q"].append((10.4, 0.2, [0.3125 - pad] + [0.0] * 6, None))
    assert gate(0.2) is True and gate.at == (pytest.approx(0.4), pytest.approx(0.3125))
    assert gate.highest == pytest.approx(0.3125)
    # a new execution is watched afresh
    client.last_execution = {
        "sent_at": 20.0,
        "sent_traj": object(),
        "feedback_q": [(20.1, 0.1, [0.30 - pad] + [0.0] * 6, None)],
    }
    assert gate(0.1) is False and gate.at is None


def test_pads_above_with_no_execution_says_nothing():
    assert pc.PadsAbove(_Chain(), SimpleNamespace(), 0.3)(0.5) is False
    assert pc.PadsAbove(_Chain(), SimpleNamespace(last_execution={"sent_traj": None}), 0.3)(0.5) is False
