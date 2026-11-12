"""The client hands a guard the same /joint_states message once per callback
it spins (~1 kHz feedback against ~46 Hz joint states): the verdicts must be
the same, but gravity is evaluated once per message and the trip report's
run-up holds messages, not copies."""

from box_opening.runtime.guards import CompensatedGuard, TorqueGuard


def test_a_repeated_message_costs_one_gravity_evaluation_and_one_runup_row():
    calls = []

    def gravity(q):
        calls.append(list(q))
        return [0.1 * q[3], 0.0, 0.0, 0.0]

    g = CompensatedGuard(3.0, gravity, reference=([0.0] * 4, [0.0] * 7))
    g.on_progress(0.1)
    q = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    for _ in range(20):  # one message, polled 20 times
        assert not g.on_efforts([0.1, 0.0, 0.0, 0.0], list(q))
    assert len(calls) == 2  # the reference, then this message once
    q2 = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]
    hits = [g.on_efforts([3.5, 0.0, 0.0, 0.0], list(q2)) for _ in range(5)]
    assert hits == [True] * 5  # the verdict on a repeat is the verdict on the message
    assert len(calls) == 3
    rep = g.trip_report()
    assert len(rep["recent"]) == 2 and rep["dev_nm"] == 3.3


def test_a_raw_guard_keeps_each_message_once_and_judges_every_poll():
    g = TorqueGuard(3.0, arm_after=0.5)
    g.on_progress(0.1)
    assert not g.on_efforts([0.0] * 4)  # the baseline
    for p in (0.2, 0.3, 0.6):  # one over-threshold message, polled across arm_after
        g.on_progress(p)
        tripped = g.on_efforts([4.0, 0.0, 0.0, 0.0])
    assert tripped  # polled once armed, it trips as it always did
    assert len(g.trip_report()["recent"]) == 2
