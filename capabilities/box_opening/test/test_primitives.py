from box_opening.primitives.core import Place, PlanState
from box_opening.runtime.legs import Kind


def state():
    return PlanState(joints=[0.0] * 7, chain=0)


def test_place_sequence_and_release(ctx):
    import pytest

    from box_opening.primitives.core import (
        CARRY_CLEAR_M,
        SETDOWN_OVERDRIVE_M,
        tcp_z,
    )

    c = ctx
    target_z = -0.07 + c.model.lid_dims[2]
    legs, _ = Place([0.45, -0.25, target_z], [0.5, 0.5, 0.5, 0.5]).plan(c, state())
    kinds = [leg.kind for leg in legs]
    assert kinds == [Kind.MOTION, Kind.MOTION, Kind.GRIPPER]
    transit, descend, open_ = legs
    assert descend.guard.trip == "setdown"
    assert open_.chain == descend.chain + 1  # contact breaks the chain
    assert open_.gripper_cmd == 0.0
    # the set-down plans in an interaction world capped under ITS contact
    interaction = [kw for k, kw in c.worlds.pushes if k == "interaction"]
    assert interaction and interaction[-1]["contact_z"] == pytest.approx(target_z)
    # the carried lid hangs below the fingertips, invisible to the
    # planner: the transit hover must clear the container top by a
    # lid-height plus margin (field 2026-08-26: lid clipped the box line)
    carry_floor = c.cpose.xyz[2] + c.model.dims[2] + c.model.lid_dims[2] + CARRY_CLEAR_M
    assert transit.target[1][2] == pytest.approx(tcp_z(carry_floor))
    # success is the TOUCH: the stroke overdrives past nominal surface
    # contact so an exact-height 'arrived' can't slip through untripped
    assert descend.target[1][2] == pytest.approx(target_z - SETDOWN_OVERDRIVE_M)
