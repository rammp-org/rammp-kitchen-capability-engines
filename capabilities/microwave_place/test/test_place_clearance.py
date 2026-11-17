"""Regression checks for microwave placement approach clearance."""

import pytest

from microwave_place import microwave_place_node, place_motion


def test_default_insertion_keeps_container_bottom_25_mm_over_glass():
    floor_z = 0.2
    target = [0.7, -0.1, floor_z + microwave_place_node.PARAMS["above"]]

    assert place_motion.insertion_bottom_clearance(target, floor_z) == pytest.approx(0.025)


def test_insertion_clearance_uses_modeled_container_bottom():
    assert place_motion.insertion_bottom_clearance([0.0, 0.0, 0.3], 0.2) == pytest.approx(
        0.1 - place_motion.BOX_BELOW
    )
