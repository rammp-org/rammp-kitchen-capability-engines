"""The computed pre-press spot (press_button.prepress_pose) and check_fingertip's guard."""

import numpy as np
import pytest

pb = pytest.importorskip("button_press.press_button")  # needs rclpy + rammp interfaces
from button_press import Abort  # noqa: E402
from button_press.panel_frame import panel_frame, to_panel  # noqa: E402

ORIGIN, R = panel_frame(np.array([0.6, 0.05, 0.25]), np.array([-1.0, 0.2, 0.1]))


def test_the_spot_is_straight_in_with_the_fingertips_at_the_standoff():
    p, Rt = pb.prepress_pose(ORIGIN, R)
    assert np.allclose(Rt[:, 2], -R[:, 2])  # tool z into the panel, along the normal
    assert np.allclose(Rt @ Rt.T, np.eye(3)) and np.isclose(np.linalg.det(Rt), 1.0)
    tip = p + Rt[:, 2] * pb.FINGERTIP_PAST_TOOL_M
    assert np.allclose(tip, ORIGIN + pb.STANDOFF_M * R[:, 2])  # on the normal through the button
    assert pb.fingertip_out_of_panel(p, Rt, ORIGIN, R) == pytest.approx(pb.STANDOFF_M)
    pb.check_fingertip(p, Rt, ORIGIN, R, "pre-press spot")


def test_aim_moves_the_spot_across_the_panel_only():
    p0, R0 = pb.prepress_pose(ORIGIN, R)
    p, Rt = pb.prepress_pose(ORIGIN, R, aim_mm=(3.0, -5.0))
    assert np.allclose(to_panel(p, ORIGIN, R) - to_panel(p0, ORIGIN, R), [0.003, -0.005, 0.0])
    assert np.allclose(Rt, R0)


def test_staging_is_further_out_along_the_normal():
    p0, _ = pb.prepress_pose(ORIGIN, R)
    p, _ = pb.prepress_pose(ORIGIN, R, out=0.08)
    assert np.allclose(p - p0, 0.08 * R[:, 2])


def test_a_spot_into_the_panel_is_refused():
    _, Rt = pb.prepress_pose(ORIGIN, R)
    p = ORIGIN - 0.01 * R[:, 2] - Rt[:, 2] * pb.FINGERTIP_PAST_TOOL_M  # fingertip 1 cm in
    with pytest.raises(Abort, match="INTO the panel"):
        pb.check_fingertip(p, Rt, ORIGIN, R, "pre-press spot")
