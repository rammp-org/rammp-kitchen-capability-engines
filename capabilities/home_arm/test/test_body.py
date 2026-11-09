"""The planner's chair and user boxes, and the step out of their padding."""

import numpy as np
import pytest
from conftest import write_model
from home_arm import body
from rammp_arm import JOINTS

HOME = [0.4425, -1.5718, -2.9033, -2.1048, -0.7412, -0.9194, 2.1631]


def _model(tmp_path, urdf_text, boxes, **kw):
    model = body.load(write_model(tmp_path, boxes, **kw))
    chain = body.Chain(urdf_text, model.base_link, JOINTS)
    return model.place(chain, "arm_tool_frame")


def test_the_boxes_are_padded_as_the_planner_pads_them_and_the_base_is_left_out(tmp_path):
    model = body.load(write_model(tmp_path, [("slab", (0.5, 0.0, 0.3), (0.2, 0.4, 0.06))]))
    assert model.padding_m == body.WORLD_PADDING_M
    assert model.lo[0] == pytest.approx([0.38, -0.22, 0.25])
    assert model.hi[0] == pytest.approx([0.62, 0.22, 0.35])
    assert "arm_base_link" not in model.spheres
    assert model.spheres["arm_forearm_link"][1] == pytest.approx([0.065, 0.055])  # the buffer added
    assert model.base_link == "arm_mount"
    model = body.load(
        write_model(tmp_path, [("slab", (0.5, 0.0, 0.3), (0.2, 0.4, 0.06))], padding=0.0)
    )
    assert model.lo[0] == pytest.approx([0.4, -0.2, 0.27])


def test_files_that_cannot_be_read_or_place_nothing_are_errors_never_skips(tmp_path):
    with pytest.raises(body.BodyModelError, match="cannot read"):
        body.load(tmp_path / "planner.yaml")
    (tmp_path / "planner.yaml").write_text("robot: robot.yaml\n")
    with pytest.raises(body.BodyModelError, match="names no layers"):
        body.load(tmp_path / "planner.yaml")
    path = write_model(tmp_path, [])
    with pytest.raises(body.BodyModelError, match="hold no obstacle"):
        body.load(path)
    (tmp_path / "layer_test.yaml").write_text(
        "obstacles:\n"
        "  - {name: tilted, position: [0, 0, 0], dims: [1, 1, 1], pose: [0, 0, 0, 1, 0, 0, 0]}\n"
    )
    with pytest.raises(body.BodyModelError, match="not an axis-aligned box"):
        body.load(path)


def test_the_urdf_must_hold_the_arm_joints_and_the_sphere_links(tmp_path, urdf_text):
    with pytest.raises(body.BodyModelError, match="no revolute or continuous joints"):
        body.Chain(urdf_text, "arm_mount", JOINTS + ["arm_joint_99"])
    model = body.load(write_model(tmp_path, [("slab", (0.5, 0.0, 0.3), (0.2, 0.4, 0.06))]))
    model.spheres["arm_no_such_link"] = model.spheres["arm_forearm_link"]
    with pytest.raises(body.BodyModelError, match="no path from arm_mount"):
        model.place(body.Chain(urdf_text, "arm_mount", JOINTS), "arm_tool_frame")


def test_depth_is_how_far_the_deepest_sphere_stands_in_the_padding(tmp_path, urdf_text):
    model = _model(tmp_path, urdf_text, [("far", (-2.0, 2.0, 0.2), (0.1, 0.1, 0.1))])
    # nothing near: every sphere well outside
    assert model.depth(HOME)[0] < -0.5
    # a box round the forearm's first sphere: that sphere is in it by its radius and the padding
    centre = model._sphere_centres(np.atleast_2d(HOME))[0]
    k = (
        0
        if "arm_half_arm_2_link" not in model._links
        else len(model.spheres["arm_half_arm_2_link"][0])
    )
    model = _model(tmp_path, urdf_text, [("round", tuple(centre[k]), (0.01, 0.01, 0.01))])
    assert model.depth(HOME)[0] == pytest.approx(0.065 + 0.005 + 0.02, abs=1e-6)


def _hand_in_a_slab(tmp_path, urdf_text, below_mm):
    """A slab just under the gripper's lowest sphere at HOME, its top `below_mm` under that sphere's
    bottom (negative: the sphere dips into the slab itself)."""
    probe = _model(tmp_path, urdf_text, [("far", (-2.0, 2.0, 0.2), (0.1, 0.1, 0.1))])
    C = probe._sphere_centres(np.atleast_2d(HOME))[0]
    r = probe._radii
    low = int(np.argmin(C[:, 2] - r))
    top = C[low, 2] - r[low] - below_mm / 1000.0
    box = ("slab", (C[low, 0], C[low, 1], top - 0.03), (0.30, 0.30, 0.06))
    return _model(tmp_path, urdf_text, [box])


def test_a_start_in_the_padding_steps_out_never_deeper_the_tool_held_and_ends_clear(
    tmp_path, urdf_text
):
    model = _hand_in_a_slab(
        tmp_path, urdf_text, below_mm=8.0
    )  # 8 mm over the slab: 12 mm into its padding
    assert model.depth(HOME)[0] == pytest.approx(0.012, abs=1e-6)
    path, why = model.escape(HOME)
    assert path is not None, why
    depth = model.depth(path)
    assert np.all(np.diff(depth) <= 1e-6)  # never deeper
    assert depth[-1] <= -body.ESCAPE_CLEAR_M + 1e-9 or depth[-1] <= -body.ESCAPE_OUT_M
    z = model.tool_z(path)
    assert z.min() >= z[0] - body.ESCAPE_DROP_M - 1e-9  # the tool held
    assert np.abs(path - np.asarray(HOME)).max() <= body.ESCAPE_MAX_RAD + body.ESCAPE_STEP_RAD


def test_a_start_clear_of_the_padding_needs_no_step(tmp_path, urdf_text):
    model = _hand_in_a_slab(tmp_path, urdf_text, below_mm=40.0)
    path, why = model.escape(HOME)
    assert why == "" and len(path) == 1


def test_where_every_way_out_goes_deeper_or_down_the_step_is_refused_saying_why(
    tmp_path, urdf_text
):
    probe = _model(tmp_path, urdf_text, [("far", (-2.0, 2.0, 0.2), (0.1, 0.1, 0.1))])
    C = probe._sphere_centres(np.atleast_2d(HOME))[0]
    lo, hi = C.min(axis=0), C.max(axis=0)
    # one box round the whole arm: no 0.6 rad step gets it out
    box = ("everything", tuple((lo + hi) / 2), tuple(hi - lo + 0.4))
    model = _model(tmp_path, urdf_text, [box])
    path, why = model.escape(HOME)
    assert path is None
    assert "inside the padding round the chair and user" in why and "goes deeper" in why


def test_a_path_from_outside_must_not_enter_the_padding_and_one_from_inside_not_go_deeper(
    tmp_path, urdf_text
):
    model = _hand_in_a_slab(tmp_path, urdf_text, below_mm=8.0)  # HOME 12 mm into the padding
    path, _ = model.escape(HOME)
    assert model.path_refusal(path) == ""  # out, never deeper
    assert "into the padding" in model.path_refusal(path[::-1])  # back in from outside: refused
