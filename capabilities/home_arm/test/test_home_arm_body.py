"""The planner's chair and user boxes, and the step out of their padding."""

# isort: off
import home_arm_testlib  # noqa: F401  (first: puts home_arm and capability_engine on sys.path)

# isort: on
import numpy as np
import pytest
from home_arm import body
from home_arm_testlib import urdf_text_fixture, write_model  # noqa: F401  (the fixture urdf_text)
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
    (tmp_path / "layer_test.yaml").write_text(
        "obstacles:\n"
        "  - {name: turned, position: [0, 0, 0], dims: [1, 1, 1], rpy_deg: [0, 0, 30]}\n"
    )
    with pytest.raises(body.BodyModelError, match="not an axis-aligned box"):
        body.load(path)


def test_the_padding_is_read_where_the_planner_reads_it(tmp_path):
    slab = [("slab", (0.5, 0.0, 0.3), (0.2, 0.4, 0.06))]
    assert body.load(write_model(tmp_path, slab, padding=0.03)).padding_m == pytest.approx(0.03)
    # rammp-curobo reads planner.world_padding only: one at the top level is not the planner's
    path = write_model(tmp_path, slab)
    path.write_text(path.read_text() + "world_padding: 0.03\n")
    with pytest.raises(body.BodyModelError, match="top level"):
        body.load(path)
    path.write_text("robot: robot.yaml\nlayers: [layer_test.yaml]\nplanner:\n  world_padding: -1\n")
    with pytest.raises(body.BodyModelError, match="not >= 0"):
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


def test_an_arm_inside_a_box_itself_is_not_stepped_out_of(tmp_path, urdf_text):
    probe = _model(tmp_path, urdf_text, [("far", (-2.0, 2.0, 0.2), (0.1, 0.1, 0.1))])
    C = probe._sphere_centres(np.atleast_2d(HOME))[0]
    lo, hi = C.min(axis=0), C.max(axis=0)
    # one box round the whole arm: every sphere deep in the box itself, not in its padding
    box = ("everything", tuple((lo + hi) / 2), tuple(hi - lo + 0.4))
    model = _model(tmp_path, urdf_text, [box])
    path, why = model.escape(HOME)
    assert path is None and "inside a chair or user box itself" in why


class _Slides:
    """A stand-in chain of two links, each a sphere's: link k slides along x by joint k, link b a
    metre over link a (the tool)."""

    root, limits = "base", [(None, None), (None, None)]

    def has(self, link):
        return link in ("a", "b")

    def frames(self, Q, links):
        Q = np.atleast_2d(np.asarray(Q, float))
        out = {}
        for link in links:
            T = np.broadcast_to(np.eye(4), (len(Q), 4, 4)).copy()
            T[:, 0, 3] = Q[:, "ab".index(link)]
            T[:, 2, 3] = float("ab".index(link))
            out[link] = T
        return out

    def tool_z(self, Q, link):
        return self.frames(Q, [link])[link][:, 2, 3]


def _walls(*padded, padding=0.02):
    """A model of two 50 mm spheres (links a and b of _Slides) and the PADDED boxes `padded`
    [(lo, hi)], each a wall across a's line."""
    one = (np.zeros((1, 3)), np.array([0.05]))
    lo = np.array([[x0, -1.0, -0.5] for x0, _ in padded])
    hi = np.array([[x1, 1.0, 0.5] for _, x1 in padded])
    names = [f"wall{k}" for k in range(len(padded))]
    model = body.BodyModel("test", names, lo, hi, {"a": one, "b": one}, padding)
    return model.place(_Slides(), "a")


def test_where_every_way_out_goes_deeper_the_step_is_refused_saying_why():
    # sphere a between two walls, 12 mm into the padding of each: either way is deeper into one
    model = _walls((-1.0, -0.038), (0.038, 1.0))
    assert model.depth([0.0, 0.0])[0] == pytest.approx(0.012)
    path, why = model.escape([0.0, 0.0])
    assert path is None
    assert "inside the padding round the chair and user" in why and "goes deeper" in why


def test_no_plan_free_step_from_a_start_that_stands_in_a_box_itself():
    """ "Never deeper" holds the deepest sphere alone: from a start deeper than the padding another
    link could be taken as deep, into the box. From a start in the padding no link can (every
    sphere stands at most as deep as the deepest). So only a start in the padding is stepped out
    of, and a start in a box itself is refused, for the step out and for any other plan-free
    path."""
    wall = (0.038, 1.0)  # padded: its own face 20 mm further, at 0.058
    model = _walls(wall)
    assert model.depth([0.0, 0.0])[0] == pytest.approx(0.012)  # a 12 mm into the padding
    path, why = model.escape([0.0, 0.0])
    assert path is not None and why == "" and model.depth(path[-1])[0] <= -body.ESCAPE_OUT_M
    assert model.in_a_box(0.012) == "" and model.in_a_box(0.02) == ""
    deep = [0.018, 0.0]  # a 30 mm into the padded wall: 10 mm into the wall itself
    assert model.depth(deep)[0] == pytest.approx(0.030)
    path, why = model.escape(deep)
    assert path is None and "10 mm inside a chair or user box itself" in why
    # ... and no other plan-free path either, even one that only backs out
    out = np.array([deep, [0.008, 0.0], [-0.03, 0.0]])
    assert "inside a chair or user box itself" in model.path_refusal(out)
    assert model.path_refusal(out[1:]) == ""  # from 20 mm in, the padding's edge: allowed


def test_a_path_from_outside_must_not_enter_the_padding_and_one_from_inside_not_go_deeper(
    tmp_path, urdf_text
):
    model = _hand_in_a_slab(tmp_path, urdf_text, below_mm=8.0)  # HOME 12 mm into the padding
    path, _ = model.escape(HOME)
    assert model.path_refusal(path) == ""  # out, never deeper
    assert "into the padding" in model.path_refusal(path[::-1])  # back in from outside: refused
