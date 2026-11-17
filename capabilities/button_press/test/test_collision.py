"""collision.py: the microwave and counter boxes from the panel frame, cuRobo's world (pushed, refused, put
back) and the clearance check of a leg on an arm model with collision meshes."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from button_press import Abort
from button_press import collision as col
from button_press.panel_frame import panel_frame
from button_press.scene_finder import FACE_CORNER_TO_START_MM, FRONT_H_M, FRONT_W_M

BUTTON = np.array([0.70, -0.30, 0.15])  # start_30s, in arm_mount
N_OUT = np.array([-1.0, 0.2, 0.0]) / np.linalg.norm([-1.0, 0.2, 0.0])  # out of the panel, toward the arm


def box_frame(b):
    """(centre, 3x3 axes as columns, half dims) of a box dict."""
    return (
        np.asarray(b["position"]),
        Rotation.from_euler("xyz", b["rpy_deg"], degrees=True).as_matrix(),
        np.asarray(b["dims"]) / 2,
    )


def local(b, p):
    c, Rb, _ = box_frame(b)
    return Rb.T @ (np.asarray(p) - c)


def inside(b, p, tol=1e-6):
    return bool(np.all(np.abs(local(b, p)) <= box_frame(b)[2] + tol))


def boxes(target="start_30s", origin=BUTTON, n=N_OUT):
    return {b["name"]: b for b in col.microwave_boxes(*panel_frame(origin, n), target)}


# ---- the boxes ---------------------------------------------------------------------------------


def test_the_body_starts_at_the_button_plane_and_goes_back():
    b = boxes()["microwave"]
    c, Rb, h = box_frame(b)
    assert np.allclose(Rb[:, 0], N_OUT)  # box x along the (level) normal
    assert np.isclose(local(b, BUTTON)[0], h[0], atol=1e-4)  # the button is on the box's front face
    assert np.isclose(2 * h[0], col.MICROWAVE_DEPTH_M)
    assert inside(b, BUTTON - 0.01 * N_OUT) and not inside(b, BUTTON + 0.002 * N_OUT)


def test_the_front_is_placed_from_the_buttons_corner_offset():
    out, right, up = col.level_axes(N_OUT)
    dx, dy = np.asarray(FACE_CORNER_TO_START_MM) / 1000.0
    corner = BUTTON - dx * right - dy * up  # bottom-right of the front
    b = boxes()["microwave"]
    pad = col.FRONT_PAD_M
    assert inside(b, corner - 0.01 * out)
    assert inside(b, corner - FRONT_W_M * right + FRONT_H_M * up - 0.01 * out)  # top-left
    assert not inside(b, corner + (pad + 0.002) * right - 0.01 * out)  # past the right edge
    assert not inside(b, corner + (FRONT_H_M + pad + 0.002) * up - 0.01 * out)  # above the top


@pytest.mark.parametrize("target", ["power_level", "timer_clock", "stop_eco", "start_30s"])
def test_every_target_gives_the_same_microwave(target):
    """The same microwave seen through any of its buttons: the boxes don't move with the target."""
    from button_press.scene_finder import BUTTON_FROM_DISPLAY_MM

    out, right, up = col.level_axes(N_OUT)
    rel = (np.asarray(BUTTON_FROM_DISPLAY_MM[target]) - np.asarray(BUTTON_FROM_DISPLAY_MM["start_30s"])) / 1000.0
    moved = boxes(target, BUTTON + rel[0] * right + rel[1] * up)
    for name, b in boxes().items():
        assert np.allclose(moved[name]["position"], b["position"], atol=1e-4)


def test_the_counter_is_under_the_front_and_reaches_toward_the_arm():
    out, right, up = col.level_axes(N_OUT)
    dy = FACE_CORNER_TO_START_MM[1] / 1000.0
    counter = boxes()["counter"]
    c, Rb, h = box_frame(counter)
    top = c[2] + h[2]
    assert np.isclose(top, BUTTON[2] - dy - col.FEET_M, atol=1e-4)
    assert np.isclose(
        local(counter, BUTTON)[0], h[0] - col.COUNTER_OUT_M, atol=1e-4
    )  # COUNTER_OUT_M in front of the face


def test_a_tilted_normal_still_gives_upright_boxes():
    for b in boxes(n=N_OUT + np.array([0.0, 0.0, 0.3])).values():
        assert b["rpy_deg"][:2] == [0.0, 0.0]


def test_a_vertical_normal_is_refused():
    with pytest.raises(Abort, match="vertical"):
        col.level_axes([0.05, 0.0, 1.0])


# ---- dense ---------------------------------------------------------------------------------------


def test_dense_steps_at_most_one_degree_the_short_way_round():
    a = np.zeros(7)
    b = np.array([np.radians(179), 0.3, np.radians(-179), 0.0, 0.0, 0.0, 0.0])
    b[0], b[2] = np.radians(179), np.radians(-179)
    a[0], a[2] = np.radians(-179), np.radians(179)  # J1/J3: 2 deg apart across +-180
    qs = col.dense([a, b])
    steps = [np.max(np.abs(col._short_delta(x, y))) for x, y in zip(qs[:-1], qs[1:], strict=True)]
    assert max(steps) <= np.radians(col.LEG_STEP_DEG) + 1e-9
    assert len(qs) - 1 == int(np.ceil(0.3 / np.radians(col.LEG_STEP_DEG)))  # J2's 0.3 rad sets it, not J1/J3


# ---- cuRobo's world ------------------------------------------------------------------------------


class FakeArm:
    def __init__(self, ok=True, kw=True):
        self.calls, self.ok, self.kw = [], ok, kw

    def set_world(self, world, **kwargs):
        if kwargs and not self.kw:
            raise TypeError("set_world() got an unexpected keyword argument")
        self.calls.append((world, kwargs))
        return self.ok, "" if self.ok else "bad world"


def test_push_writes_the_scene_and_loads_it_by_the_planners_path(tmp_path):
    arm = FakeArm()
    w = col.PlannerWorld(arm, tmp_path, "/planner/worlds", ["/config/floor.yaml", ""], "world.yaml")
    bx = list(boxes().values())
    w.push(bx, "scene estimate")
    scene = yaml.safe_load((tmp_path / col.SCENE_NAME).read_text())
    assert scene == {"base_frame": "arm_mount", "layers": ["/config/floor.yaml"], "obstacles": bx}
    assert arm.calls == [(f"/planner/worlds/{col.SCENE_NAME}", {})]
    w.push(bx, "again")  # the same boxes: not sent again
    assert len(arm.calls) == 1
    w.restore()
    assert arm.calls[-1] == ("world.yaml", {"even_if_cancelled": True})
    w.restore()  # nothing pushed since: nothing sent
    assert len(arm.calls) == 2


def test_restore_without_even_if_cancelled_falls_back(tmp_path):
    arm = FakeArm(kw=False)
    w = col.PlannerWorld(arm, tmp_path)
    w.push(list(boxes().values()), "x")
    w.restore()
    assert arm.calls[-1] == ("world.yaml", {})


def test_a_refused_world_aborts_and_is_not_restored(tmp_path):
    arm = FakeArm(ok=False)
    w = col.PlannerWorld(arm, tmp_path)
    with pytest.raises(Abort, match="refused"):
        w.push(list(boxes().values()), "x")
    assert w.loaded is None
    w.restore()
    assert len(arm.calls) == 1


def test_a_relative_world_dir_is_refused(tmp_path):
    with pytest.raises(Abort, match="not absolute"):
        col.PlannerWorld(FakeArm(), "worlds").push(list(boxes().values()), "x")


# ---- the clearance check, on the Cornell arm model (the test image's /fixtures) ------------------

CORNELL = Path("/fixtures/cornell/robot.urdf")
GOLDENS = Path(__file__).resolve().parents[3] / "tests" / "arm" / "fixtures" / "sheppy_goldens.json"


@pytest.fixture
def cornell():
    if not CORNELL.is_file():
        pytest.skip("no /fixtures/cornell (make test-arm's image)")
    from capability_engine import sim

    meta = json.loads(GOLDENS.read_text())["meta"]
    pose = meta["robot_base_pose"]
    sim.reset_for_tests()
    s = sim.make_sim(
        str(CORNELL),
        base_pose=sim.Pose(tuple(pose["position"]), tuple(pose["orientation_xyzw"])),
        ee_link=meta["ee_link"],
        initial_joints=meta["initial_joints"],
    )
    yield sim, s
    sim.reset_for_tests()


def test_a_leg_into_a_box_is_refused_and_one_clear_of_it_passes(cornell):
    sim, (scene, rb) = cornell
    q = np.array([0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57])
    tool, _ = sim.tool_pose(scene, rb, q)
    tool = np.asarray(tool)
    far = {"name": "far", "position": (tool + [0.0, 0.0, 2.0]).tolist(), "rpy_deg": [0, 0, 0], "dims": [0.1] * 3}
    hit = {"name": "hit", "position": tool.tolist(), "rpy_deg": [0, 0, 0], "dims": [0.05] * 3}
    c = col.Clearance([far], sim_robot=(scene, rb))
    try:
        d, _ = c.check_leg("clear", [q, q + 0.05])
        assert d > col.CLEAR_MIN_M
    finally:
        c.close()
    c = col.Clearance([hit], sim_robot=(scene, rb))
    try:
        with pytest.raises(Abort, match="hit"):
            c.check_leg("into it", [q, q + 0.05])
    finally:
        c.close()
    assert c.bodies == []


def test_a_model_that_disagrees_with_the_gating_sim_is_refused(cornell):
    sim, (scene, rb) = cornell
    q = np.zeros(7)
    p_sim, q_sim = sim.tool_pose(scene, rb, q)
    c = col.Clearance([], sim_robot=(scene, rb))
    c.check_model(SimpleNamespace(fk=lambda q: (np.asarray(p_sim), np.asarray(q_sim))), q)  # agrees
    with pytest.raises(Abort, match="driver's model"):
        c.check_model(SimpleNamespace(fk=lambda q: (np.asarray(p_sim) + 0.01, np.asarray(q_sim))), q)
