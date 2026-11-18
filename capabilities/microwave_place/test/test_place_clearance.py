"""Regression checks for fail-closed microwave insertion clearance."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rammp_arm

from microwave_place import place_motion


class Model:
    def fk(self, _q, link="tool"):
        return np.eye(4)


class FlangeState:
    frame_id = rammp_arm.FRAME
    position = np.zeros(3)
    orientation = np.array([0.0, 0.0, 0.0, 1.0])
    stamp_ns = 1_000_000_000


def joint_state(stamp_ns=1_000_000_000):
    return SimpleNamespace(position=np.zeros(7), stamp_ns=stamp_ns)


def synthetic_geometry():
    surfaces = [
        {"name": role, "role": role, "type": "box", "center_m": [0, 0, 0], "size_m": [0.1, 0.1, 0.1]}
        for role in sorted(place_motion.REQUIRED_SURFACE_ROLES)
    ]
    attachments = [
        {"name": role, "role": role, "type": "box", "center_m": [0, 0, 0], "size_m": [0.1, 0.1, 0.1]}
        for role in sorted(place_motion.REQUIRED_ATTACHMENT_ROLES)
    ]
    return {
        "version": 1,
        "synthetic": True,
        "uncertainty": {name: 0.001 for name in place_motion.UNCERTAINTY_FIELDS},
        "turntable_contact": {"max_gap_m": 0.002, "max_penetration_m": 0.001},
        "robot_collision_links": {"flange": ["synthetic_flange"], "gripper": ["synthetic_gripper"]},
        "surfaces": surfaces,
        "attachments": attachments,
    }


class SyntheticChecker:
    """No robot geometry: deterministic pair distances, explicitly limited to unit tests."""

    def __init__(self, collision=None):
        self.geometry = synthetic_geometry()
        self.collision = collision
        self.calls = []

    def assembly_clearances(self, q, **kwargs):
        self.calls.append(kwargs)
        phase = int(round(float(np.asarray(q)[0])))
        result = [(0.10, "gripper vs cavity", "robot", "surface")]
        if phase in (3, 4):
            result.append((0.0, "container vs turntable", "container", "turntable"))
        if self.collision is not None and phase == self.collision[0]:
            result.append(self.collision[1])
        return result


def synthetic_phases():
    names = ("side shift", "transit", "insertion", "lowering", "release", "retreat")
    return [(name, [np.full(7, index, dtype=float)], 0.5, index < 4, index < 4) for index, name in enumerate(names)]


def test_collision_free_complete_trajectory_validates_all_six_phases():
    checker = SyntheticChecker()
    worst, required = place_motion._collision_path(checker, synthetic_phases(), 0.01)

    assert set(worst) == {"side shift", "transit", "insertion", "lowering", "release", "retreat"}
    assert required == pytest.approx(0.014)
    assert [call["include_outer"] for call in checker.calls] == [True, True, False, False, False, False]


def test_container_rim_contact_is_rejected():
    checker = SyntheticChecker((2, (-0.001, "container vs rim", "container", "rim")))
    with pytest.raises(place_motion.StageFailed, match="insertion: container vs rim"):
        place_motion._collision_path(checker, synthetic_phases(), 0.01)


def test_container_turntable_penetration_is_rejected():
    checker = SyntheticChecker((3, (-0.002, "container vs turntable", "container", "turntable")))
    with pytest.raises(place_motion.StageFailed, match="lowering: container vs turntable"):
        place_motion._collision_path(checker, synthetic_phases(), 0.01)


def test_gripper_interior_wall_collision_is_rejected():
    checker = SyntheticChecker((1, (-0.001, "gripper vs left_wall", "robot", "left_wall")))
    with pytest.raises(place_motion.StageFailed, match="transit: gripper vs left_wall"):
        place_motion._collision_path(checker, synthetic_phases(), 0.01)


def test_invalid_or_missing_geometry_fails_closed(tmp_path):
    with pytest.raises(place_motion.InsertionCollisionModelUnavailable, match="not configured"):
        place_motion.load_collision_geometry("")
    with pytest.raises(place_motion.InsertionCollisionModelUnavailable, match="does not exist"):
        place_motion.load_collision_geometry(tmp_path / "missing.yaml")
    with pytest.raises(place_motion.InsertionCollisionModelUnavailable, match="marked synthetic"):
        place_motion.validate_collision_geometry(synthetic_geometry())
    assert place_motion.validate_collision_geometry(synthetic_geometry(), allow_synthetic=True)["synthetic"]


def test_real_snorlax_inventory_names_verified_links_and_precise_missing_measurements():
    import yaml

    path = Path(__file__).parents[1] / "config/snorlax_collision_geometry.yaml"
    geometry = yaml.safe_load(path.read_text())

    assert geometry["robot_collision_links"]["flange"] == []
    assert geometry["robot_collision_links"]["gripper"] == [
        "arm_robotiq_85_base_link",
        "arm_robotiq_85_left_knuckle_link",
        "arm_robotiq_85_right_knuckle_link",
        "arm_robotiq_85_left_finger_link",
        "arm_robotiq_85_right_finger_link",
        "arm_robotiq_85_left_inner_knuckle_link",
        "arm_robotiq_85_right_inner_knuckle_link",
        "arm_robotiq_85_left_finger_tip_link",
        "arm_robotiq_85_right_finger_tip_link",
    ]
    with pytest.raises(place_motion.InsertionCollisionModelUnavailable) as error:
        place_motion.load_collision_geometry(path)
    for field in geometry["missing_measurements"]:
        assert field in str(error.value)


def test_empty_urdf_flange_list_requires_measured_attachment():
    geometry = synthetic_geometry()
    geometry["robot_collision_links"]["flange"] = []
    with pytest.raises(place_motion.InsertionCollisionModelUnavailable, match="measured primitive with role 'flange'"):
        place_motion.validate_collision_geometry(geometry, allow_synthetic=True)

    geometry["attachments"].append(
        {
            "name": "measured_flange",
            "role": "flange",
            "type": "box",
            "center_m": [0, 0, 0],
            "size_m": [0.1, 0.1, 0.1],
        }
    )
    assert (
        place_motion.validate_collision_geometry(geometry, allow_synthetic=True)["robot_collision_links"]["flange"]
        == []
    )


def _validation_inputs(stamp_ns=1_000_000_000):
    q = np.zeros(7)
    checker = SyntheticChecker()
    state = SimpleNamespace(position=q, stamp_ns=stamp_ns)
    flange = SimpleNamespace(stamp_ns=stamp_ns)
    cand = {"shift": [q]}
    plan = SimpleNamespace(q=np.asarray([q]))
    res = {
        "line": [np.full(7, 2.0)],
        "descent": (np.array([0.2]), [np.full(7, 3.0)]),
        "floor_z": 0.15,
        "tilt_line": [np.full(7, 4.0)],
        "back_out": [np.full(7, 5.0)],
    }
    cfg = {"collision_state_max_age_s": 0.25, "clear_min": 0.01}
    return checker, state, flange, cand, plan, res, cfg


def test_stale_robot_state_refuses_collision_authorization(monkeypatch):
    checker, state, flange, cand, plan, res, cfg = _validation_inputs()
    monkeypatch.setattr(place_motion, "check_flange_model", lambda *_args: None)
    with pytest.raises(place_motion.Refused, match="robot state is stale"):
        place_motion.validate_place_collision(
            Model(), checker, np.zeros(7), cand, plan, res, cfg, state, flange, 1_300_000_001
        )


def test_trajectory_state_mismatch_refuses_collision_authorization(monkeypatch):
    checker, state, flange, cand, plan, res, cfg = _validation_inputs()
    state.position = np.radians(np.full(7, 0.31))
    monkeypatch.setattr(place_motion, "check_flange_model", lambda *_args: None)
    with pytest.raises(place_motion.Refused, match="trajectory/state mismatch"):
        place_motion.validate_place_collision(
            Model(), checker, np.zeros(7), cand, plan, res, cfg, state, flange, 1_100_000_000
        )


def test_place_refuses_without_full_assembly_collision_geometry():
    with pytest.raises(place_motion.InsertionCollisionModelUnavailable) as error:
        place_motion.require_full_assembly_insertion_model(None)

    message = str(error.value)
    assert message.startswith("place execution is disabled")
    assert "current trajectory, robot state and full assembly" in message


def test_plan_place_in_remains_available_for_kinematic_dry_run(monkeypatch):
    q0 = np.zeros(7)
    model = Model()
    cand = {
        "pre": np.zeros(3),
        "target": np.array([0.1, 0.0, 0.3]),
        "heading": np.array([1.0, 0.0, 0.0]),
        "insert_m": 0.1,
    }
    monkeypatch.setattr(place_motion, "check_line", lambda *args, **kwargs: None)
    monkeypatch.setattr(place_motion, "plan_line", lambda *args, **kwargs: [q0, q0])
    monkeypatch.setattr(
        place_motion,
        "check_path",
        lambda *args, **kwargs: ({"transit": (0.02, "arm vs outer box")}, 0.0),
    )
    monkeypatch.setattr(place_motion, "check_tilt", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(place_motion, "end_pose_error", lambda *args: (0.0, 0.0))
    monkeypatch.setattr(
        place_motion,
        "plan_down",
        lambda *args, **kwargs: ((np.array([0.3, 0.2]), [q0, q0]), [q0], 0),
    )
    monkeypatch.setattr(place_motion, "contact_index", lambda *args: 0)
    monkeypatch.setattr(place_motion, "plan_tilt", lambda *args: [q0])
    monkeypatch.setattr(place_motion, "plan_level_back_out", lambda *args: [q0])

    res = place_motion.plan_place_in(model, object(), q0, cand, [q0, q0], 0.2, {"clear_min": 0.01, "tilt_max_deg": 2.0})

    assert res["transit_outer_box_clearance"]["transit"] == (0.02, "arm vs outer box")


def test_dry_run_summary_distinguishes_estimate_from_unvalidated_clearance():
    report = place_motion.insertion_collision_report([0.0, 0.0, 0.3], 0.2, {"transit": (0.02, "arm vs outer box")})

    assert report["kinematic_status"] == "FEASIBLE"
    assert report["insertion_collision_status"] == "UNVALIDATED"
    assert report["insertion_bottom_clearance_cm"] == pytest.approx(5.0)
    assert report["transit_outer_box_clearance_cm"] == {"transit": [2.0, "arm vs outer box"]}
    assert report["insertion_collision_missing_geometry"] == list(place_motion.INSERTION_COLLISION_GEOMETRY_REQUIRED)


def test_saved_plan_uses_same_unvalidated_summary_keys():
    class PlanModel:
        def fk(self, _q, link="tool"):
            transform = np.eye(4)
            transform[:3, :3] = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
            return transform

    q0 = np.zeros(7)
    cand = {
        "target": np.array([0.1, 0.0, 0.3]),
        "pre": np.array([0.0, 0.0, 0.3]),
        "heading": np.array([1.0, 0.0, 0.0]),
        "offset_deg": 0.0,
        "shift_cm": 3.0,
        "shift": [q0],
        "quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "insert_m": 0.1,
    }
    loc = dict(point=[0.0, 0.0, 0.0], frames=3, detected=3, spread_m=0.0, spread_deg=0.0, fit={})
    oven = SimpleNamespace(point=np.zeros(3), n=np.array([1.0, 0.0, 0.0]), floor_z=0.2)
    plan = SimpleNamespace(q=np.array([q0, q0]), t=np.array([0.0, 1.0]))
    res = dict(
        line=[q0, q0],
        descent=(np.array([0.3, 0.2]), [q0, q0]),
        transit_outer_box_clearance={"transit": (0.02, "arm vs outer box")},
        tilt=0.0,
        end=(0.0, 0.0),
    )

    summary, record = place_motion.summarize(
        PlanModel(),
        q0,
        loc,
        oven,
        "world.yaml",
        {},
        [],
        cand,
        plan,
        res,
        [],
        {"insert_speed": 0.16},
    )

    assert summary["kinematic_status"] == "FEASIBLE"
    assert summary["insertion_collision_status"] == "UNVALIDATED"
    assert summary["insertion_bottom_clearance_cm"] == pytest.approx(5.0)
    assert record["summary"] == summary


def test_execution_guard_precedes_any_arm_access():
    with pytest.raises(place_motion.InsertionCollisionModelUnavailable):
        place_motion.run_place(
            arm=None,
            model=None,
            q0=None,
            oven=None,
            cand=None,
            plan=None,
            cfg=None,
            ctx=None,
            log=None,
            descents_dir=None,
            res=None,
            validation=None,
        )


def test_execution_rechecks_collision_bound_start_tolerance(monkeypatch):
    class Arm:
        def not_ready(self):
            return None

        def joint_states(self):
            return SimpleNamespace(position=np.radians(np.full(7, 0.31)))

        def ee_state(self):
            return object()

    validation = place_motion.CollisionValidation("token", (1, 1), 0.01, {})
    monkeypatch.setattr(place_motion, "check_flange_model", lambda *_args: None)
    with pytest.raises(place_motion.Refused, match=r"0\.31 deg.*max 0\.3"):
        place_motion.run_place(
            Arm(),
            Model(),
            np.zeros(7),
            SimpleNamespace(floor_z=0.0),
            {"heading": np.array([1.0, 0.0, 0.0])},
            SimpleNamespace(q=np.zeros((2, 7))),
            {},
            None,
            None,
            None,
            {},
            validation,
        )


def test_insertion_bottom_clearance_is_only_the_modeled_box_estimate():
    assert place_motion.insertion_bottom_clearance([0.0, 0.0, 0.3], 0.2) == pytest.approx(0.1 - place_motion.BOX_BELOW)


def test_approach_stop_indices_are_exactly_the_two_segment_joins():
    shift = [np.zeros(7), np.ones(7)]
    transit = np.array([np.ones(7), np.full(7, 2.0), np.full(7, 3.0)])
    line = [transit[-1], np.full(7, 4.0)]

    knots = place_motion.approach_knots(shift, transit, line)

    assert place_motion.approach_stop_indices(shift, transit, line) == (1, 3)
    assert knots[0] == pytest.approx(shift[-1])
    assert knots[2] == pytest.approx(transit[-1])


def test_approach_trajectory_stops_at_both_segment_joins_without_blending():
    directions = (
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([np.cos(np.deg2rad(72.7)), np.sin(np.deg2rad(72.7))]),
        np.array([0.0, 1.0]),
        np.array([np.cos(np.deg2rad(150.4)), np.sin(np.deg2rad(150.4))]),
    )
    path_2d = np.vstack(([0.0, 0.0], np.cumsum(directions, axis=0)))
    path = np.pad(path_2d, ((0, 0), (0, 5)))
    stops = (2, 4)

    _, global_positions, global_velocities = rammp_arm.joint_trajectory(path[0], path[1:], speed_scale=0.5)
    for stop in stops:
        index = int(np.argmin(np.linalg.norm(global_positions - path[stop], axis=1)))
        assert np.linalg.norm(global_velocities[index + 1] - global_velocities[index - 1]) > 0.1

    times, positions, velocities = rammp_arm.joint_trajectory(path[0], path[1:], speed_scale=0.5, stop_at=stops)

    accelerations = np.gradient(velocities, times, axis=0)
    for stop in stops:
        index = int(np.argmin(np.linalg.norm(positions - path[stop], axis=1)))
        assert positions[index] == pytest.approx(path[stop])
        assert velocities[index] == pytest.approx(np.zeros(7), abs=1e-12)
        assert accelerations[index] == pytest.approx(np.zeros(7), abs=0.1)

    assert positions[-1] == pytest.approx(path[-1])
    assert velocities[-1] == pytest.approx(np.zeros(7), abs=1e-12)

    for q in positions:
        distances = []
        for start, end in zip(path[:-1], path[1:], strict=True):
            delta = end - start
            fraction = np.clip((q - start) @ delta / (delta @ delta), 0.0, 1.0)
            distances.append(np.linalg.norm(q - (start + fraction * delta)))
        assert min(distances) < 1e-10


def test_joint_trajectory_rejects_non_interior_or_unsorted_stops():
    with pytest.raises(ValueError, match="interior knots"):
        rammp_arm.joint_trajectory(np.zeros(7), [np.ones(7), np.full(7, 2.0)], stop_at=(2,))

    with pytest.raises(ValueError, match="unique, increasing"):
        rammp_arm.joint_trajectory(
            np.zeros(7),
            [np.ones(7), np.full(7, 2.0), np.full(7, 3.0)],
            stop_at=(2, 1),
        )


@pytest.mark.parametrize("tool_z", [0.120, 0.134])
def test_flange_fk_uses_active_urdf_parent_link_not_tool_z(tmp_path, monkeypatch, tool_z):
    import pybullet as p

    urdf = f"""<robot name="test_arm">
      <link name="arm_mount"/>
      <link name="arm_end_effector_link"/>
      <joint name="arm_end_effector_fixed" type="fixed">
        <parent link="arm_mount"/><child link="arm_end_effector_link"/>
      </joint>
      <link name="arm_tool_frame"/>
      <joint name="arm_tool_frame_joint" type="fixed">
        <origin xyz="0 0 {tool_z}" rpy="0 0 0"/>
        <parent link="arm_end_effector_link"/><child link="arm_tool_frame"/>
      </joint>
    </robot>"""
    path = tmp_path / "driver.urdf"
    path.write_text(urdf)
    client = p.connect(p.DIRECT)
    robot = p.loadURDF(str(path), useFixedBase=True, physicsClientId=client)
    model = place_motion.Model.__new__(place_motion.Model)
    model.scene = place_motion.sim.Scene(place_motion.sim.Pose((0, 0, 0), (0, 0, 0, 1)))
    model.rb = place_motion.sim.SimRobot(robot, 1, client)
    model._flange_link_id = place_motion._flange_link_index(p, model.rb)
    model.flange_to_tip = place_motion.FLANGE_TO_TIP
    monkeypatch.setattr(place_motion.sim, "ARM", [])

    try:
        assert model.fk([], link="tool")[2, 3] == pytest.approx(tool_z)
        assert model.fk([], link="flange")[2, 3] == pytest.approx(0.0)
        assert model.fk([], link="tip")[2, 3] == pytest.approx(place_motion.FLANGE_TO_TIP)
    finally:
        p.disconnect(physicsClientId=client)


class DriverDescriptionBullet:
    JOINT_FIXED = 4

    def __init__(self, parent=b"arm_end_effector_link", fixed=True):
        tool_joint = [0] * 17
        tool_joint[1] = b"arm_tool_frame_joint"
        tool_joint[2] = self.JOINT_FIXED if fixed else 0
        tool_joint[12] = b"arm_tool_frame"
        tool_joint[16] = 0
        flange_joint = [0] * 17
        flange_joint[12] = parent
        self.joints = [flange_joint, tool_joint]

    def getNumJoints(self, *_args, **_kwargs):
        return len(self.joints)

    def getJointInfo(self, _robot_id, index, **_kwargs):
        return self.joints[index]

    def getBodyInfo(self, *_args, **_kwargs):
        return (b"robot_root", b"robot")


def test_flange_link_resolves_from_confirmed_driver_description():
    bullet = DriverDescriptionBullet()
    robot = SimpleNamespace(robot_id=1, physics_client_id=0)
    assert place_motion._flange_link_index(bullet, robot) == 0


@pytest.mark.parametrize(
    ("bullet", "message"),
    [
        (DriverDescriptionBullet(parent=b"wrong_link"), "parent must be arm_end_effector_link"),
        (DriverDescriptionBullet(fixed=False), "must be attached by fixed"),
    ],
)
def test_flange_link_refuses_incorrect_driver_frame_definition(bullet, message):
    with pytest.raises(place_motion.Refused, match=message):
        place_motion._flange_link_index(bullet, SimpleNamespace(robot_id=1, physics_client_id=0))


def test_flange_pose_must_match_synchronized_driver_state():
    measured = FlangeState()
    position_error, orientation_error = place_motion.check_flange_model(Model(), joint_state(), measured)
    assert position_error == pytest.approx(0.0)
    assert orientation_error == pytest.approx(0.0)

    measured.position = np.array([0.01, 0.0, 0.0])
    with pytest.raises(place_motion.Refused, match="URDF FK mismatch"):
        place_motion.check_flange_model(Model(), joint_state(), measured)


def test_flange_check_fails_closed_for_unsynchronized_or_missing_timestamps():
    measured = FlangeState()
    with pytest.raises(place_motion.Refused, match="timestamps differ"):
        place_motion.check_flange_model(Model(), joint_state(measured.stamp_ns + 10_000_001), measured)

    with pytest.raises(place_motion.Refused, match="no valid ROS timestamp"):
        place_motion.check_flange_model(Model(), joint_state(0), measured)


def test_flange_check_fails_closed_for_wrong_ee_frame():
    measured = FlangeState()
    measured.frame_id = "arm_base_link"
    with pytest.raises(place_motion.Refused, match="not 'arm_mount'"):
        place_motion.check_flange_model(Model(), joint_state(), measured)
