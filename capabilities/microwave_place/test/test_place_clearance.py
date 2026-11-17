"""Regression checks for fail-closed microwave insertion clearance."""

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


def test_place_refuses_without_full_assembly_collision_geometry():
    with pytest.raises(place_motion.InsertionCollisionModelUnavailable) as error:
        place_motion.require_full_assembly_insertion_model()

    message = str(error.value)
    for component in ("held container", "wrist-camera", "turntable", "side walls", "ceiling"):
        assert component in message


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
    assert (
        "held-container collision geometry and tool-relative offset" in report["insertion_collision_missing_geometry"]
    )


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
        )


def test_insertion_bottom_clearance_is_only_the_modeled_box_estimate():
    assert place_motion.insertion_bottom_clearance([0.0, 0.0, 0.3], 0.2) == pytest.approx(0.1 - place_motion.BOX_BELOW)


def test_flange_pose_must_match_driver_model_before_motion():
    measured = FlangeState()
    position_error, orientation_error = place_motion.check_flange_model(Model(), [0.0] * 7, measured)
    assert position_error == pytest.approx(0.0)
    assert orientation_error == pytest.approx(0.0)

    measured.position = np.array([0.01, 0.0, 0.0])
    with pytest.raises(place_motion.Refused, match="URDF FK mismatch"):
        place_motion.check_flange_model(Model(), [0.0] * 7, measured)
