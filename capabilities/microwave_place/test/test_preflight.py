"""Offline checks for collision-config inventory and preflight reporting."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from microwave_place import place_motion, preflight


def urdf(*links):
    body = "".join(
        f'<link name="{name}"><collision><origin xyz="0 0 0"/><geometry>'
        f'<mesh filename="{name}.stl"/></geometry></collision></link>'
        for name in links
    )
    return f'<robot name="test">{body}</robot>'


def complete_geometry():
    surfaces = [
        {"name": role, "role": role, "type": "box", "center_m": [0, 0, 0], "size_m": [0.1, 0.1, 0.1]}
        for role in sorted(place_motion.REQUIRED_SURFACE_ROLES)
    ]
    attachments = [
        {"name": role, "role": role, "type": "box", "center_m": [0, 0, 0], "size_m": [0.1, 0.1, 0.1]}
        for role in [*sorted(place_motion.REQUIRED_ATTACHMENT_ROLES), "flange"]
    ]
    return {
        "version": 1,
        "missing_measurements": [],
        "uncertainty": {name: 0.001 for name in place_motion.UNCERTAINTY_FIELDS},
        "turntable_contact": {"max_gap_m": 0.002, "max_penetration_m": 0.001},
        "robot_collision_links": {"flange": [], "gripper": ["gripper"]},
        "surfaces": surfaces,
        "attachments": attachments,
    }


def test_active_urdf_inventory_preserves_collision_shape_source_and_origin():
    inventory = preflight.urdf_collision_inventory(urdf("gripper"))

    assert inventory == [
        {
            "link": "gripper",
            "collision_shapes": [
                {
                    "type": "mesh",
                    "origin_xyz": ["0", "0", "0"],
                    "origin_rpy": ["0", "0", "0"],
                    "filename": "gripper.stl",
                }
            ],
        }
    ]


def test_active_urdf_inventory_accepts_xml_namespaces():
    namespaced = urdf("gripper").replace("<robot ", '<robot xmlns="urn:robot" ')

    assert preflight.urdf_collision_inventory(namespaced)[0]["link"] == "gripper"


def test_collision_report_accepts_complete_config_and_active_urdf(tmp_path):
    path = tmp_path / "geometry.yaml"
    path.write_text(yaml.safe_dump(complete_geometry()))

    report = preflight.collision_configuration_report(path, urdf("gripper"))

    assert report["complete"] is True
    assert report["validation_error"] == ""
    assert report["configured_links_missing_from_active_urdf"] == []


def test_collision_report_keeps_real_missing_measurements_actionable():
    path = Path(__file__).parents[1] / "config/snorlax_collision_geometry.yaml"
    data = yaml.safe_load(path.read_text())
    report = preflight.collision_configuration_report(path, urdf(*data["robot_collision_links"]["gripper"]))

    assert report["complete"] is False
    assert report["missing_measurements"] == data["missing_measurements"]
    assert report["configured_links_missing_from_active_urdf"] == []
    assert "is incomplete; missing measurements" in report["validation_error"]


def test_collision_report_identifies_configured_link_missing_from_urdf(tmp_path):
    path = tmp_path / "geometry.yaml"
    path.write_text(yaml.safe_dump(complete_geometry()))

    report = preflight.collision_configuration_report(path, urdf("some_other_link"))

    assert report["complete"] is False
    assert report["configured_links_missing_from_active_urdf"] == ["gripper"]


def test_diagnostic_faults_reports_error_level_and_nonzero_fault_counter():
    diagnostics = SimpleNamespace(
        status=[
            SimpleNamespace(level=0, name="ok", message="healthy", values={"faults": "0"}),
            SimpleNamespace(level=2, name="driver", message="faulted", values={"faults": "0"}),
            SimpleNamespace(level=0, name="controller", message="latched", values={"faults": "3"}),
        ]
    )

    assert preflight.diagnostic_faults(diagnostics) == [
        "driver: faulted (faults=0)",
        "controller: latched (faults=3)",
    ]


def test_preflight_check_reports_failures_but_propagates_cancel():
    assert preflight.check(lambda: {"ok": True}) == {"status": "PASS", "detail": {"ok": True}}
    failure = preflight.check(lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    assert failure == {"status": "FAIL", "detail": "offline", "error_type": "RuntimeError"}

    Cancelled = type("Cancelled", (Exception,), {})
    with pytest.raises(Cancelled):
        preflight.check(lambda: (_ for _ in ()).throw(Cancelled()))


def test_preflight_report_persists_pass_fail_summary(tmp_path):
    checks = {
        "ros": {"status": "PASS", "detail": {}},
        "geometry": {"status": "FAIL", "detail": "missing"},
    }

    path = preflight.save_report(tmp_path, checks)

    assert path.parent == tmp_path / "preflight"
    assert '"ready": false' in path.read_text()
