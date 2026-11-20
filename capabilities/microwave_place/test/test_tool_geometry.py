"""Regression checks for the Kinova sensor-puck tool and fingertip transforms."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[3]


def _assignment(tree, name):
    return next(
        statement
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in statement.targets)
    )


def test_tool_flange_and_fingertip_geometry():
    arm_tree = ast.parse((ROOT / "capability_engine/rammp_arm.py").read_text())
    motion_tree = ast.parse((ROOT / "capabilities/microwave_place/microwave_place/place_motion.py").read_text())
    model = next(node for node in motion_tree.body if isinstance(node, ast.ClassDef) and node.name == "Model")
    fk = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "fk")

    sim = SimpleNamespace()
    scope = {"np": np, "Rotation": Rotation, "sim": sim}
    code = ast.Module(
        body=[
            _assignment(arm_tree, "TOOL_Z"),
            _assignment(motion_tree, "FLANGE_TO_TIP"),
            fk,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), "tool_geometry", "exec"), scope)

    assert scope["TOOL_Z"] == 0.134
    assert scope["FLANGE_TO_TIP"] == 0.19355
    np.testing.assert_allclose(scope["FLANGE_TO_TIP"] - scope["TOOL_Z"], 0.05955)

    tool_position = np.array([0.41, -0.22, 0.63])
    instance = SimpleNamespace(scene=None, rb=None, flange_to_tip=scope["FLANGE_TO_TIP"])
    rotations = [
        Rotation.identity(),
        Rotation.from_euler("xy", [35.0, -20.0], degrees=True),
    ]
    for rotation in rotations:
        quat = rotation.as_quat()
        sim.tool_pose = lambda *_: (tool_position, quat)
        tool = scope["fk"](instance, [0.0] * 7, "tool")
        flange = scope["fk"](instance, [0.0] * 7, "flange")
        tip = scope["fk"](instance, [0.0] * 7, "tip")
        tool_z = rotation.as_matrix()[:, 2]

        np.testing.assert_allclose(tool[:3, 3], tool_position)
        np.testing.assert_allclose(flange[:3, 3], tool_position - 0.134 * tool_z)
        np.testing.assert_allclose(tip[:3, 3], tool_position + 0.05955 * tool_z)
