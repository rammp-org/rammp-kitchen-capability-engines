"""The level place's hand directions over the spot (node._hand_headings): hand_angles_deg in order, then straight out
from the arm's base unless a listed angle is within 5 deg of it. The method is loaded through `ast` (node.py imports
ROS at module level), as test_place_container_ui_flow.py does."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from place_container import plan

NODE = Path(__file__).resolve().parents[1] / "place_container" / "node.py"


def hand_headings():
    tree = ast.parse(NODE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Capability")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_hand_headings")
    scope = {"np": np, "plan": plan}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(NODE), "exec"), scope)
    return scope["_hand_headings"]


def run(angles, spot):
    return hand_headings()(SimpleNamespace(cfg={"hand_angles_deg": angles}), np.array(spot))


def test_the_listed_angles_come_first_in_order_then_straight_out():
    out = run([-30.0, -15.0, 0.0], [0.80, 0.30])  # straight out from the base: +21 deg
    assert [n for n, _ in out][:3] == ["-30 deg", "-15 deg", "+0 deg"]
    assert out[-1][0].startswith("+21 deg") and "straight out" in out[-1][0]
    np.testing.assert_allclose(out[0][1], [np.cos(np.radians(-30)), np.sin(np.radians(-30)), 0.0])
    for _, h in out:
        assert abs(np.linalg.norm(h) - 1.0) < 1e-9 and h[2] == 0.0


def test_straight_out_is_not_tried_twice():
    out = run([-30.0, -12.0], [0.80, -0.15])  # straight out: -10.6 deg, within 5 of -12
    assert len(out) == 2 and not any("straight out" in n for n, _ in out)


def test_an_empty_list_is_the_old_way_only():
    out = run([], [0.90, -0.091])
    assert len(out) == 1 and "straight out" in out[0][0]
    np.testing.assert_allclose(out[0][1], plan.approach_direction([0.90, -0.091]))
