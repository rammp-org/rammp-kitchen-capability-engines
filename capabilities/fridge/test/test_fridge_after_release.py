"""push_open.after_release: what follows the release, from the plan checked before the swing or planned again; offline
(the function loaded through `ast`, with stand-ins for planning)."""

import ast
import types
from pathlib import Path

import numpy as np
import pytest

PUSH_OPEN = Path(__file__).resolve().parents[1] / "fridge" / "push_open.py"


class Refused(Exception):
    pass


def _after_release():
    tree = ast.parse(PUSH_OPEN.read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "after_release"]
    pl = types.SimpleNamespace(START_TOL_DEG=1.0, _short_delta=lambda a, b: np.asarray(b) - np.asarray(a))
    scope = dict(np=np, pl=pl, Refused=Refused)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(PUSH_OPEN), "exec"), scope)
    return scope["after_release"]


class After:
    def __init__(self, q0, ok=True):
        self.motion = types.SimpleNamespace(q0=np.asarray(q0, float))
        self.ok, self.planned = ok, []

    def plan(self, planner, door, q, pose, from_arm=False):
        self.planned.append(door)
        return self.ok, "planned again"


class Arm:
    def get_state(self):
        return {"position": np.zeros(7), "ee_pos": np.zeros(7)}


PLANNER = types.SimpleNamespace(search=lambda: None)


def test_the_arm_at_the_checked_plans_start_runs_it():
    after = After(np.zeros(7))
    assert "checked before the swing" in _after_release()(after, PLANNER, Arm(), {"door_open_deg": 60})
    assert after.planned == []


def test_a_short_swing_plans_again_against_the_door_as_measured():
    """11-20: the swing ran to its end with the door at 54 of 60. The arm may be near the checked plan's start, but
    that plan was for a door at 60: execute_open sets `replan`, and only a new plan against the measured door runs."""
    after = After(np.zeros(7))
    after.replan = True
    door = {"door_open_deg": 54.2}
    assert "planned again" in _after_release()(after, PLANNER, Arm(), door)
    assert after.planned == [door]


def test_a_short_swing_whose_new_plan_fails_refuses():
    after = After(np.zeros(7), ok=False)
    after.replan = True
    with pytest.raises(Refused, match="gripper open"):
        _after_release()(after, PLANNER, Arm(), {"door_open_deg": 54.2})
