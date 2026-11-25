"""The stationary hook keeps the checked push plan until its full route passes."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[1] / "fridge" / "open_door.py"


class Refused(Exception):
    pass


def setup_hook():
    tree = ast.parse(MODULE.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_plan_hook")
    after = Mock()
    after.plan.return_value = True, "released"
    after.motion.joined.q = np.zeros((3, 7))
    check = Mock()
    scope = dict(
        np=np,
        S=SimpleNamespace(bar_point=Mock(return_value=np.array([0.8, -0.2, 0.4]))),
        HOOK_ANGLE_DEG=94.0,
        HOOK_LAGS_DEG=(45.0, 55.0),
        HOOK_EXIT_LIMIT_MARGIN_DEG=9.0,
        time=SimpleNamespace(monotonic=lambda: 10.0),
        check_open=check,
        Refused=Refused,
        SweepBlocked=Refused,
        pl=SimpleNamespace(OverBudget=Refused),
        say=Mock(),
        swing_goals=Mock(),
        HookBackOff=lambda: after,
        opened=lambda door, angle: dict(door, door_open_deg=angle),
        closed_door=lambda geom, h, p, cand: {"closed_hold_pos": cand["bar"]},
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(MODULE), "exec"), scope)
    model = Mock()
    plan = dict(
        cand=dict(flip=False, h=np.zeros(3), pos=np.zeros(3), quat=np.array([0.0, 0.0, 0.0, 1.0])),
        model=model,
        approach=SimpleNamespace(q_end=np.zeros(7)),
    )
    planner = Mock()
    planner.deadline = None
    planner.limit_margin.return_value = 12.0
    planner.motion.return_value = (
        SimpleNamespace(q_end=np.ones(7), goals=[SimpleNamespace(pos=np.ones(3), quat=np.array([0.0, 0.0, 0.0, 1.0]))]),
        "",
    )
    return scope["_plan_hook"], planner, plan, after, check


def test_hook_uses_the_bar_anchor_and_joins_swing_and_exterior_exit():
    plan_hook, planner, baseline, after, check = setup_hook()
    bar = np.array([0.8, -0.2, 0.4])
    result = plan_hook(planner, baseline, {"n": np.array([-1.0, 0.0, 0.0])})
    assert result["cand"]["hook"]
    np.testing.assert_array_equal(result["door"]["closed_hold_pos"], bar)
    assert result["swing_deg"] == 94.0
    assert result["after"] is after
    assert planner.join.call_count == 2
    check.assert_called_once_with(baseline["model"], planner.world, (96.0,), None)
    assert planner.deadline is None
    assert "hook" not in baseline["cand"]


def test_hook_retains_the_baseline_if_the_exterior_exit_fails():
    plan_hook, planner, baseline, after, _ = setup_hook()
    after.plan.return_value = False, "blocked"
    assert plan_hook(planner, baseline, {"n": np.array([-1.0, 0.0, 0.0])}) is None
    planner.join.assert_not_called()
    assert "hook" not in baseline["cand"]


def test_hook_declines_when_withdrawal_has_no_tracking_margin():
    plan_hook, planner, baseline, _, _ = setup_hook()
    planner.limit_margin.return_value = 8.0
    assert plan_hook(planner, baseline, {"n": np.array([-1.0, 0.0, 0.0])}) is None


def test_hook_refuses_if_the_door_sweep_gate_is_blocked():
    plan_hook, planner, baseline, _, check = setup_hook()
    check.side_effect = Refused("leg sweep blocked")
    assert plan_hook(planner, baseline, {"n": np.array([-1.0, 0.0, 0.0])}) is None
    planner.motion.assert_not_called()
    assert planner.deadline is None


@pytest.mark.parametrize(
    ("swing_ok", "end_closure"),
    [(False, 0.8), (True, 0.0)],
    ids=["driver-stop", "fingers-reopened"],
)
def test_failed_hook_skips_release_and_withdrawal(swing_ok, end_closure):
    from contextlib import nullcontext

    tree = ast.parse(MODULE.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "execute_open")
    push = Mock()
    scope = dict(
        np=np,
        F=SimpleNamespace(GRIP_DURING_SWING=True, PULL_GAINS={}, PULL_PATH_TOL_RAD=0.15, SWING_END_TOL_M=0.03),
        pl=SimpleNamespace(_short_delta=lambda a, b: np.asarray(b) - np.asarray(a)),
        dc=SimpleNamespace(DOOR_FILE=Mock()),
        relocalize=Mock(),
        save_door_geometry=Mock(),
        say=Mock(),
        ArcTrace=lambda arm: nullcontext(),
        HookLoadTrace=lambda arm, door: nullcontext(),
        hand_angle=lambda door, pose: -94.0,
        Refused=Refused,
        P=push,
        GRASP_TRACK_ABORT_M=0.03,
        REPLAN_DEG=1.0,
        HOOK_ANGLE_DEG=94.0,
        HOOK_CLOSURE=0.82,
        HOOK_FORCE=0.15,
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(MODULE), "exec"), scope)

    arm = Mock(spec=["get_state", "gripper"])
    opened = dict(position=np.zeros(7), ee_pos=np.zeros(7), gripper_pos=0.0)
    closed = dict(opened, gripper_pos=0.8)
    arm.get_state.side_effect = [opened, opened, opened, closed, dict(closed, gripper_pos=end_closure)]
    planner = Mock()
    planner.run.side_effect = [True, swing_ok]
    plan = dict(
        cand=dict(pos=np.zeros(3), hook=True),
        model=SimpleNamespace(df=SimpleNamespace(sign=-1)),
        door={},
        hinge_fields={},
        approach=Mock(),
        after=Mock(),
        swing_deg=94.0,
        swing=SimpleNamespace(q0=np.zeros(7), joined=SimpleNamespace(t=[0.0, 5.0])),
    )
    assert not scope["execute_open"](arm, planner, plan, fast=True)
    arm.gripper.assert_called_once_with(0.82, force=0.15)
    push._release.assert_not_called()
    assert planner.run.call_count == 2


def test_hook_preserves_the_callers_deadline_for_the_fallback():
    tree = ast.parse(MODULE.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_select_open_plan"
    )
    planner = SimpleNamespace(deadline=28.0)
    deadlines = []

    def decline_hook(*args):
        deadlines.append(planner.deadline)

    def check_fallback():
        deadlines.append(planner.deadline)
        return "fast"

    scope = dict(
        HOOK_PREFLIGHT_S=12.0,
        time=SimpleNamespace(monotonic=lambda: 10.0),
        _hook_preflight=decline_hook,
        say=Mock(),
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(MODULE), "exec"), scope)
    assert scope["_select_open_plan"](planner, {}, None, None, True, check_fallback) == "fast"
    assert deadlines == [22.0, 28.0]
    assert planner.deadline == 28.0
