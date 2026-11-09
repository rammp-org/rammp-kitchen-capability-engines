"""go_home's flow, with a stand-in arm: what is planned, what is sent, and how it fails."""

# isort: off
import home_arm_testlib  # noqa: F401  (first: puts home_arm and capability_engine on sys.path)
# isort: on

import functools
from types import SimpleNamespace

import numpy as np
import pytest
from home_arm import node
from home_arm.node import Capability

from capability_engine import EngineError

HOME = np.array(node.DEFAULT_HOME)
START = np.array([0.3, -0.9, -2.6, -1.8, -0.4, -0.7, 1.9])


def _plan(goal, ok=True, end=None):
    if not ok:
        return SimpleNamespace(
            success=False, message="MotionGenStatus.FINETUNE_TRAJOPT_FAIL", t=[], q=[], qd=[]
        )
    end = np.asarray(goal if end is None else end, float)
    return SimpleNamespace(
        success=True,
        message="",
        t=np.array([0.0, 2.0]),
        q=np.array([end, end]),
        qd=np.zeros((2, 7)),
    )


class Arm:
    """plan_to_joints answers from `answers` (callables of (goal, start) -> plan, in order, the last
    repeating); execute_joint_trajectory records what is sent and answers `codes` in order."""

    def __init__(self, joints=START, answers=(lambda g, s: _plan(g),), codes=(0,)):
        self.joints, self.answers, self.codes = (
            np.asarray(joints, float),
            list(answers),
            list(codes),
        )
        self.planned, self.sent = [], []

    def raw_joints(self):
        return self.joints

    def plan_to_joints(self, goal, start):
        self.planned.append((np.asarray(goal, float), np.asarray(start, float)))
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return answer(np.asarray(goal, float), np.asarray(start, float))

    def execute_joint_trajectory(self, names, points):
        self.sent.append(list(points))
        code = self.codes.pop(0) if len(self.codes) > 1 else self.codes[0]
        return SimpleNamespace(error_code=code, error_string="" if code == 0 else "ABORTED")


class Ctx:
    def __init__(self):
        self.reports = []

    def progress(self, phase, pct, message="", pausable=True):
        self.reports.append((phase, pct, pausable))

    def checkpoint(self):
        pass


class Model:
    """A body model stand-in: `inside` (depth of the start, m) and the escape it offers."""

    def __init__(self, inside=-0.05, escape=None, why=""):
        self.inside, self._escape, self.why = inside, escape, why

    def depth(self, Q):
        Q = np.atleast_2d(Q)
        return np.array([self.inside if np.allclose(q, START) else -0.05 for q in Q])

    def escape(self, q0):
        return self._escape, self.why

    def path_refusal(self, path):
        return ""


LIMITS = [
    (None, None),
    (-2.24, 2.24),
    (None, None),
    (-2.57, 2.57),
    (None, None),
    (-2.09, 2.09),
    (None, None),
]


class Chain:
    """The URDF's limits, and the tool's height: `drop` m lower per rad of joint 2 travel."""

    def __init__(self, drop=0.0):
        self.limits, self.drop = LIMITS, drop

    def tool_z(self, Q, link):
        Q = np.atleast_2d(Q)
        return 0.5 - self.drop * np.abs(Q[:, 1] - Q[0, 1])


def _engine(arm, model=None, chain=None, **params):
    values = {
        "home_joints": list(HOME),
        "body_layers": "x",
        "plan_tries": 3,
        "tool_link": "arm_tool_frame",
    }
    values.update(params)
    eng = SimpleNamespace(
        arm=arm,
        mock=False,
        get_parameter=lambda name: SimpleNamespace(value=values[name]),
        get_logger=lambda: SimpleNamespace(info=lambda *a: None, warning=lambda *a: None),
        _kinematics=lambda: (chain or Chain(), model),
        _judge=Capability._judge,
    )
    eng._plan_home = functools.partial(Capability._plan_home, eng)
    eng._off_the_limit = functools.partial(Capability._off_the_limit, eng)
    eng._plan = functools.partial(Capability._plan, eng)
    eng._fly = functools.partial(Capability._fly, eng)
    return eng


def _go(eng, **params):
    return Capability.execute(eng, "go_home", params, Ctx())


def test_home_is_planned_from_where_the_arm_stands_then_sent_once():
    arm = Arm()
    out = _go(_engine(arm, Model()))
    assert out["moved"] and out["planned"] == "directly" and not out["stepped_out"]
    assert len(arm.planned) == 1 and np.allclose(arm.planned[0][1], START)
    assert len(arm.sent) == 1


def test_a_dry_run_plans_and_sends_nothing():
    arm = Arm()
    out = _go(_engine(arm, Model()), dry_run=True)
    assert out["dry_run"] and not out["moved"] and arm.sent == []


def test_an_arm_already_home_is_left_alone():
    arm = Arm(joints=HOME + 0.005)
    out = _go(_engine(arm, Model()))
    assert out == {"home": True, "moved": False, "off_rad": 0.005}
    assert arm.planned == [] and arm.sent == []


def test_home_is_asked_for_again_and_a_plan_ending_elsewhere_counts_as_failed():
    other = START.copy()
    arm = Arm(
        answers=[
            lambda g, s: _plan(g, ok=False),
            lambda g, s: _plan(g, end=other),
            lambda g, s: _plan(g),
        ]
    )
    out = _go(_engine(arm, Model()))
    assert out["planned"] == "directly, at try 3" and len(arm.planned) == 3 and len(arm.sent) == 1


def test_where_home_never_plans_directly_it_goes_through_a_waypoint_both_legs_in_one_goal():
    def answer(goal, start):
        direct = np.allclose(start, START) and np.allclose(goal[[1, 3, 5]], HOME[[1, 3, 5]])
        return _plan(goal, ok=not direct)

    arm = Arm(answers=[answer])
    out = _go(_engine(arm, Model()))
    assert out["planned"] == "through a waypoint 50% of the way"
    w = arm.planned[3][0]
    assert w == pytest.approx((START + node.homing.on_branch(HOME, START)) / 2)
    assert len(arm.sent) == 1 and len(arm.sent[0]) == 3  # start, waypoint, home: one goal


def test_where_nothing_plans_it_fails_before_anything_moves():
    arm = Arm(answers=[lambda g, s: _plan(g, ok=False)])
    with pytest.raises(EngineError) as e:
        _go(_engine(arm, Model()))
    assert e.value.error_code == "no_way_home" and "jog the arm clear by hand" in str(e.value)
    assert arm.sent == []
    assert len(arm.planned) == 6  # 3 direct, then each waypoint's first leg


def test_a_start_in_the_padding_steps_out_first_and_home_is_planned_from_the_steps_end():
    end = START + np.array([0.0, 0.1, 0.0, 0.05, 0.0, 0.0, 0.0])
    path = np.array([START, (START + end) / 2, end])
    arm = Arm()
    out = _go(_engine(arm, Model(inside=0.012, escape=path)))
    assert out["stepped_out"] and out["moved"]
    assert np.allclose(arm.planned[0][1], end)  # planned from where the step ends
    assert len(arm.sent) == 2  # the step, then home
    assert np.allclose(arm.sent[0][-1][1], end) and np.allclose(arm.sent[0][0][2], 0)


def test_no_way_out_of_the_padding_fails_before_anything_moves():
    arm = Arm()
    with pytest.raises(EngineError) as e:
        _go(_engine(arm, Model(inside=0.03, escape=None, why="every way out goes deeper")))
    assert e.value.error_code == "start_in_padding" and arm.planned == [] and arm.sent == []


def test_a_goal_the_driver_does_not_complete_fails_the_action_and_nothing_more_is_sent():
    path = np.array([START, START + 0.05])
    arm = Arm(codes=[-9])
    with pytest.raises(EngineError) as e:
        _go(_engine(arm, Model(inside=0.01, escape=path)))
    assert e.value.error_code == "motion_failed" and len(arm.sent) == 1


def test_bad_params_are_refused():
    with pytest.raises(EngineError) as e:
        _go(_engine(Arm(), Model()), dry_run="yes")
    assert e.value.error_code == "bad_params" and e.value.recoverable
    with pytest.raises(EngineError) as e:
        _go(_engine(Arm(), Model()), speed=2)
    assert e.value.error_code == "bad_params"
    with pytest.raises(EngineError) as e:
        _go(_engine(Arm(), Model(), home_joints=[0.0] * 6))
    assert e.value.error_code == "bad_config"


def test_a_joint_too_near_its_limit_steps_off_it_along_the_joint_line_and_home_plans_from_there():
    near = START.copy()
    near[5] = 2.09 - 0.094  # joint 6, 0.094 rad from its limit: the planner refuses it as a start
    arm = Arm(joints=near)
    out = _go(_engine(arm, Model()))
    assert out["planned"] == "from the end of the step off the limit" and out["stepped_out"]
    step_end = arm.planned[0][1]
    assert 2.09 - step_end[5] >= node.homing.LIMIT_FIRST_RAD - 1e-9  # clear of the margin
    assert len(arm.sent) == 2 and np.allclose(arm.sent[0][-1][1], step_end)


def test_a_step_off_the_limit_that_would_lower_the_tool_is_refused_before_anything_moves():
    near = START.copy()
    near[5] = 2.09 - 0.094
    arm = Arm(joints=near)
    with pytest.raises(EngineError) as e:
        _go(_engine(arm, Model(), chain=Chain(drop=1.0)))
    assert e.value.error_code == "near_limit" and "lowers the tool" in str(e.value)
    assert arm.sent == []


def test_planning_stops_at_its_budget_and_fails_before_anything_moves(monkeypatch):
    clock = iter([0.0] + [node.PLAN_BUDGET_S + 1.0] * 20)
    monkeypatch.setattr(node.time, "monotonic", lambda: next(clock))
    arm = Arm(answers=[lambda g, s: _plan(g, ok=False)])
    with pytest.raises(EngineError) as e:
        _go(_engine(arm, Model()))
    assert e.value.error_code == "no_way_home" and "60 s of planning" in str(e.value)
    assert arm.planned == [] and arm.sent == []
