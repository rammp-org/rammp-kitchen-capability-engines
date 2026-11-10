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


def _plan(goal, start=None, ok=True, end=None, message="MotionGenStatus.FINETUNE_TRAJOPT_FAIL"):
    """A plan from `start` to `goal` (or `end`), straight, two samples."""
    if not ok:
        return SimpleNamespace(success=False, message=message, t=[], q=[], qd=[])
    end = np.asarray(goal if end is None else end, float)
    start = end if start is None else np.asarray(start, float)
    return SimpleNamespace(
        success=True,
        message="",
        t=np.array([0.0, 2.0]),
        q=np.array([start, end]),
        qd=np.zeros((2, 7)),
    )


def _swinging(goal, start, joint=0, out=1.0):
    """A plan that ends at `goal` but takes `joint` `out` rad past the start on the way."""
    mid = (np.asarray(start, float) + goal) / 2
    mid[joint] = start[joint] + out
    return SimpleNamespace(
        success=True,
        message="",
        t=np.array([0.0, 1.0, 2.0]),
        q=np.array([start, mid, goal]),
        qd=np.zeros((3, 7)),
    )


class Arm:
    """plan_to_joints answers from `answers` (callables of (goal, start) -> plan, in order, the last
    repeating); execute_joint_trajectory records what is sent with its speed_scale and answers
    `codes` in order, a completed goal leaving the arm at its last point. joint_states reads
    `readings` in order (the last repeating)."""

    def __init__(
        self, joints=START, answers=(lambda g, s: _plan(g, s),), codes=(0,), readings=None
    ):
        self.readings = [np.asarray(q, float) for q in (readings or [joints])]
        self.answers, self.codes = list(answers), list(codes)
        self.planned, self.sent, self.speeds = [], [], []

    def joint_states(self):
        q = self.readings.pop(0) if len(self.readings) > 1 else self.readings[0]
        return SimpleNamespace(position=q)

    def plan_to_joints(self, goal, start):
        self.planned.append((np.asarray(goal, float), np.asarray(start, float)))
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return answer(np.asarray(goal, float), np.asarray(start, float))

    def execute_joint_trajectory(self, names, points, speed_scale=1.0):
        self.sent.append(list(points))
        self.speeds.append(speed_scale)
        code = self.codes.pop(0) if len(self.codes) > 1 else self.codes[0]
        if code == 0:
            self.readings = [np.asarray(points[-1][1], float)]
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


def _engine(arm, model=None, chain=None, no_model="", **params):
    values = {
        "home_joints": list(HOME),
        "body_layers": "x",
        "plan_tries": 3,
        "tool_link": "arm_tool_frame",
        "speed_scale": 0.3,
    }
    values.update(params)
    eng = SimpleNamespace(
        arm=arm,
        mock=False,
        get_parameter=lambda name: SimpleNamespace(value=values[name]),
        get_logger=lambda: SimpleNamespace(info=lambda *a: None, warning=lambda *a: None),
        _kinematics=lambda: (chain or Chain(), model, no_model),
        _judge=Capability._judge,
    )
    for name in ("_joints", "_route", "_plan_home", "_off_the_limit", "_plan", "_fly"):
        setattr(eng, name, functools.partial(getattr(Capability, name), eng))
    return eng


@pytest.fixture(autouse=True)
def _plan_timeout(monkeypatch):
    monkeypatch.setattr(node, "_plan_timeout_s", lambda: 30.0)


def _go(eng, **params):
    return Capability.execute(eng, "go_home", params, Ctx())


def test_home_is_planned_from_where_the_arm_stands_then_sent_once_at_its_speed_scale():
    arm = Arm()
    out = _go(_engine(arm, Model()))
    assert out["moved"] and out["planned"] == "directly" and not out["stepped_out"]
    assert len(arm.planned) == 1 and np.allclose(arm.planned[0][1], START)
    assert len(arm.sent) == 1 and arm.speeds == [0.3] and out["speed_scale"] == 0.3
    assert out["duration_s"] == pytest.approx(2.0 / 0.3, abs=0.01)


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
            lambda g, s: _plan(g, s, ok=False),
            lambda g, s: _plan(g, s, end=other),
            lambda g, s: _plan(g, s),
        ]
    )
    out = _go(_engine(arm, Model()))
    assert out["planned"] == "directly, at try 3" and len(arm.planned) == 3 and len(arm.sent) == 1


def _direct(goal, start):
    return np.allclose(start, START) and np.allclose(goal, node.homing.on_branch(HOME, START))


def test_where_home_never_plans_directly_it_goes_through_a_waypoint_both_legs_in_one_goal():
    arm = Arm(answers=[lambda g, s: _plan(g, s, ok=not _direct(g, s))])
    out = _go(_engine(arm, Model()))
    assert out["planned"] == "through a waypoint 50% of the way"
    w = arm.planned[3][0]
    assert w == pytest.approx((START + node.homing.on_branch(HOME, START)) / 2)
    assert len(arm.sent) == 1 and len(arm.sent[0]) == 3  # start, waypoint, home: one goal


def test_a_plan_home_that_swings_wide_is_refused_and_home_goes_through_the_midpoint():
    # flatbot 2026-11-10: every plan home took joint_1 ~0.9 rad out past its start and back, the
    # hand toward the user; asking again got the same plan, the midpoint's legs did not swing
    def answer(goal, start):
        return _swinging(goal, start) if _direct(goal, start) else _plan(goal, start)

    arm = Arm(answers=[answer])
    out = _go(_engine(arm, Model()))
    assert out["planned"] == "through a waypoint 50% of the way"
    assert len(arm.planned) == 3  # asked directly once: the planner answers alike
    assert len(arm.sent) == 1
    j1 = [p[1][0] for p in arm.sent[0]]
    assert max(j1) <= max(START[0], HOME[0]) + 1e-9  # no swing out past the start


def test_where_every_plan_swings_wide_nothing_moves():
    arm = Arm(answers=[lambda g, s: _swinging(g, s)])
    with pytest.raises(EngineError) as e:
        _go(_engine(arm, Model()))
    assert e.value.error_code == "no_way_home" and "swings joint 1" in str(e.value)
    assert arm.sent == []


def test_a_waypoints_second_leg_is_planned_from_where_its_first_leg_ends():
    def answer(goal, start):
        if _direct(goal, start):
            return _plan(goal, start, ok=False)
        if np.allclose(start, START):  # the first leg ends 5 mrad short of the waypoint
            return _plan(goal, start, end=goal + np.r_[0.0, 0.005, 0, 0, 0, 0, 0])
        return _plan(goal, start)

    arm = Arm(answers=[answer])
    out = _go(_engine(arm, Model()))
    assert out["planned"] == "through a waypoint 50% of the way"
    w = arm.planned[3][0]
    assert np.allclose(arm.planned[4][1], w + np.r_[0.0, 0.005, 0, 0, 0, 0, 0])
    q = np.array([p[1] for p in arm.sent[0]])
    assert np.abs(np.diff(q, axis=0)).max() < 1.0  # joined without a jump


def test_where_nothing_plans_it_fails_before_anything_moves():
    arm = Arm(answers=[lambda g, s: _plan(g, s, ok=False)])
    with pytest.raises(EngineError) as e:
        _go(_engine(arm, Model()))
    assert e.value.error_code == "no_way_home" and "jog the arm clear by hand" in str(e.value)
    assert arm.sent == []
    assert len(arm.planned) == 6  # 3 direct, then each waypoint's first leg


def _refuses_start(at, message="MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION"):
    return lambda g, s: _plan(g, s, ok=not np.allclose(s, at), message=message)


def test_a_start_in_the_padding_steps_out_first_and_home_is_planned_from_the_steps_end():
    end = START + np.array([0.0, 0.1, 0.0, 0.05, 0.0, 0.0, 0.0])
    path = np.array([START, (START + end) / 2, end])
    arm = Arm(answers=[_refuses_start(START)])
    out = _go(_engine(arm, Model(inside=0.012, escape=path)))
    assert out["stepped_out"] and out["moved"]
    assert np.allclose(arm.planned[0][1], START)  # the planner is asked first: it refuses
    assert np.allclose(arm.planned[1][1], end)  # then planned from where the step ends
    assert len(arm.sent) == 2  # the step, then home
    assert np.allclose(arm.sent[0][-1][1], end) and np.allclose(arm.sent[0][0][2], 0)
    assert arm.speeds == [1.0, 0.3]  # the step at its own slow timing, home at speed_scale


def test_a_start_the_planner_takes_is_planned_from_even_inside_the_engines_padding():
    # the engine's copy of the boxes says padding, the planner's own world (another profile's,
    # say) does not: the planner is the authority, nothing steps out
    arm = Arm()
    out = _go(_engine(arm, Model(inside=0.012, escape=np.array([START, START + 0.05]))))
    assert out["planned"] == "directly" and not out["stepped_out"] and len(arm.sent) == 1


def test_no_way_out_of_the_padding_fails_before_anything_moves():
    arm = Arm(answers=[_refuses_start(START)])
    with pytest.raises(EngineError) as e:
        _go(_engine(arm, Model(inside=0.03, escape=None, why="every way out goes deeper")))
    assert e.value.error_code == "start_in_padding" and len(arm.planned) == 1 and arm.sent == []


def test_a_step_that_stops_short_fails_before_home_is_sent():
    end = START + 0.05
    arm = Arm(answers=[_refuses_start(START)])
    eng = _engine(arm, Model(inside=0.01, escape=np.array([START, end])))
    fly = eng._fly

    def short(points, what, speed):  # the driver reports success, the arm stands 0.04 short
        fly(points, what, speed)
        arm.readings = [end - 0.04]

    eng._fly = short
    with pytest.raises(EngineError) as e:
        _go(eng)
    assert e.value.error_code == "motion_failed" and "short of its end" in str(e.value)
    assert len(arm.sent) == 1


def test_a_goal_the_driver_does_not_complete_fails_the_action_and_nothing_more_is_sent():
    path = np.array([START, START + 0.05])
    arm = Arm(answers=[_refuses_start(START)], codes=[-9])
    with pytest.raises(EngineError) as e:
        _go(_engine(arm, Model(inside=0.01, escape=path)))
    assert e.value.error_code == "motion_failed" and len(arm.sent) == 1


def test_without_a_body_model_a_start_the_planner_refuses_says_why():
    arm = Arm(answers=[_refuses_start(START)])
    with pytest.raises(EngineError) as e:
        _go(_engine(arm, None, no_model="body model: cannot read x"))
    assert e.value.error_code == "body_model" and "cannot read x" in str(e.value)
    assert arm.sent == []
    arm = Arm()  # ... and a start the planner takes goes home without one
    assert _go(_engine(arm, None, no_model="body model: cannot read x"))["planned"] == "directly"


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
    for bad in (0.0, 1.5):
        with pytest.raises(EngineError) as e:
            _go(_engine(Arm(), Model(), speed_scale=bad))
        assert e.value.error_code == "bad_config"


def test_an_arm_moved_while_planned_for_is_planned_for_again_from_where_it_stands():
    moved = START + np.r_[0.1, 0, 0, 0, 0, 0, 0]
    arm = Arm(readings=[START, moved])
    out = _go(_engine(arm, Model()))
    assert len(arm.planned) == 2 and np.allclose(arm.planned[1][1], moved)
    assert np.allclose(arm.sent[0][0][1], moved) and "planned for again" in out["notes"][0]


def test_an_arm_that_keeps_moving_is_not_sent_anything():
    arm = Arm(readings=[START, START + 0.1, START + 0.2])
    with pytest.raises(EngineError) as e:
        _go(_engine(arm, Model()))
    assert e.value.error_code == "arm_moving" and e.value.recoverable and arm.sent == []


def test_a_joint_too_near_its_limit_steps_off_it_along_the_joint_line_and_home_plans_from_there():
    near = START.copy()
    near[5] = 2.09 - 0.094  # joint 6, 0.094 rad from its limit: the planner refuses it as a start
    arm = Arm(joints=near)
    out = _go(_engine(arm, Model()))
    assert out["planned"] == "from the end of the step off the limit" and out["stepped_out"]
    step_end = arm.planned[0][1]
    assert 2.09 - step_end[5] >= node.homing.LIMIT_FIRST_RAD - 1e-9  # clear of the margin
    assert len(arm.sent) == 2 and np.allclose(arm.sent[0][-1][1], step_end)


def test_off_a_limit_home_is_asked_for_again_before_a_longer_step():
    near = START.copy()
    near[5] = 2.09 - 0.094
    arm = Arm(joints=near, answers=[lambda g, s: _plan(g, s, ok=False), lambda g, s: _plan(g, s)])
    out = _go(_engine(arm, Model()))
    assert np.allclose(arm.planned[0][1], arm.planned[1][1])  # the same, shortest step, twice
    assert "directly, at try 2" in out["notes"][-1]


def test_a_step_off_the_limit_that_would_lower_the_tool_is_refused_before_anything_moves():
    near = START.copy()
    near[5] = 2.09 - 0.094
    arm = Arm(joints=near)
    with pytest.raises(EngineError) as e:
        _go(_engine(arm, Model(), chain=Chain(drop=1.0)))
    assert e.value.error_code == "near_limit" and "lowers the tool" in str(e.value)
    assert arm.sent == []


def test_a_plan_is_asked_for_only_while_a_whole_plan_timeout_fits_in_the_budget(monkeypatch):
    # 31 s in: a plan that waited its whole 30 s would end past the 60 s budget
    clock = iter([0.0] + [31.0] * 40)
    monkeypatch.setattr(node.time, "monotonic", lambda: next(clock))
    arm = Arm(answers=[lambda g, s: _plan(g, s, ok=False)])
    with pytest.raises(EngineError) as e:
        _go(_engine(arm, Model()))
    assert e.value.error_code == "no_way_home" and "60 s of planning" in str(e.value)
    assert arm.planned == [] and arm.sent == []


def test_a_way_home_too_slow_for_the_action_is_refused_before_anything_moves():
    def slow(goal, start):
        plan = _plan(goal, start)
        plan.t = np.array([0.0, 40.0])  # 133 s at speed_scale 0.3
        return plan

    for dry in (False, True):
        arm = Arm(answers=[slow])
        with pytest.raises(EngineError) as e:
            _go(_engine(arm, Model()), dry_run=dry)
        assert e.value.error_code == "too_slow" and arm.sent == []
