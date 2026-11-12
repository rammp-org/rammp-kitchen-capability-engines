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
        self.planned, self.sent, self.speeds, self.path_tols = [], [], [], []

    def joint_states(self):
        q = self.readings.pop(0) if len(self.readings) > 1 else self.readings[0]
        return SimpleNamespace(position=q)

    def plan_to_joints(self, goal, start):
        self.planned.append((np.asarray(goal, float), np.asarray(start, float)))
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return answer(np.asarray(goal, float), np.asarray(start, float))

    def execute_joint_trajectory(self, names, points, speed_scale=1.0, path_tolerance=None):
        self.sent.append(list(points))
        self.speeds.append(speed_scale)
        self.path_tols.append(path_tolerance)
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
    for name in (
        "_joints",
        "_route",
        "_plan_home",
        "_to_waypoint",
        "_off_the_limit",
        "_plan",
        "_pause_point",
        "_fits",
        "_fly",
    ):
        setattr(eng, name, functools.partial(getattr(Capability, name), eng))
    return eng


@pytest.fixture(autouse=True)
def _plan_timeout(monkeypatch):
    monkeypatch.setattr(node, "_plan_timeout_s", lambda: 30.0)
    monkeypatch.setattr(node, "_settle_s", lambda: 0.0)  # no waiting for the stand-in's joints


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
    # 3 direct, each waypoint's first leg, then the first leg of the line in 4 and in 6
    assert len(arm.planned) == 8
    assert "nor along it in 4/6 legs" in str(e.value)


def test_a_waypoints_plan_may_end_near_it_and_one_that_ends_on_another_arm_may_not():
    # box_opening's planner rig, 2026-11-12: cuRobo's FK-pose fallback ended 0.08-0.32 rad from the
    # waypoint (the same arm along its self-motion) and was refused; another arm ends 0.9-4.3 off
    off = np.r_[0.2, 0, 0, 0, -0.2, 0, 0.2]

    def answer(goal, start):
        if _direct(goal, start):
            return _plan(goal, start, ok=False)
        if np.allclose(start, START):  # a waypoint's plan
            half = np.allclose(goal, (START + node.homing.on_branch(HOME, START)) / 2)
            return _plan(goal, start, end=goal + (4.5 * off if half else off))
        return _plan(goal, start)

    arm = Arm(answers=[answer])
    out = _go(_engine(arm, Model()))
    assert out["planned"] == "through a waypoint 25% of the way"  # the 50% one ended on another arm
    w = arm.planned[4][0]
    assert np.allclose(arm.planned[5][1], w + off)  # home planned from where the plan ended
    assert np.allclose(arm.planned[5][0], node.homing.on_branch(HOME, w + off))


def _long_legs_swing(longest):
    """Plans every leg, but one whose largest joint moves more than `longest` rad swings joint 1."""

    def answer(goal, start):
        far = float(np.abs(np.asarray(goal, float) - np.asarray(start, float)).max())
        return _swinging(goal, start) if far > longest else _plan(goal, start)

    return answer


@pytest.mark.parametrize("share, n", [(0.30, 4), (0.20, 6)])
def test_where_no_single_waypoint_plans_home_goes_along_the_joint_line_in_short_legs(share, n):
    # an arm whose elbow must straighten and fold the other way: cuRobo swings wide on long legs
    span = float(np.abs(node.homing.on_branch(HOME, START) - START).max())
    arm = Arm(answers=[_long_legs_swing(share * span)])
    out = _go(_engine(arm, Model()))
    assert out["planned"] == f"in {n} legs along the joint line (no single waypoint planned)"
    legs = arm.planned[-n:]
    assert np.allclose(legs[0][1], START) and np.allclose(legs[-1][0], HOME)
    for (goal, _start), (_goal, start) in zip(legs, legs[1:], strict=False):
        assert np.allclose(goal, start)  # each leg from where the one before ended
    assert len(arm.sent) == 1 and len(arm.sent[0]) == n + 1  # one goal through every waypoint


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


def test_params_that_are_no_json_object_are_bad_params_not_an_exception():
    """The base class hands execute() whatever JSON the goal carried (REVIEW_RULES O4)."""
    for bad in (None, [], ["dry_run"], 3, True):
        with pytest.raises(EngineError) as e:
            Capability.execute(_engine(Arm(), Model()), "go_home", bad, Ctx())
        assert e.value.error_code == "bad_params" and e.value.recoverable


def test_an_arm_whose_state_does_not_come_fails_by_name_before_anything_is_planned():
    """self.arm's own errors are not EngineErrors: left alone, the action would fail as a bare
    "exception" (REVIEW_RULES U4)."""
    from capability_engine import StateTimeout

    arm = Arm()

    def no_state():
        raise StateTimeout("no /joint_states, /ee_state or /gripper_state")

    arm.joint_states = no_state
    with pytest.raises(EngineError) as e:
        _go(_engine(arm, Model()))
    assert e.value.error_code == "arm_state" and e.value.recoverable
    assert "no /joint_states" in str(e.value) and arm.planned == [] and arm.sent == []


@pytest.mark.parametrize("error", ["Refused", "EStopped", "ControlTaken"])
def test_a_goal_the_arm_refuses_fails_the_action_as_motion_failed_and_is_not_sent_again(error):
    """A rejected goal, an e-stop or control taken raises from self.arm (REVIEW_RULES C5, U4): the
    action fails by name, and nothing is sent after it."""
    from capability_engine import driver_state

    arm = Arm()
    tried = []

    def refuse(names, points, speed_scale=1.0, path_tolerance=None):
        tried.append(len(points))
        raise getattr(driver_state, error)("execute_joint_trajectory: goal REJECTED")

    arm.execute_joint_trajectory = refuse
    with pytest.raises(EngineError) as e:
        _go(_engine(arm, Model()))
    assert e.value.error_code == "motion_failed" and not e.value.recoverable
    assert "REJECTED" in str(e.value) and len(tried) == 1


def test_the_mock_plans_nothing_and_a_dry_run_of_it_sends_nothing():
    sent = []
    arm = SimpleNamespace(go_to_joint_config=lambda q, label: sent.append(label) or True)
    eng = _engine(arm, Model())
    eng.mock = True
    dry = _go(eng, dry_run=True)
    assert (dry["home"], dry["moved"], dry["mock"], dry["dry_run"]) == (False, False, True, True)
    assert sent == []
    out = _go(eng)
    assert (out["home"], out["moved"], out["mock"]) == (True, False, True) and sent == [
        "home (mock)"
    ]
    # the real path's payload, key for key: what is built on the mock runs on the robot
    real = _go(_engine(Arm(), Model()))
    assert set(real) <= set(out) and set(_go(_engine(Arm(), Model()), dry_run=True)) <= set(dry)


def test_a_planner_that_does_not_answer_is_said_so_and_not_asked_through_every_waypoint():
    """self.arm turns "no action server", "no answer" and "no result in time" into a failed plan
    whose message begins with the call's name: that is no word on the way home (REVIEW_RULES U4)."""
    down = "plan_to_joints: no /rammp_curobo/plan_to_joints action server (the cuRobo planner)"
    arm = Arm(answers=[lambda g, s: _plan(g, s, ok=False, message=down)])
    with pytest.raises(EngineError) as e:
        _go(_engine(arm, Model()))
    assert e.value.error_code == "planner_unavailable" and e.value.recoverable
    assert "no /rammp_curobo/plan_to_joints action server" in str(e.value)
    assert len(arm.planned) == 1 and arm.sent == []
    # ... off a joint limit too, where a failed home is otherwise answered with a longer step
    near = HOME.copy()
    near[1] = LIMITS[1][0] + 0.05
    arm = Arm(joints=near, answers=[lambda g, s: _plan(g, s, ok=False, message=down)])
    with pytest.raises(EngineError) as e:
        _go(_engine(arm, Model()))
    assert e.value.error_code == "planner_unavailable" and len(arm.planned) == 1


def test_a_pause_is_no_part_of_the_planning_or_the_action_budget(monkeypatch):
    """ctx.checkpoint() blocks while the action is paused, and the planning step is pausable: the
    time it took is given back, so a long pause does not end as no_way_home or too_slow."""
    now = [0.0]
    monkeypatch.setattr(node.time, "monotonic", lambda: now[0])

    class Paused(Ctx):
        each = 20.0

        def checkpoint(self):
            now[0] += self.each  # paused this long at every checkpoint

    # 60 s of pauses before the first plan: on the wall clock no whole plan_timeout would fit the
    # 60 s of planning any more. The pause is not planning, and the home is planned and flown.
    arm = Arm()
    out = Capability.execute(_engine(arm, Model()), "go_home", {}, Paused())
    assert out["home"] and out["moved"] and len(arm.sent) == 1


def test_a_pause_that_leaves_no_time_for_the_way_home_sends_nothing_and_says_so(monkeypatch):
    """Whatever cancels a long action counts its pauses (the orchestrator's execute_timeout): a way
    home that no longer fits is refused before anything is sent, not cancelled half way."""
    now = [0.0]
    monkeypatch.setattr(node.time, "monotonic", lambda: now[0])

    class Paused(Ctx):
        def checkpoint(self):
            now[0] += 500.0

    arm = Arm()
    with pytest.raises(EngineError) as e:
        Capability.execute(_engine(arm, Model()), "go_home", {}, Paused())
    assert e.value.error_code == "paused_too_long" and e.value.recoverable and arm.sent == []


def test_inside_the_engines_padding_a_plan_that_fails_past_its_start_is_asked_again_not_stepped():
    """Only a start the planner refuses is stepped out of: a plan that failed for another reason,
    swung or ended elsewhere is a planner that takes this start."""
    path = np.array([START, START + 0.05])
    for first in (
        lambda g, s: _plan(g, s, ok=False),  # FINETUNE_TRAJOPT_FAIL: the planner took the start
        lambda g, s: _swinging(g, s),
        lambda g, s: _plan(g, s, end=g + 0.4),
    ):
        arm = Arm(answers=[first, lambda g, s: _plan(g, s)])
        out = _go(_engine(arm, Model(inside=0.012, escape=path)))
        assert not out["stepped_out"] and len(arm.sent) == 1
        assert all(np.allclose(s, START) for _g, s in arm.planned[:2])
    # ... and a start it refuses for its limit margin is stepped out of, as one in collision is
    arm = Arm(answers=[_refuses_start(START, "START_IN_LIMIT_MARGIN: jog it away from the limit")])
    assert _go(_engine(arm, Model(inside=0.012, escape=path)))["stepped_out"]


def test_every_goal_is_sent_under_the_drivers_divergence_guard(monkeypatch):
    """A goal that names no path_tolerance runs with the driver's guard off (its executor checks a
    joint only where the tolerance is > 0): a blocked arm would be driven on to the end of its
    clock. Both the plan-free step and the way home carry arm.path_tol_rad, as it is set."""
    import rammp_arm

    monkeypatch.setattr(rammp_arm, "PATH_TOL_RAD", 0.11)  # as arm.path_tol_rad sets it at startup
    path = np.array([START, START + 0.05])
    arm = Arm(answers=[_refuses_start(START)])
    out = _go(_engine(arm, Model(inside=0.012, escape=path)))
    assert out["stepped_out"] and len(arm.sent) == 2 and arm.path_tols == [0.11, 0.11]


def test_home_is_said_only_of_an_arm_that_is_read_there():
    """The driver ends a goal when its clock runs out, wherever the arm is."""
    arm = Arm()
    short = HOME + np.array([0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])

    def held_back(names, points, speed_scale=1.0, path_tolerance=None):
        arm.sent.append(list(points))
        arm.readings = [short]  # SUCCESSFUL, the arm 0.2 rad short on joint 2
        return SimpleNamespace(error_code=0, error_string="")

    arm.execute_joint_trajectory = held_back
    with pytest.raises(EngineError) as e:
        _go(_engine(arm, Model()))
    assert e.value.error_code == "motion_failed" and "0.200 rad (joint 2) from the home" in str(
        e.value
    )
    assert len(arm.sent) == 1
    out = _go(_engine(Arm(), Model()))  # ... and an arm that got there says how near
    assert out["home"] and out["off_rad"] == 0.0


def test_joints_that_cannot_be_read_after_a_goal_flew_fail_as_motion_failed_not_arm_state():
    """arm_state says nothing was planned or sent. After a goal flew that is not so: the arm moved
    and where it stopped is not known (REVIEW_RULES U4)."""
    from capability_engine import StateTimeout

    arm = Arm()
    fly = arm.execute_joint_trajectory

    def fly_then_lose_state(names, points, speed_scale=1.0, path_tolerance=None):
        res = fly(names, points, speed_scale=speed_scale, path_tolerance=path_tolerance)

        def stale():
            raise StateTimeout("/joint_states is stale: last message 0.80 s ago")

        arm.joint_states = stale
        return res

    arm.execute_joint_trajectory = fly_then_lose_state
    with pytest.raises(EngineError) as e:
        _go(_engine(arm, Model()))
    assert e.value.error_code == "motion_failed" and not e.value.recoverable
    assert "the way home was flown" in str(e.value) and len(arm.sent) == 1


def test_off_a_limit_with_layers_that_cannot_be_read_nothing_is_stepped_unchecked():
    """The plan-free step off a joint limit is held out of the chair and user by the body model.
    Where body_layers names one that cannot be read, the step is refused; where the deployment
    says there is none (body_layers empty), it flies on the tool's height alone."""
    near = HOME.copy()
    near[1] = LIMITS[1][0] + 0.05
    arm = Arm(joints=near)
    with pytest.raises(EngineError) as e:
        _go(
            _engine(arm, None, no_model="body model: cannot read /config/robot/curobo/planner.yaml")
        )
    assert e.value.error_code == "body_model" and "cannot be checked against the chair" in str(
        e.value
    )
    assert arm.planned == [] and arm.sent == []
    arm = Arm(joints=near)
    out = _go(_engine(arm, None, no_model=node.NO_LAYERS))
    assert out["home"] and len(arm.sent) == 2  # the step off the limit, then the way home


def test_a_pause_asked_for_before_the_arm_is_sent_anything_is_taken_there_and_the_arm_read_again(
    monkeypatch,
):
    """self.arm's own checkpoint would block on a pause inside the send, the plans already judged
    and the arm last read before it. The pause is taken at the last pausable moment instead: an
    arm moved meanwhile is sent nothing, and one that stood still goes home."""
    now = [0.0]
    monkeypatch.setattr(node.time, "monotonic", lambda: now[0])

    def paused_last(arm, moved_to=None):
        class Last(Ctx):
            def checkpoint(self):
                if self.reports and self.reports[-1][2] is False:  # after the first unpausable step
                    now[0] += 5.0
                    if moved_to is not None:
                        arm.readings = [np.asarray(moved_to, float)]

        return Last()

    arm = Arm()
    with pytest.raises(EngineError) as e:
        Capability.execute(_engine(arm, Model()), "go_home", {}, paused_last(arm, START + 0.1))
    assert e.value.error_code == "arm_moving" and e.value.recoverable and arm.sent == []
    assert "while the action was paused" in str(e.value)
    arm = Arm()
    out = Capability.execute(_engine(arm, Model()), "go_home", {}, paused_last(arm))
    assert out["home"] and len(arm.sent) == 1


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
