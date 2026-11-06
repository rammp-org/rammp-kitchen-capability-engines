"""fridge.base_drive: the fixed-distance wheelchair move, and the open and close driving the base
(`drive_base`) on the sim. A fake base and a fake clock stand in for self.base and time; one test
drives the engine's BaseInterface in mock mode. Nothing here can reach a real base or arm."""

import json
import math
import re
import types
from functools import partial

import numpy as np
import pytest
import test_fridge_planning as planning
from door_common import common as dc
from fridge import base_drive, close_door, constants, open_door, push_open

from capability_engine import BaseError, StateTimeout
from capability_engine.engine import Cancelled

FRIDGE_DOOR, TELEOP_START, _Arm = planning.FRIDGE_DOOR, planning.TELEOP_START, planning._Arm
door_sim = planning.door_sim  # the planning tests' fixture
CFG = dict(base_drive.PARAMS)
TOL = CFG["base_tolerance"]


class Clock:
    """time for base_drive: sleep() advances it and lets the fake base move."""

    def __init__(self):
        self.t, self.on_sleep = 0.0, []

    def monotonic(self):
        return self.t

    def sleep(self, dt):
        for f in self.on_sleep:
            f(dt)
        self.t += dt


class Ctx:
    def __init__(self):
        self.cancel = False

    def cancelled(self):
        return self.cancel

    def checkpoint(self):
        if self.cancel:
            raise Cancelled()


class FakeBase:
    """self.base: the commanded vx (base_link) moves the pose while the clock runs, as the MIB
    would; `odom` may rewrite what odometry() reports (a stuck, drifting or reversed reading);
    `coast` m more once stopped."""

    def __init__(self, clock, ctx, yaw=0.0, odom=None, coast=0.0):
        self.ctx, self.x, self.y, self.yaw, self.vx = ctx, 0.0, 0.0, yaw, 0.0
        self.sent, self.modes, self.odom, self.coast = [], [], odom, coast
        clock.on_sleep.append(self._advance)

    def _advance(self, dt):
        self.x += self.vx * math.cos(self.yaw) * dt
        self.y += self.vx * math.sin(self.yaw) * dt

    def request_mode(self, mode, profile=None):
        self.modes.append((mode, profile))

    def send_twist(self, vx, wz, vy=0.0):
        if self.ctx.cancelled():
            raise Cancelled()
        self.sent.append(vx)
        self.vx = vx

    def stop(self):
        if self.vx and self.coast:
            self.x += math.copysign(self.coast, self.vx) * math.cos(self.yaw)
            self.y += math.copysign(self.coast, self.vx) * math.sin(self.yaw)
        self.sent.append(0.0)
        self.vx = 0.0

    def odometry(self):
        x, y = (self.x, self.y) if self.odom is None else self.odom(self.x, self.y)
        q = types.SimpleNamespace(x=0.0, y=0.0, z=math.sin(self.yaw / 2), w=math.cos(self.yaw / 2))
        return types.SimpleNamespace(
            pose=types.SimpleNamespace(
                pose=types.SimpleNamespace(
                    position=types.SimpleNamespace(x=x, y=y, z=0.0), orientation=q
                )
            )
        )


class Arm:
    estopped = False

    def control_status(self):
        return types.SimpleNamespace(estopped=self.estopped)


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(base_drive, "time", c)
    monkeypatch.setattr(base_drive, "say", lambda m: None)
    return c


def _move(base, arm, ctx, move, speed=0.05, cfg=CFG, execute=True):
    return base_drive.move_chair(base, arm, ctx, move, speed=speed, cfg=cfg, execute=execute)


@pytest.mark.parametrize(
    "move,speed", [(0.30, 0.03), (0.05, 0.03), (-0.10, 0.05), (0.20, 0.15), (-0.35, 0.10)]
)
def test_it_drives_the_distance_within_the_tolerance_and_ends_still(clock, move, speed):
    ctx = Ctx()
    base = FakeBase(clock, ctx)
    moved = _move(base, Arm(), ctx, move, speed)
    assert moved == pytest.approx(move, abs=TOL)
    assert -base.x == pytest.approx(move, abs=TOL)  # back is -x
    assert base.sent[-1] == 0.0 and base.vx == 0.0
    assert max(abs(v) for v in base.sent) == pytest.approx(speed)
    assert all(v * move <= 0 for v in base.sent)  # never the other way
    assert base.modes == [("autonomous", "low")]


def test_back_is_along_the_chairs_own_heading(clock):
    ctx = Ctx()
    base = FakeBase(clock, ctx, yaw=math.pi / 2)
    assert _move(base, Arm(), ctx, 0.20) == pytest.approx(0.20, abs=TOL)
    assert (base.x, base.y) == pytest.approx((0.0, -0.20), abs=TOL)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"execute": False}, "dry run"),
        ({"move": 0.40}, "base_max_distance"),
        ({"speed": 0.20}, "base_speed"),
        ({"speed": 0.0}, "base_speed"),
    ],
)
def test_it_refuses_before_moving(clock, kwargs, match):
    ctx = Ctx()
    base = FakeBase(clock, ctx)
    args = {"move": 0.30, "speed": 0.05, "execute": True, **kwargs}
    with pytest.raises(dc.Refused, match=match):
        _move(base, Arm(), ctx, args["move"], args["speed"], execute=args["execute"])
    assert base.sent == [] and base.modes == []


def test_an_arm_e_stopped_before_it_refuses(clock):
    ctx, arm = Ctx(), Arm()
    arm.estopped = True
    base = FakeBase(clock, ctx)
    with pytest.raises(dc.Refused, match="e-stopped"):
        _move(base, arm, ctx, 0.30)
    assert base.sent == [] and base.modes == []


def test_an_arm_e_stop_mid_drive_stops_the_base(clock):
    ctx, arm = Ctx(), Arm()
    base = FakeBase(clock, ctx)
    clock.on_sleep.append(lambda dt: setattr(arm, "estopped", clock.t >= 1.0))
    with pytest.raises(BaseError, match="e-stopped"):
        _move(base, arm, ctx, 0.30)
    assert base.sent[-1] == 0.0 and base.vx == 0.0
    assert -base.x == pytest.approx(0.05, abs=0.005)  # 1 s at 0.05 m/s, then nothing


def test_a_cancel_mid_drive_stops_the_base(clock):
    ctx = Ctx()
    base = FakeBase(clock, ctx)
    clock.on_sleep.append(lambda dt: setattr(ctx, "cancel", clock.t >= 1.0))
    with pytest.raises(Cancelled):
        _move(base, Arm(), ctx, 0.30)
    assert base.sent[-1] == 0.0 and base.vx == 0.0


@pytest.mark.parametrize(
    "odom,match",
    [
        (lambda x, y: (0.0, 0.0), "did not cover"),  # odometry stuck at the start
        (lambda x, y: (-x, y), "wrong way"),  # reversed
        (lambda x, y: (x, 2.0 * x), "more than asked"),  # sliding sideways
    ],
)
def test_bad_odometry_stops_the_base(clock, odom, match):
    ctx = Ctx()
    base = FakeBase(clock, ctx, odom=odom)
    with pytest.raises(BaseError, match=match):
        _move(base, Arm(), ctx, 0.30)
    assert base.sent[-1] == 0.0 and base.vx == 0.0
    # however the odometry lies, it never drives on for long: at most the timeout's worth
    assert abs(base.x) <= 0.30 + 0.05 * CFG["base_timeout_margin"] + 1e-9


def test_stuck_odometry_stops_at_the_timeout(clock):
    ctx = Ctx()
    base = FakeBase(clock, ctx, odom=lambda x, y: (0.0, 0.0))
    with pytest.raises(BaseError):
        _move(base, Arm(), ctx, 0.10, speed=0.05)
    assert clock.t == pytest.approx(0.10 / 0.05 + CFG["base_timeout_margin"], abs=0.1)


def test_ending_further_off_than_the_tolerance_fails(clock):
    ctx = Ctx()
    base = FakeBase(clock, ctx, coast=0.02)  # 2 cm on after the stop
    with pytest.raises(BaseError, match="from the distance"):
        _move(base, Arm(), ctx, 0.30)
    assert base.vx == 0.0


def test_the_tolerance_is_a_parameter(clock):
    ctx = Ctx()
    base = FakeBase(clock, ctx, coast=0.02)
    assert _move(base, Arm(), ctx, 0.30, cfg={**CFG, "base_tolerance": 0.03}) == pytest.approx(
        0.30, abs=0.03
    )


def test_check_refuses_without_odometry_or_the_arms_status(clock):
    ctx = Ctx()
    base = FakeBase(clock, ctx)
    base_drive.check(base, Arm())

    def no_odometry():
        raise BaseError("no /base/odom")

    base.odometry = no_odometry
    with pytest.raises(dc.Refused, match="no /base/odom"):
        base_drive.check(base, Arm())

    class NoStatus:
        def control_status(self):
            raise StateTimeout("no /control_status")

    with pytest.raises(dc.Refused, match="no /control_status"):
        base_drive.check(FakeBase(clock, ctx), NoStatus())


def test_it_drives_the_engines_base_interface_in_mock_mode():
    """The real self.base (mock: dead reckoning, nothing sent), real time: the calls base_drive
    makes are its API."""
    import rclpy
    from rclpy.callback_groups import ReentrantCallbackGroup

    from capability_engine.base import BaseInterface

    context = rclpy.context.Context()  # its own: the engine tests may hold the default one
    rclpy.init(context=context)
    node = rclpy.create_node("base_drive_test", context=context)
    try:
        base = BaseInterface(node, ReentrantCallbackGroup(), mock=True)
        ctx = Ctx()
        base.begin_action(ctx)
        try:
            moved = _move(base, Arm(), ctx, 0.10, speed=0.15)
            p = base.odometry().pose.pose.position
        finally:
            base.end_action()
        assert moved == pytest.approx(0.10, abs=TOL)
        assert p.x == pytest.approx(-0.10, abs=TOL)
    finally:
        node.destroy_node()
        rclpy.shutdown(context=context)


# --- the open and the close, driving the base ---


def _driver(clock, arm, fail_at_s=None):
    """drive_chair as the node makes it, on a fake base; `fail_at_s`: the MIB leaves AUTONOMOUS
    then (a joystick touch)."""
    ctx = Ctx()
    base = FakeBase(clock, ctx)
    if fail_at_s is not None:
        send = base.send_twist

        def send_twist(vx, wz, vy=0.0):
            if clock.t >= fail_at_s:
                raise BaseError("the MIB left AUTONOMOUS")
            send(vx, wz, vy)

        base.send_twist = send_twist
    return base, partial(base_drive.move_chair, base, arm, ctx, speed=0.05, cfg=CFG, execute=True)


def test_the_open_drives_the_base_for_the_push_and_pushes_in_one_goal(door_sim, clock, monkeypatch):
    scene, rb = door_sim
    arm = _Arm(scene, rb, TELEOP_START)
    base, drive = _driver(clock, arm)
    assert open_door.run(
        arm, False, swing_deg=50, drive_chair=drive
    )  # a dry run: says it will drive, drives nothing
    assert "DRIVES THE BASE" in push_open.NEXT_STEP and base.sent == [] and arm.calls == []
    m = re.search(r"DRIVES THE BASE ~(\d+) cm (back|forward)", push_open.NEXT_STEP)
    planned = int(m.group(1)) * (1 if m.group(2) == "back" else -1)

    seen = {}  # the door file as the push starts: after the pull, and the drive it recorded
    push = push_open.push_after_chair_move

    def recorded(*args):
        seen.update(json.loads(dc.DOOR_FILE.read_text()))
        return push(*args)

    monkeypatch.setattr(push_open, "push_after_chair_move", recorded)
    assert open_door.run(arm, True, swing_deg=50, drive_chair=drive)
    assert [c[0] for c in arm.calls] == [
        "curobo",
        "joints",
        "gripper",
        "joints",
        "gripper",
        "joints",
        "joints",
    ]
    assert push_open.NEXT_STEP == ""  # nothing asked of the person: opened
    moved = -base.x
    assert abs(moved * 100 - planned) <= 5 + 1
    door = json.loads(dc.DOOR_FILE.read_text())
    assert (
        door["door_open_deg"] == constants.PUSH_OPEN_TO_DEG
        and door["push_after_chair_move"] is None
    )
    assert seen["push_after_chair_move"] == pytest.approx(moved)  # the drive really made
    assert door["hinge"][0] == pytest.approx(seen["hinge"][0] + moved)  # the door where it now is


def test_an_open_whose_drive_stops_leaves_no_push_waiting(door_sim, clock):
    scene, rb = door_sim
    arm = _Arm(scene, rb, TELEOP_START)
    base, drive = _driver(clock, arm, fail_at_s=0.5)
    with pytest.raises(BaseError):
        open_door.run(arm, True, swing_deg=50, drive_chair=drive)
    assert base.vx == 0.0 and arm.gripper_pos == 0.0  # off the handle, base stopped
    assert json.loads(dc.DOOR_FILE.read_text()).get("push_after_chair_move") is None
    with pytest.raises(
        dc.Refused, match="no push waiting"
    ):  # a chair_moved goal can't push the wrong door
        open_door.run(arm, True, chair_moved=True)


def test_the_close_drives_the_base_back_then_closes(door_sim, clock):
    scene, rb = door_sim
    arm = _Arm(scene, rb, TELEOP_START)
    dc.DOOR_FILE.write_text(json.dumps(FRIDGE_DOOR))
    base, drive = _driver(clock, arm)
    assert close_door.run(arm, False, drive_chair=drive)  # a dry run drives nothing
    assert base.sent == [] and arm.calls == []
    assert close_door.run(arm, True, drive_chair=drive)  # no chair_moved needed
    assert -base.x == pytest.approx(constants.CHAIR_BACK_M, abs=TOL)
    assert arm.calls and all(c[0] == "joints" for c in arm.calls)
    assert json.loads(dc.DOOR_FILE.read_text())["door_open_deg"] == constants.CLOSE_TO_DEG


def test_the_close_refuses_to_drive_if_the_arm_would_meet_the_door(door_sim, clock, monkeypatch):
    scene, rb = door_sim
    arm = _Arm(scene, rb, TELEOP_START)
    dc.DOOR_FILE.write_text(json.dumps(FRIDGE_DOOR))
    base, drive = _driver(clock, arm)
    monkeypatch.setattr(push_open, "_clear_through_move", lambda *a: False)
    with pytest.raises(dc.Refused, match="open door"):
        close_door.run(arm, True, drive_chair=drive)
    assert base.sent == [] and arm.calls == []


def test_the_close_with_chair_moved_never_drives(door_sim, clock):
    scene, rb = door_sim
    arm = _Arm(scene, rb, TELEOP_START)
    dc.DOOR_FILE.write_text(json.dumps(FRIDGE_DOOR))
    base, drive = _driver(clock, arm)
    assert close_door.run(arm, True, chair_moved=True, drive_chair=drive)
    assert base.sent == []


def test_the_clearance_check_passes_for_the_arm_where_the_close_starts(door_sim):
    scene, rb = door_sim
    arm = _Arm(scene, rb, TELEOP_START)
    close_door.clear_through_drive(
        arm, FRIDGE_DOOR, {}, constants.CHAIR_BACK_M, constants.CHAIR_BACK_DIR
    )
    assert np.isfinite(arm.q).all()
