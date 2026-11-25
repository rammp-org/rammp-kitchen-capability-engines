"""Drive the production docking loop against a simulated differential base."""

import math
import sys
import threading
from collections import deque
from contextlib import contextmanager
from types import SimpleNamespace as Obj
from unittest.mock import MagicMock

import numpy as np
import pytest
from door_common.common import Refused
from fridge import arrival, base_drive, docking

from capability_engine import BaseError, Cancelled


class VirtualBase:
    """Integrate body twists at the axle, and report the offset odometry datum."""

    def __init__(self, pose, ctx):
        self.pose = np.array(pose, float)
        self.time = 0.0
        self.command = np.zeros(2)
        self.commands = []
        self.mode_calls = 0
        self.estopped = False
        self.ctx = ctx
        self.on_sleep = lambda: None

    def monotonic(self):
        return self.time

    def sleep(self, dt):
        self.time += dt
        self.pose[:2] += docking.rotation(self.pose[2])[:, 0] * self.command[0] * dt
        self.pose[2] += self.command[1] * dt
        self.on_sleep()

    def request_mode(self, mode, *, profile):
        assert mode == "autonomous"
        assert profile == "low"
        self.mode_calls += 1

    def send_twist(self, vx, wz):
        self.ctx.checkpoint()
        self.command[:] = vx, wz
        self.commands.append((vx, wz))

    def stop(self):
        self.command[:] = 0, 0
        self.commands.append((0, 0))

    def odometry(self):
        yaw = self.pose[2]
        x, y = self.pose[:2] - docking.rotation(yaw) @ docking.ODOM_TO_AXLE
        q = Obj(x=0, y=0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))
        return Obj(
            pose=Obj(pose=Obj(position=Obj(x=x, y=y), orientation=q)),
            twist=Obj(twist=Obj(linear=Obj(x=self.command[0]), angular=Obj(z=self.command[1]))),
        )

    def joint_states(self):
        return Obj(position=np.zeros(7))

    def control_status(self):
        return Obj(estopped=self.estopped)


class Ctx:
    cancel = pause = False

    def checkpoint(self):
        if self.cancel:
            raise Cancelled()

    def paused(self):
        return self.pause


def run_dock(
    base,
    ctx,
    guard=lambda *_: None,
    *,
    assume_odom_to_axle=docking.ODOM_TO_AXLE,
    on_route=None,
    confirm=None,
    arrival_check=None,
    arrival_relook=None,
):
    mount = np.asarray(docking.AXLE_TO_MOUNT[:2])

    def look():
        turn = docking.rotation(-base.pose[2])
        handle = turn @ (docking.TARGET_HANDLE + mount - base.pose[:2]) - mount
        normal = -(docking.rotation(docking.TARGET_YAW - base.pose[2])[:, 0])
        return dict(h_raw=np.r_[handle, 0.326], n=np.r_[normal, 0.0]), []

    return docking.dock(
        base,
        base,
        ctx,
        look,
        guard,
        speed=0.12,
        cfg=base_drive.PARAMS,
        axle_to_mount=mount,
        odom_to_axle=np.asarray(assume_odom_to_axle),
        on_motion=lambda: None,
        on_route=on_route,
        confirm=confirm,
        arrival_check=arrival_check,
        arrival_relook=arrival_relook,
        clock=base.monotonic,
        sleep=base.sleep,
    )


def test_depth_allows_reversing_a_padding_only_arm_overlap(monkeypatch):
    y, z = np.meshgrid(np.linspace(-0.26, 0.26, 27), np.linspace(0.43, 0.75, 20))
    face = np.column_stack((np.full(y.size, 0.52), y.ravel(), z.ravel()))
    monkeypatch.setattr(arrival, "depth_points", lambda _: face)
    chair = [dict(name="chair", position=[0.10, 0.0, -0.10], dims=[0.30, 0.40, 0.40])]
    arm = [dict(name="arm", position=[0.405, 0.0, 0.50], dims=[0.20, 0.14, 0.14])]
    reverse = arrival.reverse_distance(None, chair, arm, padding=0.02)
    assert reverse == pytest.approx(0.185, abs=0.01)
    assert 0.52 + reverse - 0.505 >= 0.19


def test_close_arrival_backs_straight_before_looking_or_preparing_arm():
    ctx = Ctx()
    base = VirtualBase((0.12, 0.0, 0.0), ctx)
    events = []

    def assess():
        assert not base.commands
        events.append("depth")
        return 0.25

    def prepare(*_):
        nonzero = [(vx, wz) for vx, wz in base.commands if vx or wz]
        assert nonzero and all(vx < 0 and wz == 0 for vx, wz in nonzero)
        assert base.pose[0] == pytest.approx(-0.13, abs=docking.TRANSLATION_TOL)
        events.append("prepare")

    run_dock(base, ctx, on_route=prepare, arrival_check=assess)
    assert events == ["depth", "prepare"]
    assert base.mode_calls == 1


def test_close_arrival_stops_on_unexpected_arm_motion():
    ctx = Ctx()
    base = VirtualBase((0.12, 0.0, 0.0), ctx)
    joints = np.zeros(7)
    base.joint_states = lambda: Obj(position=joints.copy())
    base.on_sleep = lambda: joints.fill(0.02)

    with pytest.raises(BaseError, match="arm moved during arrival recovery"):
        run_dock(base, ctx, arrival_check=lambda: 0.25)
    assert base.commands[-1] == (0, 0)


def test_dock_checks_and_executes_rotate_translate_rotate():
    ctx = Ctx()
    base = VirtualBase((-0.15, -0.10, -0.25), ctx)
    routes = []
    (geom, _), corrections = run_dock(base, ctx, lambda _, route: routes.append(route))
    assert corrections == len(routes) == 1
    assert base.mode_calls == 1
    assert len(routes[0]) == 3
    assert np.linalg.norm(geom["h_raw"][:2] - docking.TARGET_HANDLE) <= docking.POSITION_TOLERANCE
    assert abs(base.pose[2]) <= docking.YAW_TOLERANCE
    assert 0 < max(abs(v) for v, _ in base.commands) <= 0.05
    assert 0 < max(abs(w) for _, w in base.commands) <= docking.MAX_YAW_SPEED
    assert base.commands[-1] == (0, 0)


def test_ready_follows_only_the_terminal_leg_and_closes_on_failure():
    ctx = Ctx()
    base = VirtualBase((-0.15, -0.10, -0.25), ctx)
    events = []

    @contextmanager
    def follow(clock):
        events.append("stream opened")

        def step():
            if base.command[1]:
                events.append("turn and arm in progress")
                raise BaseError("Arm tracking")

        try:
            yield step
        finally:
            events.append("stream closed")

    def on_route(geom, q, delta, yaw, mount, route):
        assert len(route) == 3
        expected = tuple(docking.route_samples(route[:-1]))[-1]
        return lambda: Obj(follow=follow, route_at=expected)

    with pytest.raises(BaseError, match="Arm tracking"):
        run_dock(base, ctx, on_route=on_route)
    assert events == ["stream opened", "turn and arm in progress", "stream closed"]
    assert base.commands[-1] == (0, 0)


def test_dock_retries_wrist_from_park_after_ready_pose_occlusion():
    ctx = Ctx()
    base = VirtualBase((-0.15, -0.10, -0.25), ctx)
    events = []

    @contextmanager
    def follow(clock):
        events.append("ready")
        yield lambda: None

    def park(ctx, clock, sleep, stationary):
        stationary()
        events.append("parked")

    def on_route(geom, q, delta, yaw, mount, route):
        expected = tuple(docking.route_samples(route[:-1]))[-1]
        return lambda: Obj(follow=follow, park=park, route_at=expected)

    def confirm(observation):
        if events[-1] == "ready":
            events.append("occluded")
            raise Refused("Handle obscured by ready pose")
        events.append("wrist confirmed")
        return observation

    _, corrections = run_dock(base, ctx, on_route=on_route, confirm=confirm)
    assert corrections == 1
    assert events == ["ready", "occluded", "parked", "wrist confirmed"]


def test_dock_at_target_does_not_request_mode_or_twists():
    ctx = Ctx()
    base = VirtualBase((0.0, 0.0, 0.0), ctx)
    assert run_dock(base, ctx)[1] == 0
    assert base.mode_calls == 0
    assert base.commands == []


def test_over_budget_and_blocked_routes_refuse_before_mode_request():
    ctx = Ctx()
    far = VirtualBase((-0.50, 0.0, 0.0), ctx)
    with pytest.raises(Refused, match="distance or heading budget"):
        run_dock(far, ctx)
    near = VirtualBase((-0.10, 0.0, 0.0), ctx)

    def deny(*_):
        raise Refused("test leg obstruction")

    with pytest.raises(Refused, match="leg obstruction"):
        run_dock(near, ctx, deny)
    assert far.mode_calls == near.mode_calls == 0
    assert far.commands == near.commands == []


@pytest.mark.parametrize(
    "fault,error",
    [
        ("cancel", Cancelled),
        ("pause", BaseError),
        ("estopped", BaseError),
    ],
)
def test_dock_stops_on_interruption(fault, error):
    ctx = Ctx()
    base = VirtualBase((-0.10, -0.05, 0.0), ctx)
    target = base if fault == "estopped" else ctx
    base.on_sleep = lambda: setattr(target, fault, True)
    with pytest.raises(error):
        run_dock(base, ctx)
    assert base.commands[0] != (0, 0)
    assert base.commands[-1] == (0, 0)


def test_rejected_direct_path_can_use_a_guarded_retreat():
    seen = []

    def guard(_, route):
        seen.append(route)
        if len(route) < 4:
            raise Refused("obstructed")

    route, distance, turn = docking.checked_route(
        np.array([-0.15, -0.035]),
        0.1,
        0.35,
        math.pi,
        None,
        guard,
    )
    assert route[0] == (-0.08, 0.0)
    assert len(seen) >= 2
    assert distance <= 0.35 and turn <= math.pi
    position, yaw = list(docking.route_samples(route))[-1]
    assert position == pytest.approx((-0.15, -0.035))
    assert yaw == pytest.approx(0.1)


def test_dock_rejects_the_legacy_urdf_root_as_an_odom_datum():
    ctx = Ctx()
    base = VirtualBase((-0.10, -0.05, 0.0), ctx)
    with pytest.raises(BaseError, match="departed the checked docking segment"):
        run_dock(base, ctx, assume_odom_to_axle=(0.48575, -0.002625))
    assert base.commands[-1] == (0, 0)


def test_camera_flushes_both_cached_and_in_flight_pre_motion_frames(monkeypatch):
    for name in ("open3d", "message_filters"):
        monkeypatch.setitem(sys.modules, name, MagicMock())
    from door_common.perception import WristCamera

    camera = WristCamera.__new__(WristCamera)
    camera._lock = threading.Lock()
    camera._sam3 = None
    camera._latest = "cached"
    camera._recent = deque(["cached"])
    camera.node = Obj(get_clock=lambda: Obj(now=lambda: Obj(nanoseconds=2_000_000_000)))

    def frame(ns):
        sec, nanosec = divmod(ns, 1_000_000_000)
        return Obj(header=Obj(stamp=Obj(sec=sec, nanosec=nanosec)))

    camera.require_frames_after_now()
    rgb, info = frame(2_010_000_000), frame(2_010_000_000)
    camera._on_frame(rgb, info, frame(1_997_000_000))
    assert camera._latest is None and not camera._recent
    depth = frame(2_014_000_000)
    camera._on_frame(rgb, info, depth)
    assert camera._latest == (rgb, info, depth)


def test_manual_guidance_relooks_after_steps_without_requesting_drive_control():
    ctx = Ctx()
    base = VirtualBase((-0.16, 0.0, 0.0), ctx)
    mount = np.asarray(docking.AXLE_TO_MOUNT[:2])
    target = docking.TARGET_HANDLE + np.array([0.025, -0.02])
    observations, prompts, checked = [], [], []

    def look():
        observations.append(base.pose.copy())
        handle = target - base.pose[:2]
        normal = -docking.rotation(docking.TARGET_YAW)[:, 0]
        return dict(h_raw=np.r_[handle, 0.393], n=np.r_[normal, 0.0]), []

    def guide_driver(distance, turn):
        prompts.append((distance, turn))
        assert turn == 0.0
        base.pose[0] += 0.9 * distance

    (geom, _), steps = docking.guide(
        base,
        base,
        ctx,
        look,
        lambda _, route: checked.append(route),
        axle_to_mount=mount,
        odom_to_axle=np.asarray(docking.ODOM_TO_AXLE),
        show=guide_driver,
        target_handle=target,
        clock=base.monotonic,
        sleep=base.sleep,
    )
    assert steps == len(prompts) == len(checked) == 2
    assert len(observations) == 3
    assert geom["h_raw"][:2] == pytest.approx(target, abs=docking.POSITION_TOLERANCE)
    assert base.commands == []
    assert base.mode_calls == 0


def test_guided_lateral_route_keeps_a_shallow_view_and_reaches_target():
    displacement, yaw = np.array([-0.11, -0.10]), 0.09
    *_, shallow, reversed_shallow = docking.guided_routes(displacement, yaw)
    for route in (shallow, reversed_shallow):
        poses = list(docking.route_samples(route))
        assert poses[-1][0] == pytest.approx(displacement)
        assert poses[-1][1] == pytest.approx(yaw)
        assert (
            max(abs(heading - yaw) for _, heading in poses) <= docking.GUIDED_LOOK_HEADING + 1e-10
        )
    distance, turn = docking.guided_step(displacement, yaw, None, lambda *_: None)
    assert distance == 0.0
    assert turn == docking.GUIDED_TURN


def test_autonomous_dock_uses_shallow_route_when_steep_turns_are_blocked():
    displacement = np.array([0.0, 0.07])

    def guard(_, route):
        if any(abs(angle) > docking.GUIDED_LOOK_HEADING for _, angle in route):
            raise Refused("Steep turn blocked")

    route, distance, heading = docking.checked_route(displacement, 0.0, 0.35, math.pi, None, guard)
    position, yaw = list(docking.route_samples(route))[-1]
    assert position == pytest.approx(displacement)
    assert yaw == pytest.approx(0.0)
    assert distance <= 0.35 and heading <= math.pi
