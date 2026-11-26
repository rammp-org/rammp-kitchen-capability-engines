"""Instrumented Gen1.5 differential drive driven by the MIB's body-frame twist API."""

import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class DriveProfile:
    linear_limit: float
    yaw_limit: float
    linear_accel: float
    yaw_accel: float


# RAMMP-mib-fw MIBConfig.hpp, fcec972.
LOW = DriveProfile(0.35, 1.0, 0.5, 1.0)
NORMAL = DriveProfile(0.75, 2.0, 1.0, 2.0)
HIGH = DriveProfile(1.0, 3.14159, 2.0, 4.0)
TRACK_M = 0.558
WHEEL_RADIUS_M = 0.127
WATCHDOG_S = 0.300


def _limited(value, limit):
    return min(limit, max(-limit, value))


def _integrate(pose, velocity, yaw_rate, dt):
    x, y, yaw = pose
    mid = yaw + yaw_rate * dt / 2
    arc = 1.0 if not yaw_rate else math.sin(yaw_rate * dt / 2) / (yaw_rate * dt / 2)
    forward = arc * velocity * dt
    return x + forward * math.cos(mid), y + forward * math.sin(mid), yaw + yaw_rate * dt


class DifferentialDrive:
    def __init__(
        self,
        *,
        time=0.0,
        axle_pose=(0.0, 0.0, 0.0),
        odom_to_axle=(0.0, 0.0),
        latency_s=0.04,
        traction=(0.98, 0.99),
        encoder_scale=(1.002, 0.998),
        profile=LOW,
    ):
        self.time = time
        self.truth = self.encoder_pose = axle_pose
        self.odom_to_axle = odom_to_axle
        self.latency_s = latency_s
        self.traction = traction
        self.encoder_scale = encoder_scale
        self.profile = profile
        self.autonomous = False
        self.twists = []
        self.samples = []
        self._pending = deque()
        self._last = -math.inf
        self._target = self._velocity = (0.0, 0.0)
        self.encoder_v = self.encoder_w = 0.0

    def set_mode(self, time, autonomous):
        self.advance(time)
        if not autonomous:
            self._pending.clear()
            self._last = -math.inf
        self.autonomous = autonomous

    def command(self, time, vx, wz, vy=0.0):
        self.advance(time)
        self.twists.append((time, vx, vy, wz, self.autonomous))
        if self.autonomous:
            self._pending.append((time + self.latency_s, vx, wz))

    def advance(self, time):
        if time < self.time:
            raise ValueError("Differential drive clock ran backwards")
        while self.time < time:
            while self._pending and self._pending[0][0] <= self.time:
                self._last, vx, wz = self._pending.popleft()
                self._target = vx, wz
            expires = self._last + WATCHDOG_S
            active = self.autonomous and self.time < expires
            end = min(time, self.time + 0.01)
            if self._pending:
                end = min(end, self._pending[0][0])
            if active:
                end = min(end, expires)
            dt = end - self.time
            self._step(dt, self._target if active else (0.0, 0.0))
            self.time = end

    def _step(self, dt, target):
        vx, wz = self._velocity
        profile = self.profile
        vx += _limited(_limited(target[0], profile.linear_limit) - vx, profile.linear_accel * dt)
        wz += _limited(_limited(target[1], profile.yaw_limit) - wz, profile.yaw_accel * dt)
        self._velocity = vx, wz
        left = (vx - wz * TRACK_M / 2) / WHEEL_RADIUS_M
        right = (vx + wz * TRACK_M / 2) / WHEEL_RADIUS_M
        self.truth, _, _ = self._roll(self.truth, left, right, self.traction, dt)
        self.encoder_pose, self.encoder_v, self.encoder_w = self._roll(
            self.encoder_pose, left, right, self.encoder_scale, dt
        )

    @staticmethod
    def _roll(pose, left, right, scales, dt):
        left_mps = left * WHEEL_RADIUS_M * scales[0]
        right_mps = right * WHEEL_RADIUS_M * scales[1]
        v, w = (left_mps + right_mps) / 2, (right_mps - left_mps) / TRACK_M
        return _integrate(pose, v, w, dt), v, w

    def odometry(self, time):
        self.advance(time)
        x, y, yaw = self.encoder_pose
        ax, ay = self.odom_to_axle
        pose = (
            x - ax * math.cos(yaw) + ay * math.sin(yaw),
            y - ax * math.sin(yaw) - ay * math.cos(yaw),
            yaw,
        )
        v, w = self.encoder_v, self.encoder_w
        sample = (*pose, v + w * ay, -w * ax, w)
        self.samples.append((time, *sample, *self.truth))
        return sample
