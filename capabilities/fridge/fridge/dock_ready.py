"""A free-space ready pose checked against all appliance positions on the terminal docking leg."""

import math
from contextlib import contextmanager
from itertools import product

import numpy as np
from capability_engine import BaseError
from door_common.common import say

from . import constants as F
from . import docking, open_door
from . import planning as pl
from .world import FridgeModel

# Cover scene fit, base tracking, interpolation between route samples and arm tracking.
SWEEP_PADDING_M = 0.10
START_TOL_RAD = math.radians(1)
TRACK_TOL_RAD = math.radians(3)
TRACK_TOL_M = 0.02
PERIOD_S = 0.05


def _terminal_poses(route):
    before = tuple(docking.route_samples(route[:-1]))[-1]
    axle, yaw = before
    for shift, turn in docking.route_samples(route[-1:]):
        yield axle + docking.rotation(yaw) @ shift, yaw + turn


def appliance_envelope(model, route, mount):
    """One containing box per appliance box in the moving arm_mount frame."""
    poses = tuple(_terminal_poses(route))
    corners = np.asarray(list(product((-0.5, 0.5), repeat=2)))
    boxes = []
    for box in model.full(0.0):
        centre = np.asarray(box["position"], float)
        size = np.asarray(box["dims"], float)
        angle = math.radians(box["rpy_deg"][2])
        points = (corners * size[:2]) @ docking.rotation(angle).T + centre[:2]
        axis = angle - poses[-1][1]
        basis = docking.rotation(axis)
        samples = np.vstack([((points + mount - axle) @ docking.rotation(yaw) - mount) @ basis for axle, yaw in poses])
        low, high = samples.min(axis=0), samples.max(axis=0)
        boxes.append(
            dict(
                name=f"dock_sweep/{box['name']}",
                position=[*((low + high) / 2 @ basis.T).tolist(), float(centre[2])],
                dims=[*(high - low + 2 * SWEEP_PADDING_M).tolist(), float(size[2] + 2 * SWEEP_PADDING_M)],
                rpy_deg=[0.0, 0.0, math.degrees(axis)],
            )
        )
    return boxes


def plan_ready(planner, geom, predicted, joints, route, mount):
    """Plan from the parked arm against the complete envelope and occupant layers."""
    if not planner.arm.gripper_state().position <= 0.2:
        return None
    chair, _ = planner.world.door_sweep_boxes()
    grasp = "side" if F.GRASP == "auto" else F.GRASP
    candidates = open_door.grasp_candidates(predicted, grasp)
    if not chair or not candidates:
        return None
    envelope = appliance_envelope(geom["closed_model"], route, mount)
    envelope.extend(
        dict(
            box,
            name=f"dock_sweep/{box['name']}",
            rpy_deg=box.get("rpy_deg", [0.0, 0.0, 0.0]),
            dims=(np.asarray(box["dims"]) + 2 * TRACK_TOL_M).tolist(),
        )
        for box in chair
    )
    for cand in candidates[:2]:
        model = FridgeModel(open_door.closed_door(predicted, cand["h"], cand.get("ref", cand["pos"]), cand))
        nominal_goal = open_door.approach_goals(model, cand, open_door.standoffs(cand)[0])[0]
        nominal, _ = planner.leg(nominal_goal, joints)
        if nominal is None or len(nominal.q) < 3:
            continue
        for fraction in (0.7, 0.45):
            q = nominal.q[int(fraction * (len(nominal.q) - 1))]
            point, quat = planner.tool(q)
            goal = pl.to_joints(q, point, quat, "dock ready", envelope, blend=0.0)
            ready, _ = planner.motion([goal], joints)
            if ready is not None:
                planner.join(ready, envelope, world_label="the appliance swept on the terminal leg")
                end = tuple(docking.route_samples(route[:-1]))[-1]
                return ReadyMotion(planner.arm, planner.fk, ready, end)
    say("  arm ready pose unavailable: no clear approach prefix through the terminal leg")
    return None


class ReadyMotion:
    """Stream a checked arm prefix from the same control loop that sends base twists."""

    def __init__(self, arm, fk, motion, route_at):
        self.arm = arm
        self.fk = fk
        plan = motion.joined
        self.t = np.r_[0.0, plan.t]
        self.q = np.vstack((motion.q0, plan.q))
        self.route_at = route_at
        self.elapsed = 0.0

    def _track(self, target, radians):
        actual = np.asarray(self.arm.joint_states().position, float)
        if (
            not np.isfinite(actual).all()
            or np.max(np.abs(actual - target)) > radians
            or np.max(np.linalg.norm(self.fk.joint_points(actual) - self.fk.joint_points(target), axis=1)) > TRACK_TOL_M
        ):
            raise BaseError("Arm departed the checked docking path")

    def _target(self, elapsed):
        return np.array([np.interp(elapsed, self.t, joint) for joint in self.q.T])

    @contextmanager
    def follow(self, clock):
        if not self.arm.gripper_state().position <= 0.2:
            yield None
            return
        self._track(self.q[0], START_TOL_RAD)
        with self.arm.stream("joint_position") as stream:
            start = clock()

            def step():
                self._track(self._target(self.elapsed), TRACK_TOL_RAD)
                self.elapsed = min(clock() - start, self.t[-1])
                stream.send(self._target(self.elapsed))

            say("  arm ready motion during final docking leg")
            yield step

    def park(self, ctx, clock, sleep, stationary):
        """Reverse the checked prefix before another base correction."""
        if self.elapsed == 0.0:
            return
        with self.arm.stream("joint_position") as stream:
            start = clock()
            last = self._target(self.elapsed)
            while True:
                if ctx.paused():
                    raise BaseError("Paused while parking the arm for another docking correction")
                stationary()
                self._track(last, TRACK_TOL_RAD)
                elapsed = max(0.0, self.elapsed - (clock() - start))
                last = self._target(elapsed)
                stream.send(last)
                if elapsed == 0.0:
                    # The parked-arm guard will check the measured joints before allowing the next twist.
                    until = clock() + 0.3
                    while clock() < until:
                        stationary()
                        stream.send(last)
                        if np.max(np.abs(np.asarray(self.arm.joint_states().position) - last)) <= START_TOL_RAD:
                            return
                        sleep(PERIOD_S)
                    raise BaseError("The arm did not return to its checked docking start")
                sleep(PERIOD_S)
