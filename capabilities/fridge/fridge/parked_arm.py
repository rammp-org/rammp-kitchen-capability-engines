"""CPU collision spheres from the deployed cuRobo model and published robot URDF."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from .kinematics import JOINTS, ROOT, _floats

# Two stationary Nov 16 TF captures varied at most 0.000054° per joint.
# Allow 0.1° per joint for encoder repeatability and unmeasured parked motion.
JOINT_ALLOWANCE = math.radians(0.1)
# Cover the full URDF range of the gripper (cuRobo normally locks it open).
GRIPPER_STEP = 0.1  # rad, with a separate between-sample displacement bound


class ParkedArm:
    def __init__(self, planner_config, urdf):
        path = Path(planner_config)
        planner = yaml.safe_load(path.read_text())
        path = path.parent / planner["robot"]
        kin = yaml.safe_load(path.read_text())["robot_cfg"]["kinematics"]
        if kin["base_link"] != ROOT or kin["urdf_path"] != "from_robot_description":
            raise ValueError("collision sphere model uses a different arm frame or URDF")
        self.buffer = float(kin["collision_sphere_buffer"])
        if not math.isfinite(self.buffer) or self.buffer < 0:
            raise ValueError("invalid collision sphere buffer")

        groups = kin["collision_spheres"]
        if isinstance(groups, str):
            groups = yaml.safe_load((path.parent / groups).read_text())["collision_spheres"]
        links = kin["collision_link_names"]
        if not links or set(links) != set(groups):
            raise ValueError("incomplete collision sphere groups")
        self.groups = []
        joints = ET.fromstring(urdf).findall("joint")
        by_child = {joint.find("child").get("link"): joint for joint in joints}
        by_name = {joint.get("name"): joint for joint in joints}
        ((gripper, parked),) = kin["lock_joints"].items()
        limit = by_name[gripper].find("limit")
        lo, hi = float(limit.get("lower")), float(limit.get("upper"))
        if not np.isfinite([lo, hi, parked]).all() or not lo <= parked <= hi or not lo < hi:
            raise ValueError("invalid parked gripper range")
        count = math.ceil((hi - lo) / GRIPPER_STEP)
        self.gripper = np.linspace(lo, hi, count + 1)
        active = [*JOINTS, gripper]
        covered = set()

        for link in links:
            spheres = groups[link]
            centres = np.asarray([s["center"] for s in spheres], float)
            radii = np.asarray([s["radius"] for s in spheres], float)
            if (
                centres.shape != (len(radii), 3)
                or not len(radii)
                or not np.isfinite(centres).all()
                or not np.all(np.isfinite(radii) & (radii > 0))
            ):
                raise ValueError(f"invalid collision spheres for {link}")
            reach = np.linalg.norm(centres, axis=1)
            allowance = np.zeros(len(radii))
            steps = []
            visited = set()
            while link != ROOT:
                if link in visited:
                    raise ValueError("cyclic URDF link chain")
                visited.add(link)
                joint = by_child[link]
                link = joint.find("parent").get("link")
                origin = joint.find("origin")
                xyz = _floats(origin, "xyz", "0 0 0")
                orient = Rotation.from_euler("xyz", _floats(origin, "rpy", "0 0 0"))
                kind = joint.get("type")
                axis, index, scale, offset = None, None, 1.0, 0.0
                if kind in ("continuous", "revolute"):
                    driver = joint.get("name")
                    mimic = joint.find("mimic")
                    if mimic is not None:
                        driver = mimic.get("joint")
                        scale = float(mimic.get("multiplier", "1"))
                        offset = float(mimic.get("offset", "0"))
                    index = active.index(driver)
                    covered.add(driver)
                    axis = _floats(joint.find("axis"), "xyz", "1 0 0")
                    length = np.linalg.norm(axis)
                    if not np.isfinite([length, scale, offset]).all() or length <= 0:
                        raise ValueError("invalid collision link joint")
                    axis /= length
                    half_step = (hi - lo) / (2 * count) if driver == gripper else JOINT_ALLOWANCE
                    allowance += abs(scale) * half_step * reach
                elif kind != "fixed":
                    raise ValueError(f"unsupported collision link joint {kind}")
                reach += np.linalg.norm(xyz)
                steps.append((xyz, orient, axis, index, scale, offset))
            moving_fingers = any(step[3] == len(JOINTS) for step in steps)
            self.groups.append((centres, radii + self.buffer + allowance, steps, moving_fingers))
        if covered != set(active):
            raise ValueError("collision model does not cover all articulated joints")

    def spheres(self, q):
        q = np.asarray(q, float)
        if q.shape != (len(JOINTS),) or not np.isfinite(q).all():
            raise ValueError("invalid parked joints")
        clouds = []
        for centres, radii, steps, moving_fingers in self.groups:
            for finger in self.gripper if moving_fingers else (self.gripper[0],):
                positions = centres
                joints = [*q, finger]
                for xyz, orient, axis, index, scale, offset in steps:
                    if axis is not None:
                        positions = Rotation.from_rotvec(axis * (scale * joints[index] + offset)).apply(positions)
                    positions = orient.apply(positions) + xyz
                clouds.append(np.column_stack((positions, radii)))
        return np.concatenate(clouds)
