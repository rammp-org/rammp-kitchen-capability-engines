"""The arm's tool frame is its description's own.

The same arm is described two ways: the gripper bolted on the flange (flatbot's own description until
2026-11-17: every recorded run, and every test that replays one; arm_tool_frame 0.120 m out), and with
Kinova's sensor puck under it (flatbot's and gen_1_5's since rammp-deployments #149: the gripper's base
14 mm out, arm_tool_frame at 0.134). The planner plans arm_tool_frame of the description it is handed.
Until 2026-11-19 box_opening's chain stopped at the flange and held its tool frame 0.120 past it whatever
the description said: right on the first description only.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import SENSOR_PUCK_M as PUCK_M
from conftest import arm_poses, both_chains, flange_pose, flatbot_urdf, puck_urdf

from box_opening.constants import HOME, TOOL_FROM_FLANGE_STOCK_M
from box_opening.runtime import approach, planner
from box_opening.runtime.kinematics import TOOL_FROM_EE_M, ArmChain, GravityModel, mat_to_quat_xyzw

pytestmark = pytest.mark.usefixtures("approach_state")


def test_the_tool_frame_is_the_descriptions_own_and_nothing_moves_on_the_one_the_runs_were_flown_on(tmp_path):
    """The chain reads where the description hangs arm_tool_frame on the flange, where it held 0.120 for
    every description. On flatbot's 10-21 URDF that IS 0.120: the same chain to the last bit. With the
    sensor puck the tool frame is 14 mm further along the hand, the flange where it was."""
    plain, puck = both_chains(tmp_path)
    held = ArmChain(root="arm_mount", path=flatbot_urdf(), tool_from_tip_m=TOOL_FROM_EE_M)  # as it was built
    assert plain.tool_described and plain.tool_from_flange_m == TOOL_FROM_EE_M == TOOL_FROM_FLANGE_STOCK_M == 0.120
    assert puck.tool_described and puck.tool_from_flange_m == pytest.approx(0.134, abs=1e-12)
    for q in arm_poses():
        (R0, t0), (R1, t1), (R2, t2) = held.fk(q), plain.fk(q), puck.fk(q)
        assert np.array_equal(R0, R1) and np.array_equal(t0, t1)  # bit for bit: no recorded run replays differently
        assert np.array_equal(plain.tool_positions([q]), held.tool_positions([q]))
        assert np.allclose(R2, R1, atol=1e-12) and np.allclose(t2, t1 + PUCK_M * np.asarray(R1)[:, 2], atol=1e-12)
        Rf0, f0 = flange_pose(plain, q)
        Rf1, f1 = flange_pose(puck, q)
        assert np.allclose(f0, f1, atol=1e-12) and np.allclose(Rf0, Rf1, atol=1e-12)


def test_the_tool_frame_is_where_the_planner_plans_on_both_descriptions(tmp_path, monkeypatch):
    """The planner's ee_link is arm_tool_frame of the description it is handed (robot.yaml: urdf_path
    from_robot_description). A pose it plans ends with THAT frame on the pose, and the client refuses a
    pose plan that ends more than 6 mm from its pose by the arm's own FK (PLAN_END_M): with 0.120 held
    on the puck description every pose plan would have been refused as another goal's, 14 mm off."""
    from conftest import traj

    from box_opening.runtime.client import PlannerClient

    for chain, path in zip(both_chains(tmp_path), (flatbot_urdf(), puck_urdf(tmp_path)), strict=True):
        tree = GravityModel(path=path, root="arm_mount")  # walks every link of the URDF by its own joints
        monkeypatch.setattr(approach, "_chain", lambda chain=chain: chain)
        for q in arm_poses()[:6]:
            T = tree._frames(q)[0]["arm_tool_frame"]  # the planner's ee_link, off the description itself
            R, t = chain.fk(q)
            assert np.allclose(t, T[:3, 3], atol=1e-9) and np.allclose(R, T[:3, :3], atol=1e-9)
            plan = SimpleNamespace(success=True, trajectory=traj(HOME, q), message="")
            off, turn = planner.pose_end_error(plan.trajectory, T[:3, 3], mat_to_quat_xyzw(T[:3, :3]))
            assert off < 1e-8 and turn < 1e-6
            client = PlannerClient.__new__(PlannerClient)
            assert client._checked_pose(plan, T[:3, 3], mat_to_quat_xyzw(T[:3, :3])) is plan
    # what stood before: the tool frame held 0.120 out on the description with the puck
    held = ArmChain(root="arm_mount", path=puck_urdf(tmp_path), tool_from_tip_m=TOOL_FROM_EE_M)
    monkeypatch.setattr(approach, "_chain", lambda: held)
    q = arm_poses()[0]
    T = GravityModel(path=puck_urdf(tmp_path), root="arm_mount")._frames(q)[0]["arm_tool_frame"]
    off, _turn = planner.pose_end_error(traj(HOME, q), T[:3, 3], mat_to_quat_xyzw(T[:3, :3]))
    assert off == pytest.approx(PUCK_M, abs=1e-9) and off > PlannerClient.PLAN_END_M


def test_a_urdf_with_no_tool_frame_of_its_own_is_given_the_stock_grippers(tmp_path):
    """The bench's URDF stops at end_effector_link: its tool frame is the 2F-85's on the flange, 0.120."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(flatbot_urdf())
    root = tree.getroot()
    for j in root.findall("joint"):
        if j.get("name") == "arm_tool_frame_joint":
            root.remove(j)
    bare = tmp_path / "no_tool_frame.urdf"
    tree.write(bare)
    chain = ArmChain(root="arm_mount", path=str(bare))
    assert not chain.tool_described and chain.tool_from_flange_m == TOOL_FROM_EE_M
    q = arm_poses()[0]
    assert np.array_equal(chain.fk(q)[1], ArmChain(root="arm_mount", path=flatbot_urdf()).fk(q)[1])


@pytest.mark.parametrize("xyz, rpy", [("0.01 0 0.134", "0 0 0"), ("0 0 0.134", "0 0 1.5708"), ("0 0 -0.02", "0 0 0")])
def test_a_tool_frame_set_aside_or_turned_from_the_flanges_axis_is_refused(tmp_path, xyz, rpy):
    """Everything known of the hand is laid out along the flange's z from a tool frame with the flange's
    axes: a description that has it elsewhere gets no chain, so no run starts on it."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(flatbot_urdf())
    for j in tree.getroot().findall("joint"):
        if j.get("name") == "arm_tool_frame_joint":
            j.find("origin").set("xyz", xyz)
            j.find("origin").set("rpy", rpy)
    odd = tmp_path / "odd_tool_frame.urdf"
    tree.write(odd)
    with pytest.raises(RuntimeError, match="arm_tool_frame is not straight out of arm_end_effector_link"):
        ArmChain(root="arm_mount", path=str(odd))
