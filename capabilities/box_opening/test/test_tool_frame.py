"""Box opening's tool frame stays where its heights were tuned; the planner's is the description's own.

The same arm is described two ways: the gripper bolted on the flange (flatbot's own description until
2026-11-17: every recorded run, and every test that replays one; arm_tool_frame 0.120 m out), and with
Kinova's sensor puck under it (flatbot's and gen_1_5's since rammp-deployments #149: the gripper's base
14 mm out, arm_tool_frame at 0.134). The puck was on the arm all along: the hand did not move when its
description did. Every height and offset in the package was settled by flight in a tool frame 0.120 m
out of the flange, so that frame is kept on every description and the flange stands for every target
where it stood when it flew. The planner plans the description's own tool frame: a pose handed to it
is moved by the rigid step between the two, and its plan's end is judged there.
"""

import xml.etree.ElementTree as ET
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import SENSOR_PUCK_M as PUCK_M
from conftest import arm_poses, both_chains, flange_pose, flatbot_urdf, puck_urdf, traj

from box_opening.constants import (
    HEIGHTS_POINT_FROM_FLANGE_M,
    HOME,
    TCP_OFFSET_M,
    TIP_TO_TOOL_M,
    TOOL_FROM_FLANGE_STOCK_M,
)
from box_opening.models.container import ContainerModel, load_press_demo
from box_opening.perception.d405 import quat_to_mat
from box_opening.runtime import approach, body, fingers, planner
from box_opening.runtime.client import PlannerClient
from box_opening.runtime.kinematics import TOOL_FROM_EE_M, ArmChain, GravityModel, ToolFrameError, mat_to_quat_xyzw
from box_opening.tasks import cli_common, press_demo
from box_opening.tasks import reach_press as rp

pytestmark = pytest.mark.usefixtures("approach_state")

PKG = Path(__file__).resolve().parents[1]
PINK = "capabilities/box_opening/config/containers/ankou_pink.yaml"
FLATBOT_WORLD = str(PKG / "config" / "world_flatbot.yaml")
CONTAINER = str(PKG / "config" / "containers" / "ankou_pink.yaml")


def _edited(tmp_path, name, edit, src=None):
    """A URDF in `tmp_path`: `src` (None: the description with the puck) with `edit(root, joints by name)` done."""
    tree = ET.parse(src or puck_urdf(tmp_path))
    root = tree.getroot()
    edit(root, {j.get("name"): j for j in root.findall("joint")})
    out = Path(tmp_path) / ("%s.urdf" % name)
    tree.write(out)
    return str(out)


def _hang_tool_on(parent, xyz, rpy="0 0 0", typ="fixed"):
    def edit(_root, joints):
        j = joints["arm_tool_frame_joint"]
        j.set("type", typ)
        j.find("parent").set("link", parent)
        j.find("origin").set("xyz", xyz)
        j.find("origin").set("rpy", rpy)
        if typ != "fixed":
            ET.SubElement(j, "axis", xyz="0 0 1")

    return edit


def _ee_of_the_urdf(path, q, link="arm_tool_frame"):
    """(R, t) of `link` at joints `q`, by a walk of every link of the URDF that knows nothing of the chain's frames."""
    T = GravityModel(path=path, root="arm_mount")._frames(q)[0][link]
    return T[:3, :3], T[:3, 3]


# ---- the frame the heights were tuned in ---------------------------------------------------------


def test_the_flange_stands_where_it_flew_for_every_target_whichever_description_the_run_is_on(tmp_path):
    """The chain's tool frame is 0.120 m out of the flange on both descriptions, so its FK and every
    joint solution of a target are the same numbers, bit for bit, as on the description every run flew
    on. With the tool frame taken from the description (0.134 with the puck) every target would stand
    the flange 14 mm further back along the hand than it flew: the fingertips 14 mm higher straight
    down, and at 30 deg of tilt 12 mm higher and 7 mm nearer the arm."""
    plain, puck = both_chains(tmp_path)
    for chain in (plain, puck):
        assert chain.tool_from_flange_m == TOOL_FROM_EE_M == TOOL_FROM_FLANGE_STOCK_M == 0.120
        assert chain.tool_from_flange_m + TCP_OFFSET_M == pytest.approx(HEIGHTS_POINT_FROM_FLANGE_M) == 0.131
    for q in arm_poses():
        (R0, t0), (R1, t1) = plain.fk(q), puck.fk(q)
        assert np.array_equal(R0, R1) and np.array_equal(t0, t1)
        assert np.array_equal(plain.tool_positions([q]), puck.tool_positions([q]))
        assert np.array_equal(plain._fk_jac(q)[2], puck._fk_jac(q)[2])
    # the reach-out's own targets for a button, at the tilts it presses at: the look pose, the stroke's start
    # (the pre-press hover) and its bottom, solved by each chain; and where that puts the flange
    model, travel = ContainerModel.load(PINK), load_press_demo(PINK).travel_m
    button = (0.66, -0.20, 0.31)
    for tilt in (0, 15, 30):
        quat, start, bottom, look = rp.reach_geometry(model, button, tilt, travel)
        got = []
        for chain in (plain, puck):
            q_look = chain.nearest_solution(list(HOME), look, quat)
            assert q_look is not None
            line, why = chain.straight_line(q_look, start, quat)
            assert why is None, why
            down, why = chain.straight_line(line[-1], bottom, quat)
            assert why is None, why
            got.append((q_look, line, down, flange_pose(chain, line[-1])[1]))
        assert np.array_equal(got[0][0], got[1][0])
        assert np.array_equal(np.asarray(got[0][1]), np.asarray(got[1][1]))
        assert np.array_equal(np.asarray(got[0][2]), np.asarray(got[1][2]))
        # the hover as it flew: the point the heights go by REACH_START_M over the button, the flange 131 mm
        # back along the hand from it (9 mm aside for the leaning pad's edge)
        R = np.asarray(quat_to_mat(*quat), float)
        want = np.asarray(button) + [0.0, 0.0, rp.REACH_START_M] - R @ [0.0, -0.009, HEIGHTS_POINT_FROM_FLANGE_M]
        assert np.allclose(got[1][3], want, atol=2e-4)


# ---- the planner's frame -------------------------------------------------------------------------


def test_a_pose_goes_to_the_planner_at_its_own_tool_frame_and_its_plans_end_is_judged_there(tmp_path, monkeypatch):
    """The planner plans arm_tool_frame of the description it is handed (robot.yaml ee_link). On the
    description with the puck that link is 14 mm along the hand from box opening's tool frame: the pose
    handed over is the one that puts box opening's frame where it was asked, and the plan that comes
    back is held to cuRobo's own success test at the planner's frame. Without the step the planner would
    bring ITS frame to the pose, the flange 14 mm off, and such a plan is refused."""
    plain, puck = both_chains(tmp_path)
    assert plain.ee_link == puck.ee_link == "arm_tool_frame" and plain.ee_named_by == "the arm's description"
    assert plain.ee_is_tool and not puck.ee_is_tool
    assert np.array_equal(puck.ee_in_tool[0], np.eye(3)) and np.allclose(puck.ee_in_tool[1], [0, 0, PUCK_M], atol=1e-12)
    for chain, path in ((plain, flatbot_urdf()), (puck, puck_urdf(tmp_path))):
        monkeypatch.setattr(approach, "_chain", lambda chain=chain: chain)
        client = PlannerClient.__new__(PlannerClient)
        client._plan_pose = None
        for q in arm_poses()[:6]:
            R, t = chain.fk(q)  # where box opening wants ITS tool frame
            quat = mat_to_quat_xyzw(R)
            Re, te = _ee_of_the_urdf(path, q)  # where that has the planner's link, off the description itself
            sent = []

            def call(_client, goal, q=q, sent=sent):
                sent.append(goal)
                return SimpleNamespace(success=True, trajectory=traj(HOME, q), message="")

            monkeypatch.setattr(client, "_call", call, raising=False)
            res = client._plan_pose_once(list(t), quat, list(HOME))
            assert res.success  # ends where it was asked, by the planner's frame
            p = sent[0].target.position
            assert np.allclose([p.x, p.y, p.z], te, atol=1e-9)
            o = sent[0].target.orientation
            assert np.allclose(quat_to_mat(o.x, o.y, o.z, o.w), Re, atol=1e-9)
            assert np.allclose(chain.ee_fk(q)[1], te, atol=1e-9) and np.allclose(chain.ee_fk(q)[0], Re, atol=1e-9)
            if chain is plain:  # one frame: the very numbers, untouched
                assert chain.planner_pose(t, quat)[0] is t and chain.planner_pose(t, quat)[1] is quat
                assert [p.x, p.y, p.z] == [float(v) for v in t]
    # a plan that brings the PLANNER's frame to the pose box opening meant for its own: the flange 14 mm off
    monkeypatch.setattr(approach, "_chain", lambda: puck)
    client = PlannerClient.__new__(PlannerClient)
    q = arm_poses()[0]
    R, t = puck.fk(q)
    q_off = puck.solve_pose(q, np.asarray(t) - PUCK_M * np.asarray(R)[:, 2], np.asarray(R))  # its ee on the pose
    assert q_off is not None
    monkeypatch.setattr(
        client,
        "_call",
        lambda _c, _g: SimpleNamespace(success=True, trajectory=traj(HOME, q_off), message=""),
        raising=False,
    )
    client._plan_pose = None
    res = client._plan_pose_once(list(t), mat_to_quat_xyzw(R), list(HOME))
    assert not res.success and "14.00 mm" in res.message and "not flown" in res.message
    # the run record's own measure of a pose plan stays in box opening's frame: what the hand does
    off, _turn = planner.pose_end_error(traj(HOME, q), t, mat_to_quat_xyzw(R))
    assert off < 1e-9


def test_the_planners_frame_is_found_through_whatever_fixed_joints_hang_it_on_the_flange(tmp_path):
    """The description's tool frame is the planner's wherever its fixed joints hang it: on the flange,
    on the puck's link, on the gripper's base. One that moves against the flange is refused, and no
    chain is built. A frame at the flange itself (a description with no gripper) or a hair aside is a
    frame like any other: only the step to it changes."""
    q = arm_poses()[1]
    same_place = {
        "on_the_puck": _hang_tool_on("arm_sensor_puck_link", "0 0 0.134"),
        "on_the_grippers_base": _hang_tool_on("arm_robotiq_85_base_link", "0 0 0.120"),
        "a_hair_aside": _hang_tool_on("arm_end_effector_link", "0 1e-7 0.134"),
    }
    for name, edit in same_place.items():
        path = _edited(tmp_path, name, edit)
        chain = ArmChain(root="arm_mount", path=path)
        assert chain.ee_link == "arm_tool_frame" and np.allclose(chain.ee_in_tool[1], [0, 0, PUCK_M], atol=1e-6), name
        assert np.array_equal(chain.fk(q)[1], ArmChain(root="arm_mount", path=flatbot_urdf()).fk(q)[1])
        assert np.allclose(chain.ee_fk(q)[1], _ee_of_the_urdf(path, q)[1], atol=1e-9)
    at_flange = ArmChain(
        root="arm_mount", path=_edited(tmp_path, "at_flange", _hang_tool_on("arm_end_effector_link", "0 0 0"))
    )
    assert np.allclose(at_flange.ee_in_tool[1], [0, 0, -0.120], atol=1e-12) and at_flange.tool_from_flange_m == 0.120
    turned_path = _edited(tmp_path, "turned", _hang_tool_on("arm_end_effector_link", "0.01 0 0.134", rpy="0 0 1.5708"))
    turned = ArmChain(root="arm_mount", path=turned_path)
    R, t = turned.fk(q)
    xyz, quat = turned.planner_pose(t, mat_to_quat_xyzw(R))
    Re, te = _ee_of_the_urdf(turned_path, q)
    assert np.allclose(xyz, te, atol=1e-9) and np.allclose(quat_to_mat(*quat), Re, atol=1e-6)
    # a tool frame that moves against the flange: no chain
    for name, edit in (
        ("on_a_joint_that_turns", _hang_tool_on("arm_end_effector_link", "0 0 0.134", typ="revolute")),
        ("on_the_forearm", _hang_tool_on("arm_forearm_link", "0 0 0.134")),
    ):
        with pytest.raises(
            ToolFrameError, match="arm_tool_frame .* cannot be placed from the flange arm_end_effector_link"
        ):
            ArmChain(root="arm_mount", path=_edited(tmp_path, name, edit))


def test_the_planners_own_config_says_which_link_it_plans(tmp_path):
    """Where the planner's robot file is read (the body model), its ee_link is the link: a description
    that names its tool frame otherwise is still handed poses at the right place, and one whose URDF
    cannot place that link from the flange stops the load, before anything moves."""
    from test_body import STOCK_HAND, _model_files

    def rename(root, joints):
        joints["arm_tool_frame_joint"].find("child").set("link", "arm_tcp")
        for link in root.findall("link"):
            if link.get("name") == "arm_tool_frame":
                link.set("name", "arm_tcp")

    path = _edited(tmp_path, "tcp", rename)
    chain = ArmChain(root="arm_mount", path=path)
    assert chain.ee_link is None and chain.ee_is_tool  # no link by the usual name: taken for the tool frame
    files = _model_files(
        tmp_path, [("far", (-1.0, -1.0, -1.0), (0.1, 0.1, 0.1))], robot=STOCK_HAND + "    ee_link: arm_tcp\n"
    )
    model = body.load(files)
    assert model.ee_link == "arm_tcp"
    model.check_for(chain, path, "arm_mount")
    assert chain.ee_link == "arm_tcp" and "the planner's robot file" in chain.ee_named_by
    assert np.allclose(chain.ee_in_tool[1], [0, 0, PUCK_M], atol=1e-12)
    # the usual link, named by the config: the same frame on the description without the puck
    files = _model_files(
        tmp_path, [("far", (-1.0, -1.0, -1.0), (0.1, 0.1, 0.1))], robot=STOCK_HAND + "    ee_link: arm_tool_frame\n"
    )
    plain = ArmChain(root="arm_mount", path=flatbot_urdf())
    body.load(files).check_for(plain, flatbot_urdf(), "arm_mount")
    assert plain.ee_is_tool and plain.ee_link == "arm_tool_frame"
    # a link this URDF does not have
    files = _model_files(
        tmp_path, [("far", (-1.0, -1.0, -1.0), (0.1, 0.1, 0.1))], robot=STOCK_HAND + "    ee_link: arm_grasp\n"
    )
    with pytest.raises(
        body.BodyModelError, match="the planner's tool frame arm_grasp .* cannot be placed from the flange"
    ):
        body.load(files).check_for(ArmChain(root="arm_mount", path=flatbot_urdf()), flatbot_urdf(), "arm_mount")


def test_a_urdf_with_no_tool_frame_of_its_own_takes_the_planners_frame_for_the_tool_frame(tmp_path):
    """The bench's URDF stops at end_effector_link: the planner's ee_link there is the 2F-85's on the
    flange, 0.120 m out, which is box opening's own."""

    def strip(root, joints):
        root.remove(joints["arm_tool_frame_joint"])

    chain = ArmChain(root="arm_mount", path=_edited(tmp_path, "no_tool_frame", strip, src=flatbot_urdf()))
    assert chain.ee_link is None and chain.ee_is_tool and chain.tool_from_flange_m == TOOL_FROM_EE_M
    q = arm_poses()[0]
    assert np.array_equal(chain.fk(q)[1], ArmChain(root="arm_mount", path=flatbot_urdf()).fk(q)[1])


# ---- a run on a description it cannot plan with ----------------------------------------------------


class _Client:
    joint_names = ["joint_%d" % i for i in range(1, 8)]

    def __init__(self):
        self._base_frame = None

    def joints(self):
        return list(HOME)


def test_a_run_stops_before_anything_moves_on_a_tool_frame_the_urdf_does_not_fix_to_the_flange(
    tmp_path, monkeypatch, capfd
):
    """With the body check on, the body model's load builds the chain and stops the run. With it off
    (a bench, body_layers "") nothing built the chain before the first motion needed it: the run now
    stops where it says its hand, whichever CLI it is."""
    import sys

    odd = _edited(tmp_path, "odd", _hang_tool_on("arm_end_effector_link", "0 0 0.134", typ="revolute"))
    monkeypatch.setenv("BOX_OPENING_STATE", str(tmp_path))
    monkeypatch.setattr(cli_common, "init_runtime", lambda execute: (object(), _Client()))
    monkeypatch.setattr(cli_common, "_arm_urdf_path", lambda node, frame: odd)
    called = []
    monkeypatch.setattr(press_demo, "home_first", lambda *a, **k: called.append("home_first") or True)
    monkeypatch.setattr(press_demo.Runner, "run", lambda self, *a, **k: called.append("run") or [])
    monkeypatch.setattr(sys, "argv", ["press_demo", "--container", CONTAINER, "--bench-world", FLATBOT_WORLD])
    with pytest.raises(
        SystemExit, match=r"\[press_demo\] not starting: .*arm_tool_frame .*cannot be placed.*Nothing moved"
    ):
        press_demo.main()
    assert called == []
    with pytest.raises(SystemExit, match="not starting: .*arm_tool_frame .*cannot be placed.*Nothing moved"):
        cli_common.build_ctx(Namespace(container=CONTAINER, bench_world=FLATBOT_WORLD, execute=False))
    # and on a description it can plan with, both say where the hand is before they go on
    monkeypatch.setattr(cli_common, "_arm_urdf_path", lambda node, frame: puck_urdf(tmp_path))
    capfd.readouterr()
    cli_common.build_ctx(Namespace(container=CONTAINER, bench_world=FLATBOT_WORLD, execute=False))
    assert " hand: tool frame 120 mm out of the flange" in capfd.readouterr().out
    monkeypatch.setattr(cli_common, "load_body_or_exit", lambda *a, **k: sys.exit("the test ends here"))
    with pytest.raises(SystemExit, match="the test ends here"):
        press_demo.main()
    assert "[press_demo] hand: tool frame 120 mm out of the flange" in capfd.readouterr().out and called == []


# ---- what rides the gripper ------------------------------------------------------------------------


def test_the_finger_tip_links_ride_the_gripper_and_are_read_where_the_description_has_them(tmp_path, monkeypatch):
    """A touch's height is read off the finger-tip links in TF (client.contact_xyz). They ride the
    gripper: 8 mm short of the tool frame with the gripper on the flange, 6 mm PAST it on a description
    with the puck. The contact point worked out from them is the same point of the hand either way."""
    plain, puck = both_chains(tmp_path)
    assert fingers.gripper_out_m(plain) == 0.0 and fingers.tip_to_tool_m(plain) == TIP_TO_TOOL_M
    assert fingers.gripper_out_m(puck) == pytest.approx(PUCK_M, abs=1e-12)
    assert fingers.tip_to_tool_m(puck) == pytest.approx(TIP_TO_TOOL_M - PUCK_M, abs=1e-12)
    assert fingers.gripper_out_m(object()) == 0.0  # a stand-in chain: the gripper on the flange
    model = ContainerModel.load(PINK)
    quat = rp.reach_quat(model, (0.6, -0.2), 30)
    R = np.asarray(quat_to_mat(*quat), float)
    contact = np.array([0.6, -0.2, 0.31])
    tool = np.asarray(rp.tool_for_contact(contact, quat))
    for chain, path in ((plain, flatbot_urdf()), (puck, puck_urdf(tmp_path))):
        monkeypatch.setattr(approach, "_chain", lambda chain=chain: chain)
        monkeypatch.setattr(approach, "chain_ready", lambda: (True, ""))
        q = chain.solve_pose(chain.nearest_solution(list(HOME), tool, quat), tool, R)
        frames = GravityModel(path=path, root="arm_mount", gripper_rad=0.8)._frames(q)[0]
        tips = 0.5 * (
            frames["arm_robotiq_85_left_finger_tip_link"][:3, 3] + frames["arm_robotiq_85_right_finger_tip_link"][:3, 3]
        )
        # (the links' own origins shut are 0.2 mm off the 8 mm the constant has had since 2026-09-03)
        assert np.allclose(rp.contact_from_tips(tips, quat), contact, atol=5e-4)
