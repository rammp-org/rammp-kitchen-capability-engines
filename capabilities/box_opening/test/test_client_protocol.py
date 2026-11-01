import inspect

from box_opening.runtime.client import PlannerClient


def test_protocol_surface():
    required = {
        "joints": [],
        "wrist_efforts": [],
        "efforts_present": [],
        "plan_to_pose": ["xyz", "quat_xyzw", "start_joints"],
        "plan_to_joints": ["q7", "start_joints"],
        "execute": ["traj", "speed", "guard"],
        "set_world": ["path_or_name"],
        "motion_enabled": [],
        "gripper_cmd": ["position"],
        "gripper_send": ["position"],
        "gripper_join": ["handle"],
        "contact_xyz": ["timeout_s"],
        # reach_press asks for it by name (getattr): renamed, nothing would prime
        "prime_compliant": ["q"],
        # the runner asks for them by name (getattr): renamed, no arrival's
        # settle would be noted
        "joint_history": ["since"],
        "goals_sent": [],
    }
    for name, params in required.items():
        fn = getattr(PlannerClient, name)
        sig = list(inspect.signature(fn).parameters)[1:]  # drop self
        for p in params:
            assert p in sig, "%s missing param %s" % (name, p)
