"""Home with the door OPEN (after the push to 90): the hand straight off the door's inner face (spare-joint IK, impedance),
then cuRobo home with its world's door and handle turned to the door's angle about the hinge pin (the fridge body
unchanged). Run: ./side_docker.sh home_open_door.py /work/fridge_open_home.json <door deg>"""
import hashlib, json, os, sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from capability_engine import sim
urdf = open("/work/robot.urdf").read(); sim.set_model_provider(lambda: urdf)
import rclpy
import rammp_arm
from door_common import common as dc
from door_common import open_door as shared
from fridge import open_door, side_grasp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # the helpers beside this script; /work holds the data
import pull_lib
from pull_back_gains import HARDEST

AWAY_M, STEP_M = 0.12, 0.02
open_door.configure("/tmp/side_general", 30, True, False); sim.set_keep_out(None); shared.configure(HANDLE_SOURCE="sam3")
home = json.load(open(sys.argv[1]))["joints_rad"]
deg = float(sys.argv[2])
p = json.load(open("/work/side_plan_general.json"))
n = np.array([p["normal"][0], p["normal"][1], 0.0]); n /= np.linalg.norm(n)
hinge, sgn = np.asarray(p["hinge"], float), int(p["hinge_sign"])
turn = R.from_euler("z", -sgn * np.radians(deg))
inner_away = -turn.apply(n)  # off the open door's inner face, away from it
# cuRobo's world: the planner's own boxes for this detection, the door + handle turned about the hinge pin
world_path, obs = side_grasp.side_world(p["door"], np.asarray(p["handle"], float), n)
moving = ("fridge_door", "handle_bar", "handle_mount_top", "handle_mount_bottom")
lines = ["# fridge side grasp: the door OPEN %.0f deg (home_open_door.py)" % deg, "base_frame: arm_mount", "obstacles:"]
for o in obs:
    pos, rpy = np.asarray(o["position"], float), list(o["rpy_deg"])
    if o["name"] in moving:
        pos[:2] = hinge[:2] + turn.apply(np.r_[pos[:2] - hinge[:2], 0.0])[:2]
        rpy = [rpy[0], rpy[1], round(rpy[2] - sgn * deg, 3)]
    lines += [f"- name: {o['name']}", f"  position: {[round(float(v), 4) for v in pos]}", f"  rpy_deg: {rpy}",
              f"  dims: {o['dims']}"]
lines += ["objects: []", "targets: []", ""]
text = "\n".join(lines)
wpath = os.path.join(os.path.dirname(world_path), f"fridge_side_open-{hashlib.sha1(text.encode()).hexdigest()[:10]}.yaml")
open(wpath, "w").write(text)
scene, rb = dc.make_sim()
rclpy.init(); node = rclpy.create_node("fridge_home_open_door_cu")
import threading
from rclpy.executors import MultiThreadedExecutor
ex = MultiThreadedExecutor(); ex.add_node(node); threading.Thread(target=ex.spin, daemon=True).start()
cu = side_grasp.CuroboClient(node)
ai = rammp_arm.RammpArm("fridge_home_open_door")
try:
    st = ai.get_state(); ee = np.asarray(st["ee_pos"], float); q = np.asarray(st["position"], float)
    k = int(np.ceil(AWAY_M / STEP_M))
    poses = [(ee[:3] + inner_away * AWAY_M * i / k, ee[3:7]) for i in range(1, k + 1)]
    knots, worst, why = pull_lib.swing_knots_ns(scene, rb, q, poses)
    if knots is None:
        raise SystemExit(f"not moving: off the door doesn't solve: {why}")
    sc = min((pull_lib.self_clearance(rb, kq) for kq in knots), key=lambda t: t[0])
    if sc[0] < 0.04:
        raise SystemExit(f"not moving: off the door folds the arm ({sc[0] * 100:.1f} cm, {sc[1]})")
    print(f"before: tool {np.round(ee[:3], 3)}; MOVING: {AWAY_M * 100:.0f} cm off the open door's inner face along "
          f"{np.round(inner_away, 3)} (impedance)", flush=True)
    ai.move_joints(knots, cart_lengths=[STEP_M] * len(knots), impedance=HARDEST, speed_scale=1.0)
    print("off: tool", np.round(ai.get_state()["ee_pos"][:3], 3), flush=True)
    cu.set_world(wpath)
    print(f"cuRobo world: the door {deg:.0f} deg open ({os.path.basename(wpath)}); MOVING: cuRobo home ...", flush=True)
    ok = ai.go_to_joint_config(home, "home", speed_scale=0.5)
    st = ai.get_state()
    print("after:", ok, "tool", np.round(st["ee_pos"][:3], 3), "joints", np.round(np.degrees(st["position"]), 1))
finally:
    ai.close()
