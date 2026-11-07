"""cuRobo home with its world's door CLOSED again (side_grasp.side_world of the last plan's detection).
Run: ./side_docker.sh home_closed.py /work/fridge_open_home.json"""
import json, sys, threading
import numpy as np
from capability_engine import sim
urdf = open("/work/robot.urdf").read(); sim.set_model_provider(lambda: urdf)
import rclpy
from rclpy.executors import MultiThreadedExecutor
import rammp_arm
from door_common import open_door as shared
from fridge import open_door, side_grasp
open_door.configure("/tmp/side_general", 30, True, False); sim.set_keep_out(None); shared.configure(HANDLE_SOURCE="sam3")
home = json.load(open(sys.argv[1]))["joints_rad"]
p = json.load(open("/work/side_plan_general.json"))
n = np.array([p["normal"][0], p["normal"][1], 0.0]); n /= np.linalg.norm(n)
world, _ = side_grasp.side_world(p["door"], np.asarray(p["handle"], float), n)
rclpy.init(); node = rclpy.create_node("fridge_home_closed_cu")
ex = MultiThreadedExecutor(); ex.add_node(node); threading.Thread(target=ex.spin, daemon=True).start()
side_grasp.CuroboClient(node).set_world(world)
ai = rammp_arm.RammpArm("fridge_home_closed")
try:
    st = ai.get_state()
    print("before: tool", np.round(st["ee_pos"][:3], 3), "; cuRobo world: the door closed; MOVING: cuRobo home ...", flush=True)
    ok = ai.go_to_joint_config(home, "home", speed_scale=0.5)
    st = ai.get_state()
    print("after:", ok, "tool", np.round(st["ee_pos"][:3], 3), "joints", np.round(np.degrees(st["position"]), 1))
finally:
    ai.close()
