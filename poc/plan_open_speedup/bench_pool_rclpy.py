# ruff: noqa: E501, T201  -- PoC benchmark code, not capability code
"""PoC 1 under the engine's threading: plan_open with PLAN_WORKERS forked while an rclpy node spins on a
MultiThreadedExecutor in background threads (timers and a publisher busy the whole time), as in a capability engine.
Checks the plan is the serial one and no fork hangs. usage: bench_pool_rclpy.py <urdf> <serial.json> <workers> [repeats]"""

import json
import sys
import threading
import time

import rclpy
from door_common import open_door
from rclpy.executors import MultiThreadedExecutor
from setup_case import GEOM, MAX_SWING, MIN_SWING, Q0, after_swing, make, summary
from std_msgs.msg import String

rclpy.init()
node = rclpy.create_node("pool_fork_check")
pub = node.create_publisher(String, "~/chatter", 10)
ticks = [0]


def tick():
    ticks[0] += 1
    pub.publish(String(data=str(ticks[0])))
    node.get_logger().debug("tick")


node.create_timer(0.005, tick)
node.create_timer(0.007, tick)
ex = MultiThreadedExecutor(num_threads=4)
ex.add_node(node)
threading.Thread(target=ex.spin, daemon=True).start()

scene, rb = make(sys.argv[1])
serial = json.load(open(sys.argv[2]))["plan"]
open_door.PLAN_WORKERS = int(sys.argv[3])
for _ in range(int(sys.argv[4]) if len(sys.argv) > 4 else 1):
    t0, k0 = time.perf_counter(), ticks[0]
    plan, why = open_door.plan_open(scene, rb, GEOM, Q0, MAX_SWING, MIN_SWING, after_swing())
    dt = time.perf_counter() - t0
    print(
        f"RESULT rclpy-threaded pool x{open_door.PLAN_WORKERS}: {dt:.2f} s; identical to serial: {summary(plan) == serial};"
        f" executor ticks during it {ticks[0] - k0}"
    )
ex.shutdown()
rclpy.shutdown()
