# ruff: noqa: E501, T201  -- PoC benchmark code, not capability code
"""PoC 1: plan_open with its tilts searched in a fork pool (open_door.PLAN_WORKERS). Checks the plan is the serial
one, bit for bit. usage: bench_pool.py <urdf> <serial.json> <out.json> <workers> [repeats]"""

import json
import sys
import time

from door_common import open_door
from setup_case import GEOM, MAX_SWING, MIN_SWING, Q0, after_swing, make, summary

scene, rb = make(sys.argv[1])
serial = json.load(open(sys.argv[2]))["plan"]
open_door.PLAN_WORKERS = int(sys.argv[4])
times = []
for _ in range(int(sys.argv[5]) if len(sys.argv) > 5 else 1):
    t = time.perf_counter()
    plan, why = open_door.plan_open(scene, rb, GEOM, Q0, MAX_SWING, MIN_SWING, after_swing())
    times.append(time.perf_counter() - t)
same = summary(plan) == serial
print(
    f"RESULT pool x{open_door.PLAN_WORKERS}: {' '.join(f'{x:.2f}' for x in times)} s; swing {plan and plan['swing']};"
    f" identical to serial: {same}"
)
json.dump({"times": times, "plan": summary(plan), "identical": same}, open(sys.argv[3], "w"))
