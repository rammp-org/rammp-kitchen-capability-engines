# ruff: noqa: E501, T201  -- PoC benchmark code, not capability code
"""Baseline: door_common.open_door.plan_open as it is (serial PyBullet). Writes the plan to argv[2]."""

import json
import sys
import time

from door_common import open_door
from setup_case import GEOM, MAX_SWING, MIN_SWING, Q0, after_swing, make, summary

scene, rb = make(sys.argv[1])
times = []
for _ in range(int(sys.argv[3]) if len(sys.argv) > 3 else 1):
    t = time.perf_counter()
    plan, why = open_door.plan_open(scene, rb, GEOM, Q0, MAX_SWING, MIN_SWING, after_swing())
    times.append(time.perf_counter() - t)
print(
    f"RESULT serial plan_open: {' '.join(f'{x:.2f}' for x in times)} s; swing {plan and plan['swing']}; why {why}"
)
json.dump({"times": times, "plan": summary(plan)}, open(sys.argv[2], "w"))
