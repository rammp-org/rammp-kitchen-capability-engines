# ruff: noqa: E501, T201  -- PoC benchmark code, not capability code
"""PoC 2 benchmark: plan_open_batched against the cuRobo IK server. usage: bench_batched.py <urdf> <out.json> [repeats]"""

import json
import sys
import time

from door_common.curobo_ik import CuroboIK
from door_common.curobo_ik import plan_open_curobo as plan_open_batched
from setup_case import GEOM, MAX_SWING, MIN_SWING, Q0, after_swing, make, summary

scene, rb = make(sys.argv[1])
ik = CuroboIK("/data/ik.sock")
runs = []
for _ in range(int(sys.argv[3]) if len(sys.argv) > 3 else 1):
    ik.calls, ik.gpu_ms, ik.wall_ms, ik.problems = 0, 0.0, 0.0, 0
    st = {}
    t = time.perf_counter()
    plan, why = plan_open_batched(
        scene, rb, GEOM, Q0, MAX_SWING, MIN_SWING, after_swing(), ik, stats=st
    )
    st["wall_s"] = time.perf_counter() - t
    st.update(
        ik_calls=ik.calls, ik_problems=ik.problems, ik_gpu_ms=ik.gpu_ms, ik_wall_ms=ik.wall_ms
    )
    runs.append(st)
    print(
        f"RESULT batched {st['wall_s']:.2f} s (IK {ik.calls} calls, {ik.problems} problems, gpu {ik.gpu_ms:.0f} ms,"
        f" round trip {ik.wall_ms:.0f} ms; grasps {st['a_s']:.2f} s, approaches {st['l_s']:.2f} s, swings {st['s_s']:.2f} s);"
        f" swing {plan and plan['swing']}; {len(st.get('candidates', []))} candidates; why {why}"
    )
print("RESULT candidates", runs[-1].get("candidates"))
json.dump({"runs": runs, "plan": summary(plan)}, open(sys.argv[2], "w"))
