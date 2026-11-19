# ruff: noqa: E501, T201  -- PoC check, not capability code
"""The engine-side planner switch, offline: open_door._timed_plan with COMPARE_PLANNERS and each PLANNER, the run-log
notes it writes, and _replan_arc with each planner. usage: check_switch.py <urdf> <socket>"""

import sys

import numpy as np
from door_common import open_door as od
from setup_case import GEOM, MAX_SWING, MIN_SWING, Q0, after_swing, make

scene, rb = make(sys.argv[1])
od.IK_SOCKET = sys.argv[2]
notes = []
od.PLAN_LOG = lambda kind, **kw: notes.append((kind, kw))
od.COMPARE_PLANNERS, od.COMPARE_WORKERS = True, 8
for planner, workers in (("pybullet", 1), ("pybullet", 8), ("curobo", 1)):
    od.PLANNER, od.PLAN_WORKERS = planner, workers
    notes.clear()
    plan, why = od._timed_plan(scene, rb, GEOM, Q0, MAX_SWING, MIN_SWING, after_swing())
    print(f"RESULT {od.planner_name()}: why {why}; notes:")
    for kind, kw in notes:
        p = kw["plan"] or {}
        print(
            f"RESULT   {kind:12} {kw['planner']:6} total {kw['total_s']:.2f} s search {kw['search_s']} swing {p.get('swing')} tilt {p.get('pitch')} arc pts {len(p.get('arc_deg', []))}"
        )
    ser = [kw["plan"] for k, kw in notes if k == "plan_compare" and kw["planner"] == "serial"][0]
    pool = [kw["plan"] for k, kw in notes if k == "plan_compare" and kw["planner"] == "pool"][0]
    print(f"RESULT   pool plan == serial plan: {pool == ser}")
    wps = plan["wps"]
    ok, q_end, msg, arc = od._replan_arc(scene, rb, wps, plan["leg"][-1])
    print(
        f"RESULT   _replan_arc ({od.planner_name()}): ok {ok}, {len(arc)} pts, {msg}, max step {np.degrees(np.abs(np.diff(np.asarray([plan['leg'][-1], *arc]), axis=0))).max():.1f} deg"
    )
od.COMPARE_PLANNERS = False
