# ruff: noqa: E501, T201  -- PoC benchmark code, not capability code
"""Serial PyBullet vs batched cuRobo on the same case: the candidates each finds and the chosen plan's paths,
measured the same way for both (the sim's FK for pose error, cuRobo's spheres for self-collision).
usage: compare.py <urdf> <out.json>   (needs the IK server)"""

import json
import sys
import time

import numpy as np
from capability_engine.sim import tool_pose
from door_common import open_door as od
from ik_client import CuroboIK
from plan_open_batched import plan_open_batched
from scipy.spatial.transform import Rotation as R
from setup_case import GEOM, MAX_SWING, MIN_SWING, Q0, after_swing, make

scene, rb = make(sys.argv[1])
ik = CuroboIK()

seen = {}
search = od._search_tilts


def capture(*a, **k):
    seen["cands"] = search(*a, **k)
    return seen["cands"]


od._search_tilts = capture
t = time.perf_counter()
serial, _ = od.plan_open(scene, rb, GEOM, Q0, MAX_SWING, MIN_SWING, after_swing())
t_serial = time.perf_counter() - t
serial_c = [(c["side"], c["pitch"], c["swing"], round(c["margin"], 1)) for c in seen["cands"]]
st = {}
batched, _ = plan_open_batched(
    scene, rb, GEOM, Q0, MAX_SWING, MIN_SWING, after_swing(), ik, stats=st
)


def metrics(plan):
    """The chosen plan's line-up -> straight-in -> swing joint path, against the poses it was asked for."""
    g = plan["grasp"]
    tg = [np.asarray(x) for x in np.linspace(plan["pre_p"], g.position, len(plan["leg"]) + 1)[1:]]
    targets = [(x, np.asarray(g.orientation)) for x in tg] + [
        (np.asarray(w[:3]), np.asarray(w[3:])) for w in plan["wps"]
    ]
    qs = [np.asarray(q) for q in [*plan["leg"], *plan["arc"]]]
    pe, re = [], []
    for q, (pos, quat) in zip(qs, targets, strict=True):
        p_, q_ = tool_pose(scene, rb, q)
        pe.append(np.linalg.norm(p_ - pos) * 1000)
        re.append(np.degrees((R.from_quat(q_).inv() * R.from_quat(quat)).magnitude()))
    arc = np.degrees(np.asarray([plan["leg"][-1], *plan["arc"]]))
    leg = np.degrees(np.asarray([plan["q_pre"], *plan["leg"]]))
    steps_arc = np.abs(np.diff(arc, axis=0))
    path = np.asarray([plan["q_pre"], *plan["leg"], *plan["arc"]])
    sc = ik.self_collision(path)
    return {
        "tilt": plan["pitch"],
        "side": plan["side"],
        "swing": plan["swing"],
        "standoff": plan["standoff"],
        "guard_margin_deg": round(plan["margin"], 1),
        "pos_err_mm_max": round(max(pe), 2),
        "pos_err_mm_mean": round(float(np.mean(pe)), 2),
        "rot_err_deg_max": round(max(re), 2),
        "rot_err_deg_mean": round(float(np.mean(re)), 2),
        "swing_max_step_deg": round(float(steps_arc.max()), 2),
        "swing_joint_travel_deg": round(float(steps_arc.sum()), 1),
        "straight_in_max_step_deg": round(
            float(np.abs((np.diff(leg, axis=0) + 180) % 360 - 180).max()), 2
        ),
        "self_collision_max": round(float(np.max(sc)), 4),
        "n_waypoints": len(qs),
    }


out = {
    "serial": {"time_s": round(t_serial, 2), "candidates": serial_c, "plan": metrics(serial)},
    "batched": {
        "time_s": round(st["total_s"], 2),
        "candidates": st["candidates"],
        "plan": metrics(batched),
        "stats": {k: v for k, v in st.items() if k != "candidates"},
    },
}
for k in ("serial", "batched"):
    print(
        f"RESULT {k}: {out[k]['time_s']} s, {len(out[k]['candidates'])} candidates, plan {out[k]['plan']}"
    )
s_set = {c[:3] for c in serial_c}
b_set = {c[:3] for c in st["candidates"]}
print(
    f"RESULT (side, tilt, swing) found by both {len(s_set & b_set)}, serial only {sorted(s_set - b_set)},"
    f" batched only {sorted(b_set - s_set)}"
)
json.dump(out, open(sys.argv[2], "w"), indent=1, default=float)
