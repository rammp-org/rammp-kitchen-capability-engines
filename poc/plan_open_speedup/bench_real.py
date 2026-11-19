# ruff: noqa: E501, T201  -- PoC benchmark code, not capability code
"""plan_open as the engine runs it on flatbot: the chair-and-user collision check on (node.py enables it), the swing
tried from SWING_MAX_TRY_DEG down, and the door from a real detection (snorlax, 2026-11-19 16:11, arm at the view pose).
Times serial, pool and curobo, and where curobo's time goes. usage: bench_real.py <urdf> <socket> [chair 0/1]"""

import sys
import time

import numpy as np
from door_common import common as dc
from door_common import open_door as od
from door_common.curobo_ik import plan_open_curobo
from setup_case import HANDLE_QUAT, after_swing, make

GEOM = {  # 20261119_161112_open's detection (the engine's log)
    "h_raw": np.array([0.6271, -0.3542, 0.4128]),
    "orient": HANDLE_QUAT,
    "n": np.array([-0.999, 0.033, 0.0]),
    "hinge": np.array([0.6842, 0.0032, 0.4138]),
    "hinge_seen": np.array([0.6658, -0.0171, 0.4138]),
    "door_z": np.array([0.302, 0.557]),
}
Q0 = np.radians((21.4, -43.8, -92.2, -110.6, -58.8, -93.8, -165.7))  # the view pose

scene, rb = make(sys.argv[1])
if len(sys.argv) < 4 or sys.argv[3] != "0":
    dc.enable_chair_collision()
calls = [0, 0.0]
from capability_engine import sim  # noqa: E402

_ko = sim._KEEP_OUT
if _ko is not None:

    def counted(rb_, q):
        t0 = time.perf_counter()
        try:
            return _ko(rb_, q)
        finally:
            calls[0] += 1
            calls[1] += time.perf_counter() - t0

    sim.set_keep_out(counted)
print(
    f"chair collision {'on' if _ko else 'off'}; swing {od.SWING_MAX_TRY_DEG} down to {od.SWING_MIN_DEG}"
)


def run(name, fn):
    calls[0], calls[1] = 0, 0.0
    t0 = time.perf_counter()
    plan, why = fn()
    dt = time.perf_counter() - t0
    p = f"tilt {plan['pitch']} side {plan['side']} swing {plan['swing']}" if plan else why
    print(f"{name:7} {dt:6.2f} s  keep-out checks {calls[0]:6d} ({calls[1]:.2f} s)  -> {p}")
    return plan


args = (scene, rb, GEOM, Q0, od.SWING_MAX_TRY_DEG, od.SWING_MIN_DEG, after_swing())
od.PLAN_WORKERS = 1
serial = run("serial", lambda: od.plan_open(*args))
od.PLAN_WORKERS = 8
pool = run("pool", lambda: od.plan_open(*args))
od.PLAN_WORKERS = 1
od.IK_SOCKET = sys.argv[2]
ik = od._curobo_ik()
st = {}
cur = run("curobo", lambda: plan_open_curobo(*args, ik, st))
print(
    "curobo stages: "
    + ", ".join(f"{k} {st[k]:.2f} s" for k in ("a_s", "l_s", "s_s") if k in st)
    + f"; IK {ik.calls} calls, {ik.problems} problems, {ik.wall_ms / 1e3:.2f} s wall, {ik.gpu_ms / 1e3:.2f} s GPU"
)

same = lambda a, b: all(np.allclose(a[k], b[k]) for k in ("q_pre", "leg", "arc"))  # noqa: E731
print(f"curobo plan == pool plan: {same(cur, pool)}")
# the fallback approach (_CuroboFirst._other_approach) re-plans the swing with the planner in use
for planner in ("pybullet", "curobo"):
    od.PLANNER = planner
    new = od._with_approach(
        scene, rb, cur, (cur["standoff"], cur["pre_p"], cur["q_pre"], cur["leg"])
    )
    print(f"_with_approach, {planner}: {new and new['msg']}")
od.PLANNER = "pybullet"
