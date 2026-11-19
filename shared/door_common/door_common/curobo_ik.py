"""Batched cuRobo IK for plan_open (open_door.PLANNER = "curobo"): a PROOF OF CONCEPT for side-by-side tests, not
for merge. The IK comes from a standalone cuRobo process over a Unix socket (poc/plan_open_speedup/curobo_ik_server.py),
standing in for a batched IK service the planner doesn't have yet.

plan_open's search with its IK solved by cuRobo in batches instead of ~5,000 PyBullet solves one by one.

Same candidates, same gates, same pick as door_common.open_door.plan_open; only where the joints come from changes.
PyBullet chains each straight-in step and swing waypoint from the one before, and that seeding matters (seeding a
whole swing from the grasp joints loses most of them to the jump gate), so the batches keep it: every candidate's
path advances one waypoint per call, all candidates in the same call.

  A. one call: every grasp tilt x every configuration seed (_config_seeds + GRASP_EXTRA_SEEDS_DEG), each seeded with
     its seed -> the distinct grasp configurations, gated as _ik_within_guards and deduplicated as there;
  L. one call: every configuration's line-up at every standoff, seeded with its grasp joints; then the straight-ins,
     one call per step (~13), each seeded with the step before;
  S. the swings (each swing angle tried) from each configuration's straight-in end, one call per waypoint (~16).

A path drops out at the first step that fails a gate (IK error from cuRobo's own FK, jump, J4/J6, wrap, reach,
keep-out), then the trajectory keep-out checks run on the survivors, all in plan_open's order.
"""

import time
from multiprocessing.connection import Client

import numpy as np
from capability_engine import sim
from capability_engine.sim import CONTINUOUS, continuous_ok, short_delta, wrap_joints

from door_common import common as dc
from door_common import open_door as od
from door_common.common import Refused, keep_out_message, quiet


class CuroboIK:
    """The IK server's client: ik() batched IK, fk(), self_collision(); counts calls and time."""

    def __init__(self, sock, wait_s=300):
        self.sock, self.wait_s = sock, wait_s
        self._connect()
        self.calls, self.gpu_ms, self.wall_ms, self.problems = 0, 0.0, 0.0, 0

    def _connect(self):
        t0 = time.time()
        while True:
            try:
                self.conn = Client(self.sock, family="AF_UNIX")
                return
            except (FileNotFoundError, ConnectionRefusedError):
                if time.time() - t0 > self.wait_s:
                    raise Refused(
                        f"no cuRobo IK server at {self.sock} (poc/plan_open_speedup/snorlax/run_ik_server.sh)"
                    ) from None
                time.sleep(1)

    def _ask(self, *req):
        """One request; once more on a fresh connection if the server restarted since the last."""
        try:
            self.conn.send(req)
            return self.conn.recv()
        except (EOFError, OSError):
            self._connect()
            self.conn.send(req)
            return self.conn.recv()

    def ik(self, pos, quat, seeds):
        t0 = time.perf_counter()
        r = self._ask("ik", np.asarray(pos, float), np.asarray(quat, float), np.asarray(seeds, float))
        self.calls += 1
        self.problems += len(pos)
        self.gpu_ms += r["ms"]
        self.wall_ms += (time.perf_counter() - t0) * 1e3
        return r

    def fk(self, q):
        return self._ask("fk", np.asarray(q, float))

    def self_collision(self, q):
        return self._ask("self", np.asarray(q, float))["d"]


def _near_seed(q, seed):
    """`q` with each continuous joint taken the turn nearest `seed` (cuRobo returns them anywhere in +-2 pi)."""
    q = np.array(q, dtype=float)
    for j in CONTINUOUS:
        q[j] = seed[j] + (q[j] - seed[j] + np.pi) % (2 * np.pi) - np.pi
    return q


def _solve(ik, pos, quat, seeds):
    """One batched call: (joints nearest their seeds, position errors)."""
    if not pos:
        return np.zeros((0, 7)), np.zeros(0)
    r = ik.ik(np.array(pos, float), np.array(quat, float), np.array(seeds, float))
    return np.array([_near_seed(q, s) for q, s in zip(r["q"], seeds, strict=True)]), np.asarray(r["pos_err"])


def _chain(ik, starts, paths, gate):
    """Chained IK for many paths at once: path i's waypoint k (pos, quat) seeded from its waypoint k-1 (`starts[i]`
    for the first), one call per waypoint index. `gate(q_prev, q, err, pos)` -> (joints to keep, or None if the step
    fails). Returns each path's joints, or None where a step failed."""
    q = [np.asarray(s, float) for s in starts]
    out = [[] for _ in starts]
    alive, k = list(range(len(starts))), 0
    while alive:
        alive = [i for i in alive if k < len(paths[i])]
        if not alive:
            break
        sols, errs = _solve(
            ik,
            [paths[i][k][0] for i in alive],
            [paths[i][k][1] for i in alive],
            [q[i] for i in alive],
        )
        nxt = []
        for i, nq, err in zip(alive, sols, errs, strict=True):
            kept = gate(q[i], nq, float(err), paths[i][k][0])
            if kept is None:
                out[i] = None
                continue
            out[i].append(kept)
            q[i] = kept
            nxt.append(i)
        alive, k = nxt, k + 1
    return out


def _guards_ok(q):
    return abs(np.degrees(q[3])) <= sim.J4_GUARD_DEG and abs(np.degrees(q[5])) <= sim.J6_GUARD_DEG


def _grasp_ok(rb, q, err, keep_out=True):
    """_ik_within_guards' gates: the wrapped joints, or None. `keep_out` False leaves the chair check (the costly
    one) to the caller."""
    q = wrap_joints(q)
    if err > sim.MAX_IK_ERR or not _guards_ok(q) or (keep_out and keep_out_message(rb, q) is not None):
        return None
    return np.asarray(q, float)


def _leg_step(q, nq, err, pos):
    """plan_cartesian's gates on one straight-in step."""
    nq = wrap_joints(nq)
    jump = float(np.degrees(np.max(np.abs((nq - q + np.pi) % (2 * np.pi) - np.pi))))
    if err > sim.MAX_IK_ERR or jump > sim.MAX_STEP_JUMP_DEG or not _guards_ok(nq) or not continuous_ok(q, nq):
        return None
    return np.asarray(nq, float)


def _make_arc_step(rb, keep_out=True):
    def step(q, nq, err, pos):
        """_check_arc's gates on one swing waypoint (its joints raw, as PyBullet's: not wrapped). `keep_out` False
        leaves the chair check to the caller."""
        nq = _near_seed(nq, q)
        if (
            err > od.SWING_IK_PROBE_M
            or np.linalg.norm(pos) > od.SWING_MAX_REACH
            or float(np.max(np.degrees(np.abs(nq - q)))) > od.SWING_MAX_JUMP_DEG
            or abs(np.degrees(nq[5])) > od.J6_GUARD_DEG
            or abs(np.degrees(nq[3])) > sim.J4_GUARD_DEG
            or (keep_out and keep_out_message(rb, nq) is not None)
        ):
            return None
        return nq

    return step


def plan_open_curobo(scene, rb, geom, q0, max_swing, min_swing, after_swing, ik, stats=None):
    """plan_open (not `quick`) with batched cuRobo IK (`ik`: ik_client.CuroboIK). Returns (plan, why)."""
    st = stats if stats is not None else {}
    t_all = time.perf_counter()
    n, hinge, orient, h_raw = geom["n"], geom["hinge"], geom["orient"], geom["h_raw"]
    swings = list(range(max_swing, min(min_swing, max_swing) - 1, -5))
    if swings[-1] != min(min_swing, max_swing):
        swings.append(min(min_swing, max_swing))
    face_pt = geom.get("hinge_seen")
    sides = (0,) if face_pt is None else od.GRASP_SIDES_DEG
    tilts = []
    for side, pitch in [(sd, pt) for sd in sides for pt in od.GRASP_PITCHES_DEG]:
        try:
            with quiet():
                h, g = od.grasp_pose(h_raw, orient, n, pitch, side)
        except Refused:
            continue
        tilts.append(
            dict(
                side=side,
                pitch=pitch,
                h=h,
                g=g,
                g_p=np.asarray(g.position),
                g_q=np.asarray(g.orientation),
            )
        )

    # A: every tilt's configuration seeds, one call
    t0 = time.perf_counter()
    pos, quat, seeds, owner = [], [], [], []
    for ti, t in enumerate(tilts):
        aim = float(np.degrees(np.arctan2(t["g_p"][1], t["g_p"][0])))
        extra = [np.radians([aim if v is None else v for v in s]) for s in od.GRASP_EXTRA_SEEDS_DEG]
        for s in od._config_seeds(q0) + extra:
            pos.append(t["g_p"]), quat.append(t["g_q"]), seeds.append(s), owner.append(ti)
    qa, ea = _solve(ik, pos, quat, seeds)
    for t in tilts:
        t["configs"] = []
    for ti, q, err in zip(owner, qa, ea, strict=True):
        t = tilts[ti]
        q_g = _grasp_ok(rb, q, float(err))
        if q_g is not None and all(
            np.degrees(np.max(np.abs(short_delta(c, q_g)))) > od.SAME_CONFIG_DEG for c in t["configs"]
        ):
            if not t["side"] or od._fingers_in_front(rb, q_g, face_pt, n) >= od.GRASP_SIDE_FACE_CLEAR_M:
                t["configs"].append(q_g)
    jobs = [dict(t=t, q_g=q_g) for t in tilts for q_g in t["configs"]]
    st["a_s"] = time.perf_counter() - t0

    # L: line-ups at every standoff (one call), then the straight-ins chained
    t0 = time.perf_counter()
    tries = []  # (job, standoff, pre_p, straight-in targets)
    for job in jobs:
        t = job["t"]
        for standoff in od.PREGRASP_STANDOFFS:
            pre_p = od.pregrasp_point(t["g"], standoff)
            if od._height_ok(pre_p[2]):
                k_n = max(1, int(np.ceil(np.linalg.norm(t["g_p"] - pre_p) / dc.STEP_M)))
                tgts = [(pre_p + (t["g_p"] - pre_p) * k / k_n, t["g_q"]) for k in range(1, k_n + 1)]
                tries.append((job, standoff, pre_p, tgts))
    qp, ep = _solve(ik, [x[2] for x in tries], [x[0]["t"]["g_q"] for x in tries], [x[0]["q_g"] for x in tries])
    q_pres = []
    for q, err in zip(qp, ep, strict=True):
        q_pre = _grasp_ok(rb, q, float(err), keep_out=False)  # the chair check below, only where it decides
        if q_pre is not None and od._near_limit(rb, q_pre, od.CUROBO_LIMIT_MARGIN_DEG):
            q_pre = None
        q_pres.append(q_pre)
    live = [i for i, q in enumerate(q_pres) if q is not None]
    legs = dict(
        zip(
            live,
            _chain(ik, [q_pres[i] for i in live], [tries[i][3] for i in live], _leg_step),
            strict=True,
        )
    )
    for job in jobs:
        job["appr"] = None
    for i, (job, standoff, pre_p, _) in enumerate(tries):  # in standoff order: the first that passes, as plan_open
        leg = legs.get(i)
        if job["appr"] is not None or leg is None or keep_out_message(rb, q_pres[i]) is not None:
            continue
        g_p = job["t"]["g_p"]
        lengths = od._approach_lengths(g_p, pre_p, leg)
        ko = (
            dc.trajectory_keep_out(q_pres[i], leg, lengths)
            if od.APPROACH_EXACT
            else od._sent_keep_out(q_pres[i], leg, lengths)
        )
        if ko is None:
            job["appr"] = (standoff, pre_p, q_pres[i], leg)
    st["l_s"] = time.perf_counter() - t0

    # S: every swing of every configuration with an approach, chained
    t0 = time.perf_counter()
    arcs = [(job, d, od._swing_waypoints(job["t"]["g"], hinge, d)) for job in jobs if job["appr"] for d in swings]
    sols = _chain(
        ik,
        [job["appr"][3][-1] for job, _, _ in arcs],
        [[(w[:3], w[3:]) for w in wps] for _, _, wps in arcs],
        _make_arc_step(rb, keep_out=False),  # the chair check below, only on the swings that decide
    )
    st["s_s"] = time.perf_counter() - t0

    cands = []
    for job in jobs:  # plan_open's order: tilts, configurations; the largest swing that passes
        if not job["appr"]:
            continue
        standoff, pre_p, q_pre, leg = job["appr"]
        t = job["t"]
        for (j2, d, wps), arc in zip(arcs, sols, strict=True):
            if (
                j2 is not job
                or arc is None
                or any(keep_out_message(rb, q) is not None for q in arc)
                or od._swing_keep_out(leg[-1], arc) is not None
            ):
                continue
            cands.append(
                dict(
                    pitch=t["pitch"],
                    side=t["side"],
                    swing=d,
                    grasp=t["g"],
                    h=t["h"],
                    standoff=standoff,
                    pre_p=pre_p,
                    q_pre=q_pre,
                    leg=leg,
                    wps=wps,
                    arc=arc,
                    margin=od._guard_margin([q_pre, *leg, *arc]),
                    msg="",
                    q_g=job["q_g"],
                    q0=q0,
                    in_front=(t["g_p"], n, od.IN_FRONT_TOL_M),
                )
            )
            break
    if not cands:
        st["total_s"] = time.perf_counter() - t_all
        return (
            None,
            "no tilt and arm configuration gives a grasp + swing that passes the sim checks",
        )
    if od.PREFER_NEAR_START:
        cands.sort(
            key=lambda c: (
                -c["swing"],
                c["side"],
                od._joint_dist_deg(q0, c["q_pre"]) // od.NEAR_START_BUCKET_DEG,
                -c["margin"],
            )
        )
    else:
        cands.sort(key=lambda c: (c["swing"], -c["side"], c["margin"]), reverse=True)
    st["candidates"] = [(c["side"], c["pitch"], c["swing"], round(c["margin"], 1)) for c in cands]
    st["search_s"] = time.perf_counter() - t_all
    reason = None
    for c in cands[: od.SEARCH_AFTER_SWING_MAX]:
        ok, m = after_swing.check(scene, rb, geom, c, reason is None)
        c["after"] = m
        if ok:
            st["total_s"] = time.perf_counter() - t_all
            return c, None
        reason = reason or f"tilt {c['pitch']} deg, swing {c['swing']} deg fits but {m}"
    st["total_s"] = time.perf_counter() - t_all
    return None, reason


def check_arc_curobo(rb, wps, q0, ik):
    """open_door._check_arc with the waypoints' IK from cuRobo, chained: (ok, q_end, msg, arc)."""
    arc = _chain(ik, [np.asarray(q0, float)], [[(w[:3], w[3:]) for w in wps]], _make_arc_step(rb))[0]
    if arc is None:
        return False, np.asarray(q0, float), "a waypoint fails a gate (cuRobo IK)", []
    return True, arc[-1], f"all {len(wps)} waypoints OK (cuRobo IK), final J6 {np.degrees(arc[-1][5]):.1f}deg", arc
