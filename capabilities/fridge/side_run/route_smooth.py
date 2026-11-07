"""The reach-in's junction between cuRobo's path and the merge (11-06: the joint path turned 123 deg there -- J1/J2/J4/J6
reversing -- and the arm all but stopped; the merge's first knots wiggled J2-J5): cut back into cuRobo's tail and forward
into the merge, joined by a straight joint-space blend; the cut with the smallest direction turn at both of its ends
whose blend keeps >= DOOR_MIN_M off the door model and >= SELF_MIN_M between the arm's own links. The slide is untouched."""
import numpy as np
from door_common import common as dc
import pull_lib

DOOR_MIN_M, SELF_MIN_M = 0.02, 0.04
BACK_KNOTS = 30  # how far back into cuRobo's path a cut may go
STEP_DEG = 0.5  # blend knots


def _turn(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na < 1e-9 or nb < 1e-9 else float(np.degrees(np.arccos(np.clip(a @ b / na / nb, -1, 1))))


W = 3  # knots each side over which a direction is taken (the knots are only ~0.5 deg apart)


def _dir_in(path, i):
    """The direction arriving at knot i (over the W knots before it)."""
    return path[i] - path[max(0, i - W)]


def _dir_out(path, i):
    return path[min(len(path) - 1, i + W)] - path[i]


def shortcut(scene, rb, path, seg, door):
    """path: [q_start, *route] (continuous joints), seg: the route's labels (len(path) - 1). Returns (path, seg, report)."""
    path = [np.asarray(q, float) for q in path]
    i_m, i_s = seg.index("merge") + 1, seg.index("slide") + 1  # path indices of the first merge / slide knot
    before = max(_turn(_dir_in(path, i), _dir_out(path, i)) for i in range(i_m - 2, i_m + 6))  # the junction as it is
    cands = []
    for ia in range(i_m, max(1, i_m - BACK_KNOTS) - 1, -1):
        for ib in range(i_m + 1, i_s):
            d = path[ib] - path[ia]
            t = max(_turn(_dir_in(path, ia), d), _turn(d, _dir_out(path, ib)))
            if t < before - 10.0:  # only a real improvement
                cands.append((t, ib - ia, ia, ib))
    cands.sort()
    bodies = dc.DoorFrame(door, door["open_sign"]).bodies(scene, rb, 0.0)
    try:
        for t, _, ia, ib in cands[:200]:
            d = path[ib] - path[ia]
            n = max(2, int(np.ceil(np.degrees(np.max(np.abs(d))) / STEP_DEG)))
            blend = [path[ia] + d * k / n for k in range(1, n)]
            if any(dc.clearance(rb, q, bodies)[0] < DOOR_MIN_M for q in blend):
                continue
            if any(pull_lib.self_clearance(rb, q)[0] < SELF_MIN_M for q in blend[::2]):
                continue
            new = path[:ia + 1] + blend + path[ib:]
            new_seg = seg[:ia] + ["blend"] * len(blend) + seg[ib - 1:]
            return new, new_seg, (f"junction turn {before:.0f} -> {t:.0f} deg: cut {i_m - ia} cuRobo knots and "
                                  f"{ib - i_m} merge knots, {len(blend)} blend knots")
    finally:
        dc.drop(rb, bodies)
    return path, seg, f"junction turn {before:.0f} deg: no cleaner cut clears the door and the arm -- unchanged"


# --- one smooth curve instead of pieces (11-06: no junction) --------------------------
BLEND_TRIES = (4, 6, 8, 12, 16, 24)  # interior spline knots: the smoothest that keeps clear wins
ENTRY_KNOTS = 4  # the slide's first knots in the fit, heavily weighted: the curve enters the slide along it
PIN_W = 1e4


def blend(scene, rb, path, seg, door):
    """Everything before the slide (cuRobo's path + the merge) replaced by ONE least-squares cubic spline in joint space
    through those knots: pinned to the start, entering the slide along the slide (its first ENTRY_KNOTS weighted in), the
    fewest interior knots (BLEND_TRIES) whose curve keeps >= DOOR_MIN_M off the door model and >= SELF_MIN_M between the
    arm's links. Returns (path, seg, report); unchanged (with why) if none does."""
    from scipy.interpolate import make_lsq_spline
    P = np.array([np.asarray(q, float) for q in path])
    i_s = seg.index("slide") + 1  # the slide's first knot in path
    fit = P[: i_s + ENTRY_KNOTS]
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(fit, axis=0), axis=1))]
    w = np.ones(len(fit)); w[0] = PIN_W; w[i_s:] = PIN_W
    bodies = dc.DoorFrame(door, door["open_sign"]).bodies(scene, rb, 0.0)
    try:
        for m in BLEND_TRIES:
            inner = np.linspace(0, s[i_s], m + 2)[1:-1]  # the free knots only before the slide
            t = np.r_[[s[0]] * 4, inner, [s[-1]] * 4]
            try:
                spl = make_lsq_spline(s, fit, t, k=3, w=w)
            except (ValueError, np.linalg.LinAlgError):
                continue
            n = max(2, int(np.ceil(np.degrees(np.max(np.abs(np.diff(fit, axis=0)), axis=1).sum()) / STEP_DEG)))
            ss = np.linspace(0.0, s[i_s], n)
            curve = spl(ss)
            curve[0] = P[0]
            dev = float(np.max(np.linalg.norm(curve[-1] - P[i_s])))
            if np.degrees(np.max(np.abs(curve[-1] - P[i_s]))) > 0.5:
                continue  # must arrive at the slide's start
            if any(dc.clearance(rb, q, bodies)[0] < DOOR_MIN_M for q in curve[::2]):
                continue
            if any(pull_lib.self_clearance(rb, q)[0] < SELF_MIN_M for q in curve[::4]):
                continue
            new = [P[0]] + [q for q in curve[1:-1]] + [P[i_s]] + [q for q in P[i_s + 1:]]
            new_seg = ["blend"] * (len(curve) - 1) + seg[i_s - 1:]
            gone = np.degrees(max(np.min(np.linalg.norm(P[: i_s + 1] - q, axis=1)) for q in curve))
            return new, new_seg, (f"cuRobo's path + the merge as one spline ({m} interior knots, {len(curve)} knots), "
                                  f"off the original by up to {gone:.1f} deg (joint space)")
    finally:
        dc.drop(rb, bodies)
    return list(P), seg, "no spline through the approach keeps clear of the door and the arm -- the pieces as planned"
