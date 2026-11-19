# ruff: noqa: E501, T201  -- PoC analysis script, not capability code
"""Side-by-side numbers from the PoC microwave runs (run.json + arm.npz in each runs/<time>_open/).

  planning: each run's planner and time ("plan"), and with compare_planners every planner on the same detection
            ("plan_compare": time, the plan picked, and whether the pool's plan is the serial one);
  the swing, when it ran: its duration, peak joint speed and acceleration, peak joint torque, how far the hand left
            the commanded arc, the circle fit's rms, how far from the last waypoint it ended, the planned largest
            step between waypoints.

usage: compare_runs.py <runs dir> [--since YYYYMMDD_HHMMSS]   (numpy only)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def events(run, kind):
    return [e for e in run["events"] if e["kind"] == kind]


def planned(plan):
    """Largest step between swing waypoints and total joint travel (deg), from the logged plan."""
    if not plan:
        return None, None
    arc = np.asarray([plan["leg_deg"][-1], *plan["arc_deg"]])
    d = np.abs(np.diff(arc, axis=0))
    return float(d.max()), float(d.sum())


def swing_metrics(d, run):
    """The executed swing, from the arm samples between swing_start and swing_done."""
    s, e = events(run, "swing_start"), events(run, "swing_done")
    if not s or not e or not (d / "arm.npz").exists():
        return None
    a = np.load(d / "arm.npz")
    t = a["t"]
    m = (t >= s[-1]["t"]) & (t <= e[-1]["t"])
    if m.sum() < 5:
        return None
    tt, vel, eff = t[m], np.degrees(a["velocity"][m]), a["effort"][m]
    acc = np.diff(vel, axis=0) / np.diff(tt)[:, None]
    path = events(run, "swing_path")
    return dict(
        duration_s=float(e[-1]["t"] - s[-1]["t"]),
        peak_joint_speed_dps=float(np.abs(vel).max()),
        peak_joint_accel_dps2=float(np.abs(acc).max()),
        peak_torque_nm=float(np.abs(eff).max()),
        off_arc_cm=100 * path[-1]["max_off_arc_m"] if path else None,
        fit_rms_mm=1000 * path[-1]["fit_rms_m"] if path else None,
        end_err_cm=100 * e[-1]["end_err_m"],
        swing_ok=e[-1]["ok"],
    )


def fmt(x, nd=2):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main():
    root = Path(sys.argv[1])
    since = sys.argv[sys.argv.index("--since") + 1] if "--since" in sys.argv else ""
    rows, compare, by = [], [], defaultdict(list)
    for d in sorted(root.glob("*_open")):
        if d.name < since or not (d / "run.json").exists():
            continue
        run = json.loads((d / "run.json").read_text())
        args = run.get("args", {})
        for c in events(run, "plan_compare"):
            compare.append((d.name, c))
        p = events(run, "plan")
        if not p:
            continue
        p = p[-1]
        step, travel = planned(p.get("plan"))
        sw = swing_metrics(d, run) if args.get("execute") else None
        end = events(run, "end")
        row = dict(
            run=d.name,
            planner=p["planner"],
            execute=bool(args.get("execute")),
            plan_s=p["total_s"],
            search_s=p.get("search_s"),
            swing=(p.get("plan") or {}).get("swing"),
            tilt=(p.get("plan") or {}).get("pitch"),
            margin=(p.get("plan") or {}).get("margin_deg"),
            planned_max_step=step,
            planned_travel=travel,
            ok=end[-1]["ok"] if end else None,
            **(sw or {}),
        )
        rows.append(row)
        by[(row["planner"], row["execute"])].append(row)

    if compare:
        print("== every planner on the same detection (compare_planners) ==")
        print(
            f"{'run':22} {'planner':7} {'plan s':>7} {'search s':>8} {'swing':>5} {'tilt':>4} {'margin':>6} {'max step':>8}  same as serial"
        )
        serial = {}
        for name, c in compare:
            plan = c.get("plan")
            if c["planner"] == "serial":
                serial[name] = plan
            same = "" if c["planner"] == "serial" else ("yes" if plan == serial.get(name) else "no")
            step, _ = planned(plan)
            print(
                f"{name:22} {c['planner']:7} {c['total_s']:7.2f} {fmt(c.get('search_s')):>8} {fmt((plan or {}).get('swing')):>5} "
                f"{fmt((plan or {}).get('pitch')):>4} {fmt((plan or {}).get('margin_deg'), 1):>6} {fmt(step, 1):>8}  {same}"
            )
        times = defaultdict(list)
        for _, c in compare:
            times[c["planner"]].append(c["total_s"])
        print(
            "mean plan time: "
            + ", ".join(f"{k} {np.mean(v):.2f} s (n={len(v)})" for k, v in times.items())
        )

    if rows:
        print("\n== runs ==")
        cols = [
            "planner",
            "execute",
            "plan_s",
            "swing",
            "tilt",
            "margin",
            "planned_max_step",
            "duration_s",
            "peak_joint_speed_dps",
            "peak_joint_accel_dps2",
            "peak_torque_nm",
            "off_arc_cm",
            "fit_rms_mm",
            "end_err_cm",
            "swing_ok",
            "ok",
        ]
        print("run".ljust(22) + " ".join(c[:10].rjust(10) for c in cols))
        for r in rows:
            print(r["run"].ljust(22) + " ".join(fmt(r.get(c)).rjust(10) for c in cols))
        print("\n== per planner (mean +- std) ==")
        num = [
            "plan_s",
            "planned_max_step",
            "duration_s",
            "peak_joint_speed_dps",
            "peak_joint_accel_dps2",
            "peak_torque_nm",
            "off_arc_cm",
            "fit_rms_mm",
            "end_err_cm",
        ]
        for (planner, ex), rs in sorted(by.items()):
            parts = []
            for c in num:
                v = [r[c] for r in rs if r.get(c) is not None]
                if v:
                    parts.append(f"{c} {np.mean(v):.2f}+-{np.std(v):.2f}")
            oks = sum(bool(r.get("ok")) for r in rs)
            print(
                f"{planner} ({'executed' if ex else 'dry run'}, {len(rs)} runs, {oks} ok): "
                + ", ".join(parts)
            )
    if not rows and not compare:
        print(f"no open runs with planner logs under {root}")


if __name__ == "__main__":
    main()
