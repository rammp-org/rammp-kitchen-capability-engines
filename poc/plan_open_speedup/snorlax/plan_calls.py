# ruff: noqa: E501, T201  -- PoC analysis script, not capability code
"""Where each open's planning time went: the search, then the cuRobo checks after it, and how many plans the planner
was asked for while it planned (sheppy-planner's log on stdin). usage:
  docker logs sheppy-planner 2>&1 | python3 plan_calls.py <runs dir> [--since YYYYMMDD_HHMMSS]"""

import datetime
import json
import re
import sys
from pathlib import Path

LINE = re.compile(r"\[(INFO|ERROR|WARN)\] \[(\d+\.\d+)\] \[rammp_curobo\]: (.*)")

asks, fails = [], []
for line in sys.stdin:
    m = LINE.search(line)
    if m and m.group(3).startswith("Planning..."):
        asks.append(float(m.group(2)))
    elif m and m.group(3).startswith("Plan failed"):
        fails.append((float(m.group(2)), m.group(3).split(":")[0].replace("Plan failed — ", "")))

root = Path(sys.argv[1])
since = sys.argv[sys.argv.index("--since") + 1] if "--since" in sys.argv else ""
print(
    f"{'run':22} {'planner':7} {'exec':>4} {'push':>4} {'plan s':>7} {'search':>7} {'after':>6} {'asks':>5} {'failed':>6}  failures"
)
for d in sorted(root.glob("*_open")):
    if d.name < since or not (d / "run.json").exists():
        continue
    run = json.loads((d / "run.json").read_text())
    start = (
        datetime.datetime.strptime(run["start"], "%Y-%m-%d %H:%M:%S")
        .replace(tzinfo=datetime.timezone.utc)
        .timestamp()
    )
    args = run.get("args", {})
    for e in run["events"]:
        if e["kind"] != "plan":
            continue
        t1 = start + e["t"] + 1.0  # run.json's start is to the second
        t0 = t1 - 1.0 - e["total_s"]
        n = sum(t0 <= t <= t1 for t in asks)
        f = [k for t, k in fails if t0 <= t <= t1]
        kinds = ", ".join(f"{k} x{f.count(k)}" for k in sorted(set(f)))
        search = e.get("search_s")
        after = e["total_s"] - search if search is not None else None
        print(
            f"{d.name:22} {e['planner']:7} {str(bool(args.get('execute')))[0]:>4} {str(bool(args.get('push_open')))[0]:>4} "
            f"{e['total_s']:7.2f} {search if search is not None else float('nan'):7.2f} {after if after is not None else float('nan'):6.2f} {n:5d} {len(f):6d}  {kinds}"
        )
