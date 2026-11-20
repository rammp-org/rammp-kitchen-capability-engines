"""Grasps that have worked before, tried first by the open's search.

The open's search (open_door._search) takes the first grasp whose whole open plans, so the order it tries them in
decides how long planning takes: on 11-19 the grasp it used (side, tilt -10, the hand 30 deg behind the door) was the
second tried, but it was only chosen after seven more had failed. This keeps a record of what worked and puts it first:

- every open that plans is recorded (`plan`), and again once its swing finished on the arm (`arm`), in FILE_NAME next
  to the door file, the last KEEP of them;
- constants.PROVEN_SEEDS (`seed`) are what planned in the test logs before this record existed;
- the grasps on record go first, the arm's before the planner's before the seeds, then the nearest placement (where the
  handle was and which way the door faced), then the newest; each with the line-up and the hand's lag that worked
  tried first too. Every other grasp follows in its usual order, so nothing is left out: a grasp that worked elsewhere
  costs at most one failed try here.

Pure (numpy and json only): open_door wires it in.
"""

import json
import os
import time
from pathlib import Path

import numpy as np

FILE_NAME = "proven_opens.json"
KEEP = 50  # records kept, the newest
SOURCE_RANK = {"arm": 0, "plan": 1, "seed": 2}  # which kind of proof is tried first
YAW_PER_M_DEG = 200.0  # placement distance: this many deg of door yaw count as 1 m of handle position (10 deg: 5 cm)


def load(path):
    """The records in `path` (a list of dicts), [] if there is none or it can't be read."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return []
    return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []


def record(path, rec):
    """Add `rec` to `path` (the newest KEEP kept). Returns an error message, or "" when saved. Never raises: a record
    that can't be written only means the next search doesn't know about it."""
    try:
        path = Path(path)
        recs = (load(path) + [dict(rec, when=rec.get("when") or time.strftime("%Y-%m-%dT%H:%M:%S"))])[-KEEP:]
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(json.dumps(recs, indent=1))
        os.replace(tmp, path)
        return ""
    except Exception as e:  # noqa: BLE001 -- the open goes on either way
        return repr(e)


def make_record(grasp, found, handle, normal, source):
    """The record for a planned open's grasp. `found`: the plan's dict (cand, standoff, lag, hint)."""
    cand = found["cand"]
    return dict(
        grasp=grasp,
        pitch=float(cand["pitch"]),
        flip=bool(cand.get("flip", False)),
        lag=float(found.get("lag") or 0.0),
        standoff=float(found["standoff"]),
        config=found.get("hint") or "",
        handle=[round(float(x), 4) for x in handle],
        normal=[round(float(x), 4) for x in normal],
        source=source,
    )


def _placement_gap(rec, handle, normal):
    """How far `rec`'s placement is from this one (m, the door's yaw counted in), inf if either is unknown."""
    if handle is None or normal is None or not rec.get("handle") or not rec.get("normal"):
        return float("inf")
    d = float(np.linalg.norm(np.asarray(rec["handle"], float)[:2] - np.asarray(handle, float)[:2]))
    a, b = np.asarray(rec["normal"], float)[:2], np.asarray(normal, float)[:2]
    cos = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))
    return d + float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))) / YAW_PER_M_DEG


def preferred(records, grasp, handle=None, normal=None):
    """The grasps on record for `grasp` ("side" / "front"), best first: [(key, rec)] with key (pitch, flip), one per
    grasp (its best record), ranked by SOURCE_RANK, then placement gap, then newest."""
    recs = [r for r in records if r.get("grasp") == grasp and "pitch" in r]
    recs.sort(key=lambda r: str(r.get("when") or ""), reverse=True)  # newest first, kept among equals below (stable)
    recs.sort(key=lambda r: (SOURCE_RANK.get(r.get("source"), len(SOURCE_RANK)), _placement_gap(r, handle, normal)))
    out, seen = [], set()
    for r in recs:
        key = (round(float(r["pitch"])), bool(r.get("flip", False)))
        if key not in seen:
            seen.add(key)
            out.append((key, r))
    return out


def order(cands, prefs):
    """`cands` (open_door's grasp candidates) with the ones on record first, in `prefs`' order, then the rest in their
    own order. Each one on record carries it as cand["proven"] (its standoff and lag are tried first)."""
    index = {key: i for i, (key, _) in enumerate(prefs)}
    first, rest = [], []
    for c in cands:
        key = (round(float(c["pitch"])), bool(c.get("flip", False)))
        if key in index:
            first.append((index[key], dict(c, proven=prefs[index[key]][1])))
        else:
            rest.append(c)
    return [c for _, c in sorted(first, key=lambda x: x[0])] + rest


def first(value, values):
    """`values` (a sequence) with `value` moved to the front, if it is one of them (within 1e-6)."""
    values = list(values)
    if value is not None:
        for i, v in enumerate(values):
            if abs(float(v) - float(value)) < 1e-6:
                return [v] + values[:i] + values[i + 1 :]
    return values


def describe(rec):
    """`rec` in a few words, for the log."""
    bits = [f"tilt {rec['pitch']:+.0f} deg"]
    if rec.get("lag"):
        bits.append(f"hand {rec['lag']:.0f} deg behind")
    if rec.get("standoff"):
        bits.append(f"line-up {rec['standoff'] * 100:.0f} cm")
    source = {"arm": "ran on the arm", "plan": "planned", "seed": "in the test logs"}.get(rec.get("source"), "")
    when = f" {rec['when'][:10]}" if rec.get("when") else ""
    return f"{', '.join(bits)} ({source}{when})"
