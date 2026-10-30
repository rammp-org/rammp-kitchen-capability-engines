"""Refining the scene camera's calibration from what the wrist measured.

Every staging run recorded one pair: where the scene camera put the button
top and where the wrist's aim found it, both in base_link — until the
staging mode was scrapped (2026-10-28): the reach-out records none, so the
pairs on file are the last there are. The pairs are the calibration's own residuals, and a
rigid correction fitted to them is the refinement — no marker, no motion.

With few pairs, or pairs all at one spot, only a translation is
identifiable; a rotation needs pairs spread over the table (>= 4, spanning
>= 0.15 m). The fit says which it did.

The correction is HORIZONTAL: x, y and a turn about the vertical. The
scene camera's box-top height is not the arm's — with the calibration
fitted to put the table exactly at its surveyed height and level, the pink
lid still read ~3 cm low (2026-09-24) — so a 3-D fit to the pairs tilts
the table to chase it. Height and tilt belong to the table plane the scene
camera sees every run, not to the pairs.

A pair measures the calibration it was recorded under, and no other: once
a correction is written, the pairs it was fitted from describe a camera
that no longer exists (calibrated_at, load_pairs' `since`).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

MIN_PAIRS_ROTATION = 4
MIN_SPREAD_M = 0.15


@dataclass(frozen=True)
class Refinement:
    D: np.ndarray  # 4x4 correction, base -> base: p_wrist ~= D @ p_scene
    mode: str  # "translation" | "rigid"
    n: int
    rms_before_mm: float
    rms_after_mm: float
    spread_m: float


def calibrated_at(calib_path):
    """When the calibration in `calib_path` took effect (unix time): the
    `refined_at` the refinement script writes, else the file's own time —
    or None when there is no such file."""
    p = Path(calib_path)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            at = (yaml.safe_load(f) or {}).get("refined_at")
    except (OSError, yaml.YAMLError):
        at = None
    return float(at) if at is not None else p.stat().st_mtime


def load_pairs(path, synthetic=None, since=None):
    """(scene_xyz, wrist_xyz) pairs from a residuals.jsonl; `synthetic`
    is a wrist point to drop (the e2e harness's stub box, before the
    harness got its own file). With `since` (calibrated_at), only the pairs
    recorded under the calibration in force: two pairs from before the
    2026-09-17 correction sat in the file for days, still reporting the
    24 mm it had fixed."""
    scene, wrist = [], []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if since is not None and float(r.get("t", 0.0)) < since:
                continue
            w = [float(v) for v in r["wrist_top"]]
            if synthetic is not None and np.allclose(w, synthetic, atol=1e-6):
                continue
            scene.append([float(v) for v in r["scene_top"]])
            wrist.append(w)
    return np.asarray(scene, float).reshape(-1, 3), np.asarray(wrist, float).reshape(-1, 3)


def spread_m(pts):
    if len(pts) < 2:
        return 0.0
    d = pts[:, None, :2] - pts[None, :, :2]
    return float(np.sqrt((d**2).sum(-1)).max())


def fit(scene, wrist):
    """The horizontal correction D with wrist ~= D @ scene in x and y: a
    turn about the vertical and a shift (2-D Kabsch) when the pairs allow a
    rotation, a shift otherwise. Heights are not fitted (module note); the
    rms figures are horizontal."""
    scene = np.asarray(scene, float)
    wrist = np.asarray(wrist, float)
    n = len(scene)
    if n == 0:
        raise ValueError("no pairs")
    s2, w2 = scene[:, :2], wrist[:, :2]
    before = float(np.sqrt(np.mean(np.sum((s2 - w2) ** 2, axis=1))))
    sp = spread_m(scene)
    D = np.eye(4)
    if n >= MIN_PAIRS_ROTATION and sp >= MIN_SPREAD_M:
        cs, cw = s2.mean(0), w2.mean(0)
        H = (s2 - cs).T @ (w2 - cw)
        U, _S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        D[:2, :2] = R
        D[:2, 3] = cw - R @ cs
        mode = "rigid"
    else:
        D[:2, 3] = (w2 - s2).mean(0)
        mode = "translation"
    fitted = s2 @ D[:2, :2].T + D[:2, 3]
    after = float(np.sqrt(np.mean(np.sum((fitted - w2) ** 2, axis=1))))
    return Refinement(D=D, mode=mode, n=n, rms_before_mm=1000 * before, rms_after_mm=1000 * after, spread_m=sp)


def apply(T_base_link, refinement):
    """The refined base_link -> scene_camera_link transform."""
    return refinement.D @ np.asarray(T_base_link, float)


def write_refinement(calib_path, refinement, backup_dir, how):
    """Write the refined calibration into `calib_path` (a copy of the old
    one kept in `backup_dir`), stamped refined_at so the pairs it was fitted
    from retire (calibrated_at). `how` goes into its note. Returns the
    backup's path."""
    import shutil
    import time

    from box_opening.perception.d405 import mat_to_quat_xyzw, quat_to_mat
    from box_opening.perception.scene_calib import transform

    calib_path = Path(calib_path)
    doc = yaml.safe_load(calib_path.read_text())
    T2 = apply(transform(quat_to_mat(*doc["quat_xyzw"]), doc["xyz"]), refinement)
    backup = Path(backup_dir) / ("camera_scene.yaml.bak-%s" % time.strftime("%Y%m%d-%H%M%S"))
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(calib_path, backup)
    doc["xyz"] = [round(float(v), 5) for v in T2[:3, 3]]
    doc["quat_xyzw"] = [round(float(v), 6) for v in mat_to_quat_xyzw(T2[:3, :3])]
    doc["refined_at"] = round(time.time(), 1)
    doc["note"] = (
        str(doc.get("note", "")).strip()
        + " Refined %s from %d wrist pairs (%s fit, %s; rms %.1f -> %.1f mm)."
        % (
            time.strftime("%Y-%m-%d %H:%M"),
            refinement.n,
            refinement.mode,
            how,
            refinement.rms_before_mm,
            refinement.rms_after_mm,
        )
    ).strip()
    calib_path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return backup
