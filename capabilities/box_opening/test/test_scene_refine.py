"""Refining the scene calibration from wrist pairs."""

import json

import numpy as np
import pytest

from box_opening.perception.scene_refine import apply, fit, load_pairs, spread_m


def _rot_z(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_pairs_at_one_spot_fit_a_translation_only():
    """...and a HORIZONTAL one: the scene camera's box-top height is not the
    arm's (the pink lid reads ~3 cm low with the table exactly right,
    2026-09-24), so the pairs' heights are not fitted — the table plane
    owns height and tilt."""
    scene = np.array([[0.57, -0.03, 0.065], [0.58, -0.03, 0.066]])
    wrist = scene + np.array([0.006, -0.004, 0.024])
    r = fit(scene, wrist)
    assert r.mode == "translation" and r.n == 2
    assert np.allclose(r.D[:3, 3], [0.006, -0.004, 0.0], atol=1e-9)
    assert np.allclose(r.D[:3, :3], np.eye(3))
    assert r.rms_after_mm < 1e-6 and r.rms_before_mm == pytest.approx(7.2, abs=0.1)


def test_spread_pairs_recover_a_rotation_and_a_shift():
    rng = np.random.default_rng(1)
    scene = np.c_[rng.uniform(0.35, 0.65, 6), rng.uniform(-0.3, 0.3, 6), rng.uniform(0.06, 0.09, 6)]
    R, t = _rot_z(1.5), np.array([0.02, -0.01, 0.03])
    wrist = scene @ R.T + t + rng.normal(0, 0.0005, scene.shape)
    r = fit(scene, wrist)
    assert r.mode == "rigid" and r.spread_m > 0.15
    assert np.allclose(r.D[:3, :3], R, atol=2e-3) and np.allclose(r.D[:2, 3], t[:2], atol=2e-3)
    assert r.D[2, 3] == 0.0  # the height is the table plane's (above)
    assert r.rms_after_mm < 1.5
    T = np.eye(4)
    T[:3, 3] = [0.0, 0.6, 0.38]
    T2 = apply(T, r)
    assert np.allclose(T2, r.D @ T)


def test_synthetic_harness_pairs_are_dropped(tmp_path):
    f = tmp_path / "residuals.jsonl"
    rows = [
        {"scene_top": [0.457, -0.047, 0.085], "wrist_top": [0.4595, -0.0497, 0.085]},
        {"scene_top": [0.5738, -0.0294, 0.0651], "wrist_top": [0.5809, -0.033, 0.0919]},
    ]
    f.write_text("".join(json.dumps(r) + "\n" for r in rows) + "not json\n")
    scene, wrist = load_pairs(f, synthetic=[0.4595, -0.0497, 0.085])
    assert len(scene) == 1 and wrist[0][2] == pytest.approx(0.0919)
    assert spread_m(scene) == 0.0


def test_pairs_recorded_under_an_earlier_calibration_are_retired(tmp_path):
    """Found 2026-09-21: two of the four pairs on file were recorded on
    09-17 at 13:33 and 14:10 — BEFORE that evening's correction (+24 mm in z)
    was written at 16:03. They still said the scene camera read 22-27 mm
    low; mixed with today's pairs they made the "mean offset" the mission
    prints meaningless and would have gone into the next fit. A pair counts
    only if it was recorded after the calibration it is judged against took
    effect."""
    import os

    from box_opening.perception.scene_refine import calibrated_at

    f = tmp_path / "residuals.jsonl"
    rows = [
        {"t": 1000.0, "scene_top": [0.5738, -0.0294, 0.0651], "wrist_top": [0.5809, -0.033, 0.0919]},
        {"t": 3000.0, "scene_top": [0.4353, -0.0327, 0.0907], "wrist_top": [0.4357, -0.0451, 0.085]},
        {"scene_top": [0.5, 0.0, 0.09], "wrist_top": [0.5, 0.0, 0.09]},  # no time on it: cannot be trusted either
    ]
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert len(load_pairs(f)[0]) == 3  # no calibration named: everything on file
    scene, wrist = load_pairs(f, since=2000.0)
    assert len(scene) == 1 and wrist[0][1] == pytest.approx(-0.0451)

    # when a calibration took effect: what the refinement wrote, else the file's own time
    calib = tmp_path / "camera_scene.yaml"
    calib.write_text("xyz: [0, 0, 0]\nquat_xyzw: [0, 0, 0, 1]\n")
    os.utime(calib, (2500.0, 2500.0))
    assert calibrated_at(calib) == pytest.approx(2500.0)
    calib.write_text("xyz: [0, 0, 0]\nquat_xyzw: [0, 0, 0, 1]\nrefined_at: 2000.0\n")
    assert calibrated_at(calib) == pytest.approx(2000.0)
    assert calibrated_at(tmp_path / "missing.yaml") is None
