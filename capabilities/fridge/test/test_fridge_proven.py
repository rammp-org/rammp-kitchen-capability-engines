"""fridge.proven (the grasps that worked before, tried first) and open_door's search order, offline.

open_door's functions are loaded through `ast` with stand-ins for what they use (it imports door_common, cuRobo's
planning and the camera); proven.py is pure."""

import ast
import types
from pathlib import Path

import pytest

from fridge import proven

OPEN_DOOR = Path(__file__).resolve().parents[1] / "fridge" / "open_door.py"

SIDE = dict(grasp="side", pitch=-10.0, flip=False, lag=30.0, standoff=0.15)


def _cands(*pitches, side=True):
    return [dict(pitch=p, flip=False, side=side) for p in pitches]


def test_records_round_trip_and_keep_the_newest(tmp_path):
    path = tmp_path / "d" / proven.FILE_NAME
    assert proven.load(path) == []
    for k in range(proven.KEEP + 5):
        assert proven.record(path, dict(SIDE, pitch=float(k), source="plan")) == ""
    recs = proven.load(path)
    assert len(recs) == proven.KEEP and recs[-1]["pitch"] == proven.KEEP + 4 and recs[0]["when"]


def test_an_unreadable_record_is_no_record(tmp_path):
    path = tmp_path / proven.FILE_NAME
    path.write_text("{not json")
    assert proven.load(path) == []
    path.write_text('{"a": 1}')
    assert proven.load(path) == []


def test_the_arm_beats_the_planner_beats_the_seeds_then_the_nearest_placement():
    here, normal = [0.77, -0.29, 0.38], [-0.98, 0.19, 0.0]
    recs = [
        dict(SIDE, pitch=0.0, source="seed"),
        dict(SIDE, pitch=10.0, source="plan", handle=here, normal=normal, when="2026-11-19T10:00:00"),
        dict(SIDE, pitch=20.0, source="arm", handle=[1.2, 0.2, 0.38], normal=normal, when="2026-11-18T10:00:00"),
        dict(SIDE, pitch=30.0, source="arm", handle=here, normal=normal, when="2026-11-17T10:00:00"),
        dict(SIDE, pitch=40.0, source="plan", grasp="front"),  # the other grasp: never offered
    ]
    keys = [k for k, _ in proven.preferred(recs, "side", here, normal)]
    assert keys == [(30, False), (20, False), (10, False), (0, False)]


def test_one_record_per_grasp_its_best():
    recs = [
        dict(SIDE, lag=20.0, source="plan", when="2026-11-19T10:00:00"),
        dict(SIDE, lag=30.0, source="arm", when="2026-11-01T10:00:00"),
        dict(SIDE, lag=25.0, source="plan", when="2026-11-20T10:00:00"),
    ]
    ((key, rec),) = proven.preferred(recs, "side")
    assert key == (-10, False) and rec["lag"] == 30.0  # the arm's, though older


def test_the_grasps_on_record_go_first_and_nothing_is_dropped():
    cands = _cands(0, -10, 10, -20, 20)
    prefs = proven.preferred([dict(SIDE, pitch=20.0, source="arm"), dict(SIDE, source="seed")], "side")
    out = proven.order(cands, prefs)
    assert [c["pitch"] for c in out] == [20, -10, 0, 10, -20]
    assert out[0]["proven"]["source"] == "arm" and out[1]["proven"]["source"] == "seed"
    assert "proven" not in out[2]
    assert proven.order(cands, []) == cands


def test_first_moves_one_value_to_the_front():
    assert proven.first(30.0, (25, 30, 20)) == [30, 25, 20]
    assert proven.first(0.12, (0.15, 0.12, 0.09)) == [0.12, 0.15, 0.09]
    assert proven.first(None, (25, 30)) == [25, 30]
    assert proven.first(17.0, (25, 30)) == [25, 30]  # not one of them: the usual order


def _load(names, **scope):
    tree = ast.parse(OPEN_DOOR.read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(OPEN_DOOR), "exec"), scope)
    return scope


F = types.SimpleNamespace(
    SIDE_SLIDE_M=(0.15, 0.12, 0.09), PREGRASP_STANDOFF_M=(0.12, 0.08), SIDE_HAND_LAGS_DEG=(25, 30, 20), SIDE_CONFIGS=2
)


def test_the_line_up_and_lag_that_worked_are_tried_first():
    od = _load({"standoffs", "hand_lags"}, F=F, proven=proven)
    side = dict(side=True, proven=dict(standoff=0.09, lag=30.0))
    assert od["standoffs"](side) == [0.09, 0.15, 0.12]
    assert od["hand_lags"](side) == [30, 25, 20]
    assert od["standoffs"](dict(side=True)) == [0.15, 0.12, 0.09]
    assert od["hand_lags"](dict(side=False, proven=dict(lag=30.0))) == [0.0]


class Infeasible(Exception):
    pass


def _search_scope(said):
    return _load(
        {"_search"}, F=F, say=said.append, seeds=lambda q0: [("J3 -60", None)], Infeasible=Infeasible, why_not=str
    )


def test_the_search_takes_the_first_whole_open_and_tries_nothing_after_it():
    said, tried = [], []

    def grasp_and_swing(cand, hint_name=None, seed=None, lineup=None):
        tried.append(cand["pitch"])
        return dict(cand=cand)  # every swing plans

    def after_swing(found):
        return None if found["cand"]["pitch"] == -10 else dict(found, done=True)

    search = _search_scope(said)["_search"]
    plan = search(_cands(-10, 0, 10, 20), grasp_and_swing, after_swing, None, [])
    assert plan["cand"]["pitch"] == 0 and tried == [-10, 0]  # -10's push failed: the next grasp, then done


def test_with_no_swing_at_all_it_tries_the_other_configurations():
    said, calls = [], []

    def grasp_and_swing(cand, hint_name=None, seed=None, lineup=None):
        calls.append((cand["pitch"], hint_name))
        return dict(cand=cand) if hint_name else None

    search = _search_scope(said)["_search"]
    plan = search(_cands(0, 10, side=False), grasp_and_swing, lambda f: dict(f, done=True), None, [])
    assert plan["cand"]["pitch"] == 0 and calls == [(0, None), (10, None), (0, "J3 -60")]


def test_nothing_plans_refuses():
    search = _search_scope([])["_search"]
    with pytest.raises(Infeasible):
        search(_cands(0, 10, side=False), lambda *a, **k: None, lambda f: None, None, [])
