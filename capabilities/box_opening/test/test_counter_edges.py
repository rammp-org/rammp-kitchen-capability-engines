"""The cabinet from the cameras: the counter's edges the table map shows (clear_table.counter_edges)
and the solid below them (counter_keep_out), keep-out boxes for the body check (runtime/body
Body.with_obstacles). No edge where the map saw none."""

import math

import numpy as np
import pytest

from box_opening.perception import clear_table as ct

TOP = 0.389  # the chair study's 0.91 m counter, in arm_mount
FLOOR = -0.52


def _flat(x0, x1, y0, y1, z, step=0.006):
    xs, ys = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step), indexing="ij")
    return np.c_[xs.ravel(), ys.ravel(), np.full(xs.size, float(z))]


def _turned(pts, deg, about):
    a = math.radians(deg)
    R = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    c = np.array([about[0], about[1], 0.0])
    return (np.asarray(pts) - c) @ R.T + c


def _side_counter(face_y=0.06, x=(0.20, 0.80), back=0.50, top=TOP):
    """The chair study's side counter as a camera sees it: its top from the face back, the floor in
    front of it (y < face_y), its face from the floor up."""
    face = np.concatenate([_flat(x[0], x[1], face_y - 0.002, face_y + 0.002, z) for z in np.arange(FLOOR, top, 0.03)])
    return np.concatenate([_flat(x[0], x[1], face_y, back, top), _flat(x[0], x[1], face_y - 0.20, face_y, FLOOR), face])


def test_the_side_counters_edge_is_found_where_the_face_is_the_counter_behind_it():
    tmap = ct.table_map([(_side_counter(), TOP)], (0.50, 0.236))
    (e,) = ct.counter_edges(tmap)
    assert abs(e.point[1] - 0.06) <= 0.5 * ct.CELL_M  # the face lies within half a cell of the boundary
    assert e.out == pytest.approx((0.0, -1.0), abs=1e-6)  # past the edge: toward the arm, -y
    assert abs(e.along[0]) == pytest.approx(1.0) and e.length_m == pytest.approx(0.60, abs=2 * ct.CELL_M)
    assert e.point[0] == pytest.approx(0.50, abs=ct.CELL_M) and e.depth_m >= 0.40


def test_an_edge_at_any_heading_and_both_edges_of_a_corner():
    pts = _turned(_side_counter(), 20.0, (0.50, 0.236))
    (e,) = ct.counter_edges(ct.table_map([(pts, TOP)], (0.50, 0.236)))
    assert math.degrees(math.atan2(e.out[1], e.out[0])) == pytest.approx(-70.0, abs=1.5)
    top = _flat(0.30, 0.90, -0.50, 0.0, 0.22)
    floor = np.concatenate([_flat(0.10, 0.30, -0.50, 0.20, FLOOR), _flat(0.30, 0.90, 0.0, 0.20, FLOOR)])
    edges = ct.counter_edges(ct.table_map([(np.concatenate([top, floor]), 0.22)], (0.60, -0.25)))
    outs = sorted(tuple(np.round(e.out, 3)) for e in edges)
    assert outs == [(-1.0, 0.0), (0.0, 1.0)]
    assert {round(e.point[0], 2) for e in edges if e.out[0] < -0.5} == {0.29}  # within a cell of x 0.30


def test_speckle_and_a_groove_in_the_table_are_no_edge():
    rng = np.random.default_rng(0)
    table = _flat(0.30, 0.90, -0.50, 0.0, 0.22)
    specks = [_flat(x, x + 0.03, y, y + 0.03, 0.10) for x, y in rng.uniform((0.35, -0.45), (0.85, -0.05), (8, 2))]
    assert ct.counter_edges(ct.table_map([(np.concatenate([table] + specks), 0.22)], (0.6, -0.25))) == []
    # flatbot's bench 2026-11-03 16:32: a line of points below the table across it, the table going on past it
    front = _flat(0.36, 0.42, -0.50, 0.0, 0.22)
    gap = _flat(0.42, 0.44, -0.50, 0.0, 0.10)
    back = _flat(0.44, 0.95, -0.50, 0.0, 0.22)
    tmap = ct.table_map([(np.concatenate([front, gap, back]), 0.22)], (0.6, -0.25))
    assert tmap.off.sum() >= ct.EDGE_MIN_CELLS and ct.counter_edges(tmap) == []
    assert ct.counter_edges(None) == []


def test_past_the_counters_shadow_the_floor_still_marks_its_edge_and_nothing_seen_past_it_is_none():
    """A camera over the counter sees its top up to the edge, then nothing (the counter hides the
    floor below its edge), then the floor: the chair study's wrist at the side look saw it 8-13
    cells past the counter's last."""
    top = _flat(0.20, 0.80, 0.06, 0.50, TOP)
    far_floor = _flat(0.20, 0.80, -0.20, -0.10, FLOOR)  # 16 cm of shadow
    (e,) = ct.counter_edges(ct.table_map([(np.concatenate([top, far_floor]), TOP)], (0.50, 0.236)))
    assert e.out == pytest.approx((0.0, -1.0), abs=1e-6) and abs(e.point[1] - 0.06) <= ct.CELL_M
    beyond = _flat(0.20, 0.80, -0.50, -0.30, FLOOR)  # past EDGE_SHADOW_M: not known to be past this edge
    assert ct.counter_edges(ct.table_map([(np.concatenate([top, beyond]), TOP)], (0.50, 0.236))) == []
    assert ct.counter_edges(ct.table_map([(top, TOP)], (0.50, 0.236))) == []  # the view's own end


def test_a_short_edge_is_no_edge():
    pts = _side_counter(x=(0.401, 0.455))  # 3-4 cells of it
    assert ct.counter_edges(ct.table_map([(pts, TOP)], (0.44, 0.236))) == []


def test_an_edge_broken_where_the_floor_past_it_was_hidden_is_an_edge_either_side():
    """Something standing at the counter's edge, or the box near it, hides the floor past part of it:
    the edge's line breaks in two runs, and each is an edge of its own. Only the longest was kept
    before, here the part beyond the box, never the part between it and the arm's base."""
    top = _flat(0.10, 0.90, 0.06, 0.50, TOP)
    floor = _flat(0.10, 0.90, -0.20, 0.06, FLOOR)
    floor = floor[~((floor[:, 0] > 0.36) & (floor[:, 0] < 0.46))]  # hidden over 10 cm of it
    edges = ct.counter_edges(ct.table_map([(np.concatenate([top, floor]), TOP)], (0.55, 0.30)))
    assert len(edges) == 2 and all(e.out == pytest.approx((0.0, -1.0), abs=0.01) for e in edges)
    far, near = edges  # longest first
    assert near.point[0] - 0.5 * near.length_m == pytest.approx(0.10, abs=ct.CELL_M)
    assert near.point[0] + 0.5 * near.length_m == pytest.approx(0.36, abs=ct.CELL_M)
    assert far.point[0] - 0.5 * far.length_m == pytest.approx(0.46, abs=ct.CELL_M)
    assert far.point[0] + 0.5 * far.length_m == pytest.approx(0.90, abs=ct.CELL_M)


def test_a_sinks_rims_do_not_crowd_out_a_short_side_edge():
    """A corner of the counter seen 14 cm along its side, and a 30 cm sink in it: the sink's four rims
    hold more boundary cells than the side edge, which three lines (the cap before) left out."""
    top = _flat(0.10, 0.90, 0.06, 0.60, TOP)
    sink = (top[:, 0] > 0.40) & (top[:, 0] < 0.70) & (top[:, 1] > 0.22) & (top[:, 1] < 0.52)
    pts = np.concatenate(
        [
            top[~sink],
            _flat(0.40, 0.70, 0.22, 0.52, TOP - 0.18),
            _flat(-0.20, 0.10, -0.20, 0.20, FLOOR),  # the floor left of the counter seen only near its front
            _flat(0.10, 0.90, -0.20, 0.06, FLOOR),
        ]
    )
    edges = ct.counter_edges(ct.table_map([(pts, TOP)], (0.40, 0.30)))
    side = [e for e in edges if e.out[0] < -0.9 and e.point[0] < 0.30]  # not the sink's far rim, also out -x
    assert len(side) == 1 and side[0].point[0] == pytest.approx(0.10, abs=ct.CELL_M), [e.summary() for e in edges]
    assert any(e.out[1] < -0.99 and e.point[1] == pytest.approx(0.06, abs=ct.CELL_M) for e in edges)  # the front


def test_the_densest_band_counts_the_last_bin_alone_too():
    """Each direction's offsets are binned EDGE_LINE_TOL_M wide and every bin with the next is a band:
    the last bin is one too, on its own — it was left out where a gap came before it, and a line with
    the largest offset was then found only at a direction off its own, if at all."""
    normals = np.array([[0.0, 1.0]])
    tol = ct.EDGE_LINE_TOL_M
    P = np.array([[0.0, 0.5 * tol], [0.0, 3.5 * tol]] + [[0.1 * k, 9.5 * tol] for k in range(6)])
    most, a, k, q = ct._densest_band(P, normals)
    assert (most, a) == (6, 0) and k <= 9 <= k + 1 and list(q[:, 0]) == [0, 3] + [9] * 6


def test_the_keep_out_holds_the_edge_and_the_solid_behind_it_to_the_floor():
    (e,) = ct.counter_edges(ct.table_map([(_side_counter(), TOP)], (0.50, 0.236)))
    edge, under = ct.counter_keep_out([e], TOP, pad_m=0.02)
    assert edge["name"] == "cabinet/edge0" and under["name"] == "cabinet/counter0"
    for b in (edge, under):
        R = np.asarray(b["rotation"])
        assert np.allclose(R.T @ R, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)
        lo = np.asarray(b["centre"])[2] - b["half"][2]
        assert lo == pytest.approx(TOP - ct.COUNTER_SOLID_M) and lo < FLOOR  # solid to the floor
    # the edge: where it may be (the boundary, half a cell either side), up to the counter's top
    assert edge["centre"][1] - edge["half"][1] <= 0.06 <= edge["centre"][1] + edge["half"][1]
    assert edge["centre"][2] + edge["half"][2] == pytest.approx(TOP)
    # the counter: behind the edge, padded it stands COUNTER_TOP_CLEAR_M (5 mm under) from the top
    assert under["centre"][2] + under["half"][2] + 0.02 == pytest.approx(TOP + ct.COUNTER_TOP_CLEAR_M)
    assert under["centre"][1] - under["half"][1] == pytest.approx(edge["centre"][1] + edge["half"][1])
    assert ct.counter_keep_out([], TOP) == []


def test_the_bench_maps_grooves_are_no_edge_and_its_side_edge_is(tmp_path):
    """Flatbot's bench as its 2026-11-03 runs saw it (their room maps' past-the-edge cells): a line of
    them across the table 0.44 m out, the table going on in front of it (16:32), and the table's
    side 0.18 m to the left, nothing seen past it (15:54, the wrist at the look)."""
    near = _flat(0.38, 0.42, -0.50, -0.20, 0.22)
    groove = _flat(0.42, 0.44, -0.47, -0.25, 0.15)
    table = _flat(0.44, 0.98, -0.50, 0.17, 0.22)
    side = _flat(0.62, 0.88, 0.17, 0.21, 0.15)
    edges = ct.counter_edges(ct.table_map([(np.concatenate([near, groove, table, side]), 0.22)], (0.61, -0.13)))
    assert len(edges) == 1 and edges[0].out == pytest.approx((0.0, 1.0), abs=0.02)
    assert edges[0].point[1] == pytest.approx(0.17, abs=ct.CELL_M)
