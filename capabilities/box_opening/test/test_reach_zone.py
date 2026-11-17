"""The line-up's boundary (reach_zone.Zone) and the proof that fills it (tasks/reach_probe)."""

import math

import numpy as np
import pytest
from conftest import CFG, flatbot_urdf

from box_opening import guide
from box_opening.reach_zone import FAILS, UNASKED, WORKS, Zone


def _zone(works=None, x=(0.0, 1.0), y=(-0.5, 0.5)):
    """A grid over x, y, every cell asked: `works(cx, cy)` says which work (None: left unasked)."""
    z = Zone.blank(x, y, lid_z=0.30, key="k")
    if works is not None:
        for i in range(z.cells.shape[0]):
            for j in range(z.cells.shape[1]):
                z.set(i, j, works(*z.centre(i, j)))
    return z


def _band(cx, cy):
    """Works from 0.50 to 0.80 m out, 0.20 m to either side."""
    return 0.475 < cx < 0.825 and abs(cy) < 0.225


# -- the grid ----------------------------------------------------------------------------
def test_a_blank_grid_spans_its_ends_and_nothing_in_it_is_asked():
    z = Zone.blank((0.0, 0.95), (-0.95, 0.95))
    assert z.cells.shape == (20, 39)
    assert z.centre(0, 0) == (0.0, -0.95)
    assert z.centre(19, 38) == pytest.approx((0.95, 0.95))
    assert (z.cells == UNASKED).all()
    assert z.counts() == {"works": 0, "fails": 0, "unasked": 20 * 39}


def test_a_place_is_its_nearest_cells_and_outside_the_grid_nothing_works():
    z = _zone(_band)
    assert z.index((0.62, 0.02)) == z.index((0.60, 0.0)) == (12, 10)
    assert z.index((0.626, 0.0)) == (13, 10)
    assert z.index((1.03, 0.0)) is None and z.index((0.5, -0.53)) is None
    assert z.state((0.6, 0.0)) == WORKS and z.state((0.3, 0.0)) == FAILS
    assert z.state((1.2, 0.0)) == FAILS  # outside the grid
    assert not z.inside((1.2, 0.0)) and z.clear_m((1.2, 0.0)) == 0.0


def test_inside_is_a_working_cell_with_nothing_that_fails_within_the_margin():
    z = _zone(_band)
    # the band's cells run x 0.50..0.80: their outer sides are at 0.475 and 0.825
    assert z.clear_m((0.65, 0.0)) == pytest.approx(0.175)
    assert z.inside((0.65, 0.0))
    assert z.clear_m((0.50, 0.0)) == pytest.approx(0.025)
    assert z.state((0.50, 0.0)) == WORKS and not z.inside((0.50, 0.0))  # on the boundary: not in it
    assert z.inside((0.51, 0.0))  # 35 mm from the cells that fail
    assert not z.inside((0.65, 0.21))  # 15 mm from the side
    assert z.inside((0.50, 0.0), margin_m=0.02)


def test_an_unasked_cell_is_not_inside_and_holds_its_neighbours_out():
    z = _zone(_band)
    i, j = z.index((0.65, 0.0))
    z.cells[i, j] = UNASKED
    z._deep.clear()
    assert not z.inside((0.65, 0.0))
    assert not z.inside((0.70, 0.0))  # its neighbour's centre is 25 mm from it
    assert z.inside((0.75, 0.0))


def test_a_grid_that_reaches_its_own_edge_is_not_inside_there():
    z = _zone(lambda cx, cy: True, x=(0.0, 0.5), y=(-0.2, 0.2))
    assert z.inside((0.25, 0.0))
    assert z.clear_m((0.5, 0.0)) == pytest.approx(0.025)
    assert not z.inside((0.5, 0.0))  # 25 mm from the grid's outside


def test_a_deployment_may_trust_less_than_the_proof_finds():
    z = _zone(_band)
    near = z.within(far_m=0.70)
    assert near.inside((0.60, 0.0)) and not near.inside((0.75, 0.0)) and z.inside((0.75, 0.0))
    assert near.state((0.75, 0.0)) == FAILS and near.to_zone((0.95, 0.0)) == pytest.approx((-0.30, 0.0))
    assert not z.within(near_m=0.60).inside((0.55, 0.0)) and z.within(near_m=0.60).inside((0.70, 0.0))
    assert (z.within().cells == z.cells).all() and z.within().key == "k"


# -- which way to it ---------------------------------------------------------------------
def test_to_zone_points_at_the_nearest_cell_that_is_inside():
    z = _zone(_band)
    assert z.to_zone((0.65, 0.0)) is None  # inside already
    dx, dy = z.to_zone((0.95, 0.0))  # too far out: the nearest deep cell is the band's far row but one
    assert (dx, dy) == pytest.approx((-0.20, 0.0))
    dx, dy = z.to_zone((0.30, 0.0))
    assert (dx, dy) == pytest.approx((0.25, 0.0))
    dx, dy = z.to_zone((0.65, 0.40))  # off to the arm's left: back toward the middle
    assert dx == pytest.approx(0.0) and dy == pytest.approx(-0.25)
    assert _zone(lambda cx, cy: False).to_zone((0.6, 0.0)) is None  # nothing works anywhere


def test_the_guide_turns_it_into_the_users_move():
    z = _zone(_band)
    assert guide.instruction(z.to_zone((0.95, 0.0)))[:2] == ("closer", pytest.approx(0.20))
    assert guide.instruction(z.to_zone((0.30, 0.0)))[:2] == ("back", pytest.approx(0.25))
    assert guide.instruction(z.to_zone((0.65, 0.40)))[0] == "left"  # the box stands to the left: toward it
    assert guide.instruction(z.to_zone((0.65, -0.40)))[0] == "right"


def test_reach_says_nothing_of_a_place_it_cannot_judge_yet():
    z = _zone()
    assert z.reach((0.6, 0.0)) is None  # its own cell unasked
    z.set(*z.index((0.6, 0.0)), False)
    assert not z.done() and z.reach((0.6, 0.0)) is None  # fails, and what works is not known yet
    full = _zone(lambda cx, cy: False)
    assert full.done()
    r = full.reach((0.6, 0.0))
    assert r is not None and not r.inside and r.to_zone is None  # asked everywhere: the arm cannot
    band = _zone(_band)
    assert band.reach((0.65, 0.0), planned=True) == guide.Reach(True, None, True)
    out = band.reach((0.95, 0.0))
    assert not out.inside and out.to_zone == pytest.approx((-0.20, 0.0)) and out.planned is None


def test_a_line_up_from_too_far_to_the_question():
    """The grid's words through the guide: move closer, hold still, then the question."""
    z, g = _zone(_band), guide.Guide()
    t = 0.0
    say = g.update(t, guide.Fix(t, (0.95, 0.0)), z.reach((0.95, 0.0)))
    assert say.state == guide.MOVE and say.message == "Move closer, about 20 cm"
    for k in range(1, 16):  # the box where it works, standing still
        t = 1.0 + 0.1 * k
        say = g.update(t, guide.Fix(t, (0.65, 0.0)), z.reach((0.65, 0.0), planned=True))
    assert say.state == guide.READY and g.press(guide.OPEN, t)


# -- filling it ----------------------------------------------------------------------------
def test_the_boxs_own_cell_is_asked_first_then_the_lattice_outward_from_it():
    z = _zone()
    order = z.order(focus=(0.62, 0.07))
    assert order[0] == z.index((0.62, 0.07)) == (12, 11)  # off the lattice, and still first
    lattice = order[1:]
    assert all(i % 2 == 0 and j % 2 == 0 for i, j in lattice)
    assert len(lattice) == 11 * 11
    d = [math.hypot(z.centre(i, j)[0] - 0.62, z.centre(i, j)[1] - 0.07) for i, j in lattice]
    assert d == sorted(d)


def test_off_the_lattice_only_the_cells_beside_one_that_works_are_asked():
    z = _zone()
    for i in range(0, z.cells.shape[0], 2):
        for j in range(0, z.cells.shape[1], 2):
            z.set(i, j, False)
    assert z.order() == [] and z.done()  # the lattice found nothing: nothing more is asked
    z.set(12, 10, True)
    beside = sorted(z.order())
    assert beside == sorted((12 + di, 10 + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di + dj) % 2 or di % 2)
    assert not z.done()


def test_a_working_cells_surroundings_are_asked_before_the_lattice_goes_on():
    """inside() needs a cell's neighbours: they are asked as soon as it works, so a box in place is judged
    after nine proofs and not after the whole lattice."""
    z = _zone()
    own = z.index((0.62, 0.07))
    z.set(*own, True)  # the box's own cell, asked first, works
    order = z.order(focus=(0.62, 0.07))
    assert sorted(order[:8]) == sorted((own[0] + di, own[1] + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj)
    assert all(i % 2 == 0 and j % 2 == 0 for i, j in order[8:])  # then the lattice, outward from the box
    for i, j in order[:8]:
        z.set(i, j, True)
    assert z.inside(z.centre(*own)) and z.reach(z.centre(*own)).inside


def test_fill_finds_the_boundary_without_asking_every_cell(monkeypatch):
    from box_opening.tasks import reach_probe

    asked = []

    def proof(_ctx, _cfg, top, table_z=None):
        asked.append(top)
        return math.hypot(top[0] - 0.65, top[1]) < 0.18, "", 15  # works within 18 cm of (0.65, 0)

    monkeypatch.setattr(reach_probe, "press_plans", proof)
    z = Zone.blank((0.0, 0.95), (-0.95, 0.95), lid_z=0.30)
    assert reach_probe.fill(z, None, None, focus=(0.9, 0.3), n=1) == 1
    assert asked[0][:2] == pytest.approx(z.centre(*z.index((0.9, 0.3)))) and asked[0][2] == 0.30
    n = 1 + reach_probe.fill(z, None, None, focus=(0.9, 0.3))
    assert z.done() and n == len(asked) < z.cells.size // 2  # 20 x 39 cells; the lattice is a quarter of them
    for x, y in ((0.65, 0.0), (0.60, 0.05), (0.70, -0.05)):
        assert z.inside((x, y))
    for x, y in ((0.65, 0.20), (0.40, 0.0), (0.90, 0.0)):
        assert not z.inside((x, y))
    # every cell that works was asked, and so was every cell beside one
    works = np.argwhere(z.cells == WORKS)
    assert len(works) == sum(
        1 for i in range(20) for j in range(39) if math.hypot(*np.subtract(z.centre(i, j), (0.65, 0.0))) < 0.18
    )
    assert z.to_zone((0.9, 0.3)) is not None


def test_fill_stops_at_its_budget(monkeypatch):
    from box_opening.tasks import reach_probe

    monkeypatch.setattr(reach_probe, "press_plans", lambda *_a, **_k: (False, "no", None))
    ticks = iter(range(100))  # a clock that reads 0, 1, 2, ...: one tick as it starts, one before each cell
    z = Zone.blank((0.0, 0.5), (-0.5, 0.5), lid_z=0.30)
    assert reach_probe.fill(z, None, None, budget_s=3, clock=lambda: next(ticks)) == 2


def test_cells_past_the_deployments_bound_are_not_asked_and_not_waited_for():
    """reach_probe --far-m: the engine throws away what works past line_up_far_m, so no proof is spent there."""
    z = _zone()
    capped = z.order(focus=(0.95, 0.0), far_m=0.80)
    assert capped and all(math.hypot(*z.centre(i, j)) <= 0.80 + 1e-9 for i, j in capped)
    assert z.index((0.95, 0.0)) not in capped and z.order(focus=(0.95, 0.0))[0] == z.index((0.95, 0.0))
    assert z.index((0.80, 0.0)) in capped  # a centre on the bound is within it, as Zone.within has it
    assert capped == [ij for ij in z.order(focus=(0.95, 0.0)) if ij in capped]  # the order is the same
    assert z.order(focus=(0.95, 0.0), far_m=None) == z.order(focus=(0.95, 0.0))
    for i, j in capped:
        z.set(i, j, False)
    assert z.order(far_m=0.80) == [] and z.done(far_m=0.80) and not z.done()
    z.set(*z.index((0.80, 0.0)), True)  # a cell that works on the bound: its neighbours further out stay unasked
    beside = z.order(far_m=0.80)
    assert beside and all(math.hypot(*z.centre(i, j)) <= 0.80 + 1e-9 for i, j in beside)
    assert len(beside) < len([ij for ij in z.order() if abs(ij[0] - 16) <= 1 and abs(ij[1] - 10) <= 1])


def test_the_engines_copy_does_not_wait_for_cells_past_its_bound():
    """What the probe leaves unasked past the bound is outside for the engine, not something still to come:
    with everything nearer asked and nothing working, the panel says the arm cannot, not 'checking'."""
    z = _zone()
    for i, j in z.order(far_m=0.80):
        z.set(i, j, False)
    assert z.reach((0.60, 0.0)) is None  # the whole grid: cells are still to be asked
    near = z.within(far_m=0.80)
    assert near.state((0.95, 0.0)) == FAILS and z.state((0.95, 0.0)) == UNASKED
    assert near.done() and near.counts()["unasked"] > 0  # off the lattice, nearer than the bound: never asked
    r = near.reach((0.60, 0.0))
    assert r is not None and not r.inside and r.to_zone is None
    # nearer than a near bound the proof is still asked, and still waited for
    assert z.within(near_m=0.30).state((0.15, 0.0)) == UNASKED


def test_fill_leaves_what_is_past_the_bound_unasked(monkeypatch):
    from box_opening.tasks import reach_probe

    asked = []

    def proof(_ctx, _cfg, top, table_z=None):
        asked.append(top)
        return math.hypot(top[0] - 0.65, top[1]) < 0.18, "", 15  # works within 18 cm of (0.65, 0): out to 0.80

    monkeypatch.setattr(reach_probe, "press_plans", proof)
    z = Zone.blank((0.0, 0.95), (-0.95, 0.95), lid_z=0.30)
    n = reach_probe.fill(z, None, None, focus=(0.9, 0.3), far_m=0.70)
    assert n == len(asked) and z.done(far_m=0.70) and not z.done()
    assert max(math.hypot(x, y) for x, y, _z in asked) <= 0.70 + 1e-9
    assert z.state((0.65, 0.0)) == WORKS and z.state((0.75, 0.0)) == UNASKED  # works there, and was not asked
    assert z.within(far_m=0.70).inside((0.60, 0.0))
    more = reach_probe.fill(z, None, None, focus=(0.9, 0.3))  # the same grid under no bound: the rest is asked
    assert more > 0 and z.done() and z.state((0.75, 0.0)) == WORKS


# -- for the engine's line-up: the probe's serve mode ------------------------------------------
@pytest.fixture
def served(monkeypatch, tmp_path):
    """reach_probe.Server on a stand-in proof (works within 18 cm of (0.65, 0)) and a small blank grid; its
    clock ticks 0.3 s a proof. Returns (server, what it said, the places it proved)."""
    from box_opening.tasks import reach_probe

    said, proved, now = [], [], [0.0]

    def proof(_ctx, _cfg, top, table_z=None, q_from=None):
        proved.append((tuple(round(float(v), 3) for v in top), table_z))
        now[0] += 0.3
        return math.hypot(top[0] - 0.65, top[1]) < 0.18, "no" * 400, 15

    monkeypatch.setattr(reach_probe, "press_plans", proof)
    monkeypatch.setattr(reach_probe, "reach_span", lambda model, lid_z: ((0.0, 0.95), (-0.5, 0.5)))
    server = reach_probe.Server(None, None, "k", zone_dir=tmp_path, say=said.append, clock=lambda: now[0])
    server.ctx = type("C", (), {"model": None})()
    return server, said, proved


def test_the_server_waits_for_a_box_then_says_the_boundary_kept_for_its_height(served, tmp_path):
    from box_opening.tasks import reach_probe

    server, said, proved = served
    assert server.step() is False and said == [] and proved == []  # no box yet: nothing to ask
    kept = _zone(_band, x=(0.0, 0.95))
    kept.lid_z = 0.32
    reach_probe.save_zone(kept, tmp_path)
    server.hear({"box": [0.9, 0.3, 0.315], "table_z": 0.22, "prove": False})
    server.step()
    assert list(said[0]) == ["zone"] and Zone.from_dict(said[0]["zone"]).counts() == kept.counts()
    assert proved == []  # everything worth asking was asked on an earlier day


def test_the_server_fills_the_boundary_from_the_box_outward_and_says_it_as_it_goes(served, tmp_path):
    from box_opening.tasks import reach_probe

    server, said, proved = served
    server.hear({"box": [0.62, 0.07, 0.315], "table_z": 0.22, "prove": False})
    assert server.step() is True
    assert Zone.from_dict(said[0]["zone"]).counts()["unasked"] == 20 * 21  # the blank grid, said at once
    assert proved == [((0.60, 0.05, 0.32), None)]  # the box's own cell, at its centre and the grid's height
    for _ in range(40):
        server.step()
    zones = [Zone.from_dict(m["zone"]) for m in said if "zone" in m]
    assert len(zones) >= 10 and zones[-1].counts()["works"] > zones[1].counts()["works"]
    assert len(zones) <= 1 + 41 * 0.3 / reach_probe.ZONE_SAY_S + 1  # no oftener than its period
    assert all(table_z is None for _top, table_z in proved)  # a cell is asked on the box's own foot
    assert Zone.from_json(reach_probe.zone_path("k", 0.315, tmp_path).read_text()).counts()["unasked"] < 20 * 21
    while server.step():
        pass
    assert server.zone.done() and server.step() is False
    server.close()
    assert Zone.from_json(reach_probe.zone_path("k", 0.315, tmp_path).read_text()).counts() == server.zone.counts()
    assert Zone.from_dict(said[-1]["zone"]).counts() == server.zone.counts()


def test_the_boxs_own_place_is_proven_first_where_it_is_asked_for_and_once(served):
    server, said, proved = served
    server.hear({"box": [0.652, 0.013, 0.3149], "table_z": 0.2159, "prove": True})
    server.step()
    assert proved == [((0.652, 0.013, 0.315), 0.2159)]  # the box where it stands, on the table as measured
    (proof,) = (m["proof"] for m in said if "proof" in m)
    assert proof["ok"] is True and proof["xyz"] == [0.652, 0.013, 0.3149] and proof["tilt_deg"] == 15
    assert len(proof["why"]) == 300 and proof["s"] == pytest.approx(0.3)
    server.step()
    assert proved[1][1] is None  # proven: the next proof is a cell of the boundary
    server.hear({"box": [0.66, 0.02, 0.3149], "table_z": 0.2159, "prove": True})  # 11 mm on: the same place
    server.step()
    assert sum(1 for m in said if "proof" in m) == 1
    server.hear({"box": [0.70, 0.02, 0.3149], "table_z": 0.2159, "prove": True})  # the chair moved: proven again
    server.step()
    assert [m["proof"]["xyz"][0] for m in said if "proof" in m] == [0.652, 0.70]
    server.hear({"box": [0.70, 0.02, 0.3149], "table_z": 0.2159, "prove": False})
    n = len(proved)
    server.step()
    assert len(proved) == n + 1 and proved[-1][1] is None  # not asked for: the boundary goes on filling


VIEW = [0.4425, -1.5718, -2.9033, -2.1048, -0.2, -1.4, 2.6]  # an arm at rest but for its wrist, turned to look


@pytest.fixture
def starts(served, monkeypatch):
    """What each of the served stand-in's proofs was started from: its q_from, None where it was given none."""
    from box_opening.tasks import reach_probe

    stand_in, got = reach_probe.press_plans, []

    def proof(*a, **k):
        got.append(k.get("q_from"))
        return stand_in(*a, **k)

    monkeypatch.setattr(reach_probe, "press_plans", proof)
    return got


def test_the_boxs_own_proof_starts_from_the_joints_the_engine_names_and_the_cells_from_home(served, starts):
    """q_from: where the run will start its reach from after a wrist line-up, the wrist turned to its view.
    The box's own proof is made from there and says so; the boundary's cells are one grid for every start."""
    server, said, proved = served
    server.hear({"box": [0.652, 0.013, 0.3149], "table_z": 0.2159, "prove": True, "q_from": VIEW})
    server.step()
    assert starts == [VIEW] and proved == [((0.652, 0.013, 0.315), 0.2159)]
    (proof,) = (m["proof"] for m in said if "proof" in m)
    assert proof["q_from"] is True and proof["ok"] is True and proof["xyz"] == [0.652, 0.013, 0.3149]
    for _ in range(3):
        server.step()  # proven: the boundary's cells, each from HOME
    assert starts[1:] == [None, None, None] and all(table_z is None for _top, table_z in proved[1:])
    server.hear({"box": [0.652, 0.013, 0.3149], "table_z": 0.2159, "prove": True, "q_from": list(VIEW)})
    server.step()
    assert sum(1 for m in said if "proof" in m) == 1  # the same place from the same start: not proven again


def test_a_box_line_without_q_from_is_proven_from_home_as_before_and_says_so(served, starts):
    server, said, _proved = served
    server.hear({"box": [0.652, 0.013, 0.3149], "table_z": 0.2159, "prove": True})
    server.step()
    assert starts == [None] and [m["proof"]["q_from"] for m in said if "proof" in m] == [False]
    # the engine learns the view's joints: the same place is proven again, from there, once
    server.hear({"box": [0.652, 0.013, 0.3149], "table_z": 0.2159, "prove": True, "q_from": VIEW})
    server.step()
    server.step()
    assert starts == [None, VIEW, None] and [m["proof"]["q_from"] for m in said if "proof" in m] == [False, True]
    # and a line with none again drops them: each line says the whole of what it asks
    server.hear({"box": [0.70, 0.02, 0.3149], "table_z": 0.2159, "prove": True})
    server.step()
    assert starts[-1] is None and [m["proof"]["q_from"] for m in said if "proof" in m][-1] is False


@pytest.mark.parametrize("said_from", [VIEW[:6], VIEW + [0.0], "home", [None] * 7, VIEW[:6] + [float("nan")], 7])
def test_a_q_from_that_is_not_seven_joints_is_not_used_and_the_answer_says_so(said_from, served, starts, monkeypatch):
    from box_opening.tasks import reach_probe

    warned = []
    monkeypatch.setattr(reach_probe.log, "warning", warned.append)
    server, said, _proved = served
    server.hear({"box": [0.652, 0.013, 0.3149], "table_z": 0.2159, "prove": True, "q_from": said_from})
    server.step()
    assert starts == [None] and [m["proof"]["q_from"] for m in said if "proof" in m] == [False]
    assert len(warned) == 1 and "q_from" in warned[0]


def test_a_lid_at_another_height_gets_its_own_boundary(served, tmp_path):
    from box_opening.tasks import reach_probe

    server, said, _proved = served
    server.hear({"box": [0.65, 0.0, 0.315]})
    for _ in range(3):
        server.step()
    server.hear({"box": [0.65, 0.0, 0.36]})
    server.step()
    assert server.zone.lid_z == 0.36
    assert reach_probe.zone_path("k", 0.315, tmp_path).exists()  # the first is kept as the second is begun
    heights = [m["zone"]["lid_z"] for m in said if "zone" in m]
    assert heights[0] == 0.32 and heights[-1] == 0.36 and heights == sorted(heights)


def test_a_grid_goes_on_serving_a_lid_read_a_little_past_its_buckets_edge():
    """Each frame reads the lid's height anew, and a lid near an odd centimetre reads either side of it."""
    from box_opening import reach_zone

    z = _zone(_band)  # asked for a lid at 0.30: its own bucket is 0.29 to 0.31
    assert z.serves(0.30) and z.serves(0.291) and z.serves(0.309)
    assert z.serves(0.311) and z.serves(0.314) and z.serves(0.289) and z.serves(0.286)  # past the edge, in the margin
    assert not z.serves(0.316) and not z.serves(0.284)  # well outside: another grid's
    assert not z.serves(0.32) and not z.serves(0.28) and not z.serves(0.36)
    # a grid never serves the height its neighbour was asked at, and a height always has a grid that serves it
    assert 0 < reach_zone.LID_Z_HOLD_M < reach_zone.LID_Z_BUCKET_M / 2
    for mm in range(250, 400):
        own = _zone()
        own.lid_z = reach_zone.lid_bucket(mm / 1000.0)
        assert own.serves(mm / 1000.0)
    assert not Zone.blank((0.0, 1.0), (-0.5, 0.5)).serves(0.30)  # asked at no height


def test_a_lid_read_either_side_of_a_buckets_edge_does_not_swap_the_boundary(served, tmp_path):
    from box_opening.tasks import reach_probe

    server, said, proved = served
    for k in range(12):  # 0.311 is the 0.32 grid's, 0.309 the 0.30 grid's: frame about
        server.hear({"box": [0.65, 0.0, 0.309 if k % 2 else 0.311]})
        server.step()
    assert [m["zone"]["lid_z"] for m in said if "zone" in m][:1] == [0.32]
    assert {m["zone"]["lid_z"] for m in said if "zone" in m} == {0.32} and server.zone.lid_z == 0.32
    assert len({top for top, _t in proved}) == 12 and {top[2] for top, _t in proved} == {0.32}  # one grid's cells
    assert not reach_probe.zone_path("k", 0.30, tmp_path).exists()  # no second grid was begun
    server.hear({"box": [0.65, 0.0, 0.304]})  # 16 mm under: well outside, the 0.30 grid's
    server.step()
    assert server.zone.lid_z == 0.30 and proved[-1][0] == (0.65, 0.0, 0.30)
    assert _asked(Zone.from_json(reach_probe.zone_path("k", 0.32, tmp_path).read_text())) == 12  # the first is kept
    server.hear({"box": [0.65, 0.0, 0.3145]})  # back over the edge, inside the margin: it stays
    server.step()
    assert server.zone.lid_z == 0.30 and _asked(server.zone) == 2
    server.hear({"box": [0.65, 0.0, 0.3155]})  # past the margin: the 0.32 grid again, as it was kept
    server.step()
    assert server.zone.lid_z == 0.32 and _asked(server.zone) == 13


def test_the_served_loop_hears_the_engines_lines_and_ends_when_it_says_or_goes(served):
    import os

    from box_opening.tasks import reach_probe

    server, said, _proved = served
    r, w = os.pipe()
    os.write(w, b'{"box": [0.65, 0.0, 0.315], "prove": true}\nnot a line of the engine\'s\n{"quit": true}\n')
    reach_probe.serve(server, r)
    assert server.quit and [m["proof"]["ok"] for m in said if "proof" in m] == []  # quit heard before any proof
    server.quit = False
    os.write(w, b'{"box": [0.65, 0.0, 0.315], "prove": true}\n')
    os.close(w)  # the engine is gone: the pipe's end is its last word
    reach_probe.serve(server, r)
    os.close(r)
    assert [m["proof"]["ok"] for m in said if "proof" in m] == [True]


def test_the_server_asks_no_cell_past_the_bound_it_was_started_with(served, tmp_path):
    from box_opening.tasks import reach_probe

    server, said, proved = served
    server.far_m = 0.70
    server.hear({"box": [0.90, 0.0, 0.315], "prove": False})  # the user is still too far
    while server.step():
        pass
    assert proved and max(math.hypot(x, y) for (x, y, _z), _t in proved) <= 0.70 + 1e-9
    assert server.zone.done(far_m=0.70) and server.zone.state((0.65, 0.0)) == WORKS
    server.close()
    # the bound is no part of what a grid is: it is kept under the same name, and read back under another
    assert sorted(p.name for p in tmp_path.iterdir()) == ["zone_k_z+0320.json"]
    free = reach_probe.Server(server.ctx, None, "k", zone_dir=tmp_path, say=said.append, clock=lambda: 0.0)
    assert free.far_m is None
    free.hear({"box": [0.90, 0.0, 0.315], "prove": False})
    n = len(proved)
    free.step()
    assert free.zone.counts()["works"] >= server.zone.counts()["works"]  # what was asked is not asked again
    assert proved[n][0][:2] == (0.90, 0.0)  # the box's own cell, past the old bound


@pytest.fixture
def main_on_stand_ins(monkeypatch, tmp_path):
    """reach_probe.main with no robot behind it: the stand-in proof (works within 18 cm of (0.65, 0)), a
    small grid, the key "k", grids kept in tmp_path, and its own process left at its priority. Returns
    (run, served): run(*more) is main's exit code for --urdf, --container, --zone-dir and `more`; `served`
    holds the Server of each --serve, which is not looped."""
    import os
    import signal

    from box_opening.tasks import reach_probe

    served = []
    monkeypatch.setattr(os, "nice", lambda _by: 0)
    monkeypatch.setattr(signal, "signal", lambda _signum, _handler: None)  # the test process keeps its own SIGTERM
    monkeypatch.setattr(reach_probe, "setup", lambda *_a: (type("C", (), {"model": None})(), None))
    monkeypatch.setattr(reach_probe, "zone_key", lambda *_a, **_k: "k")
    monkeypatch.setattr(reach_probe, "reach_span", lambda model, lid_z: ((0.0, 0.95), (-0.5, 0.5)))
    monkeypatch.setattr(
        reach_probe,
        "press_plans",
        lambda _ctx, _cfg, top, table_z=None: (math.hypot(top[0] - 0.65, top[1]) < 0.18, "", 15),
    )
    monkeypatch.setattr(reach_probe, "serve", served.append)

    def run(*more):
        return reach_probe.main(["--urdf", "arm.urdf", "--container", "box.yaml", "--zone-dir", str(tmp_path), *more])

    return run, served


def test_far_m_on_the_command_line_bounds_the_served_probe_and_the_offline_fill(main_on_stand_ins, tmp_path, capsys):
    from box_opening.tasks import reach_probe

    run, served = main_on_stand_ins
    assert run("--serve", "--far-m", "0.8") == 0 and served[-1].far_m == 0.8
    assert run("--serve") == 0 and served[-1].far_m is None
    assert run("--serve", "--far-m", "0") == 0 and served[-1].far_m is None  # 0: no bound
    assert run("--lid-z", "0.315", "--far-m", "0.7") == 0  # ends: what is past the bound is not waited for
    kept = Zone.from_json(reach_probe.zone_path("k", 0.315, tmp_path).read_text())
    assert kept.done(far_m=0.7) and not kept.done() and kept.state((0.75, 0.0)) == UNASKED
    assert '"done": true' in capsys.readouterr().out
    assert run("--lid-z", "0.315") == 0  # the same file under no bound: the rest of it is asked
    whole = Zone.from_json(reach_probe.zone_path("k", 0.315, tmp_path).read_text())
    assert whole.done() and whole.state((0.75, 0.0)) == WORKS
    assert sorted(p.name for p in tmp_path.iterdir()) == ["zone_k_z+0320.json"]


# -- however the probe is ended, what it asked is kept -----------------------------------------
def _asked(zone):
    return zone.cells.size - zone.counts()["unasked"]


def test_quit_keeps_the_cells_asked_since_the_last_save(served, tmp_path, monkeypatch):
    """The engine's {"quit": true}, heard after the proof in flight: fewer cells than a periodic save needs."""
    import os

    from box_opening.tasks import reach_probe

    server, _said, proved = served
    r, w = os.pipe()
    stand_in = reach_probe.press_plans

    def proof(*a, **k):
        if len(proved) == 2:
            os.write(w, b'{"quit": true}\n')  # said while the third proof is in flight
        return stand_in(*a, **k)

    monkeypatch.setattr(reach_probe, "press_plans", proof)
    os.write(w, b'{"box": [0.65, 0.0, 0.315]}\n')
    reach_probe.serve(server, r)
    os.close(r)
    os.close(w)
    assert len(proved) == 3 < reach_probe.SAVE_EVERY
    kept = Zone.from_json(reach_probe.zone_path("k", 0.315, tmp_path).read_text())
    assert _asked(kept) == 3 and (kept.cells == server.zone.cells).all()


def test_a_terminate_keeps_the_cells_asked_since_the_last_save(served, tmp_path, monkeypatch):
    """The engine ends its children with SIGTERM and kills them 2 s later; a proof may take that long. The
    probe ends at the signal, inside the proof in flight, and keeps the grid on its way out."""
    import os
    import signal

    from box_opening.tasks import reach_probe

    server, said, proved = served
    stand_in = reach_probe.press_plans

    def proof(*a, **k):
        if len(proved) == 3:
            os.kill(os.getpid(), signal.SIGTERM)
        return stand_in(*a, **k)

    monkeypatch.setattr(reach_probe, "press_plans", proof)
    r, w = os.pipe()
    os.write(w, b'{"box": [0.65, 0.0, 0.315]}\n')
    before = signal.getsignal(signal.SIGTERM)
    assert hasattr(reach_probe, "end_on_terminate")  # without it the signal would end the test run itself
    try:
        reach_probe.end_on_terminate(server)
        with pytest.raises(SystemExit) as end:
            reach_probe.serve(server, r)
        os.kill(os.getpid(), signal.SIGTERM)  # a second one, the probe already ending: nothing more
    finally:
        signal.signal(signal.SIGTERM, before)
        os.close(r)
        os.close(w)
    assert end.value.code == 0 and server.quit
    kept = Zone.from_json(reach_probe.zone_path("k", 0.315, tmp_path).read_text())
    assert _asked(kept) in (3, 4) and _asked(kept) < reach_probe.SAVE_EVERY  # the proof in flight may be lost
    assert (kept.cells == server.zone.cells).all()
    assert Zone.from_dict(said[-1]["zone"]).counts() == kept.counts()  # and said, as at any end


def test_the_served_probe_is_started_to_end_on_a_terminate(main_on_stand_ins, monkeypatch):
    import signal

    run, served = main_on_stand_ins
    set_ = []
    monkeypatch.setattr(signal, "signal", lambda signum, handler: set_.append((signum, handler)))
    assert run("--serve") == 0
    ((signum, handler),) = set_
    assert signum == signal.SIGTERM and not served[0].quit
    with pytest.raises(SystemExit):
        handler(signal.SIGTERM, None)
    assert served[0].quit
    handler(signal.SIGTERM, None)  # already ending: the save under way is not cut short


def test_the_engines_own_way_of_ending_a_real_probe_keeps_its_grid(tmp_path):
    """End to end on flatbot's arm: reach_probe --serve as the engine's child (line_up.Child), a box said,
    and once a few cells are asked the engine's stop(), a terminate with a proof in flight. The probe ends by
    itself (exit 0, not the kill 2 s later) and the grid it leaves holds every cell it said it had asked."""
    import sys
    import time

    from box_opening import line_up
    from box_opening.tasks import reach_probe

    argv = [sys.executable, "-m", "box_opening.tasks.reach_probe", "--serve", "--urdf", flatbot_urdf()]
    argv += ["--body-layers", "", "--container", str(CFG), "--zone-dir", str(tmp_path), "--far-m", "0.8"]
    logged = []
    child = line_up.Child(argv, None, log=logged.append)
    try:
        child.send({"box": [0.65, -0.10, 0.315], "table_z": 0.216, "prove": False})
        zones, t0 = [], time.monotonic()
        while (not zones or _asked(zones[-1]) < 2) and child.alive() and time.monotonic() - t0 < 120.0:
            zones += [Zone.from_dict(m["zone"]) for _t, m in child.messages() if "zone" in m]
            time.sleep(0.05)
        assert zones and 2 <= _asked(zones[-1]) < reach_probe.SAVE_EVERY, logged[-5:]
        assert list(tmp_path.iterdir()) == []  # nothing kept yet: fewer cells than a periodic save
    finally:
        child.stop()
    assert child.proc.returncode == 0, logged[-5:]
    (path,) = tmp_path.iterdir()
    kept = Zone.from_json(path.read_text())
    assert kept.key == zones[-1].key and kept.lid_z == 0.32 and _asked(kept) >= _asked(zones[-1])
    assert ((kept.cells == zones[-1].cells) | (zones[-1].cells == UNASKED)).all()


def test_a_grid_that_cannot_be_kept_is_still_served(served, monkeypatch):
    """A folder that is not the probe's to write (or a full disk) costs the next line-up its proofs, not
    this one its answers: the probe says so in its log and goes on."""
    from box_opening.tasks import reach_probe

    def refused(_zone, _where=None):
        raise PermissionError("not this user's folder")

    warned = []
    monkeypatch.setattr(reach_probe, "save_zone", refused)
    monkeypatch.setattr(reach_probe.log, "warning", warned.append)
    server, said, proved = served
    server.hear({"box": [0.65, 0.0, 0.315]})
    for _ in range(reach_probe.SAVE_EVERY + 5):
        assert server.step() is True
    server.close()
    assert len(proved) == reach_probe.SAVE_EVERY + 5
    assert len(warned) == 2 and all("not kept" in w and "not this user's folder" in w for w in warned)
    assert _asked(Zone.from_dict(said[-1]["zone"])) == len(proved)


def test_the_probe_lowers_its_own_priority_before_it_works_and_works_on_where_it_may_not(monkeypatch, capsys):
    """It runs proofs back to back beside the arm's driver: whatever it is asked, it is niced first."""
    import os

    from box_opening.tasks import reach_probe

    did = []
    monkeypatch.setattr(os, "nice", lambda by: did.append(("nice", by)))
    monkeypatch.setattr(reach_probe, "setup", lambda *_a: did.append("setup") or (None, None))
    monkeypatch.setattr(reach_probe, "press_plans", lambda *_a, **_k: did.append("proof") or (True, "", 15))
    argv = ["--urdf", "arm.urdf", "--container", "box.yaml", "--at", "0.65", "0.0", "0.315"]
    assert reach_probe.main(argv) == 0
    assert did == [("nice", 10), "setup", "proof"] and reach_probe.NICE == 10

    def refused(_by):
        raise PermissionError("not this user's to change")

    monkeypatch.setattr(os, "nice", refused)
    assert reach_probe.main(argv) == 0  # a proof at the priority it was started with, not none
    monkeypatch.delattr(os, "nice")  # a platform that has no niceness
    assert reach_probe.main(argv) == 0
    assert capsys.readouterr().out.count('"ok": true') == 3


# -- kept on disk --------------------------------------------------------------------------
def test_a_grid_is_kept_and_read_back_only_for_what_it_was_asked_of(tmp_path):
    from box_opening.tasks import reach_probe

    z = _zone(_band)
    z.cells[3, 4] = UNASKED
    back = Zone.from_json(z.to_json())
    assert (back.cells == z.cells).all() and (back.x0, back.y0, back.step, back.lid_z, back.key) == (
        0.0,
        -0.5,
        0.05,
        0.30,
        "k",
    )
    assert reach_probe.lid_bucket(0.3149) == 0.32 and reach_probe.lid_bucket(0.3051) == 0.30
    path = reach_probe.save_zone(z, tmp_path)
    assert path == reach_probe.zone_path("k", 0.309, tmp_path) and path.name == "zone_k_z+0300.json"
    assert (reach_probe.load_zone("k", 0.305, None, tmp_path).cells == z.cells).all()


def test_a_save_writes_beside_the_grid_under_the_savers_own_name_and_moves_it_in_whole(tmp_path, monkeypatch):
    """Never a half-written grid under the grid's name, and two probes on one grid (the engine's and an
    offline fill) never the same temporary file: the second to move it in would find it gone."""
    import os

    from box_opening.tasks import reach_probe

    z = _zone(_band)
    path = reach_probe.save_zone(z, tmp_path)
    moved, move = [], os.replace
    monkeypatch.setattr(os, "replace", lambda src, dst: moved.append((str(src), str(dst))))
    monkeypatch.setattr(os, "getpid", lambda: 111)
    z.set(0, 0, True)
    reach_probe.save_zone(z, tmp_path)
    # the move not made (the probe was killed here): the grid's own file is the whole earlier one
    assert Zone.from_json(path.read_text()).state((0.0, -0.5)) == FAILS
    monkeypatch.setattr(os, "getpid", lambda: 222)
    reach_probe.save_zone(z, tmp_path)
    (first, to), (second, _to) = moved
    assert to == str(path) and first != second and {os.path.dirname(first), os.path.dirname(second)} == {str(tmp_path)}
    move(first, path)
    move(second, path)  # each its own file: neither finds it gone
    assert Zone.from_json(path.read_text()).state((0.0, -0.5)) == WORKS
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def _kept_text(**changed):
    """The text of a kept grid (the band's, lid 0.30, key "k") with these fields of it changed."""
    import json

    d = _zone(_band).to_dict()
    d.update(changed)
    return json.dumps(d)


NOT_A_GRID = {
    "empty": "",
    "cut short": _kept_text()[:200],
    "a list": "[]",
    "null": "null",
    "a number": "7",
    "no x0": _kept_text(x0=None),
    "x0 not a number": _kept_text(x0="NaN"),
    "no step": _kept_text(step=0.0),
    "no height": _kept_text(lid_z=None),
    "another height's grid": _kept_text(lid_z=0.36),
    "no rows": _kept_text(rows=[]),
    "empty rows": _kept_text(rows=["", ""]),
    "ragged rows": _kept_text(rows=["##", "#"]),
    "a cell that is no state": _kept_text(rows=["#x"]),
    "rows a number": _kept_text(rows=7),
    "another key's grid": _kept_text(key="another robot"),
    "not text": "\x00\x00\x00",
}


@pytest.mark.parametrize("text", list(NOT_A_GRID.values()), ids=list(NOT_A_GRID))
def test_a_kept_file_that_is_not_a_whole_grid_is_no_grid_kept(text, tmp_path, monkeypatch):
    from box_opening.tasks import reach_probe

    monkeypatch.setattr(reach_probe, "reach_span", lambda model, lid_z: ((0.0, 0.95), (-0.5, 0.5)))
    path = reach_probe.zone_path("k", 0.30, tmp_path)
    path.write_text(text)
    zone = reach_probe.load_zone("k", 0.30, None, tmp_path)
    assert zone.cells.shape == (20, 21) and (zone.cells == UNASKED).all() and zone.lid_z == 0.30 and zone.key == "k"
    assert reach_probe.save_zone(zone, tmp_path) == path  # and it is written over
    assert (reach_probe.load_zone("k", 0.30, None, tmp_path).cells == zone.cells).all()
    path.write_text(_kept_text())
    assert reach_probe.load_zone("k", 0.30, None, tmp_path).counts()["works"] > 0  # a whole one is read


def test_the_picture_reads_as_the_user_sees_the_table():
    z = Zone.blank((0.0, 0.1), (-0.05, 0.05))
    z.set(2, 2, True)  # far, to the arm's left
    z.set(0, 0, False)  # near, to its right
    assert z.picture().splitlines() == ["x 0.10  #??", "x 0.05  ???", "x 0.00  ??."]


# -- the proof itself, on flatbot's arm ------------------------------------------------------
@pytest.fixture
def flatbot():
    from box_opening.runtime import approach
    from box_opening.tasks import reach_probe

    ctx, cfg = reach_probe.setup(flatbot_urdf(), "arm_mount", "", CFG)
    yield ctx, cfg
    approach.set_chain_root("base_link")
    approach.set_body(None)


def test_a_press_plans_in_front_of_the_arm_and_not_past_its_reach(flatbot):
    from box_opening.tasks import reach_probe

    ctx, cfg = flatbot
    ok, why, tilt = reach_probe.press_plans(ctx, cfg, (0.65, -0.10, 0.315), table_z=0.216)
    assert ok and why == "" and tilt in (15, 20, 25, 30, 35)
    ok, why, tilt = reach_probe.press_plans(ctx, cfg, (1.20, 0.0, 0.315), table_z=0.216)
    assert not ok and tilt is None and why == "past everything the arm reaches"
    assert ctx.client.live != [], "the stand-in holds the joints the proof left it at"


def test_the_proof_is_made_from_the_start_it_is_given_a_wrist_turned_to_look(flatbot, monkeypatch):
    """press_plans' q_from on the arm's own kinematics: the proof before the look is asked from those joints,
    and a press in front of the arm still plans from a wrist turned as the line-up's view turns it."""
    from box_opening.constants import HOME
    from box_opening.tasks import reach_press, reach_probe

    ctx, cfg = flatbot
    view = [h + d for h, d in zip(HOME, (0.0, 0.0, 0.0, 0.0, 0.5, -0.5, 0.4), strict=True)]
    asked_from, plan_reach = [], reach_press.plan_reach

    def spy(ctx_, cfg_, button, q_from, **kw):
        asked_from.append(list(q_from))
        return plan_reach(ctx_, cfg_, button, q_from, **kw)

    monkeypatch.setattr(reach_press, "plan_reach", spy)
    ok, why, tilt = reach_probe.press_plans(ctx, cfg, (0.65, -0.10, 0.315), table_z=0.216, q_from=view)
    assert asked_from[0] == pytest.approx(view)
    assert ok and why == "" and tilt in (15, 20, 25, 30, 35)
    asked_from.clear()
    assert reach_probe.press_plans(ctx, cfg, (0.65, -0.10, 0.315), table_z=0.216)[0]
    assert asked_from[0] == pytest.approx(HOME)  # none given: HOME, as the cells are asked


def test_the_grids_ground_is_the_arms_own_reach(flatbot):
    from box_opening.tasks import reach_probe

    ctx, _cfg = flatbot
    (x0, x1), (y0, y1) = reach_probe.reach_span(ctx.model, 0.315)
    assert x0 == 0.0 and y0 == -y1 and 0.85 <= x1 <= 1.10  # the Gen3 with its gripper: about 0.9 m
    zone = reach_probe.load_zone("none kept", 0.315, ctx.model, where="/nonexistent")
    assert zone.lid_z == 0.32 and (zone.cells == UNASKED).all()


def test_the_stand_in_refuses_what_the_planner_refuses(flatbot):
    from box_opening.constants import HOME
    from box_opening.tasks import reach_probe

    ctx, _cfg = flatbot
    client = ctx.client
    assert isinstance(client, reach_probe.DryClient)
    plan = client.plan_to_joints(list(HOME), list(HOME))
    assert plan.success and list(plan.trajectory.points[-1].positions) == pytest.approx(HOME)
    lo, hi = next((lo, hi) for lo, hi in client.chain.limits if lo is not None and hi is not None)
    j = next(k for k, (lo_, hi_) in enumerate(client.chain.limits) if lo_ is not None and hi_ is not None)
    goal = list(HOME)
    goal[j] = hi - 0.02  # inside the planner's limit margin
    refused = client.plan_to_joints(goal, list(HOME))
    assert not refused.success and refused.message == "GOAL_IN_LIMIT_MARGIN (joint_%d)" % (j + 1)
    assert lo < hi


def test_the_key_follows_the_container_and_the_chair_model(flatbot, tmp_path):
    from box_opening.tasks import reach_probe

    key = reach_probe.zone_key(flatbot_urdf(), CFG)
    assert key == reach_probe.zone_key(flatbot_urdf(), CFG) and len(key) == 16
    other = tmp_path / "other.yaml"
    other.write_text(open(CFG).read() + "\n# another container\n")
    assert reach_probe.zone_key(flatbot_urdf(), str(other)) != key


# the modules a proof runs that the key did not follow until 2026-11-16 (the review's zone-key-holes)
PROOF_SOURCE = [
    "constants",
    "runtime.branches",
    "runtime.client",
    "runtime.retime",
    "runtime.guards",
    "primitives.core",
    "primitives.look",
    "models.container",
]


@pytest.mark.parametrize("name", PROOF_SOURCE + ["tasks.reach_press", "tasks.depth_press", "runtime.kinematics"])
def test_the_key_follows_the_source_of_what_the_proof_runs(name, flatbot, tmp_path, monkeypatch):
    """A grid asked under other code is not read back as current: a cell that fails is never asked again."""
    import importlib

    from box_opening.tasks import reach_probe

    key = reach_probe.zone_key(flatbot_urdf(), CFG)
    module = importlib.import_module("box_opening." + name)
    changed = tmp_path / "changed.py"
    changed.write_text(open(module.__file__).read() + "\n# a change\n")
    monkeypatch.setattr(module, "__file__", str(changed))
    assert reach_probe.zone_key(flatbot_urdf(), CFG) != key
    monkeypatch.undo()
    assert reach_probe.zone_key(flatbot_urdf(), CFG) == key


def test_the_key_follows_the_arms_spheres_and_the_hands_in_the_chair_model(flatbot, tmp_path, monkeypatch):
    """The chair model holds the arm's spheres out of its boxes: the planner's robot file's, and the hand's
    from box_opening's own robot-body config. The key followed the boxes alone."""
    from box_opening.runtime import approach, body
    from box_opening.tasks import reach_probe

    hand = tmp_path / "robot_body.yaml"
    hand.write_text("hand_spheres: {}\n")

    def model(radius=0.05, centre_z=0.10, link="forearm_link", hand_source=str(hand)):
        return body.BodyModel(
            source="planner.yaml",
            base_link="arm_mount",
            names=["chair/seat"],
            lo=np.array([[-0.3, 0.2, -0.5]]),
            hi=np.array([[0.3, 0.6, -0.1]]),
            spheres={link: (np.array([[0.0, 0.0, centre_z]]), np.array([radius]))},
            padding_m=0.01,
            buffer_m=0.004,
            hand=[],
            hand_source=hand_source,
        )

    def key_with(**changed):
        monkeypatch.setattr(approach, "body_model", lambda: model(**changed))
        return reach_probe.zone_key(flatbot_urdf(), CFG)

    key = key_with()
    assert key == key_with()
    keys = {key, key_with(radius=0.06), key_with(centre_z=0.12), key_with(link="bracelet_link")}
    keys.add(key_with(hand_source=None))  # the planner's spheres alone
    hand.write_text("hand_spheres: {}\nnear_user: {}\n")
    keys.add(key_with())
    assert len(keys) == 6


def test_the_key_of_a_grid_asked_on_a_table_is_not_the_served_grids(flatbot):
    """The served grid's cells are asked with the box on its own foot. An offline fill with --table-z asks
    them on that table: another grid, under another key."""
    from box_opening.tasks import reach_probe

    key = reach_probe.zone_key(flatbot_urdf(), CFG)
    assert reach_probe.zone_key(flatbot_urdf(), CFG, table_z=None) == key
    on_table = reach_probe.zone_key(flatbot_urdf(), CFG, table_z=0.216)
    assert on_table != key and len(on_table) == 16
    assert on_table == reach_probe.zone_key(flatbot_urdf(), CFG, table_z=0.216)
    assert on_table != reach_probe.zone_key(flatbot_urdf(), CFG, table_z=0.25)


def test_a_fill_asked_on_a_table_is_not_the_grid_the_line_up_reads(main_on_stand_ins, tmp_path, monkeypatch):
    """... so a line-up never reads a grid asked on some table back as its own."""
    from box_opening.tasks import reach_probe

    run, served = main_on_stand_ins
    monkeypatch.setattr(
        reach_probe, "zone_key", lambda _urdf, _container, table_z=None: "k" if table_z is None else "k%g" % table_z
    )
    assert run("--lid-z", "0.315", "--table-z", "0.216") == 0
    assert [p.name for p in tmp_path.iterdir()] == ["zone_k0.216_z+0320.json"]
    assert run("--serve") == 0 and served[-1].key == "k"
    assert run("--lid-z", "0.315") == 0  # no table named: the grid the line-up reads
    assert sorted(p.name for p in tmp_path.iterdir()) == ["zone_k0.216_z+0320.json", "zone_k_z+0320.json"]
