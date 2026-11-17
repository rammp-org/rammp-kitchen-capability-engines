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


# -- for the engine's line-up: the probe's serve mode ------------------------------------------
@pytest.fixture
def served(monkeypatch, tmp_path):
    """reach_probe.Server on a stand-in proof (works within 18 cm of (0.65, 0)) and a small blank grid; its
    clock ticks 0.3 s a proof. Returns (server, what it said, the places it proved)."""
    from box_opening.tasks import reach_probe

    said, proved, now = [], [], [0.0]

    def proof(_ctx, _cfg, top, table_z=None):
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
