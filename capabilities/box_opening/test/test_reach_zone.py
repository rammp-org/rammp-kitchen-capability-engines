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
