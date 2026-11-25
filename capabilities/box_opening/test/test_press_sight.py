"""The sight before the press (tasks/reach_press.fly_press_way_in) on flatbot's own kinematics: the free air to
the hand turned down with the wrist lens over the aimed button, the button sighted in frames rendered from the
lens where the arm stands, and the level line and the stroke from there. Pure kinematics and pictures: no
planner, no arm. Not flown."""

import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest
from test_button_aim import BUTTON_R, OAK_HW, OAK_K, SEAM_W, render
from test_button_sight import _lid_frame
from test_reach_press import SCENE_1519, _at, _on_flatbot

from box_opening.constants import FINGERS_PAST_MODEL_M
from box_opening.tasks import depth_press as dp
from box_opening.tasks import reach_press as rp

pytestmark = pytest.mark.usefixtures("home_20261021")

LENS = (0.0, 0.065, -0.105)  # flatbot's OAK in the hand (config/camera_oak_wrist.yaml wrist_lens_in_hand_m)
# A button 0.62 m out, 20 deg right: pressed straight down, and the level line from the sight flies to 25 mm
# either way of it. The sight pose stands the hand 65 mm further out than the press (the lens over the
# button, not the fingers): out past ~0.70 m on flatbot the hand there leans the button out of the picture
# or its level line meets a joint limit, and there is no sight
# (test_where_the_arm_reaches_no_sight_pose_the_way_in_is_todays)
NEAR = _at((0.62, -20.0))


class _Runner:
    """Flies what it is handed in an instant, the arm ending at its last point (or `stop`'s leg's start:
    that leg did not arrive; with `off`, its first execution ends that far off its last point in each joint,
    as an arrival in the joint impedance does); records each execution, each wait for a still arm and each
    note."""

    def __init__(self, ctx, stop=None, off=None, restless=()):
        self.ctx, self.stop, self.off, self.restless = ctx, stop, off, restless
        self.runs, self.stills, self.notes = [], [], []

    def run(self, legs, execute=True, **kw):
        self.runs.append(list(legs))
        out = []
        for lg in legs:
            pts = lg.traj.points
            if lg.name == self.stop:
                self.ctx.client.live = [float(v) for v in pts[len(pts) // 2].positions]
                out.append(SimpleNamespace(name=lg.name, ok=False, flown_traj=lg.traj, progress=0.5))
                return out
            self.ctx.client.live = [float(v) for v in pts[-1].positions]
            out.append(SimpleNamespace(name=lg.name, ok=True, flown_traj=lg.traj, progress=1.0))
        if self.off is not None and len(self.runs) == 1:
            self.ctx.client.live = [q + d for q, d in zip(self.ctx.client.live, self.off, strict=True)]
        return out

    def wait_still(self, why):
        """(settle_s, timed_out) as Runner.wait_still says it: timed out before what `restless` names."""
        self.stills.append(why)
        return (1.2, True) if why in self.restless else (0.0, False)

    def note(self, kind, **kw):
        self.notes.append((kind, kw))


class _Eyes:
    """The wrist camera: each frame rendered from where the lens stands at the arm's live joints (its
    mount turned `mount_deg` about the hand's x), the box's button at `true_xy` with its top at `top_z`;
    `ringed`: the lid as it is, its 60 mm moulded ring and rim round the seam (test_button_sight._lid_frame),
    not test_button_aim's render, which has no ring; `frames` overrides what each snapshot holds (a list of
    callables, one per frame, last repeated), and `stamps(n)` the second the n-th is stamped at (default n)."""

    def __init__(self, ctx, chain, true_xy, top_z, height, mount_deg=0.0, frames=None, stamps=None, ringed=False):
        self.ctx, self.chain, self.true_xy, self.top_z, self.height = ctx, chain, true_xy, top_z, height
        self.ringed = ringed
        a = math.radians(mount_deg)
        self.r_hc = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]])
        self.k, self.dist, self.n, self.frames, self.cache, self.stamps = OAK_K, None, 0, frames, {}, stamps
        self.color = np.zeros((*OAK_HW, 3), np.uint8)
        self.grab = self

    def lens(self):
        R, t = (np.asarray(v, float) for v in self.chain.fk(self.ctx.client.joints()))
        return t + R @ np.asarray(LENS), R @ self.r_hc

    def render(self):
        cam, rot = self.lens()
        key = tuple(np.round(cam, 6))
        if key not in self.cache and self.ringed:
            off = (self.true_xy[0] - cam[0], self.true_xy[1] - cam[1])
            self.cache[key] = _lid_frame(self.ctx.model, off, cam[2] - self.top_z, ss=2, rot=rot, k=self.k)
        if key not in self.cache:
            box = [(tuple(self.true_xy), self.top_z, 0.0)]
            self.cache[key] = render(cam, rot, box, self.top_z - self.height)[0]
        return self.cache[key]

    def snapshot(self):
        self.n += 1
        make = None if self.frames is None else self.frames[min(self.n, len(self.frames)) - 1]
        color = self.render() if make is None else make(self)
        sec = self.n if self.stamps is None else self.stamps(self.n)
        stamp = SimpleNamespace(sec=sec, nanosec=0)
        return SimpleNamespace(color=color, color_stamp=stamp, depth=None, depth_stamp=None, k=self.k, dist=None)


def _node(now_ns=0):
    return SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=now_ns)))


def _way(ctx, monkeypatch, button):
    """flatbot, the pink canister sized as the render's seam (test_button_sight's pink), the press's way in
    onto `button` solved as _reach_press solves it: (chain, cfg, way, legs)."""
    chain, cfg, home = _on_flatbot(ctx, monkeypatch)
    plan, why = rp.plan_reach(ctx, cfg, button, home, depth=True)
    assert why is None, why
    way, why, kind = rp.solve_way_in(ctx, cfg, button, plan.tilt_deg, plan.look_leg.goal_joints)
    assert way is not None, (why, kind)
    legs = rp.way_in_legs(ctx, cfg, way)
    ctx.client.live = [float(v) for v in legs[0].traj.points[0].positions]  # at the look pose
    ctx.model = dataclasses.replace(ctx.model, button_diameter_m=2.0 * BUTTON_R + SEAM_W)
    return chain, cfg, way, legs


def _eyes(ctx, monkeypatch, chain, true_xy, top_z, **kw):
    """The run's wrist camera (_Eyes) wired as fly_press_way_in reads it, the lens in the hand as TF would
    give it, nothing spun, nothing saved."""
    from box_opening.tasks import press_demo

    eyes = _Eyes(ctx, chain, true_xy, top_z, ctx.model.dims[2], **kw)
    monkeypatch.setattr(dp, "lens_in_hand", lambda ctx_, grab, still_since_ns=None: (list(LENS), eyes.r_hc))
    monkeypatch.setattr(press_demo, "_spin_detect", lambda node: None)
    saved = []
    monkeypatch.setattr(press_demo, "save_aim_frame", lambda grab, s, tag="aim", frame=None: saved.append(tag))
    monkeypatch.setattr(rp, "SIGHT_S", 30.0)  # frames render in tenths of a second here, not 15 a second
    eyes.saved = saved
    return (_node(), eyes, None, None)


def _point_xy(chain, q, point_tool):
    R, t = (np.asarray(v, float) for v in chain.fk(q))
    return (t + R @ np.asarray(point_tool, float))[:2]


def _fly(ctx, cfg, way, legs, button, eyes, runner):
    return rp.fly_press_way_in(ctx, cfg, runner, way, legs, button[2], eyes)


def _today(ctx, monkeypatch, button):
    """The legs and their stroke today's way in sends onto `button` (fly_way_in, the switch off)."""
    _chain, cfg, way, legs = _way(ctx, monkeypatch, button)
    runner = _Runner(ctx)
    monkeypatch.setattr(rp, "SIGHT_BEFORE_PRESS", False)
    rp.fly_press_way_in(ctx, cfg, runner, way, legs, button[2], None)
    monkeypatch.setattr(rp, "SIGHT_BEFORE_PRESS", True)
    return runner.runs, way.stroke


def _start_xy(chain, stroke, point_tool=dp.DOWN_CONTACT_TOOL):
    """Where `point_tool` stands in xy at the stroke's start."""
    return _point_xy(chain, stroke.legs[0][1][0], point_tool)


# ---- followed ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("off", [(0.0, 0.0), (0.012, 0.0), (0.0, -0.012), (-0.018, 0.018), (0.025, 0.0)])
def test_the_real_pads_come_down_over_the_button_the_sight_saw(ctx, monkeypatch, off):
    """Cases 1 and 8: a button pressed with the hand exactly down, truly `off` from the aim. Two
    executions, the free air to the sight and then the level line and the stroke; at the stroke's start
    the real pad ends (FINGERS_PAST_MODEL_M past the heights point) stand within 2 mm of the button's true
    centre. The button the run goes by after it is the sighted one (the box is not moved: gate G2); the
    stroke the hold, the rise and the way home read is the one that flew."""
    button = NEAR
    chain, cfg, way, legs = _way(ctx, monkeypatch, button)
    true = (button[0] + off[0], button[1] + off[1])
    eyes = _eyes(ctx, monkeypatch, chain, true, button[2])
    cpose = ctx.cpose
    runner = _Runner(ctx)
    res, contact, names, over_box = _fly(ctx, cfg, way, legs, button, eyes, runner)
    assert len(runner.runs) == 2 and over_box and all(r.ok for r in res)
    assert names == ["sight:tail", "depth:approach", "depth:press"] == [lg.name for lg in runner.runs[1]]
    assert contact is not None
    assert "the sight's plan" in runner.stills and "the sight" in runner.stills
    sight = dict(runner.notes)["sight"]
    assert sight["planned"] and sight["followed"], sight
    pads = _start_xy(chain, way.stroke, rp.SIGHT_PADS_TOOL)
    assert math.hypot(pads[0] - true[0], pads[1] - true[1]) < 0.002, (pads, true)
    assert runner.runs[1][-1].traj.points[-1].positions == pytest.approx(way.stroke.legs[-1][1][-1])
    assert way.sighted == pytest.approx(true, abs=0.002)
    assert ctx.cpose == cpose and eyes[1].saved == ["sight"]  # the box is not moved: the lid's room is keyed to it
    assert sight["part"] == 1.0 and sight["plan_ms"] >= 0.0 and sight["look_ms"] >= 0.0
    assert sight["dx_mm"] == pytest.approx(1000 * off[0], abs=2.0) and sight["how"] in ("seam+rim", "seam")
    assert set(sight) >= {"below_mm", "expect_below_mm", "r_px", "lens_in_hand_mm", "hits", "frames"}


def test_a_hand_short_of_straight_down_with_its_mount_off_still_comes_down_over_it(ctx, monkeypatch):
    """Case 2: the hand turned 5 deg short of straight down at the sight (no walk in: the turn, 5 deg short,
    as where the walk does not fly), the camera's mount 1.5 deg off the hand's axes as TF gives it, the
    button 12 mm off the aim: the real pad ends within 2 mm of it. They lean out 20 x sin(5 deg) = 1.7 mm
    past the heights point: a press by the heights point would put them there."""
    button = NEAR
    chain, cfg, way, legs = _way(ctx, monkeypatch, button)
    true = (button[0] + 0.012, button[1] - 0.004)
    eyes = _eyes(ctx, monkeypatch, chain, true, button[2], mount_deg=1.5)
    monkeypatch.setattr(dp, "walk_down_path", lambda *a, **k: (None, None, None, "no walk in"))
    monkeypatch.setattr(dp, "COMPLIANT_TURN_SHORT_DEG", (5.0,))
    runner = _Runner(ctx)
    _fly(ctx, cfg, way, legs, button, eyes, runner)
    sight = dict(runner.notes)["sight"]
    assert sight["followed"], sight
    assert dp._lean_deg(way.stroke.quat) == pytest.approx(5.0, abs=0.2)
    pads = _start_xy(chain, way.stroke, rp.SIGHT_PADS_TOOL)
    assert math.hypot(pads[0] - true[0], pads[1] - true[1]) < 0.002, (pads, true)
    # the pads' point, not the heights point, is what was put over it: that one stands 1.7 mm further off
    heights = _start_xy(chain, way.stroke)
    off_pads = math.hypot(pads[0] - true[0], pads[1] - true[1])
    assert math.hypot(heights[0] - true[0], heights[1] - true[1]) > off_pads + 0.001


@pytest.mark.parametrize("short", [None, 5.0])
def test_over_a_lid_with_its_ring_the_real_pads_come_down_over_the_button(ctx, monkeypatch, short):
    """The followed sight over the lid as it is (its 60 mm moulded ring and rim round the seam), the hand
    straight down and 5 deg short at the sight, the button 19 mm off the aim: the real pad ends within 2 mm
    of it. The ring and the seam are where Hough's radius took a circle across both arcs (round 2)."""
    button = NEAR
    chain, cfg, way, legs = _way(ctx, monkeypatch, button)
    true = (button[0] + 0.018, button[1] - 0.006)
    eyes = _eyes(ctx, monkeypatch, chain, true, button[2], ringed=True)
    if short is not None:
        monkeypatch.setattr(dp, "walk_down_path", lambda *a, **k: (None, None, None, "no walk in"))
        monkeypatch.setattr(dp, "COMPLIANT_TURN_SHORT_DEG", (short,))
    runner = _Runner(ctx)
    _fly(ctx, cfg, way, legs, button, eyes, runner)
    sight = dict(runner.notes)["sight"]
    assert sight["followed"], sight
    assert dp._lean_deg(way.stroke.quat) == pytest.approx(short or 0.0, abs=0.2)
    pads = _start_xy(chain, way.stroke, rp.SIGHT_PADS_TOOL)
    assert math.hypot(pads[0] - true[0], pads[1] - true[1]) < 0.002, (pads, true)


def test_frames_shot_before_the_arm_stood_still_are_not_judged(ctx, monkeypatch):
    """The sight reads only frames stamped after the arm stood still (the node's clock then): two frames
    stamped before it, the button in them 20 mm off where it is (the arm still moving), then frames of
    where it stands. The button followed is the one where it stands."""
    button = NEAR
    chain, cfg, way, legs = _way(ctx, monkeypatch, button)
    true = (button[0] + 0.012, button[1])
    eyes = _eyes(ctx, monkeypatch, chain, true, button[2], frames=[_shifted(0.020), _shifted(0.020), None])
    eyes = (_node(now_ns=2_500_000_000), *eyes[1:])  # frames 1 and 2 (1 s, 2 s) before it, 3 on after
    runner = _Runner(ctx)
    _fly(ctx, cfg, way, legs, button, eyes, runner)
    sight = dict(runner.notes)["sight"]
    assert sight["followed"] and sight["dx_mm"] == pytest.approx(12.0, abs=1.5), sight
    assert sight["frames"] == sight["hits"] == 2 and eyes[1].n >= 4


# ---- not planned: today's legs, exactly ----------------------------------------------------------------------


def _same(runs, today):
    """Whether `runs` sent exactly today's executions: the same legs, point for point."""

    def flat(rs):
        return [(lg.name, [tuple(p.positions) for p in lg.traj.points], lg.speed) for r in rs for lg in r]

    return len(runs) == len(today) and flat(runs) == flat(today)


@pytest.mark.parametrize("why", ["no eyes", "no grab", "air in position", "lens not in TF", "the plan raises"])
def test_where_no_sight_is_planned_the_way_in_is_todays_exactly(ctx, monkeypatch, why):
    """Case 5: no wrist camera on the run, no frame source, the free air in position, no camera pose for the
    lens, a plan that raises: the run says why ("not planned ... the way in as before") and the legs sent
    are exactly today's, one execution."""
    button = SCENE_1519
    today, _stroke = _today(ctx, monkeypatch, button)
    chain, cfg, way, legs = _way(ctx, monkeypatch, button)
    eyes = _eyes(ctx, monkeypatch, chain, button[:2], button[2])
    if why == "no eyes":
        eyes = None
    elif why == "no grab":
        eyes = (eyes[0], SimpleNamespace(), None, None)
    elif why == "air in position":
        monkeypatch.setenv(rp.AIR_CONTROL_ENV, "position")
        today, _stroke = _today(ctx, monkeypatch, button)
        _chain, cfg, way, legs = _way(ctx, monkeypatch, button)
    elif why == "lens not in TF":
        monkeypatch.setattr(dp, "lens_in_hand", lambda *a, **k: None)
    else:

        def boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(rp, "walk_or_turn", boom)
    runner = _Runner(ctx)
    _fly(ctx, cfg, way, legs, button, eyes, runner)
    assert _same(runner.runs, today)
    note = dict(runner.notes)["sight"]
    assert note["planned"] is False and note["why"]
    # the plan's time where it was asked (the lens, the plan that raises), else none
    assert (note["plan_ms"] is None) == (why in ("no eyes", "no grab", "air in position")), note


@pytest.mark.parametrize(
    "r_az, why",
    [
        ((0.75, -20.0), "not in view"),  # pressed 8 deg short: the sight pose 20 deg short
        ((0.85, -20.0), "not in view"),  # 22 deg
        ((0.75, -10.0), "joint limit"),  # the level line from the sight
    ],
)
def test_where_the_arm_reaches_no_sight_pose_the_way_in_is_todays(ctx, monkeypatch, r_az, why):
    """Case 2's other half and case 5: buttons 0.75-0.85 m out, pressed leaning. The sight pose stands the
    hand 65 mm further out: leaning as far as it must there, the button grown by the 30 mm the sight may find
    it off does not fit in the picture, or the level line from it meets a joint limit. No sight is planned,
    the run says why, and today's legs fly."""
    button = _at(r_az)
    today, _stroke = _today(ctx, monkeypatch, button)
    chain, cfg, way, legs = _way(ctx, monkeypatch, button)
    eyes = _eyes(ctx, monkeypatch, chain, button[:2], button[2])
    runner = _Runner(ctx)
    _fly(ctx, cfg, way, legs, button, eyes, runner)
    note = dict(runner.notes)["sight"]
    assert note["planned"] is False and why in note["why"], note
    assert note["plan_ms"] >= 0.0  # its solves paid all the same
    assert _same(runner.runs, today)


# the flatbot OAK's colour intrinsics as its frames record them (640x400)
OAK_FLATBOT_K = np.array([[398.4, 0.0, 320.0], [0.0, 398.4, 200.0], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize("axis, most", [((1.0, 0.0, 0.0), 22.0), ((0.0, 1.0, 0.0), 10.0)])
def test_the_in_view_check_is_the_picture_not_a_lean(axis, most):
    """_sight_in_view, the lens 0.19 m over the pink box's top (SIGHT_ABOVE_M and the lens 0.11 m behind the
    heights point), the camera leaning about the lens's vertical: the button grown by SIGHT_MAX_OFF_M must
    stand EDGE_PX inside the frame. Leaning along the frame's long side the sight is planned to 22 deg,
    along its short side to 10 deg: the press leans along the short side."""
    import cv2
    from test_button_aim import PINK, ROT_DOWN

    from box_opening.models.container import ContainerModel

    pink = ContainerModel.load(PINK)
    aim, top = (0.60, -0.20), 0.30
    lens = (aim[0], aim[1], top + 0.19)

    def leaning(deg):
        return cv2.Rodrigues(np.asarray(axis) * math.radians(deg))[0] @ ROT_DOWN

    args = (OAK_FLATBOT_K, None, OAK_HW, aim, top, pink)
    assert rp._sight_in_view(ROT_DOWN, lens, *args) is None
    assert rp._sight_in_view(leaning(most), lens, *args) is None
    assert "would show at" in rp._sight_in_view(leaning(most + 1.0), lens, *args)


# ---- planned, not followed: the press where today's lands ----------------------------------------------------


def _blank(eyes):
    return np.zeros((*OAK_HW, 3), np.uint8)


def _shifted(dx):
    """A frame of the button `dx` (m, base x) from where it truly is: frames that never agree."""

    def make(eyes):
        cam, rot = eyes.lens()
        box = [((eyes.true_xy[0] + dx, eyes.true_xy[1]), eyes.top_z, 0.0)]
        return render(cam, rot, box, eyes.top_z - eyes.height)[0]

    return make


@pytest.mark.parametrize("case", ["nothing", "never agree", "35 mm off", "raises", "its line refused"])
def test_a_sight_not_followed_presses_where_todays_would(ctx, monkeypatch, case):
    """Case 6, on 15:19's button: the sight finds nothing, its frames never agree, the button is seen 35 mm
    from the aim, the sight raises, or the level line over what it saw is not one the arm flies (12 mm
    further out: a joint limit down the line): the press goes by the aim,
    the heights point at the stroke's start where today's would stand (within 0.5 mm), the run goes on
    (two executions, the stroke flown), the run says why, and the button and the box are the aimed ones."""
    button = SCENE_1519
    _today_runs, today_stroke = _today(ctx, monkeypatch, button)
    chain, cfg, way, legs = _way(ctx, monkeypatch, button)
    true = (button[0] + {"35 mm off": 0.035, "its line refused": 0.012}.get(case, 0.0), button[1])
    frames = None
    if case == "nothing":
        frames = [_blank]
    elif case == "never agree":
        frames = [_shifted(0.0), _shifted(0.006), _shifted(0.012), _shifted(0.018), _blank]
    eyes = _eyes(ctx, monkeypatch, chain, true, button[2], frames=frames)
    if frames is not None:  # then blank frames until the time is up
        monkeypatch.setattr(rp, "SIGHT_S", 0.3 if case == "nothing" else 3.0)
    if case == "raises":
        from box_opening.perception import button_sight

        def boom(*a, **k):
            raise ValueError("a frame of the wrong shape")

        monkeypatch.setattr(button_sight, "sight_button", boom)
    cpose = ctx.cpose
    runner = _Runner(ctx)
    res, contact, names, over_box = _fly(ctx, cfg, way, legs, button, eyes, runner)
    note = dict(runner.notes)["sight"]
    assert note["planned"] and not note["followed"] and note["why"], note
    assert len(runner.runs) == 2 and over_box and all(r.ok for r in res) and contact is not None
    assert names == ["sight:tail", "depth:approach", "depth:press"]
    a, b = _start_xy(chain, way.stroke), _start_xy(chain, today_stroke)
    assert math.hypot(a[0] - b[0], a[1] - b[1]) < 0.0005, (a, b)
    assert way.sighted is None and ctx.cpose == cpose
    # how long the arm stood for the sight, whatever came of it: where nothing was seen, its whole window
    assert note["look_ms"] >= (1000 * rp.SIGHT_S if case == "nothing" else 0.0), note
    if case == "35 mm off":
        assert "35 mm from the aim" in note["why"]
    if case == "never agree":
        assert "none agreeing" in note["why"], note
    if case == "raises":
        assert "ValueError" in note["why"]
    if case == "its line refused":
        assert "joint limit" in note["why"] and note["dx_mm"] == pytest.approx(12.0, abs=2.0)


def test_a_sight_not_followed_with_the_hand_leaning_presses_by_the_heights_point_over_the_aim(ctx, monkeypatch):
    """Case 6 with the hand 5 deg short of straight down at the sight (the turn, as where no walk in flies),
    the arm arrived there a few mm off the plan as the joint impedance leaves it: the button seen 35 mm off
    is not followed, the level line is solved again from where the arm stands, and the point the heights go
    by, not the real pads (1.7 mm further out), stands over the aimed xy at the stroke's start: where
    today's press comes down."""
    button = NEAR
    chain, cfg, way, legs = _way(ctx, monkeypatch, button)
    eyes = _eyes(ctx, monkeypatch, chain, (button[0] + 0.035, button[1]), button[2])
    monkeypatch.setattr(dp, "walk_down_path", lambda *a, **k: (None, None, None, "no walk in"))
    monkeypatch.setattr(dp, "COMPLIANT_TURN_SHORT_DEG", (5.0,))
    runner = _Runner(ctx, off=[0.004, -0.003, 0.0, 0.003, 0.0, 0.0, 0.0])
    _fly(ctx, cfg, way, legs, button, eyes, runner)
    sight = dict(runner.notes)["sight"]
    assert sight["planned"] and not sight["followed"], sight
    assert dp._lean_deg(way.stroke.quat) == pytest.approx(5.0, abs=0.2)
    heights = _start_xy(chain, way.stroke)
    assert math.hypot(heights[0] - button[0], heights[1] - button[1]) < 0.0005, heights
    pads = _start_xy(chain, way.stroke, rp.SIGHT_PADS_TOOL)
    assert math.hypot(pads[0] - button[0], pads[1] - button[1]) > 0.001


# ---- the first execution stops part-way ----------------------------------------------------------------------


def test_a_free_air_that_did_not_arrive_has_no_sight_and_no_stroke(ctx, monkeypatch):
    """Case 7: the free air to the sight stops part-way: no sight, no second execution, not over the box;
    the callers handle it as the air split's free air that did not arrive."""
    button = SCENE_1519
    chain, cfg, way, legs = _way(ctx, monkeypatch, button)
    eyes = _eyes(ctx, monkeypatch, chain, button[:2], button[2])
    plan = rp._sight_way(ctx, cfg, _Runner(ctx), way, legs, button[2], eyes)[0]
    runner = _Runner(ctx, stop=plan.legs[-1].name)
    res, contact, names, over_box = _fly(ctx, cfg, way, legs, button, eyes, runner)
    assert len(runner.runs) == 1 and not res[-1].ok and contact is None and over_box is False
    assert names == [lg.name for lg in plan.legs] and eyes[1].n == 0
    assert "sight" not in dict(runner.notes)


# ---- the switch ------------------------------------------------------------------------------------------------


def test_with_the_switch_off_the_way_in_is_one_execution_and_no_sight(ctx, monkeypatch):
    """Case 9: SIGHT_BEFORE_PRESS off, the wrist camera there all the same: fly_way_in's one execution,
    nothing waited for, nothing noted, no frame looked at."""
    button = SCENE_1519
    chain, cfg, way, legs = _way(ctx, monkeypatch, button)
    eyes = _eyes(ctx, monkeypatch, chain, button[:2], button[2])
    monkeypatch.setattr(rp, "SIGHT_BEFORE_PRESS", False)
    runner = _Runner(ctx)
    _res, _contact, names, over_box = _fly(ctx, cfg, way, legs, button, eyes, runner)
    assert len(runner.runs) == 1 and runner.runs[0] == legs and names == [lg.name for lg in legs]
    assert over_box is False and runner.notes == [] and runner.stills == [] and eyes[1].n == 0


def test_the_pads_point_is_past_the_heights_point_by_the_fingers():
    assert rp.SIGHT_PADS_TOOL[2] - dp.DOWN_CONTACT_TOOL[2] == pytest.approx(FINGERS_PAST_MODEL_M)
    assert rp.SIGHT_PADS_TOOL[:2] == tuple(dp.DOWN_CONTACT_TOOL[:2])


# ---- after a followed sight: the callers (reach_lid_off.run_full) -------------------------------------------


def test_after_a_followed_sight_the_press_once_more_and_the_grip_go_by_the_sighted_button(ctx, monkeypatch, tmp_path):
    """Case 8, the full run's side: the first press's sight followed 12 mm out (a stand-in for
    fly_press_way_in that flies today's way in and says so), the knob not up: the press once more is solved
    onto the sighted button (and sights again: the run's cameras handed to it), and the lid comes off by it.
    The knob is still judged against the first aim: two of the wrist's own readings from the look pose."""
    from test_reach_lid_off import BUTTON, UP, _full_flow

    from box_opening.tasks import reach_lid_off as rlo

    seen = (BUTTON[0] + 0.012, BUTTON[1] - 0.003)
    flown = []

    def fly(ctx_, cfg, runner, way, legs, top_z, eyes=None, lookahead=None):
        out = rp.fly_way_in(ctx_, runner, legs, way.stroke, top_z, lookahead=lookahead)
        flown.append(eyes)
        if len(flown) == 1:
            way.sighted = seen
        return out

    def spy(real, calls, at):
        def f(*a, **k):
            calls.append(tuple(a[at]))
            return real(*a, **k)

        return f

    solved, took, looked = [], [], []
    monkeypatch.setattr(rp, "fly_press_way_in", fly)
    monkeypatch.setattr(rp, "solve_way_in", spy(rp.solve_way_in, solved, 2))
    monkeypatch.setattr(rlo, "take_lid_off", spy(rlo.take_lid_off, took, 4))
    monkeypatch.setattr(rlo, "knob_look", spy(rlo.knob_look, looked, 4))
    aims = [(BUTTON, 0.4), (BUTTON, 2.0), (BUTTON, 1.5), (UP, 8.0)]
    code, _rec, _chain, _rows = _full_flow(ctx, monkeypatch, tmp_path, aims)
    assert code == 0 and len(flown) == 2 and all(e is not None for e in flown)
    first = solved[0]
    assert solved[-1] == pytest.approx((*seen, first[2])) and took == [pytest.approx((*seen, first[2]))]
    assert looked and all(b == first for b in looked)


@pytest.mark.parametrize("case", ["followed", "in part", "not followed", "not planned"])
def test_the_flight_view_says_what_the_sight_did_under_the_press(ctx, monkeypatch, capsys, case):
    """The sight's own lines, as a run says them, read by the flight view (flight_view.py) as the push's
    are: said under the press's line, in a person's words."""
    from box_opening.flight_view import FlightView

    button = {"followed": NEAR, "in part": EDGE}.get(case, SCENE_1519)
    chain, cfg, way, legs = _way(ctx, monkeypatch, button)
    out_ = np.asarray(button[:2]) / math.hypot(*button[:2])
    true = {
        "followed": (button[0] + 0.012, button[1]),
        "in part": tuple(np.asarray(button[:2]) + 0.020 * out_),
    }.get(case, (button[0] + 0.035, button[1]))
    eyes = _eyes(ctx, monkeypatch, chain, true, button[2])
    _fly(ctx, cfg, way, legs, button, None if case == "not planned" else eyes, _Runner(ctx))
    said = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[press_demo] SIGHT: ")]
    out = []
    view = FlightView(write=out.append, clock=lambda: 0.0)
    view.start()
    view.feed("[press_demo] ONE MOTION from the look pose to below the button: depth:approach, depth:press")
    for line in said:
        view.feed(line)
    notes = [line.strip() for line in out if line.strip().startswith("note")]
    if case == "followed":
        assert len(notes) == 1 and notes[0].startswith("note  looked before the stroke: the button 12 mm from the aim")
        assert notes[0].endswith("the pads brought over it")
    elif case == "in part":
        assert len(notes) == 1 and notes[0].startswith("note  looked before the stroke: the button 20 mm from the aim")
        assert "the pads brought 50% of the way to it (the whole way: the level line over it is not one" in notes[0]
    elif case == "not followed":
        why = "the button sighted 35 mm from the aim, past 30 mm"
        assert notes == ["note  looked before the stroke, pressed by the aim: " + why]
    else:
        assert notes == ["note  no look before the stroke: no wrist camera on this run"]


# ---- round 1: what the review found ------------------------------------------------------------------------


def _aim_kept(chain, way, button):
    """The heights point at the stroke's start over the aimed xy (within 0.5 mm): the press by the aim."""
    heights = _start_xy(chain, way.stroke)
    return math.hypot(heights[0] - button[0], heights[1] - button[1]) < 0.0005


def test_a_followed_sight_after_an_arrival_off_the_plan_solves_its_line_from_where_the_arm_stands(ctx, monkeypatch):
    """The arm arrives at the sight a few mrad off its plan, as the joint impedance leaves it: the level line
    over the button sighted starts where the arm stands, not where the plan ended, and the real pad ends
    still come down within 2 mm of it."""
    button = NEAR
    chain, cfg, way, legs = _way(ctx, monkeypatch, button)
    true = (button[0] + 0.012, button[1] - 0.004)
    eyes = _eyes(ctx, monkeypatch, chain, true, button[2])
    off = [0.004, -0.003, 0.0, 0.003, 0.0, 0.0, 0.0]
    runner = _Runner(ctx, off=off)
    _fly(ctx, cfg, way, legs, button, eyes, runner)
    assert dict(runner.notes)["sight"]["followed"]
    arrived = [float(v) + d for v, d in zip(runner.runs[0][-1].traj.points[-1].positions, off, strict=True)]
    tail = runner.runs[1][0]
    assert tail.name == "sight:tail"
    assert [float(v) for v in tail.traj.points[0].positions] == pytest.approx(arrived, abs=1e-6)
    pads = _start_xy(chain, way.stroke, rp.SIGHT_PADS_TOOL)
    assert math.hypot(pads[0] - true[0], pads[1] - true[1]) < 0.002, (pads, true)


def test_an_arm_not_at_rest_at_the_sight_presses_by_the_aim(ctx, monkeypatch):
    """The wait for rest at the sight timed out: no frame is judged, the press goes by the aim (the heights
    point over it), and the run says the arm did not come to rest."""
    button = NEAR
    chain, cfg, way, legs = _way(ctx, monkeypatch, button)
    eyes = _eyes(ctx, monkeypatch, chain, (button[0] + 0.012, button[1]), button[2])
    runner = _Runner(ctx, restless=("the sight",))
    res, _contact, names, over_box = _fly(ctx, cfg, way, legs, button, eyes, runner)
    note = dict(runner.notes)["sight"]
    assert not note["followed"] and "did not come to rest" in note["why"], note
    assert note["look_ms"] is None  # not looked for
    assert eyes[1].n == 0 and len(runner.runs) == 2 and over_box and all(r.ok for r in res)
    assert names == ["sight:tail", "depth:approach", "depth:press"] and _aim_kept(chain, way, button)
    assert way.sighted is None


@pytest.mark.parametrize("what", ["the stroke", "the line", "the legs"])
def test_an_error_between_the_sight_and_the_stroke_presses_by_the_aim(ctx, monkeypatch, what):
    """An Exception between the first execution's end and the second's start (solving the stroke, the level
    line, building the legs) is no reason not to press: the press goes by the aim, two executions, and the
    run says what raised."""
    button = NEAR
    chain, cfg, way, legs = _way(ctx, monkeypatch, button)
    eyes = _eyes(ctx, monkeypatch, chain, (button[0] + 0.012, button[1]), button[2])
    mod, name = {"the stroke": (dp, "compliant_stroke"), "the line": (rp, "_line"), "the legs": (rp, "_full_world")}[
        what
    ]
    real, calls = getattr(mod, name), []

    def once(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return real(*a, **k)

    class Arrived(_Runner):
        def run(self, legs_, execute=True, **kw):
            out = super().run(legs_, execute, **kw)
            if len(self.runs) == 1:  # the free air arrived: from here on, the first call raises
                monkeypatch.setattr(mod, name, once)
            return out

    runner = Arrived(ctx)
    res, _contact, names, over_box = _fly(ctx, cfg, way, legs, button, eyes, runner)
    note = dict(runner.notes)["sight"]
    assert calls and not note["followed"] and "RuntimeError" in note["why"], note
    assert len(runner.runs) == 2 and over_box and all(r.ok for r in res)
    assert names == ["sight:tail", "depth:approach", "depth:press"] and _aim_kept(chain, way, button)


def test_a_cancel_at_the_sight_is_not_caught(ctx, monkeypatch):
    """A cancel (a BaseException) at the sight goes up as a cancel: no press flies."""
    button = NEAR
    chain, cfg, way, legs = _way(ctx, monkeypatch, button)
    eyes = _eyes(ctx, monkeypatch, chain, button[:2], button[2])
    from box_opening.tasks import press_demo

    def cancel(node):
        raise KeyboardInterrupt

    monkeypatch.setattr(press_demo, "_spin_detect", cancel)
    runner = _Runner(ctx)
    with pytest.raises(KeyboardInterrupt):
        _fly(ctx, cfg, way, legs, button, eyes, runner)
    assert len(runner.runs) == 1


@pytest.mark.parametrize("apart, followed", [(0.005, False), (0.001, True)])
def test_frames_that_each_find_the_button_must_agree_within_2_mm(ctx, monkeypatch, apart, followed):
    """Frames that each find the button, `apart` from the one before: 5 mm apart none agree and the press
    goes by the aim; 1 mm apart they agree and it is followed."""
    button = NEAR
    chain, cfg, way, legs = _way(ctx, monkeypatch, button)
    frames = [_shifted(0.0), _shifted(apart), _shifted(2 * apart), _shifted(3 * apart), _blank]
    eyes = _eyes(ctx, monkeypatch, chain, (button[0] + 0.012, button[1]), button[2], frames=frames)
    monkeypatch.setattr(rp, "SIGHT_S", 3.0)
    runner = _Runner(ctx)
    _fly(ctx, cfg, way, legs, button, eyes, runner)
    note = dict(runner.notes)["sight"]
    assert note["followed"] is followed, note
    if not followed:
        assert "none agreeing" in note["why"] and _aim_kept(chain, way, button)


def test_one_frame_delivered_again_is_one_frame(ctx, monkeypatch):
    """The grabber hands the same frame (one stamp) again and again: it is judged once, and one frame is not
    two that agree."""
    button = NEAR
    chain, cfg, way, legs = _way(ctx, monkeypatch, button)
    eyes = _eyes(ctx, monkeypatch, chain, (button[0] + 0.012, button[1]), button[2], stamps=lambda n: 1)
    monkeypatch.setattr(rp, "SIGHT_S", 0.5)
    runner = _Runner(ctx)
    _fly(ctx, cfg, way, legs, button, eyes, runner)
    note = dict(runner.notes)["sight"]
    assert not note["followed"] and note["why"].startswith("1 frame(s)"), note
    assert eyes[1].n > 2 and _aim_kept(chain, way, button)


# a button near the edge of where a sight is planned: seen 20 mm further out than aimed, the level line the
# whole way meets a joint limit 79 mm down it, half the way flies (0.75 does not)
EDGE = _at((0.70, -10.0))


def test_a_correction_the_arm_cannot_fly_whole_is_flown_in_part(ctx, monkeypatch):
    """The button seen 20 mm further out than aimed, near the edge of the arm's reach: the level line the
    whole way does not fly, the first of 0.75, 0.5, 0.25 of the way that does is flown, the real pad ends
    come down over that point, and the run says how much and why not the whole."""
    button = EDGE
    chain, cfg, way, legs = _way(ctx, monkeypatch, button)
    out = np.asarray(button[:2]) / math.hypot(*button[:2])
    true = tuple(np.asarray(button[:2]) + 0.020 * out)
    eyes = _eyes(ctx, monkeypatch, chain, true, button[2])
    runner = _Runner(ctx)
    _fly(ctx, cfg, way, legs, button, eyes, runner)
    note = dict(runner.notes)["sight"]
    assert note["followed"] and note["part"] in rp.SIGHT_PARTS[1:] and note["whole_why"], note
    seen = np.asarray(way.sighted)
    want = np.asarray(button[:2]) + note["part"] * (seen - np.asarray(button[:2]))
    pads = _start_xy(chain, way.stroke, rp.SIGHT_PADS_TOOL)
    assert math.hypot(*(pads - want)) < 0.0005, (pads, want)
    assert math.hypot(*(seen - np.asarray(true))) < 0.002


def _q(n):
    """Joints labelled `n`: the n-th step along the way in."""
    return [float(n)] * 7


@pytest.mark.parametrize("walked", [True, False])
def test_the_way_back_after_a_sight_is_one_path_from_the_stroke_to_the_aims_look_pose(monkeypatch, walked):
    """_as_flown on a hand-built way, each joint config labelled by its step along the way in: the sight's
    move from the aim's look pose (0) to its own (2), its way down (walked from there; or the slide and then
    the turn), and the level line flown after it (to 6). The way back (reach_lid_off._down_backwards) is then
    one unbroken path, each line starting where the last ended, from the tail's end (6) down every step to
    the move's first point (0), and _look_joints gives that point: where the aim was read."""
    from box_opening.tasks import reach_lid_off as rlo

    monkeypatch.setattr(dp, "compliant_line", lambda ctx, name, pts, quat, speed, world: (name, pts))
    monkeypatch.setattr(rp, "free_line_speed", lambda ctx: 1.0)
    plan = SimpleNamespace(quat=[0.0, 0.0, 0.0, 1.0])
    if walked:
        down, slide = dp.WayDown([_q(2), _q(3), _q(4)], plan.quat, 0.0, True), None
    else:
        down, slide = dp.WayDown([_q(3), _q(4)], plan.quat, 0.0, False), [_q(2), _q(3)]
    sight = SimpleNamespace(way_in=rp.WayIn([_q(0), _q(1), _q(2)], plan, down, slide, None), q=_q(4))
    way = rp.WayIn([_q(9)], SimpleNamespace(quat=None), dp.WayDown([_q(9)], None, 0.0, walked), None, None)
    rp._as_flown(way, sight, [_q(5), _q(6)])
    assert way.move == [] and way.plan is plan
    lines = rlo._down_backwards(None, way, None)
    path = lines[0][1]
    for _name, pts in lines[1:]:
        assert pts[0] == path[-1], lines  # each line starts where the last ended
        path = path + pts[1:]
    assert path == [_q(n) for n in range(6, -1, -1)], lines
    assert rlo._look_joints(way) == _q(0)


def test_a_full_run_with_the_sight_flies_its_way_back_and_its_second_press(ctx, monkeypatch, tmp_path, capsys):
    """run_full on flatbot's chain with the sight flown (no stand-in for it) and the runner's own gates on
    every execution: the first press sighted and followed, its hold not felt, the way back from where the
    stroke started, reversing the lines that flew, the sight's own move too, to the look pose the aim was
    taken from, where the knob is looked at; the knob not up, the press once more onto the sighted button,
    sighting again, and back there; the lid then taken off by the sighted button. No execution is refused,
    nothing stops."""
    import json

    from box_opening.runtime.runner import Runner
    from box_opening.tasks import reach_lid_off as rlo

    button = NEAR
    chain, cfg, way, legs = _way(ctx, monkeypatch, button)
    true = (button[0] + 0.012, button[1] - 0.004)
    eyes = _eyes(ctx, monkeypatch, chain, true, button[2])
    knobs = [rlo.KnobLook(found=True, up_mm=0.4), rlo.KnobLook(found=True, up_mm=9.0)]
    looked, took, holds = [], [], []

    def knob_look(node, ctx_, runner_, watcher, b, yaw, lid_before, press):
        looked.append((press, [float(v) for v in ctx_.client.joints()]))
        return knobs.pop(0)

    def take_lid_off(ctx_, cfg_, runner_, eyes_, at, *a, **k):
        took.append(tuple(at))
        return 0

    def hold(ctx_, cfg_, runner_, stroke, top_z, seen, contact=None):
        holds.append([float(v) for v in ctx_.client.joints()])
        return dp.DepthPress(free=True, contact=False, felt=False, judged=False)  # nothing felt: back as it came

    monkeypatch.setattr(rlo, "knob_look", knob_look)
    monkeypatch.setattr(rlo, "take_lid_off", take_lid_off)
    monkeypatch.setattr(rlo, "room_ahead", lambda ctx_, eyes_: None)
    monkeypatch.setattr(rlo, "_fingers_shut", lambda ctx_, cfg_, runner_: None)
    monkeypatch.setattr(rp, "_home_legs", lambda ctx_, cfg_, start=None: [])
    monkeypatch.setattr(dp, "refuse_press", lambda *a, **k: None)
    monkeypatch.setattr(dp, "compliant_hold", hold)
    runner = Runner(ctx.client, log_dir=tmp_path)
    code = rlo.run_full(ctx, cfg, runner, eyes, way, legs, button, 0.0, None)
    out = capsys.readouterr().out
    assert code == 0, out[-3000:]
    assert "STOP" not in out
    rows = [json.loads(line) for f in sorted(tmp_path.glob("**/*.jsonl")) for line in open(f)]
    sights = [r for r in rows if r.get("leg") == "sight"]
    assert [(r["planned"], r["followed"]) for r in sights] == [(True, True), (True, True)], sights
    # the knob looked at from the look pose the aim was taken from, after each press: both readings from
    # the one pose (the sight's own look pose stands 65 mm further out and 20 mm higher)
    aim_pose = [float(v) for v in legs[0].traj.points[0].positions]
    assert len(holds) == 2 and [p for p, _q in looked] == [1, 2]
    for _press, q in looked:
        assert q == pytest.approx(aim_pose, abs=1e-9), (q, aim_pose)
    assert out.count("from the button back to the look pose") == 2
    assert way.sighted is not None and math.hypot(way.sighted[0] - true[0], way.sighted[1] - true[1]) < 0.002
    assert len(took) == 1 and math.hypot(took[0][0] - true[0], took[0][1] - true[1]) < 0.002
    assert took[0][2] == button[2]


def test_a_part_whose_stroke_the_arm_does_not_fly_is_not_taken(ctx, monkeypatch):
    """A correction whose level line flies but whose stroke from there is one the runner would refuse
    (depth_press.line_refusal on the stroke's own lines, here a stand-in refusing the first stroke solved
    after the sight): that part is not flown, the next (0.75 of the way) is, and the run says why the
    whole way was not."""
    button = NEAR
    chain, cfg, way, legs = _way(ctx, monkeypatch, button)
    true = (button[0] + 0.012, button[1] - 0.004)
    eyes = _eyes(ctx, monkeypatch, chain, true, button[2])
    real_stroke, real_refusal = dp.compliant_stroke, dp.line_refusal
    refused = []

    def stroke(*a, **k):
        out = real_stroke(*a, **k)
        if not refused and out[0] is not None:
            refused.extend(id(line) for _name, line, _speed in out[0].legs)
        return out

    def refusal(chain_, line, *a, **k):
        return "a stroke the runner would refuse" if id(line) in refused else real_refusal(chain_, line, *a, **k)

    class Arrived(_Runner):
        def run(self, legs_, execute=True, **kw):
            out = super().run(legs_, execute, **kw)
            if len(self.runs) == 1:  # the free air arrived: the first stroke solved from here is refused
                monkeypatch.setattr(dp, "compliant_stroke", stroke)
                monkeypatch.setattr(dp, "line_refusal", refusal)
            return out

    runner = Arrived(ctx)
    _fly(ctx, cfg, way, legs, button, eyes, runner)
    note = dict(runner.notes)["sight"]
    assert refused and note["followed"] and note["part"] == 0.75, note
    assert note["whole_why"].endswith(": a stroke the runner would refuse"), note
    assert not note["whole_why"].startswith("the level line"), note


def test_a_frame_older_than_the_last_judged_is_not_judged(ctx, monkeypatch):
    """The grabber hands an older frame after a newer one (a stamp going back): it is not judged, so one
    frame and an older copy of it are not two that agree."""
    button = NEAR
    chain, cfg, way, legs = _way(ctx, monkeypatch, button)
    eyes = _eyes(
        ctx, monkeypatch, chain, (button[0] + 0.012, button[1]), button[2], stamps=lambda n: 10 if n == 1 else 5
    )
    monkeypatch.setattr(rp, "SIGHT_S", 0.5)
    runner = _Runner(ctx)
    _fly(ctx, cfg, way, legs, button, eyes, runner)
    note = dict(runner.notes)["sight"]
    assert not note["followed"] and note["why"].startswith("1 frame(s)"), note
    assert eyes[1].n > 2 and _aim_kept(chain, way, button)
