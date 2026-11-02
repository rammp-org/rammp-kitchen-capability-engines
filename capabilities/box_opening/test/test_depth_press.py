"""The depth press (tasks/depth_press): once the hand is over the box, the
height from the wrist camera's own depth, the press from the arm's torques."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from box_opening.constants import GAINS_STIFF, IMPEDANCE_KQ
from box_opening.models.container import ContainerModel, load_press_demo
from box_opening.perception.d405 import quat_to_mat
from box_opening.tasks import depth_press as dp
from box_opening.tasks import reach_press as rp

PINK = "capabilities/box_opening/config/containers/ankou_pink.yaml"
BUTTON = (0.495, -0.16, 0.24)


@pytest.fixture(scope="module")
def pink():
    return ContainerModel.load(PINK)


# ---- what the wrist camera measures, in its own frame -------------------------


def test_the_button_is_found_in_the_lens_frame_with_no_transform_at_all(pink):
    from test_button_aim import BOX_H, BUTTON_XY, OAK_K, ROT_DOWN, _h0, render

    h0 = _h0(pink)
    b = np.r_[BUTTON_XY, h0]
    for off in ((0.0, 0.0), (0.03, -0.02), (-0.04, 0.025)):
        cam = b + np.r_[off[0], off[1], 0.26]
        color, depth = render(cam, ROT_DOWN, [(BUTTON_XY, h0, 0.0)], h0 - BOX_H)
        s, why = dp.find_in_camera(color, depth, OAK_K, pink)
        assert why is None, why
        want = ROT_DOWN.T @ (b - cam)  # where the button is in the camera's frame
        assert (s.x, s.y) == pytest.approx((want[0], want[1]), abs=0.002)
        assert s.lid_depth == pytest.approx(0.26, abs=0.003) and abs(s.up_mm) < 2.0
    cam = b + np.r_[0.0, 0.0, 0.26]
    color, depth = render(cam, ROT_DOWN, [(BUTTON_XY, h0, 0.012)], h0 - BOX_H)
    s, why = dp.find_in_camera(color, depth, OAK_K, pink)
    assert why is None, why
    assert s.up_mm == pytest.approx(12.0, abs=3.0) and s.button_depth < s.lid_depth  # the knob up


def test_a_look_that_knows_where_the_button_must_be_takes_no_circle_anywhere_else(pink):
    """Flatbot 2026-10-23 17:30: after a 21 mm centring move the look took a
    circle on the cabinet 288 mm away and 99 mm deeper, in the frames Hough
    missed the button in. Told where the button must be, the finder takes
    nothing else — and still finds the button there."""
    from test_button_aim import BOX_H, BUTTON_XY, OAK_K, ROT_DOWN, _h0, render

    h0 = _h0(pink)
    b = np.r_[BUTTON_XY, h0]
    cam = b + np.r_[0.0, 0.0, 0.26]
    want = ROT_DOWN.T @ (b - cam)
    here = (want[0], want[1], 0.26)
    decoy = (BUTTON_XY[0] + 0.06, BUTTON_XY[1] - 0.12)  # another round lid's button, 10 cm lower, 13 cm off
    # only the decoy in view, as in the frames Hough missed the button in
    color, depth = render(cam, ROT_DOWN, [(decoy, h0 - 0.10, 0.0)], h0 - BOX_H)
    s, why = dp.find_in_camera(color, depth, OAK_K, pink)
    assert why is None and s.lid_depth == pytest.approx(0.36, abs=0.005)  # what the look used to take
    s, why = dp.find_in_camera(color, depth, OAK_K, pink, expect=here)
    assert s is None and why
    # both in view: the button, where it must be
    color, depth = render(cam, ROT_DOWN, [(BUTTON_XY, h0, 0.0), (decoy, h0 - 0.10, 0.0)], h0 - BOX_H)
    s, why = dp.find_in_camera(color, depth, OAK_K, pink, expect=(want[0] + 0.01, want[1] - 0.01, 0.265))
    assert why is None and (s.x, s.y) == pytest.approx((want[0], want[1]), abs=0.002)
    # ... and nothing on a lid at another depth, even right where the button should be
    s, why = dp.find_in_camera(color, depth, OAK_K, pink, expect=(want[0], want[1], 0.26 + 2 * dp.LOOK_EXPECT_DEPTH_M))
    assert s is None


def test_the_depth_under_the_fingers_is_read_where_the_fingers_line_meets_the_surface(pink):
    from test_button_aim import BOX_H, BUTTON_XY, OAK_K, ROT_DOWN, _h0, render

    # ROT_DOWN: the camera's +x is base -y, its +y base -x. A lens at
    # (lx, ly) from the fingers' line, in the camera's axes, stands at
    # fingers - ly * base_x... — here the fingers' line runs through the button
    h0 = _h0(pink)
    for lens_xy in ((0.0, 0.10), (0.0, 0.065), (0.02, 0.06)):
        for height in (0.24, 0.30, 0.40):
            off = ROT_DOWN @ np.r_[lens_xy, 0.0]  # the lens from the fingers' line, in base
            cam = np.r_[BUTTON_XY[0] + off[0], BUTTON_XY[1] + off[1], h0 + height]
            color, depth = render(cam, ROT_DOWN, [(BUTTON_XY, h0, 0.0)], h0 - BOX_H)
            assert dp.depth_under_fingers(depth, OAK_K, lens_xy) == pytest.approx(height, abs=0.004)
            # and the button itself is in view where the fingers' line meets it
            s, why = dp.find_in_camera(color, depth, OAK_K, pink)
            assert why is None, why
            assert dp.under_fingers(s, lens_xy) == pytest.approx((0.0, 0.0), abs=0.003)
            assert s.lid_depth == pytest.approx(height, abs=0.004)
    # nearer than the frame's edge can see on that line: no reading
    flat = np.full((400, 640), 0.15)
    assert dp.depth_under_fingers(flat, OAK_K, (0.0, 0.10)) is None
    # on the lens's own axis it is the depth straight ahead
    assert dp.depth_under_fingers(np.full((400, 640), 0.33), OAK_K, (0.0, 0.0)) == pytest.approx(0.33, abs=0.003)


def test_a_camera_turned_in_the_hand_reads_under_the_fingers_through_its_own_axes(pink):
    """The bench D405 sits 180 deg about its lens axis: the sighting, in the
    camera's axes, is turned into the hand's before the lens offset is added,
    and the fingers' line is projected through the camera's axes."""
    from test_button_aim import BOX_H, BUTTON_XY, OAK_K, ROT_DOWN, _h0, render

    h0 = _h0(pink)
    for lens_xy in ((0.0, 0.075), (0.02, 0.06)):
        for height in (0.24, 0.30):
            off = ROT_DOWN @ np.r_[lens_xy, 0.0]  # the hand's axes are ROT_DOWN's, the camera turned in it
            cam = np.r_[BUTTON_XY[0] + off[0], BUTTON_XY[1] + off[1], h0 + height]
            color, depth = render(cam, ROT_DOWN @ RZ180, [(BUTTON_XY, h0, 0.0)], h0 - BOX_H)
            assert dp.depth_under_fingers(depth, OAK_K, lens_xy, RZ180) == pytest.approx(height, abs=0.004)
            s, why = dp.find_in_camera(color, depth, OAK_K, pink)
            assert why is None, why
            assert dp.under_fingers(s, lens_xy, RZ180) == pytest.approx((0.0, 0.0), abs=0.003)
            # the camera's axes dropped: twice the lens offset, the wrong way
            assert np.hypot(*dp.under_fingers(s, lens_xy)) == pytest.approx(2 * np.hypot(*lens_xy), abs=0.004)


def test_the_lens_in_the_hand_keeps_the_cameras_axes(ctx, monkeypatch):
    from box_opening.perception import depth_source
    from box_opening.runtime import approach

    arm = _Arm()
    monkeypatch.setattr(approach, "_chain", lambda: arm)
    quat = rp.reach_quat(ContainerModel.load(PINK), BUTTON[:2], 0)
    ctx.client.live = [0.5, -0.1, 0.6] + list(quat)
    R, t = arm.fk(ctx.client.live)
    monkeypatch.setattr(depth_source, "camera_pose_at", lambda g: (R @ RZ180, t + R @ np.r_[0.0, 0.075, -0.125]))
    lens, r_hc = dp.lens_in_hand(ctx, None)
    assert lens == pytest.approx([0.0, 0.075, -0.125]) and r_hc == pytest.approx(RZ180)


def test_the_lid_is_the_nearest_flat_surface():
    depth = np.full((400, 640), 0.42)  # the table
    depth[150:250, 240:400] = 0.31  # the lid, nearer
    depth[::7, ::5] = 0.0  # holes
    g = dp.lid_depth_guesses(depth)
    assert g[0] == pytest.approx(0.31, abs=0.003) and g[1] == pytest.approx(0.42, abs=0.003)
    assert dp.lid_depth_guesses(np.zeros((400, 640))) == []


# ---- the press, the arm faked ---------------------------------------------------


class _Arm:
    """A chain whose joints ARE the tool pose — position then quaternion — so
    every line and turn flies and fk follows the arm."""

    def __init__(self, refuse_turn=False, refuse_first=0):
        self.lines, self.turns, self.refuse_turn = [], [], refuse_turn
        self.refuse_first = refuse_first  # the first this many turn attempts (exactly down, 2 deg short ...) refused
        self.trace = []  # legs, looks and at-rest reads, in order (_press)

    def fk(self, q):
        return np.asarray(quat_to_mat(*q[3:7]), float), np.asarray(q[:3], float)

    def straight_line(self, q0, xyz, quat):
        self.lines.append(list(xyz))
        return [list(q0), [float(v) for v in xyz] + [float(v) for v in quat]], None

    def walk(self, q0, xyz, quat, step_m=None, step_rad=None, max_step_rad=None):
        if self.refuse_turn:
            return None, "a joint limit 40% of the way"
        if self.refuse_first > 0:
            self.refuse_first -= 1
            return None, "a 0.26 rad jump in the last step"
        self.turns.append((list(xyz), list(quat)))
        return [list(q0), [float(v) for v in xyz] + [float(v) for v in quat]], None


class _Gravity:
    def wrist_holding(self, q):
        return [0.0] * 4

    def tip_load(self, q, p):
        return [0.48, 0.0, 0.2, 0.0]


LENS = (0.0, 0.065, -0.105)  # the lens from the tool frame, in the hand: 65 mm beside the fingers, 105 mm behind
EYE = np.eye(3)
RZ180 = np.diag([-1.0, -1.0, 1.0])  # the bench D405: 180 deg about its lens axis
# where TF's lens puts the fingertips in front of it with LENS (116 mm): the lid
# 270 mm below the lens is met 154 mm down
TOUCH_AT = 0.27 - (dp.DOWN_CONTACT_TOOL[2] - LENS[2])


def _sighting(x=0.0, y=-LENS[1], lid=0.27, up=0.4):
    """The button seen at (x, y) from the lens axis: by default under the fingers."""
    return dp.LensSighting(x=x, y=y, lid_depth=lid, button_depth=lid - up / 1000.0, up_mm=up, raw=None)


def _seen(hand_off=(0.0, 0.0), depth=0.27, lens=LENS, r_hc=EYE, up=0.4):
    """A look that sees one fixed button wherever the fake arm stands: its
    lid `hand_off` from the fingers' line, in the hand's axes, `depth` below
    the lens along the tool, from the hand at the first look. The sighting is in
    the camera's own axes (r_hc: the camera's axes in the hand's)."""
    world = {}

    def at(ctx, up=up):
        q = ctx.client.joints()
        R, t = np.asarray(quat_to_mat(*q[3:7]), float), np.asarray(q[:3], float)
        if "b" not in world:
            world["b"] = t + R @ np.r_[hand_off[0], hand_off[1], lens[2] + depth]
        c = np.asarray(r_hc).T @ (R.T @ (world["b"] - t) - np.asarray(lens))
        s = dp.LensSighting(
            x=float(c[0]),
            y=float(c[1]),
            lid_depth=float(c[2]),
            button_depth=float(c[2]) - up / 1000.0,
            up_mm=up,
            raw=None,
        )
        return s, None, "2/2"

    at.world = world
    at.after = lambda ctx: at(ctx, up=12.0)  # the same button, the knob up
    return at


def _press(
    ctx,
    monkeypatch,
    tmp_path,
    looks,
    tilt=30,
    touch_at=TOUCH_AT,
    push="touch",
    refuse_turn=False,
    lens=LENS,
    r_hc=EYE,
    measured=LENS,
    refuse_first=0,
    here=None,
    turned=None,
    knob_up_mm=None,
    aimed=None,
    compliant=False,
    top_z=None,
    held_up_m=0.0,
):
    """press_down_from_here with the hand `tilt` deg over BUTTON (at the
    reach-out's stroke start, 6 cm above TF's lid at 0.40), the wrist camera
    scripted by `looks` (a (sighting, z, status) or a _seen) with TF's lens
    at `lens`, turned `r_hc`, against the `measured` one in the camera
    config, and the fingertips meeting the lid `touch_at` down the stroke;
    `aimed`, the caller's aim. Every at-rest read is a distinct reading,
    ("rest", n, where) in arm.trace. Returns (DepthPress, events, arm,
    legs, start tool pose)."""
    from box_opening.runtime import approach
    from box_opening.runtime.runner import Runner

    cfg = load_press_demo(PINK)
    ctx.model = ContainerModel.load(PINK)
    ctx.press_cfg = cfg
    arm = _Arm(refuse_turn, refuse_first)
    monkeypatch.setattr(approach, "_chain", lambda: arm)
    monkeypatch.setattr(approach, "_gravity", lambda: _Gravity())
    quat = rp.reach_quat(ctx.model, BUTTON[:2], tilt)
    start = rp.tool_for_contact((BUTTON[0], BUTTON[1], 0.46), quat)  # TF's stroke start: 6 cm over its lid at 0.40
    ctx.client.live = list(start) + list(quat)
    events, legs = [], {}
    seq = list(looks)
    arm.reads = []

    def settled(*a, **k):
        live = list(ctx.client.live)
        if held_up_m and events and events[-1] == "depth:press":
            live[2] += held_up_m  # the spring held the pads this far above the stroke's end
        got = ([0.001 * (len(arm.reads) + 1), 0.0, 0.0, 0.0], live)
        arm.reads.append(got)
        return got

    ctx.client.settled_efforts = settled
    real_settled = rp._settled
    monkeypatch.setattr(
        rp, "_settled", lambda ctx_, where: (arm.trace.append("rest:" + where), real_settled(ctx_, where))[1]
    )

    arm.expects = []  # each look's `expect`, in order — one given as a function of the lens, at the look

    def look(node, watcher, model, tag, runner=None, ctx=None, settled=False, expect=None):
        events.append("look:" + tag)
        arm.trace.append("look:%s%s" % (tag, " (settled)" if settled else ""))
        arm.expects.append(expect((list(lens), np.asarray(r_hc, float))) if callable(expect) else expect)
        item = seq.pop(0)
        s, z, status = item(ctx) if callable(item) else item[:3]
        return s, z, (list(lens), np.asarray(r_hc, float)), status

    monkeypatch.setattr(dp, "look", look)

    class R(Runner):
        def run(self, legs_, execute, lookahead=None):
            for lg in legs_:
                events.append(lg.name)
                arm.trace.append(lg.name)
                legs[lg.name] = lg
            name = legs_[-1].name if legs_ else None
            top = list(ctx.client.live)
            # the stroke flies fast, then slowly (depth:approach, depth:touch):
            # `touch_at` is down from where the stroke began, and a lid higher
            # than the fast stretch reaches is met on it
            if name == "depth:approach":
                self.stroke_top = top
            stroke_top = getattr(self, "stroke_top", None) or top
            meets = touch_at is not None and (
                name == "depth:touch"
                or (name == "depth:approach" and stroke_top[2] - legs_[-1].traj.points[-1].positions[2] >= touch_at)
            )
            if meets:
                ctx.client.exec_script = [
                    ("touch", {"message": "torque guard trip", "progress": 0.7, "torque_peak": 3.2})
                ]
            if name == "press:push":
                ctx.client.exec_script = [
                    (push, {"message": "x", "progress": 0.6 if push == "touch" else 1.0, "torque_peak": 7.4})
                ]
            out = super().run(legs_, execute, lookahead)
            if meets:
                ctx.client.live = [stroke_top[0], stroke_top[1], stroke_top[2] - touch_at] + stroke_top[
                    3:
                ]  # stopped on the lid
            if name == "depth:touch" or meets:
                self.stroke_top = None
            return out

    runner = R(ctx.client, log_dir=tmp_path)
    contact = rp.CONTACT_TOOL if tilt else dp.DOWN_CONTACT_TOOL
    watcher = SimpleNamespace(grab=SimpleNamespace(wrist_lens_in_hand_m=None if measured is None else list(measured)))
    result = dp.press_down_from_here(
        None,
        ctx,
        cfg,
        runner,
        watcher,
        contact,
        here=here,
        turned=turned,
        knob_up_mm=knob_up_mm,
        aimed=aimed,
        compliant=compliant,
        top_z=top_z,
    )
    return result, events, arm, legs, start


def test_over_the_box_it_turns_down_measures_the_depth_and_presses_by_feel(ctx, monkeypatch, tmp_path):
    looks = [(_sighting(), 0.27, "2/2"), (_sighting(up=12.0), 0.255, "2/2")]
    result, events, arm, legs, start = _press(ctx, monkeypatch, tmp_path, looks)
    assert dp.exit_code(result) == 0 and result.free and result.felt and result.popped
    assert events == [
        "depth:turn",
        "look:depth-look",
        "depth:approach",
        "depth:touch",
        "press:push",
        "depth:unload",
        "depth:rise",
        "look:depth-after",
    ]
    # TURN: straight down, about the fingertip — it stays where it was over the button
    ((goal, quat),) = arm.turns
    R0 = rp._rot(rp.reach_quat(ctx.model, BUTTON[:2], 30))
    tip = np.asarray(start) + R0 @ np.asarray(rp.CONTACT_TOOL)
    Rd = rp._rot(quat)
    assert Rd[2, 2] == pytest.approx(-1.0)
    assert np.asarray(goal) + Rd @ np.asarray(dp.DOWN_CONTACT_TOOL) == pytest.approx(tip)
    # TOUCH: straight down, fast to TOUCH_SLOW_M short of where the depth and
    # TF's mount put the lid (2026-10-23: all of it at press_speed met the lid
    # at 12.4 Nm), then slowly, no farther than the depth plus travel_m
    cfg = load_press_demo(PINK)
    fast, stroke = arm.lines[0], arm.lines[1]
    ahead_tf = dp.DOWN_CONTACT_TOOL[2] - LENS[2]
    assert fast[:2] == pytest.approx(list(goal[:2])) and goal[2] - fast[2] == pytest.approx(
        0.27 - ahead_tf - dp.TOUCH_SLOW_M
    )
    assert stroke[:2] == pytest.approx(list(goal[:2])) and goal[2] - stroke[2] == pytest.approx(0.27 + cfg.travel_m)
    g = legs["depth:touch"].guard
    assert g.trip == "touch" and g.gravity is not None and g.reference is not None
    assert legs["depth:approach"].guard.trip == "touch" and legs["depth:approach"].guard.reference is g.reference
    # PUSH: button_travel_m from where the torques felt the lid, backstopped
    assert (
        arm.lines[2][2] == pytest.approx(goal[2] - TOUCH_AT - cfg.button_travel_m)
        and legs["press:push"].guard.trip == "press"
    )
    # ... its backstop the TOTAL off the free-air reading over the box (touch_nm), not touch_nm - contact_nm on
    # top of a first sample the touch's overshoot already loaded; the recoil carries on back up the touch
    gp = legs["press:push"].guard
    assert gp.touch_nm == pytest.approx(ctx.model.touch_nm) and gp.gravity is not None
    assert gp.reference is g.reference and gp.reference is not None
    assert legs["press:push"].recoil_via is legs["depth:touch"].traj
    # RISE: back to where the touch began, judged for a pull at the fingertip
    assert arm.lines[-1][2] == pytest.approx(goal[2]) and legs["depth:rise"].guard.hang is not None


def test_only_the_lines_up_or_in_place_fly_faster_and_every_guard_is_as_before(ctx, monkeypatch, tmp_path):
    """Run 20:04 flew the turn (3.3 mm of tool travel) in 7.43 s and the rise
    in 5.56 s at CONTACT_SPEED. The turn pivots about the fingertip and the
    rise starts after the at-rest check: FREE_LINE_SPEED. Off the button and
    the push, as before; the touch flies at press_speed to TOUCH_SLOW_M short
    of the lid and at TOUCH_SPEED onto it (2026-10-23: 12.4 Nm at 0.35)."""
    from box_opening.constants import CONTACT_SPEED, FREE_LINE_SPEED

    cfg = load_press_demo(PINK)
    looks = [(_sighting(x=0.02), 0.27, "2/2"), (_sighting(), 0.27, "2/2"), (_sighting(up=12.0), None, "2/2")]
    result, events, arm, legs, _s = _press(ctx, monkeypatch, tmp_path, looks)
    assert dp.exit_code(result) == 0 and "depth:centre" in events
    speeds = {name: lg.speed for name, lg in legs.items()}
    assert speeds == {
        "depth:turn": FREE_LINE_SPEED,
        "depth:centre": FREE_LINE_SPEED,
        "depth:approach": cfg.press_speed,
        "depth:touch": dp.TOUCH_SPEED,
        "press:push": cfg.press_speed,
        "depth:unload": CONTACT_SPEED,
        "depth:rise": FREE_LINE_SPEED,
    }
    assert dp.CENTRE_SPEED == FREE_LINE_SPEED and CONTACT_SPEED < FREE_LINE_SPEED < cfg.press_speed
    over_box, down = arm.reads[0], arm.reads[1]  # 'over the box', then 'pointing down' after the turn
    turn, rise, unload = legs["depth:turn"].guard, legs["depth:rise"].guard, legs["depth:unload"].guard
    assert (turn.touch_nm, turn.trip, turn.reference, turn.hang) == (rp.DRAG_NM, "obstruction", over_box, None)
    assert turn.gravity is not None
    for g in (rise, unload):
        assert (g.touch_nm, g.trip, g.reference) == (ctx.model.touch_nm, "obstruction", down)
        assert g.hang[1] == rp.HANG_N and g.gravity is not None
    assert legs["depth:touch"].guard.touch_nm == cfg.contact_nm and legs["depth:touch"].guard.reference == down


def test_one_at_rest_read_where_the_arm_stands_and_a_fresh_one_after_a_turn(ctx, monkeypatch, tmp_path):
    """Each at-rest read holds the arm 0.45 s. The turn moved: its own read
    pointing down is the touch's reference."""
    looks = [(_sighting(), 0.27, "2/2"), (_sighting(up=12.0), None, "2/2")]
    _r, _e, arm, legs, _s = _press(ctx, monkeypatch, tmp_path, looks)
    assert arm.trace[:4] == [
        "rest:over the box",
        "depth:turn",
        "rest:over the button, pointing down",
        "look:depth-look (settled)",
    ]
    assert legs["depth:touch"].guard.reference == arm.reads[1]


def test_at_staging_one_at_rest_read_before_the_look(ctx, monkeypatch, tmp_path):
    looks = [(_sighting(), 0.27, "2/2"), (_sighting(up=12.0), None, "2/2")]
    result, _e, arm, legs, _s = _press(ctx, monkeypatch, tmp_path, looks, tilt=0)
    assert dp.exit_code(result) == 0
    assert arm.trace[:2] == ["rest:over the box", "look:depth-look (settled)"]  # no second read: the turn did not move
    assert legs["depth:touch"].guard.reference == arm.reads[0]


def test_the_callers_read_is_the_reference_when_nothing_moved_since(ctx, monkeypatch, tmp_path):
    """The reach-out read the stroke's start a moment before: the same
    reading at the same joints, not read again."""
    here = ([0.5, 0.0, 0.0, 0.0], None)
    looks = [(_sighting(), 0.27, "2/2"), (_sighting(up=12.0), None, "2/2")]
    _r, _e, arm, legs, _s = _press(ctx, monkeypatch, tmp_path, looks, here=here)
    assert arm.trace[:2] == ["depth:turn", "rest:over the button, pointing down"]
    assert legs["depth:turn"].guard.reference == here


def test_a_hand_that_slid_in_turning_is_not_turned_again(ctx, monkeypatch, tmp_path):
    """The reach-out's walk in landed 2 deg short of straight down: turning
    to exactly down from there is the wrist's singular case. No turn, and
    the caller's read is the reference. CENTRE takes the lean out instead:
    the button on the tilted fingers' line is off the fingertip's vertical."""
    here = ([0.5, 0.0, 0.0, 0.0], None)
    short = dp.down_quat(ContainerModel.load(PINK), BUTTON[:2], 2.0)
    seen = _seen((0.0, 0.0))
    looks = [seen, seen, seen.after]
    result, events, arm, legs, _s = _press(ctx, monkeypatch, tmp_path, looks, tilt=2, here=here, turned=short)
    assert dp.exit_code(result) == 0 and arm.turns == [] and "depth:turn" not in events and "depth:centre" in events
    assert arm.trace[0] == "look:depth-look (settled)" and not any(t.startswith("rest:over") for t in arm.trace)
    assert legs["depth:touch"].guard.reference == here and legs["depth:touch"].target[2] == pytest.approx(short)


def test_the_push_backstop_is_a_ceiling_off_free_air_not_on_top_of_the_touch_overshoot(ctx, monkeypatch, tmp_path):
    """Run 2026-10-22 20:04 replayed (joint_4, the trip reports): over the box
    it read 8.211 Nm in free air; the 0.35 touch overshot to 6.1 off that, so
    the push's first sample (2.738) was already 5.5 Nm in. The raw backstop
    (touch_nm - contact_nm = 7 Nm over THAT) rode to -4.972: 13.2 Nm off free
    air against touch_nm 10, on the lid's edge. Judged off the free-air
    reading the push trips at the first sample past touch_nm."""
    from dataclasses import replace

    from box_opening.runtime.guards import CompensatedGuard, TorqueGuard

    looks = [(_sighting(), 0.27, "2/2"), (_sighting(up=12.0), 0.255, "2/2")]
    _r, _e, _a, legs, _s = _press(ctx, monkeypatch, tmp_path, looks)
    cfg = load_press_demo(PINK)
    free_air = [8.211, -0.419, 0.926, -0.254]
    push = [
        [2.738, -0.857, 0.754, -0.41],
        [-2.857, -1.583, 2.064, -0.421],
        [-3.72, -1.562, 2.129, -0.439],
        [-4.204, -1.577, 2.297, -0.415],
        [-4.972, -1.727, 2.288, -0.428],
    ]
    q = [0.0] * 7

    def first_trip(guard):
        guard.on_progress(0.01)
        for i, eff in enumerate(push):
            if guard.on_efforts(eff, q) if isinstance(guard, CompensatedGuard) else guard.on_efforts(eff):
                return i, free_air[0] - eff[0]
        return None, None

    spec = replace(legs["press:push"].guard, reference=(free_air, q))
    now = CompensatedGuard(spec.touch_nm, spec.gravity, reference=spec.reference)
    was = TorqueGuard(ctx.model.touch_nm - cfg.contact_nm)
    i_now, total_now = first_trip(now)
    i_was, total_was = first_trip(was)
    assert (i_was, round(total_was, 1)) == (4, 13.2)  # what the run did
    assert i_now == 1 and total_now < total_was  # the first sample the stream had past 10 Nm (11.1)
    assert all(free_air[0] - e[0] <= ctx.model.touch_nm for e in push[:i_now])


def test_the_touch_measures_how_far_in_front_of_the_lens_the_fingertips_are(ctx, monkeypatch, tmp_path):
    import json

    looks = [(_sighting(lid=0.27), 0.27, "2/2"), (_sighting(up=12.0), None, "2/2")]
    result, _e, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, looks, touch_at=0.155)
    note = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl"))) if '"depth_touch"' in line][0]
    assert note["down_mm"] == pytest.approx(155.0) and note["fingertips_ahead_of_lens_mm"] == pytest.approx(115.0)
    assert result.popped


def test_at_staging_the_hand_already_points_down_and_does_not_turn(ctx, monkeypatch, tmp_path):
    looks = [(_sighting(), 0.27, "2/2"), (_sighting(up=12.0), None, "2/2")]
    result, events, arm, _l, _s = _press(ctx, monkeypatch, tmp_path, looks, tilt=0)
    assert "depth:turn" not in events and arm.turns == [] and dp.exit_code(result) == 0


def test_without_the_button_in_view_it_does_not_press_blind(ctx, monkeypatch, tmp_path):
    result, events, arm, _l, _s = _press(ctx, monkeypatch, tmp_path, [(None, 0.31, "the button in none")])
    assert result.free and "not pressing blind" in result.why and "depth:touch" not in events
    assert arm.lines == [] and dp.exit_code(result) == 1


# ---- the caller's aim: where the first look expects the button, and where no look confirms it -----

AIMED = (BUTTON[0], BUTTON[1], 0.46 - TOUCH_AT + 0.0004)  # under the turned-down pads, 0.4 mm up off its lid
DEPTH_1433 = Path(__file__).parent / "data" / "wrist_depth_look_flatbot_20261027_1433.npz"


def test_the_first_look_expects_the_button_where_the_callers_aim_put_it(ctx, monkeypatch, tmp_path):
    """Flatbot 2026-10-27 14:33: the fingers stood over the tilted look's
    button, and the first close-up, told nothing, took a circle on the
    lid's edge 64 mm off it. Handed the aim, the first look is told where
    that button must be seen from where the hand stands."""
    looks = [(_sighting(), 0.27, "2/2"), (_sighting(up=12.0), 0.255, "2/2")]
    result, _e, arm, _l, _s = _press(ctx, monkeypatch, tmp_path, looks, aimed=AIMED, knob_up_mm=0.4)
    assert dp.exit_code(result) == 0
    assert arm.expects[0] == pytest.approx((0.0, -LENS[1], 0.27), abs=1e-6)  # under the fingers, its lid 0.27 down


@pytest.mark.parametrize("z_under", [0.2804, 0.26])
def test_where_no_look_confirms_the_button_the_aim_stands_in(ctx, monkeypatch, tmp_path, z_under):
    """14:33 again: turned down there, the lens 65 mm beside the fingers sees
    the button under them at the frame's top edge, its ring cut off, and no
    look confirms it. The aim stands in where the camera's own depth under
    the fingers backs it (14:33: the button's face 11 mm deeper than the
    aim's): the press goes straight down there, fast to TOUCH_SLOW_M short
    of the nearer of the two lids, and the after-look is told where the aim
    put the button."""
    import json

    looks = [(None, z_under, "9 frame(s), the button in none"), (_sighting(up=12.0), None, "2/2")]
    result, events, arm, _l, _s = _press(ctx, monkeypatch, tmp_path, looks, aimed=AIMED, knob_up_mm=0.4)
    assert dp.exit_code(result) == 0 and result.felt and result.popped and not result.moved
    assert events == [
        "depth:turn",
        "look:depth-look",
        "depth:approach",
        "depth:touch",
        "press:push",
        "depth:unload",
        "depth:rise",
        "look:depth-after",
    ]
    ((goal, _q),) = arm.turns
    fast = arm.lines[0]
    ahead_tf = dp.DOWN_CONTACT_TOOL[2] - LENS[2]
    assert fast[:2] == pytest.approx(list(goal[:2]))  # over the aim already: no centring
    assert goal[2] - fast[2] == pytest.approx(min(0.27, z_under + 0.0004) - ahead_tf - dp.TOUCH_SLOW_M)
    assert arm.expects[1] == pytest.approx((0.0, -LENS[1], 0.27), abs=1e-6)
    rows = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl")))]
    (note,) = (r for r in rows if r.get("leg") == "depth_trusted_aim")
    assert note["ok"] and note["under_fingers_mm"] == pytest.approx(1000 * z_under)
    assert [r for r in rows if r.get("leg") == "depth_centred"][0]["by_aim"] is True


@pytest.mark.parametrize(
    "z_under, aimed, why",
    [
        (None, AIMED, "the camera reads no depth under the fingers"),
        (0.32, AIMED, "the surface under the fingers 320 mm below the lens, the aimed button 270"),
        (0.27, (AIMED[0] + 0.04, AIMED[1], AIMED[2]), "the aimed button is 40 mm from under the fingers"),
    ],
)
def test_the_aim_does_not_stand_in_where_the_camera_does_not_back_it(ctx, monkeypatch, tmp_path, z_under, aimed, why):
    looks = [(None, z_under, "9 frame(s), the button in none")]
    result, events, arm, _l, _s = _press(ctx, monkeypatch, tmp_path, looks, aimed=aimed, knob_up_mm=0.4)
    assert result.free and why in result.why and "not pressing on the aim" in result.why
    assert events == ["depth:turn", "look:depth-look"] and arm.lines == [] and dp.exit_code(result) == 1


def test_an_aim_a_few_mm_off_is_centred_on_by_the_arms_own_kinematics(ctx, monkeypatch, tmp_path):
    """No look after the move — none would confirm the button there either:
    the arm's own kinematics put the aim under the fingers."""
    off = (AIMED[0] + 0.008, AIMED[1] - 0.006, AIMED[2])
    looks = [(None, 0.27, "9 frame(s), the button in none"), (_sighting(up=12.0), None, "2/2")]
    result, events, arm, _l, _s = _press(ctx, monkeypatch, tmp_path, looks, aimed=off, knob_up_mm=0.4)
    assert dp.exit_code(result) == 0
    assert events[:5] == ["depth:turn", "look:depth-look", "depth:centre", "depth:approach", "depth:touch"]
    assert arm.lines[0][:2] == pytest.approx(list(off[:2])) and arm.lines[1][:2] == pytest.approx(list(off[:2]))


@pytest.mark.parametrize("miss, moved", [(0.010, False), (0.020, True)])
def test_after_a_press_by_the_aim_its_own_miss_is_no_move(ctx, monkeypatch, tmp_path, miss, moved):
    """The after-look's button against the aim's holds the aim's own miss
    (14:33: ~10 mm): only a box slid farther than SLIDE_OK_M has moved."""
    looks = [(None, 0.2804, "9 frame(s), the button in none"), (_sighting(y=-LENS[1] - miss, up=12.0), None, "2/2")]
    result, _e, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, looks, aimed=AIMED, knob_up_mm=0.4)
    assert result.moved is moved and result.shift_m == pytest.approx(miss, abs=1e-4)
    assert dp.exit_code(result) == (1 if moved else 0)


def test_the_1433_close_up_takes_the_lids_edge_and_backs_the_aim(ctx, monkeypatch, pink):
    """The 14:33 first close-up itself (flatbot 2026-10-27), the hand turned
    down over the tilted look's button, the lid 0.18 m below the lens. Told
    nothing, the finder takes a circle more than LOOK_EXPECT_M from where
    the aim puts the button (the run took one 64 mm off, and flew there).
    Told the aim, no circle: the button's ring runs off the top of the
    frame. The depth under the fingers backs the aim — the button's face 11
    mm deeper than the aim's top, its lid ring at the aim's height."""
    import cv2

    from box_opening.runtime import approach
    from box_opening.runtime.kinematics import mat_to_quat_xyzw

    f = np.load(DEPTH_1433)
    color = cv2.imdecode(f["color_png"], cv2.IMREAD_COLOR)
    depth = f["depth_mm"].astype(np.float32) / 1000.0
    rot, cam = np.asarray(f["rot_cam"], float), np.asarray(f["trans_cam"], float)
    # the hand where the frame was shot: flatbot's OAK keeps the hand's axes, its lens LENS from the tool frame
    arm = _Arm()
    monkeypatch.setattr(approach, "_chain", lambda: arm)
    ctx.client.live = list(cam - rot @ np.asarray(LENS)) + list(mat_to_quat_xyzw(rot))
    pair = (list(LENS), EYE)
    aim = dp.aim_seen(ctx, f["aimed"], pair, float(f["up_mm"]))
    assert (aim.x, aim.y) == pytest.approx((0.0, -LENS[1]), abs=0.001)  # right under the fingers
    assert aim.lid_depth == pytest.approx(0.176, abs=0.002)
    s, why = dp.find_in_camera(color, depth, f["k"], pink, dist=f["dist"])
    assert s is not None and np.hypot(s.x - aim.x, s.y - aim.y) > dp.LOOK_EXPECT_M
    s, why = dp.find_in_camera(color, depth, f["k"], pink, dist=f["dist"], expect=dp.where_seen(aim))
    assert s is None and "ring around it is not lid" in why
    z = dp.depth_under_fingers(depth, f["k"], LENS)
    assert z == pytest.approx(0.187, abs=0.002)
    assert dp.aim_disagrees(ctx, aim, pair, z) is None


def test_from_over_the_box_the_fingers_are_moved_over_the_button_the_camera_sees(ctx, monkeypatch, tmp_path):
    """Flatbot 2026-10-22 20:04: TF's tilted aim put the press 45 mm off the
    button's centre, on the lid's edge. Straight down, the camera sees the
    button from its own axis, LENS[:2] beside the fingers: the hand moves by
    the difference, and looks again."""
    looks = [
        (_sighting(x=0.02, y=-LENS[1] + 0.03), 0.27, "2/2"),
        (_sighting(x=0.001), 0.27, "2/2"),
        (_sighting(up=12.0), None, "2/2"),
    ]
    result, events, arm, legs, start = _press(ctx, monkeypatch, tmp_path, looks)
    assert dp.exit_code(result) == 0
    assert events[:6] == [
        "depth:turn",
        "look:depth-look",
        "depth:centre",
        "look:depth-look",
        "depth:approach",
        "depth:touch",
    ]
    # the re-look follows a move with no at-rest read: it waits for still frames
    assert [t for t in arm.trace if t.startswith("look:")][:2] == ["look:depth-look (settled)", "look:depth-look"]
    assert "look:depth-after (settled)" in arm.trace
    goal, quat = arm.turns[0]
    R = rp._rot(quat)
    centre = np.asarray(goal) + R[:, 0] * 0.02 + R[:, 1] * 0.03  # along the hand's own x and y
    assert arm.lines[0] == pytest.approx(list(centre))
    assert legs["depth:centre"].guard.trip == "obstruction" and legs["depth:centre"].plan_free
    # the stroke then goes straight down from over the button
    assert arm.lines[1][:2] == pytest.approx(list(centre[:2]))


def test_each_look_after_a_move_or_the_press_is_told_where_the_button_must_be(ctx, monkeypatch, tmp_path):
    """The centring move goes level by what the camera asked, the hand's
    attitude held: the look after it is told where that button, unmoved,
    must be seen from there (2026-10-23 17:30 took a cabinet's circle). The
    after-look stands where the last look did: told where it saw it."""
    seen = _seen(hand_off=(0.02, 0.03))
    got = []

    def again(ctx_):
        out = seen(ctx_)
        got.append(out[0])
        return out

    looks = [seen, again, seen.after]
    result, events, arm, _l, _s = _press(ctx, monkeypatch, tmp_path, looks)
    assert dp.exit_code(result) == 0
    assert [e for e in events if e.startswith("look:")] == ["look:depth-look", "look:depth-look", "look:depth-after"]
    first, after_move, after_press = arm.expects
    assert first is None  # nothing is known before the first look
    assert after_move == pytest.approx((got[0].x, got[0].y, got[0].lid_depth), abs=1e-6)
    assert after_press == pytest.approx(dp.where_seen(got[0]), abs=1e-9)
    assert result.seen_at is not None  # ... and a look again after it, where the after-look saw it


def test_a_button_far_from_under_the_fingers_is_not_chased(ctx, monkeypatch, tmp_path):
    result, events, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, [(_sighting(y=-LENS[1] + 0.12), 0.27, "2/2")])
    assert result.free and "farther than a re-centre" in result.why and "depth:centre" not in events


def test_a_move_that_does_not_bring_the_button_under_the_fingers_stops(ctx, monkeypatch, tmp_path):
    looks = [(_sighting(x=0.03), 0.27, "2/2"), (_sighting(x=0.029), 0.27, "2/2")]
    result, events, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, looks)
    assert result.free and "did not bring it under" in result.why and "depth:touch" not in events


def test_the_touch_reports_where_the_mount_in_tf_puts_the_fingertips(ctx, monkeypatch, tmp_path):
    import json

    looks = [(_sighting(lid=0.27), 0.27, "2/2"), (_sighting(up=12.0), None, "2/2")]
    _press(ctx, monkeypatch, tmp_path, looks, touch_at=0.15)
    note = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl"))) if '"depth_touch"' in line][0]
    assert note["fingertips_ahead_of_lens_mm"] == pytest.approx(120.0)
    assert note["tf_says_mm"] == pytest.approx(1000 * (dp.DOWN_CONTACT_TOOL[2] - LENS[2]))


def test_a_hand_that_cannot_point_down_there_stops_with_nothing_else_flown(ctx, monkeypatch, tmp_path):
    result, events, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, [], refuse_turn=True)
    assert result.free and "no straight-down hand" in result.why and events == [] and dp.exit_code(result) == 1


def test_no_wrist_frame_at_all_presses_nothing(ctx, monkeypatch, tmp_path):
    result, events, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, [(None, None, "no frame")])
    assert result.free and "not pressing blind" in result.why and "depth:touch" not in events


def test_an_open_box_is_never_pressed_the_pads_feel_its_lid_beside_the_knob(ctx, monkeypatch, tmp_path):
    """17:27: the knob already 10 mm up, and the run stopped — pressing it
    would shut the box. Now nothing presses it: the closed pads go halfway
    across the lid's ring along the hand's x, feel the lid there, and come
    back up; the box is put at the button's centre, at the felt lid's height,
    the knob up — what the grip goes down to."""
    from box_opening.models.container import from_container

    result, events, arm, legs, start = _press(ctx, monkeypatch, tmp_path, [(_sighting(up=14.0), 0.27, "2/2")])
    assert events == [
        "depth:turn",
        "look:depth-look",
        "depth:beside",
        "depth:approach",
        "depth:touch",
        "depth:unload",
        "depth:rise",
    ]
    assert "press:push" not in events
    assert result.free and result.contact and result.popped and not result.felt and result.why is None
    assert dp.exit_code(result) == 0
    goal, quat = arm.turns[0]
    R = rp._rot(quat)
    d = dp.lid_touch_offset_m(ctx.model)
    beside = np.asarray(goal) + R[:, 0] * d  # along the hand's x: the pads' narrow side across the ring
    assert arm.lines[0] == pytest.approx(list(beside))
    assert legs["depth:beside"].guard.trip == "obstruction" and legs["depth:beside"].plan_free
    assert arm.lines[1][:2] == pytest.approx(list(beside[:2]))  # the touch straight down from there
    assert legs["depth:touch"].guard.trip == "touch"
    # the box: the button's centre — where the fingers stood before the move — at the lid's felt height
    lid_z = goal[2] - TOUCH_AT + dp.DOWN_CONTACT_TOOL[2] * R[2, 2]
    button = from_container(ctx.cpose, ctx.model.button_offset)
    assert button[:2] == pytest.approx(list(goal[:2]), abs=1e-6) and button[2] == pytest.approx(lid_z, abs=1e-6)
    assert result.touch_xyz == pytest.approx(tuple(button), abs=1e-4)
    # back up where the touch began, judged for a pull at the fingertip, as after a press
    assert arm.lines[-1][2] == pytest.approx(goal[2]) and legs["depth:rise"].guard.hang is not None


def test_the_aim_before_the_depth_press_saw_the_knob_up_and_that_is_enough(ctx, monkeypatch, tmp_path):
    """Either reading is enough: pressing an open box shuts it, while feeling
    the lid of a shut one costs a grip that closes on air and stops."""
    result, events, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, [(_sighting(up=1.0), 0.27, "2/2")], knob_up_mm=10.0)
    assert "depth:beside" in events and "press:push" not in events and result.popped
    result, events, _a, _l, _s = _press(
        ctx, monkeypatch, tmp_path, [(_sighting(up=1.0), 0.27, "2/2"), (_sighting(up=0.5), None, "2/2")], knob_up_mm=2.0
    )
    assert "depth:beside" not in events and "press:push" in events


def test_the_pads_feel_an_open_lid_halfway_across_its_ring_clear_of_the_knob_and_the_edge(pink):
    d = dp.lid_touch_offset_m(pink)
    assert d == pytest.approx(0.5 * (0.022 + 0.0525))  # Ankou: halfway between the button's edge and the lid's
    assert d - dp.PAD_HALF_M[0] - 0.022 >= dp.LID_TOUCH_CLEAR_M
    assert 0.0525 - np.hypot(d + dp.PAD_HALF_M[0], dp.PAD_HALF_M[1]) >= dp.LID_TOUCH_CLEAR_M
    # a lid with no ring round its button to speak of: not felt, not pressed
    narrow = SimpleNamespace(button_diameter_m=0.044, dims=(0.07, 0.07, 0.12), button_offset=(0.0, 0.0, 0.12))
    assert dp.lid_touch_offset_m(narrow) is None


def test_an_open_box_with_no_lid_ring_to_feel_is_not_pressed(ctx, monkeypatch, tmp_path):
    monkeypatch.setattr(dp, "lid_touch_offset_m", lambda model: None)
    result, events, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, [(_sighting(up=14.0), 0.27, "2/2")])
    assert result.free and "OPEN" in result.why and "depth:touch" not in events and "depth:beside" not in events


def test_no_contact_all_the_way_down_rises_and_says_so(ctx, monkeypatch, tmp_path):
    result, events, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, [(_sighting(), 0.27, "2/2")], touch_at=None)
    assert result.free and "no contact" in result.why and "press:push" not in events
    assert events[-2:] == ["depth:unload", "depth:rise"]


def test_a_contact_where_these_fingertips_cannot_be_is_not_the_lid(ctx, monkeypatch, tmp_path):
    # the lid 30 cm below the lens, met 3 cm down: the fingertips would be 27 cm in front of the lens
    result, events, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, [(_sighting(lid=0.30), None, "2/2")], touch_at=0.03)
    assert result.free and "not where" in result.why and "press:push" not in events


def test_a_push_the_torques_never_felt_bottom_out_is_not_a_press(ctx, monkeypatch, tmp_path):
    looks = [(_sighting(), 0.27, "2/2"), (_sighting(up=0.5), None, "2/2")]
    result, _e, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, looks, push="arrived")
    assert not result.felt and not result.popped and dp.exit_code(result) == 1


def test_a_box_the_bench_camera_sees_moved_is_moved_the_way_it_went(ctx, monkeypatch, tmp_path):
    """The bench D405 is turned 180 deg about its lens axis: a button seen
    12 mm further along the camera's x moved 12 mm along the hand's -x."""
    from box_opening.models.container import from_container

    bench = (0.0, 0.075, -0.125)
    looks = [(_sighting(y=bench[1]), 0.27, "2/2"), (_sighting(x=0.012, y=bench[1], up=12.0), None, "2/2")]
    result, _e, arm, _l, _s = _press(
        ctx,
        monkeypatch,
        tmp_path,
        looks,
        lens=bench,
        r_hc=RZ180,
        measured=bench,
        touch_at=0.27 - (dp.DOWN_CONTACT_TOOL[2] - bench[2]),
    )
    assert result.moved
    R = rp._rot(arm.turns[0][1])
    want = np.r_[BUTTON[0], BUTTON[1]] - R[:2, 0] * 0.012  # along the hand's -x
    assert from_container(ctx.cpose, ctx.model.button_offset)[:2] == pytest.approx(list(want))


def test_the_home_after_a_depth_press_that_stopped_says_why_first():
    """17:30: a depth press stopped over the box went home announcing "DEPTH
    PRESS done" and told why only after the home."""
    stopped = dp.DepthPress(free=True, why="moving over the button did not bring it under the fingers (21 -> 288 mm)")
    assert dp.home_why(stopped) == "STOP: " + stopped.why
    assert "MOVED" in dp.home_why(dp.DepthPress(free=True, contact=True, moved=True))
    assert dp.home_why(dp.DepthPress(free=True, contact=True, felt=True, popped=True)) == ""


def test_a_box_the_press_moved_stops_the_run(ctx, monkeypatch, tmp_path):
    looks = [(_sighting(), 0.27, "2/2"), (_sighting(x=0.012, up=12.0), None, "2/2")]
    result, _e, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, looks)
    assert result.moved and dp.exit_code(result) == 1


ARRIVED = ("arrived", {"message": "ok", "progress": 1.0})
TURNED_DOWN = [1.0, 0.0, 0.0, 0.0]  # tool_frame's z along the world's -z (x, y, z, w): test_reach_press.STRAIGHT_DOWN


def _turns_down(chain, model, q, contact_tool, shorts=dp.TURN_SHORT_DEG):
    """dp.turn_down_path for test_reach_press's fake chain, whose walks all
    refuse: the turn after the slide flies, exactly down."""
    return [list(q), [float(v) + 0.01 for v in q]], list(TURNED_DOWN), 0.0, None


def _depth_flow(ctx, monkeypatch, tmp_path, press, compliant=False, walk_in=False, turn=_turns_down, **kw):
    """test_reach_press._flow with --depth-press: `press` stands in for
    press_down_from_here (called with the reach's ctx) and, in a compliant
    run, for the hold at the end of its one-execution way in
    (dp.compliant_hold: `press` is the hold's verdict); the stroke before
    it is compliant_stroke's on the fake chain. The fake chain's fk stands
    at z 0.45 over the aimed button at 0.40 (`chain`, a kw for _flow, may
    be another). A `compliant` run's press is proven before the arm moves
    (rp.way_down): the fake chain's walks refuse — unless `walk_in`, then
    they fly — and the turn after the slide is `turn`'s
    (dp.turn_down_path)."""
    from test_reach_press import _Chain, _flow

    if compliant:
        monkeypatch.setattr(dp, "turn_down_path", turn)
    if walk_in:
        monkeypatch.setattr(_Chain, "walk", lambda self, q0, xyz, quat, **k: ([list(q0), [v + 0.01 for v in q0]], None))
    got = {}
    compliant_run = compliant

    def fake(
        node,
        ctx_,
        cfg,
        runner,
        watcher,
        contact_tool,
        here=None,
        turned=None,
        knob_up_mm=None,
        aimed=None,
        compliant=False,
        top_z=None,
    ):
        got.update(
            contact=contact_tool,
            knob_up_mm=knob_up_mm,
            aimed=aimed,
            compliant=compliant,
            top_z=top_z,
            client_compliant=getattr(ctx_.client, "compliant", False),
        )
        return press(ctx_)

    def hold(ctx_, cfg, runner, stroke, top_z, seen):
        got.update(top_z=top_z, stroke=stroke, client_compliant=getattr(ctx_.client, "compliant", False))
        return press(ctx_)

    monkeypatch.setattr(dp, "press_down_from_here", fake)
    if compliant:
        monkeypatch.setattr(dp, "compliant_hold", hold)
    real = rp.run_reach_press

    def with_flag(node, ctx_, cfg, runner, watcher, args, scene_fix, locator):
        return real(
            node,
            ctx_,
            cfg,
            runner,
            watcher,
            SimpleNamespace(execute=True, press_only=True, depth_press=True, compliant=compliant_run),
            scene_fix,
            locator,
        )

    monkeypatch.setattr(rp, "run_reach_press", with_flag)
    kw.setdefault("touch", ARRIVED)  # no TF stroke flies: the 4th motion is the way up or home
    code, rec, chain, home = _flow(ctx, monkeypatch, tmp_path, **kw)
    return code, rec, chain, home, got


def _popped(ctx_):
    return dp.DepthPress(free=True, contact=True, felt=True, popped=True)


def _lost(ctx_):
    """Touched and pushed, then the wrist camera did not find the box again."""
    return dp.DepthPress(free=False, contact=True, felt=True, lost=True, why="the wrist camera does not find it")


def _measured(ctx_, z=0.30):
    """The depth press as it leaves ctx: the box put where the fingertips met it."""
    from box_opening.models.container import from_container

    b = from_container(ctx_.cpose, ctx_.model.button_offset)
    ctx_.cpose = rp.cpose_for_button((b[0], b[1], z), ctx_.cpose.yaw, ctx_.model.button_offset)
    return dp.DepthPress(free=True, contact=True, felt=True, popped=True, touch_xyz=(b[0], b[1], z))


def test_the_reach_out_hands_over_to_the_depth_press_once_over_the_box(ctx, monkeypatch, tmp_path):
    """After the slide in, the TF stroke is not flown: the depth press is.
    Standing within HOME_CLEAR_M of the box's padded top, the hand goes
    straight up clear of it before home is planned — once."""
    import test_reach_press as trp

    from box_opening.worlds import planned_top_z

    code, rec, chain, _h, got = _depth_flow(ctx, monkeypatch, tmp_path, _popped)
    assert code == 0 and got["contact"] == rp.CONTACT_TOOL
    # the look's aim goes with it: where the button is, and its knob
    assert got["aimed"] == pytest.approx(trp.BUTTON) and got["knob_up_mm"] == pytest.approx(0.4)
    assert rec.events == ["reach:look", "aim:reach-aim", "reach:re-aim", "reach:slide-in", "depth:clear", "home"]
    assert "press:down" not in rec.events
    up = [xyz for xyz, _q in chain.lines][-1]
    assert up[:2] == pytest.approx([0.5, -0.4]) and up[2] == pytest.approx(
        planned_top_z(ctx.model, ctx.cpose) + rp.HOME_CLEAR_M
    )
    g = rec.legs["depth:clear"].guard
    assert (
        g.trip == "obstruction"
        and g.touch_nm == rp.DRAG_NM
        and g.hang is not None
        and rec.legs["depth:clear"].plan_free
    )


# ---- CENTRE on a mount that is known, in world xy ----------------------------------


def test_a_mount_in_tf_that_disagrees_with_the_measured_lens_is_not_centred_on(ctx, monkeypatch, tmp_path):
    """Flatbot 2026-10-22 20:04: TF's hand-measured mount put the lens 100 mm
    beside the fingers, the truth ~65. CENTRE on it 'converges' with the
    press 35 mm off, on the lid's edge — it stops with nothing pressed."""
    tf_dev = (0.0, 0.100, -0.22)
    result, events, arm, _l, _s = _press(ctx, monkeypatch, tmp_path, [_seen((0.0, 0.0))], lens=tf_dev)
    assert result.free and "mount" in result.why and "+100" in result.why and "+65" in result.why
    assert events == ["depth:turn", "look:depth-look"] and arm.lines == [] and dp.exit_code(result) == 1


def test_no_measured_lens_in_the_camera_config_is_not_centred_on(ctx, monkeypatch, tmp_path):
    result, events, arm, _l, _s = _press(ctx, monkeypatch, tmp_path, [_seen((0.02, 0.0))], measured=None)
    assert result.free and "wrist_lens_in_hand_m" in result.why
    assert events == ["depth:turn", "look:depth-look"] and arm.lines == []


def test_a_mount_in_tf_within_5_mm_of_the_measured_lens_centres(ctx, monkeypatch, tmp_path):
    tf = (0.002, LENS[1] + 0.003, LENS[2] + 0.01)
    seen = _seen((0.02, 0.03), lens=tf)
    looks = [seen, seen, seen.after]
    result, events, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, looks, lens=tf)
    assert dp.exit_code(result) == 0
    assert events[:6] == [
        "depth:turn",
        "look:depth-look",
        "depth:centre",
        "look:depth-look",
        "depth:approach",
        "depth:touch",
    ]


def test_a_turn_short_of_straight_down_centres_the_vertical_stroke_not_the_tool_axis(ctx, monkeypatch, tmp_path):
    """Exactly down sits on the wrist's singularity, the turn stops 5 deg
    short. The button on the tool axis 277 mm below the lens is 14 mm off the
    fingertip's vertical: CENTRE moves it under the vertical stroke, level."""
    seen = _seen((0.0, 0.0), depth=0.277)
    looks = [seen, seen, seen.after]
    result, events, arm, legs, _s = _press(
        ctx, monkeypatch, tmp_path, looks, refuse_first=2, touch_at=0.277 - (dp.DOWN_CONTACT_TOOL[2] - LENS[2])
    )
    goal, quat = arm.turns[0]
    R = rp._rot(quat)
    assert np.degrees(np.arccos(-R[2, 2])) == pytest.approx(5.0, abs=0.01)
    assert dp.exit_code(result) == 0 and "depth:centre" in events
    b = seen.world["b"]
    tip = np.asarray(arm.lines[0]) + R @ np.asarray(dp.DOWN_CONTACT_TOOL)
    assert np.hypot(*(tip[:2] - b[:2])) <= dp.CENTRE_TOL_M
    assert np.hypot(*(np.asarray(goal[:2]) - arm.lines[0][:2])) == pytest.approx(0.014, abs=0.001)
    # level: every waypoint of the centring move at the height it began
    for p in legs["depth:centre"].traj.points:
        assert p.positions[2] == pytest.approx(goal[2], abs=0.001)
    # and the stroke goes straight down from there
    assert arm.lines[1][:2] == pytest.approx(arm.lines[0][:2])


def test_the_bench_cameras_turn_in_the_hand_is_kept_when_centring(ctx, monkeypatch, tmp_path):
    """The bench D405, 180 deg about its lens axis: with the button under the
    fingers there is no move; 30 mm off in the hand's +y, the move goes 30 mm
    toward it — not 100 mm away."""
    bench = (0.0, 0.075, -0.125)
    seen = _seen((0.0, 0.0), lens=bench, r_hc=RZ180)
    looks = [seen, seen.after]
    result, events, arm, _l, start = _press(
        ctx,
        monkeypatch,
        tmp_path,
        looks,
        tilt=0,
        lens=bench,
        r_hc=RZ180,
        measured=bench,
        touch_at=0.27 - (dp.DOWN_CONTACT_TOOL[2] - bench[2]),
    )
    assert dp.exit_code(result) == 0 and "depth:centre" not in events
    seen = _seen((0.0, 0.03), lens=bench, r_hc=RZ180)
    looks = [seen, seen, seen.after]
    result, events, arm, _l, start = _press(
        ctx,
        monkeypatch,
        tmp_path,
        looks,
        tilt=0,
        lens=bench,
        r_hc=RZ180,
        measured=bench,
        touch_at=0.27 - (dp.DOWN_CONTACT_TOOL[2] - bench[2]),
    )
    assert dp.exit_code(result) == 0 and "depth:centre" in events
    R = rp._rot(rp.reach_quat(ctx.model, BUTTON[:2], 0))
    want = np.asarray(start[:3]) + R @ np.r_[0.0, 0.03, 0.0]
    assert arm.lines[0][:2] == pytest.approx(list(want[:2]), abs=0.001)


def test_a_touch_far_from_where_the_mount_puts_the_fingertips_pushes_nothing(ctx, monkeypatch, tmp_path):
    ahead_tf = dp.DOWN_CONTACT_TOOL[2] - LENS[2]
    looks = [(_sighting(lid=0.27), 0.27, "2/2")]
    result, events, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, looks, touch_at=0.27 - (ahead_tf - 0.05))
    assert result.free and result.contact and "not pushing" in result.why and "press:push" not in events
    assert events[-2:] == ["depth:unload", "depth:rise"]
    # 12 mm (#128's mount against the 20:04 touch): pushed
    looks = [(_sighting(lid=0.27), 0.27, "2/2"), (_sighting(up=12.0), None, "2/2")]
    result, events, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, looks, touch_at=0.27 - (ahead_tf + 0.012))
    assert "press:push" in events and dp.exit_code(result) == 0


def test_a_box_the_touch_put_lower_needs_no_line_up_before_home(ctx, monkeypatch, tmp_path):
    """20:04: the touch found the lid 76 mm under TF's. With the box put
    there, the hand at the stroke's start is clear of it and home plans."""
    code, rec, _c, home, _g = _depth_flow(ctx, monkeypatch, tmp_path, _measured)
    assert code == 0 and rec.events[-2:] == ["reach:slide-in", "home"]
    from box_opening.models.container import from_container

    assert from_container(home["cpose"], ctx.model.button_offset)[2] == pytest.approx(0.30)  # home planned around it


@pytest.mark.parametrize("compliant", [False, True])
def test_a_line_up_that_does_not_fly_falls_back_to_the_back_off_and_logs_each_refusal(
    ctx, monkeypatch, tmp_path, capsys, compliant
):
    """A compliant run's clear goes up and back toward the arm
    (rp.clear_targets): where neither of its lines flies, the same back-off
    as a guarded run's, unguarded, and at a guarded run's FREE_LINE_SPEED,
    not a compliant run's faster free-air lines': it flies where the planner
    refused the start, the arm perhaps inside the box's padded volume. The
    back-off says what the planner refused, not always "inside the box's
    padded volume"."""
    import json

    import test_reach_press as trp
    from test_reach_press import _Chain

    from box_opening.constants import FREE_LINE_SPEED
    from box_opening.worlds import planned_top_z

    real = _Chain.straight_line

    def no_way_up(self, q0, xyz, quat):
        if xyz[2] > 0.49:  # the way up (0.50, 0.505 compliant) and the move to the look (0.54: planned): a joint limit
            self.lines.append((list(xyz), list(quat)))
            return None, "joint_6 limit"
        return real(self, q0, xyz, quat)

    monkeypatch.setattr(_Chain, "straight_line", no_way_up)
    if compliant:
        monkeypatch.setattr(_Chain, "R", trp.STRAIGHT_DOWN)
    code, rec, chain, _h, _g = _depth_flow(ctx, monkeypatch, tmp_path, _popped, compliant=compliant, home_refusals=2)
    assert code == 1 and rec.events[-1] == "reach:back-off" and "depth:clear" not in rec.events
    rows = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl")))]
    refused = [r for r in rows if r["leg"] == "home_refused"]
    assert len(refused) == 2 and all("INVALID_START" in r["why"] for r in refused)
    (clear,) = (r for r in rows if r["leg"] == "depth_clear")
    assert clear["flown"] is False and "joint_6 limit" in clear["why"]
    assert (rec.legs["reach:back-off"].guard is None) is compliant
    assert rec.legs["reach:back-off"].speed == FREE_LINE_SPEED
    ups = [xyz for xyz, _q in chain.lines if xyz[2] > 0.49]  # the clear's last, after the refused move to the look
    top = planned_top_z(ctx.model, ctx.cpose)
    if compliant:
        # the hand straight down: CLEAR_EXTRA_M over the guarded run's height, 50 then 150 mm back toward the arm
        u = np.asarray(ctx.cpose.xyz[:2]) / np.hypot(*ctx.cpose.xyz[:2])
        assert ups[-2:] == [
            pytest.approx([*(np.array([0.5, -0.4]) - b * u), top + rp.HOME_CLEAR_M + rp.CLEAR_EXTRA_M])
            for b in (rp.CLEAR_BACK_M, rp.REACH_PULLBACK_M)
        ]
    else:
        assert ups[-1:] == [pytest.approx([0.5, -0.4, top + rp.HOME_CLEAR_M])]  # straight up, as before
    said = capsys.readouterr().out
    assert "the planner refused home from here (MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION" in said
    assert "inside the box's padded volume" not in said


def test_a_depth_press_whose_home_failed_never_exits_0(ctx, monkeypatch, tmp_path, capsys):
    """The knob popped, but the arm did not get home: it holds over the box."""
    monkeypatch.setattr(rp, "_go_home", lambda *a, **k: False)
    code, _r, _c, _h, _g = _depth_flow(ctx, monkeypatch, tmp_path, _measured)
    assert code == 1 and "did not get home" in capsys.readouterr().out


def test_a_box_no_camera_finds_after_the_press_holds_the_arm(ctx, monkeypatch, tmp_path, capsys):
    code, rec, _c, _h, _g = _depth_flow(ctx, monkeypatch, tmp_path, _lost, scene_after="none")
    assert code == 1 and "home" not in rec.events and rec.events[-1] == "scene"
    assert "neither camera" in capsys.readouterr().out


def test_a_box_only_the_scene_camera_finds_after_the_press_goes_home_around_it(ctx, monkeypatch, tmp_path):
    from box_opening.models.container import from_container

    code, rec, _c, home, _g = _depth_flow(ctx, monkeypatch, tmp_path, _lost)
    assert code == 1  # the knob was not judged
    assert rec.events.count("scene") == 1 and rec.events[-1] == "home"
    assert from_container(home["cpose"], ctx.model.button_offset) == pytest.approx([0.66, -0.57, 0.40])
    # the scene camera saw it 50 mm away: home is planned around the box shifted by that
    code, rec, _c, home, _g = _depth_flow(ctx, monkeypatch, tmp_path / "moved", _lost, scene_after="moved")
    assert code == 1 and rec.events[-1] == "home"
    assert from_container(home["cpose"], ctx.model.button_offset) == pytest.approx([0.71, -0.57, 0.40])


def test_a_box_the_depth_press_moved_is_a_stop_before_the_arm_goes_home(ctx, monkeypatch, tmp_path, capsys):
    def moved(ctx_):
        return dp.DepthPress(free=True, contact=True, felt=True, popped=True, moved=True)

    code, rec, _c, _h, _g = _depth_flow(ctx, monkeypatch, tmp_path, moved)
    out = capsys.readouterr().out
    assert code == 1 and rec.events[-1] == "home"
    assert out.index("STOP: the press MOVED the box") < out.index("\nhome ")


def test_the_mission_record_keeps_what_the_reach_and_the_depth_press_found(ctx, monkeypatch, tmp_path):
    import json

    from box_opening.detection_set import MissionFrames

    ctx.mission_frames = MissionFrames(tmp_path / "missions", "ankou_pink.yaml", 0.2933, now=0)
    code, _r, _c, _h, _g = _depth_flow(ctx, monkeypatch, tmp_path, _measured)
    assert code == 0
    ctx.mission_frames.frames.append(("scene_0", {"x": np.zeros(1)}))
    doc = json.loads((ctx.mission_frames.close(exit_code=code) / "truth.json").read_text())
    # the aim through TF is a fix, never trusted truth; the arm's own touch is kept beside it
    assert doc["truth_source"] == "fix" and doc["truth_xyz"] == pytest.approx([0.66, -0.57, 0.40])
    assert (
        doc["stage"] == "pressed" and doc["popped"] is True and doc["touch_xyz"] == pytest.approx([0.66, -0.57, 0.30])
    )
    assert doc["table_z"] == pytest.approx(0.2933)


# ---- what the depth press leaves behind ---------------------------------------


def test_the_touch_puts_the_box_where_the_fingertips_met_it(ctx, monkeypatch, tmp_path):
    """TF's lid at 0.40 (BUTTON 0.24 here); the fingertips met it 240 mm
    down from 0.46, where the wrist camera saw the lid (so the touch passes
    the mount check): every world after the touch holds the box there, and
    the table is left as the scene camera measured it."""
    from box_opening.models.container import from_container

    ctx.model = ContainerModel.load(PINK)
    ctx.cpose = rp.cpose_for_button(BUTTON, 0.3, ctx.model.button_offset)
    ctx.worlds.table_top_z = 0.10
    lid = 0.24 + (dp.DOWN_CONTACT_TOOL[2] - LENS[2])
    looks = [(_sighting(lid=lid), lid, "2/2"), (_sighting(up=12.0), None, "2/2")]
    result, _e, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, looks, touch_at=0.24)
    assert result.popped and result.free
    assert from_container(ctx.cpose, ctx.model.button_offset) == pytest.approx([BUTTON[0], BUTTON[1], 0.22])
    assert ctx.cpose.yaw == 0.3 and result.touch_xyz == pytest.approx((BUTTON[0], BUTTON[1], 0.22))
    assert ctx.worlds.table_top_z == 0.10
    rise = [kw for kind, kw in ctx.worlds.pushes if kind == "full"][-1]  # the rise's world
    assert from_container(rise["cpose"], ctx.model.button_offset)[2] == pytest.approx(0.22)
    assert result.ref is not None  # the free-air reading the way home is guarded against


def test_a_box_the_wrist_camera_loses_after_a_contact_holds_the_arm(ctx, monkeypatch, tmp_path):
    looks = [(_sighting(), 0.27, "2/2"), (None, None, "the button in none")]
    result, events, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, looks)
    assert result.contact and result.felt and result.lost and not result.free
    assert dp.exit_code(result) == 1 and events[-1] == "look:depth-after"


def test_a_box_the_press_moved_is_moved_in_the_world_by_what_the_camera_saw(ctx, monkeypatch, tmp_path):
    from box_opening.models.container import from_container

    looks = [(_sighting(), 0.27, "2/2"), (_sighting(x=0.012, up=12.0), None, "2/2")]
    result, _e, arm, _l, _s = _press(ctx, monkeypatch, tmp_path, looks)
    assert result.moved
    R = rp._rot(arm.turns[0][1])
    want = np.r_[BUTTON[0], BUTTON[1]] + R[:2, 0] * 0.012  # along the hand's own x
    assert from_container(ctx.cpose, ctx.model.button_offset)[:2] == pytest.approx(list(want))


def test_the_scene_camera_is_asked_only_when_the_wrist_lost_the_box(ctx, tmp_path):
    from conftest import runner as make_runner

    from box_opening.models.container import ContainerPose

    r = make_runner(ctx.client, tmp_path)
    fix = SimpleNamespace(top_xyz=(0.6, -0.2, 0.4))
    asked = []
    loc = SimpleNamespace(locate=lambda s: asked.append(s) or SimpleNamespace(top_xyz=(0.6, -0.2, 0.4)))
    seen = dp.DepthPress(free=True, contact=True, popped=True)
    assert dp.after_scene_check(ctx, r, seen, fix, loc) is seen and asked == []
    lost = dp.DepthPress(free=False, contact=True, lost=True, why="x")
    assert dp.after_scene_check(ctx, r, lost, fix, None) is lost  # no scene camera: the arm holds
    ctx.cpose = ContainerPose(xyz=(0.6, -0.2, 0.28), yaw=0.0)
    out = dp.after_scene_check(ctx, r, lost, fix, loc)
    assert out.free and not out.moved and len(asked) == 1 and ctx.cpose.xyz == (0.6, -0.2, 0.28)
    tipped = SimpleNamespace(locate=lambda s: SimpleNamespace(top_xyz=(0.6, -0.2, 0.37)))
    out = dp.after_scene_check(ctx, r, lost, fix, tipped)
    assert out.free and out.moved


def test_an_arm_that_did_not_get_home_is_never_a_success(ctx, capsys):
    popped = dp.DepthPress(free=True, contact=True, felt=True, popped=True)
    assert dp.finish(ctx, popped, True) == 0
    assert dp.finish(ctx, popped, False) == 1 and "did not get home" in capsys.readouterr().out
    held = dp.DepthPress(free=False, contact=True, why="the way up stopped")
    assert dp.finish(ctx, held, False) == 1 and "did not get home" not in capsys.readouterr().out


def test_the_reach_out_slides_in_turning_down_in_one_leg(ctx, monkeypatch, tmp_path):
    """Run 20:04: the slide in (4.56 s), two at-rest reads and the turn
    (7.43 s) — one walk does both, as slow and as guarded as the slide."""
    import test_reach_press as trp

    from box_opening.constants import CONTACT_SPEED

    got = {}

    def press(
        node,
        ctx_,
        cfg,
        runner,
        watcher,
        contact_tool,
        here=None,
        turned=None,
        knob_up_mm=None,
        aimed=None,
        compliant=False,
        top_z=None,
    ):
        got.update(here=here, turned=turned)
        # as the real press does: the free-air reading it was handed is the one home is judged against
        return dp.DepthPress(free=True, contact=True, felt=True, popped=True, ref=here)

    monkeypatch.setattr(dp, "press_down_from_here", press)
    walks = []

    def walk(self, q0, xyz, quat, step_m=None, step_rad=None, max_step_rad=None):
        walks.append((list(xyz), list(quat), step_m, step_rad, max_step_rad))
        return [list(q0), [v + 0.01 for v in q0]], None

    monkeypatch.setattr(trp._Chain, "walk", walk, raising=False)
    real = rp.run_reach_press
    monkeypatch.setattr(
        rp,
        "run_reach_press",
        lambda n, c, cfg, r, w, a, f, loc: real(
            n, c, cfg, r, w, SimpleNamespace(execute=True, press_only=True, depth_press=True), f, loc
        ),
    )
    code, rec, chain, _h = trp._flow(ctx, monkeypatch, tmp_path, touch=ARRIVED)
    assert code == 0 and rec.events == [
        "reach:look",
        "aim:reach-aim",
        "reach:re-aim",
        "reach:slide-in",
        "depth:clear",
        "home",
    ]
    leg = rec.legs["reach:slide-in"]
    ((goal, down, step_m, step_rad, jump),) = walks
    assert leg.speed == CONTACT_SPEED and leg.guard.reference == trp.AT_LOOK2 and leg.guard.touch_nm == rp.DRAG_NM
    assert leg.target[2] == pytest.approx(down) and got["turned"] == pytest.approx(down)
    assert (step_m, step_rad, jump) == (dp.TURN_STEP_M, dp.TURN_STEP_RAD, dp.TURN_MAX_JUMP_RAD)
    assert rp._rot(down)[2, 2] == pytest.approx(-1.0)  # exactly down, the first of TURN_SHORT_DEG
    # the pads end where the tilted fingertip would have ended the slide
    quat = rp.reach_quat(ctx.model, trp.BUTTON[:2], rp.REACH_TILTS_DEG[0])
    start = rp.tool_for_contact((trp.BUTTON[0], trp.BUTTON[1], trp.BUTTON[2] + rp.REACH_START_M), quat)
    tip = np.asarray(start) + rp._rot(quat) @ np.asarray(rp.CONTACT_TOOL)
    assert np.asarray(goal) + rp._rot(down) @ np.asarray(dp.DOWN_CONTACT_TOOL) == pytest.approx(tip)
    # one at-rest read after it, handed over: the depth press reads nothing again
    assert got["here"] == trp.AT_START and rec.reads == 4
    # the slide is solved by plan_reach and reach_from_here, to prove it flies, and not again from the
    # live joints; the stroke, which the depth press never flies, only by plan_reach
    assert sum(xyz == pytest.approx(start) for xyz, _q in chain.lines) == 2
    bottom = rp.tool_for_contact((trp.BUTTON[0], trp.BUTTON[1], trp.BUTTON[2] - load_press_demo(PINK).travel_m), quat)
    assert sum(xyz == pytest.approx(bottom) for xyz, _q in chain.lines) == 1


@pytest.mark.parametrize("compliant", [False, True])
def test_the_depth_press_checks_the_aim_as_the_wrist_measured_it(ctx, monkeypatch, tmp_path, compliant):
    """Flatbot 2026-10-27 15:19: over the box the guarded depth press found
    the aim 175.9 mm below the lens and the wrist's own depth under the
    fingers 190.0 — 14 mm, inside LOOK_EXPECT_DEPTH_M. The aim's press
    point raised by the wrist's 21.6 mm table offset (ButtonAimer.fix) would
    read 36 mm off that same depth and be refused. So the guarded depth
    press gets the aim as the wrist measured it, while the box's world and
    the press's top, guarded or compliant, go by the raised one."""
    import test_reach_press as trp

    from box_opening.models.container import from_container

    measured = (trp.BUTTON[0], trp.BUTTON[1], trp.BUTTON[2] - 0.0216)
    code, _rec, _chain, _h, got = _depth_flow(
        ctx, monkeypatch, tmp_path, _popped, compliant=compliant, first=(trp.BUTTON, 0.4, measured)
    )
    assert code == 0 and got["top_z"] == pytest.approx(trp.BUTTON[2])
    if not compliant:  # a compliant press has no look and no guard to judge the aim against
        assert got["compliant"] is False and got["aimed"] == pytest.approx(measured)
    kw = [kw for kind, kw in ctx.worlds.pushes if kind == "interaction" and kw.get("tag") == "reach"][-1]
    assert from_container(kw["cpose"], ctx.model.button_offset) == pytest.approx(list(trp.BUTTON))


def test_where_no_walk_turns_down_on_the_way_it_slides_then_turns(ctx, monkeypatch, tmp_path):
    import test_reach_press as trp

    got = {}

    def press(
        node,
        ctx_,
        cfg,
        runner,
        watcher,
        contact_tool,
        here=None,
        turned=None,
        knob_up_mm=None,
        aimed=None,
        compliant=False,
        top_z=None,
    ):
        got.update(here=here, turned=turned)
        return dp.DepthPress(free=True, contact=True, felt=True, popped=True)

    monkeypatch.setattr(dp, "press_down_from_here", press)
    tried = []
    monkeypatch.setattr(
        trp._Chain,
        "walk",
        lambda self, q0, xyz, quat, **k: (tried.append(quat), (None, "a joint limit"))[1],
        raising=False,
    )
    real = rp.run_reach_press
    monkeypatch.setattr(
        rp,
        "run_reach_press",
        lambda n, c, cfg, r, w, a, f, loc: real(
            n, c, cfg, r, w, SimpleNamespace(execute=True, press_only=True, depth_press=True), f, loc
        ),
    )
    code, rec, _c, _h = trp._flow(ctx, monkeypatch, tmp_path, touch=ARRIVED)
    assert code == 0 and len(tried) == len(dp.TURN_SHORT_DEG)
    quat = rp.reach_quat(ctx.model, trp.BUTTON[:2], rp.REACH_TILTS_DEG[0])
    assert rec.legs["reach:slide-in"].target[2] == pytest.approx(quat)  # the tilted slide: the depth press turns
    assert got["turned"] is None and got["here"] == trp.AT_START


def test_the_walk_in_on_flatbot_turns_down_over_the_button_no_lower_than_the_slide(ctx, monkeypatch):
    """Flatbot's URDF, run 20:04's aimed button at 35 deg (the walk
    modelled before it was flown: 31 points, the fingertip within 0.6 mm of
    the slide's height)."""
    from box_opening.constants import HOME
    from box_opening.runtime import approach
    from box_opening.runtime.kinematics import ArmChain

    chain = ArmChain(path="capabilities/box_opening/test/data/arm_flatbot_20261021.urdf", root="arm_mount")
    model = ContainerModel.load(PINK)
    ctx.model = model
    button, tilt = (0.4833, -0.1575, 0.4044), 35
    quat, start, _bottom, look = rp.reach_geometry(model, button, tilt, 0.015)
    q_look = chain.nearest_solution(HOME, look, quat)
    assert q_look is not None
    ctx.client.live = [float(v) for v in q_look]
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    pts, down, why = dp.walk_in_down(ctx, start, quat)
    assert why is None, why
    tip = np.asarray(start) + rp._rot(quat) @ np.asarray(rp.CONTACT_TOOL)  # the slid-in fingertip
    R, t = chain.fk(pts[-1])
    R = np.asarray(R, float)
    lean = np.degrees(np.arccos(np.clip(-R[2, 2], -1.0, 1.0)))
    assert lean < 1.0 + max(dp.TURN_SHORT_DEG)
    assert np.linalg.norm(np.asarray(t) + R @ np.asarray(dp.DOWN_CONTACT_TOOL) - tip) < 0.001
    lowest = []
    for q in pts:
        Rq, tq = chain.fk(q)
        Rq = np.asarray(Rq, float)
        lowest.append(
            min(float((np.asarray(tq) + Rq @ np.asarray(p))[2]) for p in (rp.CONTACT_TOOL, dp.DOWN_CONTACT_TOOL))
        )
    assert min(lowest) >= tip[2] - 0.001  # never nearer the lid than the slide
    assert (
        max(float(np.abs(np.asarray(b) - np.asarray(a)).max()) for a, b in zip(pts, pts[1:], strict=False))
        <= dp.TURN_MAX_JUMP_RAD
    )


def test_a_look_straight_after_an_at_rest_read_takes_frames_at_once(monkeypatch):
    """look() waits LOOK_SETTLE_S for still frames; after an at-rest read
    the arm has been still 0.45 s already. Either way a frame shot before
    `after` never counts."""
    import time as _time

    from box_opening.tasks import press_demo

    monkeypatch.setattr(dp, "LOOK_SETTLE_S", 0.5)
    monkeypatch.setattr(press_demo, "save_aim_frame", lambda *a, **k: None)
    clock = {"ns": 1000}
    grab = SimpleNamespace(
        color=np.zeros((4, 4, 3)), depth=np.zeros((4, 4)), k=np.eye(3), color_stamp=SimpleNamespace(sec=0, nanosec=1000)
    )  # shot before the look began
    node = SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=clock["ns"])))

    def spin(_node):
        clock["ns"] += 10
        if clock["ns"] % 1000 == 0:  # a new frame now and then, stamped now
            grab.color_stamp = SimpleNamespace(sec=0, nanosec=clock["ns"])

    monkeypatch.setattr(press_demo, "_spin_detect", spin)
    # the stale frame sees the button elsewhere: counted, it would not agree
    monkeypatch.setattr(
        dp,
        "find_in_camera",
        lambda c, d, k, m, dist=None, expect=None: (
            _sighting(x=0.05 if grab.color_stamp.nanosec == 1000 else 0.0),
            None,
        ),
    )
    for settled in (True, False):
        clock["ns"], grab.color_stamp = 1000, SimpleNamespace(sec=0, nanosec=1000)
        t0 = _time.monotonic()
        s, _z, _lens, status = dp.look(node, SimpleNamespace(grab=grab), None, "t", settled=settled)
        took = _time.monotonic() - t0
        assert s is not None and status == "2/2 frames", (settled, status)
        assert (took < dp.LOOK_SETTLE_S) if settled else (took >= dp.LOOK_SETTLE_S), (settled, took)


def test_whatever_the_fast_stretch_meets_is_not_pushed(ctx, monkeypatch, tmp_path):
    """The fast stretch is a guarded touch too, and ends TOUCH_SLOW_M short
    of the lid — more than AHEAD_TOL_M, so every contact the press accepts
    is met slowly. Something met on the fast stretch stands too far above
    the lid: the slow stretch is not flown, nothing is pushed."""
    assert dp.TOUCH_SLOW_M > dp.AHEAD_TOL_M
    looks = [(_sighting(), 0.27, "2/2"), (_sighting(up=12.0), 0.255, "2/2")]
    result, events, arm, legs, start = _press(ctx, monkeypatch, tmp_path, looks, touch_at=0.08)
    assert "depth:approach" in events and "depth:touch" not in events and "press:push" not in events
    assert result.free and "not pushing" in result.why


def test_at_rest_the_lens_is_placed_even_when_tf_lags_the_newest_frame(monkeypatch):
    """Flatbot 2026-10-23 16:51: right over the box, both of the depth look's
    frames were newer than the newest arm TF, the lookup at their stamps
    failed, and the press stopped "no camera pose for the wrist's frames".
    The arm had not moved since the look began, so the camera's pose a
    little earlier — never before the arm stopped — places the lens."""
    from rclpy.time import Time

    from box_opening.runtime import approach

    stamp_ns = 1_000_000_000_000
    latest_ns = stamp_ns - 40_000_000  # TF is 40 ms behind the frame

    class Buffer:
        def lookup_transform(self, root, parent, when):
            if when.nanoseconds > latest_ns:
                raise RuntimeError("extrapolation into the future")
            return SimpleNamespace(
                transform=SimpleNamespace(
                    translation=SimpleNamespace(x=0.0, y=0.065, z=-0.105),
                    rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                )
            )

    grab = SimpleNamespace(
        tf_buffer=Buffer(),
        parent="wrist_camera_color_optical_frame",
        _base_frame="arm_mount",
        mount_xyz=[0.0, 0.0, 0.0],
        mount_quat=[0.0, 0.0, 0.0, 1.0],
        stamp_offset_s=0.0,
        color_stamp=Time(nanoseconds=stamp_ns).to_msg(),
    )
    monkeypatch.setattr(approach, "_chain", lambda: SimpleNamespace(fk=lambda q: (np.eye(3), np.zeros(3))))
    ctx = SimpleNamespace(client=SimpleNamespace(joints=lambda: [0.0] * 7))
    assert dp.lens_in_hand(ctx, grab) is None  # at the frame's own stamp: TF has nothing yet
    lens, r_hc = dp.lens_in_hand(ctx, grab, still_since_ns=stamp_ns - 1_000_000_000)
    assert lens == pytest.approx([0.0, 0.065, -0.105])
    # ... but never from before the arm stopped
    assert dp.lens_in_hand(ctx, grab, still_since_ns=stamp_ns - 20_000_000) is None


# ---- the compliant press: impedance, one stroke, nothing it reads stops it --------

COMPLIANT_TOP = 0.40  # the button's top under the harness's pads (TF's lid)


def _compliant(ctx, monkeypatch, tmp_path, held_up_m, **kw):
    import json

    ctx.client.compliant = True
    result, events, arm, legs, start = _press(
        ctx, monkeypatch, tmp_path, [], touch_at=None, compliant=True, top_z=COMPLIANT_TOP, held_up_m=held_up_m, **kw
    )
    rows = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl")))]
    (note,) = (r for r in rows if r.get("leg") == "depth_compliant_press")
    return result, events, arm, legs, note


def test_a_compliant_press_is_one_stroke_held_and_back_up_with_no_look_and_no_guard(ctx, monkeypatch, tmp_path):
    """Owner 2026-10-27: impedance, and nothing that stops the arm on a bad
    reading or a hit. The stroke flies as one execution to below the
    button's bottom — as far as the spring needs to push COMPLIANT_PRESS_N
    there — under the STIFF tuning, is held, and comes back up to where it
    began. No look (the harness has none to give), no guard on any leg.
    The stroke meets the box by design: from it on every world, home's
    included, pads the box for a shove, as a guarded touch does. The turn
    before it and the rise after it, in free air, fly at a compliant run's
    COMPLIANT_FREE_LINE_SPEED."""
    from box_opening.constants import COMPLIANT_FREE_LINE_SPEED
    from box_opening.primitives.core import CONTACT_SHIFT_PAD_M

    cfg = load_press_demo(PINK)
    assert ctx.contact_pad == 0.0
    result, events, arm, legs, note = _compliant(ctx, monkeypatch, tmp_path, held_up_m=0.0)
    assert events == ["depth:turn", "depth:approach", "depth:press", "depth:rise"]
    assert len(ctx.client.executed) == 3  # the turn, the stroke (approach and press merged), the rise
    assert all(g is None for g in ctx.client.guards)
    assert ctx.client.gains_sent == [None, dp.COMPLIANT_GAINS, dp.COMPLIANT_GAINS]
    # half their speed (owner 2026-10-28): 0.075 and 0.175 for the pink box
    assert legs["depth:press"].speed == dp.COMPLIANT_STROKE_SLOWDOWN * dp.TOUCH_SPEED == pytest.approx(0.075)
    assert legs["depth:approach"].speed == dp.COMPLIANT_STROKE_SLOWDOWN * cfg.press_speed == pytest.approx(0.175)
    assert legs["depth:turn"].speed == legs["depth:rise"].speed == COMPLIANT_FREE_LINE_SPEED
    k = note["stiffness_n_per_m"]
    over = min(max(dp.COMPLIANT_PRESS_N / k, dp.COMPLIANT_OVERSHOOT_M[0]), dp.COMPLIANT_OVERSHOOT_M[1])
    assert note["end_z"] == pytest.approx(COMPLIANT_TOP - cfg.button_travel_m - over, abs=1e-4)
    ((goal, quat),) = arm.turns
    assert legs["depth:rise"].goal_joints == pytest.approx(list(goal) + list(quat))  # back up to where the stroke began
    # the pads reached the stroke's end: nothing pushed back, nothing pressed
    assert not result.felt and not result.judged and dp.exit_code(result) == 1
    assert ctx.contact_pad == CONTACT_SHIFT_PAD_M


def test_a_compliant_press_held_short_of_its_end_pressed_with_the_spring(ctx, monkeypatch, tmp_path):
    """The button's bottom held the pads short of the stroke's end: the
    spring pushed stiffness x that, and the run says so — 0, the knob not
    looked at."""
    result, _e, _a, _l, note = _compliant(ctx, monkeypatch, tmp_path, held_up_m=0.05)
    assert result.felt and result.press_n == pytest.approx(note["stiffness_n_per_m"] * 0.05, rel=0.02)
    assert result.free and dp.exit_code(result) == 0
    assert result.touch_xyz is not None


def test_the_stiffness_at_the_pads_is_the_joint_springs_seen_through_the_arm():
    """A chain whose joints ARE the pads' position: straight down it is just
    joint 3's spring."""

    class Cartesian:
        def fk(self, q):
            return np.eye(3), np.asarray(q[:3], float) - np.asarray(dp.DOWN_CONTACT_TOOL, float)

    assert dp.fingertip_stiffness_z(Cartesian(), [0.5, -0.1, 0.4]) == pytest.approx(dp.COMPLIANT_KQ[2])


def test_the_press_models_the_spring_its_legs_fly():
    """The press's stiffness at the pads, its overshoot and press_n are
    modelled on the spec its legs send — the compliant run's, RIGID — from
    the one table of the driver's tunings: no hand copy of the driver's
    numbers to drift."""
    from box_opening.constants import COMPLIANT_RUN_GAINS, GAINS_RIGID

    assert dp.COMPLIANT_GAINS is COMPLIANT_RUN_GAINS is GAINS_RIGID
    assert dp.COMPLIANT_KQ == IMPEDANCE_KQ[GAINS_RIGID] == (400.0,) * 7


STIFF_KQ = IMPEDANCE_KQ[GAINS_STIFF]  # Kq 160/60: the spring the 10-27 press tests were laid out on


def _pin_stiff(monkeypatch):
    """Model the press on STIFF for a test whose chain or numbers were laid
    out on it: dp.COMPLIANT_KQ and COMPLIANT_GAINS, which the hold's
    reading and _held_at take at call time. compliant_stroke's and
    stroke_bend's kq defaults were bound at import, to RIGID's: returns
    STIFF_KQ, to hand them."""
    monkeypatch.setattr(dp, "COMPLIANT_KQ", STIFF_KQ)
    monkeypatch.setattr(dp, "COMPLIANT_GAINS", GAINS_STIFF)
    return STIFF_KQ


# ---- the compliant stroke's bend: the springs push straight down at the button's bottom ------


def _pads_at(chain, q):
    """The pads (DOWN_CONTACT_TOOL) at joints `q`, from `chain`'s fk alone."""
    R, t = chain.fk(list(q))
    return np.asarray(t, float) + np.asarray(R, float) @ np.asarray(dp.DOWN_CONTACT_TOOL, float)


def _straight_end(chain, pts, bottom):
    """The last of a stroke leg's joint points `pts` whose pads stand no
    lower than `bottom`: where its straight part ends, at the button's
    bottom — past it a press is bent (dp.stroke_bend)."""
    return [q for q in pts if _pads_at(chain, q)[2] >= bottom - 1e-4][-1]


def _held_at(chain, q_ref, b):
    """Where the joint springs (dp.COMPLIANT_KQ: RIGID, or STIFF where a
    test pins it) about the reference `q_ref` hold a 7-joint
    arm with the pads pinned at `b` — the driver's law at rest, gravity
    cancelled: Kq (q - q_ref) = J(q)^T f, the pads at b — solved for f, the
    button's push on the pads. Its own Jacobian (central differences of
    `chain`'s fk): nothing of depth_press's model in it. Returns (the held
    joints, f in world axes)."""
    kq = np.asarray(dp.COMPLIANT_KQ, float)
    q_ref = np.asarray(q_ref, float)

    def jac(q):
        J = np.zeros((3, 7))
        for i in range(7):
            d = np.zeros(7)
            d[i] = 1e-6
            J[:, i] = (_pads_at(chain, q + d) - _pads_at(chain, q - d)) / 2e-6
        return J

    q, f = q_ref.copy(), np.zeros(3)
    for _ in range(30):
        J = jac(q)
        f = f + np.linalg.solve(J @ np.diag(1.0 / kq) @ J.T, np.asarray(b, float) - _pads_at(chain, q))
        q = q_ref + (J.T @ f) / kq
    assert np.linalg.norm(_pads_at(chain, q) - np.asarray(b, float)) < 1e-7
    return q, f


def _hold_note(ctx, monkeypatch, log_dir, chain, stroke, held):
    """dp.compliant_hold on `chain` with the arm held at joints `held`, on
    the button whose top is COMPLIANT_TOP: (its verdict, its run record's
    depth_compliant_press row)."""
    import json

    from box_opening.runtime import approach
    from box_opening.runtime.runner import Runner

    monkeypatch.setattr(approach, "_chain", lambda: chain)
    ctx.client.settled_efforts = lambda *a, **k: ([0.0] * 4, [float(v) for v in held])
    result = dp.compliant_hold(
        ctx, load_press_demo(PINK), Runner(ctx.client, log_dir=log_dir), stroke, COMPLIANT_TOP, {}
    )
    rows = [json.loads(line) for line in open(next(log_dir.glob("run-*.jsonl")))]
    (note,) = (r for r in rows if r.get("leg") == "depth_compliant_press")
    return result, note


class _Coupled:
    """A chain whose pads are A q[:3] — joints 4-7 move nothing — and
    whose A couples x and z: a push straight down moves its pads sideways
    too, as flatbot's do (C = A Kq^-1 A^T, not diagonal). Its straight
    lines end where they are asked to; the hand's axes are the world's."""

    def __init__(self, xz, zx=0.0):
        self.A = np.array([[1.0, 0.0, xz], [0.0, 1.0, 0.0], [zx, 0.0, 1.0]])

    def fk(self, q):
        return np.eye(3), self.A @ np.asarray(q[:3], float) - np.asarray(dp.DOWN_CONTACT_TOOL, float)

    def straight_line(self, q0, xyz, quat):
        q = [float(v) for v in q0]
        q[:3] = [float(v) for v in np.linalg.solve(self.A, np.asarray(xyz, float) + np.asarray(dp.DOWN_CONTACT_TOOL))]
        return [[float(v) for v in q0], q], None


Q_COUPLED = [0.5, -0.1, 0.5, 0.0, 0.0, 0.0, 0.0]  # where a _Coupled stroke starts: its pads 0.50-0.55 m up


@pytest.mark.parametrize("stiffer", [1.0, 4.0])
def test_a_bent_stroke_ends_where_the_joint_springs_push_straight_down_at_the_buttons_bottom(
    ctx, monkeypatch, tmp_path, stiffer
):
    """The driver's spring is a joint spring, and on an arm whose pads'
    compliance couples (_Coupled), a reference driven straight down past
    the button pushes sideways too. The press leg goes straight down to the
    button's bottom (q_b), then bends on in joint space to q_end with
    Kq (q_end - q_b) = -F J_z(q_b)^T: the arm held at the bottom, the
    springs push F straight down there and nothing sideways, the hold's
    own reading included. F is COMPLIANT_PRESS_N where the pads' reference
    then drops within COMPLIANT_OVERSHOOT_M (`stiffer` 4: 634 N/m at the
    pads, 24 mm), else scaled to its bound (STIFF on this chain: 158 N/m,
    60 mm). No new leg: the events are as before. STIFF pinned
    (_pin_stiff): this chain's numbers were laid out on it."""
    cfg = load_press_demo(PINK)
    chain = _Coupled(xz=0.25, zx=0.1)
    kq = stiffer * np.asarray(_pin_stiff(monkeypatch), float)
    stroke, why = dp.compliant_stroke(chain, cfg, Q_COUPLED, TURNED_DOWN, COMPLIANT_TOP, kq=tuple(kq))
    assert why is None and stroke.bent and stroke.bend_why is None
    assert [name for name, _p, _s in stroke.legs] == ["depth:approach", "depth:press"]
    bottom = COMPLIANT_TOP - cfg.button_travel_m
    press = stroke.legs[-1][1]
    q_b = np.asarray(_straight_end(chain, press, bottom), float)
    assert list(_pads_at(chain, q_b)) == pytest.approx(list(_pads_at(chain, Q_COUPLED)[:2]) + [bottom], abs=1e-9)
    q_end = np.asarray(stroke.q_end, float)
    assert list(press[-1]) == pytest.approx(list(q_end), abs=1e-12)  # the press leg's last point
    # the bend's waypoints: no joint more than BEND_STEP_RAD apart, q_b to q_end on one joint-space line
    tail = np.asarray(press[max(i for i, q in enumerate(press) if np.abs(np.asarray(q) - q_b).max() < 1e-12) :])
    assert len(tail) >= 3 and np.abs(np.diff(tail, axis=0)).max() <= dp.BEND_STEP_RAD + 1e-12
    d = q_end - q_b
    s = (tail - q_b) @ d / (d @ d)
    assert np.abs(tail - q_b - np.outer(s, d)).max() < 1e-12 and np.all(np.diff(s) > 0) and s[-1] == pytest.approx(1.0)
    J = np.hstack([chain.A, np.zeros((3, 4))])  # the pads' translation Jacobian, and their compliance through it
    assert dp.fingertip_jacobian(chain, q_b) == pytest.approx(J, abs=1e-9)
    assert dp.fingertip_compliance(chain, q_b, tuple(kq)) == pytest.approx(J @ np.diag(1.0 / kq) @ J.T, abs=1e-12)
    J_z = J[2]  # the pads' height: z + 0.1 x
    k = 1.0 / float(J_z @ (J_z / kq))
    assert stroke.k == pytest.approx(k, rel=1e-6)
    force = k * stroke.over  # the chain is linear: the drop is F C_zz exactly
    assert force == pytest.approx(min(dp.COMPLIANT_PRESS_N, k * dp.COMPLIANT_OVERSHOOT_M[1]), rel=1e-6)
    assert kq * (q_end - q_b) == pytest.approx(-force * J_z, abs=1e-9)
    # the static force at the bottom: straight down, F
    f = np.linalg.lstsq(dp.fingertip_jacobian(chain, q_b).T, kq * (q_end - q_b), rcond=None)[0]
    assert f == pytest.approx([0.0, 0.0, -force], abs=1e-9)
    # the bend moved the pads' reference sideways, as the coupling asks: 0.35 of the drop on this chain
    aside = np.hypot(*(_pads_at(chain, q_end) - _pads_at(chain, q_b))[:2])
    assert stroke.bend_m == pytest.approx(aside) and aside == pytest.approx(0.35 * stroke.over / 1.01, rel=1e-6)
    if stiffer == 1.0:  # the hold reads the springs that fly: COMPLIANT_KQ, held at the bottom
        result, note = _hold_note(ctx, monkeypatch, tmp_path, chain, stroke, q_b)
        assert result.felt and result.press_n == pytest.approx(force, abs=1e-6)
        assert note["press_n"] == pytest.approx(force, abs=0.05) and note["sideways_n"] == 0.0
        assert note["bent"] and note["bend_mm"] == pytest.approx(1000 * aside, abs=0.05)
        assert note["held_off_aim_mm"] == [0.0, 0.0] and "bend_why" not in note


def test_where_the_pads_compliance_does_not_couple_the_bend_is_straight_down(ctx, monkeypatch, tmp_path):
    """The harness's arm (_Arm: its joints are the tool's pose) turned
    straight down: its pads' compliance is diagonal, so the bend's end lies
    straight below the button's bottom — only the pads' height moves, in
    joint space too — and the press ends where it did before the bend,
    the spring there pushing what it did (the compliant press tests above,
    unchanged)."""
    _result, _events, arm, legs, note = _compliant(ctx, monkeypatch, tmp_path, held_up_m=0.0)
    ((goal, quat),) = arm.turns
    cfg = load_press_demo(PINK)
    stroke, why = dp.compliant_stroke(arm, cfg, list(goal) + list(quat), quat, COMPLIANT_TOP)
    assert why is None and stroke.bent and stroke.bend_m == pytest.approx(0.0, abs=1e-9)
    q_b = np.asarray(_straight_end(arm, stroke.legs[-1][1], COMPLIANT_TOP - cfg.button_travel_m), float)
    step = np.asarray(stroke.q_end, float) - q_b
    assert step[2] == pytest.approx(-stroke.over, abs=1e-9) and np.abs(np.delete(step, 2)).max() < 1e-9
    assert list(legs["depth:press"].traj.points[-1].positions) == pytest.approx(stroke.q_end)
    assert note["bent"] and note["bend_mm"] == 0.0 and note["sideways_n"] == 0.0


def test_a_bend_past_bend_max_m_flies_straight_and_the_run_says_why(ctx, monkeypatch, tmp_path, capsys):
    """A bend that would move the pads' reference more than BEND_MAX_M
    sideways — here 60 mm, the pads' x coupled one to one with their height
    — is near a degenerate compliance, and in free air the pads would end
    that far off the button: the press flies straight on down as before
    the bend, as far below the bottom as the spring there needs, within
    COMPLIANT_OVERSHOOT_M, and the console and the run record say why. Held
    at the bottom, that straight reference shoves the pads sideways: the
    hold says so. STIFF pinned (_pin_stiff): this chain's 60 mm and 19.2 N
    were laid out on it."""
    cfg = load_press_demo(PINK)
    chain = _Coupled(xz=1.0)
    stroke, why = dp.compliant_stroke(chain, cfg, Q_COUPLED, TURNED_DOWN, COMPLIANT_TOP, kq=_pin_stiff(monkeypatch))
    assert why is None and not stroke.bent and stroke.bend_m == 0.0
    assert stroke.bend_why == "a bend would move the pads' reference 60 mm sideways, past BEND_MAX_M's 30"
    assert [name for name, _p, _s in stroke.legs] == ["depth:approach", "depth:press"]
    bottom = COMPLIANT_TOP - cfg.button_travel_m
    assert stroke.over == pytest.approx(dp.COMPLIANT_OVERSHOOT_M[1]) and stroke.end_z == pytest.approx(bottom - 0.06)
    end = _pads_at(chain, stroke.q_end)
    assert list(end) == pytest.approx(list(_pads_at(chain, Q_COUPLED)[:2]) + [bottom - 0.06], abs=1e-9)  # straight
    assert list(stroke.legs[-1][1][-1]) == pytest.approx(stroke.q_end)
    assert "; not bent: a bend would move the pads' reference 60 mm sideways" in dp.press_text(stroke, cfg)
    q_b = np.linalg.solve(chain.A, [end[0], end[1], bottom])  # the pads held at the bottom, over the aim
    result, note = _hold_note(ctx, monkeypatch, tmp_path, chain, stroke, list(q_b) + [0.0] * 4)
    assert not note["bent"] and note["bend_why"] == stroke.bend_why and note["bend_mm"] == 0.0
    # held there the springs push 19.2 N down and 9.6 N sideways (A^T f = Kq (q_end - q_b))
    assert note["press_n"] == pytest.approx(19.2, abs=0.05) and note["sideways_n"] == pytest.approx(9.6, abs=0.05)
    assert result.felt and "PRESSED with ~19 N, 10 N sideways" in capsys.readouterr().out


def test_a_bend_whose_end_leaves_a_joints_limits_flies_straight():
    """The bend's waypoints are no IK solutions, so no line's limit check
    sees them: an end outside a joint's limits (within_limits, as the arm's
    own chain has) is no bend to fly, and the stroke flies straight on
    down as before the bend. STIFF pinned: its 60 mm bound was laid out on
    it."""
    cfg = load_press_demo(PINK)
    bottom = COMPLIANT_TOP - cfg.button_travel_m

    class Limited(_Coupled):
        def within_limits(self, q):
            return float((self.A @ np.asarray(q[:3], float))[2]) >= bottom - 0.01  # the pads' height, on this chain

    stroke, why = dp.compliant_stroke(Limited(xz=0.25), cfg, Q_COUPLED, TURNED_DOWN, COMPLIANT_TOP, kq=STIFF_KQ)
    assert why is None and not stroke.bent and stroke.bend_why == "the bend's end is outside a joint's limits"
    assert stroke.end_z == pytest.approx(bottom - dp.COMPLIANT_OVERSHOOT_M[1])
    assert list(_pads_at(Limited(xz=0.25), stroke.q_end)[:2]) == pytest.approx(
        list(_pads_at(Limited(xz=0.25), Q_COUPLED)[:2]), abs=1e-9
    )  # straight below the aim
    assert dp.stroke_bend(_Coupled(xz=0.25), Q_COUPLED, kq=STIFF_KQ)[0] is not None  # unlimited, it bends


def test_a_bend_is_no_bend_where_the_pads_do_not_move_with_the_joints(monkeypatch):
    """A chain whose pads stand still whatever its joints (test_reach_press's
    _Chain) has no compliance to bend along: the stroke flies straight as
    before the bend, its spring infinitely stiff, the least overshoot."""
    import test_reach_press as trp

    monkeypatch.setattr(trp._Chain, "R", trp.STRAIGHT_DOWN)
    cfg = load_press_demo(PINK)
    stroke, why = dp.compliant_stroke(trp._Chain(), cfg, [0.0] * 7, TURNED_DOWN, 0.40)
    assert why is None and not stroke.bent and stroke.bend_why == "the pads do not move with the joints here"
    assert stroke.k == float("inf") and stroke.end_z == pytest.approx(0.40 - cfg.button_travel_m - 0.015)
    q_end, aside, why = dp.stroke_bend(trp._Chain(), [0.0] * 7)
    assert q_end is None and aside == 0.0 and why == stroke.bend_why


def test_a_hand_already_at_the_buttons_bottom_flies_straight_and_one_below_the_strokes_end_is_refused():
    """The bend starts where the press leg's straight part meets the
    button's bottom. A hand whose pads already stand within 5 mm of it, or
    below it (a wrong top_z, or the hand already in the button), has no
    such part: bent from where it stands, the springs' reference would be
    driven that much deeper still. It flies straight on down as before the
    bend, and pads already below where that stroke ends are refused, as
    they were before it. STIFF pinned: this chain's numbers were laid out
    on it."""
    cfg = load_press_demo(PINK)
    chain = _Coupled(xz=0.25, zx=0.1)
    bottom = COMPLIANT_TOP - cfg.button_travel_m
    x, y = _pads_at(chain, Q_COUPLED)[:2]
    at = list(np.linalg.solve(chain.A, [x, y, bottom + 0.002])) + [0.0] * 4  # the pads 2 mm above the bottom
    stroke, why = dp.compliant_stroke(chain, cfg, at, TURNED_DOWN, COMPLIANT_TOP, kq=STIFF_KQ)
    assert why is None and not stroke.bent and stroke.bend_m == 0.0
    assert stroke.bend_why == (
        "the pads already stand 2 mm above the button's bottom, with no stroke down to it to " "bend from"
    )
    assert [name for name, _p, _s in stroke.legs] == ["depth:press"]
    # straight on down, as far below the bottom as the spring needs (STIFF's 158 N/m: the 60 mm bound)
    assert stroke.over == pytest.approx(dp.COMPLIANT_OVERSHOOT_M[1]) and stroke.end_z == pytest.approx(bottom - 0.06)
    assert list(_pads_at(chain, stroke.q_end)) == pytest.approx([x, y, bottom - 0.06], abs=1e-9)
    below = list(np.linalg.solve(chain.A, [x, y, bottom - 0.088])) + [0.0] * 4  # 88 mm below it
    assert dp.compliant_stroke(chain, cfg, below, TURNED_DOWN, COMPLIANT_TOP, kq=STIFF_KQ) == (
        None,
        "the pads already stand 28 mm below where the press would end",
    )


def test_a_compliant_reach_out_reads_nothing_at_rest_and_no_leg_can_stop_it(ctx, monkeypatch, tmp_path):
    """The reach-out of a compliant run: no at-rest read, no weight check at
    the look, no stopping guard on the re-aim, the slide in, the turn, the
    stroke or the way up, and the press stroked and judged at the higher
    top — the scene camera's, as at 15:19 when the wrist's aim stood 24 mm
    low and the touch met the scene's. With no guard to stop them in time,
    the free-air lines fly at COMPLIANT_FREE_LINE_SPEED — not the
    CONTACT_SPEED 16:23 crawled 10.5 s of them at, nor a guarded run's
    FREE_LINE_SPEED — and each is labelled with the full world: the runner
    refuses an unguarded leg faster than CONTACT_SPEED in any other (spec
    §6), and the arm would hold at the look pose with nothing pressed. The
    clear before home goes up and back toward the arm
    (rp.clear_targets), planned while the way in flew (rp.plan_way_out):
    with the hand straight down, CLEAR_EXTRA_M over HOME_CLEAR_M and
    CLEAR_BACK_M back. Straight down already within TOUCH_SLOW_M of the
    button, the stroke has no fast stretch."""
    import json

    import test_reach_press as trp

    from box_opening.constants import COMPLIANT_FREE_LINE_SPEED
    from box_opening.worlds import planned_top_z

    monkeypatch.setattr(trp._Chain, "R", trp.STRAIGHT_DOWN)
    low = (trp.BUTTON[0], trp.BUTTON[1], trp.BUTTON[2] - 0.024)
    code, rec, chain, _h, got = _depth_flow(ctx, monkeypatch, tmp_path, _popped, compliant=True, first=(low, 0.4))
    assert code == 0 and rec.reads == 0
    assert rec.events == [
        "reach:look",
        "aim:reach-aim",
        "reach:re-aim",
        "reach:slide-in",
        "depth:turn",
        "depth:press",
        "depth:clear",
        "home",
    ]
    assert all(
        rec.legs[n].guard is None
        for n in ("reach:re-aim", "reach:slide-in", "depth:turn", "depth:press", "depth:clear")
    )
    for n in ("reach:re-aim", "reach:slide-in", "depth:turn", "depth:clear"):
        assert rec.legs[n].speed == COMPLIANT_FREE_LINE_SPEED and rec.legs[n].world.startswith("full"), (
            n,
            rec.legs[n].speed,
            rec.legs[n].world,
        )
    up = [xyz for xyz, _q in chain.lines][-1]  # from the fake fk's (0.5, -0.4, 0.45)
    u = np.asarray(ctx.cpose.xyz[:2]) / np.hypot(*ctx.cpose.xyz[:2])
    assert up[:2] == pytest.approx(list(np.array([0.5, -0.4]) - rp.CLEAR_BACK_M * u))
    assert up[2] == pytest.approx(planned_top_z(ctx.model, ctx.cpose) + rp.HOME_CLEAR_M + rp.CLEAR_EXTRA_M)
    rows = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl")))]
    assert not [r for r in rows if r.get("outcome") == "refused"]
    (out,) = (r for r in rows if r.get("leg") == "reach_way_out")
    assert out["planned"] and out["clear_back_mm"] == pytest.approx(1000 * rp.CLEAR_BACK_M) and out["why"] is None
    assert out["legs"] == ["depth:clear", "home"]  # the fake fk stands still: nothing to rise
    assert not [r for r in rows if r.get("leg") == "depth_clear"]  # no clear solved after the hold
    assert got["client_compliant"] and got["top_z"] == pytest.approx(trp.BUTTON[2])
    # the stroke solved for that top: the fake chain's pads infinitely stiff, the least overshoot past its bottom
    cfg = load_press_demo(PINK)
    assert got["stroke"].end_z == pytest.approx(trp.BUTTON[2] - cfg.button_travel_m - dp.COMPLIANT_OVERSHOOT_M[0])
    assert not ctx.client.compliant  # the client is as it was once the run is over


def test_a_compliant_walk_in_flies_at_the_free_line_speed_too(ctx, monkeypatch, tmp_path):
    """Where the slide in turns the hand down on the way (the walk, as at
    20:04), the walk is the compliant run's slide in: unguarded, at
    COMPLIANT_FREE_LINE_SPEED, labelled with the full world, and not refused."""
    import test_reach_press as trp

    from box_opening.constants import COMPLIANT_FREE_LINE_SPEED

    monkeypatch.setattr(
        trp._Chain, "walk", lambda self, q0, xyz, quat, **k: ([list(q0), [v + 0.01 for v in q0]], None), raising=False
    )
    code, rec, _chain, _h, got = _depth_flow(ctx, monkeypatch, tmp_path, _popped, compliant=True)
    assert code == 0 and got["client_compliant"] and rec.reads == 0
    assert rec.events == [
        "reach:look",
        "aim:reach-aim",
        "reach:re-aim",
        "reach:slide-in",
        "depth:approach",
        "depth:press",
        "depth:clear",
        "home",
    ]
    leg = rec.legs["reach:slide-in"]
    assert rp._rot(leg.target[2])[2, 2] == pytest.approx(-1.0)  # the walk's: the hand turned straight down
    assert leg.guard is None and leg.speed == COMPLIANT_FREE_LINE_SPEED and leg.world.startswith("full"), (
        leg.speed,
        leg.world,
    )


def test_a_compliant_way_in_flown_leg_by_leg_flies_its_free_air_lines_at_the_compliant_speed_too(
    ctx, monkeypatch, tmp_path
):
    """Where the way in's legs do not make one execution (_not_one_execution)
    a press-only compliant run flies them leg by leg, each solved again from
    where the arm stands: the re-aim, the slide in, the press, then the
    clear before home (_clear_up_and_back). Its free-air lines fly as the one
    execution's do — unguarded, at COMPLIANT_FREE_LINE_SPEED — never at the
    CONTACT_SPEED or FREE_LINE_SPEED of a guarded run."""
    from box_opening.constants import COMPLIANT_FREE_LINE_SPEED

    monkeypatch.setattr(rp, "_not_one_execution", lambda ctx_, legs: "its legs do not meet")
    code, rec, _chain, _h, got = _depth_flow(ctx, monkeypatch, tmp_path, _popped, compliant=True)
    assert code == 0 and got["compliant"] and got["client_compliant"] and rec.reads == 0
    assert rec.events == ["reach:look", "aim:reach-aim", "reach:re-aim", "reach:slide-in", "depth:clear", "home"]
    for n in ("reach:re-aim", "reach:slide-in", "depth:clear"):
        lg = rec.legs[n]
        assert lg.guard is None and lg.speed == COMPLIANT_FREE_LINE_SPEED and lg.world.startswith("full"), (
            n,
            lg.speed,
            lg.world,
        )


def test_a_compliant_reach_out_primes_the_impedance_once_before_the_reach_is_planned(
    ctx, monkeypatch, tmp_path, capsys
):
    """Flatbot 2026-10-27 16:23: the look, the run's first compliant goal,
    took 1.857 s against its 1.60 s preview — the driver's switch out of
    POSITION and its spring fading in under the look. The hold that primes
    the joint impedance goes out once, before the reach is planned, at the
    very read the look's plan starts from: the hold's reference is the
    look's first waypoint."""
    reads = []
    real_joints = ctx.client.joints

    def joints():  # until the arm first moves, each read its own sample, 0.1 mrad on from the last
        q = real_joints()
        if ctx.client.executed:
            return q
        reads.append([v + 1e-4 * (len(reads) + 1) for v in q])
        return list(reads[-1])

    monkeypatch.setattr(ctx.client, "joints", joints)
    planned = []
    real_plan = rp.plan_reach

    def plan_reach(ctx_, cfg, button, q_from, *a, **k):
        planned.append((list(q_from), len(ctx_.client.primes)))
        return real_plan(ctx_, cfg, button, q_from, *a, **k)

    monkeypatch.setattr(rp, "plan_reach", plan_reach)
    code, rec, _chain, _h, got = _depth_flow(ctx, monkeypatch, tmp_path, _popped, compliant=True)
    assert code == 0 and got["client_compliant"]
    (q0,) = ctx.client.primes
    assert q0 in reads  # one read ...
    first_from, primed_by_then = planned[0]
    assert primed_by_then == 1 and first_from == q0  # ... primed before the reach is planned from it
    assert list(rec.legs["reach:look"].traj.points[0].positions) == q0  # the look starts at the hold
    assert "the driver enters its joint impedance now, while the reach is planned" in capsys.readouterr().out


@pytest.mark.parametrize("compliant", [False, True])
def test_a_reach_out_re_times_every_motion_on_its_own_runs_profile(ctx, monkeypatch, tmp_path, compliant):
    """press_demo builds the runner on a position client; run_reach_press
    makes the client compliant only then, and puts it back when the run
    ends. Every motion a compliant reach-out re-times — the look, the way in,
    the way out — eases at COMPLIANT_ACCEL / COMPLIANT_DECEL under the
    wrist's COMPLIANT_QD_CEILING (Runner.retime_params reads the flag at
    each re-timing: a flag kept from when the runner was built would fly
    the whole run on the position profile); a guarded one, in position,
    keeps RetimeParams' and no ceiling."""
    from box_opening.constants import COMPLIANT_ACCEL, COMPLIANT_DECEL, COMPLIANT_QD_CEILING
    from box_opening.runtime import runner as runner_mod
    from box_opening.runtime.retime import RetimeParams

    seen = []  # the profile each re-timing is handed: a group's, in flight and preview, and a free-air line's
    in_line = []
    real_group, real_line = runner_mod.retime_group, runner_mod.line_profile

    def retime_group(trajs, speeds, vmax, params=RetimeParams()):  # noqa: B008 - frozen dataclass
        if not in_line:  # line_profile's own re-timings scale the ease by the line's speed: what it is handed counts
            seen.append(params)
        return real_group(trajs, speeds, vmax, params)

    def line_profile(leg, params, time_scale=1.0):
        seen.append(params)
        in_line.append(leg)
        try:
            return real_line(leg, params, time_scale)
        finally:
            in_line.pop()

    monkeypatch.setattr(runner_mod, "retime_group", retime_group)
    monkeypatch.setattr(runner_mod, "line_profile", line_profile)
    assert not getattr(ctx.client, "compliant", False)  # as press_demo built its runner
    code, rec, _chain, _h, got = _depth_flow(ctx, monkeypatch, tmp_path, _popped, compliant=compliant)
    assert code == 0 and got["client_compliant"] is compliant and seen
    plain = RetimeParams()
    want = (COMPLIANT_ACCEL, COMPLIANT_DECEL, COMPLIANT_QD_CEILING) if compliant else (plain.accel, plain.decel, None)
    assert {(p.accel, p.decel, p.qd_ceiling) for p in seen} == {want}
    assert not ctx.client.compliant  # put back once the run is over


def test_a_guarded_reach_out_is_as_before(ctx, monkeypatch, tmp_path):
    from box_opening.constants import CONTACT_SPEED

    code, rec, _chain, _h, got = _depth_flow(ctx, monkeypatch, tmp_path, _popped)
    # at rest, at the look, at the look again, over the button, and over the box for the home
    assert code == 0 and not got["compliant"] and rec.reads == 5
    assert ctx.client.primes == []  # every leg in POSITION: no joint impedance to prime
    assert rec.legs["reach:slide-in"].guard is not None
    # toward the box slow enough for its guard to stop it, in the interaction world
    for n in ("reach:re-aim", "reach:slide-in"):
        assert rec.legs[n].speed == CONTACT_SPEED and rec.legs[n].world.startswith("interaction"), (
            n,
            rec.legs[n].speed,
            rec.legs[n].world,
        )
    assert rec.legs["depth:clear"].speed == CONTACT_SPEED and rec.legs["depth:clear"].guard is not None


@pytest.mark.parametrize("walk_in", [False, True])
def test_a_compliant_press_is_proven_for_the_scene_button_and_again_for_the_aimed_one(
    ctx, monkeypatch, tmp_path, walk_in
):
    """Before the look, from the rest pose, for the scene camera's button;
    after the aim, before the re-aim, for the aimed one: the walk in from
    the look pose, else the slide and the turn, then the stroke. The run
    records say which, and the aimed one's flies as it was proven."""
    import json

    code, rec, _chain, _h, got = _depth_flow(ctx, monkeypatch, tmp_path, _popped, compliant=True, walk_in=walk_in)
    assert code == 0 and got["top_z"] == pytest.approx(0.40)
    assert rec.events == [
        "reach:look",
        "aim:reach-aim",
        "reach:re-aim",
        "reach:slide-in",
        *([] if walk_in else ["depth:turn"]),
        "depth:approach",
        "depth:press",
        "depth:clear",
        "home",
    ]
    rows = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl")))]
    (plan,) = (r for r in rows if r["leg"] == "reach_plan")
    assert plan["proof"] == "scene" and plan["down"] == {"walked": walk_in, "short_deg": 0.0}
    (way,) = (r for r in rows if r["leg"] == "reach_way_in")
    assert (way["tilt_deg"], way["walked"], way["short_deg"]) == (rp.REACH_TILTS_DEG[0], walk_in, 0.0)
    # the fake chain's pads are infinitely stiff: the stroke ends the least overshoot below the button's bottom
    cfg = load_press_demo(PINK)
    assert way["end_z"] == pytest.approx(0.40 - cfg.button_travel_m - dp.COMPLIANT_OVERSHOOT_M[0], abs=1e-4)
    # ... and the stroke that flew and was held is that proof's: not solved again
    assert got["stroke"].end_z == pytest.approx(way["end_z"], abs=1e-4) and got["stroke"].k == way["k"]
    assert list(rec.legs["depth:press"].traj.points[-1].positions) == pytest.approx(
        [float(v) for v in got["stroke"].legs[-1][1][-1]]
    )
    order = [r["leg"] for r in rows]
    assert (
        order.index("reach_plan")
        < order.index("reach:look")
        < order.index("reach_way_in")
        < order.index("reach:re-aim")
    )


def test_a_compliant_run_whose_aimed_button_no_way_down_reaches_goes_home_from_the_look(
    ctx, monkeypatch, tmp_path, capsys
):
    """16:23 2026-10-27: the aim put the button 12 mm from the scene
    camera's, and the arm flew the re-aim and a 7.4 s slide in before the
    turn down was asked for — "no straight-down hand reaches here" — then
    the clear, home refused INVALID_START, a 15 cm back-off and home. Now
    the aimed button's way down is solved before the re-aim: where neither
    the walk in nor the turn after the slide turns the hand down, the arm
    goes home from the look pose, in free air: no re-aim, no slide in, no
    clear, no back-off, nothing pressed."""
    import json

    turns = []

    def scene_only(chain, model, q, contact_tool, shorts=dp.TURN_SHORT_DEG):
        turns.append(list(q))
        if len(turns) == 1:  # plan_reach's, for the scene camera's button
            return _turns_down(chain, model, q, contact_tool, shorts)
        return None, None, None, "0 deg short: IK did not converge 73% of the way"

    code, rec, _chain, _h, got = _depth_flow(ctx, monkeypatch, tmp_path, _popped, compliant=True, turn=scene_only)
    assert code == 1 and rec.events == ["reach:look", "aim:reach-aim", "home"] and got == {}
    assert len(turns) == 1 + 1 + len(rp.TURN_LOW_ABOVE_M)  # the scene's; the aimed one's at the slide's end, then lower
    out = capsys.readouterr().out
    assert (
        "STOP: no press reaches the aimed button (the way down (walk in: 0 deg short: no turning walk modelled; "
        "2 deg short: no turning walk modelled; 5 deg short: no turning walk modelled; turn after the slide: 0 deg "
        "short: IK did not converge 73% of the way; turned 45 mm above the button: 0 deg short: IK did not converge "
        "73% of the way; turned 35 mm above the button: 0 deg short: IK did not converge 73% of the way)) — nothing "
        "was pressed — returning home"
    ) in out
    rows = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl")))]
    assert not [r for r in rows if r["leg"] in ("reach_way_in", "home_refused")]


# ---- a compliant reach-out's way in: one execution from the aimed look pose ----------------


def _sinking_chain():
    """test_reach_press._Chain whose straight lines end where they were
    asked to go: the compliant stroke ends below where it began, and the
    rise back up to its start has a line to fly (the plain chain's fk
    stands still: nothing to rise). Each line's end is its own joints, so
    a line solved before the flight never stands in for one flown."""
    from test_reach_press import _Chain

    class Sinking(_Chain):
        def __init__(self):
            super().__init__()
            self.at = {}

        def straight_line(self, q0, xyz, quat):
            pts, why = super().straight_line(q0, xyz, quat)
            if pts is not None:
                pts[-1][-1] += 1e-6 * (len(self.at) + 1)
                self.at[tuple(float(v) for v in pts[-1])] = [float(v) for v in xyz]
            return pts, why

        def fk(self, q):
            R, t = super().fk(q)
            return R, np.asarray(self.at.get(tuple(float(v) for v in q), t), float)

    return Sinking()


def _way_in_ends(ctx, monkeypatch, outcome, info):
    """The fake client with the first goal sent under COMPLIANT_GAINS — the
    way in's one execution — scripted to end `outcome` with `info`."""
    real = ctx.client.execute
    state = {"first": True}

    def execute(traj, speed, guard=None, while_running=None, stop_when=None, gains=None):
        if gains == dp.COMPLIANT_GAINS and state["first"]:
            state["first"] = False
            ctx.client.exec_script.insert(0, (outcome, dict(info)))
        return real(traj, speed, guard=guard, while_running=while_running, stop_when=stop_when, gains=gains)

    monkeypatch.setattr(ctx.client, "execute", execute)


@pytest.mark.parametrize("walk_in", [False, True])
def test_a_compliant_reach_out_flies_its_way_in_as_one_execution(ctx, monkeypatch, tmp_path, walk_in):
    """In a compliant run nothing is read between the re-aim, the slide in,
    the turn and the stroke, yet each flew as a goal of its own that ended
    at rest. From the aimed look pose they are ONE runner.run, merged into
    one execution under COMPLIANT_GAINS, each starting at the last waypoint
    of the one before as solve_way_in chained them. Then the hold, and the
    rise back up to the stroke's start, the clear and home as one execution
    too (test_a_compliant_reach_out_flies_from_the_button_to_home_as_one_execution).
    A walk in turns the hand down on its way: no turn."""
    import json

    import test_reach_press as trp

    from box_opening.constants import COMPLIANT_FREE_LINE_SPEED
    from box_opening.runtime.legs import merge_groups

    cfg = load_press_demo(PINK)
    code, rec, _chain, _h, got = _depth_flow(
        ctx, monkeypatch, tmp_path, _popped, compliant=True, walk_in=walk_in, chain=_sinking_chain()
    )
    assert code == 0 and rec.reads == 0
    way_in = ["reach:re-aim", "reach:slide-in", *([] if walk_in else ["depth:turn"]), "depth:approach", "depth:press"]
    assert rec.events == ["reach:look", "aim:reach-aim", *way_in, "depth:rise", "depth:clear", "home"]
    legs = [rec.legs[n] for n in way_in]
    assert len(merge_groups(legs)) == 1
    assert (way_in, [dp.COMPLIANT_GAINS]) in rec.runs  # one runner.run, one execution, under STIFF
    for a, b in zip(legs, legs[1:], strict=False):
        assert list(b.traj.points[0].positions) == list(a.traj.points[-1].positions), (a.name, b.name)
    for lg in legs:
        assert lg.guard is None and lg.gains == dp.COMPLIANT_GAINS and lg.plan_free and lg.world.startswith("full"), (
            lg.name,
            lg.guard,
            lg.gains,
            lg.world,
        )
    speeds = {n: COMPLIANT_FREE_LINE_SPEED for n in way_in[:-2]}
    speeds.update(
        {
            "depth:approach": dp.COMPLIANT_STROKE_SLOWDOWN * cfg.press_speed,
            "depth:press": dp.COMPLIANT_STROKE_SLOWDOWN * dp.TOUCH_SPEED,
        }
    )
    assert {lg.name: lg.speed for lg in legs} == speeds
    # then the rise back up to where the stroke began, the clear and home: one run, one execution
    assert rec.runs[-1] == (["depth:rise", "depth:clear", "home"], [dp.COMPLIANT_GAINS])
    rise = rec.legs["depth:rise"]
    assert rise.gains == dp.COMPLIANT_GAINS and rise.speed == COMPLIANT_FREE_LINE_SPEED and rise.guard is None
    assert list(rise.traj.points[0].positions) == list(legs[-1].traj.points[-1].positions)
    assert rise.target[1] == pytest.approx(list(got["stroke"].t0))
    # the hold judged the stroke that flew, at the button's top
    assert got["top_z"] == pytest.approx(trp.BUTTON[2]) and got["client_compliant"]
    assert [float(v) for v in got["stroke"].legs[-1][1][-1]] == list(legs[-1].traj.points[-1].positions)
    rows = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl")))]
    (flight,) = (r for r in rows if r["leg"] == "reach_way_in_flight")
    assert flight["legs"] == way_in and flight["outcome"] == "arrived" and flight["duration_s"] > 0.0
    fracs = flight["seg_start_fracs"]
    assert len(fracs) == len(way_in) and fracs[0] == 0.0 and fracs == sorted(fracs) and fracs[-1] <= 1.0
    assert flight["stops"] == len([v for v in flight["junction_speeds"] if v < 0.04])
    # every member's row carries the one execution's result
    member = [r for r in rows if r["leg"] in way_in and r["kind"] == "motion"]
    assert [r["leg"] for r in member] == way_in and len({r["t_wall"] for r in member}) == 1
    (slid,) = (r for r in rows if r["leg"] == "reach_slid_in")
    assert slid["turned_down"] is walk_in
    assert ctx.contact_pad > 0.0  # the stroke met the box by design: home pads it for a shove


def test_a_way_in_that_did_not_start_goes_home_from_the_look_pose(ctx, monkeypatch, tmp_path, capsys):
    """The driver turned the one execution down: nothing moved, the arm
    stands at the look pose in free air. Home from there, around the box
    as it was never touched — no clear, no back-off, nothing pressed."""
    import json

    _way_in_ends(ctx, monkeypatch, "failed", {"message": "goal rejected by the driver", "progress": 0.0})
    code, rec, _chain, home, got = _depth_flow(ctx, monkeypatch, tmp_path, _popped, compliant=True)
    assert code == 1 and got == {}  # no hold
    assert rec.events[-1] == "home" and "depth:clear" not in rec.events and "depth:rise" not in rec.events
    assert rec.events[:2] == ["reach:look", "aim:reach-aim"] and rec.events.count("home") == 1
    assert home["pad"] == 0.0  # nothing touched: no shove pad around the box
    out = capsys.readouterr().out
    assert "STOP: the way in did not start (goal rejected by the driver) — nothing was pressed — returning home" in out
    rows = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl")))]
    (flight,) = (r for r in rows if r["leg"] == "reach_way_in_flight")
    assert flight["outcome"] == "failed"


def test_a_way_in_that_stopped_part_way_holds_the_arm_and_says_in_which_leg(ctx, monkeypatch, tmp_path, capsys):
    """The execution stopped 60% of the way: the arm stands somewhere on the
    way in, maybe over the box — it holds there, no home, no hold. The STOP
    names the leg in flight by where each began (seg_start_fracs)."""
    import json

    from box_opening.tasks.reach_press import _phase_at

    _way_in_ends(ctx, monkeypatch, "failed", {"message": "PATH_TOLERANCE_VIOLATED", "progress": 0.6})
    code, rec, _chain, _h, got = _depth_flow(ctx, monkeypatch, tmp_path, _popped, compliant=True)
    assert code == 1 and got == {} and "home" not in rec.events
    assert rec.events[-1] == "depth:press"  # the way in was the last thing handed to the runner
    rows = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl")))]
    (flight,) = (r for r in rows if r["leg"] == "reach_way_in_flight")
    phase = _phase_at(flight["legs"], flight["seg_start_fracs"], 0.6)
    assert phase != flight["legs"][0]  # 0.6 of the way is past the re-aim
    assert (
        "STOP: the way in stopped during %s (PATH_TOLERANCE_VIOLATED) — the arm holds" % phase
        in capsys.readouterr().out
    )
    # the last leg begun by then: at its own start fraction, a leg is in flight
    names, starts = ["a", "b", "c"], [0.0, 0.3, 0.7]
    assert [_phase_at(names, starts, p) for p in (0.0, 0.29, 0.3, 0.69, 0.7, 1.0)] == ["a", "a", "b", "b", "c", "c"]


def test_a_way_in_the_runner_refuses_goes_home_from_the_look_pose(ctx, monkeypatch, tmp_path, capsys):
    """The runner's own gates refuse the one execution before anything is
    sent (its trajectory sanity gate on the turn, say): nothing moved, as
    when the driver turns the goal down, so home from the look pose with no
    shove pad — and no flight row, the runner having re-timed nothing for
    it (its last_retime is still the look's)."""
    import json

    from box_opening.runtime.runner import Runner

    real = Runner._refusal

    def refusal(self, leg):
        if leg.name == "depth:turn":
            return "trajectory sanity gate: arm_joint_6 inside its limit margin"
        return real(self, leg)

    monkeypatch.setattr(Runner, "_refusal", refusal)
    code, rec, _chain, home, got = _depth_flow(ctx, monkeypatch, tmp_path, _popped, compliant=True)
    assert code == 1 and got == {} and home["pad"] == 0.0
    way_in = next(run for run in rec.runs if run[0][0] == "reach:re-aim")
    assert "depth:turn" in way_in[0] and way_in[1] == []  # handed over as one run, nothing sent
    assert rec.events[-1] == "home" and "depth:clear" not in rec.events and "depth:rise" not in rec.events
    assert (
        "STOP: the way in did not start (trajectory sanity gate: arm_joint_6 inside its limit margin) — nothing "
        "was pressed — returning home"
    ) in capsys.readouterr().out
    rows = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl")))]
    assert not [r for r in rows if r["leg"] == "reach_way_in_flight"]
    (refused,) = (r for r in rows if r.get("outcome") == "refused")
    assert refused["leg"] == "depth:turn"


def test_why_a_way_in_is_not_one_execution(ctx, monkeypatch):
    """The way in flies as one execution only where its legs make one merge
    group and the first starts where the arm stands (_not_one_execution);
    anything else is said, and the run flies it leg by leg."""
    from test_reach_press import _Chain

    from box_opening.constants import FREE_LINE_SPEED
    from box_opening.runtime import approach
    from box_opening.tasks.reach_press import _not_one_execution

    monkeypatch.setattr(approach, "_chain", lambda: _Chain())
    q = [[0.01 * k] * 7 for k in range(4)]
    legs = [
        dp.compliant_line(ctx, name, q[i : i + 2], TURNED_DOWN, FREE_LINE_SPEED, ("full", "full.yaml"))
        for i, name in enumerate(("reach:re-aim", "reach:slide-in", "depth:press"))
    ]
    ctx.client.live = list(q[0])
    assert _not_one_execution(ctx, legs) is None
    # under the client's run default (None) rather than the spec the others name: a goal of its own
    legs[2].gains = None
    assert _not_one_execution(ctx, legs) == "depth:press does not continue reach:slide-in as one goal"
    legs[2].gains = dp.COMPLIANT_GAINS
    # the arm 2 mrad from the first leg's start: the driver would jump there
    ctx.client.live = [0.002] + list(q[0][1:])
    assert _not_one_execution(ctx, legs) == "the arm stands 0.002 rad from where reach:re-aim starts"
    assert _not_one_execution(ctx, []) == "no line to fly"


def test_the_compliant_stroke_solved_ahead_is_the_one_the_press_flies(ctx, monkeypatch, tmp_path):
    """compliant_stroke from the hand turned down is press_compliant's own
    stroke — its legs, their ends and speeds, its stiffness and its end —
    and turn_down_path is the turn the press flies: what the reach-out
    solves before the arm moves is what then flies."""
    result, _events, arm, legs, note = _compliant(ctx, monkeypatch, tmp_path, held_up_m=0.0)
    ((goal, quat),) = arm.turns
    stroke, why = dp.compliant_stroke(arm, load_press_demo(PINK), list(goal) + list(quat), quat, COMPLIANT_TOP)
    assert why is None and [name for name, _pts, _speed in stroke.legs] == ["depth:approach", "depth:press"]
    for name, pts, speed in stroke.legs:
        assert legs[name].speed == speed
        assert list(legs[name].traj.points[-1].positions) == pytest.approx([float(v) for v in pts[-1]])
    assert stroke.k == pytest.approx(note["stiffness_n_per_m"], abs=0.05)
    assert stroke.end_z == pytest.approx(note["end_z"], abs=5e-5)
    # ... and the turn's path, with the compliant press's leans, is the turn the press flew
    pts, down, short, why = dp.turn_down_path(
        _Arm(), ctx.model, legs["depth:turn"].traj.points[0].positions, rp.CONTACT_TOOL, dp.COMPLIANT_TURN_SHORT_DEG
    )
    assert why is None and short == 0.0 and down == pytest.approx(quat)
    assert list(pts[-1]) == pytest.approx(list(legs["depth:turn"].traj.points[-1].positions))
    assert "lean_deg" not in note  # exactly down: nothing to say


def _lean_deg(quat):
    return float(np.degrees(np.arccos(np.clip(-rp._rot(quat)[2, 2], -1.0, 1.0))))


@pytest.mark.parametrize("refused, lean", [(1, 2.0), (2, 5.0), (5, 12.0), (6, 15.0)])
def test_a_compliant_press_turns_as_far_as_15_deg_short_where_nearer_down_does_not_fly(
    ctx, monkeypatch, tmp_path, capsys, refused, lean
):
    """Flatbot 2026-10-27 16:23: the slid-in fingertip 0.78 m out, where no
    hand within 8 deg of straight down is found, and the press stopped "no
    straight-down hand reaches here" with nothing pressed. A compliant press
    tries COMPLIANT_TURN_SHORT_DEG, smallest lean first: with the turns up
    to 10 deg short refused it turns 12 deg short, about the fingertip — the
    pads' centre where the tilted fingertip stood, over the aim — and
    strokes straight down from there. From LEAN_SAID_DEG on the console and
    the run record say the lean."""
    result, events, arm, legs, note = _compliant(ctx, monkeypatch, tmp_path, held_up_m=0.0, refuse_first=refused)
    assert events == ["depth:turn", "depth:approach", "depth:press", "depth:rise"]
    ((goal, quat),) = arm.turns
    assert _lean_deg(quat) == pytest.approx(lean, abs=1e-6)
    R = rp._rot(quat)
    # the harness's slide ended with the tilted fingertip at (BUTTON xy, 0.46)
    assert np.asarray(goal) + R @ np.asarray(dp.DOWN_CONTACT_TOOL) == pytest.approx([BUTTON[0], BUTTON[1], 0.46])
    bottom = COMPLIANT_TOP - load_press_demo(PINK).button_travel_m
    for name in ("depth:approach", "depth:press"):  # vertical, the lean held, to the button's bottom ...
        pts = [list(p.positions) for p in legs[name].traj.points]
        end = [q for q in pts if dp._pads_z(arm, q) >= bottom - 1e-9][-1]
        assert end[:2] == pytest.approx(list(goal[:2])) and end[3:] == pytest.approx(list(quat))
    # ... and the press bent on past it in joint space (dp.stroke_bend), the springs' reference below it: the
    # fake's pads only drop, its first three joints their own
    end = list(legs["depth:press"].traj.points[-1].positions)
    assert end[:2] == pytest.approx(list(goal[:2])) and dp._pads_z(arm, end) < bottom and end[2] < goal[2]
    out = capsys.readouterr().out
    assert "TURN: %.0f deg short of straight down (no hand nearer straight down flies here)" % lean in out
    said = "PRESS (compliant), the hand leaning %.0f deg back toward the arm: straight down" % lean
    if lean >= dp.LEAN_SAID_DEG:
        assert said in out and note["lean_deg"] == lean
    else:
        assert said not in out and "PRESS (compliant): straight down" in out and "lean_deg" not in note


def _detour(arc):
    """_Arm.walk flying every turn by way of a detour of each joint in turn,
    out and back no further than the runner's sanity gate lets a joint
    swing (dp.line_refusal), `arc(lean)` rad of joint travel in all."""

    def walk(self, q0, xyz, quat, step_m=None, step_rad=None, max_step_rad=None):
        end = [float(v) for v in xyz] + [float(v) for v in quat]
        h = 0.5 * (arc(_lean_deg(quat)) - float(np.linalg.norm(np.subtract(end, q0)))) / len(end)
        self.turns.append((list(xyz), list(quat)))
        pts = [list(q0), end]
        for j in range(len(end)):
            pts += [end[:j] + [end[j] + h] + end[j + 1 :], end]
        return pts, None

    return walk


def test_a_turn_of_more_joint_arc_than_turn_max_arc_rad_is_refused_though_it_flies(ctx, monkeypatch, tmp_path):
    """The walks in from 16:23's re-aimed look that fly only by swinging
    joint_1 round run 2.65-3.21 rad of joint arc; every turn in place that
    flies at the 30 deg tilt flatbot's boxes plan at, 0.49-1.91. A turn in
    place longer than TURN_MAX_ARC_RAD is refused like one that does not
    fly — the next lean is tried — and a compliant press none of whose
    turns is short enough flies nothing. The walk in from the look pose
    has no such bound."""
    model = ContainerModel.load(PINK)
    quat = rp.reach_quat(model, BUTTON[:2], 30)
    start = rp.tool_for_contact((BUTTON[0], BUTTON[1], 0.46), quat)
    q = list(start) + list(quat)
    assert dp.joint_arc([[0.0, 0.0], [3.0, 4.0], [3.0, 0.0]]) == pytest.approx(9.0)
    assert dp.joint_arc([q]) == 0.0
    monkeypatch.setattr(_Arm, "walk", _detour(lambda lean: dp.TURN_MAX_ARC_RAD + 0.3))
    pts, down, short, why = dp.turn_down_path(_Arm(), model, q, rp.CONTACT_TOOL, dp.COMPLIANT_TURN_SHORT_DEG)
    assert pts is None and why == "; ".join(
        "%.0f deg short: a turn of 2.30 rad of joint arc" % s for s in dp.COMPLIANT_TURN_SHORT_DEG
    )
    # long at the first leans, within the bound from 8 deg short: turned 8 deg short
    monkeypatch.setattr(_Arm, "walk", _detour(lambda lean: 2.4 if lean < 7.0 else dp.TURN_MAX_ARC_RAD - 0.05))
    pts, down, short, why = dp.turn_down_path(_Arm(), model, q, rp.CONTACT_TOOL, dp.COMPLIANT_TURN_SHORT_DEG)
    assert why is None and short == 8.0 and dp.joint_arc(pts) == pytest.approx(dp.TURN_MAX_ARC_RAD - 0.05)
    # the walk in from the look pose: as long as it flies
    monkeypatch.setattr(_Arm, "walk", _detour(lambda lean: 2.7))
    pts, down, short, why = dp.walk_down_path(_Arm(), model, q, start, quat)
    assert why is None and short == 0.0 and dp.joint_arc(pts) == pytest.approx(2.7)
    # a compliant press: no turn short enough, nothing flown
    ctx.client.compliant = True
    result, events, _arm, _legs, _s = _press(
        ctx, monkeypatch, tmp_path, [], touch_at=None, compliant=True, top_z=COMPLIANT_TOP
    )
    assert events == [] and ctx.client.executed == []
    assert result.free and result.why == "no straight-down hand reaches here (%s)" % "; ".join(
        "%.0f deg short: a turn of 2.70 rad of joint arc" % s for s in dp.COMPLIANT_TURN_SHORT_DEG
    )


def test_a_walk_the_runner_would_refuse_is_no_walk_and_the_next_lean_is_tried(ctx, monkeypatch, tmp_path):
    """The 2026-10-28 merge review of #53 + #55: a walk the arm's IK solves
    can swing a joint out and back further than its net change and
    SANITY_MARGIN_RAD, and the runner refuses that leg before it moves
    (Runner._refusal). turn_walk takes a lean's walk only where the
    runner's gate passes the leg it flies as — in the runner's own words
    where it does not — and tries the next lean."""
    from box_opening.constants import FREE_LINE_SPEED, SANITY_MARGIN_RAD
    from box_opening.primitives.core import _full_world
    from box_opening.runtime import approach
    from box_opening.runtime.runner import Runner

    model = ContainerModel.load(PINK)
    quat = rp.reach_quat(model, BUTTON[:2], 30)
    start = rp.tool_for_contact((BUTTON[0], BUTTON[1], 0.46), quat)
    q = list(start) + list(quat)
    walks = {}

    def walk(self, q0, xyz, quat_, step_m=None, step_rad=None, max_step_rad=None):
        # joint 1 out past both ends and back: refused exactly down, not 2 deg short
        end = [float(v) for v in xyz] + [float(v) for v in quat_]
        lean = round(_lean_deg(quat_))
        out = list(end)
        out[0] = max(q0[0], end[0]) + SANITY_MARGIN_RAD + (0.05 if lean == 0 else -0.05)
        walks[lean] = [list(q0), out, end]
        return walks[lean], None

    monkeypatch.setattr(_Arm, "walk", walk)
    arm = _Arm()
    monkeypatch.setattr(approach, "_chain", lambda: arm)
    pts, down, short, why = dp.walk_down_path(arm, model, q, start, quat)
    assert why is None and short == 2.0 and pts == walks[2]
    runner = Runner(ctx.client, log_dir=tmp_path)

    def refusal(path):
        return runner._refusal(dp.compliant_line(ctx, "reach:slide-in", path, down, FREE_LINE_SPEED, _full_world(ctx)))

    assert refusal(pts) is None
    gate = refusal(walks[0])
    assert gate.startswith("trajectory sanity gate: joint_1 excursion ")
    tip = np.asarray(start) + rp._rot(quat) @ np.asarray(rp.CONTACT_TOOL)  # walk_down_path's
    assert dp.turn_walk(arm, model, q, tip, (0.0,)) == (None, None, None, "0 deg short: " + gate)


def _swung(q0, end):
    """A line from `q0` to `end` by way of joint 1 out past both ends and
    back, 0.05 rad further than the runner's sanity gate allows."""
    from box_opening.constants import SANITY_MARGIN_RAD

    out = list(end)
    out[0] = max(q0[0], end[0]) + SANITY_MARGIN_RAD + 0.05
    return [list(q0), out, list(end)]


def test_a_way_in_whose_slide_stroke_or_move_the_runner_would_refuse_is_no_way_in(ctx, monkeypatch):
    """The slide before a turn, the stroke and the move to the aimed
    button's look pose fly as legs of their own too (rp.way_in_legs), each
    under the runner's sanity gate: the proof takes none it refuses — no
    way down, or no straight line to the look pose, which the planner may
    plan instead."""
    from box_opening.runtime import approach

    ctx.model = ContainerModel.load(PINK)
    cfg = load_press_demo(PINK)
    quat, start, _bottom, look = rp.reach_geometry(ctx.model, BUTTON, 30, cfg.travel_m)
    q_look, q_start = list(look) + list(quat), list(start) + list(quat)

    def way_down(slide):
        arm = _Arm(refuse_first=len(dp.TURN_SHORT_DEG))  # no walk in: the slide, then the turn
        monkeypatch.setattr(approach, "_chain", lambda: arm)
        return rp.way_down(ctx, cfg, q_look, slide, start, quat, BUTTON[2])

    way, stroke, why = way_down([q_look, q_start])
    assert (
        why is None and not way.walked and [name for name, _p, _s in stroke.legs] == ["depth:approach", "depth:press"]
    )
    way, stroke, why = way_down(_swung(q_look, q_start))
    assert way is None and why.startswith("the slide in: trajectory sanity gate: joint_1 excursion "), why
    monkeypatch.setattr(_Arm, "straight_line", lambda self, q0, xyz, quat_: (_swung(q0, list(xyz) + list(quat_)), None))
    way, stroke, why = way_down([q_look, q_start])
    assert way is None and why.startswith("depth:approach: trajectory sanity gate: joint_1 excursion "), why
    q_off = [look[0] - 0.02] + list(look[1:]) + list(quat)  # 20 mm off the aimed button's look pose
    way, why, kind = rp.solve_way_in(ctx, cfg, BUTTON, 30, q_off)
    assert way is None and kind == "lines"
    assert why.startswith("the move to its look pose (trajectory sanity gate: joint_1 excursion "), why


def _leans_tried(monkeypatch):
    """_Arm.walk refusing every walk, the lean of each it was asked for kept."""
    leans = []

    def walk(self, q0, xyz, quat, step_m=None, step_rad=None, max_step_rad=None):
        leans.append(round(_lean_deg(quat), 6))
        return None, "IK did not converge 70% of the way"

    monkeypatch.setattr(_Arm, "walk", walk)
    return leans


def test_the_guarded_press_turns_no_further_than_5_deg_short(ctx, monkeypatch, tmp_path):
    """Its close-up look and DEPTH assume the lens points down along the
    fingers, and 5 deg short already puts the button 14 mm off the tool
    axis: the guarded press tries TURN_SHORT_DEG only — (0, 2, 5) — and
    where none flies it stops with nothing flown. A compliant press there
    tries every one of COMPLIANT_TURN_SHORT_DEG, in order."""
    leans = _leans_tried(monkeypatch)
    result, events, _arm, _l, _s = _press(ctx, monkeypatch, tmp_path, [])
    assert dp.TURN_SHORT_DEG == (0.0, 2.0, 5.0) and leans == [0.0, 2.0, 5.0]
    assert events == [] and result.free and result.why.startswith("no straight-down hand reaches here (")
    assert "8 deg short" not in result.why
    leans.clear()
    ctx.client.compliant = True
    result, events, _arm, _l, _s = _press(
        ctx, monkeypatch, tmp_path, [], touch_at=None, compliant=True, top_z=COMPLIANT_TOP
    )
    assert leans == list(dp.COMPLIANT_TURN_SHORT_DEG) and events == [] and result.free
    # smallest lean first, the guarded ones first of all: what they reach keeps its lean
    every = dp.COMPLIANT_TURN_SHORT_DEG
    assert every[:3] == dp.TURN_SHORT_DEG and list(every) == sorted(every) and every[-1] == 15.0


def test_the_walk_in_tries_only_the_guarded_leans_in_a_compliant_run_too(ctx, monkeypatch, tmp_path):
    """From 16:23's re-aimed look the walks 10-15 deg short "fly" only by
    swinging joint_1 1.6-1.9 rad, the elbow travelling 247-312 mm:
    walk_in_down, and way_down's walk that proves it, try TURN_SHORT_DEG
    only and carry no arc bound, in a compliant run too. Only the turn
    after the slide — way_down's proof of it, the scene camera's button's
    and the aimed one's — leans further."""
    import test_reach_press as trp

    from box_opening.runtime import approach

    tried = []
    real = dp.turn_walk

    def turn_walk(chain, model, q, tip, shorts=dp.TURN_SHORT_DEG, max_arc_rad=None):
        tried.append((tuple(shorts), max_arc_rad))
        return real(chain, model, q, tip, shorts, max_arc_rad)

    monkeypatch.setattr(dp, "turn_walk", turn_walk)
    turns = []

    def turn(chain, model, q, contact_tool, shorts=dp.TURN_SHORT_DEG):
        turns.append((tuple(shorts), tuple(contact_tool)))
        return _turns_down(chain, model, q, contact_tool, shorts)

    # the fake chain's walks all refuse: the scene button's proof and the
    # aimed one's — each a walk, then the turn after the slide about the
    # fingertip — and the flight flies the aimed one's, solving none again
    code, rec, _chain, _h, got = _depth_flow(ctx, monkeypatch, tmp_path, _popped, compliant=True, turn=turn)
    assert code == 0 and got["client_compliant"]
    assert rec.events == [
        "reach:look",
        "aim:reach-aim",
        "reach:re-aim",
        "reach:slide-in",
        "depth:turn",
        "depth:approach",
        "depth:press",
        "depth:clear",
        "home",
    ]
    assert tried == [(dp.TURN_SHORT_DEG, None)] * 2
    assert turns == [(dp.COMPLIANT_TURN_SHORT_DEG, rp.CONTACT_TOOL)] * 2
    # walk_in_down itself, from the look pose: the guarded leans and no others
    leans = _leans_tried(monkeypatch)
    arm = _Arm()
    monkeypatch.setattr(approach, "_chain", lambda: arm)
    quat, start, _bottom, look = rp.reach_geometry(ctx.model, trp.BUTTON, 30, load_press_demo(PINK).travel_m)
    ctx.client.live = list(look) + list(quat)
    pts, down, why = dp.walk_in_down(ctx, start, quat)
    assert pts is None and down is None and leans == [0.0, 2.0, 5.0]
    assert why == "; ".join("%.0f deg short: IK did not converge 70%% of the way" % s for s in dp.TURN_SHORT_DEG)


def test_on_flatbot_1623s_aimed_button_is_pressed_turning_12_deg_short_after_the_slide(ctx, monkeypatch):
    """16:23 2026-10-27 replayed on flatbot's URDF: the look the planner
    took for the scene camera's button, then the re-aim and the slide in to
    the aimed one as straight lines, the slid-in fingertip 0.780 m out.
    There the turn with TURN_SHORT_DEG refuses, as it did live ("no
    straight-down hand reaches here (0 deg short: IK did not converge 73%
    of the way; ...)"), and so does the walk in from the re-aimed look.
    With COMPLIANT_TURN_SHORT_DEG it flies 12 deg short: 11 points, every
    step within TURN_MAX_JUMP_RAD, 1.46 rad of joint arc, about the
    fingertip — the pads' centre where the fingertip was, no point of the
    pads nearer the lid than 5 cm. The compliant stroke flies from its end,
    straight down to the button's bottom, RIGID's spring ~1320 N/m at the
    pads there (~1150 at the turn's end; STIFF's 523 and 457), then bent
    on."""
    from box_opening.constants import HOME
    from box_opening.primitives.look import look_joints
    from box_opening.runtime import approach
    from box_opening.runtime.branches import planner_branch
    from box_opening.runtime.kinematics import ArmChain

    chain = ArmChain(path="capabilities/box_opening/test/data/arm_flatbot_20261021.urdf", root="arm_mount")
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    model = ContainerModel.load(PINK)
    cfg = load_press_demo(PINK)
    scene, aimed = (0.7248, -0.2860, 0.3125), (0.7213, -0.2970, 0.3125)
    quat_s, _start_s, _bottom_s, look_s = rp.reach_geometry(model, scene, 30, cfg.travel_m)
    # the look pose's joints as client._plan_pose_near picks them
    q_look = chain.nearest_solution(planner_branch(HOME), look_s, quat_s, seeds=(look_joints(HOME),))
    assert q_look is not None
    quat, start, _bottom, look = rp.reach_geometry(model, aimed, 30, cfg.travel_m)
    move, why = chain.straight_line(q_look, look, quat)  # the re-aim
    assert why is None, why
    slide, why = chain.straight_line(move[-1], start, quat)
    assert why is None, why
    R, t = chain.fk(slide[-1])
    tip = np.asarray(t) + np.asarray(R) @ np.asarray(rp.CONTACT_TOOL)
    assert np.hypot(*tip[:2]) == pytest.approx(0.780, abs=0.001)
    # what 16:23 had: no turn, and no walk in either
    pts, _down, _short, why = dp.turn_down_path(chain, model, slide[-1], rp.CONTACT_TOOL, dp.TURN_SHORT_DEG)
    assert pts is None and why.startswith("0 deg short: IK did not converge"), why
    pts, _down, _short, why = dp.walk_down_path(chain, model, move[-1], start, quat)
    assert pts is None, why
    # a compliant press's leans
    pts, down, short, why = dp.turn_down_path(chain, model, slide[-1], rp.CONTACT_TOOL, dp.COMPLIANT_TURN_SHORT_DEG)
    assert why is None and max(dp.TURN_SHORT_DEG) < short <= 15.0, why
    assert short == 12.0 and len(pts) == 11  # as modelled before this was flown
    assert dp.joint_arc(pts) <= dp.TURN_MAX_ARC_RAD and dp.joint_arc(pts) == pytest.approx(1.46, abs=0.02)
    assert (
        max(float(np.abs(np.asarray(b) - np.asarray(a)).max()) for a, b in zip(pts, pts[1:], strict=False))
        <= dp.TURN_MAX_JUMP_RAD
    )
    R1, t1 = chain.fk(pts[-1])
    R1 = np.asarray(R1, float)
    assert np.degrees(rp.hand_lean(R1)) == pytest.approx(12.0, abs=0.05)
    assert np.linalg.norm(np.asarray(t1) + R1 @ np.asarray(dp.DOWN_CONTACT_TOOL) - tip) < 0.001
    lowest = []
    for q in pts:
        Rq, tq = chain.fk(q)
        Rq = np.asarray(Rq, float)
        lowest.append(
            min(float((np.asarray(tq) + Rq @ np.asarray(p))[2]) for p in (rp.CONTACT_TOOL, dp.DOWN_CONTACT_TOOL))
        )
    assert min(lowest) - aimed[2] >= 0.05
    stroke, why = dp.compliant_stroke(chain, cfg, pts[-1], down, aimed[2])
    assert why is None, why
    assert 1000.0 <= stroke.k <= 1500.0 and [name for name, _p, _s in stroke.legs] == ["depth:approach", "depth:press"]
    bottom = aimed[2] - cfg.button_travel_m
    assert stroke.end_z < bottom
    for _name, p, _speed in stroke.legs:  # straight down from the turn's end, the lean held, to the button's bottom
        Rp, tp = chain.fk(_straight_end(chain, p, bottom))
        assert list(tp[:2]) == pytest.approx(list(t1[:2]), abs=1e-3)
        assert np.degrees(rp.hand_lean(np.asarray(Rp, float))) == pytest.approx(12.0, abs=0.1)
    # ... then bent on (dp.stroke_bend): the springs' reference 15 mm below the bottom — COMPLIANT_OVERSHOOT_M's
    # least: COMPLIANT_PRESS_N would drop it only 11 mm — and 7 mm aside, where they push ~19.8 N straight down on
    # the pads held at the bottom (STIFF: 28 mm, 14 mm aside, 15 N; 15:19's in
    # test_on_flatbot_a_bent_press_pushes_straight_down_where_a_straight_one_shoves_the_pads_aside)
    assert stroke.bent and stroke.bend_why is None and stroke.bend_m == pytest.approx(0.007, abs=0.001)
    assert stroke.over == pytest.approx(dp.COMPLIANT_OVERSHOOT_M[0], abs=0.0005)
    assert stroke.k == pytest.approx(1320.0, abs=20.0) and stroke.k * stroke.over == pytest.approx(19.8, abs=0.3)
    _held, f = _held_at(chain, stroke.q_end, _pads_at(chain, _straight_end(chain, stroke.legs[-1][1], bottom)))
    assert f == pytest.approx(
        [0.0, 0.0, 20.1], abs=0.1
    )  # the button's push on the pads: straight up (19.9 at wrist Kq 200)


# ---- a compliant run's clear before home: up as far as the lean needs, and back -----------


def _clear(ctx, monkeypatch, tmp_path, R, chain=None):
    """rp._clear_over_box in a compliant run, from the fake chain's
    (0.5, -0.4, 0.45) with the hand at `R`, the box at test_reach_press's
    BUTTON (its padded top at 0.44). Returns (what it returned, the legs it
    flew, the chain, the run record's depth_clear note or None)."""
    import json

    import test_reach_press as trp

    from box_opening.runtime import approach
    from box_opening.runtime.runner import Runner

    ctx.model = ContainerModel.load(PINK)
    ctx.cpose = rp.cpose_for_button(trp.BUTTON, 0.0, ctx.model.button_offset)
    ctx.client.compliant = True
    chain = chain or trp._Chain()
    chain.R = np.asarray(R, float)
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    flown = []

    class Rec(Runner):
        def run(self, legs, execute, lookahead=None):
            flown.extend(legs)
            return super().run(legs, execute, lookahead)

    got = rp._clear_over_box(ctx, Rec(ctx.client, log_dir=tmp_path), None)
    notes = [r for p in tmp_path.glob("run-*.jsonl") for r in map(json.loads, open(p)) if r.get("leg") == "depth_clear"]
    return got, flown, chain, notes[-1] if notes else None


def _tilted30():
    import test_reach_press as trp

    return rp._rot(rp.reach_quat(ContainerModel.load(PINK), trp.BUTTON[:2], 30))


def _back_of(xyz):
    """How far back a target stands from the fake fk's (0.5, -0.4)."""
    return float(np.hypot(xyz[0] - 0.5, xyz[1] + 0.4))


def test_a_compliant_clear_from_a_leaning_hand_rises_as_far_as_the_lean_needs_and_backs_toward_the_arm(
    ctx, monkeypatch, tmp_path
):
    """16:23 2026-10-27: the hand 30 deg off straight down, the clear rose to
    HOME_CLEAR_M over the padded cuboid, and the planner's lowest gripper
    spheres, 0.045 sin 30 deg lower than straight down, stood 7 mm inside
    it: home INVALID_START, then a 15 cm back-off. Now one line, the hand's
    attitude held, to HOME_CLEAR_M + 0.045 sin 30 deg + 0.005 over it and
    50 mm back toward the arm: unguarded, at COMPLIANT_FREE_LINE_SPEED, in
    the full world."""
    from box_opening.constants import COMPLIANT_FREE_LINE_SPEED
    from box_opening.worlds import planned_top_z

    R30 = _tilted30()
    got, flown, chain, note = _clear(ctx, monkeypatch, tmp_path, R30)
    assert got is True
    (leg,) = flown
    assert leg.name == "depth:clear" and leg.plan_free
    assert leg.speed == COMPLIANT_FREE_LINE_SPEED and leg.guard is None and leg.world.startswith("full"), (
        leg.speed,
        leg.guard,
        leg.world,
    )
    ((xyz, quat),) = chain.lines
    u = np.asarray(ctx.cpose.xyz[:2]) / np.hypot(*ctx.cpose.xyz[:2])
    want = planned_top_z(ctx.model, ctx.cpose) + rp.HOME_CLEAR_M + 0.045 * np.sin(np.radians(30.0)) + 0.005
    assert xyz == pytest.approx([*(np.array([0.5, -0.4]) - 0.05 * u), want])
    assert rp._rot(quat) == pytest.approx(R30, abs=1e-9)
    assert note["flown"] and note["back_mm"] == pytest.approx(50.0) and note["lean_deg"] == pytest.approx(30.0)
    assert note["up_mm"] == pytest.approx(1000 * (want - 0.45), abs=0.1) and note["why"] is None


def test_where_the_first_line_up_and_back_does_not_fly_the_clear_goes_back_as_far_as_the_look(
    ctx, monkeypatch, tmp_path
):
    """The second target is REACH_PULLBACK_M back, as high: where 16:23's
    planner took home after its 15 cm back-off."""
    import test_reach_press as trp

    first = lambda xyz, quat: abs(_back_of(xyz) - rp.CLEAR_BACK_M) < 1e-9  # noqa: E731
    got, flown, chain, note = _clear(ctx, monkeypatch, tmp_path, _tilted30(), chain=trp._Chain(refuse=[first]))
    assert got is True and [lg.name for lg in flown] == ["depth:clear"]
    (a, _qa), (b, _qb) = chain.lines
    assert _back_of(a) == pytest.approx(rp.CLEAR_BACK_M) and _back_of(b) == pytest.approx(rp.REACH_PULLBACK_M)
    assert b[2] == pytest.approx(a[2])
    assert note["flown"] and note["back_mm"] == pytest.approx(1000 * rp.REACH_PULLBACK_M)
    assert note["why"] == "50 mm back: IK did not converge"


@pytest.mark.parametrize(
    "rooms, flown_back",
    [
        ((0.160, 0.368), rp.REACH_PULLBACK_M),  # 15:19's box turned down at the scene camera's 0.3111 (flatbot's URDF)
        ((0.264, 0.444), rp.CLEAR_BACK_M),  # ... at the wrist aim's 0.2881: the first leaves room
        ((0.150, 0.170), rp.CLEAR_BACK_M),  # neither does: the first, and home's own clear-margin step after it
    ],
)
def test_a_clear_that_would_leave_a_joint_too_near_its_limit_for_home_goes_on_to_the_next_target(
    ctx, monkeypatch, tmp_path, capsys, rooms, flown_back
):
    """A line up and back that flies but ends with a joint nearer its limit
    than home plans from (CLEAR_MARGIN_ROOM_RAD, _home_legs) would cost the
    clear-margin step the clear is there to save: the next target is
    tried, the first that leaves room is flown — else the first that
    flies, and no line is solved past the one flown. The console says
    why, as the run log does."""
    import test_reach_press as trp

    from box_opening.primitives.core import CLEAR_MARGIN_ROOM_RAD

    class Folding(trp._Chain):
        """joint_6 ends `rooms` from its upper limit at the two targets."""

        limits = [(None, None)] * 5 + [(-2.09, 2.09), (None, None)]

        def straight_line(self, q0, xyz, quat):
            self.lines.append((list(xyz), list(quat)))
            end = list(q0)
            end[5] = 2.09 - rooms[0 if _back_of(xyz) < 0.1 else 1]
            return [list(q0), end], None

    got, flown, chain, note = _clear(ctx, monkeypatch, tmp_path, trp.STRAIGHT_DOWN, chain=Folding())
    assert got is True
    (leg,) = flown
    room = rooms[0 if flown_back == rp.CLEAR_BACK_M else 1]
    assert 2.09 - leg.goal_joints[5] == pytest.approx(room) and note["back_mm"] == pytest.approx(1000 * flown_back)
    assert len(chain.lines) == (1 if rooms[0] >= CLEAR_MARGIN_ROOM_RAD else 2)
    said = capsys.readouterr().out
    assert "up 55 mm and %.0f mm back toward the arm (the hand leaning 0 deg)" % (1000 * flown_back) in said
    if rooms[0] < CLEAR_MARGIN_ROOM_RAD:
        assert note["why"].startswith("50 mm back: joint_6 %.3f rad from its limit at its end" % rooms[0])
        assert "before home (%s)" % note["why"] in said
    else:
        assert note["why"] is None
        assert "padded volume, before home\n" in said


def test_a_hand_already_as_high_as_the_clear_goes_is_not_moved(ctx, monkeypatch, tmp_path):
    import test_reach_press as trp

    class High(trp._Chain):
        def fk(self, q):
            return np.array(self.R, float), np.array([0.5, -0.4, 0.60])

    got, flown, chain, note = _clear(ctx, monkeypatch, tmp_path, _tilted30(), chain=High())
    assert got is True and flown == [] and chain.lines == [] and note is None


@pytest.mark.parametrize("case", ["16:23 tilted", "15:19 turned down"])
def test_on_flatbot_the_compliant_clear_leaves_home_nothing_to_refuse_or_step_off(ctx, monkeypatch, tmp_path, case):
    """Flatbot's URDF, from 16:23's tilted slide end (the turn down refused
    there) and from 15:19's box with the hand walked in turned down at the
    scene camera's top, as a compliant run presses it. From each, home is
    refused as things stand: the planner's gripper spheres reach into the
    padded box. The first target's straight line flies, and the line the
    clear flies ends with the lowest of the four gripper-base spheres
    (robot_gen3_2f85.yaml arm_robotiq_85_base_link (+-0.045, 0, 0.145) and
    (0, +-0.045, 0.145), at end_effector_link, r 0.018 + buffer 0.005) 15
    mm or more over the padded cuboid, never lower on the way, and every
    limited joint CLEAR_MARGIN_ROOM_RAD or more from its limit: no
    INVALID_START, no clear-margin step. At 15:19's box the 50 mm line
    leaves arm_joint_6 0.160 rad: the 150 mm one is flown."""
    from box_opening.constants import HOME
    from box_opening.primitives.core import CLEAR_MARGIN_ROOM_RAD
    from box_opening.primitives.look import look_joints
    from box_opening.runtime import approach
    from box_opening.runtime.branches import planner_branch
    from box_opening.runtime.kinematics import TOOL_FROM_EE_M, ArmChain, mat_to_quat_xyzw
    from box_opening.runtime.runner import Runner
    from box_opening.worlds import planned_top_z

    chain = ArmChain(path="capabilities/box_opening/test/data/arm_flatbot_20261021.urdf", root="arm_mount")
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    model = ContainerModel.load(PINK)
    ctx.model = model
    tilted = case.startswith("16:23")
    button, yaw_deg = ((0.7213, -0.2970, 0.3125), 36.9) if tilted else ((0.6617, -0.1972, 0.3111), 85.9)
    quat, start, _bottom, look = rp.reach_geometry(model, button, 30, load_press_demo(PINK).travel_m)
    # the look pose's joints as client._plan_pose_near picks them
    q_look = chain.nearest_solution(planner_branch(HOME), look, quat, seeds=(look_joints(HOME),))
    ctx.client.live = [float(v) for v in q_look]
    if tilted:
        pts, why = chain.straight_line(ctx.client.live, start, quat)  # the slide in
    else:
        pts, _down, why = dp.walk_in_down(ctx, start, quat)
    assert why is None, why
    ctx.client.live = q0 = [float(v) for v in pts[-1]]
    ctx.cpose = rp.cpose_for_button(button, np.radians(yaw_deg), model.button_offset)
    ctx.client.compliant = True
    top = planned_top_z(model, ctx.cpose)

    def lowest(q):  # the four gripper-base spheres' lowest point
        R, t = chain.fk(q)
        return min(
            float((np.asarray(t) + np.asarray(R) @ np.r_[x, y, 0.145 - TOOL_FROM_EE_M])[2])
            for x, y in ((0.045, 0.0), (-0.045, 0.0), (0.0, 0.045), (0.0, -0.045))
        ) - (0.018 + 0.005)

    def room(q):
        return min(min(q[j] - lo, hi - q[j]) for j, (lo, hi) in enumerate(chain.limits) if lo is not None)

    assert lowest(q0) < top  # among the padded box: home from here is INVALID_START
    R, t = chain.fk(q0)
    first, second = rp.clear_targets(ctx, R, t)
    line, why = chain.straight_line(q0, first, mat_to_quat_xyzw(R))
    assert why is None, why
    assert lowest(line[-1]) - top >= 0.015
    flown = []

    class Rec(Runner):
        def run(self, legs, execute, lookahead=None):
            flown.extend(legs)
            return super().run(legs, execute, lookahead)

    assert rp._clear_over_box(ctx, Rec(ctx.client, log_dir=tmp_path), None) is True
    (leg,) = flown
    path = [list(p.positions) for p in leg.traj.points]
    assert lowest(path[-1]) - top >= 0.015 and room(path[-1]) >= CLEAR_MARGIN_ROOM_RAD, (
        lowest(path[-1]) - top,
        room(path[-1]),
    )
    assert min(lowest(q) for q in path) >= lowest(q0) - 1e-4  # it only rises
    _R, end = chain.fk(path[-1])
    if tilted:
        assert list(end) == pytest.approx(first, abs=1e-4)
    else:
        assert room(line[-1]) < CLEAR_MARGIN_ROOM_RAD  # the first: arm_joint_6 0.160 rad from its limit
        assert list(end) == pytest.approx(second, abs=1e-4)


# ---- a compliant reach-out's way home: from the hold to home as one execution -------------


class _PadsChain:
    """A chain whose first three joints ARE the pads' position: fk puts the
    tool DOWN_CONTACT_TOOL above them, the hand's axes the world's. `sag(q)`
    lowers the pads by that much more (m)."""

    def __init__(self, sag=None):
        self.sag = sag

    def fk(self, q):
        t = np.asarray(q[:3], float) - np.asarray(dp.DOWN_CONTACT_TOOL, float)
        if self.sag is not None:
            t[2] -= self.sag(q)
        return np.eye(3), t


def _stroke_100mm(j4=0.0):
    """A 100 mm compliant stroke on _PadsChain, the pads 2 mm a sample from
    0.50 down to 0.40: the approach to 0.44, then the press, joint_4 at
    `j4` all the way."""
    q = [[0.6, -0.2, round(0.50 - 0.002 * k, 6), j4, 0.0, 0.0, 0.0] for k in range(51)]
    legs = [("depth:approach", q[:31], 0.35), ("depth:press", q[30:], dp.TOUCH_SPEED)]
    pad0 = np.asarray(q[0][:3], float)
    return dp.StrokePlan(
        legs, 440.0, 0.03, 0.40, pad0 - np.asarray(dp.DOWN_CONTACT_TOOL), pad0, list(q[0]), list(TURNED_DOWN)
    )


# the arm the spring held: the pads 30 mm above the stroke's end and a little aside, other joints off it too
HELD = [0.602, -0.201, 0.43, 0.02, -0.03, 0.01, 0.0]


def test_the_way_back_up_from_a_compliant_hold_joins_the_stroke_15_mm_above_the_held_pads():
    """The spring holds the arm 0.040-0.052 rad off the stroke's commanded
    end (flatbot's URDF, STIFF), past the runner's DRIFT_REPLAN_RAD: a way
    up from there is refused, and the arm stays pressing on the button. The
    stroke's sample nearest the held arm lies near its end, and a way up
    joined there presses 23-30 mm down again first. The way back up starts at
    the held arm, joins the stroke at its first sample RISE_JOIN_ABOVE_M or
    more above the held pads, and retraces the stroke to its first sample,
    exactly stroke.q0: where the clear planned while the way in flew
    starts. A 100 mm stroke, the pads held 30 mm above its end."""
    stroke = _stroke_100mm()
    rise = dp.rise_back_up(_PadsChain(), stroke, HELD)
    assert rise[0] == pytest.approx(HELD)
    assert rise[1][2] == pytest.approx(0.446)  # 16 mm up; the sample before it, 0.444, is 14 mm
    assert rise[1][2] - HELD[2] >= dp.RISE_JOIN_ABOVE_M > rise[1][2] - 0.002 - HELD[2]
    assert rise[1:] == [q for q in stroke.legs[0][1] if q[2] >= 0.446 - 1e-9][::-1]  # the stroke's own samples
    assert rise[-1] == stroke.q0 == stroke.legs[0][1][0]  # exactly its first sample
    assert len(rise) == 1 + 28  # 0.446 to 0.500 in 2 mm, the junction of the approach and the press once
    # never below the held pads: at its points, nor along the step from the held arm to the stroke
    assert min(q[2] for q in rise) == pytest.approx(HELD[2])
    step = np.subtract(rise[1], rise[0])
    assert all(float((np.asarray(rise[0]) + s * step)[2]) >= HELD[2] - 1e-12 for s in np.linspace(0.0, 1.0, 21))
    assert float(np.abs(step).max()) <= dp.RISE_FIRST_STEP_MAX_RAD


def test_a_way_back_up_with_a_long_first_step_or_nothing_to_join_is_not_one():
    """A step from the held arm to the stroke that moves a joint more than
    RISE_FIRST_STEP_MAX_RAD is no rise to fly blind; pads held so high that
    no sample stands RISE_JOIN_ABOVE_M over them have no stroke to join.
    Either is None, and the rise is a straight line from where the arm
    stands."""
    stroke = _stroke_100mm()
    assert dp.rise_back_up(_PadsChain(), stroke, HELD[:3] + [0.24] + HELD[4:]) is not None
    assert dp.rise_back_up(_PadsChain(), stroke, HELD[:3] + [0.26] + HELD[4:]) is None
    assert dp.rise_back_up(_PadsChain(), stroke, HELD[:2] + [0.4849] + HELD[3:]) is not None  # the start, 15.1 mm up
    assert dp.rise_back_up(_PadsChain(), stroke, HELD[:2] + [0.4851] + HELD[3:]) is None  # ... 14.9 mm


def test_a_way_back_up_whose_step_to_the_stroke_dips_the_pads_is_not_one():
    """A step that undoes the spring's deflection by lowering the pads on
    the way — here joint_4 sagging them by q4 (0.3 - q4): from the arm held
    at joint_4 0.1 to the stroke at 0.3 it passes 0.15, 2.5 mm lower than
    held — would press again. RISE_CHECK_POINTS along the step find it."""
    sag = _PadsChain(sag=lambda q: float(q[3]) * (0.3 - float(q[3])))
    stroke = _stroke_100mm(j4=0.3)
    held = [0.6, -0.2, 0.45, 0.1, 0.0, 0.0, 0.0]  # the pads at 0.45 - 0.02: 0.43
    assert dp.rise_back_up(sag, stroke, held) is None
    held = [0.6, -0.2, 0.43, 0.3, 0.0, 0.0, 0.0]  # held on the stroke's joint_4: no dip
    assert dp.rise_back_up(sag, stroke, held)[1][2] == pytest.approx(0.446)


def test_the_held_arm_is_written_on_the_strokes_side_of_the_wrap():
    """The arm's reading is wrapped into (-pi, pi], the stroke's samples
    are not: a continuous joint read a whole turn from the stroke is the
    same arm, and the rise starts on the stroke's branch (here joint_1,
    which _PadsChain reads as the pads' x)."""
    stroke = _stroke_100mm()
    wrapped = [HELD[0] + 2 * np.pi] + HELD[1:]
    rise = dp.rise_back_up(_PadsChain(), stroke, wrapped)
    assert rise[0][0] == pytest.approx(HELD[0])
    assert np.asarray(rise) == pytest.approx(np.asarray(dp.rise_back_up(_PadsChain(), stroke, HELD)))


def _flatbot_turned_down(scene, aimed):
    """A compliant reach-out on flatbot's URDF as 10-27's flew: the look the
    planner took for the scene camera's button `scene`, the re-aim and the
    slide in to the `aimed` one as straight lines at 30 deg, then the turn
    down after the slide (COMPLIANT_TURN_SHORT_DEG). Returns (chain, model,
    cfg, the turn's points, the turned hand's quat)."""
    from box_opening.constants import HOME
    from box_opening.primitives.look import look_joints
    from box_opening.runtime.branches import planner_branch
    from box_opening.runtime.kinematics import ArmChain

    chain = ArmChain(path="capabilities/box_opening/test/data/arm_flatbot_20261021.urdf", root="arm_mount")
    model, cfg = ContainerModel.load(PINK), load_press_demo(PINK)
    quat_s, _start_s, _bottom_s, look_s = rp.reach_geometry(model, scene, 30, cfg.travel_m)
    q_look = chain.nearest_solution(planner_branch(HOME), look_s, quat_s, seeds=(look_joints(HOME),))
    quat, start, _bottom, look = rp.reach_geometry(model, aimed, 30, cfg.travel_m)
    move, _why = chain.straight_line(q_look, look, quat)
    slide, _why = chain.straight_line(move[-1], start, quat)
    turn, down, _short, why = dp.turn_down_path(chain, model, slide[-1], rp.CONTACT_TOOL, dp.COMPLIANT_TURN_SHORT_DEG)
    assert why is None, why
    return chain, model, cfg, turn, down


def test_on_flatbot_a_bent_press_pushes_straight_down_where_a_straight_one_shoves_the_pads_aside(
    ctx, monkeypatch, tmp_path
):
    """15:19 2026-10-27 on flatbot's URDF: the hand turned exactly down over
    the aimed button and the compliant stroke solved from there, the arm
    held by the button's bottom (_held_at: the driver's joint spring at
    rest, the pads pinned on the bottom under the aim). Flown straight on
    down, as before the bend (BEND_MAX_M 0), RIGID's springs push the pads
    19.5 N down and 6.2 N sideways there (STIFF: 16.7 and 5.1) — free to
    slide, across the 44 mm button. Bent (dp.stroke_bend), straight down
    and nothing sideways: the held arm is the bend's start (to the straight
    line's IK tolerance), and the hold (dp.compliant_hold) reads it so. k at
    the bottom is 1166 N/m (STIFF: 467): COMPLIANT_PRESS_N would drop the
    reference only 13 mm, so it drops COMPLIANT_OVERSHOOT_M's least, 15 mm,
    and the springs push 17.5 N. The bend moves the pads' reference 5 mm
    aside (STIFF: 11 mm, 32 mm below), its end 0.022 rad off the bottom's
    joints (STIFF: 0.046)."""
    import test_reach_press as trp

    chain, _model, cfg, turn, down = _flatbot_turned_down(trp.SCENE_1519, trp.AIMED_1519)
    assert rp.hand_lean(rp._rot(down)) < 1e-6  # exactly down
    bottom = trp.AIMED_1519[2] - cfg.button_travel_m
    real = dp.BEND_MAX_M
    pushes, notes = {}, {}
    for bent in (False, True):
        monkeypatch.setattr(dp, "BEND_MAX_M", real if bent else 0.0)
        stroke, why = dp.compliant_stroke(chain, cfg, turn[-1], down, trp.AIMED_1519[2])
        assert why is None and stroke.bent == bent, why
        held, f = _held_at(chain, stroke.q_end, [stroke.pad0[0], stroke.pad0[1], bottom])
        pushes[bent] = -f  # the pads' push on the button
        (tmp_path / str(bent)).mkdir()
        result, notes[bent] = _hold_note(ctx, monkeypatch, tmp_path / str(bent), chain, stroke, held)
        assert result.press_n == pytest.approx(f[2], abs=0.01) and notes[bent]["felt"]
        assert notes[bent]["sideways_n"] == pytest.approx(np.hypot(*f[:2]), abs=0.06)
    straight, bent = pushes[False], pushes[True]
    # a straight press shoves the pads 9.2 N aside with the wrist at Kq 400 (6.2 at 200); the bent
    # one, which flies, none
    assert -straight[2] == pytest.approx(20.3, abs=0.3) and np.hypot(*straight[:2]) == pytest.approx(9.2, abs=0.3)
    assert -bent[2] == pytest.approx(17.5, abs=0.1) and np.hypot(*bent[:2]) < 0.05
    assert -bent[2] == pytest.approx(stroke.k * stroke.over, abs=0.1)  # the bend's force, scaled to its bound
    q_b = np.asarray(_straight_end(chain, stroke.legs[-1][1], bottom), float)
    assert np.abs(held - q_b).max() < 1e-4  # held at the bottom, the arm stands where the bend starts
    assert notes[True]["press_n"] == pytest.approx(17.5) and notes[True]["sideways_n"] == 0.0
    assert notes[True]["bent"] and notes[True]["held_off_aim_mm"] == pytest.approx([0.0, 0.0], abs=0.1)
    assert stroke.k == pytest.approx(1166.0, abs=5.0)
    assert stroke.over == pytest.approx(dp.COMPLIANT_OVERSHOOT_M[0], abs=0.0005)
    assert stroke.bend_m == pytest.approx(0.005, abs=0.001) and notes[True]["bend_mm"] == pytest.approx(5.0, abs=1.0)
    assert float(np.abs(np.asarray(stroke.q_end) - q_b).max()) == pytest.approx(0.022, abs=0.003)
    assert (
        "PRESS (compliant): straight down 72 mm to the button's bottom, its reference on 15 mm below it and 5 mm "
        "aside, where the springs push ~17 N straight down on the pads held there (1166 N/m at the pads); held "
        "0.3 s"
    ) in dp.press_text(stroke, cfg)


def _flatbot_1623_hold():
    """16:23's aimed button on flatbot's URDF, the hand turned 12 deg short
    after the slide and the compliant stroke solved from there (as in
    test_on_flatbot_1623s_aimed_button_is_pressed_turning_12_deg_short_after_the_slide),
    and the arm held by the button: the joint springs (dp.COMPLIANT_KQ as it
    stands: _pin_stiff) about the stroke's end (stroke.q_end), the pads
    pinned where the stroke's straight part met the button's bottom
    (_held_at). Returns (chain, model, cfg, the aimed
    button, the stroke, the held joints, pads(q): the pads' position at
    joints q)."""
    aimed = (0.7213, -0.2970, 0.3125)
    chain, model, cfg, turn, down = _flatbot_turned_down((0.7248, -0.2860, 0.3125), aimed)
    stroke, why = dp.compliant_stroke(chain, cfg, turn[-1], down, aimed[2], kq=dp.COMPLIANT_KQ)
    assert why is None, why

    def pads(q):
        return _pads_at(chain, q)

    b = pads(_straight_end(chain, stroke.legs[-1][1], aimed[2] - cfg.button_travel_m))
    held, _f = _held_at(chain, stroke.q_end, b)
    return chain, model, cfg, aimed, stroke, held, pads


@pytest.mark.parametrize("spring", ["RIGID", "STIFF"])
@pytest.mark.parametrize("bent", [True, False])
def test_on_flatbot_the_way_back_up_from_1623s_hold_only_rises(monkeypatch, bent, spring):
    """16:23's press on flatbot's URDF, the arm held by the button's bottom
    (_flatbot_1623_hold), under the RIGID spring that flies and STIFF
    (_pin_stiff), which the way back up was laid out on. Bent
    (dp.stroke_bend), the held arm IS the stroke's own sample there, where
    the bend starts — the springs' static balance — 0.018 rad off the
    commanded end (STIFF: 0.035). Flown straight on down, as a bend past
    BEND_MAX_M flies, under STIFF it stands past DRIFT_REPLAN_RAD from the
    commanded end, and the stroke's sample nearest it in joint space 15 mm
    and more below the held pads (20 mm); RIGID holds it 0.023 rad off,
    inside the drift gate, its nearest sample still 12 mm below, a second
    press. Either way the way back up joins the stroke 15-17 mm above them,
    its step there well inside RISE_FIRST_STEP_MAX_RAD, the pads never
    lower than held on it."""
    from box_opening.constants import DRIFT_REPLAN_RAD

    if spring == "STIFF":
        _pin_stiff(monkeypatch)
    if not bent:
        monkeypatch.setattr(dp, "BEND_MAX_M", 0.0)
    chain, _model, _cfg, _aimed, stroke, held, pads = _flatbot_1623_hold()
    assert stroke.bent == bent
    q_end = np.asarray(stroke.q_end, float)
    assert list(q_end) == [float(v) for v in stroke.legs[-1][1][-1]]  # the stroke's last joints
    floor = float(pads(held)[2])
    assert floor - float(pads(q_end)[2]) == pytest.approx(stroke.over, abs=0.002)
    samples = np.asarray([q for _n, pts, _s in stroke.legs for q in pts], float)
    nearest = samples[int(np.argmin(np.linalg.norm(samples - held, axis=1)))]
    if bent:
        off = {"RIGID": 0.018, "STIFF": 0.035}[spring]
        assert float(np.abs(held - q_end).max()) == pytest.approx(off, abs=0.003)
        assert float(np.abs(nearest - held).max()) < 1e-6  # joined there, no lower first
    elif spring == "STIFF":
        assert float(np.abs(held - q_end).max()) > DRIFT_REPLAN_RAD
        assert floor - float(pads(nearest)[2]) > 0.015  # joined there, a second press (20 mm)
    else:
        assert float(np.abs(held - q_end).max()) == pytest.approx(0.023, abs=0.003)
        assert float(np.abs(held - q_end).max()) < DRIFT_REPLAN_RAD
        assert floor - float(pads(nearest)[2]) == pytest.approx(0.012, abs=0.002)  # joined there, a second press
    rise = dp.rise_back_up(chain, stroke, list(held))
    assert rise is not None and rise[-1] == [float(v) for v in stroke.q0]
    assert 0.015 <= float(pads(rise[1])[2]) - floor <= 0.017
    step = np.subtract(rise[1], rise[0])
    assert float(np.abs(step).max()) < 0.2
    assert all(float(pads(np.asarray(rise[0]) + s * step)[2]) >= floor for s in np.linspace(0.0, 1.0, 51))


@pytest.mark.parametrize("spring", ["RIGID", "STIFF"])
@pytest.mark.parametrize("pinned", [False, True], ids=["free air", "pinned at the bottom"])
@pytest.mark.parametrize("case", ["15:19", "16:23", "0.76 m out"])
def test_on_flatbot_a_compliant_press_rises_back_up_its_bent_stroke(ctx, monkeypatch, tmp_path, case, pinned, spring):
    """press_compliant's own rise (dp.compliant_rise: the reach-out's
    leg-by-leg press) on flatbot's URDF, the hand turned down after the
    slide (_flatbot_turned_down) and the stroke bent (dp.stroke_bend), under
    the RIGID spring that flies and STIFF (_pin_stiff), which the defect
    below was found on. The bend's end is no IK solution: in free air —
    nothing under the pads, the arm at the stroke's end — under STIFF the
    hand stands 3-6 deg off the attitude the stroke held, and a straight
    line back to its start takes that whole turn in its first 2 mm step:
    0.11 rad at 15:19's box, 0.13 at 16:23's, refused past MAX_STEP_RAD
    0.76 m out, at (0.75, -0.15) under a lid at 0.31 — the arm held over
    the box, let go, and never went home (the 2026-10-28 review of #60).
    RIGID's bend is about half as deep and aside: the hand 1.8-2.2 deg off,
    the line's first step 0.05, 0.07 and 0.07 rad, none refused, still the
    whole turn in one step. Under either it rises back up its own stroke
    (rise_back_up), as the one-execution way home does: one goal under
    COMPLIANT_GAINS from where the arm stands to exactly the stroke's
    start, through the stroke's own samples, joined 16-19 mm above the
    pads, its first step under 0.03 rad, the pads never lower than they
    stood, nothing let go. Pinned by the button's bottom (_held_at), where
    the straight line flies, it goes the same way, joined 16 mm up, its
    first step under 0.07 rad."""
    import test_reach_press as trp

    from box_opening.runtime import approach
    from box_opening.runtime.runner import Runner

    scene, aimed, yaw_deg, first = {  # the straight line's first step from the free-air end, rad
        "15:19": (trp.SCENE_1519, trp.AIMED_1519, 85.9, {"STIFF": 0.114, "RIGID": 0.051}),
        "16:23": (trp.SCENE_1623, (0.7213, -0.2970, 0.3125), 36.9, {"STIFF": 0.134, "RIGID": 0.061}),
        "0.76 m out": ((0.75, -0.15, 0.31), (0.75, -0.15, 0.31), 0.0, {"STIFF": None, "RIGID": 0.067}),  # None: refused
    }[case]
    first = first[spring]
    kq = _pin_stiff(monkeypatch) if spring == "STIFF" else dp.COMPLIANT_KQ
    chain, model, cfg, turn, down = _flatbot_turned_down(scene, aimed)
    stroke, why = dp.compliant_stroke(chain, cfg, turn[-1], down, aimed[2], kq=kq)
    assert why is None and stroke.bent, why
    if pinned:
        b = _pads_at(chain, _straight_end(chain, stroke.legs[-1][1], aimed[2] - cfg.button_travel_m))
        start = [float(v) for v in _held_at(chain, stroke.q_end, b)[0]]
        held = dp.DepthPress(free=True, contact=True, felt=True, judged=False, press_n=dp.COMPLIANT_PRESS_N)
    else:
        start = [float(v) for v in stroke.q_end]
        held = dp.DepthPress(free=True, judged=False, press_n=0.0)
    line, why = chain.straight_line(start, [float(v) for v in stroke.t0], stroke.quat)
    if pinned:
        assert line is not None, why  # what the rise flew before: no worse
    elif first is None:
        assert line is None and "near a singularity" in why, why
    else:
        assert float(np.abs(np.subtract(line[1], line[0])).max()) == pytest.approx(first, abs=0.005)
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    ctx.model = model
    ctx.cpose = rp.cpose_for_button(aimed, np.radians(yaw_deg), model.button_offset)
    ctx.client.live = list(start)
    ctx.client.compliant = True
    asked = _releases(ctx)
    sent = []
    real = ctx.client.execute

    def execute(traj, speed, **kw):
        sent.append(traj)
        return real(traj, speed, **kw)

    monkeypatch.setattr(ctx.client, "execute", execute)
    result = dp.compliant_rise(ctx, Runner(ctx.client, log_dir=tmp_path), stroke, held)
    assert result is held and asked == []
    (goal,) = sent
    assert ctx.client.gains_sent == [dp.COMPLIANT_GAINS]
    path = np.asarray([p.positions for p in goal.points], float)
    assert np.abs(path[0] - start).max() < 1e-9 and list(path[-1]) == [float(v) for v in stroke.q0]
    samples = np.asarray([q for _n, pts, _s in stroke.legs for q in pts], float)
    assert all(float(np.abs(samples - q).max(axis=1).min()) < 1e-9 for q in path[1:])  # the stroke's own
    floor = float(_pads_at(chain, start)[2])
    assert dp.RISE_JOIN_ABOVE_M <= float(_pads_at(chain, path[1])[2]) - floor < (0.016 if pinned else 0.02)
    assert float(np.abs(path[1] - path[0]).max()) < (0.07 if pinned else 0.03)
    assert min(float(_pads_at(chain, q)[2]) for q in path) >= floor - 1e-9


def test_on_flatbot_1623s_way_home_from_the_hold_is_one_goal_the_drivers_gate_takes(ctx, monkeypatch, tmp_path):
    """16:23's press on flatbot's URDF, the arm held by the button
    (_flatbot_1623_hold), its way out planned from the stroke's start
    (rp.plan_way_out) with home stood in by what the client hands the
    planner — planner_branch(start), the rest pose the short way round from
    there — as a joint line. _way_home sends ONE goal, from the held arm to
    the rest pose, that the runner's gates and the driver's own
    (driver.refusal: the start, the velocity limits, the continuity) take:
    no continuous joint turning round in it, the pads never lower than the
    spring held them on the way up, and the rise flowing on into the clear
    (a 143 deg turn in joint space, inside retime's 150 deg reversal). The
    run record's row for that execution says when each leg began."""
    import json
    import math

    from box_opening.constants import JOINT_VMAX
    from box_opening.primitives.core import CONTACT_SHIFT_PAD_M
    from box_opening.runtime import approach, driver
    from box_opening.runtime.branches import CONTINUOUS_JOINTS, nearest_branch, planner_branch
    from box_opening.runtime.legs import Kind, Leg
    from box_opening.runtime.runner import Runner
    from box_opening.tasks.press_demo import rest_joints

    chain, model, cfg, aimed, stroke, held, pads = _flatbot_1623_hold()
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    ctx.model = model
    ctx.cpose = rp.cpose_for_button(aimed, np.radians(36.9), model.button_offset)
    ctx.client.live = [float(v) for v in held]
    ctx.client.compliant = True
    ctx.contact_pad = CONTACT_SHIFT_PAD_M  # set before the way in flew

    def plan(ctx_, state, name, target, world, speed):
        start = planner_branch(state.joints)
        goal = nearest_branch(target[1], start)
        pts = [[a + (b - a) * k / 40 for a, b in zip(start, goal, strict=False)] for k in range(41)]
        leg = Leg(
            name=name,
            kind=Kind.MOTION,
            traj=approach.line_trajectory(ctx_.client.joint_names, pts),
            speed=speed,
            guard=None,
            world=world[0],
            world_path=str(world[1]),
            chain=state.chain,
            target=target,
            goal_joints=list(pts[-1]),
        )
        return leg, state

    monkeypatch.setattr(rp, "_plan_motion", plan)
    way, why = rp.plan_way_out(ctx, cfg, stroke.q0, stroke.quat)
    assert why is None and way.back_m == pytest.approx(rp.CLEAR_BACK_M), why
    sent = []
    real = ctx.client.execute

    def execute(traj, speed, **kw):
        sent.append(traj)
        return real(traj, speed, **kw)

    monkeypatch.setattr(ctx.client, "execute", execute)
    pressed = dp.DepthPress(free=True, contact=True, felt=True, judged=False, press_n=15.0)
    result, homed = rp._way_home(
        ctx, cfg, Runner(ctx.client, log_dir=tmp_path), stroke, pressed, list(held), (way, None), None, None
    )
    assert homed and result.free and len(sent) == 1
    (goal,) = sent
    assert driver.refusal(goal, list(held), JOINT_VMAX) is None
    path = np.asarray([p.positions for p in goal.points], float)
    assert np.abs(path[0] - held).max() < 1e-9
    assert all(
        abs(math.remainder(float(a) - float(b), 2 * math.pi)) < 1e-9
        for a, b in zip(path[-1], rest_joints(cfg), strict=False)
    )
    for j in CONTINUOUS_JOINTS:
        assert float(np.ptp(path[:, j])) < math.pi + 0.5, j  # no whole turn at a junction
    k0 = next(k for k, q in enumerate(path) if np.abs(q - stroke.q0).max() < 1e-9)  # the stroke's start
    floor = float(pads(held)[2])
    assert min(float(pads(q)[2]) for q in path[: k0 + 1]) >= floor - 1e-9
    rows = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl")))]
    (flight,) = (r for r in rows if r["leg"] == "reach_way_out_flight")
    assert flight["legs"] == ["depth:rise", "depth:clear", "home"] and flight["outcome"] == "arrived"
    assert flight["junction_speeds"][0] >= 0.04  # the rise flows into the clear: no stop between them


def _compliant_goal_ends(ctx, monkeypatch, n, outcome=None, info=None, trajs=None):
    """The fake client with the `n`th goal sent under COMPLIANT_GAINS (the
    way in's is the first) scripted to end `outcome` with `info`; every
    trajectory sent appended to `trajs`."""
    real = ctx.client.execute
    count = {"n": 0}

    def execute(traj, speed, guard=None, while_running=None, stop_when=None, gains=None):
        if trajs is not None:
            trajs.append(traj)
        if gains == dp.COMPLIANT_GAINS:
            count["n"] += 1
            if count["n"] == n and outcome is not None:
                ctx.client.exec_script.insert(0, (outcome, dict(info)))
        return real(traj, speed, guard=guard, while_running=while_running, stop_when=stop_when, gains=gains)

    monkeypatch.setattr(ctx.client, "execute", execute)


WAY_IN = ["reach:re-aim", "reach:slide-in", "depth:turn", "depth:approach", "depth:press"]
WAY_OUT = ["depth:rise", "depth:clear", "home"]


def _after_the_hold(rec):
    """The runs after the way in's: (their legs' names, the gains each execution was sent)."""
    return rec.runs[rec.runs.index((WAY_IN, [dp.COMPLIANT_GAINS])) + 1 :]


def test_a_compliant_reach_out_flies_from_the_button_to_home_as_one_execution(ctx, monkeypatch, tmp_path, capsys):
    """After the hold every 10-27 run that reached the box took extra moves
    to get home: a rise, a clear, home planned over the box (refused at
    14:33 and 16:23), then a clear-margin step or a 15 cm back-off. Now the
    way home is planned while the way in flies — the clear from the
    stroke's start up and back toward the arm, home from the clear's end,
    the box already padded for a shove — and after the hold it is ONE
    runner.run and one execution: the rise back up the stroke from where
    the spring held the arm, the clear and home, under COMPLIANT_GAINS, no
    planner asked after the hold."""
    import json

    from box_opening.constants import COMPLIANT_FREE_LINE_SPEED
    from box_opening.primitives.core import CONTACT_SHIFT_PAD_M

    code, rec, _chain, home, got = _depth_flow(
        ctx, monkeypatch, tmp_path, _popped, compliant=True, chain=_sinking_chain()
    )
    assert code == 0
    assert rec.events == ["reach:look", "aim:reach-aim", *WAY_IN, *WAY_OUT]
    assert _after_the_hold(rec) == [(WAY_OUT, [dp.COMPLIANT_GAINS])]  # one run, one execution
    legs = [rec.legs[n] for n in WAY_OUT]
    for a, b in zip(legs, legs[1:], strict=False):
        assert list(b.traj.points[0].positions) == list(a.traj.points[-1].positions), (a.name, b.name)
    for lg in legs:
        assert lg.guard is None and lg.gains == dp.COMPLIANT_GAINS and lg.world.startswith("full"), lg.name
    assert rec.legs["depth:rise"].speed == rec.legs["depth:clear"].speed == COMPLIANT_FREE_LINE_SPEED
    # home was planned from the clear's end while the way in flew — the way in handed over, the arm still at
    # its start, nothing of the way home yet — around the box padded for the stroke's shove
    assert home["start"] == list(rec.legs["depth:clear"].traj.points[-1].positions)
    assert rec.events[: home["at"]] == ["reach:look", "aim:reach-aim", *WAY_IN]
    assert home["live"] == list(rec.legs["reach:re-aim"].traj.points[0].positions)
    assert home["pad"] == CONTACT_SHIFT_PAD_M
    # the rise from the held arm (the fake hold leaves it at the stroke's end) to the stroke's start
    stroke = got["stroke"]
    rise = rec.legs["depth:rise"]
    assert list(rise.traj.points[0].positions) == list(rec.legs["depth:press"].traj.points[-1].positions)
    assert list(rise.traj.points[-1].positions) == [float(v) for v in stroke.q0]
    assert list(rec.legs["depth:clear"].traj.points[0].positions) == [float(v) for v in stroke.q0]
    rows = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl")))]
    (out,) = (r for r in rows if r["leg"] == "reach_way_out")
    assert out["planned"] and out["legs"] == WAY_OUT and out["rise_on_stroke"] and out["why"] is None
    assert out["clear_back_mm"] == pytest.approx(1000 * rp.CLEAR_BACK_M)
    assert not [r for r in rows if r["leg"] in ("depth_clear", "home_refused")]
    # its members log the one execution's t_wall alike: the flight row says where each began, as the way in's does
    (flight,) = (r for r in rows if r["leg"] == "reach_way_out_flight")
    assert flight["legs"] == WAY_OUT and flight["outcome"] == "arrived" and flight["duration_s"] > 0.0
    fracs = flight["seg_start_fracs"]
    assert len(fracs) == len(WAY_OUT) and fracs[0] == 0.0 and fracs == sorted(fracs) and fracs[-1] <= 1.0
    assert flight["stops"] == len([v for v in flight["junction_speeds"] if v < 0.04])
    member = [r for r in rows if r["leg"] in WAY_OUT and r["kind"] == "motion"]
    assert [r["leg"] for r in member] == WAY_OUT and len({r["t_wall"] for r in member}) == 1
    assert round(member[0]["t_wall"], 3) == flight["t_wall"]
    said = capsys.readouterr().out
    assert "ONE MOTION from the button to home: depth:rise, depth:clear, home — up to the stroke's start" in said


def test_a_home_planned_a_turn_off_the_clear_is_written_on_its_branch_and_still_one_execution(
    ctx, monkeypatch, tmp_path
):
    """The planner is handed planner_branch(start), and its plan can start
    a whole turn from the clear's raw end — joint_3 2 pi off here. Merged as
    written, one goal would turn joint_3 a full revolution at the junction
    (legs.MERGE_GAP_RAD); the plan is moved onto the clear's end
    (approach.on_branch_of): the same motion to the arm, one execution."""
    trajs = []
    _compliant_goal_ends(ctx, monkeypatch, 0, trajs=trajs)
    code, rec, _chain, home, _got = _depth_flow(
        ctx, monkeypatch, tmp_path, _popped, compliant=True, chain=_sinking_chain(), home_turns=1
    )
    assert code == 0 and _after_the_hold(rec) == [(WAY_OUT, [dp.COMPLIANT_GAINS])]
    leg = rec.legs["home"]
    clear_end = list(rec.legs["depth:clear"].traj.points[-1].positions)
    assert list(leg.traj.points[0].positions) == pytest.approx(clear_end) and home["start"] == clear_end
    assert leg.goal_joints == pytest.approx([0.0] * 7)  # the rest pose as planned, a turn back
    q3 = [p.positions[2] for p in trajs[-1].points]  # the one goal sent: no turn of joint_3 in it
    assert max(q3) - min(q3) < 0.5


def test_where_no_way_home_was_planned_ahead_the_rise_flies_alone_then_home_as_before(
    ctx, monkeypatch, tmp_path, capsys
):
    """The lookahead found no way home: home refused from the ends of both
    clears it tried. The rise flies on its own — the pads never left on the
    button — then home from over the box as before the one execution: the
    clear up and back from where the arm stands, then home planned."""
    import json

    code, rec, _chain, _home, _got = _depth_flow(
        ctx, monkeypatch, tmp_path, _popped, compliant=True, chain=_sinking_chain(), home_refusals=2
    )
    assert code == 0
    assert [names for names, _g in _after_the_hold(rec)] == [["depth:rise"], ["depth:clear"], ["home"]]
    rows = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl")))]
    (out,) = (r for r in rows if r["leg"] == "reach_way_out")
    assert not out["planned"] and out["legs"] == ["depth:rise"] and out["clear_back_mm"] is None
    assert out["why"].count("home refused (MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION: the start)") == 2
    assert out["why"].startswith("50 mm back: home refused") and "; 150 mm back: home refused" in out["why"]
    (clear,) = (r for r in rows if r["leg"] == "depth_clear")
    assert clear["flown"]  # solved after the hold, from where the rise left the arm
    assert "no way home was planned while the way in flew (50 mm back: home refused" in capsys.readouterr().out


def test_a_way_home_the_runner_refuses_flies_the_rise_alone_first(ctx, monkeypatch, tmp_path, capsys):
    """The runner's gates refuse the one execution before anything moves
    (its sanity gate on home, say): the pads are still on the button. The
    rise flies on its own first, then home from over the box as before."""
    import json

    from box_opening.runtime.runner import Runner

    real = Runner._refusal
    refused = []

    def refusal(self, leg):
        if leg.name == "home" and not refused:
            refused.append(leg.name)
            return "trajectory sanity gate: arm_joint_3 excursion"
        return real(self, leg)

    monkeypatch.setattr(Runner, "_refusal", refusal)
    code, rec, _chain, _home, _got = _depth_flow(
        ctx, monkeypatch, tmp_path, _popped, compliant=True, chain=_sinking_chain()
    )
    assert code == 0 and refused == ["home"]
    after = _after_the_hold(rec)
    assert after[0] == (WAY_OUT, [])  # handed over as one run, nothing sent
    assert after[1] == (["depth:rise"], [dp.COMPLIANT_GAINS])  # then the rise, alone
    assert [names for names, _g in after[2:]] == [["depth:clear"], ["home"]]
    assert (
        "the way home did not start (trajectory sanity gate: arm_joint_3 excursion) — up off the button on its "
        "own, then home from over the box"
    ) in capsys.readouterr().out
    rows = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl")))]
    assert not [r for r in rows if r["leg"] == "reach_way_out_flight"]  # nothing of it flew


def test_a_way_home_refused_as_the_arm_left_the_held_reading_rises_from_where_the_arm_stands(
    ctx, monkeypatch, tmp_path, capsys
):
    """The runner refuses a plan-free line that no longer starts where the
    arm stands (DRIFT_REPLAN_RAD): the arm moved after the hold was read,
    and the one execution never started — the pads still on the button. The
    rise flown on its own is solved again from where the arm stands: the
    held reading's would be refused the same way, and the arm would hold
    there pressing on the button. Then home from over the box, from the
    stroke's start. A 100 mm stroke on _PadsChain, the arm 0.05 rad off the
    held reading on joint_4 (which its pads do not follow)."""
    from conftest import leg as make_leg

    from box_opening.constants import DRIFT_REPLAN_RAD
    from box_opening.runtime import approach
    from box_opening.runtime.runner import Runner

    monkeypatch.setattr(approach, "_chain", lambda: _PadsChain())
    stroke = _stroke_100mm()
    live = list(HELD)
    live[3] += DRIFT_REPLAN_RAD + 0.01
    ctx.client.live = list(live)
    clear = [list(stroke.q0), [v + 0.01 for v in stroke.q0]]
    way = rp.WayOut(clear, list(TURNED_DOWN), [make_leg("home", start=clear[-1], end=[0.0] * 7)], rp.CLEAR_BACK_M)
    runs = []

    class Rec(Runner):
        def run(self, legs, execute, lookahead=None):
            out = super().run(legs, execute, lookahead)
            runs.append(([lg.name for lg in legs], list(legs[0].traj.points[0].positions), [r.outcome for r in out]))
            return out

    homes = []
    monkeypatch.setattr(
        rp, "_home_from_over_box", lambda ctx_, cfg_, runner_, result: homes.append(list(ctx_.client.joints())) or True
    )
    pressed = dp.DepthPress(free=True, contact=True, felt=True, judged=False, press_n=15.0)
    result, homed = rp._way_home(
        ctx,
        load_press_demo(PINK),
        Rec(ctx.client, log_dir=tmp_path),
        stroke,
        pressed,
        list(HELD),
        (way, None),
        None,
        None,
    )
    assert homed and result.free and result.why is None
    (names, start, outcome), (names2, start2, outcome2) = runs
    assert names == WAY_OUT and start == pytest.approx(HELD) and outcome == ["refused"]
    assert names2 == ["depth:rise"] and start2 == pytest.approx(live) and outcome2 == ["arrived"]
    assert homes == [[float(v) for v in stroke.q0]]  # up the stroke to its start, then home from over the box
    assert "a plan-free line no longer starts where the arm is" in capsys.readouterr().out


def test_a_way_home_whose_home_steps_off_a_limit_first_flies_that_step_as_a_goal_of_its_own(
    ctx, monkeypatch, tmp_path, capsys
):
    """Home planned from the clear's end with a joint too near its limit for
    the planner starts with home_arm's plan-free step (_home_legs,
    ClearMarginFree): guarded, under the client's run default, and home
    after it on the next chain, planned from the step's end — a turn off it
    on joint_7 here, as planner_branch can leave it. The rise and the clear
    fly as one goal, the step as one of its own, then home, moved onto the
    step's end: three goals of one runner.run, the guard kept on the step
    (it only watches in a compliant run)."""
    from conftest import leg as make_leg

    from box_opening.runtime import approach
    from box_opening.runtime.guards import GuardSpec
    from box_opening.runtime.runner import Runner

    monkeypatch.setattr(approach, "_chain", lambda: _PadsChain())
    stroke = _stroke_100mm()
    ctx.client.live = list(HELD)
    clear = [list(stroke.q0), [v + 0.01 for v in stroke.q0]]
    stepped = [v + 0.02 for v in clear[-1]]
    step = make_leg(
        "clear-margin",
        start=clear[-1],
        end=stepped,
        speed=0.15,
        world="bench",
        guard=GuardSpec(touch_nm=4.0, trip="obstruction"),
    )
    turn = [0.0] * 6 + [2.0 * np.pi]
    home = make_leg("home", start=list(np.add(stepped, turn)), end=list(np.add([0.0] * 7, turn)), chain=1)
    way = rp.WayOut(clear, list(TURNED_DOWN), [step, home], rp.CLEAR_BACK_M)
    pressed = dp.DepthPress(free=True, contact=True, felt=True, judged=False, press_n=15.0)
    result, homed = rp._way_home(
        ctx,
        load_press_demo(PINK),
        Runner(ctx.client, log_dir=tmp_path),
        stroke,
        pressed,
        list(HELD),
        (way, None),
        None,
        None,
    )
    assert homed and result.why is None
    assert ctx.client.gains_sent == [dp.COMPLIANT_GAINS, None, dp.COMPLIANT_GAINS]
    assert ctx.client.guards[0] is None and ctx.client.guards[1] is not None and ctx.client.guards[2] is None
    assert step.guard is not None and step.gains is None and home.gains == dp.COMPLIANT_GAINS
    assert list(home.traj.points[0].positions) == pytest.approx(stepped)  # on the step's end, the turn taken off
    assert "in 3 goals from the button to home: depth:rise, depth:clear, clear-margin, home" in capsys.readouterr().out


def test_a_way_home_that_stops_part_way_holds_the_arm_and_says_in_which_leg(ctx, monkeypatch, tmp_path, capsys):
    """The one execution stopped just after it began: the arm holds where
    it stopped, near the button — no home — and the STOP names the leg in
    flight, as the run record's row for that execution does."""
    import json

    _compliant_goal_ends(ctx, monkeypatch, 2, "failed", {"message": "PATH_TOLERANCE_VIOLATED", "progress": 0.01})
    code, rec, _chain, _home, _got = _depth_flow(
        ctx, monkeypatch, tmp_path, _popped, compliant=True, chain=_sinking_chain()
    )
    assert code == 1 and _after_the_hold(rec) == [(WAY_OUT, [dp.COMPLIANT_GAINS])]
    out = capsys.readouterr().out
    assert "STOP: the way home stopped during depth:rise (PATH_TOLERANCE_VIOLATED) — the arm holds" in out
    assert "did not get home" not in out  # one STOP, the arm held where it stopped
    rows = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl")))]
    (flight,) = (r for r in rows if r["leg"] == "reach_way_out_flight")
    assert flight["legs"] == WAY_OUT and flight["outcome"] == "failed" and flight["seg_start_fracs"][1] > 0.01


def test_a_way_home_whose_later_goal_did_not_start_goes_home_from_where_the_arm_stands(
    ctx, monkeypatch, tmp_path, capsys
):
    """Home planned 10 mrad off the clear's end flies as a goal of its own
    after the rise and the clear (legs.MERGE_GAP_RAD). The driver turned
    that goal down: the arm stands where the clear left it, in free air off
    the button, and home is planned from there as before the one
    execution."""
    real = rp._home_legs

    def off_the_clear(ctx_, cfg_, start=None):
        legs = real(ctx_, cfg_, start=start)
        if start is not None:  # planned ahead, from the clear's end
            p = legs[0].traj.points[0]
            p.positions = [float(v) + (0.01 if j == 1 else 0.0) for j, v in enumerate(p.positions)]
        return legs

    monkeypatch.setattr(rp, "_home_legs", off_the_clear)
    _compliant_goal_ends(ctx, monkeypatch, 3, "failed", {"message": "goal rejected by the driver", "progress": 0.0})
    code, rec, _chain, _home, _got = _depth_flow(
        ctx, monkeypatch, tmp_path, _popped, compliant=True, chain=_sinking_chain()
    )
    assert code == 0
    after = _after_the_hold(rec)
    assert after[0] == (WAY_OUT, [dp.COMPLIANT_GAINS, dp.COMPLIANT_GAINS])  # [rise, clear], then home turned down
    assert after[1:] == [(["home"], [None])]  # planned from where the arm stands
    out = capsys.readouterr().out
    assert "in 2 goals from the button to home: depth:rise, depth:clear, home" in out
    assert (
        "the way home's home did not start (goal rejected by the driver) — home from where the arm stands, over "
        "the box"
    ) in out


# ---- a compliant stop lets go: the springs hold where the arm stands ------------------------
# With the stop-the-arm checks off, Ctrl+C and the e-stop are the only stops, and the driver lets
# go of nothing itself: a cancel keeps its joint impedance pulling toward the last reference, its
# feed-forward on (kinova-gen3-driver 94736cd supervisor.cpp:278-296), and no watchdog lets go
# (cmd_timeout_s 0). client.release moves that reference to where the arm stands (rp.let_go).


def _releases(ctx):
    """Every release the run asks of the fake client, as how many goals it
    had executed by then."""
    asked = []
    ctx.client.release = lambda: asked.append(len(ctx.client.executed)) or True
    return asked


def test_a_ctrl_c_in_the_compliant_hold_lets_go_once_and_still_stops_the_run(ctx, monkeypatch, tmp_path):
    """Ctrl+C while the hold reads (settled_efforts) lands with no goal in
    flight: abort.py raises at once, and press_demo exits 130. The springs
    would push on into the box toward the stroke's end, 27-72 mm below the
    button's top, for as long as the driver runs. The press lets go once,
    after the turn and the stroke, and the Ctrl+C still ends the run."""
    asked = _releases(ctx)
    real_hold = dp.compliant_hold

    def hold(ctx_, cfg, runner, stroke, top_z, seen):
        def ctrl_c(*a, **k):
            raise KeyboardInterrupt

        ctx_.client.settled_efforts = ctrl_c
        return real_hold(ctx_, cfg, runner, stroke, top_z, seen)

    monkeypatch.setattr(dp, "compliant_hold", hold)
    ctx.client.compliant = True
    with pytest.raises(KeyboardInterrupt):
        _press(ctx, monkeypatch, tmp_path, [], touch_at=None, compliant=True, top_z=COMPLIANT_TOP)
    assert asked == [2]  # the turn and the stroke flown, no rise


@pytest.mark.parametrize("way_up", ["no line", "stopped"])
def test_a_compliant_press_whose_way_up_does_not_fly_lets_go_once(ctx, monkeypatch, tmp_path, way_up):
    """The button held the pads 50 mm above the stroke's end, and no way up
    flies: neither back up the stroke (rise_back_up) nor a straight line
    from the held arm (compliant_rise's `up is None`), or the rise stopped
    part-way. The arm holds, free False, and lets go once: after the hold,
    or after the rise that stopped."""
    asked = _releases(ctx)
    if way_up == "no line":
        monkeypatch.setattr(dp, "rise_back_up", lambda chain, stroke, q_held: None)
        monkeypatch.setattr(rp, "_line", lambda ctx_, xyz, quat: (None, "IK did not converge 40% of the way"))
        why = "the way up is not a straight line the arm can fly (IK did not converge 40% of the way) — the arm holds"
    else:
        ctx.client.exec_script = [ARRIVED, ARRIVED, ("failed", {"message": "PATH_TOLERANCE_VIOLATED", "progress": 0.4})]
        why = "the way up stopped (PATH_TOLERANCE_VIOLATED) — the arm holds"
    result, events, _arm, _legs, note = _compliant(ctx, monkeypatch, tmp_path, held_up_m=0.05)
    assert note["felt"] and not result.free and result.why == why and dp.exit_code(result) == 1
    assert asked == ([2] if way_up == "no line" else [3])
    assert events[-1] == ("depth:press" if way_up == "no line" else "depth:rise")


@pytest.mark.parametrize("refused_by", ["the runner", "the driver"])
def test_a_compliant_stroke_that_did_not_start_lets_nothing_go_and_the_arm_may_go_home(
    ctx, monkeypatch, tmp_path, refused_by
):
    """The runner's gates refused the stroke, or the driver turned its goal
    down (no progress): nothing moved, and the hand stands over the button
    in free air where the turn left it. Nothing to let go of, no hold read,
    no shove to pad the box for, and free: home still runs
    (press_down_from_here's callers)."""
    from box_opening.runtime.runner import Runner

    asked = _releases(ctx)
    if refused_by == "the runner":
        real = Runner._refusal
        why = "trajectory sanity gate: arm_joint_6 inside its limit margin"
        monkeypatch.setattr(Runner, "_refusal", lambda self, leg: why if leg.name == "depth:press" else real(self, leg))
    else:
        why = "goal rejected by the driver"
        ctx.client.exec_script = [ARRIVED, ("failed", {"message": why, "progress": 0.0})]  # the turn, the stroke
    ctx.client.compliant = True
    result, events, arm, _legs, _start = _press(
        ctx, monkeypatch, tmp_path, [], touch_at=None, compliant=True, top_z=COMPLIANT_TOP
    )
    assert result.free and result.why == "the press stroke did not start (%s) — nothing was pressed" % why
    assert asked == [] and arm.reads == [] and ctx.contact_pad == 0.0
    assert events == ["depth:turn", "depth:approach", "depth:press"]  # handed over, nothing after
    assert dp.exit_code(result) == 1


def test_a_compliant_stroke_with_no_legs_goes_on_to_the_hold_and_the_rise(ctx, monkeypatch, tmp_path):
    """compliant_stroke has no legs to give where the pads already stand
    within 5 mm of the stroke's end (a top_z far above the real button):
    the runner flies nothing and has no result. That is no refused stroke,
    and no crash on an empty result: the hold reads (nothing pressed), the
    arm already stands at the stroke's start so no way up flies, it is
    free, and nothing is let go."""
    import dataclasses

    asked = _releases(ctx)
    real = dp.compliant_stroke

    def at_its_end(chain, cfg, q0, quat, top_z, kq=dp.COMPLIANT_KQ):
        stroke, why = real(chain, cfg, q0, quat, top_z, kq)
        return dataclasses.replace(stroke, legs=[], q_end=list(q0)), why

    monkeypatch.setattr(dp, "compliant_stroke", at_its_end)
    ctx.client.compliant = True
    result, events, arm, _legs, _start = _press(
        ctx, monkeypatch, tmp_path, [], touch_at=None, compliant=True, top_z=COMPLIANT_TOP
    )
    assert result.free and result.why is None and not result.felt and dp.exit_code(result) == 1
    assert asked == [] and len(arm.reads) == 1 and events == ["depth:turn"]


def test_a_compliant_stroke_that_stopped_part_way_lets_go_once_and_holds(ctx, monkeypatch, tmp_path):
    """The stroke's goal failed half-way down: the springs pull on toward
    its next point, maybe on the button already. The arm holds, free
    False, the box padded for a shove; it lets go once, and no hold is
    read."""
    from box_opening.primitives.core import CONTACT_SHIFT_PAD_M

    asked = _releases(ctx)
    ctx.client.exec_script = [ARRIVED, ("failed", {"message": "PATH_TOLERANCE_VIOLATED", "progress": 0.5})]
    ctx.client.compliant = True
    result, events, arm, _legs, _start = _press(
        ctx, monkeypatch, tmp_path, [], touch_at=None, compliant=True, top_z=COMPLIANT_TOP
    )
    assert not result.free and result.why == "the press stroke stopped (PATH_TOLERANCE_VIOLATED) — the arm holds"
    assert asked == [2] and arm.reads == [] and ctx.contact_pad == CONTACT_SHIFT_PAD_M
    assert events == ["depth:turn", "depth:approach", "depth:press"]


@pytest.mark.parametrize(
    "soft, ends, asked_after",
    [
        (True, "stopped", [1]),  # the turn's goal failed 40% of the way, over the button
        (True, "not started", []),  # ... turned down by the driver: at rest where the slide in left it
        (False, "stopped", []),  # --guarded: a POSITION goal holds where it stopped
    ],
    ids=["compliant, stopped", "compliant, not started", "guarded, stopped"],
)
def test_a_turn_down_that_stopped_part_way_over_the_button_lets_go_once_in_a_compliant_press(
    ctx, monkeypatch, tmp_path, soft, ends, asked_after
):
    """Where the slide in did not turn the hand down (no walk in), the press
    turns it first (turn_down), over the button. A compliant turn stopped
    part-way leaves the springs pulling toward its next point, its
    feed-forward on: the arm holds, free False, nothing pressed, and it
    lets go once, after the turn — no stroke, no hold. A turn the driver
    turned down moved nothing, and nothing is let go; nor in a guarded
    press."""
    asked = _releases(ctx)
    why = "PATH_TOLERANCE_VIOLATED" if ends == "stopped" else "goal rejected by the driver"
    ctx.client.exec_script = [("failed", {"message": why, "progress": 0.4 if ends == "stopped" else 0.0})]
    ctx.client.compliant = soft
    result, events, arm, _legs, _start = _press(
        ctx, monkeypatch, tmp_path, [], touch_at=None, compliant=soft, top_z=COMPLIANT_TOP if soft else None
    )
    assert not result.free and result.why == "the turn met something (%s)" % why and dp.exit_code(result) == 1
    assert asked == asked_after and events == ["depth:turn"] and ctx.contact_pad == 0.0
    assert len(arm.reads) == (0 if soft else 1)  # a guarded press reads its reference before the turn


@pytest.mark.parametrize(
    "ends, code, asked_after",
    [
        ("the way in stopped part-way", 1, [2]),  # the look, then the way in: stopped 60% of the way
        ("the way home stopped", 1, [3]),  # ... the way in, the hold, then the way home, just begun
        ("the way in did not start", 1, []),  # at the look pose, at rest: home from there
        ("home", 0, []),  # pressed and home: nothing let go after it, nor between its goals
    ],
    ids=["way in stopped", "way home stopped", "way in not started", "home"],
)
def test_a_compliant_reach_out_lets_go_where_it_stopped_loaded_or_part_way_and_nowhere_else(
    ctx, monkeypatch, tmp_path, ends, code, asked_after
):
    """A way in stopped part-way leaves the springs pulling toward the next
    point of a path cut short, maybe over the box; a way home stopped, toward
    the stroke's end below the button or the rise's next point. Each lets go
    once, and the arm holds there. Where nothing moved, or the run got home,
    the arm stands at rest at a leg's end in free air: a release there would
    only add its own sag (none in run_reach_press's finally)."""
    asked = _releases(ctx)
    if ends == "the way in stopped part-way":
        _way_in_ends(ctx, monkeypatch, "failed", {"message": "PATH_TOLERANCE_VIOLATED", "progress": 0.6})
    elif ends == "the way in did not start":
        _way_in_ends(ctx, monkeypatch, "failed", {"message": "goal rejected by the driver", "progress": 0.0})
    elif ends == "the way home stopped":
        _compliant_goal_ends(ctx, monkeypatch, 2, "failed", {"message": "PATH_TOLERANCE_VIOLATED", "progress": 0.01})
    got, _rec, _chain, _home, _got = _depth_flow(
        ctx, monkeypatch, tmp_path, _popped, compliant=True, chain=_sinking_chain()
    )
    assert got == code and asked == asked_after


STOPPED = ("failed", {"message": "PATH_TOLERANCE_VIOLATED", "progress": 0.4})
REJECTED = ("failed", {"message": "goal rejected by the driver", "progress": 0.0})


@pytest.mark.parametrize(
    "fallback, ends, code, asked_after",
    [
        ("none planned", {}, 0, []),  # the look, the way in, the rise alone, the clear, home: nothing let go
        ("none planned", {4: STOPPED}, 1, [4]),  # ... the clear from over the box stopped 40% of the way
        ("none planned", {5: STOPPED}, 1, [5]),  # ... home from the clear's end stopped
        ("home off the clear", {4: REJECTED}, 0, []),  # [rise, clear], its home turned down, home from there
        ("home off the clear", {4: REJECTED, 5: STOPPED}, 1, [5]),  # ... and that home stopped
    ],
    ids=[
        "none planned, home",
        "none planned, clear stopped",
        "none planned, home stopped",
        "later goal, home",
        "later goal, home stopped",
    ],
)
def test_a_compliant_reach_outs_fallback_home_from_over_the_box_lets_go_where_it_stopped(
    ctx, monkeypatch, tmp_path, fallback, ends, code, asked_after
):
    """Where no way home was planned ahead (home refused from both clears'
    ends: the rise alone, _rise_then_home), or a later goal of it did not
    start, the way home is home from over the box (_home_from_over_box):
    the clear up and back, then home, goals of their own. Either stopped
    part-way leaves the springs pulling toward its next point, its
    feed-forward on, as the one execution's stop does: free, but not home.
    It lets go once, after that goal, and the arm holds there. A fallback
    that gets home lets nothing go."""
    kw = {}
    if fallback == "none planned":
        kw["home_refusals"] = 2
    else:
        real = rp._home_legs

        def off_the_clear(ctx_, cfg_, start=None):
            legs = real(ctx_, cfg_, start=start)
            if start is not None:  # planned ahead, from the clear's end: 10 mrad off it, a goal of its own
                p = legs[0].traj.points[0]
                p.positions = [float(v) + (0.01 if j == 1 else 0.0) for j, v in enumerate(p.positions)]
            return legs

        monkeypatch.setattr(rp, "_home_legs", off_the_clear)
    asked = _releases(ctx)
    real_execute = ctx.client.execute

    def execute(traj, speed, guard=None, while_running=None, stop_when=None, gains=None):
        n = len(ctx.client.executed) + 1  # every goal sent, the look's the first
        if n in ends:
            ctx.client.exec_script.insert(0, (ends[n][0], dict(ends[n][1])))
        return real_execute(traj, speed, guard=guard, while_running=while_running, stop_when=stop_when, gains=gains)

    monkeypatch.setattr(ctx.client, "execute", execute)
    got, rec, _chain, _home, _got = _depth_flow(
        ctx, monkeypatch, tmp_path, _popped, compliant=True, chain=_sinking_chain(), **kw
    )
    assert got == code and asked == asked_after
    after = [names for names, _g in _after_the_hold(rec)]
    assert after[0] == (["depth:rise"] if fallback == "none planned" else WAY_OUT)  # the fallback, not home in one


@pytest.mark.parametrize(
    "home, asked_after", [(None, []), (STOPPED, [3]), (REJECTED, [])], ids=["home", "home stopped", "home not started"]
)
def test_a_compliant_way_in_that_did_not_start_lets_go_only_where_its_home_from_the_look_pose_stopped(
    ctx, monkeypatch, tmp_path, home, asked_after
):
    """The driver turned the way in's goal down: the arm stands at rest at
    the look pose, in free air, nothing pressed, and goes home from there
    (_go_home). Home stopped part-way leaves the springs pulling toward its
    next point, its feed-forward on: the run lets go once, after it, and
    the arm holds there. Home, or turned down too, nothing is let go."""
    asked = _releases(ctx)
    ends = {2: REJECTED, 3: home}  # the look, the way in, home
    real_execute = ctx.client.execute

    def execute(traj, speed, guard=None, while_running=None, stop_when=None, gains=None):
        n = len(ctx.client.executed) + 1
        if ends.get(n) is not None:
            ctx.client.exec_script.insert(0, (ends[n][0], dict(ends[n][1])))
        return real_execute(traj, speed, guard=guard, while_running=while_running, stop_when=stop_when, gains=gains)

    monkeypatch.setattr(ctx.client, "execute", execute)
    got, rec, _chain, _home, _got = _depth_flow(
        ctx, monkeypatch, tmp_path, _popped, compliant=True, chain=_sinking_chain()
    )
    assert got == 1 and asked == asked_after
    assert rec.events == ["reach:look", "aim:reach-aim", *WAY_IN, "home"] and len(ctx.client.executed) == 3


def test_a_ctrl_c_in_a_compliant_reach_outs_hold_lets_go_once_and_still_stops_the_run(ctx, monkeypatch, tmp_path):
    """The one-execution reach-out's hold, between its way in and its way
    home: a Ctrl+C there lets go once, after the way in, and still ends the
    run, the client restored."""
    asked = _releases(ctx)

    def ctrl_c(ctx_):
        raise KeyboardInterrupt  # in the hold's read: no goal in flight, abort.py raises at once

    with pytest.raises(KeyboardInterrupt):
        _depth_flow(ctx, monkeypatch, tmp_path, ctrl_c, compliant=True, chain=_sinking_chain())
    assert asked == [2] and not ctx.client.compliant


@pytest.mark.parametrize(
    "rooms, back",
    [
        ((0.160, 0.368), rp.REACH_PULLBACK_M),  # 15:19's box turned down at the scene camera's 0.3111 (flatbot's URDF)
        ((0.264, 0.444), rp.CLEAR_BACK_M),  # ... at the wrist aim's 0.2881: the first leaves room
        ((0.150, 0.170), rp.CLEAR_BACK_M),  # neither does: the first, home's own clear-margin rule after it
    ],
)
def test_the_way_out_planned_ahead_takes_the_clear_that_leaves_home_room_to_plan_from(ctx, monkeypatch, rooms, back):
    """plan_way_out picks the clear as _clear_up_and_back does: the first of
    clear_targets whose line flies and leaves every joint room for home to
    plan from, else the first whose line flies, no line solved past the one
    taken — and plans home from that clear's end, not from where the arm
    stands."""
    import test_reach_press as trp

    from box_opening.primitives.core import CLEAR_MARGIN_ROOM_RAD
    from box_opening.runtime import approach

    class Folding(trp._Chain):
        """joint_6 ends `rooms` from its upper limit at the two targets."""

        limits = [(None, None)] * 5 + [(-2.09, 2.09), (None, None)]

        def straight_line(self, q0, xyz, quat):
            self.lines.append((list(xyz), list(quat)))
            end = list(q0)
            end[5] = 2.09 - rooms[0 if _back_of(xyz) < 0.1 else 1]
            return [list(q0), end], None

    ctx.model = ContainerModel.load(PINK)
    ctx.cpose = rp.cpose_for_button(trp.BUTTON, 0.0, ctx.model.button_offset)
    chain = Folding()
    chain.R = trp.STRAIGHT_DOWN
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    q0 = [0.1] * 7
    way, why = rp.plan_way_out(ctx, load_press_demo(PINK), q0, TURNED_DOWN)
    assert why is None and way.back_m == pytest.approx(back) and way.quat == TURNED_DOWN
    assert way.clear[0] == q0 and 2.09 - way.clear[-1][5] == pytest.approx(rooms[0 if back == rp.CLEAR_BACK_M else 1])
    assert len(chain.lines) == (1 if rooms[0] >= CLEAR_MARGIN_ROOM_RAD else 2)
    (home,) = way.home
    assert ctx.client.joint_starts == [way.clear[-1]] and ctx.client.live == [0.0] * 7  # from the clear's end
    assert list(home.traj.points[0].positions) == pytest.approx(way.clear[-1])
    if rooms[0] < CLEAR_MARGIN_ROOM_RAD:
        assert way.passed.startswith("50 mm back: joint_6 %.3f rad from its limit at its end" % rooms[0])
    else:
        assert way.passed is None


def test_a_way_out_whose_home_does_not_plan_from_a_clear_takes_the_next_or_none(ctx, monkeypatch):
    """A home the planner refuses from the first clear's end is planned
    from the next one's; refused from both, there is no way out to fly."""
    import test_reach_press as trp

    from box_opening.runtime import approach

    class Refused:
        success = False
        message = "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION"
        trajectory = None

    ctx.model = ContainerModel.load(PINK)
    ctx.cpose = rp.cpose_for_button(trp.BUTTON, 0.0, ctx.model.button_offset)
    chain = trp._Chain()
    chain.R = trp.STRAIGHT_DOWN
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    cfg = load_press_demo(PINK)
    ctx.client.plans = [Refused()]
    way, why = rp.plan_way_out(ctx, cfg, [0.1] * 7, TURNED_DOWN)
    assert why is None and way.back_m == pytest.approx(rp.REACH_PULLBACK_M)
    assert way.passed == "50 mm back: home refused (MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION)"
    ctx.client.plans = [Refused(), Refused()]
    way, why = rp.plan_way_out(ctx, cfg, [0.1] * 7, TURNED_DOWN)
    assert way is None and why == (
        "50 mm back: home refused (MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION); "
        "150 mm back: home refused (MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION)"
    )
