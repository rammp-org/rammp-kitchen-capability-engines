"""The reach-out's lid off (tasks/reach_lid_off): the compliant reach-out
press as the whole run — the knob looked at from the look pose, the press
once more where it is not up, the grip, the lift, the carry and the
set-down — with the arm, the cameras and the planner faked; and its
reachability on flatbot's URDF for the boxes of 2026-10-27/28."""

import json
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import setdown_flown as flown
import test_reach_press as trp
from conftest import fixture_path, flatbot_urdf

from box_opening.constants import FINGERS_PAST_MODEL_M, GAINS_MEDIUM, HOME_SPEED
from box_opening.models.container import ContainerModel, ContainerPose, load_press_demo
from box_opening.tasks import depth_press as dp
from box_opening.tasks import reach_lid_off as rlo
from box_opening.tasks import reach_press as rp

# The 2026-10-21 HOME (conftest.HOME_20261021): this file's recorded runs and poses were built on it.
pytestmark = pytest.mark.usefixtures("home_20261021")

PINK = trp.PINK
BUTTON = trp.BUTTON  # (0.66, -0.57, 0.40): the fake fk's box
UP = (BUTTON[0], BUTTON[1], BUTTON[2] + 0.009)  # the aim's press point on a popped knob: its top
KNOB_ON = 0.39  # the knuckle closed on the knob (bench 2026-08-26: 0.387)
AIR = 0.79  # ... closed on air


def test_the_grip_goes_by_the_lid_where_the_hold_held_the_pads_not_by_the_scene_cameras_top():
    """Owner 2026-10-30: the scene camera gives a general location. The lid
    the grip goes by is GRIP_LID_ABOVE_HELD_M (4 mm) above where the hold
    held the pads, whatever the press's top: 10:58's scene camera put the
    top 15.8 mm above the touch, the grip went 7.4 mm above it (the button's
    travel) plus the lid's rise, and closed on air. Without a hold, the
    press's top."""
    assert rlo.GRIP_LID_ABOVE_HELD_M == pytest.approx(0.004)
    assert rlo.grip_lid_z(0.3097) == 0.3097
    for top in (0.3097, 0.300, 0.3426):  # 13:43's top within the travel, one below the pads, 10:58's 15.8 mm above
        assert rlo.grip_lid_z(top, 0.3056) == pytest.approx(0.3056 + 0.004)
    assert rlo.grip_lid_z(0.3426, 0.3268) == pytest.approx(0.3308)  # 10:58: 3.5 mm lower than it went (0.3343)


class _Chain(trp._Chain):
    """test_reach_press's fake chain, with an IK for the lid's spot."""

    def __init__(self, refuse=(), unreachable=()):
        super().__init__(refuse)
        self.unreachable = unreachable
        self.solved = []

    def nearest_solution(self, q_start, xyz, quat, seeds=()):
        self.solved.append(list(xyz))
        if any(what(xyz) for what in self.unreachable):
            return None
        return np.zeros(7)


class _Rec(trp._Rec):
    def __init__(self):
        super().__init__()
        self.plans = []  # (leg name, how many events came before it was planned)
        self.planning = 0  # plans in progress now
        self.most_at_once = 0  # ... and the most there ever were
        self.aim_hits = []  # (each aim's tag, the agreeing hits it asked for: None, the reach-out's)
        self.setdowns = []  # (each set-down's spot, what its watch felt), in order


class _Felt:
    """reach_lid_off.SetdownContact's stand-in: what the set-down's watch felt, as the release asks
    it (credible, refusal, record)."""

    WHY = {
        "never": "the touch ended 8 mm under where the plan has the lid meet the table and the arm felt no table",
        "held": "something held the lid up 60 mm over where the plan has it meet the table (more than 35)",
        "failed": "the set-down's watch failed (boom) and the descent was stopped there",
    }

    def __init__(self, how):
        self.how = how

    def credible(self):
        return self.how == "met"

    def refusal(self):
        return None if self.how == "met" else (self.how, self.WHY[self.how])

    def record(self):
        met = self.how in ("met", "held")
        out = {
            "watched": True,
            "met": met,
            "credible": self.credible(),
            "verdict": None if self.credible() else self.how,
        }
        if met:
            out.update(above_top_mm=12.0 if self.how == "met" else 60.0, cut=True, cut_mm=20.0)
        return out


def _walk_or_turn(ctx_, q_look, slide, start, quat, top_z=None):
    """reach_press.walk_or_turn for the fake chain: the turn in place where
    the slide ends, as test_reach_press._way_down's."""
    turn = [[float(v) for v in slide[-1]], [float(v) + 0.01 for v in slide[-1]]]
    return dp.WayDown(turn, list(trp.DOWN_QUAT), 0.0, False), None


def _full_flow(
    ctx,
    monkeypatch,
    tmp_path,
    aims,
    knuckle=KNOB_ON,
    slipped=False,
    press_only=False,
    chain=None,
    extent=None,
    ends=None,
    home_refusals=0,
    wrist=None,
    scene="table",
    refuse_carry=False,
    rise=None,
    at_spot_look=None,
    one_motion=True,
    from_press=False,
    setdown="met",
    touch_table=False,
):
    """run_reach_press as a full run (compliant --depth-press, no
    --press-only), everything faked: `aims` the wrist's aims in order, each
    (press point, knob mm) or None — the look's, then each after a press
    (and its recheck); `knuckle` what closing on the knob reads (a list:
    each close's in turn, the last for every one after), `slipped`
    whether the fingers read that still at the table; `extent` the table
    the scene camera saw (no room map: the lid goes beside the box, in its
    zone); `ends`: {n: (outcome, info)} for the n-th goal sent; `wrist`
    the wrist's points at the spot look, `scene` the scene camera's (the
    default: a flat table 0.3 m round the box, _table(r=0.3); None: no camera data,
    and so no table measured for the lid to go on); `setdown` what the
    set-down's watch felt (_Felt: "met", the lid met the table where the
    plan has it; "never", "held", "failed": it did not, and the lid is not
    let go; a _Felt of the test's own; a list: each spot's in turn, the last
    for every one after; None: the real watch, which the fake
    client feeds nothing). The
    table the set-down goes by is the cameras' (the fake chain's fk stands
    still, and its fingers' hold says nothing of the lid's top) unless
    `touch_table`: then setdown_table_z's own, the cameras' table moved by
    where that fk has the fingers hold the knob (_touch_table_z);
    `rise` depth_press.rise_back_up's stand-in (None:
    back up the stroke); `at_spot_look()` what else happens at the spot
    look; `one_motion` False: the grip's lookahead plans the lift alone,
    as before the close flew on to the set-down — the lift, then the spot
    look, then the carry; `from_press` True: the first press's hold goes
    straight on to the grip (GRIP_FROM_PRESS), else the way back, the look
    and the grip from the look pose, as after that grip missed. Returns
    (exit code, the record, the chain, the run log's rows)."""
    from conftest import leg as make_leg

    from box_opening.runtime import approach
    from box_opening.runtime.runner import Runner

    monkeypatch.setenv("BOX_OPENING_STATE", str(tmp_path))  # never the robot's own gripper.json
    rec = _Rec()
    cfg = load_press_demo(PINK)
    ctx.model = ContainerModel.load(PINK)
    ctx.press_cfg = cfg
    ctx.lid_drop = ctx.lid_drops = None  # the run resolves its own
    ctx.worlds.table_extent = extent
    chain = chain or _Chain()
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    monkeypatch.setattr(approach, "_gravity", lambda: trp._Gravity())
    monkeypatch.setattr(rp, "oblique_aim_available", lambda: True)
    refusals = {"home": home_refusals}

    def plan(ctx_, state, name, target, world, speed):
        rec.plans.append((name, len(rec.events)))
        rec.planning += 1  # the planner serves one plan at a time
        rec.most_at_once = max(rec.most_at_once, rec.planning)
        try:
            if name == "lid:carry" and refuse_carry:
                raise RuntimeError("planning failed for lid:carry: MotionGenStatus.IK_FAIL")
            if name == "home" and refusals["home"] > 0:
                refusals["home"] -= 1
                raise RuntimeError("planning failed for home: MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION")
            start = list(state.joints)
            lg = make_leg(
                name,
                start=start,
                end=[0.0] * 7 if name == "home" else [v + 0.05 for v in start],
                speed=speed,
                world=world[0],
            )
            lg.target = target
            return lg, state
        finally:
            rec.planning -= 1

    monkeypatch.setattr(rp, "_plan_motion", plan)
    if not touch_table:
        monkeypatch.setattr(rlo, "setdown_table_z", lambda ctx_, table_z, fix_, q_grip=None: (float(table_z), None))
    if setdown is not None:
        felts = list(setdown) if isinstance(setdown, list | tuple) else [setdown]

        def felt(ctx_, legs, carry):
            how = felts.pop(0) if len(felts) > 1 else felts[0]
            carry.contact = how if isinstance(how, _Felt) else _Felt(how)
            rec.setdowns.append((carry.drop, carry.contact.how))
            return carry.contact

        monkeypatch.setattr(rlo, "watch_setdown", felt)
    if isinstance(scene, str):
        scene = _table(r=0.3)  # a small one: every spot on it is judged in each of these runs
    monkeypatch.setattr(rlo, "GRIP_FROM_PRESS", from_press)
    if not one_motion:
        monkeypatch.setattr(
            rlo, "grip_ahead", lambda ctx_, cfg_, eyes, q, quat, room=None: (*rlo.plan_lift(ctx_, cfg_, q, quat), None)
        )
    monkeypatch.setattr(rp, "way_down", trp._way_down([], lambda tilt, tip: True))
    monkeypatch.setattr(rp, "walk_or_turn", _walk_or_turn)
    # the arm held on the button 32 mm above the stroke's end, the way up back along it
    monkeypatch.setattr(
        dp, "rise_back_up", rise or (lambda chain_, stroke, q, join_above_m=None: [list(q), list(stroke.q0)])
    )
    held_pad = (BUTTON[0], BUTTON[1], BUTTON[2] - 0.006)

    def hold(ctx_, cfg_, runner_, stroke, top_z, seen, contact=None):
        rec.events.append("hold")
        seen.update(held_joints=[float(v) + 0.002 for v in stroke.legs[-1][1][-1]], touch_xyz=held_pad)
        return dp.DepthPress(free=True, contact=True, felt=True, judged=False, press_n=15.0)

    monkeypatch.setattr(dp, "compliant_hold", hold)
    monkeypatch.setattr(
        rlo,
        "spot_look",
        lambda node, watcher, runner_: (rec.events.append("spot-look"), at_spot_look and at_spot_look(), wrist)[2],
    )

    class R(Runner):
        def run(self, legs, execute, lookahead=None):
            for lg in legs:
                rec.events.append(lg.name)
                rec.legs[lg.name] = lg
            sent = len(getattr(self.client, "gains_sent", []))
            out = super().run(legs, execute, lookahead)
            rec.runs.append(([lg.name for lg in legs], getattr(self.client, "gains_sent", [])[sent:]))
            return out

    runner = R(ctx.client, log_dir=tmp_path)
    queue = list(aims)

    def aim(node, watcher, yaw, lid_z, prior_xy, runner_, tag, min_hits=None):
        rec.events.append("aim:" + tag)
        rec.aim_hits.append((tag, min_hits))
        got = queue.pop(0) if queue else None
        if got is None:
            return None, "no circle", None, None
        return (got[0], 0.0), "3/3 frames", got[1], got[0]

    monkeypatch.setattr(rp, "_aim", aim)
    closes = list(knuckle) if isinstance(knuckle, list | tuple) else [knuckle]
    # the fingers as press_demo's press:close left them before the run: shut on nothing
    held = {"on": True, "reads": AIR}

    def gripper_cmd(pos):
        if pos is None:  # query only
            return True, (AIR if slipped else held["reads"]) if held["on"] else 0.0, False
        held["on"] = pos > 0.5
        if held["on"]:
            held["reads"] = closes.pop(0) if len(closes) > 1 else closes[0]
        return True, held["reads"] if held["on"] else 0.0, False

    ctx.client.gripper_cmd = gripper_cmd
    ctx.client.lets_go = []
    ctx.client.release = lambda: ctx.client.lets_go.append(len(ctx.client.executed)) or True
    if ends:
        trp._goal_ends(ctx, monkeypatch, ends)
    fix = SimpleNamespace(
        pose=ContainerPose(xyz=(BUTTON[0], BUTTON[1], BUTTON[2] - 0.121), yaw=0.0),
        top_xyz=BUTTON,
        table_points=scene,
        table_z_scene=BUTTON[2] - 0.121,
    )
    locator = SimpleNamespace(
        prime=lambda: None,
        locate=lambda s: (rec.events.append("scene"), fix)[1],
    )
    args = SimpleNamespace(execute=True, press_only=press_only, depth_press=True, compliant=True)
    watcher = SimpleNamespace(table_z=BUTTON[2] - 0.121)
    code = rp.run_reach_press(None, ctx, cfg, runner, watcher, args, fix, locator)
    rows = []
    for f in sorted(tmp_path.glob("run-*.jsonl")):
        rows += [json.loads(line) for line in open(f)]
    return code, rec, chain, rows


WAY_IN = ["reach:re-aim", "reach:slide-in", "depth:turn", "depth:press"]
BACK = ["depth:rise", "depth:turn-back", "reach:slide-out"]
GRIP = ["grip:open", "reach:re-aim", "reach:slide-in", "depth:turn", "grip:approach", "grip:down"]
CLOSE = ["grip:close", "lid:lift", "lid:clear"]
CARRY = ["lid:carry", "place:down", "place:touch"]
HOME = ["place:open", "place:retreat", "home"]
# ... flown as two: the release riding the retreat, then home, slow, under MEDIUM (core.as_home)
RELEASED, HOMED = ["place:open", "place:retreat"], ["home"]


def _executions(rec):
    """Each runner.run's legs with the gains of every execution it sent."""
    return [(names, gains) for names, gains in rec.runs]


class _Video:
    """camera_video.CameraVideo's stand-in: what the run says on it, and when."""

    def __init__(self, sources, out_dir, rec=None, ctx=None):
        self.sources, self.out_dir, self.rec, self.ctx = [n for n, _g in sources], out_dir, rec, ctx
        self.said, self.started, self.finished, self.stopped = [], None, None, None

    def start(self):
        self.started = list(self.rec.events)
        return self

    def mark(self, text):
        self.said.append(text)

    def overlay(self, name, uv, text=""):
        self.said.append("overlay %s" % name)

    def finish(self, caption=None):
        self.finished = (caption, len(self.ctx.client.lets_go))
        return "/captures/lid-video-x.mp4"

    def stop(self):
        self.stopped = len(self.ctx.client.lets_go)


def _videoed(ctx, monkeypatch):
    """A full run that records the lid video, as a run off the robot's tests
    does not: the video made for it (reach_press._reach_press's) kept."""
    from box_opening.perception import camera_video

    made = []
    monkeypatch.setenv("BOX_OPENING_VIDEO", "1")

    def make(sources, out_dir):
        made.append(_Video(sources, out_dir, made_rec[0], ctx))
        return made[-1]

    made_rec = [None]
    monkeypatch.setattr(camera_video, "CameraVideo", make)
    return made, made_rec


def test_a_full_run_records_both_cameras_from_the_grip_on_says_each_step_and_writes_it_at_the_end(
    ctx, monkeypatch, tmp_path, capsys
):
    """Owner 2026-10-29: "create a video of what the wrist and scene cameras
    see during the lid placement so i can see too". A full --execute run
    makes it of the wrist and the scene cameras; it starts after the press,
    as the grip's way down begins; its caption says each step; the end is
    its last picture, and the console says where the file is."""
    made, made_rec = _videoed(ctx, monkeypatch)
    real = _Rec

    def rec_kept():
        r = real()
        made_rec[0] = r
        return r

    monkeypatch.setattr(sys.modules[__name__], "_Rec", rec_kept)
    code, rec, _chain, _rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)])
    assert code == 0
    (v,) = made
    assert v.sources == ["wrist", "scene"]
    assert "hold" in v.started and "grip:open" not in v.started  # after the press, before the grip
    steps = [t.split(" ")[0].rstrip(":") for t in v.said if not t.startswith("overlay")]
    assert steps == ["GRIP", "CLOSE", "RELEASE"] and "ONE MOTION: LIFT, CLEAR, CARRY to" in v.said[1]
    assert v.finished[0] == "the end: the lid set down, the arm home"
    assert "VIDEO: what the wrist and scene cameras saw from the grip on: /captures/lid-video-x.mp4" in (
        capsys.readouterr().out
    )


def test_a_ctrl_c_with_the_lid_in_the_fingers_lets_go_first_and_writes_no_video(ctx, monkeypatch, tmp_path):
    """Review on #23 (REVIEW_RULES C1): a cancel (the engine's Ctrl+C) must
    end at once. The arm lets go first, as before; the video's sampling
    stops and nothing is encoded (seconds of ffmpeg before the cancelled
    result). Any other way out still writes it, with how the run stopped."""
    made, made_rec = _videoed(ctx, monkeypatch)
    real = _Rec

    def rec_kept():
        r = real()
        made_rec[0] = r
        return r

    def ctrl_c():
        raise KeyboardInterrupt

    monkeypatch.setattr(sys.modules[__name__], "_Rec", rec_kept)
    with pytest.raises(KeyboardInterrupt):
        _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], at_spot_look=ctrl_c, one_motion=False)
    (v,) = made
    assert v.finished is None and v.stopped is not None and v.stopped >= 1  # let go, then sampling stopped
    made.clear()

    def stop_line():
        raise SystemExit(1)

    with pytest.raises(SystemExit):
        _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], at_spot_look=stop_line, one_motion=False)
    (v,) = made
    caption, lets_go = v.finished
    assert caption == "stopped: SystemExit" and lets_go >= 1  # the arm let go before the video was written


def test_a_knob_up_after_the_press_is_gripped_lifted_carried_set_down_and_the_arm_goes_home(
    ctx, monkeypatch, tmp_path, capsys
):
    """The full run: the way in, the hold, the way back to the look pose
    (the rise and the way down backwards), the look, then the grip's way
    down with the fingers opening on it; the close, then the lift, the
    clear, the carry and the set-down in one motion (owner 2026-10-29:
    "seemlesly grab the lid and move it to the safe location without
    stopping or interuptions"); the release, the retreat and home — five
    executions after the look's, every one RIGID and unguarded. The lid's
    spot, its carry and home after the set-down all planned while the hand
    came down to the knob, on the room judged while the press flew: no
    spot look, no plan at rest."""
    modes = []
    real_execute = ctx.client.execute
    monkeypatch.setattr(
        ctx.client, "execute", lambda *a, **k: (modes.append(ctx.client.compliant), real_execute(*a, **k))[1]
    )
    code, rec, chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)])
    assert code == 0
    assert rec.events == [
        "reach:look",
        "aim:reach-aim",
        *WAY_IN,
        "hold",
        *BACK,
        "aim:reach-after",
        *GRIP,
        *CLOSE,
        *CARRY,
        *HOME,
    ]
    runs = _executions(rec)
    assert [names for names, _g in runs] == [["reach:look"], WAY_IN, BACK, GRIP, CLOSE + CARRY, RELEASED, HOMED]
    # one goal each, every one under the run's one gains spec: the lift to the set-down one execution; the
    # release rides the retreat (sent, not waited on); home after it flies under MEDIUM, as every home does
    for names, gains in runs[1:-1]:
        assert gains == [dp.COMPLIANT_GAINS], names
    assert runs[-1] == (HOMED, [GAINS_MEDIUM]) and rec.legs["home"].speed == HOME_SPEED
    assert modes == [True] * 7 and rec.legs["place:open"].defer_join
    assert all(lg.guard is None for lg in rec.legs.values() if lg.kind.value == "motion")
    # the carry and home after the set-down planned once each, while the grip's way down flew (the record then
    # ended with its legs); home before that from the look pose while the way back flew, for a stop there
    planned = {name: at for name, at in rec.plans}
    gripping = rec.events.index("grip:close")
    assert planned["lid:carry"] == planned["home"] == gripping
    assert [name for name, _at in rec.plans].count("lid:carry") == 1
    assert [at for name, at in rec.plans if name == "home"] == [rec.events.index("aim:reach-after"), gripping]
    # the spot chosen ahead, said before the close: on the table the scene camera measured, where it reads clear
    legs = [r["leg"] for r in rows]
    assert legs.index("reach_grip") < legs.index("lid_spot") < legs.index("reach_grip_result")
    (spot,) = (r for r in rows if r["leg"] == "lid_spot")
    assert (spot["tier"], spot["mapped"], spot["chosen"]) == ("clear", True, "ahead, with the scene camera")
    assert spot["table"]["off_table"] == 0 and spot["table"]["z"] == pytest.approx(TABLE_Z, abs=1e-4)
    assert spot["room_s"] is not None  # the room judged while the press flew
    # of ahead_s, the wait for that room; the spots whose reach was judged, their IK's time; the
    # lookahead's overrun (None: this client says nothing of what it sent)
    assert 0.0 <= spot["room_wait_s"] <= spot["ahead_s"] and spot["spot_checks"] >= 1 and spot["spot_ik_s"] >= 0.0
    assert spot["host_stood_s"] is None
    # the wrist's look-pose frame was to be pooled into the room (shadow): this aim kept none
    assert spot["pooled"]["chose"] is False and spot["pooled"]["why"] == "no aim kept its frame"
    # the retreat flown from where the arm stood at the release, held at its start; home as planned ahead
    (back,) = (r for r in rows if r["leg"] == "retreat_from_live")
    assert (back["home_ahead"], back["home_flies"], back["hold_s"]) == (True, True, rlo.RETREAT_HOLD_S)
    assert rec.legs["place:retreat"].start_hold_s == rlo.RETREAT_HOLD_S
    assert spot["xyz"] == pytest.approx(list(ctx.lid_drop.xyz), abs=1e-4)
    hover = rec.legs["lid:carry"].target[1]
    assert hover[:2] == pytest.approx(list(ctx.lid_drop.xyz[:2]))
    assert ctx.lid_at is ctx.lid_drop  # the placed lid is in every world after the release
    (knob,) = (r for r in rows if r["leg"] == "reach_knob")
    assert (knob["press"], knob["popped"], knob["looks"], knob["knob_up_mm"]) == (1, True, 1, 9.0)
    # the grip goes by the lid GRIP_LID_ABOVE_HELD_M above where the hold
    # held the pads (6 mm below the top), whatever the top
    (grip,) = (r for r in rows if r["leg"] == "reach_grip")
    lid = BUTTON[2] - 0.006 + rlo.GRIP_LID_ABOVE_HELD_M
    assert grip["lid_z"] == pytest.approx(lid) and grip["pads_z"] == pytest.approx(lid - rlo.GRIP_BELOW_LID_M)
    out = capsys.readouterr().out
    assert "CLOSE, then ONE MOTION: lid:lift, lid:clear, lid:carry, place:down, place:touch" in out
    assert "LID GRIPPED" in out and out.rstrip().endswith("arm home")
    assert ctx.client.lets_go == []  # nothing stopped: nothing let go


def test_with_the_free_air_in_position_the_arm_stands_while_the_springs_come_in_and_the_press_is_as_ever(
    ctx, monkeypatch, tmp_path
):
    """BOX_OPENING_AIR_CONTROL=position (owner 2026-11-16, the arm "a little floppy" on its way to the button:
    "do 1 and 3"): the look and the way in's free-air lines fly in position control, slower and eased; the
    arm then stands at the stroke's start, the driver is put in its joint impedance there and its spring
    waited in; and the stroke, the hold and everything after fly on the spring exactly as they did."""
    from box_opening.constants import AIR_ACCEL, AIR_DECEL, AIR_LINE_SPEED, AIR_LOOK_SPEED, AIR_SWITCH_S

    monkeypatch.setenv("BOX_OPENING_AIR_CONTROL", "position")
    waits = []
    monkeypatch.setattr(rp.time, "sleep", lambda s: waits.append(s))
    modes = []
    real_execute = ctx.client.execute
    monkeypatch.setattr(
        ctx.client, "execute", lambda *a, **k: (modes.append(ctx.client.compliant), real_execute(*a, **k))[1]
    )
    ways, real_fly = [], rp.fly_way_in

    def fly(ctx_, runner, legs, stroke, top_z, lookahead=None):
        ways.append({lg.name: lg for lg in legs})  # the press's own legs (the grip's way down has their names too)
        return real_fly(ctx_, runner, legs, stroke, top_z, lookahead)

    monkeypatch.setattr(rp, "fly_way_in", fly)
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)])
    assert code == 0
    runs = _executions(rec)
    air, stroke = WAY_IN[:-1], WAY_IN[-1:]
    (way_in,) = ways
    # the run as it was, but for the way in: two executions where it was one
    assert [names for names, _g in runs] == [["reach:look"], air, stroke, BACK, GRIP, CLOSE + CARRY, RELEASED, HOMED]
    assert rec.events[: 2 + len(WAY_IN)] == ["reach:look", "aim:reach-aim", *WAY_IN]
    # in position: the look, and the free air of the way in. On the spring: the stroke and all after it
    assert modes[:3] == [False, False, True] and all(modes[3:])
    assert runs[1][1] == [None] and runs[2][1] == [dp.COMPLIANT_GAINS]
    for names, gains in runs[3:-1]:
        assert gains == [dp.COMPLIANT_GAINS], names
    look = rec.legs["reach:look"]
    assert (look.speed, look.compliant, look.gains, look.ease) == (AIR_LOOK_SPEED, False, None, (AIR_ACCEL, AIR_DECEL))
    for name in air:
        leg = way_in[name]
        assert (leg.compliant, leg.gains, leg.ease) == (False, None, (AIR_ACCEL, AIR_DECEL)), name
        assert leg.speed == AIR_LINE_SPEED and not leg.fold and leg.guard is None
    press = way_in["depth:press"]
    assert (press.compliant, press.gains, press.ease) == (None, dp.COMPLIANT_GAINS, None)
    # the grip's way down to the knob stays one motion on the spring (its lines have the same names)
    assert all(rec.legs[name].compliant is None and rec.legs[name].gains == dp.COMPLIANT_GAINS for name in air)
    # between them: stood still, the driver's impedance entered where the arm stands, its fade-in waited
    legs = [r["leg"] for r in rows]
    (switch,) = (r for r in rows if r["leg"] == "air_switch")
    assert switch["primed"] is True and switch["wait_s"] == AIR_SWITCH_S and waits.count(AIR_SWITCH_S) == 1
    assert switch["tries"] == 1 and rp.AIR_PRIME_RETRY_S not in waits  # sent at the first ask: no second
    assert legs.index("depth:turn") < legs.index("air_switch") < legs.index("depth:press")
    assert ctx.client.primes == [None]  # that one hold, of the joints it reads then; none at the run's start
    # the press is watched as an execution of its own, and its flight is the stroke's
    (flight,) = (r for r in rows if r["leg"] == "reach_way_in_flight")
    assert flight["legs"] == stroke


def test_in_position_a_free_air_part_that_stops_brings_no_spring_and_no_stroke(ctx, monkeypatch, tmp_path):
    monkeypatch.setenv("BOX_OPENING_AIR_CONTROL", "position")
    monkeypatch.setattr(rp.time, "sleep", lambda s: None)
    # the second goal sent is the way in's free air: it stops part way
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], ends={2: ("failed", {"error": "the driver stopped it"})}
    )
    assert code == 1
    assert [names for names, _g in _executions(rec)][:2] == [["reach:look"], WAY_IN[:-1]]
    assert "depth:press" not in rec.events and not [r for r in rows if r["leg"] == "air_switch"]
    assert ctx.client.primes == []


def _hosting(ctx, monkeypatch):
    """Whether each execution the fake client is sent hosts planning (execute's while_running), in order."""
    hosted = []
    real_execute = ctx.client.execute

    def execute(traj, speed, guard=None, while_running=None, stop_when=None, gains=None):
        hosted.append(while_running is not None)
        return real_execute(traj, speed, guard=guard, while_running=while_running, stop_when=stop_when, gains=gains)

    monkeypatch.setattr(ctx.client, "execute", execute)
    return hosted


@pytest.mark.parametrize("air", ["impedance", "position"])
def test_a_stroke_flown_on_its_own_hosts_no_planning_and_the_way_home_is_planned_after_the_hold(
    ctx, monkeypatch, tmp_path, capsys, air
):
    """--press-only plans its way home while the way in flies (reach_press._fly_way_in: the runner's
    lookahead), and the client runs what an execution hosts before it first polls that execution's
    stop_when. As one motion (flown 2026-11-12) that is at the re-aim's start, seconds before the stroke.
    With the free air in position the stroke is an execution of its own, and the planning hosted there
    left the press's contact watch unpolled for as long as the planner took, the pads on their way down to
    the lid (review 2026-11-16). There the stroke hosts nothing: the way home is planned after the hold —
    up off the button, then home from over the box."""
    monkeypatch.setenv("BOX_OPENING_AIR_CONTROL", air)
    monkeypatch.setattr(rp.time, "sleep", lambda s: None)
    hosted = _hosting(ctx, monkeypatch)
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4)], press_only=True)
    assert code == 0
    # each execution with the legs of the runner.run that sent it
    flown = [names for names, gains in _executions(rec) for _goal in gains]
    assert len(flown) == len(hosted)
    held = rec.events.index("hold")
    (home_planned,) = (at for name, at in rec.plans if name == "home")
    (way_out,) = (r for r in rows if r["leg"] == "reach_way_out")
    if air == "impedance":
        # as flown: the one motion hosts it, and the way home is in hand at the hold
        assert [names for names, h in zip(flown, hosted, strict=True) if h] == [WAY_IN]
        assert home_planned <= held and way_out["planned"] is True
        return
    assert flown[:3] == [["reach:look"], WAY_IN[:-1], WAY_IN[-1:]]
    assert not any(hosted)  # the stroke's execution least of all: its watch is polled from its first sample
    assert home_planned > held and way_out["planned"] is False
    assert rec.events[held:] == ["hold", "depth:rise", "depth:clear", "home"]
    out = capsys.readouterr().out
    assert "no way home was planned while the way in flew (the stroke flew on its own" in out
    assert "the lookahead plan failed" not in out  # none was asked for: nothing failed


@pytest.mark.parametrize("press_only", [False, True], ids=["the full run", "the press alone"])
@pytest.mark.parametrize("home_refusals", [0, 2], ids=["home", "home refused"])
def test_in_position_a_stroke_that_does_not_start_leaves_from_over_the_box_the_clear_first(
    ctx, monkeypatch, tmp_path, capsys, press_only, home_refusals
):
    """As one motion, "the way in did not start" could only mean the arm still stood at the look pose, and
    home is planned straight from there. With the free air in position that air has flown when the
    stroke's own execution is turned down (the driver, the client's gates, the runner's): the hand stands
    turned down at the stroke's start, 35-60 mm over the lid (review 2026-11-16). It leaves as every other
    stop over the box does: up and back clear of the box first, then home — and says the stroke did not
    start, nothing pressed. An arm that then does not get home is said to hold, never left as a stop a
    retry may follow."""
    monkeypatch.setenv("BOX_OPENING_AIR_CONTROL", "position")
    monkeypatch.setattr(rp.time, "sleep", lambda s: None)
    # the goals sent: the look, the way in's free air, then the stroke, which the driver turns down
    code, rec, _chain, rows = _full_flow(
        ctx,
        monkeypatch,
        tmp_path,
        [(BUTTON, 0.4), (UP, 9.0)],
        press_only=press_only,
        ends={3: trp.REJECTED},
        home_refusals=home_refusals,
    )
    assert code == 1
    after = ["depth:clear", "home"] if not home_refusals else ["depth:clear", "reach:back-off"]
    assert rec.events == ["reach:look", "aim:reach-aim", *WAY_IN, *after]
    turned = list(rec.legs["depth:turn"].traj.points[-1].positions)
    assert list(rec.legs["depth:clear"].traj.points[0].positions) == pytest.approx(turned)  # from the stroke's start
    assert ctx.client.primes == [None] and "hold" not in rec.events  # the spring came in; nothing was pressed
    out = capsys.readouterr().out
    assert "the way in did not start" not in out
    said = "STOP: the press stroke did not start (goal rejected by the driver) — nothing was pressed"
    assert said + " — returning home" in out
    last = [line for line in out.splitlines() if "STOP" in line][-1]  # what the node reads the run's end from
    if home_refusals:
        assert "the arm did not get home — it holds" in last
        assert ctx.client.lets_go == [len(ctx.client.executed)]  # let go where it stands, after its last goal
    else:
        assert "nothing was pressed" in last and "holds" not in last
        assert ctx.client.lets_go == []


@pytest.mark.parametrize("press_only", [False, True], ids=["the full run", "the press alone"])
def test_in_position_a_spring_the_driver_was_not_asked_for_flies_no_stroke_and_the_arm_goes_home(
    ctx, monkeypatch, tmp_path, capsys, press_only
):
    """The stroke flies on the driver's joint impedance, entered at rest at its start by the hold
    client.prime_compliant sends. Where that hold did not go out (no fresh /joint_states, the driver's
    action server not answering) the stroke flew all the same, the first impedance goal after a position
    one, on a spring still fading in: behind its reference, which the contact watch reads as a box met
    early (review 2026-11-16). It is asked for once more after AIR_PRIME_RETRY_S; still not sent, no stroke
    is flown: nothing pressed, said for what it is, and the arm goes home from over the box."""
    from box_opening.constants import AIR_SWITCH_S

    monkeypatch.setenv("BOX_OPENING_AIR_CONTROL", "position")
    waits = []
    monkeypatch.setattr(rp.time, "sleep", lambda s: waits.append(s))
    asked = []  # each ask: how many goals had gone out by then
    monkeypatch.setattr(ctx.client, "prime_compliant", lambda q=None: asked.append(len(ctx.client.executed)) or False)
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], press_only=press_only)
    assert code == 1
    assert asked == [2, 2]  # twice, the look and the free air flown, nothing sent between
    assert waits.count(rp.AIR_PRIME_RETRY_S) == 1 and AIR_SWITCH_S not in waits  # no spring to wait in
    # no stroke: its legs never reached the runner, and nothing was held on the button
    assert rec.events == ["reach:look", "aim:reach-aim", *WAY_IN[:-1], "depth:clear", "home"]
    (switch,) = (r for r in rows if r["leg"] == "air_switch")
    assert (switch["primed"], switch["tries"], switch["wait_s"]) == (False, 2, 0.0)
    assert not [r for r in rows if r["leg"] in ("reach_way_in_flight", "press_refused")]
    out = capsys.readouterr().out
    assert "the way in did not start" not in out
    last = [line for line in out.splitlines() if "STOP" in line][-1]  # what the node reads the run's end from
    assert "STOP: the press stroke did not start (the driver did not enter its joint impedance" in last
    assert "nothing was pressed" in last and "holds" not in last
    assert ctx.client.lets_go == []  # home: nothing let go


def test_in_position_a_spring_not_asked_for_at_first_is_asked_for_once_more_and_the_press_flies(
    ctx, monkeypatch, tmp_path
):
    """A hiccup of the joint stream or of the driver's action server at the one moment the hold is sent
    does not cost the run: asked once more after AIR_PRIME_RETRY_S, sent, its fade-in waited as ever, and
    the press and all after it fly as they do."""
    from box_opening.constants import AIR_SWITCH_S

    monkeypatch.setenv("BOX_OPENING_AIR_CONTROL", "position")
    waits = []
    monkeypatch.setattr(rp.time, "sleep", lambda s: waits.append(s))
    answers = [False, True]
    monkeypatch.setattr(ctx.client, "prime_compliant", lambda q=None: answers.pop(0))
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)])
    assert code == 0 and answers == []
    assert [names for names, _g in _executions(rec)][:3] == [["reach:look"], WAY_IN[:-1], WAY_IN[-1:]]
    (switch,) = (r for r in rows if r["leg"] == "air_switch")
    assert (switch["primed"], switch["tries"], switch["wait_s"]) == (True, 2, AIR_SWITCH_S)
    # the wait before the second ask, then the spring's fade-in, in that order, each once
    assert [s for s in waits if s in (rp.AIR_PRIME_RETRY_S, AIR_SWITCH_S)] == [rp.AIR_PRIME_RETRY_S, AIR_SWITCH_S]


def test_in_position_a_free_air_part_that_does_not_start_still_goes_home_from_the_look_pose(
    ctx, monkeypatch, tmp_path, capsys
):
    """The free air's own execution turned down: nothing moved, the arm stands at the look pose, and home
    is planned from there as it always was — no clear, which is for a hand over the box."""
    monkeypatch.setenv("BOX_OPENING_AIR_CONTROL", "position")
    monkeypatch.setattr(rp.time, "sleep", lambda s: None)
    code, rec, _chain, _rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], ends={2: trp.REJECTED}
    )
    assert code == 1
    assert rec.events == ["reach:look", "aim:reach-aim", *WAY_IN[:-1], "home"]
    assert ctx.client.primes == [] and ctx.client.lets_go == []
    assert (
        "STOP: the way in did not start (goal rejected by the driver) — nothing was pressed" in capsys.readouterr().out
    )


def test_where_nothing_was_planned_ahead_the_lift_flies_alone_and_the_spot_is_looked_at_after_it(
    ctx, monkeypatch, tmp_path, capsys
):
    """No spot whose carry and home both plan found while the hand came
    down to the knob (here: none looked for): as before, the close, the
    lift and the clear, then the lid's one stop, the spot look; the carry
    and the set-down; the release, the retreat and home — six executions
    after the look's, every one unguarded and RIGID but home's, slow under
    MEDIUM (core.as_home)."""
    code, rec, chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], one_motion=False)
    assert code == 0
    assert rec.events == [
        "reach:look",
        "aim:reach-aim",
        *WAY_IN,
        "hold",
        *BACK,
        "aim:reach-after",
        *GRIP,
        *CLOSE,
        "spot-look",
        *CARRY,
        *HOME,
    ]
    runs = _executions(rec)
    assert [names for names, _g in runs] == [["reach:look"], WAY_IN, BACK, GRIP, CLOSE, CARRY, RELEASED, HOMED]
    # one goal each, every one under the run's one gains spec, home under its own
    for names, gains in runs[1:-1]:
        assert gains == [dp.COMPLIANT_GAINS], names
    assert runs[-1] == (HOMED, [GAINS_MEDIUM])
    assert all(lg.guard is None for lg in rec.legs.values() if lg.kind.value == "motion")
    # the carry planned while the lift and the clear flew (carry_ahead), before the spot look, and flown: the spot
    # chosen with the lid lifted is its own; home while the retreat flew, not while the set-down did (its watch
    # is polled from its first sample: test_the_set_down_hosts_no_planning_...)
    planned = {name: at for name, at in rec.plans}
    assert planned["lid:carry"] == rec.events.index("lid:clear") + 1 == rec.events.index("spot-look")
    assert [name for name, _at in rec.plans].count("lid:carry") == 1
    assert planned["home"] == rec.events.index("place:retreat") + 1 == rec.events.index("home")
    # ... and from the look pose while the way back flew, for a stop there: unused, the lid coming off
    assert [at for name, at in rec.plans if name == "home"] == [rec.events.index("aim:reach-after"), planned["home"]]
    # nothing chose the lid's spot before the lift: it is chosen at the spot look, on the table the cameras measured
    assert [r["leg"] for r in rows].index("lid_spot") > [r["leg"] for r in rows].index("reach_grip_result")
    (spot,) = (r for r in rows if r["leg"] == "lid_spot")
    assert spot["tier"] == "clear" and spot["mapped"] is True
    assert spot["xyz"] == pytest.approx(list(ctx.lid_drop.xyz), abs=1e-4)
    hover = rec.legs["lid:carry"].target[1]
    assert hover[:2] == pytest.approx(list(ctx.lid_drop.xyz[:2]))
    assert ctx.lid_at is ctx.lid_drop  # the placed lid is in every world after the release
    (knob,) = (r for r in rows if r["leg"] == "reach_knob")
    assert (knob["press"], knob["popped"], knob["looks"], knob["knob_up_mm"]) == (1, True, 1, 9.0)
    # the grip goes by the lid GRIP_LID_ABOVE_HELD_M above where the hold
    # held the pads (6 mm below the top), whatever the top
    (grip,) = (r for r in rows if r["leg"] == "reach_grip")
    lid = BUTTON[2] - 0.006 + rlo.GRIP_LID_ABOVE_HELD_M
    assert grip["lid_z"] == pytest.approx(lid) and grip["pads_z"] == pytest.approx(lid - rlo.GRIP_BELOW_LID_M)
    out = capsys.readouterr().out
    assert "LID GRIPPED" in out and out.rstrip().endswith("arm home")
    assert ctx.client.lets_go == []  # nothing stopped: nothing let go


def test_a_lid_that_rose_with_its_knob_up_is_the_box_open_and_the_grip_follows_it_up(
    ctx, monkeypatch, tmp_path, capsys
):
    """2026-10-28 16:15:19: the press let the lid go and it rose 9.6 mm, the
    button 5.8 mm from where it was aimed, its knob 7.8 mm up — read as "the
    press moved the box" and the arm went home with the box open. A lid
    LID_POP_M higher with the button in place is the box open, and its knob
    up is something to hold: no press once more, and the grip goes by the
    lid where it now stands."""
    risen = (BUTTON[0] + 0.0058, BUTTON[1], BUTTON[2] - 0.0004 + 0.0096 + 0.0078)  # on the first aim's lid
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (risen, 7.8)])
    assert code == 0
    runs = _executions(rec)
    assert [names for names, _g in runs] == [
        ["reach:look"],
        WAY_IN,
        BACK,
        GRIP,
        CLOSE + CARRY,
        RELEASED,
        HOMED,
    ]  # pressed once
    (knob,) = (r for r in rows if r["leg"] == "reach_knob")
    assert (knob["popped"], knob["lid_rose"], knob["knob_up_mm"], knob["stop"]) == (True, True, 7.8, None)
    assert knob["lid_dz_mm"] == pytest.approx(9.6)
    (grip,) = (r for r in rows if r["leg"] == "reach_grip")
    assert grip["lid_z"] == pytest.approx(
        BUTTON[2] - 0.006 + rlo.GRIP_LID_ABOVE_HELD_M + 0.0096
    )  # the held-pads lid raised by the rise
    out = capsys.readouterr().out
    assert "knob UP 7.8 mm, and its LID ROSE +10 mm — the box is open" in out and "LID GRIPPED" in out


@pytest.mark.parametrize(
    "knob_mm, dz, shift", [(-0.6, 0.0082, 0.0046), (0.9, 0.0094, 0.0030)], ids=["16:23:53", "16:15:52"]
)
def test_a_lid_that_rose_with_its_knob_flush_is_neither_gripped_nor_pressed_again_the_arm_goes_home(
    ctx, monkeypatch, tmp_path, capsys, knob_mm, dz, shift
):
    """2026-10-28 16:23:53: the press let the lid go and it rose 8.2 mm, the
    button 4.6 mm from where it was aimed, its knob -0.6 mm above it.
    Gripped as the box open, the fingers closed on air (0.768 against the
    band [0.30, 0.55]), opened, backed out and went home: a flush knob gives
    them nothing to hold — nor would 16:15:52's (the lid 9.4 mm up, the
    button 3.0 mm off, its knob 0.9). The look stops the run: home from the
    look pose, the fingers never opened, no press once more (a released lid
    is not pressed again), exit 1 — the record keeps the rise and says why."""
    risen = (BUTTON[0] + shift, BUTTON[1], BUTTON[2] - 0.0004 + dz + knob_mm / 1000)  # on the first aim's lid
    aims = [(BUTTON, 0.4), (risen, knob_mm), (risen, knob_mm)]
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, aims, knuckle=AIR)
    assert code == 1
    assert rec.events == [
        "reach:look",
        "aim:reach-aim",
        *WAY_IN,
        "hold",
        *BACK,
        "aim:reach-after",
        "aim:reach-after",
        "home",
    ]
    assert ctx.client.gripper_sent == [] and ctx.client.lets_go == []
    (knob,) = (r for r in rows if r["leg"] == "reach_knob")
    assert (knob["popped"], knob["lid_rose"], knob["knob_up_mm"], knob["looks"]) == (False, True, knob_mm, 2)
    assert knob["lid_dz_mm"] == pytest.approx(1000 * dz) and knob["shift_mm"] == pytest.approx(1000 * shift)
    assert knob["stop"].startswith("the lid rose %+.0f mm (released) but its knob is not up" % (1000 * dz))
    assert not [r for r in rows if r["leg"] in ("reach_grip", "reach_grip_result", "lid_spot")]
    out = capsys.readouterr().out
    assert "knob %.1f mm above the lid, and its LID ROSE %+.0f mm — the box is open" % (knob_mm, 1000 * dz) in out
    assert (
        "STOP: the lid rose %+.0f mm (released) but its knob is not up (%.1f mm above the lid) — the fingers have "
        "nothing to hold, so the lid is not taken off; the box is left open, and the next run would take it for "
        "shut and press it (its aim judges a box by its knob alone) — returning home" % (1000 * dz, knob_mm)
    ) in out
    assert "pressing once more" not in out and "GRIP" not in out


def test_a_lid_that_rose_at_the_press_once_more_with_its_knob_flush_stops_the_run_there(
    ctx, monkeypatch, tmp_path, capsys
):
    """Not up after the first press, its lid where it was: pressed once more
    (owner 2026-10-28). That press lets the lid go, its knob flush: the same
    stop — home from the look pose, nothing gripped."""
    risen = (BUTTON[0], BUTTON[1], BUTTON[2] + 0.0099)  # the lid +9.4 mm on the first aim's -0.4, its knob 0.9 above
    aims = [(BUTTON, 0.4), (BUTTON, 2.0), (BUTTON, 1.5), (risen, 0.9), (risen, 0.9)]
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, aims)
    assert code == 1
    press = [*WAY_IN, "hold", *BACK]
    assert rec.events == [
        "reach:look",
        "aim:reach-aim",
        *press,
        "aim:reach-after",
        "aim:reach-after",
        *press,
        "aim:reach-after",
        "aim:reach-after",
        "home",
    ]
    knobs = [(r["press"], r["popped"], r["lid_rose"], r["stop"] is None) for r in rows if r["leg"] == "reach_knob"]
    assert knobs == [(1, False, False, True), (2, False, True, False)]
    out = capsys.readouterr().out
    assert "pressing once more" in out and "STOP: the lid rose +9 mm (released) but its knob is not up" in out
    assert "pressed 2 times" not in out and ctx.client.gripper_sent == []


@pytest.mark.parametrize("off_mm", [-0.1, 0.0], ids=["just under", "at"])
def test_a_risen_lid_is_gripped_with_its_knob_up_as_a_shut_one_and_under_that_the_stop_says_not_up(
    ctx, monkeypatch, tmp_path, capsys, off_mm
):
    """On a risen lid the knob is up where it is on a shut one
    (press_demo.knob_is_up, KNOB_UP_MIN_MM or more): gripped. Just under,
    flush or partly up, it is not up: the stop, which says so with the
    reading."""
    from box_opening.tasks.press_demo import KNOB_UP_MIN_MM

    knob_mm, up = KNOB_UP_MIN_MM + off_mm, off_mm >= 0
    risen = (BUTTON[0], BUTTON[1], BUTTON[2] - 0.0004 + 0.009 + knob_mm / 1000)  # the lid +9 mm on the first aim's -0.4
    aims = [(BUTTON, 0.4), (risen, knob_mm)] + ([] if up else [(risen, knob_mm)])  # a knob not up is read twice
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, aims)
    (knob,) = (r for r in rows if r["leg"] == "reach_knob")
    assert (code, knob["popped"], knob["lid_rose"], knob["stop"] is None) == (0 if up else 1, up, True, up)
    assert any(n.startswith("grip:") for n in rec.events) == up
    said = "STOP: the lid rose +9 mm (released) but its knob is not up (%.1f mm above the lid)" % knob_mm
    assert (said in capsys.readouterr().out) == (not up)


@pytest.mark.parametrize("dz", [-0.009, 0.040])
def test_a_lid_lower_or_far_higher_after_the_press_is_still_the_box_moved(ctx, monkeypatch, tmp_path, dz):
    """Lower than before the press, or higher than a released lid rises
    (LID_POP_M): the press moved the box — home, nothing gripped."""
    after = (BUTTON[0], BUTTON[1], BUTTON[2] - 0.0004 + dz + 0.0009)
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (after, 0.9), (after, 0.9)])
    assert code == 1
    (knob,) = (r for r in rows if r["leg"] == "reach_knob")
    assert (knob["popped"], knob["lid_rose"]) == (False, False)
    assert knob["stop"].startswith("the press moved the box: its lid")
    assert not any(n.startswith("grip:") for n in rec.events)


def test_a_knob_not_up_is_looked_at_again_then_pressed_once_more_and_its_lid_taken_off(
    ctx, monkeypatch, tmp_path, capsys
):
    """Not up after the press, nor at the look again: the same press once
    more from the look pose, its way in solved again from where the arm
    stands, the hold, the way back and the look; up — the lid comes off."""
    aims = [(BUTTON, 0.4), (BUTTON, 2.0), (BUTTON, 1.5), (UP, 8.0)]
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, aims)
    assert code == 0
    press = [*WAY_IN, "hold", *BACK]
    assert rec.events == [
        "reach:look",
        "aim:reach-aim",
        *press,
        "aim:reach-after",
        "aim:reach-after",
        *press,
        "aim:reach-after",
        *GRIP,
        *CLOSE,
        *CARRY,
        *HOME,
    ]
    knobs = [(r["press"], r["popped"], r["looks"], r["knob_up_mm"]) for r in rows if r["leg"] == "reach_knob"]
    assert knobs == [(1, False, 2, 2.0), (2, True, 1, 8.0)]
    assert "pressing once more — the same press, from the look pose" in capsys.readouterr().out


def test_a_knob_still_not_up_after_the_second_press_sends_the_arm_home_with_nothing_gripped(
    ctx, monkeypatch, tmp_path, capsys
):
    aims = [(BUTTON, 0.4), (BUTTON, 2.0), (BUTTON, 1.5), (BUTTON, 2.8), (BUTTON, 2.5)]  # flush: under 3.0
    code, rec, _chain, _rows = _full_flow(ctx, monkeypatch, tmp_path, aims)
    assert code == 1
    press = [*WAY_IN, "hold", *BACK]
    assert rec.events == [
        "reach:look",
        "aim:reach-aim",
        *press,
        "aim:reach-after",
        "aim:reach-after",
        *press,
        "aim:reach-after",
        "aim:reach-after",
        "home",
    ]
    out = capsys.readouterr().out
    assert "STOP: pressed 2 times and the knob is not up (2.8 mm above the lid) — the lid is not taken off" in out
    assert ctx.client.gripper_sent == []  # the fingers never opened


@pytest.mark.parametrize(
    "aims, presses",
    [
        ([(BUTTON, 0.4), (BUTTON, 2.0), (BUTTON, 4.2)], 1),  # the recheck reads 4.2: the highest decides
        ([(BUTTON, 0.4), (BUTTON, 5.6), (BUTTON, 5.1)], 1),
        ([(BUTTON, 0.4), (BUTTON, 2.0), (BUTTON, 1.5), (BUTTON, 3.0), (BUTTON, 2.5)], 2),  # 3.0 is not flush
    ],
    ids=["after the first press", "read twice in between", "after the second press"],
)
def test_a_knob_neither_flush_nor_up_after_a_press_is_not_pressed_again_but_gripped_as_an_open_boxs(
    ctx, monkeypatch, tmp_path, capsys, aims, presses
):
    """The wrist study of 2026-11-05 (I-Y2, H5): the tilted look reads
    popped knobs down to 5.0 mm, and in its physics study a second press on
    a popped knob latched the box again. After a press only a knob read
    flush (below press_demo.KNOB_SHUT_BELOW_MM) is pressed again; one read
    3.0 mm or more, but under KNOB_UP_MIN_MM, is gripped as an open box's."""
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, aims)
    assert code == 0
    press = [*WAY_IN, "hold", *BACK]
    looked = ["aim:reach-after", "aim:reach-after"]
    assert rec.events == [
        "reach:look",
        "aim:reach-aim",
        *(press + looked) * presses,
        *GRIP,
        *CLOSE,
        *CARRY,
        *HOME,
    ]
    knobs = [(r["press"], r["popped"], r["call"]) for r in rows if r["leg"] == "reach_knob"]
    assert knobs == ([(1, False, "shut"), (2, False, "band")] if presses == 2 else [(1, False, "band")])
    want = max(v for _p, v in aims[-2:])
    assert "the knob is neither flush nor up (%.1f mm above the lid): not pressing it again" % want in (
        capsys.readouterr().out
    )


def test_a_knob_whose_height_the_look_does_not_read_after_a_press_is_not_pressed_again(
    ctx, monkeypatch, tmp_path, capsys
):
    """The button found after the press, but no look read the knob's
    height: it is not judged, so it is neither pressed again nor gripped."""
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (BUTTON, None), (BUTTON, None)])
    assert code == 1 and rec.events[-3:] == ["aim:reach-after", "aim:reach-after", "home"]
    assert rec.events.count("depth:press") == 1 and ctx.client.gripper_sent == []
    (knob,) = (r for r in rows if r["leg"] == "reach_knob")
    assert (knob["knob_up_mm"], knob["call"]) == (None, "unread")
    assert "STOP: the wrist read no height for the knob after press 1 — it is not pressed again" in (
        capsys.readouterr().out
    )


BACK_OUT = ["grip:up", "depth:turn-back", "reach:slide-out"]  # the grip's way down backwards, to the look pose


def test_fingers_that_close_on_air_back_out_shut_the_way_they_came(ctx, monkeypatch, tmp_path, capsys):
    """The close reads air (outside the grip band, at its target: nothing
    between the pads): nothing is lifted. The fingers stay shut (no 0.7 s
    open first), the hand backs out the grip's way down to the look pose,
    the wrist aims at the knob again there (_regrip) — here it finds none,
    so the second grip comes down where the first went, GRIP_RETRY_LOWER_M
    lower; on air again, it backs out once more and goes home: exit 1."""
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], knuckle=AIR)
    assert code == 1
    assert rec.events == [
        "reach:look",
        "aim:reach-aim",
        *WAY_IN,
        "hold",
        *BACK,
        "aim:reach-after",
        *GRIP,
        *CLOSE,
        *CARRY,
        *BACK_OUT,
        "aim:reach-regrip",
        *GRIP,
        *CLOSE,
        *CARRY,
        *BACK_OUT,
        "home",
    ]
    assert (CLOSE + CARRY, []) in _executions(rec)  # the close's run returned at the close: handed, nothing sent
    assert [r["leg"] for r in rows if r["leg"].startswith(("lid:", "place:"))] == []
    results = [r for r in rows if r["leg"] == "reach_grip_result"]
    assert [(r["ok"], r["retry"]) for r in results] == [(False, True), (False, False)] and "0.790" in results[0][
        "detail"
    ]
    out = capsys.readouterr().out
    assert "GRIP MISSED" in out and "coming down 10 mm lower" in out
    assert (
        "the wrist does not find the button again from the look pose (no circle): gripping once more where the "
        "first grip went" in out
    )


# ---- a grip on air, tried once more at the knob aimed at again -----------------

# 12:03:52 2026-10-29: the grip came down where the press was aimed, 11 mm
# short of the knob; from the look pose the wrist sees it there
REGRIP = (BUTTON[0] + 0.011, BUTTON[1], BUTTON[2] + 0.009)


def _grip_ways(monkeypatch, refuse=()):
    """solve_grip_way with the button each call went by kept, the calls
    numbered in `refuse` (the first is 1) unsolved."""
    real, asked = rlo.solve_grip_way, []

    def solve(ctx_, cfg_, button, tilt, q_from, lid_z):
        asked.append((tuple(float(v) for v in button), float(lid_z), tilt))
        if len(asked) in refuse:
            return None, "no straight-down hand reaches the knob"
        return real(ctx_, cfg_, button, tilt, q_from, lid_z)

    monkeypatch.setattr(rlo, "solve_grip_way", solve)
    return asked


def test_a_grip_on_air_is_aimed_again_from_the_look_pose_and_gripped_once_more_where_the_knob_is(
    ctx, monkeypatch, tmp_path, capsys
):
    """12:03:52 2026-10-29: the fingers closed on air (0.789 against the
    band [0.30, 0.55]) 11 mm short of the knob, and the run went home, exit
    1. Now the hand backs out to the look pose (the fingers, shut on nothing, stay shut), the wrist
    aims at the knob again from there — once — and the hand comes down to
    the knob it sees, the first grip's tilt kept, GRIP_RETRY_LOWER_M (1 cm)
    lower (owner 2026-10-30): the second close holds it, and the lid comes
    off as ever, exit 0. The record says each step."""
    asked = _grip_ways(monkeypatch)
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0), (REGRIP, 9.0)], knuckle=[AIR, KNOB_ON]
    )
    assert code == 0
    assert rec.events == [
        "reach:look",
        "aim:reach-aim",
        *WAY_IN,
        "hold",
        *BACK,
        "aim:reach-after",
        *GRIP,
        *CLOSE,
        *CARRY,
        *BACK_OUT,
        "aim:reach-regrip",
        *GRIP,
        *CLOSE,
        *CARRY,
        *HOME,
    ]
    assert rec.events.count("aim:reach-regrip") == 1 and rec.events.count("grip:close") == 2
    # the re-aim at the popped knob asks for REGRIP_AIM_HITS agreeing hits; every other aim the reach-out's own
    assert [h for tag, h in rec.aim_hits if tag == "reach-regrip"] == [rlo.REGRIP_AIM_HITS] == [3]
    assert all(h is None for tag, h in rec.aim_hits if tag != "reach-regrip")
    # the second way down solved for the re-aimed button, at the first's height, lid and tilt
    (first, lid1, tilt1), (second, lid2, tilt2) = asked
    assert first == pytest.approx(BUTTON) and second == pytest.approx((REGRIP[0], REGRIP[1], BUTTON[2]))
    assert tilt2 == tilt1 and lid2 == pytest.approx(lid1 - rlo.GRIP_RETRY_LOWER_M) and rlo.GRIP_RETRY_LOWER_M == 0.010
    grips = [r for r in rows if r["leg"] == "reach_grip"]
    assert grips[1]["lid_z"] == pytest.approx(grips[0]["lid_z"] - 0.010, abs=1e-4)
    assert grips[1]["pads_z"] == pytest.approx(grips[0]["pads_z"] - 0.010, abs=1e-4)
    grips = [r for r in rows if r["leg"] == "reach_grip"]
    assert [(g["attempt"], g["button"][:2]) for g in grips] == [
        (1, pytest.approx(list(BUTTON[:2]))),
        (2, pytest.approx(list(REGRIP[:2]))),
    ]
    (regrip,) = (r for r in rows if r["leg"] == "reach_regrip")
    assert (regrip["found"], regrip["outcome"], regrip["knob_up_mm"]) == (True, "grip again", 9.0)
    assert regrip["shift_mm"] == pytest.approx(11.0) and regrip["xyz"] == pytest.approx(list(REGRIP))
    results = [
        (r["attempt"], r["ok"], r.get("on_air"), r.get("retry")) for r in rows if r["leg"] == "reach_grip_result"
    ]
    assert results == [(1, False, True, True), (2, True, None, None)]
    out = capsys.readouterr().out
    assert "GRIP MISSED" in out and "RE-AIMED from the look pose" in out and "GRIP ONCE MORE" in out
    assert "LID GRIPPED" in out and out.rstrip().endswith("arm home")
    assert ctx.client.lets_go == []  # nothing stopped part-way: nothing let go


def test_a_grip_on_air_twice_backs_out_and_goes_home_after_one_retry_only(ctx, monkeypatch, tmp_path, capsys):
    """The second close reads air as well: the hand backs out to the look
    pose once more, the fingers shut, and goes home from there, exit 1 —
    never a third grip, nor a second re-aim, whatever the wrist would see."""
    plans = _home_plans(monkeypatch)
    aims = [(BUTTON, 0.4), (UP, 9.0), (REGRIP, 9.0), (REGRIP, 9.0)]
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, aims, knuckle=AIR)
    assert code == 1
    assert rec.events == [
        "reach:look",
        "aim:reach-aim",
        *WAY_IN,
        "hold",
        *BACK,
        "aim:reach-after",
        *GRIP,
        *CLOSE,
        *CARRY,
        *BACK_OUT,
        "aim:reach-regrip",
        *GRIP,
        *CLOSE,
        *CARRY,
        *BACK_OUT,
        "home",
    ]
    results = [(r["attempt"], r["ok"], r["on_air"], r["retry"]) for r in rows if r["leg"] == "reach_grip_result"]
    assert results == [(1, False, True, True), (2, False, True, False)]
    assert len([r for r in rows if r["leg"] == "reach_regrip"]) == 1
    # home flown is the one planned while the second back-out flew, from where it ends
    assert rec.legs["home"] is plans[-1][1][-1] and plans[-1][0] == pytest.approx(
        rec.legs["reach:slide-out"].goal_joints
    )
    out = capsys.readouterr().out
    assert "GRIP FAILED AGAIN, 10 mm lower" in out
    assert "STOP: the grip failed twice, the second time 10 mm lower (grip 0.790" in out


def test_a_re_aim_that_finds_no_knob_grips_once_more_where_the_first_went(ctx, monkeypatch, tmp_path, capsys):
    """A re-aim that finds no button sends the second grip where the first
    went — the press's and the look's aim — GRIP_RETRY_LOWER_M lower: it
    holds the knob, and the lid comes off."""
    asked = _grip_ways(monkeypatch)
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0), None], knuckle=[AIR, KNOB_ON]
    )
    assert code == 0 and rec.events.count("grip:close") == 2
    (first, lid1, _t1), (second, lid2, _t2) = asked
    assert second == pytest.approx(first) and lid2 == pytest.approx(lid1 - rlo.GRIP_RETRY_LOWER_M)
    (note,) = (r for r in rows if r["leg"] == "reach_regrip")
    assert (note["found"], note["outcome"]) == (False, "grip again where the first went")
    assert (
        "the wrist does not find the button again from the look pose (no circle): gripping once more where the "
        "first grip went" in capsys.readouterr().out
    )


@pytest.mark.parametrize("shift, regrips", [(0.0193, True), (0.045, False)], ids=["19 mm: 10:58", "45 mm"])
def test_a_knob_the_re_aim_finds_nudged_is_gripped_where_it_is_and_one_far_off_is_the_box_moved(
    ctx, monkeypatch, tmp_path, capsys, shift, regrips
):
    """2026-10-30 10:58: the fingers closed over the knob, and the re-aim —
    once it no longer took the lid's 60 mm ring for the knob — saw it 19 mm
    from where the first grip went: the lid nudged. Within REGRIP_MAX_M the
    second grip goes where the knob is now, GRIP_RETRY_LOWER_M lower;
    farther, the box moved, and the arm goes home from the look pose."""
    asked = _grip_ways(monkeypatch)
    nudged = (BUTTON[0] + shift, BUTTON[1], BUTTON[2] + 0.009)
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0), (nudged, 7.7)], knuckle=[AIR, KNOB_ON]
    )
    (note,) = (r for r in rows if r["leg"] == "reach_regrip")
    if regrips:
        assert code == 0 and rec.events.count("grip:close") == 2 and note["outcome"] == "grip again"
        (_first, lid1, _t1), (second, lid2, _t2) = asked
        assert second[:2] == pytest.approx(nudged[:2]) and lid2 == pytest.approx(lid1 - rlo.GRIP_RETRY_LOWER_M)
        assert "RE-AIMED from the look pose" in capsys.readouterr().out
    else:
        assert code == 1 and rec.events[-1] == "home" and note["outcome"] == "home" and len(asked) == 1
        assert "mm from where the first grip went — the box moved" in capsys.readouterr().out


def test_a_re_aim_with_no_way_down_to_the_knob_sends_the_arm_home_from_the_look_pose(
    ctx, monkeypatch, tmp_path, capsys
):
    """After a close on air the re-aim finds the knob, but no way down to it
    solves: home from the look pose, flying the home planned while the
    back-out flew (no planner call at rest), exit 1 — nothing gripped again."""
    _grip_ways(monkeypatch, (2,))
    plans = _home_plans(monkeypatch)
    aims = [(BUTTON, 0.4), (UP, 9.0), (REGRIP, 9.0)]
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, aims, knuckle=[AIR, KNOB_ON])
    assert code == 1
    assert rec.events == [
        "reach:look",
        "aim:reach-aim",
        *WAY_IN,
        "hold",
        *BACK,
        "aim:reach-after",
        *GRIP,
        *CLOSE,
        *CARRY,
        *BACK_OUT,
        "aim:reach-regrip",
        "home",
    ]
    (note,) = (r for r in rows if r["leg"] == "reach_regrip")
    assert (note["found"], note["outcome"]) == (True, "grip again")
    assert [r["attempt"] for r in rows if r["leg"] == "reach_grip_refused"] == [2]
    (ahead,) = (r for r in rows if r["leg"] == "home_ahead")
    assert ahead["flown"] is True and rec.legs["home"] is plans[-1][1][-1]
    assert (
        "STOP: the hand does not come down to the re-aimed knob (no straight-down hand reaches the knob) — the "
        "lid is not taken off" in capsys.readouterr().out
    )


def test_a_grip_that_fails_but_not_on_air_is_not_tried_again(ctx, monkeypatch, tmp_path, capsys):
    """Fingers stopped wide, below the grip band, closed on something larger
    than the knob: not a miss a re-aim mends. They open, the hand backs out
    and goes home, as before: no re-aim. Nor is a gripper that failed, or an
    open, ever a close on air (_on_air)."""
    from box_opening.runtime.runner import LegResult

    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0), (REGRIP, 9.0)], knuckle=0.2
    )
    assert code == 1
    assert rec.events[-len(BACK_OUT) - 5 :] == [*CARRY, "grip:open", *BACK_OUT, "home"]
    assert "aim:reach-regrip" not in rec.events and not [r for r in rows if r["leg"] == "reach_regrip"]
    (result,) = (r for r in rows if r["leg"] == "reach_grip_result")
    assert (result["on_air"], result["retry"]) == (False, False)
    assert "GRIP FAILED" in capsys.readouterr().out
    band = (0.30, 0.55)
    assert rlo._on_air(LegResult("grip:close", "arrived", False, "", gripper_pos=0.789), band)
    assert not rlo._on_air(LegResult("grip:close", "arrived", False, "", gripper_pos=0.2), band)
    assert not rlo._on_air(LegResult("grip:close", "failed", False, "", gripper_pos=0.789), band)
    assert not rlo._on_air(LegResult("grip:open", "arrived", False, "", gripper_pos=0.789), band)
    assert not rlo._on_air(LegResult("grip:close", "arrived", True, "band unchecked"), band)


class _ReachingChain(_Chain):
    """_Chain whose fk puts the hand where the straight line that ended at
    those joints went (its IK reaching the target), so a line from where the
    arm stands to somewhere else has a length; and whose turning walk flies
    (the way back to the look pose from a rise off the stroke)."""

    def __init__(self):
        super().__init__()
        self.at = {}

    def straight_line(self, q0, xyz, quat, max_step_rad=None):
        pts, why = super().straight_line(q0, xyz, quat, max_step_rad)
        if pts:
            self.at[tuple(round(float(v), 9) for v in pts[-1])] = list(xyz)
        return pts, why

    def walk(self, q0, xyz, quat, step_m=None, step_rad=None, max_step_rad=None):
        return self.straight_line(q0, xyz, quat)

    def fk(self, q):
        R, t = super().fk(q)
        return R, np.array(self.at.get(tuple(round(float(v), 9) for v in q), t), float)


FREE_AIR_RUNS = {
    "lid off": {},
    "grip on air": dict(knuckle=AIR),
    # no way back up the stroke: a straight line to its start, then the walk from there to the look pose
    "off the stroke": dict(rise=lambda chain_, stroke, q, join_above_m=None: None, chain=_ReachingChain()),
    # no home plans after the set-down, the carry's lookahead's nor again (plan_home_after): the retreat alone,
    # then home from up there
    "no home after the set-down": {},
    # the release's retreat and home turned down by the driver (the 6th goal: the lift to the set-down is the
    # 5th): the retreat alone, then home
    "the retreat and home refused": dict(ends={6: trp.REJECTED}, chain=_ReachingChain()),
}


@pytest.mark.parametrize("case", list(FREE_AIR_RUNS))
def test_the_full_runs_free_air_lines_fly_at_the_compliant_speed_and_only_under_the_wrists_ceiling(
    ctx, monkeypatch, tmp_path, case
):
    """Every plan-free line of the full run that moves in free air — the
    re-aim, the slide in and the turn of the way in and of the grip's, the
    rise, the turn back and the slide out (or, from a rise off the stroke,
    the walk back to the look pose), the grip's way back up after a close on
    air, the clear with the lid and the retreat from it (with home, or alone
    where home does not plan or fly) — flies at COMPLIANT_FREE_LINE_SPEED,
    and every execution carrying one is re-timed under the wrist's
    COMPLIANT_QD_CEILING and meets its legs at COMPLIANT_JUNCTION_SPEED
    (Runner.retime_params). The legs that meet the button, the knob or the
    table keep their own speeds, the planned ones TRANSIT_SPEED, home
    HOME_SPEED (constants: every home flies slow, under MEDIUM). Outside
    the run, the client as press_demo built it, free_line_speed is a
    guarded run's FREE_LINE_SPEED."""
    from box_opening.constants import (
        COMPLIANT_FREE_LINE_SPEED,
        COMPLIANT_JUNCTION_SPEED,
        COMPLIANT_QD_CEILING,
        FREE_LINE_SPEED,
        HOME_SPEED,
        TRANSIT_SPEED,
    )
    from box_opening.runtime import runner as runner_mod
    from box_opening.runtime.retime import RetimeParams
    from box_opening.runtime.runner import Runner

    flown, timed = [], []  # (name, speed) of every motion leg handed to the runner; (speeds, params) per re-timing
    real_run, real_group = Runner.run, runner_mod.retime_group

    def run(self, legs, execute, lookahead=None):
        flown.extend((lg.name, lg.speed) for lg in legs if lg.kind.value == "motion")
        return real_run(self, legs, execute, lookahead)

    def retime_group(trajs, speeds, vmax, params=RetimeParams(), folds=None, step_caps=None):  # noqa: B008 - frozen dataclass
        timed.append((list(speeds), params))
        return real_group(trajs, speeds, vmax, params, folds, step_caps)

    monkeypatch.setattr(Runner, "run", run)
    monkeypatch.setattr(runner_mod, "retime_group", retime_group)
    if case == "no home after the set-down":
        monkeypatch.setattr(
            rlo,
            "plan_home_after",
            lambda ctx_, cfg_, carry: (
                None,
                "home after the set-down "
                "(planning failed for home: MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION)",
            ),
        )
    assert rp.free_line_speed(ctx) == FREE_LINE_SPEED
    cfg = load_press_demo(PINK)
    code, rec, _chain, _rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], **FREE_AIR_RUNS[case])
    lid_off = case != "grip on air"
    assert code == (0 if lid_off else 1)
    want = {
        n: COMPLIANT_FREE_LINE_SPEED
        for n in (
            "reach:re-aim",
            "reach:slide-in",
            "depth:turn",
            "depth:rise",
            "depth:turn-back",
            "reach:slide-out",
            "grip:up",
            "lid:clear",
            "place:retreat",
            "place:down",
        )
    }  # the set-down's way down: free air
    # the stroke's own speed: here the faked way down's (test_reach_press._way_down), its press at TOUCH_SPEED
    want.update(
        {
            "depth:press": dp.TOUCH_SPEED,
            "grip:approach": cfg.grip_speed,
            "grip:down": dp.TOUCH_SPEED,
            "lid:lift": cfg.lift_speed,
            "place:touch": rlo.PLACE_TOUCH_SPEED,
            "reach:look": TRANSIT_SPEED,
            "lid:carry": TRANSIT_SPEED,
            "home": HOME_SPEED,
        }
    )
    assert [(n, s) for n, s in flown if s != want[n]] == []
    names = {n for n, _s in flown}
    assert ("grip:up" in names) is not lid_off and ("place:retreat" in names) is lid_off
    runs = [legs for legs, _gains in rec.runs]
    if case == "off the stroke":
        # the walk from where the rise ends (here where the arm stands: the fake hand is at the stroke's start
        # already), not the way down backwards
        assert runs[2] == ["reach:slide-out"], runs
    elif case == "no home after the set-down":
        assert runs[-2:] == [["place:open", "place:retreat"], ["home"]], runs
    elif case == "the retreat and home refused":
        assert runs[-3:] == [RELEASED, ["place:retreat"], ["home"]], runs
    fast = [params for speeds, params in timed if COMPLIANT_FREE_LINE_SPEED in speeds]
    assert fast and all(
        p.qd_ceiling == COMPLIANT_QD_CEILING and p.junction_cruise == COMPLIANT_JUNCTION_SPEED for p in fast
    )
    assert not ctx.client.compliant and rp.free_line_speed(ctx) == FREE_LINE_SPEED  # put back once the run is over


def test_a_lid_that_slipped_on_the_way_is_released_the_arm_goes_home_and_the_run_says_so(
    ctx, monkeypatch, tmp_path, capsys
):
    """The knuckle reads shut past the knob at the table: the lid left the fingers on the way. They open,
    the arm goes home, and the run ends STOP, the lid slipped out. Whatever the set-down's watch felt
    there was not the lid in the fingers meeting the table: the run does not say "SET-DOWN CONTACT: the
    lid met the table" of it (the flight view and the node read that line as the lid on the table; review
    2026-11-17), it says what was felt beside the slip."""
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], slipped=True)
    assert code == 1 and rec.events[-3:] == HOME
    (setdown,) = (r for r in rows if r["leg"] == "reach_setdown")
    assert setdown["kept"] is False and setdown["credible"] is True  # the watch felt something; the lid was gone
    out = capsys.readouterr().out
    assert "SET-DOWN CONTACT" not in out and "DONE" not in out
    assert (
        "the lid slipped on the way (the set-down felt a contact +12 mm from where the plan put the table: not the "
        "lid's, which had left the fingers)"
    ) in out
    assert out.rstrip().endswith(
        "STOP: the arm is home, but the lid slipped out of the fingers on the way to [%.2f, %.2f]"
        % tuple(ctx.lid_drop.xyz[:2])
    )


def test_a_box_the_press_moved_is_not_gripped(ctx, monkeypatch, tmp_path, capsys):
    moved = (BUTTON[0] + 0.02, BUTTON[1], BUTTON[2] + 0.009)
    code, rec, _chain, _rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (moved, 9.0)])
    assert code == 1 and rec.events[-2:] == ["aim:reach-after", "home"]
    assert "STOP: the press MOVED the box 20 mm" in capsys.readouterr().out


def test_a_button_the_wrist_no_longer_finds_after_the_press_is_gripped_for_not_pressed_again(
    ctx, monkeypatch, tmp_path, capsys
):
    """2026-11-12, flatbot, twice: the button popped, the finder (a flat disc at the lid's height) lost it once it
    stood up, and both runs stopped there with the box open. Where it was pressed is known and the fingers tell a
    knob from air: the run goes on to the grip, and never presses a knob it could not judge."""
    code, rec, _chain, _rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), None, None])
    assert rec.events.count("depth:press") == 1
    after = rec.events[rec.events.index("aim:reach-after") :]
    assert after[:2] == ["aim:reach-after", "aim:reach-after"] and "grip:close" in after, rec.events
    assert code == 0 and rec.events[-1] == "home"
    assert "the knob is not judged after press 1" in capsys.readouterr().out


def test_an_open_box_has_its_lid_taken_off_unpressed(ctx, monkeypatch, tmp_path, capsys):
    """The aim reads the knob up before anything is pressed: pressing it
    would shut the box. The full run takes the lid off from the look pose,
    no stroke flown."""
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(UP, 9.0)])
    assert code == 0
    assert rec.events == ["reach:look", "aim:reach-aim", *GRIP, *CLOSE, *CARRY, *HOME]
    assert [(r["button_above_lid_mm"], r["opened"]) for r in rows if r["leg"] == "open_box"] == [(9.0, True)]
    (grip,) = (r for r in rows if r["leg"] == "reach_grip")
    assert grip["lid_z"] == pytest.approx(BUTTON[2])  # the scene's top, above the aim's knob less its height
    assert "ALREADY OPEN" in capsys.readouterr().out


def test_an_open_box_s_grip_on_air_is_tried_once_more_at_the_knob_aimed_at_again(ctx, monkeypatch, tmp_path):
    """The open box's lid, taken off unpressed (reach_press._reach_press
    calls take_lid_off itself), gets the same one retry after a close on
    air: the re-aim from the look pose, the second grip there."""
    asked = _grip_ways(monkeypatch)
    again = (UP[0] + 0.011, UP[1], UP[2])
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(UP, 9.0), (again, 9.0)], knuckle=[AIR, KNOB_ON])
    assert code == 0
    assert rec.events == [
        "reach:look",
        "aim:reach-aim",
        *GRIP,
        *CLOSE,
        *CARRY,
        *BACK_OUT,
        "aim:reach-regrip",
        *GRIP,
        *CLOSE,
        *CARRY,
        *HOME,
    ]
    assert [b[:2] for b, _lid, _tilt in asked] == [pytest.approx(UP[:2]), pytest.approx(again[:2])]


def test_with_press_only_an_open_box_still_stops_the_run(ctx, monkeypatch, tmp_path):
    code, rec, _chain, _rows = _full_flow(ctx, monkeypatch, tmp_path, [(UP, 9.0)], press_only=True)
    assert code == 1 and rec.events == ["reach:look", "aim:reach-aim", "home"]


def test_a_knob_read_between_flush_and_up_then_flush_is_pressed(ctx, monkeypatch, tmp_path):
    """The look reads the knob 4.4 mm up (the highest flush button on
    record), the second aim from the same pose 1.0: shut, pressed — on the
    second aim's button."""
    aims = [(BUTTON, 4.4), (BUTTON, 1.0), (UP, 9.0)]
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, aims)
    assert code == 0
    assert rec.events == [
        "reach:look",
        "aim:reach-aim",
        "aim:reach-recheck",
        *WAY_IN,
        "hold",
        *BACK,
        "aim:reach-after",
        *GRIP,
        *CLOSE,
        *CARRY,
        *HOME,
    ]
    (call,) = (r for r in rows if r["leg"] == "knob_call")
    assert (call["reads_mm"], call["call"], call["pressed"]) == ([4.4, 1.0], "shut", True)
    assert not [r for r in rows if r["leg"] == "open_box"]


@pytest.mark.parametrize(
    "aims, call",
    [
        ([(BUTTON, 4.4), (UP, 5.0)], "band"),  # in between twice: taken for open
        ([(BUTTON, 5.6), None], "band"),  # the second aim finds no button: the first reading stands
        ([(BUTTON, 5.0), (UP, 8.0)], "open"),
    ],
    ids=["in between twice", "no second reading", "then up"],
)
def test_a_knob_not_read_flush_on_a_second_look_is_not_pressed_and_its_lid_comes_off(
    ctx, monkeypatch, tmp_path, capsys, aims, call
):
    """The wrist study of 2026-11-05 (I-Y2, M17, H5): one tilted look at
    6.0 mm read popped knobs 5.0-9.7 mm, and an open box read shut was
    pressed shut 16 % of the time. Between 3.0 and 6.0 mm the wrist aims
    once more; still not flush, the box is not pressed: the full run takes
    its lid off as an open box's, from the look pose, no stroke flown."""
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, aims)
    assert code == 0
    assert rec.events == ["reach:look", "aim:reach-aim", "aim:reach-recheck", *GRIP, *CLOSE, *CARRY, *HOME]
    assert "depth:press" not in rec.events
    deciding = aims[1] or aims[0]
    (note,) = (r for r in rows if r["leg"] == "knob_call")
    assert (note["looks"], note["call"], note["pressed"]) == (2, call, False)
    assert note["reads_mm"] == [a[1] for a in aims if a is not None]
    assert [(r["button_above_lid_mm"], r["opened"], r["call"]) for r in rows if r["leg"] == "open_box"] == [
        (deciding[1], True, call)
    ]
    out = capsys.readouterr().out
    assert ("NOT CALLED SHUT" in out) == (call == "band") and ("ALREADY OPEN" in out) == (call == "open")


def test_with_press_only_a_knob_read_between_flush_and_up_twice_stops_the_run(ctx, monkeypatch, tmp_path, capsys):
    code, rec, _chain, _rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 4.4), (BUTTON, 3.5)], press_only=True)
    assert code == 1 and rec.events == ["reach:look", "aim:reach-aim", "aim:reach-recheck", "home"]
    assert "STOP: the knob reads 4.4 then 3.5 mm above its lid — neither flush (below 3) nor up (6" in (
        capsys.readouterr().out
    )


def test_a_knob_whose_height_no_look_reads_before_the_press_is_not_pressed(ctx, monkeypatch, tmp_path, capsys):
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, None), (BUTTON, None)])
    assert code == 1 and rec.events == ["reach:look", "aim:reach-aim", "aim:reach-recheck", "home"]
    (note,) = (r for r in rows if r["leg"] == "knob_call")
    assert (note["reads_mm"], note["call"], note["pressed"]) == ([None, None], "unread", False)
    assert "STOP: the wrist read no height for the knob on two looks" in capsys.readouterr().out


# ---- home, planned from the look pose while the way back flies -----------------

MOVED = (BUTTON[0] + 0.02, BUTTON[1], BUTTON[2] + 0.009)  # the aim after a press that moved the box 20 mm
NOT_UP_TWICE = [(BUTTON, 0.4), (BUTTON, 2.0), (BUTTON, 1.5), (BUTTON, 2.8), (BUTTON, 2.5)]  # flush: under 3.0


def _home_plans(monkeypatch, refuse=()):
    """reach_press._home_legs with each call kept as (where it plans home
    from — None: where the arm stands —, the legs it returned, None where
    refused); the calls numbered in `refuse` (the first is 1) refused, as
    the planner can."""
    real, plans = rp._home_legs, []

    def home_legs(ctx_, cfg_, start=None):
        at = None if start is None else [float(v) for v in start]
        if len(plans) + 1 in refuse:
            plans.append((at, None))
            raise RuntimeError("planning failed for home: MotionGenStatus.TRAJOPT_FAIL")
        plans.append((at, real(ctx_, cfg_, start=start)))
        return plans[-1][1]

    monkeypatch.setattr(rp, "_home_legs", home_legs)
    return plans


@pytest.mark.parametrize(
    "aims, presses, grip_way",
    [
        ([(BUTTON, 0.4), (MOVED, 9.0)], 1, True),
        (NOT_UP_TWICE, 2, True),
        ([(BUTTON, 0.4), (UP, 9.0)], 1, False),
    ],
    # (a button not found after the press is no stop since 2026-11-12: the run goes on to the grip)
    ids=["the box moved", "not up after the second press", "no way down to the knob"],
)
def test_a_stop_at_the_look_pose_flies_the_home_planned_while_the_way_back_flew(
    ctx, monkeypatch, tmp_path, aims, presses, grip_way
):
    """2026-10-28 16:15:35 and 16:16:09: the knob looked at, the run
    stopped, and the arm stood 0.21 and 0.23 s at the look pose while home
    was planned. Home is planned from the look pose while each way back
    flies — a second press's from where its own ends — and the stop flies
    the last: no planner call at rest. So does a knob up that no way down
    to it reaches: the arm has not left the look pose."""
    if not grip_way:
        monkeypatch.setattr(rlo, "solve_grip_way", lambda *a, **k: (None, "no straight-down hand reaches the knob"))
    plans = _home_plans(monkeypatch)
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, aims)
    assert code == 1 and rec.events[-1] == "home"
    backs = [i + 1 for i, e in enumerate(rec.events) if e == "reach:slide-out"]  # where each way back's legs end
    assert len(backs) == presses
    # each home planned while a way back flew (the record then ended with its legs), none since
    assert [at for name, at in rec.plans if name == "home"] == backs
    assert [at is not None for at, _legs in plans] == [True] * presses
    assert plans[-1][0] == pytest.approx(rec.legs["reach:slide-out"].goal_joints)  # from where the last ends
    assert rec.legs["home"] is plans[-1][1][-1]
    (ahead,) = (r for r in rows if r["leg"] == "home_ahead")
    assert ahead["flown"] is True and ahead["off_rad"] == pytest.approx(0.0)


def test_the_set_down_is_cut_where_the_lid_meets_the_table(ctx, monkeypatch, tmp_path, capsys):
    """Owner 2026-11-02: the arm "presses the lid down onto the table
    waaaaaaay too hard ... the lid should just be let go of when it's ~1 cm
    above the table". The lift-to-set-down execution's lead carries a
    SetdownContact that cuts the touch where the lid meets the table
    (SETDOWN_FROM_CONTACT), and the reach_setdown row says what it saw. Here
    the real watch, and a client that reports no flight: nothing watched, so
    nothing says the lid met the table, and (owner 2026-11-17) the fingers
    do not open there."""
    from box_opening.tasks.press_contact import PressContact

    assert rlo.SETDOWN_FROM_CONTACT is True
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], setdown=None)
    lead = rec.legs["lid:lift"]  # the one motion's lead: lift, clear, carry, down, touch
    assert isinstance(lead.stop_when, rlo.SetdownContact) and isinstance(lead.stop_when, PressContact)
    assert lead.stop_when.cut is True and lead.stop_when.final
    (sd,) = (r for r in rows if r["leg"] == "reach_setdown")
    assert sd["contact"]["cut_on"] is True and sd["contact"]["watched"] is False and sd["contact"]["met"] is False
    assert sd["credible"] is False and sd["contact"]["verdict"] == "never"
    assert code == 1 and "place:open" not in rec.events
    out = capsys.readouterr().out
    assert "the set-down was not watched" in out
    # ... nor is the lid taken on to another spot (owner 2026-11-19: "it should keep trying spots"): a
    # set-down that was not watched says nothing of its spot, and the next would fly as blind. The box
    assert (
        rec.events.count("place:touch") == 1 and "no other spot is tried: a set-down that could not be watched" in out
    )
    assert out.rstrip().endswith("the lid is back on the box, and the arm is home")


@pytest.mark.parametrize("felt", ["never", "held"])
def test_a_set_down_that_felt_no_table_keeps_the_lid_and_takes_it_on_to_another_spot(
    ctx, monkeypatch, tmp_path, capsys, felt
):
    """Owner 2026-11-17: "the first run, the arm dropped the lid off the table": the fingers open only
    where the watch felt the lid meet the table (SetdownContact.credible). And 2026-11-19, asked whether
    a set-down that feels no table should put the lid back on the box or try another spot first: "it
    should keep trying spots, assume tere will always be a spot for the lid". The touch ended and
    nothing was felt, or something held the lid up well above the table: the lid is NOT let go and NOT
    taken back to the box. Up with it the way the retreat goes, the fingers shut; that spot and the
    lid's own footprint round it set aside; the next spot chosen by the same rules; carried there and
    set down the same watched way — here that one feels the table, and the run ends as any good run:
    exit 0, the lid on the table at the second spot, said."""
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY, scene=_table(), setdown=[felt, "met"]
    )
    assert code == 0
    after = rec.events[rec.events.index("place:touch") + 1 :]
    # never back over the box, never onto it. Where something the map did not show held the lid up, the wrist
    # looks at the table from over that spot before the next is chosen; where nothing was felt, no look
    assert after == ["lid:back-up", *(["spot-look"] if felt == "held" else []), *CARRY, *HOME]
    assert not [r for r in rows if r["leg"] == "lid_put_back"]
    looks = [r for r in rows if r["leg"] == "lid_look_again"]
    assert [r["why"] for r in looks] == (
        ["something the map did not show held the lid up at spot 1"] if felt == "held" else []
    )
    # the way back up is the retreat's own line from where the arm stands, the fingers shut: no hold for them
    assert rec.legs["lid:back-up"].start_hold_s == 0.0 and rec.legs["lid:back-up"].gains == dp.COMPLIANT_GAINS
    (first, how1), (second, how2) = rec.setdowns
    assert (how1, how2) == (felt, "met")
    # off the footprint that was left: the lid's own where nothing was felt; the lid's and the hand's (its
    # farthest corner 99 mm from the tool, whichever way it turns next) where something held the lid up
    apart = max(ctx.model.lid_dims[:2]) if felt == "never" else 2 * rlo.HAND_REACH_M
    assert rlo.HAND_REACH_M == pytest.approx(0.0986, abs=1e-4)
    assert math.hypot(second.xyz[0] - first.xyz[0], second.xyz[1] - first.xyz[1]) >= apart - 1e-9
    assert ctx.lid_at is ctx.lid_drop is second  # where the lid lies: the spot that took it
    sd = [(r["spot"], r["credible"], r["contact"]["verdict"]) for r in rows if r["leg"] == "reach_setdown"]
    assert sd == [(1, False, felt), (2, True, None)]
    (left,) = (r for r in rows if r["leg"] == "lid_spot_left")
    assert (left["spot"], left["verdict"], left["next"], left["why_the_box"]) == (1, felt, "another spot", None)
    assert left["xyz"] == pytest.approx(list(first.xyz), abs=1e-4) and left["time_left_s"] is None  # nothing times it
    spots = [r for r in rows if r["leg"] == "lid_spot"]
    assert [r["spot"] for r in spots] == [1, 2] and spots[1]["xyz"] == pytest.approx(list(second.xyz), abs=1e-4)
    assert [(s["spot"], s["verdict"]) for s in spots[1]["left"]] == [(1, felt)] and spots[1]["searched"] == [
        "the tiers"
    ]
    assert spots[1]["table"]["aside"] >= 1  # the spots within the lid's footprint of the first were passed over
    out = capsys.readouterr().out
    assert "SET-DOWN: no table felt at spot 1, [%.2f, %.2f] — " % (first.xyz[0], first.xyz[1]) in out
    assert "the lid is not let go: up with it and on to another spot" in out and "back onto the box" not in out
    assert "; spot 2 of this run, after spot 1 at [%.2f, %.2f]: " % (first.xyz[0], first.xyz[1]) in out
    assert "SPOT 2, ONE MOTION: lid:carry, place:down, place:touch — to [" in out
    assert out.count("SET-DOWN CONTACT") == 1 and out.count("RELEASE as the retreat starts") == 1
    assert "DONE — box open, lid on the table at [%.2f, %.2f]" % (second.xyz[0], second.xyz[1]) in out
    assert "spot 2 of this run (no table was felt at the 1 before it)" in out and out.rstrip().endswith("arm home")


def test_a_set_down_that_could_not_be_watched_is_not_tried_elsewhere_and_the_lid_goes_back_on_the_box(
    ctx, monkeypatch, tmp_path, capsys
):
    """The watch failed: that says nothing of the spot, and a set-down at the next would fly as blind —
    its touch to its planned end, 8 mm under the plan, with nothing watching for the table. No other
    spot is tried: the lid is NOT let go; back up with it the way the retreat goes, back over the box,
    down onto it, the fingers open there, the hand off the knob, home — the one place known to take
    the lid. Exit 1, and the run says the lid is back on the box, and why no other spot was tried."""
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], setdown="failed")
    assert code == 1 and "place:open" not in rec.events and "place:retreat" not in rec.events
    after = rec.events[rec.events.index("place:touch") + 1 :]
    assert after == ["lid:back-up", "lid:back-over", "lid:put-back", "lid:release", "lid:rise-off", "home"]
    # the way back up is the retreat's own line from where the arm stands, the fingers shut: no hold for them
    assert rec.legs["lid:back-up"].start_hold_s == 0.0 and rec.legs["lid:back-up"].gains == dp.COMPLIANT_GAINS
    (sd,) = (r for r in rows if r["leg"] == "reach_setdown")
    assert sd["credible"] is False and sd["kept"] is True and sd["contact"]["verdict"] == "failed"
    (back,) = (r for r in rows if r["leg"] == "lid_put_back")
    assert back["ok"] is True and back["why"].startswith("the set-down felt no table (")
    assert not [r for r in rows if r["leg"] == "lid_spot_left"]
    assert ctx.lid_lifted is None and ctx.lid_at is None  # out of the fingers, on the box: no lid lies at the spot
    out = capsys.readouterr().out
    assert (
        "SET-DOWN: no table felt — " in out and "the lid is not let go: back up with it, and back onto the box" in out
    )
    assert "(no other spot is tried: a set-down that could not be watched says nothing of this spot" in out
    assert "SET-DOWN CONTACT" not in out and "DONE" not in out and "RELEASE as the retreat starts" not in out
    assert out.rstrip().endswith("the lid is back on the box, and the arm is home")


class _WalksBack(_Chain):
    """A chain that refuses the next `refuse_next` straight lines (the hand does not hold their attitude
    from where it stands) and, once `walk_ok`, walks to a pose turning as it goes."""

    refuse_next, walk_ok = 0, False

    def __init__(self):
        super().__init__()
        self.walked = []

    def straight_line(self, q0, xyz, quat, max_step_rad=None):
        if self.refuse_next:
            self.refuse_next -= 1
            return None, "a joint would jump 0.4 rad over one step"
        return super().straight_line(q0, xyz, quat, max_step_rad)

    def walk(self, q0, xyz, quat, step_m=None, step_rad=None, max_step_rad=None):
        if not self.walk_ok:
            return super().walk(q0, xyz, quat, step_m, step_rad, max_step_rad)
        self.walked.append((list(quat), step_m, step_rad))
        return [list(q0), [v + 0.01 for v in q0]], None


def test_the_way_back_over_the_box_turns_the_hand_back_where_no_line_holds_the_grips_attitude(
    ctx, monkeypatch, tmp_path
):
    """The set-down's hand stands turned to its spot, or leaning: from there no straight line holds the
    attitude the knob was gripped at (put_lid_back's way back over the box was only ever flown from the
    clear's end, which has it). Refused so, the way back over the box is the arm's own walk, turning back
    as it goes, in the carry walk's steps; the lid still goes down onto the box and the arm home."""
    chain = _WalksBack()
    real = rlo.put_lid_back

    def back(ctx_, cfg_, runner_, why):
        chain.refuse_next, chain.walk_ok = 1, True
        return real(ctx_, cfg_, runner_, why)

    monkeypatch.setattr(rlo, "put_lid_back", back)
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], chain=chain, setdown="failed"
    )
    assert code == 1 and "place:open" not in rec.events
    after = rec.events[rec.events.index("place:touch") + 1 :]
    assert after == ["lid:back-up", "lid:back-over", "lid:put-back", "lid:release", "lid:rise-off", "home"]
    ((quat, step_m, step_rad),) = chain.walked
    assert quat == pytest.approx(list(rec.legs["lid:lift"].target[2]))  # back to the attitude the knob was held at
    assert (step_m, step_rad) == (rlo.CARRY_WALK_STEP_M, rlo.CARRY_WALK_STEP_RAD)
    (put,) = (r for r in rows if r["leg"] == "lid_put_back")
    assert put["ok"] is True


def test_a_lid_that_met_the_table_is_let_go_there_and_only_then_the_run_says_it_is_on_the_table(
    ctx, monkeypatch, tmp_path, capsys
):
    """The watch felt the lid meet the table where the plan has it: the fingers open as the retreat
    starts, and the run's DONE says the lid is on the table, let go where the arm felt it."""
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)])
    assert code == 0 and rec.events[-3:] == HOME and "lid:back-up" not in rec.events
    (sd,) = (r for r in rows if r["leg"] == "reach_setdown")
    assert sd["credible"] is True
    out = capsys.readouterr().out
    assert "SET-DOWN CONTACT: the lid met the table +12 mm from where the plan put it" in out
    assert "DONE — box open, lid on the table at [" in out and "let go where the arm felt it meet the table" in out


def test_a_set_down_that_felt_no_table_and_cannot_fly_back_up_holds_the_lid(ctx, monkeypatch, tmp_path, capsys):
    """No line back up solves from where the arm stands: nothing is let go, the arm holds the lid
    where it is, and says so (a person clears it)."""
    monkeypatch.setattr(rlo, "retreat_from_live", lambda ctx_, carry, world: (None, "IK did not converge"))
    for felt in ("never", "failed"):  # on to another spot, or back onto the box: both begin with the way up
        code, rec, _chain, _rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], setdown=felt)
        assert code == 1 and rec.events[-1] == "place:touch" and "place:open" not in rec.events
        assert "no way back up solves from where the arm stands (IK did not converge) — the arm holds the lid" in (
            capsys.readouterr().out
        )


def test_the_set_down_watch_starts_above_where_the_plan_has_the_lid_meet_the_table(monkeypatch):
    """watch_setdown: the plan has the lid meet the table SETDOWN_PAST_M
    over the touch's end (carry.over_m: the touch goes on that far under it,
    for the arm to feel a table that stands where the plan has it); the
    watch begins SETDOWN_WATCH_M above the plan's table — where the slow
    stretch begins — on the execution's last stretch below there (final),
    from its first motion leg (not a gripper leg before it), cuts where the
    lid meets (past 0), and stops the descent where it fails: it is told
    the joints the way down ends at, where the slow stretch begins."""
    from box_opening.runtime import approach
    from box_opening.runtime.legs import Kind
    from box_opening.tasks.press_contact import pads_z

    class Chain:
        def fk(self, q):
            return np.eye(3), np.array([0.0, 0.0, float(q[0])])

    monkeypatch.setattr(approach, "_chain", lambda: Chain())
    close = SimpleNamespace(kind=Kind.GRIPPER, stop_when=None)
    lift = SimpleNamespace(kind=Kind.MOTION, stop_when=None)
    touch = SimpleNamespace(kind=Kind.MOTION, stop_when=None, goal_joints=[0.25] + [0.0] * 6)
    down = SimpleNamespace(kind=Kind.MOTION, stop_when=None, goal_joints=[0.338] + [0.0] * 6)
    carry = SimpleNamespace(legs=[SimpleNamespace(), down, touch], over_m=rlo.SETDOWN_PAST_M, contact=None)
    ctx = SimpleNamespace(client=SimpleNamespace())
    got = rlo.watch_setdown(ctx, [close, lift, down, touch], carry)
    end = pads_z(Chain(), touch.goal_joints)
    assert lift.stop_when is got is carry.contact and close.stop_when is None
    assert isinstance(got, rlo.SetdownContact) and got.abort_on_failure is True
    assert got.end_z == pytest.approx(end) and got.top_z == pytest.approx(end + rlo.SETDOWN_PAST_M)
    assert got.arm_z == pytest.approx(end + rlo.SETDOWN_PAST_M + rlo.SETDOWN_WATCH_M) and got.past_m == 0.0
    assert got.cut is True and got.final is True and rlo.SETDOWN_WATCH_M == rlo.SETDOWN_SLOW_M
    assert (got.settle_s, got.credible_m, got.held_m) == (0.6, 0.035, 0.015)
    assert got.slow_q == down.goal_joints and down.stop_when is None
    assert not got.credible() and got.refusal()[0] == "never"  # nothing fed: nothing says the lid met the table


def test_every_press_is_cut_at_its_contact_and_watched_all_the_way_down(ctx, monkeypatch, tmp_path):
    """Contact detection rides each press's way in (its lead's stop_when). Review round 2: every
    press's stroke ends as far past where the pads met the box as it was planned past the top, a
    press once more (the knob not up after the first) too; each watches its whole descent
    (final, from where its stroke begins), refuses a contact off the band, and stops on a watch
    that fails."""
    from box_opening.tasks import press_contact as pc

    real, watched = dp.watch_press, []

    def watch(ctx_, legs, stroke, top_z, cut=True):
        got = real(ctx_, legs, stroke, top_z, cut=cut)
        watched.append((legs[0].name, legs[0].stop_when is got, got))
        return got

    monkeypatch.setattr(dp, "watch_press", watch)
    code, rec, _chain, _rows = _full_flow(ctx, monkeypatch, tmp_path, NOT_UP_TWICE)
    assert code == 1
    assert [(lead, on, got.cut) for lead, on, got in watched] == [("reach:re-aim", True, True)] * 2
    for _lead, _on, got in watched:
        assert isinstance(got, pc.PressContact) and got.final and got.abort_on_failure
        # the band, both ends up by what flatbot's fingers reach past the pads it measures (constants, 2026-11-17)
        assert got.early_m == pytest.approx(pc.CONTACT_EARLY_M + FINGERS_PAST_MODEL_M)
        assert got.late_m == pytest.approx(pc.CONTACT_LATE_M - FINGERS_PAST_MODEL_M)
        assert got.slow_z == pytest.approx(got.top_z + dp.TOUCH_SLOW_M) and got.arm_z >= got.slow_z


def test_the_drivers_force_is_logged_at_the_press_and_the_set_down(ctx, monkeypatch, tmp_path):
    """Owner 2026-11-04: log the driver's estimate of the force on the tool (/ee_state) beside the
    run's own (the springs' pull at the hold; the set-down's), never instead of it."""
    asked = []

    def ee_force(since=None, span_s=None):
        asked.append((since, span_s))
        return {"mean_n": [0.4, -0.3, 18.2], "peak_up_n": 21.0, "peak_n": 21.1, "samples": 15}

    monkeypatch.setattr(ctx.client, "ee_force", ee_force, raising=False)
    code, _rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)])
    assert code == 0
    (sd,) = (r for r in rows if r["leg"] == "reach_setdown")  # the hold's: test_depth_press
    assert sd["driver_force"]["at_release"]["mean_n"] == [0.4, -0.3, 18.2]
    assert (None, 0.2) in asked


@pytest.mark.parametrize("verdict", ["early", "late", "never", "failed"])
def test_a_press_refused_by_its_contact_goes_back_up_and_home(ctx, monkeypatch, tmp_path, capsys, verdict):
    """Review round 2: a contact well off the cameras' top, none at all, or a watch that failed:
    the press is not pressed on, the springs let go where the stroke stopped, the arm goes back
    to the look pose and home, and the run STOPs with why (the node's contact_* codes)."""
    real = dp.watch_press
    why = {
        "early": "the press met something 60 mm above the button's top as the cameras saw it (more than 45): "
        "something else is in the way — the press was stopped there; nothing was pressed",
        "late": "the pads met nothing down to 28 mm below the button's top as the cameras saw it (more than 25): "
        "the box is not where it was seen — the press was stopped there",
        "never": "the press stroke ended with the pads meeting nothing: the box is not where it was seen",
        "failed": "the press's contact watch failed (ValueError: x) — the press was stopped there",
    }[verdict]

    def watch(ctx_, legs, stroke, top_z, cut=True):
        got = real(ctx_, legs, stroke, top_z, cut=cut)
        got.refusal = lambda: (verdict, why)
        return got

    monkeypatch.setattr(dp, "watch_press", watch)
    let_go = []
    monkeypatch.setattr(rp, "let_go", lambda ctx_: let_go.append(1))
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)])
    assert code == 1 and rec.events[-1] == "home"
    assert "reach:slide-out" in rec.events and not any(e.startswith("grip:") for e in rec.events)
    (refused,) = (r for r in rows if r["leg"] == "press_refused")
    assert refused["verdict"] == verdict and refused["why"] == why
    assert not any(r["leg"] == "depth_compliant_press" for r in rows)  # no hold: nothing pressed on
    assert let_go and ("STOP: " + why) in capsys.readouterr().out


@pytest.mark.parametrize("push, flown", [(0.03, True), (0.05, False)])
def test_the_home_planned_on_the_way_back_flies_only_from_where_it_starts(
    ctx, monkeypatch, tmp_path, capsys, push, flown
):
    """The arm pushed `push` rad on joint_2 off the look pose while it
    looked: within DRIFT_REPLAN_RAD the home planned while the way back
    flew is flown, as the runner flies any planned leg from there; beyond,
    home is planned from where the arm stands — a plan from somewhere else
    is never flown."""
    real = rlo.knob_look

    def pushed(node, ctx_, *rest):
        ctx_.client.live[1] += push
        return real(node, ctx_, *rest)

    monkeypatch.setattr(rlo, "knob_look", pushed)
    plans = _home_plans(monkeypatch)
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (MOVED, 9.0)])
    assert code == 1 and rec.events[-2:] == ["aim:reach-after", "home"]
    (ahead,) = (r for r in rows if r["leg"] == "home_ahead")
    assert ahead["flown"] is flown and ahead["off_rad"] == pytest.approx(push)
    look = plans[0][0]  # where the way back ended
    here = [v + (push if j == 1 else 0.0) for j, v in enumerate(look)]
    if flown:
        assert len(plans) == 1 and rec.legs["home"] is plans[0][1][-1]
    else:
        assert [at for at, _legs in plans] == [look, None] and rec.legs["home"] is plans[1][1][-1]
        assert rec.legs["home"].traj.points[0].positions == pytest.approx(here)
        assert "home is planned again from here" in capsys.readouterr().out


@pytest.mark.parametrize(
    "aims, knuckle, ends, refuse, at_rest",
    [
        ([(BUTTON, 0.4), (UP, 9.0)], AIR, None, (), False),
        ([(BUTTON, 0.4), (UP, 9.0)], AIR, None, (2, 3), True),  # two back-outs (the second grip, 1 cm lower)
        ([(BUTTON, 0.4), (UP, 9.0)], KNOB_ON, {4: trp.REJECTED}, (), True),
        (NOT_UP_TWICE, KNOB_ON, None, (2,), True),
        (NOT_UP_TWICE[:3], KNOB_ON, {4: trp.REJECTED}, (), True),
    ],
    ids=[
        "backed out",
        "backed out, its home refused",
        "the grip's way down did not start",
        "pressed once more, its home refused",
        "the way in once more did not start",
    ],
)
def test_the_home_planned_on_the_way_back_is_dropped_once_the_arm_moves_on(
    ctx, monkeypatch, tmp_path, aims, knuckle, ends, refuse, at_rest
):
    """The grip's way down to the knob, or a press once more, sends the arm
    off the look pose; it comes back within DRIFT_REPLAN_RAD of where the
    home planned on the way back starts (the fake re-aim is 0.01 rad; a
    way down, or in, that did not start, 0). That home is never flown
    after: the stop flies the one planned while the arm came back (the
    back-out's), else one planned where the arm stands, the run log saying
    none was in hand."""
    from box_opening.constants import DRIFT_REPLAN_RAD

    plans = _home_plans(monkeypatch, refuse)
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, aims, knuckle=knuckle, ends=ends, one_motion=False)
    assert code == 1 and rec.events[-1] == "home"
    first = plans[0]  # the way back's
    home = rec.legs["home"]
    assert max(abs(a - b) for a, b in zip(home.traj.points[0].positions, first[0], strict=False)) <= DRIFT_REPLAN_RAD
    assert home is not first[1][-1] and home is plans[-1][1][-1]
    assert (plans[-1][0] is None) is at_rest  # at rest: from where the arm stands
    (ahead,) = (r for r in rows if r["leg"] == "home_ahead")
    assert ahead["flown"] is not at_rest and (ahead["off_rad"] is None) is at_rest  # none in hand: off_rad None


# ---- the lid's spot, chosen with the lid lifted: never a refusal ---------------

BOX_XY = BUTTON[:2]
SMALL = (BOX_XY[0] - 0.22, BOX_XY[0] + 0.22, BOX_XY[1] - 0.22, BOX_XY[1] + 0.22)  # no spot beside the box fits
ROOMY = (BOX_XY[0] - 0.6, BOX_XY[0] + 0.6, BOX_XY[1] - 0.6, BOX_XY[1] + 0.6)
TABLE_Z = BUTTON[2] - 0.121


def _table(xy=BOX_XY, r=0.6, step=0.01, z=TABLE_Z):
    """A flat table's points round `xy`, and the box standing on it."""
    g = np.arange(-r, r + 1e-9, step)
    X, Y = np.meshgrid(g + xy[0], g + xy[1], indexing="ij")
    flat = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], -1)
    box = flat[np.hypot(flat[:, 0] - xy[0], flat[:, 1] - xy[1]) < 0.0525].copy()
    flat = flat[np.hypot(flat[:, 0] - xy[0], flat[:, 1] - xy[1]) >= 0.0525]
    box[:, 2] = z + 0.121
    return np.concatenate([flat, box])


def _wall(pts, inner=0.15, outer=0.45, tall=0.30):
    """`pts` with everything from `inner` to `outer` round the box standing
    `tall`: no room for the lid anywhere the hand reaches."""
    pts = pts.copy()
    d = np.hypot(pts[:, 0] - BOX_XY[0], pts[:, 1] - BOX_XY[1])
    pts[(d > inner) & (d < outer), 2] = TABLE_Z + tall
    return pts


def test_a_small_table_no_longer_stops_the_run_before_anything_moves(ctx, monkeypatch, tmp_path, capsys):
    """2026-10-28: five full runs stopped "nowhere to set the lid down"
    before any motion. Nothing is resolved at the start now: the lid comes
    off, and with no spot beside the box in the set-down zone and no camera
    data, it goes to the nearest table point the hand reaches, off the box —
    placed, exit 0, and said once, as information."""
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=SMALL)
    assert code == 0 and rec.events[:2] == ["reach:look", "aim:reach-aim"] and rec.events[-3:] == HOME
    (spot,) = (r for r in rows if r["leg"] == "lid_spot")
    assert spot["tier"] == "nearest"
    out = capsys.readouterr().out
    assert "nowhere to set the lid down" not in out and "STOP" not in out
    assert "the nearest table point the hand reaches, off the box" in out
    x0, x1, y0, y1 = SMALL
    assert x0 < ctx.lid_drop.xyz[0] < x1 and y0 < ctx.lid_drop.xyz[1] < y1  # on the table
    assert math.hypot(ctx.lid_drop.xyz[0] - BOX_XY[0], ctx.lid_drop.xyz[1] - BOX_XY[1]) > 0.105  # off the box


def test_the_nearest_spot_keeps_the_lid_off_the_box_s_holder(ctx, monkeypatch, tmp_path, capsys):
    """Sweep lid_spot-g-3: 2026-10-30 13:35 set the lid down 0.120 m from
    the box, on the blue holder's flange, and the holder tipped. The
    'nearest' rings now start with the lid's rim NEAREST_CLEAR_M off the
    box, centred on the box where the press held its button."""
    from box_opening.tasks.press_demo import container_half

    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=SMALL)
    assert code == 0
    (spot,) = (r for r in rows if r["leg"] == "lid_spot")
    lid_r = 0.5 * max(ctx.model.lid_dims[:2])
    first = container_half(ctx.model) + lid_r + rlo.NEAREST_CLEAR_M
    assert spot["tier"] == "nearest" and first == pytest.approx(0.155, abs=0.005)
    cx, cy = rlo.nearest_centre(ctx)
    assert math.hypot(ctx.lid_drop.xyz[0] - cx, ctx.lid_drop.xyz[1] - cy) >= first - 1e-6
    assert ctx.held_xy == pytest.approx(BUTTON[:2])  # the fake hold held the pads on the button


def test_no_spot_of_any_tier_lands_the_lid_inside_the_no_land_radius(ctx, monkeypatch):
    """Owner 2026-11-04: "in run 2 the lid accidentally landed on the rim. we need to make a no
    land radius around the box" — 12:40's lid set down 114 mm from where the press held the
    button. Whatever tier offers a spot (here the clear tier's first, 0.12 m off, as 12:40's
    ring was), one whose disc comes within NO_LAND_CLEAR_M of the box's footprint round where the
    press held its button, or where ctx.cpose has the box, is passed over and never counted as
    a try; the first spot outside both is taken, and the run record says how far outside."""
    from box_opening.models.container import from_container
    from box_opening.runtime import approach
    from box_opening.tasks.press_demo import container_half

    r = rlo.no_land_radius(ctx.model)
    assert r == pytest.approx(container_half(ctx.model) + 0.5 * max(ctx.model.lid_dims[:2]) + rlo.NO_LAND_CLEAR_M)
    assert rlo.no_land_radius(ContainerModel.load(PINK)) == pytest.approx(0.155)  # the pink canister: 50 mm rim to rim
    bx, by = float(ctx.cpose.xyz[0]), float(ctx.cpose.xyz[1])
    button = from_container(ctx.cpose, ctx.model.button_offset)
    ctx.held_xy = (float(button[0]), float(button[1]) - 0.040)  # the aim 40 mm off where the press held it
    cx, cy = rlo.box_centres(ctx)[1]
    assert (cx, cy) == pytest.approx((bx, by - 0.040))
    rim = ContainerPose(xyz=(cx, cy - 0.114, -0.07), yaw=0.0)  # 12:40's: 114 mm from the held button
    aimed = ContainerPose(xyz=(bx, by - 0.180, -0.07), yaw=0.0)  # 180 mm from the aim, 140 from the held one
    clear = ContainerPose(xyz=(bx, by + 0.200, -0.07), yaw=0.0)  # outside both
    assert rlo.box_clearance(ctx, rim) < 0 and rlo.box_clearance(ctx, aimed) < 0 < rlo.box_clearance(ctx, clear)
    from box_opening.perception.clear_table import Spot

    def three(m, cpose, worlds, tmap_, table_z, judged_out=None, hand_at=None, cache=None):
        judged_out.extend(Spot(d.xyz[:2], 0.0, 0.02, "nothing", 0.0) for d in (rim, aimed, clear))
        return [], None

    monkeypatch.setattr(rlo, "lid_spots", three)
    monkeypatch.setattr(rlo, "TIER_TRIES", dict(rlo.TIER_TRIES, clear=1))  # one try: the passed-over are not tries
    checked = []
    tmap = _standing_ring((bx, by), 0.5, 0.6)  # seen, nothing standing near the box

    def reach(ctx_, d, q, seeds=()):
        checked.append(d)
        return None

    drop, tier, _says = next(rlo.spot_candidates(ctx, tmap, -0.07, [0.0] * 7, reach=reach))
    assert (drop, tier) == (clear, "clear") and checked == [clear]
    note = rlo.no_land_note(ctx, drop)
    assert note["radius_mm"] == pytest.approx(1000 * r, abs=0.1) and note["clear_mm"] > 0 and len(note["centres"]) == 2
    # and where no room was judged (the tiers of 12:40's run: beside the box, nearest): every spot offered
    # lies outside the radius
    monkeypatch.setattr(rlo, "lid_spots", lambda *a, **k: ([], "none judged"))
    monkeypatch.setattr(approach, "_chain", lambda: _Chain())
    offered = list(rlo.spot_candidates(ctx, tmap, -0.07, [0.0] * 7, reach=lambda *a, **k: None))
    assert offered and all(rlo.box_clearance(ctx, d) >= 0.0 for d, _t, _s in offered)


def _standing_ring(centre, inner, outer, cell=0.02, tall=0.03):
    """A TableMap with something `tall` standing from `inner` to `outer`
    round `centre` (a holder's flange), the rest seen bare."""
    from box_opening.perception.clear_table import TableMap

    n = 60
    x0, y0 = centre[0] - n * cell / 2, centre[1] - n * cell / 2
    xs = x0 + (np.arange(n) + 0.5) * cell
    ys = y0 + (np.arange(n) + 0.5) * cell
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    d = np.hypot(X - centre[0], Y - centre[1])
    tall_ = np.where((d >= inner) & (d <= outer), tall, np.nan)
    return TableMap(x0, y0, cell, tall_, np.ones((n, n), bool), np.zeros((n, n), bool), tuple(centre))


def _bare(monkeypatch, ctx):
    """A table seen bare all round ctx's box (a TableMap), no room judged on it (lid_spots finds
    none): the tiers that follow the room's — beside the box, the nearest — on a measured table."""
    monkeypatch.setattr(rlo, "lid_spots", lambda *a, **k: ([], "none judged"))
    return _standing_ring((float(ctx.cpose.xyz[0]), float(ctx.cpose.xyz[1])), 5.0, 6.0)


def test_the_nearest_rings_centre_on_the_held_button_and_pass_over_what_stands(ctx):
    from box_opening.models.container import from_container

    button = from_container(ctx.cpose, ctx.model.button_offset)
    bx, by = float(ctx.cpose.xyz[0]), float(ctx.cpose.xyz[1])
    assert rlo.nearest_centre(ctx) == (bx, by)  # no press held: the box as aimed
    ctx.held_xy = (float(button[0]) + 0.007, float(button[1]) - 0.017)  # 13:35's press, 18 mm off the aim
    assert rlo.nearest_centre(ctx) == pytest.approx((bx + 0.007, by - 0.017))
    ctx.held_xy = (float(button[0]) + 0.05, float(button[1]))  # not this box where it is now
    assert rlo.nearest_centre(ctx) == (bx, by)
    ctx.held_xy = (float(button[0]) + 0.007, float(button[1]) - 0.017)
    cx, cy = rlo.nearest_centre(ctx)
    pts = rlo._nearest_points(ctx, -0.07)
    r0 = min(math.hypot(p.xyz[0] - cx, p.xyz[1] - cy) for p in pts)
    lid_r = 0.5 * max(ctx.model.lid_dims[:2])
    # a flange standing out to 0.12 m: no spot whose disc (and LID_CLEAR_M) covers it
    flange = _standing_ring((cx, cy), 0.06, 0.12)
    kept = rlo._nearest_points(ctx, -0.07, flange)
    near = min(math.hypot(p.xyz[0] - cx, p.xyz[1] - cy) for p in kept)
    assert near > r0 and near - lid_r - rlo.clear_table.LID_CLEAR_M > 0.12 - 0.02  # past the flange's cells
    assert len(kept) < len(pts)


def _aim_frame(ctx, table_off=-0.02, h=0.5):
    """watcher.last_aim_frame as press_demo keeps it: a wrist frame looking
    straight down from `h` over the box onto a bare table `table_off` below
    the scene camera's (the box's base), and that offset as the aim read it."""
    from types import SimpleNamespace

    bx, by, bz = (float(v) for v in ctx.cpose.xyz)
    k = np.array([[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]])
    depth = np.full((240, 320), h + bz - (bz + table_off), np.float32)
    rot, trans = np.diag([1.0, -1.0, -1.0]), np.array([bx, by, bz + h])
    snap = SimpleNamespace(depth=depth, k=k, color=None, color_stamp=None)
    return "reach-aim", (snap, (rot, trans), {}), table_off


def test_the_look_poses_wrist_frame_is_a_cloud_on_its_own_table_moved_as_the_box_was(ctx):
    """Sweep lid_spot-pool-aim-frame, as the safety verifier corrected it:
    the wrist's cloud on its own table plane (it sits 7-68 mm below the
    scene camera's) and moved AIM_FORWARD_M as the box was from the aim; the
    fingers and the box taken out."""
    from types import SimpleNamespace

    from box_opening.perception import clear_table
    from box_opening.tasks.press_demo import container_half

    watcher = SimpleNamespace(last_aim_frame=_aim_frame(ctx))
    table_z = float(ctx.cpose.xyz[2])
    (pts, tz), why = rlo.aim_cloud(ctx, watcher, table_z)
    assert why is None and tz == pytest.approx(table_z - 0.02)
    assert pts[:, 2] == pytest.approx(table_z - 0.02, abs=1e-6)  # the table the wrist sees, where it reads it
    bx, by = float(ctx.cpose.xyz[0]), float(ctx.cpose.xyz[1])
    _tag, (snap, (rot, trans), _x), _off = watcher.last_aim_frame
    raw = clear_table.depth_points(snap.depth, snap.k, rot, trans)
    raw = raw[np.hypot(raw[:, 0] - bx, raw[:, 1] - by) > container_half(ctx.model) + clear_table.BOX_CLEAR_M]
    d = math.hypot(bx, by)
    shift = np.array([rp.AIM_FORWARD_M * bx / d, rp.AIM_FORWARD_M * by / d])
    assert len(pts) == len(raw) and pts[:, :2] == pytest.approx(raw[:, :2] + shift)
    for kept, why_none in (
        (None, "no aim kept its frame"),
        (("reach-aim", None, -0.02), "the aim's frame had no pose"),
        ((*_aim_frame(ctx)[:2], float("nan")), "the wrist read no table of its own at the aim"),
    ):
        assert rlo.aim_cloud(ctx, SimpleNamespace(last_aim_frame=kept), table_z) == (None, why_none)


def test_the_pooled_room_map_is_built_beside_the_scene_cameras_and_does_not_choose(ctx, monkeypatch, tmp_path):
    """Shadow mode (POOL_AIM_FRAME False): the room is judged on the scene
    camera's map as before, the pooled one built beside it; switched on,
    the pooled one is what the room is judged on."""
    from types import SimpleNamespace

    from box_opening.runtime import approach

    monkeypatch.setattr(approach, "_chain", lambda: SimpleNamespace())  # no arm to leave out of the scene's cloud
    bx, by, bz = (float(v) for v in ctx.cpose.xyz)
    g = np.arange(-0.5, 0.5, 0.01)
    X, Y = np.meshgrid(g + bx, g + by, indexing="ij")
    scene_pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, bz)], -1)
    scene_pts = scene_pts[(scene_pts[:, 1] - by) < -0.2]  # it sees the table only on one side
    scene_fix = SimpleNamespace(table_points=scene_pts, table_z_scene=bz, pose=ctx.cpose)
    watcher = SimpleNamespace(table_z=bz, last_aim_frame=_aim_frame(ctx, h=0.45))
    judged = []
    from box_opening.tasks import press_demo

    monkeypatch.setattr(
        rlo, "lid_spots", lambda m, cpose, worlds, tmap, table_z, cache=None: judged.append(tmap) or ([], "none")
    )
    for pool in (False, True):
        monkeypatch.setattr(rlo, "POOL_AIM_FRAME", pool)
        room = rlo.room_ahead(ctx, (None, watcher, scene_fix, None))
        room.thread.join(10.0)
        assert room.pooled is not None and room.scene_map is not None and room.aim_points > 0
        assert room.tmap is (room.pooled if pool else room.scene_map) and judged[-1] is room.tmap
        assert room.pooled.seen.sum() > room.scene_map.seen.sum()  # the wrist sees what the scene camera did not
    from box_opening.runtime.runner import Runner

    runner = Runner(ctx.client, log_dir=tmp_path, side_files=True)  # the maps: a run that keeps its frames
    drop = rlo.ContainerPose(xyz=(bx + 0.2, by, bz), yaw=0.0)
    note = rlo.pooled_note(ctx, runner, room, drop)
    assert "maps" not in rlo.pooled_note(ctx, Runner(ctx.client, log_dir=tmp_path / "plain"), room, drop)
    assert note["chose"] is True and note["pooled_seen"] > note["scene_seen"] and note["aim_points"] == room.aim_points
    assert note["spot"] == pytest.approx([bx + 0.2, by, 0.0])
    press_demo.wait_capture_writes()  # the maps are written off the motion's way
    maps = np.load(tmp_path / note["maps"])
    assert {"scene_tall", "scene_seen", "pooled_tall", "pooled_seen", "pooled_origin", "box_xy"} <= set(maps.files)
    assert int(maps["pooled_seen"].sum()) == note["pooled_seen"]
    assert rlo.pooled_note(ctx, runner, None, drop) is None  # no room judged ahead: nothing to compare


def test_the_retreat_from_the_lid_flies_from_where_the_arm_stands_held_at_its_start(ctx, monkeypatch):
    """Sweep planning-retreat-from-live, as verified: the planned retreat
    starts where the touch was sent, and the arm stands 8-43 mrad off that
    at the release — the 40 mrad drift gate refused two of seven. It is
    solved from the joints read, on the planned line's branch, and holds
    RETREAT_HOLD_S at its start; the home planned ahead flies from its end
    only within the drift gate."""
    from types import SimpleNamespace

    from box_opening.constants import DRIFT_REPLAN_RAD
    from box_opening.runtime import approach

    chain = _Chain()
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    planned = [[0.1] * 7, [0.11] * 7]
    carry = SimpleNamespace(retreat=planned, quat=[0.0, 1.0, 0.0, 0.0])
    world = ("full", "full.yaml")
    live = [0.1] * 7
    live[3] += 0.0428  # 14:26's 42.8 mrad
    ctx.client.joints = lambda: list(live)
    lg, off = rlo.retreat_from_live(ctx, carry, world)
    assert lg.name == "place:retreat" and lg.start_hold_s == rlo.RETREAT_HOLD_S == 0.25
    assert list(lg.traj.points[0].positions) == pytest.approx(live)  # from where the arm stands
    assert off == pytest.approx(0.0428) and off > DRIFT_REPLAN_RAD  # the fake line ends as far off as it starts
    live[2] = 0.1 + 2 * math.pi  # joint_3, continuous, read a turn round: put on the planned line's branch
    lg, off = rlo.retreat_from_live(ctx, carry, world)
    assert lg.traj.points[0].positions[2] == pytest.approx(0.1) and off == pytest.approx(0.0428)
    stuck = SimpleNamespace(fk=chain.fk, straight_line=lambda q0, xyz, quat: (None, "IK did not converge"))
    monkeypatch.setattr(approach, "_chain", lambda: stuck)
    assert rlo.retreat_from_live(ctx, carry, world) == (None, "IK did not converge")


def test_a_retreat_from_where_the_arm_stands_that_ends_past_the_drift_gate_plans_home_after_it(
    ctx, monkeypatch, tmp_path
):
    """The home planned ahead starts where the planned retreat ends: one
    from the live joints ending past DRIFT_REPLAN_RAD of that is not flown
    (the runner would refuse it with the lid let go), home is planned from
    where the retreat ends, while it flies."""
    real = rlo.retreat_from_live
    monkeypatch.setattr(rlo, "retreat_from_live", lambda c, carry, world: (real(c, carry, world)[0], 0.05))
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)])
    assert code == 0
    (back,) = (r for r in rows if r["leg"] == "retreat_from_live")
    assert (back["home_ahead"], back["home_flies"]) == (True, False)
    homes = [at for name, at in rec.plans if name == "home"]
    assert homes[-1] == rec.events.index("place:retreat") + 1  # planned as the retreat flew, from where it ends
    assert rec.events[-3:] == HOME


def test_a_press_only_run_chooses_no_lid_spot(ctx, monkeypatch, tmp_path, capsys):
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4)], extent=SMALL, press_only=True)
    assert code == 0 and ctx.lid_drop is None and not [r for r in rows if r["leg"] == "lid_spot"]
    assert "spot-look" not in rec.events


def test_a_clear_spot_the_wrist_sees_after_the_lift_is_where_the_lid_goes(ctx, monkeypatch, tmp_path, capsys):
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY, wrist=_table(), one_motion=False
    )
    assert code == 0
    (spot,) = (r for r in rows if r["leg"] == "lid_spot")
    assert spot["tier"] == "clear" and spot["mapped"] is True and spot["wrist_points"] > 0
    assert "where the cameras see room for it and the hand" in capsys.readouterr().out
    d = math.hypot(ctx.lid_drop.xyz[0] - BOX_XY[0], ctx.lid_drop.xyz[1] - BOX_XY[1])
    assert d >= 0.19  # lid_place_min_clear from the box
    assert not [r for r in rows if r["leg"] == "seen_obstacles"]  # the map is the mission's: no planner world gets it


def test_with_nothing_but_what_stands_round_the_box_the_lid_goes_back_on_the_box(ctx, monkeypatch, tmp_path, capsys):
    """Something standing all round the box, as far as the hand reaches (_wall): until 2026-11-17
    "nothing reads clear ... the most room seen" set the lid down on top of it, or (16:40) in front of
    the table's edge. No spot has table the cameras measured under the whole lid: the lid is taken
    nowhere. It goes back on the box, and the run says why."""
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY, wrist=_wall(_table()), one_motion=False
    )
    assert code == 1 and "lid:carry" not in rec.events and "place:open" not in rec.events
    after = rec.events[rec.events.index("spot-look") + 1 :]
    assert after[:4] == ["lid:back-over", "lid:put-back", "lid:release", "lid:rise-off"] and after[-1] == "home"
    (spot,) = (r for r in rows if r["leg"] == "lid_spot")
    assert (
        spot["xyz"] is None and spot["table"]["off_table"] > 0 and "on what stands there" in spot["table"]["first_off"]
    )
    # ... having looked wider first (owner 2026-11-19): the table past the wall is bare and measured, but
    # the wall, taller than the lid is carried, stands in the lid's way to every place on it
    assert spot["searched"] == ["the tiers", "wider"] and spot["table"]["blocked"] > 0
    out = capsys.readouterr().out
    assert "looking wider — the whole of the table the cameras measured" in out
    assert "PUT BACK: no carry plans to a spot on the table the cameras measured (of the spots looked at, " in out
    assert "not on the table the cameras measured" in out and "with something standing in the lid's way there" in out
    assert out.rstrip().endswith("the lid is back on the box, and the arm is home")


def test_with_no_camera_data_the_lid_is_taken_nowhere_and_goes_back_on_the_box(ctx, monkeypatch, tmp_path, capsys):
    """No camera's points of the table at all (it was "no camera shows the table: beside the box", a
    spot derived from the box alone, in the bench's zone): nothing measured a table this run, so the
    lid has nowhere it may be let go, whatever extent is on record. Lifted, looked for a spot with the
    wrist, none: back on the box."""
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY, scene=None
    )
    assert code == 1 and "place:open" not in rec.events and "lid:carry" not in rec.events
    (spot,) = (r for r in rows if r["leg"] == "lid_spot")
    assert (spot["xyz"], spot["tier"], spot["mapped"]) == (None, None, False)
    assert "no camera's map shows the table there" in spot["table"]["first_off"]
    out = capsys.readouterr().out
    assert "PUT BACK: no carry plans to a spot on the table the cameras measured" in out
    assert out.rstrip().endswith("the lid is back on the box, and the arm is home")


def test_a_clear_spot_the_scene_camera_sees_is_where_the_lid_goes_its_room_judged_while_the_press_flew(
    ctx, monkeypatch, tmp_path, capsys
):
    """The scene camera's table round the box, nothing on it: the lid goes
    to a clear spot, chosen before the close, and the close flies on to the
    set-down. Every spot's room is judged on the thread the press flies
    beside (room_ahead): the grip's lookahead ranks them from its cache."""
    import threading

    from box_opening.perception import clear_table

    judged_on = []

    def counted(real):
        def judge(*a, **k):
            judged_on.append(threading.current_thread().name)
            return real(*a, **k)

        return judge

    # spot by spot or in batches (clear_table.BATCHED_SPOTS): wherever the room is judged
    monkeypatch.setattr(clear_table, "clearance", counted(clear_table.clearance))
    monkeypatch.setattr(clear_table, "clearance_batch", counted(clear_table.clearance_batch))
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY, scene=_table()
    )
    assert code == 0 and "spot-look" not in rec.events
    assert [names for names, _g in _executions(rec)][-3:] == [CLOSE + CARRY, RELEASED, HOMED]
    (spot,) = (r for r in rows if r["leg"] == "lid_spot")
    assert (spot["tier"], spot["mapped"], spot["chosen"]) == ("clear", True, "ahead, with the scene camera")
    assert judged_on and set(judged_on) == {"lid-room"}
    assert "where the cameras see room for it and the hand" in capsys.readouterr().out
    d = math.hypot(ctx.lid_drop.xyz[0] - BOX_XY[0], ctx.lid_drop.xyz[1] - BOX_XY[1])
    assert d >= 0.19  # lid_place_min_clear from the box


def test_nothing_clear_the_scene_camera_sees_still_sets_the_lid_down_at_the_most_room_in_one_motion(
    ctx, monkeypatch, tmp_path, capsys
):
    """Nothing reads clear — every spot has a part of the hand nearer something than its margin
    asks — but the table is measured and bare under the lid and the hand: the spot with the most
    room is still where the lid goes (the fallback of 2026-10-28), in one motion."""
    from box_opening.perception.clear_table import Spot

    def short(m, cpose, worlds, tmap, table_z, judged_out=None, hand_at=None, cache=None):
        if judged_out is not None:
            judged_out.extend(
                Spot((BOX_XY[0] - 0.10, BOX_XY[1] + dy), 0.0, room, "the wrist camera", 0.0)
                for dy, room in ((0.20, -0.012), (-0.20, -0.004), (0.24, -0.030))
            )
        return [], "nowhere has room for the lid and the hand among what the cameras see"

    monkeypatch.setattr(rlo, "lid_spots", short)
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY)
    assert code == 0 and [names for names, _g in _executions(rec)][-3:] == [CLOSE + CARRY, RELEASED, HOMED]
    (spot,) = (r for r in rows if r["leg"] == "lid_spot")
    assert spot["tier"] == "most room" and spot["xyz"][:2] == pytest.approx([BOX_XY[0] - 0.10, BOX_XY[1] - 0.20])
    out = capsys.readouterr().out
    assert "nothing reads clear" in out and "the wrist camera 4 mm short" in out and "STOP" not in out
    assert out.rstrip().endswith("arm home")


def test_a_spot_home_does_not_plan_from_after_the_set_down_is_passed_over(ctx, monkeypatch, tmp_path):
    """2026-10-29 14:45: after the set-down j6 stood 0.09 rad from its
    limit and home would not plan from there: the arm held the lid's spot.
    Home after the set-down is planned with the carry, while the hand
    comes down to the knob: a spot it does not plan from is passed over,
    the next one's carry and home planned, and the lid goes there in one
    motion."""
    real, asked = rlo.plan_home_after, []

    def home_after(ctx_, cfg_, carry):
        asked.append(carry.drop)
        if len(asked) == 1:
            return None, "home after the set-down (planning failed for home: MotionGenStatus.INVALID_START_STATE)"
        return real(ctx_, cfg_, carry)

    monkeypatch.setattr(rlo, "plan_home_after", home_after)
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY)
    assert code == 0 and [names for names, _g in _executions(rec)][-3:] == [CLOSE + CARRY, RELEASED, HOMED]
    assert len(asked) == 2 and not rlo._same_spot(asked[0], asked[1])
    (spot,) = (r for r in rows if r["leg"] == "lid_spot")
    assert "home after the set-down" in spot["refused"]
    assert spot["xyz"][:2] == pytest.approx(list(asked[1].xyz[:2]), abs=1e-4) and ctx.lid_drop is asked[1]


def test_the_lid_s_way_to_the_table_stopped_part_way_holds_the_lid_and_lets_go(ctx, monkeypatch, tmp_path, capsys):
    """The lift to the set-down is one goal (the 5th); stopped part way by
    the driver, the arm holds the lid where it stopped, the springs let go
    (rp.let_go): exit 1, never a release or home."""
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], ends={5: trp.STOPPED})
    assert code == 1 and ctx.client.lets_go
    assert "place:open" not in rec.events and "home" not in rec.events[rec.events.index("grip:close") :]
    assert "STOP: the lid's way from the box to the table stopped during" in capsys.readouterr().out


def test_the_room_judged_while_the_press_flew_is_the_box_s_only_where_it_still_stands():
    """The room is judged round the box where ctx.cpose had it: the box
    since put elsewhere (a re-aim), or the judging failed, it is judged
    again."""
    room = rlo.Room(at=(0.5, -0.4, 0.2))
    here = SimpleNamespace(cpose=ContainerPose(xyz=(0.5, -0.4, 0.2), yaw=0.0))
    assert rlo.room_for(here, room) is room and rlo.room_for(here, None) is None
    moved = SimpleNamespace(cpose=ContainerPose(xyz=(0.505, -0.4, 0.2), yaw=0.0))
    assert rlo.room_for(moved, room) is None
    room.at = None
    assert rlo.room_for(here, room) is None


def test_the_wrist_points_of_the_held_lid_are_not_an_obstacle(ctx, monkeypatch):
    """At the spot look the wrist sees the lid in the fingers, ~0.1 m over
    the table under the hand: taken out (wrist_table), it stands nowhere on
    the map; left in, it would stand there and block the spots beside it."""
    from box_opening.perception import clear_table
    from box_opening.runtime import approach

    chain = _Chain()  # fk: the tool at [0.5, -0.4, 0.45], the hand straight up (R eye): pads 11 mm above it
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    ctx.model = ContainerModel.load(PINK)
    ctx.cpose = ContainerPose(xyz=(BOX_XY[0], BOX_XY[1], TABLE_Z), yaw=0.0)
    table = _table()
    pads = np.array([0.5, -0.4])
    g = np.arange(-0.05, 0.0501, 0.004)
    X, Y = np.meshgrid(g + pads[0], g + pads[1], indexing="ij")
    disc = np.hypot(X - pads[0], Y - pads[1]) < 0.0525
    lid = np.stack([X[disc], Y[disc], np.full(int(disc.sum()), TABLE_Z + 0.10)], -1)
    held = np.concatenate([table, lid])
    scene = SimpleNamespace(pose=ctx.cpose, table_points=table, table_z_scene=TABLE_Z)

    def standing_at_pads(tmap):
        X_, Y_ = tmap.centres()
        near = np.hypot(X_ - pads[0], Y_ - pads[1]) < 0.04
        return int(np.isfinite(tmap.tall[near]).sum())

    raw = clear_table.table_map([(table, TABLE_Z), (held, TABLE_Z)], BOX_XY)
    assert standing_at_pads(raw) > 0  # counted, the held lid stands on the map
    tmap = rlo.spot_map(ctx, held, [0.0] * 7, scene, None, TABLE_Z, lens_in_hand=(0.0, 0.0, 0.5))
    assert standing_at_pads(tmap) == 0
    assert len(rlo.wrist_table(ctx, held, [0.0] * 7, (0.0, 0.0, 0.5))) < len(table)  # the lid's column and the box out


def test_a_ctrl_c_stops_the_lookahead_between_two_spots_and_while_it_waits_for_the_room(ctx, monkeypatch):
    """The grip's lookahead (carry_ahead) checks the abort flag before each spot's carry and
    each reach check, and while it waits for the room judged during the press: a Ctrl+C there
    goes out at once, never after the rest of the planning."""
    import threading
    import time

    from box_opening.runtime.abort import AbortFlag, LookaheadAborted

    ctx.client._abort = AbortFlag()
    ctx.client._abort.requested = True
    planned = []
    monkeypatch.setattr(rlo, "plan_carry", lambda *a, **k: planned.append(1) or (None, "x"))
    monkeypatch.setattr(rlo, "spot_candidates", lambda *a, **k: iter([(SimpleNamespace(xyz=(0.5, 0.0, 0.2)), "t", "")]))
    monkeypatch.setattr(rlo, "spot_map", lambda *a, **k: None)
    monkeypatch.setattr(rlo, "_table_z", lambda *a, **k: 0.2)
    with pytest.raises(LookaheadAborted):
        rlo.carry_ahead(ctx, load_press_demo(PINK), (None, None, None, None), [0.0] * 7)
    assert planned == []
    room = rlo.Room(at=tuple(float(v) for v in ctx.cpose.xyz))
    done = threading.Event()
    room.thread = threading.Thread(target=done.wait, args=(5.0,), daemon=True)
    room.thread.start()
    t0 = time.monotonic()
    with pytest.raises(LookaheadAborted):
        rlo.room_for(ctx, room)
    assert time.monotonic() - t0 < 1.0  # not the room's 5 s
    done.set()


def test_the_carry_search_in_two_parts_finds_what_carry_ahead_does_and_plans_only_in_its_second(ctx, monkeypatch):
    """Sweep motion_timing-g-06: the grip's lookahead split at a plan
    boundary. CarrySearch.start finds the first spot the hand reaches and
    plans nothing; finish plans the carry and home there, trying the next
    spots where they are refused — the same answer carry_ahead gives in one
    go (it is the two, one after the other)."""
    cfg = load_press_demo(PINK)
    spots = [(SimpleNamespace(xyz=(0.5 + 0.01 * i, 0.0, 0.2), yaw=0.0), "clear", "says %d" % i) for i in range(4)]
    asked, plans = [], []

    def candidates(*a, **k):
        for s in spots:
            asked.append(s[2])
            yield s

    def carry(ctx_, cfg_, q, drop):
        plans.append(drop.xyz[0])
        return (None, "refused") if drop.xyz[0] < 0.515 else (SimpleNamespace(legs=[], retreat=[]), None)

    monkeypatch.setattr(rlo, "spot_candidates", candidates)
    monkeypatch.setattr(rlo, "spot_map", lambda *a, **k: None)
    monkeypatch.setattr(rlo, "_table_z", lambda *a, **k: 0.2)
    monkeypatch.setattr(rlo, "plan_carry", carry)
    monkeypatch.setattr(rlo, "plan_home_after", lambda ctx_, cfg_, c: (["home"], None))
    eyes = (None, None, None, None)
    search = rlo.CarrySearch(ctx, cfg, eyes, [0.0] * 7, home=True).start()
    assert asked == ["says 0"] and plans == []  # the first spot found, nothing planned
    got = search.finish()
    assert plans == pytest.approx([0.5, 0.51, 0.52]) and asked == ["says 0", "says 1", "says 2"]
    assert got.drop.xyz[0] == pytest.approx(0.52) and got.home == ["home"] and len(got.refused) == 2
    asked.clear()
    plans.clear()
    once = rlo.carry_ahead(ctx, cfg, eyes, [0.0] * 7, home=True)
    assert (once.drop.xyz, once.tier, once.refused, once.home) == (got.drop.xyz, got.tier, got.refused, got.home)
    # the rise's half with no lift hands that on; the way down's makes nothing of it
    assert rlo.grip_ahead_down((None, "no lift", None)) == (None, "no lift", None)


def test_no_carry_that_plans_puts_the_lid_back_on_the_box_and_goes_home(ctx, monkeypatch, tmp_path, capsys):
    """Every spot the hand reaches refused by the planner (CARRY_TRIES of the
    tiers', then as many of the whole table's): the
    arm stands holding the lid over the box. It does not stay there: the lid
    goes back where it came from, back over the box and straight down to
    where the knob was held, the fingers open, the hand rises off the knob
    and goes home. Exit 1 (the lid did not come off), the arm home."""
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY, refuse_carry=True
    )
    assert code == 1 and "lid:carry" not in rec.events and "place:open" not in rec.events
    (spot,) = (r for r in rows if r["leg"] == "lid_spot")
    # ... CARRY_TRIES of the tiers' spots, and as many more of the wider search's (owner 2026-11-19)
    assert spot["xyz"] is None and spot["refused"].count("IK_FAIL") == 2 * rlo.CARRY_TRIES
    assert spot["searched"] == ["the tiers", "wider"]
    after = rec.events[rec.events.index("spot-look") + 1 :]
    assert after[:4] == ["lid:back-over", "lid:put-back", "lid:release", "lid:rise-off"] and after[-1] == "home"
    (back,) = (r for r in rows if r["leg"] == "lid_put_back")
    assert back["ok"] is True and back["why"].startswith("no carry plans to a spot on the table the cameras measured")
    assert ctx.lid_lifted is None and not ctx.client.lets_go  # out of the fingers; nothing stopped part way
    out = capsys.readouterr().out
    assert "PUT BACK: no carry plans to a spot on the table the cameras measured" in out
    assert "the lid is back on the box, and the arm is home" in out and "the arm holds" not in out.split("PUT BACK")[-1]


def test_a_carry_the_run_refuses_before_it_starts_puts_the_lid_back(ctx, monkeypatch, tmp_path, capsys):
    """The carry refused by the run's own checks before anything went out
    (here its start gate): the arm stands at the clear holding the lid, and
    the lid goes back on the box, as when no carry plans."""
    from box_opening.runtime.runner import Runner

    real = Runner._refusal
    monkeypatch.setattr(
        Runner,
        "_refusal",
        lambda self, leg: "the arm is 0.06 rad from the start" if leg.name == "lid:carry" else real(self, leg),
    )
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY, one_motion=False
    )
    assert code == 1 and "place:open" not in rec.events
    after = rec.events[rec.events.index("place:touch") + 1 :]  # after the carry's legs, refused as one
    assert after[:4] == ["lid:back-over", "lid:put-back", "lid:release", "lid:rise-off"] and after[-1] == "home"
    assert "PUT BACK: the lid's way to the table was refused before it started" in capsys.readouterr().out


def test_the_spot_look_takes_the_first_frame_shot_once_the_arm_stands_still(monkeypatch):
    """The spot look waited a blind 0.3 s (depth_press.look's settle) once
    the lift and the clear arrived, whatever the arm did: after 15:45's
    looks it stood still only at +0.80 s. It now waits for the arm to
    stand still (runner.wait_still, noted before 'spot-look'), then takes
    the first frame shot after that — never one shot while the arm settled
    — with no blind settle on top."""
    from box_opening.tasks import press_demo as pd

    clock = {"ns": 1_000_000_000}
    grab = SimpleNamespace(color_stamp=SimpleNamespace(sec=0, nanosec=900_000_000), depth=object())
    node = SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=clock["ns"])))
    events = []

    def wait_still(before):
        events.append(("still", before))
        clock["ns"] += 800_000_000  # the arm settles 0.8 s
        grab.color_stamp = SimpleNamespace(sec=1, nanosec=700_000_000)  # a frame shot while it settled

    def spin(node_):
        events.append("spin")
        clock["ns"] += 20_000_000
        if events.count("spin") == 2:  # the first frame shot after the wait
            grab.color_stamp = SimpleNamespace(sec=1, nanosec=850_000_000)

    monkeypatch.setattr(pd, "_spin_detect", spin)
    monkeypatch.setattr(pd, "wrist_points", lambda watcher: ("points of", watcher.grab.color_stamp.nanosec))
    notes = []
    runner = SimpleNamespace(wait_still=wait_still, note=lambda kind, **kw: notes.append((kind, kw)))
    got = rlo.spot_look(node, SimpleNamespace(grab=grab), runner)
    assert events == [("still", "spot-look"), "spin", "spin"]
    assert got == ("points of", 850_000_000)
    assert notes == [("spot_look", {"points": 2, "why": None, "frame": None})]  # no colour image: nothing to keep


def test_a_spot_look_with_no_points_says_why(monkeypatch):
    """10:28 2026-10-29: the spot look gave the room map no points and the
    record could not say why. It now notes it: no new frame once the arm
    stood still, no pose for the frame, or what the depth held — the lid in
    the fingers nearer the lens than the hand's cut, nothing read, or past
    2 m."""
    from box_opening.perception import depth_source
    from box_opening.tasks import press_demo as pd

    clock = {"ns": 1_000_000_000}
    node = SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=clock["ns"])))
    notes = []
    runner = SimpleNamespace(wait_still=lambda before: None, note=lambda kind, **kw: notes.append((kind, kw)))
    # no frame newer than the arm standing still
    grab = SimpleNamespace(color_stamp=SimpleNamespace(sec=0, nanosec=500_000_000), depth=np.zeros((4, 4)))
    monkeypatch.setattr(pd, "_spin_detect", lambda node_: None)
    monkeypatch.setattr(rlo, "SPOT_FRAME_S", 0.05)
    assert rlo.spot_look(node, SimpleNamespace(grab=grab), runner) is None
    assert notes[-1][1]["points"] == 0 and notes[-1][1]["why"].startswith("no new wrist frame within 0.1 s")
    # a frame, no points: why, from the frame
    depth = np.zeros((10, 10), np.float32)
    depth[:4] = 0.12  # the lid, 12 cm off the lens
    depth[4:6] = 2.5  # past the cut
    grab = SimpleNamespace(color_stamp=SimpleNamespace(sec=2, nanosec=0), depth=depth, k=np.eye(3))
    monkeypatch.setattr(pd, "wrist_points", lambda watcher: np.zeros((0, 3)))
    monkeypatch.setattr(depth_source, "camera_pose_at", lambda g: (np.eye(3), np.zeros(3)))
    assert len(rlo.spot_look(node, SimpleNamespace(grab=grab), runner)) == 0
    assert notes[-1] == (
        "spot_look",
        {
            "points": 0,
            "frame": None,
            "why": (
                "no depth in range: of 100 pixels 40 read nothing, 40 nearer than %.2f m, 20 past 2 m"
                % rlo.clear_table.WRIST_NEAREST_M
            ),
        },
    )
    monkeypatch.setattr(depth_source, "camera_pose_at", lambda g: None)
    monkeypatch.setattr(pd, "wrist_points", lambda watcher: None)
    assert rlo.spot_look(node, SimpleNamespace(grab=grab), runner) is None  # asked until SPOT_FRAME_S
    assert notes[-1][1]["points"] == 0
    assert notes[-1][1]["why"].startswith("no camera pose at the new frames' stamps (TF) within")
    # TF lands after the frame (2 of 6 logged spot looks gave up here with ~0.9 s left): asked again
    asked = []

    def late(g):
        asked.append(1)
        return None if len(asked) < 3 else (np.eye(3), np.zeros(3))

    monkeypatch.setattr(depth_source, "camera_pose_at", late)
    # wrist_points places the frame by the same pose: none until TF has it (two asks in)
    monkeypatch.setattr(pd, "wrist_points", lambda watcher: None if len(asked) < 2 else np.zeros((5, 3)))
    monkeypatch.setattr(rlo, "SPOT_FRAME_S", 1.0)
    assert len(rlo.spot_look(node, SimpleNamespace(grab=grab), runner)) == 5 and notes[-1][1]["points"] == 5
    assert rlo.spot_look(None, SimpleNamespace(grab=grab), runner) is None and notes[-1][1]["why"] == "no node to spin"
    assert rlo.spot_look(node, SimpleNamespace(), runner) is None and notes[-1][1]["why"] == "no wrist camera"


# 2026-10-29 12:02:48: where the arm stood with the lid lifted, at the spot
# look (the carry's first point), the button its after-look aimed at, and
# the scene camera's table. The carry from there turned the hand 91 deg
# about the vertical (by FK) to a spot with 46 mm to spare, joint_7 151.
AT_1202_SPOT_LOOK = [0.1754, -0.1941, 3.0832, -1.0629, -0.0708, -1.7567, 1.6355]
BUTTON_1202 = (0.6422, -0.009, 0.3117)
TABLE_1202 = 0.2163


def test_the_hands_turn_to_each_spot_is_counted_from_where_the_arm_holds_the_lid(ctx, monkeypatch):
    """Flatbot's own kinematics at 12:02's spot look: the hand faces, by
    FK, within a degree of the reach-out's grip at the button
    (reach_press.reach_quat, what clear_lid_drops takes without the arm).
    spot_candidates counts the hand's turn to each spot from where the FK
    has it face, and says it: on open table it comes down unturned, the
    hand turning ~1 deg. Joint_7 turned on 1.4 rad, the hand faces 80 deg
    otherwise, and the spot it comes down at is the quarter turn nearest
    that — not the press attitude unturned, 80 deg away."""
    from box_opening.geometry import ang_diff
    from box_opening.perception import clear_table
    from box_opening.runtime.kinematics import mat_to_quat_xyzw

    chain, _cfg, home = trp._on_flatbot(ctx, monkeypatch)
    ctx.cpose = rp.cpose_for_button(BUTTON_1202, 0.0, ctx.model.button_offset)
    box = ctx.cpose.xyz[:2]
    tmap = clear_table.table_map([(_table(xy=box, z=TABLE_1202), TABLE_1202)], box)
    pressing = clear_table.hand_yaw(ctx.model.press_quat(ctx.cpose.xyz))

    def facing(q):
        return clear_table.hand_yaw(mat_to_quat_xyzw(np.asarray(chain.fk(q)[0], float)))

    grip = clear_table.hand_yaw(rp.reach_quat(ctx.model, BUTTON_1202[:2], 0.0))
    assert abs(math.degrees(ang_diff(facing(AT_1202_SPOT_LOOK), grip))) < 1.0
    drop, tier, says = next(rlo.spot_candidates(ctx, tmap, TABLE_1202, AT_1202_SPOT_LOOK, seeds=(home,)))
    turn = math.degrees(ang_diff(pressing, facing(AT_1202_SPOT_LOOK)))
    assert (tier, drop.yaw) == ("clear", 0.0) and abs(turn) < 2.0
    assert "the hand turns %+.0f deg)" % turn in says
    turned = list(AT_1202_SPOT_LOOK)
    turned[6] += 1.4
    assert abs(math.degrees(ang_diff(facing(turned), facing(AT_1202_SPOT_LOOK)))) == pytest.approx(80.2, abs=1.0)
    nearest = min(clear_table.TURNS, key=lambda t: abs(ang_diff(pressing + t, facing(turned))))
    assert abs(nearest) == pytest.approx(math.pi / 2)
    drop, tier, says = next(rlo.spot_candidates(ctx, tmap, TABLE_1202, turned, seeds=(home,)))
    assert tier == "clear" and drop.yaw == pytest.approx(nearest), says


def test_the_lid_s_spot_is_mapped_and_its_carry_planned_from_where_the_arm_stood_still(ctx, monkeypatch, tmp_path):
    """The joints the wrist's frame is mapped from (spot_map) and the carry
    planned from (spot_candidates, plan_carry) are read after the spot
    look's wait for a still arm, where its frame was shot: not before it,
    the arm still settling from the lift and the clear."""
    from box_opening.runtime.runner import Runner

    stood = [0.01 * (k + 1) for k in range(7)]
    seen = []

    def look(node, watcher, runner_):
        ctx.client.live = list(stood)  # the arm came to rest while the spot look waited
        return None

    def candidates(ctx_, tmap, table_z, q, seeds=(), reach=None, spot_z=None, passed=None, **more):
        seen.append(("spot_candidates", list(q)))
        yield ContainerPose(xyz=(0.45, -0.25, -0.027), yaw=0.0), "beside the box", "beside the box"

    monkeypatch.setattr(rlo, "spot_look", look)
    monkeypatch.setattr(rlo, "spot_map", lambda ctx_, wrist, q, *rest: seen.append(("spot_map", list(q))))
    monkeypatch.setattr(rlo, "spot_candidates", candidates)
    monkeypatch.setattr(
        rlo, "plan_carry", lambda ctx_, cfg_, q, drop: (seen.append(("plan_carry", list(q))), (None, "refused"))[1]
    )
    eyes = (None, SimpleNamespace(table_z=-0.027), None, None)
    assert rlo.choose_and_carry(ctx, load_press_demo(PINK), Runner(ctx.client, log_dir=tmp_path), eyes) == 1
    assert seen == [("spot_map", stood), ("spot_candidates", stood), ("plan_carry", stood)]


def test_the_carry_planned_while_the_lift_flew_is_planned_again_where_the_lids_own_map_keeps_it_out(
    ctx, monkeypatch, tmp_path, capsys
):
    """The carry carry_ahead planned while the lift flew was held out of the boxes in force then; the
    lid's own map, read after the lift (press_demo.keep_out_counter), may show a counter edge the
    scene camera's did not. Where that carry would now go into its keep-out (carry_refusal) it is
    not flown, even to the same spot: the carry is planned again here, against the boxes in force
    now, and the record says why."""
    from box_opening.runtime.runner import Runner

    drop = ContainerPose(xyz=(0.45, -0.25, -0.027), yaw=0.0)
    seen, notes = [], []
    monkeypatch.setattr(rlo, "spot_look", lambda node, watcher, runner_: None)
    monkeypatch.setattr(rlo, "spot_map", lambda *a, **k: None)
    monkeypatch.setattr(rlo, "spot_candidates", lambda *a, **k: iter([(drop, "clear", "clear")]))
    stale = "place:touch (the cabinet: arm_robotiq_85_base_link 9 mm into cabinet/edge0)"
    monkeypatch.setattr(rlo, "carry_refusal", lambda carry: (seen.append(("checked", carry)), stale)[1])
    monkeypatch.setattr(rlo, "plan_carry", lambda ctx_, cfg_, q, d: (seen.append(("plan_carry", d)), (None, "no"))[1])
    ahead = rlo.CarryAhead([float(v) for v in ctx.client.joints()], drop=drop, tier="clear", carry="planned ahead")
    runner = Runner(ctx.client, log_dir=tmp_path)
    monkeypatch.setattr(runner, "note", lambda kind, **kw: notes.append((kind, kw)))
    eyes = (None, SimpleNamespace(table_z=-0.027), None, None)
    assert rlo.choose_and_carry(ctx, load_press_demo(PINK), runner, eyes, ahead=ahead) == 1
    assert seen == [("checked", "planned ahead"), ("plan_carry", drop)]  # the same spot, planned again
    (spot,) = (kw for kind, kw in notes if kind == "lid_spot")
    assert spot["ahead"]["flown"] is False and spot["ahead"]["why"] == "the lid's own map keeps it out: " + stale
    assert "the carry planned while the lift flew is not flown: the lid's own map keeps it out" in (
        capsys.readouterr().out
    )


# ---- the carry planned while the lift flies ------------------------------------


# the scene camera's table, 0.8 m across round the box: room beside it
NEAR = dict(extent=ROOMY, scene=_table(r=0.4))


def _lift_flight(rec):
    """Where the lift's runner.run handed its legs: the plans made while the
    lift and the clear flew are those noted with this many events before."""
    return rec.events.index("lid:clear") + 1


def test_the_carry_planned_while_the_lift_flies_is_flown_where_the_spot_chosen_after_it_is_the_same(
    ctx, monkeypatch, tmp_path, capsys
):
    """13:37 2026-10-29: the arm stood 3.2 s with the lid in the fingers, 2.38
    s of it choosing the spot and planning the carry. While the lift and the
    clear fly, the scene camera's map, its spots the hand reaches and the
    carry to the best are made (carry_ahead) — the one planner call in that
    flight. After the lift the spot look and the map with the wrist's frame
    in it choose the spot as before; it is the same, the arm where the
    clear was to end: that carry flies, no planner call at rest, the
    reach verdicts taken from it. The record says so and what it saved."""
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], one_motion=False, wrist=_table(r=0.4), **NEAR
    )
    assert code == 0 and rec.events[-3:] == HOME
    flight = _lift_flight(rec)
    assert [name for name, at in rec.plans if at == flight] == ["lid:carry"]  # the lift's flight: the carry alone
    # nothing planned from the spot look until the carry flies (then home, while it flies)
    assert [name for name, at in rec.plans if flight < at <= rec.events.index("lid:carry")] == []
    assert rec.most_at_once == 1  # never two plans at once
    (spot,) = (r for r in rows if r["leg"] == "lid_spot")
    ahead = spot["ahead"]
    assert (ahead["flown"], ahead["why"], ahead["off_rad"]) == (True, None, 0.0)
    assert spot["tier"] == "clear" and spot["xyz"] == pytest.approx(ahead["xyz"])
    assert ahead["reused"] >= 1 and ahead["saved_s"] >= 0.0
    assert spot["carry_s"] == 0.0 and spot["ik_s"] == 0.0  # no plan, no reach judged again at rest
    assert all(spot[k] >= 0.0 for k in ("look_s", "map_s", "candidates_s"))
    assert "the carry planned while the lift flew goes" in capsys.readouterr().out


def test_the_carry_planned_while_the_lift_flies_is_not_flown_where_the_wrist_shows_the_spot_taken(
    ctx, monkeypatch, tmp_path, capsys
):
    """The scene camera sees room beside the box; the wrist, the lid lifted,
    sees something standing all round it (_wall): the spot is chosen on what
    the wrist shows (owner: chosen with the lid lifted), not the scene's —
    the carry planned while the lift flew is not flown, and the record says
    why. With something standing wherever the lid could go, it goes nowhere:
    back on the box (until 2026-11-17 it went on top of what stood there,
    "the most room")."""
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], one_motion=False, wrist=_wall(_table(r=0.4)), **NEAR
    )
    assert code == 1 and "lid:carry" not in rec.events and "place:open" not in rec.events
    flight = _lift_flight(rec)
    carries = [at for name, at in rec.plans if name == "lid:carry"]
    assert carries == [flight]  # planned ahead once, on the scene camera's map; none at rest: no spot to plan to
    assert rec.most_at_once == 1
    (spot,) = (r for r in rows if r["leg"] == "lid_spot")
    ahead = spot["ahead"]
    assert spot["xyz"] is None and ahead["flown"] is False and ahead["why"] == "no carry plans here"
    out = capsys.readouterr().out
    assert "the carry planned while the lift flew is not flown" in out
    assert out.rstrip().endswith("the lid is back on the box, and the arm is home")


@pytest.mark.parametrize("push, flown", [(0.03, True), (0.05, False)])
def test_the_carry_planned_while_the_lift_flies_is_flown_only_from_where_it_starts(
    ctx, monkeypatch, tmp_path, push, flown
):
    """The arm stands `push` rad on joint_2 off where the clear was to end
    when the spot look is done: within DRIFT_REPLAN_RAD the carry planned
    ahead flies, as the runner flies any planned leg from there; beyond,
    the carry is planned from where the arm stands and the reach of each
    spot judged again — a plan from somewhere else is never flown."""
    from box_opening.constants import DRIFT_REPLAN_RAD

    def pushed():
        ctx.client.live[1] += push  # while the spot look waited

    code, rec, _chain, rows = _full_flow(
        ctx,
        monkeypatch,
        tmp_path,
        [(BUTTON, 0.4), (UP, 9.0)],
        one_motion=False,
        wrist=_table(r=0.4),
        at_spot_look=pushed,
        **NEAR,
    )
    assert code == 0
    (spot,) = (r for r in rows if r["leg"] == "lid_spot")
    ahead = spot["ahead"]
    assert ahead["off_rad"] == pytest.approx(push) and ahead["flown"] is flown
    carries = [at for name, at in rec.plans if name == "lid:carry"]
    assert len(carries) == (1 if flown else 2)
    start = rec.legs["lid:carry"].traj.points[0].positions
    here = ctx.client.exec_starts[-3]  # the carry's execution, before the retreat's and home's
    assert max(abs(a - b) for a, b in zip(start, here, strict=False)) <= DRIFT_REPLAN_RAD
    if not flown:
        assert ahead["why"].startswith("the arm stands 0.050 rad from where it was planned from")
        assert ahead["reused"] == 0 and spot["ik_s"] >= 0.0 and spot["carry_s"] >= 0.0


def test_after_a_grip_once_more_the_carry_is_the_one_planned_while_its_own_lift_flew(ctx, monkeypatch, tmp_path):
    """A first close on air returns before its lift: nothing is planned
    ahead for it. The grip once more lifts the lid, and the carry planned
    while that lift flew is the one flown — one carry plan in the run."""
    code, rec, _chain, rows = _full_flow(
        ctx,
        monkeypatch,
        tmp_path,
        [(BUTTON, 0.4), (UP, 9.0), (REGRIP, 9.0)],
        knuckle=[AIR, KNOB_ON],
        one_motion=False,
        **NEAR,
    )
    assert code == 0 and rec.events.count("grip:close") == 2
    second_lift = len(rec.events) - rec.events[::-1].index("lid:clear")  # events before the second lift's plans
    assert [at for name, at in rec.plans if name == "lid:carry"] == [second_lift]
    (spot,) = (r for r in rows if r["leg"] == "lid_spot")
    assert spot["ahead"]["flown"] is True and rec.most_at_once == 1


def test_a_lookahead_result_is_told_how_long_its_host_stood_for_it():
    """The grip's lookahead (lift, why, CarryAhead) or the lift's
    (CarryAhead): the CarryAhead learns its host's stand at its end
    (Runner.lookahead_stood_s), for the lid_spot row."""
    from types import SimpleNamespace

    placed = rlo.CarryAhead([0.0] * 7)
    got = rlo._lookahead(SimpleNamespace(lookahead_result=("lift", None, placed), lookahead_stood_s=0.31))
    assert got == ("lift", None, placed) and placed.host_stood_s == 0.31
    alone = rlo.CarryAhead([0.0] * 7)
    assert rlo._lookahead(SimpleNamespace(lookahead_result=alone, lookahead_stood_s=0.02)) is alone
    assert alone.host_stood_s == 0.02
    assert rlo._lookahead(SimpleNamespace(lookahead_result=(None, "no lift", None), lookahead_stood_s=0.5)) == (
        None,
        "no lift",
        None,
    )
    assert rlo._lookahead(SimpleNamespace(lookahead_result=None)) is None  # a runner that keeps no stand


def test_a_spot_is_the_carry_planned_ahead_s_own_within_a_centimetre_and_the_same_turn():
    a = ContainerPose(xyz=(0.60, -0.32, 0.216), yaw=0.0)
    assert rlo._same_spot(a, ContainerPose(xyz=(0.607, -0.327, 0.216), yaw=2 * math.pi))
    assert not rlo._same_spot(a, ContainerPose(xyz=(0.611, -0.32, 0.216), yaw=0.0))
    assert not rlo._same_spot(a, ContainerPose(xyz=(0.60, -0.32, 0.216), yaw=math.pi / 2))
    assert rlo._spot_key(a) == rlo._spot_key(ContainerPose(xyz=(0.60001, -0.32, 0.3), yaw=2 * math.pi))


def test_the_fingers_close_on_the_knob_once_the_arm_stands_still(ctx, monkeypatch, tmp_path):
    """16:23 2026-10-28: grip:close went out about 30 ms after the way down
    to the knob "arrived" — its trajectory's clock run out, the arm 2.3 mm
    off its end. The close now goes out once the arm stands still
    (Leg.send_when_still): after the way down and the open that rode it,
    before the fingers move, noted 'still' before grip:close. Nothing else
    waits so: home flies in the run's own mode, impedance (core.as_home),
    so no mode change waits for it."""
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)])
    assert code == 0
    assert [name for name, lg in rec.legs.items() if lg.send_when_still] == ["grip:close"]
    # after the look, the way in, the way back and the way down to the knob
    assert ctx.client.stills == [4]
    legs = [r["leg"] for r in rows]
    stills = [i for i, name in enumerate(legs) if name == "still"]
    assert [rows[i]["before"] for i in stills] == ["grip:close"]
    assert legs.index("grip:down") < legs.index("grip:open") < stills[0] < legs.index("grip:close")


# ---- flatbot's URDF: the five boxes of the owner's brief ------------------------

FLATBOT_BOXES = [(0.51, -0.27), (0.57, -0.19), (0.716, -0.13), (0.668, -0.33), (0.60, -0.49)]
LID = 0.3125  # 16:23's lid; the scene's table ~0.10 below it


def _held_at_bottom(chain, stroke, top_z, cfg):
    """The press leg's sample whose pads stand nearest the button's bottom:
    where the spring holds the arm on a button that stops it there."""
    pts = stroke.legs[-1][1]
    bottom = top_z - cfg.button_travel_m
    return min(pts, key=lambda q: abs(dp._pads_z(chain, q) - bottom))


def _plan_by_ik(chain, home):
    """rp._plan_motion's stand-in on flatbot: the planner stood in for by
    the arm's IK, a goal it reaches flown as two points."""

    def plan_ik(ctx_, state, name, target, world, speed):
        if target[0] == "joints":
            q = list(target[1])
        else:
            q = chain.nearest_solution(list(state.joints), target[1], target[2], seeds=(home,))
            assert q is not None, name
        return rp.line_leg(
            ctx_,
            name,
            [list(state.joints), [float(v) for v in q]],
            target[2] if len(target) > 2 else [0, 0, 0, 1],
            speed,
            None,
            world,
        ), state

    return plan_ik


@pytest.mark.parametrize("name", ["ankou_pink", "oxo_pop", "oxo_pop_small"])
def test_the_set_downs_slow_stretch_begins_below_every_containers_hover(name):
    """The carry's hover (Place.hover_for: hover_standoff and a lid over
    where the plan has the lid meet the table, at least) stands above where
    the set-down's slow stretch begins, SETDOWN_SLOW_M over that: the way
    down from it goes down. And the touch's end is under the plan's table,
    SETDOWN_PAST_M: far enough for the watch's rises (3 and 5 mm of the
    reference past the touch) where the table stands as planned, and not so
    far that the lid's own thickness is gone (the fingertips stay over the
    table)."""
    from box_opening.tasks import press_contact as pc

    model = ContainerModel.load("capabilities/box_opening/config/containers/%s.yaml" % name)
    assert model.hover_standoff + model.lid_dims[2] > rlo.SETDOWN_SLOW_M + 0.01
    assert (
        max(pc.CONTACT_RISE_M, pc.CONTACT_SLOW_RISE_M) < rlo.SETDOWN_PAST_M < model.lid_dims[2] - rlo.GRIP_BELOW_LID_M
    )
    assert rlo.SETDOWN_CREDIBLE_M < rlo.SETDOWN_SLOW_M - 0.03  # the settle's 33 mm end above the band


@pytest.mark.parametrize("shortest_walk", [False, True], ids=["today's posture", "the quickest walks' posture"])
@pytest.mark.parametrize("box", FLATBOT_BOXES, ids=["%.3f,%.2f" % b for b in FLATBOT_BOXES])
def test_on_flatbot_the_whole_lid_off_is_reachable_for_every_box_of_the_brief(
    ctx, monkeypatch, tmp_path, box, shortest_walk
):
    """Flatbot's own kinematics, the pink canister's lid at 0.3125: the
    press is proven at 30 deg; from the hold the way back to the look pose
    — the rise and the way down backwards — is one the runner flies; the
    look pose sees the button at the frame's centre ~0.30 m off the lens,
    where the straight-down depth look could not stand (no turned-down hand
    over the far boxes rises the 56-69 mm it needs above the stroke's
    start); the grip comes down to the lid; the lift, the clear, the carry
    (IK standing in for the planner), the set-down and the retreat are
    lines the runner flies; and a spot beside the box is one the hand
    reaches straight down. All of it as well from the posture the look
    takes for the quickest walks where one is quicker (rp.quickest_look,
    run_reach_press's proof; at one of the five): every leg after the look
    starts from it."""
    from box_opening.runtime.runner import Runner
    from box_opening.tasks import press_demo

    chain, cfg, home = trp._on_flatbot(ctx, monkeypatch)
    button = (box[0], box[1], LID)
    ctx.cpose = rp.cpose_for_button(button, 0.0, ctx.model.button_offset)
    ctx.worlds.table_top_z = LID - 0.10  # the scene camera's table (press_demo.adopt_scene_table)
    if shortest_walk:
        monkeypatch.setattr(rp, "LOOK_POSTURE_BUDGET_S", math.inf)  # every proof, however loaded the machine
    plan, why = rp.plan_reach(ctx, cfg, button, home, depth=True, shortest_walk=shortest_walk)
    assert why is None and plan.tilt_deg == 30
    assert plan.look_leg.goal_joints == pytest.approx(plan.look_joints, abs=1e-9)
    if shortest_walk:
        # over the transit and the one walk a run flies since #109 (rp.LOOK_POSTURE_WALKS), one of the five takes
        # another elbow (over three walks two did, 3.1 and 1.8 s quicker: (0.60, -0.49) was the other)
        assert (plan.posture["chosen"] > 0) is (box == (0.668, -0.33)), plan.posture
    way, why, _kind = rp.solve_way_in(ctx, cfg, button, plan.tilt_deg, plan.look_leg.goal_joints)
    assert way is not None, why
    runner = Runner(ctx.client, log_dir=tmp_path)
    q_held = _held_at_bottom(chain, way.stroke, LID, cfg)
    back, kind, why = rlo.way_back_legs(ctx, way, q_held)
    assert kind is None, why
    assert [(lg.name, runner._refusal(lg)) for lg in back] == [(lg.name, None) for lg in back]
    assert back[-1].goal_joints == pytest.approx(rlo._look_joints(way))
    assert len(rlo.merge_groups(back)) == 1  # one execution
    # the look: the button at the frame's centre, ~0.30 m off the lens (the wrist's OAK: K of 10-27's frames)
    R, t = chain.fk(rlo._look_joints(way))
    lens = np.asarray(t) + np.asarray(R) @ np.array([0.0, 0.065, -0.105])
    c = np.asarray(R).T @ (np.asarray(button) - lens)
    assert 0.27 < float(np.linalg.norm(c)) < 0.33 and abs(c[0] / c[2]) < 0.1 and abs(c[1] / c[2]) < 0.1
    if math.hypot(*box) > 0.70:
        # no straight-down depth look at 245 mm from the lid: the hand at the stroke's start rises too little
        R0, t0 = chain.fk(way.stroke.q0)
        lens0 = np.asarray(t0) + np.asarray(R0) @ np.array([0.0, 0.065, -0.105])
        need = 0.245 - (float(lens0[2]) - LID)
        up, _why = chain.straight_line(way.stroke.q0, [t0[0], t0[1], t0[2] + need], way.stroke.quat)
        assert up is None
    # the grip straight from the press: up off the hold onto the stroke's line and down it, lines the runner flies
    rise, down, why = rlo.grip_from_hold_lines(ctx, way, q_held)
    assert why is None, why
    lines = [
        dp.compliant_line(ctx, "grip:rise", rise, way.stroke.quat, 0.55, rlo._full_world(ctx)),
        dp.compliant_line(ctx, "grip:down", down, way.stroke.quat, dp.TOUCH_SPEED, rlo._full_world(ctx)),
    ]
    assert [(lg.name, runner._refusal(lg)) for lg in lines] == [(lg.name, None) for lg in lines]
    assert dp._pads_z(chain, down[-1]) == pytest.approx(
        dp._pads_z(chain, q_held) + rlo.GRIP_LID_ABOVE_HELD_M - rlo.GRIP_BELOW_LID_M, abs=5e-4
    )
    assert dp._pads_z(chain, rise[-1]) == pytest.approx(dp._pads_z(chain, q_held) + rlo.GRIP_RISE_M, abs=5e-4)
    # the grip, from the look pose the way back ends at
    lid_z = rlo.grip_lid_z(LID, dp._pads_z(chain, q_held))
    grip, why = rlo.solve_grip_way(ctx, cfg, button, plan.tilt_deg, back[-1].goal_joints, lid_z)
    assert grip is not None, why
    legs = rp.way_in_legs(ctx, cfg, grip)
    assert [(lg.name, runner._refusal(lg)) for lg in legs] == [(lg.name, None) for lg in legs]
    assert dp._pads_z(chain, legs[-1].goal_joints) == pytest.approx(lid_z - rlo.GRIP_BELOW_LID_M, abs=5e-4)
    # the spot: the first beside the box the hand reaches straight down
    spots = press_demo.lid_drop_candidates(ctx.model, ctx.cpose, LID - 0.10)
    spots = [ContainerPose(xyz=tuple(xyz), yaw=0.0) for xyz, _d in spots]
    reached = [d for d in spots if rlo.drop_reachable(ctx, d, home) is None]
    assert reached, [rlo.drop_reachable(ctx, d, home) for d in spots]

    monkeypatch.setattr(rp, "_plan_motion", _plan_by_ik(chain, home))
    lift, why = rlo.plan_lift(ctx, cfg, legs[-1].goal_joints, grip.stroke.quat)
    assert lift is not None, why
    assert [lg.name for lg in lift.legs] == ["lid:lift", "lid:clear"]
    q_look = lift.legs[-1].goal_joints
    # where the lid goes, from the spot look: the first the hand reaches from there
    drop, tier, _says = next(rlo.spot_candidates(ctx, _bare(monkeypatch, ctx), LID - 0.10, q_look, seeds=(home,)))
    assert tier == "beside the box" and type(drop) is ContainerPose  # straight down: no lean was judged
    carry, why = rlo.plan_carry(ctx, cfg, q_look, drop)
    assert carry is not None, why
    # the way down to SETDOWN_SLOW_M over where the plan has the lid meet the table, the touch on down to
    # SETDOWN_PAST_M under it (cut where the lid meets the table: watch_setdown)
    _quat, contact, _hover = rlo.setdown_poses(ctx, drop)
    _R, t_down = chain.fk(carry.legs[1].goal_joints)
    _R, t_touch = chain.fk(carry.legs[2].goal_joints)
    assert t_down[2] == pytest.approx(contact[2] + rlo.SETDOWN_SLOW_M, abs=5e-4)
    assert t_touch[2] == pytest.approx(contact[2] - rlo.SETDOWN_PAST_M, abs=5e-4)
    assert carry.over_m == rlo.SETDOWN_PAST_M and carry.k_z > 0.0
    home_legs, why = rlo.plan_home_after(ctx, cfg, carry)
    assert home_legs is not None, why
    names = [lg.name for lg in carry.legs + home_legs]
    assert names[:4] == ["lid:carry", "place:down", "place:touch", "place:retreat"] and names[-1] == "home"
    plan_free = [lg for lg in lift.legs + carry.legs + home_legs if lg.name not in ("lid:carry", "home")]
    assert [(lg.name, runner._refusal(lg)) for lg in plan_free] == [(lg.name, None) for lg in plan_free]
    # the close flies on to the set-down: the lift, the clear, the carry and the set-down are one execution,
    # re-timed as a compliant run flies it, that the driver's gates pass
    from box_opening.constants import JOINT_VMAX
    from box_opening.runtime.legs import merge_groups
    from box_opening.runtime.retime import check_like_executor, retime_group

    one = [*lift.legs, *carry.legs]
    assert [[lg.name for lg in g] for g in merge_groups(one)] == [[lg.name for lg in one]]
    ctx.client.compliant = True
    flown, _info = retime_group([lg.traj for lg in one], [lg.speed for lg in one], JOINT_VMAX, runner.retime_params())
    assert check_like_executor(flown, JOINT_VMAX) == []
    # the lid lifted straight up off the rim, then up and back toward the arm, clear of the box's padded volume
    lift_leg, clear = lift.legs
    _R, t_grip = chain.fk(legs[-1].goal_joints)
    _R, t_lift = chain.fk(lift_leg.goal_joints)
    assert t_lift[2] - t_grip[2] == pytest.approx(rlo.lift_off_m(ctx.model), abs=1e-4)
    assert lift.lift_m == pytest.approx(rlo.lift_off_m(ctx.model))  # the full lift, never the shorter one
    assert t_lift[:2] == pytest.approx(t_grip[:2], abs=1e-4)
    _R, t_clear = chain.fk(clear.goal_joints)
    from box_opening.worlds import planned_top_z

    assert t_clear[2] >= planned_top_z(ctx.model, ctx.cpose) + rp.HOME_CLEAR_M
    assert math.hypot(*t_clear[:2]) < math.hypot(*t_lift[:2])  # back toward the arm


@pytest.mark.parametrize("box", FLATBOT_BOXES, ids=["%.3f,%.2f" % b for b in FLATBOT_BOXES])
def test_on_flatbot_the_free_air_lines_fly_faster_under_the_wrists_ceiling_and_stop_nowhere_new(
    ctx, monkeypatch, tmp_path, box
):
    """2026-10-28 16:23: the walks between the tilted look pose and the hand
    turned down over the button — the way in, the way back, the grip's way
    in and its back out — were 15.9 s of the run's 29.4 at FREE_LINE_SPEED.
    A compliant run flies the same lines at COMPLIANT_FREE_LINE_SPEED,
    re-timed as the runner flies them (Runner.retime_params). Flatbot's own
    kinematics, every box of the brief: each execution carrying one — the
    way in, the way back, the grip in, the back out, the lift and the clear,
    the retreat and home (the retreat alone where home first steps a joint
    off its limit, guarded) — timed as the runner flies it, against the
    same lines at FREE_LINE_SPEED under compliant-timing's profile (no
    junction ceiling): it is shorter and stops where it stopped there, no
    more, no less; it meets its legs no faster (2 %: the profile's rounding
    sees the faster samples beyond the junction's three) and asks the
    driver's quintic for no more acceleration than 10 % over what it asked,
    or 10 rad/s^2; the press stroke is entered no faster than before; in no
    interval does a wrist joint run past COMPLIANT_QD_CEILING, nor any other
    past 0.9 x its execution's fastest leg's speed x its limit (0.69 rad/s
    on the walks); and the runner's and the driver's gates pass every one."""
    from dataclasses import replace

    from jtc_model import peak_accel

    from box_opening.constants import (
        COMPLIANT_FREE_LINE_SPEED,
        COMPLIANT_JUNCTION_SPEED,
        COMPLIANT_QD_CEILING,
        FREE_LINE_SPEED,
        JOINT_VMAX,
    )
    from box_opening.primitives.core import _full_world
    from box_opening.runtime.legs import merge_groups
    from box_opening.runtime.retime import check_like_executor, retime_group
    from box_opening.runtime.runner import Runner, line_profile
    from box_opening.runtime.stamps import secs

    def as_flown(group, p, slow=False):
        # what Runner._run_motion sends for the execution `group` (one profile over it, through the fold its
        # legs mark, Leg.fold; a lone free-air line its own) and retime_group's report of it; its free-air lines
        # at FREE_LINE_SPEED where `slow`
        group = [
            replace(lg, speed=FREE_LINE_SPEED) if slow and lg.speed == COMPLIANT_FREE_LINE_SPEED else lg for lg in group
        ]
        if len(group) > 1:
            return retime_group(
                [lg.traj for lg in group], [lg.speed for lg in group], JOINT_VMAX, p, [lg.fold for lg in group]
            )
        traj = line_profile(group[0], p)
        return traj, {"duration_s": secs(traj.points[-1].time_from_start), "junction_speeds": [], "stops": 0}

    chain, cfg, home = trp._on_flatbot(ctx, monkeypatch)
    ctx.client.compliant = True  # as run_reach_press sets it: every leg below is a compliant run's
    button = (box[0], box[1], LID)
    ctx.cpose = rp.cpose_for_button(button, 0.0, ctx.model.button_offset)
    plan, why = rp.plan_reach(ctx, cfg, button, home, depth=True)
    assert why is None and plan.tilt_deg == 30
    way, why, _kind = rp.solve_way_in(ctx, cfg, button, plan.tilt_deg, plan.look_leg.goal_joints)
    assert way is not None, why
    q_held = _held_at_bottom(chain, way.stroke, LID, cfg)
    back, kind, why = rlo.way_back_legs(ctx, way, q_held)
    assert kind is None, why
    grip, why = rlo.solve_grip_way(
        ctx, cfg, button, plan.tilt_deg, back[-1].goal_joints, rlo.grip_lid_z(LID, dp._pads_z(chain, q_held))
    )
    assert grip is not None, why
    grip_in = rp.way_in_legs(ctx, cfg, grip)

    monkeypatch.setattr(rp, "_plan_motion", _plan_by_ik(chain, home))
    lift, why = rlo.plan_lift(ctx, cfg, grip_in[-1].goal_joints, grip.stroke.quat)
    assert lift is not None, why
    drop, _tier, _says = next(
        rlo.spot_candidates(ctx, _bare(monkeypatch, ctx), LID - 0.10, lift.legs[-1].goal_joints, seeds=(home,))
    )
    carry, why = rlo.plan_carry(ctx, cfg, lift.legs[-1].goal_joints, drop)
    assert carry is not None, why
    retreat, why = rlo.plan_home_after(ctx, cfg, carry)
    assert retreat is not None, why
    executions = {
        "way in": rp.way_in_legs(ctx, cfg, way),
        "way back": back,
        "grip in": grip_in,
        "back out": rlo._grip_backwards(ctx, grip, _full_world(ctx)),
        "lift": lift.legs,
        "retreat": retreat,
    }
    runner = Runner(ctx.client, log_dir=tmp_path)
    params = runner.retime_params()
    assert params.qd_ceiling == COMPLIANT_QD_CEILING and params.junction_cruise == COMPLIANT_JUNCTION_SPEED
    wrist = np.asarray(COMPLIANT_QD_CEILING[4:], float)
    timed = []
    for what, legs in executions.items():
        assert any(lg.speed == COMPLIANT_FREE_LINE_SPEED for lg in legs), (what, [lg.speed for lg in legs])
        plan_free = [lg for lg in legs if lg.plan_free]
        assert [(lg.name, runner._refusal(lg)) for lg in plan_free] == [(lg.name, None) for lg in plan_free], what
        timed += [(what, g) for g in merge_groups(legs) if any(lg.speed == COMPLIANT_FREE_LINE_SPEED for lg in g)]
    assert {what for what, _g in timed} == set(executions)
    for what, legs in timed:
        flown, info = as_flown(legs, params)
        was, before = as_flown(legs, replace(params, junction_cruise=None), slow=True)
        assert info["duration_s"] < before["duration_s"] and info["stops"] == before["stops"], (what, info, before)
        assert all(
            v <= 1.02 * b + 1e-9 for v, b in zip(info["junction_speeds"], before["junction_speeds"], strict=False)
        ), (what, info["junction_speeds"], before["junction_speeds"])
        assert peak_accel(flown)[0] <= max(1.1 * peak_accel(was)[0], 10.0), (what, peak_accel(flown), peak_accel(was))
        assert check_like_executor(flown, JOINT_VMAX) == [], what
        q = np.asarray([p.positions for p in flown.points])
        t = np.asarray([secs(p.time_from_start) for p in flown.points])
        # each interval's joint speed, what the ceilings bound (the stamps whole nanoseconds: 4e-8 of a 25 ms one);
        # the driver's quintic between two samples can run past it where the path turns, as it did at 0.3
        qd = np.abs(np.diff(q, axis=0)) / np.diff(t)[:, None]
        assert (qd[:, 4:] <= wrist * (1 + 1e-6)).all(), (what, qd[:, 4:].max(axis=0))
        arm = 0.9 * max(lg.speed for lg in legs) * np.asarray(JOINT_VMAX[:4], float)  # the planned home's: 1.0
        assert (qd[:, :4] <= arm * (1 + 1e-6)).all(), (what, qd[:, :4].max(axis=0))
        if what == "way in":
            k = next(i for i, lg in enumerate(legs) if lg.name in ("depth:approach", "depth:press"))
            assert info["junction_speeds"][k - 1] <= before["junction_speeds"][k - 1] + 1e-9, (info, before)


def test_the_lid_comes_down_faster_touches_the_table_gently_and_the_hand_turns_at_the_wrists_pace():
    """Owner 2026-10-29: "we also need to speed up the lid placement". The
    set-down's way down is free air and flies at the free-air speed (it was
    setdown_speed, 0.35), its touch at PLACE_TOUCH_SPEED, the press's own
    again (twice it set the lid down too hard, owner 2026-10-29 14:45);
    the hand's own turn (arm_joint_7), measured on 12:02's carry following
    0.45 rad/s 22 mrad behind, is capped at 0.9, the other wrist joints at
    0.45 as before."""
    from box_opening.constants import COMPLIANT_FREE_LINE_SPEED, COMPLIANT_QD_CEILING

    assert rlo.PLACE_TOUCH_SPEED == pytest.approx(dp.TOUCH_SPEED)
    assert COMPLIANT_FREE_LINE_SPEED > load_press_demo(PINK).setdown_speed
    assert COMPLIANT_QD_CEILING[4:] == (0.45, 0.45, 0.9)


def test_the_fingers_come_down_to_the_knob_opened_to_its_width_not_fully(ctx, monkeypatch, tmp_path):
    """2026-10-29 14:31:59: fully open (~85 mm), the fingertips came down
    46-49 mm from the knob's axis, on the rim of a lid only 52.5 mm in
    radius, and closed on it (0.147, "on the box"). The grip's way down
    opens them to the knob's width and GRIP_SIDE_CLEAR_M either side (64 mm
    for the pink canister's 44 mm knob: 32 mm off its axis, 20 mm inside the
    lid's edge); the release at the table still opens them fully."""
    from box_opening.constants import GRIPPER_CMD_CLOSED, GRIPPER_CMD_OPEN

    code, rec, _chain, _rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)])
    assert code == 0
    m = ctx.model
    # 2026-11-12 19:12:49: open to the knob and 10 mm either side (64 mm), a finger came down at the edge of
    # a knob the press had aimed 7.5 mm off and the box was shut again. The pads now stand midway between
    # the knob's edge and the lid's: 74.5 mm for the 44 mm knob on the 105 mm lid
    width = 0.5 * (m.button_diameter_m + m.lid_dims[0])
    assert width == pytest.approx(0.0745) and width > m.button_diameter_m + 2 * rlo.GRIP_SIDE_CLEAR_M
    assert rec.legs["grip:open"].gripper_cmd == pytest.approx(rlo.grip_open_cmd(m))
    assert GRIPPER_CMD_OPEN < rlo.grip_open_cmd(m) < GRIPPER_CMD_CLOSED
    assert rlo.grip_open_cmd(m) == pytest.approx(GRIPPER_CMD_CLOSED * (1 - width / rlo.GRIPPER_OPEN_WIDTH_M))
    # the fingertips' place across the knob: as far inside the lid's edge as off the knob's, more than its aim was off
    lid_r = 0.5 * m.lid_dims[0]
    assert lid_r - width / 2 == pytest.approx(width / 2 - m.button_diameter_m / 2) and lid_r - width / 2 >= 0.015
    # a lid no wider than its knob and the clearance: the knob's width and GRIP_SIDE_CLEAR_M either side, as before
    small = SimpleNamespace(button_diameter_m=0.044, lid_dims=(0.050, 0.050, 0.02))
    assert rlo.grip_open_cmd(small) == pytest.approx(GRIPPER_CMD_CLOSED * (1 - 0.064 / rlo.GRIPPER_OPEN_WIDTH_M))
    assert rec.legs["place:open"].gripper_cmd == GRIPPER_CMD_OPEN  # the lid let go fully at the table


# ---- the grip's close-up record ----------------------------------------------------------


@pytest.mark.parametrize(
    "name, want",
    [
        ("wrist_grip_close_flatbot_20261030_1008_held.jpg", (345.5, 395.5, 48.5)),
        ("wrist_grip_close_flatbot_20261030_1059_air.jpg", (287.5, 389.5, 62.2)),
    ],
    ids=["10:08 held", "10:58 on air"],
)
def test_the_knobs_circle_is_found_in_the_grips_close_up(name, want):
    """2026-10-30's close-ups, the fingers open round the knob once the arm
    stood still (rebuilt at the wrist's 640 x 400 from the lid videos, the
    JPEG's): the knob's circle at the frame's bottom edge — 10:08's, held,
    centred at the edge; 10:58's second, on air, ~58 px to its left. A
    circle mostly past the edge is found to a few px (the held one's centre
    moved 6 px and its radius 7 between the video's frame and this JPEG):
    a record, not yet a correction."""
    import cv2

    img = cv2.imread(str(fixture_path(name)))
    got = rlo.grip_knob_circle(img)
    assert got is not None
    assert got[0] == pytest.approx(want[0], abs=3) and got[1] == pytest.approx(want[1], abs=3)
    assert got[2] == pytest.approx(want[2], abs=3)
    assert rlo.grip_knob_circle(np.full_like(img, 128)) is None  # no knob, no circle


def test_the_grips_close_up_is_kept_with_where_the_knob_sits_and_whether_it_held(monkeypatch):
    """Taken once the arm stands still before the close (Leg.when_still):
    the frame kept on a thread, the knob's circle measured on another, its
    place in the camera from its size (the button's 44 mm), noted with the
    grip's result."""
    import cv2

    from box_opening.tasks import press_demo

    img = cv2.imread(str(fixture_path("wrist_grip_close_flatbot_20261030_1059_air.jpg")))
    kept = []
    monkeypatch.setattr(press_demo, "_save_wrist_frame", lambda g, folder, caption, **kw: kept.append(folder) or folder)
    K = np.array([[398.4, 0.0, 338.0], [0.0, 398.3, 203.2], [0.0, 0.0, 1.0]])
    watcher = SimpleNamespace(grab=SimpleNamespace(color=img, k=K))
    shot = rlo._GripShot(watcher, SimpleNamespace(button_diameter_m=0.044), 2)
    notes = []
    runner = SimpleNamespace(note=lambda leg, **kw: notes.append((leg, kw)))
    shot.note(runner, 2, False, "grip 0.769 vs band [0.3, 0.55]")
    ((leg, row),) = notes
    assert leg == "grip_close_look" and row["attempt"] == 2 and row["held"] is False
    assert len(kept) == 1 and "wrist-grip-close-" in str(kept[0]) and str(kept[0]).endswith("-a2")
    u, v, r = row["knob_px"]
    z = 398.4 * 0.022 / r
    assert row["knob_cam_mm"] == pytest.approx(
        [1000 * z * (u - 338.0) / 398.4, 1000 * z * (v - 203.2) / 398.3, 1000 * z], abs=0.2
    )
    # no wrist frame: noted all the same, nothing kept
    notes.clear()
    rlo._GripShot(SimpleNamespace(), SimpleNamespace(button_diameter_m=0.044), 1).note(runner, 1, True, "grip 0.41")
    assert notes == [
        (
            "grip_close_look",
            dict(
                attempt=1,
                held=True,
                detail="grip 0.41",
                frame=None,
                knob_px=None,
                knob_cam_mm=None,
                why="no wrist frame",
            ),
        )
    ]


def test_each_grip_close_is_recorded_once_the_arm_stands_still(ctx, monkeypatch, tmp_path):
    """The full run's close rides the runner's wait for the arm to stand
    still (Leg.when_still), and the record says which attempt and whether
    it held: here a close on air, then the grip once more that held."""
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0), (REGRIP, 9.0)], knuckle=[AIR, KNOB_ON]
    )
    assert code == 0 and rec.legs["grip:close"].when_still is not None
    looks = [(r["attempt"], r["held"], r["why"]) for r in rows if r["leg"] == "grip_close_look"]
    assert looks == [(1, False, "no wrist frame"), (2, True, "no wrist frame")]


# ---- the grip straight from the press ---------------------------------------------------

FROM_PRESS = ["grip:rise", "grip:open"]  # the open sent under the rise once the pads leave the knob


def test_the_grip_straight_from_the_press_takes_the_lid_off_with_no_way_back_and_no_look(
    ctx, monkeypatch, tmp_path, capsys
):
    """Owner 2026-10-30: after the first press's hold the hand rises
    GRIP_RISE_M, the fingers opening to the knob's width as it does once the
    pads have left the knob (owner 2026-11-02: not on it), comes
    down the stroke's line to the grip's height — GRIP_LID_ABOVE_HELD_M
    over where the hold held the pads, less GRIP_BELOW_LID_M — and closes:
    no way back to the look pose, no look, no way down from it (10:08's
    were 2.6 + 0.7 + 3.2 s). The lift, the spot, its carry and home are
    planned while the hand rises; the fingers are open before it comes
    down; the rest as from the look pose."""
    from box_opening.tasks.press_contact import PadsAbove

    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], from_press=True)
    assert code == 0
    assert rec.events == [
        "reach:look",
        "aim:reach-aim",
        *WAY_IN,
        "hold",
        *FROM_PRESS,
        "grip:down",
        *CLOSE,
        *CARRY,
        *HOME,
    ]
    runs = [names for names, _g in _executions(rec)]
    assert runs == [["reach:look"], WAY_IN, FROM_PRESS, ["grip:down"], CLOSE + CARRY, RELEASED, HOMED]
    planned = {name: at for name, at in rec.plans}
    # the lookahead split over the rise and the way down (grip_ahead_rise, grip_ahead_down): the lift and
    # the first spot while the hand rose, the carry and home after it while it came down
    assert planned["lid:carry"] == planned["home"] == rec.events.index("grip:down") + 1
    (spot,) = (r for r in rows if r["leg"] == "lid_spot")
    assert spot["chosen"] == "ahead, with the scene camera" and spot["spot_checks"] >= 1
    assert rec.legs["grip:open"].defer_join and rec.legs["grip:open"].gripper_cmd == pytest.approx(
        rlo.grip_open_cmd(ctx.model)
    )
    held = BUTTON[2] - 0.006  # the fake hold's pads
    opened = rec.legs["grip:open"]
    assert opened.send_with_previous_motion and isinstance(opened.send_when, PadsAbove)
    assert opened.send_when.z == pytest.approx(held + rlo.GRIP_OPEN_ABOVE_HELD_M)
    # the fake client keeps no feedback: the gate never said so, and the open went out at the rise's top
    (open_row,) = (r for r in rows if r["leg"] == "grip_open")
    assert open_row["under_rise"] is False and open_row["above_held_mm"] == pytest.approx(37.5)
    # the way down is watched for where the open fingertips meet the lid, and stops there (past 0)
    from box_opening.tasks.press_contact import PressContact

    watch = rec.legs["grip:down"].stop_when
    assert isinstance(watch, PressContact) and watch.cut is True and watch.past_m == 0.0 and watch.contact is None
    assert not [r for r in rows if r["leg"] == "grip_touch"]  # nothing met: the way down ended where it always did
    (grip,) = (r for r in rows if r["leg"] == "reach_grip")
    assert (grip["attempt"], grip["from_press"]) == (0, True)
    assert grip["lid_z"] == pytest.approx(held + rlo.GRIP_LID_ABOVE_HELD_M)
    assert grip["pads_z"] == pytest.approx(held + rlo.GRIP_LID_ABOVE_HELD_M - rlo.GRIP_BELOW_LID_M)
    assert not [r for r in rows if r["leg"] in ("reach_knob", "reach-after")]  # no look
    (result,) = (r for r in rows if r["leg"] == "reach_grip_result")
    assert result["ok"] is True and result["attempt"] == 0
    out = capsys.readouterr().out
    assert "GRIP STRAIGHT FROM THE PRESS" in out and out.rstrip().endswith("arm home")


def test_the_grip_from_the_press_is_up_and_down_the_strokes_line(ctx, monkeypatch):
    """grip_from_hold_lines: up GRIP_RISE_M from the hold onto the stroke's
    line, under the exit's looser gate (EXIT_MAX_STEP_RAD), then down it to
    the pads GRIP_LID_ABOVE_HELD_M - GRIP_BELOW_LID_M over where they were
    held under the line's own (MAX_STEP_RAD), the stroke's attitude."""
    from box_opening.runtime.kinematics import EXIT_MAX_STEP_RAD

    asked, gates = [], []

    class Chain:
        def fk(self, q):
            return np.eye(3), np.array([0.6 + q[0], -0.2 + q[1], 0.30 + q[2]])

        def straight_line(self, q0, xyz, quat, max_step_rad=None):
            asked.append((list(xyz), list(quat)))
            gates.append(max_step_rad)
            return [list(q0), [xyz[0] - 0.6, xyz[1] + 0.2, xyz[2] - 0.30] + [0.0] * 4], None

    from box_opening.runtime import approach

    monkeypatch.setattr(approach, "_chain", lambda: Chain())
    monkeypatch.setattr(dp, "line_refusal", lambda chain, pts: None)
    way = SimpleNamespace(stroke=SimpleNamespace(q0=[0.004, -0.003, 0.05, 0, 0, 0, 0], quat=[0, 1, 0, 0]))
    rise, down, why = rlo.grip_from_hold_lines(ctx, way, [0.0, 0.0, 0.0, 0, 0, 0, 0])  # held, 4-3 mm off the line
    assert why is None
    (top, q_top), (bottom, q_bottom) = asked
    assert top == pytest.approx([0.604, -0.203, 0.30 + rlo.GRIP_RISE_M])
    assert bottom == pytest.approx([0.604, -0.203, 0.30 + rlo.GRIP_LID_ABOVE_HELD_M - rlo.GRIP_BELOW_LID_M])
    assert q_top == q_bottom == [0, 1, 0, 0] and rise[0] == [0.0] * 7 and down[0] == rise[-1]
    assert gates == [EXIT_MAX_STEP_RAD, None]  # None: the way down to the knob is asked under MAX_STEP_RAD


def test_the_rise_off_the_hold_clears_a_popped_knob_on_a_risen_lid_and_leans_where_straight_does_not_fly(
    ctx, monkeypatch
):
    """Flatbot 2026-11-12 19:12:49 (the owner: "it just didnt raise high enough before it came back down to
    grab it, reclosing the box"): the rise was 25 mm, the pads got 22.9 mm up, and a popped knob's top stands
    ~25 mm over the hold (the stroke, the released lid's rise, the knob over it). The rise is the stroke, the
    most a released lid rises and a knob's height; where that line does not fly, the same with the hand
    leaning further back by its top and back to the stroke's attitude on the way down, then
    GRIP_RISE_MIN_M each way; where none flies, the first refusal."""
    assert rlo.GRIP_RISE_M == pytest.approx(rlo.KNOB_STROKE_M + rlo.LID_POP_M[1] + rlo.KNOB_STROKE_M)
    assert rlo.GRIP_RISE_M >= 0.040 > rlo.GRIP_RISE_MIN_M > rlo.GRIP_RISE_LEAST_M > 0.025
    assert rlo.GRIP_OPEN_ABOVE_HELD_M > 0.025  # the fingers spread over the popped knob's top, not on it
    lines, walks = [], []

    class Chain:
        def __init__(self, straight):
            self.straight = straight

        def fk(self, q):
            return np.eye(3), np.array([0.6 + q[0], -0.2 + q[1], 0.30 + q[2]])

        def straight_line(self, q0, xyz, quat, max_step_rad=None):
            lines.append(round(xyz[2] - 0.30, 4))
            if not self.straight(xyz[2] - 0.30):
                return None, "the chair and user: arm_half_arm_1_link 0 mm into person/lap"
            return [list(q0), [xyz[0] - 0.6, xyz[1] + 0.2, xyz[2] - 0.30] + [0.0] * 4], None

        def walk(self, q0, xyz, quat, step_m=None, max_step_rad=None):
            walks.append((round(xyz[2] - 0.30, 4), [round(v, 4) for v in quat]))
            return [list(q0), [xyz[0] - 0.6, xyz[1] + 0.2, xyz[2] - 0.30] + [0.0] * 4], None

    from box_opening.geometry import rotate_about_world_axis
    from box_opening.runtime import approach

    monkeypatch.setattr(dp, "line_refusal", lambda chain, pts: None)
    way = SimpleNamespace(stroke=SimpleNamespace(q0=[0.0, 0.0, 0.05, 0, 0, 0, 0], quat=[0.0, 1.0, 0.0, 0.0]))
    held = [0.0] * 7
    low = rlo.GRIP_LID_ABOVE_HELD_M - rlo.GRIP_BELOW_LID_M
    # straight up flies: the full rise, no lean asked
    monkeypatch.setattr(approach, "_chain", lambda: Chain(lambda dz: True))
    rise, down, why = rlo.grip_from_hold_lines(ctx, way, held)
    assert why is None and lines == [pytest.approx(rlo.GRIP_RISE_M), pytest.approx(low)] and walks == []
    # no straight line up: the full rise leaning the least, and back to the stroke's attitude coming down
    lines.clear()
    monkeypatch.setattr(approach, "_chain", lambda: Chain(lambda dz: False))
    rise, down, why = rlo.grip_from_hold_lines(ctx, way, held)
    leaned = rotate_about_world_axis(
        way.stroke.quat,
        (-0.2 / math.hypot(0.6, 0.2), -0.6 / math.hypot(0.6, 0.2), 0.0),
        math.radians(rlo.LIFT_LEAN_DEG[0]),
    )
    assert why is None and lines == [pytest.approx(rlo.GRIP_RISE_M)]
    assert walks == [
        (pytest.approx(rlo.GRIP_RISE_M), [pytest.approx(v, abs=1e-4) for v in leaned]),
        (pytest.approx(low), [0.0, 1.0, 0.0, 0.0]),
    ]
    assert rise[-1][2] == pytest.approx(rlo.GRIP_RISE_M) and down[0] == rise[-1] and down[-1][2] == pytest.approx(low)
    # nothing flies: the first refusal, the full straight rise's
    Chain.walk = lambda self, q0, xyz, quat, step_m=None, max_step_rad=None: (None, "IK did not converge")
    rise, down, why = rlo.grip_from_hold_lines(ctx, way, held)
    assert rise is None and down is None and why.startswith("the rise off the button (the chair and user")


def test_a_leaning_grip_comes_down_further_out_so_the_pads_low_half_takes_the_knob(ctx, monkeypatch):
    """A hand leaning back toward the arm holds its pads' ends on a slant (fingers.pad_corner_below_m): with
    their middle over the knob the corner nearest the arm meets the lid first and the pads take the knob's
    side that much higher — 2.8 mm at 15 deg, 4.6 at 25, 5.5 at 30, of a knob 7.8 mm proud. grip_shift_m puts
    the pads' middle that far out past the knob's centre that their ends over it stand no more than
    GRIP_PAD_OVER_M over the corner, and never nearer their edge than GRIP_PAD_EDGE_M; a hand nearly
    straight down is not shifted. The grip straight from the press ends its way down there, on the stroke's
    line where that does not fly; the grip from the look pose comes down to a point that far further out."""
    from box_opening.runtime import approach, fingers

    half = 0.5 * fingers.PAD_WIDTH_M
    assert fingers.pad_corner_below_m(15.0) == pytest.approx(0.0028, abs=1e-4)
    assert (
        fingers.pad_corner_below_m(30.0) == pytest.approx(0.0055, abs=1e-4) and fingers.pad_corner_below_m(0.0) == 0.0
    )
    model = ContainerModel.load(PINK)
    at = (0.6, -0.2)

    def shift(lean):
        return rlo.grip_shift_m(rp.reach_quat(model, at, lean))

    assert shift(0.0) == 0.0 and shift(10.0) == 0.0
    assert shift(15.0) == pytest.approx(half - rlo.GRIP_PAD_OVER_M / math.sin(math.radians(15.0)), abs=2e-4)
    assert 0.003 < shift(15.0) < shift(20.0) < shift(25.0) < shift(30.0) <= half - rlo.GRIP_PAD_EDGE_M
    assert shift(60.0) == pytest.approx(half - rlo.GRIP_PAD_EDGE_M)
    # the grip straight from the press: the way down ends that far out from the arm, past the stroke's line
    lines = []

    class Chain:
        refuse_out = False

        def fk(self, q):
            return np.eye(3), np.array([at[0] + q[0], at[1] + q[1], 0.30 + q[2]])

        def straight_line(self, q0, xyz, quat, max_step_rad=None):
            lines.append([round(float(v), 5) for v in xyz])
            if self.refuse_out and math.hypot(xyz[0], xyz[1]) > math.hypot(*at) + 1e-4:
                return None, "IK did not converge 80% of the way"
            return [list(q0), [xyz[0] - at[0], xyz[1] - at[1], xyz[2] - 0.30] + [0.0] * 4], None

        def walk(self, q0, xyz, quat, step_m=None, max_step_rad=None):
            return None, "no walk"

    chain = Chain()
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    monkeypatch.setattr(dp, "line_refusal", lambda chain_, pts: None)
    quat = rp.reach_quat(model, at, 25.0)
    way = SimpleNamespace(stroke=SimpleNamespace(q0=[0.0, 0.0, 0.05, 0, 0, 0, 0], quat=quat))
    rise, down, why = rlo.grip_from_hold_lines(ctx, way, [0.0] * 7)
    out = rlo.grip_shift_m(quat)
    d = math.hypot(*at)
    assert why is None and lines[0][:2] == pytest.approx(list(at))  # the rise: straight up the stroke's line
    assert lines[1][:2] == pytest.approx([at[0] * (1 + out / d), at[1] * (1 + out / d)], abs=1e-5)
    assert math.hypot(lines[1][0] - at[0], lines[1][1] - at[1]) == pytest.approx(out, abs=1e-5)
    # where no way down ends out there, the stroke's own line, as a hand straight down comes down
    lines.clear()
    chain.refuse_out = True
    rise, down, why = rlo.grip_from_hold_lines(ctx, way, [0.0] * 7)
    assert why is None and lines[-1][:2] == pytest.approx(list(at)) and len(lines) > 2
    # the grip from the look pose: solved for the knob, then for the point further out, which is the one flown
    asked = []
    leaning = SimpleNamespace(pts=[[0.0] * 7], quat=quat, walked=True, slide=None, short_deg=25.0)
    monkeypatch.setattr(
        rp,
        "reach_from_here",
        lambda ctx_, cfg_, button, tilt, need_stroke=True, q=None: (
            asked.append(tuple(button)),
            ([], SimpleNamespace(slide_pts=[[0.0] * 7], start_xyz=[0, 0, 0], quat=quat), None),
        )[1],
    )
    monkeypatch.setattr(rp, "walk_or_turn", lambda *a, **k: (leaning, None))
    monkeypatch.setattr(rlo, "grip_descent", lambda *a, **k: (SimpleNamespace(legs=[]), None))
    cfg = load_press_demo(PINK)
    got, why = rlo.solve_grip_way(ctx, cfg, (at[0], at[1], 0.36), 30, [0.0] * 7, 0.36)
    assert why is None and len(asked) == 2 and asked[0] == pytest.approx((at[0], at[1], 0.36))
    assert asked[1] == pytest.approx((at[0] * (1 + out / d), at[1] * (1 + out / d), 0.36), abs=1e-6)
    assert got.plan.slide_pts == [[0.0] * 7]


def test_open_fingertips_stand_above_shut_ones_by_the_2f85s_linkage_and_the_touch_rise_follows():
    """fingers.pads_above_shut, from flatbot's URDF: 13.5 mm fully open, 2.3 mm on the 44 mm knob (0.41 rad),
    nothing shut. touch_rise_m: once the open fingertips have met the lid the hand comes back up what closing
    onto the knob brings the pads out, and GRIP_TOUCH_CLEAR_M, so they close round the knob's base."""
    from box_opening.runtime import fingers

    assert fingers.pads_above_shut(0.0) == pytest.approx(0.0135, abs=3e-4)
    assert fingers.pads_above_shut(0.41) == pytest.approx(0.0023, abs=3e-4)
    assert fingers.pads_above_shut(0.8) == 0.0 and fingers.pads_above_shut(1.5) == 0.0
    model = ContainerModel.load(PINK)
    open_h = fingers.pads_above_shut(rlo.grip_open_cmd(model))
    knob_h = fingers.pads_above_shut(rlo._width_cmd(model.button_diameter_m))
    assert 0.008 < open_h < 0.012 and 0.002 < knob_h < 0.004
    assert rlo.touch_rise_m(model) == pytest.approx(open_h - knob_h + rlo.GRIP_TOUCH_CLEAR_M)


def test_fingertips_that_meet_the_lid_on_the_way_down_stop_there_and_close_round_the_knobs_base(
    ctx, monkeypatch, tmp_path, capsys
):
    """The way down to the knob ends where the open fingertips meet something — a released lid stands
    ~10 mm higher than the hold says, and pressed on down it shut the box again (2026-11-12 19:12:49). The
    hand comes back up touch_rise_m (grip:settle) and the fingers close there; the lift planned ahead from
    the way down's planned end, lower, is not flown: the close plans its own from where the hand stands,
    the carry chosen after it."""
    from box_opening.tasks import press_contact

    class Met(press_contact.PressContact):
        def __call__(self, progress):
            if self.final:  # the set-down's own watch: as it is
                return super().__call__(progress)
            self.contact, self.how = (0.4, BUTTON[2] + 0.011, 0.004), "fast"  # the pads' height where they met
            return False  # the fake client flies the leg to its end: the stop itself is the real client's

    monkeypatch.setattr(press_contact, "PressContact", Met)
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], from_press=True)
    assert code == 0
    i = rec.events.index("grip:down")
    assert rec.events[i : i + 5] == ["grip:down", "grip:settle", "grip:close", "lid:lift", "lid:clear"]
    runs = [names for names, _g in _executions(rec)]
    assert ["grip:settle"] in runs and CLOSE in runs and CLOSE + CARRY not in runs  # the lift from where it stands
    (met,) = (r for r in rows if r["leg"] == "grip_touch")
    held = BUTTON[2] - 0.006
    assert met["how"] == "fast" and met["over_held_mm"] == pytest.approx(1000 * (BUTTON[2] + 0.011 - held), abs=0.1)
    assert met["rise_mm"] == pytest.approx(1000 * rlo.touch_rise_m(ctx.model), abs=0.1)
    assert "the open fingertips met the lid" in capsys.readouterr().out


def test_a_hand_that_did_not_rise_clear_of_the_knob_does_not_come_down_on_it(ctx, monkeypatch, tmp_path, capsys):
    """The arm measured its pads less than GRIP_RISE_LEAST_M over the hold at the rise's top: the fingers may
    not have left the popped knob, and the way down would press it shut. No way down: back to the look pose,
    the look to judge the knob, the grip from there."""
    monkeypatch.setattr(rlo, "GRIP_RISE_LEAST_M", 1.0)  # the fake arm never gets there
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], from_press=True)
    (refused,) = (r for r in rows if r["leg"] == "reach_grip_refused" and r.get("from_press"))
    assert refused["why"].startswith("rose ")
    i = rec.events.index("grip:rise")
    assert "grip:down" not in rec.events[i : i + 3]  # the rise, the open, and no way down after them
    assert "not coming down on it" in capsys.readouterr().out
    assert code == 0  # the look saw the knob up and the grip from the look pose took the lid off


def test_a_grip_from_the_press_on_air_goes_back_to_the_look_pose_and_grips_from_there(ctx, monkeypatch, tmp_path):
    """The close straight from the press reads air: nothing lifted; the
    hand goes back up the stroke and down backwards to the look pose (the
    fingers shut on nothing: not opened first), the look sees the knob up,
    and the grip from there (attempt 1) takes the lid off."""
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], knuckle=[AIR, KNOB_ON], from_press=True
    )
    assert code == 0
    assert rec.events == [
        "reach:look",
        "aim:reach-aim",
        *WAY_IN,
        "hold",
        *FROM_PRESS,
        "grip:down",
        *CLOSE,
        *CARRY,
        *BACK,
        "aim:reach-after",
        *GRIP,
        *CLOSE,
        *CARRY,
        *HOME,
    ]
    results = [(r["attempt"], r["ok"], r.get("from_press")) for r in rows if r["leg"] == "reach_grip_result"]
    assert results == [(0, False, True), (1, True, None)]
    looks = [(r["attempt"], r["held"]) for r in rows if r["leg"] == "grip_close_look"]
    assert looks == [(0, False), (1, True)]


def test_a_knob_not_up_after_the_grip_from_the_press_is_pressed_once_more_from_the_look_pose(
    ctx, monkeypatch, tmp_path, capsys
):
    """The knob did not pop: the grip straight from the press closes on
    air, the hand goes back to the look pose, the look reads the knob down
    (twice), and the press once more is as ever: its way back, the look,
    and the grip from the look pose — no second grip from the press."""
    aims = [(BUTTON, 0.4), (BUTTON, 2.0), (BUTTON, 1.5), (UP, 8.0)]
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, aims, knuckle=[AIR, KNOB_ON], from_press=True)
    assert code == 0
    press = [*WAY_IN, "hold"]
    assert rec.events == [
        "reach:look",
        "aim:reach-aim",
        *press,
        *FROM_PRESS,
        "grip:down",
        *CLOSE,
        *CARRY,
        *BACK,
        "aim:reach-after",
        "aim:reach-after",
        *press,
        *BACK,
        "aim:reach-after",
        *GRIP,
        *CLOSE,
        *CARRY,
        *HOME,
    ]
    assert rec.events.count("grip:rise") == 1
    assert "pressing once more — the same press, from the look pose" in capsys.readouterr().out


RIM = 0.147  # the knuckle closed on the box's rim: below the grip band, on something else
CHORD = 0.65  # ... stalled between the band's top and an empty close: on something 4-26 mm thick


def test_the_fingers_are_shut_before_the_press_once_more(ctx, monkeypatch, tmp_path):
    """A grip straight from the press that closed on something else (the rim)
    backs out with the fingers opened; the look then reads the knob down. The
    press once more must not fly with open fingers (an open aperture strikes
    the lid and can still read "pressed"): a press:close at the look pose
    first, which reads shut, then the press, the look and the grip."""
    aims = [(BUTTON, 0.4), (BUTTON, 2.0), (BUTTON, 1.5), (UP, 8.0)]
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, aims, knuckle=[RIM, AIR, KNOB_ON], from_press=True)
    assert code == 0
    first, second = [i for i, e in enumerate(rec.events) if e == "reach:re-aim"][:2]
    between = rec.events[first:second]
    assert between.count("press:close") == 1 and between.index("press:close") > between.index("grip:close")
    assert rec.events.count("press:close") == 1


def test_fingers_that_stay_open_stop_the_press_once_more(ctx, monkeypatch, tmp_path, capsys):
    """The close before the press once more does not read shut (here it
    stops on something): nothing is pressed, home."""
    aims = [(BUTTON, 0.4), (BUTTON, 2.0), (BUTTON, 1.5), (UP, 8.0)]
    code, rec, _chain, _rows = _full_flow(ctx, monkeypatch, tmp_path, aims, knuckle=[RIM, RIM], from_press=True)
    assert code == 1 and rec.events[-1] == "home"
    assert rec.events.count("depth:press") == 1  # the first press only
    assert "the fingers are not shut; nothing was pressed" in capsys.readouterr().out


def test_a_grip_that_stalls_on_something_opens_before_it_backs_out(ctx, monkeypatch, tmp_path):
    """A close that stops above the grip band but short of shut (CHORD) counts
    as on air for the retry, yet something stands between the pads: they open
    before the hand backs up the stroke, or it would carry that back up. A
    close that reached its target (AIR) backs out shut: nothing to carry."""
    code, rec, _chain, _rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], knuckle=[CHORD, KNOB_ON], from_press=True
    )
    assert code == 0
    after = rec.events[rec.events.index("grip:close") :]
    assert after[after.index("grip:close") + 1 : after.index("depth:rise")].count("grip:open") == 1
    code, rec, _chain, _rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], knuckle=[AIR, KNOB_ON], from_press=True
    )
    after = rec.events[rec.events.index("grip:close") :]
    assert code == 0 and "grip:open" not in after[: after.index("depth:rise")]


def test_where_the_grip_from_the_press_does_not_fly_the_way_back_and_the_look_follow(
    ctx, monkeypatch, tmp_path, capsys
):
    """No straight rise off the button, or none down to the knob: nothing
    moves for it; the way back to the look pose, the look and the grip from
    there, as ever."""
    monkeypatch.setattr(
        rlo, "grip_from_hold_lines", lambda ctx_, way, q: (None, None, "the rise off the button (no IK)")
    )
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], from_press=True)
    assert code == 0
    assert rec.events == [
        "reach:look",
        "aim:reach-aim",
        *WAY_IN,
        "hold",
        *BACK,
        "aim:reach-after",
        *GRIP,
        *CLOSE,
        *CARRY,
        *HOME,
    ]
    (refused,) = (r for r in rows if r["leg"] == "reach_grip_refused")
    assert refused["attempt"] == 0 and refused["from_press"] is True
    assert (
        "no grip straight from the press (the rise off the button (no IK)): back to the look pose"
        in capsys.readouterr().out
    )


def test_with_the_grip_from_the_press_the_video_records_from_the_press_on(ctx, monkeypatch, tmp_path):
    """The lid video starts as the first press does, the grip following its
    hold: before the way in, and its end the run's."""
    made, made_rec = _videoed(ctx, monkeypatch)
    real = _Rec

    def rec_kept():
        r = real()
        made_rec[0] = r
        return r

    monkeypatch.setattr(sys.modules[__name__], "_Rec", rec_kept)
    code, rec, _chain, _rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], from_press=True)
    assert code == 0
    (v,) = made
    assert "hold" not in v.started and "reach:re-aim" not in v.started
    steps = [t.split(" ")[0].rstrip(":") for t in v.said if not t.startswith("overlay")]
    assert steps == ["GRIP", "CLOSE", "RELEASE"] and "straight from the press" in v.said[0]
    assert v.finished[0] == "the end: the lid set down, the arm home"


# ---- a high counter (the chair study's 0.86 / 0.91 m side cabinets) ----------------------


class _HandAt(_Chain):
    """The fake chain with the hand standing at height `z` (fk), at `R` (the
    fake's own where none is said), and a walk that flies (straight to its
    end, its asks kept)."""

    def __init__(self, z, refuse=(), unreachable=(), R=None):
        super().__init__(refuse, unreachable)
        self.z = z
        if R is not None:
            self.R = R
        self.walks = []

    def fk(self, q):
        return np.array(self.R, float), np.array([0.5, -0.4, self.z])

    def walk(self, q0, xyz, quat, step_m=None, step_rad=None, max_step_rad=None):
        self.walks.append(dict(xyz=list(xyz), step_m=step_m, step_rad=step_rad, max_step_rad=max_step_rad))
        return [list(q0), [v + 0.01 for v in q0]], None


def _carry_floor(ctx):
    from box_opening.primitives.core import CARRY_CLEAR_M, tcp_z

    m = ctx.model
    return tcp_z(ctx.cpose.xyz[2] + m.dims[2] + m.lid_dims[2] + CARRY_CLEAR_M)


def _grip_z(ctx, above=0.0):
    """The straight-down hand's height (trp.STRAIGHT_DOWN) with the closed pads GRIP_BELOW_LID_M
    under a lid whose top stands `above` over the cameras' top: the knob as the grip holds it."""
    m = ctx.model
    return ctx.cpose.xyz[2] + m.dims[2] + above - rlo.GRIP_BELOW_LID_M + dp.DOWN_CONTACT_TOOL[2]


def test_the_clear_after_the_lift_goes_only_as_high_as_the_carrys_floor_where_todays_does_not_fly(ctx, monkeypatch):
    """The chair study's 0.86 / 0.91 m side counters (study/fixes/highcounter): the clear up to
    clear_targets' height met joint_6's limit in every row whose lift flew and no clear did. There
    the clear goes up to the carry's floor (the held lid's underside CARRY_CLEAR_M over the box,
    lift_clear_targets), back toward the arm as far as clear_targets' first; where clear_targets'
    flies, as on every bench run, it is the one flown, nothing else asked. The box's top under that
    floor is the lid's as the grip took it where that stands higher than the cameras': the study's
    rows that met the lid 45 mm over their top carried it 0.5-5.0 mm over the rim from the clear
    the cameras' top alone put at the lift's top."""
    from box_opening.runtime import approach
    from box_opening.worlds import planned_top_z

    floor = _carry_floor(ctx)
    want = planned_top_z(ctx.model, ctx.cpose) + rp.HOME_CLEAR_M + rp.CLEAR_EXTRA_M  # the fake hand: no lean
    assert want > floor + 0.005
    cfg = load_press_demo(PINK)
    grip = _grip_z(ctx)  # the knob held where the cameras put the lid's top
    assert grip < floor - 0.05
    # the bench: clear_targets' first flies — the lift, then that one line, as before
    chain = _HandAt(grip, R=trp.STRAIGHT_DOWN)
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    lift, why = rlo.plan_lift(ctx, cfg, [0.0] * 7, [0, 0, 0, 1])
    assert why is None and [lg.name for lg in lift.legs] == ["lid:lift", "lid:clear"]
    assert [xyz[2] for xyz, _q in chain.lines] == pytest.approx([grip + rlo.lift_off_m(ctx.model), want])
    assert lift.back_m == pytest.approx(rp.CLEAR_BACK_M) and lift.lift_m == pytest.approx(rlo.lift_off_m(ctx.model))
    # a high counter: no line up to clear_targets' flies (joint_6), the one up to the carry's floor does
    lifted = (0.5, -0.4)
    over = [lambda xyz, q: xyz[2] > floor + 1e-6 and tuple(xyz[:2]) != lifted]
    chain = _HandAt(grip, R=trp.STRAIGHT_DOWN, refuse=over)
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    lift, why = rlo.plan_lift(ctx, cfg, [0.0] * 7, [0, 0, 0, 1])
    assert why is None and lift.back_m == pytest.approx(rp.CLEAR_BACK_M)
    assert [round(xyz[2], 6) for xyz, _q in chain.lines[1:]] == [round(want, 6)] * 2 + [round(floor, 6)]
    # the lid gripped 45 mm over the cameras' top (the study's highest contact): the floor stands
    # over the lid's top as gripped, never the cameras' alone
    todays = [lambda xyz, q: abs(xyz[2] - want) < 1e-6 and tuple(xyz[:2]) != lifted]
    chain = _HandAt(_grip_z(ctx, 0.045), R=trp.STRAIGHT_DOWN, refuse=todays)
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    lift, why = rlo.plan_lift(ctx, cfg, [0.0] * 7, [0, 0, 0, 1])
    assert why is None and lift.back_m == pytest.approx(rp.CLEAR_BACK_M)
    assert [round(xyz[2], 6) for xyz, _q in chain.lines[1:]] == [round(want, 6)] * 2 + [round(floor + 0.045, 6)]
    # and where neither flies, both are said
    chain = _HandAt(grip, R=trp.STRAIGHT_DOWN, refuse=[lambda xyz, q: tuple(xyz[:2]) != lifted])
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    lift, why = rlo.plan_lift(ctx, cfg, [0.0] * 7, [0, 0, 0, 1])
    assert lift is None and "no way up and back" in why and "up to the carry's floor: 50 mm back" in why


def test_a_lift_that_does_not_fly_its_full_height_flies_the_shorter_one_and_says_how_far(ctx, monkeypatch):
    """The chair study's high counters: the 40 mm lift met joint_6's limit 24-40 mm up in 67 of 225
    rows; where it does not fly, the lift goes LIFT_OFF_MIN_CLEAR_M past the cap (35 mm for the
    pink canister's 25 mm cap), and the LiftOff says how far (lift_m: the put-back's rise off the
    knob goes as far). Where neither flies, the full lift's refusal is the one said."""
    from box_opening.runtime import approach

    cfg = load_press_demo(PINK)
    full, short = rlo.lift_off_m(ctx.model), rlo.lift_off_m(ctx.model, rlo.LIFT_OFF_MIN_CLEAR_M)
    assert full > short and full - short == pytest.approx(rlo.LIFT_OFF_CLEAR_M - rlo.LIFT_OFF_MIN_CLEAR_M)
    lifted = (0.5, -0.4)
    chain = _HandAt(0.45, refuse=[lambda xyz, q: tuple(xyz[:2]) == lifted and xyz[2] > 0.45 + short + 1e-6])
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    lift, why = rlo.plan_lift(ctx, cfg, [0.0] * 7, [0, 0, 0, 1])
    assert why is None and lift.lift_m == pytest.approx(short) and rlo._lift_m(ctx, lift) == pytest.approx(short)
    assert [xyz[2] for xyz, _q in chain.lines[:2]] == pytest.approx([0.45 + full, 0.45 + short])
    chain = _HandAt(0.45, refuse=[lambda xyz, q: tuple(xyz[:2]) == lifted])
    chain.walk = lambda q0, xyz, quat, **k: (None, "IK did not converge")  # nor leaning further back
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    lift, why = rlo.plan_lift(ctx, cfg, [0.0] * 7, [0, 0, 0, 1])
    assert lift is None and why.startswith("the lift off the canister is not a straight line")
    assert rlo._lift_m(ctx, rlo.LiftOff([])) == pytest.approx(full)  # a LiftOff that says none: the full lift


def test_a_lift_no_straight_line_flies_goes_up_with_the_hand_leaning_further_back(ctx, monkeypatch):
    """Flatbot 2026-11-12 18:12: the knob held at the arm's reach (the box 0.71 m out, the hand
    leaning 30 deg), the lift ran out 20-28 mm up at that lean and the run backed out of an open
    box. There the lift goes the same height with the hand leaning LIFT_LEAN_DEG further back
    toward the arm by its top — the least lean first, the full lift before the shorter — and the
    clear and all after it hold that attitude (LiftOff.quat). A lift that flies straight asks no
    lean; a lean that would end the lid's low edge at the rim is not flown."""
    from box_opening.geometry import rotate_about_world_axis
    from box_opening.runtime import approach, kinematics

    cfg = load_press_demo(PINK)
    full = rlo.lift_off_m(ctx.model)
    lifted = (0.5, -0.4)
    quat = [0.0, 0.0, 0.0, 1.0]
    chain = _HandAt(0.45)
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    lift, why = rlo.plan_lift(ctx, cfg, [0.0] * 7, quat)
    assert why is None and chain.walks == [] and lift.quat == quat  # straight up flies: no lean asked
    # no line straight up at the hand's own attitude: the walk, leaning the least, to the full height
    asked = []
    chain = _HandAt(0.45, refuse=[lambda xyz, q: tuple(xyz[:2]) == lifted and tuple(q) == tuple(quat)])
    walk = chain.walk
    chain.walk = lambda q0, xyz, q, **k: (asked.append(list(q)), walk(q0, xyz, q, **k))[1]
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    lift, why = rlo.plan_lift(ctx, cfg, [0.0] * 7, quat)
    d = math.hypot(*lifted)
    leaned = list(
        rotate_about_world_axis(quat, (lifted[1] / d, -lifted[0] / d, 0.0), math.radians(rlo.LIFT_LEAN_DEG[0]))
    )
    assert why is None and lift.lift_m == pytest.approx(full) and lift.quat == pytest.approx(leaned)
    assert asked == [pytest.approx(leaned)]
    assert chain.walks == [
        dict(
            xyz=[0.5, -0.4, pytest.approx(0.45 + full)],
            step_m=kinematics.STEP_M,
            step_rad=None,
            max_step_rad=kinematics.MAX_STEP_RAD,
        )
    ]
    assert [leg.name for leg in lift.legs] == ["lid:lift", "lid:clear"]
    assert all(leg.target[2] == pytest.approx(leaned) for leg in lift.legs)  # the clear at the lift's top attitude
    # the lid's low edge: its radius times the lean's sine, of the clearance past the cap
    lid_r = 0.5 * max(ctx.model.lid_dims[0], ctx.model.lid_dims[1])
    assert rlo._lean_keeps_clear(ctx.model, rlo.LIFT_LEAN_DEG[0], rlo.LIFT_OFF_CLEAR_M)
    assert not rlo._lean_keeps_clear(
        ctx.model, math.degrees(math.asin(rlo.LIFT_OFF_MIN_CLEAR_M / lid_r)) + 0.1, rlo.LIFT_OFF_MIN_CLEAR_M
    )


def test_where_no_hand_reaches_a_spot_straight_down_the_same_spots_follow_leaning(ctx, monkeypatch):
    """Every tier straight down first, as the bench's runs chose (a straight-down spot reached is
    offered before any lean is judged); only past them all the same spots again as LeanSpots, the
    hand leaning SETDOWN_LEAN_DEG — the chair study's high counters, where no straight-down hand
    reaches its carry height over any spot. (A table seen bare all round, no room judged on it:
    the spots beside the box and the nearest.)"""
    from box_opening.runtime import approach

    monkeypatch.setattr(approach, "_chain", lambda: _Chain())
    bare = _bare(monkeypatch, ctx)
    asked = []

    def nothing_straight_down(ctx_, d, q, seeds=()):
        asked.append(d)
        return "no straight-down hand reaches its carry height 0.514" if rlo._spot_lean(d) == 0.0 else None

    gen = rlo.spot_candidates(ctx, bare, -0.07, [0.0] * 7, reach=nothing_straight_down)
    drop, tier, says = next(gen)
    assert isinstance(drop, rlo.LeanSpot) and drop.lean_deg == rlo.SETDOWN_LEAN_DEG and "leaning 20 deg" in says
    straight = [d for d in asked if rlo._spot_lean(d) == 0.0]
    assert straight and asked[: len(straight)] == straight and asked[len(straight)] is drop
    assert drop.xyz == straight[0].xyz and drop.yaw == straight[0].yaw and tier in ("beside the box", "nearest")
    assert not rlo._same_spot(drop, straight[0]) and rlo._spot_key(drop) != rlo._spot_key(straight[0])
    # a carry planned ahead to the spot straight down, the one chosen there leaning: the note says why
    ahead = rlo.CarryAhead([0.0] * 7, drop=straight[0], carry=object())
    note = rlo._ahead_note(ahead, 0.0, False, (drop, tier, says), rlo._Reach())
    assert note["why"] == "the spot chosen with the lid lifted is 0 mm from it, the hand leaning otherwise"
    # one straight-down spot reached, the third: it is offered, before any lean is judged
    asked.clear()

    def third_only(ctx_, d, q, seeds=()):
        asked.append(d)
        return None if len(asked) == 3 else "no straight-down hand reaches its carry height 0.514"

    drop, _tier, _says = next(rlo.spot_candidates(ctx, bare, -0.07, [0.0] * 7, reach=third_only))
    assert type(drop) is ContainerPose and len(asked) == 3 and all(rlo._spot_lean(d) == 0.0 for d in asked)


def test_a_leaning_set_down_lands_the_lid_as_gripped_its_centre_over_the_spot_its_lowest_point_on_the_table(
    ctx, monkeypatch
):
    """Flatbot's URDF, the pink canister's lid gripped straight down: setdown_poses for a LeanSpot
    leans the hand SETDOWN_LEAN_DEG back toward the arm, facing the spot; the lid, level as the
    grip took it, has its bottom's centre over the spot and its lowest point on the table where the
    tool meets it, the hover over that at least Place.hover_for's, the carry floor under the lid's
    lowest point. Straight down (lean 0) it is setdown_poses' own straight-down answer."""
    from box_opening.constants import HOME
    from box_opening.perception.d405 import quat_to_mat
    from box_opening.primitives.look import look_joints
    from box_opening.runtime import approach
    from box_opening.runtime.kinematics import ArmChain

    chain = ArmChain(path=flatbot_urdf(), root="arm_mount")
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    ctx.model = ContainerModel.load(PINK)
    m = ctx.model
    box = FLATBOT_BOXES[0]
    ctx.cpose = rp.cpose_for_button((box[0], box[1], LID), 0.0, m.button_offset)
    grip_quat = rp.reach_quat(m, box, 0.0)  # straight down, facing the box
    pads = np.array([box[0], box[1], LID - rlo.GRIP_BELOW_LID_M])
    tool = pads - rp._rot(grip_quat) @ np.asarray(dp.DOWN_CONTACT_TOOL)
    q_grip = chain.nearest_solution(list(HOME), tool, grip_quat, seeds=(look_joints(HOME),))
    assert q_grip is not None
    spot = (box[0] - 0.05, box[1] - 0.20, LID - 0.121)
    lean = rlo.LeanSpot(xyz=spot, yaw=0.3)
    quat, contact, hover = rlo.setdown_poses(ctx, lean, held=q_grip)
    R = np.asarray(quat_to_mat(*quat), float)
    assert math.degrees(rp.hand_lean(R)) == pytest.approx(rlo.SETDOWN_LEAN_DEG, abs=1e-6)
    assert quat == pytest.approx(rp.reach_quat(m, spot[:2], rlo.SETDOWN_LEAN_DEG))
    # the lid in the tool frame as gripped: level, its top GRIP_BELOW_LID_M over the pads
    Rg = np.asarray(chain.fk(q_grip)[0], float)
    top = np.asarray(dp.DOWN_CONTACT_TOOL) + Rg.T @ np.array([0.0, 0.0, rlo.GRIP_BELOW_LID_M])
    n = R @ (Rg.T @ np.array([0.0, 0.0, 1.0]))
    bottom = np.asarray(contact) + R @ top - m.lid_dims[2] * n
    assert bottom[:2] == pytest.approx(spot[:2], abs=1e-9)
    tilt = math.acos(float(n[2]))
    assert math.degrees(tilt) == pytest.approx(rlo.SETDOWN_LEAN_DEG, abs=0.5)  # a level grip: tilted by the lean
    assert bottom[2] - 0.5 * m.lid_dims[0] * math.sin(tilt) == pytest.approx(spot[2], abs=1e-9)  # lowest point
    assert hover[:2] == pytest.approx(contact[:2]) and hover[2] >= contact[2] + m.hover_standoff + m.lid_dims[2] - 1e-9
    # leaning 0, held straight down: the straight-down set-down's own contact and hover
    flat = rlo.LeanSpot(xyz=spot, yaw=0.0, lean_deg=0.0)
    assert rlo.setdown_poses(ctx, flat, held=q_grip) == rlo.setdown_poses(ctx, ContainerPose(xyz=spot, yaw=0.0))
    _q, c0, h0 = rlo._lean_setdown_poses(ctx, flat, q_grip)
    _q, c1, h1 = rlo.setdown_poses(ctx, ContainerPose(xyz=spot, yaw=0.0))
    assert c0 == pytest.approx(c1, abs=1e-5) and h0 == pytest.approx(h1, abs=1e-5)  # the IK's grip to 1e-4 rad
    with pytest.raises(ValueError):
        rlo.setdown_poses(ctx, lean)  # leaning, the lid's grip is needed


def test_a_carry_the_planner_refuses_inside_its_limit_margin_is_the_arms_own_walk(ctx, monkeypatch):
    """The chair study's high counters: the planner refused the carry START_IN_LIMIT_MARGIN or
    GOAL_IN_LIMIT_MARGIN in 58 of the rows whose lid came up (joint_6 0.026-0.085 rad from its
    limit where the clear ends). There the carry flies as the arm's own walk to the same hover
    (walk_carry: CARRY_WALK_STEP_M, CARRY_WALK_STEP_RAD, kinematics.MAX_STEP_RAD), a plan-free
    'lid:carry' at the free-air lines' speed, the set-down's lines after it as ever; any other
    refusal is the carry's refusal, as before."""
    from box_opening.runtime import approach
    from box_opening.runtime import kinematics as kin

    cfg = load_press_demo(PINK)
    chain = _HandAt(0.45)
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    drop = ContainerPose(xyz=(0.40, -0.30, -0.07), yaw=0.0)
    why_planner = ["START_IN_LIMIT_MARGIN: start: arm_joint_6 is 0.043 rad from its position limit"]

    def refuses(ctx_, state, name, target, world, speed):
        raise RuntimeError("planning failed for %s: %s" % (name, why_planner[0]))

    monkeypatch.setattr(rp, "_plan_motion", refuses)
    carry, why = rlo.plan_carry(ctx, cfg, [0.0] * 7, drop)
    assert why is None and [lg.name for lg in carry.legs] == ["lid:carry", "place:down", "place:touch"]
    walk = carry.legs[0]
    assert walk.plan_free and walk.speed == pytest.approx(rp.free_line_speed(ctx)) and walk.gains == dp.COMPLIANT_GAINS
    _quat, _contact, hover = rlo.setdown_poses(ctx, drop)
    assert chain.walks == [
        dict(xyz=hover, step_m=rlo.CARRY_WALK_STEP_M, step_rad=rlo.CARRY_WALK_STEP_RAD, max_step_rad=kin.MAX_STEP_RAD)
    ]
    why_planner[0] = "GOAL_IN_LIMIT_MARGIN: goal: arm_joint_6 is 0.020 rad from its position limit"
    carry, why = rlo.plan_carry(ctx, cfg, [0.0] * 7, drop)
    assert why is None and carry.legs[0].plan_free and len(chain.walks) == 2
    why_planner[0] = "MotionGenStatus.IK_FAIL"
    carry, why = rlo.plan_carry(ctx, cfg, [0.0] * 7, drop)
    assert carry is None and why == "the carry (MotionGenStatus.IK_FAIL)" and len(chain.walks) == 2


def test_a_carry_the_planner_swings_wide_is_the_arms_own_walk_before_anything_is_merged_with_it(ctx, monkeypatch):
    """Flatbot 2026-11-12 18:42: the planner's carry took arm_joint_2 0.444 rad against the 0.444 the
    runner's sanity gate allows. The gate judges every leg of an execution before anything moves, so it
    refused the close, the lift and the set-down merged with that carry, and the run read the refusal as
    a grip that missed: the knob was up and no finger closed on it. The carry is judged by that gate
    where it is planned: one that swings is the arm's own walk to the same hover (walk_carry), and a spot
    neither reaches is refused there, for the next."""
    import copy

    from conftest import leg as make_leg

    from box_opening.primitives.core import PlanState
    from box_opening.runtime import approach
    from box_opening.runtime.guards import sanity_violations

    cfg = load_press_demo(PINK)
    chain = _HandAt(0.45)
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    drop = ContainerPose(xyz=(0.40, -0.30, -0.07), yaw=0.0)

    def swings(ctx_, state, name, target, world, speed):
        start = list(state.joints)
        lg = make_leg(name, start=start, end=[v + 0.05 for v in start], speed=speed, world=world[0])
        out = copy.deepcopy(lg.traj.points[0])
        out.positions = [start[0], start[1] + 0.05 + rlo.SANITY_MARGIN_RAD + 0.01, *start[2:]]
        lg.traj.points.insert(1, out)  # joint 2 out past the gate's allowance and back
        lg.target = target
        return lg, state

    monkeypatch.setattr(rp, "_plan_motion", swings)
    assert sanity_violations(
        swings(ctx, PlanState(joints=[0.0] * 7, chain=0), "lid:carry", None, ("full",), 1.0)[0].traj, 0.35
    )
    carry, why = rlo.plan_carry(ctx, cfg, [0.0] * 7, drop)
    assert why is None and [lg.name for lg in carry.legs] == ["lid:carry", "place:down", "place:touch"]
    assert carry.legs[0].plan_free and len(chain.walks) == 1  # the walk, to the same hover
    assert chain.walks[0]["xyz"] == rlo.setdown_poses(ctx, drop)[2]
    assert not sanity_violations(carry.legs[0].traj, rlo.SANITY_MARGIN_RAD)
    # no walk there either: the spot's carry is refused where it is planned, and says both
    chain.walk = lambda q0, xyz, quat, **k: (None, "IK did not converge 40% of the way")
    carry, why = rlo.plan_carry(ctx, cfg, [0.0] * 7, drop)
    assert carry is None and "swings wide" in why and "walked: IK did not converge" in why


def test_the_way_down_to_a_knob_is_planned_for_the_lid_as_touched_where_that_stands_higher(ctx, monkeypatch):
    """Flatbot 2026-11-12 18:42: the arm's touch put the lid 44 mm over where the wrist camera put the
    button. The grip's way in from the look pose, planned for the camera's height, ended with the pads 22
    mm over the grip's end, too low to come down from ("the hand does not come down to the knob"). The
    way in is planned for the higher of the two; a lid the touch puts at or under the cameras' button
    changes nothing."""
    cfg = load_press_demo(PINK)
    asked = []

    def reach(ctx_, cfg_, button, tilt, need_stroke=True, q=None):
        asked.append(tuple(button))
        return None, None, "stop here"

    monkeypatch.setattr(rp, "reach_from_here", reach)
    assert rlo.solve_grip_way(ctx, cfg, (0.5, -0.4, 0.313), 30.0, [0.0] * 7, 0.357) == (None, "stop here")
    assert rlo.solve_grip_way(ctx, cfg, (0.5, -0.4, 0.313), 30.0, [0.0] * 7, 0.300) == (None, "stop here")
    assert asked == [(0.5, -0.4, 0.357), (0.5, -0.4, 0.313)]


def test_the_scene_cloud_is_moved_by_the_cameras_error_at_the_box_but_not_by_a_rough_fit(ctx):
    """_scene_cloud moves the scene camera's table points by where the arm found the box against
    where the scene put it: the camera's error. A ROUGH fix (a box the frame cut, scene_source.
    SCENE_ROUGH) is off by its own fit, which is not the camera's: its cloud is kept as measured."""
    from box_opening.models.container import ContainerPose

    table = np.array([[0.40, -0.30, TABLE_Z], [0.45, -0.35, TABLE_Z]])
    off = ContainerPose(xyz=(ctx.cpose.xyz[0] - 0.03, ctx.cpose.xyz[1], ctx.cpose.xyz[2]), yaw=ctx.cpose.yaw)
    careful = SimpleNamespace(pose=off, table_points=table, table_z_scene=TABLE_Z)
    pts, tz = rlo._scene_cloud(ctx, careful, None, TABLE_Z)
    assert tz == TABLE_Z and pts[:, 0] == pytest.approx(table[:, 0] + 0.03)
    rough = SimpleNamespace(pose=off, table_points=table, table_z_scene=TABLE_Z, rough=True)
    pts, _tz = rlo._scene_cloud(ctx, rough, None, TABLE_Z)
    assert pts == pytest.approx(table)


# ---- 2026-11-17: the lid is let go only resting on the table (owner: "the arm dropped the lid off the table") ----
# What the runs of 2026-11-13 and 11-17 recorded, replayed (setdown_flown: the numbers, off their run files).


class _Z:
    """A chain whose first joint is the tool frame's height; the hand straight down."""

    def fk(self, q):
        return np.diag([1.0, -1.0, -1.0]), np.array([0.0, 0.0, float(q[0])])


def _run1(monkeypatch, chain=None):
    """2026-11-17 16:40's inputs to the lid's spot: the box where the run had it, the table as the wrist
    survey's fix measured it, the room map it chose on."""
    from box_opening.runtime import approach

    ctx = SimpleNamespace(
        model=ContainerModel.load(PINK),
        cpose=ContainerPose(xyz=(*flown.RUN1_BOX_XY, flown.RUN1_TABLE_Z), yaw=0.0),
        worlds=SimpleNamespace(table_extent=flown.RUN1_EXTENT),
        held_xy=None,
        client=None,
    )
    monkeypatch.setattr(approach, "_chain", lambda: chain or _Chain())
    return ctx, flown.run1_map()


def test_run_1s_spot_in_front_of_the_tables_edge_is_no_longer_a_spot(monkeypatch):
    """Flatbot 2026-11-17 16:40: "lid goes to [0.52, -0.42, 0.205] ... nothing reads clear (nowhere in x
    0.28 to 0.65 ... has room ...): the most room seen, the lid 67 mm short", with the table measured over
    x 0.64..1.15. On that run's own map and extent: that spot is not on the table (nothing was seen under
    it; 12 cm in front of the extent), the search now reaches as far out as the box (x 0.78), and every
    spot offered, whatever its tier, has the whole lid on table the cameras measured."""
    ctx, tmap = _run1(monkeypatch)
    m = ctx.model
    lid_r = 0.5 * m.lid_dims[0]
    old = ContainerPose(xyz=(*flown.RUN1_SPOT, flown.RUN1_TABLE_Z), yaw=0.0)
    assert "past where it may on the table the cameras measured (x 0.64..1.15" in rlo.spot_off_table(ctx, tmap, old)
    ctx.worlds.table_extent = None  # ... and by the map alone
    assert "did not see" in rlo.spot_off_table(ctx, tmap, old)
    ctx.worlds.table_extent = flown.RUN1_EXTENT
    x0, x1, y0, y1 = rlo.spot_zone(m, ctx.cpose, ctx.worlds, tmap)
    assert x0 >= 0.64 + lid_r + 0.03 - 0.02 and x1 == pytest.approx(flown.RUN1_BOX_XY[0])  # it was 0.28..0.65
    passed = rlo.SpotsPassed()
    offered = list(
        rlo.spot_candidates(ctx, tmap, flown.RUN1_TABLE_Z, flown.RUN1_Q_CLEAR, reach=lambda *a: None, passed=passed)
    )
    assert offered and offered[0][1] in ("clear", "most room")
    assert {tier for _d, tier, _s in offered} <= {"clear", "most room", "beside the box", "nearest"}
    for d, _tier, _says in offered:
        assert flown.RUN1_EXTENT[0] + lid_r + 0.03 <= d.xyz[0] <= flown.RUN1_EXTENT[1] - lid_r - 0.03
        assert flown.RUN1_EXTENT[2] + lid_r + 0.03 <= d.xyz[1] <= flown.RUN1_EXTENT[3] - lid_r - 0.03
        assert rlo.spot_off_table(ctx, tmap, d, flown.RUN1_Q_CLEAR) is None
        assert math.hypot(d.xyz[0] - old.xyz[0], d.xyz[1] - old.xyz[1]) > 0.15
    assert passed.off_table > 0  # the spots in front of the edge and on what stands were looked at, and passed over
    # where the hand reaches none of them, none is offered: the caller puts the lid back on the box
    passed = rlo.SpotsPassed()
    none = list(
        rlo.spot_candidates(ctx, tmap, flown.RUN1_TABLE_Z, flown.RUN1_Q_CLEAR, reach=lambda *a: "no IK", passed=passed)
    )
    assert none == [] and passed.unreachable > 0 and "the hand does not reach" in passed.says()


def test_on_flatbots_bare_arm_run_1s_lid_goes_to_a_spot_on_the_measured_table_or_back_on_the_box(monkeypatch, capsys):
    """The same, with flatbot's arm (its URDF, the joints the carry began at): the first spot the hand
    reaches — straight down, else leaning — is on the table the wrist measured, or there is none and
    the lid goes back on the box. Which, and where, is printed for the record.

    The BARE arm: its kinematics and joint limits alone. The chair and its user are not in it (the
    chain has no body model, and approach.body_check is off), nor the counter's keep-out under the
    edges that run's maps showed (two boxes in force then: its counter_edges rows) — the test data
    holds neither flatbot's body layers nor its planner. On the robot both are in force for the hover
    and the line down (kinematics.body_refusal; the same day's 16:43 run was refused by exactly that
    check, "arm_half_arm_1_link 0 mm into person/lap"), so the spot printed here is where the arm's
    own reach puts the lid, not a spot the robot's checks are known to pass: there it may be passed
    over for another, or the lid put back on the box."""
    from box_opening.runtime import approach
    from box_opening.runtime.kinematics import ArmChain

    chain = ArmChain(root="arm_mount", path=flatbot_urdf())
    ctx, tmap = _run1(monkeypatch, chain)
    monkeypatch.setattr(approach, "body_check", lambda: None, raising=False)
    passed = rlo.SpotsPassed()
    got = next(rlo.spot_candidates(ctx, tmap, flown.RUN1_TABLE_Z, flown.RUN1_Q_CLEAR, passed=passed), None)
    with capsys.disabled():
        sys.stdout.write(
            "\nrun 1 replayed on flatbot's BARE arm (its URDF; no chair and user body, no counter keep-out): %s (%s)\n"
            % (
                "no spot the hand reaches on the measured table: the lid goes back on the box"
                if got is None
                else "the lid goes to [%.2f, %.2f] (%s; lean %.0f deg)"
                % (got[0].xyz[0], got[0].xyz[1], got[1], rlo._spot_lean(got[0])),
                passed.says(),
            )
        )
    if got is not None:
        assert rlo.spot_off_table(ctx, tmap, got[0], flown.RUN1_Q_CLEAR) is None
        assert got[0].xyz[0] >= flown.RUN1_EXTENT[0] + 0.5 * ctx.model.lid_dims[0] + 0.03


def test_a_box_inside_the_bench_s_reach_band_is_searched_round_as_before():
    """spot_zone for a box the straight-down hand's band holds (press_demo.DROP_X_M, x 0.28..0.65) on a
    table measured all round: the zone clear_lid_drops searched, the same lattice — so a bench run's
    spots are the spots it had."""
    from box_opening.tasks import press_demo as pd

    m = ContainerModel.load(PINK)
    cpose = ContainerPose(xyz=(0.45, -0.25, 0.0), yaw=0.0)
    bare = _standing_ring((0.45, -0.25), 5.0, 6.0)
    x0, x1, y0, y1 = rlo.spot_zone(m, cpose, SimpleNamespace(table_extent=None), bare)
    reach = pd.lid_place_min_clear(m) + 0.20
    assert (x0, x1) == pytest.approx(pd.DROP_X_M) and y1 == pytest.approx(-0.25 + reach)
    assert ((x0 - pd.DROP_X_M[0]) / 0.02) % 1.0 == pytest.approx(0.0, abs=1e-6)
    assert (((y0 - (-0.25 - pd.CLEAR_SPOTS_Y_M)) / 0.02 + 0.5) % 1.0) - 0.5 == pytest.approx(0.0, abs=1e-6)
    # a box past the band: as far out as the box itself, on the table the fix measured
    far = ContainerPose(xyz=(0.78, -0.13, 0.0), yaw=0.0)
    bare = _standing_ring((0.78, -0.13), 5.0, 6.0)
    zone = rlo.spot_zone(m, far, SimpleNamespace(table_extent=(0.64, 1.15, -0.64, 0.21)), bare)
    assert zone[0] >= 0.64 + 0.0525 + 0.03 - 0.02 and zone[1] == pytest.approx(0.78) and zone[3] <= 0.21 - 0.0825


def test_the_table_the_set_down_goes_by_is_the_cameras_moved_by_where_the_fingers_hold_the_lid(monkeypatch, capsys):
    """2026-11-17 16:40: the wrist's fix had the lid's top at 0.2938 and the table at 0.2046 (89 mm
    under it); the press met the lid 45 mm over the cameras' top, and the knob was gripped with the pads
    at 0.3465. The cameras' heights were ~5 cm low at the box, the table's with them, and the set-down's
    plan stood that far under the table. The table is as far under the lid's top as the cameras saw it,
    the top where the fingers hold it: 0.2593, 55 mm over the cameras'. (11-13 01:52 and 01:53: 46 and
    54 mm.) No grip, a rough fix, or a difference past TOUCH_TABLE_MAX_M: the cameras' table."""
    from box_opening.runtime import approach

    monkeypatch.setattr(approach, "_chain", lambda: _Z())
    grip = [0.3465 + dp.DOWN_CONTACT_TOOL[2]] + [0.0] * 6  # the pads at 0.3465
    fix = SimpleNamespace(top_xyz=(0.756, -0.1165, flown.RUN1_FIX_TOP_Z), table_z_scene=flown.RUN1_TABLE_Z)
    ctx = SimpleNamespace(lid_lifted=None)
    z, moved = rlo.setdown_table_z(ctx, flown.RUN1_TABLE_Z, fix, grip)
    assert z == pytest.approx(0.3465 + rlo.GRIP_BELOW_LID_M - (flown.RUN1_FIX_TOP_Z - flown.RUN1_TABLE_Z))
    assert moved == pytest.approx(0.0547, abs=2e-4) and z == pytest.approx(0.2593, abs=2e-4)
    # the lift's own grip, once the lid is in the fingers
    ctx.lid_lifted = SimpleNamespace(q_grip=grip)
    assert rlo.setdown_table_z(ctx, flown.RUN1_TABLE_Z, fix) == (pytest.approx(z), pytest.approx(moved))
    # nothing to go by: the cameras' own
    ctx.lid_lifted = None
    assert rlo.setdown_table_z(ctx, flown.RUN1_TABLE_Z, fix) == (flown.RUN1_TABLE_Z, None)
    assert rlo.setdown_table_z(ctx, flown.RUN1_TABLE_Z, None, grip) == (flown.RUN1_TABLE_Z, None)
    rough = SimpleNamespace(top_xyz=fix.top_xyz, table_z_scene=fix.table_z_scene, rough=True)
    assert rlo.setdown_table_z(ctx, flown.RUN1_TABLE_Z, rough, grip) == (flown.RUN1_TABLE_Z, None)
    way_off = SimpleNamespace(top_xyz=(0.0, 0.0, flown.RUN1_FIX_TOP_Z - 0.04), table_z_scene=flown.RUN1_TABLE_Z)
    capsys.readouterr()
    assert rlo.setdown_table_z(ctx, flown.RUN1_TABLE_Z, way_off, grip) == (flown.RUN1_TABLE_Z, None)
    assert "95 mm from where the cameras' table and the box's height put it (more than 80)" in capsys.readouterr().out


def _flown(name, watch):
    """One recorded set-down (setdown_flown.SETDOWNS_FLOWN) fed to `watch(chain, client, plan, arm, end)`'s
    watch as the client fed it: the trajectory sent, then each kept sample of the feedback in order.
    Returns (the watch, the fraction it stopped the descent at or None)."""
    rec = flown.SETDOWNS_FLOWN[name]
    traj = SimpleNamespace(
        points=[
            SimpleNamespace(
                positions=[z + dp.DOWN_CONTACT_TOOL[2]] + [0.0] * 6,
                time_from_start=SimpleNamespace(sec=int(t), nanosec=int(round((t - int(t)) * 1e9))),
            )
            for t, z in rec["traj"]
        ]
    )
    info = {"sent_at": 0.0, "sent_traj": traj, "feedback_q": []}
    client = SimpleNamespace(last_execution=info)
    det = watch(_Z(), client, rec["plan"], rec["plan"] + rlo.SETDOWN_WATCH_M, rec["end"])
    stopped = None
    for t, frac, z in rec["fed"]:
        info["feedback_q"].append((t, frac, [z + dp.DOWN_CONTACT_TOOL[2]] + [0.0] * 6, None))
        if det(frac) and stopped is None:
            stopped = frac
    return det, stopped


@pytest.mark.parametrize("name", sorted(flown.SETDOWNS_FLOWN))
def test_the_three_contacts_recorded_as_the_slow_stretch_began_are_not_taken_for_the_table(name):
    """2026-11-13 01:52, 01:53 and 11-17 16:40: the set-down's watch reported "the lid met the table"
    62.3-64.2 mm over the plan, 74-165 ms after it armed, each time — the last with the lid over the
    floor. Their flights replayed (the pads' height, reference and measured): the watch as it was
    (PressContact, armed where the slow stretch begins, judging from its first sample) meets the same
    contact the run recorded, from the arm catching up with its reference as that slows; the set-down's
    own watch (SetdownContact) judges nothing while the arm settles, takes nothing in the record for the
    table, stops nothing, and would not have opened the fingers."""
    from box_opening.tasks.press_contact import PressContact

    old, stopped = _flown(
        name, lambda chain, client, plan, arm, end: PressContact(chain, client, plan, arm, 0.0, end, final=True)
    )
    above_mm, at_s = flown.SETDOWNS_FLOWN[name]["old"]
    assert old.contact is not None and old.how == "fast"
    assert 1000 * (old.contact[1] - old.top_z) == pytest.approx(above_mm, abs=0.3)
    assert old.contact[0] == pytest.approx(at_s, abs=0.005) and 0.07 < old.contact[0] - old.armed_t < 0.17
    new, stopped = _flown(name, rlo.SetdownContact)
    assert new.armed_t == pytest.approx(old.armed_t) and new.samples == old.samples  # the same samples, watched
    assert new.contact is None and new.early == [] and stopped is None and new.f_stop is None
    assert not new.credible() and new.refusal()[0] == "never"
    rec = new.record()
    assert (rec["watched"], rec["met"], rec["credible"], rec["verdict"]) == (True, False, False, "never")


def _descent(
    table=None,
    catch_up=0.0,
    bump=None,
    top=0.20,
    noise=0.0002,
    seed=0,
    under=0.009,
    rate=None,
    delay=0.0,
    polled=True,
    chain=None,
    slow_q=False,
    sinks=0.0,
    firm=None,
    told=None,
):
    """A set-down's last stretch flown over a table: the reference's pads from 12 cm over where the plan
    has the lid meet the table (`top`) to SETDOWN_PAST_M under it, fast (135 mm/s) to where the slow
    stretch begins and at 55 mm/s from there, as flown 2026-11-17; the pads measured `under` it (9 mm), coming
    up on it by `catch_up` (m) over the 0.35 s after the slow stretch begins (the arm catching up) — or, with
    `rate` (m/s), at that pace from `delay` s after it begins until all of `catch_up` has come (the recorded
    flights' own: _flown_lag) —, held up by a table at `table` (the pads' height there; None: none), and by
    `bump`: (the pads' height it begins at, how far it lifts them, for how many mm of the reference).
    `sinks`: what stands at `table` yields — the pads sink that part of the way their free path goes under
    its top (0: rigid, a table; 0.5: a bag, a hand) — down to `firm` (the pads' height it gives no further
    at; None: nowhere). `told` (a dict) is told of the flight: "touch_ref", the reference's pads where the
    measured ones first met what stands at `table`; "free", where the pads would stand at the flight's last
    sample with nothing under them (what stands at `table` is pushed down `sinks` of the way from its top
    to there, the hand's spring deflected by the rest).
    `polled` False: the client keeps the flight's feedback and the watch is not polled while it flies
    (something the execution hosted), then reads it once it is over (read_rest). `chain`: the kinematics
    it measures by (None: test_press_contact's, joint 1 the tool's height); `slow_q` True: the watch is told
    the joints the slow stretch begins at, as watch_setdown tells it. Returns (the watch, the fraction it
    stopped at or None, the reference's pads there)."""
    import test_press_contact as tpc

    arm, end = top + rlo.SETDOWN_SLOW_M, top - rlo.SETDOWN_PAST_M
    fast = np.arange(top + 0.12, arm, -0.135 * 0.01)
    slow = np.arange(arm, end, -0.055 * 0.01)
    z = np.concatenate([fast, slow, [end]])
    t_slow = 0.02 + len(fast) * 0.01
    rng = np.random.default_rng(seed)
    client = SimpleNamespace(last_execution=None)
    det = rlo.SetdownContact(
        chain or tpc._Chain(), client, top, arm, end, slow_q=[arm - tpc.PAD] + [0.0] * 6 if slow_q else None
    )

    def measured(t, zr):
        since = t - t_slow - delay
        came_up = catch_up * min(max(since / 0.35, 0.0), 1.0) if rate is None else min(max(rate * since, 0.0), catch_up)
        free = zr - under + came_up + float(rng.normal(0.0, noise))
        if bump is not None and bump[0] - bump[2] < zr <= bump[0]:
            free += bump[1] * min((bump[0] - zr) / 0.004, 1.0)
        if told is not None:
            told["free"] = free
        if table is None or free >= table:
            return free
        if told is not None:
            told.setdefault("touch_ref", zr)
        sunk = table - sinks * (table - free)
        return sunk if firm is None else max(sunk, firm)

    stopped, info = tpc._fly(det if polled else (lambda frac: False), client, z, 0.01, measured)
    if not polled:
        det.read_rest()
    z_stop = (
        None
        if stopped is None
        else float(np.interp(stopped * (0.02 + (len(z) - 1) * 0.01), 0.02 + np.arange(len(z)) * 0.01, z))
    )
    return det, stopped, z_stop


def _flown_lag(name):
    """How one recorded set-down's pads lagged their reference (setdown_flown.SETDOWNS_FLOWN[name], the
    three flown with the watch): (the most they stood under it, m; how far they had come up on it by the
    cut, m; how fast they were still coming up there, m/s, over the record's last 0.1 s; how long after
    the slow stretch began they stood lowest, s)."""
    rec = flown.SETDOWNS_FLOWN[name]
    tt, tz = (np.array(v, float) for v in zip(*rec["traj"], strict=True))
    fed = np.array(rec["fed"], float)
    gap = fed[:, 2] - np.interp(fed[:, 1] * tt[-1], tt, tz)
    slow = int(np.argmax(np.interp(fed[:, 1] * tt[-1], tt, tz) <= rec["plan"] + rlo.SETDOWN_SLOW_M))
    low = slow + int(np.argmin(gap[slow:]))
    last = fed[:, 0] >= fed[-1, 0] - 0.1
    rate = float(np.polyfit(fed[last, 0], gap[last], 1)[0])
    return -float(gap.min()), float(gap[-1] - gap[low]), rate, float(fed[low, 0] - fed[slow, 0])


def test_the_arm_catching_up_as_the_slow_stretch_begins_is_not_the_table_and_the_table_under_it_is():
    """The pads come up 5 mm on the reference in the 0.35 s after the slow stretch begins (17-22 mm/s:
    2026-11-17's), nothing under the lid: nothing is met, the touch flies to its end 8 mm under the plan,
    and the lid may not be let go. The same with a table where the lid meets it 13 mm over the plan (the
    mean of the 19 set-downs flown to their end): met there, the descent stopped within a few millimetres
    of it, and that contact is the one the fingers open on."""
    det, stopped, _z = _descent(catch_up=0.005)
    assert stopped is None and det.contact is None and det.early == [] and not det.credible()
    assert det.refusal() == (
        "never",
        "the touch ended 8 mm under where the plan has the lid meet the table and the arm felt no table",
    )
    det, stopped, z_stop = _descent(table=0.20 + 0.013, catch_up=0.005)
    assert stopped is not None and det.credible() and det.verdict is None and det.refusal() is None
    assert det.contact[1] == pytest.approx(0.213, abs=0.002) and det.how == "fast"
    assert 0.213 - 0.009 - 0.012 < z_stop < 0.213 + 0.009  # the reference stopped as the rise was seen
    rec = det.record()
    assert rec["credible"] and rec["met"] and rec["cut"] and rec["above_top_mm"] == pytest.approx(13.0, abs=2.0)
    assert (rec["settle_s"], rec["credible_above_mm"], rec["held_mm"], rec["early"]) == (0.6, 35.0, 15.0, [])
    assert (rec["stall_mm"], rec["free_mm"]) == ([12.0, 2.0], [10.0, 1.5])
    assert rec["began_above_mm"] == pytest.approx(13.0, abs=2.0)  # the pads came freely down to the table


@pytest.mark.parametrize("table_mm", [0, 5, 10, 13, 20, 26])
def test_a_rise_well_above_the_table_is_not_the_table_and_the_descent_goes_on_to_it(table_mm):
    """After the settle, with the reference 43 mm over where the plan has the lid meet the table (the pads,
    4 mm under it by then, more than SETDOWN_CREDIBLE_M over the plan), the pads come up 5 mm on the
    reference and fall back: a rise the press's rule takes for a contact. It is not the table: noted
    (`early`), the watch settles again, the descent goes on, slowly — and the table is the table, wherever
    in the recorded range it stands (the 19 set-downs flown onto one: -0.4..+26 mm over the plan, mean +13).

    The lag is the flown ones' (the pads 9 mm under the reference, coming up 5 mm as the slow stretch
    begins). Review 2026-11-17: while the watch settles again its rules are blind for 33 mm of the descent,
    +40 down to +7 mm over the plan; a table there (+10, +13, +20, +26) is seen only as the pads held up,
    which was counted from the least gap since the watch ARMED and dated to that sample: "something held
    the lid up 69 mm over where the plan has it meet the table", the lid carried back to the box from a
    table it stood on. (This test passed with the pads 2 mm under the reference and no catching up.)"""
    table = 0.20 + table_mm / 1000.0
    det, stopped, z_stop = _descent(table=table, bump=(0.243, 0.005, 0.012), catch_up=0.005)
    assert stopped is not None and det.credible() and det.verdict is None and det.refusal() is None
    assert det.contact[1] == pytest.approx(table, abs=0.002)
    assert len(det.early) == 1 and det.early[0][1] == pytest.approx(0.239, abs=0.003)
    rec = det.record()
    assert rec["early"][0][1] == pytest.approx(39.0, abs=3.0) and rec["early"][0][2] == "fast"
    assert rec["above_top_mm"] == pytest.approx(table_mm, abs=2.0) and rec["credible"] is True and rec["cut"]
    # seen as the pads stalled where the watch still settled again (blind), by a rise where it had settled
    assert det.how == ("held" if table_mm >= 10 else "fast")
    # ... where the pads last came freely down before it is the table too: the rise above the band fell back
    assert rec["began_above_mm"] == pytest.approx(table_mm, abs=2.0)
    # ... and stopped no more than SETDOWN_STALL_REF_M past where the pads met it (4 mm under the
    # reference), never at the touch's end
    assert table + 0.004 - z_stop < rlo.SETDOWN_STALL_REF_M + 0.001 and z_stop > 0.20 - rlo.SETDOWN_PAST_M + 0.002


def test_a_rise_well_above_the_table_with_no_table_under_it_is_not_let_go_on():
    """The same rise, and nothing under the lid: the touch flies to its end, nothing is met, and the early
    rise is said in why the lid is not let go."""
    det, stopped, _z = _descent(bump=(0.243, 0.005, 0.012), catch_up=0.005)
    assert stopped is None and not det.credible() and len(det.early) == 1 and det.verdict is None
    assert det.refusal()[0] == "never"
    assert "rises +" in det.refusal()[1] and "were not taken for the table" in det.refusal()[1]


@pytest.mark.parametrize("name", sorted(flown.SETDOWNS_FLOWN))
@pytest.mark.parametrize("table_mm", [None, 0, 13, 26])
def test_with_each_recorded_flights_own_lag_caught_up_all_the_way_the_table_is_the_table(name, table_mm):
    """The three flights recorded with the watch were cut 0.27-0.33 s after the slow stretch began, the
    pads still coming up on the reference: how far the catching up goes is not on record. Here each one's
    own lag (the most its pads stood under the reference: 8.6, 8.5 and 13.8 mm; its pace at the cut: 29,
    18 and 26 mm/s; from where it stood lowest) caught up ALL the way, to no lag at all. In free air
    nothing is met and nothing stopped. Over a table 0, 13 or 26 mm over the plan the lid is let go on it,
    at its height. With 2026-11-17's 13.8 mm the "held up" count, made from the least gap since the watch
    armed, stood 1.2 mm short of SETDOWN_HELD_M once the arm had caught up, fired on the table's first
    millimetres before the rise rule could, and dated the contact to where the watch armed: "held 66 mm
    over the plan" (review 2026-11-17: "the same with no early rise once the catch-up reaches 13 mm")."""
    under, risen, rate, low_at = _flown_lag(name)
    assert 0.0075 < under < 0.0145 and 0.0035 < risen < 0.0065 and 0.015 < rate < 0.032 and low_at < 0.1
    table = None if table_mm is None else 0.20 + table_mm / 1000.0
    det, stopped, z_stop = _descent(table=table, under=under, catch_up=under, rate=rate, delay=low_at)
    if table is None:
        assert stopped is None and det.contact is None and det.verdict is None and not det.credible()
        assert det.refusal()[0] == "never"
        return
    assert stopped is not None and det.credible() and det.refusal() is None
    assert det.contact[1] == pytest.approx(table, abs=0.002) and det.record()["above_top_mm"] == pytest.approx(
        table_mm, abs=2.0
    )
    assert table - z_stop < 0.008  # the reference stopped as the rise was seen: a few millimetres past the table


@pytest.mark.parametrize("table_mm", [0, 13, 26])
def test_a_catching_up_that_outlasts_the_settle_is_not_the_table_and_the_table_after_it_is(table_mm):
    """2026-11-17's lag once more (the pads 13.8 mm under the reference, coming up 26 mm/s), the catching
    up beginning 0.25 s after the slow stretch does: still under way when the settle ends, 0.6 s in. What
    rises after that is met as a rise that began with the pads more than SETDOWN_CREDIBLE_M over the plan
    (where the catching up began): not the table, noted, the watch settles again — and the table under
    it, met while it does, is the table: the catching up is over 37 mm over the plan and the pads come
    freely down the 11 mm and more from there to it. It was "something held the lid up" at the height the
    watch armed at: all 13.8 mm of the catching up counted toward SETDOWN_HELD_M."""
    under, _risen, rate, _low_at = _flown_lag("20261117-164059")
    table = 0.20 + table_mm / 1000.0
    det, stopped, z_stop = _descent(table=table, under=under, catch_up=under, rate=rate, delay=0.25)
    assert len(det.early) == 1 and det.early[0][1] == pytest.approx(0.20 + 0.0525, abs=0.004)  # where it began
    assert stopped is not None and det.credible() and det.verdict is None
    assert det.contact[1] == pytest.approx(table, abs=0.002)
    assert det.record()["began_above_mm"] == pytest.approx(table_mm, abs=2.0)
    assert table - z_stop < rlo.SETDOWN_STALL_REF_M + 0.001
    # and in free air: nothing met, the lid not let go
    det, stopped, _z = _descent(under=under, catch_up=under, rate=rate, delay=0.25)
    assert stopped is None and not det.credible() and det.refusal()[0] == "never" and len(det.early) == 1


def test_something_holding_the_lid_up_above_where_the_table_can_be_stops_the_descent_and_is_not_let_go_on():
    """The pads stop 60 mm over where the plan has the lid meet the table, as the arm still settles, and
    the reference goes on: 12 mm of it (SETDOWN_STALL_REF_M) with the pads no lower and whatever holds the
    lid up is no catching up. Above SETDOWN_CREDIBLE_M it is not the table: the descent is stopped there
    (the spring pressing no harder), and the lid is not let go on it. The table itself met that early — 30
    mm over the plan, inside the band — is the table."""
    det, stopped, z_stop = _descent(table=0.26)
    assert stopped is not None and det.verdict == "held" and not det.credible() and det.how == "held"
    # 10 mm of the reference on (the pads may come down 2 of the 12) and three samples, no more
    assert 0.26 + 0.009 - 0.0125 < z_stop < 0.26 + 0.009 - 0.0095
    verdict, why = det.refusal()
    assert verdict == "held" and "something held the lid up 60 mm over where the plan has it meet the table" in why
    assert det.record()["credible"] is False and det.record()["verdict"] == "held"
    det, stopped, _z = _descent(table=0.23)
    assert stopped is not None and det.credible() and det.contact[1] == pytest.approx(0.23, abs=0.002)
    # with the arm catching up as flown (5 mm of it before the pads are held): the height said is where the
    # pads STAND held, not where the gap was least (where the watch armed: it said "69 mm")
    det, stopped, z_stop = _descent(table=0.26, catch_up=0.005)
    assert stopped is not None and det.verdict == "held" and not det.credible()
    assert "something held the lid up 60 mm over where the plan has it meet the table" in det.refusal()[1]
    assert det.record()["above_top_mm"] == pytest.approx(60.0, abs=1.0)


@pytest.mark.parametrize("thing_mm", [40, 45, 50])
def test_a_rise_above_the_band_that_goes_on_is_something_holding_the_lid_up_and_stops_the_descent(thing_mm):
    """Something stands 40-50 mm over where the plan has the lid meet the table: met as a rise once the
    watch has settled, above SETDOWN_CREDIBLE_M, so not the table — noted, and the watch settles again.
    But the pads stand on it: SETDOWN_STALL_REF_M of the reference's descent with the pads no lower and
    it is the lid held up. The descent stops there — the reference no further past where the pads met the
    thing than that — the lid is not let go, and the height said is the thing's. (At 50 mm the pads meet
    it while the watch still settles, and they have stalled before a rise rule is asked.)"""
    thing = 0.20 + thing_mm / 1000.0
    det, stopped, z_stop = _descent(table=thing, catch_up=0.005)
    assert stopped is not None and det.verdict == "held" and not det.credible() and det.how == "held"
    assert [how for _t, _z, how, _seen in det.early] == ([] if thing_mm == 50 else ["fast"])
    assert all(z == pytest.approx(thing, abs=0.002) for _t, z, _how, _seen in det.early)
    assert det.contact[1] == pytest.approx(thing, abs=0.002)
    assert "something held the lid up %d mm over where the plan has it meet the table" % thing_mm in det.refusal()[1]
    assert thing + 0.004 - z_stop < rlo.SETDOWN_STALL_REF_M + 0.001  # the pads met it 4 mm under the reference


def test_what_the_pads_came_up_by_while_the_watch_settled_is_not_counted_once_it_has_settled():
    """The arm catches up 14.7 mm while the watch settles — 0.3 mm short of SETDOWN_HELD_M — and, flying
    on, the pads come up one millimetre more with the reference 25 mm over the plan (too little and too
    slow for either rise rule; the gap wanders that much in free air). Nothing holds the lid up: "held
    up" is counted from where the pads last came freely down — after the catching up, which is over by
    then. Counted from the least gap since the watch armed and dated where the pads stand, that
    millimetre read as the lid held up 27 mm over the plan — inside the band, so "the table", and the
    fingers would have opened in free air. The table under it is still the table."""
    det, stopped, _z = _descent(catch_up=0.0147, bump=(0.225, 0.001, 0.040))
    assert stopped is None and det.contact is None and det.verdict is None and det.early == []
    assert not det.credible() and det.refusal()[0] == "never"
    det, stopped, _z = _descent(table=0.213, catch_up=0.0147, bump=(0.225, 0.001, 0.040))
    assert stopped is not None and det.credible() and det.contact[1] == pytest.approx(0.213, abs=0.002)
    assert det.how == "fast"


def test_a_catching_up_of_the_whole_held_height_inside_the_settle_stops_the_descent_and_lets_nothing_go():
    """What the watch cannot tell apart, said plainly: pads that come up on the reference by all of
    SETDOWN_HELD_M while the watch still settles read as held up, whether a table does it or the arm's
    own catching up. 14 mm of it in free air is nothing; 15.5 mm stops the descent in free air, well
    above the band, and the lid goes back on the box — never let go. The recorded flights had come up
    3.8-6.0 mm by their cut, from 8.5-13.8 mm under the reference."""
    det, stopped, _z = _descent(catch_up=0.014)
    assert stopped is None and det.contact is None and det.verdict is None
    det, stopped, z_stop = _descent(catch_up=0.0155)
    assert stopped is not None and det.verdict == "held" and not det.credible()
    assert det.contact[1] > 0.20 + rlo.SETDOWN_CREDIBLE_M and z_stop > 0.20 + rlo.SETDOWN_CREDIBLE_M


# ---- one thing holds the lid up from where the pads last came freely down (second review 2026-11-17) ------


def _lags():
    """The lags the yielding and rigid things are flown with: the tests' own (the pads 9 mm under the
    reference, 5 mm caught up in 0.35 s), and each recorded flight's (setdown_flown.SETDOWNS_FLOWN: 8.6,
    8.5 and 13.8 mm under, at its own pace from where it stood lowest) caught up all the way. As
    _descent's keywords."""
    own = [dict(under=0.009, catch_up=0.005)]
    for name in sorted(flown.SETDOWNS_FLOWN):
        under, _risen, rate, low_at = _flown_lag(name)
        own.append(dict(under=under, catch_up=under, rate=rate, delay=low_at))
    return own


@pytest.mark.parametrize("thing_mm", [40, 48, 56])
@pytest.mark.parametrize("sinks", [0.2, 0.35, 0.5, 0.65])
def test_something_that_yields_first_met_above_where_the_table_can_be_is_never_the_table(sinks, thing_mm):
    """A bag, a cloth, a hand 40-56 mm over where the plan has the lid meet the table: the pads sink a
    fifth to two thirds of the way their free path goes under its top, so the gap goes on rising and the
    pads go on coming down, into the band. Second review 2026-11-17: the rise was noted "not the table",
    the watch settled again, the "held up" count began afresh where that settle ended and the contact was
    dated where the pads stood by then — inside the band: "the lid met the table", the fingers opened on
    it (met at +48, the pads sinking half-way: let go at +30, the reference 37 mm past the first touch;
    at +56, two thirds: after 67 mm). What holds the pads up is one thing from where they last came
    freely down, however far they have sunk into it: met above the band it is never the table. The
    descent stops with the hand's spring no more than SETDOWN_HELD_M and three samples deflected, the lid
    is not let go, and the words say where it was met — with the tests' lag and with each recorded
    flight's own (where the arm was still catching up as the pads met it, that is one rise with the
    catching up, begun higher still)."""
    thing = 0.20 + thing_mm / 1000.0
    for lag in _lags():
        told = {}
        det, stopped, _z = _descent(table=thing, sinks=sinks, told=told, **lag)
        assert stopped is not None and det.verdict == "held" and not det.credible(), lag
        assert det.how == "risen" and len(det.early) <= 1  # one thing: noted once at most, then held up by it
        rec = det.record()
        assert rec["verdict"] == "held" and rec["credible"] is False and rec["met"] is True
        # where the pads last came freely down: its top (the first millimetre of a rise is not yet one), or
        # higher; where they stand as the descent stops: on it, or sunk into it
        assert rec["began_above_mm"] > max(thing_mm - 3.0, 1000 * rlo.SETDOWN_CREDIBLE_M), lag
        assert rec["above_top_mm"] < thing_mm + 0.5, lag
        why = det.refusal()[1]
        assert "that is not the table, and the descent was stopped there" in why
        stands, began = 1000 * (det.contact[1] - 0.20), 1000 * (det.began[1] - 0.20)
        if stands > 1000 * rlo.SETDOWN_CREDIBLE_M:
            assert "something held the lid up %.0f mm over where the plan has it meet" % stands in why
        else:  # sunk to inside the band: said with where it was met
            assert "something met %.0f mm over where the plan has the lid meet the table" % began in why
            assert "still held it up, %.0f mm over it" % stands in why
        assert (1.0 - sinks) * (thing - told["free"]) < rlo.SETDOWN_HELD_M + 0.003, lag


def test_something_that_yields_and_then_gives_no_further_inside_the_band_is_still_not_the_table():
    """The same, and what yields gives no further once the pads have sunk to 30 mm over the plan (a bag
    with something in it): they stop there, inside the band, where a table could be. They have not come
    freely down since they met it 40 mm over the plan: it is that, not the table, whichever of the rules
    sees them held up."""
    det, stopped, z_stop = _descent(table=0.240, sinks=0.65, firm=0.230, catch_up=0.005)
    assert stopped is not None and det.verdict == "held" and not det.credible()
    assert det.how in ("held", "risen") and det.contact[1] == pytest.approx(0.230, abs=0.001)
    rec = det.record()
    assert rec["began_above_mm"] == pytest.approx(40.0, abs=3.0) and rec["above_top_mm"] == pytest.approx(30.0, abs=1.0)
    assert z_stop > 0.20 + 0.010  # stopped on it: the reference never came down to the plan's table
    assert det.refusal()[1].startswith(
        "something met %.0f mm over where the plan has the lid meet the table (more than 35) still held it up, 30 mm "
        "over it: that is not the table" % rec["began_above_mm"]
    )


def test_a_rise_that_goes_on_is_noted_once_and_where_it_began(monkeypatch):
    """What the `early` rows say of one thing that goes on holding the lid up. With a settle of 0.1 s the
    fast rule sees the same rise twice before the gap has risen SETDOWN_HELD_M (something met 56 mm over
    the plan, the pads sinking half-way): one thing, noted once. And with one of 1.0 s the first rule to
    see something met 48 mm over the plan (the pads sinking seven tenths of the way) is asked once they
    have sunk to inside the band, and dates its rise there: the note says where the pads last came
    freely down, 46-48 mm over the plan, not a height inside the band that "was not taken for the table"
    (nor, of a rise whose rule has it above the band, wherever in it the rule's own window began)."""
    monkeypatch.setattr(rlo, "SETDOWN_SETTLE_S", 0.1)
    det, stopped, _z = _descent(table=0.256, sinks=0.5)
    assert stopped is not None and det.verdict == "held" and det.how == "risen"
    assert [how for _t, _z, how, _seen in det.early] == ["fast"]
    assert det.early[0][1] == pytest.approx(0.256, abs=0.002)
    monkeypatch.setattr(rlo, "SETDOWN_SETTLE_S", 1.0)
    det, stopped, _z = _descent(table=0.248, sinks=0.7)
    assert stopped is not None and det.verdict == "held" and det.how == "risen"
    assert [how for _t, _z, how, _seen in det.early] == ["fast"]
    assert det.early[0][1] == pytest.approx(0.247, abs=0.0015) and det.early[0][3] - det.early[0][0] > 0.5
    assert (
        "(rises +%.0f mm over it were not taken for the table)" % (1000 * (det.early[0][1] - 0.20)) in det.refusal()[1]
    )


_RIGID_LAGS = {
    "the tests' lag": dict(under=0.009, catch_up=0.005),
    "2026-11-13's, not yet caught up": dict(under=0.0085, catch_up=0.0),
    "2026-11-17's, not yet caught up": dict(under=0.0138, catch_up=0.0),
    "2026-11-17's, caught up as at its cut": dict(under=0.0138, catch_up=0.006),
}


@pytest.mark.parametrize("lag", list(_RIGID_LAGS.values()), ids=list(_RIGID_LAGS))
@pytest.mark.parametrize("thing_mm", [36, 40, 44, 48, 52, 54, 60])
def test_something_rigid_above_where_the_table_can_be_is_pressed_no_further_than_the_pads_stalling_shows(thing_mm, lag):
    """Something rigid 36-60 mm over where the plan has the lid meet the table — or the table itself
    where the plan stands that far under it: setdown_table_z does not move the plan for a rough fix or
    with no grip height, and 2026-11-13's and 11-17's plans stood 46-55 mm low. Second review 2026-11-17:
    met while the watch still settled, the "held up" count was dropped just short of its 15 mm where the
    settle ended and begun again, and the reference went on up to 33 mm past the contact (54 mm over the
    plan with 11-13's lag; 32 mm at +48 with 11-17's), some 52 N, before the descent stopped. It had been
    17 mm; the owner, 2026-11-02, of ~60 N: "presses the lid down onto the table waaaaaaay too hard". The
    pads' own descent says it: they have not come down while the reference came down SETDOWN_STALL_REF_M
    (12 mm). Stopped no further past the contact than that, whatever the lag and wherever it stands, the
    lid not let go, the height said its own."""
    thing = 0.20 + thing_mm / 1000.0
    told = {}
    det, stopped, z_stop = _descent(table=thing, told=told, **lag)
    # stalled on it; or, where the arm was still catching up as the pads met it, the gap risen SETDOWN_HELD_M
    # with what it came up by: sooner still
    assert stopped is not None and det.verdict == "held" and not det.credible() and det.how in ("held", "risen")
    assert det.contact[1] == pytest.approx(thing, abs=0.002)
    assert "something held the lid up %d mm over where the plan has it meet the table" % thing_mm in det.refusal()[1]
    pressed = told["touch_ref"] - z_stop  # how far the reference went on past where the pads met it
    assert 0.007 < pressed < rlo.SETDOWN_STALL_REF_M + 0.001 < 0.017


@pytest.mark.parametrize("lag_mm", [16, 20, 24, 26])
def test_a_catching_up_of_more_than_the_held_height_is_never_the_table_in_free_air(lag_mm):
    """The pads 16-26 mm under the reference as the slow stretch begins (the most on record is 13.8) and
    coming up on it by three quarters or all of that, 18-40 mm/s from 0, 0.1 or 0.25 s after it begins:
    nothing under the lid. Second review 2026-11-17: what came up inside the settle was forgotten where it
    ended, and the rest, rising with the pads inside the band by then, was taken for the table by the
    fast rule — 24 mm at 26 mm/s from 0.1 s: "the lid met the table" 35 mm over the plan, in free air
    (before the first review's fix that one read "held"; 16 mm at 18 mm/s from 0.25 s was let go on by
    both). A catching up is one rise from where the slow stretch begins, the pads well above the band
    there: never the table, whatever its size and pace. Past SETDOWN_HELD_M of it the descent may stop
    ("held": something met above the band, wherever the pads stand by then) and the lid goes back on the
    box."""
    under = lag_mm / 1000.0
    seen = set()
    for part in (0.75, 1.0):
        for rate in (0.018, 0.026, 0.040):
            for delay in (0.0, 0.1, 0.25):
                det, stopped, z_stop = _descent(under=under, catch_up=part * under, rate=rate, delay=delay)
                assert not det.credible(), (part, rate, delay)
                assert det.verdict in (None, "held") and det.refusal()[0] in ("never", "held")
                if det.verdict == "held":
                    assert det.began[1] > 0.20 + rlo.SETDOWN_CREDIBLE_M + 0.003, (part, rate, delay)
                seen.add(det.refusal()[0])
    assert "held" in seen  # ... which is what more than SETDOWN_HELD_M of it reads as


def test_the_recorded_lag_caught_up_slower_than_on_record_is_never_the_table_however_long_it_takes():
    """How long the catching up lasts is not on record (the three flights were cut 0.27-0.33 s in, the pads
    coming up 18-29 mm/s). 2026-11-17's lag (13.8 mm under the reference), three quarters or all of it
    caught up at 10 or 14 mm/s from 0, 0.1 or 0.25 s after the slow stretch begins: still rising 0.7-1.6 s
    in, with the pads inside the band and the settle over. Nothing is under the lid. Dated by the
    rule that saw it, that rise began inside the band: "the lid met the table" 29-35 mm over the plan, in
    free air, in 20 of these 36 flights (the first work's open item: "if it outlasts 0.6 s, a false rise
    inside the 35 mm band would still open the fingers"). It is one rise from where the slow stretch
    begins, the pads 48 mm or more over the plan there: noted, never the table. (One of 9 mm/s or slower,
    0.16 mm a mm of the descent, is no faster than free air's own drift and is not told from it.)"""
    under, _risen, _rate, _low_at = _flown_lag("20261117-164059")
    band = 0.20 + rlo.SETDOWN_CREDIBLE_M
    noted = 0
    for part in (0.75, 1.0):
        for rate in (0.010, 0.014):
            for delay in (0.0, 0.1, 0.25):
                for seed in (0, 1, 2):
                    det, stopped, _z = _descent(under=under, catch_up=part * under, rate=rate, delay=delay, seed=seed)
                    case = (part, rate, delay, seed)
                    assert stopped is None and not det.credible() and det.refusal()[0] == "never", case
                    assert len(det.early) <= 1 and all(z > band + 0.010 for _t, z, _how, _seen in det.early), case
                    noted += len(det.early)
    assert noted >= 20  # the rise rules did see them: it is where they began that says what they are


def _least_descent(rows):
    """The least the pads came down in any SETDOWN_STALL_REF_M of the reference's descent over `rows`,
    (the reference's pads, the measured pads) in order: what the watch's stall rule is asked (m)."""
    least = None
    for i, (ref, pads) in enumerate(rows):
        was = next((p for r, p in reversed(rows[:i]) if r >= ref + rlo.SETDOWN_STALL_REF_M), None)
        if was is not None:
            least = was - pads if least is None else min(least, was - pads)
    return least


def test_in_no_recorded_set_down_do_the_pads_stall_before_the_table_is_felt():
    """What SETDOWN_STALL_REF_M and SETDOWN_STALL_PADS_M rest on (held up: the pads came down less than 2
    mm while the reference came down 12). The three set-downs flown with the watch, the arm catching up
    with its reference from where the slow stretch begins to the cut: the pads came down 6.9-8.3 mm in
    every 12 mm of the reference's descent (55-69 % of its pace). The 19 flown onto the table before it,
    from where the watch arms to where it stops the descent (or the touch's end): 3.9 mm or more — the
    least where the rise rules had just felt the table (10-30 14:26), 6.4 mm before any table (11-04
    12:40, its gap rising 6.7 mm in free air 50-70 mm over the plan)."""
    three = {}
    for name, rec in flown.SETDOWNS_FLOWN.items():
        tt, tz = (np.array(v, float) for v in zip(*rec["traj"], strict=True))
        rows = [(float(np.interp(frac * tt[-1], tt, tz)), z) for _t, frac, z in rec["fed"]]
        three[name] = _least_descent([(ref, z) for ref, z in rows if ref <= rec["plan"] + rlo.SETDOWN_WATCH_M])
    assert sorted(round(1000 * v, 1) for v in three.values()) == pytest.approx([6.9, 7.9, 8.3], abs=0.15)
    older = {}
    for name in flown.SETDOWNS_OLDER:
        _det, at = _older(name)
        end = -rlo.SETDOWN_PAST_M if at is None else at
        rows = [(ref, pads) for _t, ref, pads in flown.older_samples(name) if end <= ref <= rlo.SETDOWN_WATCH_M]
        older[name] = _least_descent(rows)
    assert min(older.values()) == pytest.approx(0.0039, abs=0.0002) == older["20261030-142611"]
    assert min(min(three.values()), min(older.values())) > 1.9 * rlo.SETDOWN_STALL_PADS_M


# ---- the watch that fails, and the feedback no poll read (review 2026-11-17) ------------------------------


class _Breaks:
    """test_press_contact's chain (joint 1 the tool's height) whose kinematics raise once `broken`."""

    broken = False

    def fk(self, q):
        if self.broken:
            raise RuntimeError("boom")
        return np.eye(3), np.array([0.0, 0.0, float(q[0])])


def test_a_set_down_watch_that_fails_as_it_watches_stops_the_descent_there_and_the_lid_is_not_let_go():
    """press_contact's watch of the set-down flew its planned end where it failed. A lid is let go only on
    a contact now, and a planned end is no contact: a watch that fails on the stretch it watches stops the
    descent where it failed — the reference does not go on down onto a table nothing watches for."""

    class BreaksAfter(_Breaks):
        """Sound for `calls` of its kinematics: the trajectory's heights (191 points), then some 50 samples
        of the slow stretch."""

        def __init__(self, calls):
            self.left = calls

        def fk(self, q):
            self.left -= 1
            self.broken = self.broken or self.left < 0
            return super().fk(q)

    det, stopped, z_stop = _descent(chain=BreaksAfter(100000), slow_q=True, catch_up=0.005)
    assert stopped is None and det.failed is None and det.armed_t is not None  # sound: a free descent, watched
    det, stopped, z_stop = _descent(chain=BreaksAfter(300), slow_q=True, catch_up=0.005)
    assert det.failed == "RuntimeError: boom" and det.verdict == "failed" and det.failed_when == "watching"
    assert stopped is not None and 0.20 + 0.030 < z_stop < 0.20 + rlo.SETDOWN_SLOW_M - 0.005  # where it failed
    assert not det.credible() and det.refusal() == (
        "failed",
        "the set-down's watch failed (RuntimeError: boom) and the descent was stopped there",
    )
    rec = det.record()
    assert (rec["credible"], rec["verdict"], rec["failed"]) == (False, "failed", "RuntimeError: boom")


def test_a_set_down_watch_that_fails_before_the_descent_stops_the_descent_not_the_lift_or_the_carry(capsys):
    """The watch is the stop_when of the whole one motion — the lift, the clear, the carry, the way down,
    the touch — polled from its first sample. Where it failed there (it cannot make the heights of the
    trajectory sent, on its first poll) it cancelled the flight on the spot: the lid half lifted or in
    mid-carry, and the way back from there a line nothing had checked against the box (review 2026-11-17).
    The lift and the carry fly on. The descent is what it stops: where the slow stretch begins, found in the
    trajectory sent by the joints the way down ends at (no kinematics: those are what failed). The lid is
    not let go."""
    chain = _Breaks()
    chain.broken = True
    det, stopped, z_stop = _descent(chain=chain, slow_q=True)
    assert det.failed == "RuntimeError: boom" and det.verdict == "failed" and det.failed_when == "before"
    assert det.traj is None and det.armed_t is None  # it never had the trajectory's heights, and never watched
    # not cancelled above the slow stretch: stopped where it begins, SETDOWN_SLOW_M over the plan
    assert stopped is not None and z_stop == pytest.approx(0.20 + rlo.SETDOWN_SLOW_M, abs=0.001)
    assert det.stopped_frac == pytest.approx(det.fail_from, abs=0.01)
    assert not det.credible() and det.refusal() == (
        "failed",
        "the set-down's watch failed (RuntimeError: boom) and the descent was stopped where its slow stretch begins",
    )
    assert "the descent will be stopped where its slow stretch begins; the lid will not be let go" in (
        capsys.readouterr().out
    )
    # joints further from the sent trajectory than SLOW_Q_TOL_RAD are no place in it: nothing is stopped
    det.slow_q = [v + 2 * rlo.SLOW_Q_TOL_RAD for v in det.slow_q]
    assert det._descent_from() is None


def test_a_set_down_watch_that_fails_with_no_way_to_tell_the_descent_flies_the_planned_end_and_lets_nothing_go():
    """Failed before the descent, and nothing says where that begins (no joints of the slow stretch's
    start, or a trajectory it cannot read): the flight is not cancelled where it happens to be. It flies its
    planned end, as the set-down did before it was watched, and the lid is not let go; the run says so."""
    client = SimpleNamespace(last_execution={"sent_at": 0.0, "sent_traj": object(), "feedback_q": [(0.0, 0.0, [0.0])]})
    for slow_q in (None, [0.28] + [0.0] * 6):
        det = rlo.SetdownContact(_Breaks(), client, 0.20, 0.28, 0.192, slow_q=slow_q)
        assert [det(f) for f in (0.1, 0.4, 0.9, 1.0)] == [False] * 4  # the lift, the carry, the descent: all fly
        assert det.verdict == "failed" and det.failed_when == "before" and det.fail_from is None
        assert not det.credible() and det.stopped_frac is None
        assert det.refusal() == (
            "failed",
            "the set-down's watch failed (AttributeError: 'object' object has no attribute 'points') before the "
            "descent, and where that begins was not known: the touch flew to its planned end with nothing watching "
            "for the table",
        )


def test_the_feedback_no_poll_read_is_read_once_the_flight_is_over_and_says_whether_the_lid_met_the_table():
    """client.execute polls a stop_when only once what its goal hosts has run. A watch never polled while
    the set-down flew had read nothing when the release asked it, and the run said "the set-down was not
    watched (no feedback of its last stretch)" of feedback the client had kept; a lid that stood on the
    table went back on the box (review 2026-11-17). read_rest, once the execution is over, reads what was
    kept and not read: the watch then says what ALL of the flight's samples say. Nothing is cut by it —
    the reference flew to the touch's end — and its record says how it knows."""
    det, stopped, _z = _descent(table=0.213, catch_up=0.005, polled=False)
    assert stopped is None and det.stopped_frac is None  # not polled: nothing was stopped
    assert det.credible() and det.refusal() is None and det.contact[1] == pytest.approx(0.213, abs=0.002)
    rec = det.record()
    assert rec["read_after_flight"] == rec["samples"] > 100 and rec["seen_after_flight"] is True
    assert (rec["watched"], rec["met"], rec["credible"], rec["cut"]) == (True, True, True, False)
    # over nothing: the touch ended and no table was felt, said of a flight that WAS read
    det, _stopped, _z = _descent(catch_up=0.005, polled=False)
    assert not det.credible() and det.record()["watched"] is True and det.record()["seen_after_flight"] is False
    assert det.refusal() == (
        "never",
        "the touch ended 8 mm under where the plan has the lid meet the table and the arm felt no table",
    )
    # a watch polled all the way has nothing left to read, and what it saw it saw in flight
    det, stopped, _z = _descent(table=0.213, catch_up=0.005)
    assert stopped is not None and det.read_rest() == 0 and det.record()["seen_after_flight"] is False
    assert det.record()["cut"] is True
    # ... and what another execution left with the client since is not this set-down's to read
    det.client.last_execution = {"sent_at": 9.0, "sent_traj": object(), "feedback_q": [(9.0, 0.5, [0.0])]}
    assert det.read_rest() == 0 and det.credible() and det.failed is None
    # no feedback kept at all: truly not watched
    det = rlo.SetdownContact(_Breaks(), SimpleNamespace(last_execution=None), 0.20, 0.28, 0.192)
    assert det.read_rest() == 0 and det.refusal()[1].startswith("the set-down was not watched (no feedback")
    # a watch that fails reading it: said, and the lid is not let go
    chain = _Breaks()
    det, _stopped, _z = _descent(table=0.213, chain=chain, polled=False)  # reads fine ...
    assert det.credible()
    chain.broken = True
    det, _stopped, _z = _descent(table=0.213, chain=chain, polled=False)  # ... and not
    assert (
        not det.credible()
        and det.failed_when == "after"
        and det.refusal()
        == (
            "failed",
            "the set-down's watch failed (RuntimeError: boom) reading what was kept of the flight once it was over: "
            "nothing says the lid met the table",
        )
    )


class _FeltOnceRead(_Felt):
    """A set-down's watch that no poll fed while it flew: what it felt is known only once it has read what
    the client kept (read_rest), and then it is the lid on the table."""

    def __init__(self):
        super().__init__("never")
        self.read = 0

    def read_rest(self):
        self.read += 1
        self.how = "met"
        return 87

    def record(self):
        out = super().record()
        out.update(read_after_flight=87 if self.read else 0, seen_after_flight=bool(self.read))
        if self.read:
            out["cut"] = False
        return out


def test_the_release_reads_what_the_watch_has_not_read_before_it_asks_whether_the_lid_met_the_table(
    ctx, monkeypatch, tmp_path, capsys
):
    """_release_and_home reads the rest of the flight's feedback into the watch first: a lid that met the
    table while no poll came is let go there, the run saying how it knows; it was carried back to the box
    as "not watched"."""
    watch = _FeltOnceRead()
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], setdown=watch)
    assert watch.read == 1 and code == 0 and rec.events[-3:] == HOME and "lid:back-up" not in rec.events
    (sd,) = (r for r in rows if r["leg"] == "reach_setdown")
    assert sd["credible"] is True and sd["contact"]["seen_after_flight"] is True
    assert (
        "SET-DOWN CONTACT: the lid met the table +12 mm from where the plan put it; a cut would have ended the "
        "touch 20 mm short of its planned end (seen only in the feedback read once the flight was over: nothing "
        "was cut)"
    ) in capsys.readouterr().out


# ---- the set-down hosts no planning (review 2026-11-17) ---------------------------------------------------


@pytest.mark.parametrize("one_motion", [True, False], ids=["the spot chosen ahead", "the spot chosen after the lift"])
def test_the_set_down_hosts_no_planning_home_is_planned_before_it_or_while_the_retreat_flies(
    ctx, monkeypatch, tmp_path, one_motion
):
    """The client runs what an execution hosts before it first polls that execution's stop_when
    (client.execute), so the package hosts nothing on the press's stroke (reach_press.fly_way_in). The
    set-down's execution carries the watch that stops the descent where the lid meets the table. With the
    spot chosen after the lift (choose_and_carry) it hosted the home plan: for as long as that took the
    watch saw nothing — the reference going on to the touch's end, 8 mm under the plan, onto a table it had
    met — and a plan that outlasted the flight left it never polled (review 2026-11-17). Neither way of
    setting the lid down hosts anything now. Home after it: planned with the carry while the hand came
    down to the knob (the spot chosen ahead), or while the retreat flies, from where it ends, and flown
    from there with no plan at rest."""
    hosted = _hosting(ctx, monkeypatch)
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], one_motion=one_motion)
    assert code == 0
    flown_ = [tuple(names) for names, gains in _executions(rec) for _goal in gains]
    assert len(flown_) == len(hosted)
    by = dict(zip(flown_, hosted, strict=True))
    setdown = tuple(CLOSE + CARRY) if one_motion else tuple(CARRY)
    assert setdown in by and by[setdown] is False  # the watch is polled from the execution's first sample
    home_planned = [at for name, at in rec.plans if name == "home"][-1]
    if one_motion:
        assert by[tuple(RELEASED)] is False and home_planned <= rec.events.index("grip:close")
        return
    # the lift's execution hosts the carry's plan (no watch on it); the retreat's, home's
    assert by[tuple(CLOSE)] is True and by[tuple(RELEASED)] is True
    assert home_planned == rec.events.index("place:retreat") + 1 == rec.events.index("home")
    (ahead,) = (r for r in rows if r["leg"] == "home_ahead")
    assert ahead["flown"] is True  # ... and flown from where the retreat ended: no planner call at rest
    assert rec.legs["home"].traj.points[0].positions == pytest.approx(rec.legs["place:retreat"].goal_joints)


def test_a_home_that_does_not_plan_while_the_retreat_flies_is_planned_once_more_at_rest(ctx, monkeypatch, tmp_path):
    """The home hosted by the retreat is refused (the planner can): the lid is let go and the hand back up
    all the same, and home is planned once more where the arm stands, as it was before."""
    plans = _home_plans(monkeypatch, refuse=(2,))  # the 1st is the way back's, from the look pose
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], one_motion=False)
    assert code == 0 and rec.events[-3:] == HOME
    assert [(at is not None, legs is not None) for at, legs in plans] == [(True, True), (True, False), (False, True)]
    assert plans[1][0] == pytest.approx(rec.legs["place:retreat"].goal_joints)
    assert not [r for r in rows if r["leg"] == "home_ahead"]  # none in hand: planned at rest


# ---- the table the set-down goes by, in the whole run (review 2026-11-17) ---------------------------------


def _touch_table_z(chain):
    """Where _full_flow's `touch_table` has the set-down's table: setdown_table_z's rule on the fake arm,
    whose fk stands the pads at one height whatever the joints, and on _full_flow's fix, the lid's top
    121 mm over its table — as far under the lid's top as the cameras saw it, the top where the fingers
    hold it."""
    from box_opening.tasks.press_contact import pads_z

    return pads_z(chain, [0.0] * 7) + rlo.GRIP_BELOW_LID_M - (BUTTON[2] - TABLE_Z)


def _setdown_by(ctx, chain, rows, table):
    """The run's set-down went by the table at `table`, not the cameras': the lid's spot, the row that
    records it, and the lines flown down onto it."""
    (spot,) = (r for r in rows if r["leg"] == "lid_spot")
    assert spot["xyz"][2] == pytest.approx(table, abs=1e-4) and ctx.lid_drop.xyz[2] == pytest.approx(table)
    assert spot["table"]["z"] == pytest.approx(TABLE_Z, abs=1e-4)  # the cameras' own, kept beside it
    assert spot["table"]["setdown_z"] == pytest.approx(table, abs=1e-4)
    # the way down ends SETDOWN_SLOW_M over where the lid meets THAT table, the touch SETDOWN_PAST_M under it
    _quat, contact, _hover = rlo.setdown_poses(ctx, ctx.lid_drop)
    cameras = ContainerPose(xyz=(*ctx.lid_drop.xyz[:2], TABLE_Z), yaw=ctx.lid_drop.yaw)
    assert contact[2] - rlo.setdown_poses(ctx, cameras)[1][2] == pytest.approx(table - TABLE_Z)
    ends = [xyz[2] for xyz, _q in chain.lines if xyz[:2] == pytest.approx(contact[:2])]
    assert any(z == pytest.approx(contact[2] + rlo.SETDOWN_SLOW_M) for z in ends)
    assert any(z == pytest.approx(contact[2] - rlo.SETDOWN_PAST_M) for z in ends)
    assert not any(z < contact[2] - rlo.SETDOWN_PAST_M - 1e-6 for z in ends)  # and nothing flies lower
    return spot


@pytest.mark.parametrize("from_press", [False, True], ids=["gripped from the look pose", "gripped from the press"])
def test_the_one_motion_sets_the_lid_down_by_the_table_moved_to_where_the_fingers_hold_the_lid(
    ctx, monkeypatch, tmp_path, capsys, from_press
):
    """2026-11-13 and 11-17: the set-down's plan stood 46-55 mm under the table, built on the wrist
    camera's heights; setdown_table_z moves the table by where the fingers hold the lid's top. The function
    was tested alone and every run in these tests flew with it switched off (review 2026-11-17: with
    spot_candidates made to ignore it, every test stayed green). Here it is in the whole run. The fake
    arm's fingers hold the knob 63 mm over where the cameras' table and the box's height put the lid's top:
    the spot planned while the hand comes down — before any lift is on the run's context, by the joints
    the knob is gripped at handed to the lookahead (grip_ahead, grip_ahead_rise) —, the plan's contact and
    the touch's end all stand 63 mm over the cameras' table, and the run says so."""
    code, rec, chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], touch_table=True, from_press=from_press
    )
    assert code == 0 and "spot-look" not in rec.events
    table = _touch_table_z(chain)
    assert table == pytest.approx(TABLE_Z + 0.063, abs=5e-4)
    spot = _setdown_by(ctx, chain, rows, table)
    assert spot["chosen"] == "ahead, with the scene camera"
    out = capsys.readouterr().out
    assert "the table the lid is set down by: %.3f, 63 mm over the cameras' %.3f" % (table, TABLE_Z) in out
    assert "lid goes to [%.2f, %.2f, %.3f]" % (*ctx.lid_drop.xyz[:2], table) in out


@pytest.mark.parametrize("ahead", [True, False], ids=["the carry planned as the lift flew", "the carry planned there"])
def test_the_spot_chosen_after_the_lift_is_set_down_by_the_table_moved_to_where_the_fingers_hold_the_lid(
    ctx, monkeypatch, tmp_path, ahead
):
    """The same where the spot is chosen with the lid lifted (choose_and_carry): the table moved by the
    lift's own grip (ctx.lid_lifted) — for the carry planned while the lift flew (carry_ahead, which is
    handed no grip there), and for the one planned at the spot look where none was — the spot, its row
    and the set-down's lines on it."""
    if not ahead:
        monkeypatch.setattr(rlo, "carry_ahead", lambda *a, **k: None)
    code, rec, chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], touch_table=True, one_motion=False
    )
    assert code == 0 and "spot-look" in rec.events
    spot = _setdown_by(ctx, chain, rows, _touch_table_z(chain))
    assert (spot["ahead"] is not None and spot["ahead"]["flown"] is True) if ahead else spot["ahead"] is None
    # planned while the lift flew, or at the spot look
    planned = [at for name, at in rec.plans if name == "lid:carry"]
    assert planned == [rec.events.index("spot-look") + (0 if ahead else 1)]


# ---- the 19 set-downs flown onto the table, through the watch (review 2026-11-17) -------------------------


def _older(name, past=None):
    """One of the 19 set-downs flown onto the table before the watch (setdown_flown.SETDOWNS_OLDER) fed to
    the set-down's watch as the client fed it — the plan where setdown_table_z has it, the touch ending
    `past` under it (None: SETDOWN_PAST_M) —: each kept sample down to where the reference stands at that
    end, or to the flight's own end where that is higher. Returns (the watch, where the reference's pads
    stood over the plan as it stopped the descent, or None)."""
    top, past = 0.25, rlo.SETDOWN_PAST_M if past is None else past
    rows = [(0.02 + t, ref, pads) for t, ref, pads in flown.older_samples(name)]
    total = rows[-1][0]
    traj = SimpleNamespace(
        points=[
            SimpleNamespace(
                positions=[top + ref + dp.DOWN_CONTACT_TOOL[2]] + [0.0] * 6,
                time_from_start=SimpleNamespace(sec=int(t), nanosec=int(round((t - int(t)) * 1e9))),
            )
            for t, ref, _pads in rows
        ]
    )
    info = {"sent_at": 0.0, "sent_traj": traj, "feedback_q": []}
    det = rlo.SetdownContact(_Z(), SimpleNamespace(last_execution=info), top, top + rlo.SETDOWN_WATCH_M, top - past)
    for t, ref, pads in rows:
        if det.armed_t is not None and ref < -past:
            break
        info["feedback_q"].append((t, t / total, [top + pads + dp.DOWN_CONTACT_TOOL[2]] + [0.0] * 6, None))
        if det(t / total):
            return det, ref
    return det, None


# Where the watch has each of them meet the table (mm over the plan, the rule's own dating): the 17 it feels
# with the touch ending SETDOWN_PAST_M under the plan, as replayed before the second review of 2026-11-17
_OLDER_FELT_MM = {
    "20261029-120248": 12.0,
    "20261029-120459": 15.9,
    "20261029-133204": 8.8,
    "20261029-133716": 7.0,
    "20261029-143035": 0.6,
    "20261029-145127": 9.8,
    "20261029-150250": 12.6,
    "20261030-100830": 24.0,
    "20261030-133221": 8.3,
    "20261030-133311": 16.2,
    "20261030-133506": 21.4,
    "20261030-133548": 13.7,
    "20261030-142532": 11.8,
    "20261030-142611": 23.5,
    "20261104-123929": 9.8,
    "20261104-124050": 7.3,
    "20261104-124212": 8.2,
}


def test_the_set_downs_flown_onto_the_table_are_felt_as_the_table_before_the_touch_ends_at_todays_pace():
    """The 19 set-downs flown onto the table before the watch (2026-10-29 to 11-04), replayed through it
    with the plan where setdown_table_z has it and the touch ending SETDOWN_PAST_M (8 mm) under that.

    None reads as something holding the lid up, and none as a rise above the band: a real table is the
    table, -0.4..+26 mm over the plan. Of the 18 that kept the lid, the 12 flown at today's touch pace
    (53-61 mm/s) are all felt, the reference still 4 mm or more from the touch's end (the least: 10-29
    12:04, a lid that settled slowly, felt 3.4 mm under the plan). Of the 6 flown with the touch at twice
    that (104-125 mm/s, #89: the owner found it "too hard"), five are felt, 2.0-6.4 mm under the plan;
    10-29 14:45 is not — its lid met the table 2 mm over the plan with the pads 6 mm behind the reference
    and settled as the hand came on, the rise seen only 14 mm under the plan. Review 2026-11-17: that one
    would have been carried back to the box. The touch is not made deeper for it (SETDOWN_PAST_M's
    note); this pins what 8 mm does and does not feel. The one whose lid had slipped out on the way
    (10-30 13:28) is not felt at all."""
    older = flown.SETDOWNS_OLDER
    felt, began = {}, {}
    for name in older:
        det, at = _older(name)
        rec = det.record()
        assert rec["verdict"] in (None, "never") and rec["early"] == [], name  # never "held", never "early"
        assert det.credible() == (at is not None)
        felt[name] = (at, rec.get("above_top_mm"))
        began[name] = rec.get("began_above_mm")
    kept = [name for name in sorted(older) if older[name]["kept"]]
    today = [name for name in kept if older[name]["touch"] < 70]
    faster = [name for name in kept if older[name]["touch"] > 100]
    assert (len(older), len(kept), len(today), len(faster)) == (19, 18, 12, 6)
    assert all(felt[name][0] is not None for name in today)
    assert min(felt[name][0] for name in today) > -(rlo.SETDOWN_PAST_M - 0.004)
    assert [name for name in faster if felt[name][0] is None] == ["20261029-144545"]
    assert all(-0.0065 < felt[name][0] < -0.0015 for name in faster if felt[name][0] is not None)
    assert all(-1.0 < above < 27.0 for at, above in felt.values() if at is not None)  # where each met the table
    # ... each at the height the rule that felt it dates it to, as before the second review of 2026-11-17
    # (mm over the plan), the reference where it was then (the same samples stop the descent) ...
    assert {name: above for name, (at, above) in felt.items() if at is not None} == pytest.approx(
        _OLDER_FELT_MM, abs=0.11
    )
    # ... and what holds the pads up there is the table alone: they had come freely down to +1..+27 mm over
    # the plan before it, inside the band every time (the nearest 8 mm under its edge: 10-30 10:08)
    assert all(1.0 < began[name] < 27.5 for name in _OLDER_FELT_MM) and began["20261030-100830"] == max(
        began[name] for name in _OLDER_FELT_MM
    )
    assert felt["20261030-132845"][0] is None and not older["20261030-132845"]["kept"]
    # 14:45's is felt only by a touch that goes 14 mm under the plan
    assert _older("20261029-144545", past=0.012)[1] is None
    det, at = _older("20261029-144545", past=0.016)
    assert det.credible() and at == pytest.approx(-0.014, abs=0.001)


# ---- spot after spot (owner 2026-11-19: "it should keep trying spots, assume tere will always be a spot for the
# lid") ---------------------------------------------------------------------------------------------------------


def _lid_wide(ctx):
    return max(ctx.model.lid_dims[:2])


def _after_first_touch(rec):
    return rec.events[rec.events.index("place:touch") + 1 :]


def test_spot_after_spot_no_table_felt_at_any_ends_on_the_box_after_as_many_spots_as_a_run_tries(
    ctx, monkeypatch, tmp_path, capsys
):
    """No spot's set-down feels the table (a run by hand: nothing times it). The lid goes on from spot
    to spot — each chosen by the same rules, each a lid's width or more from every spot left before it,
    each on table the cameras measured — and never an unbounded number of them: after SETDOWN_SPOTS the
    lid goes back on the box, the last resort, and the run says how it ended. It is never let go."""
    monkeypatch.delenv(rlo.RUN_BUDGET_ENV, raising=False)
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY, scene=_table(), setdown="never"
    )
    assert code == 1 and "place:open" not in rec.events
    assert rec.events.count("place:touch") == rlo.SETDOWN_SPOTS == len(rec.setdowns) == 6
    assert rec.events.count("lid:back-up") == rlo.SETDOWN_SPOTS  # up from each; from the last, on to the box
    assert rec.events[-5:] == ["lid:back-over", "lid:put-back", "lid:release", "lid:rise-off", "home"]
    spots = [d for d, _how in rec.setdowns]
    for i, a in enumerate(spots):
        assert rlo.box_clearance(ctx, a) >= 0.0
        for b in spots[:i]:
            assert math.hypot(a.xyz[0] - b.xyz[0], a.xyz[1] - b.xyz[1]) >= _lid_wide(ctx)
    left = [r for r in rows if r["leg"] == "lid_spot_left"]
    assert [r["spot"] for r in left] == [1, 2, 3, 4, 5, 6]
    assert [r["next"] for r in left] == ["another spot"] * 5 + ["the box"]
    assert left[-1]["why_the_box"] == "6 spots is as many as a run tries (SETDOWN_SPOTS)"
    (back,) = (r for r in rows if r["leg"] == "lid_put_back")
    assert back["ok"] is True and "no other spot is tried: 6 spots is as many as a run tries" in back["why"]
    out = capsys.readouterr().out
    assert out.count("the lid is not let go: up with it and on to another spot") == 5
    assert "back onto the box (no other spot is tried: 6 spots is as many as a run tries (SETDOWN_SPOTS))" in out
    # what the line before each set-down promises is what follows one that feels no table: the last spot's, the box
    flies = {int(t.split()[2].rstrip(",")): t for t in out.splitlines() if t.startswith("[press_demo] SPOT ")}
    assert sorted(flies) == [2, 3, 4, 5, 6]
    assert all(flies[n].endswith("felt nowhere, up again with it and on to another spot") for n in (2, 3, 4, 5))
    assert flies[6].endswith("felt nowhere, back onto the box (the last of the 6 spots a run tries)")
    assert "SET-DOWN CONTACT" not in out and "DONE" not in out
    assert out.rstrip().endswith("the lid is back on the box, and the arm is home")


def _budget(monkeypatch, left_s, go=False, s=90.0):
    """The engine's hand-over (node.budget_env), as if the run had `left_s` of its `s` seconds left."""
    import time

    told = {"s": s, "from": time.monotonic() - (s - left_s), "go": go}
    monkeypatch.setenv(rlo.RUN_BUDGET_ENV, json.dumps(told))


@pytest.mark.parametrize("left_s, goes_on", [(60.0, True), (45.0, True), (25.0, False), (5.0, False)])
def test_no_further_spot_is_started_that_the_runs_time_does_not_hold_with_the_way_back_to_the_box(
    ctx, monkeypatch, tmp_path, capsys, left_s, goes_on
):
    """The engine stops a run at its budget with a Ctrl+C wherever it is (node.RUN_BUDGET_S), and a run
    cut with the lid in its fingers is the arm holding it. A further spot is started only where the
    run's time still holds it and, should it feel no table either, the way back to the box, the release
    and home after it: 35 s by the records (CHOICE_S + SPOT_S + BOX_S + END_S). With less left, the lid
    goes back on the box at once — even where that is less than the way back takes: the box is still
    the best there is — and the run says how much time was left."""
    assert rlo.CHOICE_S + rlo.SPOT_S + rlo.BOX_S + rlo.END_S == pytest.approx(35.0)
    _budget(monkeypatch, left_s)
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY, scene=_table(), setdown=["never", "met"]
    )
    (left,) = (r for r in rows if r["leg"] == "lid_spot_left")
    assert left["time_left_s"] == pytest.approx(left_s, abs=5.0)  # (less what the run took up to there)
    out = capsys.readouterr().out
    if goes_on:
        assert (
            code == 0 and _after_first_touch(rec) == ["lid:back-up", *CARRY, *HOME] and left["next"] == "another spot"
        )
        assert "s of the run's time left)" in out
        return
    assert code == 1 and "place:open" not in rec.events and left["next"] == "the box"
    assert _after_first_touch(rec) == [
        "lid:back-up",
        "lid:back-over",
        "lid:put-back",
        "lid:release",
        "lid:rise-off",
        "home",
    ]
    assert "s of the run's time are left: another spot and, should it feel no table either, the way back" in out
    assert "take 35 (12 and 23, on record)" in out
    assert out.rstrip().endswith("the lid is back on the box, and the arm is home")


def _later_reach(monkeypatch, says, on_check=lambda: None):
    """Every reach check of a spot after the first (it is asked with the lid's own joints) answers
    `says`, `on_check()` called at each; the first spot's are the real ones. Returns the checks made."""
    real, made = rlo.drop_reachable, []

    def reach(ctx_, drop, q, seeds=(), held=None, stands=None):
        if held is None:
            return real(ctx_, drop, q, seeds)
        on_check()
        made.append(drop)
        return says

    monkeypatch.setattr(rlo, "drop_reachable", reach)
    return made


def _later_plans(monkeypatch):
    """rlo.plan_carry, the plans asked for a spot after the first kept ((spot, kwargs)); all planned as ever."""
    real, asked = rlo.plan_carry, []

    def plan(ctx_, cfg_, q, drop, **kw):
        if kw:
            asked.append((drop, kw))
        return real(ctx_, cfg_, q, drop, **kw)

    monkeypatch.setattr(rlo, "plan_carry", plan)
    return asked


def test_the_clock_is_asked_all_through_the_choice_and_the_box_is_taken_with_its_time_still_in_hand(
    ctx, monkeypatch, tmp_path, capsys
):
    """The choice of the next spot is bounded by the run's clock, not only its beginning: where the hand
    reaches spot after spot of the table no better than the last (each reach check 0.4 s of the run's time
    here), the clock is asked before every one of them, and the moment what is left no longer holds a spot
    and the way back to the box after it (31 s) the choosing stops and the box is taken — with the box's
    own 23 s still in hand. As first written the tiers were gone through with no clock at all (~100 reach
    checks, 4 plans: ~17 s at the robot's rates) and the put-back began with whatever was left: the test
    that stood here pinned one begun with 20 s (the review of 2026-11-19)."""
    clock = {"left": 40.0}
    monkeypatch.setattr(rlo.RunClock, "left_s", lambda self: clock["left"])
    checks = _later_reach(monkeypatch, "no hand reaches it", lambda: clock.update(left=clock["left"] - 0.4))
    plans = _later_plans(monkeypatch)
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY, scene=_table(), setdown=["never", "met"]
    )
    assert code == 1 and "place:open" not in rec.events and len(rec.setdowns) == 1
    assert _after_first_touch(rec) == [
        "lid:back-up",
        "lid:back-over",
        "lid:put-back",
        "lid:release",
        "lid:rise-off",
        "home",
    ]
    gate = rlo.SPOT_S + rlo.BOX_S + rlo.END_S
    assert gate == pytest.approx(31.0) and len(checks) == 23 and plans == []  # 40 s down to under 31, 0.4 s a check
    assert gate - 0.4 - 1e-9 <= clock["left"] < gate  # stopped at the first check the clock said no to
    assert clock["left"] >= rlo.BOX_S + rlo.END_S  # the put-back begins with what it needs, and its margin
    spots = [r for r in rows if r["leg"] == "lid_spot"]
    assert spots[1]["xyz"] is None and spots[1]["timed_out"] is True and spots[1]["time_left_s"] < gate
    out = capsys.readouterr().out
    assert "and no other spot is tried: the run's time ran out while one was looked for (31 s are left;" in out
    assert out.rstrip().endswith("the lid is back on the box, and the arm is home")
    view = _flight_view(out, code)  # the flight view says why the box was taken
    i = _view_line(view, r" *\d+ SET DOWN  FAIL  spot 1: no table felt, lid kept")
    assert view[i + 1] == "             next  putting the lid back on the box (no time left for another spot)"
    assert view[i + 2].endswith("PUT BACK  ok    lid back on the box, arm home")


def test_a_choice_the_clock_has_no_time_for_asks_no_reach_and_no_plan(ctx, monkeypatch, tmp_path, capsys):
    """The time left as the spot is left holds another (40 s), and by the time the arm stands over it the
    clock says 5: not one reach check is begun and not one plan asked (98 reach checks were, in the review's
    probe of the code before it) — the box at once, since nothing else is possible."""
    left = iter([40.0, 40.0, 40.0])  # before the set-down flies, as the spot is left, and for its row
    monkeypatch.setattr(rlo.RunClock, "left_s", lambda self: next(left, 5.0))
    checks = _later_reach(monkeypatch, None)
    plans = _later_plans(monkeypatch)
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY, scene=_table(), setdown=["never", "met"]
    )
    assert code == 1 and checks == [] and plans == [] and "spot-look" not in rec.events
    assert _after_first_touch(rec) == [
        "lid:back-up",
        "lid:back-over",
        "lid:put-back",
        "lid:release",
        "lid:rise-off",
        "home",
    ]
    spots = [r for r in rows if r["leg"] == "lid_spot"]
    assert spots[1]["timed_out"] is True and spots[1]["searched"] == []
    assert "the run's time ran out while one was looked for (5 s are left;" in capsys.readouterr().out


@pytest.mark.parametrize("gone_in, plans", [("a reach check", 0), ("a plan", 1)])
def test_the_time_is_asked_before_each_plan_too(ctx, monkeypatch, tmp_path, gone_in, plans):
    """A reach check is an IK walk and a plan can take a second and more. The time runs out during the
    reach check that passes a later spot: its carry is not planned. It runs out during a plan the planner
    refuses: no second spot is looked at, no second plan asked."""
    clock = {"left": 40.0}
    monkeypatch.setattr(rlo.RunClock, "left_s", lambda self: clock["left"])
    if gone_in == "a reach check":
        _later_reach(monkeypatch, None, lambda: clock.update(left=30.0))
    real, asked = rlo.plan_carry, []

    def plan(ctx_, cfg_, q, drop, **kw):
        if not kw:
            return real(ctx_, cfg_, q, drop)
        asked.append(drop)
        clock["left"] = 30.0
        return None, "the carry (walked: no; planned: no)"

    monkeypatch.setattr(rlo, "plan_carry", plan)
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY, scene=_table(), setdown=["never", "met"]
    )
    assert code == 1 and len(asked) == plans and len(rec.setdowns) == 1
    spots = [r for r in rows if r["leg"] == "lid_spot"]
    assert spots[1]["timed_out"] is True and spots[1]["searched"] == ["the tiers"]


@pytest.mark.parametrize("flight, goes_on", [(3.0, True), (20.2, False)])
def test_the_time_before_a_carry_is_sent_is_asked_for_that_carrys_own_flight(
    ctx, monkeypatch, tmp_path, capsys, flight, goes_on
):
    """The time is asked once more before the chosen spot's carry is sent — for the flight this carry is
    as the runner will fly it (flight_s) and the way back up after it (SPOT_UP_S), not for the 8 s a spot
    takes on record: with 45 s left a carry of 3 s goes (5.5 and the box's 23), one of 20 s does not (it
    and the way back to the box need 45.7), and the lid goes back on the box from over the spot it was
    left at. The line before the set-down that does fly says what is known as it is said: the run is
    timed, 45 s are left now, and another spot follows only where the time then still holds one."""
    monkeypatch.setattr(rlo.RunClock, "left_s", lambda self: 45.0)
    monkeypatch.setattr(rlo, "flight_s", lambda runner, legs: flight if len(legs) == 3 else 1.0)
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY, scene=_table(), setdown=["never", "met"]
    )
    spots = [r for r in rows if r["leg"] == "lid_spot"]
    assert spots[1]["flight_s"] == pytest.approx(flight) and spots[1]["tier"] is not None
    out = capsys.readouterr().out
    if goes_on:
        assert code == 0 and spots[1]["xyz"] is not None and len(rec.setdowns) == 2
        (line,) = (text for text in out.splitlines() if text.startswith("[press_demo] SPOT 2, ONE MOTION:"))
        assert line.endswith(
            "felt nowhere, up again with it, and on to another spot only where the run's time then still holds "
            "one and the way back to the box after it (45 s are left now, 35 are asked then), else back onto the box"
        )
        return
    assert code == 1 and spots[1]["xyz"] is None and len(rec.setdowns) == 1
    assert _after_first_touch(rec) == [
        "lid:back-up",
        "lid:back-over",
        "lid:put-back",
        "lid:release",
        "lid:rise-off",
        "home",
    ]
    assert (
        "and no other spot is tried: 45 s of the run's time are left" in out
        and "take 46 (23 by this spot's own carry and 23, on record)" in out
    )
    assert "SPOT 2, ONE MOTION:" not in out


def test_the_runner_says_how_long_a_carry_flies_before_it_is_sent(ctx, monkeypatch, tmp_path):
    """flight_s: the carry's legs as the runner will re-time them (its preview's own sum), None where
    that cannot be told."""
    from conftest import leg as make_leg

    from box_opening.runtime.runner import Runner

    runner = Runner(ctx.client, log_dir=tmp_path)
    a = make_leg("lid:carry", start=[0.0] * 7, end=[0.3] * 7, speed=1.0)
    b = make_leg("place:down", start=[0.3] * 7, end=[0.35] * 7, speed=0.55)
    one, both = rlo.flight_s(runner, [a]), rlo.flight_s(runner, [a, b])
    assert one is not None and 0.0 < one < both < 30.0
    assert rlo.flight_s(runner, [SimpleNamespace(kind=a.kind, traj=None, guard=None)]) is None


def test_the_run_reads_its_time_from_what_the_engine_handed_it_and_from_its_own_go(tmp_path, capsys):
    """RunClock: the engine's hand-over (node.budget_env) is the seconds the run has, the time they count
    from on the host's monotonic clock (the engine and the run are two processes on one host: no step of
    the wall clock moves it), and whether they start again at each go (a goal sent by hand:
    node.Capability._follow counts such a run's budget from its last go). The go's time is the run's own
    record's — the 'line_up' row of stage 'go' press_demo.line_up_watch writes as it hears one, stamped by
    the wall clock and put on the engine's clock by what the two differ by; never later than now, never
    before the budget's own start. Nothing handed: nothing times the run."""
    now, wall = [1000.0], [5000.0]

    def clock(env, runner=None):
        return rlo.RunClock(runner, env=env, now=lambda: now[0], wall=lambda: wall[0])

    nothing = clock({})
    assert nothing.left_s() is None and nothing.unreadable is None
    told = {"s": 90.0, "from": 940.0, "go": False}
    assert clock({rlo.RUN_BUDGET_ENV: json.dumps(told)}).left_s() == pytest.approx(30.0)
    now[0], wall[0] = 1040.0, 5040.0
    assert clock({rlo.RUN_BUDGET_ENV: json.dumps(told)}).left_s() == pytest.approx(-10.0)  # past it: said as it is
    wall[0] = 9999.0  # the wall clock stepped: nothing of the budget moves with it
    assert clock({rlo.RUN_BUDGET_ENV: json.dumps(told)}).left_s() == pytest.approx(-10.0)
    # a goal sent by hand: from its last go, where the run has had one
    now[0], wall[0] = 1000.0, 5000.0
    record = tmp_path / "run-x.jsonl"
    runner = SimpleNamespace(_log_path=record)
    by_hand = {rlo.RUN_BUDGET_ENV: json.dumps(dict(told, go=True))}
    assert clock(by_hand, SimpleNamespace(_log_path=None)).left_s() == pytest.approx(30.0)  # no record yet
    assert clock(by_hand, runner).left_s() == pytest.approx(30.0)  # ... or none that can be read: from its start

    def rows(*stamps):
        record.write_text("".join(json.dumps(r) + "\n" for r in stamps) + "not json\n")

    rows(
        {"t": 4950.0, "leg": "line_up", "kind": "note", "stage": "go"},
        {"t": 4960.0, "leg": "scene_fix", "kind": "note"},
        {"t": 4985.0, "leg": "line_up", "kind": "note", "stage": "go"},  # guided in the middle of the run: its last
        {"t": 4990.0, "leg": "line_up", "kind": "note", "stage": "stop"},
    )
    by_hand_clock = clock(by_hand, runner)
    assert by_hand_clock.left_s() == pytest.approx(75.0)  # the go 15 s ago on either clock
    # ... which moves nothing under the orchestrator's 120 s
    assert clock({rlo.RUN_BUDGET_ENV: json.dumps(told)}, runner).left_s() == pytest.approx(30.0)
    # a row written since is read (the record is read again only when it has grown: every reach check asks)
    now[0], wall[0] = 1010.0, 5010.0
    assert by_hand_clock.left_s() == pytest.approx(65.0)
    rows({"t": 5008.0, "leg": "line_up", "kind": "note", "stage": "go"}, {"t": 5009.0, "leg": "still", "kind": "note"})
    assert by_hand_clock.left_s() == pytest.approx(88.0)
    # a go dated after now (the wall clock stepped back since) is now; one before the budget began is its start
    rows({"t": 5030.0, "leg": "line_up", "kind": "note", "stage": "go"})
    assert clock(by_hand, runner).left_s() == pytest.approx(90.0)
    rows({"t": 4000.0, "leg": "line_up", "kind": "note", "stage": "go"})
    assert clock(by_hand, runner).left_s() == pytest.approx(20.0)


@pytest.mark.parametrize(
    "handed",
    [
        "",
        "ninety",
        "null",
        "0",
        "true",
        '"x"',
        "[90]",
        "{}",
        '{"s": 90.0}',
        '{"s": "90", "from": "1"}',
        '{"s": true, "from": 1}',
        '{"s": NaN, "from": 1}',
        '{"s": 1e999, "from": 1}',
        '{"s": 90, "from": null}',
        '{"s": 90, "from": 1, "go": "no"}',
    ],
)
def test_a_budget_that_is_there_and_cannot_be_read_fails_closed(handed, capsys):
    """Only an absent budget means nothing times the run (the command line's). One that is there and does
    not read as one means an engine IS timing this run and the run does not know by how much: its clock
    says nothing is left, so no further spot is ever started on it, and it says what it was handed. As
    first written every one of these read as "nothing handed", and such a run would have gone from spot to
    spot with the engine's Ctrl+C due (the review of 2026-11-19)."""
    clock = rlo.RunClock(env={rlo.RUN_BUDGET_ENV: handed}, now=lambda: 10.0)
    assert clock.left_s() == 0.0 and clock.unreadable is not None and repr(handed) in clock.unreadable
    assert "cannot be read" in capsys.readouterr().out
    search = rlo.SpotSearch(SimpleNamespace(), None, None, None, None, 0.0, 0.0)
    search.clock = clock
    assert "cannot be read" in search.no_more(0.0) and search.out_of_time()
    assert search.after_this() == ("box", "the run's time cannot be read")
    assert clock.known_s() is None  # the 0.0 is the rule's, not a time: never said or recorded as one
    # ... while a budget that reads, however little it leaves, is a budget
    assert rlo.RunClock(env={rlo.RUN_BUDGET_ENV: '{"s": -5, "from": 0}'}, now=lambda: 10.0).left_s() == -15.0
    assert rlo.RunClock(env={rlo.RUN_BUDGET_ENV: '{"s": 90, "from": 1}'}, now=lambda: 10.0).from_go is False


def test_a_budget_stamped_on_another_clock_is_one_that_cannot_be_read(capsys):
    """A budget that reads as one and counts from a time this run cannot have been started at on this
    host's monotonic clock: later than now (one stamped by time.time() read as 1.8e9 s left; one 60 s
    ahead as 150 s left of a 90 s budget), or older than the run's own seconds and an hour. Both were
    taken as they stood and failed OPEN — every spot tried, the engine's Ctrl+C due (the second review of
    2026-11-19). They are budgets that cannot be read: nothing is left, no further spot."""
    now = 5000.0

    def clock(since, s=90.0):
        told = json.dumps({"s": s, "from": since, "go": False})
        return rlo.RunClock(env={rlo.RUN_BUDGET_ENV: told}, now=lambda: now)

    for since, why in (
        (1.8e9, "after now on this host's monotonic clock"),  # time.time()'s
        (now + 60.0, "it counts from 60 s after now"),
        (now + 1.5, "after now"),
        (now - 90.0 - 3601.0, "no run is that old"),
    ):
        stamped = clock(since)
        assert stamped.left_s() == 0.0 and stamped.known_s() is None, since
        assert why in stamped.unreadable and "cannot be read" in capsys.readouterr().out
        search = rlo.SpotSearch(SimpleNamespace(), None, None, None, None, 0.0, 0.0)
        search.clock = stamped
        assert "cannot be read" in search.no_more(0.0) and search.out_of_time()
    assert (rlo.RUN_FROM_AHEAD_S, rlo.RUN_FROM_OLD_S) == (1.0, 3600.0)
    # the engine's rounding of its own clock, a run past its time, a long one: budgets, read as they stand
    assert clock(now + 0.0004).left_s() == pytest.approx(90.0, abs=1e-3)
    assert clock(now - 200.0).left_s() == pytest.approx(-110.0)
    assert clock(now - 90.0 - 3599.0).left_s() == pytest.approx(-3599.0)
    assert clock(now - 3000.0, s=-5.0).left_s() == pytest.approx(-3005.0)  # (a budget a line-up used up: its hour too)
    assert clock(now - 30.0).unreadable is None and clock(now - 30.0).known_s() == pytest.approx(60.0)


@pytest.mark.parametrize("handed", ["ninety", "the wall clock's"])
def test_a_run_whose_budget_cannot_be_read_takes_the_lid_to_no_second_spot(ctx, monkeypatch, tmp_path, capsys, handed):
    """The same, in the run — a budget that does not read, and one stamped by time.time(): its first
    set-down feels no table, and the lid goes back on the box. Nothing the run says or records gives a
    time it does not know: not "0 s left"."""
    import time

    if handed == "the wall clock's":
        handed = json.dumps({"s": 90.0, "from": round(time.time(), 3), "go": False})
    monkeypatch.setenv(rlo.RUN_BUDGET_ENV, handed)
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY, scene=_table(), setdown=["never", "met"]
    )
    assert code == 1 and len(rec.setdowns) == 1 and "place:open" not in rec.events
    assert _after_first_touch(rec) == [
        "lid:back-up",
        "lid:back-over",
        "lid:put-back",
        "lid:release",
        "lid:rise-off",
        "home",
    ]
    (left,) = (r for r in rows if r["leg"] == "lid_spot_left")
    assert left["next"] == "the box" and "cannot be read" in left["why_the_box"] and left["time_left_s"] is None
    out = capsys.readouterr().out
    # said before the set-down flew, too: no "on to another spot" promised of a run that will try none
    assert "felt nowhere, back onto the box (the run's time cannot be read)" in out
    assert "on to another spot" not in out and "0 s are left" not in out and "0 s of the run's time" not in out
    assert out.rstrip().endswith("the lid is back on the box, and the arm is home")
    view = _flight_view(out, code)
    i = _view_line(view, r" *\d+ SET DOWN  FAIL  spot 1: no table felt, lid kept")
    assert view[i + 1] == "             next  carrying the lid back to the box (the run's time cannot be read)"


def test_a_first_choice_ended_by_a_time_that_cannot_be_read_says_that_and_no_seconds(
    ctx, monkeypatch, tmp_path, capsys
):
    """No spot of the tiers has a carry that plans, and the run's time cannot be read: the wider search
    is not begun (nothing says the box's own time is in hand), the lid goes back on the box, and the
    put-back says why as it is known — the time cannot be read. It said "0 s are left", the rule's
    fail-closed answer taken for a time (the second review of 2026-11-19)."""
    monkeypatch.setenv(rlo.RUN_BUDGET_ENV, "ninety")
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY, refuse_carry=True
    )
    assert code == 1 and "lid:carry" not in rec.events
    (spot,) = (r for r in rows if r["leg"] == "lid_spot")
    assert spot["searched"] == ["the tiers"]
    out = capsys.readouterr().out
    assert "and no more of the table is looked at: the run's time cannot be read" in out
    assert "s are left" not in out
    assert out.rstrip().endswith("the lid is back on the box, and the arm is home")


def test_the_line_before_a_set_down_says_only_what_is_known_as_it_is_said(ctx, monkeypatch, tmp_path, capsys):
    """What follows a set-down that feels no table depends on the run's clock THEN, which is not known as
    the set-down is sent: the line predicted it from the planned flight, and a set-down that something
    stops early ends sooner than planned — with 36 s left it said "back onto the box" and the run, 36 s
    still in hand as the spot was left, went on to another (the second review of 2026-11-19). The line
    says the box only where that is certain (the last spot, a time that cannot be read); of a timed run,
    another spot only where the time then still holds one; of a run nothing times, another spot."""
    search = rlo.SpotSearch(SimpleNamespace(), None, None, None, None, 0.0, 0.0)
    search.clock = SimpleNamespace(unreadable=None, left_s=lambda: None)
    assert search.after_this() is None
    search.clock = SimpleNamespace(unreadable=None, left_s=lambda: 12.0)
    assert search.after_this() == ("time", 12.0)  # however little: what the clock says then is then's
    search.spots = rlo.SETDOWN_SPOTS
    assert search.after_this() == ("box", "the last of the 6 spots a run tries")
    carry = SimpleNamespace(over_m=0.008)
    assert rlo._touch_says(carry).endswith("felt nowhere, up again with it and on to another spot")
    assert rlo._touch_says(carry, ("box", "why")).endswith("felt nowhere, back onto the box (why)")
    timed = rlo._touch_says(carry, ("time", 36.4))
    assert timed.endswith(
        "felt nowhere, up again with it, and on to another spot only where the run's time then still holds one "
        "and the way back to the box after it (36 s are left now, 35 are asked then), else back onto the box"
    )
    # in the run: 36 s left all along, something holds the lid up at the first spot, and the lid goes on
    monkeypatch.setattr(rlo.RunClock, "left_s", lambda self: 36.0)
    code, rec, _chain, _rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY, scene=_table(), setdown=["held", "met"]
    )
    assert code == 0 and [how for _d, how in rec.setdowns] == ["held", "met"]
    out = capsys.readouterr().out
    said = [text for text in out.splitlines() if "ONE MOTION" in text and "felt nowhere" in text]
    assert len(said) == 2 and all(text.endswith(timed.split("felt nowhere, ")[1]) for text in said)
    assert "back onto the box (the run's time" not in out


def test_the_runs_own_line_up_writes_the_go_row_its_clock_reads():
    """The clock's go is press_demo.line_up_watch's own note: this pins the row it reads to the line
    that writes it (a renamed stage would silently count every by-hand run from its start)."""
    import inspect

    from box_opening.tasks import press_demo as pd

    assert 'runner.note("line_up", stage="go"' in inspect.getsource(pd.line_up_watch)


def test_a_spot_the_lid_was_left_at_is_set_aside_with_the_lids_own_footprint_round_it(ctx, monkeypatch):
    """SpotSearch.aside: a spot whose lid would lie on any of the place the lid was held over at a spot
    it was left at (their centres nearer than the lid is wide) is set aside for the run; spot_candidates
    passes such spots over, counted, never a try. Nothing left: nothing set aside."""
    from box_opening.runtime import approach

    monkeypatch.setattr(approach, "_chain", lambda: _Chain())
    bare = _bare(monkeypatch, ctx)
    search = rlo.SpotSearch(ctx, None, None, None, bare, -0.07, -0.07)
    offered = [d for d, _t, _s in rlo.spot_candidates(ctx, bare, -0.07, [0.0] * 7, reach=lambda *a: None)]
    first = offered[0]
    assert not search.aside(first)
    was = search.leave(first, "beside the box", "never", "the touch ended ...")
    assert (was.n, was.drop, was.verdict) == (1, first, "never") and "spot 1 at [" in was.says()
    wide = _lid_wide(ctx)
    near = ContainerPose(xyz=(first.xyz[0] + wide - 0.001, first.xyz[1], -0.07), yaw=0.0)
    far = ContainerPose(xyz=(first.xyz[0] + wide + 0.001, first.xyz[1], -0.07), yaw=0.0)
    assert search.aside(first) and search.aside(near) and not search.aside(far)
    passed = rlo.SpotsPassed()
    again = [
        d
        for d, _t, _s in rlo.spot_candidates(
            ctx, bare, -0.07, [0.0] * 7, reach=lambda *a: None, aside=search.aside, passed=passed
        )
    ]
    assert again and passed.aside >= 2  # straight down and leaning, at the least
    for d in again:
        assert math.hypot(d.xyz[0] - first.xyz[0], d.xyz[1] - first.xyz[1]) >= wide
    assert "within the footprint left at a spot where no table was felt" in passed.says()
    # the same rules in the same order, less what is set aside: every spot offered before and not set aside
    # is offered again, in the order it had (a tier that lost spots goes further down its own list)
    keys = [rlo._spot_key(d) for d in offered]
    now = [rlo._spot_key(d) for d in again]
    assert [k for k in now if k in keys] == [k for k, d in zip(keys, offered, strict=True) if not search.aside(d)]


def test_past_every_tier_the_whole_measured_table_is_gone_through_nearest_the_hand_first(ctx, monkeypatch):
    """wider_candidates, for where no tier's spot has a carry that plans: every place on the table the
    cameras measured — not only round the box, not only the first few of a tier — SPOT_APART_M apart,
    the nearest where the hand holds the lid first, each through what every spot must pass (the
    no-land radius, the measured table under the whole lid and the hand, what is set aside), straight
    down first, then leaning. The reach of every place that passes is asked; where those are more than
    WHOLE_TABLE_TRIES, of that many of them evenly from the nearest to the farthest: the whole of the
    table, not the corner of it nearest the hand."""
    from box_opening.runtime import approach

    chain = _Chain()  # its hand stands at [0.5, -0.4]
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    hand = (0.5, -0.4)
    bare = _bare(monkeypatch, ctx)  # 1.2 m across, all of it seen bare
    bx, by = float(ctx.cpose.xyz[0]), float(ctx.cpose.xyz[1])
    tiers = [d for d, _t, _s in rlo.spot_candidates(ctx, bare, -0.07, [0.0] * 7, reach=lambda *a: None)]
    far_tier = max(math.hypot(d.xyz[0] - bx, d.xyz[1] - by) for d in tiers)
    assert far_tier < 0.46  # the tiers look no further from the box than this (NEAREST_MAX_M)
    passed = rlo.SpotsPassed()
    wide = list(rlo.wider_candidates(ctx, bare, -0.07, [0.0] * 7, reach=lambda *a: None, passed=passed))
    assert {tier for _d, tier, _s in wide} == {"the whole table"} == {rlo.WHOLE_TABLE}
    assert "anywhere on the table the cameras measured" in wide[0][2]
    straight = [d for d, _t, _s in wide if rlo._spot_lean(d) == 0.0]
    assert len(straight) == rlo.WHOLE_TABLE_TRIES == 48 and [d for d, _t, _s in wide[:48]] == straight
    out = [math.hypot(d.xyz[0] - hand[0], d.xyz[1] - hand[1]) for d in straight]
    assert out == sorted(out)  # the nearest the hand first
    # the whole table: from one side of what the map saw to the other, far past where the tiers look
    x0, x1, y0, y1 = rlo.clear_table.table_bounds(bare)
    xs, ys = [d.xyz[0] for d in straight], [d.xyz[1] for d in straight]
    assert min(xs) < x0 + 0.3 and max(xs) > x1 - 0.3 and min(ys) < y0 + 0.3 and max(ys) > y1 - 0.3
    assert max(math.hypot(d.xyz[0] - bx, d.xyz[1] - by) for d in straight) > far_tier
    every = rlo._table_points(ctx, bare, -0.07, near=hand)
    assert len(every) > 10 * 48 and passed.no_land > 0  # of some 700 places, the box's own among them, passed over
    for i, d in enumerate(straight):
        assert type(d) is ContainerPose and rlo.box_clearance(ctx, d) >= 0.0
        assert rlo.spot_off_table(ctx, bare, d, [0.0] * 7) is None
        for e in straight[:i]:
            assert math.hypot(d.xyz[0] - e.xyz[0], d.xyz[1] - e.xyz[1]) >= rlo.clear_table.SPOT_APART_M - 1e-9
    # then the same table with the hand leaning SETDOWN_LEAN_DEG; the fake chain's hand does not hold the
    # lid level, so nothing past that lean lands as it must (landing_tilt_deg): none of those is offered
    leaning = [d for d, _t, _s in wide[48:]]
    assert len(leaning) == 48 and all(
        isinstance(d, rlo.LeanSpot) and d.lean_deg == rlo.SETDOWN_LEAN_DEG for d in leaning
    )
    assert "past every spot straight down: the hand leaning 20 deg" in wide[48][2] and passed.tilted > 0
    # a table with no more places than a pass asks the reach of: every one of them that passes, in order
    small = _standing_ring((bx, by), 5.0, 6.0)
    X, Y = small.centres()
    seen = (np.abs(X - (bx + 0.35)) < 0.18) & (np.abs(Y - by) < 0.18)  # a patch of table 0.36 m across
    small = type(small)(small.x0, small.y0, small.cell, small.tall, seen, small.off, small.box_xy)
    pts = rlo._table_points(ctx, small, -0.07, near=hand)
    roomless = rlo.SpotsPassed()
    few = rlo.wider_candidates(ctx, small, -0.07, [0.0] * 7, reach=lambda *a: None, passed=roomless)
    few = [d for d, _t, _s in few if rlo._spot_lean(d) == 0.0]
    assert 4 < len(few) < len(pts) < rlo.WHOLE_TABLE_TRIES  # (near the patch's edge the hand would hang past it)

    def fits(d):
        """On the measured table, the whole lid and the hand; and with a clear spot's room there."""
        lid_r, lid_h = 0.5 * max(ctx.model.lid_dims[:2]), ctx.model.lid_dims[2]
        room = rlo.clear_table.clearance(small, d.xyz[:2], rlo.setdown_poses(ctx, d)[0], lid_r, lid_h)[0]
        return rlo.spot_off_table(ctx, small, d, [0.0] * 7) is None and room >= 0.0

    assert [d.xyz for d in few] == [d.xyz for d in pts if fits(d)]
    # ... the room a 'clear' spot has, which the whole table's places were not asked at all (the review of
    # 2026-11-19): on the measured table by the gate, and nearer what was not seen than the lid or the hand
    # may come — passed over, counted, said
    gated = [d for d in pts if rlo.spot_off_table(ctx, small, d, [0.0] * 7) is None]
    assert roomless.tight == len(gated) - len(few) > 0
    assert "%d with less room for the lid and the hand than a spot must have" % roomless.tight in roomless.says()
    assert rlo._thinned(list(range(10)), 48) == list(range(10))
    assert rlo._thinned(list(range(96)), 48) == list(range(0, 96, 2))
    # a spot offered before is not offered again, and what is set aside is passed over here too
    skip = {rlo._spot_key(straight[0])}
    aside = lambda d: d.xyz[:2] == straight[1].xyz[:2]  # noqa: E731
    got = [
        d
        for d, _t, _s in rlo.wider_candidates(
            ctx, bare, -0.07, [0.0] * 7, reach=lambda *a: None, skip=skip, aside=aside
        )
    ]
    assert straight[0].xyz not in [d.xyz for d in got if rlo._spot_lean(d) == 0.0]
    assert straight[1].xyz not in [d.xyz for d in got]
    # the run's time, asked before each reach check, ends the search
    asked = []
    stop = lambda: len(asked) >= 3  # noqa: E731
    reach = lambda c, d, q, seeds=(): asked.append(d) or "no"  # noqa: E731
    assert list(rlo.wider_candidates(ctx, bare, -0.07, [0.0] * 7, reach=reach, stop=stop)) == [] and len(asked) == 3
    # no map: no place on any table
    assert list(rlo.wider_candidates(ctx, None, -0.07, [0.0] * 7, reach=lambda *a: None)) == []


def test_a_place_with_something_standing_in_the_lids_way_to_it_is_passed_over(ctx, monkeypatch):
    """The lid's way to a wider spot, and to any spot from one it was left at, is the straight line
    from where the hand holds it, asked against what the map shows standing taller than the lid is
    carried: the planner is told nothing of what stands on the table, and a 'clear' spot's room holds
    the way from the box only. A wall between the hand and a place: passed over, counted, never a try."""
    from box_opening.perception.clear_table import TableMap
    from box_opening.runtime import approach

    chain = _Chain()  # its hand stands at [0.5, -0.4]
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    monkeypatch.setattr(rlo, "lid_spots", lambda *a, **k: ([], "none judged"))
    hand = (0.5, -0.4)
    wall_y = hand[1] + 0.25
    ctx.cpose = ContainerPose(xyz=(hand[0], hand[1] + 0.45, ctx.cpose.xyz[2]), yaw=0.0)  # the box past the wall
    # a table seen bare 1.2 m across round the hand, a wall 0.30 m tall right across it
    bare = _standing_ring(hand, 5.0, 6.0)
    _X, Y = bare.centres()
    tall = np.where(np.abs(Y - wall_y) < 0.021, 0.30, np.nan)
    walled = TableMap(bare.x0, bare.y0, bare.cell, tall, bare.seen, bare.off, bare.box_xy)

    def behind(d):
        return d.xyz[1] > wall_y

    passed = rlo.SpotsPassed()
    wide = [
        d for d, _t, _s in rlo.wider_candidates(ctx, walled, -0.07, [0.0] * 7, reach=lambda *a: None, passed=passed)
    ]
    assert wide and passed.blocked > 0 and not any(behind(d) for d in wide)
    assert "with something standing in the lid's way there" in passed.says()
    # the tiers ask it only of a carry that starts from a spot the lid was left at (`way`)
    tiers = [d for d, _t, _s in rlo.spot_candidates(ctx, walled, -0.07, [0.0] * 7, reach=lambda *a: None)]
    asked = [d for d, _t, _s in rlo.spot_candidates(ctx, walled, -0.07, [0.0] * 7, reach=lambda *a: None, way=True)]
    assert any(behind(d) for d in tiers) and asked and not any(behind(d) for d in asked)


def _leaning_grip(ctx, at_xy, lean_deg):
    """A chain whose hand holds the lid leaning `lean_deg` back toward the arm, facing `at_xy`."""
    from box_opening.perception.d405 import quat_to_mat

    return _HandAt(0.45, R=np.asarray(quat_to_mat(*rp.reach_quat(ctx.model, at_xy, lean_deg)), float))


@pytest.mark.parametrize("grip_lean, offered", [(0.0, False), (9.0, False), (11.0, True), (30.0, True)])
def test_the_hand_leans_past_20_deg_only_last_and_only_where_the_lid_still_lands_as_at_20(
    ctx, monkeypatch, grip_lean, offered
):
    """SETDOWN_LEAN_MAX_DEG: past every spot straight down and at SETDOWN_LEAN_DEG the hand leans as far
    as the press's own look pose does (reach_press.REACH_TILTS_DEG's first, 30 deg). 20 was chosen for
    how the lid lands — tilted 20 deg at most, its far edge falling 36 mm — and that is kept: a hand
    leaning 30 deg is offered only where the lid, level as the grip took it, lands tilted no more than
    20 (landing_tilt_deg): where the grip itself leaned 10 deg or more the same way, as it does over
    the far boxes that need this."""
    from box_opening.runtime import approach

    assert rlo.SETDOWN_LEAN_MAX_DEG == 30.0 == rp.REACH_TILTS_DEG[0] and rlo.SETDOWN_LEAN_DEG == 20.0
    bare = _bare(monkeypatch, ctx)
    spot = rlo._table_points(ctx, bare, -0.07)[0]
    chain = _leaning_grip(ctx, spot.xyz[:2], grip_lean)
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    deep = rlo.LeanSpot(xyz=spot.xyz, yaw=0.0, lean_deg=rlo.SETDOWN_LEAN_MAX_DEG)
    assert rlo.landing_tilt_deg(ctx, deep, [0.0] * 7) == pytest.approx(30.0 - grip_lean, abs=1e-3)
    asked = []

    def only_past_20(ctx_, d, q, seeds=()):
        asked.append(rlo._spot_lean(d))
        return None if rlo._spot_lean(d) > rlo.SETDOWN_LEAN_DEG else "no hand reaches its carry height"

    passed = rlo.SpotsPassed()
    got = next(rlo.wider_candidates(ctx, bare, -0.07, [0.0] * 7, reach=only_past_20, passed=passed), None)
    assert asked[0] == 0.0 and 20.0 in asked and asked == sorted(asked)  # straight down, then 20, and only then 30
    if not offered:
        assert got is None and 30.0 not in asked and passed.tilted > 0
        assert "where the hand leaning 30 deg would land the lid tilted more than 20" in passed.says()
        return
    drop, tier, says = got
    assert isinstance(drop, rlo.LeanSpot) and drop.lean_deg == 30.0
    assert rlo.landing_tilt_deg(ctx, drop, [0.0] * 7) <= 20.5
    assert "past every spot at 20 deg too: the hand leaning 30 deg" in says
    # ... and no tier of spot_candidates ever leans past 20
    assert all(
        rlo._spot_lean(d) <= 20.0
        for d, _t, _s in rlo.spot_candidates(ctx, bare, -0.07, [0.0] * 7, reach=lambda *a: None)
    )


class _HandAtQ(_Chain):
    """A chain whose tool stands at its first three joints' values (x, y, z), the hand leaning the
    fourth's degrees back toward the arm, facing `facing` (reach_press.reach_quat)."""

    def __init__(self, model, facing):
        super().__init__()
        self.model, self.facing = model, facing

    def fk(self, q):
        from box_opening.perception.d405 import quat_to_mat

        R = np.asarray(quat_to_mat(*rp.reach_quat(self.model, self.facing, float(q[3]))), float)
        return R, np.array([float(q[0]), float(q[1]), float(q[2])])


def test_a_carry_that_would_bring_the_lid_under_the_top_it_came_off_over_the_canister_is_refused(ctx, monkeypatch):
    """lid_low_over_box: leaning 30 deg the chair study's carry "put the lid below the rim over the
    canister" in 6 rows (SETDOWN_LEAN_DEG's note: why 20 was chosen), and a carry from one spot to the
    next may cross the box, which stands open by then. Anywhere on such a carry's way where the lid's
    disc stands over the box, its lowest point stays over the top it came off, or the carry is refused
    (plan_carry: asked of every carry to a hand leaning past 20 deg, and wherever `over_box` says)."""
    from conftest import leg as make_leg

    from box_opening.runtime import approach
    from box_opening.tasks.press_contact import pads_z
    from box_opening.tasks.press_demo import container_half

    m = ctx.model
    bx, by = float(ctx.cpose.xyz[0]), float(ctx.cpose.xyz[1])
    chain = _HandAtQ(m, (bx, by))
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    top = float(ctx.cpose.xyz[2]) + float(m.dims[2])
    lid_r, lid_h = 0.5 * max(m.lid_dims[:2]), float(m.lid_dims[2])
    held = [bx, by, 0.5, 0.0, 0.0, 0.0, 0.0]  # the lid hangs level in a hand straight down
    tool_over_pads = 0.5 - pads_z(chain, held)

    def at(x, y, low):
        """Joints with the level lid's lowest point `low` over the box's top, its centre at (x, y)."""
        return [x, y, top + low + lid_h - rlo.GRIP_BELOW_LID_M + tool_over_pads, 0.0, 0.0, 0.0, 0.0]

    # the lid hangs LID_HANG_M lower than the model's cap puts it (the 19 set-downs flown to their end met the
    # table 13 mm over the plan in the mean): every height here is asked of the lid that much lower. The
    # check had no margin at all (the review of 2026-11-19)
    hang = rlo.LID_HANG_M
    assert hang == pytest.approx(0.013)
    assert rlo.lid_low_over_box(ctx, [at(bx, by, 0.04), at(bx + 0.3, by, 0.04)], held) is None  # as a hover holds it
    assert rlo.lid_low_over_box(ctx, [at(bx, by, hang + 0.001)], held) is None
    assert "would pass 12 mm under the top it came off" in rlo.lid_low_over_box(ctx, [at(bx, by, 0.001)], held)
    why = rlo.lid_low_over_box(ctx, [at(bx + 0.3, by, 0.04), at(bx, by, hang - 0.010), at(bx - 0.3, by, 0.04)], held)
    assert why == "the lid's lowest point would pass 10 mm under the top it came off, over the open canister"
    # between its waypoints too: two of them clear of the box either side, the straight joint line between them over it
    low = hang - 0.010
    assert rlo.lid_low_over_box(ctx, [at(bx - 0.3, by, low), at(bx + 0.3, by, low)], held) == why
    edge = container_half(m) + lid_r
    assert rlo.lid_low_over_box(ctx, [at(bx + edge - 0.005, by, low)], held) is not None  # its rim over the box's
    assert rlo.lid_low_over_box(ctx, [at(bx + edge + 0.005, by, low)], held) is None  # beside the box: not asked
    # the lid's top as the fingers took it, where that stands higher than the cameras' top of the box
    ctx.lid_lifted = SimpleNamespace(q_grip=at(bx, by, 0.030 - lid_h))  # gripped with its top 30 mm over theirs
    assert rlo.lid_low_over_box(ctx, [at(bx, by, hang + 0.020)], held) is not None
    assert rlo.lid_low_over_box(ctx, [at(bx, by, hang + 0.031)], held) is None
    ctx.lid_lifted = None
    # a hand leaning 30 deg: the lid, level as gripped, hangs its low edge lower. Where a leaning set-down
    # has the lid's lowest point meet a table at the height of the box's top (setdown_poses' own answer for
    # such a spot, over the box), that is exactly where this draws the line
    over = rlo.LeanSpot(xyz=(bx, by, top), yaw=0.0, lean_deg=30.0)
    _quat, contact, _hover = rlo.setdown_poses(ctx, over, held=held)
    assert rlo.lid_low_over_box(ctx, [[contact[0], contact[1], contact[2] + hang, 30.0, 0.0, 0.0, 0.0]], held) is None
    lower = [contact[0], contact[1], contact[2] + hang - 0.005, 30.0, 0.0, 0.0, 0.0]
    assert "would pass 5 mm under the top it came off" in rlo.lid_low_over_box(ctx, [lower], held)
    # plan_carry asks it of a hand leaning past 20 deg and wherever over_box says, and of no other carry
    monkeypatch.setattr(approach, "_chain", lambda: _Chain())
    cfg = load_press_demo(PINK)
    asked = []
    monkeypatch.setattr(rlo, "lid_low_over_box", lambda ctx_, pts, held_: asked.append(list(held_)) or "too low")

    def plan(ctx_, state, name, target, world, speed):
        lg = make_leg(name, start=list(state.joints), end=[v + 0.05 for v in state.joints], speed=speed, world=world[0])
        lg.target = target
        return lg, state

    monkeypatch.setattr(rp, "_plan_motion", plan)
    drop = ContainerPose(xyz=(0.40, -0.30, -0.07), yaw=0.0)
    q = [0.1] * 7
    carry, why = rlo.plan_carry(ctx, cfg, q, drop)
    assert carry is not None and why is None and asked == []  # the first spot's carry: as it was
    carry, why = rlo.plan_carry(ctx, cfg, q, rlo.LeanSpot(xyz=drop.xyz, yaw=0.0, lean_deg=20.0))
    assert carry is not None and asked == []
    # ... and where the planner's path is refused for it, the arm's own walk is tried before the spot is
    # refused (this chain has none)
    carry, why = rlo.plan_carry(ctx, cfg, q, drop, held=[0.2] * 7, over_box=True)
    assert carry is None and why == "the carry (too low; walked: no turning walk modelled)" and asked == [[0.2] * 7]
    carry, why = rlo.plan_carry(ctx, cfg, q, rlo.LeanSpot(xyz=drop.xyz, yaw=0.0, lean_deg=30.0))
    assert carry is None and why == "the carry (too low; walked: no turning walk modelled)"
    assert asked[-1] == q  # held: where the carry starts


def test_from_a_spot_left_the_next_is_chosen_with_the_lid_as_the_grip_took_it_and_kept_off_the_canister(
    ctx, monkeypatch, tmp_path
):
    """The first carry begins where the clear ends, the hand as the grip had it: the lid hangs level
    there, and setdown_poses takes the carry's start for that. A carry from a spot the lid was left at
    begins with the hand as that set-down had it (turned to the spot, leaning): the joints the lid hangs
    level at are still the first carry's start (SpotSearch.held), handed to the reach checks and to
    plan_carry, with over_box: such a carry's way may cross the open canister."""
    real_plan, real_reach, plans, reaches = rlo.plan_carry, rlo.drop_reachable, [], []

    def plan(ctx_, cfg_, q, drop, **kw):
        plans.append((list(q), kw))
        return real_plan(ctx_, cfg_, q, drop, **kw)

    def reach(ctx_, drop, q, seeds=(), held=None, stands=None):
        reaches.append((list(q), held, stands))
        return real_reach(ctx_, drop, q, seeds, held=held, stands=stands)

    monkeypatch.setattr(rlo, "plan_carry", plan)
    monkeypatch.setattr(rlo, "drop_reachable", reach)
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY, scene=_table(), setdown=["never", "met"]
    )
    assert code == 0 and len(plans) == 2
    clear_end = rec.legs["lid:clear"].goal_joints
    (q1, kw1), (q2, kw2) = plans
    assert q1 == pytest.approx(clear_end) and kw1 == {}  # the first spot's: planned as it always was
    assert q2 != pytest.approx(q1) and kw2["over_box"] is True and kw2["held"] == pytest.approx(clear_end)
    # ... and asked what a later carry is asked (its own path's height and what stands under it, a leaning
    # hand's way down), the lid as the grip took it there too, over the table the set-down goes by
    way = kw2["way"]
    assert (
        isinstance(way, rlo.LaterWay) and way.held == pytest.approx(clear_end) and way.table_z == pytest.approx(TABLE_Z)
    )
    first = [(held, stands) for q, held, stands in reaches if q == pytest.approx(q1)]
    later = [(held, stands) for q, held, stands in reaches if q == pytest.approx(q2)]
    assert first and all(h is None and st is None for h, st in first)
    assert later and all(h == pytest.approx(clear_end) and st is not None for h, st in later)


def _no_other_spot_on_the_first_map(monkeypatch):
    """The tiers and the wider search give nothing for a carry from a spot left, on the first map they
    are asked about: the map in hand has no other spot. Returns the maps asked about, in order."""
    real, maps = rlo.spot_candidates, []

    def tiers(ctx_, tmap, *a, **k):
        if k.get("aside") is not None:
            if not any(tmap is seen for seen in maps):
                maps.append(tmap)
            if tmap is maps[0]:
                return iter(())
        return real(ctx_, tmap, *a, **k)

    real_wider = rlo.wider_candidates
    monkeypatch.setattr(rlo, "spot_candidates", tiers)
    monkeypatch.setattr(
        rlo,
        "wider_candidates",
        lambda ctx_, tmap, *a, **k: iter(()) if maps and tmap is maps[0] else real_wider(ctx_, tmap, *a, **k),
    )
    return maps


def test_where_the_map_in_hand_has_no_other_spot_the_wrist_looks_from_where_the_arm_stands_and_the_search_goes_again(
    ctx, monkeypatch, tmp_path, capsys
):
    """The spot chosen ahead was chosen on the scene camera's map alone, made before the press from 14 cm
    over the table. Its set-down felt no table, and that map has no other spot for the lid: before the
    box, the wrist looks at the table from where the arm stands with the lid (the map may be stale, and
    from above the wrist sees table the scene camera does not), its frame is pooled in, and the search
    goes again on that map. Here it shows a spot, and the lid goes there."""
    maps = _no_other_spot_on_the_first_map(monkeypatch)
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY, wrist=_table(), setdown=["never", "met"]
    )
    assert code == 0 and _after_first_touch(rec) == ["lid:back-up", "spot-look", *CARRY, *HOME]
    assert len(maps) == 2 and maps[1] is not maps[0]
    (look,) = (r for r in rows if r["leg"] == "lid_look_again")
    assert look["wrist_points"] > 0 and look["mapped"] is True and look["looks"] == 1
    spots = [r for r in rows if r["leg"] == "lid_spot"]
    assert spots[1]["looked_again"] is True and spots[1]["searched"] == ["the tiers", "wider", "the tiers"]
    out = capsys.readouterr().out
    assert "no spot on the map in hand, which was made before the arm stood here: the wrist looks" in out


def test_a_wrist_look_that_adds_nothing_leaves_the_box_and_the_arm_looks_once_a_stand(
    ctx, monkeypatch, tmp_path, capsys
):
    """No other spot on the map in hand, and the wrist's look from where the arm stands gives no points:
    no spot is left to try, and the lid goes back on the box from over the spot it was left at — said
    with every spot it was taken to. And a map made with the wrist's frame from where the arm stands is
    not looked at again from there (the spot look after the lift: choose_and_carry's)."""
    _no_other_spot_on_the_first_map(monkeypatch)
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY, setdown=["never", "met"]
    )
    assert code == 1 and "place:open" not in rec.events and len(rec.setdowns) == 1
    assert _after_first_touch(rec) == [
        "lid:back-up",
        "spot-look",
        "lid:back-over",
        "lid:put-back",
        "lid:release",
        "lid:rise-off",
        "home",
    ]
    (look,) = (r for r in rows if r["leg"] == "lid_look_again")
    assert look["wrist_points"] == 0 and look["mapped"] is False
    out = capsys.readouterr().out
    assert "PUT BACK: no table was felt under the lid (spot 1 at [" in out
    assert "and no other spot on the table the cameras measured has a carry that plans" in out
    assert out.rstrip().endswith("the lid is back on the box, and the arm is home")
    # the spot look after the lift made the map where the arm stands: no second look from there
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY, refuse_carry=True, one_motion=False
    )
    assert code == 1 and rec.events.count("spot-look") == 1


def test_on_run_1s_map_the_spots_after_the_first_are_all_on_the_measured_table_and_off_what_was_left(
    monkeypatch, capsys
):
    """Flatbot 2026-11-17 16:40 once more (its map, its extent, flatbot's bare arm from the joints the
    carry began at): where the first spot's set-down feels no table, what follows is never the spot
    that run let the lid go at, in front of the table's edge — every spot the tiers and the wider search
    offer has the whole lid on table the cameras measured, a lid's width from the spot left. Which
    spots there are at each lean is printed for the record: at 20 deg the hand reaches two, 4 cm apart,
    so the second spot is one the hand leans 30 deg to. (No body model here, and so no leaning hand's
    body asked of what stands: the run asks it, and three of these six are then no spot —
    test_on_run_1s_map_the_carry_to_every_further_spot_is_the_walk_where_the_joint_space_way_is_refused.)"""
    from box_opening.runtime import approach
    from box_opening.runtime.kinematics import ArmChain

    chain = ArmChain(root="arm_mount", path=flatbot_urdf())
    ctx, tmap = _run1(monkeypatch, chain)
    monkeypatch.setattr(approach, "body_check", lambda: None, raising=False)
    q = flown.RUN1_Q_CLEAR
    first = next(rlo.spot_candidates(ctx, tmap, flown.RUN1_TABLE_Z, q))[0]
    search = rlo.SpotSearch(ctx, None, None, None, tmap, flown.RUN1_TABLE_Z, flown.RUN1_TABLE_Z, held=q)
    search.leave(first, "clear", "never", "the touch ended ...")
    common = dict(aside=search.aside, held=q)
    tiers = list(rlo.spot_candidates(ctx, tmap, flown.RUN1_TABLE_Z, q, way=True, **common))
    passed = rlo.SpotsPassed()
    wider = list(rlo.wider_candidates(ctx, tmap, flown.RUN1_TABLE_Z, q, passed=passed, **common))
    old = ContainerPose(xyz=(*flown.RUN1_SPOT, flown.RUN1_TABLE_Z), yaw=0.0)
    wide = max(ctx.model.lid_dims[:2])
    for d, _tier, _says in tiers + wider:
        assert rlo.spot_off_table(ctx, tmap, d, q) is None
        assert math.hypot(d.xyz[0] - first.xyz[0], d.xyz[1] - first.xyz[1]) >= wide
        assert math.hypot(d.xyz[0] - old.xyz[0], d.xyz[1] - old.xyz[1]) > 0.15
        assert rlo._spot_lean(d) <= 20.0 or rlo.landing_tilt_deg(ctx, d, q) <= 20.5
    with capsys.disabled():
        sys.stdout.write(
            "\nrun 1 replayed on flatbot's BARE arm, the first spot [%.2f, %.2f] (lean %.0f) left: the tiers then "
            "offer %d spot(s); the wider search %s (%s)\n"
            % (
                first.xyz[0],
                first.xyz[1],
                rlo._spot_lean(first),
                len(tiers),
                ", ".join(
                    "[%.2f, %.2f] %s lean %.0f" % (d.xyz[0], d.xyz[1], t, rlo._spot_lean(d)) for d, t, _s in wider
                )
                or "none",
                passed.says(),
            )
        )


def test_the_wider_search_and_the_second_look_never_take_the_time_the_way_back_to_the_box_needs(
    ctx, monkeypatch, tmp_path, capsys
):
    """The first spot's tiers are gone through whatever the run's clock says, as they always were. The
    wider search and the wrist's second look are work a run did not do before 2026-11-19, and with less
    left than a spot and the way back to the box take (31 s) they are not begun: the lid goes back on
    the box as it did, with the time that takes still in hand. The clock is asked before the wider
    search's own work, too: going through the table's places took 1-3 s before the first reach check
    asked it (the review of 2026-11-19)."""
    _budget(monkeypatch, 25.0)
    real, went = rlo._table_points, []
    monkeypatch.setattr(rlo, "_table_points", lambda *a, **k: went.append(1) or real(*a, **k))
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY, refuse_carry=True
    )
    assert code == 1 and "lid:carry" not in rec.events and rec.events.count("spot-look") == 1
    (spot,) = (r for r in rows if r["leg"] == "lid_spot")
    assert spot["refused"].count("IK_FAIL") == rlo.CARRY_TRIES  # the tiers' alone: no wider spot was asked for
    assert went == [] and spot["searched"] == ["the tiers"]  # ... and none of the table's places gone through
    out = capsys.readouterr().out
    assert "and no more of the table is looked at: the run's time holds no spot found later" in out  # why, said
    assert " s are left, they take 31)" in out
    assert out.rstrip().endswith("the lid is back on the box, and the arm is home")


def test_the_wrists_second_look_is_not_begun_where_the_runs_time_does_not_hold_a_spot_after_it(
    ctx, monkeypatch, tmp_path, capsys
):
    """No other spot on the map in hand, and by then less of the run's time than a spot and the way back
    to the box take: the wrist does not look again (a second or two with the arm standing, and a map to
    judge), and the lid goes back on the box from over the spot it was left at."""
    _no_other_spot_on_the_first_map(monkeypatch)
    left = iter([40.0, 40.0] + [20.0] * 10)
    monkeypatch.setattr(rlo.RunClock, "left_s", lambda self: next(left))
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY, wrist=_table(), setdown=["never", "met"]
    )
    assert code == 1 and "spot-look" not in rec.events and len(rec.setdowns) == 1
    assert _after_first_touch(rec) == [
        "lid:back-up",
        "lid:back-over",
        "lid:put-back",
        "lid:release",
        "lid:rise-off",
        "home",
    ]
    assert not [r for r in rows if r["leg"] == "lid_look_again"]
    assert capsys.readouterr().out.rstrip().endswith("the lid is back on the box, and the arm is home")


def test_a_spot_search_plans_to_no_spot_twice_and_each_search_has_its_own_carry_tries(ctx, monkeypatch, tmp_path):
    """SpotSearch.pick: the tiers' spots, CARRY_TRIES carries refused at most; then the wider search's,
    as many again (kept off the canister: plan_carry's over_box); then, where the map in hand was not
    made with the wrist's frame from where the arm stands, the wrist's look and both once more on the
    new map — and no spot a carry was refused to is planned to a second time: the planner's answer
    from the same joints stands."""
    from box_opening.runtime.runner import Runner
    from box_opening.tasks import press_demo as pd

    cfg = load_press_demo(PINK)
    ctx.model = ContainerModel.load(PINK)
    spots = [ContainerPose(xyz=(0.40 + 0.01 * k, -0.30, -0.07), yaw=0.0) for k in range(14)]
    planned, maps = [], []
    monkeypatch.setattr(rlo, "plan_carry", lambda ctx_, cfg_, q, d, **kw: (planned.append((d, kw)), (None, "no"))[1])

    def tiers(ctx_, tmap, *a, **k):
        maps.append(("the tiers", tmap))
        return iter([(d, "clear", "clear") for d in spots[:6]])

    def wider(ctx_, tmap, *a, **k):
        maps.append(("wider", tmap))
        return iter([(d, rlo.WHOLE_TABLE, "anywhere") for d in spots[4:]])

    monkeypatch.setattr(rlo, "spot_candidates", tiers)
    monkeypatch.setattr(rlo, "wider_candidates", wider)
    q = [float(v) for v in ctx.client.joints()]
    runner = Runner(ctx.client, log_dir=tmp_path)
    # the map made where the arm stands (the spot look after the lift): the tiers, then wider, and no second look
    search = rlo.SpotSearch(ctx, cfg, (None, None, None, None), runner, "map 1", -0.07, -0.07, looked_from=q)
    got = search.pick(runner, q)
    assert got.carry is None and got.chosen is None and len(got.refused) == 2 * rlo.CARRY_TRIES == 8
    assert (got.searched, got.looked) == (["the tiers", "wider"], False)
    assert [d for d, _kw in planned] == spots[:8]  # four of the tiers', then the next four: none twice
    assert [kw for _d, kw in planned[:4]] == [{}] * 4  # the first spot's tiers: planned as they always were
    for _d, kw in planned[4:]:  # the wider search's: off the canister, and asked what a later carry is
        assert kw["held"] is None and kw["over_box"] is True and isinstance(kw["way"], rlo.LaterWay)
    # a map not made from here (the scene camera's alone): the wrist looks, and both go once more on its map
    planned.clear()
    maps.clear()
    monkeypatch.setattr(rlo, "spot_look", lambda node, watcher, runner_: np.zeros((5, 3)))
    monkeypatch.setattr(rlo, "spot_map", lambda *a, **k: (maps.append(("spot_map", len(k["more"]))), "map 2")[1])
    monkeypatch.setattr(pd, "keep_out_counter", lambda *a, **k: None)
    search = rlo.SpotSearch(ctx, cfg, (None, SimpleNamespace(grab=None), None, None), runner, "map 1", -0.07, -0.07)
    got = search.pick(runner, q)
    assert got.carry is None and got.looked is True and search.tmap == "map 2" and search.looked_from == q
    assert got.searched == ["the tiers", "wider", "the tiers", "wider"]
    assert maps == [
        ("the tiers", "map 1"),
        ("wider", "map 1"),
        ("spot_map", 1),
        ("the tiers", "map 2"),
        ("wider", "map 2"),
    ]
    assert [d for d, _kw in planned] == spots[:12] and len(got.refused) == 12  # ... and still none twice
    # a carry that plans ends the search there
    monkeypatch.setattr(rlo, "plan_carry", lambda ctx_, cfg_, q_, d, **kw: ("the carry", None))
    got = rlo.SpotSearch(ctx, cfg, (None, None, None, None), runner, "map 1", -0.07, -0.07, looked_from=q).pick(
        runner, q
    )
    assert (got.carry, got.chosen, got.searched) == ("the carry", (spots[0], "clear", "clear"), ["the tiers"])


def test_a_spots_reach_is_judged_once_from_the_same_joints(ctx, monkeypatch):
    """_Reach keeps its own verdicts too: a spot asked about again in the same search (the tiers gone
    through once more after the wrist's look) is not solved again, and the lid's own joints (`held`)
    go to each check."""
    calls = []
    monkeypatch.setattr(
        rlo, "drop_reachable", lambda ctx_, d, q, seeds=(), held=None: calls.append((d, held)) or "no hand reaches it"
    )
    reach = rlo._Reach(held=[0.3] * 7)
    d = ContainerPose(xyz=(0.4, -0.3, -0.07), yaw=0.0)
    assert reach(ctx, d, [0.0] * 7) == "no hand reaches it" == reach(ctx, d, [0.0] * 7)
    assert calls == [(d, [0.3] * 7)] and len(reach.found) == 1


# ---- the fix round of 2026-11-19: what a carry after the first is asked, the leaning hand, "held" --------------


class _Dips(_HandAtQ):
    """_HandAtQ whose tool also sinks 0.2 m times the sine of pi times its fifth joint: between two
    waypoints at one height with that joint at 0 and 1, the straight joint line dips 0.2 m."""

    def fk(self, q):
        R, t = super().fk(q)
        return R, t - np.array([0.0, 0.0, 0.2 * math.sin(math.pi * float(q[4]))])


def _block(tmap, xy, tall, half=0.03):
    """`tmap` with a thing `tall` standing on the square of half-width `half` round `xy`."""
    from box_opening.perception.clear_table import TableMap

    X, Y = tmap.centres()
    on = (np.abs(X - xy[0]) <= half) & (np.abs(Y - xy[1]) <= half)
    return TableMap(
        tmap.x0, tmap.y0, tmap.cell, np.where(on, tall, tmap.tall), tmap.seen | on, tmap.off & ~on, tmap.box_xy
    )


def test_a_carry_after_the_first_is_asked_its_own_path_at_carry_height_over_the_table_and_what_stands_on_it(
    ctx, monkeypatch
):
    """lid_way_refusal (the review of 2026-11-19): the carry from one spot to the next crosses the table
    from a set-down's attitude, the planner's world holds no table, and a joint-space line between two
    hovers left the straight chord by 36-174 mm and sank up to 117 mm, to 80 mm over the table, on
    flatbot's URDF. Along the path as the arm flies it, between its waypoints too: the lid's lowest point
    never more than CARRY_SAG_M under the lower of its two ends, and nothing the map shows standing under
    the path it sweeps taller than the lid passes there (the lid LID_HANG_M lower than the model's cap)."""
    from box_opening.runtime import approach
    from box_opening.tasks.press_contact import pads_z

    m = ctx.model
    chain = _Dips(m, (0.6, 0.0))
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    table = -0.07
    lid_h = float(m.lid_dims[2])
    held = [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]
    tool_over_pads = 0.5 - pads_z(chain, held)

    def at(x, y, over, dip=0.0):
        """Joints with the level lid's lowest point (the model's) `over` the table, its centre at (x, y)."""
        return [x, y, table + over + lid_h - rlo.GRIP_BELOW_LID_M + tool_over_pads, 0.0, dip, 0.0, 0.0]

    bare = _standing_ring((0.5, 0.0), 5.0, 6.0)
    a, b = (0.3, -0.3), (0.7, 0.3)
    # a straight line at carry height, as the arm's own walk between two hovers flies it
    level = [at(*a, 0.15), at(0.5, 0.0, 0.15), at(*b, 0.15)]
    assert rlo.lid_way_refusal(ctx, bare, level, held, table) is None
    # ... sinking no more than CARRY_SAG_M under its lower end on the way is still that carry
    assert rlo.CARRY_SAG_M == pytest.approx(0.010)
    assert rlo.lid_way_refusal(ctx, bare, [at(*a, 0.15), at(0.5, 0.0, 0.142), at(*b, 0.17)], held, table) is None
    why = rlo.lid_way_refusal(ctx, bare, [at(*a, 0.15), at(0.5, 0.0, 0.10), at(*b, 0.17)], held, table)
    assert why == "the lid would sink 50 mm under the height it is carried at on the way, to 87 mm over the table"
    # between its waypoints too: both at carry height, the joint line between them dipping 0.2 m
    why = rlo.lid_way_refusal(ctx, bare, [at(*a, 0.15), at(*b, 0.15, dip=1.0)], held, table)
    assert why is not None and why.startswith("the lid would sink 200 mm under the height it is carried at")
    assert len(rlo._path_states([at(*a, 0.15), at(*b, 0.15, dip=1.0)])) == 51  # no joint more than 0.02 rad a state
    # what stands under the path: taller than the lid passes there (it hangs 13 mm lower than the model's cap;
    # the map's own slack, 20 mm, on top), and only under the path it sweeps
    low_thing, tall_thing = _block(bare, (0.5, 0.0), 0.10), _block(bare, (0.5, 0.0), 0.13)
    assert rlo.lid_way_refusal(ctx, low_thing, level, held, table) is None  # 137 mm over the table, 37 over the thing
    why = rlo.lid_way_refusal(ctx, tall_thing, level, held, table)
    assert why is not None and "the lid would pass" in why and "the lid 137 mm over the table there" in why
    assert rlo.lid_way_refusal(ctx, _block(bare, (0.7, -0.3), 0.30), level, held, table) is None  # off the path
    # the hand over the lid too: something its camera would meet, far taller than the lid passes
    why = rlo.lid_way_refusal(ctx, _block(bare, (0.5, 0.12), 0.40), level, held, table)
    assert why is not None and "the wrist camera would pass" in why
    # the table the set-down goes by, where that stands higher than the cameras': less room over the thing
    assert rlo.lid_way_refusal(ctx, low_thing, level, held, table + 0.03) is not None
    assert rlo.lid_way_refusal(ctx, None, level, held, table) is None  # no map: the height alone


class _Walks(_Chain):
    """The fake chain with a walk that flies (straight to its end)."""

    def __init__(self):
        super().__init__()
        self.walks = []

    def walk(self, q0, xyz, quat, step_m=None, step_rad=None, max_step_rad=None):
        self.walks.append(list(xyz))
        return [list(q0), [v + 0.01 for v in q0]], None


def test_a_carry_after_the_first_is_the_arms_own_walk_first_and_the_planners_only_where_it_passes_the_same(
    ctx, monkeypatch, capsys
):
    """plan_carry with a LaterWay: the walk first — a straight line from hover to hover at their height —
    and the planner's path only where no walk flies, each asked the same (the canister, the carry's height,
    what stands under it); a spot neither is a way to is refused with both reasons; a leaning set-down's
    way down is asked too. And with no LaterWay, a planned carry refused for the canister is flown as the
    walk where that clears it, before the spot is refused: on 2026-11-17 16:40's map the joint-space way to
    every further spot brought the lid under the canister's top and the walk to each cleared it, and the
    refusal returned without trying it (the review of 2026-11-19)."""
    from conftest import leg as make_leg

    from box_opening.runtime import approach

    cfg = load_press_demo(PINK)
    chain = _Walks()
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    planned, paths, descents = [], [], []

    def plan(ctx_, state, name, target, world, speed):
        planned.append(name)
        lg = make_leg(name, start=list(state.joints), end=[v + 0.05 for v in state.joints], speed=speed, world=world[0])
        lg.target = target
        return lg, state

    monkeypatch.setattr(rp, "_plan_motion", plan)
    drop = ContainerPose(xyz=(0.40, -0.30, -0.07), yaw=0.0)
    q = [0.1] * 7

    class Way:
        refuse = {}  # how many points a refused path has -> why

        def path(self, pts):
            paths.append(len(pts))
            return self.refuse.get(len(pts))

        def descent(self, d, pts):
            descents.append((d, len(pts)))
            return self.refuse.get("descent")

    way = Way()
    carry, why = rlo.plan_carry(ctx, cfg, q, drop, held=[0.2] * 7, way=way)
    assert (
        why is None and carry.legs[0].plan_free and planned == [] and len(chain.walks) == 1
    )  # the walk: no plan asked
    assert paths == [2] and descents == [(drop, 4)]  # its own path asked; the way down and the touch, 2 points each
    assert "the lid's way to [0.40, -0.30] is the arm's own walk, a straight line at the height it is carried at" in (
        capsys.readouterr().out
    )
    # the walk's path refused: the planner's, asked the same
    way.refuse = {2: "the lid would sink 40 mm"}
    paths.clear()
    carry, why = rlo.plan_carry(ctx, cfg, q, drop, held=[0.2] * 7, way=way)
    assert carry is None and planned == ["lid:carry"] and paths == [2, 2]
    assert why == "the carry (walked: the lid would sink 40 mm; planned: the lid would sink 40 mm)"
    # ... and taken where it passes (this planner's path has as many points: told apart by the walk failing)
    way.refuse = {}
    monkeypatch.setattr(chain, "walk", lambda *a, **k: (None, "IK did not converge 40% of the way"))
    carry, why = rlo.plan_carry(ctx, cfg, q, drop, held=[0.2] * 7, way=way)
    assert why is None and not carry.legs[0].plan_free
    assert "is the planner's (no walk flies: IK did not converge 40% of the way)" in capsys.readouterr().out
    # a planner that refuses: both said
    monkeypatch.setattr(
        rp, "_plan_motion", lambda *a: (_ for _ in ()).throw(RuntimeError("x: MotionGenStatus.IK_FAIL"))
    )
    carry, why = rlo.plan_carry(ctx, cfg, q, drop, held=[0.2] * 7, way=way)
    assert (
        carry is None
        and why == "the carry (walked: IK did not converge 40% of the way; planned: MotionGenStatus.IK_FAIL)"
    )
    # a leaning set-down's way down refused
    monkeypatch.setattr(rp, "_plan_motion", plan)
    way.refuse = {"descent": "the hand leaning there would come 12 mm into what stands 150 mm tall"}
    carry, why = rlo.plan_carry(ctx, cfg, q, drop, held=[0.2] * 7, way=way)
    assert (
        carry is None and why == "the set-down (the hand leaning there would come 12 mm into what stands 150 mm tall)"
    )
    # no LaterWay (a hand leaning past 20 deg; over_box): the planner's carry first, and where the canister
    # refuses it the walk before the spot is refused
    chain = _Walks()
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    asked = []
    monkeypatch.setattr(
        rlo, "lid_low_over_box", lambda ctx_, pts, held_: asked.append(1) or (None if len(asked) > 1 else "too low")
    )
    carry, why = rlo.plan_carry(ctx, cfg, q, drop, held=[0.2] * 7, over_box=True)
    assert why is None and carry.legs[0].plan_free and len(asked) == 2 and len(chain.walks) == 1
    assert "the planner's carry is no way for the lid (too low): the arm's own walk instead" in capsys.readouterr().out


# flatbot's own robot file from the forearm out: the collision spheres of the forearm, the wrist's two links,
# the bracelet and the 2F-85, copied 2026-11-19 from the file its planner and its body check read on the robot
# (config/robots/flatbot_fwd8_cams1117/curobo/robot.yaml of its deployment). The tests of what the hand's own
# body is asked stood on test_body's STOCK_HAND, which has no sphere on either wrist link, and so pinned an
# accept/refuse line the robot does not have: the wrist is most of what leans arm-side (the review of 2026-11-19).
FLATBOT_ARM_END = """robot_cfg:
  kinematics:
    base_link: arm_mount
    collision_sphere_buffer: 0.005
    collision_spheres:
      arm_forearm_link:
      - {center: [0.0, 0.0, 0.0], radius: 0.06}
      - {center: [0.0, -0.06, 0.0], radius: 0.058}
      - {center: [0.0, -0.12, 0.0], radius: 0.058}
      - {center: [0.0, -0.17, 0.0], radius: 0.055}
      arm_spherical_wrist_1_link:
      - {center: [0.0, 0.0, 0.0], radius: 0.06}
      - {center: [0.0, 0.0, -0.085], radius: 0.06}
      arm_spherical_wrist_2_link:
      - {center: [0.0, 0.0, 0.0], radius: 0.055}
      - {center: [0.0, -0.085, 0.0], radius: 0.055}
      arm_bracelet_link:
      - {center: [0.0, 0.0, -0.045], radius: 0.045}
      - {center: [0.0, -0.05, -0.045], radius: 0.045}
      - {center: [0.045, 0.0, -0.05], radius: 0.036}
      arm_robotiq_85_base_link:
      - {center: [0.0, 0.0, 0.04], radius: 0.04}
      - {center: [0.0, 0.0, 0.055], radius: 0.042}
      - {center: [3.2e-05, 0.04, 0.06], radius: 0.032}
      - {center: [-3.2e-05, -0.04, 0.06], radius: 0.032}
      - {center: [-0.04, 3.2e-05, 0.06], radius: 0.032}
      - {center: [0.04, -3.2e-05, 0.06], radius: 0.032}
      - {center: [0.0, 0.0, 0.078], radius: 0.035}
      - {center: [3.6e-05, 0.045, 0.1], radius: 0.026}
      - {center: [-3.6e-05, -0.045, 0.1], radius: 0.026}
      - {center: [-0.045, 3.6e-05, 0.1], radius: 0.026}
      - {center: [0.045, -3.6e-05, 0.1], radius: 0.026}
      - {center: [3.6e-05, 0.045, 0.13], radius: 0.022}
      - {center: [-3.6e-05, -0.045, 0.13], radius: 0.022}
      - {center: [-0.045, 3.6e-05, 0.13], radius: 0.022}
      - {center: [0.045, -3.6e-05, 0.13], radius: 0.022}
      - {center: [3.6e-05, 0.045, 0.145], radius: 0.018}
      - {center: [-3.6e-05, -0.045, 0.145], radius: 0.018}
      - {center: [-0.045, 3.6e-05, 0.145], radius: 0.018}
      - {center: [0.045, -3.6e-05, 0.145], radius: 0.018}
      arm_robotiq_85_left_finger_link:
      - {center: [-0.010026, -4.3e-05, 0.019663], radius: 0.03}
      arm_robotiq_85_left_finger_tip_link:
      - {center: [-0.021578, -3.8e-05, 0.031998], radius: 0.02}
      arm_robotiq_85_right_finger_link:
      - {center: [0.010026, 4.1e-05, 0.019663], radius: 0.03}
      arm_robotiq_85_right_finger_tip_link:
      - {center: [0.021578, 3.7e-05, 0.031998], radius: 0.02}
"""


def _body_on(chain, tmp_path, monkeypatch, robot=FLATBOT_ARM_END):
    """flatbot's body model on `chain` (`robot`: its robot file's spheres, flatbot's own from the forearm
    out; this checkout's hand spheres), no chair: what approach.body_check() hands the run on the robot."""
    from pathlib import Path

    import test_body as tb

    from box_opening.runtime import approach, body

    pkg = Path(__file__).resolve().parents[1]
    model = body.load(
        tb._model_files(tmp_path, [("far", (5.0, 5.0, 5.0), (0.1, 0.1, 0.1))], robot=robot),
        robot_body=pkg / "config" / "robot_body.yaml",
    )
    bd = model.check_for(chain, flatbot_urdf(), "arm_mount")
    monkeypatch.setattr(approach, "body_check", lambda: bd, raising=False)
    monkeypatch.setattr(rlo, "_HAND_MODEL_SAID", set())  # (what a run says once of its body model: each test's own)
    return bd


def _leaning_down(ctx, chain, lean, xy):
    """A leaning set-down where run 1's box and table stood: (the spot, the joint line down to its touch's
    end from the hover, the unit vector from the arm to the spot)."""
    q, table = list(flown.RUN1_Q_CLEAR), flown.RUN1_TABLE_Z
    d = rlo.LeanSpot(xyz=(xy[0], xy[1], table), yaw=0.0, lean_deg=lean)
    quat, contact, hover = rlo.setdown_poses(ctx, d, held=q)
    qh = chain.nearest_solution(q, hover, quat, seeds=())
    down, _why = chain.straight_line(qh, [contact[0], contact[1], contact[2] - rlo.SETDOWN_PAST_M], quat)
    assert down is not None
    return d, down, np.array(xy) / math.hypot(*xy)


@pytest.mark.parametrize(
    "lean, xy, back_mm, tall, link",
    [
        (30.0, (0.87, 0.03), 120, 0.20, "arm_spherical_wrist_2_link"),
        (30.0, (0.87, 0.03), 250, 0.30, "arm_spherical_wrist_1_link"),  # only the wrist's first link reaches it
        (20.0, (0.74, -0.338), 110, 0.20, "arm_spherical_wrist_2_link"),
        (20.0, (0.74, -0.338), 250, 0.30, "arm_spherical_wrist_1_link"),
    ],
)
def test_a_leaning_hand_is_held_out_of_what_stands_arm_side_of_its_spot_by_its_own_body(
    monkeypatch, tmp_path, lean, xy, back_mm, tall, link
):
    """The gate's hand (clear_table.HAND_PARTS) is a hand straight down: two rectangles over the spot.
    Leaning back toward the arm the hand's body stands arm-side of the spot, and the rectangles stay where
    they were. Where the box and the table of 2026-11-17 16:40 stood, with flatbot's arm and flatbot's own
    spheres of the wrist and hand (the reviews of 2026-11-19, measured again): a thing 6 cm square standing
    11-25 cm arm-side of the lid's middle — the gate takes the spot, and the wrist comes 28-53 mm into it
    (20 cm tall 110-120 mm back: the wrist's second link, 38 mm at 20 deg and 48 at 30; 30 cm tall 250 mm
    back, which nothing but the wrist's first link reaches: 28 and 53). hand_into_standing asks the body
    model's spheres of the hand and wrist at the set-down's own joints, all the way down: that spot is
    refused; the same spot with nothing standing there, or with a 12 cm thing 200 mm back, is not; and with
    no body model nothing can tell, which is itself the refusal."""
    from box_opening.runtime import approach
    from box_opening.runtime.kinematics import ArmChain

    chain = ArmChain(root="arm_mount", path=flatbot_urdf())
    ctx, _map = _run1(monkeypatch, chain)
    tmap = _standing_ring(flown.RUN1_BOX_XY, 5.0, 6.0)  # that run's box and table, seen bare 1.2 m across
    bd = _body_on(chain, tmp_path, monkeypatch)
    links, missing = rlo._hand_links(bd)
    assert missing == [] and "arm_forearm_link" not in links
    assert links >= {
        "arm_spherical_wrist_1_link",
        "arm_spherical_wrist_2_link",
        "arm_bracelet_link",
        "arm_robotiq_85_base_link",
        "arm_end_effector_link",
    }
    q, table = list(flown.RUN1_Q_CLEAR), flown.RUN1_TABLE_Z
    d, down, u = _leaning_down(ctx, chain, lean, xy)
    thing = _block(tmap, tuple(np.array(xy) - 0.001 * back_mm * u), tall)
    assert rlo.spot_off_table(ctx, thing, d, q) is None  # the gate, a hand straight down: it takes the spot
    why = rlo.hand_into_standing(ctx, thing, down, table)
    assert why is not None and why.startswith("the hand leaning there would come ")
    assert " mm into what stands %d mm tall" % (1000 * tall) in why and "(%s; 20 mm is kept)" % link in why
    assert rlo.hand_into_standing(ctx, tmap, down, table) is None  # nothing standing there: the spot is taken
    low = _block(tmap, tuple(np.array(xy) - 0.200 * u), 0.12)  # ... nor with a 12 cm thing 200 mm arm-side
    assert rlo.spot_off_table(ctx, low, d, q) is None and rlo.hand_into_standing(ctx, low, down, table) is None
    way = rlo.LaterWay(ctx, thing, table, q)
    assert way.descent(d, down) == why
    assert way.descent(ContainerPose(xyz=d.xyz, yaw=0.0), down) is None  # a hand straight down is the gate's own
    # the reach check of a later spot asks it (drop_reachable's `stands`), and plan_carry of the lines it flies
    assert rlo.drop_reachable(ctx, d, q, stands=way.descent) == why
    assert rlo.drop_reachable(ctx, d, q) is None
    monkeypatch.setattr(approach, "body_check", lambda: None, raising=False)
    assert rlo.hand_into_standing(ctx, tmap, down, table) == (
        "no body model of the hand to hold it, leaning, out of what stands by the spot"
    )


def test_a_body_model_with_no_spheres_on_the_wrist_is_no_model_of_the_hand_for_the_lids_further_spots(
    monkeypatch, tmp_path, capsys
):
    """A Body knows only the links its sphere file names. With a file that gives the wrist's two links no
    sphere (test_body's STOCK_HAND: the forearm, the bracelet, the 2F-85) the check of a leaning hand asked
    the bracelet and the gripper and passed whatever the wrist came into: a 30 cm thing 250 mm arm-side of a
    30 deg spot was taken with the wrist's centre line 3 mm from it — the same thing flatbot's own file
    refuses by 53 mm (the second review of 2026-11-19). Such a model is no body model for what is asked of
    the hand past the first spot: no leaning set-down is taken (the refusal says what the file lacks), a
    later carry is asked of the lid and a top-down hand alone, and the run says so, once."""
    import test_body as tb

    from box_opening.runtime.kinematics import ArmChain

    chain = ArmChain(root="arm_mount", path=flatbot_urdf())
    ctx, _map = _run1(monkeypatch, chain)
    bare = _standing_ring(flown.RUN1_BOX_XY, 5.0, 6.0)
    q, table = list(flown.RUN1_Q_CLEAR), flown.RUN1_TABLE_Z
    d, down, u = _leaning_down(ctx, chain, 30.0, (0.87, 0.03))
    thing = _block(bare, tuple(np.array((0.87, 0.03)) - 0.250 * u), 0.30)
    (tmp_path / "own").mkdir()
    (tmp_path / "stock").mkdir()
    _body_on(chain, tmp_path / "own", monkeypatch)
    assert " mm into what stands 300 mm tall" in rlo.hand_into_standing(ctx, thing, down, table)
    capsys.readouterr()
    bd = _body_on(chain, tmp_path / "stock", monkeypatch, robot=tb.STOCK_HAND)
    links, missing = rlo._hand_links(bd)
    assert missing == ["the wrist's first link", "the wrist's second link"]
    assert "arm_bracelet_link" in links and "arm_spherical_wrist_1_link" not in links
    lacks = "the body model has no spheres on the wrist's first link or the wrist's second link"
    refusal = lacks + ": no model of the hand to hold it, leaning, out of what stands by the spot"
    assert rlo.hand_into_standing(ctx, thing, down, table) == refusal
    assert rlo.hand_into_standing(ctx, bare, down, table) == refusal  # nothing standing: still no model of it
    way = rlo.LaterWay(ctx, thing, table, q)
    assert way.descent(d, down) == refusal and rlo.drop_reachable(ctx, d, q, stands=way.descent) == refusal
    # a carry is asked by the checks that need no model of the hand: this one of the hand's body is not made
    assert rlo.hand_way_refusal(ctx, thing, down, table) is None
    out = capsys.readouterr().out
    assert out.count(lacks) == 1  # said once, however often it was asked
    assert "no leaning set-down is taken past the first spot's own" in out
    # a body that does not say what moves its links, or on an arm of fewer joints than a wrist has: none of it
    assert rlo._hand_links(SimpleNamespace()) == (
        set(),
        ["the wrist's first link", "the wrist's second link", "the hand"],
    )
    paths = {"hand": [(None, None, 0, 0), (None, None, 1, 1)]}
    assert rlo._hand_links(SimpleNamespace(paths=paths, index={"a": 0, "b": 1}))[1] == [
        "the wrist's first link",
        "the wrist's second link",
        "the hand",
    ]


def test_on_run_1s_map_the_carry_to_every_further_spot_is_the_walk_where_the_joint_space_way_is_refused(
    ctx, monkeypatch, tmp_path, capsys
):
    """Flatbot 2026-11-17 16:40's map, its arm, the lid as gripped there, from the hover over the first
    spot once it is left. The planner is stood in for by a joint-space line to the next hover (no planner
    offline). To every further spot the search offers, that line brings the lid 23-26 mm under the top it
    came off, over the open canister, and under the height it is carried at; the arm's own walk to each
    clears both, the lid 150 mm over the table all the way and the hand's own body clear of what stands.
    plan_carry flies the walk — it returned the canister's refusal without trying it, and the run would
    have stood 15-20 s with the lid and then put it back on the box (the review of 2026-11-19). And the
    further spots are fewer than they were: three of the six the first version offered had the leaning
    hand come down within a finger of what stands 55-65 mm tall beside them."""
    from conftest import leg as make_leg

    from box_opening.runtime import approach
    from box_opening.runtime.kinematics import ArmChain

    chain = ArmChain(root="arm_mount", path=flatbot_urdf())
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    _body_on(chain, tmp_path, monkeypatch)
    ctx.model = ContainerModel.load(PINK)
    ctx.cpose = ContainerPose(xyz=(*flown.RUN1_BOX_XY, flown.RUN1_TABLE_Z), yaw=0.0)
    ctx.worlds.table_extent = flown.RUN1_EXTENT
    ctx.held_xy = None
    ctx.lid_lifted = SimpleNamespace(q_grip=flown.RUN1_Q_GRIP)
    tmap, table, q = flown.run1_map(), flown.RUN1_TABLE_Z, list(flown.RUN1_Q_CLEAR)
    cfg = load_press_demo(PINK)
    first = next(rlo.spot_candidates(ctx, tmap, table, q))[0]
    quat1, _contact1, hover1 = rlo.setdown_poses(ctx, first, held=q)
    q1 = [float(v) for v in chain.nearest_solution(q, hover1, quat1, seeds=())]
    search = rlo.SpotSearch(ctx, cfg, None, None, tmap, table, table, held=q)
    search.leave(first, "clear", "never", "the touch ended ...")
    way = search.way(q)
    reach = rlo._Reach(held=q, stands=way.descent)
    common = dict(aside=search.aside, held=q, reach=reach)
    spots = list(rlo.spot_candidates(ctx, tmap, table, q1, way=True, **common))
    spots += list(rlo.wider_candidates(ctx, tmap, table, q1, **common))
    assert 1 <= len(spots) <= 5 and all(rlo._spot_lean(d) == 30.0 for d, _t, _s in spots)
    assert sum("the hand leaning there would come" in (why or "") for why, _took in reach.found.values()) >= 1
    lines = []

    def joint_line(ctx_, state, name, target, world, speed):
        """The planner's stand-in: the straight joint line to the nearest solution of the goal pose."""
        goal = chain.nearest_solution(list(state.joints), target[1], target[2], seeds=())
        lg = make_leg(name, start=list(state.joints), end=[float(v) for v in goal], speed=speed, world=world[0])
        lg.target = target
        lines.append([list(state.joints), [float(v) for v in goal]])
        return lg, state

    monkeypatch.setattr(rp, "_plan_motion", joint_line)
    said = []
    for d, tier, _says in spots:
        # the joint-space way, asked alone: refused for the canister and for the height it sinks to
        carry, why = rlo.plan_carry(ctx, cfg, q1, d, held=q)
        assert lines and rlo.lid_low_over_box(ctx, lines[-1], q) is not None
        assert rlo.lid_way_refusal(ctx, tmap, lines[-1], q, table) is not None
        # ... a hand leaning past 20 deg is asked the canister of it even so, and the walk is flown instead
        assert why is None and carry.legs[0].plan_free
        # the carry a later spot is planned with: the walk first, the planner not asked at all
        n = len(lines)
        carry, why = rlo.plan_carry(ctx, cfg, q1, d, held=q, over_box=True, way=way)
        assert why is None and carry.legs[0].plan_free and len(lines) == n
        walked = [list(p.positions) for p in carry.legs[0].traj.points]
        assert rlo.lid_low_over_box(ctx, walked, q) is None and rlo.lid_way_refusal(ctx, tmap, walked, q, table) is None
        assert rlo.hand_way_refusal(ctx, tmap, walked, table) is None  # the hand's own body, all along it
        _xy, low = rlo._lid_along(ctx, rlo._path_states(walked), q)
        said.append(
            "[%.2f, %.2f] %s lean %.0f: walked, the lid %.0f mm over the table at the least (the joint-space way: %s)"
            % (
                d.xyz[0],
                d.xyz[1],
                tier,
                rlo._spot_lean(d),
                1000 * (low.min() - table),
                rlo.lid_low_over_box(ctx, lines[-1], q),
            )
        )
    with capsys.disabled():
        sys.stdout.write(
            "\nrun 1 replayed on flatbot's BARE arm with its hand's body, from over the first spot: %s\n"
            % "; ".join(said)
        )


# ---- after "held": the wrist's look, the hand's footprint, the same height twice ------------------------------


class _HeldAt(_Felt):
    """A set-down's watch that felt something hold the lid up `above_mm` over where the plan has the table."""

    def __init__(self, above_mm):
        super().__init__("held")
        self.above_mm = float(above_mm)

    def refusal(self):
        return ("held", "something held the lid up %.0f mm over where the plan has it meet the table" % self.above_mm)

    def record(self):
        return dict(super().record(), above_top_mm=self.above_mm)


def test_where_something_held_the_lid_up_the_hands_footprint_is_set_aside_with_the_lids(ctx, monkeypatch):
    """SpotSearch.aside after a "held": something stands at that spot which the map did not show, and no
    part of the next lid or hand may come down over the footprint that was left there — the lid's disc
    and the hand's (HAND_REACH_M round where its tool stood: the wrist camera's corner, 99 mm, whichever
    way the hand is turned next). The lid's width alone let the next spot lie rim to rim with whatever
    held the lid, 108 mm away for a 105 mm lid (the review of 2026-11-19). After a "never" the lid's own
    footprint is what is set aside, as it was. A leaning hand stands arm-side of the lid's middle, and
    its footprint with it."""
    from box_opening.runtime import approach

    monkeypatch.setattr(approach, "_chain", lambda: _Chain())
    wide, reach = _lid_wide(ctx), rlo.HAND_REACH_M
    lid_r = 0.5 * wide
    assert reach == pytest.approx(0.0986, abs=1e-4) and 0.10 <= wide <= 0.105

    def at(x, y=0.0, lean=0.0):
        d = ContainerPose(xyz=(0.5 + x, y, -0.07), yaw=0.0)
        return d if not lean else rlo.LeanSpot(xyz=d.xyz, yaw=0.0, lean_deg=lean)

    never = rlo.SpotSearch(ctx, None, None, None, None, -0.07, -0.07)
    never.leave(at(0.0), "clear", "never", "nothing felt")
    assert never.left[0].hand_xy == pytest.approx((0.5, 0.0))  # a hand straight down: over the lid's middle
    assert never.aside(at(wide - 0.001)) and not never.aside(at(wide + 0.001))  # the lid's own width, as before
    held = rlo.SpotSearch(ctx, None, None, None, None, -0.07, -0.07)
    held.leave(at(0.0), "clear", "held", "something held the lid up", above_m=0.06)
    # two hands straight down: neither's farthest corner over the other's footprint
    assert held.aside(at(wide + 0.003))  # rim to rim with what held the lid: where the next spot lay before
    assert held.aside(at(2 * reach - 0.002)) and not held.aside(at(2 * reach + 0.002))
    assert held.aside(at(0.0, 2 * reach - 0.002)) and not held.aside(at(0.0, -2 * reach - 0.002))
    # a leaning hand stands arm-side of its lid (here 60 mm toward -x): the footprint left goes with it, and
    # the next spot's own hand is where ITS tool will stand
    real = rlo.setdown_poses

    def poses(ctx_, d, held=None):
        if rlo._spot_lean(d):
            return None, (d.xyz[0] - 0.06, d.xyz[1], 0.2), None
        return real(ctx_, d, held=held)

    monkeypatch.setattr(rlo, "setdown_poses", poses)
    lean = rlo.SpotSearch(ctx, None, None, None, None, -0.07, -0.07, held=[0.0] * 7)
    lean.leave(at(0.0, lean=20.0), "clear", "held", "something held the lid up", above_m=0.06)
    assert lean.left[0].hand_xy == pytest.approx((0.44, 0.0))
    far, arm_side = 2 * reach + 0.002, -(0.06 + 2 * reach)
    assert not lean.aside(at(far - 0.04))  # past the lid's far rim: 159 mm from the lid, 219 from where the hand stood
    assert lean.aside(at(arm_side + 0.002)) and not lean.aside(at(arm_side - 0.002))  # arm-side: from the hand
    # ... a next spot the hand leans to, its tool 60 mm arm-side of it: that hand's disc off the lid's that was left
    edge = lid_r + reach + 0.06
    assert lean.aside(at(edge - 0.002, lean=20.0)) and not lean.aside(at(edge + 0.002, lean=20.0))


def test_two_set_downs_held_up_at_the_same_height_are_the_table_itself_and_no_third_spot_is_pressed_on(
    ctx, monkeypatch, tmp_path, capsys
):
    """Something the map did not show held the lid up at one spot, and at the next, a hand's width away
    and looked at by the wrist in between, something held it up at the same height (within SAME_HELD_M,
    15 mm): that is probably the table itself, standing higher than the run has it. Each such set-down
    presses 12-16 mm into what it met before it is told from the table, and going on would press on it
    spot after spot, up to SETDOWN_SPOTS times (the review of 2026-11-19): the run does not. It says so
    — on the console, in the run's record (lid_spot_left.same_height_as) — and takes the lid back to the
    box. The lid is not let go on what held it: nothing the run has says where the table is but the plan
    that put it 60 mm lower."""
    assert rlo.SAME_HELD_M == pytest.approx(0.015)
    code, rec, _chain, rows = _full_flow(
        ctx,
        monkeypatch,
        tmp_path,
        [(BUTTON, 0.4), (UP, 9.0)],
        extent=ROOMY,
        scene=_table(),
        wrist=_table(),
        setdown=[_HeldAt(60.0), _HeldAt(72.0), "met"],
    )
    assert code == 1 and "place:open" not in rec.events and [how for _d, how in rec.setdowns] == ["held", "held"]
    after = _after_first_touch(rec)
    second = after.index("place:touch")
    assert after[:second] == ["lid:back-up", "spot-look", "lid:carry", "place:down"]  # the look before spot 2
    # from the second touch's end straight back: up, over the box, down onto it, and home — no third spot
    assert after[second + 1 :] == [
        "lid:back-up",
        "lid:back-over",
        "lid:put-back",
        "lid:release",
        "lid:rise-off",
        "home",
    ]
    one, two = (r for r in rows if r["leg"] == "lid_spot_left")
    assert (one["verdict"], one["above_mm"], one["same_height_as"], one["next"]) == ("held", 60.0, None, "another spot")
    assert (two["verdict"], two["above_mm"], two["same_height_as"], two["next"]) == ("held", 72.0, 1, "the box")
    assert "probably the table itself, standing higher than the run has it" in two["why_the_box"]
    out = capsys.readouterr().out
    assert (
        "no other spot is tried: something held the lid up at the same height at two spots, +60 mm from where the "
        "plan has the table at spot 1 and +72 mm at spot 2: that is probably the table itself"
    ) in out
    assert out.rstrip().endswith("the lid is back on the box, and the arm is home")
    # the flight view reads those lines of the run's own
    view = _flight_view(out, code)
    i = _view_line(view, r" *\d+ SET DOWN  FAIL  spot 2 \(\d+ cm from the box\): something held the lid up, lid kept")
    assert view[i - 1].endswith("SET DOWN  FAIL  spot 1: something held the lid up, lid kept")
    assert view[i + 1] == (
        "             next  carrying the lid back to the box (held up at the same height twice: probably the "
        "table, higher than planned)"
    )
    assert view[i + 2] == "             next  putting the lid back on the box"
    assert view[i + 3].endswith("PUT BACK  ok    lid back on the box, arm home")
    assert "lid: on the box" in view[-1]


def test_set_downs_held_up_at_different_heights_are_different_things_and_the_lid_goes_on(
    ctx, monkeypatch, tmp_path, capsys
):
    """Held up 60 mm over the plan at one spot and 90 mm at the next: two things the map did not show,
    not the table. The wrist looks after each, and the lid goes on to a third spot, a hand's width from
    both, where it meets the table and is let go."""
    code, rec, _chain, rows = _full_flow(
        ctx,
        monkeypatch,
        tmp_path,
        [(BUTTON, 0.4), (UP, 9.0)],
        extent=ROOMY,
        scene=_table(),
        wrist=_table(),
        setdown=[_HeldAt(60.0), _HeldAt(90.0), "met"],
    )
    assert code == 0 and [how for _d, how in rec.setdowns] == ["held", "held", "met"]
    assert _after_first_touch(rec) == ["lid:back-up", "spot-look", *CARRY, "lid:back-up", "spot-look", *CARRY, *HOME]
    looks = [r for r in rows if r["leg"] == "lid_look_again"]
    assert [r["why"] for r in looks] == [
        "something the map did not show held the lid up at spot 1",
        "something the map did not show held the lid up at spot 2",
    ]
    assert [r["same_height_as"] for r in rows if r["leg"] == "lid_spot_left"] == [None, None]
    (a, _), (b, _), (c, _) = rec.setdowns
    for p, q in ((a, b), (a, c), (b, c)):
        assert math.hypot(p.xyz[0] - q.xyz[0], p.xyz[1] - q.xyz[1]) >= 2 * rlo.HAND_REACH_M - 1e-9
    out = capsys.readouterr().out
    assert out.count("something the map did not show held the lid up at spot") == 2
    view = _flight_view(out, code)
    i = _view_line(view, r" *\d+ SET DOWN  ok    spot 3 \(\d+ cm from the box\): let go where the lid met the table")
    assert "SET DOWN  FAIL  spot 2 (" in view[i - 1] and "SET DOWN  FAIL  spot 1: " in view[i - 2]
    assert view[-1].startswith(
        "RESULT  DONE — box open, lid on the table (spot 3: no table felt at 2 before), arm home"
    )


# ---- the run's own console lines, read by the flight view ------------------------------------------------------


def _flight_view(out, code, **how):
    """The flight view's lines (box_opening/flight_view.py) for a run whose console said `out`, ended `code`."""
    from box_opening.flight_view import FlightView

    lines = []
    view = FlightView(write=lines.append, clock=lambda: 0.0)
    view.start()
    for text in out.splitlines():
        view.feed(text)
    view.finish(code, **how)
    return lines


def _view_line(view, pattern):
    """Which of the flight view's lines is the one that reads `pattern` (a regular expression of the whole line)."""
    import re

    (i,) = (k for k, line in enumerate(view) if re.fullmatch(pattern, line))
    return i


# ---- what a carry after the first is asked, in the run ---------------------------------------------------------


def _once_a_spot_is_left(monkeypatch, then=lambda: None):
    """A list that is empty until the run's first spot is left (another_spot entered; `then()` called there)."""
    later, real = [], rlo.another_spot

    def another(*a, **k):
        later.append(1)
        then()
        return real(*a, **k)

    monkeypatch.setattr(rlo, "another_spot", another)
    return later


def test_in_the_run_the_lids_straight_way_from_a_spot_left_is_asked_of_every_next_spot(
    ctx, monkeypatch, tmp_path, capsys
):
    """From a spot the lid was left at, something the map shows standing in the lid's straight way to
    every other spot: each is passed over before its reach is asked or a carry planned (_offers' `way`,
    which SpotSearch.pick hands the tiers of every spot after the first — nothing but a unit test of
    _offers held that, and the run asking nothing of the way passed every test: the review of
    2026-11-19), the run says so, and the lid goes back on the box. The first spot's own way is the
    room's, judged from the box, as it was."""
    real = rlo.clear_table.carry_clearance_fast
    later = _once_a_spot_is_left(monkeypatch)

    def way(tmap, a, b, *rest):
        if later:
            return -0.05, "the lid", (float(b[0]), float(b[1]))
        return real(tmap, a, b, *rest)

    monkeypatch.setattr(rlo.clear_table, "carry_clearance_fast", way)
    checks = _later_reach(monkeypatch, None)
    plans = _later_plans(monkeypatch)
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY, scene=_table(), setdown=["never", "met"]
    )
    assert code == 1 and len(rec.setdowns) == 1 and "place:open" not in rec.events
    assert checks == [] and plans == []  # no spot was a try
    spots = [r for r in rows if r["leg"] == "lid_spot"]
    assert spots[0]["table"]["blocked"] in (None, 0) and spots[1]["table"]["blocked"] > 0
    out = capsys.readouterr().out
    assert "%d with something standing in the lid's way there" % spots[1]["table"]["blocked"] in out
    assert out.rstrip().endswith("the lid is back on the box, and the arm is home")


def test_in_the_run_every_carry_after_the_first_is_asked_the_canister_and_its_own_path_and_the_first_is_not(
    ctx, monkeypatch, tmp_path, capsys
):
    """What plan_carry asks of a carry's own path — the canister under it (lid_low_over_box), its height
    all the way and what stands under it (lid_way_refusal) — is asked of every carry the run plans once a
    spot is left, and of none before: the first spot's carry is planned as it was flown before
    2026-11-19 (of 22 recorded first carries 4 would not pass, the report's numbers). Here every later
    carry's path would sink: none is flown, each spot's refusal says so, and the lid goes back on the box."""
    later = _once_a_spot_is_left(monkeypatch)
    asked = {"first": 0, "box": 0, "way": 0}

    def low(ctx_, pts, held):
        asked["box" if later else "first"] += 1
        return None

    def sinks(ctx_, tmap, pts, held, table_z):
        asked["way" if later else "first"] += 1
        return "the lid would sink 40 mm under the height it is carried at on the way, to 110 mm over the table"

    monkeypatch.setattr(rlo, "lid_low_over_box", low)
    monkeypatch.setattr(rlo, "lid_way_refusal", sinks)
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY, scene=_table(), setdown=["never", "met"]
    )
    assert code == 1 and len(rec.setdowns) == 1 and "place:open" not in rec.events
    assert asked["first"] == 0 and asked["box"] >= 1 and asked["way"] == asked["box"]
    spots = [r for r in rows if r["leg"] == "lid_spot"]
    assert spots[1]["xyz"] is None
    # this arm has no walk: said, with why the planner's own path is no way either
    assert (
        "the carry (walked: no turning walk modelled; planned: the lid would sink 40 mm under the height it is "
        "carried at on the way, to 110 mm over the table)"
    ) in spots[1]["refused"]
    out = capsys.readouterr().out
    assert "and no other spot on the table the cameras measured has a carry that plans" in out
    assert out.rstrip().endswith("the lid is back on the box, and the arm is home")


def test_in_the_run_the_lid_goes_from_spot_to_spot_by_the_arms_own_walk_where_one_flies(
    ctx, monkeypatch, tmp_path, capsys
):
    """From a spot the lid was left at, the carry to the next is the arm's own walk — a straight line
    from hover to hover at their height, in the carry walk's steps — where one flies: the planner, which
    knows no table, is not asked for a carry again. Where none flies (every other run of these tests: the
    fake arm has no walk) the planner's path is taken only once it has passed the same checks, and the
    run says which it was."""
    chain = _WalksBack()
    _once_a_spot_is_left(monkeypatch, then=lambda: setattr(chain, "walk_ok", True))
    code, rec, _chain, rows = _full_flow(
        ctx,
        monkeypatch,
        tmp_path,
        [(BUTTON, 0.4), (UP, 9.0)],
        chain=chain,
        extent=ROOMY,
        scene=_table(),
        setdown=["never", "met"],
    )
    assert code == 0 and len(rec.setdowns) == 2
    assert _after_first_touch(rec) == ["lid:back-up", *CARRY, *HOME]
    touch = rec.events.index("place:touch")
    carries = [before for name, before in rec.plans if name == "lid:carry"]
    assert carries and all(before <= touch for before in carries)  # the first spot's alone: none planned after it
    ((_quat, step_m, step_rad),) = chain.walked
    assert (step_m, step_rad) == (rlo.CARRY_WALK_STEP_M, rlo.CARRY_WALK_STEP_RAD)
    second = rec.setdowns[1][0]
    out = capsys.readouterr().out
    assert (
        "the lid's way to [%.2f, %.2f] is the arm's own walk, a straight line at the height it is carried at"
        % (second.xyz[0], second.xyz[1])
    ) in out
    # ... and with no walk, the planner's, said so
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY, scene=_table(), setdown=["never", "met"]
    )
    assert code == 0
    assert "is the planner's (no walk flies: no turning walk modelled): it keeps the carry's height" in (
        capsys.readouterr().out
    )


def test_the_flight_view_reads_the_runs_own_lines_from_spot_to_spot(ctx, monkeypatch, tmp_path, capsys):
    """The flight view reads the run's console by pattern (flight_view.py), and its tests feed it lines
    typed into them: a word changed in the run's line and the view goes quiet with every test green (the
    review of 2026-11-19). Here the lines are the run's own, as _full_flow's run prints them: a first
    spot that feels no table, a second that takes the lid; and the same run cut by the engine's budget —
    with the lid on its way to spot 2, and while the next spot was still being chosen."""
    code, rec, _chain, _rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY, scene=_table(), setdown=["never", "met"]
    )
    out = capsys.readouterr().out
    assert code == 0
    view = _flight_view(out, code)
    i = _view_line(view, r" *\d+ SET DOWN  FAIL  spot 1: no table felt, lid kept")
    assert (
        _view_line(view, r" *\d+ SET DOWN  ok    spot 2 \(\d+ cm from the box\): let go where the lid met the table")
        == i + 1
    )
    assert view[-1].startswith(
        "RESULT  DONE — box open, lid on the table (spot 2: no table felt at 1 before), arm home"
    )
    # cut by the budget with the carry to spot 2 in flight: the console as far as that carry's own line
    lines = out.splitlines()
    (flies,) = (k for k, line in enumerate(lines) if line.startswith("[press_demo] SPOT 2, ONE MOTION: "))
    cut = _flight_view("\n".join(lines[: flies + 1]), 130, timed_out=True, held=True)
    j = _view_line(cut, r" *\d+ CARRY     FAIL  stopped: the run took too long, the lid on its way to spot 2")
    assert cut[j - 1].endswith("SET DOWN  FAIL  spot 1: no table felt, lid kept")
    assert cut[j + 1].startswith("RESULT  FAILED at step") and "(CARRY)" in cut[j + 1]
    assert "lid: in the hand" in cut[-1]
    # ... and cut while the next spot was being chosen: as far as the line that left spot 1
    (left,) = (k for k, line in enumerate(lines) if line.startswith("[press_demo] SET-DOWN: no table felt at spot 1"))
    cut = _flight_view("\n".join(lines[: left + 1]), 130, timed_out=True, held=True)
    j = _view_line(cut, r" *\d+ SET DOWN  FAIL  stopped: the run took too long, the next spot for the lid being chosen")
    assert cut[j - 1].endswith("SET DOWN  FAIL  spot 1: no table felt, lid kept") and "lid: in the hand" in cut[-1]


class _OneSphereBody:
    """A body model with one sphere on a hand link, where joints (x, y, z, ...) put it; one on each link of
    the wrist, half a metre over it; and one on the forearm, under the table (runtime/body.Body's
    spheres_at, paths and index, as reach_lid_off asks them)."""

    index = {"joint_%d" % k: k for k in range(7)}
    paths = {
        "hand": [(None, None, k, k) for k in range(7)],
        "wrist_2": [(None, None, k, k) for k in range(6)],
        "wrist_1": [(None, None, k, k) for k in range(5)],
        "forearm": [(None, None, k, k) for k in range(4)],
    }

    def __init__(self, radius=0.05):
        self.radius = radius

    def spheres_at(self, q):
        x, y, z = float(q[0]), float(q[1]), float(q[2])
        centres = np.array([[x, y, z], [x, y, z + 0.5], [x, y, z + 0.6], [x, y, -1.0]])
        return centres, np.array([self.radius, 0.01, 0.01, 0.5]), ["hand", "wrist_2", "wrist_1", "forearm"]


def test_a_carry_after_the_first_keeps_the_hands_own_body_out_of_what_stands_along_its_way(ctx, monkeypatch):
    """lid_way_refusal asks the lid's disc and a hand straight down over it. From one set-down's attitude
    to the next the hand leans and turns on the way, and its body is not over the lid: hand_way_refusal
    asks the body model's own spheres of the hand and wrist, all along the path as the arm flies it
    (between its waypoints too), by the margin a leaning set-down's hand keeps. A LaterWay asks both of
    every carry (path). Only the hand's and the wrist's links are asked (the forearm's sphere here stands
    under the table all the way); with no body model the lid's own check stands alone."""
    from box_opening.runtime import approach

    body = _OneSphereBody()
    monkeypatch.setattr(approach, "body_check", lambda: body, raising=False)
    monkeypatch.setattr(approach, "_chain", lambda: _HandAtQ(ctx.model, (0.6, 0.0)))
    assert rlo._hand_links(body) == ({"hand", "wrist_1", "wrist_2"}, [])
    table = -0.07
    bare = _standing_ring((0.5, 0.0), 5.0, 6.0)
    post = _block(bare, (0.5, 0.0), 0.30, half=0.011)  # 4 cm square, 0.30 m tall
    a, b = [0.2, 0.2, table + 0.40, 0, 0, 0, 0], [0.8, 0.2, table + 0.40, 0, 0, 0, 0]
    assert rlo.hand_way_refusal(ctx, bare, [a, b], table) is None  # nothing stands
    assert rlo.hand_way_refusal(ctx, post, [a, b], table) is None  # 0.19 m beside the post all the way
    # both ends clear of it and the straight joint line between them through it: asked between the waypoints
    a, b = [0.2, 0.0, table + 0.25, 0, 0, 0, 0], [0.8, 0.0, table + 0.25, 0, 0, 0, 0]
    why = rlo.hand_way_refusal(ctx, post, [a, b], table)
    assert why is not None and why.startswith("the hand would pass 50 mm into what stands 300 mm tall at [0.")
    assert why.endswith("(hand; 20 mm is kept) on the way")

    # over its top by less than is kept, and by more
    def over(mm):
        z = table + 0.35 + 0.001 * mm
        return [[0.2, 0.0, z, 0, 0, 0, 0], [0.8, 0.0, z, 0, 0, 0, 0]]

    assert "would pass within 10 mm of what stands 300 mm tall" in rlo.hand_way_refusal(ctx, post, over(10), table)
    assert rlo.hand_way_refusal(ctx, post, over(25), table) is None
    # a LaterWay's path asks the lid's way first, then the hand's
    way = rlo.LaterWay(ctx, post, table, [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(rlo, "lid_way_refusal", lambda *a_: None)
    assert way.path([a, b]) == why
    monkeypatch.setattr(rlo, "lid_way_refusal", lambda *a_: "the lid would sink")
    assert way.path([a, b]) == "the lid would sink"
    monkeypatch.setattr(approach, "body_check", lambda: None, raising=False)
    assert rlo.hand_way_refusal(ctx, post, [a, b], table) is None


def test_the_clock_is_asked_before_a_searchs_own_work_and_of_every_place_it_looks_at(ctx, monkeypatch):
    """spot_candidates' and wider_candidates' `stop` (the run's clock: SpotSearch.out_of_time). It was
    asked only before a reach check, of a place that had passed everything else: judging the room on the
    map, going through the table's places and passing over hundreds of them (off the table, set aside)
    went by without the clock — 1-3 s on a bare table, far more where little of it passes (the review of
    2026-11-19). It is asked before any of that work, and of every place looked at."""
    from box_opening.runtime import approach

    monkeypatch.setattr(approach, "_chain", lambda: _Chain())
    bare = _bare(monkeypatch, ctx)
    work = []
    for name in ("_spot_tiers", "_table_points"):
        real = getattr(rlo, name)
        monkeypatch.setattr(rlo, name, lambda *a, _real=real, _name=name, **k: work.append(_name) or _real(*a, **k))
    q, no = [0.0] * 7, lambda: True
    assert list(rlo.spot_candidates(ctx, bare, -0.07, q, reach=lambda *a: None, stop=no)) == []
    assert list(rlo.wider_candidates(ctx, bare, -0.07, q, reach=lambda *a: None, stop=no)) == []
    assert work == []
    # every place set aside, none a try: the clock is still asked of each, and ends it
    asked, reaches, passed = [], [], rlo.SpotsPassed()

    def stop():
        asked.append(1)
        return len(asked) > 20

    got = rlo.wider_candidates(
        ctx, bare, -0.07, q, reach=lambda *a: reaches.append(1), stop=stop, aside=lambda d: True, passed=passed
    )
    assert list(got) == [] and reaches == [] and len(asked) == 21
    assert 0 < passed.aside + passed.no_land <= 19  # (the first ask is before the work; of some 700 places)
    asked.clear()
    got = rlo.spot_candidates(ctx, bare, -0.07, q, reach=lambda *a: reaches.append(1), stop=stop, aside=lambda d: True)
    assert list(got) == [] and reaches == [] and len(asked) == 21


# --- the look again, over the lid the wrist sees (RAMMP 1.5, 2026-11-20) -------------------------------------
def _wrist_lid(dx=0.09, dy=-0.16, dz=-0.03):
    """The wrist's own find from the look pose: the lid `dx`, `dy` from where the look was aimed (the scene
    camera's place), its own table under it."""
    top = (BUTTON[0] + dx, BUTTON[1] + dy, BUTTON[2] + dz)
    table = top[2] - 0.093
    return SimpleNamespace(
        pose=ContainerPose(xyz=(top[0], top[1], table), yaw=0.0),
        top_xyz=top,
        table_points=_table(xy=top[:2], r=0.3, z=table),
        table_z_scene=table,
        table_extent=None,
        source="wrist",
        rough=False,
    )


LOOK_FRAMES = []  # each look's frame kept as its aim found no button (press_demo.dump_wrist_frame's tags)


def _lids(monkeypatch, *finds):
    """rp.lid_from_here scripted: each call's (fix, why) in turn; what it was asked with. The frame a look
    that found no button is kept as (press_demo.dump_wrist_frame) is counted in LOOK_FRAMES, not written."""
    from box_opening.tasks import press_demo

    asked, left = [], list(finds)
    LOOK_FRAMES.clear()
    monkeypatch.setattr(press_demo, "dump_wrist_frame", lambda watcher, tag: LOOK_FRAMES.append(tag))

    def lid_from_here(node, ctx_, runner_, watcher, args, locator, scene_fix, aimed_xy):
        asked.append((tuple(aimed_xy), getattr(scene_fix, "source", "scene"), locator is not None))
        fix, why = left.pop(0) if left else (None, "the wrist's frame from the look pose: no counter")
        return fix, why, None if fix is None else press_demo.BY_ONLY

    monkeypatch.setattr(rp, "lid_from_here", lid_from_here)
    return asked


def test_a_look_that_finds_no_button_looks_again_over_the_lid_the_wrist_sees(ctx, monkeypatch, tmp_path, capsys):
    """RAMMP 1.5, 2026-11-20 13:40-13:44: the scene camera's place 16-19 cm from the lid, the look short of it,
    the lid whole in the frame's corner, and four runs ended "no button from the look pose". The wrist's own
    finder has the lid in that frame: the look is planned again over it, flown, and the aim asked there; the run
    goes on with the wrist's lid as its box, on the wrist's own table, the scene camera asked nothing after."""
    lid = _wrist_lid()
    asked = _lids(monkeypatch, (lid, None))
    planned = []
    real = rp.plan_reach
    monkeypatch.setattr(
        rp,
        "plan_reach",
        lambda c, cfg, button, q, **k: (planned.append((c.cpose, tuple(button), k)), real(c, cfg, button, q, **k))[1],
    )
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [None, (BUTTON, 0.4), (UP, 9.0)])
    assert code == 0
    assert rec.events == [
        "reach:look",
        "aim:reach-aim",
        "reach:look",
        "aim:reach-aim",
        *WAY_IN,
        "hold",
        *BACK,
        "aim:reach-after",
        *GRIP,
        *CLOSE,
        *CARRY,
        *HOME,
    ]
    assert asked == [(BUTTON[:2], "scene", True)] and LOOK_FRAMES == ["reach-look"]  # the first look's, kept
    # the look again: planned for the wrist's lid, with the box where the wrist has it and the press proven there
    first, again = planned[0], planned[1]
    assert first[1] == pytest.approx(BUTTON) and first[2] == {"depth": True, "looks": None, "shortest_walk": True}
    assert again[0] is lid.pose and again[1] == pytest.approx(lid.top_xyz) and again[2] == {"depth": True}
    assert ctx.worlds.table_top_z == pytest.approx(lid.table_z_scene)  # the wrist's own table from here on
    (row,) = (r for r in rows if r["leg"] == "look_again")
    assert (row["n"], row["planned"], row["had"], row["from_aimed_mm"]) == (1, True, "the scene camera", 184)
    assert row["lid_xyz"] == pytest.approx(list(lid.top_xyz), abs=1e-4)
    out = capsys.readouterr().out
    assert (
        "LOOK AGAIN 1 of 2: no button where the scene camera had the box; the wrist sees the lid at [0.750, -0.730, "
        "0.370], 184 mm from there — the look moves over it" in out
    )
    # the counter's edges read again on everything the run has seen of it, each camera named once
    assert "COUNTER (the scene camera and the wrist from a look pose)" in out
    assert "COUNTER (the scene camera, the wrist from a look pose and the wrist at the look)" in out
    assert "the wrist at the look and the wrist at the look" not in out
    assert "from the wrist's own find" in out and "from the scene camera's" not in out  # (the AIMED line's prior)
    assert out.rstrip().endswith("arm home")


def test_a_look_again_from_the_scene_cameras_place_measures_the_two_cameras_once(ctx, monkeypatch, tmp_path):
    """The look before was aimed where the scene camera had the box and the wrist has that lid elsewhere: one
    container by both cameras, handed to the check of the two (perception/camera_check.adopt) with the scene
    camera's fix and the wrist's, before the wrist's lid becomes the run's box. A second look again starts from
    the wrist's own place: no pair of the two cameras, nothing measured."""
    from box_opening.perception import camera_check

    a, b = _wrist_lid(), _wrist_lid(dx=0.16, dy=-0.10)
    _lids(monkeypatch, (a, None), (b, None))
    pairs = []

    def adopt(c, locator, scene_fix, wrist_fix, runner=None, by=""):
        pairs.append((getattr(scene_fix, "source", "scene"), wrist_fix, locator is not None, by))

    monkeypatch.setattr(camera_check, "adopt", adopt)
    code, _rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [None, None, None])
    assert code == 2 and [r["n"] for r in rows if r["leg"] == "look_again"] == [1, 2]
    ((source, wrist, had_locator, by),) = pairs
    assert source != "wrist" and wrist is a and had_locator and by


def test_two_looks_again_at_most_and_the_stop_says_so(ctx, monkeypatch, tmp_path, capsys):
    """A lid the wrist's finder has and the aim finds no button on is no container this run opens: after two
    looks again the STOP it was, the arm home, the looks counted in its words."""
    a, b = _wrist_lid(), _wrist_lid(dx=0.16, dy=-0.10)
    asked = _lids(monkeypatch, (a, None), (b, None), (_wrist_lid(dx=0.0, dy=-0.30), None))
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [None, None, None])
    assert code == 2
    assert rec.events == [
        "reach:look",
        "aim:reach-aim",
        "reach:look",
        "aim:reach-aim",
        "reach:look",
        "aim:reach-aim",
        "home",
    ]
    # each look again is asked from where the look before it was aimed, with the wrist's lid as the run's box
    assert asked == [(BUTTON[:2], "scene", True), (a.top_xyz[:2], "wrist", False)]
    assert [r["n"] for r in rows if r["leg"] == "look_again"] == [1, 2] and LOOK_FRAMES == ["reach-look"] * 3
    out = capsys.readouterr().out
    assert "LOOK AGAIN 2 of 2: no button where the wrist's own look had the box" in out
    assert (
        "STOP: no button from the look pose, nor from 2 looks again (no circle; each look again over the lid the "
        "wrist's frame showed; no look again: a reach looks again no more than 2 times) — nothing was pressed — "
        "returning home" in out
    )


def test_with_no_lid_in_the_wrists_frame_the_stop_is_as_it_was_and_says_why_no_look_again(
    ctx, monkeypatch, tmp_path, capsys
):
    asked = _lids(monkeypatch, (None, "the wrist's frame from the look pose: wrist camera: no counter in 1 frame"))
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [None])
    assert code == 2 and rec.events == ["reach:look", "aim:reach-aim", "home"] and len(asked) == 1
    assert not [r for r in rows if r["leg"] == "look_again"] and LOOK_FRAMES == ["reach-look"]
    assert (
        "STOP: no button from the look pose (no circle; no look again: the wrist's frame from the look pose: wrist "
        "camera: no counter in 1 frame) — nothing was pressed — returning home" in capsys.readouterr().out
    )


def test_a_lid_whose_look_does_not_plan_is_no_way_in_from_the_look_pose(ctx, monkeypatch, tmp_path, capsys):
    """The wrist sees the lid and no press plans for it from where the chair stands: the arm goes home, and that
    is what the run says (by hand: its STOP; guided: NO WAY IN, the chair is moved and the run looks again)."""
    real = rp.plan_reach
    for guided, want in ((False, 1), (True, rp.NO_WAY_IN)):
        _lids(monkeypatch, (_wrist_lid(), None))
        calls = []

        def plan_reach(c, cfg, button, q, _calls=calls, **k):
            _calls.append(tuple(button))
            return real(c, cfg, button, q, **k) if len(_calls) == 1 else (None, "30 deg: the look pose (IK_FAIL)")

        monkeypatch.setattr(rp, "plan_reach", plan_reach)
        real_flow = rp.run_reach_press

        def with_guide(node, ctx_, cfg, runner, watcher, args, fix, locator, _g=guided, _flow=real_flow):
            args.guide = _g
            return _flow(node, ctx_, cfg, runner, watcher, args, fix, locator)

        monkeypatch.setattr(rp, "run_reach_press", with_guide)
        code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path / str(guided), [None, (BUTTON, 0.4)])
        monkeypatch.setattr(rp, "run_reach_press", real_flow)
        assert code == want and rec.events == ["reach:look", "aim:reach-aim", "home"]
        (row,) = (r for r in rows if r["leg"] == "look_again")
        assert row["planned"] is False and row["why"] == "30 deg: the look pose (IK_FAIL)"
        out = capsys.readouterr().out
        word = "NO WAY IN" if guided else "STOP"
        assert (
            "%s: no reach-out press plans for the lid the wrist sees from the look pose, at [0.750, -0.730] (30 deg: "
            "the look pose (IK_FAIL)) — nothing was pressed — returning home" % word in out
        )
        assert ctx.no_way_in.startswith("no reach-out press plans for the lid the wrist sees")
        assert ctx.no_way_in_at == pytest.approx(_wrist_lid().top_xyz)  # the place the guidance is handed


def test_a_look_again_is_flown_only_where_the_runs_time_still_holds_it(ctx, monkeypatch, tmp_path, capsys):
    """The engine stops a run at its budget wherever it is. A look again and all a run does after its look need
    RELOOK_S + RELOOK_REST_S: with less left, none is flown, and the STOP says so."""
    import time

    asked = _lids(monkeypatch, (_wrist_lid(), None))
    short = rp.RELOOK_S + rp.RELOOK_REST_S - 5.0
    monkeypatch.setenv(rlo.RUN_BUDGET_ENV, json.dumps({"s": short, "from": round(time.monotonic(), 3), "go": False}))
    code, rec, _chain, _rows = _full_flow(ctx, monkeypatch, tmp_path, [None, (BUTTON, 0.4)])
    assert code == 2 and rec.events == ["reach:look", "aim:reach-aim", "home"] and asked == []
    assert "no look again: no time left for a look again (the run has " in capsys.readouterr().out
    # with the time for it, it is flown
    monkeypatch.setenv(rlo.RUN_BUDGET_ENV, json.dumps({"s": 90.0, "from": round(time.monotonic(), 3), "go": False}))
    code, rec, _chain, _rows = _full_flow(ctx, monkeypatch, tmp_path / "b", [None, (BUTTON, 0.4), (UP, 9.0)])
    assert code == 0 and rec.events[:4] == ["reach:look", "aim:reach-aim", "reach:look", "aim:reach-aim"]


def test_a_look_again_that_stops_part_way_ends_the_run_there(ctx, monkeypatch, tmp_path):
    _lids(monkeypatch, (_wrist_lid(), None))
    ends = {2: ("aborted", {"message": "stopped", "progress": 0.4})}  # the second goal sent: the look again
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [None, (BUTTON, 0.4)], ends=ends)
    assert code == 1 and rec.events == ["reach:look", "aim:reach-aim", "reach:look"]
    assert ctx.client.lets_go  # a compliant leg stopped part way is let go of there


def test_after_a_look_again_the_press_starts_from_the_higher_of_the_two_cameras_lid_tops(
    ctx, monkeypatch, tmp_path, capsys
):
    """Review of 2026-11-20: after a look again the "higher top" guard compared the wrist's top with the wrist's
    aim and the scene camera's top of the same lid was dropped (the wrist read the lid 39-58 mm under the scene
    camera's that day, and reads low on flatbot). The scene camera's top of the lid the look again took goes
    with it, and the press starts from the higher; the counter map after the aim is the last look's own, from
    the joints its frame was shot at."""
    lid = _wrist_lid()  # its top 30 mm under the scene camera's (BUTTON's z)
    _lids(monkeypatch, (lid, None))
    maps = []
    real = rlo.counter_map
    monkeypatch.setattr(
        rlo,
        "counter_map",
        lambda c, w, fix, q, tz, aim=False, more=(): maps.append((fix, list(q), aim, [f for f, _q in more]))
        or real(c, w, fix, q, tz, aim=aim, more=more),
    )
    low = (BUTTON[0], BUTTON[1], BUTTON[2] - 0.05)  # the aim's own height for the button, 50 mm under the scene's
    up = (low[0], low[1], low[2] + 0.009)  # ... and for the popped knob after the press, by the same camera
    code, rec, _chain, _rows = _full_flow(ctx, monkeypatch, tmp_path, [None, (low, 0.4), (up, 9.0)])
    assert code == 0 and ctx.scene_top_z == pytest.approx(BUTTON[2])
    out = capsys.readouterr().out
    assert "the scene camera puts the lid 50 mm higher than the wrist's aim: pressing from that height" in out
    assert "the wrist's own find puts the lid" not in out
    # the three maps: the first look's (the scene camera's fix), the look again's, and the aim's after it
    assert [(m[2], len(m[3])) for m in maps] == [(False, 0), (False, 1), (True, 1)]
    assert maps[1][0] is lid and maps[2][0] is lid and maps[1][1] == maps[2][1] != maps[0][1]  # shot from the look pose
    assert maps[1][3][0].top_xyz == pytest.approx(BUTTON)  # ... pooled with the scene camera's


def test_a_wrist_fix_presses_from_the_scene_cameras_top_only_where_main_handed_it_one(
    ctx, monkeypatch, tmp_path, capsys
):
    """The run's own fix the wrist's (its look from rest): with the scene camera's top of the same lid
    (ctx.scene_top_z: main, after the wrist's look) the press starts from the higher of the three; without one,
    from the higher of the wrist's find and its aim, as before, and said as what it is."""
    real = rp.run_reach_press
    low = (BUTTON[0], BUTTON[1], BUTTON[2] - 0.05)
    up = (low[0], low[1], low[2] + 0.009)  # the popped knob after the press, as the same camera reads it
    for find_dz, scene_top, said in (
        # the wrist's find 30 mm under the scene camera's top of the same lid: the scene camera's is the higher
        (-0.03, BUTTON[2], "the scene camera puts the lid 50 mm higher than the wrist's aim"),
        (0.0, None, "the wrist's own find puts the lid 50 mm higher than the wrist's aim"),
        (0.0, BUTTON[2] - 0.10, "the wrist's own find puts the lid 50 mm higher than the wrist's aim"),
    ):

        def by_the_wrist(node, ctx_, cfg, runner, watcher, args, fix, locator, _dz=find_dz, _top=scene_top, _real=real):
            fix.source, ctx_.scene_top_z = "wrist", _top
            fix.top_xyz = (BUTTON[0], BUTTON[1], BUTTON[2] + _dz)
            return _real(node, ctx_, cfg, runner, watcher, args, fix, None)

        monkeypatch.setattr(rp, "run_reach_press", by_the_wrist)
        where = tmp_path / ("%s-%s" % (find_dz, scene_top))
        code, _rec, _chain, _rows = _full_flow(ctx, monkeypatch, where, [(low, 0.4), (up, 9.0)])
        monkeypatch.setattr(rp, "run_reach_press", real)
        out = capsys.readouterr().out
        assert code == 0 and said + ": pressing from that height" in out, (scene_top, out[-1500:])
        assert "from the wrist's own find" in out  # the AIMED line names the prior it measured from
    # a scene fix flown and aimed, as ever: its own top, said as the scene camera's
    code, _rec, _chain, _rows = _full_flow(ctx, monkeypatch, tmp_path / "scene", [(low, 0.4), (up, 9.0)])
    out = capsys.readouterr().out
    assert code == 0 and "the scene camera puts the lid 50 mm higher than the wrist's aim" in out
    assert "0 mm from the scene camera's" in out and "from the wrist's own find" not in out
