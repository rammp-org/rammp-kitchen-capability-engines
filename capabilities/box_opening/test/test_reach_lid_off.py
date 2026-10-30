"""The reach-out's lid off (tasks/reach_lid_off): the compliant reach-out
press as the whole run — the knob looked at from the look pose, the press
once more where it is not up, the grip, the lift, the carry and the
set-down — with the arm, the cameras and the planner faked; and its
reachability on flatbot's URDF for the boxes of 2026-10-27/28."""

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import test_reach_press as trp

from box_opening.models.container import ContainerModel, ContainerPose, load_press_demo
from box_opening.tasks import depth_press as dp
from box_opening.tasks import reach_lid_off as rlo
from box_opening.tasks import reach_press as rp

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
    scene=None,
    refuse_carry=False,
    rise=None,
    at_spot_look=None,
    one_motion=True,
    from_press=False,
):
    """run_reach_press as a full run (compliant --depth-press, no
    --press-only), everything faked: `aims` the wrist's aims in order, each
    (press point, knob mm) or None — the look's, then each after a press
    (and its recheck); `knuckle` what closing on the knob reads (a list:
    each close's in turn, the last for every one after), `slipped`
    whether the fingers read that still at the table; `extent` the table
    the scene camera saw (no room map: the lid goes beside the box, in its
    zone); `ends`: {n: (outcome, info)} for the n-th goal sent; `wrist`
    the wrist's points at the spot look, `scene` the scene camera's (None:
    no camera data); `rise` depth_press.rise_back_up's stand-in (None:
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

    def hold(ctx_, cfg_, runner_, stroke, top_z, seen):
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

    def aim(node, watcher, yaw, lid_z, prior_xy, runner_, tag):
        rec.events.append("aim:" + tag)
        got = queue.pop(0) if queue else None
        if got is None:
            return None, "no circle", None, None
        return (got[0], 0.0), "3/3 frames", got[1], got[0]

    monkeypatch.setattr(rp, "_aim", aim)
    closes = list(knuckle) if isinstance(knuckle, list | tuple) else [knuckle]
    held = {"on": True, "reads": closes[0]}

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
        owl=SimpleNamespace(enable=lambda: None, disable=lambda: None),
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
# ... flown as two: the release riding the retreat, then home in POSITION (_in_position)
RELEASED, HOMED = ["place:open", "place:retreat"], ["home"]


def _executions(rec):
    """Each runner.run's legs with the gains of every execution it sent."""
    return [(names, gains) for names, gains in rec.runs]


class _Video:
    """camera_video.CameraVideo's stand-in: what the run says on it, and when."""

    def __init__(self, sources, out_dir, rec=None, ctx=None):
        self.sources, self.out_dir, self.rec, self.ctx = [n for n, _g in sources], out_dir, rec, ctx
        self.said, self.started, self.finished = [], None, None

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


def test_a_ctrl_c_with_the_lid_in_the_fingers_lets_go_first_then_writes_the_video(ctx, monkeypatch, tmp_path):
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
    caption, lets_go = v.finished
    assert caption == "stopped: KeyboardInterrupt" and lets_go >= 1  # the arm let go before the video was written


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
    # release rides the retreat (sent, not waited on), and home after it flies in POSITION
    for names, gains in runs[1:]:
        assert gains == [dp.COMPLIANT_GAINS], names
    assert modes == [True] * 6 + [False] and rec.legs["place:open"].defer_join
    assert all(lg.guard is None for lg in rec.legs.values() if lg.kind.value == "motion")
    # the carry and home after the set-down planned once each, while the grip's way down flew (the record then
    # ended with its legs); home before that from the look pose while the way back flew, for a stop there
    planned = {name: at for name, at in rec.plans}
    gripping = rec.events.index("grip:close")
    assert planned["lid:carry"] == planned["home"] == gripping
    assert [name for name, _at in rec.plans].count("lid:carry") == 1
    assert [at for name, at in rec.plans if name == "home"] == [rec.events.index("aim:reach-after"), gripping]
    # the spot chosen ahead, said before the close: no camera data, so beside the box
    legs = [r["leg"] for r in rows]
    assert legs.index("reach_grip") < legs.index("lid_spot") < legs.index("reach_grip_result")
    (spot,) = (r for r in rows if r["leg"] == "lid_spot")
    assert (spot["tier"], spot["mapped"], spot["chosen"]) == ("beside the box", False, "ahead, with the scene camera")
    assert spot["room_s"] is not None  # the room judged while the press flew
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


def test_where_nothing_was_planned_ahead_the_lift_flies_alone_and_the_spot_is_looked_at_after_it(
    ctx, monkeypatch, tmp_path, capsys
):
    """No spot whose carry and home both plan found while the hand came
    down to the knob (here: none looked for): as before, the close, the
    lift and the clear, then the lid's one stop, the spot look; the carry
    and the set-down; the release, the retreat and home — six executions
    after the look's, every one RIGID and unguarded."""
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
    # one goal each, every one under the run's one gains spec
    for names, gains in runs[1:]:
        assert gains == [dp.COMPLIANT_GAINS], names
    assert all(lg.guard is None for lg in rec.legs.values() if lg.kind.value == "motion")
    # the carry planned while the lift and the clear flew (carry_ahead), before the spot look, and flown: the spot
    # chosen with the lid lifted is its own; home while the carry flew
    planned = {name: at for name, at in rec.plans}
    assert planned["lid:carry"] == rec.events.index("lid:clear") + 1 == rec.events.index("spot-look")
    assert [name for name, _at in rec.plans].count("lid:carry") == 1
    assert rec.events.index("place:touch") < planned["home"] <= rec.events.index("place:open")
    # ... and from the look pose while the way back flew, for a stop there: unused, the lid coming off
    assert [at for name, at in rec.plans if name == "home"] == [rec.events.index("aim:reach-after"), planned["home"]]
    # nothing chose the lid's spot before the lift: no camera data, so beside the box
    assert [r["leg"] for r in rows].index("lid_spot") > [r["leg"] for r in rows].index("reach_grip_result")
    (spot,) = (r for r in rows if r["leg"] == "lid_spot")
    assert spot["tier"] == "beside the box" and spot["mapped"] is False
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
        "nothing to hold, so the lid is not taken off; the box is left open: lift the lid off by hand, or shut it, "
        "before the next run (which would take it for shut and press it) — returning home" % (1000 * dz, knob_mm)
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
    aims = [(BUTTON, 0.4), (BUTTON, 2.0), (BUTTON, 1.5), (BUTTON, 3.0), (BUTTON, 2.5)]
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
    assert "STOP: pressed 2 times and the knob is not up (3.0 mm above the lid) — the lid is not taken off" in out
    assert ctx.client.gripper_sent == []  # the fingers never opened


BACK_OUT = ["grip:up", "depth:turn-back", "reach:slide-out"]  # the grip's way down backwards, to the look pose


def test_fingers_that_close_on_air_open_again_and_back_out_the_way_they_came(ctx, monkeypatch, tmp_path, capsys):
    """The close reads air (outside the grip band): nothing is lifted. The
    fingers open, the hand backs out the grip's way down to the look pose,
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
        "grip:open",
        *BACK_OUT,
        "aim:reach-regrip",
        *GRIP,
        *CLOSE,
        *CARRY,
        "grip:open",
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
    1. Now the fingers open, the hand backs out to the look pose, the wrist
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
        "grip:open",
        *BACK_OUT,
        "aim:reach-regrip",
        *GRIP,
        *CLOSE,
        *CARRY,
        *HOME,
    ]
    assert rec.events.count("aim:reach-regrip") == 1 and rec.events.count("grip:close") == 2
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
    """The second close reads air as well: the fingers open, the hand backs
    out to the look pose once more and goes home from there, exit 1 — never
    a third grip, nor a second re-aim, whatever the wrist would see."""
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
        "grip:open",
        *BACK_OUT,
        "aim:reach-regrip",
        *GRIP,
        *CLOSE,
        *CARRY,
        "grip:open",
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
        "grip:open",
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

    def straight_line(self, q0, xyz, quat):
        pts, why = super().straight_line(q0, xyz, quat)
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
    table keep their own speeds, the planned ones TRANSIT_SPEED. Outside
    the run, the client as press_demo built it, free_line_speed is a
    guarded run's FREE_LINE_SPEED."""
    from box_opening.constants import (
        COMPLIANT_FREE_LINE_SPEED,
        COMPLIANT_JUNCTION_SPEED,
        COMPLIANT_QD_CEILING,
        FREE_LINE_SPEED,
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

    def retime_group(trajs, speeds, vmax, params=RetimeParams()):  # noqa: B008 - frozen dataclass
        timed.append((list(speeds), params))
        return real_group(trajs, speeds, vmax, params)

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
            "home": TRANSIT_SPEED,
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
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], slipped=True)
    assert code == 1 and rec.events[-3:] == HOME
    (setdown,) = (r for r in rows if r["leg"] == "reach_setdown")
    assert setdown["kept"] is False
    assert "the lid slipped out of the fingers" in capsys.readouterr().out


def test_a_box_the_press_moved_is_not_gripped(ctx, monkeypatch, tmp_path, capsys):
    moved = (BUTTON[0] + 0.02, BUTTON[1], BUTTON[2] + 0.009)
    code, rec, _chain, _rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (moved, 9.0)])
    assert code == 1 and rec.events[-2:] == ["aim:reach-after", "home"]
    assert "STOP: the press MOVED the box 20 mm" in capsys.readouterr().out


def test_a_box_the_wrist_no_longer_finds_after_the_press_is_not_pressed_again(ctx, monkeypatch, tmp_path, capsys):
    code, rec, _chain, _rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), None, None])
    assert code == 1 and rec.events[-3:] == ["aim:reach-after", "aim:reach-after", "home"]
    assert rec.events.count("depth:press") == 1
    assert "the knob is not judged" in capsys.readouterr().out


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
        "grip:open",
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


# ---- home, planned from the look pose while the way back flies -----------------

MOVED = (BUTTON[0] + 0.02, BUTTON[1], BUTTON[2] + 0.009)  # the aim after a press that moved the box 20 mm
NOT_UP_TWICE = [(BUTTON, 0.4), (BUTTON, 2.0), (BUTTON, 1.5), (BUTTON, 3.0), (BUTTON, 2.5)]


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
        ([(BUTTON, 0.4), None, None], 1, True),
        (NOT_UP_TWICE, 2, True),
        ([(BUTTON, 0.4), (UP, 9.0)], 1, False),
    ],
    ids=["the box moved", "the box not found", "not up after the second press", "no way down to the knob"],
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


def test_nothing_clear_still_sets_the_lid_down_at_the_most_room_seen(ctx, monkeypatch, tmp_path, capsys):
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY, wrist=_wall(_table()), one_motion=False
    )
    assert code == 0 and rec.events[-3:] == HOME
    (spot,) = (r for r in rows if r["leg"] == "lid_spot")
    assert spot["tier"] == "most room"
    out = capsys.readouterr().out
    assert "nothing reads clear" in out and "STOP" not in out and out.rstrip().endswith("arm home")


def test_no_camera_data_still_sets_the_lid_down_beside_the_box(ctx, monkeypatch, tmp_path, capsys):
    code, _rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY)
    assert code == 0
    (spot,) = (r for r in rows if r["leg"] == "lid_spot")
    assert (spot["tier"], spot["mapped"], spot["wrist_points"]) == ("beside the box", False, None)
    assert "no camera shows the table: beside the box" in capsys.readouterr().out


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
    real = clear_table.clearance

    def counted(*a, **k):
        judged_on.append(threading.current_thread().name)
        return real(*a, **k)

    monkeypatch.setattr(clear_table, "clearance", counted)
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
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY, scene=_wall(_table())
    )
    assert code == 0 and [names for names, _g in _executions(rec)][-3:] == [CLOSE + CARRY, RELEASED, HOMED]
    (spot,) = (r for r in rows if r["leg"] == "lid_spot")
    assert spot["tier"] == "most room"
    out = capsys.readouterr().out
    assert "nothing reads clear" in out and "STOP" not in out and out.rstrip().endswith("arm home")


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


def test_no_carry_that_plans_holds_the_lid_and_says_so(ctx, monkeypatch, tmp_path, capsys):
    """Every spot the hand reaches refused by the planner: the run stops
    holding the lid (exit 1) after CARRY_TRIES, not before the run."""
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY, refuse_carry=True
    )
    assert code == 1 and "lid:carry" not in rec.events
    (spot,) = (r for r in rows if r["leg"] == "lid_spot")
    assert spot["xyz"] is None and spot["refused"].count("IK_FAIL") == rlo.CARRY_TRIES
    assert "STOP: no carry to a spot the lid could go to plans" in capsys.readouterr().out


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
    rlo.spot_look(node, SimpleNamespace(grab=grab), runner)
    assert notes[-1][1]["why"] == "no camera pose at the frame's stamp (TF)"
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

    def candidates(ctx_, tmap, table_z, q, seeds=(), reach=None):
        seen.append(("spot_candidates", list(q)))
        yield SimpleNamespace(xyz=(0.45, -0.25, -0.027)), "beside the box", "beside the box"

    monkeypatch.setattr(rlo, "spot_look", look)
    monkeypatch.setattr(rlo, "spot_map", lambda ctx_, wrist, q, *rest: seen.append(("spot_map", list(q))))
    monkeypatch.setattr(rlo, "spot_candidates", candidates)
    monkeypatch.setattr(
        rlo, "plan_carry", lambda ctx_, cfg_, q, drop: (seen.append(("plan_carry", list(q))), (None, "refused"))[1]
    )
    eyes = (None, SimpleNamespace(table_z=-0.027), None, None)
    assert rlo.choose_and_carry(ctx, load_press_demo(PINK), Runner(ctx.client, log_dir=tmp_path), eyes) == 1
    assert seen == [("spot_map", stood), ("spot_candidates", stood), ("plan_carry", stood)]


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
    the wrist shows (owner: chosen with the lid lifted), not the scene's, and
    the carry is planned there at rest as before — the one planned while
    the lift flew is not flown, and the record says why."""
    code, rec, _chain, rows = _full_flow(
        ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], one_motion=False, wrist=_wall(_table(r=0.4)), **NEAR
    )
    assert code == 0
    flight = _lift_flight(rec)
    carries = [at for name, at in rec.plans if name == "lid:carry"]
    assert carries[0] == flight and len(carries) == 2 and carries[1] > rec.events.index("spot-look")
    assert rec.most_at_once == 1
    (spot,) = (r for r in rows if r["leg"] == "lid_spot")
    ahead = spot["ahead"]
    assert spot["tier"] == "most room" and ahead["flown"] is False
    assert ahead["why"].startswith("the spot chosen with the lid lifted is ")
    assert math.hypot(spot["xyz"][0] - ahead["xyz"][0], spot["xyz"][1] - ahead["xyz"][1]) > rlo.AHEAD_SAME_SPOT_M
    assert rec.legs["lid:carry"].target[1][:2] == pytest.approx(spot["xyz"][:2])  # the carry flown is to the new spot
    assert "the carry planned while the lift flew is not flown" in capsys.readouterr().out


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
    waits so."""
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)])
    assert code == 0
    assert [name for name, lg in rec.legs.items() if lg.send_when_still] == ["grip:close"]
    # after the look, the way in, the way back and the way down to the knob; and before home in POSITION
    assert ctx.client.stills == [4, 6]
    legs = [r["leg"] for r in rows]
    stills = [i for i, name in enumerate(legs) if name == "still"]
    assert [rows[i]["before"] for i in stills] == ["grip:close", "home in position"]
    assert legs.index("grip:down") < legs.index("grip:open") < stills[0] < legs.index("grip:close")
    assert legs.index("place:retreat") < stills[1] < legs.index("home")


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


def test_the_lid_is_pressed_onto_the_table_no_harder_than_its_push_and_never_past_two_millimetres(monkeypatch):
    """Owner 2026-10-29: the lid is set down "too hard aka the arm is moving
    too far down towards the table and pressing against it abnormally
    hard". 6 mm past the table at the hand's 1150-2290 N/m was 11-30 N on
    it. The touch ends SETDOWN_PUSH_N (5 N) on the spring straight down at
    the fingertips past where the lid meets the table, at most
    SETDOWN_OVER_M (2 mm): a stiff hand goes less deep, a soft one no
    deeper than 2 mm."""
    assert rlo.SETDOWN_PUSH_N == pytest.approx(5.0) and rlo.SETDOWN_OVER_M == pytest.approx(0.002)
    for k, over in ((5000.0, 0.001), (2500.0, 0.002), (1150.0, 0.002), (float("inf"), 0.0)):
        monkeypatch.setattr(dp, "fingertip_stiffness_z", lambda chain, q, k=k: k)
        got, k_z = rlo.setdown_over(None, None)
        assert got == pytest.approx(over) and k_z == k


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
    run_reach_press's proof; at two of the five): every leg after the look
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
    if shortest_walk:  # two of the five take another elbow: 3.1 and 1.8 s quicker over the transit and three walks
        assert (plan.posture["chosen"] > 0) is (box in ((0.668, -0.33), (0.60, -0.49))), plan.posture
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
    drop, tier, _says = next(rlo.spot_candidates(ctx, None, LID - 0.10, q_look, seeds=(home,)))
    assert tier == "beside the box"
    carry, why = rlo.plan_carry(ctx, cfg, q_look, drop)
    assert carry is not None, why
    # the touch ends past where the lid meets the table no deeper than SETDOWN_PUSH_N on the hand's spring there
    _quat, contact, _hover = rlo.setdown_poses(ctx, drop)
    _R, t_touch = chain.fk(carry.legs[2].goal_joints)
    assert t_touch[2] == pytest.approx(contact[2] - carry.over_m, abs=5e-4)
    assert 0.0 < carry.over_m <= rlo.SETDOWN_OVER_M and carry.k_z * carry.over_m <= rlo.SETDOWN_PUSH_N + 1e-9
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
        # what Runner._run_motion sends for the execution `group` (one profile over it, a lone free-air line its
        # own) and retime_group's report of it; its free-air lines at FREE_LINE_SPEED where `slow`
        group = [
            replace(lg, speed=FREE_LINE_SPEED) if slow and lg.speed == COMPLIANT_FREE_LINE_SPEED else lg for lg in group
        ]
        if len(group) > 1:
            return retime_group([lg.traj for lg in group], [lg.speed for lg in group], JOINT_VMAX, p)
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
    drop, _tier, _says = next(rlo.spot_candidates(ctx, None, LID - 0.10, lift.legs[-1].goal_joints, seeds=(home,)))
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
    width = m.button_diameter_m + 2 * rlo.GRIP_SIDE_CLEAR_M
    assert width == pytest.approx(0.064)
    assert rec.legs["grip:open"].gripper_cmd == pytest.approx(rlo.grip_open_cmd(m))
    assert GRIPPER_CMD_OPEN < rlo.grip_open_cmd(m) < GRIPPER_CMD_CLOSED
    assert rlo.grip_open_cmd(m) == pytest.approx(GRIPPER_CMD_CLOSED * (1 - 0.064 / rlo.GRIPPER_OPEN_WIDTH_M))
    # the fingertips' place across the knob: inside the lid's edge by more than the knob's aim can be off
    lid_r = 0.5 * m.lid_dims[0]
    assert lid_r - width / 2 >= 0.02
    assert width / 2 - m.button_diameter_m / 2 >= 0.01
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

    img = cv2.imread(str(Path(__file__).parent / "data" / name))
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

    img = cv2.imread(str(Path(__file__).parent / "data" / "wrist_grip_close_flatbot_20261030_1059_air.jpg"))
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

FROM_PRESS = ["grip:open", "grip:rise"]


def test_the_grip_straight_from_the_press_takes_the_lid_off_with_no_way_back_and_no_look(
    ctx, monkeypatch, tmp_path, capsys
):
    """Owner 2026-10-30: after the first press's hold the hand rises
    GRIP_RISE_M, the fingers opening to the knob's width as it does, comes
    down the stroke's line to the grip's height — GRIP_LID_ABOVE_HELD_M
    over where the hold held the pads, less GRIP_BELOW_LID_M — and closes:
    no way back to the look pose, no look, no way down from it (10:08's
    were 2.6 + 0.7 + 3.2 s). The lift, the spot, its carry and home are
    planned while the hand rises; the fingers are open before it comes
    down; the rest as from the look pose."""
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
    assert planned["lid:carry"] == planned["home"] == rec.events.index("grip:down")  # while the hand rose
    assert rec.legs["grip:open"].defer_join and rec.legs["grip:open"].gripper_cmd == pytest.approx(
        rlo.grip_open_cmd(ctx.model)
    )
    (grip,) = (r for r in rows if r["leg"] == "reach_grip")
    held = BUTTON[2] - 0.006  # the fake hold's pads
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
    line, then down it to the pads GRIP_LID_ABOVE_HELD_M - GRIP_BELOW_LID_M
    over where they were held, the stroke's attitude."""
    asked = []

    class Chain:
        def fk(self, q):
            return np.eye(3), np.array([0.6 + q[0], -0.2 + q[1], 0.30 + q[2]])

        def straight_line(self, q0, xyz, quat):
            asked.append((list(xyz), list(quat)))
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
