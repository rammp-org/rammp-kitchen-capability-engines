import math
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import FakeClient

from box_opening.models.container import ContainerPose
from box_opening.runtime.legs import Kind

CFG = "capabilities/box_opening/config/containers/oxo_pop.yaml"


def names(legs):
    return [leg.name for leg in legs]


def test_entry_points_registered():
    setup = Path("capabilities/box_opening/setup.py").read_text()
    for ep in ["box_opening_node", "press_demo", "home_arm", "preflight", "owl_detector"]:
        assert ep + " = " in setup


def test_home_arm_plans_home_in_the_bench_world(ctx):
    """The isolated recovery home knows no container pose: it plans above
    the unseen-container band, like the mission's own recovery home."""
    from box_opening.constants import HOME, TRANSIT_SPEED
    from box_opening.tasks import home_arm

    c = ctx
    c.cpose = None
    legs = home_arm.build_legs(c)
    assert names(legs) == ["home"]
    assert legs[0].world == "bench" and legs[0].speed == TRANSIT_SPEED
    assert legs[0].target == ("joints", list(HOME))


def test_home_arm_with_a_known_table_plans_guarded_without_a_caution(ctx, capsys):
    from box_opening.tasks import home_arm

    legs = home_arm.build_legs(ctx)
    assert legs[0].world == "bench"
    assert "CAUTION" not in capsys.readouterr().out


class _PlanCounter:
    """Wraps a FakeClient's plan_to_joints: counts the plans, and records
    whether anything had been printed before each one."""

    def __init__(self, client, capsys, succeed=True):
        self.n = 0
        self.out = ""
        self.printed_first = []
        inner = client.plan_to_joints

        def plan(q7, start_joints):
            self.n += 1
            self.out += capsys.readouterr().out
            self.printed_first.append(bool(self.out))
            if not succeed:

                class Bad:
                    success = False
                    message = "INVALID_START_STATE_WORLD_COLLISION"

                return Bad()
            return inner(q7, start_joints)

        client.plan_to_joints = plan


def test_home_arm_on_a_tableless_world_cautions_first_and_plans_once(ctx, capsys):
    """Flatbot 2026-10-22: world_flatbot.yaml has no table and home_arm
    measures none, so the 'guarded' bench world was the floor alone, the
    same as the bare one — a recovery home planned blind with no caution,
    and a refused plan re-planned in the identical world."""
    from box_opening.constants import HOME
    from box_opening.tasks import home_arm

    c = ctx
    c.worlds.table_top_z = None
    plans = _PlanCounter(c.client, capsys)
    legs = home_arm.build_legs(c)
    assert names(legs) == ["home"] and legs[0].world == "bench_bare"
    assert legs[0].target == ("joints", list(HOME))
    assert [kw.get("model") for kind, kw in c.worlds.pushes if kind == "bench"] == [None]
    assert plans.n == 1 and plans.printed_first == [True]  # the caution came BEFORE the plan
    out = plans.out + capsys.readouterr().out
    assert "CAUTION" in out and "neither the table nor the box" in out and "--lift-first" in out


def test_home_arm_on_a_tableless_world_refused_raises_without_a_second_plan(ctx, capsys):
    import pytest

    from box_opening.tasks import home_arm

    c = ctx
    c.worlds.table_top_z = None
    plans = _PlanCounter(c.client, capsys, succeed=False)
    with pytest.raises(RuntimeError):
        home_arm.build_legs(c)
    assert plans.n == 1


def _demo_cfg():
    from box_opening.models.container import load_press_demo

    return load_press_demo(CFG)


def test_the_search_looks_then_sweeps_each_way(ctx):
    """The look is the arm's own rest pose with the wrist turned down — not a
    bench Cartesian pose, which measured 3.89 s against 1.72 s — and when it
    sees nothing the base pans each way to find the box."""
    import pytest

    from box_opening.constants import HOME
    from box_opening.models.container import load_press_demo
    from box_opening.primitives.look import look_joints
    from box_opening.tasks import press_demo

    cfg = load_press_demo(CFG)
    steps = press_demo.search_targets(cfg)
    assert [name for name, _q, _speed in steps] == ["look", "sweep:left", "sweep:right"]
    assert steps[0][1] == pytest.approx(look_joints(HOME))
    # the sweep is slower than the look: it exists to see, not to arrive
    assert steps[1][2] < steps[0][2]


def test_a_search_leg_is_joint_space_and_ends_on_a_sighting(ctx):
    """Joint-space so no bench frame is involved, bench world so a transit
    before detection stays above anything that could be standing there, and
    stop_when so the motion ends the moment the camera sees a box."""
    from box_opening.constants import HOME, TRANSIT_SPEED
    from box_opening.primitives.look import look_joints
    from box_opening.tasks import press_demo

    seen = {"fix": None}

    class Watcher:
        def coarse_fix(self):
            return seen["fix"]

    leg = press_demo.build_search_leg(ctx, ("look", look_joints(HOME), TRANSIT_SPEED), [0.0] * 7, Watcher())
    assert leg.target[0] == "joints"
    assert leg.world == "bench" and leg.kind is Kind.MOTION
    assert leg.stop_when() is False
    seen["fix"] = ((0.45, 0.0, 0.085), 0.0)
    assert leg.stop_when() is True


def test_press_demo_no_tag_home_uses_the_bench_world(ctx):
    from conftest import FakeStore

    from box_opening.tasks import press_demo

    c = replace(ctx, worlds=FakeStore())
    home = press_demo.build_home_leg(c, [0.0] * 7)
    assert home.world == "bench"


def test_lid_place_clearance_gate_threshold(ctx):
    import pytest

    from box_opening.tasks import press_demo

    c = ctx
    need = press_demo.lid_place_min_clear(c.model)
    # both footprint half-diagonals + gripper-body room; the field runs on
    # the 2.9-inch box bracketed it: IK_FAIL at 0.073 m separation, clean
    # plan at 0.162 m (0.156 for that box). The shipped 4.1-inch box needs
    # 0.197, and the axis spots at that distance planned 32/32 (2026-09-17)
    assert 0.073 < need < 0.25
    import math

    assert need == pytest.approx((math.hypot(*c.model.dims[:2]) + math.hypot(*c.model.lid_dims[:2])) / 2 + 0.05)


def _first_drop(m, cpose, table_z, prefer_xy):
    """The first of lid_drop_candidates, or (None, None): what
    press_demo.box_relative_lid_drop returned, before it went with the
    staging run (2026-10-28)."""
    from box_opening.tasks import press_demo

    spots = press_demo.lid_drop_candidates(m, cpose, table_z, prefer_xy)
    return spots[0] if spots else (None, None)


def test_the_lid_goes_down_beside_the_box_wherever_the_box_is(ctx):
    """The drop spot is derived from the DETECTED box every run, never
    surveyed: a fixed base-frame point assumes a fixed table, which is the
    thing a wheelchair takes away. The configured lid_place survives only as
    the side to try first, and the table height comes from the camera."""
    from box_opening.tasks import press_demo

    c = ctx
    m = c.model
    need = press_demo.lid_place_min_clear(m)
    prefer = [0.45, -0.25, -0.027]  # the configured side: -y
    for box_xy in ((0.45, 0.1), (0.396, -0.24), (0.45, -0.25), (0.35, 0.2)):
        box = ContainerPose(xyz=(box_xy[0], box_xy[1], -0.02), yaw=0.0)
        xyz, _side = _first_drop(m, box, -0.031, prefer)
        assert xyz is not None
        assert math.hypot(xyz[0] - box_xy[0], xyz[1] - box_xy[1]) >= need - 1e-9
        assert press_demo.DROP_X_M[0] <= xyz[0] <= press_demo.DROP_X_M[1]
        assert press_demo.DROP_Y_M[0] <= xyz[1] <= press_demo.DROP_Y_M[1]
        assert xyz[2] == -0.031  # the table the camera measured, not the yaml
    # the preferred side is the one taken when it fits
    middle = ContainerPose(xyz=(0.45, 0.0, -0.02), yaw=0.0)
    xyz, _side = _first_drop(m, middle, -0.027, prefer)
    assert xyz[1] < 0.0
    # a box well outside the set-down zone leaves nowhere to put the lid:
    # refuse and say so, rather than invent a spot the arm cannot reach
    far_out = ContainerPose(xyz=(0.90, 0.0, -0.02), yaw=0.0)
    xyz, _side = _first_drop(m, far_out, -0.027, prefer)
    assert xyz is None


def test_the_lid_side_snaps_to_the_base_axis_nearest_the_preferred_one(ctx):
    """Field 2026-09-17: box at (0.5625, -0.0247), lid_place (0.45, -0.25)
    — the exact direction is 27 deg off -y, and the carry pose there had no
    IK in any attitude (the planner's padded container is a base-aligned
    cuboid, so a diagonal spot sits nearer its corner). The -y axis spot
    beside the same box plans."""
    import pytest

    from box_opening.tasks import press_demo

    m = ctx.model
    need = press_demo.lid_place_min_clear(m)
    box = ContainerPose(xyz=(0.5625, -0.0247, -0.027), yaw=math.radians(10.3))
    xyz, side = _first_drop(m, box, -0.027, [0.45, -0.25, -0.027])
    assert side == (0.0, -1.0)
    assert xyz[0] == pytest.approx(0.5625) and xyz[1] == pytest.approx(-0.0247 - need)
    # the base-axis spots through the box first, preferred side first; the
    # diagonals only after every axis spot (2026-10-23)
    spots = press_demo.lid_drop_candidates(m, box, -0.027, [0.45, -0.25, -0.027])
    dirs = [d for _xyz, d in spots]
    assert dirs[0] == (0.0, -1.0)
    axis = [d in ((0.0, -1.0), (0.0, 1.0), (1.0, 0.0), (-1.0, 0.0)) for d in dirs]
    assert axis == sorted(axis, reverse=True)


def test_a_box_at_the_edge_of_the_seen_table_gets_a_diagonal_drop_spot():
    """Flatbot 2026-10-23 16:31: the box at [0.39, -0.20] on the edge of the
    table the scene camera saw, the set-down zone x 0.43-0.63, y -0.63 to
    -0.23: every axis spot fell outside it ("no lid drop spot ... move the
    box"). A diagonal one, LID_DIAG_EXTRA_M farther out, fits."""
    from box_opening.models.container import ContainerModel, ContainerPose
    from box_opening.tasks import press_demo

    m = ContainerModel.load("capabilities/box_opening/config/containers/ankou_pink.yaml")
    box = ContainerPose(xyz=(0.391, -0.196, 0.184), yaw=0.0)
    zone = (0.43, 0.63, -0.63, -0.23)
    ((xyz, d),) = press_demo.lid_drop_candidates(m, box, 0.184, zone=zone)
    assert d[0] > 0 and d[1] < 0  # toward +x, -y: into the seen table
    assert math.hypot(xyz[0] - 0.391, xyz[1] + 0.196) == pytest.approx(
        press_demo.lid_place_min_clear(m) + press_demo.LID_DIAG_EXTRA_M
    )
    assert zone[0] <= xyz[0] <= zone[1] and zone[2] <= xyz[1] <= zone[3]


def test_the_lid_goes_where_the_cameras_saw_room_not_onto_the_microwave(monkeypatch):
    """Flatbot 2026-10-23 17:57 (test/data/table_room_flatbot_20261023_1757
    .npz): with the scene camera's points moved by its error at the box and
    the wrist's frame from staging, the drops are on the open table beyond
    the box, on the table the box stands on — and the 17:57 spot, on the
    microwave's front edge, is none of them."""
    import math
    from types import SimpleNamespace

    import numpy as np

    from box_opening.models.container import ContainerModel
    from box_opening.tasks import press_demo

    f = np.load("capabilities/box_opening/test/data/table_room_flatbot_20261023_1757.npz")
    pink = ContainerModel.load("capabilities/box_opening/config/containers/ankou_pink.yaml")
    table_z = float(f["table_z"])
    scene_fix = SimpleNamespace(
        table_points=f["scene_mm"] / 1000.0,
        table_z_scene=table_z,
        pose=ContainerPose(xyz=(*f["scene_box"], table_z), yaw=0.0),
    )
    cpose = ContainerPose(xyz=(*f["wrist_box"], table_z), yaw=0.0)
    monkeypatch.setattr(press_demo, "wrist_points", lambda watcher: f["wrist_mm"] / 1000.0)
    tmap = press_demo.room_map(None, scene_fix, cpose, at_staging=True)
    drops, spots, why = press_demo.clear_lid_drops(pink, cpose, tmap, table_z)
    assert drops and why is None
    assert all(d.xyz[2] == table_z for d in drops)
    assert [d.yaw for d in drops] == [s.turn for s in spots]
    assert all(math.hypot(d.xyz[0] - 0.56, d.xyz[1] + 0.37) > 0.15 for d in drops)


def test_the_hands_turn_to_a_spot_counts_the_bearing_it_gripped_the_knob_at():
    """The reach-out grips the knob with the press attitude turned to the
    button's bearing from the arm (reach_press.reach_quat), and sets the
    lid down at the press attitude fixed in the world, turned
    (place_quat): from a box 60 deg round to the right the hand turns 60
    deg to come down unturned, and 30 deg to come down a quarter turned
    back. With room either way the quarter turn goes first, and each spot
    says how far the hand turns to it. Given where the hand faces (the
    arm's FK, reach_lid_off.spot_candidates), that is what counts."""
    import numpy as np

    from box_opening.geometry import ang_diff
    from box_opening.models.container import ContainerModel
    from box_opening.perception import clear_table
    from box_opening.tasks import press_demo

    pink = ContainerModel.load("capabilities/box_opening/config/containers/ankou_pink.yaml")
    table_z = 0.2163
    box = (0.30, -0.52)
    g = np.arange(-0.6, 0.6001, 0.006)
    X, Y = np.meshgrid(g + box[0], g + box[1], indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, table_z)], -1)
    pts[np.hypot(pts[:, 0] - box[0], pts[:, 1] - box[1]) < 0.0525, 2] += 0.121  # the box
    tmap = clear_table.table_map([(pts, table_z)], box)
    cpose = ContainerPose(xyz=(box[0], box[1], table_z), yaw=0.0)
    pressing = clear_table.hand_yaw(pink.press_quat(cpose.xyz))  # fixed in the world
    bearing = math.atan2(box[1], box[0])  # the button is over the box's centre
    assert math.degrees(bearing) == pytest.approx(-60.0, abs=0.1)
    drops, spots, why = press_demo.clear_lid_drops(pink, cpose, tmap, table_z)
    assert drops and why is None
    for d, s in zip(drops, spots, strict=False):
        down = clear_table.hand_yaw(press_demo.place_quat(pink, d))
        assert abs(ang_diff(s.rotation, down - (pressing + bearing))) < 1e-9
    assert drops[0].yaw == pytest.approx(-math.pi / 2)
    assert math.degrees(spots[0].rotation) == pytest.approx(-30.0, abs=0.1)
    # the hand facing as the press attitude does: no turn is the least
    drops, spots, why = press_demo.clear_lid_drops(pink, cpose, tmap, table_z, hand_at=pressing)
    assert drops[0].yaw == 0.0 and spots[0].rotation == pytest.approx(0.0, abs=1e-9)


def test_shipped_config_speeds_are_guard_safe():
    """The shipped values are what actually runs at the bench."""
    import pytest

    from box_opening.models.container import load_press_demo

    cfg = load_press_demo(CFG)
    assert cfg.lift_speed == pytest.approx(0.35)
    assert cfg.setdown_speed == pytest.approx(0.35)
    # 0.15 -> 0.35 (owner, 2026-09-16): the set-down has run guarded at 0.35
    # since 2026-09-02; the grip descent gets the same
    assert cfg.grip_speed == pytest.approx(0.35)
    assert cfg.detect_period_s == pytest.approx(0.05)
    # every contact/carry speed stays inside the guard-limited band
    for v in (cfg.grip_speed, cfg.setdown_speed, cfg.lift_speed):
        assert 0.0 < v <= 0.5


def test_park_tool_down_rests_at_the_look_pose(ctx):
    """open_box.park_tool_down: the mission ends tool-down instead of at the
    factory HOME, saving the 2.4-2.9 rad wrist flip twice per run. That rest
    is now the LOOK pose — HOME's own wrist turned down — not the surveyed
    joint vector over a fixed bench point it used to be, so parking costs the
    search nothing and survives the arm being moved."""
    import math

    from box_opening.constants import HOME, REST_TOL_RAD
    from box_opening.primitives.look import look_joints
    from box_opening.tasks import press_demo

    cfg = _demo_cfg()
    assert cfg.park_tool_down is False  # the run ends at HOME (owner 2026-09-04)
    assert press_demo.rest_joints(cfg) == list(HOME)
    on = replace(cfg, park_tool_down=True)
    park = look_joints(HOME)
    assert press_demo.rest_joints(on) == park  # the switch still works
    # parked there, the search's first step is already done
    assert press_demo.search_targets(on)[0][1] == park
    # "already parked" is judged against the server's own start gate
    assert press_demo.rest_distance(park, park) == 0.0
    nudged = list(park)
    nudged[3] += REST_TOL_RAD * 2
    assert press_demo.rest_distance(nudged, park) > REST_TOL_RAD
    assert press_demo.rest_distance([math.pi] + park[1:], park) > 1.0


def test_home_arm_falls_back_to_the_bare_table_when_the_band_refuses_the_start(ctx):
    """A recovery home usually starts INSIDE the unseen-container band (the
    arm is holding at a press or a grip), which made the guarded bench
    world refuse every post-contact home; it now falls back to the bare
    table with a caution (review 2026-09-02)."""
    from box_opening.tasks import home_arm

    class RefusesOnce(FakeClient):
        def __init__(self):
            super().__init__()
            self.n = 0

        def plan_to_joints(self, q7, start_joints):
            self.n += 1
            if self.n == 1:

                class Bad:
                    success = False
                    message = "INVALID_START_STATE_WORLD_COLLISION"

                return Bad()
            return super().plan_to_joints(q7, start_joints)

    c = ctx
    c.client = RefusesOnce()
    legs = home_arm.build_legs(c)
    assert legs[0].name == "home" and legs[0].world == "bench_bare"
    kinds = [kw.get("model") is None for kind, kw in c.worlds.pushes if kind == "bench"]
    assert kinds == [False, True]  # guarded band first, then the bare table


class _Readiness:
    """Client double for the pre-actuation readiness check."""

    def __init__(self, planner=(True,), driver=True, tf=True, others=()):
        self.planner = list(planner)
        self.driver = driver
        self.tf = tf
        self.others = list(others)
        self.planner_waits = []
        self._base_frame = "arm_mount"

    def tf_ready(self, target, source, timeout_s=2.0):
        return self.tf

    def other_arm_clients(self):
        return self.others

    def planner_reachable(self, timeout_s=5.0):
        self.planner_waits.append(timeout_s)
        return self.planner.pop(0) if len(self.planner) > 1 else self.planner[0]

    def driver_reachable(self, timeout_s=5.0):
        return self.driver


def test_a_run_refuses_before_any_motion_when_the_planner_is_not_up():
    """Field 2026-09-16: the stack had just been restarted, the gripper
    closed, and THEN the run died on `set_world service unavailable` — cuRobo
    takes 17-21 s to load after sheppy starts it, and set_world waits 5 s.
    Readiness is checked before anything is actuated, and says what to do."""
    from box_opening.tasks import press_demo

    why = press_demo.readiness_refusal(_Readiness(planner=(False,)), execute=True)
    assert why is not None and "planner" in why and "nothing moved" in why.lower()


def test_a_planner_still_loading_is_waited_for_not_failed():
    """A restart race is a pause, not a failure: the first probe is short, and
    a planner that comes up within the load window lets the run proceed."""
    from box_opening.tasks import press_demo

    client = _Readiness(planner=(False, True))
    assert press_demo.readiness_refusal(client, execute=True) is None
    assert client.planner_waits[0] <= 1.0
    assert client.planner_waits[1] >= 20.0  # longer than cuRobo's measured load


def test_the_driver_is_required_only_when_executing():
    from box_opening.tasks import press_demo

    down = _Readiness(driver=False)
    why = press_demo.readiness_refusal(down, execute=True)
    assert why is not None and "driver" in why
    assert press_demo.readiness_refusal(_Readiness(driver=False), execute=False) is None


def test_a_run_refuses_before_any_motion_without_the_arms_tf_or_beside_adl():
    """Bench 2026-09-25: box_opening had stopped (no /tf at all) and the run
    flew to staging before it could not place a single wrist frame; and the
    ADL runtime, which moves this arm on its own, was running. Both now stop
    the run before anything moves, and name the one command that fixes both."""
    from box_opening.tasks import press_demo

    why = press_demo.readiness_refusal(_Readiness(tf=False), execute=True)
    assert why is not None and "TF" in why and "sheppy up box-opening" in why and "Nothing moved" in why
    why = press_demo.readiness_refusal(_Readiness(others=["rammp_adl_runtime"]), execute=True)
    assert why is not None and "rammp_adl_runtime" in why and "sheppy up box-opening" in why
    assert press_demo.readiness_refusal(_Readiness(), execute=True) is None
    assert press_demo.readiness_refusal(_Readiness(tf=False, others=["rammp_adl_runtime"]), execute=False) is None


def test_the_tf_gate_asks_for_the_worlds_wrist_camera_frame():
    """Flatbot 2026-10-22: the gate looked up the D405 default's
    end_effector_link, which flatbot (arm_-prefixed, OAK wrist) never
    publishes, and refused a healthy stack."""
    from box_opening.tasks import press_demo

    asked = []
    client = _Readiness()
    client.tf_ready = lambda target, source, timeout_s=2.0: asked.append(source) or True
    assert press_demo.readiness_refusal(client, execute=True, camera_cfg_name="camera_oak_wrist.yaml") is None
    assert asked == ["wrist_camera_color_optical_frame"]
    asked.clear()
    press_demo.readiness_refusal(client, execute=True)
    assert asked == ["end_effector_link"]  # the D405 default, unchanged


def _pairs_file(tmp_path, offsets_mm, t0=3000.0, spots=((0.325, -0.291), (0.312, -0.158), (0.406, -0.219))):
    """A calibration file (in force from t=2000) and a residuals file with
    one pair per spot: the scene camera's box `offsets_mm` from the wrist's."""
    import json
    import os

    calib = tmp_path / "camera_scene.yaml"
    calib.write_text("xyz: [-0.0767, 0.6836, 0.3883]\nquat_xyzw: [0.026522, 0.043627, -0.28454, 0.957304]\n")
    os.utime(calib, (2000.0, 2000.0))
    f = tmp_path / "residuals.jsonl"
    rows = []
    for i, ((x, y), (dx, dy)) in enumerate(zip(spots, offsets_mm, strict=False)):
        rows.append({"t": t0 + i, "scene_top": [x + dx / 1000.0, y + dy / 1000.0, 0.062], "wrist_top": [x, y, 0.095]})
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return f, calib


def test_the_scene_camera_is_trusted_on_what_it_last_measured(tmp_path):
    """Bench 2026-09-24: the cabinet-door tag refused the scene camera at
    6 px (a 3 s wrist search) while the camera had not moved — and at 5 px
    it would have flown to a box 12 cm off. What decides now is the last
    pair recorded under the calibration in force."""
    from box_opening.tasks import press_demo

    f, calib = _pairs_file(tmp_path, [(-126, -28), (-3, 2)])  # 12 cm off, then corrected
    assert press_demo.scene_trust(f, calib) == (True, None)
    f, calib = _pairs_file(tmp_path, [(-3, 2), (-118, -31)])
    trusted, why = press_demo.scene_trust(f, calib)
    assert trusted is False and "122 mm" in why
    f, calib = _pairs_file(tmp_path, [(-118, -31)], t0=1000.0)  # made under an earlier calibration
    assert press_demo.scene_trust(f, calib) == (True, None)
    assert press_demo.scene_trust(tmp_path / "none.jsonl", calib) == (True, None)


def test_the_touch_height_says_where_the_pads_landed():
    """The verdicts, at the heights the bench has produced. Pads on the
    CLOSED button: -3.7 .. +6.2 mm (a 5-12 mm "rim" band once refused the
    good +6.2 contact — never again). Pads on the knob ALREADY UP: +9.6
    (09-17), +10.8 (09-21 13:35) and +17.7 (09-17). The first two were read
    as "pads on a ring" for four days and pushed: that push is what shuts an
    open box. This knob stands about 10 mm proud with the lid on."""
    from box_opening.tasks.press_demo import touch_verdict

    for off in (-0.0037, -0.0008, 0.0023, 0.0062, 0.0075):
        assert touch_verdict(off) == "press", off
    for off in (0.0096, 0.0108, 0.0177):
        assert touch_verdict(off) == "popped", off
    assert touch_verdict(0.060) == "foreign"


def test_the_camera_says_the_knob_is_up_at_the_heights_this_knob_reaches():
    """Bench 2026-09-21 14:16, the owner: "it pressed the button and opened
    it but then stopped and went down again and CLOSED the box". The pop
    check had read [11.1, 10.1, 11.1] mm on three fresh frames — the knob
    was up — and called it NOT POPPED, because "popped" began at 12 mm (a
    number from one +17.7 mm touch). Every reading of this knob up is
    9-11 mm (9.2, 10.1, 11.1; +10 against its own lid in an aim frame);
    every reading of it down is within a millimetre of zero (-0.8, 0.2)."""
    from box_opening.tasks.press_demo import knob_is_up

    for up_mm in (9.2, 10.1, 11.1, 16.0):
        assert knob_is_up(up_mm), up_mm
    for up_mm in (-0.8, 0.2, 1.0, 4.0, None):
        assert not knob_is_up(up_mm), up_mm


def test_a_stop_above_the_box_still_gets_home(ctx, tmp_path, capsys):
    """Bench 2026-09-21 14:16: "home plan refused (... the start state
    collides with the world ...) — arm holds". A recovery home is planned in
    the blind bench world, whose whole placement band is blocked to
    container height — and the arm was at the hop, fingers 50 mm above the
    lid, INSIDE that band: refused by construction, every time a run stops
    over the box. The mission knows where this box is. It rises out of the
    corridor it came down (the re-press's own retreat, in the button's
    interaction world) and goes home from staging height."""
    import re

    import pytest
    from conftest import runner as make_runner

    from box_opening.tasks import press_demo

    c = ctx
    cfg = _demo_cfg()
    c.press_cfg = cfg

    class Refused:
        success = False
        message = "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION"
        trajectory = None

    c.client.plans = [Refused()]  # the first plan asked for — home from the hop — is refused
    assert press_demo.try_home(c, make_runner(c.client, tmp_path), True, "STOP: test") is True
    said = capsys.readouterr().out
    assert "arm holds" not in said and "rising over the box first" in said
    # the rise flows into the home: ONE execution, like the mission's own retreat + home
    assert len(c.client.executed) == 1
    assert re.search(r"retreat:rise .*interaction_button[^\n]*\nhome ", said)
    assert c.client.joint_targets[-1] == pytest.approx(press_demo.rest_joints(cfg))
    assert c.client.live == pytest.approx(press_demo.rest_joints(cfg))  # and it got there
    assert c.lid_at is None


def test_a_stop_with_no_box_located_holds_honestly(ctx, tmp_path, capsys):
    """... and when there is no located box to rise over, a refused home
    still leaves the arm holding with an honest line, as before."""
    from conftest import runner as make_runner

    from box_opening.tasks import press_demo

    c = ctx
    c.cpose = None

    class Refused:
        success = False
        message = "INVALID_START_STATE_WORLD_COLLISION"
        trajectory = None

    c.client.plans = [Refused()]
    assert press_demo.try_home(c, make_runner(c.client, tmp_path), True, "STOP: test") is False
    assert "arm holds" in capsys.readouterr().out and c.client.executed == []


def test_a_home_that_was_refused_or_did_not_arrive_says_so(ctx, tmp_path):
    """try_home's answer is whether the arm got home: the depth press exits
    on it (a popped knob with the arm held over the box is not a success)."""
    from conftest import runner as make_runner

    from box_opening.tasks import press_demo

    class Refused:
        success = False
        message = "INVALID_START_STATE_WORLD_COLLISION"
        trajectory = None

    ctx.press_cfg = _demo_cfg()
    assert press_demo.try_home(ctx, make_runner(ctx.client, tmp_path), True, "STOP: test") is True
    ctx.client.exec_script = [("failed", {"message": "driver aborted the goal", "progress": 0.2})]
    assert press_demo.try_home(ctx, make_runner(ctx.client, tmp_path), True, "STOP: test") is False
    ctx.client.plans = [Refused(), Refused()]  # from here, and the rise over the box
    assert press_demo.try_home(ctx, make_runner(ctx.client, tmp_path), True, "STOP: test") is False


def test_a_stop_with_a_joint_on_its_limit_steps_off_it_and_gets_home(ctx, tmp_path, capsys):
    """Flatbot 2026-10-23: a staging at the reach limit left arm_joint_6
    0.100 rad from its limit. The planner refuses a start inside its margin,
    so the recovery home AND the rise over the box were refused and the arm
    held. It now steps plan-free along the joint line toward the rest pose
    first, as home_arm does, and plans home from where the step ends."""
    import pytest
    from conftest import runner as make_runner

    from box_opening.constants import PLANNER_LIMIT_MARGIN_RAD
    from box_opening.primitives.core import CLEAR_MARGIN_FIRST_RAD
    from box_opening.runtime import approach
    from box_opening.tasks import press_demo

    c = ctx
    cfg = _demo_cfg()
    c.press_cfg = cfg
    rest = press_demo.rest_joints(cfg)
    lo, hi = approach._chain().limits[5]
    start = list(rest)
    start[5] = lo + 0.0999  # arm_joint_6 a hair inside the planner's margin, as it was
    c.client.live = list(start)

    class InMargin:
        success = False
        message = "START_IN_LIMIT_MARGIN: start: arm_joint_6 is 0.100 rad from its position limit"
        trajectory = None

    def planner(real):  # refuses any start inside its margin, as rammp-curobo does
        def plan(*a, **kw):
            q = kw.get("start_joints", a[-1])
            return InMargin() if min(q[5] - lo, hi - q[5]) < PLANNER_LIMIT_MARGIN_RAD else real(*a, **kw)

        return plan

    c.client.plan_to_joints = planner(c.client.plan_to_joints)
    c.client.plan_to_pose = planner(c.client.plan_to_pose)
    assert press_demo.try_home(c, make_runner(c.client, tmp_path), True, "STOP: test") is True
    said = capsys.readouterr().out
    assert "joint_6 is 0.100 rad from its limit" in said and "a plan-free step" in said
    assert "arm holds" not in said
    # nothing was planned from the refused start: home from the step's end
    end = c.client.joint_starts[0]
    assert min(end[5] - lo, hi - end[5]) == pytest.approx(CLEAR_MARGIN_FIRST_RAD, abs=1e-6)
    # the guarded plan-free step and the planned home fly apart, and it got home
    assert len(c.client.executed) == 2
    assert c.client.live == pytest.approx(rest)


def test_the_mission_record_keeps_the_table_the_scene_camera_measured(ctx, tmp_path):
    """Flatbot (no survey): the record was built before the scene measured
    the table, and every truth.json said table_z NaN."""
    import json
    from types import SimpleNamespace

    import numpy as np

    from box_opening.detection_set import MissionFrames
    from box_opening.tasks import press_demo

    worlds = SimpleNamespace(table_top_z=None, table_extent=None)
    watcher = SimpleNamespace(table_z=None)
    ctx.mission_frames = MissionFrames(tmp_path / "set", "ankou_pink.yaml", worlds.table_top_z, now=0)
    fix = SimpleNamespace(table_z_scene=0.2933, table_extent=(0.3, 0.9, -0.5, 0.2))
    press_demo.adopt_scene_table(ctx, worlds, watcher, fix)
    assert worlds.table_top_z == watcher.table_z == 0.2933 and worlds.table_extent == fix.table_extent
    ctx.mission_frames.frames.append(("scene_0", {"x": np.zeros(1)}))
    doc = json.loads((ctx.mission_frames.close(exit_code=1) / "truth.json").read_text())
    assert doc["table_z"] == pytest.approx(0.2933)
    # a surveyed table is the one kept; an unmeasured one changes nothing
    worlds2 = SimpleNamespace(table_top_z=0.05, table_extent=None)
    press_demo.adopt_scene_table(ctx, worlds2, watcher, fix)
    assert worlds2.table_top_z == 0.05
    worlds3 = SimpleNamespace(table_top_z=None, table_extent=None)
    press_demo.adopt_scene_table(ctx, worlds3, watcher, SimpleNamespace(table_z_scene=float("nan"), table_extent=None))
    assert worlds3.table_top_z is None


def test_an_open_box_is_not_pressed_shut():
    """What the mission says when the aim finds the button already up — and
    that it says nothing when the button is down or was not measured."""
    from box_opening.tasks.press_demo import refuse_open_box

    for up_mm in (8.0, 9.0, 10.3):  # the aim frames of 13:35, 14:14:04 and 14:14:42, all pressed shut
        why = refuse_open_box(up_mm)
        assert why is not None and "already" in why and "%.0f mm" % up_mm in why
    for up_mm in (-1.0, 0.0, 1.0, 4.0, None):
        assert refuse_open_box(up_mm) is None


def test_the_push_bound_leaves_room_for_the_buttons_travel():
    """The push ends at the button's stop (the force), and 10 mm is the
    room it has to get there — 4 mm from a 3 Nm contact was not enough on
    the 4.1-inch box (owner 2026-09-17)."""
    from box_opening.models.container import load_press_demo

    cfg = load_press_demo("capabilities/box_opening/config/containers/oxo_pop.yaml")
    assert cfg.button_travel_m == 0.010
    assert cfg.contact_nm == 3.0 and cfg.touch_nm if hasattr(cfg, "touch_nm") else True


STUCK = [0.031, 0.721, 3.125, -2.281, -0.001, 0.267, 1.568]  # refused as a start, 2026-09-17


def test_home_arm_can_lift_straight_up_before_planning(ctx):
    """The pose the planner refuses as a start (fingers 40 mm above the
    table after a jog) is lifted plan-free, straight up with the attitude
    held, and HOME is planned from where the lift ends."""
    import numpy as np
    import pytest

    from box_opening.runtime.kinematics import ArmChain
    from box_opening.tasks import home_arm

    c = ctx
    c.client.live = list(STUCK)
    legs = home_arm.build_legs(c, lift_first_m=0.08)
    assert [x.name for x in legs] == ["lift-free", "home"]
    lift = legs[0]
    assert lift.guard is not None and lift.guard.trip == "obstruction"
    chain = ArmChain()
    R0, t0 = chain.fk(STUCK)
    pts = [p.positions for p in lift.traj.points]
    assert np.allclose(pts[0], STUCK)
    for q in pts:
        R, t = chain.fk(q)
        assert np.hypot(t[0] - t0[0], t[1] - t0[1]) < 1e-4  # straight up
        assert np.allclose(R, R0, atol=2e-3)  # attitude held
    _R, t_end = chain.fk(pts[-1])
    assert t_end[2] - t0[2] == pytest.approx(0.08, abs=1e-4)
    assert legs[1].name == "home"
    # HOME was planned from the lift's END, not from the stuck pose
    assert c.client.joint_starts[-1] == pytest.approx(list(lift.goal_joints))


def test_home_arm_without_a_lift_is_unchanged(ctx):
    from box_opening.tasks import home_arm

    legs = home_arm.build_legs(ctx)
    assert [x.name for x in legs] == ["home"]


def test_residual_pairs_go_next_to_a_custom_calibration(tmp_path):
    """The e2e harness's synthetic pairs must never land in the real file
    (they did, ~90 of them, 2026-09-17)."""
    from box_opening.tasks import press_demo

    assert press_demo.residuals_path_for(None) == press_demo.RESIDUALS_FILE
    custom = tmp_path / "camera_scene.yaml"
    assert press_demo.residuals_path_for(str(custom)) == tmp_path / "residuals.jsonl"


def _hop_frame(ctx, knob_up_m):
    """A synthetic wrist frame from the hop: the camera looking straight
    down from 17.5 cm above the lid, 75 mm off the tool axis (the mount),
    the lid a plane and the button a 45 mm disc standing `knob_up_m` above
    it. Returns (grab, (rot, trans))."""
    from types import SimpleNamespace

    import numpy as np

    from box_opening.models.container import from_container

    button = np.array(from_container(ctx.cpose, ctx.model.button_offset), float)
    k = np.array([[434.9, 0.0, 424.0], [0.0, 434.3, 237.8], [0.0, 0.0, 1.0]])
    rot = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    trans = button + np.array([0.075, 0.0, 0.175])
    vv, uu = np.mgrid[0:480, 0:848].astype(float)
    depth = np.full((480, 848), 0.175, np.float32)
    # a pixel's ray meets the knob's top plane at range (0.175 - up); it is
    # knob where that point lies within the disc
    zc = 0.175 - knob_up_m
    pts = (
        np.stack([(uu - k[0, 2]) / k[0, 0] * zc, (vv - k[1, 2]) / k[1, 1] * zc, np.full_like(uu, zc)], -1) @ rot.T
        + trans
    )
    depth[np.hypot(pts[..., 0] - button[0], pts[..., 1] - button[1]) < 0.0225] = zc
    grab = SimpleNamespace(
        color=np.zeros((480, 848, 3), np.uint8),
        depth=depth,
        k=k,
        color_stamp=SimpleNamespace(sec=100, nanosec=0),
    )
    return grab, (rot, trans)


def _aim_frame_env(ctx, tmp_path, monkeypatch):
    """press_demo with captures in tmp_path, a hop frame and its pose, a
    sighting on it, and write_frame held until the returned gate is set.
    Returns (press_demo, grab, pose, sighting, gate)."""
    import threading
    from types import SimpleNamespace

    from box_opening.perception import depth_source
    from box_opening.tasks import press_demo

    monkeypatch.setattr(press_demo, "CAPTURES_DIR", tmp_path)
    grab, pose = _hop_frame(ctx, 0.0)
    monkeypatch.setattr(depth_source, "camera_pose_at", lambda g, offset_s=None: pose)
    gate, real = threading.Event(), press_demo.write_frame

    def held(folder, arrays, **kw):
        assert gate.wait(5.0)
        return real(folder, arrays, **kw)

    monkeypatch.setattr(press_demo, "write_frame", held)
    sighting = SimpleNamespace(uv=(424.0, 238.0), r_px=40.0, diameter_m=0.045)
    return press_demo, grab, pose, sighting, gate


def test_the_aim_frame_is_written_off_the_critical_path(ctx, tmp_path, monkeypatch):
    """run-20261022-200421: every aim waited ~45 ms for its evidence frame's
    zlib before the next leg could start. The aim returns its folder at
    once; the frame — the one the aim was taken on, not a newer one the
    grabber has since put in its place — lands a moment later."""
    import numpy as np

    press_demo, grab, pose, sighting, gate = _aim_frame_env(ctx, tmp_path, monkeypatch)
    color, depth = grab.color, grab.depth
    folder = press_demo.save_aim_frame(grab, sighting, "aim")
    try:
        assert folder is not None and folder.is_dir() and not (folder / "frame_000.npz").exists()
        # the grabber's next message: new arrays, as d405.py assigns them
        grab.color, grab.depth = np.full_like(color, 255), np.zeros_like(depth)
    finally:
        gate.set()
    press_demo.wait_capture_writes()
    saved = np.load(folder / "frame_000.npz")
    assert np.allclose(saved["trans_cam"], pose[1])
    assert np.array_equal(saved["color"], color) and np.array_equal(saved["depth"], depth)
    assert (folder / "frame_000.jpg").exists()


def test_the_aim_frame_prunes_old_captures_after_its_write(ctx, tmp_path, monkeypatch):
    import numpy as np

    press_demo, grab, pose, sighting, gate = _aim_frame_env(ctx, tmp_path, monkeypatch)
    for i in range(press_demo.AIM_CAPTURES_KEPT + 1):
        (tmp_path / ("wrist-aim-20000101-0000%02d" % i)).mkdir()
    gate.set()
    folder = press_demo.save_aim_frame(grab, sighting, "aim")
    press_demo.wait_capture_writes()
    kept = sorted(tmp_path.glob("wrist-aim-*"))
    assert len(kept) == press_demo.AIM_CAPTURES_KEPT and folder in kept
    assert np.load(folder / "frame_000.npz")["depth"].shape == (480, 848)


def test_the_mission_frames_wait_for_the_aim_frame_they_copy(ctx, tmp_path, monkeypatch):
    """MissionFrames.close copies the aim's folder in as "staging": a run
    that ends straight after the aim must not copy it half written."""
    import threading

    import numpy as np

    from box_opening.detection_set import MissionFrames

    press_demo, grab, pose, sighting, gate = _aim_frame_env(ctx, tmp_path / "captures", monkeypatch)
    folder = press_demo.save_aim_frame(grab, sighting, "aim")
    frames = MissionFrames(tmp_path / "missions", "ankou_pink.yaml", 0.05)
    frames.located([0.4, 0.0, 0.15], "aim", aim_capture=folder)
    timer = threading.Timer(0.2, gate.set)
    timer.start()
    try:
        press_demo.keep_mission_frames(frames, None)
    finally:
        gate.set()
        timer.cancel()
    staging = frames.dir / "staging"
    assert np.allclose(np.load(staging / "frame_000.npz")["trans_cam"], pose[1])
    assert (staging / "frame_000.jpg").exists()


def test_the_pink_canister_pushes_firmer_and_holds():
    """The owner, 2026-09-23: "press down harder so it can open the box".
    Its push stopped at the 4.5 Nm backstop after 4.9 mm and the box stayed
    shut. For this box: a 7 Nm backstop (touch_nm 10 - contact 3), 12 mm of
    room, and a 0.3 s hold on the button. The OXOs keep what opened them."""
    from box_opening.models.container import ContainerModel, load_press_demo

    pink = "capabilities/box_opening/config/containers/ankou_pink.yaml"
    m, cfg = ContainerModel.load(pink), load_press_demo(pink)
    assert m.touch_nm - cfg.contact_nm == pytest.approx(7.0)
    assert cfg.button_travel_m == pytest.approx(0.012) and cfg.push_hold_s == pytest.approx(0.3)
    oxo = load_press_demo("capabilities/box_opening/config/containers/oxo_pop.yaml")
    assert oxo.push_hold_s == 0.0 and oxo.button_travel_m == pytest.approx(0.010)


def test_the_run_starts_by_going_home_unless_already_there(ctx, tmp_path, capsys):
    """Owner, 2026-09-25: the first thing a run does is send the arm to its
    exact rest pose, if it is not already there — one joint goal to HOME,
    planned in the bench world like home_arm's."""
    from conftest import runner as make_runner

    from box_opening.constants import HOME
    from box_opening.tasks import press_demo

    ctx.client.live = list(HOME)
    ctx.client.live[2] = HOME[2] - 2 * math.pi + 0.004  # home as the driver reports it (-3.1412): already there
    assert press_demo.home_first(ctx, make_runner(ctx.client, tmp_path), True, list(HOME))
    assert ctx.client.executed == [] and ctx.client.joint_targets == []

    ctx.client.live = [v + 0.3 for v in HOME]  # anywhere else
    assert press_demo.home_first(ctx, make_runner(ctx.client, tmp_path), True, list(HOME))
    assert len(ctx.client.executed) == 1
    assert ctx.client.joint_targets[-1] == pytest.approx(list(HOME), abs=1e-9)
    assert "HOME FIRST" in capsys.readouterr().out


# the joint spring's hold after an impedance home: 0.03 rad off at worst, as
# flatbot 2026-10-27 16:24 printed it (the other joints' offsets are made up),
# joint_3 read through the driver's wrap
_HOLD = [0.0, 0.03, -0.006, -0.02, 0.0, 0.012, 0.0]


def _held(home):
    q = [h + d for h, d in zip(home, _HOLD, strict=False)]
    q[2] += 2 * math.pi  # -3.143 is past -pi: the driver reports joint_3 there as +3.140
    return q


def _rows(log_dir):
    import json

    return [json.loads(line) for f in sorted(log_dir.glob("run-*.jsonl")) for line in open(f)]


def test_a_compliant_run_starts_from_the_impedance_hold_it_finds(ctx, tmp_path, capsys):
    """Flatbot 2026-10-27 16:24: the run after the first compliant run found
    the arm 0.03 rad off HOME (the spring's hold) and flew a POSITION home
    before anything else: a 0.14 s plan, 0.55 s flown (0.25 s of it the
    switch out of impedance), the switch back still to come. Within
    REST_TOL_RAD (main's tolerance for a compliant depth press) it starts
    from there: nothing planned or flown, the offsets logged."""
    from conftest import runner as make_runner

    from box_opening.constants import HOME, REST_TOL_RAD
    from box_opening.tasks import press_demo

    ctx.client.live = _held(HOME)
    assert press_demo.home_first(ctx, make_runner(ctx.client, tmp_path), True, list(HOME), tol=REST_TOL_RAD)
    assert ctx.client.executed == [] and ctx.client.joint_targets == [] and ctx.worlds.pushes == []
    out = capsys.readouterr().out
    assert "HOME FIRST" not in out
    assert "0.030 rad from its rest pose — the joint spring's hold after an impedance run; starting from there" in out
    (note,) = _rows(tmp_path)
    assert note["leg"] == "start_offset" and note["kind"] == "note"
    assert note["rad"] == pytest.approx(0.03, abs=1e-4)
    assert note["per_joint"] == pytest.approx(_HOLD, abs=1e-4)  # wrap-aware, signed live - rest


def test_the_same_hold_is_flown_home_with_the_default_tolerance(ctx, tmp_path, capsys):
    """Every other run (--guarded, no --depth-press, --no-scene) keeps
    AT_REST_RAD: from the same 0.03 rad, HOME FIRST flies, as before."""
    from conftest import runner as make_runner

    from box_opening.constants import HOME
    from box_opening.tasks import press_demo

    ctx.client.live = _held(HOME)
    assert press_demo.home_first(ctx, make_runner(ctx.client, tmp_path), True, list(HOME))
    assert len(ctx.client.executed) == 1 and len(ctx.client.joint_targets) == 1
    assert ctx.client.joint_targets[-1] == pytest.approx(list(HOME), abs=1e-9)
    assert "HOME FIRST: the arm is 0.03 rad from its rest pose" in capsys.readouterr().out
    assert [r["leg"] for r in _rows(tmp_path)] == ["home"]  # no start_offset note


def test_a_start_beyond_the_compliant_tolerance_goes_home_first_either_way(ctx, tmp_path, capsys):
    """0.06 rad off is past REST_TOL_RAD too: HOME FIRST flies whichever
    tolerance the run passed."""
    from conftest import runner as make_runner

    from box_opening.constants import HOME, REST_TOL_RAD
    from box_opening.tasks import press_demo

    for i, tol in enumerate((press_demo.AT_REST_RAD, REST_TOL_RAD)):
        ctx.client.live = list(HOME)
        ctx.client.live[1] += 0.06
        log_dir = tmp_path / str(i)
        assert press_demo.home_first(ctx, make_runner(ctx.client, log_dir), True, list(HOME), tol=tol)
        assert len(ctx.client.executed) == i + 1, tol
        assert ctx.client.joint_targets[-1] == pytest.approx(list(HOME), abs=1e-9)
        out = capsys.readouterr().out
        assert "HOME FIRST: the arm is 0.06 rad" in out and "spring's hold" not in out, tol
        assert [r["leg"] for r in _rows(log_dir)] == ["home"], tol


def test_the_start_line_tells_a_stiff_hold_from_the_rest_pose(ctx, tmp_path, capsys):
    """STIFF doubles MEDIUM's Kq, so its hold should sit about 0.015 rad off
    HOME: past AT_REST_RAD, and printed so — two decimals made it "0.01",
    AT_REST_RAD's own value, on a start that is not at rest."""
    from conftest import runner as make_runner

    from box_opening.constants import HOME, REST_TOL_RAD
    from box_opening.tasks import press_demo

    ctx.client.live = list(HOME)
    ctx.client.live[1] += 0.015
    assert press_demo.home_first(ctx, make_runner(ctx.client, tmp_path), True, list(HOME), tol=REST_TOL_RAD)
    assert ctx.client.executed == []
    assert "the arm rests 0.015 rad from its rest pose" in capsys.readouterr().out


def test_a_home_the_bench_world_refuses_moves_nothing_and_names_the_recovery(ctx, tmp_path, capsys):
    """An arm down at the table is home_arm's case (bare-world and
    lift-first fallbacks, the operator's eye on the bench): the run does not
    start, and says so."""
    from conftest import runner as make_runner

    from box_opening.constants import HOME
    from box_opening.tasks import press_demo

    class Refused:
        success = False
        message = "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION"
        trajectory = None

    ctx.client.live = [v + 0.3 for v in HOME]
    ctx.client.plans = [Refused()]
    assert not press_demo.home_first(ctx, make_runner(ctx.client, tmp_path), True, list(HOME))
    assert ctx.client.executed == []
    assert "home_arm --execute" in capsys.readouterr().out


def test_detect_only_reports_the_scene_cameras_fixes_with_nothing_moving(monkeypatch, capsys):
    """--detect-only is the scene camera's answer, repeated: the wrist does
    not find the box (owner, 2026-10-22), and nothing is flown to ask."""
    from types import SimpleNamespace

    from box_opening.tasks import press_demo

    clock = [0.0]
    monkeypatch.setattr(press_demo.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(press_demo.time, "sleep", lambda s: None)

    class Locator:
        last_why = None
        calls = 0

        def locate(self, timeout_s):
            clock[0] += 2.0  # a locate takes OWL ticks
            self.calls += 1
            if self.calls == 2:
                self.last_why = "the scene OWL saw no box in 5.0 s"
                return None
            top = (0.42, 0.22 + 0.001 * self.calls, 0.21)
            return SimpleNamespace(
                pose=SimpleNamespace(xyz=(0.42, 0.22, 0.09), yaw=0.1),
                top_xyz=top,
                n_points=120,
                score=0.44,
                table_z_scene=0.089,
                table_extent=(0.3, 0.9, -0.2, 0.6),
            )

    assert press_demo.scene_detect_report(Locator(), duration_s=15.0) == 7
    out = capsys.readouterr().out
    assert "nothing moves" in out and "no fix: the scene OWL saw no box" in out
    assert "7 fix(es), spread [0, 7, 0] mm" in out

    class Blind:
        last_why = "scene camera streams missing: depth"

        def locate(self, timeout_s):
            clock[0] += 5.0
            return None

    assert press_demo.scene_detect_report(Blind(), duration_s=15.0) == 0
    out = capsys.readouterr().out
    assert out.count("no fix: scene camera streams missing") == 1  # said once, not every try
    assert "NO BOX — scene camera streams missing: depth" in out


def test_the_scene_line_names_the_detector_and_the_sightings_behind_the_fix():
    from types import SimpleNamespace

    from box_opening.tasks import press_demo

    seg = SimpleNamespace(score=0.881, source="segmenter", n_sightings=2, sources=("segmenter", "segmenter"))
    assert press_demo.scene_source_label(seg) == "SEG 0.88, 2 sightings (segmenter+segmenter)"
    mixed = SimpleNamespace(score=0.31, source="owl", n_sightings=2, sources=("segmenter", "owl"))
    assert press_demo.scene_source_label(mixed) == "OWL 0.31, 2 sightings (segmenter+owl)"
    assert press_demo.scene_source_label(SimpleNamespace(score=0.44)) == "OWL 0.44, 1 sighting (owl)"  # an older fix


def test_the_lid_is_set_down_on_the_table_the_camera_measured():
    """Flatbot: no surveyed bench, so no fixed set-down zone either — the
    lid goes beside the box on the table measured this run, whole."""
    from types import SimpleNamespace

    from box_opening.models.container import ContainerModel
    from box_opening.tasks import press_demo

    m = ContainerModel.load(CFG)
    measured = SimpleNamespace(table_extent=(0.45, 1.10, -0.90, -0.10))
    x0, x1, y0, y1 = press_demo.drop_zone(m, measured)
    inset = 0.5 * max(m.lid_dims[:2]) + press_demo.DROP_EDGE_M
    assert (x0, x1, y0, y1) == pytest.approx((0.45 + inset, 1.10 - inset, -0.90 + inset, -0.10 - inset))
    box = ContainerPose(xyz=(0.70, -0.55, 0.30), yaw=0.0)
    spots = press_demo.lid_drop_candidates(m, box, 0.30, zone=(x0, x1, y0, y1))
    assert spots and all(x0 <= s[0] <= x1 and y0 <= s[1] <= y1 for s, _d in spots)
    assert [d for _s, d in spots] != [
        d for _s, d in press_demo.lid_drop_candidates(m, box, 0.30)
    ]  # not the bench's zone
    assert press_demo.drop_zone(m, SimpleNamespace(table_extent=None)) == (*press_demo.DROP_X_M, *press_demo.DROP_Y_M)
    # too few table points for an extent: the ring square assumed around the
    # pink canister still leaves the lid somewhere to go
    import numpy as np

    from box_opening.perception.scene_source import seen_table_extent

    pink = ContainerModel.load("capabilities/box_opening/config/containers/ankou_pink.yaml")
    ring, assumed = seen_table_extent(np.zeros((0, 3)), 0.30, box.xyz[:2])
    assert assumed and press_demo.lid_drop_candidates(
        pink, box, 0.30, zone=press_demo.drop_zone(pink, SimpleNamespace(table_extent=ring))
    )


def test_the_staging_aim_builds_its_aimer_the_straight_down_way_when_no_reach_out_priors_are_given(monkeypatch):
    """Review 2026-10-22: aim_button_at_staging always passed lid_z/prior_xy,
    and a ButtonAimer without them raised TypeError at every straight-down
    aim — with the arm already at staging. Unset, they are not passed."""
    from types import SimpleNamespace

    from box_opening.perception import button_aim
    from box_opening.tasks import press_demo

    made = []

    class Aimer:
        last_why = None

        def __init__(self, grab, model, table_z, **kw):
            made.append(kw)

        def status(self):
            return "still frames"

    monkeypatch.setattr(button_aim, "ButtonAimer", Aimer)
    w = SimpleNamespace(grab=SimpleNamespace(color=None, depth=None, k=None, color_stamp=None), model=None, table_z=0.1)
    got, status = press_demo.aim_button_at_staging(None, w, 0.0, timeout_s=0)
    assert got is None and "still frames" in status and made == [{}]
    press_demo.aim_button_at_staging(None, w, 0.0, timeout_s=0, lid_z=0.2, prior_xy=(0.8, 0.0))
    assert made[-1] == {"lid_z": 0.2, "prior_xy": (0.8, 0.0)}
    press_demo.aim_button_at_staging(None, w, 0.0, timeout_s=0, lid_z=0.2, prior_xy=(0.8, 0.0), min_hits=2)
    assert made[-1] == {"lid_z": 0.2, "prior_xy": (0.8, 0.0), "min_hits": 2}


def test_the_aim_judges_frames_back_to_back_and_waits_only_when_none_is_new(monkeypatch, capsys):
    """Flatbot 2026-10-27: a 20 ms sleep after every tick, the 45-75 ms ones
    that judged a tilted frame too, held the reach-out aims to 10.1-14.0 of
    the wrist OAK's 15 frames a second. A tick that took a frame, a hit or
    not, is followed by the next at once; one that took none sleeps —
    nothing new, or a colour frame whose depth has not landed, which the
    next tick tries again. The fix comes back as it did, and the AIM line
    and the note say which hits agreed and when each came."""
    from types import SimpleNamespace

    from box_opening.perception import button_aim
    from box_opening.tasks import press_demo

    events, made = [], []

    class Aimer:
        """Ticks as scripted (each takes a frame or not); the last fixes, on
        its first and third hits."""

        last_why = None

        def __init__(self, grab, model, table_z, **kw):
            made.append(kw)
            self.frames = self.hits = 0
            self.hit_s, self.above_lid, self.pair_dt_ms = [], [], []
            self.script = [
                "took frame 1",
                "nothing new",
                "frame 2's colour without its depth",
                "took frame 2",
                "took frame 3, the camera moving",
                "took frame 4",
            ]
            self.last_sighting = SimpleNamespace(
                uv=(320.0, 200.0), r_px=30.0, diameter_m=0.0441, oblique=True, disc_fill=0.8
            )

        def tick(self):
            what = self.script.pop(0)  # IndexError: ticked on past the fix
            events.append(what)
            if what.startswith("took"):
                self.frames += 1
                if "moving" not in what:
                    self.hits += 1
                    self.hit_s.append(0.07 * self.frames)
                    self.above_lid.append({1: 0.0012, 2: float("nan"), 3: 0.0014}[self.hits])
            return not self.script

        def fix(self, measured=False):
            return (0.7213, -0.297, 0.2914) if measured else (0.7213, -0.297, 0.3095)

        def fix_hits(self):
            return [0, 2]

        def button_above_lid_m(self):
            return 0.0013

        def table_offset_m(self, raw=False):
            return float("nan")

        def status(self):
            return "2 of 3 hits agree in 4 frames"

    monkeypatch.setattr(button_aim, "ButtonAimer", Aimer)
    monkeypatch.setattr(press_demo, "_spin_detect", lambda node: None)
    monkeypatch.setattr(press_demo.time, "sleep", lambda s: events.append("sleep"))
    notes = []
    runner = SimpleNamespace(note=lambda leg, **kw: notes.append((leg, kw)))
    grab = SimpleNamespace(color=None, depth=None, k=None, color_stamp=None)  # no frame for the capture
    w = SimpleNamespace(grab=grab, model=None, table_z=0.2139)
    out = {}
    got, status = press_demo.aim_button_at_staging(
        None,
        w,
        0.6,
        timeout_s=5.0,
        runner=runner,
        tag="reach-aim",
        out=out,
        lid_z=0.3125,
        prior_xy=(0.7248, -0.286),
        min_hits=2,
    )
    assert events == [
        "took frame 1",
        "nothing new",
        "sleep",
        "frame 2's colour without its depth",
        "sleep",
        "took frame 2",
        "took frame 3, the camera moving",
        "took frame 4",
    ]
    assert made == [{"lid_z": 0.3125, "prior_xy": (0.7248, -0.286), "min_hits": 2}]
    assert got == ((0.7213, -0.297, 0.3095), 0.6) and status == "2 of 3 hits agree in 4 frames"
    assert w.last_aim == got[0]
    assert out == {"button_up_mm": pytest.approx(1.3), "measured_xyz": (0.7213, -0.297, 0.2914)}
    assert "pressing its centre; 2 of 3 hits agree (4 frames, " in capsys.readouterr().out
    ((leg, kw),) = notes
    assert leg == "reach-aim" and (kw["hits"], kw["frames"]) == (3, 4)
    assert kw["agreeing"] == [0, 2] and kw["hit_s"] == [0.07, 0.14, 0.28] and kw["above_lid_mm"] == [1.2, None, 1.4]


@pytest.mark.parametrize(
    "off,off_raw,z,words,off_mm,raw_mm",
    [
        (
            -0.0216,
            -0.0216,
            0.3101,
            "the wrist reads its table -22 mm from the scene's, the button's top raised 22 mm",
            -21.6,
            -21.6,
        ),
        (
            float("nan"),
            -0.0452,
            0.2885,
            "the wrist reads its table -45 mm from the scene's (not taken out)",
            None,
            -45.2,
        ),
    ],
)
def test_the_aim_says_and_keeps_the_wrists_table_offset(monkeypatch, capsys, off, off_raw, z, words, off_mm, raw_mm):
    """Flatbot 2026-10-27 15:19: the wrist read its table 21.6 mm below the
    scene camera's, and its aim 0.2885 where the touch met the lid at 0.3122.
    The AIM line says what the wrist read, the press point's z is raised by
    an offset taken out, and the note keeps the z measured and the offset
    as read even when it was not taken out: drift shows before it fails."""
    from types import SimpleNamespace

    import numpy as np

    from box_opening.perception import button_aim, depth_source
    from box_opening.tasks import press_demo

    frames = [0]

    def snapshot():  # a new still frame each read
        frames[0] += 1
        st = SimpleNamespace(sec=frames[0], nanosec=0)
        return SimpleNamespace(
            color=np.zeros((4, 4, 3), np.uint8),
            color_stamp=st,
            depth=np.ones((4, 4), np.float32),
            depth_stamp=st,
            k=np.eye(3),
            dist=None,
        )

    rot, cam = np.diag([1.0, -1.0, -1.0]), np.array([0.536, -0.144, 0.574])
    monkeypatch.setattr(depth_source, "camera_pose_at", lambda g, offset_s=None, stamp=None: (rot, cam))
    seen = button_aim.ButtonSighting(
        xyz=(0.661, -0.1969, 0.2885),
        uv=(277.0, 163.2),
        r_px=31.6,
        range_m=0.36,
        diameter_m=0.0501,
        above_lid_m=0.001,
        lid_z_m=0.2895,
        incidence_deg=25.4,
        oblique=True,
        disc_fill=0.72,
        table_offset_m=off,
        table_offset_raw_m=off_raw,
    )
    monkeypatch.setattr(button_aim, "find_button", lambda *a, **kw: (seen, None))
    monkeypatch.setattr(press_demo, "_spin_detect", lambda node: None)
    notes = []
    runner = SimpleNamespace(note=lambda leg, **kw: notes.append((leg, kw)))
    grab = SimpleNamespace(snapshot=snapshot, color=None, depth=None, k=None, color_stamp=None, stamp_offset_s=0.0)
    w = SimpleNamespace(
        grab=grab, model=SimpleNamespace(dims=(0.105, 0.105, 0.121), button_diameter_m=0.044), table_z=0.2143
    )
    out = {}
    got, _status = press_demo.aim_button_at_staging(
        None, w, 0.0, timeout_s=5.0, runner=runner, tag="reach-aim", out=out, lid_z=0.3111, prior_xy=(0.6754, -0.1812)
    )
    assert got is not None and got[0] == pytest.approx((0.661, -0.1969, z), abs=1e-4)
    assert w.last_aim == pytest.approx(got[0])  # the press point the press goes on (aim_source)
    # ... and the caller is told it as the wrist measured it (reach_press._aim)
    assert out["measured_xyz"] == pytest.approx((0.661, -0.1969, 0.2885), abs=1e-6)
    assert out["button_up_mm"] == pytest.approx(1.0)
    assert words in capsys.readouterr().out
    ((leg, kw),) = notes
    assert leg == "reach-aim" and kw["press_xyz"] == [0.661, -0.1969, z] and kw["measured_z"] == 0.2885
    assert (kw["table_offset_mm"], kw["table_offset_raw_mm"]) == (off_mm, raw_mm)
