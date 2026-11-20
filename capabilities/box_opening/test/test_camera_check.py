"""The scene camera checked against the wrist camera (perception/camera_check): what one pair measures, when a
pair is one, and what is turned by it for the rest of a run."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from box_opening.models.container import ContainerPose
from box_opening.perception import camera_check as cc
from box_opening.perception.scene_source import SceneFix, SceneLocator

# RAMMP 1.5, 2026-11-20 13:39-13:44: the scene lens as the team's gen_1_5 TF has it, and the five times both
# cameras had the one container (the runs' own SCENE and WRIST SURVEY lines), the chair at four places
LENS = (0.151329, -0.571954)
PAIRS = (
    ((0.912, 0.000), (0.988, -0.201)),
    ((0.680, -0.034), (0.770, -0.192)),
    ((0.681, -0.035), (0.770, -0.193)),
    ((0.540, -0.049), (0.635, -0.188)),
    ((0.559, -0.078), (0.643, -0.208)),
)
# the gen_1_5 mount of that day, T(arm_mount <- the scene camera's colour frame), as a run's saved frame held it
T_MOUNT = np.array(
    [
        [0.4190646993844396, -0.0393058726493597, 0.9071051902095488, 0.151329],
        [-0.9073110393937678, 0.01953227068193164, 0.4200061525694697, -0.571954],
        [-0.03422653245700108, -0.999036305034647, -0.027477367026150334, 0.391385],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _measures():
    return [cc.measure(LENS, (*s, 0.443), (*w, 0.405), t=1000.0 + i)[0] for i, (s, w) in enumerate(PAIRS)]


def test_the_five_pairs_of_2026_11_20_are_one_steady_turn_about_the_scene_lens():
    """155 to 215 mm apart on the table, by how far the container stood; 13 to 15 deg about the scene lens
    wherever it stood, and 28 to 36 mm nearer along the line. The height is said as it was read."""
    got = _measures()
    assert all(d is not None for d in got)
    assert [round(d.turn_deg, 1) for d in got] == [-13.0, -13.9, -13.9, -14.9, -14.0]
    assert [round(1000 * d.along_m) for d in got] == [-36, -28, -29, -34, -29]
    assert [round(1000 * d.apart_m) for d in got] == [215, 182, 181, 168, 155]
    assert all(d.dz_m == pytest.approx(-0.038) for d in got)
    assert "-13.9 deg about its lens and 28 mm nearer along it (182 mm apart at 0.75 m)" in got[1].words()
    assert got[1].words().endswith("the wrist's lid 38 mm lower")
    assert got[1].row() == {
        "t": 1001.0,
        "turn_deg": -13.94,
        "along_mm": -28.2,
        "dz_mm": -38.0,
        "range_m": 0.754,
        "apart_mm": 181.8,
    }


def test_each_pairs_measure_puts_the_other_four_within_a_few_centimetres_of_the_wrist():
    """The replay the fix was asked for: the scene camera's numbers turned by what ONE pair measured (and
    that container's own fix moved along the line) land where the wrist has the container at the other
    chair places too, where a shift fitted to one place would not."""
    got = _measures()
    worst = 0.0
    for i, d in enumerate(got):
        for j, (s, w) in enumerate(PAIRS):
            turned = cc.turn_xyz(LENS, d.turn_rad, (*s, 0.443), d.along_m)
            err = math.hypot(turned[0] - w[0], turned[1] - w[1])
            assert turned[2] == 0.443  # heights are never moved by it
            if i == j:
                assert err < 1e-9  # (the pair it was measured from: onto the wrist's by construction)
            worst = max(worst, err)
            # a shift alone, fitted at the same pair, is off by as much as the container's place has changed
            shifted = (s[0] + PAIRS[i][1][0] - PAIRS[i][0][0], s[1] + PAIRS[i][1][1] - PAIRS[i][0][1])
            assert err <= math.hypot(shifted[0] - w[0], shifted[1] - w[1]) + 0.012
    assert worst < 0.035, worst
    # with the turn alone (no along-line move: what every OTHER thing the scene camera sees gets), within 5 cm
    alone = max(
        math.hypot(*(np.array(cc.turn_xyz(LENS, d.turn_rad, (*s, 0.0))[:2]) - w)) for d in got for s, w in PAIRS
    )
    assert alone < 0.05, alone
    # and untouched they were 155-215 mm apart
    assert min(math.hypot(s[0] - w[0], s[1] - w[1]) for s, w in PAIRS) > 0.15


def test_a_pair_that_could_not_be_one_container_is_no_measure():
    d, why = cc.measure(LENS, (0.70, 0.30, 0.44), (0.70, -0.25, 0.40))  # 37 deg apart about the lens
    assert d is None and "more than 25 deg" in why and "not taken for one container seen twice" in why
    d, why = cc.measure(LENS, (0.60, -0.10, 0.44), (0.95, 0.25, 0.40))  # the same direction, 0.5 m further
    assert d is None and "apart along the scene camera's line, more than 200" in why
    d, why = cc.measure(LENS, (0.25, -0.50, 0.44), (0.30, -0.45, 0.40))  # at the lens: no direction
    assert d is None and "gives no direction" in why
    assert cc.measure(LENS, (float("nan"), 0.0, 0.4), (0.7, 0.0, 0.4))[0] is None
    # heights that are not known are not made up
    assert math.isnan(cc.measure(LENS, (0.68, -0.03), (0.77, -0.19))[0].dz_m)


def test_the_turn_is_about_the_lens_so_the_camera_stays_where_its_mount_has_it():
    D = cc.rotation(LENS, math.radians(-14.0))
    T = D @ T_MOUNT
    assert np.allclose(T[:3, 3], T_MOUNT[:3, 3]) and np.allclose(T[2, :3], T_MOUNT[2, :3])  # its place and its tilt
    was = math.degrees(math.atan2(T_MOUNT[1, 2], T_MOUNT[0, 2]))  # where its optical axis points on the table
    assert math.degrees(math.atan2(T[1, 2], T[0, 2])) == pytest.approx(was - 14.0)
    p = np.array([0.3, -0.1, 0.9, 1.0])  # a point in the camera's frame: lifted with it, it is turned about the lens
    assert np.allclose((T @ p)[:3], cc.turn_xyz(LENS, math.radians(-14.0), (T_MOUNT @ p)[:3]))


def _locator(mode=cc.RUN, T=T_MOUNT, only_one=None):
    """A scene locator as built bare, placed by TF at `T` (its link the colour frame), with a check of its own.
    `only_one`: what it answers where asked whether it holds one container of the kind (None: it cannot be asked,
    as a locator of this branch cannot)."""
    loc = SceneLocator.__new__(SceneLocator)
    loc.grab = SimpleNamespace(link_frame="scene", lookup=lambda parent, child: T, link_to_color=lambda: np.eye(4))
    loc.T_base_link, loc.base_frame = None, "arm_mount"
    loc.check = cc.CameraCheck(mode=mode, clock=lambda: 5000.0)
    if only_one is not None:
        loc.only_one = lambda: only_one
    return loc


def _fix(xy, z=0.443, table=0.343, source="segmenter", rough=False):
    pts = np.array([[xy[0] + 0.1, xy[1], table], [xy[0], xy[1] + 0.1, table]], np.float32)
    return SceneFix(
        pose=ContainerPose(xyz=(xy[0], xy[1], table), yaw=0.1),
        top_xyz=(xy[0], xy[1], z),
        n_points=190,
        score=0.8,
        table_z_scene=table,
        table_extent=(xy[0] - 0.2, xy[0] + 0.2, xy[1] - 0.3, xy[1] + 0.3),
        table_points=pts,
        source=source,
        rough=rough,
    )


class _Notes:
    def __init__(self):
        self.rows = []

    def note(self, kind, **kw):
        self.rows.append((kind, kw))


def _seen(loc, xy):
    """The scene camera's fix of a container its mount has at `xy`, as loc.locate() returns it NOW: lifted
    through the placement in force (the turn) and moved along its line (own_fix)."""
    return cc.own_fix(loc, _fix(cc.turn_xyz(LENS, loc.check.turn_rad, (*xy, 0.443))[:2]))


def test_a_wrist_find_taken_for_the_scene_cameras_container_turns_the_scene_camera_for_the_rest_of_the_run(capsys):
    """adopt(): measured, said, noted, warned of once, in force on the locator (every later number it lifts) and
    told to the run's context. The team's mount itself is what it was (raw)."""
    loc, notes, ctx = _locator(), _Notes(), SimpleNamespace()
    assert np.allclose(loc.base_to_color(), T_MOUNT)  # nothing in force: as its mount has it
    wrist = _fix(PAIRS[1][1], z=0.385, table=0.293, source="wrist")
    d = cc.adopt(ctx, loc, _fix(PAIRS[1][0]), wrist, notes, "r1", paired=cc.BY_TIE)
    assert d.turn_deg == pytest.approx(-13.94, abs=0.01) and loc.check.by == "run" and loc.check.measured is d
    assert ctx.cameras_in_force == cc.in_force(loc) == (d.turn_rad, d.along_m)
    T = loc.base_to_color()
    assert np.allclose(T, cc.rotation(LENS, loc.check.turn_rad) @ T_MOUNT) and np.allclose(T[:3, 3], T_MOUNT[:3, 3])
    assert np.allclose(loc.base_to_color(raw=True), T_MOUNT) and np.allclose(loc.base_to_link(raw=True), T_MOUNT)
    assert np.allclose(cc.raw_mount(loc), T_MOUNT) and np.allclose(
        loc.base_to_link(), T
    )  # (its link is its colour frame)
    # the same container again later in the run, as the turned locator returns it: no second warning
    assert (
        cc.adopt(ctx, loc, _seen(loc, PAIRS[4][0]), (*PAIRS[4][1], 0.38), notes, by="r1", paired=cc.BY_ONE) is not None
    )
    said = [m for m in capsys.readouterr().out.splitlines() if "CAMERAS" in m or "WARNING" in m]
    assert "the scene camera has the container at [0.680, -0.034] and the wrist camera at [0.770, -0.192]" in said[0]
    assert "(one container: the selection's tie took the wrist's find)" in said[0]
    assert "the wrist's lid 58 mm lower" in said[0]
    assert said[0].endswith("turned -13.9 deg about its lens for the rest of this run")
    assert "WARNING: the scene camera's mount looks 14 deg off (it puts the container 182 mm" in said[1]
    assert "tell the team" in said[1] and sum("WARNING" in m for m in said) == 1  # once a run
    assert "the scene camera has the container at [0.559, -0.078]" in said[2]  # as the MOUNT has it, not what is left
    kind, row = notes.rows[0]
    assert kind == "camera_check" and row["measured"] and row["turn_deg"] == -13.94 and row["by"] == "run"
    assert row["scene_xy"] == [0.68, -0.034] and row["wrist_xy"] == [0.77, -0.192] and row["run"] == "r1"
    assert row["paired"] == cc.BY_TIE and notes.rows[1][1]["paired"] == cc.BY_ONE
    # the run's record of a later scene look names the turn its numbers were lifted with
    loc._look, loc.last_timing, loc.instance_id = None, {}, None
    assert loc.look_record()["camera_check"] == {"mode": "run", "by": "run", "turn_deg": -13.96, "along_mm": -28.7}


def test_a_fix_returned_with_a_turn_in_force_is_measured_as_the_mount_itself_has_it():
    """The run's second look is lifted through the turned placement and moved along its line. Measured against the
    wrist again, what is in force afterwards is the scene camera's OWN disagreement, not what the first left."""
    loc = _locator()
    first = cc.adopt(None, loc, _fix(PAIRS[1][0]), (*PAIRS[1][1], 0.40), paired=cc.BY_TIE)
    second = cc.adopt(None, loc, _seen(loc, PAIRS[3][0]), (*PAIRS[3][1], 0.40), paired=cc.BY_TIE)
    assert first.turn_deg == pytest.approx(-13.94, abs=0.01) and second.turn_deg == pytest.approx(-14.94, abs=0.01)
    assert second.along_m == pytest.approx(-0.034, abs=0.001) and loc.check.turn_rad == second.turn_rad


def test_a_pair_is_measured_only_where_something_establishes_that_it_is_one_container(capsys):
    """The review of 2026-11-20: an unselected run takes any wrist find, and a lid 15 cm beside a healthy camera's
    box then measured as a mount 12 deg off: a wrong warning and a wrong turn. adopt measures only what its
    caller says is one container, and says why where it does not."""
    loc, notes = _locator(), _Notes()
    beside = (PAIRS[1][0][0] + 0.02, PAIRS[1][0][1] - 0.15, 0.40)  # another lid, 15 cm to the side
    # what it would read as: a mount some 10 deg off (the review's own case, -12.0), from a camera that is right
    assert cc.measure(LENS, (*PAIRS[1][0], 0.44), beside)[0].turn_deg == pytest.approx(-10.2, abs=0.3)
    assert cc.adopt(None, loc, _fix(PAIRS[1][0]), beside, notes, by="r") is None  # (the caller did not say)
    paired, why = cc.corroborated(loc, [])
    assert cc.adopt(None, loc, _fix(PAIRS[1][0]), beside, notes, by="r", paired=paired, why_not=why) is None
    out = capsys.readouterr().out
    assert "CAMERAS: the two cameras' places are not measured against each other: the caller does not say" in out
    assert "not measured against each other: no selection ties the wrist's find to the scene camera's box, and" in out
    assert "WARNING" not in out and loc.check.by == "" and np.allclose(loc.base_to_color(), T_MOUNT)
    assert [r[1]["measured"] for r in notes.rows] == [False, False] and "no selection" in notes.rows[1][1]["why"]
    # what establishes it: a selection's tie that took the find ...
    assert cc.corroborated(loc, None, tied=True) == (cc.BY_TIE, "")
    # ... or, with no selection, ONE container of the kind in the scene camera's view AND no second lid-sized top
    # in the wrist's frames. A locator that cannot be asked (this branch's own) is no corroboration
    assert cc.corroborated(_locator(), [])[1].endswith(
        "the scene camera cannot say it holds only one container of the kind"
    )
    assert cc.corroborated(_locator(only_one=False), [])[1].endswith("does not hold exactly one container of the kind")
    assert cc.corroborated(_locator(only_one=True), None)[1].endswith("what else the wrist's frames hold is not known")
    assert cc.corroborated(_locator(only_one=True), [(0.81, 0.02)])[1].endswith(
        "the wrist's frames hold another lid-sized top at [0.810, 0.020]"
    )
    assert cc.corroborated(_locator(only_one=True), []) == (cc.BY_ONE, "")
    broken = _locator()
    broken.only_one = lambda: 1 / 0
    assert cc.corroborated(broken, [])[0] == ""  # what cannot be counted is not "only one"


def test_nothing_is_carried_from_run_to_run_and_keep_is_no_mode(tmp_path, monkeypatch):
    """camera_check: "run" (the default: unset, and any word it does not know) measures and turns within the run;
    "none" leaves the scene camera as its mount has it. Nothing is read from or written to the state folder in
    either, and "keep" (the measure carried to the next run) is not a mode: withdrawn in review."""
    monkeypatch.setenv("BOX_OPENING_STATE", str(tmp_path))
    for word in ("", "run", "keep", "true", "1", "on", "whatever"):
        assert cc.mode_of({"BOX_OPENING_CAMERA_CHECK": word}) == cc.RUN, word
    for word in ("none", "None", " off ", "0", "false", "no", "disabled"):
        assert cc.mode_of({"BOX_OPENING_CAMERA_CHECK": word}) == cc.NONE, word
    assert cc.mode_of({}) == cc.RUN and cc.MODES == ("run", "none") and "not available" in cc.NO_KEEP
    monkeypatch.delenv("BOX_OPENING_CAMERA_CHECK", raising=False)
    assert cc.CameraCheck().mode == cc.RUN and cc.CameraCheck(mode="keep").mode == cc.RUN
    run = _locator()
    d = cc.adopt(None, run, _fix(PAIRS[3][0]), (*PAIRS[3][1], 0.40), by="a", paired=cc.BY_TIE)
    assert d is not None and np.allclose(run.base_to_color(), cc.rotation(LENS, d.turn_rad) @ T_MOUNT)
    nxt = _locator()  # the next run: from the mount as it is
    assert np.allclose(nxt.base_to_color(), T_MOUNT) and cc.in_force(nxt) == (0.0, 0.0) and nxt.check.by == ""
    off = _locator(mode=cc.NONE)
    assert cc.adopt(None, off, _fix(PAIRS[3][0]), (*PAIRS[3][1], 0.40), paired=cc.BY_TIE) is None
    assert np.allclose(off.base_to_color(), T_MOUNT) and cc.own_fix(off, _fix((0.6, 0.0))).top_xyz[:2] == (0.6, 0.0)
    assert list(tmp_path.rglob("*")) == []  # nothing was written anywhere under the state folder


def test_what_is_no_measure_changes_nothing_and_never_ends_a_run():
    loc, notes, tie = _locator(), _Notes(), {"paired": cc.BY_TIE}
    assert cc.adopt(None, loc, None, (0.7, 0.0, 0.4), **tie) is None  # the scene camera had no fix
    assert cc.adopt(None, loc, _fix((0.7, 0.0), rough=True), (0.75, -0.1, 0.4), **tie) is None  # a rough one
    assert cc.adopt(None, loc, _fix((0.7, 0.0), source="wrist"), (0.75, -0.1, 0.4), **tie) is None  # not the scene's
    assert cc.adopt(None, loc, _fix((0.70, 0.30)), (0.70, -0.25, 0.4), notes, **tie) is None  # 37 deg apart
    assert cc.adopt(None, loc, _fix((0.70, 0.30)), (float("nan"), 0.0, 0.4), notes, **tie) is None  # not a place
    assert [r[1]["measured"] for r in notes.rows] == [False, False] and "more than 25 deg" in notes.rows[0][1]["why"]
    assert cc.adopt(None, SimpleNamespace(), _fix((0.7, 0.0)), (0.75, -0.1, 0.4), **tie) is None  # no check on it
    assert cc.adopt(None, None, _fix((0.7, 0.0)), (0.75, -0.1, 0.4), **tie) is None
    broken = _locator()
    broken.grab = SimpleNamespace(link_frame="scene", lookup=lambda *_: 1 / 0, link_to_color=lambda: np.eye(4))
    assert cc.adopt(None, broken, _fix((0.7, 0.0)), (0.75, -0.1, 0.4), **tie) is None  # said, never raised
    frozen = cc.adopt(object(), loc, _fix(PAIRS[1][0]), (*PAIRS[1][1], 0.4), **tie)  # a context that holds nothing
    assert frozen is not None and loc.check.by == "run"


def test_a_fix_made_before_the_correction_was_known_can_be_put_where_the_correction_has_it():
    """turned_fix: for a caller that holds a scene fix from before a measure (the look pose's own find)."""
    d = _measures()[1]
    fix = _fix(PAIRS[1][0])
    got = cc.turned_fix(fix, LENS, d.turn_rad, d.along_m)
    assert got.top_xyz[:2] == pytest.approx(PAIRS[1][1]) and got.pose.xyz[:2] == pytest.approx(PAIRS[1][1])
    assert got.top_xyz[2] == fix.top_xyz[2] and got.pose.xyz[2] == fix.pose.xyz[2] and got.table_z_scene == 0.343
    assert got.pose.yaw == pytest.approx(0.1 + d.turn_rad)
    want = np.array([cc.turn_xyz(LENS, d.turn_rad, p) for p in fix.table_points])
    assert np.allclose(got.table_points, want, atol=1e-6) and got.table_points.dtype == np.float32
    x0, x1, y0, y1 = got.table_extent  # the box round the turned one: it holds every turned corner
    corners = [cc.turn_xyz(LENS, d.turn_rad, (x, y, 0.0)) for x in fix.table_extent[:2] for y in fix.table_extent[2:]]
    assert all(x0 - 1e-9 <= c[0] <= x1 + 1e-9 and y0 - 1e-9 <= c[1] <= y1 + 1e-9 for c in corners)
    assert cc.turned_fix(fix, LENS, 0.0) is fix and cc.turned_fix(None, LENS, 0.1) is None


def test_every_fix_and_every_sighting_the_locator_gives_is_placed_alike(capsys):
    """SceneLocator.locate returns its look's fix through own_fix, and tells each sighting the same way: the fix
    before a press and the one after it are placed alike, and a line-up's proof (of a sighting) and the run's plan
    (of the fix) are of one place. As first written a sighting was told 28-36 mm from where the fix then stood,
    and a proof speaks for a box within 15 mm (line_up.PROOF_NEAR_M)."""
    from box_opening import line_up
    from box_opening.perception.scene_source import Sighting

    loc = _locator()
    raw = _fix((0.68, -0.03))
    loc._locate = lambda timeout_s: raw
    assert loc.locate(1.0) is raw  # nothing in force
    loc.check.turn_rad, loc.check.along_m, loc.check.by = math.radians(-14.0), -0.030, "run"
    got = loc.locate(1.0)
    was = math.hypot(raw.top_xyz[0] - LENS[0], raw.top_xyz[1] - LENS[1])
    now = math.hypot(got.top_xyz[0] - LENS[0], got.top_xyz[1] - LENS[1])
    assert now == pytest.approx(was - 0.030) and got.top_xyz[2] == raw.top_xyz[2]  # 30 mm nearer its lens, no lower
    assert math.atan2(got.top_xyz[1] - LENS[1], got.top_xyz[0] - LENS[0]) == pytest.approx(
        math.atan2(raw.top_xyz[1] - LENS[1], raw.top_xyz[0] - LENS[0])
    )  # (along its own line: the turn is in what the look lifts, not added here)
    assert got.pose.xyz[:2] == pytest.approx(got.top_xyz[:2]) and got.table_z_scene == raw.table_z_scene
    # ... and a sighting of the same lift, as it is told
    told = []
    loc.on_sighting, loc._clock = lambda s, age: told.append(s), lambda: 10.0
    sight = Sighting("segmenter", 0.8, raw.top_xyz, raw.pose, 190, (), 0.343, "side", (1, 2, 3, 4), (1, 2, 3, 4))
    msg = SimpleNamespace(stamp_ns=None, recv_s=9.9)
    loc._tell(sight, msg, (10.0, "f", (640, 400), None))
    assert told[0].top == pytest.approx(got.top_xyz) and told[0].pose.xyz == pytest.approx(got.pose.xyz)
    assert math.hypot(told[0].top[0] - got.top_xyz[0], told[0].top[1] - got.top_xyz[1]) < line_up.PROOF_NEAR_M / 100
    rough = Sighting("segmenter", 0.8, raw.top_xyz, raw.pose, 30, ("left",), 0.343, "rough", (1, 2, 3, 4), None)
    loc._tell(rough, msg, (10.0, "f", (640, 400), None))
    assert told[1] is rough  # (a rough one's place is its own fit's, as a rough fix's is)
    # what it was BEFORE the correction, for measuring and for what is kept of the run
    assert cc.raw_xyz(loc, got.top_xyz) == pytest.approx(cc.turn_xyz(LENS, math.radians(14.0), raw.top_xyz))
    assert cc.kept_row(loc, got.top_xyz) == {
        "turn_deg": -14.0,
        "along_mm": -30.0,
        "by": "run",
        "fix_xyz_run": [round(v, 5) for v in got.top_xyz],
    }
    loc._locate = lambda timeout_s: None
    assert loc.locate(1.0) is None
    bare = SceneLocator.__new__(SceneLocator)  # one built with no check at all (a test's): as it always was
    bare._locate = lambda timeout_s: raw
    assert (
        bare.locate(1.0) is raw and cc.kept_row(bare) is None and cc.raw_xyz(bare, (1.0, 2.0, 3.0)) == (1.0, 2.0, 3.0)
    )
    stand_in = SimpleNamespace(base_to_link=lambda: "its own")  # a stand-in that knows no raw mount: the one it has
    assert cc.raw_mount(stand_in) == "its own"
    assert capsys.readouterr().out == ""


def test_a_scene_fix_held_across_a_measure_can_be_put_where_the_new_correction_has_it():
    """refix: a caller that flies the scene camera's fix, has the wrist find the same lid part way (the look
    pose), and goes on using the fix it holds (the look again's pooled counter)."""
    loc = _locator()
    assert cc.in_force(loc) == (0.0, 0.0) and cc.in_force(SimpleNamespace()) == (0.0, 0.0)
    held = _fix(PAIRS[1][0])  # returned with nothing in force
    was = cc.in_force(loc)
    d = cc.adopt(None, loc, held, (*PAIRS[1][1], 0.40), paired=cc.BY_TIE)
    assert cc.in_force(loc) == (d.turn_rad, d.along_m)
    now = cc.refix(loc, held, was)
    assert now.top_xyz[:2] == pytest.approx(PAIRS[1][1]) and now.top_xyz[2] == held.top_xyz[2]
    want = np.array([cc.turn_xyz(LENS, d.turn_rad, p) for p in held.table_points])
    assert np.allclose(now.table_points, want, atol=1e-6)  # its table points turned with it (not moved along)
    # ... which is where a look made now puts the same container: turned by the locator, moved along by own_fix
    assert _seen(loc, PAIRS[1][0]).top_xyz[:2] == pytest.approx(now.top_xyz[:2])
    assert cc.refix(loc, held, cc.in_force(loc)) is held  # nothing changed since: as it is
    wrist = _fix((0.6, 0.0), source="wrist")
    assert cc.refix(loc, wrist, was) is wrist and cc.refix(loc, None, was) is None
