"""The scene camera checked against the wrist camera (perception/camera_check): what one pair measures, what
is turned by it, and the guess the next run starts from."""

import json
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


def _locator(tmp_path, mode=cc.KEEP, T=T_MOUNT):
    """A scene locator as built bare, placed by TF at `T` (its link the colour frame), with a check of its own."""
    loc = SceneLocator.__new__(SceneLocator)
    loc.grab = SimpleNamespace(link_frame="scene", lookup=lambda parent, child: T, link_to_color=lambda: np.eye(4))
    loc.T_base_link, loc.base_frame = None, "arm_mount"
    loc.check = cc.CameraCheck(mode=mode, path=tmp_path / "scene_calib" / cc.FILE_NAME, clock=lambda: 5000.0)
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


def test_a_wrist_find_taken_for_the_scene_cameras_container_turns_the_scene_camera_for_the_rest_of_the_run(
    tmp_path, capsys
):
    """adopt(): measured, said, noted, warned of once, in force on the locator (every later number it lifts),
    and kept for the next run. The team's mount itself is what it was (raw)."""
    loc, notes = _locator(tmp_path), _Notes()
    assert np.allclose(loc.base_to_color(), T_MOUNT)  # nothing in force: as its mount has it
    wrist = _fix(PAIRS[1][1], z=0.405, table=0.313, source="wrist")
    d = cc.adopt(None, loc, _fix(PAIRS[1][0]), wrist, notes, "r1")
    assert d.turn_deg == pytest.approx(-13.94, abs=0.01) and loc.check.by == "run"
    T = loc.base_to_color()
    assert np.allclose(T, cc.rotation(LENS, loc.check.turn_rad) @ T_MOUNT) and np.allclose(T[:3, 3], T_MOUNT[:3, 3])
    assert np.allclose(loc.base_to_color(raw=True), T_MOUNT) and np.allclose(loc.base_to_link(raw=True), T_MOUNT)
    assert np.allclose(loc.base_to_link(), T)  # (its link is its colour frame here)
    # the same container again later in the run, as the turned locator lifts it: no second warning
    seen = cc.own_fix(loc, _fix(cc.turn_xyz(LENS, loc.check.turn_rad, (*PAIRS[4][0], 0.443))[:2]))
    assert cc.adopt(None, loc, seen, (*PAIRS[4][1], 0.40), notes, by="r1") is not None
    said = [m for m in capsys.readouterr().out.splitlines() if "CAMERAS" in m or "WARNING" in m]
    assert "the scene camera has the container at [0.680, -0.034] and the wrist camera at [0.770, -0.192]" in said[0]
    assert "turned -13.9 deg about its lens for the rest of this run, and the next starts from that" in said[0]
    assert "WARNING: the scene camera's mount looks 14 deg off" in said[1] and "tell the team" in said[1]
    assert sum("WARNING" in m for m in said) == 1  # once a run
    kind, row = notes.rows[0]
    assert kind == "camera_check" and row["measured"] and row["turn_deg"] == -13.94 and row["by"] == "run"
    assert row["scene_xy"] == [0.68, -0.034] and row["wrist_xy"] == [0.77, -0.192] and row["guess_deg"] is None
    # what the next run reads: both measures, under this mount
    doc = json.loads((tmp_path / "scene_calib" / cc.FILE_NAME).read_text())
    assert [r["by"] for r in doc["measures"]] == ["r1", "r1"] and doc["mount"] == cc.mount_key(T_MOUNT)
    # the run's record of a later scene look names the turn its numbers were lifted with
    loc._look, loc.last_timing, loc.instance_id = None, {}, None
    assert loc.look_record()["camera_check"]["by"] == "run"


def test_a_fix_lifted_with_a_turn_in_force_is_measured_as_the_mount_itself_has_it(tmp_path):
    """The run's second look is lifted through the turned placement. Measured against the wrist again, what
    is in force afterwards is the scene camera's OWN disagreement, not the remainder added to nothing."""
    loc = _locator(tmp_path)
    first = cc.adopt(None, loc, _fix(PAIRS[1][0]), (*PAIRS[1][1], 0.40))
    # the same container at the next chair place, as the turned locator now lifts it
    seen = cc.own_fix(loc, _fix(cc.turn_xyz(LENS, loc.check.turn_rad, (*PAIRS[3][0], 0.443))[:2]))
    second = cc.adopt(None, loc, seen, (*PAIRS[3][1], 0.40))
    assert first.turn_deg == pytest.approx(-13.94, abs=0.01) and second.turn_deg == pytest.approx(-14.94, abs=0.01)
    assert loc.check.turn_rad == second.turn_rad


def test_the_next_run_starts_from_the_last_ones_measure_as_a_guess_and_its_own_replaces_it(tmp_path):
    first = _locator(tmp_path)
    for (s, w), t in zip(PAIRS[:3], (100.0, 200.0, 300.0), strict=True):
        first.check._clock = lambda t=t: t
        seen = cc.own_fix(first, _fix(cc.turn_xyz(LENS, first.check.turn_rad, (*s, 0.443))[:2]))
        cc.adopt(None, first, seen, (*w, 0.40), by="a")
    nxt = _locator(tmp_path)
    T = nxt.base_to_color()  # the first number it lifts: the guess is read, with the mount it is keyed on
    g = nxt.check.guess
    assert nxt.check.by == "guess" and g.n == 3 and math.degrees(g.turn_rad) == pytest.approx(-13.9, abs=0.01)
    assert np.allclose(T, cc.rotation(LENS, g.turn_rad) @ T_MOUNT)
    assert g.words(now=300.0 + 12 * 60).startswith("-13.9 deg about its lens, 29 mm nearer along it")
    assert g.words(now=300.0 + 12 * 60).endswith(
        "the last 3 runs measured against the wrist camera, the newest 12 min ago"
    )
    # the scene camera's own fix of the container is moved along its line by the guess; a wrist's or a rough one not
    fix = _fix(cc.turn_xyz(LENS, g.turn_rad, (*PAIRS[4][0], 0.443))[:2])
    own = cc.own_fix(nxt, fix)
    assert math.hypot(own.top_xyz[0] - PAIRS[4][1][0], own.top_xyz[1] - PAIRS[4][1][1]) < 0.01
    assert own.top_xyz[2] == fix.top_xyz[2] and own.table_z_scene == fix.table_z_scene and own.pose.xyz[2] == 0.343
    assert np.array_equal(own.table_points, fix.table_points)  # (the table round it is not moved along)
    assert cc.own_fix(nxt, _fix((0.6, 0.0), source="wrist")).top_xyz[:2] == (0.6, 0.0)
    assert cc.own_fix(nxt, _fix((0.6, 0.0), rough=True)).top_xyz[:2] == (0.6, 0.0)
    # its own pair (the fix as its look returned it): in force, kept beside the others (4 now), near the guess
    cc.adopt(None, nxt, own, (*PAIRS[4][1], 0.40), by="b")
    assert nxt.check.by == "run" and math.degrees(nxt.check.turn_rad) == pytest.approx(-14.0, abs=0.1)
    doc = json.loads(nxt.check.path.read_text())
    assert [r["by"] for r in doc["measures"]] == ["a", "a", "a", "b"]


def test_a_run_that_measures_far_from_the_guess_says_so_and_keeps_only_its_own(tmp_path):
    """The camera was moved again (or put right) between two runs: the older measures describe a camera that
    no longer is."""
    first = _locator(tmp_path)
    cc.adopt(None, first, _fix(PAIRS[1][0]), (*PAIRS[1][1], 0.40), by="old")
    nxt = _locator(tmp_path)
    nxt.base_to_color()
    assert nxt.check.by == "guess"
    # the camera now agrees with the wrist: its own placement is the wrist's, returned with the guess in it
    seen = cc.own_fix(nxt, _fix(cc.turn_xyz(LENS, nxt.check.turn_rad, (0.70, -0.10, 0.44))[:2]))
    notes = _Notes()
    d = cc.adopt(None, nxt, seen, (0.70, -0.10, 0.40), notes, by="new")
    assert abs(d.turn_deg) < 0.01 and nxt.check.turn_rad == d.turn_rad
    assert notes.rows[0][1]["from_guess_deg"] == pytest.approx(13.94, abs=0.01)
    assert [r["by"] for r in json.loads(nxt.check.path.read_text())["measures"]] == ["new"]


def test_a_guess_belongs_to_the_mount_it_was_measured_under(tmp_path):
    """The team puts the mount right, or the stack is another robot's: the guess is dropped, not carried over."""
    first = _locator(tmp_path)
    cc.adopt(None, first, _fix(PAIRS[1][0]), (*PAIRS[1][1], 0.40))
    moved = cc.rotation(LENS, math.radians(-14.0)) @ T_MOUNT  # the mount as a recalibration would write it
    other = _locator(tmp_path, T=moved)
    assert np.allclose(other.base_to_color(), moved) and other.check.by == "" and other.check.guess is None
    assert cc.load_guess(other.check.path, cc.mount_key(moved))[1].startswith(
        "the scene camera's mount is not the one the last runs' measure against the wrist camera was made under"
    )
    assert cc.same_mount(cc.mount_key(T_MOUNT), cc.mount_key(T_MOUNT))
    shifted = T_MOUNT.copy()
    shifted[0, 3] += 0.005
    assert not cc.same_mount(cc.mount_key(T_MOUNT), cc.mount_key(shifted))
    assert not cc.same_mount(cc.mount_key(T_MOUNT), None) and not cc.same_mount([1, 2], cc.mount_key(T_MOUNT))
    # a file that is not one, or whose numbers no pair could give: no guess, and no error
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    assert cc.load_guess(path, cc.mount_key(T_MOUNT)) == (None, None)
    path.write_text(json.dumps({"mount": cc.mount_key(T_MOUNT), "measures": [{"turn_deg": 60.0, "along_mm": 0.0}]}))
    assert cc.load_guess(path, cc.mount_key(T_MOUNT))[0] is None
    assert cc.load_guess(tmp_path / "none.json", cc.mount_key(T_MOUNT)) == (None, None)


def test_only_the_newest_five_measures_are_kept_and_the_guess_is_their_median(tmp_path):
    path, key = tmp_path / "c.json", cc.mount_key(T_MOUNT)
    for i, deg in enumerate((-10.0, -14.0, -14.2, -13.8, -30.0 / 3, -14.1, -13.9)):
        cc.save_measure(path, key, cc.Disagreement(math.radians(deg), -0.03, t=float(i)), by=str(i))
    doc = json.loads(path.read_text())
    assert [r["by"] for r in doc["measures"]] == ["2", "3", "4", "5", "6"] and cc.GUESS_KEPT == 5
    g, why = cc.load_guess(path, key)
    assert why is None and g.n == 5 and math.degrees(g.turn_rad) == pytest.approx(-13.9) and g.t == 6.0


def test_the_check_can_be_run_without_the_guess_or_not_at_all(tmp_path, monkeypatch):
    """camera_check: "run" measures and turns within the run and reads and keeps nothing; "none" leaves the
    scene camera as its mount has it, as before 2026-11-20."""
    seeded = _locator(tmp_path)
    cc.adopt(None, seeded, _fix(PAIRS[1][0]), (*PAIRS[1][1], 0.40), by="seed")
    kept = seeded.check.path.read_text()
    run = _locator(tmp_path, mode=cc.RUN)
    assert np.allclose(run.base_to_color(), T_MOUNT) and run.check.guess is None  # no guess read
    d = cc.adopt(None, run, _fix(PAIRS[3][0]), (*PAIRS[3][1], 0.40))
    assert d is not None and np.allclose(run.base_to_color(), cc.rotation(LENS, d.turn_rad) @ T_MOUNT)
    assert run.check.path.read_text() == kept  # ... and none kept
    off = _locator(tmp_path, mode=cc.NONE)
    assert cc.adopt(None, off, _fix(PAIRS[3][0]), (*PAIRS[3][1], 0.40)) is None
    assert np.allclose(off.base_to_color(), T_MOUNT) and off.check.path.read_text() == kept
    assert cc.own_fix(off, _fix((0.6, 0.0))).top_xyz[:2] == (0.6, 0.0)
    for word, mode in (("", cc.KEEP), ("keep", cc.KEEP), ("run", cc.RUN), ("none", cc.NONE), ("off", cc.NONE)):
        assert cc.mode_of({"BOX_OPENING_CAMERA_CHECK": word}) == mode
    assert cc.mode_of({}) == cc.KEEP and cc.MODES == ("keep", "run", "none")
    monkeypatch.setenv("BOX_OPENING_STATE", str(tmp_path / "state"))
    monkeypatch.setenv("BOX_OPENING_CAMERA_CHECK", "run")
    fresh = cc.CameraCheck()
    assert fresh.mode == cc.RUN and fresh.path == tmp_path / "state" / "scene_calib" / "camera_check.json"


def test_what_is_no_measure_changes_nothing_and_never_ends_a_run(tmp_path):
    loc, notes = _locator(tmp_path), _Notes()
    assert cc.adopt(None, loc, None, (0.7, 0.0, 0.4)) is None  # the scene camera had no fix
    assert cc.adopt(None, loc, _fix((0.7, 0.0), rough=True), (0.75, -0.1, 0.4)) is None  # a rough one: no measure
    assert cc.adopt(None, loc, _fix((0.7, 0.0), source="wrist"), (0.75, -0.1, 0.4)) is None  # not the scene camera's
    assert cc.adopt(None, loc, _fix((0.70, 0.30)), (0.70, -0.25, 0.4), notes) is None  # two things, not one twice
    assert notes.rows == [("camera_check", {"measured": False, "why": notes.rows[0][1]["why"], **loc.check.row()})]
    assert cc.adopt(None, SimpleNamespace(), _fix((0.7, 0.0)), (0.75, -0.1, 0.4)) is None  # a locator with no check
    assert cc.adopt(None, None, _fix((0.7, 0.0)), (0.75, -0.1, 0.4)) is None
    broken = _locator(tmp_path)
    broken.grab = SimpleNamespace(link_frame="scene", lookup=lambda *_: 1 / 0, link_to_color=lambda: np.eye(4))
    assert cc.adopt(None, broken, _fix((0.7, 0.0)), (0.75, -0.1, 0.4)) is None  # said, never raised
    assert loc.check.by == "" and not loc.check.path.exists() and np.allclose(loc.base_to_color(), T_MOUNT)


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


def test_every_fix_the_locator_returns_is_placed_by_the_check_in_force(tmp_path):
    """SceneLocator.locate returns its look's fix through own_fix: the fix before a press and the one after it
    are placed alike, whatever is in force, and a locator with nothing in force returns it untouched."""
    loc = _locator(tmp_path)
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
    loc._locate = lambda timeout_s: None
    assert loc.locate(1.0) is None
    bare = SceneLocator.__new__(SceneLocator)  # one built with no check at all (a test's): as it always was
    bare._locate = lambda timeout_s: raw
    assert bare.locate(1.0) is raw


def test_a_scene_fix_held_across_a_measure_can_be_put_where_the_new_correction_has_it(tmp_path):
    """refix: a caller that flies the scene camera's fix, has the wrist find the same lid part way (the look
    pose), and goes on comparing later scene looks with the fix it holds."""
    loc = _locator(tmp_path)
    assert cc.in_force(loc) == (0.0, 0.0) and cc.in_force(SimpleNamespace()) == (0.0, 0.0)
    held = _fix(PAIRS[1][0])  # returned with nothing in force
    was = cc.in_force(loc)
    d = cc.adopt(None, loc, held, (*PAIRS[1][1], 0.40))
    assert cc.in_force(loc) == (d.turn_rad, d.along_m)
    now = cc.refix(loc, held, was)
    assert now.top_xyz[:2] == pytest.approx(PAIRS[1][1]) and now.top_xyz[2] == held.top_xyz[2]
    # ... which is where a look made now puts the same container: turned by the locator, moved along by own_fix
    again = cc.own_fix(loc, _fix(cc.turn_xyz(LENS, d.turn_rad, (*PAIRS[1][0], 0.443))[:2]))
    assert again.top_xyz[:2] == pytest.approx(now.top_xyz[:2])
    assert cc.refix(loc, held, cc.in_force(loc)) is held  # nothing changed since: as it is
    wrist = _fix((0.6, 0.0), source="wrist")
    assert cc.refix(loc, wrist, was) is wrist and cc.refix(loc, None, was) is None
