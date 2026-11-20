"""The scene camera checked against the wrist camera, within a run.

Both cameras place the container in the arm's frame: the scene camera by its mount (the robot's TF, or
the calibration file), the wrist camera by the arm that carries it. The arm goes where the wrist camera
sees, so wherever the two have the same container, how far apart they have it is the scene camera's
error as the arm meets it.

2026-11-20, RAMMP 1.5, five sightings of one container by both, the chair at four places (the scene
lens at [0.151, -0.572]):

    scene [0.912,  0.000]  wrist [0.988, -0.201]   215 mm apart   -13.0 deg about the lens, -36 mm along
    scene [0.680, -0.034]  wrist [0.770, -0.192]   182 mm         -13.9 deg, -28 mm
    scene [0.681, -0.035]  wrist [0.770, -0.193]   181 mm         -13.9 deg, -29 mm
    scene [0.540, -0.049]  wrist [0.635, -0.188]   168 mm         -14.9 deg, -34 mm
    scene [0.559, -0.078]  wrist [0.643, -0.208]   155 mm         -14.0 deg, -29 mm

The distance between them changes with how far the container stands; the angle about the scene camera's
lens changes much less. So the disagreement is measured as what ONE pair gives without assuming more:
the turn about the vertical through the scene lens that brings the scene camera's direction to the
container onto the wrist's, how much nearer or farther along that line the wrist has it, and how much
higher.

WHAT THAT IS AND IS NOT. It is not a calibration. One pair cannot tell a turn about the vertical from a
tilt or a shift, and the five pairs do not say the mount is only turned: a free rigid fit to them is
-10.4 deg about a point 294 mm behind the lens, and the scene camera's table reads 0.155 m higher per
metre of range, so the mount is tilted too (the review of 2026-11-20). What the five pairs establish:
with one pair's measure applied to the other four, the container itself comes out within 31 mm of where
the wrist has it (48 mm with the turn alone); with the other four's median applied to the fifth, 5-17 mm
(28-39 mm with the turn alone). Untouched it was 155-215 mm off. A thing elsewhere on the table comes
out 3-6 cm off (the turn about the lens and the free fit differ 25-67 mm across it). Better than 15 cm,
not exact, and none of it is flown.

WHEN A PAIR IS MEASURED. Only where the two fixes are known to be of ONE container (adopt's `paired`,
which the caller fills: corroborated): a selection's tie took the wrist's find, or the scene camera
holds exactly one container of the kind and the wrist's frames no second lid-sized top. A lid 15 cm to
the side of a healthy camera's box would otherwise measure as -12 deg: a wrong warning and a wrong turn.

WHAT IS DONE WITH IT, for the rest of the run only (the owner's rules: nothing recalibrated, the team's
TF and files untouched, the scene measured each run): the scene camera's placement as box opening reads
it (SceneLocator.base_to_color, and everything lifted through it afterwards: its fixes and its
sightings in a line-up, the line of sight a selection is tied by, the points a later look takes) is
turned by that angle about the vertical through its lens, and the container's own place (a fix, a
sighting: own_xyz) is moved along the line by the along-line difference. Heights are never moved.

WHAT A RUN SEES OF IT. A run reads its scene camera again after the wrist's find only where the chair is
guided and it looks a second time, or where a look again pools what the scene camera saw. In an ordinary
run what this gives is the CAMERAS line, the WARNING to tell the team (the flight view shows both), and
a second round that starts corrected.

NOTHING IS CARRIED FROM RUN TO RUN. A measure kept as the next run's starting guess was built and
withdrawn in review (2026-11-20): a guess that outlives the camera's being put right puts the scene
camera's fix of one container onto its neighbour, the arm opens the neighbour, and the pair that then
measures "confirms" the guess. No rule on the file's age or on its rows' agreement makes that sound, so
"keep" is not a mode: every run starts from the mount as it is, and nothing is read from or written to
the state folder.

Set by BOX_OPENING_CAMERA_CHECK (constants.CAMERA_CHECK_ENV; the node's camera_check): "run" (the
default, and what an unset or unknown word is read as), or "none" (nothing measured, nothing turned).

Pure numpy; the ROS side is the locator's.
"""

import math
import os
import time
from dataclasses import dataclass, replace

import numpy as np

from box_opening import LOGGER as log
from box_opening.constants import CAMERA_CHECK_ENV

RUN, NONE = "run", "none"
MODES = (RUN, NONE)
# asked for by name and not given: a correction carried from one run to the next (the module's note)
KEEP = "keep"
NO_KEEP = (
    '"keep" (a correction carried from one run to the next) is not available: it can confirm itself on the '
    "wrong container"
)
# what "none" is also spelled as by hand; every other word, and none at all, is RUN
NONE_WORDS = (NONE, "off", "0", "false", "no", "disabled")

# A pair is one container seen by both cameras only where the two directions from the scene lens are no
# further apart than this. 2026-11-20 measured 13-15 deg and the camera was found turned ~15 deg once
# before (flatbot, 2026-11-17); at 25 deg the scene camera would have the box 0.3 m off at 0.7 m, which
# is no camera to correct in silence.
TURN_MAX_DEG = 25.0
# ... and no further apart along that line than the range error a selection is tied within
# (press_demo.SELECTED_ALONG_M): what was seen was 28-36 mm (above) and 80 mm (2026-11-11).
ALONG_MAX_M = 0.20
# A fix nearer the lens than this gives no direction worth the name: a centimetre of either camera's own
# scatter is 2.3 deg there. Nothing the arm opens stands that near the scene camera.
NEAR_LENS_M = 0.25
# A turn past this is said as a WARNING, once a run: at 0.7 m it is 37 mm, half of what ties a wrist find
# to a selection (press_demo.SELECTED_ACROSS_M, 75 mm) and most of what scene_trust distrusts (50 mm).
TURN_WARN_DEG = 3.0

# What establishes that a wrist find and the scene camera's fix are one container (corroborated)
BY_TIE = "the selection's tie took the wrist's find"
BY_ONE = "the scene camera holds one container of the kind and the wrist's frames one lid-sized top"


def mode_of(env=None):
    """RUN or NONE by $BOX_OPENING_CAMERA_CHECK: NONE for "none" and the words that mean it by hand
    (NONE_WORDS), RUN for everything else, unset and "keep" included (the node refuses "keep" and any
    word it does not know before a run starts: node.Capability._camera_check)."""
    word = str((os.environ if env is None else env).get(CAMERA_CHECK_ENV, "")).strip().lower()
    return NONE if word in NONE_WORDS else RUN


@dataclass(frozen=True)
class Disagreement:
    """How the scene camera's placement of a container differs from the wrist's, about the scene lens.
    `turn_rad`: the turn about the vertical through the lens that brings the scene camera's direction to
    it onto the wrist's (counter-clockwise from above positive). `along_m`: the wrist's distance from
    the lens less the scene camera's (negative: the wrist has it nearer). `dz_m`: the wrist's lid top
    less the scene camera's (nan where one of them has none). `range_m`: the scene camera's distance
    from its lens to it. `apart_m`: how far apart the two had it on the table. `t`: when (unix)."""

    turn_rad: float
    along_m: float
    dz_m: float = float("nan")
    range_m: float = float("nan")
    apart_m: float = float("nan")
    t: float = 0.0

    @property
    def turn_deg(self):
        return math.degrees(self.turn_rad)

    def words(self):
        """'-13.9 deg about its lens and 29 mm nearer along it (182 mm apart at 0.75 m), the wrist's lid
        36 mm lower'."""
        out = "%+.1f deg about its lens and %.0f mm %s along it" % (
            self.turn_deg,
            1000 * abs(self.along_m),
            "nearer" if self.along_m < 0 else "farther",
        )
        if math.isfinite(self.apart_m) and math.isfinite(self.range_m):
            out += " (%.0f mm apart at %.2f m)" % (1000 * self.apart_m, self.range_m)
        if math.isfinite(self.dz_m):
            out += ", the wrist's lid %.0f mm %s" % (1000 * abs(self.dz_m), "lower" if self.dz_m < 0 else "higher")
        return out

    def row(self):
        return {
            "t": round(float(self.t), 1),
            "turn_deg": round(self.turn_deg, 2),
            "along_mm": round(1000 * self.along_m, 1),
            "dz_mm": None if not math.isfinite(self.dz_m) else round(1000 * self.dz_m, 1),
            "range_m": None if not math.isfinite(self.range_m) else round(self.range_m, 3),
            "apart_mm": None if not math.isfinite(self.apart_m) else round(1000 * self.apart_m, 1),
        }


def measure(lens_xy, scene_xyz, wrist_xyz, t=None):
    """(Disagreement, None) of one pair: where the scene camera has a container (`scene_xyz`, as its
    mount places it: nothing of an earlier correction in it) and where the wrist has the same one, both
    in the arm's frame, `lens_xy` the scene lens there. (None, why) for a pair that is no measure: a fix
    too near the lens to give a direction (NEAR_LENS_M), the two directions further apart than
    TURN_MAX_DEG, or the two ranges than ALONG_MAX_M. Whether the two ARE one container is the
    caller's to know (a selection's tie, one container in each view): these bounds only refuse what
    could not be."""
    lens = np.asarray(lens_xy, float)[:2]
    s = np.asarray(scene_xyz, float)
    w = np.asarray(wrist_xyz, float)
    ds, dw = s[:2] - lens, w[:2] - lens
    rs, rw = float(np.hypot(*ds)), float(np.hypot(*dw))
    if not (np.isfinite(ds).all() and np.isfinite(dw).all()):
        return None, "a fix that is not a place"
    if min(rs, rw) < NEAR_LENS_M:
        return None, "a fix %.2f m from the scene lens gives no direction (under %.2f m)" % (min(rs, rw), NEAR_LENS_M)
    turn = math.atan2(ds[0] * dw[1] - ds[1] * dw[0], ds[0] * dw[0] + ds[1] * dw[1])
    along = rw - rs
    apart = float(np.hypot(*(w[:2] - s[:2])))
    if abs(math.degrees(turn)) > TURN_MAX_DEG:
        return None, (
            "the two cameras have it %.1f deg apart about the scene lens (%.0f mm), more than %.0f deg: not taken "
            "for one container seen twice" % (math.degrees(turn), 1000 * apart, TURN_MAX_DEG)
        )
    if abs(along) > ALONG_MAX_M:
        return None, (
            "the two cameras have it %.0f mm apart along the scene camera's line, more than %.0f: not taken for "
            "one container seen twice" % (1000 * abs(along), 1000 * ALONG_MAX_M)
        )
    dz = float(w[2] - s[2]) if len(s) > 2 and len(w) > 2 and np.isfinite(s[2]) and np.isfinite(w[2]) else float("nan")
    return Disagreement(float(turn), float(along), dz, rs, apart, time.time() if t is None else float(t)), None


def rotation(lens_xy, turn_rad):
    """The 4x4 that turns the arm's frame by `turn_rad` about the vertical through `lens_xy`: applied to
    T(arm <- scene camera), the camera stays where it is and what it sees is turned about it."""
    c, s = math.cos(turn_rad), math.sin(turn_rad)
    x, y = float(lens_xy[0]), float(lens_xy[1])
    D = np.eye(4)
    D[:2, :2] = [[c, -s], [s, c]]
    D[0, 3] = x - (c * x - s * y)
    D[1, 3] = y - (s * x + c * y)
    return D


def turn_xyz(lens_xy, turn_rad, xyz, along_m=0.0):
    """A place the scene camera gave, turned `turn_rad` about the vertical through its lens and moved
    `along_m` along the line from the lens to it; its height as it was."""
    p = np.asarray(xyz, float)
    lens = np.asarray(lens_xy, float)[:2]
    c, s = math.cos(turn_rad), math.sin(turn_rad)
    d = p[:2] - lens
    d = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
    r = float(np.hypot(*d))
    if along_m and r > 1e-6:
        d = d * ((r + float(along_m)) / r)
    return (float(lens[0] + d[0]), float(lens[1] + d[1]), *(float(v) for v in p[2:]))


def turned_fix(fix, lens_xy, turn_rad, along_m=0.0):
    """A scene_source.SceneFix made BEFORE a correction was known, as the correction has it: its lid's
    top and its pose turned about the lens and moved along the line (turn_xyz), its yaw turned, its
    table points and the extent they were seen over turned (not moved along: own_xyz's note). Heights,
    and everything else of it, as they were."""
    if fix is None or (not turn_rad and not along_m):
        return fix
    top = turn_xyz(lens_xy, turn_rad, fix.top_xyz, along_m)
    pose = fix.pose
    if pose is not None:
        pose = replace(pose, xyz=turn_xyz(lens_xy, turn_rad, pose.xyz, along_m), yaw=float(pose.yaw) + float(turn_rad))
    points, extent = getattr(fix, "table_points", None), getattr(fix, "table_extent", None)
    D = rotation(lens_xy, turn_rad)
    if points is not None and len(points) and turn_rad:
        p = np.asarray(points, float)
        points = np.c_[p[:, :2] @ D[:2, :2].T + D[:2, 3], p[:, 2:]].astype(np.float32)
    if extent is not None and turn_rad:
        x0, x1, y0, y1 = (float(v) for v in extent)
        corners = np.array([[x0, y0], [x0, y1], [x1, y0], [x1, y1]]) @ D[:2, :2].T + D[:2, 3]
        extent = (
            float(corners[:, 0].min()),
            float(corners[:, 0].max()),
            float(corners[:, 1].min()),
            float(corners[:, 1].max()),
        )
    return replace(fix, top_xyz=top, pose=pose, table_points=points, table_extent=extent)


# ------------------------------------------------------------------------------------- what a locator holds
class CameraCheck:
    """The correction in force for one scene locator, for one run: `turn_rad` about the vertical through
    the lens (what SceneLocator.base_to_color turns its placement by) and `along_m` (own_xyz), none
    until the run's own pair is measured (adopt); `by`: "" (none) or "run"."""

    def __init__(self, mode=None, clock=time.time):
        self.mode = mode_of() if mode is None else (NONE if mode == NONE else RUN)
        self._clock = clock
        self.turn_rad = 0.0
        self.along_m = 0.0
        self.by = ""
        self.measured = None  # this run's newest Disagreement (the scene camera's own placement against the wrist's)
        self.warned = False

    @property
    def active(self):
        return self.mode != NONE

    def adopt(self, d):
        """This run's own measure `d` (the scene camera's own placement against the wrist's) in force
        from here on."""
        self.measured, self.turn_rad, self.along_m, self.by = d, float(d.turn_rad), float(d.along_m), "run"

    def row(self):
        """For the run's record."""
        return {
            "mode": self.mode,
            "by": self.by,
            "turn_deg": round(math.degrees(self.turn_rad), 2),
            "along_mm": round(1000 * self.along_m, 1),
        }


def lens_xy(locator):
    """The scene lens in the arm's frame, (x, y), or None where the locator cannot place its camera. The
    turn is about the vertical through it, so it is the same with and without one."""
    T = locator.base_to_color() if locator is not None else None
    return None if T is None else (float(T[0, 3]), float(T[1, 3]))


def in_force(locator):
    """(turn_rad, along_m) of the correction `locator` places its fixes with now; (0.0, 0.0) with none."""
    check = getattr(locator, "check", None)
    if check is None or not check.active:
        return 0.0, 0.0
    return float(check.turn_rad), float(check.along_m)


def raw_mount(locator):
    """T(arm <- the scene camera's link) as the MOUNT has it (TF, or the calibration file), nothing of a
    correction in it; a stand-in locator that knows no such thing is asked for the one it has. None
    where it cannot place its camera."""
    try:
        return locator.base_to_link(raw=True)
    except TypeError:
        return locator.base_to_link()


def own_xyz(locator, xyz):
    """A place the scene camera gave for THE CONTAINER (lifted through the locator's placement, so the
    turn is in it), as the correction in force has the container itself: moved along the line from the
    lens by the along-line difference. Only the container's own place is moved so: the scene camera sees
    a lid from its side, and what that does to its range says nothing of the table round it. Its height
    as it was. `xyz` itself where nothing is in force or the camera cannot be placed."""
    _turn, along = in_force(locator)
    if not along:
        return xyz
    lens = lens_xy(locator)
    return xyz if lens is None else turn_xyz(lens, 0.0, xyz, along)


def raw_xyz(locator, xyz):
    """A container's place as SceneLocator.locate (or a told sighting) gave it, as the MOUNT itself has
    it: the along-line move and the turn in force taken out again. For measuring the camera against the
    wrist (adopt), and for what is kept of a run for work on the camera's mount (a mission's saved
    frames): a later refit must see the camera's whole error, not what a correction left of it. `xyz`
    itself where nothing is in force."""
    turn, along = in_force(locator)
    if not turn and not along:
        return xyz
    lens = lens_xy(locator)
    if lens is None:
        return xyz
    return turn_xyz(lens, -turn, turn_xyz(lens, 0.0, xyz, -along))


def kept_row(locator, fix_xyz=None):
    """The correction in force as it is kept beside what a run saves of its scene camera (a mission's
    frames: the mount and the fix are saved as the mount has them, raw_mount and raw_xyz), under its own
    name: its turn and along-line move, and the container's place as the run itself went by it
    (`fix_xyz`, as locate() returned it). None where nothing is in force."""
    turn, along = in_force(locator)
    if not turn and not along:
        return None
    return {
        "turn_deg": round(math.degrees(turn), 3),
        "along_mm": round(1000 * along, 1),
        "by": str(getattr(getattr(locator, "check", None), "by", "")),
        "fix_xyz_run": None if fix_xyz is None else [round(float(v), 5) for v in fix_xyz],
    }


def own_fix(locator, fix):
    """The scene camera's fix of the container as the correction in force has the container itself
    (own_xyz: its top and its pose moved along the line of sight; the turn is already in what the look
    lifted). `fix` itself where nothing is in force, the fix is the wrist's, or a rough one (its place is
    its own fit's). SceneLocator.locate returns every fix through this, and tells every sighting the
    same way (own_sighting): a line-up's proof and the run's plan are of one place."""
    if fix is None or getattr(fix, "source", "") == "wrist" or getattr(fix, "rough", False):
        return fix
    top = own_xyz(locator, fix.top_xyz)
    if top is fix.top_xyz:
        return fix
    pose = fix.pose
    if pose is not None:
        pose = replace(pose, xyz=own_xyz(locator, pose.xyz))
    return replace(fix, top_xyz=top, pose=pose)


def own_sighting(locator, sighting):
    """A scene_source.Sighting as it is TOLD (SceneLocator.on_sighting: a line-up's watch), placed as the
    fix made of it would be (own_fix): one placement for both. A rough one as it is. The sightings the
    look itself votes on are not these: it votes on what it lifted, and its fix is moved once."""
    if sighting is None or getattr(sighting, "lift", "") == "rough":
        return sighting
    top = own_xyz(locator, sighting.top)
    if top is sighting.top:
        return sighting
    pose = sighting.pose
    if pose is not None:
        pose = replace(pose, xyz=own_xyz(locator, pose.xyz))
    return replace(sighting, top=tuple(top), pose=pose)


def refix(locator, fix, was):
    """A scene fix that `locator`.locate() returned while the correction `was` was in force (in_force's
    answer then), as the correction in force NOW has it: for a caller that holds such a fix across an
    adopt() and goes on using it (turned_fix by what changed). `fix` itself where nothing changed, it
    is not the scene camera's, or the camera cannot be placed."""
    now, lens = in_force(locator), lens_xy(locator)
    if fix is None or lens is None or getattr(fix, "source", "") == "wrist":
        return fix
    return turned_fix(fix, lens, now[0] - float(was[0]), now[1] - float(was[1]))


def corroborated(locator, others, tied=False):
    """What establishes that a wrist find and the scene camera's fix are ONE container: (its words, "")
    or ("", why nothing does). `tied`: a selection's tie took the wrist's find (the run's own test of
    exactly that). Else, with no selection: the scene camera holds exactly one container of the kind
    (the locator's only_one(), asked with getattr: a locator that cannot say is no corroboration) AND
    the wrist's frames hold no second lid-sized top (`others`: the tops besides the find, [] for none;
    None: not known, which is no corroboration either). Without one of the two a pair is two cameras
    each seeing something, and a lid 15 cm beside a healthy camera's box would be measured as a mount
    12 deg off."""
    if tied:
        return BY_TIE, ""
    no_tie = "no selection ties the wrist's find to the scene camera's box, and "
    one = getattr(locator, "only_one", None)
    try:
        alone = bool(one()) if callable(one) else None
    except Exception:  # noqa: BLE001 - what cannot be counted is not "only one"
        alone = None
    if alone is None:
        return "", no_tie + "the scene camera cannot say it holds only one container of the kind"
    if not alone:
        return "", no_tie + "the scene camera does not hold exactly one container of the kind"
    if others is None:
        return "", no_tie + "what else the wrist's frames hold is not known"
    if len(others):
        return "", no_tie + "the wrist's frames hold another lid-sized top at [%.3f, %.3f]" % (
            float(others[0][0]),
            float(others[0][1]),
        )
    return BY_ONE, ""


def adopt(ctx, locator, scene_fix, wrist_fix, runner=None, by="", paired="", why_not=""):
    """A wrist find was just taken for the container the scene camera has at `scene_fix` (a fix as
    `locator`.locate() returned it, with the correction that is in force NOW: nothing may have changed
    the check between that look and this call): measure the two against each other and put the result
    in force for the rest of the run (the module's note). `paired`: what establishes that the two ARE
    one container (corroborated's words, or the caller's own); empty, and nothing is measured,
    `why_not` said. `wrist_fix`: the wrist's fix, or its (x, y, z). `by`: the run, for its record.
    Returns the Disagreement now in force, or None where nothing was measured (the check off, no scene
    fix or a rough one, a pair nothing establishes, a pair that is no measure: said). `ctx` is told the
    correction in force (ctx.cameras_in_force), for what was placed by the scene camera's numbers
    before it (press_demo.keep_out_counter, counter_after_adopt). Never raises and never ends a run:
    what it cannot do is a line in the log."""
    try:
        return _adopt(ctx, locator, scene_fix, wrist_fix, runner, by, paired, why_not)
    except Exception as e:  # noqa: BLE001 - a check that failed is no reason to stop a run
        log.info("[press_demo] CAMERAS: not checked against each other (%s: %s)" % (type(e).__name__, e))
        return None


def _adopt(ctx, locator, scene_fix, wrist_fix, runner, by, paired, why_not):
    check = getattr(locator, "check", None)
    if check is None or not check.active or scene_fix is None or wrist_fix is None:
        return None
    if getattr(scene_fix, "source", "") == "wrist" or getattr(scene_fix, "rough", False):
        return None  # (no scene camera's fix, or one placed roughly from a box its frame cut: no measure)
    note = getattr(runner, "note", None)
    if not paired:
        why = str(why_not or "the caller does not say what makes the two one container")
        log.info("[press_demo] CAMERAS: the two cameras' places are not measured against each other: %s" % why)
        if note is not None:
            note("camera_check", measured=False, why=why, run=str(by), **check.row())
        return None
    lens = lens_xy(locator)
    if lens is None:
        return None
    w = tuple(float(v) for v in getattr(wrist_fix, "top_xyz", wrist_fix))
    # the scene camera's OWN placement of it: what its look moved it along the line by and the turn it was
    # lifted with, taken out again
    s = raw_xyz(locator, tuple(float(v) for v in scene_fix.top_xyz))
    d, why = measure(lens, s, w, t=check._clock())
    if d is None:
        log.info("[press_demo] CAMERAS: no measure of the scene camera against the wrist's: %s" % why)
        if note is not None:
            note("camera_check", measured=False, why=why, run=str(by), **check.row())
        return None
    check.adopt(d)
    if ctx is not None:
        try:
            ctx.cameras_in_force = in_force(locator)
        except Exception:  # noqa: BLE001 - a context that cannot hold it: the keep-out then drops nothing
            pass
    log.info(
        "[press_demo] CAMERAS: the scene camera has the container at [%.3f, %.3f] and the wrist camera at [%.3f, "
        "%.3f] (one container: %s): they disagree by %s. The scene camera's numbers are turned %+.1f deg about "
        "its lens for the rest of this run" % (s[0], s[1], w[0], w[1], paired, d.words(), d.turn_deg)
    )
    if abs(d.turn_deg) > TURN_WARN_DEG and not check.warned:
        check.warned = True
        log.info(
            "[press_demo] WARNING: the scene camera's mount looks %.0f deg off (it puts the container %.0f mm from "
            "where the wrist camera, which the arm follows, has it): tell the team. Box opening turns its own "
            "reading of it for this run; everything else that uses the scene camera's positions is off by as much"
            % (abs(d.turn_deg), 1000 * d.apart_m)
        )
    if note is not None:
        note(
            "camera_check",
            measured=True,
            paired=str(paired),
            run=str(by),
            scene_xy=[round(s[0], 4), round(s[1], 4)],
            wrist_xy=[round(w[0], 4), round(w[1], 4)],
            lens_xy=[round(lens[0], 4), round(lens[1], 4)],
            **d.row(),
            **{k: v for k, v in check.row().items() if k in ("mode", "by")},
        )
    return d
