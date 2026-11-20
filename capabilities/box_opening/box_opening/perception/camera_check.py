"""The scene camera checked against the wrist camera, each run.

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

The distance between them changes with how far the container stands; the ANGLE about the scene camera's
lens does not. That is a camera turned on its mount, and a shift fitted to one place (scene_refine's
translation) would be wrong at the next. So the disagreement is measured as what one pair gives without
assuming more: the turn about the vertical through the scene lens that brings the scene camera's
direction to the container onto the wrist's, how much nearer or farther along that line the wrist has
it, and how much higher. One pair cannot tell a turn about the vertical from a tilt, and does not try:
the turn is about the vertical (as scene_refine's correction is horizontal), and what a tilt would add
shows in the heights, which are said and not corrected.

What is done with it (the owner's rules: nothing recalibrated, the team's TF and files untouched, the
scene measured each run):

  for the rest of the run   the scene camera's placement as box opening reads it (SceneLocator
                            .base_to_color, and everything lifted through it: its fixes, its sightings
                            in a line-up, the line of sight a selection is tied by, the points a later
                            look or the lid's room map takes) is turned by that angle about the vertical
                            through its lens. The along-line difference is applied to that container's
                            own fix only (own_fix): the scene camera sees the lid from its side, and
                            what that does to its range is the lid's, not the table's. Heights are never
                            moved by it.
  for the next run's start  the measure is kept in box opening's state folder (camera_check.json, the
                            newest GUESS_KEPT under one mount) as a GUESS: the next run's scene numbers
                            start turned by their median, said with its age each time, until that run's
                            own pair replaces it. A guess belongs to the mount it was measured against
                            (the numbers TF or the calibration file gave): another mount, and it is
                            dropped. A run that measures more than GUESS_FAR_DEG from the guess says so
                            and keeps only its own.

Set by BOX_OPENING_CAMERA_CHECK (constants.CAMERA_CHECK_ENV; the node's camera_check): "keep" (the
default), "run" (measured and applied in the run, no guess read or kept), "none" (nothing measured,
nothing turned: the scene camera as its mount has it). Words a YAML file does not read as true or false.

Pure numpy and a small JSON file; the ROS side is the locator's.
"""

import json
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from box_opening import LOGGER as log
from box_opening.constants import CAMERA_CHECK_ENV, state_dir

KEEP, RUN, NONE = "keep", "run", "none"
MODES = (KEEP, RUN, NONE)
FILE_NAME = "camera_check.json"

# A pair is one container seen by both cameras only where the two directions from the scene lens are no
# further apart than this. 2026-11-20 measured 13-15 deg and the camera was found turned ~15 deg once
# before (flatbot, 2026-11-17); at 25 deg the scene camera would have the box 0.3 m off at 0.7 m, which
# is no camera to correct in silence, and two things that far apart are as likely two things.
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
GUESS_KEPT = 5  # the newest measures kept under one mount; the guess is their median
# A run's own measure this far from the guess: the camera has moved since, or one of the two paired two
# different things. Said, and the older measures are dropped (they describe a camera that no longer is).
# The five pairs above spread 1.9 deg.
GUESS_FAR_DEG = 3.0
# The mount a guess was measured against is still the mount where it reads within these: a static TF
# and a calibration file give the same numbers every time, so anything past rounding is another mount.
MOUNT_XYZ_TOL_M = 0.002
MOUNT_TURN_TOL_DEG = 0.2


def mode_of(env=None):
    """KEEP, RUN or NONE by $BOX_OPENING_CAMERA_CHECK: unset or anything else, KEEP (the node refuses a
    value it does not know before a run starts: node.Capability._camera_check)."""
    word = str((os.environ if env is None else env).get(CAMERA_CHECK_ENV, "")).strip().lower()
    if word in (NONE, "off", "0", "false", "no"):
        return NONE
    return RUN if word == RUN else KEEP


def file_path(env=None):
    return state_dir(env) / "scene_calib" / FILE_NAME


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
    table points and the extent they were seen over turned (not moved along: own_fix's note). Heights,
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


# ------------------------------------------------------------------------------------------- the guess
def mount_key(T_base_link):
    """A mount as the guess is keyed on it: T(arm <- the scene camera's link) as TF or the calibration
    file gave it, xyz and the rotation's rows."""
    T = np.asarray(T_base_link, float)
    return [round(float(v), 5) for v in T[:3, 3]] + [round(float(v), 6) for v in T[:3, :3].ravel()]


def same_mount(a, b):
    """Whether two mount_keys are one mount: within MOUNT_XYZ_TOL_M and MOUNT_TURN_TOL_DEG."""
    try:
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.shape != (12,) or b.shape != (12,):
            return False
        if float(np.abs(a[:3] - b[:3]).max()) > MOUNT_XYZ_TOL_M:
            return False
        R = a[3:].reshape(3, 3).T @ b[3:].reshape(3, 3)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))))
        return angle <= MOUNT_TURN_TOL_DEG
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class Guess:
    """What the runs before measured under this mount: the median of the newest `n` (turn, along), and
    when the newest was made."""

    turn_rad: float
    along_m: float
    n: int
    t: float

    def words(self, now=None):
        age = max(0.0, (time.time() if now is None else float(now)) - self.t)
        if age < 5400:
            ago = "%.0f min ago" % (age / 60.0)
        elif age < 2 * 86400:
            ago = "%.0f h ago" % (age / 3600.0)
        else:
            ago = "%.0f days ago" % (age / 86400.0)
        return "%+.1f deg about its lens, %.0f mm %s along it: what %s measured against the wrist camera, %s" % (
            math.degrees(self.turn_rad),
            1000 * abs(self.along_m),
            "nearer" if self.along_m < 0 else "farther",
            "the last run" if self.n == 1 else "the last %d runs" % self.n,
            ago if self.n == 1 else "the newest " + ago,
        )


def _read(path):
    try:
        with open(path) as f:
            doc = json.load(f)
        return doc if isinstance(doc, dict) else None
    except (OSError, ValueError):
        return None


def load_guess(path, key):
    """(Guess, None) from `path` for the mount `key`, or (None, why there is none): no file, a file that
    cannot be read, measures made under another mount (dropped: said, the file left for the next
    measure to replace), or none in it."""
    doc = _read(path)
    if doc is None:
        return None, None
    rows = [r for r in doc.get("measures") or [] if isinstance(r, dict)]
    if not rows:
        return None, None
    if not same_mount(doc.get("mount"), key):
        return None, (
            "the scene camera's mount is not the one the last runs' measure against the wrist camera was made "
            "under (%d kept): dropped, the scene camera as its mount has it" % len(rows)
        )
    try:
        rows = rows[-GUESS_KEPT:]
        turn = float(np.median([float(r["turn_deg"]) for r in rows]))
        along = float(np.median([float(r["along_mm"]) for r in rows]))
        newest = max(float(r.get("t") or 0.0) for r in rows)
    except (KeyError, TypeError, ValueError):
        return None, "the last runs' measure against the wrist camera cannot be read: not used"
    if abs(turn) > TURN_MAX_DEG or abs(along) > 1000 * ALONG_MAX_M:
        return None, "the last runs' measure against the wrist camera is past what a pair may be: not used"
    return Guess(math.radians(turn), along / 1000.0, len(rows), newest), None


def save_measure(path, key, d, fresh=False, by=""):
    """`d` (a Disagreement, the scene camera's own placement against the wrist's) added to the measures
    kept under the mount `key`, the newest GUESS_KEPT of them; `fresh`: the older ones are dropped
    first (they describe a camera that no longer is). A file of another mount is replaced. A file that
    cannot be written is a line in the log, never the run's end."""
    path = Path(path)
    doc = _read(path) or {}
    rows = [r for r in doc.get("measures") or [] if isinstance(r, dict)]
    if fresh or not same_mount(doc.get("mount"), key):
        rows = []
    rows = (rows + [dict(d.row(), by=str(by))])[-GUESS_KEPT:]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"format": 1, "mount": list(key), "measures": rows}, indent=1) + "\n")
        tmp.replace(path)
    except OSError as e:
        log.info("[scene] (the cameras' disagreement was not kept for the next run: %s)" % e)
        return False
    return True


# ------------------------------------------------------------------------------------- what a locator holds
class CameraCheck:
    """The correction in force for one scene locator: `turn_rad` about the vertical through the lens
    (what SceneLocator.base_to_color turns its placement by), `along_m` (own_fix), and where they came
    from, `by`: "" (none), "guess" (the runs before, until this one measures) or "run" (this run's own
    pair). settle() reads the guess once, when the mount is first known; adopt() takes a run's
    measure."""

    def __init__(self, mode=None, path=None, clock=time.time):
        self.mode = mode_of() if mode is None else mode
        self.path = Path(path) if path is not None else file_path()
        self._clock = clock
        self.turn_rad = 0.0
        self.along_m = 0.0
        self.by = ""
        self.guess = None  # the Guess read at the start, kept for the word on how far the run's own is from it
        self.key = None  # the mount it is all keyed on
        self.measured = None  # this run's own Disagreement (the scene camera's own placement against the wrist's)
        self.warned = False
        self._settled = False

    @property
    def active(self):
        return self.mode != NONE

    def settle(self, T_base_link):
        """Once, with the mount as TF or the calibration file gives it: the guess of the runs before,
        where there is one for this mount (mode KEEP), in force and said."""
        if self._settled or T_base_link is None:
            return
        self._settled = True
        self.key = mount_key(T_base_link)
        if self.mode != KEEP:
            return
        guess, why = load_guess(self.path, self.key)
        if why:
            log.info("[scene] %s" % why)
        if guess is None:
            return
        self.guess, self.turn_rad, self.along_m, self.by = guess, guess.turn_rad, guess.along_m, "guess"
        log.info(
            "[scene] the scene camera's numbers start turned %s. A guess, until this run's wrist camera has the "
            "container too" % guess.words(self._clock())
        )

    def adopt(self, d, by=""):
        """This run's own measure `d` (the scene camera's own placement against the wrist's) in force
        from here on, and kept for the next run's start (mode KEEP). Returns how far it is from the guess
        it replaces (deg), or None where there was none."""
        far = None if self.guess is None else abs(d.turn_deg - math.degrees(self.guess.turn_rad))
        self.measured, self.turn_rad, self.along_m, self.by = d, d.turn_rad, d.along_m, "run"
        if self.mode == KEEP and self.key is not None:
            save_measure(self.path, self.key, d, fresh=far is not None and far > GUESS_FAR_DEG, by=by)
        return far

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


def own_fix(locator, fix):
    """The scene camera's fix of the container as the correction in force has the container itself:
    moved along the line from the lens by the along-line difference (CameraCheck.along_m). The turn is
    already in it (the locator lifted it through base_to_color). Only this container's own place is
    moved: the scene camera sees a lid from its side, and what that does to its range says nothing of
    the table round it. Heights as they were. `fix` itself where nothing is in force, the fix is the
    wrist's, or a rough one (its place is its own fit's). SceneLocator.locate returns every fix through
    this."""
    check = getattr(locator, "check", None)
    if fix is None or check is None or not check.active or not check.along_m:
        return fix
    if getattr(fix, "source", "") == "wrist" or getattr(fix, "rough", False):
        return fix
    lens = lens_xy(locator)
    if lens is None:
        return fix
    top = turn_xyz(lens, 0.0, fix.top_xyz, check.along_m)
    pose = fix.pose
    if pose is not None:
        pose = replace(pose, xyz=turn_xyz(lens, 0.0, pose.xyz, check.along_m))
    return replace(fix, top_xyz=top, pose=pose)


def adopt(ctx, locator, scene_fix, wrist_fix, runner=None, by=""):
    """A wrist find was just taken for the container the scene camera has at `scene_fix` (a fix as
    `locator`.locate() returned it, with the correction that is in force NOW: nothing may have changed
    the check between that look and this call): measure the two against each other and put the result
    in force (the module's note).
    `wrist_fix`: the wrist's fix, or its (x, y, z). Returns the Disagreement now in force, or None
    where nothing was measured (the check off, no scene fix or a rough one, a pair that is no measure:
    said). Never raises and never ends a run: what it cannot do is a line in the log."""
    try:
        return _adopt(ctx, locator, scene_fix, wrist_fix, runner, by)
    except Exception as e:  # noqa: BLE001 - a check that failed is no reason to stop a run
        log.info("[press_demo] CAMERAS: not checked against each other (%s: %s)" % (type(e).__name__, e))
        return None


def _adopt(ctx, locator, scene_fix, wrist_fix, runner, by):
    check = getattr(locator, "check", None)
    if check is None or not check.active or scene_fix is None or wrist_fix is None:
        return None
    if getattr(scene_fix, "source", "") == "wrist" or getattr(scene_fix, "rough", False):
        return None  # (no scene camera's fix, or one placed roughly from a box its frame cut: no measure)
    lens = lens_xy(locator)
    if lens is None:
        return None
    w = tuple(float(v) for v in getattr(wrist_fix, "top_xyz", wrist_fix))
    # the scene camera's OWN placement of it: what its look moved it along the line by (own_fix) and the
    # turn it was lifted with, taken out again
    s = turn_xyz(lens, 0.0, tuple(float(v) for v in scene_fix.top_xyz), -check.along_m)
    s = turn_xyz(lens, -check.turn_rad, s)
    d, why = measure(lens, s, w, t=check._clock())
    note = getattr(runner, "note", None)
    if d is None:
        log.info("[press_demo] CAMERAS: no measure of the scene camera against the wrist's: %s" % why)
        if note is not None:
            note("camera_check", measured=False, why=why, **check.row())
        return None
    had, guess = check.by, check.guess
    far = check.adopt(d, by=by)
    log.info(
        "[press_demo] CAMERAS: the scene camera has the container at [%.3f, %.3f] and the wrist camera at [%.3f, "
        "%.3f]: they disagree by %s. The scene camera's numbers are turned %+.1f deg about its lens for the rest "
        "of this run%s"
        % (
            s[0],
            s[1],
            w[0],
            w[1],
            d.words(),
            d.turn_deg,
            ""
            if check.mode != KEEP
            else ", and the next starts from that"
            if check.key is not None
            else " (not kept: the mount it was measured under is not known)",
        )
    )
    if had == "guess" and far is not None:
        log.info(
            "[press_demo] CAMERAS: the guess this run started from (%+.1f deg) was %.1f deg from what it measures%s"
            % (
                math.degrees(guess.turn_rad),
                far,
                ""
                if far <= GUESS_FAR_DEG
                else ": more than %.0f. The camera has moved since, or one of the two paired two different things; "
                "this run's own is used and the older ones are dropped" % GUESS_FAR_DEG,
            )
        )
    if abs(d.turn_deg) > TURN_WARN_DEG and not check.warned:
        check.warned = True
        log.info(
            "[press_demo] WARNING: the scene camera's mount looks %.0f deg off (it puts the container %.0f mm from "
            "where the wrist camera, which the arm follows, has it): tell the team. Box opening turns its own "
            "reading of it; everything else that uses the scene camera's positions is off by as much"
            % (abs(d.turn_deg), 1000 * d.apart_m)
        )
    if note is not None:
        note(
            "camera_check",
            measured=True,
            scene_xy=[round(s[0], 4), round(s[1], 4)],
            wrist_xy=[round(w[0], 4), round(w[1], 4)],
            lens_xy=[round(lens[0], 4), round(lens[1], 4)],
            guess_deg=None if guess is None else round(math.degrees(guess.turn_rad), 2),
            from_guess_deg=None if far is None else round(far, 2),
            **d.row(),
            **{k: v for k, v in check.row().items() if k in ("mode", "by")},
        )
    return d
