"""Box opening's perception as Foxglove shows it, while a run flies.

press_demo runs it on its perception node, on a thread of its own (run,
from press_demo.start_viz's thread). From then on the look
(SceneLocator.locate), the wrist's aim and its looks at the knob after a
press (press_demo.aim_button_at_staging), the compliant press
(depth_press.compliant_stroke, as the reach-out's way in and the full run's
presses fly it) and the lid off (reach_lid_off: the grip's way down and the
lift, the lid's spot and the map it was chosen on) hand it what they
computed anyway, and the console's own lines say what was decided. Under
/box_opening/viz:

    scene/annotated/compressed  the scene frame the look judged: every box each detector gave the gate,
                                numbered, the fix, passed (no fix from it) or refused and why, the fix's
                                lid and holder projected into it, a second place that passed too; the
                                SCENE line under it
    wrist/annotated/compressed  the wrist frame the aim judged — the first aim's, or a knob look's after a
                                press: the button it found, the lid round it, where the scene camera put
                                the button, the hits; the table it read; the AIM or AFTER PRESS line
    markers                     the box fix, its second places, the table the look measured, the aimed
                                button, the stroke, the grip and the lift, the lid's spot and what the
                                cameras saw standing round it, in the planning frame
    status                      the phase and the last decision: the console's own line

markers and status are latched (transient local, depth 1), and so are the
images once drawn: a Foxglove that connects late still shows the last of
each. The capability's docs/foxglove/box_opening_perception.json lays them out beside the
cameras, the segmenter's overlay, TF and the arm (docs/HARDWARE_BRINGUP.md
§4a).

None of it stops the run (owner: not a millisecond slower, never a stop
for it). A run with it off never imports this module (press_demo's
start_viz and viz_hook); with it on, the mission's thread only starts the
viz's thread (~0.3 ms), which imports it (~1 ms: no dataclass here) once
the mission's thread waits and goes on as its worker (run). The mission's
thread only hands references over — frames the grabbers replace whole,
results already made — into one slot per kind, a newer item replacing one
not yet taken: a few microseconds, never a wait, never a wake; the
console's lines go out as they did. Making the publishers, drawing,
encoding and publishing are that one thread's, and as it shares the
mission's GIL it works only while the mission's thread waits on the
driver, a frame or a plan (PERIOD_S, IDLE_RUN_FRAC): never while it aims,
solves or re-times. A mission thread that wakes inside a step no
checkpoint breaks waits for its end: the marker type's import (3-5 ms, up
to 17 with the Jetson loaded; once a run, after a quiet second:
IMPORT_QUIET_S) and a publish (the lid spot's the longest, about 2 ms:
MAP_STEP). An image nobody subscribes to is not drawn or encoded:
the newest is kept, and drawn for the first subscriber. stop(), once the
run is over, has the worker publish what is left (a NO BOX look, the DONE
line) ungated — not when the run unwinds from a Ctrl+C — and waits for its
end, a second Ctrl+C or not. Every entry point is wrapped: a failure is
said once and never reaches the mission. It is off unless
BOX_OPENING_VIZ=1 turns it on (constants.viz_enabled).

Importable with no ROS: rclpy and the messages are imported by the worker
(Viz._making) and the marker builder.
"""

import array
import math
import re
import sys
import textwrap
import threading
import time

import numpy as np

from box_opening import LOGGER as log
from box_opening.constants import viz_enabled as enabled

NS = "/box_opening/viz"
SCENE_TOPIC = NS + "/scene/annotated/compressed"
WRIST_TOPIC = NS + "/wrist/annotated/compressed"
MARKER_TOPIC = NS + "/markers"
STATUS_TOPIC = NS + "/status"
JPEG_QUALITY = 80
MAX_W_PX = 960  # a drawn image wider than this is scaled down to it before encoding (the OAKs' 640 are not)
# The aim judges up to 15 frames a second: one without a hit is handed over
# at most this often; every hit and the fix are handed over as they come
WRIST_PERIOD_S = 0.2
# The worker shares the mission's GIL, so it takes it only while the
# mission's thread waits — on the driver, a frame, a plan — never while it
# computes. It wakes every PERIOD_S, and only then (a hand-over never wakes
# it: every wake is a GIL switch, ~0.1 ms of a busy thread's time), and
# works only when the mission's thread ran under IDLE_RUN_FRAC of the time
# since its last wake (its schedstat): a thread spinning on the arm's state
# through a motion runs a few percent, one aiming or solving ~100 %.
# Measured on the aim's own find_button back to back (2026-10-28): drawing
# whenever an item came slowed it 2-4 % with Foxglove watching, a gate
# polled every 20 ms 0.5 %; this way nothing is drawn until it pauses.
PERIOD_S = 0.25
IDLE_RUN_FRAC = 0.2
# ... and it steps aside this long between the parts of one piece of work,
# so a mission thread that wakes meanwhile takes the GIL at once; there it
# drops the rest of its batch, for the next wait, once that thread has run
# over BUSY_RUN_FRAC of the time since the batch began: work that started
# late in a wait does not run on into the computing that follows it. It is
# not niced: a niced worker the kernel preempts while it holds the GIL only
# keeps a waking mission thread waiting the longer (and niced it bought no
# shorter wait: 2026-10-28)
YIELD_S = 0.0002
BUSY_RUN_FRAC = 0.3
LINES_KEPT = 50  # console lines waiting for the worker, at most
CLOSE_WAIT_S = 0.5  # stop() waits this long at most for the worker's last batch (_run: 2 ms; both images watched 13-27)
# ... and once it stops waiting (that ran out, or a Ctrl+C cut it short),
# this long for the worker to stop at its next step, through any further
# Ctrl+C: the longest step it cannot leave part-way (an encode, a
# publisher; the marker type's import ~27 ms) — a daemon thread still
# inside OpenCV when the interpreter goes takes the process down with it
# ("terminate called without an active exception", the exit code lost)
QUIT_WAIT_S = 0.05
# The marker type's import (_making's first step: /opt/ros ships no .pyc, so
# its modules compile from source, each compile 1-4.5 ms of C holding the
# GIL that no checkpoint breaks into) waits for the mission's thread to have
# run under IMPORT_RUN_FRAC at every wake of the last IMPORT_QUIET_S: a wait
# through the startup's checks or a motion, never a short pause between two
# computes, nor a run of waits with a short compute in each (one 30 ms in
# 0.4 s passes the gate's 20 %)
IMPORT_QUIET_S = 1.0
IMPORT_RUN_FRAC = 0.05
PREFIX = "[press_demo]"
_clock = time.monotonic  # the tests drive it

# What the status mirrors: a press_demo line whose words open in capitals —
# SCENE:, AIM:, REACH-OUT:, ONE MOTION ..., PRESS ..., AFTER PRESS 1:, GRIP:,
# CLOSE, LID GRIPPED, RELEASE, STOP:, NO BOX, DONE — and the few that open
# in lower case and decide something: the detect-only report's, the press
# once more (reach_lid_off's AGAIN), where the lid goes (its SPOT).
# Compiled at its first use (said, on the worker's thread): 0.2 ms of the
# import otherwise
PHASE_RE = r"[A-Z][A-Z-]+(?: [A-Z][A-Z-]+)*"
_LOWER_PHASES = (
    ("no fix", "LOOK"),
    ("detect-only done", "DONE"),
    ("preview ends", "DONE"),
    ("the knob is not up", "AGAIN"),
    ("lid goes to", "SPOT"),
)
# which console lines caption which image: the one it says after the item —
# the wrist's, its AIM line, then the box already open (ALREADY OPEN, after
# the first aim) or, after a press, the knob look's verdict (AFTER PRESS)
CAPTION_PHASE = {"scene": ("SCENE",), "wrist": ("AIM", "ALREADY OPEN", "AFTER PRESS")}
# whose button and lid an aim was given (press_demo.aim_button_at_staging's
# prior_xy and lid_z), by its tag: the scene camera's, but a knob look after
# a press (reach_press._aim's "reach-after") is given the button the press
# was aimed at, its top the press's
SCENE_PRIOR = ("the scene camera's button", "the scene camera's lid")
PRIOR_WORDS = {"reach-after": ("where the press was aimed", "the press's top")}
# what a failure is said of (Viz.failed)
FAILED = {
    "scene": "the annotated scene",
    "wrist": "the annotated wrist frame",
    "press": "the stroke's markers",
    "grip": "the grip's markers",
    "spot": "the lid spot's markers",
    "markers": "the markers",
    "status": "the status",
    "viz": "making the viz's publishers",
}

# drawing (BGR)
GREEN, RED, ORANGE, GREY = (60, 205, 60), (60, 60, 235), (0, 165, 255), (165, 165, 165)
YELLOW, CYAN, WHITE, MAGENTA, DIM = (0, 225, 255), (230, 200, 60), (240, 240, 240), (200, 80, 230), (110, 110, 110)
VERDICT_BGR = {"fix": GREEN, "passed": YELLOW, "two places": ORANGE, "refused": RED, "not judged": GREY}
WON = ("fix", "passed", "two places")  # the verdicts of the box the gate passed: drawn bold, on top
STRIP_BGR = (28, 28, 28)
FONT_SCALE = 0.42
LINE_PX = 16
PAD_PX = 6
WHY_MAX = 120  # a refusal is cut to this many characters in the image
SRC_TAG = {"segmenter": "SEG", "owl": "OWL"}
KIND_WORDS = {
    "band": "the band above its box",
    "grown": "its box grown up (the frame cuts it)",
    "lid": "its box, a lid stacked on the holder's",
    "as_is": "its box",
}

# markers (RGBA)
HOLDER_RGBA = (0.25, 0.45, 0.95, 0.35)  # lid_room_view's box
TABLE_RGBA = (0.55, 0.80, 0.95, 0.18)
RIVAL_RGBA = (1.0, 0.60, 0.10, 0.9)
BUTTON_RGBA = (0.25, 0.90, 0.35, 0.95)
HIT_RGBA = (1.0, 1.0, 1.0, 0.8)
PRIOR_RGBA = (1.0, 0.90, 0.20, 0.9)
APPROACH_RGBA = (1.0, 1.0, 1.0, 0.9)
PRESS_RGBA = (1.0, 0.45, 0.10, 0.95)
SPRING_RGBA = (0.85, 0.15, 0.15, 0.9)
GRIP_RGBA = (0.75, 0.45, 1.0, 0.95)
LIFT_RGBA = (0.30, 0.85, 0.95, 0.9)
TEXT_RGBA = (1.0, 1.0, 1.0, 1.0)
TEXT_M = 0.018  # a label's height (m)
# The spot look's map is drawn as the tops of the squares something stands
# in (clear_table.TableMap.standing), one CUBE_LIST, this many made between
# two checkpoints. Measured 2026-10-28 (this Jetson, niced, cores 0-9) on
# test_reach_lid_off's wall round the box, 1407 of the map's 85 x 85: a
# point 3.8 us, the markers 6.5 ms in 18 steps of at most 0.45 ms; the
# publish, one step, holds the GIL 1.6 ms converting the array to C (1.9 ms
# beside 25 bare markers, about the rest of a full run's array; those 25
# alone 0.84 ms) — once a run, as the carry flies, and under the 3-5 ms the
# marker type's import keeps a waking mission thread waiting
MAP_STEP = 100


def said(line):
    """(phase, text) for a console line the status mirrors, else None: a
    press_demo line that opens with a word in capitals (PHASE_RE), or one of
    the few that open in lower case and decide something (_LOWER_PHASES).
    text is the line without its "[press_demo]"."""
    if not line.startswith(PREFIX):
        return None
    text = line[len(PREFIX) :].strip()
    m = re.match(PHASE_RE, text)  # re keeps the compiled pattern
    if m is not None:
        return m.group(0), text
    for start, phase in _LOWER_PHASES:
        if text.startswith(start):
            return phase, text
    return None


def ascii_text(s):
    """`s` as the image's font can write it (Hershey: ASCII only)."""
    s = str(s).replace("—", "-").replace("–", "-").replace("°", " deg").replace("·", ".")
    return s.encode("ascii", "replace").decode("ascii")


# --- what was judged ---------------------------------------------------------


class _Record:
    """Named fields (__slots__) its class's __init__ sets: what is handed
    over and judged here. Not dataclasses — defining the four was ~3.4 ms
    of this module's import, at every launch (review 2026-10-28)."""

    __slots__ = ()

    def replace(self, **kw):
        """A copy, the fields in `kw` changed (dataclasses.replace's)."""
        new = object.__new__(type(self))
        for name in self.__slots__:
            setattr(new, name, kw.pop(name) if name in kw else getattr(self, name))
        if kw:
            raise TypeError("%s has no field %s" % (type(self).__name__, ", ".join(kw)))
        return new

    def __repr__(self):
        return "%s(%s)" % (type(self).__name__, ", ".join("%s=%r" % (n, getattr(self, n)) for n in self.__slots__))


class Cand(_Record):
    """One box the look's gate judged (judged_cands): its detector
    ("segmenter" | "owl"), score, the box the gate lifted (x0, y0, x1, y1),
    the detector's own box (the holder under a band, else the box itself),
    its kind (segmenter_source.SceneCand.kind; "" for the OWL's), the
    detector's class where it says one, the verdict ("fix" | "passed" |
    "two places" | "refused" | "not judged") and why."""

    __slots__ = ("source", "score", "box", "det_box", "kind", "cls", "verdict", "why")

    def __init__(self, source, score, box, det_box, kind, cls, verdict, why):
        self.source, self.score, self.box, self.det_box = source, score, box, det_box
        self.kind, self.cls, self.verdict, self.why = kind, cls, verdict, why

    @property
    def tag(self):
        """'SEG 0.88 FIX': what the image writes at the box."""
        word = {"fix": "FIX", "passed": "PASSED", "two places": "2 PLACES", "refused": "X", "not judged": "-"}[
            self.verdict
        ]
        return "%s %.2f %s" % (SRC_TAG.get(self.source, self.source), self.score, word)

    def line(self):
        """What the strip under the image says of it."""
        what = " ".join(w for w in (SRC_TAG.get(self.source, self.source), self.cls, "%.2f" % self.score) if w)
        if self.kind in KIND_WORDS and self.kind != "as_is":
            what += ", %s" % KIND_WORDS[self.kind]
        return "%s: %s - %s" % (what, self.verdict.upper(), self.why)


def judged_cands(judged, boxes=None, fix=None):
    """[Cand] for every candidate of the last message each detector gave
    SceneLocator.locate's gate (its last_judged: source -> (candidates,
    SceneBoxFix, frame, T_base_color)), the segmenter's first: the box the
    gate passed — "fix" where the look's fix (`fix`, the SceneFix locate
    returned) was voted from that detector's sightings, "passed" where it
    was not (a pass is one sighting: no fix came of it, or the other
    detector's did), "two places" when another passed too, RIVAL_M away —
    the ones it refused, each with the gate's own last reason for it (the
    SceneBoxFix.tried that SceneBoxFix.tried_boxes gives this very box: a
    score says too little, two boxes 0.314 and 0.311 both read 0.31 in a
    reason; none where the two do not pair one to one), and the ones after
    the winner it never had to judge. `boxes`, the locator's last_boxes,
    names the segmenter's classes."""
    from box_opening.perception.scene_source import SRC_LABEL, box_key

    voted = set() if fix is None else {getattr(fix, "source", None), *(getattr(fix, "sources", ()) or ())}
    out = []
    for source in sorted(judged, key=lambda s: s != "segmenter"):
        cands, got = judged[source][0], judged[source][1]
        label = SRC_LABEL.get(source, source)
        won = None if got.pose is None else got.box  # the candidate's own box: matched by identity, as the reasons
        lifted = (
            None
            if won is None
            else "%s lift, %d lid points, its top at [%.3f, %.3f, %.3f]" % (got.lift, got.n, *got.top)
        )
        classes = {}
        if source == "segmenter":
            classes = {round(float(r[1]), 6): str(r[0]) for r in (boxes or {}).get(source, ()) if len(r) == 6}
        # each reason beside its box; reasons that do not pair with boxes
        # one to one tell no box anything — never another box's reason
        tried_boxes = getattr(got, "tried_boxes", ())
        paired = len(got.tried) == len(tried_boxes)
        reasons = tuple(zip(got.tried, tried_boxes, strict=False)) if paired else ()
        for c in cands:
            score, box = float(c[0]), box_key(c[1])
            det = getattr(c, "det_box", None)
            tail = " (%s %.2f)" % (label, score)
            whys = [t[: -len(tail)] if t.endswith(tail) else t for t, b in reasons if b is c[1]]
            passed = won is not None and c[1] is won
            if passed and got.rivals:
                d = min(math.hypot(r[0] - got.top[0], r[1] - got.top[1]) for r in got.rivals)
                verdict = "two places"
                why = "it passed, and so did another %.0f mm from it: the OWL settles which" % (1000 * d)
            elif passed and source in voted:
                verdict, why = "fix", lifted
            elif passed:
                verdict = "passed"
                why = "it passed (%s), but %s" % (
                    lifted,
                    "no fix came of it: see the caption"
                    if fix is None
                    else "the fix is the %s's" % SRC_LABEL.get(fix.source, fix.source),
                )
            elif whys:
                verdict, why = "refused", whys[-1]
            elif paired:
                verdict, why = "not judged", "a box ahead of it passed"
            else:
                verdict, why = "not judged", "the gate's reasons do not say which box each was"
            if len(why) > WHY_MAX:
                why = why[: WHY_MAX - 3] + "..."
            out.append(
                Cand(
                    source,
                    score,
                    box,
                    box if det is None else box_key(det),
                    str(getattr(c, "kind", "") or ""),
                    classes.get(round(score, 6), ""),
                    verdict,
                    why,
                )
            )
    return out


def scene_caption(look):
    """The scene image's caption until the console's SCENE line says it."""
    fix = look.fix
    if fix is None:
        return "NO BOX - %s" % (look.why or "the scene camera found no box")
    from box_opening.tasks.press_demo import scene_source_label

    p = fix.pose.xyz
    return "SCENE: box at [%.3f, %.3f], its lid's top at %.3f, the table at %.3f (%s)%s" % (
        p[0],
        p[1],
        fix.top_xyz[2],
        fix.table_z_scene,
        scene_source_label(fix),
        "; %s" % fix.partial if getattr(fix, "partial", "") else "",
    )


# --- projecting and drawing --------------------------------------------------


def project(pts, R, t, k, dist=None):
    """The pixels (N, 2) of base-frame points `pts` in a camera whose
    optical frame stands at rotation `R` and origin `t` in the base frame,
    with intrinsics `k` and distortion `dist` as its grabber keeps them;
    NaN for a point behind the camera."""
    import cv2

    pc = (np.asarray(pts, float).reshape(-1, 3) - np.asarray(t, float).ravel()) @ np.asarray(R, float)
    K = np.asarray(k, float)
    d = None if dist is None else np.asarray(dist, float).ravel()
    if d is not None and d.size and np.any(d != 0):
        uv = cv2.projectPoints(pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, d)[0].reshape(-1, 2)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = np.c_[K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]]
    uv = np.asarray(uv, float)
    uv[pc[:, 2] <= 0.01] = np.nan
    return uv


def ring(xy, r, z, n=48):
    """n points (n, 3) round a horizontal circle of radius r about xy at height z."""
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.c_[float(xy[0]) + r * np.cos(a), float(xy[1]) + r * np.sin(a), np.full(n, float(z))]


def outline(model, xy, z, yaw=0.0):
    """The container's outline at height z about xy: the lid's circle (a
    round model), else its footprint's rectangle turned `yaw`."""
    if model.round:
        return ring(xy, 0.5 * float(model.lid_dims[0]), z)
    hx, hy = 0.5 * float(model.dims[0]), 0.5 * float(model.dims[1])
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [float(xy[0]) + c * u - s * v, float(xy[1]) + s * u + c * v, float(z)]
            for u, v in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))
        ]
    )


class Busy(Exception):
    """The mission's thread computes again: the worker's batch waits (Viz._check)."""


_checkpoint = None  # set by the worker for its batch (Viz.step, Viz._make); raises Busy


def _yield():
    """Inside a batch (Viz.step, Viz._make): the worker's checkpoint — it
    steps aside for the mission's thread, should that want the GIL now, and
    stops here (Busy) should that compute again, or stop() have it quit
    (Viz._check, Viz._check_last). Outside one, nothing."""
    if _checkpoint is not None:
        _checkpoint()


def _px(p):
    return int(round(float(np.clip(p[0], -1e5, 1e5)))), int(round(float(np.clip(p[1], -1e5, 1e5))))


def _poly(img, uv, bgr, thick=1, closed=True):
    """The finite runs of the pixel path `uv` drawn on `img`."""
    import cv2

    pts = np.clip(np.asarray(uv, float), -1e5, 1e5)
    ok = np.all(np.isfinite(pts), axis=1)
    if closed and ok.all() and len(pts) > 2:
        cv2.polylines(img, [np.round(pts).astype(np.int32)], True, bgr, thick, cv2.LINE_AA)
        return
    run = []
    for p, good in zip(pts, ok, strict=False):
        if good:
            run.append(p)
            continue
        if len(run) > 1:
            cv2.polylines(img, [np.round(run).astype(np.int32)], False, bgr, thick, cv2.LINE_AA)
        run = []
    if len(run) > 1:
        cv2.polylines(img, [np.round(run).astype(np.int32)], False, bgr, thick, cv2.LINE_AA)


def _cross(img, uv, bgr, size=14, thick=2):
    import cv2

    if np.all(np.isfinite(uv)):
        cv2.drawMarker(img, _px(uv), bgr, cv2.MARKER_CROSS, size, thick, cv2.LINE_AA)


def _label(img, text, org, bgr, taken=None):
    """`text` with its baseline's left end at `org`, on a dark box, kept
    inside the image; moved up (then down) off the boxes of the labels in
    `taken`, to which it adds its own."""
    import cv2

    text = ascii_text(text)
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1)
    x = min(max(int(org[0]), 0), max(img.shape[1] - w - 2, 0))
    y0 = min(max(int(org[1]), h + 2), img.shape[0] - base - 1)
    step = h + base + 3
    for y in [y0 - i * step for i in range(4)] + [y0 + i * step for i in range(1, 4)]:
        y = min(max(y, h + 2), img.shape[0] - base - 1)
        box = (x - 1, y - h - 2, x + w + 1, y + base)
        if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in taken or ()):
            break
    if taken is not None:
        taken.append(box)
    cv2.rectangle(img, box[:2], box[2:], STRIP_BGR, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, bgr, 1, cv2.LINE_AA)


def _strip(img, lines):
    """`img` with `lines` — text, or (text, bgr) — written under it on a
    dark strip of its width, each wrapped to it."""
    import cv2

    w = img.shape[1]
    sample = "the scene camera 0123"
    sample_w = cv2.getTextSize(sample, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1)[0][0]
    per = max(20, int((w - 2 * PAD_PX) * len(sample) / sample_w))  # characters to a row
    rows = []
    for ln in lines:
        text, bgr = ln if isinstance(ln, tuple) else (ln, WHITE)
        rows += [(r, bgr) for r in (textwrap.wrap(ascii_text(text), per) or [""])]
    strip = np.full((PAD_PX + LINE_PX * len(rows), w, 3), STRIP_BGR, np.uint8)
    for i, (r, bgr) in enumerate(rows):
        cv2.putText(
            strip, r, (PAD_PX, LINE_PX * (i + 1) - 2), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, bgr, 1, cv2.LINE_AA
        )
    return np.vstack([img, strip])


def scene_image(look, model):
    """The look (SceneLook) as a BGR image, or None without a frame: the
    frame its gate judged with every candidate box numbered — green the fix,
    yellow passed (no fix from it), orange two places, red refused, grey not
    judged, the detector's own box under a band thin grey — the fix's lid
    (its outline at its top, its centre a cross) and holder (its outline on
    the table) projected into it,
    the table it measured cyan, each of two places an orange cross; under
    it the SCENE line and every candidate's verdict."""
    import cv2

    from box_opening.perception.scene_source import TABLE_RING_M

    order = [s for s in (getattr(look.fix, "source", None), "segmenter", "owl") if s in look.judged]
    color, T = (look.judged[order[0]][2], look.judged[order[0]][3]) if order else (look.color, None)
    if color is None:
        return None
    img = np.ascontiguousarray(color).copy()
    cands = judged_cands(look.judged, look.boxes, look.fix)
    _yield()
    cam = None if T is None or look.k is None else (np.asarray(T, float)[:3, :3], np.asarray(T, float)[:3, 3])

    def drawn(pts):
        return project(pts, cam[0], cam[1], look.k, look.dist)

    fix = look.fix
    if fix is not None and cam is not None and math.isfinite(fix.table_z_scene):
        tz, ext = float(fix.table_z_scene), fix.table_extent
        if ext is not None:
            x0, x1, y0, y1 = ext
            table = np.array([[x0, y0, tz], [x1, y0, tz], [x1, y1, tz], [x0, y1, tz]])
            table = np.concatenate(
                [
                    np.linspace(a, b, 12, endpoint=False)  # its edges, in pieces the lens bends
                    for a, b in zip(table, np.roll(table, -1, 0), strict=False)
                ]
            )
        else:
            table = ring(fix.top_xyz, TABLE_RING_M[1], tz)
        _poly(img, drawn(table), CYAN, 1)
        _yield()
    taken = []  # the labels first, the boxes over them: no box is hidden under another's label
    for i, c in enumerate(cands, 1):
        _label(img, "%d %s" % (i, c.tag), (c.box[0], c.box[1] - 4), VERDICT_BGR[c.verdict], taken)
    for c in cands:
        if c.det_box != c.box:
            cv2.rectangle(img, _px(c.det_box[:2]), _px(c.det_box[2:]), DIM, 1, cv2.LINE_AA)
    for c in sorted(cands, key=lambda c: c.verdict in WON):  # the winner's on top
        cv2.rectangle(
            img, _px(c.box[:2]), _px(c.box[2:]), VERDICT_BGR[c.verdict], 2 if c.verdict in WON else 1, cv2.LINE_AA
        )
    _yield()
    if fix is not None and cam is not None:
        top, yaw = fix.top_xyz, float(fix.pose.yaw)
        _poly(img, drawn(outline(model, top, top[2], yaw)), GREEN, 2)  # the lid's top
        _poly(img, drawn(outline(model, top, fix.pose.xyz[2], yaw)), GREEN, 1)  # the holder on the table
        _cross(img, drawn([top])[0], GREEN)
        _yield()
    if cam is not None:
        for _cands, got, *_rest in look.judged.values():
            if got.pose is not None and got.rivals:
                for p in (got.top, *got.rivals):
                    uv = drawn([p])[0]
                    _cross(img, uv, ORANGE, 18)
                    if np.all(np.isfinite(uv)):
                        _label(img, "two places", (uv[0] + 8, uv[1] - 8), ORANGE)
    _yield()
    lines = [look.caption or scene_caption(look)]
    if not cands:
        lines.append(("no detector box was judged in this look", GREY))
    lines += [("%d %s" % (i, c.line()), VERDICT_BGR[c.verdict]) for i, c in enumerate(cands, 1)]
    lines.append(
        (
            "green the fix, yellow passed (no fix from it), red refused, orange two places, grey not judged; "
            "thin grey the detector's own box, cyan the table the fix stands on",
            DIM,
        )
    )
    return _strip(img, lines)


def _table_words(s):
    """What the newest hit read of the table the wrist sees (button_aim's
    table offset), or ""."""
    raw = getattr(s, "table_offset_raw_m", float("nan"))
    if not math.isfinite(raw):
        return ""
    taken = math.isfinite(getattr(s, "table_offset_m", float("nan")))
    return "table: the wrist reads its table %+.0f mm from the scene's (%s)" % (
        1000 * raw,
        "taken out" if taken else "not taken out",
    )


def wrist_image(tick, model):
    """The aim's tick (AimTick) as a BGR image, or None without a frame: the
    frame the aimer judged, and the hit it made in that frame — the
    button's circle green, the press point (its centre) a red cross, the lid
    round it (magenta, at its ring's measured height) — then, through that
    frame's own pose, the button the aim was given (a yellow cross at the
    lid height it was given: the scene camera's, or after a press where the
    press was aimed, prior_said) and every hit's press point so far (white
    dots; the fix's agreeing ones green). A frame without a hit draws no
    button: an older hit's is another frame's. Under it the AIM line (then
    ALREADY OPEN, or after a press the AFTER PRESS one: CAPTION_PHASE), this
    frame's verdict, the lid and the table its hit read."""
    import cv2

    if tick.color is None:
        return None
    img = np.ascontiguousarray(tick.color).copy()
    s = tick.sighting
    cam = tick.cam if tick.cam is not None and tick.k is not None else None

    def drawn(pts):
        return project(pts, cam[0], cam[1], tick.k, tick.dist)

    agree = []
    if tick.done is not None:
        agree = list(tick.done.fix_hits() or [])
    _yield()
    if cam is not None and tick.prior_xy is not None and tick.lid_z is not None:
        uv = drawn([(tick.prior_xy[0], tick.prior_xy[1], tick.lid_z)])[0]
        if np.all(np.isfinite(uv)):
            cv2.drawMarker(img, _px(uv), YELLOW, cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)
            _label(img, tick.prior_said[0], (uv[0] + 10, uv[1] + 16), YELLOW)
    if cam is not None and tick.points:
        for i, uv in enumerate(drawn(tick.points)):
            if np.all(np.isfinite(uv)):
                cv2.circle(img, _px(uv), 4 if i in agree else 2, GREEN if i in agree else WHITE, -1, cv2.LINE_AA)
    if s is not None:
        if cam is not None:
            lid_z = s.lid_z_m  # the ring the oblique aim read; straight down, the disc less its height above it
            if not math.isfinite(lid_z):
                lid_z = s.xyz[2] - (s.above_lid_m if math.isfinite(s.above_lid_m) else 0.0)
            off = np.asarray(model.button_offset, float)[:2]  # a round lid's button is at its centre: no yaw needed
            _poly(img, drawn(outline(model, np.asarray(s.xyz[:2]) - off, lid_z)), MAGENTA, 1)  # the lid
            _poly(img, drawn(ring(s.xyz[:2], 0.5 * float(model.button_diameter_m), s.xyz[2])), GREEN, 2)
        else:
            cv2.circle(img, _px(s.uv), int(round(s.r_px)), GREEN, 2, cv2.LINE_AA)
        _cross(img, s.uv, RED, 22)
    _yield()
    lines = [
        tick.caption
        or "AIM: %d hit(s) in %d frames, %.1f s; a fix takes %d that agree"
        % (tick.hits, tick.frames or 0, tick.t_aim, tick.need)
    ]
    if tick.hit and s is not None:
        up = "%+.1f mm" % (1000 * s.above_lid_m) if math.isfinite(s.above_lid_m) else "height unknown"
        tilted = ""
        if s.oblique:
            tilted = ", seen %.0f deg off its normal" % s.incidence_deg
            if math.isfinite(s.disc_fill):
                tilted += ", depth on %.0f%% of it" % (100 * s.disc_fill)
        lines.append(
            ("this frame: the button, %.0f mm across, %s above its lid%s" % (1000 * s.diameter_m, up, tilted), GREEN)
        )
    elif tick.done is None:
        lines.append(("this frame: %s" % (tick.why or "no button"), RED))
    if s is not None:
        lid = ("the lid ring at z %.3f" % s.lid_z_m) if math.isfinite(s.lid_z_m) else ""
        if lid and tick.lid_z is not None:
            lid += ", %s %.3f (%+.0f mm)" % (tick.prior_said[1], tick.lid_z, 1000 * (s.lid_z_m - tick.lid_z))
        lines += [(w, MAGENTA if w is lid else CYAN) for w in (lid, _table_words(s)) if w]
    if tick.done is not None:
        xyz, measured = tick.done.fix(), tick.done.fix(measured=True)
        if xyz is not None:
            lines.append(
                (
                    "FIX: %d of %d hits agree - the press point [%.3f, %.3f, %.3f] (the wrist measured z %.3f)"
                    % (len(agree), tick.hits, *xyz, measured[2]),
                    GREEN,
                )
            )
    return _strip(img, lines)


def jpeg(img):
    """`img` as JPEG bytes (a uint8 array), scaled down to MAX_W_PX wide first."""
    import cv2

    h, w = img.shape[:2]
    if w > MAX_W_PX:
        img = cv2.resize(img, (MAX_W_PX, max(1, int(round(h * MAX_W_PX / w)))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("the JPEG encoder refused a %dx%d image" % (w, h))
    return buf


# --- markers -----------------------------------------------------------------


def _at(m, xyz):
    m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in xyz)
    return m


def markers(shown, model, frame, stamp):
    """What the markers topic shows, as a MarkerArray in `frame`. From the
    last look (shown["scene"], a SceneLook): its fix's holder and lid at the
    fix (lid_room_view's colours), the lid's top, the table it measured
    (where the scene saw it, else the ring it was read from), the two places
    of a message that passed in two. From the aim (shown["aim"], an
    AimTick): the button at its press point (the fix's, else the newest
    hit's), each hit, where the scene camera put the button. From the press
    (shown["press"], a PressPlan): its stroke — the pads at its start, the
    approach to TOUCH_SLOW_M over the button's top, the press to its
    bottom, on to where the springs' reference ends (a bent stroke's aside,
    bent_end). From the grip (shown["grip"], a GripPlan): the closed pads'
    way down from where it begins (their x, y by the arm's own kinematics,
    _pads_at: with no chain it is not drawn), the approach to GRIP_SLOW_M
    over where it ends, down to it, and the lift straight up lift_off_m.
    From the lid's spot (shown["spot"], a SpotPick): the tops of what the
    cameras saw standing on the table (the map it was chosen on), the lid
    where it goes down, its tier and why. Each part labelled.
    Its first marker deletes the last array's."""
    from geometry_msgs.msg import Point
    from visualization_msgs.msg import Marker

    from box_opening.perception.scene_source import TABLE_RING_M
    from box_opening.tasks.lid_room_view import LID_RGBA, SHORT_RGB, MarkerList, yawed

    ml = MarkerList(frame, stamp)

    def add(*a):
        _yield()  # a marker is ~0.1 ms of Python: the mission's thread may take the GIL between any two
        return ml.add(*a)

    lid_h = float(model.lid_dims[2])
    shape = Marker.CYLINDER if model.round else Marker.CUBE

    def text(ns, xyz, words):
        _at(add(ns, Marker.TEXT_VIEW_FACING, TEXT_RGBA, (0.0, 0.0, TEXT_M)), xyz).text = ascii_text(words)

    look = shown.get("scene")
    fix = None if look is None else look.fix
    if fix is not None:
        top, base, yaw = fix.top_xyz, fix.pose.xyz, float(fix.pose.yaw)
        tall = max(float(top[2]) - float(base[2]) - lid_h, 0.005)
        holder = _at(
            add("box: holder", shape, HOLDER_RGBA, (model.dims[0], model.dims[1], tall)),
            (top[0], top[1], float(base[2]) + 0.5 * tall),
        )
        lid = _at(
            add("box: lid", shape, LID_RGBA, (model.lid_dims[0], model.lid_dims[1], lid_h)),
            (top[0], top[1], float(top[2]) - 0.5 * lid_h),
        )
        if not model.round:
            yawed(holder, yaw)
            yawed(lid, yaw)
        _at(add("box: lid top", Marker.SPHERE, (0.2, 0.9, 0.3, 1.0), (0.012, 0.012, 0.012)), top)
        from box_opening.tasks.press_demo import scene_source_label

        text(
            "box: label",
            (top[0], top[1], float(top[2]) + 0.06),
            "SCENE: %s: box at [%.3f, %.3f], lid top %.3f" % (scene_source_label(fix), base[0], base[1], top[2]),
        )
        tz = float(fix.table_z_scene)
        if math.isfinite(tz):
            ext = fix.table_extent
            if ext is not None:
                x0, x1, y0, y1 = ext
                _at(
                    add("table (measured)", Marker.CUBE, TABLE_RGBA, (x1 - x0, y1 - y0, 0.002)),
                    (0.5 * (x0 + x1), 0.5 * (y0 + y1), tz),
                )
            else:
                r = TABLE_RING_M[1]
                _at(add("table (measured)", Marker.CYLINDER, TABLE_RGBA, (2 * r, 2 * r, 0.002)), (top[0], top[1], tz))
            text("table (measured)", (float(base[0]) + 0.12, float(base[1]), tz + 0.01), "table %.3f" % tz)
    if look is not None:
        for _cands, got, *_rest in look.judged.values():
            if got.pose is not None and got.rivals:
                m = add("two places", Marker.SPHERE_LIST, RIVAL_RGBA, (0.03, 0.03, 0.03))
                m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in (got.top, *got.rivals)]
                text("two places", (got.top[0], got.top[1], float(got.top[2]) + 0.05), "two places pass as the box")
    aim = shown.get("aim")
    if aim is not None:
        got = None if aim.done is None else aim.done.fix()
        s = aim.sighting
        at = got if got is not None else (None if s is None else s.xyz)
        if at is not None:
            d = float(model.button_diameter_m)
            _at(add("aim: button", Marker.CYLINDER, BUTTON_RGBA, (d, d, 0.003)), at)
            text(
                "aim: button",
                (at[0], at[1], float(at[2]) + 0.035),
                "%s [%.3f, %.3f, %.3f]" % ("AIMED" if got is not None else "aim: a hit", *at),
            )
        if aim.points:
            m = add("aim: hits", Marker.SPHERE_LIST, HIT_RGBA, (0.006, 0.006, 0.006))
            m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in aim.points]
        if aim.prior_xy is not None and aim.lid_z is not None:
            _at(
                add("aim: " + aim.prior_said[0], Marker.SPHERE, PRIOR_RGBA, (0.01, 0.01, 0.01)),
                (aim.prior_xy[0], aim.prior_xy[1], aim.lid_z),
            )
    press = shown.get("press")
    if press is not None:
        from box_opening.tasks.depth_press import TOUCH_SLOW_M

        st = press.stroke
        x, y, z0 = (float(v) for v in st.pad0)
        top = float(press.top_z)
        bottom = top - float(press.travel_m)
        slow = min(top + TOUCH_SLOW_M, z0)
        _at(add("press: stroke start", Marker.SPHERE, APPROACH_RGBA, (0.01, 0.01, 0.01)), (x, y, z0))
        # straight down to the bottom; the springs' reference on from there
        # straight down too, or a bent stroke's to where its joints end
        # (bent_end), else not drawn at all — straight down it would not be
        end = bent_end(st) if st.bent else (x, y, float(st.end_z))
        for ns, rgba, a, b in (
            ("press: approach", APPROACH_RGBA, (x, y, z0), (x, y, slow)),
            ("press: down to the bottom", PRESS_RGBA, (x, y, slow), (x, y, bottom)),
            ("press: the springs' reference", SPRING_RGBA, (x, y, bottom), end),
        ):
            if b is not None and math.dist(a, b) > 1e-4:
                m = add(ns, Marker.LINE_STRIP, rgba, (0.003, 0.0, 0.0))
                m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in (a, b)]
        _at(add("press: button bottom", Marker.SPHERE, PRESS_RGBA, (0.008, 0.008, 0.008)), (x, y, bottom))
        aside = ""
        if st.bent:
            aside = ", %.0f mm aside%s" % (1000 * st.bend_m, "" if end is not None else " (bent: its end not drawn)")
        text(
            "press: label",
            (x + 0.05, y, top + 0.02),
            "PRESS: %.0f mm down to the button's bottom %.3f, the springs' "
            "reference on to %.3f%s; ~%.0f N (%.0f N/m)"
            % (1000 * (z0 - bottom), bottom, float(st.end_z), aside, st.k * st.over, st.k),
        )
    grip = shown.get("grip")
    start = None if grip is None else _pads_at(grip.stroke.q0)  # where the closed pads begin their way down
    if start is not None:
        from box_opening.tasks.reach_lid_off import GRIP_SLOW_M, lift_off_m

        x, y, z0 = start
        end = float(grip.stroke.pad_z)
        slow = min(end + GRIP_SLOW_M, z0)
        up = end + lift_off_m(model)  # straight up with the knob held (reach_lid_off.plan_lift), then the clear
        for ns, rgba, a, b in (
            ("grip: approach", APPROACH_RGBA, z0, slow),
            ("grip: down to the knob", GRIP_RGBA, slow, end),
            ("grip: the lift", LIFT_RGBA, end, up),
        ):
            if abs(a - b) > 1e-4:
                m = add(ns, Marker.LINE_STRIP, rgba, (0.003, 0.0, 0.0))
                m.points = [Point(x=x, y=y, z=float(z)) for z in (a, b)]
        _at(add("grip: the closed pads", Marker.SPHERE, GRIP_RGBA, (0.008, 0.008, 0.008)), (x, y, end))
        text(
            "grip: label",
            (x + 0.05, y, up + 0.02),
            "GRIP: the closed pads down to %.3f, %.0f mm under the lid at "
            "%.3f; then %.0f mm straight up, and up and back toward the arm"
            % (end, 1000 * (grip.lid_z - end), grip.lid_z, 1000 * (up - end)),
        )
    spot = shown.get("spot")
    if spot is not None:
        tz, sx, sy = float(spot.table_z), float(spot.drop.xyz[0]), float(spot.drop.xyz[1])
        cells = () if spot.tmap is None else spot.tmap.standing()
        if len(cells):
            c = float(spot.tmap.cell)
            m = add(
                "lid spot: what the cameras saw standing",
                Marker.CUBE_LIST,
                (*SHORT_RGB, 0.6),
                (0.9 * c, 0.9 * c, 0.004),
            )
            pts = []
            for i in range(0, len(cells), MAP_STEP):
                _yield()  # up to ~1400 squares on a crowded table: made a step at a time, as the markers are
                pts += [Point(x=float(px), y=float(py), z=tz + float(h)) for px, py, h in cells[i : i + MAP_STEP]]
            m.points = pts
        lid = _at(
            add("lid spot", shape, LID_RGBA, (model.lid_dims[0], model.lid_dims[1], lid_h)), (sx, sy, tz + 0.5 * lid_h)
        )
        if not model.round:
            yawed(lid, float(spot.drop.yaw))
        text(
            "lid spot: label",
            (sx, sy, tz + lid_h + 0.05),
            "LID SPOT (%s): [%.2f, %.2f] - %s" % (spot.tier, sx, sy, spot.says),
        )
    return ml.array


def _pads_at(q):
    """The pads (depth_press.DOWN_CONTACT_TOOL) in the world with the arm at
    joints `q`, by the arm's own kinematics, or None: no q, or no chain —
    the run's is built by the time a stroke or a grip is planned
    (approach._chain()), and none is built here."""
    from box_opening.runtime import approach

    chain = approach._CHAIN
    if q is None or chain is None:
        return None
    from box_opening.tasks.depth_press import _pads

    return tuple(float(v) for v in _pads(chain, q))


def bent_end(stroke):
    """Where a bent stroke's springs' reference ends (depth_press.StrokePlan:
    the pads at its last joints, q_end), or None: no q_end, or no chain
    (_pads_at)."""
    return _pads_at(getattr(stroke, "q_end", None))


# --- what the mission hands over ----------------------------------------------


class SceneLook(_Record):
    """One look, as scene() hands it over: when (t, _clock), its SceneFix
    (fix) or why there is none (SceneLocator.last_why), what the gate
    judged (last_judged), the segmenter's raw boxes, their classes there
    (last_boxes), the grabber's newest frame (drawn on when no message was
    judged), its intrinsics; the console's SCENE line once it is said
    (caption), and whether it is drawn as it stands."""

    __slots__ = ("t", "fix", "why", "judged", "boxes", "color", "k", "dist", "caption", "drawn")

    def __init__(self, t, fix, why, judged, boxes, color, k, dist, caption=None, drawn=False):
        self.t, self.fix, self.why, self.judged, self.boxes = t, fix, why, judged, boxes
        self.color, self.k, self.dist, self.caption, self.drawn = color, k, dist, caption, drawn


class AimTick(_Record):
    """One tick of the aim, as aim() hands it over: when (t), the frame the
    aimer judged (its colour, intrinsics) and its camera's pose (cam: R, t,
    or None), the hit it made in that very frame (sighting: a
    ButtonSighting, or None — an older hit's pixel is not this frame's),
    whether this tick hit, else why its frame was refused (last_why), the
    hits, frames and agreeing hits a fix takes (need), the time since the
    aimer was made (t_aim, s), the lid height and the button it was given
    (or None) and whose they are (prior_said: SCENE_PRIOR, or PRIOR_WORDS by
    the aim's tag), every hit's press point so far, the aimer once it made
    its fix (done: it changes no more); the console's AIM line once it is
    said, then its verdict's (caption: CAPTION_PHASE), and whether it is
    drawn as it stands."""

    __slots__ = (
        "t",
        "color",
        "k",
        "dist",
        "cam",
        "sighting",
        "hit",
        "why",
        "hits",
        "frames",
        "need",
        "t_aim",
        "lid_z",
        "prior_xy",
        "points",
        "done",
        "prior_said",
        "caption",
        "drawn",
    )

    def __init__(
        self,
        t,
        color,
        k,
        dist,
        cam,
        sighting,
        hit,
        why,
        hits,
        frames,
        need,
        t_aim,
        lid_z,
        prior_xy,
        points,
        done=None,
        prior_said=SCENE_PRIOR,
        caption=None,
        drawn=False,
    ):
        self.t, self.color, self.k, self.dist, self.cam, self.sighting = t, color, k, dist, cam, sighting
        self.hit, self.why, self.hits, self.frames, self.need, self.t_aim = hit, why, hits, frames, need, t_aim
        self.lid_z, self.prior_xy, self.points, self.done = lid_z, prior_xy, points, done
        self.prior_said, self.caption, self.drawn = prior_said, caption, drawn


class PressPlan(_Record):
    """The compliant stroke, as press() hands it over: when (t), the
    depth_press.StrokePlan, the button's top it presses (top_z) and its
    bottom below that (travel_m, cfg.button_travel_m)."""

    __slots__ = ("t", "stroke", "top_z", "travel_m")

    def __init__(self, t, stroke, top_z, travel_m):
        self.t, self.stroke, self.top_z, self.travel_m = t, stroke, top_z, travel_m


class GripPlan(_Record):
    """The grip's way down, as grip() hands it over: when (t), the
    reach_lid_off.GripDescent (the closed pads' reference from the hand
    turned down at its q0 straight down to its pad_z) and the lid's height
    it goes by (lid_z)."""

    __slots__ = ("t", "stroke", "lid_z")

    def __init__(self, t, stroke, lid_z):
        self.t, self.stroke, self.lid_z = t, stroke, lid_z


class SpotPick(_Record):
    """Where the lid goes, as spot() hands it over: when (t), the table map
    it was chosen on (a clear_table.TableMap, or None: no camera showed the
    table), the spot (a ContainerPose on the table), its tier and what it
    says (reach_lid_off.spot_candidates), and the table it goes down on
    (table_z, the map's heights above it)."""

    __slots__ = ("t", "tmap", "drop", "tier", "says", "table_z")

    def __init__(self, t, tmap, drop, tier, says, table_z):
        self.t, self.tmap, self.drop, self.tier, self.says, self.table_z = t, tmap, drop, tier, says, table_z


class _Mirror:
    """sys.stdout as it was, every whole press_demo line written to it also
    handed to `hand` (Viz.said_line). The console comes first: what is
    written goes on as it would have, and the mirror never raises."""

    def __init__(self, out, hand):
        self._out, self._hand, self._part = out, hand, ""
        self._lock = threading.Lock()

    def write(self, s):
        n = self._out.write(s)
        try:
            with self._lock:
                *lines, part = (self._part + s).split("\n")
                self._part = part[-2000:]
            for ln in lines:
                if ln.startswith(PREFIX):
                    self._hand(ln)
        except Exception:  # noqa: BLE001 -- the console must never fail for the viz
            pass
        return n

    def flush(self):
        return self._out.flush()

    def __getattr__(self, name):
        return getattr(self._out, name)


class Viz:
    """The slots the mission's thread hands items into, and the one worker
    that makes the publishers (_making: a step at a time, as it draws) and
    draws and publishes the items (step) — on a thread of its own
    (`thread`), else on the one that calls _run (run), or none (the tests
    call step). `ran`: how long (ns) the mission's thread has run, the gate
    (press_demo.thread_runtime; None: no gate); `pubs`: the tests'
    publishers."""

    def __init__(self, node, model, frame, pubs=None, thread=True, ran=None):
        self.node, self.model, self.frame = node, model, frame
        self.pubs = pubs
        self._ran = ran
        self._window = None  # (its run time, when) at the worker's last wake (_idle)
        self._quiet_since = None  # ... since when every wake found it under IMPORT_RUN_FRAC (_idle, _quiet)
        self._begun = None  # ... and when the worker's batch began (_check)
        self._markers_due = False  # what the markers show changed since they were published
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._slots = {}  # kind -> the newest item not yet taken: a newer one replaces it
        self._lines = []  # (when, line): what the console said since the worker last took them
        self._held = {}  # "scene" / "wrist" -> its newest item, drawn, or kept for a first subscriber
        self._shown = {}  # what the markers show: "scene", "aim", "press", "grip", "spot"
        self._failed = set()  # what has failed, said once each
        self._aim_seen = (None, 0, 0.0)  # (the aimer, its hits, when) last handed over: aim()'s throttle
        self._out = self._mirror = None
        self.closed = False
        self.quit = False  # stop() no longer waits: the worker stops at its next step (_check, _check_last)
        self.ctrl_c = False  # a Ctrl+C came into close()'s wait (stop says so)
        self.done = threading.Event()  # set as _run ends: what close() waits on (_wait)
        self.thread = None
        if thread:
            self.thread = threading.Thread(target=self._run, name="viz", daemon=True)
            self.thread.start()

    # -- the mission's side: hand over, never wait, never wake the worker -
    def put(self, kind, item):
        with self._lock:
            self._slots[kind] = item

    def said_line(self, line):
        with self._lock:
            self._lines.append((_clock(), line))
            del self._lines[:-LINES_KEPT]

    def mirror(self):
        """The console's lines to the status from here on (close undoes it)."""
        self._out = sys.stdout
        self._mirror = sys.stdout = _Mirror(self._out, self.said_line)

    def aim_tick(self, aimer, fixed, frames_before, tag=None):
        frames = getattr(aimer, "frames", None)
        if not fixed and frames == frames_before:
            return  # it took no new frame
        hits, now = int(getattr(aimer, "hits", 0)), _clock()
        last, last_hits, last_t = self._aim_seen
        hit = hits != (last_hits if last is aimer else 0)
        if not (fixed or hit) and last is aimer and now - last_t < WRIST_PERIOD_S:
            return
        self._aim_seen = (aimer, hits, now)
        # the frame the aimer judged (its snapshot, by reference) and that
        # frame's pose (ButtonAimer.last_frame), not the grabber's newest;
        # its hit only when it hit in it: an older hit's pixel is another frame's
        f, cam = getattr(aimer, "last_frame", None) or (None, None)
        self.put(
            "wrist",
            AimTick(
                t=now,
                color=getattr(f, "color", None),
                k=getattr(f, "k", None),
                dist=getattr(f, "dist", None),
                cam=cam,
                sighting=aimer.last_sighting if hit or fixed else None,
                hit=hit,
                why=None if hit else aimer.last_why,
                hits=hits,
                frames=frames,
                need=aimer._need(),
                t_aim=now - aimer.t0,
                lid_z=aimer.lid_z,
                prior_xy=aimer.prior_xy,
                points=list(aimer.points),
                done=aimer if fixed else None,
                prior_said=PRIOR_WORDS.get(tag, SCENE_PRIOR),
            ),
        )

    def failed(self, what, e):
        """Say once that `what` (a FAILED key) failed; never raises."""
        if self.closed or what in self._failed:
            return
        self._failed.add(what)
        try:
            log.info(
                "[viz] (%s failed: %s — said once; the run goes on)"
                % (FAILED.get(what, what), str(e) or type(e).__name__)
            )
        except Exception:  # noqa: BLE001
            pass

    def close(self, last=True):
        """The console as it was, then the worker stopped: `last`, after its
        last batch (_run), CLOSE_WAIT_S at most; not `last` (the run unwinds
        from a Ctrl+C), at its next step, nothing more published. Once
        close() stops waiting — that ran out, or a second Ctrl+C came into
        it — the worker quits at its next step all the same, and close()
        gives it QUIT_WAIT_S to, whatever Ctrl+C comes meanwhile: it is
        never left inside OpenCV for the interpreter's teardown."""
        self.closed = True
        self.quit = self.quit or not last
        self._wake.set()
        if self._mirror is not None and sys.stdout is self._mirror:
            sys.stdout = self._out
        th = self.thread
        if th is None or th is threading.current_thread():
            return
        if not self.quit:
            self._wait(CLOSE_WAIT_S, through=False)
        self.quit = True
        self._wait(QUIT_WAIT_S, through=True)

    def _wait(self, s, through):
        """Up to `s` for the worker's end (done). A Ctrl+C meanwhile
        (KeyboardInterrupt) is kept (ctrl_c) and has the worker quit; it
        ends the wait, but `through` it waits on to its end. Not
        Thread.join: on this robot's Python (3.10.12) a join that a
        KeyboardInterrupt cuts short marks the live thread stopped
        (bpo-45274's handler), and every later join returns at once."""
        until = time.monotonic() + s
        while True:
            try:
                left = until - time.monotonic()
                if left <= 0 or self.done.wait(left):
                    return
            except KeyboardInterrupt:
                self.ctrl_c = self.quit = True
                if not through:
                    return

    # -- the worker's side ------------------------------------------------
    def _making(self):
        """Making the publishers, a step at a time: each yield is a
        checkpoint (_make), as drawing has. The steps, each held whole
        (measured 2026-10-28, press_demo loaded, niced): visualization_msgs'
        import — /opt/ros ships no .pyc, so its twelve modules compile from
        source each run, ~27 ms of the GIL, each compile ~1-4.5 ms of C no
        switch interval breaks into: begun only once the mission's thread
        has been all but still for IMPORT_QUIET_S (_quiet) — then the marker
        builder's module, the markers' C side (~1.7 ms), and each publisher
        (0.4-1.1 ms). Then self.pubs."""
        from visualization_msgs.msg import MarkerArray

        yield
        import importlib

        importlib.import_module("box_opening.tasks.lid_room_view")  # the marker builder's (markers): loaded here
        yield
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from rclpy.qos_event import PublisherEventCallbacks
        from rclpy.type_support import check_for_type_support
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import String

        check_for_type_support(MarkerArray)  # its C side, and Marker's: else in its create_publisher
        yield
        keep = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        pubs = {}
        for kind, msg_type, topic in (
            ("scene", CompressedImage, SCENE_TOPIC),
            ("wrist", CompressedImage, WRIST_TOPIC),
            ("markers", MarkerArray, MARKER_TOPIC),
            ("status", String, STATUS_TOPIC),
        ):
            # no QoS event handlers: rclpy's default (an incompatible-QoS
            # warning) is a waitable on the node, which the perception
            # executor would wait on at every wake of the run — four of
            # them cost its thread ~70 us a wake, ~1-2 % of find_button
            # beside it (measured 2026-10-28; with none the node's waitables
            # stay as they were)
            pubs[kind] = self.node.create_publisher(
                msg_type, topic, keep, event_callbacks=PublisherEventCallbacks(use_default_callbacks=False)
            )
            yield
        self.pubs = pubs

    def _make(self, making):
        """The next steps of `making` (_making), the mission's thread
        waiting (_run): a checkpoint between each, until that thread
        computes again (Busy) — the rest wait for its next wait, as a
        batch's do (step). False once a step failed (said): no viz."""
        global _checkpoint

        self._begin()
        _checkpoint = self._check
        try:
            for _step in making:
                _yield()
        except Busy:
            pass
        except Exception as e:  # noqa: BLE001 -- what is handed over waits in its slots, unread
            self.failed("viz", e)
            return False
        finally:
            _checkpoint = None
        return True

    def _run(self):
        """The worker, until close(): done set as it ends, however it ends."""
        try:
            self._work()
        finally:
            self.done.set()

    def _work(self):
        self._idle()  # the first window starts now
        making = None
        while not self.closed:
            self._wake.wait(PERIOD_S)  # set by close() alone
            if self.closed or not self._idle():
                continue
            if self.pubs is None:
                if making is None:  # its first step, the marker type's import, waits the longer (IMPORT_QUIET_S)
                    if not self._quiet():
                        continue
                    making = self._making()
                if not self._make(making):
                    return
            else:
                self.step()
            self._idle()  # the next window starts after this work
        # stop(): the run is over (press_demo's finally) and its thread waits
        # in close(), so what it handed over last and the worker has not yet
        # taken — the NO BOX look, the last detect-only look, the DONE line —
        # goes out now, ungated. Not without publishers: made now, they would
        # go with the node before a Foxglove found them. Not when the run
        # unwinds from a Ctrl+C (quit): nothing waits for a picture then
        if self.pubs is not None and not self.quit:
            try:
                self.step(final=True)
            except Exception:  # noqa: BLE001 -- the run is over: nothing left to say it to
                pass

    def _idle(self):
        """Did the mission's thread wait since the last call — run under
        IDLE_RUN_FRAC of the time between? True where its run time cannot
        be read (no gate); False on the first call (no window yet). Since
        when it ran under IMPORT_RUN_FRAC at every call is kept (_quiet)."""
        ran = self._ran
        if ran is None:
            return True
        try:
            now = (ran(), _clock())
        except Exception:  # noqa: BLE001 -- no reading, no gate
            return True
        last, self._window = self._window, now
        if last is None:
            return False
        ran, dt = now[0] - last[0], (now[1] - last[1]) * 1e9
        if ran >= IMPORT_RUN_FRAC * dt:
            self._quiet_since = None
        elif self._quiet_since is None:
            self._quiet_since = last[1]
        return ran < IDLE_RUN_FRAC * dt

    def _quiet(self):
        """Has the mission's thread run under IMPORT_RUN_FRAC at every wake
        (_idle) of the last IMPORT_QUIET_S? Always, with no gate."""
        return self._ran is None or (self._quiet_since is not None and _clock() - self._quiet_since >= IMPORT_QUIET_S)

    def _guard(self, what, fn, *a):
        try:
            fn(*a)
        except Busy:
            raise
        except Exception as e:  # noqa: BLE001 -- nothing drawn may reach the run
            self.failed(what, e)

    def step(self, final=False):
        """One batch, the mission's thread waiting (_run): whatever was
        handed over since the last — the newest of each kind — then the
        markers and the images kept for a first subscriber, until the
        mission's thread computes again (Busy): what is left waits for its
        next wait. `final`: the run is over — all of it, never stepping
        aside, unless stop() stops waiting for it (_check_last)."""
        global _checkpoint

        with self._lock:
            items, self._slots = self._slots, {}
            lines, self._lines = self._lines, []
        if not items and not lines and not self._markers_due and all(item.drawn for item in self._held.values()):
            return
        if "scene" in items:
            look = items["scene"]
            self._shown = {"scene": look}  # a new look: the aim, the press and the lid off before it are gone
            self._held["scene"] = look
            self._markers_due = True
        if "wrist" in items:
            tick = items["wrist"]
            self._held["wrist"] = tick
            if tick.hit or tick.done is not None:
                self._shown["aim"], self._markers_due = tick, True
        for kind in ("press", "grip", "spot"):
            if kind in items:
                self._shown[kind], self._markers_due = items[kind], True
        if lines:  # after the items: a line said after one of this batch captions it
            self._guard("status", self._said, lines)
        self._begin()
        _checkpoint = self._check_last if final else self._check
        try:
            if self._markers_due:
                _yield()
                self._guard("markers", self._publish_markers)
                self._markers_due = False
            for kind in ("scene", "wrist"):
                _yield()
                self._guard(kind, self._draw, kind)
        except Busy:
            pass
        finally:
            _checkpoint = None

    def _begin(self):
        try:
            self._begun = None if self._ran is None else (self._ran(), _clock())
        except Exception:  # noqa: BLE001 -- no reading: no checkpoint
            self._begun = None

    def _check(self):
        """A batch's checkpoint (_yield): YIELD_S aside, so a mission thread
        that wakes meanwhile takes the GIL at once, then Busy once stop()
        closed the viz (its last batch does it all) or has the worker quit,
        or once that thread has run over BUSY_RUN_FRAC of the time since
        this batch began (over a millisecond of it)."""
        time.sleep(YIELD_S)
        if self.quit or self.closed:
            raise Busy()
        if self._begun is None:
            return
        try:
            ran, t = self._ran(), _clock()
        except Exception:  # noqa: BLE001
            return
        if t - self._begun[1] > 0.001 and ran - self._begun[0] > BUSY_RUN_FRAC * (t - self._begun[1]) * 1e9:
            raise Busy()

    def _check_last(self):
        """The last batch's checkpoint: no stepping aside, the run being
        over; Busy once stop() no longer waits for it."""
        if self.quit:
            raise Busy()

    def _said(self, lines):
        """The newest line the status mirrors (said) published; a SCENE,
        AIM, ALREADY OPEN or AFTER PRESS line said after the image it
        belongs to (CAPTION_PHASE) made that image's caption, to be drawn
        again."""
        from std_msgs.msg import String

        last = None
        for t, line in lines:
            got = said(line)
            if got is None:
                continue
            phase, last = got
            for kind, caption_phases in CAPTION_PHASE.items():
                item = self._held.get(kind)
                if phase in caption_phases and item is not None and item.t <= t:
                    item.caption, item.drawn = last, False
        if last is not None:
            self.pubs["status"].publish(String(data=last))

    def _stamp(self):
        if self.node is None:
            from builtin_interfaces.msg import Time

            return Time()
        return self.node.get_clock().now().to_msg()

    def _publish_markers(self):
        self.pubs["markers"].publish(markers(self._shown, self.model, self.frame, self._stamp()))

    def _wanted(self, kind):
        """Is the kept image of `kind` undrawn, and does someone subscribe to it?"""
        item = self._held.get(kind)
        return item is not None and not item.drawn and self.pubs[kind].get_subscription_count() > 0

    def _draw(self, kind):
        """The kept image of `kind` drawn and published, once, when someone
        subscribes to it; otherwise left kept, undrawn."""
        from sensor_msgs.msg import CompressedImage

        if not self._wanted(kind):
            return
        item = self._held[kind]
        item.drawn = True  # one try per caption: a frame that fails to draw is not tried every period
        try:
            img = scene_image(item, self.model) if kind == "scene" else wrist_image(item, self.model)
            if img is not None:
                _yield()  # the encode is its longest call: a checkpoint before it
        except Busy:
            item.drawn = False  # interrupted, not failed: the next wait draws it
            raise
        if img is None:
            return
        msg = CompressedImage()
        msg.header.stamp = self._stamp()
        msg.format = "jpeg"
        msg.data = array.array("B", jpeg(img).tobytes())  # an array.array: a bytes value is checked byte by byte
        self.pubs[kind].publish(msg)


# --- the entry points the mission calls: each a no-op without a viz, none raises

_VIZ = None  # the run's viz, set last (press_demo.viz_hook reads it: never a module still loading)
_LOCK = threading.Lock()  # run() setting _VIZ and stop() taking it: one at a time


def _new(node, model, frame, env, ran, thread):
    """A Viz on `node` mirroring the console's lines from here on, or None:
    turned off ($BOX_OPENING_VIZ), or no node to publish on (said)."""
    if not enabled(env):
        return None
    v = None
    try:
        if not callable(getattr(node, "create_publisher", None)):
            raise TypeError("%s publishes nothing" % type(node).__name__)
        v = Viz(node, model, frame, thread=thread, ran=ran)
        v.mirror()
    except Exception as e:  # noqa: BLE001 -- no viz is no reason to stop a run
        if v is not None:
            v.close()
        log.info("[viz] (no Foxglove view of this run: %s)" % (str(e) or type(e).__name__))
        return None
    return v


def start(node, model, frame, env=None, ran=None):
    """A viz on `node` for `model`, its markers in the planning frame
    `frame`, its worker on a thread of its own, gated by `ran` (the
    mission's thread's run time; None: no gate) — or None (_new).
    press_demo's run is run's instead, on a thread press_demo starts."""
    global _VIZ
    stop()
    v = _new(node, model, frame, env, ran, thread=True)
    _VIZ = v
    return v


def run(node, model, frame, ran, stopped, env=None):
    """press_demo's viz thread from here on (press_demo._run_viz): the
    run's viz on `node` (its perception node: publishing needs no spin) for
    `model`, its markers in the planning frame `frame`, and its worker run
    on this very thread until stop(), gated by `ran` (how long, ns, the
    mission's thread has run). Nothing, should press_demo have stopped it
    first (`stopped`, an Event: stop_viz sets it before it calls stop)."""
    global _VIZ
    with _LOCK:
        if stopped.is_set():
            return
        v = _new(node, model, frame, env, ran, thread=False)
        if v is None:
            return
        v.thread = threading.current_thread()
        _VIZ = v
    v._run()


def stop(last=True):
    """The console as it was before start, the worker stopped (Viz.close):
    `last`, after it published what is left; not `last` — the run unwinds
    from a Ctrl+C — at once. Never raises, and never changes how the run
    ends: a second Ctrl+C here cuts the wait short (said), the worker quits
    at its next step and is waited for, and press_demo's finally goes on to
    the perception executor's shutdown."""
    global _VIZ
    with _LOCK:
        v, _VIZ = _VIZ, None
    if v is None:
        return
    try:
        v.close(last)
    except (Exception, KeyboardInterrupt) as e:  # noqa: BLE001 -- a Ctrl+C between close()'s own waits
        try:
            v.ctrl_c = v.ctrl_c or isinstance(e, KeyboardInterrupt)
            v.close(False)  # the worker quits at its next step, and is waited for
        except (Exception, KeyboardInterrupt):  # noqa: BLE001
            pass
    if v.ctrl_c:
        try:
            log.info("[viz] (a Ctrl+C: what was left is not published)")
        except (Exception, KeyboardInterrupt):  # noqa: BLE001
            pass


def scene(locator, fix):
    """The look SceneLocator.locate just made (`fix`, or None): its frame,
    every candidate and the gate's verdicts (locator.last_judged), its fix."""
    v = _VIZ
    if v is None:
        return
    try:
        g = locator.grab
        v.put(
            "scene",
            SceneLook(
                t=_clock(),
                fix=fix,
                why=None if fix is not None else locator.last_why,
                judged=dict(getattr(locator, "last_judged", None) or {}),
                boxes=dict(getattr(locator, "last_boxes", None) or {}),
                color=getattr(g, "color", None),
                k=getattr(g, "k", None),
                dist=getattr(g, "dist", None),
            ),
        )
    except Exception as e:  # noqa: BLE001
        v.failed("scene", e)


def aim(aimer, fixed, frames_before=None, tag=None):
    """The tick of `aimer` (a ButtonAimer) aim_button_at_staging just ran
    (`fixed`: it made the fix; `frames_before`: aimer.frames before it;
    `tag`: its own, which says whose prior it was given, PRIOR_WORDS):
    handed over when it took a new frame — every hit and the fix, other
    frames at most every WRIST_PERIOD_S — with the frame it judged."""
    v = _VIZ
    if v is None:
        return
    try:
        v.aim_tick(aimer, fixed, frames_before, tag)
    except Exception as e:  # noqa: BLE001
        v.failed("wrist", e)


def press(stroke, top_z, cfg):
    """The compliant stroke (depth_press.compliant_stroke's) about to fly
    onto the button whose top is at `top_z`."""
    v = _VIZ
    if v is None:
        return
    try:
        v.put("press", PressPlan(_clock(), stroke, float(top_z), float(cfg.button_travel_m)))
    except Exception as e:  # noqa: BLE001
        v.failed("press", e)


def grip(stroke, lid_z):
    """The grip's way down (reach_lid_off.grip_descent's) about to fly onto
    the knob of the lid at `lid_z`, the fingers opening as it does."""
    v = _VIZ
    if v is None:
        return
    try:
        v.put("grip", GripPlan(_clock(), stroke, float(lid_z)))
    except Exception as e:  # noqa: BLE001
        v.failed("grip", e)


def spot(tmap, drop, tier, says, table_z):
    """Where the lid goes (reach_lid_off.choose_and_carry's spot `drop`, of
    `tier`, saying `says`), chosen on the map `tmap` with the lid lifted:
    the carry there about to fly, down onto the table at `table_z`."""
    v = _VIZ
    if v is None:
        return
    try:
        v.put("spot", SpotPick(_clock(), tmap, drop, str(tier), str(says), float(table_z)))
    except Exception as e:  # noqa: BLE001
        v.failed("spot", e)
