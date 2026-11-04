"""The scene camera's box from sheppy's always-on scene segmenter.

sheppy's scene_segmenter (rammp-perception's RF-DETR-Seg S node) watches
the scene camera at 5 Hz and publishes every processed frame's boxes on
/perception/object_detections, whoever reads them. Box opening reads them
instead of inferring a model of its own: the scene OWL took 1.3-1.4 s an
attempt and missed the box outright when its pale lid scored under the
0.12 floor (flatbot 2026-10-27 16:22 and 16:24, "the scene OWL saw no box
in 15.0 s"); RF-DETR found it in 37/37 detbench frames at its own 0.25,
top-1 in all of them, with no other "food container" anywhere, in 36 ms.

RF-DETR boxes the canister's blue HOLDER as a COCO bowl ("food
container"), and the gate refuses a holder box as it is: its top reads
36-55 mm below lid height. The lid stands ON the holder, so the band above
the holder box is where the lid is (lid_band) — the object's own geometry,
not a place in the room. A box RF-DETR draws on the lid itself, a second
"bowl" stacked on the holder, goes to the gate as it is (stacked_on); a
holder box the frame cuts at its side is grown upward instead, so the
gate's partial lift still finishes it from the lid's arc in view.

The container is the one the user selected: given its instance_id (the
orchestrator's selection, perception's tracked id such as food_container_2),
only that object's boxes are read, with a box stacked on it (its lid), never
another container's on the table. Without one (a standalone run), every box
of the container's classes is a candidate.

The id is how it is found at the first look, not after (review round 2):
perception drops a track it has not seen for track_buffer_frames (10, ~2 s
at flatbot's 5 Hz), and the box it acquires again gets a new id, as when
the arm hides the box through a press. From the first fix on, the rung
follows the selected container by where it was boxed (SegmenterRung.follow,
which every fix moves on): a frame without its id takes the box of its
classes that overlaps that one most, FOLLOW_IOU or more (the scene camera
does not move), and says which id it took (SegFrame.followed).

Every candidate still goes through box opening's own depth gate
(scene_source.scene_fix_from_boxes): a box only ever says WHERE to look.

Importable with no ROS at all: rclpy and the interfaces are imported
inside SegmenterRung.
"""

import math
import time
from typing import NamedTuple

from box_opening.perception.scene_source import frame_edges

SEGMENTER_TOPIC = "/perception/object_detections"
SEG_SILENT_S = 1.0  # no message for this long (since subscribing, or since the last) = silent
# ... but no publisher on the graph at all is "absent" at once: a profile that
# does not select the segmenter is told so at once, not after a second's wait.
# The graph is asked at most this often, and only while no message has come.
SEG_GRAPH_S = 0.5
SEG_FRESH_S = 1.0  # newest segmenter frame this much older than the grabber's newest colour frame = stale
SEG_TOP_K = 3  # the best boxes of one message the gate tries
# the lid band, in holder-box widths/heights: this fraction of its width each
# side of its centre, from this fraction of its height above its top to this
# fraction below it. Any band above the holder works (8 settings, width
# 0.6-1.0 and 0.35-0.6 of the height above it, all 34/37 on detbench).
BAND_HALF_W, BAND_ABOVE, BAND_BELOW = 0.36, 0.45, 0.10
CUT_GROW = 0.5  # a holder box cut at the frame's side: grown up by this much of its height
STACK_LO, STACK_HI = 0.15, 0.5  # a stacked lid's bottom edge lies in [top - LO*h, top + HI*h] of the box under it
STATE_VISIBLE = 0  # rammp_perception_interfaces/DetectedObject.STATE_VISIBLE
# a frame without the selected id takes, from the first fix on, the box that overlaps where the
# selected container was boxed at the last fix most, this much or more (intersection over union):
# its own box again under a new id. A lid boxed on top of it overlaps it far less
FOLLOW_IOU = 0.3

_LATEST = object()  # state()/describe(): read the rung's newest frame


def stamp_ns(stamp):
    """A ROS stamp (sec, nanosec) as integer nanoseconds; None for None."""
    if stamp is None:
        return None
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class SegFrame(NamedTuple):
    """One segmenter message, as the rung keeps it (the 0.5 MB label image is not kept)."""

    recv_s: float  # time.monotonic() when it arrived
    stamp_ns: object  # its camera frame's stamp (int ns), or None
    frame_id: str  # its camera frame's frame_id
    wh: tuple  # its label image's (width, height); (0, 0) when it sent no masks
    objects: tuple  # ((perception_class, score, (x0, y0, x1, y1)), ...) of the rung's classes
    n_objects: int  # every object it published, of any class
    selected: object = None  # with an instance_id to follow: whether that object was in it (else None)
    mine: object = None  # ... and its own box (x0, y0, x1, y1), as read: what the rung follows from a fix
    followed: object = None  # the id it took by where the selected one was (its own id not in it), else None


class SceneCand(NamedTuple):
    """A segmenter candidate for the gate. Fields 0-2 read as the gate's
    (score, box[, holder])."""

    score: float
    box: tuple  # what the gate lifts: the lid band, the grown cut box, or the box as it is
    holder: object  # the uncut holder box that arms the gate's looks-past refusal, or None
    det_box: tuple  # the detector's own box, clamped to the frame
    kind: str  # "lid" (stacked, as is) | "band" (above an uncut holder) | "grown" (a cut holder) | "as_is"


def clamp_box(b, wh):
    """The pixel box [x0, y0, x1, y1] clamped to a w x h image, as floats."""
    w, h = float(wh[0]), float(wh[1])
    return (
        min(max(float(b[0]), 0.0), w),
        min(max(float(b[1]), 0.0), h),
        min(max(float(b[2]), 0.0), w),
        min(max(float(b[3]), 0.0), h),
    )


def stacked_on(b, others):
    """Does box `b` stand on one of `others`? RF-DETR boxes the lid as a
    second bowl on top of the holder's (flatbot 2026-10-22 18:32): its
    centre over the other box, its top above the other's, its bottom edge
    near the other's top, and smaller than it."""
    x0, y0, x1, y1 = (float(v) for v in b[:4])
    cx = 0.5 * (x0 + x1)
    for o in others:
        if o is b:
            continue
        ox0, oy0, ox1, oy1 = (float(v) for v in o[:4])
        oh = oy1 - oy0
        over = ox0 <= cx <= ox1 and y0 < oy0  # its centre over the other box, its top above the other's
        if over and oy0 - STACK_LO * oh <= y1 <= oy0 + STACK_HI * oh and (y1 - y0) < oh:
            return True
    return False


def lid_band(box, image_wh):
    """(band, anchored): where the lid stands on a holder box. Cut by the
    frame at its left or right side (the gate's own 3 px rule): the holder
    box grown upward, the cut edge kept so the gate's partial lift runs on
    it (anchored False — the holder lies inside it). Otherwise the band
    above the holder's top (anchored True: the gate refuses a band that
    looks past its holder)."""
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    w, h = x1 - x0, y1 - y0
    edges = frame_edges((x0, y0, x1, y1), image_wh)
    if "left" in edges or "right" in edges:
        return clamp_box((x0, y0 - CUT_GROW * h, x1, y1), image_wh), False
    cx = 0.5 * (x0 + x1)
    return clamp_box(
        (cx - BAND_HALF_W * w, y0 - BAND_ABOVE * h, cx + BAND_HALF_W * w, y0 + BAND_BELOW * h), image_wh
    ), True


def segmenter_candidates(objects, classes, image_wh, box_kind="holder", k=SEG_TOP_K):
    """[SceneCand] for the gate, best first, from one message's objects
    [(perception_class, score, (x0, y0, x1, y1))]: the k best of `classes`,
    clamped to the frame; with box_kind "as_is" every one as it is (the box
    is the container itself), else every stacked lid first as it is, then
    every other box as its lid band. A raw holder box is never offered: the
    gate reads its top 36-55 mm below lid height, and offering raw boxes
    beside the bands made a fix 105 mm off (detbench 163029)."""
    if box_kind not in ("holder", "as_is"):
        raise ValueError("box_kind must be 'holder' or 'as_is', not %r" % (box_kind,))
    keep = sorted(((float(s), clamp_box(b, image_wh)) for c, s, b in objects if c in classes), key=lambda t: -t[0])[:k]
    if box_kind == "as_is":
        return [SceneCand(s, b, None, b, "as_is") for s, b in keep]
    boxes = [b for _s, b in keep]
    stacked = [stacked_on(b, boxes[:i] + boxes[i + 1 :]) for i, b in enumerate(boxes)]
    lids = [SceneCand(s, b, None, b, "lid") for (s, b), on in zip(keep, stacked, strict=False) if on]
    rest = []
    for (s, b), on in zip(keep, stacked, strict=False):
        if on:
            continue
        band, anchored = lid_band(b, image_wh)
        rest.append(SceneCand(s, band, b if anchored else None, b, "band" if anchored else "grown"))
    return lids + rest


def box_iou(a, b):
    """Intersection over union of two pixel boxes (x0, y0, x1, y1)."""
    ix = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
    iy = max(0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])))
    inter = ix * iy
    union = (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1])) + (float(b[2]) - float(b[0])) * (
        float(b[3]) - float(b[1])
    )
    return inter / union if union > 0.0 else 0.0


def frame_of(msg, classes, now_s, instance_id=None, near=None):
    """A rammp_perception_interfaces/ObjectDetections message as a SegFrame:
    only the VISIBLE objects of `classes` with a finite, non-empty box, a
    score in [0, 1] and (when it sent masks) a box inside its label image
    to within a pixel. With `instance_id`, of those only the object of that
    id and a box stacked on one of its boxes (stacked_on: the lid, boxed as
    a second bowl on its holder). With `near` too (where it was boxed at the
    last fix: SegmenterRung.follow), a frame without that id takes the box
    that overlaps `near` most, FOLLOW_IOU or more, as the selected one."""
    li = getattr(msg, "label_image", None)
    wh = (int(li.width), int(li.height)) if li is not None else (0, 0)
    kept = []
    n = 0
    for o in msg.objects:
        n += 1
        if int(o.state) != STATE_VISIBLE or o.perception_class not in classes:
            continue
        box = tuple(float(v) for v in o.box)
        if len(box) != 4 or not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
            continue
        score = float(o.score)
        if not 0.0 <= score <= 1.0:  # NaN fails this too
            continue
        if wh != (0, 0) and (box[0] < -1.0 or box[1] < -1.0 or box[2] > wh[0] + 1.0 or box[3] > wh[1] + 1.0):
            continue
        kept.append((str(o.perception_class), score, box, str(getattr(o, "instance_id", ""))))
    selected = mine_box = followed = None
    if instance_id:
        mine = [k for k in kept if k[3] == instance_id]
        if not mine and near is not None and kept:
            best = max(kept, key=lambda k: box_iou(k[2], near))
            if box_iou(best[2], near) >= FOLLOW_IOU:
                mine, followed = [best], best[3]
        selected = bool(mine)
        mine_box = mine[0][2] if mine else None
        lids = [k for k in kept if k not in mine and stacked_on(k[2], [m[2] for m in mine])]
        kept = mine + lids
    objects = tuple(k[:3] for k in kept)
    return SegFrame(
        float(now_s), stamp_ns(msg.header.stamp), str(msg.header.frame_id), wh, objects, n, selected, mine_box, followed
    )


def _import_types():
    from rammp_perception_interfaces.msg import DetectedObject, ObjectDetections

    return ObjectDetections, DetectedObject


def _has_field(cls, name):
    fields = getattr(cls, "get_fields_and_field_types", None)
    if callable(fields):
        return name in fields()
    return hasattr(cls, name)


class SegmenterRung:
    """The scene segmenter's newest frame, as box opening's scene locator
    reads it. Subscribes once and keeps only the newest message (state()
    says whether it may be used: from the scene camera, at its size, fresh).
    `instance_id`: the selected container's id, whose boxes alone are read
    (frame_of); None or "" reads every box of `classes`. `available` False,
    with `why`, when this install cannot read the messages at all."""

    def __init__(self, node, classes, import_types=None, instance_id=None):
        self.node = node
        self.classes = tuple(classes)
        self.instance_id = instance_id or None
        self.near = None  # where the selected container was boxed at the last fix (follow); None before one
        self.latest = None  # SegFrame, replaced whole by the callback (one attribute store)
        self.count = 0
        self.available = False
        self.why = None
        self.cb_error = None  # the last message the callback could not read, as repr: it keeps the frame before
        self._graph = (None, None)  # (when, publisher count) of the last graph query (publishers)
        self.t_subscribed = time.monotonic()
        try:
            ObjectDetections, DetectedObject = (import_types or _import_types)()
        except ImportError as e:
            self.why = (
                "rammp_perception_interfaces does not import (%s): this install does not chain " "~/ros2_ws/install" % e
            )
            return
        if not _has_field(DetectedObject, "size"):
            self.why = (
                "its DetectedObject predates rammp-interfaces-ros2#16 (no size): it would misread the "
                "running images' messages"
            )
            return
        from rclpy.qos import qos_profile_sensor_data

        # best effort is compatible with the node's reliable publisher; a
        # late message is worth nothing here, a newer one comes in 0.2 s
        self._sub = node.create_subscription(ObjectDetections, SEGMENTER_TOPIC, self._cb, qos_profile_sensor_data)
        self.t_subscribed = time.monotonic()
        self.available = True

    def _cb(self, msg):
        # on the perception thread: one atomic attribute store, the label image
        # dropped. Never raises: an exception here would end the executor's
        # spin, and with it every camera the mission's perception thread
        # serves. A message it cannot read keeps the frame before, which goes
        # silent in SEG_SILENT_S: the look then says the segmenter is silent.
        try:
            frame = frame_of(msg, self.classes, time.monotonic(), self.instance_id, self.near)
        except Exception as e:  # noqa: BLE001 -- someone else's node publishes this
            self.cb_error = repr(e)
            return
        self.latest = frame
        self.count += 1

    def follow(self, box):
        """From a fix on, follow the selected container by where it was boxed then (`box`, its own
        box in that fix's frame: SegFrame.mine) as well as by its id (frame_of's `near`)."""
        if box is not None:
            self.near = tuple(float(v) for v in box[:4])  # one attribute store: the callback reads it

    def publishers(self, now_s):
        """How many publishers the graph has on SEGMENTER_TOPIC (asked at
        most every SEG_GRAPH_S), or None when the node cannot say."""
        t, n = self._graph
        if t is not None and now_s - t < SEG_GRAPH_S:
            return n
        try:
            n = int(self.node.count_publishers(SEGMENTER_TOPIC))
        except Exception:  # noqa: BLE001 -- a stub node, or a graph query that failed: no answer, the timer decides
            n = None
        self._graph = (now_s, n)
        return n

    def state(self, now_s, color_frame_id=None, color_wh=None, color_stamp_ns=None, frame=_LATEST):
        """ "unavailable" | "absent" | "waiting" | "silent" | "wrong_camera" |
        "wrong_size" | "stale" | "live" for the newest frame (or `frame`),
        against the scene camera's own newest colour frame (its frame_id,
        (w, h), stamp)."""
        if not self.available:
            return "unavailable"
        m = self.latest if frame is _LATEST else frame
        if m is None:
            if self.publishers(now_s) == 0:
                return "absent"  # nothing on the graph publishes it: no word is coming
            return "waiting" if now_s - self.t_subscribed <= SEG_SILENT_S else "silent"
        if now_s - m.recv_s > SEG_SILENT_S:
            return "silent"
        if color_frame_id is not None and m.frame_id != color_frame_id:
            return "wrong_camera"
        if tuple(m.wh) != (0, 0) and color_wh is not None and tuple(m.wh) != tuple(color_wh):
            return "wrong_size"
        if color_stamp_ns is not None and m.stamp_ns is not None and color_stamp_ns - m.stamp_ns > SEG_FRESH_S * 1e9:
            return "stale"
        return "live"

    def describe(self, state, color_frame_id=None, color_wh=None, color_stamp_ns=None, frame=_LATEST):
        """One line for the operator on a segmenter that is not live."""
        m = self.latest if frame is _LATEST else frame
        if state == "unavailable":
            return self.why
        if state in ("absent", "silent"):
            said = (
                "ABSENT: nothing publishes %s" % SEGMENTER_TOPIC
                if state == "absent"
                else "SILENT: no %s in %.1f s" % (SEGMENTER_TOPIC, SEG_SILENT_S)
            )
            if self.cb_error:
                said += " (a message it sent could not be read: %s)" % self.cb_error
            return said + "; select scene_segmenter: rfdetr in the sheppy profile"
        if state == "waiting":
            return "no message yet, in its first %.1f s" % SEG_SILENT_S
        if m is None:
            return state
        if state == "wrong_camera":
            return "it watches '%s', not the scene camera ('%s')%s" % (
                m.frame_id,
                color_frame_id,
                "; the mock?" if not m.frame_id else "",
            )
        if state == "wrong_size":
            return "its frames are %dx%d, the scene camera's %dx%d" % (m.wh[0], m.wh[1], color_wh[0], color_wh[1])
        if state == "stale":
            return "its newest frame is %.1f s older than the scene camera's" % ((color_stamp_ns - m.stamp_ns) / 1e9)
        return state
