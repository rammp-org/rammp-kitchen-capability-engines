"""Local open-vocabulary bbox: OWLv2 on the Jetson, no network in the loop.

The wheelchair will not always have internet (deployment constraint,
2026-09-01), so the semantic gate needs a local answer. OWLv2 runs in the
persistent owl_detector node (perception/owl_node.py); this module is the
mission side: the enable gate, the topic rung, and the pure helpers.

Measured on capture 20260901-130610 (8 scan-pose frames): bbox stable to
+/-1 px across frames, centred on the true box, score 0.235-0.249
against a 0.12-0.18 false-positive floor. Phrasing matters more than the
model: "a small white square box" finds it, "a food storage container"
does not — which is why the queries are CONFIG, plural, and the best
score across them wins.

Same contract and the same humility as the Claude backend: a returned
roi only ever NARROWS where the depth detector looks; every geometric
honesty gate still stands behind it. There is deliberately NO in-process
model: a cold load costs tens of seconds against a 10 s detect budget and
parked a second OWLv2 on the planner's GPU (review 2026-09-02).
"""

from pathlib import Path

# The container the mission was run with, announced to BOTH owl_detector
# instances: their prompts and score floor live in the container YAML, and
# the launch starts them with none of their own — they loaded oxo_pop.yaml
# whatever `press_demo --container` said (2026-09-22). One latched topic:
# a node that starts later still gets the last announcement.
CONTAINER_TOPIC = "/box_opening/owl_container"


def announce_container(node, path):
    """Publish the ABSOLUTE path of the container YAML the mission uses,
    latched, for the owl_detector nodes to load their prompts from."""
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub = node.create_publisher(
        String, CONTAINER_TOPIC, latched
    )  # the node keeps it: the latched message lives as long as the publisher
    pub.publish(String(data=str(Path(path).resolve())))
    return pub


def container_changed(current, announced):
    """Does an announced container path name a different file from the one
    in use? An empty announcement is no announcement."""
    if not announced:
        return False
    return Path(announced).resolve() != Path(current).resolve()


def owl_floor(cfg, camera):
    """The score floor for `camera`'s detector, from the container config."""
    return float(cfg.owl_min_score_scene if camera == "scene" else cfg.owl_min_score)


def owl_detect(proc, model, rgb, queries, floor, device="cuda", k=None):
    """OWLv2 on one RGB frame: the top boxes above `floor`, best first —
    [(score, [x0, y0, x1, y1])]. What the owl_detector node publishes, and
    what an offline replay of recorded frames computes the same way. The
    frame must be upright: OWLv2 scored flatbot's pink lid 0.04 upside down,
    0.44 upright (2026-10-22), and the cameras now rotate on the device
    (rammp-deployments #133)."""
    import torch

    h, w = rgb.shape[:2]
    inputs = proc(text=[list(queries)], images=[rgb], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_object_detection(
        out, threshold=float(floor), target_sizes=torch.tensor([[h, w]]).to(device)
    )[0]
    top = top_boxes(
        res["scores"].tolist(),
        res["labels"].tolist(),
        [b.tolist() for b in res["boxes"]],
        floor,
        **({} if k is None else {"k": k}),
    )
    return top


def topics_for(camera):
    """(bbox topic, enable topic) of the owl_detector instance watching
    `camera`. The wrist keeps the original names; the scene camera's
    instance gets its own pair, so the two never answer each other's
    questions. The ONE place these names are written: the node, the rung
    and the stubs all ask here."""
    if camera == "wrist":
        return "/box_opening/owl_bbox", "/box_opening/owl_enable"
    if camera == "scene":
        return "/box_opening/owl_bbox_scene", "/box_opening/owl_enable_scene"
    raise ValueError("camera must be 'wrist' or 'scene', not %r" % (camera,))


BBOX_TOPIC = topics_for("wrist")[0]  # [x0,y0,x1,y1,score,frame_age_s] per box
TOPIC_FRESH_S = 3.0
# a bbox may gate the depth watcher only when its FRAME is this recent:
# the node stamps frame time, inference is ~0.65 s, and a box seen while
# the camera was still moving must not gate a parked frame
ROI_FRESH_S = 1.5


BOX_FIELDS = 6  # x0, y0, x1, y1, score, frame_age_s — repeated per candidate
TOP_K = 3


def top_boxes(scores, labels, boxes, min_score, k=TOP_K):
    """The k highest-scoring boxes above the floor, best first: [(score,
    [x0, y0, x1, y1]), ...]. Two containers in the scene scored 0.255 and
    0.255 (2026-09-17) and the single best box flipped between them every
    tick; the consumer decides by geometry which one is its container."""
    got = [
        (float(s), [int(v) for v in b])
        for s, _l, b in zip(scores, labels, boxes, strict=False)
        if float(s) >= float(min_score)
    ]
    got.sort(key=lambda t: -t[0])
    return got[:k]


def bboxes_in_msg(m):
    """Every candidate in a bbox message (BOX_FIELDS per box), best first;
    a heartbeat (score < 0) yields none. The first six fields are the best
    box, so consumers that read only those see what they always did."""
    if m is None:
        return []
    out = []
    for i in range(0, len(m) - BOX_FIELDS + 1, BOX_FIELDS):
        b = list(m[i : i + BOX_FIELDS])
        if b[4] >= 0.0:
            out.append(b)
    return out


# A heartbeat's score slot: the node is up and has frames but saw nothing
# (HEARTBEAT), or is up and has had NO camera frame at all (BLIND — a
# camera down or a subscription gone quiet; it used to send nothing then,
# and read as a dead node: bench 2026-09-22).
HEARTBEAT = -1.0
BLIND = -2.0
# ... and BLIND only once frames have really stopped: an inference (0.65 s)
# outlasts the 0.5 s tick, so the tick after it routinely finds no new
# frame — sending BLIND then made the mission give up on a node that was
# fine (2026-09-23, two runs). The node re-subscribes at the same age.
BLIND_AFTER_S = 3.0


def blind_heartbeat_due(stale_s):
    """Should a node whose last new frame is `stale_s` old say BLIND?"""
    return float(stale_s) > BLIND_AFTER_S


def classify_bbox_msg(m, now, fresh_s=TOPIC_FRESH_S):
    """One topic message -> "bbox" | "alive" | "blind" | "stale". Pure, testable."""
    if m is None or now - m[5] > fresh_s:
        return "stale"
    if m[4] >= 0.0:
        return "bbox"
    return "blind" if m[4] <= BLIND else "alive"


def roi_from_bbox(m, shape, pad):
    """Padded, image-clamped roi from a bbox message. Pure."""
    h, w = shape[:2]
    x0, y0, x1, y1 = m[:4]
    return (
        max(0, int(x0) - pad),
        max(0, int(y0) - pad),
        min(w - 1, int(x1) + pad),
        min(h - 1, int(y1) + pad),
    )


class OwlRung:
    """The owl rung: reads the persistent owl_detector node's topic and
    owns the enable gate the node listens to.

    Call enable() around a detect window and disable() when the fix
    commits — inference outside the window only slows the planner. The
    node heartbeats every tick even with nothing seen, so the rung can
    wait for a live node's answer instead of guessing:

        fresh bbox       -> use it
        fresh heartbeat  -> node alive: keep waiting (it answers ~2 Hz)
        neither, ever    -> node absent: decline at once (plain depth is
                            the floor; nothing in-process to fall back to)

    A live node that finishes waiting having seen NO box is trusted: the
    rung declines — two models disagreeing about the same frames helps
    nobody. While enabled, every fresh bbox live-gates the depth watcher
    so its roi-gated samples are already collected when the arm parks.
    """

    def __init__(self, node, cfg, watcher_holder=None, camera="wrist"):
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Bool, Float32MultiArray

        self.node = node
        self.cfg = cfg
        self.camera = camera
        self.watcher_holder = watcher_holder or {}
        self.latest = None
        # the newest message that carried a BOX, apart from the newest
        # message: a heartbeat landing just after a box must not hide it
        # from a reader that polls (fresh_box)
        self.latest_box = None
        self.enabled = False
        bbox_topic, enable_topic = topics_for(camera)
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._enable_pub = node.create_publisher(Bool, enable_topic, latched)
        self._Bool = Bool
        node.create_subscription(Float32MultiArray, bbox_topic, self._cb, 1)

    def _now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def fresh_box(self, fresh_s=TOPIC_FRESH_S):
        """The newest box message while it is fresh, or None — whatever
        heartbeats arrived after it (2026-09-23: "the scene OWL saw no box"
        with the box in view; a heartbeat had replaced it within
        milliseconds, before the locate loop looked)."""
        m = self.latest_box
        if m is None or classify_bbox_msg(m, self._now(), fresh_s) != "bbox":
            return None
        return m

    def enable(self):
        self.enabled = True
        self._enable_pub.publish(self._Bool(data=True))

    def disable(self):
        self.enabled = False
        self._enable_pub.publish(self._Bool(data=False))

    def _cb(self, msg):
        m = list(msg.data)
        # the node sends the frame's AGE (float32-safe); keep an absolute
        # frame time on this clock so freshness is a real subtraction
        m[5] = self._now() - max(0.0, float(m[5]))
        self.latest = m
        if m[4] >= 0.0:
            self.latest_box = m
        w = self.watcher_holder.get("watcher")
        if (
            w is not None
            and self.enabled
            and m[4] >= 0.0
            and self._now() - m[5] <= ROI_FRESH_S
            and w.grab.color is not None
        ):
            # offered, not written: the watcher applies it only once the
            # camera has been still since before this bbox's frame
            w.offer_roi(roi_from_bbox(m, w.grab.color.shape, int(self.cfg.vlm_pad_px)), m[5])

    def __call__(self, color_rgb, cfg_):
        import time as _t

        import rclpy as _r

        saw_alive = False
        deadline = _t.monotonic() + 2.0
        while _t.monotonic() < deadline:
            box = self.fresh_box()
            if box is not None:
                roi = roi_from_bbox(box, color_rgb.shape, int(cfg_.vlm_pad_px))
                return roi, "OWL node bbox (%d,%d)-(%d,%d) score %.2f" % (
                    *roi,
                    box[4],
                )
            kind = classify_bbox_msg(self.latest, self._now())
            if kind == "blind":
                # up, but no camera frame has reached it: waiting cannot help
                return None, (
                    "OWL node is live but has had no %s frame — the camera, or its subscription "
                    "(see its log)" % self.camera
                )
            saw_alive = saw_alive or kind == "alive"
            if not saw_alive and self.latest is None and _t.monotonic() > deadline - 1.5:
                break  # nothing at all in 0.5 s: no node on the graph
            _r.spin_once(self.node, timeout_sec=0.1)
        if saw_alive:
            return None, "OWL node is live and sees no container top"
        return None, "owl_detector node not running — plain depth"
