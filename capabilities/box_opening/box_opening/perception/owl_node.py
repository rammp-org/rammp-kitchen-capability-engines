"""owl_detector — persistent local bbox service for the semantic gate.

    ros2 run box_opening owl_detector

The OWL model costs tens of seconds to load, and a CLI that loads it per
run makes every mission wait (field 2026-09-01: the detect phase sat on
the preload). This node loads ONCE — at launch, overlapping the
planner's own GPU init — then runs the detector on the live colour
stream at a gentle cadence and publishes the best bbox:

    /box_opening/owl_bbox   std_msgs/Float32MultiArray
                                  [x0, y0, x1, y1, score, frame_age_s]

Published only when something clears vlm.owl_min_score; the stamp is the
FRAME time so the mission can ignore stale sightings. Inference runs only
while the mission has enabled it (a latched Bool on owl_enable, auto-off
after 30 s): outside its detect windows the node heartbeats and leaves
the GPU to the planner. This is the architecture the deployment target
wants: an always-on perception service whose output the mission consumes.

Queries, model, and threshold come from the same container YAML the
mission uses (`container` parameter), so there is exactly one place to
tune them.
"""

import time
import warnings

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32MultiArray, String

from box_opening.perception.owl_source import (
    BLIND,
    CONTAINER_TOPIC,
    HEARTBEAT,
    blind_heartbeat_due,
    container_changed,
    owl_detect,
    owl_floor,
    topics_for,
)

# The mission enables inference only around its detect windows: OWLv2 at
# 100 % GPU duty doubled every cuRobo solve (0.22 s -> 0.47 s measured
# offline 2026-09-02). A stale enable cannot pin the GPU forever either.
ENABLE_MAX_S = 30.0
STALE_S = 3.0  # no new frame for this long -> re-subscribe
RESUB_S = 5.0  # and not more often than this


class OwlDetector(Node):
    def __init__(self, name="owl_detector", **node_kwargs):
        # `name` and `node_kwargs` (parameter_overrides): the engine node hosts two of these
        # in its process, the wrist's and the scene camera's (node.Capability._start_owl)
        super().__init__(name, **node_kwargs)
        from box_opening.models.container import load_press_demo
        from box_opening.tasks.cli_common import default_container_yaml

        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        container = self.declare_parameter("container", "").value
        # which camera this instance watches: the wrist D405 (the mission's
        # close-range aim) or the fixed scene camera (finding the box before
        # the arm moves). One model per instance; each infers only inside
        # its own enable window, so two instances never load the GPU at once.
        self.camera = str(self.declare_parameter("camera", "wrist").value)
        # camera_config: name of the camera YAML this instance reads (e.g.
        # "camera_oak_wrist.yaml" / "camera_oak_scene.yaml" on flatbot).
        self._camera_cfg_name = str(self.declare_parameter("camera_config", "").value) or None
        bbox_topic, enable_topic = topics_for(self.camera)
        # the score floor; the container yaml's owl_min_score unless the
        # launch says otherwise. The scene instance runs lower: from a metre
        # the bench box scores 0.18-0.23 against a 0.18 floor (2026-09-16),
        # and a geometry gate behind it (the lid slab) rejects false boxes
        self.min_score = float(self.declare_parameter("min_score", -1.0).value)
        # 2 Hz: at 1 Hz the brief mid-scan view of the box could fall
        # between ticks; inference is ~0.65 s so this saturates only
        # while frames actually change
        period = float(self.declare_parameter("period_s", 0.5).value)
        # No input downscale knob: OWLv2's processor resizes every frame to
        # its fixed 960x960 before the model, so a 1280x720 frame and a
        # 640x360 one both cost 0.62 s on the Orin (measured 2026-09-17),
        # and the smaller one only scores lower. The time is the model's.
        cfg_path = container or default_container_yaml()
        self.cfg = load_press_demo(str(cfg_path))
        self._cfg_path = str(cfg_path)
        # a floor the launch set outlives a container change; one taken
        # from the YAML follows the YAML
        self._min_score_from_launch = self.min_score >= 0
        if self.min_score < 0:
            self.min_score = owl_floor(self.cfg, self.camera)

        self.get_logger().info("loading %s ..." % self.cfg.owl_model)
        t0 = time.monotonic()
        import torch  # noqa: F401
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        from transformers import logging as hf_logging

        hf_logging.set_verbosity_error()
        self._proc = Owlv2Processor.from_pretrained(self.cfg.owl_model)
        self._model = Owlv2ForObjectDetection.from_pretrained(self.cfg.owl_model).eval().cuda()
        # One inference on a blank frame before saying "ready": the first
        # CUDA call pays kernel compilation and allocation, and a mission
        # that enabled a cold instance saw nothing inside its 3 s window
        # (field 2026-09-16).
        import numpy as np

        blank = np.zeros((720, 1280, 3), np.uint8)
        inputs = self._proc(text=[list(self.cfg.owl_queries)], images=[blank], return_tensors="pt").to("cuda")
        with torch.no_grad():
            self._model(**inputs)
        torch.cuda.synchronize()
        self.get_logger().info(
            "owl_detector (%s camera) ready in %.1f s — %s at %.1f Hz, min score %.2f"
            % (
                self.camera,
                time.monotonic() - t0,
                self.cfg.owl_model,
                1.0 / period,
                self.min_score,
            )
        )

        if self.camera == "scene":
            from box_opening.perception.d405 import camera_config
            from box_opening.perception.scene import SceneGrabber

            _scfg = camera_config(self._camera_cfg_name) if self._camera_cfg_name else None
            self.grab = SceneGrabber(self, keep=1, need_depth=False, scene_cfg=_scfg)
        else:
            from box_opening.perception.d405 import D405Grabber

            self.grab = D405Grabber(self, need_depth=False, cfg_name=self._camera_cfg_name)
        self.pub = self.create_publisher(Float32MultiArray, bbox_topic, 1)
        self._last_stamp = None
        self._last_new_frame_t = time.monotonic()
        self._last_resub_t = 0.0
        self._enabled_until = 0.0
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(Bool, enable_topic, self._on_enable, latched)
        # the mission's container (owl_source.announce_container): its
        # prompts and floor replace this node's own the moment it is run
        self.create_subscription(String, CONTAINER_TOPIC, self._on_container, latched)
        self._timer = self.create_timer(period, self._tick)

    def _on_container(self, msg):
        from box_opening.models.container import load_press_demo

        path = str(msg.data)
        if not container_changed(self._cfg_path, path):
            return
        try:
            cfg = load_press_demo(path)
        except Exception as e:  # a bad announcement must not take the detector down
            self.get_logger().error(
                "container %s announced by the mission could not be loaded (%s) — keeping %s"
                % (path, e, self._cfg_path)
            )
            return
        self.cfg, self._cfg_path = cfg, path
        if not self._min_score_from_launch:
            self.min_score = owl_floor(cfg, self.camera)
        self.get_logger().info(
            "container -> %s: queries %s, min score %.2f" % (path, list(cfg.owl_queries), self.min_score)
        )

    def _on_enable(self, msg):
        rising = bool(msg.data) and not self.enabled
        self._enabled_until = (time.monotonic() + ENABLE_MAX_S) if msg.data else 0.0
        if rising:
            # infer the newest frame NOW, not at the next tick: waiting
            # cost every locate 0-0.5 s before its 0.65 s inference. The
            # reset leaves the timer overdue when the inference returns,
            # so the next one follows back to back as it always did.
            self._timer.reset()
            self._tick()

    @property
    def enabled(self):
        return time.monotonic() < self._enabled_until

    def _tick(self):
        g = self.grab
        now_m = time.monotonic()
        stamp = None if g.color_stamp is None else (g.color_stamp.sec, g.color_stamp.nanosec)
        if g.color is None or stamp is None or stamp == self._last_stamp:
            # no NEW frame. The camera may be down — or this process's
            # subscription has gone quiet while the topic streams (a
            # launch-started scene instance did that minutes after finding
            # the box, twice, 2026-09-16, while a fresh subscriber elsewhere
            # got 30 Hz). Re-subscribe after STALE_S, then every RESUB_S.
            stale = now_m - self._last_new_frame_t
            # once frames have really stopped, say so every tick: the
            # mission's rung must tell "up but blind" from "not running"
            # (bench 2026-09-22). Not on one quiet tick: an inference
            # outlasts the tick, and BLIND then ended good waits (09-23).
            if blind_heartbeat_due(stale):
                blind = Float32MultiArray()
                blind.data = [0.0, 0.0, 0.0, 0.0, BLIND, 0.0]
                self.pub.publish(blind)
            if stale > STALE_S and now_m - self._last_resub_t > RESUB_S:
                self._last_resub_t = now_m
                g.resubscribe()
                self.get_logger().warning(
                    "no new %s frame for %.0f s — re-subscribed (#%d)" % (self.camera, stale, g.resubscribes)
                )
            return
        if self._last_stamp is not None and now_m - self._last_new_frame_t > STALE_S:
            self.get_logger().info("%s frames back after %.0f s" % (self.camera, now_m - self._last_new_frame_t))
        self._last_stamp = stamp
        self._last_new_frame_t = now_m
        frame_t = g.color_stamp.sec + g.color_stamp.nanosec * 1e-9
        if not self.enabled:
            # heartbeat only: the mission's rung must still tell "node
            # alive, idle" from "node absent"
            msg = Float32MultiArray()
            msg.data = [0.0, 0.0, 0.0, 0.0, HEARTBEAT, 0.0]
            self.pub.publish(msg)
            return

        # D405Grabber stores BGR; the processor expects RGB (measured
        # harmless on the capture set, but it is the wrong buffer)
        top = owl_detect(self._proc, self._model, g.color[:, :, ::-1], self.cfg.owl_queries, self.min_score)
        best = top[0] if top else None
        msg = Float32MultiArray()
        now = self.get_clock().now().nanoseconds * 1e-9
        # slot 5 is the frame's AGE at publish, never an absolute time: a
        # float32 at ~1.7e9 s has 128 s resolution, which made any
        # freshness test on an absolute stamp a coin flip (review
        # 2026-09-02). The rung rebuilds an absolute frame time on its
        # own float64 clock. A bbox is 0.65-1.15 s old by the time it
        # lands, and the mission must not gate a parked frame with a box
        # seen while moving.
        if best is None:
            # heartbeat: the mission can tell "node alive, keep waiting"
            # from "node absent, fall back" (field 2026-09-01: without
            # this, one missed window cost a cold in-process model load)
            msg.data = [0.0, 0.0, 0.0, 0.0, HEARTBEAT, 0.0]
        else:
            # the best box first (six fields: what every consumer read
            # before), then the runners-up: a scene with two containers
            # gives the locator both, and its footprint gate picks
            age = max(0.0, now - frame_t)
            data = []
            for score, (x0, y0, x1, y1) in top:
                data += [float(x0), float(y0), float(x1), float(y1), float(score), age]
            msg.data = data
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = OwlDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
