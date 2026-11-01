"""Container model: every mission target derives from this + a detected pose.

Every pose-directed leg uses the press attitude from the container config
(top-down: the tool z must point at the lid to press its button), either
fixed in the world or steered to the target's bearing (press_yaw_steer).
"""

import math
from dataclasses import dataclass

import yaml

from box_opening.constants import TRANSIT_SPEED
from box_opening.geometry import euler_deg_to_quat_xyzw, yaw_about_world_z
from box_opening.primitives.look import SWEEP_RAD


@dataclass(frozen=True)
class ContainerPose:
    xyz: tuple
    yaw: float


@dataclass(frozen=True)
class ContainerModel:
    dims: tuple
    lid_dims: tuple
    button_offset: tuple
    button_diameter_m: float
    touch_nm: float
    press_attitude_rpy_deg: tuple
    hover_standoff: float  # carry/set-down hover above a target (Place)
    measure_me: bool
    # True: the attitude's yaw is steered to the box's bearing from the base
    # (the "bearing family" of 2026-08-25). False: the attitude is FIXED in
    # the world, whatever the box's bearing — the wrist then keeps one
    # orientation from LOOK through approach and press instead of turning
    # up to 105 deg on the way to staging (owner, 2026-09-16: "the arm
    # keeps flipping"). Reach checked on the live planner: 20/20 positions
    # over the table, from HOME and from LOOK, straight descents everywhere.
    press_yaw_steer: bool = True
    # True: the lid is a disc lid_dims[0] across (the Ankou canister): the
    # scene camera may finish a box its frame cuts off from the arc of the
    # lid's side it does see (perception/scene_source.lid_circle)
    round: bool = False

    def press_quat(self, xyz):
        """The press/grip/place attitude at a base-frame point (xyzw)."""
        yaw = math.atan2(float(xyz[1]), float(xyz[0])) if self.press_yaw_steer else 0.0
        return attitude_quat(self.press_attitude_rpy_deg, yaw)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            dims=tuple(raw["dims"]),
            lid_dims=tuple(raw["lid_dims"]),
            button_offset=tuple(raw["button_offset"]),
            button_diameter_m=float(raw.get("button_diameter_m", 0.036)),
            touch_nm=float(raw["press"]["touch_nm"]),
            press_attitude_rpy_deg=tuple(raw["press_attitude_rpy_deg"]),
            hover_standoff=float(raw["hover_standoff"]),
            measure_me=bool(raw.get("measure_me", True)),
            press_yaw_steer=bool(raw.get("press_yaw_steer", True)),
            round=bool(raw.get("round", False)),
        )


def from_container(cpose, offset):
    """Container-frame offset -> base_link, rotated by the container yaw."""
    c, s = math.cos(cpose.yaw), math.sin(cpose.yaw)
    ox, oy, oz = offset
    return [
        cpose.xyz[0] + c * ox - s * oy,
        cpose.xyz[1] + s * ox + c * oy,
        cpose.xyz[2] + oz,
    ]


def attitude_quat(rpy_deg, yaw):
    """Primitive attitude from config rpy, yaw-steered about world z (xyzw)."""
    return list(yaw_about_world_z(euler_deg_to_quat_xyzw(rpy_deg), yaw))


@dataclass(frozen=True)
class PressDemoCfg:
    """Knobs of the camera-driven press flow (owner design 2026-08-24)."""

    travel_m: float  # how far below the estimated button the TOUCH may search
    contact_nm: float  # first-contact threshold of the touch stage
    button_travel_m: float  # the bounded push from the measured contact
    push_hold_s: float  # held on the button after a good push, before the recoil
    press_speed: float
    staging_m: float  # full-world approach height above the button
    scan_xyz: tuple
    search_arc_rad: float  # base-yaw half-arc the sweep pans each way
    timeout_s: float
    min_hits: int
    tol_m: float
    window_s: float
    fresh_s: float
    grip_band: tuple  # gripper feedback band = holding the popped button
    lift_m: float
    lift_speed: float
    grip_speed: float  # descent onto the popped button (ramp at the bench)
    grip_hop_m: float  # retreat height above the plane between press and grip
    setdown_speed: float  # guarded set-down; the trip IS the success
    setdown_touch_nm: float  # set-down trip threshold (gentler than the press)
    park_tool_down: bool  # rest at the LOOK pose between runs, not factory HOME
    require_button_circle: bool  # only button-circle sightings may commit a fix
    confirm_in_flight: bool  # the wrist confirms the box DURING the approach; no stop
    recentre_max_moves: int  # moves over the button at staging before pressing (0: aim once, press)
    press_offset_xy: tuple  # base-frame trim of the press target (m)
    warp_fast_speed: float  # free-air scale of a warped descent
    warp_slow_frac: float  # final path fraction at contact speed
    grip_clear_m: float  # fingertip stop height above the lid plane
    grip_offset_xy: tuple  # base-frame grasp trim (bench-measured bias)
    detect_period_s: float  # detector tick; min_hits * this = commit floor
    detect_source: str  # 'vlm' | 'depth'
    vlm_backends: tuple  # ladder order, e.g. ('owl', 'claude')
    vlm_model: str
    owl_model: str
    owl_queries: tuple
    owl_min_score: float
    owl_min_score_scene: float  # the scene camera's floor: the box is small from a metre
    vlm_target: str  # what to ask Claude to box — plain English
    vlm_timeout_s: float
    vlm_pad_px: int
    # the scene camera's box from sheppy's always-on scene segmenter
    # (perception/segmenter_source): the classes it boxes this container as,
    # () for none — the scene OWL alone then finds it
    segmenter_classes: tuple = ()
    segmenter_box: str = "holder"  # "holder": its box is the holder the lid stands on; "as_is": the container


SCENE_DETECT_KEYS = ("segmenter_classes", "segmenter_box")


def scene_detect_cfg(raw):
    """(segmenter_classes, segmenter_box) from a container config's
    scene_detect block, validated; ((), "holder") without one."""
    sd = raw.get("scene_detect")
    if sd is None:
        sd = {}
    if not isinstance(sd, dict):
        raise ValueError("scene_detect must be a mapping (%s)" % ", ".join(SCENE_DETECT_KEYS))
    unknown = sorted(set(sd) - set(SCENE_DETECT_KEYS))
    if unknown:
        raise ValueError(
            "scene_detect: unknown key(s) %s (known: %s)" % (", ".join(map(str, unknown)), ", ".join(SCENE_DETECT_KEYS))
        )
    classes = sd.get("segmenter_classes")
    if classes is None:
        classes = []
    if not isinstance(classes, list) or not all(isinstance(c, str) and c.strip() for c in classes):
        raise ValueError(
            "scene_detect.segmenter_classes must be a list of the scene segmenter's class names "
            "(e.g. [food container]); an empty list or none means the scene OWL alone"
        )
    box = sd.get("segmenter_box", "holder")
    if box not in ("holder", "as_is"):
        raise ValueError("scene_detect.segmenter_box must be 'holder' or 'as_is'")
    return tuple(c.strip() for c in classes), box


def load_press_demo(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    pd = raw["press_demo"]
    segmenter_classes, segmenter_box = scene_detect_cfg(raw)
    cfg = PressDemoCfg(
        travel_m=float(pd["travel_m"]),
        contact_nm=float(raw["press"].get("contact_nm", 3.0)),
        button_travel_m=float(raw["press"].get("button_travel_m", 0.004)),
        push_hold_s=float(raw["press"].get("push_hold_s", 0.0)),
        press_speed=float(pd["speed"]),
        staging_m=float(pd["staging_m"]),
        scan_xyz=tuple(raw["scan"]["xyz"]),
        search_arc_rad=float(raw["scan"].get("search_arc_rad", SWEEP_RAD)),
        timeout_s=float(raw["detect"]["timeout_s"]),
        min_hits=int(raw["detect"]["min_hits"]),
        tol_m=float(raw["detect"]["tol_m"]),
        window_s=float(raw["detect"]["window_s"]),
        fresh_s=float(raw["detect"]["fresh_s"]),
        grip_band=tuple(raw["open_box"]["grip_band"]),
        lift_m=float(raw["open_box"]["lift_m"]),
        lift_speed=float(raw["open_box"]["lift_speed"]),
        grip_speed=float(raw["open_box"].get("grip_speed", 0.15)),
        grip_hop_m=float(raw["open_box"].get("grip_hop_m", 0.12)),
        setdown_speed=float(raw["open_box"].get("setdown_speed", 0.15)),
        setdown_touch_nm=float(raw["open_box"].get("setdown_touch_nm", 4.0)),
        park_tool_down=bool(raw["open_box"].get("park_tool_down", False)),
        require_button_circle=bool(raw["detect"].get("require_button_circle", True)),
        confirm_in_flight=bool(raw["detect"].get("confirm_in_flight", False)),
        recentre_max_moves=int(raw["detect"].get("recentre_max_moves", 2)),
        press_offset_xy=tuple(float(v) for v in raw["open_box"].get("press_offset_xy", [0.0, 0.0])),
        warp_fast_speed=float(raw["open_box"].get("warp_fast_speed", 0.0)),
        warp_slow_frac=float(raw["open_box"].get("warp_slow_frac", 0.3)),
        grip_clear_m=float(raw["open_box"]["grip_clear_m"]),
        grip_offset_xy=tuple(raw["open_box"]["grip_offset_xy"]),
        detect_period_s=float(raw["detect"].get("period_s", 0.15)),
        detect_source=str(raw["detect"].get("source", "depth")),
        vlm_backends=tuple(raw.get("vlm", {}).get("backends", ("claude",))),
        vlm_model=str(raw.get("vlm", {}).get("model", "claude-opus-5")),
        owl_model=str(raw.get("vlm", {}).get("owl_model", "google/owlv2-base-patch16-ensemble")),
        owl_queries=tuple(raw.get("vlm", {}).get("owl_queries", ("a small white square box",))),
        owl_min_score=float(raw.get("vlm", {}).get("owl_min_score", 0.18)),
        owl_min_score_scene=float(raw.get("vlm", {}).get("owl_min_score_scene", 0.12)),
        vlm_target=str(raw.get("vlm", {}).get("target", "the food-storage container")),
        vlm_timeout_s=float(raw.get("vlm", {}).get("timeout_s", 20.0)),
        vlm_pad_px=int(raw.get("vlm", {}).get("pad_px", 20)),
        segmenter_classes=segmenter_classes,
        segmenter_box=segmenter_box,
    )
    if not raw["open_box"]["grip_band"][0] < raw["open_box"]["grip_band"][1] < 0.8:
        raise ValueError(
            "open_box.grip_band must be (lo, hi) with hi < 0.8 — 0.8 is the "
            "closed-on-air feedback and can never mean 'holding the button'"
        )
    if len(cfg.press_offset_xy) != 2 or any(abs(v) > 0.02 for v in cfg.press_offset_xy):
        raise ValueError(
            "open_box.press_offset_xy must be two values within +/-0.02 m — a "
            "bigger trim means the fix or the mount is wrong, not the pads"
        )
    if not 0.5 <= cfg.contact_nm < float(raw["press"]["touch_nm"]):
        raise ValueError(
            "press.contact_nm must be at least 0.5 Nm and below press.touch_nm "
            "(the touch finds the surface; touch_nm is the push's backstop)"
        )
    if not 0.001 <= cfg.button_travel_m <= 0.015:
        raise ValueError("press.button_travel_m must be 1-15 mm (the button's own travel)")
    if not 0.0 <= cfg.push_hold_s <= 2.0:
        raise ValueError("press.push_hold_s must be 0-2 s")
    if not 0.0 <= cfg.grip_clear_m <= 0.02:
        raise ValueError(
            "open_box.grip_clear_m must be in [0, 0.02] m — 0 puts the "
            "fingertips ON the lid plane, more than 2 cm closes above the knob"
        )
    if any(abs(v) > 0.02 for v in cfg.grip_offset_xy):
        raise ValueError(
            "open_box.grip_offset_xy is a mm-scale bias trim, not an offset "
            "— |each| must be <= 0.02 m (re-measure the mount instead)"
        )
    if cfg.travel_m <= 0:
        raise ValueError("press_demo.travel_m must be positive")
    if not 0.0 < cfg.press_speed <= 1.0:
        raise ValueError("press_demo.speed outside (0, 1]")
    # Contact-leg speeds are bounded well below transit: the torque guard
    # only arms after the first execution feedback, so the descent travels
    # speed * executor_poll before a baseline exists (BASELINE_TRAVEL_M is
    # the 10 mm budget). 0.5 keeps that inside budget even if the poll
    # regresses to 0.1 s. Raise the poll rate, not this bound.
    for field in ("grip_speed", "setdown_speed", "lift_speed"):
        v = getattr(cfg, field)
        if not 0.0 < v <= 0.5:
            raise ValueError(
                "open_box.%s must be in (0, 0.5] — contact/carry legs are " "guard-limited, not transit legs" % field
            )
    # The free-air part of a guarded descent may run as fast as a transit
    # leg, and no faster: past TRANSIT_SPEED the arm would be moving quicker
    # toward something it is about to touch than it ever moves in the open.
    # The bound was 0.6 and was raised to 0.75 with it (owner, 2026-09-15),
    # which is where the chained press approach already cruises.
    if cfg.warp_fast_speed and not 0.0 < cfg.warp_fast_speed <= TRANSIT_SPEED:
        raise ValueError(
            "open_box.warp_fast_speed must be in (0, %g] — the free-air part "
            "of a GUARDED descent, never faster than a transit leg" % TRANSIT_SPEED
        )
    if not 0.0 < cfg.warp_slow_frac < 1.0:
        raise ValueError("open_box.warp_slow_frac must be in (0, 1)")
    if cfg.detect_source not in ("depth", "vlm"):
        raise ValueError("detect.source must be 'depth' or 'vlm'")
    if not 0 <= int(raw["detect"].get("recentre_max_moves", 2)) <= 5:
        raise ValueError("detect.recentre_max_moves must be 0-5")
    bad = set(cfg.vlm_backends) - {"owl", "claude"}
    if bad or (cfg.detect_source == "vlm" and not cfg.vlm_backends):
        raise ValueError("vlm.backends must be a non-empty subset of ['owl', 'claude']")
    if not 0.0 < cfg.owl_min_score < 1.0:
        raise ValueError("vlm.owl_min_score must be in (0, 1)")
    if not 0.0 < cfg.owl_min_score_scene < 1.0:
        raise ValueError("vlm.owl_min_score_scene must be in (0, 1)")
    if not 0.0 < cfg.vlm_timeout_s <= 60.0:
        raise ValueError("vlm.timeout_s must be in (0, 60] s")
    if not 0.0 < cfg.detect_period_s <= 0.5:
        raise ValueError("detect.period_s must be in (0, 0.5] s")
    # staging is anchored at the BUTTON TOP; the full-world obstacle tops
    # out at container top + err-tall (2 cm) + collision padding (2 cm),
    # and the finger collision spheres reach ~6 cm below tool_frame.
    # 0.10 above the cuboid gap is the MEASURED plans/fails boundary
    # (margin probe vs the real planner, 2026-08-25).
    clearance = float(raw["dims"][2]) - float(raw["button_offset"][2]) + 0.10
    if cfg.staging_m < clearance:
        raise ValueError(
            "press_demo.staging_m %.3f cannot clear the container cuboid: "
            "dims.z - button_offset.z + 0.10 (err-tall + padding + gripper "
            "spheres, measured 2026-08-25) = %.3f m needed" % (cfg.staging_m, clearance)
        )
    return cfg
