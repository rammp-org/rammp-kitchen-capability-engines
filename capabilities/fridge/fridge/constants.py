"""The fridge's values for door_common: what is true of this fridge and this rig, not of doors.

Everything not here is door_common's. MEASURED changes with the fridge and the wheelchair, so each of its values is also
a ROS parameter (lower case); the rest was tuned on the arm and needs code to change.
"""

from door_common import common as dc

INCH = 0.0254

# --- measured on the fridge being opened (per fridge: the node's parameters) ---
MEASURED = {
    # the detected handle point -> door's front surface, perpendicular to the door: the door's face, its inside and
    # the hinge are placed from the handle by it. Measured 11-06: the bar's back edge 2.5 in from the face, the bar
    # ~0.5 in deep (front ~3.0 in). Still 0.070 (the STL's bar middle) until the camera's handle point is measured
    # against the bar's real front (docs/TESTING.md, "Before the next run")
    "handle_centre_to_face_m": 0.070,
    # door's front surface -> hinge pin axis: door thickness 1.25 in (measured 09-20, pin assumed at the back face) +
    # 1.5 cm (hinge side crept forward). If the pin depth is measured directly, use it alone.
    "pin_behind_face_m": 1.25 * INCH + 0.015,
    # handle bar centre line -> hinge pin axis, along the door at handle height (measured 09-20)
    "hinge_along_door_m": 14.5 * INCH,
    # the door's full width, hinge-side edge to free edge (~18 in, at the top of the door). Too small = the plans pass
    # closer to the real free edge than they think
    "door_width_m": 18 * INCH,
    # front face to the innermost point of its inside, bins / shelves included (~4.2 in by hand, may be slightly off).
    # Modelled as a slab this deep across the whole door: it sets the room behind the open door for the push-open
    "door_depth_m": 4.2 * INCH,
    # door top / bottom above / below the middle of the handle bar (measured 10-24). The close pushes only within these
    "door_above_handle_m": 15 * INCH,
    "door_below_handle_m": 15.5 * INCH,
    # the handle bar's grippable straight length, between its mounts: 8 in, measured on the fridge (11-06). It limits
    # how far above or below the middle of the bar the open may grasp (GRASP_END_MARGIN_M)
    "handle_length_m": 8 * INCH,
    "body_depth_m": 0.50,  # cabinet behind the door
    # LEVEL MOUNT ONLY (dojo, which has no wheelchair model): no part of the arm or gripper may be more than this far
    # behind the arm base (arm-base x) while on its right (y < 0) -- the stand-in for the chair (10-24: a close ran the
    # arm into the back of the chair; user: ~8 in estimated, then 12, then 15). On the 30-deg mount the wheelchair and
    # its occupant are the cuRobo planner's own collision layers instead (chair.py)
    "chair_keep_out_behind_m": 15 * INCH,
}

# --- grasp: the microwave's corrections on the Jetson rig, the values every fridge run so far used (10-22, 10-24: the
# door files' handle-to-grasp is 4.8 cm); nothing has checked them on this handle ---
# along the approach axis (negative = past the detected handle). 11-05: the microwave's 0.088 grabbed too deep on
# the fridge handle -> 3.8 cm (1.5 in) shallower
GRIP_EXT = 0.126
VERTICAL_CORR = 0.0  # world z
GRASP_X_ADJUST = 0.040  # along the approach axis
GRASP_FORCE = 0.5  # a normal close (the microwave's 0.25 is loosened on purpose, to let its bar turn)
# the open's grasp_offset_m (up from the middle of the bar) keeps the middle of the finger pads at least this far in
# from either end of handle_length_m: half the 2F-85 pad's ~2.2 cm width, plus ~2 cm for the detected middle being off
GRASP_END_MARGIN_M = 0.03

# --- how the motions are planned (planning.py, world.py): free-space legs by cuRobo, in a collision world of the fridge
# (FridgeModel: cabinet, door at its angle, handle) plus the planner's own chair layers and the scene camera, the arm's
# configuration cuRobo's to choose; legs on or along the door by local IK along their exact path, checked by cuRobo;
# each motion one trajectory joined by cuRobo. The poses below are the only places a motion is told to pass ---
GRASP_PITCHES_DEG = (15, 30, 45)  # the grasp's nose-down tilts tried (configure sets them per mount), each with the
# hand either way up (fingers across the bar both ways): the first of these whose whole open plans is used. Only
# 15-45 deg: under 15 the fingers don't grip the bar firmly enough to pull the door (user, 2026-11-08)
PREGRASP_STANDOFF_M = (0.12, 0.08)  # line up this far straight out from the grasp, then straight in (handle out of
# the world for that leg only); the first that plans
ALT_CONFIG_SEEDS = (("J3", -60.0), ("J3", 60.0), ("wrist", 0.0))  # if no grasp's swing plans from the configuration
# cuRobo picks from the start joints, these: the start joints with the elbow (J3) turned, or the wrist flipped (J5+180,
# -J6, J7+180), as cuRobo's start for planning the line-up pose -- other arm configurations at the same grasp
SWING_CANDIDATES = 3  # what follows the swing is planned for at most this many grasps whose approach + swing plan
PLAN_BUDGET_S = 120.0  # the time one search may take (the whole plan, or a re-plan after the arm or the base moved:
# the time spent moving never counts); past it the search refuses
HAND_WINDOW_U_M = 0.15  # onto and off the handle: the door slab within this of the hand, along the door ...
HAND_WINDOW_Z_M = 0.15  # ... and up or down, is out of cuRobo's check (the handle with it): the hand is there
SWING_DEG = 90  # the full pull (push_open false) without a swing_deg. Exactly this swing or the run refuses: no smaller
# fallback

# --- how the open moves (the side grasp's 11-06 hardware runs, side_run/; run_general.py): the swing held compliant, the
# gripper closing as the pull starts and opening while the arm still moves, everything after the release one motion ---
PULL_GAINS = {  # the driver's stiffest joint impedance (kinova-gen3-driver gains.h: kq <= 400 N m/rad, torque_limit <=
    # the URDF's effort limits), critically damped: the swing and everything after the release run under it
    "kq": (400.0,) * 7,
    "zeta": 1.0,
    "torque_limit": (39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0),
}
PULL_PATH_TOL_RAD = 0.15  # the swing's tracking limit (rad, per joint)
GRIP_DURING_SWING = True  # the gripper closes on the bar as the swing starts (the 2F-85 takes ~1 s), no stop on the
# handle; False: it closes first, then the swing
RELEASE_ARC_DEG = 8.0  # the pull runs on this far along the door's arc while the gripper opens (a stopped hand that
# opened pushed the door back shut, 11-06) ...
OPEN_DURING_S = 2.0  # ... for at least this long past the pull's end ...
OPEN_LEAD_S = 0.3  # ... the gripper starting to open this long before the hand passes the pull's end ...
GRIP_OPEN_SPEED = 0.2  # ... slowly
AFTER_RELEASE_PATH_TOL_RAD = 0.25  # back off + round the door + the push, one motion under PULL_GAINS: this tracking
# limit (free space until the push; impedance lags the path more the faster it goes: 11-06 at ~1 rad/s, past 0.15 rad)
# the scene camera on its post (not in the planner's chair layers; 11-06 the arm hit it): arm_mount ->
# scene_camera_color_optical_frame from TF on flatbot, kept out of every fridge world as a box this big (30-deg mount)
SCENE_CAMERA_XYZ = (0.132, -0.441, 0.328)
SCENE_CAMERA_BOX_M = 0.18

# --- release: let go, then back straight out of the handle ---
BACK_OFF_M = 0.08

# --- after the full pull (push_open false): back round the door to its inner side, so the arm ends beside the open
# door. cuRobo routes there from the back-off; if it finds no route, it is given one point past the door's free edge ---
AROUND_EDGE_CLEAR_M = (0.15,)  # that point: this far past the free edge (tool point) ...
AROUND_OUT_M = 0.15  # ... and at least this far off the door's mid-plane, on its outside
AROUND_END_M = (0.25, 0.30)  # the end: this far from the hinge along the door, this far off its inside, facing the
# fridge, upright: beside the open door, in front of the fridge opening

# --- the push-open, the open's default (push_open.py): pull part way holding the handle, let go, push the door the
# rest of the way with the side of the hand: in behind the door, fingers along it toward the hinge, rolled vertical
# (facing the hand into the cabinet puts the gripper through the door). cuRobo routes from the back-off to the push
# start round the open door (a point past its free edge only if it finds no route), then onto the inner face and round
PUSH_OPEN_PULL_DEG = 60  # the pull. 11-05 sim sweep (PyBullet planner): 60 gave the most chair placements where the
# open and the close plan; kept for cuRobo until it is swept again
PUSH_OPEN_TO_DEG = 90  # push the door on to this
PUSH_OPEN_MIN_PUSH_DEG = 5  # ... at least this far past the pull: the pull is at most PUSH_OPEN_TO_DEG - this
PUSH_OPEN_EDGE_CLEAR_M = 0.16  # the point past the free edge (tool point), when cuRobo needs one
PUSH_OPEN_GAP_M = 0.10  # the push start: this far off the door's inner face (tool point) ...
PUSH_OPEN_DZ_M = -0.10  # ... this far above the handle's height (negative: below; configure sets it from MOUNTS) ...
PUSH_OPEN_AT_M = (0.35, 0.30, 0.40, 0.25)  # ... this far from the hinge along the door (tried in order) ...
PUSH_OPEN_APPROACH_DEG = (0, 20)  # ... the fingers this far off "along the door toward the hinge", toward pointing
# into the cabinet (tried in order, each with the tool's y up and down) ...
PUSH_OPEN_ROLL_DEG = 90  # ... rolled this far about the approach (90: fingers open vertically, the flat side to the door)
PUSH_OPEN_PRELOAD_M = 0.01  # the push: the side of the hand this far past touching the inner face ...
PUSH_DOOR_PANEL_M = 0.025  # ... checked (world.FridgeModel.pushed) against the door where it is at each step: its outer
# panel this thick (the front face, the outer part of the free edge: the real door is ~3.2 cm, the rest of the model its
# bins) and its inner part except round the hand. 11-08 on cuRobo: the arm's spheres never reached a 3 cm panel while
# pushing, 15 mm into a 5 cm one ...
PUSH_DOOR_WINDOW_M = (0.08, 0.25, 0.15)  # ... the window: this far toward the hinge and toward the free edge from the
# hand, and up and down (11-08: 5 cm toward the hinge reached the inner part by 19 mm; these, nothing)
PUSH_OPEN_POINT_DEG = 2  # ... held in the door's frame, turned about the hinge (a pose every this much door) ...
PUSH_OPEN_AWAY_M = 0.10  # ... then straight off the inner face this far (to the right): out of the planner's padding
# round the door (6 cm left the hand inside it, so nothing could be planned from there) ...
PUSH_OPEN_FINISH_U = 0.5  # ... then to in front of the fridge, facing into it, upright (as the grasp): this fraction of
# the door's width from the hinge (the middle of the opening) ...
PUSH_OPEN_FINISH_OUT_M = (0.25, 0.18, 0.12)  # ... this far out from the closed door's face (tool point; the first that
# plans: nearer the arm, the elbow (J4) runs into its limit facing forwards), at the door's mid-height ...
PUSH_OPEN_FINISH_CLEAR_M = 0.03  # ... part of the push's one leg (no stop, no blend), the hand's real box this clear of
# the open door and its handle all the way. If it doesn't plan, the push-open ends at the step-off
# the real 2F-85 (open) in the tool frame, from its collision meshes in the driver's URDF (measured 11-06): fingertips
# +-8.0 cm across (tool x), the body +-3.8 cm thick (tool y; the bracelet +-4.2 cm), the tips 1.4 cm past the tool
# frame, the bracelet's back 21 cm behind it. Where the hand meets the door when it pushes (cuRobo's spheres are
# bigger: it keeps them 2 cm off everything else)
HAND_BOX_TOOL_M = ((-0.080, 0.080), (-0.042, 0.042), (-0.21, 0.045))
# after the pull and the back-off off the handle, the push may need the wheelchair moved first (the open then stops and
# asks for it; an open goal with chair_moved pushes). Tried in this order, the first after which the push plans:
CHAIR_MOVES_M = (0.0, 0.05, -0.05, 0.10, -0.10, 0.15, -0.15, 0.20)  # + back (away from the fridge), - forward
CHAIR_MOVE_STEPS = 3  # while the chair moves, the arm (held) must clear the door at this many points along the move

# --- close: the hand pushes the open door round toward shut with whatever it holds, never tilting: cuRobo routes from
# where the hand is to the swing start (past the hinge, beside the open door), the swing follows the arc about the
# hinge, spiralling in, until the door is at CLOSE_TO_DEG (it swings shut by itself), and it backs straight out ---
CHAIR_BACK_M = 0.30  # what a close goal's chair_back used to default to (the goal now leaves it to the plan) ...
# ... without a chair_back, the close tries these wheelchair moves (m, + back, - forward), smallest first, out to the
# base's base_max_distance, and moves the chair by the first whose plan passes every gate
CLOSE_CHAIR_MOVES_M = tuple(
    sorted({round(s * k * 0.05, 2) for k in range(8) for s in (1, -1)}, key=lambda m: (abs(m), m < 0))
)
CHAIR_BACK_DIR = (1.0, 0.0)  # ... which moves the recorded fridge this way in the arm's frame (arm-base x = the
# chair's forward): the node's chair_back_dir parameter. The plan uses the door from the open, shifted by this
CLOSE_TO_DEG = 45.0  # swing the door shut until it is open this far, then back out (no torque push)
CLOSE_SPEED = 0.7  # the close goal's default speed: slower, the swing doesn't carry the door shut from CLOSE_TO_DEG
CLOSE_Z_MARGIN_M = 0.08  # without a push_z the close tries swing heights within the door, this far in from its top and
CLOSE_Z_STEP_M = 0.05  # bottom, every this far, nearest the hand first ...
CLOSE_Z_TRIES = 3  # ... at most this many
CLOSE_YAWS_DEG = (0, 45, -45, 90, -90, 135, -135, 180)  # the close may turn the hand about vertical by these (tried
# in order, at each height): its tilt kept, so a held container stays upright
CLOSE_START_GAP_M = 0.06  # the swing start: the hand this far off the open door (sideways past the hinge until it is)
CLOSE_POINT_DEG = 2.0  # the swing: a pose every this much of the hand's way round the hinge
CLOSE_HAND_HALF_M = 0.04  # the hand's half-width along the door: the swing must end clear of the handle by ...
CLOSE_HANDLE_MARGIN_M = 0.02  # ... this
CLOSE_RETREAT_M = (0.20, 0.15, 0.10)  # back straight out this far after the swing (the first that plans, at most
# the goal's retreat_dist)
CLOSE_MAX_TILT_DEG = 2.0  # the close never tilts the hand -- no roll, no pitch; turning about vertical is fine (it may
# hold a container). Every leg is planned holding the tilt (HOLD_LEVEL, which the planner checks to its own 2 deg) and the
# joined trajectory is checked again: within this of where it starts, or it refuses
CONTAINER_M = 6.5 * INCH  # the container held for the close (measured 10-24): ~a 6.5 in diameter sphere (11-06)
CONTAINER_CLEARANCE_M = 0.5 * INCH  # safety margin the container must keep from the door/handle/body (11-06)
CLOSE_SETTINGS = dict(  # the close's fixed settings; a goal may set the GOAL_SETTINGS among them
    push_radius=0.30,  # the hand's distance from the hinge during the swing (handle at ~37 cm)
    push_radius_end=0.28,
    side_past_hinge=0.08,  # the swing start, sideways past the hinge
    retreat_dist=0.20,
    # 11-06: the container is a 6.5 in sphere that sticks out ~6.5 in past the gripper tips; it is not in the
    # collision model. The plan keeps the hand back by that reach + the 0.5 in margin instead: the door is planned
    # shifted this far in x, so the container -- not the hand -- reaches the door
    hold_offset_x=-(CONTAINER_M + CONTAINER_CLEARANCE_M),
)

# --- what depends on the arm's mount (the node's mount_tilt_deg). Everything else plans in arm_mount: level, at the arm
# base's origin. A 30-deg lean moves the shoulder ~14 cm forward, so the arm reaches further and needs the fridge
# further away. 0: grasps 0.87-0.92 m out on the arm (09-20), placement from sim. 30: sim only (flatbot model, handle
# ~0.31 m above the arm base, door square to the chair). ---
MOUNTS = {
    0: dict(
        max_reach=0.94,
        placement_x=(0.50, 0.56),
        placement_y=(-0.45, -0.39),
        push_open_dz_m=-0.10,  # the push 10 cm below the handle
        grasp_pitches_deg=(15, 25),  # 15-45 only (GRASP_PITCHES_DEG)
    ),
    30: dict(
        max_reach=1.10,
        # where open and close both planned with the PyBullet planner (11-05 sweep); not yet re-swept with cuRobo
        placement_x=(0.82, 0.90),
        placement_y=(-0.44, -0.16),
        push_open_dz_m=0.10,  # 10 cm ABOVE the handle
        grasp_pitches_deg=(15, 30, 45),  # 15-45 only (GRASP_PITCHES_DEG)
    ),
}
PLACEMENT_X, PLACEMENT_Y = MOUNTS[0]["placement_x"], MOUNTS[0]["placement_y"]  # configure sets them
PLACEMENT_YAW_DEG = (-10.0, 10.0)


def configure(data_dir, mount_tilt_deg=0):
    """Point door_common's paths at `data_dir`, name the appliance and pick the mount's placement. No door-size
    defaults: every fridge door file carries its own (open_door.configure's DOOR_MODEL)."""
    global PLACEMENT_X, PLACEMENT_Y, PUSH_OPEN_DZ_M, GRASP_PITCHES_DEG
    if mount_tilt_deg not in MOUNTS:
        raise ValueError(f"mount_tilt_deg must be one of {sorted(MOUNTS)}, not {mount_tilt_deg!r}")
    PLACEMENT_X, PLACEMENT_Y = MOUNTS[mount_tilt_deg]["placement_x"], MOUNTS[mount_tilt_deg]["placement_y"]
    PUSH_OPEN_DZ_M = MOUNTS[mount_tilt_deg]["push_open_dz_m"]
    GRASP_PITCHES_DEG = MOUNTS[mount_tilt_deg]["grasp_pitches_deg"]
    dc.configure_paths(data_dir)
    dc.configure(APPLIANCE="fridge")
