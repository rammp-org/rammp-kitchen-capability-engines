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

# --- the start pose (action go_to_start, sent before an open): the web app's Fridge pose, the start every fridge sim
# plans from (fridge-grasp-sims: FRIDGE_START), clear of the planner's wheelchair and occupant layers. Joints J1..J7
# (deg); the node's start_joints_deg parameter, per robot ---
START_JOINTS_DEG = (103.04, -30.7, -169.63, -123.05, 101.64, 110.04, 2.5)  # 11-12: the start of the first open that executed through (run
# 20261112_194816_open, flatbot); was the web app's Fridge pose (98.8, 10.2, 179.5, -94.7, 98.7, 100.2, 16.1)
START_REACHED_DEG = 2.0  # every joint already this close (self.arm's GO_TO_TRACK_DEG): nothing moves

# --- grasp: the microwave's corrections on the Jetson rig, the values every fridge run so far used (10-22, 10-24: the
# door files' handle-to-grasp is 4.8 cm); nothing has checked them on this handle ---
# along the approach axis (negative = past the detected handle). 11-05: the microwave's 0.088 grabbed too deep on
# the fridge handle -> 3.8 cm (1.5 in) shallower
# 11-09: 0.126 stopped the fingertips 3-4.6 cm SHORT of the bar (both executed runs missed it). Measured: the
# fingertip ends are 4.3 cm past arm_tool_frame (robotiq finger_tip mesh reaches 5.1 cm past the tip link, which is
# 0.8 cm behind the tool frame), not door_common's FINGERTIP_PAST_TOOL 6.2 cm; the detected bar front was right
# to <0.5 cm (fingers teleoped onto it: /fridge_data/true_handle_20261109.json). A level grasp now puts the
# fingertips 3 cm past the bar's front (tool = bar front - 1.3 cm). Backup constants.py.bak_20261109.
# 11-13: - 0.014 for the sensor puck (rammp-deployments af2396b, #149): the puck was on the arm when 0.053 was
# tuned, but the model put arm_tool_frame 0.120 m past the flange, so every grasp went 14 mm deeper than planned
# and 0.053 held back those 14 mm. The model now has arm_tool_frame at 0.134 m and parks the flange 14 mm further
# back by itself, so 0.053 would grasp 14 mm shallower than the grasp that worked: take the 14 mm out again.
GRIP_EXT = 0.053 - 0.014
VERTICAL_CORR = 0.0  # world z
GRASP_X_ADJUST = 0.040  # along the approach axis
GRASP_FORCE = 0.5  # a normal close (the microwave's 0.25 is loosened on purpose, to let its bar turn)
# the open's grasp_offset_m (up from the middle of the bar) keeps the middle of the finger pads at least this far in
# from either end of handle_length_m: half the 2F-85 pad's ~2.2 cm width, plus ~2 cm for the detected middle being off
GRASP_END_MARGIN_M = 0.03

# --- which grasp the open uses (the node's `grasp` parameter; configure sets GRASP): "side" (side_grasp.py, this
# branch's default: in along the door from the hinge side, one finger behind the bar) or "front" (fingers across the
# bar, straight in from in front of it: feature/15-fridge's). Everything after the grasp is the same code ---
GRASPS = ("side", "front")
GRASP = "side"
# --- the side grasp (side_grasp.py). Its pads' place is the one that ran on the arm on 11-06 (89504f3's
# side_grasp_poses, with the then GRIP_EXT 0.126 - GRASP_X_ADJUST 0.040 - GRASP_PADS_PAST_TOOL 0.042 - SIDE_DEEPER_M
# 0.055), written out because GRIP_EXT has changed since (the front grasp's, 11-09). It was found with the SCENE
# camera's handle point; the wrist camera's (detect_camera wrist, the default) is unchecked for it: watch where the
# fingers close ---
SIDE_INSERT_DEG = 70.0  # the hand slides in this far off the door normal: 20 deg off the face, from in front of it
SIDE_PADS_INTO_M = 0.011  # the pads' line: this far behind the detected handle point, into the door ...
SIDE_PADS_PAST_BAR_M = 0.05  # ... and the pads this far further along the approach (11-06: 17 cm in from a waypoint
# 12 cm short of the bar: the fingers wrap the bar deep in the hand)
SIDE_TILTS_DEG = (0, -10, 10, -20, 20, -30, 30, -40, 40)  # nose-down (+) / nose-up (-) tilts about the finger axis,
# tried in this order -- level first (what ran on the arm), then out to +-40 (user, 11-10). The pads stay put
SIDE_SLIDE_M = (0.15, 0.12, 0.09)  # line up this far straight back along the (level) approach from the grasp, then
# slide in; the first that plans. 0.15: the fingertips run 8 cm straight before the bar (11-06's slide); 0.09: 2 cm
SIDE_HAND_LAGS_DEG = (25, 30, 20, 15, 10, 5, 0)  # in the swing the hand turns about the bar this much less than the
# door, the first whose swing and back-off plan (the bar turns in the closed gripper, held compliant: the side branch's
# hand_turns, 11-06 on the arm; turned fully with the door, J6 runs into its limit). In this order: 25-30 is what
# planned in the 11-10 sims (11-09 detections), smaller turns rarely did -- each one tried costs a swing
SIDE_SWING_CHECK_DEG = 6.0  # the swing is checked against the door every this much (side_grasp.swing_worlds): the door
# where that stretch starts, so at most this far behind the hand (~4 cm at the handle, less toward the hinge)
SIDE_BACK_OFF_M = 0.07  # off the handle: back along the hand's own axis this far -- the bar sits ~1 cm past the tool
# point, the fingertips 4.3 cm: 3.5 cm frees it, + 3.5 cm -- (further bends the wrist, J6, into its limit) ...
SIDE_BACK_OUT_M = 0.08  # ... then (cuRobo's path) this far straight out from the door's face (side branch, 11-06: "then
# out from the open door's face"), clear of cuRobo's padding round the door ...
SIDE_UNBEND_DEG = 45.0  # ... the hand turned about vertical on the way, up to this far, toward the way the arm reaches
# (the wrist, J6, is near its limit after the swing) ... then (cuRobo) to where the front grasp's back-off ends
SIDE_CONFIGS = 2  # if no grasp gives a whole open, each grasp is tried again from at most this many other arm
# configurations (side_grasp.config_seeds), the most room at the wrist (J6) first
SIDE_WINDOW_HINGE_M = 0.40  # sliding in, holding and backing off, the door within this far toward the hinge from the
# handle (and HAND_WINDOW_U_M past it, HAND_WINDOW_Z_M up and down) is in cuRobo's world only as its inner part ...
SIDE_FACE_RECESS_M = 0.05  # ... its face this much deeper than the model's: cuRobo's 2 cm padding and fat spheres
# call the finger behind the bar a collision with the real face (side branch, 11-06: its goal 2 cm out of a face 2 cm
# deeper still collided, 3 cm planned)

# --- how the motions are planned (planning.py, world.py): free-space legs by cuRobo, in a collision world of the fridge
# (FridgeModel: cabinet, door at its angle, handle) plus the planner's own chair layers (scene camera included), the arm's
# configuration cuRobo's to choose; legs on or along the door by local IK along their exact path, checked by cuRobo;
# each motion one trajectory joined by cuRobo. The poses below are the only places a motion is told to pass ---
GRASP_PITCHES_DEG = (25, 30, 40, -25, -30, -40)  # the grasp's tilts tried, + nose-down, - nose-up (configure sets
# them per mount), each with the hand either way up (fingers across the bar both ways): the first of these whose whole
# open plans is used. Only 25-40 deg either way: under 15 the fingers don't grip the bar firmly enough to pull the door
# (user, 2026-11-08); 11-09 (user): at least 25, and over 40 runs the wrist camera into the handle; 11-10 (user):
# nose-up 25-40 too (-15 not valid)
PREGRASP_STANDOFF_M = (0.12, 0.08)  # line up this far straight out from the grasp, then straight in (handle out of
# the world for that leg only); the first that plans
ALT_CONFIG_SEEDS = (("J3", -60.0), ("J3", 60.0), ("wrist", 0.0))  # if no grasp's swing plans from the configuration
# cuRobo picks from the start joints, these: the start joints with the elbow (J3) turned, or the wrist flipped (J5+180,
# -J6, J7+180), as cuRobo's start for planning the line-up pose -- other arm configurations at the same grasp
SWING_CANDIDATES = 3  # what follows the swing is planned for at most this many grasps whose approach + swing plan
PLAN_BUDGET_S = 240.0  # the time one search may take (the whole plan, or a re-plan after the arm or the base moved:
# the time spent moving never counts); past it the search refuses. 120 until 11-09: a whole push-open search that ends
# in a real refusal took 117 s in sim (11-09 detections, the Jetson's GPU), one that planned 98 s: twice that, so a
# search ends with an answer, not out of time
HAND_WINDOW_U_M = 0.15  # onto and off the handle: the door slab within this of the hand, along the door ...
HAND_WINDOW_Z_M = 0.15  # ... and up or down, is out of cuRobo's check (the handle with it): the hand is there
SWING_DEG = 90  # the full pull (push_open false) without a swing_deg. Exactly this swing or the run refuses: no smaller
# fallback

# --- how the open moves (the side grasp's 11-06 hardware runs, side_run/; run_general.py): the swing held compliant, the
# gripper closing as the pull starts, the swing stopping at its angle, then the gripper opening, everything after the
# release one motion ---
PULL_GAINS = {  # joint impedance (kinova-gen3-driver gains.h: kq <= 400 N m/rad, torque_limit <= the URDF's effort
    # limits), critically damped: the swing and everything after the release run under it. The shoulder and elbow
    # (j1-j4) at the driver's stiffest, to carry the door; the wrist (j5-j7) at the driver's own stiff profile, 60. At
    # 400 the wrist, the hand clamped on a handle that the door holds to its own arc, turned any mismatch between the
    # planned arc and the door's into torque: 400 N m/rad x 0.9 deg = 6 N m, the driver's contact stop at its loose
    # profile (j5-j7). 11-13: two of three pulls stopped -11 "unexpected contact" late in the swing, j6 ~6 N m off the
    # pull that finished, j1-j4 well inside theirs. At 60 the same mismatch is ~0.9 N m.
    "kq": (400.0, 400.0, 400.0, 400.0, 60.0, 60.0, 60.0),
    "zeta": 1.0,
    "torque_limit": (39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0),
}
PULL_PATH_TOL_RAD = 0.15  # the swing's tracking limit (rad, per joint)
GRIP_DURING_SWING = True  # the gripper closes on the bar as the swing starts (the 2F-85 takes ~1 s), no stop on the
# handle; False: it closes first, then the swing
# the release (user, 11-09): the swing comes to a STOP at exactly swing_deg, the gripper then opens with the hand still
# (confirmed open before anything else moves) -- the hand never pulls the door on past the angle asked for. Until 11-09
# the pull ran on 8 deg further while the gripper opened (RELEASE_ARC_DEG): the door ended ~68 deg on a 60 deg pull,
# was let go still moving and swung on into the fingers as they backed off. Stopped while held, it has no momentum left
GRIP_OPEN_SPEED = 0.2  # the gripper opens this slowly
AFTER_RELEASE_PATH_TOL_RAD = 0.25  # back off + round the door + the push, one motion under PULL_GAINS: this tracking
# limit (free space until the push; impedance lags the path more the faster it goes: 11-06 at ~1 rad/s, past 0.15 rad)

# --- release: once the gripper is open, back straight out of the handle (the first leg of what follows the swing) ---
BACK_OFF_M = 0.12  # 0.08 until 11-09: with the deeper grasp (GRIP_EXT 0.053) it left the hand 22 mm inside the
# handle's padding in cuRobo's world, so no route to the push could start (sim, the 11-09 detections)

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
# ... coming to rest at exactly PUSH_OPEN_TO_DEG (the door isn't held: a hand still moving there lets it coast past; user,
# 11-09: it must land on the angle at any speed), its last PUSH_EASE_DEG at most PUSH_END_JOINT_SPEED (rad/s, the
# fastest joint) whatever the goal's speed. Not yet tuned on the arm: if the door still coasts past 90, lower it ...
PUSH_EASE_DEG = 10.0
PUSH_END_JOINT_SPEED = 0.10
PUSH_OPEN_AWAY_M = 0.10  # ... then, from rest, straight off the inner face this far (to the right): out of the planner's padding
# round the door (6 cm left the hand inside it, so nothing could be planned from there) ...
PUSH_OPEN_FINISH_U = 0.5  # ... then to in front of the fridge, facing into it, upright (as the grasp): this fraction of
# the door's width from the hinge (the middle of the opening) ...
PUSH_OPEN_FINISH_OUT_M = (0.25, 0.18, 0.12)  # ... this far out from the closed door's face (tool point; the first that
# plans: nearer the arm, the elbow (J4) runs into its limit facing forwards), at the door's mid-height ...
PUSH_OPEN_FINISH_CLEAR_M = 0.03  # ... in the step-off's leg (no stop between), the hand's real box this clear of
# the open door and its handle all the way. If it doesn't plan, the push-open ends at the step-off
# the real 2F-85 (open) in the tool frame, from its collision meshes in the driver's URDF (measured 11-06): fingertips
# +-8.0 cm across (tool x), the body +-3.8 cm thick (tool y; the bracelet +-4.2 cm), the tips 1.4 cm past the tool
# frame, the bracelet's back 21 cm behind it. Where the hand meets the door when it pushes (cuRobo's spheres are
# bigger: it keeps them 2 cm off everything else)
HAND_BOX_TOOL_M = ((-0.080, 0.080), (-0.042, 0.042), (-0.21, 0.045))
# after the pull and the back-off off the handle, the push may need the wheelchair moved first (the open then stops and
# asks for it; an open goal with chair_moved pushes). Tried in this order, the first after which the push plans:
CHAIR_MOVES_M = tuple(  # + back (away from the fridge), - forward: 0 to +-35 cm in 5 cm steps, smallest first (as the
    # close's CLOSE_CHAIR_MOVES_M; up to +20 / -15 cm until 11-11)
    sorted({round(s * k * 0.05, 2) for k in range(8) for s in (1, -1)}, key=lambda m: (abs(m), m < 0))
)
CHAIR_MOVE_STEPS = 3  # while the chair moves, the arm (held) must clear the door at this many points along the move

# --- close: the hand (and the container it may hold) pushes the open door round toward shut, never tilting. cuRobo routes
# from where the hand is to a push start behind the door -- tried at several distances from the hinge, furthest first
# (leverage) -- then the push follows the arc about the hinge at that distance until whatever part of the hand or the
# container touches first has the door at CLOSE_TO_DEG (it swings shut by itself), and it backs straight out. Only the
# push is scripted: it is contact, which cuRobo (every plan kept 2 cm off everything) can't plan. The door is where the
# door file says (the push-open leaves it at PUSH_OPEN_TO_DEG) ---
CHAIR_BACK_M = 0.30  # what a close goal's chair_back used to default to (the goal now leaves it to the plan) ...
# ... without a chair_back, the close tries these wheelchair moves (m, + back, - forward), smallest first, out to the
# base's base_max_distance, and moves the chair by the first whose plan passes every gate
CLOSE_CHAIR_MOVES_M = tuple(
    sorted({round(s * k * 0.05, 2) for k in range(8) for s in (1, -1)}, key=lambda m: (abs(m), m < 0))
)
CHAIR_BACK_DIR = (1.0, 0.0)  # ... which moves the recorded fridge this way in the arm's frame (arm-base x = the
# chair's forward): the node's chair_back_dir parameter. The plan uses the door from the open, shifted by this
CLOSE_TO_DEG = 45.0  # swing the door shut until it is open this far, then back out (no torque push)
CLOSE_PLAN_ATTEMPTS = 3  # the after-move close is planned afresh up to this many times before it refuses: cuRobo's
# routes are random and it doesn't see the held container (11-12: identical dry runs, one refused, one planned)
CLOSE_SPEED = 0.7  # the close goal's default speed: slower, the swing doesn't carry the door shut from CLOSE_TO_DEG
CLOSE_Z_MARGIN_M = 0.08  # without a push_z the close searches push heights over the whole door, this far in from its
CLOSE_Z_STEP_M = 0.05  # top and bottom, every this far (nearest the hand first)
CLOSE_YAWS_DEG = (0, 45, -45, 90, -90, 135, -135, 180)  # the close may turn the hand about vertical by these (tried
# in order): its tilt kept, so a held container stays upright. Any part of the hand or the container may push
CLOSE_PITCHES_DEG = (0, 45, 90)  # with an empty hand, also tipped nose-down this far (the container stays level): any
# part of the hand may push at any angle (user, 11-09), and a hand tipped down reaches behind the open door far more
# easily than a level one at the edge of the arm's reach (11-10 sim)
CLOSE_RADII_FROM_EDGE_M = (-0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20)  # without a push_radius, the push's distance
# from the hinge (the tool point's): the door's length less each of these, furthest out first -- leverage. Negative:
# past the free edge -- the hand's or the container's edge catches the door's edge (user, 11-10), and the push starts
# just round it rather than all the way behind the door
CLOSE_START_GAP_M = 0.02  # the push start: the hand and the container this clear of the door. cuRobo keeps its own 2 cm
# from it too (with spheres bigger than the hand); this checks the container, which cuRobo doesn't see (6 cm until
# 11-09, with no stated reason)
CLOSE_START_STEP_DEG = 2.0  # the push start: round the hinge past the open door, every this much ...
CLOSE_START_MAX_DEG = 175.0  # ... up to this (180: flat against the cabinet's side)
CLOSE_TRIES = 12  # per wheelchair move, at most this many push starts (that pass the geometry) are planned by cuRobo ...
CLOSE_TRIES_PER_RADIUS = 3  # ... at most this many at each distance from the hinge (furthest first), and only push
# starts the arm can reach with the hand turned that way (its own IK, CLOSE_IK_SEEDS seeds; none beyond the mount's
# max_reach): an unreachable distance or hand turn can't use up the tries (11-10: the 3 nearest the arm were hand turns
# the wrist can't make out there, all IK_FAIL, while one further along planned)
CLOSE_IK_SEEDS = 5  # ... the search's IK (close_door.search_starts: every start of a distance from the hinge from every
# seed, solved together) starts from the arm's joints and these, skipping a start whose wrist is further from the
# shoulder than the arm stretches ...
CLOSE_SEARCH_MAX_IK = 400  # ... at most this many push starts per distance ...
CLOSE_ARM_RADIUS_M = 0.06  # ... and keeps a start only if the arm there (its links as capsules this thick) is this
CLOSE_ARM_CLEAR_M = 0.01  # clear of the door and the cabinet: the hand alone clearing it isn't enough
CLOSE_POINT_DEG = 2.0  # the push: a pose every this much of the hand's way round the hinge
CLOSE_PUSH_END_RADIUS_M = 0.30  # the push draws in to this far from the hinge by the end (close_door.sweep)
CLOSE_RETREAT_M = (0.20, 0.15, 0.10)  # back straight out this far after the push (the first that plans, at most
# the goal's retreat_dist)
CLOSE_MAX_TILT_DEG = 2.0  # the close never tilts the hand -- no roll, no pitch; turning about vertical is fine (it may
# hold a container). Every leg is planned holding the tilt (HOLD_LEVEL, which the planner checks to its own 2 deg) and the
# joined trajectory is checked again: within this of where it starts, or it refuses
CONTAINER_M = 6.5 * INCH  # the container held for the close (measured 10-24): ~a 6.5 in diameter sphere (11-06) ...
CONTAINER_CLEARANCE_M = 0.5 * INCH  # ... which keeps this from the door and cabinet on the way to the push (11-06) ...
CONTAINER_CENTRE_TOOL_M = 0.043 + CONTAINER_M / 2  # ... held sticking out ~6.5 in past the fingertips (11-06), which
# are 4.3 cm past the tool frame (11-09): its centre this far along the tool's z, carried with the hand. cuRobo doesn't
# see it: the close checks it itself (the push start, the route there, the push), and lets it push. The fridge is never
# shifted to stand in for it (until 11-12 the whole fridge was planned 17.8 cm nearer the arm instead, which put the
# push-open's end pose inside the door, 11-11)
CLOSE_HOLDING_GRIPPER = 0.2  # the close carries the container exactly when the gripper reads more closed than this at
# its start (0 open, 1 closed: 11-11, 0.004 with an empty open hand, 0.965 holding the container); the open never does
CLOSE_SETTINGS = dict(  # the close's fixed settings; a goal may set the GOAL_SETTINGS among them
    push_radius=None,  # the tool's distance from the hinge in the push (m); None: CLOSE_RADII_FROM_EDGE_M, furthest first
    retreat_dist=0.20,
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
        # the box the cells below span: the handle x, y where the open planned (cuRobo sweep, next comment)
        placement_x=(0.65, 1.05),
        placement_y=(-0.30, 0.00),
        # every handle x, y (5 cm grid) where the cuRobo open (60 deg pull, then the push-open, a chair move between
        # them if it needed one) planned in sim, the handle 0.315 m above the arm base (as detected on the robot on
        # 11-09): the placement advice sends the wheelchair toward the nearest of them (open_door.placement_advice).
        # 11-10 sweep on bcdc46e + the 11-09 uncommitted changes (GRIP_EXT 0.053, BACK_OFF_M 0.12) with the tilts below
        # (+-25-40) and the push's chair move up to +-35 cm (it never needed more than 5 cm); every refusal re-checked
        # in a fresh planner process (fridge-grasp-sims: flatbot model with the arm base 8 in further forward on the
        # chair, the web app's Fridge start pose, door square to the chair); the close was not part of it
        placement_cells=(
            (0.65, -0.15), (0.65, -0.10), (0.65, -0.05), (0.70, -0.15), (0.70, -0.10), (0.70, -0.05),
            (0.75, -0.25), (0.75, -0.20), (0.75, -0.15), (0.75, -0.10), (0.75, -0.05), (0.80, -0.25),
            (0.80, -0.20), (0.80, -0.15), (0.80, -0.10), (0.80, -0.05), (0.80, 0.00), (0.85, -0.30),
            (0.85, -0.25), (0.85, -0.20), (0.85, -0.15), (0.85, -0.10), (0.85, -0.05), (0.90, -0.30),
            (0.90, -0.25), (0.90, -0.20), (0.90, -0.15), (0.90, -0.10), (0.90, 0.00), (0.95, -0.25),
            (0.95, -0.20), (0.95, -0.15), (0.95, -0.05), (0.95, 0.00), (1.00, -0.25), (1.00, -0.20),
            (1.00, -0.15), (1.00, -0.10), (1.00, -0.05), (1.05, -0.15), (1.05, -0.10),
        ),
        # ... of those, the cells where the open plans only with the push's wheelchair move: an open goal with
        # base_moves false (no wheelchair move planned) can't start there, and its placement advice leaves them out
        cells_needing_base_move=(
            (0.65, -0.15), (0.65, -0.10), (0.65, -0.05),  # the nearest row
            (0.70, -0.15), (0.75, -0.25), (0.90, -0.10),
        ),
        # the side grasp's (grasp: side): every handle x, y where the side grasp's open (60 deg pull, any tilt of
        # -40..40 in 10s, a wheelchair move of up to +-35 cm before the push if it needed one -- all but 2 of the 231
        # tilt / spot pairs that planned needed one, 15-35 cm back mostly) planned in sim, the handle 0.315 m above the
        # arm base: fridge-grasp-sims results/side_tilt_z0.315.jsonl (plots/front_vs_side_any_tilt_z0.315.png, 47
        # spots), 11-10, the side branch's own planner (89504f3 + the tilt and the chair move) on the real cuRobo core,
        # the same model and grid as the front grasp's; the close was not part of it
        side_placement_x=(0.65, 0.90),
        side_placement_y=(-0.50, -0.05),
        side_placement_cells=(
            (0.65, -0.45), (0.65, -0.40), (0.65, -0.35), (0.65, -0.30), (0.65, -0.25), (0.65, -0.20),
            (0.65, -0.15), (0.70, -0.50), (0.70, -0.45), (0.70, -0.40), (0.70, -0.35), (0.70, -0.30),
            (0.70, -0.25), (0.70, -0.20), (0.70, -0.15), (0.75, -0.50), (0.75, -0.45), (0.75, -0.40),
            (0.75, -0.35), (0.75, -0.30), (0.75, -0.25), (0.75, -0.20), (0.75, -0.15), (0.75, -0.10),
            (0.75, -0.05), (0.80, -0.50), (0.80, -0.45), (0.80, -0.40), (0.80, -0.35), (0.80, -0.30),
            (0.80, -0.25), (0.80, -0.20), (0.80, -0.15), (0.80, -0.10), (0.80, -0.05), (0.85, -0.40),
            (0.85, -0.35), (0.85, -0.30), (0.85, -0.25), (0.85, -0.20), (0.85, -0.15), (0.85, -0.10),
            (0.85, -0.05), (0.90, -0.40), (0.90, -0.35), (0.90, -0.30), (0.90, -0.20),
        ),
        push_open_dz_m=0.10,  # 10 cm ABOVE the handle
        grasp_pitches_deg=(25, 30, 40, -25, -30, -40),  # 25-40 either way (GRASP_PITCHES_DEG): nose-down first; 11-09:
        # 45 ran the wrist camera into the handle
    ),
}
PLACEMENT_X, PLACEMENT_Y = MOUNTS[0]["placement_x"], MOUNTS[0]["placement_y"]  # configure sets them
PLACEMENT_CELLS = MOUNTS[0].get("placement_cells")  # ... and this: None, the box above is the placement
CELLS_NEEDING_BASE_MOVE = MOUNTS[0].get("cells_needing_base_move", ())  # ... and this
PLACEMENT_YAW_DEG = (-10.0, 10.0)

# The UI's panels (node.py's UI_ACTIONS): the goal keeps the arm, gripper, base and the UI while one waits, so both
# waits are bounded. A notice (Done: done, or failed) waits this long before the goal ends anyway ...
UI_NOTICE_TIMEOUT_S = 30.0
# ... and a question (Continue after a wheelchair move, or try again) this long before the goal fails no_answer
UI_PROMPT_TIMEOUT_S = 600.0


def configure(data_dir, mount_tilt_deg=0, grasp="side"):
    """Point door_common's paths at `data_dir`, name the appliance and pick the mount's and the grasp's placement. No
    door-size defaults: every fridge door file carries its own (open_door.configure's DOOR_MODEL)."""
    global PLACEMENT_X, PLACEMENT_Y, PLACEMENT_CELLS, CELLS_NEEDING_BASE_MOVE, PUSH_OPEN_DZ_M, GRASP_PITCHES_DEG, GRASP
    if mount_tilt_deg not in MOUNTS:
        raise ValueError(f"mount_tilt_deg must be one of {sorted(MOUNTS)}, not {mount_tilt_deg!r}")
    if grasp not in GRASPS:
        raise ValueError(f"grasp must be one of {', '.join(GRASPS)}, not {grasp!r}")
    GRASP = grasp
    mount = MOUNTS[mount_tilt_deg]
    # the side grasp's own placement where a sim gave one; else the front grasp's (the level mount has no side sim)
    key = "side_" if grasp == "side" and "side_placement_x" in mount else ""
    PLACEMENT_X, PLACEMENT_Y = mount[f"{key}placement_x"], mount[f"{key}placement_y"]
    PLACEMENT_CELLS = mount.get(f"{key}placement_cells")
    # the cells base_moves false leaves out of the advice: the front grasp's only. The side grasp has no such list (its
    # sim mostly planned only with a wheelchair move), so with base_moves false its advice may name a cell whose open
    # then refuses -- before any motion
    CELLS_NEEDING_BASE_MOVE = mount.get(f"{key}cells_needing_base_move", ())
    PUSH_OPEN_DZ_M = MOUNTS[mount_tilt_deg]["push_open_dz_m"]
    GRASP_PITCHES_DEG = MOUNTS[mount_tilt_deg]["grasp_pitches_deg"]
    dc.configure_paths(data_dir)
    dc.configure(APPLIANCE="fridge")
