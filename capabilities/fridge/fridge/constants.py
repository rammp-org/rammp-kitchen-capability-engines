"""The fridge's values for door_common: what is true of this fridge and this rig, not of doors.

Everything not here is door_common's. MEASURED changes with the fridge and the wheelchair, so each of its values is also
a ROS parameter (lower case); the rest was tuned on the arm and needs code to change.
"""

from door_common import common as dc

INCH = 0.0254

# --- measured on the fridge being opened (per fridge: the node's parameters) ---
MEASURED = {
    # middle of the handle bar (where the fingers close) -> door's front surface, perpendicular to the door, gap behind
    # the bar included (measured 10-22; was 0.020, estimated by eye)
    "handle_centre_to_face_m": 0.053,
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
    "door_above_handle_m": 12.5 * INCH,
    "door_below_handle_m": 18 * INCH,
    "body_depth_m": 0.50,  # cabinet behind the door
    # The arm is mounted on the chair's left and the person sits just behind it, to its right (10-24: a close ran the
    # arm into the back of the chair). No part of the arm or gripper may be more than this far behind the arm base
    # (arm-base x) while on its right (y < 0). User, 10-24: ~8 in estimated, then 12, then 15; the real teleop start has
    # the forearm ~14 in behind the base: inside 12, clear at 15.
    "chair_keep_out_behind_m": 15 * INCH,
}

# --- grasp: the microwave's corrections on the Jetson rig, the values every fridge run so far used (10-22, 10-24: the
# door files' handle-to-grasp is 4.8 cm); nothing has checked them on this handle ---
GRIP_EXT = 0.088  # along the approach axis (negative = past the detected handle)
VERTICAL_CORR = 0.0  # world z
GRASP_X_ADJUST = 0.040  # along the approach axis
GRASP_FORCE = 0.5  # a normal close (the microwave's 0.25 is loosened on purpose, to let its bar turn)

# --- handle position gates: the microwave's defaults (y >= -0.50, z <= 0.65) would refuse the fridge handles of 09-20/
# 10-21 ---
PLAUSIBLE_Y = (-0.60, 0.60)
PLAUSIBLE_Z = (0.25, 0.75)

# --- swing ---
SWING_DEG = 90  # the swing without a swing_deg. Exactly this swing or the run refuses: no smaller fallback

# --- release: let go, then back straight out of the handle ---
BACK_OFF_M = 0.06

# --- after the full pull (the default open): back round the door to its inner side (open_door.plan_release_around), so
# the arm ends beside the open door, not stuck outside it. All of it is planned with the grasp + swing ---
AROUND_EDGE_CLEAR_M = (0.10, 0.15, 0.20)  # back along the door to this far past its free edge (tool point), tried in
# order; every link also stays >= MIN_CLEAR from the door model the whole way
AROUND_OUT_M = 0.15  # ... at least this far off the door's mid-plane on the outside while going back ...
AROUND_IN_M = 0.25  # ... round the edge to this far off it on the inside, turning to face the fridge ...
AROUND_END_M = (0.25, 0.30)  # ... and in to this far from the hinge along the door, this far off its inside: beside
# the open door, in front of the fridge opening

# --- the push-open option (push_open.py): pull a little holding the handle, let go, push the door the rest of the way,
# with the hand turned PARALLEL to the door: round its free edge at hand height (a fridge door is too tall to go over),
# in behind it with the fingers pointing along it toward the hinge, rolled vertical, and the side of the hand pushing
# the inner face. (Facing the hand into the cabinet, as the microwave's demo does, puts the gripper through the door.)
PUSH_OPEN_PULL_DEG = 50  # the pull. An offline sim search (10-24 detection, 4.2 in door) found parallel hand poses
# behind the door from 35 (2 poses, 2.8 cm clear) to 60; 50 has many (75 of 720, up to 7.3 cm clear)
PUSH_OPEN_TO_DEG = 90  # push the door on to this
PUSH_OPEN_OFF_HANDLE_M = BACK_OFF_M  # let go, then straight out off the handle this far ...
PUSH_OPEN_EDGE_CLEAR_M = 0.16  # ... along the open door to this far past its free edge (tool point) ...
PUSH_OPEN_GAP_M = 0.10  # ... round it to this far off the door's inner face (tool point), turning the hand ...
PUSH_OPEN_DZ_M = -0.10  # ... and going this far below the handle's height (the search's best poses) ...
PUSH_OPEN_AT_M = (0.40, 0.35, 0.30, 0.25)  # ... in along the inner face to this far from the hinge (tried in order) ...
PUSH_OPEN_APPROACH_DEG = (0, 15, -15, 30)  # ... the fingers this far off "along the door toward the hinge", toward
# pointing into the cabinet (tried in order, each with the tool's y up and down) ...
PUSH_OPEN_ROLL_DEG = 90  # ... rolled this far about the approach (90: fingers open vertically)
PUSH_OPEN_PRELOAD_M = 0.01  # the push: the side of the hand this far past touching the inner face (sim) ...
PUSH_OPEN_ARC_STEP_DEG = 10  # ... held in the door's frame, turned about the hinge in these steps

# --- close: door_common's push-close, reached by backing out in front of the open door's free edge and going left
# behind it, from further back, and only until the door is at CLOSE_TO_DEG (it swings shut by itself) ---
CHAIR_BACK_M = 0.30  # the wheelchair is moved back this far first ...
CHAIR_BACK_DIR = (1.0, 0.0)  # ... which moves the recorded fridge this way in the arm's frame (arm-base x = the
# chair's forward): the node's chair_back_dir parameter. The plan uses the door from the open, shifted by this
CLOSE_TO_DEG = 45.0  # swing the door shut until it is open this far, then back out (no torque push)
CLOSE_Z_MARGIN_M = 0.08  # without a push_z the close tries swing heights within the door, this far in from its top and
CLOSE_Z_STEP_M = 0.05  # bottom, every this far, nearest the hand first
CONTAINER_M = 6.5 * INCH  # the tupperware held for the close (measured 10-24): the hand is kept back by this much
CLOSE_MAX_TILT_DEG = 2.0  # the close never tilts the hand -- no roll, no pitch; turning about vertical is fine (it may
# hold a container). Every planned configuration must keep the tool's tilt within this of where it starts, or it refuses
CLOSE_SETTINGS = dict(  # replace door_common.close_door.CLOSE_DEFAULTS
    push_radius=0.30,  # the hand's distance from the hinge during the swing (handle at ~37 cm)
    push_radius_end=0.28,
    push_stop_short=0.0,  # the swing ends AT the CLOSE_TO_DEG door model ...
    push_max=0.0,  # ... and there is no final push
    keep_orientation=True,  # holding a container
    hold_offset_x=-CONTAINER_M,
)

# --- what depends on the arm's mount (the node's mount_tilt_deg). Everything else plans in arm_mount, which is level
# and shares the arm base's origin on every robot. The reach limits are measured from that origin, but a 30-deg lean
# moves the shoulder ~14 cm forward, so the same arm reaches further from it and its upper arm needs the fridge further
# away. 0: 09-20 grasps were 0.87-0.92 m out (door_common: 0.915 grasp / 0.90 swing), placement as passed in sim with a
# push-open after a 70-deg pull from the Fridge start pose. 30: sim only (10-24 door shifted, flatbot model, 90-deg open
# + route round the door + close), never run on the arm. ---
MOUNTS = {
    0: dict(max_reach=0.94, swing_max_reach=0.94, placement_x=(0.50, 0.56), placement_y=(-0.45, -0.39)),
    30: dict(max_reach=1.10, swing_max_reach=1.10, placement_x=(0.68, 0.76), placement_y=(-0.48, -0.38)),
}
PLACEMENT_X, PLACEMENT_Y = MOUNTS[0]["placement_x"], MOUNTS[0]["placement_y"]  # configure sets them
PLACEMENT_YAW_DEG = (-10.0, 10.0)


def configure(data_dir, mount_tilt_deg=0):
    """Point door_common's paths at `data_dir`, name the appliance and pick the mount's placement. No door-size
    defaults: every fridge door file carries its own (open_door.configure's DOOR_MODEL)."""
    global PLACEMENT_X, PLACEMENT_Y
    if mount_tilt_deg not in MOUNTS:
        raise ValueError(f"mount_tilt_deg must be one of {sorted(MOUNTS)}, not {mount_tilt_deg!r}")
    PLACEMENT_X, PLACEMENT_Y = MOUNTS[mount_tilt_deg]["placement_x"], MOUNTS[mount_tilt_deg]["placement_y"]
    dc.configure_paths(data_dir)
    dc.configure(APPLIANCE="fridge")
