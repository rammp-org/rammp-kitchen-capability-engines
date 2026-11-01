"""The microwave's values for door_common: what is true of this microwave and this rig, not of doors."""

from door_common import common as dc

BENCH_DATA_DIR = "/tmp/microwave"  # the bench scripts' data_dir (the node's is its data_dir parameter)
YOLO_MODEL = "yolo26s.pt"  # dev stand-in (the node's yolo_model parameter): the demo uses a different model

# --- grasp (tuned on the arm, see NOTES.md) ---
GRIP_EXT = 0.07  # along the approach axis (negative = past the detected handle). Was -0.010 (rchi-cpu-5);
# Jetson+sheppy+Capybara calib 10-07 touch check: planned grasp 9.8 cm too deep (0.088); snorlax 10-28: 0.100
VERTICAL_CORR = 0.0  # world z. 0.025 (rchi-cpu-5) -> 0.046 (10-07 touch) -> 0: the detector now returns the
# MIDDLE of the bar, not a point near its bottom end
GRASP_X_ADJUST = 0.040  # along the approach axis

HINGE_BEHIND_FACE = 0.016  # real pivot vs the detected hinge (the door's visible edge): this far behind the door
HINGE_PAST_EDGE = 0.021  # face and this far beyond the edge, away from the handle. Measured 10-07 from the
# hand's path in a stiff 50-deg swing (circle fit rms 0.6 mm; r 36.0 vs 33.7 cm)

# Park ("end") pose for the open and close tasks, joints in rad. Hand-placed by teleop on the Jetson
# rig, 2026-10-07 (was park_pose.json v4). HARDCODED FOR NOW: per-robot, so this can change -- likely
# to a fragment param or one of the driver's presets (go_to_preset).
PARK_JOINTS = [
    0.6047337092622715,
    0.6574053764067224,
    -2.7310591810191354,
    -2.245420837282717,
    1.0414940588270571,
    1.366535599611345,
    1.2981626670490867,
]

# door-size defaults: the door model of a door file without its own (common.door_dims)
DOOR_T = 0.04  # door thickness
DOOR_PAST_HANDLE = 0.06  # free edge beyond the handle. Was 0.122 (detector span 47.2 - 35.0 cm), but that
# span includes the control panel: on 09-25 the hand passed the 90-deg door's edge
# at x ~0.27, which only fits a door ending <= ~6 cm past the handle.
DOOR_Z = (0.25, 0.80)  # unmeasured -> deliberately generous (fallback: no closed_handle in the door file)
DOOR_Z_HANDLE_Z = 0.537  # closed-handle z when DOOR_Z was set (09-23); the model keeps DOOR_Z's extent
# relative to the handle, so a lower/higher microwave (09-28: handle ~0.26) is covered
BODY_DEPTH = 0.40


def configure(data_dir):
    """Point door_common's paths at `data_dir` and give it the values above."""
    dc.configure_paths(data_dir)
    dc.configure(
        APPLIANCE="microwave",
        PARK_JOINTS=PARK_JOINTS,
        DOOR_T=DOOR_T,
        DOOR_PAST_HANDLE=DOOR_PAST_HANDLE,
        DOOR_Z=DOOR_Z,
        DOOR_Z_HANDLE_Z=DOOR_Z_HANDLE_Z,
        BODY_DEPTH=BODY_DEPTH,
    )
