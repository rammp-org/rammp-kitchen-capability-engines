"""The look: pointing the camera at the table before anything is known.

A turn, not a flight. Measured on the real planner from HOME (2026-09-15):
flying to the old fixed scan pose 3.89 s, re-orienting the tool in place
without travelling 4.01 s — both swing about 3 rad, because wrist-flat and
tool-down are different IK families — against 3.42 s for this, which moves
two joints. Defined against the arm's OWN rest pose rather than a bench
Cartesian pose, so it holds wherever the chair is standing.

WHAT MUST BE IN FRAME IS THE BOX TOP, not the table, and the whole of it:
the top sits dims.z nearer the camera, so it is seen through a
proportionally smaller window, and a top touching the image border is
refused (depth_source's border gate) because a clipped footprint cannot be
measured. Sizing the look by what it sees of the TABLE is how the first
field run missed a box (2026-09-15: the box stood at base [0.48, 0.12], the
look's lid-height frame reached x 0.55, and the top's far edge at 0.54 left
10 mm — the semantic model found it jammed against the frame edge while the
plateau geometry found nothing in 234 frames).

From HOME this puts the camera at [0.49, 0.00, 0.479] looking straight
down. At LID height the frame covers x 0.28-0.71 and y +/-0.38 (TF,
2026-09-15); the NARROW axis is x, because the image's long axis lies along
base y. With the sweep below that covers the placement band.

The lift does two jobs. Turning the wrist alone leaves the fingertips at
z 0.154 — above the pre-detection keep-out band (top z 0.105), but inside
cuRobo's collision activation distance, and the goal is refused outright
(IK_FAIL, live planner 2026-09-15: refused at lift 0.00 and 0.10, solved
from 0.15 up). Past that, every further radian of lift buys view. Turning
the wrist by the same amount again keeps the camera pointing straight
down, so the view stays nadir — the geometry the plateau is measured in.
"""

import math

ELBOW = 3  # joint_4, zero-based
WRIST_PITCH = 5  # joint_6, zero-based
# rest (tool forward, wrist flat) -> tool down
TILT_RAD = -math.pi / 2
# Elbow opening that lifts the hand clear of the keep-out band AND lifts
# the camera far enough to see a whole box top; the wrist takes the same
# angle back, so the tool still points down. 0.15 is where the goal starts
# planning at all, so this is not a floor — it is a view: 3.42 s against
# 2.74 s at 0.25, for 2.8x the lid-height area.
LIFT_RAD = 0.8
# Base-yaw arc each way when the look sees nothing. At the look the camera
# stands 0.49 m out from the base, so 0.6 rad carries the view centre to
# [0.41, -0.28] while keeping it nadir — reaching the corners of the
# placement band the bench world blocks before anything has been detected,
# and swinging the frame so its narrow axis falls somewhere new.
SWEEP_RAD = 0.6


_WRIST_PITCH_LIMIT = 1.95  # cuRobo joint limit 2.09 minus 0.10 rad margin, with 0.04 slack


def look_joints(rest, lift=LIFT_RAD):
    """`rest` with the hand lifted and the tool turned to look down.

    j6 is clamped inside the planner's joint-limit margin so the goal
    always plans on flatbot, where HOME j6 is -0.683 (vs +0.960 on the
    bench) and the unclamped formula would reach -3.054."""
    q = [float(v) for v in rest]
    q[ELBOW] += lift
    q[WRIST_PITCH] = max(-_WRIST_PITCH_LIMIT, min(_WRIST_PITCH_LIMIT, q[WRIST_PITCH] + TILT_RAD - lift))
    return q


def sweep_targets(look, arc=SWEEP_RAD):
    """Base-yaw targets that pan the look across the table, one each way.

    Together they cover [-arc, +arc] through the middle; the search stops at
    the first coarse fix, so the second is only reached when the first side
    was empty."""
    out = []
    for delta in (arc, -arc):
        q = [float(v) for v in look]
        q[0] += delta
        out.append(q)
    return out
