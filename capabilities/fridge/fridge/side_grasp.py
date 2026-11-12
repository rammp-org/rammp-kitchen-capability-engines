"""The side grasp (the node's `grasp: side`, this branch's default): the hand comes in along the door from the hinge
side of the handle, one finger between the bar and the door, and closes across the bar from its front and back.

Seen from above it is the front grasp's hand turned 90 deg about the vertical (away from the hinge), then turned back
SIDE_INSERT_DEG off the door normal (20 deg off the door's face, coming from in front of it): the approach (tool z)
runs along the door toward the free edge, the fingers close along the door normal. The hand slides in along that
approach (level, as the front grasp's straight-in) from a line-up point SIDE_SLIDE_M back, so the fingers travel
along their own axis and every point before the grasp is shallower than the grasp. Tilted (SIDE_TILTS_DEG) nose-down
(+) or nose-up (-) about the finger axis, the pads stay where the level hand puts them.

The pads' place on the bar is what ran on the arm on 11-06 (side_grasp.side_grasp_poses on 89504f3, which used the then
GRIP_EXT 0.126): SIDE_PADS_INTO_M behind the detected handle point, then SIDE_PADS_PAST_BAR_M further along the
approach. It is written out here because GRIP_EXT (the front grasp's) has changed since.

cuRobo keeps 2 cm (and its spheres) off every box, so it would call the finger behind the bar a collision with the door
and the handle. Where the hand slides in, holds and backs off, its worlds leave the handle out and, in a window from the
hand's side of the hinge to past the handle, keep only the door's inner part: its face SIDE_FACE_RECESS_M deeper than
the model's (hold_world). The swing is checked against the door where it is at each step (swing_worlds).

In the swing the hand turns about the bar up to SIDE_HAND_LAGS_DEG less than the door (swing_poses): turned fully with
it, the wrist (J6) runs into its limit. After the release the hand backs off along its own axis, out from the door and
round to where the front grasp's back-off ends (push_open.back_off_goals): what follows is the front grasp's.
"""

import numpy as np
from door_common import open_door as shared
from door_common.common import Refused
from scipy.spatial.transform import Rotation as R

from . import constants as F


def side_frame(n, hinge, h):
    """(into the door, along the door toward the hinge, hinge sign): horizontal unit vectors for outward door normal
    `n`, hinge point `hinge` and handle `h`. The sign is +1 when the hinge is on the handle's left as seen from the arm
    (toward the hinge = n x up), -1 on its right."""
    out = np.array([n[0], n[1], 0.0], float)
    out /= np.linalg.norm(out)
    left = np.cross(out, [0.0, 0.0, 1.0])
    sign = 1 if float(np.dot(np.asarray(hinge, float) - np.asarray(h, float), left)) >= 0 else -1
    return -out, sign * left, sign


def front_hand(orient, n):
    """The level front grasp's hand (door_common.open_door.grasp_pose at no tilt, without its gates): the detector's
    handle orientation flipped to approach, yawed to face the door square."""
    quat = R.from_quat(orient) * R.from_euler("y", np.pi)
    z0, into = quat.as_matrix()[:, 2], -np.asarray(n, float)
    yaw = float(np.arctan2(z0[0] * into[1] - z0[1] * into[0], z0[0] * into[0] + z0[1] * into[1]))
    return R.from_euler("z", yaw) * quat


def grasp_pose(h, orient, n, hinge, tilt_deg):
    """The side grasp for handle `h` (its height corrected already), handle orientation `orient`, outward door normal
    `n`, hinge `hinge`, tilted `tilt_deg` nose-down about the finger axis: (tool position, quat xyzw, pads point).
    Raises Refused if the hand's frame doesn't come out as the side grasp's (a detection the geometry can't hold)."""
    into, toward_hinge, sign = side_frame(n, hinge, h)
    # the front grasp's hand turned 90 deg about the vertical, away from the hinge: its approach (into the door) becomes
    # "along the door toward the free edge", its finger axis (along the door) the door normal
    rot = R.from_euler("z", -90 * sign, degrees=True) * front_hand(orient, n)
    approach = rot.as_matrix()[:, 2]
    if np.dot(approach, -toward_hinge) < 0.999:
        raise Refused(f"side grasp: the hand's approach {np.round(approach, 3)} isn't along the door -- refusing")
    off = np.radians(90.0 - F.SIDE_INSERT_DEG)  # the insert: along the door, turned this far into it
    want = -toward_hinge * np.cos(off) + into * np.sin(off)
    turn = float(np.arctan2(np.cross(approach, want)[2], np.dot(approach, want)))
    rot = R.from_euler("z", turn) * rot
    level = rot.as_matrix()[:, 2]
    if np.dot(level, want) < 0.999:
        raise Refused(f"side grasp: the hand's approach {np.round(level, 3)} isn't the insert {np.round(want, 3)}")
    pads = np.asarray(h, float) + into * F.SIDE_PADS_INTO_M + level * F.SIDE_PADS_PAST_BAR_M
    rot = rot * R.from_euler("x", tilt_deg, degrees=True)
    tool = pads - rot.as_matrix()[:, 2] * shared.GRASP_PADS_PAST_TOOL
    return tool, rot.as_quat(), pads


def unbend(pos, quat):
    """The hand at `pos` turned about vertical toward the arm's reach (from the arm base out to `pos`), by at most
    SIDE_UNBEND_DEG: the wrist (J6) bends less the more the hand points the way the arm reaches."""
    approach = R.from_quat(quat).as_matrix()[:, 2]
    a, r = np.asarray(approach[:2], float), np.asarray(pos[:2], float)
    if np.linalg.norm(a) < 1e-6 or np.linalg.norm(r) < 1e-6:
        return np.asarray(quat, float)
    turn = float(np.arctan2(a[0] * r[1] - a[1] * r[0], a @ r))
    turn = float(np.clip(turn, -np.radians(F.SIDE_UNBEND_DEG), np.radians(F.SIDE_UNBEND_DEG)))
    return (R.from_euler("z", turn) * R.from_quat(quat)).as_quat()


def bar_point(h, n):
    """Where the hand holds the bar (the pivot it turns about in the swing): SIDE_PADS_INTO_M behind the detected
    handle point `h`, at its height -- the pads go SIDE_PADS_PAST_BAR_M on past it, so the bar sits deep in the hand."""
    into = -np.array([n[0], n[1], 0.0], float)
    return np.asarray(h, float) + into / np.linalg.norm(into) * F.SIDE_PADS_INTO_M


def slide_direction(df, deg):
    """The way the hand slides in (horizontal unit vector), with the door open `deg`: along the door toward its free
    edge, turned 90 - SIDE_INSERT_DEG into it -- in the door's frame, as it was when the door was closed."""
    off = np.radians(90.0 - F.SIDE_INSERT_DEG)
    rot = df._rot(deg)
    d = (rot @ df.e) * np.cos(off) - (rot @ df.n) * np.sin(off)
    return np.r_[d, 0.0]


def swing_poses(cand, df, swing, lag, degs):
    """The tool poses of the swing at the door angles `degs` (to `swing`): the bar carried round the hinge with the
    door, the hand turning about it evenly to `lag` deg short of the door (the bar turns that much in the closed
    gripper, held compliant: side branch, 11-06 on the arm, so the wrist (J6) keeps room)."""
    bar0, tool0, q0 = (np.asarray(cand[k], float) for k in ("bar", "pos", "quat"))
    out = []
    for d in degs:
        bar = df.hinge + R.from_euler("z", df.sign * d, degrees=True).apply(bar0 - df.hinge)
        bar[2] = bar0[2]
        turn = R.from_euler("z", df.sign * (d - lag * d / swing), degrees=True)
        out.append((bar + turn.apply(tool0 - bar0), (turn * R.from_quat(q0)).as_quat()))
    return out


def door_reference(h, n):
    """The point the door file is built on (closed_grasp_pos): where the level FRONT grasp's tool would be, in front of
    the handle. The door's frame (DoorFrame: the handle's place along the door, the face) is the front grasp's, so the
    close and the push-open read a side grasp's door file as they read the front's."""
    into = -np.array([n[0], n[1], 0.0], float)
    into /= np.linalg.norm(into)
    return np.asarray(h, float) - into * (F.GRIP_EXT - F.GRASP_X_ADJUST)


def candidates(geom):
    """The side grasps tried, in order: SIDE_TILTS_DEG, each with the hand up (as on the arm on 11-06). Dicts {pitch,
    flip, h, pos, quat, side, ref}: `ref` the door file's reference (door_reference). A grasp past MAX_REACH is
    skipped, as the front grasp's are."""
    h = np.asarray(geom["h_raw"], float) + np.array([0.0, 0.0, shared.VERTICAL_CORR])
    n, hinge = np.asarray(geom["n"], float), np.asarray(geom["hinge"], float)
    front = front_hold(geom)
    out, far = [], []
    for tilt in F.SIDE_TILTS_DEG:
        pos, quat, _ = grasp_pose(h, geom["orient"], n, hinge, tilt)
        if shared.MAX_REACH is not None and np.linalg.norm(pos) > shared.MAX_REACH:
            far.append(f"{tilt:+.0f}")
            continue
        out.append(
            dict(
                pitch=tilt,
                flip=False,
                h=h,
                pos=pos,
                quat=quat,
                side=True,
                ref=door_reference(h, n),
                bar=bar_point(h, n),
                front=front,
            )
        )
    if far:
        from door_common.common import say

        say(f"  side grasp tilts {', '.join(far)} deg: the grasp is past the arm's reach ({shared.MAX_REACH} m)")
    return out


def front_hold(geom):
    """Where the FRONT grasp would hold the closed door's handle (its first tilt, GRASP_PITCHES_DEG[0], hand up), as
    [x y z qx qy qz qw]: the side grasp's back-off ends where the front's would, carried round with the door (push_open
    .back_off_goals)."""
    from door_common.common import quiet

    rot = front_hand(geom["orient"], geom["n"]) * R.from_euler("x", F.GRASP_PITCHES_DEG[0], degrees=True)
    try:
        with quiet():
            _, g = shared.grasp_pose(
                np.asarray(geom["h_raw"], float), geom["orient"], np.asarray(geom["n"], float), F.GRASP_PITCHES_DEG[0]
            )
        return [*map(float, g.position), *map(float, g.orientation)]
    except Refused:  # past MAX_REACH: the same pose, without door_common's gate
        h = np.asarray(geom["h_raw"], float) + np.array([0.0, 0.0, shared.VERTICAL_CORR])
        return [*map(float, door_reference(h, geom["n"])), *map(float, rot.as_quat())]


def name(cand, lag=None):
    return f"side grasp, tilt {cand['pitch']:+.0f} deg" + (f", the hand {lag:.0f} deg behind the door" if lag else "")


CONFIG_J1_DEG = (0.0, -30.0, 30.0)  # config_seeds: the start joints with J1 and J3 turned by these (the side branch's
CONFIG_J3_DEG = (
    0.0,
    -30.0,
    30.0,
    -60.0,
    60.0,
    -90.0,
    90.0,
    -120.0,
    120.0,
    180.0,
)  # GEN_J1/J3_OFFSETS_DEG, 11-06), each
# with the wrist flipped too
SAME_CONFIG_DEG = 10.0  # two configurations closer than this (every joint) are one


def config_seeds(q0):
    """[(name, joints)]: IK seeds round `q0` for the side grasp's other arm configurations, nearest first."""
    out = []
    for flip in (False, True):
        for d1 in CONFIG_J1_DEG:
            for d3 in CONFIG_J3_DEG:
                q = np.array(q0, float)
                q[0] += np.radians(d1)
                q[2] += np.radians(d3)
                if flip:
                    q[4], q[5], q[6] = q[4] + np.pi, -q[5], q[6] + np.pi
                out.append(
                    (f"J1 {d1:+.0f} J3 {d3:+.0f}{' wrist flipped' if flip else ''}", q, abs(d1) + abs(d3) + 180 * flip)
                )
    out.sort(key=lambda s: s[2])
    return [(name, q) for name, q, _ in out]


# --- what cuRobo checks the hand against while it is on (or sliding onto, or backing off) the handle ------------------


def window(model):
    """The part of the door (u0, u1, z0, z1; u along it from the hinge) the hand may be in front of while it slides in,
    holds the handle or backs off: from SIDE_WINDOW_HINGE_M toward the hinge from the handle to HAND_WINDOW_U_M past it,
    HAND_WINDOW_Z_M up and down from the handle."""
    df = model.df
    z = float(model.door.get("handle_mid_z", model.door["closed_handle"][2]))
    return (
        df.u_handle - F.SIDE_WINDOW_HINGE_M,
        df.u_handle + F.HAND_WINDOW_U_M,
        z - F.HAND_WINDOW_Z_M,
        z + F.HAND_WINDOW_Z_M,
    )


def hold_world(model, degs):
    """The door open at each of `degs` as the side-grasping hand may meet it, and the cabinet: the slab whole outside
    window(); inside it only the door's inner part, its face SIDE_FACE_RECESS_M deeper than the model's; no handle."""
    df = model.df
    front = df.t / 2
    out = model.body()
    for k, deg in enumerate(degs):
        out += model.slab(deg, tag=f"door{k}", window=window(model))
        out += model.slab(deg, tag=f"door{k}_recessed", v=(-front, front - F.SIDE_FACE_RECESS_M))
    return out


def swing_worlds(model, degs):
    """For each pose of the swing (the door open `degs` there, in order from the closed door): the (label, world)
    the way to it is checked against -- the door where it is when that stretch of SIDE_SWING_CHECK_DEG starts
    (hold_world): never ahead of the hand (the door opens toward the arm, so a door further open than the hand would be
    inside it), at most SIDE_SWING_CHECK_DEG behind it. The hand turns with the door, but the arm behind it doesn't:
    alongside the door, it could meet it. One world per stretch: each is a world load and a cuRobo check."""
    out, start, current = [], 0.0, None
    for d in degs:
        if current is None or d - start > F.SIDE_SWING_CHECK_DEG + 1e-9:
            start = out[-1][2] if out else 0.0
            current = (f"the door at {start:.0f} deg (the hand's window: its face recessed, no handle)",
                       hold_world(model, (start,)))
        out.append((*current, d))
    return [(label, world) for label, world, _ in out]
