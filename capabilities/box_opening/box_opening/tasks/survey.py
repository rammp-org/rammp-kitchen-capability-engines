"""The wrist survey: the container from the wrist camera, where the scene camera cannot see it.

Parked side-on to a kitchen counter the seated user cannot go under (the chair study's only
setup that reaches the box, 2026-11-06), the counter stands on the arm's side (+y in
arm_mount), and the scene camera, fixed and facing +x, does not see it: at x 0.3-0.7 m its view
reaches no further toward +y than y -0.36 to +0.04 (the scene study's frame-edge map), while
every box the side counter's sweep pressed stood at y 0.25-0.65. Until now that run stopped
with NO BOX before anything moved.

With the survey on (the node's wrist_survey, press_demo --wrist-survey; off by default), a run
whose scene camera finds no container looks for it with the wrist camera instead:

1. SURVEY POSES (survey_views): one or two poses for the wrist camera over the arm's side, chosen
   on this run from the arm's own kinematics and the body model, nothing about the room: the
   region a box may stand in on that side (SIDE_X_M, SIDE_Y_M) at any counter height in
   COUNTER_Z_M — the counter's height is unknown until seen, so a pose is scored on every height
   of the band at once. Candidate views (the camera's place, how far it tilts from straight
   down and toward where, which way the image's long side lies) are scored by the lids they show
   whole (`sees`) and the lids whose button the finder could check from there (`checks`: within
   the finder's incidence and range, wrist_finder.CHECK_*), over the cells the arm could press
   (a tool straight down over the lid inside the arm's reach, ArmChain.past_reach_m); the best
   are solved by the arm's IK, the elbow swung along its self-motion where that clears, and kept
   only where the arm stands clear of the chair and its user (runtime/body, strictly), clear of
   the COUNTER BAND by KEEP_ABOVE_M — every place a counter of the band could stand on the arm's
   side (its face no nearer than the chair's widest box there plus EDGE_GAP_M) — inside the
   planner's joint limits, and where its way there from the pose before, as a straight joint line,
   holds to the band and off the user as the flight's is held (Band.path_refusal). A second pose
   only where it adds SECOND_GAIN of the first's score and its joints fly from the first's within
   SECOND_MAX_S.
2. THE LOOK: each pose is planned by the planner as the reach-out's look is (the joint goal, the
   full world, TRANSIT_SPEED), its way there held to the counter band and off the user, every
   sphere of the arm (Band.path_refusal), flown, and the arm let stand still; SURVEY_FUSE_FRAMES
   still frames from there are fused per pixel (button_aim.StillDepth's fusion) into one frame,
   kept as a capture (wrist-survey-*).
3. THE FINDER (perception/wrist_finder.find_container) on every frame so far. A lid-sized top no
   view could check the button on, or a lid the view cut, is a place to LOOK AGAIN at
   (look_again_view: the camera LOOK_AGAIN_RANGE_M from its lid, at most LOOK_AGAIN_TILT_DEG off
   straight down, the arm no higher than the survey's own poses reached: over the box it may stand
   under an upper cabinet the frames looking down never show), at most LOOK_AGAIN_MAX places, then
   the finder once more on all the frames.
4. THE HAND-OFF: its fix (a SceneFix, source "wrist") is the reach-out's prior, as the scene
   camera's was — the tilted wrist look, the aim and the press unchanged — and its clouds the lid
   spot's map (the fix's table_points, as the scene camera's were); every survey frame's points
   are kept on the run's Ctx (ctx.survey_clouds) for what else needs the counter (the cabinet
   edge's keep-out). Nothing found, or two places that pass, is a STOP before anything is
   pressed: NO BOX — wrist survey: why.

A run with the orchestrator's selection (--instance-id) does not survey: the selection is one of
the scene camera's detections, and the wrist cannot tell which container the user selected
(docs/interface.md: the interface's gap).

Its heights are the wrist camera's own (wrist_finder's note: the lid's height above its counter
is right, both read low by the wrist mount's error, more the more the view tilts). The press goes
by the higher of the survey's top and the aim's, and the press's contact band (45 mm above, 25 mm
below) judges the rest: a lid the wrist put too low is refused there (contact_early), never pressed
from a guess.
"""

import math
import os
import time
from dataclasses import dataclass, field, replace

import numpy as np

from box_opening import LOGGER as log

# --- where the survey looks -----------------------------------------------------------------
# The box's centre on the arm's side, arm_mount (m): beside the chair parked side-on, its
# counter's near face at y 0.06-0.14 (the chair study's side scenes) and the box 0.10 m back from
# it or more; x from beside the shoulder (nearer the mount the reach-out's look stands over the
# arm's own base: the study's side sweeps pressed nothing at x 0.20, from x 0.25 on) to past the
# arm's reach. A cell is scored only where the arm could press it (survey_grid); outside, a box
# is seen only where a view happens to show it.
SIDE_X_M = (0.15, 0.90)
SIDE_Y_M = (0.10, 0.70)
# The counter heights the survey is built for (arm_mount z, m): 0.72-0.97 m above flatbot's floor,
# the study's three kitchen counters (z 0.239, 0.339, 0.389) inside
COUNTER_Z_M = (0.20, 0.45)
GRID_M = 0.05
HEIGHTS = 6  # counter heights sampled over COUNTER_Z_M
# A counter's side face stands no nearer the arm than the chair's widest box on the arm's side
# (seat/armrest_arm_*, y 0.038 in flatbot's layers) and this gap (the study's nearest, 0.02)
EDGE_GAP_M = 0.02
CHAIR_LAYERS = ("platform", "seat")
# A survey pose keeps every arm sphere this far clear of the counter band (Band): the highest
# counter it is built for, wherever on the arm's side it stands
KEEP_ABOVE_M = 0.05
BAND_NAME = "survey/counter_band"
BAND_REACH_M = 1.5  # the band runs this far along x and out in y: past anything the arm reaches
# ... but for the upper arm, shoulder to elbow: mounted 58 mm from the band's nearest face, no
# posture of flatbot's holds it KEEP_ABOVE_M off (its root 4-12 mm inside the band at every
# posture), so a survey pose keeps its spheres no nearer than the rest pose has them (Band). The
# forearm, the wrist and the hand, which do reach out over a counter, keep KEEP_ABOVE_M
REST_NEAR_LINKS = ("half_arm",)

# --- the wrist camera ---------------------------------------------------------------------------
# Its optical axes in the tool frame: the deployed mount (rammp-deployments cameras.yaml: quat
# [-0.5, -0.5, -0.5, 0.5] on arm_end_effector_link, the OAK's images turned 180 deg on the device),
# looking along the tool, the image's long side along -x of the tool, its top toward +y
OPTICAL_IN_TOOL = np.diag([-1.0, -1.0, 1.0])
# the lens from tool_frame, where the camera config gives none (camera_oak_wrist.yaml
# wrist_lens_in_hand_m: [0, 0.065, -0.105], the fitted mount)
LENS_IN_TOOL_M = (0.0, 0.065, -0.105)
# the wrist OAK's colour stream, where the camera has sent none yet: 640 x 400, fy 398 px
NOMINAL_K = ((398.0, 0.0, 320.0), (0.0, 398.0, 200.0), (0.0, 0.0, 1.0))
NOMINAL_SIZE = (400, 640)  # rows, columns
# The OAK's stereo leaves the colour image's left columns without depth (60 px on 640), and the
# closed fingers can reach into its bottom rows: a lid counts as in view only clear of both
VIEW_LEFT_PX = 60
VIEW_BOTTOM_FRAC = 0.10
VIEW_MARGIN_PX = 8  # its whole rim this far inside the rest (wrist_finder.EDGE_PX 3, with slack)
# a lid nearer than this has no stereo depth (the OAK's ~0.2 m; the finder drops the hand at 0.15)
MIN_RANGE_M = 0.25
SEE_MAX_RANGE_M = 0.90  # ... and farther than this too coarse for the lid band (the finder's 1.0 m)
# The finder checks the button within 50 deg of straight down and 0.70 m (wrist_finder.CHECK_*);
# the survey counts a check only inside these, for the real lid's height against the model's
CHECK_INCIDENCE_DEG = 45.0
CHECK_RANGE_M = 0.65
RIM_POINTS = 12

# --- the views ----------------------------------------------------------------------------------
# Candidates: the lens at each place of the CAM_* grid, tilted TILTS_DEG off straight down toward
# the scored cells' middle and AZIMUTHS_DEG either side of it, the image's long side across the
# tilt or along it. Offline (fix/survey's feasmap: flatbot's URDF and chair model, from HOME), of
# 1061 views with the lens at x 0-0.6, y -0.1-0.35, z 0.50-0.74, 0-50 deg, 48 cleared the chair,
# its user and the counter band, 34 of them with the lens at y 0.20, z 0.66-0.74, tilted 20-50
# deg (none lower over the arm's side: the band; none straight down: the reach). The grid is there
CAM_X_M = (0.00, 0.15, 0.30, 0.45)
CAM_Y_M = (0.20,)
CAM_Z_M = (0.66, 0.70, 0.74)
TILTS_DEG = (20.0, 35.0, 50.0)
AZIMUTHS_DEG = (-45.0, 0.0, 45.0)
SEE_WEIGHT = 0.6
MAX_VIEWS = 2
SECOND_GAIN = 0.40  # a second view only where it adds this share of the first's score ...
SECOND_MAX_S = 4.0  # ... and its joints fly from the first's within this (branches.joint_time, full speed)
VIEW_POOL = 3  # the first this many candidates that clear are weighed, best score first ...
GAIN_SLACK = 0.9  # ... and of those within this share of the best, the quickest to fly to is taken
IK_TRIES = 30  # candidates solved by IK at most, best first, per view chosen
# a candidate's wrist this far inside the arm's reach at least (ArmChain.past_reach_m's bound: the
# limits and the links' offsets end it sooner)
REACH_MARGIN_M = 0.05
POSTURE_SWING_RAD = 2.4  # the elbow walked this far each way along its self-motion for a clear posture
POSTURE_SWING_TRIES = 2  # ... from the quickest IK postures this many
# a posture's way there, as a straight joint line, is judged this finely when a pose is chosen (clear_posture's
# `way`; the planner's own path is judged at body.CHECK_STEP_RAD when it flies: fly_view)
WAY_STEP_RAD = 0.05
# The survey's IK: the arm's own (kinematics.ArmChain.solve_pose) from the rest pose, its look and
# IK_SEEDS postures drawn once (IK_SEED: the same draw every run), IK_ROUNDS rounds of damped steps
# each; solutions closer than IK_SAME_RAD are one posture
IK_SEEDS = 10
IK_SEED = 11
IK_ROUNDS = 3
IK_DAMPING = 0.01
IK_SAME_RAD = 0.05
PERSON_LAYERS = ("person",)

# --- the look again -----------------------------------------------------------------------------
# The lens this far off straight down over the place, nearest straight down first (the wrist's
# heights err more the more it tilts: wrist_finder's note), and this far from its lid top; past
# 25 deg only where nearer straight down no posture clears or reaches (a box 0.6 m out and more:
# the sim's side counters, 11-06), inside the finder's 50 deg for the button with margin
LOOK_AGAIN_TILT_DEG = (0.0, 15.0, 25.0, 35.0, 40.0)
LOOK_AGAIN_RANGE_M = (0.35, 0.42, 0.30)
LOOK_AGAIN_MAX = 2  # places looked at again

# --- the frames ---------------------------------------------------------------------------------
SURVEY_FUSE_FRAMES = 3  # still frames fused per pose (button_aim.StillDepth: a glossy lid's holes)
# ... within this long of the arm standing still (3 s was too short once: flatbot 2026-11-11 19:27, the
# run started 4 s after the wrist camera's driver, "no new wrist frame", and the scene camera's box was taken)
SURVEY_FRAME_S = 10.0
SURVEY_CAPTURE_TAG = "survey"
# A view is asked of the planner at most this many times where the client refuses what it planned:
# its joint plan at times ends on another turn of the wrist (its FK-pose fallback; private rig,
# 2026-11-12: 2 of 31 wrist-turn plans, 0.14 and 0.31 rad off on joint 7), and asked again it plans
SURVEY_PLAN_TRIES = 3


@dataclass(frozen=True)
class Camera:
    """The wrist camera as the survey plans with it: the lens in the tool frame, its optical axes
    there, its intrinsics and image size (rows, columns)."""

    lens: np.ndarray
    optical: np.ndarray
    k: np.ndarray
    size: tuple

    def view(self, R_tool, t_tool):
        """(the optical axes in the base frame, the lens's position) with the tool at (R, t)."""
        R = np.asarray(R_tool, float)
        return R @ self.optical, np.asarray(t_tool, float) + R @ self.lens

    def tool_for(self, cam_xyz, look_dir, wide_dir):
        """The tool pose (R, t) that puts the lens at `cam_xyz` looking along `look_dir`, the
        image's long side (optical x) along `wide_dir` as near as it can lie across the view."""
        d = np.asarray(look_dir, float)
        d = d / np.linalg.norm(d)
        w = np.asarray(wide_dir, float)
        w = w - (w @ d) * d
        n = float(np.linalg.norm(w))
        if n < 1e-6:
            return None
        w = w / n
        R_opt = np.stack([w, np.cross(d, w), d], 1)
        R_tool = R_opt @ self.optical.T
        return R_tool, np.asarray(cam_xyz, float) - R_tool @ self.lens


def camera(lens=None, k=None, size=None):
    """The Camera: the lens from the camera config's wrist_lens_in_hand_m (`lens`), the
    intrinsics the camera sent (`k`, `size`), each nominal where not given."""
    return Camera(
        lens=np.asarray(LENS_IN_TOOL_M if lens is None else lens, float).reshape(3),
        optical=OPTICAL_IN_TOOL.copy(),
        k=np.asarray(NOMINAL_K if k is None else k, float).reshape(3, 3),
        size=tuple(int(v) for v in (NOMINAL_SIZE if size is None else size)),
    )


def camera_of(watcher, camera_cfg_name=None):
    """The Camera for this run: the camera config's lens, the grabber's newest intrinsics."""
    lens = None
    try:
        from box_opening.perception.d405 import camera_config

        lens = camera_config(camera_cfg_name).get("wrist_lens_in_hand_m") if camera_cfg_name else None
    except Exception:  # noqa: BLE001 - a config without it: the fitted mount's
        lens = None
    g = getattr(watcher, "grab", None)
    k = getattr(g, "k", None)
    color = getattr(g, "color", None)
    size = None if color is None or np.ndim(color) < 2 else np.shape(color)[:2]
    return camera(lens=lens, k=k if k is not None and size is not None else None, size=size if k is not None else None)


@dataclass(frozen=True)
class Grid:
    """The lids the survey is scored on: every cell of the side region at every counter height of
    the band, (N, 3) lid-top centres, the counter's z under each, and each one's weight (1: the
    arm could press there, 0: not)."""

    lids: np.ndarray
    counter_z: np.ndarray
    weight: np.ndarray
    lid_r: float


def survey_grid(chain, model, x=SIDE_X_M, y=SIDE_Y_M, z=COUNTER_Z_M, step=GRID_M, heights=HEIGHTS):
    """The Grid for `model` (its lid's top dims.z above the counter): a cell weighs 1 where a
    tool straight down over its lid stands inside the arm's reach (ArmChain.past_reach_m)."""
    xs = np.arange(x[0], x[1] + 1e-9, step)
    ys = np.arange(y[0], y[1] + 1e-9, step)
    zs = np.linspace(z[0], z[1], int(heights))
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    cz = Z.ravel()
    lids = np.stack([X.ravel(), Y.ravel(), cz + float(model.dims[2])], 1)
    down = (1.0, 0.0, 0.0, 0.0)  # the tool straight down
    w = np.array([1.0 if chain.past_reach_m(p, down) <= 0.0 else 0.0 for p in lids])
    r = 0.5 * float(max(model.lid_dims[0], model.lid_dims[1]))
    return Grid(lids=lids, counter_z=cz, weight=w, lid_r=r)


def cover(cam, R_opt, cam_xyz, grid):
    """(checks, sees), (N,) bools over grid.lids: the lid's whole rim in the view clear of its
    edges (VIEW_*), its centre CHECK_MARGIN_PX inside, between MIN_RANGE_M and SEE_MAX_RANGE_M
    (`sees`); and within CHECK_INCIDENCE_DEG of straight down and CHECK_RANGE_M too (`checks`)."""
    from box_opening.perception import wrist_finder as wf

    P = np.asarray(grid.lids, float)
    c = np.asarray(cam_xyz, float)
    R = np.asarray(R_opt, float)
    rows, cols = cam.size
    k = cam.k
    a = np.linspace(0.0, 2.0 * math.pi, RIM_POINTS, endpoint=False)
    rim = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], 1) * grid.lid_r  # (M, 3)
    pts = P[:, None, :] + rim[None]  # (N, M, 3)
    pc = (pts - c) @ R  # into the optical frame
    zc = pc[..., 2]
    front = (zc > 1e-3).all(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = k[0, 0] * pc[..., 0] / zc + k[0, 2]
        v = k[1, 1] * pc[..., 1] / zc + k[1, 2]
    inside = (
        (u >= VIEW_LEFT_PX + VIEW_MARGIN_PX)
        & (u <= cols - 1 - VIEW_MARGIN_PX)
        & (v >= VIEW_MARGIN_PX)
        & (v <= rows * (1.0 - VIEW_BOTTOM_FRAC) - VIEW_MARGIN_PX)
    ).all(axis=1)
    pc0 = (P - c) @ R
    with np.errstate(divide="ignore", invalid="ignore"):
        u0 = k[0, 0] * pc0[:, 0] / pc0[:, 2] + k[0, 2]
        v0 = k[1, 1] * pc0[:, 1] / pc0[:, 2] + k[1, 2]
    m = wf.CHECK_MARGIN_PX
    centred = (pc0[:, 2] > 0) & (u0 >= m) & (u0 <= cols - 1 - m) & (v0 >= m) & (v0 <= rows - 1 - m)
    rng = np.linalg.norm(c - P, axis=1)
    sees = front & inside & centred & (rng >= MIN_RANGE_M) & (rng <= SEE_MAX_RANGE_M)
    up = (c - P)[:, 2] / np.maximum(rng, 1e-9)
    inc = np.degrees(np.arccos(np.clip(up, -1.0, 1.0)))
    checks = sees & (inc <= CHECK_INCIDENCE_DEG) & (rng <= CHECK_RANGE_M)
    return checks, sees


BAND_LAYER = BAND_NAME.split("/", 1)[0]


class Band:
    """The counter band (the module's note) as the survey keeps the arm from it: every place a
    counter of COUNTER_Z_M could stand on the arm's side — from below the floor to the band's top,
    its face no nearer the arm than the chair's widest box on that side (CHAIR_LAYERS) and
    EDGE_GAP_M, out past anything the arm reaches. Robot-body geometry and the survey's own
    envelope, never a counter: nothing of the room is in it. At a survey pose (refusal) every
    checked arm sphere keeps KEEP_ABOVE_M clear of it, but for the upper arm's (REST_NEAR_LINKS),
    which keep no nearer than the arm's rest pose `rest` has them (DEEPER_M): its root stands 4-12
    mm inside the band at every posture of flatbot's (the arm mounted 58 mm from the nearest face
    the band allows, its shoulder 0.20 m under the highest top), as the chair's layers were carved
    round the base and the shoulder. The forearm, the wrist and the hand keep KEEP_ABOVE_M however
    near the rest pose has them (flatbot's HOME holds the hand 5-50 mm over the band's top, its
    forearm and wrist 18-28 mm inside it). On the way there every sphere is held to it
    (path_refusal)."""

    def __init__(self, check, rest, top_z=COUNTER_Z_M[1], keep_m=KEEP_ABOVE_M):
        from box_opening.runtime.body import DEEPER_M

        lo, hi = self._box(check, float(top_z))
        self.top_z, self.keep_m = float(top_z), float(keep_m)
        self.body = check.with_obstacles([(BAND_NAME, (lo + hi) / 2.0, (hi - lo) / 2.0)], pad_m=0.0)
        d, self.links = self.body.sphere_distances(np.atleast_2d(np.asarray(rest, float)), layers=(BAND_LAYER,))
        self.rest_m = d[0]
        near = np.array([_bare(lk).startswith(REST_NEAR_LINKS) for lk in self.links], bool)
        self.required = np.where(near, np.minimum(self.keep_m, self.rest_m - DEEPER_M), self.keep_m)
        # the user's boxes and their padding, as the planner pads them, for the way there (path_refusal)
        self.user_layers = tuple(lay for lay in PERSON_LAYERS if lay in check.layers)
        self.user_pad_m = float(check.model.padding_m)
        # the highest any sphere may rise on the way (path_refusal), once the survey's own poses set it
        self.ceiling_z = None

    where = "beside the chair"

    def _box(self, check, top_z):
        """(lo, hi) of the band's box: beside the chair, from its widest box on the arm's side out."""
        self.y_edge = float(chair_edge_y(check) + EDGE_GAP_M)
        return (
            np.array([-BAND_REACH_M, self.y_edge, -BAND_REACH_M]),
            np.array([BAND_REACH_M, BAND_REACH_M, top_z]),
        )

    def refusal(self, Q):
        """Why joint states Q ((n, 7), or one (7,)) bring a sphere nearer the band than kept, or None."""
        Q = np.atleast_2d(np.asarray(Q, float))
        d, links = self.body.sphere_distances(Q, layers=(BAND_LAYER,))
        bad = d < self.required[None, :]
        if not bad.any():
            return None
        i = int(np.argwhere(bad.any(axis=1))[0][0])
        s = int(np.argmax(np.where(bad[i], self.required - d[i], -np.inf)))
        return "%s %.0f mm %s where a counter up to %.2f m may stand %s (kept %.0f mm%s)%s" % (
            links[s],
            1000 * abs(float(d[i, s])),
            "inside" if d[i, s] < 0 else "from",
            self.top_z,
            self.where,
            1000 * max(float(self.required[s]), 0.0) if self.required[s] > 0 else 0.0,
            "" if self.required[s] >= self.keep_m else ", as near as the rest pose has it",
            "" if len(Q) == 1 else ", %.0f%% of the way" % (100.0 * i / max(1, len(Q) - 1)),
        )

    def path_refusal(self, traj_or_points, step_rad=None):
        """Why a planned trajectory (or joint points), joined by straight joint lines sampled every
        `step_rad` (body.CHECK_STEP_RAD) as the body check samples a path, takes a sphere of the
        arm (every one the body check holds: the base and the shoulder are not among them) into the
        band or nearer the user than the planner pads them (PERSON_LAYERS), or any sphere above
        `ceiling_z` where that is set (a look again's way: no higher than the survey's own poses
        reached), or None. The body check's rule for a line out of where the arm stands
        (runtime/body; _first_nearer): a sphere outside where the path starts never enters, one
        inside may move in it or out of it, never DEEPER_M deeper. At the rest pose the upper arm,
        the forearm and the wrist already stand inside the band's worst case (its highest top at
        its nearest face), which no counter the arm rests beside can be; a way out of the rest pose
        that sweeps the forearm deeper meets a lower counter there (the sim's straight line to the
        first pose chosen when only the wrist and the hand were judged: the forearm 10 mm into the
        0.91 m counter 20 mm from the chair), and the first chosen once the band held every sphere
        crossed the user's torso box (the forearm 20 mm inside it). The chair's own boxes are not
        judged on the way: flatbot's HOME stands inside seat/armrest_arm (the chair study's
        problem 1), and every way out of it reads deeper there at its start; the planner holds
        them."""
        from box_opening.runtime.body import CHECK_STEP_RAD

        pts = getattr(traj_or_points, "points", None)
        P = np.array([list(p.positions) for p in pts] if pts is not None else [list(q) for q in traj_or_points], float)
        if P.ndim != 2 or not len(P):
            return None
        step = CHECK_STEP_RAD if step_rad is None else float(step_rad)
        rows = [P[:1]]
        for a, b in zip(P[:-1], P[1:], strict=False):
            m = max(1, int(math.ceil(float(np.abs(b - a).max()) / step)))
            rows.append(a + (np.arange(1, m + 1) / m)[:, None] * (b - a))
        Q = np.concatenate(rows)
        if self.ceiling_z is not None:
            C, r, _links = self.body._centres(Q)
            tops = (C[..., 2] + r[None, :]).max(axis=1)
            if float(tops.max()) > self.ceiling_z + 1e-3:
                i = int(np.argmax(tops > self.ceiling_z + 1e-3))
                return "the arm %.0f mm higher than the survey's own poses reached, %.0f%% of the way" % (
                    1000 * (float(tops[i]) - self.ceiling_z),
                    100.0 * i / max(1, len(Q) - 1),
                )
        d, links = self.body.sphere_distances(Q, layers=(BAND_LAYER,))
        hit = _first_nearer(d, 0.0)
        if hit is not None:
            i, s = hit
            return "%s %.0f mm into where a counter up to %.2f m may stand %s%s, %.0f%% of the way" % (
                links[s],
                1000 * abs(float(d[i, s])),
                self.top_z,
                self.where,
                "" if d[0, s] >= 0.0 else " (%.0f mm where it started)" % (1000 * abs(float(d[0, s]))),
                100.0 * i / max(1, len(Q) - 1),
            )
        if not getattr(self, "user_layers", ()):
            return None
        d, links = self.body.sphere_distances(Q, layers=self.user_layers)
        hit = _first_nearer(d, self.user_pad_m)
        if hit is None:
            return None
        i, s = hit
        dm = 1000 * float(d[i, s])
        near = ("%.1f mm inside" % -dm) if dm < 0 else ("%.1f mm from" % dm)
        return "%s %s the user (the planner keeps %.0f mm)%s, %.0f%% of the way" % (
            links[s],
            near,
            1000 * self.user_pad_m,
            "" if d[0, s] >= self.user_pad_m else " (%.1f mm where it started)" % (1000 * float(d[0, s])),
            100.0 * i / max(1, len(Q) - 1),
        )


def _sampled(traj_or_points, step_rad=None):
    """A planned trajectory's (or joint points') states joined by straight joint lines sampled every
    `step_rad` (body.CHECK_STEP_RAD), as Band.path_refusal samples a way: (n, 7), or None with none."""
    from box_opening.runtime.body import CHECK_STEP_RAD

    pts = getattr(traj_or_points, "points", None)
    P = np.array([list(p.positions) for p in pts] if pts is not None else [list(q) for q in traj_or_points], float)
    if P.ndim != 2 or not len(P):
        return None
    step = CHECK_STEP_RAD if step_rad is None else float(step_rad)
    rows = [P[:1]]
    for a, b in zip(P[:-1], P[1:], strict=False):
        m = max(1, int(math.ceil(float(np.abs(b - a).max()) / step)))
        rows.append(a + (np.arange(1, m + 1) / m)[:, None] * (b - a))
    return np.concatenate(rows)


def _first_nearer(d, keep_m):
    """The first (state, sphere) of a way's signed distances `d` ((n, S), m) nearer than `keep_m`,
    where that sphere started no nearer, or DEEPER_M nearer than it started where it did (the body
    check's rule for a line out of where the arm stands, Body.first_hit), or None."""
    from box_opening.runtime.body import DEEPER_M

    allowed = np.where(d[0] < keep_m, d[0] - DEEPER_M, keep_m)
    bad = d < allowed[None, :]
    if not bad.any():
        return None
    i = int(np.argwhere(bad.any(axis=1))[0][0])
    return i, int(np.argmax(np.where(bad[i], allowed - d[i], -np.inf)))


def _bare(link):
    return link[4:] if link.startswith("arm_") else link


def chair_edge_y(check):
    """The furthest the chair's own boxes (CHAIR_LAYERS, unpadded) reach toward +y: where a
    counter beside it can stand no nearer than."""
    m = check.model
    hi = np.asarray(m.hi, float) - float(m.padding_m)
    ys = [float(h[1]) for n, h in zip(m.names, hi, strict=False) if n.split("/", 1)[0] in CHAIR_LAYERS]
    return max(ys) if ys else 0.0


@dataclass
class SurveyView:
    """A pose the wrist camera surveys from: the joints, where the lens stands and how it looks,
    and the grid lids it checks and sees from there."""

    name: str
    joints: list
    cam_xyz: tuple
    rot_cam: np.ndarray
    checks: np.ndarray
    sees: np.ndarray
    tilt_deg: float
    swing_rad: float = 0.0  # how far the elbow was swung from the IK's posture to clear
    person_m: float = float("nan")  # the arm's least distance to the user's boxes there
    score: float = 0.0
    what: str = ""


def _score(grid, checks, sees):
    w = grid.weight
    return float((w * (checks + SEE_WEIGHT * (sees & ~checks))).sum())


def _tilt(R_opt):
    return float(np.degrees(np.arccos(np.clip(-float(np.asarray(R_opt)[2, 2]), -1.0, 1.0))))


def candidates(chain, cam, grid):
    """Every candidate view as (what, R_tool, t_tool, R_opt, lens, checks, sees, score), best
    score first: the lens at each place of the CAM_* grid, straight down or tilted TILTS_DEG
    toward the scored cells' middle and AZIMUTHS_DEG either side, the image's long side across
    the tilt or along it — each one whose tool pose stands REACH_MARGIN_M inside everything the
    arm reaches (ArmChain.past_reach_m: no IK solves one past it)."""
    from box_opening.runtime.kinematics import mat_to_quat_xyzw

    w = grid.weight > 0
    mid = grid.lids[w].mean(axis=0) if w.any() else grid.lids.mean(axis=0)
    out = []
    for cx in CAM_X_M:
        for cy in CAM_Y_M:
            for cz in CAM_Z_M:
                c = np.array([cx, cy, cz])
                toward = math.atan2(float(mid[1] - cy), float(mid[0] - cx))
                for tilt in TILTS_DEG:
                    for az in AZIMUTHS_DEG if tilt > 0 else (0.0,):
                        phi = toward + math.radians(az)
                        h = np.array([math.cos(phi), math.sin(phi), 0.0])
                        t = math.radians(tilt)
                        d = math.sin(t) * h + np.array([0.0, 0.0, -math.cos(t)])
                        for roll, wide in (("across", np.cross([0.0, 0.0, 1.0], h)), ("along", h)):
                            pose = cam.tool_for(c, d, wide)
                            if pose is None:
                                continue
                            R_tool, t_tool = pose
                            if chain.past_reach_m(t_tool, mat_to_quat_xyzw(R_tool)) > -REACH_MARGIN_M:
                                continue
                            R_opt, lens = cam.view(R_tool, t_tool)
                            checks, sees = cover(cam, R_opt, lens, grid)
                            score = _score(grid, checks, sees)
                            if score <= 0.0:
                                continue
                            what = "lens [%.2f, %.2f, %.2f], %.0f deg off straight down%s, long side %s" % (
                                cx,
                                cy,
                                cz,
                                tilt,
                                "" if tilt == 0 else " toward %.0f deg" % math.degrees(phi),
                                roll,
                            )
                            out.append((what, R_tool, t_tool, R_opt, lens, checks, sees, score))
    out.sort(key=lambda c: -c[7])
    return out


def ik_seeds(rest):
    """The postures the survey's IK starts from: the rest pose, its look (primitives.look), and
    IK_SEEDS more drawn once over the arm's joint range (a fixed draw: the same every run)."""
    from box_opening.primitives.look import look_joints

    rng = np.random.default_rng(IK_SEED)
    lo = np.array([-math.pi, -2.0, -math.pi, -2.3, -math.pi, -1.9, -math.pi])
    out = [np.array(rest, float), np.array(look_joints(rest), float)]
    out += [lo + (-2.0 * lo) * rng.random(7) for _ in range(IK_SEEDS)]
    return out


def solutions(chain, seeds, t_tool, R_tool):
    """The arm's IK solutions for the tool at (t, R): solve_pose from each seed, IK_ROUNDS rounds
    of its damped steps at most; one per posture (any joint IK_SAME_RAD apart)."""
    out = []
    for seed in seeds:
        q = np.array(seed, float)
        for _ in range(IK_ROUNDS):
            got = chain.solve_pose(q, t_tool, R_tool, damping=IK_DAMPING)
            if got is not None:
                q = got
                break
            q = _damped_steps(chain, q, t_tool, R_tool)
        else:
            continue
        if all(float(np.abs(q - o).max()) > IK_SAME_RAD for o in out):
            out.append(q)
    return out


def _damped_steps(chain, q, t_tool, R_tool, n=12, damping=0.05):
    """`n` damped least-squares steps toward the tool at (t, R) from `q` (solve_pose's own, kept
    when they do not converge): a far seed's next round starts nearer."""
    from box_opening.runtime.kinematics import rotvec

    q = np.array(q, float)
    for _ in range(n):
        R, t, J = chain._fk_jac(q)
        err = np.r_[np.asarray(t_tool, float) - t, rotvec(np.asarray(R_tool, float) @ R.T)]
        q = q + J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), err)
    return q


def clear_posture(
    chain, check, band, q_from, t_tool, R_tool, seeds=None, swing_rad=POSTURE_SWING_RAD, ceiling_z=None, way=False
):
    """(joints, how far the elbow was swung, None) holding the tool at (t, R) clear of the chair
    and its user (`check`, strictly: Body.pose_refusal) and of the counter band (`band`: a Band),
    inside the planner's joint limits, (`ceiling_z`) no sphere's top above that, and (`way`) its
    way there from `q_from` as a straight joint line held to the band and off the user
    (band.path_refusal, as the flight's is: the planner's own is judged again when it flies): of
    the arm's IK solutions
    (reach() from `q_from`, then solutions() from `seeds`), the quickest to fly to from `q_from`
    that clears as it is, else the nearest along its self-motion that clears; or (None, 0, why)."""
    from box_opening.constants import PLANNER_LIMIT_MARGIN_RAD
    from box_opening.runtime.branches import CONTINUOUS_JOINTS, PLANNER_CONTINUOUS_LIMIT_RAD, joint_time, nearest_branch
    from box_opening.runtime.kinematics import mat_to_quat_xyzw

    quat = mat_to_quat_xyzw(R_tool)
    found = []
    q = chain.reach(list(q_from), t_tool, quat)
    if q is not None:
        found.append(np.array(q, float))
    for qq in solutions(chain, ik_seeds(q_from) if seeds is None else seeds, t_tool, R_tool):
        if all(float(np.abs(qq - o).max()) > IK_SAME_RAD for o in found):
            found.append(qq)
    if not found:
        return None, 0.0, "no IK"

    def fit(qq):
        qq = nearest_branch(qq, q_from)
        if not chain.within_limits(qq, margin=PLANNER_LIMIT_MARGIN_RAD):
            return None, "a joint within %.2f rad of its limit" % PLANNER_LIMIT_MARGIN_RAD
        if any(abs(qq[i]) > PLANNER_CONTINUOUS_LIMIT_RAD for i in CONTINUOUS_JOINTS):
            return None, "a continuous joint past the planner's bound"
        why = check.pose_refusal(qq) or band.refusal(qq)
        if why is None and ceiling_z is not None:
            top = sphere_top_z(check, qq)
            if top > ceiling_z + 1e-3:
                why = "the arm %.0f mm higher than the survey's own poses reached" % (1000 * (top - ceiling_z))
        if why is None and way:
            on_way = band.path_refusal([list(q_from), list(qq)], step_rad=WAY_STEP_RAD)
            why = None if on_way is None else "its way there: %s" % on_way
        return (None, why) if why else (qq, None)

    found.sort(key=lambda qq: joint_time(q_from, nearest_branch(qq, q_from)))
    why = None
    for qq in found:
        got, why_q = fit(list(qq))
        if got is not None:
            return got, 0.0, None
        why = why or why_q
    for qq in found[:POSTURE_SWING_TRIES]:
        for s, qs in sorted(chain.self_motion(qq, swing_rad), key=lambda sq: abs(sq[0])):
            got, _why = fit(list(qs))
            if got is not None:
                return got, float(s), None
    return (
        None,
        0.0,
        "%s (%d IK posture%s, and along the elbow's swing, %.1f rad each way)"
        % (
            why,
            len(found),
            "" if len(found) == 1 else "s",
            swing_rad,
        ),
    )


def _view(name, q, R_opt, lens, checks, sees, swing, check, what, score=0.0):
    d = check.distance(q, layers=PERSON_LAYERS) if set(PERSON_LAYERS) <= set(check.layers) else None
    return SurveyView(
        name=name,
        joints=[float(v) for v in q],
        cam_xyz=tuple(float(v) for v in lens),
        rot_cam=np.asarray(R_opt, float),
        checks=checks,
        sees=sees,
        tilt_deg=_tilt(R_opt),
        swing_rad=float(swing),
        person_m=float("nan") if d is None else float(d.m),
        score=float(score),
        what=what,
    )


def survey_views(chain, check, q_from, cam, model, rest=None, grid=None, max_views=MAX_VIEWS):
    """(views, why, stats): up to `max_views` SurveyViews, greedily, each from the first VIEW_POOL
    candidates (best score first, IK_TRIES at most) the arm holds clear (clear_posture: the chair and
    its user, the counter band as the rest pose `rest`, None HOME, has it, and the way there from
    `q_from` or the view before, as a straight joint line): the first view the best
    of them; a second, adding SECOND_GAIN of the first's score and flying from it within
    SECOND_MAX_S, the quickest to fly to of those within GAIN_SLACK of the best gain. ([], why,
    stats) where none clears. `stats`: the candidates, the IK tries, the share of the scored cells
    the views check and see, the seconds it took."""
    from box_opening.constants import HOME
    from box_opening.runtime.branches import joint_time

    t0 = time.monotonic()
    grid = survey_grid(chain, model) if grid is None else grid
    band = Band(check, list(HOME) if rest is None else rest)
    cands = candidates(chain, cam, grid)
    total = float(grid.weight.sum())
    stats = {"candidates": len(cands), "scored_cells": int(total), "ik": 0}
    views, refused = [], []
    have_c = np.zeros(len(grid.lids), bool)
    have_s = np.zeros(len(grid.lids), bool)
    q_prev = list(q_from)
    for k in range(int(max_views)):
        base = _score(grid, have_c, have_s)
        gains = sorted(
            ((_score(grid, have_c | c[5], have_s | c[6]) - base, i) for i, c in enumerate(cands)), key=lambda g: -g[0]
        )
        pool = []  # (gain, transit s, view): the first VIEW_POOL that clear, best gain first
        for gain, i in gains[:IK_TRIES]:
            if gain <= (SECOND_GAIN * views[0].score if k else 0.0) or len(pool) >= VIEW_POOL:
                break
            what, R_tool, t_tool, R_opt, lens, checks, sees, _s = cands[i]
            stats["ik"] += 1
            q, swing, why = clear_posture(chain, check, band, q_prev, t_tool, R_tool, way=True)
            if q is None:
                refused.append("%s: %s" % (what, why))
                continue
            transit = joint_time(q_prev, q)
            if k and transit > SECOND_MAX_S:
                refused.append("%s: %.1f s from the first view" % (what, transit))
                continue
            pool.append(
                (
                    gain,
                    transit,
                    _view("survey:look-%d" % (k + 1), q, R_opt, lens, checks, sees, swing, check, what, gain),
                )
            )
        if not pool:
            break
        # the first: the best score; a second: of those within GAIN_SLACK of the best, the quickest to fly to
        best = max(g for g, _t, _v in pool)
        if k == 0:
            got = max(pool, key=lambda c: c[0])[2]
        else:
            got = min((c for c in pool if c[0] >= GAIN_SLACK * best), key=lambda c: (c[1], -c[0]))[2]
        views.append(got)
        have_c |= got.checks
        have_s |= got.sees
        q_prev = list(got.joints)
    stats["checks"] = round(float((grid.weight * have_c).sum()) / max(total, 1.0), 3)
    stats["sees"] = round(float((grid.weight * have_s).sum()) / max(total, 1.0), 3)
    stats["choose_s"] = round(time.monotonic() - t0, 3)
    if views:
        return views, None, stats
    why = "no survey pose the arm holds clear of the chair, its user and where a counter may stand (%s)" % (
        "; ".join(refused[:3]) or "no candidate the arm reaches sees a lid"
    )
    return [], why, stats


def sphere_top_z(check, q):
    """The highest any of the arm's checked spheres reaches at joints `q` (m, base z)."""
    C, r, _links = check.spheres_at(q)
    return float((C[:, 2] + r).max())


def look_again_view(chain, check, q_from, cam, model, xy, top_z, rest=None, n=1, ceiling_z=None):
    """A SurveyView over the place `xy` whose lid top stands at `top_z`: the lens LOOK_AGAIN_RANGE_M
    from it, tilted LOOK_AGAIN_TILT_DEG off straight down at most, back toward the arm, the lid
    whole in view and its button checkable (cover), the arm clear as for a survey pose and no
    higher than `ceiling_z` (run_survey: the survey poses' own highest sphere — over the box the
    arm may stand under what the counter carries, an upper cabinet, which the frames looking down
    never show), nor its way there from `q_from` as a straight joint line (clear_posture's `way`,
    Band.path_refusal; the planner's own is judged again when it flies). (view, None) or (None,
    why)."""
    from box_opening.constants import HOME

    p = np.array([float(xy[0]), float(xy[1]), float(top_z)])
    r = 0.5 * float(max(model.lid_dims[0], model.lid_dims[1]))
    one = Grid(lids=p[None], counter_z=np.array([float(top_z) - float(model.dims[2])]), weight=np.ones(1), lid_r=r)
    band = Band(check, list(HOME) if rest is None else rest)
    band.ceiling_z = ceiling_z
    a = math.atan2(-p[1], -p[0])
    back = np.array([math.cos(a), math.sin(a), 0.0])
    refused = []
    for tilt in LOOK_AGAIN_TILT_DEG:
        for rng in LOOK_AGAIN_RANGE_M:
            t = math.radians(tilt)
            c = p + rng * (math.sin(t) * back + np.array([0.0, 0.0, math.cos(t)]))
            for roll, wide in (("across", np.cross([0.0, 0.0, 1.0], back)), ("along", back)):
                pose = cam.tool_for(c, p - c, wide)
                if pose is None:
                    continue
                R_tool, t_tool = pose
                R_opt, lens = cam.view(R_tool, t_tool)
                checks, sees = cover(cam, R_opt, lens, one)
                if not checks[0]:
                    continue
                # its way there too, as a straight joint line (fly_view judges the planner's own)
                q, swing, why = clear_posture(chain, check, band, q_from, t_tool, R_tool, ceiling_z=ceiling_z, way=True)
                if q is None:
                    refused.append("%.2f m, %.0f deg, long side %s: %s" % (rng, tilt, roll, why))
                    continue
                what = "over [%.3f, %.3f] from %.2f m, %.0f deg off straight down" % (p[0], p[1], rng, tilt)
                return _view("survey:look-again-%d" % n, q, R_opt, lens, checks, sees, swing, check, what), None
    return None, "no look again over [%.3f, %.3f] the arm holds clear (%s)" % (
        p[0],
        p[1],
        "; ".join(refused[:2]) or "no view within %.0f deg shows its button" % max(LOOK_AGAIN_TILT_DEG),
    )


def look_again_places(find, d_lid):
    """Where the finder asks to look again (at most LOOK_AGAIN_MAX, the strongest first): each
    passing place no view could check the button on, and each lid-sized top a view cut that no
    such place stands at. [(xy, lid top z)]."""
    out = []
    for q in getattr(find, "places", ()):
        if q.button == "unchecked":
            out.append(((float(q.xy[0]), float(q.xy[1])), float(q.top_z), max(p.n_slab for p in q.patches)))
    for p in getattr(find, "patches", ()):
        if p.why.endswith("cut off by the view"):
            if all(math.hypot(p.xy[0] - x, p.xy[1] - y) >= 0.5 * d_lid for (x, y), _z, _n in out):
                out.append(((float(p.xy[0]), float(p.xy[1])), float(p.top_z), int(p.n_slab)))
    out.sort(key=lambda o: -o[2])
    return [(xy, z) for xy, z, _n in out[:LOOK_AGAIN_MAX]]


# --- the run's survey -----------------------------------------------------------------------------


# --- the front: the wrist turned where the arm rests -----------------------------------------
# Facing a table (the owner's demo, 2026-11-11) the box stands where the arm presses it, in front
# of the arm: x 0.45-0.80, y -0.25..+0.25 (the demo reach map, study/demo/reach). The scene camera
# stands at y -0.61 looking straight ahead and shows a lid whole only for y <= -0.30 at x 0.80: no
# cell of the map is both seen whole and pressed, and where it does place a box its mount puts it
# 7-9 cm out (2026-11-11 18:35: the box found at [0.787, -0.240], "no reach-out press plans").
# The wrist camera rides the arm. At flatbot's HOME it already looks forward, 8 deg under level,
# from 0.64 m up; the box is just under its view. So the wrist TURNS, the arm above it standing
# as it rests (owner: "don't do a 'look down' pose. the arm would just need to angle itself so it
# can see the scene"): the three wrist joints (FRONT_JOINTS) alone, solved on this run so the
# lens's axis meets a point of the front region at a lid's height, the image level.
FRONT_X_M = (0.45, 0.85)  # the box's centre in front of the arm, arm_mount (m): the reach map's columns
FRONT_Y_M = (-0.40, 0.45)  # ... and its rows: the arm presses out to y +0.45 at x 0.60 (study/demo/reach)
# the table in front: a 29 in table stands at z 0.216 (the demo), the 11-11 folding table at 0.25
FRONT_TABLE_Z_M = (0.15, 0.35)
FRONT_JOINTS = (4, 5, 6)  # joints 5-7
FRONT_MAX_RAD = 1.0  # no wrist joint turns further than this from where the arm rests
# Where the lens's axis may be put, (x, y) in arm_mount at a lid's height: each solved on this run by
# the wrist alone (turn_to) and the views chosen from them by the cells each adds (angled_views).
# One view does not show the front: the lens stands 0.26 m over a lid on a 29 in table, 0.21 m
# behind the nearest row, so a row 0.50 m wide lies 50 deg either side of it, and the OAK's left
# 60 px (the +y side) hold no depth. Offline (study/demo/wristlook, the reach map's 86 places the
# arm opens the box at): the one view flown until 2026-11-12, aimed at [0.65, -0.05], showed 42 of
# them whole and the finder on its rendered frames placed the box at 62 (34 of its fixes from a
# lid the view cut, up to 16 mm out); from the views chosen here, a cut lid looked at again, at
# all 86 (3 mm out at worst).
FRONT_AIM_X_M = (0.45, 0.60)
FRONT_AIM_Y_M = (-0.30, -0.15, 0.0, 0.15, 0.30)
FRONT_MAX_VIEWS = 4  # views looked from at most, the one that shows the most first
FRONT_MIN_GAIN = 0.03  # a further view only where it shows this share of the scored cells no view before does
# A lid a view cuts is a place to look again at, the wrist turned to put it in the middle (as the
# side survey looks again from above a place): at most this many times a run
FRONT_AGAIN_MAX = 2
# The hand no nearer the user than this at a front view, the hint's and a look again's alike, nor
# on its way there (m): the least distance from a sphere of the links the wrist's turn moves
# (HAND_LINKS: the two wrist links, the bracelet, the gripper with its fingers and the wrist camera;
# the sphere's buffered surface) to the user's UNPADDED boxes (PERSON_LAYERS; Body.sphere_distances).
# A view that would bring the hand nearer is not flown, and says so. 0.0: no floor of the look's own
# — the body check's alone (the planner's 20 mm padding and the spheres' 5 mm), as the look was
# first flown. On flatbot_fwd8 the hand rests 204 mm from the user; the views chosen bring it to
# 155, 204, 75 (the one turned toward the user's side: the fingertips over the lap's corner) and
# 179 mm, a look again to 50 mm at the nearest (a lid at [0.45, -0.25]). What a floor costs
# (study/demo/wristlook, the 86 places the arm opens the box at, rendered frames): 0.075 or 0.100
# still finds the box at all 86, other views chosen in the near one's place and 8-9 fixes then from
# a lid a view cut that no look again may turn to (8 mm out at worst); 0.150 loses the 5 places
# nearest the user's side; 0.200 loses 39. Never asked to stand farther than the hand rests
# (FrontBand).
# BOX_OPENING_FRONT_HAND_USER_M in the run's environment sets it for a deployment (metres; the robot's own, as its
# body model is): unset, empty or unreadable, none.
FRONT_HAND_USER_ENV = "BOX_OPENING_FRONT_HAND_USER_M"


def _hand_user_floor(env=None):
    """The hand's floor from the user the environment names (FRONT_HAND_USER_ENV), 0.0 where it names none or
    what it names is no finite number >= 0."""
    raw = str((os.environ if env is None else env).get(FRONT_HAND_USER_ENV, "")).strip()
    try:
        v = float(raw) if raw else 0.0
    except ValueError:
        return 0.0
    return v if math.isfinite(v) and v >= 0.0 else 0.0


FRONT_HAND_USER_M = _hand_user_floor()
HAND_LINKS = ("spherical_wrist", "bracelet", "end_effector", "robotiq")  # every link past joint 5, by its stem
FRONT_ROLL_WEIGHT = 0.5  # the image's long side held level, against the aim's (unit: half an image)
FRONT_TURN_WEIGHT = 0.05  # ... and the least turn that does it
FRONT_ITERS = 60
FRONT_AIM_TOL = 0.25  # the aim within this share of half the image of its centre, or no view


class FrontBand(Band):
    """The band in front of the chair: every place a table of FRONT_TABLE_Z_M could stand before
    the user's knees (the furthest the chair's and the user's boxes reach toward +x, and
    EDGE_GAP_M), right across. The turn keeps the forearm, the wrist and the hand KEEP_ABOVE_M
    over its top, or as high as the rest pose has them where that is nearer."""

    where = "in front of the chair"

    def __init__(self, check, rest, top_z=FRONT_TABLE_Z_M[1], keep_m=KEEP_ABOVE_M, hand_user_m=None):
        super().__init__(check, rest, top_z=top_z, keep_m=keep_m)
        from box_opening.runtime.body import DEEPER_M

        # every sphere as near as the rest pose has it, where that is nearer than the keep: the
        # arm rests over the user's lap, and a table's edge may stand under its hand
        self.required = np.minimum(self.keep_m, self.rest_m - DEEPER_M)
        # the hand's floor from the user (FRONT_HAND_USER_M; 0: none), never farther than it rests
        floor = float(FRONT_HAND_USER_M if hand_user_m is None else hand_user_m)
        self.hand_user_m = 0.0
        if floor > 0.0 and self.user_layers:
            d, _links = self._hand_m(rest)
            self.hand_user_m = min(floor, float(d.min()) - DEEPER_M) if d.size else 0.0

    def _hand_m(self, Q):
        """((n, H) the distances of the hand's spheres (HAND_LINKS) to the user's unpadded boxes at
        each row of Q, their links)."""
        d, links = self.body.sphere_distances(np.atleast_2d(np.asarray(Q, float)), layers=self.user_layers)
        hand = np.array([_bare(lk).startswith(HAND_LINKS) for lk in links], bool)
        return d[:, hand], [lk for lk, h in zip(links, hand, strict=False) if h]

    def refusal(self, Q):
        """Band.refusal, and the hand no nearer the user than its floor (hand_user_m)."""
        why = super().refusal(Q)
        if why is not None or self.hand_user_m <= 0.0:
            return why
        d, links = self._hand_m(Q)
        if not d.size or float(d.min()) >= self.hand_user_m:
            return None
        i, s = np.unravel_index(int(np.argmin(d)), d.shape)
        return "the hand (%s) %.0f mm from the user, nearer than the look keeps it (%.0f mm)%s" % (
            _bare(links[s]),
            1000 * float(d[i, s]),
            1000 * self.hand_user_m,
            "" if len(d) == 1 else ", %.0f%% of the way" % (100.0 * i / max(1, len(d) - 1)),
        )

    def path_refusal(self, traj_or_points, step_rad=None):
        """Band.path_refusal, and the hand held to its floor from the user along the way (a sphere
        nearer than the floor where the way starts never goes nearer: _first_nearer)."""
        why = super().path_refusal(traj_or_points, step_rad)
        if why is not None or self.hand_user_m <= 0.0:
            return why
        Q = _sampled(traj_or_points, step_rad)
        if Q is None:
            return None
        d, links = self._hand_m(Q)
        hit = _first_nearer(d, self.hand_user_m) if d.size else None
        if hit is None:
            return None
        i, s = hit
        return "the hand (%s) %.0f mm from the user, nearer than the look keeps it (%.0f mm), %.0f%% of the way" % (
            _bare(links[s]),
            1000 * float(d[i, s]),
            1000 * self.hand_user_m,
            100.0 * i / max(1, len(Q) - 1),
        )

    def _box(self, check, top_z):
        m = check.model
        hi = np.asarray(m.hi, float) - float(m.padding_m)
        layers = CHAIR_LAYERS + PERSON_LAYERS
        xs = [float(h[0]) for n, h in zip(m.names, hi, strict=False) if n.split("/", 1)[0] in layers]
        self.x_edge = float((max(xs) if xs else 0.0) + EDGE_GAP_M)
        return (
            np.array([self.x_edge, -BAND_REACH_M, -BAND_REACH_M]),
            np.array([BAND_REACH_M, BAND_REACH_M, top_z]),
        )


def _lens(chain, cam, q):
    """(the optical axes in the base frame, the lens's position) at joints `q`."""
    R, t = chain.fk(q)
    return cam.view(R, t)


def turn_to(chain, cam, rest, aim_xyz, joints=FRONT_JOINTS, max_rad=FRONT_MAX_RAD):
    """The joints nearest `rest` that differ from it only in `joints` and put `aim_xyz` at the
    middle of the wrist camera's image, the image level (damped least squares from `rest` and
    from a turn of the first of them either way; the best), and how far the aim then lies from
    the image's middle, in halves of the image: (q, miss)."""
    rest = np.asarray(rest, float)
    aim = np.asarray(aim_xyz, float)
    idx = list(joints)
    rows, cols = cam.size
    k = cam.k

    def residual(dq):
        q = rest.copy()
        q[idx] += dq
        R_opt, lens = _lens(chain, cam, q)
        p = (aim - lens) @ R_opt
        z = max(float(p[2]), 1e-3)
        behind = 0.0 if p[2] > 1e-3 else 5.0
        return np.array(
            [
                (k[0, 0] * p[0] / z + k[0, 2] - 0.5 * cols) / (0.5 * cols) + behind,
                (k[1, 1] * p[1] / z + k[1, 2] - 0.5 * rows) / (0.5 * rows) + behind,
                FRONT_ROLL_WEIGHT * float(R_opt[2, 0]),
                *(FRONT_TURN_WEIGHT * dq),
            ]
        )

    best = None
    for start in (0.0, 0.5, -0.5):
        dq = np.zeros(len(idx))
        dq[0] = start
        for _ in range(FRONT_ITERS):
            r = residual(dq)
            J = np.stack([(residual(dq + e) - r) / 1e-5 for e in 1e-5 * np.eye(len(idx))], 1)
            step = np.linalg.solve(J.T @ J + 1e-3 * np.eye(len(idx)), -J.T @ r)
            dq = np.clip(dq + np.clip(step, -0.2, 0.2), -max_rad, max_rad)
            if float(np.abs(step).max()) < 1e-5:
                break
        r = residual(dq)
        cost = float(r @ r)
        if best is None or cost < best[0]:
            best = (cost, dq.copy(), float(np.hypot(r[0], r[1])))
    q = rest.copy()
    q[idx] += best[1]
    return [float(v) for v in q], best[2]


def _aimed(chain, check, band, cam, grid, rest, aim_xyz, name, what):
    """The rest pose with its wrist turned at `aim_xyz` (turn_to), judged as a pose: (SurveyView,
    None) where the aim lands (FRONT_AIM_TOL), the joints stand inside the planner's limits and the
    arm stands clear of the chair and its user (strictly, as every pose box_opening chooses) and
    of the front band; else (None, why). Its way there is the caller's to judge (_way)."""
    from box_opening.constants import PLANNER_LIMIT_MARGIN_RAD

    q, miss = turn_to(chain, cam, rest, aim_xyz)
    if miss > FRONT_AIM_TOL:
        return None, "no turn of the wrist within %.1f rad puts it in view" % FRONT_MAX_RAD
    if not chain.within_limits(q, margin=PLANNER_LIMIT_MARGIN_RAD):
        return None, "past a joint limit"
    why = check.pose_refusal(q) or band.refusal(q)
    if why is not None:
        return None, why
    R_opt, lens = _lens(chain, cam, q)
    checks, sees = cover(cam, R_opt, lens, grid)
    return _view(name, q, R_opt, lens, checks, sees, 0.0, check, what), None


def _way(check, band, q_prev, q):
    """Why the straight joint line from `q_prev` to `q` is not flown (the chair, its user, the
    front band), or None."""
    n = max(2, int(math.ceil(float(np.abs(np.subtract(q, q_prev)).max()) / WAY_STEP_RAD)) + 1)
    line = np.linspace(np.asarray(q_prev, float), np.asarray(q, float), n)
    why = check.refusal(line) or band.path_refusal(line)
    return None if why is None else "its way there: %s" % why


def angled_views(chain, check, q_from, cam, model, rest=None, max_views=FRONT_MAX_VIEWS, hint=None):
    """(views, why, stats) as survey_views gives them, for a box in FRONT of the arm: the rest
    pose `rest` (None: HOME) with its wrist turned (turn_to) so the camera's axis meets a point of
    the front region at a lid's height over the middle of FRONT_TABLE_Z_M. The points are the
    FRONT_AIM_* grid's; a view is a candidate where the aim lands, the joints stand inside the
    planner's limits and the arm stands clear of the chair, its user and the front band (_aimed).
    The views are chosen by what they add: the one that shows the most scored cells whole first,
    then the one that adds the most to it, while one adds FRONT_MIN_GAIN of the cells and its
    straight joint line from the view before holds to the chair, the user and the band; at most
    `max_views`.

    `hint`: (x, y) where something says the box may stand (the scene camera's fix, rough or cut
    as it may be): the wrist looks there first, where a turn reaches it, and the chosen views
    follow for what that one does not show."""
    from box_opening.constants import HOME

    t0 = time.monotonic()
    rest = list(HOME) if rest is None else list(rest)
    grid = survey_grid(chain, model, x=FRONT_X_M, y=FRONT_Y_M, z=FRONT_TABLE_Z_M)
    band = FrontBand(check, rest)
    total = float(grid.weight.sum())
    lid_z = 0.5 * (FRONT_TABLE_Z_M[0] + FRONT_TABLE_Z_M[1]) + float(model.dims[2])
    aims = [(float(ax), float(ay)) for ax in FRONT_AIM_X_M for ay in FRONT_AIM_Y_M]
    stats = {"candidates": len(aims) + (hint is not None), "scored_cells": int(total), "ik": 0}
    views, refused = [], []
    have_c = np.zeros(len(grid.lids), bool)
    have_s = np.zeros(len(grid.lids), bool)
    q_prev = list(q_from)

    def take(view, gain):
        nonlocal have_c, have_s, q_prev
        views.append(replace(view, score=float(gain)))
        have_c |= view.checks
        have_s |= view.sees
        q_prev = list(view.joints)

    if hint is not None:
        hx, hy = float(hint[0]), float(hint[1])
        what = "the wrist turned to look where the box may stand, at [%.2f, %.2f]" % (hx, hy)
        view, why = _aimed(chain, check, band, cam, grid, rest, (hx, hy, lid_z), "survey:angle-hint", what)
        stats["ik"] += 1
        why = why if view is None else _way(check, band, q_prev, view.joints)
        if why is None:
            take(view, _score(grid, view.checks, view.sees))
        else:
            refused.append("%s: %s" % (what, why))
    cands = []
    for ax, ay in aims:
        what = "the wrist turned to look at [%.2f, %.2f]" % (ax, ay)
        view, why = _aimed(chain, check, band, cam, grid, rest, (ax, ay, lid_z), "survey:angle", what)
        stats["ik"] += 1
        if view is None:
            refused.append("%s: %s" % (what, why))
        else:
            cands.append(view)
    n_hint = len(views)
    while cands and len(views) < int(max_views) + n_hint:
        gains = [_score(grid, have_c | v.checks, have_s | v.sees) - _score(grid, have_c, have_s) for v in cands]
        i = int(np.argmax(gains))
        view = cands.pop(i)
        if gains[i] < FRONT_MIN_GAIN * SEE_WEIGHT * total:
            if not views:
                refused.append("%s: it shows no lid the arm presses" % view.what)
            break
        why = _way(check, band, q_prev, view.joints)
        if why is not None:
            refused.append("%s: %s" % (view.what, why))
            continue
        take(replace(view, name="survey:angle-%d" % (len(views) - n_hint + 1)), gains[i])
    stats["checks"] = round(float((grid.weight * have_c).sum()) / max(total, 1.0), 3)
    stats["sees"] = round(float((grid.weight * have_s).sum()) / max(total, 1.0), 3)
    stats["refused"] = refused[:6]
    # the views the hand's floor from the user kept the wrist from (FRONT_HAND_USER_M), said by the run
    stats["hand_kept_mm"] = round(1000 * band.hand_user_m, 1)
    stats["hand_refused"] = sum(": the hand (" in r or "its way there: the hand (" in r for r in refused)
    stats["choose_s"] = round(time.monotonic() - t0, 3)
    if views:
        return views, None, stats
    return [], "no turn of the wrist shows the table in front (%s)" % "; ".join(refused[:3]), stats


def again_place(find, d_lid, looked=()):
    """Where the front look turns to look again: the strongest lid-sized top the views so far cut
    (a passing place every sighting of which is cut, or a top refused only as cut off by the
    view) that stands no nearer than half a lid to a place in `looked`: ((x, y), lid top z), or
    None."""
    passing, refused = [], []
    for q in getattr(find, "places", ()):
        if q.button != "none" and q.patches and all(getattr(p, "cut", False) for p in q.patches):
            passing.append(((float(q.xy[0]), float(q.xy[1])), float(q.top_z), max(p.n_slab for p in q.patches)))
    for p in getattr(find, "patches", ()):
        if getattr(p, "why", "").endswith("cut off by the view"):
            refused.append(((float(p.xy[0]), float(p.xy[1])), float(p.top_z), int(p.n_slab)))
    # a place that passes cut before a top refused as cut, each the strongest first
    out = sorted(passing, key=lambda o: -o[2]) + sorted(refused, key=lambda o: -o[2])
    for xy, z, _n in out:
        if all(math.hypot(xy[0] - x, xy[1] - y) >= 0.5 * d_lid for x, y in looked):
            return xy, z
    return None


def again_view(chain, check, q_from, cam, model, xy, top_z, rest=None, n=1):
    """(view, None) for a look again at the lid a view cut, at `xy` with its top at `top_z`: the
    rest pose with the wrist turned to put it in the middle of the image, judged as every front
    view is (_aimed) and its straight joint line from `q_from` too — or (None, why)."""
    from box_opening.constants import HOME

    rest = list(HOME) if rest is None else list(rest)
    grid = Grid(
        lids=np.array([[xy[0], xy[1], top_z]], float),
        counter_z=np.array([top_z - float(model.dims[2])]),
        weight=np.ones(1),
        lid_r=0.5 * float(max(model.lid_dims[0], model.lid_dims[1])),
    )
    band = FrontBand(check, rest)
    what = "the wrist turned to the lid a view cut, at [%.2f, %.2f]" % (xy[0], xy[1])
    view, why = _aimed(chain, check, band, cam, grid, rest, (xy[0], xy[1], top_z), "survey:angle-again-%d" % n, what)
    if view is not None:
        why = _way(check, band, q_from, view.joints)
        if why is None and not bool(view.sees[0]):
            why = "the turn does not show it whole"
    if why is not None:
        return None, "no look again at [%.2f, %.2f]: %s" % (xy[0], xy[1], why)
    return view, None


@dataclass
class Survey:
    """What the survey did and found: `fix` (a SceneFix, source "wrist") or None with `why`; `find`
    the finder's last answer; `clouds` every frame's points (ctx.survey_clouds); `stopped` where a
    motion stopped part way (the arm holds there)."""

    fix: object = None
    why: str = ""
    find: object = None
    views: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    clouds: list = field(default_factory=list)
    stopped: bool = False
    preview: bool = False
    timing: dict = field(default_factory=dict)


def grab_frame(node, watcher, runner, tag, n=SURVEY_FUSE_FRAMES, timeout_s=SURVEY_FRAME_S):
    """One wrist frame from where the arm stands: up to `n` still frames shot after it stood still
    (runner.wait_still), each paired with its depth and posed by TF at its stamp, fused per pixel
    (button_aim.StillDepth); the newest one's colour and pose. (wrist_finder.WristFrame, its
    snapshot, its pose, None), or (None, None, None, why) where none came in `timeout_s`."""
    from box_opening.perception.button_aim import StillDepth
    from box_opening.perception.d405 import frame_snapshot
    from box_opening.perception.depth_source import camera_is_still, camera_pose_at, depth_key, depth_paired
    from box_opening.perception.wrist_finder import WristFrame
    from box_opening.tasks.press_demo import _spin_detect

    g = getattr(watcher, "grab", None)
    if g is None:
        return None, None, None, "no wrist camera"
    if runner is not None:
        runner.wait_still(tag)
    after_ns = node.get_clock().now().nanoseconds if node is not None else None
    fused = StillDepth(n)
    last_stamp, last_depth, got, why = None, None, None, "no new wrist frame"
    t_end = time.monotonic() + float(timeout_s)
    while time.monotonic() < t_end:
        if node is not None:
            _spin_detect(node)
        f = frame_snapshot(g)
        if f.color is None or f.depth is None or f.k is None or f.color_stamp is None:
            time.sleep(0.02)
            continue
        stamp = (int(f.color_stamp.sec), int(f.color_stamp.nanosec))
        if stamp == last_stamp or (after_ns is not None and stamp[0] * 10**9 + stamp[1] <= after_ns):
            time.sleep(0.02)
            continue
        ok, why_pair = depth_paired(f, last_depth)
        if not ok:
            why = why_pair
            time.sleep(0.01)
            continue
        cam = camera_pose_at(g, stamp=f.color_stamp)
        if cam is None:  # TF lands after the frame: asked again next time round
            why = "no camera pose for the frame (TF)"
            time.sleep(0.01)
            continue
        last_stamp, last_depth = stamp, depth_key(f)
        if got is not None and not camera_is_still(got[1], cam):
            fused.reset()
        got = (f, cam, fused.add(cam, f.depth))
        if len(fused.depths) >= int(n):
            break
    if got is None:
        return None, None, None, why
    f, cam, depth = got
    frame = WristFrame(
        color=f.color,
        depth=np.asarray(depth, np.float32),
        k=np.asarray(f.k, float),
        rot_cam=np.asarray(cam[0], float),
        trans_cam=np.asarray(cam[1], float),
        dist=None if getattr(f, "dist", None) is None else np.asarray(f.dist, float),
        tag=tag,
    )
    return frame, f, cam, None


def save_frame(watcher, snap, cam, frame, caption):
    """A survey frame as a capture (wrist-<SURVEY_CAPTURE_TAG>-<stamp>, its fused depth as
    depth_fused: wrist_finder.frame_from_capture replays it), written off this thread."""
    from box_opening.tasks.press_demo import CAPTURES_DIR, _prune_captures, _save_wrist_frame, capture_stamp

    folder = CAPTURES_DIR / ("wrist-%s-%s" % (SURVEY_CAPTURE_TAG, capture_stamp()))
    extra = {"depth_fused": np.asarray(frame.depth, np.float32), "fused_frames": np.array([SURVEY_FUSE_FRAMES])}
    return _save_wrist_frame(
        getattr(watcher, "grab", None),
        folder,
        "%s: %s" % (frame.tag, caption),
        background=True,
        after=lambda: _prune_captures(SURVEY_CAPTURE_TAG),
        frame=(snap, cam, extra),
    )


def fly_view(ctx, runner, view, band, execute):
    """`view` planned as the reach-out's look is (the joint goal, the full world, TRANSIT_SPEED),
    its way there held to the counter band (Band.path_refusal), and flown: ("flown" | "refused" |
    "stopped", why)."""
    from box_opening.constants import TRANSIT_SPEED
    from box_opening.primitives.core import PlanRefused, PlanState, _full_world, _plan_motion

    q0 = ctx.client.joints()
    leg = None
    for _ask in range(SURVEY_PLAN_TRIES):
        try:
            leg, _st = _plan_motion(
                ctx,
                PlanState(joints=list(q0), chain=0),
                view.name,
                ("joints", list(view.joints)),
                _full_world(ctx),
                TRANSIT_SPEED,
            )
            break
        except PlanRefused as e:  # a plan that ends elsewhere than asked: asked again (SURVEY_PLAN_TRIES)
            why = str(e)
        except RuntimeError as e:
            return "refused", str(e)
    if leg is None:
        return "refused", why
    why = band.path_refusal(leg.traj)
    if why is not None:
        return "refused", "its way there: %s" % why
    res = runner.run([leg], execute=execute)
    if any(not r.ok for r in res):
        last = res[-1]
        # the runner's own gates refuse a leg before it moves (Runner._refusal: the sanity gate, an
        # unarmed client): not flown, the arm where it stood; anything else stopped it part way
        return ("refused" if last.outcome == "refused" else "stopped"), "%s (%s)" % (last.outcome, last.detail)
    return "flown", None


def run_survey(node, ctx, runner, watcher, execute, rest=None, camera_cfg_name=None, front=False, hint=None):
    """The survey (the module's note) from where the arm stands, its rest pose `rest`: a Survey.
    Each step is noted ('wrist_survey'): the views chosen, and per view where it looked from and
    what the finder said.

    `front`: the box stands in front of the arm (angled_views): the wrist turned where the arm
    rests, no pose over the arm's side and no look again from above a place. From there the lid
    is seen 50-65 deg off straight down, where no button is checked (wrist_finder.CHECK_*): the
    finder goes by the lid's geometry alone, and the reach-out's own look and aim, nearer and
    steeper, find the button or stop the run, as they do for the scene camera's box. The views
    are looked from in turn (the one at `hint`, (x, y) where the scene camera has the box, first)
    until one shows the lid whole; a lid a view cuts is looked at again, the wrist turned to put
    it in the middle (again_view, FRONT_AGAIN_MAX), and is a fix only where that look shows it
    whole or could not be flown."""
    from box_opening.constants import HOME
    from box_opening.perception import wrist_finder as wf
    from box_opening.runtime import approach

    t0 = time.monotonic()
    out = Survey()
    check = approach.body_check()
    if check is None:
        out.why = "the survey keeps its poses clear of the chair by the body model, and it is off (body_layers)"
        return out
    rest = list(HOME) if rest is None else list(rest)
    chain, cam, model = approach._chain(), camera_of(watcher, camera_cfg_name), ctx.model
    if front:
        views, why, stats = angled_views(chain, check, ctx.client.joints(), cam, model, rest=rest, hint=hint)
    else:
        views, why, stats = survey_views(chain, check, ctx.client.joints(), cam, model, rest=rest)
    runner.note("wrist_survey", stage="views", front=bool(front), views=[_view_row(v) for v in views], why=why, **stats)
    if not views:
        out.why = why
        return _done(out, None, t0)
    if front:
        log.info(
            "[press_demo] WRIST LOOK: the wrist turns where the arm rests (%s rad on joints %s) for its camera to "
            "see the table in front, %d view%s until one shows the box: %.0f%% of the cells the arm presses there, "
            "at every table height of %.2f-%.2f m (%.2f s to choose)"
            % (
                "; ".join("/".join("%+.2f" % (v.joints[j] - rest[j]) for j in FRONT_JOINTS) for v in views),
                "-".join(str(j + 1) for j in (FRONT_JOINTS[0], FRONT_JOINTS[-1])),
                len(views),
                "" if len(views) == 1 else "s",
                100 * stats["sees"],
                FRONT_TABLE_Z_M[0],
                FRONT_TABLE_Z_M[1],
                stats["choose_s"],
            )
        )
        if stats.get("hand_refused"):
            log.info(
                "[press_demo] WRIST LOOK: %d turn%s of the wrist not looked from: the hand would come nearer the user "
                "than the look keeps it (%.0f mm)"
                % (stats["hand_refused"], "" if stats["hand_refused"] == 1 else "s", stats["hand_kept_mm"])
            )
    else:
        log.info(
            "[press_demo] WRIST SURVEY: the scene camera finds no container — %d pose%s for the wrist camera over "
            "the arm's side, checking the button on %.0f%% of the cells the arm presses and seeing %.0f%%, at every "
            "counter height of %.2f-%.2f m (%.2f s to choose)"
            % (
                len(views),
                "" if len(views) == 1 else "s",
                100 * stats["checks"],
                100 * stats["sees"],
                COUNTER_Z_M[0],
                COUNTER_Z_M[1],
                stats["choose_s"],
            )
        )
    band = FrontBand(check, rest) if front else Band(check, rest)
    frames = []
    d_lid = float(max(model.lid_dims[0], model.lid_dims[1]))
    ceiling = max(sphere_top_z(check, v.joints) for v in views)  # a look again never rises above the survey

    def look(view):
        """Fly to `view` and judge what the wrist sees there: the finder's answer, or None."""
        state, why = fly_view(ctx, runner, view, band, execute)
        out.views.append(view)
        if state != "flown":
            runner.note("wrist_survey", stage=view.name, flown=False, why=why)
            log.info("[press_demo] WRIST SURVEY %s (%s) not flown: %s" % (view.name, view.what, why))
            out.why = (
                "the move to %s stopped part way: %s" % (view.name, why)
                if state == "stopped"
                else "%s (%s) not flown: %s" % (view.name, view.what, why)
            )
            out.stopped = state == "stopped"
            return None
        if not execute:
            out.preview = True
            return None
        frame, snap, pose, why = grab_frame(node, watcher, runner, view.name)
        if frame is None:
            runner.note("wrist_survey", stage=view.name, flown=True, frame=False, why=why)
            log.info("[press_demo] WRIST SURVEY %s: no frame (%s)" % (view.name, why))
            out.why = "no wrist frame at %s (%s)" % (view.name, why)
            return None
        frames.append(frame)
        out.clouds.append(_cloud(frame))
        t_find = time.monotonic()
        find = wf.find_container(frames, model, check_buttons=not front)
        folder = save_frame(watcher, snap, pose, frame, find.why)
        runner.note(
            "wrist_survey",
            stage=view.name,
            flown=True,
            frame=True,
            cam=[round(float(v), 4) for v in frame.trans_cam],
            tilt_deg=round(_tilt(frame.rot_cam), 1),
            found=find.fix is not None,
            why=find.why,
            places=[[round(q.xy[0], 4), round(q.xy[1], 4), q.button] for q in find.places],
            confidence=round(float(find.confidence), 2),
            find_s=round(time.monotonic() - t_find, 3),
            capture=None if folder is None else str(folder),
        )
        log.info("[press_demo] WRIST SURVEY %s (%s): %s" % (view.name, view.what, find.why))
        return find

    find = None
    queue = list(views)
    looked, unshown, pending, n_again = [], [], None, 0
    while queue:
        view = queue.pop(0)
        got = look(view)
        find = got if got is not None else find
        if out.stopped or out.preview:
            return _done(out, find, t0)
        if pending is not None:
            # the look again's own frame: the lid whole where the view before cut one, or not
            if got is not None and not _shown_whole(got, pending, len(frames) - 1, d_lid):
                unshown.append(pending)
            pending = None
        if front and find is not None and find.fix is not None and _near(find.fix.top_xyz, unshown, d_lid):
            xy = find.fix.top_xyz
            find = replace(
                find,
                fix=None,
                why="wrist camera: a lid-sized top at [%.3f, %.3f] that a view cut, and the look again at it showed "
                "no lid whole there" % (xy[0], xy[1]),
            )
        fix = find.fix if find is not None else None
        if fix is not None and not (front and getattr(fix, "partial", "")):
            return _done(out, find, t0)
        if front and find is not None and n_again < FRONT_AGAIN_MAX:
            place = again_place(find, d_lid, looked)
            if place is not None:
                n_again += 1
                looked.append(place[0])
                again, why = again_view(
                    chain, check, ctx.client.joints(), cam, model, place[0], place[1], rest=rest, n=n_again
                )
                if again is not None:
                    pending = place[0]
                    queue.insert(0, again)
                    continue
                runner.note(
                    "wrist_survey", stage="look-again", xy=[round(v, 4) for v in place[0]], flown=False, why=why
                )
                log.info("[press_demo] WRIST LOOK: %s" % why)
        if fix is not None:  # a lid every view so far cut, and no look again at it to fly
            return _done(out, find, t0)
    if find is not None and not front:
        band.ceiling_z = ceiling  # nor on its way there
        for i, (xy, top_z) in enumerate(look_again_places(find, d_lid)):
            again, why = look_again_view(
                chain, check, ctx.client.joints(), cam, model, xy, top_z, rest=rest, n=i + 1, ceiling_z=ceiling
            )
            if again is None:
                runner.note("wrist_survey", stage="look-again", xy=[round(v, 4) for v in xy], flown=False, why=why)
                log.info("[press_demo] WRIST SURVEY: %s" % why)
                continue
            got = look(again)
            find = got if got is not None else find
            if out.stopped or (find is not None and find.fix is not None):
                break
    return _done(out, find, t0)


def _near(xyz, places, d_lid):
    """Is (x, y) of `xyz` within half a lid of one of `places` [(x, y), ...]?"""
    return any(math.hypot(xyz[0] - x, xyz[1] - y) < 0.5 * d_lid for x, y in places)


def _shown_whole(find, xy, i_frame, d_lid):
    """Did frame `i_frame` of `find` show a lid whole (a passing patch the view does not cut)
    within half a lid of `xy`?"""
    return any(
        p.frame == i_frame and not p.why and not p.cut and math.hypot(p.xy[0] - xy[0], p.xy[1] - xy[1]) < 0.5 * d_lid
        for p in getattr(find, "patches", ())
    )


def _done(out, find, t0):
    """`out` with the finder's last answer: its fix, or why there is none (the finder's, else the
    last view's that was not looked from)."""
    out.find = find
    if find is not None:
        out.fix = find.fix
        if find.fix is None and not out.stopped:
            out.why = find.why
    elif not out.why:
        out.why = "a preview: no frames" if out.preview else "no survey view was looked from"
    out.timing["survey_s"] = round(time.monotonic() - t0, 2)
    return out


def _cloud(frame):
    """A survey frame's points in the base frame (wrist_finder.lift: no hand, nothing past 1 m), the
    lens and the frame's tag: what ctx.survey_clouds keeps for what else needs the counter."""
    from box_opening.perception import wrist_finder as wf

    pts, _uv, _edge = wf.lift(frame)
    return {"points": pts.astype(np.float32), "cam_xyz": tuple(float(v) for v in frame.trans_cam), "tag": frame.tag}


def _view_row(v):
    return {
        "name": v.name,
        "joints": [round(float(q), 4) for q in v.joints],
        "cam": [round(float(c), 3) for c in v.cam_xyz],
        "tilt_deg": round(float(v.tilt_deg), 1),
        "swing_rad": round(float(v.swing_rad), 2),
        "person_mm": None if not math.isfinite(v.person_m) else round(1000 * v.person_m, 1),
        "what": v.what,
    }
