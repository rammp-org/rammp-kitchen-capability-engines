"""The fridge's handle detector: door_common's, with the handle from its own depth and a door plane that must fit it.

At grasp range the door fills the frame, and on the 10-21 frames of this fridge YOLO gave 'cell phone' on the handle and
'chair' on the door, so YOLO narrows nothing. Colour-blind. The door's hinge edge isn't in view, so this detector
reports no hinge at all: the hinge used for planning is measured from the grasp (hinge_measured), and door_common's two
looks only have to agree on the handle.

With SAM 3 (handle_source "sam3"):
1. The instance (pick_handle_mask): the largest whose outline, at the distance its own depth gives, measures like a
   handle (loose height, width and width / height ranges), not the best-scoring -- on 11-04 a screwdriver on the
   bench and sideways handles across the room were picked over it. None handle-like: the look is a miss, never the
   best-scoring one.
2. The handle (_handle_from_mask): from the mask's own depth -- the nearest surface in each row of it, the bar's front.
   The handle reads true; the black glossy door around it can read only its reflection (11-04: the door at 0.9-1.1 m
   behind a handle at 0.5 m), so nothing about the handle's position comes from the door plane any more.
3. The door plane (_door_plane): fit on the door beside the handle, on its hinge side (door_region; the handle is near
   the free edge, past which the box would take in the bench and wall), and kept only if the handle stands
   handle_centre_to_face_m x DOOR_STANDOFF_MIN_FRACTION to handle_centre_to_face_m + DOOR_STANDOFF_SLACK_M in front of
   it, the way a handle sits on its door, with a horizontal normal facing the camera. Else that plane's points are left
   out and the next plane is fit. A true door is the first plane and passes; a reflection fails and, with nothing else
   on the door, the look fails, saying why.
With depth alone (handle_source "depth") the door plane comes from the whole image, as door_common's detector does it.

door_square_to_arm (testing only): the door is taken to face the arm squarely -- its normal along the arm base's -x --
and, with SAM 3, no door plane is fit at all (the door's depth on 11-04 was its reflection). A door yawed off that gets
a wrong hinge and swing.
"""

import door_common.perception
import numpy as np
from door_common.common import say
from door_common.perception import (  # noqa: F401 -- WristCamera re-exported for node.py
    DEPTH_RANGE_M,
    HANDLE_MASK_MIN_PIXELS,
    MAX_DOOR_NORMAL_VERTICAL,
    PLANE_DIST_M,
    WristCamera,
)

# what a SAM 3 instance (its outline, at the distance its own depth gives) must measure to be taken for the handle --
# loose, so only what plainly isn't a handle is left out. 11-04's wrong picks: a screwdriver on the bench (3 x 6 cm),
# a door's lever handle and a drawer pull across the room (no depth there; within it, ~4 and ~2 cm tall). The printed
# handle (data/RAMMP.Fridge.Handle5.STL) is 24.8 cm tall, 6.4 cm wide and stands 7.9 cm off the door: its outline is
# ~19-25 cm tall (the straight bar or all of it), 6-10 cm wide from head-on to 45 deg and the side; it measured 14-24
# x 6-15 cm on 11-04. No distance limit: the planner's reach and IK decide what is too far
HANDLE_HEIGHT_M = (0.08, 0.50)  # vertical extent (5th-95th pct): ~1/3 to 2x the handle's
HANDLE_WIDTH_M = (0.02, 0.25)  # horizontal extent: ~1/3 of the narrowest view to 2.5x the widest
# width / height: the handle stands upright, 0.26 (head-on) to ~0.53 (45 deg; 0.37-0.89 measured on 11-04's rebuilt
# masks); a handle mounted sideways is wider than tall (a door's lever ~3, a drawer pull ~10, a 20 x 10 cm bar 2), a
# door edge or a pole much narrower than tall. ~2.6x below the narrowest, 1.4x above the widest measured
HANDLE_WIDTH_TO_HEIGHT = (0.10, 1.25)
BAR_FRONT_DEPTH_M = 0.02  # per mask row, the bar's front: within this of the row's nearest point (the rest: the door
#   or its reflection seen past the bar's edges)
BAR_FRONT_MIN_POINTS = 50  # fewer valid depth points on the bar: the handle from the door plane instead (door_common)
DOOR_STANDOFF_MIN_FRACTION = 0.5  # a plane is the door if the handle's front stands from handle_centre_to_face_m x
#   this (11-04: closer is the handle or its mount, 0.2-0.8 cm; the door was 7-9 cm behind it) ...
DOOR_STANDOFF_SLACK_M = 0.10  # ... to handle_centre_to_face_m + this in front of it (11-04: its reflection, 35-55 cm)
DOOR_FACING_MAX_DEG = 70.0  # ... and it faces the camera within this (the camera starts square-on to the door; 11-04: a
#   plane at the handle's standoff facing sideways, the door's edge)
MAX_DOOR_PLANES = 4  # planes tried in the door region before the look fails
DOOR_REGION_WIDTHS = 3.0  # the region: this many mask widths beside the handle on its hinge side ...
DOOR_REGION_ROWS = 0.25  # ... over the mask's rows, plus this fraction of its height above and below
DOOR_REGION_MIN_PIXELS = 2000  # smaller (the hinge side out of view): a box round the handle instead
SQUARE_DOOR_NORMAL = np.array([-1.0, 0.0, 0.0])  # door_square_to_arm: the door's outward normal, back along the arm's x


def _facing_deg(normal, up_cam):
    """How far (deg) the plane's normal, horizontally, is from pointing straight back at the camera (camera frame)."""
    n = normal - up_cam * float(normal @ up_cam)
    back = -(np.array([0.0, 0.0, 1.0]) - up_cam * float(up_cam[2]))  # back along the camera's horizontal heading
    c = float(n @ back) / max(np.linalg.norm(n) * np.linalg.norm(back), 1e-9)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _handle_like(height, width):
    """Why an instance measuring `height` x `width` m is not the handle, or ""."""
    for what, v, (lo, hi) in (("tall", height, HANDLE_HEIGHT_M), ("wide", width, HANDLE_WIDTH_M)):
        if not lo <= v <= hi:
            return f"{v * 100:.0f} cm {what} (handle {lo * 100:.0f}-{hi * 100:.0f})"
    lo, hi = HANDLE_WIDTH_TO_HEIGHT
    if not lo * height <= width <= hi * height:
        how = "sideways" if width > hi * height else "too narrow"
        return f"{how}: {width * 100:.0f} cm wide, {height * 100:.0f} cm tall (handle width / height {lo:g}-{hi:g})"
    return ""


class HandleDetector(door_common.perception.HandleDetector):
    check_door_vertical = False  # _door_plane checks the normal itself, so it can try the next plane instead

    def __init__(self, handle_centre_to_face_m=0.070, door_square_to_arm=False):
        self.min_standoff = handle_centre_to_face_m * DOOR_STANDOFF_MIN_FRACTION
        self.max_standoff = handle_centre_to_face_m + DOOR_STANDOFF_SLACK_M
        self.last_miss = ""  # why the last look found nothing, for the refusal
        self.door_square_to_arm = door_square_to_arm

    def pick_handle_mask(self, instances, frame=None):
        """Of SAM 3's [(score, mask, class)] on `frame` (its depth_mm, K, base_T_cam): the largest instance that
        measures like a handle (_instance_size, _handle_like). None if none does (says why, and last_miss): never the
        best-scoring instead -- on 11-04 that was a screwdriver on the bench, or a sideways handle on a door or
        cabinet across the room. An instance needs depth on it to be measured: one without (beyond the depth range,
        like those across the room) is not taken -- the handle within reach always had it (11-04: every look)."""
        if not instances:
            return None
        found, why = [], []
        for score, mask, label in instances:
            px = int(np.count_nonzero(mask))
            if px < HANDLE_MASK_MIN_PIXELS:
                why.append(f"{label} {score:.2f}: {px} px")
                continue
            size = None if frame is None else self._instance_size(mask, frame)
            if size is None:
                why.append(f"{label} {score:.2f}: no depth on it to measure (out of range?)")
                continue
            problem = _handle_like(*size[:2])
            if problem:
                why.append(f"{label} {score:.2f}: {problem}")
                continue
            height, width, rng = size
            found.append((px, score, mask, label, f"{height * 100:.0f} x {width * 100:.0f} cm at {rng:.2f} m"))
        if not found:
            self.last_miss = f"no SAM 3 instance of {len(instances)} is handle-like ({'; '.join(why)})"
            say(f"  {self.last_miss}")
            return None
        px, score, mask, label, what = max(found, key=lambda t: t[0])
        say(f"  SAM 3: '{label}' {score:.2f}, {px} px, {what} -- of {len(instances)} instance(s)")
        return mask

    def _instance_size(self, mask, frame):
        """An instance's size: its whole silhouette (every mask pixel, as SAM 3 outlines it) placed at the distance of
        its front surface (_front_points) -- (vertical extent, horizontal extent, that distance), in m, along the arm
        base's vertical (so a rolled camera reads the same). Needs only enough depth on the instance for its distance:
        a glossy handle often has depth on a few rows only. None without that depth or the camera pose."""
        T = frame.get("base_T_cam")
        if T is None or frame.get("depth_mm") is None:
            return None
        front = self._front_points(mask, frame["depth_mm"], frame["K"])
        if front is None:
            return None
        d0 = float(np.median(front[:, 2]))  # camera-frame depth of its front surface
        vs, us = np.nonzero(mask)
        K = frame["K"]
        rays = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones(len(us))], axis=1)
        base = (rays * d0) @ np.asarray(T)[:3, :3].T  # arm-base orientation (extents only: no translation needed)
        z = base[:, 2]
        xy = base[:, :2] - base[:, :2].mean(axis=0)
        across = xy @ np.linalg.svd(xy, full_matrices=False)[2][0]  # along the main horizontal direction
        return (
            float(np.percentile(z, 95) - np.percentile(z, 5)),
            float(np.percentile(across, 95) - np.percentile(across, 5)),
            float(np.median(np.linalg.norm(front, axis=1))),
        )

    def _bar_front(self, mask, depth, K, up_cam):
        """The handle (camera frame) from the mask's own depth: per row, the points within BAR_FRONT_DEPTH_M of the
        row's nearest; their median, at the middle of their vertical extent (5th-95th pct), as door_common's depth
        handle. Returns (handle, (lo, hi) up base z (camera-frame offsets), points) or None if too few."""
        pts = self._front_points(mask, depth, K)
        if pts is None:
            return None
        up = pts @ up_cam
        lo, hi = float(np.percentile(up, 5)), float(np.percentile(up, 95))
        handle = np.median(pts, axis=0)
        return handle + up_cam * ((lo + hi) / 2 - float(handle @ up_cam)), (lo, hi), len(pts)

    @staticmethod
    def _front_points(mask, depth, K):
        """The mask's front surface (camera frame): per row, its points within BAR_FRONT_DEPTH_M of the row's nearest
        (the rest: what is seen past its edges). None with fewer than BAR_FRONT_MIN_POINTS."""
        vs, us = np.nonzero(mask)
        d = depth[vs, us] / 1000.0
        ok = ~np.isnan(d) & (d >= DEPTH_RANGE_M[0]) & (d <= DEPTH_RANGE_M[1])
        vs, us, d = vs[ok], us[ok], d[ok]
        if len(d) < BAR_FRONT_MIN_POINTS:
            return None
        nearest = np.full(int(vs.max()) + 1, np.inf)
        np.minimum.at(nearest, vs, d)
        front = d <= nearest[vs] + BAR_FRONT_DEPTH_M
        if front.sum() < BAR_FRONT_MIN_POINTS:
            return None
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        return np.stack([(us[front] - cx) / fx * d[front], (vs[front] - cy) / fy * d[front], d[front]], axis=1)

    def _handle_from_mask(self, mask, depth, K, normal, offset, up_cam, z0):
        """The bar's front from its own depth (detect found it, before the door plane); door_common's (the mask on the
        door plane) when the bar had too little depth."""
        if self._bar is None:
            return super()._handle_from_mask(mask, depth, K, normal, offset, up_cam, z0)
        lo, hi = self._bar_extent
        say(
            f"  handle from its own depth: {self._bar_points} pts on the bar's front, z {lo + z0:.3f} to "
            f"{hi + z0:.3f} -> middle {(lo + hi) / 2 + z0:.3f}; "
            f"{float(self._bar @ normal + offset) * 100:.1f} cm in front of the door plane"
        )
        return self._bar

    def _hinge_side(self, base_T_cam):
        """+1 if the door's hinge is toward the image's +u (right) from the handle, -1 if left: the hinge lies along
        the door from the handle (hinge_measured's 'along'), with the door square-ish to the camera."""
        Rb = base_T_cam[:3, :3]
        fwd = Rb[:, 2].copy()
        fwd[2] = 0.0
        n_out = -fwd / np.linalg.norm(fwd)  # the door's outward normal, roughly back toward the camera
        along = np.cross(n_out, [0.0, 0.0, 1.0])  # from the handle toward the hinge, in the arm's frame
        a, b = self._bar, self._bar + Rb.T @ (along * 0.2)
        return 1.0 if (b[0] / b[2] - a[0] / a[2]) > 0 else -1.0

    def door_region(self, rgb, box, debug):
        """Depth handle source: the whole image. SAM 3: the door beside the handle on its hinge side -- DOOR_REGION_
        WIDTHS mask widths of it over the mask's rows (+ DOOR_REGION_ROWS) -- or, too little of that in view, a box
        round the handle; less any plane already tried and refused."""
        h, w = rgb.shape[:2]
        mask = self._mask
        if mask is None or not mask.any():
            region = np.ones((h, w), dtype=bool)
        else:
            vs, us = np.nonzero(mask)
            u0, u1, v0, v1 = int(us.min()), int(us.max()), int(vs.min()), int(vs.max())
            mw, dv = u1 - u0 + 1, int((v1 - v0 + 1) * DOOR_REGION_ROWS)
            r0, r1 = max(0, v0 - dv), min(h, v1 + dv + 1)
            region = np.zeros((h, w), dtype=bool)
            if self._bar is not None:
                if self._hinge_side(self._base_T_cam) > 0:
                    region[r0:r1, min(w, u1 + 4) : min(w, u1 + 4 + int(DOOR_REGION_WIDTHS * mw))] = True
                else:
                    region[r0:r1, max(0, u0 - 4 - int(DOOR_REGION_WIDTHS * mw)) : max(0, u0 - 4)] = True
            if region.sum() < DOOR_REGION_MIN_PIXELS:
                region[r0:r1, max(0, u0 - mw) : min(w, u1 + mw + 1)] = True
                region &= ~mask
        if self._exclude is not None:
            region &= ~self._exclude
        return region

    def _door_plane(self, rgb, K, depth, base_T_cam, box, debug):
        """door_common's plane fit, kept only if it can be this handle's door (see the module docstring); else its
        points are left out and the next plane is fit, up to MAX_DOOR_PLANES."""
        self._exclude = None
        refused = []
        for _ in range(MAX_DOOR_PLANES):
            plane = super()._door_plane(rgb, K, depth, base_T_cam, box, debug)
            if plane is None:
                break
            _, _, normal, offset, up_cam, _, _ = plane
            vertical = abs(float(normal @ up_cam))
            standoff = None if self._bar is None else float(self._bar @ normal + offset)
            facing = _facing_deg(normal, up_cam)
            if vertical > MAX_DOOR_NORMAL_VERTICAL:
                refused.append(f"{np.degrees(np.arcsin(min(vertical, 1.0))):.0f} deg from vertical")
            elif standoff is not None and not self.min_standoff <= standoff <= self.max_standoff:
                refused.append(f"handle {standoff * 100:+.0f} cm in front")
            elif facing > DOOR_FACING_MAX_DEG:
                refused.append(f"facing {facing:.0f} deg off the camera")
            else:
                if refused:
                    say(f"  door plane: fit {len(refused) + 1} (refused before it: {'; '.join(refused)})")
                return plane
            # leave this plane's points out of the next fit
            vv, uu = np.mgrid[0 : depth.shape[0], 0 : depth.shape[1]]
            dd = depth / 1000.0
            pts = np.stack([(uu - K[0, 2]) / K[0, 0] * dd, (vv - K[1, 2]) / K[1, 1] * dd, dd], axis=-1)
            on = (dd > 0) & (np.abs(pts @ normal + offset) < 1.5 * PLANE_DIST_M)
            self._exclude = on if self._exclude is None else self._exclude | on
        self.last_miss = (
            f"no door plane fits the handle ({'; '.join(refused) or 'too few depth points'}; the handle stands "
            f"{self.min_standoff * 100:.0f} to {self.max_standoff * 100:.0f} cm off its door). If the door is glossy, "
            "its depth may be its reflection"
        )
        say(f"  {self.last_miss}")
        return None

    def detect(self, rgb, K, depth, base_T_cam, box=None, handle_mask=None, debug=None):
        """door_common's, with the handle from its own depth when SAM 3 gave a mask (found before the door plane, which
        must then fit it), and no hinge: the fridge's hinge edge isn't in view; the hinge used for planning is measured
        from the grasp (hinge_measured). Clearing it also skips door_common's cross-look hinge-agreement check."""
        self._mask, self._base_T_cam, self._exclude, self._bar = handle_mask, base_T_cam, None, None
        self.last_miss = ""
        if handle_mask is not None and base_T_cam is not None:
            bar = self._bar_front(handle_mask, depth, K, base_T_cam[2, :3])
            if bar is not None:
                self._bar, self._bar_extent, self._bar_points = bar
        if self.door_square_to_arm and self._bar is not None:
            return self._square_door(base_T_cam)
        det = super().detect(rgb, K, depth, base_T_cam, box=box, handle_mask=handle_mask, debug=debug)
        if det is not None:
            det["hinge"] = None
            if self.door_square_to_arm:
                det["normal"] = SQUARE_DOOR_NORMAL.copy()
                say("  door normal assumed square to the arm (door_square_to_arm)")
        return det

    def _square_door(self, base_T_cam):
        """door_square_to_arm with the handle from its own depth: the detection without any door plane."""
        Rb, tb = base_T_cam[:3, :3], base_T_cam[:3, 3]
        lo, hi = self._bar_extent
        z0 = float(tb[2])
        say(
            f"  handle from its own depth: {self._bar_points} pts on the bar's front, z {lo + z0:.3f} to {hi + z0:.3f}"
            f" -> middle {(lo + hi) / 2 + z0:.3f}; door normal assumed square to the arm (door_square_to_arm), no "
            "door plane fit"
        )
        return {
            "handle": Rb @ self._bar + tb,
            "quat": door_common.perception.HANDLE_QUAT,
            "normal": SQUARE_DOOR_NORMAL.copy(),
            "hinge": None,
            "door_z": None,
            "door_mid": None,
        }
