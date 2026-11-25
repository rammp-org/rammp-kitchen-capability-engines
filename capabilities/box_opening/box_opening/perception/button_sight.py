"""The button sighted from over it: where its centre is from the wrist lens, by colour alone.

The press's sight (tasks/reach_press: the hand turned down with the wrist lens over the aimed button,
standing still) takes one colour frame and asks where the button's centre is, in the base's xy, from the
lens's vertical. No depth: the wrist OAK's stereo reads nothing nearer than ~24 cm, and the lens stands
15-24 cm over the button there.

The frame is turned, undistorted and rotated in one remap, into a virtual camera at the same centre looking
straight down the world's z. Every level circle is a circle there whatever the hand's lean: a circle of
radius R (m) centred (X, Y) from the lens's vertical, Z below the lens, shows with radius r = f R / Z at
c + f (X, Y) / Z. The offset is (centre - c) R / r: f cancels, and the scale is the circle's own size
against the model's button. Its range f R / r is a record only (the seam's two edges, or one of them, set
the size to a few per cent), fine for scale in xy, no measure of a height.

Which circle is the seam is decided from colour alone, so it is decided by rules the recorded frames
taught (E4): the lid's moulded 60 mm ring and its rim are concentric with the seam, and the seam is not
always seen (a steep view, the fingers over it). Classical circle fitting: Hough gives the centres only;
about each, every circle is read from the edge pixels' distances (one direction of brightness at a time),
and a circle counts only where one edge of its own, fitted alone until the fit no longer moves, goes round
ten sixteenths of it.
"""

import math
from dataclasses import dataclass

import numpy as np

# HoughCircles on the median-blurred grey of the virtual picture, as the recorded frames' experiment ran it
# (E4: at these settings a centre was found on every pink frame that shows the lid), in one pass over the
# radii looked for. Only its centres are used, the SCAN_KEEP strongest: the lid's circles share one, and
# Hough finds it from whichever of them shows (its radius, one per centre, is not used). The lid's centre
# was Hough's first or second on every recorded frame that shows the seam and every render; six is three
# times that
HOUGH_PARAM1 = 80
HOUGH_PARAM2 = 30
SCAN_KEEP = 6
# The radii looked for: from SCAN_SPAN[0] of the seam's radius expected to SCAN_SPAN[1] of the lid's (the
# seam's, where the lid is not round): past the seam window's 0.80-1.30 and the rim's 12 % either way
SCAN_SPAN = (0.7, 1.4)
# ... within this of where the button is expected, beyond the flow's 30 mm (reach_press.SIGHT_MAX_OFF_M):
# a button a little past it is seen, and refused for being there, not missed
SEARCH_M = 0.04
# Circles whose centres stand within FAMILY_PX, or FAMILY_FRAC of the smaller one's radius, are one family:
# the seam, the ring and the rim of one lid (E4's gate)
FAMILY_PX = 6.0
FAMILY_FRAC = 0.12
# The seam groove is dark: its inner edge darkens going outward and its outer edge brightens, two circles
# about one centre the groove's width apart, 2 mm on the model's 22 mm radius (0.09). On the recorded
# frames its one-sign edges stand 2.75-4.8 px apart at 30-36 px (0.09-0.13, the blur spreading them);
# 0.15 is a little over the widest. Two uses (grooves_merged): a darkening edge and the brightening one
# outside it within GROOVE_FRAC of its radius are one groove, their mean the circle (the lid's 1.5 mm
# moulded ring is one too); and two edges of one sign within it are one edge seen twice, the better
# carried kept
GROOVE_FRAC = 0.15
# seam+rim: the family holds a circle button_diameter_m / lid diameter times another, within this: the
# recorded frames' seam-and-rim pairs stand 9-11 % off the model's 0.42 (15:19, 10:10, 14:04), the 60 mm
# ring against the rim (0.57) 35 %
RATIO_TOL = 0.12
# seam: the innermost, within this of the radius expected from the range the arm expects (the circle as its
# own edge fits it). Lopsided: the tops have read low, so the real seam has shown up to 1.3 times larger
# than expected, and the lid's 60 mm moulded ring is 1.35 times the seam. What is left: where the seam
# cannot be seen and the top is modelled some 6 mm or more too high, the ring passes as the seam, its scale
# making the offset about a quarter short (5 mm on 20). Narrower, real seams seen with the top read low
# would be refused, the case the sight is for; a seam that can be seen is taken first (a seam-and-rim pair
# in any family, then a seam by its size)
SEAM_WINDOW = (0.80, 1.30)
# Only a circle wholly inside the frame is a circle: one the frame's edge cuts fits its arc at more than one
# size and centre (the 14:33 close-up's seam, cut by the top edge, took a 51 px circle 13 px off its centre
# as well as its own 62). EDGE_PX inside the edge: past the OAK's dark rectification border, 1-4 px on the
# recorded frames, twice the widest. Checked at WHOLE_POINTS round each circle: a point every 7.5 deg, at
# most 18 px apart on the largest circle looked for on the recorded frames (133 px)
EDGE_PX = 8
WHOLE_POINTS = 48
# An edge pixel belongs to a circle about a centre where its gradient runs within 45 deg of the radius
# (cos 45 deg): a circle's own edge runs across it, a straight edge through it does not. A centre d px off
# the circle's turns its edge's gradient up to asin(d / r) off the radius: 21 deg for the recorded frames'
# worst, a Hough centre 10 px off a 28 px seam; 45 is twice that
RADIAL_COS = math.sqrt(0.5)
# The circles about a centre: the edge pixels' distances from it, one direction of brightness at a time,
# counted PEAK_PX wide (an edge read from a centre a pixel or two off spreads that much; from one further off
# a peak is only a start, and the second read, about the fitted centre, is sharp); each local most with at
# least PEAK_FRAC of the circle's round in pixels is a candidate. Low on purpose, under the 10 / 16 x 0.5
# (0.31) a circle carried holds at least: carried is the test
PEAK_PX = 3
PEAK_FRAC = 0.25
# carried: a candidate fitted by least squares to the edge pixels of its own brightening or darkening, first
# those within FIT_PX (a peak's PEAK_PX), then those within EDGE_FIT_PX, again and again until the
# pixels it holds no longer change. Most settle in 2-12 rounds; on the recorded frames a few walk on 0.1 px a
# round, some still moving at 30 (circles of 47-90 px, none the seam). Those settle by 60 on circles other
# starts reach in fewer rounds; stopped at 30 they stand 0.5-0.9 px short of them, where they began and not
# where their edge is (14:04), so a fit not settled in FIT_ROUNDS is refused, not taken where it stopped.
# Where it ends is not bounded: a Hough centre on the recorded frames stands 5-10 px off the
# seam's (16:23's 10), and the fit walks there; what it ends on must then carry it
FIT_PX = 3.0
EDGE_FIT_PX = 1.5  # an edge Canny draws one pixel wide, laid on a circle to the pixel's half either side
FIT_ROUNDS = 30
# carried measures the pixels near where its fit stood, cut again where it walks REACH_PX from there: the
# same pixels as the whole picture's, whatever REACH_PX is. A cut measures every pixel once; most walks
# stay within 4 px of their start, and 2-6 cost the same on the recorded frames (14:04: 29.3-29.8 ms)
REACH_PX = 4.0
# ... and its edge must go round it: in SUPPORT_MIN of SUPPORT_SECTORS equal sectors, each holding
# SUPPORT_ARC times as many pixels as the sector's arc (Canny's edge has 1-1.4 per pixel of arc: half lets a
# sector count where the edge is broken, and the few stray pixels of a sector it misses do not). A circle
# drawn across two concentric circles' arcs d px apart in radius stands d/2 off their centre and leaves each
# arc by (d/2)(1 - cos t): within EDGE_FIT_PX of an edge over some 6 sectors of 16, never 9. Ten: on 9
# the circle is not held by its arc (the 10:59 regrip frame, the fingers over 7 sectors: fits begun a pixel
# apart on the seam's one edge end 6 px apart, and its other edge's fit, carried in 9, stands 12 mm off the
# seam), while the recorded seams in view are carried in 11-16
SUPPORT_SECTORS = 16
SUPPORT_MIN = 10
SUPPORT_ARC = 0.5
MIN_BELOW_M = 0.08  # closer than this the button fills the frame (button_aim.MIN_RANGE_M)
# One circle reached from two candidates (two centres, two peaks): the fits agree within 0.3 px (one edge
# fitted from four starts 2 px off, on the renders and 16:23); 1 px is about three times that
SAME_PX = 1.0


@dataclass
class ButtonSight:
    """The button's centre from the lens's vertical, in the base's axes (m): `off_xy`; how far below the lens
    its circle's size puts it (`below_m`, a record only); `how` it was told from the lid's other circles
    ('seam+rim' or 'seam'); its radius in the virtual picture (`r_px`); and, to draw it on the frame as
    given, its centre there (`uv`) and the button's size (`diameter_m`)."""

    off_xy: tuple
    below_m: float
    how: str
    r_px: float
    uv: tuple
    diameter_m: float


@dataclass
class Edges:
    """A picture's edge pixels (Canny's), at `x`, `y`, and its brightness gradient there (`gx`, `gy`, `g`
    its size): made once a frame (edges_of)."""

    x: np.ndarray
    y: np.ndarray
    gx: np.ndarray
    gy: np.ndarray
    g: np.ndarray


def virtual_down(R_base_cam):
    """The virtual camera's axes in the base, as columns: z straight down, x the camera's own x laid level
    (the picture turned as little as it can be), y completing them. None where the camera's x stands
    within ~6 deg of vertical: no level x to lay it along."""
    r = np.asarray(R_base_cam, float)
    z = np.array([0.0, 0.0, -1.0])
    x = r[:, 0] - (r[:, 0] @ z) * z
    n = float(np.linalg.norm(x))
    if n < 0.1:
        return None
    x /= n
    return np.column_stack([x, np.cross(z, x), z])


def edges_of(gray):
    """The Edges of grey picture `gray`: Canny's at Hough's own thresholds, the gradient Sobel's."""
    import cv2

    ys, xs = np.nonzero(cv2.Canny(gray, HOUGH_PARAM1 / 2.0, HOUGH_PARAM1))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)[ys, xs].astype(float)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)[ys, xs].astype(float)
    return Edges(xs.astype(float), ys.astype(float), gx, gy, np.hypot(gx, gy))


def _centres(gray, lo, hi):
    """(u, v) of the SCAN_KEEP strongest centres HoughCircles finds in `gray` for radii `lo` to `hi` px."""
    import cv2

    found = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=max(12.0, lo),
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=int(math.floor(lo)),
        maxRadius=int(math.ceil(hi)),
    )
    return [] if found is None else [(float(u), float(v)) for u, v, _r in found[0][:SCAN_KEEP, :3]]


def _radial(e, u, v):
    """(each edge pixel's distance from (u, v), its brightening outward: +1, -1, or 0 where its gradient
    does not run along the radius within RADIAL_COS)."""
    dx, dy = e.x - u, e.y - v
    rho = np.hypot(dx, dy)
    along = (e.gx * dx + e.gy * dy) / np.maximum(rho, 1e-9)
    return rho, np.where(np.abs(along) >= RADIAL_COS * e.g, np.sign(along), 0.0)


def candidates(e, centre, lo, hi):
    """(u, v, r, sign) of every circle about `centre` whose edge pixels of one sign (+1 brightening
    outward, -1 darkening) stand out at one distance from it, radii `lo` to `hi` px."""
    u, v = centre
    rho, sign = _radial(e, u, v)
    edges = np.arange(math.floor(lo) - PEAK_PX, math.ceil(hi) + PEAK_PX + 1, 1.0)
    out = []
    for s in (1.0, -1.0):
        rr = rho[sign == s]
        n = np.convolve(np.histogram(rr, edges)[0], np.ones(PEAK_PX), "valid").astype(float)
        mid = edges[: len(n)] + PEAK_PX / 2.0
        frac = n / (2.0 * math.pi * np.maximum(mid, 1.0))
        left, right = np.r_[-1.0, n[:-1]], np.r_[n[1:], -1.0]
        for i in np.flatnonzero((frac >= PEAK_FRAC) & (n > left) & (n >= right) & (lo <= mid) & (mid <= hi)):
            near = rr[np.abs(rr - mid[i]) <= PEAK_PX / 2.0]
            out.append((u, v, float(near.mean()), s))
    return out


def carried(e, circle, sign, rho=None):
    """`circle` (u, v, r) fitted to its own edge (`sign`: +1 brightening outward, -1 darkening; Edges `e`)
    until the pixels it holds no longer change, or None where that edge does not carry it: the fit still
    moving after FIT_ROUNDS, or its pixels within EDGE_FIT_PX in under SUPPORT_MIN of SUPPORT_SECTORS
    sectors. Else (u, v, r, the sectors they lie in: _sectors). The fit is the edge's own, not where it
    began. `rho`: the pixels' distances from `circle`'s centre, where the caller has them.

    Each round measures only the pixels near where the fit stood when they were last cut (_Near), and cuts
    them again where it has walked REACH_PX from there: the pixels it holds are the same as on the whole
    picture, so is the fit. Every fit runs to its own end, also one that passes close to a circle already
    found: stopping it there loses one of two edges of a sign a few pixels apart (a shadow's beside the
    seam's) to whichever is fitted first, and the seam then moves with the order of Hough's centres."""
    u0, v0, r0 = (float(c) for c in circle[:3])
    fit, band, held, near = (u0, v0, r0), FIT_PX, None, None
    if rho is not None:
        near = _Near(e, fit, rho)
    least = SUPPORT_MIN * SUPPORT_ARC * 2.0 * math.pi * r0 / SUPPORT_SECTORS
    for _ in range(FIT_ROUNDS):
        if near is None or near.left(fit):
            near = _Near(e, fit)
        on = near.held(fit, band, sign)
        if np.count_nonzero(on) < max(least, 3):
            return None
        if band == EDGE_FIT_PX and held is not None and _same_pixels(near, on, *held):
            break
        held = (near, on) if band == EDGE_FIT_PX else None
        fit = near.fit(on)
        band = EDGE_FIT_PX
    else:
        return None  # still moving: stopped where it began, not where its edge is (FIT_ROUNDS)
    on = near.held(fit, EDGE_FIT_PX, sign)
    n = _sectors(fit, near.x[on], near.y[on])
    return (*fit, n) if n >= SUPPORT_MIN else None


def _same_pixels(near, on, near0, on0):
    """Whether pixels `on` of _Near `near` are pixels `on0` of `near0`."""
    return np.array_equal(on, on0) if near is near0 else np.array_equal(near.ids[on], near0.ids[on0])


class _Near:
    """The pixels of Edges `e` that a circle within REACH_PX of `at` (u, v, r: its centre moved and its
    radius changed, together) can hold within FIT_PX: a pixel d from that circle's centre stands within d
    of it from `at`'s, so within REACH_PX + FIT_PX of `at` (and a pixel more, for rounding). In the
    picture's order, their indices `ids`."""

    def __init__(self, e, at, rho=None):
        self.at = at
        if rho is None:
            rho = np.hypot(e.x - at[0], e.y - at[1])
        keep = np.abs(rho - at[2]) <= REACH_PX + FIT_PX + 1.0
        self.ids = np.flatnonzero(keep)
        self.x, self.y, self.gx, self.gy = e.x[keep], e.y[keep], e.gx[keep], e.gy[keep]
        # for held: the pixels as x + i y, and the gradient's conjugate either sign, so that the real part
        # of its product with a pixel's offset from a centre is the gradient along that offset
        self.z = self.x + 1j * self.y
        self.g_out = self.gx - 1j * self.gy
        self.g_in = -self.g_out
        # for the fit: each pixel's x, y, x^2 + y^2 and 1, from `at`'s centre (near the circle's, so the
        # normal equations stand well)
        a, b = self.x - at[0], self.y - at[1]
        self.terms = np.stack([a, b, a * a + b * b, np.ones_like(a)], axis=1)
        # _radial's test, its sign's gradient along the radius at least RADIAL_COS of the gradient and not
        # 0: at least this (the least float over 0 where RADIAL_COS g is 0)
        self.cg = np.maximum(RADIAL_COS * e.g[keep], np.nextafter(0.0, 1.0))

    def left(self, fit):
        """Whether `fit` stands further than REACH_PX from where these pixels were cut."""
        return math.hypot(fit[0] - self.at[0], fit[1] - self.at[1]) + abs(fit[2] - self.at[2]) > REACH_PX

    def fit(self, on):
        """The circle (u, v, r) through pixels `on` by least squares (the algebraic fit: x^2 + y^2 = p x +
        q y + c), from its normal equations."""
        t = self.terms[on]
        (xx, xy, xz, x), (_, yy, yz, y), (_, _, _, z), (_, _, _, n) = (t.T @ t).tolist()
        p, q, c = _solve3(((xx, xy, x), (xy, yy, y), (x, y, n)), (xz, yz, z))
        u, v = p / 2.0, q / 2.0
        return self.at[0] + u, self.at[1] + v, math.sqrt(max(c + u * u + v * v, 0.0))

    def held(self, fit, band, sign):
        """Which of these pixels circle `fit` holds: within `band` of it, their gradient of `sign` along
        its radius (_radial's)."""
        d = self.z - complex(fit[0], fit[1])
        rho = np.abs(d)
        along = (self.g_out if sign > 0 else self.g_in) * d
        return (np.abs(rho - fit[2]) <= band) & (along.real / np.maximum(rho, 1e-9) >= self.cg)


def _solve3(m, rhs):
    """x with m x = rhs, for 3 x 3 `m` (rows) by Cramer's rule; (0, 0, inf) where m is singular (the points
    on a line: no circle through them)."""
    (a, b, c), (d, e, f), (g, h, i) = m
    co = (e * i - f * h, f * g - d * i, d * h - e * g)
    det = a * co[0] + b * co[1] + c * co[2]
    if not abs(det) > 1e-12 * abs(a * e * i):
        return 0.0, 0.0, math.inf
    r0, r1, r2 = rhs
    return (
        (r0 * co[0] + b * (f * r2 - r1 * i) + c * (r1 * h - e * r2)) / det,
        (a * (r1 * i - f * r2) + r0 * co[1] + c * (d * r2 - r1 * g)) / det,
        (a * (e * r2 - r1 * h) + b * (r1 * g - d * r2) + r0 * co[2]) / det,
    )


def _sectors(circle, x, y):
    """How far edge pixels (x, y) go round `circle`: in how many of SUPPORT_SECTORS equal sectors they are
    SUPPORT_ARC times as many as the sector's arc has pixels."""
    u, v, r = circle[:3]
    if not len(x):
        return 0
    k = (np.arctan2(y - v, x - u) % (2.0 * math.pi)) // (2.0 * math.pi / SUPPORT_SECTORS)
    counts = np.bincount(k.astype(int).clip(0, SUPPORT_SECTORS - 1), minlength=SUPPORT_SECTORS)
    return int((counts >= SUPPORT_ARC * 2.0 * math.pi * r / SUPPORT_SECTORS).sum())


def circles_about(e, centres, lo, hi):
    """(u, v, r, sign, sectors) of every circle about `centres` (Hough's) that its own edge carries
    (carried): the candidates about each, each carried; then the same about the centre of each circle so
    found, where Hough's was some pixels off and a weaker circle's edge spread too thin there to stand out."""
    found, read = [], []
    for ring in (list(centres), None):
        if ring is None:
            ring = [f[:2] for f in found]
        for c in ring:
            if any(math.hypot(c[0] - p[0], c[1] - p[1]) <= SAME_PX for p in read):
                continue
            read.append(c)
            rho = None
            for u, v, r, s in candidates(e, c, lo, hi):
                rho = np.hypot(e.x - u, e.y - v) if rho is None else rho
                fit = carried(e, (u, v, r), s, rho)
                if fit is not None and not any(_same(fit, s, f) for f in found):
                    found.append((*fit[:3], s, fit[3]))
    return found


def _same(fit, sign, other):
    """Whether circle `fit` of edge `sign` is `other` (u, v, r, sign, ...) to SAME_PX."""
    return sign == other[3] and all(abs(a - b) <= SAME_PX for a, b in zip(fit[:3], other[:3], strict=True))


def _in_frame(m1, m2, shape):
    """Which pixels of the virtual picture (remap maps `m1`, `m2`) show the frame as given, EDGE_PX inside
    its edge: a boolean image."""
    h, w = shape
    return (m1 >= EDGE_PX) & (m1 <= w - 1 - EDGE_PX) & (m2 >= EDGE_PX) & (m2 <= h - 1 - EDGE_PX)


def _whole(circle, inside):
    """Whether all of `circle` (u, v, r, ...) lies on pixels the frame shows (`inside`, _in_frame's)."""
    u, v, r = circle[:3]
    a = np.linspace(0.0, 2.0 * math.pi, WHOLE_POINTS, endpoint=False)
    us, vs = np.round(u + r * np.cos(a)).astype(int), np.round(v + r * np.sin(a)).astype(int)
    h, w = inside.shape
    ok = (us >= 0) & (us < w) & (vs >= 0) & (vs < h)
    return bool(ok.all() and inside[vs, us].all())


def families(circles):
    """Concentric groups of `circles`, each smallest first: a circle joins the first family whose smallest
    circle's centre stands within FAMILY_PX or FAMILY_FRAC of that circle's radius of its own."""
    fams = []
    for c in sorted(circles, key=lambda c: c[2]):
        for f in fams:
            if math.hypot(c[0] - f[0][0], c[1] - f[0][1]) < max(FAMILY_PX, FAMILY_FRAC * f[0][2]):
                f.append(c)
                break
        else:
            fams.append([c])
    return fams


def grooves_merged(fam):
    """`fam` ((u, v, r, sign, sectors)) as circles (u, v, r), smallest first. Of edges of one sign within
    GROOVE_FRAC in radius, the one carried round more sectors (one edge seen twice: a shadow beside it, a
    Hough centre's walk ending a pixel or two apart); then each darkening edge with the brightening one
    outside it within GROOVE_FRAC taken as one, their mean (a groove: the seam, the lid's ring); every
    other edge as it is."""
    kept = []
    for a in sorted(fam, key=lambda c: (-c[4], c[2])):
        if not any(b[3] == a[3] and abs(b[2] - a[2]) <= GROOVE_FRAC * min(a[2], b[2]) for b in kept):
            kept.append(a)
    kept.sort(key=lambda c: c[2])
    out, used = [], set()
    for i, a in enumerate(kept):
        if i in used:
            continue
        outer = (j for j in range(i + 1, len(kept)) if a[3] < 0 < kept[j][3] and kept[j][2] <= a[2] * (1 + GROOVE_FRAC))
        j = next(outer, None)
        if j is None:
            out.append(tuple(float(x) for x in a[:3]))
        else:
            used.add(j)
            out.append(tuple(float((x + y) / 2.0) for x, y in zip(a[:3], kept[j][:3], strict=True)))
    return sorted(out, key=lambda c: c[2])


def which_seam(fam, model, r_expected, by=None):
    """(the seam of family `fam` (circles (u, v, r), smallest first, grooves merged), how it was told) or
    (None, why): the smaller of a seam-and-rim pair where the lid is round (RATIO_TOL), else the innermost
    where its size is the one expected (`r_expected` px; SEAM_WINDOW). `by`: 'rim' or 'size' for that way
    alone."""
    if by != "size" and getattr(model, "round", False):
        want = float(model.button_diameter_m) / float(model.lid_dims[0])
        pairs = [(abs(a[2] / b[2] - want) / want, a) for i, a in enumerate(fam) for b in fam[i + 1 :] if b[2] > a[2]]
        err, seam = min(pairs, default=(None, None), key=lambda p: p[0])
        if err is not None and err <= RATIO_TOL:
            return seam, "seam+rim"
    if by == "rim":
        return None, "no seam-and-rim pair"
    inner = fam[0]
    ratio = inner[2] / float(r_expected)
    if SEAM_WINDOW[0] <= ratio <= SEAM_WINDOW[1]:
        return inner, "seam"
    return None, "circles of %s px about one centre, the innermost %.2f times the seam's %.0f px expected" % (
        "/".join("%.0f" % c[2] for c in fam),
        ratio,
        float(r_expected),
    )


@dataclass
class Down:
    """A frame turned to look straight down (look_down): the grey picture (`gray`, median-blurred) and
    which of its pixels show the frame (`inside`, _in_frame's); its focal length `f` and the lens's vertical
    in it (`c`); the circles looked for (`lo` to `hi` px) and the seam's radius expected (`r_seam`); the
    virtual camera's axes in the base (`R_v`) and the camera's rays into its (`R_vc`); and, to put a point
    back on the frame, the camera's intrinsics and distortion (`k`, `dist`)."""

    gray: np.ndarray
    inside: np.ndarray
    f: float
    c: np.ndarray
    lo: float
    hi: float
    r_seam: float
    R_v: np.ndarray
    R_vc: np.ndarray
    k: np.ndarray
    dist: np.ndarray


def look_down(color, k, dist, R_base_cam, model, expect_off, expect_below):
    """(Down, None) for colour frame `color` (BGR) from a camera with intrinsics `k`, distortion `dist` and
    axes `R_base_cam` (its columns in the base), the button expected `expect_off` (base xy, m) from the
    lens's vertical and `expect_below` (m) below it; or (None, why)."""
    import cv2

    if color is None or k is None:
        return None, "no frame"
    below = float(expect_below)
    if not below >= MIN_BELOW_M:
        return None, "the button expected %.0f mm below the lens: too near to sight" % (1000 * below)
    R_v = virtual_down(R_base_cam)
    if R_v is None:
        return None, "the camera's x axis stands near vertical: no level picture to turn it to"
    kk = np.asarray(k, float)
    f = float(kk[0, 0])
    r_seam = f * float(model.button_diameter_m) / 2.0 / below
    lid_r = float(model.lid_dims[0]) / 2.0 if getattr(model, "round", False) else float(model.button_diameter_m) / 2.0
    lo, hi = SCAN_SPAN[0] * r_seam, SCAN_SPAN[1] * f * lid_r / below
    # the virtual picture: a window about where the button is expected, wide enough for the largest circle
    # looked for centred SEARCH_M from there; the lens's vertical at c, wherever that falls
    e = R_v.T @ np.array([float(expect_off[0]), float(expect_off[1]), 0.0])
    half = int(math.ceil(hi + f * SEARCH_M / below))
    c = np.array([half - f * e[0] / below, half - f * e[1] / below])
    k_v = np.array([[f, 0.0, c[0]], [0.0, f, c[1]], [0.0, 0.0, 1.0]])
    R_vc = R_v.T @ np.asarray(R_base_cam, float)  # camera rays into the virtual camera's
    d = np.zeros(5) if dist is None else np.asarray(dist, float).ravel()
    m1, m2 = cv2.initUndistortRectifyMap(kk, d, R_vc, k_v, (2 * half, 2 * half), cv2.CV_32FC1)
    img = cv2.remap(np.ascontiguousarray(color), m1, m2, cv2.INTER_LINEAR)
    gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    inside = _in_frame(m1, m2, np.asarray(color).shape[:2])
    return Down(gray, inside, f, c, lo, hi, r_seam, R_v, R_vc, kk, d), None


def sight_button(color, k, dist, R_base_cam, model, expect_off, expect_below):
    """The button's centre in colour frame `color` (BGR) from a camera with intrinsics `k`, distortion
    `dist` and axes `R_base_cam` (its columns in the base): (ButtonSight, None) or (None, why).

    `expect_off` (base xy, m: where the button is expected from the lens's vertical) and `expect_below`
    (m: how far below the lens its top is expected) choose and gate the circle only, never the answer:
    in the picture turned straight down (look_down), the circles each carried by an edge of its own about
    Hough's centres (circles_about), wholly in the frame, in concentric families; the seam (which_seam) by
    a seam-and-rim pair in any family first, then by its size alone in any, each way the family nearest
    where the button is expected first."""
    import cv2

    view, why = look_down(color, k, dist, R_base_cam, model, expect_off, expect_below)
    if view is None:
        return None, why
    gray, lo, hi, c, f, r_seam = view.gray, view.lo, view.hi, view.c, view.f, view.r_seam
    centres = _centres(gray, lo, hi)
    if not centres:
        return None, "no circle of %.0f-%.0f px radius about where the button is expected" % (lo, hi)
    edges = edges_of(gray)
    found = circles_about(edges, centres, lo, hi)
    if not found:
        return None, "no circle about where the button is expected that its own edge carries"
    whole = [p for p in found if _whole(p, view.inside)]
    if not whole:
        return None, "circles of %s px about where the button is expected, none whole inside the frame" % "/".join(
            "%.0f" % p[2] for p in sorted(found, key=lambda p: p[2])
        )
    # the seam: told by a seam-and-rim pair in any family before by its size alone in any, each way the family
    # nearest where the button is expected first: a stray family nearer it does not hide the button's, and a
    # circle of about the size expected is not taken over a pair that shows the seam
    half = gray.shape[0] / 2.0
    fams = sorted(families(whole), key=lambda g: math.hypot(g[0][0] - half, g[0][1] - half))
    first, seam = None, None
    for by in ("rim", "size"):
        for fam in fams:
            seam, how = which_seam(grooves_merged(fam), model, r_seam, by=by)
            if seam is not None:
                break
            if by == "size":
                first = first or how
        if seam is not None:
            break
    if seam is None:
        return None, first or "no seam"
    u, v, r = seam
    R = float(model.button_diameter_m) / 2.0
    o = np.array([u - c[0], v - c[1], 0.0]) * (R / r)  # in the virtual camera's axes, z left 0: level
    off = view.R_v @ o
    ray = view.R_vc.T @ np.array([(u - c[0]) / f, (v - c[1]) / f, 1.0])
    uv, _jac = cv2.projectPoints(ray.reshape(1, 1, 3), np.zeros(3), np.zeros(3), view.k, view.dist)
    return (
        ButtonSight(
            off_xy=(float(off[0]), float(off[1])),
            below_m=f * R / r,
            how=how,
            r_px=float(r),
            uv=(float(uv[0, 0, 0]), float(uv[0, 0, 1])),
            diameter_m=float(model.button_diameter_m),
        ),
        None,
    )
