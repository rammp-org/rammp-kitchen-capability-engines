"""Where the container can stand for the arm to open it: the boundary the line-up guides the user into.

The owner, 2026-11-16: "I want it to have a boundary of where it CAN and CANT open the box, and tell the
user to either move closer, further away, more left, or right."

A Zone is that boundary as a grid over the table, in the arm's frame (x ahead of the arm, y to its left),
for a lid at one height: each cell WORKS, FAILS or is UNASKED. Nothing in it is typed in: a cell's word is
the run's own proof that a press plans for a box standing there (tasks/reach_probe.press_plans: the same
plan_reach and way in the run makes before it moves, on the robot's own URDF and chair model), and the
lid's height is the cameras' that run. What this module adds is only how the grid is read:

    inside(xy)    the box's cell works, and nothing that does not work is nearer than MARGIN_M: the
                  cameras place a box to a few centimetres, and a box on the boundary is not in it
    to_zone(xy)   from the box to the nearest place that is inside, for guide.instruction's words
    order(focus)  which cells to ask next: the box's own, a coarse lattice outward from it, then only
                  the cells along the boundary found (a proof takes about a second on the robot's
                  computer; the whole grid would take a quarter of an hour, the boundary a few minutes)

Unasked cells are not inside. A grid part way through says None of a place it cannot yet judge (reach):
the panel says it is checking, never that the arm cannot.

Not in a Zone: where the lid goes once it is off. A run picks that from what the wrist camera sees with
the lid lifted, and at a table whose near edge is far from the user it can find no room (simulation,
2026-11-16: with the edge 0.55 m out, boxes nearer than 13-18 cm to it stopped at the set-down). The
run's own stop stays the last word there.

This module moves nothing and imports no kinematics: the engine reads it, a child process fills it.
"""

import json
import math
from dataclasses import dataclass, field

import numpy as np

from box_opening.guide import Reach

WORKS, FAILS, UNASKED = 1, 0, -1

STEP_M = 0.05  # a cell's side
MARGIN_M = 0.03  # inside: nothing that does not work is nearer than this
COARSE = 2  # the lattice asked first: every COARSE-th cell each way
_CHARS = {WORKS: "#", FAILS: ".", UNASKED: "?"}
_STATES = {c: s for s, c in _CHARS.items()}


@dataclass
class Zone:
    """The grid: cell (i, j)'s centre is (x0 + i * step, y0 + j * step). `lid_z`: the lid's top the cells
    were asked at (m, the arm's frame). `key`: what they were asked of (the robot's URDF, its chair model,
    the container and the proof's own code: reach_probe.zone_key); a Zone is only read where it matches."""

    x0: float
    y0: float
    step: float
    cells: np.ndarray  # (nx, ny) int8 of WORKS / FAILS / UNASKED
    lid_z: float = None
    key: str = ""
    _deep: dict = field(default_factory=dict, repr=False, compare=False)  # margin -> the cells inside by it

    # -- making one ----------------------------------------------------------------
    @classmethod
    def blank(cls, x_span, y_span, step=STEP_M, lid_z=None, key=""):
        """Every cell unasked, over x_span and y_span ((low, high), m; both ends are cell centres)."""
        nx = int(round((float(x_span[1]) - float(x_span[0])) / step)) + 1
        ny = int(round((float(y_span[1]) - float(y_span[0])) / step)) + 1
        cells = np.full((max(nx, 1), max(ny, 1)), UNASKED, np.int8)
        return cls(float(x_span[0]), float(y_span[0]), float(step), cells, lid_z, key)

    def to_json(self):
        rows = ["".join(_CHARS[int(v)] for v in row) for row in self.cells]
        return json.dumps(
            {"x0": self.x0, "y0": self.y0, "step": self.step, "lid_z": self.lid_z, "key": self.key, "rows": rows}
        )

    @classmethod
    def from_json(cls, text):
        d = json.loads(text)
        cells = np.array([[_STATES[c] for c in row] for row in d["rows"]], np.int8)
        return cls(float(d["x0"]), float(d["y0"]), float(d["step"]), cells, d.get("lid_z"), d.get("key", ""))

    # -- cells ---------------------------------------------------------------------
    def centre(self, i, j):
        return self.x0 + i * self.step, self.y0 + j * self.step

    def index(self, xy):
        """The cell (i, j) that holds `xy`, or None outside the grid."""
        i = int(math.floor((float(xy[0]) - self.x0) / self.step + 0.5))
        j = int(math.floor((float(xy[1]) - self.y0) / self.step + 0.5))
        nx, ny = self.cells.shape
        return (i, j) if 0 <= i < nx and 0 <= j < ny else None

    def state(self, xy):
        """WORKS, FAILS or UNASKED for the cell that holds `xy`; outside the grid nothing works."""
        ij = self.index(xy)
        return FAILS if ij is None else int(self.cells[ij])

    def set(self, i, j, works):
        self.cells[i, j] = WORKS if works else FAILS
        self._deep.clear()

    def counts(self):
        return {
            name: int((self.cells == s).sum()) for name, s in (("works", WORKS), ("fails", FAILS), ("unasked", UNASKED))
        }

    # -- reading it ----------------------------------------------------------------
    def clear_m(self, xy):
        """How far `xy` is from the nearest ground that does not work: a cell that fails or is unasked, or
        the grid's own outside. 0 where its own cell does not work."""
        x, y = float(xy[0]), float(xy[1])
        if self.state(xy) != WORKS:
            return 0.0
        nx, ny = self.cells.shape
        half = self.step / 2.0
        # the grid's outside: its outer cells' far sides
        out = min(
            x - (self.x0 - half),
            (self.x0 + (nx - 1) * self.step + half) - x,
            y - (self.y0 - half),
            (self.y0 + (ny - 1) * self.step + half) - y,
        )
        bad = np.argwhere(self.cells != WORKS)
        if len(bad) == 0:
            return max(0.0, out)
        dx = np.maximum(np.abs(self.x0 + bad[:, 0] * self.step - x) - half, 0.0)
        dy = np.maximum(np.abs(self.y0 + bad[:, 1] * self.step - y) - half, 0.0)
        return max(0.0, min(out, float(np.hypot(dx, dy).min())))

    def inside(self, xy, margin_m=MARGIN_M):
        """The arm opens a box at `xy`, by this grid: its cell works and nothing that does not is within
        `margin_m`."""
        return self.state(xy) == WORKS and self.clear_m(xy) >= margin_m

    def _deep_cells(self, margin_m):
        """(n, 2) centres of the cells whose own centre is inside by `margin_m`."""
        got = self._deep.get(margin_m)
        if got is None:
            keep = [
                self.centre(i, j)
                for i, j in np.argwhere(self.cells == WORKS)
                if self.inside(self.centre(i, j), margin_m)
            ]
            got = self._deep[margin_m] = np.array(keep, float).reshape(-1, 2)
        return got

    def to_zone(self, xy, margin_m=MARGIN_M):
        """From a box at `xy` to the nearest cell centre that is inside: (dx, dy), m. None where `xy` is
        inside already, or where no cell is known to be."""
        if self.inside(xy, margin_m):
            return None
        deep = self._deep_cells(margin_m)
        if len(deep) == 0:
            return None
        d = deep - np.array([float(xy[0]), float(xy[1])])
        k = int(np.hypot(d[:, 0], d[:, 1]).argmin())
        return float(d[k, 0]), float(d[k, 1])

    def reach(self, xy, planned=None, margin_m=MARGIN_M):
        """guide.Reach for a box at `xy` (`planned`: the proof for the box where it stands, where one was
        made). None where the grid cannot say yet: the box's own cell is unasked, or it is outside
        everything known to work while cells that may still work are unasked."""
        if self.state(xy) == UNASKED:
            return None
        if self.inside(xy, margin_m):
            return Reach(True, None, planned)
        to = self.to_zone(xy, margin_m)
        if to is None and not self.done():
            return None
        return Reach(False, to, planned)

    def within(self, near_m=None, far_m=None):
        """A copy in which no cell works whose centre is nearer the arm's base than `near_m` or further
        than `far_m` (None: no bound): what a deployment trusts of what the proof finds. On 2026-11-12 the
        arm opened boxes 0.63 and 0.68 m out; the proof plans presses to about 0.90 m that have never
        flown."""
        cells = self.cells.copy()
        nx, ny = cells.shape
        r = np.hypot((self.x0 + np.arange(nx) * self.step)[:, None], (self.y0 + np.arange(ny) * self.step)[None, :])
        out = np.zeros(cells.shape, bool)
        if near_m is not None:
            out |= r < float(near_m) - 1e-9  # a centre on the bound is within it
        if far_m is not None:
            out |= r > float(far_m) + 1e-9
        cells[out & (cells == WORKS)] = FAILS
        return Zone(self.x0, self.y0, self.step, cells, self.lid_z, self.key)

    # -- filling it ----------------------------------------------------------------
    def _beside_working(self):
        """The unasked cells with a working cell among their eight neighbours: the boundary's own."""
        works = np.pad(self.cells == WORKS, 1)
        near = np.zeros(self.cells.shape, bool)
        nx, ny = self.cells.shape
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di or dj:
                    near |= works[1 + di : 1 + di + nx, 1 + dj : 1 + dj + ny]
        return near & (self.cells == UNASKED)

    def order(self, focus=None):
        """The unasked cells worth asking, in the order to ask them: the cell that holds `focus` (the box),
        then the coarse lattice, then the cells beside one that works; each nearest `focus` first. A cell
        off the lattice with no working neighbour is never asked: it stays unasked, and so not inside."""
        fx, fy = (self.x0, self.y0) if focus is None else (float(focus[0]), float(focus[1]))
        own = None if focus is None else self.index(focus)
        beside = self._beside_working()
        out = []
        for i, j in np.argwhere(self.cells == UNASKED):
            i, j = int(i), int(j)
            if own == (i, j):
                tier = 0
            elif i % COARSE == 0 and j % COARSE == 0:
                tier = 1
            elif beside[i, j]:
                tier = 2
            else:
                continue
            cx, cy = self.centre(i, j)
            out.append((tier, math.hypot(cx - fx, cy - fy), i, j))
        return [(i, j) for _tier, _d, i, j in sorted(out)]

    def done(self):
        """No cell is left worth asking, but perhaps the box's own (order()'s first)."""
        unasked = self.cells == UNASKED
        if not unasked.any():
            return True
        nx, ny = self.cells.shape
        lattice = np.zeros(self.cells.shape, bool)
        lattice[::COARSE, ::COARSE] = True
        return not (unasked & (lattice | self._beside_working())).any()

    def picture(self):
        """The grid as text, far rows first and the arm's left on the left: # works, . fails, ? unasked."""
        nx, ny = self.cells.shape
        return "\n".join(
            "x %.2f  %s"
            % (self.x0 + i * self.step, "".join(_CHARS[int(self.cells[i, j])] for j in range(ny - 1, -1, -1)))
            for i in range(nx - 1, -1, -1)
        )
