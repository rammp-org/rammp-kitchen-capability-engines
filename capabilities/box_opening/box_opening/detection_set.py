"""A recorded detection set, the parts with no robot in them: where the box
is placed (the grid), the frames' format, and the manifest that pins the
set — every file's sha256 and every placement's truth. Two writers: the
recorder (tasks/record_detection_set.py), a grid placed by hand, and every
--execute mission (press_demo), each run a placement of the MISSIONS_SET.
The campaign's evaluator (onyx/tools/evaluation) checks a set against its
manifest. Importable with no ROS at all."""

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

MISSIONS_SET = "missions"  # the set every --execute mission adds its placement to

# The grid, over the reachable table (the scene camera sees all of it: 187/187
# points of a finer grid, 2026-09-23). Snake order keeps each hover short.
GRID_X = (0.30, 0.40, 0.50, 0.60, 0.70)
GRID_Y = (-0.30, -0.15, 0.0, 0.15, 0.30)


def grid_points(xs=GRID_X, ys=GRID_Y):
    """[(x, y)] in snake order: along y, reversing every other row of x."""
    out = []
    for i, x in enumerate(xs):
        row = list(ys) if i % 2 == 0 else list(reversed(ys))
        out += [(float(x), float(y)) for y in row]
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(set_dir):
    """manifest.json: every file's sha256 and every placement's truth — what
    the evaluator checks the set against, and what the campaign commits. A
    placement without its truth.json (a run or a recording that stopped
    while writing it) is not part of the set."""
    set_dir = Path(set_dir)
    files = {}
    placements = []
    for p in sorted(set_dir.glob("p*")):
        if not (p / "truth.json").is_file():
            continue
        truth = json.loads((p / "truth.json").read_text())
        placements.append(dict(truth, id=p.name))
        for f in sorted(p.rglob("*")):
            if f.is_file():
                files[str(f.relative_to(set_dir))] = sha256(f)
    doc = {"set": set_dir.name, "placements": placements, "files": files}
    (set_dir / "manifest.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    return doc


def scene_frame(scene, T_base_link):
    """The scene camera's capture as the scene locator uses it, raw: colour
    (BGR), the median depth of the kept frames, both intrinsics, the depth
    -> colour extrinsic, link -> colour, and the calibration in force — for
    write_frame. None when a stream or a transform is missing.

    Nothing is copied or computed here: the grabber's frames are held (it
    replaces its arrays, it never writes into them) and the median is
    deferred to write_frame — the mission takes this with the arm about to
    move."""
    from box_opening.perception.scene import median_depth

    T_cd, T_lc = scene.depth_to_color(), scene.link_to_color()
    if scene.missing() or T_cd is None or T_lc is None or T_base_link is None:
        return None
    depths = list(scene.depths)
    return dict(
        color=scene.color,
        depth=lambda: median_depth(depths),
        k=np.asarray(scene.k, float),
        kd=np.asarray(scene.kd, float),
        dist=np.asarray(scene.dist if scene.dist is not None else [], float),
        T_color_depth=np.asarray(T_cd, float),
        T_link_color=np.asarray(T_lc, float),
        T_base_link=np.asarray(T_base_link, float),
    )


def wrist_frame(grab, pose):
    """One wrist frame — colour, depth, K, the camera's pose `pose` ((rot,
    trans) at the frame's stamp: depth_source.camera_pose_at) — for
    write_frame. The frames are held, not copied (as scene_frame)."""
    rot, trans = pose
    dist = getattr(grab, "dist", None)
    return dict(
        color=grab.color,
        depth=grab.depth,
        k=np.asarray(grab.k, float),
        rot_cam=np.asarray(rot, float),
        trans_cam=np.asarray(trans, float),
        stamp=np.array([grab.color_stamp.sec, grab.color_stamp.nanosec]),
        dist=np.asarray(dist if dist is not None else [], float),
    )


def write_frame(folder, arrays, compressed=True):
    """`arrays` (scene_frame / wrist_frame; a value may be a callable, taken
    now) as folder/frame_000.npz. Returns the folder."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {k: (v() if callable(v) else v) for k, v in arrays.items()}
    (np.savez_compressed if compressed else np.savez)(folder / "frame_000.npz", **arrays)
    return folder


def truth_trusted(doc):
    """A mission's truth counts only when the press went where it was aimed
    and the button answered: aimed by the close-up aim at the button
    (press_demo.aim_button_at_staging), and the knob seen popped afterwards
    or the lid gripped and placed. A fix from further up, a press that met
    something else, a run that ended before the button answered — the
    frames are kept, the truth is not scored."""
    return (
        doc.get("truth_source") == "aim"
        and doc.get("truth_xyz") is not None
        and (doc.get("popped") is True or doc.get("stage") == "done")
    )


def truth_note(doc):
    """Why a mission's truth is, or is not, scored (truth_trusted)."""
    if truth_trusted(doc):
        return "button truth recorded"
    if doc.get("truth_xyz") is None:
        return "no button truth: no press point (the box was not found, or the run stopped first)"
    if doc.get("truth_source") != "aim":
        return "no button truth: the press was not aimed close up"
    return "no button truth: the button was not seen to pop"


class MissionFrames:
    """One mission, as a placement of the MISSIONS_SET: what the scene
    camera saw before the arm moved, what the wrist saw at each search step
    it stood at, and — when truth_trusted — where the button really was,
    with the close-up aim's own frame (staging/, the circle drawn) to check
    it by eye.

    Frames are HELD during the run (scene_frame, wrist_frame: no copy, no
    disk) and written by close(), after the run has ended: the mission's
    torque guard runs in the same process, and nothing is written while it
    watches a stroke. Written uncompressed (~7 MB a scene frame, ~3 MB a
    wrist frame): ~0.1 s at the end of a run, where compressing cost ~0.5 s."""

    def __init__(self, set_dir, container, table_z, now=None):
        self.dir = Path(set_dir) / time.strftime("p%Y%m%d-%H%M%S", time.localtime(now))
        self.frames = []  # (folder name, arrays), in the order seen
        self.aim_capture = None  # the close-up aim's saved frame (press_demo.save_aim_frame), copied in
        self.doc = {
            "source": "mission",
            "container": container,
            "table_z": float(table_z) if table_z is not None else float("nan"),
            "stage": "started",
            "truth_xyz": None,
            "truth_source": None,
            "popped": None,
        }

    def scene(self, arrays, fix_xyz=None, score=None, why=None, source=None, boxes=None):
        """The scene camera's capture at the locate, and its answer (the
        box's top) or why it had none; which detector the fix came from
        ("segmenter" | "owl") and, per detector, the raw boxes of the last
        message the locate gated (SceneLocator.last_boxes) — so a live run
        replays exactly. `scene_owl` stays the score behind the fix, for the
        evaluator."""
        if arrays is not None:
            self.frames.append(("scene_%d" % sum(n.startswith("scene_") for n, _ in self.frames), arrays))
        self.doc.update(
            scene_fix_xyz=None if fix_xyz is None else [round(float(v), 5) for v in fix_xyz],
            scene_owl=None if score is None else round(float(score), 3),
            scene_why=why,
            scene_source=source,
            scene_boxes={str(k): [list(b) for b in v] for k, v in (boxes or {}).items()},
        )

    def wrist(self, pose, arrays):
        """A wrist frame the search stood at, at search step `pose` (look,
        sweep:left, sweep:right)."""
        prefix = "wrist_%s_" % pose
        self.frames.append((prefix + str(sum(n.startswith(prefix) for n, _ in self.frames)), arrays))

    def located(self, xyz, source, aim_capture=None):
        """The press point the run went on with, where it came from ("aim":
        the close-up aim at the button; "fix": anything else) and, for an
        aim, the folder its frame was saved in."""
        self.doc.update(truth_xyz=[round(float(v), 5) for v in xyz], truth_source=source, stage="located")
        self.aim_capture = aim_capture

    def note(self, **facts):
        """What the run established later: stage="pressed" / "done", popped=."""
        self.doc.update(facts)

    def close(self, exit_code=None):
        """Write the frames and truth.json. Returns the placement's folder,
        or None when nothing was seen (no frame to keep)."""
        aim = Path(self.aim_capture) if self.aim_capture is not None else None
        if aim is not None and not aim.is_dir():
            aim = None  # pruned from captures/ already
        if not self.frames and aim is None:
            return None
        for name, arrays in self.frames:
            write_frame(self.dir / name, arrays, compressed=False)
        names = [n for n, _ in self.frames]
        if aim is not None:
            shutil.copytree(aim, self.dir / "staging")
            names.append("staging")
        self.doc["exit_code"] = exit_code
        self.doc["truth_suspect"] = not truth_trusted(self.doc)
        self.doc["truth_note"] = truth_note(self.doc)
        self.doc["frames"] = names
        (self.dir / "truth.json").write_text(json.dumps(self.doc, indent=1))
        return self.dir
