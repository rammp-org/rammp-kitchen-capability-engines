"""The detection sets' pure parts: where the recorder asks for the box, the
manifest the evaluator checks a set against, and the record every mission
keeps of what its cameras saw (MissionFrames)."""

import json
from types import SimpleNamespace

import numpy as np


def test_the_grid_covers_the_reachable_table_in_snake_order():
    from box_opening.detection_set import GRID_X, GRID_Y, grid_points

    pts = grid_points()
    assert len(pts) == len(GRID_X) * len(GRID_Y) == 25
    assert len(set(pts)) == len(pts)
    # each hover is short: consecutive points are grid neighbours
    for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
        assert abs(x1 - x0) + abs(y1 - y0) <= max(max(GRID_X) - min(GRID_X), 0.16)
        assert (x0 == x1) or (y0 == y1)
    assert min(p[0] for p in pts) == 0.30 and max(p[0] for p in pts) == 0.70
    assert min(p[1] for p in pts) == -0.30 and max(p[1] for p in pts) == 0.30


def test_the_manifest_pins_every_file_and_carries_every_truth(tmp_path):
    from box_opening.detection_set import sha256, write_manifest

    for name, truth in (
        ("p000", {"container": "ankou_pink.yaml", "truth_xyz": [0.4, 0.0, 0.094]}),
        ("p900", {"container": None, "truth_xyz": None}),
    ):
        d = tmp_path / name
        (d / "scene_0").mkdir(parents=True)
        (d / "scene_0" / "frame_000.npz").write_bytes(b"frame " + name.encode())
        (d / "truth.json").write_text(json.dumps(truth))
    (tmp_path / "p001" / "scene_0").mkdir(parents=True)  # stopped while it was being written
    (tmp_path / "p001" / "scene_0" / "frame_000.npz").write_bytes(b"half")
    doc = write_manifest(tmp_path)
    assert [p["id"] for p in doc["placements"]] == ["p000", "p900"]
    assert doc["placements"][1]["container"] is None
    assert doc["files"]["p000/scene_0/frame_000.npz"] == sha256(tmp_path / "p000/scene_0/frame_000.npz")
    assert set(doc["files"]) == {
        "p000/scene_0/frame_000.npz",
        "p000/truth.json",
        "p900/scene_0/frame_000.npz",
        "p900/truth.json",
    }
    assert json.loads((tmp_path / "manifest.json").read_text()) == doc


class _SceneGrab:
    """A scene grabber double: two kept depth frames, streams and TF up."""

    def __init__(self, missing=()):
        self.colors = [np.full((4, 6, 3), 7, np.uint8)]
        self.depths = [np.full((4, 6), 0.5, np.float32), np.full((4, 6), 0.7, np.float32)]
        self.k = self.kd = np.eye(3)
        self.dist = None
        self._missing = list(missing)

    @property
    def color(self):
        return self.colors[-1]

    def missing(self):
        return self._missing

    def depth_to_color(self):
        return np.eye(4)

    def link_to_color(self):
        return np.eye(4)


def test_a_scene_frame_is_held_not_copied_and_its_median_is_taken_when_written(tmp_path):
    from box_opening.detection_set import scene_frame, write_frame

    g = _SceneGrab()
    arrays = scene_frame(g, np.eye(4))
    assert arrays["color"] is g.colors[-1]  # held: no copy on the mission's clock
    assert callable(arrays["depth"])  # the median waits for the write
    g.depths.append(np.full((4, 6), 9.0, np.float32))  # the grabber moves on
    z = np.load(write_frame(tmp_path / "scene_0", arrays) / "frame_000.npz")
    assert np.allclose(z["depth"], 0.6)  # the frames kept when it was taken
    assert set(z.files) == {"color", "depth", "k", "kd", "dist", "T_color_depth", "T_link_color", "T_base_link"}
    assert scene_frame(_SceneGrab(missing=["depth"]), np.eye(4)) is None
    assert scene_frame(g, None) is None  # no calibration, no frame


def _wrist_arrays():
    from box_opening.detection_set import wrist_frame

    grab = SimpleNamespace(
        color=np.zeros((4, 6, 3), np.uint8),
        depth=np.full((4, 6), 0.4, np.float32),
        k=np.eye(3),
        color_stamp=SimpleNamespace(sec=5, nanosec=6),
        dist=None,
    )
    return wrist_frame(grab, (np.eye(3), np.zeros(3)))


def test_a_mission_keeps_its_frames_and_scores_its_truth_only_when_the_press_landed(tmp_path):
    from box_opening.detection_set import MissionFrames, write_manifest

    rec = MissionFrames(tmp_path / "missions", "ankou_pink.yaml", 0.05, now=0)
    rec.scene(scene_frame_arrays(), fix_xyz=[0.41, 0.02, 0.15], score=0.52)
    rec.wrist("look", _wrist_arrays())
    rec.wrist("sweep:left", _wrist_arrays())
    rec.wrist("look", _wrist_arrays())
    aim = tmp_path / "captures" / "wrist-aim-1"  # the aim's own frame, as press_demo saved it
    aim.mkdir(parents=True)
    (aim / "frame_000.jpg").write_bytes(b"jpeg")
    rec.located([0.4, 0.0, 0.15], "aim", aim_capture=aim)
    rec.note(popped=True)
    rec.note(stage="pressed")
    out = rec.close(exit_code=0)
    doc = json.loads((out / "truth.json").read_text())
    assert doc["frames"] == ["scene_0", "wrist_look_0", "wrist_sweep:left_0", "wrist_look_1", "staging"]
    assert (out / "staging" / "frame_000.jpg").read_bytes() == b"jpeg"  # the truth, checkable by eye
    assert doc["truth_suspect"] is False and doc["truth_xyz"] == [0.4, 0.0, 0.15]
    assert doc["scene_fix_xyz"] == [0.41, 0.02, 0.15] and doc["exit_code"] == 0
    for name in doc["frames"][:-1]:
        assert (out / name / "frame_000.npz").is_file()
    # the evaluator's view of it: a box placement with its truth
    man = write_manifest(tmp_path / "missions")
    assert [p["id"] for p in man["placements"]] == [out.name]
    assert man["placements"][0]["container"] == "ankou_pink.yaml"


def scene_frame_arrays():
    from box_opening.detection_set import scene_frame

    return scene_frame(_SceneGrab(), np.eye(4))


def test_a_missions_truth_counts_only_from_a_close_up_aim_the_button_answered():
    from box_opening.detection_set import truth_note, truth_trusted

    aimed = {"truth_xyz": [0.4, 0.0, 0.15], "truth_source": "aim"}
    assert truth_trusted(dict(aimed, popped=True, stage="pressed"))
    assert truth_trusted(dict(aimed, popped=None, stage="done"))  # the lid came off: it popped
    assert not truth_trusted(dict(aimed, popped=False, stage="pressed"))
    assert not truth_trusted(dict(aimed, popped=None, stage="pressed"))  # nothing saw it pop
    assert not truth_trusted(dict(aimed, truth_source="fix", popped=True, stage="done"))  # aimed from further up
    assert not truth_trusted({"truth_xyz": None, "stage": "started"})
    assert "no press point" in truth_note({"truth_xyz": None})
    assert "not aimed close up" in truth_note(dict(aimed, truth_source="fix"))
    assert "not seen to pop" in truth_note(dict(aimed, popped=False))


def test_a_mission_that_saw_nothing_keeps_nothing_and_a_miss_is_never_an_empty_scene(tmp_path):
    from box_opening.detection_set import MissionFrames

    assert MissionFrames(tmp_path, "ankou_pink.yaml", 0.05, now=0).close(exit_code=1) is None
    assert not any(tmp_path.iterdir())
    # an aim whose capture was pruned since: nothing to copy, nothing breaks
    rec = MissionFrames(tmp_path / "m", "ankou_pink.yaml", 0.05, now=0)
    rec.wrist("look", _wrist_arrays())
    rec.located([0.4, 0.0, 0.15], "aim", aim_capture=tmp_path / "gone")
    assert json.loads((rec.close(exit_code=0) / "truth.json").read_text())["frames"] == ["wrist_look_0"]
    # the scene saw the table, nobody found the box: the frame is kept, and
    # flagged — the evaluator skips it rather than counting an empty table
    rec = MissionFrames(tmp_path, "ankou_pink.yaml", 0.05, now=0)
    rec.scene(scene_frame_arrays(), why="the scene OWL saw no box in 5.0 s")
    doc = json.loads((rec.close(exit_code=2) / "truth.json").read_text())
    assert doc["truth_suspect"] is True and doc["truth_xyz"] is None
    assert doc["scene_why"].startswith("the scene OWL") and doc["exit_code"] == 2
    assert doc["scene_source"] is None and doc["scene_boxes"] == {}


def test_a_mission_keeps_which_detector_found_the_box_and_the_boxes_it_gated(tmp_path):
    """The raw boxes of the last message the locate gated, per detector: a
    live run replays exactly (SceneLocator.last_boxes)."""
    from box_opening.detection_set import MissionFrames

    rec = MissionFrames(tmp_path, "ankou_pink.yaml", 0.05, now=0)
    boxes = {
        "segmenter": [["food container", 0.88, 157.75, 216.38, 255.62, 260.5]],
        "owl": [[0.31, 170.0, 195.0, 243.0, 222.0]],
    }
    rec.scene(scene_frame_arrays(), fix_xyz=[0.661, -0.265, 0.31], score=0.88, source="segmenter", boxes=boxes)
    boxes["segmenter"].clear()  # the locator's next look: what the mission kept does not change with it
    doc = json.loads((rec.close(exit_code=0) / "truth.json").read_text())
    assert (
        doc["scene_source"] == "segmenter" and doc["scene_owl"] == 0.88
    )  # the score behind the fix, for the evaluator
    assert doc["scene_boxes"] == {
        "segmenter": [["food container", 0.88, 157.75, 216.38, 255.62, 260.5]],
        "owl": [[0.31, 170.0, 195.0, 243.0, 222.0]],
    }


_CONFIG = __import__("pathlib").Path(__file__).resolve().parents[1] / "config"


def test_the_recorder_refuses_flatbot_before_anything_starts(tmp_path, monkeypatch):
    """world_flatbot.yaml has no surveyed table (the mission measures it per
    run), and sheppy_task.sh always passes it: the recorder refused with a
    TypeError on lid_z after rclpy and the Runner were up. It refuses
    before rclpy now, saying nothing moved."""
    import pytest

    from box_opening.tasks import cli_common
    from box_opening.tasks import record_detection_set as rds
    from box_opening.worlds import WorldStore

    flatbot = str(_CONFIG / "world_flatbot.yaml")
    why = rds.table_refusal(WorldStore(flatbot, out_dir=tmp_path))
    assert why and "needs a surveyed table" in why and "Nothing moved" in why

    def no_rclpy(execute):
        raise AssertionError("must refuse before rclpy")

    monkeypatch.setenv("BOX_OPENING_STATE", str(tmp_path))
    monkeypatch.setattr(cli_common, "init_runtime", no_rclpy)
    monkeypatch.setattr(
        "sys.argv",
        [
            "record_detection_set",
            "--bench-world",
            flatbot,
            "--container",
            str(_CONFIG / "containers" / "ankou_pink.yaml"),
        ],
    )
    with pytest.raises(SystemExit) as e:
        rds.main()
    assert "needs a surveyed table" in str(e.value.code)


def test_the_bench_world_is_not_refused(tmp_path):
    from box_opening.tasks import record_detection_set as rds
    from box_opening.worlds import WorldStore

    worlds = WorldStore(str(_CONFIG / "world_bench.yaml"), out_dir=tmp_path)
    assert isinstance(worlds.table_top_z, float)
    assert rds.table_refusal(worlds) is None


def test_the_recorder_plans_and_watches_in_the_worlds_frames():
    """The planning root and the wrist camera's config come from the world
    (arm_mount and the OAK on flatbot), as build_ctx and press_demo do."""
    import inspect

    from box_opening.tasks import record_detection_set as rds

    src = inspect.getsource(rds.main)
    assert "client._base_frame = worlds.base_frame" in src
    call = src[src.index("BoxTopWatcher(pnode") :]
    call = call[: call.index(")\n")]
    assert "base_frame=worlds.base_frame" in call and "camera_cfg_name=worlds.wrist_camera_config" in call
