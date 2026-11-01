"""The lid-off's video of what the wrist and the scene cameras see
(perception/camera_video): owner 2026-10-29, "create a video of what the
wrist and scene cameras see during the lid placement so i can see too"."""

import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from box_opening.models.container import ContainerPose
from box_opening.perception import camera_video as cvid
from box_opening.tasks import reach_lid_off as rlo


def _cam(grey, h, w, stamp=1):
    return SimpleNamespace(color=np.full((h, w, 3), grey, np.uint8), color_stamp=stamp)


def test_the_two_cameras_side_by_side_under_the_caption_are_written_as_an_mp4_that_plays_in_real_time(tmp_path):
    """Each camera's newest frame fitted into its panel (the wrist's 4:3,
    the scene's 16:9), wrist left and scene right under the caption; the
    MP4 plays as long as the recording ran, the last picture held a
    second, and the newest VIDEOS_KEPT are kept."""
    wrist, scene = _cam(80, 480, 640), _cam(160, 720, 1280)
    for i in range(cvid.VIDEOS_KEPT + 2):
        (tmp_path / ("lid-video-20261029-0000%02d.mp4" % i)).write_bytes(b"old")
    v = cvid.CameraVideo([("wrist", wrist), ("scene", scene)], tmp_path, fps=20).start()
    v.mark("GRIP: the fingers open on the way down to the knob")
    t0 = time.monotonic()
    while time.monotonic() - t0 < 0.6:
        scene.color_stamp += 1
        time.sleep(0.02)
    path = v.finish("the end: the lid set down, the arm home")
    assert path is not None and path.suffix == ".mp4" and path.stat().st_size > 0
    cap = cv2.VideoCapture(str(path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, img = cap.read()
    assert ok and (w, h) == (2 * cvid.PANEL_W, cvid.CAPTION_H + cvid.PANEL_H)
    assert n == len(v.timeline()) and n / 20.0 == pytest.approx(0.6 + 1.0, abs=0.25)
    mid = cvid.CAPTION_H + cvid.PANEL_H // 2
    assert int(img[mid, cvid.PANEL_W // 2].mean()) == pytest.approx(80, abs=12)  # the wrist, left
    assert int(img[mid, cvid.PANEL_W + cvid.PANEL_W // 2].mean()) == pytest.approx(160, abs=12)  # the scene, right
    # the 16:9 scene fills its 16:9 panel; the 4:3 wrist fills its height, black beside it
    assert int(img[mid, cvid.PANEL_W + 10].mean()) == pytest.approx(160, abs=12)
    assert int(img[mid, 10].mean()) < 20
    assert len(list(tmp_path.glob("lid-video-*"))) == cvid.VIDEOS_KEPT and path.exists()
    assert v.finish("again") == path  # once only


def test_a_camera_with_no_frame_is_a_black_panel_and_no_camera_records_nothing(tmp_path):
    v = cvid.CameraVideo(
        [("wrist", SimpleNamespace(color=None, color_stamp=None)), ("scene", _cam(200, 720, 1280))], tmp_path
    )
    v._t0 = time.monotonic()
    img = cv2.imdecode(np.frombuffer(v.compose(time.monotonic()), np.uint8), cv2.IMREAD_COLOR)
    mid = cvid.CAPTION_H + cvid.PANEL_H // 2
    assert int(img[mid, cvid.PANEL_W // 2].mean()) < 20 and int(img[mid, cvid.PANEL_W * 3 // 2].mean()) > 180
    none = cvid.CameraVideo([("wrist", None), ("scene", None)], tmp_path).start()
    assert none._thread is None and none.finish("the end") is None and not list(tmp_path.iterdir())


def test_a_sample_taken_late_shows_the_picture_before_it_so_the_video_keeps_real_time():
    v = cvid.CameraVideo([], ".", fps=10)
    v.frames = [(0.0, b"a"), (0.25, b"b"), (0.30, b"c")]
    assert v.timeline() == [b"a", b"a", b"a", b"c"]


def test_runs_record_it_unless_told_not_to():
    assert cvid.enabled({}) and cvid.enabled({"BOX_OPENING_VIDEO": "1"})
    assert not cvid.enabled({"BOX_OPENING_VIDEO": "0"})


def test_the_lid_s_spot_is_drawn_where_the_scene_camera_sees_that_place():
    """The scene camera 1 m over the spot looking straight down (fx 500,
    centre 320,240); the box the scene camera put 20 mm further along x
    than the wrist's aim: the rim is drawn 20 mm along x of the spot (10
    px), the lid's radius round it."""
    marks = {}
    video = SimpleNamespace(overlay=lambda name, uv, text: marks.update({name: (np.asarray(uv), text)}))
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = [0.5, 0.0, 1.0]
    loc = SimpleNamespace(
        base_to_color=lambda: T, grab=SimpleNamespace(k=np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]]))
    )
    ctx = SimpleNamespace(
        lid_video=video,
        scene_locator=loc,
        model=SimpleNamespace(lid_dims=(0.10, 0.10, 0.03)),
        cpose=ContainerPose(xyz=(0.70, 0.0, 0.0), yaw=0.0),
    )
    fix = SimpleNamespace(pose=ContainerPose(xyz=(0.72, 0.0, 0.0), yaw=0.0))
    rlo._video_spot(ctx, (None, None, fix, None), ContainerPose(xyz=(0.5, 0.0, 0.0), yaw=0.0))
    uv, text = marks["scene"]
    assert text == "the lid goes here"
    assert uv.mean(axis=0) == pytest.approx([330.0, 240.0], abs=0.5)
    assert np.linalg.norm(uv - uv.mean(axis=0), axis=1) == pytest.approx(25.0, abs=0.5)
    # no scene camera's pose: nothing drawn, nothing raised
    marks.clear()
    loc.base_to_color = lambda: None
    rlo._video_spot(ctx, (None, None, fix, None), ContainerPose(xyz=(0.5, 0.0, 0.0), yaw=0.0))
    assert not marks
