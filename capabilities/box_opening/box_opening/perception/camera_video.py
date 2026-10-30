"""A video of what the wrist and the scene cameras see, side by side, while
the lid comes off and is set down (owner 2026-10-29: "create a video of what
the wrist and scene cameras see during the lid placement so i can see too").

The frames come from the grabbers the mission already holds, so nothing new
is subscribed: VIDEO_FPS times a second each camera's newest colour frame is
fitted into a PANEL_W x PANEL_H panel, the panels side by side under a
caption (the seconds since the start and the step under way), and the
picture is kept JPEG-encoded in memory. Nothing is written while the arm
moves (detection_set.MissionFrames' rule): finish() writes the MP4 once the
run has ended, with ffmpeg's H.264 where there is an ffmpeg, else OpenCV's.
The sampling thread runs at nice 19 (the arm driver's real-time loop has
CPU 11 to itself, isolcpus). BOX_OPENING_VIDEO=0 turns it off."""

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

VIDEO_FPS = 10
PANEL_W, PANEL_H = 640, 360
CAPTION_H = 30
# at most this long: a run that stops holding the lid keeps its first minutes
VIDEO_MAX_S = 180.0
JPEG_QUALITY = 85
VIDEOS_KEPT = 20  # newest videos kept of each tag; older ones are pruned
STALE_S = 1.0  # a camera with no new frame for this long says so on its panel
ENCODE_TIMEOUT_S = 300.0


def enabled(env=None):
    """Whether runs record the video: on unless BOX_OPENING_VIDEO=0."""
    env = os.environ if env is None else env
    return env.get("BOX_OPENING_VIDEO", "1").strip() != "0"


def _stamp_key(stamp):
    """A frame stamp as something == compares: (sec, nanosec) of a ROS Time."""
    if stamp is None:
        return None
    sec = getattr(stamp, "sec", None)
    return stamp if sec is None else (int(sec), int(getattr(stamp, "nanosec", 0)))


def _fit(img):
    """`img` (BGR, or grey) scaled to fit a PANEL_W x PANEL_H panel, centred:
    (the panel, the scale, (x, y) of the image's corner in it)."""
    import cv2

    panel = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
    img = np.asarray(img)
    if img.ndim == 2:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    s = min(PANEL_W / float(w), PANEL_H / float(h))
    sw, sh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    x, y = (PANEL_W - sw) // 2, (PANEL_H - sh) // 2
    panel[y : y + sh, x : x + sw] = cv2.resize(img[:, :, :3], (sw, sh), interpolation=cv2.INTER_AREA)
    return panel, s, (x, y)


def _text(img, text, org, scale=0.55, color=(255, 255, 255)):
    """`text` at `org`, white on a black outline: readable on any picture."""
    import cv2

    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


class CameraVideo:
    """The cameras `sources` ((name, grabber with .color and .color_stamp),
    left to right) recorded from start() to finish(), the MP4 written into
    `out_dir` as <tag>-<when it started>.mp4. mark() sets the caption;
    overlay() draws a line on one camera's panel (the lid's spot)."""

    def __init__(self, sources, out_dir, tag="lid-video", fps=VIDEO_FPS, max_s=VIDEO_MAX_S):
        self.sources = [(str(n), g) for n, g in sources if g is not None]
        self.out_dir = Path(out_dir)
        self.tag = tag
        self.fps = float(fps)
        self.max_frames = int(max_s * self.fps)
        self.caption = ""
        self.frames = []  # (seconds since start, JPEG bytes), in order
        self.path = None  # the MP4 finish() wrote
        self._overlays = {}  # source name -> (polyline in its full frame's pixels, text)
        self._seen = {}  # source name -> (its newest stamp, when that stamp first came)
        self._stop = threading.Event()
        self._thread = None
        self._t0 = None
        self._when = None
        self._done = False

    def start(self):
        """Start recording (on a thread of its own). Returns self."""
        if self._thread is None and self.sources:
            self._t0 = time.monotonic()
            self._when = time.strftime("%Y%m%d-%H%M%S")
            self._thread = threading.Thread(target=self._loop, name="camera-video", daemon=True)
            self._thread.start()
        return self

    def mark(self, text):
        """The caption from now on: the step under way."""
        self.caption = str(text)

    def overlay(self, name, uv, text=""):
        """A closed line through pixels `uv` (N x 2, the camera `name`'s full
        frame) drawn on its panel from now on, `text` beside it; None clears it."""
        self._overlays[name] = None if uv is None else (np.asarray(uv, float).reshape(-1, 2), str(text))

    def _loop(self):
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 19)  # this thread only (Linux)
        except (AttributeError, OSError):
            pass
        period = 1.0 / self.fps
        nxt = time.monotonic()
        while not self._stop.is_set() and len(self.frames) < self.max_frames:
            now = time.monotonic()
            try:
                jpg = self.compose(now)
            except Exception:  # a picture that cannot be made is one frame short, never the run's problem
                jpg = None
            if jpg is not None:
                self.frames.append((now - self._t0, jpg))
            nxt = max(nxt + period, time.monotonic())  # behind, it drops samples: finish() keeps the timeline
            self._stop.wait(max(0.0, nxt - time.monotonic()))

    def compose(self, now):
        """One picture: each camera's panel, the caption over them. JPEG bytes."""
        import cv2

        canvas = np.zeros((CAPTION_H + PANEL_H, PANEL_W * len(self.sources), 3), np.uint8)
        for i, (name, g) in enumerate(self.sources):
            img = getattr(g, "color", None)
            key = _stamp_key(getattr(g, "color_stamp", None))
            last = self._seen.get(name)
            if last is None or last[0] != key:
                self._seen[name] = last = (key, now)
            label = name
            if img is None:
                panel = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
                label += " - no frame"
            else:
                panel, s, (x0, y0) = _fit(img)
                ov = self._overlays.get(name)
                if ov is not None and len(ov[0]):
                    pts = np.round(ov[0] * s + [x0, y0]).astype(np.int32)
                    cv2.polylines(panel, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2, cv2.LINE_AA)
                    if ov[1]:
                        top = pts[np.argmin(pts[:, 1])]
                        _text(panel, ov[1], (int(top[0]) - 40, max(14, int(top[1]) - 8)), 0.5, (0, 255, 0))
                if key is not None and now - last[1] > STALE_S:
                    label += " - no new frame for %.1f s" % (now - last[1])
            _text(panel, label, (8, PANEL_H - 10))
            canvas[CAPTION_H:, i * PANEL_W : (i + 1) * PANEL_W] = panel
        _text(canvas, "%5.1f s  %s" % (now - self._t0, self.caption), (8, CAPTION_H - 9), 0.6)
        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return buf.tobytes() if ok else None

    def stop(self):
        """Stop sampling; nothing is written."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)

    def timeline(self):
        """The frames as the video plays them, VIDEO_FPS a second: each slot
        the newest picture taken by then (a sample the thread was late for
        shows the one before it, so the video runs in real time)."""
        out, j = [], 0
        if not self.frames:
            return out
        for k in range(int(self.frames[-1][0] * self.fps) + 1):
            while j + 1 < len(self.frames) and self.frames[j + 1][0] <= k / self.fps:
                j += 1
            out.append(self.frames[j][1])
        return out

    def finish(self, caption=None, hold_s=1.0):
        """Stop and write the MP4: its path, or None with no picture taken or
        none written. `caption`: the last picture's, held `hold_s` (how the
        run ended). Once only. Run it once the arm no longer needs this
        process."""
        if self._done:
            return self.path
        self._done = True
        self.stop()
        if caption is not None and self.frames:
            self.mark(caption)
            now = time.monotonic()
            try:
                jpg = self.compose(now)
            except Exception:  # the last picture is the one before
                jpg = None
            if jpg is not None:
                self.frames += [(now - self._t0, jpg), (now - self._t0 + hold_s, jpg)]
        frames = self.timeline()
        if not frames:
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / ("%s-%s.mp4" % (self.tag, self._when))
        self.path = _ffmpeg(frames, self.fps, path) or _opencv(frames, self.fps, path)
        _prune(self.out_dir, self.tag)
        return self.path


def _ffmpeg(frames, fps, path):
    """The JPEGs `frames` into an H.264 MP4 at `path` by ffmpeg (at nice 19):
    `path`, or None without an ffmpeg or where it failed."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        return None
    cmd = [
        exe,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "image2pipe",
        "-framerate",
        "%g" % fps,
        "-c:v",
        "mjpeg",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "26",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    nice = shutil.which("nice")
    try:
        r = subprocess.run(
            ([nice, "-n", "19"] if nice else []) + cmd,
            input=b"".join(frames),
            capture_output=True,
            timeout=ENCODE_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return path if r.returncode == 0 and path.exists() and path.stat().st_size > 0 else None


def _opencv(frames, fps, path):
    """The JPEGs `frames` by OpenCV: an MP4 (mp4v) at `path`, else a motion
    JPEG .avi beside it. The file's path, or None."""
    import cv2

    first = cv2.imdecode(np.frombuffer(frames[0], np.uint8), cv2.IMREAD_COLOR)
    size = (first.shape[1], first.shape[0])
    for p, code in ((path, "mp4v"), (path.with_suffix(".avi"), "MJPG")):
        w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*code), fps, size)
        if not w.isOpened():
            continue
        for f in frames:
            w.write(cv2.imdecode(np.frombuffer(f, np.uint8), cv2.IMREAD_COLOR))
        w.release()
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def _prune(out_dir, tag, kept=VIDEOS_KEPT):
    """Keep the newest `kept` of <tag>-* in `out_dir`."""
    try:
        for f in sorted(Path(out_dir).glob("%s-*" % tag))[:-kept]:
            f.unlink()
    except OSError:
        pass
