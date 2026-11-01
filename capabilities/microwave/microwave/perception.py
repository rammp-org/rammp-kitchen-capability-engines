"""The microwave's handle detector: door_common's, with YOLO or boxer finding where the microwave is.

1. YOLO box around the microwave (COCO 'microwave', else the classes it confuses it with) -- or, with boxer, every
   pixel, keeping the wrist-camera points inside boxer's 3D box.
2. The rest is door_common.perception.HandleDetector: door plane, handle, hinge.

NOTE: YOLO (step 1) is a development stand-in only -- the actual demo will NOT use YOLO; a different
detection model will provide the microwave box. Don't tune the task around YOLO's quirks (e.g. the
red/blue swap in `_box`).
"""

import door_common.perception
import numpy as np
from door_common.common import say
from door_common.perception import WristCamera  # noqa: F401 -- re-exported for node.py and detection_debug

from .constants import YOLO_MODEL

COCO_MICROWAVE_CLASS_ID = 68
# COCO classes YOLO confuses this microwave with (bus 5, train 6, suitcase 28, oven 69,
# refrigerator 72), used only when it finds no 'microwave' at all.
YOLO_FALLBACK_CLASS_IDS = [5, 6, 28, 69, 72]
BOX_THRESHOLD = 0.3

# boxer crop: keep wrist-camera points inside the microwave's 3D box grown by this much along its (x, y, z) axes --
# sideways for the handle in front of the door (boxer's box may not include it), a little upward --
BOX_MARGIN_M = np.array([0.10, 0.10, 0.02])
# -- and from this far ABOVE the box's floor: the table it stands on is in no fit (its points in front of the door
# fall inside the door plane's tolerance and drag the door's edge, i.e. the hinge, out sideways and down)
BOX_FLOOR_CLEAR_M = 0.02
BOX_MIN_POINTS = 500


class HandleDetector(door_common.perception.HandleDetector):
    def __init__(self, model=YOLO_MODEL):
        self._model, self._yolo = model, None

    def _box(self, rgb):
        """Highest-confidence microwave box (x1, y1, x2, y2), or None.
        Tries the image as given AND with red/blue swapped: this red microwave often scores far
        higher swapped (10-24: 0.20 vs 0.77, 0.49 vs 0.67 -- red + boxy reads as bus/train)."""
        if self._yolo is None:
            from ultralytics import YOLO  # here, not at import: loads torch, which the mock and boxer runs never need

            self._yolo = YOLO(self._model)
        res = None
        for img in (rgb, np.ascontiguousarray(rgb[:, :, ::-1])):
            r = self._yolo.predict(img, classes=[COCO_MICROWAVE_CLASS_ID], conf=BOX_THRESHOLD, verbose=False)[0]
            if len(r.boxes) and (res is None or float(r.boxes.conf.max()) > float(res.boxes.conf.max())):
                res = r
        if res is None:
            res = self._yolo.predict(rgb, classes=YOLO_FALLBACK_CLASS_IDS, conf=BOX_THRESHOLD, verbose=False)[0]
            if len(res.boxes) == 0:
                return None
            best = int(res.boxes.conf.argmax())
            say(
                f"  (YOLO fallback: no 'microwave'; using '{res.names[int(res.boxes.cls[best])]}' "
                f"{float(res.boxes.conf[best]):.2f})"
            )
        else:
            best = int(res.boxes.conf.argmax())
            say(f"  microwave {float(res.boxes.conf[best]):.2f}")
        return res.boxes.xyxy[best].cpu().numpy().astype(int)

    def door_region(self, rgb, box, debug):
        """The whole image when cropping to boxer's 3D box, else YOLO's 2D box."""
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        if box is not None:
            mask[:] = True
        else:
            b2d = self._box(rgb)
            if b2d is None:
                say("  no microwave box")
                return None
            x1, y1, x2, y2 = b2d
            if debug is not None:
                debug["box2d"] = (int(x1), int(y1), int(x2), int(y2))
            mask[y1:y2, x1:x2] = True
        return mask

    def crop_points(self, pts, box, base_R, base_t):
        """With `box`: the points inside it, grown by BOX_MARGIN_M."""
        if box is None:
            return pts
        local = (pts @ base_R.T + base_t - box["center"]) @ box["R"]  # base -> the box's own axes
        lo, hi = -box["size"] / 2 - BOX_MARGIN_M, box["size"] / 2 + BOX_MARGIN_M
        lo[2] = -box["size"][2] / 2 + BOX_FLOOR_CLEAR_M  # box z is up (boxer's boxes are gravity-aligned)
        pts = pts[np.all((local >= lo) & (local <= hi), axis=1)]
        say(
            f"  boxer {float(box['score']):.2f}: box {np.round(box['center'], 3)} size {np.round(box['size'], 3)}"
            f" -> {len(pts)} wrist-camera points inside"
        )
        if len(pts) < BOX_MIN_POINTS:
            say(f"  fewer than {BOX_MIN_POINTS} points in the box -- is the microwave in the wrist camera's view?")
            return None
        return pts
