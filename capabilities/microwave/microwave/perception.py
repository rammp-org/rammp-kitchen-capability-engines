"""The microwave's handle detector: door_common's, with a box saying where the microwave is.

1. Where the microwave is: a 3D box -- boxer's, or (open, without boxer) the door the scene camera found inside the
   scene segmenter's 2D microwave box -- keeping the wrist-camera points inside it; or a 2D box from another detector
   (`region_2d`: the scene segmenter's, for the scene camera's own look at the door). No box: no detection, unless
   `whole_image` (detection_debug's replay of saved frames, which keep no box).
2. The rest is door_common.perception.HandleDetector: door plane, handle, hinge.
"""

import door_common.perception
import numpy as np
from door_common.common import say
from door_common.perception import WristCamera  # noqa: F401 -- re-exported for node.py and detection_debug

# boxer crop: keep wrist-camera points inside the microwave's 3D box grown by this much along its (x, y, z) axes --
# sideways for the handle in front of the door (boxer's box may not include it), a little upward --
BOX_MARGIN_M = np.array([0.10, 0.10, 0.02])
# -- and from this far ABOVE the box's floor: the table it stands on is in no fit (its points in front of the door
# fall inside the door plane's tolerance and drag the door's edge, i.e. the hinge, out sideways and down)
BOX_FLOOR_CLEAR_M = 0.06
BOX_MIN_POINTS = 500


class HandleDetector(door_common.perception.HandleDetector):
    def __init__(self, whole_image=False):
        self.region_2d = None  # (x1, y1, x2, y2) from another detector (the scene segmenter)
        self.whole_image = whole_image  # no box: fit the whole image (the replay tool) instead of refusing

    def door_region(self, rgb, box, debug):
        """The whole image when cropping to a 3D box, else `region_2d` when set, else (whole_image) the whole image;
        otherwise None: nothing says where the microwave is."""
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        if box is not None or (self.region_2d is None and self.whole_image):
            mask[:] = True
            return mask
        if self.region_2d is None:
            say("  no microwave box (no boxer box, no scene-camera door) -- nothing to look for the handle in")
            return None
        x1, y1, x2, y2 = self.region_2d
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
            f"  {box.get('source', 'boxer')} {float(box['score']):.2f}: box {np.round(box['center'], 3)} "
            f"size {np.round(box['size'], 3)} -> {len(pts)} wrist-camera points inside"
        )
        if len(pts) < BOX_MIN_POINTS:
            say(f"  fewer than {BOX_MIN_POINTS} points in the box -- is the microwave in the wrist camera's view?")
            return None
        return pts
