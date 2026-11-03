"""The fridge's handle detector: door_common's, searching the whole image, with no door-edge guess.

At grasp range the door fills the frame, and on the 10-21 frames of this fridge YOLO gave 'cell phone' on the handle and
'chair' on the door, so nothing narrows the image first. Colour-blind. The door's hinge edge isn't in view, so the hinge
is the measured one (hinge_measured) and door_common's two looks compare that, not an edge read off the image.
"""

import door_common.perception
import numpy as np
from door_common.perception import WristCamera  # noqa: F401 -- re-exported for node.py


class HandleDetector(door_common.perception.HandleDetector):
    def __init__(self, hinge_offset):
        self.hinge_offset = hinge_offset  # from the handle, in the door's frame: into it, along it, up

    def door_region(self, rgb, box, debug):
        return np.ones(rgb.shape[:2], dtype=bool)

    def detect(self, *args, **kwargs):
        """door_common's, with `hinge` the measured hinge from this look's handle and door normal."""
        det = super().detect(*args, **kwargs)
        if det is not None:
            n = det["normal"]
            into, along, up = self.hinge_offset
            det["hinge"] = det["handle"] - n * into + np.cross(n, [0.0, 0.0, 1.0]) * along + np.array([0.0, 0.0, up])
        return det
