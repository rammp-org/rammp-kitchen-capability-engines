"""The fridge's handle detector: door_common's, searching the whole image.

At grasp range the door fills the frame, and on the 10-21 frames of this fridge YOLO gave 'cell phone' on the handle and
'chair' on the door, so nothing narrows the image first. Colour-blind.
"""

import door_common.perception
import numpy as np
from door_common.perception import WristCamera  # noqa: F401 -- re-exported for node.py


class HandleDetector(door_common.perception.HandleDetector):
    def door_region(self, rgb, box, debug):
        return np.ones(rgb.shape[:2], dtype=bool)
