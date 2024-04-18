from typing import List

import numpy as np

from ..Stereo import RgbdPoint, StereoPoint


class KeyFrameInsertion:
    def __init__(self, points: List):
        self.points = points

    def __call__(self, **kwargs):
        if "depth_map" in kwargs.keys():
            frame = RgbdPoint(**kwargs)
        else:
            frame = StereoPoint(**kwargs)

        # decide whether to add frame to psoses
        # if you do
        frame.compute_features2d()
        frame.compute_features3d()
        if len(self.points) == 0: 
            frame.x = np.eye(4)
        self.points.append(frame)
        return True
