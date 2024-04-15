import numpy as np 
from typing import List 
from Stereo import StereoPoint, RgbdPoint


class KeyFrameInsertion: 
    def __init__(self, poses: List):
        self.poses = poses

    def process(self, **kwargs):
        if "depth_map" in kwargs.keys():
            frame = RgbdPoint(**kwargs)
        else: 
            frame = StereoPoint(**kwargs)

        # decide whether to add frame to poses 
        # if you do 
        insert = True 
        if insert: 
            self.poses.append(frame)
            return True 
        else: 
            return False




