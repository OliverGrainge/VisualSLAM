from typing import List, Union

import numpy as np


class PointCloud:
    def __init__(self, poses: List, matching_threshold=250) -> None:
        self.poses = poses
        self.pointcloud = []
        self.pointdescriptors = []

    def __call__(self):
        pass

            

