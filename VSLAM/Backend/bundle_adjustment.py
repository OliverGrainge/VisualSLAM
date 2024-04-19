from typing import List, Union

import numpy as np


from typing import List, Union
from ..MotionEstimation import MotionEstimation3D2D, MotionEstimation3D3D
from ..utils import homogenize, unhomogenize
import numpy as np
from scipy.optimize import least_squares
import cv2
from typing import Tuple, Dict


def pointspose2params(poses: List, points: np.ndarray) -> Tuple[np.ndarray, int]:
    params = []
    for pose in poses:
        rvec, tvec = unhomogenize(pose)
        params += list(rvec)
        params += list(tvec)
    params += list(points.flatten())
    n_poses = len(poses)
    return np.array(params), n_poses


def params2posepoints(params: np.ndarray, n_poses) -> List[np.ndarray]:
    pose_params = params[:n_poses * 6]
    point_params = params[n_poses * 6:]
    poses = []
    for i in n_poses: 
        rvec = pose_params[i*6:i*6+3]
        tvec = pose_params[i*6+3:i*6+6]
        poses.append(homogenize(rvec, tvec))
    points = np.array(point_params).reshape(-1, 3)
    return poses, points


class BundleAdjustment:
    def __init__(self, points: List, map: Dict, loop_closures: List, window: int = 5) -> None:
        self.poses = points
        self.loop_closures = loop_closures
        self.window = window


    def data_assosiation(self):
        if len(self.points[-self.window:]) < self.window:
            return None
        
        matchIdxs = []
        desc3d = np.vstack(self.map["local_descriptors"])
        for frame in self.points[-self.window:-1]:
            frame_matches = []
            desc2d = frame.descriptors_2d
            matches = frame.feature_matcher(desc3d, desc2d)
            for match in matches:
                frame_matches.append([match.queryIdx, match.trainIdx])
            matchIdxs.append(np.array(frame_matches))
        return matchIdxs




    def optimize(
        loop_detection: Union[np.ndarray, bool] = False, window: Union[None, int] = None
    ) -> None:
        """
        performs
        """
        pass



