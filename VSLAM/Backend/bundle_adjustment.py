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
        params += list(rvec.flatten())
        params += list(tvec.flatten())
    params += list(points.flatten())
    return np.array(params)


def params2posepoints(params: np.ndarray, n_poses) -> List[np.ndarray]:
    pose_params = params[:n_poses * 6]
    point_params = params[n_poses * 6:]
    poses = []
    for i in range(n_poses): 
        rvec = pose_params[i*6:i*6+3]
        tvec = pose_params[i*6+3:i*6+6]
        poses.append(homogenize(rvec, tvec))
    points = np.array(point_params).reshape(-1, 3)
    return poses, points


def huber_loss(residual, delta=1.0):
    if np.abs(residual) <= delta:
        return 0.5 * residual**2
    else:
        return delta * (np.abs(residual) - 0.5 * delta)


def tukey_biweight(residual, c=4.685):
    if np.abs(residual) <= c:
        x_c = residual / c
        return (c**2 / 6) * (1 - (1 - x_c**2)**3)
    else:
        return (c**2 / 6)


def cauchy_loss(residual, c=2.384):
    return c**2 / 2 * np.log(1 + (residual / c)**2)





class BundleAdjustment:
    def __init__(self, points: List, map: Dict, loop_closures: List, window: int = 10) -> None:
        self.points = points
        self.loop_closures = loop_closures
        self.window = window
        self.map = map


    def data_assosiation(self):
        if len(self.points[-self.window:]) < self.window:
            return None
        # k, i, j
        # k = the frame
        # i = the point 
        # j = the point in 3d 
        des3d = self.map["local_descriptors"][:50]
        print("============================== MAP SIZE", len(des3d))
        corr = []

        for frame in self.points[-self.window:]:
            matches = frame.feature_matcher(
                des1=des3d,
                des2=frame.descriptors_2d,
                apply_lowe=True
            )
            point_corr = []
            for match in matches[:40]:
                j = match.queryIdx
                i = match.trainIdx
                point_corr.append([i, j])
            corr.append(np.array(point_corr))
        return corr


    def cost_function(self, params, corr, n_poses, frames):
        errors = []
        poses, points = params2posepoints(params, n_poses)
        for idx, frame_match in enumerate(corr):
            rvec, tvec = unhomogenize(poses[idx])
            point3d = points[frame_match[:, 1]]
            point2d = np.array([frames[idx].keypoints_2d[i].pt for i in frame_match[:, 0]])
            proj, _ = cv2.projectPoints(point3d.T, rvec, tvec, frames[idx].K, np.zeros(5))
            res = point2d.flatten() - proj.flatten()
            for val in res: 
                if np.abs(val) < 10:
                    errors.append(huber_loss(val))
       # print(np.mean(errors), np.min(errors), np.max(errors))
        return np.array(errors)


    def __call__(
        self, loop_detection: Union[np.ndarray, bool] = False, window: Union[None, int] = None
    ) -> None:
        """
        performs
        """
        if len(self.points[-self.window:]) < self.window:
            return
        
        corr = self.data_assosiation()
        poses = [frame.x for frame in self.points[-self.window:]]
        n_poses = len(poses)
        points = self.map["local_points"][:50]
        params = pointspose2params(poses, points)

        
        self.cost_function(params, corr, n_poses, self.points[-self.window:])
        
        
        result = least_squares(
            self.cost_function,
            params,
            args=(
                corr, 
                n_poses,
                self.points[-self.window:],

            ), 
            max_nfev=1,
            verbose=1,
            #method='lm',
            #ftol=1e-2   
        )
        

        new_poses, new_points = params2posepoints(result.x, n_poses)
        self.map["local_points"][:50] = new_points
        for idx, point in enumerate(self.points[-self.window:]):
            movement = np.linalg.norm(point.x[:3, 3] - new_poses[idx][:3, 3])
            if movement < 0.1:
                print("optimizing pose")
                point.x = new_poses[idx]
        #print(np.allclose(poses[0].flatten(), new_poses[0].flatten()))




