from typing import List, Union
from ..MotionEstimation import MotionEstimation3D2D, MotionEstimation3D3D
from ..utils import homogenize, unhomogenize
import numpy as np
from scipy.optimize import least_squares
import cv2


def pose2params(poses: List) -> np.ndarray:
    params = []
    for pose in poses:
        rvec, tvec = unhomogenize(pose)
        params += list(rvec.flatten())
        params += list(tvec.flatten())
    return np.array(params)

def params2pose(params: np.ndarray) -> List[np.ndarray]:
    n_poses = len(params) // 6
    poses = []
    for idx in range(n_poses):
        poses.append(
            homogenize(
                params[idx * 6 : idx * 6 + 3],
                params[idx * 6 + 3 : idx * 6 + 6],
            )
        )
    return poses

def huber_loss(residual, delta=1.0):
    if np.abs(residual) <= delta:
        return 0.5 * residual**2
    else:
        return delta * (np.abs(residual) - 0.5 * delta)


class PoseGraphOptimization:
    def __init__(self, points: List, transformations: List, loop_closures: List, window=4) -> None:
        self.points = points
        self.transformations = transformations
        self.loop_closures = loop_closures
        self.window=window
        self.motion_estimation = MotionEstimation3D2D(self.points, self.transformations)

    def __call__(
        self, loop_detection: Union[np.ndarray, None] = None
    ) -> None:
        """
        performs global optimization if loop closure detected,
        otherwise performs local optimization
        """
        pose_graph = self.get_pose_graph()
        if pose_graph is not None: 
            poses = self.get_poses()
            params = pose2params(poses)
            try:
                result = least_squares(
                    self.cost_function,
                    params, 
                    args=(pose_graph,),
                )
                opt_poses = params2pose(result.x)
                for i, pt in enumerate(self.points[-self.window:]):
                    if np.linalg.norm(pt.x[:3, 3] - opt_poses[i][:3, 3]) < 0.02:
                        pt.x = opt_poses[i]
                        print("optimizing pose")
            except: 
                pass
                

    @staticmethod
    def cost_function(params, pose_graph) -> np.ndarray:
        errors = []
        poses = params2pose(params)
        for i in range(len(pose_graph)):
            for j in range(len(pose_graph[0])):
                if i != j and pose_graph[i][j] is not None:
                    pred = poses[i] @ pose_graph[i][j]
                    targ = poses[j]
                    rvec_pred, _ = cv2.Rodrigues(pred[:3, :3])
                    tvec_pred = pred[:3, 3].flatten()
                    rvec_targ, _ = cv2.Rodrigues(targ[:3, :3])
                    tvec_targ = targ[:3, 3].flatten()
                    errors += [huber_loss(val) for val in (rvec_pred.flatten() - rvec_targ.flatten())]
                    errors += [huber_loss(val) for val in (tvec_pred - tvec_targ)]
        return np.array(errors)


    def get_poses(self):
        return [pt.x for pt in self.points[-self.window:]]
    
    def get_pose_graph(self):
        n = len(self.points[-self.window:])
        if n < self.window: 
            return None 
        
        pose_graph = [[None for _ in range(self.window)] for _ in range(self.window)]
        for i in range(self.window):
            for j in range(self.window):
                if i != j: 
                    T = self.motion_estimation.estimate(
                        self.points[-self.window:][i], 
                        self.points[-self.window:][j]
                    )
                    pose_graph[i][j] = T 


        return pose_graph 


