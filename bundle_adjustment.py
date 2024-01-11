import scipy
import numpy as np
from collections import namedtuple, deque



class BundleAdjustment: 
    def __init__(self, cam_calib: np.ndarray, window_size: int=3):
        self.cam_calib = cam_calib
        self.points3d = deque(maxsize=window_size)
        self.features3d = deque(maxsize=window_size)
        self.features2d = deque(maxsize=window_size)
        self.camera_poses = deque(maxsize=window_size)
        self.camera_obs = deque(maxsize=window_size)
        self.data_association = []    

    def insert_observations(self, 
                            camera_pose: np.ndarray, 
                            obs_pos: np.ndarray, 
                            points_pos: np.ndarray, 
                            point_feature: np.ndarray): 

        self.camera_poses.insert(0, camera_pose)
        self.points3d.insert(0, points_pos)
        self.features3d.insert(0, point_feature)
        self.camera_obs.insert(0, obs_pos)

        self.data_association()

    def data_assosiation(self):
        all_3d_features = np.hstack([feat for feat in list(self.features3d)])
        for features2d in list(self.features2d):
            matches = self.matcher.match(features2d, all_3d_features)
            self.data_association.append(np.array([[m.queryIdx, m.trainIdx] for m in matches]))
        
    def cost_function(self):
        error = 0
        camera_poses = list(self.camera_poses)
        all_3d_points = np.hstack([points for points in list(self.points3d)])
        points2d = list(self.camera_obs)
        for i, camera_pos in enumerate(camera_poses): 
            points3d_viewed = all_3d_points[self.data_association[i][:, 1]]
            points2d_obs = self.camera_obs[i][self.data_association[:, 0]]
            for j, point3d in enumerate(points3d_viewed):
                proj = self.cam_calib @ camera_pos @ point3d
                error += np.sum((points2d_obs[j] - proj)**2)
        return error

    def optimizer()






        
