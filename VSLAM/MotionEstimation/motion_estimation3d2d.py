import numpy as np
import cv2
from ..utils import homogenize
from typing import List

class MotionEstimation3D2D: 
    def __init__(self, points: List, transformations: List): 
        self.points = points
        self.transformations=transformations

    def estimate(self, frame1, frame2):
        matches = frame1.feature_matcher(des1=frame1.descriptors_3d, des2=frame2.descriptors_2d)
        points3d_sorted = np.zeros((len(matches), 3), dtype=np.float32)
        points2d_sorted = np.zeros((len(matches), 2), dtype=np.float32)
        points_desc = np.zeros((len(matches), frame1.descriptors_3d.shape[1]), dtype=np.float32)
        for i, match in enumerate(matches):
            points3d_sorted[i, :] = frame1.keypoints_3d[match.queryIdx]
            points2d_sorted[i, :] = frame2.keypoints_2d[match.trainIdx].pt
            points_desc[i, :] = frame1.descriptors_3d[match.queryIdx]

        if len(points3d_sorted) < 8:
            return None
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
            points3d_sorted,
            points2d_sorted,
            frame2.K,
            np.zeros(4),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None
        if len(inliers)/len(points3d_sorted) < 0.9:
            return None
        T = np.linalg.inv(homogenize(rotation_vector, translation_vector))
        return T


    def __call__(self):
        if len(self.points) <= 1: 
            return
        frame1, frame2 = self.points[-2], self.points[-1]
        matches = frame1.feature_matcher(des1=frame1.descriptors_3d, des2=frame2.descriptors_2d)
        points3d_sorted = np.zeros((len(matches), 3), dtype=np.float32)
        points2d_sorted = np.zeros((len(matches), 2), dtype=np.float32)
        points_desc = np.zeros((len(matches), frame1.descriptors_3d.shape[1]), dtype=np.float32)
        for i, match in enumerate(matches):
            points3d_sorted[i, :] = frame1.keypoints_3d[match.queryIdx]
            points2d_sorted[i, :] = frame2.keypoints_2d[match.trainIdx].pt
            points_desc[i, :] = frame1.descriptors_3d[match.queryIdx]

        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
            points3d_sorted,
            points2d_sorted,
            frame2.K,
            np.zeros(4),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            raise Exception("PnP algorithm failed")
        T = np.linalg.inv(homogenize(rotation_vector, translation_vector))
        self.transformations.append(T)
        new_pose = self.points[-2].x @ T
        self.points[-1].x = new_pose



