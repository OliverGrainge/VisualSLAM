import numpy as np
import cv2
from typing import Union
from typing import List 
from ..utils import homogenize

class MotionEstimation3D3D: 
    def __init__(self, points: List, transformations: List): 
        self.points = points
        self.transformations=transformations

    def __call__(self):
        if len(self.points) <= 1: 
            return
        frame1, frame2 = self.points[-2], self.points[-1]
        matches = frame1.feature_matcher(des1=frame1.descriptors_3d, des2=frame2.descriptors_3d)
        frame1_pts = np.zeros((len(matches), 3), dtype=np.float32)
        frame2_pts = np.zeros((len(matches), 3), dtype=np.float32)
        for i, match in enumerate(matches):
            frame1_pts[i, :] = frame1.keypoints_3d[match.queryIdx]
            frame2_pts[i, :] = frame2.keypoints_3d[match.trainIdx]

        success, T, inliers = cv2.estimateAffine3D(frame1_pts, frame2_pts, ransacThreshold=0.05, confidence=0.99)


        if not success: 
            raise Exception("Affine Transformation not found")
        T = np.vstack((T, np.array([0, 0, 0, 1])))
        T = np.linalg.inv(T)
        self.transformations.append(T)
        new_pose = self.points[-2].x @ T
        self.points[-1].x = new_pose


