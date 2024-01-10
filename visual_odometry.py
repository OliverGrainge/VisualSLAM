
from PIL import Image
from typing import Tuple, List
import numpy as np
import utils
from PIL import Image
import cv2


class BaseOdometry: 
    def __init__(self, config: dict, intrinsic_calib: np.ndarray, projection_calib: np.ndarray):
        self.config = config
        self.intrinsic_calib = intrinsic_calib
        self.projection_calib = projection_calib

    @staticmethod
    def _from_transf(R, t):
        
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        
        return T


class VisualOdometryBase(BaseOdometry):
    def __init__(self, config: dict, intrinsic_calib: np.ndarray, projection_calib: np.ndarray):
        super().__init__(config=config, intrinsic_calib=intrinsic_calib, projection_calib=projection_calib)
        self.feature_extractor = utils.get_feature_extractor(config)
        self.feature_matcher = utils.get_feature_matcher(config)

    def extract_features(self, image: Image) -> Tuple[np.ndarray, np.ndarray]:
        image = np.array(image)
        keypoints, descriptors = self.feature_extractor.detectAndCompute(image, None)
        return (keypoints, descriptors)

    def match_features(self, desc1: Tuple[np.ndarray, np.ndarray], desc2: Tuple[np.ndarray, np.ndarray]) -> List:
        matches = self.feature_matcher.match(desc1[1], desc2[1])
        matches = sorted(matches, key=lambda x: x.distance)
        pts1 = np.float32([desc1[0][m.queryIdx].pt for m in matches])
        pts2 = np.float32([desc2[0][m.trainIdx].pt for m in matches])
        desc1 = np.float32([desc1[1][m.queryIdx] for m in matches])
        desc2 = np.float32([desc2[1][m.trainIdx] for m in matches])
        return [pts1, pts2], [desc1, desc2]

    def triangulate_points(self, R: np.ndarray, t: np.ndarray, matches: List[np.ndarray], features: List[np.ndarray]):
        # Triangulate points
        points1_norm = cv2.undistortPoints(np.expand_dims(matches[0], axis=1), self.intrinsic_calib, None)
        points2_norm = cv2.undistortPoints(np.expand_dims(matches[1], axis=1), self.intrinsic_calib, None)
        # Combine rotation and translation into a 3x4 transformation matrix
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = np.hstack((R, t))
        # Triangulate points to get 4D homogeneous coordinates
        points_4d = cv2.triangulatePoints(P1, P2, points1_norm, points2_norm)
        # Convert homogeneous coordinates to Cartesian coordinates
        points_3d = cv2.convertPointsFromHomogeneous(points_4d.T)
        # Extract the 3D coordinates
        triangulated_points = points_3d.squeeze()
        triangulated_features = features[0]
        return triangulated_points, triangulated_features




class VisualOdometry(VisualOdometryBase):
    def __init__(self, config: dict, intrinsic_calib: np.ndarray, projection_calib: np.ndarray, initial_pose: np.ndarray):
        super().__init__(config=config, intrinsic_calib=intrinsic_calib, projection_calib=projection_calib)
        self.config = config
        self.poses = []
        self.current_pose = initial_pose
        self.poses.append(initial_pose)
        self.mapping_points = []
        self.mapping_features = []
        self.frame_count = 0
        self.old_frame = 0
        self.old_features = 0

    def get_pose(self, matches: Tuple[np.ndarray, np.ndarray], features: np.ndarray) -> np.ndarray:
        E, mask = cv2.findEssentialMat(matches[0], matches[1], 
                                    focal=self.intrinsic_calib[0,0],
                                    pp=(self.intrinsic_calib[0, 2], self.intrinsic_calib[1, 2]),
                                    method=cv2.RANSAC,
                                    prob=0.999,
                                    threshold=1)
        matches[0] = matches[0][mask.ravel() == 1]
        matches[1] = matches[1][mask.ravel() == 1]
        features[0] = features[0][mask.ravel() == 1]
        features[1] = features[1][mask.ravel() == 1]

        _, R, t, mask = cv2.recoverPose(E, matches[0], matches[1], 
                                focal=self.intrinsic_calib[0,0],
                                pp=(self.intrinsic_calib[0, 2], self.intrinsic_calib[1, 2]),
                                )
        points, point_features = self.triangulate_points(R, t, matches, features)
        self.mapping_points.append(points)
        self.mapping_features.append(point_features)

        transformation_matrix = self._from_transf(R, np.squeeze(t))
        return transformation_matrix

    def process_image(self, image: Image) -> None:
        if self.frame_count % self.config["skip_frame"] == 0:
            if isinstance(self.old_frame, int):
                self.frame_count += 1
                self.old_frame = image
                self.old_features = self.extract_features(image)
                return 0
                
            self.frame_count += 1
            new_features = self.extract_features(image)
            matches, sorted_features = self.match_features(self.old_features, new_features)
            if matches[0] is not None: 
                if len(matches[0]) > 20 and len(matches[1]) > 20:
                    transf = self.get_pose(matches, sorted_features)
                    self.current_pose = np.matmul(self.current_pose, np.linalg.inv(transf))
                    self.poses.append(np.concatenate((self.current_pose,
                                                    np.array([[0., 0., 0., 1.]])),
                                                    axis=0))
            self.old_frame = image
            self.old_features = new_features
        else:
            self.frame_count += 1
            return None
    

