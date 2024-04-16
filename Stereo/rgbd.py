from typing import List

import numpy as np
from PIL import Image
from utils import get_feature_detector, get_feature_matcher


class RgbdPoint:
    """
    A class representing a point in an RGB-D image with methods to compute 2D and 3D features.

    Attributes:
        image (Image.Image): The RGB image associated with this point.
        depth_map (Image.Image): The depth map image providing depth information for each pixel in the RGB image.
        K (np.ndarray): The intrinsic camera calibration matrix containing focal lengths and optical center coordinates.
        keypoints_2d (list or None): List of 2D keypoints detected in the RGB image, None if not yet computed.
        descriptors_2d (np.ndarray or None): Array of descriptors for the 2D keypoints, None if not yet computed.
        global_descriptor_2d (Any or None): A global descriptor for the RGB image, None if not yet computed.
        keypoints_3d (np.ndarray or None): Array of 3D points corresponding to the 2D keypoints, None if not computed.
        descriptors_3d (np.ndarray or None): Array of descriptors corresponding to the 3D keypoints, None if not computed.
        feature_detector (Callable): A function to detect features and compute descriptors in the RGB image.
        feature_matcher (Callable): A function to match features between different sets of descriptors.

    Methods:
        compute_features2d(): Detects 2D keypoints and computes their descriptors in the RGB image.
        compute_features3d(): Computes 3D keypoints from 2D keypoints using the depth map and the intrinsic matrix.
    """
    def __init__(self, image: Image.Image, depth_map: Image.Image, K: np.ndarray):
        """
        Initializes an RgbdPoint instance with an RGB image, a depth map, and camera intrinsic parameters.

        Parameters:
            image (Image.Image): The RGB image for feature detection.
            depth_map (Image.Image): The depth map image providing depth data correlated to the RGB image.
            K (np.ndarray): The intrinsic camera calibration matrix.
        """
        # image data
        self.image = image
        self.depth_map = depth_map

        # camera parameters 
        self.K = K # intrinsic calibration parameters
        self.X = # extrinsic transformation in world co-ordinates

        # 2d features
        self.keypoints_2d = None
        self.descriptors_2d = None
        self.global_descriptor_2d = None

        # 2d features
        self.keypoints_3d = None
        self.descriptors_3d = None

        # feature computation
        self.feature_detector = get_feature_detector()
        self.feature_matcher = get_feature_matcher()

    def transform_points3d(self):
        """
        Transforms the 3D keypoints based on the camera's pose. This method applies the pose transformation 
        matrix (self.X) to the 3D keypoints (self.keypoints_3d) to convert them from the camera coordinate 
        system to the world coordinate system or vice versa, depending on the definition of self.X.

        The transformation is done in homogeneous coordinates to accommodate translation. This method 
        assumes that self.X is a valid 4x4 homogeneous transformation matrix and self.keypoints_3d are
        the points in 3D space needing transformation.

        Returns:
            None: The keypoints are transformed in-place, modifying the self.keypoints_3d attribute.

        Raises:
            AssertionError: If the points are not represented in homogeneous coordinates.
        """
        if self.X is None or self.keypoints_3d is None: 
            return 

        points = np.hstack((self.keypoints_3d, np.ones((len(self.keypoints_3d), 1))))
        assert points.shape[1] == 4, "points must be represented in homogenous co-ordinates"
        self.keypoints_3d = np.dot(self.X, self.keypoints_3d.T).T
        self.keypoints_3d = self.keypoints_3d[:, :3] / self.keypoints_3d[:, 3].reshape(-1, 1)

    def compute_features2d(self):
        """
        Detects keypoints and computes their descriptors in the RGB image using the assigned feature detector.
        Results are stored in the keypoints_2d and descriptors_2d attributes.
        """
        if self.keypoints_2d is None or self.descriptors_2d is None:
            kp, des = self.feature_detector(self.image)
            self.keypoints_2d = kp
            self.descriptors_2d = des

    def compute_features3d(self):
        """
        Computes 3D keypoints from the 2D keypoints using the depth map and camera intrinsic parameters.
        Corresponding descriptors are derived from the 2D descriptors. Ensures 2D features are computed first.
        """
        if not self.keypoints_2d or not self.descriptors_2d:
            self.compute_features2d()

        # Convert depth map to numpy array for indexing
        depth_array = np.array(self.depth_map)

        # Prepare to collect 3D points
        keypoints_3d = []
        mask = []
        for kp in self.keypoints_2d:
            x, y = int(kp.pt[0]), int(kp.pt[1])  # Keypoint coordinates rounded to nearest integer
            depth = depth_array[y, x]  # Get the depth value at this pixel

            if depth > 0:  # Ignore keypoints with no depth information
                # Convert (x, y, depth) to (X, Y, Z) in camera coordinates
                mask.append(True)
                X = (x - self.K[0, 2]) * depth / self.K[0, 0]
                Y = (y - self.K[1, 2]) * depth / self.K[1, 1]
                Z = depth
                keypoints_3d.append([X, Y, Z])
            mask.append(False)

        self.keypoints_3d = np.array(keypoints_3d)  # Convert list to array for use in further processing
        self.descriptors_3d = self.descriptors_2d[np.array(mask)]

