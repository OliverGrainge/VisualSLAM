from typing import List, Union

import numpy as np
from PIL import Image
from utils import get_feature_detector, get_feature_matcher
import cv2 
from utils import projection_matrix, homogenize, unhomogenize


class StereoPoint:
    """
    A class to manage and process stereo image data to extract and utilize 2D and 3D features.

    Attributes:
        image (Image.Image): Left image of the stereo camera pair as a PIL image.
        right_image (Image.Image): Right image of the stereo camera pair as a PIL image.
        Kl (np.ndarray): Intrinsic calibration matrix of the left camera.
        Kr (np.ndarray): Intrinsic calibration matrix of the right camera.
        T (np.ndarray): Transformation matrix representing the pose changes from the left to right camera.
        X (np.ndarray, optional): The pose of the left camera in homogeneous coordinates. Defaults to None.
        keypoints_2d (np.ndarray): Detected 2D keypoints in the left image. None until computed.
        descriptors_2d (np.ndarray): Descriptors for the 2D keypoints in the left image. None until computed.
        global_descriptor_2d (np.ndarray): A global descriptor for the left image, if applicable. None by default.
        keypoints_3d (np.ndarray): 3D points computed from stereo image pair. None until computed.
        descriptors_3d (np.ndarray): Descriptors associated with the 3D keypoints. None until computed.
        feature_detector: Callable function to detect features in an image.
        feature_matcher: Callable function to match features between two images.
    
    Methods:
        compute_features2d(): Detects 2D keypoints and descriptors in the left image.
        compute_features3d(): Triangulates 3D points from matched 2D keypoints in stereo images.
    """
    def __init__(
        self,
        image: Image.Image,
        right_image: Image.Image,
        T: np.ndarray,
        Kl: np.ndarray,
        Kr: np.ndarray,
        X: Union[None, np.ndarray] = None,
    ):
        """
        Initializes the StereoPoint object with the given images, transformation matrix, and camera intrinsic parameters.

        Parameters:
            image (Image.Image): The left image from the stereo camera setup.
            right_image (Image.Image): The right image from the stereo camera setup.
            T (np.ndarray): A 4x4 transformation matrix from the left camera to the right camera.
            Kl (np.ndarray): A 3x3 intrinsic camera matrix for the left camera.
            Kr (np.ndarray): A 3x3 intrinsic camera matrix for the right camera.
            X (np.ndarray, optional): The pose of the left camera in homogeneous coordinates.
        """
        # imaging data and calibrations
        self.image = image  # left PIL image of stereo camera
        self.right_image = right_image  # right PIL image of stereo camera
        self.Kl = Kl  # intrinsic parameters of left camera
        self.Kr = Kr  # intrinsc parameters of right camera
        self.T = T  # trasformation between left and right image image

        # pose of self.image
        self.X = X  # pose of camera in homogenous co-ordinates (4x4 matrix)

        # 2d features
        self.keypoints_2d = None
        self.descriptors_2d = None
        self.global_descriptor_2d = None

        # 3d features
        self.keypoints_3d = None
        self.descriptors_3d = None

        # feature extraction
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
        Computes the 2D keypoints and descriptors in the left image using the configured feature detector.
        This method populates the keypoints_2d and descriptors_2d attributes.
        """
        if self.keypoints_2d is None or self.descriptors_2d is None:
            kp, des = self.feature_detector(self.image)
            self.keypoints_2d = kp
            self.descriptors_2d = des

    def compute_features3d(self):
        """
        Triangulates 3D points from the 2D feature correspondences between the left and right stereo images.
        This method uses the configured feature detector and matcher to first find matches and then triangulates
        these to compute 3D coordinates. It also filters out 3D points that are beyond a threshold distance,
        which is a multiple of the stereo baseline. This helps in removing distant and potentially unreliable points.
        """
        if self.keyspoints_2d is None:
            kp, des = self.feature_detector(self.image)
            self.keypoints_2d = kp
            self.descriptors_2d = des

        right_keypoints_2d, right_descriptors_2d = self.feature_detector(
            self.right_image
        )
        left_kp, left_des, right_kp, _ = self.feature_matcher(
            kp1=self.keypoints_2d,
            des1=self.descriptors_2d,
            kp2=right_keypoints_2d,
            des2=right_descriptors_2d,
        )

        rvec, tvec = unhomogenize(self.T)
        Pl = projection_matrix(rvec=np.zeros(3), tvec=np.zeros(3), K=self.Kr)
        Pr = projection_matrix(rvec=rvec, tvec=tvec, K=self.Kl)
        points_3d_hom = cv2.triangulatePoints(Pl, Pr, left_kp, right_kp)
        points_3d = points_3d_hom[:3] / points_3d_hom[3]

        baseline = np.linalg.norm(tvec)
        max_depth = baseline * 9
        filtered_mask = np.array([True if point[2] < max_depth else False for point in points_3d])
        keypoints_3d = points_3d.T
        self.keypoints_3d = self.keypoints_3d[filtered_mask]
        self.descriptors_3d = left_des[filtered_mask]
