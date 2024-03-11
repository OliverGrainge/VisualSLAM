import cv2
import numpy as np
from PIL import Image
from typing import Tuple


class Point:
    def __init__(
        self,
        image_left: Image.Image,
        image_right: Image.Image,
        left_projection_matrix: np.ndarray,
    ):
        self.image_left = image_left
        self.image_right = image_right
        self.left_projection_matrix = left_projection_matrix

        self.feature_detector = cv2.ORB_create()

        self.left_points2d_pose = None
        self.right_points2d_pos = None
        self.left_points2d_desc = None
        self.right_points2d_desc = None
        self.points3d = None
        self.point_descriptors = None

        self.featurepoints2d()

    @staticmethod 
    def homgenize(rotation: np.ndarry, translation: np.ndarray) -> np.ndarray:
        transformation = np.eye(4)
        transformation[:3, :3] = rotation.squeeze()
        transformation[:3, 3] = translation 
        return transformation 

    def featurepoints2d(self) -> None:
        image = np.array(self.image_left)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        (
            self.left_points2d_pose,
            self.left_points2d_desc,
        ) = self.feature_detector.detectAndCompute(gray, None)

        image = np.array(self.image_right)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        (
            self.right_points2d_pose,
            self.right_points2d_desc,
        ) = self.feature_detector.detectAndCompute(gray, None)

    def projection_matrix(self) -> np.ndarray:
        return self.left_projection_matrix

    def rotation(
        self,
    ) -> np.ndarray:
        K, R, t, _, _, _, _ = cv2.decomposeProjectionMatrix(self.left_projection_matrix)
        return R

    def translation(
        self,
    ) -> np.ndarray:
        K, R, t, _, _, _, _ = cv2.decomposeProjectionMatrix(self.left_projection_matrix)
        return t


class PosePoint(Point):
    def __init__(
        self,
        image_left: Image.Image,
        image_right: Image.Image,
        left_projection_matrix: np.ndarray,
        right_projection_matrix: np.ndarray,
    ):
        super().__init__(
            image_left=image_left,
            image_right=image_right,
            left_projection_matrix=left_projection_matrix,
        )
        self.right_projection_matrix = right_projection_matrix
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        self.points3d_pos = None
        self.points2d_desc = None

        self.triangulate()

    def triangulate(self):
        self.featurepoints2d()
        matches = self.matcher.match(self.left_points2d_desc, self.right_points2d_desc)
        dist_fn = lambda x: x.distance
        matches = sorted(matches, key=dist_fn)

        points_left = np.zeros((len(matches), 2), dtype=np.float32)
        points_right = np.zeros((len(matches), 2), dtype=np.float32)
        points_desc = np.zeros((len(matches), self.left_points2d_desc.shape[1]), dtype=np.float32)
        for i, match in enumerate(matches):
            points_left[i, :] = self.left_points2d_pose[match.queryIdx].pt
            points_right[i, :] = self.right_points2d_pose[match.trainIdx].pt
            points_desc[i, :] = self.left_points_desc[match.queryIdx]

        points_left_transposed = points_left.T
        points_right_transposed = points_right.T

        point_4d_hom = cv2.triangulatePoints(
            self.left_projection_matrix,
            self.right_projection_matrix,
            points_left_transposed,
            points_right_transposed,
        )
        points_3d = point_4d_hom[:3] / point_4d_hom[3]
        self.points3d_pos = points_3d.T
        self.points3d_desc = points_desc

    def projection_matrix(self) -> Tuple[np.ndarray]:
        return (self.left_projection_matrix, self.right_projection_matrix)

    def rotation(
        self,
    ) -> Tuple[np.ndarray]:
        Kl, Rl, tl, _, _, _, _ = cv2.decomposeProjectionMatrix(
            self.left_projection_matrix
        )
        Kr, Rr, tr, _, _, _, _ = cv2.decomposeProjectionMatrix(
            self.right_projection_matrix
        )
        return Rl, Rr

    def translation(
        self,
    ) -> Tuple[np.ndarray]:
        Kl, Rl, tl, _, _, _, _ = cv2.decomposeProjectionMatrix(
            self.left_projection_matrix
        )
        Kr, Rr, tr, _, _, _, _ = cv2.decomposeProjectionMatrix(
            self.left_projection_matrix
        )
        return tl, tr

    def intrinsic_camera_cal(self) -> Tuple[np.ndarray]:
        Kl, Rl, tl, _, _, _, _ = cv2.decomposeProjectionMatrix(
            self.left_projection_matrix
        )
        Kr, Rr, tr, _, _, _, _ = cv2.decomposeProjectionMatrix(
            self.left_projection_matrix
        )
        return Kl, Kr





class OdometryPoint(PosePoint): 
    def __init__(
        self,
        image_left: Image.Image,
        image_right: Image.Image,
        left_projection_matrix: np.ndarray,
        right_projection_matrix: np.ndarray,
    ):
        super().__init__(
            image_left=image_left,
            image_right=image_right,
            left_projection_matrix=left_projection_matrix,
            right_projection_matrix=right_projection_matrix
        )


    def relative_transformation(self, points3d: np.ndarray, desc3d: np.ndarray):
        matches = self.matcher.match(desc3d, self.left_points2d_desc)
        dist_fn = lambda x: x.distance
        matches = sorted(matches, key=dist_fn)

        points3d_sorted = np.zeros((len(matches), 3), dtype=np.float32)
        points2d_sorted = np.zeros((len(matches), 2), dtype=np.float32)
        points_desc = np.zeros((len(matches), desc3d.shape[1]), dtype=np.float32)

        for i, match in enumerate(matches):
            points3d_sorted[i, :] = points3d[match.queryIdx].pt
            points2d_sorted[i, :] = self.left_points2d_pose[match.trainIdx].pt
            points_desc[i, :] = desc3d[match.queryIdx]

        K_l, K_r = self.intrinsic_camera_cal()
        dist_coeffs = np.zeros(4)
        success, rotation_vector, translation_vector = cv2.solvePnP(points3d_sorted, points2d_sorted, K_l, dist_coeffs)
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        transformation = self.homgenize(rotation_matrix, translation_vector)
        return transformation

