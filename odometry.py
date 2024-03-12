import cv2
import numpy as np
from PIL import Image
from typing import Tuple
from collections import deque
from point_features import SIFT
from typing import List


class Point:
    def __init__(
        self,
        image_left: Image.Image,
        image_right: Image.Image,
        left_projection: np.ndarray,
    ):
        self.image_left = image_left
        self.image_right = image_right
        self.left_projection = left_projection

        self.feature_detector = SIFT()

        self.left_points2d_pose = None
        self.right_points2d_pose = None
        self.left_points2d_desc = None
        self.right_points2d_desc = None
        self.points3d = None
        self.point_descriptors = None

        self.featurepoints2d()

    @staticmethod
    def homgenize(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
        transformation = np.eye(4)
        transformation[:3, :3] = rotation.squeeze()
        transformation[:3, 3] = translation.squeeze()
        return transformation

    def featurepoints2d(self) -> None:
        (
            self.left_points2d_pose,
            self.left_points2d_desc,
        ) = self.feature_detector(self.image_left)

        (
            self.right_points2d_pose,
            self.right_points2d_desc,
        ) = self.feature_detector(self.image_right)

    def projection_matrix(self) -> np.ndarray:
        return self.left_projection

    def rotation(
        self,
    ) -> np.ndarray:
        K, R, t, _, _, _, _ = cv2.decomposeProjectionMatrix(self.left_projection)
        return R

    def translation(
        self,
    ) -> np.ndarray:
        K, R, t, _, _, _, _ = cv2.decomposeProjectionMatrix(self.left_projection)
        return t


class PosePoint(Point):
    def __init__(
        self,
        image_left: Image.Image,
        image_right: Image.Image,
        left_projection: np.ndarray,
        right_projection: np.ndarray,
    ):
        super().__init__(
            image_left=image_left,
            image_right=image_right,
            left_projection=left_projection,
        )
        self.right_projection = right_projection
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.points3d_pos = None
        self.points2d_desc = None

        self.triangulate()

    def triangulate(self):
        matches = self.matcher.knnMatch(
            self.left_points2d_desc, self.right_points2d_desc, k=2
        )
        # dist_fn = lambda x: x.distance
        # matches = sorted(matches, key=dist_fn)
        # Apply Lowe's ratio test
        good_matches = []
        ratio_threshold = 0.75  # Commonly used threshold; adjust based on your dataset
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
        matches = good_matches
        """
        matched_image = cv2.drawMatches(
            np.array(self.image_left),
            self.left_points2d_pose,
            np.array(self.image_right),
            self.right_points2d_pose,
            matches[:40],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("Matched Points", matched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
        points_left = np.zeros((len(matches), 2), dtype=self.left_points2d_desc.dtype)
        points_right = np.zeros((len(matches), 2), dtype=self.right_points2d_desc.dtype)
        points_desc = np.zeros(
            (len(matches), self.left_points2d_desc.shape[1]), dtype=np.float32
        )
        for i, match in enumerate(matches):
            points_left[i, :] = self.left_points2d_pose[match.queryIdx].pt
            points_right[i, :] = self.right_points2d_pose[match.trainIdx].pt
            points_desc[i, :] = self.left_points2d_desc[match.queryIdx]

        points_left_transposed = points_left.T
        points_right_transposed = points_right.T

        point_4d_hom = cv2.triangulatePoints(
            self.left_projection,
            self.right_projection,
            points_left_transposed,
            points_right_transposed,
        )

        points_3d = point_4d_hom[:3] / point_4d_hom[3]
        self.points3d_pos = points_3d.T
        self.points3d_desc = points_desc

    def projection_matrix(self) -> Tuple[np.ndarray]:
        return (self.left_projection, self.right_projection)

    def rotation(
        self,
    ) -> Tuple[np.ndarray]:
        Kl, Rl, tl, _, _, _, _ = cv2.decomposeProjectionMatrix(self.left_projection)
        Kr, Rr, tr, _, _, _, _ = cv2.decomposeProjectionMatrix(self.right_projection)
        return Rl, Rr

    def translation(
        self,
    ) -> Tuple[np.ndarray]:
        Kl, Rl, tl, _, _, _, _ = cv2.decomposeProjectionMatrix(self.left_projection)
        Kr, Rr, tr, _, _, _, _ = cv2.decomposeProjectionMatrix(self.left_projection)
        return tl, tr

    def intrinsic_camera_cal(self) -> Tuple[np.ndarray]:
        Kl, Rl, tl, _, _, _, _ = cv2.decomposeProjectionMatrix(self.left_projection)
        Kr, Rr, tr, _, _, _, _ = cv2.decomposeProjectionMatrix(self.left_projection)
        return Kl, Kr


class OdometryPoint(PosePoint):
    def __init__(
        self,
        image_left: Image.Image,
        image_right: Image.Image,
        left_projection: np.ndarray,
        right_projection: np.ndarray,
    ):
        super().__init__(
            image_left=image_left,
            image_right=image_right,
            left_projection=left_projection,
            right_projection=right_projection,
        )
        self.transformation = None

    def relative_transformation(self, points3d: np.ndarray, desc3d: np.ndarray):
        matches = self.matcher.knnMatch(desc3d, self.left_points2d_desc, k=2)
        dist_fn = lambda x: x.distance
        # matches = sorted(matches, key=dist_fn)
        good_matches = []
        ratio_threshold = 0.75  # Commonly used threshold; adjust based on your dataset
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
        matches = good_matches

        points3d_sorted = np.zeros((len(matches), 3), dtype=np.float32)
        points2d_sorted = np.zeros((len(matches), 2), dtype=np.float32)
        points_desc = np.zeros((len(matches), desc3d.shape[1]), dtype=np.float32)

        for i, match in enumerate(matches):
            points3d_sorted[i, :] = points3d[match.queryIdx]
            points2d_sorted[i, :] = self.left_points2d_pose[match.trainIdx].pt
            points_desc[i, :] = desc3d[match.queryIdx]

        K_l, K_r = self.intrinsic_camera_cal()
        dist_coeffs = np.zeros(4)

        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
            points3d_sorted,
            points2d_sorted,
            K_l,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            raise Exception("PnP algorithm failed")
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        transformation = self.homgenize(rotation_matrix, translation_vector)
        self.transformation = transformation
        return transformation


class BundleAdjustment:
    def __init__(
        self,
    ):
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    @staticmethod
    def reprojection_error(
        params,
        n_cameras,
        n_points,
        camera_indices,
        point_indices,
        observed_points,
        camera_model,
    ):
        """Calculate reprojection errors."""
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))
        points_projected = np.zeros(observed_points.shape)

        for i in range(len(camera_indices)):
            camera_index = camera_indices[i]
            point_index = point_indices[i]
            R, _ = cv2.Rodrigues(camera_params[camera_index, :3])
            t = camera_params[camera_index, 3:6].reshape(3, 1)
            point_3d_hom = np.hstack([points_3d[point_index], [1]])
            point_2d_hom = camera_model @ (R @ point_3d_hom[:3] + t)
            points_projected[i] = point_2d_hom[:2] / point_2d_hom[2]

        return (points_projected - observed_points).ravel()

    def data_association(self, points2d_desc: List[np.ndarray], points3d_desc: np.ndarray) -> Tuple[np.ndarray]:
        point_indices = []
        camera_indices = []
        for camera_idx, desc2d in enumerate(points2d_desc):
            matches = self.matcher.knnMatch(desc2d, points3d_desc, k=2)
            good_matches = []
            ratio_threshold = 0.75  # Commonly used threshold; adjust based on your dataset
            for m, n in matches:
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
            point_indices += [match.trainIdx for match in good_matches]
            camera_indices += [camera_idx for _ in good_matches]
        return camera_indices, point_indices

    @staticmethod
    def poses2params(poses: List[np.ndarray]) -> np.ndarray:
        params = []
        for pose in poses:
            translation = list(pose[:3, 3])
            rotation, _ = cv2.Rodrigues(pose[:3, :3])
            rotation = list(rotation)
            params += rotation
            params += translation
        return np.array(params)


class StereoOdometry:
    def __init__(
        self,
        left_projection: np.ndarray,
        right_projection: np.ndarray,
        initial_pose: np.ndarray,
        window_size: int = 10,
    ):
        self.left_projection = left_projection
        self.right_projection = right_projection
        self.initial_pose = initial_pose
        self.window_size = window_size

        self.poses = [initial_pose]
        self.transformations = []
        self.odometry_points = []
        self.point_cloud_pos = []
        self.point_could_desc = []

    def process_images(self, image_left: Image.Image, image_right: Image.Image):
        odom_point = OdometryPoint(
            image_left, image_right, self.left_projection, self.right_projection
        )
        if len(self.odometry_points) == 0:
            self.odometry_points.append(odom_point)
            return
        else:
            transformation = odom_point.relative_transformation(
                self.odometry_points[-1].points3d_pos,
                self.odometry_points[-1].points3d_desc,
            )
            transformation = np.linalg.inv(transformation)
            self.transformations.append(transformation)
            self.poses.append(self.poses[-1] @ transformation)
            self.odometry_points.append(odom_point)

    def get_trajectory(self):
        return self.poses

    def get_map(self):
        return np.vstack([pt.points3d_pos for pt in self.odometry_points])
