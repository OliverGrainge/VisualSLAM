import cv2
import numpy as np
from PIL import Image
from typing import Tuple
from collections import deque
from point_features import SIFT
from typing import List
import bundle_adjustment
from scipy.optimize import least_squares
import time


def homgenize(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transformation = np.eye(4)
    translation = translation.squeeze()
    if len(translation) == 4:
        translation = translation[:3] / translation[3]
    transformation[:3, :3] = rotation.squeeze()
    transformation[:3, 3] = translation.squeeze()
    return transformation


class OdometryPoint:
    def __init__(
        self,
        image_left: Image.Image,
        image_right: Image.Image,
        left_projection: np.ndarray,
        right_projection: np.ndarray,
        Kl: np.ndarray,
        Kr: np.ndarray,
    ):
        self.image_left = image_left
        self.image_right = image_right
        self.left_projection = left_projection
        self.right_projection = right_projection

        self.Kl = Kl
        self.Kr = Kr
        self.feature_detector = SIFT()
        self.matcher = cv2.BFMatcher()
        self.featurepoints2d()
        self.triangulate()

    def featurepoints2d(self) -> None:
        (
            self.left_points2d_pose,
            self.left_points2d_desc,
        ) = self.feature_detector(self.image_left)

        (
            self.right_points2d_pose,
            self.right_points2d_desc,
        ) = self.feature_detector(self.image_right)

    def triangulate(self):
        matches = self.matcher.knnMatch(
            self.left_points2d_desc, self.right_points2d_desc, k=2
        )
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


def get_matches(matcher, desc1: np.ndarray, desc2: np.ndarray, top_N=False):
    matches = matcher.knnMatch(desc1, desc2, k=2)
    # matches = sorted(matches, key=dist_fn)
    good_matches = []
    ratio_threshold = 0.75  # Commonly used threshold; adjust based on your dataset
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
    matches = good_matches

    #if top_N is not None:
    #    dist_fn = lambda x: x.distance
    #    matches = sorted(matches, key=dist_fn)
    #    matches = matches[:top_N]
    return matches

def project_points(points_3D, camera_matrix, rvec, tvec):
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    # Transform 3D points to the camera coordinate system
    points_3D = points_3D.reshape(-1, 3)  # Ensure points_3D is Nx3
    points_cam = R @ points_3D.T + tvec.reshape(-1, 1)
    # Project points onto the image plane
    points_proj = camera_matrix @ points_cam
    # Apply perspective division
    points_proj = points_proj[:2] / points_proj[2]
    # Transpose to get the final 2D points in shape Nx2
    points_2D = points_proj.T
    return points_2D


def relative_transformation(
    matcher, K_l, points3d_pos, points3d_desc, points2d_pos, points2d_desc
):
    matches = get_matches(matcher, points3d_desc, points2d_desc)
    points3d_sorted = np.zeros((len(matches), 3), dtype=np.float32)
    points2d_sorted = np.zeros((len(matches), 2), dtype=np.float32)
    points_desc = np.zeros((len(matches), points3d_desc.shape[1]), dtype=np.float32)

    for i, match in enumerate(matches):
        points3d_sorted[i, :] = points3d_pos[match.queryIdx]
        points2d_sorted[i, :] = points2d_pos[match.trainIdx].pt
        points_desc[i, :] = points3d_desc[match.queryIdx]

    success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
        points3d_sorted,
        points2d_sorted,
        K_l,
        np.zeros(4),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise Exception("PnP algorithm failed")

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    transformation = homgenize(rotation_matrix, translation_vector)
    return transformation





class StereoOdometry:
    def __init__(
        self,
        left_projection: np.ndarray,
        right_projection: np.ndarray,
        Kl: np.ndarray,
        Kr: np.ndarray,
        initial_pose: np.ndarray,
        window_size: int = 2,
    ):
        self.left_projection = left_projection
        self.right_projection = right_projection
        self.left_to_right = self.right_projection - self.left_projection
        self.initial_pose = initial_pose
        self.window_size = window_size
        self.Kl = Kl
        self.Kr = Kr

        self.poses = [initial_pose]
        self.transformations = []
        self.cumulative_transfrom = None
        self.points = []

        self.matcher = cv2.BFMatcher()

    def process_images(self, image_left: Image.Image, image_right: Image.Image) -> None:
        if len(self.points) == 0:
            point = OdometryPoint(
                image_left,
                image_right,
                self.left_projection,
                self.right_projection,
                self.Kl,
                self.Kr,
            )
            self.points.append(point)
        else:
            point = OdometryPoint(
                image_left,
                image_right,
                self.left_projection,
                self.right_projection,
                self.Kl,
                self.Kr,
            )
            transformation = relative_transformation(
                self.matcher,
                self.Kl,
                self.points[-1].points3d_pos,
                self.points[-1].points3d_desc,
                point.left_points2d_pose,
                point.left_points2d_desc,
            )
            transformation = np.linalg.inv(transformation)
            self.transformations.append(transformation)
            if self.cumulative_transfrom is None:
                self.cumulative_transfrom = transformation
            else:
                self.cumulative_transfrom = self.cumulative_transfrom @ transformation
            # self.poses.append(self.poses[-1] @ transformation)
            new_pose = self.initial_pose @ self.cumulative_transfrom
            self.left_projection = self.Kl @ self.cumulative_transfrom[:3, :]
            self.right_projection = self.Kr @ (self.cumulative_transfrom[:3, :] + self.left_to_right[:3, :])
        
            #=====================================================================================================
            self.left_projection = np.hstack([transformation[:3, :3], transformation[:3, 3].reshape(-1, 1)])
            self.left_projection = self.Kl @ self.left_projection
            self.right_projection = self.Kr @ self.right_projection 
            self.right_projection = self.left_projection + self.left_to_right
            """
                # Convert rotation vector to rotation matrix
            R = transformation[:3, :3]
            # Transform 3D points to the camera coordinate system
            points_3D = points_3D.reshape(-1, 3)  # Ensure points_3D is Nx3
            points_cam = R @ points_3D.T + tvec.reshape(-1, 1)
            # Project points onto the image plane
            points_proj = camera_matrix @ points_cam
            # Apply perspective division
            points_proj = points_proj[:2] / points_proj[2]
            # Transpose to get the final 2D points in shape Nx2
            points_2D = points_proj.T
            return points_2D
            """
            #=====================================================================================================
            point = OdometryPoint(
                image_left,
                image_right,
                self.left_projection,
                self.right_projection,
                self.Kl,
                self.Kr,
            )
            self.poses.append(new_pose)
            self.points.append(point)
            # self.bundle_adjust()

    def bundle_adjust(self) -> None:
        # 2d array of the 3d point positions
        max_3d_points = 100
        points3d_pos = self.odometry_points[-1].points3d_pos
        # 2d array of the 3d point descriptors
        points3d_desc = self.odometry_points[-1].points3d_desc
        points3d_desc = points3d_desc[:max_3d_points]
        # 3x3 matrix of the camera intrinsic parameters
        camera_model, _ = self.odometry_points[-1].intrinsic_camera_cal()
        # A list of the left camera project matrices
        camera_poses = self.poses[-self.window_size :]
        # number of camera images in the bundle
        n_cameras = len(camera_poses)
        # a list of the 2d observation descriptors
        points2d_desc = [
            pt.left_points2d_desc for pt in self.odometry_points[-self.window_size :]
        ]
        # a list of the 2d observation position arrays
        points2d_pos = []
        for op in self.odometry_points[-self.window_size :]:
            point_pos = []
            for kp in op.left_points2d_pose:
                point_pos.append([kp.pt[0], kp.pt[1]])
            point_pos = np.array(point_pos)
            points2d_pos.append(point_pos)

        # ============== Setup the parameters for optimization ====================
        # params[pitch_1, roll_1, yaw_1, tx_1, ty_1, tz_1, pitch_2, roll_2, ...... x1, y1, yz ]
        params = bundle_adjustment.points2params(camera_poses, points3d_pos)
        # camera_indices indicates which image each 2d observaion corresponds to
        # camera_indices[0, 0, 0, 0, 1, 1, 1, 1, .... n_cameras, n_cameras]
        # points_indices indicates which 3d point the descriptor coressponds to
        # point_indices[1, 4, 32, 14, 58, ... ]
        (
            points2d_pos,
            camera_indices,
            point_indices,
        ) = bundle_adjustment.data_association(
            self.odometry_points[-1].matcher, points2d_pos, points2d_desc, points3d_desc
        )

        error = bundle_adjustment.reprojection_error(
            params, n_cameras, camera_indices, point_indices, points2d_pos, camera_model
        )

        st = time.time()
        result = least_squares(
            bundle_adjustment.reprojection_error,
            params,
            args=(n_cameras, camera_indices, point_indices, points2d_pos, camera_model),
            verbose=0,
            method="lm",
            ftol=1e-06,
            xtol=1e-06,
            gtol=1e-06,
            max_nfev=200,
        )
        # **options)
        optimized_params = result.x

        optimized_error = bundle_adjustment.reprojection_error(
            optimized_params,
            n_cameras,
            camera_indices,
            point_indices,
            points2d_pos,
            camera_model,
        )
        poses, points3d_pos = bundle_adjustment.params2points(
            optimized_params, n_cameras
        )
        self.poses[-self.window_size :] = poses
        # self.odometry_points[-1].points3d_pos = points3d_pos
        et = time.time()
        print("=========")
        print(
            "optimized Adjustment Error:",
            np.abs(optimized_error).mean(),
            np.abs(error).mean(),
            st - et,
        )

    def get_trajectory(self):
        return self.poses

    def get_map(self):
        return np.vstack([pt.points3d_pos for pt in self.odometry_points])
