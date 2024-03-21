import time
from collections import deque
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from sklearn.metrics.pairwise import cosine_similarity
from loop_detector import LoopDetector

from point_features import SIFT
from pose_graph_optimization import optimize_poses
from utils import (
    decompose_projection,
    get_config,
    get_feature_detector,
    get_feature_matcher,
    get_matches,
    homogenize,
    projection_matrix,
    relative_transformation,
    sort_matches,
    transform_points3d,
    unhomogenize,
)

np.set_printoptions(precision=3, suppress=True)

config = get_config()


class StereoPoint:
    """
    A class designed to handle stereo image processing to extract 2D features, match them across stereo pairs,
    and triangulate to find 3D points.

    Attributes:
        image_left (Image.Image): The left image of the stereo pair.
        image_right (Image.Image): The right image of the stereo pair.
        left_projection (Image.Image): The projection matrix for the left camera.
        right_projection (Image.Image): The projection matrix for the right camera.
        K_l (np.ndarray): The intrinsic camera matrix for the left camera.
        K_r (np.ndarray): The intrinsic camera matrix for the right camera.
        left_kp: Key points from the left image. Initialized as None and set by `featurepoints2d`.
        right_kp: Key points from the right image. Initialized as None and set by `featurepoints2d`.
        left_desc2d: Descriptors for key points in the left image. Initialized as None and set by `featurepoints2d`.
        right_desc2d: Descriptors for key points in the right image. Initialized as None and set by `featurepoints2d`.
        points3d: 3D points triangulated from matched key points. Initialized as None and set by `triangulate`.
        desc3d: Descriptors associated with 3D points. Initialized as None and set by `triangulate`.
        matcher: The feature matcher used to match descriptors across stereo images.
        feature_detector: The feature detector used to detect key points and descriptors in images.
        stereo: The stereo block matcher used for generating disparity maps.

    Parameters:
        image_left (Image.Image): The left image of the stereo pair.
        image_right (Image.Image): The right image of the stereo pair.
        left_projection (Image.Image): The projection matrix for the left camera.
        right_projection (Image.Image): The projection matrix for the right camera.
        K_l (np.ndarray): The intrinsic camera matrix for the left camera.
        K_r (np.ndarray): The intrinsic camera matrix for the right camera.
    """
    def __init__(
        self,
        image_left: Image.Image,
        image_right: Image.Image,
        left_projection: Image.Image,
        right_projection: Image.Image,
        K_l: np.ndarray,
        K_r: np.ndarray,
    ):
        """
        Initializes the StereoPoint object with the given images, projection matrices, and intrinsic camera matrices.
        """

        self.image_left = image_left
        self.image_right = image_right
        self.left_projection = left_projection
        self.right_projection = right_projection
        self.K_l = K_l
        self.K_r = K_r

        self.left_kp = None
        self.right_kp = None
        self.left_desc2d = None
        self.right_desc2d = None
        self.points3d = None
        self.desc3d = None

        self.matcher = get_feature_matcher()
        self.feature_detector = get_feature_detector()
        self.stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=15)

    def featurepoints2d(self) -> None:
        """
        Detects and extracts 2D feature points and their descriptors from both left and right images.
        """
        (
            self.left_kp,
            self.left_desc2d,
        ) = self.feature_detector(self.image_left)

        (
            self.right_kp,
            self.right_desc2d,
        ) = self.feature_detector(self.image_right)

    def triangulate(self) -> None:
        """
        Performs stereo triangulation on matched feature points to compute 3D points.
        This method uses the matched 2D feature points from both images, the camera projection matrices,
        and the intrinsic matrices to triangulate and generate 3D points.
        """
        matches = get_matches(self.matcher, self.left_desc2d, self.right_desc2d)
        points_left, points_right, points_desc = sort_matches(
            matches, self.left_kp, self.right_kp, self.left_desc2d
        )

        points_left_transposed = points_left.T
        points_right_transposed = points_right.T
        point_4d_hom = cv2.triangulatePoints(
            self.left_projection,
            self.right_projection,
            points_left_transposed,
            points_right_transposed,
        )
        point_4d_hom = point_4d_hom.T
        self.points3d = point_4d_hom[:, :3] / point_4d_hom[:, 3].reshape(-1, 1)
        self.desc3d = points_desc.astype(self.left_desc2d.dtype)
        """
        matched_image = cv2.drawMatches(
            np.array(self.image_left),
            self.left_kp,
            np.array(self.image_right),
            self.right_kp,
            matches[:40],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("Matched Points", matched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """


class StereoOdometry:
    """
    A class for computing stereo odometry, integrating features such as pose graph optimization (PGO) and loop closure detection.
    It processes stereo images to estimate the camera's motion and reconstructs a 3D trajectory over time.

    Attributes:
        left_projection (np.ndarray): The projection matrix for the left camera.
        right_projection (np.ndarray): The projection matrix for the right camera.
        initial_pose (np.ndarray): The initial pose of the camera.
        pgo (bool): Indicates whether pose graph optimization is enabled.
        window (int): The size of the window for windowed pose graph optimization.
        loop_closure (bool): Indicates whether loop closure detection is enabled.
        K_l, K_r (np.ndarray): The intrinsic camera matrices for the left and right cameras, respectively.
        R_l, R_r (np.ndarray): The rotation matrices for the left and right cameras, respectively.
        T_l, T_r (np.ndarray): The translation vectors for the left and right cameras, respectively.
        left_to_right (np.ndarray): The translation from the left to the right camera.
        loop_detector (LoopDetector): The loop detector instance used if loop closure is enabled.
        poses (list): A list of poses (4x4 transformation matrices) estimated so far.
        cumulative_transform_inv (np.ndarray): The inverse of the cumulative transformation matrix.
        cumulative_transform (np.ndarray): The cumulative transformation matrix.
        all_cumulative_transforms (list): A list of all cumulative transformation matrices.
        points (list): A list of `StereoPoint` objects representing stereo image pairs processed.
        transforms (list): A list of transformation matrices between consecutive stereo image pairs.
        points3d (list): A list of 3D points triangulated from stereo image pairs.
        desc3d (list): A list of descriptors associated with the 3D points.
        loop_closures (list): A list of loop closures detected.

    Parameters:
        left_projection (np.ndarray): The projection matrix for the left camera.
        right_projection (np.ndarray): The projection matrix for the right camera.
        initial_pose (np.ndarray): The initial pose of the camera.
        pgo (bool): Whether pose graph optimization is enabled.
        window (int): The size of the window for windowed pose graph optimization.
        loop_closure (bool): Whether loop closure detection is enabled.
    """
    def __init__(
        self,
        left_projection: np.ndarray,
        right_projection: np.ndarray,
        initial_pose: np.ndarray,
        pgo: bool = config["pose_graph_optimization"],
        window: int = config["window_size"],
        loop_closure: bool = config["loop_closure"],
    ):
        """
        Initializes the StereoOdometry object with the given parameters.
        """
        self.left_projection = left_projection
        self.right_projection = right_projection
        self.initial_projection_left = left_projection
        self.initial_projection_right = right_projection
        self.initial_pose = initial_pose
        self.window = window
        self.pgo = pgo
        self.loop_closure = loop_closure
        self.K_l, self.R_l, self.T_l = decompose_projection(self.left_projection)
        self.K_r, self.R_r, self.T_r = decompose_projection(self.right_projection)
        self.left_to_right = self.T_r - self.T_l

        if self.loop_closure:
            self.loop_detector = LoopDetector()

        self.poses = [initial_pose]
        self.cumulative_transform_inv = None
        self.cumulative_transform = None
        self.all_cumulative_transforms = []

        self.points = []
        self.transforms = []
        self.points3d = []
        self.desc3d = []
        self.loop_closures = []

    def process_images(self, image_left: Image.Image, image_right: Image.Image) -> None:
        """
        Processes a pair of stereo images to estimate the camera's motion and update the 3D map.

        Parameters:
            image_left (Image.Image): The left image of the stereo pair.
            image_right (Image.Image): The right image of the stereo pair.
        """
        if len(self.points) == 0:
            point = StereoPoint(
                image_left,
                image_right,
                self.initial_projection_left,
                self.initial_projection_right,
                self.K_l,
                self.K_r,
            )
            point.featurepoints2d()
            point.triangulate()

            self.points.append(point)
            self.points3d.append(point.points3d)
            self.desc3d.append(point.desc3d)
        else:
            new_point = StereoPoint(
                image_left,
                image_right,
                self.initial_projection_left,
                self.initial_projection_right,
                self.K_l,
                self.K_r,
            )
            new_point.featurepoints2d()
            new_point.triangulate()
            prev_point = self.points[-1]
            rvec, tvec = relative_transformation(
                prev_point.matcher,
                prev_point.points3d,
                prev_point.desc3d,
                new_point.left_kp,
                new_point.left_desc2d,
                new_point.K_l,
            )

            if self.cumulative_transform is None:
                self.cumulative_transform = self.initial_pose @ homogenize(rvec, tvec)
                self.all_cumulative_transforms.append(self.cumulative_transform)
            else:
                self.cumulative_transform = self.cumulative_transform @ homogenize(
                    rvec, tvec
                )
                self.all_cumulative_transforms.append(self.cumulative_transform)

            self.transforms.append(np.linalg.inv(homogenize(rvec, tvec)))
            new_pose = self.poses[-1] @ np.linalg.inv(homogenize(rvec, tvec))
            self.points3d.append(
                transform_points3d(np.linalg.inv(new_pose), new_point.points3d)
            )
            self.desc3d.append(new_point.desc3d)
            self.poses.append(new_pose)
            self.points.append(new_point)
            if self.pgo:
                self.windowed_pose_graph_optimization()
            if self.loop_closure:
                sucess, match_idx, match_dist = self.loop_detector(image_left)
                if sucess:
                    print(sucess)
                    rvec, tvec = relative_transformation(
                        self.points[-1].matcher,
                        self.points[-1].points3d,
                        self.points[-1].desc3d,
                        self.points[-1].left_kp,
                        self. points[match_idx].left_desc2d,
                        self.K_l,
                    )
                    T = np.linalg.inv(homogenize(rvec, tvec))
                    self.loop_closures.append((len(self.points), match_idx, T))
                    self.global_pose_graph_optimization()
                """
                if sucess:
                    import matplotlib.pyplot as plt 
                    fig, axs = plt.subplots(1, 2)
                    axs[0].imshow(np.array(image_left), cmap='gray')
                    axs[1].imshow(np.array(self.points[match_idx].image_left), cmap='gray')
                    plt.show()
                """

            # ======================================

    def get_trajectory(self):
        """
        Returns the estimated trajectory of the camera as a list of poses.

        Returns:
            list: A list of poses (4x4 transformation matrices) estimated so far.
        """
        return self.poses

    def get_map(self):
        """
        Returns the reconstructed 3D map as a list of 3D points.

        Returns:
            list: A list of 3D points reconstructed from the stereo image pairs.
        """
        return [pt.points3d for pt in self.points]

    def windowed_pose_graph_optimization(self):
        """
        Performs windowed pose graph optimization to refine the poses within a specified window.
        """
        points = self.points[-self.window :]
        poses = self.poses[-self.window :]
        pose_graph = [[None for _ in range(len(points))] for _ in range(len(points))]
        i = 0
        for j in range(len(points)):
            if i != j:
                rvec, tvec = relative_transformation(
                    points[i].matcher,
                    points[i].points3d,
                    points[i].desc3d,
                    points[j].left_kp,
                    points[j].left_desc2d,
                    self.K_l,
                )
                # pose_j ~ pose_i @ pose_graph[i][j]
                pose_graph[i][j] = np.linalg.inv(homogenize(rvec, tvec))
        opt_poses = optimize_poses(poses, pose_graph)
        self.poses[-self.window :] = opt_poses


    def global_pose_graph_optimization(self):
        """
        Performs global pose graph optimization to refine all the estimated poses based on loop closures and sequential transformations.
        """ 
        print("======================== PERFORMING GLOBAL OPTIMIZATION ===========================")
        # pose_j ~ poise_i @ pose_graph[i][j]
        pose_graph = [[None for _ in range(len(self.points))] for _ in range(len(self.points))]
        # fill the graph with sequential transformations 
        for idx, T in enumerate(self.transforms):
            pose_graph[idx + 1][idx] = T
        # fill the graph with loop closures
        print("========================= ", len(self.loop_closures), "  Loop Closures ==============")
        for lc in self.loop_closures: 
            pose_graph[lc[0]][lc[1]] = lc[2]

        opt_poses = optimize_poses(self.poses, pose_graph)
        self.poses = opt_poses




