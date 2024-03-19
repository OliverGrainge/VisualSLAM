import cv2
import numpy as np
from PIL import Image
from typing import Tuple
from collections import deque
from point_features import SIFT
from typing import List
from scipy.optimize import least_squares
import time
from sklearn.metrics.pairwise import cosine_similarity
from utils import get_matches, sort_matches, homogenize, unhomogenize, transform_points3d, decompose_projection, projection_matrix, relative_transformation
from pose_graph_optimization import optimize_poses
np.set_printoptions(precision=3, suppress=True)



class StereoPoint:
    def __init__(
        self,
        image_left: Image.Image,
        image_right: Image.Image,
        left_projection: Image.Image,
        right_projection: Image.Image,
        K_l: np.ndarray,
        K_r: np.ndarray,
    ):
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

        self.matcher = cv2.BFMatcher()
        self.feature_detector = SIFT()
        self.stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=15)

    def featurepoints2d(self) -> None:
        (
            self.left_kp,
            self.left_desc2d,
        ) = self.feature_detector(self.image_left)

        (
            self.right_kp,
            self.right_desc2d,
        ) = self.feature_detector(self.image_right)

    def triangulate(self) -> None:
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
        self.desc3d = points_desc
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
    def __init__(
        self,
        left_projection: np.ndarray,
        right_projection: np.ndarray,
        initial_pose: np.ndarray,
        window: int=10,
    ):
        self.left_projection = left_projection
        self.right_projection = right_projection
        self.initial_projection_left = left_projection
        self.initial_projection_right = right_projection
        self.initial_pose = initial_pose
        self.window = window
        self.K_l, self.R_l, self.T_l = decompose_projection(self.left_projection)
        self.K_r, self.R_r, self.T_r = decompose_projection(self.right_projection)
        self.left_to_right = self.T_r - self.T_l

        self.poses = [initial_pose]
        self.cumulative_transform_inv = None
        self.cumulative_transform = None
        self.all_cumulative_transforms = []

        self.points = []
        self.transforms = []
        self.points3d = []
        self.desc3d = []

    def process_images(self, image_left: Image.Image, image_right: Image.Image) -> None:
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


            new_pose = self.poses[-1] @ np.linalg.inv(homogenize(rvec, tvec))
            self.points3d.append(transform_points3d(np.linalg.inv(new_pose), new_point.points3d))
            self.desc3d.append(new_point.desc3d)
            self.poses.append(new_pose)
            self.points.append(new_point)
            self.pose_graph_optimization()
            # ======================================

    def get_trajectory(self):
        return self.poses

    def get_map(self): 
        return [pt.points3d for pt in self.points]

    def pose_graph_optimization(self):
        points = self.points[-self.window:]
        poses = self.poses[-self.window:]
        pose_graph = [[None for _ in range(len(points))] for _ in range(len(points))]
        i = 0
        for j in range(len(points)): 
            if i != j:
                rvec, tvec = relative_transformation(points[i].matcher,
                                            points[i].points3d,
                                            points[i].desc3d,
                                            points[j].left_kp,
                                            points[j].left_desc2d,
                                            self.K_l)
                # pose_j ~ pose_i @ pose_graph[i][j]
                pose_graph[i][j] = np.linalg.inv(homogenize(rvec, tvec))
        opt_poses = optimize_poses(poses, pose_graph)
        self.poses[-self.window:] = opt_poses


        

        




    


    