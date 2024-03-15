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
from sklearn.metrics.pairwise import cosine_similarity

np.set_printoptions(precision=3, suppress=True)


def get_matches(
    matcher, desc1: np.ndarray, desc2: np.ndarray, ratio_threshold=0.75, top_N=None
):
    matches = matcher.knnMatch(desc1, desc2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
    matches = good_matches
    if top_N is not None:
        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[:top_N]
    return matches


def sort_matches(matches: List, left_kp: List, right_kp: List, left_desc: np.ndarray):
    points_left = np.zeros((len(matches), 2))
    points_right = np.zeros((len(matches), 2))
    points_desc = np.zeros((len(matches), left_desc.shape[1]), dtype=np.float32)
    for i, match in enumerate(matches):
        points_left[i, :] = left_kp[match.queryIdx].pt
        points_right[i, :] = right_kp[match.trainIdx].pt
        points_desc[i, :] = left_desc[match.queryIdx]
    return points_left, points_right, points_desc


def homogenize(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transformation = np.eye(4)
    R, _ = cv2.Rodrigues(rotation)
    transformation[:3, :3] = R.squeeze()
    transformation[:3, 3] = translation.squeeze()
    return transformation


def unhomogenize(pose: np.ndarray) -> Tuple[np.ndarray]:
    assert pose.shape[0] == 4
    assert pose.shape[1] == 4
    rot = pose[:3, :3]
    rvec, _ = cv2.Rodrigues(rot)
    tvec = pose[:3, 3]
    return rvec, tvec


def transform_points3d(trans: np.ndarray, points: np.ndarray):
    points = np.hstack((points, np.ones((len(points), 1))))
    assert points.shape[1] == 4
    tpoints = np.dot(trans, points.T).T
    tpoints = tpoints[:, :3] / tpoints[:, 3].reshape(-1, 1)
    return tpoints


def decompose_projection(proj: np.ndarray) -> np.ndarray:
    K, R, T = cv2.decomposeProjectionMatrix(proj)[:3]
    T = T.flatten()
    T = T[:3] / T[3]
    T = -T
    return (K, R, T.reshape(-1, 1))


def projection_matrix(rvec: np.ndarray, tvec: np.ndarray, k: np.ndarray):
    assert len(rvec.squeeze()) == 3
    assert len(tvec.squeeze()) == 3
    proj = np.eye(4)[:3, :]
    rmat, _ = cv2.Rodrigues(rvec)
    assert rmat.shape[0] == 3
    assert rmat.shape[1] == 3
    proj[:3, :3] = rmat
    proj[:3, 3] = tvec.squeeze()
    proj = np.dot(k, proj)
    return proj


def relative_transformation(
    matcher,
    points3d: np.ndarray,
    desc3d: np.ndarray,
    left_kp: List,
    desc2d: np.ndarray,
    K_l: np.ndarray,
):
    matches = get_matches(matcher, desc3d, desc2d)
    points3d_sorted = np.zeros((len(matches), 3), dtype=np.float32)
    points2d_sorted = np.zeros((len(matches), 2), dtype=np.float32)
    points_desc = np.zeros((len(matches), desc3d.shape[1]), dtype=np.float32)
    for i, match in enumerate(matches):
        points3d_sorted[i, :] = points3d[match.queryIdx]
        points2d_sorted[i, :] = left_kp[match.trainIdx].pt
        points_desc[i, :] = desc3d[match.queryIdx]

    success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
        points3d_sorted,
        points2d_sorted,
        K_l,
        np.zeros(4),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise Exception("PnP algorithm failed")
    return rotation_vector, translation_vector


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


class StereoOdometry:
    def __init__(
        self,
        left_projection: np.ndarray,
        right_projection: np.ndarray,
        initial_pose: np.ndarray,
        window: int=3,
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
            self.bundle_adjustment()
            # ======================================

    def get_trajectory(self):
        return self.poses

    def get_map(self): 
        return [pt.points3d for pt in self.points]

    @staticmethod 
    def get_pos(keypoints: List) -> np.ndarray:
        pos = []
        for kp in keypoints: 
            pos.append(np.array([kp.pt[0], kp.pt[1]]))
        return np.vstack(pos)
    
    @staticmethod
    def data_association(matcher, desc3d: np.ndarray, points2d: np.ndarray, desc2d: np.ndarray) -> Tuple[np.ndarray]: 
        """
        This function takes a set of 3d descriptors aswell as 2d points and descriptors. 
        It returns a set of points that match the 3d points and a set of indices as to which 
        points they are 
        
        """
        matches = get_matches(matcher, desc2d, desc3d)
        sorted_points2d = np.zeros((len(matches), 2))
        point_indices = np.zeros(len(matches)).astype(int)
        for idx, match in enumerate(matches):
            sorted_points2d[idx, :] = points2d[match.queryIdx]
            point_indices[idx] = match.trainIdx
        
        return sorted_points2d, point_indices


    @staticmethod
    def reproject(points3d: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray):
        p_points, _ = cv2.projectPoints(points3d, rvec, tvec, K, np.zeros(4))
        return p_points.squeeze()

    @staticmethod
    def pose2params(poses: np.ndarray): 
        params = []
        for pose in poses: 
            rvec, tvec = unhomogenize(pose)
            params += list(rvec.flatten())
            params += list(tvec.flatten())
        return np.array(params)

    @staticmethod 
    def params2pose(params: np.ndarray):
        assert len(params) % 6 == 0
        poses = []
        for idx in range(len(params) // 6): 
            p = params[idx*6:idx*6 + 6]
            rvec = np.array(p[:3])
            tvec = np.array(p[3:])
            pose = homogenize(rvec, tvec)
            poses.append(pose)
        return poses


    def ba_cost(self, params: np.ndarray,
                n_camera: int, 
                point_indices: List[np.ndarray],
                sorted_points2d: List[np.ndarray],
                points3d: np.ndarray, 
                camera_model: np.ndarray):
        all_errors = []
        for idx in range(n_camera): 
            pose = params[idx * 6: idx * 6 + 6]
            rvec = pose[:3]
            tvec = pose[3:]
            indices = point_indices[idx]
            pt2d = sorted_points2d[idx]
            pt3d = points3d[indices]
            p_p2d, _ = cv2.projectPoints(pt3d, rvec, tvec, camera_model, np.zeros(4))
            p_p2d = p_p2d.squeeze()
            errors = list((pt2d - p_p2d).flatten())
            all_errors += errors
        return np.array(all_errors)

    def bundle_adjustment(self): 
        points = self.points[-self.window:]
        poses = self.poses[-self.window:]
        points3d = self.points3d[-self.window:]
        points3d_desc = self.desc3d[-self.window:]
        points2d_pos = [self.get_pos(pt.left_kp) for pt in points]
        points2d_desc = [pt.left_desc2d for pt in points]
        
        # make a single point cloud from the last up to date image
        points3d = points3d[0]
        points3d_desc = points3d_desc[0]
        #points3d = np.vstack(points3d)
        #points3d_desc = np.vstack(points3d_desc)

        point_indices = []
        sorted_points2d = []
        for idx in range(len(points)):
            pt2d, indices = self.data_association(points[idx].matcher,
                                                  points3d_desc, 
                                                  points2d_pos[idx],
                                                  points2d_desc[idx])
            point_indices.append(indices)
            sorted_points2d.append(pt2d)

        params = self.pose2params(poses)
        n_camera = len(points)
        args = (n_camera, point_indices, sorted_points2d, points3d, self.K_l)
        errors = self.ba_cost(params, *args)
        result = least_squares(self.ba_cost, params, args=args, method='lm')
        errors_opt = self.ba_cost(result.x, *args)
        poses = self.params2pose(result.x)
        self.poses[-self.window:] = poses

    
    """
    def bundle_adjustment(self): 
        points = self.points[-self.window:]
        poses = self.poses[-self.window:]
        points3d = self.points3d[-self.window:]
        points3d_desc = self.desc3d[-self.window:]
        points2d_pos = [self.get_pos(pt.left_kp) for pt in points]
        points2d_desc = [pt.left_desc2d for pt in points]
        
        # make a single point cloud from the last up to date image
        #points3d = points3d[0]
        #points3d_desc = points3d_desc[0]
        points3d = np.vstack(points3d)
        points3d_desc = np.vstack(points3d_desc)

        point_indices = []
        sorted_points2d = []
        for idx in range(len(points)):
            pt2d, indices = self.data_association(points[idx].matcher,
                                                  points3d_desc, 
                                                  points2d_pos[idx],
                                                  points2d_desc[idx])
            point_indices.append(indices)
            sorted_points2d.append(pt2d)

        params = self.pose2params(poses)
        n_camera = len(points)

        
        for idx in range(n_camera):
            pose = params[idx * 6: idx * 6 + 6]
            rvec = pose[:3]
            tvec = pose[3:]
            indices = point_indices[idx]
            pt2d = sorted_points2d[idx]
            pt3d = points3d[indices]
            p_p2d = self.reproject(pt3d, rvec, tvec, self.K_l)
            errors = (pt2d - p_p2d).flatten()
            print("hello", np.abs(errors).mean())
    """

    """
    def bundle_adjustment(self): 
        points = self.points[-self.window:]
        poses = self.poses[-self.window:]
        points3d = self.points3d[-self.window:]
        points3d_desc = self.desc3d[-self.window:]
        points2d_pos = [self.get_pos(pt.left_kp) for pt in points]
        points2d_desc = [pt.left_desc2d for pt in points]

        camera_indices = []
        sorted_points2d = []
        for idx in range(len(points)):
            pt2d, indices = self.data_association(points[idx].matcher,
                                                  points3d_desc[idx], 
                                                  points2d_pos[idx],
                                                  points2d_desc[idx])
            camera_indices.append(indices)
            sorted_points2d.append(pt2d)

        params = self.pose2params(poses)
        n_camera = len(points)

        for idx in range(n_camera):
            pose = params[idx * 6: idx * 6 + 6]
            rvec = pose[:3]
            tvec = pose[3:]
            indices = camera_indices[idx]
            pt2d = sorted_points2d[idx]
            pt3d = points3d[idx][indices]
            p_p2d = self.reproject(pt3d, rvec, tvec, self.K_l)
            errors = (pt2d - p_p2d).flatten()
            print(np.abs(errors).mean())
    """

        

        
        


        



    