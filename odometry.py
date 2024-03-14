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


def get_matches(matcher, desc1: np.ndarray, desc2: np.ndarray, ratio_threshold=0.75, top_N=None): 
        matches = matcher.knnMatch(
            desc1, desc2, k=2
        )
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
        matches = good_matches
        if top_N is not None: 
            matches = sorted(matches, key=lambda x:x.distance)
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
    tpoints = np.dot(points, trans)
    tpoints = tpoints[:, :3] / tpoints[:, 3].reshape(-1, 1)
    return tpoints


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



def relative_transformation(matcher, points3d: np.ndarray, desc3d: np.ndarray, left_kp: List, desc2d: np.ndarray, K_l: np.ndarray):
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
        points_left, points_right, points_desc = sort_matches(matches, self.left_kp, self.right_kp, self.left_desc2d)

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
        window_size: int = 10,
    ):
        self.left_projection = left_projection
        self.right_projection = right_projection
        self.initial_projection_left = left_projection 
        self.initial_projection_right = right_projection
        self.left_to_right = (right_projection[:3, 3] - left_projection[:3, 3]).reshape(-1, 1)
        self.initial_pose = initial_pose
        self.window_size = window_size
        self.K_l = cv2.decomposeProjectionMatrix(self.left_projection)[0]
        self.K_r = cv2.decomposeProjectionMatrix(self.right_projection)[0]

        self.poses = [initial_pose]
        self.cumulative_transform = None
        self.points = []
        self.odometry_points = []

    def process_images(self, image_left: Image.Image, image_right: Image.Image) -> None:
        if len(self.points) == 0:
            point = StereoPoint(image_left, image_right, self.left_projection, self.right_projection, self.K_l, self.K_r)   
            point.featurepoints2d()
            point.triangulate()
            self.points.append(point)
        else:
            new_point = StereoPoint(image_left, image_right, self.initial_projection_left, self.initial_projection_right, self.K_l, self.K_r)
            new_point.featurepoints2d()
            prev_point = self.points[-1]
            rvec, tvec = relative_transformation(prev_point.matcher, prev_point.points3d, prev_point.desc3d, new_point.left_kp, new_point.left_desc2d, new_point.K_l)
            if self.cumulative_transform is None: 
                self.cumulative_transform = np.linalg.inv(homogenize(rvec, tvec))
            else: 
                self.cumulative_transform = self.cumulative_transform @ np.linalg.inv(homogenize(rvec, tvec))
            
            new_pose_left = self.poses[-1] @ np.linalg.inv(homogenize(rvec, tvec))
            new_point.triangulate()
            self.poses.append(new_pose_left)
            self.points.append(new_point)
            #self.bundle_adjust()


    def get_trajectory(self): 
        return self.poses
"""
    def bundle_adjust(self) -> None:
        import matplotlib.pyplot as plt
        # 2d array of the 3d point positions
        

        max_3d_points_per_image = 10
        all_points3d = [pt.points3d[:max_3d_points_per_image] for pt in self.points[-self.window_size:]]
        all_points_old = all_points3d
        all_desc3d = [pt.desc3d[:max_3d_points_per_image] for pt in self.points[-self.window_size:]]
        all_poses = self.poses[-self.window_size:]
        all_points3d = [transform_points3d(all_poses[idx], all_points3d[idx]) for idx in range(len(all_poses))]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ts = np.vstack([p[:3, 3] for p in self.poses])
        ax.plot(ts[:, 0], ts[:, 1], ts[:, 2], markersize=10, label="trajectory")
        for idx, point in enumerate(all_points3d):
            ax.scatter(point[:, 0], point[:, 1], point[:, 2], label="triangulated " + str(idx))
        plt.legend()
        plt.show() 
        
        #plt.scatter(all_poses[0][:3, 0], all_poses[0][:3, 1], all_poses[0][:3, 2], label="Origin")
        all_points3d = [transform_points3d(all_poses[idx], all_points3d[idx]) for idx in range(len(all_poses))]
        #plt.scatter(all_poses[1][:3, 0], all_poses[1][:3, 1], all_poses[1][:3, 2], label="T2")
        #plt.scatter(all_points3d[1][:, 0], all_points3d[1][:, 1], all_points3d[1][:, 2], label="transformed")

        points3d = np.vstack(all_points3d)
        desc3d = np.vstack(all_desc3d)
        # 3x3 matrix of the camera intrinsic parameters
        camera_model = self.K_l
        # A list of the left camera project matrices
        camera_poses = self.poses[-self.window_size :]
        # number of camera images in the bundle
        n_cameras = len(camera_poses)
        # a list of the 2d observation descriptors
        points2d_desc = [
            pt.left_desc2d for pt in self.points[-self.window_size :]
        ]
        # a list of the 2d observation position arrays
        points2d_pos = []
        for op in self.points[-self.window_size :]:
            point_pos = []
            for kp in op.left_kp:
                point_pos.append([kp.pt[0], kp.pt[1]])
            point_pos = np.array(point_pos)
            points2d_pos.append(point_pos)

        # ============== Setup the parameters for optimization ====================
        # params[pitch_1, roll_1, yaw_1, tx_1, ty_1, tz_1, pitch_2, roll_2, ...... x1, y1, yz ]
        points3d = transform_points3d(self.poses[-1], points3d)
        params = bundle_adjustment.points2params(camera_poses, points3d)
        # camera_indices indicates which image each 2d observaion corresponds to
        # camera_indices[0, 0, 0, 0, 1, 1, 1, 1, .... n_cameras, n_cameras]
        # points_indices indicates which 3d point the descriptor coressponds to
        # point_indices[1, 4, 32, 14, 58, ... ]
        points2d_pos, camera_indices, point_indices = bundle_adjustment.data_association(
            self.points[-1].matcher, points2d_pos, points2d_desc, desc3d
        )

        error = bundle_adjustment.reprojection_error(
            params, 
            n_cameras, 
            camera_indices,
            point_indices,
            points2d_pos,
            camera_model,
        )


        st = time.time()
        print(len(camera_indices), len(point_indices), len(points2d_pos), len(params))
        result = least_squares(bundle_adjustment.reprojection_error,
                params,
                args=(
                    n_cameras, 
                    camera_indices, 
                    point_indices,
                    points2d_pos,
                    camera_model
                ),
                verbose=0,
                method='lm', 
                ftol=1e-06,
                xtol=1e-06,
                gtol=1e-06)
                #**options)
        optimized_params = result.x 
        
        optimized_error = bundle_adjustment.reprojection_error(
            optimized_params, 
            n_cameras, 
            camera_indices, 
            point_indices, 
            points2d_pos, 
            camera_model,
        )
        print("======", np.abs(error).mean(), np.abs(optimized_error).mean())
        poses, point3d_pose = bundle_adjustment.params2points(optimized_params, len(camera_poses))
        #self.poses[-self.window_size:] = poses     
        print(len(self.poses[-self.window_size:]), len(camera_poses))
        """


