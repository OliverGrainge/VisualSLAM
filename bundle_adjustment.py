import cv2
from typing import List, Tuple
import numpy as np


def data_association(
    matcher,
    points2d_pos: List[np.ndarray],
    points2d_desc: List[np.ndarray],
    points3d_desc: np.ndarray,
    max_matches_per_image: int=10
) -> Tuple[np.ndarray]:
    point_indices = []
    camera_indices = []
    observations = []
    for camera_idx, desc2d in enumerate(points2d_desc):
        matches = matcher.knnMatch(desc2d, points3d_desc, k=2)
        good_matches = []
        ratio_threshold = 0.75  # Commonly used threshold; adjust based on your dataset
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                        good_matches.append(m)
        good_matches = sorted(good_matches, key=lambda x: x.distance)
        good_matches = good_matches[:max_matches_per_image]
        observations += [list(points2d_pos[camera_idx][match.queryIdx]) for match in good_matches]
        point_indices += [match.trainIdx for match in good_matches]
        camera_indices += [camera_idx for _ in good_matches]
    return np.array(observations), np.array(camera_indices), np.array(point_indices)


def points2params(poses: List[np.ndarray], points3d_pos: np.ndarray) -> np.ndarray:
    params = []
    for pose in poses:
        translation = list(pose[:3, 3].flatten())
        rotation, _ = cv2.Rodrigues(pose[:3, :3])
        rotation = list(rotation.flatten())
        params += rotation
        params += translation
    params += list(points3d_pos.flatten())
    return np.array(params)

def params2points(params: np.ndarray, n_poses: int) -> Tuple[np.ndarray]:
    poses = [np.eye(4)]
    camera_params = params[:n_poses *6]
    point_params = params[n_poses * 6:]
    for camera_idx in range(n_poses):
        rot, _ = cv2.Rodrigues(camera_params[camera_idx*6:camera_idx*6 + 3])
        trans = camera_params[camera_idx*6 + 3: camera_idx*6 + 6]
        poses[camera_idx][:3, :3] = rot 
        poses[camera_idx][:3, 3] = trans
    point_params = point_params.reshape(-1, 3)
    return poses, point_params


"""
    camera_indices: 
        indiciates which 2d observations are from which camera

    point_indices: 
        which 3d points each observation corresponds to
"""


def reprojection_error(
    params: np.ndarray,
    n_cameras: int,
    camera_indices: np.ndarray,
    point_indices: np.ndarray,
    points2d: np.ndarray,
    camera_model: np.ndarray,
):  
    residuals = []
    points3d = params[n_cameras * 6 :].reshape(-1, 3)
    for camera_idx in range(n_cameras):
        rvec = params[camera_idx * 6 : camera_idx * 6 + 3]
        tvec = params[camera_idx * 6 + 3 : camera_idx * 6 + 6]
        observation_mask = camera_indices == camera_idx
        camera_observations = points2d[observation_mask]
        camera_points3d = points3d[point_indices[observation_mask]]
        projected_points, _ = cv2.projectPoints(
            camera_points3d, rvec, tvec, camera_model, np.zeros(4)
        )
        projected_points = projected_points.squeeze()
        res = camera_observations - projected_points
        residuals += list(res.flatten())
    return np.array(residuals)

