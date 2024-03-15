import cv2
from typing import List, Tuple
import numpy as np


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


def data_association(
    matcher,
    points2d_pos: List[np.ndarray],
    points2d_desc: List[np.ndarray],
    points3d_desc: np.ndarray,
    max_matches_per_image: int = 1000,
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
        observations += [
            list(points2d_pos[camera_idx][match.queryIdx]) for match in good_matches
        ]
        point_indices += [match.trainIdx for match in good_matches]
        camera_indices += [camera_idx for _ in good_matches]
    return np.array(observations), np.array(camera_indices), np.array(point_indices)


def points2params(poses: List[np.ndarray]) -> np.ndarray:
    params = []
    for pose in poses:
        rvec, tvec = unhomogenize(pose)
        params += list(rvec.flatten())
        params += list(tvec.flatten())
    return np.array(params)


def params2points(params: np.ndarray, n_cameras: int) -> Tuple[np.ndarray]:
    poses = [np.eye(4) for _ in range(n_cameras)]
    camera_params = params[: n_cameras * 6]
    for camera_idx in range(n_cameras):
        rvec = camera_params[camera_idx * 6 : camera_idx * 6 + 3]
        tvec = camera_params[camera_idx * 6 + 3 : camera_idx * 6 + 6]
        poses.append(homogenize(rvec, tvec))
    return poses


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
    points3d: np.ndarray,
    camera_model: np.ndarray,
):
    residuals = []
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
