import cv2
import numpy as np
from PIL import Image
from typing import Tuple
from collections import deque
from point_features import SIFT
from typing import List
from scipy.optimize import least_squares
import time
import yaml
from sklearn.metrics.pairwise import cosine_similarity

np.set_printoptions(precision=3, suppress=True)


def get_config(): 
    with open("config.yaml", 'r') as file:
        data = yaml.safe_load(file)
    return data


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
    return rvec.flatten(), tvec


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
    matches = get_matches(matcher, desc3d, desc2d, top_N=None)
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