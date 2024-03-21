import time
from collections import deque
from typing import List, Tuple

import cv2
import numpy as np
import yaml
from PIL import Image
from scipy.optimize import least_squares
from sklearn.metrics.pairwise import cosine_similarity

import point_features
from point_features import SIFT

np.set_printoptions(precision=3, suppress=True)


def get_config():
    """
    Loads and returns the configuration data from a YAML file named 'config.yaml'.

    Returns:
        dict: A dictionary containing the configuration data.
    """

    with open("config.yaml", "r") as file:
        data = yaml.safe_load(file)
    return data


def get_feature_detector():
    """
    Creates and returns a feature detector object based on the configuration specified in 'config.yaml'.

    The configuration file should specify the type of feature detector to use under 'feature_detector'.

    Returns:
        point_features.ORB or point_features.SIFT: An instance of the specified feature detector.

    Raises:
        NotImplementedError: If the feature detector specified in the configuration is not supported.
    """
    config = get_config()
    if "orb" in config["feature_detector"].lower():
        return point_features.ORB()
    elif "sift" in config["feature_detector"].lower():
        return point_features.SIFT()
    else:
        raise NotImplementedError


def get_feature_matcher():
    """
    Creates and returns a feature matcher object based on the configuration specified in 'config.yaml'.

    The configuration file should specify the type of feature detector to use under 'feature_detector', 
    which determines the appropriate feature matcher to be returned.

    Returns:
        cv2.BFMatcher: An instance of a brute force matcher with the appropriate norm type.

    Raises:
        NotImplementedError: If the feature matcher for the specified feature detector is not supported.
    """

    config = get_config()
    if "orb" in config["feature_detector"].lower():
        return cv2.BFMatcher(cv2.NORM_HAMMING)
    elif "sift" in config["feature_detector"].lower():
        return cv2.BFMatcher(cv2.NORM_L2)
    else:
        raise NotImplementedError


def get_matches(
    matcher, desc1: np.ndarray, desc2: np.ndarray, ratio_threshold=0.75, top_N=None
) -> List[cv2.KeyPoint.KeyPoint]:
    """
    Finds and filters matches between two sets of descriptors based on the ratio test and optionally selects the top N matches.

    Parameters:
        matcher: The matcher object to use for finding matches.
        desc1 (np.ndarray): The first set of descriptors.
        desc2 (np.ndarray): The second set of descriptors.
        ratio_threshold (float, optional): The threshold for the Lowe's ratio test. Defaults to 0.75.
        top_N (int, optional): The number of top matches to select based on their distance. If None, all good matches are returned.

    Returns:
        list: A list of filtered matches.
    """
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


def sort_matches(matches: List, left_kp: List, right_kp: List, left_desc: np.ndarray) -> Tuple[List[cv2.KeyPoint.KeyPoint], List[cv2.KeyPoint.KeyPoint], np.ndarray]:
    """
    Sorts matches and extracts corresponding points and descriptors.

    Parameters:
        matches (List): A list of matches.
        left_kp (List): Key points from the left image.
        right_kp (List): Key points from the right image.
        left_desc (np.ndarray): Descriptors from the left image.

    Returns:
        tuple: Three numpy arrays containing the sorted points from the left and right images, and the sorted descriptors from the left image.
    """
    points_left = np.zeros((len(matches), 2))
    points_right = np.zeros((len(matches), 2))
    points_desc = np.zeros((len(matches), left_desc.shape[1]), dtype=np.float32)
    for i, match in enumerate(matches):
        points_left[i, :] = left_kp[match.queryIdx].pt
        points_right[i, :] = right_kp[match.trainIdx].pt
        points_desc[i, :] = left_desc[match.queryIdx]
    return points_left, points_right, points_desc


def homogenize(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    Converts rotation and translation vectors into a homogeneous transformation matrix.

    Parameters:
        rotation (np.ndarray): The rotation vector.
        translation (np.ndarray): The translation vector.

    Returns:
        np.ndarray: A 4x4 homogeneous transformation matrix.
    """
    transformation = np.eye(4)
    R, _ = cv2.Rodrigues(rotation)
    transformation[:3, :3] = R.squeeze()
    transformation[:3, 3] = translation.squeeze()
    return transformation


def unhomogenize(pose: np.ndarray) -> Tuple[np.ndarray]:
    """
    Converts a 4x4 homogeneous transformation matrix into rotation and translation vectors.

    Parameters:
        pose (np.ndarray): A 4x4 homogeneous transformation matrix.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Rotation and translation vectors extracted from the transformation matrix.
    """
    assert pose.shape[0] == 4
    assert pose.shape[1] == 4
    rot = pose[:3, :3]
    rvec, _ = cv2.Rodrigues(rot)
    tvec = pose[:3, 3]
    return rvec.flatten(), tvec


def transform_points3d(trans: np.ndarray, points: np.ndarray):
    """
    Applies a 4x4 transformation matrix to a set of 3D points.

    Parameters:
        trans (np.ndarray): A 4x4 transformation matrix.
        points (np.ndarray): A set of 3D points.

    Returns:
        np.ndarray: The transformed 3D points.
    """
    points = np.hstack((points, np.ones((len(points), 1))))
    assert points.shape[1] == 4
    tpoints = np.dot(trans, points.T).T
    tpoints = tpoints[:, :3] / tpoints[:, 3].reshape(-1, 1)
    return tpoints


def decompose_projection(proj: np.ndarray) -> np.ndarray:
    """
    Decomposes a projection matrix into its intrinsic matrix, rotation matrix, and translation vector components.

    Parameters:
        proj (np.ndarray): A projection matrix.

    Returns:
        tuple: The intrinsic matrix, rotation matrix, and translation vector.
    """
    K, R, T = cv2.decomposeProjectionMatrix(proj)[:3]
    T = T.flatten()
    T = T[:3] / T[3]
    T = -T
    return (K, R, T.reshape(-1, 1))


def projection_matrix(rvec: np.ndarray, tvec: np.ndarray, k: np.ndarray):
    """
    Constructs a projection matrix from rotation and translation vectors and an intrinsic matrix.

    Parameters:
        rvec (np.ndarray): A rotation vector.
        tvec (np.ndarray): A translation vector.
        k (np.ndarray): An intrinsic matrix.

    Returns:
        np.ndarray: The constructed projection matrix.
    """
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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimates the relative camera pose (rotation and translation vectors) between a 3D scene and its 2D image projection using the Perspective-n-Point (PnP) algorithm.

    This function matches 3D descriptors with 2D descriptors, sorts the correspondences to align 3D points with their 2D projections, and then applies the PnP algorithm to estimate the relative pose.

    Parameters:
        matcher: The matcher object used to find matches between descriptors.
        points3d (np.ndarray): A Numpy array of 3D points in the scene.
        desc3d (np.ndarray): A Numpy array of descriptors corresponding to the 3D points.
        left_kp (List): A list of keypoints in the 2D image, typically obtained from a feature detector.
        desc2d (np.ndarray): A Numpy array of descriptors corresponding to the keypoints in the 2D image.
        K_l (np.ndarray): The intrinsic camera matrix of the 2D image's capturing camera.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the rotation vector and translation vector that represent the estimated camera pose.

    Raises:
        Exception: If the PnP algorithm fails to find a valid solution.

    Note:
        The PnP algorithm is applied using RANSAC for robustness against outliers.
    """
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
