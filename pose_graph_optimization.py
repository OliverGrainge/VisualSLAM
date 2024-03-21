from typing import List

import numpy as np
from scipy.optimize import least_squares

from utils import homogenize, unhomogenize


def compute_error(
    pose_params: np.ndarray, poses_graph_transformations: List[List[np.ndarray]]
):
    """
    Computes the residual errors between predicted poses and target poses based on the transformations defined in a pose graph.

    Parameters:
        pose_params (np.ndarray): A flat array of pose parameters (rotation vector followed by translation vector) for all poses.
        poses_graph_transformations (List[List[np.ndarray]]): A 2D list of transformation matrices defining the relative poses between poses in the graph.

    Returns:
        np.ndarray: A flat array of residual errors for rotation and translation vectors.
    """
    errors = []
    n_poses = len(pose_params) // 6
    poses = [
        homogenize(pose_params[i * 6 : i * 6 + 3], pose_params[i * 6 + 3 : i * 6 + 6])
        for i in range(n_poses)
    ]
    for i in range(n_poses):
        for j in range(n_poses):
            if i != j and poses_graph_transformations[i][j] is not None:
                targ_rvec, targ_tvec = unhomogenize(poses[j])
                pred_rvec, pred_tvec = unhomogenize(
                    poses[i] @ poses_graph_transformations[i][j]
                )
                res_rvec = list(targ_rvec - pred_rvec)
                res_tvec = list(targ_tvec - pred_tvec)
                errors += res_rvec
                errors += res_tvec
    return np.array(errors)


def pose2params(poses: List[np.ndarray]):
    """
    Converts a list of pose matrices into a flat array of pose parameters.

    Parameters:
        poses (List[np.ndarray]): A list of 4x4 pose matrices.

    Returns:
        np.ndarray: A flat array of pose parameters, where each set of six parameters corresponds to the rotation vector and translation vector of a pose.
    """

    params = []
    for pose in poses:
        rvec, tvec = unhomogenize(pose)
        params += list(rvec.flatten())
        params += list(tvec.flatten())
    return np.array(params)


def params2pose(pose_params: np.ndarray):
    """
    Converts a flat array of pose parameters back into a list of pose matrices.

    Parameters:
        pose_params (np.ndarray): A flat array of pose parameters, where each set of six parameters corresponds to the rotation vector and translation vector of a pose.

    Returns:
        List[np.ndarray]: A list of 4x4 pose matrices constructed from the provided pose parameters.
    """
    n_poses = len(pose_params) // 6
    poses = []
    for idx in range(n_poses):
        poses.append(
            homogenize(
                pose_params[idx * 6 : idx * 6 + 3],
                pose_params[idx * 6 + 3 : idx * 6 + 6],
            )
        )
    return poses


def optimize_poses(
    poses: List[np.ndarray], pose_graph_transformation: List[List[np.ndarray]]
):
    """
    Optimizes poses given an initial guess for each pose and a pose graph defining the relative transformations between poses.

    Parameters:
        poses (List[np.ndarray]): An initial guess for the poses to be optimized.
        pose_graph_transformation (List[List[np.ndarray]]): A 2D list of transformation matrices defining the relative poses between poses in the graph.

    Returns:
        List[np.ndarray]: A list of optimized pose matrices.
    """
    pose_params = pose2params(poses)
    result = least_squares(
        compute_error, pose_params, args=(pose_graph_transformation,)
    )
    optimized_poses = params2pose(result.x)
    return optimized_poses
