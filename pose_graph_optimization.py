from typing import List

import numpy as np
from scipy.optimize import least_squares

from utils import homogenize, unhomogenize


def compute_error(
    pose_params: np.ndarray, poses_graph_transformations: List[List[np.ndarray]]
):
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
    params = []
    for pose in poses:
        rvec, tvec = unhomogenize(pose)
        params += list(rvec.flatten())
        params += list(tvec.flatten())
    return np.array(params)


def params2pose(pose_params: np.ndarray):
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
    """Optimize poses given a pose graph and initial guesses for each pose."""
    pose_params = pose2params(poses)
    result = least_squares(
        compute_error, pose_params, args=(pose_graph_transformation,)
    )
    optimized_poses = params2pose(result.x)
    return optimized_poses
