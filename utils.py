import yaml
import numpy as np
import cv2
from typing import Tuple

def get_config():
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)


def get_feature_matcher(**kwargs):
    config = get_config()
    module_name = "Matching"
    feature_matcher = __import__(module_name, fromlist=[config["feature_matcher"]])
    feature_matcher = getattr(feature_matcher, config["feature_matcher"])
    return feature_matcher(**kwargs)


def get_feature_detector(**kwargs):
    config = get_config()
    module_name = "Features.LocalFeatures"
    feature_detector = __import__(module_name, fromlist=[config["feature_detector"]])
    feature_detector = getattr(feature_detector, config["feature_detector"])
    return feature_detector(**kwargs)


def projection_matrix(rvec: np.ndarray, tvec: np.ndarray, k):
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


def homogenize(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    transformation = np.eye(4)
    R, _ = cv2.Rodrigues(rvec)
    transformation[:3, :3] = R.squeeze()
    transformation[:3, 3] = tvec.squeeze()
    return transformation


def unhomogenize(T: np.ndarray) -> Tuple[np.ndarray]:
    assert T.shape[0] == 4
    assert T.shape[1] == 4
    rot = T[:3, :3]
    rvec, _ = cv2.Rodrigues(rot)
    tvec = T[:3, 3]
    return rvec.flatten(), tvec