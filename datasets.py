from glob import glob
from os.path import join
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import utils


class KittiDataset:
    def __init__(self, config: dict, sequence: str = "01"):
        super().__init__()
        self.data_dir = config["datasets_directory"]
        self.sequence_dir = join(
            self.data_dir, "kitti/data_odometry_gray/dataset/sequences/"
        )
        self.image_paths_left = sorted(
            glob(self.sequence_dir + sequence + "/image_0/*.png")
        )
        self.image_paths_right = sorted(
            glob(self.sequence_dir + sequence + "/image_1/*.png")
        )
        self.K_l, self.P_l, self.K_r, self.P_r = self._load_calib(
            self.sequence_dir + sequence + "/calib.txt"
        )
        self.poses = self._load_poses(self.sequence_dir + "poses/" + sequence + ".txt")

        assert self.poses.shape[0] == len(self.image_paths_left)

    @staticmethod
    def _load_calib(filepath) -> Tuple[np.ndarray]:
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K_l (ndarray): Intrinsic parameters for left camera. Shape (3,3)
        P_l (ndarray): Projection matrix for left camera. Shape (3,4)
        K_r (ndarray): Intrinsic parameters for right camera. Shape (3,3)
        P_r (ndarray): Projection matrix for right camera. Shape (3,4)
        """
        data = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.split(":", 1)[1]
                numbers = [float(number) for number in line.split()]
                data.append(numbers)

        P_l = np.array(data[0]).reshape(3, 4)
        P_r = np.array(data[1]).reshape(3, 4)
        K_l = P_l[:3, :3]
        K_r = P_r[:3, :3]
        return K_l, P_l, K_r, P_r

    @staticmethod
    def _load_poses(filepath) -> np.ndarray:
        """
        Loads the GT poses

        Parameters
        ----------
        filepath (str): The file path to the poses file

        Returns
        -------
        poses (ndarray): The GT poses. Shape (n, 4, 4)
        """
        poses = []
        with open(filepath, "r") as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=" ")
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return np.array(poses)

    def __len__(self):
        return len(self.image_paths_left)

    def load_images(self, idx: int) -> Tuple[Image.Image]:
        img_left = Image.open(self.image_paths_left[idx])
        img_right = Image.open(self.image_paths_right[idx])
        return (img_left, img_right)

    def initial_pose(self):
        return self.poses[0]

    def projection_matrix(self) -> Tuple[np.ndarray]:
        return (self.P_l, self.P_r)

    def intrinsic_calib(self) -> Tuple[np.ndarray]:
        k1, r1, t1, _, _, _, _ = cv2.decomposeProjectionMatrix(self.P_l)
        k2, r2, t2, _, _, _, _ = cv2.decomposeProjectionMatrix(self.P_r)
        return k1, k2

    def ground_truth(self):
        return self.poses


if __name__ == "__main__":
    config = utils.get_config()
    ds = KittiDataset(config)
    imgl, imgr = ds.load_image(10)
    x = ds.initial_pose()
    Pl, Pr = ds.projection_calib()
    k1, k2 = ds.intrinsic_calib()
    gt = ds.ground_truth()
