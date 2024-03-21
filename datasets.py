from glob import glob
from os.path import join
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import utils


class KittiDataset:
    """
    A class for handling and loading data from the KITTI odometry dataset, including stereo images, calibration parameters, and ground truth poses.

    Attributes:
        data_dir (str): Directory path to the dataset.
        sequence_dir (str): Path to the sequence directory within the KITTI dataset.
        image_paths_left (list of str): Sorted list of paths to left camera images.
        image_paths_right (list of str): Sorted list of paths to right camera images.
        K_l (np.ndarray): Intrinsic camera matrix for the left camera.
        P_l (np.ndarray): Projection matrix for the left camera.
        K_r (np.ndarray): Intrinsic camera matrix for the right camera.
        P_r (np.ndarray): Projection matrix for the right camera.
        poses (np.ndarray): Ground truth poses for the sequence.

    Parameters:
        config (dict): Configuration dictionary containing dataset settings, including the datasets directory.
        sequence (str, optional): The sequence number to load. Defaults to "09".
    """
    def __init__(self, config: dict, sequence: str = "09"):
        """
        Initializes the KittiDataset object by loading image paths, calibration data, and ground truth poses for the specified sequence.
        """
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
        Loads the calibration data from a specified file.

        Parameters:
            filepath (str): Path to the calibration file.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Returns the intrinsic camera matrix for the left camera (K_l),
            the projection matrix for the left camera (P_l), the intrinsic camera matrix for the right camera (K_r),
            and the projection matrix for the right camera (P_r).
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
        Loads the ground truth poses from a specified file.

        Parameters:
            filepath (str): Path to the file containing the ground truth poses.

        Returns:
            np.ndarray: An array of ground truth poses.
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
        """
        Returns the number of stereo image pairs in the dataset sequence.

        Returns:
            int: The number of stereo image pairs.
        """

        return len(self.image_paths_left)

    def load_images(self, idx: int) -> Tuple[Image.Image]:
        """
        Loads a pair of stereo images at the specified index.

        Parameters:
            idx (int): The index of the image pair to load.

        Returns:
            Tuple[Image.Image, Image.Image]: A tuple containing the left and right images as PIL.Image objects.
        """
        img_left = Image.open(self.image_paths_left[idx])
        img_right = Image.open(self.image_paths_right[idx])
        return (img_left, img_right)

    def initial_pose(self) -> np.ndarray:
        """
        Returns the initial pose from the ground truth poses.

        Returns:
            np.ndarray: The initial pose as a 4x4 transformation matrix.
        """
        return self.poses[0]

    def projection_matrix(self) -> Tuple[np.ndarray]:
        """
        Returns the projection matrices for the left and right cameras.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The projection matrices (P_l, P_r).
        """
        return (self.P_l, self.P_r)

    def intrinsic_calib(self) -> Tuple[np.ndarray]:
        """
        Decomposes the projection matrices to retrieve the intrinsic calibration matrices for the left and right cameras.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The intrinsic calibration matrices for the left and right cameras (K_l, K_r).
        """
        k1, r1, t1, _, _, _, _ = cv2.decomposeProjectionMatrix(self.P_l)
        k2, r2, t2, _, _, _, _ = cv2.decomposeProjectionMatrix(self.P_r)
        return k1, k2

    def ground_truth(self) -> List[np.ndarray]:
        """
        Returns the ground truth poses for the sequence.

        Returns:
            List[np.ndarray]: The ground truth poses.
        """
        return self.poses


if __name__ == "__main__":
    config = utils.get_config()
    ds = KittiDataset(config)
    imgl, imgr = ds.load_image(10)
    x = ds.initial_pose()
    Pl, Pr = ds.projection_calib()
    k1, k2 = ds.intrinsic_calib()
    gt = ds.ground_truth()
