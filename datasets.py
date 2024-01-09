
from PIL import Image
from typing import Tuple
import numpy as np
from glob import glob
from os.path import join
import pandas as pd
import utils
import cv2





class KittiSampleDataset: 
    def __init__(self, config: dict, sequence: str="05"):
        super().__init__()
        self.data_dir = config["datasets_directory"]
        self.sequence_dir = join(self.data_dir, "kitti/data_odometry_gray/dataset/sequences/")
        self.image_paths = sorted(glob(self.sequence_dir + sequence + "/image_0/*.png"))
        self.calib = utils.read_calib_file(self.sequence_dir + sequence + "/calib.txt")
        self.poses = utils.read_pose_file(self.sequence_dir + "poses/" + sequence + ".txt")

        assert self.poses.shape[0] == len(self.image_paths)

    def __len__(self): 
        return len(self.image_paths)

    def load_image(self, idx: int) -> Image:
        img = Image.open(self.image_paths[idx])
        return img
    
    def initial_pose(self): 
        return self.poses[0]
    
    def projection_calib(self) -> np.ndarray:
        return self.calib.to_numpy()[0].reshape(3, 4)

    def intrinsic_calib(self) -> np.ndarray:
        k1, r1, t1, _, _, _, _= cv2.decomposeProjectionMatrix(self.calib.to_numpy()[0].reshape(3, 4))
        return k1

    def ground_truth(self):
        return self.poses


if __name__ == "__main__":
    config = utils.get_config()
    ds = KittiSampleDataset(config)