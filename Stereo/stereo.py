from typing import List, Union

import numpy as np
from PIL import Image


class StereoPoint:
    def __init__(
        self,
        image: Image.Image,
        right_image: Image.Image,
        T: np.ndarray,
        K: np.ndarray,
        pose: Union[None, np.ndarray] = None,
    ):
        # imaging data and calibrations
        self.image = image
        self.right_image = right_image
        self.K = K
        self.T = T

        # pose of self.image
        self.pose = pose

        # 2d features
        self.keypoints_2d = None
        self.descriptors_2d = None
        self.global_descriptor_2d = None

        # 2d features
        self.keypoints_3d = None
        self.descriptors_3d = None

    def compute_features2d(self):
        """
        Computes the feature points in 2d
        """
        pass

    def compute_features3d(self):
        """
        uses the 2d features to triangulate
        """
        pass
