from typing import Tuple

import cv2
import numpy as np
from PIL import Image


class SIFT:
    def __init__(self):
        self.feature_detector = cv2.SIFT_create()

    def __call__(self, image: Image.Image) -> Tuple:
        image = np.array(image)
        if image.ndim > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        (
            left_points2d_pose,
            left_points2d_desc,
        ) = self.feature_detector.detectAndCompute(gray, None)

        return left_points2d_pose, left_points2d_desc
