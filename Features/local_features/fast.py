from typing import Tuple

import cv2
import numpy as np
from PIL import Image


class FAST:
    def __init__(self):
        self.fast = cv2.FastFeatureDetector_create()
        self.dtype = np.uint8

    def __call__(self, image: Image.Image) -> Tuple:
        image.convert("L")
        image = np.array(image)
        kp, des = self.fast.detectAndCompute(image, None)
        return kp, des
