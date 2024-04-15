from typing import Tuple

import cv2
import numpy as np
from PIL import Image


class SIFT:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.dtype = np.float32

    def __call__(self, image: Image.Image) -> Tuple:
        image.convert("L")
        image = np.array(image)
        kp, des = self.sift.detectAndCompute(image, None)
        return kp, des
