from typing import Tuple

import cv2
import numpy as np
from PIL import Image


class AKAZE:
    def __init__(self):
        self.akaze = cv2.AKAZE_create()

    def __call__(self, image: Image.Image) -> Tuple:
        image.convert("L")
        image = np.array(image)
        kp, des = self.akaze.detectAndCompute(image, None)
        return kp, des
