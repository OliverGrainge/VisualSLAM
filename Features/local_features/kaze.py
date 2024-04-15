from typing import Tuple

import cv2
import numpy as np
from PIL import Image


class KAZE:
    def __init__(self):
        self.kaze = cv2.KAZE_create()
        self.dtype = np.float32

    def __call__(self, image: Image.Image) -> Tuple:
        image.convert("L")
        image = np.array(image)
        kp, des = self.kaze.detectAndCompute(image, None)
        return kp, des