import cv2 
import numpy as np 
from typing import Tuple
from PIL import Image


class ORB: 
    def __init__(self):
        self.orb = cv2.ORB_create()

    def __call__(self, image: Image.Image) -> Tuple:
        image.convert('L')
        image = np.array(image)
        kp, des = self.orb.detectAndCompute(image, None)
        return kp, des