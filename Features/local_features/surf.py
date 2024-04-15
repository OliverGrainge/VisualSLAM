import cv2 
import numpy as np 
from typing import Tuple
from PIL import Image


class SURF: 
    def __init__(self):
        self.surf = cv2.xfeatures2d.SURF_create()

    def __call__(self, image: Image.Image) -> Tuple:
        image.convert('L')
        image = np.array(image)
        kp, des = self.surf.detectAndCompute(image, None)
        return kp, des