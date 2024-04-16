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



class KAZEBlocks:
    def __init__(self, grid_size=(6, 8)):
        self.kaze = cv2.KAZE_create()
        self.grid_size = grid_size
        self.dtype = np.float32

    def __call__(self, image: Image.Image) -> Tuple:
        image.convert("L")
        image = np.array(image)
        height, width, channels = image.shape
        block_height = height // self.grid_size[0]
        block_width = width // self.grid_size[1]

        keypoints_all = []
        descriptors_all = []

        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                start_x = j * block_width
                start_y = i * block_height
                end_x = start_x + block_width
                end_y = start_y + block_height
                if i == self.grid_size[0] - 1:
                    end_y = height
                if j == self.grid_size[1] - 1:
                    end_x = width

                mask = np.zeros((height, width), dtype=np.uint8)
                mask[start_y:end_y, start_x:end_x] = 255
                keypoints, descriptors = self.kaze.detectAndCompute(image, mask)
                keypoints_all.extend(keypoints)
                if descriptors is not None:
                    if len(descriptors_all) == 0:
                        descriptors_all = descriptors
                    else:
                        descriptors_all = np.vstack((descriptors_all, descriptors))

        return keypoints_all, descriptors_all