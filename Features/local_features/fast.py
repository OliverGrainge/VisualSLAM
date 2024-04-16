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


class FASTBlocks:
    def __init__(self):
        self.fast = cv2.FastFeatureDetector_create()
        self.dtype = np.uint8

    def __call__(self, image: Image.Image) -> Tuple:
        image.convert("L")
        image = np.array(image)
        height, width = image.shape
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
                keypoints, descriptors = self.fast.detectAndCompute(image, mask)
                
                keypoints_all.extend(keypoints)
                if descriptors is not None:
                    if descriptors_all == []:
                        descriptors_all = descriptors
                    else:
                        descriptors_all = np.vstack((descriptors_all, descriptors))

        return keypoints_all, descriptors_all