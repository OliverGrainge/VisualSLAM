from typing import List

import cv2
import numpy as np
from typing import Union, Tuple, List


class DenseMatcher:
    def __init__(self, lowes_ratio=0.75):
        self.bf = cv2.BFMatcher()
        self.lowes_ratio = lowes_ratio

    def __call__(self, **kwargs) -> Union[Tuple, List]:
        if "des1" not in kwargs.keys() or "des2" not in kwargs.keys():
            raise Exception(
                "DenseyMatcher.__call__ must have des1 and des2 and arguments"
            )
        matches = self.bf.knnMatch(kwargs["des1"], kwargs["des2"], k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < self.lowes_ratio * n.distance:
                good_matches.append(m)
        good_matches = sorted(good_matches, key=lambda x: x.distance)
        if "kp1" in kwargs.keys() and "kp2" in kwargs.keys():
            kp1 = np.float32([kwargs["kp1"][m.queryIdx].pt for m in good_matches]).T
            kp2 = np.float32([kwargs["kp2"][m.trainIdx].pt for m in good_matches]).T
            des1 = np.float32([kwargs["des1"][m.queryIdx] for m in good_matches]).T
            des2 = np.float32([kwargs["des2"][m.trainIdx] for m in good_matches]).T
            return kp1, des1, kp2, des2
        else:
            return good_matches
