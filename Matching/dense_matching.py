from typing import List

import cv2
import numpy as np


class DenseMatcher:
    def __init__(self, lowes_ratio=0.75):
        self.bf = cv2.BFMatcher()
        self.lowes_ratio = lowes_ratio

    def __call__(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        matches = self.bf.knnMatch(desc1, desc2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < self.lowes_ratio * n.distance:
                good_matches.append(m)
        good_matches = sorted(good_matches, key= lambda x: x.distance)
        return good_matches
