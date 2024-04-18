from typing import List, Union

import numpy as np
from PIL import Image


class EigenPlaces:
    def __init__(self, poses: List):
        self.poses = poses
        self.descriptor_map = None

    def __call__(self) -> Union[None, int]:
        """
        Takes the last entry into poses and finds a place match
        """
