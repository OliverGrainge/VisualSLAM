from PIL import Image
import numpy as np
from typing import List

class RgbdPoint: 
    def __init__(self, image: Image.Image, depth_map: Image.Image):
        self.image = image
        self.depth_map = depth_map
 
        # 2d features
        self.keypoints_2d = None 
        self.descriptors_2d = None
        self.global_descriptor_2d = None

        #2d features
        self.keypoints_3d = None 
        self.descriptors_3d = None
    

    def compute_features2d(self):
        """
        Computes the feature points in 2d
        """
        pass
    
    def compute_features3d(self):
        """
        uses the 2d features to triangulate
        """
        pass 