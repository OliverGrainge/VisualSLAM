import numpy as np 
from typing import List, Union 



class PointCloud: 
    def __init__(self, poses: List) -> None:
        self.poses = poses


    def view(self):
        """
        renders the point cloud
        """
        pass 