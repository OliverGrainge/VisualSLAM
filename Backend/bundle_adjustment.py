import numpy as np 
from typing import List, Union 



class BundleAdjustment: 
    def __init__(self, poses: List, loop_closures: List, window: int=5) -> None:
        self.poses = poses 
        self.loop_closures = loop_closures
        self.window = window


    def optimize(loop_detection: Union[np.ndarray, bool]=False, window: Union[None, int]=None) -> None:
        """
        performs
        """
        pass 