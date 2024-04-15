import numpy as np 
from typing import List, Union 



class BundleAdjustment: 
    def __init__(self, poses: List, window: int=5) -> None:
        self.poses = poses 
        self.window = window


    def optimize(global_opt: bool=False, window: Union[None, int]=None) -> None:
        """
        performs either global or local optimization on the 
        poses. 
        """
        pass 