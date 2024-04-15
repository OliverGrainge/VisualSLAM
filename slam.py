import numpy as np 


class VSLAM: 
    def __init__(self, **args):
        self.keyframe_insertion = None
        self.loop_closure = None 
        self.bundle_adjustment = None 
        self.pose_graph_optimization = None 


    def __call__: