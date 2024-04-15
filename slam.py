import numpy as np 
from Backend import PoseGraphOptimization, BundleAdjustment
from LoopClosure import EigenPlaces
from Mapping import PointCloud
from KeyFrame import KeyFrameInsertion
from MotionEstimation import threeDtwoD


class VSLAM: 
    def __init__(self, **args):
        self.poses = []
        self.loop_closures = []

        self.keyframe_insertion = KeyFrameInsertion(self.poses)
        self.loop_closure = EigenPlaces(self.poses)
        self.bundle_adjustment = BundleAdjustment(self.poses, self.loop_closures) 
        self.pose_graph_optimization = PoseGraphOptimization(self.poses, self.loop_closures)
        self.motion_estimation = threeDtwoD
        self.map = PointCloud(self.poses)


    def __call__(self, **kwargs) -> None:
        keyframe = self.keyframe_insertion(**kwargs)
        if keyframe:
            self.motion_estimation()
            self.pose_graph_optimization(local=True)
            self.bundle_adjustment(local=True)
            
            loop_detection = self.loop_closure(keyframe)
            self.bundle_adjustment(loop_detection)
            #self.pose_graph_optimization(closure)
