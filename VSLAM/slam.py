import numpy as np
from collections import defaultdict

from VSLAM.Backend import PoseGraphOptimization, BundleAdjustment
from VSLAM.KeyFrame import KeyFrameInsertion
from VSLAM.LoopClosure import EigenPlaces
from VSLAM.Mapping import PointCloud
from VSLAM.MotionEstimation import MotionEstimation3D3D, MotionEstimation3D2D


class VSLAM:
    def __init__(self):
        self.points = []
        self.loop_closures = []
        self.transformations = []
        self.map = defaultdict(list)

        self.keyframe_insertion = KeyFrameInsertion(self.points)
        #self.loop_closure = EigenPlaces(self.poses)
        self.bundle_adjustment = BundleAdjustment(self.points, self.map, self.loop_closures)
        self.pose_graph_optimization = PoseGraphOptimization(
            self.points, self.transformations, self.loop_closures
        )
        self.motion_estimation = MotionEstimation3D2D(self.points, self.transformations)
        self.mapping = PointCloud(self.points, self.map)

    def __call__(self, **kwargs) -> None:
        keyframe = self.keyframe_insertion(**kwargs)
        if keyframe:
            self.motion_estimation()
            self.mapping()
            #self.pose_graph_optimization()
            self.bundle_adjustment()

        #    loop_detection = self.loop_closure(keyframe)
         #   self.bundle_adjustment(loop_detection)
            # self.pose_graph_optimization(closure)

    def get_trajectory(self):
        return np.array([pt.x[:3, 3] for pt in self.points])
