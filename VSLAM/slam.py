import numpy as np

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

        self.keyframe_insertion = KeyFrameInsertion(self.points)
        #self.loop_closure = EigenPlaces(self.poses)
        #self.bundle_adjustment = BundleAdjustment(self.poses, self.loop_closures)
        #self.pose_graph_optimization = PoseGraphOptimization(
        #    self.poses, self.loop_closures
        #)
        self.motion_estimation = MotionEstimation3D2D(self.points, self.transformations)
        #self.map = PointCloud(self.poses)

    def __call__(self, **kwargs) -> None:
        keyframe = self.keyframe_insertion(**kwargs)
        if keyframe:
            self.motion_estimation()
        #    self.pose_graph_optimization(local=True)
        #    self.bundle_adjustment(local=True)

        #    loop_detection = self.loop_closure(keyframe)
         #   self.bundle_adjustment(loop_detection)
            # self.pose_graph_optimization(closure)

    def get_trajectory(self):
        return np.array([pt.x[:3, 3] for pt in self.points])
