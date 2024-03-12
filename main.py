from datasets import KittiDataset
from odometry import StereoOdometry
from test import Evaluate


if __name__ == "__main__":
    eval = Evaluate(StereoOdometry, KittiDataset)
    eval.view_trajectories()
