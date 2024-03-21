from test import Evaluate

from datasets import KittiDataset
from odometry import StereoOdometry

if __name__ == "__main__":
    eval = Evaluate(StereoOdometry, KittiDataset, num_samples=None)
    eval.view_trajectories()
