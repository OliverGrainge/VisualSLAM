from test import Evaluate

from datasets import KittiDataset
from odometry import StereoOdometry

if __name__ == "__main__":
    eval = Evaluate(StereoOdometry, KittiDataset, num_samples=750)
    eval.view_trajectories()
