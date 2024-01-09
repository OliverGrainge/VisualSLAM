
from datasets import KittiSampleDataset
from visual_odometry import VisualOdometry
import utils
import numpy as np
import matplotlib.pyplot as plt
from evaluation import EvalVO

def main():
    config = utils.get_config()
    eval = EvalVO(VisualOdometry, KittiSampleDataset, config, debug=True)
    eval.eval()
    eval.plot_trajectory()
    print(eval.mean_tracking_error())


if __name__ == "__main__":
    main()