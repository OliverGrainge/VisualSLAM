

from PIL import Image
from typing import Tuple
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class EvalVO: 
    def __init__(self, method, dataset, config, debug=False):
        if debug==True: 
            self.dataset = dataset(config)
            self.dataset.image_paths = self.dataset.image_paths[:300]
            self.dataset.poses = self.dataset.poses[:300]
        else:
            self.dataset = dataset(config)
        self.ground_truth = self.dataset.ground_truth()
        self.intrinsic_calib = self.dataset.intrinsic_calib()
        self.projection_calib = self.dataset.projection_calib()
        self.initial_pose = self.dataset.initial_pose()
    
        self.pbar = tqdm(total=self.dataset.__len__())
        self.vo = method(config=config,
                         intrinsic_calib=self.intrinsic_calib,
                         projection_calib=self.projection_calib,
                         initial_pose=self.initial_pose)
    
    def eval(self) -> None:
        for idx in range(self.dataset.__len__()):
            image = self.dataset.load_image(idx)
            self.vo.process_image(image)
            self.pbar.update(1)

    def plot_trajectory(self) -> None:
        # Extracting the translation components (x, y, z) from each matrix
        positions = [matrix[:3, 3] for matrix in self.vo.poses]
        positions = np.array(positions)  # Convert to numpy array for easier handling
        gt_positions = [matrix[:3, 3] for matrix in self.dataset.ground_truth()]
        gt_positions = np.array(gt_positions)

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plotting the path
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker='o', label="Estimated Path")
        ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], marker='x', label="Ground Truth")

        # Setting labels
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

        plt.title("3D Pose Path")
        plt.legend()
        plt.show()


    def mean_tracking_error(self) -> float:
        positions = [matrix[:3, 3] for matrix in self.vo.poses]
        positions = np.array(positions)  # Convert to numpy array for easier handling
        gt_positions = [matrix[:3, 3] for matrix in self.ground_truth]
        gt_positions = np.array(gt_positions)
        differences = positions - gt_positions 
        errors = np.linalg.norm(differences, axis=1)
        mean_error = np.mean(errors)
        return mean_error

        