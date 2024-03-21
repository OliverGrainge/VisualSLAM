import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import utils


class Evaluate:
    """
    A class to evaluate the performance of stereo odometry against a ground truth dataset.
    It processes a dataset with a given odometry algorithm and compares the estimated trajectories to the ground truth.

    Attributes:
        config (dict): Configuration settings obtained from a utility function.
        dataset: An instance of a dataset class initialized with the configuration settings.
        odometry: An instance of an odometry algorithm initialized with projection matrices and an initial pose.
        num_samples (int): The number of samples from the dataset to be evaluated.

    Parameters:
        odometry: The odometry class to be used for motion estimation.
        dataset: The dataset class to be used for evaluation.
        num_samples (int, optional): The number of samples to evaluate. If None, evaluates on the entire dataset.
    """
    def __init__(self, odometry, dataset, num_samples=None):
        """
        Initializes the evaluation process by setting up the dataset, odometry algorithm,
        and processing the specified number of images.
        """
        self.config = utils.get_config()
        self.dataset = dataset(self.config)
        left_proj, right_proj = self.dataset.projection_matrix()
        initial_pos = self.dataset.initial_pose()
        self.odometry = odometry(left_proj, right_proj, initial_pos)

        if num_samples is None:
            num_samples = len(self.dataset)
        for idx in tqdm(range(num_samples)):
            image_left, image_right = self.dataset.load_images(idx)
            self.odometry.process_images(image_left, image_right)
        self.num_samples = num_samples

    @staticmethod
    def translation_vector(pose: np.ndarray) -> np.ndarray:
        """
        Extracts the translation vector from a pose matrix.

        Parameters:
            pose (np.ndarray): A pose matrix from which the translation vector is extracted.

        Returns:
            np.ndarray: The translation vector of the pose.
        """
        return pose[:3, 3]

    def view_trajectories(self):
        """
        Visualizes the ground truth and estimated trajectories in a 3D plot. Additionally,
        it computes and displays the mean absolute error (MAE) between the ground truth and estimated trajectories.
        """
        gt = self.dataset.ground_truth()
        gt_pose = np.array([self.translation_vector(pose) for pose in gt])
        pose_graph = self.odometry.get_trajectory()
        tracked_pose = np.array([self.translation_vector(pose) for pose in pose_graph])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(
            gt_pose[: self.num_samples, 0],
            gt_pose[: self.num_samples, 1],
            gt_pose[: self.num_samples, 2],
            marker="o",
            label="Ground Truth",
            markersize=2,
        )

        ax.plot(
            tracked_pose[: self.num_samples, 0],
            tracked_pose[: self.num_samples, 1],
            tracked_pose[: self.num_samples, 2],
            marker="o",
            label="Tracked",
            markersize=2,
        )
        """
        all_pts = self.odometry.get_map()
        for points3d in all_pts: 
            idx = np.random.choice(points3d.shape[0], size=100, replace=True)
            points3d = points3d[idx]
            ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], alpha=0.2, s=10)
        """

        total_error = np.mean(
            np.abs(gt_pose[: self.num_samples] - tracked_pose[: self.num_samples])
        )

        # Set labels and legend
        ax.set_xlim(min(gt_pose[:self.num_samples, 0]), max(gt_pose[:self.num_samples, 0]))
        ax.set_ylim(min(gt_pose[:self.num_samples, 1]), max(gt_pose[:self.num_samples, 1]))
        ax.set_zlim(min(gt_pose[:self.num_samples, 2]), max(gt_pose[:self.num_samples, 2]))
        ax.set_title(f"MAE: {total_error:.2f}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()

        # Show plot
        plt.show()

    def tracking_mae(self):
        """
        Computes the Mean Absolute Error (MAE) between the ground truth and estimated trajectories.

        Returns:
            float: The mean absolute error of the tracking.
        """
        tracked_pose = np.array([self.translation_vector(pose) for pose in pose_graph])
        gt_pose = np.array([self.translation_vector(pose) for pose in gt])
        total_error = np.mean(
            np.abs(gt_pose[: self.num_samples] - tracked_pose[: self.num_samples])
        )
        return total_error
