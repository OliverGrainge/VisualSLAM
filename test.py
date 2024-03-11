import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils


class Evaluate:
    def __init__(self, odometry, dataset, num_samples=None):
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
        return pose[:3, 3]

    def view_trajectories(self):
        gt = self.dataset.ground_truth()
        gt_pose = np.array([self.translation_vector(pose) for pose in gt])
        tracked_pose = np.array(
            [self.translation_vector(pose) for pose in self.odometry.poses]
        )

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

        # Set labels and legend
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()

        # Show plot
        plt.show()

    def trajectory_error(self):
        pass
