import numpy as np
import cv2

np.set_printoptions(precision=3, suppress=True)
K = np.array([[718.856, 0.0, 607.1928], [0.0, 718.856, 185.2157], [0.0, 0.0, 1.0]])


# the initial pose
initial_pose = np.eye(4)

transformation = np.eye(4)
transformation[0, 3] = 1.0
next_pose = initial_pose @ np.linalg.inv(transformation)
# the next pose
# next_pose = np.copy(initial_pose)
# next_pose[0, 3] = -1.0

# the 3d point
point3d = np.array([[0.1, 0.1, 0.1]])


# 1. project the 3d point into the initial pose.
proj = K @ initial_pose[:3, :]
point2d = (proj @ np.hstack((point3d, np.array([[1]]))).T).T
point2d = point2d[:, :2] / point2d[:, 2].reshape(-1, 1)

# 2. translate the points
tpoint3d = (transformation @ np.hstack((point3d, np.array([[1]]))).T).T
tpoint3d = tpoint3d[:, :3] / tpoint3d[:, 3].reshape(-1, 1)

# 3. project the translated point into next pose camera
tproj = K @ next_pose[:3, :]
tpoint2d = (tproj @ np.hstack((tpoint3d, np.array([[1]]))).T).T
tpoint2d = tpoint2d[:, :2] / tpoint2d[:, 2].reshape(-1, 1)

print(tpoint2d, point2d)
