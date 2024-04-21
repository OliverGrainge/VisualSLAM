from typing import List, Union, Dict

import numpy as np


class PointCloud:
    def __init__(self, points: List, map: Dict, matching_threshold=250, window: int=2, max_map_size: int=400) -> None:
        self.points = points
        self.map = map
        self.window = window
        self.max_map_size = max_map_size
        self.id = 0

    def __call__(self):
        
        if len(self.map["local_points"]) == 0:
            self.map["local_points"] = self.points[-1].keypoints_3d
            self.map["local_descriptors"] = self.points[-1].descriptors_3d
            self.map["local_ids"] = np.zeros(len(self.points[-1].keypoints_3d))
            self.id += 1

        else:
            if len(self.map["local_points"]) == 1:
                matches = self.points[-1].feature_matcher(
                    des1=self.points[-1].descriptors_3d,
                    des2=self.map["local_descriptors"],
                    apply_lowe=False
                )[:int(self.max_map_size/self.window)]
            else:
                matches = self.points[-1].feature_matcher(
                    des1=self.points[-1].descriptors_3d,
                    des2=self.map["local_descriptors"],
                    apply_lowe=False
                )[:int(self.max_map_size/self.window)]

            points_to_add = []
            descriptors_to_add = []
            for match in matches:
                if match.distance > 100:
                    points_to_add.append(self.points[-1].keypoints_3d[match.queryIdx])
                    descriptors_to_add.append(self.points[-1].descriptors_3d[match.queryIdx])

            if len(descriptors_to_add) > 0:
                points_to_add = np.vstack(points_to_add)
                points_to_add = self.points[-1].transform_points3d(points_to_add, self.points[-1].x)
                self.map["local_descriptors"] = np.vstack((self.map["local_descriptors"], np.vstack(descriptors_to_add)))
                self.map["local_points"] = np.vstack((self.map["local_points"], points_to_add))
                self.map["local_ids"] = np.concatenate((self.map["local_ids"], (np.zeros(len(descriptors_to_add)) + self.id)))
                self.id += 1
                
            if self.id > self.window:
                idx = np.argmax(self.map["local_ids"]==1)
                self.map["local_descriptors"] = self.map["local_descriptors"][idx:]
                self.map["local_points"] = self.map["local_points"][idx:]
                self.map["local_ids"] = self.map["local_ids"][idx:]
                self.map["local_ids"] = self.map["local_ids"] - 1
                self.id -= 1
        
        assert self.map["local_descriptors"].shape[0] == self.map["local_points"].shape[0]
            

