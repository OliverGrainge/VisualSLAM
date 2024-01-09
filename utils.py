import yaml
import cv2
import numpy as np
import pandas as pd

def get_config() -> dict:
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)
    

def get_feature_extractor(config: dict):
    if config["feature_extractor"] == "orb":
        return cv2.ORB_create()
    elif config["feature_extractor"] == "sift":
        return cv2.SIFT_create()
    else: 
        raise NotImplementedError
    

def get_feature_matcher(config: dict): 
    if config["feature_matcher"] == "bf":
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        return matcher
    else: 
        raise NotImplementedError
    

def read_calib_file(filepath):
    # Dictionary to hold the data
    calib_data = {}

    # Read the file line by line
    with open(filepath, 'r') as file:
        for line in file:
            key, value = line.split(':', 1)
            # Convert the string of values into a numpy array
            calib_data[key] = np.fromstring(value, sep=' ')

    # Convert the dictionary into a pandas DataFrame for easier manipulation
    calib_df = pd.DataFrame.from_dict(calib_data, orient='index')
    return calib_df


def read_pose_file(file_path):
    # List to hold all pose matrices
    poses = []

    # Read the file line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into values and convert them to floats
            values = [float(x) for x in line.split()]

            # Reshape the values into a 3x4 matrix
            pose_matrix = np.array(values).reshape((3, 4))

            # Append the matrix to the list of poses
            poses.append(pose_matrix)

    return np.array(poses)
