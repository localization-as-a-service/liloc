import open3d
import numpy as np
import utils.pointcloud as pointcloud
import tqdm
import os


'''
    Reads the output of the FCGF algorithm returns the keypoints and the features.
    pcd_only: if True, only the keypoints are returned
'''
def get_features(feature_file, voxel_size, pcd_only=False):
    data = np.load(feature_file)
    pcd = pointcloud.make_pcd(data["keypts"])
    pointcloud.compute_normals(pcd, voxel_size)
    
    if pcd_only:
        return pcd
    
    scores = data["scores"]
    features = open3d.registration.Feature()
    features.data = data["features"].T
    
    return pcd, features, scores