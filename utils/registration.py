import os
import copy
import open3d

import numpy as np
import utils.FCGF as FCGF
import utils.pointcloud as pointcloud

from scipy.signal import argrelmin
from PIL import Image


def estimation_method(p2p):
    if p2p:
        return open3d.registration.TransformationEstimationPointToPoint(False)  
    else: 
        return open3d.registration.TransformationEstimationPointToPlane()


def exec_icp(source, target, threshold, trans_init, max_iteration=30, p2p=True):
    return open3d.registration.registration_icp(
        source, target, threshold, trans_init, estimation_method(p2p),
        open3d.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    )


def exec_ransac(source, target, source_feat, target_feat, n_ransac, threshold, p2p=True):
    return open3d.registration.registration_ransac_based_on_feature_matching(
        source, target, source_feat, target_feat, threshold,
        estimation_method(p2p), n_ransac,
        [
            open3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            open3d.registration.CorrespondenceCheckerBasedOnDistance(threshold)
        ],
        open3d.registration.RANSACConvergenceCriteria(4000000, 600))


def compute_fpfh(pcd, voxel_size, down_sample=True, compute_normals=False):
    if down_sample:
        pcd = open3d.voxel_down_sample(pcd, voxel_size)
        
    if compute_normals: pointcloud.compute_normals(pcd, voxel_size)

    radius_feature = voxel_size * 5
    pcd_fpfh = open3d.registration.compute_fpfh_feature(
        pcd, open3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd, pcd_fpfh


def describe(source, target, reg_result, end="\n"):
    print(f"Keypts: [{len(source.points)}, {len(target.points)}]", end="\t")
    print(f"No of matches: {len(reg_result.correspondence_set)}", end="\t")
    print(f"Fitness: {reg_result.fitness:.4f}", end="\t")
    print(f"Inlier RMSE: {reg_result.inlier_rmse:.4f}", end=end)
    

def view(source, target, T):
    p1 = copy.deepcopy(source)
    p2 = copy.deepcopy(target)
    
    p1.paint_uniform_color([1, 0.706, 0])
    p2.paint_uniform_color([0, 0.651, 0.929])
    
    p1.transform(T)
    
    open3d.visualization.draw_geometries([p1, p2])
    
    
def find_cutoffs(std_values, target_fps, min_std, threshold):
    cutoffs = argrelmin(std_values, order=target_fps // 2)[0]
    return cutoffs[np.where(np.abs(std_values[cutoffs] - min_std) < threshold)[0]]


def get_cutoff_sequence(std_values, target_fps, min_std, threshold, cutoff_margin):
    cutoffs = find_cutoffs(std_values, target_fps, min_std, threshold)
    # add the first and last frame to the cutoffs
    cutoffs = np.concatenate([[0], cutoffs, [len(std_values) - 1]])
    # add a margin to the cutoffs
    cutoffs = [[cutoffs[i] + cutoff_margin, cutoffs[i + 1] - cutoff_margin] for i in range(len(cutoffs) - 1)]
    # check if the first frame in the last cutoff is the last frame in the sequence 
    cutoffs[-1][0] = min(len(std_values) - 1, cutoffs[-1][0])
    # remove invalid cutoffs
    cutoffs = [c for c in cutoffs if c[0] < c[1]]
    return cutoffs


def calc_std(depth_img_file, depth_scale):
    depth_img = Image.open(depth_img_file).convert("I")
    depth_img = np.array(depth_img) / depth_scale
    return np.std(depth_img)


def register_fragments_local(sequence_dir, sequence_ts, t, voxel_size):
    src_feature_file = os.path.join(sequence_dir, f"{sequence_ts[t]}.secondary.npz")
    tgt_feature_file = os.path.join(sequence_dir, f"{sequence_ts[t + 1]}.secondary.npz")

    source = FCGF.get_features(src_feature_file, pcd_only=True)
    target = FCGF.get_features(tgt_feature_file, pcd_only=True)
    
    source, source_fpfh = compute_fpfh(source, voxel_size, down_sample=False)
    target, target_fpfh = compute_fpfh(target, voxel_size, down_sample=False)
    
    source.paint_uniform_color(np.random.random(3).tolist())
    
    global_reg = exec_ransac(source, target, source_fpfh, target_fpfh, n_ransac=4, threshold=0.05)
    local_reg = exec_icp(source, target, threshold=0.05, trans_init=global_reg.transformation, max_iteration=30)
    
    return source, target, local_reg
