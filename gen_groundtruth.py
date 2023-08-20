import open3d
import numpy as np
import pandas as pd
import os
import tqdm
import copy
import glob

from scipy.signal import argrelmin
from PIL import Image

import utils.registration as registration
import utils.grid_search_unopt as grid_search
import utils.fread as fread
import utils.FCGF as FCGF
import utils.registration as registration
import utils.functions as functions
import utils.transform as transform
import utils.pointcloud as pointcloud


def register_frames(dataset_dir, experiment, trial, subject, sequence, voxel_size, out_dir):
    sequence_dir = os.path.join(dataset_dir, experiment, trial, str(voxel_size), subject, sequence)
    sequence_ts = fread.get_timstamps(sequence_dir, ext=".secondary.npz")
    num_frames = len(sequence_ts)
    
    file_name = f"{experiment}__{trial}__{subject}__{sequence}"
    
    if os.path.exists(os.path.join(out_dir, f"{file_name}.pose.npz")):
        return
    
    print("     :: Number of frames: {}".format(num_frames))
    
    local_pcds = []
    local_feat = []
    
    print("     :: Caching local PCDs and features.")

    for t in tqdm.trange(num_frames):
        feature_file = os.path.join(sequence_dir, f"{sequence_ts[t]}.secondary.npz")
        pcd, features, _ = FCGF.get_features(feature_file, voxel_size)
        local_pcds.append(pcd)
        local_feat.append(features)
    
    print("     :: Registering local PCDs.")
    
    local_t = [np.identity(4)]

    for t in tqdm.trange(num_frames - 1):
        source, source_feat = copy.deepcopy(local_pcds[t + 1]), local_feat[t + 1]
        target, target_feat = copy.deepcopy(local_pcds[t]), local_feat[t]

        ransac_reg = registration.exec_ransac(source, target, source_feat, target_feat, n_ransac=3, threshold=0.05)
        icp_reg = registration.exec_icp(source, target, threshold=0.05, trans_init=ransac_reg.transformation, max_iteration=200)
        
        local_t.append(icp_reg.transformation)
        
    print("     :: Calculating transformations.")
    
    trajectory_t = [np.identity(4)]

    for t in tqdm.trange(1, num_frames):
        trajectory_t.append(np.dot(trajectory_t[t - 1], local_t[t]))

    print("     :: Applying transformations.")
        
    for i in tqdm.trange(num_frames):
        local_pcds[i].transform(trajectory_t[i])
    
    trajectory = pointcloud.merge_pcds(local_pcds, voxel_size)
    trajectory = pointcloud.remove_outliers(trajectory)

    print("     :: Saving trajectory.")
    
    # open3d.visualization.draw_geometries([trajectory])
    open3d.io.write_point_cloud(os.path.join(out_dir, f"{file_name}.trajectory.pcd"), trajectory)
    np.savez(os.path.join(out_dir, f"{file_name}.pose.npz"), local_t=local_t)
    
    print("     :: Done.")


def find_candidate_global_pos(std_values, delta):
    global_pos = [0]
    prev_t = 0
    for current_t in range(len(std_values)):
        if np.abs(std_values[current_t] - std_values[prev_t]) > delta:
            global_pos.append(current_t)
            prev_t = current_t
            
    return global_pos


def find_cutoffs(std_values, target_fps, min_std, threshold):
    cutoffs = argrelmin(std_values, order=target_fps // 2)[0]
    return cutoffs[np.where(np.abs(std_values[cutoffs] - min_std) < threshold)[0]]
    
    
def create_fragments(dataset_dir, experiment, trial, subject, sequence, voxel_size, out_dir):
    min_std = 0.5
    threshold = 0.5
    target_fps = 20
    cutoff_margin = 5 # frames
    
    sequence_dir = f"data/raw_data/{experiment}/{trial}/secondary/{subject}/{sequence}/frames"
    feature_dir = os.path.join("data/features", experiment, trial, str(voxel_size), subject, sequence)

    sequence_ts = fread.get_timstamps(feature_dir, ext=".secondary.npz")
    num_frames = len(sequence_ts)

    file_name = f"{experiment}__{trial}__{subject}__{sequence}"
    
    # sequence_ts = fread.get_timstamps_from_images(sequence_dir, ext=".depth.png")
    # sequence_ts = helpers.sample_timestamps(sequence_ts, target_fps)
    # num_frames = len(sequence_ts)
    
    print("     :: Caclulating Std. of frames.")
    
    std_values = []

    for t in tqdm.trange(len(sequence_ts)):
        depth_img = Image.open(os.path.join(sequence_dir, f"frame-{sequence_ts[t]}.depth.png")).convert("I")
        depth_img = np.array(depth_img) / 4000
        std_values.append(np.std(depth_img))
        
    std_values = np.array(std_values)

    # global_pos = find_candidate_global_pos(std_values, delta=0.2)
    
    print("     :: Caclulating cut-off frames.")
    
    cutoffs = find_cutoffs(std_values, target_fps, min_std, threshold)
    cutoffs = np.concatenate([[0], cutoffs, [num_frames - 1]])
    cutoffs = list(zip(cutoffs[:-1] + cutoff_margin, cutoffs[1:] - cutoff_margin))
    
    if not os.path.exists(os.path.join(f"data/trajectories/groundtruth/{experiment}", f"{file_name}.pose.npz")):
        print("File not found!")
        return
        
    local_t = np.load(os.path.join(f"data/trajectories/groundtruth/{experiment}", f"{file_name}.pose.npz"))["local_t"]
    
    print(f"     :: Caching {num_frames} local PCDs.")
    
    local_pcds = []

    for t in tqdm.trange(num_frames):
        feature_file = os.path.join(feature_dir, f"{sequence_ts[t]}.secondary.npz")
        pcd, _, _ = FCGF.get_features(feature_file, voxel_size)
        local_pcds.append(pcd)
        
    print("     :: Making fragments.")
        
    fragments = []
    for start_t, end_t in tqdm.tqdm(cutoffs):
        trajectory_t = [np.identity(4)]

        for t in range(start_t + 1, end_t):
            trajectory_t.append(np.dot(trajectory_t[t - start_t - 1], local_t[t]))
        
        fragment = []
        for t in range(start_t, end_t):
            local_temp = copy.deepcopy(local_pcds[t])
            local_temp.transform(trajectory_t[t - start_t])
            fragment.append(local_temp)
            
        fragment = pointcloud.merge_pcds(fragment, 0.03)
        fragments.append(fragment)
    
    print("     :: Saving fragments.")
    
    fragments_dir = "data/fragments"
    
    if not os.path.exists(os.path.join(fragments_dir, experiment)):
        os.makedirs(os.path.join(fragments_dir, experiment))
        
    for i, fragment in enumerate(fragments):
        open3d.io.write_point_cloud(os.path.join(fragments_dir, experiment, f"{file_name}__{i:02d}.pcd"), fragment)


def register_fragments(dataset_dir, experiment, trial, subject, sequence, voxel_size, out_dir):
    file_name = f"{experiment}__{trial}__{subject}__{sequence}"
    
    fragment_files = glob.glob(os.path.join(f"data/fragments/{experiment}", f"{file_name}__*.npz"))
    fragment_files = sorted(fragment_files, key=lambda f: int(os.path.basename(f).split(".")[0].split("__")[-1]))
    
    fragments = []
    fragment_t = []

    for i in tqdm.trange(len(fragment_files)):
        fragment_pcd, global_pcd, refine_reg = grid_search.global_registration(
            src_feature_file=fragment_files[i],
            tgt_feature_file="data/reference/larc_kitchen_v5.npz",
            cell_size=4,
            voxel_size=0.03,
            refine_enabled=True
        )
        
        fragments.append(fragment_pcd)
        fragment_t.append(refine_reg.transformation if refine_reg else np.identity(4))
    
    np.savez(os.path.join(out_dir, f"{file_name}.fragment.pose.npz"), fragment_t=fragment_t)
    
    
def seperate_poses(dataset_dir, experiment, trial, subject, sequence, voxel_size, out_dir):
    file_name = f"{experiment}__{trial}__{subject}__{sequence}"
    
    fragment_t = np.load(os.path.join(f"data/trajectories/groundtruth/{experiment}", f"{file_name}.fragment.pose.npz"))["fragment_t"]
    fragment_files = glob.glob(os.path.join(f"data/fragments/{experiment}", f"{file_name}__*.npz"))
    fragment_files = sorted(fragment_files, key=lambda f: int(os.path.basename(f).split(".")[0].split("__")[-1]))

    for i in range(len(fragment_t)):
        np.savetxt(os.path.normpath(fragment_files[i].replace("npz", "txt")), fragment_t[i])
        

def disassemble_fragments(dataset_dir, experiment, trial, subject, sequence, voxel_size, out_dir):
    min_std = 0.5
    threshold = 0.5
    target_fps = 20
    cutoff_margin = 5 # frames
    
    sequence_dir = f"data/raw_data/{experiment}/{trial}/secondary/{subject}/{sequence}/frames"
    feature_dir = os.path.join("data/features", experiment, trial, str(voxel_size), subject, sequence)

    sequence_ts = fread.get_timstamps(feature_dir, ext=".secondary.npz")
    num_frames = len(sequence_ts)

    file_name = f"{experiment}__{trial}__{subject}__{sequence}"
    
    if os.path.exists(os.path.join(out_dir, f"{file_name}.gtpose.npz")):
        return
    
    print("     :: Caclulating Std. of frames.")
    
    std_values = []

    for t in range(len(sequence_ts)):
        depth_img = Image.open(os.path.join(sequence_dir, f"frame-{sequence_ts[t]}.depth.png")).convert("I")
        depth_img = np.array(depth_img) / 4000
        std_values.append(np.std(depth_img))
        
    std_values = np.array(std_values)
    
    print("     :: Caclulating cut-off frames.")
    
    cutoffs = find_cutoffs(std_values, target_fps, min_std, threshold)
    cutoffs = np.concatenate([[0], cutoffs, [num_frames - 1]])
    cutoffs = list(zip(cutoffs[:-1] + cutoff_margin, cutoffs[1:] - cutoff_margin))
    
    if not os.path.exists(os.path.join(f"data/trajectories/groundtruth/{experiment}", f"{file_name}.pose.npz")):
        print("File not found!")
        return
    
    print("     :: Converting fragment transformations to global.")
        
    local_t = np.load(os.path.join(f"data/trajectories/groundtruth/{experiment}", f"{file_name}.pose.npz"))["local_t"]
    
    fragment_files = glob.glob(os.path.join(f"data/fragments/{experiment}", f"{file_name}__*.pcd"))
    fragment_files = sorted(fragment_files, key=lambda f: int(os.path.basename(f).split(".")[0].split("__")[-1]))

    fragment_t = [np.loadtxt(fragment_files[i].replace("pcd", "txt")) for i in range(len(fragment_files)) if os.path.exists(fragment_files[i].replace("pcd", "txt"))]
    
    trajectory_t = [np.identity(4) for _ in range(num_frames)]
    global_t = [np.zeros((4, 4)) for _ in range(num_frames)]
    
    if len(fragment_t) < len(cutoffs):
        fragment_ids = [int(os.path.basename(f).split(".")[0].split("__")[-1]) for f in fragment_files]
        cutoffs = [cutoffs[fi] for fi in fragment_ids]
    
    for fragment_ind, (start_t, end_t) in enumerate(cutoffs):
        for t in range(start_t + 1, end_t):
            trajectory_t[t] = np.dot(trajectory_t[t - 1], local_t[t])
            
        for t in range(start_t, end_t):
            global_t[t] = np.dot(fragment_t[fragment_ind], trajectory_t[t])
    
    np.savez(os.path.join(out_dir, f"{file_name}.gtpose.npz"), local_t=global_t)
    

def validate_groundtruth(dataset_dir, experiment, trial, subject, sequence, voxel_size, out_dir):
    sequence_dir = os.path.join(dataset_dir, experiment, trial, str(voxel_size), subject, sequence)

    sequence_ts = fread.get_timstamps(sequence_dir, ext=".secondary.npz")
    num_frames = len(sequence_ts)

    file_name = f"{experiment}__{trial}__{subject}__{sequence}"
    try:
        local_t = np.load(os.path.join(out_dir, f"{file_name}.gtpose.npz"))["local_t"]
        target = open3d.io.read_point_cloud("data/reference/larc_kitchen_v5.pcd")
        
        refined_pcds = []

        for t in range(num_frames):
            feature_file = os.path.join(sequence_dir, f"{sequence_ts[t]}.secondary.npz")
            source = FCGF.get_features(feature_file, voxel_size, pcd_only=True)
            source.paint_uniform_color(pointcloud.random_color())
            source.transform(local_t[t])
            refined_pcds.append(source)

        source = pointcloud.merge_pcds(refined_pcds, voxel_size)

        # open3d.visualization.draw_geometries([source])

        source.paint_uniform_color([1, 0.706, 0])
        target.paint_uniform_color([0, 0.651, 0.929])

        open3d.visualization.draw_geometries([source, target])
        
    except FileNotFoundError:
        print("Unabel to find refined transformations.")
        return       


if __name__ == "__main__":
    VOXEL_SIZE = 0.05
    ROOT_DIR = "data/features"
    EXPERIMENT = "exp_12"
    OUT_DIR = f"data/trajectories/groundtruth/{EXPERIMENT}"

    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    
    for trial in os.listdir(os.path.join(ROOT_DIR, EXPERIMENT)):
        for subject in os.listdir(os.path.join(ROOT_DIR, EXPERIMENT, trial, str(VOXEL_SIZE))):
            for sequence in os.listdir(os.path.join(ROOT_DIR, EXPERIMENT, trial, str(VOXEL_SIZE), subject)):
                print(f"Processing: {EXPERIMENT} >> {trial} >> {subject} >> {sequence}")
                disassemble_fragments(ROOT_DIR, EXPERIMENT, trial, subject, sequence, VOXEL_SIZE, OUT_DIR)
    
    
    # pcd_files = glob.glob(os.path.join(OUT_DIR, "*.pcd"))
    # for pcd_file in pcd_files:
    #     open3d.visualization.draw_geometries([open3d.io.read_point_cloud(pcd_file)])
    