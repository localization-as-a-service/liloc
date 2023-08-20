import numpy as np
import pandas as pd
import os
import tqdm
import open3d

import utils.pointcloud as pointcloud
import utils.fread as fread
import utils.pointcloud as pointcloud
import utils.FCGF as FCGF

from utils.config import Config
from time import sleep



def local_trajectory(config: Config):
    output_file = config.get_output_file(f"{config.get_file_name()}.npz")
    
    if not os.path.exists(output_file):
        print("No local trajectory found for this sequence.")
        return
        
    data = np.load(output_file)
    
    sequence_ts = data["sequence_ts"]
    trajectory_t = data["trajectory_t"]
    local_t = data["local_t"]
    
    feature_dir = config.get_feature_dir()
    
    num_frames = len(sequence_ts)
    
    local_pcds = []
    
    for t in tqdm.trange(num_frames):
        if np.sum(local_t[t]) == 4:
            continue
        
        feature_file = os.path.join(feature_dir, f"{sequence_ts[t]}.secondary.npz")
        pcd = FCGF.get_features(feature_file, config.voxel_size, pcd_only=True)
        pcd.paint_uniform_color(pointcloud.random_color())
        pcd.transform(trajectory_t[t])
        local_pcds.append(pcd)
        
    trajectory_pcd = pointcloud.merge_pcds(local_pcds, config.voxel_size)
    pointcloud.view(trajectory_pcd)
    
    
def visualize_trajectory(config: Config):
    feature_dir = config.get_feature_dir()
    output_file_name = config.get_file_name()

    if not os.path.exists(config.get_output_file(f"{output_file_name}.npz")):
        print("No global trajectory found for this sequence.")
        return

    registrations = np.load(config.get_output_file(f"{output_file_name}.npz"))

    sequence_ts = registrations["sequence_ts"]
    global_t = registrations["global_t"]

    num_frames = len(sequence_ts)

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    global_pcd = FCGF.get_features(os.path.join(feature_dir, f"{sequence_ts[0]}.global.npz"), config.voxel_size, pcd_only=True)
    global_pcd.paint_uniform_color([0, 0.651, 0.929])

    local_pcd = FCGF.get_features(os.path.join(feature_dir, f"{sequence_ts[0]}.secondary.npz"), config.voxel_size, pcd_only=True)
    local_pcd.transform(global_t[0])
    local_pcd.paint_uniform_color([1, 0.706, 0])

    trajectory = [global_t[i][:3, 3].tolist() for i in range(num_frames) if np.sum(global_t[i]) != 4]
    lines = [[i, i + 1] for i in range(len(trajectory) - 1)]
    colors = [[1, 0, 0] for i in range(len(lines))]

    line_set = open3d.geometry.LineSet()

    vis.add_geometry(global_pcd)
    vis.add_geometry(local_pcd)
    vis.add_geometry(line_set)
    
    skipped_frames = 0

    for i in range(num_frames):
        if np.sum(global_t[i]) == 4: 
            skipped_frames += 1
            continue
        
        global_pcd_t = FCGF.get_features(os.path.join(feature_dir, f"{sequence_ts[i]}.global.npz"), config.voxel_size, pcd_only=True)
        global_pcd.points = global_pcd_t.points
        global_pcd.paint_uniform_color([0, 0.651, 0.929])
        
        local_pcd_t = FCGF.get_features(os.path.join(feature_dir, f"{sequence_ts[i]}.secondary.npz"), config.voxel_size, pcd_only=True)
        local_pcd_t.transform(global_t[i])
        
        local_pcd.points = local_pcd_t.points
        local_pcd.paint_uniform_color([1, 0.706, 0])
        
        line_set.points = open3d.utility.Vector3dVector(trajectory[:i+1])
        line_set.lines = open3d.utility.Vector2iVector(lines[:i - skipped_frames])
        line_set.colors = open3d.utility.Vector3dVector(colors[:i - skipped_frames])
        
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
        sleep(0.03)
        
    vis.destroy_window()

    
if __name__ == "__main__":
    config = Config(
        sequence_dir="data/raw_data",
        feature_dir="data/features",
        output_dir="data/trajectories/trajectory/FPFH_outlier_removed",
        experiment="exp_12",
        trial="trial_2",
        subject="subject-1",
        sequence="04",
        groundtruth_dir="data/trajectories/groundtruth",
    )
    
    config.voxel_size=0.05
    config.target_fps=20
    config.min_std=0.5
    
    visualize_trajectory(config)
    
    # for trial in os.listdir(os.path.join(config.feature_dir, config.experiment)):
    #     config.trial = trial
    #     for subject in os.listdir(os.path.join(config.feature_dir, config.experiment, config.trial, str(config.voxel_size))):
    #         config.subject = subject    
    #         for sequence in os.listdir(os.path.join(config.feature_dir, config.experiment, config.trial, str(config.voxel_size), config.subject)):
    #             config.sequence = sequence
    #             print(f"Processing: {config.experiment} >> {config.trial} >> {config.subject} >> {config.sequence}")
    #             visualize_trajectory(config)
    
    
    
        
    