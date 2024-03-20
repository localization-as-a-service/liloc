import open3d
import numpy as np
import pandas as pd
import os
import glob
import tqdm
import copy
import time

import utils.registration as registration
import utils.functions as functions
import utils.transform as transform
import utils.pointcloud as pointcloud
import utils.fread as fread
import utils.FCGF as FCGF

from utils.config import Config
from scipy.ndimage import gaussian_filter1d
from PIL import Image


def fpfh_local_registration(config: Config):
    output_file = config.get_output_file(config.get_file_name() + ".npz")
    
    if os.path.exists(output_file):
        print(f"-- File {output_file} already exists. Skipping.")
        return
    
    feature_dir = config.get_feature_dir()

    sequence_ts = fread.get_timstamps(feature_dir, ext=".secondary.npz")
    sequence_ts = fread.sample_timestamps(sequence_ts, config.target_fps)
    
    # elapsed_ts = sequence_ts - sequence_ts[0]
    # startt_idx = np.argwhere(elapsed_ts >= 4000)[0][0]
    # sequence_ts = sequence_ts[startt_idx:]
    num_frames = len(sequence_ts)
    print(f"-- Number of frames: {num_frames}")
    
    print("-- Caching local PCDs and features.")
    
    local_pcds = []
    fpfh_feats = []
    
    # start_time = time.time()

    for t in tqdm.trange(num_frames):
        feature_file = os.path.join(feature_dir, f"{sequence_ts[t]}.secondary.npz")
        pcd = FCGF.get_features(feature_file, config.voxel_size, pcd_only=True)
        pcd, fpfh = registration.compute_fpfh(pcd, config.voxel_size, down_sample=False)
        local_pcds.append(pcd)
        fpfh_feats.append(fpfh) 
        
    # end_time = time.time()
    # print(f"-- Caching took {end_time - start_time} seconds.")
    # print(f"-- Average time per frame: {(end_time - start_time) / num_frames} seconds.")
    # print(f"-- Itr/second: {num_frames / (end_time - start_time)}")
        
    print("-- Registering local PCDs.")
    
    local_t = [np.identity(4)]
    # start_time = time.time()
    
    for t in tqdm.trange(num_frames - 1):
        source, source_fpfh = copy.deepcopy(local_pcds[t + 1]), fpfh_feats[t + 1]
        target, target_fpfh = copy.deepcopy(local_pcds[t]), fpfh_feats[t]

        reg_result = registration.exec_ransac(source, target, source_fpfh, target_fpfh, n_ransac=4, threshold=0.05)
        reg_result = registration.exec_icp(source, target, threshold=0.05, trans_init=reg_result.transformation, max_iteration=200, p2p=False)

        local_t.append(reg_result.transformation)
    
    # end_time = time.time()
    # print(f"-- Caching took {end_time - start_time} seconds.")
    # print(f"-- Average time per frame: {(end_time - start_time) / num_frames} seconds.")
    # print(f"-- Itr/second: {num_frames / (end_time - start_time)}")
    
    print("-- Refining Trajectory.")
    
    trajectory_t = [np.identity(4)]

    for t in tqdm.trange(1, num_frames):
        trajectory_t.append(np.dot(trajectory_t[t - 1], local_t[t]))
        
    print("-- Saving Trajectory.")
    
    np.savez_compressed(output_file, sequence_ts=sequence_ts, local_t=local_t, trajectory_t=trajectory_t)
        
    
def imu_pcd_fused_registration(config: Config, calib_period: int = 4):
    output_file = config.get_output_file(f"{config.get_file_name()}.npz")
    
    if os.path.exists(output_file):
        print(f"-- File {output_file} already exists. Skipping.")
        return
    
    feature_dir = config.get_feature_dir()
    motion_dir = config.get_motion_dir()

    sequence_ts = fread.get_timstamps(feature_dir, ext=".secondary.npz")
    sequence_ts = fread.sample_timestamps(sequence_ts, config.target_fps)
    # num_frames = len(sequence_ts)

    accel_df = pd.read_csv(os.path.join(motion_dir, "accel.csv"))
    gyro_df = pd.read_csv(os.path.join(motion_dir, "gyro.csv"))

    accel_df.drop_duplicates("timestamp", inplace=True)
    gyro_df.drop_duplicates("timestamp", inplace=True)
    imu_df = pd.merge(accel_df, gyro_df, on="timestamp", suffixes=("a", "g"))

    frame_rate = accel_df.shape[0] / (accel_df.timestamp.values[-1] - accel_df.timestamp.values[0]) * 1000
    win_len = int(frame_rate * calib_period) # 4 seconds window

    # compute dt in seconds
    imu_df.loc[:, "dt"] = np.concatenate([[0], (imu_df.timestamp.values[1:] - imu_df.timestamp.values[:-1]) / 1000])
    # remove first row as the dt is 0
    imu_df = imu_df.iloc[1:]
    # reset index in pandas data frame
    imu_df.reset_index(drop=True, inplace=True)

    # Fill 0 for displacement, angles, and coordinates
    imu_df.loc[:, "x"] = np.zeros(len(imu_df))
    imu_df.loc[:, "y"] = np.zeros(len(imu_df))
    imu_df.loc[:, "z"] = np.zeros(len(imu_df))

    sigma = 2
    # apply gaussian filter to smooth acceleration and gyro data
    imu_df.loc[:, "xa"] = gaussian_filter1d(imu_df.xa.values, sigma=sigma)
    imu_df.loc[:, "ya"] = gaussian_filter1d(imu_df.ya.values, sigma=sigma)
    imu_df.loc[:, "za"] = gaussian_filter1d(imu_df.za.values, sigma=sigma)
    imu_df.loc[:, "xg"] = gaussian_filter1d(imu_df.xg.values, sigma=sigma)
    imu_df.loc[:, "yg"] = gaussian_filter1d(imu_df.yg.values, sigma=sigma)
    imu_df.loc[:, "zg"] = gaussian_filter1d(imu_df.zg.values, sigma=sigma)

    # gravity vector calculation    
    gravity = imu_df.iloc[:win_len, [1, 2, 3]].mean().values
    # removing gravity
    imu_df[["xa", "ya", "za"]] = imu_df[["xa", "ya", "za"]] - gravity
    # moving average filter        
    accel_mavg = imu_df[["xa", "ya", "za"]].rolling(window=win_len).mean()
    accel_mavg.fillna(0, inplace=True)
    
    imu_df[["xa", "ya", "za"]] = imu_df[["xa", "ya", "za"]] - accel_mavg
    # remove the stabilization data
    imu_df = imu_df.iloc[win_len:].copy()

    # load ground truth trajectory
    start_t = np.where(sequence_ts == functions.nearest(sequence_ts, imu_df.timestamp.values[0]))[0][0]
    sequence_ts = sequence_ts[start_t:]
    num_frames = len(sequence_ts)
    
    # calculate the standard deviation of the depth data
    std_values = []

    for t in range(len(sequence_ts)):
        depth_img_file = os.path.join(config.get_sequence_dir(), f"frame-{sequence_ts[t]}.depth.png")
        std_values.append(registration.calc_std(depth_img_file, 4000))
        
    std_values = np.array(std_values)
        
    cutoffs = registration.get_cutoff_sequence(std_values, config.target_fps, config.min_std, config.threshold, config.cutoff_margin)
    
    local_pcds = []

    for t in tqdm.trange(num_frames):
        feature_file = os.path.join(feature_dir, f"{sequence_ts[t]}.secondary.npz")
        pcd = FCGF.get_features(feature_file, config.voxel_size, pcd_only=True)
        # pcd = pointcloud.preprocess(pcd, config.voxel_size)
        local_pcds.append(pcd)

    local_t = [np.identity(4) for _ in range(num_frames)]

    # start_time = time.time()
    num_iterations = 0
    for start_c, end_c in cutoffs:
        # first frame registration with FPFH
        source, source_fpfh = registration.compute_fpfh(copy.deepcopy(local_pcds[start_c + 1]), config.voxel_size, down_sample=False)
        target, target_fpfh = registration.compute_fpfh(copy.deepcopy(local_pcds[start_c]), config.voxel_size, down_sample=False)

        reg_result = registration.exec_ransac(source, target, source_fpfh, target_fpfh, n_ransac=4, threshold=0.05)
        reg_result = registration.exec_icp(source, target, 0.05, reg_result.transformation, 200)

        # registration.describe(source, target, reg_result)
        # registration.view(source, target, reg_result.transformation)

        velocity = reg_result.transformation[:3, 3] / (sequence_ts[start_c + 1] - sequence_ts[start_c]) * 1000
        
        local_t[start_c + 1] = reg_result.transformation

        for t in tqdm.trange(start_c + 1, end_c - 1):
            start_t, end_t = t, t + 1
            
            imu_slice_df = imu_df[(imu_df.timestamp >= sequence_ts[start_t]) & (imu_df.timestamp <= sequence_ts[end_t])]
            
            # calculate displacement and rotation
            rotation_matrix = np.identity(4)
            translation = np.zeros(3)

            for i in range(len(imu_slice_df)):
                v = imu_slice_df.iloc[i].values
                
                dt = v[7]
                
                # current displacement and rotation
                da = np.degrees([v[j + 4] * dt for j in range(3)])
                
                acceleration = imu_slice_df.iloc[i, [1, 2, 3]].values

                d = [(velocity[j] * dt) + (0.5 * acceleration[j] * dt * dt) for j in range(3)]
                d = np.dot(rotation_matrix, np.array([*d, 1]))
                
                translation = translation + d[:3]
                velocity = [velocity[j] + acceleration[j] * dt for j in range(3)]
                
                rotation_matrix = transform.rotate_transformation_matrix(rotation_matrix, da[0], da[1], da[2])
                
            trans_mat = np.identity(4)
            trans_mat[:3, 3] = translation
            trans_mat[:3, :3] = rotation_matrix[:3, :3]
            
            source = copy.deepcopy(local_pcds[end_t])
            target = copy.deepcopy(local_pcds[start_t])
            
            refined_transform = registration.exec_icp(source, target, 0.05, trans_mat, 200).transformation
            
            velocity = refined_transform[:3, 3] * 1e3 / (sequence_ts[end_t] - sequence_ts[start_t])
            
            local_t[end_t] = refined_transform
            num_iterations += 1
        
    
    # end_time = time.time()
    # print(f"Total time: {end_time - start_time}")
    # print(f"Average time: {(end_time - start_time) / num_iterations}")
    # print(f"Itr/sec: {num_iterations / (end_time - start_time)}")
            
    local_t = np.array(local_t)
    
    trajectory_t = [np.identity(4) for _ in range(num_frames)]

    for start_c, end_c in cutoffs:
        for t in range(start_c + 1, end_c):
            trajectory_t[t] = np.dot(trajectory_t[t - 1], local_t[t])
            
        # trajectory_pcd = []

        # for t in range(start_c, end_c):
        #     pcd = copy.deepcopy(local_pcds[t])
        #     pcd.transform(trajectory_t[t])
        #     trajectory_pcd.append(pcd)

        # trajectory = pointcloud.merge_pcds(trajectory_pcd, config.voxel_size)
        # open3d.visualization.draw_geometries([trajectory])
    
    np.savez_compressed(output_file, sequence_ts=sequence_ts, local_t=local_t, trajectory_t=trajectory_t)
    
    
def normalize_timestamps(source_file, target_file):
    if not os.path.exists(source_file) or not os.path.exists(target_file):
        return
    
    # rename source file
    os.rename(source_file, source_file.replace(".npz", ".bak.npz"))
    
    source = np.load(source_file.replace(".npz", ".bak.npz"))
    target = np.load(target_file)
    
    source_ts = source["sequence_ts"]
    target_ts = target["sequence_ts"]
    
    local_t = source["local_t"]
    trajectory_t = source["trajectory_t"]
    
    target_t = target_ts[0]
    
    # if len(source_ts) == len(target_ts):
    #     return
        
    idx = np.argwhere(source_ts == target_t)[0][0]
    
    source_ts = source_ts[idx:]
    local_t = local_t[idx:]
    trajectory_t = trajectory_t[idx:]
    
    np.savez_compressed(source_file, sequence_ts=source_ts, local_t=local_t, trajectory_t=trajectory_t)
    
    
    
if __name__ == "__main__":
    config = Config(
        sequence_dir="data/raw_data",
        feature_dir="data/features",
        output_dir="data/trajectories/local/FPFH_outlier_removed_0.05",
        experiment="exp_12",
        trial="trial_1",
        subject="subject-1",
        sequence="01",
        groundtruth_dir="data/trajectories/groundtruth",
    )
    
    # config.target_fps=20
    # config.min_std=0.5 # 0.5 for exp 12, 1.0 for exp 13
    # for voxel_size in [0.03, 0.05, 0.08]:
    #     config.voxel_size=voxel_size
    #     print(f"Voxel Size: {config.voxel_size}")
    #     for i in range(1, 11):
    #         print(f"Iteration: {i}")
    #         config.output_dir=f"data/trajectories/local/{i}/IMU_PCD_{config.voxel_size}"
        
    #         for trial in os.listdir(os.path.join(config.feature_dir, config.experiment)):
    #             config.trial = trial
    #             for subject in os.listdir(os.path.join(config.feature_dir, config.experiment, config.trial, str(config.voxel_size))):
    #                 config.subject = subject    
    #                 for sequence in os.listdir(os.path.join(config.feature_dir, config.experiment, config.trial, str(config.voxel_size), config.subject)):
    #                     config.sequence = sequence
    #                     print(f"Processing: {config.experiment} >> {config.trial} >> {config.subject} >> {config.sequence}")
    #                     imu_pcd_fused_registration(config)
        
    # for i in range(1, 11):
    #     print(f"Iteration: {i}")
        
    #     target_files = glob.glob(f"data/trajectories/local/{i}/IMU_PCD_0.08/exp_12/*.npz")
    
    #     for target_file in tqdm.tqdm(target_files):
    #         source_file = target_file.replace("IMU_PCD", "FPFH")
    #         normalize_timestamps(source_file, target_file)
        

    # target_files = glob.glob("data/trajectories/local/IMU_PCD_outlier_removed/exp_13/*.npz")
    
    # for target_file in tqdm.tqdm(target_files):
    #     source_file = target_file.replace("IMU_PCD", "FPFH")
    #     normalize_timestamps(source_file, target_file)