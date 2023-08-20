import numpy as np
import pandas as pd
import open3d
import copy
import tqdm
import os
import time

import utils.registration as registration
import utils.fread as fread
import utils.FCGF as FCGF
import utils.grid_search_unopt as grid_search
import utils.transform as transform

from concurrent.futures import ThreadPoolExecutor
from PIL import Image



def global_registration(src_feature_file, tgt_feature_file, voxel_size, refine_enabled=False):
    source, source_feat, _ = FCGF.get_features(src_feature_file, voxel_size)
    target, target_feat, _ = FCGF.get_features(tgt_feature_file, voxel_size)

    reg_result = registration.exec_ransac(source, target, source_feat, target_feat, n_ransac=3, threshold=0.05)
    
    if refine_enabled and not reg_result:
        reg_result = registration.exec_icp(source, target, threshold=0.05, trans_init=reg_result.transformation, max_iteration=200)
    
    return source, target, reg_result


def compare_global_grid_optimized(root_feature_dir, experiment, trial, subject, sequence, voxel_size):
    file_name = f"{experiment}__{trial}__{subject}__{sequence}"
    output_dir = f"results/compression/global_registration/{voxel_size}"
    
    if os.path.exists(os.path.join(output_dir, f"{file_name}.npz")): return
    
    groundtruth_data = np.load(os.path.join("data/trajectories/groundtruth", experiment, f"{file_name}.gtpose.npz"))
    
    feature_dir = os.path.join(root_feature_dir, experiment, trial, str(voxel_size), subject, sequence)

    sequence_ts = fread.get_timstamps(feature_dir, ext=".secondary.npz")
    delta = (sequence_ts[1:] - sequence_ts[:-1])
    target_fps = 1e3 / np.mean(delta)

    print(f"Target FPS: {target_fps}")
    num_frames = len(sequence_ts)

    sequence_inds = np.arange(0, num_frames, int(target_fps * 0.8))
    groundtruth_t = groundtruth_data["local_t"]
    
    
    estimated_t = [[], []]
    execution_times = [[], []]

    for t in tqdm.tqdm(sequence_inds):
        try:
            source_feature_file = os.path.join(feature_dir, f"{sequence_ts[t]}.secondary.npz")
            compressed_source_feature_file = os.path.join(feature_dir.replace(experiment, f"{experiment}_compressed"), f"{sequence_ts[t]}.secondary.npz")
            target_feature_file = os.path.join(feature_dir, f"{sequence_ts[t]}.global.npz")
            
            start_t = time.time()
            _, _, reg_result = grid_search.global_registration(source_feature_file, target_feature_file, voxel_size, cell_size=2, refine_enabled=True)
            end_t = time.time()
            estimated_t[0].append(reg_result.transformation if reg_result else np.identity(4))
            execution_times[0].append(end_t - start_t)
            
            start_t = time.time()
            _, _, reg_result = grid_search.global_registration(compressed_source_feature_file, target_feature_file, voxel_size, cell_size=2, refine_enabled=True)
            end_t = time.time()
            estimated_t[1].append(reg_result.transformation if reg_result else np.identity(4))
            execution_times[1].append(end_t - start_t)
            
            # source = FCGF.get_features(source_feature_file, voxel_size, pcd_only=True)
            # target = FCGF.get_features(target_feature_file, voxel_size, pcd_only=True)
            
            # registration.view(source, target, groundtruth_t[t])
        except FileNotFoundError:
            estimated_t[0].append(np.identity(4))
            estimated_t[1].append(np.identity(4))
        
    translation_error = [[], []]
    rotation_error = [[], []]

    for t in tqdm.tqdm(range(len(sequence_inds))):
        
        if np.sum(groundtruth_t[sequence_inds[t]]) == 0: 
            for i in range(2):
                translation_error[i].append(-1)
                rotation_error[i].append(-1)
            continue
        
        for i in range(2):
            er, et = transform.calc_error(estimated_t[i][t], groundtruth_t[sequence_inds[t]])
            translation_error[i].append(et)
            rotation_error[i].append(er)
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    np.savez_compressed(os.path.join(output_dir, f"{file_name}.npz"),
                        sequence_inds=sequence_inds, 
                        sequence_ts=sequence_ts,
                        translation_error=translation_error,
                        rotation_error=rotation_error
    )
    
if __name__ == "__main__":
    root_feature_dir = "data/features"

    experiment = "exp_12"
    trial = "trial_1"
    subject = "subject-1"

    voxel_size = 0.05
    target_fps = 20
    
    for sequence in os.listdir(os.path.join(root_feature_dir, experiment, trial, str(voxel_size), subject)):
        print(f"Experiment: {experiment}, Trial: {trial}, Subject: {subject}, Sequence: {sequence}")
        compare_global_grid_optimized(root_feature_dir, experiment, trial, subject, sequence, voxel_size)