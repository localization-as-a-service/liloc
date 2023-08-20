import open3d
import numpy as np
import pandas as pd
import os
import glob
import tqdm
import copy

import utils.registration as registration
import utils.fread as fread
import utils.transform as transform
import utils.grid_search as grid_search
import utils.functions as functions

from time import sleep
from PIL import Image
from utils.config import Config


def evaluate(config: Config):
    file_name = config.get_file_name()
    
    if not os.path.exists(config.get_output_file(f"{file_name}.npz")):
        print("Unable to find trajectory data. Skipping.")
        return
    
    if not os.path.exists(os.path.join(config.get_groundtruth_dir(), f"{file_name}.gtpose.npz")):
        print("Unable to find groundtruth data. Skipping.")
        return

    groundtruth_data = np.load(os.path.join(config.get_groundtruth_dir(), f"{file_name}.gtpose.npz"))
    estimated_data = np.load(config.get_output_file(f"{file_name}.npz"))
    
    feature_dir = config.get_feature_dir()
    sequence_ts = fread.get_timstamps(feature_dir, ext=".secondary.npz")
    sequence_ts = fread.sample_timestamps(sequence_ts, config.target_fps)
    
    target_ts = estimated_data["sequence_ts"]

    target_inds = [np.argwhere(sequence_ts == target_ts[i])[0][0] for i in range(len(target_ts))]

    found_correct_global, found_correct_global_at = estimated_data["info"]
    estimated_t = estimated_data["global_t"]
    groundtruth_t = groundtruth_data["local_t"][target_inds]
    
    translation_error = []
    rotation_error = []

    for t in range(len(estimated_t)):
        if np.sum(estimated_t[t]) == 4: continue
        if np.sum(groundtruth_t[t]) == 0: continue
        
        er, et = transform.calc_error(estimated_t[t], groundtruth_t[t])
        translation_error.append(et)
        rotation_error.append(er)
        
    print(f"Translation error: {np.mean(translation_error):.3f} ({np.std(translation_error):.3f})", end="\t")
    print(f"Rotation error: {np.mean(rotation_error):.3f} ({np.std(rotation_error):.3f})")
    
    return [np.mean(translation_error), np.mean(rotation_error), found_correct_global, found_correct_global_at]

if __name__ == "__main__":
    config = Config(
        sequence_dir="../local-registration/data/raw_data",
        feature_dir="../local-registration/data/features",
        output_dir="../local-registration/data/trajectories/trajectory/IMU_PCD",
        experiment="exp_11",
        trial="trial_1",
        subject="subject-1",
        sequence="01",
        groundtruth_dir="../local-registration/data/trajectories/groundtruth",
    )
    
    config.voxel_size=0.03
    config.target_fps=20
    config.min_std=0.5
    
    for i in range(3, 6)
    
    for trial in os.listdir(os.path.join(config.feature_dir, config.experiment)):
        config.trial = trial
        for subject in os.listdir(os.path.join(config.feature_dir, config.experiment, config.trial, str(config.voxel_size))):
            config.subject = subject    
            for sequence in os.listdir(os.path.join(config.feature_dir, config.experiment, config.trial, str(config.voxel_size), config.subject)):
                config.sequence = sequence
                print(f"Processing: {config.experiment} >> {config.trial} >> {config.subject} >> {config.sequence}")
                evaluate(config)