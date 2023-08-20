import numpy as np
import pandas as pd
import open3d
import copy
import tqdm
import os

import utils.registration as registration
import utils.fread as fread
import utils.FCGF as FCGF
import utils.transform as transform
import utils.grid_search as grid_search


def compute_registration_stats(tgt_feature_file, experiment, frames_per_sequence=50, cell_size=3):
    groundtruth_file = f"results/groundtruth/groundtruth_{experiment}.csv"
    
    global_pcd = FCGF.get_features(tgt_feature_file, pcd_only=True)
    global_pcd.paint_uniform_color([0, 0.651, 0.929])

    grid_points = grid_search.get_grid(global_pcd, cell_size)

    groundtruth_df = pd.read_csv(groundtruth_file)
    groundtruth_df = groundtruth_df[groundtruth_df.correctness]

    sequences = groundtruth_df.loc[:, ["trial", "subject", "sequence"]].values

    for trial, subject, sequence in sequences:
        output_file = f"results/stats/{experiment}_larc_3cams_v2_cell_2/{trial}__{subject}__{sequence:02d}.csv"

        if os.path.exists(output_file): continue 

        pose_file = f"data/trajectories/groundtruth/{experiment}/{experiment}__{trial}__{subject}__{sequence:02d}.pose.refined.npz"
        local_t = np.load(pose_file)["local_t"]
        
        sequence_dir = f"data/features/{experiment}/{trial}/{0.03}/{subject}/{sequence:02d}"

        sequence_ts = fread.get_timstamps(sequence_dir, ext=".secondary.npz")
        num_frames = len(sequence_ts)
        
        frame_inds = np.random.randint(0, num_frames, frames_per_sequence)
        
        data = []

        for t in tqdm.tqdm(frame_inds):
            # reg_result = None
            source, source_feat, _ = FCGF.get_features(os.path.join(sequence_dir, f"{sequence_ts[t]}.secondary.npz"))

            for i, p in enumerate(grid_points):
                target_cell, target_cell_feat, _ = grid_search.get_cell_features(tgt_feature_file, p, cell_size)

                if len(target_cell.points) < 2000:
                    data.append([sequence_ts[t], i, 0, -1, -1, -1, -1])
                    continue

                rsc_res = registration.exec_global_reg(source, target_cell, source_feat, target_cell_feat, n_ransac=3, threshold=0.05)
                
                er, et = helpers.calc_error(rsc_res.transformation, local_t[t])
                data.append([sequence_ts[t], i, len(rsc_res.correspondence_set), rsc_res.inlier_rmse, rsc_res.fitness, et, er])
                
        stats = pd.DataFrame(data, columns=["id", "grid_index", "num_inliers", "inlier_rmse", "fitness", "translation_error", "rotation_error"])
        stats.to_csv(output_file, index=False)
        
        
def compute_ransac_stats(tgt_feature_file, experiment, cell_size=3):
    groundtruth_file = f"results/groundtruth/groundtruth_{experiment}.csv"
    
    ransac_n = 3
    ransac_max_iterations = 100000

    estimation_method = open3d.registration.TransformationEstimationPointToPoint(False)
    
    global_pcd = FCGF.get_features(tgt_feature_file, pcd_only=True)
    global_pcd.paint_uniform_color([0, 0.651, 0.929])

    grid_points = grid_search.get_grid(global_pcd, cell_size)

    groundtruth_df = pd.read_csv(groundtruth_file)
    groundtruth_df = groundtruth_df[groundtruth_df.correctness]

    sequences = groundtruth_df.loc[:, ["trial", "subject", "sequence"]].values

    for trial, subject, sequence in sequences:
        output_file = f"results/ransac_n_stats/{experiment}_reference_stationary/{trial}__{subject}__{sequence:02d}.csv"

        if os.path.exists(output_file): continue 

        sequence_dir = f"data/features/{experiment}/{trial}/{0.03}/{subject}/{sequence:02d}"
        stats_file = f"results/stats/{experiment}_reference_stationary/{trial}__{subject}__{sequence:02d}.csv"
        pose_file = f"data/trajectories/groundtruth/{experiment}/{experiment}__{trial}__{subject}__{sequence:02d}.pose.refined.npz"

        stats = pd.read_csv(stats_file)
        stats.loc[:, "correctness"] = stats.apply(lambda x: x.translation_error < 0.3 and x.rotation_error < 3, axis=1)
        stats = stats[stats.correctness]

        local_t = np.load(pose_file)["local_t"]

        sequence_ts = fread.get_timstamps(sequence_dir, ext=".secondary.npz")

        data = []

        for sequence_t, grid_id in stats.values[:, :2]:
            t = np.where(sequence_ts == sequence_t)[0][0]
            target, target_features, _ = FCGF.get_features(os.path.join(sequence_dir, f"{sequence_t}.secondary.npz"))
            source, source_features, _ = grid_search.get_cell_features(tgt_feature_file, grid_points[grid_id], cell_size)

            similar_features = np.ones(len(source.points)) * -1
            kdtree_feature = open3d.geometry.KDTreeFlann(target_features)

            for i in range(ransac_max_iterations):
                ransac_corres = []
                sample_ids = np.random.randint(0, len(source.points), ransac_n)
                for source_sample_id in sample_ids:
                    if similar_features[source_sample_id] == -1:
                        _, indices, dists = kdtree_feature.search_knn_vector_xd(source_features.data[:, source_sample_id], 1)
                        similar_features[source_sample_id] = indices.pop()
                    ransac_corres.append([source_sample_id, similar_features[source_sample_id]])

                ransac_corres = open3d.utility.Vector2iVector(ransac_corres)
                transformation = estimation_method.compute_transformation(source, target, ransac_corres)

                er, et = helpers.calc_error(helpers.inv_transform(transformation), local_t[t])
                
                if er < 3 and et < 0.2:
                    data.append([sequence_t, t, grid_id, *sample_ids, er, et])
                    
        df = pd.DataFrame(data, columns=["id", "frame", "grid_index", "x1", "x2", "x3", "rotation_error", "translation_error"])
        df.to_csv(output_file, index=False)
        
        
if __name__ == "__main__":
    tgt_feature_file = "data/reference/larc_kitchen_3cams_v2.npz"
    compute_registration_stats(tgt_feature_file, experiment="exp_3", frames_per_sequence=100, cell_size=2)
    
        
        
    