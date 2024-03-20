import os
import open3d
import numpy as np
import tqdm

import utils.pointcloud as pointcloud
import utils.fread as fread
import utils.functions as functions

from utils.depth_camera import DepthCamera


def convert_to_point_clouds(dataset_dir, subject_id=1, device_id=3, aligned=True):
    """
    Go through the dataset structure and convert all the depth images to point clouds

    Args:
        dataset_dir (str): the directory contains the raw captures
    """
    
    out_dir = dataset_dir.replace("raw_data", "point_clouds")

    if not os.path.exists(out_dir): os.makedirs(out_dir)

    # Load pre-computed camera extrinsic parameters
    pose_device_0 = np.loadtxt(os.path.join(dataset_dir, "global/transformations/device-0.txt"))
    pose_device_1 = np.loadtxt(os.path.join(dataset_dir, "global/transformations/device-1.txt"))
    pose_device_2 = np.loadtxt(os.path.join(dataset_dir, "global/transformations/device-2.txt"))

    # Load pre-computed camera intrinsic parameters
    device_0 = DepthCamera("device-0", os.path.join(dataset_dir, f"../metadata/device-0{'-aligned' if aligned else ''}.json"))
    device_1 = DepthCamera("device-1", os.path.join(dataset_dir, f"../metadata/device-1{'-aligned' if aligned else ''}.json"))
    device_2 = DepthCamera("device-2", os.path.join(dataset_dir, f"../metadata/device-2{'-aligned' if aligned else ''}.json"))
    # Secondary camera
    device_3 = DepthCamera("device-3", os.path.join(dataset_dir, f"../metadata/device-{device_id}-aligned.json"))

    # Iterate through the secondary directory
    subject = f"subject-{subject_id}"
    for seq_id in os.listdir(os.path.join(dataset_dir, "secondary", subject)):
        sequence_dir = os.path.join(dataset_dir, "secondary", subject, seq_id, "frames")
        seq_out_dir = os.path.join(out_dir, subject, seq_id)

        if not os.path.exists(seq_out_dir): os.makedirs(seq_out_dir)

        sequence_ts = fread.get_timstamps_from_images(sequence_dir, ".depth.png")
        # sequence_ts = fread.sample_timestamps(sequence_ts, 20)

        device_0_ts = fread.get_timstamps_from_images(os.path.join(dataset_dir, "global", "device-0"), ".depth.png")
        device_1_ts = fread.get_timstamps_from_images(os.path.join(dataset_dir, "global", "device-1"), ".depth.png")
        device_2_ts = fread.get_timstamps_from_images(os.path.join(dataset_dir, "global", "device-2"), ".depth.png")
        
        for t in tqdm.tqdm(sequence_ts):
            if os.path.exists(os.path.join(seq_out_dir, f"{t}.global.pcd")) and os.path.exists(os.path.join(seq_out_dir, f"{t}.secondary.pcd")):
                continue
            
            pcd_g0 = device_0.depth_to_point_cloud(os.path.join(dataset_dir, "global", "device-0", f"frame-{functions.nearest(device_0_ts, t)}.depth.png"))
            pcd_g1 = device_1.depth_to_point_cloud(os.path.join(dataset_dir, "global", "device-1", f"frame-{functions.nearest(device_1_ts, t)}.depth.png"))
            pcd_g2 = device_2.depth_to_point_cloud(os.path.join(dataset_dir, "global", "device-2", f"frame-{functions.nearest(device_2_ts, t)}.depth.png"))
            
            pcd_g0.transform(pose_device_0)
            pcd_g1.transform(pose_device_1)
            pcd_g2.transform(pose_device_2)
            
            global_pcd = pcd_g0 + pcd_g1 + pcd_g2
            global_pcd = open3d.geometry.voxel_down_sample(global_pcd, voxel_size=0.03)
            
            secondary_pcd = device_3.depth_to_point_cloud(os.path.join(sequence_dir, f"frame-{t}.depth.png"))
            secondary_pcd = open3d.geometry.voxel_down_sample(secondary_pcd, voxel_size=0.03)
            
            global_pcd = pointcloud.remove_outliers(global_pcd)
            secondary_pcd = pointcloud.remove_outliers(secondary_pcd)
            
            open3d.io.write_point_cloud(os.path.join(seq_out_dir, f"{t}.global.pcd"), global_pcd)
            open3d.io.write_point_cloud(os.path.join(seq_out_dir, f"{t}.secondary.pcd"), secondary_pcd)


if __name__ == "__main__":
    convert_to_point_clouds("data/raw_data/exp_13/trial_4", subject_id=1, device_id=3, aligned=False)
