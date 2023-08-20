import numpy as np
import open3d


def merge_pcds(pcds, voxel_size):
    global_pcd = open3d.geometry.PointCloud()

    for local_pcd in pcds:
        global_pcd += local_pcd
    
    return open3d.geometry.voxel_down_sample(global_pcd, voxel_size)


def random_color():
    return np.random.rand(3)


def rgb(r, g, b):
    return r / 255.0, g / 255.0, b / 255.0


def compute_normals(pcd, voxel_size):
    radius_normal = voxel_size * 2
    open3d.geometry.estimate_normals(pcd, open3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    return pcd


def remove_outliers(pcd, nb_neighbors=80, std_ratio=0.5):
    _, ind = open3d.geometry.statistical_outlier_removal(pcd, nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return open3d.geometry.select_down_sample(pcd, ind)
    
   
def make_pcd(points):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    return pcd


def view(pcd):
    open3d.visualization.draw_geometries([pcd])