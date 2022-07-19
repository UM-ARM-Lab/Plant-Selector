"""
This script is for evaluation of the weed extraction algorithm.
"""
import os
from statistics import mode

import numpy as np
import open3d as o3d

import helpers as hp


def calculate_weed_centroid(list_files, centroids):
    row = 0
    for file in list_files:
        points = np.load(file)

        # All the weed extraction algorithm

        pcd_points = points[:, :3]
        float_colors = points[:, 3]

        pcd_colors = np.array((0, 0, 0))
        for x in float_colors:
            rgb = hp.float_to_rgb(x)
            pcd_colors = np.vstack((pcd_colors, rgb))

        pcd_colors = pcd_colors[1:, :] / 255

        # Filter the point cloud so that only the green points stay
        # Get the indices of the points with g parameter greater than x
        r_low, g_low, b_low = 0.1, 0.3, 0.1
        r_high, g_high, b_high = 0.8, 0.8, 0.6
        green_points_indices = np.where((pcd_colors[:, 0] > r_low) & (pcd_colors[:, 0] < r_high) &
                                        (pcd_colors[:, 1] > g_low) & (pcd_colors[:, 1] < g_high) &
                                        (pcd_colors[:, 2] > b_low) & (pcd_colors[:, 2] < b_high))

        if len(green_points_indices[0]) == 1:
            print("No green points found. Try again.")
            return

        # Save xyzrgb info in green_points (type: numpy array)
        green_points_xyz = pcd_points[green_points_indices]
        green_points_rgb = pcd_colors[green_points_indices]

        # Create Open3D point cloud for green points
        green_pcd = o3d.geometry.PointCloud()
        # Save xyzrgb info in green_pcd (type: open3d.PointCloud)
        green_pcd.points = o3d.utility.Vector3dVector(green_points_xyz)
        green_pcd.colors = o3d.utility.Vector3dVector(green_points_rgb)

        # Apply radius outlier filter to green_pcd
        _, ind = green_pcd.remove_radius_outlier(nb_points=7, radius=0.007)

        if len(green_points_indices[0]) == 0:
            print("Not enough points. Try again.")
            return

        # Just keep the inlier points in the point cloud
        green_pcd = green_pcd.select_by_index(ind)
        green_pcd_points = np.asarray(green_pcd.points)

        # Apply DBSCAN to green points
        labels = np.array(green_pcd.cluster_dbscan(eps=0.007, min_points=15))  # This is actually pretty good

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters == 0:
            print("Not enough points. Try again.")
            return
        # Get labels of the biggest cluster
        biggest_cluster_indices = np.where(labels[:] == mode(labels))
        # Just keep the points that correspond to the biggest cluster (weed)
        green_pcd_points = green_pcd_points[biggest_cluster_indices]

        # Get coordinates of the weed centroid
        weed_centroid = np.mean(green_pcd_points, axis=0)
        centroids[row, :] = weed_centroid
        row += 1

    return centroids


def compute_distances(centroids, manual_labels):
    return np.linalg.norm(manual_labels[:, :3] - centroids[:, :3])


def get_manual_labels(list_files_manual_labels, manual_labels):
    row = 0
    for label in list_files_manual_labels:
        return


def main():
    """
    STEPS:

    DONE:
    - Run through the point clouds of "weed_eval/" folder
    - Get the centroid and save it in an array.
    - Get the manual labels.
    - Get the distance between each one.
    - Print results, graphs, etc.
    :return:
    """
    # Create paths for pcs and manual labels
    path_pcs = "/home/miguel/catkin_ws/src/plant_selector/weed_eval/pcs"
    path_manual_labels = "/home/miguel/catkin_ws/src/plant_selector/weed_eval/manual_labels"

    # Return the list of files and directories present in a specified directory path.
    list_files_pcs = os.listdir(path_pcs)
    list_files_manual_labels = os.listdir(path_manual_labels)

    # Create empty arrays for centroids and manual_labels
    centroids = np.zeros(shape=(len(list_files_pcs), 3))
    manual_labels = np.zeros(shape=(len(list_files_manual_labels), 3))

    # Fill in the arrays
    centroids = calculate_weed_centroid(list_files_pcs, centroids)
    manual_labels = get_manual_labels(list_files_manual_labels, manual_labels)

    error = compute_distances(centroids, manual_labels)
    print(np.mean(error))


if __name__ == '__main__':
    main()
