#!/usr/bin/env python
import ctypes
import struct

import hdbscan
import numpy as np
from matplotlib import colors

import ros_numpy
import rospy
import sensor_msgs
from geometry_msgs.msg import Point
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointField
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker
import open3d as o3d
from statistics import mode

import helpers as hp


def calculate_weed_centroid(points):
    # All the weed extraction algorithm
    pcd_points = points[:, :3]
    float_colors = points[:, 3]

    pcd_colors = np.array((0, 0, 0))
    for x in float_colors:
        rgb = hp.float_to_rgb(x)
        pcd_colors = np.vstack((pcd_colors, rgb))

    # Alternate green color filter
    pcd_colors = pcd_colors[1:, :]

    # Filter the point cloud so that only the green points stay
    # Get the indices of the points with g parameter greater than x
    green_points_indices = np.where((pcd_colors[:, 1] - pcd_colors[:, 0] > pcd_colors[:, 1] / 12.0) &
                                    (pcd_colors[:, 1] - pcd_colors[:, 2] > pcd_colors[:, 1] / 12.0))
    green_points_xyz = pcd_points[green_points_indices]
    green_points_rgb = pcd_colors[green_points_indices]

    r_low, g_low, b_low = 10, 20, 10
    r_high, g_high, b_high = 240, 240, 240
    green_points_indices = np.where((green_points_rgb[:, 0] > r_low) & (green_points_rgb[:, 0] < r_high) &
                                    (green_points_rgb[:, 1] > g_low) & (green_points_rgb[:, 1] < g_high) &
                                    (green_points_rgb[:, 2] > b_low) & (green_points_rgb[:, 2] < b_high))

    if len(green_points_indices[0]) == 1:
        rospy.loginfo("No green points found. Try again.")
        return None, None

    # Save xyzrgb info in green_points (type: numpy array)
    green_points_xyz = green_points_xyz[green_points_indices]
    green_points_rgb = green_points_rgb[green_points_indices]

    # Create Open3D point cloud for green points
    green_pcd = o3d.geometry.PointCloud()
    # Save xyzrgb info in green_pcd (type: open3d.PointCloud)
    green_pcd.points = o3d.utility.Vector3dVector(green_points_xyz)
    green_pcd.colors = o3d.utility.Vector3dVector(green_points_rgb)

    # Apply radius outlier filter to green_pcd
    _, ind = green_pcd.remove_radius_outlier(nb_points=7, radius=0.007)

    if len(ind) == 0:
        print("Not enough points. Try again.")
        return None, None

    # Just keep the inlier points in the point cloud
    green_pcd = green_pcd.select_by_index(ind)
    green_pcd_points = np.asarray(green_pcd.points)

    # Apply DBSCAN to green points
    labels = np.array(green_pcd.cluster_dbscan(eps=0.0055, min_points=15))  # This is actually pretty good

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters == 0:
        print("Not enough points. Try again.")
        return None, None

    # Get labels of the biggest cluster
    biggest_cluster_indices = np.where(labels[:] == mode(labels))
    # Just keep the points that correspond to the biggest cluster (weed)
    green_pcd_points = green_pcd_points[biggest_cluster_indices]

    # Get coordinates of the weed centroid
    weed_centroid = np.mean(green_pcd_points, axis=0)

    dirt_indices = np.arange(0, len(pcd_points))
    # These are the indices for dirt
    dirt_indices = np.setdiff1d(dirt_indices, green_points_indices)
    # Save xyzrgb info in dirt_points (type: numpy array) from remaining indices of green points filter
    dirt_points_xyz = pcd_points[dirt_indices]
    dirt_points_rgb = pcd_colors[dirt_indices]
    # Create PC for dirt points
    dirt_pcd = o3d.geometry.PointCloud()
    # Save points and color to the point cloud
    dirt_pcd.points = o3d.utility.Vector3dVector(dirt_points_xyz)
    dirt_pcd.colors = o3d.utility.Vector3dVector(dirt_points_rgb)

    # Apply plane segmentation function from open3d and get the best inliers
    plane_model, best_inliers = dirt_pcd.segment_plane(distance_threshold=0.0005,
                                                       ransac_n=3,
                                                       num_iterations=1000)

    if len(best_inliers) == 0:
        rospy.loginfo("Can't find dirt, Select both weed and dirt.")
        return None, None

    [a, b, c, _] = plane_model
    if a < 0:
        normal = np.asarray([a, b, c])
    else:
        normal = -np.asarray([a, b, c])

    return weed_centroid, normal
