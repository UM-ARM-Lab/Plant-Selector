#!/usr/bin/env python
from audioop import avg
import ctypes
from os import fdatasync
import struct

import hdbscan
from statistics import mode
from math import atan, pi
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.decomposition import PCA
import rospy
from sensor_msgs import point_cloud2 as pc2
from tf.transformations import rotation_matrix

import clustering_tests as ct


def calculate_weed_centroid(points):
    # All the weed extraction algorithm
    pcd_points = points[:, :3]
    float_colors = points[:, 3]

    pcd_colors = np.array((0, 0, 0))
    for x in float_colors:
        rgb = float_to_rgb(x)
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

    # disp_pcd = o3d.geometry.PointCloud()
    # disp_pcd.points = o3d.utility.Vector3dVector(green_pcd_points)
    # disp_pcd.paint_uniform_color([0, 0, 0])
    # o3d.visualization.draw_geometries([disp_pcd])

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
    
    # inlier_cloud = dirt_pcd.select_by_index(best_inliers)
    # outlier_cloud = dirt_pcd.select_by_index(best_inliers, invert=True)
    # inlier_cloud.paint_uniform_color([1, 0, 0])
    # outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    [a, b, c, _] = plane_model
    if a < 0:
        normal = np.asarray([a, b, c])
    else:
        normal = -np.asarray([a, b, c])

    return weed_centroid, normal


def predict_weed_pose(selection):
    # Load point cloud and visualize it
    points = np.array(list(pc2.read_points(selection)))

    if points.shape[0] == 0:
        rospy.loginfo("You selected no points, select a few points")
        return None

    # weed_centroid, normal = ct.RANSAC_calculate_pose(points)
    weed_centroid, normal = ct.kmeans_calculate_pose(points)
    weed_centroid_original, normal_original = calculate_weed_centroid(points)
    print("\nOG wc: ", weed_centroid_original)
    print("new wc: ", weed_centroid)
    print("OG n: ", normal_original)
    print("new n: ", normal)

    if weed_centroid is None:
        return None

    # Currently only for zed
    if normal[2] > 0:
        normal = -normal

    phi = atan(normal[1] / normal[2])
    if phi < pi / 2:
        phi = phi + pi - 2 * phi
    theta = atan(normal[0] / -normal[2])

    # Construct transformation matrix from camera to tool of end effector
    camera2tool = np.eye(4)
    camera2tool[:3, :3] = (rotation_matrix(phi, np.asarray([1, 0, 0])) @
                           rotation_matrix(theta, np.asarray([0, 1, 0])))[:3, :3]
    camera2tool[:3, 3] = weed_centroid

    return camera2tool


def cluster_filter(points):
    # TODO: The eps value here might want to somehow change dynamically where points further away can have clusters
    #  more spread out?
    # The eps value really depends on how good the video quality is and how far away points are from each other
    # clustering = DBSCAN(eps=0.015, min_samples=20).fit(points)
    clustering = hdbscan.HDBSCAN(min_cluster_size=30, gen_min_span_tree=True, allow_single_cluster=1).fit(points)
    # labels = clusterer.labels_
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # If there are no clusters, return
    if n_clusters == 0:
        rospy.loginfo("Invalid selection for branch selection")
        return None

    # Find the cluster closest to the user
    closest_cluster = 0
    closest_cluster_dist = np.inf
    # TODO: Figure out how to get the actual center of camera so it isnt hardcoded camera_location = np.array((0, 0, 0))
    camera_location = np.array([0, 0, 0])
    for x in range(n_clusters):
        sel_indices = np.argwhere(labels == x).squeeze(1)
        this_cluster = points[sel_indices]
        cluster_center = np.sum(this_cluster[:, :3], axis=0) / this_cluster.shape[0]
        dist = np.linalg.norm(cluster_center - camera_location)
        if dist < closest_cluster_dist:
            closest_cluster_dist = dist
            closest_cluster = x

    sel_indices = np.argwhere(labels == closest_cluster).squeeze(1)
    best_selection = points[sel_indices]
    return best_selection[:, :3]


def predict_branch_pose(selection): 
    points = np.array(list(pc2.read_points(selection)))

    # Perform Depth Filter
    points_xyz = cluster_filter(points)

    if points_xyz is None:
        return None

    # Create Open3D point cloud for green points
    pcd = o3d.geometry.PointCloud()
    # Save xyzrgb info in green_pcd (type: open3d.PointCloud)
    pcd.points = o3d.utility.Vector3dVector(points_xyz)

    # Apply plane segmentation function from open 3d and get the best inliers
    _, best_inliers = pcd.segment_plane(distance_threshold=0.01,
                                        ransac_n=3,
                                        num_iterations=1000)
    # Just save and continue working with the inlier points defined by the plane segmentation function
    inlier_points = points_xyz[best_inliers]
    # Get the centroid of the inlier points
    # In Cartesian coordinates, the centroid is just the mean of the components. That is, axis=0 runs down the rows,
    # so at the end you get the mean of x, y and z components (centroid)
    inliers_centroid = np.mean(inlier_points, axis=0)

    # Apply PCA and get just one principal component
    pca = PCA(n_components=1)
    # Fit the PCA to the inlier points
    pca.fit(inlier_points)
    # The first component (vector) is the normal of the plane we are looking for
    normal = pca.components_[0]

    # Since point cloud is relative to camera, the cam position is 0 0 0
    camera_position = np.array([0, 0, 0])
    # This is the "main vector" going from the camera to the centroid of the PC
    camera_to_centroid = inliers_centroid - camera_position

    # Call the project function to get the cut direction vector
    cut_direction = project(camera_to_centroid, normal)
    # Normalize the projected vector
    cut_direction_normalized = cut_direction / np.linalg.norm(cut_direction)
    # Cross product between normalized cut director vector and the normal of the plane to obtain the
    # 2nd principal component
    cut_y = np.cross(cut_direction_normalized, normal)

    # Get 3x3 rotation matrix
    # The first row is the x-axis of the tool frame in the camera frame
    camera2tool_rot = np.array([normal, cut_y, cut_direction_normalized]).T

    # Construct transformation matrix from camera to tool of end effector
    camera2tool = np.zeros([4, 4])
    camera2tool[:3, :3] = camera2tool_rot
    camera2tool[:3, 3] = inliers_centroid
    camera2tool[3, 3] = 1

    return camera2tool


def float_to_rgb(float_rgb):
    """ Converts a packed float RGB format to an RGB list

        Args:
            float_rgb: RGB value packed as a float

        Returns:
            color (list): 3-element list of integers [0-255,0-255,0-255]
    """
    s = struct.pack('>f', float_rgb)
    i = struct.unpack('>l', s)[0]
    pack = ctypes.c_uint32(i).value

    r = (pack & 0x00FF0000) >> 16
    g = (pack & 0x0000FF00) >> 8
    b = (pack & 0x000000FF)

    color = np.array([r, g, b]).astype(np.float)

    return color


def project(u, n):
    """
    This functions projects a vector "u" to a plane "n" following a mathematical equation.

    :param u: vector that is going to be projected. (numpy array)
    :param n: normal vector of the plane (numpy array)
    :return: vector projected onto the plane (numpy array)
    """
    return u - np.dot(u, n) / np.linalg.norm(n) * n


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix