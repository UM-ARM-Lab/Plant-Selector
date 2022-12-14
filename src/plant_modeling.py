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

import find_centroids as fc


def predict_weed_pose(selection, ret_mult=False):
    # Load point cloud and visualize it
    points = np.array(list(pc2.read_points(selection)))

    if points.shape[0] == 0:
        rospy.loginfo("You selected no points, select a few points")
        return None

    # weed_centroid, normal = fc.DBSCAN_calculate_pose(points, return_multiple_grasps=ret_mult)
    weed_centroid, normal = fc.FRG_calculate_pose(points, return_multiple_grasps=ret_mult)
    # weed_centroid, normal = fc.color_calculate_pose(points, return_multiple_grasps=ret_mult)
    if weed_centroid is None:
        return None

    # Currently only for zed
    if normal[2] > 0:
        normal = -normal

    phi = atan(normal[1] / normal[2])
    if phi < pi / 2:
        phi = phi + pi - 2 * phi
    theta = atan(normal[0] / -normal[2])

    # Construct transformation matrices from camera to tool of end effector
    # If multiple grasps are returned, make a list of them. If not, dont
    if weed_centroid.ndim > 1:
        camera2tool_list = []
        for i in range(weed_centroid.shape[0]):
            camera2tool = np.eye(4)
            camera2tool[:3, :3] = (rotation_matrix(phi, np.asarray([1, 0, 0])) @
                                rotation_matrix(theta, np.asarray([0, 1, 0])))[:3, :3]
            camera2tool[:3, 3] = weed_centroid[i, :]
            camera2tool_list.append(camera2tool)
        return camera2tool_list
    else: 
        camera2tool = np.eye(4)
        camera2tool[:3, :3] = (rotation_matrix(phi, np.asarray([1, 0, 0])) @
                            rotation_matrix(theta, np.asarray([0, 1, 0])))[:3, :3]
        camera2tool[:3, 3] = weed_centroid
        return [camera2tool]


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