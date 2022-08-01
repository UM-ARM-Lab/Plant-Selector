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


def publish_pc_no_color(publisher, points, frame_id):
    """
    Args:
        publisher: ros publisher
        points: an Nx3 array
        frame_id: the frame to publish in

    Returns: a PointCloud2 message ready to be published to rviz

    """
    header = Header(frame_id=frame_id)
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1)
              # PointField('rgb', 12, PointField.FLOAT32, 1)
              ]
    pc2_msg = point_cloud2.create_cloud(header, fields, points)
    publisher.publish(pc2_msg)


def publish_pc_with_color(publisher, points, frame_id):
    """
    Args:
        publisher: ros publisher
        points: an Nx4 array
        frame_id: the frame to publish in

    Returns: a PointCloud2 message ready to be published to rviz

    """
    header = Header(frame_id=frame_id)
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1),
              PointField('rgb', 12, PointField.FLOAT32, 1)
              ]
    pc2_msg = point_cloud2.create_cloud(header, fields, points)
    publisher.publish(pc2_msg)


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

    color = [r, g, b]

    return color


def cluster_filter(pc):
    points = np.array(list(sensor_msgs.point_cloud2.read_points(pc)))

    if points.shape[0] == 0:
        rospy.loginfo("No points selected")
        return

    # Perform a color filter
    # points = helpers.green_color_filter(points)

    # TODO: The eps value here might want to somehow change dynamically where points further away can have clusters more spread out?
    # The eps value really depends on how good the video quality is and how far away points are from each other
    # clustering = DBSCAN(eps=0.015, min_samples=20).fit(points)
    clustering = hdbscan.HDBSCAN(min_cluster_size=30, gen_min_span_tree=True, allow_single_cluster=1).fit(points)
    # labels = clusterer.labels_
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # If there are no clusters, return
    if n_clusters == 0:
        rospy.loginfo("Invalid selection for branch selection")
        return

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
    return best_selection


# TODO: Probably throw errors instead of returning None twice
def green_color_filter(points):
    """
    Filters out points that are not green
    Args:
        points: an Nx4 numpy array where the 4th col is color

    Returns: an Nx4 numpy array that only has green points

    """
    pcd_points = points[:, :3]
    float_colors = points[:, 3]

    pcd_colors = np.array((0, 0, 0))
    for x in float_colors:
        rgb = float_to_rgb(x)
        pcd_colors = np.vstack((pcd_colors, rgb))

    pcd_colors = pcd_colors[1:, :]

    # Filter the point cloud so that only the green points stay
    # Get the indices of the points with g parameter greater than x
    green_points_indices = np.where((pcd_colors[:, 1] - pcd_colors[:, 0] > pcd_colors[:, 1] / 10.0) &
                                    (pcd_colors[:, 1] - pcd_colors[:, 2] > pcd_colors[:, 1] / 10.0))

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
        rospy.loginfo("Not enough points. Try again.")
        return None, None

    # Just keep the inlier points in the point cloud
    green_pcd = green_pcd.select_by_index(ind)
    green_pcd_points = np.asarray(green_pcd.points)

    # Apply DBSCAN to green points
    labels = np.array(green_pcd.cluster_dbscan(eps=0.0055, min_points=15))  # This is actually pretty good

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters == 0:
        rospy.loginfo("Not enough points. Try again.")
        return None, None

    # Get labels of the biggest cluster
    biggest_cluster_indices = np.where(labels[:] == mode(labels))
    # Just keep the points that correspond to the biggest cluster (weed)
    green_pcd_points = green_pcd_points[biggest_cluster_indices]

    # Get the points that were not green
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

    return green_pcd_points, dirt_pcd


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


def rviz_arrow(arrow_pub, frame_id, start, direction, name, thickness=0.008, length_scale=0.15, color='w'):
    color_msg = ColorRGBA(*colors.to_rgba(color))

    # Define ROS message
    msg = Marker()
    msg.type = Marker.ARROW
    msg.action = Marker.ADD
    msg.ns = name
    msg.header.frame_id = frame_id
    msg.color = color_msg

    # Define endpoint of the arrow, given by the start point, the direction and a length_scale parameter
    end = start + direction * length_scale
    # Construct ROS message for the start and end of the arrow
    msg.points = [
        ros_numpy.msgify(Point, start),
        ros_numpy.msgify(Point, end),
    ]
    msg.pose.orientation.w = 1
    msg.scale.x = thickness
    msg.scale.y = thickness * 2

    # Publish message
    arrow_pub.publish(msg)


def plot_plane(plane_pub, frame_id, center, normal, size: float = 0.1, res: float = 0.001):
    """
    This function plots a plane in Rviz.
    Args:
        plane_pub: ros publisher
        frame_id: frame id of plane to publish
        center: center of plane
        normal: normal to the plane
        size: how large the plane is
        res: how "dense" the plane is
    """
    # Get three orthogonal vectors
    # Create a random vector from the normal vector
    r = normal + [1, 0, 0]
    # Normalize normal vector
    r = r / np.linalg.norm(r)
    # Normalize normal vector
    v0 = normal / np.linalg.norm(normal)
    # The other two orthogonal vectors
    v1 = np.cross(v0, r)
    v2 = np.cross(v0, v1)

    # Define the size and resolution of the plane
    t = np.arange(-size, size, res)

    # Construct 't' by 3 matrix
    v1s = t[:, None] * v1[None, :]
    v2s = t[:, None] * v2[None, :]

    # Construct a 't' by 't' by 3 matrix for the plane
    v1s_repeated = np.tile(v1s, [t.size, 1, 1])
    # Define the points that will construct the plane
    points = center + v1s_repeated + v2s[:, None]
    # Flatten the points
    points_flat = points.reshape([-1, 3])

    # Call the function to plot plane as a PC
    publish_pc_no_color(plane_pub, points_flat[:, :3], frame_id)


def project(u, n):
    """
    This functions projects a vector "u" to a plane "n" following a mathematical equation.

    :param u: vector that is going to be projected. (numpy array)
    :param n: normal vector of the plane (numpy array)
    :return: vector projected onto the plane (numpy array)
    """
    return u - np.dot(u, n) / np.linalg.norm(n) * n
