#!/usr/bin/env python
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointField
from std_msgs.msg import Header
import ctypes
import struct
import numpy as np


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


# TODO: Eventually make this filter take in the upper/lowerbounds so it isnt just green
def green_color_filter(points):
    """
    Filters out points that are not green
    Args:
        points: an Nx4 numpy array where the 4th col is color

    Returns: an Nx4 numpy array that only has green points

    """
    float_colors = points[:, 3]

    pcd_colors = np.array((0, 0, 0))
    for x in float_colors:
        rgb = float_to_rgb(x)
        pcd_colors = np.vstack((pcd_colors, rgb))

    pcd_colors = pcd_colors[1:, :] / 255

    # Filter the point cloud so that only the green points stay
    # Get the indices of the points with g parameter greater than x
    r_low, g_low, b_low = 0, 0.6, 0
    r_high, g_high, b_high = 1, 1, 1
    green_points_indices = np.where((pcd_colors[:, 0] > r_low) & (pcd_colors[:, 0] < r_high) &
                                    (pcd_colors[:, 1] > g_low) & (pcd_colors[:, 1] < g_high) &
                                    (pcd_colors[:, 2] > b_low) & (pcd_colors[:, 2] < b_high))

    if len(green_points_indices[0]) == 1:
        r_low, g_low, b_low = 0, 0.3, 0
        r_high, g_high, b_high = 1, 1, 1
        green_points_indices = np.where((pcd_colors[:, 0] > r_low) & (pcd_colors[:, 0] < r_high) &
                                        (pcd_colors[:, 1] > g_low) & (pcd_colors[:, 1] < g_high) &
                                        (pcd_colors[:, 2] > b_low) & (pcd_colors[:, 2] < b_high))

    return points[green_points_indices]
