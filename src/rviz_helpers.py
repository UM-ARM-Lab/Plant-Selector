#!/usr/bin/env python
import numpy as np
from matplotlib import colors

import ros_numpy
from geometry_msgs.msg import Point
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointField
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker


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
