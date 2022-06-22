"""
    This code loads a PointCloud and estimates the best pose for a gripper to grasp the object.
    Created by: Miguel Munoz
    First created: 07 june 2022
"""

# Import necessary libraries and functions
import numpy as np
import open3d as o3d
from matplotlib import colors
from sklearn.decomposition import PCA

import ros_numpy
import rospy
import tf.transformations
from arc_utilities.tf2wrapper import TF2Wrapper
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

camera_frame = 'zed2i_left_camera_frame'


def project(u, n):
    """
    This functions projects a vector "u" to a plane "n" following a mathematic equation.
    :param u: vector that is going to be projected. (numpy array)
    :param n: normal vector of the plane (numpy array)
    :return: vector projected onto the plane (numpy array)
    """
    return u - np.dot(u, n) / np.linalg.norm(n) * n


def plot_pointcloud_rviz(pub: rospy.Publisher, xs, ys, zs):
    """
    This function plots pointcloud in Rviz.
    :param pub: ROS Publisher
    :param xs: x coordinates of points
    :param ys: y coordinates of points
    :param zs: z coordinates of points
    :return: None.
    """
    # Create a list of the coordinates of the points
    list_of_tuples = [(x, y, z) for x, y, z in zip(xs, ys, zs)]
    # Define name and type
    dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32)]
    # Define a tuple with coordinates and type
    np_record_array = np.array(list_of_tuples, dtype=dtype)
    # Construct ROS message
    msg = ros_numpy.msgify(PointCloud2, np_record_array, frame_id=camera_frame, stamp=rospy.Time.now())
    # Publish the message
    pub.publish(msg)


def rviz_arrow(pub, start, direction, name, thickness=0.004, length_scale=0.2, color='r'):
    """
    This function displays an arrow in Rviz.
    :param pub: ROS Publisher
    :param start: vector with coordinates of the start point (origin)
    :param direction: vector that defines de direction of the arrow
    :param name: namespace to display in Rviz
    :param thickness: thickness of the arrow
    :param length_scale: length of the arrow
    :return: None
    """
    color_msg = ColorRGBA(*colors.to_rgba(color))

    # Define ROS message
    msg = Marker()
    msg.type = Marker.ARROW
    msg.action = Marker.ADD
    msg.ns = name
    msg.header.frame_id = camera_frame
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
    pub.publish(msg)


def plot_plane(pub, centroid, normal, size: float = 1, res: float = 0.01):
    """
    This function plots a plane in Rviz.
    :param pub: ROS publisher
    :param centroid: centroid of the inliers
    :param normal: normal vector of the plane
    :param size: size of the plane
    :param res: resolution of the plane (there will be a point every "res" distance)
    :return: None
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
    points = centroid + v1s_repeated + v2s[:, None]
    # Flatten the points
    points_flat = points.reshape([-1, 3])

    # Call the function to plot plane as a PC
    plot_pointcloud_rviz(pub, points_flat[:, 0], points_flat[:, 1], points_flat[:, 2])


def main():
    """
    STEPS:
    - load the selected point cloud
    - compute the grasp/cut pose and publish it so we can see it in rviz
        - get the inliers using open3d plane segmentation
        - compute the centroid of the inliers
        - run PCA on the inliers
        - get the first component
        - define the plane (the 1st component is the normal of the plane)
        - visualize the plane
        - take the gripper to the centroid
        - define the orientation of the gripper
            - project the camera-to-plant vector into the plane
    """
    # Initialize ros node
    rospy.init_node("branch_cutting_demo")
    # Initialize publishers for arrow, and PCs
    # SYNTAX: pub = rospy.Publisher('topic_name', geometry_msgs.msg.Point, queue_size=10)
    # Second argument was imported in the beginning
    arrow_pub = rospy.Publisher("arrow", Marker, queue_size=10)
    src_pub = rospy.Publisher("source", PointCloud2, queue_size=10)
    inliers_pub = rospy.Publisher("inliers", PointCloud2, queue_size=10)
    plane_pub = rospy.Publisher("plane", PointCloud2, queue_size=10)

    # Load PC that was previously selected by the interactive marker in Rviz
    pcd = o3d.io.read_point_cloud("../pcs/branch-cutting/seg-01.pcd")

    # Transform open3d PC to numpy array
    points = np.asarray(pcd.points)
    # Apply plane segmentation function from open3d and get the best inliers
    _, best_inliers = pcd.segment_plane(distance_threshold=0.0005,
                                        ransac_n=3,
                                        num_iterations=1000)
    # Just save and continue working with the inlier points defined by the plane segmentation function
    inlier_points = points[best_inliers]
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

    # Since camera position is lost, define an approximate position
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

    tfw = TF2Wrapper()
    while True:
        # Get 3x3 rotation matrix
        # The first row is the x-axis of the tool frame in the camera frame
        camera2tool_rot = np.array([normal, cut_y, cut_direction_normalized]).T

        # Construct transformation matrix from camera to tool of end effector
        camera2tool = np.zeros([4, 4])
        camera2tool[:3, :3] = camera2tool_rot
        camera2tool[:3, 3] = inliers_centroid
        camera2tool[3, 3] = 1

        # Get transformation matrix between tool and end effector
        tool2ee = tfw.get_transform(parent="left_tool", child="end_effector_left")
        # Chain effect: get transformation matrix from camera to end effector
        camera2ee = camera2tool @ tool2ee

        # Rviz commands
        tfw.send_transform_matrix(camera2ee, parent=camera_frame, child='end_effector_left')
        # Call rviz_arrow function to first component, cut direction and second component
        rviz_arrow(arrow_pub, inliers_centroid, normal, name='first component', length_scale=0.04, color='r')
        rviz_arrow(arrow_pub, inliers_centroid, cut_y, name='cut y', length_scale=0.05, color='g')
        rviz_arrow(arrow_pub, inliers_centroid, cut_direction, name='cut direction', length_scale=0.05, color='b')

        # Call plot_plane function to visualize plane in Rviz
        plot_plane(plane_pub, inliers_centroid, normal, size=0.05, res=0.001)
        # Call plot_pointcloud_rviz function to visualize PCs in Rviz
        plot_pointcloud_rviz(src_pub, points[:, 0], points[:, 1], points[:, 2])
        plot_pointcloud_rviz(inliers_pub, inlier_points[:, 0], inlier_points[:, 1], inlier_points[:, 2])
        rospy.sleep(1)


if __name__ == '__main__':
    main()
