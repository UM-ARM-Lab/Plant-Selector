#!/usr/bin/env python
# Import useful libraries and functions
import argparse
import sys

import open3d as o3d
import numpy as np
import pcl as pcl

import rospy
import ros_numpy
from math import atan, sin, cos, pi
from sensor_msgs.msg import PointCloud2
from statistics import mode
from std_msgs.msg import ColorRGBA
from matplotlib import colors
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from arc_utilities.tf2wrapper import TF2Wrapper
from sklearn.decomposition import PCA
from sensor_msgs import point_cloud2 as pc2
from std_msgs.msg import String
import struct
import ctypes
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointField
from std_msgs.msg import Header

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

    color = [r,g,b]

    return color

def plot_pointcloud_rviz(pub, points, frame_id):
    """
    Args:
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
    pub.publish(pc2_msg)


class PlantExtractor:
    def __init__(self, camera_frame):
        # Initialize publishers for PCs, arrow and planes
        # SYNTAX: pub = rospy.Publisher('topic_name', geometry_msgs.msg.Point, queue_size=10)
        # Second argument was imported in the beginning
        self.src_pub = rospy.Publisher("source_pc", PointCloud2, queue_size=10)
        self.inliers_pub = rospy.Publisher("inliers_pc", PointCloud2, queue_size=10)
        self.arrow_pub = rospy.Publisher("normal", Marker, queue_size=10)
        self.plane_pub = rospy.Publisher("plane", PointCloud2, queue_size=10)

        rospy.Subscriber("/plant_selector/mode", String, self.mode_change)

        self.frame_id = camera_frame

        # Set the default mode to branch
        self.mode = "Branch"
        self.pc_sub = rospy.Subscriber("/plant_selector/filtered", PointCloud2, self.select_plant)

    def mode_change(self, new_mode):
        """
        Callback to a new mode type from /plant_selector/mode. Modes can be either None, Branch, Weed, or Cancel.
        :param new_mode: ros string msg
        """
        self.mode = new_mode.data
        if self.mode == "Branch":
            self.pc_sub = rospy.Subscriber("/plant_selector/filtered", PointCloud2, self.select_plant)
        elif self.mode == "Weed":
            self.pc_sub = rospy.Subscriber("/rviz_selected_points", PointCloud2, self.select_plant)
        rospy.loginfo("New mode: " + self.mode)

    def select_plant(self, selection):
        """
        Callback to a new pointcloud message to /rviz_selected_points. This func calls a function to handle
        branch or weed extraction depending on the type of mode is currently active
        :param selection: ros pointcloud2 message
        """
        rospy.loginfo("About to detect plant in \"" + self.mode + "\" mode")
        if self.mode == "Branch":
            self.select_branch(selection)
        elif self.mode == "Weed":
            self.select_weed(selection)

    def project(self, u, n):
        """
        This functions projects a vector "u" to a plane "n" following a mathematical equation.
        :param u: vector that is going to be projected. (numpy array)
        :param n: normal vector of the plane (numpy array)
        :return: vector projected onto the plane (numpy array)
        """
        return u - np.dot(u, n) / np.linalg.norm(n) * n

    def rviz_arrow(self, start, direction, name, thickness, length_scale, color):
        """
        This function displays an arrow in Rviz.
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
        msg.header.frame_id = self.frame_id
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
        self.arrow_pub.publish(msg)

    def plot_plane(self, centroid, normal, size: float = 1, res: float = 0.01):
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
        plot_pointcloud_rviz(self.plane_pub, points_flat[:, :3], self.frame_id)

    def select_branch(self, selection):
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
        # TODO: Need to figure out either how to do different plane segmentation on a numpy array, or convert ros to pcl
        # Transform open3d PC to numpy array
        points = np.array(list(pc2.read_points(selection)))

        pcd_points = points[:, :3]
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

        # Save xyzrgb info in green_points (type: numpy array)
        green_points_xyz = pcd_points[green_points_indices]
        green_points_rgb = pcd_colors[green_points_indices]

        # Create Open3D point cloud for green points
        green_pcd = o3d.geometry.PointCloud()
        # Save xyzrgb info in green_pcd (type: open3d.PointCloud)
        green_pcd.points = o3d.utility.Vector3dVector(green_points_xyz)
        green_pcd.colors = o3d.utility.Vector3dVector(green_points_rgb)

        # Apply plane segmentation function from open3d and get the best inliers
        _, best_inliers = green_pcd.segment_plane(distance_threshold=0.01,
                                                  ransac_n=3,
                                                  num_iterations=1000)
        # Just save and continue working with the inlier points defined by the plane segmentation function
        inlier_points = pcd_points[best_inliers]
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
        cut_direction = self.project(camera_to_centroid, normal)
        # Normalize the projected vector
        cut_direction_normalized = cut_direction / np.linalg.norm(cut_direction)
        # Cross product between normalized cut director vector and the normal of the plane to obtain the
        # 2nd principal component
        cut_y = np.cross(cut_direction_normalized, normal)

        tfw = TF2Wrapper()
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

        for x in range(10):
            tfw.send_transform_matrix(camera2ee, parent=self.frame_id, child='end_effector_left')
            rospy.sleep(0.1)

        # Rviz commands
        # Call rviz_arrow function to first component, cut direction and second component
        # self.rviz_arrow(inliers_centroid, normal, name='first component', length_scale=0.04, color='r', thickness=0.008)
        # self.rviz_arrow(inliers_centroid, cut_y, name='cut y', length_scale=0.05, color='g', thickness=0.008)
        # self.rviz_arrow(inliers_centroid, cut_direction, name='cut direction', length_scale=0.05, color='b', thickness=0.008)

        # Call plot_plane function to visualize plane in Rviz
        # self.plot_plane(inliers_centroid, normal, size=0.05, res=0.001)
        # Call plot_pointcloud_rviz function to visualize PCs in Rviz
        plot_pointcloud_rviz(self.src_pub, points[:, :3], self.frame_id)
        plot_pointcloud_rviz(self.inliers_pub, inlier_points[:, :3], self.frame_id)
        # Call rviz_arrow function to see normal of the plane
        self.rviz_arrow(inliers_centroid, normal, name='normal', thickness=0.008, length_scale=0.15,
                        color='r')
        # Call plot_plane function to visualize plane in Rviz
        self.plot_plane(inliers_centroid, normal, size=0.1, res=0.001)

    def select_weed(self, selection):
        # Load point cloud and visualize it
        # TODO: Figure out how to get the rgb part of the ros msg
        points = np.array(list(pc2.read_points(selection)))
        pcd_points = points[:, :3]
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

        # Save xyzrgb info in green_points (type: numpy array)
        green_points_xyz = pcd_points[green_points_indices]
        green_points_rgb = pcd_colors[green_points_indices]

        # Create Open3D point cloud for green points
        green_pcd = o3d.geometry.PointCloud()
        # Save xyzrgb info in green_pcd (type: open3d.PointCloud)
        green_pcd.points = o3d.utility.Vector3dVector(green_points_xyz)
        green_pcd.colors = o3d.utility.Vector3dVector(green_points_rgb)
        # Apply radius outlier filter to green_pcd
        _, ind = green_pcd.remove_radius_outlier(nb_points=5, radius=0.02)
        # Just keep the inlier points in the point cloud
        green_pcd = green_pcd.select_by_index(ind)
        green_pcd_points = np.asarray(green_pcd.points)

        # Apply DBSCAN to green points
        labels = np.array(green_pcd.cluster_dbscan(eps=0.02, min_points=10))

        """
        # Color clusters and visualize them
        max_label = labels.max()
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        green_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([green_pcd])
        """

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
        [a, b, c, _] = plane_model
        # Just save and continue working with the inlier points defined by the plane segmentation function
        inlier_dirt_points = dirt_points_xyz[best_inliers]
        # Get centroid of dirt
        inlier_dirt_centroid = np.mean(inlier_dirt_points, axis=0)
        # The a, b, c coefficients of the plane equation are the components of the normal vector of that plane
        normal = np.asarray([a, b, c])
        if normal[2] > 0:
            normal = -normal

        phi = atan(normal[1] / normal[2])
        if phi < pi/2:
            phi = phi + pi - 2 * phi
        theta = atan(normal[0] / -normal[2])

        Rx = np.asarray([[1, 0, 0],
                         [0, cos(phi), -sin(phi)],
                         [0, sin(phi), cos(phi)]])
        Ry = np.asarray([[cos(theta), 0, sin(theta)],
                         [0, 1, 0],
                         [-sin(theta), 0, cos(theta)]])

        frame2vector_rot = Rx @ Ry

        """
        # Apply PCA and get just one principal component
        pca = PCA(n_components=3)
        # Fit the PCA to the inlier points
        pca.fit(inlier_dirt_points)
        # The third component (vector) is the normal of the plane of the dirt we are looking for
        third_comp = pca.components_[2]
        """

        tfw = TF2Wrapper()
        # Construct transformation matrix from camera to tool of end effector
        camera2tool = np.zeros([4, 4])
        camera2tool[:3, :3] = frame2vector_rot
        camera2tool[:3, 3] = weed_centroid
        camera2tool[3, 3] = 1

        # Define transformation matrix from tool to end effector
        tool2ee = tfw.get_transform(parent="left_tool", child="end_effector_left")
        # Define transformation matrix from camera to end effector
        camera2ee = camera2tool @ tool2ee
        # Display gripper
        for x in range(10):
            tfw.send_transform_matrix(camera2ee, parent=self.frame_id, child='end_effector_left')
            rospy.sleep(0.1)
        # Call plot_pointcloud_rviz function to visualize PCs in Rviz
        # Visualize all the point cloud as "source"
        plot_pointcloud_rviz(self.src_pub, pcd_points[:, :3], self.frame_id)
        # Visualize filtered green points as "inliers"
        plot_pointcloud_rviz(self.inliers_pub, green_pcd_points[:, :3], self.frame_id)
        # Call rviz_arrow function to see normal of the plane
        self.rviz_arrow(inlier_dirt_centroid, normal, name='normal', thickness=0.008, length_scale=0.15,
                        color='w')
        # Call plot_plane function to visualize plane in Rviz
        self.plot_plane(inlier_dirt_centroid, normal, size=0.1, res=0.001)


def main():
    rospy.init_node("plant_extraction")

    parser = argparse.ArgumentParser()
    parser.add_argument('frame', type=str)
    args = parser.parse_args(rospy.myargv(sys.argv[1:]))

    plant_extractor = PlantExtractor(args.frame)
    rospy.spin()


if __name__ == '__main__':
    main()
