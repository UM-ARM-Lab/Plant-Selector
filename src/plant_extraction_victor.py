#!/usr/bin/env python
# Import useful libraries and functions
from math import atan, sin, cos, pi
from re import I
from statistics import mode

import numpy as np
import open3d as o3d
from matplotlib import colors
from sklearn.decomposition import PCA

import helpers as hp
import ros_numpy
import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from geometry_msgs.msg import Point
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA
from std_msgs.msg import String
from visualization_msgs.msg import Marker
import hdbscan

# Victor stuff
from arm_robots.victor import Victor
from victor_hardware_interface_msgs.msg import ControlMode
from arm_robots.robot_utils import make_follow_joint_trajectory_goal, PlanningResult, PlanningAndExecutionResult, \
    ExecutionResult, is_empty_trajectory, merge_joint_state_and_scene_msg

class PlantExtractor:
    def __init__(self):
        """
        Initialize publishers for PCs, arrow and planes.

        :param camera_frame: Name of the camera frame to be used.
        """
        # Initialize publishers for PCs, arrow and planes
        # SYNTAX: pub = rospy.Publisher('topic_name', geometry_msgs.msg.Point, queue_size=10)
        self.src_pub = rospy.Publisher("source_pc", PointCloud2, queue_size=10)
        self.inliers_pub = rospy.Publisher("inliers_pc", PointCloud2, queue_size=10)
        self.arrow_pub = rospy.Publisher("normal", Marker, queue_size=10)
        self.plane_pub = rospy.Publisher("plane", PointCloud2, queue_size=10)

        self.green_pub = rospy.Publisher("green", PointCloud2, queue_size=10)
        self.remove_rad_pub = rospy.Publisher("removed_rad", PointCloud2, queue_size=10)

        rospy.Subscriber("/plant_selector/mode", String, self.mode_change)

        self.frame_id = str(rospy.get_param("frame_id"))
        self.tfw = TF2Wrapper()

        # Victor Code
        self.victor = Victor()
        self.victor.set_control_mode(control_mode=ControlMode.JOINT_POSITION, vel=0.1)
        self.victor.connect()

        rospy.sleep(1)
        self.victor.open_left_gripper()
        rospy.sleep(1)
        self.victor.open_right_gripper()
        rospy.sleep(1)

        # Set the default mode to branch
        self.mode = "Branch"
        self.branch_pc_sub = rospy.Subscriber("/rviz_selected_points", PointCloud2, self.select_branch)
        self.weed_pc_sub = rospy.Subscriber("/rviz_selected_points", PointCloud2, self.select_weed)

        # Fixing first selection
        # TODO: This isn't ideal, probs a better way to do this
        ident_matrix = np.eye(4)
        for _ in range(10):
            self.tfw.send_transform_matrix(ident_matrix, parent=self.frame_id, child='end_effector_left')
            rospy.sleep(0.05)
            
    def mode_change(self, new_mode):
        """
        Callback to a new mode type from /plant_selector/mode. Modes can be either Branch or Weed.

        :param new_mode: Ros string message
        """
        self.mode = new_mode.data
        rospy.loginfo("New mode: " + self.mode)

    @staticmethod
    def project(u, n):
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
        :param color: color of the arrow
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
        hp.publish_pc_no_color(self.plane_pub, points_flat[:, :3], self.frame_id)

    def select_branch(self, selection):
        """
        This function selects the branch and gives a pose for the gripper.

        :param selection: selected pointcloud from rviz
        :return: None.
        """
        if self.mode != "Branch":
            return

        points_xyz = hp.cluster_filter(selection)[:, :3]

        # Transform open3d PC to numpy array

        # Create Open3D point cloud for green points
        pcd = o3d.geometry.PointCloud()
        # Save xyzrgb info in green_pcd (type: open3d.PointCloud)
        pcd.points = o3d.utility.Vector3dVector(points_xyz)

        # Apply plane segmentation function from open3d and get the best inliers
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

        # Get 3x3 rotation matrix
        # The first row is the x-axis of the tool frame in the camera frame
        camera2tool_rot = np.array([normal, cut_y, cut_direction_normalized]).T

        # Construct transformation matrix from camera to tool of end effector
        camera2tool = np.zeros([4, 4])
        camera2tool[:3, :3] = camera2tool_rot
        camera2tool[:3, 3] = inliers_centroid
        camera2tool[3, 3] = 1

        tfw = TF2Wrapper()
        # Get transformation matrix between tool and end effector
        tool2ee = tfw.get_transform(parent="left_tool", child="end_effector_left")
        # map2cam = tfw.get_transform(parent="map", child=self.frame_id)
        # Chain effect: get transformation matrix from camera to end effector
        camera2ee = camera2tool @ tool2ee  # Put map2cam first once we add in map part
        tfw.send_transform_matrix(camera2ee, parent=self.frame_id, child='end_effector_left')

        # Rviz commands
        # Call plot_pointcloud_rviz function to visualize PCs in Rviz
        hp.publish_pc_no_color(self.src_pub, points_xyz, self.frame_id)
        hp.publish_pc_no_color(self.inliers_pub, inlier_points, self.frame_id)
        # Call rviz_arrow function to see normal of the plane
        self.rviz_arrow(inliers_centroid, normal, name='normal', thickness=0.008, length_scale=0.15,
                        color='r')
        # Call plot_plane function to visualize plane in Rviz
        self.plot_plane(inliers_centroid, normal, size=0.1, res=0.001)

    def select_weed(self, selection):
        """
        This function extracts the weed points and gives a pose for the gripper.

        :param selection: Selected pointcloud in Rviz.
        :return: None.
        """
        if self.mode != "Weed":
            return

        # Load point cloud and visualize it
        points = np.array(list(pc2.read_points(selection)))

        if points.shape[0] == 0:
            rospy.loginfo("Select points")
            return

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
            rospy.loginfo("No green points found. Try again.")
            return

        # Save xyzrgb info in green_points (type: numpy array)
        green_points_xyz = pcd_points[green_points_indices]
        green_points_rgb = pcd_colors[green_points_indices]

        hp.publish_pc_no_color(self.green_pub, green_points_xyz, self.frame_id)

        # Create Open3D point cloud for green points
        green_pcd = o3d.geometry.PointCloud()
        # Save xyzrgb info in green_pcd (type: open3d.PointCloud)
        green_pcd.points = o3d.utility.Vector3dVector(green_points_xyz)
        green_pcd.colors = o3d.utility.Vector3dVector(green_points_rgb)

        # Apply radius outlier filter to green_pcd
        _, ind = green_pcd.remove_radius_outlier(nb_points=7, radius=0.007)

        if len(green_points_indices[0]) == 0:
            rospy.loginfo("Not enough points. Try again.")
            return

        # Just keep the inlier points in the point cloud
        green_pcd = green_pcd.select_by_index(ind)
        green_pcd_points = np.asarray(green_pcd.points)
        hp.publish_pc_no_color(self.remove_rad_pub, green_pcd_points, self.frame_id)

        # Apply DBSCAN to green points
        labels = np.array(green_pcd.cluster_dbscan(eps=0.007, min_points=15))  # This is actually pretty good

        # This is pretty good but struggles in some edge cases
        # clusterer = hdbscan.HDBSCAN(min_cluster_size=30, gen_min_span_tree=True, min_samples=1, allow_single_cluster=1)
        # # clusterer = hdbscan.HDBSCAN(min_cluster_size=8, min_samples=1, allow_single_cluster=1)
        # clusterer.fit(green_pcd_points)
        # labels = clusterer.labels_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters == 0:
            rospy.loginfo("Not enough points. Try again.")
            return

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

        [a, b, c, _] = plane_model
        # Just save and continue working with the inlier points defined by the plane segmentation function
        inlier_dirt_points = dirt_points_xyz[best_inliers]
        # Get centroid of dirt
        inlier_dirt_centroid = np.mean(inlier_dirt_points, axis=0)
        # The a, b, c coefficients of the plane equation are the components of the normal vector of that plane
        normal = np.asarray([a, b, c])

        # Currently only for zed
        if normal[2] > 0:
            normal = -normal

        phi = atan(normal[1] / normal[2])
        if phi < pi/2:
            phi = phi + pi - 2 * phi
        theta = atan(normal[0] / -normal[2])

        # Construct transformation matrix from camera to tool of end effector
        camera2tool = np.eye(4)
        camera2tool[:3, 3] = weed_centroid

        # Victor Stuff
        vicroot2cam = self.tfw.get_transform(parent='victor_root', child=self.frame_id)
        result = vicroot2cam @ camera2tool
        
        # plan_exec_res = self.victor.plan_to_pose(self.victor.right_arm_group, self.victor.right_tool_name, [result[0, 3], result[1, 3], result[2, 3], phi, -theta, 0])
        plan_exec_res = self.victor.plan_to_pose(self.victor.right_arm_group, self.victor.right_tool_name, [result[0, 3], result[1, 3], result[2, 3], phi, -theta, 0])
        was_success = plan_exec_res.planning_result.success
        if was_success == False:
            rospy.loginfo('Cant find path, will try with left arm.')
            
            plan_exec_res = self.victor.plan_to_pose(self.victor.left_arm_group, self.victor.left_tool_name, [result[0, 3], result[1, 3], result[2, 3], phi, -theta, 0])
            was_success = plan_exec_res.planning_result.success
            if was_success == False:
                rospy.loginfo("can't find a path")

        self.visualize_gripper(phi, theta, weed_centroid)
        self.publish_weed_debug_data(dirt_points_xyz, green_pcd_points, inlier_dirt_centroid, normal)


    def visualize_gripper(self, phi, theta, weed_centroid):
        # Generate Rotation Matrices
        rx = np.asarray([[1, 0, 0],
                            [0, cos(phi), -sin(phi)],
                            [0, sin(phi), cos(phi)]])
        ry = np.asarray([[cos(theta), 0, sin(theta)],
                            [0, 1, 0],
                            [-sin(theta), 0, cos(theta)]])

        # Combine
        frame_2_vector_rot = rx @ ry

        # Create camera to tool matrix
        camera_2_tool = np.eye(4)
        camera_2_tool[:3, :3] = frame_2_vector_rot
        camera_2_tool[:3, 3] = weed_centroid

        # Define transformation matrix from tool to end effector
        tool2ee = self.tfw.get_transform(parent="left_tool", child="end_effector_left")
        
        # Get transform from camera to end effector
        camera_2_ee = camera_2_tool @ tool2ee

        # Display gripper
        self.tfw.send_transform_matrix(camera_2_ee, parent=self.frame_id, child='end_effector_left')



    def publish_weed_debug_data(self, dirt_points_xyz, green_pcd_points, inlier_dirt_centroid, normal):
        hp.publish_pc_no_color(self.src_pub, dirt_points_xyz[:, :3], self.frame_id)
        # Visualize filtered green points as "inliers"
        hp.publish_pc_no_color(self.inliers_pub, green_pcd_points[:, :3], self.frame_id)
        # Call rviz_arrow function to see normal of the plane
        self.rviz_arrow(inlier_dirt_centroid, normal, name='normal', thickness=0.008, length_scale=0.15,
                        color='w')
        # Call plot_plane function to visualize plane in Rviz
        self.plot_plane(inlier_dirt_centroid, normal, size=0.1, res=0.001)


def main():
    """
    This main function constantly waits for any selection of points in Rviz.

    :return: None.
    """
    rospy.init_node("plant_extraction")
    plant_extractor = PlantExtractor()
    rospy.spin()


if __name__ == '__main__':
    main()
