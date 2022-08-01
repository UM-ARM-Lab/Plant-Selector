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
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker
from tf.transformations import euler_from_matrix, rotation_matrix
import hdbscan

# Val stuff
from arm_robots.hdt_michigan import Val
from geometry_msgs.msg import Pose


class PlantExtractor:
    def __init__(self):
        # Initialize publishers for PCs, arrow and planes
        self.src_pub = rospy.Publisher("source_pc", PointCloud2, queue_size=10)
        self.inliers_pub = rospy.Publisher("inliers_pc", PointCloud2, queue_size=10)
        self.arrow_pub = rospy.Publisher("normal", Marker, queue_size=10)
        self.plane_pub = rospy.Publisher("plane", PointCloud2, queue_size=10)

        rospy.Subscriber("/plant_selector/mode", String, self.mode_change)
        rospy.Subscriber("/plant_selector/verification", Bool, self.move_robot)

        self.frame_id = str(rospy.get_param("frame_id"))
        self.tfw = TF2Wrapper()

        # Val Code
        self.val = Val(raise_on_failure=True)
        self.val.connect()
        self.val.plan_to_joint_config('both_arms', 'bent')
        self.goal = None
        self.plan_exec_res = None

        rospy.sleep(1)
        self.val.open_left_gripper()
        rospy.sleep(1)
        self.val.open_right_gripper()
        rospy.sleep(1)

        # Set the default mode to branch
        self.mode = "Branch"
        self.plant_pc_sub = rospy.Subscriber("/rviz_selected_points", PointCloud2, self.plant_extraction)

    def mode_change(self, new_mode):
        self.mode = new_mode.data
        rospy.loginfo("New mode: " + self.mode)
        self.val.plan_to_joint_config('both_arms', 'bent')

    def move_robot(self, is_verified):
        if is_verified.data:
            self.val_execute()

    def plant_extraction(self, pc):
        if self.mode == "Branch":
            self.select_branch(pc)
        elif self.mode == "Weed":
            self.select_weed(pc)

    def select_branch(self, selection):
        # Perform Depth Filter
        points_xyz = hp.cluster_filter(selection)[:, :3]

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
        cut_direction = hp.project(camera_to_centroid, normal)
        # Normalize the projected vector
        cut_direction_normalized = cut_direction / np.linalg.norm(cut_direction)
        # Cross product between normalized cut director vector and the normal of the plane to obtain the
        # 2nd principal component
        cut_y = np.cross(cut_direction_normalized, normal)

        # Get 3x3 rotation matrix
        # The first row is the x-axis of the tool frame in the camera frame
        camera2tool_rot = np.array([normal, cut_y, cut_direction_normalized]).T

        # Visualize gripper. TODO: Can make all of this a lot cleaner
        # Construct transformation matrix from camera to tool of end effector
        camera2tool = np.zeros([4, 4])
        camera2tool[:3, :3] = camera2tool_rot
        camera2tool[:3, 3] = inliers_centroid
        camera2tool[3, 3] = 1

        # Visualize Point Clouds
        self.publish_pc_data(points_xyz, inlier_points, inliers_centroid, normal)
        self.visualize_gripper_urdf(camera2tool)

        # get transform from world to cam
        world2cam = self.tfw.get_transform(parent="world", child=self.frame_id)
        world2tool = world2cam @ camera2tool
        x_rot, y_rot, z_rot = euler_from_matrix(world2tool[:3, :3])

        val2cam = self.tfw.get_transform(parent='world', child=self.frame_id)
        result = val2cam @ camera2tool
        self.goal = [result[0, 3], result[1, 3], result[2, 3], x_rot, y_rot, z_rot]

        self.val_plan()

    def select_weed(self, selection):
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
            return

        # Save xyzrgb info in green_points (type: numpy array)
        green_points_xyz = green_points_xyz[green_points_indices]
        green_points_rgb = green_points_rgb[green_points_indices]

        # Create Open3D point cloud for green points
        green_pcd = o3d.geometry.PointCloud()
        # Save xyzrgb info in green_pcd (type: open3d.PointCloud)
        green_pcd.points = o3d.utility.Vector3dVector(green_points_xyz)
        green_pcd.colors = o3d.utility.Vector3dVector(green_points_rgb)

        # Apply radius outlier filter to green_pcd
        _, ind = green_pcd.remove_radius_outlier(nb_points=2, radius=0.007)

        if len(ind) == 0:
            rospy.loginfo("Not enough points. Try again.")
            return

        # Just keep the inlier points in the point cloud
        green_pcd = green_pcd.select_by_index(ind)
        green_pcd_points = np.asarray(green_pcd.points)

        # Apply DBSCAN to green points
        labels = np.array(green_pcd.cluster_dbscan(eps=0.007, min_points=15))  # This is actually pretty good

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
            return

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
        camera2tool[:3, :3] = (rotation_matrix(phi, np.asarray([1, 0, 0])) @
                               rotation_matrix(theta, np.asarray([0, 1, 0])))[:3, :3]
        camera2tool[:3, 3] = weed_centroid

        self.visualize_gripper_urdf(camera2tool)

        # Victor Stuff
        world2cam = self.tfw.get_transform(parent='world', child=self.frame_id)
        world2tool = world2cam @ camera2tool

        # Visualize pcs and gripper
        self.publish_pc_data(dirt_points_xyz, green_pcd_points, inlier_dirt_centroid, normal)

        # Plan to the pose
        x_rot, y_rot, z_rot = euler_from_matrix(world2tool[:3, :3])
        self.goal = [world2tool[0, 3], world2tool[1, 3], world2tool[2, 3], x_rot, y_rot, z_rot]

        self.val_plan()

    def val_plan(self):
        self.val.set_execute(False)
        # Find a plan and execute it
        self.plan_exec_res = self.val.plan_to_pose('right_side', self.val.right_tool_name, self.goal)
        was_success = self.plan_exec_res.planning_result.success
        # If there is no possible plan, try the left arm
        if was_success:
            print("Found a path!")
        else:
            rospy.loginfo("Can't find path.")
        
        # Return to zero state at the end, no matter what
        rospy.sleep(2)

    def val_execute(self):
        self.val.set_execute(True)
        exec_res = self.val.follow_arms_joint_trajectory(self.plan_exec_res.planning_result.plan.joint_trajectory)
        print(f"The execution was: {exec_res}")

        # Send the gripper away!
        end_effector_to_void = np.eye(4)
        end_effector_to_void[:3, 3] = 1000
        self.tfw.send_transform_matrix(end_effector_to_void, 'hdt_michigan_root', 'red_end_effector_left')

        # Grasping
        rospy.sleep(2)
        self.val.close_right_gripper()

        # Go back to base!
        rospy.sleep(5)
        self.val.plan_to_joint_config('both_arms', 'bent')

        rospy.sleep(2)
        self.val.open_right_gripper()

    def visualize_gripper_urdf(self, camera2tool):
        # Get transformation matrix between tool and end effector
        tool2ee = self.tfw.get_transform(parent="red_left_tool", child="red_end_effector_left")
        # Chain effect: get transformation matrix from camera to end effector
        camera2ee = camera2tool @ tool2ee
        self.tfw.send_transform_matrix(camera2ee, parent=self.frame_id, child='red_end_effector_left')

    def publish_pc_data(self, dirt_points_xyz, green_pcd_points, inlier_dirt_centroid, normal):
        # Visualize entire selected area
        hp.publish_pc_no_color(self.src_pub, dirt_points_xyz[:, :3], self.frame_id)
        # Visualize filtered green points as "inliers"
        hp.publish_pc_no_color(self.inliers_pub, green_pcd_points[:, :3], self.frame_id)
        # Call rviz_arrow function to see normal of the plane
        hp.rviz_arrow(self.arrow_pub, self.frame_id, inlier_dirt_centroid, normal, name='normal')
        # Call plot_plane function to visualize plane in Rviz
        hp.plot_plane(self.plane_pub, self.frame_id, inlier_dirt_centroid, normal)


def main():
    rospy.init_node("plant_extraction")
    plant_extractor = PlantExtractor()
    rospy.spin()


if __name__ == '__main__':
    main()
