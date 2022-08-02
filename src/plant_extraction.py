#!/usr/bin/env python
# Import useful libraries and functions
from math import atan, sin, cos, pi

import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA

import helpers as hp
import plant_helpers as ph
import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from std_msgs.msg import Bool
from tf.transformations import euler_from_matrix, rotation_matrix
import argparse
import sys

# Robot stuff
from arm_robots.hdt_michigan import Val
from arm_robots.victor import Victor
from victor_hardware_interface_msgs.msg import ControlMode


class PlantExtractor:
    def __init__(self, robot):
        rospy.Subscriber("/plant_selector/mode", String, self.mode_change)
        rospy.Subscriber("/plant_selector/verification", Bool, self.move_robot)

        self.frame_id = str(rospy.get_param("frame_id"))
        self.tfw = TF2Wrapper()

        self.robot = robot
        if self.robot is not None:
            self.robot.connect()
        self.default_pose = str(rospy.get_param("default_pose"))
        self.robot_to_default_pose()
        self.ask_for_verif_pub = rospy.Publisher("/plant_selector/ask_for_verification", Bool, queue_size=10)

        self.goal = None
        self.plan_exec_res = None

        # Set the default mode to branch
        self.mode = "Weed"
        rospy.Subscriber("/rviz_selected_points", PointCloud2, self.plant_extraction)

        rospy.sleep(1)
        self.hide_red_gripper()

    def mode_change(self, new_mode):
        self.mode = new_mode.data
        rospy.loginfo("New mode: " + self.mode)
        self.robot_to_default_pose()

    def move_robot(self, is_verified):
        if is_verified.data:
            self.robot_execute()

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

        # Construct transformation matrix from camera to tool of end effector
        camera2tool = np.zeros([4, 4])
        camera2tool[:3, :3] = camera2tool_rot
        camera2tool[:3, 3] = inliers_centroid
        camera2tool[3, 3] = 1

        self.visualize_red_gripper(camera2tool)
        self.robot_plan(camera2tool)

    def select_weed(self, selection):
        # Load point cloud and visualize it
        points = np.array(list(pc2.read_points(selection)))

        if points.shape[0] == 0:
            rospy.loginfo("Select points")
            return

        weed_centroid, normal = ph.calculate_weed_centroid(points)

        if weed_centroid is None:
            return

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

        self.visualize_red_gripper(camera2tool)
        self.robot_plan(camera2tool)

    def robot_plan(self, camera2tool):
        if self.robot is None:
            return

        # figure out gripper pose in world frame
        world2cam = self.tfw.get_transform(parent='world', child=self.frame_id)
        world2tool = world2cam @ camera2tool

        # Plan to the pose
        x_rot, y_rot, z_rot = euler_from_matrix(world2tool[:3, :3])

        self.goal = [world2tool[0, 3], world2tool[1, 3], world2tool[2, 3], x_rot, y_rot, z_rot]
        self.robot.set_execute(False)
        # Find a plan and execute it
        self.plan_exec_res = self.robot.plan_to_pose(self.robot.right_arm_group, self.robot.right_tool_name, self.goal)
        was_success = self.plan_exec_res.planning_result.success

        if was_success:
            # Send a message to rviz panel which prompts the user to verify the execution plan
            msg = Bool()
            msg.data = True
            self.ask_for_verif_pub.publish(msg)
        else:
            rospy.loginfo("Can't find path.")

    def robot_execute(self):
        if self.robot is None:
            return

        # Attempt to go to the goal
        self.robot.set_execute(True)
        exec_res = self.robot.follow_arms_joint_trajectory(self.plan_exec_res.planning_result.plan.joint_trajectory)
        print(f"The execution was: {exec_res.success}")

        # Grasping
        rospy.sleep(2)
        self.robot.close_right_gripper()

        rospy.sleep(5)
        self.robot_to_default_pose()

    def robot_to_default_pose(self):
        self.hide_red_gripper()
        if self.robot is None:
            return

        # Go back to default!
        self.robot.set_execute(True)
        self.robot.plan_to_joint_config('both_arms', self.default_pose)

        rospy.sleep(1)
        self.robot.open_right_gripper()
        rospy.sleep(1)
        self.robot.open_left_gripper()

    def visualize_red_gripper(self, camera2tool):
        # Get transformation matrix between tool and end effector
        tool2ee = self.tfw.get_transform(parent="red_left_tool", child="red_end_effector_left")
        # Chain effect: get transformation matrix from camera to end effector
        camera2ee = camera2tool @ tool2ee
        self.tfw.send_transform_matrix(camera2ee, parent=self.frame_id, child='red_end_effector_left')

    def hide_red_gripper(self):
        # Because you can't directly hide a urdf, just send the urdf very far away :D
        end_effector_to_void = np.eye(4)
        end_effector_to_void[:3, 3] = 1000
        self.tfw.send_transform_matrix(end_effector_to_void, 'zed2i_base_link', 'red_end_effector_left')

    # def publish_pc_data(self, dirt_points_xyz, green_pcd_points, inlier_dirt_centroid, normal):
    #     # Visualize entire selected area
    #     hp.publish_pc_no_color(self.src_pub, dirt_points_xyz[:, :3], self.frame_id)
    #     # Visualize filtered green points as "inliers"
    #     hp.publish_pc_no_color(self.inliers_pub, green_pcd_points[:, :3], self.frame_id)
    #     # Call rviz_arrow function to see normal of the plane
    #     hp.rviz_arrow(self.arrow_pub, self.frame_id, inlier_dirt_centroid, normal, name='normal')
    #     # Call plot_plane function to visualize plane in Rviz
    #     hp.plot_plane(self.plane_pub, self.frame_id, inlier_dirt_centroid, normal)


def main():
    rospy.init_node("plant_extraction")

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--robot', type=str)
    args = parser.parse_args(rospy.myargv(sys.argv[1:]))

    robot = None
    if args.robot == "Val":
        robot = Val(raise_on_failure=True)
    elif args.robot == "Victor":
        robot = Victor()
        robot.set_control_mode(control_mode=ControlMode.JOINT_POSITION, vel=0.1)

    plant_extractor = PlantExtractor(robot)
    rospy.spin()


if __name__ == '__main__':
    main()
