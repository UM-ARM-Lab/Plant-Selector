#!/usr/bin/env python
# Import useful libraries and functions
from statistics import mode

import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA

import helpers as hp
import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from tf.transformations import rotation_matrix
from visualization_msgs.msg import Marker


class PlantExtractor:
    def __init__(self):
        """
        Initialize publishers for PCs, arrow and planes.
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

        # Display gripper
        self.visualize_gripper(inliers_centroid, normal)

        # Publish rviz data
        self.publish_debug_data(points_xyz, inlier_points, inliers_centroid, normal)

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

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters == 0:
            rospy.loginfo("Not enough points. Try again.")
            return

        # Get labels of the biggest cluster
        biggest_cluster_indices = np.where(labels[:] == mode(labels))
        # Just keep the points that correspond to the biggest cluster (weed)
        green_pcd_points = green_pcd_points[biggest_cluster_indices]
        # Get coordinates of the weed centroid
        # print(f"x-distance: {np.mean(green_pcd_points, axis=0)[0]}")

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
        dirt_pcd_send = o3d.geometry.PointCloud()
        # Save points and color to the point cloud
        dirt_pcd_send.points = o3d.utility.Vector3dVector(inlier_dirt_points)
        # The a, b, c coefficients of the plane equation are the components of the normal vector of that plane
        if a < 0:
            normal = np.asarray([a, b, c])
        else:
            normal = -np.asarray([a, b, c])

        # Display gripper
        self.visualize_gripper(weed_centroid, normal)

        # Publish rviz data
        self.publish_debug_data(dirt_points_xyz, green_pcd_points, inlier_dirt_centroid, normal)

    def visualize_gripper(self, camera2target, normal):
        # Call the project function to get the cut direction vector
        cut_direction = hp.project(camera2target, normal)
        # Normalize the projected vector
        cut_direction_normalized = cut_direction / np.linalg.norm(cut_direction)
        # Cross product between normalized cut director vector and the normal of the plane to obtain the
        # 2nd principal component
        cut_y = np.cross(cut_direction_normalized, normal)

        # Get 3x3 rotation matrix
        # The first row is the x-axis of the tool frame in the camera frame
        camera2tool_rot = np.array([normal, cut_y, cut_direction_normalized]).T

        # Construct transformation matrix from camera to tool of end effector
        camera2tool = np.eye(4)
        camera2tool[:3, :3] = camera2tool_rot

        # If selecting weed, apply a dynamic angle for the robot
        if self.mode == 'Weed':
            max_len_arm, angle = 1, -60
            # Get the rotation matrix from the parameters above
            rotation = rotation_matrix(np.deg2rad(camera2target[0] * angle / max_len_arm), cut_y, point=camera2target)
            # Get new camera2tool matrix
            camera2tool = rotation @ camera2tool
        # Assign translation for the transformation matrix
        camera2tool[:3, 3] = camera2target

        # Define transformation matrix from tool to end effector
        tool2ee = self.tfw.get_transform(parent="left_tool", child="end_effector_left")

        # Get transform from camera to end effector
        camera2ee = camera2tool @ tool2ee

        # Display gripper
        self.tfw.send_transform_matrix(camera2ee, parent=self.frame_id, child='end_effector_left')

    def publish_debug_data(self, source, inliers, origin, normal):
        hp.publish_pc_no_color(self.src_pub, source[:, :3], self.frame_id)
        hp.publish_pc_no_color(self.inliers_pub, inliers[:, :3], self.frame_id)
        hp.rviz_arrow(self.frame_id, self.arrow_pub, origin, normal, name='normal', thickness=0.008, length_scale=0.15,
                      color='w')
        hp.plot_plane(self.frame_id, self.plane_pub, origin, normal, size=0.1, res=0.001)


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
