"""
This script is for evaluation of the weed extraction algorithm.
"""
import os
from statistics import mode

import numpy as np
import open3d as o3d

import helpers as hp
import rospy
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker


def calculate_weed_centroid(points):
    # All the weed extraction algorithm
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
        print("No green points found. Try again.")
        return None

    # Save xyzrgb info in green_points (type: numpy array)
    green_points_xyz = pcd_points[green_points_indices]
    green_points_rgb = pcd_colors[green_points_indices]

    # Create Open3D point cloud for green points
    green_pcd = o3d.geometry.PointCloud()
    # Save xyzrgb info in green_pcd (type: open3d.PointCloud)
    green_pcd.points = o3d.utility.Vector3dVector(green_points_xyz)
    green_pcd.colors = o3d.utility.Vector3dVector(green_points_rgb)

    # Apply radius outlier filter to green_pcd
    _, ind = green_pcd.remove_radius_outlier(nb_points=7, radius=0.007)

    if len(green_points_indices[0]) == 0:
        print("Not enough points. Try again.")
        return None

    # Just keep the inlier points in the point cloud
    green_pcd = green_pcd.select_by_index(ind)
    green_pcd_points = np.asarray(green_pcd.points)

    # Apply DBSCAN to green points
    labels = np.array(green_pcd.cluster_dbscan(eps=0.007, min_points=15))  # This is actually pretty good

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters == 0:
        print("Not enough points. Try again.")
        return None
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
    dirt_pcd_send = o3d.geometry.PointCloud()
    # Save points and color to the point cloud
    dirt_pcd_send.points = o3d.utility.Vector3dVector(inlier_dirt_points)
    # The a, b, c coefficients of the plane equation are the components of the normal vector of that plane
    if a < 0:
        normal = np.asarray([a, b, c])
    else:
        normal = -np.asarray([a, b, c])

    return weed_centroid, normal


class WeedMetrics:
    def __init__(self, data_directory, pred_model):
        """
                Initialize publishers for PCs, arrow and planes.
            """
        # Initialize publishers for PCs
        self.pc_pub = rospy.Publisher("point_cloud", PointCloud2, queue_size=10)
        self.centroid_pub = rospy.Publisher("centroid", PointCloud2, queue_size=10)
        self.stem_pub = rospy.Publisher("stem", PointCloud2, queue_size=10)
        self.arrow_pub = rospy.Publisher("normal_rotations", Marker, queue_size=10)

        self.data_directory = data_directory
        self.pc_filenames = os.listdir(self.data_directory + 'pcs/')
        self.manual_labels = None
        self.get_manual_labels()

        self.pred_model = pred_model
        self.error = None
        self.skipped_weeds = 0
        self.skipped_weeds_filenames = []

        self.frame_id = "zed2i_left_camera_frame"

    def run_eval(self):
        # Start with an empty array
        pred_stems = []
        normals = []
        parent_directory = self.data_directory + "pcs/"
        # Fill the array with predicted stem
        for file in self.pc_filenames:
            points = np.load(parent_directory + file)
            stem_pred, normal = self.pred_model(points)

            # if there is a valid prediction, add it, otherwise, ignore
            if stem_pred is not None:
                pred_stems.append(stem_pred)
                normals.append(normal)
            else:
                # We did not get a prediction, ignore this case
                self.skipped_weeds += 1
                self.skipped_weeds_filenames.append(file)

        pred_stems = np.asarray(pred_stems)
        normals = np.asarray(normals)

        # Remove the files in the array of file names that were unable to make a prediction
        # Remove the stems of weeds that are associated with a point cloud that couldn't make a stem prediction
        i = 0
        for pc_file in self.pc_filenames:
            if pc_file in self.skipped_weeds_filenames:
                self.pc_filenames.remove(pc_file)
                self.manual_labels = np.delete(self.manual_labels, i, 0)
            else:
                i += 1

        # Reshape
        self.manual_labels = self.manual_labels.reshape(self.manual_labels.shape[0], 3)

        self.compute_distances(pred_stems)
        self.metric_printer()

        for sample in range(len(self.error)):
            mat = hp.rotation_matrix_from_vectors(normals[sample], np.asarray([0, 0, 1]))
            normal_rot = mat.dot(normals[sample])

            file = parent_directory + self.pc_filenames[sample]
            pc = np.load(file)
            mean_pc = np.mean(pc[:, :3], axis=0)
            pc_norm = pc
            pc_norm[:, :3] = pc[:, :3] - mean_pc
            pc_trans = np.transpose(pc_norm[:, :3])
            for point in range(pc_trans.shape[1]):
                pc_norm[point, :3] = np.matmul(mat, pc_trans[:, point])
            pred_stem_norm = pred_stems[sample].reshape(1, 3) - mean_pc
            true_stem_norm = self.manual_labels[sample].reshape(1, 3) - mean_pc

            hp.publish_pc_with_color(self.pc_pub, pc_norm, self.frame_id)
            hp.publish_pc_no_color(self.centroid_pub, pred_stem_norm, self.frame_id)
            hp.publish_pc_no_color(self.stem_pub, true_stem_norm, self.frame_id)
            hp.rviz_arrow(self.frame_id, self.arrow_pub, np.asarray([0, 0, 0]), normals[sample], name='normal',
                          thickness=0.008, length_scale=0.15, color='r')
            hp.rviz_arrow(self.frame_id, self.arrow_pub, np.asarray([0, 0, 0]), normal_rot, name='normal_rot',
                          thickness=0.008, length_scale=0.15, color='b')
            input(f"Currently viewing {str(self.pc_filenames[sample])}. Press enter to see next sample.")

    def compute_distances(self, centroids):
        self.error = np.linalg.norm(self.manual_labels[:, :3] - centroids[:, :3], axis=1)

    def get_manual_labels(self):
        parent_directory = self.data_directory + "manual_labels/"
        true_stem_filenames = os.listdir(parent_directory)

        self.manual_labels = []
        for label in true_stem_filenames:
            self.manual_labels.append(np.load(parent_directory + label)[:, :3])
        self.manual_labels = np.asarray(self.manual_labels)

    def metric_printer(self):
        print(f"Mean: {np.mean(self.error)}")
        print(f"Median: {np.median(self.error)}")
        print(f"Std: {np.std(self.error)}")

        if self.skipped_weeds != 0:
            print(f"Unable to make predictions on the following {self.skipped_weeds} files:")
            print(self.skipped_weeds_filenames)


def main():
    rospy.init_node('weed_eval')

    # Create paths for pcs and manual labels
    data_directory = "/home/miguel/catkin_ws/src/plant_selector/weed_eval/"

    # Run the evaluation
    evaluator = WeedMetrics(data_directory, calculate_weed_centroid)
    evaluator.run_eval()


if __name__ == '__main__':
    main()
