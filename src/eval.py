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


def calculate_weed_centroid(list_files, centroids, path_pcs):
    row = 0
    for file in list_files:
        file = path_pcs + file
        points = np.load(file)

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
            return

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
            return

        # Just keep the inlier points in the point cloud
        green_pcd = green_pcd.select_by_index(ind)
        green_pcd_points = np.asarray(green_pcd.points)

        # Apply DBSCAN to green points
        labels = np.array(green_pcd.cluster_dbscan(eps=0.007, min_points=15))  # This is actually pretty good

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters == 0:
            print("Not enough points. Try again.")
            return
        # Get labels of the biggest cluster
        biggest_cluster_indices = np.where(labels[:] == mode(labels))
        # Just keep the points that correspond to the biggest cluster (weed)
        green_pcd_points = green_pcd_points[biggest_cluster_indices]

        # Get coordinates of the weed centroid
        weed_centroid = np.mean(green_pcd_points, axis=0)
        centroids[row, :] = weed_centroid
        row += 1

    return centroids


class WeedMetrics:
    def __init__(self, path_pcs, path_manual_labels, pred_model):
        """
                Initialize publishers for PCs, arrow and planes.
            """
        # Initialize publishers for PCs
        self.pc_pub = rospy.Publisher("point_cloud", PointCloud2, queue_size=10)
        self.centroid_pub = rospy.Publisher("centroid", PointCloud2, queue_size=10)
        self.stem_pub = rospy.Publisher("stem", PointCloud2, queue_size=10)

        self.list_files_pcs = os.listdir(path_pcs)
        self.list_files_manual_labels = os.listdir(path_manual_labels)

        self.path_pcs = path_pcs
        self.pred_model = pred_model
        self.error = None

        self.centroids = np.zeros(shape=(len(self.list_files_pcs), 3))
        self.manual_labels = np.zeros(shape=(len(self.list_files_manual_labels), 3))

        self.get_manual_labels(self.list_files_manual_labels, path_manual_labels)

        # self.frame_id = str(rospy.get_param("zed2i_left_camera_frame"))
        self.frame_id = "zed2i_left_camera_frame"

    def run_eval(self):
        pred_centroids = self.pred_model(self.list_files_pcs, self.centroids, self.path_pcs)
        self.compute_distances(pred_centroids)
        self.metric_printer()

        for sample in range(len(self.error)):
            file = self.path_pcs + self.list_files_pcs[sample]
            pc = np.load(file)
            mean_pc = np.mean(pc[:, :3], axis=0)
            pc_norm = pc
            pc_norm[:, :3] = pc[:, :3] - mean_pc
            centroid_norm = pred_centroids[sample].reshape(1, 3) - mean_pc
            manual_labels = self.manual_labels[sample].reshape(1, 3) - mean_pc

            hp.publish_pc_with_color(self.pc_pub, pc_norm, self.frame_id)
            hp.publish_pc_no_color(self.centroid_pub, centroid_norm, self.frame_id)
            hp.publish_pc_no_color(self.stem_pub, manual_labels, self.frame_id)
            input(f"Sample {sample + 1}")

    def compute_distances(self, centroids):
        self.error = np.linalg.norm(self.manual_labels[:, :3] - centroids[:, :3], axis=1)

    def get_manual_labels(self, list_files_manual_labels, path_manual_labels):
        row = 0
        for label in list_files_manual_labels:
            self.manual_labels[row, :3] = np.load(path_manual_labels + label)[:, :3]
            row += 1

    def metric_printer(self):
        print(f"Mean: {np.mean(self.error)}")
        print(f"Median: {np.median(self.error)}")
        print(f"Std: {np.std(self.error)}")


def main():
    rospy.init_node('weed_eval')

    # Create paths for pcs and manual labels
    path_pcs = "/home/miguel/catkin_ws/src/plant_selector/weed_eval/pcs/"
    path_manual_labels = "/home/miguel/catkin_ws/src/plant_selector/weed_eval/manual_labels/"

    # Run the evaluation
    evaluator = WeedMetrics(path_pcs, path_manual_labels, calculate_weed_centroid)
    evaluator.run_eval()


if __name__ == '__main__':
    main()
