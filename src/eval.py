#!/usr/bin/env python
import os

import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2

import argparse
import sys

import plant_modeling as pm
import rviz_helpers as rh


class WeedMetrics:
    def __init__(self, data_directory, pred_model, gripper_size):
        # Initialize publishers for PCs
        self.pc_pub = rospy.Publisher("point_cloud", PointCloud2, queue_size=10)
        self.centroid_pub = rospy.Publisher("centroid", PointCloud2, queue_size=10)
        self.stem_pub = rospy.Publisher("stem", PointCloud2, queue_size=10)

        self.data_directory = data_directory
        self.pc_filenames = os.listdir(self.data_directory + 'pcs/')
        self.manual_labels = None

        self.gripper_size = gripper_size
        self.pred_model = pred_model
        self.error = None
        self.successes = None

        self.skipped_weeds = 0
        self.skipped_weeds_filenames = []

        # This is an arbitrary camera frame because we are publishing the point clouds ourselves.
        self.frame_id = "zed2i_left_camera_frame"

    def run_eval(self):
        # Start with an empty array
        good_pc_filenames = []
        pred_stems = []
        normals = []
        pc_parent_directory = self.data_directory + "pcs/"
        # Fill the array with predicted stem
        for file in self.pc_filenames:
            points = np.load(pc_parent_directory + file)
            stem_pred, normal = self.pred_model(points)

            # if there is a valid prediction, add it, otherwise, ignore
            if stem_pred is not None:
                pred_stems.append(stem_pred)
                normals.append(normal)
                good_pc_filenames.append(file)
            else:
                # We did not get a prediction, ignore this case
                self.skipped_weeds += 1
                self.skipped_weeds_filenames.append(file)

        pred_stems = np.asarray(pred_stems)
        normals = np.asarray(normals)

        # Remove the files in the array of file names that were unable to make a prediction
        # Remove the stems of weeds that are associated with a point cloud that couldn't make a stem prediction
        self.manual_labels = []
        man_labels_parent_directory = self.data_directory + "manual_labels/"
        for filename in good_pc_filenames:
            self.manual_labels.append(np.load(man_labels_parent_directory + filename)[:, :3])
        self.manual_labels = np.asarray(self.manual_labels)
        self.manual_labels = self.manual_labels.reshape(self.manual_labels.shape[0], 3)

        self.compute_distances(pred_stems)
        self.metric_printer()

        for sample in range(len(self.error)):
            file = pc_parent_directory + good_pc_filenames[sample]
            pc = np.load(file)
            mean_pc = np.mean(pc[:, :3], axis=0)
            pc_norm = pc
            pc_norm[:, :3] = pc[:, :3] - mean_pc
            pc_trans = np.transpose(pc_norm[:, :3])

            # This rotation makes the plane of the dirt always be pointing in the z direction which makes things easier to view
            mat = pm.rotation_matrix_from_vectors(normals[sample], np.asarray([0, 0, 1]))
            for point in range(pc_trans.shape[1]):
                pc_norm[point, :3] = np.matmul(mat, pc_trans[:, point])
            pred_stem_norm = np.matmul(mat, np.transpose(pred_stems[sample].reshape(1, 3) - mean_pc)).transpose()
            true_stem_norm = np.matmul(mat, np.transpose(self.manual_labels[sample].reshape(1, 3) - mean_pc)).transpose()

            rh.publish_pc_with_color(self.pc_pub, pc_norm, self.frame_id)
            rh.publish_pc_no_color(self.centroid_pub, pred_stem_norm, self.frame_id)
            rh.publish_pc_no_color(self.stem_pub, true_stem_norm, self.frame_id)
            input(f"Currently viewing {str(good_pc_filenames[sample])}. Error of {self.error[sample]}.\n"
                  f"Press enter to see next sample.")

        # Clear out the prediction/actual to clear up rviz
        rh.publish_pc_no_color(self.centroid_pub, np.array([]), self.frame_id)
        rh.publish_pc_no_color(self.stem_pub, np.array([]), self.frame_id)

        print("\n\n\n\nNow viewing the cases where a weed stem could not be predicted.")
        for no_pred_weed in self.skipped_weeds_filenames:
            file = pc_parent_directory + no_pred_weed
            pc = np.load(file)
            mean_pc = np.mean(pc[:, :3], axis=0)
            pc_norm = pc
            pc_norm[:, :3] = pc[:, :3] - mean_pc
            rh.publish_pc_with_color(self.pc_pub, pc_norm, self.frame_id)
            input(f"Currently viewing {no_pred_weed}. Could not make a prediction.")

    def compute_distances(self, centroids):
        self.error = np.linalg.norm(self.manual_labels[:, :3] - centroids[:, :3], axis=1)
        self.successes = np.sum(np.where(self.error < self.gripper_size / 2, 1, 0))

    def get_manual_labels(self):
        parent_directory = self.data_directory + "manual_labels/"
        true_stem_filenames = os.listdir(parent_directory)

        self.manual_labels = []
        for label in true_stem_filenames:
            self.manual_labels.append(np.load(parent_directory + label)[:, :3])
        self.manual_labels = np.asarray(self.manual_labels)

    def metric_printer(self):
        print("-------------------------------------")
        print("METRICS")
        print("-------------------------------------")
        print(f"Mean: {np.mean(self.error)}")
        print(f"Median: {np.median(self.error)}")
        print(f"Std: {np.std(self.error)}\n")
        print(f"Theoretical Successes: {str(self.successes)}")
        print(f"Theoretical Failures: {str(len(self.error) - self.successes )}")
        print(f"Theoretical Success Rate: {str(self.successes / len(self.error))}")
        print("-------------------------------------")

        print(f"\n\nUnable to make predictions on the following {self.skipped_weeds} files:")
        if self.skipped_weeds != 0:
            print(self.skipped_weeds_filenames, "\n\n")


# To run this node, make sure to give an argument of the location of the parent directory of data which should have a
# sub folders of both 'manual_labels' and 'pcs' which should be filled with weed data.
def main():
    rospy.init_node('weed_eval')

    parser = argparse.ArgumentParser()
    parser.add_argument('weed_directory', type=str)
    args = parser.parse_args(rospy.myargv(sys.argv[1:]))

    # Run the evaluation
    evaluator = WeedMetrics(args.weed_directory, pm.calculate_weed_centroid, gripper_size=0.015)
    evaluator.run_eval()


if __name__ == '__main__':
    main()
