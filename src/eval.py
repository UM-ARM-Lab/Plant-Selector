#!/usr/bin/env python
import os

import numpy as np
import csv
import rospy
from sensor_msgs.msg import PointCloud2

import argparse
import sys

import plant_modeling as pm
import find_centroids as fc
import rviz_helpers as rh


class WeedMetrics:
    def __init__(self, data_directory, pred_model, gripper_size, return_multiple_grasps):
        # Initialize publishers for PCs
        self.pc_pub = rospy.Publisher("point_cloud", PointCloud2, queue_size=10)
        self.centroid_pub = rospy.Publisher("centroid", PointCloud2, queue_size=10)
        self.stem_pub = rospy.Publisher("stem", PointCloud2, queue_size=10)

        self.data_directory = data_directory
        self.pc_filenames = os.listdir(self.data_directory + 'single_pcs/')
        self.label_names_folder = 'single_manual_labels/'
        self.pcs_names_folder = 'single_pcs/'
        self.manual_labels = None

        self.gripper_size = gripper_size
        self.pred_model = pred_model
        self.error = None
        self.successes = None

        self.last_metrics = []
        self.best_metrics = []
        self.current_metrics = []

        self.basepath = os.path.dirname(__file__)

        self.last_file = os.path.abspath(os.path.join(self.basepath, "..", "weed_eval/past_metrics/eval_last.csv"))
        self.best_file = os.path.abspath(os.path.join(self.basepath, "..", "weed_eval/past_metrics/eval_best.csv"))
        self.current_file = os.path.abspath(os.path.join(self.basepath, "..", "weed_eval/past_metrics/color.csv"))
        self.all_data_file = os.path.abspath(os.path.join(self.basepath, "..", "weed_eval/test_eval.csv"))

        self.skipped_weeds = 0
        self.skipped_weeds_filenames = []

        # This is an arbitrary camera frame because we are publishing the point clouds ourselves.
        self.frame_id = "zed2i_left_camera_frame"

    def run_eval(self):
        # Start with an empty array
        good_pc_filenames = []
        pred_stems = []
        normals = []
        pc_parent_directory = self.data_directory + self.pcs_names_folder
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
        man_labels_parent_directory = self.data_directory + self.label_names_folder
        for filename in good_pc_filenames:
            self.manual_labels.append(np.load(man_labels_parent_directory + filename)[:, :3])
        self.manual_labels = np.asarray(self.manual_labels)
        self.manual_labels = self.manual_labels.reshape(self.manual_labels.shape[0], 3)

        self.compute_distances(pred_stems)
        self.metric_printer()

        # with open(self.all_data_file, 'w', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(self.pc_filenames)
        #     writer.writerow(self.manual_labels)
        #     writer.writerow(pred_stems)
        #     writer.writerow(self.error)

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
        parent_directory = self.data_directory + self.label_names_folder
        true_stem_filenames = os.listdir(parent_directory)

        self.manual_labels = []
        for label in true_stem_filenames:
            self.manual_labels.append(np.load(parent_directory + label)[:, :3])
        self.manual_labels = np.asarray(self.manual_labels)


    def compare_previous(self):
        # extract last and best metrics from files
        self.last_metrics = np.genfromtxt(self.last_file, delimiter=',')[1,:]
        self.best_metrics = np.genfromtxt(self.best_file, delimiter=',')[1,:]

        self.current_metrics = np.array([np.mean(self.error), np.median(self.error), np.std(self.error), self.successes, 
            (len(self.error) - self.successes), (self.successes / len(self.error))])
     

        # compare last and best info to current info, first row is improvement from last, second is improvement from best
        self.improvement = np.zeros((2,6))
        for i in range(self.improvement.shape[0]):
            for j in range(self.improvement.shape[1]):
                
                if i == 0:
                    if j == 3 or j == 4:
                        self.improvement[i, j] = self.current_metrics[j] - self.last_metrics[j]
                    else:
                        self.improvement[i, j] = 100 * (self.current_metrics[j] - self.last_metrics[j]) / self.last_metrics[j]
                else:
                    if j == 3 or j == 4:
                        self.improvement[i, j] = self.current_metrics[j] - self.best_metrics[j]
                    else:
                        self.improvement[i, j] = 100 * (self.current_metrics[j] - self.best_metrics[j]) / self.best_metrics[j]
        return 0
    

    def update_metrics(self):
        labels = ["mean", "median", "std", "successes", "failures", "success rate"]
        
        # if current mean is better than best, replace best with current
        if self.best_metrics[0] > self.current_metrics[0]:
            self.best_metrics = self.current_metrics
            with open(self.best_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(labels)
                writer.writerow(self.current_metrics)

        
        # replace last metrics with current
        self.last_metrics = self.current_metrics
        with open(self.last_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(labels)
            writer.writerow(self.current_metrics)

        # # Save current metrics
        # with open(self.current_file, 'w', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(labels)
        #     writer.writerow(self.current_metrics)


    def metric_printer(self):
        self.compare_previous()
        
        print("--------------------------------------------------")
        print("METRICS (in meters)")
        print("--------------------------------------------------")
        print(f"Mean: {self.current_metrics[0]}")
        print(f"\t{round(self.improvement[0,0], 4)}% change from last time")
        print(f"\t{round(self.improvement[1,0], 4)}% change from best trial")

        print(f"Median: {self.current_metrics[1]}")
        print(f"\t{round(self.improvement[0,1], 4)}% change from last time")
        print(f"\t{round(self.improvement[1,1], 4)}% change from best trial")

        print(f"Std: {self.current_metrics[2]}")
        print(f"\t{round(self.improvement[0,2], 4)}% change from last time")
        print(f"\t{round(self.improvement[1,2], 4)}% change from best trial\n")


        print(f"Theoretical Successes: {self.current_metrics[3]}")
        print(f"\tLast trial there were {self.last_metrics[3]} successes")
        print(f"\tThe best trial had {self.best_metrics[3]} successes")

        print(f"Theoretical Failures: {self.current_metrics[4]}")
        print(f"\tLast trial there were {self.last_metrics[4]} failures")
        print(f"\tThe best trial had {self.best_metrics[4]} failures")

        print(f"Theoretical Success Rate: {self.current_metrics[5]}")
        print(f"\t{round(self.improvement[0,5], 4)}% change from last time")
        print(f"\t{round(self.improvement[1,5], 4)}% change from best trial")
        print("--------------------------------------------------")

        self.update_metrics()

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
    # evaluator = WeedMetrics(args.weed_directory, fc.color_calculate_pose, gripper_size=0.015)
    # evaluator = WeedMetrics(args.weed_directory, fc.DBSCAN_calculate_pose, gripper_size=0.015, return_multiple_grasps=False)
    evaluator = WeedMetrics(args.weed_directory, fc.FRG_calculate_pose, gripper_size=0.015, return_multiple_grasps=False)
    evaluator.run_eval()


if __name__ == '__main__':
    main()
