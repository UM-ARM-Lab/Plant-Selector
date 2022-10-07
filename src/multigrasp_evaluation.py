import os
from turtle import fd

import numpy as np
import open3d as o3d
import pandas as pd
import csv

import plant_modeling as pm
import clustering_tests as ct


def evaluate_predictions(all_preds, manual_labels):
    '''!
    Compares predicitons to manual labels, calculates metrics

    @param all_preds   a list of arrays containting predicted centroids
    @param algorithm   a list of arrays containing actual centroids

    @return dataframe
    '''

    gripper_size = 0.015
    success_threshold = gripper_size / 2.0

    total_weeds = 0
    successful_grasps = 0
    attempted_grasps = 0
    successful_errors = []
    for i in range(len(all_preds)):
        total_weeds += manual_labels[i].shape[0]
        attempted_grasps += all_preds[i].shape[0]

        for j in range(all_preds[i].shape[0]):
            current_pred = all_preds[i][j, :]

            for k in range(manual_labels[i].shape[0]):
                current_label = manual_labels[i][k, :]
                current_error = np.linalg.norm(current_label - current_pred)
                if (abs(current_error) <= success_threshold):
                    successful_grasps += 1
                    successful_errors.append(current_error)
    data = [(total_weeds, attempted_grasps, successful_grasps, successful_grasps/attempted_grasps, successful_grasps/total_weeds, np.mean(successful_errors))]
    df = pd.DataFrame(data, columns=['TotalWeeds', 'Attempts', 'Successes', 'AttemptsSuccessRate', 'WeedsSuccessRate', 'MeanError'])
    return df


def main():
    display = False
    
    data_directory = '/home/amasse/catkin_ws/src/plant_selector/weed_eval/'
    pcs_folder = 'multi_pcs/'
    manual_labels_folder = 'multi_manual_labels/'

    pc_filenames = ['1_multi.npy','2_multi.npy','3_multi.npy','4_multi.npy']
    label_filenames = ['1_multi_label.npy','2_multi_label.npy','3_multi_label.npy','4_multi_label.npy']

    algorithms = ['kmeans-optimized', 'kmeans-redmean', 'kmeans-euclidean', 'bi-kmeans', 'spectral', 'ward', 'npc']
    # algorithms = ['color-segmentation']

    all_dfs = []
    for alg in algorithms:
        pcds = []
        all_preds = []
        for file in pc_filenames:
            points = np.load(data_directory + pcs_folder + file)
            pcd, array, colors = ct.array_2_pc(points)
            pcds.append(pcd)
            # stem_preds, normal = ct.DBSCAN_calculate_pose(points, algorithm=alg, return_multiple_grasps=True)
            stem_preds, normal = pm.calculate_weed_centroid(points, return_multiple_grasps=True)
            stem_preds = np.asarray(stem_preds)
            all_preds.append(stem_preds)
        

        manual_labels = []
        for file in label_filenames:
            manual_labels.append(np.load(data_directory + manual_labels_folder + file))
        
        df = evaluate_predictions(all_preds, manual_labels)
        df["Method"] = alg
        all_dfs.append(df)

    full_frame = pd.concat(all_dfs)
    full_frame.to_csv('/home/amasse/catkin_ws/src/plant_selector/weed_eval/multi_grasp_data.csv')

    if display == True:
        for i in range(len(pc_filenames)):
            preds_pcd = o3d.geometry.PointCloud()
            preds_pcd.points = o3d.utility.Vector3dVector(all_preds[i])
            preds_pcd.paint_uniform_color([1, 0.1, 0.1])
            manual_pcd = o3d.geometry.PointCloud()
            manual_pcd.points = o3d.utility.Vector3dVector(manual_labels[i])
            manual_pcd.paint_uniform_color([0.1, 0.1, 1])
            o3d.visualization.draw_geometries([pcds[i], preds_pcd, manual_pcd])

    return 0


if __name__ == "__main__":
    main()