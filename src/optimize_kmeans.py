from turtle import distance
import numpy as np
from scipy import optimize
import os
import clustering_tests as ct
import open3d as o3d


def get_segmentation(weights, training_directory, pc_filenames):
    pc_folder = training_directory + 'original_selections/'
    
    labels_list = []
    pc_list = []
    for file in pc_filenames:
            points = np.load(pc_folder + file)
            pcd_points = points[:, :3]
            float_colors = points[:, 3]
            pcd_colors = np.array((0, 0, 0))
            for x in float_colors:
                rgb = ct.float_to_rgb(x)
                pcd_colors = np.vstack((pcd_colors, rgb))
            pcd_colors = np.delete(pcd_colors, 0, 0)
            pcd_array = np.hstack((pcd_points, pcd_colors))

            cents, labels = ct.kmeans_from_scratch(pcd_array, 2, weights)
            labels_list.append(labels)
            pc_list.append(pcd_points)


    # seperate by number of points in segment and only return weeds
    weeds_list = []
    for i in range(len(labels_list)):
        if (np.count_nonzero(labels_list[i] == 1)) < (len(labels_list[i]) - np.count_nonzero(labels_list[i] == 1)):
            weeds_label = 1
        else:
            weeds_label = 0
        
        weed_idx = np.argwhere(labels_list[i] == weeds_label)
        weed = np.take(pc_list[i], weed_idx, axis=0)
        weeds_list.append(weed)

    return weeds_list


def calculate_cost(weights, training_directory, pc_filenames, manual_labels_filenames):
    weeds_list = get_segmentation(weights, training_directory, pc_filenames)

    manual_labels_folder = training_directory + 'manually_segmented/'
    manual_labels_list = []
    for file in manual_labels_filenames:
        pcd = o3d.io.read_point_cloud(manual_labels_folder + file)
        manual_labels_list.append(np.asarray(pcd.points))

    # Find the average distances between the two point clouds
    distances = []
    for i in range(len(weeds_list)):
        avg_weeds = np.mean(weeds_list[i], axis=0)
        avg_manuals = np.mean(manual_labels_list[i], axis=0)
        dist = np.linalg.norm(avg_manuals-avg_weeds)
        distances.append(dist)

    # Determine avg distances
    avg_dist = np.mean(distances)

    return avg_dist


def main():
    starting_weights = [0, 0, 0, 0, 0, 0]
    bounds = [(0, 10000), (0, 10000), (0, 10000), (0, 10000), (0, 10000), (0, 10000)]

    training_directory = "/home/amasse/catkin_ws/src/plant_selector/segmentation_training/"
    # pc_filenames = os.listdir(training_directory + 'original_selections/')

    pc_filenames = []
    manual_labels_filenames = []
    names_list = ['2', '5', '12', '14', '20', '22', '28', '30', '32', '33', '34', '36', '39' ,'40', '41']
    for i in range(len(names_list)):
        pc_filenames.append(''.join((names_list[i], ".npy")))
        manual_labels_filenames.append(''.join((names_list[i], ".ply")))

    # d = calculate_cost(starting_weights, training_directory, pc_filenames, manual_labels_filenames)

    results = dict()
    results['shgo'] = optimize.shgo(calculate_cost, bounds, args=(training_directory, pc_filenames, manual_labels_filenames), options={'disp': True})
    best_weights = results['shgo'].x
    print("The best weights are: ", best_weights)

    minimized_dist = calculate_cost(best_weights, training_directory, pc_filenames, manual_labels_filenames)
    print("The smallest distance is: ", minimized_dist)
    return 0


if __name__ == "__main__":
    main()

