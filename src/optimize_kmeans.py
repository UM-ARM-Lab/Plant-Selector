"""!
@brief Uses training data to generate weights for kmeans distance function that will create "optimal" solutions
"""

from turtle import distance
import numpy as np
from scipy import optimize
import os
import find_centroids as fc
import open3d as o3d


def get_segmentation(weights, training_directory, pc_filenames):
    '''!
    Uses given weights and training data to do weighted kmeans.

    @param weights    a list containing 6 weights for weighted kmeans implementation. weights must be >= 0
    @param training_directory   a string with the directory to the training data
    @param pc_filenames    a list of strings containing the names of the pointclouds being used for training. eg: "79.ply"

    @return list of weeds where each element contains all weeds from the corresponding file
    '''
    pc_folder = training_directory + '/training_selections/'
    
    # For every file, load it and do weighted kmeans. Put the labels and point clouds into a list
    labels_list = []
    pc_list = []
    for file in pc_filenames:
            points = np.load(pc_folder + file)
            pcd_points = points[:, :3]
            float_colors = points[:, 3]
            pcd_colors = np.array((0, 0, 0))
            for x in float_colors:
                rgb = fc.float_to_rgb(x)
                pcd_colors = np.vstack((pcd_colors, rgb))
            pcd_colors = np.delete(pcd_colors, 0, 0)
            pcd_array = np.hstack((pcd_points, pcd_colors))

            cents, labels = fc.kmeans_from_scratch(pcd_array, 2, weights)
            labels_list.append(labels)
            pc_list.append(pcd_points)


    # Seperate by number of points in segment and only return weeds
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
    '''!
    Evaluate the cost given the weights. Uses the weights to do initial segmentation on the training point clouds and then finds the average distance between the manual labels and the centroids of the segmented clouds.

    @param weights   a list of 6 weights, corresponding to x, y, z, r, g, and b
    @param training_directory   a string with the directory to the training data
    @param pc_filenames    a list of strings containing the names of the pointclouds being used for training. eg: "79.ply"
    @param manual_label_filenames   a list of strings containing the names of the manual labels for each pointcloud in pc_filenames. eg. "79.npy"

    @return the cost
    '''
    # Do initial segmentation using weighted kmeans and provided weights
    weeds_list = get_segmentation(weights, training_directory, pc_filenames)

    # Organize manual labels into a list of 3d points
    manual_labels_folder = training_directory + '/training_manually_segmented/'
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
    '''!
    Finds and prints the optimal weights for the distance function used in kmeans. These can be input into DBSCAN_calculate_pose or FRG_calculate_pose when using algorithm='kmeans-optimized'
    '''
    
    # Starting weights and bounds for the optimization function
    starting_weights = [0, 0, 0, 0, 0, 0]
    bounds = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]


    # Location of the training data
    training_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "segmentation_training"))

    # Make a list of file names for the point clouds and their manual labels
    pc_filenames = []
    manual_labels_filenames = []
    names_list = ['2', '5', '79', '93', '94', '95', '99', '100', '104']
    for i in range(len(names_list)):
        pc_filenames.append(''.join((names_list[i], ".npy")))
        manual_labels_filenames.append(''.join((names_list[i], ".ply")))


    # Optimize the cost function to find the optimal weights for kmeans distance function
    results = dict()
    results['shgo'] = optimize.shgo(calculate_cost, bounds, args=(training_directory, pc_filenames, manual_labels_filenames), options={'disp': True})
    best_weights = results['shgo'].x
    print()
    print("The best weights are: ", best_weights)
    print()

    minimized_dist = calculate_cost(best_weights, training_directory, pc_filenames, manual_labels_filenames)
    print("The smallest distance is: ", minimized_dist)
    return 0


if __name__ == "__main__":
    main()

