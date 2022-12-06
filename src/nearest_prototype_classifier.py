"""!
@brief Uses pre-segmented data to create a nearest prototype classifier for initial segmentation step
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
import find_centroids as fc


def get_training_points(path_to_training, file_numbers):
    '''!
    Returns lists of points for each category

    @param path_to_training   path to training data
    @param file_numbers   list of file numbers to consider (list of strings)

    @return weeds, dirt, rocks
    '''

    weeds_training_set = np.zeros((1, 6))
    rocks_training_set = np.zeros((1, 6))
    dirt_training_set = np.zeros((1, 6))
    for file_num in file_numbers:
        weed_file = path_to_training + file_num + '_weeds.ply'
        rock_file = path_to_training + file_num + '_rocks.ply'
        dirt_file = path_to_training + file_num + '_dirt.ply'

        weeds_points = np.asarray(o3d.io.read_point_cloud(weed_file).points)
        weeds_colors = np.asarray(o3d.io.read_point_cloud(weed_file).colors)
        weeds_all = np.hstack((weeds_points, weeds_colors))
        weeds_training_set = np.vstack((weeds_training_set, weeds_all))

        rocks_points = np.asarray(o3d.io.read_point_cloud(rock_file).points)
        rocks_colors = np.asarray(o3d.io.read_point_cloud(rock_file).colors)
        rocks_all = np.hstack((rocks_points, rocks_colors))
        rocks_training_set = np.vstack((rocks_training_set, rocks_all))

        dirt_points = np.asarray(o3d.io.read_point_cloud(dirt_file).points)
        dirt_colors = np.asarray(o3d.io.read_point_cloud(dirt_file).colors)
        dirt_all = np.hstack((dirt_points, dirt_colors))
        dirt_training_set = np.vstack((dirt_training_set, dirt_all))
    
    weeds_training_set = np.delete(weeds_training_set, 0, 0)
    rocks_training_set = np.delete(rocks_training_set, 0, 0)
    dirt_training_set = np.delete(dirt_training_set, 0, 0)
    
    return weeds_training_set, dirt_training_set, rocks_training_set


def get_zhsv(all_data):
    '''!
    Convert x, y, z, r, g, b data to z, h, s, v

    @param all_data   list of points in [x, y, z, r, g, b] form

    @return list of points in [z, h, s ,v] form
    '''
    # select only colors and reorder from RGB to BGR
    all_colors = np.array([all_data[:,5], all_data[:,4], all_data[:,3]]).T

    # convert colors from 0-1 floats to 0-255 ints
    all_colors = (all_colors * 255.0).astype(np.uint8)


    # Convert all colors to hsv
    for row in range(all_colors.shape[0]):
        bgr_color = np.uint8([[all_colors[row, :]]])
        hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0, 0, :]
        for col in range(all_colors.shape[1]):
            all_colors[row, col] = hsv_color[col]

    # Assemble once more
    zhsv = np.hstack((all_data[:,2].reshape((all_colors.shape[0], 1)), all_colors))

    # Make everying scale of 0-1
    normalizing_array = np.array([1, 179, 255, 255])
    zhsv = zhsv / normalizing_array[None, :]

    # Return only columns that matter
    # zhsv = zhsv[:,2:]

    return zhsv


def npc_segment_weeds(unclassified_data, use_hsv=False):
    '''!
    Use nearest prototype classifier to segment weeds

    @param unclassified_data   list of points to classify
    @param use_hsv   True/False to choose whether or not to convert to hsv

    @return list of label numbers for input data
    '''
    
    path_to_training = '/home/amasse/catkin_ws/src/plant_selector/segmentation_training/npc_training_data/'
    # possible file numbers ['20', '22', '77', '79', '95', '104]
    file_numbers = ['22']

    plot_training_rgb = False
    

    # Define the set of labels Y:
    Y = ['weeds', 'dirt', 'rocks']


    # Import the sets of data for training classifier
    weeds_training_set, dirt_training_set, rocks_training_set = get_training_points(path_to_training, file_numbers)


    # Get only z, h, s, and v values if desired
    if use_hsv == True:
        weeds_training_set = get_zhsv(weeds_training_set)
        dirt_training_set = get_zhsv(dirt_training_set)
        rocks_training_set = get_zhsv(rocks_training_set)
        unclassified_data = get_zhsv(unclassified_data)
    elif plot_training_rgb == True:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(weeds_training_set[:,3],weeds_training_set[:,4],weeds_training_set[:,5], color='green')
        ax.scatter(rocks_training_set[:,3],rocks_training_set[:,4],rocks_training_set[:,5], color='orange')
        ax.scatter(dirt_training_set[:,3],dirt_training_set[:,4],dirt_training_set[:,5], color='black')
        ax.set_xlabel("Red")
        ax.set_ylabel("Green")
        ax.set_zlabel("Blue")
        plt.show()



    # Comupte the per-class centroids
    weeds_centroid = np.mean(weeds_training_set, axis=0)
    rocks_centroid = np.mean(rocks_training_set, axis=0)
    dirt_centroid = np.mean(dirt_training_set, axis=0)
    cents = np.array([weeds_centroid, dirt_centroid, rocks_centroid])


    # For each point in the unclassified data, find the class label that minimizes the distance to centroid
    labels = []
    label_nums = []
    for point in range(unclassified_data.shape[0]):
        dist = []
        for centroid in range(cents.shape[0]):
            dist.append(np.linalg.norm(unclassified_data[point] - cents[centroid]))
        labels.append(Y[np.argmin(dist)])
        label_nums.append(np.argmin(dist))

    label_nums = np.asarray(label_nums)
    
    return label_nums


def main():
    file_loc = '/home/amasse/catkin_ws/src/plant_selector/segmentation_training/all_selections/79.npy'

    # Load in data we want to classify
    pcd, unclassified_data, _ = fc.array_2_pc(np.load(file_loc))
    # o3d.visualization.draw_geometries([pcd], window_name="Original")

    label_nums = npc_segment_weeds(unclassified_data, use_hsv=False)

    # Color and visualize
    max_label = max(label_nums)
    colors = plt.get_cmap("Paired")(label_nums / (max_label if max_label > 0 else 1))
    label_nums = np.asarray(label_nums)
    colors[label_nums == 1] = 0
    colors[label_nums == 2] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([pcd], window_name="Initial Segmentation")
    return


if __name__ == "__main__":
    main()