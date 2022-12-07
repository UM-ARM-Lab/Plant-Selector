"""!
@brief Finds the centroids of weeds from provided point cloud selection
"""

from audioop import avg
import ctypes
from os import fdatasync
import struct
from turtle import fd

import hdbscan
from statistics import mode
from math import atan, pi
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import open3d as o3d
import scipy as sp
import random
import itertools
import more_itertools
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, BisectingKMeans
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from scipy import optimize

import rospy
from sensor_msgs import point_cloud2 as pc2
from tf.transformations import rotation_matrix

import nearest_prototype_classifier as npc
import facet_region_growing as frg
import leaf_axis as la
import gen_alg as ga


def color_calculate_pose(points, return_multiple_grasps=False):
    '''!
    Uses ratios between colors and DBSCAN to calculate the poses for weeds in the provided selection

    @param points   a list of points from the point cloud from the selection
    @param return_multiple_grasps   Boolean indicating whether or not to return multiple grasps for selections of multiple weeds

    @return list of weed centroids, normal associated with dirt
    '''

    # All the weed extraction algorithm
    pcd_points = points[:, :3]
    float_colors = points[:, 3]

    pcd_colors = np.array((0, 0, 0))
    for x in float_colors:
        rgb = float_to_rgb(x) * 255.0
        pcd_colors = np.vstack((pcd_colors, rgb))

    # Alternate green color filter
    pcd_colors = pcd_colors[1:, :]

    # Filter the point cloud so that only the green points stay
    # Get the indices of the points with g parameter greater than x
    green_points_indices = np.where((pcd_colors[:, 1] > pcd_colors[:, 0] * (12.0 / 11.0)) &
                                    (pcd_colors[:, 1] > pcd_colors[:, 2] * (12.0 / 11.0)))
    green_points_xyz = pcd_points[green_points_indices]
    green_points_rgb = pcd_colors[green_points_indices]

    r_low, g_low, b_low = 10, 20, 10
    r_high, g_high, b_high = 240, 240, 240
    green_points_indices = np.where((green_points_rgb[:, 0] > r_low) & (green_points_rgb[:, 0] < r_high) &
                                    (green_points_rgb[:, 1] > g_low) & (green_points_rgb[:, 1] < g_high) &
                                    (green_points_rgb[:, 2] > b_low) & (green_points_rgb[:, 2] < b_high))

    if len(green_points_indices[0]) == 1:
        rospy.loginfo("No green points found. Try again.")
        return None, None

    # Save xyzrgb info in green_points (type: numpy array)
    green_points_xyz = green_points_xyz[green_points_indices]
    green_points_rgb = green_points_rgb[green_points_indices]

    # Create Open3D point cloud for green points
    green_pcd = o3d.geometry.PointCloud()
    # Save xyzrgb info in green_pcd (type: open3d.PointCloud)
    green_pcd.points = o3d.utility.Vector3dVector(green_points_xyz)
    green_pcd.colors = o3d.utility.Vector3dVector(green_points_rgb)
    

    # Apply radius outlier filter to green_pcd
    _, ind = green_pcd.remove_radius_outlier(nb_points=7, radius=0.007)

    if len(ind) == 0:
        print("Not enough points. Try again.")
        return None, None

    # Just keep the inlier points in the point cloud
    green_pcd = green_pcd.select_by_index(ind)
    green_pcd_points = np.asarray(green_pcd.points)
    # green_pcd.paint_uniform_color([0.01, 0.5, 0.01])
    # o3d.visualization.draw_geometries([green_pcd], window_name="Initial Segmentation")

    # Apply DBSCAN to green points
    labels = np.array(green_pcd.cluster_dbscan(eps=0.0055, min_points=15))  # This is actually pretty good
    max_label = labels.max()
    colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    green_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([green_pcd])

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters == 0:
        print("Not enough points. Try again.")
        return None, None

    # If we want multiple centroids, return multiple, if not return only largest
    if return_multiple_grasps == True:
        weeds_segments = labels_to_dict(green_pcd, labels)
        weed_centroid = calculate_centroids(weeds_segments)
    else:
        biggest_cluster_indices = np.where(labels[:] == mode(labels))
        green_pcd_points = green_pcd_points[biggest_cluster_indices]
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
        return None, None

    [a, b, c, _] = plane_model
    if a < 0:
        normal = np.asarray([a, b, c])
    else:
        normal = -np.asarray([a, b, c])

    return weed_centroid, normal


def DBSCAN_calculate_pose(points, algorithm='npc', weights=[0,100,100,0,100,0], return_multiple_grasps=False):
    '''!
    Uses DBSCAN to cluster weeds and calculate pose

    @param points   a list of points from the point cloud from the selection
    @param algorithm   string containing algorithm choice. options include 'kmeans-optimized', 'kmeans-redmean', 'kmeans-euclidean', 'bi-kmeans', 'spectral', 'ward', 'npc'
    @param weights    a list containing 6 weights for weighted kmeans implementation. weights must be >= 0
    @param return_multiple_grasps   Boolean indicating whether or not to return multiple grasps for selections of multiple weeds

    @return list of weed centroids, normal associated with dirt
    '''
    # Do initial segmentation
    weeds, dirt = initial_segmentation(points, algorithm, weights)
    
    weeds_array = np.asarray(weeds.points)
    if weeds_array.shape[0] < 2:
        return None, None
    # weeds.paint_uniform_color([0.01, 0.5, 0.01])set_partitions
    # o3d.io.write_point_cloud("/home/amasse/catkin_ws/src/plant_selector/weed_eval/segmented_weeds2.pcd", weeds)
    # o3d.visualization.draw_geometries([weeds])


    # Calculate ideal epsilon based on data (lol this makes things worse)
    epsilon = calculate_epsilon(weeds_array)


    # Run dbscan to cluster into individual weeds, color them
    labels = np.array(weeds.cluster_dbscan(eps=epsilon, min_points=10))
    max_label = labels.max()
    colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    weeds.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([weeds])
    

    # Seperate the weeds into dict of individual clusters
    weeds_segments_all = labels_to_dict(weeds, labels)


    # Remove weeds with less than 15 points
    weeds_segments = {}
    count = 0
    for i in range(len(weeds_segments_all)):
        if len(weeds_segments_all[i].points) >= 15:
            weeds_segments[count] = weeds_segments_all[i]
            count += 1
    

    # Find list of centroids of weeds and dirt normal
    weeds_centroids = calculate_centroids(weeds_segments)
    normal = calculate_normal(dirt)

    if len(weeds_centroids) < 1:
        return None, None
    

    # If we want multiple grasps, return multiple grasps
    if return_multiple_grasps == True:
        return weeds_centroids, normal


    # Remove centroid guesses that do not lie within a weed
    weeds_centroids = remove_false_positives(weeds_centroids, weeds_array)


    # For now only return the largest weed
    weeds_sizes = {}
    for i in range(len(weeds_centroids)):
        weeds_sizes[i] = np.shape(np.asarray(weeds_segments[i].points))[0]
    if len(weeds_sizes) < 1:
        return None, None
    largest_segment = max(weeds_sizes, key=weeds_sizes.get)


    return weeds_centroids[largest_segment], normal


def FRG_calculate_pose(points, algorithm='npc', weights=[0,100,100,0,100,0], proximity_thresh=0.05, return_multiple_grasps=False):
    '''!
    Uses Facet Region Growing method and PCA to return the pose for the gripper

    @param points   a list of points from the point cloud from the selection
    @param algorithm   string containing algorithm choice. options include 'kmeans-optimized', 'kmeans-redmean', 'kmeans-euclidean', 'bi-kmeans', 'spectral', 'ward', 'npc
    @param weights    a list containing 6 weights for weighted kmeans implementation. weights must be >= 0
    @param return_multiple_grasps   Boolean indicating whether or not to return multiple grasps for selections of multiple weeds

    @return list of weed centroids, normal associated with dirt
    '''
    print("\nUsing facet region growing with {}...\n".format(algorithm))

    # Do initial segmentation and convert to numpy array
    all_weeds, dirt = initial_segmentation(points, algorithm, weights)
    all_weeds_array = np.asarray(all_weeds.points)
    if all_weeds_array.shape[0] < 2:
        return None, None

    # Use facet region growing to get individual leaves (provided there are enough points)
    if all_weeds_array.shape[0] >= 30:
        # Run dbscan to cluster into individual weeds, color them
        labels = np.array(all_weeds.cluster_dbscan(eps=0.0065, min_points=5))
        max_label = labels.max()
        colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        all_weeds.colors = o3d.utility.Vector3dVector(colors[:, :3])
        # o3d.visualization.draw_geometries([all_weeds], window_name="Clustering")

        # Seperate the weeds into dict of individual clusters
        all_weeds_segments = labels_to_dict(all_weeds, labels)

        # Remove clusters under 20 points and put them in a list, combine large ones
        small_weeds = []
        small_centroids = np.array([0, 0, 0])
        weeds = o3d.geometry.PointCloud()
        for i in range(len(all_weeds_segments)):
            if len(all_weeds_segments[i].points) < 30:
                current_weed = np.asarray(all_weeds_segments[i].points)
                small_weeds.append(current_weed)
                small_centroids = np.vstack((small_centroids, np.mean(current_weed, axis=0)))
            else:
                weeds += all_weeds_segments[i]

        if not np.array_equal(np.array([0, 0, 0]), small_centroids):
            small_centroids = np.delete(small_centroids, 0, 0)

        # Format and display only large weeds (that will go into facet region growing)
        weeds_array = np.asarray(weeds.points)
        weeds.paint_uniform_color([0.01, 0.5, 0.01])
        # o3d.visualization.draw_geometries([weeds], window_name="Only Large")

        if len(weeds_array) > 0:
            # Do facet region growing to find leaves
            leaves = frg.facet_leaf_segmentation(weeds_array)

            # Format+display FRG
            leaf_pcs = []
            for i in range(len(leaves)):
                color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
                leaf = o3d.geometry.PointCloud()
                leaf.points = o3d.utility.Vector3dVector(leaves[i])
                leaf.paint_uniform_color(list(color[:3]))
                leaf_pcs.append(leaf)
            # o3d.visualization.draw_geometries(
                # [leaf_pcs[i] for i in range(len(leaves))], window_name="FRG")

            
            # If there is only one leaf we dont want to do the rest of FRG. 
            if len(leaves) == 1:
                weeds_segments = {0: leaf_pcs[0]}
            else:
                # Get leaf axes and base points and associated leaves
                edges = np.array([0, 0, 0, 0, 0, 0])
                for leaf in leaves:
                    _, current_edges= la.get_axis_and_ends(leaf)
                    edges = np.vstack((edges, current_edges))
                edges = np.delete(edges, 0, 0)
                num_edges = edges.shape[0]


                # Genetic algorithm
                # n_iter = 100 # define the total iterations
                # n_bits = 3 * num_edges # bits
                # n_pop = 10 # define the population size
                # r_cross = 0.5 # crossover rate
                # r_mut = 1.0 / float(n_bits) # mutation rate
                # best_S_bits, score = ga.genetic_algorithm(genetic_fitness_function, edges, leaves, n_bits, n_iter, n_pop, r_cross, r_mut)
                # best_S_string = "".join([str(best_S_bits[x]) for x in range(0, len(best_S_bits))])
                # best_S_int = [int(best_S_string[i:i+3], 2) for i in range(0, len(best_S_string), 3)]
                # best_S = reduce_S(best_S_int)
                # print(best_S)
                
                # Find a way to convert best_S_int so that it has no gaps

                # Use brute force to find optimal weed assignment based on cost function from least squares
                best_S, lowest_cost, _ = brute_force_optimize(edges, leaves)

                # Convert optimal weed assignment to dictionary of pointclouds
                weeds_segments = assemble_weeds_from_label(leaves, best_S)

                # Solve for centroids and normals using usual methods, remove obvious false positives
                weeds_centroids = calculate_centroids(weeds_segments)
                if not np.array_equal(np.array([0, 0, 0]), small_centroids):
                    weeds_centroids = np.vstack((weeds_centroids, np.array(small_centroids)))
                if len(weeds_segments) > 1:
                    weeds_centroids = remove_false_positives(weeds_centroids, all_weeds_array)
        else:
            weeds_centroids = small_centroids
    else:
        weeds_segments = {0: all_weeds}
        weeds_centroids = calculate_centroids(weeds_segments)
        normal = calculate_normal(dirt)
        return weeds_centroids, normal
    
    normal = calculate_normal(dirt)
    if len(weeds_centroids) < 1:
        return None, None
    
    # o3d.visualization.draw_geometries(
    #     [weeds_segments[i] for i in range(len(weeds_segments))], window_name="Weed Combinations")

    # If we want multiple grasps, return multiple grasps
    if return_multiple_grasps == True:
        return weeds_centroids, normal
    else:
        return weeds_centroids[0], normal


def brute_force_optimize(E, leaves, max_weeds=5):
    '''!
    Brute force optimization of cost functions to find best assignment S of leaves to weeds

    @param E   An Nx6 matrix where each row corresponds to a line (as defined by two 3d points) representing a leaf
    @param leaves   a list where each element is a numpy array containing xyz points for each leaf
    
    @return best assignments S, cost for that assignment, centroid locations from least squares for S
    '''

    # Find all possible S for the number of edges and max number of possible weeds
    num_edges = len(E)
    combinations = find_combinations(num_edges, max_weeds, allow_singles=True)

    # Test all S with least squares to find the lowest cost
    lowest_cost = math.inf
    best_centroids = 0
    all_costs = []
    best_S = 0
    all_S = []
    all_times = 0
    for i in range(len(combinations)):
        current_S = list(combinations[i])
        all_S.append(np.array(current_S).reshape(1, len(current_S))[0])
        st = time.time()
        # D_cost = distance_cost(E, leaves, current_S)
        LS_cost1, LS_cost2, current_centroids = LstSqrs_cost(current_S, E, leaves, return_cents = True)
        et = time.time()
        all_times += et-st
        current_cost = LS_cost2
        all_costs.append(current_cost)
        
        if current_cost < lowest_cost:
            lowest_cost = current_cost
            best_centroids = current_centroids
            best_S = current_S
    # print("Avg Cost Eval Time: ", all_times/len(combinations))
    return best_S, lowest_cost, best_centroids


def find_combinations(num_edges, max_weeds, allow_singles=True):
    '''!
    Use more_itertools to return a list of possible solutions, S

    @param num_edges   the number of edges (or facets) we are trying to find a solution for. This dictates the length of S
    @param max_weeds   the maximum number of weeds we are assuming exist in the selection. This dictates the highest possible value in S
    @param allow_singles    optional, boolean, True if you want to allow S to allow leaves to be assigned alone to their own weed, False otherwise

    @return a dictionary of edges, a dictionary of points
    '''

    # Create all possible partition combinations and convert them to form of S
    combinations = []
    for i in range(1, max_weeds+1):
        current_possible_parts = list(more_itertools.set_partitions(list(range(num_edges)), i))

        for n in range(len(current_possible_parts)):
            current_part = current_possible_parts[n]
            keep_S = True
            S = [0] * num_edges

            for j in range(len(current_part)):
                weed_assignment = current_part[j]

                # Allow or disallow S' that contain singles
                if allow_singles == True:
                    for k in weed_assignment:
                        S[k] = j
                else:
                    if len(weed_assignment) > 1:
                        for k in weed_assignment:
                            S[k] = j
                    else:
                        keep_S = False

            if keep_S == True:
                combinations.append(S)
    
    return combinations


def genetic_fitness_function(S_bits, E, leaves):
    '''!
    The least squares cost function but formatted for use with the genetic algorithm.

    @param S_bits   a list of bits where each set of 3 is binary for a single integer value between 0 and 7
    @param E   An Nx6 matrix where each row corresponds to a line (as defined by two 3d points) representing a leaf
    @param leaves   a list where each element is a numpy array containing xyz points for each leaf

    @return least squares mean distance cost (goes up as solution becomes more correct, as is required by the genetic algorithm)
    '''
    
    # Convert S_bits to integers
    S_string = "".join([str(S_bits[x]) for x in range(0, len(S_bits))])
    S_ints = [int(S_string[i:i+3], 2) for i in range(0, len(S_string), 3)]
    S_ints = reduce_S(S_ints)

    # Evaluate cost function of choice
    LS_cost1, LS_cost2 = LstSqrs_cost(S_ints, E, leaves)

    return -1 * LS_cost2


def LstSqrs_cost(S, E, leaves, return_cents=False, default_res=3.5, default_dist=0.07):
    '''!
    Uses least squares to find the centroids of plants given their leaf assignments

    @param S   a list of integer labels corresponding to each leaf
    @param E   An Nx6 matrix where each row corresponds to a line (as defined by two 3d points) representing a leaf
    @param leaves   a list where each element is a numpy array containing xyz points for each leaf
    @param default_res   optional, a float assigning a default least squares residual in the event that least squares cannot find one (usually for cases where there is a single leaf alone in an assignment)
    @param default_dist   optional, a float assigning a default least squares mean distance in the event that least squares cannot find one (usually for cases where there is a single leaf alone in an assignment)

    @return sum of the residuals from least squares, sum of distances to mean locations, list of centroids for each weed
    '''
    
    S = [round(i) for i in S]
    S = [int(i) for i in S]
    S = np.array(S)
    
    # Seperate E into weeds based on S
    plants, plants_points = solution_to_plants(E, leaves, S)

    # Do least squares for every plant and store the centroids and residuals in lists
    centroids_list = []
    residuals = []
    dist_to_mean_list = []
    count = 0
    for i in range(len(plants)):
        edges = plants[i]
        weed = plants_points[i]
        
        # If the current plant index actually has a plant in it, do the stuff
        if np.any(edges):
            # Calculate matricies A and b for Least Squares and solve it for each plant
            # A is the sum of all A_i for each edge, e_i 
            A = np.array([0, 0, 0])
            b = np.array([[0]])
            mean_leaves = np.array([0, 0, 0])
            for e in range(edges.shape[0]):
                mean_leaf_location = np.mean(weed[e], axis=0)
                mean_leaves = np.vstack((mean_leaves, mean_leaf_location))
                point_a = edges[e,:3]
                point_b = edges[e, 3:]
                d = point_b - point_a
                f = (2 * d) / np.linalg.norm(d)**2
                R = np.dot(-point_a, d)
                A_i = np.array([[2-f[0]*d[0], -f[0]*d[1], -f[0]*d[2]], [-f[1]*d[0], 2-f[1]*d[1], -f[1]*d[2]], [-f[2]*d[0], -f[2]*d[1], 2-f[2]*d[2]]])
                b_i = np.array([[(2 * point_a[0]) - (f[0] * R)], [(2 * point_a[1]) - (f[1] * R)], [(2 * point_a[2]) - (f[2] * R)]])
                A = np.vstack((A, A_i))
                b = np.vstack((b, b_i))
        
            mean_leaves = np.delete(mean_leaves, 0, 0)
            A = np.delete(A, 0, 0)
            b = np.delete(b, 0, 0)

            # Perform least squares to get centroid guess and residuals
            cent, res, _, _ = np.linalg.lstsq(A, b, rcond=None)
            centroids_list.append(list(cent.reshape((3,))))

            # Find sum of distances of LS predicted centroid to mean locations of each leaf in a plant
            all_dists_to_leaves = np.linalg.norm(mean_leaves - cent.reshape((1,3)), axis=1)
            dists_to_leaves = np.sum(all_dists_to_leaves)
            

            # If no residual is given (ie the plant assignment only contains one leaf), set residual to default value
            if len(res) > 0:
                residuals.append(res[0])
                dist_to_mean_list.append(dists_to_leaves)
                count += 1
            else:
                residuals.append(default_res)
                dist_to_mean_list.append(default_dist)


    # Return appropriate values
    if return_cents == True:
        # return np.sum(residuals)/count, np.array(centroids_list)
        return np.sum(residuals), sum(dist_to_mean_list), np.array(centroids_list)
    else:
        # return np.sum(residuals)/count
        return np.sum(residuals), sum(dist_to_mean_list)


def solution_to_plants(E, leaves, S):
    '''!
    Use the solution S to create a dictionary of edges and a dictionary of points

    @param E   a Hx6 numpy array where each row is a set of edges representing a facet. The first 3 values are coords of endpoint 1 and the last 3 values are coords of endpoint 2
    @param leaves   a list of leaves. Each leaf is an Gx3 numpy array where each row is a point in the leaf.
    @param S    a list of integers representing weed assignments (S is similar to the labels output of DBSCAN)

    @return a dictionary of edges, a dictionary of points
    '''

    plants = {}
    plant_points = {}
    for i in range(S.max()+1):
        plant_exists = False
        idx = np.where(S[:] == i)[0]
        current_plant = np.array([0, 0, 0, 0, 0, 0])
        current_plant_points = []
        for j in idx:
            current_plant = np.vstack((current_plant, E[j, :]))
            current_plant_points.append(leaves[j])
            plant_exists = True
        
        # If the current S doesnt assign current index to anything set it to an empty list
        if plant_exists:
            current_plant = np.delete(current_plant, 0, 0)
            plants[i] = current_plant
            plant_points[i] = current_plant_points
        else:
            plants[i] = np.array([])
            plant_points[i] = np.array([])
    return plants, plant_points


def reduce_S(S):
    '''!
    If S contains non-consecutive numbers, we can fix that.

    @param S    a list of integers representing weed assignments (S is similar to the labels output of DBSCAN)

    @return a fixed version of S
    '''
    new_S = np.zeros((len(S),))
    S = np.array(S)
    count = 0
    done_list = []
    for i in range(len(S)):
        current = S[i]
        idx = np.where(S == current)[0]
        if idx.size != 0:
            for j in idx:
                if j not in done_list:
                    done_list.append(j)
                    new_S[j] = count
            count += 1
        
    return list(new_S.astype(np.int64))


def distance_cost(E, leaves, S):
    '''!
    Calculate the value of the distance cost function for a given solution

    @param E   a Hx6 numpy array where each row is a set of edges representing a facet. The first 3 values are coords of endpoint 1 and the last 3 values are coords of endpoint 2
    @param leaves   a list of leaves. Each leaf is an Gx3 numpy array where each row is a point in the leaf.
    @param S    a list of integers representing weed assignments (S is similar to the labels output of DBSCAN)

    @return a float that decreases as the leaves in a weed assignment get closer together
    '''

    # Format S into 
    S = [round(i) for i in S]
    S = [int(i) for i in S]
    S = np.array(S)


    # Create dict of edges according to proposed S
    plants, plants_points = solution_to_plants(E, leaves, S)
    
    
    all_plant_costs = []
    for i in range(len(plants)):
        # Get list of all possible combinations of points within the plant
        possible_combos = list(itertools.combinations(list(range(len(plants[i]))), 2))

        edge_set = plants[i]

        # If the current plant only has one leaf it has no distance cost. Otherwise:
        if len(edge_set) >= 2:
            # For each possible combo find min dist:
            all_min_dists = []
            for j in range(len(possible_combos)):
                current_combo = possible_combos[j]
                all_dists = np.array([
                    np.linalg.norm(edge_set[current_combo[0]][0] - edge_set[current_combo[1]][0]),
                    np.linalg.norm(edge_set[current_combo[0]][0] - edge_set[current_combo[1]][1]),
                    np.linalg.norm(edge_set[current_combo[0]][1] - edge_set[current_combo[1]][0]),
                    np.linalg.norm(edge_set[current_combo[0]][1] - edge_set[current_combo[1]][1])])
                min_dist = np.amin(all_dists)
                all_min_dists.append(min_dist)
            
            # Find the avg min distance for all combinations within a single plant
            plant_cost = np.mean(all_min_dists)
            # plant_cost = np.sum(all_min_dists)
        
            all_plant_costs.append(plant_cost)

    # Distance cost is sum of all plant costs. The closer together the base points of leaves the smaller the cost
    d_cost = sum(all_plant_costs) * 1000

    return d_cost


def remove_false_positives(centroids, weeds_array):
    '''!
    Delete centroid guesses that are not near any weed points

    @param centroids   a numpy array where each row is a centroid guess
    @param weeds_array   a numpy array where each row is a point in the point cloud of all weeds (ie. excluding dirt and rocks)

    @return a numpy array containing only centroids near weeds
    '''
    new_centroids = np.array([0, 0, 0])

    # go through each of the points in the centroids and determine if close to the weeds pointcloud
    min_dist = math.inf
    for i in range(centroids.shape[0]):
        current_cent = centroids[i, :]

        for j in range(weeds_array.shape[0]):
            dist = np.linalg.norm(current_cent - weeds_array[j, :])
            if dist < min_dist:
                min_dist = dist
            
        if min_dist < 0.0015:
            new_centroids = np.vstack((new_centroids, current_cent))

    new_centroids = np.delete(new_centroids, 0, 0)

    return new_centroids


def initial_segmentation(points, algorithm='npc', weights=[0,100,100,0,100,0]):
    '''!
    Does initial segmentation --> Uses specified algorithm to seperate weeds from dirt

    @param points   a list of points from the point cloud from the selection
    @param algorithm   optional, string containing algorithm choice. options include 'kmeans-optimized', 'kmeans-redmean', 'kmeans-euclidean', 'bi-kmeans', 'spectral', 'ward', 'npc'
    @param weights    optional, a list containing 6 weights for weighted kmeans implementation. weights must be >= 0

    @return point cloud of weeds, point cloud of dirt
    '''
    
    # Create and format point cloud
    pcd, pcd_array, pcd_colors = array_to_pc(points)
    # o3d.visualization.draw_geometries([pcd], window_name="Original")


    # Save pc to file if desired
    file_loc = "/home/amasse/catkin_ws/src/plant_selector/weed_eval/"
    # np.save(file_loc + "saved.npy", points)
    # o3d.io.write_point_cloud(file_loc + "saved.pcd", pcd)


    # Do clustering on original point cloud, based on desired method
    k = 2
    using_npc = False
    if algorithm == 'kmeans-optimized':
        cents, labels = kmeans_from_scratch(pcd_array, k, weights)
    elif algorithm == 'kmeans-redmean':
        cents, labels = kmeans_from_scratch(pcd_array, k, redmean=True)
    elif algorithm == 'kmeans-euclidean':
        kmeans = KMeans(n_clusters=k, random_state=0).fit(pcd_colors)
        labels = kmeans.labels_
    elif algorithm == 'bi-kmeans':
        bi_kmeans = BisectingKMeans(n_clusters=k, random_state=0).fit(pcd_array)
        labels = bi_kmeans.labels_
    elif algorithm == 'spectral':
        sc = SpectralClustering(n_clusters=k,
            assign_labels='discretize',
            random_state=0).fit(pcd_array)
        labels = sc.labels_
    elif algorithm == 'ward':
        ward = AgglomerativeClustering(linkage="ward", n_clusters=k).fit(pcd_array)
        labels = ward.labels_
    elif algorithm == 'npc':
        labels = npc.npc_segment_weeds(pcd_array)
        using_npc = True


    # Color and display according to labels from initial segmentation step
    max_label = labels.max()
    if max_label < 1:
        return None, None
    colors = plt.get_cmap("Paired")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([pcd], window_name="Initial Segmentation")


    # Seperate segments into dict of idividual clusters
    segments = labels_to_dict(pcd, labels)


    # Compare by sizes to get pcds for weeds and dirt, filter out outliers
    # This makes the assumption that the dirt pc is larger than the weeds pc
    weeds, dirt = separate_by_size(segments, using_npc)
    _, ind = weeds.remove_radius_outlier(nb_points=7, radius=0.007)
    if len(ind) != 0:
        weeds = weeds.select_by_index(ind)
    
    return weeds, dirt


def kmeans_from_scratch(X, k, weights=[1, 1, 1, 1, 1, 1], redmean=False):
    '''!
    Implements kmeans from scratch
    Mostly stolen from https://medium.com/nerd-for-tech/k-means-python-implementation-from-scratch-8400f30b8e5c

    @param X   a list of points from the point cloud from the selection
    @param k   desired number of clusters

    @return labels and their centroids
    '''
    diff = 1
    labels = np.zeros(X.shape[0])
   
    centroid_idx = np.random.choice(X.shape[0], k)
    centroids = np.zeros((k,6))
    for i in range(k):
        for j in range(6):
            centroids[i, j] = X[centroid_idx[i], j]
    
    while diff:
        # for each observation
        for i, row in enumerate(X):
            mn_dist = float('inf') # dist of the point from all centroids
            
            for idx, centroid in enumerate(centroids):
                # Use redmean distance (just another distance metric in color space)
                if redmean == True:
                    rc255 = int(centroid[3] * 255)
                    rr255 = int(row[3] * 255)

                    rc = centroid[3]
                    rr = row[3]
                    gc = centroid[4]
                    gr = row[4]
                    bc = centroid[5]
                    br = row[5]
                    
                    r_bar = 0.5 * (rc255 - rr255)
                    if r_bar < 128:
                        d = np.sqrt(2*(rc-rr)**2 + 4*(gc-gr)**2 + 3*(bc-br)**2)
                    else:
                        d = np.sqrt(3*(rc-rr)**2 + 4*(gc-gr)**2 + 2*(bc-br)**2)

                # Use weighted distance with weight calculated in optimize_kmeans.py
                else:
                    d = np.sqrt(
                        (weights[0] * (centroid[0] - row[0])**2) +
                        (weights[1] * (centroid[1] - row[1])**2) +
                        (weights[2] * (centroid[2] - row[2])**2) +
                        (weights[3] * (centroid[3] - row[3])**2) +
                        (weights[4] * (centroid[4] - row[4])**2) +
                        (weights[5] * (centroid[5] - row[5])**2))
                
                if mn_dist > d: # store closest centroid
                    mn_dist = d
                    labels[i] = idx
            
        new_centroids = pd.DataFrame(X).groupby(by=labels).mean().values

        # if centroids are same then leave
        if np.count_nonzero(centroids-new_centroids) == 0:
            diff = 0
        else:
            centroids = new_centroids

    labels = labels.astype(int)

    return centroids, labels


def separate_by_height(segments, height_threshold=0.001):
    '''!
    Seperates dirt point cloud and weed point clouds by avg height

    @param segments   dict containing all segments of point cloud
    @param height_threshold   float

    @return weeds, dirt pointclouds containing weeds and dirt respecively
    '''

    # Assume the first segment is dirt (because it is the largest)
    dirt = segments[0]

    # Find all heights of each segment
    segment_heights = {}
    for i in range(len(segments)):
        segment_heights[i] = np.average(np.asarray(segments[i].points)[:,2])
    
    dirt_height = segment_heights[0]

    # Compare all heights, add all weeds to pcd, paint them all black
    weeds = o3d.geometry.PointCloud()
    for i in range(len(segments)):
        if segment_heights[i] > dirt_height + height_threshold:
            weeds += segments[i]
    weeds.paint_uniform_color([0, 0, 0])

    return weeds, dirt


def array_to_pc(points):
    '''!
    Take in array of points and format it into an open3d point cloud

    @param points   list of points

    @return open3d pc, array of points, array of colors
    '''
    pcd_points = points[:, :3]
    float_colors = points[:, 3]
    pcd_colors = np.array((0, 0, 0))
    for x in float_colors:
        rgb = float_to_rgb(x)
        pcd_colors = np.vstack((pcd_colors, rgb))
    pcd_colors = np.delete(pcd_colors, 0, 0)
    pcd_array = np.hstack((pcd_points, pcd_colors))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    return pcd, pcd_array, pcd_colors


def separate_by_size(segments, using_npc=False):
    '''!
    Seperates dirt point cloud and weed point clouds by number of pts

    @param segments   dict containing all segments of point cloud

    @return weeds, dirt:  pointclouds containing weeds and dirt respecively
    '''

    if using_npc == True:
        weeds = segments[0]
        if len(segments) > 2:
            if np.shape(np.asarray(segments[1].points))[0] > np.shape(np.asarray(segments[2].points))[0]:
                dirt = segments[1]
            else:
                dirt = segments[2]
        else:
            dirt = segments[1]
    else:
        if np.shape(np.asarray(segments[0].points))[0] > np.shape(np.asarray(segments[1].points))[0]:
            weeds = segments[1]
            dirt = segments[0]
        else:
            weeds = segments[0]
            dirt = segments[1]

    return weeds, dirt


def labels_to_dict(pcd, labels):
    '''!
    Seperats pcd into dict of clusters based on labels

    @param pcd   the open3d point cloud in question
    @param labels   list of labels (eg from dbscan or kmeans, list of integers)

    @return dictionary containing pointclouds by cluster
    '''
    
    new_dict = {}
    for i in range(labels.max() + 1):
        idx = np.where(labels[:] == i)[0]
        new_dict[i] = pcd.select_by_index(idx)
    return new_dict


def assemble_weeds_from_label(leaves, best_S):
    '''!
    Assembles dictionary of point clouds representing weeds given labels from bestS for each leaf

    @param leaves   a list of numpy arrays. each numpy array is a leaf (usually from facet region growing)
    @param best_S   a list of integers. each element is the weed label for one leaf in leaves

    @return dictionary of point clouds where each point cloud is a weed
    '''

    new_dict = {}
    best_S = np.array(best_S)
    for i in range(np.max(best_S) + 1):
        idx = np.where(best_S[:] == i)[0]
        current_weed = []

        for j in idx:
            current_weed.append(leaves[j])

        current_weed = np.concatenate(current_weed, axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(current_weed)
        color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        # color= colors[i]
        pcd.paint_uniform_color(list(color[:3]))
        new_dict[i] = pcd
    
    return new_dict


def calculate_epsilon(pcd_array):
    '''!
    Uses method described here:
    https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc
    and kneed package to find the ideal epsilon for our data
    Currently out of use because "ideal" epsilon does not neccisarily produce correct weeds. Hand tuned value is 0.004.

    @param pcd_array   numpy array containing x,y, and z points from pt cloud

    @return ideal epsilon to use for DBSCAN based on data
    '''

    return 0.004
    # Calculate the distance from all points to nearest point
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(pcd_array)
    distances, indices = nbrs.kneighbors(pcd_array)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]

    # Find the elbow/knee associated with that distance
    kn = KneeLocator(x=range(len(distances)), y=distances, curve='convex')
    knee_idx = kn.knee
    epsilon = distances[knee_idx]

    if epsilon < 0.004:
        epsilon = 0.004

    return epsilon


def calculate_centroids(segments):
    '''!
    Calculates the centroid of each segment by avg position

    @param segments   dict containing point cloud segments

    @return list of centroids corresponding to each segment
    '''
    centroids = np.zeros((len(segments), 3))
    for i in range(len(segments)):
        centroids[i] = np.mean(np.asarray(segments[i].points), axis=0)
    return centroids


def calculate_normal(dirt):
    '''!
    Finds normal from dirt point cloud

    @param dirt   the open3d point cloud containing dirt

    @return normal of dirt
    '''

    # Apply RANSAC from open3d and get the best inliers
    plane_model, best_inliers = dirt.segment_plane(distance_threshold=0.0005,
                                                       ransac_n=3,
                                                       num_iterations=1000)

    if len(best_inliers) == 0:
        rospy.loginfo("Can't find dirt, Select both weed and dirt.")
        return None, None
    
    # inlier_cloud = dirt_pcd.select_by_index(best_inliers)
    # outlier_cloud = dirt_pcd.select_by_index(best_inliers, invert=True)
    # inlier_cloud.paint_uniform_color([1, 0, 0])
    # outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    [a, b, c, _] = plane_model
    if a < 0:
        normal = np.asarray([a, b, c])
    else:
        normal = -np.asarray([a, b, c])

    return normal


def float_to_rgb(float_rgb):
    '''!
    Converts a packed float RGB format to an RGB list

    @param float_rgb   RGB value packed as a float

    @return color (list): 3-element list of integers [0-255,0-255,0-255]
    '''

    s = struct.pack('>f', float_rgb)
    i = struct.unpack('>l', s)[0]
    pack = ctypes.c_uint32(i).value

    r = (pack & 0x00FF0000) >> 16
    g = (pack & 0x0000FF00) >> 8
    b = (pack & 0x000000FF)

    color = np.array([r, g, b]).astype(np.float) / 255.0

    return color