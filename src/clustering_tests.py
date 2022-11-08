from audioop import avg
import ctypes
from os import fdatasync
import struct
from turtle import fd

import hdbscan
from statistics import mode
from math import atan, pi
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import open3d as o3d
import scipy as sp
import random
import itertools
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, BisectingKMeans
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from scipy import optimize

import rospy
from sensor_msgs import point_cloud2 as pc2
from tf.transformations import rotation_matrix

import nearest_prototype_classifier_test as npc
import facet_region_growing as frg
import leaf_axis as la


def DBSCAN_calculate_pose(points, algorithm='npc', weights=[0,100,100,0,100,0], return_multiple_grasps=False):
    '''!
    Uses DBSCAN to cluster weeds and calculate pose

    @param points   a list of points from the point cloud from the selection
    @param algorithm   string containing algorithm choice. options include 'kmeans-optimized', 'kmeans-redmean', 'kmeans-euclidean', 'bi-kmeans', 'spectral', 'ward', 'npc
    @param weights    a list containing 6 weights for weighted kmeans implementation. weights must be >= 0
    @param return_multiple_grasps   Boolean indicating whether or not to return multiple grasps for selections of multiple weeds

    @return list of weed centroids, normal associated with dirt
    '''
    # Do initial segmentation
    weeds, dirt = initial_segmentation(points, algorithm, weights)
    
    weeds_array = np.asarray(weeds.points)
    if weeds_array.shape[0] < 2:
        return None, None
    # weeds.paint_uniform_color([0.01, 0.5, 0.01])
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
    weeds_segments = labels_to_dict(weeds, labels)
    


    # Find list of centroids of weeds and dirt normal
    weeds_centroids = calculate_centroids(weeds_segments)
    normal = calculate_normal(dirt)

    if len(weeds_centroids) < 1:
        return None, None


    # If we want multiple grasps, return multiple grasps
    if return_multiple_grasps == True:
        return weeds_centroids, normal



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

    # Do initial segmentation and convert to numpy array
    weeds, dirt = initial_segmentation(points, algorithm, weights)
    weeds_array = np.asarray(weeds.points)
    if weeds_array.shape[0] < 2:
        return None, None

    weeds.paint_uniform_color([0.01, 0.5, 0.01])
    o3d.visualization.draw_geometries([weeds])

    # Use facet region growing to get individual leaves
    leaves = frg.facet_leaf_segmentation(weeds_array)


    # Display leaf segments
    leaf_pcs = []
    for i in range(len(leaves)):
        color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        leaf = o3d.geometry.PointCloud()
        leaf.points = o3d.utility.Vector3dVector(leaves[i])
        leaf.paint_uniform_color(list(color[:3]))
        leaf_pcs.append(leaf)
    o3d.visualization.draw_geometries(
        [leaf_pcs[i] for i in range(len(leaves))])


    # Get leaf axes and base points
    edges = np.array([0, 0, 0, 0, 0, 0])
    for leaf in leaves:
        _, current_edges = la.get_axis_and_ends(leaf)
        edges = np.vstack((edges, current_edges))
    edges = np.delete(edges, 0, 0)
    

    # S0 = [math.floor(len(leaf_pcs) / 2)] * len(leaf_pcs)
    # S0 = [0] * len(leaf_pcs)
    # result = optimize.minimize(LstSqrs_find_centroid, S0, args=(edges), options={'disp': True})
    # best_S = result.x

    # bounds = [(0.0, len(leaf_pcs))] * len(leaf_pcs)
    # cons = ({'type': 'eq', 'fun':is_int})
    # results = dict()
    # results['shgo'] = optimize.shgo(LstSqrs_find_centroid, bounds, args=([edges]), constraints=cons, options={'disp': True})
    # best_S = results['shgo'].x

    # res, weeds_centroids = LstSqrs_find_centroid(correct_S, edges, return_cents=True)

    best_S, lowest_cost, weeds_centroids = brute_force_optimize(edges)
    print("Best S: ", best_S)
    print("Lowest Cost: ", lowest_cost)

    normal = calculate_normal(dirt)

    if len(weeds_centroids) < 1:
        return None, None


    # If we want multiple grasps, return multiple grasps
    if return_multiple_grasps == True:
        return weeds_centroids, normal
    else:
        return weeds_centroids[0], normal


def is_int(S):
    # Determine if all values in S are integers
    for s in S:
        if math.floor(s) != s:
            return False
    
    return True


def brute_force_optimize(E, max_weeds=5):
    # Find all possible S for the number of edges and max number of possible weeds
    num_edges = len(E)
    combinations = list(itertools.combinations_with_replacement(range(max_weeds), num_edges))

    # Test all S with least squares to find the lowest cost
    lowest_cost = math.inf
    best_centroids = 0
    best_S = 0
    for i in range(len(combinations)):
        current_S = list(combinations[i])
        current_cost, current_centroids = LstSqrs_find_centroid(current_S, E, return_cents = True)
        
        if current_cost < lowest_cost:
            lowest_cost = current_cost
            best_centroids = current_centroids
            best_S = current_S
    
    
    return best_S, lowest_cost, best_centroids


def LstSqrs_find_centroid(S, E, return_cents=False):
    '''!
    Uses least squares to find the centroids of plants given their leaf assignments

    @param S   a list of integer labels corresponding to each leaf
    @param E   An Nx6 matrix where each row corresponds to a line (as defined by two 3d points) representing a leaf

    @return sum of the residuals from least squares, list of centroids for each weed
    '''
    
    S = [round(i) for i in S]
    S = [int(i) for i in S]
    S = np.array(S)
    
    # Seperate E into weeds based on S
    plants = {}
    for i in range(S.max()+1):
        plant_exists = False
        idx = np.where(S[:] == i)[0]
        current_plant = np.array([0, 0, 0, 0, 0, 0])
        for j in idx:
            current_plant = np.vstack((current_plant, E[j, :]))
            plant_exists = True
        
        # If the current S doesnt assign current index to anything set it to an empty list
        if plant_exists:
            current_plant = np.delete(current_plant, 0, 0)
            plants[i] = current_plant
        else:
            plants[i] = np.array([])
    
   
    
    # Do least squares for every plant and store the centroids and residuals in lists
    centroids_list = []
    residuals = []
    for i in range(len(plants)):
        edges = plants[i]
        
        # If the current plant index actually has a plant in it, do the stuff
        if np.any(edges):
            # Calculate matricies A and b for Least Squares and solve it for each plant
            # A is the sum of all A_i for each edge, e_i 
            A = np.array([0, 0, 0])
            b = np.array([[0]])
            for e in range(edges.shape[0]):
                point_a = edges[e,:3]
                point_b = edges[e, 3:]
                d = point_b - point_a
                f = (2 * d) / np.linalg.norm(d)**2
                R = np.dot(-point_a, d)
                A_i = np.array([[2-f[0]*d[0], -f[0]*d[1], -f[0]*d[2]], [-f[1]*d[0], 2-f[1]*d[1], -f[1]*d[2]], [-f[2]*d[0], -f[2]*d[1], 2-f[2]*d[2]]])
                b_i = np.array([[(2 * point_a[0]) - (f[0] * R)], [(2 * point_a[1]) - (f[1] * R)], [(2 * point_a[2]) - (f[2] * R)]])
                A = np.vstack((A, A_i))
                b = np.vstack((b, b_i))
        
            A = np.delete(A, 0, 0)
            b = np.delete(b, 0, 0)

            # Perform least squares to get centroid guess cent and residuals
            cent, res, _, _ = np.linalg.lstsq(A, b, rcond=None)
            centroids_list.append(list(cent))

            # If no residual is givn (ie the plant assignment only contains one leaf), set residual to 0
            if len(res) > 0:
                residuals.append(res[0])
            else:
                residuals.append(0)

    # Remove empty values
    if return_cents == True:
        return sum(residuals), centroids_list
    else:
        return sum(residuals)


def initial_centroid_guess(E):
    total_points = 2 * len(E)
    sum_points = np.array([0, 0, 0])
    for edge in E:
        sum_points = sum_points + edge[0]
        sum_points = sum_points + edge[1]

    return sum_points / total_points


def initial_segmentation(points, algorithm='npc', weights=[0,100,100,0,100,0]):
    '''!
    Does initial segmentation --> Uses specified algorithm to seperate weeds from dirt

    @param points   a list of points from the point cloud from the selection
    @param algorithm   string containing algorithm choice. options include 'kmeans-optimized', 'kmeans-redmean', 'kmeans-euclidean', 'bi-kmeans', 'spectral', 'ward', 'npc
    @param weights    a list containing 6 weights for weighted kmeans implementation. weights must be >= 0

    @return point cloud of weeds, point cloud of dirt
    '''
    
    # Create and format point cloud
    pcd, pcd_array, pcd_colors = array_2_pc(points)
    # o3d.visualization.draw_geometries([pcd], window_name="Original")
    # np.save("/home/amasse/catkin_ws/src/plant_selector/weed_eval/saved.npy", points)
    # o3d.io.write_point_cloud("/home/amasse/catkin_ws/src/plant_selector/weed_eval/saved.pcd", pcd)


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


    max_label = labels.max()
    if max_label < 1:
        return None, None
    colors = plt.get_cmap("Paired")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    # colors[labels == 1] = 0
    # colors[labels == 2] = 0.5
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([pcd], window_name="Initial Segmentation")

    

    # Seperate segments into dict of idividual clusters
    segments = labels_to_dict(pcd, labels)



    # Compare by sizes to get pcds for weeds and dirt, filter out outliers
    weeds, dirt = separate_by_size(segments, using_npc)
    _, ind = weeds.remove_radius_outlier(nb_points=7, radius=0.007)
    if len(ind) != 0:
        weeds = weeds.select_by_index(ind)
    
    return weeds, dirt


def calculate_radian(vector1, vector2):
    dot = np.dot(vector1, vector2)
    norm = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    cos = dot / norm
    return np.arccos(np.clip(cos, -1.0, 1.0))


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
                # weighted distance function
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


def remove_height_outliers(pcd_points, pcd_colors, max_height = -0.2):
    '''!
    Removes all point cloud points above max_height

    @param pcd_points   [x,3] numpy array containing x, y, and z coords
    @param pcd_colors   [x,3] numpy array containing r, g, and b colors
    @param max_height   float containing height cutoff. default -0.2

    @return pcd_points, pcd_colors with points above max_height removed
    '''
    pcd_array = np.hstack((pcd_points, pcd_colors))

    filtered_array = np.delete(pcd_array, np.where(
        pcd_array[:,2] > max_height)[0], axis=0)

    pcd_points = filtered_array[:, :3]
    pcd_colors = filtered_array[:, 3:]

    return pcd_points, pcd_colors


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


def array_2_pc(points):
    '''!
    take in array of points and format it to be usable

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
    # pcd_points, pcd_colors = remove_height_outliers(pcd_points, pcd_colors)
    pcd_array = np.hstack((pcd_points, pcd_colors))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    return pcd, pcd_array, pcd_colors


def separate_by_size(segments, using_npc=False):
    '''!
    Seperates dirt point cloud and weed point clouds by number of pts

    @param segments   dict containing all segments of point cloud

    @return weeds, dirt pointclouds containing weeds and dirt respecively
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
    @param labels   list of labels (eg from dbscan or kmeans)

    @return dictionary containing pointclouds by cluster
    '''
    
    new_dict = {}
    for i in range(labels.max() + 1):
        idx = np.where(labels[:] == i)[0]
        new_dict[i] = pcd.select_by_index(idx)
    return new_dict


def calculate_epsilon(pcd_array):
    '''!
    Uses method described here:
    https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc
    and kneed package to find the ideal epsilon for our data

    @param pcd_array   numpy array containing x,y, and z points from pt cloud

    @return ideal epsilon to use for DBSCAN based on data
    '''
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
    # print("Calculated eps: ", epsilon)

    epsilon = 0.004

    # if epsilon < 0.004:
    #     epsilon = 0.004
        # print("switched to default epsilon")

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
    """ Converts a packed float RGB format to an RGB list

        Args:
            float_rgb: RGB value packed as a float

        Returns:
            color (list): 3-element list of integers [0-255,0-255,0-255]
    """
    s = struct.pack('>f', float_rgb)
    i = struct.unpack('>l', s)[0]
    pack = ctypes.c_uint32(i).value

    r = (pack & 0x00FF0000) >> 16
    g = (pack & 0x0000FF00) >> 8
    b = (pack & 0x000000FF)

    color = np.array([r, g, b]).astype(np.float) / 255.0

    return color