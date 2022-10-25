from audioop import avg
import ctypes
from os import fdatasync
import struct
from turtle import fd

import hdbscan
from statistics import mode
from math import atan, pi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import open3d as o3d
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, BisectingKMeans
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

import rospy
from sensor_msgs import point_cloud2 as pc2
from tf.transformations import rotation_matrix

import nearest_prototype_classifier_test as npc


def HDBSCAN_kmeans_calculate_pose(points):
    '''!
    Uses kmeans and HDBSCAN to cluster points and return the pose for the gripper

    @param points   a list of points from the point cloud from the selection

    @return list of weed centroids, normal associated with dirt
    '''
    # Create and format point cloud
    pcd, pcd_array, pcd_colors = array_2_pc(points)


    # Do kmeans clustering on original point cloud, color results
    k = 2
    kmeans = KMeans(n_clusters=k, random_state=0).fit(pcd_array)
    labels = kmeans.labels_
    max_label = labels.max()
    colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])


    # Seperate segments into dict of idividual clusters
    segments = labels_to_dict(pcd, labels)


    # Compare by sizes to get pcds for weeds and dirt
    weeds, dirt = separate_by_size(segments)
    weeds_array = np.asarray(weeds.points)


    # Run hdbscan to cluster into individual weeds, color them
    clustering = hdbscan.HDBSCAN(
        min_cluster_size=10,
        gen_min_span_tree=True,
        allow_single_cluster=1,
        algorithm="prims_balltree").fit(weeds_array)
    labels = clustering.labels_
    max_label = labels.max()
    colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 1
    weeds.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([weeds], window_name="prims_balltree")


    # If we fail to find more than one weed cluster, leave
    if max_label < 1:
        return None, None
    

    # Seperate the weeds into dict of individual clusters
    weeds_segments = labels_to_dict(weeds, labels)
    # o3d.visualization.draw_geometries(
    #     [weeds_segments[i] for i in range(len(weeds_segments))], window_name="HDBSCAN")
    

    # Find list of centroids of weeds and dirt normal
    weeds_centroids = calculate_centroids(weeds_segments)
    normal = calculate_normal(dirt)


    # For now only return the largest weed
    weeds_sizes = {}
    for i in range(len(weeds_centroids)):
        weeds_sizes[i] = np.shape(np.asarray(weeds_segments[i].points))[0]
    largest_segment = max(weeds_sizes, key=weeds_sizes.get)

    return weeds_centroids[largest_segment], normal


def DBSCAN_calculate_pose(points, algorithm='npc', weights=[0,100,100,0,100,0], return_multiple_grasps=False):
    '''!
    Uses kmeans and DBSCAN to cluster points and return the pose for the gripper

    @param points   a list of points from the point cloud from the selection
    @param algorithm   string containing algorithm choice. options include 'kmeans-optimized', 'kmeans-redmean', 'kmeans-euclidean', 'bi-kmeans', 'spectral', 'ward', 'npc

    @return list of weed centroids, normal associated with dirt
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
    weeds_array = np.asarray(weeds.points)
    if weeds_array.shape[0] < 2:
        return None, None
    # weeds.paint_uniform_color([0.01, 0.5, 0.01])
    o3d.io.write_point_cloud("/home/amasse/catkin_ws/src/plant_selector/weed_eval/segmented_weeds2.pcd", weeds)
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