from audioop import avg
import ctypes
from os import fdatasync
import struct

import hdbscan
from statistics import mode
from math import atan, pi
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import rospy
from sensor_msgs import point_cloud2 as pc2
from tf.transformations import rotation_matrix


def kmeans_calculate_pose(points):
    '''!
    Uses kmeans and DBSCAN to cluster points and return the pose for the gripper
    MUCH better than RANSAC_calculate_pose

    @param points   a list of points from the point cloud from the selection

    @return list of weed centroids, normal associated with dirt
    '''
    # Create and format point cloud
    pcd_points = points[:, :3]
    float_colors = points[:, 3]
    pcd_colors = np.array((0, 0, 0))
    for x in float_colors:
        rgb = float_to_rgb(x)
        pcd_colors = np.vstack((pcd_colors, rgb))
    pcd_colors = np.delete(pcd_colors, 0, 0)
    pcd_points, pcd_colors = remove_height_outliers(pcd_points, pcd_colors)
    pcd_array = np.hstack((pcd_points, pcd_colors))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

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

    # Run dbscan to cluster into individual weeds, color them
    labels = np.array(weeds.cluster_dbscan(eps=0.004, min_points=10))
    max_label = labels.max()
    colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 1
    weeds.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # If we fail to find more than one weed cluster, leave
    if max_label < 1:
        return None, None
    
    # Seperate the weeds into dict of individual clusters
    weeds_segments = labels_to_dict(weeds, labels)
    o3d.visualization.draw_geometries(
        [weeds_segments[i] for i in range(len(weeds_segments))])
    

    # Find list of centroids of weeds and dirt normal
    weeds_centroids = calculate_centroids(weeds_segments)
    normal = calculate_normal(dirt)

    # For now only return the largest weed
    weeds_sizes = {}
    for i in range(len(weeds_centroids)):
        weeds_sizes[i] = np.shape(np.asarray(weeds_segments[i].points))[0]
    largest_segment = max(weeds_sizes, key=weeds_sizes.get)

    return weeds_centroids[largest_segment], normal


def RANSAC_calculate_pose(points):
    '''!
    Uses RANSAC and DBSCAN to cluster points and return the pose for the gripper
    Filters out smaller weeds, segmentation not great.

    @param points   a list of points from the point cloud from the selection

    @return list of weed centroids, normal associated with dirt
    '''
    # Create and format point cloud
    pcd_points = points[:, :3]
    float_colors = points[:, 3]
    pcd_colors = np.array((0, 0, 0))
    for x in float_colors:
        rgb = float_to_rgb(x)
        pcd_colors = np.vstack((pcd_colors, rgb))
    pcd_colors = np.delete(pcd_colors, 0, 0)
    pcd_points, pcd_colors = remove_height_outliers(pcd_points, pcd_colors)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

    # Do initial segmentation
    segments = RANSAC_segment_pcd(pcd, dist_thresh=0.005, epsilon=0.004)

    # Compare by heights to get pcds for weeds and dirt
    weeds, dirt = separate_by_height(segments, height_threshold=0.01)
    o3d.visualization.draw_geometries([weeds])

    # Run dbscan again to cluster into individual weeds, color them
    labels = np.array(weeds.cluster_dbscan(eps=0.004, min_points=5))
    max_label = labels.max()
    colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    weeds.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([weeds])

    # If we fail to find more than one weed cluster, leave
    if max_label < 1:
        return None, None
    
    # Seperate the weeds into dict of individual clusters
    weeds_segments = labels_to_dict(pcd, labels)

    # Find list of centroids of weeds and dirt normal
    weeds_centroids = calculate_centroids(weeds_segments)
    normal = calculate_normal(dirt)

    # For now only return the largest weed
    weeds_sizes = {}
    for i in range(len(weeds_centroids)):
        weeds_sizes[i] = np.shape(np.asarray(weeds_segments[i].points))[0]
    largest_segment = max(weeds_sizes, key=weeds_sizes.get)

    return weeds_centroids[largest_segment], normal


def RANSAC_segment_pcd(pcd, max_planes=100, dist_thresh=0.005, epsilon=0.004):
    '''!
    A test for the segmentation method presented in:
    https://towardsdatascience.com/how-to-automate-3d-point-cloud-segmentation-and-clustering-with-python-343c9039e4f5
   
    @param pcd   the open3d point cloud we want to segment
    @param max_planes   max number of planes for RANSAC loop
    @param dist_thresh   RANSAC distance threshold parameter
    @param epsilon   DBSCAN epsilon parameter

    @return dictionary of segments
    '''
    
    segment_models = {}
    segments = {}

    '''
    SEGMENTATION LOOP
    Run RANSAC and store inliers in segments dict
    Run DBSCAN on the inliers
    Count how many points are in each cluster and store in candidates
    Select the cluster with the most points and store in best_candidates
    Throw the clusters that are not the best candidate back into rest to be considered next iteration
    '''
    rest = pcd
    counter = 0
    for i in range(max_planes):
        colors = plt.get_cmap("tab20")(i)

        try:
            segment_models[i], inliers = rest.segment_plane(
                distance_threshold = dist_thresh,
                ransac_n = 3,
                num_iterations = 1000)
            segments[i] = rest.select_by_index(inliers)

            labels = np.array(segments[i].cluster_dbscan(eps=epsilon, min_points = 10))
    
            candidates = [len(np.where(labels==j)[0]) for j in np.unique(labels)]
            best_candidate = int(np.unique(labels)[np.where(candidates == np.max(candidates))[0]])

            rest = rest.select_by_index(inliers, invert=True) + segments[i].select_by_index(list(np.where(labels!=best_candidate)[0]))
            segments[i] = segments[i].select_by_index(list(np.where(labels== best_candidate)[0]))
            segments[i].paint_uniform_color(list(colors[:3]))

            counter += 1

        except:
            # print("No more segments can be made. Leaving loop...\n")
            break
     
    # print("RANSAC loop found ", counter, " segments.\n")

    # Draw the results
    o3d.visualization.draw_geometries(
        [segments[i] for i in range(counter)] + [rest])
    
    return segments


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


def separate_by_size(segments):
    '''!
    Seperates dirt point cloud and weed point clouds by number of pts

    @param segments   dict containing all segments of point cloud

    @return weeds, dirt pointclouds containing weeds and dirt respecively
    '''

    # Assume the largest segment is dirt

    # Find all sizes of each segment
    segment_sizes = {}
    for i in range(len(segments)):
        segment_sizes[i] = np.shape(np.asarray(segments[i].points))[0]
    largest_segment = max(segment_sizes, key=segment_sizes.get)

    # Catagorize based on size
    if largest_segment == 0:
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