import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm
import open3d as o3d
import matplotlib.pyplot as plt
import math
import random


def spacial_characteristics(points, K=8, plane_thresh=0.003):
    # Algorithm 1: IPCA for spacial characteristics of each point
    tree = KDTree(points)
    normals = []
    smoothness = []
    print("\nComputing Smoothness and Normals:")
    for i in tqdm(range(points.shape[0])):
        # Find K nearest neighbors to current point, add to list of points [3, K]
        distances, indices = tree.query(points[i,:], K)
        neighbors_list = points[indices, :].T

        changing_neighbors = True

        while changing_neighbors == True:
            # Compute covariance matrix, C, of neighbors_list
            C_i = (1.0 / K) * np.dot(neighbors_list, neighbors_list.T)

            # Compute eigenvalues and eigenvectors of C
            e_vals, e_vects = np.linalg.eig(C_i)
            idx = e_vals.argsort()
            e_vals = np.flip(e_vals[idx])
            e_vects = np.fliplr(e_vects[:, idx])

            # Extract the normal and smoothness of our current point
            n_i = e_vects[:, 2]
            s_i = e_vals[1] / e_vals[2]

            # Find the distance between current point and all in neighbors_lost, remove point is dist greater than sigma
            for n in range(neighbors_list.shape[1]):
                current_dist = np.linalg.norm(np.dot(n_i.T, (neighbors_list[:,n] - points[i,:]))) / np.linalg.norm(n_i)
                if current_dist > plane_thresh:
                    new_neighbors_list = np.delete(neighbors_list, n, axis=1)
                else:
                    new_neighbors_list = neighbors_list
            
            # When the size of our neighbors list is no longer changing (ie. it is stable), append current normal and smoothness
            if new_neighbors_list.shape[1] == neighbors_list.shape[1]:
                changing_neighbors = False
            neighbors_list = new_neighbors_list
        normals.append(n_i)
        smoothness.append(s_i)

    characteristics = [normals, smoothness]
    return characteristics



def coarse_segmentation(points, characteristics, K, grow_thresh, angle_thresh, plane_thresh):
    seed_points = []
    labels = -1 * np.ones((points.shape[0]))
    label_counter = 0
    used = np.zeros((points.shape[0]))
    smoothness = np.asarray(characteristics[1])
    normals = np.asarray(characteristics[0])


    tree = KDTree(points)


    # Assign labels to each point
    print("\nCoarse Segmentation:")
    for i in tqdm(range(points.shape[0])):
        # If we have not used this point yet
        if used[i] == 0:
            # Mark this point as used
            used[i] = 1

            # Find the neighbors of our current point
            distances, indices = tree.query(points[i, :], K)
            neighbor_smoothness = smoothness[indices]
            neighbor_normals = normals[indices]
            neighbors_list = points[indices, :].T

            # Pick out the neighbor with the highest smoothness, add it to list of seed_points
            idx = neighbor_smoothness.argsort()
            seed_normal = neighbor_normals[idx[0]]
            seed_point = neighbors_list[:, idx[0]]
            seed_points.append(seed_point)
            labels[i] = label_counter
    

            # Use the seed point and the three conditions to grow facet
            for k in range(points.shape[0]):
                # Do only if we have not used this point yet
                if used[k] == 0:
                    
                    # If the current point meets all 3 conditions w.r.t. the seed point, the point belongs to the region of that seed
                    cond1 = np.abs(np.linalg.norm(points[k, :] - seed_point)) <= grow_thresh
                    cond2 = np.abs(np.arccos(np.clip(np.dot(normals[k, :], seed_normal), -1, 1))) <= angle_thresh
                    cond3 = np.linalg.norm(np.dot(seed_normal.T, (points[k, :] - seed_point))) / np.linalg.norm(seed_normal) <= plane_thresh
                    if cond1 and cond2 and cond3:
                        # Mark this point as used and update label
                        labels[k] = label_counter
                        used[k] = 1
            
            label_counter += 1

    return seed_points, labels



def kmeans_refinement(points, seed_points, sphere_radius):
    # KMeans Facet Refinement


    d = np.ones((points.shape[0])) * math.inf
    unstable_clusters = True

    # Repeat finding new seed points until the clusters are not moving anymore
    print("\nK-Means Facet Refinement:")
    while unstable_clusters:
        K_labels = -1 * np.ones((points.shape[0]))
        for s in range(len(seed_points)):
            # Find the distance from current seed point to all points in cloud and also all other seeds
            all_dists = np.abs(np.linalg.norm(seed_points[s] - points, axis=1))
            seed_dists = np.abs(np.linalg.norm(seed_points[s] - seed_points, axis=1))


            # Select the points within sphere_radius
            possible_idxs = np.argwhere(all_dists < sphere_radius)[:, 0]
            seed_idxs = np.argwhere(seed_dists < sphere_radius)[:, 0]
                

            # Classify points close to seed into that facet
            for i in possible_idxs:
                Distance = []
                for c in seed_idxs:
                    Distance = np.abs(np.linalg.norm(seed_points[c] - points[i]))
                    if Distance <= d[i]:
                        d[i] = Distance
                        # print("Point ", i, " closest to seed ", c)
                        K_labels[i] = c


        # Cluster unclustered points to nearest seed point
        for i in range(points.shape[0]):
            if K_labels[i] == -1.0:
                dists = np.abs(np.linalg.norm(points[i] - seed_points, axis=1))
                K_labels[i] = np.argmin(dists)
        
        

        # print(seed_points)
        # Set new seed points to be the centroids of the facets
        no_change = 0
        for i in range(len(seed_points)):
            idx = np.where(K_labels[:] == i)[0]

            if idx.any():
                facet = points[idx]
                new_seed_point = np.mean(facet, axis=0)
            else:
                new_seed_point = seed_points[i]

            # Count number of centroids that remain approximately the same
            m_seed_point = np.round(np.linalg.norm(seed_points[i]), decimals=3)
            m_new_seed_point = np.round(np.linalg.norm(new_seed_point), decimals=3)
            if m_seed_point == m_new_seed_point:
                no_change += 1

            seed_points[i] = new_seed_point
        

        # Keep going only if centroids are not changing
        if no_change >= len(seed_points):
            unstable_clusters = False

    # print("Found ", max(K_labels), " labels.")
    return K_labels



def facet_over_segmentation(points, characteristics, K=8, grow_thresh=0.006, angle_thresh=0.85, plane_thresh=0.005, sphere_radius=0.005):
    # Algorithm 2: Facet over-segmentation
    seed_points, coarse_labels = coarse_segmentation(points, characteristics, K, grow_thresh, angle_thresh, plane_thresh)
    labels = kmeans_refinement(points, seed_points, sphere_radius)
    return coarse_labels



def get_facets_from_labels(points, labels, characteristics):
    facets = []
    normals = np.asarray(characteristics[0])

    for i in range(int(labels.max()) + 1):
        idx = np.where(labels[:] == i)[0]
        facets.append([points[idx], normals[idx]])
    return facets



def are_adjacent(f_i, f_j, adj_thresh):
    adj_count = 0
    for i in range(len(f_i)):
        all_dists = np.abs(np.linalg.norm(f_i[i] - f_j, axis=1)) - adj_thresh
        in_range = all_dists[all_dists <= 0]
        if np.any(in_range):
            adj_count += 1
    
    if adj_count > 0:
        return True
    else:
        return False



def facet_region_growing(facets, grow_thresh=0.085, adj_thresh=0.0015):
    used = np.zeros((len(facets)))
    leaves = []

    print("\nFacet Region Growing:")
    for k in tqdm(range(len(facets))):
        if used[k] == 0:
            # Append the current facet to the queue and mark it as used
            A = []
            A.append(facets[k])
            used[k] = 1

            # Grow the facet until the queue is empty then move onto next facet
            while len(A) > 0:
                # Remove the first facet from the queue
                facet_i = A.pop(0)
                current_leaf = []

                for j in range(len(facets)):
                    # print("j = ", j)
                    if used[j] == 0:

                        # Calculate distance between facet_i and f_j
                        facet_i_centroid = np.mean(facet_i[0], axis=0)
                        facet_i_avg_normal = np.mean(facet_i[1], axis=0)
                        facet_j_centroid = np.mean(facets[j][0], axis=0)
                        d = np.linalg.norm(np.dot(facet_i_avg_normal.T, (facet_j_centroid - facet_i_centroid))) / np.linalg.norm(facet_i_avg_normal)


                        # If this facet is adjacent and close enough to the previous one, grow the facet
                        adj = are_adjacent(facet_i[0], facets[j][0], adj_thresh)
                        if d < grow_thresh and adj:
                            f = np.append(facet_i[0], facets[j][0], axis=0)
                            n = np.append(facet_i[1], facets[j][1], axis=0)
                            current_leaf = [f,n]
                            facet_i = current_leaf
                            used[j] = 1
                            
                
                if len(current_leaf) > 0:
                    A.append(current_leaf)
            leaves.append(facet_i[0])
    return leaves



def facet_leaf_segmentation(points, min_points = 10):
    # Leaf segmentation via facet region growing
    # Method presented here: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6263610/
    
    characteristics = spacial_characteristics(points)
    labels = facet_over_segmentation(points, characteristics)
    facets = get_facets_from_labels(points, labels, characteristics)
    all_leaves = facet_region_growing(facets)

    good_leaves = []
    for leaf in all_leaves:
        if len(leaf) > min_points:
            good_leaves.append(leaf)

    return good_leaves



def main():
    # file_loc = '/home/amasse/catkin_ws/src/plant_selector/segmentation_training/leaf_model_1.ply'
    file_loc = '/home/amasse/catkin_ws/src/plant_selector/weed_eval/segmented_weeds.pcd'
    # file_loc = '/home/amasse/catkin_ws/src/plant_selector/weed_eval/larger_weeds.pcd'
    weeds = o3d.io.read_point_cloud(file_loc)
    pcd_points = np.asarray(weeds.points)

    leaves = facet_leaf_segmentation(pcd_points)

    # Convert leaves into point clouds and display
    leaf_pcs = []
    for i in range(len(leaves)):
        color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        leaf = o3d.geometry.PointCloud()
        leaf.points = o3d.utility.Vector3dVector(leaves[i])
        leaf.paint_uniform_color(list(color[:3]))
        leaf_pcs.append(leaf)

    o3d.visualization.draw_geometries(
        [leaf_pcs[i] for i in range(len(leaves))])

    
    # facet_pcs = []
    # for i in range(len(facets)):
    #     color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
    #     fac = o3d.geometry.PointCloud()
    #     fac.points = o3d.utility.Vector3dVector(facets[i][0])
    #     fac.paint_uniform_color(list(color[:3]))
    #     facet_pcs.append(fac)

    # o3d.visualization.draw_geometries(
    #     [facet_pcs[i] for i in range(len(facets))])



if __name__ == "__main__":
    main()