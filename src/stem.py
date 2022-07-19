"""
finding-stem.py
This python file is used to test the algorithm to find the actual stem of the weed.
Created by: Miguel Munoz
Date: July 11th, 2022
"""
from skspatial.objects import Plane, Point, Vector
from skspatial.plotting import plot_3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
import plant_extraction
import sys, ast

import rospy


def proj_point2plane(point_in_plane, point_to_project, normal):
    # Return the projection of the point to the plane
    return point_to_project - np.dot(point_to_project - point_in_plane, normal) * normal


def plot_plane_points(point_in_plane, normal, size):
    # Get d parameter for plane equation
    d = -point_in_plane.dot(normal)
    # Define x and y limits for the plane
    x_lims = np.linspace(point_in_plane[0] - size, point_in_plane[0] + size, 10)
    y_lims = np.linspace(point_in_plane[1] - size, point_in_plane[1] + size, 10)
    # Create meshgrid for plane
    xx, yy = np.meshgrid(x_lims, y_lims)
    # Get z value of every point
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
    return xx, yy, z


# Open PCD and get np array of points
pcd = o3d.io.read_point_cloud('/home/christianforeman/catkin_ws/src/plant_selector/bags/weed-selection.pcd')
dirt_pcd = o3d.io.read_point_cloud('/home/christianforeman/catkin_ws/src/plant_selector/bags/dirt_points.pcd')
plane_model, best_inliers = dirt_pcd.segment_plane(distance_threshold=0.0005,
                                                   ransac_n=3,
                                                   num_iterations=1000)
[a, b, c, _] = plane_model
normal = np.asarray([a, b, c])

points = np.asarray(pcd.points)

# o3d.visualization.draw_geometries([pcd])

# _, filtered_points = pcd.remove_radius_outlier(nb_points=8, radius=0.005)
# points = raw_points[filtered_points]

# Define a point in the plane, as well as its normal
point_in_plane = np.asarray(np.mean(points, axis=0))
# normal = np.asarray([0.6410022562021247, -0.16509240724820431, -0.7495736152058574])

# Create empty array for the projected points
points_projected = np.zeros(points.shape)
for point in range(len(points)):
    # Go through each point in points
    point_to_project = points[point]
    # Project the ith point in the plane
    point_projected = proj_point2plane(point_in_plane, point_to_project, normal)
    # Append each projected point to the array
    points_projected[point] = point_projected

# Create o3d PointCloud
weed_projected = o3d.geometry.PointCloud()
# Save points to new PointCloud
weed_projected.points = o3d.utility.Vector3dVector(points_projected)

o3d.io.write_point_cloud('/home/christianforeman/catkin_ws/src/plant_selector/bags/sup.pcd', weed_projected)

gen_centroid = np.mean(points_projected, axis=0)

# Apply DBSCAN and get labels
labels = np.array(weed_projected.cluster_dbscan(eps=0.002, min_points=4))
# Define number of leaves
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"DBSCAN found: {n_clusters} clusters")
colors = plt.get_cmap("tab20")(labels / (n_clusters if n_clusters > 0 else 1))
colors[labels < 0] = 0
weed_projected.colors = o3d.utility.Vector3dVector(colors[:, :3])
# o3d.visualization.draw_geometries([weed_projected])

# Define coordinates for each leaf and find centroid
leaves = list()
pcas = list()
fitted_pcas = list()
origins = list()
pcs = list()
vectors = list()
# centroids = list()
for cluster in range(n_clusters):
    leaves.append(points_projected[np.where(labels[:] == cluster)])
    pcas.append(PCA(n_components=1))
    fitted_pcas.append(pcas[cluster].fit(leaves[cluster]))
    origins.append(np.mean(leaves[cluster], axis=0))
    pcs.append(pcas[cluster].components_[0])
    vectors.append(np.append(origins[cluster], pcs[cluster]))
    # centroids.append(np.mean(leaves[cluster], axis=0))

# centroid_of_centroids = np.mean(centroids[:], axis=0)
# centroids = np.asarray(centroids)

soa = np.asarray(vectors)
X, Y, Z, U, V, W = zip(*soa)

# Create plane function
xx, yy, z = plot_plane_points(point_in_plane, normal, size=0.01)

# Plot both plane and points
fig = plt.figure(1)
handlers = ['Projected points', 'General Centroid', 'Centroid of centroids', 'General centroid', 'Leaves to stem']
ax = fig.add_subplot(projection='3d')
# Plot the surface
ax.plot_surface(xx, yy, z, alpha=0.2)
# Plot the points
# Projected points (green)
ax.scatter(points_projected[:, 0], points_projected[:, 1], points_projected[:, 2],
           c='g', s=50, label=handlers[0])
# # Centroids per leaf (red)
# ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
#            c='r', s=100, label=handlers[1])
# # Centroid of centroids (black)
# ax.scatter(centroid_of_centroids[0], centroid_of_centroids[1], centroid_of_centroids[2],
#            c='k', s=100, label=handlers[2])
# General centroid of the hole weed (blue)
ax.scatter(gen_centroid[0], gen_centroid[1], gen_centroid[2],
           c='b', s=100, label=handlers[1])
# # Plot lines connecting individual centroids to centroid of centroids
# for centroid in range(n_clusters):
#     if centroid == 0:
#         ax.plot([centroids[centroid][0], centroid_of_centroids[0]],
#                 [centroids[centroid][1], centroid_of_centroids[1]],
#                 [centroids[centroid][2], centroid_of_centroids[2]],
#                 'k--', label=handlers[4])
#     else:
#         ax.plot([centroids[centroid][0], centroid_of_centroids[0]],
#                 [centroids[centroid][1], centroid_of_centroids[1]],
#                 [centroids[centroid][2], centroid_of_centroids[2]],
#                 'k--', label='_nolegend_')
# ax.quiver(X, Y, Z, U, V, W)
plt.title("PCA per Leaf", fontsize=20, fontweight='bold')
ax.set_xlabel("x-coordinates", fontsize=20)
ax.set_ylabel("y-coordinates", fontsize=20)
ax.set_zlabel("z-coordinates", fontsize=20)
ax.xaxis.set_tick_params(labelsize=7)
ax.yaxis.set_tick_params(labelsize=7)
ax.zaxis.set_tick_params(labelsize=7)
ax.xaxis.labelpad = 20
ax.yaxis.labelpad = 20
ax.zaxis.labelpad = 20
ax.legend(loc='best', fontsize='large')
plt.show()