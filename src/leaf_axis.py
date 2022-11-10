from email.mime import base
import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pyquaternion import Quaternion


def calculate_radian(vector1, vector2):
    dot = np.dot(vector1, vector2)
    norm = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    cos = dot / norm
    return np.arccos(np.clip(cos, -1.0, 1.0))



def pca_pointcloud(pointcloud):
    """
    # 1. Apply PCA (component_1 -> l-axis )
    # 2. Calculate the angle between the centroid of p_init and determine the l-axis
    """
    centroid = np.mean(pointcloud, axis=0)

    # Translate the pointcloud to its centroid
    pointcloud_0 = pointcloud - centroid

    pca = PCA(n_components=3)
    pca.fit(pointcloud_0)
    component_1, component_2, component_3 = pca.components_
    
    # PCA does not gurantee the right hand coordinate vector direction
    # so we calclate the the 2nd pc by cross calculation of 1st pc and 3rd pc
    # this calculation is not neccesary for leaf flattening, but we calculate it to check the 
    # calculation is correct by visualizing the vectors.
    component_2 = np.cross(component_1, component_3) * -1.0

    # assure the component_1 vector (l-axis) to point the centroid
    check_component_1_radian = calculate_radian(centroid[0:2], component_1[0:2])
    if check_component_1_radian >= np.pi / 2.0:
        component_1 = -1.0 * component_1
        component_3 = -1.0 * component_3 
        # print("Component 1 has been reversed.") 
    
    check_component_3_radian = calculate_radian(np.array([0.0, 0.0, 1.0]), component_3)
    if check_component_3_radian >= np.pi / 2.0:
        component_2 = -1.0 * component_2
        component_3 = -1.0 * component_3
        # print("Component 3 has been reversed.")
    
    return component_1, component_2, component_3



def find_end_points(c1, c2, points, end_thresh=0.0015):
    # We want to translate the centroid of the leaf to the origin
    trans_points = points - np.mean(points, axis=0)


    # Then we make c1 into a unit vector
    c1 = c1 / np.linalg.norm(c1)
    c2 = c2 / np.linalg.norm(c2)
    z = np.array([0, 0, 1])
    x = np.array([1, 0, 0])

    # Find rotations to make axes align: https://dev.to/ku6ryo/how-to-align-two-xyz-coordinates-1ecn
    
    # Align c1 to z
    rot_axis_z = np.cross(c1, z) / np.linalg.norm(np.cross(c1, z))
    theta_z = np.arccos(np.dot(c1, z))
    qz = Quaternion(axis=rot_axis_z, angle=theta_z)
    R_z = qz.rotation_matrix

    # Align c2 to x
    rotated_c2 = qz.rotate(c2)
    rot_axis_x = np.cross(rotated_c2, x) / np.linalg.norm(np.cross(rotated_c2, x))
    theta_x = np.arccos(np.dot(rotated_c2, x))
    qx = Quaternion(axis=rot_axis_x, angle=theta_x)
    R_x = qx.rotation_matrix

    # Rotate all points to match axes
    rot_points1 = (R_z @ trans_points.T).T
    rot_points = (R_x @ rot_points1.T).T


    # Find base and tip points based on largest and smallest z value within a certain x
    within_thresh1 = np.where(np.abs(rot_points[:,0]) < end_thresh, rot_points[:,2], np.amax(rot_points[:,2]))
    end_idx1 = np.argmin(within_thresh1)
    end_point1 = points[end_idx1, :]

    within_thresh2 = np.where(np.abs(rot_points[:,0]) < end_thresh, rot_points[:,2], np.amin(rot_points[:,2]))
    end_idx2 = np.argmax(within_thresh2)
    end_point2 = points[end_idx2, :]

    return np.hstack((end_point1, end_point2))



def get_axis_and_ends(points, return_points=False):
    # Get axis that aligns with the main vein of the leaf and find the points associated with the tip and base of leaf
    # End points are returned as a vector of 6 where the first three values are end point 1 and the last three values are end point 2
    l_axis, h_axis, d_axis = pca_pointcloud(points)
    end_points = find_end_points(l_axis, h_axis, points)

    if return_points==True:
        return l_axis, end_points, points
    else:
        return l_axis, end_points


def main():
    M = o3d.io.read_point_cloud('/home/amasse/catkin_ws/src/plant_selector/segmentation_training/leaf_model_2.ply')
    leaf = np.asarray(M.points)

    c1, end_points = get_axis_and_ends(leaf)

    end_point1 = end_points[0]
    end_point2 = end_points[1]

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(leaf[:,0],leaf[:,1],leaf[:,2], color='green')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.quiver(end_point1[0], end_point1[1], end_point1[2], c1[0], c1[1], c1[2], length=0.0075, color='red')
    ax.quiver(end_point2[0], end_point2[1], end_point2[2], -c1[0], -c1[1], -c1[2], length=0.0075, color='blue')
    plt.show()



if __name__ == "__main__":
    main()