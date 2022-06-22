# Import useful libraries and functions
import open3d as o3d
import numpy as np
import rospy
import ros_numpy
import matplotlib.pyplot as plt
from math import atan, sin, cos
from sensor_msgs.msg import PointCloud2
from statistics import mode
from std_msgs.msg import ColorRGBA
from matplotlib import colors
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from arc_utilities.tf2wrapper import TF2Wrapper
from sklearn.decomposition import PCA

camera_frame = 'zed2i_left_camera_frame'


class WeedExtractor:
    def __init__(self, camera_frame):
        rospy.init_node("weed_extraction")
        # Initialize publishers for PCs, arrow and planes
        # SYNTAX: pub = rospy.Publisher('topic_name', geometry_msgs.msg.Point, queue_size=10)
        # Second argument was imported in the beginning
        self.src_pub = rospy.Publisher("source_weed", PointCloud2, queue_size=10)
        self.inliers_pub = rospy.Publisher("inliers_weed", PointCloud2, queue_size=10)
        self.arrow_pub = rospy.Publisher("normal", Marker, queue_size=10)
        self.plane_pub = rospy.Publisher("dirt_plane", PointCloud2, queue_size=10)

        self.selection_sub = rospy.Subscriber("/rviz_selected_points", PointCloud2, queue_size=10)

        self.frame_id = camera_frame

    def plot_pointcloud_rviz(self, pub: rospy.Publisher, xs, ys, zs):
        """
        This function plots pointcloud in Rviz.
        :param pub: ROS Publisher
        :param xs: x coordinates of points
        :param ys: y coordinates of points
        :param zs: z coordinates of points
        :return: None.
        """
        # Create a list of the coordinates of the points
        list_of_tuples = [(x, y, z) for x, y, z in zip(xs, ys, zs)]
        # Define name and type
        dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32)]
        # Define a tuple with coordinates and type
        np_record_array = np.array(list_of_tuples, dtype=dtype)
        # Construct ROS message
        msg = ros_numpy.msgify(PointCloud2, np_record_array, frame_id=camera_frame, stamp=rospy.Time.now())
        # Publish the message
        pub.publish(msg)

    def rviz_arrow(self, start, direction, name, thickness, length_scale, color):
        """
        This function displays an arrow in Rviz.
        :param pub: ROS Publisher
        :param start: vector with coordinates of the start point (origin)
        :param direction: vector that defines de direction of the arrow
        :param name: namespace to display in Rviz
        :param thickness: thickness of the arrow
        :param length_scale: length of the arrow
        :return: None
        """
        color_msg = ColorRGBA(*colors.to_rgba(color))

        # Define ROS message
        msg = Marker()
        msg.type = Marker.ARROW
        msg.action = Marker.ADD
        msg.ns = name
        msg.header.frame_id = camera_frame
        msg.color = color_msg

        # Define endpoint of the arrow, given by the start point, the direction and a length_scale parameter
        end = start + direction * length_scale
        # Construct ROS message for the start and end of the arrow
        msg.points = [
            ros_numpy.msgify(Point, start),
            ros_numpy.msgify(Point, end),
        ]
        msg.pose.orientation.w = 1
        msg.scale.x = thickness
        msg.scale.y = thickness * 2

        # Publish message
        self.arrow_pub.publish(msg)

    def plot_plane(self, centroid, normal, size: float = 1, res: float = 0.01):
        """
        This function plots a plane in Rviz.
        :param pub: ROS publisher
        :param centroid: centroid of the inliers
        :param normal: normal vector of the plane
        :param size: size of the plane
        :param res: resolution of the plane (there will be a point every "res" distance)
        :return: None
        """
        # Get three orthogonal vectors
        # Create a random vector from the normal vector
        r = normal + [1, 0, 0]
        # Normalize normal vector
        r = r / np.linalg.norm(r)
        # Normalize normal vector
        v0 = normal / np.linalg.norm(normal)
        # The other two orthogonal vectors
        v1 = np.cross(v0, r)
        v2 = np.cross(v0, v1)

        # Define the size and resolution of the plane
        t = np.arange(-size, size, res)

        # Construct 't' by 3 matrix
        v1s = t[:, None] * v1[None, :]
        v2s = t[:, None] * v2[None, :]

        # Construct a 't' by 't' by 3 matrix for the plane
        v1s_repeated = np.tile(v1s, [t.size, 1, 1])
        # Define the points that will construct the plane
        points = centroid + v1s_repeated + v2s[:, None]
        # Flatten the points
        points_flat = points.reshape([-1, 3])

        # Call the function to plot plane as a PC
        self.plot_pointcloud_rviz(self.plane_pub, points_flat[:, 0], points_flat[:, 1], points_flat[:, 2])


    def select_weed(self, selection):
        # Load point cloud and visualize it
        pcd = o3d.io.read_point_cloud("../pcs/weed-extraction/weed-09.pcd")
        # o3d.visualization.draw_geometries([pcd])

        # Get numpy array of xyz and rgb values of the point cloud
        pcd_points = np.asarray(pcd.points)
        pcd_colors = np.asarray(pcd.colors)

        # Filter the point cloud so that only the green points stay
        # Get the indices of the points with g parameter greater than x
        r_low, g_low, b_low = 0, 0.6, 0
        r_high, g_high, b_high = 1, 1, 1
        green_points_indices = np.where((pcd_colors[:, 0] > r_low) & (pcd_colors[:, 0] < r_high) &
                                        (pcd_colors[:, 1] > g_low) & (pcd_colors[:, 1] < g_high) &
                                        (pcd_colors[:, 2] > b_low) & (pcd_colors[:, 2] < b_high))

        h = len(green_points_indices[0])
        if len(green_points_indices[0]) == 1:
            r_low, g_low, b_low = 0, 0.3, 0
            r_high, g_high, b_high = 1, 1, 1
            green_points_indices = np.where((pcd_colors[:, 0] > r_low) & (pcd_colors[:, 0] < r_high) &
                                            (pcd_colors[:, 1] > g_low) & (pcd_colors[:, 1] < g_high) &
                                            (pcd_colors[:, 2] > b_low) & (pcd_colors[:, 2] < b_high))

        # Save xyzrgb info in green_points (type: numpy array)
        green_points_xyz = pcd_points[green_points_indices]
        green_points_rgb = pcd_colors[green_points_indices]

        # Create Open3D point cloud for green points
        green_pcd = o3d.geometry.PointCloud()
        # Save xyzrgb info in green_pcd (type: open3d.PointCloud)
        green_pcd.points = o3d.utility.Vector3dVector(green_points_xyz)
        green_pcd.colors = o3d.utility.Vector3dVector(green_points_rgb)
        # Apply radius outlier filter to green_pcd
        _, ind = green_pcd.remove_radius_outlier(nb_points=5, radius=0.02)
        # Just keep the inlier points in the point cloud
        green_pcd = green_pcd.select_by_index(ind)
        green_pcd_points = np.asarray(green_pcd.points)

        # Apply DBSCAN to green points
        labels = np.array(green_pcd.cluster_dbscan(eps=0.02, min_points=10))

        """
        # Color clusters and visualize them
        max_label = labels.max()
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        green_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([green_pcd])
        """

        # Get labels of the biggest cluster
        biggest_cluster_indices = np.where(labels[:] == mode(labels))
        # Just keep the points that correspond to the biggest cluster (weed)
        green_pcd_points = green_pcd_points[biggest_cluster_indices]
        # Get coordinates of the weed centroid
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
        [a, b, c, _] = plane_model
        # Just save and continue working with the inlier points defined by the plane segmentation function
        inlier_dirt_points = dirt_points_xyz[best_inliers]
        # Get centroid of dirt
        inlier_dirt_centroid = np.mean(inlier_dirt_points, axis=0)
        # The a, b, c coefficients of the plane equation are the components of the normal vector of that plane
        normal = np.asarray([a, b, c])

        phi = -atan(normal[1] / normal[2])
        theta = atan(normal[0] / normal[2])
        psi = atan(normal[1] / normal[0])

        Rx = np.asarray([[1, 0, 0],
                         [0, cos(phi), -sin(phi)],
                         [0, sin(phi), cos(phi)]])
        Ry = np.asarray([[cos(theta), 0, sin(theta)],
                         [0, 1, 0],
                         [-sin(theta), 0, cos(theta)]])
        Rz = np.asarray([[cos(psi), -sin(psi), 0],
                         [sin(psi), cos(psi), 0],
                         [0, 0, 1]])

        frame2vector_rot = Rx @ Ry @ Rz

        """
        # Apply PCA and get just one principal component
        pca = PCA(n_components=3)
        # Fit the PCA to the inlier points
        pca.fit(inlier_dirt_points)
        # The third component (vector) is the normal of the plane of the dirt we are looking for
        third_comp = pca.components_[2]
        """

        tfw = TF2Wrapper()

        while True:
            # Construct transformation matrix from camera to tool of end effector
            camera2tool = np.zeros([4, 4])
            camera2tool[:3, :3] = frame2vector_rot
            camera2tool[:3, 3] = weed_centroid
            camera2tool[3, 3] = 1

            # Define transformation matrix from tool to end effector
            tool2ee = tfw.get_transform(parent="left_tool", child="end_effector_left")
            # Define transformation matrix from camera to end effector
            camera2ee = camera2tool @ tool2ee
            # Display gripper
            tfw.send_transform_matrix(camera2ee, parent=camera_frame, child='end_effector_left')

            # Call plot_pointcloud_rviz function to visualize PCs in Rviz
            # Visualize all the point cloud as "source"
            self.plot_pointcloud_rviz(self.src_pub,
                                 pcd_points[:, 0],
                                 pcd_points[:, 1],
                                 pcd_points[:, 2])
            # Visualize filtered green points as "inliers"
            self.plot_pointcloud_rviz(self.inliers_pub,
                                 green_pcd_points[:, 0],
                                 green_pcd_points[:, 1],
                                 green_pcd_points[:, 2])
            # Call rviz_arrow function to see normal of the plane
            self.rviz_arrow(inlier_dirt_centroid, normal, name='normal', thickness=0.008, length_scale=0.15,
                       color='w')
            # Call plot_plane function to visualize plane in Rviz
            self.plot_plane(inlier_dirt_centroid, normal, size=0.1, res=0.001)
            rospy.sleep(1)


def main():
    weed_extractor = WeedExtractor("zed2i_left_camera_frame")
    rospy.spin()

if __name__ == '__main__':
    main()
