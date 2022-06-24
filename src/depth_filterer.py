#!/usr/bin/env python
import rospy
import sensor_msgs.point_cloud2
from sensor_msgs.msg import PointCloud2
import numpy as np
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointField
from std_msgs.msg import Header
from sklearn.cluster import DBSCAN


def points_to_pc2_msg(points, frame_id):
    """
    Args:
        points: an Nx4 array
        frame_id: the frame to publish in

    Returns: a PointCloud2 message ready to be published to rviz

    """
    header = Header(frame_id=frame_id)
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1),
              PointField('rgb', 12, PointField.FLOAT32, 1)
              ]
    pc2_msg = point_cloud2.create_cloud(header, fields, points)
    return pc2_msg


class Filterer:
    def __init__(self):
        rospy.init_node('filterer', anonymous=True)
        rospy.Subscriber("/rviz_selected_points", PointCloud2, self.cluster_filter)
        self.filter_pub = rospy.Publisher("/plant_selector/filtered", PointCloud2, queue_size=10)

    def cluster_filter(self, pc):
        points = np.array(list(sensor_msgs.point_cloud2.read_points(pc)))
        clustering = DBSCAN(eps=0.015, min_samples=30).fit(points)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Find the cluster closest to the user
        closest_cluster = 0
        closest_cluster_dist = np.inf
        # TODO: Figure out how to get the actual center of camera
        camera_location = np.array((0, 0, 0))
        for x in range(n_clusters):
            sel_indicies = np.argwhere(labels == x).squeeze(1)
            this_cluster = points[sel_indicies]
            cluster_center = np.sum(this_cluster[:, :3], axis=0) / this_cluster.shape[0]
            dist = np.linalg.norm(cluster_center - camera_location)
            if dist < closest_cluster_dist:
                closest_cluster_dist = dist
                closest_cluster = x

        sel_indicies = np.argwhere(labels == closest_cluster).squeeze(1)
        best_selection = points[sel_indicies]
        msg = points_to_pc2_msg(best_selection, "zed2i_left_camera_frame")
        self.filter_pub.publish(msg)


def main():
    filter_one = Filterer()
    rospy.spin()


if __name__ == '__main__':
    main()