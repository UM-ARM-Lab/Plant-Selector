#!/usr/bin/env python
import rospy
import sensor_msgs.point_cloud2
from sensor_msgs.msg import PointCloud2
import numpy as np
from sklearn.cluster import DBSCAN
import helpers


class Filterer:
    def __init__(self):
        rospy.init_node('filterer', anonymous=True)
        rospy.Subscriber("/rviz_selected_points", PointCloud2, self.cluster_filter)
        self.filter_pub = rospy.Publisher("/plant_selector/filtered", PointCloud2, queue_size=10)
        self.frame_id = "zed2i_left_camera_frame"

    def cluster_filter(self, pc):
        points = np.array(list(sensor_msgs.point_cloud2.read_points(pc)))

        # Perform a color filter
        points = helpers.green_color_filter(points)

        # TODO: The eps value here might want to somehow change dynamically where points further away can have clusters more spread out?
        # The eps value really depends on how good the video quality is and how far away points are from each other
        clustering = DBSCAN(eps=0.015, min_samples=30).fit(points)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Find the cluster closest to the user
        closest_cluster = 0
        closest_cluster_dist = np.inf
        # TODO: Figure out how to get the actual center of camera so it isnt hardcoded
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
        helpers.publish_pc_with_color(self.filter_pub, best_selection, self.frame_id)


def main():
    filter_one = Filterer()
    rospy.spin()


if __name__ == '__main__':
    main()