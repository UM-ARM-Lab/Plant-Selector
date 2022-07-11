import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2
from nav_msgs.msg import Odometry


class Kalman:
    def __init__(self):
        # Setup ROS stuff
        self.pc_topic = str(rospy.get_param("pc_source"))
        self.odom_topic = str(rospy.get_param("odom_source"))
        rospy.Subscriber(self.pc_topic, PointCloud2, self.pointcloud_callback)
        rospy.Subscriber(self.odom_topic, Odometry, self.pointcloud_callback)

    # Unsure if this will be necessary, but will keep for now, may want a most recent pointcloud or something like that
    def pointcloud_callback(self, pc):
        return None

    def odometry_callback(self, odom):
        return None

    # This function actually calculates the new location for the gripper
    def filter_tracking(self):
        return None


def main():
    rospy.init_node("kalman_filter")
    kalman = Kalman()
    rospy.spin()


if __name__ == '__main__':
    main()


