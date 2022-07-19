import rospy
import numpy as np

import sensor_msgs.point_cloud2
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2
from geometry_msgs.msg import Point

from enum import Enum


class SelectionType(Enum):
    NONE = 0
    WEED_PC = 1
    WEED_CENTER = 2


class WeedDataCollector:
    def __init__(self):
        # Setup ROS stuff
        rospy.Subscriber("/rviz_selected_points", PointCloud2, self.selection_callback)

        # Initialize parent directory for getting data
        self.parent_directory = "/home/christianforeman/catkin_ws/src/plant_selector/weed_pcs/"

        # Need to find a way to initialize this number
        self.number = 1
        self.selection_type = SelectionType.WEED_PC

    # Unsure if this will be necessary, but will keep for now, may want a most recent pointcloud or something like that
    def selection_callback(self, pc):
        points = np.array(list(sensor_msgs.point_cloud2.read_points(pc)))
        filename = self.parent_directory

        if self.selection_type == SelectionType.WEED_PC:
            # Create filename and save
            filename += "weed_" + str(self.number) + "_pc.npy"
            np.save(filename, points)

            # Next selection should be for a weed center
            self.selection_type = SelectionType.WEED_CENTER
        else:
            # Create filename and save
            filename += "weed_" + str(self.number) + "_center.npy"
            np.save(filename, points)

            # Next Selection should be a brand new weed pc
            self.selection_type = SelectionType.WEED_PC
            self.number += 1


def main():
    rospy.init_node("weed_data_collector")
    kalman = WeedDataCollector()
    rospy.spin()


if __name__ == '__main__':
    main()


