import rospy
import numpy as np

import sensor_msgs.point_cloud2
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from arc_utilities.tf2wrapper import TF2Wrapper


class Kalman:
    def __init__(self):
        # Setup ROS stuff
        self.frame_id = str(rospy.get_param("frame_id"))
        self.pc_topic = str(rospy.get_param("pc_source"))
        self.odom_topic = str(rospy.get_param("odom_source"))
        rospy.Subscriber(self.pc_topic, PointCloud2, self.pointcloud_callback)
        rospy.Subscriber(self.odom_topic, Odometry, self.odometry_callback)

        self.points = None
        self.current_odom = dict({'position': Point(0, 0, 0), 'angular': Point(0, 0, 0)})
        self.diff = Point(0, 0, 0)
        self.tfw = TF2Wrapper()
        self.transform = np.eye(4)

    # Unsure if this will be necessary, but will keep for now, may want a most recent pointcloud or something like that
    def pointcloud_callback(self, pc):
        self.points = np.array(list(sensor_msgs.point_cloud2.read_points(pc)))
        return None

    def odometry_callback(self, odom):
        # Get the odometry data
        new_odom = odom.pose.pose

        # If the odometry is no different, don't run the kalman filter. TODO: CHECK IF THIS IS WHAT I WANT TO DO
        if not self.is_diff_odom(new_odom):
            return
        print("Moving")
        self.filter_tracking()

    # This function actually calculates the new location for the gripper
    def filter_tracking(self):
        # Prediction = A(t) @ x(t-1)  +  B(t) @ u(t)
        #              4 x 4 @ 4 x 1  + 4 x 4 @ 4 x 1
        #                    Last Pos        Cur Action
        #
        #              Calculate Sigma Bar t
        # Perhaps the vals with bar on top is based purely on dynamics and then with no bar is when estimated noise
        # is added in?
        # R(t) is a 4x4 matrix where the diagonal is full of values where you have to tune. This represents the noise

        # Correction:
        #              Calculate the Kalman Gain
        #              Correct the prediction
        #              Update Sigma(t)
        #
        # Need to figure out what C(t) matrix is. "Describes how to map the state xt to an observed zt"

        # Note: Apparently you are supposed to kind of do a random initialization for the values for noise.
        # Should definitely look into how people initialize them
        # After this implementation, I will need to account for rotation. Should be easier once I figure out how the
        # regular kalman filter works and then move to the extended kalman filter

        # Just a Test for how to send transforms to the gripper
        self.transform[0, 3] = self.current_odom['position'].x
        self.transform[1, 3] = self.current_odom['position'].y
        self.transform[2, 3] = self.current_odom['position'].z
        self.tfw.send_transform_matrix(self.transform, parent=self.frame_id, child='end_effector_left')

    def is_diff_odom(self, new_odom):
        x_diff = new_odom.position.x - self.current_odom['position'].x
        y_diff = new_odom.position.y - self.current_odom['position'].y
        z_diff = new_odom.position.z - self.current_odom['position'].z
        self.diff = Point(x_diff, y_diff, z_diff)

        if abs(x_diff) < 0.03 and abs(y_diff) < 0.03 and abs(z_diff) < 0.03:
            return False
        # TODO: Add Angular components here as well
        self.current_odom['position'] = new_odom.position
        return True


def main():
    rospy.init_node("kalman_filter")
    kalman = Kalman()
    rospy.spin()


if __name__ == '__main__':
    main()


