#! /usr/bin/env python
import colorama
import numpy as np

from arc_utilities import ros_init
from arm_robots.hdt_michigan import Val
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler
import rospy
from arc_utilities.tf2wrapper import TF2Wrapper

ask_before_moving = True


def myinput(msg):
    global ask_before_moving
    if ask_before_moving:
        input(msg)


@ros_init.with_ros("basic_motion")
def main():
    np.set_printoptions(suppress=True, precision=0, linewidth=200)
    colorama.init(autoreset=True)

    val = Val(raise_on_failure=True)
    val.connect()

    # val.open_left_gripper()
    val.close_left_gripper()
    # val.open_right_gripper()
    val.close_right_gripper()

    print("press enter if prompted")

    # Plan to joint config
    rospy.sleep(1)
    val.plan_to_joint_config('both_arms', 'bent')

    # Plan to pose
    rospy.sleep(1)
    camera2tool = np.eye(4)
    camera2tool[:3, 3] = [0.85, -0.25, -0.1]

    tfw = TF2Wrapper()
    val2cam = tfw.get_transform(parent='world', child='zed2i_left_camera_frame')
    result = val2cam @ camera2tool

    # tfw.send_transform_matrix(camera2tool, parent='zed2i_base_link', child='end_effector_left')

    goal = [result[0, 3], result[1, 3], result[2, 3], 1.5707, 0, 3.14]
    # Need to figure out how to plan from the torso
    print(val.right_arm_group)
    # val.plan_to_pose('right_side', val.right_tool_name, goal)

    val.disconnect()


if __name__ == "__main__":
    main()
