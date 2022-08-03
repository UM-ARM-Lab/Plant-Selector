#!/usr/bin/env python
# Import useful libraries and functions
import numpy as np
import plant_modeling as pm
import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from std_msgs.msg import Bool
from tf.transformations import euler_from_matrix

# Robot stuff
from arm_robots.hdt_michigan import Val

# THIS SCRIPT IS MEANT FOR ONLY VAL!!!
class RepetitivePlantExtractor:
    def __init__(self):
        rospy.Subscriber("/plant_selector/mode", String, self.mode_change)
        rospy.Subscriber("/plant_selector/verification", Bool, self.move_robot)
        rospy.Subscriber("/rviz_selected_points", PointCloud2, self.plant_extraction)
        rospy.Subscriber("/plant_selector/hide_gripper", Bool, self.hide_gripper_callback)
        self.ask_for_verif_pub = rospy.Publisher("/plant_selector/ask_for_verification", Bool, queue_size=10)

        # Set the default mode to branch
        self.mode = "Weed"

        self.camera_frame_id = str(rospy.get_param("camera_frame_id"))
        self.tfw = TF2Wrapper()

        self.robot = Val(raise_on_failure=True)
        self.robot.connect()
        self.auto_move = bool(rospy.get_param("return_to_default_automatically"))
        print(self.auto_move)
        self.default_pose = str(rospy.get_param("default_pose"))
        self.robot_to_default_pose()

        self.goal = None
        self.plan_exec_res = None

        rospy.sleep(1)
        self.hide_red_gripper()

    def mode_change(self, new_mode):
        self.mode = new_mode.data
        rospy.loginfo("New mode: " + self.mode)
        self.robot_to_default_pose()

    def move_robot(self, is_verified):
        if is_verified.data:
            self.robot_execute()

    def plant_extraction(self, pc):
        if self.mode == "Branch":
            self.select_branch(pc)
        elif self.mode == "Weed":
            self.select_weed(pc)

    def select_branch(self, selection):
        camera2tool = pm.predict_branch_pose(selection)
        if camera2tool is None:
            rospy.loginfo("Unable to make a branch prediction")
            return
        self.visualize_red_gripper(camera2tool)
        self.robot_plan(camera2tool)

    def select_weed(self, selection):
        camera2tool = pm.predict_weed_pose(selection)
        if camera2tool is None:
            rospy.loginfo("Unable to make a weed prediction")
            return
        self.visualize_red_gripper(camera2tool)
        self.robot_plan(camera2tool)

    def robot_plan(self, camera2tool):
        if self.robot is None:
            return

        # figure out gripper pose in world frame
        world2cam = self.tfw.get_transform(parent='world', child=self.camera_frame_id)
        world2tool = world2cam @ camera2tool

        # Plan to the pose
        x_rot, y_rot, z_rot = euler_from_matrix(world2tool[:3, :3])

        # Plan to a pose that is around 20 centimeters higher than directed this acts as a "home" pose between multiple attempts
        # at grasping
        self.goal = [world2tool[0, 3], world2tool[1, 3], world2tool[2, 3] + 0.2, x_rot, y_rot, z_rot]
        self.robot.set_execute(False)

        # Find a plan and execute it
        was_success = True
        try:
            self.plan_exec_res = self.robot.plan_to_pose(self.robot.left_arm_group, self.robot.left_tool_name, self.goal)
        except:
            rospy.loginfo("Can't find a valid plan.")
            was_success = False

        if was_success:
            # Send a message to rviz panel which prompts the user to verify the execution plan
            msg = Bool()
            msg.data = True
            self.ask_for_verif_pub.publish(msg)
        else:
            msg = Bool()
            msg.data = False
            self.ask_for_verif_pub.publish(msg)

    def robot_execute(self):
        # Attempt to go to the goal
        self.robot.set_execute(True)
        self.robot.follow_arms_joint_trajectory(self.plan_exec_res.planning_result.plan.joint_trajectory)

        return_cords = self.goal[:3]
        weed_cords = self.goal[:3]
        weed_cords[2] -= 0.17
        self.robot.store_current_tool_orientations([self.robot.left_tool_name])
        for _ in range(5):
            # Attempt to go for it
            self.robot.follow_jacobian_to_position(self.robot.left_arm_group, [self.robot.left_tool_name], [[weed_cords]], vel_scaling=1.0)

            # Grasping
            rospy.sleep(1)
            self.robot.close_left_gripper()

            # Return to "home position"
            self.robot.follow_jacobian_to_position(self.robot.left_arm_group, [self.robot.left_tool_name], [[return_cords]], vel_scaling=1.0)

            # open
            rospy.sleep(1)
            self.robot.open_left_gripper()

        rospy.sleep(5)
        self.robot_to_default_pose()

    def robot_to_default_pose(self):
        self.hide_red_gripper()
        if self.auto_move == False:
            return

        # Go back to default!
        self.robot.set_execute(True)
        self.robot.plan_to_joint_config('both_arms', self.default_pose)

        rospy.sleep(1)
        self.robot.open_left_gripper()

    def visualize_red_gripper(self, camera2tool):
        # Get transformation matrix between tool and end effector
        tool2ee = self.tfw.get_transform(parent="red_left_tool", child="red_end_effector_left")
        # Chain effect: get transformation matrix from camera to end effector
        camera2ee = camera2tool @ tool2ee
        self.tfw.send_transform_matrix(camera2ee, parent=self.camera_frame_id, child='red_end_effector_left')

    def hide_gripper_callback(self, msg):
        # This function is used as callback of mainpanel when someone presses "Hide Red Gripper"
        # This is probably not the best way to do this but, I can't think of a better way
        self.hide_red_gripper()

    def hide_red_gripper(self):
        # Because you can't directly hide a urdf, just send the urdf very far away :D
        end_effector_to_void = np.eye(4)
        end_effector_to_void[:3, 3] = 1000
        self.tfw.send_transform_matrix(end_effector_to_void, 'zed2i_base_link', 'red_end_effector_left')


def main():
    rospy.init_node("repetitive_plant_extraction")
    plant_extractor = RepetitivePlantExtractor()
    rospy.spin()


if __name__ == '__main__':
    main()
