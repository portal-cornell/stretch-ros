#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger
import time
import hello_helpers.hello_misc as hm
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
# place
from hal_skills_aruco import HalSkillsPlace

# ppo fixed
from hal_skills_ppo_eval import HalSkillsNode
from broadcast_aruco_hal import BroadcastArucoHAL
import sys

import argparse

class Move(hm.HelloNode):
    """
    A class that sends Twist messages to move the Stretch robot forward.
    """
    def __init__(self):
        """
        Function that initializes the subscriber.
        :param self: The self reference.
        """
        hm.HelloNode.__init__(self)
        self.node_name = rospy.get_name()
        rospy.loginfo("{0} started".format(self.node_name))
        self.pub = rospy.Publisher('/stretch/cmd_vel', Twist, queue_size=1) #/stretch_diff_drive_controller/cmd_vel for gazebo
        s = rospy.ServiceProxy("/switch_to_navigation_mode", Trigger)
        resp = s()
        print(resp)

        self.trajectory_client = actionlib.SimpleActionClient(
            "/stretch_controller/follow_joint_trajectory", FollowJointTrajectoryAction
        )

        rospy.init_node("hal_move")

        self.hal_skills_pick = HalSkillsNode()

        # rgb_topic = "/head_camera/color/image_raw"
        # cam_info_topic = "/head_camera/aligned_depth_to_color/camera_info"
        # pc_topic = "/head_camera/aligned_depth_to_color/image_raw"
        # cam_frame = "camera_depth_optical_frame"
        # self.aruco_broadcaster = BroadcastArucoHAL(
        #     rgb_topic=rgb_topic,
        #     cam_info_topic=cam_info_topic,
        #     pc_topic=pc_topic,
        #     cam_frame=cam_frame,
        # )

    def move_x(self, speed):
        """
        Function that publishes Twist messages
        :param self: The self reference.

        :publishes command: Twist message.
        """
        command = Twist()
        command.linear.x = speed
        command.linear.y = 0.0
        command.linear.z = 0.0
        command.angular.x = 0.0
        command.angular.y = 0.0
        command.angular.z = 0.0
        self.pub.publish(command)
    
    def close_grip(self):
        point = JointTrajectoryPoint()
        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.goal_time_tolerance = rospy.Time(1.0)
        trajectory_goal.trajectory.joint_names = ["joint_gripper_finger_left"]
        point.positions = [-0.2]
        trajectory_goal.trajectory.points = [point]
        self.trajectory_client.send_goal(trajectory_goal)
        grip_change_time = 2
        rospy.sleep(grip_change_time)

    def initial_arm_move(self):
        rospy.loginfo("Set arm")
        pose = {
            "wrist_extension": 0.05,
            "joint_lift": 0.3,  # for cabinet -0.175
            "joint_wrist_pitch": 0.0,
            "joint_wrist_yaw": 2.8,
        }
        self.move_to_pose(pose)
        self.close_grip()
        return True
    
    def initial_arm_place(self):
        rospy.loginfo("Set arm")
        pose = {
            "wrist_extension": 0.05,
            "joint_lift": 0.3,  # for cabinet -0.175
            "joint_wrist_pitch": 0.0,
            "joint_wrist_yaw": 0.0,
        }
        self.move_to_pose(pose)
        self.close_grip()
        return True
    
    def retract_arm_primitive(self):
        rospy.loginfo("Set arm")
        pose = {
            "wrist_extension": 0.05
        }
        self.move_to_pose(pose)
        return True
    
    def timed_move(self, duration, speed):
        start = time.time()
        t = 0
        while t < duration:
            current = time.time()
            t = current - start
            self.move_x(speed)
        self.move_x(0)
    
    def main(self, place_loc):
        """
        Robot starts at table and looking at the objects. Hal_prediction_node and depth_bs needs to be running for pick
        place_loc: List of strings of form ["x", "y", "z"] for place location in the fridge in aruco frame
        """

        self.hal_skills_pick.main()

        self.retract_arm_primitive()
        self.initial_arm_move()

        s = rospy.ServiceProxy("/switch_to_navigation_mode", Trigger)
        resp = s()
        print(resp)

        self.timed_move(7, -0.1)

        self.initial_arm_place()

        # self.aruco_broadcaster.predict()
        # input()
        self.hal_skills_place = HalSkillsPlace(place_loc)
        self.hal_skills_place.main()

        self.retract_arm_primitive()
        self.initial_arm_move()

        s = rospy.ServiceProxy("/switch_to_navigation_mode", Trigger)
        resp = s()
        print(resp)

        self.timed_move(7, 0.1)

        return
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="main_slighting")

    parser.add_argument("-i", type=int, required=True)

    args = parser.parse_args()


    move_obj = Move()
    # place_loc = ["0.1", "-0.1557", "-0.07"]  # right next to the bottom left Aruco tag
    # place_loc = ["0.1", "0.1557", "-0.07"]   # right next to the top left Aruco tag
    # place_loc = ["0.1", "0.1557", "-0.07"]   # right next to the top left Aruco tag

    place_locations = [
        ["0.35", "-0.1257", "-0.07"], # right next to the bottom left Aruco tag,
        ["0.1", "0.1557", "-0.03"],   # right next to the top left Aruco tag,
        ["0.4", "0.1557", "-0.03"],   # right next to the top right Aruco tag,
        ["0.35", "-0.1557", "-0.03"],   # right next to the bottom right Aruco tag,
        ["0.3", "-0.1557", "-0.03"],   # middle of middle shelf and deeper
    ]

    move_obj.main(place_locations[args.i])