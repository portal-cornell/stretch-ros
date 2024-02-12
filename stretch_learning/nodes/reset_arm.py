#!/usr/bin/env python3

import time
import rospy
import math
import argparse
import message_filters
import actionlib
from pathlib import Path
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryGoal, FollowJointTrajectoryAction
from custom_msg_python.msg import Keypressed
from stretch_learning.msg import PickPrompt
import hello_helpers.hello_misc as hm
import matplotlib.pyplot as plt
import pickle as pl
import keyboard
import matplotlib.animation as animation

from PIL import Image as Im
import torch
import torch.nn as nns
import numpy as np


from std_srvs.srv import Trigger
from geometry_msgs.msg import PointStamped

from copy import deepcopy
import tf2_py
import tf2_ros
from geometry_msgs.msg import TransformStamped, PoseWithCovarianceStamped
from tf2_msgs.msg import TFMessage
from collections import deque


class HalSkillsNode(hm.HelloNode):
    def __init__(self):
        hm.HelloNode.__init__(self)
        self.debug_mode = False
        self.trajectory_client = actionlib.SimpleActionClient(
            "/stretch_controller/follow_joint_trajectory", FollowJointTrajectoryAction
        )

        self.mode = "position"  #'manipulation' #'navigation'
        # self.mode = "navigation"                              NOTE: changed frm  navigation to position, since navigation does not support base movement

        self.joint_states = None
        self.joint_states_data = None

        self.joint_states_subscriber = rospy.Subscriber(
            "/stretch/joint_states", JointState, self.joint_states_callback
        )

        self.publisher = rospy.Publisher(
            "/amcl_pose", data_class=PoseWithCovarianceStamped, queue_size=1
        )

        self.init_node()
        self.rate = 10.0
        self.rosrate = rospy.Rate(self.rate)

    def init_node(self):
        rospy.init_node("hal_skills_node")
        self.node_name = rospy.get_name()
        rospy.loginfo("{0} started".format(self.node_name))

    def joint_states_callback(self, msg):
        self.joint_states = msg
        js_msg = list(zip(msg.name, msg.position, msg.velocity, msg.effort))
        js_msg = sorted(js_msg, key=lambda x: x[0])
        js_arr = []
        for idx, (name, pos, vel, eff) in enumerate(js_msg):
            js_arr.extend([pos, vel, eff])
        joint_states_data_raw = torch.from_numpy(np.array(js_arr, dtype=np.float32))
        if len(joint_states_data_raw.size()) <= 1:
            self.joint_states_data = joint_states_data_raw.unsqueeze(0)
        else:
            self.joint_states_data = joint_states_data_raw

    # -----------------pick_pantry() initial configs-----------------#
    def move_arm_pick_pantry(self):
        rospy.loginfo("Set arm")
        self.pick_starting_height = 0.998
        self.joint_lift_index = self.joint_states.name.index("joint_lift")
        pose = {
            "wrist_extension": 0.05,
            "joint_lift": self.pick_starting_height - 0.45,  # for cabinet -0.175
            "joint_wrist_pitch": 0.0,
            "joint_wrist_yaw": 0.0,
            "joint_wrist_roll": 0.0,
            # "joint_wrist_yaw": -0.09,
        }
        self.base_gripper_yaw = -0.09
        self.gripper_len = 0.22
        self.move_to_pose(pose)
        return True

    def move_head_pick_pantry(self):
        tilt = -0.4358
        pan = -1.751
        rospy.loginfo("Set head pan")
        pose = {"joint_head_pan": pan, "joint_head_tilt": tilt}
        self.move_to_pose(pose)
        return True

    def open_grip(self):
        point = JointTrajectoryPoint()
        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.goal_time_tolerance = rospy.Time(1.0)
        trajectory_goal.trajectory.joint_names = ["joint_gripper_finger_left"]
        point.positions = [0.22]
        trajectory_goal.trajectory.points = [point]
        self.trajectory_client.send_goal(trajectory_goal)
        grip_change_time = 2
        rospy.sleep(grip_change_time)

    def pick_pantry_initial_config(self, rate):
        done_head_pan = False
        # import pdb; pdb.set_trace()

        while not done_head_pan:
            if self.joint_states:
                done_head_pan = self.move_head_pick_pantry()
            rate.sleep()
        done_initial_config = False
        while not done_initial_config:
            if self.joint_states:
                done_initial_config = self.move_arm_pick_pantry()
            rate.sleep()
        self.open_grip()

    def reset_pick_pantry(self):
        print("start of reseting pick pantry")
        self.pick_pantry_initial_config(self.rosrate)

    def reset_pick_bowl(self):
        pose_sequence = [
            {
                "wrist_extension": 0.05,
                "joint_gripper_finger_left": 0.22,
                "joint_lift": 1.09,  # for cabinet -0.175
                "joint_wrist_pitch": -0.56,
                "joint_wrist_yaw": 0.0,
                "joint_wrist_roll": 1.59,
            },
            {"wait": 1.0},
            {"wrist_extension": 0.24},
            {"wait": 1.0},
            {
                "joint_gripper_finger_left": 0.10715,
                # "joint_gripper_finger_right": 0.10715,
            },
            {"wait": 1.0},
            {"joint_wrist_pitch": -0.807},
            {"wait": 1.0},
            {
                "joint_gripper_finger_left": -0.105,
                # "joint_gripper_finger_right": -0.105,
            },
            {"wait": 1.0},
        ]

        for pose in pose_sequence:
            if "wait" in pose:
                rospy.sleep(pose["wait"])
            else:
                self.move_to_pose(pose)

        return True

    def main(self):
        print("start of main")
        rate = rospy.Rate(self.rate)

        print("switching to position mode")
        s = rospy.ServiceProxy("/switch_to_position_mode", Trigger)
        resp = s()
        print(resp)

        print("start of reset")

        self.pick_pantry_initial_config(rate)

    def publishing(self):
        message = PoseWithCovarianceStamped()
        message.pose.pose.position.x = 2.815
        message.pose.pose.position.y = 1.05
        message.pose.pose.orientation.z = 0.707
        message.pose.pose.orientation.w = 0.707

        message._md5sum = "953b798c0f514ff060a53a3498ce6246"
        print("here")
        self.publisher.publish(message)


if __name__ == "__main__":
    node = HalSkillsNode()
    node.main()
    # node.publishing()
    # node.reset_pick_bowl()
    # node.reset_pick_pantry()
