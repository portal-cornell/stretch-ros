#!/usr/bin/env python3

import time
import cv2
import csv
import rospy
import tf
import math
import argparse
import message_filters
import actionlib
from pathlib import Path
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryGoal, FollowJointTrajectoryAction
from custom_msg_python.msg import Keypressed
import hello_helpers.hello_misc as hm
import matplotlib.pyplot as plt
import pickle as pl
import keyboard
import matplotlib.animation as animation
from moviepy.editor import VideoFileClip
from PIL import Image as Im
import torch
import torch.nn as nns
import numpy as np
from torchvision import transforms
from cv_bridge import CvBridge, CvBridgeError
import ppo_utils
import open_clip
import copy
import sys
import tf2_ros
from std_srvs.srv import Trigger
from geometry_msgs.msg import PointStamped

joint_labels_for_img = [
    "gripper_aperture",
    "joint_arm_l0",
    "joint_arm_l1",
    "joint_arm_l2",
    "joint_arm_l3",
    "joint_gripper_finger_left",
    "joint_gripper_finger_right",
    "joint_head_pan",
    "joint_head_tilt",
    "joint_lift",
    "joint_wrist_pitch",
    "joint_wrist_roll",
    "joint_wrist_yaw",
    "wrist_extension",
]

device = "cuda" if torch.cuda.is_available() else "cpu"


class HalSkills(hm.HelloNode):
    def __init__(self, goal_pos):
        hm.HelloNode.__init__(self)
        self.debug_mode = False
        self.rate = 10.0
        self.trajectory_client = actionlib.SimpleActionClient(
            "/stretch_controller/follow_joint_trajectory", FollowJointTrajectoryAction
        )

        self.step_size = "medium"
        self.rad_per_deg = math.pi / 180.0
        self.small_deg = 3.0
        self.small_rad = self.rad_per_deg * self.small_deg
        self.small_translate = 0.005  # 0.02
        self.medium_deg = 6.0
        self.medium_rad = self.rad_per_deg * self.medium_deg * (3 / 5)
        self.medium_translate = 0.04 * (3 / 5)
        self.mode = "position"  #'manipulation' #'navigation'
        # self.mode = "navigation"                              NOTE: changed frm  navigation to position, since navigation does not support base movement

        self.home_odometry = None
        self.odometry = None
        self.joint_states = None
        self.wrist_image = None
        self.head_image = None
        self.joint_states_data = None
        self.cv_bridge = CvBridge()

        self.goal_pos = list(map(float, goal_pos))
        self.goal_tensor = torch.Tensor(self.goal_pos).to(device)

        # self.goal_tensor[3:5] = self.odom_to_js(self.goal_tensor[3:5])

        self.init_node()  # comment out when using move

    def init_node(self):
        rospy.init_node("hal_skills_node")
        self.node_name = rospy.get_name()
        rospy.loginfo("{0} started".format(self.node_name))

    def quaternion_to_angle(self, quaternion):
        z, w = quaternion[2], quaternion[3]
        t0 = 2.0 * w * z
        t1 = 1.0 - 2.0 * z * z
        return math.atan2(t0, t1)

    def odom_callback(self, msg):  # NOTE: callback to set odometry value
        pose = msg.pose.pose
        raw_odometry = [
            pose.position.x,
            pose.position.y,
            self.quaternion_to_angle(
                [
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ]
            ),
        ]
        if self.home_odometry is None:
            self.home_odometry = torch.from_numpy(np.array(raw_odometry))

        self.odometry = torch.from_numpy(np.array(raw_odometry)) - self.home_odometry
        self.odometry = self.odom_to_js(self.odometry)

    def joint_states_callback(self, msg):
        self.joint_states = msg
        js_msg = list(zip(msg.name, msg.position, msg.velocity, msg.effort))
        js_msg = sorted(js_msg, key=lambda x: x[0])
        js_arr = []
        for idx, (name, pos, vel, eff) in enumerate(js_msg):
            js_arr.extend([pos, vel, eff])
        self.joint_states_data = torch.from_numpy(np.array(js_arr, dtype=np.float32))
        if len(self.joint_states_data.size()) <= 1:
            self.joint_states_data = self.joint_states_data.unsqueeze(0)

    def odom_to_js(self, odom_data):
        """
        Takes in odom data [x, y, theta] from /odom topic and flips the x and y
        """
        odom_data[0] = odom_data[0] * -1
        odom_data[1] = odom_data[1] * -1
        return odom_data

    def rotate_and_translate(self, point, angle, tx, ty):
        x, y = point

        # Rotate the point
        x_rot = x * math.cos(angle) - y * math.sin(angle)
        y_rot = x * math.sin(angle) + y * math.sin(angle)

        # Translate in the rotated coordinates
        x_trans = x_rot + tx
        y_trans = y_rot + ty

        return (x_trans, y_trans)

    def new_end_eff_to_xyz(self, joint_state):
        extension = joint_state[0]
        yaw = joint_state[1]
        lift = joint_state[2]

        gripper_len = 0.23
        base_gripper_yaw = -0.09

        yaw_delta = yaw - base_gripper_yaw
        gripper_y_offset = gripper_len * np.cos(yaw_delta)
        gripper_x_offset = gripper_len * -np.sin(yaw_delta)

        x = gripper_x_offset + 0.064
        y = gripper_y_offset + extension + 0.1822
        z = lift + 0.111

        return np.array([x.item(), y.item(), z])

    def rotate_odom(self, coord, angle, pivot):
        x1 = (
            math.cos(angle) * (coord[0] - pivot[0])
            - math.sin(angle) * (coord[1] - pivot[1])
            + pivot[0]
        )
        y1 = (
            math.sin(angle) * (coord[0] - pivot[0])
            + math.cos(angle) * (coord[1] - pivot[1])
            + pivot[1]
        )

        return (x1, y1)

    @torch.no_grad()
    def main(self, reset=True):
        print("start of main")

        print("switching to position mode")
        s = rospy.ServiceProxy("/switch_to_position_mode", Trigger)
        resp = s()
        print(resp)

        self.odom_subscriber = rospy.Subscriber(
            "/odom", Odometry, self.odom_callback
        )  # NOTE: subscribe to odom

        self.joint_states_subscriber = rospy.Subscriber(
            "/stretch/joint_states", JointState, self.joint_states_callback
        )
        self.curr_ee_ps_publisher = rospy.Publisher(
            "hal_skills_final/curr_ee_ps", PointStamped, queue_size=10
        )
        self.curr_ee_ps_ref_publisher = rospy.Publisher(
            "hal_skills_final/curr_ee_ps_ref", PointStamped, queue_size=10
        )
        self.goal_point_publisher = rospy.Publisher(
            "track_xyz/goal_point", PointStamped, queue_size=10
        )
        self.shifted_ps_publisher = rospy.Publisher(
            "hal_skills_final/shifted_ps", PointStamped, queue_size=10
        )
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)

        rate = rospy.Rate(self.rate)

        if reset:
            print("start of reset")

        print("start of loop")

        # node = hm.HelloNode.quick_create('transformations')
        listener = tf.TransformListener()
        from_frame_rel = "base_link"
        to_frame_rel = "link_grasp_center"

        rospy.sleep(1.0)
        rate = rospy.Rate(1)

        # node.move_to_pose({'joint_arm': 0.5}, return_before_done=True)

        while not rospy.is_shutdown():
            if self.joint_states is not None and self.odometry is not None:
                print("-" * 80)

                joint_pos = self.joint_states.position
                lift_idx, wrist_idx, yaw_idx = (
                    self.joint_states.name.index("joint_lift"),
                    self.joint_states.name.index("wrist_extension"),
                    self.joint_states.name.index("joint_wrist_yaw"),
                )
                end_eff_tensor = torch.Tensor(
                    self.new_end_eff_to_xyz(
                        [
                            joint_pos[wrist_idx],
                            joint_pos[yaw_idx],
                            joint_pos[lift_idx],
                        ]
                    )
                )
                print(end_eff_tensor)

                goal_point_msg = PointStamped()
                goal_point_msg.header.frame_id = "aruco_frame_depth"
                goal_point_msg.point.x = self.goal_pos[0]
                goal_point_msg.point.y = self.goal_pos[1]
                goal_point_msg.point.z = self.goal_pos[2]
                point_in_target_frame = listener.transformPoint("base_link", goal_point_msg)
                print(point_in_target_frame)
                self.goal_point_publisher.publish(point_in_target_frame)

                try:
                    gripper_finger_right = self.buffer.lookup_transform(
                        "base_link", "link_gripper_fingertip_right", rospy.Time()
                    )
                    gripper_finger_left = self.buffer.lookup_transform(
                        "base_link", "link_gripper_fingertip_left", rospy.Time()
                    )
                    # odom = self.buffer.lookup_transform(
                    #     "base_link", "link_mast", rospy.Time()
                    # )
                    # gripper_finger_left = self.buffer.lookup_transform(
                    #     "base_link", "link_gripper_fingertip_left", rospy.Time()
                    # )
                except (
                    tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException,
                ) as e:
                    print(e)
                    continue

                end_eff_tensor_ref = torch.Tensor(
                    [
                        # trans.transform.translation.x - 0.09,
                        # trans.transform.translation.y - 0.05,
                        # trans.transform.translation.z - 0.06,
                        (
                            gripper_finger_left.transform.translation.x
                            + gripper_finger_right.transform.translation.x
                        )
                        / 2,
                        # gripper_finger_left.transform.translation.y,
                        (
                            gripper_finger_left.transform.translation.y
                            + gripper_finger_right.transform.translation.y
                        )
                        / 2,
                        (
                            gripper_finger_left.transform.translation.z
                            + gripper_finger_right.transform.translation.z
                        )
                        / 2,
                    ]
                )
                print(f'end_eff_trans: {end_eff_tensor_ref=}')
                point = PointStamped()
                point.header.frame_id = "base_link"
                point.point.x = end_eff_tensor_ref[0]
                point.point.y = end_eff_tensor_ref[1]
                point.point.z = end_eff_tensor_ref[2]
                self.curr_ee_ps_ref_publisher.publish(point)
                # end_eff_odom = torch.Tensor(
                #     [
                #         # trans.transform.translation.x - 0.09,
                #         # trans.transform.translation.y - 0.05,
                #         # trans.transform.translation.z - 0.06,
                #         odom.transform.translation.x,
                #         # gripper_finger_left.transform.translation.y,
                #         odom.transform.translation.y,
                #         odom.transform.translation.z,
                #     ]
                # )
                # print(end_eff_odom)

                point = PointStamped()
                point.header.frame_id = "base_link"
                point.point.x = -end_eff_tensor[0]
                point.point.y = -end_eff_tensor[1]
                point.point.z = end_eff_tensor[2]
                self.curr_ee_ps_publisher.publish(point)
                # print(f"Curr: {end_eff_tensor=}")
                # print(end_eff_odom)

                joint_pos = self.joint_states.position
                lift_idx, wrist_idx, yaw_idx = (
                    self.joint_states.name.index("joint_lift"),
                    self.joint_states.name.index("wrist_extension"),
                    self.joint_states.name.index("joint_wrist_yaw"),
                )
                wrist_angle = joint_pos[yaw_idx]

                hyp = 0.06 * math.sqrt(2)

                # rate.sleep()

                # joint_pos = self.joint_states.position
                # lift_idx, wrist_idx, yaw_idx = (
                #     self.joint_states.name.index("joint_lift"),
                #     self.joint_states.name.index("wrist_extension"),
                #     self.joint_states.name.index("joint_wrist_yaw"),
                # )

                # end_eff_tensor = torch.Tensor(
                #     self.new_end_eff_to_xyz(
                #         [
                #             joint_pos[wrist_idx],
                #             joint_pos[yaw_idx],
                #             joint_pos[lift_idx],
                #             self.odometry[0],
                #             self.odometry[1],
                #             self.odometry[2],
                #         ]
                #     )
                # )

                # goal_pos_tensor = torch.Tensor(
                #     self.new_end_eff_to_xyz(self.goal_tensor)
                # )

                # print(f"goal tensor:  {goal_pos_tensor}")
                # print(f"Current EE pos: {end_eff_tensor}")

            rate.sleep()


def get_args():
    parser = argparse.ArgumentParser(description="main_slighting")
    parser.add_argument("--goal_pos", type=str, required=False, default="0,0")
    return parser.parse_args(rospy.myargv()[1:])


if __name__ == "__main__":
    args = get_args()
    goal_pos = args.goal_pos.split(",")

    node = HalSkills(goal_pos)
    # node.start()
    node.main()
