#!/usr/bin/env python3
import pdb
import time
import cv2
import csv
import rospy
import math
import argparse
import message_filters
import actionlib
from pathlib import Path
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from std_msgs.msg import String
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
from cv_bridge import CvBridge, CvBridgeError
import ppo_utils
import copy
import json
from ppo import MLP
import requests
import io


def load_ppo_model(pth_path):
    model = MLP(**ppo_utils.config)
    model = ppo_utils.load_pth_file_to_model(model, pth_path)
    return model


from stretch_learning.srv import Pick, PickResponse

# simulation import
import simulate
import base64

device = "cuda" if torch.cuda.is_available() else "cpu"

RUNNING = -1
SUCCESS = 1
NOT_STARTED = 2

INDEX_TO_KEYPRESSED = {
    # noop
    0: "noop",
    # arm up
    1: "arm up",
    # arm down
    2: "arm down",
    # arm out
    3: "arm out",
    # arm in
    4: "arm in",
    # base forward
    5: "base forward",
    # base back
    6: "base back",
    # base rotate left
    7: "base rotate left",
    # base rotate right
    8: "base rotate right",
    # gripper right
    9: "gripper left",
    # gripper left
    10: "gripper right",
    # gripper down
    11: "gripper down",
    # gripper up
    12: "gripper up",
    # gripper roll right
    13: "gripper roll right",
    # gripper roll left
    14: "gripper roll left",
    # gripper open
    15: "gripper open",
    # gripper close
    16: "gripper close",
}

kp_mapping = ["Arm out", "Arm in", "Gripper left", "Gripper right"]

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


url = "http://192.168.0.243:8000/predict"


class HalSkills(hm.HelloNode):
    def __init__(self, skill_name, model_type, train_type, goal_pos, is_2d, use_delta):
        hm.HelloNode.__init__(self)
        self.debug_mode = False
        self.rate = 10.0
        self.trajectory_client = actionlib.SimpleActionClient(
            "/stretch_controller/follow_joint_trajectory", FollowJointTrajectoryAction
        )

        self.is_2d = True
        self.use_delta = use_delta

        self.step_size = "medium"
        self.rad_per_deg = math.pi / 180.0
        self.small_deg = 3.0
        self.small_rad = self.rad_per_deg * self.small_deg
        self.small_translate = 0.005  # 0.02
        self.medium_deg = 6.0
        self.medium_rad = self.rad_per_deg * self.medium_deg
        self.medium_translate = 0.04
        self.mode = "position"  #'manipulation' #'navigation'

        self.goal_pos = list(map(float, goal_pos))
        self.goal_tensor = torch.Tensor(self.goal_pos).to(device)

        pth_path = "/home/strech/catkin_ws/src/stretch_ros/stretch_learning/checkpoints/ppo_point_and_shoot/policy.pth"
        self.model_ppo = load_ppo_model(pth_path)
        self.model_ppo.eval()

        self.init_node()

    def init_node(self):
        rospy.init_node("hal_skills_node")
        self.node_name = rospy.get_name()
        rospy.loginfo("{0} started".format(self.node_name))

    def goal_pos_callback(self, msg):
        self.goal_pos_pred = msg
        print(self.goal_pos_pred)

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

    def wrist_image_callback(self, ros_rgb_image):
        try:
            self.raw_wrist_image = self.cv_bridge.imgmsg_to_cv2(ros_rgb_image, "rgb8")
            # self.raw_wrist_image = cv2.rotate(
            #     self.raw_wrist_image, cv2.ROTATE_90_CLOCKWISE
            # )
            self.wrist_image = self.img_transform(self.raw_wrist_image)
            self.wrist_image = self.wrist_image.unsqueeze(0)
        except CvBridgeError as error:
            print(error)

    def head_image_callback(self, ros_rgb_image):
        try:
            self.raw_head_image = self.cv_bridge.imgmsg_to_cv2(ros_rgb_image, "rgb8")
            self.raw_head_image = cv2.rotate(
                self.raw_head_image, cv2.ROTATE_90_CLOCKWISE
            )
            self.head_image = self.img_transform(self.raw_head_image)
            self.head_image = self.head_image.unsqueeze(0)
        except CvBridgeError as error:
            print(error)

    def index_to_keypressed(self, index):
        _index_to_keypressed = {
            # noop
            0: "_",
            # arm up
            1: "8",
            # arm down
            2: "2",
            # arm out
            3: "w",
            # arm in
            4: "x",
            # base forward
            5: "4",
            # base back
            6: "6",
            # base rotate left
            7: "7",
            # base rotate right
            8: "9",
            # gripper right
            10: "a",
            # gripper left
            9: "d",
            # gripper down
            11: "c",
            # gripper up
            12: "v",
            # gripper roll right
            13: "o",
            # gripper roll left
            14: "p",
            # gripper open
            15: "0",
            # gripper close
            16: "5",
        }
        return _index_to_keypressed[index]

    def get_deltas(self):
        if self.step_size == "small":
            deltas = {"rad": self.small_rad, "translate": self.small_translate}
        if self.step_size == "medium":
            deltas = {"rad": self.medium_rad, "translate": self.medium_translate}
        if self.step_size == "big":
            deltas = {"rad": self.big_rad, "translate": self.big_translate}
        return deltas

    def get_command(self, c):
        keypressed_publisher = rospy.Publisher("key_pressed", Keypressed, queue_size=20)
        rospy.Rate(1)
        msg = Keypressed()
        msg.timestamp = (str)(rospy.get_rostime().to_nsec())

        # 8 or up arrow
        if c == "8" or c == "\x1b[A":
            command = {"joint": "joint_lift", "delta": self.get_deltas()["translate"]}
            msg.keypressed = "8"
            keypressed_publisher.publish(msg)
        # 2 or down arrow
        if c == "2" or c == "\x1b[B":
            command = {"joint": "joint_lift", "delta": -self.get_deltas()["translate"]}
            msg.keypressed = "2"
            keypressed_publisher.publish(msg)
        if self.mode == "manipulation":
            # 4 or left arrow
            if c == "4" or c == "\x1b[D":
                command = {
                    "joint": "joint_mobile_base_translation",
                    "delta": self.get_deltas()["translate"],
                }
                msg.keypressed = "4"
                keypressed_publisher.publish(msg)
            # 6 or right arrow
            if c == "6" or c == "\x1b[C":
                command = {
                    "joint": "joint_mobile_base_translation",
                    "delta": -self.get_deltas()["translate"],
                }
                msg.keypressed = "6"
                keypressed_publisher.publish(msg)
        elif self.mode == "position":
            # 4 or left arrow
            if c == "4" or c == "\x1b[D":
                command = {
                    "joint": "translate_mobile_base",
                    "inc": self.get_deltas()["translate"],
                }
                msg.keypressed = "4"
                keypressed_publisher.publish(msg)
            # 6 or right arrow
            if c == "6" or c == "\x1b[C":
                command = {
                    "joint": "translate_mobile_base",
                    "inc": -self.get_deltas()["translate"],
                }
                msg.keypressed = "6"
                keypressed_publisher.publish(msg)
            # 1 or end key
            if c == "7" or c == "\x1b[H":
                command = {
                    "joint": "rotate_mobile_base",
                    "inc": self.get_deltas()["rad"],
                }
                msg.keypressed = "7"
                keypressed_publisher.publish(msg)
            # 3 or pg down 5~
            if c == "9" or c == "\x1b[5":
                command = {
                    "joint": "rotate_mobile_base",
                    "inc": -self.get_deltas()["rad"],
                }
                msg.keypressed = "9"
                keypressed_publisher.publish(msg)
        elif self.mode == "navigation":
            rospy.loginfo("ERROR: Navigation mode is not currently supported.")

        if c == "w" or c == "W":
            command = {
                "joint": "wrist_extension",
                "delta": self.get_deltas()["translate"],
            }
            msg.keypressed = "w"
            keypressed_publisher.publish(msg)
        if c == "x" or c == "X":
            command = {
                "joint": "wrist_extension",
                "delta": -self.get_deltas()["translate"],
            }
            msg.keypressed = "x"
            keypressed_publisher.publish(msg)
        if c == "d" or c == "D":
            command = {"joint": "joint_wrist_yaw", "delta": -self.get_deltas()["rad"]}
            msg.keypressed = "d"
            keypressed_publisher.publish(msg)
        if c == "a" or c == "A":
            command = {"joint": "joint_wrist_yaw", "delta": self.get_deltas()["rad"]}
            msg.keypressed = "a"
            keypressed_publisher.publish(msg)
        if c == "v" or c == "V":
            command = {"joint": "joint_wrist_pitch", "delta": -self.get_deltas()["rad"]}
            msg.keypressed = "v"
            keypressed_publisher.publish(msg)
        if c == "c" or c == "C":
            command = {"joint": "joint_wrist_pitch", "delta": self.get_deltas()["rad"]}
            msg.keypressed = "c"
            keypressed_publisher.publish(msg)
        if c == "p" or c == "P":
            command = {"joint": "joint_wrist_roll", "delta": -self.get_deltas()["rad"]}
            msg.keypressed = "p"
            keypressed_publisher.publish(msg)
        if c == "o" or c == "O":
            command = {"joint": "joint_wrist_roll", "delta": self.get_deltas()["rad"]}
            msg.keypressed = "o"
            keypressed_publisher.publish(msg)
        if c == "5" or c == "\x1b[E" or c == "g" or c == "G":
            # grasp
            command = {
                "joint": "joint_gripper_finger_left",
                "delta": -self.get_deltas()["rad"],
            }
            msg.keypressed = "5"
            keypressed_publisher.publish(msg)
        if c == "0" or c == "\x1b[2" or c == "r" or c == "R":
            # release
            command = {
                "joint": "joint_gripper_finger_left",
                "delta": self.get_deltas()["rad"],
            }
            msg.keypressed = "0"
            keypressed_publisher.publish(msg)
        if c == "i" or c == "I":
            # head up
            command = {
                "joint": "joint_head_tilt",
                "delta": (2.0 * self.get_deltas()["rad"]),
            }
            msg.keypressed = "i"
            keypressed_publisher.publish(msg)
        if c == "," or c == "<":
            # head down
            command = {
                "joint": "joint_head_tilt",
                "delta": -(2.0 * self.get_deltas()["rad"]),
            }
            msg.keypressed = ","
            keypressed_publisher.publish(msg)
        if c == "j" or c == "J":
            command = {
                "joint": "joint_head_pan",
                "delta": (2.0 * self.get_deltas()["rad"]),
            }
            msg.keypressed = "j"
            keypressed_publisher.publish(msg)
        if c == "l" or c == "L":
            command = {
                "joint": "joint_head_pan",
                "delta": -(2.0 * self.get_deltas()["rad"]),
            }
            msg.keypressed = "l"
            keypressed_publisher.publish(msg)
        if c == "b" or c == "B":
            rospy.loginfo("process_keyboard.py: changing to BIG step size")
            self.step_size = "big"
            msg.keypressed = "b"
            keypressed_publisher.publish(msg)
        if c == "m" or c == "M":
            rospy.loginfo("process_keyboard.py: changing to MEDIUM step size")
            self.step_size = "medium"
            msg.keypressed = "m"
            keypressed_publisher.publish(msg)
        if c == "s" or c == "S":
            rospy.loginfo("process_keyboard.py: changing to SMALL step size")
            self.step_size = "small"
            msg.keypressed = "s"
            keypressed_publisher.publish(msg)
        if c == "q" or c == "Q":
            rospy.loginfo("keyboard_teleop exiting...")
            rospy.signal_shutdown("Received quit character (q), so exiting")

        ####################################################

        return command

    def send_command(self, command):
        joint_state = self.joint_states
        if (joint_state is not None) and (command is not None):
            point = JointTrajectoryPoint()
            point.time_from_start = rospy.Duration(0.0)
            trajectory_goal = FollowJointTrajectoryGoal()
            trajectory_goal.goal_time_tolerance = rospy.Time(1.0)

            joint_name = command["joint"]
            trajectory_goal.trajectory.joint_names = [joint_name]
            if "inc" in command:
                inc = command["inc"]
                new_value = inc
            elif "delta" in command:
                joint_index = joint_state.name.index(joint_name)
                joint_value = joint_state.position[joint_index]
                delta = command["delta"]
                new_value = joint_value + delta
            point.positions = [new_value]
            trajectory_goal.trajectory.points = [point]
            trajectory_goal.trajectory.header.stamp = rospy.Time.now()
            print(trajectory_goal)
            self.trajectory_client.send_goal(trajectory_goal)

    ######################### Hard Coded Commands #########################
    def grasp_primitive(self):
        point = JointTrajectoryPoint()
        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.goal_time_tolerance = rospy.Time(1.0)
        trajectory_goal.trajectory.joint_names = ["joint_gripper_finger_left"]
        point.positions = [-0.2]
        trajectory_goal.trajectory.points = [point]
        self.trajectory_client.send_goal(trajectory_goal)
        grip_change_time = 2
        rospy.sleep(grip_change_time)

    def lift_arm_primitive(self):
        point = JointTrajectoryPoint()
        joint_lift_index = self.joint_states.name.index("joint_lift")
        curr_lift = self.joint_states.position[joint_lift_index]
        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.goal_time_tolerance = rospy.Time(1.0)
        trajectory_goal.trajectory.joint_names = ["joint_lift"]
        point.positions = [curr_lift + 0.1]
        trajectory_goal.trajectory.points = [point]
        self.trajectory_client.send_goal(trajectory_goal)
        arm_lift_time = 1
        rospy.sleep(arm_lift_time)

    def retract_arm_primitive(self):
        # point = JointTrajectoryPoint()
        # trajectory_goal = FollowJointTrajectoryGoal()
        # trajectory_goal.goal_time_tolerance = rospy.Time(1.0)
        # trajectory_goal.trajectory.joint_names = ["joint_arm_l0"]
        # point.positions = [0.005]
        # trajectory_goal.trajectory.points = [point]
        # self.trajectory_client.send_goal(trajectory_goal)
        kp = self.index_to_keypressed(4)

        self.send_command(kp)
        self.send_command(kp)
        self.send_command(kp)
        self.send_command(kp)
        arm_lift_time = 2.5
        rospy.sleep(arm_lift_time)

    # -----------------pick_pantry() initial configs-----------------#
    def move_arm_pick_pantry(self):
        rospy.loginfo("Set arm")
        self.pick_starting_height = 0.998
        self.joint_lift_index = self.joint_states.name.index("joint_lift")
        pose = {
            "wrist_extension": 0.01,
            "joint_lift": self.pick_starting_height - 0.2,  # for cabinet 0.175
            "joint_wrist_pitch": 0.2,
            "joint_wrist_yaw": -0.09,
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
        print("1")
        while not done_head_pan:
            print("2")
            if self.joint_states:
                done_head_pan = self.move_head_pick_pantry()
            rate.sleep()
        done_initial_config = False
        while not done_initial_config:
            if self.joint_states:
                done_initial_config = self.move_arm_pick_pantry()
            rate.sleep()
        self.open_grip()

    # -----------------pick_table() initial configs-----------------#
    def move_arm_pick_table(self):
        rospy.loginfo("Set arm")
        self.pick_starting_height = 0.9096
        self.joint_lift_index = self.joint_states.name.index("joint_lift")
        pose = {
            "wrist_extension": 0.01,
            "joint_lift": self.pick_starting_height,
            "joint_wrist_pitch": 0.2,
            "joint_wrist_yaw": -0.09,
        }
        self.move_to_pose(pose)
        return True

    def pick_table_initial_config(self, rate):
        done_head_pan = False
        while not done_head_pan:
            if self.joint_states:
                done_head_pan = self.move_head_pick()
            rate.sleep()
        done_initial_config = False
        while not done_initial_config:
            if self.joint_states:
                done_initial_config = self.move_arm_pick_table()
            rate.sleep()
        self.open_grip()

    def check_pick_termination(self):
        self.joint_lift_index = self.joint_states.name.index("joint_lift")
        curr_height = self.joint_states.position[self.joint_lift_index]
        if curr_height - self.pick_starting_height > 0.05:
            return True
        return False

    # -----------------place_table() initial configs-----------------#
    def move_arm_place_table(self):
        rospy.loginfo("Set arm")
        self.gripper_finger = self.joint_states.name.index("joint_gripper_finger_left")
        self.wrist_ext_indx = self.joint_states.name.index("joint_arm_l3")
        self.place_starting_extension = 0.01
        pose = {
            "wrist_extension": 0.01,
            "joint_lift": 0.9769,
            "joint_wrist_pitch": 0.1948,
            "joint_wrist_yaw": -0.089,
        }
        self.move_to_pose(pose)
        return True

    def move_head_place_table(self):
        tilt = -0.1612
        pan = -1.757
        rospy.loginfo("Set head pan")
        pose = {"joint_head_pan": pan, "joint_head_tilt": tilt}
        self.move_to_pose(pose)
        return True

    def place_table_initial_config(self, rate):
        done_head_pan = False
        while not done_head_pan:
            if self.joint_states:
                done_head_pan = self.move_head_place_table()
            rate.sleep()
        done_initial_config = False
        while not done_initial_config:
            if self.joint_states:
                done_initial_config = self.move_arm_place_table()
            rate.sleep()

    def check_place_termination(self):
        # TODO place termination should also have a arm extension perhaps
        if (
            self.joint_states.position[self.gripper_finger] > 0.19
            and self.joint_states.position[self.wrist_ext_indx] * 4 > 0.04812 * 4
        ):
            return True
        return False

    # -----------------open_drawer() initial configs-----------------#
    def move_arm_open_drawer(self):
        rospy.loginfo("Set arm")
        self.gripper_finger = self.joint_states.name.index("joint_gripper_finger_left")
        self.wrist_ext_indx = self.joint_states.name.index("joint_arm_l3")
        self.place_starting_extension = 0.01
        pose = {
            "wrist_extension": 0.01,
            "joint_lift": 0.451,
            "joint_wrist_pitch": -0.3076,
            "joint_wrist_yaw": 0.0,
        }
        self.move_to_pose(pose)
        return True

    def move_head_open_drawer(self):
        tilt = -0.8673
        pan = -1.835
        rospy.loginfo("Set head pan")
        pose = {"joint_head_pan": pan, "joint_head_tilt": tilt}
        self.move_to_pose(pose)
        return True

    def open_drawer_initial_config(self, rate):
        done_head_pan = False
        while not done_head_pan:
            if self.joint_states:
                done_head_pan = self.move_head_open_drawer()
            rate.sleep()
        done_initial_config = False
        while not done_initial_config:
            if self.joint_states:
                done_initial_config = self.move_arm_open_drawer()
            rate.sleep()

    def start(self):
        self.action_status = NOT_STARTED
        self.skill_name = "pick_pantry_all"
        self.train_type = "bc_oracle"

        s = rospy.Service("pick_server", Pick, self.callback)
        rospy.loginfo("Pick server has started")
        rospy.spin()

    def callback(self, req):
        if self.action_status == NOT_STARTED:
            # call hal_skills
            self.action_status = RUNNING
            if not req.is_pick:
                self.skill_name = "place_table"
        self.main()

        return PickResponse(self.action_status)

    def end_eff_to_xy(self, extension, yaw):
        # extension, yaw = deltas
        yaw_delta = -(yaw - self.base_gripper_yaw)  # going right is more negative
        x = self.gripper_len * np.sin(yaw_delta)
        y = self.gripper_len * np.cos(yaw_delta) + extension
        return [x, y]

    def end_eff_to_xyz(self, joint_state):
        extension = joint_state[0]
        yaw = joint_state[1]
        lift = joint_state[2]

        gripper_len = 0.22
        base_gripper_yaw = -0.09
        yaw_delta = -(yaw - base_gripper_yaw)  # going right is more negative
        y = gripper_len * torch.cos(torch.tensor([yaw_delta])) + extension
        x = gripper_len * torch.sin(torch.tensor([yaw_delta]))
        z = lift

        return np.array([x.item(), y.item(), z])

    def project_point(self, angle, test_point):
        # Unit vector coordinates
        unit_vector = (math.cos(angle), math.sin(angle))

        # Dot product of the test point and unit vector
        dot_product = test_point[0] * unit_vector[0] + test_point[1] * unit_vector[1]

        # Projected point coordinates
        projected_point = (dot_product * unit_vector[0], dot_product * unit_vector[1])

        if np.sign(projected_point[0]) == 0 or np.sign(projected_point[0]) == np.sign(
            unit_vector[0]
        ):
            return True
        else:
            return False

    def is_point_in_half_circle(self, rotation_angle, center, radius, test_point):
        center_offset = 0.005
        unit_vector = np.array([math.cos(rotation_angle), math.sin(rotation_angle)])
        center += center_offset * unit_vector

        # Translate the test point coordinates relative to the center of the circle
        translated_point = [test_point[0] - center[0], test_point[1] - center[1]]

        # Calculate the projection of the translated point onto a vector defined by the rotation angle
        projection = self.project_point(rotation_angle, translated_point)

        if projection and np.linalg.norm(translated_point) <= radius:
            return True
        else:
            return False

    @torch.no_grad()
    def main(self):
        print("start of main")
        self.joint_states_subscriber = rospy.Subscriber(
            "/stretch/joint_states", JointState, self.joint_states_callback
        )
        self.wrist_image_subscriber = message_filters.Subscriber(
            "/wrist_camera/color/image_raw", Image
        )
        self.wrist_image_subscriber.registerCallback(self.wrist_image_callback)
        self.head_image_subscriber = message_filters.Subscriber(
            "/head_camera/color/image_raw", Image
        )
        self.head_image_subscriber.registerCallback(self.head_image_callback)
        goal_pos_sub = rospy.Subscriber(
            "/hal_prediction_node/goal_pos", String, self.goal_pos_callback
        )

        rate = rospy.Rate(self.rate)

        print("start of reset")
        # get hal to starting position for pick
        if "pick" in self.skill_name or "point" in self.skill_name:
            self.pick_pantry_initial_config(rate)
            # pass
        elif self.skill_name == "place_table":
            self.place_table_initial_config(rate)
        elif self.skill_name == "open_drawer":
            self.open_drawer_initial_config(rate)
        else:
            raise NotImplementedError
        print("start of loop")
        keypresses = []
        img_count = 0
        times = []

        joint_pos = ["wrist_extension", "joint_wrist_yaw", "joint_lift"]

        kp_reduced_mapping = {
            # arm out
            0: 3,
            # arm in
            1: 4,
            # gripper right
            2: 9,
            # gripper left
            3: 10,
            # arm up
            4: 1,
            # arm down
            5: 2,
        }
        initial_pos = None
        onpolicy_pos = []
        onpolicy_kp = []
        goal = (0.0, 0.0)
        # while not rospy.is_shutdown():
        goal_tensors = []
        end_eff_tensors = []
        # for _ in range(70):
        # ref_text_tokenized = tokenizer(["salt"])
        # text = ["salt"]
        mode = False
        fixed_goal_pos_tensor = torch.Tensor((0.368, 0.289, 0.823))
        use_fixed = False

        prev_inputs_wrist = None
        prev_inputs_head = None

        # goal_pos_tensor = torch.Tensor(
        #     self.end_eff_to_xyz(self.goal_tensor.cpu().detach().numpy().tolist())
        # )

        i = 0
        count = 0
        goal_sum = torch.tensor(0)
        for i in range(100):
            if (
                self.joint_states is not None
                and self.wrist_image is not None
                and self.head_image is not None
            ):
                # check delta to determine skill termination
                # if "pick" in self.skill_name and self.check_pick_termination():
                #     rospy.loginfo("\n\n***********Pick Completed***********\n\n")
                #     self.action_status = SUCCESS
                #     print(times)
                #     return 1
                # elif "place" in self.skill_name and self.check_place_termination():
                #     rospy.loginfo("\n\n***********Place Completed***********\n\n")
                #     self.action_status = SUCCESS
                #     return 1

                # # if not, continue with next command
                # if len(self.joint_states_data.size()) <= 1:
                #     print(self.joint_states_data)
                #     continue

                continue

                # TODO: at each iteration, the goal pos should be available in
                # the variable: self.goal_pos_pred.

                # if i == 0:
                #     goal_pos_tensor[0] = self.goal_pos[0][0]
                goal_pos_tensor = fixed_goal_pos_tensor
                goal_tensors.append(goal_pos_tensor)
                end_eff_tensors.append(end_eff_tensor)
                print(f"Current EE pos: {end_eff_tensor}")

                if self.is_point_in_half_circle(
                    joint_pos[yaw_idx], goal_pos_tensor[:2], 0.025, end_eff_tensor[:2]
                ) and (np.abs(goal_pos_tensor[2] - end_eff_tensor[2]) < 0.035):
                    print("Got to goal!!")
                    self.grasp_primitive()
                    break

                inp = goal_pos_tensor - end_eff_tensor

                start = time.time()
                print(f"Input to model {inp}")
                prediction = self.model_ppo(inp)
                print(f"{prediction=}")
                print(f"ppo: {time.time() - start}")
                times.append(time.time() - start)

                keypressed_index = torch.argmax(prediction).item()
                keypressed = kp_reduced_mapping[keypressed_index]
                keypressed = self.index_to_keypressed(keypressed)

                _pos = (goal_pos_tensor - end_eff_tensor).cpu().numpy()
                print(f"pos: {_pos}")
                onpolicy_pos.append(_pos)
                onpolicy_kp.append(keypressed_index)

                if self.debug_mode:
                    img_count += 1
                    img_path = Path(img_dir, f"{img_count:04}.jpg")
                    cv2.imwrite(str(img_path), self.raw_image)
                    keypresses.append(str(keypressed_index))
                    with open(csv_path, "w", encoding="UTF8", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(keypresses)

                if keypressed == "_":
                    # noop
                    print("NOOP")
                    continue

                command = self.get_command(keypressed)

                print(f"{rospy.Time().now()}, {keypressed_index=}, {command=}")
                self.send_command(command)
                # time.sleep(1)
            rate.sleep()
        timestr = time.strftime("%Y%m%d-%H%M%S")
        if self.is_2d:
            title = f"{timestr}_2d"
        elif self.use_delta:
            title = f"{timestr}_use_delta"
        else:
            title = f"{timestr}_no_delta"
        print(goal_tensors)
        print(end_eff_tensors)
        fig = plt.figure()
        ax = plt.axes(xlim=(-1, 1), ylim=(-1, 1))

        def animate(i):
            ax.clear()  # clear axes
            ax.set_xlim(-0.05, 0.05)
            ax.set_ylim(0.4, 0.6)

            ax.plot(goal_tensors[i][0], goal_tensors[i][1], "ro")
            # ax.plot(end_eff_tensors[i][0], end_eff_tensors[i][1], 'go')

            return ax

        anim = animation.FuncAnimation(
            fig, animate, frames=len(goal_tensors), interval=1000
        )
        anim.save("points.gif", writer="imagemagick", fps=10)
        clip = VideoFileClip("points.gif")
        # Save the resulting MP4 file
        clip.write_videofile("points.mp4", fps=10)
        for i in range(len(goal_tensors)):
            plt.plot([goal_tensors[i][0], goal_tensors[i][1]], color="red")
            plt.plot([end_eff_tensors[i][0], end_eff_tensors[i][1]], color="green")
            plt.show()
        start_ext = initial_pos[0]
        start_yaw = initial_pos[1]
        goal_ext = self.goal_pos[0][0]
        goal_yaw = self.goal_pos[0][1]
        ckpt_path = "/home/strech/catkin_ws/src/stretch_ros/stretch_learning/checkpoints/pick_pepper_salt/d745bda3_epoch=100_val_loss=0.000173.pt"
        save_fig_path = "/home/strech/catkin_ws/src/stretch_ros/stretch_learning/nodes/plots/sep_6_graphs"
        sim_x, sim_y, sim_onpolicy_kp = simulate.run_simulate(
            start_ext, start_yaw, goal_ext, goal_yaw, ckpt_path, save_fig_path
        )

        self.overlay_plot(
            sim_x,
            sim_y,
            sim_onpolicy_kp,
            onpolicy_pos,
            onpolicy_kp,
            goal,
            title,
            save_dir="plots",
        )
        # self.overlay_plot(self, sim_x, sim_y, onpolicy_kp, pts, labels, goal, title, file=None, save_dir='temp')
        # self.decision_boundary(
        #     onpolicy_pos, onpolicy_kp, goal, title=title, save_dir="plots"
        # )

    # def decision_boundary(self, pts, labels, goal, title, file=None, save_dir="temp"):
    #     pts = np.array(pts)
    #     print(f'pts: {pts}')
    #     labels = np.array(labels)
    #     x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    #     y_min, y_max = pts[:, 1].min(), pts[:, 1].max()

    #     print(f'x_min: {x_min}')
    #     print(f'x_max: {x_max}')

    #     scatter = plt.scatter(
    #         pts[:, 0], pts[:, 1], c=labels, cmap="viridis", s=5, alpha=1
    #     )
    #     plt.plot(goal[0], goal[1], marker="*", markersize=15, color="red")
    #     fig_handle = plt.figure()
    #     handles, _ = scatter.legend_elements()
    #     filtered_kp_mapping = [kp_mapping[i] for i in np.unique(labels)]
    #     plt.legend(handles, filtered_kp_mapping, title="Classes")
    #     plt.xlabel("Relative x")
    #     plt.ylabel("Relative y")
    #     # plt.xlim(x_min - 0.5, x_max + 0.5)
    #     # plt.ylim(y_min - 0.5, y_max + 0.5)
    #     plt.title(title)
    #     print(f"Saving as decision_boundary{'_' + file if file else ''}.png")

    #     save_dir = '/home/strech/catkin_ws/src/stretch_ros/stretch_learning/nodes/plots/sep_5_graphs'
    #     save_path = Path(save_dir, f"{title.replace(' ', '_')}.png")
    #     save_path.parent.mkdir(exist_ok=True)
    #     plt.savefig(
    #         # f'decision_boundary{"_" + file if file else ""}.png',
    #         save_path,
    #         dpi=300,
    #         bbox_inches="tight",
    #     )

    #     pl.dump(fig_handle, file(f"{title}.pickle", "wb"))
    #     plt.close()
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from pathlib import Path
    # import pickle as pl

    def decision_boundary(self, pts, labels, goal, title, file=None, save_dir="temp"):
        import matplotlib

        matplotlib.use("Agg")
        pts = np.array(pts)
        labels = np.array(labels)

        print(f"shape: {pts.shape}")

        x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()

        scatter = plt.scatter(
            pts[:, 0], -pts[:, 1], c=labels, cmap="viridis", s=5, alpha=1
        )
        plt.plot(goal[0], -goal[1], marker="*", markersize=10, color="red")
        plt.plot(pts[0, 0], -pts[0, 1], marker="o", markersize=8, color="green")
        handles, _ = scatter.legend_elements()
        plt.legend(handles, [kp_mapping[i] for i in np.unique(labels)], title="Classes")

        plt.xlabel("Relative x")
        plt.ylabel("Relative y")
        plt.xlim(x_min - 0.5, x_max + 0.5)
        plt.ylim(-(y_max + 0.5), -(y_min - 0.5))
        # plt.xlim(-0.6,0.8)
        # plt.ylim(-0.4,0.8)
        plt.title(title)

        save_dir = "/home/strech/catkin_ws/src/stretch_ros/stretch_learning/nodes/plots/sep_5_graphs"
        save_path = Path(save_dir, f"{title.replace(' ', '_')}.png")
        save_path.parent.mkdir(exist_ok=True)
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        with open(f"{title}.pickle", "wb") as pickle_file:
            pl.dump(fig_handle, pickle_file)

        plt.close()

    def get_decision_boundary_points(self, mins, maxs, res):
        ranges = [
            np.arange(mins[i], maxs[i] + res[i], res[i]) for i in range(len(mins))
        ]
        grids = np.meshgrid(*ranges, indexing="ij")
        pts = np.stack(grids, axis=-1).reshape(-1, len(mins))

        predicted_kps = []
        pts = np.stack(grids, axis=-1).reshape(-1, len(mins))
        for p in pts:
            dx, dy = self.goal_pos[0] - p[0], self.goal_pos[1] - p[1]
            inp = np.append(p, [dx, dy])
            inp = torch.from_numpy(inp).float().unsqueeze(0).to(device)
            predicted_action = self.model(inp)
            predicted_kp = torch.argmax(predicted_action).item()
            predicted_kps.append([predicted_kp])
        predicted_kps = np.array(predicted_kps).flatten()
        # pts = np.flip(pts, axis=0)
        # pts = -pts
        # pts[:, 0] = -pts[:, 0]

        return pts, predicted_kps

    def overlay_plot(
        self,
        sim_x,
        sim_y,
        onpolicy_kp,
        pts,
        labels,
        goal,
        title,
        file=None,
        save_dir="temp",
    ):
        import matplotlib

        matplotlib.use("Agg")
        sim_kp_mapping = [
            "Sim Arm out",
            "Sim Arm in",
            "Sim Gripper left",
            "Sim Gripper right",
        ]

        pts = np.array(pts)
        labels = np.array(labels)
        print(f"shape: {pts.shape}")

        x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()

        scatter = plt.scatter(
            pts[:, 0], -pts[:, 1], c=labels, cmap="viridis", s=5, alpha=1
        )
        plt.plot(goal[0], -goal[1], marker="*", markersize=10, color="red")
        plt.plot(pts[0, 0], -pts[0, 1], marker="o", markersize=8, color="green")
        handles, _ = scatter.legend_elements()
        plt.legend(handles, [kp_mapping[i] for i in np.unique(labels)], title="Classes")

        plt.xlabel("Relative x")
        plt.ylabel("Relative y")
        x_lim = [x_min - 0.5, x_max + 0.5]
        y_lim = [-(y_max + 0.5), -(y_min - 0.5)]
        plt.xlim(x_lim[0], x_lim[1])
        plt.ylim(y_lim[0], y_lim[1])
        # plt.xlim(-0.6,0.8)
        # plt.ylim(-0.4,0.8)
        plt.title(title)

        # plotting sim

        scatter = plt.scatter(
            sim_x, sim_y, c=onpolicy_kp, cmap="viridis", s=2, alpha=0.3
        )
        handles2, _ = scatter.legend_elements()
        plt.legend(
            handles + handles2,
            [kp_mapping[i] for i in np.unique(labels)]
            + [sim_kp_mapping[i] for i in np.unique(onpolicy_kp)],
            title="Classes",
        )

        # plotting decision boundary
        # print(f'x_lims: {x_lim}')
        # print(f'y_lims: {y_lim}')
        # dec_bound_pts, dec_bound_kps = self.get_decision_boundary_points([x_min-0.45, -(y_max + 0.45)], [x_max+0.45, -(y_min - 0.45)], [0.005, 0.005])
        # print(f'decisionn_boundary x_lims: {np.min(dec_bound_pts[:, 0])}, {np.max(dec_bound_pts[:, 0])}')
        # print(f'decisionn_boundary y_lims: {np.min(dec_bound_pts[:, 1])}, {np.max(dec_bound_pts[:, 1])}')

        # scatter = plt.scatter(dec_bound_pts[:, 0], dec_bound_pts[:, 1], c=dec_bound_kps, cmap="viridis", s=5, alpha=0.01)

        save_dir = "/home/strech/catkin_ws/src/stretch_ros/stretch_learning/nodes/plots/sep_18_graphs"
        save_path = Path(save_dir, f"{title.replace(' ', '_')}.png")
        save_path.parent.mkdir(exist_ok=True)
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        with open(f"{title}.pickle", "wb") as pickle_file:
            pl.dump(fig_handle, pickle_file)

        plt.close()


# Usage example:
# decision_boundary(pts, labels, goal, "Decision Boundary", file=None, save_dir="temp")


def get_args():
    supported_skills = [
        "pick_pantry",
        "place_table",
        "open_drawer",
        "pick_pepper",
        "pick_pantry_all",
        "pick_whisk",
        "pick_whisk_pantry",
        "pick_salt",
        "point_shoot",
    ]
    supported_models = ["visuomotor_bc", "irl"]
    supported_types = [
        # visuomotor_bc
        "reg",
        "reg-no-vel",
        "end-eff",
        "end-eff-img",
        "end-eff-img-ft",
        "bc_top",
        "bc_all",
        "bc_oracle",
        "resnet_bc_all",
        "resnet_bc_oracle",
        # irl
        "iql",
    ]

    parser = argparse.ArgumentParser(description="main_slighting")
    parser.add_argument(
        "--skill_name", type=str, choices=supported_skills, default="pick_salt"
    )
    parser.add_argument(
        "--input_dim", type=str, choices=["4d", "2d"], required=False, default="2d"
    )
    parser.add_argument("--use_delta", action="store_true", default=True)
    parser.add_argument(
        "--model_type", type=str, choices=supported_models, default="visuomotor_bc"
    )
    parser.add_argument(
        "--train_type", type=str, choices=supported_types, default="bc_oracle"
    )
    parser.add_argument("--goal_pos", type=str, required=False, default="0,0")
    return parser.parse_args(rospy.myargv()[1:])


if __name__ == "__main__":
    args = get_args()
    skill_name = args.skill_name
    model_type = args.model_type
    train_type = args.train_type
    goal_pos = args.goal_pos.split(",")
    is_2d = args.input_dim == "2d"
    use_delta = args.use_delta

    node = HalSkills(skill_name, model_type, train_type, goal_pos, is_2d, use_delta)
    # node.start()
    node.main()
