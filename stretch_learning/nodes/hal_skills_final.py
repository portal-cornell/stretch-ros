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
from moviepy.editor import VideoFileClip
from PIL import Image as Im
import torch
import torch.nn as nns
import numpy as np
from torchvision import transforms
from cv_bridge import CvBridge, CvBridgeError
import ppo_utils
import open_clip

from std_srvs.srv import Trigger
from geometry_msgs.msg import PointStamped
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from copy import deepcopy
import tf2_py
import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from collections import deque
import copy
# import owl_nms_clip as onc


def load_ppo_model(pth_path):
    model = ppo_utils.MLP(**ppo_utils.config)
    model = ppo_utils.load_pth_file_to_model(model, pth_path)
    return model


from transformers import OwlViTProcessor

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
tokenizer = open_clip.get_tokenizer("EVA02-B-16")

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

kp_mapping = [
    "Arm out",
    "Arm in",
    "Gripper right",
    "Gripper left",
    "Lift Up",
    "Lift Down",
    "Base Left",
    "Base Right",
    "Base Rotate Left",
    "Base Rotate Right",
]

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


class HalSkillsNode(hm.HelloNode):
    def __init__(self):
        hm.HelloNode.__init__(self)
        self.debug_mode = False
        self.rate = 1.0
        self.time = time.time()
        self.trajectory_client = actionlib.SimpleActionClient(
            "/stretch_controller/follow_joint_trajectory", FollowJointTrajectoryAction
        )
        self.device = "cpu"
        self.step_size = "medium"
        self.rad_per_deg = math.pi / 180.0
        self.small_deg = 3.0
        self.small_rad = self.rad_per_deg * self.small_deg
        self.small_translate = 0.005  # 0.02
        self.medium_deg = 6.0
        self.medium_rad = self.rad_per_deg * self.medium_deg * (3 / 5)
        self.medium_translate = 0.03 * (3 / 5)
        self.mode = "position"  #'manipulation' #'navigation'
        # self.mode = "navigation"                              NOTE: changed frm  navigation to position, since navigation does not support base movement

        self.home_odometry = None
        self.odometry = None
        self.joint_states = None
        self.joint_states_data = None
        self.wrist_image = None
        self.goal_pos_pred = None
        self.goal_pos_count = 0
        self.head_image = None
        self.cv_bridge = CvBridge()
        self.pick_prompt = None

        OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
        OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

        self.img_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
            ]
        )

        # pth_path = "/home/strech/catkin_ws/src/stretch_ros/stretch_learning/checkpoints/ppo_point_and_shoot/policy_fixed_reduced.pth"
        pth_path = "/home/strech/catkin_ws/src/stretch_ros/stretch_learning/checkpoints/ppo_point_and_shoot/policy_base.pth"
        # pth_path = "/home/strech/catkin_ws/src/stretch_ros/stretch_learning/checkpoints/ppo_point_and_shoot/policy.pth"

        self.model_ppo = load_ppo_model(pth_path)
        self.model_ppo.eval()

        self.rate = 10.0
        self.pick_prompt_publisher = rospy.Publisher(
            "/stretch/pick_prompt", String, queue_size=1
        )
        self.goal_point_publisher = rospy.Publisher(
            "/hal_skills_final/goal_point", PointStamped, queue_size=1
        )
        self.curr_ee_ps_publisher = rospy.Publisher(
            "hal_skills_final/curr_ee_ps", PointStamped, queue_size=1
        )
        self.subscribe()

        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)

        self.prev_odom = None
        self.curr_odom = None
        # self.init_node()  # comment out when using move

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

    def goal_pos_callback(self, msg):
        self.goal_pos_count += 1
        if self.goal_pos_count <= 20:
            goal_pos_str = msg.data
            goal_pos_x, goal_pos_y, goal_pos_z = goal_pos_str.split(",")
            goal_pos_x = goal_pos_x.split("[[")[1].strip()
            goal_pos_y = goal_pos_y.strip()
            goal_pos_z = goal_pos_z.split("]]")[0].strip()
            self.goal_pos_pred = torch.from_numpy(
                np.array([float(goal_pos_x), float(goal_pos_y), float(goal_pos_z)])
            )
            self.goal_pos_pred.to(self.device).unsqueeze(0)

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

    def head_image_callback(self, ros_rgb_image):
        try:
            self.raw_head_image = self.cv_bridge.imgmsg_to_cv2(ros_rgb_image, "bgr8")
            self.head_image = self.img_transform(self.raw_head_image)
            self.head_image = self.head_image.unsqueeze(0)
        except CvBridgeError as error:
            print(error)

    def convert_coordinates(self, point, angle):
        # Create the rotation matrix using the angle
        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )

        # Convert the point to the new coordinate system
        new_point = np.dot(rotation_matrix, point)

        if np.sign(new_point[0]) >= 0:
            return True
        else:
            return False

    def is_point_in_half_circle(self, rotation_angle, center, radius, test_point):
        center_offset = 0
        rotated_angle = np.copy(rotation_angle)
        rotated_angle += np.pi / 2
        unit_vector = np.array([math.cos(rotated_angle), math.sin(rotated_angle)])
        center_copy = np.copy(center)
        center_copy += center_offset * unit_vector

        # Translate the test point coordinates relative to the center of the circle
        translated_point = [
            test_point[0] - center_copy[0],
            test_point[1] - center_copy[1],
        ]

        # Calculate the projection of the translated point onto a vector defined by the rotation angle
        projection = self.convert_coordinates(translated_point, rotated_angle)

        if projection and np.linalg.norm(translated_point) <= radius:
            return True
        else:
            return False

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

    def lift_arm_primitive(self):
        point = JointTrajectoryPoint()
        joint_lift_index = self.joint_states.name.index("joint_lift")
        curr_lift = self.joint_states.position[joint_lift_index]
        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.goal_time_tolerance = rospy.Time(1.0)
        trajectory_goal.trajectory.joint_names = ["joint_lift"]
        point.positions = [curr_lift + 0.02]
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
            "wrist_extension": 0.05,
            "joint_lift": self.pick_starting_height - 0.45,  # for cabinet -0.175
            "joint_wrist_pitch": 0.0,
            "joint_wrist_yaw": 0.0,
            # "joint_wrist_pitch": 0.2,
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

        # s = rospy.Service("pick_server", Pick, self.callback)
        rospy.loginfo("Pick server has started")
        rospy.spin()

    def callback(self, req):
        if self.action_status == NOT_STARTED:
            # call hal_skills
            self.action_status = RUNNING
            if not req.is_pick:
                self.skill_name = "place_table"
        self.main()

    # def odom_to_js(self, odom_data):
    #     """
    #     Takes in odom data [x, y, theta] from /odom topic and flips the x and y
    #     """
    #     odom_data[0] = odom_data[0] * -1
    #     odom_data[1] = odom_data[1] * -1
    #     return odom_data

    def odom_to_js(self, odom_data):
        """
        Takes in odom data [x, y, theta] from /odom topic and rotates +90 degrees
        """
        tmp = deepcopy(odom_data[0])
        odom_data[0] = deepcopy(odom_data[1]) * -1
        odom_data[1] = tmp
        return odom_data

    def end_eff_to_xyz(self, joint_state):
        extension = joint_state[0]
        yaw = joint_state[1]
        lift = joint_state[2]
        base_x = joint_state[3]
        base_y = joint_state[4]
        base_angle = joint_state[5]

        gripper_len = 0.27
        base_gripper_yaw = -0.09

        # find cx, cy in base frame
        # point = (0.1, 0.17)
        point = (0.03, 0.17)
        pivot = (0, 0)
        cx, cy = self.rotate_odom(point, base_angle, pivot)

        # cx, cy in origin frame
        cx += base_x
        cy += base_y

        extension_y_offset = extension * np.cos(base_angle)
        extension_x_offset = extension * -np.sin(base_angle)
        yaw_delta = yaw - base_gripper_yaw
        gripper_y_offset = gripper_len * np.cos(yaw_delta + base_angle)
        gripper_x_offset = gripper_len * -np.sin(yaw_delta + base_angle)

        x = cx + extension_x_offset + gripper_x_offset
        y = cy + extension_y_offset + gripper_y_offset
        z = lift

        return np.array([x.item(), y.item(), z])

    def _find_mode(self, buffer):
        vals, counts = np.unique(buffer, return_counts=True)
        index = np.argmax(counts)
        return vals[index]

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

    def pick_prompt_callback(self, pick_prompt_msg):
        self.pick_prompt = pick_prompt_msg.data.lower()

    def handover(self, _):
        # lift arm
        pose = {
            "joint_lift": 0.98,
            "wrist_extension": 0.35,
        }
        self.move_to_pose(pose)
        self.open_grip()
        rospy.sleep(1)
        self.close_grip()

    def subscribe(self):
        self.odom_subscriber = rospy.Subscriber(
            "/odom", Odometry, self.odom_callback
        )  # NOTE: subscribe to odom

        self.joint_states_subscriber = rospy.Subscriber(
            "/stretch/joint_states", JointState, self.joint_states_callback
        )
        self.goal_pos_subscriber = rospy.Subscriber(
            "/hal_prediction_node/goal_pos", String, self.goal_pos_callback
        )
        self.pick_prompt_subscriber = rospy.Subscriber(
            "/stretch/pick_prompt", String, self.pick_prompt_callback
        )
        self.head_image_subscriber = message_filters.Subscriber(
            "/head_camera/color/image_raw",
            Image
            # "/camera2/color/image_raw",
            # Image,
        )
        self.head_image_subscriber.registerCallback(self.head_image_callback)

    @torch.no_grad()
    def main(self, prompt, reset):
        # rospy.init_node("hal_skills_node")
        # self.node_name = rospy.get_name()
        # rospy.loginfo("{0} started".format(self.node_name))
        print("start of main")

        print("switching to position mode")
        s = rospy.ServiceProxy("/switch_to_position_mode", Trigger)
        resp = s()
        print(resp)

        rate = rospy.Rate(self.rate)
        # rospy.sleep(15)
        if self.debug_mode:
            print("\n\n********IN DEBUG MODE**********\n\n")
            cwd = "/home/strech/catkin_ws/src/stretch_ros/stretch_learning"
            csv_path = Path(cwd, "kp.csv")
            img_dir = Path(cwd, "images")
            if not img_dir.exists():
                img_dir.mkdir()
            else:
                for img in img_dir.glob("*.png"):
                    img.unlink()

        # self.close_grip()
        # self.retract_arm_primitive()
        # self.lift_arm_primitive()

        print("start of reset")

        # get hal to starting position for pick
        if reset:
            self.pick_pantry_initial_config(rate)

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
            # base forward
            6: 5,
            # base back
            7: 6,
            # base rotate left
            8: 7,
            # base rotate right
            9: 8,
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
        fixed_goal_pos_tensor = (0, 0, 0)
        use_fixed = False

        prev_inputs_wrist = None
        prev_inputs_head = None
        step = 0
        head_img_raw_copy = copy.deepcopy(self.raw_head_image)
        # head_img_raw_copy = cv2.resize(head_img_raw_copy, (1000, 1000))
        head_rgb_img = copy.deepcopy(Im.fromarray(head_img_raw_copy).convert("RGB"))
        head_img_copy = copy.deepcopy(self.head_image)
        # inputs_head = self.processor(
        #     text=[self.pick_prompt] * 1,
        #     # images=Im.fromarray(self.raw_head_image),
        #     images=[Im.fromarray(head_img_raw_copy)] * 1,
        #     return_tensors="pt",
        # ).to(self.device)
        # head_boxes, head_scores, head_labels = onc.owl_vit(
        #     head_rgb_img, copy.deepcopy(self.pick_prompt)
        # )

        # (
        #     head_new_boxes,
        #     head_boxes_nms,
        #     head_scores,
        #     head_labels,
        # ) = onc.nms_clip(
        #     head_rgb_img,
        #     head_boxes,
        #     head_scores,
        #     head_labels,
        #     copy.deepcopy(self.pick_prompt),
        # )

        joint_pos = self.joint_states.position
        lift_idx, wrist_idx, yaw_idx = (
            self.joint_states.name.index("joint_lift"),
            self.joint_states.name.index("wrist_extension"),
            self.joint_states.name.index("joint_wrist_yaw"),
        )

        js_data = [
            (x, y) for (x, y) in zip(self.joint_states.name, self.joint_states.position)
        ]
        js_data.sort(key=lambda x: x[0])
        js_data = [x[1] for x in js_data]
        js_data = torch.tensor(js_data).unsqueeze(0)

        fixed_goal = None
        total_goal = np.zeros(3)
        goal_count = 0

        self.home_odometry = None

        goal_pred_x = []
        goal_pred_y = []
        goal_pred_z = []

        prev_odom = self.odometry[:2]

        keypressed_index = None
        in_image_x = False

        height_pred_buffers = deque(maxlen=10)
        lower_shelf_preds, top_shelf_preds = 0, 0
        fixed_height = None
        SECOND_SHELF_Z = 0.69
        TOP_SHELF_Z = 0.89

        prev_odom = self.buffer.lookup_transform("odom", "base_link", rospy.Time())

        prev_x = prev_odom.transform.translation.x
        prev_y = prev_odom.transform.translation.y
        prev_z = prev_odom.transform.translation.z

        self.prev_odom = np.array([prev_x, prev_y, prev_z])

        # listener = tf.TransformListener()
        # from_frame_rel = "centered_base_link"
        # to_frame_rel = "link_grasp_center"
        # while step < 500 and not rospy.is_shutdown():
        self.time = time.time()
        while not rospy.is_shutdown():
            # while (
            #     self.goal_pos_pred is None
            #     or self.pick_prompt != prompt
            #     and not rospy.is_shutdown()
            # ):
            #     # TODO: should be cleaner way of doing this with service
            self.pick_prompt_publisher.publish(prompt)
            print(f"{self.goal_pos_pred=}")

            if (
                self.joint_states_data is not None
                and self.goal_pos_pred is not None
                and self.head_image is not None
                and self.odometry is not None
            ):
                # step += 1

                # continue
                # fixed goal pred
                # print(f"FIXED GOAL: {fixed_goal}")
                # curr_odom = self.odometry[:2]
                # change = curr_odom - prev_odom
                # prev_odom = deepcopy(curr_odom)
                # if fixed_goal is not None and in_image_x:
                #     self.goal_pos_pred = deepcopy(fixed_goal)

                #     self.goal_pos_pred[:2] -= change
                #     print("-" * 80)
                # else:
                #     fixed_goal = deepcopy(self.goal_pos_pred)
                #     # fixed_goal[:2] -= self.odometry[:2]

                if (
                    self.joint_states_data is not None
                    and len(self.joint_states_data.size()) <= 1
                ):
                    print("HERE")
                    # print(self.joint_states_data)
                    continue

                # print("-" * 80)

                try:
                    gripper_finger_right = self.buffer.lookup_transform(
                        "base_link", "link_gripper_fingertip_right", rospy.Time()
                    )
                    gripper_finger_left = self.buffer.lookup_transform(
                        "base_link", "link_gripper_fingertip_left", rospy.Time()
                    )
                except (
                    tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException,
                ) as e:
                    print(e)
                    continue

                end_eff_tensor = torch.Tensor(
                    [
                        (
                            gripper_finger_left.transform.translation.x
                            + gripper_finger_right.transform.translation.x
                        )
                        / 2,
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

                # end_eff_tensor[0] -= 0.01
                # end_eff_tensor[1] -= 0.01

                curr_odom = self.buffer.lookup_transform(
                    "odom", "base_link", rospy.Time()
                )

                curr_x = curr_odom.transform.translation.x
                curr_y = curr_odom.transform.translation.y
                curr_z = curr_odom.transform.translation.z

                self.curr_odom = np.array([curr_x, curr_y, curr_z])

                change = self.curr_odom - self.prev_odom
                # print(f"{change=}")
                local_goal_pos = deepcopy(self.goal_pos_pred)

                local_goal_pos = local_goal_pos + change
                # local_goal_pos = deepcopy(fixed_goal)
                # @NAT: add any offsets here
                local_goal_pos[0] = -(local_goal_pos[0])
                local_goal_pos[1] = -(local_goal_pos[1] + 0.03)  # 0.17
                # local_goal_pos[2] += 0.05
                # local_goal_pos[2] = (
                #     SECOND_SHELF_Z if local_goal_pos[-1].item() < 0.78 else TOP_SHELF_Z
                # )

                # if robot predicts 10 top shelf, go top. else go bottom

                # if local_goal_pos[2] == SECOND_SHELF_Z:
                #     height_pred_buffers.append(SECOND_SHELF_Z)
                # elif local_goal_pos[2] == TOP_SHELF_Z:
                #     height_pred_buffers.append(TOP_SHELF_Z)
                # local_goal_pos[2] = self._find_mode(height_pred_buffers)
                # print(height_pred_buffers)
                # print(local_goal_pos[2])

                # if "mustard" in prompt.lower():
                #     local_goal_pos[2] = SECOND_SHELF_Z
                #     # SECOND_SHELF_Z
                # elif "relish" in prompt.lower():
                #     local_goal_pos[2] = TOP_SHELF_Z

                # goal_tensors.append(local_goal_pos)
                # end_eff_tensors.append(end_eff_tensor)

                # print(
                #     f"{trans.transform.translation.x}, {trans.transform.translation.y}, {trans.transform.translation.z}"
                # )

                point = PointStamped()
                point.header.frame_id = "base_link"
                point.point.x = local_goal_pos[0]
                point.point.y = local_goal_pos[1]
                point.point.z = local_goal_pos[2]
                self.goal_point_publisher.publish(point)
                print(f"Pred: {local_goal_pos=}")

                point = PointStamped()
                point.header.frame_id = "base_link"
                point.point.x = end_eff_tensor[0]
                point.point.y = end_eff_tensor[1]
                point.point.z = end_eff_tensor[2]
                self.curr_ee_ps_publisher.publish(point)
                print(f"Curr: {end_eff_tensor=}")

                # print(f"goal tensor:  {local_goal_pos}")
                # print(f"Current EE pos: {end_eff_tensor}")

                # step = 50
                # step += 1
                # if step <= 20:
                #     # drop first 15 to stabilize
                #     continue
                # print(f"{point=}")
                # print()
                # continue
                local_goal_pos[:2] = -local_goal_pos[:2]
                end_eff_tensor[:2] = -end_eff_tensor[:2]
                in_z = np.abs(local_goal_pos[2] - end_eff_tensor[2]) < 0.018
                deep_y = (end_eff_tensor[1] - local_goal_pos[1] >= 0.001) and (
                    end_eff_tensor[1] - local_goal_pos[1]
                ) <= 0.024
                in_y = (
                    np.abs(end_eff_tensor[1] - local_goal_pos[1]) < 0.001
                ) or deep_y  # 0.01
                in_x = np.abs(local_goal_pos[0] - end_eff_tensor[0]) < 0.012  # 0.012
                in_image_x = (
                    np.abs(local_goal_pos[0] - end_eff_tensor[0]) < 0.1
                )  # 0.012
                full_extend = self.joint_states.position[wrist_idx] > 0.20

                joint_eff = self.joint_states.effort
                wrist_yaw_eff_idx = self.joint_states.name.index("joint_wrist_roll")
                wrist_yaw_eff = joint_eff[wrist_yaw_eff_idx]
                yaw_eff = 0
                if yaw_eff > 0 or np.abs(wrist_yaw_eff) > 1e-4:
                    yaw_eff += 1

                print(f"{in_x=}, {in_y=}, {in_z=}, {yaw_eff=}, {full_extend=}")
                # if (in_circle or yaw_eff >= 1) and in_z:
                if (in_z and in_y and in_x) or yaw_eff >= 2:
                    # print(f"{in_circle=}")
                    # print(f"{in_half_circle=}")
                    print("Got to goal!!")
                    self.close_grip()
                    self.lift_arm_primitive()
                    break

                print(f"{end_eff_tensor=}")
                print(f"{local_goal_pos=}")

                inp = local_goal_pos - end_eff_tensor
                # inp[0] = -inp[0]
                # inp[1] = -inp[1]
                # inp = torch.cat((inp,torch.tensor([self.odometry[2]])),dim=0)

                inp = inp.to(torch.float32)

                start = time.time()
                print(f"Input to model {inp}")
                prediction = self.model_ppo(inp)
                print(f"{prediction=}")
                print(f"ppo: {time.time() - start}")
                times.append(time.time() - start)

                keypressed_index = torch.argmax(prediction).item()
                keypressed = kp_reduced_mapping[keypressed_index]
                keypressed = self.index_to_keypressed(keypressed)

                _pos = (local_goal_pos - end_eff_tensor).cpu().numpy()
                print(f"pos: {_pos}")
                onpolicy_pos.append(_pos)
                onpolicy_kp.append(keypressed_index)

                if keypressed == "_":
                    # noop
                    print("NOOP")
                    continue

                command = self.get_command(keypressed)

                print(f"{rospy.Time().now()}, {keypressed_index=}, {command=}")
                self.send_command(command)

            if keypressed_index is not None:
                if keypressed_index > 5:
                    # rate.sleep()
                    rospy.sleep(2)
                else:
                    rate.sleep()
            else:
                rate.sleep()
            print(f"time of loop = {time.time() - self.time}")
            self.time = time.time()

        # publish empty prompt to stop
        # pick_prompt_msg.pick_prompt = None
        # pick_prompt_msg.do_pick = False
        self.pick_prompt_publisher.publish(prompt)

        self.prev_odom = deepcopy(self.curr_odom)


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


if __name__ == "__main__":
    node = HalSkillsNode()
    # node.start()
    node.main()
