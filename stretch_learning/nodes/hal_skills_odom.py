#!/usr/bin/env python3

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
from ppo import MLP
import sys

from std_srvs.srv import Trigger

import hal_controller_full_train
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO


def load_ppo_model(pth_path):
    model = MLP(**ppo_utils.config)
    model = ppo_utils.load_pth_file_to_model(model, pth_path)
    return model


from transformers import OwlViTProcessor

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
tokenizer = open_clip.get_tokenizer("EVA02-B-16")

# BC imports
from r3m import load_r3m
from bc.owlvit_model_v3 import BC

# from bc.model_bc_trained import BC as BC_Trained

# IQL imports
from iql.img_js_net import ImageJointStateNet
from iql.iql import load_iql_trainer

from stretch_learning.srv import Pick, PickResponse

# simulation import
import simulate

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

        self.skill_name = skill_name
        self.model_type = model_type
        self.train_type = train_type

        print("\n\nskill_name: ", self.skill_name)
        print("model_type: ", self.model_type)
        print("train_type: ", self.train_type)
        print()

        ckpt_dir = Path(
            f"~/catkin_ws/src/stretch_ros/stretch_learning/checkpoints",
            self.skill_name,
            self.model_type,
            self.train_type,
        ).expanduser()

        # _id = "c56ea4ee"
        # ckpt_dir = Path(
        #     f"~/catkin_ws/src/stretch_ros/stretch_learning/checkpoints",
        #     self.skill_name,
        #     self.model_type,
        #     f"{self.train_type}_{_id}",
        # ).expanduser()

        if self.model_type == "visuomotor_bc":
            self.model = self.load_bc_model(ckpt_dir)
        elif self.model_type == "irl" and self.train_type == "iql":
            self.model = self.load_iql_model(ckpt_dir)

        self.goal_pos = list(map(float, goal_pos))
        self.goal_tensor = torch.Tensor(self.goal_pos).to(device)

        self.goal_tensor[3:5] = self.odom_to_js(self.goal_tensor[3:5])

        pth_path = "/home/strech/catkin_ws/src/stretch_ros/stretch_learning/checkpoints/ppo_point_and_shoot/policy.pth"
        self.model_ppo = load_ppo_model(pth_path)
        self.model_ppo.eval()

        self.init_node()  # comment out when using move

    def load_iql_model(self, ckpt_dir):
        img_comp_dims = 32
        joint_pos, joint_vel, joint_force = True, True, True
        img_js_net = ImageJointStateNet(
            img_comp_dims, joint_pos, joint_vel, joint_force
        )

        state_dim, action_dim = 32 + 14 * 3 + 3, 17
        max_action = 1
        n_hidden = 3
        iql_deterministic = True
        model = load_iql_trainer(
            device,
            iql_deterministic,
            state_dim,
            action_dim,
            max_action,
            n_hidden,
            img_js_net,
        )

        ckpts = [
            ckpt for ckpt in Path(ckpt_dir).glob("*.pt") if "last" not in ckpt.stem
        ]
        ckpts.sort(key=lambda x: float(x.stem.split("_", 3)[2]))
        ckpt_path = ckpts[0]
        ckpt_path = "/home/strech/catkin_ws/src/stretch_ros/stretch_learning/checkpoints/pick_pepper_salt/d745bda3_epoch=100_val_loss=0.000173.pt"
        print(f"Loading checkpoint from {str(ckpt_path)}.\n")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        state_dict = torch.load(
            "/home/strech/catkin_ws/src/stretch_ros/stretch_learning/checkpoints/pick_pantry/visuomotor_bc/bc_oracle/epoch=400_success=1.000.pt",
            map_location=torch.device("cpu"),
        )
        for key in list(state_dict.keys()):
            state_dict[key.replace("fc.", "")] = state_dict.pop(key)
        model.fc_last.load_state_dict(state_dict, strict=False)

        model.img_js_net.eval()
        model.actor.eval()
        return model

    def load_bc_model(self, ckpt_dir):
        # print(ckpt_dir)
        # ckpts = [ckpt for ckpt in ckpt_dir.glob("*.pt") if "last" not in ckpt.stem]
        # ckpts += [ckpt for ckpt in ckpt_dir.glob("*.ckpt") if "last" not in ckpt.stem]
        # # import pdb; pdb.set_trace()
        # if "val_acc" in str(ckpts[0]):
        #     ckpts.sort(key=lambda x: float(x.stem.split("val_acc=")[1]), reverse=True)
        # else:
        #     ckpts.sort(
        #         key=lambda x: float(x.stem.split("combined_acc=")[1]), reverse=True
        #     )

        # ckpt_path = ckpts[-4]
        # ckpt_path = Path(ckpt_dir, "last.ckpt")
        # print(f"Loading checkpoint from {str(ckpt_path)}.\n")

        # state_dict = torch.load(ckpt_path, map_location=device)
        # modified_dict = {}
        # for key, value in state_dict.items():
        #     key = key.replace("_orig_mod.", "")
        #     modified_dict[key] = value
        # model.load_state_dict(modified_dict)

        # ckpt_dir = Path("/home/strech/catkin_ws/src/stretch_ros/stretch_learning/checkpoints/point_shoot/ckpts")
        # ckpts = [ckpt for ckpt in ckpt_dir.glob("*.pt")]
        # ckpt_path = ckpts[-1]
        ckpt_path = Path(
            "/home/strech/catkin_ws/src/stretch_ros/stretch_learning/checkpoints/pick_pepper_salt/f3bfc93c_epoch=130_val_loss=0.000744.pt"
        )

        model = BC(num_classes=4, device="cpu")

        print(f"Loading checkpoint from {str(ckpt_path)}.\n")
        state_dict = torch.load(ckpt_path, map_location=device)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(missing_keys)
        print(unexpected_keys)

        # state_dict = torch.load(
        #     "/home/strech/catkin_ws/src/stretch_ros/stretch_learning/checkpoints/pick_pantry/visuomotor_bc/bc_oracle/epoch=400_success=1.000.pt",
        #     map_location=torch.device("cpu"),
        # )
        # for key in list(state_dict.keys()):
        #     state_dict[key.replace("fc.", "")] = state_dict.pop(key)
        # model.fc_last.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        return model

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

    def wrist_image_callback(self, ros_rgb_image):
        try:
            self.raw_wrist_image = self.cv_bridge.imgmsg_to_cv2(ros_rgb_image, "bgr8")
            self.wrist_image = self.img_transform(self.raw_wrist_image)
            self.wrist_image = self.wrist_image.unsqueeze(0)
        except CvBridgeError as error:
            print(error)

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
            "joint_lift": self.pick_starting_height - 0.55,  # for cabinet -0.175
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

    def odom_to_js(self, odom_data):
        """
        Takes in odom data [x, y, theta] from /odom topic and flips the x and y
        """
        odom_data[0] = odom_data[0] * -1
        odom_data[1] = odom_data[1] * -1
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
    def main(self):
        # rospy.init_node("hal_skills_node")
        # self.node_name = rospy.get_name()
        # rospy.loginfo("{0} started".format(self.node_name))
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
        self.wrist_image_subscriber = message_filters.Subscriber(
            "/wrist_camera/color/image_raw", Image
        )
        self.wrist_image_subscriber.registerCallback(self.wrist_image_callback)
        self.head_image_subscriber = message_filters.Subscriber(
            "/head_camera/color/image_raw", Image
        )
        self.head_image_subscriber.registerCallback(self.head_image_callback)

        rate = rospy.Rate(self.rate)

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

        # self.grasp_primitive()
        # self.retract_arm_primitive()
        # self.lift_arm_primitive()

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

        goal_pos_tensor = torch.Tensor(self.end_eff_to_xyz(self.goal_tensor))

        print(f"Goal XYZ: {goal_pos_tensor}")
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

        end_eff_tensor = torch.Tensor(
            self.end_eff_to_xyz(
                [
                    joint_pos[wrist_idx],
                    joint_pos[yaw_idx],
                    joint_pos[lift_idx],
                    self.odometry[0],
                    self.odometry[1],
                    self.odometry[2],
                ]
            )
        )

        _pos = (goal_pos_tensor - end_eff_tensor).cpu().numpy()
        print(f"delta_pos: {_pos}")
        onpolicy_pos.append(_pos)

        while step < 500:
            if (
                self.joint_states is not None
                # and self.wrist_image is not None
                # and self.head_image is not None
                and self.odometry is not None
            ):
                # check delta to determine skill termination
                if "pick" in self.skill_name and self.check_pick_termination():
                    rospy.loginfo("\n\n***********Pick Completed***********\n\n")
                    self.action_status = SUCCESS
                    print(times)
                    return 1
                elif "place" in self.skill_name and self.check_place_termination():
                    rospy.loginfo("\n\n***********Place Completed***********\n\n")
                    self.action_status = SUCCESS
                    return 1

                # if not, continue with next command
                if len(self.joint_states_data.size()) <= 1:
                    print(self.joint_states_data)
                    continue

                # print(f"joint state shape:  {self.joint_states_data.shape}")

                print("-" * 80)
                # print(f"Current goal position: {self.goal_tensor}")

                # create current end-effector

                joint_pos = self.joint_states.position
                lift_idx, wrist_idx, yaw_idx = (
                    self.joint_states.name.index("joint_lift"),
                    self.joint_states.name.index("wrist_extension"),
                    self.joint_states.name.index("joint_wrist_yaw"),
                )

                js_data = [
                    (x, y)
                    for (x, y) in zip(
                        self.joint_states.name, self.joint_states.position
                    )
                ]
                js_data.sort(key=lambda x: x[0])
                js_data = [x[1] for x in js_data]
                js_data = torch.tensor(js_data).unsqueeze(0)

                end_eff_tensor = torch.Tensor(
                    self.end_eff_to_xyz(
                        [
                            joint_pos[wrist_idx],
                            joint_pos[yaw_idx],
                            joint_pos[lift_idx],
                            self.odometry[0],
                            self.odometry[1],
                            self.odometry[2],
                        ]
                    )
                )

                # Wait for a keypress
                # key = cv2.waitKey(0)
                # print(key)
                # if key == ord("s"):
                #     mode = True
                #     print("salt")
                # elif key == ord("p"):
                #     mode = False
                #     print("pepper")
                # if mode:
                #     text = ["salt"]
                #     ref_text_tokenized = tokenizer(text)

                # else:
                #     text = ["pepper"]
                #     ref_text_tokenized = tokenizer(text)
                # print(33333333333)
                # self.raw_head_image = np.zeros_like(self.raw_head_image)
                # self.head_image = torch.zeros_like(self.head_image)
                # inputs_head = processor(
                #     text=text,
                #     images=Im.fromarray(self.raw_head_image),
                #     return_tensors="pt",
                # ).to("cpu")

                # self.raw_wrist_image = np.zeros_like(self.raw_wrist_image)
                # self.wrist_image = torch.zeros_like(self.wrist_image)
                # inputs_wrist = processor(
                #     text=text,
                #     images=Im.fromarray(self.raw_wrist_image),
                #     return_tensors="pt",
                # ).to("cpu")

                # start = time.time()
                # if prev_inputs_wrist:
                #     print(
                #         f"{torch.norm(prev_inputs_wrist['pixel_values']-inputs_wrist['pixel_values'])=}"
                #     )
                # prev_inputs_wrist = copy.deepcopy(inputs_wrist)
                # print(f"{inputs_head=}")
                # print(f"{inputs_wrist=}")
                # with torch.no_grad(), torch.cuda.amp.autocast_mode.autocast():
                #     _, self.goal_pos = self.model.forward_special(
                #         inputs_head,
                #         inputs_wrist,
                #         self.wrist_image,
                #         self.head_image,
                #         ref_text_tokenized,
                #         js_data,
                #     )
                # print(time.time() - start)

                # goal_prediction = self.model.forward_special(
                #     inputs_head, self.head_image, ref_text_tokenized
                # )
                # goal_pos_tensor = torch.Tensor(self.end_eff_to_xy(*self.goal_pos))
                # goal_pos_tensor = self.goal_pos[0]
                # print(goal_pos_tensor.shape)
                # print(end_eff_tensor.shape)
                ###############
                # plt.xlim(-1,1)
                # plt.ylim(-1,1)
                # goal_tensors.append(goal_pos_tensor)
                # goal_pos_tensor[1] = 0.47

                # print(end_eff_tensor)
                # plt.plot(goal_pos_tensor[0],goal_pos_tensor[1],'ro')
                # plt.plot(end_eff_tensor[0],end_eff_tensor[1],'go')
                # goal_ext = 0.4
                # goal_yaw = 0.52
                # gx,gy = self.end_eff_to_xy(goal_ext,goal_yaw)
                # print(gx,gy)
                # plt.plot(gx,gy, color="yellow", marker="*",markersize=14)

                # plt.savefig("yolo.png")
                # continue
                ######################

                goal_pos_tensor = torch.Tensor(self.end_eff_to_xyz(self.goal_tensor))

                goal_tensors.append(goal_pos_tensor)
                end_eff_tensors.append(end_eff_tensor)

                print(f"goal tensor:  {goal_pos_tensor}")
                print(f"Current EE pos: {end_eff_tensor}")

                # pred_gy = goal_pos_tensor[1]
                # curr_gy = end_eff_tensor[1]
                # if use_fixed == False:
                #     fixed_goal_pos_tensor = goal_pos_tensor
                #     use_fixed = True
                #     print("Using Fixed")
                if initial_pos is None:
                    initial_pos = [
                        joint_pos[wrist_idx],
                        joint_pos[yaw_idx],
                        joint_pos[lift_idx],
                        self.odometry[0],
                        self.odometry[1],
                        self.odometry[2],
                    ]

                in_z = np.abs(goal_pos_tensor[2] - end_eff_tensor[2]) < 0.025
                in_circle = (
                    np.linalg.norm(goal_pos_tensor[:2] - end_eff_tensor[:2]) < 0.02
                )
                in_half_circle = self.is_point_in_half_circle(
                    self.odometry[2], goal_pos_tensor[:2], 0.025, end_eff_tensor[:2]
                )
                if (in_circle or in_half_circle) and in_z:
                    print(in_circle)
                    print(in_half_circle)
                    print("Got to goal!!")
                    self.grasp_primitive()
                    break

                # if torch.norm(goal_pos_tensor[:2] - end_eff_tensor[:2]) < 0.03:
                #     print("Got to goal!!")
                #     self.grasp_primitive()
                #     # self.lift_arm_primitive()
                #     # self.retract_arm_primitive()
                #     break
                # if self.is_2d:
                #     if use_fixed:
                #         inp = fixed_goal_pos_tensor - end_eff_tensor
                #     else:
                #         inp = torch.cat(
                #             (
                #                 goal_pos_tensor - end_eff_tensor,
                #                 torch.tensor(0).unsqueeze(0),
                #             ),
                #             dim=-1,
                #         )
                # elif self.use_delta:
                #     inp = torch.cat((end_eff_tensor, goal_pos_tensor - end_eff_tensor))
                # else:
                #     inp = torch.cat((end_eff_tensor, self.goal_tensor))
                inp = goal_pos_tensor - end_eff_tensor
                inp = inp.unsqueeze(0)
                # self.joint_states_data = torch.cat((self.joint_states_data, args.user_coordinate
                start = time.time()

                print(f"Input to model {inp}")
                prediction = self.model_ppo(inp)
                print(f"{prediction=}")

                # times.append(time.time() - start)
                print(time.time() - start)

                # prediction = torch.nn.functional.softmax(prediction).flatten()
                # dist = torch.distributions.Categorical(prediction)
                # keypressed_index = dist.sample().item()
                # keypressed = self.index_to_keypressed(keypressed_index)

                # best3 = torch.argsort(prediction).flatten().flip(0)[:3]
                # probs = torch.nn.functional.softmax(prediction).flatten()

                # print(f"PREDICTION: {INDEX_TO_KEYPRESSED[keypressed_index]}")
                # print("Best 3")
                # for i in range(3):
                #     print(f"Prediction #{i+1}: {INDEX_TO_KEYPRESSED[best3[i].item()]}, {probs[best3[i]]}")
                # print("-"*50)

                keypressed_index = torch.argmax(prediction).item()
                keypressed = kp_reduced_mapping[keypressed_index]
                keypressed = self.index_to_keypressed(keypressed)

                # _pos = self.end_eff_to_xy(
                #     (self.goal_tensor - end_eff_tensor).cpu().numpy()
                # )
                _pos = (goal_pos_tensor - end_eff_tensor).cpu().numpy()

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
                time_start = time.time()
                self.send_command(command)
                print("command time: " + str(time.time() - time_start))
                step += 1
                # time.sleep(1)

            rate.sleep()
        timestr = time.strftime("%Y%m%d-%H%M%S")
        if self.is_2d:
            title = f"{timestr}_2d"
        elif self.use_delta:
            title = f"{timestr}_use_delta"
        else:
            title = f"{timestr}_no_delta"

        fig = plt.figure()
        ax = plt.axes(xlim=(-1, 1), ylim=(-1, 1))

        # def animate(i):
        #     ax.clear()  # clear axes
        #     ax.set_xlim(-0.05, 0.05)
        #     ax.set_ylim(0.4, 0.6)

        #     ax.plot(goal_tensors[i][0], goal_tensors[i][1], "ro")
        #     # ax.plot(end_eff_tensors[i][0], end_eff_tensors[i][1], 'go')

        #     return ax

        # anim = animation.FuncAnimation(
        #     fig, animate, frames=len(goal_tensors), interval=1000
        # )
        # anim.save("points.gif", writer="imagemagick", fps=10)
        # clip = VideoFileClip("points.gif")
        # # Save the resulting MP4 file
        # clip.write_videofile("points.mp4", fps=10)
        # for i in range(len(goal_tensors)):
        #     plt.plot([goal_tensors[i][0], goal_tensors[i][1]], color="red")
        #     plt.plot([end_eff_tensors[i][0], end_eff_tensors[i][1]], color="green")
        #     plt.show()
        # start_ext = initial_pos[0]
        # start_yaw = initial_pos[1]
        # goal_ext = self.goal_pos[0][0]
        # goal_yaw = self.goal_pos[0][1]
        # ckpt_path = "/home/strech/catkin_ws/src/stretch_ros/stretch_learning/checkpoints/pick_pepper_salt/d745bda3_epoch=100_val_loss=0.000173.pt"
        # save_fig_path = "/home/strech/catkin_ws/src/stretch_ros/stretch_learning/nodes/plots/base_ppo"
        # sim_x, sim_y, sim_onpolicy_kp = simulate.run_simulate(
        #     start_ext, start_yaw, goal_ext, goal_yaw, ckpt_path, save_fig_path
        # )

        # hyperparameters = {
        #     "gamma": (1 - 0.0763619061360584),
        #     "max_grad_norm": 0.7876864744938158,
        #     "n_steps": 128,
        #     "learning_rate": 0.00021175129610218643,
        #     "ent_coef": 0.08297853628113681,
        #     "gae_lambda": (1 - 0.005645587760538626),
        #     "policy_kwargs": {
        #         # "net_arch": {"pi": [64, 64], "vf": [64, 64], "activation_fn":'relu'}
        #         # "net_arch": {"pi": [64], "vf": [64]}
        #         "net_arch": {
        #             "pi": [100, 100, 100],
        #             "vf": [100, 100, 100],
        #             "activation_fn": "relu",
        #         }
        #         # "net_arch": {"pi": [100, 100, 100, 100], "vf": [100, 100, 100, 100], "activation_fn":'tanh'}
        #     },
        # }

        # num_cpu = 8
        # env_id = "HalControllerEnv"
        # vec_env = make_vec_env(env_id, n_envs=num_cpu)
        # model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda", **hyperparameters)
        # model.load(
        #     "/home/strech/catkin_ws/src/stretch_ros/stretch_learning/checkpoints/ppo_point_and_shoot/ppo_full_2600_small_relu.zip"
        # )
        env = hal_controller_full_train.HalControllerEnv(max_steps=step)
        print(initial_pos)
        success_rate, state_list, action_list, _ = hal_controller_full_train.evaluate(
            self.model_ppo,
            env,
            start=initial_pos,
            goal=self.goal_tensor,
        )
        state_list = np.array(state_list)
        print(f"onpolicy initial:  {onpolicy_pos[0]}")
        self.overlay_plot(
            state_list,
            action_list,
            onpolicy_kp,
            onpolicy_pos,
            "ppo_rotate_1500_small_tanh",
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

    def plot_traj_no_save(
        self, kp_mapping, state_list, action_list, goal, s, alpha, title
    ):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.grid()

        st = np.array(state_list)
        plot_x = st[:, 0]
        plot_y = st[:, 1]
        plot_z = st[:, 2]
        sim_scatter = ax.scatter(
            plot_x[:-1],
            plot_y[:-1],
            plot_z[:-1],
            s=s,
            c=action_list,
            cmap="viridis",
            alpha=alpha,
        )
        handles, _ = sim_scatter.legend_elements()

        filtered_kp_mapping = [kp_mapping[i] for i in np.unique(action_list)]
        plt.legend(handles, filtered_kp_mapping, title="Key Presses")

        # Plot the start
        plt.plot(
            [plot_x[0]],
            [plot_y[0]],
            [plot_z[0]],
            marker=".",
            markersize=15,
            color="pink",
            label="start",
        )

        # Plot the goal
        plt.plot(
            [goal[0]],
            [goal[1]],
            [goal[2]],
            marker="*",
            markersize=15,
            color="red",
            label="goal",
        )

        # Plot the end of trajectory with no action
        plt.plot(
            [plot_x[-1]],
            [plot_y[-1]],
            [plot_z[-1]],
            marker=".",
            markersize=5,
            color="orange",
            label="end",
        )

        ax.set_xlim(-0.25, 0.25)
        ax.set_ylim(-0.68, 0.68)
        ax.set_zlim(-1, 1)
        plt.title("HAL Controller Sim with PPO")

        save_dir = "/home/strech/catkin_ws/src/stretch_ros/stretch_learning/nodes/plots/base_ppo"
        save_path = Path(save_dir, f"{title.replace(' ', '_')}.png")
        save_path.parent.mkdir(exist_ok=True)
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    def overlay_plot(
        self,
        sim_states,
        sim_actions,
        onpolicy_kp,
        pts,
        title,
        save_dir="temp",
    ):
        import matplotlib

        print(f"sim actions: {sim_actions}")
        matplotlib.use("Agg")
        sim_kp_mapping = [
            "Sim Arm out",
            "Sim Arm in",
            "Sim Gripper right",
            "Sim Gripper left",
            "Sim Lift Up",
            "Sim Lift Down",
            "Sim Base Left",
            "Sim Base Right",
            "Sim Base Rotate Left",
            "Sim Base Rotate Right",
        ]

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.grid()

        st = np.array(sim_states)
        plot_x = st[:, 0]
        plot_y = st[:, 1]
        plot_z = st[:, 2]
        sim_scatter = ax.scatter(
            plot_x[:-1],
            plot_y[:-1],
            plot_z[:-1],
            s=2,
            c=sim_actions,
            cmap="viridis",
            alpha=1,
        )
        handles, _ = sim_scatter.legend_elements()

        # Plot the start
        plt.plot(
            [plot_x[0]],
            [plot_y[0]],
            [plot_z[0]],
            marker=".",
            markersize=15,
            color="pink",
            label="start",
        )

        # Plot the goal
        plt.plot(
            [0],
            [0],
            [0],
            marker="*",
            markersize=15,
            color="red",
            label="goal",
        )

        # Plot the end of trajectory with no action
        plt.plot(
            [plot_x[-1]],
            [plot_y[-1]],
            [plot_z[-1]],
            marker=".",
            markersize=5,
            color="orange",
            label="end",
        )

        # on policy plots
        st = np.array(pts)
        plot_x = st[:, 0]
        plot_y = st[:, 1]
        plot_z = st[:, 2]
        sim_scatter = ax.scatter(
            plot_x[:-1],
            plot_y[:-1],
            plot_z[:-1],
            s=1,
            c=onpolicy_kp,
            cmap="viridis",
            alpha=0.3,
        )
        handles2, _ = sim_scatter.legend_elements()

        # Plot the start
        plt.plot(
            [plot_x[0]],
            [plot_y[0]],
            [plot_z[0]],
            marker=".",
            markersize=15,
            color="pink",
            label="start",
        )

        # Plot the goal
        plt.plot(
            [0],
            [0],
            [0],
            marker="*",
            markersize=15,
            color="red",
            label="goal",
        )

        # Plot the end of trajectory with no action
        plt.plot(
            [plot_x[-1]],
            [plot_y[-1]],
            [plot_z[-1]],
            marker=".",
            markersize=5,
            color="orange",
            label="end",
        )

        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)

        plt.legend(
            handles2 + handles,
            [kp_mapping[i] for i in np.unique(onpolicy_kp)]
            + [sim_kp_mapping[i] for i in np.unique(sim_actions)],
            title="Classes",
        )

        save_dir = "/home/strech/catkin_ws/src/stretch_ros/stretch_learning/nodes/plots/base_ppo"
        save_path = Path(save_dir, f"{title.replace(' ', '_')}.png")
        save_path.parent.mkdir(exist_ok=True)
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
        # with open(f"{title}.pickle", "wb") as pickle_file:
        #     pl.dump(fig_handle, pickle_file)

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
