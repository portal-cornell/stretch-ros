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
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryGoal, FollowJointTrajectoryAction
from custom_msg_python.msg import Keypressed
import hello_helpers.hello_misc as hm

import torch
import numpy as np
import scipy
from torchvision import transforms
from cv_bridge import CvBridge, CvBridgeError

# BC imports
from r3m import load_r3m
from bc.seq_model import BC_Seq

# from bc.model_bc_trained import BC as BC_Trained

# IQL imports
from iql.img_js_net import ImageJointStateNet
from iql.iql import load_iql_trainer

from stretch_learning.srv import Pick, PickResponse

device = "cuda" if torch.cuda.is_available() else "cpu"

RUNNING = -1
SUCCESS = 1
NOT_STARTED = 2

joint_labels = [
    "gripper_aperture_pos",
    "gripper_aperture_vel",
    "gripper_aperture_eff",
    "joint_arm_l0_pos",
    "joint_arm_l0_vel",
    "joint_arm_l0_eff",
    "joint_arm_l1_pos",
    "joint_arm_l1_vel",
    "joint_arm_l1_eff",
    "joint_arm_l2_pos",
    "joint_arm_l2_vel",
    "joint_arm_l2_eff",
    "joint_arm_l3_pos",
    "joint_arm_l3_vel",
    "joint_arm_l3_eff",
    "joint_gripper_finger_left_pos",
    "joint_gripper_finger_left_vel",
    "joint_gripper_finger_left_eff",
    "joint_gripper_finger_right_pos",
    "joint_gripper_finger_right_vel",
    "joint_gripper_finger_right_eff",
    "joint_head_pan_pos",
    "joint_head_pan_vel",
    "joint_head_pan_eff",
    "joint_head_tilt_pos",
    "joint_head_tilt_vel",
    "joint_head_tilt_eff",
    "joint_lift_pos",
    "joint_lift_vel",
    "joint_lift_eff",
    "joint_wrist_pitch_pos",
    "joint_wrist_pitch_vel",
    "joint_wrist_pitch_eff",
    "joint_wrist_roll_pos",
    "joint_wrist_roll_vel",
    "joint_wrist_roll_eff",
    "joint_wrist_yaw_pos",
    "joint_wrist_yaw_vel",
    "joint_wrist_yaw_eff",
    "wrist_extension_pos",
    "wrist_extension_vel",
    "wrist_extension_eff",
]

class HalSkills(hm.HelloNode):
    def __init__(self, skill_name, model_type, train_type, goal_pos, a_hor):
        hm.HelloNode.__init__(self)
        self.debug_mode = False
        self.rate = 10.0
        self.trajectory_client = actionlib.SimpleActionClient(
            "/stretch_controller/follow_joint_trajectory", FollowJointTrajectoryAction
        )

        self.a_hor = a_hor
        self.step_size = "medium"
        self.rad_per_deg = math.pi / 180.0
        self.small_deg = 3.0
        self.small_rad = self.rad_per_deg * self.small_deg
        self.small_translate = 0.005  # 0.02
        self.medium_deg = 6.0
        self.medium_rad = self.rad_per_deg * self.medium_deg
        self.medium_translate = 0.04
        self.mode = "navigation"

        self.joint_states = None
        self.wrist_image = torch.empty(0)
        self.head_image = torch.empty(0)
        self.joint_states_data = torch.empty(0)
        self.cv_bridge = CvBridge()

        self.img_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
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


        if self.model_type == "visuomotor_bc":
            self.model = self.load_bc_model(ckpt_dir)


        goal_pos = list(map(float, goal_pos))
        self.goal_tensor = torch.Tensor(goal_pos).to(device)

        self.init_node()

    def load_bc_model(self, ckpt_dir):
        ckpt_path = Path("/home/strech/catkin_ws/src/stretch_ros/stretch_learning/checkpoints/point_shoot/20230826-222805/epoch=830_val_loss=0.276.pt")
        print(f"Loading {ckpt_path.stem}")
        
        model = BC_Seq(action_horizon=self.a_hor, loss_fn=None, accuracy=None)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()

        return model

    def init_node(self):
        rospy.init_node("hal_skills_node")
        self.node_name = rospy.get_name()
        rospy.loginfo("{0} started".format(self.node_name))

    def update_horizon(self, horizon, update):
        update = update.squeeze(0).unsqueeze(0)
        if len(horizon) == args.o_hor:
            horizon = horizon[1:]
        elif len(horizon) == 0:
            horizon = update
            return horizon
        horizon = torch.cat((horizon, update), axis=0)
        return horizon

    def joint_states_callback(self, msg):
        self.joint_states = msg
        js_msg = list(zip(msg.name, msg.position, msg.velocity, msg.effort))
        js_msg = sorted(js_msg, key=lambda x: x[0])
        js_arr = []
        for idx, (name, pos, vel, eff) in enumerate(js_msg):
            js_arr.extend([pos, vel, eff])
        joint_states_data = torch.from_numpy(np.array(js_arr, dtype=np.float32))
        if len(joint_states_data) <= 1:
            joint_states_data = joint_states_data.unsqueeze(0)
        self.joint_states_data = self.update_horizon(self.joint_states_data, joint_states_data)

    def wrist_image_callback(self, ros_rgb_image):
        try:
            self.raw_wrist_image = self.cv_bridge.imgmsg_to_cv2(ros_rgb_image, "bgr8")
            wrist_image = self.img_transform(self.raw_wrist_image)
            wrist_image = wrist_image.unsqueeze(0)
        except CvBridgeError as error:
            print(error)
        self.wrist_image = self.update_horizon(self.wrist_image, wrist_image)

    def head_image_callback(self, ros_rgb_image):
        try:
            self.raw_head_image = self.cv_bridge.imgmsg_to_cv2(ros_rgb_image, "bgr8")
            head_image = self.img_transform(self.raw_head_image)
            head_image = head_image.unsqueeze(0)
        except CvBridgeError as error:
            print(error)
        self.head_image = self.update_horizon(self.head_image, head_image)

    def get_deltas(self):
        if self.step_size == "small":
            deltas = {"rad": self.small_rad, "translate": self.small_translate}
        if self.step_size == "medium":
            deltas = {"rad": self.medium_rad, "translate": self.medium_translate}
        if self.step_size == "big":
            deltas = {"rad": self.big_rad, "translate": self.big_translate}
        return deltas

    def get_command(self, delta_js):
        """"""
        keypressed_publisher = rospy.Publisher("key_pressed", Keypressed, queue_size=20)
        rospy.Rate(1)
        msg = Keypressed()
        msg.timestamp = (str)(rospy.get_rostime().to_nsec())

        # # 8 or up arrow
        # if c == "8" or c == "\x1b[A":
        #     command = {"joint": "joint_lift", "delta": self.get_deltas()["translate"]}
        #     msg.keypressed = "8"
        #     keypressed_publisher.publish(msg)
        
        filter = ["joint_lift_pos", "wrist_extension_pos", "joint_wrist_yaw_pos"]
        command = {}
        for i in range(len(filter)-1, -1, -1):
            joint_name = filter[i]
            print(f"{joint_name}: {delta_js[i]}")
            if abs(delta_js[i]) > 1e-5:
                command[joint_name] = delta_js[i]

        ####################################################

        return command

    def send_command(self, command):
        joint_state = self.joint_states
        if (joint_state is not None) and (command is not None):
            point = JointTrajectoryPoint()
            point.time_from_start = rospy.Duration(0.0)
            trajectory_goal = FollowJointTrajectoryGoal()
            # trajectory_goal.goal_time_tolerance = rospy.Time(1.0)

            trajectory_goal.trajectory.joint_names = []
            trajectory_goal.trajectory.points = []
            print(f"{command=}")
            for joint_name in command.keys():
                delta = command[joint_name]
                if "_pos" not in joint_name or delta < 1e-3:
                    continue
                trajectory_goal.trajectory.joint_names.append(joint_name[:-4])

                joint_index = joint_state.name.index(joint_name[:-4])
                joint_value = joint_state.position[joint_index]
                delta = command[joint_name]
                new_value = joint_value + delta
                
                point.positions = [new_value]
                trajectory_goal.trajectory.points.append(point)
                self.trajectory_client.send_goal(trajectory_goal)
                trajectory_goal = FollowJointTrajectoryGoal()
                print("HERE as wel?")


        trajectory_goal.trajectory.header.stamp = rospy.Time.now()

    # -----------------pick_pantry() initial configs-----------------#
    def move_arm_pick_pantry(self):
        rospy.loginfo("Set arm")
        self.pick_starting_height = 0.968
        self.joint_lift_index = self.joint_states.name.index("joint_lift")
        pose = {
            "wrist_extension": 0.01,
            "joint_lift": self.pick_starting_height,  # for cabinet
            "joint_wrist_pitch": 0.2,
            "joint_wrist_yaw": -0.09,
        }
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
            # "joint_wrist_yaw": -0.09,
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
        curr_height = self.joint_states.position[self.joint_lift_index]
        if curr_height - self.pick_starting_height > 0.07:
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

    @torch.no_grad()
    def main(self):
        # rospy.init_node("hal_skills_node")
        # self.node_name = rospy.get_name()
        # rospy.loginfo("{0} started".format(self.node_name))
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
        print("start of reset")
        # get hal to starting position for pick
        if "pick" in self.skill_name or "point" in self.skill_name:
            self.pick_pantry_initial_config(rate)
        elif self.skill_name == "place_table":
            self.place_table_initial_config(rate)
        elif self.skill_name == "open_drawer":
            self.open_drawer_initial_config(rate)
        else:
            raise NotImplementedError
        print("start of loop")
        img_count = 0
        times = []
        prediction = torch.empty(0,4)

        while not rospy.is_shutdown():
            # print("not shutdown")
            # print("js ===== ", self.joint_states_data.shape)
            # print("wrist image ===", self.wrist_image.shape)
            # print("head image ===", self.head_image.shape)
            if self.joint_states is not None:
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

                print("-"*80)
                print(f"Current goal position: {self.goal_tensor}")

                # create current end-effector
                joint_pos = self.joint_states.position
                lift_idx, wrist_idx, yaw_idx = self.joint_states.name.index("joint_lift"), self.joint_states.name.index("wrist_extension"), self.joint_states.name.index("joint_wrist_yaw")
                end_eff_tensor = torch.Tensor([joint_pos[lift_idx], joint_pos[wrist_idx], joint_pos[yaw_idx]])
                print(f"Current EE pos: {end_eff_tensor}")
                inp = torch.cat((end_eff_tensor, self.goal_tensor)).unsqueeze(0)

                start = time.time()
                prediction = self.model(inp).squeeze(0)
                prediction = prediction.view((self.a_hor, -1))
                prediction = torch.sum(prediction, dim=0).numpy()
                times.append(time.time() - start)
            
                command = self.get_command(prediction)
                prediction = prediction[1:]
                print(command)
                self.send_command(command)
            rate.sleep()


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
        "point_shoot"
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
        "--skill_name", type=str, choices=supported_skills, default="pick_pantry"
    )
    parser.add_argument(
        "--model_type", type=str, choices=supported_models, default="visuomotor_bc"
    )
    parser.add_argument(
        "--train_type", type=str, choices=supported_types, default="bc_oracle"
    )
    parser.add_argument(
        "--o_hor", type=int, default=1
    )
    parser.add_argument(
        "--a_hor", type=int, default=8
    )
    parser.add_argument("--goal_pos", type=str, required=True)
    return parser.parse_args(rospy.myargv()[1:])


if __name__ == "__main__":
    args = get_args()
    skill_name = args.skill_name
    model_type = args.model_type
    train_type = args.train_type
    a_hor = args.a_hor
    goal_pos = args.goal_pos.split(",")

    node = HalSkills(skill_name, model_type, train_type, goal_pos, a_hor)
    # node.start()
    node.main()
