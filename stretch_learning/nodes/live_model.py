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
from torchvision import transforms
from cv_bridge import CvBridge, CvBridgeError

# BC imports
from r3m import load_r3m
from bc.model import BC

# from bc.model_bc_trained import BC as BC_Trained

# IQL imports
from iql.img_js_net import ImageJointStateNet
from iql.iql import load_iql_trainer

from stretch_learning.srv import Pick, PickResponse

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

class HalSkills(hm.HelloNode):
    def __init__(self, skill_name, model_type, train_type):
        hm.HelloNode.__init__(self)
        self.debug_mode = False
        self.rate = 10.0
        self.trajectory_client = actionlib.SimpleActionClient(
            "/stretch_controller/follow_joint_trajectory", FollowJointTrajectoryAction
        )
        
        self.mode = "navigation"

        self.joint_states = None
        self.wrist_image = None
        self.head_image = None
        self.joint_states_data = None
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
        elif self.model_type == "irl" and self.train_type == "iql":
            self.model = self.load_iql_model(ckpt_dir)

        self.init_node()

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
        print(f"Loading checkpoint from {str(ckpt_path)}.\n")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.img_js_net.eval()
        model.actor.eval()
        return model

    def load_bc_model(self, ckpt_dir):
        print(ckpt_dir)
        ckpts = [ckpt for ckpt in ckpt_dir.glob("*.pt") if "last" not in ckpt.stem]
        ckpts += [ckpt for ckpt in ckpt_dir.glob("*.ckpt") if "last" not in ckpt.stem]
        # import pdb; pdb.set_trace()
        if "val_acc" in str(ckpts[0]):
            ckpts.sort(key=lambda x: float(x.stem.split("val_acc=")[1]), reverse=True)
        else:
            ckpts.sort(
                key=lambda x: float(x.stem.split("combined_acc=")[1]), reverse=True
            )

        ckpt_path = ckpts[0]
        # ckpt_path = Path(ckpt_dir, "last.ckpt")
        print(f"Loading checkpoint from {str(ckpt_path)}.\n")

        model = BC(
            skill_name=self.skill_name,
            joint_state_dims=14 * 3,
            state_action_dims=17,
            device=device,
            use_joints=False
        )
        state_dict = torch.load(ckpt_path, map_location=device)
        modified_dict = {}
        for key, value in state_dict.items():
            key = key.replace("_orig_mod.", "")
            modified_dict[key] = value
        model.load_state_dict(modified_dict)
        model.to(device)
        model.eval()
        return model

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
        self.joint_states_data = torch.from_numpy(np.array(js_arr, dtype=np.float32))
        if len(self.joint_states_data.size()) <= 1:
            self.joint_states_data = self.joint_states_data.unsqueeze(0)

    def wrist_image_callback(self, ros_rgb_image):
        try:
            self.wrist_raw_image = self.cv_bridge.imgmsg_to_cv2(ros_rgb_image, "bgr8")
            self.wrist_image = self.img_transform(self.wrist_raw_image)
            self.wrist_image = self.wrist_image.unsqueeze(0)
        except CvBridgeError as error:
            print(error)
    def head_image_callback(self, ros_rgb_image):
        try:
            self.head_raw_image = self.cv_bridge.imgmsg_to_cv2(ros_rgb_image, "bgr8")
            self.head_image = self.img_transform(self.head_raw_image)
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
            9: "a",
            # gripper left
            10: "d",
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

        ####################################################

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

        print("start of loop")
        times = []
        while not rospy.is_shutdown():
            # print("not shutdown")
            # print (f"joint states: {self.joint_states is not None}")
            # print (f"wrist image: {self.wrist_image is not None}")
            # print (f"head image: {self.head_image is not None}")
            if self.joint_states is not None and self.wrist_image is not None and self.head_image is not None:

                # if not, continue with next command
                if len(self.joint_states_data.size()) <= 1:
                    print(self.joint_states_data)
                    continue

                if self.model_type == "visuomotor_bc":
                    start = time.time()
                    prediction = self.model(self.wrist_image, self.joint_states_data, image2=self.head_image)
                    times.append(time.time() - start)
                elif self.train_type == "iql":
                    observation = self.model.img_js_net(
                        self.rbg_image, self.joint_states_data
                    )
                    prediction = self.model.actor.act(observation)
                else:
                    raise NotImplementedError
                keypressed_index = torch.argmax(prediction).item()
                best3 = torch.argsort(prediction).flatten().flip(0)[:3]
                probs = torch.nn.functional.softmax(prediction).flatten()

                # print(f"PREDICTION: {INDEX_TO_KEYPRESSED[keypressed_index]}")
                print("Best 3")
                for i in range(3):
                    print(f"Prediction #{i+1}: {INDEX_TO_KEYPRESSED[best3[i].item()]}, {probs[best3[i]]}")
                print("-"*50)

                # print(f"{rospy.Time().now()}, {keypressed_index=}")
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
    return parser.parse_args(rospy.myargv()[1:])


if __name__ == "__main__":
    args = get_args()
    skill_name = args.skill_name
    model_type = args.model_type
    train_type = args.train_type

    node = HalSkills(skill_name, model_type, train_type)
    # node.start()
    node.main()
