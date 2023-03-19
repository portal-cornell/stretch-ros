#!/usr/bin/env python3

import rospy
import math
import argparse
import message_filters
import actionlib
from pathlib import Path
from model_bc import BC
from model_bc_trained import BC as BC_Trained
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from control_msgs.msg import FollowJointTrajectoryGoal, FollowJointTrajectoryAction
from trajectory_msgs.msg import JointTrajectoryPoint
import hello_helpers.hello_misc as hm

import torch
import numpy as np
from torchvision import transforms
from cv_bridge import CvBridge, CvBridgeError

from r3m import load_r3m
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

r3m = load_r3m("resnet18")
                                           
class HalSkills(hm.HelloNode):
    def __init__(self, args):
        hm.HelloNode.__init__(self)
        self.rate = 10.0
        self.trajectory_client = actionlib.SimpleActionClient('/stretch_controller/follow_joint_trajectory', FollowJointTrajectoryAction)

        self.step_size = "medium"
        self.rad_per_deg = math.pi/180.0
        self.small_deg = 3.0
        self.small_rad = self.rad_per_deg * self.small_deg
        self.small_translate = 0.005  #0.02
        self.medium_deg = 6.0
        self.medium_rad = self.rad_per_deg * self.medium_deg
        self.medium_translate = 0.04
        # self.mode = 'position' #'manipulation' #'navigation'
        self.mode = 'navigation'

        self.joint_states = None
        self.rbg_image = None
        self.joint_states_data = None
        self.cv_bridge = CvBridge()

        self.img_transform = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])            

        self.skill_name = args.skill_name
        self.model_type = args.model_type
        self.train_type = args.train_type

        print("\n\nskill_name: ", self.skill_name)
        print("model_type: ", self.model_type)
        print("train_type: ", self.train_type)
        print()

        ckpt_dir = Path(f"~/catkin_ws/src/stretch_ros/stretch_learning/checkpoints", 
                        args.skill_name, args.model_type, args.train_type).expanduser()
        # ckpt_name = "model_name=0-epoch=221-val_acc=1.00.ckpt"
        ckpt_name = "last.ckpt"
        ckpt_path = Path(ckpt_dir, ckpt_name)
        print(f"Loading checkpoint from {str(ckpt_path)}.\n")
        if "trained" in args.train_type:
            self.model = BC_Trained.load_from_checkpoint(ckpt_path)
        else:
            self.model = BC.load_from_checkpoint(ckpt_path)
        self.model.eval()

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

    def image_callback(self, ros_rgb_image):
        try:
            raw_image = self.cv_bridge.imgmsg_to_cv2(ros_rgb_image, 'bgr8')
            self.rbg_image = self.img_transform(raw_image)
            self.rbg_image = self.rbg_image.unsqueeze(0)
        except CvBridgeError as error:
            print(error)

    def index_to_keypressed(self, index):
        _index_to_keypressed = {
            # noop
            0: '_',
            # arm up
            1: "8",
            # arm down
            2: "2",
            # arm out
            3: 'w',
            # arm in
            4: 'x',
            # base forward
            5: '4',
            # base back
            6: '6',
            # base rotate left
            7: '7',
            # base rotate right
            8: '9',
            # gripper right
            9: 'a',
            # gripper left
            10: 'd',
            # gripper down
            11: 'c',
            # gripper up
            12: 'v',
            # gripper roll right
            13: 'o',
            # gripper roll left
            14: 'p',
            # gripper open
            15: '0',
            # gripper close
            16: '5'
        }
        return _index_to_keypressed[index]

    def get_deltas(self):
        if self.step_size == 'small':
            deltas = {'rad': self.small_rad, 'translate': self.small_translate}
        if self.step_size == 'medium':
            deltas = {'rad': self.medium_rad, 'translate': self.medium_translate} 
        if self.step_size == 'big':
            deltas = {'rad': self.big_rad, 'translate': self.big_translate} 
        return deltas

    def get_command(self, c):
        # 8 or up arrow
        if c == '8' or c == '\x1b[A':
            command = {'joint': 'joint_lift', 'delta': self.get_deltas()['translate']}
        # 2 or down arrow
        if c == '2' or c == '\x1b[B':
            command = {'joint': 'joint_lift', 'delta': -self.get_deltas()['translate']}
        if self.mode == 'manipulation':
            # 4 or left arrow
            if c == '4' or c == '\x1b[D':
                command = {'joint': 'joint_mobile_base_translation', 'delta': self.get_deltas()['translate']}
            # 6 or right arrow
            if c == '6' or c == '\x1b[C':
                command = {'joint': 'joint_mobile_base_translation', 'delta': -self.get_deltas()['translate']}
        elif self.mode == 'position':
            # 4 or left arrow
            if c == '4' or c == '\x1b[D':
                command = {'joint': 'translate_mobile_base', 'inc': self.get_deltas()['translate']}
            # 6 or right arrow
            if c == '6' or c == '\x1b[C':
                command = {'joint': 'translate_mobile_base', 'inc': -self.get_deltas()['translate']}
            # 1 or end key 
            if c == '7' or c == '\x1b[H':
                command = {'joint': 'rotate_mobile_base', 'inc': self.get_deltas()['rad']}
            # 3 or pg down 5~
            if c == '9' or c == '\x1b[5':
                command = {'joint': 'rotate_mobile_base', 'inc': -self.get_deltas()['rad']}
        elif self.mode == 'navigation':
            rospy.loginfo('ERROR: Navigation mode is not currently supported.')

        if c == 'w' or c == 'W':
            command = {'joint': 'wrist_extension', 'delta': self.get_deltas()['translate']}
        if c == 'x' or c == 'X':
            command = {'joint': 'wrist_extension', 'delta': -self.get_deltas()['translate']}
        if c == 'd' or c == 'D':
            command = {'joint': 'joint_wrist_yaw', 'delta': -self.get_deltas()['rad']}
        if c == 'a' or c == 'A':
            command = {'joint': 'joint_wrist_yaw', 'delta': self.get_deltas()['rad']}
        if c == 'v' or c == 'V':
            command = {'joint': 'joint_wrist_pitch', 'delta': -self.get_deltas()['rad']}
        if c == 'c' or c == 'C':
            command = {'joint': 'joint_wrist_pitch', 'delta': self.get_deltas()['rad']}
        if c == 'p' or c == 'P':
            command = {'joint': 'joint_wrist_roll', 'delta': -self.get_deltas()['rad']}
        if c == 'o' or c == 'O':
            command = {'joint': 'joint_wrist_roll', 'delta': self.get_deltas()['rad']}
        if c == '5' or c == '\x1b[E' or c == 'g' or c == 'G':
            # grasp
            command = {'joint': 'joint_gripper_finger_left', 'delta': -self.get_deltas()['rad']}
        if c == '0' or c == '\x1b[2' or c == 'r' or c == 'R':
            # release
            command = {'joint': 'joint_gripper_finger_left', 'delta': self.get_deltas()['rad']}
        if c == 'i' or c == 'I':
            command = {'joint': 'joint_head_tilt', 'delta': (2.0 * self.get_deltas()['rad'])}
        if c == ',' or c == '<':
            command = {'joint': 'joint_head_tilt', 'delta': -(2.0 * self.get_deltas()['rad'])}
        if c == 'j' or c == 'J':
            command = {'joint': 'joint_head_pan', 'delta': (2.0 * self.get_deltas()['rad'])}
        if c == 'l' or c == 'L':
            command = {'joint': 'joint_head_pan', 'delta': -(2.0 * self.get_deltas()['rad'])}
        if c == 'b' or c == 'B':
            rospy.loginfo('process_keyboard.py: changing to BIG step size')
            self.step_size = 'big'
        if c == 'm' or c == 'M':
            rospy.loginfo('process_keyboard.py: changing to MEDIUM step size')
            self.step_size = 'medium'
        if c == 's' or c == 'S':
            rospy.loginfo('process_keyboard.py: changing to SMALL step size')
            self.step_size = 'small'
        return command

    def send_command(self, command):
        joint_state = self.joint_states
        if (joint_state is not None) and (command is not None):
            point = JointTrajectoryPoint()
            point.time_from_start = rospy.Duration(0.0)
            trajectory_goal = FollowJointTrajectoryGoal()
            trajectory_goal.goal_time_tolerance = rospy.Time(1.0)
            
            joint_name = command['joint']
            trajectory_goal.trajectory.joint_names = [joint_name]
            if 'inc' in command:
                inc = command['inc']
                new_value = inc
            elif 'delta' in command:
                joint_index = joint_state.name.index(joint_name)
                joint_value = joint_state.position[joint_index]
                delta = command['delta']
                new_value = joint_value + delta
            point.positions = [new_value]
            trajectory_goal.trajectory.points = [point]
            trajectory_goal.trajectory.header.stamp = rospy.Time.now()
            self.trajectory_client.send_goal(trajectory_goal)

    #-----------------pick_pantry() initial configs-----------------#
    def move_arm_pick_pantry(self):
        rospy.loginfo("Set arm")
        self.pick_starting_height = 0.828
        self.joint_lift_index = self.joint_states.name.index("joint_lift")
        pose = {'wrist_extension': 0.01,
                'joint_lift': self.pick_starting_height,  # for cabinet
                'joint_wrist_pitch': 0.2,
                'joint_wrist_yaw': -0.09}
        self.move_to_pose(pose)
        return True

    def move_head_pick_pantry(self):
        tilt = -0.4358
        pan = -1.751
        rospy.loginfo("Set head pan")
        pose = {'joint_head_pan': pan, 'joint_head_tilt': tilt}
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

    #-----------------pick_table() initial configs-----------------#
    def move_arm_pick_table(self):
        rospy.loginfo("Set arm")
        self.pick_starting_height = 0.9096
        self.joint_lift_index = self.joint_states.name.index("joint_lift")
        pose = {'wrist_extension': 0.01,
                'joint_lift': self.pick_starting_height,
                'joint_wrist_pitch': 0.2,
                'joint_wrist_yaw': -0.09}
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
        if curr_height - self.pick_starting_height > 0.05:
            return True
        return False

    #-----------------place_table() initial configs-----------------#
    def move_arm_place_table(self):
        rospy.loginfo("Set arm")
        self.gripper_finger = self.joint_states.name.index("joint_gripper_finger_left")
        self.wrist_ext_indx = self.joint_states.name.index("joint_arm_l3")
        self.place_starting_extension = 0.01
        pose = {'wrist_extension': 0.01,
                'joint_lift': 0.9589,
                'joint_wrist_pitch': 0.1948,
                'joint_wrist_yaw': -0.089}
        self.move_to_pose(pose)
        return True

    def move_head_place_table(self):
        tilt = -0.1612
        pan = -1.757
        rospy.loginfo("Set head pan")
        pose = {'joint_head_pan': pan, 'joint_head_tilt': tilt}
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
        self.gripper_finger = self.joint_states.name.index("joint_gripper_finger_left")
        if self.joint_states.position[self.gripper_finger] > 0.19:
            return True
        return False



    def main(self):
        rospy.init_node("hal_skills_node")
        self.node_name = rospy.get_name()
        rospy.loginfo("{0} started".format(self.node_name))

        self.joint_states_subscriber = rospy.Subscriber('/stretch/joint_states', JointState, self.joint_states_callback)
        self.rgb_image_subscriber = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.rgb_image_subscriber.registerCallback(self.image_callback)

        rate = rospy.Rate(self.rate)

        # get hal to starting position for pick
        if self.skill_name == "pick_pantry":
            self.pick_pantry_initial_config(rate)
        elif self.skill_name == "pick_table":
            self.pick_table_initial_config(rate)
        elif self.skill_name ==  "place_table":
            self.place_table_initial_config(rate)


        while not rospy.is_shutdown():
            if self.joint_states is not None and self.rbg_image is not None:
                # check delta to determine skill termination
                if "pick" in self.skill_name and self.check_pick_termination():
                    rospy.loginfo("\n\n***********Pick Completed***********\n\n")
                    return 0
                elif "place" in self.skill_name and self.check_place_termination():
                    rospy.loginfo("\n\n***********Place Completed***********\n\n")
                    return 0

                # if not, continue with next command
                if len(self.joint_states_data.size()) <= 1:
                    print(self.joint_states_data)
                prediction = self.model(self.rbg_image, self.joint_states_data)
                keypressed_index = torch.argmax(prediction).item()
                keypressed = self.index_to_keypressed(keypressed_index)

                if keypressed == "_":
                    # noop
                    print("NOOP")
                    continue

                command = self.get_command(keypressed)
                print(f'{rospy.Time().now()}, {keypressed_index=}, {command=}')
                self.send_command(command)
            rate.sleep()

def get_args():
    supported_skills = ["pick_pantry", "place_table"]
    supported_models = ["visuomotor_bc"]
    supported_types = ["reg", "reg-no-vel", "end-eff", "end-eff-img", "end-eff-img-comp-2", \
                       "ee-img-trained", "ee-ic2-trained"]

    parser = argparse.ArgumentParser(description="main_lighting")
    parser.add_argument("--skill_name", type=str, choices=supported_skills, default="pick")
    parser.add_argument("--model_type", type=str, choices=supported_models, default="visuomotor_bc")
    parser.add_argument("--train_type", type=str, choices=supported_types, default="reg")
    return parser.parse_args(rospy.myargv()[1:])


if __name__ == '__main__':
    node = HalSkills(get_args())
    node.main()