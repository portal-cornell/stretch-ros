#!/usr/bin/env python3

import rospy
import math
import message_filters
import actionlib
from pathlib import Path
from model_lightning import BC
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from control_msgs.msg import FollowJointTrajectoryGoal, FollowJointTrajectoryAction
from trajectory_msgs.msg import JointTrajectoryPoint
import hello_helpers.hello_misc as hm

import torch
import numpy as np
from torchvision import transforms
from cv_bridge import CvBridge, CvBridgeError


class OpenDrawer(hm.HelloNode):
    def __init__(self):
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
        self.mode = 'position' #'manipulation' #'navigation'

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
        ckpt_path = Path("~/catkin_ws/src/stretch_ros/stretch_learning/checkpoints/r3m.ckpt").expanduser()
        self.model = BC.load_from_checkpoint(ckpt_path)

    def joint_states_callback(self, msg):
        self.joint_states = msg
        js_msg = list(zip(msg.name, msg.position, msg.velocity, msg.effort))
        js_msg = sorted(js_msg, key=lambda x: x[0])
        js_arr = []
        for (_, pos, vel, eff) in js_msg:
            js_arr.extend([pos, vel, eff])
        joint_states = torch.from_numpy(np.array(js_arr, dtype=np.float32))
        self.joint_states_data = joint_states.unsqueeze(0)

    def image_callback(self, ros_rgb_image):
        try:
            raw_image = self.cv_bridge.imgmsg_to_cv2(ros_rgb_image, 'bgr8')
            self.rbg_image = self.img_transform(raw_image)
            self.rbg_image = self.rbg_image.unsqueeze(0)
        except CvBridgeError as error:
            print(error)

    def move_to_initial_configuration(self):
        # TODO: needs to be changed
        initial_pose = {'wrist_extension': 0.01,
                        'joint_wrist_yaw': 1.570796327,
                        'gripper_aperture': 0.0}
        rospy.loginfo('Move to the initial configuration for drawer opening.')
        self.move_to_pose(initial_pose) 

    def index_to_keypressed(self, index):
        _index_to_keypressed = {
            # base forward
            0: '4',
            # base back
            1: '6',
            # base rotate left
            2: '7',
            # base rotate right
            3: '9',
            # arm up
            4: '8',
            # arm down
            5:'2',
            # arm out
            6:'w',
            # arm in
            7: 'x',
            # gripper right
            8: 'a',
            # gripper left
            9: 'd',
            # gripper down
            10: 'c',
            # gripper up
            11: 'v',
            # gripper roll right
            12: 'o',
            # gripper roll left
            13: 'p'
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

    def main(self):
        rospy.init_node("open_drawer_bc")
        self.node_name = rospy.get_name()
        rospy.loginfo("{0} started".format(self.node_name))

        self.joint_states_subscriber = rospy.Subscriber('/stretch/joint_states', JointState, self.joint_states_callback)
        self.rgb_image_subscriber = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.rgb_image_subscriber.registerCallback(self.image_callback)

        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            if self.joint_states is not None and self.rbg_image is not None:
                prediction = self.model(self.rbg_image, self.joint_states_data)
                keypressed_index = torch.argmax(prediction).item()
                keypressed = self.index_to_keypressed(keypressed_index)
                command = self.get_command(keypressed)
                print(f'{rospy.Time().now()}, {keypressed_index=}, {command=}')
                self.send_command(command)
            rate.sleep()


if __name__ == '__main__':
    node = OpenDrawer()
    node.main()