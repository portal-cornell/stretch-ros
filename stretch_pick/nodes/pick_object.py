#!/usr/bin/env python3

import rospy
import threading
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction

from argparse import ArgumentParser

import hello_helpers.hello_misc as hm
import stretch_funmap.navigate as nv

class PickObjectNode(hm.HelloNode):

    def __init__(self):
        hm.HelloNode.__init__(self)
        self.rate = 10.0
        self.trajectory_client = actionlib.SimpleActionClient('/stretch_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.joint_states = None
        self.joint_states_lock = threading.Lock()
        self.move_base = nv.MoveBase(self)
        self.letter_height_m = 0.2
        self.wrist_position = None
        self.lift_position = None
        self.manipulation_view = None
        self.debug_directory = None

    def joint_states_callback(self, joint_states):
        with self.joint_states_lock: 
            self.joint_states = joint_states
        wrist_position, wrist_velocity, wrist_effort = hm.get_wrist_state(joint_states)
        self.wrist_position = wrist_position
        lift_position, lift_velocity, lift_effort = hm.get_lift_state(joint_states)
        self.lift_position = lift_position
        self.left_finger_position, temp1, temp2 = hm.get_left_finger_state(joint_states)

    def move_to_initial_configuration(self):
        initial_pose = {'joint_wrist_pitch': 0.17487380981896308,
                        'joint_wrist_roll': -0.0046019423636569235,
                        'joint_gripper_finger_right': 0.21431472632290202,
                        'joint_gripper_finger_left': 0.21431472632290202,
                        'joint_right_wheel': 0.0,
                        'joint_left_wheel': 0.0,
                        'joint_lift': 0.8157716393952612,
                        'joint_arm_l3': 0.0005087813420959571,
                        'joint_arm_l2': 0.0005087813420959571,
                        'joint_arm_l1': 0.0005087813420959571,
                        'joint_arm_l0': 0.0005087813420959571,
                        'joint_wrist_yaw': -0.18727348785437203,
                        'joint_head_pan': -1.8047646014497567,
                        'joint_head_tilt': -0.9130266522276587}
        rospy.loginfo('Move to the initial configuration for drawer opening.')
        self.move_to_pose(initial_pose)

    def execute(self):
        print("\n\n-------Starting pick() skill-------")
        rospy.init_node("pick")
        self.move_to_initial_configuration()

if __name__ == '__main__':
    try:
        parser = ArgumentParser()
        node = PickObjectNode()
        node.execute()
    except KeyboardInterrupt:
        rospy.loginfo('Interrupt received, shutting down.')