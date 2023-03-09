#!/usr/bin/env python3

import rospy
import math
import threading
import actionlib
from control_msgs.msg import FollowJointTrajectoryGoal, FollowJointTrajectoryAction
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState

from argparse import ArgumentParser

import hello_helpers.hello_misc as hm
import stretch_funmap.navigate as nv

class PickObjectNode(hm.HelloNode):

    def __init__(self):
        hm.HelloNode.__init__(self)
        self.rate = 10.0
        self.trajectory_client = actionlib.SimpleActionClient('/stretch_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.joints = ['joint_lift']
        self.joint_states = None
        self.joint_states_lock = threading.Lock()
        self.move_base = nv.MoveBase(self)
        self.letter_height_m = 0.2
        self.wrist_position = None
        self.lift_position = None
        self.manipulation_view = None
        self.debug_directory = None

        # limiting params
        self.min_lift = 0.3
        self.max_lift = 1.0
        self.min_extension = 0.01
        self.max_extension = 0.5
        self.gripper_close = -0.30
        self.gripper_open = 0.22

        self.rad = 4.5 * math.pi/180
        self.translate = 0.04
        self.gripper_len = 0.25

        # TODO: hardcoded based on joint_lift, joint_arm_, joint_wrist_yaw
        # joint_lift = 0.8284666114772404
        # joint_arm_l0 = 0.08061580522710103
        # joint_wrist_yaw = -0.17768610793008674 (radians, moving in direction behind body is negative)
        self.x = 0.081*4*math.tan(-0.0)
        self.y = 0.081 * 4
        self.z = 0.828

    def joint_states_callback(self, joint_states):
        with self.joint_states_lock: 
            self.joint_states = joint_states
        self.wrist_position, _, _ = hm.get_wrist_state(joint_states)
        self.lift_position, _, _ = hm.get_lift_state(joint_states)
        self.left_finger_position, _, _ = hm.get_left_finger_state(joint_states)
        self.wrist_yaw, _, _ = hm.get_wrist_yaw(joint_states)

    def move_to_initial_configuration(self):
        # self.joint_name_to_target_pos("wrist_extension", 0.01)
        # self.joint_name_to_target_pos("joint_lift", self.z)
        # self.joint_name_to_target_pos("joint_wrist_pitch", 0.2)
        # self.joint_name_to_target_pos("joint_wrist_yaw", -0.14)

        # arm configs
        rospy.loginfo("Set arm")
        pose = {'wrist_extension': 0.01,
                'joint_lift': self.z,
                'joint_wrist_pitch': 0.2,
                'joint_wrist_yaw': -0.09}
        self.move_to_pose(pose)
        return True

    def move_head(self):
        tilt = -0.4358
        pan = -1.751
        rospy.loginfo("Set head pan")
        pose = {'joint_head_pan': pan, 'joint_head_tilt': tilt}
        self.move_to_pose(pose)
        return True

    def change_grip(self, new_grip_pos):
        point = JointTrajectoryPoint()
        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.goal_time_tolerance = rospy.Time(1.0)
        trajectory_goal.trajectory.joint_names = ["joint_gripper_finger_left"]
        point.positions = [new_grip_pos]
        trajectory_goal.trajectory.points = [point]
        self.trajectory_client.send_goal(trajectory_goal)
        grip_change_time = 2
        rospy.sleep(grip_change_time)

    def extend_arm(self):
        # extend, rotate = self.y, self.x
        y_dist = self.y - self.wrist_position + self.gripper_len
        x_dist = self.x
        rads_to_object_center = math.tan(x_dist/y_dist)

        if (self.y - self.wrist_position) < 0:
            return True
        if (rads_to_object_center < self.wrist_yaw-self.left_finger_position):
            # print("right:", self.left_finger_position + self.wrist_yaw)
            command = {'joint': 'joint_wrist_yaw', 'delta': rads_to_object_center-self.wrist_yaw}
        elif (rads_to_object_center > self.wrist_yaw+self.left_finger_position):
            # print("left:", self.wrist_yaw-self.left_finger_position)
            command = {'joint': 'joint_wrist_yaw', 'delta': rads_to_object_center-self.wrist_yaw}
        else:
            # print("FOWARD BY: ", self.translate)
            command = {'joint': 'wrist_extension', 'delta': self.translate}
        self.send_command(command)
        return False

    def end_pick(self):
        command = {'joint': 'joint_lift', 'delta': self.translate}
        self.send_command(command)

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
        print("\n\n-------Starting pick() skill-------")
        rospy.init_node("pick_engineered")
        self.joint_states_subscriber = rospy.Subscriber('/stretch/joint_states', JointState, self.joint_states_callback)

        rate = rospy.Rate(self.rate)

        done_head_pan = False
        while not done_head_pan:
            if self.joint_states:
                done_head_pan = self.move_head()
            rate.sleep()

        done_initial_config = False
        while not done_initial_config:
            if self.joint_states:        
                done_initial_config = self.move_to_initial_configuration()
            rate.sleep()

        self.change_grip(self.gripper_open)

        done_extending = False
        while not done_extending:
            if self.joint_states:
                done_extending = self.extend_arm()
            rate.sleep()

        self.change_grip(self.gripper_close)
        self.end_pick()

        # done_closing = False
        # while not done_closing:
        #     if self.joint_states:
        #         done_closing = self.change_grip(self.gripper_close)
        #     rate.sleep()

        # done_pick = False
        # while not done_pick:
        #     if self.joint_states:
        #         done_pick = self.end_pick()
        #     rate.sleep()

        raise KeyboardInterrupt

        # TODO get x, y, z (will call move_head here)


if __name__ == '__main__':
    try:
        parser = ArgumentParser()
        node = PickObjectNode()
        node.main()
    except KeyboardInterrupt:
        rospy.loginfo('Interrupt received, shutting down.')