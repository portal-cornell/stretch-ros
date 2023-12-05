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

from engineered_skills.srv import Place,PlaceResponse

RUNNING = -1
SUCCESS = 1
NOT_STARTED = 2

class PlaceObjectNode(hm.HelloNode):

    def __init__(self):

        """
        TO RUN:
        0. In one terminal:
            yolov5

        1. In two separate terminals:
            sc
            hr

        2. In third terminal: 
            place
        """
        
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
        self.max_lift = 1.1
        self.min_extension = 0.01
        self.max_extension = 0.5
        self.gripper_close = -0.30
        self.gripper_open = 0.22

        self.rad = 4.5 * math.pi/180
        self.translate = 0.04
        self.gripper_len = 0.25

        self.arm_ext_target = 0.10 * 4
        self.arm_lower_target = 0.9505

    def init_node(self):
        rospy.init_node("place_node")
        self.node_name = rospy.get_name()
        rospy.loginfo("{0} started".format(self.node_name))
        

    def start(self):
        self.init_node()
        self.action_status = NOT_STARTED

        s = rospy.Service('place_server', Place, self.callback)
        rospy.loginfo("Place server has started")
        rospy.spin()

    def callback(self, req):
        if self.action_status == NOT_STARTED:
            # call hal_skills
            self.action_status = RUNNING

        self.main()
            
        return PlaceResponse(self.action_status)

    def joint_states_callback(self, joint_states):
        with self.joint_states_lock: 
            self.joint_states = joint_states
        self.wrist_position, _, _ = hm.get_wrist_state(joint_states)
        self.lift_position, _, _ = hm.get_lift_state(joint_states)
        self.left_finger_position, _, _ = hm.get_left_finger_state(joint_states)
        self.wrist_yaw, _, _ = hm.get_wrist_yaw(joint_states)

    def move_to_initial_configuration(self):
        rospy.loginfo("Set arm")
        pose = {'wrist_extension': 0.01,
                'joint_lift': 1.5,
                'joint_wrist_pitch': 0.1948,
                'joint_wrist_yaw': -0.089}
        self.move_to_pose(pose)
        return True

    def move_head(self):
        tilt = -0.1612
        pan = -1.757
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
        pose = {'wrist_extension': self.arm_ext_target}
        self.move_to_pose(pose)
        rospy.sleep(2.5)
        return True

    def lower_arm(self):
        pose = {'joint_lift': self.arm_lower_target}
        self.move_to_pose(pose)
        rospy.sleep(1)
        return True


    def main(self):
        print("\n\n-------Starting place() skill-------")
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

        self.change_grip(self.gripper_close)

        done_extending = False
        while not done_extending:
            if self.joint_states:
                done_extending = self.extend_arm()
            rate.sleep()

        done_lowering = False
        while not done_lowering:
            if self.joint_states:
                done_lowering = self.lower_arm()
            rate.sleep()

        self.change_grip(self.gripper_open)
        
        self.action_status=SUCCESS
        return 1 

    def start_no_service(self):
        self.init_node()
        rospy.loginfo("Place server has started, not waiting on service")
        self.main()


if __name__ == '__main__':
    try:
        parser = ArgumentParser()
        node = PlaceObjectNode()
        node.start()

        # node.start_no_service()
    except KeyboardInterrupt:
        rospy.loginfo('Interrupt received, shutting down.')