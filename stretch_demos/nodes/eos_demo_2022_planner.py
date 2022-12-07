#!/usr/bin/env python3

# Import modules
import rospy
import actionlib
import sys
import time
import subprocess
import signal
import os

# We need the MoveBaseAction and MoveBaseGoal from the move_base_msgs package.
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

# We're going to use the quaternion type here.
from geometry_msgs.msg import Quaternion, PoseWithCovarianceStamped, Twist

# tf includes a handy set of transformations to move between Euler angles and quaternions (and back).
from tf import transformations

from basic_navigation import StretchNavigation
from basic_move import Move
from basic_joint_mover import JointMover

HOME = (-1.85, 2.77, -0.85)

TABLE = (-0.397, 0.024, -0.85)

MOVE_WITH_CART_POSE = (-0.397, 0.024, 0.5)

class DemoPlanner:
    def __init__(self):
        self.all_status = {"went_to_table": False, 
                            "pos_ready_for_cart": False,
                            "went_back_home": False,
                            }

        self.amcl_listener = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.get_position_in_map)

        self.pos = None

    def get_position_in_map(self, msg):
        self.pos = msg.pose.pose.position
        print(f"new pos: {self.pos}")

    def done_callback(self, curr_action):
        def callback(status, result):
            self.all_status[curr_action] = True
            if status == actionlib.GoalStatus.SUCCEEDED:
                rospy.loginfo('{0}: SUCCEEDED in reaching the goal.'.format(curr_action))
            else:
                rospy.loginfo('{0}: FAILED in reaching the goal.'.format(curr_action))    

        return callback    

    
if __name__ == '__main__':
    """
    TO RUN:

    1. In one terminal:
        roslaunch stretch_demos eos_demo_2022.launch map_yaml:=${HELLO_FLEET_PATH}/maps/mapping_eod_11_14.yaml

    2. Then run:
        python3 /home/strech/catkin_ws/src/stretch_ros/stretch_demos/nodes/eos_demo_2022_planner.py

    3. to get a better point based on rviz: rostopic echo /clicked_point
    """

    # Initialize the node, and call it "navigation"
    rospy.init_node('demo_planner', argv=sys.argv)

    planner = DemoPlanner()

    # Declare a `StretchNavigation` object
    # nav = StretchNavigation()

    # joint_mover = JointMover()

    base_mover = Move()

    flag = True
    count = 0
    # while flag:
    #     base_mover.move_x(1.0)
    #     check = input('s')
    #     if check == 's': flag = False
    #     count += 1
    # print("count", count)

    while count < 50:
        print(count)
        base_mover.move_x(1.0)
        time.sleep(0.5)
        count += 1

    """RESET ARM ONLY"""
    # joint_mover.reset_arm()
    """RESET ARM ONLY"""
    
    # nav.go_to((-0.397, 0.024, -1.0), planner.done_callback("went_to_table"))  # go roughly to the table
    
    # input("waiting to start grasp node launch file")

    # p = subprocess.Popen(["roslaunch", "stretch_demos", "grasp_object_demo_2022.launch"])

    # input("started the grasp node launch file")

    # os.kill(p.pid, signal.SIGINT)

    # input("killed the grasp node launch file")

    # nav.go_to(MOVE_WITH_CART_POSE, planner.done_callback("pos_ready_for_cart"))
    
    # joint_mover.lift_arm_before_extend()

    # time.sleep(3)

    # joint_mover.extend_arm_over_cart()

    # time.sleep(2.5)

    # """Need to test the time"""
    # joint_mover.open_gripper()

    # time.sleep(2)

    # joint_mover.rotate_wrist_before_latching_cart()

    # time.sleep(2)
    # """Need to test the time"""

    # joint_mover.lower_arm_to_cart()

    # time.sleep(2)

    # nav.go_to(HOME, planner.done_callback("went_back_home"))  # go roughly to the table