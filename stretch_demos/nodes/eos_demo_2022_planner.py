#!/usr/bin/env python3

# Import modules
import rospy
import actionlib
import sys

# We need the MoveBaseAction and MoveBaseGoal from the move_base_msgs package.
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

# We're going to use the quaternion type here.
from geometry_msgs.msg import Quaternion

# tf includes a handy set of transformations to move between Euler angles and quaternions (and back).
from tf import transformations

from basic_navigation import StretchNavigation

HOME = (-1.85, 2.77, -0.85)

TABLE = (-0.397, 0.024, -0.85)

class DemoPlanner:
    def __init__(self):
        self.all_status = {"went_to_table": False}

    def done_callback(self, curr_action):
        def callback(status, result):
            if status == actionlib.GoalStatus.SUCCEEDED:
                rospy.loginfo('{0}: SUCCEEDED in reaching the goal.'.format(self.__class__.__name__))
                self.all_status[curr_action] = True
            else:
                rospy.loginfo('{0}: FAILED in reaching the goal.'.format(self.__class__.__name__))    

        return callback    


if __name__ == '__main__':
    """
    TO RUN:

    1. In one terminal:
        roslaunch stretch_navigation navigation.launch map_yaml:=${HELLO_FLEET_PATH}/maps/mapping_eod_11_14.yaml

    2. Then run:
        python3 /home/strech/catkin_ws/src/stretch_ros/stretch_demos/nodes/eos_demo_2022_planner.py

    3. to get a better point based on rviz: rostopic echo /clicked_point
    """

    # Initialize the node, and call it "navigation"
    rospy.init_node('demo_planner', argv=sys.argv)

    planner = DemoPlanner()

    # Declare a `StretchNavigation` object
    nav = StretchNavigation()
    
    nav.go_to(HOME, planner.done_callback("went_to_table"))  # go roughly to the table

    # Don't need this. Go to already makes planner sleep
    # while not planner.all_status["went_to_table"]:
    #     print("========================================================")
    #     print("       Waiting for Stretch to go to the table")
    #     print("========================================================")
