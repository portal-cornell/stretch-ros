#!/usr/bin/env python3

# Import modules
import rospy
import actionlib
import sys
import time
import math

# We need the MoveBaseAction and MoveBaseGoal from the move_base_msgs package.
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

# We're going to use the quaternion type here.
from geometry_msgs.msg import Quaternion, PoseWithCovarianceStamped, Twist

# tf includes a handy set of transformations to move between Euler angles and quaternions (and back).
from tf import transformations

from utils.basic_navigation import StretchNavigation
from utils.basic_move import Move
from utils.basic_joint_mover import JointMover

import hello_helpers.hello_misc as hm
from sensor_msgs.msg import JointState

from control_msgs.msg import FollowJointTrajectoryAction


from engineered_skills.srv import Move,MoveResponse

locations = {
"HOME": (0.0, 0.0, 0.0),
"TABLE": (1.9, -.782, math.pi/2),
"SALT": (0.26, -1.73, -1*math.pi/2),
"PEPPER": (0.26, -2.39, -1*math.pi/2)
}

MOVE_WITH_CART_POSE = (-0.397, 0.024, 0.5)

RUNNING = -1
SUCCESS = 1
NOT_STARTED = 2

class DemoPlanner(hm.HelloNode):
    def __init__(self):
        rospy.init_node("hal_move_node")
        self.node_name = rospy.get_name()
        rospy.loginfo("{0} started".format(self.node_name))

        self.all_status = {"went_to_table": False, 
                            "pos_ready_for_cart": False,
                            "went_back_home": False,
                            }

        self.amcl_listener = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.get_position_in_map)

        self.pos = None
        self.nav = StretchNavigation()
        self.joint_mover = JointMover()
        self.rate = 10.0
        self.joint_states = None
        self.joint_states_subscriber = rospy.Subscriber('/stretch/joint_states', JointState, self.joint_states_callback)
        self.trajectory_client = actionlib.SimpleActionClient('/stretch_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    
    def start(self):

        self.action_status = NOT_STARTED

        s = rospy.Service('move_server', Move, self.callback)
        rospy.loginfo("Move server has started")
        rospy.spin()
        # self.main()

    def main(self, loc):
        self.pick_pantry_initial_config(rospy.Rate(self.rate))
        coords = locations[loc]
        self.nav.go_to(coords, self.done_callback(f"went_to_{loc.lower()}"))  # go roughly to the table


    def callback(self, req):
        if self.action_status == NOT_STARTED:
            # call hal_skills
            self.action_status = RUNNING
            self.main(req.loc)
            self.action_status = NOT_STARTED
            return MoveResponse(SUCCESS)

        return MoveResponse(self.action_status)
    
    def move_arm_pick_pantry(self):
        rospy.loginfo("Set arm")
        self.pick_starting_height = 0.828
        self.joint_lift_index = self.joint_states.name.index("joint_lift")
        pose = {'wrist_extension': 0.01}
        self.move_to_pose(pose)
        return True
    
    def pick_pantry_initial_config(self, rate):
        done_initial_config = False
        while not done_initial_config:
            if self.joint_states:        
                done_initial_config = self.move_arm_pick_pantry()
            rate.sleep()
        # self.open_grip()

    def joint_states_callback(self, msg):
        self.joint_states = msg

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
        roslaunch engineered_skills demo_sp23.launch map_yaml:=${HELLO_FLEET_PATH}/maps/spring2023demo.yaml

    2. Then run:
        python3 /home/strech/new_ws/src/stretch_ros/engineered_skills/nodes/move.py

    3. to get a better point based on rviz: rostopic echo /clicked_point
    """

    # Initialize the node, and call it "navigation"
    # rospy.init_node('demo_planner', argv=sys.argv)
    planner = DemoPlanner()

    # planner.start()
    planner.nav.go_to((1.3038, -0.7492, (math.pi/4)), None)
    # planner.main("TABLE")
    # planner.main("PEPPER")
    # planner.main("TABLE")
    # planner.main("SALT")
    # planner.main("HOME")


    # Declare a `StretchNavigation` object


    """RESET ARM ONLY"""
    # joint_mover.reset_arm()
    """RESET ARM ONLY"""
    
    # joint_mover.lift_arm_before_extend()

    # time.sleep(3)

    # joint_mover.extend_arm_before_latching_cart()

    # time.sleep(2.5)

    # joint_mover.lower_arm_to_cart()

    # time.sleep(2)

    # nav.go_to(HOME, planner.done_callback("went_back_home"))  # go roughly to the table