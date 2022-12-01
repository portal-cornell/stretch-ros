#!/usr/bin/env python3

# Import modules
import rospy
import actionlib
import sys

# We need the MoveBaseAction and MoveBaseGoal from the move_base_msgs package.
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

# We're going to use the quaternion type here.
from geometry_msgs.msg import Quaternion, PoseWithCovarianceStamped, Twist

# tf includes a handy set of transformations to move between Euler angles and quaternions (and back).
from tf import transformations

from basic_navigation import StretchNavigation
from basic_move import Move

HOME = (-1.85, 2.77, -0.85)

TABLE = (-0.397, 0.024, -0.85)

MOVE_WITH_CART_POSE = (-0.397, 0.024, 0.5)

class DemoPlanner:
    def __init__(self):
        self.all_status = {"went_to_table": False, 
                            "went_back_home": False,
                            "latch_cart: raised_arm": False,
                            "latch_cart: extended_arm": False,
                            "reset_arm": False}

        self.amcl_listener = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.get_position_in_map)

        self.pos = None

    def get_position_in_map(self, msg):
        self.pos = msg.pose.pose.position
        print(f"new pos: {self.pos}")

    def done_callback(self, curr_action):
        def callback(status, result):
            print(self.all_status[curr_action])
            self.all_status[curr_action] = True
            if status == actionlib.GoalStatus.SUCCEEDED:
                rospy.loginfo('{0}: SUCCEEDED in reaching the goal.'.format(self.__class__.__name__))
            else:
                rospy.loginfo('{0}: FAILED in reaching the goal.'.format(self.__class__.__name__))    

        return callback    

# import rospy
import time
from control_msgs.msg import FollowJointTrajectoryGoal, FollowJointTrajectoryAction
from trajectory_msgs.msg import JointTrajectoryPoint
import hello_helpers.hello_misc as hm

class MultiPointCommand():
    """
    A class that sends multiple joint trajectory goals to the stretch robot.
    """
    def __init__(self):
        # hm.HelloNode.__init__(self)
        self.trajectory_client = actionlib.SimpleActionClient('/stretch_controller/follow_joint_trajectory', FollowJointTrajectoryAction)

        server_reached = self.trajectory_client.wait_for_server(timeout=rospy.Duration(60.0))
        if not server_reached:
            rospy.signal_shutdown('Unable to connect to arm action server. Timeout exceeded.')
            sys.exit()

    def issue_one_target_command(self, joint_name_list, position_list, task_name):
        """
        Function that makes an action call and sends multiple joint trajectory goals
        to the joint_lift, wrist_extension, and joint_wrist_yaw.
        :param self: The self reference.
        """
        point = JointTrajectoryPoint()
        point.positions = position_list
        # point.positions = [0.7, 
        #                         3.232259253198204e-05, 3.232259253198204e-05, 3.232259253198204e-05, 3.232259253198204e-05, 
        #                         2.9995715989780476, 
        #                         -0.013805827090970769]

        # point1 = JointTrajectoryPoint()
        # point1.positions = [0.9269133518339224, 
        #                         0.08809500066835729, 0.08809500066835729, 0.08809500066835729, 0.08809500066835729, 
        #                         -1.3371199201069839, 
        #                         -0.8866408953979006]

        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.trajectory.joint_names = joint_name_list
        trajectory_goal.trajectory.points = [point0]
        trajectory_goal.trajectory.header.stamp = rospy.Time(0.0)
        trajectory_goal.trajectory.header.frame_id = 'base_link'

        self.trajectory_client.send_goal(trajectory_goal, done_cb=callback)
        
        rospy.loginfo(f'[{task_name}] Sent list of goals = {trajectory_goal}')
        self.trajectory_client.wait_for_result()


    def lift_arm_before_extend(self):
        target_position_list = [0.7]
        joint_name_list = ["joint_lift"]

        self.issue_one_target_command(joint_name_list, target_position_list, "lift arm")

    def extend_arm_before_latching_cart(self):
        target_position_list = [0.9269133518339224, 
                                0.08809500066835729, 0.08809500066835729, 0.08809500066835729, 0.08809500066835729, 
                                -1.3371199201069839, 
                                -0.8866408953979006]
        joint_name_list = ['joint_lift', 
                                'joint_arm_l3', 'joint_arm_l2', 'joint_arm_l1', 'joint_arm_l0', 
                                'joint_wrist_yaw',
                                'joint_wrist_pitch']

        self.issue_one_target_command(joint_name_list, target_position_list, "extend arm")    

    def lower_arm_to_cart(self):
        target_position_list = [0.685428316319051]
        joint_name_list = ["joint_lift"]

        self.issue_one_target_command(joint_name_list, target_position_list, "latch cart")

    def lower_arm_to_cart(self):
        """
        Function that makes an action call and sends multiple joint trajectory goals
        to the joint_lift, wrist_extension, and joint_wrist_yaw.
        :param self: The self reference.
        """

        point = JointTrajectoryPoint()
        point.positions = [0.685428316319051]

        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.trajectory.joint_names = ['joint_lift']

        trajectory_goal.trajectory.points = [point]
        trajectory_goal.trajectory.header.stamp = rospy.Time(0.0)
        trajectory_goal.trajectory.header.frame_id = 'base_link'

        try:
            # self.trajectory_client.send_goal(trajectory_goal, done_cb = callback)
            self.trajectory_client.send_goal(trajectory_goal)
        except:
            pass
        rospy.loginfo('Sent list of goals = {0}'.format(trajectory_goal))
        self.trajectory_client.wait_for_result()
   

    def issue_multipoint_command_2(self, callback):
        """
        Function that makes an action call and sends multiple joint trajectory goals
        to the joint_lift, wrist_extension, and joint_wrist_yaw.
        :param self: The self reference.
        """
        point0 = JointTrajectoryPoint()
        point0.positions = [0.8, 
                                3.232259253198204e-05, 3.232259253198204e-05, 3.232259253198204e-05, 3.232259253198204e-05, 
                                2.9995715989780476, 
                                -0.013805827090970769]

        point1 = JointTrajectoryPoint()
        point1.positions = [0.9269133518339224, 
                                0.08809500066835729, 0.08809500066835729, 0.08809500066835729, 0.08809500066835729, 
                                -1.3371199201069839, 
                                -0.8866408953979006]

        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.trajectory.joint_names = ['joint_lift', 
                                                    'joint_arm_l3', 'joint_arm_l2', 'joint_arm_l1', 'joint_arm_l0', 
                                                    'joint_wrist_yaw',
                                                    'joint_wrist_pitch']
        trajectory_goal.trajectory.points = [point1]
        trajectory_goal.trajectory.header.stamp = rospy.Time(0.0)
        trajectory_goal.trajectory.header.frame_id = 'base_link'


        self.trajectory_client.send_goal(trajectory_goal, callback)
        rospy.loginfo('Sent list of goals = {0}'.format(trajectory_goal))
        self.trajectory_client.wait_for_result()


    def issue_multipoint_command_0(self, callback):
        """
        Function that makes an action call and sends multiple joint trajectory goals
        to the joint_lift, wrist_extension, and joint_wrist_yaw.
        :param self: The self reference.
        """
        point0 = JointTrajectoryPoint()
        point0.positions = [0.8, 
                                3.232259253198204e-05, 3.232259253198204e-05, 3.232259253198204e-05, 3.232259253198204e-05, 
                                2.9995715989780476, 
                                -0.013805827090970769]

        point = JointTrajectoryPoint()
        point.positions = [0.3, 
                                3.232259253198204e-05, 3.232259253198204e-05, 3.232259253198204e-05, 3.232259253198204e-05, 
                                2.9995715989780476, 
                                -0.013805827090970769]

        point1 = JointTrajectoryPoint()
        point1.positions = [0.9269133518339224, 
                                0.08809500066835729, 0.08809500066835729, 0.08809500066835729, 0.08809500066835729, 
                                -1.3371199201069839, 
                                -0.8866408953979006]

        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.trajectory.joint_names = ['joint_lift', 
                                                    'joint_arm_l3', 'joint_arm_l2', 'joint_arm_l1', 'joint_arm_l0', 
                                                    'joint_wrist_yaw',
                                                    'joint_wrist_pitch']
        trajectory_goal.trajectory.points = [point]
        trajectory_goal.trajectory.header.stamp = rospy.Time(0.0)
        trajectory_goal.trajectory.header.frame_id = 'base_link'

        try:
            # self.trajectory_client.send_goal(trajectory_goal, done_cb = callback)
            self.trajectory_client.send_goal_and_wait(trajectory_goal)
        except:
            pass
        rospy.loginfo('Sent list of goals = {0}'.format(trajectory_goal))
        self.trajectory_client.wait_for_result()


    def lower_arm_to_cart(self):
        """
        Function that makes an action call and sends multiple joint trajectory goals
        to the joint_lift, wrist_extension, and joint_wrist_yaw.
        :param self: The self reference.
        """

        point = JointTrajectoryPoint()
        point.positions = [0.685428316319051]

        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.trajectory.joint_names = ['joint_lift']

        trajectory_goal.trajectory.points = [point]
        trajectory_goal.trajectory.header.stamp = rospy.Time(0.0)
        trajectory_goal.trajectory.header.frame_id = 'base_link'

        try:
            # self.trajectory_client.send_goal(trajectory_goal, done_cb = callback)
            self.trajectory_client.send_goal(trajectory_goal)
        except:
            pass
        rospy.loginfo('Sent list of goals = {0}'.format(trajectory_goal))
        self.trajectory_client.wait_for_result()


    def main(self):
        """
        Function that initiates the multipoint_command function.
        :param self: The self reference.
        """
        # hm.HelloNode.main(self, 'multipoint_command', 'multipoint_command', wait_for_first_pointcloud=False)
        rospy.loginfo('issuing multipoint command...')

        self.issue_multipoint_command_0()

        # self.issue_multipoint_command()
        # time.sleep(2)
        # input("waiting before moving to next step")

        # self.issue_multipoint_command_2()
        # time.sleep(2)



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
    nav = StretchNavigation()
    
    nav.go_to(TABLE, planner.done_callback("went_to_table"))  # go roughly to the table

    # Don't need this. Go to already makes planner sleep
    # while not planner.all_status["went_to_table"]:
    #     print("========================================================")
    #     print("       Waiting for Stretch to go to the table")
    #     print("========================================================")

    # input("Waiting to get into the pose to move with cart")

    nav.go_to(MOVE_WITH_CART_POSE, planner.done_callback("went_to_table"))

    # lift, arms, wrist yaw, pitch\
    # [0.9269133518339224, 
    #     0.08809500066835729, 0.08809500066835729, 0.08809500066835729, 0.08809500066835729, 
    #     -1.3371199201069839, 
    #     -0.8866408953979006]

    # try:
    #     node = MultiPointCommand()
    #     node.main()
    # except KeyboardInterrupt:
    #     rospy.loginfo('interrupt received, so shutting down')
    
    joint_mover = MultiPointCommand()
    
    # joint_mover.issue_multipoint_command_0(planner.done_callback("reset_arm"))

    joint_mover.issue_multipoint_command(planner.done_callback("latch_cart: raised_arm"))

    time.sleep(4.5)

    joint_mover.issue_multipoint_command_2(planner.done_callback("latch_cart: extended_arm"))

    time.sleep(3)

    joint_mover.lower_arm_to_cart()

    time.sleep(2.5)

    # while not planner.all_status["latch_cart: extended_arm"]:
    #     print("waiting to finish extending arm")   

    # input("Waiting to get arm adjusted")

    # # base_motion = Move()

    # # while planner.pos.x > -0.74 and planner.pos.y > -0.1:
    # #     print("Moving stretch backward")
    # #     base_motion.move_x(speed=-0.1)

    # # print("Completed")

    # # input("Got to target position")

    nav.go_to(HOME, planner.done_callback("went_back_home"))  # go roughly to the table
