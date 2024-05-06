import math
import rospy
from move_ps.basic_navigation import StretchNavigation
import hello_helpers.hello_misc as hm
from std_srvs.srv import Trigger
import torch
import actionlib
import tf2_ros
import time
from geometry_msgs.msg import Twist

# place
from hal_skills_aruco import HalSkillsPlace

# ppo fixed
from hal_skills_ppo_eval import HalSkillsNode

# ppo full
# from hal_skills_final_odom import HalSkillsNode

# from hal_skills_pred_odom import HalSkills
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryGoal, FollowJointTrajectoryAction
from stretch_funmap.navigate import MoveBase


device = "cuda" if torch.cuda.is_available() else "cpu"

FRIDGE_LOC = [0, -0.7492, 0]
TABLE_LOC = [0, 0, 0]

# skill status
RUNNING = -1
SUCCESS = 1
NOT_STARTED = 2


class Hal(hm.HelloNode):
    def __init__(self):
        hm.HelloNode.__init__(self)

        self.trajectory_client = actionlib.SimpleActionClient(
            "/stretch_controller/follow_joint_trajectory", FollowJointTrajectoryAction
        )

        rospy.init_node("hal_move_skills_node")
        self.node_name = rospy.get_name()
        rospy.loginfo("{0} started".format(self.node_name))

        self.pub = rospy.Publisher('/stretch/cmd_vel', Twist, queue_size=1) #/stretch_diff_drive_controller/cmd_vel for gazebo

        self.hal_skills_pick = HalSkillsNode()
        self.hal_skills_pick.subscribe()

        self.nav = StretchNavigation()
        self.rate = 10.0

        self.action_status = NOT_STARTED
    
    def close_grip(self):
        point = JointTrajectoryPoint()
        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.goal_time_tolerance = rospy.Time(1.0)
        trajectory_goal.trajectory.joint_names = ["joint_gripper_finger_left"]
        point.positions = [-0.2]
        trajectory_goal.trajectory.points = [point]
        self.trajectory_client.send_goal(trajectory_goal)
        grip_change_time = 2
        rospy.sleep(grip_change_time)

    def initial_arm_move(self):
        rospy.loginfo("Set arm")
        pose = {
            "wrist_extension": 0.05,
            "joint_lift": 0.3,  # for cabinet -0.175
            "joint_wrist_pitch": 0.0,
            "joint_wrist_yaw": 2.8,
        }
        self.move_to_pose(pose)
        self.close_grip()
        return True
    
    def initial_arm_place(self):
        rospy.loginfo("Set arm")
        pose = {
            "wrist_extension": 0.05,
            "joint_lift": 0.3,  # for cabinet -0.175
            "joint_wrist_pitch": 0.0,
            "joint_wrist_yaw": 0.0,
        }
        self.move_to_pose(pose)
        self.close_grip()
        return True
    
    def retract_arm_primitive(self):
        rospy.loginfo("Set arm")
        pose = {
            "wrist_extension": 0.05
        }
        self.move_to_pose(pose)
        return True
    
    def pick_goal(self):
        """
        reset parameter defaults to False: when False, Hal will reset its arm and execute policy assuming it's already in reset position
        """
        self.hal_skills_pick.main()
        self.retract_arm_primitive()
    
    def place_goal(self, place_loc):
        """
        reset parameter defaults to False: when False, Hal will reset its arm and execute policy assuming it's already in reset position
        """
        self.initial_arm_place()

        self.hal_skills_place = HalSkillsPlace(place_loc)
        self.hal_skills_place.main()

        self.retract_arm_primitive()
    
    def move_table_callback(self, status, result):
        print("at table")
        self.action_status = SUCCESS
        self.current_location = TABLE_LOC
    
    def move_table(self):
        """
        !!!! MAKE SURE ARM IS RESETTED BEFORE EXECUTING MOVE
        """
        # this is to execute move
        s = rospy.ServiceProxy("/switch_to_navigation_mode", Trigger)
        resp = s()
        print(resp)
        
        self.initial_arm_move()
        self.nav.go_to(TABLE_LOC, self.move_table_callback)
    
    def move_fridge_callback(self, status, result):
        print("at fridge")
        self.action_status = SUCCESS
        self.current_location = FRIDGE_LOC
    
    def move_fridge(self):
        """
        !!!! MAKE SURE ARM IS RESETTED BEFORE EXECUTING MOVE
        """
        # this is to execute move
        s = rospy.ServiceProxy("/switch_to_navigation_mode", Trigger)
        resp = s()
        print(resp)
        
        self.initial_arm_move()
        self.nav.go_to(FRIDGE_LOC, self.move_fridge_callback)
    
    def grocery_main(self, place_loc):
        self.pick_goal()

        self.move_fridge()

        self.place_goal(place_loc)

        self.move_table()

        self.action_status = SUCCESS

if __name__ == "__main__":
    hal = Hal()
    hal.move_fridge()
    
