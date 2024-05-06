import math
import rospy
from move_ps.basic_navigation import StretchNavigation
import hello_helpers.hello_misc as hm
from std_srvs.srv import Trigger
import torch
import actionlib
import tf2_ros
import time
import numpy as np
from geometry_msgs.msg import Twist, Pose2D

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

FRIDGE_LOC = [1.3038, -0.7492, math.pi/4]
TABLE_LOC = [1.3038, -0.2, math.pi/4]
HOME_LOC = (0.1, 0, 0)

TABLE_HEIGHT = 0.97  # arbitrary table height
TABLE_WRIST_EXT = 0.4
SHELF_HEIGHT = 0.80
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

        self.curr_pose = None
        self.linear_gain = 0.8
        self.angular_gain = 0.8
        self.dist_threshold = 0.02
        self.ang_threshold = 0.02
        self.max_linear_velocity = 0.1
        self.max_angular_velocity = 0.1
        self.turn_in_place_gain = 0.3

        self.curr_pose_sub = rospy.Subscriber(
            "/pose2D",
            Pose2D,
            self.curr_pose_callback,
        )


    def move_x(self, speed):
        """
        Function that publishes Twist messages
        :param self: The self reference.

        :publishes command: Twist message.
        """
        command = Twist()
        command.linear.x = speed
        command.linear.y = 0.0
        command.linear.z = 0.0
        command.angular.x = 0.0
        command.angular.y = 0.0
        command.angular.z = 0.0
        self.pub.publish(command)
    
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
    
    def timed_move(self, duration, speed):
        start = time.time()
        t = 0
        while t < duration:
            current = time.time()
            t = current - start
            self.move_x(speed)
        self.move_x(0)
    
    def curr_pose_callback(self, msg):
        self.curr_pose = [msg.x, msg.y, msg.theta]
    
    def at_xy_goal(self, curr, goal):
        if np.linalg.norm([curr[0] - goal[0], curr[1] - goal[1]]) < self.dist_threshold:
            return True
        else:
            return False
    
    def at_angle_goal(self, curr, goal):
        if np.abs(curr[2] - goal[2]) < self.ang_threshold:
            return True
        else:
            return False
    
    def reset_velocity(self):
        """
        Function that publishes Twist messages
        :param self: The self reference.

        :publishes command: Twist message.
        """
        command = Twist()
        command.linear.x = 0.0
        command.linear.y = 0.0
        command.linear.z = 0.0
        command.angular.x = 0.0
        command.angular.y = 0.0
        command.angular.z = 0.0
        self.pub.publish(command)
    
    def move_velocity(self, linear_x, angular_z):
        """
        Function that publishes Twist messages
        :param self: The self reference.

        :publishes command: Twist message.
        """
        command = Twist()
        command.linear.x = linear_x
        command.linear.y = 0.0
        command.linear.z = 0.0
        command.angular.x = 0.0
        command.angular.y = 0.0
        command.angular.z = angular_z
        self.pub.publish(command)

    def pid_move(self, goal):
        """2 Stage approach
        Stage 1 - Approach the goal ASAP
        Stage 2 - Rotate the base to match the angle defined in the goal positon 
        
        Parameter:
            goal (list): [x, y, theta]
        """
        self.initial_arm_move()
        s = rospy.ServiceProxy("/switch_to_navigation_mode", Trigger)
        resp = s()
        print(resp)
        print("Go to goal")

        #  Stage 1
        while not rospy.is_shutdown() and not self.at_xy_goal(self.curr_pose, goal):
            if self.curr_pose is not None:
                diff_vector = [goal[0] - self.curr_pose[0], goal[1] - self.curr_pose[1]]
                print(f"{self.curr_pose=}")
                print(f"{diff_vector=}")
                angle_to_goal = np.arctan2(diff_vector[1], diff_vector[0])
                print(f"{angle_to_goal=}")
                distance_error = np.linalg.norm(diff_vector)
                angle_error = angle_to_goal - self.curr_pose[2] # Difference between orientation and the angle btw the robot and the goal
                print(f"{angle_error=}")

                linear_velocity = np.clip(self.linear_gain * distance_error * np.cos(angle_error), 
                                          a_min=-self.max_linear_velocity, a_max=self.max_linear_velocity)
                angular_velocity = np.clip(self.angular_gain * np.abs(angle_error) * (np.sign(linear_velocity) if linear_velocity > 0.005 else 1) * np.sin(angle_error), 
                                            a_min=-self.max_angular_velocity, a_max=self.max_angular_velocity)

                print(f"{linear_velocity=}")
                print(f"{angular_velocity=}")
                self.move_velocity(linear_velocity, angular_velocity)

                print("-" * 20)

                rospy.sleep(1)
                self.reset_velocity()
            else:
                print("Retrieving current pose")

        print("Turn in place")
        # Stage 2
        while not rospy.is_shutdown() and not self.at_angle_goal(self.curr_pose, goal):
            angle_error = goal[2] - self.curr_pose[2] # Difference between orientation and the angle btw the robot and the goal

            if np.abs(goal[2]) > np.pi / 2 and np.abs(self.curr_pose[2]) > np.pi / 2 and np.sign(goal[2]) != np.sign(self.curr_pose):
                angle_error = (-1 * np.sign(angle_error) * 360) + angle_error

            angular_velocity = np.clip(self.turn_in_place_gain * angle_error, a_min=-self.max_angular_velocity, a_max=self.max_angular_velocity)

            print(f"{self.curr_pose=}")
            print(f"{angle_error=}")
            print(f"{angular_velocity=}")
            self.move_velocity(0.0, angular_velocity)

            rospy.sleep(1)
            self.reset_velocity()

    
    def pick_goal(self, prompt):
        """
        reset parameter defaults to False: when False, Hal will reset its arm and execute policy assuming it's already in reset position
        """
        self.hal_skills_pick.main(prompt=prompt)
        self.retract_arm_primitive()
    
    def place_goal(self, place_loc):
        """
        reset parameter defaults to False: when False, Hal will reset its arm and execute policy assuming it's already in reset position
        """
        self.initial_arm_place()

        self.hal_skills_place = HalSkillsPlace(place_loc)
        self.hal_skills_place.main()

        self.retract_arm_primitive()
    
    def move_table(self):
        self.initial_arm_move()
        s = rospy.ServiceProxy("/switch_to_navigation_mode", Trigger)
        resp = s()
        print(resp)

        self.timed_move(7.5, 0.1)

    def move_fridge(self):
        self.initial_arm_move()
        s = rospy.ServiceProxy("/switch_to_navigation_mode", Trigger)
        resp = s()
        print(resp)

        self.timed_move(7.5, -0.1)
    
    def grocery_main(self, place_loc):
        self.pick_goal()

        self.move_fridge()

        self.place_goal(place_loc)

        self.move_table()

        self.action_status = SUCCESS

if __name__ == "__main__":
    hal = Hal()

    # hal.pid_move(goal=[-0.7, 0.0, 0.1])
    hal.pid_move(goal=[0.0, 0.0, -0.2])
    
