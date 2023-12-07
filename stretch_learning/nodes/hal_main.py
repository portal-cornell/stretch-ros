import math
import rospy
from move_ps.basic_navigation import StretchNavigation
import hello_helpers.hello_misc as hm
from std_srvs.srv import Trigger
import torch
import actionlib
import tf2_ros
import time


from hal_skills_final import HalSkillsNode

# from hal_skills_pred_odom import HalSkills
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryGoal, FollowJointTrajectoryAction
from stretch_funmap.navigate import MoveBase


device = "cuda" if torch.cuda.is_available() else "cpu"

# pantry_mustard = (2.72585, 1.27955, math.pi / 2)
# pantry_ketchup = (2.72585, 0.90955, math.pi / 2)
# pantry = pantry_mustard
# pantry = (2.83585, 1.08955, math.pi / 2)  # actual pantry pos
PANTRY_LOC = [2.80, 1.08, math.pi / 2]
# pantry = (2.00585, 0.96955, math.pi / 2)# experimentation
BETWEEN_TABLE_PANTRY_LOC = (2.0, 1.2, -math.pi / 4)
TABLE_LOC = (1.55, 2.01, math.pi)
HOME_LOC = (0.1, 0, 0)

TABLE_HEIGHT = 0.98  # arbitrary table height
TABLE_WRIST_EXT = 0.38
# skill status
RUNNING = -1
SUCCESS = 1
NOT_STARTED = 2


class Hal(hm.HelloNode):
    def __init__(self):
        hm.HelloNode.__init__(self)
        # self.main("hal_move_skills_node", "hal_move_skills_node", wait_for_first_pointcloud=False)

        self.trajectory_client = actionlib.SimpleActionClient(
            "/stretch_controller/follow_joint_trajectory", FollowJointTrajectoryAction
        )

        rospy.init_node("hal_move_skills_node")
        self.node_name = rospy.get_name()
        rospy.loginfo("{0} started".format(self.node_name))

        self.hal_skills = HalSkillsNode()
        self.hal_skills.subscribe()

        self.nav = StretchNavigation()
        self.rate = 10.0
        self.current_location = HOME_LOC

        self.action_status = NOT_STARTED
        self.pick_prompt = None

        self.pantry_moves = 0

    def pick_goal(self, prompt, reset=False):
        """
        reset parameter defaults to False: when False, Hal will reset its arm and execute policy assuming it's already in reset position
        """
        prompt = prompt.lower()
        if "mustard" in prompt:
            prompt = "yellow mustard"
        elif "reslish" in prompt:
            prompt = "dill relish"

        self.hal_skills.main(reset=reset, prompt=prompt)

    def spin_towards_goal(self, goal_loc):
        """
        spin robot so that it is in the direction of the goal
        """
        return
        rospy.set_param("/move_base/TrajectoryPlannerROS/yaw_goal_tolerance", 0.20)
        curr_x, curr_y, _ = self.current_location
        goal_x, goal_y, _ = goal_loc
        centerred_x, centerred_y = goal_x - curr_x, goal_y - curr_y
        new_orientation = math.atan2(centerred_y, centerred_x)
        print(f"Spinning towards {new_orientation}")
        self.nav.go_to((curr_x, curr_y, new_orientation), self.orient_callback)

    def orient_callback(self, status, result):
        self.action_status = SUCCESS

    def move_between_table_pantry_callback(self, status, result):
        self.current_location = BETWEEN_TABLE_PANTRY_LOC

    def move_pantry(self):
        """
        !!!! MAKE SURE ARM IS RESETTED BEFORE EXECUTING MOVE
        """
        # this is to execute move
        s = rospy.ServiceProxy("/switch_to_navigation_mode", Trigger)
        resp = s()
        rospy.set_param("/move_base/TrajectoryPlannerROS/xy_goal_tolerance", 0.05)
        rospy.set_param(
            "/move_base/TrajectoryPlannerROS/yaw_goal_tolerance", 0.20
        )  # 10 degrees
        print(resp)

        # intermediary point if going to pantry from table
        if self.current_location == TABLE_LOC:
            self.spin_towards_goal(goal_loc=BETWEEN_TABLE_PANTRY_LOC)
            self.nav.go_to(
                BETWEEN_TABLE_PANTRY_LOC, self.move_between_table_pantry_callback
            )

        self.spin_towards_goal(goal_loc=PANTRY_LOC)
        self.nav.go_to(
            PANTRY_LOC, self.move_pantry_callback
        )  # go roughly to the pantry

        # calibrate angles
        rospy.set_param("/move_base/TrajectoryPlannerROS/xy_goal_tolerance", 0.05)
        rospy.set_param(
            "/move_base/TrajectoryPlannerROS/yaw_goal_tolerance", 0.02
        )  # 10 degrees
        rospy.set_param(
            "/move_base/TrajectoryPlannerROS/max_vel_theta", 0.2
        )  # 10 degrees
        rospy.set_param(
            "/move_base/TrajectoryPlannerROS/acc_lim_theta", 0.2
        )  # 10 degrees
        self.nav.go_to(PANTRY_LOC, self.angle_calib_callback)

        # resetting params
        rospy.set_param(
            "/move_base/TrajectoryPlannerROS/yaw_goal_tolerance", 0.09
        )  # reset params
        rospy.set_param(
            "/move_base/TrajectoryPlannerROS/max_vel_theta", 1.0
        )  # 10 degrees
        rospy.set_param(
            "/move_base/TrajectoryPlannerROS/acc_lim_theta", 1.0
        )  # 10 degrees

    def angle_calib_callback(self, status, result):
        print("done angle calibration")

        self.action_stats = SUCCESS

    def move_pantry_callback(self, status, result):
        print("at pantry")
        self.current_location = PANTRY_LOC
        # PANTRY_LOC[1] -= 0.01
        PANTRY_LOC[2] += 0.06
        # self.action_status = SUCCESS

    def move_table(self):
        """
        !!!! MAKE SURE ARM IS RESETTED BEFORE EXECUTING MOVE
        """
        # this is to execute move
        # self.move_table_callback(None, None)
        # return
        rospy.ServiceProxy("/switch_to_position_mode", Trigger)
        self._lift_arm_primitive()
        rospy.set_param("/move_base/TrajectoryPlannerROS/xy_goal_tolerance", 0.05)
        s = rospy.ServiceProxy("/switch_to_navigation_mode", Trigger)
        resp = s()
        print(resp)
        self.spin_towards_goal(goal_loc=TABLE_LOC)
        self.nav.go_to(TABLE_LOC, self.move_table_callback)  # go roughly to the table

    def move_table_callback(self, status, result):
        print("at table")
        self.current_location = TABLE_LOC
        self.action_status = SUCCESS

    def move_home(self):
        """
        !!!! MAKE SURE ARM IS RESETTED BEFORE EXECUTING MOVE
        """
        # this is to execute move
        s = rospy.ServiceProxy("/switch_to_navigation_mode", Trigger)
        resp = s()
        print(resp)
        self.spin_towards_goal(goal_loc=HOME_LOC)
        self.nav.go_to(HOME_LOC, self.move_home_callback)  # go roughly to the table

    def move_home_callback(self, status, result):
        print("at home")
        self.action_status = SUCCESS
        self.current_location = HOME_LOC

    def reset_pick_pantry(self):
        print("resetting pick pantry")
        rate = rospy.Rate(self.rate)
        rospy.ServiceProxy("/switch_to_position_mode", Trigger)
        self.pick_pantry_initial_config(rate)

    def move_arm_pick_pantry(self):
        rospy.loginfo("Set arm")
        self.pick_starting_height = 0.998
        self.joint_lift_index = self.hal_skills.joint_states.name.index("joint_lift")
        pose = {
            "wrist_extension": 0.01,
            "joint_lift": self.pick_starting_height - 0.55,  # for cabinet -0.175
            "joint_wrist_pitch": 0.2,
            "joint_wrist_yaw": -0.09,
        }
        self.base_gripper_yaw = -0.09
        self.gripper_len = 0.22
        self.move_to_pose(pose)
        return True

    def move_head_pick_pantry(self):
        print("in move_head_pick_pantry")
        tilt = -0.4358
        pan = -1.751
        rospy.loginfo("Set head pan")
        pose = {"joint_head_pan": pan, "joint_head_tilt": tilt}
        self.move_to_pose(pose)
        return True

    def open_grip(self):
        point = JointTrajectoryPoint()
        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.goal_time_tolerance = rospy.Time(1.0)
        trajectory_goal.trajectory.joint_names = ["joint_gripper_finger_left"]
        point.positions = [0.22]
        trajectory_goal.trajectory.points = [point]
        self.trajectory_client.send_goal(trajectory_goal)
        grip_change_time = 2
        rospy.sleep(grip_change_time)

    def pick_pantry_initial_config(self, rate):
        done_head_pan = False
        # import pdb; pdb.set_trace()

        print("in pick pantry initial config")
        while not done_head_pan:
            if self.hal_skills.joint_states:
                print("got joint states")
                done_head_pan = self.move_head_pick_pantry()
            rate.sleep()
        print("done with head pan")
        done_initial_config = False
        while not done_initial_config:
            if self.hal_skills.joint_states:
                done_initial_config = self.move_arm_pick_pantry()
            rate.sleep()
        print("done with  move arm")
        self.open_grip()

    def _lift_arm_primitive(self):
        s = rospy.ServiceProxy("/switch_to_position_mode", Trigger)
        resp = s()
        print(resp)

        # retract arm and go to height
        rospy.loginfo("Retract arm")
        pose = {
            "wrist_extension": 0.01,
            "joint_lift": 1.0,  # for cabinet -0.175
            "joint_wrist_pitch": 0.2,
            "joint_wrist_yaw": -0.09,
        }
        self.move_to_pose(pose)

    def retract_arm_primitive(self):
        s = rospy.ServiceProxy("/switch_to_position_mode", Trigger)
        resp = s()
        print(resp)

        # retract arm and go to height
        rospy.loginfo("Retract arm")
        pose = {
            "wrist_extension": 0.01,
        }
        self.move_to_pose(pose)

    def place_table(self):
        # move left
        # self.move_table()
        # go to height
        pose = {
            "wrist_extension": TABLE_WRIST_EXT,
        }
        self.move_to_pose(pose)
        time.sleep(0.5)
        pose = {
            "joint_lift": TABLE_HEIGHT,  # for cabinet -0.175
        }
        self.move_to_pose(pose)
        self.open_grip()
        # retract wrist
        pose = {
            "wrist_extension": 0.01,
        }
        self.move_to_pose(pose)

        self.action_status = SUCCESS


if __name__ == "__main__":
    hal = Hal()
    # hal.reset_pick_pantry()
    # is_reset = True

    # hal.move_pantry()
    # rospy.sleep(2)
    # hal.pick_goal(reset=False, prompt="kosher salt")
    # hal.move_table()
    # hal.place_table()
    # exit()
