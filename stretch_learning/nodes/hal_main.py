import math
import rospy
from move_ps.basic_navigation import StretchNavigation
import hello_helpers.hello_misc as hm
from std_srvs.srv import Trigger
import torch
import actionlib
import tf2_ros


from hal_skills_odom_integrate import HalSkills
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryGoal, FollowJointTrajectoryAction
from stretch_funmap.navigate import MoveBase 


device = "cuda" if torch.cuda.is_available() else "cpu"

pantry = (2.90585, 0.96955, math.pi / 2) # actual pantry pos
# pantry = (2.00585, 0.96955, math.pi / 2)# experimentation
table = ( 1.1854, 1.8176, math.pi)
home = (0.1, 0, 0)

TABLE_HEIGHT = 0.66     # arbitrary table height


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

        self.hal_skills = HalSkills()
        self.hal_skills.subscribe() 

        self.nav = StretchNavigation()
        self.rate = 10.0

        
        # self.hal_skills.register()
        # this following line is to execute the policy
        # node.main()

        # this is to execute move
        # s = rospy.ServiceProxy("/switch_to_navigation_mode", Trigger)
        # resp = s()
        # print(resp)

        # self.move_base = MoveBase(self) 
         


    def pick_goal(self, reset):
        """
        reset parameter defaults to False: when False, Hal will reset its arm and execute policy assuming it's already in reset position
        """

        # self.hal_skills.goal_tensor[3:5] = self.hal_skills.odom_to_js(
        #     self.hal_skills.goal_tensor[3:5]
        # )
        self.hal_skills.main(reset=reset)

    def move_pantry(self):
        """
        !!!! MAKE SURE ARM IS RESETTED BEFORE EXECUTING MOVE
        """
        # this is to execute move
        s = rospy.ServiceProxy("/switch_to_navigation_mode", Trigger)
        resp = s()
        print(resp)

        self.nav.go_to(pantry, self.move_pantry_callback)  # go roughly to the table

    def move_pantry_callback(self, status, resuslt):
        print("at pantry")

    def move_table(self):
        """
        !!!! MAKE SURE ARM IS RESETTED BEFORE EXECUTING MOVE
        """
        # this is to execute move
        s = rospy.ServiceProxy("/switch_to_navigation_mode", Trigger)
        resp = s()
        print(resp)

        self.nav.go_to(table, self.move_table_callback)  # go roughly to the table

    def move_table_callback(self, status, resuslt):
        print("at table")

    def move_home(self):
        """
        !!!! MAKE SURE ARM IS RESETTED BEFORE EXECUTING MOVE
        """
        # this is to execute move
        s = rospy.ServiceProxy("/switch_to_navigation_mode", Trigger)
        resp = s()
        print(resp)

        self.nav.go_to(home, self.move_home_callback)  # go roughly to the table

    def move_home_callback(self, status, resuslt):
        print("at home")

    def reset_pick_pantry(self):
        print('resetting pick pantry')
        rate = rospy.Rate(self.rate)
        self.pick_pantry_initial_config(rate)

    def move_arm_pick_pantry(self):
        rospy.loginfo("Set arm")
        self.pick_starting_height = 0.998
        self.joint_lift_index = self.hal_skills.joint_states.name.index("joint_lift")
        pose = {
            "wrist_extension": 0.01,
            "joint_lift": self.pick_starting_height - 0.35,  # for cabinet -0.175
            "joint_wrist_pitch": 0.2,
            "joint_wrist_yaw": -0.09,
        }
        self.base_gripper_yaw = -0.09
        self.gripper_len = 0.22
        self.move_to_pose(pose)
        return True

    def move_head_pick_pantry(self):
        print('in move_head_pick_pantry')
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

        print('in pick pantry initial config')
        while not done_head_pan:
            if self.hal_skills.joint_states:
                print('got joint states')
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

    def place_table(self): 
        s = rospy.ServiceProxy("/switch_to_position_mode", Trigger)
        resp = s()
        print(resp)

        # retract arm and go to height
        rospy.loginfo("Retract arm")
        pose = {
            "wrist_extension": 0.01,
            "joint_lift": 0.93,  # for cabinet -0.175
            "joint_wrist_pitch": 0.2,
            "joint_wrist_yaw": -0.09,
        }
        self.move_to_pose(pose)
        # move left 
        
        # self.move_base.forward(0.1, publish_visualizations=False, detect_obstacles=False)

        self.move_table() 

        # go to height
        pose = {
            "wrist_extension": TABLE_HEIGHT,
            "joint_lift": 0.6,  # for cabinet -0.175
        }
        self.move_to_pose(pose)


if __name__ == "__main__":
    hal = Hal()
    hal.reset_pick_pantry()
    is_reset = True

    hal.move_pantry() 
    hal.pick_goal(reset=(not is_reset))
    hal.place_table()
    exit()

    command = '0.336,0.246,0.85'
    goal_pos = command.split(",")
    goal_pos = list(map(float, goal_pos))
    hal.pick_goal(goal_pos, reset=not is_reset)
    is_reset = False

    # while True:
    #     command = input()
    #     command = command.split(" ")

    #     if command[0] == "pick":
    #         goal_pos = command[1].split(",")
    #         hal.pick_goal(goal_pos, reset=not is_reset)
    #         is_reset = False
    #     elif command == "move":
    #         if not is_reset:
    #             hal.reset_pick_pantry()
    #             is_reset = True

    #         if command[1] == "pantry":
    #             hal.move_pantry()
    #         elif command[1] == "home":
    #             hal.move_home()
    #     else:
    #         print("command not supported")
