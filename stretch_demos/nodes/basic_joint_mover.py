import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryGoal, FollowJointTrajectoryAction
from trajectory_msgs.msg import JointTrajectoryPoint

class JointMover():
    """
    A class that sends multiple joint trajectory goals to the stretch robot.
    """
    def __init__(self):
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

        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.trajectory.joint_names = joint_name_list
        trajectory_goal.trajectory.points = [point]
        trajectory_goal.trajectory.header.stamp = rospy.Time(0.0)
        trajectory_goal.trajectory.header.frame_id = 'base_link'

        self.trajectory_client.send_goal(trajectory_goal)
        
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

    def reset_arm(self):
        target_position_list = [0.3, 
                                3.232259253198204e-05, 3.232259253198204e-05, 3.232259253198204e-05, 3.232259253198204e-05, 
                                2.9995715989780476, 
                                -0.013805827090970769]
        joint_name_list = ['joint_lift', 
                                'joint_arm_l3', 'joint_arm_l2', 'joint_arm_l1', 'joint_arm_l0', 
                                'joint_wrist_yaw',
                                'joint_wrist_pitch']

        self.issue_one_target_command(joint_name_list, target_position_list, "reset arm")       