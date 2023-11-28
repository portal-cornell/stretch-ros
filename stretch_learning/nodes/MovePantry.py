import math
import rospy
from move_ps.basic_navigation import StretchNavigation
import hello_helpers.hello_misc as hm
from std_srvs.srv import Trigger

from hal_skills_odom import HalSkills

pantry = (-1.695, 1.516, -math.pi / 2)
home = (0, 0, 0)


class Mover(hm.HelloNode):
    def __init__(self):
        rospy.init_node("hal_move_node")
        self.node_name = rospy.get_name()
        rospy.loginfo("{0} started".format(self.node_name))

        print("get here")
        self.nav = StretchNavigation()
        print("get here")
        self.rate = 10.0

        s = rospy.ServiceProxy("/switch_to_navigation_mode", Trigger)
        resp = s()
        print(resp)

    def callback(self, status, resuslt):
        goal_pos = ["0.32", "-0.2", "0.83", "0", "0", "0"]
        node = HalSkills(
            "pick_salt", "visuomotor_bc", "bc_oracle", goal_pos, "2d", True
        )
        node.main()
        print("done")

    def main(self):
        # self.pick_pantry_initial_config(rospy.Rate(self.rate))
        coords = pantry
        self.nav.go_to(coords, self.callback)  # go roughly to the table


m = Mover()
m.main()
