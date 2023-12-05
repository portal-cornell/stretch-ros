import math
import rospy
from utils.basic_navigation import StretchNavigation
import hello_helpers.hello_misc as hm

from std_srvs.srv import Trigger

pantry = (-1.695, 1.516, -math.pi / 2)
home = (0.3, 0, 0)


class Mover(hm.HelloNode):
    def __init__(self):
        rospy.init_node("hal_move_node")
        self.node_name = rospy.get_name()
        rospy.loginfo("{0} started".format(self.node_name))

        print("get here")
        self.nav = StretchNavigation()
        print("get here")
        self.rate = 5.0

    def callback(self, status, result):
        print("done")

    def main(self):
        # self.pick_pantry_initial_config(rospy.Rate(self.rate))
        coords = home
        self.nav.go_to(coords, self.callback)  # go roughly to the table


m = Mover()
m.main()
