#!/usr/bin/env python3

import rospy
import sys
import time
import numpy as np

from hal_main import Hal
from stretch_learning.msg import PickPrompt
from stretch_learning.srv import (
    HalMove,
    HalPick,
    HalPlace,
    HalHandover,
    HalMoveResponse,
    HalPickResponse,
    HalPlaceResponse,
    HalHandoverResponse,
)
from geometry_msgs.msg import PoseStamped, Point, Quaternion, PoseWithCovariance
from std_msgs.msg import String
from std_srvs.srv import Trigger

# from dora_perception.msg import Object

RUNNING = -1
SUCCESS = 1
# FAILURE = 0
NOT_STARTED = 2


class TaskServer:
    def __init__(self):
        self.hal = Hal()
        self.action_status = NOT_STARTED

        self.move_location_to_function = {
            "home": self.hal.move_home,
            "pantry": self.hal.move_pantry,
            "table": self.hal.move_table,
            "shelf": self.hal.move_shelf,
        }
        self.place_location_to_function = {
            "table": self.hal.place_table,
            "shelf": self.hal.place_shelf,
        }
        print("init")

    def start(self):
        # start services
        s_move = rospy.Service("HalMove", HalMove, self.handle_move)
        s_place = rospy.Service("HalPlace", HalPlace, self.handle_place)
        s_pick = rospy.Service("HalPick", HalPick, self.handle_pick)
        s_handover = rospy.Service("HalHandover", HalHandover, self.handle_handover)

        rospy.spin()

    def handle_move(self, req):
        print("Move callback called")
        # rospy.sleep(15)
        if self.action_status == NOT_STARTED:
            # return HalMoveResponse(SUCCESS)
            self.action_status = RUNNING
            self.hal.retract_arm_primitive()
            if hasattr(req, "location"):
                self.move_location_to_function[req.location.lower()]()
            elif req == "table":
                self.hal.move_table()
            elif req == "pantry":
                self.hal.move_pantry()
            elif req == "shelf":
                self.hal.move_shelf()
            self.action_status = NOT_STARTED
            return HalMoveResponse(SUCCESS)
        return HalMoveResponse(self.action_status)

    def handle_place(self, req):
        print("Place callback called")
        if self.action_status == NOT_STARTED:
            # return HalPlaceResponse(SUCCESS)
            self.action_status = RUNNING
            if hasattr(req, "location"):
                self.place_location_to_function[req.location.lower()]()
            elif req == "table":
                self.hal.place_table()
            elif req == "shelf":
                self.hal.place_shelf()
            # self.hal.retract_arm_primitive()
            self.action_status = NOT_STARTED
            return HalPlaceResponse(SUCCESS)
        return HalPlaceResponse(self.action_status)

    def handle_pick(self, req):
        print("Pick callback called")

        if self.action_status == NOT_STARTED:
            # return HalPickResponse(SUCCESS)
            self.action_status = RUNNING
            # self.hal.reset_pick_pantry()
            if hasattr(req, "object"):
                self.hal.pick_goal(prompt=req.object.lower())
            else:
                self.hal.pick_goal(prompt=req.lower())
            self.action_status = NOT_STARTED
            return HalPickResponse(SUCCESS)
        return HalPickResponse(self.action_status)

    def handle_handover(self, _):
        print("Handover callback called")
        if self.action_status == NOT_STARTED:
            self.action_status = RUNNING
            self.hal.handover(None)
            self.action_status = NOT_STARTED
            return HalHandoverResponse(SUCCESS)
        return HalHandoverResponse(self.action_status)

    def drop_obj(self):
        pose_sequence = [
            {
                "joint_lift": 0.96,  # for cabinet -0.175
            },
            {"wait": 0.1},
            {
                "wrist_extension": 0.28,
            },
            {"wait": 0.1},
            {
                "joint_lift": 0.92,  # for cabinet -0.175
            },
            {"wait": 0.1},
            {
                "joint_gripper_finger_left": 0.22,
            },
        ]

        for pose in pose_sequence:
            if "wait" in pose:
                rospy.sleep(pose["wait"])
            else:
                self.hal.move_to_pose(pose)
        return True

    def retract_for_move(self):
        self.hal.retract_arm_primitive()


if __name__ == "__main__":

    # os.system("roslaunch engineered_skills demo_sp23.launch map_yaml:=/home/strech/stretch_user/maps/12012023.yaml")
    ts = TaskServer()
    # ts.handle_move("pantry")
    # ts.handle_pick("blue dish soap")
    # ts.handle_pick("kosher salt with blue cap")
    # ts.start()

    # exit()
    import sys

    # import subprocess

    print("here")
    if sys.argv[1] == "1":
        print("1")
        ts.handle_pick("red ketchup")
    elif sys.argv[1] == "2":
        print("2")
        ts.handle_pick("chicken broth box")
        #
        # ts.handle_move("table")
        # ts.handle_place("table")
        # time.sleep(5)

        # ts.handle_handover(None)
        # ts.handle_move("shelf")
        # ts.hal.hal_skills.move_shelf_primitive()
        # ts.handle_place("shelf")
    else:
        ts.drop_obj()
