#!/usr/bin/env python3

import rospy

from hal_main_test import Hal
from stretch_learning.srv import (
    HalMove,
    HalPick,
    HalPlace,
    HalHandover,
    HalMoveResponse,
    HalPickResponse,
    HalPlaceResponse,
)

RUNNING = -1
SUCCESS = 1
NOT_STARTED = 2

class TaskServer:
    def __init__(self):
        self.hal = Hal()
        self.action_status = NOT_STARTED

        self.move_location_to_function = {
            "fridge": self.hal.move_fridge,
            "table": self.hal.move_table,
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
        if self.action_status == NOT_STARTED:
            self.action_status = RUNNING
            if hasattr(req, "location"):
                self.move_location_to_function[req.location.lower()]()
            elif req == "table":
                self.hal.move_table()
            elif req == "fridge":
                self.hal.move_fridge()
            self.action_status = NOT_STARTED
            return HalMoveResponse(SUCCESS)
        return HalMoveResponse(self.action_status)

    def handle_place(self, req):
        print("Place callback called")
        if self.action_status == NOT_STARTED:
            self.action_status = RUNNING
            self.hal.place_goal(req)
            self.action_status = NOT_STARTED
            return HalPlaceResponse(SUCCESS)
        return HalPlaceResponse(self.action_status)

    def handle_pick(self):
        print("Pick callback called")
        if self.action_status == NOT_STARTED:
            self.action_status = RUNNING
            self.hal.pick_goal()
            self.action_status = NOT_STARTED
            return HalPickResponse(SUCCESS)
        return HalPickResponse(self.action_status)

    def retract_for_move(self):
        self.hal.retract_arm_primitive()


if __name__ == "__main__":
    ts = TaskServer()
    ts.handle_move("table")
    
