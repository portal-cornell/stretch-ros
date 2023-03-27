#!/usr/bin/env python

from __future__ import print_function

from engineered_skills.srv import MoveAction,MoveActionResponse
import rospy
import time

RUNNING = -1
SUCCESS = 1

class MoveActionServer:
    def __init__(self):
        # TODO: these are just dummy constraints to mimic that it takes time to do an action
        self._start_time = 0
        self._time_constr = 5

        rospy.init_node('move_action_server')
        rospy.loginfo("Move Action server node created")
        s = rospy.Service('move_action', MoveAction, self.callback)
        rospy.loginfo("Move Action server has started")
        rospy.spin()

    def callback(self, req):
        if self._start_time == 0:
            # the action has not started yet
            # print(f"    [starting] go_to({req.loc})")
            rospy.loginfo(f"    [starting] go_to({req.loc})")
            self._start_time = time.time()
            return MoveActionResponse(RUNNING)
        else:
            if (time.time() - self._start_time) < self._time_constr:
                # still executing the action
                return MoveActionResponse(RUNNING)
            else:
                self._start_time = 0  # reset for next time use
                # print(f"    [success] go_to({req.loc})")
                rospy.loginfo(f"    [success] go_to({req.loc})")
                return MoveActionResponse(SUCCESS)

if __name__ == "__main__":
    MoveActionServer()
