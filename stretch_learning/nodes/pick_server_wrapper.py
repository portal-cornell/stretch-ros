#!/usr/bin/env python

from __future__ import print_function

from stretch_learning.srv import Pick,PickResponse
from hal_skills import start_skills_from_server
import rospy


RUNNING = -1
SUCCESS = 1
NOT_STARTED = 2

class PickServer:
    def __init__(self):
        # skill specific attributes
        self.skill_name = "pick_pantry"
        self.model_type = "visuomotor_bc"
        self.train_type = "end-eff-img-no-rec"
        
        self.action_status = NOT_STARTED

        rospy.init_node('pick_server_wrapper')
        rospy.loginfo("Pick server node created")
        s = rospy.Service('pick_server', Pick, self.callback)
        rospy.loginfo("Pick server has started")
        rospy.spin()


    def update_status(self, new_status):
        if new_status in [RUNNING, SUCCESS]:
            self.action_status = new_status
        else:
            rospy.loginfo("Invalid status from Pick server")

    def callback(self, req):
        if self.action_status == NOT_STARTED:
            # call hal_skills
            start_skills_from_server(self.skill_name, self.model_type, 
                                     self.train_type, self.update_status)
            self.action_status = RUNNING
    
        return PickResponse(self.action_status)


if __name__ == "__main__":
    PickServer()
