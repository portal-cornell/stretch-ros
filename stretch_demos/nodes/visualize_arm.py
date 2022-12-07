#!/usr/bin/env python3

import numpy as np
import argparse as ap
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Point
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped
import math

# p1, p2, p3 are the rigid body points
global p1
global p2
global p3

# create an object called line_segment_1 and line_segment_2, each with two rigid bodies
p1 = None
p2 = None
p3 = None


def euler_to_quaternion(euler):
    roll, pitch, yaw = euler
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - \
        np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - \
        np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]


class ArmNode():

    def __init__(self):
        pass

    def distance(self, p1, p2):
        dist = (p1.x-p2.x)**2+(p1.y-p2.y)**2+(p1.z-p2.z)**2
        return math.sqrt(dist)

    def create_line_segment(self, p1, p2):
        markerArray = MarkerArray()
        count = 0

        marker1 = Marker()
        marker1.id = count
        marker1.lifetime = rospy.Duration()
        marker1.header.frame_id = "map"
        marker1.type = marker1.SPHERE
        marker1.action = marker1.ADD
        marker1.scale.x = 0.1
        marker1.scale.y = 0.1
        marker1.scale.z = 0.1
        marker1.color.a = 1.0
        marker1.color.r = 1.0
        marker1.color.g = 1.0
        marker1.color.b = 0.0
        marker1.pose.orientation.w = 1.0
        marker1.pose.position.x = p1.x
        marker1.pose.position.y = p1.y
        marker1.pose.position.z = p1.z
        markerArray.markers.append(marker1)
        count += 1

        marker2 = Marker()
        marker2.id = count
        marker2.lifetime = rospy.Duration()
        marker2.header.frame_id = "map"
        marker2.type = marker2.SPHERE
        marker2.action = marker2.ADD
        marker2.scale.x = 0.1
        marker2.scale.y = 0.1
        marker2.scale.z = 0.1
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 1.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = p2.x
        marker2.pose.position.y = p2.y
        marker2.pose.position.z = p2.z
        markerArray.markers.append(marker2)
        count += 1

        cylinder = Marker()
        cylinder.id = count
        cylinder.header.stamp = rospy.Time.now()
        cylinder.header.frame_id = "map"
        cylinder.type = cylinder.ARROW
        cylinder.scale.x = 0.05
        cylinder.scale.y = 0
        cylinder.scale.z = 0
        cylinder.points = [p1, p2]
        cylinder.color.r = 1.0
        cylinder.color.a = 1.0
        count += 1
        markerArray.markers.append(cylinder)

        return markerArray

    def callback1(self, data):
        global p1
        p1 = data.pose.position

    def callback2(self, data):
        global p2
        p2 = data.pose.position

    def callback3(self, data):
        global p3
        p3 = data.pose.position

    def main(self):
        global p1
        global p2
        global p3
        rospy.init_node('register', anonymous=True)
        rospy.Subscriber("/vrpn_client_node/wrist/pose",
                         PoseStamped, self.callback1)
        rospy.Subscriber("/vrpn_client_node/elbow/pose",
                         PoseStamped, self.callback2)
        rospy.Subscriber("/vrpn_client_node/shoulder/pose",
                         PoseStamped, self.callback3)

        wrist_to_elbow_publisher = rospy.Publisher(
            "/wrist_to_elbow", MarkerArray)
        elbow_to_shoulder_publisher = rospy.Publisher(
            "/elbow_to_shoulder", MarkerArray)
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if p1 is None or p3 is None or p2 is None:
                continue
            wrist_to_elbow = self.create_line_segment(p1, p2)
            wrist_to_elbow_publisher.publish(wrist_to_elbow)
            elbow_to_shoulder = self.create_line_segment(p2, p3)
            elbow_to_shoulder_publisher.publish(elbow_to_shoulder)
            rate.sleep()


if __name__ == '__main__':
    try:
        parser = ap.ArgumentParser(description='Create line segments')
        args, unknown = parser.parse_known_args()
        node = ArmNode()
        node.main()
    except rospy.ROSInterruptException:
        pass
