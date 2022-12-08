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

count = 0

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

def transform(p):
    import copy
    q = copy.deepcopy(p)
    q.z = p.y
    q.x = -p.x+1.5
    q.y = p.z-0.5

    return q


def createSphere(p, scale = 0.075):
    global count
    marker = Marker()
    marker.id = count
    marker.lifetime = rospy.Duration()
    marker.header.frame_id = "map"
    marker.type = marker.SPHERE
    marker.scale.x = scale 
    marker.scale.y = scale 
    marker.scale.z = scale 
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 0.0
    marker.color.b = 1.0
    marker.pose.orientation.w = 1.0
    marker.pose.position.x = p.x
    marker.pose.position.y = p.y
    marker.pose.position.z = p.z
    count += 1
    return marker

def createArrow(p1, p2, scale = 0.05):
    global count
    marker = Marker()
    marker.id = count
    marker.lifetime = rospy.Duration()
    marker.header.frame_id = "map"
    marker.type = marker.ARROW
    marker.scale.x = scale
    marker.scale.y = 0
    marker.scale.z = 0
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.points = [p1, p2]
    count += 1
    return marker

class ArmNode():

    def __init__(self):
        pass

    def distance(self, p1, p2):
        dist = (p1.x-p2.x)**2+(p1.y-p2.y)**2+(p1.z-p2.z)**2
        return math.sqrt(dist)

    def create_line_segment(self, p1, p2, scale = 0.05):
        markerArray = MarkerArray()
        count = 0

        marker1 = createSphere(p1)
        markerArray.markers.append(marker1)
        count += 1

        marker2 = createSphere(p2)
        markerArray.markers.append(marker2)
        count += 1

        cylinder = createArrow(p1, p2, scale = scale)
        count += 1
        markerArray.markers.append(cylinder)

        return markerArray

    def callback1(self, data):
        global p1
        p1 = transform(data.pose.position)

    def callback2(self, data):
        global p2
        p2 = transform(data.pose.position)

    def callback3(self, data):
        global p3
        p3 = transform(data.pose.position)

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
            "/wrist_to_elbow", MarkerArray, queue_size=1)
        elbow_to_shoulder_publisher = rospy.Publisher(
            "/elbow_to_shoulder", MarkerArray, queue_size=1)
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if p1 is None or p3 is None or p2 is None:
                continue
            wrist_to_elbow = self.create_line_segment(p1, p2, scale = 0.05)
            wrist_to_elbow_publisher.publish(wrist_to_elbow)
            elbow_to_shoulder = self.create_line_segment(p2, p3, scale = 0.07)
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
