#!/usr/bin/env python

import rospy
from tf2_ros import TransformBroadcaster, TransformStamped
import numpy as np
import tf.transformations as tft

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

if __name__ == '__main__':
    # Initialize the ROS node
    rospy.init_node('frame_transformer')

    # Create a transform broadcaster
    broadcaster = TransformBroadcaster()

    # Define the frames for the motion capture system and robot
    mocap_frame = 'world'
    robot_frame = 'map'
    point1 = [0, 0.0, 0.0]
    point2 = [2.10, 0, 1.03]
    # point
    transformation = tft.translation_matrix(point2) @ tft.inverse_matrix(tft.translation_matrix(point1))
    translation = tft.translation_from_matrix(transformation)
    rotation = tft.quaternion_from_matrix(transformation)
    # print(transformation)
    # input()
    # Broadcast the transformation between the motion capture and robot frames
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        # # Define the transformation from the motion capture to robot frame
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = robot_frame
        transform.child_frame_id = mocap_frame
        transform.transform.translation.x = translation[0]
        transform.transform.translation.y = translation[1]
        transform.transform.translation.z = translation[2]
        transform.transform.rotation.x = rotation[0]
        transform.transform.rotation.y = rotation[1]
        transform.transform.rotation.z = rotation[2]
        transform.transform.rotation.w = rotation[3]

        # Broadcast the transformation
        broadcaster.sendTransform(transform)

        rate.sleep()