#!/usr/bin/env python3

import numpy as np
import scipy.ndimage as nd
import scipy.signal as si
import cv2
import math
import stretch_funmap.max_height_image as mh
import stretch_funmap.segment_max_height_image as sm
import stretch_funmap.ros_max_height_image as rm
import hello_helpers.hello_misc as hm
import ros_numpy as rn
import rospy
import os



class CustomGraspPlanner():
    def __init__(self, tf2_buffer, debug_directory=None):
        self.debug_directory = debug_directory
        print('CustomGraspPlanner __init__: self.debug_directory =', self.debug_directory)
        
        # # Define the volume of interest for planning using the current
        # # view.
        
        # # How far to look ahead.
        # look_ahead_distance_m = 2.0
        # # Robot's width plus a safety margin.
        # look_to_side_distance_m = 1.3

        # m_per_pix = 0.006
        # pixel_dtype = np.uint8 

        # # stretch (based on HeadScan in mapping.py)
        # robot_head_above_ground = 1.13
        # # After calibration, the floor is lower for stretch than for
        # # Django, so I've lowered the acceptable floor range even
        # # more. This is merits more thought. Is there something
        # # wrong with the calibration or is this to be expected?
        # # How consistent will it be with different floor types?
        # # How will the robot handle floor slope due to calibration
        # # / hardware issues?
        # lowest_distance_below_ground = 0.03
        # voi_height_m = robot_head_above_ground + lowest_distance_below_ground

        # robot_right_edge_m = 0.2
        # voi_side_x_m = 2.0 * look_to_side_distance_m
        # voi_side_y_m = look_ahead_distance_m
        
        # voi_axes = np.identity(3)
        # voi_origin = np.array([-(voi_side_x_m/2.0), -(voi_side_y_m + robot_right_edge_m), -lowest_distance_below_ground])

        # # Define the VOI using the base_link frame
        # old_frame_id = 'base_link'
        # voi = rm.ROSVolumeOfInterest(old_frame_id, voi_origin, voi_axes, voi_side_x_m, voi_side_y_m, voi_height_m)
        # # Convert the VOI to the map frame to handle mobile base changes
        # new_frame_id = 'map'
        # lookup_time = rospy.Time(0) # return most recent transform
        # timeout_ros = rospy.Duration(0.1)
        # stamped_transform =  tf2_buffer.lookup_transform(new_frame_id, old_frame_id, lookup_time, timeout_ros)
        # points_in_old_frame_to_new_frame_mat = rn.numpify(stamped_transform.transform)
        # voi.change_frame(points_in_old_frame_to_new_frame_mat, new_frame_id)
        # self.voi = voi
        # self.max_height_im = rm.ROSMaxHeightImage(self.voi, m_per_pix, pixel_dtype)
        # print(self.max_height_im.voi)
        # self.max_height_im.print_info()
        # self.updated = False

        self.point_cloud = None

    def point_cloud_callback(self, point_cloud):
        self.point_cloud = point_cloud

    def main(self, node_name, node_topic_namespace, wait_for_first_pointcloud=True):
        rospy.init_node(node_name)
        self.node_name = rospy.get_name()
        rospy.loginfo("{0} started".format(self.node_name))

        # self.trajectory_client = actionlib.SimpleActionClient('/stretch_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        server_reached = self.trajectory_client.wait_for_server(timeout=rospy.Duration(60.0))
        if not server_reached:
            rospy.signal_shutdown('Unable to connect to arm action server. Timeout exceeded.')
            sys.exit()
        
        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer)
        
        self.point_cloud_subscriber = rospy.Subscriber('/camera/depth/color/points', PointCloud2, self.point_cloud_callback)
        self.point_cloud_pub = rospy.Publisher('/' + node_topic_namespace + '/point_cloud2', PointCloud2, queue_size=1)

        rospy.wait_for_service('/stop_the_robot')
        rospy.loginfo('Node ' + self.node_name + ' connected to /stop_the_robot service.')
        self.stop_the_robot_service = rospy.ServiceProxy('/stop_the_robot', Trigger)
        
        if wait_for_first_pointcloud:
            # Do not start until a point cloud has been received
            point_cloud_msg = self.point_cloud
            print('Node ' + node_name + ' waiting to receive first point cloud.')
            while point_cloud_msg is None:
                rospy.sleep(0.1)
                point_cloud_msg = self.point_cloud
            print('Node ' + node_name + ' received first point cloud, so continuing.')
