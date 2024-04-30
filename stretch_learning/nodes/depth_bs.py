#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import CameraInfo, PointCloud2
from sensor_msgs import point_cloud2 as pc2
import cv2
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image
from std_msgs.msg import Header
import sys
import os
import numpy as np
from std_msgs.msg import String
import pyrealsense2 as rs2
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

if not hasattr(rs2, "intrinsics"):
    import pyrealsense2.pyrealsense2 as rs2
# from depth_to_color import ImageProcessor
# from perception.msg import Object, ObjectList
from geometry_msgs.msg import PointStamped
from std_msgs.msg import (
    String,
    Int32,
    Int32MultiArray,
    MultiArrayLayout,
    MultiArrayDimension,
)
import tf

CAMERA_DIR = "left"


# Listens to the depth topics and projects pixels to 3D
# Publishes point cloud to object_point_cloud topic
class ImageListener:
    def __init__(self, depth_image_topic, depth_info_topic, header):
        self.bridge = CvBridge()
        self.intrinsics = None
        self.image_processor = None
        self.point_cloud_map = {}
        self.color_image = None
        self.depth_image = None
        self.center_array = None
        self.result = None
        self.mask_array = None
        self.header = header
        self.listener = tf.TransformListener()

        self.sub = rospy.Subscriber(
            depth_image_topic, msg_Image, self.imageDepthCallback
        )
        self.sub_info = rospy.Subscriber(
            depth_info_topic, CameraInfo, self.imageDepthInfoCallback
        )
        self.color_sub = rospy.Subscriber(
            f"/head_camera/color/image_raw", msg_Image, self.colorImageCallback
        )
        self.center_sub = rospy.Subscriber(
            f"/hal_prediction_node/center", String, self.centerCallback
        )
        self.mask = rospy.Subscriber(
            f"/hal_prediction_node/mask", msg_Image, self.maskCallback
        )

        self.xyz_pub = rospy.Publisher(
            f"/hal_prediction_node/xyz", String, queue_size=1
        )

        self.xyz_point_pub = rospy.Publisher(
            "/hal_prediction_node/point", PointStamped, queue_size=1
        )
        # visualize this topic
        self.base_link_point = rospy.Publisher(
            "/hal_prediction_node/base_link_point", PointStamped, queue_size=1
        )
        self.point_cloud_publisher = rospy.Publisher(
            "/mask_point_cloud", PointCloud2, queue_size=1
        )

    def imageDepthCallback(self, data):
        self.depth_image = self.bridge.imgmsg_to_cv2(data, data.encoding)

    def centerCallback(self, data):
        center_list = list(map(int, data.data.split(" ")))
        self.center_array = np.array(center_list)

    def maskCallback(self, data):
        self.mask_array = self.bridge.imgmsg_to_cv2(data, data.encoding)

    def imageDepthInfoCallback(self, cameraInfo):
        if self.intrinsics is not None:
            return
        self.intrinsics = rs2.intrinsics()
        self.intrinsics.width = cameraInfo.width
        self.intrinsics.height = cameraInfo.height
        self.intrinsics.ppx = cameraInfo.K[2]
        self.intrinsics.ppy = cameraInfo.K[5]
        self.intrinsics.fx = cameraInfo.K[0]
        self.intrinsics.fy = cameraInfo.K[4]
        if cameraInfo.distortion_model == "plumb_bob":
            self.intrinsics.model = rs2.distortion.brown_conrady
        elif cameraInfo.distortion_model == "equidistant":
            self.intrinsics.model = rs2.distortion.kannala_brandt4
        self.intrinsics.coeffs = [i for i in cameraInfo.D]

    def colorImageCallback(self, image_msg):
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")

        # Rotate image 180 degrees
        new_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        new_img = Image.fromarray(cv2.rotate(new_img, cv2.ROTATE_180))
        self.color_image = new_img

    def filter_outliers(self, point_cloud):
        # Calculate the mean and standard deviation
        mean = np.mean(point_cloud, axis=0)
        std_dev = np.std(point_cloud, axis=0)

        # Calculate the Z-scores
        z_scores = np.abs((point_cloud - mean) / std_dev)

        # Filter out outliers
        filtered_point_cloud = point_cloud[(z_scores < 2).all(axis=1)]

        return filtered_point_cloud

    def get_pcl(self):
        print(self.intrinsics, self.center_array, self.mask_array)
        if self.intrinsics is None:
            return
        if self.center_array is None:
            return
        if self.mask_array is None:
            return
        # xs, ys = np.nonzero(self.mask_array)
        # self.point_cloud_map = []
        # for idx in range(len(xs)):
        #     i, j = xs[idx], ys[idx]
        #     result = (
        #         np.array(
        #             rs2.rs2_deproject_pixel_to_point(
        #                 self.intrinsics, [j, i], self.depth_image[i, j]
        #             )
        #         )
        #         / 1000
        #     )
        #     self.point_cloud_map.append([result[2], -result[0], -result[1]])
            # self.point_cloud_map.append([result[2], -1 * result[0], -1 * result[1]])
        result = (
            np.array(
                rs2.rs2_deproject_pixel_to_point(
                    self.intrinsics,
                    [self.center_array[1], self.center_array[0]],
                    self.depth_image[self.center_array[0], self.center_array[1]],
                )
            )
            / 1000
        )

        self.result = result

    def publish_pcls(self):
        if self.result is None:
            return
        print(self.result[0])
        print(self.result[1])
        print(self.result[2])
        point = PointStamped()
        point.header.frame_id = "camera_aligned_depth_to_color_frame"
        point.point.x = self.result[2]
        point.point.y = -self.result[0]
        point.point.z = -self.result[1]
        # point_in_target_frame = self.listener.transformPoint("base_link", point)
        point_in_target_frame = self.listener.transformPoint("odom", point)
        print("HFUH")
        self.base_link_point.publish(point_in_target_frame)
        self.xyz_point_pub.publish(point)
        self.xyz_pub.publish(f"{self.result[0]}, {self.result[1]}, {self.result[2]}")
        # cloud = pc2.create_cloud_xyz32(self.header, self.point_cloud_map)
        # self.point_cloud_publisher.publish(cloud)


def main():
    depth_image_topic = f"/head_camera/aligned_depth_to_color/image_raw"
    depth_info_topic = f"/head_camera/aligned_depth_to_color/camera_info"
    rate = rospy.Rate(5)  # ROS Rate at 5Hz

    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = f"camera_aligned_depth_to_color_frame"
    listener = ImageListener(depth_image_topic, depth_info_topic, header)
    # listener.image_processor = ImageProcessor(texts=list(listener.object_to_publisher_map.keys()))

    import time

    while not rospy.is_shutdown():
        print("ushdsfu")
        listener.get_pcl()
        listener.publish_pcls()
        rate.sleep()


if __name__ == "__main__":
    print(f'here')
    node_name = "image_listener_node"
    rospy.init_node(node_name)
    main()
