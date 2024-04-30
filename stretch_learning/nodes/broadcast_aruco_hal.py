#!/usr/bin/env python
import tf

import rospy
import numpy as np
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped

from scipy.spatial.transform import Rotation as R

from threading import Lock

from cam_local import CamLocal

import pyrealsense2 as rs2


class BroadcastArucoHAL:
    def __init__(self, rgb_topic, cam_info_topic, pc_topic, cam_frame):
        self.lock = Lock()

        self.hal_rgb = None
        self.depth_image = None
        self.camera_info_K = None
        self.camera_info_D = None
        self.intrinsics = None

        self.rgb_topic = rgb_topic
        self.cam_info_topic = cam_info_topic
        self.pc_topic = pc_topic
        self.cam_frame = cam_frame

        self.points = None

        # image transforms
        self.cv_bridge = CvBridge()

        # (x, y, z) projected point of object in aruco frame
        self.aruco_publisher = rospy.Publisher(
            "/broadcast_hal_aruco/aruco_point", PointStamped, queue_size=1
        )
        self.center_publisher = rospy.Publisher(
            "/broadcast_hal_aruco/center_aruco", Image, queue_size=1
        )
        self.xyz_point_pub = rospy.Publisher(
            "/broadcast_hal_aruco/point", PointStamped, queue_size=1
        )
        self.base_link_point = rospy.Publisher(
            "/broadcast_hal_arcuo/base_link_point", PointStamped, queue_size=1
        )
        self.goal_point_pub = rospy.Publisher(
            "/broadcast_hal_arcuo/goal_point", PointStamped, queue_size=1
        )

        # spinup
        self.rate = 10.0

        # rospy.init_node("broadcast_aruco_node")

        self.listener = tf.TransformListener()

    def rgb_image_callback(self, ros_rgb_image):
        self.hal_rgb = self.cv_bridge.imgmsg_to_cv2(ros_rgb_image, "bgr8")
        # self.hal_rgb = ndimage.rotate(raw_head, -90)  # -96
        self.hal_rgb_arr = self.hal_rgb[:, :, ::-1]
        # self.hal_rgb = Im.fromarray(self.hal_rgb_arr)

    def rgb_info_callback(self, camera_info):
        self.camera_info_K = np.array(camera_info.K).reshape([3, 3])
        self.camera_info_D = np.array(camera_info.D)

    def depth_info_callback(self, cameraInfo):
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

    def depth_image_callback(self, data):
        self.depth_image = self.cv_bridge.imgmsg_to_cv2(data, data.encoding)

    def get_pcl(self, center):
        if self.intrinsics is None:
            return

        result = (
            np.array(
                rs2.rs2_deproject_pixel_to_point(
                    self.intrinsics,
                    [center[0], center[1]],
                    self.depth_image[center[1], center[0]],
                )
            )
            / 1000
        )

        self.result = result

    def publish_pcls(self):
        if self.result is None:
            return
        point = PointStamped()
        point.header.frame_id = "camera_aligned_depth_to_color_frame"
        point.point.x = self.result[2]
        point.point.y = -self.result[0]
        point.point.z = -self.result[1]
        point_in_target_frame = self.listener.transformPoint(self.cam_frame, point)
        self.base_link_point.publish(point_in_target_frame)
        self.xyz_point_pub.publish(point)

        return point, point_in_target_frame

    def predict(self):
        print("Predicting")
        self.head_image_subscriber = message_filters.Subscriber(
            "/head_camera/color/image_raw", Image
        )
        self.head_image_subscriber.registerCallback(self.rgb_image_callback)
        self.head_cam_info_sub = message_filters.Subscriber(
            "/head_camera/aligned_depth_to_color/camera_info", CameraInfo
        )
        self.head_cam_info_sub.registerCallback(self.rgb_info_callback)

        self.depth_info = rospy.Subscriber(
            "/head_camera/aligned_depth_to_color/camera_info",
            CameraInfo,
            self.depth_info_callback,
        )
        self.depth_cam = rospy.Subscriber(
            "/head_camera/aligned_depth_to_color/image_raw",
            Image,
            self.depth_image_callback,
        )

        br = tf.TransformBroadcaster()

        rate = rospy.Rate(self.rate)
        self.goals_published = 0
        while not rospy.is_shutdown():
            if (
                self.hal_rgb is not None
                and self.camera_info_K is not None
                and self.camera_info_D is not None
                and self.depth_image is not None
            ):
                with self.lock:
                    hal_cam_localizer = CamLocal(
                        self.hal_rgb_arr, self.camera_info_K, self.camera_info_D
                    )
                    hal_aruco_trans, aruco_hal_trans, center = (
                        hal_cam_localizer.localize_cam()
                    )
                    self.get_pcl(center)

                    point_cam, point_base = self.publish_pcls()

                    print(f"{hal_aruco_trans=}")
                    aruco_translation_depth = (
                        point_base.point.x,
                        point_base.point.y,
                        point_base.point.z,
                    )
                    aruco_translation = (
                        hal_aruco_trans[0, 3],
                        hal_aruco_trans[1, 3],
                        hal_aruco_trans[2, 3],
                    )

                    r = R.from_matrix(hal_aruco_trans[:3, :3])
                    r_quat = r.as_quat()

                    br.sendTransform(
                        aruco_translation_depth,
                        r_quat,
                        rospy.Time.now(),
                        "aruco_frame_depth",
                        self.cam_frame,
                    )

                print(
                    f"=========================Broadcasting Aruco Frame==================="
                )
            rate.sleep()


if __name__ == "__main__":
    rgb_topic = "/head_camera/color/image_raw"
    cam_info_topic = "/head_camera/aligned_depth_to_color/camera_info"
    pc_topic = "/head_camera/aligned_depth_to_color/image_raw"
    cam_frame = "camera_depth_optical_frame"
    aruco_broadcaster = BroadcastArucoHAL(
        rgb_topic=rgb_topic,
        cam_info_topic=cam_info_topic,
        pc_topic=pc_topic,
        cam_frame=cam_frame,
    )
    aruco_broadcaster.predict()
