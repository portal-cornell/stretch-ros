import cv2
import numpy as np


class CamLocal:
    """
    This class uses an image with aruco markers
    and camera intrinsic parameters K and D to
    calcuate the transformation matrix between
    aruco marker frame with ID 0 to the camera frame.
    """

    def __init__(self, img, K, D):
        """
        img: RGB Image of aruco marker
        K: Camera intrinsic parameters K (3, 3)
        D: Camera distortion parameters D (5, 1)
        """
        self.img = img
        self.camera_info_K = K
        self.camera_info_D = D

    def inv_mat(self, rot, trans):
        """
        Transpose of rotation matrix (3, 3) is the same as the inverse
        Negates the translation vector
        """
        return rot.T, -1 * trans

    def make_trans(self, rot, trans):
        """
        rot (3, 3): rotation matrix
        trans (3, 1): translation vector
        output (4, 4): transformation matrix
        """
        trans_mat = np.zeros((4, 4))
        trans_mat[3, 3] = 1
        trans_mat[:3, :3] = rot
        trans_mat[:3, 3] = trans.flatten()
        return trans_mat
    
    def get_center_aruco(self, markerCorner):
        corners = markerCorner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners
        # convert each of the (x, y)-coordinate pairs to integers
        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))

        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        cY = int((topLeft[1] + bottomRight[1]) / 2.0)

        return (cX, cY)

    def estimate_pose_single_markers(self, corners, marker_size, mtx, distortion):
        """
        This will estimate the rvec and tvec for each of the marker corners detected by:
        corners, ids, rejectedImgPoints = detector.detectMarkers(image)
        corners - is an array of detected corners for each detected marker in the image
        marker_size - is the size of the detected markers
        mtx - is the camera matrix
        distortion - is the camera distortion matrix
        RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
        """
        marker_points = np.array(
            [
                [-marker_size / 2, marker_size / 2, 0],
                [marker_size / 2, marker_size / 2, 0],
                [marker_size / 2, -marker_size / 2, 0],
                [-marker_size / 2, -marker_size / 2, 0],
            ],
            dtype=np.float32,
        )
        trash = []
        rvecs = []
        tvecs = []
        for c in corners:
            nada, R, t = cv2.solvePnP(
                marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE
            )
            rvecs.append(R)
            tvecs.append(t)
            trash.append(nada)
        return rvecs, tvecs, trash

    def localize_cam(self, dict_type=cv2.aruco.DICT_4X4_50, marker_size=0.04):
        """
        self.img: RGB image with ArUco Marker
        self.camera_info_K: camera intrinsic parameters K
        self.camera_info_D: camera distortion parameter D
        Calculates the transformation matrices between a detected ArUco marker and camera frames
        """
        dictionary = cv2.aruco.getPredefinedDictionary(dict_type)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        corners, ids, _ = detector.detectMarkers(
            self.img
        )  # detect the markers in the image
        print(ids)

        corners = np.array(corners)[
            ids.flatten().argsort(kind="stable")
        ]  # sort the corners based on ID (0, 1, 2, 3)

        if np.all(ids != None):
            # Debugging tool to draw the frames in the image
            # for c in corners:
            #     rvec, tvec, _ = self.estimate_pose_single_markers(
            #         c, marker_size, self.camera_info_K, self.camera_info_D
            #     )
            #     rvec, jac = cv2.Rodrigues(
            #         np.array(rvec[0])
            #     )  # convert (3, 1) to (3, 3) rotation matrix
            #     trans_mat = self.make_trans(
            #         rvec, tvec[0]
            #     )  # construct a (4, 4) transformation matrix
            #     # draw the frames in the image
            #     cv2.drawFrameAxes(
            #         self.img,
            #         self.camera_info_K,
            #         self.camera_info_D,
            #         rvec,
            #         tvec[0],
            #         0.01,
            #     )
            #     cv2.imshow("frame", self.img)
            #     cv2.waitKey(0)

            # Use ArUco marker ID 0 as the origin
            cx, cy = self.get_center_aruco(corners[0])
            rvec, tvec, _ = self.estimate_pose_single_markers(
                corners[0], marker_size, self.camera_info_K, self.camera_info_D
            )
            rvec, jac = cv2.Rodrigues(np.array(rvec[0]))

            # Pose of the aruco marker with respect to HAL Cam
            hal_aruco_trans = self.make_trans(rvec, tvec[0])

            # Pose of HAL Cam with respect to the aruco marker
            rvec, tvec = self.inv_mat(rvec, tvec[0])
            aruco_hal_trans = self.make_trans(rvec, tvec)
            return hal_aruco_trans, aruco_hal_trans, (cx, cy)
