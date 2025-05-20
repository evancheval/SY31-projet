#! /usr/bin/env python3

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class Projector(Node):
    def __init__(self):
        super().__init__("projector")

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, "projections", 1)

        # Initialize the node parameters
        # P, K and D are the camera parameters found by ROS calibration
        # (P: projection matrix, K: intrinsic parameters and D: distortion parameters, see CameraInfo for more details)
        self.P = self.K = self.D = None
        self.axis = np.array([[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, -1, 1]], dtype=int).T

        # Constants for the checkerboard detection and projection
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.board_size = (7, 5)
        self.objp = np.zeros((np.prod(self.board_size), 3), np.float32)
        self.objp[:, :2] = np.mgrid[: self.board_size[0], : self.board_size[1]].T.reshape(-1, 2)

        # Subscriber to the input topic. self.callback is called when a message is received
        self.sub_info = self.create_subscription(CameraInfo, "camera_info", self.callback_info, 1)
        self.sub_img = self.create_subscription(Image, "image_rect", self.callback_img, 1)

    def callback_info(self, msg: CameraInfo):
        """Retrieves the camera intrinsic parameters from the ROS calibration."""

        self.D = np.array(msg.d, dtype=np.float32).flatten()
        self.K = np.array(msg.k, dtype=np.float32).reshape((3, 3))
        self.P = np.array(msg.p, dtype=np.float32).reshape((3, 4))

    def callback_img(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().warn(f"ROS->OpenCV {e}")
            return

        # As we use the calibration information from ROS, die if it does not exist
        if self.P is None or self.P[0, 0] == 0:
            self.get_logger().warn("Camera is not calibrated")
            return

        # Determine the transformation from camera to checkerboard
        checkerboard_transfo = self.find_checkerboard_transfo(img)
        if checkerboard_transfo is not None:
            # TODO: Transform each axis points on the checkerboard and project to camera

            img = cv2.line(img, tuple(axis[:, 0]), tuple(axis[:, 1]), (255, 0, 0), 5)
            img = cv2.line(img, tuple(axis[:, 0]), tuple(axis[:, 2]), (0, 255, 0), 5)
            img = cv2.line(img, tuple(axis[:, 0]), tuple(axis[:, 3]), (0, 0, 255), 5)

        # Convert OpenCV -> ROS Image and publish
        try:
            msg_out = self.bridge.cv2_to_imgmsg(img, "bgr8")
        except CvBridgeError as e:
            self.get_logger().warn(f"OpenCV->ROS {e}")
            return

        self.pub.publish(msg_out)

    def find_checkerboard_transfo(self, img: np.ndarray) -> np.ndarray:
        """Finds the transformation to a checkerboard in the image.
        returns: 4x4 homogeneous transform such that H = R|t if a checkerboard is found, None otherwise
        """
        # TODO: Convert to greyscale

        # TODO: Find the corners in the image


def main(args=None):
    import rclpy

    rclpy.init(args=args)
    node = Projector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
