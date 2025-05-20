#! /usr/bin/env python3

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image


class Detector(Node):
    def __init__(self):
        super().__init__("detector")

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, "detections", 1)

        # Determine min and max pixel values

        #mask HSV
        #blue mask (HSV) : 205/360 hue -> 160 to 230 -> OpenCV : 80 to 115
        self.blue_min = np.array([[80, 120, 80]], dtype=np.uint8)
        self.blue_max = np.array([[115, 230, 250]], dtype=np.uint8)

        self.sub = self.create_subscription(Image, "image_rect", self.callback, 1)

    def callback(self, msg: Image):
        """Process the images going on image_rect"""

        # Convert ROS -> OpenCV
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().warn(f"ROS->OpenCV {e}")
            return

        img_out = self.detect(img)

        # Convert OpenCV -> ROS
        try:
            format = "bgr8" if img_out.ndim == 3 else "mono8"
            msg_out = self.bridge.cv2_to_imgmsg(img_out, format)
        except CvBridgeError as e:
            self.get_logger().warn(f"ROS->OpenCV {e}")
            return

        self.pub.publish(msg_out)

    def detect(self, img: np.ndarray) -> np.ndarray:
        # Filter pixels based on their value
        #if HSV image
        img = cv2.cvtColor(img, 40)

        mask = cv2.inRange(img, self.blue_min[0], self.blue_max[0])
        # Convert the image to HSV color space
        for i in range(1,len(self.blue_min)):        
            mask = mask + cv2.inRange(img, self.blue_min[i], self.blue_min[i])

        

        #if HSV image
        img = cv2.cvtColor(img, 54)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(img, contours, -1, (0,255,0), 3,8, hierarchy)
        
        areamax = 0
        imax = 0
        if not(len(contours)==0):
            for i in range(len(contours)):
                area = cv2.contourArea(contours[i])
                if areamax < area:
                    areamax = area
                    imax = i
            cv2.drawContours(img, contours, imax, (0,255,0), 3,8)
            # centerx = np.mean(contours[imax][0])
            # centery = np.mean(contours[imax][1])
            M = np.mean(contours[imax],axis=0)[0]
            cv2.circle(img,(int(M[0]),(int(M[1]))),2,(0,255,255),2)

        return img


def main(args=None):
    import rclpy

    rclpy.init(args=args)
    node = Detector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
