#! /usr/bin/env python3

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from turtlebot3_msgs.msg import SensorState

#test
class Detector(Node):
    def __init__(self):
        super().__init__("detector")

        self.distance : np.float32 = 500.0

        self.bridge = CvBridge()
        self.pub_detect = self.create_publisher(Image, "detections", 1)
        # self.pub_chat = self.create_publisher(String, "chatter", 1)

        # Determine min and max pixel values

        #mask HSV
        #blue mask (HSV) : 205/360 hue -> 160 to 230 -> OpenCV : 80 to 115
        self.blue_min = np.array([[80, 120, 80]], dtype=np.uint8)
        self.blue_max = np.array([[115, 230, 250]], dtype=np.uint8)

        self.red_min = np.array([[0, 100, 10], [140, 100, 10]], dtype=np.uint8)
        self.red_max = np.array([[10, 230, 245], [180, 230, 245]], dtype=np.uint8)

        self.sub_camera = self.create_subscription(Image, "image_rect", self.callback, 1)
        self.sub_sonar = self.create_subscription(SensorState, "sensor_state", self.callback_sonar, 1)

    def callback_sonar(self, msg: SensorState) :
        self.distance = msg.sonar

    def callback(self, msg: Image):
        """Process the images going on image_rect"""

        if self.distance < 600.0 and self.distance > 10.0:

            # Convert ROS -> OpenCV
            try:
                img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except CvBridgeError as e:
                self.get_logger().warn(f"ROS->OpenCV {e}")
                return

            img_out = self.detect(img, False)

            # Convert OpenCV -> ROS
            try:
                format = "bgr8" if img_out.ndim == 3 else "mono8"
                msg_out = self.bridge.cv2_to_imgmsg(img_out, format)
            except CvBridgeError as e:
                self.get_logger().warn(f"ROS->OpenCV {e}")
                return

            self.pub_detect.publish(msg_out)

    def detect(self, img: np.ndarray, direction_fleche_sujet: bool = True) -> np.ndarray:
        # Filter pixels based on their value
        #if HSV image
        img = cv2.cvtColor(img, 40)

        

        mask_blue = cv2.inRange(img, self.blue_min[0], self.blue_max[0])
        mask_red = cv2.inRange(img, self.red_min[0], self.red_max[0])
        # Convert the image to HSV color space
        for i in range(1,len(self.red_min)):        
            mask_red = mask_red + cv2.inRange(img, self.red_min[i], self.red_max[i])

        

        #if HSV image
        img = cv2.cvtColor(img, 54)

        contours_blue, hierarchy_blue = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_red, hierarchy_red = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(img, contours, -1, (0,255,0), 3,8, hierarchy)
        
        areamax_red = 0
        areamax_blue = 0
        imax_red = 0
        imax_blue = 0
        if not(len(contours_blue)==0 and len(contours_red)==0):
            for i in range(len(contours_blue)):
                area = cv2.contourArea(contours_blue[i])
                if areamax_blue < area:
                    areamax_blue = area
                    imax_blue = i
            for i in range(len(contours_red)):
                area = cv2.contourArea(contours_red[i])
                if areamax_red < area:
                    areamax_red = area
                    imax_red = i
            if areamax_red > areamax_blue :
                areamax = areamax_red
                imax = imax_red
                contours = contours_red
                if direction_fleche_sujet:
                    self.get_logger().info("Gauche")
                else:
                    self.get_logger().info("Droite")
            else:
                areamax = areamax_blue
                imax = imax_blue
                contours = contours_blue
                if direction_fleche_sujet:
                    self.get_logger().info("Droite")
                else:
                    self.get_logger().info("Gauche")

            if areamax > 1.0:
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
