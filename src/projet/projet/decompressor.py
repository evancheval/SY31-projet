#! /usr/bin/env python3

from pathlib import Path

import cv2 as cv
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from yaml import safe_load


class Decompressor(Node):
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    qos_info = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

    def __init__(self):
        super().__init__("decompressor")

        self.pub = self.create_publisher(Image, "image_raw", 10)
        self.pub_info = self.create_publisher(CameraInfo, "camera_info", self.qos_info)

        self.info = CameraInfo()
        self.timer = self.create_timer(1.0, self.update_info)
        self.sub = self.create_subscription(
            CompressedImage, "image_robot", self.callback, self.qos
        )

    def update_info(self):
        calib_path = Path.home() / "sy31_projet/src/projet/calib.yaml"

        if not calib_path.exists():
            # Le calibrateur de caméra de ROS place sa calibration dans /tmp
            # Copie dans le répertoire personnel si ce n'est pas déjà fait
            calib_path_tmp = Path("/tmp/calibrationdata.tar.gz")
            if not calib_path_tmp.exists():
                return

            import tarfile

            tar = tarfile.open(calib_path_tmp, "r:gz")
            tar = tar.extractfile("ost.yaml")
            if tar is None:
                return
            with open(calib_path, "w") as f:
                f.write(tar.read().decode())

        with open(calib_path, "r") as f:
            calib = safe_load(f)

        self.info.width = calib["image_width"]
        self.info.height = calib["image_height"]
        self.info.k = calib["camera_matrix"]["data"]
        self.info.d = calib["distortion_coefficients"]["data"]
        self.info.r = calib["rectification_matrix"]["data"]
        self.info.p = calib["projection_matrix"]["data"]
        self.info.distortion_model = calib["distortion_model"]

    def callback(self, msg: CompressedImage):
        data = np.asarray(msg.data, dtype=np.uint8)

        msg_out = Image()
        img = cv.imdecode(data, cv.IMREAD_UNCHANGED)
        msg_out.data = img.flatten().tolist()
        msg_out.header = msg.header
        msg_out.encoding = msg.format.split(";")[0]
        msg_out.height = img.shape[0]
        msg_out.width = img.shape[1]
        msg_out.step = img.shape[1] * img.shape[2]
        self.info.header = msg.header

        self.pub_info.publish(self.info)
        self.pub.publish(msg_out)


def main(args=None):
    rclpy.init(args=args)
    node = Decompressor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
