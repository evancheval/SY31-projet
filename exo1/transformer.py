#!/usr/bin/env python3

import numpy as np
from rclpy import qos
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import create_cloud

PC2FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name="cluster", offset=16, datatype=PointField.FLOAT32, count=1),
]


class Transformer(Node):
    def __init__(self):
        super().__init__("transformer")
        lidar_qos =  qos.QoSProfile(depth=10, reliability=qos.QoSReliabilityPolicy.BEST_EFFORT)
        self.pub = self.create_publisher(PointCloud2, "points", 10)
        self.sub = self.create_subscription(LaserScan, "scan", self.callback, lidar_qos)

    def callback(self, msg: LaserScan):
        xy = []
        intensities = []
        for i, theta in enumerate(np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)):
            # TODO: Remove points too close
            if (msg.ranges[i]<0.1): # 10 cm = 0.1 m
                continue       

            # To prevent bugs with the arc shape
            if (msg.ranges[i]>0.99):
                continue         

            # TODO: Polar to Cartesian transformation
            x = msg.ranges[i] * np.cos(theta)
            y = msg.ranges[i] * np.sin(theta)
            xy.append((x,y))
            intensities.append(msg.intensities[i])
            
        zeros = np.zeros((len(xy), 1))
        intensities = np.reshape(intensities, (len(intensities), 1))
        points = np.hstack((xy, zeros, intensities, zeros))
        self.pub.publish(create_cloud(msg.header, PC2FIELDS, points))


def main(args=None):
    import rclpy

    rclpy.init(args=args)
    node = Transformer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
