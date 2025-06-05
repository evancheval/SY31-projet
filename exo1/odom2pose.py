#!/usr/bin/env python3

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, MagneticField
from turtlebot3_msgs.msg import SensorState
from transforms3d.euler import euler2quat, quat2euler
from sensor_msgs_py.point_cloud2 import create_cloud
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField


class Odom2Pose(Node):
    # Constants
    ENCODER_RESOLUTION = 4096
    WHEEL_RADIUS = 0.033
    WHEEL_SEPARATION = 0.160
    MAG_OFFSET = np.pi / 2.0 - 0.07

    def __init__(self):
        super().__init__("odom_to_pose")

        # Variables
        self.x_odom, self.y_odom, self.O_odom = 0.0, 0.0, 0.0
        self.x_gyro, self.y_gyro, self.O_gyro = 0.0, 0.0, 0.0
        self.prev_left_encoder = 0.0
        self.prev_right_encoder = 0.0
        self.v = 0.0
        self.traj_enco = []
        self.traj_gyro = []

        # Publishers
        self.pub_enco = self.create_publisher(PoseStamped, "/pose_enco", 10)
        self.pub_gyro = self.create_publisher(PoseStamped, "/pose_gyro", 10)
        self.pub_enco_traj = self.create_publisher(PointCloud2, "/traj_enco", 10)
        self.pub_gyro_traj = self.create_publisher(PointCloud2, "/traj_gyro", 10)

        # Subscribers
        self.sub_gyro = self.create_subscription(Imu, "/imu", self.callback_gyro, 10)
        self.sub_enco = self.create_subscription(
            SensorState, "/sensor_state", self.callback_enco, 10
        )
        


    @staticmethod
    def coordinates_to_message(x: float, y: float, O: float, t: Time) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = t
        msg.header.frame_id = "odom"
        msg.pose.position.x = x
        msg.pose.position.y = y
        [
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
        ] = euler2quat(0.0, 0.0, O)
        return msg

    def dt_from_stamp(self, stamp: Time, field: str) -> float:
        t = stamp.sec + stamp.nanosec / 1e9
        dt = t - getattr(self, field) if hasattr(self, field) else 0.0
        setattr(self, field, t)
        return dt

    def callback_enco(self, sensor_state: SensorState):
        # Compute the differential in encoder count
        dq_left = sensor_state.left_encoder - self.prev_left_encoder
        dq_right = sensor_state.right_encoder - self.prev_right_encoder
        self.prev_left_encoder = sensor_state.left_encoder
        self.prev_right_encoder = sensor_state.right_encoder

        dt = self.dt_from_stamp(sensor_state.header.stamp, "prev_enco_t")
        if dt <= 0:
            return

        # TODO: Compute the linear and angular velocity (self.v and w)
        N = 4096
        dphi_g = dq_left * np.pi *2 / N
        dphi_d = dq_right * np.pi *2 / N

        phi_point_g = dphi_g / dt
        phi_point_d = dphi_d / dt

        r = 33 * 10**-3 # rayon des roues en mètre
        L = 80 * 10**-3 # demi entre-axe du robot


        self.v = r / 2 * (phi_point_d+phi_point_g)
        self.w = r / (2 * L) * (phi_point_d - phi_point_g)

        # TODO: Update x_odom, y_odom and O_odom accordingly
        self.x_odom = self.x_odom + self.v * dt * np.cos(self.O_odom)
        self.y_odom = self.y_odom + self.v * dt * np.sin(self.O_odom)
        self.O_odom = self.O_odom + self.w * dt

        self.pub_enco.publish(
            Odom2Pose.coordinates_to_message(
                self.x_odom, self.y_odom, self.O_odom, sensor_state.header.stamp
            )
        )

        self.traj_enco.append(np.hstack((self.x_odom,self.y_odom, 0, 0, 0)))


        # Création du header pour le PointCloud2
        header = sensor_state.header
        header.frame_id = "odom"

        # Définition des champs du PointCloud2 (x, y, z, intensity, ring)
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='ring', offset=16, datatype=PointField.FLOAT32, count=1),
        ]

        self.pub_enco_traj.publish(create_cloud(header, fields, self.traj_enco))

    def callback_gyro(self, gyro: Imu):
        dt = self.dt_from_stamp(gyro.header.stamp, "prev_gyro_t")
        if dt <= 0:
            return

        # TODO: retrieve the angular velocity
        self.w = gyro.angular_velocity.z

        # TODO: update O_gyro, x_gyro and y_gyro accordingly (using encoders' self.v)
        self.O_gyro = self.O_gyro + self.w*dt
        self.x_gyro = self.x_gyro + self.v * dt*np.cos(self.O_gyro)
        self.y_gyro = self.y_gyro + self.v * dt * np.sin(self.O_gyro)

        quat = [self.x_gyro, self.y_gyro, 0, self.O_gyro]
        eulZYZ = quat2euler(quat, 'szxz')
        # print(eulZYZ)

        self.pub_gyro.publish(
            Odom2Pose.coordinates_to_message(
                self.x_gyro, self.y_gyro, self.O_gyro, gyro.header.stamp
            )
        )

        self.traj_gyro.append(np.hstack((self.x_gyro,self.y_gyro, 0, 50, 50)))


        # Création du header pour le PointCloud2
        header = gyro.header
        header.frame_id = "odom"

        # Définition des champs du PointCloud2 (x, y, z, intensity, ring)
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='ring', offset=16, datatype=PointField.FLOAT32, count=1),
        ]

        self.pub_gyro_traj.publish(create_cloud(header, fields, self.traj_gyro))


def main(args=None):
    try:
        rclpy.init(args=args)
        node = Odom2Pose()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
