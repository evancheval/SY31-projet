#!/usr/bin/env python3

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from turtlebot3_msgs.msg import SensorState
from transforms3d.euler import euler2quat
from sensor_msgs_py.point_cloud2 import create_cloud
from sensor_msgs.msg import PointCloud2, PointField


class Odom2Pose(Node):
    # Constantes
    ENCODER_RESOLUTION = 4096
    WHEEL_RADIUS = 0.033
    WHEEL_SEPARATION = 0.160
    MAG_OFFSET = np.pi / 2.0 - 0.07

    def __init__(self):
        super().__init__("odom_to_pose")

        # Initialisation des variables
        self.x_odom, self.y_odom, self.O_odom = 0.0, 0.0, 0.0
        self.x_gyro, self.y_gyro, self.O_gyro = 0.0, 0.0, 0.0
        self.prev_left_encoder = 0.0
        self.prev_right_encoder = 0.0
        # v est la vitesse linéaire du robot
        self.v = 0.0
        self.traj_enco = []
        self.traj_gyro = []

        # Initialisation des écarts-types pour les poses de l'encodeur et du gyroscope
        self.s_enco = 0.2
        self.s_gyro = 0.05
        self.Lambda = self.s_gyro**2 / (self.s_enco**2 + self.s_gyro**2)

        # Publication des positions estimées par les encodeurs et le gyromètre (sous forme de quaternions)
        self.pub_enco = self.create_publisher(PoseStamped, "/pose_enco", 10)
        self.pub_gyro = self.create_publisher(PoseStamped, "/pose_gyro", 10)
        # Publication des trajectoires estimées par les encodeurs et le gyromètre
        self.pub_enco_traj = self.create_publisher(PointCloud2, "/traj_enco", 10)
        self.pub_gyro_traj = self.create_publisher(PointCloud2, "/traj_gyro", 10)
        self.pub_moy_traj = self.create_publisher(
            PointCloud2, "/traj_moy_enco_gyro", 10
        )

        # Récupération des données des encodeurs et du gyromètre (données par le robot)
        self.sub_gyro = self.create_subscription(Imu, "/imu", self.callback_gyro, 10)
        self.sub_enco = self.create_subscription(
            SensorState, "/sensor_state", self.callback_enco, 10
        )

    @staticmethod
    def coordinates_to_message(x: float, y: float, O: float, t: Time) -> PoseStamped:
        """Transforme les coordonnées en message"""
        msg = PoseStamped()
        msg.header.stamp = t
        # Frame à sélectionner si visualisation dans rviz2
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
        """Calcule le laps de temps écoulé entre stamp et le dernier message reçu (field)"""
        t = stamp.sec + stamp.nanosec / 1e9
        dt = t - getattr(self, field) if hasattr(self, field) else 0.0
        setattr(self, field, t)
        return dt

    def callback_enco(self, sensor_state: SensorState):
        """Calcule la position estimée par l'encodeur"""

        # Calculer la différence des compteurs d'encodeurs sur les roues
        dq_left = sensor_state.left_encoder - self.prev_left_encoder
        dq_right = sensor_state.right_encoder - self.prev_right_encoder
        self.prev_left_encoder = sensor_state.left_encoder
        self.prev_right_encoder = sensor_state.right_encoder

        dt = self.dt_from_stamp(sensor_state.header.stamp, "prev_enco_t")
        if dt <= 0:
            return

        N = 4096
        dphi_g = dq_left * np.pi * 2 / N
        dphi_d = dq_right * np.pi * 2 / N

        phi_point_g = dphi_g / dt
        phi_point_d = dphi_d / dt

        r = 33 * 10**-3  # rayon des roues en mètre
        L = 80 * 10**-3  # demi entre-axe du robot

        # Vitesse linéaire
        self.v = r / 2 * (phi_point_d + phi_point_g)
        # Vitesse angulaire
        self.w = r / (2 * L) * (phi_point_d - phi_point_g)

        self.x_odom = self.x_odom + self.v * dt * np.cos(self.O_odom)
        self.y_odom = self.y_odom + self.v * dt * np.sin(self.O_odom)
        self.O_odom = self.O_odom + self.w * dt

        self.pub_enco.publish(
            Odom2Pose.coordinates_to_message(
                self.x_odom, self.y_odom, self.O_odom, sensor_state.header.stamp
            )
        )

        # Intensité = 0 (couleur rouge sur Rviz2)
        self.traj_enco.append(np.hstack((self.x_odom, self.y_odom, 0, 0, 0)))

        # Création du header pour le PointCloud2
        header = sensor_state.header
        header.frame_id = "base_scan"

        # Définition des champs du PointCloud2 (x, y, z, intensity, ring)
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(
                name="intensity", offset=12, datatype=PointField.FLOAT32, count=1
            ),
            PointField(name="ring", offset=16, datatype=PointField.FLOAT32, count=1),
        ]

        self.pub_enco_traj.publish(create_cloud(header, fields, self.traj_enco))
        self.calcul_traj_moyen_enco_gyro(header)

    def callback_gyro(self, gyro: Imu):
        """Calcule la position estimée par le gyroscope"""

        dt = self.dt_from_stamp(gyro.header.stamp, "prev_gyro_t")
        if dt <= 0:
            return

        self.w = gyro.angular_velocity.z

        self.O_gyro = self.O_gyro + self.w * dt
        self.x_gyro = self.x_gyro + self.v * dt * np.cos(self.O_gyro)
        self.y_gyro = self.y_gyro + self.v * dt * np.sin(self.O_gyro)

        self.pub_gyro.publish(
            Odom2Pose.coordinates_to_message(
                self.x_gyro, self.y_gyro, self.O_gyro, gyro.header.stamp
            )
        )

        # Intensité = 50 (couleur bleue sur Rviz2)
        self.traj_gyro.append(np.hstack((self.x_gyro, self.y_gyro, 0, 50, 50)))

        # Création du header pour le PointCloud2
        header = gyro.header
        header.frame_id = "base_scan"

        # Définition des champs du PointCloud2 (x, y, z, intensity, ring)
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(
                name="intensity", offset=12, datatype=PointField.FLOAT32, count=1
            ),
            PointField(name="ring", offset=16, datatype=PointField.FLOAT32, count=1),
        ]

        self.pub_gyro_traj.publish(create_cloud(header, fields, self.traj_gyro))

        self.calcul_traj_moyen_enco_gyro(header)

    def calcul_traj_moyen_enco_gyro(self, header):
        """Va publier sur le topic /traj_moy_enco_gyro une meilleure estimation de la trajectoire
        grâce aux encodeurs et gyroscope grâce à une fusion de données par moyennes pondérées.
        """

        if len(self.traj_enco) == 0 or len(self.traj_gyro) == 0:
            return
        if len(self.traj_enco) != len(self.traj_gyro):
            return

        liste_pt_traj_moy = []
        for i in range(len(self.traj_enco)):
            pt_enco = self.traj_enco[i]
            pt_gyro = self.traj_gyro[i]
            pt_moy = self.Lambda * np.array(pt_enco) + (1 - self.Lambda) * np.array(
                pt_gyro
            )
            liste_pt_traj_moy.append(pt_moy)

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(
                name="intensity", offset=12, datatype=PointField.FLOAT32, count=1
            ),
            PointField(name="ring", offset=16, datatype=PointField.FLOAT32, count=1),
        ]

        header.frame_id = "base_scan"
        self.pub_moy_traj.publish(create_cloud(header, fields, liste_pt_traj_moy))


def main(args=None):
    try:
        rclpy.init(args=args)
        node = Odom2Pose()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
