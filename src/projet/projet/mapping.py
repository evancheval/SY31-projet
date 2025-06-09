#! /usr/bin/env python3

import numpy as np
import time
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from transforms3d.euler import quat2euler
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import create_cloud, read_points_numpy


class MappingClass(Node):

    def __init__(self):
        super().__init__("mapping")

        self.pub_map = self.create_publisher(PointCloud2, "map", 10)
        self.pub_map_enco = self.create_publisher(PointCloud2, "map_enco", 10)
        self.pub_map_gyro = self.create_publisher(PointCloud2, "map_gyro", 10)

        # Nombre minimum de points récupérés par le lidar pour ajouter à la carte
        self.min_points = 5
        # Initialisation des transformations
        # ref_T est la transformation de référence entre le repère fixe et le repère du Lidar
        self.ref_T = np.eye(3)
        self.previous_ref_T = np.eye(3)
        # T_enco est la transformation entre le repère du Lidar et le repère de l'encodeur
        self.T_enco = np.eye(3)
        # T_gyro est la transformation entre le repère du Lidar et le repère du gyroscope
        self.T_gyro = np.eye(3)
        # T_lidar est la transformation entre le repère du Lidar et le repère fixe
        # (on en a besoin pour corriger les erreurs de rotation du Lidar)
        self.T_lidar = np.eye(3)

        # last_angle est l'angle de rotation du Lidar lors du dernier message reçu
        # Il est utilisé pour détecter une rotation et initier la correction des erreurs de rotation du Lidar
        self.last_angle = None

        # Initialisation des cartes
        self.map: np.ndarray = None
        self.map_enco: np.ndarray = None
        self.map_gyro: np.ndarray = None

        # Distance minimale entre deux points Lidar pour qu'ils soient considérés comme différents
        self.dist_min = 0.01

        # Indicateurs pour savoir si on mappe l'encodeur et/ou le gyroscope
        self.mapping_enco = True
        self.mapping_gyro = True

        # Compteur de mapping (utile si on veut visualiser le mapping itérativement dans RViz2)
        self.n_map = 0

        # Sauvegarde du dernier et de l'antépénultième message Lidar reçu pour effectuer la correction des erreurs de rotation
        self.previous_msg_map = None
        self.last_msg = None

        # Création d'un timer pour traiter le mapping,
        # afin de ne pas être surchargé par les messages du Lidar
        self.timer = self.create_timer(1.0, self.process_latest_msg)

        # Initialisation des variables pour les poses de l'encodeur et du gyroscope
        self.x_enco, self.y_enco, self.O_enco = None, None, None
        self.x_gyro, self.y_gyro, self.O_gyro = None, None, None
        # Initialisation des poses de l'encodeur et du gyroscope lors du premier mapping effectué.
        # Ces valeurs sont utilisées pour calculer la transformation T_enco et T_gyro,
        # on les considérera comme les poses des repères fixes de l'encodeur et du gyroscope.
        self.x_enco_0, self.y_enco_0, self.O_enco_0 = None, None, None
        self.x_gyro_0, self.y_gyro_0, self.O_gyro_0 = None, None, None

        # Initialisation des écarts-types pour les poses de l'encodeur et du gyroscope
        self.s_enco = 0.2
        self.s_gyro = 0.05
        self.Lambda = self.s_gyro**2 / (self.s_enco**2 + self.s_gyro**2)

        # Récupération des poses de l'encodeur et du gyroscope publiées par le noeud odom2pose
        self.sub_enco = self.create_subscription(
            PoseStamped, "/pose_enco", self.callback_enco, 10
        )
        self.sub_gyro = self.create_subscription(
            PoseStamped, "/pose_gyro", self.callback_gyro, 10
        )
        # Récupération des points du Lidar publiés par le noeud transformer
        self.sub_map = self.create_subscription(
            PointCloud2, "points", self.mapping_callback, 10
        )

    def mapping_callback(self, msg: PointCloud2):
        """Enregistre le dernier message reçu pour le traitement ultérieur."""
        self.last_msg = msg
        # On enregistre l'orientation associée au dernier message reçu.
        # On procède par fusion avec les moyennes pondérées des poses de l'encodeur et du gyroscope
        if self.O_enco is not None and self.O_gyro is not None:
            self.last_angle = (
                self.Lambda * self.O_enco + (1 - self.Lambda) * self.O_gyro
            )

    def callback_enco(self, msg: PoseStamped):
        """Enregistre la dernière pose reçue de l'encodeur, et mets à jour la matrice de transformation T_enco."""
        self.x_enco = msg.pose.position.x
        self.y_enco = msg.pose.position.y
        quat = [
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
        ]
        self.O_enco = quat2euler(quat)[2]

        # Mise à jour de la matrice de transformation T_enco (z=0 car on travaille en 2D)
        if self.x_enco_0 != None:
            self.T_enco = MappingClass.get_transform(
                np.array([self.x_enco, self.y_enco, self.O_enco]),
                np.array([self.x_enco_0, self.y_enco_0, self.O_enco_0]),
            )

        # self.get_logger().info(f"Encoder pose: x={self.x_enco}, y={self.y_enco}, O={self.O_enco}")

    def callback_gyro(self, msg: PoseStamped):
        """Enregistre la dernière pose reçue du gyroscope, et mets à jour la matrice de transformation T_gyro."""
        self.x_gyro = msg.pose.position.x
        self.y_gyro = msg.pose.position.y
        quat = [
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
        ]
        self.O_gyro = quat2euler(quat)[2]

        # Mise à jour de la matrice de transformation T_gyro (z=0 car on travaille en 2D)
        if self.x_gyro_0 is not None:
            self.T_gyro = MappingClass.get_transform(
                np.array([self.x_gyro, self.y_gyro, self.O_gyro]),
                np.array([self.x_gyro_0, self.y_gyro_0, self.O_gyro_0]),
            )
            
        # self.get_logger().info(f"Gyro pose: x={self.x_gyro}, y={self.y_gyro}, O={self.O_gyro}")

    def process_latest_msg(self):
        """Traite le dernier message reçu du Lidar pour mettre à jour la carte."""
        if self.last_msg is None or self.x_enco is None or self.x_gyro is None:
            return
        else:
            # self.get_logger().info("Traitement du dernier message reçu...")
            start_time = time.time()
            msg = self.last_msg
            # Réinitialisation du dernier message pour éviter de le traiter plusieurs fois
            self.last_msg = None

        xy: np.ndarray = read_points_numpy(msg, ["x", "y"])

        if xy.shape[0] <= self.min_points:
            return

        # Fusion avec des moyennes pondérées
        self.ref_T = self.Lambda * self.T_enco + (1 - self.Lambda) * self.T_gyro
        if self.mapping_enco:
            self.ref_T = self.T_enco.copy()
        elif self.mapping_gyro:
            self.ref_T = self.T_gyro.copy()

        if self.map is None:
            # Premier mapping
            self.map = xy
            self.n_map += 5
        else:
            xy_fixed = MappingClass.apply_transform(xy, self.ref_T)

            #########################################################################################
            # Correction de l'erreur de rotation du Lidar par ICP
            # Non inclus car diminue la précision du mapping
            #########################################################################################
            # if self.previous_msg_map is not None and abs(self.last_angle - self.O_gyro) > 0.1:
                # self.get_logger().info(f"Correction de l'erreur de rotation du Lidar par ICP")
                # # Correction de l'erreur de rotation du Lidar
                # ref_points : np.ndarray = read_points_numpy(self.previous_msg_map, ["x", "y"])
                # ref_points_fixed = MappingClass.apply_transform(ref_points, self.previous_ref_T)

                # plus_proche_voisin = MappingClass.find_nearest_neighbors_euclidian(xy_fixed, ref_points_fixed)
                # distances = np.linalg.norm(xy_fixed - plus_proche_voisin, axis=1)
                # # Sélectionner les points qui sont suffisamment proches de leur plus proche voisin
                # mask = distances < 0.05
                # xy_fixed = xy_fixed[mask]

                # self.previous_ref_T = self.ref_T.copy()
                # # pose_vect3 = MappingClass.estimate_pose_from_points(xy_fixed)
                # # ref_T = np.eye(3)
                # # ref_T[0:2, 2] = pose_vect3[0:2,0]
                # # Get T between ref and xy
                # T_current = MappingClass.apply_icp(xy_fixed, ref_points_fixed)
                # # T_current = ref_T @ T_current
                # # R_current = T_current[:2, :2]
                # # T_current[0:2, 2] = np.zeros((2,))
                # # Previous T -> current T
                # self.T_lidar = self.T_lidar.copy() @ T_current
                # # new_xy_fixed = MappingClass.apply_transform(xy.copy(), self.ref_T.copy() @ self.T_lidar)
                # # self.previous_msg_map = msg
                # # map_msg = np.hstack(
                # #     (
                # #         ref_points_fixed,
                # #         np.zeros((len(ref_points_fixed), 1)),
                # #         np.ones((len(ref_points_fixed), 1)),
                # #         np.ones((len(ref_points_fixed), 1)),
                # #     )
                # # )
                # # map_msg = np.vstack((map_msg, 
                # #                      np.hstack(
                # #                          (
                # #                             new_xy_fixed,
                # #                             np.zeros((len(new_xy_fixed), 1)),
                # #                             np.ones((len(new_xy_fixed), 1))*2,
                # #                             np.ones((len(new_xy_fixed), 1))*2
                # #                          )
                # #                      )))
                # # map_msg = np.vstack((map_msg,
                # #                      np.hstack(
                # #                          (
                # #                             ref_points,
                # #                             np.zeros((len(ref_points), 1)),
                # #                             np.zeros((len(ref_points), 1))
                # #                          )
                # #                      )))
                
                # # self.pub_map.publish(create_cloud(msg.header, msg.fields, map_msg))
                # # return
            #########################################################################################

            xy_fixed = MappingClass.apply_transform(xy_fixed.copy(), self.T_lidar)

            # Trouver les plus proches voisins pour tous les points
            nearest_neighbors = MappingClass.find_nearest_neighbors_euclidian(
                xy_fixed, self.map
            )
            distances = np.linalg.norm(xy_fixed - nearest_neighbors, axis=1)
            # Sélectionner les points qui sont suffisamment éloignés de leur plus proche voisin
            mask = distances > self.dist_min
            new_points = xy_fixed[mask]
            if new_points.shape[0] > 0:
                self.map = np.vstack((self.map, new_points))
                self.n_map += 5

        # Publication de la carte
        # self.get_logger().info(f"done. Map size: {len(self.map)} points")
        self.previous_msg_map = msg
        map_msg = np.hstack(
            (
                self.map,
                np.zeros((len(self.map), 1)),
                np.ones((len(self.map), 1)) * self.n_map,
                np.zeros((len(self.map), 1)),
            )
        )
        self.pub_map.publish(create_cloud(msg.header, msg.fields, map_msg))

        # Mappings individuels pour l'encodeur et le gyroscope
        if self.mapping_enco:
            if self.map_enco is None:
                # Définition de l'origine du repère fixe de l'encodeur et premier mapping
                self.map_enco = xy
                self.x_enco_0 = self.x_enco
                self.y_enco_0 = self.y_enco
                self.O_enco_0 = self.O_enco
            else:
                # Transformation des points du repère du robot au repère fixe en utilisant la pose de l'encodeur
                xy_fixed_enco = MappingClass.apply_transform(
                    xy, self.T_enco
                )
                nearest_neighbors = MappingClass.find_nearest_neighbors_euclidian(
                    xy_fixed_enco, self.map_enco
                )
                distances = np.linalg.norm(xy_fixed_enco - nearest_neighbors, axis=1)
                # Sélectionner les points qui sont suffisamment éloignés de leur plus proche voisin
                mask = distances > self.dist_min
                new_points = xy_fixed_enco[mask]
                if new_points.shape[0] > 0:
                    self.map_enco = np.vstack((self.map_enco, new_points))
            # Publication de la carte
            # self.get_logger().info(f"done. Map_enco size: {len(self.map_enco)} points")
            map_msg_enco = np.hstack(
                (
                    self.map_enco,
                    np.zeros((len(self.map_enco), 1)),
                    np.ones((len(self.map_enco), 1)) * self.n_map,
                    np.zeros((len(self.map_enco), 1)),
                )
            )
            self.pub_map_enco.publish(
                create_cloud(msg.header, msg.fields, map_msg_enco)
            )

        if self.mapping_gyro:
            if self.map_gyro is None:
                # Définition de l'origine du repère fixe du gyroscope et premier mapping
                self.map_gyro = xy
                self.x_gyro_0 = self.x_gyro
                self.y_gyro_0 = self.y_gyro
                self.O_gyro_0 = self.O_gyro
            else:
                # Transformation des points du repère du Lidar au repère fixe en utilisant la pose du gyroscope
                xy_fixed_gyro = MappingClass.apply_transform(
                    xy, self.T_gyro @ self.T_lidar
                )
                nearest_neighbors = MappingClass.find_nearest_neighbors_euclidian(
                    xy_fixed_gyro, self.map_gyro
                )
                distances = np.linalg.norm(xy_fixed_gyro - nearest_neighbors, axis=1)
                # Sélectionner les points qui sont suffisamment éloignés de leur plus proche voisin
                mask = distances > self.dist_min
                new_points = xy_fixed_gyro[mask]
                if new_points.shape[0] > 0:
                    self.map_gyro = np.vstack((self.map_gyro, new_points))
            # Publication de la carte
            # self.get_logger().info(f"done. Map_gyro size: {len(self.map_gyro)} points")
            map_msg_gyro = np.hstack(
                (
                    self.map_gyro,
                    np.zeros((len(self.map_gyro), 1)),
                    np.ones((len(self.map_gyro), 1)) * self.n_map,
                    np.zeros((len(self.map_gyro), 1)),
                )
            )
            self.pub_map_gyro.publish(
                create_cloud(msg.header, msg.fields, map_msg_gyro)
            )
        elapsed = time.time() - start_time
        self.get_logger().info(f"Temps écoulé : {elapsed} secondes")


    #########################################################################################
    # Correction de l'erreur de rotation du Lidar par ICP
    # Non inclus car diminue la précision du mapping
    #########################################################################################
    # @staticmethod
    # def estimate_pose_from_points(xy: np.ndarray) -> np.ndarray:
    #     """Renvoie le centre du cluster sous forme de matrice (2*1)."""
    #     xy_c = np.ndarray([2,1])
    #     xy_c[:,0] = np.mean(xy[:,0]), np.mean(xy[:,1])
    #     return xy_c

    # @staticmethod
    # def best_fit_transform(S, D):
    #     """Trouve la meilleure transformation (T) entre S et D.
    #     :param S: nuage de points source
    #     :param D: nuage de points destination
    #     :return: matrice de transformation (4*4) entre source et destination
    #     """
    #     # Centre de masse
    #     centroid_S = np.mean(S, axis=0)
    #     centroid_D = np.mean(D, axis=0)

    #     # Centrer les nuages de points à l'origine
    #     SS = S - centroid_S
    #     DD = D - centroid_D
    #     H = SS.T @ DD

    #     # Si la matrice de covariance H contient des NaN ou des Inf, il y a un problème
    #     if not np.all(np.isfinite(H)):
    #         return np.eye(4)

    #     try:
    #         U, _, VT = np.linalg.svd(H)
    #     except np.linalg.LinAlgError:
    #         # La SVD n'a pas convergé
    #         return np.eye(4)

    #     R_mat = VT.T @ U.T

    #     # Réflexion détectée
    #     if np.linalg.det(R_mat) < 0:
    #         VT[2, :] *= -1
    #         R_mat = VT.T @ U.T

    #     t = centroid_D - R_mat @ centroid_S

    #     T = np.eye(4)
    #     T[:3, :3] = R_mat
    #     T[:3, 3] = t
    #     return T

    # @staticmethod
    # def apply_icp(S: np.ndarray, D: np.ndarray, max_iter: int = 20, tol: float = 1e-10):
    #     """Appliquer l'ICP de S à D.
    #     :param S: nuage de points source
    #     :param D: nuage de points destination
    #     :param max_iter: Nombre maximal d'itérations
    #     :param tol: Tolérance d'erreur entre 2 erreurs consécutives avant d'arrêter le processus
    #     :return: matrice de transformation (4*4)
    #     """
    #     T = np.eye(3)
    #     error_prev = float("-inf")
    #     error_current = float("inf")
    #     ST = S
    #     D3D = np.hstack((D, np.zeros((D.shape[0], 1))))
    #     i = 0

    #     while i < max_iter and abs(error_current - error_prev) > tol:
    #         ST3D = np.hstack((ST, np.zeros((ST.shape[0], 1))))
    #         voisins_ST_dans_D = MappingClass.find_nearest_neighbors_euclidian(ST, D)
    #         voisins_ST_dans_D3D = np.hstack((voisins_ST_dans_D, np.zeros((voisins_ST_dans_D.shape[0], 1))))
    #         T_local3D = MappingClass.best_fit_transform(ST3D, voisins_ST_dans_D3D)
    #         T_local = np.eye(3)
    #         T_local[:2,:2] = T_local3D[:2, :2]
    #         T_local[:2, 2] = T_local3D[:2, 3]
    #         ST = MappingClass.apply_transform(ST, T_local)
    #         T = T @ T_local
    #         error_prev = error_current

    #         error_current = np.mean(
    #             MappingClass.find_nearest_neighbors_euclidian(ST, D)
    #         )
    #         i += 1

    #     return T

    #########################################################################################

    @staticmethod
    def apply_transform(xy: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Appliquer la transformation T sur les points.
        :param points: matrice (N*2) des coordonnées x, y
        :param T: matrice de transformation (3*3)
        :return: matrice (N*2) des coordonnées x, y transformées
        """
        xy_homogeneous = np.hstack((xy, np.ones((xy.shape[0], 1))))
        return (T @ xy_homogeneous.T).T[:, :2]

    @staticmethod
    def find_nearest_neighbors_euclidian(S: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Pour chaque point de S, trouve le point le plus proche dans D
        :param S: nuage de points source
        :param D: nuage de points destination
        :return: liste des points correspondants dans D
        """
        matched = []

        for s_p in S:
            # Calculer la distance euclidienne entre le point s_p et tous les points de D
            distances = np.linalg.norm(D - s_p, axis=1)
            closest_index = np.argmin(distances)
            # Ajouter le point le plus proche à la liste des correspondances
            matched.append(D[closest_index])

        return np.array(matched)

    @staticmethod
    def get_transform(Lidar: np.ndarray, Fixed: np.ndarray) -> np.ndarray:
        """Trouve la transformation (T) du repère Lidar (x,y, O) au repère fixe (x,y, O).
        :Lidar point : [xL, yL, OL]
        :Fixed point : [xF, yF, OF]
        :return: matrice de transformation (3*3) entre Lidar et fixe
        """
        T = np.eye(3)
        # Translation
        T[0:2, 2] =  Lidar[0:2] - Fixed[0:2]
        # Rotation
        OL = Lidar[2]  # Orientation du robot dans son repère (repère Lidar)
        OF = Fixed[2]  # Orientation du robot dans le repère fixe
        THETA = OL - OF
        T[0:2, 0:2] = [[np.cos(THETA), -np.sin(THETA)], [np.sin(THETA), np.cos(THETA)]]
        return T


def main(args=None):
    import rclpy

    rclpy.init(args=args)
    node = MappingClass()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
