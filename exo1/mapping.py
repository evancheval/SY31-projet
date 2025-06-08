#! /usr/bin/env python3

from builtin_interfaces.msg import Time
import numpy as np
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Imu
from turtlebot3_msgs.msg import SensorState
from transforms3d.euler import euler2quat, quat2euler
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud, read_points_numpy


class MappingClass(Node):
    # Constants
    ENCODER_RESOLUTION = 4096
    WHEEL_RADIUS = 0.033
    WHEEL_SEPARATION = 0.160
    MAG_OFFSET = np.pi / 2.0 - 0.07

    def __init__(self):
        super().__init__("mapping")
        self.pub_map = self.create_publisher(PointCloud2, "map", 10)
        self.pub_map_enco = self.create_publisher(PointCloud2, "map_enco", 10)
        self.pub_map_gyro = self.create_publisher(PointCloud2, "map_gyro", 10)

        self.min_points = 5
        self.ref_T = np.eye(4)
        self.T_enco = np.eye(4)
        self.T_gyro = np.eye(4)
        self.T_lidar = np.eye(4)
        self.last_angle = None
        self.map : np.ndarray = None
        self.map_enco : np.ndarray = None
        self.map_gyro : np.ndarray = None
        self.n_map = 0
        self.previous_msg_mapped = None
        self.last_msg = None
        self.timer = self.create_timer(0.5, self.process_latest_msg)

        self.x_odom, self.y_odom, self.O_odom = None, None, None
        self.x_gyro, self.y_gyro, self.O_gyro = None, None, None
        self.prev_left_encoder = 0.0
        self.prev_right_encoder = 0.0
        self.v = 0.0
        self.x_odom_0, self.y_odom_0, self.O_odom_0 = None, None, None
        self.x_gyro_0, self.y_gyro_0, self.O_gyro_0 = None, None, None


        self.sub_enco = self.create_subscription(SensorState, "/sensor_state", self.callback_enco, 10)

        # self.sub_enco = self.create_subscription(PoseStamped, "/pose_enco", self.callback_enco, 10)
        self.sub_gyro = self.create_subscription(PoseStamped, "/pose_gyro", self.callback_gyro, 10)
        

        self.sub_map = self.create_subscription(PointCloud2, "points", self.mapping_callback, 10)

    def mapping_callback(self, msg: PointCloud2):
        self.last_msg = msg  # Just store the latest message
        self.last_angle = self.O_gyro

    def dt_from_stamp(self, stamp: Time, field: str) -> float:
        t = stamp.sec + stamp.nanosec / 1e9
        dt = t - getattr(self, field) if hasattr(self, field) else 0.0
        setattr(self, field, t)
        return dt

    def callback_enco(self, msg: SensorState):
        """Callback for the encoder pose."""
        ##################################
        # if self.last_msg is not None and abs(msg.header.stamp.sec%10*1e9 + msg.header.stamp.nanosec - self.last_msg.header.stamp.sec%10*1e9 - self.last_msg.header.stamp.nanosec) > 1e8:
        #     return
        # self.x_odom = msg.pose.position.x
        # self.y_odom = msg.pose.position.y
        # quat = [
        #     msg.pose.orientation.w,
        #     msg.pose.orientation.x,
        #     msg.pose.orientation.y,
        #     msg.pose.orientation.z
        # ]
        # self.O_odom = quat2euler(quat)[2]
        ##################################
        if self.x_odom is None or self.y_odom is None or self.O_odom is None:
            self.x_odom = 0.0
            self.y_odom = 0.0
            self.O_odom = 0.0
        sensor_state = msg
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
        newx = self.x_odom + self.v * dt * np.cos(self.O_odom)
        newy = self.y_odom + self.v * dt * np.sin(self.O_odom)
        newO = self.O_odom + self.w * dt

        if self.x_odom_0 is not None:
            self.T_enco = MappingClass.get_transform(
                np.array([newx, newy, 0, newO]),
                np.array([self.x_odom_0, self.y_odom_0, 0, self.O_odom_0]))
        
        self.x_odom = newx
        self.y_odom = newy
        self.O_odom = newO

        # self.get_logger().info(f"Encoder pose: x={self.x_odom}, y={self.y_odom}, O={self.O_odom}")

        return
    
    def callback_gyro(self, msg: PoseStamped):
        """Callback for the gyro pose."""
        # if self.last_msg is not None and abs(msg.header.stamp.sec%10*1e9 + msg.header.stamp.nanosec - self.last_msg.header.stamp.sec%10*1e9 - self.last_msg.header.stamp.nanosec) > 1e4:
        #     return
        self.x_gyro = msg.pose.position.x
        self.y_gyro = msg.pose.position.y
        quat = [
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z
        ]
        self.O_gyro = quat2euler(quat)[2]

        if self.x_gyro_0 is None:
            self.x_gyro_0 = self.x_gyro
            self.y_gyro_0 = self.y_gyro
            self.O_gyro_0 = self.O_gyro
        
        return
    
    def process_latest_msg(self):
        if self.last_msg is None:
            return
        else:
            # self.get_logger().info("Processing latest message...")
            msg = self.last_msg
            self.last_msg = None
        
        mapping_enco = True
        mapping_gyro = True

        xyz : np.ndarray = read_points_numpy(msg, ["x", "y", "z"])

        # Too small: remove reference and wait for a new one
        if xyz.shape[0] <= self.min_points:
            return
            
        
        
        # Get T from Lidar repere to fixed repere
        if mapping_gyro:
            if (self.x_gyro is None or self.y_gyro is None or self.O_gyro is None):
                return
            T_current_gyro = MappingClass.get_transform(np.array([self.x_gyro, self.y_gyro, 0, self.O_gyro]),
                                                   np.array([self.x_gyro_0, self.y_gyro_0, 0, self.O_gyro_0]))
        if not(mapping_enco or mapping_gyro):
            raise ValueError("No mapping method selected")


        
        
        
        if mapping_enco and mapping_gyro:
            # Fusion avec des moyennes pondérées
            sigma_enco = 0.05
            sigma_gyro = 0.05
            mylambda = sigma_gyro**2 / (sigma_enco**2 + sigma_gyro**2)
            self.ref_T = mylambda * self.T_enco + (1 - mylambda) * T_current_gyro
        elif mapping_enco:
            self.ref_T = self.T_enco.copy()
        elif mapping_gyro:
            self.ref_T = T_current_gyro.copy()
        

        

       
        
        dist_min = 0.01
        # On commence à mapper si on a au moins 'min_points' points
        if self.map is None :
            self.map = xyz
            self.n_map += 5
        else:
            xyz_fixed = MappingClass.apply_transform(xyz, self.ref_T)
            # if self.previous_msg_mapped is not None and abs(self.last_angle - self.O_gyro) > 0.01:
            #     self.get_logger().info(f"test")
            #     # Correction de l'erreur de rotation du Lidar
            #     ref_points : np.ndarray = read_points_numpy(self.previous_msg_mapped, ["x", "y", "z"])
            #     ref_points_fixed = MappingClass.apply_transform(ref_points, self.ref_T)
            #     # pose_vect3 = MappingClass.estimate_pose_from_points(xyz)
            #     # ref_T = np.eye(4)
            #     # ref_T[0:3, 3] = pose_vect3[0:3,0]
            #     # Get T between ref and xyz
            #     T_current = MappingClass.apply_icp(ref_points_fixed,xyz_fixed)
            #     R_current = T_current[:3, :3]
            #     T_current[0:3, 3] = np.zeros((3,))
            #     # Previous T -> current T
            #     self.T_lidar = self.T_lidar.copy() @ T_current
                
            
            # Find nearest neighbors for all points
            nearest_neighbors = MappingClass.find_nearest_neighbors_euclidian(xyz_fixed, self.map)
            # Compute distances
            distances = np.linalg.norm(xyz_fixed - nearest_neighbors, axis=1)
            # Select points that are far enough from their nearest neighbor
            mask = distances > dist_min
            new_points = xyz_fixed[mask]
            if new_points.shape[0] > 0:
                self.map = np.vstack((self.map, new_points))
                self.n_map += 5
            
        # Publish map
        # self.get_logger().info(f"done. Map size: {len(self.map)} points")
        self.previous_msg_mapped = msg
        map_msg = np.hstack((self.map, np.ones((len(self.map), 1))*self.n_map, np.zeros((len(self.map), 1))))
        self.pub_map.publish(create_cloud(msg.header, msg.fields, map_msg))


        if mapping_enco:
            if self.map_enco is None :
                self.map_enco = xyz
                self.x_odom_0 = self.x_odom
                self.y_odom_0 = self.y_odom
                self.O_odom_0 = self.O_odom
            else:
                # Transform points from robot repere to fixed repere using encoder pose
                xyz_fixed_enco = MappingClass.apply_transform(xyz, self.T_enco @ self.T_lidar)
                # Find nearest neighbors for all points
                nearest_neighbors = MappingClass.find_nearest_neighbors_euclidian(xyz_fixed_enco, self.map_enco)
                # Compute distances
                distances = np.linalg.norm(xyz_fixed_enco - nearest_neighbors, axis=1)
                # Select points that are far enough from their nearest neighbor
                mask = distances > dist_min
                new_points = xyz_fixed_enco[mask]
                if new_points.shape[0] > 0:
                    self.map_enco = np.vstack((self.map_enco, new_points))
            # Publish map
            # self.get_logger().info(f"done. Map_enco size: {len(self.map_enco)} points")
            map_msg_enco = np.hstack((self.map_enco, np.ones((len(self.map_enco), 1))*self.n_map, np.zeros((len(self.map_enco), 1))))
            self.pub_map_enco.publish(create_cloud(msg.header, msg.fields, map_msg_enco))

        if mapping_gyro:
            if self.map_gyro is None :
                self.map_gyro = xyz
            else:
                # Transform points from Lidar repere to fixed repere using gyroscope pose
                xyz_fixed_gyro = MappingClass.apply_transform(xyz, T_current_gyro @ self.T_lidar)
                # Find nearest neighbors for all points
                nearest_neighbors = MappingClass.find_nearest_neighbors_euclidian(xyz_fixed_gyro, self.map_gyro)
                # Compute distances
                distances = np.linalg.norm(xyz_fixed_gyro - nearest_neighbors, axis=1)
                # Select points that are far enough from their nearest neighbor
                mask = distances > dist_min
                new_points = xyz_fixed_gyro[mask]
                if new_points.shape[0] > 0:
                    self.map_gyro = np.vstack((self.map_gyro, new_points))
            # Publish map
            # self.get_logger().info(f"done. Map_gyro size: {len(self.map_gyro)} points")
            map_msg_gyro = np.hstack((self.map_gyro, np.ones((len(self.map_gyro), 1))*self.n_map, np.zeros((len(self.map_gyro), 1))))
            self.pub_map_gyro.publish(create_cloud(msg.header, msg.fields, map_msg_gyro))

    @staticmethod
    def estimate_pose_from_points(xyz: np.ndarray) -> np.ndarray:
        """Return the cluster's center as a (3*1) matrix."""
        # TODO: Return the cluster's center as a (3*1) matrix
        xyz_c = np.ndarray([3,1])
        xyz_c[:,0] = np.mean(xyz[:,0]), np.mean(xyz[:,1]), np.mean(xyz[:,2])
        return xyz_c

    @staticmethod
    def best_fit_transform(S, D):
        """Finds the best transform (T) between S and D.
        :param S: source pointcloud
        :param D: destination pointcloud
        :return: (4*4) transform matrix between source and destination
        """
        # Mass center
        centroid_S = np.mean(S, axis=0)
        centroid_D = np.mean(D, axis=0)

        # Center pointclouds in zero
        SS = S - centroid_S
        DD = D - centroid_D
        H = SS.T @ DD

        # If the covariance matrix H has NaN or Inf, something is wrong
        if not np.all(np.isfinite(H)):
            return np.eye(4)

        try:
            U, _, VT = np.linalg.svd(H)
        except np.linalg.LinAlgError:
            # SVD did not converge
            return np.eye(4)

        R_mat = VT.T @ U.T

        # Reflecion detected
        if np.linalg.det(R_mat) < 0:
            VT[2, :] *= -1
            R_mat = VT.T @ U.T

        t = centroid_D - R_mat @ centroid_S

        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = t
        return T

    @staticmethod
    def apply_icp(S: np.ndarray, D: np.ndarray, max_iter: int = 20, tol: float = 1e-10):
        """Apply ICP from S to D.
        :param S: source pointcloud
        :param D: destination pointcloud
        :param max_iter: How many iterations to do at most
        :param tol: Error tolereance between 2 consecutive errors before stopping the process
        :return: (4*4) transform matrix
        """
        T = np.eye(4)
        error_prev = float("-inf")
        error_current = float("inf")
        ST = S
        i = 0

        while i < max_iter and abs(error_current - error_prev) > tol:
            # TODO: Complete ICP process
            voisins_ST_dans_D = MappingClass.find_nearest_neighbors_euclidian(ST, D)
            T_local = MappingClass.best_fit_transform(ST, voisins_ST_dans_D)
            ST = MappingClass.apply_transform(ST, T_local)
            T = T @ T_local
            error_prev = error_current


            error_current = np.mean(MappingClass.find_nearest_neighbors_euclidian(ST, D))
            i += 1

        return T

    @staticmethod
    def apply_transform(xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Apply transform T on points.
        :param points: (N*3) matrix of x, y and z coordinates
        :param T: (4*4) transformation matrix
        :return: (N*3) matrix of transformed x, y and z coordinates
        """
        xyz_homogeneous = np.hstack((xyz, np.ones((xyz.shape[0], 1))))
        return (T @ xyz_homogeneous.T).T[:, :3]

    @staticmethod
    def find_nearest_neighbors_euclidian(S: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Find for each point of S, the closest point in D
        :param S: source pointcloud
        :param D: destination pointcloud
        :return: list of points
        """
        matched = []

        # TODO: let point 's_p' in S, add to 'matched' list, the closest point in 'D' of 'p_s'
        for s_p in S:
            # Calculate the distance from s_p to all points in D
            distances = np.linalg.norm(D - s_p, axis=1)
            # Find the index of the closest point in D
            closest_index = np.argmin(distances)
            # Append the closest point to matched list
            matched.append(D[closest_index])

        return np.array(matched)

    @staticmethod
    def get_transform(Lidar : np.ndarray, Fixed : np.ndarray) -> np.ndarray:
        """Finds the transform (T) from repere Lidar (x,y,z=0, O) to repere fixe (x,y,z=0, O).
        :Lidar point : [xL, yL, zL, OL]
        :Fixed point : [xF, yF, zF, OF]
        :return: (4*4) transform matrix between Lidar and fixed
        """
        T = np.eye(4)
        T[0:3,3] = -Fixed[0:3] + 1*Lidar[0:3]
        # T[0:3,3] = -Lidar[0:3]
        OL = Lidar[3]
        OF = Fixed[3]
        THETA = OL-OF
        T[0:2, 0:2] = [[np.cos(THETA), -np.sin(THETA)],
                       [np.sin(THETA), np.cos(THETA)]]
        # T[0:2, 0:2] = [[np.cos(OL), -np.sin(OL)],
        #                [np.sin(OL), np.cos(OL)]]
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
