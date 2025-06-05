#! /usr/bin/env python3

import numpy as np
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud, read_points_numpy


class ICPPoseEstimator(Node):
    def __init__(self):
        super().__init__("icp_pose_estimator")
        self.pub_map = self.create_publisher(PointCloud2, "map", 10)

        self.min_points = 5
        self.ref_points = None
        self.ref_T = np.eye(4)
        self.map = []
        self.n_map = 0

        self.create_subscription(PointCloud2, "points", self.maping_callback, 10)

    def maping_callback(self, msg: PointCloud2):
        xyz = read_points_numpy(msg, ["x", "y", "z"])

        # Too small: remove reference and wait for a new one
        if xyz.shape[0] <= self.min_points:
            self.ref_points = None

        # Replace the reference if necessary
        if self.ref_points is None:
            # Reference needs more than 'min_points' xyz
            if xyz.shape[0] > self.min_points:
                pose_vect3 = ICPPoseEstimator.estimate_pose_from_points(xyz)
                self.ref_T = np.eye(4)
                self.ref_T[0:3, 3] = pose_vect3[0:3,0]
                self.ref_points = xyz
            else:
                return
            
        # On commence à mapper si on a au moins 'min_points' points
        if len(self.map) == 0:
            self.map = np.hstack((self.ref_points, np.zeros((len(self.ref_points), 1)), np.zeros((len(self.ref_points), 1)))).tolist()
            self.n_map += 5
            


        #########################

        # if (self.ref_points is not None):
        #     if (xyz.shape[0]<self.ref_points.shape[0]):
        #         self.get_logger().info(f"ref_points trop grand : len(xyz) = {len(xyz)}, len(self.ref_points) = {len(self.ref_points)}")
        #         self.ref_points=self.ref_points[:xyz.shape[0], :]
        #     if (xyz.shape[0]>self.ref_points.shape[0]):
        #         self.get_logger().info(f"xyz trop grand : len(xyz) = {len(xyz)}, len(self.ref_points) = {len(self.ref_points)}")
        #         xyz= xyz[:self.ref_points.shape[0],:]

        #########################
        

        # Get T between ref and xyz
        T_current = ICPPoseEstimator.apply_icp(self.ref_points, xyz, 100)

        # Previous T -> current T
        self.ref_T = self.ref_T @ T_current
        self.ref_points = xyz

        
        # Récupérer les xyz (les 3 premiers éléments) de chaque point de self.map
        map_xyz = np.array([point[:3] for point in self.map])
        map_on_robot_actual_repere = ICPPoseEstimator.apply_transform(map_xyz, self.ref_T)
        nearest_neighbors = ICPPoseEstimator.find_nearest_neighbors_euclidian(xyz, map_on_robot_actual_repere)
        # Update map with the new points
        added_points_to_map = False
        new_points = np.hstack((xyz, np.ones((len(xyz), 1))*self.n_map, np.zeros((len(xyz), 1))))
        for i in range(len(nearest_neighbors)):
            if np.linalg.norm(xyz[i] - nearest_neighbors[i]) > 0.05:
                self.map.append(new_points[i])
                if not(added_points_to_map):
                    self.n_map += 5
                    added_to_map = True


        # Publish map
        # self.get_logger().info(f"done. Map size: {len(self.map)} points")
        self.pub_map.publish(create_cloud(msg.header, msg.fields, self.map))

    @staticmethod
    def estimate_pose_from_points(xyz: np.ndarray) -> np.ndarray:
        """Return the cluster's center as a (3*1) matrix."""
        # TODO: Return the cluster's center as a (3*1) matrix
        xyz_c = np.ndarray([3,1])
        xyz_c[:,0] = np.mean(xyz[:,0]), np.mean(xyz[:,1]), np.mean(xyz[:,2])
        return xyz_c

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
            voisins_ST_dans_D = ICPPoseEstimator.find_nearest_neighbors_euclidian(ST, D)
            T_local = ICPPoseEstimator.best_fit_transform(ST, voisins_ST_dans_D)
            ST = ICPPoseEstimator.apply_transform(ST, T_local)
            T = T @ T_local
            error_prev = error_current


            error_current = np.mean(ICPPoseEstimator.find_nearest_neighbors_euclidian(ST, D))
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


def main(args=None):
    import rclpy

    rclpy.init(args=args)
    node = ICPPoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
