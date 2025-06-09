#! /usr/bin/env python3

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from turtlebot3_msgs.msg import SensorState


class Detector(Node):
    def __init__(self):
        super().__init__("detector")

        # pour gérer les affichages
        self.instruction = ""
        self.instruction2 = ""

        # Initialisation du laps de temps parcouru par le son du sonar au mur (en µs)
        # Initialisation à 1000 µs pour éviter les erreurs de détection au démarrage
        self.dt: np.float32 = 1000.0
        # Taille du filtre médian pour le traitement de l'erreur sur le sonar
        self.median_filter_size = 10
        self.last_n_dt = []

        self.bridge = CvBridge()
        # Publisher pour les images traitées (renvoie l'image si pas de détection)
        self.pub_detect = self.create_publisher(Image, "detections", 1)
        # Publisher pour les instructions de navigation
        self.pub_instructions = self.create_publisher(String, "instructions", 1)

        # Déterminer les valeurs minimales et maximales des pixels à filtrer
        # Utilisation de tableaux à deux dimensions pour le cas des couleurs à deux intervalles (comme le rouge)
        # Masque flèche bleue (HSV) :
        # 205/360 hue -> intervalle entre 160 et 230
        #             -> OpenCV : 80 à 115
        self.blue_min = np.array([[80, 80, 80]], dtype=np.uint8)
        self.blue_max = np.array([[115, 255, 250]], dtype=np.uint8)

        # Masque flèche rouge (HSV) :
        # 0/360 hue -> intervalle entre 0 et 20 et entre 280 et 360 (deux intervalles, d'où le tableau à deux lignes)
        #             -> OpenCV : 0 à 10 et 140 à 180
        self.red_min = np.array([[0, 100, 10], [140, 100, 10]], dtype=np.uint8)
        self.red_max = np.array([[10, 230, 245], [180, 230, 245]], dtype=np.uint8)

        # Abonnement aux messages de la caméra et du sonar
        # Pour la caméra, possibilité de choisir "image_rect" plutôt que "image_raw"
        # si le noeud image_proc est utilisé (à décommenter dans projet.launch.xml)
        # "image_raw" est publié manuellement dans le noeud decompress
        # "image_rect" est publié automatiquement par le noeud image_proc grâce à un fichier de calibration

        # self.sub_camera = self.create_subscription(Image, "image_rect", self.callback_image, 1)
        self.sub_camera = self.create_subscription(
            Image, "image_raw", self.callback_image, 1
        )
        # Abonnement au sonar
        self.sub_sonar = self.create_subscription(
            SensorState, "sensor_state", self.callback_sonar, 1
        )

    def callback_sonar(self, msg: SensorState):
        """Met à jour le laps de temps parcouru par le son du sonar au mur (en µs)"""
        self.dt = msg.sonar
        if len(self.last_n_dt) >= self.median_filter_size:
            self.last_n_dt.pop()
            self.last_n_dt.append(self.dt)

    def callback_image(self, msg: Image):
        """Procède à la détection des flèches dans l'image reçue,
        seulement si on s'approche du mur"""

        # pour éviter que le message de direction s'affiche plus d'une fois
        if self.dt > 1000:
            self.instruction = ""
            self.instruction2 = ""

        # Si self.dt = 650 µs, alors d'après la documentation du constructeur,
        # pour un TurtleBot3 Burger, la distance au mur est de
        # 340 m/s * 650 µs / 2 = 0.1105 m = 11.05 cm.
        # On minore self.dt pour éviter les valeurs aberrantes.
        if self.dt < 650.0 and self.dt > 10.0:
            # Convertir l'image ROS en OpenCV
            try:
                img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except CvBridgeError as e:
                self.get_logger().warn(f"ROS->OpenCV {e}")
                return

            img_out = self.detect(img, False)

            # Convertir l'image OpenCV en image ROS
            try:
                format = "bgr8" if img_out.ndim == 3 else "mono8"
                msg_out = self.bridge.cv2_to_imgmsg(img_out, format)
            except CvBridgeError as e:
                self.get_logger().warn(f"ROS->OpenCV {e}")
                return

            self.pub_detect.publish(msg_out)
        else:
            # Publication de l'image classique si la distance est trop grande :
            # la détection n'a pas lieu, pas besoin d'indications de directions.
            self.pub_detect.publish(msg)

    def detect(
        self, img: np.ndarray, direction_fleche_sujet: bool = True
    ) -> np.ndarray:
        """
        Détecte la flèche dans l'image et indique la direction à prendre en fonction de sa couleur.
        Le paramètre `direction_fleche_sujet` indique si la flèche est orientée de la même manière
        qu'indiqué dans le sujet (True : rouge = gauche et bleu = droite),
        ou si elle est orientée dans le sens opposé (False : rouge = droite et bleu = gauche).
        """

        # Convertit l'image en HSV si elle est en BGR,
        # pour appliquer les masques de couleur.
        # BGR2HSV = 40
        img = cv2.cvtColor(img, 40)

        mask_blue = cv2.inRange(img, self.blue_min[0], self.blue_max[0])
        mask_red = cv2.inRange(img, self.red_min[0], self.red_max[0])
        # Pour le rouge, on doit ajouter les deux intervalles
        # car OpenCV ne gère pas les intervalles de couleur circulaires.
        for i in range(1, len(self.red_min)):
            mask_red = mask_red + cv2.inRange(img, self.red_min[i], self.red_max[i])

        # Reconvertit l'image en BGR pour l'affichage,
        # HSV2BGR = 54
        img = cv2.cvtColor(img, 54)

        contours_blue, _ = cv2.findContours(
            mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        contours_red, _ = cv2.findContours(
            mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        areamax_red, areamax_blue, imax_red, imax_blue = 0, 0, 0, 0
        if not (len(contours_blue) == 0 and len(contours_red) == 0):
            # Trouve le contour le plus grand pour chaque couleur.
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
            # Si les deux couleurs sont détectées (peut arriver dans certains cas),
            # on compare les aires maximales : la couleur qui a la plus grande aire
            # est celle qui est la plus proche, soit celle qui donne l'instruction de navigation.
            if areamax_red > areamax_blue:
                areamax = areamax_red
                imax = imax_red
                contours = contours_red
                if direction_fleche_sujet:
                    direction = "Gauche"
                else:
                    direction = "Droite"
            else:
                areamax = areamax_blue
                imax = imax_blue
                contours = contours_blue
                if direction_fleche_sujet:
                    direction = "Droite"
                else:
                    direction = "Gauche"

            # Différent message à afficher
            if self.dt < 500 and direction != self.instruction2:
                self.instruction2 = direction
                self.get_logger().info(f"{direction} URGENT !")
                # self.pub_instructions.publish(String(f"{direction} URGENT !"))

            if direction != self.instruction:
                self.instruction = direction
                self.get_logger().info(direction)
                self.pub_instructions.publish(String(direction))

            

            if areamax > 1.0:
                cv2.drawContours(img, contours, imax, (0, 255, 0), 3, 8)
                # Dessine un cercle sur l'image au centre de la flèche détectée
                M = np.mean(contours[imax], axis=0)[0]
                cv2.circle(img, (int(M[0]), (int(M[1]))), 2, (0, 255, 255), 2)

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