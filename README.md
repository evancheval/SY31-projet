# SY31-projet - Cartographie guidée
Ce projet, réalisé dans le cadre de l'UV SY31 (Capteurs pour les systèmes intelligents) à l'Université de Technologie de Compiègne, a pour but de réaliser une cartographie guidée d'un labyrinthe en utilisant un robot équipé d'un Lidar. Le robot se déplace dans l'environnement et collecte des données de distance, qui sont ensuite utilisées pour créer une carte, et donne des indications de navigation en fonction d'instructions dans l'environnement (des flèches pour guider le chemin).
Binôme : 
- LOPEZ VILESPY Etienne
- CHEVALERIAS Evan

## Mise en place
### Machine
Ubuntu 22.04 LTS
### Robot
Robot Turtlebot3 Burger
### Prérequis
- Python 3.10.12 au moins
- ros2
- [colcon package](https://colcon.readthedocs.io/en/released/user/installation.html)

### Installation
Dans un terminal (depuis la racine de votre utilisateur de préférence : Home), suivez les étapes suivantes :
1. Cloner le projet dans le workspace (ici nommé `sy31_projet_elvec`):
    ```bash
    git clone https://github.com/evancheval/SY31-projet.git sy31_projet_elvec
    cd sy31_projet_elvec
    ```
2. Build le projet :
    ```bash
    colcon build --symlink-install
    ```
3. Source le workspace :
    ```bash
    source install/setup.bash
    ```

## Utilisation
### Lancer le robot
Pour lancer le robot, il faut d'abord s'assurer que le robot est allumé et connecté au réseau.
### Exécution
Pour lancer le projet, il faut exécuter la commande suivante dans un terminal (dans le dossier `sy31_projet_elvec`) :
```bash
ros2 launch projet projet.launch.xml
```

Le launch est lancé. Pour visualiser la cartographie, il faut lancer RViz2 dans un autre terminal (dans le dossier `sy31_projet_elvec`). Pour visualiser directement, vous pouvez utiliser le fichier de configuration RViz2 fourni dans le projet :
```bash
rviz2 -f base_scan -d src/projet/rviz2_config.rviz
```

Pour visualiser l'image caméra en temps réel, vous pouvez décommenter la ligne `12` dans le fichier `projet.launch.xml` :
```xml
<node pkg="rqt_image_view" exec="rqt_image_view" name="rqt_image_view" output="screen" />
```
Pour voir la détection, choisissez le topic `detections` dans le menu déroulant en haut à gauche de la fenêtre Rqt qui s'ouvre à l'exécution.

Pour déplacer le robot, vous pouvez utiliser les commandes suivantes dans un autre terminal (dans le dossier `sy31_projet_elvec`) :
```bash
export ROS_DOMAIN_ID=6
TURTLEBOT3_MODEL=burger ros2 run turtlebot3_teleop teleop_keyboard
```

Pour vérifier que la connexion est établie, vous pouvez afficher la liste des topics. Vous devriez voir au moins `/sensor_state` et `/cmd_vel`. Si ce n’est pas le cas, cela signifie probablement que vous avez oublié d’exécuter export ROS_DOMAIN_ID=6 ou que le robot ne s’est pas initialisé correctement.
Dans ce dernier cas, vous pouvez redémarrer ses nœuds en exécutant
```bash
ssh root@192.168.111.1 -t "systemctl restart bringup"
```

### Exemple
Pour tester le projet, vous pouvez lancer dans un autre terminal (toujours dans le dossier `sy31_projet_elvec`) le bag de données fourni dans le projet :	
```bash
ros2 bag play laby2/laby2_0.db3
```
