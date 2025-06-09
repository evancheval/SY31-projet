from setuptools import find_packages, setup

package_name = 'projet'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['projet.launch.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='evancheval',
    maintainer_email='evan.chevalerias@etu.utc.fr',
    description='Projet SY31 - Cartographie Guidée',
    license='CC BY-SA 4.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "decompressor = projet.decompressor:main",
            "detections = projet.detect:main",
            "transformer = projet.transformer:main",
            "icp_pose_estimator = projet.icp_pose_estimator:main",
            "odom2pose = projet.odom2pose:main",
            "mapping = projet.mapping:main",            
        ],
    },
)
