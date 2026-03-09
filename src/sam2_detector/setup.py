from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'sam2_detector'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         [f'resource/{package_name}']),
        (f'share/{package_name}',              ['package.xml']),
        (f'share/{package_name}/config',       glob('config/*.yaml')),
        (f'share/{package_name}/launch',       glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='GroundingDINO + SAM2 detector ROS 2 node for object navigation',
    license='MIT',
    entry_points={
        'console_scripts': [
            f'sam2_detector = {package_name}.sam2_detector_node:main',
        ],
    },
)
