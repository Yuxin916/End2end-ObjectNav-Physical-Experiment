import os
from glob import glob
from setuptools import setup

package_name = 'unitree_g1_sdk_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tsaisplus',
    maintainer_email='tsaisplus0x0@gmail.com',
    description='ROS2 cmd_vel -> Unitree G1 unitree_sdk2 LocoClient bridge (DDS).',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'g1_sdk_control = unitree_g1_sdk_bridge.g1_sdk_control:main',
        ],
    },
)
