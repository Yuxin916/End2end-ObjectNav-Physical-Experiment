from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'vln_bridge'
package_files = []

for root, _, files in os.walk(os.path.join(package_name, 'helper')):
    for filename in files:
        package_files.append(os.path.relpath(os.path.join(root, filename), package_name))

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    package_data={
        package_name: package_files,
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config',
            glob('config/*.yaml')),
        ('share/' + package_name + '/launch',
            glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='VLM object navigation bridge for ROS 2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vlm_navigator = vln_bridge.vlm_navigator_node:main',
        ],
    },
)
