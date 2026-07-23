import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_tof_core = get_package_share_directory('tof_core')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    world = LaunchConfiguration('world')
    robot_urdf = LaunchConfiguration('robot_urdf')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': ['-r ', world],
        }.items(),
    )


    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/tof_sim/tof@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            '/tof_sim/tof/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked',
        ],
        output='screen',
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'tof_robot',
            '-file', robot_urdf,
            '-x', '-2.0',
            '-y', '0.0',
            '-z', '0.08',
        ],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value=os.path.join(pkg_tof_core, 'worlds', 'tof_shapes.sdf'),
            description='Gazebo Sim world file.',
        ),
        DeclareLaunchArgument(
            'robot_urdf',
            default_value=PathJoinSubstitution([
                FindPackageShare('tof_sim'),
                'urdf',
                'tof_robot.urdf',
            ]),
            description='URDF file to spawn into the world.',
        ),
        gazebo,
        bridge,
        spawn_robot,
    ])
