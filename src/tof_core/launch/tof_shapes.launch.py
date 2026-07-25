import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_irc_table = get_package_share_directory('irc_table')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    world = LaunchConfiguration('world')
    robot_urdf = LaunchConfiguration('robot_urdf')
    foxglove = LaunchConfiguration('foxglove')
    foxglove_address = LaunchConfiguration('foxglove_address')
    foxglove_port = LaunchConfiguration('foxglove_port')
    lidar_frame = LaunchConfiguration('lidar_frame')
    initial_x = LaunchConfiguration('initial_x')
    initial_y = LaunchConfiguration('initial_y')
    initial_z = LaunchConfiguration('initial_z')
    initial_yaw = LaunchConfiguration('initial_yaw')

    gazebo_resource_path = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(pkg_irc_table, 'models'),
    )
    ignition_resource_path = AppendEnvironmentVariable(
        'IGN_GAZEBO_RESOURCE_PATH',
        os.path.join(pkg_irc_table, 'models'),
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': ['-r ', world],
        }.items(),
    )

    bridge = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='ros_gz_bridge',
                executable='parameter_bridge',
                arguments=[
                    '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                    '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
                    '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
                    '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
                    '/tof_sim/raw_tof@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
                    '/tof_sim/raw_tof/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
                ],
                parameters=[{
                    'qos_overrides./cmd_vel.subscriber.reliability': 'reliable',
                }],
                output='screen',
            ),
        ],
    )

    robot_description = ParameterValue(
        Command(['cat ', robot_urdf]),
        value_type=str,
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
        }],
        output='screen',
    )

    tof_frame_republisher = Node(
        package='tof_core',
        executable='tof_frame_republisher.py',
        name='tof_frame_republisher',
        parameters=[{
            'use_sim_time': True,
            'target_frame': ParameterValue(lidar_frame, value_type=str),
            'scan_in': '/tof_sim/raw_tof',
            'scan_out': '/tof_sim/tof',
            'points_in': '/tof_sim/raw_tof/points',
            'points_out': '/tof_sim/tof/points',
        }],
        output='screen',
    )

    irc_table_marker_publisher = Node(
        package='tof_core',
        executable='irc_table_marker_publisher.py',
        name='irc_table_marker_publisher',
        parameters=[{
            'use_sim_time': True,
        }],
        output='screen',
    )

    cmd_vel_publisher = Node(
        package='tof_util',
        executable='cmd_vel_publisher.py',
        name='cmd_vel_publisher',
        output='screen',
    )

    foxglove_bridge = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        parameters=[{
            'address': ParameterValue(foxglove_address, value_type=str),
            'port': ParameterValue(foxglove_port, value_type=int),
            'use_sim_time': True,
        }],
        output='screen',
        condition=IfCondition(foxglove),
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'tof_robot',
            '-file', robot_urdf,
            '-x', initial_x,
            '-y', initial_y,
            '-z', initial_z,
            '-Y', initial_yaw,
        ],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value=os.path.join(pkg_irc_table, 'worlds', 'irc_table.sdf'),
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
        DeclareLaunchArgument(
            'foxglove',
            default_value='true',
            description='Start foxglove_bridge.',
        ),
        DeclareLaunchArgument(
            'foxglove_address',
            default_value='0.0.0.0',
            description='Bind address used by foxglove_bridge.',
        ),
        DeclareLaunchArgument(
            'foxglove_port',
            default_value='8765',
            description='WebSocket port used by foxglove_bridge.',
        ),
        DeclareLaunchArgument(
            'lidar_frame',
            default_value='tof_sensor_link',
            description='Frame ID used for the public ToF scan and point cloud.',
        ),
        DeclareLaunchArgument(
            'initial_x',
            default_value='-0.25',
            description='Initial robot X position.',
        ),
        DeclareLaunchArgument(
            'initial_y',
            default_value='-0.25',
            description='Initial robot Y position.',
        ),
        DeclareLaunchArgument(
            'initial_z',
            default_value='0.29',
            description='Initial robot Z position.',
        ),
        DeclareLaunchArgument(
            'initial_yaw',
            default_value='0.0',
            description='Initial robot yaw in radians.',
        ),
        gazebo_resource_path,
        ignition_resource_path,
        gazebo,
        bridge,
        robot_state_publisher,
        tof_frame_republisher,
        irc_table_marker_publisher,
        cmd_vel_publisher,
        foxglove_bridge,
        spawn_robot,
    ])
