import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
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


SET_POSE_SERVICE = '/world/tof_shapes/set_pose'


def generate_launch_description():
    pkg_tof_core = get_package_share_directory('tof_core')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    world = LaunchConfiguration('world')
    robot_urdf = LaunchConfiguration('robot_urdf')
    foxglove = LaunchConfiguration('foxglove')
    foxglove_address = LaunchConfiguration('foxglove_address')
    foxglove_port = LaunchConfiguration('foxglove_port')
    lidar_frame = LaunchConfiguration('lidar_frame')
    motion = LaunchConfiguration('motion')
    cmd_vel_topic = LaunchConfiguration('cmd_vel_topic')
    motion_rate = LaunchConfiguration('motion_rate')
    cmd_vel_timeout = LaunchConfiguration('cmd_vel_timeout')
    initial_x = LaunchConfiguration('initial_x')
    initial_y = LaunchConfiguration('initial_y')
    initial_z = LaunchConfiguration('initial_z')
    initial_yaw = LaunchConfiguration('initial_yaw')
    odom_frame = LaunchConfiguration('odom_frame')
    base_frame = LaunchConfiguration('base_frame')

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
                    '/tof_sim/raw_tof@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
                    '/tof_sim/raw_tof/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
                    SET_POSE_SERVICE + '@ros_gz_interfaces/srv/SetEntityPose',
                ],
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

    foxglove_bridge = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        parameters=[{
            'address': ParameterValue(foxglove_address, value_type=str),
            'port': ParameterValue(foxglove_port, value_type=int),
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
        ],
        output='screen',
    )

    cmd_vel_motion_driver = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='tof_core',
                executable='tof_cmd_vel_motion.py',
                name='tof_cmd_vel_motion',
                parameters=[{
                    'use_sim_time': True,
                    'service_name': SET_POSE_SERVICE,
                    'entity_name': 'tof_robot',
                    'cmd_vel_topic': ParameterValue(cmd_vel_topic, value_type=str),
                    'rate_hz': ParameterValue(motion_rate, value_type=float),
                    'cmd_timeout': ParameterValue(cmd_vel_timeout, value_type=float),
                    'initial_x': ParameterValue(initial_x, value_type=float),
                    'initial_y': ParameterValue(initial_y, value_type=float),
                    'initial_z': ParameterValue(initial_z, value_type=float),
                    'initial_yaw': ParameterValue(initial_yaw, value_type=float),
                    'frame_id': ParameterValue(odom_frame, value_type=str),
                    'child_frame_id': ParameterValue(base_frame, value_type=str),
                }],
                output='screen',
                condition=IfCondition(motion),
            ),
        ],
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
            'motion',
            default_value='true',
            description='Enable the planar /cmd_vel motion driver.',
        ),
        DeclareLaunchArgument(
            'cmd_vel_topic',
            default_value='/cmd_vel',
            description='Twist topic used to drive the robot.',
        ),
        DeclareLaunchArgument(
            'motion_rate',
            default_value='30.0',
            description='Pose update rate in Hz.',
        ),
        DeclareLaunchArgument(
            'cmd_vel_timeout',
            default_value='0.5',
            description='Seconds before stale cmd_vel commands are treated as zero.',
        ),
        DeclareLaunchArgument(
            'initial_x',
            default_value='-2.0',
            description='Initial robot X position.',
        ),
        DeclareLaunchArgument(
            'initial_y',
            default_value='0.0',
            description='Initial robot Y position.',
        ),
        DeclareLaunchArgument(
            'initial_z',
            default_value='0.0',
            description='Initial robot Z position.',
        ),
        DeclareLaunchArgument(
            'initial_yaw',
            default_value='0.0',
            description='Initial robot yaw in radians.',
        ),
        DeclareLaunchArgument(
            'odom_frame',
            default_value='odom',
            description='Odometry frame for TF, pose, and path output.',
        ),
        DeclareLaunchArgument(
            'base_frame',
            default_value='base_footprint',
            description='Robot base frame driven by cmd_vel.',
        ),
        gazebo,
        bridge,
        robot_state_publisher,
        tof_frame_republisher,
        foxglove_bridge,
        spawn_robot,
        cmd_vel_motion_driver,
    ])
