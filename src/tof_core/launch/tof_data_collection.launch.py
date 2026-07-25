from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    linear_x = LaunchConfiguration('linear_x')
    linear_y = LaunchConfiguration('linear_y')
    linear_z = LaunchConfiguration('linear_z')
    angular_x = LaunchConfiguration('angular_x')
    angular_y = LaunchConfiguration('angular_y')
    angular_z = LaunchConfiguration('angular_z')
    duration = LaunchConfiguration('duration')
    frequency = LaunchConfiguration('frequency')
    bag_output_dir = LaunchConfiguration('bag_output_dir')
    bag_name_prefix = LaunchConfiguration('bag_name_prefix')

    cmd_vel_bag_runner = Node(
        package='tof_util',
        executable='cmd_vel_bag_runner.py',
        name='cmd_vel_bag_runner',
        parameters=[{
            'use_sim_time': True,
            'linear_x': ParameterValue(linear_x, value_type=float),
            'linear_y': ParameterValue(linear_y, value_type=float),
            'linear_z': ParameterValue(linear_z, value_type=float),
            'angular_x': ParameterValue(angular_x, value_type=float),
            'angular_y': ParameterValue(angular_y, value_type=float),
            'angular_z': ParameterValue(angular_z, value_type=float),
            'duration': ParameterValue(duration, value_type=float),
            'frequency': ParameterValue(frequency, value_type=float),
            'bag_output_dir': bag_output_dir,
            'bag_name_prefix': bag_name_prefix,
        }],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'linear_x',
            default_value='0.2',
            description='Commanded linear x velocity in m/s.',
        ),
        DeclareLaunchArgument(
            'linear_y',
            default_value='0.0',
            description='Commanded linear y velocity in m/s.',
        ),
        DeclareLaunchArgument(
            'linear_z',
            default_value='0.0',
            description='Commanded linear z velocity in m/s.',
        ),
        DeclareLaunchArgument(
            'angular_x',
            default_value='0.0',
            description='Commanded angular x velocity in rad/s.',
        ),
        DeclareLaunchArgument(
            'angular_y',
            default_value='0.0',
            description='Commanded angular y velocity in rad/s.',
        ),
        DeclareLaunchArgument(
            'angular_z',
            default_value='0.0',
            description='Commanded angular z velocity in rad/s.',
        ),
        DeclareLaunchArgument(
            'duration',
            default_value='10.0',
            description='CmdVel action duration in seconds.',
        ),
        DeclareLaunchArgument(
            'frequency',
            default_value='20.0',
            description='CmdVel publish frequency in Hz.',
        ),
        DeclareLaunchArgument(
            'bag_output_dir',
            default_value='bags/tof_cnn',
            description='Directory where rosbag output folders are created.',
        ),
        DeclareLaunchArgument(
            'bag_name_prefix',
            default_value='cmd_vel_tof',
            description='Prefix used for timestamped rosbag output folders.',
        ),
        cmd_vel_bag_runner,
    ])
