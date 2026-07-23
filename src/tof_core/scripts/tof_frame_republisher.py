#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2


class ToFFrameRepublisher(Node):
    def __init__(self):
        super().__init__('tof_frame_republisher')

        self.declare_parameter('target_frame', 'tof_sensor_link')
        self.declare_parameter('scan_in', '/tof_sim/raw_tof')
        self.declare_parameter('scan_out', '/tof_sim/tof')
        self.declare_parameter('points_in', '/tof_sim/raw_tof/points')
        self.declare_parameter('points_out', '/tof_sim/tof/points')

        self._target_frame = self.get_parameter(
            'target_frame'
        ).get_parameter_value().string_value
        scan_in = self.get_parameter('scan_in').get_parameter_value().string_value
        scan_out = self.get_parameter('scan_out').get_parameter_value().string_value
        points_in = self.get_parameter('points_in').get_parameter_value().string_value
        points_out = self.get_parameter('points_out').get_parameter_value().string_value

        self._scan_pub = self.create_publisher(
            LaserScan,
            scan_out,
            qos_profile_sensor_data,
        )
        self._points_pub = self.create_publisher(
            PointCloud2,
            points_out,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            LaserScan,
            scan_in,
            self._republish_scan,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2,
            points_in,
            self._republish_points,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f'Republishing ToF topics with frame_id={self._target_frame}'
        )

    def _republish_scan(self, msg):
        msg.header.frame_id = self._target_frame
        self._scan_pub.publish(msg)

    def _republish_points(self, msg):
        msg.header.frame_id = self._target_frame
        self._points_pub.publish(msg)


def main():
    rclpy.init()
    node = ToFFrameRepublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
