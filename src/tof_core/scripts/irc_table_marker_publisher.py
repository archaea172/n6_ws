#!/usr/bin/env python3

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from visualization_msgs.msg import Marker


class IrcTableMarkerPublisher(Node):
    def __init__(self):
        super().__init__('irc_table_marker_publisher')

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self._publisher = self.create_publisher(
            Marker,
            '/visualization_marker',
            qos,
        )
        self._marker = self._create_marker()

        self.create_timer(1.0, self._publish_marker)
        self._publish_marker()

        self.get_logger().info(
            'Publishing IRC table marker in frame odom from '
            'package://tof_sim/meshes/irc_table.glb'
        )

    def _create_marker(self):
        marker = Marker()
        marker.header.frame_id = 'odom'
        marker.ns = 'irc_table'
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.pose.position.z = -0.26
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.lifetime = Duration(seconds=0.0).to_msg()
        marker.frame_locked = False
        marker.mesh_resource = 'package://tof_sim/meshes/irc_table.glb'
        marker.mesh_use_embedded_materials = True
        return marker

    def _publish_marker(self):
        self._marker.header.stamp = self.get_clock().now().to_msg()
        self._publisher.publish(self._marker)


def main():
    rclpy.init()
    node = IrcTableMarkerPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
