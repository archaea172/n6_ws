#!/usr/bin/env python3
from __future__ import annotations

import math
import time
from collections import deque
from typing import Deque, Tuple

import rclpy
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from tf2_ros import TransformBroadcaster


def quaternion_from_euler(roll: float, pitch: float, yaw: float):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    q = type('Quaternion', (), {})()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


class ToFMotion3D(Node):
    def __init__(self) -> None:
        super().__init__('tof_motion_3d')

        self.service_name = self.declare_parameter(
            'service_name', '/world/tof_shapes/set_pose'
        ).value
        self.entity_name = self.declare_parameter('entity_name', 'tof_robot').value
        self.frame_id = self.declare_parameter('frame_id', 'world').value
        self.child_frame_id = self.declare_parameter('child_frame_id', 'base_footprint').value
        self.mode = self.declare_parameter('mode', 'figure8').value
        self.heading_mode = self.declare_parameter('heading_mode', 'tangent').value

        self.rate_hz = float(self.declare_parameter('rate_hz', 30.0).value)
        self.period = float(self.declare_parameter('period', 18.0).value)
        self.center_x = float(self.declare_parameter('center_x', -1.6).value)
        self.center_y = float(self.declare_parameter('center_y', 0.0).value)
        self.center_z = float(self.declare_parameter('center_z', 0.35).value)
        self.radius_x = float(self.declare_parameter('radius_x', 1.6).value)
        self.radius_y = float(self.declare_parameter('radius_y', 1.0).value)
        self.z_amplitude = float(self.declare_parameter('z_amplitude', 0.18).value)
        self.roll_amplitude = float(self.declare_parameter('roll_amplitude', 0.08).value)
        self.pitch_amplitude = float(self.declare_parameter('pitch_amplitude', 0.12).value)
        self.yaw_offset = float(self.declare_parameter('yaw_offset', 0.0).value)
        self.path_length = int(self.declare_parameter('path_length', 600).value)
        self.publish_tf = bool(self.declare_parameter('publish_tf', True).value)

        self.rate_hz = max(self.rate_hz, 1.0)
        self.period = max(self.period, 1.0)
        self.path_length = max(self.path_length, 2)

        self.client = self.create_client(SetEntityPose, self.service_name)
        self.pose_pub = self.create_publisher(PoseStamped, '/tof_robot/pose', 10)
        self.path_pub = self.create_publisher(Path, '/tof_robot/path', 10)
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None
        self.path: Deque[PoseStamped] = deque(maxlen=self.path_length)
        self.pending_request = None
        self.start_time = time.monotonic()
        self.last_wait_log = 0.0

        self.timer = self.create_timer(1.0 / self.rate_hz, self.on_timer)
        self.get_logger().info(
            f'3D motion enabled for {self.entity_name}: mode={self.mode}, '
            f'service={self.service_name}'
        )

    def on_timer(self) -> None:
        now_wall = time.monotonic()
        if not self.client.service_is_ready():
            if now_wall - self.last_wait_log > 2.0:
                self.get_logger().warn(f'Waiting for {self.service_name}')
                self.last_wait_log = now_wall
            return

        if self.pending_request is not None and not self.pending_request.done():
            return
        self.pending_request = None

        elapsed = now_wall - self.start_time
        pose = self.make_pose(elapsed)
        stamp = self.get_clock().now().to_msg()

        request = SetEntityPose.Request()
        request.entity.name = self.entity_name
        request.entity.type = Entity.MODEL
        request.pose = pose
        self.pending_request = self.client.call_async(request)
        self.pending_request.add_done_callback(self.on_set_pose_done)

        stamped = PoseStamped()
        stamped.header.stamp = stamp
        stamped.header.frame_id = self.frame_id
        stamped.pose = pose
        self.pose_pub.publish(stamped)

        self.path.append(stamped)
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = self.frame_id
        path.poses = list(self.path)
        self.path_pub.publish(path)

        if self.tf_broadcaster is not None:
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = self.frame_id
            transform.child_frame_id = self.child_frame_id
            transform.transform.translation.x = pose.position.x
            transform.transform.translation.y = pose.position.y
            transform.transform.translation.z = pose.position.z
            transform.transform.rotation = pose.orientation
            self.tf_broadcaster.sendTransform(transform)

    def on_set_pose_done(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001 - keep the timer alive after bridge hiccups.
            self.get_logger().warn(f'Set pose request failed: {exc}')
            self.pending_request = None
            return

        if not response.success:
            self.get_logger().warn('Gazebo rejected the set_pose request')
        self.pending_request = None

    def make_pose(self, elapsed: float) -> Pose:
        x, y, z, roll, pitch, yaw = self.trajectory(elapsed)
        q = quaternion_from_euler(roll, pitch, yaw)

        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.x = q.x
        pose.orientation.y = q.y
        pose.orientation.z = q.z
        pose.orientation.w = q.w
        return pose

    def trajectory(self, elapsed: float) -> Tuple[float, float, float, float, float, float]:
        phase = 2.0 * math.pi * elapsed / self.period
        omega = 2.0 * math.pi / self.period

        if self.mode == 'orbit':
            x = self.center_x + self.radius_x * math.cos(phase)
            y = self.center_y + self.radius_y * math.sin(phase)
            vx = -self.radius_x * omega * math.sin(phase)
            vy = self.radius_y * omega * math.cos(phase)
        elif self.mode == 'sweep':
            x = self.center_x + self.radius_x * math.sin(phase)
            y = self.center_y + self.radius_y * math.sin(0.5 * phase)
            vx = self.radius_x * omega * math.cos(phase)
            vy = 0.5 * self.radius_y * omega * math.cos(0.5 * phase)
        else:
            x = self.center_x + self.radius_x * math.sin(phase)
            y = self.center_y + 0.5 * self.radius_y * math.sin(2.0 * phase)
            vx = self.radius_x * omega * math.cos(phase)
            vy = self.radius_y * omega * math.cos(2.0 * phase)

        z = max(0.05, self.center_z + self.z_amplitude * math.sin(0.5 * phase + 0.4))
        roll = self.roll_amplitude * math.sin(1.7 * phase)
        pitch = self.pitch_amplitude * math.sin(1.3 * phase + 0.2)

        if self.heading_mode == 'center':
            yaw = math.atan2(self.center_y - y, self.center_x - x)
        elif self.heading_mode == 'fixed':
            yaw = self.yaw_offset
        else:
            yaw = math.atan2(vy, vx) + self.yaw_offset

        return x, y, z, roll, pitch, yaw


def main() -> None:
    rclpy.init()
    node = ToFMotion3D()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
