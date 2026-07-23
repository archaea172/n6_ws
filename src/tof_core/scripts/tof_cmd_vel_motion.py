#!/usr/bin/env python3
from __future__ import annotations

import math
import time
from collections import deque
from typing import Deque, Tuple

import rclpy
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from tf2_ros import TransformBroadcaster


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_from_yaw(yaw: float):
    q = type('Quaternion', (), {})()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


class ToFCmdVelMotion(Node):
    def __init__(self) -> None:
        super().__init__('tof_cmd_vel_motion')

        self.service_name = self.declare_parameter(
            'service_name', '/world/tof_shapes/set_pose'
        ).value
        self.entity_name = self.declare_parameter('entity_name', 'tof_robot').value
        self.cmd_vel_topic = self.declare_parameter('cmd_vel_topic', '/cmd_vel').value
        self.frame_id = self.declare_parameter('frame_id', 'odom').value
        self.child_frame_id = self.declare_parameter('child_frame_id', 'base_footprint').value

        self.rate_hz = float(self.declare_parameter('rate_hz', 30.0).value)
        self.cmd_timeout = float(self.declare_parameter('cmd_timeout', 0.5).value)
        self.x = float(self.declare_parameter('initial_x', -2.0).value)
        self.y = float(self.declare_parameter('initial_y', 0.0).value)
        self.z = float(self.declare_parameter('initial_z', 0.0).value)
        self.yaw = float(self.declare_parameter('initial_yaw', 0.0).value)
        self.path_length = int(self.declare_parameter('path_length', 600).value)
        self.publish_tf = bool(self.declare_parameter('publish_tf', True).value)

        self.rate_hz = max(self.rate_hz, 1.0)
        self.cmd_timeout = max(self.cmd_timeout, 0.0)
        self.path_length = max(self.path_length, 2)

        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self.last_cmd_time = None
        self.last_update_time = None
        self.pending_request = None
        self.last_wait_log = 0.0
        self.path: Deque[PoseStamped] = deque(maxlen=self.path_length)

        self.client = self.create_client(SetEntityPose, self.service_name)
        self.create_subscription(Twist, self.cmd_vel_topic, self.on_cmd_vel, 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/tof_robot/pose', 10)
        self.path_pub = self.create_publisher(Path, '/tof_robot/path', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        self.timer = self.create_timer(1.0 / self.rate_hz, self.on_timer)
        self.get_logger().info(
            f'2D cmd_vel motion enabled for {self.entity_name}: '
            f'topic={self.cmd_vel_topic}, service={self.service_name}'
        )

    def on_cmd_vel(self, msg: Twist) -> None:
        self.vx = msg.linear.x
        self.vy = msg.linear.y
        self.wz = msg.angular.z
        self.last_cmd_time = self.get_clock().now()

    def on_timer(self) -> None:
        now_wall = time.monotonic()
        if not self.client.service_is_ready():
            self.last_update_time = None
            if now_wall - self.last_wait_log > 2.0:
                self.get_logger().warn(f'Waiting for {self.service_name}')
                self.last_wait_log = now_wall
            return

        if self.pending_request is not None and not self.pending_request.done():
            return
        self.pending_request = None

        now = self.get_clock().now()
        if self.last_update_time is None:
            dt = 0.0
        else:
            dt = (now - self.last_update_time).nanoseconds * 1e-9
            if dt < 0.0 or dt > 1.0:
                dt = 0.0
        self.last_update_time = now

        vx, vy, wz = self.active_twist(now)
        self.integrate(vx, vy, wz, dt)

        pose = self.make_pose()
        stamp = now.to_msg()

        request = SetEntityPose.Request()
        request.entity.name = self.entity_name
        request.entity.type = Entity.MODEL
        request.pose = pose
        self.pending_request = self.client.call_async(request)
        self.pending_request.add_done_callback(self.on_set_pose_done)

        self.publish_state(stamp, pose, vx, vy, wz)

    def active_twist(self, now) -> Tuple[float, float, float]:
        if self.last_cmd_time is None:
            return 0.0, 0.0, 0.0
        age = (now - self.last_cmd_time).nanoseconds * 1e-9
        if self.cmd_timeout > 0.0 and age > self.cmd_timeout:
            return 0.0, 0.0, 0.0
        return self.vx, self.vy, self.wz

    def integrate(self, vx: float, vy: float, wz: float, dt: float) -> None:
        heading = self.yaw
        dx_body = vx * dt
        dy_body = vy * dt

        self.x += math.cos(heading) * dx_body - math.sin(heading) * dy_body
        self.y += math.sin(heading) * dx_body + math.cos(heading) * dy_body
        self.yaw = normalize_angle(heading + wz * dt)

    def make_pose(self) -> Pose:
        q = quaternion_from_yaw(self.yaw)
        pose = Pose()
        pose.position.x = self.x
        pose.position.y = self.y
        pose.position.z = self.z
        pose.orientation.x = q.x
        pose.orientation.y = q.y
        pose.orientation.z = q.z
        pose.orientation.w = q.w
        return pose

    def publish_state(self, stamp, pose: Pose, vx: float, vy: float, wz: float) -> None:
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

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_frame_id
        odom.pose.pose = pose
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = wz
        self.odom_pub.publish(odom)

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


def main() -> None:
    rclpy.init()
    node = ToFCmdVelMotion()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
