import threading
import time

from geometry_msgs.msg import Twist
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from tof_msgs.action import CmdVel


class CmdVelPublisher(Node):
    def __init__(self):
        super().__init__('cmd_vel_publisher')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self._active_goal = False
        self._active_goal_lock = threading.Lock()
        self._action_server = ActionServer(
            self,
            CmdVel,
            'cmd_vel_action',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )

    def destroy(self):
        self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        if goal_request.frequency <= 0.0:
            self.get_logger().warn('Rejecting CmdVel goal: frequency must be greater than 0.')
            return GoalResponse.REJECT

        if goal_request.duration < 0.0:
            self.get_logger().warn('Rejecting CmdVel goal: duration must be non-negative.')
            return GoalResponse.REJECT

        with self._active_goal_lock:
            if self._active_goal:
                self.get_logger().warn('Rejecting CmdVel goal: another goal is already active.')
                return GoalResponse.REJECT

            self._active_goal = True

        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request for CmdVel goal.')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        request = goal_handle.request
        result = CmdVel.Result()

        try:
            self.get_logger().info(
                f'Executing CmdVel goal for {request.duration:.3f}s at {request.frequency:.3f}Hz.'
            )

            if request.duration == 0.0:
                self.publish_stop()
                goal_handle.succeed()
                return result

            period = 1.0 / request.frequency
            deadline = time.monotonic() + request.duration

            while time.monotonic() < deadline:
                if goal_handle.is_cancel_requested:
                    self.publish_stop()
                    goal_handle.canceled()
                    self.get_logger().info('CmdVel goal canceled.')
                    return result

                self.publisher_.publish(request.vel)
                self._sleep_until_next_publish(period, deadline)

            self.publish_stop()
            goal_handle.succeed()
            self.get_logger().info('CmdVel goal succeeded.')
            return result
        finally:
            with self._active_goal_lock:
                self._active_goal = False

    def publish_stop(self):
        self.publisher_.publish(Twist())

    def _sleep_until_next_publish(self, period, deadline):
        end_time = min(time.monotonic() + period, deadline)
        while time.monotonic() < end_time:
            time.sleep(min(0.01, end_time - time.monotonic()))


def main_cmd_vel_publisher(args=None):
    rclpy.init(args=args)
    cmd_vel_publisher = CmdVelPublisher()
    executor = MultiThreadedExecutor()

    try:
        rclpy.spin(cmd_vel_publisher, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        cmd_vel_publisher.destroy()
        if rclpy.ok():
            rclpy.shutdown()
