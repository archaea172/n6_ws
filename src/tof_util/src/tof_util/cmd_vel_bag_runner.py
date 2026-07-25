import datetime as dt
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from geometry_msgs.msg import Twist
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String
from tof_msgs.action import CmdVel


DEFAULT_BAG_TOPICS = [
    '/clock',
    '/cmd_vel',
    '/odom',
    '/tf',
    '/tf_static',
    '/tof_sim/tof',
    '/tof_sim/tof/points',
    '/tof_sim/raw_tof',
    '/tof_sim/raw_tof/points',
    '/data_collection/event',
]

GOAL_STATUS_NAMES = {
    0: 'unknown',
    1: 'accepted',
    2: 'executing',
    3: 'canceling',
    4: 'succeeded',
    5: 'canceled',
    6: 'aborted',
}


class CmdVelBagRunner(Node):
    def __init__(self):
        super().__init__('cmd_vel_bag_runner')

        self.declare_parameter('action_name', 'cmd_vel_action')
        self.declare_parameter('event_topic', '/data_collection/event')
        self.declare_parameter('bag_output_dir', 'bags/tof_cnn')
        self.declare_parameter('bag_name_prefix', 'cmd_vel_tof')
        self.declare_parameter('topics', DEFAULT_BAG_TOPICS)

        self.declare_parameter('linear_x', 0.2)
        self.declare_parameter('linear_y', 0.0)
        self.declare_parameter('linear_z', 0.0)
        self.declare_parameter('angular_x', 0.0)
        self.declare_parameter('angular_y', 0.0)
        self.declare_parameter('angular_z', 0.0)
        self.declare_parameter('duration', 10.0)
        self.declare_parameter('frequency', 20.0)

        self.declare_parameter('action_server_timeout_sec', 15.0)
        self.declare_parameter('rosbag_startup_delay_sec', 2.0)
        self.declare_parameter('action_result_timeout_margin_sec', 30.0)
        self.declare_parameter('post_action_record_delay_sec', 0.5)
        self.declare_parameter('rosbag_shutdown_timeout_sec', 10.0)

        self._action_name = self.get_parameter('action_name').value
        self._event_topic = self.get_parameter('event_topic').value
        self._topics = self._get_topics()
        self._bag_path = self._make_bag_path()

        self._duration = float(self.get_parameter('duration').value)
        self._frequency = float(self.get_parameter('frequency').value)
        self._action_server_timeout_sec = float(
            self.get_parameter('action_server_timeout_sec').value
        )
        self._rosbag_startup_delay_sec = float(
            self.get_parameter('rosbag_startup_delay_sec').value
        )
        self._action_result_timeout_margin_sec = float(
            self.get_parameter('action_result_timeout_margin_sec').value
        )
        self._post_action_record_delay_sec = float(
            self.get_parameter('post_action_record_delay_sec').value
        )
        self._rosbag_shutdown_timeout_sec = float(
            self.get_parameter('rosbag_shutdown_timeout_sec').value
        )

        self._action_client = ActionClient(self, CmdVel, self._action_name)
        self._event_pub = self.create_publisher(String, self._event_topic, 10)
        self._bag_process = None

    def run(self):
        self._validate_parameters()
        action_succeeded = False

        try:
            self._start_rosbag()
            self._sleep(self._rosbag_startup_delay_sec)
            self._publish_event('bag_started')

            status = self._send_cmd_vel_goal()
            status_name = GOAL_STATUS_NAMES.get(status, f'unknown_{status}')
            self._publish_event('action_finished', status=status_name)
            action_succeeded = status == 4

            if action_succeeded:
                self.get_logger().info('CmdVel action succeeded.')
            else:
                self.get_logger().error(f'CmdVel action finished with status: {status_name}')

            return action_succeeded
        except Exception as exc:
            self.get_logger().error(f'CmdVel bag run failed: {exc}')
            self._publish_event('run_failed', error=str(exc))
            return False
        finally:
            if self._bag_process is not None:
                self._publish_event('bag_stopping')
                self._sleep(self._post_action_record_delay_sec)
                self._stop_rosbag()

    def _validate_parameters(self):
        if self._frequency <= 0.0:
            raise ValueError('frequency must be greater than 0.0')

        if self._duration < 0.0:
            raise ValueError('duration must be non-negative')

        if not self._topics:
            raise ValueError('topics must contain at least one topic name')

    def _get_topics(self):
        topics = self.get_parameter('topics').value
        if isinstance(topics, str):
            return [topic.strip() for topic in topics.split(',') if topic.strip()]

        return [str(topic) for topic in topics]

    def _make_bag_path(self):
        output_dir = Path(self.get_parameter('bag_output_dir').value).expanduser()
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir

        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        bag_name = f'{self.get_parameter("bag_name_prefix").value}_{timestamp}'
        return output_dir / bag_name

    def _start_rosbag(self):
        command = [
            'ros2',
            'bag',
            'record',
            '--use-sim-time',
            '-o',
            str(self._bag_path),
            *self._topics,
        ]

        self.get_logger().info(f'Starting rosbag: {" ".join(command)}')
        self._bag_process = subprocess.Popen(
            command,
            preexec_fn=os.setsid,
        )

        self._sleep(0.2)
        if self._bag_process.poll() is not None:
            raise RuntimeError(
                f'ros2 bag record exited early with code {self._bag_process.returncode}'
            )

    def _send_cmd_vel_goal(self):
        self.get_logger().info(f'Waiting for action server: {self._action_name}')
        if not self._action_client.wait_for_server(
            timeout_sec=self._action_server_timeout_sec
        ):
            raise RuntimeError(
                f'action server was not available after '
                f'{self._action_server_timeout_sec:.1f}s'
            )

        goal = CmdVel.Goal()
        goal.vel = self._make_twist()
        goal.duration = self._duration
        goal.frequency = self._frequency

        self._publish_event('action_goal_sent')
        send_goal_future = self._action_client.send_goal_async(goal)
        goal_handle = self._wait_for_future(
            send_goal_future,
            self._action_server_timeout_sec,
            'CmdVel goal request',
        )

        if not goal_handle.accepted:
            self._publish_event('action_finished', status='rejected')
            raise RuntimeError('CmdVel action goal was rejected')

        result_timeout = self._duration + self._action_result_timeout_margin_sec
        result_future = goal_handle.get_result_async()
        result_response = self._wait_for_future(
            result_future,
            result_timeout,
            'CmdVel action result',
        )
        return int(result_response.status)

    def _make_twist(self):
        twist = Twist()
        twist.linear.x = float(self.get_parameter('linear_x').value)
        twist.linear.y = float(self.get_parameter('linear_y').value)
        twist.linear.z = float(self.get_parameter('linear_z').value)
        twist.angular.x = float(self.get_parameter('angular_x').value)
        twist.angular.y = float(self.get_parameter('angular_y').value)
        twist.angular.z = float(self.get_parameter('angular_z').value)
        return twist

    def _wait_for_future(self, future, timeout_sec, description):
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(f'timed out waiting for {description}')

            rclpy.spin_once(self, timeout_sec=min(0.1, remaining))

        if not future.done():
            raise RuntimeError(f'{description} did not complete')

        return future.result()

    def _publish_event(self, event_name, **fields):
        message = String()
        tokens = [
            event_name,
            f'bag_path={self._bag_path}',
            *[f'{key}={value}' for key, value in fields.items()],
        ]
        message.data = ' '.join(tokens)
        self._event_pub.publish(message)
        self.get_logger().info(message.data)

    def _stop_rosbag(self):
        process = self._bag_process
        self._bag_process = None

        if process.poll() is not None:
            self.get_logger().info(
                f'rosbag process already exited with code {process.returncode}'
            )
            return

        self.get_logger().info('Stopping rosbag with SIGINT.')
        os.killpg(os.getpgid(process.pid), signal.SIGINT)

        try:
            process.wait(timeout=self._rosbag_shutdown_timeout_sec)
            self.get_logger().info(f'rosbag stopped with code {process.returncode}')
        except subprocess.TimeoutExpired:
            self.get_logger().warn('rosbag did not stop after SIGINT; sending SIGTERM.')
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.get_logger().error('rosbag did not stop after SIGTERM; killing it.')
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait(timeout=2.0)

    def _sleep(self, seconds):
        end_time = time.monotonic() + max(0.0, seconds)
        while rclpy.ok() and time.monotonic() < end_time:
            rclpy.spin_once(self, timeout_sec=min(0.1, end_time - time.monotonic()))


def main_cmd_vel_bag_runner(args=None):
    rclpy.init(args=args)
    runner = CmdVelBagRunner()

    try:
        success = runner.run()
    except KeyboardInterrupt:
        runner.get_logger().info('Interrupted.')
        success = False
    finally:
        runner.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return 0 if success else 1


def main(args=None):
    return main_cmd_vel_bag_runner(args=args)


if __name__ == '__main__':
    sys.exit(main())
