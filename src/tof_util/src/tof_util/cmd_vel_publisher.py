import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class CmdVelPublisher(Node):
    def __init__(self):
        super().__init__('cmd_vel_publisher')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.publish_cmd_vel)

    def publish_cmd_vel(self):
        msg = Twist()
        msg.linear.x = 0.5  # Set linear velocity in x direction
        msg.angular.z = 0.1  # Set angular velocity around z axis
        self.publisher_.publish(msg)

def main_cmd_vel_publisher(args=None):
    rclpy.init(args=args)
    cmd_vel_publisher = CmdVelPublisher()
    rclpy.spin(cmd_vel_publisher)
    cmd_vel_publisher.destroy_node()
    rclpy.shutdown()
