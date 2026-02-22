#!/usr/bin/env python3
from __future__ import annotations

import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class FakeInputs(Node):
    def __init__(self) -> None:
        super().__init__("fake_inputs")

        self.pub_robot = self.create_publisher(PoseStamped, "/robot_pose", 10)
        self.pub_target = self.create_publisher(PoseStamped, "/target_pose", 10)

        self.t0 = time.time()
        self.timer = self.create_timer(0.05, self.on_timer)  # 20 Hz

        self.get_logger().info("FakeInputs started. Publishing /robot_pose and /target_pose")

    def on_timer(self) -> None:
        t = time.time() - self.t0

        # Robot pose: small circle around (-1, -1)
        robot = PoseStamped()
        robot.header.stamp = self.get_clock().now().to_msg()
        robot.header.frame_id = "map"
        robot.pose.position.x = -1.0 + 0.3 * math.cos(t)
        robot.pose.position.y = -1.0 + 0.3 * math.sin(t)
        robot.pose.position.z = 0.0
        robot.pose.orientation.w = 1.0

        # Target pose: fixed point (2, 2)
        target = PoseStamped()
        target.header.stamp = robot.header.stamp
        target.header.frame_id = "odom"
        target.pose.position.x = 2.0
        target.pose.position.y = 2.0
        target.pose.position.z = 0.0
        target.pose.orientation.w = 1.0

        self.pub_robot.publish(robot)
        self.pub_target.publish(target)


def main() -> None:
    rclpy.init()
    node = FakeInputs()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
