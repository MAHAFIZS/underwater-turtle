#!/usr/bin/env python3
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


@dataclass
class Params:
    topic: str = "/target_pose"
    frame_id: str = "odom"
    rate_hz: float = 20.0

    # center
    cx: float = 0.0
    cy: float = 0.0

    # circle / figure8
    radius: float = 1.5
    speed: float = 0.35  # m/s along the curve (approx)

    # wander bounds (IMPORTANT)
    bound_min_x: float = -2.3
    bound_max_x: float = 2.3
    bound_min_y: float = -2.3
    bound_max_y: float = 2.3

    # keep away from walls by this margin
    wall_margin: float = 0.35

    wander_step: float = 0.035
    wander_bias: float = 0.85


class FakeInputs(Node):
    def __init__(self) -> None:
        super().__init__("fake_inputs")

        self.p = Params()

        # ROS params
        self.declare_parameter("mode", "wander")  # circle|figure8|wander
        self.declare_parameter("topic", self.p.topic)
        self.declare_parameter("frame_id", self.p.frame_id)
        self.declare_parameter("rate_hz", self.p.rate_hz)

        self.declare_parameter("cx", self.p.cx)
        self.declare_parameter("cy", self.p.cy)
        self.declare_parameter("radius", self.p.radius)
        self.declare_parameter("speed", self.p.speed)

        self.declare_parameter("bound_min_x", self.p.bound_min_x)
        self.declare_parameter("bound_max_x", self.p.bound_max_x)
        self.declare_parameter("bound_min_y", self.p.bound_min_y)
        self.declare_parameter("bound_max_y", self.p.bound_max_y)
        self.declare_parameter("wall_margin", self.p.wall_margin)

        self.declare_parameter("wander_step", self.p.wander_step)
        self.declare_parameter("wander_bias", self.p.wander_bias)

        # load params
        self.mode = str(self.get_parameter("mode").value).strip().lower()
        self.p.topic = str(self.get_parameter("topic").value)
        self.p.frame_id = str(self.get_parameter("frame_id").value)
        self.p.rate_hz = float(self.get_parameter("rate_hz").value)

        self.p.cx = float(self.get_parameter("cx").value)
        self.p.cy = float(self.get_parameter("cy").value)
        self.p.radius = float(self.get_parameter("radius").value)
        self.p.speed = float(self.get_parameter("speed").value)

        self.p.bound_min_x = float(self.get_parameter("bound_min_x").value)
        self.p.bound_max_x = float(self.get_parameter("bound_max_x").value)
        self.p.bound_min_y = float(self.get_parameter("bound_min_y").value)
        self.p.bound_max_y = float(self.get_parameter("bound_max_y").value)
        self.p.wall_margin = float(self.get_parameter("wall_margin").value)

        self.p.wander_step = float(self.get_parameter("wander_step").value)
        self.p.wander_bias = float(self.get_parameter("wander_bias").value)

        # derived safe bounds (inside wall margin)
        self._xmin = self.p.bound_min_x + self.p.wall_margin
        self._xmax = self.p.bound_max_x - self.p.wall_margin
        self._ymin = self.p.bound_min_y + self.p.wall_margin
        self._ymax = self.p.bound_max_y - self.p.wall_margin

        # publisher
        self.pub_target = self.create_publisher(PoseStamped, self.p.topic, 10)

        # state
        self._theta = 0.0
        self._wx = self.p.cx
        self._wy = self.p.cy
        self._wdir = random.uniform(-math.pi, math.pi)

        period = 1.0 / max(self.p.rate_hz, 1e-6)
        self.timer = self.create_timer(period, self.on_timer)

        self.get_logger().info(
            f"FakeInputs publishing '{self.p.topic}' in frame '{self.p.frame_id}', mode='{self.mode}'. "
            f"safe_bounds x[{self._xmin:.2f},{self._xmax:.2f}] y[{self._ymin:.2f},{self._ymax:.2f}]"
        )

    def _publish_target(self, x: float, y: float) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.p.frame_id
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        self.pub_target.publish(msg)

    def _clamp_in_safe_box(self) -> None:
        # clamp inside safe bounds
        self._wx = min(max(self._wx, self._xmin), self._xmax)
        self._wy = min(max(self._wy, self._ymin), self._ymax)

    def on_timer(self) -> None:
        dt = 1.0 / max(self.p.rate_hz, 1e-6)

        if self.mode == "circle":
            r = max(self.p.radius, 1e-6)
            omega = self.p.speed / r
            self._theta += omega * dt
            x = self.p.cx + r * math.cos(self._theta)
            y = self.p.cy + r * math.sin(self._theta)
            # clamp so circle never commands outside walls
            x = min(max(x, self._xmin), self._xmax)
            y = min(max(y, self._ymin), self._ymax)
            self._publish_target(x, y)
            return

        if self.mode == "figure8":
            r = self.p.radius
            omega = self.p.speed / max(r, 1e-6)
            self._theta += omega * dt
            x = self.p.cx + r * math.sin(self._theta)
            y = self.p.cy + 0.5 * r * math.sin(2.0 * self._theta)
            x = min(max(x, self._xmin), self._xmax)
            y = min(max(y, self._ymin), self._ymax)
            self._publish_target(x, y)
            return

        # default: wander (bounded)
        bias = min(max(self.p.wander_bias, 0.0), 1.0)
        self._wdir = bias * self._wdir + (1.0 - bias) * random.uniform(-math.pi, math.pi)

        self._wx += self.p.wander_step * math.cos(self._wdir)
        self._wy += self.p.wander_step * math.sin(self._wdir)

        # bounce at safe bounds by reflecting direction
        if self._wx <= self._xmin:
            self._wx = self._xmin
            self._wdir = math.pi - self._wdir
        if self._wx >= self._xmax:
            self._wx = self._xmax
            self._wdir = math.pi - self._wdir
        if self._wy <= self._ymin:
            self._wy = self._ymin
            self._wdir = -self._wdir
        if self._wy >= self._ymax:
            self._wy = self._ymax
            self._wdir = -self._wdir

        self._clamp_in_safe_box()
        self._publish_target(self._wx, self._wy)


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