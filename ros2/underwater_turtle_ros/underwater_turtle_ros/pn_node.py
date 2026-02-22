# underwater_turtle_ros/pn_node.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from underwater_turtle_ros.controllers.pn import PNController, PNParams, clamp, wrap_angle


@dataclass
class NodeParams:
    # Topics
    odom_topic: str = "/odom"
    target_pose_topic: str = "/target_pose"
    scan_topic: str = "/scan"
    cmd_vel_topic: str = "/cmd_vel"

    # Loop rate
    rate_hz: float = 20.0

    # Command limits (stability-first)
    # NOTE: TurtleBot3 can tip/flip in sim if you keep v too high near collisions.
    v_max: float = 0.15
    omega_abs_max: float = 0.70

    # Obstacle safety (front cone)
    obs_slow_dist: float = 0.55      # start blending avoidance when closer than this
    obs_stop_dist: float = 0.32      # full avoidance at/inside this (emergency uses 0.85*obs_stop_dist)
    front_cone_deg: float = 40.0     # +/- degrees for front cone
    avoid_turn_omega: float = 0.45   # turn rate while avoiding (signed by chosen side)

    # Scan geometry
    # Your scan angles are 0..2pi; for TurtleBot3, front is typically 0 rad.
    # If your setup is rotated, change this, but start with 0.0.
    front_center_rad: float = 0.0

    # Contact freeze (prevents "push into wall + flip")
    contact_dist: float = 0.16       # a bit above range_min=0.12
    contact_hold_s: float = 0.50     # seconds to command (0,0) after contact


class PNNode(Node):
    def __init__(self) -> None:
        super().__init__("underwater_turtle_pn_node")

        # Params (could later be wired to ROS parameters/YAML)
        self.p = NodeParams()

        # Controller params (PN)
        # Keep PN caps aligned with node caps for predictability
        pn_params = PNParams()
        pn_params.v_abs_max = min(float(pn_params.v_abs_max), float(self.p.v_max))
        pn_params.omega_abs_max = min(float(pn_params.omega_abs_max), float(self.p.omega_abs_max))
        self.ctrl = PNController(pn_params)

        # State buffers
        self._odom: Optional[Odometry] = None
        self._target: Optional[PoseStamped] = None
        self._scan: Optional[LaserScan] = None
        self._t_last: Optional[float] = None

        # ---- Obstacle smoothing / hysteresis memory ----
        self._min_front_filt: float = float("inf")
        self._avoid_side: int = 0            # +1 left, -1 right, 0 unknown
        self._avoid_side_until: float = 0.0  # hold time until (sec)

        # ---- Contact freeze memory ----
        self._contact_until: float = 0.0

        # ---- Tuning knobs (keep as node fields for quick iteration) ----
        self._obs_alpha: float = 0.45        # EMA for min_front (0..1), higher = smoother
        self._side_hold_s: float = 0.80      # keep chosen side stable
        self._side_margin_m: float = 0.20    # only switch if clearly better
        self._v_min_near: float = 0.06       # minimum speed during avoidance blend (prevents "crawl lock")

        # Additional node-level coupling: reduce v when turning hard (stability)
        self._turn_v_omega1: float = 0.40
        self._turn_v_cap1: float = 0.08
        self._turn_v_omega2: float = 0.60
        self._turn_v_cap2: float = 0.05

        # Pub/Sub
        self.pub_cmd = self.create_publisher(Twist, self.p.cmd_vel_topic, 10)

        self.create_subscription(Odometry, self.p.odom_topic, self._on_odom, 10)
        self.create_subscription(PoseStamped, self.p.target_pose_topic, self._on_target, 10)
        self.create_subscription(LaserScan, self.p.scan_topic, self._on_scan, 10)

        # Timer loop
        period = 1.0 / max(self.p.rate_hz, 1e-6)
        self.timer = self.create_timer(period, self._step)

        self.get_logger().info(
            f"PNNode odom='{self.p.odom_topic}', target='{self.p.target_pose_topic}', scan='{self.p.scan_topic}', cmd='{self.p.cmd_vel_topic}'"
        )
        self.get_logger().info(
            f"Front cone centered at {self.p.front_center_rad:.3f} rad (0 typically forward for 0..2pi scans)."
        )

    # ---------------- Callbacks ----------------

    def _on_odom(self, msg: Odometry) -> None:
        self._odom = msg

    def _on_target(self, msg: PoseStamped) -> None:
        self._target = msg

    def _on_scan(self, msg: LaserScan) -> None:
        self._scan = msg

    # ---------------- Helpers ----------------

    @staticmethod
    def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
        # yaw from quaternion
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _smoothstep(x: float) -> float:
        x = clamp(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _obs_weight(self, d: float) -> float:
        """
        Smooth weight w(d): 0 far, 1 near.
        Uses obs_slow_dist as "start influence" and obs_stop_dist as "full influence".
        """
        sd = float(self.p.obs_slow_dist)
        st = float(self.p.obs_stop_dist)
        if d <= st:
            return 1.0
        if d >= sd:
            return 0.0
        t = (sd - d) / max(sd - st, 1e-6)  # 0..1
        return self._smoothstep(t)

    def _choose_side(self, min_left: float, min_right: float, now: float) -> int:
        """
        Choose +1 (left) or -1 (right) with hold + margin to prevent flip-flop.
        """
        if now < self._avoid_side_until and self._avoid_side != 0:
            return self._avoid_side

        m = float(self._side_margin_m)

        if min_left > min_right + m:
            side = +1
        elif min_right > min_left + m:
            side = -1
        else:
            side = self._avoid_side if self._avoid_side != 0 else -1

        self._avoid_side = side
        self._avoid_side_until = now + float(self._side_hold_s)
        return side

    def _front_cone_metrics(self) -> Tuple[float, float, float]:
        """
        Returns (min_front, min_left, min_right) inside front cone.

        Works with scans that span 0..2pi by using wrap_angle around front_center_rad.
        """
        if self._scan is None:
            return float("inf"), float("inf"), float("inf")

        scan = self._scan
        cone = math.radians(float(self.p.front_cone_deg))
        center = float(self.p.front_center_rad)

        min_front = float("inf")
        min_left = float("inf")
        min_right = float("inf")

        for i, r in enumerate(scan.ranges):
            if r is None:
                continue
            if math.isinf(r) or math.isnan(r) or r <= 0.0:
                continue

            ang = scan.angle_min + i * scan.angle_increment
            err = wrap_angle(ang - center)  # [-pi, pi]

            if abs(err) > cone:
                continue

            if r < min_front:
                min_front = r
            if err >= 0.0:
                if r < min_left:
                    min_left = r
            else:
                if r < min_right:
                    min_right = r

        return min_front, min_left, min_right

    def _publish(self, v: float, omega: float) -> None:
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(omega)
        self.pub_cmd.publish(msg)

    # ---------------- Main loop ----------------

    def _step(self) -> None:
        if self._odom is None or self._target is None:
            return

        # Time / dt
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._t_last is None:
            self._t_last = now
            return
        dt = max(now - self._t_last, 1e-6)
        self._t_last = now

        # Robot state from odom
        ox = self._odom.pose.pose.position.x
        oy = self._odom.pose.pose.position.y
        q = self._odom.pose.pose.orientation
        psi = self._yaw_from_quat(q.x, q.y, q.z, q.w)

        # Target
        tx = self._target.pose.position.x
        ty = self._target.pose.position.y

        # --- PN control ---
        omega_cmd, v_cmd = self.ctrl.compute(
            tx=tx, ty=ty,
            x=ox, y=oy, psi=psi,
            v_max=float(self.p.v_max),
            dt=float(dt),
        )

        # Node-level safety clamps
        omega_cmd = clamp(float(omega_cmd), -float(self.p.omega_abs_max), float(self.p.omega_abs_max))
        v_cmd = clamp(float(v_cmd), 0.0, float(self.p.v_max))

        # --- Obstacle metrics ---
        min_front, min_left, min_right = self._front_cone_metrics()

        # EMA filter for min_front to reduce jitter from single rays
        if math.isfinite(min_front):
            a = clamp(float(self._obs_alpha), 0.0, 0.99)
            if not math.isfinite(self._min_front_filt):
                self._min_front_filt = float(min_front)
            else:
                self._min_front_filt = a * float(min_front) + (1.0 - a) * float(self._min_front_filt)

        d_front = float(self._min_front_filt) if math.isfinite(self._min_front_filt) else float(min_front)

        # ---- Contact freeze (prevents pushing/torquing into obstacle and flipping) ----
        if d_front < float(self.p.contact_dist):
            self._contact_until = now + float(self.p.contact_hold_s)

        if now < self._contact_until:
            self._publish(0.0, 0.0)
            return

        # ---- Smooth avoidance blend ----
        w = self._obs_weight(d_front)  # 0..1

        if w > 0.0:
            side = self._choose_side(min_left, min_right, now)  # +left / -right
            omega_avoid = float(side) * abs(float(self.p.avoid_turn_omega))

            # blend omega (no snap switching)
            omega_cmd = (1.0 - w) * float(omega_cmd) + w * float(omega_avoid)

            # blend v: slow but keep floor (avoid crawl lock)
            v_cmd = (1.0 - w) * float(v_cmd) + w * float(self._v_min_near)

        # ---- Extra stability: reduce v when turning hard ----
        omega_abs = abs(float(omega_cmd))
        if omega_abs > float(self._turn_v_omega1):
            v_cmd = min(float(v_cmd), float(self._turn_v_cap1))
        if omega_abs > float(self._turn_v_omega2):
            v_cmd = min(float(v_cmd), float(self._turn_v_cap2))

        # True emergency stop (very close) - keep small to avoid chatter
        emergency = float(self.p.obs_stop_dist) * 0.85
        if d_front < emergency:
            v_cmd = 0.0

        # final clamps
        omega_cmd = clamp(float(omega_cmd), -float(self.p.omega_abs_max), float(self.p.omega_abs_max))
        v_cmd = clamp(float(v_cmd), 0.0, float(self.p.v_max))

        self._publish(v_cmd, omega_cmd)


def main() -> None:
    rclpy.init()
    node = PNNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._publish(0.0, 0.0)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()