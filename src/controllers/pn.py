# underwater_turtle_ros/controllers/pn.py
from __future__ import annotations

import math
from dataclasses import dataclass


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class PNParams:
    # PN core
    pn_N: float = 6.57
    pn_k_los: float = 4.57
    pn_los_rate_alpha: float = 0.6     # EMA smoothing (0..1), higher = more smoothing
    pn_vc_gate: float = 0.05           # closing-speed scale for gating los_rate term

    # Command smoothing / safety
    omega_cmd_rate_max: float = 2.0    # rad/s^2 command slew limit (prevents snap turns)

    # Forward speed shaping (base)
    v_base: float = 0.12               # m/s
    v_gain: float = 0.05               # m/s per meter

    # Hard caps (Gazebo-friendly)
    omega_abs_max: float = 0.8         # rad/s hard cap
    v_abs_max: float = 0.22            # m/s hard cap

    # Slow down while turning / misaligned
    slow_turn_omega: float = 0.5       # rad/s: begin slowing v above this
    slow_turn_err: float = 0.6         # rad: begin slowing v above this (~35 deg)
    v_turn_min: float = 0.03           # m/s crawl speed while turning

    # Linear accel limiting (prevents impulse flips)
    v_cmd_rate_max: float = 0.35       # m/s^2 slew limit for v

    # Near-goal speed caps
    near_goal_1: float = 0.5           # m
    near_goal_v1: float = 0.12         # m/s cap inside near_goal_1
    near_goal_2: float = 0.2           # m
    near_goal_v2: float = 0.06         # m/s cap inside near_goal_2


class PNController:
    """
    Proportional Navigation controller + stabilizers:
      - LOS-rate EMA filter
      - closing-speed gate on PN term
      - omega_cmd slew-rate limiting
      - omega hard cap
      - turn-aware speed coupling (slow down when |omega| or heading error is large)
      - v_cmd slew-rate limiting
      - near-goal speed caps

    Keeps internal memory (LOS prev, filtered LOS rate, prev omega cmd, prev v cmd).
    """

    def __init__(self, params: PNParams | None = None):
        self.p = params or PNParams()
        self._los_prev = 0.0
        self._los_rate_filt = 0.0
        self._omega_cmd_prev = 0.0
        self._v_cmd_prev = 0.0
        self._initialized = False

    @staticmethod
    def _closing_speed(
        *,
        dx: float,
        dy: float,
        vxf: float,
        vyf: float,
        vtx: float,
        vty: float,
    ) -> float:
        """
        Closing speed along line-of-sight.
        Returns Vc = -r_dot (positive means closing).
        """
        r = math.hypot(dx, dy)
        if r < 1e-6:
            return 0.0
        rhat_x = dx / r
        rhat_y = dy / r
        vrx = vxf - vtx
        vry = vyf - vty
        r_dot = rhat_x * vrx + rhat_y * vry
        return float(-r_dot)  # Vc

    def reset(self) -> None:
        self._los_prev = 0.0
        self._los_rate_filt = 0.0
        self._omega_cmd_prev = 0.0
        self._v_cmd_prev = 0.0
        self._initialized = False

    def compute(
        self,
        *,
        tx: float,
        ty: float,
        x: float,
        y: float,
        psi: float,
        v_max: float,
        dt: float,
        # optional extras for gating
        fish_vx: float = 0.0,
        fish_vy: float = 0.0,
        turtle_vx_ground: float = 0.0,
        turtle_vy_ground: float = 0.0,
        **_ignored,
    ) -> tuple[float, float]:
        dx = tx - x
        dy = ty - y

        # Line-of-sight angle
        los = math.atan2(dy, dx)

        # Initialize memory on first call
        if not self._initialized:
            self._los_prev = los
            self._omega_cmd_prev = 0.0
            self._v_cmd_prev = 0.0
            self._initialized = True

        # LOS rate (wrapped) and EMA filter
        los_rate = wrap_angle(los - self._los_prev) / max(dt, 1e-6)
        self._los_prev = los

        a = clamp(self.p.pn_los_rate_alpha, 0.0, 0.99)
        self._los_rate_filt = a * self._los_rate_filt + (1.0 - a) * los_rate
        los_rate_use = self._los_rate_filt

        # Heading error to LOS
        err = wrap_angle(los - psi)

        # Closing-speed gating for PN term
        Vc = self._closing_speed(
            dx=dx,
            dy=dy,
            vxf=fish_vx,
            vyf=fish_vy,
            vtx=turtle_vx_ground,
            vty=turtle_vy_ground,
        )
        v_gate = max(float(self.p.pn_vc_gate), 1e-6)
        gate = clamp(Vc / v_gate, 0.0, 1.0)

        # PN omega command
        omega_cmd = float(self.p.pn_k_los) * float(err) + (float(self.p.pn_N) * float(gate)) * float(los_rate_use)

        # omega slew-rate limiting
        rate_max = max(float(self.p.omega_cmd_rate_max), 0.0)
        if rate_max > 0.0:
            dmax = rate_max * max(dt, 1e-6)
            omega_cmd = clamp(omega_cmd, self._omega_cmd_prev - dmax, self._omega_cmd_prev + dmax)

        # hard cap omega
        omega_cmd = clamp(float(omega_cmd), -float(self.p.omega_abs_max), float(self.p.omega_abs_max))
        self._omega_cmd_prev = float(omega_cmd)

        # Distance to target
        dist = math.hypot(dx, dy)

        # Base forward speed
        v_cap = min(float(v_max), float(self.p.v_abs_max))
        v_nom = float(self.p.v_base) + float(self.p.v_gain) * float(dist)
        v_nom = clamp(v_nom, 0.0, v_cap)

        # Near-goal caps
        if dist < float(self.p.near_goal_1):
            v_cap = min(v_cap, float(self.p.near_goal_v1))
        if dist < float(self.p.near_goal_2):
            v_cap = min(v_cap, float(self.p.near_goal_v2))
        v_nom = clamp(v_nom, 0.0, v_cap)

        # Turn-aware speed coupling
        err_abs = abs(err)
        omega_abs = abs(omega_cmd)

        if float(self.p.slow_turn_err) > 1e-6:
            s_err = clamp(1.0 - (err_abs / float(self.p.slow_turn_err)), 0.0, 1.0)
        else:
            s_err = 1.0

        if float(self.p.slow_turn_omega) > 1e-6:
            s_om = clamp(1.0 - (omega_abs / float(self.p.slow_turn_omega)), 0.0, 1.0)
        else:
            s_om = 1.0

        s = min(s_err, s_om)

        v_cmd = float(self.p.v_turn_min) + (v_nom - float(self.p.v_turn_min)) * s
        v_cmd = clamp(v_cmd, 0.0, v_cap)

        # Linear slew-rate limiting
        amax = max(float(self.p.v_cmd_rate_max), 0.0)
        if amax > 0.0:
            dvmax = amax * max(dt, 1e-6)
            v_cmd = clamp(v_cmd, self._v_cmd_prev - dvmax, self._v_cmd_prev + dvmax)

        self._v_cmd_prev = float(v_cmd)

        return float(omega_cmd), float(v_cmd)