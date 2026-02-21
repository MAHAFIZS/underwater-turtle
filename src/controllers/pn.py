# src/controllers/pn.py
from __future__ import annotations
import math
from dataclasses import dataclass

from src.turtle_robot import wrap_angle, clamp


@dataclass
class PNParams:
    pn_N: float = 6.57
    pn_k_los: float = 4.57
    pn_los_rate_alpha: float = 0.6     # EMA smoothing (0..1), higher = more smoothing
    pn_vc_gate: float = 0.05           # closing-speed scale for gating los_rate term
    omega_cmd_rate_max: float = 2.0    # rad/s^2 command slew limit
    v_base: float = 0.20
    v_gain: float = 0.07


class PNController:
    """
    Proportional Navigation controller + vortex stabilizers:
      - LOS-rate EMA filter
      - closing-speed gate on PN term
      - omega_cmd slew-rate limiting (prevents snap turns)

    Keeps its own internal memory (LOS prev, filtered LOS rate, prev omega cmd).
    """

    def __init__(self, params: PNParams | None = None):
        self.p = params or PNParams()
        self._los_prev = 0.0
        self._los_rate_filt = 0.0
        self._omega_cmd_prev = 0.0
        self._initialized = False

    @staticmethod
    def _closing_speed(*, dx: float, dy: float,
                       vxf: float, vyf: float,
                       vtx: float, vty: float) -> float:
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
        self._initialized = False

    def compute(self, *,
                tx: float, ty: float,
                x: float, y: float, psi: float,
                v_max: float,
                dt: float,
                # optional extras for gating
                fish_vx: float = 0.0, fish_vy: float = 0.0,
                turtle_vx_ground: float = 0.0, turtle_vy_ground: float = 0.0,
                **_ignored) -> tuple[float, float]:
        dx = tx - x
        dy = ty - y

        los = math.atan2(dy, dx)

        if not self._initialized:
            self._los_prev = los
            self._initialized = True

        los_rate = wrap_angle(los - self._los_prev) / max(dt, 1e-6)
        self._los_prev = los

        a = clamp(self.p.pn_los_rate_alpha, 0.0, 0.99)
        self._los_rate_filt = a * self._los_rate_filt + (1.0 - a) * los_rate
        los_rate_use = self._los_rate_filt

        err = wrap_angle(los - psi)

        # Closing-speed gating
        Vc = self._closing_speed(dx=dx, dy=dy,
                                 vxf=fish_vx, vyf=fish_vy,
                                 vtx=turtle_vx_ground, vty=turtle_vy_ground)
        v_gate = max(float(self.p.pn_vc_gate), 1e-6)
        gate = clamp(Vc / v_gate, 0.0, 1.0)

        omega_cmd = self.p.pn_k_los * err + (self.p.pn_N * gate) * los_rate_use

        # omega slew-rate limiting
        rate_max = max(float(self.p.omega_cmd_rate_max), 0.0)
        if rate_max > 0.0:
            dmax = rate_max * dt
            omega_cmd = clamp(omega_cmd, self._omega_cmd_prev - dmax, self._omega_cmd_prev + dmax)
        self._omega_cmd_prev = float(omega_cmd)

        dist = math.hypot(dx, dy)
        v_cmd = clamp(self.p.v_base + self.p.v_gain * dist, self.p.v_base, v_max)
        return omega_cmd, v_cmd