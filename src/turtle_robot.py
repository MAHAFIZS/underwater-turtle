# src/turtle_robot.py
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class TurtleState:
    x: float = -4.0
    y: float = -4.0
    psi: float = 0.0
    v: float = 0.35
    omega: float = 0.0  # internal (smoothed) turn rate

    # For PN:
    los_prev: float = 0.0
    los_rate_filt: float = 0.0

    # For command rate limiting:
    omega_cmd_prev: float = 0.0


@dataclass
class TurtleParams:
    v_max: float = 0.78
    omega_max: float = 0.75
    omega_alpha: float = 3.0
    v_alpha: float = 1.25

    # limit how fast commanded omega is allowed to change (rad/s^2)
    omega_cmd_rate_max: float = 2.0

    # Current compensation: 0=perfect (no drift), 1=none (full drift)
    current_comp: float = 1.0

    # Controller
    controller: str = "pure"  # "pure" or "pn"

    # PN gains
    pn_N: float = 4.5
    pn_k_los: float = 3.0

    # Filter for LOS rate (0..1), higher = more smoothing
    # Uses EMA: filt = a*filt + (1-a)*new
    pn_los_rate_alpha: float = 0.6

    # gating strength for PN LOS-rate term based on closing speed.
    # gate = clamp(Vc / (v_gate + eps), 0..1)
    pn_vc_gate: float = 0.05  # m/s


class TurtleRobot:
    """
    Robot wrapper:
      step(dt, fish, current):
        1) sonar measurement (may be missing)
        2) EKF predict/update
        3) controller computes (omega_cmd, v_cmd) toward estimated fish position
        4) dynamics integrate turtle state + (possibly compensated) current
    """

    def __init__(self, state: TurtleState, params: TurtleParams, sonar, ekf, R: np.ndarray):
        self.state = state
        self.params = params
        self.sonar = sonar
        self.ekf = ekf
        self.R = R

        self.last_meas_ok: bool = False
        self.last_z: np.ndarray | None = None

    # --- internal adapter: sonar/ekf expect a turtle-like object with x,y,psi
    def _turtle_view(self):
        class _T:
            pass

        t = _T()
        t.x = self.state.x
        t.y = self.state.y
        t.psi = self.state.psi
        return t

    def _current_xy(self, current) -> tuple[float, float]:
        """
        Backward-compatible current access.

        Supports:
          - objects with .cx/.cy
          - objects with .cx0/.cy0 (older day5 CurrentField)
          - dict-like {"cx":..., "cy":...} or {"cx0":..., "cy0":...}
          - None
        """
        if current is None:
            return 0.0, 0.0

        # dict-like
        if isinstance(current, dict):
            if "cx" in current and "cy" in current:
                return float(current["cx"]), float(current["cy"])
            if "cx0" in current and "cy0" in current:
                return float(current["cx0"]), float(current["cy0"])
            return 0.0, 0.0

        # attribute-like
        if hasattr(current, "cx") and hasattr(current, "cy"):
            return float(current.cx), float(current.cy)
        if hasattr(current, "cx0") and hasattr(current, "cy0"):
            return float(current.cx0), float(current.cy0)

        return 0.0, 0.0

    def sense(self, fish) -> tuple[bool, np.ndarray]:
        ok, z = self.sonar.measure(self._turtle_view(), fish)
        self.last_meas_ok = ok
        self.last_z = z.copy() if ok else None
        return ok, z

    def estimate(self, dt: float, ok: bool, z: np.ndarray) -> None:
        self.ekf.predict(dt)
        if ok:
            self.ekf.update(z, self._turtle_view(), self.R)

    def _control_pure(self) -> tuple[float, float]:
        tx = float(self.ekf.x[0])
        ty = float(self.ekf.x[1])
        dx = tx - self.state.x
        dy = ty - self.state.y

        desired = math.atan2(dy, dx)
        err = wrap_angle(desired - self.state.psi)

        k_psi = 1.05
        omega_cmd = k_psi * err

        dist = math.hypot(dx, dy)
        v_cmd = clamp(0.20 + 0.07 * dist, 0.20, self.params.v_max)
        return omega_cmd, v_cmd

    def _closing_speed(self, dx: float, dy: float, current=None) -> float:
        """
        Estimate closing speed Vc along the line of sight.
        Positive Vc means we are closing in (distance decreasing).
        Uses EKF fish velocity estimate and turtle approximate ground velocity.
        """
        r = math.hypot(dx, dy)
        if r < 1e-6:
            return 0.0

        rhat_x = dx / r
        rhat_y = dy / r

        # Fish velocity estimate from EKF state [x, y, vx, vy]
        vxf = float(self.ekf.x[2]) if len(self.ekf.x) >= 4 else 0.0
        vyf = float(self.ekf.x[3]) if len(self.ekf.x) >= 4 else 0.0

        # Turtle body velocity in world frame (relative-to-water)
        vtx = self.state.v * math.cos(self.state.psi)
        vty = self.state.v * math.sin(self.state.psi)

        # Add remaining drift (same comp used in apply()) to approximate ground velocity
        curx, cury = self._current_xy(current)
        comp = clamp(float(getattr(self.params, "current_comp", 1.0)), 0.0, 1.0)
        vtx += comp * curx
        vty += comp * cury

        # Relative velocity: fish - turtle
        vrx = vxf - vtx
        vry = vyf - vty

        # Range rate: r_dot = rhat · v_rel
        r_dot = rhat_x * vrx + rhat_y * vry

        # Closing speed: Vc = -r_dot
        return float(-r_dot)

    def _control_pn(self, dt: float, current=None) -> tuple[float, float]:
        tx = float(self.ekf.x[0])
        ty = float(self.ekf.x[1])
        dx = tx - self.state.x
        dy = ty - self.state.y

        los = math.atan2(dy, dx)  # line-of-sight angle in world frame
        los_rate = wrap_angle(los - self.state.los_prev) / max(dt, 1e-6)
        self.state.los_prev = los

        # Filter LOS rate
        a = clamp(self.params.pn_los_rate_alpha, 0.0, 0.99)
        self.state.los_rate_filt = a * self.state.los_rate_filt + (1 - a) * los_rate
        los_rate_use = self.state.los_rate_filt

        # Heading error to LOS
        err = wrap_angle(los - self.state.psi)

        # Closing-speed gating
        Vc = self._closing_speed(dx, dy, current=current)
        v_gate = max(float(getattr(self.params, "pn_vc_gate", 0.20)), 1e-6)
        gate = clamp(Vc / v_gate, 0.0, 1.0)

        # PN-style turn command
        omega_cmd = self.params.pn_k_los * err + (self.params.pn_N * gate) * los_rate_use

        dist = math.hypot(dx, dy)
        v_cmd = clamp(0.20 + 0.07 * dist, 0.20, self.params.v_max)
        return omega_cmd, v_cmd

    def control(self, dt: float, current=None) -> tuple[float, float]:
        ctrl = (self.params.controller or "pure").lower()
        if ctrl == "pn":
            return self._control_pn(dt, current=current)
        return self._control_pure()

    def apply(self, dt: float, omega_cmd: float, v_cmd: float, current) -> None:
        # command rate limiting (prevents snap turns)
        rate_max = max(float(getattr(self.params, "omega_cmd_rate_max", 2.0)), 0.0)
        if rate_max > 0.0:
            dmax = rate_max * dt
            omega_cmd = clamp(omega_cmd, self.state.omega_cmd_prev - dmax, self.state.omega_cmd_prev + dmax)
        self.state.omega_cmd_prev = float(omega_cmd)

        # hard limit
        omega_cmd = clamp(omega_cmd, -self.params.omega_max, self.params.omega_max)

        # turning inertia
        self.state.omega += (omega_cmd - self.state.omega) * (1 - math.exp(-self.params.omega_alpha * dt))
        self.state.psi = wrap_angle(self.state.psi + self.state.omega * dt)

        # speed inertia
        v_cmd = clamp(v_cmd, 0.0, self.params.v_max)
        self.state.v += (v_cmd - self.state.v) * (1 - math.exp(-self.params.v_alpha * dt))

        # current compensation
        comp = clamp(float(getattr(self.params, "current_comp", 1.0)), 0.0, 1.0)
        curx, cury = self._current_xy(current)
        cx = comp * curx
        cy = comp * cury

        # integrate + drift
        self.state.x += (self.state.v * math.cos(self.state.psi) + cx) * dt
        self.state.y += (self.state.v * math.sin(self.state.psi) + cy) * dt

    def step(self, dt: float, fish, current, **_ignored) -> None:
        ok, z = self.sense(fish)
        self.estimate(dt, ok, z)
        omega_cmd, v_cmd = self.control(dt, current=current)
        self.apply(dt, omega_cmd, v_cmd, current)