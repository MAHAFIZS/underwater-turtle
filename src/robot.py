from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass

def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

@dataclass
class TurtleState:
    x: float
    y: float
    psi: float
    v: float
    omega: float = 0.0  # internal turn-rate state

@dataclass
class TurtleParams:
    v_max: float = 0.78
    omega_max: float = 0.75
    omega_alpha: float = 3.0
    v_alpha: float = 1.25

class TurtleRobot:
    """
    Robot wrapper:
    - state
    - estimator (ekf)
    - sensor (sonar)
    - controller (guidance law)
    - dynamics (apply_control)
    """
    def __init__(self, state: TurtleState, params: TurtleParams, sonar, ekf, R: np.ndarray):
        self.state = state
        self.params = params
        self.sonar = sonar
        self.ekf = ekf
        self.R = R

        # debug outputs (optional)
        self.last_meas_ok = False
        self.last_z = None

    def sense(self, fish) -> tuple[bool, np.ndarray]:
        ok, z = self.sonar.measure(self._as_turtle_view(), fish)
        self.last_meas_ok = ok
        self.last_z = z if ok else None
        return ok, z

    def estimate(self, dt: float, meas_ok: bool, z: np.ndarray):
        self.ekf.predict(dt)
        if meas_ok:
            self.ekf.update(z, self._as_turtle_view(), self.R)

    def control(self) -> tuple[float, float]:
        # chase estimated fish position
        tx, ty = float(self.ekf.x[0]), float(self.ekf.x[1])
        dx = tx - self.state.x
        dy = ty - self.state.y

        desired = math.atan2(dy, dx)
        err = wrap_angle(desired - self.state.psi)

        k_psi = 1.05
        omega_cmd = k_psi * err

        dist = math.hypot(dx, dy)
        v_cmd = clamp(0.20 + 0.07 * dist, 0.20, self.params.v_max)
        return omega_cmd, v_cmd

    def apply_control(self, dt: float, omega_cmd: float, v_cmd: float, current) -> None:
        omega_cmd = clamp(omega_cmd, -self.params.omega_max, self.params.omega_max)

        # turning inertia
        self.state.omega += (omega_cmd - self.state.omega) * (1 - math.exp(-self.params.omega_alpha * dt))
        self.state.psi = wrap_angle(self.state.psi + self.state.omega * dt)

        # speed inertia
        v_cmd = clamp(v_cmd, 0.0, self.params.v_max)
        self.state.v += (v_cmd - self.state.v) * (1 - math.exp(-self.params.v_alpha * dt))

        # kinematics + current
        self.state.x += (self.state.v * math.cos(self.state.psi) + current.cx) * dt
        self.state.y += (self.state.v * math.sin(self.state.psi) + current.cy) * dt

    def step(self, dt: float, fish, current) -> None:
        ok, z = self.sense(fish)
        self.estimate(dt, ok, z)
        omega_cmd, v_cmd = self.control()
        self.apply_control(dt, omega_cmd, v_cmd, current)

    # helper: adapt to your sonar/ekf expecting a Turtle-like object
    def _as_turtle_view(self):
        class _T:
            pass
        t = _T()
        t.x = self.state.x
        t.y = self.state.y
        t.psi = self.state.psi
        return t
