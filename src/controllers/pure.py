# src/controllers/pure.py
from __future__ import annotations
import math
from dataclasses import dataclass

from src.turtle_robot import wrap_angle, clamp


@dataclass
class PureParams:
    k_psi: float = 1.05
    v_base: float = 0.20
    v_gain: float = 0.07


class PurePursuitController:
    """
    Stateless pure pursuit heading controller.
    Input: turtle pose + estimated fish position
    Output: (omega_cmd, v_cmd)
    """

    def __init__(self, params: PureParams | None = None):
        self.p = params or PureParams()

    def compute(self, *, tx: float, ty: float, x: float, y: float, psi: float,
                v_max: float, dt: float, **_ignored) -> tuple[float, float]:
        dx = tx - x
        dy = ty - y
        desired = math.atan2(dy, dx)
        err = wrap_angle(desired - psi)

        omega_cmd = self.p.k_psi * err

        dist = math.hypot(dx, dy)
        v_cmd = clamp(self.p.v_base + self.p.v_gain * dist, self.p.v_base, v_max)
        return omega_cmd, v_cmd