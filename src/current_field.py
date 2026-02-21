from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass
class CurrentField:
    mode: str = "vortex"     # vortex | shear | gust
    strength: float = 0.20   # overall scale (m/s-ish)
    omega: float = 0.35
    cx0: float = 0.0
    cy0: float = 0.0
    core: float = 1.0

    def vel(self, x: float, y: float, t: float) -> tuple[float, float]:
        s = float(self.strength)

        if self.mode == "shear":
            u = s * (0.6 * y) * (1.0 + 0.25 * math.sin(self.omega * t))
            v = s * (0.15 * math.sin(0.7 * x + 0.5 * math.cos(self.omega * t)))
            return u, v

        if self.mode == "gust":
            u = s * (0.8 + 0.35 * math.sin(self.omega * t + 0.7 * x))
            v = s * (0.2 * math.cos(self.omega * t + 0.9 * y))
            return u, v

        dx = x - self.cx0
        dy = y - self.cy0
        r2 = dx * dx + dy * dy
        denom = (r2 + self.core * self.core)

        k = s * (1.0 + 0.25 * math.sin(self.omega * t))
        u = -k * dy / denom
        v =  k * dx / denom
        return u, v
