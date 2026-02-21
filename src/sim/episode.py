# src/sim/episode.py
from __future__ import annotations
import math
import random
from dataclasses import dataclass

import numpy as np

from src.turtle_robot import wrap_angle, clamp
from src.current_field import CurrentField


@dataclass
class World:
    xmin: float = -5.0
    xmax: float = 5.0
    ymin: float = -5.0
    ymax: float = 5.0


@dataclass
class Fish:
    x: float = 2.0
    y: float = 2.0
    heading: float = math.pi
    speed: float = 0.62
    v_nom: float = 0.62
    v_burst: float = 1.25
    turn_rate_max: float = 1.25
    wander_turn_std: float = 0.40
    burst_prob_per_s: float = 0.07
    burst_time_left: float = 0.0
    alive: bool = True

    def step(self, dt: float, t: float, world: World, cur: CurrentField) -> None:
        if not self.alive:
            return

        if self.burst_time_left <= 0.0:
            if random.random() < self.burst_prob_per_s * dt:
                self.burst_time_left = random.uniform(0.9, 1.9)
        else:
            self.burst_time_left -= dt

        target_speed = self.v_burst if self.burst_time_left > 0.0 else self.v_nom
        beta = 2.0
        self.speed += (target_speed - self.speed) * (1 - math.exp(-beta * dt))

        d_heading = random.gauss(0.0, self.wander_turn_std) * dt
        d_heading = clamp(d_heading, -self.turn_rate_max * dt, self.turn_rate_max * dt)
        self.heading = wrap_angle(self.heading + d_heading)

        cx, cy = cur.vel(self.x, self.y, t)
        self.x += (self.speed * math.cos(self.heading) + cx) * dt
        self.y += (self.speed * math.sin(self.heading) + cy) * dt

        # bounce
        if self.x < world.xmin:
            self.x = world.xmin
            self.heading = wrap_angle(math.pi - self.heading)
        elif self.x > world.xmax:
            self.x = world.xmax
            self.heading = wrap_angle(math.pi - self.heading)

        if self.y < world.ymin:
            self.y = world.ymin
            self.heading = wrap_angle(-self.heading)
        elif self.y > world.ymax:
            self.y = world.ymax
            self.heading = wrap_angle(-self.heading)


@dataclass
class Sonar:
    sigma_r: float = 0.07
    sigma_b: float = math.radians(3.5)
    fov: float = math.radians(70.0)
    dropout: float = 0.12

    def measure(self, turtle_like, fish: Fish) -> tuple[bool, np.ndarray]:
        if not fish.alive:
            return False, np.zeros(2)

        dx = fish.x - turtle_like.x
        dy = fish.y - turtle_like.y
        r_true = math.hypot(dx, dy)
        b_true = wrap_angle(math.atan2(dy, dx) - turtle_like.psi)

        if abs(b_true) > self.fov:
            return False, np.zeros(2)
        if random.random() < self.dropout:
            return False, np.zeros(2)

        r = r_true + random.gauss(0.0, self.sigma_r)
        b = wrap_angle(b_true + random.gauss(0.0, self.sigma_b))
        return True, np.array([r, b], dtype=float)


class FishEKF:
    """State: [x, y, vx, vy]."""

    def __init__(self, x0: np.ndarray, P0: np.ndarray):
        self.x = x0.astype(float).copy()
        self.P = P0.astype(float).copy()

    def predict(self, dt: float, q_pos: float = 0.16, q_vel: float = 0.8) -> None:
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=float)
        self.x = F @ self.x
        Q = np.diag([q_pos * dt * dt, q_pos * dt * dt, q_vel * dt, q_vel * dt])
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray, turtle_like, R: np.ndarray) -> None:
        xf, yf, vxf, vyf = self.x
        dx = xf - turtle_like.x
        dy = yf - turtle_like.y
        r = max(math.hypot(dx, dy), 1e-6)

        b = wrap_angle(math.atan2(dy, dx) - turtle_like.psi)
        h = np.array([r, b], dtype=float)

        dr_dxf = dx / r
        dr_dyf = dy / r
        db_dxf = -dy / (r * r)
        db_dyf = dx / (r * r)

        H = np.array([[dr_dxf, dr_dyf, 0, 0],
                      [db_dxf, db_dyf, 0, 0]], dtype=float)

        y = z - h
        y[1] = wrap_angle(float(y[1]))

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P


@dataclass
class TurtlePlantParams:
    v_max: float = 0.78
    omega_max: float = 0.75
    omega_alpha: float = 3.0
    v_alpha: float = 1.25
    current_comp: float = 0.70  # 0=perfect cancel, 1=no cancel


@dataclass
class TurtlePlantState:
    x: float = -4.0
    y: float = -4.0
    psi: float = 0.0
    v: float = 0.35
    omega: float = 0.0


class TurtlePlant:
    """Dynamics only. Controller is external (ROS-ready)."""

    def __init__(self, state: TurtlePlantState | None = None, params: TurtlePlantParams | None = None):
        self.s = state or TurtlePlantState()
        self.p = params or TurtlePlantParams()

    def ground_velocity(self, cur: CurrentField, t: float) -> tuple[float, float]:
        # body velocity relative-to-water
        vx = self.s.v * math.cos(self.s.psi)
        vy = self.s.v * math.sin(self.s.psi)

        # remaining drift after compensation
        curx, cury = cur.vel(self.s.x, self.s.y, t)
        comp = clamp(float(self.p.current_comp), 0.0, 1.0)
        vx += comp * curx
        vy += comp * cury
        return vx, vy

    def step(self, dt: float, omega_cmd: float, v_cmd: float, cur: CurrentField, t: float) -> None:
        omega_cmd = clamp(float(omega_cmd), -self.p.omega_max, self.p.omega_max)
        v_cmd = clamp(float(v_cmd), 0.0, self.p.v_max)

        # first-order turn dynamics
        self.s.omega += (omega_cmd - self.s.omega) * (1 - math.exp(-self.p.omega_alpha * dt))
        self.s.psi = wrap_angle(self.s.psi + self.s.omega * dt)

        # first-order speed dynamics
        self.s.v += (v_cmd - self.s.v) * (1 - math.exp(-self.p.v_alpha * dt))

        # integrate with current (post-comp remaining drift)
        curx, cury = cur.vel(self.s.x, self.s.y, t)
        comp = clamp(float(self.p.current_comp), 0.0, 1.0)
        self.s.x += (self.s.v * math.cos(self.s.psi) + comp * curx) * dt
        self.s.y += (self.s.v * math.sin(self.s.psi) + comp * cury) * dt


class TurtleAgent:
    """Plant + sonar + EKF. Controller supplied externally each step."""

    def __init__(self, plant: TurtlePlant, sonar: Sonar, ekf: FishEKF, R: np.ndarray):
        self.plant = plant
        self.sonar = sonar
        self.ekf = ekf
        self.R = R

    def _turtle_view(self):
        class _T:
            pass
        t = _T()
        t.x = self.plant.s.x
        t.y = self.plant.s.y
        t.psi = self.plant.s.psi
        return t

    def step(self, dt: float, t: float, fish: Fish, cur: CurrentField, controller) -> None:
        ok, z = self.sonar.measure(self._turtle_view(), fish)

        self.ekf.predict(dt)
        if ok:
            self.ekf.update(z, self._turtle_view(), self.R)

        tx, ty = float(self.ekf.x[0]), float(self.ekf.x[1])
        fish_vx = float(self.ekf.x[2]) if len(self.ekf.x) >= 4 else 0.0
        fish_vy = float(self.ekf.x[3]) if len(self.ekf.x) >= 4 else 0.0

        turtle_vx, turtle_vy = self.plant.ground_velocity(cur, t)

        omega_cmd, v_cmd = controller.compute(
            tx=tx, ty=ty,
            x=self.plant.s.x, y=self.plant.s.y, psi=self.plant.s.psi,
            v_max=self.plant.p.v_max,
            dt=dt,
            fish_vx=fish_vx, fish_vy=fish_vy,
            turtle_vx_ground=turtle_vx, turtle_vy_ground=turtle_vy,
        )

        self.plant.step(dt, omega_cmd, v_cmd, cur, t)


def make_episode(*, field: str, cur_strength: float, comp: float) -> tuple[World, CurrentField, TurtleAgent, Fish]:
    world = World()
    cur = CurrentField(mode=field, strength=float(cur_strength), cx0=0.0, cy0=0.0, core=1.8)

    fish = Fish()
    sonar = Sonar()
    R = np.diag([sonar.sigma_r**2, sonar.sigma_b**2])

    x0 = np.array([fish.x + 0.7, fish.y - 0.6, 0.0, 0.0])
    P0 = np.diag([2.2, 2.2, 1.2, 1.2])
    ekf = FishEKF(x0, P0)

    plant = TurtlePlant(params=TurtlePlantParams(current_comp=float(comp)))
    agent = TurtleAgent(plant=plant, sonar=sonar, ekf=ekf, R=R)
    return world, cur, agent, fish


def run_one_episode(*, field: str, cur_strength: float, comp: float, controller,
                    seed: int, dt: float, timeout_s: float,
                    r_capture: float, t_hold: float) -> tuple[bool, float]:
    random.seed(seed)
    np.random.seed(seed)

    world, cur, agent, fish = make_episode(field=field, cur_strength=cur_strength, comp=comp)

    # allow controller to reset between episodes (PN needs it)
    if hasattr(controller, "reset"):
        controller.reset()

    hold = 0.0
    t = 0.0
    steps = int(timeout_s / dt)

    for _ in range(steps):
        t += dt

        fish.step(dt, t, world, cur)
        agent.step(dt, t, fish, cur, controller)

        # bounds for turtle
        agent.plant.s.x = clamp(agent.plant.s.x, world.xmin, world.xmax)
        agent.plant.s.y = clamp(agent.plant.s.y, world.ymin, world.ymax)

        d = math.hypot(fish.x - agent.plant.s.x, fish.y - agent.plant.s.y)
        if d < r_capture:
            hold += dt
        else:
            hold = max(0.0, hold - 2.0 * dt)

        if hold >= t_hold:
            return True, t

    return False, timeout_s