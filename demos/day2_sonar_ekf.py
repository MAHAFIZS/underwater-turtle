# day2_sonar_ekf.py
# Day 2: Add sonar-like sensing (range+bearing + FOV + dropout) and EKF fish tracking.
#
# Run:
#   py demos\day2_sonar_ekf.py
#
# Deps:
#   pip install numpy matplotlib

from __future__ import annotations
import math
import random
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class World:
    xmin: float = -5.0
    xmax: float = 5.0
    ymin: float = -5.0
    ymax: float = 5.0


@dataclass
class Current:
    cx: float = 0.15
    cy: float = 0.05


@dataclass
class Turtle:
    x: float = -4.0
    y: float = -4.0
    psi: float = 0.0
    v: float = 0.35
    v_max: float = 0.75
    omega_max: float = 0.75

    # internal states for inertia
    _omega: float = 0.0

    def step(self, dt: float, omega_cmd: float, v_cmd: float, cur: Current) -> None:
        omega_cmd = clamp(omega_cmd, -self.omega_max, self.omega_max)

        # turning inertia
        omega_alpha = 3.0
        self._omega += (omega_cmd - self._omega) * (1 - math.exp(-omega_alpha * dt))
        self.psi = wrap_angle(self.psi + self._omega * dt)

        # speed inertia
        v_cmd = clamp(v_cmd, 0.0, self.v_max)
        alpha_v = 1.3
        self.v += (v_cmd - self.v) * (1 - math.exp(-alpha_v * dt))

        # kinematics + current drift
        self.x += (self.v * math.cos(self.psi) + cur.cx) * dt
        self.y += (self.v * math.sin(self.psi) + cur.cy) * dt


@dataclass
class Fish:
    x: float = 2.0
    y: float = 2.0
    heading: float = math.pi
    speed: float = 0.6
    v_nom: float = 0.6
    v_burst: float = 1.2
    turn_rate_max: float = 1.2
    wander_turn_std: float = 0.35
    burst_prob_per_s: float = 0.06
    burst_time_left: float = 0.0

    def step(self, dt: float, world: World, cur: Current) -> None:
        # burst logic
        if self.burst_time_left <= 0.0:
            if random.random() < self.burst_prob_per_s * dt:
                self.burst_time_left = random.uniform(0.8, 1.8)
        else:
            self.burst_time_left -= dt

        target_speed = self.v_burst if self.burst_time_left > 0.0 else self.v_nom
        beta = 2.0
        self.speed += (target_speed - self.speed) * (1 - math.exp(-beta * dt))

        # wander
        d_heading = random.gauss(0.0, self.wander_turn_std) * dt
        d_heading = clamp(d_heading, -self.turn_rate_max * dt, self.turn_rate_max * dt)
        self.heading = wrap_angle(self.heading + d_heading)

        # integrate with current
        self.x += (self.speed * math.cos(self.heading) + cur.cx) * dt
        self.y += (self.speed * math.sin(self.heading) + cur.cy) * dt

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


# -------------------------
# Sonar measurement model
# -------------------------
@dataclass
class Sonar:
    sigma_r: float = 0.06          # meters
    sigma_b: float = math.radians(3.0)  # radians
    fov: float = math.radians(70.0)     # +/- 70 deg
    dropout: float = 0.10               # probability per measurement

    def measure(self, turtle: Turtle, fish: Fish) -> tuple[bool, np.ndarray]:
        dx = fish.x - turtle.x
        dy = fish.y - turtle.y
        r_true = math.hypot(dx, dy)
        b_true = wrap_angle(math.atan2(dy, dx) - turtle.psi)

        # FOV gate
        if abs(b_true) > self.fov:
            return False, np.zeros(2)

        # dropout
        if random.random() < self.dropout:
            return False, np.zeros(2)

        r = r_true + random.gauss(0.0, self.sigma_r)
        b = wrap_angle(b_true + random.gauss(0.0, self.sigma_b))
        return True, np.array([r, b], dtype=float)


# -------------------------
# EKF for fish tracking
# state x = [xf, yf, vxf, vyf]
# -------------------------
class FishEKF:
    def __init__(self, x0: np.ndarray, P0: np.ndarray):
        self.x = x0.astype(float).copy()
        self.P = P0.astype(float).copy()

    def predict(self, dt: float, q_pos: float = 0.15, q_vel: float = 0.6) -> None:
        # Constant velocity model
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=float)

        self.x = F @ self.x

        # Process noise (simple diagonal)
        Q = np.diag([q_pos*dt*dt, q_pos*dt*dt, q_vel*dt, q_vel*dt])
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray, turtle: Turtle, R: np.ndarray) -> None:
        # Measurement function h(x): range + bearing from turtle pose
        xf, yf, vxf, vyf = self.x
        dx = xf - turtle.x
        dy = yf - turtle.y
        r = math.hypot(dx, dy)
        r = max(r, 1e-6)

        b = wrap_angle(math.atan2(dy, dx) - turtle.psi)
        h = np.array([r, b], dtype=float)

        # Jacobian H = dh/dx
        dr_dxf = dx / r
        dr_dyf = dy / r
        db_dxf = -dy / (r*r)
        db_dyf =  dx / (r*r)

        H = np.array([
            [dr_dxf, dr_dyf, 0, 0],
            [db_dxf, db_dyf, 0, 0],
        ], dtype=float)

        # Innovation
        y = z - h
        y[1] = wrap_angle(float(y[1]))

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P


def chase_controller_to_point(turtle: Turtle, target_xy: tuple[float, float]) -> tuple[float, float]:
    tx, ty = target_xy
    dx = tx - turtle.x
    dy = ty - turtle.y

    desired = math.atan2(dy, dx)
    err = wrap_angle(desired - turtle.psi)

    k_psi = 1.1
    omega_cmd = k_psi * err

    dist = math.hypot(dx, dy)
    v_cmd = clamp(0.25 + 0.08 * dist, 0.20, turtle.v_max)
    return omega_cmd, v_cmd


def main() -> None:
    random.seed(7)
    np.random.seed(7)

    world = World()
    cur = Current(cx=0.15, cy=0.05)

    turtle = Turtle()
    fish = Fish()
    sonar = Sonar()

    dt = 0.05
    trail_len = 220

    # EKF init: start near fish with big uncertainty
    x0 = np.array([fish.x + 0.5, fish.y - 0.4, 0.0, 0.0])
    P0 = np.diag([1.5, 1.5, 1.0, 1.0])
    ekf = FishEKF(x0, P0)

    R = np.diag([sonar.sigma_r**2, sonar.sigma_b**2])

    turtle_hist, fish_hist, est_hist = [], [], []

    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(world.xmin, world.xmax)
    ax.set_ylim(world.ymin, world.ymax)
    ax.set_title("Day 2: Turtle vs Fish with Sonar + EKF (2D)")
    ax.grid(True, alpha=0.3)

    # current arrow
    cur_scale = 3.0
    ax.arrow(world.xmin + 0.7, world.ymax - 0.7, cur.cx*cur_scale, cur.cy*cur_scale,
             head_width=0.15, length_includes_head=True)
    ax.text(world.xmin + 0.6, world.ymax - 0.4, "current", fontsize=10)

    # artists
    turtle_pt, = ax.plot([], [], marker="o", markersize=10, linestyle="None")
    fish_pt, = ax.plot([], [], marker="x", markersize=10, linestyle="None")
    est_pt, = ax.plot([], [], marker="s", markersize=7, linestyle="None")  # EKF estimate

    turtle_trail, = ax.plot([], [], linewidth=1.5, alpha=0.8)
    fish_trail, = ax.plot([], [], linewidth=1.0, alpha=0.6)
    est_trail, = ax.plot([], [], linewidth=1.0, alpha=0.6)

    heading_line, = ax.plot([], [], linewidth=2.0, alpha=0.9)
    meas_ray, = ax.plot([], [], linewidth=1.0, alpha=0.7)  # sonar ray when measurement exists

    info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=10)

    def init():
        for artist in [turtle_pt, fish_pt, est_pt, turtle_trail, fish_trail, est_trail, heading_line, meas_ray]:
            artist.set_data([], [])
        info.set_text("")
        return turtle_pt, fish_pt, est_pt, turtle_trail, fish_trail, est_trail, heading_line, meas_ray, info

    def update(_frame):
        # Step fish (truth)
        fish.step(dt, world, cur)

        # Sonar measurement (range+bearing) - may be missing
        got_meas, z = sonar.measure(turtle, fish)

        # EKF predict/update
        ekf.predict(dt)
        if got_meas:
            ekf.update(z, turtle, R)

        xf_hat, yf_hat = float(ekf.x[0]), float(ekf.x[1])

        # Control turtle towards estimated fish position (not truth)
        omega_cmd, v_cmd = chase_controller_to_point(turtle, (xf_hat, yf_hat))
        turtle.step(dt, omega_cmd, v_cmd, cur)
        turtle.x = clamp(turtle.x, world.xmin, world.xmax)
        turtle.y = clamp(turtle.y, world.ymin, world.ymax)

        # trails
        turtle_hist.append((turtle.x, turtle.y))
        fish_hist.append((fish.x, fish.y))
        est_hist.append((xf_hat, yf_hat))
        if len(turtle_hist) > trail_len:
            turtle_hist.pop(0); fish_hist.pop(0); est_hist.pop(0)

        # update markers
        turtle_pt.set_data([turtle.x], [turtle.y])
        fish_pt.set_data([fish.x], [fish.y])
        est_pt.set_data([xf_hat], [yf_hat])

        tx, ty = zip(*turtle_hist)
        fx, fy = zip(*fish_hist)
        ex, ey = zip(*est_hist)
        turtle_trail.set_data(tx, ty)
        fish_trail.set_data(fx, fy)
        est_trail.set_data(ex, ey)

        # heading line
        L = 0.6
        heading_line.set_data(
            [turtle.x, turtle.x + L * math.cos(turtle.psi)],
            [turtle.y, turtle.y + L * math.sin(turtle.psi)],
        )

        # measurement ray visualization
        if got_meas:
            r, b = float(z[0]), float(z[1])
            ang_world = wrap_angle(turtle.psi + b)
            mx = turtle.x + r * math.cos(ang_world)
            my = turtle.y + r * math.sin(ang_world)
            meas_ray.set_data([turtle.x, mx], [turtle.y, my])
        else:
            meas_ray.set_data([], [])

        dist_true = math.hypot(fish.x - turtle.x, fish.y - turtle.y)
        dist_est = math.hypot(xf_hat - turtle.x, yf_hat - turtle.y)

        info.set_text(
            f"sonar: {'OK' if got_meas else 'MISS'}\n"
            f"dist true: {dist_true:.2f} m\n"
            f"dist est : {dist_est:.2f} m\n"
            f"turtle v  : {turtle.v:.2f} m/s"
        )

        return turtle_pt, fish_pt, est_pt, turtle_trail, fish_trail, est_trail, heading_line, meas_ray, info

    ani = FuncAnimation(fig, update, init_func=init, interval=int(dt * 1000), blit=True)
    plt.show()


if __name__ == "__main__":
    main()
