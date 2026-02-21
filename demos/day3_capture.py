# day3_capture.py
# Day 3: Sonar + EKF + turtle interception + CAPTURE logic (medium difficulty).
#
# Run:
#   py demos\day3_capture.py
#
# Controls:
#   ]  increase current magnitude
#   [  decrease current magnitude
#   r  reset episode
#   q  quit
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
    cx: float = 0.18
    cy: float = 0.06

    def set_mag(self, mag: float) -> None:
        # keep direction constant, scale magnitude
        base = math.hypot(self.cx, self.cy)
        if base < 1e-9:
            self.cx, self.cy = mag, 0.0
            return
        s = mag / base
        self.cx *= s
        self.cy *= s

    def mag(self) -> float:
        return math.hypot(self.cx, self.cy)


@dataclass
class Turtle:
    x: float = -4.0
    y: float = -4.0
    psi: float = 0.0
    v: float = 0.35
    v_max: float = 0.78
    omega_max: float = 0.75
    _omega: float = 0.0

    def step(self, dt: float, omega_cmd: float, v_cmd: float, cur: Current) -> None:
        omega_cmd = clamp(omega_cmd, -self.omega_max, self.omega_max)

        # turning inertia
        omega_alpha = 3.0
        self._omega += (omega_cmd - self._omega) * (1 - math.exp(-omega_alpha * dt))
        self.psi = wrap_angle(self.psi + self._omega * dt)

        # speed inertia
        v_cmd = clamp(v_cmd, 0.0, self.v_max)
        alpha_v = 1.25
        self.v += (v_cmd - self.v) * (1 - math.exp(-alpha_v * dt))

        # kinematics + current drift
        self.x += (self.v * math.cos(self.psi) + cur.cx) * dt
        self.y += (self.v * math.sin(self.psi) + cur.cy) * dt


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
    burst_prob_per_s: float = 0.07  # medium
    burst_time_left: float = 0.0
    alive: bool = True

    def step(self, dt: float, world: World, cur: Current) -> None:
        if not self.alive:
            return

        # burst logic
        if self.burst_time_left <= 0.0:
            if random.random() < self.burst_prob_per_s * dt:
                self.burst_time_left = random.uniform(0.9, 1.9)
        else:
            self.burst_time_left -= dt

        target_speed = self.v_burst if self.burst_time_left > 0.0 else self.v_nom
        beta = 2.0
        self.speed += (target_speed - self.speed) * (1 - math.exp(-beta * dt))

        # wander
        d_heading = random.gauss(0.0, self.wander_turn_std) * dt
        d_heading = clamp(d_heading, -self.turn_rate_max * dt, self.turn_rate_max * dt)
        self.heading = wrap_angle(self.heading + d_heading)

        # integrate
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


@dataclass
class Sonar:
    sigma_r: float = 0.07
    sigma_b: float = math.radians(3.5)
    fov: float = math.radians(70.0)
    dropout: float = 0.12

    def measure(self, turtle: Turtle, fish: Fish) -> tuple[bool, np.ndarray]:
        if not fish.alive:
            return False, np.zeros(2)

        dx = fish.x - turtle.x
        dy = fish.y - turtle.y
        r_true = math.hypot(dx, dy)
        b_true = wrap_angle(math.atan2(dy, dx) - turtle.psi)

        if abs(b_true) > self.fov:
            return False, np.zeros(2)
        if random.random() < self.dropout:
            return False, np.zeros(2)

        r = r_true + random.gauss(0.0, self.sigma_r)
        b = wrap_angle(b_true + random.gauss(0.0, self.sigma_b))
        return True, np.array([r, b], dtype=float)


class FishEKF:
    def __init__(self, x0: np.ndarray, P0: np.ndarray):
        self.x = x0.astype(float).copy()
        self.P = P0.astype(float).copy()

    def predict(self, dt: float, q_pos: float = 0.16, q_vel: float = 0.8) -> None:
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=float)

        self.x = F @ self.x
        Q = np.diag([q_pos*dt*dt, q_pos*dt*dt, q_vel*dt, q_vel*dt])
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray, turtle: Turtle, R: np.ndarray) -> None:
        xf, yf, vxf, vyf = self.x
        dx = xf - turtle.x
        dy = yf - turtle.y
        r = math.hypot(dx, dy)
        r = max(r, 1e-6)

        b = wrap_angle(math.atan2(dy, dx) - turtle.psi)
        h = np.array([r, b], dtype=float)

        dr_dxf = dx / r
        dr_dyf = dy / r
        db_dxf = -dy / (r*r)
        db_dyf =  dx / (r*r)

        H = np.array([
            [dr_dxf, dr_dyf, 0, 0],
            [db_dxf, db_dyf, 0, 0],
        ], dtype=float)

        y = z - h
        y[1] = wrap_angle(float(y[1]))

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P


def chase_controller(turtle: Turtle, target_xy: tuple[float, float]) -> tuple[float, float]:
    tx, ty = target_xy
    dx = tx - turtle.x
    dy = ty - turtle.y
    desired = math.atan2(dy, dx)
    err = wrap_angle(desired - turtle.psi)

    k_psi = 1.05
    omega_cmd = k_psi * err

    dist = math.hypot(dx, dy)
    v_cmd = clamp(0.20 + 0.07 * dist, 0.20, turtle.v_max)

    return omega_cmd, v_cmd


def make_episode(cur_mag: float) -> tuple[World, Current, Turtle, Fish, Sonar, FishEKF, np.ndarray]:
    world = World()
    cur = Current(cx=0.18, cy=0.06)
    cur.set_mag(cur_mag)

    turtle = Turtle()
    fish = Fish()

    sonar = Sonar()
    R = np.diag([sonar.sigma_r**2, sonar.sigma_b**2])

    x0 = np.array([fish.x + 0.7, fish.y - 0.6, 0.0, 0.0])
    P0 = np.diag([2.2, 2.2, 1.2, 1.2])
    ekf = FishEKF(x0, P0)

    return world, cur, turtle, fish, sonar, ekf, R


def main() -> None:
    random.seed(7)
    np.random.seed(7)

    dt = 0.05
    trail_len = 250

    # Capture settings
    r_capture = 0.30
    t_hold = 2.0
    hold_timer = 0.0
    captured = False
    elapsed = 0.0
    episode_timeout = 45.0  # seconds

    cur_mag = 0.19  # medium default
    world, cur, turtle, fish, sonar, ekf, R = make_episode(cur_mag)

    turtle_hist, fish_hist, est_hist = [], [], []

    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(world.xmin, world.xmax)
    ax.set_ylim(world.ymin, world.ymax)
    ax.grid(True, alpha=0.3)

    title = ax.set_title("")
    info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=10)

    turtle_pt, = ax.plot([], [], marker="o", markersize=10, linestyle="None")
    fish_pt, = ax.plot([], [], marker="x", markersize=10, linestyle="None")
    est_pt, = ax.plot([], [], marker="s", markersize=7, linestyle="None")

    turtle_trail, = ax.plot([], [], linewidth=1.5, alpha=0.8)
    fish_trail, = ax.plot([], [], linewidth=1.0, alpha=0.6)
    est_trail, = ax.plot([], [], linewidth=1.0, alpha=0.6)

    heading_line, = ax.plot([], [], linewidth=2.0, alpha=0.9)
    meas_ray, = ax.plot([], [], linewidth=1.0, alpha=0.7)

    capture_circle = plt.Circle((0, 0), r_capture, fill=False, linewidth=1.5, alpha=0.8)
    ax.add_patch(capture_circle)

    def reset_episode():
        nonlocal world, cur, turtle, fish, sonar, ekf, R
        nonlocal turtle_hist, fish_hist, est_hist
        nonlocal hold_timer, captured, elapsed

        world, cur, turtle, fish, sonar, ekf, R = make_episode(cur_mag)
        turtle_hist, fish_hist, est_hist = [], [], []
        hold_timer = 0.0
        captured = False
        elapsed = 0.0

    def on_key(event):
        nonlocal cur_mag
        if event.key == "]":
            cur_mag = clamp(cur_mag + 0.05, 0.0, 0.55)
            reset_episode()
        elif event.key == "[":
            cur_mag = clamp(cur_mag - 0.05, 0.0, 0.55)
            reset_episode()
        elif event.key in ["r", "R"]:
            reset_episode()
        elif event.key in ["q", "Q"]:
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    def init():
        for artist in [turtle_pt, fish_pt, est_pt, turtle_trail, fish_trail, est_trail, heading_line, meas_ray]:
            artist.set_data([], [])
        info.set_text("")
        return turtle_pt, fish_pt, est_pt, turtle_trail, fish_trail, est_trail, heading_line, meas_ray, info

    def update(_frame):
        nonlocal hold_timer, captured, elapsed

        elapsed += dt

        # Title
        title.set_text(f"Day 3: Capture with Sonar+EKF | current={cur.mag():.2f} m/s  ([ / ] to change)")

        if not captured and elapsed < episode_timeout:
            # Step fish
            fish.step(dt, world, cur)

            # sonar measurement
            got_meas, z = sonar.measure(turtle, fish)

            # EKF
            ekf.predict(dt)
            if got_meas:
                ekf.update(z, turtle, R)

            xf_hat, yf_hat = float(ekf.x[0]), float(ekf.x[1])

            # control to estimate
            omega_cmd, v_cmd = chase_controller(turtle, (xf_hat, yf_hat))
            turtle.step(dt, omega_cmd, v_cmd, cur)

            turtle.x = clamp(turtle.x, world.xmin, world.xmax)
            turtle.y = clamp(turtle.y, world.ymin, world.ymax)

            # Capture check (use TRUE distance for capture)
            dist_true = math.hypot(fish.x - turtle.x, fish.y - turtle.y)
            if dist_true < r_capture:
                hold_timer += dt
            else:
                hold_timer = max(0.0, hold_timer - 2.0 * dt)

            if hold_timer >= t_hold:
                captured = True
                fish.alive = False  # freeze fish

        # trails
        turtle_hist.append((turtle.x, turtle.y))
        fish_hist.append((fish.x, fish.y))
        est_hist.append((float(ekf.x[0]), float(ekf.x[1])))

        if len(turtle_hist) > trail_len:
            turtle_hist.pop(0); fish_hist.pop(0); est_hist.pop(0)

        # artists update
        turtle_pt.set_data([turtle.x], [turtle.y])
        fish_pt.set_data([fish.x], [fish.y])
        est_pt.set_data([float(ekf.x[0])], [float(ekf.x[1])])

        tx, ty = zip(*turtle_hist)
        fx, fy = zip(*fish_hist)
        ex, ey = zip(*est_hist)

        turtle_trail.set_data(tx, ty)
        fish_trail.set_data(fx, fy)
        est_trail.set_data(ex, ey)

        # heading
        L = 0.6
        heading_line.set_data(
            [turtle.x, turtle.x + L * math.cos(turtle.psi)],
            [turtle.y, turtle.y + L * math.sin(turtle.psi)],
        )

        # capture circle centered at turtle
        capture_circle.center = (turtle.x, turtle.y)

        # sonar ray (visual only if fish alive)
        if fish.alive:
            got_meas, z = sonar.measure(turtle, fish)
            if got_meas:
                r, b = float(z[0]), float(z[1])
                ang_world = wrap_angle(turtle.psi + b)
                mx = turtle.x + r * math.cos(ang_world)
                my = turtle.y + r * math.sin(ang_world)
                meas_ray.set_data([turtle.x, mx], [turtle.y, my])
            else:
                meas_ray.set_data([], [])
        else:
            meas_ray.set_data([], [])

        # info text
        dist_true = math.hypot(fish.x - turtle.x, fish.y - turtle.y)
        status = "CAPTURED ✅" if captured else ("TIMEOUT ❌" if elapsed >= episode_timeout else "TRACKING...")
        info.set_text(
            f"status: {status}\n"
            f"dist true: {dist_true:.2f} m\n"
            f"hold: {hold_timer:.2f}/{t_hold:.2f} s\n"
            f"time: {elapsed:.1f}/{episode_timeout:.0f} s\n"
            f"keys: [ ] current, r reset, q quit"
        )

        return turtle_pt, fish_pt, est_pt, turtle_trail, fish_trail, est_trail, heading_line, meas_ray, info

    ani = FuncAnimation(fig, update, init_func=init, interval=int(dt * 1000), blit=True)
    plt.show()


if __name__ == "__main__":
    main()
