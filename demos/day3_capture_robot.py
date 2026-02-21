# demos/day3_capture_robot.py
# Same as your Day 3, but wrapped into a TurtleRobot class (sense->estimate->control->act),
# and now visualized as a turtle-robot body (oriented triangle).
#
# Run (recommended):
#   py -m demos.day3_capture_robot
#
# If you run as a script, make sure imports work (package init files present):
#   py demos/day3_capture_robot.py

from __future__ import annotations
import math
import random
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

from src.turtle_robot import TurtleRobot, TurtleState, TurtleParams, wrap_angle, clamp


def turtle_triangle(x: float, y: float, psi: float, L: float = 0.65, W: float = 0.40) -> np.ndarray:
    """
    Oriented triangle representing the turtle robot.
    Tip points forward along heading psi.
    Returns (3,2) array of polygon points in world frame.
    """
    pts = np.array(
        [
            [L / 2, 0.0],     # tip
            [-L / 2, W / 2],  # rear-left
            [-L / 2, -W / 2], # rear-right
        ],
        dtype=float,
    )
    c, s = math.cos(psi), math.sin(psi)
    R = np.array([[c, -s], [s, c]], dtype=float)
    return (pts @ R.T) + np.array([x, y], dtype=float)


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

    def step(self, dt: float, world: World, cur: Current) -> None:
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

        self.x += (self.speed * math.cos(self.heading) + cur.cx) * dt
        self.y += (self.speed * math.sin(self.heading) + cur.cy) * dt

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
    def __init__(self, x0: np.ndarray, P0: np.ndarray):
        self.x = x0.astype(float).copy()
        self.P = P0.astype(float).copy()

    def predict(self, dt: float, q_pos: float = 0.16, q_vel: float = 0.8) -> None:
        F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )
        self.x = F @ self.x
        Q = np.diag([q_pos * dt * dt, q_pos * dt * dt, q_vel * dt, q_vel * dt])
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray, turtle_like, R: np.ndarray) -> None:
        xf, yf, vxf, vyf = self.x
        dx = xf - turtle_like.x
        dy = yf - turtle_like.y
        r = math.hypot(dx, dy)
        r = max(r, 1e-6)

        b = wrap_angle(math.atan2(dy, dx) - turtle_like.psi)
        h = np.array([r, b], dtype=float)

        dr_dxf = dx / r
        dr_dyf = dy / r
        db_dxf = -dy / (r * r)
        db_dyf = dx / (r * r)

        H = np.array(
            [
                [dr_dxf, dr_dyf, 0, 0],
                [db_dxf, db_dyf, 0, 0],
            ],
            dtype=float,
        )

        y = z - h
        y[1] = wrap_angle(float(y[1]))

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P


def make_episode(cur_mag: float):
    world = World()
    cur = Current(cx=0.18, cy=0.06)
    cur.set_mag(cur_mag)

    fish = Fish()

    sonar = Sonar()
    R = np.diag([sonar.sigma_r**2, sonar.sigma_b**2])

    # EKF init
    x0 = np.array([fish.x + 0.7, fish.y - 0.6, 0.0, 0.0])
    P0 = np.diag([2.2, 2.2, 1.2, 1.2])
    ekf = FishEKF(x0, P0)

    # Robot wrapper
    state = TurtleState()
    params = TurtleParams()
    robot = TurtleRobot(state=state, params=params, sonar=sonar, ekf=ekf, R=R)

    return world, cur, robot, fish


def main() -> None:
    random.seed(7)
    np.random.seed(7)

    dt = 0.05
    trail_len = 250

    r_capture = 0.30
    t_hold = 2.0
    hold_timer = 0.0
    captured = False
    elapsed = 0.0
    episode_timeout = 45.0

    cur_mag = 0.19
    world, cur, robot, fish = make_episode(cur_mag)

    robot_hist, fish_hist, est_hist = [], [], []

    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(world.xmin, world.xmax)
    ax.set_ylim(world.ymin, world.ymax)
    ax.grid(True, alpha=0.3)

    title = ax.set_title("")
    info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=10)

    # ROBOT BODY (triangle) instead of a dot
    robot_body = Polygon([[0, 0], [0, 0], [0, 0]], closed=True, alpha=0.9)
    ax.add_patch(robot_body)

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
        nonlocal world, cur, robot, fish
        nonlocal robot_hist, fish_hist, est_hist
        nonlocal hold_timer, captured, elapsed

        world, cur, robot, fish = make_episode(cur_mag)
        robot_hist, fish_hist, est_hist = [], [], []
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
        for artist in [fish_pt, est_pt, turtle_trail, fish_trail, est_trail, heading_line, meas_ray]:
            artist.set_data([], [])
        robot_body.set_xy([[0, 0], [0, 0], [0, 0]])
        info.set_text("")
        return robot_body, fish_pt, est_pt, turtle_trail, fish_trail, est_trail, heading_line, meas_ray, info

    def update(_frame):
        nonlocal hold_timer, captured, elapsed

        elapsed += dt
        title.set_text(f"Day 3 (Robot): Capture with Sonar+EKF | current={cur.mag():.2f} m/s  ([ / ] to change)")

        if not captured and elapsed < episode_timeout:
            fish.step(dt, world, cur)

            # robot step (sense->estimate->control->act)
            robot.step(dt, fish, cur)

            # clamp inside bounds
            robot.state.x = clamp(robot.state.x, world.xmin, world.xmax)
            robot.state.y = clamp(robot.state.y, world.ymin, world.ymax)

            # capture check uses TRUE distance
            dist_true = math.hypot(fish.x - robot.state.x, fish.y - robot.state.y)
            if dist_true < r_capture:
                hold_timer += dt
            else:
                hold_timer = max(0.0, hold_timer - 2.0 * dt)

            if hold_timer >= t_hold:
                captured = True
                fish.alive = False

        # trails
        robot_hist.append((robot.state.x, robot.state.y))
        fish_hist.append((fish.x, fish.y))
        est_hist.append((float(robot.ekf.x[0]), float(robot.ekf.x[1])))

        if len(robot_hist) > trail_len:
            robot_hist.pop(0)
            fish_hist.pop(0)
            est_hist.pop(0)

        # Update robot body (triangle)
        robot_body.set_xy(turtle_triangle(robot.state.x, robot.state.y, robot.state.psi))

        # other markers
        fish_pt.set_data([fish.x], [fish.y])
        est_pt.set_data([float(robot.ekf.x[0])], [float(robot.ekf.x[1])])

        tx, ty = zip(*robot_hist)
        fx, fy = zip(*fish_hist)
        ex, ey = zip(*est_hist)

        turtle_trail.set_data(tx, ty)
        fish_trail.set_data(fx, fy)
        est_trail.set_data(ex, ey)

        # heading line (optional, helps readability)
        L = 0.6
        heading_line.set_data(
            [robot.state.x, robot.state.x + L * math.cos(robot.state.psi)],
            [robot.state.y, robot.state.y + L * math.sin(robot.state.psi)],
        )

        capture_circle.center = (robot.state.x, robot.state.y)

        # sonar ray uses robot.last_z if available
        if robot.last_meas_ok and robot.last_z is not None and fish.alive:
            r, b = float(robot.last_z[0]), float(robot.last_z[1])
            ang_world = wrap_angle(robot.state.psi + b)
            mx = robot.state.x + r * math.cos(ang_world)
            my = robot.state.y + r * math.sin(ang_world)
            meas_ray.set_data([robot.state.x, mx], [robot.state.y, my])
        else:
            meas_ray.set_data([], [])

        dist_true = math.hypot(fish.x - robot.state.x, fish.y - robot.state.y)
        status = "CAPTURED ✅" if captured else ("TIMEOUT ❌" if elapsed >= episode_timeout else "TRACKING...")
        info.set_text(
            f"status: {status}\n"
            f"dist true: {dist_true:.2f} m\n"
            f"hold: {hold_timer:.2f}/{t_hold:.2f} s\n"
            f"time: {elapsed:.1f}/{episode_timeout:.0f} s\n"
            f"sonar: {'OK' if robot.last_meas_ok else 'MISS'}\n"
            f"keys: [ ] current, r reset, q quit"
        )

        return robot_body, fish_pt, est_pt, turtle_trail, fish_trail, est_trail, heading_line, meas_ray, info

    ani = FuncAnimation(fig, update, init_func=init, interval=int(dt * 1000), blit=True)
    plt.show()


if __name__ == "__main__":
    main()
