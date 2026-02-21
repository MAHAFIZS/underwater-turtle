# demos/day4_eval.py
# Day 4: Automatic evaluation + performance plots (robustness curve)
#
# Run:
#   py -m demos.day4_eval --episodes 60
#   py -m demos.day4_eval --episodes 60 --comp 0.2
#   py -m demos.day4_eval --episodes 60 --comp 0.5
#   py -m demos.day4_eval --episodes 60 --comp 0.8
#
# Outputs:
#   results/day4/metrics.csv
#   results/day4/capture_rate_vs_current.png
#   results/day4/mean_time_vs_current.png

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.turtle_robot import TurtleRobot, TurtleState, TurtleParams, wrap_angle, clamp


# -------------------------
# Environment + target model
# -------------------------
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

        # fish feels FULL current
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
# Sonar + EKF (same as Day 3/4)
# -------------------------
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


def make_episode(cur_mag: float, comp: float):
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

    state = TurtleState()
    params = TurtleParams()
    params.current_comp = float(comp)  # turtle feels only comp * current
    robot = TurtleRobot(state=state, params=params, sonar=sonar, ekf=ekf, R=R)

    return world, cur, robot, fish


def run_one_episode(
    cur_mag: float,
    comp: float,
    seed: int,
    dt: float = 0.05,
    timeout_s: float = 45.0,
    r_capture: float = 0.30,
    t_hold: float = 2.0,
) -> tuple[bool, float]:
    random.seed(seed)
    np.random.seed(seed)

    world, cur, robot, fish = make_episode(cur_mag, comp)

    hold_timer = 0.0
    t = 0.0

    steps = int(timeout_s / dt)
    for _ in range(steps):
        t += dt

        fish.step(dt, world, cur)
        robot.step(dt, fish, cur)

        robot.state.x = clamp(robot.state.x, world.xmin, world.xmax)
        robot.state.y = clamp(robot.state.y, world.ymin, world.ymax)

        dist_true = math.hypot(fish.x - robot.state.x, fish.y - robot.state.y)

        if dist_true < r_capture:
            hold_timer += dt
        else:
            hold_timer = max(0.0, hold_timer - 2.0 * dt)

        if hold_timer >= t_hold:
            return True, t

    return False, timeout_s


def linspace_list(a: float, b: float, n: int) -> list[float]:
    if n == 1:
        return [float(a)]
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60, help="episodes per current value")
    ap.add_argument("--seed", type=int, default=7, help="base seed (episode seed = seed + idx)")
    ap.add_argument("--curr_min", type=float, default=0.00)
    ap.add_argument("--curr_max", type=float, default=0.50)
    ap.add_argument("--curr_steps", type=int, default=11)
    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--comp", type=float, default=0.5, help="turtle current_comp in [0,1] (0=perfect, 1=none)")
    args = ap.parse_args()

    comp = clamp(float(args.comp), 0.0, 1.0)

    out_dir = Path("results/day4")
    out_dir.mkdir(parents=True, exist_ok=True)

    currents = linspace_list(args.curr_min, args.curr_max, args.curr_steps)

    rows = []
    capture_rates = []
    mean_times = []

    print(f"[day4] episodes per current: {args.episodes}")
    print(f"[day4] current sweep: {currents[0]:.2f} .. {currents[-1]:.2f} ({len(currents)} steps)")
    print(f"[day4] turtle current_comp={comp:.2f} (0=perfect, 1=no compensation)")
    print(f"[day4] writing to: {out_dir.resolve()}")

    for ci, c in enumerate(currents):
        successes = 0
        times_success = []

        for ei in range(args.episodes):
            seed = args.seed + ci * 10_000 + ei
            ok, tcap = run_one_episode(cur_mag=c, comp=comp, seed=seed, timeout_s=args.timeout)
            if ok:
                successes += 1
                times_success.append(tcap)

        rate = successes / args.episodes
        mtime = float(np.mean(times_success)) if times_success else float("nan")

        capture_rates.append(rate)
        mean_times.append(mtime)

        rows.append(
            {
                "current": f"{c:.4f}",
                "episodes": str(args.episodes),
                "successes": str(successes),
                "capture_rate": f"{rate:.4f}",
                "mean_capture_time_s": f"{mtime:.4f}" if times_success else "",
                "turtle_current_comp": f"{comp:.3f}",
            }
        )

        print(f"  current={c:.2f}  capture_rate={rate:.2f}  mean_t={'NA' if not times_success else f'{mtime:.1f}s'}")

    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["current", "episodes", "successes", "capture_rate", "mean_capture_time_s", "turtle_current_comp"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[wrote] {csv_path}")

    fig1 = plt.figure()
    plt.plot(currents, capture_rates, marker="o")
    plt.xlabel("Current magnitude (m/s)")
    plt.ylabel("Capture rate")
    plt.title(f"Capture rate vs current (N={args.episodes}/point, comp={comp:.2f})")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    p1 = out_dir / "capture_rate_vs_current.png"
    fig1.savefig(p1, dpi=200, bbox_inches="tight")
    print(f"[wrote] {p1}")

    fig2 = plt.figure()
    y = np.array(mean_times, dtype=float)
    x = np.array(currents, dtype=float)
    mask = np.isfinite(y)
    if np.any(mask):
        plt.plot(x[mask], y[mask], marker="o")
    plt.xlabel("Current magnitude (m/s)")
    plt.ylabel("Mean capture time (s) [successes only]")
    plt.title(f"Mean capture time vs current (successes only, comp={comp:.2f})")
    plt.grid(True, alpha=0.3)
    p2 = out_dir / "mean_time_vs_current.png"
    fig2.savefig(p2, dpi=200, bbox_inches="tight")
    print(f"[wrote] {p2}")

    plt.show()


if __name__ == "__main__":
    main()
