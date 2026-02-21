# demos/day7_tune_pn.py
# Day 7 (Day 4): Tune PN gains (pn_N, pn_k_los) for a chosen disturbance field.
#
# Day 1 update:
#   - Fix spatial current sampling: fish uses current at fish position, turtle uses current at turtle position.
#   - (Optional) tiny one-time debug print hook (commented).
#
# Run examples:
#   py -m demos.day7_tune_pn --field shear  --episodes 30 --t_hold 0.1
#   py -m demos.day7_tune_pn --field vortex --episodes 30 --t_hold 0.1
#
# Output:
#   results/day7/tune_pn_<field>.csv
#   results/day7/heatmap_pn_robustness_<field>.png

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

    def step(self, dt: float, world: World, curx: float, cury: float) -> None:
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

        self.x += (self.speed * math.cos(self.heading) + curx) * dt
        self.y += (self.speed * math.sin(self.heading) + cury) * dt

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
    def __init__(self, x0: np.ndarray, P0: np.ndarray):
        self.x = x0.astype(float).copy()
        self.P = P0.astype(float).copy()

    def predict(self, dt: float, q_pos: float = 0.16, q_vel: float = 0.8) -> None:
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1,  0],
                      [0, 0, 0,  1]], dtype=float)
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


def current_field(field: str, strength: float, x: float, y: float) -> tuple[float, float]:
    # Deterministic flow fields (simple, stable)
    field = field.lower()

    if field == "shear":
        # drift grows with y (rightwards)
        return (strength * (y / 5.0), 0.0)

    if field == "vortex":
        # rotational around origin
        # u = -k*y, v = k*x
        k = strength * 0.9
        return (-k * (y / 5.0), k * (x / 5.0))

    if field == "gust":
        # time-varying gust: handled outside (we’ll approximate as position-based here)
        return (strength * 0.6 * math.sin(0.7 * y), strength * 0.6 * math.cos(0.7 * x))

    # default: uniform
    return (strength * 0.18, strength * 0.06)


def run_one_episode(field: str, strength: float, comp: float,
                    pn_N: float, pn_k_los: float,
                    seed: int, dt: float, timeout_s: float,
                    r_capture: float, t_hold: float) -> tuple[bool, float]:
    random.seed(seed)
    np.random.seed(seed)

    world = World()
    fish = Fish()

    sonar = Sonar()
    R = np.diag([sonar.sigma_r**2, sonar.sigma_b**2])

    x0 = np.array([fish.x + 0.7, fish.y - 0.6, 0.0, 0.0])
    P0 = np.diag([2.2, 2.2, 1.2, 1.2])
    ekf = FishEKF(x0, P0)

    state = TurtleState()
    params = TurtleParams()
    params.controller = "pn"
    params.current_comp = float(comp)
    params.pn_N = float(pn_N)
    params.pn_k_los = float(pn_k_los)

    robot = TurtleRobot(state=state, params=params, sonar=sonar, ekf=ekf, R=R)

    hold = 0.0
    t = 0.0
    steps = int(timeout_s / dt)

    for _ in range(steps):
        t += dt

        # --- Day 1 FIX: evaluate spatially varying current at each agent's position ---
        curx_t, cury_t = current_field(field, strength, robot.state.x, robot.state.y)
        curx_f, cury_f = current_field(field, strength, fish.x, fish.y)

        # Optional: one-time sanity print (keep commented unless debugging)
        # if abs(t - dt) < 1e-9:
        #     print(f"[dbg] field={field} strength={strength:.3f} cur_t=({curx_t:.3f},{cury_t:.3f}) cur_f=({curx_f:.3f},{cury_f:.3f})")

        fish.step(dt, world, curx_f, cury_f)
        robot.step(dt, fish, type("C", (), {"cx": curx_t, "cy": cury_t})())

        robot.state.x = clamp(robot.state.x, world.xmin, world.xmax)
        robot.state.y = clamp(robot.state.y, world.ymin, world.ymax)

        d = math.hypot(fish.x - robot.state.x, fish.y - robot.state.y)
        if d < r_capture:
            hold += dt
        else:
            hold = max(0.0, hold - 2.0 * dt)

        if hold >= t_hold:
            return True, t

    return False, timeout_s


def robustness_score(rates_by_current: list[float], currents: list[float], high_thr: float = 0.30) -> float:
    rates = np.array(rates_by_current, dtype=float)
    curs = np.array(currents, dtype=float)
    mean_rate = float(np.mean(rates))
    high = float(np.mean(rates[curs >= high_thr])) if np.any(curs >= high_thr) else mean_rate
    return 0.5 * mean_rate + 0.5 * high


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", type=str, default="shear", choices=["shear", "vortex", "gust"])
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--r_capture", type=float, default=0.30)
    ap.add_argument("--t_hold", type=float, default=2.0)

    ap.add_argument("--comp", type=float, default=0.70, help="fix compensation while tuning PN gains")
    ap.add_argument("--curr_min", type=float, default=0.00)
    ap.add_argument("--curr_max", type=float, default=0.50)
    ap.add_argument("--curr_steps", type=int, default=11)

    ap.add_argument("--N_min", type=float, default=2.0)
    ap.add_argument("--N_max", type=float, default=7.0)
    ap.add_argument("--N_steps", type=int, default=6)

    ap.add_argument("--k_min", type=float, default=1.0)
    ap.add_argument("--k_max", type=float, default=6.0)
    ap.add_argument("--k_steps", type=int, default=6)

    args = ap.parse_args()

    out_dir = Path("results/day7")
    out_dir.mkdir(parents=True, exist_ok=True)

    currents = np.linspace(args.curr_min, args.curr_max, args.curr_steps)
    Ns = np.linspace(args.N_min, args.N_max, args.N_steps)
    Ks = np.linspace(args.k_min, args.k_max, args.k_steps)

    heat = np.zeros((len(Ns), len(Ks)), dtype=float)

    csv_path = out_dir / f"tune_pn_{args.field}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["field", "comp", "pn_N", "pn_k_los", "episodes", "robustness", "mean_rate", "highcurrent_mean"])

        best = (-1.0, None)

        for i, N in enumerate(Ns):
            for j, k in enumerate(Ks):
                rates = []
                for ci, cur in enumerate(currents):
                    succ = 0
                    for e in range(args.episodes):
                        seed = args.seed + i * 1_000_000 + j * 10_000 + ci * 100 + e
                        ok, _tcap = run_one_episode(args.field, float(cur), float(args.comp),
                                                   float(N), float(k),
                                                   seed, args.dt, args.timeout, args.r_capture, args.t_hold)
                        succ += int(ok)
                    rates.append(succ / args.episodes)

                score = robustness_score(rates, currents.tolist(), high_thr=0.30)
                mean_rate = float(np.mean(rates))
                high_mean = float(np.mean([r for r, c in zip(rates, currents) if c >= 0.30])) if np.any(currents >= 0.30) else mean_rate

                heat[i, j] = score
                w.writerow([args.field, f"{args.comp:.3f}", f"{N:.3f}", f"{k:.3f}", args.episodes,
                            f"{score:.6f}", f"{mean_rate:.6f}", f"{high_mean:.6f}"])

                print(f"N={N:.2f} k={k:.2f} robust={score:.3f} mean={mean_rate:.3f} high={high_mean:.3f}")

                if score > best[0]:
                    best = (score, (N, k, mean_rate, high_mean))

    print(f"[wrote] {csv_path}")
    if best[1] is not None:
        N, k, mr, hm = best[1]
        print(f"[best] field={args.field} comp={args.comp:.2f}  N={N:.2f} k={k:.2f}  robust={best[0]:.3f} mean={mr:.3f} high={hm:.3f}")

    fig = plt.figure()
    plt.imshow(heat, origin="lower", aspect="auto",
               extent=[Ks[0], Ks[-1], Ns[0], Ns[-1]])
    plt.xlabel("pn_k_los")
    plt.ylabel("pn_N")
    plt.title(f"PN tuning heatmap (robustness) | field={args.field} | comp={args.comp:.2f} | episodes={args.episodes}")
    plt.colorbar(label="Robustness score")

    p = out_dir / f"heatmap_pn_robustness_{args.field}.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    print(f"[wrote] {p}")
    plt.show()


if __name__ == "__main__":
    main()