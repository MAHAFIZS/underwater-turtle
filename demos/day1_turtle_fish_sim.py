# day1_turtle_fish_sim.py
# Day 1: 2D underwater turtle + fish sim with current + live animation
#
# Run:
#   python day1_turtle_fish_sim.py
#
# Requires:
#   pip install numpy matplotlib

from __future__ import annotations
import math
import random
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def wrap_angle(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
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
    cx: float = 0.15  # m/s
    cy: float = 0.05  # m/s


@dataclass
class Turtle:
    x: float = -4.0
    y: float = -4.0
    psi: float = 0.0          # heading (rad)
    v: float = 0.35           # slower base speed (m/s)
    v_max: float = 0.75       # cap speed lower for turtle feel
    omega_max: float = 0.75   # TURN RATE LIMIT (rad/s) -> slower turning

    def step(self, dt: float, omega_cmd: float, v_cmd: float, cur: Current) -> None:
        omega = clamp(omega_cmd, -self.omega_max, self.omega_max)

        # Add heading inertia (turning lag)
        # simple first-order response on omega
        if not hasattr(self, "_omega"):
            self._omega = 0.0
        omega_alpha = 3.0  # smaller = more sluggish
        self._omega += (omega - self._omega) * (1 - math.exp(-omega_alpha * dt))
        self.psi = wrap_angle(self.psi + self._omega * dt)

        # speed inertia (more sluggish)
        v_cmd = clamp(v_cmd, 0.0, self.v_max)
        alpha_v = 1.3  # smaller = more sluggish
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

    # behavior params
    turn_rate_max: float = 1.2    # rad/s
    wander_turn_std: float = 0.35 # rad/s noise scale
    burst_prob_per_s: float = 0.06
    burst_time_left: float = 0.0

    def step(self, dt: float, world: World, cur: Current) -> None:
        # occasional burst
        if self.burst_time_left <= 0.0:
            if random.random() < self.burst_prob_per_s * dt:
                self.burst_time_left = random.uniform(0.8, 1.8)
        else:
            self.burst_time_left -= dt

        target_speed = self.v_burst if self.burst_time_left > 0.0 else self.v_nom
        beta = 2.0
        self.speed += (target_speed - self.speed) * (1 - math.exp(-beta * dt))

        # smooth wandering turns
        d_heading = random.gauss(0.0, self.wander_turn_std) * dt
        d_heading = clamp(d_heading, -self.turn_rate_max * dt, self.turn_rate_max * dt)
        self.heading = wrap_angle(self.heading + d_heading)

        # integrate with current drift
        self.x += (self.speed * math.cos(self.heading) + cur.cx) * dt
        self.y += (self.speed * math.sin(self.heading) + cur.cy) * dt

        # bounce off walls (simple reflection)
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


def naive_chase_controller(turtle: Turtle, fish: Fish) -> tuple[float, float]:
    """
    Day 1 only: gentle chase for turtle-like motion.
    """
    dx = fish.x - turtle.x
    dy = fish.y - turtle.y
    desired = math.atan2(dy, dx)
    err = wrap_angle(desired - turtle.psi)

    # gentler heading control
    k_psi = 1.1
    omega_cmd = k_psi * err

    # turtle speeds up only a bit when far
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

    dt = 0.05  # 20 Hz

    # history for a short trail
    trail_len = 200
    turtle_hist = []
    fish_hist = []

    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(world.xmin, world.xmax)
    ax.set_ylim(world.ymin, world.ymax)
    ax.set_title("Day 1: Underwater Turtle vs Fish (2D) with Current")
    ax.grid(True, alpha=0.3)

    # Draw current vector
    cur_scale = 3.0
    ax.arrow(
        world.xmin + 0.7,
        world.ymax - 0.7,
        cur.cx * cur_scale,
        cur.cy * cur_scale,
        head_width=0.15,
        length_includes_head=True,
    )
    ax.text(world.xmin + 0.6, world.ymax - 0.4, "current", fontsize=10)

    # artists
    turtle_pt, = ax.plot([], [], marker="o", markersize=10, linestyle="None")
    fish_pt, = ax.plot([], [], marker="x", markersize=10, linestyle="None")
    turtle_trail, = ax.plot([], [], linewidth=1.5, alpha=0.8)
    fish_trail, = ax.plot([], [], linewidth=1.0, alpha=0.6)
    heading_line, = ax.plot([], [], linewidth=2.0, alpha=0.9)

    info = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=10
    )

    def init():
        turtle_pt.set_data([], [])
        fish_pt.set_data([], [])
        turtle_trail.set_data([], [])
        fish_trail.set_data([], [])
        heading_line.set_data([], [])
        info.set_text("")
        return turtle_pt, fish_pt, turtle_trail, fish_trail, heading_line, info

    def update(_frame):
        # step fish
        fish.step(dt=dt, world=world, cur=cur)

        # controller (Day 1 naive)
        omega_cmd, v_cmd = naive_chase_controller(turtle, fish)
        turtle.step(dt=dt, omega_cmd=omega_cmd, v_cmd=v_cmd, cur=cur)

        # keep turtle inside bounds (soft clamp)
        turtle.x = clamp(turtle.x, world.xmin, world.xmax)
        turtle.y = clamp(turtle.y, world.ymin, world.ymax)

        # record trails
        turtle_hist.append((turtle.x, turtle.y))
        fish_hist.append((fish.x, fish.y))
        if len(turtle_hist) > trail_len:
            turtle_hist.pop(0)
        if len(fish_hist) > trail_len:
            fish_hist.pop(0)

        # update artists
        turtle_pt.set_data([turtle.x], [turtle.y])
        fish_pt.set_data([fish.x], [fish.y])

        tx, ty = zip(*turtle_hist)
        fx, fy = zip(*fish_hist)
        turtle_trail.set_data(tx, ty)
        fish_trail.set_data(fx, fy)

        # heading line
        L = 0.6
        hx = [turtle.x, turtle.x + L * math.cos(turtle.psi)]
        hy = [turtle.y, turtle.y + L * math.sin(turtle.psi)]
        heading_line.set_data(hx, hy)

        dist = math.hypot(fish.x - turtle.x, fish.y - turtle.y)
        info.set_text(
            f"distance: {dist:.2f} m\n"
            f"turtle v: {turtle.v:.2f} m/s\n"
            f"fish v: {fish.speed:.2f} m/s"
        )

        return turtle_pt, fish_pt, turtle_trail, fish_trail, heading_line, info

    ani = FuncAnimation(fig, update, init_func=init, interval=int(dt * 1000), blit=True)
    plt.show()


if __name__ == "__main__":
    main()
