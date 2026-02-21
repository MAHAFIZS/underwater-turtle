# src/eval/grid.py
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.sim.episode import run_one_episode


@dataclass
class GridArgs:
    episodes: int = 40
    seed: int = 7
    timeout: float = 45.0
    dt: float = 0.05
    r_capture: float = 0.35
    t_hold: float = 0.1

    curr_min: float = 0.00
    curr_max: float = 0.50
    curr_steps: int = 11

    comp_min: float = 0.10
    comp_max: float = 0.90
    comp_steps: int = 9


def linspace(a: float, b: float, n: int) -> np.ndarray:
    if n == 1:
        return np.array([a], dtype=float)
    return np.linspace(a, b, n)


def eval_grid(*, field: str, controller_name: str, controller_factory,
              out_dir: Path, args: GridArgs) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    currents = linspace(args.curr_min, args.curr_max, args.curr_steps)
    comps = linspace(args.comp_min, args.comp_max, args.comp_steps)

    heat = np.zeros((len(comps), len(currents)), dtype=float)

    csv_path = out_dir / f"grid_metrics_{field}_{controller_name}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["field", "controller", "comp", "current_strength", "episodes", "successes",
                    "capture_rate", "mean_capture_time_s"])

        for i, comp in enumerate(comps):
            for j, cur_strength in enumerate(currents):
                succ = 0
                times: list[float] = []

                for e in range(args.episodes):
                    controller = controller_factory()  # new per episode (clean state)
                    seed = args.seed + i * 1_000_000 + j * 10_000 + e
                    ok, tcap = run_one_episode(
                        field=field,
                        cur_strength=float(cur_strength),
                        comp=float(comp),
                        controller=controller,
                        seed=seed,
                        dt=args.dt,
                        timeout_s=args.timeout,
                        r_capture=args.r_capture,
                        t_hold=args.t_hold,
                    )
                    if ok:
                        succ += 1
                        times.append(float(tcap))

                rate = succ / args.episodes
                heat[i, j] = rate
                mt = float(np.mean(times)) if times else float("nan")

                w.writerow([field, controller_name, f"{comp:.3f}", f"{cur_strength:.3f}",
                            args.episodes, succ, f"{rate:.4f}", "" if not times else f"{mt:.3f}"])
                print(f"field={field} ctrl={controller_name} comp={comp:.2f} cur={cur_strength:.2f} rate={rate:.2f}")

    print(f"[wrote] {csv_path}")

    fig = plt.figure()
    plt.imshow(
        heat, origin="lower", aspect="auto",
        extent=[currents[0], currents[-1], comps[0], comps[-1]],
    )
    plt.xlabel("Current strength (field scale)")
    plt.ylabel("Turtle current_comp (0=perfect, 1=none)")
    plt.title(f"Capture rate heatmap | field={field} | ctrl={controller_name} | episodes={args.episodes}")
    plt.colorbar(label="Capture rate")

    png_path = out_dir / f"heatmap_capture_rate_{field}_{controller_name}.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    print(f"[wrote] {png_path}")

    plt.show()
    return csv_path, png_path