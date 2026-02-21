# demos/day6_compare.py
# Day 6 (Day 3): Compare controllers across fields using the grid CSVs from Day 5.
#
# Run:
#   py -m demos.day6_compare --episodes 40 --fields vortex,shear
#
# Outputs:
#   results/day6/summary.csv
#   results/day6/compare_mean_rate.png
#   results/day6/compare_p10_rate.png
#   results/day6/compare_highcurrent_mean.png
#   results/day6/compare_robustness.png

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class GridStats:
    field: str
    controller: str
    episodes: int
    mean_rate: float
    p10_rate: float
    best_rate: float
    highcurrent_mean: float
    robustness: float


def read_grid_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def _parse_current(row: dict) -> float:
    # support both old/new csv headers
    if "current_strength" in row:
        return float(row["current_strength"])
    if "current" in row:
        return float(row["current"])
    return float("nan")


def compute_stats(rows: list[dict], field: str, controller: str, episodes: int, high_current_thresh: float) -> GridStats:
    rates = np.array([float(r["capture_rate"]) for r in rows], dtype=float)
    currents = np.array([_parse_current(r) for r in rows], dtype=float)

    mean_rate = float(np.mean(rates))
    p10_rate = float(np.quantile(rates, 0.10))  # robust "worst-case"
    best_rate = float(np.max(rates))

    mask = currents >= high_current_thresh
    high_mean = float(np.mean(rates[mask])) if np.any(mask) else float("nan")

    # robustness: overall + high-current (equal weight)
    robustness = 0.5 * mean_rate + 0.5 * (high_mean if not np.isnan(high_mean) else mean_rate)

    return GridStats(
        field=field,
        controller=controller,
        episodes=episodes,
        mean_rate=mean_rate,
        p10_rate=p10_rate,
        best_rate=best_rate,
        highcurrent_mean=high_mean,
        robustness=robustness,
    )


def grouped_bar_plot(stats: list[GridStats], metric: str, out_path: Path, title: str) -> None:
    fields = sorted({s.field for s in stats})
    controllers = ["pure", "pn"]

    M = np.zeros((len(fields), len(controllers)), dtype=float)
    for i, f in enumerate(fields):
        for j, c in enumerate(controllers):
            s = next((x for x in stats if x.field == f and x.controller == c), None)
            M[i, j] = getattr(s, metric) if s is not None else np.nan

    x = np.arange(len(fields))
    width = 0.35

    fig = plt.figure()
    plt.title(title)
    plt.xticks(x, fields)
    plt.ylabel(metric)

    plt.bar(x - width / 2, M[:, 0], width, label="pure")
    plt.bar(x + width / 2, M[:, 1], width, label="pn")

    ymax = float(np.nanmax(M)) if np.isfinite(np.nanmax(M)) else 1.0
    plt.ylim(0.0, min(1.0, ymax * 1.15 + 1e-6))

    plt.legend()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"[wrote] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--fields", type=str, default="vortex,shear")
    ap.add_argument("--high_current", type=float, default=0.30)
    args = ap.parse_args()

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    controllers = ["pure", "pn"]

    in_dir = Path("results/day5")
    out_dir = Path("results/day6")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats: list[GridStats] = []

    for field in fields:
        for ctrl in controllers:
            csv_path = in_dir / f"grid_metrics_{field}_{ctrl}.csv"
            if not csv_path.exists():
                print(f"[skip] missing {csv_path} (run day5_heatmap first)")
                continue

            rows = read_grid_csv(csv_path)
            s = compute_stats(rows, field=field, controller=ctrl, episodes=args.episodes, high_current_thresh=args.high_current)
            all_stats.append(s)
            print(
                f"[ok] {field:>6} {ctrl:>4} "
                f"mean={s.mean_rate:.3f} p10={s.p10_rate:.3f} high={s.highcurrent_mean:.3f} robust={s.robustness:.3f}"
            )

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["field", "controller", "episodes", "mean_rate", "p10_rate", "best_rate", "highcurrent_mean", "robustness"])
        for s in all_stats:
            w.writerow(
                [s.field, s.controller, s.episodes,
                 f"{s.mean_rate:.6f}", f"{s.p10_rate:.6f}", f"{s.best_rate:.6f}",
                 f"{s.highcurrent_mean:.6f}", f"{s.robustness:.6f}"]
            )
    print(f"[wrote] {summary_path}")

    grouped_bar_plot(all_stats, "mean_rate", out_dir / "compare_mean_rate.png", "Mean capture rate (grid average)")
    grouped_bar_plot(all_stats, "p10_rate", out_dir / "compare_p10_rate.png", "10th-percentile capture rate (robust worst-case)")
    grouped_bar_plot(all_stats, "highcurrent_mean", out_dir / "compare_highcurrent_mean.png", f"High-current mean (current >= {args.high_current:.2f})")
    grouped_bar_plot(all_stats, "robustness", out_dir / "compare_robustness.png", "Robustness = 0.5*mean + 0.5*high-current mean")


if __name__ == "__main__":
    main()
