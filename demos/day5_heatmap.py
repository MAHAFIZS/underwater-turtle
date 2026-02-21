# demos/day5_heatmap.py
from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.grid import eval_grid, GridArgs
from src.controllers.pure import PurePursuitController
from src.controllers.pn import PNController


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--r_capture", type=float, default=0.35)
    ap.add_argument("--t_hold", type=float, default=0.1)

    ap.add_argument("--curr_min", type=float, default=0.00)
    ap.add_argument("--curr_max", type=float, default=0.50)
    ap.add_argument("--curr_steps", type=int, default=11)

    ap.add_argument("--comp_min", type=float, default=0.10)
    ap.add_argument("--comp_max", type=float, default=0.90)
    ap.add_argument("--comp_steps", type=int, default=9)

    ap.add_argument("--field", type=str, default="vortex", choices=["vortex", "shear", "gust"])
    ap.add_argument("--controller", type=str, default="pure", choices=["pure", "pn"])
    args = ap.parse_args()

    g = GridArgs(
        episodes=args.episodes,
        seed=args.seed,
        timeout=args.timeout,
        dt=args.dt,
        r_capture=args.r_capture,
        t_hold=args.t_hold,
        curr_min=args.curr_min,
        curr_max=args.curr_max,
        curr_steps=args.curr_steps,
        comp_min=args.comp_min,
        comp_max=args.comp_max,
        comp_steps=args.comp_steps,
    )

    out_dir = Path("results/day5")

    if args.controller == "pure":
        controller_factory = lambda: PurePursuitController()
    else:
        controller_factory = lambda: PNController()

    eval_grid(
        field=args.field,
        controller_name=args.controller,
        controller_factory=controller_factory,
        out_dir=out_dir,
        args=g,
    )


if __name__ == "__main__":
    main()