#!/usr/bin/env python3
"""Generate one window of stochastic irrigation particles for VIC PBS runs."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Generate daily irrigation particles for one VIC window.")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--n-particles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--event-probability", type=float, default=0.25)
    parser.add_argument("--irrigation-min-mm", type=float, default=5.0)
    parser.add_argument("--irrigation-max-mm", type=float, default=30.0)
    parser.add_argument("--out-csv", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for particle in range(args.n_particles):
        for date in pd.date_range(args.start_date, args.end_date, freq="D"):
            has_event = rng.random() < args.event_probability
            amount = rng.uniform(args.irrigation_min_mm, args.irrigation_max_mm) if has_event else 0.0
            rows.append(
                {
                    "particle": particle,
                    "date": date.date().isoformat(),
                    "irrigation_mm_day": float(amount),
                }
            )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print("Wrote irrigation table: {}".format(args.out_csv))


if __name__ == "__main__":
    main()

