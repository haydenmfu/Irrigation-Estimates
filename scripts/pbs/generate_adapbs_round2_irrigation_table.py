#!/usr/bin/env python3
"""Generate AdaPBS-style adapted round-2 irrigation particles.

The adapted proposal is learned from the round-1 sequential PBS posterior:
for each date, use the weighted event frequency and weighted positive-event
amount moments. A small prior-mixture component is retained to avoid collapsing
the proposal support completely.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Generate adapted AdaPBS round-2 irrigation table.")
    parser.add_argument("--round1-root", type=Path, required=True)
    parser.add_argument("--round1-output", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--n-particles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=202607061)
    parser.add_argument("--prior-mixture", type=float, default=0.20)
    parser.add_argument("--prior-event-probability", type=float, default=0.25)
    parser.add_argument("--prior-min-mm", type=float, default=5.0)
    parser.add_argument("--prior-max-mm", type=float, default=30.0)
    parser.add_argument("--min-event-probability", type=float, default=0.03)
    parser.add_argument("--max-event-probability", type=float, default=0.85)
    return parser.parse_args()


def weighted_mean(values, weights):
    weights = np.asarray(weights, dtype=float)
    values = np.asarray(values, dtype=float)
    total = weights.sum()
    if total <= 0:
        return float(np.nanmean(values))
    return float(np.sum(values * weights) / total)


def main():
    args = parse_args()
    irrigation = pd.read_csv(args.round1_root / "particle_irrigation_inputs.csv")
    weights = pd.read_csv(args.round1_output / "pbs_particle_weights.csv")
    if "window_id" not in weights.columns:
        raise ValueError("Round-1 weights must include window_id.")

    irrigation["date"] = pd.to_datetime(irrigation["date"])
    weights = weights[["window_id", "particle", "weight"]].copy()
    dates = sorted(irrigation["date"].unique())

    # Map dates to the same 7-day windows used by the Week 6 PBS postprocessor.
    start = pd.Timestamp(min(dates))
    date_window = {}
    for date in dates:
        date = pd.Timestamp(date)
        date_window[date] = int(((date - start).days // 7) + 1)

    rng = np.random.default_rng(args.seed)
    rows = []
    diagnostics = []
    for date in dates:
        date = pd.Timestamp(date)
        window_id = date_window[date]
        day = irrigation[irrigation["date"].eq(date)].merge(
            weights[weights["window_id"].eq(window_id)], on="particle", how="left"
        )
        day["weight"] = day["weight"].fillna(1.0 / max(1, day["particle"].nunique()))
        positive = day["irrigation_mm_day"] > 0
        weighted_event_prob = weighted_mean(positive.astype(float), day["weight"])
        adapted_event_prob = np.clip(
            (1.0 - args.prior_mixture) * weighted_event_prob
            + args.prior_mixture * args.prior_event_probability,
            args.min_event_probability,
            args.max_event_probability,
        )

        positives = day[positive].copy()
        if positives.empty:
            amount_mean = 0.5 * (args.prior_min_mm + args.prior_max_mm)
            amount_sd = 0.25 * (args.prior_max_mm - args.prior_min_mm)
        else:
            pos_weights = positives["weight"].to_numpy(dtype=float)
            amount_values = positives["irrigation_mm_day"].to_numpy(dtype=float)
            amount_mean = weighted_mean(amount_values, pos_weights)
            amount_var = weighted_mean((amount_values - amount_mean) ** 2, pos_weights)
            amount_sd = max(2.5, float(np.sqrt(max(amount_var, 0.0))))

        diagnostics.append(
            {
                "date": date.date().isoformat(),
                "window_id": window_id,
                "round1_weighted_event_probability": weighted_event_prob,
                "adapted_event_probability": adapted_event_prob,
                "adapted_positive_amount_mean_mm": amount_mean,
                "adapted_positive_amount_sd_mm": amount_sd,
            }
        )

        for particle in range(args.n_particles):
            if rng.random() < args.prior_mixture:
                has_event = rng.random() < args.prior_event_probability
                amount = rng.uniform(args.prior_min_mm, args.prior_max_mm) if has_event else 0.0
            else:
                has_event = rng.random() < adapted_event_prob
                if has_event:
                    amount = rng.normal(amount_mean, amount_sd)
                    amount = float(np.clip(amount, 0.0, args.prior_max_mm))
                    if amount < 0.1:
                        amount = 0.0
                else:
                    amount = 0.0
            rows.append(
                {
                    "particle": particle,
                    "date": date.date().isoformat(),
                    "irrigation_mm_day": float(amount),
                    "proposal_round": 2,
                    "proposal_type": "adapted_mixture",
                }
            )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    diag_path = args.out_csv.with_name(args.out_csv.stem + "_diagnostics.csv")
    pd.DataFrame(diagnostics).to_csv(diag_path, index=False)
    print("Wrote adapted round-2 irrigation table: {}".format(args.out_csv))
    print("Wrote adapted proposal diagnostics: {}".format(diag_path))


if __name__ == "__main__":
    main()
