#!/usr/bin/env python3
"""Score one sequential PBS window and write a resampled state plan."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


TARGET_MODES = ("cropland_only", "crop_minus_control", "joint_crop_control")


def parse_args():
    parser = argparse.ArgumentParser(description="Score one daily delta-SM PBS window on Hopper.")
    parser.add_argument("--window-root", type=Path, required=True)
    parser.add_argument("--target-csv", type=Path, required=True)
    parser.add_argument("--window-id", type=int, required=True)
    parser.add_argument("--window-start", required=True)
    parser.add_argument("--window-end", required=True)
    parser.add_argument("--target-mode", choices=TARGET_MODES, default="cropland_only")
    parser.add_argument("--delta-sigma-m3m3", type=float, default=0.075)
    parser.add_argument("--layer1-depth-mm", type=float, default=50.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resample-out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260709)
    return parser.parse_args()


def particle_number(path):
    return int(path.name.split("_")[-1])


def read_particle_daily_sm(window_root, layer1_depth_mm):
    rows = []
    for particle_dir in sorted(path for path in window_root.glob("particle_*") if path.is_dir()):
        particle = particle_number(particle_dir)
        for result_file in sorted((particle_dir / "results").glob("*.nc")):
            with xr.open_dataset(result_file) as ds:
                top = ds["OUT_SOIL_MOIST"].isel(nlayer=0) / layer1_depth_mm
                for time_i, time in enumerate(pd.to_datetime(top["time"].values)):
                    arr = top.isel(time=time_i).values
                    rows.append(
                        {
                            "particle": particle,
                            "date": time.date().isoformat(),
                            "particle_basin_SM_m3m3": float(np.nanmean(arr)),
                        }
                    )
    if not rows:
        raise RuntimeError("No VIC particle output rows found under {}".format(window_root))
    return pd.DataFrame(rows)


def build_particle_interval_deltas(particle_daily, targets):
    rows = []
    for _, interval in targets[["interval_start", "interval_end"]].drop_duplicates().iterrows():
        start = particle_daily[particle_daily["date"].eq(interval["interval_start"])][
            ["particle", "particle_basin_SM_m3m3"]
        ]
        end = particle_daily[particle_daily["date"].eq(interval["interval_end"])][
            ["particle", "particle_basin_SM_m3m3"]
        ]
        merged = start.merge(end, on="particle", suffixes=("_t0", "_t1"))
        merged["interval_start"] = interval["interval_start"]
        merged["interval_end"] = interval["interval_end"]
        merged["particle_delta_SM"] = merged["particle_basin_SM_m3m3_t1"] - merged["particle_basin_SM_m3m3_t0"]
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def residual_columns_for_mode(mode, merged):
    if mode == "cropland_only":
        valid = np.isfinite(merged["cropland_delta_SM_satellite"])
        residual = merged["cropland_delta_SM_satellite"] - merged["particle_delta_SM"]
        return [("crop_minus_particle", residual.where(valid))]
    if mode == "crop_minus_control":
        sat_anom = merged["crop_minus_control_delta_SM_satellite"]
        sim_anom = merged["particle_delta_SM"] - merged["basin0_open_loop_delta_SM"]
        valid = np.isfinite(sat_anom) & np.isfinite(sim_anom)
        residual = sat_anom - sim_anom
        return [("crop_control_minus_particle_openloop", residual.where(valid))]
    if mode == "joint_crop_control":
        crop_valid = np.isfinite(merged["cropland_delta_SM_satellite"])
        crop_resid = merged["cropland_delta_SM_satellite"] - merged["particle_delta_SM"]
        sat_anom = merged["crop_minus_control_delta_SM_satellite"]
        sim_anom = merged["particle_delta_SM"] - merged["basin0_open_loop_delta_SM"]
        anom_valid = np.isfinite(sat_anom) & np.isfinite(sim_anom)
        anom_resid = sat_anom - sim_anom
        return [
            ("crop_minus_particle", crop_resid.where(crop_valid)),
            ("crop_control_minus_particle_openloop", anom_resid.where(anom_valid)),
        ]
    raise ValueError("Unknown target mode: {}".format(mode))


def normalize_log_weights(log_values):
    log_values = np.asarray(log_values, dtype=float)
    max_log = np.nanmax(log_values)
    weights = np.exp(log_values - max_log)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return np.ones_like(log_values) / len(log_values)
    return weights / total


def summarize_irrigation(window_root, scores, window_start, window_end):
    irrigation = pd.read_csv(window_root / "particle_irrigation_inputs.csv")
    irrigation = irrigation[(irrigation["date"] >= window_start) & (irrigation["date"] <= window_end)].copy()
    weighted = irrigation.merge(scores[["particle", "weight"]], on="particle", how="left")
    weighted["weighted_irrigation_mm_day"] = weighted["irrigation_mm_day"] * weighted["weight"]
    return (
        weighted.groupby("date", as_index=False)
        .agg(
            posterior_mean_irrigation_mm=("weighted_irrigation_mm_day", "sum"),
            prior_mean_irrigation_mm=("irrigation_mm_day", "mean"),
            max_particle_irrigation_mm=("irrigation_mm_day", "max"),
        )
        .sort_values("date")
    )


def systematic_resample(weights, seed):
    rng = np.random.default_rng(seed)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    indexes = np.zeros(n, dtype=int)
    i = 0
    j = 0
    while i < n:
        if positions[i] < cumulative[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def state_file_for_particle(window_root, particle, window_end):
    token = pd.Timestamp(window_end).strftime("%Y%m%d")
    pattern = "state.{}_*.nc".format(token)
    matches = sorted((window_root / "particle_{:04d}".format(int(particle)) / "state").glob(pattern))
    if not matches:
        raise FileNotFoundError("No state file for particle {} matching {}".format(particle, pattern))
    return matches[0]


def write_resample_plan(args, scores):
    if args.resample_out is None:
        return
    ordered = scores.sort_values("particle").reset_index(drop=True)
    parent_indexes = systematic_resample(ordered["weight"].to_numpy(), args.seed)
    rows = []
    for new_particle, parent_idx in enumerate(parent_indexes):
        parent = ordered.iloc[int(parent_idx)]
        parent_particle = int(parent["particle"])
        rows.append(
            {
                "particle": new_particle,
                "parent_particle": parent_particle,
                "parent_weight": float(parent["weight"]),
                "initial_state": str(state_file_for_particle(args.window_root, parent_particle, args.window_end)),
                "source_window_id": args.window_id,
                "source_window_start": args.window_start,
                "source_window_end": args.window_end,
            }
        )
    args.resample_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.resample_out, index=False)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = pd.read_csv(args.target_csv)
    targets = targets[targets["window_id"].eq(args.window_id)].copy()
    if targets.empty:
        raise RuntimeError("No targets for window_id {} in {}".format(args.window_id, args.target_csv))

    particle_daily = read_particle_daily_sm(args.window_root, args.layer1_depth_mm)
    particle_deltas = build_particle_interval_deltas(particle_daily, targets)
    merged = particle_deltas.merge(targets, on=["interval_start", "interval_end"], how="inner")
    residual_parts = residual_columns_for_mode(args.target_mode, merged)
    merged["log_likelihood_component_sum"] = 0.0
    merged["n_residual_components"] = 0
    for name, residual in residual_parts:
        merged[name] = residual
        valid = np.isfinite(residual)
        merged.loc[valid, "log_likelihood_component_sum"] += -0.5 * (residual[valid] / args.delta_sigma_m3m3) ** 2
        merged.loc[valid, "n_residual_components"] += 1

    matches = merged[merged["n_residual_components"] > 0].copy()
    particle_ids = sorted(particle_daily["particle"].unique())
    grouped = (
        matches.groupby("particle", as_index=False)
        .agg(
            log_likelihood=("log_likelihood_component_sum", "sum"),
            n_observation_intervals=("interval_end", "nunique"),
            n_residual_components=("n_residual_components", "sum"),
        )
    )
    scores = pd.DataFrame({"particle": particle_ids}).merge(grouped, on="particle", how="left")
    scores["log_likelihood"] = scores["log_likelihood"].fillna(0.0)
    scores["n_observation_intervals"] = scores["n_observation_intervals"].fillna(0).astype(int)
    scores["n_residual_components"] = scores["n_residual_components"].fillna(0).astype(int)
    scores["prior_weight"] = 1.0 / len(scores)
    scores["log_posterior_unnormalized"] = np.log(scores["prior_weight"]) + scores["log_likelihood"]
    scores["weight"] = normalize_log_weights(scores["log_posterior_unnormalized"].to_numpy())
    scores["rank"] = scores["weight"].rank(ascending=False, method="first").astype(int)
    scores = scores.sort_values("rank").copy()

    daily = summarize_irrigation(args.window_root, scores, args.window_start, args.window_end)
    ess = 1.0 / float(np.sum(scores["weight"] ** 2))
    summary = pd.DataFrame(
        [
            {
                "window_id": args.window_id,
                "window_start": args.window_start,
                "window_end": args.window_end,
                "target_mode": args.target_mode,
                "delta_sigma_m3m3": args.delta_sigma_m3m3,
                "n_daily_observation_rows": int(len(targets)),
                "effective_sample_size": ess,
                "max_particle_weight": float(scores["weight"].max()),
                "best_particle": int(scores.iloc[0]["particle"]),
                "posterior_irrigation_sum_mm": float(daily["posterior_mean_irrigation_mm"].sum()),
                "prior_irrigation_sum_mm": float(daily["prior_mean_irrigation_mm"].sum()),
            }
        ]
    )

    matches.to_csv(args.out_dir / "particle_interval_matches.csv", index=False)
    scores.to_csv(args.out_dir / "particle_weights.csv", index=False)
    daily.to_csv(args.out_dir / "posterior_daily_irrigation.csv", index=False)
    particle_daily.to_csv(args.out_dir / "particle_daily_basin_mean_sm.csv", index=False)
    summary.to_csv(args.out_dir / "window_summary.csv", index=False)
    write_resample_plan(args, scores)
    print(summary.to_string(index=False))
    if args.resample_out is not None:
        print("Wrote resampled initial-state plan: {}".format(args.resample_out))


if __name__ == "__main__":
    main()
