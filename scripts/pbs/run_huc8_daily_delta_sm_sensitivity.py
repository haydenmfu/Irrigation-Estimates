#!/usr/bin/env python3
"""Daily/within-window HUC8 SMAP L3 delta-SM PBS sensitivity tests.

This extends the weekly delta-SM diagnostic by using all consecutive high-quality
SMAP L3 pairs within each 7-day PBS window. It still keeps irrigation estimates
on the VIC particle irrigation table, but tests several likelihood scales and
alternative delta-SM targets.
"""

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
WEEK3 = PROJECT / "Week 3"
WEEK6 = PROJECT / "Week 6"
WEEK7 = PROJECT / "Week 7"
plt.style.use(str(PROJECT / "research_report.mplstyle"))
sys.path.insert(0, str(WEEK3))

from analyze_pbs_week_particles_cellwise import read_particle_vic_cells  # noqa: E402


DEFAULT_WEEK_ROOT = (
    WEEK6
    / "data"
    / "fresh_pbs_runs_4week_N100"
    / "extracted"
    / "pbs4week_absoluteSM_mlpSatSM_threshold_N100_20260706_135223"
)
DEFAULT_ENDPOINTS = (
    WEEK7
    / "outputs"
    / "delta_sm_pre_pbs"
    / "huc8_10180009_weekly_vic_aligned"
    / "satellite_weekly"
    / "huc8_smap_l3_endpoint_observations.csv"
)
DEFAULT_CDL = (
    WEEK7
    / "outputs"
    / "delta_sm_pre_pbs"
    / "huc8_10180009_cdl_crop30_power_rain_partial_current"
    / "huc8_smap_cell_cropland_summary.csv"
)
DEFAULT_VIC_DAILY = (
    WEEK3
    / "data"
    / "VIC_basin0_outputs"
    / "dry_spottedtail_creek_vic_basin_daily_summary_for_satellite.csv"
)
DEFAULT_RAINFALL = WEEK7 / "data" / "rainfall" / "huc8_10180009_power_daily_precip_2018_2021.csv"
DEFAULT_OUT_ROOT = (
    WEEK7
    / "outputs"
    / "delta_sm_pre_pbs"
    / "huc8_10180009_weekly_vic_aligned"
    / "diagnostics"
)

TARGET_MODES = ("cropland_only", "crop_minus_control", "joint_crop_control")


def parse_args():
    parser = argparse.ArgumentParser(description="Run daily HUC8 SMAP L3 delta-SM PBS sensitivity tests.")
    parser.add_argument("--week-root", type=Path, default=DEFAULT_WEEK_ROOT)
    parser.add_argument("--endpoint-observations", type=Path, default=DEFAULT_ENDPOINTS)
    parser.add_argument("--cdl-summary", type=Path, default=DEFAULT_CDL)
    parser.add_argument("--vic-basin-daily", type=Path, default=DEFAULT_VIC_DAILY)
    parser.add_argument("--rainfall-csv", type=Path, default=DEFAULT_RAINFALL)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--cdl-year", type=int, default=2020)
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--max-pair-gap-days", type=int, default=3)
    parser.add_argument("--layer1-depth-mm", type=float, default=50.0)
    parser.add_argument(
        "--sigmas",
        default="0.015,0.025,0.035,0.050,0.075",
        help="Comma-separated delta-SM likelihood scales to test.",
    )
    parser.add_argument(
        "--target-modes",
        default=",".join(TARGET_MODES),
        help="Comma-separated target modes: cropland_only,crop_minus_control,joint_crop_control.",
    )
    parser.add_argument("--weight-mask-class", default="cropland_like")
    parser.add_argument("--control-mask-class", default="noncropland_control")
    parser.add_argument("--quality-mode", choices=["high_quality", "all_finite"], default="high_quality")
    parser.add_argument("--low-ess-threshold", type=float, default=3.0)
    return parser.parse_args()


def git_commit_hash():
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def bool_series(values):
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def parse_float_list(text):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_mode_list(text):
    modes = [item.strip() for item in text.split(",") if item.strip()]
    unknown = sorted(set(modes) - set(TARGET_MODES))
    if unknown:
        raise ValueError("Unknown target modes: {}".format(unknown))
    return modes


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def weekly_windows(start_date, end_date, window_days):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    windows = []
    cur = start
    idx = 1
    while cur <= end:
        win_end = min(cur + pd.Timedelta(days=window_days - 1), end)
        windows.append(
            {
                "window_id": idx,
                "window_start": cur.date().isoformat(),
                "window_end": win_end.date().isoformat(),
            }
        )
        cur = win_end + pd.Timedelta(days=1)
        idx += 1
    return windows


def window_for_interval(start_date, end_date, windows):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    for window in windows:
        w0 = pd.Timestamp(window["window_start"])
        w1 = pd.Timestamp(window["window_end"])
        if start >= w0 and end <= w1 and end > start:
            return window
    return None


def build_daily_satellite_delta_targets(args, windows, start_date, end_date):
    obs = pd.read_csv(args.endpoint_observations, parse_dates=["date"])
    obs["retrieval_recommended_bool"] = bool_series(obs["retrieval_recommended"])
    obs = obs[obs["soil_moisture"].notna()].copy()
    if args.quality_mode == "high_quality":
        obs = obs[obs["retrieval_recommended_bool"]].copy()
    obs = obs[(obs["date"] >= pd.Timestamp(start_date)) & (obs["date"] <= pd.Timestamp(end_date))].copy()

    cdl = pd.read_csv(args.cdl_summary)
    cdl = cdl[cdl["year"].eq(args.cdl_year)].copy()
    cdl_cols = [
        "row",
        "col",
        "mask_class",
        "crop_fraction",
        "noncropland_fraction",
        "excluded_fraction",
        "dominant_cdl_class",
    ]
    obs = obs.merge(cdl[cdl_cols], on=["row", "col"], how="left")
    obs["mask_class"] = obs["mask_class"].fillna("unknown")

    pair_rows = []
    obs = obs.sort_values(["row", "col", "period", "date"])
    for (row, col, period), group in obs.groupby(["row", "col", "period"]):
        group = group.sort_values("date").reset_index(drop=True)
        for i in range(len(group) - 1):
            a = group.iloc[i]
            b = group.iloc[i + 1]
            gap = int((pd.Timestamp(b["date"]) - pd.Timestamp(a["date"])).days)
            if gap < 1 or gap > args.max_pair_gap_days:
                continue
            window = window_for_interval(a["date"], b["date"], windows)
            if window is None:
                continue
            pair_rows.append(
                {
                    "window_id": window["window_id"],
                    "window_start": window["window_start"],
                    "window_end": window["window_end"],
                    "interval_start": pd.Timestamp(a["date"]).date().isoformat(),
                    "interval_end": pd.Timestamp(b["date"]).date().isoformat(),
                    "interval_days": gap,
                    "row": int(row),
                    "col": int(col),
                    "smap_cell_id": a["smap_cell_id"],
                    "period": period,
                    "mask_class": a["mask_class"],
                    "overlap_area_km2": float(a["overlap_area_km2"]),
                    "SMAP_SM_t0": float(a["soil_moisture"]),
                    "SMAP_SM_t1": float(b["soil_moisture"]),
                    "delta_SM_satellite": float(b["soil_moisture"] - a["soil_moisture"]),
                    "retrieval_qual_flag_t0": float(a["retrieval_qual_flag"])
                    if np.isfinite(a["retrieval_qual_flag"])
                    else np.nan,
                    "retrieval_qual_flag_t1": float(b["retrieval_qual_flag"])
                    if np.isfinite(b["retrieval_qual_flag"])
                    else np.nan,
                }
            )

    pairs = pd.DataFrame(pair_rows)
    if pairs.empty:
        return pairs, pd.DataFrame()

    agg_rows = []
    group_cols = ["window_id", "window_start", "window_end", "interval_start", "interval_end", "period", "mask_class"]
    for keys, group in pairs.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["interval_days"] = int((pd.Timestamp(row["interval_end"]) - pd.Timestamp(row["interval_start"])).days)
        row["n_pairs"] = int(len(group))
        row["area_km2"] = float(group["overlap_area_km2"].sum())
        row["delta_SM_satellite_area_weighted"] = weighted_mean(
            group["delta_SM_satellite"], group["overlap_area_km2"]
        )
        row["delta_SM_satellite_mean"] = float(group["delta_SM_satellite"].mean())
        row["delta_SM_satellite_sd"] = float(group["delta_SM_satellite"].std(ddof=1)) if len(group) > 1 else np.nan
        agg_rows.append(row)
    aggregates = pd.DataFrame(agg_rows)

    target_rows = []
    base_cols = ["window_id", "window_start", "window_end", "interval_start", "interval_end", "period"]
    for keys, group in aggregates.groupby(base_cols):
        row = dict(zip(base_cols, keys))
        crop = group[group["mask_class"].eq(args.weight_mask_class)]
        control = group[group["mask_class"].eq(args.control_mask_class)]
        if not crop.empty:
            crop_row = crop.iloc[0]
            row["cropland_delta_SM_satellite"] = float(crop_row["delta_SM_satellite_area_weighted"])
            row["cropland_n_pairs"] = int(crop_row["n_pairs"])
            row["cropland_area_km2"] = float(crop_row["area_km2"])
        else:
            row["cropland_delta_SM_satellite"] = np.nan
            row["cropland_n_pairs"] = 0
            row["cropland_area_km2"] = 0.0
        if not control.empty:
            control_row = control.iloc[0]
            row["control_delta_SM_satellite"] = float(control_row["delta_SM_satellite_area_weighted"])
            row["control_n_pairs"] = int(control_row["n_pairs"])
            row["control_area_km2"] = float(control_row["area_km2"])
        else:
            row["control_delta_SM_satellite"] = np.nan
            row["control_n_pairs"] = 0
            row["control_area_km2"] = 0.0
        row["crop_minus_control_delta_SM_satellite"] = (
            row["cropland_delta_SM_satellite"] - row["control_delta_SM_satellite"]
            if np.isfinite(row["cropland_delta_SM_satellite"]) and np.isfinite(row["control_delta_SM_satellite"])
            else np.nan
        )
        target_rows.append(row)

    targets = pd.DataFrame(target_rows).sort_values(["window_id", "interval_start", "interval_end", "period"])
    return pairs, targets


def read_particle_daily_sm(week_root, layer1_depth_mm):
    particles = read_particle_vic_cells(week_root, layer1_depth_mm)
    return (
        particles.groupby(["particle", "date"], as_index=False)
        .agg(particle_basin_SM_m3m3=("vic_layer1_m3m3", "mean"))
        .sort_values(["particle", "date"])
    )


def build_particle_interval_deltas(particle_daily, targets):
    intervals = targets[["interval_start", "interval_end"]].drop_duplicates().copy()
    rows = []
    for _, interval in intervals.iterrows():
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
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_reference_interval_deltas(targets, vic_daily_path, rainfall_path):
    daily = pd.read_csv(vic_daily_path, parse_dates=["time"]).set_index("time")
    rain = pd.read_csv(rainfall_path, parse_dates=["date"])
    rain_daily = rain.groupby("date", as_index=False)["precip_mm"].mean().set_index("date")
    rows = []
    intervals = targets[["interval_start", "interval_end"]].drop_duplicates().copy()
    for _, interval in intervals.iterrows():
        t0 = pd.Timestamp(interval["interval_start"])
        t1 = pd.Timestamp(interval["interval_end"])
        sm0 = daily.loc[t0, "basin_mean_vic_soil_moist_layer1_m3m3"]
        sm1 = daily.loc[t1, "basin_mean_vic_soil_moist_layer1_m3m3"]
        rows.append(
            {
                "interval_start": interval["interval_start"],
                "interval_end": interval["interval_end"],
                "basin0_open_loop_SM_t0": float(sm0),
                "basin0_open_loop_SM_t1": float(sm1),
                "basin0_open_loop_delta_SM": float(sm1 - sm0),
                "huc8_power_rainfall_sum_mm": float(
                    rain_daily.loc[(rain_daily.index > t0) & (rain_daily.index <= t1), "precip_mm"].sum()
                ),
                "basin0_vic_precip_sum_mm": float(
                    daily.loc[(daily.index > t0) & (daily.index <= t1), "basin_mean_out_prec"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def normalize_log_weights(log_values):
    log_values = np.asarray(log_values, dtype=float)
    max_log = np.nanmax(log_values)
    weights = np.exp(log_values - max_log)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return np.ones_like(log_values) / len(log_values)
    return weights / total


def summarize_window_irrigation(week_root, scores, window):
    irrigation = pd.read_csv(week_root / "particle_irrigation_inputs.csv")
    irrigation = irrigation[
        (irrigation["date"] >= window["window_start"]) & (irrigation["date"] <= window["window_end"])
    ].copy()
    weighted = irrigation.merge(scores[["particle", "weight"]], on="particle", how="left")
    weighted["weighted_irrigation_mm_day"] = weighted["irrigation_mm_day"] * weighted["weight"]
    daily = (
        weighted.groupby("date", as_index=False)
        .agg(
            posterior_mean_irrigation_mm=("weighted_irrigation_mm_day", "sum"),
            prior_mean_irrigation_mm=("irrigation_mm_day", "mean"),
            max_particle_irrigation_mm=("irrigation_mm_day", "max"),
        )
        .sort_values("date")
    )
    daily["window_id"] = window["window_id"]
    daily["window_start"] = window["window_start"]
    daily["window_end"] = window["window_end"]
    return daily


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


def score_sensitivity_run(args, windows, targets, particle_deltas, reference, sigma, target_mode):
    particle_ids = sorted(particle_deltas["particle"].unique())
    prior_weights = {particle: 1.0 / len(particle_ids) for particle in particle_ids}
    all_matches = []
    all_weights = []
    all_daily = []
    summary_rows = []

    target_ref = targets.merge(reference, on=["interval_start", "interval_end"], how="left")
    target_ref["obs_id"] = np.arange(len(target_ref))

    for window in windows:
        twin = target_ref[target_ref["window_id"].eq(window["window_id"])].copy()
        pwin = particle_deltas.merge(
            twin[
                [
                    "obs_id",
                    "window_id",
                    "window_start",
                    "window_end",
                    "interval_start",
                    "interval_end",
                    "period",
                    "cropland_delta_SM_satellite",
                    "control_delta_SM_satellite",
                    "crop_minus_control_delta_SM_satellite",
                    "cropland_n_pairs",
                    "control_n_pairs",
                    "basin0_open_loop_delta_SM",
                    "huc8_power_rainfall_sum_mm",
                ]
            ],
            on=["interval_start", "interval_end"],
            how="inner",
        )
        pwin = pwin[pwin["window_id"].eq(window["window_id"])].copy()

        residual_parts = residual_columns_for_mode(target_mode, pwin)
        pwin["log_likelihood_component_sum"] = 0.0
        pwin["n_residual_components"] = 0
        for name, residual in residual_parts:
            pwin[name] = residual
            valid = np.isfinite(residual)
            pwin.loc[valid, "log_likelihood_component_sum"] += -0.5 * (residual[valid] / sigma) ** 2
            pwin.loc[valid, "n_residual_components"] += 1

        matches = pwin[pwin["n_residual_components"] > 0].copy()
        grouped = (
            matches.groupby("particle", as_index=False)
            .agg(
                log_likelihood=("log_likelihood_component_sum", "sum"),
                n_observation_intervals=("obs_id", "nunique"),
                n_residual_components=("n_residual_components", "sum"),
                residual_rmse=("log_likelihood_component_sum", "size"),
            )
        )
        if matches.empty:
            grouped = pd.DataFrame({"particle": particle_ids, "log_likelihood": 0.0})
            grouped["n_observation_intervals"] = 0
            grouped["n_residual_components"] = 0

        residual_cols = [name for name, _ in residual_parts]
        if not matches.empty:
            rmse_by_particle = []
            for particle, group in matches.groupby("particle"):
                vals = []
                for col in residual_cols:
                    vals.extend(group[col].dropna().astype(float).tolist())
                rmse_by_particle.append(
                    {
                        "particle": particle,
                        "residual_rmse_delta_SM": float(np.sqrt(np.mean(np.asarray(vals) ** 2))) if vals else np.nan,
                    }
                )
            grouped = grouped.drop(columns=["residual_rmse"], errors="ignore").merge(
                pd.DataFrame(rmse_by_particle), on="particle", how="left"
            )
        else:
            grouped["residual_rmse_delta_SM"] = np.nan

        scores = pd.DataFrame({"particle": particle_ids}).merge(grouped, on="particle", how="left")
        scores["log_likelihood"] = scores["log_likelihood"].fillna(0.0)
        scores["n_observation_intervals"] = scores["n_observation_intervals"].fillna(0).astype(int)
        scores["n_residual_components"] = scores["n_residual_components"].fillna(0).astype(int)
        scores["prior_weight"] = scores["particle"].map(prior_weights).fillna(1.0 / len(particle_ids))
        scores["log_posterior_unnormalized"] = np.log(np.maximum(scores["prior_weight"], 1e-300)) + scores[
            "log_likelihood"
        ]
        scores["weight"] = normalize_log_weights(scores["log_posterior_unnormalized"].to_numpy())
        scores["rank"] = scores["weight"].rank(ascending=False, method="first").astype(int)
        scores = scores.sort_values("rank").copy()

        daily = summarize_window_irrigation(args.week_root, scores, window)
        ess = 1.0 / float(np.sum(scores["weight"] ** 2))
        max_weight = float(scores["weight"].max())
        best_particle = int(scores.iloc[0]["particle"])
        rmse = float(np.sqrt(np.nanmean(scores["residual_rmse_delta_SM"] ** 2)))
        n_intervals = int(twin[["interval_start", "interval_end", "period"]].drop_duplicates().shape[0])
        n_components = int(scores["n_residual_components"].sum() / max(1, len(particle_ids)))
        warnings = []
        if n_intervals == 0 or n_components == 0:
            warnings.append("zero usable daily delta observations")
        if ess < args.low_ess_threshold:
            warnings.append("low ESS")
        if daily["posterior_mean_irrigation_mm"].fillna(0).abs().max() < 1e-3:
            warnings.append("posterior irrigation near zero")

        scores["sigma_m3m3"] = sigma
        scores["target_mode"] = target_mode
        scores["window_id"] = window["window_id"]
        scores["window_start"] = window["window_start"]
        scores["window_end"] = window["window_end"]
        daily["sigma_m3m3"] = sigma
        daily["target_mode"] = target_mode
        if not matches.empty:
            matches["sigma_m3m3"] = sigma
            matches["target_mode"] = target_mode
            all_matches.append(matches)
        all_weights.append(scores)
        all_daily.append(daily)
        summary_rows.append(
            {
                "sigma_m3m3": sigma,
                "target_mode": target_mode,
                "window_id": window["window_id"],
                "window_start": window["window_start"],
                "window_end": window["window_end"],
                "n_daily_observation_intervals": n_intervals,
                "mean_residual_components_per_particle": n_components,
                "effective_sample_size": ess,
                "max_particle_weight": max_weight,
                "best_particle": best_particle,
                "residual_rmse_delta_SM": rmse,
                "posterior_irrigation_sum_mm": float(daily["posterior_mean_irrigation_mm"].sum()),
                "prior_irrigation_sum_mm": float(daily["prior_mean_irrigation_mm"].sum()),
                "warnings": "; ".join(warnings),
            }
        )
        prior_weights = dict(zip(scores["particle"], scores["weight"]))

    matches = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()
    weights = pd.concat(all_weights, ignore_index=True)
    daily = pd.concat(all_daily, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    return matches, weights, daily, summary


def plot_sensitivity(out_dir, summary, daily):
    totals = (
        summary.groupby(["target_mode", "sigma_m3m3"], as_index=False)
        .agg(
            posterior_total_mm=("posterior_irrigation_sum_mm", "sum"),
            prior_total_mm=("prior_irrigation_sum_mm", "sum"),
            min_ess=("effective_sample_size", "min"),
            median_ess=("effective_sample_size", "median"),
        )
        .sort_values(["target_mode", "sigma_m3m3"])
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    for mode, group in totals.groupby("target_mode"):
        ax.plot(group["sigma_m3m3"], group["posterior_total_mm"], marker="o", linewidth=1.8, label=mode)
    prior = float(totals["prior_total_mm"].max()) if len(totals) else np.nan
    if np.isfinite(prior):
        ax.axhline(prior, color="#6B7280", linestyle="--", linewidth=1.0, label="prior total")
    ax.set_xlabel("Delta-SM likelihood sigma (m3/m3)")
    ax.set_ylabel("4-week posterior irrigation (mm)")
    ax.set_title("Daily delta-SM PBS sensitivity: posterior irrigation", pad=42)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.savefig(out_dir / "daily_delta_sm_posterior_total_by_sigma_target.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    for mode, group in totals.groupby("target_mode"):
        ax.plot(group["sigma_m3m3"], group["min_ess"], marker="o", linewidth=1.8, label=mode)
    ax.set_xlabel("Delta-SM likelihood sigma (m3/m3)")
    ax.set_ylabel("Minimum weekly ESS")
    ax.set_title("Daily delta-SM PBS sensitivity: minimum ESS", pad=42)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.savefig(out_dir / "daily_delta_sm_min_ess_by_sigma_target.png", dpi=300)
    plt.close(fig)

    default = daily[(daily["sigma_m3m3"].eq(0.035))].copy()
    if default.empty and not daily.empty:
        default_sigma = sorted(daily["sigma_m3m3"].unique())[0]
        default = daily[daily["sigma_m3m3"].eq(default_sigma)].copy()
    if not default.empty:
        fig, ax = plt.subplots(figsize=(13.0, 5.2), constrained_layout=True)
        for mode, group in default.groupby("target_mode"):
            plot = group.groupby("date", as_index=False)["posterior_mean_irrigation_mm"].sum()
            ax.plot(pd.to_datetime(plot["date"]), plot["posterior_mean_irrigation_mm"], marker="o", label=mode)
        prior_daily = default.groupby("date", as_index=False)["prior_mean_irrigation_mm"].first()
        ax.plot(
            pd.to_datetime(prior_daily["date"]),
            prior_daily["prior_mean_irrigation_mm"],
            color="#6B7280",
            linestyle="--",
            label="prior mean",
        )
        ax.set_ylabel("Irrigation (mm/day)")
        ax.set_title("Daily delta-SM posterior daily irrigation at sigma 0.035", pad=42)
        ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02))
        fig.savefig(out_dir / "daily_delta_sm_posterior_daily_irrigation_sigma_0035.png", dpi=300)
        plt.close(fig)


def write_report(out_dir, metadata, targets, summary):
    totals = (
        summary.groupby(["target_mode", "sigma_m3m3"], as_index=False)
        .agg(
            posterior_total_mm=("posterior_irrigation_sum_mm", "sum"),
            prior_total_mm=("prior_irrigation_sum_mm", "sum"),
            min_ess=("effective_sample_size", "min"),
            median_ess=("effective_sample_size", "median"),
            max_weight=("max_particle_weight", "max"),
        )
        .sort_values(["target_mode", "sigma_m3m3"])
    )
    lines = [
        "# Daily HUC8 SMAP L3 Delta-SM PBS Sensitivity",
        "",
        "This run uses consecutive high-quality SMAP L3 pairs within each PBS window rather than only weekly endpoint pairs.",
        "",
        "## Configuration",
        "",
        f"- Maximum same-cell/same-period pair gap: {metadata['max_pair_gap_days']} days.",
        f"- Sigmas tested: {', '.join(str(x) for x in metadata['sigmas_m3m3'])} m3/m3.",
        f"- Target modes tested: {', '.join(metadata['target_modes'])}.",
        f"- Daily interval targets: {len(targets)} aggregate interval/period rows.",
        "",
        "Target definitions:",
        "",
        "- `cropland_only`: compare cropland satellite delta SM to particle basin delta SM.",
        "- `crop_minus_control`: compare cropland-minus-control satellite delta SM to particle-minus-open-loop VIC delta SM.",
        "- `joint_crop_control`: combine the cropland-only residual and the crop-minus-control anomaly residual.",
        "",
        "## Sensitivity Summary",
        "",
        "| Target mode | Sigma | Posterior total mm | Prior total mm | Min ESS | Median ESS | Max weight |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in totals.iterrows():
        lines.append(
            f"| {row.target_mode} | {row.sigma_m3m3:.3f} | {row.posterior_total_mm:.4f} | "
            f"{row.prior_total_mm:.4f} | {row.min_ess:.2f} | {row.median_ess:.2f} | {row.max_weight:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Plots",
            "",
            "![Posterior total by sigma and target](./daily_delta_sm_posterior_total_by_sigma_target.png)",
            "",
            "![Minimum ESS by sigma and target](./daily_delta_sm_min_ess_by_sigma_target.png)",
            "",
            "![Posterior daily irrigation at sigma 0.035](./daily_delta_sm_posterior_daily_irrigation_sigma_0035.png)",
            "",
            "## Output Tables",
            "",
            "- `daily_delta_cell_pairs.csv`",
            "- `daily_delta_targets_by_interval.csv`",
            "- `daily_delta_targets_for_hopper.csv`",
            "- `particle_interval_delta_sm.csv`",
            "- `reference_interval_delta_sm.csv`",
            "- `sensitivity_summary.csv`",
            "- `window_summary_by_run.csv`",
            "- `posterior_daily_irrigation_by_run.csv`",
            "- `particle_weights_by_run.csv`",
        ]
    )
    (out_dir / "summary_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    totals.to_csv(out_dir / "sensitivity_summary.csv", index=False)


def main():
    args = parse_args()
    args.week_root = args.week_root.resolve()
    if not args.week_root.exists():
        raise FileNotFoundError(args.week_root)

    sigmas = parse_float_list(args.sigmas)
    target_modes = parse_mode_list(args.target_modes)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / f"huc8_smap_l3_daily_delta_sm_sensitivity_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    particle_dates_probe = pd.read_csv(args.week_root / "particle_irrigation_inputs.csv")
    start_date = particle_dates_probe["date"].min()
    end_date = particle_dates_probe["date"].max()
    windows = weekly_windows(start_date, end_date, args.window_days)

    cell_pairs, targets = build_daily_satellite_delta_targets(args, windows, start_date, end_date)
    if targets.empty:
        raise RuntimeError("No daily delta-SM targets could be built.")
    particle_daily = read_particle_daily_sm(args.week_root, args.layer1_depth_mm)
    particle_deltas = build_particle_interval_deltas(particle_daily, targets)
    reference = build_reference_interval_deltas(targets, args.vic_basin_daily, args.rainfall_csv)

    all_matches = []
    all_weights = []
    all_daily = []
    all_summary = []
    for mode in target_modes:
        for sigma in sigmas:
            matches, weights, daily, summary = score_sensitivity_run(
                args, windows, targets, particle_deltas, reference, sigma, mode
            )
            if not matches.empty:
                all_matches.append(matches)
            all_weights.append(weights)
            all_daily.append(daily)
            all_summary.append(summary)

    matches_all = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()
    weights_all = pd.concat(all_weights, ignore_index=True)
    daily_all = pd.concat(all_daily, ignore_index=True)
    summary_all = pd.concat(all_summary, ignore_index=True)

    cell_pairs.to_csv(out_dir / "daily_delta_cell_pairs.csv", index=False)
    targets.to_csv(out_dir / "daily_delta_targets_by_interval.csv", index=False)
    particle_daily.to_csv(out_dir / "particle_daily_basin_mean_sm.csv", index=False)
    particle_deltas.to_csv(out_dir / "particle_interval_delta_sm.csv", index=False)
    reference.to_csv(out_dir / "reference_interval_delta_sm.csv", index=False)
    targets.merge(reference, on=["interval_start", "interval_end"], how="left").to_csv(
        out_dir / "daily_delta_targets_for_hopper.csv", index=False
    )
    matches_all.to_csv(out_dir / "particle_interval_matches_by_run.csv", index=False)
    weights_all.to_csv(out_dir / "particle_weights_by_run.csv", index=False)
    daily_all.to_csv(out_dir / "posterior_daily_irrigation_by_run.csv", index=False)
    summary_all.to_csv(out_dir / "window_summary_by_run.csv", index=False)

    metadata = {
        "run_date_time": datetime.now().isoformat(timespec="seconds"),
        "run_type": "huc8_smap_l3_daily_delta_sm_sensitivity",
        "week_root": str(args.week_root),
        "endpoint_observations": str(args.endpoint_observations.resolve()),
        "cdl_summary": str(args.cdl_summary.resolve()),
        "vic_basin_daily": str(args.vic_basin_daily.resolve()),
        "rainfall_csv": str(args.rainfall_csv.resolve()),
        "quality_mode": args.quality_mode,
        "quality_flag_high_quality_values": [0, 8],
        "date_range": {"start": start_date, "end": end_date},
        "window_days": args.window_days,
        "max_pair_gap_days": args.max_pair_gap_days,
        "sigmas_m3m3": sigmas,
        "target_modes": target_modes,
        "weight_mask_class": args.weight_mask_class,
        "control_mask_class": args.control_mask_class,
        "particle_count": int(weights_all["particle"].nunique()),
        "daily_target_rows": int(len(targets)),
        "cell_pair_rows": int(len(cell_pairs)),
        "code_commit_hash": git_commit_hash(),
    }
    (out_dir / "pbs_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    plot_sensitivity(out_dir, summary_all, daily_all)
    write_report(out_dir, metadata, targets, summary_all)
    print(out_dir)


if __name__ == "__main__":
    main()
