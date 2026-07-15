#!/usr/bin/env python3
"""Run a HUC8 SMAP-L3 delta-SM PBS comparison.

The posterior irrigation estimate still comes from the existing particle
irrigation inputs. Only the particle likelihood uses weekly delta soil moisture.
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
import matplotlib.dates as mdates
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run HUC8 SMAP-L3 delta-SM PBS comparison.")
    parser.add_argument("--week-root", type=Path, default=DEFAULT_WEEK_ROOT)
    parser.add_argument("--endpoint-observations", type=Path, default=DEFAULT_ENDPOINTS)
    parser.add_argument("--cdl-summary", type=Path, default=DEFAULT_CDL)
    parser.add_argument("--vic-basin-daily", type=Path, default=DEFAULT_VIC_DAILY)
    parser.add_argument("--rainfall-csv", type=Path, default=DEFAULT_RAINFALL)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--cdl-year", type=int, default=2020)
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--search-half-window-days", type=int, default=2)
    parser.add_argument("--layer1-depth-mm", type=float, default=50.0)
    parser.add_argument(
        "--delta-sigma-m3m3",
        type=float,
        default=0.035,
        help=(
            "Working-assumption Gaussian likelihood scale for weekly delta SM. "
            "This first-pass value was not independently calibrated for delta-SM."
        ),
    )
    parser.add_argument("--weight-mask-class", default="cropland_like")
    parser.add_argument("--control-mask-class", default="noncropland_control")
    parser.add_argument("--quality-mode", choices=["high_quality", "all_finite"], default="high_quality")
    parser.add_argument("--low-ess-threshold", type=float, default=3.0)
    parser.add_argument("--low-target-pair-threshold", type=int, default=10)
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


def safe_float(value):
    try:
        value = float(value)
    except Exception:
        return None
    return value if math.isfinite(value) else None


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


def nearest_endpoint(group, target, max_offset_days, endpoint_role):
    candidates = group.copy()
    candidates["offset_days"] = (candidates["date"] - target).dt.days
    candidates["abs_offset_days"] = candidates["offset_days"].abs()
    candidates = candidates[candidates["abs_offset_days"] <= max_offset_days].copy()
    if candidates.empty:
        return None
    if endpoint_role == "start":
        candidates["direction_penalty"] = (candidates["offset_days"] < 0).astype(int)
    else:
        candidates["direction_penalty"] = (candidates["offset_days"] > 0).astype(int)
    return candidates.sort_values(["abs_offset_days", "direction_penalty", "offset_days"]).iloc[0]


def build_satellite_delta_pairs(args, windows):
    obs = pd.read_csv(args.endpoint_observations, parse_dates=["date"])
    obs["retrieval_recommended_bool"] = bool_series(obs["retrieval_recommended"])
    obs = obs[obs["soil_moisture"].notna()].copy()
    if args.quality_mode == "high_quality":
        obs = obs[obs["retrieval_recommended_bool"]].copy()

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
    for col in ["crop_fraction", "noncropland_fraction", "excluded_fraction"]:
        obs[col] = obs[col].fillna(np.nan)

    pair_rows = []
    for window in windows:
        target_start = pd.Timestamp(window["window_start"])
        target_end = pd.Timestamp(window["window_end"])
        for (row, col, period), group in obs.groupby(["row", "col", "period"]):
            start = nearest_endpoint(group, target_start, args.search_half_window_days, "start")
            end = nearest_endpoint(group, target_end, args.search_half_window_days, "end")
            if start is None or end is None:
                continue
            if pd.Timestamp(end["date"]) <= pd.Timestamp(start["date"]):
                continue
            pair_rows.append(
                {
                    "window_id": window["window_id"],
                    "window_start": window["window_start"],
                    "window_end": window["window_end"],
                    "row": int(row),
                    "col": int(col),
                    "smap_cell_id": start["smap_cell_id"],
                    "period": period,
                    "mask_class": start["mask_class"],
                    "crop_fraction": start["crop_fraction"],
                    "noncropland_fraction": start["noncropland_fraction"],
                    "excluded_fraction": start["excluded_fraction"],
                    "overlap_area_km2": float(start["overlap_area_km2"]),
                    "target_t0": target_start.date().isoformat(),
                    "target_t1": target_end.date().isoformat(),
                    "sat_t0": pd.Timestamp(start["date"]).date().isoformat(),
                    "sat_t1": pd.Timestamp(end["date"]).date().isoformat(),
                    "sat_t0_offset_days": int(start["offset_days"]),
                    "sat_t1_offset_days": int(end["offset_days"]),
                    "SMAP_SM_t0": float(start["soil_moisture"]),
                    "SMAP_SM_t1": float(end["soil_moisture"]),
                    "delta_SM_satellite": float(end["soil_moisture"] - start["soil_moisture"]),
                    "retrieval_qual_flag_t0": safe_float(start["retrieval_qual_flag"]),
                    "retrieval_qual_flag_t1": safe_float(end["retrieval_qual_flag"]),
                    "soil_moisture_error_t0": safe_float(start["soil_moisture_error"]),
                    "soil_moisture_error_t1": safe_float(end["soil_moisture_error"]),
                    "source_file_t0": start["source_file"],
                    "source_file_t1": end["source_file"],
                }
            )

    all_pairs = pd.DataFrame(pair_rows)
    if all_pairs.empty:
        return all_pairs, pd.DataFrame(), all_pairs

    all_pairs["total_abs_offset_days"] = all_pairs["sat_t0_offset_days"].abs() + all_pairs["sat_t1_offset_days"].abs()
    all_pairs["period_rank"] = all_pairs["period"].map({"AM": 0, "PM": 1}).fillna(2)
    selected = (
        all_pairs.sort_values(["window_id", "row", "col", "total_abs_offset_days", "period_rank"])
        .drop_duplicates(["window_id", "row", "col"], keep="first")
        .drop(columns=["period_rank"])
        .reset_index(drop=True)
    )

    agg_rows = []
    for (window_id, mask_class), group in selected.groupby(["window_id", "mask_class"]):
        agg_rows.append(
            {
                "window_id": int(window_id),
                "mask_class": mask_class,
                "n_pairs": int(len(group)),
                "area_km2": float(group["overlap_area_km2"].sum()),
                "delta_SM_satellite_area_weighted": weighted_mean(
                    group["delta_SM_satellite"], group["overlap_area_km2"]
                ),
                "delta_SM_satellite_mean": float(group["delta_SM_satellite"].mean()),
                "delta_SM_satellite_median": float(group["delta_SM_satellite"].median()),
                "delta_SM_satellite_sd": float(group["delta_SM_satellite"].std(ddof=1))
                if len(group) > 1
                else np.nan,
                "mean_abs_t0_offset_days": float(group["sat_t0_offset_days"].abs().mean()),
                "mean_abs_t1_offset_days": float(group["sat_t1_offset_days"].abs().mean()),
                "max_abs_endpoint_offset_days": int(
                    max(group["sat_t0_offset_days"].abs().max(), group["sat_t1_offset_days"].abs().max())
                ),
                "periods_used": ";".join(sorted(group["period"].unique())),
                "sat_t0_dates": ";".join(sorted(group["sat_t0"].unique())),
                "sat_t1_dates": ";".join(sorted(group["sat_t1"].unique())),
            }
        )
    aggregates = pd.DataFrame(agg_rows)
    win_frame = pd.DataFrame(windows)
    aggregates = win_frame.merge(aggregates, on="window_id", how="left")
    return all_pairs, selected, aggregates


def read_reference_deltas(windows, vic_daily_path, rainfall_path):
    daily = pd.read_csv(vic_daily_path, parse_dates=["time"]).set_index("time")
    rain = pd.read_csv(rainfall_path, parse_dates=["date"])
    rain_daily = rain.groupby("date", as_index=False)["precip_mm"].mean().set_index("date")
    rows = []
    for window in windows:
        t0 = pd.Timestamp(window["window_start"])
        t1 = pd.Timestamp(window["window_end"])
        sm0 = daily.loc[t0, "basin_mean_vic_soil_moist_layer1_m3m3"]
        sm1 = daily.loc[t1, "basin_mean_vic_soil_moist_layer1_m3m3"]
        precip = daily.loc[(daily.index > t0) & (daily.index <= t1), "basin_mean_out_prec"].sum()
        huc8_rain = rain_daily.loc[(rain_daily.index > t0) & (rain_daily.index <= t1), "precip_mm"].sum()
        rows.append(
            {
                "window_id": window["window_id"],
                "window_start": window["window_start"],
                "window_end": window["window_end"],
                "basin0_open_loop_SM_t0": float(sm0),
                "basin0_open_loop_SM_t1": float(sm1),
                "basin0_open_loop_delta_SM": float(sm1 - sm0),
                "basin0_vic_precip_sum_mm": float(precip),
                "huc8_power_rainfall_sum_mm": float(huc8_rain),
            }
        )
    return pd.DataFrame(rows)


def read_particle_delta_sm(week_root, windows, layer1_depth_mm):
    particles = read_particle_vic_cells(week_root, layer1_depth_mm)
    daily = (
        particles.groupby(["particle", "date"], as_index=False)
        .agg(particle_basin_SM_m3m3=("vic_layer1_m3m3", "mean"))
        .sort_values(["particle", "date"])
    )
    rows = []
    for window in windows:
        start = daily[daily["date"].eq(window["window_start"])][["particle", "particle_basin_SM_m3m3"]]
        end = daily[daily["date"].eq(window["window_end"])][["particle", "particle_basin_SM_m3m3"]]
        merged = start.merge(end, on="particle", suffixes=("_t0", "_t1"))
        merged["window_id"] = window["window_id"]
        merged["window_start"] = window["window_start"]
        merged["window_end"] = window["window_end"]
        merged["particle_delta_SM"] = merged["particle_basin_SM_m3m3_t1"] - merged["particle_basin_SM_m3m3_t0"]
        rows.append(merged)
    return pd.concat(rows, ignore_index=True), daily


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


def score_delta_pbs(args, windows, particle_deltas, satellite_aggregates, reference):
    particle_ids = sorted(particle_deltas["particle"].unique())
    prior_weights = {particle: 1.0 / len(particle_ids) for particle in particle_ids}
    all_matches = []
    all_weights = []
    all_daily = []
    summary_rows = []
    warnings = []

    for window in windows:
        target_row = satellite_aggregates[
            (satellite_aggregates["window_id"].eq(window["window_id"]))
            & (satellite_aggregates["mask_class"].eq(args.weight_mask_class))
        ]
        target_available = not target_row.empty and np.isfinite(
            target_row.iloc[0]["delta_SM_satellite_area_weighted"]
        )
        target_delta = float(target_row.iloc[0]["delta_SM_satellite_area_weighted"]) if target_available else np.nan
        target_pairs = int(target_row.iloc[0]["n_pairs"]) if target_available else 0

        pwin = particle_deltas[particle_deltas["window_id"].eq(window["window_id"])].copy()
        pwin["prior_weight"] = pwin["particle"].map(prior_weights).fillna(1.0 / len(particle_ids))
        if target_available:
            pwin["target_delta_SM_satellite"] = target_delta
            pwin["residual_delta_SM"] = target_delta - pwin["particle_delta_SM"]
            pwin["log_likelihood"] = -0.5 * (pwin["residual_delta_SM"] / args.delta_sigma_m3m3) ** 2
        else:
            pwin["target_delta_SM_satellite"] = np.nan
            pwin["residual_delta_SM"] = np.nan
            pwin["log_likelihood"] = 0.0

        pwin["log_posterior_unnormalized"] = np.log(np.maximum(pwin["prior_weight"], 1e-300)) + pwin[
            "log_likelihood"
        ]
        pwin["weight"] = normalize_log_weights(pwin["log_posterior_unnormalized"].to_numpy())
        pwin["rank"] = pwin["weight"].rank(ascending=False, method="first").astype(int)
        scores = pwin.sort_values("rank").copy()
        daily = summarize_window_irrigation(args.week_root, scores, window)

        ref = reference[reference["window_id"].eq(window["window_id"])].iloc[0]
        crop = target_row.iloc[0] if target_available else None
        control_rows = satellite_aggregates[
            (satellite_aggregates["window_id"].eq(window["window_id"]))
            & (satellite_aggregates["mask_class"].eq(args.control_mask_class))
        ]
        control = control_rows.iloc[0] if not control_rows.empty else None
        control_delta = safe_float(control["delta_SM_satellite_area_weighted"]) if control is not None else None
        control_pairs = int(control["n_pairs"]) if control is not None and np.isfinite(control["n_pairs"]) else 0

        ess = 1.0 / float(np.sum(scores["weight"] ** 2))
        max_weight = float(scores["weight"].max())
        best_particle = int(scores.iloc[0]["particle"])
        rmse = float(np.sqrt(np.nanmean(scores["residual_delta_SM"] ** 2))) if target_available else np.nan
        window_warnings = []
        if not target_available:
            window_warnings.append("missing cropland satellite delta target")
        if target_pairs < args.low_target_pair_threshold:
            window_warnings.append(f"low cropland target pair count ({target_pairs})")
        if ess < args.low_ess_threshold:
            window_warnings.append("low ESS")
        if daily["posterior_mean_irrigation_mm"].fillna(0).abs().max() < 1e-3:
            window_warnings.append("posterior irrigation near zero")

        all_matches.append(
            scores[
                [
                    "window_id",
                    "window_start",
                    "window_end",
                    "particle",
                    "particle_basin_SM_m3m3_t0",
                    "particle_basin_SM_m3m3_t1",
                    "particle_delta_SM",
                    "target_delta_SM_satellite",
                    "residual_delta_SM",
                    "log_likelihood",
                ]
            ].copy()
        )
        all_weights.append(scores)
        all_daily.append(daily)
        summary_rows.append(
            {
                "window_id": window["window_id"],
                "window_start": window["window_start"],
                "window_end": window["window_end"],
                "weight_mask_class": args.weight_mask_class,
                "cropland_delta_SM_satellite": target_delta,
                "cropland_n_pairs": target_pairs,
                "control_delta_SM_satellite": control_delta,
                "control_n_pairs": control_pairs,
                "crop_minus_control_delta_SM": target_delta - control_delta
                if target_available and control_delta is not None
                else np.nan,
                "basin0_open_loop_delta_SM": float(ref["basin0_open_loop_delta_SM"]),
                "huc8_power_rainfall_sum_mm": float(ref["huc8_power_rainfall_sum_mm"]),
                "basin0_vic_precip_sum_mm": float(ref["basin0_vic_precip_sum_mm"]),
                "effective_sample_size": ess,
                "max_particle_weight": max_weight,
                "best_particle": best_particle,
                "residual_rmse_delta_SM": rmse,
                "posterior_irrigation_sum_mm": float(daily["posterior_mean_irrigation_mm"].sum()),
                "prior_irrigation_sum_mm": float(daily["prior_mean_irrigation_mm"].sum()),
                "warnings": "; ".join(window_warnings),
            }
        )
        warnings.extend([f"Window {window['window_id']}: {warning}" for warning in window_warnings])
        prior_weights = dict(zip(scores["particle"], scores["weight"]))

    return (
        pd.concat(all_matches, ignore_index=True),
        pd.concat(all_weights, ignore_index=True),
        pd.concat(all_daily, ignore_index=True),
        pd.DataFrame(summary_rows),
        sorted(set(warnings)),
    )


def plot_outputs(out_dir, selected_pairs, satellite_aggregates, reference, weights, daily, window_summary):
    fig, ax1 = plt.subplots(figsize=(12.8, 5.2), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.14, top=0.72)
    x = pd.to_datetime(window_summary["window_end"])
    ax1.axhline(0, color="#525252", linewidth=0.9)
    ax1.plot(x, window_summary["cropland_delta_SM_satellite"], marker="o", color="#365C8D", label="Cropland SMAP delta")
    ax1.plot(x, window_summary["control_delta_SM_satellite"], marker="o", color="#1F9E89", label="Control SMAP delta")
    ax1.plot(x, window_summary["basin0_open_loop_delta_SM"], marker="s", color="#5C6773", label="basin0 VIC open-loop delta")
    ax1.set_ylabel("Delta SM (m3/m3)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.bar(
        x,
        window_summary["basin0_vic_precip_sum_mm"],
        width=2.8,
        alpha=0.23,
        color="#B58D1D",
        label="basin0 VIC forcing precip",
    )
    ax2.set_ylabel("Precipitation (mm/week)")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.suptitle("HUC8 weekly delta SM comparison used for PBS weighting", y=0.98)
    fig.legend(
        lines + lines2,
        labels + labels2,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
    )
    fig.savefig(out_dir / "huc8_satellite_vic_rainfall_delta_comparison.png", dpi=300)
    plt.close(fig)

    d = daily.copy()
    d["date_dt"] = pd.to_datetime(d["date"])
    fig, ax = plt.subplots(figsize=(13.0, 5.0), constrained_layout=True)
    ax.bar(d["date_dt"], d["posterior_mean_irrigation_mm"], width=0.75, color="#365C8D", label="Posterior mean")
    ax.plot(d["date_dt"], d["prior_mean_irrigation_mm"], marker="o", color="#5C6773", label="Prior mean")
    ax.set_title("Posterior daily irrigation with delta-SM particle weighting", pad=38)
    ax.set_ylabel("Irrigation (mm/day)")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.savefig(out_dir / "posterior_vs_prior_daily_irrigation_delta_sm_pbs.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.2), constrained_layout=True)
    ax.bar(window_summary["window_id"].astype(str), window_summary["effective_sample_size"], color="#365C8D")
    ax.set_title("Effective sample size by delta-SM window")
    ax.set_xlabel("Window")
    ax.set_ylabel("ESS")
    fig.savefig(out_dir / "effective_sample_size_by_window.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.2, 5.0), constrained_layout=True)
    for _, sub in weights.groupby("particle"):
        ax.plot(sub["window_id"], sub["weight"], color="#C7CDD6", linewidth=0.7, alpha=0.28)
    top_particles = (
        weights.groupby("particle")["weight"].max().sort_values(ascending=False).head(10).index.tolist()
    )
    for particle in top_particles:
        sub = weights[weights["particle"].eq(particle)].sort_values("window_id")
        ax.plot(sub["window_id"], sub["weight"], marker="o", linewidth=1.6, label=f"p{int(particle)}")
    ax.set_title("Particle weights by weekly delta-SM window", pad=38)
    ax.set_xlabel("Window")
    ax.set_ylabel("Posterior weight")
    ax.legend(frameon=False, ncol=5, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.savefig(out_dir / "particle_weights_by_window.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.2, 4.8), constrained_layout=True)
    for mask_class, group in selected_pairs.groupby("mask_class"):
        plot_group = group.groupby("window_id", as_index=False).agg(n_pairs=("delta_SM_satellite", "size"))
        ax.plot(plot_group["window_id"], plot_group["n_pairs"], marker="o", label=mask_class)
    ax.set_title("Selected SMAP L3 endpoint-pair counts by mask", pad=38)
    ax.set_xlabel("Window")
    ax.set_ylabel("Cell pairs")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.savefig(out_dir / "satellite_pair_counts_by_mask.png", dpi=300)
    plt.close(fig)


def write_report(out_dir, metadata, window_summary):
    lines = [
        "# HUC8 SMAP-L3 Delta-SM PBS Comparison",
        "",
        "This run keeps the irrigation estimate on the existing particle irrigation inputs, while using weekly delta soil moisture for particle weighting.",
        "",
        "## Configuration",
        "",
        "- Satellite product: SMAP L3 enhanced passive soil moisture (`SPL3SMP_E`).",
        f"- Quality mode: {metadata['quality_mode']} (`retrieval_qual_flag` 0/8 when high-quality mode is used; NSIDC includes 8 because failed freeze/thaw retrieval does not affect soil-moisture retrieval).",
        f"- Endpoint search: nearest same-period observation within +/-{metadata['search_half_window_days']} days.",
        f"- Weighting target: area-weighted `{metadata['weight_mask_class']}` HUC8 satellite delta SM.",
        f"- Delta-SM likelihood sigma: {metadata['delta_sigma_m3m3']:.3f} m3/m3, treated as a working assumption rather than a calibrated error model.",
        "- Irrigation output: posterior weighted mean of `particle_irrigation_inputs.csv`.",
        "",
        "## Window Diagnostics",
        "",
        "| Window | Dates | Crop dSM | Crop n | Control dSM | Control n | VIC dSM | VIC P mm | ESS | Best | Posterior irr mm | Warnings |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in window_summary.iterrows():
        warnings = row.get("warnings", "")
        lines.append(
            f"| {int(row.window_id)} | {row.window_start} to {row.window_end} | "
            f"{row.cropland_delta_SM_satellite:.4f} | {int(row.cropland_n_pairs)} | "
            f"{row.control_delta_SM_satellite:.4f} | {int(row.control_n_pairs)} | "
            f"{row.basin0_open_loop_delta_SM:.4f} | {row.basin0_vic_precip_sum_mm:.2f} | "
            f"{row.effective_sample_size:.2f} | {int(row.best_particle)} | "
            f"{row.posterior_irrigation_sum_mm:.4f} | {warnings} |"
        )

    lines.extend(
        [
            "",
            "## Irrigation Result",
            "",
            f"- Total posterior mean irrigation: {metadata['total_posterior_irrigation_mm']:.4f} mm",
            f"- Total prior mean irrigation: {metadata['total_prior_irrigation_mm']:.4f} mm",
            f"- Posterior irrigation near zero for all days: {metadata['posterior_irrigation_near_zero_all_days']}",
            "",
            "## Warnings",
            "",
        ]
    )
    if metadata["warnings"]:
        lines.extend([f"- {warning}" for warning in metadata["warnings"]])
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `huc8_smap_l3_weekly_delta_all_period_pairs.csv`",
            "- `huc8_smap_l3_weekly_delta_selected_pairs.csv`",
            "- `huc8_smap_l3_weekly_delta_by_mask.csv`",
            "- `pbs_particle_delta_sm_matches.csv`",
            "- `pbs_particle_weights.csv`",
            "- `posterior_daily_irrigation.csv`",
            "- `window_summary.csv`",
            "- `pbs_run_metadata.json`",
        ]
    )
    (out_dir / "summary_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    args.week_root = args.week_root.resolve()
    if not args.week_root.exists():
        raise FileNotFoundError(args.week_root)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / f"huc8_smap_l3_delta_sm_pbs_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    particle_dates_probe = pd.read_csv(args.week_root / "particle_irrigation_inputs.csv")
    start_date = particle_dates_probe["date"].min()
    end_date = particle_dates_probe["date"].max()
    windows = weekly_windows(start_date, end_date, args.window_days)

    all_pairs, selected_pairs, satellite_aggregates = build_satellite_delta_pairs(args, windows)
    reference = read_reference_deltas(windows, args.vic_basin_daily, args.rainfall_csv)
    particle_deltas, particle_daily_sm = read_particle_delta_sm(args.week_root, windows, args.layer1_depth_mm)
    matches, weights, daily, window_summary, warnings = score_delta_pbs(
        args, windows, particle_deltas, satellite_aggregates, reference
    )

    all_pairs.to_csv(out_dir / "huc8_smap_l3_weekly_delta_all_period_pairs.csv", index=False)
    selected_pairs.to_csv(out_dir / "huc8_smap_l3_weekly_delta_selected_pairs.csv", index=False)
    satellite_aggregates.to_csv(out_dir / "huc8_smap_l3_weekly_delta_by_mask.csv", index=False)
    reference.to_csv(out_dir / "basin0_open_loop_vic_and_rainfall_weekly_reference.csv", index=False)
    particle_daily_sm.to_csv(out_dir / "particle_daily_basin_mean_sm.csv", index=False)
    particle_deltas.to_csv(out_dir / "particle_weekly_delta_sm.csv", index=False)
    matches.to_csv(out_dir / "pbs_particle_delta_sm_matches.csv", index=False)
    weights.to_csv(out_dir / "pbs_particle_weights.csv", index=False)
    daily.to_csv(out_dir / "posterior_daily_irrigation.csv", index=False)
    window_summary.to_csv(out_dir / "window_summary.csv", index=False)

    plot_outputs(out_dir, selected_pairs, satellite_aggregates, reference, weights, daily, window_summary)

    metadata = {
        "run_date_time": datetime.now().isoformat(timespec="seconds"),
        "run_type": "huc8_smap_l3_delta_sm_pbs_comparison",
        "week_root": str(args.week_root),
        "endpoint_observations": str(args.endpoint_observations.resolve()),
        "cdl_summary": str(args.cdl_summary.resolve()),
        "cdl_year": args.cdl_year,
        "vic_basin_daily": str(args.vic_basin_daily.resolve()),
        "rainfall_csv": str(args.rainfall_csv.resolve()),
        "satellite_product": "SPL3SMP_E",
        "quality_mode": args.quality_mode,
        "quality_flag_high_quality_values": [0, 8],
        "search_half_window_days": args.search_half_window_days,
        "weight_mask_class": args.weight_mask_class,
        "control_mask_class": args.control_mask_class,
        "delta_sigma_m3m3": args.delta_sigma_m3m3,
        "delta_sigma_note": (
            "Working assumption for first-pass delta-SM PBS weighting; not independently "
            "calibrated for weekly delta-SM residuals."
        ),
        "window_days": args.window_days,
        "date_range": {"start": start_date, "end": end_date},
        "window_count": len(windows),
        "particle_count": int(weights["particle"].nunique()),
        "assimilation_variable_for_weighting": "weekly_delta_soil_moisture",
        "irrigation_estimate_source": "particle_irrigation_inputs.csv",
        "total_posterior_irrigation_mm": float(daily["posterior_mean_irrigation_mm"].sum()),
        "total_prior_irrigation_mm": float(daily["prior_mean_irrigation_mm"].sum()),
        "posterior_irrigation_near_zero_all_days": bool(
            daily["posterior_mean_irrigation_mm"].fillna(0).abs().max() < 1e-3
        ),
        "warnings": warnings,
        "code_commit_hash": git_commit_hash(),
        "notes": [
            "Satellite deltas are HUC8 CDL mask aggregates.",
            "Particle deltas are basin-mean surface-layer VIC deltas from the existing N100 ensemble.",
            "Sequential reweighting is postprocessing only; VIC was not rerun between windows.",
        ],
    }
    (out_dir / "pbs_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(out_dir, metadata, window_summary)
    print(out_dir)


if __name__ == "__main__":
    main()
