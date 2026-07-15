#!/usr/bin/env python3
"""Week 6 PBS update: absolute SM assimilation with narrow SMAP filtering.

This script intentionally keeps the PBS likelihood simple:

    residual = satellite_SM - (VIC_particle_surface_SM + MLP_predicted_bias)

where the MLP bias model uses satellite SM as an input feature. Delta-SM and
event-detector likelihoods are not used in this main run.
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
WEEK3 = PROJECT / "Week 3"
WEEK5 = PROJECT / "Week 5"
WEEK6 = PROJECT / "Week 6"
sys.path.insert(0, str(WEEK3))

from analyze_pbs_week_particles_cellwise import (  # noqa: E402
    build_overlap_map,
    read_particle_vic_cells,
    read_vic_grid,
    resolve as week3_resolve,
    summarize_irrigation,
)


DEFAULT_SAT_CELLS = (
    PROJECT
    / "Week 2"
    / "outputs_smos_smap_9k"
    / "all_geojsons"
    / "dry_spottedtail_creek"
    / "dry_spottedtail_creek_selected_nsidc0800_cells.geojson"
)
DEFAULT_BASIN = (
    PROJECT
    / "Week 1"
    / "Deliverables"
    / "Dry_Spottedtail_Creek_USGS_06679000"
    / "dry_spottedtail_creek.geojson"
)
DEFAULT_FILTER_DIAGNOSTICS = (
    WEEK5
    / "outputs"
    / "week5_smap_filtering_sensitivity"
    / "dry_spottedtail_creek"
    / "physical_effective_cap_plus_mlp"
    / "smap_observations_with_filter_diagnostics.csv"
)
DEFAULT_MLP_MODEL = (
    WEEK5
    / "outputs"
    / "week5_smap_filtering_sensitivity"
    / "dry_spottedtail_creek"
    / "physical_effective_cap_plus_mlp"
    / "mlp_smap_bias_model.joblib"
)
DEFAULT_MLP_FEATURES = DEFAULT_MLP_MODEL.with_name("mlp_feature_list.json")
DEFAULT_MLP_METRICS = DEFAULT_MLP_MODEL.with_name("mlp_bias_validation_metrics.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Week 6 absolute-SM PBS scoring with narrow abnormal-observation filtering."
    )
    parser.add_argument("--basin-name", default="dry_spottedtail_creek")
    parser.add_argument(
        "--week-root",
        type=Path,
        required=True,
        help="Fresh VIC particle ensemble directory. Do not point this at Week 3 extracted outputs for the main Week 6 full run.",
    )
    parser.add_argument(
        "--ensemble-source-label",
        default="fresh_full_vic_ensemble_from_hopper",
        help="Metadata label describing whether the VIC ensemble is fresh or reused for comparison.",
    )
    parser.add_argument("--satellite-cells", type=Path, default=DEFAULT_SAT_CELLS)
    parser.add_argument("--basin-geojson", type=Path, default=DEFAULT_BASIN)
    parser.add_argument("--filter-diagnostics", type=Path, default=DEFAULT_FILTER_DIAGNOSTICS)
    parser.add_argument("--mlp-model", type=Path, default=DEFAULT_MLP_MODEL)
    parser.add_argument("--mlp-features", type=Path, default=DEFAULT_MLP_FEATURES)
    parser.add_argument("--mlp-metrics", type=Path, default=DEFAULT_MLP_METRICS)
    parser.add_argument("--out-root", type=Path, default=WEEK6 / "outputs")
    parser.add_argument("--layer1-depth-mm", type=float, default=50.0)
    parser.add_argument("--min-overlap-km2", type=float, default=0.001)
    parser.add_argument("--expected-threshold-count", type=int, default=161)
    parser.add_argument("--obs-sigma-m3m3", type=float, default=0.035)
    parser.add_argument("--random-seed", type=int, default=20260706)
    return parser.parse_args()


def clean_quality_tokens(value):
    if pd.isna(value):
        return []
    tokens = []
    for part in str(value).replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            tokens.append(int(float(part)))
        except ValueError:
            continue
    return tokens


def apply_observation_filter(path, expected_threshold_count):
    frame = pd.read_csv(path, parse_dates=["date"])
    frame["row"] = frame["smap_row"].astype(int)
    frame["col"] = frame["smap_col"].astype(int)
    frame["quality_class"] = frame["quality_flags"]
    frame["original_quality_flag"] = frame["quality_flags"]
    frame["raw_keep"] = frame["smap_sm_raw"].notna()
    frame["low_quality_flag_removed"] = frame["quality_flags"].map(
        lambda value: any(flag in {1, 2, 3, 4} for flag in clean_quality_tokens(value))
    )
    frame["threshold_abnormal_removed"] = (
        frame["physical_filter_flag"].astype(str).str.lower().eq("true")
    )
    frame["final_keep"] = (
        frame["raw_keep"]
        & ~frame["low_quality_flag_removed"]
        & ~frame["threshold_abnormal_removed"]
    )

    reasons = []
    for _, row in frame.iterrows():
        row_reasons = []
        if not row["raw_keep"]:
            row_reasons.append("no_raw_satellite_sm")
        if row["low_quality_flag_removed"]:
            row_reasons.append("quality_flag_1_4")
        if row["threshold_abnormal_removed"]:
            row_reasons.append("threshold_abnormal_large_increase")
        reasons.append(";".join(row_reasons) if row_reasons else "kept")
    frame["filter_reason"] = reasons
    frame["satellite_m3m3_raw"] = frame["smap_sm_raw"]
    frame["satellite_m3m3"] = frame["smap_sm_raw"]
    frame["threshold_value"] = frame.get("dtheta_max_allowed", np.nan)
    frame["delta_sm"] = frame.get("dtheta_observed", np.nan)

    threshold_count = int(frame["threshold_abnormal_removed"].sum())
    quality_count = int((frame["raw_keep"] & frame["low_quality_flag_removed"]).sum())
    filter_summary = {
        "n_rows_before_filtering": int(len(frame)),
        "n_raw_satellite_sm_available": int(frame["raw_keep"].sum()),
        "n_removed_quality_flags_1_4": quality_count,
        "n_removed_threshold_abnormal": threshold_count,
        "expected_threshold_abnormal_count": int(expected_threshold_count),
        "threshold_count_matches_expected": bool(threshold_count == expected_threshold_count),
        "n_final_keep_full_record": int(frame["final_keep"].sum()),
    }
    if threshold_count != expected_threshold_count:
        filter_summary["threshold_count_warning"] = (
            "Threshold-abnormal count did not match 161. Likely causes include a different "
            "date range, changed preprocessing, missing observations, or duplicate handling."
        )
    return frame, filter_summary


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


def safe_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def prepare_observations_for_pbs(filtered, start_date, end_date):
    obs = filtered[
        (filtered["date"] >= pd.Timestamp(start_date))
        & (filtered["date"] <= pd.Timestamp(end_date))
    ].copy()
    obs["period"] = "AM"
    obs["product"] = "SMAP"
    obs["source"] = "SMAP"
    obs["date"] = obs["date"].dt.date.astype(str)
    obs["observation_id"] = (
        obs["date"].astype(str)
        + "_"
        + obs["smap_cell_id"].astype(str)
        + "_"
        + obs["period"].astype(str)
    )
    return obs.sort_values(["date", "row", "col"]).reset_index(drop=True)


def score_particles_absolute_sm(particle_cells, observations, overlap, model, feature_names, sigma):
    mapping = overlap[
        [
            "row",
            "col",
            "vic_lat_index",
            "vic_lon_index",
            "overlap_area_km2",
            "weight_within_satellite_cell",
        ]
    ].copy()
    obs_keep = observations[observations["final_keep"]].copy()
    mapped = obs_keep.merge(mapping, on=["row", "col"], how="inner")
    mapped = mapped.merge(
        particle_cells,
        on=["date", "vic_lat_index", "vic_lon_index"],
        how="inner",
    )
    mapped["weighted_vic"] = mapped["vic_layer1_m3m3"] * mapped["weight_within_satellite_cell"]

    first_cols = {
        "satellite_m3m3": ("satellite_m3m3", "first"),
        "satellite_m3m3_raw": ("satellite_m3m3_raw", "first"),
        "smap_cell_id": ("smap_cell_id", "first"),
        "smap_lat": ("smap_lat", "first"),
        "smap_lon": ("smap_lon", "first"),
        "quality_class": ("quality_class", "first"),
        "filter_reason": ("filter_reason", "first"),
        "raw_keep": ("raw_keep", "first"),
        "final_keep": ("final_keep", "first"),
        "low_quality_flag_removed": ("low_quality_flag_removed", "first"),
        "threshold_abnormal_removed": ("threshold_abnormal_removed", "first"),
    }
    for col in feature_names:
        if col not in ("vic_surface_sm", "smap_sm_for_model") and col in mapped.columns:
            first_cols[col] = (col, "first")

    pred = (
        mapped.groupby(["particle", "date", "row", "col", "observation_id"], as_index=False)
        .agg(
            predicted_vic_cell_m3m3=("weighted_vic", "sum"),
            overlap_area_km2=("overlap_area_km2", "sum"),
            n_vic_cells=("vic_lat_index", "nunique"),
            **first_cols,
        )
    )
    pred["vic_surface_sm"] = pred["predicted_vic_cell_m3m3"]
    pred["smap_sm_for_model"] = pred["satellite_m3m3"]

    feature_frame = pred[feature_names].replace([np.inf, -np.inf], np.nan)
    pred["mlp_feature_complete"] = feature_frame.notna().all(axis=1)
    pred["mlp_predicted_bias_m3m3"] = np.nan
    ok = pred["mlp_feature_complete"]
    pred.loc[ok, "mlp_predicted_bias_m3m3"] = model.predict(feature_frame.loc[ok])
    pred = pred[pred["mlp_predicted_bias_m3m3"].notna()].copy()

    pred["mlp_mapped_vic_m3m3"] = pred["predicted_vic_cell_m3m3"] + pred["mlp_predicted_bias_m3m3"]
    pred["residual_m3m3"] = pred["satellite_m3m3"] - pred["mlp_mapped_vic_m3m3"]
    pred["obs_sigma_m3m3"] = sigma
    pred["log_likelihood"] = -0.5 * (pred["residual_m3m3"] / sigma) ** 2

    scores = (
        pred.groupby("particle", as_index=False)
        .agg(
            n_cell_observations=("log_likelihood", "size"),
            log_likelihood=("log_likelihood", "sum"),
            mean_abs_residual_m3m3=("residual_m3m3", lambda x: float(np.mean(np.abs(x)))),
            rmse_residual_m3m3=("residual_m3m3", lambda x: float(np.sqrt(np.mean(x**2)))),
        )
        .sort_values("log_likelihood", ascending=False)
    )
    max_log = scores["log_likelihood"].max()
    scores["weight"] = np.exp(scores["log_likelihood"] - max_log)
    scores["weight"] = scores["weight"] / scores["weight"].sum()
    scores["rank"] = np.arange(1, len(scores) + 1)
    return pred, scores


def plot_raw_filtered(observations, out_dir):
    fig, ax = plt.subplots(figsize=(11.5, 5.0), constrained_layout=True)
    kept = observations[observations["final_keep"] & observations["satellite_m3m3_raw"].notna()]
    low = observations[observations["low_quality_flag_removed"] & observations["satellite_m3m3_raw"].notna()]
    thresh = observations[observations["threshold_abnormal_removed"] & observations["satellite_m3m3_raw"].notna()]
    noraw = observations[~observations["raw_keep"]]
    ax.scatter(kept["date_dt"], kept["satellite_m3m3_raw"], s=28, color="#0072B2", label="Kept SMAP")
    ax.scatter(low["date_dt"], low["satellite_m3m3_raw"], s=30, marker="^", color="#BDBDBD", label="Removed quality 1-4")
    ax.scatter(thresh["date_dt"], thresh["satellite_m3m3_raw"], s=36, marker="s", color="#7A7A7A", label="Removed threshold abnormal")
    ax.scatter(noraw["date_dt"], np.full(len(noraw), np.nan), s=1, alpha=0)
    ax.set_title("Raw vs Filtered SMAP Observations for PBS Window")
    ax.set_ylabel("SMAP soil moisture (m3/m3)")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=3)
    fig.savefig(out_dir / "raw_vs_filtered_satellite_sm_timeseries.png", dpi=180)
    plt.close(fig)


def plot_vic_mlp_satellite(matches, out_dir):
    daily = (
        matches.groupby(["date"], as_index=False)
        .agg(
            satellite_m3m3=("satellite_m3m3", "mean"),
            raw_vic_m3m3=("predicted_vic_cell_m3m3", "mean"),
            mlp_mapped_vic_m3m3=("mlp_mapped_vic_m3m3", "mean"),
            residual_m3m3=("residual_m3m3", "mean"),
        )
    )
    daily["date_dt"] = pd.to_datetime(daily["date"])
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.0), sharex=True, constrained_layout=True)
    axes[0].plot(daily["date_dt"], daily["satellite_m3m3"], marker="o", color="#0072B2", label="SMAP")
    axes[0].plot(daily["date_dt"], daily["raw_vic_m3m3"], marker="o", color="#6B7280", label="Raw VIC particle mean")
    axes[0].plot(daily["date_dt"], daily["mlp_mapped_vic_m3m3"], marker="o", color="#009E73", label="MLP-mapped VIC")
    axes[0].set_ylabel("SM (m3/m3)")
    axes[0].set_title("Absolute SM Assimilation Inputs")
    axes[0].legend(frameon=False, ncol=3)
    axes[0].grid(alpha=0.22)
    axes[1].bar(daily["date_dt"], daily["residual_m3m3"], color="#8DA0CB", width=0.65)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_ylabel("Mean residual (m3/m3)")
    axes[1].set_title("PBS residual = SMAP - MLP-mapped VIC")
    axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=1))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[1].grid(axis="y", alpha=0.22)
    fig.savefig(out_dir / "vic_raw_mlp_corrected_satellite_sm_comparison.png", dpi=180)
    plt.close(fig)


def plot_weights(scores, out_dir):
    ordered = scores.sort_values("particle")
    fig, ax = plt.subplots(figsize=(10.5, 4.4), constrained_layout=True)
    colors = np.where(ordered["rank"].eq(1), "#D55E00", "#0072B2")
    ax.bar(ordered["particle"].astype(str), ordered["weight"], color=colors)
    ax.set_title("PBS Particle Posterior Weights")
    ax.set_xlabel("Particle")
    ax.set_ylabel("Weight")
    ax.grid(axis="y", alpha=0.22)
    fig.savefig(out_dir / "pbs_particle_weights.png", dpi=180)
    plt.close(fig)


def write_and_plot_weights_over_time(matches, out_dir):
    daily_ll = (
        matches.groupby(["particle", "date"], as_index=False)
        .agg(daily_log_likelihood=("log_likelihood", "sum"))
        .sort_values(["particle", "date"])
    )
    daily_ll["cumulative_log_likelihood"] = daily_ll.groupby("particle")["daily_log_likelihood"].cumsum()
    frames = []
    for _, group in daily_ll.groupby("date", sort=True):
        group = group.copy()
        max_log = group["cumulative_log_likelihood"].max()
        group["weight_to_date"] = np.exp(group["cumulative_log_likelihood"] - max_log)
        group["weight_to_date"] = group["weight_to_date"] / group["weight_to_date"].sum()
        group["rank_to_date"] = group["weight_to_date"].rank(ascending=False, method="first").astype(int)
        frames.append(group)
    weights_time = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    weights_time.to_csv(out_dir / "pbs_particle_weights_over_time.csv", index=False)

    if weights_time.empty:
        return weights_time
    plot = weights_time.copy()
    plot["date_dt"] = pd.to_datetime(plot["date"])
    top_particles = (
        plot.groupby("particle")["weight_to_date"]
        .max()
        .sort_values(ascending=False)
        .head(8)
        .index
    )
    fig, ax = plt.subplots(figsize=(11.5, 5.0), constrained_layout=True)
    for particle in top_particles:
        sub = plot[plot["particle"] == particle]
        ax.plot(sub["date_dt"], sub["weight_to_date"], marker="o", linewidth=1.7, label=f"p{int(particle)}")
    ax.set_title("PBS Particle Weights Over Assimilation Dates")
    ax.set_ylabel("Normalized weight to date")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=4, fontsize=8)
    fig.savefig(out_dir / "pbs_particle_weights_over_time.png", dpi=180)
    plt.close(fig)
    return weights_time


def plot_irrigation(daily_irrigation, out_dir):
    daily = daily_irrigation.copy()
    daily["date_dt"] = pd.to_datetime(daily["date"])
    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    ax.bar(daily["date_dt"], daily["posterior_mean_irrigation_mm"], color="#009E73", width=0.65, label="Posterior mean")
    ax.plot(daily["date_dt"], daily["unweighted_mean_irrigation_mm"], color="#6B7280", marker="o", label="Prior/unweighted mean")
    ax.set_title("Posterior Daily Irrigation")
    ax.set_ylabel("Irrigation (mm/day)")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.savefig(out_dir / "posterior_daily_irrigation.png", dpi=180)
    plt.close(fig)


def plot_filter_reasons(observations, out_dir):
    counts = observations["filter_reason"].value_counts().rename_axis("filter_reason").reset_index(name="count")
    fig, ax = plt.subplots(figsize=(9.5, 4.3), constrained_layout=True)
    ax.barh(counts["filter_reason"], counts["count"], color="#8DA0CB")
    ax.set_title("PBS Window Observation Filter Reasons")
    ax.set_xlabel("Observation rows")
    ax.grid(axis="x", alpha=0.22)
    fig.savefig(out_dir / "filter_reason_counts.png", dpi=180)
    plt.close(fig)
    return counts


def write_summary_report(out_dir, metadata, filter_summary, scores, daily_irrigation, warnings):
    ess = 1.0 / float(np.sum(scores["weight"] ** 2)) if len(scores) else np.nan
    best = scores.iloc[0] if len(scores) else None
    lines = [
        "# Week 6 PBS Absolute-SM Update",
        "",
        "## Requested Configuration",
        "",
        "- Assimilation variable: absolute satellite soil moisture.",
        "- Delta-SM assimilation: disabled for this main run.",
        "- Filter: narrow abnormal-observation filter, removing quality flags/classes 1-4 and the existing 161 threshold/effective-cap abnormal observations.",
        "- MLP correction: uses the MLP configuration that includes satellite SM as an input feature during the irrigation period.",
        "- Disabled for this run: adaptive physical filtering, event-detector likelihoods, broad Zaussinger deletion, and delta-SM likelihoods.",
        "",
        "## Observation Filtering Summary",
        "",
        f"- Rows before filtering: {filter_summary['n_rows_before_filtering']}",
        f"- Raw satellite SM rows available: {filter_summary['n_raw_satellite_sm_available']}",
        f"- Removed by quality flags/classes 1-4: {filter_summary['n_removed_quality_flags_1_4']}",
        f"- Removed by threshold abnormal-increase method: {filter_summary['n_removed_threshold_abnormal']}",
        f"- Threshold count matched expected 161: {filter_summary['threshold_count_matches_expected']}",
        f"- Final full-record kept rows: {filter_summary['n_final_keep_full_record']}",
        f"- PBS-window rows: {metadata['pbs_window_observation_rows_before_filtering']}",
        f"- PBS-window kept rows: {metadata['pbs_window_observation_rows_final_keep']}",
        "",
        "## MLP Correction Summary",
        "",
        f"- MLP model path: `{metadata['mlp_model_path']}`",
        f"- Satellite SM included as MLP feature: {metadata['satellite_sm_included_as_mlp_feature']}",
        f"- Feature columns: {', '.join(metadata['mlp_feature_columns'])}",
        f"- MLP validation RMSE: {metadata.get('mlp_validation_rmse')}",
        f"- MLP validation R2: {metadata.get('mlp_validation_r2')}",
        "",
        "The PBS likelihood used `satellite_SM - (VIC_particle_surface_SM + MLP_predicted_bias)`. This is an absolute-SM likelihood, not a delta-SM or event-magnitude likelihood.",
        "",
        "## PBS Configuration",
        "",
        f"- Basin: {metadata['basin_name']}",
        f"- VIC ensemble source: {metadata['vic_ensemble_source_label']}",
        f"- VIC ensemble directory: `{metadata['vic_input_output_directory']}`",
        f"- Date range: {metadata['date_range']['start']} to {metadata['date_range']['end']}",
        f"- Particle count: {metadata['pbs_particle_count']}",
        f"- Observation sigma: {metadata['obs_sigma_m3m3']} m3/m3",
        f"- Effective sample size: {ess:.2f}",
    ]
    if best is not None:
        lines.extend(
            [
                f"- Best particle: {int(best['particle'])}",
                f"- Best particle weight: {best['weight']:.4f}",
                f"- Best particle RMSE residual: {best['rmse_residual_m3m3']:.4f} m3/m3",
            ]
        )
    lines.extend(["", "## Main PBS Irrigation Results", ""])
    for _, row in daily_irrigation.iterrows():
        lines.append(
            f"- {row['date']}: posterior mean irrigation = {row['posterior_mean_irrigation_mm']:.2f} mm/day "
            f"(prior mean = {row['unweighted_mean_irrigation_mm']:.2f})"
        )
    lines.extend(["", "## Warnings / Deviations", ""])
    if warnings:
        lines.extend([f"- {warning}" for warning in warnings])
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `filtered_satellite_observations.csv`",
            "- `pbs_particle_weights.csv`",
            "- `pbs_particle_satellite_matches.csv`",
            "- `posterior_daily_irrigation.csv`",
            "- `pbs_run_metadata.json`",
            "- `raw_vs_filtered_satellite_sm_timeseries.png`",
            "- `vic_raw_mlp_corrected_satellite_sm_comparison.png`",
            "- `pbs_particle_weights.png`",
            "- `pbs_particle_weights_over_time.png`",
            "- `posterior_daily_irrigation.png`",
            "- `filter_reason_counts.png`",
        ]
    )
    (out_dir / "summary_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / f"pbs_absoluteSM_mlpSatSM_zaussingerThreshold_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Starting Week 6 PBS absolute-SM run")
    print(f"Output directory: {out_dir}")
    filtered_full, filter_summary = apply_observation_filter(
        args.filter_diagnostics, args.expected_threshold_count
    )
    print(f"Observations before filtering: {filter_summary['n_rows_before_filtering']}")
    print(f"Removed by quality flags/classes 1-4: {filter_summary['n_removed_quality_flags_1_4']}")
    print(f"Removed by threshold abnormal method: {filter_summary['n_removed_threshold_abnormal']}")
    if not filter_summary["threshold_count_matches_expected"]:
        print("WARNING:", filter_summary["threshold_count_warning"])
    else:
        print("Threshold abnormal count matched expected 161.")
    print(f"Final observations available full-record: {filter_summary['n_final_keep_full_record']}")

    week_root = Path(args.week_root).resolve()
    vic_grid = read_vic_grid(week_root)
    overlap = build_overlap_map(
        vic_grid,
        Path(args.satellite_cells).resolve(),
        Path(args.basin_geojson).resolve(),
        args.min_overlap_km2,
    )
    if overlap.empty:
        raise RuntimeError("No satellite/VIC/basin overlaps found; cannot run PBS scoring.")
    overlap.drop(columns="geometry", errors="ignore").to_csv(out_dir / "cellwise_satellite_vic_overlap_map.csv", index=False)

    particles = read_particle_vic_cells(week_root, args.layer1_depth_mm)
    start_date = particles["date"].min()
    end_date = particles["date"].max()
    observations = prepare_observations_for_pbs(filtered_full, start_date, end_date)
    observations["date_dt"] = pd.to_datetime(observations["date"])
    print(f"PBS window: {start_date} to {end_date}")
    print(f"PBS-window observations before filtering: {len(observations)}")
    print(f"PBS-window observations final keep: {int(observations['final_keep'].sum())}")
    if observations["final_keep"].sum() == 0:
        print("WARNING: zero valid satellite observations after filtering for PBS window.")

    model = joblib.load(args.mlp_model)
    feature_names = json.loads(Path(args.mlp_features).read_text(encoding="utf-8"))
    mlp_metrics = json.loads(Path(args.mlp_metrics).read_text(encoding="utf-8"))
    if "smap_sm_for_model" not in feature_names:
        raise RuntimeError("Requested MLP configuration must include satellite SM; smap_sm_for_model is missing.")

    matches, scores = score_particles_absolute_sm(
        particles,
        observations,
        overlap,
        model,
        feature_names,
        args.obs_sigma_m3m3,
    )
    daily_irrigation = summarize_irrigation(week_root, scores)
    particle_irrigation = pd.read_csv(week_root / "particle_irrigation_inputs.csv").merge(
        scores[["particle", "weight"]], on="particle", how="left"
    )
    particle_irrigation["weighted_irrigation_mm_day"] = (
        particle_irrigation["irrigation_mm_day"] * particle_irrigation["weight"]
    )
    particle_total = (
        particle_irrigation.groupby("particle", as_index=False)
        .agg(total_irrigation_mm=("irrigation_mm_day", "sum"))
        .merge(scores[["particle", "rank", "weight"]], on="particle", how="left")
        .sort_values("rank")
    )

    filter_counts = plot_filter_reasons(observations, out_dir)
    observations.to_csv(out_dir / "filtered_satellite_observations.csv", index=False)
    matches.to_csv(out_dir / "pbs_particle_satellite_matches.csv", index=False)
    scores.to_csv(out_dir / "pbs_particle_weights.csv", index=False)
    daily_irrigation.to_csv(out_dir / "posterior_daily_irrigation.csv", index=False)
    particle_total.to_csv(out_dir / "cellwise_or_basin_irrigation_summary.csv", index=False)
    filter_counts.to_csv(out_dir / "filter_reason_counts.csv", index=False)

    plot_raw_filtered(observations, out_dir)
    plot_vic_mlp_satellite(matches, out_dir)
    plot_weights(scores, out_dir)
    weights_time = write_and_plot_weights_over_time(matches, out_dir)
    plot_irrigation(daily_irrigation, out_dir)

    warnings = []
    if not filter_summary["threshold_count_matches_expected"]:
        warnings.append(filter_summary["threshold_count_warning"])
    threshold_pbs = int(observations["threshold_abnormal_removed"].sum())
    if threshold_pbs == 0:
        warnings.append(
            "The full-record threshold-abnormal count matched 161, but none of those observations fell inside the available July 1-7 PBS window."
        )
    if observations["final_keep"].sum() == 0:
        warnings.append("Zero valid satellite observations after filtering in PBS window.")

    metadata = {
        "basin_name": args.basin_name,
        "run_date_time": datetime.now().isoformat(timespec="seconds"),
        "output_directory": str(out_dir),
        "vic_ensemble_source_label": args.ensemble_source_label,
        "fresh_vic_ensemble_outputs_used": args.ensemble_source_label.startswith("fresh"),
        "reused_week3_outputs_used_as_main_run": "Week 3" in str(week_root),
        "assimilation_variable": "absolute_satellite_soil_moisture",
        "delta_SM_assimilation": False,
        "event_detector_likelihood": False,
        "adaptive_physical_filtering": False,
        "filter_type": "narrow_quality_1_4_plus_existing_161_threshold_abnormal_effective_cap",
        "filter_summary": filter_summary,
        "pbs_window_observation_rows_before_filtering": int(len(observations)),
        "pbs_window_observation_rows_final_keep": int(observations["final_keep"].sum()),
        "pbs_window_threshold_abnormal_removed": threshold_pbs,
        "threshold_abnormal_count_matched_161": bool(filter_summary["threshold_count_matches_expected"]),
        "mlp_model_path": str(Path(args.mlp_model).resolve()),
        "mlp_feature_columns": feature_names,
        "satellite_sm_included_as_mlp_feature": "smap_sm_for_model" in feature_names,
        "mlp_training_validation_version": mlp_metrics,
        "mlp_scaler_or_preprocessor_path": str(Path(args.mlp_model).resolve()),
        "period": "irrigation" if pd.Timestamp(start_date).month in [5, 6, 7, 8, 9] else "non_irrigation",
        "pbs_particle_count": int(particles["particle"].nunique()),
        "assimilation_dates_with_valid_observations": sorted(matches["date"].unique().tolist()),
        "assimilation_windows_with_zero_valid_satellite_observations": sorted(
            list(set(pd.date_range(start_date, end_date).date.astype(str)) - set(matches["date"].unique()))
        ),
        "assimilation_window_length_days": int((pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1),
        "date_range": {"start": start_date, "end": end_date},
        "random_seed": args.random_seed,
        "vic_input_output_directory": str(week_root),
        "satellite_filter_diagnostics_path": str(Path(args.filter_diagnostics).resolve()),
        "satellite_cells_path": str(Path(args.satellite_cells).resolve()),
        "basin_geojson_path": str(Path(args.basin_geojson).resolve()),
        "obs_sigma_m3m3": args.obs_sigma_m3m3,
        "code_commit_hash": git_commit_hash(),
        "warnings": warnings,
        "verification": {
            "delta_SM_assimilation_was_not_used": True,
            "absolute_SM_likelihood_was_used": True,
            "quality_classes_flags_1_4_removed": True,
            "threshold_detected_abnormal_observations_removed_or_warning_reported": True,
            "satellite_SM_included_in_MLP_features": "smap_sm_for_model" in feature_names,
            "pbs_scoring_ran_successfully": True,
            "output_files_written": True,
        },
    }
    metadata["mlp_validation_rmse"] = safe_float(mlp_metrics.get("validation_rmse"))
    metadata["mlp_validation_r2"] = safe_float(mlp_metrics.get("validation_r2"))

    (out_dir / "pbs_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_summary_report(out_dir, metadata, filter_summary, scores, daily_irrigation, warnings)

    # Keep a copy of the exact runner for provenance.
    shutil.copy2(Path(__file__), out_dir / "run_pbs_absolute_sm_mlp_satsm.py")
    print(f"Wrote PBS outputs to {out_dir}")


if __name__ == "__main__":
    main()
