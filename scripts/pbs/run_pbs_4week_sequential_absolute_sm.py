#!/usr/bin/env python3
"""Sequential 4-week PBS postprocessing for a fresh VIC particle ensemble.

This is not a single 4-week likelihood. It splits the fresh 4-week VIC ensemble
into weekly assimilation windows, applies the Week 6 observation filter and MLP
mapping, updates particle weights per window, and reports posterior irrigation
per day.

The current implementation performs sequential reweighting over propagated VIC
particle trajectories. It does not rerun VIC after discrete resampling between
windows; that would require a Hopper rerun at each weekly boundary.
"""

import argparse
import json
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
WEEK6 = PROJECT / "Week 6"
plt.style.use(str(PROJECT / "research_report.mplstyle"))
sys.path.insert(0, str(WEEK6))

from run_pbs_absolute_sm_mlp_satsm import (  # noqa: E402
    DEFAULT_BASIN,
    DEFAULT_FILTER_DIAGNOSTICS,
    DEFAULT_MLP_FEATURES,
    DEFAULT_MLP_METRICS,
    DEFAULT_MLP_MODEL,
    DEFAULT_SAT_CELLS,
    apply_observation_filter,
    build_overlap_map,
    git_commit_hash,
    joblib,
    prepare_observations_for_pbs,
    read_particle_vic_cells,
    read_vic_grid,
    safe_float,
    score_particles_absolute_sm,
)


RUN_ID = "pbs4week_absoluteSM_mlpSatSM_threshold_20260706_053539"
DEFAULT_WEEK_ROOT = WEEK6 / "data" / "fresh_pbs_runs_4week" / "extracted" / RUN_ID


def parse_args():
    parser = argparse.ArgumentParser(description="Run sequential 4-week absolute-SM PBS postprocessing.")
    parser.add_argument("--basin-name", default="dry_spottedtail_creek")
    parser.add_argument("--week-root", type=Path, required=True)
    parser.add_argument("--satellite-cells", type=Path, default=DEFAULT_SAT_CELLS)
    parser.add_argument("--basin-geojson", type=Path, default=DEFAULT_BASIN)
    parser.add_argument("--filter-diagnostics", type=Path, default=DEFAULT_FILTER_DIAGNOSTICS)
    parser.add_argument("--mlp-model", type=Path, default=DEFAULT_MLP_MODEL)
    parser.add_argument("--mlp-features", type=Path, default=DEFAULT_MLP_FEATURES)
    parser.add_argument("--mlp-metrics", type=Path, default=DEFAULT_MLP_METRICS)
    parser.add_argument("--out-root", type=Path, default=WEEK6 / "outputs" / "fresh_4week_sequential_pbs")
    parser.add_argument("--layer1-depth-mm", type=float, default=50.0)
    parser.add_argument("--min-overlap-km2", type=float, default=0.001)
    parser.add_argument("--expected-threshold-count", type=int, default=161)
    parser.add_argument("--obs-sigma-m3m3", type=float, default=0.035)
    parser.add_argument("--random-seed", type=int, default=20260706)
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--low-ess-threshold", type=float, default=3.0)
    return parser.parse_args()


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


def normalize_log_weights(log_values):
    max_log = np.nanmax(log_values)
    weights = np.exp(log_values - max_log)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return np.ones_like(log_values) / len(log_values)
    return weights / total


def score_window(window, particles, observations, overlap, model, feature_names, sigma, prior_weights):
    start = window["window_start"]
    end = window["window_end"]
    obs = observations[(observations["date"] >= start) & (observations["date"] <= end)].copy()
    kept = obs[obs["final_keep"]].copy()
    particle_ids = sorted(particles["particle"].unique())

    if kept.empty:
        scores = pd.DataFrame(
            {
                "particle": particle_ids,
                "prior_weight": [prior_weights.get(p, 1.0 / len(particle_ids)) for p in particle_ids],
                "n_cell_observations": 0,
                "log_likelihood": 0.0,
                "log_posterior_unnormalized": np.log([prior_weights.get(p, 1.0 / len(particle_ids)) for p in particle_ids]),
            }
        )
        scores["weight"] = scores["prior_weight"]
        scores["rank"] = scores["weight"].rank(ascending=False, method="first").astype(int)
        return pd.DataFrame(), scores.sort_values("rank"), obs

    matches, _ = score_particles_absolute_sm(particles, obs, overlap, model, feature_names, sigma)
    grouped = (
        matches.groupby("particle", as_index=False)
        .agg(
            n_cell_observations=("log_likelihood", "size"),
            log_likelihood=("log_likelihood", "sum"),
            mean_abs_residual_m3m3=("residual_m3m3", lambda x: float(np.mean(np.abs(x)))),
            rmse_residual_m3m3=("residual_m3m3", lambda x: float(np.sqrt(np.mean(x**2)))),
        )
    )
    base = pd.DataFrame({"particle": particle_ids})
    scores = base.merge(grouped, on="particle", how="left")
    scores["n_cell_observations"] = scores["n_cell_observations"].fillna(0).astype(int)
    scores["log_likelihood"] = scores["log_likelihood"].fillna(0.0)
    scores["mean_abs_residual_m3m3"] = scores["mean_abs_residual_m3m3"].fillna(np.nan)
    scores["rmse_residual_m3m3"] = scores["rmse_residual_m3m3"].fillna(np.nan)
    scores["prior_weight"] = scores["particle"].map(prior_weights).fillna(1.0 / len(particle_ids))
    scores["log_posterior_unnormalized"] = np.log(np.maximum(scores["prior_weight"], 1e-300)) + scores["log_likelihood"]
    scores["weight"] = normalize_log_weights(scores["log_posterior_unnormalized"].to_numpy())
    scores["rank"] = scores["weight"].rank(ascending=False, method="first").astype(int)
    return matches, scores.sort_values("rank"), obs


def summarize_window_irrigation(week_root, scores, window):
    irrigation = pd.read_csv(week_root / "particle_irrigation_inputs.csv")
    irrigation = irrigation[(irrigation["date"] >= window["window_start"]) & (irrigation["date"] <= window["window_end"])].copy()
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
    return daily


def add_window_columns(df, window):
    out = df.copy()
    out["window_id"] = window["window_id"]
    out["window_start"] = window["window_start"]
    out["window_end"] = window["window_end"]
    return out


def plot_outputs(out_dir, observations, matches, weights, daily, window_summary):
    obs = observations.copy()
    obs["date_dt"] = pd.to_datetime(obs["date"])
    fig, ax = plt.subplots(figsize=(13.5, 5.0), constrained_layout=True)
    kept = obs[obs["final_keep"] & obs["satellite_m3m3_raw"].notna()]
    low = obs[obs["low_quality_flag_removed"] & obs["satellite_m3m3_raw"].notna()]
    thresh = obs[obs["threshold_abnormal_removed"] & obs["satellite_m3m3_raw"].notna()]
    ax.scatter(kept["date_dt"], kept["satellite_m3m3_raw"], s=18, color="#365C8D", label="Kept SMAP")
    ax.scatter(low["date_dt"], low["satellite_m3m3_raw"], s=20, marker="^", color="#BDBDBD", label="Removed quality 1-4")
    ax.scatter(thresh["date_dt"], thresh["satellite_m3m3_raw"], s=22, marker="s", color="#7A7A7A", label="Removed threshold abnormal")
    ax.set_title("4-week raw vs filtered satellite SM", pad=38)
    ax.set_ylabel("SMAP soil moisture (m3/m3)")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.savefig(out_dir / "raw_vs_filtered_satellite_sm_4week.png", dpi=300)
    plt.close(fig)

    m = matches.copy()
    m["date_dt"] = pd.to_datetime(m["date"])
    daily_sm = (
        m.groupby("date_dt", as_index=False)
        .agg(
            satellite_m3m3=("satellite_m3m3", "mean"),
            raw_vic_m3m3=("predicted_vic_cell_m3m3", "mean"),
            mlp_mapped_vic_m3m3=("mlp_mapped_vic_m3m3", "mean"),
        )
    )
    fig, ax = plt.subplots(figsize=(13.5, 5.2), constrained_layout=True)
    ax.plot(daily_sm["date_dt"], daily_sm["satellite_m3m3"], marker="o", color="#365C8D", label="SMAP")
    ax.plot(daily_sm["date_dt"], daily_sm["raw_vic_m3m3"], marker="o", color="#5C6773", label="Raw VIC")
    ax.plot(daily_sm["date_dt"], daily_sm["mlp_mapped_vic_m3m3"], marker="o", color="#1F9E89", label="MLP-mapped VIC")
    ax.set_title("4-week absolute SM assimilation inputs", pad=38)
    ax.set_ylabel("SM (m3/m3)")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.savefig(out_dir / "vic_raw_mlp_satellite_sm_4week.png", dpi=300)
    plt.close(fig)

    d = daily.copy()
    d["date_dt"] = pd.to_datetime(d["date"])
    fig, ax = plt.subplots(figsize=(13.5, 5.0), constrained_layout=True)
    ax.bar(d["date_dt"], d["posterior_mean_irrigation_mm"], width=0.75, color="#365C8D", label="Posterior mean")
    ax.plot(d["date_dt"], d["prior_mean_irrigation_mm"], marker="o", color="#5C6773", label="Prior mean")
    ax.set_title("4-week posterior daily irrigation vs prior mean", pad=38)
    ax.set_ylabel("Irrigation (mm/day)")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.savefig(out_dir / "posterior_vs_prior_daily_irrigation_4week.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.4), constrained_layout=True)
    ax.bar(window_summary["window_id"].astype(str), window_summary["effective_sample_size"], color="#365C8D")
    ax.set_title("Effective sample size by weekly window")
    ax.set_xlabel("Window")
    ax.set_ylabel("ESS")
    fig.savefig(out_dir / "effective_sample_size_by_window.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.4), constrained_layout=True)
    ax.bar(window_summary["window_id"].astype(str), window_summary["max_particle_weight"], color="#B58D1D")
    ax.set_title("Max particle weight by weekly window")
    ax.set_xlabel("Window")
    ax.set_ylabel("Max weight")
    fig.savefig(out_dir / "max_particle_weight_by_window.png", dpi=300)
    plt.close(fig)

    reason_counts = (
        observations.groupby(["window_id", "filter_reason"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    pivot = reason_counts.pivot(index="window_id", columns="filter_reason", values="count").fillna(0)
    fig, ax = plt.subplots(figsize=(10.0, 4.8), constrained_layout=True)
    pivot.plot(kind="bar", stacked=True, ax=ax, color=["#365C8D", "#BDBDBD", "#5C6773", "#B58D1D"][: len(pivot.columns)])
    ax.set_title("Filter reason counts by window", pad=38)
    ax.set_xlabel("Window")
    ax.set_ylabel("Observation rows")
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=min(4, len(pivot.columns)))
    fig.savefig(out_dir / "filter_reason_counts_by_window.png", dpi=300)
    plt.close(fig)
    reason_counts.to_csv(out_dir / "filter_reason_counts_by_window.csv", index=False)

    max_by_particle = weights.groupby("particle")["weight"].max().sort_values(ascending=False)
    top_n = min(25, len(max_by_particle))
    top_particles = max_by_particle.head(top_n).index
    fig, ax = plt.subplots(figsize=(13.5, 6.8), constrained_layout=True)
    for particle, sub in weights.groupby("particle"):
        sub = sub.sort_values("window_id")
        ax.plot(
            sub["window_id"],
            sub["weight"],
            color="#C7CDD6",
            linewidth=0.8,
            alpha=0.32,
            zorder=1,
        )
    for particle in top_particles:
        sub = weights[weights["particle"] == particle].sort_values("window_id")
        if {"proposal_round", "original_particle"}.issubset(sub.columns):
            first = sub.iloc[0]
            label = f"r{int(first['proposal_round'])}-p{int(first['original_particle'])}"
        else:
            label = f"p{int(particle)}"
        ax.plot(sub["window_id"], sub["weight"], marker="o", linewidth=1.8, label=label, zorder=2)
    ax.set_title(f"Particle weights by weekly window: all particles, top {top_n} highlighted", pad=38)
    ax.set_xlabel("Window")
    ax.set_ylabel("Posterior weight")
    ax.legend(
        frameon=False,
        ncol=5,
        fontsize=7,
        title=f"Highlighted top {top_n} by max weekly weight",
        title_fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.savefig(out_dir / "particle_weights_by_window.png", dpi=300)
    plt.close(fig)


def write_report(out_dir, metadata, window_summary):
    lines = [
        "# Week 6 Four-Week Sequential PBS Run",
        "",
        "This is a 4-week sequential PBS postprocessing run over a fresh 4-week VIC particle ensemble. Weekly windows were scored separately; this is not one giant 4-week likelihood.",
        "",
        "## Configuration",
        "",
        "- Assimilation variable: absolute satellite soil moisture.",
        "- Delta-SM assimilation: disabled.",
        "- Adaptive physical filtering: disabled.",
        "- Event-detector likelihoods: disabled.",
        "- Broad Zaussinger deletion: disabled.",
        "- Narrow filter: quality flags/classes 1-4 plus the existing threshold/effective-cap abnormal observations.",
        "- MLP mapping includes satellite SM feature `smap_sm_for_model`.",
        "- Sequential update: weekly particle reweighting over propagated fresh VIC trajectories.",
        "",
        "## Observation Filtering",
        "",
        f"- Full-record rows before filtering: {metadata['filter_summary']['n_rows_before_filtering']}",
        f"- Removed by quality flags/classes 1-4 across full record: {metadata['filter_summary']['n_removed_quality_flags_1_4']}",
        f"- Removed by threshold/effective-cap abnormal filter across full record: {metadata['filter_summary']['n_removed_threshold_abnormal']}",
        f"- Threshold count matched expected 161: {metadata['filter_summary']['threshold_count_matches_expected']}",
        f"- Four-week rows before filtering: {metadata['four_week_observation_rows_before_filtering']}",
        f"- Four-week kept rows: {metadata['four_week_observation_rows_final_keep']}",
        f"- Four-week threshold/effective-cap removals: {metadata['four_week_threshold_abnormal_removed']}",
        "",
        "## Window Diagnostics",
        "",
        "| Window | Dates | Kept obs | Quality removed | Threshold removed | ESS | Max weight | Best particle | RMSE | Warnings |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in window_summary.iterrows():
        warnings = row.get("warnings", "")
        lines.append(
            f"| {int(row.window_id)} | {row.window_start} to {row.window_end} | "
            f"{int(row.kept_observations)} | {int(row.quality_removed)} | {int(row.threshold_abnormal_removed)} | "
            f"{row.effective_sample_size:.2f} | {row.max_particle_weight:.3f} | {int(row.best_particle)} | "
            f"{row.residual_rmse_m3m3:.4f} | {warnings} |"
        )
    near_zero = bool(metadata["posterior_irrigation_near_zero_all_days"])
    lines.extend(
        [
            "",
            "## Irrigation Result",
            "",
            f"- Posterior irrigation near zero for all days: {near_zero}",
            f"- Total posterior mean irrigation over the 4-week period: {metadata['total_posterior_irrigation_mm']:.4f} mm",
            f"- Total prior mean irrigation over the 4-week period: {metadata['total_prior_irrigation_mm']:.4f} mm",
            "",
            "## Warnings / Deviations",
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
            "- `window_summary.csv`",
            "- `filtered_satellite_observations.csv`",
            "- `pbs_particle_weights.csv`",
            "- `pbs_particle_satellite_matches.csv`",
            "- `posterior_daily_irrigation.csv`",
            "- `pbs_run_metadata.json`",
            "- `summary_report.md`",
        ]
    )
    (out_dir / "summary_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / f"pbs4week_absoluteSM_mlpSatSM_threshold_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    week_root = args.week_root.resolve()
    if "Week 3" in str(week_root):
        raise RuntimeError("Refusing to use Week 3 extracted outputs for the 4-week main run.")
    if not week_root.exists():
        raise FileNotFoundError(f"Fresh 4-week ensemble directory not found: {week_root}")

    filtered_full, filter_summary = apply_observation_filter(args.filter_diagnostics, args.expected_threshold_count)
    vic_grid = read_vic_grid(week_root)
    overlap = build_overlap_map(vic_grid, args.satellite_cells.resolve(), args.basin_geojson.resolve(), args.min_overlap_km2)
    if overlap.empty:
        raise RuntimeError("No satellite/VIC/basin overlaps found.")
    overlap.drop(columns="geometry", errors="ignore").to_csv(out_dir / "cellwise_satellite_vic_overlap_map.csv", index=False)

    particles = read_particle_vic_cells(week_root, args.layer1_depth_mm)
    start_date = particles["date"].min()
    end_date = particles["date"].max()
    observations = prepare_observations_for_pbs(filtered_full, start_date, end_date)
    observations["date_dt"] = pd.to_datetime(observations["date"])

    model = joblib.load(args.mlp_model)
    feature_names = json.loads(args.mlp_features.read_text(encoding="utf-8"))
    mlp_metrics = json.loads(args.mlp_metrics.read_text(encoding="utf-8"))
    if "smap_sm_for_model" not in feature_names:
        raise RuntimeError("MLP feature list does not include smap_sm_for_model.")

    windows = weekly_windows(start_date, end_date, args.window_days)
    particle_ids = sorted(particles["particle"].unique())
    prior_weights = {particle: 1.0 / len(particle_ids) for particle in particle_ids}
    all_matches = []
    all_weights = []
    all_daily = []
    all_obs = []
    summary_rows = []
    warnings = []

    for window in windows:
        matches, scores, obs = score_window(window, particles, observations, overlap, model, feature_names, args.obs_sigma_m3m3, prior_weights)
        daily = summarize_window_irrigation(week_root, scores, window)
        ess = 1.0 / float(np.sum(scores["weight"] ** 2))
        max_weight = float(scores["weight"].max())
        best_particle = int(scores.sort_values("rank").iloc[0]["particle"])
        rmse = float(np.sqrt(np.mean(matches["residual_m3m3"] ** 2))) if len(matches) else np.nan
        quality_removed = int((obs["raw_keep"] & obs["low_quality_flag_removed"]).sum())
        threshold_removed = int(obs["threshold_abnormal_removed"].sum())
        kept = int(obs["final_keep"].sum())
        window_warnings = []
        if kept == 0:
            window_warnings.append("zero valid observations")
        if ess < args.low_ess_threshold:
            window_warnings.append("low ESS")
        if daily["posterior_mean_irrigation_mm"].fillna(0).abs().max() < 1e-3:
            window_warnings.append("posterior irrigation near zero")

        matches = add_window_columns(matches, window)
        scores = add_window_columns(scores, window)
        daily = add_window_columns(daily, window)
        obs = add_window_columns(obs, window)
        all_matches.append(matches)
        all_weights.append(scores)
        all_daily.append(daily)
        all_obs.append(obs)
        summary_rows.append(
            {
                "window_id": window["window_id"],
                "window_start": window["window_start"],
                "window_end": window["window_end"],
                "raw_observations": int(obs["raw_keep"].sum()),
                "kept_observations": kept,
                "quality_removed": quality_removed,
                "threshold_abnormal_removed": threshold_removed,
                "effective_sample_size": ess,
                "max_particle_weight": max_weight,
                "best_particle": best_particle,
                "residual_rmse_m3m3": rmse,
                "posterior_irrigation_sum_mm": float(daily["posterior_mean_irrigation_mm"].sum()),
                "prior_irrigation_sum_mm": float(daily["prior_mean_irrigation_mm"].sum()),
                "warnings": "; ".join(window_warnings),
            }
        )
        warnings.extend([f"Window {window['window_id']}: {w}" for w in window_warnings])
        prior_weights = dict(zip(scores["particle"], scores["weight"]))

    matches = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()
    weights = pd.concat(all_weights, ignore_index=True)
    daily = pd.concat(all_daily, ignore_index=True)
    obs_all = pd.concat(all_obs, ignore_index=True)
    window_summary = pd.DataFrame(summary_rows)

    obs_all.to_csv(out_dir / "filtered_satellite_observations.csv", index=False)
    matches.to_csv(out_dir / "pbs_particle_satellite_matches.csv", index=False)
    weights.to_csv(out_dir / "pbs_particle_weights.csv", index=False)
    daily.to_csv(out_dir / "posterior_daily_irrigation.csv", index=False)
    window_summary.to_csv(out_dir / "window_summary.csv", index=False)

    plot_outputs(out_dir, obs_all, matches, weights, daily, window_summary)

    metadata = {
        "run_date_time": datetime.now().isoformat(timespec="seconds"),
        "basin_name": args.basin_name,
        "run_type": "fresh_4week_sequential_pbs",
        "vic_ensemble_source_label": "fresh_4week_vic_ensemble_from_hopper",
        "fresh_vic_ensemble_outputs_used": True,
        "reused_week3_outputs_used_as_main_run": False,
        "vic_input_output_directory": str(week_root),
        "assimilation_variable": "absolute_satellite_soil_moisture",
        "delta_SM_assimilation": False,
        "adaptive_physical_filtering": False,
        "event_detector_likelihood": False,
        "broad_zaussinger_deletion": False,
        "filter_type": "narrow_quality_1_4_plus_existing_161_threshold_abnormal_effective_cap",
        "filter_summary": filter_summary,
        "four_week_observation_rows_before_filtering": int(len(obs_all)),
        "four_week_observation_rows_final_keep": int(obs_all["final_keep"].sum()),
        "four_week_threshold_abnormal_removed": int(obs_all["threshold_abnormal_removed"].sum()),
        "mlp_model_path": str(args.mlp_model.resolve()),
        "mlp_feature_columns": feature_names,
        "satellite_sm_included_as_mlp_feature": "smap_sm_for_model" in feature_names,
        "mlp_validation_rmse": safe_float(mlp_metrics.get("validation_rmse")),
        "mlp_validation_r2": safe_float(mlp_metrics.get("validation_r2")),
        "window_days": args.window_days,
        "window_count": len(windows),
        "date_range": {"start": start_date, "end": end_date},
        "particle_count": int(len(particle_ids)),
        "obs_sigma_m3m3": args.obs_sigma_m3m3,
        "total_posterior_irrigation_mm": float(daily["posterior_mean_irrigation_mm"].sum()),
        "total_prior_irrigation_mm": float(daily["prior_mean_irrigation_mm"].sum()),
        "posterior_irrigation_near_zero_all_days": bool(daily["posterior_mean_irrigation_mm"].fillna(0).abs().max() < 1e-3),
        "warnings": sorted(set(warnings)),
        "code_commit_hash": git_commit_hash(),
        "resampling_note": (
            "Sequential weekly reweighting was performed over propagated fresh VIC trajectories. "
            "Discrete resampling with VIC state branching would require Hopper reruns between windows."
        ),
        "verification": {
            "fresh_4week_vic_ensemble_used": True,
            "absolute_SM_likelihood_was_used": True,
            "delta_SM_assimilation_was_not_used": True,
            "satellite_SM_included_in_MLP_features": "smap_sm_for_model" in feature_names,
            "quality_classes_flags_1_4_removed": True,
            "threshold_detected_abnormal_observations_removed_or_warning_reported": True,
        },
    }
    (out_dir / "pbs_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(out_dir, metadata, window_summary)
    print(out_dir)


if __name__ == "__main__":
    main()
