#!/usr/bin/env python3
"""AdaPBS-style pooled postprocessing for Week 6.

This pools the N=100 initial proposal and the N=100 adapted proposal, then
scores weekly absolute-SM likelihoods over the combined candidate set. It is an
AdaPBS-inspired two-proposal diagnostic, not a complete implementation of the
full AdaPBS/AMIS paper.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
WEEK6 = PROJECT / "Week 6"
sys.path.insert(0, str(WEEK6))

from run_pbs_4week_sequential_absolute_sm import (  # noqa: E402
    DEFAULT_BASIN,
    DEFAULT_FILTER_DIAGNOSTICS,
    DEFAULT_MLP_FEATURES,
    DEFAULT_MLP_METRICS,
    DEFAULT_MLP_MODEL,
    DEFAULT_SAT_CELLS,
    add_window_columns,
    apply_observation_filter,
    build_overlap_map,
    git_commit_hash,
    joblib,
    normalize_log_weights,
    plot_outputs,
    prepare_observations_for_pbs,
    read_particle_vic_cells,
    read_vic_grid,
    safe_float,
    score_particles_absolute_sm,
    weekly_windows,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run pooled two-round AdaPBS-style postprocessing.")
    parser.add_argument("--round1-root", type=Path, required=True)
    parser.add_argument("--round2-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=WEEK6 / "outputs" / "adapbs_4week_N100_pooled")
    parser.add_argument("--basin-name", default="dry_spottedtail_creek")
    parser.add_argument("--satellite-cells", type=Path, default=DEFAULT_SAT_CELLS)
    parser.add_argument("--basin-geojson", type=Path, default=DEFAULT_BASIN)
    parser.add_argument("--filter-diagnostics", type=Path, default=DEFAULT_FILTER_DIAGNOSTICS)
    parser.add_argument("--mlp-model", type=Path, default=DEFAULT_MLP_MODEL)
    parser.add_argument("--mlp-features", type=Path, default=DEFAULT_MLP_FEATURES)
    parser.add_argument("--mlp-metrics", type=Path, default=DEFAULT_MLP_METRICS)
    parser.add_argument("--layer1-depth-mm", type=float, default=50.0)
    parser.add_argument("--min-overlap-km2", type=float, default=0.001)
    parser.add_argument("--expected-threshold-count", type=int, default=161)
    parser.add_argument("--obs-sigma-m3m3", type=float, default=0.035)
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--low-ess-threshold", type=float, default=10.0)
    return parser.parse_args()


def read_round(root, proposal_round, particle_offset, layer1_depth_mm):
    particles = read_particle_vic_cells(root, layer1_depth_mm).copy()
    particles["original_particle"] = particles["particle"].astype(int)
    particles["proposal_round"] = proposal_round
    particles["particle"] = particles["original_particle"] + particle_offset

    irrigation = pd.read_csv(root / "particle_irrigation_inputs.csv")
    irrigation["original_particle"] = irrigation["particle"].astype(int)
    irrigation["proposal_round"] = proposal_round
    irrigation["particle"] = irrigation["original_particle"] + particle_offset
    return particles, irrigation


def score_window(window, particles, observations, overlap, model, feature_names, sigma, prior_weights):
    obs = observations[(observations["date"] >= window["window_start"]) & (observations["date"] <= window["window_end"])].copy()
    kept = obs[obs["final_keep"]].copy()
    particle_ids = sorted(particles["particle"].unique())
    if kept.empty:
        scores = pd.DataFrame({"particle": particle_ids})
        scores["prior_weight"] = scores["particle"].map(prior_weights).fillna(1.0 / len(particle_ids))
        scores["n_cell_observations"] = 0
        scores["log_likelihood"] = 0.0
        scores["log_posterior_unnormalized"] = np.log(np.maximum(scores["prior_weight"], 1e-300))
        scores["weight"] = scores["prior_weight"]
        scores["rank"] = scores["weight"].rank(ascending=False, method="first").astype(int)
        return pd.DataFrame(), scores.sort_values("rank"), obs

    matches, _ = score_particles_absolute_sm(particles, obs, overlap, model, feature_names, sigma)
    grouped = matches.groupby("particle", as_index=False).agg(
        n_cell_observations=("log_likelihood", "size"),
        log_likelihood=("log_likelihood", "sum"),
        mean_abs_residual_m3m3=("residual_m3m3", lambda x: float(np.mean(np.abs(x)))),
        rmse_residual_m3m3=("residual_m3m3", lambda x: float(np.sqrt(np.mean(x**2)))),
    )
    scores = pd.DataFrame({"particle": particle_ids}).merge(grouped, on="particle", how="left")
    scores["n_cell_observations"] = scores["n_cell_observations"].fillna(0).astype(int)
    scores["log_likelihood"] = scores["log_likelihood"].fillna(0.0)
    scores["mean_abs_residual_m3m3"] = scores["mean_abs_residual_m3m3"].fillna(np.nan)
    scores["rmse_residual_m3m3"] = scores["rmse_residual_m3m3"].fillna(np.nan)
    scores["prior_weight"] = scores["particle"].map(prior_weights).fillna(1.0 / len(particle_ids))
    scores["log_posterior_unnormalized"] = np.log(np.maximum(scores["prior_weight"], 1e-300)) + scores["log_likelihood"]
    scores["weight"] = normalize_log_weights(scores["log_posterior_unnormalized"].to_numpy())
    scores["rank"] = scores["weight"].rank(ascending=False, method="first").astype(int)
    return matches, scores.sort_values("rank"), obs


def summarize_irrigation(irrigation, scores, window):
    sub = irrigation[(irrigation["date"] >= window["window_start"]) & (irrigation["date"] <= window["window_end"])].copy()
    weighted = sub.merge(scores[["particle", "weight"]], on="particle", how="left")
    weighted["weighted_irrigation_mm_day"] = weighted["irrigation_mm_day"] * weighted["weight"]
    daily = weighted.groupby("date", as_index=False).agg(
        posterior_mean_irrigation_mm=("weighted_irrigation_mm_day", "sum"),
        prior_mean_irrigation_mm=("irrigation_mm_day", "mean"),
        max_particle_irrigation_mm=("irrigation_mm_day", "max"),
    )
    daily["window_id"] = window["window_id"]
    return daily


def write_report(out_dir, metadata, window_summary):
    lines = [
        "# Week 6 AdaPBS-Style N=100 + N=100 Pooled Run",
        "",
        "This is a two-round AdaPBS-inspired diagnostic. Round 1 uses the initial N=100 proposal. Round 2 uses an adapted N=100 proposal generated from the round-1 posterior irrigation signal. The postprocessor pools both rounds and performs weekly absolute-SM weighting over 200 candidate trajectories.",
        "",
        "Important limitation: this is not the full AdaPBS/AMIS algorithm from the paper because it does not compute exact proposal-density mixture denominators for the adapted irrigation generator. It is a practical two-round adaptive proposal experiment for this VIC workflow.",
        "",
        "## Configuration",
        "",
        "- Assimilation variable: absolute satellite soil moisture.",
        "- Delta-SM assimilation: disabled.",
        "- MLP includes satellite SM feature `smap_sm_for_model`.",
        "- Filter: narrow quality 1-4 plus existing 161 threshold/effective-cap abnormal observations.",
        f"- Candidate trajectories: {metadata['particle_count']} pooled particles.",
        "",
        "## Window Diagnostics",
        "",
        "| Window | Dates | Kept obs | ESS | Max weight | Best particle | Best round | RMSE | Warnings |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in window_summary.iterrows():
        lines.append(
            f"| {int(row.window_id)} | {row.window_start} to {row.window_end} | {int(row.kept_observations)} | "
            f"{row.effective_sample_size:.2f} | {row.max_particle_weight:.3f} | {int(row.best_particle)} | "
            f"{int(row.best_proposal_round)} | {row.residual_rmse_m3m3:.4f} | {row.warnings} |"
        )
    lines.extend(
        [
            "",
            "## Irrigation Result",
            "",
            f"- Total posterior mean irrigation: {metadata['total_posterior_irrigation_mm']:.4f} mm",
            f"- Total pooled prior mean irrigation: {metadata['total_prior_irrigation_mm']:.4f} mm",
            "",
            "## Weighting Formula Note",
            "",
            "The implemented pooled diagnostic uses weekly weights proportional to prior_weight * likelihood, with Gaussian absolute-SM likelihood. The report intentionally labels this as AdaPBS-style rather than exact AdaPBS because exact AMIS-style deterministic-mixture proposal-density correction is not yet implemented.",
        ]
    )
    (out_dir / "summary_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / ("adapbs4week_N100plusN100_pooled_" + timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    round1_root = args.round1_root.resolve()
    round2_root = args.round2_root.resolve()
    if not round1_root.exists() or not round2_root.exists():
        raise FileNotFoundError("Both round roots must exist before pooled AdaPBS postprocessing.")

    p1, i1 = read_round(round1_root, 1, 0, args.layer1_depth_mm)
    p2, i2 = read_round(round2_root, 2, 100000, args.layer1_depth_mm)
    particles = pd.concat([p1, p2], ignore_index=True)
    irrigation = pd.concat([i1, i2], ignore_index=True)
    irrigation["date"] = pd.to_datetime(irrigation["date"]).dt.date.astype(str)

    filtered_full, filter_summary = apply_observation_filter(args.filter_diagnostics, args.expected_threshold_count)
    vic_grid = read_vic_grid(round1_root)
    overlap = build_overlap_map(vic_grid, args.satellite_cells.resolve(), args.basin_geojson.resolve(), args.min_overlap_km2)
    overlap.drop(columns="geometry", errors="ignore").to_csv(out_dir / "cellwise_satellite_vic_overlap_map.csv", index=False)

    start_date = particles["date"].min()
    end_date = particles["date"].max()
    observations = prepare_observations_for_pbs(filtered_full, start_date, end_date)
    model = joblib.load(args.mlp_model)
    feature_names = json.loads(args.mlp_features.read_text(encoding="utf-8"))
    mlp_metrics = json.loads(args.mlp_metrics.read_text(encoding="utf-8"))

    particle_ids = sorted(particles["particle"].unique())
    # Equal mixture over the two proposal rounds at the start.
    prior_weights = {p: 1.0 / len(particle_ids) for p in particle_ids}
    windows = weekly_windows(start_date, end_date, args.window_days)

    all_matches, all_scores, all_daily, all_obs, summary_rows, warnings = [], [], [], [], [], []
    for window in windows:
        matches, scores, obs = score_window(window, particles, observations, overlap, model, feature_names, args.obs_sigma_m3m3, prior_weights)
        scores = scores.merge(
            particles[["particle", "proposal_round", "original_particle"]].drop_duplicates(),
            on="particle",
            how="left",
        )
        daily = summarize_irrigation(irrigation, scores, window)
        ess = 1.0 / float(np.sum(scores["weight"] ** 2))
        max_weight = float(scores["weight"].max())
        best = scores.sort_values("rank").iloc[0]
        rmse = float(np.sqrt(np.mean(matches["residual_m3m3"] ** 2))) if len(matches) else np.nan
        window_warnings = []
        if int(obs["final_keep"].sum()) == 0:
            window_warnings.append("zero valid observations")
        if ess < args.low_ess_threshold:
            window_warnings.append("low ESS")
        if daily["posterior_mean_irrigation_mm"].fillna(0).abs().max() < 1e-3:
            window_warnings.append("posterior irrigation near zero")

        all_matches.append(add_window_columns(matches, window))
        all_scores.append(add_window_columns(scores, window))
        all_daily.append(add_window_columns(daily, window))
        all_obs.append(add_window_columns(obs, window))
        summary_rows.append(
            {
                "window_id": window["window_id"],
                "window_start": window["window_start"],
                "window_end": window["window_end"],
                "raw_observations": int(obs["raw_keep"].sum()),
                "kept_observations": int(obs["final_keep"].sum()),
                "quality_removed": int((obs["raw_keep"] & obs["low_quality_flag_removed"]).sum()),
                "threshold_abnormal_removed": int(obs["threshold_abnormal_removed"].sum()),
                "effective_sample_size": ess,
                "max_particle_weight": max_weight,
                "best_particle": int(best["particle"]),
                "best_original_particle": int(best["original_particle"]),
                "best_proposal_round": int(best["proposal_round"]),
                "residual_rmse_m3m3": rmse,
                "posterior_irrigation_sum_mm": float(daily["posterior_mean_irrigation_mm"].sum()),
                "prior_irrigation_sum_mm": float(daily["prior_mean_irrigation_mm"].sum()),
                "warnings": "; ".join(window_warnings),
            }
        )
        warnings.extend(["Window {}: {}".format(window["window_id"], w) for w in window_warnings])
        prior_weights = dict(zip(scores["particle"], scores["weight"]))

    matches = pd.concat(all_matches, ignore_index=True)
    weights = pd.concat(all_scores, ignore_index=True)
    daily = pd.concat(all_daily, ignore_index=True)
    obs_all = pd.concat(all_obs, ignore_index=True)
    window_summary = pd.DataFrame(summary_rows)

    obs_all.to_csv(out_dir / "filtered_satellite_observations.csv", index=False)
    matches.to_csv(out_dir / "adapbs_particle_satellite_matches.csv", index=False)
    weights.to_csv(out_dir / "adapbs_particle_weights.csv", index=False)
    daily.to_csv(out_dir / "posterior_daily_irrigation.csv", index=False)
    window_summary.to_csv(out_dir / "window_summary.csv", index=False)
    plot_outputs(out_dir, obs_all, matches, weights, daily, window_summary)

    metadata = {
        "run_date_time": datetime.now().isoformat(timespec="seconds"),
        "basin_name": args.basin_name,
        "run_type": "two_round_adapbs_style_pooled_absolute_sm",
        "round1_root": str(round1_root),
        "round2_root": str(round2_root),
        "particle_count": int(len(particle_ids)),
        "round1_particle_count": int(p1["particle"].nunique()),
        "round2_particle_count": int(p2["particle"].nunique()),
        "assimilation_variable": "absolute_satellite_soil_moisture",
        "delta_SM_assimilation": False,
        "filter_type": "narrow_quality_1_4_plus_existing_161_threshold_abnormal_effective_cap",
        "filter_summary": filter_summary,
        "mlp_model_path": str(args.mlp_model.resolve()),
        "mlp_feature_columns": feature_names,
        "satellite_sm_included_as_mlp_feature": "smap_sm_for_model" in feature_names,
        "mlp_validation_rmse": safe_float(mlp_metrics.get("validation_rmse")),
        "mlp_validation_r2": safe_float(mlp_metrics.get("validation_r2")),
        "obs_sigma_m3m3": args.obs_sigma_m3m3,
        "window_days": args.window_days,
        "window_count": len(windows),
        "total_posterior_irrigation_mm": float(daily["posterior_mean_irrigation_mm"].sum()),
        "total_prior_irrigation_mm": float(daily["prior_mean_irrigation_mm"].sum()),
        "warnings": sorted(set(warnings)),
        "code_commit_hash": git_commit_hash(),
        "adapbs_limitation": "Two-round adapted proposal diagnostic; exact AMIS deterministic-mixture proposal-density correction is not yet implemented.",
    }
    (out_dir / "pbs_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(out_dir, metadata, window_summary)
    print(out_dir)


if __name__ == "__main__":
    main()
