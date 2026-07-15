#!/usr/bin/env python3
"""Build a full-season Delta-SM PBS diagnostic with purple weights and rainfall."""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
PLOTTER = SCRIPT_DIR / "plot_abolafia_rosenzweig_style_pbs_diagnostic.py"

DEFAULT_RUN_ROOT = (
    PROJECT
    / "data"
    / "season_delta_resample_rerun"
    / "season_2020_may_sep_N100"
    / "extracted"
    / "season_daily_deltaSM_resample_rerun_basin0_20200501_20200930_N100"
)
DEFAULT_SMAP_OBS = (
    PROJECT
    / "outputs"
    / "seasonal_delta_sm_pbs"
    / "huc8_10180009_2020_may_sep"
    / "satellite"
    / "huc8_smap_l3_observations_2020_may_sep.csv"
)
DEFAULT_CDL_SUMMARY = (
    PROJECT
    / "outputs"
    / "delta_sm_pre_pbs"
    / "huc8_10180009_cdl_crop30_power_rain_partial_current"
    / "huc8_smap_cell_cropland_summary.csv"
)
DEFAULT_OPEN_LOOP = (
    PROJECT
    / "data"
    / "VIC_basin0_outputs"
    / "dry_spottedtail_creek_vic_basin_daily_summary_for_satellite.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create the full-season purple PBS diagnostic with rainfall on the right y-axis."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--smap-observations", type=Path, default=DEFAULT_SMAP_OBS)
    parser.add_argument("--cdl-summary", type=Path, default=DEFAULT_CDL_SUMMARY)
    parser.add_argument("--mask-class", default="cropland_like")
    parser.add_argument("--open-loop-sm", type=Path, default=DEFAULT_OPEN_LOOP)
    parser.add_argument("--precipitation", type=Path, default=DEFAULT_OPEN_LOOP)
    parser.add_argument("--out-prefix", type=Path, default=None)
    parser.add_argument("--max-particles", type=int, default=None)
    return parser.parse_args()


def require_path(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def window_id_from_postprocess_dir(path):
    # Directory names are window_1, window_2, ...
    return int(path.name.split("_", 1)[1])


def combine_window_particle_sm(run_root, out_dir):
    frames = []
    for window_dir in sorted((run_root / "postprocess").glob("window_*"), key=window_id_from_postprocess_dir):
        path = window_dir / "particle_daily_basin_mean_sm.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["window_id"] = window_id_from_postprocess_dir(window_dir)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No particle_daily_basin_mean_sm.csv files found under {run_root / 'postprocess'}")
    out = pd.concat(frames, ignore_index=True)
    out_path = out_dir / "season_particle_daily_basin_mean_sm_for_diagnostic.csv"
    out.to_csv(out_path, index=False)
    return out_path


def build_cropland_smap_series(smap_path, cdl_path, mask_class, out_dir):
    smap = pd.read_csv(smap_path)
    cdl = pd.read_csv(cdl_path)
    cdl = cdl[cdl["year"].astype(str).eq("2020")].copy()
    cdl = cdl[cdl["mask_class"].eq(mask_class)].copy()
    if cdl.empty:
        raise RuntimeError(f"No CDL cells found for mask_class={mask_class!r}")

    needed = cdl[["smap_cell_id", "mask_class", "overlap_area_km2"]].drop_duplicates("smap_cell_id")
    merged = smap.merge(needed, on="smap_cell_id", how="inner", suffixes=("", "_cdl"))
    if "retrieval_recommended" in merged.columns:
        recommended = merged["retrieval_recommended"].astype(str).str.lower().isin(["true", "1", "1.0"])
        merged = merged[recommended].copy()
    merged["soil_moisture"] = pd.to_numeric(merged["soil_moisture"], errors="coerce")
    merged["overlap_area_km2_cdl"] = pd.to_numeric(merged["overlap_area_km2_cdl"], errors="coerce")
    merged = merged.dropna(subset=["date", "soil_moisture", "overlap_area_km2_cdl"])
    merged = merged[merged["overlap_area_km2_cdl"] > 0].copy()
    if merged.empty:
        raise RuntimeError("No finite recommended cropland SMAP observations available after CDL join.")

    rows = []
    for (date, period), group in merged.groupby(["date", "period"], dropna=False):
        weights = group["overlap_area_km2_cdl"].to_numpy(dtype=float)
        values = group["soil_moisture"].to_numpy(dtype=float)
        rows.append(
            {
                "date": date,
                "period": period,
                "soil_moisture": float(np.average(values, weights=weights)),
                "n_cells": int(group["smap_cell_id"].nunique()),
                "area_km2": float(np.nansum(weights)),
                "mask_class": mask_class,
            }
        )
    out = pd.DataFrame(rows).sort_values(["date", "period"])
    out_path = out_dir / f"season_smap_l3_{mask_class}_sm_for_diagnostic.csv"
    out.to_csv(out_path, index=False)
    return out_path


def main():
    args = parse_args()
    run_root = require_path(args.run_root).resolve()
    local_summary = require_path(run_root / "local_summary")
    out_prefix = args.out_prefix or (local_summary / "abolafia_rosenzweig_style_season_delta_sm_pbs_diagnostic")

    particle_sm = combine_window_particle_sm(run_root, local_summary)
    smap_obs = build_cropland_smap_series(
        require_path(args.smap_observations),
        require_path(args.cdl_summary),
        args.mask_class,
        local_summary,
    )

    command = [
        sys.executable,
        str(PLOTTER),
        "--run-dir",
        str(run_root),
        "--particle-sm",
        str(particle_sm),
        "--particle-sm-col",
        "particle_basin_SM_m3m3",
        "--weights",
        str(local_summary / "particle_weights_combined.csv"),
        "--observations",
        str(smap_obs),
        "--obs-sm-col",
        "soil_moisture",
        "--particle-irrigation",
        str(local_summary / "posterior_daily_irrigation_combined.csv"),
        "--window-summary",
        str(local_summary / "window_summary_combined.csv"),
        "--open-loop-sm",
        str(args.open_loop_sm),
        "--open-loop-sm-col",
        "basin_mean_vic_soil_moist_layer1_m3m3",
        "--precipitation",
        str(args.precipitation),
        "--precipitation-col",
        "basin_mean_out_prec",
        "--precipitation-label",
        "VIC forcing precipitation",
        "--out-prefix",
        str(out_prefix),
        "--title",
        "Seasonal SMAP-L3 Delta-SM PBS Diagnostic",
        "--cmap",
        "Purples",
        "--cmap-min",
        "0.22",
        "--cmap-max",
        "1.0",
        "--weight-color-scope",
        "window",
        "--weight-color-scale",
        "auto",
        "--weight-vmax-percentile",
        "98.5",
        "--min-particle-alpha",
        "0.46",
        "--max-particle-alpha",
        "0.88",
        "--min-particle-linewidth",
        "0.45",
        "--max-particle-linewidth",
        "1.20",
        "--highlight-min-scaled-weight",
        "0.82",
        "--highlight-linewidth-boost",
        "0.55",
        "--highlight-alpha",
        "0.96",
    ]
    if args.max_particles:
        command.extend(["--max-particles", str(args.max_particles)])

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
