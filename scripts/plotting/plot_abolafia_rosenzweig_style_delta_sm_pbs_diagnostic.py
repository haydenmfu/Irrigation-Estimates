#!/usr/bin/env python3
"""Build the Abolafia-Rosenzweig-style diagnostic for the HUC8 delta-SM PBS run."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = (
    PROJECT
    / "outputs"
    / "delta_sm_pre_pbs"
    / "huc8_10180009_weekly_vic_aligned"
    / "diagnostics"
    / "huc8_smap_l3_delta_sm_pbs_20260708_192928"
)
DEFAULT_OPEN_LOOP = (
    PROJECT
    / "data"
    / "VIC_basin0_outputs"
    / "dry_spottedtail_creek_vic_basin_daily_summary_for_satellite.csv"
)
PLOTTER = SCRIPT_DIR / "plot_abolafia_rosenzweig_style_pbs_diagnostic.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create the two-panel Abolafia-Rosenzweig-style plot for the HUC8 delta-SM PBS run."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--mask-class", default="cropland_like")
    parser.add_argument("--open-loop-sm", type=Path, default=DEFAULT_OPEN_LOOP)
    parser.add_argument("--precipitation", type=Path, default=DEFAULT_OPEN_LOOP)
    parser.add_argument("--precipitation-col", default="basin_mean_out_prec")
    parser.add_argument("--out-prefix", type=Path, default=None)
    parser.add_argument("--max-particles", type=int, default=None)
    parser.add_argument("--uncertainty", choices=["quantile", "std"], default="quantile")
    parser.add_argument("--cmap", default="Purples", help="Matplotlib colormap passed through to the Week 6 plotter.")
    parser.add_argument("--cmap-min", type=float, default=0.20, help="Lower colormap fraction passed through to the Week 6 plotter.")
    parser.add_argument("--cmap-max", type=float, default=1.00, help="Upper colormap fraction passed through to the Week 6 plotter.")
    return parser.parse_args()


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def particle_irrigation_path(run_dir):
    metadata_path = run_dir / "pbs_run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        week_root = metadata.get("week_root") or metadata.get("vic_input_output_directory")
        if week_root:
            path = Path(week_root) / "particle_irrigation_inputs.csv"
            if path.exists():
                return path
    raise FileNotFoundError("Could not resolve particle_irrigation_inputs.csv from run metadata.")


def build_endpoint_observation_series(run_dir, mask_class):
    pairs_path = run_dir / "huc8_smap_l3_weekly_delta_selected_pairs.csv"
    pairs = pd.read_csv(pairs_path)
    pairs = pairs[pairs["mask_class"].eq(mask_class)].copy()
    if pairs.empty:
        raise RuntimeError(f"No selected SMAP endpoint pairs found for mask_class={mask_class!r}")

    rows = []
    for endpoint, date_col, actual_col, sm_col, offset_col in [
        ("t0", "target_t0", "sat_t0", "SMAP_SM_t0", "sat_t0_offset_days"),
        ("t1", "target_t1", "sat_t1", "SMAP_SM_t1", "sat_t1_offset_days"),
    ]:
        for (window_id, target_date), group in pairs.groupby(["window_id", date_col]):
            rows.append(
                {
                    "window_id": int(window_id),
                    "endpoint": endpoint,
                    "date": target_date,
                    "soil_moisture": weighted_mean(group[sm_col], group["overlap_area_km2"]),
                    "n_pairs": int(len(group)),
                    "mean_abs_offset_days": float(group[offset_col].abs().mean()),
                    "actual_dates": ";".join(sorted(group[actual_col].astype(str).unique())),
                    "mask_class": mask_class,
                }
            )
    out = pd.DataFrame(rows).sort_values(["date", "endpoint", "window_id"])
    out_path = run_dir / f"huc8_smap_l3_{mask_class}_endpoint_sm_for_abolafia_plot.csv"
    out.to_csv(out_path, index=False)
    return out_path


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    obs_path = build_endpoint_observation_series(run_dir, args.mask_class)
    irrigation_path = particle_irrigation_path(run_dir)
    out_prefix = args.out_prefix or (run_dir / "abolafia_rosenzweig_style_delta_sm_pbs_diagnostic")

    command = [
        sys.executable,
        str(PLOTTER),
        "--run-dir",
        str(run_dir),
        "--particle-sm",
        str(run_dir / "particle_daily_basin_mean_sm.csv"),
        "--particle-sm-col",
        "particle_basin_SM_m3m3",
        "--weights",
        str(run_dir / "pbs_particle_weights.csv"),
        "--observations",
        str(obs_path),
        "--obs-sm-col",
        "soil_moisture",
        "--particle-irrigation",
        str(irrigation_path),
        "--window-summary",
        str(run_dir / "window_summary.csv"),
        "--open-loop-sm",
        str(args.open_loop_sm),
        "--open-loop-sm-col",
        "basin_mean_vic_soil_moist_layer1_m3m3",
        "--precipitation",
        str(args.precipitation),
        "--precipitation-col",
        args.precipitation_col,
        "--precipitation-label",
        "VIC forcing precipitation",
        "--out-prefix",
        str(out_prefix),
        "--title",
        "HUC8 SMAP-L3 Delta-SM PBS Diagnostic",
        "--uncertainty",
        args.uncertainty,
    ]
    if args.max_particles:
        command.extend(["--max-particles", str(args.max_particles)])
    if args.cmap:
        command.extend(["--cmap", args.cmap])
    if args.cmap_min is not None:
        command.extend(["--cmap-min", str(args.cmap_min)])
    if args.cmap_max is not None:
        command.extend(["--cmap-max", str(args.cmap_max)])

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
