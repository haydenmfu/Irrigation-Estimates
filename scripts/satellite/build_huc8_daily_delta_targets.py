#!/usr/bin/env python3
"""Build HUC8 daily within-window Delta-SM targets for Hopper PBS scoring."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
WEEK3 = PROJECT / "Week 3"
WEEK7 = PROJECT / "Week 7"

DEFAULT_OBSERVATIONS = (
    WEEK7
    / "outputs"
    / "seasonal_delta_sm_pbs"
    / "huc8_10180009_2020_may_sep"
    / "satellite"
    / "huc8_smap_l3_observations_2020_may_sep.csv"
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
DEFAULT_OUT = (
    WEEK7
    / "outputs"
    / "seasonal_delta_sm_pbs"
    / "huc8_10180009_2020_may_sep"
    / "targets"
    / "daily_delta_targets_for_hopper_2020_may_sep.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build daily HUC8 SMAP L3 Delta-SM targets for Hopper PBS.")
    parser.add_argument("--endpoint-observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--cdl-summary", type=Path, default=DEFAULT_CDL)
    parser.add_argument("--vic-basin-daily", type=Path, default=DEFAULT_VIC_DAILY)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cell-pairs-csv", type=Path, default=None)
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--start-date", default="2020-05-01")
    parser.add_argument("--end-date", default="2020-09-30")
    parser.add_argument("--cdl-year", type=int, default=2020)
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--max-pair-gap-days", type=int, default=3)
    parser.add_argument("--weight-mask-class", default="cropland_like")
    parser.add_argument("--control-mask-class", default="noncropland_control")
    parser.add_argument("--quality-mode", choices=["recommended", "all_finite"], default="recommended")
    return parser.parse_args()


def bool_series(values):
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


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


def build_cell_pairs(args, windows):
    obs = pd.read_csv(args.endpoint_observations, parse_dates=["date"])
    if "retrieval_recommended" not in obs.columns:
        raise ValueError("Observation table missing retrieval_recommended column.")
    obs["retrieval_recommended_bool"] = bool_series(obs["retrieval_recommended"])
    obs = obs[obs["soil_moisture"].notna()].copy()
    if args.quality_mode == "recommended":
        obs = obs[obs["retrieval_recommended_bool"]].copy()
    obs = obs[(obs["date"] >= pd.Timestamp(args.start_date)) & (obs["date"] <= pd.Timestamp(args.end_date))].copy()

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
    missing = sorted(set(cdl_cols) - set(cdl.columns))
    if missing:
        raise ValueError("CDL summary missing columns: {}".format(missing))
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
                    "smap_cell_id": a.get("smap_cell_id", "{}_{}".format(row, col)),
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
    return pd.DataFrame(pair_rows)


def aggregate_targets(args, pairs):
    if pairs.empty:
        return pd.DataFrame()
    agg_rows = []
    group_cols = ["window_id", "window_start", "window_end", "interval_start", "interval_end", "period", "mask_class"]
    for keys, group in pairs.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["interval_days"] = int((pd.Timestamp(row["interval_end"]) - pd.Timestamp(row["interval_start"])).days)
        row["n_pairs"] = int(len(group))
        row["area_km2"] = float(group["overlap_area_km2"].sum())
        row["delta_SM_satellite_area_weighted"] = weighted_mean(group["delta_SM_satellite"], group["overlap_area_km2"])
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
    return pd.DataFrame(target_rows).sort_values(["window_id", "interval_start", "interval_end", "period"])


def add_vic_reference(targets, vic_daily_path):
    if targets.empty:
        return targets
    daily = pd.read_csv(vic_daily_path, parse_dates=["time"]).set_index("time")
    rows = []
    for _, row in targets.iterrows():
        t0 = pd.Timestamp(row["interval_start"])
        t1 = pd.Timestamp(row["interval_end"])
        sm0 = daily.loc[t0, "basin_mean_vic_soil_moist_layer1_m3m3"]
        sm1 = daily.loc[t1, "basin_mean_vic_soil_moist_layer1_m3m3"]
        rows.append(
            {
                **row.to_dict(),
                "basin0_open_loop_SM_t0": float(sm0),
                "basin0_open_loop_SM_t1": float(sm1),
                "basin0_open_loop_delta_SM": float(sm1 - sm0),
                "basin0_vic_precip_sum_mm": float(
                    daily.loc[(daily.index > t0) & (daily.index <= t1), "basin_mean_out_prec"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    windows = weekly_windows(args.start_date, args.end_date, args.window_days)
    pairs = build_cell_pairs(args, windows)
    targets = aggregate_targets(args, pairs)
    targets = add_vic_reference(targets, args.vic_basin_daily)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(args.out_csv, index=False)
    cell_pairs_csv = args.cell_pairs_csv or args.out_csv.with_name(args.out_csv.stem + "_cell_pairs.csv")
    pairs.to_csv(cell_pairs_csv, index=False)

    metadata = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "window_days": args.window_days,
        "max_pair_gap_days": args.max_pair_gap_days,
        "quality_mode": args.quality_mode,
        "n_windows": len(windows),
        "n_cell_pairs": int(len(pairs)),
        "n_target_rows": int(len(targets)),
        "target_rows_by_window": targets.groupby("window_id").size().astype(int).to_dict() if len(targets) else {},
        "out_csv": str(args.out_csv),
        "cell_pairs_csv": str(cell_pairs_csv),
    }
    metadata_json = args.metadata_json or args.out_csv.with_suffix(".metadata.json")
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
