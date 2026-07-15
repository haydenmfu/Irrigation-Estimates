import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge VIC ET and observed satellite ET into calibration-ready daily/monthly/seasonal tables."
    )
    parser.add_argument("--vic-daily", required=True, help="Output from export_vic_et_features.py.")
    parser.add_argument("--observed-daily", required=True, help="Observed ET daily CSV, e.g. SSEBop output.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--basin-name", default="dry_spottedtail_creek")
    parser.add_argument("--irrigation-months", default="5,6,7,8,9")
    return parser.parse_args()


def rmse(obs, sim):
    valid = np.isfinite(obs) & np.isfinite(sim)
    if not valid.any():
        return np.nan
    return float(np.sqrt(np.mean((sim[valid] - obs[valid]) ** 2)))


def mae(obs, sim):
    valid = np.isfinite(obs) & np.isfinite(sim)
    if not valid.any():
        return np.nan
    return float(np.mean(np.abs(sim[valid] - obs[valid])))


def bias(obs, sim):
    valid = np.isfinite(obs) & np.isfinite(sim)
    if not valid.any():
        return np.nan
    return float(np.mean(sim[valid] - obs[valid]))


def nse(obs, sim):
    valid = np.isfinite(obs) & np.isfinite(sim)
    if valid.sum() < 2:
        return np.nan
    obs_v = obs[valid]
    sim_v = sim[valid]
    denom = np.sum((obs_v - np.mean(obs_v)) ** 2)
    if denom == 0:
        return np.nan
    return float(1 - np.sum((sim_v - obs_v) ** 2) / denom)


def kge(obs, sim):
    valid = np.isfinite(obs) & np.isfinite(sim)
    if valid.sum() < 2:
        return np.nan
    obs_v = obs[valid]
    sim_v = sim[valid]
    if np.std(obs_v) == 0 or np.std(sim_v) == 0 or np.mean(obs_v) == 0:
        return np.nan
    r = np.corrcoef(obs_v, sim_v)[0, 1]
    alpha = np.std(sim_v) / np.std(obs_v)
    beta = np.mean(sim_v) / np.mean(obs_v)
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def metrics(group):
    obs = group["observed_et_mm"].to_numpy(dtype=float)
    sim = group["vic_et_mm"].to_numpy(dtype=float)
    return pd.Series(
        {
            "n_pairs": int((np.isfinite(obs) & np.isfinite(sim)).sum()),
            "rmse_mm": rmse(obs, sim),
            "mae_mm": mae(obs, sim),
            "bias_mm": bias(obs, sim),
            "nse": nse(obs, sim),
            "kge": kge(obs, sim),
            "obs_total_mm": float(np.nansum(obs)),
            "vic_total_mm": float(np.nansum(sim)),
        }
    )


def metrics_by_season(frame):
    rows = []
    for season, group in frame.groupby("season"):
        row = metrics(group).to_dict()
        row["season"] = season
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    irrigation_months = {int(x) for x in args.irrigation_months.split(",") if x.strip()}

    vic = pd.read_csv(args.vic_daily)
    obs = pd.read_csv(args.observed_daily)

    if "date" not in vic and "time" in vic:
        vic["date"] = pd.to_datetime(vic["time"]).dt.date.astype(str)
    if "date" not in vic:
        raise SystemExit("VIC daily table must contain either 'date' or 'time'.")
    vic["date"] = pd.to_datetime(vic["date"]).dt.date.astype(str)
    obs["date"] = pd.to_datetime(obs["date"]).dt.date.astype(str)

    if "basin_mean_vic_et_total_mm" not in vic:
        if "basin_mean_out_evap" in vic:
            vic["basin_mean_vic_et_total_mm"] = vic["basin_mean_out_evap"]
        else:
            raise SystemExit(
                "VIC daily table must contain basin_mean_vic_et_total_mm or basin_mean_out_evap."
            )
    if "observed_et_mm" not in obs:
        raise SystemExit("Observed daily table must contain observed_et_mm.")

    keep_vic = [
        "date",
        "year",
        "month",
        "season",
        "basin_mean_vic_et_total_mm",
        "basin_mean_out_prec",
        "basin_mean_out_runoff",
        "basin_mean_out_baseflow",
        "n_vic_cells",
    ]
    keep_vic = [col for col in keep_vic if col in vic]
    merged = vic[keep_vic].merge(
        obs[["date", "observed_et_mm", "n_pixels", "product", "download_status"]],
        on="date",
        how="inner",
    )
    merged = merged.rename(columns={"basin_mean_vic_et_total_mm": "vic_et_mm"})
    merged["month"] = pd.to_datetime(merged["date"]).dt.month
    merged["year"] = pd.to_datetime(merged["date"]).dt.year
    merged["season"] = np.where(merged["month"].isin(irrigation_months), "irrigation", "non_irrigation")
    merged["et_error_mm"] = merged["vic_et_mm"] - merged["observed_et_mm"]

    monthly = (
        merged.groupby(["year", "month", "season"])
        .agg(
            {
                "vic_et_mm": "sum",
                "observed_et_mm": ["sum", "count"],
                "n_pixels": "mean",
            }
        )
        .reset_index()
    )
    monthly.columns = [
        "year",
        "month",
        "season",
        "vic_et_mm",
        "observed_et_mm",
        "n_pairs",
        "mean_n_pixels",
    ]
    monthly["et_error_mm"] = monthly["vic_et_mm"] - monthly["observed_et_mm"]

    seasonal = (
        monthly.groupby(["year", "season"])
        .agg({"vic_et_mm": "sum", "observed_et_mm": "sum", "n_pairs": "sum"})
        .reset_index()
    )
    seasonal["et_error_mm"] = seasonal["vic_et_mm"] - seasonal["observed_et_mm"]

    daily_metrics = metrics_by_season(merged)
    monthly_metrics = metrics_by_season(monthly)
    seasonal_metrics = metrics_by_season(seasonal)
    all_metrics = pd.concat(
        [
            daily_metrics.assign(timestep="daily"),
            monthly_metrics.assign(timestep="monthly"),
            seasonal_metrics.assign(timestep="seasonal"),
        ],
        ignore_index=True,
    )

    prefix = args.basin_name
    paths = {
        "daily": output_dir / f"{prefix}_et_calibration_daily.csv",
        "monthly": output_dir / f"{prefix}_et_calibration_monthly.csv",
        "seasonal": output_dir / f"{prefix}_et_calibration_seasonal.csv",
        "metrics": output_dir / f"{prefix}_et_calibration_metrics.csv",
        "metadata": output_dir / f"{prefix}_et_calibration_metadata.json",
    }
    merged.to_csv(paths["daily"], index=False)
    monthly.to_csv(paths["monthly"], index=False)
    seasonal.to_csv(paths["seasonal"], index=False)
    all_metrics.to_csv(paths["metrics"], index=False)
    paths["metadata"].write_text(
        json.dumps(
            {
                "vic_daily": str(args.vic_daily),
                "observed_daily": str(args.observed_daily),
                "irrigation_months": sorted(irrigation_months),
                "recommended_optimizer_objectives": [
                    "irrigation daily or monthly RMSE",
                    "irrigation seasonal bias",
                    "non-irrigation daily or monthly RMSE",
                ],
                "notes": [
                    "The optimizer should minimize RMSE and absolute bias.",
                    "KGE/NSE are reported for diagnostics but can be unstable with small seasonal samples.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for path in paths.values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
