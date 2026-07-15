#!/usr/bin/env python3
"""HUC8-scale delta-SM diagnostic with CDL cropland/control masks.

This is the larger-domain continuation of the Dry Spottedtail 12-cell diagnostic.
It intentionally selects all NSIDC-0800 9 km cells intersecting the HUC8 boundary
instead of reusing the original basin cell list.
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as raster_mask
from shapely.geometry import box


PROJECT = Path(__file__).resolve().parents[1]
WEEK2 = PROJECT / "Week 2"
WEEK7 = PROJECT / "Week 7"

DEFAULT_BOUNDARY = WEEK7 / "outputs" / "domain_boundaries" / "dry_spottedtail_parent_huc8_10180009.geojson"
DEFAULT_DRY = PROJECT / "Week 1" / "Deliverables" / "Dry_Spottedtail_Creek_USGS_06679000" / "dry_spottedtail_creek.geojson"
DEFAULT_RAW = WEEK2 / "raw_smos_smap_9k"
DEFAULT_CDL = WEEK7 / "CDL_nebraska"
DEFAULT_OUT = WEEK7 / "outputs" / "delta_sm_pre_pbs" / "huc8_10180009_cdl_crop30"

CELL_SIZE_M = 9000
BBOX_PADDING_DEGREES = 0.5
NSIDC_PATHS = [
    ("SMAP_Based", "AM", "SMAP_Based/Soil_Moisture_Retrieval_Data_AM", "_am"),
    ("SMAP_Based", "PM", "SMAP_Based/Soil_Moisture_Retrieval_Data_PM", "_pm"),
    ("SMOS_Based", "AM", "SMOS_Based/Soil_Moisture_Retrieval_Data_AM", "_am"),
    ("SMOS_Based", "PM", "SMOS_Based/Soil_Moisture_Retrieval_Data_PM", "_pm"),
]

CDL_CROP_CLASSES = {
    1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 38, 39,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
    61, 66, 67, 68, 69, 70, 71, 72, 74, 75, 76, 77,
    204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214,
    216, 217, 218, 219, 220, 221, 222, 223, 224, 225,
    226, 227, 228, 229, 230, 231, 232, 233, 234, 235,
    236, 237, 238, 239, 240, 241, 242, 243, 244, 245,
    246, 247, 248, 249, 250, 254,
}
CDL_EXCLUDE_CLASSES = {0, 81, 82, 83, 87, 88, 111, 112, 121, 122, 123, 124, 131, 190, 195}


def parse_args():
    parser = argparse.ArgumentParser(description="Run HUC8 CDL/rainfall delta-SM diagnostic.")
    parser.add_argument("--huc8-boundary", type=Path, default=DEFAULT_BOUNDARY)
    parser.add_argument("--dry-spottedtail-boundary", type=Path, default=DEFAULT_DRY)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--observations-csv", type=Path, default=None, help="Optional pre-extracted HUC8 satellite observation CSV.")
    parser.add_argument("--cdl-dir", type=Path, default=DEFAULT_CDL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--end-date", default="2021-12-31")
    parser.add_argument("--source", default="SMAP_Based", choices=["SMAP_Based", "SMOS_Based"])
    parser.add_argument("--period", default="AM", choices=["AM", "PM"])
    parser.add_argument("--crop-threshold", type=float, default=0.30)
    parser.add_argument("--control-threshold", type=float, default=0.10)
    parser.add_argument("--exclude-threshold", type=float, default=0.50)
    parser.add_argument("--sensitivity-thresholds", default="0.20,0.30,0.50")
    parser.add_argument("--low-rain-threshold-mm", type=float, default=1.0)
    parser.add_argument("--positive-delta-threshold", type=float, default=0.02)
    parser.add_argument("--max-pair-gap-days", type=float, default=10.0)
    parser.add_argument("--rainfall-csv", type=Path, default=None, help="Optional rainfall table with date,row,col,precip_mm.")
    return parser.parse_args()


def parse_date_from_filename(path):
    match = re.search(r"_(\d{8})_", Path(path).name)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d").date().isoformat()


def read_array(h5, key):
    if key not in h5:
        return None
    dataset = h5[key]
    arr = np.asarray(dataset[:])
    fill = dataset.attrs.get("_FillValue", None)
    if fill is not None and np.issubdtype(arr.dtype, np.number):
        arr = arr.astype("float64", copy=False)
        for value in np.ravel(np.asarray(fill)):
            try:
                arr[arr == float(value)] = np.nan
            except (TypeError, ValueError):
                pass
    return arr


def estimate_utm_crs(boundary_wgs84):
    centroid = boundary_wgs84.geometry.iloc[0].centroid
    zone = int((centroid.x + 180) // 6) + 1
    return "EPSG:%d" % (32600 + zone if centroid.y >= 0 else 32700 + zone)


def load_boundary(path):
    frame = gpd.read_file(path)
    if frame.empty:
        raise ValueError("Boundary has no features: %s" % path)
    if frame.crs is None:
        frame = frame.set_crs("EPSG:4326")
    return frame.dissolve().reset_index(drop=True).to_crs("EPSG:4326")


def select_huc8_cells(boundary, raw_files):
    west, south, east, north = [float(x) for x in boundary.total_bounds]
    bbox = (west - BBOX_PADDING_DEGREES, south - BBOX_PADDING_DEGREES, east + BBOX_PADDING_DEGREES, north + BBOX_PADDING_DEGREES)
    centers = {}
    for file_path in raw_files[:5]:
        with h5py.File(file_path, "r") as h5:
            for _, _, group, suffix in NSIDC_PATHS:
                lat = read_array(h5, "%s/latitude%s" % (group, suffix))
                lon = read_array(h5, "%s/longitude%s" % (group, suffix))
                if lat is None or lon is None:
                    continue
                valid = np.isfinite(lat) & np.isfinite(lon) & (lat > -9000) & (lon > -9000)
                near = valid & (lon >= bbox[0]) & (lon <= bbox[2]) & (lat >= bbox[1]) & (lat <= bbox[3])
                rows, cols = np.where(near)
                for r, c in zip(rows, cols):
                    centers[(int(r), int(c))] = (float(lat[r, c]), float(lon[r, c]))
        if centers:
            break
    if not centers:
        raise RuntimeError("No NSIDC grid centers found near HUC8 boundary.")

    rows = [r for r, _ in centers]
    cols = [c for _, c in centers]
    lats = [v[0] for v in centers.values()]
    lons = [v[1] for v in centers.values()]
    points = gpd.GeoDataFrame({"row": rows, "col": cols, "lat": lats, "lon": lons}, geometry=gpd.points_from_xy(lons, lats), crs="EPSG:4326")
    area_crs = estimate_utm_crs(boundary)
    pts_area = points.to_crs(area_crs)
    half = CELL_SIZE_M / 2
    cells = gpd.GeoDataFrame(
        pts_area.drop(columns="geometry"),
        geometry=[box(p.x - half, p.y - half, p.x + half, p.y + half) for p in pts_area.geometry],
        crs=area_crs,
    )
    boundary_area = boundary.to_crs(area_crs)
    geom = boundary_area.geometry.iloc[0]
    area_km2 = geom.area / 1_000_000
    cells["cell_area_km2"] = cells.geometry.area / 1_000_000
    cells["overlap_area_km2"] = cells.geometry.intersection(geom).area / 1_000_000
    cells = cells[cells["overlap_area_km2"] > 0].copy()
    cells["percent_of_huc8"] = 100 * cells["overlap_area_km2"] / area_km2
    cells["percent_of_cell"] = 100 * cells["overlap_area_km2"] / cells["cell_area_km2"]
    cells["smap_cell_id"] = cells["row"].astype(int).astype(str) + "_" + cells["col"].astype(int).astype(str)
    return cells, cells.to_crs("EPSG:4326"), area_km2


def dominant_value(values):
    if len(values) == 0:
        return np.nan
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[np.argmax(counts)])


def summarize_cdl_for_cells(cells, cdl_path, year, crop_threshold, control_threshold, exclude_threshold):
    rows = []
    with rasterio.open(cdl_path) as src:
        cells_src = cells.to_crs(src.crs)
        for _, cell in cells_src.iterrows():
            try:
                data, _ = raster_mask(src, [cell.geometry], crop=True, filled=True, nodata=src.nodata)
            except ValueError:
                continue
            arr = data[0]
            valid = np.isfinite(arr)
            if src.nodata is not None:
                valid &= arr != src.nodata
            if not valid.any():
                continue
            classes = arr[valid].astype(int)
            total = float(len(classes))
            crop = np.isin(classes, list(CDL_CROP_CLASSES)).sum()
            excluded = np.isin(classes, list(CDL_EXCLUDE_CLASSES)).sum()
            noncrop = total - crop - excluded
            crop_fraction = crop / total
            excluded_fraction = excluded / total
            if excluded_fraction >= exclude_threshold:
                mask_class = "excluded"
            elif crop_fraction >= crop_threshold:
                mask_class = "cropland_like"
            elif crop_fraction <= control_threshold:
                mask_class = "noncropland_control"
            else:
                mask_class = "mixed"
            rows.append(
                {
                    "year": year,
                    "row": int(cell["row"]),
                    "col": int(cell["col"]),
                    "smap_cell_id": cell["smap_cell_id"],
                    "lat": float(cell["lat"]),
                    "lon": float(cell["lon"]),
                    "overlap_area_km2": float(cell["overlap_area_km2"]),
                    "crop_fraction": float(crop_fraction),
                    "noncropland_fraction": float(max(noncrop, 0) / total),
                    "excluded_fraction": float(excluded_fraction),
                    "dominant_cdl_class": dominant_value(classes),
                    "mask_class": mask_class,
                }
            )
    return pd.DataFrame(rows)


def extract_observations(raw_files, cells, source, period):
    group_lookup = {(s, p): (g, suf) for s, p, g, suf in NSIDC_PATHS}
    group, suffix = group_lookup[(source, period)]
    records = []
    selected = cells[["row", "col", "lat", "lon", "overlap_area_km2", "smap_cell_id"]].copy()
    for file_path in raw_files:
        date = parse_date_from_filename(file_path)
        if date is None:
            continue
        with h5py.File(file_path, "r") as h5:
            sm = read_array(h5, "%s/soil_moisture%s" % (group, suffix))
            filt = read_array(h5, "%s/soil_moisture_filtered%s" % (group, suffix))
            flags = read_array(h5, "%s/retrieval_qual_flag%s" % (group, suffix))
            if sm is None:
                continue
            for _, cell in selected.iterrows():
                r = int(cell["row"])
                c = int(cell["col"])
                value = float(sm[r, c]) if np.isfinite(sm[r, c]) else np.nan
                filtered = float(filt[r, c]) if filt is not None and np.isfinite(filt[r, c]) else np.nan
                flag = float(flags[r, c]) if flags is not None and np.isfinite(flags[r, c]) else np.nan
                finite = np.isfinite(value)
                rec = {
                    "date": date,
                    "source": source,
                    "period": period,
                    "row": r,
                    "col": c,
                    "smap_cell_id": cell["smap_cell_id"],
                    "smap_lat": float(cell["lat"]),
                    "smap_lon": float(cell["lon"]),
                    "overlap_area_km2": float(cell["overlap_area_km2"]),
                    "soil_moisture": value,
                    "soil_moisture_filtered": filtered,
                    "retrieval_qual_flag": flag,
                    "retrieval_recommended": bool(finite and np.isfinite(flag) and ((int(flag) & 1) != 0)),
                    "filtered_available": bool(np.isfinite(filtered)),
                    "source_file": file_path.name,
                }
                records.append(rec)
    return pd.DataFrame(records)


def load_rainfall(path):
    if path is None or not path.exists():
        return None
    rain = pd.read_csv(path)
    rain["date"] = pd.to_datetime(rain["date"])
    if "smap_cell_id" not in rain.columns and {"row", "col"}.issubset(rain.columns):
        rain["smap_cell_id"] = rain["row"].astype(int).astype(str) + "_" + rain["col"].astype(int).astype(str)
    return rain


def compute_pairs(obs, rainfall, low_rain_threshold, positive_delta_threshold, max_gap_days):
    obs = obs.copy()
    obs["date_dt"] = pd.to_datetime(obs["date"])
    obs = obs[obs["soil_moisture"].notna()].sort_values(["smap_cell_id", "date_dt"])
    rows = []
    for cell_id, group in obs.groupby("smap_cell_id"):
        group = group.reset_index(drop=True)
        for i in range(1, len(group)):
            prev = group.iloc[i - 1]
            cur = group.iloc[i]
            dt = (cur["date_dt"] - prev["date_dt"]).days
            if dt <= 0 or dt > max_gap_days:
                continue
            rainfall_sum = np.nan
            if rainfall is not None:
                sub = rainfall[(rainfall["smap_cell_id"].eq(cell_id)) & (rainfall["date"] > prev["date_dt"]) & (rainfall["date"] <= cur["date_dt"])]
                if len(sub):
                    rainfall_sum = float(sub["precip_mm"].fillna(0).sum())
            delta = float(cur["soil_moisture"] - prev["soil_moisture"])
            low_rain = (not np.isfinite(rainfall_sum)) or rainfall_sum <= low_rain_threshold
            rows.append(
                {
                    "smap_cell_id": cell_id,
                    "row": int(cur["row"]),
                    "col": int(cur["col"]),
                    "smap_lat": float(cur["smap_lat"]),
                    "smap_lon": float(cur["smap_lon"]),
                    "date_start": prev["date"],
                    "date_end": cur["date"],
                    "dt_days": dt,
                    "rainfall_sum": rainfall_sum,
                    "delta_SM_obs": delta,
                    "residual_low_rain": delta if low_rain else np.nan,
                    "positive_unexplained_wetting": bool(low_rain and delta >= positive_delta_threshold),
                }
            )
    return pd.DataFrame(rows)


def summary_by_mask(pairs):
    if pairs.empty:
        return pd.DataFrame()
    return pairs.groupby("mask_class", as_index=False).agg(
        n_pairs=("smap_cell_id", "size"),
        n_cells=("smap_cell_id", "nunique"),
        positive_unexplained_wetting=("positive_unexplained_wetting", "sum"),
        positive_unexplained_fraction=("positive_unexplained_wetting", "mean"),
        mean_delta_SM_obs=("delta_SM_obs", "mean"),
        mean_positive_delta_SM=("delta_SM_obs", lambda x: x[x > 0].mean()),
        mean_low_rain_delta_SM=("residual_low_rain", "mean"),
    )


def summary_by_cell(pairs):
    if pairs.empty:
        return pd.DataFrame()
    return pairs.groupby(["smap_cell_id", "row", "col", "smap_lat", "smap_lon", "mask_class"], as_index=False).agg(
        n_pairs=("smap_cell_id", "size"),
        positive_unexplained_wetting=("positive_unexplained_wetting", "sum"),
        positive_unexplained_frequency=("positive_unexplained_wetting", "mean"),
        mean_delta_SM_obs=("delta_SM_obs", "mean"),
        mean_positive_delta_SM=("delta_SM_obs", lambda x: x[x > 0].mean()),
    )


def plot_domain_map(cells, cdl_2020, huc8, dry, out_path):
    plot = cells.merge(cdl_2020[["smap_cell_id", "crop_fraction", "mask_class"]], on="smap_cell_id", how="left")
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    plot.plot(ax=ax, column="crop_fraction", cmap="YlGn", edgecolor="#4D4D4D", linewidth=0.25, legend=True, vmin=0, vmax=1)
    huc8.boundary.plot(ax=ax, color="black", linewidth=1.4)
    if dry is not None:
        dry.boundary.plot(ax=ax, color="#D95F02", linewidth=1.0)
    ax.set_title("HUC8 10180009 SMAP Cells and CDL Cropland Fraction")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_timeseries(event_summary, out_path):
    if event_summary.empty:
        return
    data = event_summary.copy()
    data["date_end"] = pd.to_datetime(data["date_end"])
    fig, ax = plt.subplots(figsize=(11, 5.6), constrained_layout=True)
    for col, color, label in [
        ("mean_delta_SM_cropland_like", "#2CA25F", "Cropland-like"),
        ("mean_delta_SM_noncropland_control", "#737373", "Non-cropland/control"),
        ("mean_delta_SM_all", "#0072B2", "All cells"),
    ]:
        if col in data:
            ax.plot(data["date_end"], data[col], marker="o", markersize=3.5, linewidth=1.2, color=color, label=label)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Mean delta_SM")
    ax.set_title("HUC8 Delta-SM Time Series")
    ax.legend(frameon=False, ncol=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_scatter(pairs, out_path):
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    colors = {"cropland_like": "#2CA25F", "noncropland_control": "#737373", "mixed": "#9ECAE1", "excluded": "#BDBDBD"}
    x = pairs["rainfall_sum"] if pairs["rainfall_sum"].notna().any() else np.zeros(len(pairs))
    xlabel = "Accumulated rainfall (mm)" if pairs["rainfall_sum"].notna().any() else "Rainfall unavailable (shown at 0)"
    for klass, group in pairs.groupby("mask_class"):
        gx = group["rainfall_sum"] if group["rainfall_sum"].notna().any() else np.zeros(len(group))
        ax.scatter(gx, group["delta_SM_obs"], s=16, alpha=0.55, color=colors.get(klass, "#0072B2"), label=klass)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("delta_SM_obs")
    ax.set_title("Accumulated Rainfall vs Satellite delta_SM")
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_histogram(pairs, out_path):
    fig, ax = plt.subplots(figsize=(8, 5.6), constrained_layout=True)
    colors = {"cropland_like": "#2CA25F", "noncropland_control": "#737373", "mixed": "#9ECAE1", "excluded": "#BDBDBD"}
    for klass, group in pairs.groupby("mask_class"):
        ax.hist(group["delta_SM_obs"].dropna(), bins=40, density=True, alpha=0.42, color=colors.get(klass, "#0072B2"), label=klass)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("delta_SM_obs")
    ax.set_ylabel("Density")
    ax.set_title("HUC8 delta_SM Distribution by Mask Class")
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_cell_map(cells, cell_summary, column, title, out_path):
    plot = cells.merge(cell_summary[["smap_cell_id", column]], on="smap_cell_id", how="left")
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    plot.plot(ax=ax, column=column, cmap="magma", edgecolor="#4D4D4D", linewidth=0.25, legend=True, missing_kwds={"color": "#F0F0F0"})
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_event_summary(pairs):
    rows = []
    for (start, end), group in pairs.groupby(["date_start", "date_end"], dropna=False):
        crop = group[group["mask_class"].eq("cropland_like")]
        control = group[group["mask_class"].eq("noncropland_control")]
        rows.append(
            {
                "date_start": start,
                "date_end": end,
                "rainfall_sum": group["rainfall_sum"].mean(),
                "mean_delta_SM_all": group["delta_SM_obs"].mean(),
                "mean_delta_SM_cropland_like": crop["delta_SM_obs"].mean() if len(crop) else np.nan,
                "mean_delta_SM_noncropland_control": control["delta_SM_obs"].mean() if len(control) else np.nan,
                "residual_crop_minus_control": (
                    crop["delta_SM_obs"].mean() - control["delta_SM_obs"].mean()
                    if len(crop) and len(control)
                    else np.nan
                ),
                "n_pairs": len(group),
                "n_cropland_like_pairs": len(crop),
                "n_noncropland_control_pairs": len(control),
                "positive_unexplained_wetting_fraction": group["positive_unexplained_wetting"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["date_start", "date_end"])


def write_report(args, out_dir, raw_files, cells, cdl_summary, pairs, by_mask, warnings):
    n_cells = cells["smap_cell_id"].nunique()
    cdl_2020 = cdl_summary[cdl_summary["year"].eq(2020)] if "year" in cdl_summary else cdl_summary
    counts = cdl_2020["mask_class"].value_counts().to_dict()
    overall_frac = float(pairs["positive_unexplained_wetting"].mean()) if len(pairs) else np.nan
    crop_frac = by_mask.loc[by_mask["mask_class"].eq("cropland_like"), "positive_unexplained_fraction"]
    control_frac = by_mask.loc[by_mask["mask_class"].eq("noncropland_control"), "positive_unexplained_fraction"]
    crop_frac = float(crop_frac.iloc[0]) if len(crop_frac) else np.nan
    control_frac = float(control_frac.iloc[0]) if len(control_frac) else np.nan
    recommendation = (
        "Proceed cautiously: cropland-like cells show a higher positive low-rain delta-SM frequency than controls."
        if np.isfinite(crop_frac) and np.isfinite(control_frac) and crop_frac > control_frac * 1.25
        else "Do not treat this as a clean irrigation-like signal yet; the HUC8 diagnostic does not show a strong cropland/control separation with the currently available data."
    )

    if len(by_mask):
        table_cols = list(by_mask.columns)
        table_lines = [
            "| " + " | ".join(table_cols) + " |",
            "| " + " | ".join(["---"] * len(table_cols)) + " |",
        ]
        for _, row in by_mask.iterrows():
            values = []
            for col in table_cols:
                value = row[col]
                if isinstance(value, float):
                    values.append("%.6g" % value)
                else:
                    values.append(str(value))
            table_lines.append("| " + " | ".join(values) + " |")
        mask_table = "\n".join(table_lines)
    else:
        mask_table = "No observation pairs available."

    lines = [
        "# HUC8 Delta-SM Diagnostic Report",
        "",
        "Target HUC8: `10180009`, Middle North Platte-Scotts Bluff.",
        "",
        "## Data Coverage",
        "",
        f"- Local raw NSIDC-0800 files used: {len(raw_files)}",
        f"- Raw file date range: {parse_date_from_filename(raw_files[0]) if raw_files else 'none'} to {parse_date_from_filename(raw_files[-1]) if raw_files else 'none'}",
        f"- Requested date range: {args.start_date} to {args.end_date}",
        f"- Source/period used: `{args.source}` `{args.period}`",
        f"- Rainfall CSV provided: `{args.rainfall_csv}`" if args.rainfall_csv else "- Rainfall CSV provided: none",
        "",
        "## HUC8 SMAP/CDL Domain",
        "",
        f"- HUC8 SMAP cells selected: {n_cells}",
        f"- Cropland-like cells: {counts.get('cropland_like', 0)}",
        f"- Non-cropland/control cells: {counts.get('noncropland_control', 0)}",
        f"- Mixed cells: {counts.get('mixed', 0)}",
        f"- Excluded cells: {counts.get('excluded', 0)}",
        "",
        "## Delta-SM Results",
        "",
        f"- Observation pairs: {len(pairs)}",
        f"- Positive unexplained wetting fraction overall: {overall_frac:.4f}" if np.isfinite(overall_frac) else "- Positive unexplained wetting fraction overall: NA",
        "",
        mask_table,
        "",
        "## Decision",
        "",
        recommendation,
        "",
        "## Warnings / Deviations",
        "",
    ]
    lines.extend(["- " + w for w in warnings] if warnings else ["- None."])
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `huc8_smap_cell_cropland_summary.csv`",
            "- `huc8_delta_sm_observation_pairs.csv`",
            "- `huc8_delta_sm_summary_by_mask_class.csv`",
            "- `huc8_delta_sm_summary_by_cell.csv`",
            "- `huc8_smap_cdl_domain_map.png`",
            "- `huc8_delta_sm_timeseries.png`",
            "- `huc8_rainfall_vs_delta_sm_scatter.png`",
            "- `huc8_delta_sm_distribution_by_mask_class.png`",
            "- `huc8_positive_unexplained_wetting_frequency_map.png`",
            "- `huc8_mean_positive_delta_sm_map.png`",
        ]
    )
    (out_dir / "summary_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    warnings = []

    huc8 = load_boundary(args.huc8_boundary)
    dry = load_boundary(args.dry_spottedtail_boundary) if args.dry_spottedtail_boundary.exists() else None

    raw_files = []
    raw_dates = []
    if args.observations_csv is not None:
        if not args.observations_csv.exists():
            raise FileNotFoundError(args.observations_csv)
        obs_for_cells = pd.read_csv(args.observations_csv, nrows=1000)
        required = {"row", "col", "smap_lat", "smap_lon", "overlap_area_km2", "smap_cell_id"}
        if not required.issubset(obs_for_cells.columns):
            raise ValueError("--observations-csv must include %s" % sorted(required))
        cell_rows = (
            pd.read_csv(args.observations_csv, usecols=list(required))
            .drop_duplicates("smap_cell_id")
            .rename(columns={"smap_lat": "lat", "smap_lon": "lon"})
        )
        cells_wgs84 = gpd.GeoDataFrame(
            cell_rows,
            geometry=gpd.points_from_xy(cell_rows["lon"], cell_rows["lat"]),
            crs="EPSG:4326",
        )
        area_crs = estimate_utm_crs(huc8)
        pts_area = cells_wgs84.to_crs(area_crs)
        half = CELL_SIZE_M / 2
        cells_area = gpd.GeoDataFrame(
            pts_area.drop(columns="geometry"),
            geometry=[box(p.x - half, p.y - half, p.x + half, p.y + half) for p in pts_area.geometry],
            crs=area_crs,
        )
        cells_wgs84 = cells_area.to_crs("EPSG:4326")
        raw_dates = sorted(pd.read_csv(args.observations_csv, usecols=["date"])["date"].dropna().astype(str).unique())
    else:
        raw_files = sorted(args.raw_dir.glob("*.h5"))
        raw_files = [p for p in raw_files if parse_date_from_filename(p) and args.start_date <= parse_date_from_filename(p) <= args.end_date]
        if not raw_files:
            raise RuntimeError("No local raw NSIDC H5 files available in requested date range.")
        raw_dates = [parse_date_from_filename(p) for p in raw_files]
        if min(raw_dates) > args.start_date or max(raw_dates) < args.end_date:
            warnings.append("Local raw H5 coverage is incomplete for 2018-2021; diagnostic used available files only.")
        cells_area, cells_wgs84, huc8_area = select_huc8_cells(huc8, raw_files)
    cells_wgs84.to_file(args.out_dir / "huc8_selected_smap_cells.geojson", driver="GeoJSON")
    cells_wgs84.drop(columns="geometry").to_csv(args.out_dir / "huc8_selected_smap_cells.csv", index=False)

    cdl_frames = []
    for year in [2018, 2019, 2020, 2021]:
        cdl_path = args.cdl_dir / ("CDL_%d_31.tif" % year)
        if not cdl_path.exists():
            warnings.append("Missing CDL raster for %d: %s" % (year, cdl_path))
            continue
        cdl_frames.append(summarize_cdl_for_cells(cells_area, cdl_path, year, args.crop_threshold, args.control_threshold, args.exclude_threshold))
    cdl_summary = pd.concat(cdl_frames, ignore_index=True)
    cdl_summary.to_csv(args.out_dir / "huc8_smap_cell_cropland_summary.csv", index=False)
    cdl_2020_cell_count = cdl_summary[cdl_summary["year"].eq(2020)]["smap_cell_id"].nunique()
    selected_cell_count = cells_wgs84["smap_cell_id"].nunique()
    if cdl_2020_cell_count < selected_cell_count:
        warnings.append(
            "CDL rasters classified %d of %d selected HUC8 SMAP cells; unclassified cells are likely outside the Nebraska CDL raster coverage."
            % (cdl_2020_cell_count, selected_cell_count)
        )

    for threshold in [float(x.strip()) for x in args.sensitivity_thresholds.split(",") if x.strip()]:
        tmp = cdl_summary.copy()
        def classify(row):
            if row["excluded_fraction"] >= args.exclude_threshold:
                return "excluded"
            if row["crop_fraction"] >= threshold:
                return "cropland_like"
            if row["crop_fraction"] <= args.control_threshold:
                return "noncropland_control"
            return "mixed"
        tmp["mask_class"] = tmp.apply(classify, axis=1)
        sens = tmp.groupby(["year", "mask_class"], as_index=False).size()
        sens["crop_threshold"] = threshold
        sens.to_csv(args.out_dir / ("huc8_cdl_threshold_sensitivity_crop%02d.csv" % int(threshold * 100)), index=False)

    if args.observations_csv is not None:
        obs = pd.read_csv(args.observations_csv)
        obs = obs[(obs["source"].eq(args.source)) & (obs["period"].eq(args.period))].copy()
    else:
        obs = extract_observations(raw_files, cells_wgs84.drop(columns="geometry"), args.source, args.period)
    obs.to_csv(args.out_dir / "huc8_satellite_observations_extracted.csv", index=False)
    if obs["soil_moisture"].notna().sum() == 0:
        warnings.append("Extracted satellite observations contain no finite soil moisture values.")

    rainfall = load_rainfall(args.rainfall_csv)
    if rainfall is None:
        warnings.append("No HUC8 rainfall table was available; event flags use positive delta-SM without a true rainfall screen.")
    pairs = compute_pairs(obs, rainfall, args.low_rain_threshold_mm, args.positive_delta_threshold, args.max_pair_gap_days)
    if pairs.empty:
        warnings.append("No delta-SM observation pairs were produced.")
    cdl_for_year = cdl_summary[cdl_summary["year"].eq(2020)][["smap_cell_id", "crop_fraction", "noncropland_fraction", "excluded_fraction", "dominant_cdl_class", "mask_class"]]
    pairs = pairs.merge(cdl_for_year, on="smap_cell_id", how="left")
    pairs["mask_class"] = pairs["mask_class"].fillna("unknown")
    pairs.to_csv(args.out_dir / "huc8_delta_sm_observation_pairs.csv", index=False)

    by_mask = summary_by_mask(pairs)
    by_cell = summary_by_cell(pairs)
    events = make_event_summary(pairs)
    by_mask.to_csv(args.out_dir / "huc8_delta_sm_summary_by_mask_class.csv", index=False)
    by_cell.to_csv(args.out_dir / "huc8_delta_sm_summary_by_cell.csv", index=False)
    events.to_csv(args.out_dir / "huc8_delta_sm_event_summary.csv", index=False)

    cdl_2020 = cdl_summary[cdl_summary["year"].eq(2020)]
    plot_domain_map(cells_wgs84, cdl_2020, huc8, dry, args.out_dir / "huc8_smap_cdl_domain_map.png")
    plot_timeseries(events, args.out_dir / "huc8_delta_sm_timeseries.png")
    plot_scatter(pairs, args.out_dir / "huc8_rainfall_vs_delta_sm_scatter.png")
    plot_histogram(pairs, args.out_dir / "huc8_delta_sm_distribution_by_mask_class.png")
    if not by_cell.empty:
        plot_cell_map(cells_wgs84, by_cell, "positive_unexplained_frequency", "Positive Unexplained Wetting Frequency by SMAP Cell", args.out_dir / "huc8_positive_unexplained_wetting_frequency_map.png")
        plot_cell_map(cells_wgs84, by_cell, "mean_positive_delta_SM", "Mean Positive delta_SM by SMAP Cell", args.out_dir / "huc8_mean_positive_delta_sm_map.png")

    metadata = {
        "huc8": "10180009",
        "huc8_name": "Middle North Platte-Scotts Bluff",
        "n_huc8_smap_cells": int(cells_wgs84["smap_cell_id"].nunique()),
        "raw_files_used": len(raw_files),
        "raw_date_range": [min(raw_dates), max(raw_dates)] if raw_dates else [None, None],
        "observations_csv": str(args.observations_csv) if args.observations_csv else None,
        "requested_date_range": [args.start_date, args.end_date],
        "source": args.source,
        "period": args.period,
        "crop_threshold": args.crop_threshold,
        "control_threshold": args.control_threshold,
        "rainfall_available": rainfall is not None,
        "warnings": warnings,
    }
    (args.out_dir / "huc8_delta_sm_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(args, args.out_dir, raw_files, cells_wgs84, cdl_summary, pairs, by_mask, warnings)
    print(args.out_dir)


if __name__ == "__main__":
    main()
