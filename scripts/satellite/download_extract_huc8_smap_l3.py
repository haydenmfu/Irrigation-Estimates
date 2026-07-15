#!/usr/bin/env python3
"""Download native SMAP L3 enhanced granules and extract HUC8 cells.

This script is intentionally date-list driven. It is for targeted weekly
endpoint extraction, not calendar crawling.
"""

import argparse
import json
import re
from pathlib import Path

import earthaccess
import h5py
import numpy as np
import pandas as pd
import requests


PROJECT = Path(__file__).resolve().parents[1]
WEEK7 = PROJECT / "Week 7"
CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
DEFAULT_CELLS = (
    WEEK7
    / "outputs"
    / "delta_sm_pre_pbs"
    / "huc8_10180009_cdl_crop30_power_rain_partial_current"
    / "huc8_selected_smap_cells.csv"
)
DEFAULT_OUT = (
    WEEK7
    / "outputs"
    / "delta_sm_pre_pbs"
    / "huc8_10180009_weekly_vic_aligned"
    / "satellite_weekly"
    / "huc8_smap_l3_endpoint_observations.csv"
)
DEFAULT_AUDIT = DEFAULT_OUT.with_name("huc8_smap_l3_download_audit.csv")
DEFAULT_MANIFEST = DEFAULT_OUT.with_name("smap_l3_endpoint_manifest.csv")
DEFAULT_RAW = WEEK7 / "data" / "smap_l3_raw_huc8_weekly_endpoints"

SMAP_GROUPS = [
    ("AM", "Soil_Moisture_Retrieval_Data_AM"),
    ("PM", "Soil_Moisture_Retrieval_Data_PM"),
]
SMAP_VARIABLES = [
    "soil_moisture",
    "soil_moisture_error",
    "retrieval_qual_flag",
    "surface_flag",
    "vegetation_water_content",
    "landcover_class",
    "latitude",
    "longitude",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Download/extract HUC8 native SMAP L3 endpoint observations.")
    parser.add_argument("--dates", required=True, help="Comma-separated target dates, YYYY-MM-DD.")
    parser.add_argument("--selected-cells", type=Path, default=DEFAULT_CELLS)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--version", default="006")
    parser.add_argument("--login-strategy", default="netrc")
    parser.add_argument("--skip-login", action="store_true")
    parser.add_argument("--delete-raw-after-extract", action="store_true")
    return parser.parse_args()


def date_token(date_text):
    return date_text.replace("-", "")


def parse_date_from_filename(path):
    match = re.search(r"_(\d{8})_", Path(path).name)
    if not match:
        return None
    token = match.group(1)
    return f"{token[:4]}-{token[4:6]}-{token[6:]}"


def data_link(entry):
    links = entry.get("links", [])
    for link in links:
        href = link.get("href", "")
        if href.startswith("https://") and href.endswith(".h5") and "protected" in href:
            return href
    for link in links:
        href = link.get("href", "")
        if href.startswith("https://") and href.endswith(".h5"):
            return href
    return None


def query_cmr(date, version):
    params = {
        "short_name": "SPL3SMP_E",
        "version": version,
        "provider": "NSIDC_CPRD",
        "temporal": f"{date}T00:00:00Z,{date}T23:59:59Z",
        "page_size": 10,
    }
    response = requests.get(CMR_URL, params=params, timeout=120)
    response.raise_for_status()
    entries = response.json().get("feed", {}).get("entry", [])
    rows = []
    for entry in entries:
        rows.append(
            {
                "target_date": date,
                "title": entry.get("title"),
                "granule_ur": entry.get("granule_ur"),
                "collection_concept_id": entry.get("collection_concept_id"),
                "size_mb": entry.get("granule_size"),
                "url": data_link(entry),
            }
        )
    return rows


def build_manifest(dates, version):
    rows = []
    for date in dates:
        rows.extend(query_cmr(date, version))
    return pd.DataFrame(rows).sort_values(["target_date", "title"])


def existing_file(raw_dir, date):
    token = date_token(date)
    matches = sorted(raw_dir.glob(f"SMAP_L3_SM_P_E_{token}_*.h5"))
    return matches[0] if matches else None


def download_missing(manifest, raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    to_download = []
    for date, group in manifest.groupby("target_date"):
        existing = existing_file(raw_dir, date)
        if existing is not None:
            audit_rows.append(
                {
                    "target_date": date,
                    "granules_found": len(group),
                    "downloaded": False,
                    "local_file": str(existing),
                    "notes": "already_present",
                }
            )
            continue
        row = group[group["url"].notna()].head(1)
        if row.empty:
            audit_rows.append(
                {
                    "target_date": date,
                    "granules_found": len(group),
                    "downloaded": False,
                    "local_file": "",
                    "notes": "no_download_url",
                }
            )
            continue
        to_download.append(row.iloc[0].to_dict())

    if to_download:
        urls = [row["url"] for row in to_download]
        paths = earthaccess.download(urls, local_path=str(raw_dir))
        by_date = {parse_date_from_filename(path): Path(path) for path in paths}
        for row in to_download:
            date = row["target_date"]
            path = by_date.get(date) or existing_file(raw_dir, date)
            audit_rows.append(
                {
                    "target_date": date,
                    "granules_found": int((manifest["target_date"] == date).sum()),
                    "downloaded": path is not None,
                    "local_file": str(path) if path else "",
                    "notes": "downloaded" if path else "download_failed_or_missing_after_download",
                }
            )
    return pd.DataFrame(audit_rows).sort_values("target_date")


def product_key(group, var, period):
    if period == "PM":
        return f"{group}/{var}_pm"
    return f"{group}/{var}"


def fill_values(dataset):
    values = dataset.attrs.get("_FillValue", None)
    if values is None:
        return []
    out = []
    for value in np.ravel(np.asarray(values)):
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            pass
    return out


def clean_scalar(value, dataset):
    try:
        scalar = np.asarray(value).item()
    except ValueError:
        scalar = value
    try:
        numeric = float(scalar)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(numeric):
        return np.nan
    if any(numeric == fill for fill in fill_values(dataset)):
        return np.nan
    return numeric


def selected_row_cache(h5, group, period, cells):
    rows = sorted({int(row) for row in cells["row"]})
    cache = {}
    for var in SMAP_VARIABLES:
        key = product_key(group, var, period)
        if key not in h5:
            cache[var] = None
            continue
        dataset = h5[key]
        cache[var] = {
            "dataset": dataset,
            "rows": {row: np.asarray(dataset[row, :]) for row in rows},
        }
    return cache


def cached_value(cache, var, row, col):
    entry = cache.get(var)
    if entry is None:
        return np.nan
    return clean_scalar(entry["rows"][row][col], entry["dataset"])


def recommended_mask(flag, finite):
    if not finite or not np.isfinite(flag):
        return False
    # NSIDC SPL3SMP_E guidance marks retrieval_qual_flag 0 or 8 as high quality.
    return int(flag) in {0, 8}


def extract_records(files, selected_cells):
    cells = pd.read_csv(selected_cells)
    records = []
    for file_path in sorted(files):
        date = parse_date_from_filename(file_path)
        print(f"extracting {Path(file_path).name}", flush=True)
        with h5py.File(file_path, "r") as h5:
            for period, group in SMAP_GROUPS:
                cache = selected_row_cache(h5, group, period, cells)
                if cache.get("soil_moisture") is None or cache.get("retrieval_qual_flag") is None:
                    continue
                for _, cell in cells.iterrows():
                    r = int(cell["row"])
                    c = int(cell["col"])
                    value = cached_value(cache, "soil_moisture", r, c)
                    finite = bool(np.isfinite(value))
                    flag = cached_value(cache, "retrieval_qual_flag", r, c)
                    rec = {
                        "date": date,
                        "product": "SPL3SMP_E",
                        "source": "SMAP_L3",
                        "period": period,
                        "row": r,
                        "col": c,
                        "smap_cell_id": cell.get("smap_cell_id", f"{r}_{c}"),
                        "smap_lat": float(cell["lat"]),
                        "smap_lon": float(cell["lon"]),
                        "overlap_area_km2": float(cell["overlap_area_km2"]),
                        "soil_moisture": float(value) if finite else np.nan,
                        "retrieval_qual_flag": float(flag) if np.isfinite(flag) else np.nan,
                        "retrieval_recommended": recommended_mask(flag, finite),
                        "source_file": Path(file_path).name,
                    }
                    for var in SMAP_VARIABLES:
                        if var in {"soil_moisture", "retrieval_qual_flag"}:
                            continue
                        val = cached_value(cache, var, r, c)
                        rec[var] = float(val) if np.isfinite(val) else np.nan
                    records.append(rec)
    return pd.DataFrame(records)


def append_dedup(new, path):
    if path.exists():
        old = pd.read_csv(path)
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    key = ["date", "product", "period", "row", "col"]
    combined = combined.drop_duplicates(subset=key, keep="last").sort_values(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return combined


def main():
    args = parse_args()
    dates = sorted({d.strip() for d in args.dates.split(",") if d.strip()})
    if not args.skip_login:
        print(f"logging in with earthaccess strategy={args.login_strategy}", flush=True)
        earthaccess.login(strategy=args.login_strategy)

    print(f"querying CMR for {len(dates)} date(s)", flush=True)
    manifest = build_manifest(dates, args.version)
    args.manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest_csv, index=False)
    args.manifest_csv.with_suffix(".json").write_text(
        json.dumps(
            {
                "dates_requested": dates,
                "entries": int(len(manifest)),
                "version": args.version,
                "manifest_csv": str(args.manifest_csv),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"checking/downloading {len(manifest)} manifest row(s)", flush=True)
    audit = download_missing(manifest, args.raw_dir)
    files = [Path(p) for p in audit["local_file"].dropna().astype(str) if p and Path(p).exists()]
    print(f"extracting {len(files)} local file(s)", flush=True)
    records = extract_records(files, args.selected_cells)
    combined = append_dedup(records, args.out_csv)

    counts = (
        combined[combined["date"].isin(dates)]
        .groupby(["date", "period"], as_index=False)
        .agg(
            n_rows=("soil_moisture", "size"),
            n_finite=("soil_moisture", lambda x: int(x.notna().sum())),
            n_recommended=("retrieval_recommended", "sum"),
        )
    )
    audit = audit.merge(counts, left_on="target_date", right_on="date", how="left").drop(columns=["date"], errors="ignore")
    args.audit_csv.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.audit_csv, index=False)

    if args.delete_raw_after_extract:
        for file_path in files:
            file_path.unlink(missing_ok=True)

    print(args.out_csv)
    print("dates_requested=%d manifest_rows=%d extracted_rows=%d combined_rows=%d" % (len(dates), len(manifest), len(records), len(combined)))


if __name__ == "__main__":
    main()
