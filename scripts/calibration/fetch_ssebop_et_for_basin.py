import argparse
import datetime as dt
import json
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.mask import mask


SSEBOP_DOMAIN = "https://edcintl.cr.usgs.gov"
SSEBOP_DIR = "/downloads/sciweb1/shared/uswem/web/conus/eta/modis_eta/daily/downloads"
SSEBOP_PATTERN = "det{year}{doy:03d}.modisSSEBopETactual.zip"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download USGS SSEBop MODIS daily actual ET and extract basin-mean ET. "
            "Daily zip files contain GeoTIFF values scaled by 1000; output ET is mm/day."
        )
    )
    parser.add_argument("--basin-geojson", required=True, help="Basin polygon in GeoJSON/shapefile format.")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--basin-name", default="dry_spottedtail_creek")
    parser.add_argument("--raw-dir", default=None, help="Optional raw zip/cache directory.")
    parser.add_argument("--irrigation-months", default="5,6,7,8,9")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--keep-tifs", action="store_true", help="Keep extracted GeoTIFFs.")
    parser.add_argument(
        "--delete-raw",
        action="store_true",
        help="Delete each downloaded zip after successful extraction.",
    )
    parser.add_argument(
        "--reset-output",
        action="store_true",
        help="Ignore an existing daily CSV and rebuild the requested period.",
    )
    return parser.parse_args()


def daterange(start, end):
    day = start
    while day <= end:
        yield day
        day += dt.timedelta(days=1)


def url_for_day(day):
    doy = int(day.strftime("%j"))
    return f"{SSEBOP_DOMAIN}{SSEBOP_DIR}/{SSEBOP_PATTERN.format(year=day.year, doy=doy)}"


def download_if_needed(url, dest, timeout):
    if dest.exists() and dest.stat().st_size > 0:
        return "cached"
    with requests.get(url, stream=True, timeout=timeout) as response:
        if response.status_code == 404:
            return "missing"
        response.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(dest)
    return "downloaded"


def first_tif_from_zip(zip_path, extract_dir):
    with zipfile.ZipFile(zip_path) as zf:
        tifs = [name for name in zf.namelist() if name.lower().endswith((".tif", ".tiff"))]
        if not tifs:
            raise RuntimeError(f"No GeoTIFF found in {zip_path}")
        member = tifs[0]
        out = extract_dir / Path(member).name
        if not out.exists():
            with zf.open(member) as src, out.open("wb") as dst:
                dst.write(src.read())
        return out


def summarize_tif(tif_path, basin_wgs84):
    with rasterio.open(tif_path) as src:
        basin = basin_wgs84.to_crs(src.crs)
        data, _ = mask(src, [geom for geom in basin.geometry], crop=True, filled=False)
        arr = np.ma.asarray(data[0], dtype="float64")
        values = arr.compressed()
        nodata = src.nodata
        if nodata is not None:
            values = values[values != nodata]
        values = values[values != 9999]
        values = values / 1000.0
        values = values[np.isfinite(values)]
        if values.size == 0:
            return {
                "observed_et_mm": np.nan,
                "n_pixels": 0,
                "min_et_mm": np.nan,
                "max_et_mm": np.nan,
                "source_crs": str(src.crs),
            }
        return {
            "observed_et_mm": float(np.mean(values)),
            "n_pixels": int(values.size),
            "min_et_mm": float(np.min(values)),
            "max_et_mm": float(np.max(values)),
            "source_crs": str(src.crs),
        }


def main():
    args = parse_args()
    start = dt.datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = dt.datetime.strptime(args.end_date, "%Y-%m-%d").date()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir) if args.raw_dir else output_dir / "raw_ssebop_daily"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tif_dir = output_dir / "extracted_tifs"
    tif_dir.mkdir(parents=True, exist_ok=True)
    irrigation_months = {int(x) for x in args.irrigation_months.split(",") if x.strip()}

    basin = gpd.read_file(args.basin_geojson).to_crs("EPSG:4326")
    daily_path = output_dir / f"{args.basin_name}_ssebop_et_daily.csv"
    monthly_path = output_dir / f"{args.basin_name}_ssebop_et_monthly.csv"
    seasonal_path = output_dir / f"{args.basin_name}_ssebop_et_seasonal.csv"
    metadata_path = output_dir / f"{args.basin_name}_ssebop_et_metadata.json"

    rows = []
    completed_dates = set()
    if daily_path.exists() and not args.reset_output:
        existing = pd.read_csv(daily_path)
        if "date" in existing:
            rows = existing.to_dict("records")
            completed_dates = set(existing["date"].astype(str))
            print("Resuming with {} completed dates from {}".format(
                len(completed_dates), daily_path
            ))

    for day in daterange(start, end):
        if day.isoformat() in completed_dates:
            print("{}: already complete".format(day))
            continue
        doy = int(day.strftime("%j"))
        filename = SSEBOP_PATTERN.format(year=day.year, doy=doy)
        zip_path = raw_dir / filename
        url = url_for_day(day)
        status = download_if_needed(url, zip_path, args.timeout)
        row = {
            "date": day.isoformat(),
            "year": day.year,
            "month": day.month,
            "doy": doy,
            "season": "irrigation" if day.month in irrigation_months else "non_irrigation",
            "product": "USGS_SSEBop_MODIS_daily_ETa",
            "url": url,
            "download_status": status,
        }
        if status == "missing":
            row.update({"observed_et_mm": np.nan, "n_pixels": 0, "min_et_mm": np.nan, "max_et_mm": np.nan})
            rows.append(row)
            print(f"{day}: missing")
            pd.DataFrame(rows).sort_values("date").to_csv(daily_path, index=False)
            continue
        try:
            tif_path = first_tif_from_zip(zip_path, tif_dir)
            row.update(summarize_tif(tif_path, basin))
            if not args.keep_tifs:
                if tif_path.exists():
                    tif_path.unlink()
            if args.delete_raw and zip_path.exists():
                zip_path.unlink()
            print(f"{day}: {row['observed_et_mm']:.4f} mm/day from {row['n_pixels']} pixels ({status})")
        except Exception as exc:
            row["error"] = str(exc)
            row.update({"observed_et_mm": np.nan, "n_pixels": 0, "min_et_mm": np.nan, "max_et_mm": np.nan})
            print(f"{day}: ERROR {exc}")
        rows.append(row)
        pd.DataFrame(rows).sort_values("date").to_csv(daily_path, index=False)

    if not args.keep_tifs:
        try:
            tif_dir.rmdir()
        except OSError:
            pass

    daily = pd.DataFrame(rows).sort_values("date").drop_duplicates("date", keep="last")

    daily.to_csv(daily_path, index=False)
    monthly = (
        daily.dropna(subset=["observed_et_mm"])
        .groupby(["year", "month", "season"])
        .agg({"observed_et_mm": ["sum", "count"], "n_pixels": "mean"})
        .reset_index()
    )
    monthly.columns = [
        "year",
        "month",
        "season",
        "observed_et_mm",
        "n_days",
        "mean_pixels",
    ]
    seasonal = (
        monthly.groupby(["year", "season"])
        .agg({"observed_et_mm": "sum", "n_days": "sum", "mean_pixels": "mean"})
        .reset_index()
    )
    monthly.to_csv(monthly_path, index=False)
    seasonal.to_csv(seasonal_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "source": "USGS SSEBop MODIS Daily actual ET",
                "source_page": "https://earlywarning.usgs.gov/ssebop/modis/daily/",
                "source_template": f"{SSEBOP_DOMAIN}{SSEBOP_DIR}/{SSEBOP_PATTERN}",
                "scale_factor": "GeoTIFF values divided by 1000 to get mm/day",
                "start_date": args.start_date,
                "end_date": args.end_date,
                "irrigation_months": sorted(irrigation_months),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for path in [daily_path, monthly_path, seasonal_path, metadata_path]:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
