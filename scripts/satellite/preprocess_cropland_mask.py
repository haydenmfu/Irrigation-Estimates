#!/usr/bin/env python3
"""Preprocess cropland / non-cropland masks for delta-SM diagnostics.

This script is intentionally product-agnostic. It accepts a categorical
land-cover/cropland raster plus an analysis grid, then summarizes the dominant
mask class for each analysis cell. USDA NASS CDL is the preferred product for
this project; use --product-preset usda_nass_cdl_cultivated for the CDL
cultivated band, where 2 means cultivated and 1 means non-cultivated.

Typical use:

    python preprocess_cropland_mask.py \
      --landcover-raster path/to/cropland_or_landcover.tif \
      --analysis-grid path/to/smap_cells.geojson \
      --boundary-file path/to/boundary.geojson \
      --product-preset usda_nass_cdl_cultivated \
      --out-dir Week 7/outputs/cropland_masks/dry_spottedtail_creek

Notes:
- USDA NASS CDL cultivated band values are 1 = non-cultivated and
  2 = cultivated for 2013-2023. This is the recommended simple mask.
- ESA CCI and other products use different class codes. Pass the class lists
  explicitly instead of relying on hard-coded assumptions.
- If the boundary file contains HUC12 geometries and --boundary-level is huc8
  or huc6, the script dissolves by the first 8 or 6 digits of a HUC code field.
"""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.features import geometry_mask
from shapely.geometry import box


DEFAULT_PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_GRID = (
    DEFAULT_PROJECT
    / "Week 2"
    / "outputs_smos_smap_9k"
    / "all_geojsons"
    / "dry_spottedtail_creek"
    / "dry_spottedtail_creek_selected_nsidc0800_cells.geojson"
)
DEFAULT_BOUNDARY = (
    DEFAULT_PROJECT
    / "Week 1"
    / "Deliverables"
    / "Dry_Spottedtail_Creek_USGS_06679000"
    / "dry_spottedtail_creek.geojson"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build cropland/non-cropland masks on an analysis grid.")
    parser.add_argument("--landcover-raster", type=Path, required=True)
    parser.add_argument("--analysis-grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--boundary-file", type=Path, default=DEFAULT_BOUNDARY)
    parser.add_argument("--boundary-level", choices=["current", "huc12", "huc8", "huc6"], default="current")
    parser.add_argument("--huc-code", default=None, help="Optional HUC code to select/dissolve from a WBD layer.")
    parser.add_argument("--huc-field", default=None, help="Optional HUC field name. Auto-detected if omitted.")
    parser.add_argument(
        "--product-preset",
        choices=["none", "usda_nass_cdl_cultivated", "usda_nass_cdl_cropland"],
        default="none",
        help=(
            "Optional class preset. usda_nass_cdl_cultivated uses the CDL cultivated band "
            "(2 cultivated, 1 non-cultivated). usda_nass_cdl_cropland uses common crop-specific "
            "CDL classes and excludes water/developed/barren/wetlands/no-data classes."
        ),
    )
    parser.add_argument("--year", type=int, default=None, help="Optional CDL/product year recorded in metadata.")
    parser.add_argument("--crop-classes", default="", help="Comma-separated land-cover classes considered cropland.")
    parser.add_argument("--noncrop-classes", default="", help="Optional comma-separated classes considered non-cropland.")
    parser.add_argument("--exclude-classes", default="", help="Comma-separated classes to exclude, e.g. water/urban/no-data.")
    parser.add_argument("--crop-threshold", type=float, default=0.50)
    parser.add_argument("--exclude-threshold", type=float, default=0.50)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PROJECT / "Week 7" / "outputs" / "cropland_masks" / "dry_spottedtail_creek")
    return parser.parse_args()


def parse_classes(text):
    if not text:
        return set()
    return {int(float(x.strip())) for x in text.split(",") if x.strip()}


def apply_product_preset(args):
    """Fill class lists for common products while keeping CLI overrides possible."""
    if args.product_preset == "none":
        if not args.crop_classes:
            raise ValueError("--crop-classes is required unless --product-preset is supplied.")
        return args

    if args.product_preset == "usda_nass_cdl_cultivated":
        args.crop_classes = args.crop_classes or "2"
        args.noncrop_classes = args.noncrop_classes or "1"
        return args

    if args.product_preset == "usda_nass_cdl_cropland":
        # Crop-specific CDL classes. This intentionally includes common crop,
        # hay/pasture, fallow/idle cropland, orchard, and double-crop classes.
        # The cultivated band is preferred for a simpler cultivated/non-cultivated mask.
        args.crop_classes = args.crop_classes or (
            "1,2,3,4,5,6,10,11,12,13,14,21,22,23,24,25,26,27,28,29,30,"
            "31,32,33,34,35,36,37,38,39,41,42,43,44,45,46,47,48,49,50,"
            "51,52,53,54,55,56,57,58,59,60,61,66,67,68,69,70,71,72,74,"
            "75,76,77,204,205,206,207,208,209,210,211,212,213,214,216,"
            "217,218,219,220,221,222,223,224,225,226,227,228,229,230,"
            "231,232,233,234,235,236,237,238,239,240,241,242,243,244,"
            "245,246,247,248,249,250,254"
        )
        # Keep natural vegetation such as forest, shrubland, grassland/pasture,
        # and hay as potential non-cropland controls. Exclude only classes that
        # are poor controls for a soil-moisture irrigation diagnostic.
        args.exclude_classes = args.exclude_classes or "0,81,82,83,87,88,111,112,121,122,123,124,131,190,195"
        return args

    raise ValueError(f"Unhandled product preset: {args.product_preset}")


def detect_huc_field(frame):
    candidates = ["huc12", "HUC12", "huc_12", "HUC_12", "huc8", "HUC8", "huc6", "HUC6"]
    for col in candidates:
        if col in frame.columns:
            return col
    for col in frame.columns:
        if "huc" in col.lower():
            return col
    return None


def load_boundary(path, boundary_level, huc_code=None, huc_field=None):
    boundary = gpd.read_file(path)
    if boundary.empty:
        raise ValueError(f"Boundary file has no features: {path}")
    if boundary.crs is None:
        boundary = boundary.set_crs("EPSG:4326")

    if boundary_level in {"huc12", "huc8", "huc6"}:
        field = huc_field or detect_huc_field(boundary)
        if field and huc_code:
            boundary = boundary[boundary[field].astype(str).str.startswith(str(huc_code))]
        if field and boundary_level in {"huc8", "huc6"}:
            n = 8 if boundary_level == "huc8" else 6
            boundary = boundary.copy()
            boundary["_huc_group"] = boundary[field].astype(str).str.slice(0, n)
            if huc_code:
                boundary = boundary[boundary["_huc_group"].eq(str(huc_code)[:n])]
            boundary = boundary.dissolve(by="_huc_group", as_index=False)
        elif field and boundary_level == "huc12" and huc_code:
            boundary = boundary[boundary[field].astype(str).eq(str(huc_code))]
    if boundary.empty:
        raise ValueError("Boundary selection produced no features.")
    return boundary[["geometry"]].dissolve()


def summarize_cell(src, geom, crop_classes, noncrop_classes, exclude_classes):
    try:
        data, transform = mask(src, [geom], crop=True, filled=True, nodata=src.nodata)
    except ValueError:
        return None
    arr = data[0]
    valid = np.ones(arr.shape, dtype=bool)
    if src.nodata is not None:
        valid &= arr != src.nodata
    valid &= np.isfinite(arr)
    if not valid.any():
        return None

    classes = arr[valid].astype(int)
    total = float(len(classes))
    crop = np.isin(classes, list(crop_classes)).sum()
    exclude = np.isin(classes, list(exclude_classes)).sum()
    if noncrop_classes:
        noncrop = np.isin(classes, list(noncrop_classes)).sum()
    else:
        noncrop = total - crop - exclude
    return {
        "n_valid_pixels": int(total),
        "crop_fraction": float(crop / total),
        "noncrop_fraction": float(max(noncrop, 0) / total),
        "exclude_fraction": float(exclude / total),
    }


def main():
    args = parse_args()
    args = apply_product_preset(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    crop_classes = parse_classes(args.crop_classes)
    noncrop_classes = parse_classes(args.noncrop_classes)
    exclude_classes = parse_classes(args.exclude_classes)

    grid = gpd.read_file(args.analysis_grid)
    if grid.empty:
        raise ValueError(f"Analysis grid is empty: {args.analysis_grid}")
    if grid.crs is None:
        grid = grid.set_crs("EPSG:4326")
    boundary = load_boundary(args.boundary_file, args.boundary_level, args.huc_code, args.huc_field)

    with rasterio.open(args.landcover_raster) as src:
        boundary_src = boundary.to_crs(src.crs)
        grid_src = grid.to_crs(src.crs)
        boundary_geom = boundary_src.geometry.iloc[0]
        grid_src = grid_src[grid_src.intersects(boundary_geom)].copy()
        rows = []
        for idx, row in grid_src.iterrows():
            inter = row.geometry.intersection(boundary_geom)
            if inter.is_empty:
                continue
            summary = summarize_cell(src, inter, crop_classes, noncrop_classes, exclude_classes)
            if summary is None:
                continue
            out = row.drop(labels="geometry").to_dict()
            out["grid_index"] = idx
            out.update(summary)
            if summary["exclude_fraction"] >= args.exclude_threshold:
                mask_class = "excluded"
            elif summary["crop_fraction"] >= args.crop_threshold:
                mask_class = "cropland"
            else:
                mask_class = "noncropland"
            out["mask_class"] = mask_class
            out["geometry"] = row.geometry
            rows.append(out)

    if not rows:
        raise RuntimeError("No analysis-grid cells overlapped valid raster data.")
    out_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=grid_src.crs).to_crs("EPSG:4326")
    out_csv = args.out_dir / "cropland_mask_by_analysis_cell.csv"
    out_geojson = args.out_dir / "cropland_mask_by_analysis_cell.geojson"
    out_png = args.out_dir / "cropland_mask_quicklook.png"
    out_meta = args.out_dir / "cropland_mask_metadata.json"
    out_gdf.drop(columns="geometry").to_csv(out_csv, index=False)
    out_gdf.to_file(out_geojson, driver="GeoJSON")

    colors = {"cropland": "#2ca25f", "noncropland": "#9e9e9e", "excluded": "#4C78A8"}
    fig, ax = plt.subplots(figsize=(8.0, 7.2), constrained_layout=True)
    out_gdf.plot(ax=ax, color=out_gdf["mask_class"].map(colors), edgecolor="white", linewidth=0.8)
    boundary.to_crs("EPSG:4326").boundary.plot(ax=ax, color="black", linewidth=1.2)
    ax.set_title("Cropland mask quicklook")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=color, markersize=9, label=label)
        for label, color in colors.items()
        if label in set(out_gdf["mask_class"])
    ]
    ax.legend(handles=handles, frameon=False)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    metadata = {
        "landcover_raster": str(args.landcover_raster),
        "product_preset": args.product_preset,
        "year": args.year,
        "analysis_grid": str(args.analysis_grid),
        "boundary_file": str(args.boundary_file),
        "boundary_level": args.boundary_level,
        "huc_code": args.huc_code,
        "crop_classes": sorted(crop_classes),
        "noncrop_classes": sorted(noncrop_classes),
        "exclude_classes": sorted(exclude_classes),
        "crop_threshold": args.crop_threshold,
        "exclude_threshold": args.exclude_threshold,
        "n_cells": int(len(out_gdf)),
        "mask_counts": out_gdf["mask_class"].value_counts().to_dict(),
    }
    out_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(out_csv)
    print(out_geojson)
    print(out_png)


if __name__ == "__main__":
    main()
