#!/usr/bin/env python3
"""Create Google Earth Engine export tasks for USDA NASS CDL rasters.

This script starts Earth Engine batch exports to Google Drive. It does not
download the files directly to this machine; after the tasks finish, download
the GeoTIFFs from Drive into Week 7/data/cdl/ and pass them to
preprocess_cropland_mask.py.

Recommended first pass for this project:

    python export_usda_nass_cdl_from_earth_engine.py ^
      --boundary-file "C:/Users/f00l2/Desktop/Irrigation/Week 1/Deliverables/Dry_Spottedtail_Creek_USGS_06679000/dry_spottedtail_creek.geojson" ^
      --years 2018,2019,2020,2021 ^
      --band cultivated

The USDA/NASS/CDL cultivated band is coded as:
    1 = non-cultivated
    2 = cultivated
"""

import argparse
import json
from pathlib import Path

import geopandas as gpd


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_BOUNDARY = (
    PROJECT
    / "Week 1"
    / "Deliverables"
    / "Dry_Spottedtail_Creek_USGS_06679000"
    / "dry_spottedtail_creek.geojson"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Export USDA NASS CDL rasters from Google Earth Engine.")
    parser.add_argument("--boundary-file", type=Path, default=DEFAULT_BOUNDARY)
    parser.add_argument("--years", default="2018,2019,2020,2021")
    parser.add_argument("--band", choices=["cultivated", "cropland", "confidence"], default="cultivated")
    parser.add_argument("--drive-folder", default="irrigation_week7_cdl")
    parser.add_argument("--description-prefix", default="dry_spottedtail_cdl")
    parser.add_argument("--scale", type=float, default=30.0)
    parser.add_argument("--buffer-degrees", type=float, default=0.02, help="Small lon/lat buffer around boundary export region.")
    parser.add_argument("--dry-run", action="store_true", help="Print task metadata without starting exports.")
    return parser.parse_args()


def load_region_coordinates(path, buffer_degrees):
    boundary = gpd.read_file(path)
    if boundary.empty:
        raise ValueError("Boundary file has no features: %s" % path)
    if boundary.crs is None:
        boundary = boundary.set_crs("EPSG:4326")
    boundary = boundary.to_crs("EPSG:4326")
    if hasattr(boundary.geometry, "union_all"):
        geom = boundary.geometry.union_all()
    else:
        geom = boundary.geometry.unary_union
    if buffer_degrees:
        geom = geom.buffer(buffer_degrees)
    minx, miny, maxx, maxy = geom.bounds
    return [
        [
            [minx, miny],
            [maxx, miny],
            [maxx, maxy],
            [minx, maxy],
            [minx, miny],
        ]
    ]


def main():
    args = parse_args()
    years = [int(x.strip()) for x in args.years.split(",") if x.strip()]
    region_coords = load_region_coordinates(args.boundary_file, args.buffer_degrees)

    metadata = {
        "earth_engine_collection": "USDA/NASS/CDL",
        "band": args.band,
        "years": years,
        "boundary_file": str(args.boundary_file),
        "drive_folder": args.drive_folder,
        "scale": args.scale,
        "region": region_coords,
    }
    print(json.dumps(metadata, indent=2))

    if args.dry_run:
        return

    import ee

    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()

    region = ee.Geometry.Polygon(region_coords)
    collection = ee.ImageCollection("USDA/NASS/CDL")
    tasks = []
    for year in years:
        image = (
            collection
            .filterDate("%04d-01-01" % year, "%04d-01-01" % (year + 1))
            .first()
            .select(args.band)
            .clip(region)
        )
        file_prefix = "%s_%s_%s" % (args.description_prefix, args.band, year)
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=file_prefix,
            folder=args.drive_folder,
            fileNamePrefix=file_prefix,
            region=region,
            scale=args.scale,
            crs="EPSG:4326",
            maxPixels=1e13,
        )
        task.start()
        tasks.append({"year": year, "task_id": task.id, "file_prefix": file_prefix})
        print("Started %s: %s" % (file_prefix, task.id))

    print("\nAfter the Earth Engine tasks finish, download these GeoTIFFs from Google Drive into:")
    print(PROJECT / "Week 7" / "data" / "cdl")
    print(json.dumps(tasks, indent=2))


if __name__ == "__main__":
    main()
