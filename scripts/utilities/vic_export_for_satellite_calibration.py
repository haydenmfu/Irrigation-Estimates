from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_VARIABLES = [
    "OUT_PREC",
    "OUT_EVAP",
    "OUT_TRANSP_VEG",
    "OUT_RUNOFF",
    "OUT_BASEFLOW",
    "OUT_SOIL_MOIST",
    "OUT_SOIL_WET",
    "OUT_AIR_TEMP",
    "OUT_LAI",
    "OUT_FCANOPY",
    "OUT_ALBEDO",
    "OUT_SENSIBLE",
    "OUT_LATENT",
    "OUT_REL_HUMID",
    "OUT_WIND",
    "OUT_EVAP_CANOP",
    "OUT_EVAP_BARE",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export VIC daily grid/state variables for satellite soil-moisture comparison."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing VIC NetCDF files, e.g. /path/to/results/basin0",
    )
    parser.add_argument(
        "--pattern",
        default="fluxes.*.nc",
        help="Glob pattern for VIC NetCDF files. Default: fluxes.*.nc",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where CSV/JSON outputs will be written.",
    )
    parser.add_argument(
        "--basin-name",
        default="basin0",
        help="Short basin label used in output file names.",
    )
    parser.add_argument(
        "--layer-depths-mm",
        default=None,
        help=(
            "Optional comma-separated VIC soil layer depths in mm, e.g. 100,300,700. "
            "If provided, OUT_SOIL_MOIST is also exported as volumetric water content."
        ),
    )
    parser.add_argument(
        "--parameter-file",
        default=None,
        help=(
            "Optional VIC parameter NetCDF containing depth(layer, lat, lon). "
            "Preferred over --layer-depths-mm because it preserves per-cell depths."
        ),
    )
    parser.add_argument(
        "--variables",
        default=",".join(DEFAULT_VARIABLES),
        help="Comma-separated VIC variables to export.",
    )
    return parser.parse_args()


def parse_layer_depths(value):
    if value is None or value.strip() == "":
        return None
    depths = [float(x.strip()) for x in value.split(",") if x.strip()]
    if any(depth <= 0 for depth in depths):
        raise ValueError("All layer depths must be positive.")
    return depths


def find_layer_dim(da):
    layer_dims = [dim for dim in da.dims if "layer" in dim.lower()]
    return layer_dims[0] if layer_dims else None


def standardize_layer_depths(depth_da, target_layer_dim):
    layer_dim = find_layer_dim(depth_da)
    if layer_dim is None:
        raise ValueError("Parameter-file depth variable has no layer dimension.")
    if layer_dim != target_layer_dim:
        depth_da = depth_da.rename({layer_dim: target_layer_dim})
    return depth_da * 1000.0


def load_parameter_layer_depths(parameter_file, target_layer_dim):
    if parameter_file is None:
        return None
    params = xr.open_dataset(parameter_file)
    if "depth" not in params:
        raise ValueError(f"No 'depth' variable found in parameter file: {parameter_file}")
    return standardize_layer_depths(params["depth"], target_layer_dim)


def active_cell_mask(ds):
    if "OUT_SOIL_MOIST" in ds:
        da = ds["OUT_SOIL_MOIST"]
        reduce_dims = [dim for dim in da.dims if dim not in ("lat", "lon")]
        return da.notnull().any(dim=reduce_dims)

    first = next(iter(ds.data_vars))
    da = ds[first]
    reduce_dims = [dim for dim in da.dims if dim not in ("lat", "lon")]
    return da.notnull().any(dim=reduce_dims)


def export_grid_daily(ds, variables, layer_depths, parameter_depths_mm, output_path):
    parts = []

    for var in variables:
        if var not in ds:
            continue

        da = ds[var]
        layer_dim = find_layer_dim(da)

        if var == "OUT_SOIL_MOIST" and layer_dim:
            for i in range(da.sizes[layer_dim]):
                layer = da.isel({layer_dim: i}).drop_vars(layer_dim, errors="ignore")
                layer_name = f"vic_soil_moist_layer{i + 1}_mm"
                parts.append(layer.to_dataframe(name=layer_name).reset_index())

                if parameter_depths_mm is not None:
                    depth = parameter_depths_mm.isel({layer_dim: i})
                    volumetric = layer / depth
                    volumetric_name = f"vic_soil_moist_layer{i + 1}_m3m3"
                    parts.append(volumetric.to_dataframe(name=volumetric_name).reset_index())
                elif layer_depths is not None:
                    if i >= len(layer_depths):
                        raise ValueError(
                            f"Layer depth missing for OUT_SOIL_MOIST layer {i + 1}; "
                            f"got {len(layer_depths)} depth values."
                        )
                    volumetric = layer / layer_depths[i]
                    volumetric_name = f"vic_soil_moist_layer{i + 1}_m3m3"
                    parts.append(volumetric.to_dataframe(name=volumetric_name).reset_index())
        elif layer_dim:
            for i in range(da.sizes[layer_dim]):
                layer = da.isel({layer_dim: i}).drop_vars(layer_dim, errors="ignore")
                name = f"{var.lower()}_layer{i + 1}"
                parts.append(layer.to_dataframe(name=name).reset_index())
        else:
            name = var.lower()
            parts.append(da.to_dataframe(name=name).reset_index())

    if not parts:
        raise RuntimeError("None of the requested variables were found in the VIC dataset.")

    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on=["time", "lat", "lon"], how="outer")

    value_cols = [col for col in out.columns if col not in ("time", "lat", "lon")]
    out = out.dropna(subset=value_cols, how="all")
    out.to_csv(output_path, index=False)
    return out


def export_basin_daily_summary(grid_df, output_path):
    value_cols = [col for col in grid_df.columns if col not in ("time", "lat", "lon")]
    summary = (
        grid_df.groupby("time", as_index=False)[value_cols]
        .mean(numeric_only=True)
        .rename(columns={col: f"basin_mean_{col}" for col in value_cols})
    )
    summary["n_grid_cells"] = grid_df.groupby("time").size().to_numpy()
    summary.to_csv(output_path, index=False)
    return summary


def export_metadata(ds, variables, layer_depths, parameter_file, grid_df, output_path):
    metadata = {
        "time_start": str(pd.to_datetime(grid_df["time"]).min()),
        "time_end": str(pd.to_datetime(grid_df["time"]).max()),
        "n_times": int(grid_df["time"].nunique()),
        "n_grid_cells": int(grid_df[["lat", "lon"]].drop_duplicates().shape[0]),
        "exported_columns": list(grid_df.columns),
        "requested_variables": variables,
        "available_variables": list(ds.data_vars),
        "layer_depths_mm": layer_depths,
        "parameter_file": str(parameter_file) if parameter_file else None,
        "notes": [
            "VIC OUT_SOIL_MOIST is usually water storage per layer, commonly mm.",
            "Layer volumetric columns are only valid if --layer-depths-mm matches the VIC soil parameter file.",
            "Satellite SMAP/SMOS products report near-surface volumetric soil moisture, so layer 1 m3/m3 is usually the first comparison target.",
        ],
    }
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No NetCDF files found with pattern {args.pattern!r} in {input_dir}")

    variables = [var.strip() for var in args.variables.split(",") if var.strip()]
    layer_depths = parse_layer_depths(args.layer_depths_mm)

    ds = xr.open_mfdataset(files, combine="by_coords")
    soil_moist_layer_dim = find_layer_dim(ds["OUT_SOIL_MOIST"]) if "OUT_SOIL_MOIST" in ds else "nlayer"
    parameter_depths_mm = load_parameter_layer_depths(args.parameter_file, soil_moist_layer_dim)
    mask = active_cell_mask(ds)
    ds = ds.where(mask)

    prefix = args.basin_name
    grid_path = output_dir / f"{prefix}_vic_grid_daily_for_satellite.csv"
    summary_path = output_dir / f"{prefix}_vic_basin_daily_summary_for_satellite.csv"
    metadata_path = output_dir / f"{prefix}_vic_export_metadata.json"

    grid_df = export_grid_daily(ds, variables, layer_depths, parameter_depths_mm, grid_path)
    summary_df = export_basin_daily_summary(grid_df, summary_path)
    export_metadata(ds, variables, layer_depths, args.parameter_file, grid_df, metadata_path)

    print(f"Wrote grid daily CSV: {grid_path}")
    print(f"Wrote basin daily summary CSV: {summary_path}")
    print(f"Wrote metadata JSON: {metadata_path}")
    print(f"Rows: {len(grid_df)}")
    print(f"Unique times: {grid_df['time'].nunique()}")
    print(f"Unique lat/lon cells: {grid_df[['lat', 'lon']].drop_duplicates().shape[0]}")
    print(f"Summary rows: {len(summary_df)}")


if __name__ == "__main__":
    main()
