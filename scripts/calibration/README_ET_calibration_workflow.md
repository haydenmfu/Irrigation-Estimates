# Week 4 ET Calibration Workflow

This week pivots calibration away from soil moisture and routed runoff toward evapotranspiration (ET). The optimizer can still follow the VICRes/eNSGA-II structure, but the objective function should compare VIC `OUT_EVAP` against observed satellite ET.

## Why This Shape

The NSGA-II paper emphasizes nondominated sorting, elitism, and crowding-distance diversity. For this project that means we should preserve separate objective functions instead of collapsing all ET errors into one weighted score. A good first objective set is:

1. irrigation-season ET RMSE, May through September
2. irrigation-season ET bias
3. warm non-irrigation ET RMSE, April and October

That gives the optimizer permission to find tradeoffs: strong irrigation fit, low seasonal bias, and reasonable shoulder-season behavior. November through March remains a diagnostic because SSEBop contains extensive cold-season zeros.

## Scripts Added

### 1. Export VIC ET features

Script:

```bash
python "Week 4/Calibration/export_vic_et_features.py" \
  --input-dir /home/fs01/hmf63/Local_Irrigation/VICFiles/results/p2_cal/benchmark/basin0 \
  --pattern "fluxes.*.nc" \
  --output-dir /home/fs01/hmf63/Local_Irrigation/VICFiles/week4_et_calibration/vic_et \
  --basin-name dry_spottedtail_creek
```

Outputs:

- `dry_spottedtail_creek_vic_et_grid_daily.csv`
- `dry_spottedtail_creek_vic_et_basin_daily.csv`
- `dry_spottedtail_creek_vic_et_monthly.csv`
- `dry_spottedtail_creek_vic_et_seasonal.csv`
- `dry_spottedtail_creek_vic_et_export_metadata.json`

Primary VIC calibration column:

```text
basin_mean_vic_et_total_mm
```

This uses `OUT_EVAP` when available. The script also exports/checks the component sum:

```text
OUT_EVAP_CANOP + OUT_EVAP_BARE + OUT_TRANSP_VEG
```

### 2. Fetch observed ET from USGS SSEBop

Script:

```bash
python "Week 4/Calibration/fetch_ssebop_et_for_basin.py" \
  --basin-geojson "Week 2/data/dry_spottedtail_creek.geojson" \
  --start-date 2020-05-01 \
  --end-date 2020-09-30 \
  --output-dir "Week 4/data/observed_et/dry_spottedtail_creek" \
  --basin-name dry_spottedtail_creek
```

Outputs:

- `dry_spottedtail_creek_ssebop_et_daily.csv`
- `dry_spottedtail_creek_ssebop_et_monthly.csv`
- `dry_spottedtail_creek_ssebop_et_seasonal.csv`
- `dry_spottedtail_creek_ssebop_et_metadata.json`

Notes:

- Product: USGS SSEBop MODIS Daily actual ET.
- Native scale is about 1 km.
- GeoTIFF values are divided by 1000 to produce mm/day.
- Values outside the basin polygon are masked, not treated as zero.
- Use `--delete-raw` to remove each approximately 12 MB daily zip after extraction.
- The downloader checkpoints its daily CSV after every date and resumes completed dates.
- This is a practical first observed-ET target. OpenET would be higher-resolution, but it usually requires a Google Earth Engine/OpenET access path.

For the complete local 2018-2021 fetch and baseline comparison, run:

```powershell
cd "C:\Users\f00l2\Desktop\Irrigation\Week 4"
.\run_et_observation_pipeline.ps1
```

### 3. Build calibration-ready ET tables

Script:

```bash
python "Week 4/Calibration/build_et_calibration_table.py" \
  --vic-daily "Week 4/data/vic_et/dry_spottedtail_creek_vic_et_basin_daily.csv" \
  --observed-daily "Week 4/data/observed_et/dry_spottedtail_creek/dry_spottedtail_creek_ssebop_et_daily.csv" \
  --output-dir "Week 4/outputs/et_calibration/dry_spottedtail_creek" \
  --basin-name dry_spottedtail_creek
```

Outputs:

- `dry_spottedtail_creek_et_calibration_daily.csv`
- `dry_spottedtail_creek_et_calibration_monthly.csv`
- `dry_spottedtail_creek_et_calibration_seasonal.csv`
- `dry_spottedtail_creek_et_calibration_metrics.csv`
- `dry_spottedtail_creek_et_calibration_metadata.json`

## ET eNSGA-II Calibration Driver

Script:

```text
Week 4/Calibration/basin_calibration_eNSGAII_ET.py
```

It reuses the VICRes epsilon-NSGA-II structure, but replaces:

- routing execution
- station discharge loading
- runoff objectives

with:

- VIC execution
- `OUT_EVAP` extraction
- monthly ET objective calculation
- basin-wide NetCDF soil parameter updates

Default objectives:

1. May-September normalized monthly RMSE
2. May-September absolute relative bias
3. April and October normalized monthly RMSE

The default calibration period is 2018-2020. The year 2021 is reserved for
out-of-sample validation. November-March is retained in diagnostics but not
optimized because the daily SSEBop record contains extensive cold-season zeros.

The active parameter defaults are based on the actual `veg_p0_soilgrid.nc`
structure:

- `infilt`, `Ds`, `Ws`, and `c`: basin-wide absolute values
- `depth1`, `depth2`, and `depth3`: basin-wide layer thicknesses
- `Dsmax_multiplier`: multiplier applied to the existing spatial `Dsmax` grid

The multiplier preserves the original spatial soil pattern. The runoff
calibration script's `Dsmax` range of 0-30 and deep-layer ranges were not
carried over because they are incompatible with the supplied parameter file,
where active-cell `Dsmax` is approximately 176-898 and layer thicknesses are
0.05, 0.75, and 0.70 m.

A Hopper SLURM template is provided at:

```text
Week 4/Calibration/run_vic_et_calibration_basin0.sh
```

The default submission is a 20-evaluation pipeline pilot:

```bash
sbatch run_vic_et_calibration_basin0.sh
```

The default run label is `schemafix_pilot_N20`, keeping it separate from the
first failed pilot whose xarray rewrite changed the NetCDF `depth` dimension
order. Candidate parameter files are now copied from the original and edited
in place with netCDF4 so VIC's schema and fill values are preserved.

Hopper's Python 3.6 environment requires Platypus 1.2.0:

```bash
source /home/fs01/hmf63/Local_Irrigation/VICFiles/venv/bin/activate
pip install "platypus-opt==1.2.0"
```

Platypus 1.3 and newer require Python 3.8 or newer.

## Held-Out Validation

After the full N100 calibration is complete, evaluation 85 is selected from
the calibration-period Pareto front as the balanced compromise. Validate the
entire seven-member Pareto front on 2021 without re-tuning the selection:

```powershell
cd "C:\Users\f00l2\Desktop\Irrigation\Week 4"
.\upload_et_validation_to_hopper.ps1
```

Then on Hopper:

```bash
cd /home/fs01/hmf63/Local_Irrigation/VICFiles
sbatch run_et_pareto_validation_2021.sh
```

Retrieve the completed validation with:

```powershell
.\retrieve_et_validation_from_hopper.ps1
```

After the pilot succeeds, launch 100 evaluations with:

```bash
sbatch --export=ALL,MAX_EVALUATIONS=100,RUN_LABEL=full_N100 run_vic_et_calibration_basin0.sh
```

The 20-evaluation pilot is the initial NSGA-II population only. The
100-evaluation run adds approximately four evolutionary generations. Based on
the pilot sensitivity, the full search modestly expands `infilt` from 0.80 to
0.90 and top-layer depth from 0.20 to 0.25 m. A fixed seed (`20260623`) makes
the full run reproducible.

Before submitting it, copy the completed observed ET CSV and both calibration
scripts to Hopper:

```powershell
scp "C:/Users/f00l2/Desktop/Irrigation/Week 4/data/observed_et/dry_spottedtail_creek/dry_spottedtail_creek_ssebop_et_daily.csv" "hmf63@hopper.cac.cornell.edu:/home/fs01/hmf63/Local_Irrigation/VICFiles/week4_et_calibration/observed_et/"

scp "C:/Users/f00l2/Desktop/Irrigation/Week 4/Calibration/basin_calibration_eNSGAII_ET.py" "C:/Users/f00l2/Desktop/Irrigation/Week 4/Calibration/run_vic_et_calibration_basin0.sh" "hmf63@hopper.cac.cornell.edu:/home/fs01/hmf63/Local_Irrigation/VICFiles/"
```

## VIC Outputs Needed

For calibration runs, VIC should output at least:

```text
OUT_EVAP
OUT_TRANSP_VEG
OUT_EVAP_CANOP
OUT_EVAP_BARE
OUT_PREC
OUT_RUNOFF
OUT_BASEFLOW
OUT_SOIL_WET
OUT_AIR_TEMP
```

`OUT_RUNOFF` and `OUT_BASEFLOW` are not objective targets yet, but they are sanity checks so an ET fit does not create unreasonable water balance behavior.
