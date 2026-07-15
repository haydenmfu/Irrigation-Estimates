# Storage Cleanup Candidates

Audit date: 2026-07-12

This is a cleanup map for the irrigation project workspace. Nothing in this file has been deleted automatically. The main goal is to separate large raw/cache/archive material from outputs that are still useful for the current PBS/SMAP/VIC workflow.

## Current Largest Areas

| Path | Approx. size | Notes |
|---|---:|---|
| `Week 2/` | 52.7 GB | Mostly raw June 2020 satellite granules. |
| `Week 4/data/vic_et_calibration_runs/` | 46.8 GB | Tens of thousands of VIC calibration run text outputs. |
| `Week 7/data/` | 30+ GB and growing | Includes active full-season SMAP download plus older raw caches. |
| `.git/objects/` | 24.1 GB | Loose git objects; likely compressible with `git gc`. |
| `Week 1/data/` | 1.6 GB | 2025 CDL GeoTIFFs, likely outside the current 2018-2021 analysis period. |

## Do Not Touch While Download Is Active

The current full-season SMAP L3 download is running here:

```text
Week 7/data/smap_l3_raw_huc8_10180009_2020_may_sep/
```

It is being written by:

```text
Week 7/prepare_full_season_delta_sm_pbs.ps1 -DownloadSmap -BuildTargets
Week 7/download_extract_huc8_smap_l3.py --dates 2020-05-01 ... 2020-09-30
```

At audit time it contained completed `.h5` files plus temporary partial files. Do not delete or move this folder until the download/extraction finishes or is intentionally stopped.

## Strong Cleanup Candidates

These are the most defensible cleanup targets because they are raw downloads, obsolete partial downloads, or outside the current study period.

| Path | Approx. size | Why it is probably disposable |
|---|---:|---|
| `Week 2/raw_smos_smap_9k/` | 33.6 GB | Raw June 2020 SMOS-based SMAP files. Current workflow uses SMAP L3 HUC8 extraction, not this raw Week 2 cache. |
| `Week 2/raw_smap/` | 19.0 GB | Raw June 2020 SMAP files. Small processed outputs exist in `Week 2/outputs_smap/` (~3.7 MB). |
| `Week 3/data/satellite_raw_cache/nsidc0800/` | 2.13 GB | Two abandoned `partial_*` files, likely failed/incomplete NSIDC0800 downloads. |
| `Week 7/data/nsidc0800_raw_huc8_stream/` | 2.11 GB | Older NSIDC0800 attempt with one `.part` file; current work uses SMAP L3 product. |
| `Week 7/data/nsidc0800_raw_huc8_sparse_may_sep/` | 0.49 GB | One old `.part` file from a sparse May-September attempt. |
| `Week 1/data/CDL_2025_California.tif` | 1.19 GB | California and 2025 are outside the current Nebraska/HUC8 2018-2021 project scope. |
| `Week 1/data/CDL_2025_Nebraska.tif` | 0.38 GB | 2025 is outside the current 2018-2021 scope; keep only if you expect to revisit 2025 CDL testing. |

Potential immediate reclaim from this group: about 58.9 GB.

## Likely Archive/Delete After Verifying Outputs

These may still be useful for reproducibility, but they are not usually needed for day-to-day analysis once summaries/figures are created.

| Path | Approx. size | Check before deleting |
|---|---:|---|
| `Week 4/data/vic_et_calibration_runs/` | 46.8 GB | Keep selected summaries, figures, and configuration notes. Most size is `.txt` model output from calibration sweeps. |
| `Week 7/data/smap_l3_raw_huc8_weekly_endpoints/` | 17.9 GB | The extracted weekly endpoint table exists at `Week 7/outputs/delta_sm_pre_pbs/huc8_10180009_weekly_vic_aligned/satellite_weekly/huc8_smap_l3_endpoint_observations.csv`. Once the full-season SMAP L3 cache is complete, this July-only raw cache is probably redundant. |
| `Week 7/data/daily_delta_resample_rerun_N100/week7_daily_deltaSM_resample_rerun_N100_20260709_results.tgz` | 1.07 GB | Extracted results exist under `Week 7/data/daily_delta_resample_rerun_N100/extracted/`. Keep the archive only if you want a portable backup. |
| `Week 6/data/fresh_pbs_runs_4week_N100/*.tgz` | ~0.28 GB | Week 6 output summaries exist under `Week 6/outputs/fresh_4week_sequential_pbs_N100/`. |
| `Week 6/data/adapbs_round2_N100/*.tgz` | ~0.28 GB | Week 6 AdaPBS summaries exist under `Week 6/outputs/adapbs_4week_N100_pooled/`. |
| `Week 7/CDL_nebraska/CDL_2018_31.tif` through `CDL_2021_31.tif` | 1.1 GB | Raw CDL GeoTIFFs are useful if the CDL/SMAP cell mask needs to be rebuilt. If the mask summary is final, archive them externally rather than keeping them in the active workspace. |

## Git Repository Size

`.git/objects/` is about 24.1 GB and consists of loose objects. This may be reclaimable with a non-destructive garbage collection:

```powershell
git gc
git count-objects -vH
```

This compresses git objects but does not remove large files from git history. If `.git` remains huge after `git gc`, then large data files were probably committed at some point and would require history cleanup or a fresh repository strategy.

## Suggested First Cleanup Batch

After confirming the active full-season download is not using these paths, the safest first deletion batch would be the old raw/partial caches and out-of-window CDL files:

```powershell
$delete = @(
  "Week 2/raw_smos_smap_9k",
  "Week 2/raw_smap",
  "Week 3/data/satellite_raw_cache/nsidc0800",
  "Week 7/data/nsidc0800_raw_huc8_stream",
  "Week 7/data/nsidc0800_raw_huc8_sparse_may_sep",
  "Week 1/data/CDL_2025_California.tif",
  "Week 1/data/CDL_2025_Nebraska.tif"
)

$delete | ForEach-Object {
  if (Test-Path -LiteralPath $_) {
    Remove-Item -LiteralPath $_ -Recurse -Force
  }
}
```

Recommended follow-up after that:

1. Run `git gc`.
2. Decide whether to archive or delete `Week 4/data/vic_et_calibration_runs/`.
3. After full-season SMAP extraction is complete, consider deleting `Week 7/data/smap_l3_raw_huc8_weekly_endpoints/`.
4. Move large `.tgz` job archives to external storage if the extracted summaries are sufficient.
