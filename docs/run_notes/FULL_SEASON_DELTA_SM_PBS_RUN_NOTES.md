# Full-Season Delta-SM PBS Groundwork

Prepared: 2026-07-12

This note describes the scaffold for extending the Week 7 daily Delta-SM sequential PBS from the July 2020 proof of concept to an assumed full irrigation season, May 1-September 30, with a one-year spin-up ending immediately before the season.

## Intended Run

- Main PBS period: `2020-05-01` to `2020-09-30`
- Spin-up period: `2019-05-01` to `2020-04-30`
- Window length: 7 days
- Particles: 100 by default
- Target mode: `cropland_only`
- Delta-SM likelihood scale: `0.075 m3/m3`
- Satellite product: SMAP L3 enhanced passive soil moisture, `SPL3SMP_E`, Version 006
- Domain: HUC8 `10180009` for SMAP/CDL targets, VIC `basin0` for model runs

## New/Updated Files

- `Week 7/prepare_full_season_delta_sm_pbs.ps1`
  - Main local driver.
  - Generates the full-season SMAP date list.
  - Optionally downloads/extracts SMAP L3.
  - Optionally builds the Hopper target CSV.
  - Optionally uploads/submits the Hopper run.

- `Week 7/build_huc8_daily_delta_targets.py`
  - Builds `daily_delta_targets_for_hopper_2020_may_sep.csv` from extracted SMAP observations, CDL mask classes, and basin0 open-loop VIC daily output.

- `Week 7/run_hopper_daily_delta_sm_resample_rerun_basin0_season.sh`
  - Hopper SLURM script for the seasonal sequential PBS.
  - Runs a cold-start spin-up if no initial state is provided.
  - Then runs weekly PBS windows through the irrigation season.

- `Week 7/retrieve_season_delta_resample_rerun_results.ps1`
  - Retrieves the seasonal run archive from Hopper and summarizes it locally with the existing summarizer.

- `Week 7/pbs_prepare_vic_window_from_irrigation_table.py`
  - Added `--allow-cold-start` for spin-up/state-generation runs.

## Local Preparation Commands

Dry-run the wrapper and show expected paths:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "Week 7/prepare_full_season_delta_sm_pbs.ps1"
```

Download/extract all May-September SMAP L3 granules for HUC8 cells:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "Week 7/prepare_full_season_delta_sm_pbs.ps1" -DownloadSmap
```

Build the daily Delta-SM target CSV after the observations exist:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "Week 7/prepare_full_season_delta_sm_pbs.ps1" -BuildTargets
```

Do both steps in one command:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "Week 7/prepare_full_season_delta_sm_pbs.ps1" -DownloadSmap -BuildTargets
```

Upload scripts and targets to Hopper without submitting:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "Week 7/prepare_full_season_delta_sm_pbs.ps1" -UploadToHopper
```

Upload and submit:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "Week 7/prepare_full_season_delta_sm_pbs.ps1" -DownloadSmap -BuildTargets -UploadToHopper -Submit
```

## Expected Local Outputs

Under:

```text
Week 7/outputs/seasonal_delta_sm_pbs/huc8_10180009_2020_may_sep/
```

Expected outputs include:

- `satellite/huc8_smap_l3_observations_2020_may_sep.csv`
- `satellite/smap_l3_manifest_2020_may_sep.csv`
- `satellite/smap_l3_download_audit_2020_may_sep.csv`
- `targets/daily_delta_targets_for_hopper_2020_may_sep.csv`
- `targets/daily_delta_cell_pairs_2020_may_sep.csv`
- `targets/daily_delta_targets_for_hopper_2020_may_sep.metadata.json`

Raw SMAP granules are stored under:

```text
Week 7/data/smap_l3_raw_huc8_10180009_2020_may_sep/
```

## Hopper Behavior

The seasonal SLURM script defaults to:

```text
RUN_ID=season_daily_deltaSM_resample_rerun_basin0_20200501_20200930_N100
PBS_START=2020-05-01
PBS_END=2020-09-30
SPINUP_START=2019-05-01
SPINUP_END=2020-04-30
N_PARTICLES=100
TARGET_MODE=cropland_only
DELTA_SIGMA=0.075
```

If `INITIAL_STATE` is not provided, the Hopper script first runs a one-particle, zero-irrigation open-loop spin-up from `SPINUP_START` to `SPINUP_END`, using `--allow-cold-start`. The generated end state initializes the first PBS window.

After the spin-up, the script generates 7-day windows from `PBS_START` to `PBS_END`, runs VIC for each particle/window, scores the window against the daily Delta-SM targets, resamples end states, and carries the posterior state ensemble forward.

## Retrieval After Hopper Completes

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "Week 7/retrieve_season_delta_resample_rerun_results.ps1"
```

This expects Hopper to have created:

```text
/home/fs01/hmf63/Local_Irrigation/VICFiles/pbs_runs_season/season_daily_deltaSM_resample_rerun_basin0_20200501_20200930_N100_results.tgz
```

## Notes and Caveats

- This is prepared infrastructure; it has not yet downloaded the full season or submitted the Hopper job.
- The full-season run is much larger than the July test: about 22 weekly windows for May-September.
- With 100 particles, this implies roughly 2,200 weekly particle VIC runs, plus the spin-up.
- The `0.075 m3/m3` Delta-SM likelihood scale is still a working assumption.
- The spin-up is implemented as a zero-irrigation VIC warmup/state-generation run. That is reasonable groundwork, but the exact spin-up period and cold-start assumptions should be confirmed with the mentor before treating it as final methodology.
