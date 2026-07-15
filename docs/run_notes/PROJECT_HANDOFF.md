# Irrigation Project Handoff

Last updated: 2026-07-12

This file is meant to be the first thing a fresh Codex/chat window reads. It summarizes the current irrigation research project, where the important files are, what has already been done, and what caveats matter.

## Project Objective

Estimate irrigation water input in the Dry Spottedtail Creek / Middle North Platte-Scotts Bluff area using a VIC particle ensemble and satellite soil-moisture observations.

The current framework is a particle batch smoother (PBS):

1. Generate stochastic irrigation particles.
2. Add each particle's irrigation to VIC precipitation forcing.
3. Run VIC for each particle.
4. Compare particle soil moisture to SMAP satellite soil moisture.
5. Weight particles by likelihood.
6. Estimate posterior irrigation as the weighted mean of particle irrigation histories.

The current main test window is July 1-28, 2020, split into four weekly assimilation windows.

## Current Main Result

The current best result is the Week 7 daily Delta-SM sequential PBS run:

- Domain: VIC `basin0`; satellite comparison HUC8 `10180009`.
- Observation: SMAP L3 enhanced passive soil moisture (`SPL3SMP_E`, Version 006).
- Target: daily within-window `cropland_only` Delta-SM from HUC8 cropland-like SMAP cells.
- Likelihood scale: `sigma = 0.075 m3/m3`. This is a working assumption, not calibrated observation error.
- Algorithm: weekly state update with resampling and VIC rerun between windows.
- Prior total irrigation: `118.9795 mm`.
- Posterior total irrigation: `41.1413 mm`.
- Window ESS: `40.32, 27.85, 39.58, 22.80`.
- Interpretation: encouraging diagnostic result, clearly nonzero and not collapsed, but not yet a validated irrigation estimate.

The absolute-SM experiments are still important as negative controls:

- Absolute-SM N100 sequential run: posterior `0.0544 mm` from prior `116.4794 mm`; ESS collapsed to 1 by window 4.
- AdaPBS-style pooled absolute-SM run: posterior `1.4458 mm` from prior `76.3844 mm`.
- Interpretation: absolute SM weighting is very sensitive to level bias between VIC and SMAP.

The older weekly Delta-SM postprocessing result produced `59.5977 mm` posterior from `116.4794 mm` prior, but it did not implement weekly state resampling/rerunning. It should be treated as a diagnostic, not the main experiment.

## Current Technical Report

Primary report files:

- `Week 7/outputs/pbs_framework_technical_report/pbs_framework_technical_report.tex`
- `Week 7/outputs/pbs_framework_technical_report/pbs_framework_technical_report.pdf`
- `Week 7/outputs/pbs_framework_technical_report/figures/`

The report was recently revised so that:

- Experiment 1 is the absolute-SM sequential PBS.
- Experiment 1b is the AdaPBS-style pooled absolute-SM trial.
- Experiment 2 is now the main HUC8 SMAP L3 daily Delta-SM sequential PBS.
- The Experiment 2 section now describes only the final sequential Delta-SM experiment, not the older weekly endpoint or daily postprocessing trials.
- The final Delta-SM comparison uses the sequential resample/rerun result, not the old weekly postprocessing result.
- Precipitation references use the VIC forcing/output precipitation, not NASA POWER.
- Figure legends were adjusted to sit below titles and above plots where possible.
- The purple Abolafia-style diagnostic includes rainfall on a right-side axis.

To rebuild the PDF from the report directory:

```powershell
cd "C:\Users\f00l2\Desktop\Irrigation\Week 7\outputs\pbs_framework_technical_report"
$tectonic = "C:\Users\f00l2\Desktop\Irrigation\.tools\tectonic\tectonic-0.16.9-x86_64-pc-windows-msvc\tectonic.exe"
& $tectonic "pbs_framework_technical_report.tex" --keep-logs --keep-intermediates --outdir .
```

Known harmless compile messages:

- `inputenc package ignored with utf8 based engines`
- `microtype` footnote patch warning
- console-only `Fontconfig error: Cannot load default config file`

## Important Data and Output Locations

Main Delta-SM report/diagnostic output:

- `Week 7/outputs/delta_sm_pre_pbs/huc8_10180009_weekly_vic_aligned/diagnostics/huc8_smap_l3_delta_sm_pbs_20260708_192928/`

Important files in that folder:

- `summary_report.md`
- `DELTA_SM_PBS_FULL_REPORT.md`
- `window_summary.csv`
- `pbs_particle_weights.csv`
- `posterior_daily_irrigation.csv`
- `huc8_smap_l3_weekly_delta_selected_pairs.csv`
- `huc8_smap_l3_weekly_delta_by_mask.csv`
- `basin0_open_loop_vic_and_rainfall_weekly_reference.csv`

Main daily Delta-SM resample/rerun retrieved Hopper output:

- `Week 7/data/daily_delta_resample_rerun_N100/extracted/week7_daily_deltaSM_resample_rerun_N100_20260709/local_summary/`

Important files there:

- `summary_report.md`
- `window_summary_combined.csv`
- `posterior_daily_irrigation_combined.csv`
- `resampling_parent_state_tables_combined.csv`
- `posterior_vs_prior_daily_irrigation_resample_rerun.png`
- `effective_sample_size_by_window_resample_rerun.png`
- `resampling_parent_diversity.png`

SMAP L3 HUC8 extraction:

- `Week 7/download_extract_huc8_smap_l3.py`
- `Week 7/data/smap_l3_raw_huc8_weekly_endpoints/`
- `Week 7/outputs/delta_sm_pre_pbs/huc8_10180009_weekly_vic_aligned/satellite_weekly/huc8_smap_l3_endpoint_observations.csv`

Daily Delta-SM sensitivity output:

- `Week 7/outputs/delta_sm_pre_pbs/huc8_10180009_weekly_vic_aligned/diagnostics/huc8_smap_l3_daily_delta_sm_sensitivity_20260709_142022/`

## Important Scripts

Report and figure generation:

- `Week 7/write_delta_sm_pbs_full_report.py`
- `Week 7/plot_abolafia_rosenzweig_style_delta_sm_pbs_diagnostic.py`
- `Week 7/run_huc8_delta_sm_pbs_comparison.py`
- `Week 7/run_huc8_daily_delta_sm_sensitivity.py`
- `Week 7/summarize_week7_daily_delta_resample_rerun.py`
- `Week 6/plot_abolafia_rosenzweig_style_pbs_diagnostic.py`
- `Week 6/run_pbs_4week_sequential_absolute_sm.py`
- `research_report.mplstyle`

Hopper sequential resample/rerun workflow:

- `Week 7/pbs_generate_irrigation_window_table.py`
- `Week 7/pbs_prepare_vic_window_from_irrigation_table.py`
- `Week 7/score_hopper_daily_delta_sm_window.py`
- `Week 7/run_hopper_daily_delta_sm_resample_rerun_basin0_20260709.sh`
- `Week 7/retrieve_week7_daily_delta_resample_rerun_results.ps1`

Absolute-SM and AdaPBS-style historical scripts:

- `Week 6/run_pbs_4week_sequential_absolute_sm.py`
- `Week 6/run_pbs_absolute_sm_mlp_satsm.py`
- `Week 6/generate_adapbs_round2_irrigation_table.py`
- `Week 6/run_adapbs_4week_pooled_absolute_sm.py`

## Figure Style Notes

Current figure style aims to be more research/report-grade:

- serif body font where possible
- muted gridlines
- thin axes
- top/right spines mostly removed
- legends outside data panels or placed below titles
- colorblind-friendlier palettes where possible
- purple particle-weight scale for the Abolafia-style Delta-SM diagnostic
- rainfall plotted in sky blue with a matching right-side axis in the Abolafia-style diagnostic

The report references figures from:

- `Week 7/outputs/pbs_framework_technical_report/figures/`

If regenerating figures, make sure the refreshed PNGs are copied back into that report `figures/` folder.

As of the latest report revision, the main LaTeX report no longer includes the older weekly endpoint Delta-SM diagnostic figures or the daily sensitivity-grid figures in Experiment 2. Those files may still exist in output folders, but they are no longer part of the main Experiment 2 narrative.

## Current Caveats

These should be preserved in any scientific summary:

- The Delta-SM likelihood scale is a working assumption. It has not been calibrated.
- Daily Delta-SM targets are informative but can become too restrictive at smaller sigma values.
- The HUC8 satellite aggregation domain and VIC `basin0` model domain are not identical.
- The irrigation prior is simple: independent daily event probability `p = 0.25`, event amount Uniform(5, 30) mm/day, spatially uniform over `basin0`.
- The current run is only a 4-week July 2020 proof of concept.
- Posterior daily timing is inherited from stochastic particle proposals; weekly observations constrain trajectory plausibility more than exact daily farmer behavior.
- Absolute-SM weighting likely failed because of model-satellite level bias.
- SMAP retrieval quality flag handling should be described carefully. Earlier checks found selected weekly endpoints had quality flag 0; allowing flag 8 in extraction code did not affect those selected endpoint results. Do not casually call flag 8 "high quality" without checking product documentation.

## Recommended Next Steps

Likely next technical directions:

1. Calibrate the Delta-SM likelihood scale instead of assuming `sigma = 0.075`.
2. Extend the sequential Delta-SM PBS beyond the 4-week July 2020 test window.
3. Improve the irrigation prior using crop calendars, irrigation district information, or field/water-right constraints.
4. Check spatial consistency between HUC8 SMAP cells and the VIC `basin0` domain.
5. Test cropland-minus-control and joint crop/control targets only after checking degeneracy and error-scale assumptions.
6. Validate posterior irrigation against independent water-use, withdrawal, or field-level irrigation data if available.
7. Consider a fuller AdaPBS/AMIS implementation with proposal-density correction if adaptive proposals become central.

## Full-Season Extension Groundwork

On 2026-07-12, the next phase was scaffolded but not executed. The mentor-recommended direction is to extend the Delta-SM sequential PBS from the 4-week July 2020 proof of concept to the assumed full irrigation season, May 1-September 30, with a one-year VIC spin-up before the PBS period. Validation against online/regional irrigation data is a later step and has not yet been started.

New groundwork files:

- `Week 7/prepare_full_season_delta_sm_pbs.ps1`
  - Main PowerShell driver for local preparation.
  - Can download/extract full-season SMAP L3 data, build the daily Delta-SM Hopper target CSV, upload scripts/targets to Hopper, and optionally submit the job.
- `Week 7/build_huc8_daily_delta_targets.py`
  - Builds full-season daily within-window Delta-SM target rows from SMAP L3 observations, CDL mask classes, and basin0 open-loop VIC daily output.
- `Week 7/run_hopper_daily_delta_sm_resample_rerun_basin0_season.sh`
  - Parameterized Hopper SLURM script for May-September sequential PBS.
  - Defaults to spin-up `2019-05-01` to `2020-04-30`, then PBS `2020-05-01` to `2020-09-30`.
- `Week 7/retrieve_season_delta_resample_rerun_results.ps1`
  - Retrieves and summarizes the seasonal Hopper run after completion.
- `Week 7/FULL_SEASON_DELTA_SM_PBS_RUN_NOTES.md`
  - Human-readable command notes for the full-season workflow.

Updated helper:

- `Week 7/pbs_prepare_vic_window_from_irrigation_table.py`
  - Added `--allow-cold-start` for spin-up/state-generation runs. This comments out active `INIT_STATE` lines when no initial state is provided.

Dry-run command:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "Week 7/prepare_full_season_delta_sm_pbs.ps1"
```

Full local preparation command, once ready to download:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "Week 7/prepare_full_season_delta_sm_pbs.ps1" -DownloadSmap -BuildTargets
```

Upload and submit command:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "Week 7/prepare_full_season_delta_sm_pbs.ps1" -DownloadSmap -BuildTargets -UploadToHopper -Submit
```

Expected local full-season outputs:

- `Week 7/outputs/seasonal_delta_sm_pbs/huc8_10180009_2020_may_sep/satellite/huc8_smap_l3_observations_2020_may_sep.csv`
- `Week 7/outputs/seasonal_delta_sm_pbs/huc8_10180009_2020_may_sep/targets/daily_delta_targets_for_hopper_2020_may_sep.csv`
- `Week 7/data/smap_l3_raw_huc8_10180009_2020_may_sep/`

Important caveat: the seasonal job is much larger than the July proof-of-concept. May-September gives roughly 22 weekly windows, so N100 implies about 2,200 particle-window VIC runs plus the spin-up.

## How To Use This In A New Chat

In a new Codex window, start with:

> Read `PROJECT_HANDOFF.md` first. Then focus only on [specific task].

For report work, also open:

- `Week 7/outputs/pbs_framework_technical_report/pbs_framework_technical_report.tex`
- `Week 7/outputs/pbs_framework_technical_report/pbs_framework_technical_report.pdf`

For plotting work, also open the relevant script and the report `figures/` directory. Do not regenerate downloads or rerun Hopper jobs unless specifically requested.
