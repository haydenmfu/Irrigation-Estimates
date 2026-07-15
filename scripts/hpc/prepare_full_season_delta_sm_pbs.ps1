param(
    [int]$Year = 2020,
    [string]$Huc8 = "10180009",
    [datetime]$SeasonStart = [datetime]"2020-05-01",
    [datetime]$SeasonEnd = [datetime]"2020-09-30",
    [datetime]$SpinupStart = [datetime]"2019-05-01",
    [datetime]$SpinupEnd = [datetime]"2020-04-30",
    [int]$WindowDays = 7,
    [int]$NParticles = 100,
    [string]$TargetMode = "cropland_only",
    [double]$DeltaSigma = 0.075,
    [string]$HostName = "hopper.cac.cornell.edu",
    [string]$UserName = "hmf63",
    [string]$RemoteRoot = "/home/fs01/hmf63/Local_Irrigation/VICFiles",
    [switch]$DownloadSmap,
    [switch]$BuildTargets,
    [switch]$UploadToHopper,
    [switch]$Submit
)

$ErrorActionPreference = "Stop"

if ($SeasonEnd -lt $SeasonStart) {
    throw "SeasonEnd must be on or after SeasonStart."
}
if ($SpinupEnd -ge $SeasonStart) {
    Write-Warning "SpinupEnd is not before SeasonStart. Usually spin-up should end the day before the PBS season starts."
}

$LocalWeek7 = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $LocalWeek7
$SeasonTag = "{0}_may_sep" -f $Year
$RunLabel = "huc8_${Huc8}_${SeasonTag}"
$SeasonRoot = Join-Path $LocalWeek7 "outputs\seasonal_delta_sm_pbs\$RunLabel"
$SatelliteDir = Join-Path $SeasonRoot "satellite"
$TargetDir = Join-Path $SeasonRoot "targets"
$RawDir = Join-Path $LocalWeek7 "data\smap_l3_raw_${RunLabel}"
$LogsDir = Join-Path $SeasonRoot "logs"

$ObservationCsv = Join-Path $SatelliteDir "huc8_smap_l3_observations_${SeasonTag}.csv"
$ManifestCsv = Join-Path $SatelliteDir "smap_l3_manifest_${SeasonTag}.csv"
$AuditCsv = Join-Path $SatelliteDir "smap_l3_download_audit_${SeasonTag}.csv"
$TargetCsv = Join-Path $TargetDir "daily_delta_targets_for_hopper_${SeasonTag}.csv"
$TargetPairsCsv = Join-Path $TargetDir "daily_delta_cell_pairs_${SeasonTag}.csv"
$TargetMetadata = Join-Path $TargetDir "daily_delta_targets_for_hopper_${SeasonTag}.metadata.json"

$SelectedCells = Join-Path $LocalWeek7 "outputs\delta_sm_pre_pbs\huc8_10180009_cdl_crop30_power_rain_partial_current\huc8_selected_smap_cells.csv"
$CdlSummary = Join-Path $LocalWeek7 "outputs\delta_sm_pre_pbs\huc8_10180009_cdl_crop30_power_rain_partial_current\huc8_smap_cell_cropland_summary.csv"
$VicDaily = Join-Path $ProjectRoot "Week 3\data\VIC_basin0_outputs\dry_spottedtail_creek_vic_basin_daily_summary_for_satellite.csv"

New-Item -ItemType Directory -Force -Path $SatelliteDir, $TargetDir, $RawDir, $LogsDir | Out-Null

$Dates = New-Object System.Collections.Generic.List[string]
for ($d = $SeasonStart.Date; $d -le $SeasonEnd.Date; $d = $d.AddDays(1)) {
    $Dates.Add($d.ToString("yyyy-MM-dd"))
}
$DateCsv = ($Dates -join ",")

Write-Host "Full-season Delta-SM PBS preparation"
Write-Host "  HUC8: $Huc8"
Write-Host "  Season: $($SeasonStart.ToString('yyyy-MM-dd')) to $($SeasonEnd.ToString('yyyy-MM-dd')) ($($Dates.Count) SMAP target dates)"
Write-Host "  Spin-up: $($SpinupStart.ToString('yyyy-MM-dd')) to $($SpinupEnd.ToString('yyyy-MM-dd'))"
Write-Host "  Local season root: $SeasonRoot"
Write-Host "  Raw SMAP dir: $RawDir"
Write-Host ""

if ($DownloadSmap) {
    Write-Host "Downloading/extracting SMAP L3 observations for all season dates..."
    python (Join-Path $LocalWeek7 "download_extract_huc8_smap_l3.py") `
        --dates $DateCsv `
        --selected-cells $SelectedCells `
        --raw-dir $RawDir `
        --out-csv $ObservationCsv `
        --manifest-csv $ManifestCsv `
        --audit-csv $AuditCsv
} else {
    Write-Host "Skipping SMAP download/extraction. Use -DownloadSmap to run it."
    if (!(Test-Path -LiteralPath $ObservationCsv)) {
        Write-Warning "Observation CSV does not exist yet: $ObservationCsv"
    }
}

if ($BuildTargets) {
    if (!(Test-Path -LiteralPath $ObservationCsv)) {
        throw "Cannot build targets because observation CSV is missing: $ObservationCsv. Run with -DownloadSmap first."
    }
    Write-Host "Building daily within-window Delta-SM target CSV for Hopper..."
    python (Join-Path $LocalWeek7 "build_huc8_daily_delta_targets.py") `
        --endpoint-observations $ObservationCsv `
        --cdl-summary $CdlSummary `
        --vic-basin-daily $VicDaily `
        --out-csv $TargetCsv `
        --cell-pairs-csv $TargetPairsCsv `
        --metadata-json $TargetMetadata `
        --start-date $($SeasonStart.ToString("yyyy-MM-dd")) `
        --end-date $($SeasonEnd.ToString("yyyy-MM-dd")) `
        --cdl-year $Year `
        --window-days $WindowDays
} else {
    Write-Host "Skipping target build. Use -BuildTargets after SMAP observations are available."
    if (!(Test-Path -LiteralPath $TargetCsv)) {
        Write-Warning "Target CSV does not exist yet: $TargetCsv"
    }
}

$RunId = "season_daily_deltaSM_resample_rerun_basin0_$($SeasonStart.ToString('yyyyMMdd'))_$($SeasonEnd.ToString('yyyyMMdd'))_N$NParticles"
$RemoteTargetLeaf = "daily_delta_targets_for_hopper_season_latest.csv"
$RemoteJobLeaf = "run_hopper_daily_delta_sm_resample_rerun_basin0_season.sh"
$RemoteTarget = "${RemoteRoot}/${RemoteTargetLeaf}"
$RemoteJob = "${RemoteRoot}/${RemoteJobLeaf}"
$SubmitExport = "ALL,RUN_ID=${RunId},PBS_START=$($SeasonStart.ToString('yyyy-MM-dd')),PBS_END=$($SeasonEnd.ToString('yyyy-MM-dd')),SPINUP_START=$($SpinupStart.ToString('yyyy-MM-dd')),SPINUP_END=$($SpinupEnd.ToString('yyyy-MM-dd')),WINDOW_DAYS=${WindowDays},N_PARTICLES=${NParticles},TARGET_MODE=${TargetMode},DELTA_SIGMA=${DeltaSigma},TARGET_CSV=${RemoteTarget}"

if ($UploadToHopper) {
    if (!(Test-Path -LiteralPath $TargetCsv)) {
        throw "Cannot upload because target CSV is missing: $TargetCsv. Run with -BuildTargets first."
    }
    $Scripts = @(
        (Join-Path $LocalWeek7 "pbs_generate_irrigation_window_table.py"),
        (Join-Path $LocalWeek7 "pbs_prepare_vic_window_from_irrigation_table.py"),
        (Join-Path $LocalWeek7 "score_hopper_daily_delta_sm_window.py"),
        (Join-Path $LocalWeek7 "run_hopper_daily_delta_sm_resample_rerun_basin0_season.sh")
    )
    Write-Host "Uploading scripts and target CSV to Hopper..."
    scp $Scripts $TargetCsv "${UserName}@${HostName}:${RemoteRoot}/"
    ssh "${UserName}@${HostName}" "cd ${RemoteRoot} && mv $(Split-Path -Leaf $TargetCsv) ${RemoteTargetLeaf} && chmod +x ${RemoteJobLeaf}"

    if ($Submit) {
        Write-Host "Submitting seasonal PBS job to Hopper..."
        ssh "${UserName}@${HostName}" "cd ${RemoteRoot} && sbatch --export=${SubmitExport} ${RemoteJobLeaf}"
    } else {
        Write-Host "Upload complete. To submit manually:"
        Write-Host "ssh ${UserName}@${HostName} `"cd ${RemoteRoot} && sbatch --export=${SubmitExport} ${RemoteJobLeaf}`""
    }
} else {
    Write-Host "Skipping Hopper upload. Use -UploadToHopper to copy scripts/targets, and -Submit to submit."
}

Write-Host ""
Write-Host "Prepared paths:"
Write-Host "  Observation CSV: $ObservationCsv"
Write-Host "  Target CSV:      $TargetCsv"
Write-Host "  Target metadata: $TargetMetadata"
Write-Host "  Run ID:          $RunId"
