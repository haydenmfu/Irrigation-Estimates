param(
    [string]$RunId = "season_daily_deltaSM_resample_rerun_basin0_20200501_20200930_N100",
    [string]$HostName = "hopper.cac.cornell.edu",
    [string]$UserName = "hmf63",
    [string]$RemoteRoot = "/home/fs01/hmf63/Local_Irrigation/VICFiles",
    [string]$LocalLabel = "season_2020_may_sep_N100"
)

$ErrorActionPreference = "Stop"

$LocalWeek7 = Split-Path -Parent $MyInvocation.MyCommand.Path
$LocalData = Join-Path $LocalWeek7 "data\season_delta_resample_rerun\$LocalLabel"
$ExtractRoot = Join-Path $LocalData "extracted"
$LocalArchive = Join-Path $LocalData "${RunId}_results.tgz"

$RemoteArchive = "${RemoteRoot}/pbs_runs_season/${RunId}_results.tgz"

New-Item -ItemType Directory -Force -Path $LocalData, $ExtractRoot | Out-Null

Write-Host "Retrieving seasonal Delta-SM PBS archive from Hopper..."
Write-Host "  Remote: $RemoteArchive"
Write-Host "  Local:  $LocalArchive"
scp "${UserName}@${HostName}:${RemoteArchive}" $LocalArchive

Write-Host "Extracting archive to $ExtractRoot ..."
tar -xzf $LocalArchive -C $ExtractRoot

$ExtractedRun = Join-Path $ExtractRoot $RunId
if (!(Test-Path -LiteralPath $ExtractedRun)) {
    throw "Expected extracted run folder not found: $ExtractedRun"
}

Write-Host "Summarizing retrieved seasonal run..."
python (Join-Path $LocalWeek7 "summarize_week7_daily_delta_resample_rerun.py") --run-root $ExtractedRun

Write-Host ""
Write-Host "Extracted seasonal run:"
Write-Host $ExtractedRun
Write-Host "Local summary:"
Write-Host (Join-Path $ExtractedRun "local_summary")
