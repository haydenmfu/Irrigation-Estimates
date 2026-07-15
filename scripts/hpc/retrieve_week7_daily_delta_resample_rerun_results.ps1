$ErrorActionPreference = "Stop"

$HostName = "hopper.cac.cornell.edu"
$UserName = "hmf63"
$RunId = "week7_daily_deltaSM_resample_rerun_N100_20260709"
$RemoteRoot = "/home/fs01/hmf63/Local_Irrigation/VICFiles"
$RemoteArchive = "${RemoteRoot}/pbs_runs_week7/${RunId}_results.tgz"
$RemoteOut = "${RemoteRoot}/week7_daily_delta_resample_rerun_20260709.out"
$RemoteErr = "${RemoteRoot}/week7_daily_delta_resample_rerun_20260709.err"

$LocalWeek7 = Split-Path -Parent $MyInvocation.MyCommand.Path
$LocalData = Join-Path $LocalWeek7 "data\daily_delta_resample_rerun_N100"
$LocalArchive = Join-Path $LocalData "${RunId}_results.tgz"
$ExtractRoot = Join-Path $LocalData "extracted"

New-Item -ItemType Directory -Force -Path $LocalData | Out-Null
New-Item -ItemType Directory -Force -Path $ExtractRoot | Out-Null

Write-Host "Retrieving Hopper stdout/stderr and Week 7 daily delta-SM resample/rerun archive..."
scp "${UserName}@${HostName}:${RemoteOut}" $LocalData
scp "${UserName}@${HostName}:${RemoteErr}" $LocalData
scp "${UserName}@${HostName}:${RemoteArchive}" $LocalArchive

Write-Host "Extracting archive to $ExtractRoot ..."
tar -xzf $LocalArchive -C $ExtractRoot

$ExtractedRun = Join-Path $ExtractRoot $RunId
Write-Host ""
Write-Host "Extracted Week 7 daily delta-SM resample/rerun ensemble:"
Write-Host $ExtractedRun

Write-Host ""
Write-Host "Summarizing retrieved run..."
python (Join-Path $LocalWeek7 "summarize_week7_daily_delta_resample_rerun.py") --run-root $ExtractedRun

