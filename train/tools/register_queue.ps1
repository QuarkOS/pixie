# Registers PixelAI-v3-queue (SYSTEM): starts 2 minutes from now and at every boot; the script itself waits for v2.
if (-not $env:PIXIE_ROOT) { throw "Set PIXIE_ROOT to the Pixie data root" }
$V3 = Join-Path $env:PIXIE_ROOT 'v3'
$Queue = Join-Path $V3 'code\run_v3_queue.ps1'
$Log = Join-Path $V3 'logs\queue.log'
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$Queue`"" -WorkingDirectory $V3
$t1 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2)
$t2 = New-ScheduledTaskTrigger -AtStartup
$set = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -StartWhenAvailable
$pr = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
Register-ScheduledTask -TaskName 'PixelAI-v3-queue' -Action $action -Trigger @($t1, $t2) -Settings $set -Principal $pr -Description "Pixie v3: waits for PixelAI-v2-full to finish, then env check, baselines, v2 fine-tune, pilot a, pilot b (one GPU job at a time). Logs $Log" | Out-Null
Get-ScheduledTask -TaskName 'PixelAI-v3-queue' | Select-Object TaskName, State | Format-List
(Get-ScheduledTask -TaskName 'PixelAI-v3-queue').Triggers | Select-Object StartBoundary, Enabled | Format-Table -AutoSize
