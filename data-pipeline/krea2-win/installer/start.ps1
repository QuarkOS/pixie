# Starts sd-server (model stays loaded on GPU, no offload), the dashboard and the sprite runner.
# Unattended: restarts sd-server if it crashes; runner resumes where it stopped. Use -Stop to stop everything.
param([switch]$Stop, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot; Set-Location $Root
$Py = Join-Path $Root 'pyenv\Scripts\python.exe'
$Pids = Join-Path $Root 'data\pids.json'
New-Item -ItemType Directory -Force -Path (Join-Path $Root 'data'), (Join-Path $Root 'logs') | Out-Null

function Stop-All {
  if (Test-Path $Pids) {
    (Get-Content $Pids | ConvertFrom-Json).PSObject.Properties | ForEach-Object { Stop-Process -Id $_.Value -Force -ErrorAction SilentlyContinue }
    Remove-Item $Pids -Force
  }
  Get-CimInstance Win32_Process -Filter "Name='sd-server.exe'" | Where-Object { $_.ExecutablePath -like "$Root*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
if ($Stop) { Stop-All; Write-Host 'Stopped.'; exit 0 }
if (-not (Test-Path $Py)) { Write-Host 'Not installed yet - run install.bat first.' -ForegroundColor Red; exit 1 }
Stop-All

$cfg = Get-Content (Join-Path $Root 'config.json') -Raw | ConvertFrom-Json
$port = [int]$cfg.port; $dport = [int]$cfg.dashboard_port
$te = Join-Path $Root ("models\" + $cfg.text_encoder)
if (-not (Test-Path $te)) { Write-Host "Text encoder $($cfg.text_encoder) missing, falling back to Q4_K_M"; $te = Join-Path $Root 'models\Qwen3-VL-4B-Instruct-Q4_K_M.gguf' }
# 5070 Ti (16 GB) choices - UNTESTED on that card: everything on GPU (no --offload-to-cpu / --clip-on-cpu / --vae-on-cpu),
# model stays loaded in the server, LoRA applied at runtime, conditioning cache on. Same flags as the measured 3080 Ti run.
$srvArgs = @('--listen-ip','127.0.0.1','--listen-port',"$port",
  '--diffusion-model', (Join-Path $Root ("models\" + $cfg.diffusion_model)),
  '--llm', $te, '--vae', (Join-Path $Root 'models\wan_2.1_vae.safetensors'),
  '--lora-model-dir', (Join-Path $Root 'loras'), '--lora-apply-mode','at_runtime',
  '--disable-image-metadata','--conditioning-cache-size','64')   # no -v in production (multi-GB logs over 10 h)
if ($cfg.extra_server_args) { $srvArgs += ($cfg.extra_server_args -split '\s+' | Where-Object { $_ }) }
$env:CUDA_MODULE_LOADING = 'LAZY'
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
$quoted = $srvArgs | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }

function Start-Server {
  $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
  $p = Start-Process -FilePath (Join-Path $Root 'bin\sd-server.exe') -ArgumentList $quoted -WorkingDirectory (Join-Path $Root 'bin') `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $Root "logs\sd-server-$stamp.log") -RedirectStandardError (Join-Path $Root "logs\sd-server-$stamp.err.log")
  try { $p.PriorityClass = 'AboveNormal' } catch { }   # no admin needed
  $p
}
$srv  = Start-Server
$dash = Start-Process -FilePath $Py -ArgumentList @("`"$Root\app\dashboard.py`"", '--port', "$dport", '--data', "`"$Root\data`"") -WindowStyle Hidden -PassThru `
          -RedirectStandardOutput (Join-Path $Root 'logs\dashboard.log') -RedirectStandardError (Join-Path $Root 'logs\dashboard.err.log')
$run  = Start-Process -FilePath $Py -ArgumentList @("`"$Root\app\runner.py`"", '--config', "`"$Root\config.json`"") -WindowStyle Hidden -PassThru `
          -RedirectStandardOutput (Join-Path $Root 'logs\runner.log') -RedirectStandardError (Join-Path $Root 'logs\runner.err.log')
try { $run.PriorityClass = 'AboveNormal'; $dash.PriorityClass = 'BelowNormal' } catch { }
@{ server = $srv.Id; dashboard = $dash.Id; runner = $run.Id } | ConvertTo-Json | Set-Content $Pids
Write-Host "Dashboard: http://127.0.0.1:$dport   (first start loads ~10 GB of models; can take a few minutes)"
Write-Host "Sprites:   $Root\data\sprites32\ok   Logs: $Root\logs"
if (-not $NoBrowser) { Start-Sleep 2; Start-Process "http://127.0.0.1:$dport" }

# keep Windows from going to sleep while generating (no admin needed; released when this window/process ends)
try {
  Add-Type -Namespace K2 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
  [void][K2.Power]::SetThreadExecutionState([uint32]2147483649)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
} catch { Write-Host 'Could not block sleep; set Windows sleep to Never for the night.' -ForegroundColor Yellow }

# watchdog: keep sd-server alive until the runner finishes; leave the dashboard running afterwards
Write-Host 'Running. Keep this window open (minimise it) - it restarts sd-server if it crashes. The runner keeps the PC awake.'
Write-Host 'To stop and use the PC normally: double-click "Krea2 Sprites - Stop" on the desktop (progress is kept; Start resumes).'
while (-not $run.HasExited) {
  Start-Sleep 10
  if ($srv.HasExited) {
    Write-Host "$(Get-Date -Format T) sd-server exited (code $($srv.ExitCode)); restarting" -ForegroundColor Yellow
    $srv = Start-Server
    @{ server = $srv.Id; dashboard = $dash.Id; runner = $run.Id } | ConvertTo-Json | Set-Content $Pids
  }
}
Stop-Process -Id $srv.Id -Force -ErrorAction SilentlyContinue
try { [void][K2.Power]::SetThreadExecutionState([uint32]2147483648) } catch { }
Write-Host 'Queue finished. sd-server stopped (GPU free); dashboard still at the URL above. Stop-Krea2.bat closes it.' -ForegroundColor Green
