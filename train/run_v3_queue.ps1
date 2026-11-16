# Pixie v3 queue (scheduled task PixelAI-v3-queue). Waits until the v2 run PixelAI-v2-full has finished and
# written train_summary.json + its final checkpoints, then runs one GPU job at a time:
#   env check -> baseline rescore -> v2 fine-tune on restored data (+ rare samples) -> pilot a (pix256) -> pilot b (P50) -> compare
# Every stage writes into a NEW folder and has a done-marker, so a restart resumes where it stopped.
# Never touches the v2 run folder except reading it.
param([switch]$DryRun, [string]$V2RunOverride = '', [string]$V2TaskOverride = '', [string]$V2ProcPatternOverride = '')
# -DryRun: evaluate the v2-finished check once, list every stage with its command and done-marker state, start nothing.
#          -V2RunOverride/-V2TaskOverride/-V2ProcPatternOverride point the check at a fake folder/task for testing.
$ErrorActionPreference = 'Continue'
if (-not $env:PIXIE_ROOT) { throw "Set PIXIE_ROOT to the Pixie data root" }
$P = $env:PIXIE_ROOT
$V3 = Join-Path $P 'v3'
$PY = Join-Path $P 'venv\Scripts\python.exe'
$PYT = Join-Path $V3 'venv-triton\Scripts\python.exe'
$RUNS = Join-Path $P 'runs'
$V2RUN = Join-Path $RUNS 'v2-L12-full'
$V2TASK = 'PixelAI-v2-full'
$V2PROC = '*train_v2.py*'
if ($V2RunOverride) { $V2RUN = $V2RunOverride }; if ($V2TaskOverride) { $V2TASK = $V2TaskOverride }; if ($V2ProcPatternOverride) { $V2PROC = $V2ProcPatternOverride }
$PAL = Join-Path $P 'data\subset150k\palette.npy'
$CLIP = Join-Path $V3 'clip\clip_l14_text_l11_f32.pt'
$V2TAB = Join-Path $P 'v2\clip_tok\clip_l14_token_embedding_f16.npy'
if (-not $env:PIXIE_PYTHON) { $PushPy = 'python' } else { $PushPy = $env:PIXIE_PYTHON }
$PushScript = Join-Path $P 'tools\push_training.py'
$LOG = "$V3\logs\queue.log"; $STATUS = "$V3\queue_status.json"
if ($DryRun) { $LOG = "$V3\logs\queue_dryrun.log"; $STATUS = "$V3\queue_status_dryrun.json" }
New-Item -ItemType Directory -Force -Path "$V3\logs", "$V3\results" | Out-Null

function Log($m) { $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m"; Add-Content -Path $LOG -Value $line; Write-Host $line }
function SetStatus($stage, $state, $detail) {
  $o = [ordered]@{ stage = $stage; state = $state; detail = $detail; time = (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') }
  ($o | ConvertTo-Json -Compress) | Set-Content -Path $STATUS -Encoding UTF8
}

function V2Check {
  $summary = Test-Path "$V2RUN\train_summary.json"
  $state = "$((Get-ScheduledTask -TaskName $V2TASK -ErrorAction SilentlyContinue).State)"
  $proc = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like $V2PROC })
  $best = Test-Path "$V2RUN\best.pt"; $last = Test-Path "$V2RUN\last.pt"
  $done = $summary -and $state -ne 'Running' -and $proc.Count -eq 0 -and $best -and $last
  [pscustomobject]@{ done = $done; train_summary = $summary; task_state = $state; train_v2_procs = $proc.Count; best_pt = $best; last_pt = $last }
}

if ($DryRun) {
  $c = V2Check
  Log "DRY RUN (nothing is started). v2 run $V2RUN, task $V2TASK"
  Log ("v2 check: " + ($c | ConvertTo-Json -Compress))
  Log ("decision: " + $(if ($c.done) { 'v2 finished -> queue would start stage 1 now (after 60 s settle)' } else { 'v2 not finished -> keep waiting, GPU untouched, re-check every 60 s' }))
}

# ---- single instance
if (-not $DryRun) {
$lockPath = "$V3\queue.lock"
try { $lock = [System.IO.File]::Open($lockPath, 'OpenOrCreate', 'ReadWrite', 'None') } catch { Log "another queue instance holds the lock; exiting"; exit 0 }
Log "queue started (pid $PID)"

# ---- wait for v2 (last.pt + train_summary.json are the trainer's final two writes)
$notRunningSince = $null
while ($true) {
  $c = V2Check
  if ($c.done) { break }
  if (-not $c.train_summary -and $c.task_state -ne 'Running' -and $c.train_v2_procs -eq 0) {
    if (-not $notRunningSince) { $notRunningSince = Get-Date }
    SetStatus 'wait_v2' 'blocked' "$V2TASK is not running and has no train_summary.json since $notRunningSince; waiting, GPU untouched"
  } else {
    $notRunningSince = $null
    SetStatus 'wait_v2' 'waiting' "v2 task state $($c.task_state), train_summary $($c.train_summary), last.pt $($c.last_pt)"
  }
  Start-Sleep -Seconds 60
}
Log "v2 finished (train_summary.json + last.pt present, task $((Get-ScheduledTask -TaskName $V2TASK).State)); settling 60 s"
Start-Sleep -Seconds 60

# ---- live training panel: push v2's final snapshot once, then hand the panel to the v3 runs
if (-not (Test-Path "$V3\logs\pusher_handover.txt")) {
  & $PushPy -u $PushScript | Out-Null
  schtasks /Change /TN PixelAI-v2-training-push /DISABLE | Out-Null
  "v2 final snapshot pushed and PixelAI-v2-training-push disabled at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'); the queue pushes v3 runs with the same script (env PIXELAI_V2_RUN). Re-enable with: schtasks /Change /TN PixelAI-v2-training-push /ENABLE" | Set-Content "$V3\logs\pusher_handover.txt"
  Log "pusher handover done"
}
}

function Push($run) {
  $env:PIXELAI_V2_RUN = $run; $env:PIXELAI_V2_PILOT = "$V3\no-pilot"
  try { $p = Start-Process -FilePath $PushPy -ArgumentList '-u', $PushScript -NoNewWindow -PassThru; $p.WaitForExit(60000) | Out-Null } catch {}
}

function RunStage($name, $marker, $exe, $argStr, $outDir, $pushRun, $timeoutMin) {
  if (Test-Path $marker) { Log "stage $name already done ($marker)"; return $true }
  if ($DryRun) { Log "would run stage $name -> $outDir (marker $marker, cap $timeoutMin min): $exe $argStr"; return $true }
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  SetStatus $name 'running' $outDir
  Log "stage $name start: $exe $argStr"
  $p = Start-Process -FilePath $exe -ArgumentList $argStr -NoNewWindow -PassThru -RedirectStandardOutput "$outDir\stdout_$name.log" -RedirectStandardError "$outDir\stderr_$name.log"
  $null = $p.Handle  # keep the handle so ExitCode is readable after exit
  $t0 = Get-Date; $lastPush = (Get-Date).AddMinutes(-10)
  while (-not $p.HasExited) {
    Start-Sleep -Seconds 20
    if ($pushRun -and ((Get-Date) - $lastPush).TotalMinutes -ge 5) { Push $pushRun; $lastPush = Get-Date }
    if ($timeoutMin -gt 0 -and ((Get-Date) - $t0).TotalMinutes -gt $timeoutMin) { Log "stage $name exceeded $timeoutMin min, stopping it"; Stop-Process -Id $p.Id -Force; break }
  }
  $p.WaitForExit(); $code = $p.ExitCode
  if ($pushRun) { Push $pushRun }
  $ok = Test-Path $marker
  Log "stage $name exit $code, marker $ok, $([math]::Round(((Get-Date) - $t0).TotalMinutes, 1)) min"
  SetStatus $name ($(if ($ok) { 'done' } else { 'failed' })) "exit $code"
  return $ok
}

# 1. env check (~15 min cap)
$envDir = "$V3\results"
RunStage 'env_main' "$envDir\env_main.json" $PY "-u $V3\code\env_check.py $envDir\env_main.json" $envDir $null 15 | Out-Null
RunStage 'env_triton' "$envDir\env_triton.json" $PYT "-u $V3\code\env_check.py $envDir\env_triton.json --compile" $envDir $null 15 | Out-Null

# 2. baselines on the new probe/iid splits (v1 cooldown A, v1 cooldown B, v2 best)
RunStage 'baselines' "$V3\results\baselines.json" $PY ("-u $V3\code\rescore_v3.py --data $V3\data --cache $V3\cache --pixelgen-root $V3\code\pixelgen_copy " +
  "--v2-clip-table $V2TAB --palette $PAL --samples $V3\results\baseline_samples " +
  "--v1 v1_cooldownA=$RUNS\full1p4m-L10-cont-cooldown\best.pt --v1 v1_cooldownB_cfg=$RUNS\full1p4m-L10-final32-cfg\best.pt " +
  "--v2 v2_best=$V2RUN\best.pt --out $V3\results\baselines.json") "$V3\results" $null 30 | Out-Null

# 3. v2 fine-tune on the restored mix: 2 epochs, LR 5e-5 -> 1e-5, new folder; samples rare prompts at the end
$FT = "$RUNS\v3-v2ft-restored"
RunStage 'v2ft' "$FT\train_summary.json" $PY ("-u $V3\code\train_v3.py --arch v2ft --init $V2RUN\best.pt --v2-clip-table $V2TAB " +
  "--data $V3\data --cache $V3\cache --palette $PAL --out $FT --epochs 2 --batch 128 --lr 5e-5 --lr-min 1e-5 --warmup 200 " +
  "--dropout 0.1 --eval-at-start --eval-every-images 678174 --save-every-images 1356348 --max-hours 1.5") $FT $FT 120 | Out-Null

# 4a. pilot a: v2-size pixel model + online CLIP-L penultimate text, 8M images
$PA = "$RUNS\v3-pilot-a-pix256"
RunStage 'pilot_a' "$PA\train_summary.json" $PY ("-u $V3\code\train_v3.py --arch pix --clip $CLIP --data $V3\data --cache $V3\cache " +
  "--palette $PAL --out $PA --images 8000000 --batch 128 --lr 2e-4 --lr-min 1e-5 --warmup 1000 --layers 12 --width 256 --heads 8 " +
  "--dropout 0.1 --eval-every-images 1000000 --save-every-images 1000000 --max-hours 3.5") $PA $PA 240 | Out-Null

# 4b. pilot b: P50 (d512 x L16, 2x2 patches, slot-aware causal MLP head), 8M images
$PB = "$RUNS\v3-pilot-b-p50"
$okb = RunStage 'pilot_b' "$PB\train_summary.json" $PY ("-u $V3\code\train_v3.py --arch p50 --clip $CLIP --data $V3\data --cache $V3\cache " +
  "--palette $PAL --out $PB --images 8000000 --batch 128 --lr 3e-4 --lr-min 1.5e-5 --warmup 1000 --layers 16 --width 512 --heads 8 " +
  "--dropout 0.1 --eval-every-images 1000000 --save-every-images 1000000 --max-hours 3.5") $PB $PB 240
if (-not $DryRun -and -not $okb -and -not (Test-Path "$PB\last.pt")) {
  # only if it died before its first checkpoint (e.g. out of memory): retry with batch 64 in a separate folder
  $PB2 = "$RUNS\v3-pilot-b-p50-b64"
  Log "pilot_b failed before any checkpoint; retrying with batch 64 in $PB2"
  RunStage 'pilot_b_b64' "$PB2\train_summary.json" $PY ("-u $V3\code\train_v3.py --arch p50 --clip $CLIP --data $V3\data --cache $V3\cache " +
    "--palette $PAL --out $PB2 --images 8000000 --batch 64 --lr 2e-4 --lr-min 1e-5 --warmup 2000 --layers 16 --width 512 --heads 8 " +
    "--dropout 0.1 --eval-every-images 1000000 --save-every-images 1000000 --max-hours 4") $PB2 $PB2 270 | Out-Null
}

# 5. summary tables
if ($DryRun) { Log "would run compare_v3.py -> $V3\results\results.md; dry run finished"; exit 0 }
& $PY -u "$V3\code\compare_v3.py" $RUNS $V3 | Out-File -Encoding utf8 "$V3\logs\compare.log"
SetStatus 'all' 'done' "$V3\results\results.md"
Log "queue finished"
$lock.Close()
