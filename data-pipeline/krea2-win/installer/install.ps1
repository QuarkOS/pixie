# Krea 2 sprite pipeline - Windows installer (RTX 5070 Ti / Blackwell sm_120 target).
# Everything is installed inside this folder. Re-running is safe: finished downloads are skipped (hash-checked).
param([switch]$Q8TextEncoder, [switch]$NoStart, [switch]$NoShortcut)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$Root = $PSScriptRoot
Set-Location $Root
$Log = Join-Path $Root 'install.log'
Start-Transcript -Path $Log -Append | Out-Null

function Step($m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "ERROR: $m" -ForegroundColor Red; Stop-Transcript | Out-Null; exit 1 }

# ---- pinned downloads (verified to exist on 2026-11-14) ----
$SdTag   = 'master-1047-984eb06'   # same commit as the working 3080 Ti build; Windows CUDA build compiled with CUDA 12.8.1 for archs 61..90,100,120
$Files = @(
  @{ Name='sd-cuda.zip';  Dir='downloads'; Url="https://github.com/leejet/stable-diffusion.cpp/releases/download/$SdTag/sd-master-984eb06-bin-win-cuda12-x64.zip"; Sha='d21008534fd7ec8d4bb0bde78c728530a819a510654ba9c04bd279dab40ac8ab' },
  @{ Name='cudart.zip';   Dir='downloads'; Url="https://github.com/leejet/stable-diffusion.cpp/releases/download/$SdTag/cudart-sd-bin-win-cu12-x64.zip"; Sha='fe20366827d357c00797eebb58244dddab7fd9a348d70090c3871004c320f38d' },
  @{ Name='uv.zip';       Dir='downloads'; Url='https://github.com/astral-sh/uv/releases/download/0.13.8/uv-x86_64-pc-windows-msvc.zip'; Sha='529e78a9cec6cd7e3e014f7ecd8bda0c113969d4ed0f2efeedb85c30943325b8' },
  @{ Name='krea2_turbo-iq4_xs.gguf'; Dir='models'; Url='https://huggingface.co/gguf-org/krea-2-gguf/resolve/main/krea2_turbo-iq4_xs.gguf'; Sha='56e1bfb0318693e4d0882e48c72286b7ad98f72dc9c9e5c46a5164c6cca7c77d' },
  @{ Name='Qwen3-VL-4B-Instruct-Q4_K_M.gguf'; Dir='models'; Url='https://huggingface.co/unsloth/Qwen3-VL-4B-Instruct-GGUF/resolve/main/Qwen3-VL-4B-Instruct-Q4_K_M.gguf'; Sha='d4dcd426bfba75752a312b266b80fec8136fbaca13c62d93b7ac41fa67f0492b' },
  @{ Name='wan_2.1_vae.safetensors'; Dir='models'; Url='https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors'; Size=253815318 }
)
if ($Q8TextEncoder) {  # optional, UNTESTED: higher-quality text encoder (4.3 GB). Set config.json text_encoder to use it.
  $Files += @{ Name='Qwen3-VL-4B-Instruct-Q8_0.gguf'; Dir='models'; Url='https://huggingface.co/unsloth/Qwen3-VL-4B-Instruct-GGUF/resolve/main/Qwen3-VL-4B-Instruct-Q8_0.gguf'; Sha='e30b41d2f2cc48149dfbba1b3fb2c023c7e6bf2def75833d4d43820df75efeb3' }
}

# ---- 1. GPU / driver check ----
Step 'Checking NVIDIA GPU and driver (needs CUDA 12.8+ support, i.e. driver 570 or newer)'
$smi = (Get-Command nvidia-smi -ErrorAction SilentlyContinue).Source
if (-not $smi -and (Test-Path "$env:SystemRoot\System32\nvidia-smi.exe")) { $smi = "$env:SystemRoot\System32\nvidia-smi.exe" }
if (-not $smi) { Fail 'nvidia-smi not found. Install the latest NVIDIA Game Ready / Studio driver first: https://www.nvidia.com/Download/index.aspx' }
$hdr = ((& $smi 2>&1) | Out-String)
$gpu = ((& $smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1) | Select-Object -First 1) -as [string]
$m = [regex]::Match($hdr, 'CUDA(?: Driver| UMD)? Version\s*:?\s*([0-9]+)\.([0-9]+)')
$drv = [regex]::Match($gpu, '([0-9]{3,})\.[0-9]+\s*$')
if ($m.Success) {
  $cuda = [version]("{0}.{1}" -f $m.Groups[1].Value, $m.Groups[2].Value)
  Write-Host "GPU: $gpu   driver CUDA: $cuda"
  if ($cuda -lt [version]'12.8') { Fail "Driver supports CUDA $cuda, need 12.8+ for RTX 50-series. Update the NVIDIA driver and run install.bat again." }
} elseif ($drv.Success) {
  $dv = [int]$drv.Groups[1].Value
  Write-Host "GPU: $gpu   (CUDA version not shown by nvidia-smi; using driver version $dv)"
  if ($dv -lt 570) { Fail "NVIDIA driver $dv is too old for RTX 50-series (need 570+). Update the driver and run install.bat again." }
} else {
  Write-Host "WARNING: could not read the CUDA or driver version from nvidia-smi. Continuing anyway. nvidia-smi said:" -ForegroundColor Yellow
  Write-Host $hdr
}
if ($gpu -notmatch '50[0-9]0') { Write-Host 'Note: this package is tuned for an RTX 5070 Ti; other NVIDIA cards with 12 GB+ should also work.' -ForegroundColor Yellow }

# ---- 2. downloads (curl.exe ships with Windows 10 1803+; resumable) ----
$curl = "$env:SystemRoot\System32\curl.exe"
if (-not (Test-Path $curl)) { Fail 'curl.exe not found (needs Windows 10 1803 or newer).' }
function Get-Sha($p) { (Get-FileHash -Algorithm SHA256 -Path $p).Hash.ToLower() }
foreach ($f in $Files) {
  $dir = Join-Path $Root $f.Dir; New-Item -ItemType Directory -Force -Path $dir | Out-Null
  $dst = Join-Path $dir $f.Name; $ok = Join-Path $dir ($f.Name + '.ok')
  if ((Test-Path $dst) -and (Test-Path $ok)) { Write-Host "have $($f.Name)"; continue }
  Step "Downloading $($f.Name)"
  $tries = 0
  while ($true) {
    $tries++
    & $curl -L --fail --retry 10 --retry-delay 5 --retry-all-errors -C - -o "$dst.part" $f.Url
    if ($LASTEXITCODE -eq 0) { break }
    if ($tries -ge 5) { Fail "Download failed: $($f.Url)" }
    Write-Host "retrying ($tries)..."; Start-Sleep -Seconds 15
  }
  if ($f.Sha) {
    $h = Get-Sha "$dst.part"
    if ($h -ne $f.Sha) { Remove-Item "$dst.part" -Force; Fail "Checksum mismatch for $($f.Name) (got $h). Run install.bat again." }
  } elseif ($f.Size) {
    if ((Get-Item "$dst.part").Length -ne $f.Size) { Remove-Item "$dst.part" -Force; Fail "Size mismatch for $($f.Name). Run install.bat again." }
  }
  Move-Item -Force "$dst.part" $dst; Set-Content -Path $ok -Value 'ok'
}

# ---- 3. unpack stable-diffusion.cpp + CUDA runtime ----
function Unzip($zip, $to) {
  New-Item -ItemType Directory -Force -Path $to | Out-Null
  $tar = "$env:SystemRoot\System32\tar.exe"
  if (Test-Path $tar) { & $tar -xf $zip -C $to; if ($LASTEXITCODE -eq 0) { return } }
  Expand-Archive -Force -Path $zip -DestinationPath $to
}
$bin = Join-Path $Root 'bin'
if (-not (Test-Path (Join-Path $bin 'sd-server.exe'))) {
  Step 'Unpacking stable-diffusion.cpp (CUDA) and CUDA runtime'
  Unzip (Join-Path $Root 'downloads\sd-cuda.zip') $bin
  Unzip (Join-Path $Root 'downloads\cudart.zip') $bin
  # flatten if an archive contained a subfolder
  Get-ChildItem $bin -Recurse -File -Include *.dll,*.exe | Where-Object { $_.DirectoryName -ne $bin } | ForEach-Object { Move-Item -Force $_.FullName $bin }
}
if (-not (Test-Path (Join-Path $bin 'sd-server.exe'))) { Fail 'sd-server.exe missing after unpack.' }

# ---- 3b. bundled LoRA (rejoin split parts if the zip was split for transfer) ----
$lora = Join-Path $Root 'loras\pix32-final.safetensors'
$parts = Get-ChildItem (Join-Path $Root 'loras') -Filter 'pix32-final.safetensors.part*' -ErrorAction SilentlyContinue | Sort-Object Name
if (-not (Test-Path $lora) -and $parts) {
  $out = [IO.File]::Create($lora); foreach ($p in $parts) { $b = [IO.File]::ReadAllBytes($p.FullName); $out.Write($b, 0, $b.Length) }; $out.Close()
}
if (-not (Test-Path $lora)) { Fail 'loras\pix32-final.safetensors missing from the package.' }
if ((Get-Sha $lora) -ne '8aa3dcdbf17687a3f74cb6e474a3f91417056232e7993b28c86ddfa6b92aeebe') { Fail 'LoRA checksum mismatch - package is damaged, copy it again.' }

# ---- 4. portable Python via uv (local to this folder) ----
Step 'Setting up portable Python (uv) for extractor + dashboard'
$tools = Join-Path $Root 'tools'
if (-not (Test-Path (Join-Path $tools 'uv.exe'))) { Unzip (Join-Path $Root 'downloads\uv.zip') $tools }
$uvExe = Get-ChildItem $tools -Recurse -Filter uv.exe | Select-Object -First 1 -ExpandProperty FullName
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Root 'tools\python'
$env:UV_CACHE_DIR = Join-Path $Root 'tools\uv-cache'
$env:UV_NO_CONFIG = '1'
$venv = Join-Path $Root 'pyenv'
if (-not (Test-Path (Join-Path $venv 'Scripts\python.exe'))) {
  & $uvExe venv --python 3.12 --python-preference only-managed $venv; if ($LASTEXITCODE) { Fail 'uv venv failed' }
}
& $uvExe pip install --python (Join-Path $venv 'Scripts\python.exe') numpy==2.2.6 pillow==11.3.0 scipy==1.15.3
if ($LASTEXITCODE) { Fail 'pip install failed' }

# ---- 5. quick self-test: server binary starts and sees the GPU ----
Step 'Self-test: sd-server --help'
$ErrorActionPreference = 'Continue'
& (Join-Path $bin 'sd-server.exe') --help *> (Join-Path $Root 'logs-sdserver-help.txt')
$ErrorActionPreference = 'Stop'
if (-not (Select-String -Path (Join-Path $Root 'logs-sdserver-help.txt') -Pattern 'lora-model-dir' -Quiet)) { Fail 'sd-server.exe did not run (see logs-sdserver-help.txt). Missing VC++ runtime? Install https://aka.ms/vs/17/release/vc_redist.x64.exe' }

# ---- 6. shortcuts ----
if (-not $NoShortcut) {
  Step 'Creating desktop shortcuts'
  $ws = New-Object -ComObject WScript.Shell
  $desk = [Environment]::GetFolderPath('Desktop')
  foreach ($s in @(@('Krea2 Sprites - Start', 'Start-Krea2.bat'), @('Krea2 Sprites - Stop', 'Stop-Krea2.bat'))) {
    $lnk = $ws.CreateShortcut((Join-Path $desk ($s[0] + '.lnk')))
    $lnk.TargetPath = Join-Path $Root $s[1]; $lnk.WorkingDirectory = $Root; $lnk.Save()
  }
}
Write-Host "`nInstall finished." -ForegroundColor Green
Stop-Transcript | Out-Null
if (-not $NoStart) { & (Join-Path $Root 'start.ps1') }
