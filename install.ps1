# Install RIC Python deps + preload Silero VAD (Windows / Vision-Pipeline).
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\install.ps1
#   powershell -ExecutionPolicy Bypass -File .\install.ps1 -Python D:\Vision-Pipeline\Scripts\python.exe

param(
    [string]$Python = "D:\Vision-Pipeline\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

if (-not (Test-Path $Python)) {
    $Python = (Get-Command python -ErrorAction Stop).Source
}

Write-Host "==> Using $Python"
& $Python -m pip install -U pip wheel
& $Python -m pip install -r "$Root\requirements.txt"

Write-Host "==> Preload Silero VAD"
& $Python -c @"
import sys
from pathlib import Path
sys.path.insert(0, r'$Root')
import numpy as np
from audio.silero_vad import SileroVAD
v = SileroVAD()
r = v.classify(np.zeros(v.chunk_samples, dtype=np.float32))
print('Silero OK:', r['label'], f\"p={r['probability']:.4f}\")
"@

New-Item -ItemType Directory -Force -Path "$Root\data" | Out-Null
Write-Host "Done. Run: $Python vad_srp_phat_pipeline.py --simulate-servos"
