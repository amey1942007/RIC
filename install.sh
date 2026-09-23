#!/usr/bin/env bash
# install.sh — install RIC deps, Silero VAD model, and (on Pi) system packages.
# Usage:
#   ./install.sh              # venv + pip + preload Silero
#   ./install.sh --overlay    # also compile/install ric-inmp441-4ch DT overlay (needs sudo)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

WITH_OVERLAY=0
for arg in "$@"; do
  case "$arg" in
    --overlay) WITH_OVERLAY=1 ;;
    -h|--help)
      echo "Usage: ./install.sh [--overlay]"
      exit 0
      ;;
  esac
done

echo "==> RIC install from $ROOT"

# System packages (Debian / Raspberry Pi OS)
if command -v apt-get >/dev/null 2>&1; then
  echo "==> apt packages"
  sudo apt-get update -qq
  sudo apt-get install -y -qq \
    python3-venv python3-pip python3-tk python3-smbus \
    i2c-tools libportaudio2 portaudio19-dev \
    device-tree-compiler \
    liblgpio-dev python3-lgpio swig \
    python3-picamera2 python3-libcamera python3-pil python3-pil.imagetk \
    rpicam-apps || true

  # Enable I2C for Waveshare Pan-Tilt HAT (PCA9685 @ 0x40)
  if command -v raspi-config >/dev/null 2>&1; then
    echo "==> enabling I2C (Waveshare HAT)"
    sudo raspi-config nonint do_i2c 0 || true
  fi
fi

# Virtualenv — with system site-packages so picamera2 / libcamera (apt-only,
# used for the Camera Module 3 preview) are importable from the venv.
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "==> creating .venv (system-site-packages for picamera2)"
  python3 -m venv --system-site-packages "$ROOT/.venv"
elif grep -q 'include-system-site-packages = false' "$ROOT/.venv/pyvenv.cfg" 2>/dev/null; then
  echo "==> enabling system-site-packages on existing .venv (picamera2)"
  sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' \
    "$ROOT/.venv/pyvenv.cfg"
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
python -m pip install -U pip wheel

echo "==> pip requirements"
python -m pip install -r "$ROOT/requirements.txt"

# CPU torch wheel if pip torch is missing / broken on aarch64
python - <<'PY' || python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
import torch
print("torch", torch.__version__)
PY

ARCH="$(uname -m || true)"
if [[ "$ARCH" == "aarch64" || "$ARCH" == "armv7l" ]]; then
  echo "==> Pi extras (Waveshare Pan-Tilt HAT / Blinka / ServoKit)"
  python -m pip install adafruit-blinka adafruit-circuitpython-servokit \
    adafruit-circuitpython-pca9685 smbus2 || true
  echo "    Check HAT: sudo i2cdetect -y 1   (expect 0x40)"
  echo "==> camera check (Camera Module 3, cam 0)"
  python - <<'PY' || echo "    picamera2 not importable — GUI will fall back to rpicam-vid / no camera"
from picamera2 import Picamera2
cams = Picamera2.global_camera_info()
print("    cameras:", [c.get("Model") for c in cams] or "none detected")
PY
fi

echo "==> preload Silero VAD model"
python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import numpy as np
from audio.silero_vad import SileroVAD
v = SileroVAD()
r = v.classify(np.zeros(v.chunk_samples, dtype=np.float32))
print("Silero OK:", r["label"], f"p={r['probability']:.4f}")
PY

if [[ "$WITH_OVERLAY" -eq 1 ]]; then
  echo "==> installing ric-inmp441-4ch overlay"
  dtc -@ -I dts -O dtb -o /tmp/ric-inmp441-4ch.dtbo \
    "$ROOT/overlays/ric-inmp441-4ch-overlay.dts"
  sudo cp /tmp/ric-inmp441-4ch.dtbo /boot/firmware/overlays/ric-inmp441-4ch.dtbo
  if ! grep -q 'dtoverlay=ric-inmp441-4ch' /boot/firmware/config.txt 2>/dev/null; then
    echo "" | sudo tee -a /boot/firmware/config.txt >/dev/null
    echo "# RIC 4x INMP441" | sudo tee -a /boot/firmware/config.txt >/dev/null
    echo "dtparam=i2s=on" | sudo tee -a /boot/firmware/config.txt >/dev/null
    echo "dtoverlay=ric-inmp441-4ch" | sudo tee -a /boot/firmware/config.txt >/dev/null
  fi
  echo "Overlay installed. Reboot required: sudo reboot"
fi

mkdir -p "$ROOT/data"
echo ""
echo "Done."
echo "  GUI:      ./run_gui.sh --real-servos"
echo "  Headless: ./run_gui.sh --no-gui --real-servos"
echo "  Mics:     .venv/bin/python scripts/check_rpi_4ch_mics.py"
echo "  Servos:   .venv/bin/python scripts/servo_interactive.py  # home | limits | save"
