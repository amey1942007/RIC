# Lecturer Tracking Camera — Audio Localisation Stack

Raspberry Pi 5 pipeline: **Silero VAD → SRP-PHAT → 2D Kalman → pan/tilt**.

Mic array: **4× INMP441 at corners of a vertical 6 cm square** (plane perpendicular to the base).

## Layout

```
RIC/
  install.sh / install.ps1       # one-shot deps + Silero model
  run_gui.sh                     # launch live GUI on Pi
  vad_srp_phat_pipeline.py       # entry point
  config.py
  gui/live_dashboard.py
  audio/  control/  pipeline/
  overlays/ric-inmp441-4ch-overlay.dts
  scripts/check_rpi_4ch_mics.py
  requirements.txt
```

## Mic geometry (vertical 6 cm square)

```
        +Z (up)
   M0 ●─────────● M1
      │  6 cm   │
   M3 ●─────────● M2
          → +X
```

| Mic | I2S |
|-----|-----|
| M0 / M1 | SD1 → GPIO20 (pin 38) |
| M2 / M3 | SD2 → GPIO22 (pin 15) |

SCK→GPIO18, WS→GPIO19. **Not** GPIO21 for SD2 (that pin is TX-only).

## Install

**Raspberry Pi**

```bash
cd ~/RIC
chmod +x install.sh run_gui.sh
./install.sh              # venv, pip, Silero weights
./install.sh --overlay    # also install 4-ch I2S overlay (then reboot)
```

**Windows (Vision-Pipeline venv)**

```powershell
cd D:\RIC
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

## Run

```bash
./run_gui.sh                          # live GUI
./run_gui.sh --no-gui                 # headless
./run_gui.sh --real-servos            # pan/tilt HAT
.venv/bin/python scripts/check_rpi_4ch_mics.py
```

## Assembly reminder

Do **not** seat servos in the bracket until you have commanded them to home (0°)
via `ServoController.go_home()`.
