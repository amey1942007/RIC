# Lecturer Tracking Camera — Audio Localisation Stack

Raspberry Pi 5: **Silero VAD → SRP-PHAT → Kalman → Waveshare Pan-Tilt HAT**.

## How the pieces fit (not duplicates)

```
./run_gui.sh                          # thin shell: venv + DISPLAY
        │
        ▼
vad_srp_phat_pipeline.py              # ONLY CLI entry (--no-gui / --real-servos)
        │
        ├─ GUI path  → gui/live_dashboard.py
        │                     │
        └─ headless  ─────────┴──► pipeline/lecturer_tracker.py
                                   (real engine: capture → VAD → SRP → Kalman → HAT)
```

| File | Role |
|------|------|
| `run_gui.sh` | Convenience launcher on Pi |
| `vad_srp_phat_pipeline.py` | CLI / flags |
| `pipeline/lecturer_tracker.py` | Core tracking logic (library, not a second app) |
| `gui/live_dashboard.py` | Tk meters / DOA / pan-tilt display |

## Layout

```
RIC/
  install.sh / install.ps1 / run_gui.sh
  vad_srp_phat_pipeline.py
  config.py
  gui/live_dashboard.py
  audio/  control/  pipeline/
  overlays/ric-inmp441-4ch-overlay.dts
  scripts/check_rpi_4ch_mics.py
  scripts/servo_interactive.py   # home + limits + move (one tool)
  requirements.txt
```

## Mic wiring (Pi 5)

| Signal | Pin | GPIO |
|--------|-----|------|
| SCK | 12 | 18 |
| WS | 35 | 19 |
| SD1 (M0+M1) | 38 | 20 |
| SD2 (M2+M3) | **15** | **22** |
| 3V3 / GND | 1/17 + GND | — |

## Waveshare Pan-Tilt HAT

tilt=`S0`, pan=`S1`, I2C `0x40`. Brown=GND, Red=5V, Orange=PWM.

**Do not assemble servos until home (0°)** — use interactive tool below.

## Install / run

```bash
cd ~/RIC
chmod +x install.sh run_gui.sh
./install.sh && ./install.sh --overlay && sudo reboot

sudo i2cdetect -y 1
.venv/bin/python scripts/check_rpi_4ch_mics.py

# Servos free to spin → home, then assemble, then probe limits:
.venv/bin/python scripts/servo_interactive.py
# home | pan 90 | setpanmin 25 | settiltmax 140 | save

./run_gui.sh --real-servos
./run_gui.sh --no-gui --real-servos
```

## Mic height

Array **centre near mouth height** (~1.4–1.6 m standing). Not floor / not ceiling.
