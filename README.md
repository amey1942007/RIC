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
| `gui/live_dashboard.py` | Tk control room: camera feed, meters, radar, pan-tilt |
| `camera/pi_camera.py` | Camera Module 3 grabber (picamera2 → rpicam-vid → OpenCV) |

## GUI controls

| Control | Key | What it does |
|---------|-----|--------------|
| **MIC ON/OFF** | `M` | Mute the array — VAD/DOA stop, HAT holds position |
| **SERVO AUTO/MANUAL** | `A` | MANUAL: sliders, `←→↑↓` nudge (2°), click a bearing on the radar |
| **CENTRE** | `C` | Front pose: pan `PAN_FRONT_DEG` (90) / tilt `TILT_FRONT_DEG` (145) |
| **CAM** | — | Start / stop the Camera Module 3 preview (cam index `CAMERA_INDEX` = 0) |
| quit | `Q` | |

The camera HUD shows the commanded pan/tilt, a green marker where the
speaker is relative to the camera's aim, and a red TRACK dot while speech is
being followed.

## Layout

```
RIC/
  install.sh / install.ps1 / run_gui.sh
  vad_srp_phat_pipeline.py
  config.py
  gui/live_dashboard.py
  camera/pi_camera.py
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

Captured channel → mic is set by `MIC_CHANNEL_ORDER` in `config.py` (currently
`(1, 0, 2, 3)`, calibrated with a talker on the array's left). If you rewire,
re-run `scripts/check_rpi_4ch_mics.py --side left` from ~1 m on the left — it
prints the order to set (the mirrored order is equally consistent but swaps
left/right, so a level check alone cannot resolve it).
All channels are high-passed at `AUDIO_HIGHPASS_HZ` (120 Hz) and GCC-PHAT uses
only `SRP_BAND_HZ` (300–4000 Hz); the raw Pi feed is >98 % sub-150 Hz rumble.

## Waveshare Pan-Tilt HAT

tilt=`S0`, pan=`S1`, I2C `0x40`. Brown=GND, Red=5V, Orange=PWM.

**Do not assemble servos until home (0°)** — use interactive tool below.

Servo window (config.py):

| Axis | Min | Front | Max | Note |
|------|-----|-------|-----|------|
| Pan | 0 (left) | **90** | **150** (right) | capped at 150 — camera ribbon binds past it; az 0→+90 is compressed into 90→150 |
| Tilt | 80 (up) | **145** | 180 (down) | 145 = camera level with the array |

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
