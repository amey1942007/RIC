# Lecturer Tracking Camera — Audio Localisation Stack

Raspberry Pi 5: **Silero VAD → SRP-PHAT → Kalman → Waveshare Pan-Tilt HAT**.

Mic array: **4× INMP441**, vertical 6 cm square.  
HAT: [Waveshare Pan-Tilt HAT](https://www.waveshare.com/wiki/Pan-Tilt_HAT) (PCA9685 @ `0x40`, tilt=`S0`, pan=`S1`).

## Layout

```
RIC/
  install.sh / install.ps1
  run_gui.sh
  vad_srp_phat_pipeline.py
  config.py
  gui/live_dashboard.py
  audio/  control/  pipeline/
  overlays/ric-inmp441-4ch-overlay.dts
  scripts/check_rpi_4ch_mics.py
  scripts/home_pantilt.py      # Waveshare 0° home BEFORE assembly
  scripts/test_pantilt.py
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

HAT sits on the 40-pin header. Servos:

| Servo | HAT pad | Channel |
|-------|---------|---------|
| Tilt (B) | **S0** | 0 |
| Pan (A) | **S1** | 1 |

Wire colours: Brown→GND, Red→5V, Orange→S0/S1.

**Do not assemble servos into the bracket until home (0°) is done** — see Waveshare wiki.

## Install (on the Pi)

```bash
cd ~/RIC
chmod +x install.sh run_gui.sh
./install.sh
./install.sh --overlay
sudo reboot
```

After reboot, confirm I2C + mics:

```bash
sudo i2cdetect -y 1          # expect "40" (PCA9685)
arecord -l                   # ric-inmp441-4ch
.venv/bin/python scripts/check_rpi_4ch_mics.py
```

### HAT first-time (Waveshare order)

```bash
# 1) Servos plugged in but NOT mounted in bracket (free to spin)
.venv/bin/python scripts/home_pantilt.py

# 2) Power off → assemble bracket (don't twist tilt horn) → power on
.venv/bin/python scripts/test_pantilt.py

# 3) Live tracking → HAT
./run_gui.sh --real-servos
```

The pipeline sends SRP azimuth/elevation to the HAT every speech frame (GUI shows pan/tilt).

## Run

```bash
./run_gui.sh --real-servos     # audio + GUI + HAT
./run_gui.sh --simulate-servos # audio + GUI, no PWM
./run_gui.sh --no-gui --real-servos
```
