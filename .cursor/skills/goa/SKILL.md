---
name: goa
description: >-
  Create a complete robust pipeline with working SRP-PHAT audio location of
  speaker with the RIC 4× INMP441 + Waveshare Pan-Tilt HAT setup (vertical
  6 cm mic square), smooth servo motion that resists sudden noise, and
  Silero VAD tuned for human speech. Use when the user invokes /goa or asks
  to rebuild/harden the lecturer-tracking audio localisation pipeline.
disable-model-invocation: true
---

# /goa — RIC robust audio localisation pipeline

## Goal (verbatim)

Create a complete robust pipeline with working srp phat audio location of speaker with our current setup with the distance given and also make the servo movement smoother such that no sudden noise can interfare and also improve the silero vad paramters for best human audio detection.

## Hardware facts (do not invent)

- Pi 5, 4× INMP441, vertical 6 cm square (X–Z, faces +Y)
- SD1→GPIO20 (M0 L, M1 R), SD2→GPIO22 (M2 L, M3 R); SCK→18, WS→19
- Waveshare Pan-Tilt HAT: PCA9685 `@0x40`, tilt=`S0`/ch0, pan=`S1`/ch1
- Nominal speaker distance: `config.NOMINAL_SPEAKER_DISTANCE_M` (far-field SRP)
- Mount **array centre near mouth height (~1.4–1.6 m standing)**, not floor/ceiling

## Required behaviour when applying /goa

1. Keep entry points: `install.sh`, `run_gui.sh`, `vad_srp_phat_pipeline.py`
2. Preserve channel order M0..M3 matching I2S wiring and `config.MIC_POSITIONS`
3. Enforce smooth motion:
   - Kalman: high `KALMAN_R`, low `KALMAN_Q_*`, `KALMAN_MAX_STEP_DEG`, velocity clamp
   - Servo: `SERVO_EMA_ALPHA`, `SERVO_MAX_SPEED_DEG_S`, `DEAD_ZONE_DEG`, `CONFIDENCE_MOVE_THRESHOLD`
   - VAD hangover: `VAD_SPEECH_ON_CHUNKS` / `VAD_SPEECH_OFF_CHUNKS`
4. Clamp pan/tilt to `PAN_*` / `TILT_*` (set via `scripts/servo_interactive.py` then `save`)
5. Never move HAT on weak SRP peaks or one-shot noise (confidence + hangover gates)
6. Prefer editing `config.py` constants over scattering magic numbers

## Probe servo limits

```bash
.venv/bin/python scripts/servo_interactive.py
# pan / tilt / +pan / setpanmin / settiltmax / save
```

## Verify mics / run

```bash
.venv/bin/python scripts/check_rpi_4ch_mics.py   # 4/4 live
./run_gui.sh --real-servos
```

## VAD defaults (human speech)

Prefer stricter than 0.5 for rooms with claps/music: `VAD_THRESHOLD≈0.55`, limited AGC (`VAD_MAX_GAIN≈25`, higher `VAD_NOISE_FLOOR`), on/off chunk debounce.

## Out of scope

- Do not reintroduce deleted `tests/` live VAD harness unless user asks
- Do not put SD2 on GPIO21 (TX-only on Pi 5)
