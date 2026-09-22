"""
Lecturer Tracking Camera — shared configuration.

Mic geometry: 4× INMP441 at corners of a 6 cm square on a *vertical* plane
(perpendicular to the base), facing into the room (+Y).
"""
from __future__ import annotations

from pathlib import Path

# ── Audio ──────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000          # Silero VAD requires 16 kHz
VAD_CHUNK_SAMPLES = 512       # Silero window @ 16 kHz
VAD_THRESHOLD = 0.45          # slightly softer than 0.5 for far talkers
VAD_NORMALIZE = True          # peak-normalize quiet / distant speech before VAD
VAD_TARGET_PEAK = 0.6         # AGC target peak in [-1, 1]
VAD_NOISE_FLOOR = 0.004       # skip AGC below this peak (true silence)
VAD_MAX_GAIN = 40.0           # cap boost so hiss is not over-amplified
NUM_MICS = 4
CHANNELS = 4                  # 2× stereo I2S pairs → 4 channels
# Preferred ALSA / PortAudio name fragment for the Pi overlay card
AUDIO_DEVICE_NAME = "ricinmp4414ch"

# ── Mic geometry (vertical 6 cm square, perpendicular to base) ─────────────
# Plane = X–Z (vertical). Array faces +Y (into the room).
# Looking at the front of the array:
#
#        +Z (up)
#          ↑
#   M0 ●─────────● M1
#      │  6 cm   │
#      │         │ 6 cm
#   M3 ●─────────● M2
#          → +X (right)
#
# Channel order matches I2S wiring:
#   SD1 GPIO20 → M0 (L/GND) + M1 (R/3V3)
#   SD2 GPIO22 → M2 (L/GND) + M3 (R/3V3)
MIC_SIDE_M = 0.06
_HALF = MIC_SIDE_M / 2.0
MIC_POSITIONS = {
    0: (-_HALF, 0.0, +_HALF),   # M0 top-left
    1: (+_HALF, 0.0, +_HALF),   # M1 top-right
    2: (+_HALF, 0.0, -_HALF),   # M2 bottom-right
    3: (-_HALF, 0.0, -_HALF),   # M3 bottom-left
}
MIC_LABELS = ("M0", "M1", "M2", "M3")
SPEED_OF_SOUND = 343.0        # m/s @ ~20 °C

# Mount style for SRP direction + servo mapping
#   "vertical_stand" — array upright on base / pan-tilt pod (el↑ = up)
#   "ceiling"        — looking down into room (el↑ = toward floor)
ARRAY_MOUNT = "vertical_stand"

# ── SRP-PHAT search grid ───────────────────────────────────────────────────
AZIMUTH_MIN_DEG = -90.0
AZIMUTH_MAX_DEG = 90.0
AZIMUTH_STEP_DEG = 2.0
ELEVATION_MIN_DEG = -30.0     # below array centre
ELEVATION_MAX_DEG = 60.0      # above array centre
ELEVATION_STEP_DEG = 2.0
SRP_FRAME_SAMPLES = 2048      # FFT frame for localisation
CONFIDENCE_THRESHOLD = 0.35   # hold pose below this

# ── Kalman (smooth over fast — lecture recording) ──────────────────────────
KALMAN_DT = VAD_CHUNK_SAMPLES / SAMPLE_RATE   # ~32 ms
KALMAN_Q_POS = 0.5            # process noise on angle (deg²)
KALMAN_Q_VEL = 2.0            # process noise on velocity
KALMAN_R = 25.0               # measurement noise (high → trust model more)

# ── Servo / Waveshare Pan-Tilt HAT (PCA9685) ───────────────────────────────
# https://www.waveshare.com/wiki/Pan-Tilt_HAT
#   S0 = tilt, S1 = pan, I2C addr 0x40
PCA9685_ADDRESS = 0x40
TILT_CHANNEL = 0              # Waveshare S0 (tilt servo B)
PAN_CHANNEL = 1               # Waveshare S1 (pan servo A)
PAN_MIN_DEG = 20.0
PAN_MAX_DEG = 160.0
TILT_MIN_DEG = 30.0
TILT_MAX_DEG = 150.0
DEAD_ZONE_DEG = 2.0
POSITION_FILE = Path("/var/lib/camera/last_position.json")
# Windows / desktop fallback for development:
POSITION_FILE_DEV = Path(__file__).resolve().parent / "data" / "last_position.json"

# ── Streaming (reference defaults) ─────────────────────────────────────────
STREAM_WIDTH = 1280
STREAM_HEIGHT = 720
STREAM_FPS = 25
STREAM_BITRATE = 2_000_000

# ── GUI ────────────────────────────────────────────────────────────────────
GUI_REFRESH_MS = 50
