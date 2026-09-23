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
# Slightly stricter gate + hangover reduces clap/noise yanking the HAT
VAD_THRESHOLD = 0.58
VAD_NORMALIZE = True
VAD_TARGET_PEAK = 0.55
VAD_NOISE_FLOOR = 0.008       # ignore quieter than this for AGC
VAD_MAX_GAIN = 20.0           # limited far-field boost (less noise amp)
VAD_SPEECH_ON_CHUNKS = 4      # ~128 ms sustained speech before track
VAD_SPEECH_OFF_CHUNKS = 15    # ~480 ms hangover after speech ends
NUM_MICS = 4
CHANNELS = 4
AUDIO_DEVICE_NAME = "ricinmp4414ch"

# ── Mic geometry (vertical 6 cm square) ────────────────────────────────────
# Mount array centre near speaker *mouth height* (~1.4–1.6 m standing),
# not floor / not ceiling — balances elevation for a vertical plane.
MIC_SIDE_M = 0.06
_HALF = MIC_SIDE_M / 2.0
MIC_POSITIONS = {
    0: (-_HALF, 0.0, +_HALF),   # M0 top-left
    1: (+_HALF, 0.0, +_HALF),   # M1 top-right
    2: (+_HALF, 0.0, -_HALF),   # M2 bottom-right
    3: (-_HALF, 0.0, -_HALF),   # M3 bottom-left
}
MIC_LABELS = ("M0", "M1", "M2", "M3")
SPEED_OF_SOUND = 343.0

ARRAY_MOUNT = "vertical_stand"

# Typical lecturer distance used by /goa skill notes (metres, far-field SRP)
NOMINAL_SPEAKER_DISTANCE_M = 2.5

# ── SRP-PHAT ───────────────────────────────────────────────────────────────
AZIMUTH_MIN_DEG = -90.0
AZIMUTH_MAX_DEG = 90.0
AZIMUTH_STEP_DEG = 2.0
ELEVATION_MIN_DEG = -30.0
ELEVATION_MAX_DEG = 60.0
ELEVATION_STEP_DEG = 2.0
SRP_FRAME_SAMPLES = 2048
CONFIDENCE_THRESHOLD = 0.58   # ignore weak / noisy peaks
CONFIDENCE_MOVE_THRESHOLD = 0.65  # only move HAT above this
SRP_MEDIAN_WINDOW = 5         # median of last N peaks before Kalman
SRP_MAX_JUMP_DEG = 25.0       # reject single-frame outliers vs last good

# ── Kalman (heavy smoothing — reduce sudden jumps) ─────────────────────────
KALMAN_DT = VAD_CHUNK_SAMPLES / SAMPLE_RATE
KALMAN_Q_POS = 0.05
KALMAN_Q_VEL = 0.2
KALMAN_R = 100.0
KALMAN_MAX_STEP_DEG = 3.0
KALMAN_MAX_VEL_DEG_S = 30.0

# ── Servo / Waveshare Pan-Tilt HAT ─────────────────────────────────────────
# Probe limits with:  .venv/bin/python scripts/servo_interactive.py
PCA9685_ADDRESS = 0x40
TILT_CHANNEL = 0              # S0
PAN_CHANNEL = 1               # S1
# Pan: 0 = left, 90 = front (centered on array), 180 = right
PAN_MIN_DEG = 0.0
PAN_MAX_DEG = 180.0
# Tilt: 80 = highest (look up), 145 = eye-level front, 180 = lowest (look down)
TILT_MIN_DEG = 80.0           # top / up
TILT_MAX_DEG = 180.0          # bottom / down
TILT_FRONT_DEG = 145.0        # camera faces you when array is at eye level
TILT_INVERTED = True          # up → toward TILT_MIN (80)
DEAD_ZONE_DEG = 4.0
SERVO_MAX_SPEED_DEG_S = 35.0
SERVO_EMA_ALPHA = 0.18
POSITION_FILE = Path("/var/lib/camera/last_position.json")
POSITION_FILE_DEV = Path(__file__).resolve().parent / "data" / "last_position.json"

# ── Streaming ──────────────────────────────────────────────────────────────
STREAM_WIDTH = 1280
STREAM_HEIGHT = 720
STREAM_FPS = 25
STREAM_BITRATE = 2_000_000

GUI_REFRESH_MS = 50
