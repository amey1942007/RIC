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
# Speech gate. VAD_ENABLED=False bypasses Silero entirely and gates on plain signal
# level instead: a chunk is "active" when its peak (after high-pass) exceeds
# max(ENERGY_GATE_PEAK, ENERGY_GATE_RATIO × tracked noise floor).
# Measured on the Pi: quiet room 0.001–0.003, talker at ~1 m ≈ 0.02.
VAD_ENABLED = False
ENERGY_GATE_PEAK = 0.004      # absolute minimum peak to count as sound
ENERGY_GATE_RATIO = 3.0       # ... and at least this × the running noise floor
# Slightly stricter gate + hangover reduces clap/noise yanking the HAT
VAD_THRESHOLD = 0.58
# Measured on the Pi (Sep 23): raw INMP441 speech peaks ≈0.02 → Silero never fires
# without level normalisation (0 usable runs vs 7 with AGC). Keep True.
VAD_NORMALIZE = True
VAD_TARGET_PEAK = 0.55
VAD_NOISE_FLOOR = 0.008       # used only when VAD_NORMALIZE is True
VAD_MAX_GAIN = 20.0           # used only when VAD_NORMALIZE is True
VAD_SPEECH_ON_CHUNKS = 4      # ~128 ms sustained speech before track
VAD_SPEECH_OFF_CHUNKS = 15    # ~480 ms hangover after speech ends
NUM_MICS = 4
CHANNELS = 4
AUDIO_DEVICE_NAME = "ricinmp4414ch"
# Captured I2S channel index for each logical mic M0..M3 (positions below stay fixed).
# I2S slots: SD1 → ch0 (L) / ch1 (R), SD2 → ch2 (L) / ch3 (R). Top pair is wired
# R = top-left (M0) / L = top-right (M1); bottom pair L = bottom-right (M2) / R = bottom-left (M3).
# Set by `scripts/check_rpi_4ch_mics.py --side left` (talker on the left → negative azimuth).
# Its mirror (0, 1, 3, 2) is equally SRP-consistent but swaps left/right.
MIC_CHANNEL_ORDER = (1, 0, 2, 3)
# Measured: >98 % of raw energy is < 150 Hz (rumble / servo vibration) — no direction
# info at those wavelengths for a 6 cm array and it swamps the VAD. High-pass everything.
AUDIO_HIGHPASS_HZ = 120.0     # 4th-order Butterworth, applied to all 4 channels; 0 = off
SRP_BAND_HZ = (300.0, 4000.0) # GCC-PHAT bins outside this band are ignored (speech band)

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
BEARING_HOLD_DEG = 4.0        # ignore SRP updates smaller than this (stops spin on a static talker)
SRP_ADAPTIVE_SEARCH = True    # coarse-to-fine search (False = legacy full dense grid)
SRP_COARSE_STEP_MULTIPLIER = 3  # pass-1 step = this × AZIMUTH/ELEVATION_STEP_DEG
SRP_VOLUME_NEIGHBORS = 1      # ±N fine cells averaged per candidate ("modified SRP-PHAT"); 0 = off

# ── GCC-PHAT diffuseness-mask fallback (audio/gcc_phat_diffuse.py, not wired in) ──
GCC_DIFFUSE_THRESHOLD = 0.6   # keep T-F bins with diffuseness < this (0..1); needs room tuning
GCC_DIFFUSE_SUBFRAME = 512    # STFT sub-frame length used for CDR smoothing inside one SRP frame

# ── Diagnostics ────────────────────────────────────────────────────────────
VAD_DIAGNOSTIC_LOGGING = False  # opt-in VAD-gating validation logs (tracking drop / borderline chunks)

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
# Pan: 0 = left, 90 = front (centered on array), 150 = right-most
# PAN_MAX is capped at 150 (not 180): the camera ribbon cable binds past ~150.
# Mapping is piecewise around PAN_FRONT_DEG so "front" stays at exactly 90.
PAN_MIN_DEG = 0.0
PAN_MAX_DEG = 150.0
PAN_FRONT_DEG = 90.0          # az = 0 → this pan angle
# Which way the HAT turns for a positive azimuth. +1: az>0 → toward PAN_MAX (150);
# -1: az>0 → toward PAN_MIN (0). Set -1 on Sep 23: with the calibrated mic order the
# camera turned AWAY from a talker on the right, i.e. the pan servo runs opposite to
# the azimuth sign convention (mic order was verified separately with --side left).
PAN_DIRECTION = -1
# Tilt: 80 = highest (look up), 145 = eye-level front, 180 = lowest (look down)
TILT_MIN_DEG = 80.0           # top / up
TILT_MAX_DEG = 180.0          # bottom / down
TILT_FRONT_DEG = 145.0        # camera faces you when array is at eye level
TILT_INVERTED = True          # up → toward TILT_MIN (80)
TRACK_TILT = False            # False = pan-only: tilt held at TILT_FRONT_DEG (elevation is the weak axis)
DEAD_ZONE_DEG = 3.0
SERVO_MAX_SPEED_DEG_S = 35.0
SERVO_EMA_ALPHA = 0.18
POSITION_FILE = Path("/var/lib/camera/last_position.json")
POSITION_FILE_DEV = Path(__file__).resolve().parent / "data" / "last_position.json"

# ── Streaming ──────────────────────────────────────────────────────────────
STREAM_WIDTH = 1280
STREAM_HEIGHT = 720
STREAM_FPS = 25
STREAM_BITRATE = 2_000_000

# ── Camera (Raspberry Pi Camera Module 3 on the pan/tilt head) ─────────────
CAMERA_ENABLED = True         # show live feed in the GUI (auto-off if no camera found)
CAMERA_INDEX = 0              # Picamera2 camera number / V4L2 index (cam 0 = CSI port 0)
CAMERA_PREVIEW_WIDTH = 640    # GUI preview size (keeps Pi CPU free for SRP-PHAT)
CAMERA_PREVIEW_HEIGHT = 360
CAMERA_FPS = 20               # preview frame rate
CAMERA_HFLIP = True           # camera is mounted upside-down on the pan-tilt head:
CAMERA_VFLIP = True           # hflip + vflip = 180° rotation (keeps left/right true)

GUI_REFRESH_MS = 50
