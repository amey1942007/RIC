"""
Full lecturer-tracking pipeline:

  4-ch I2S audio → Silero VAD gate → SRP-PHAT → Kalman 2D → ServoController

Mic geometry: vertical 6 cm square (see config.MIC_POSITIONS).
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from typing import Any, Optional

import numpy as np

import config as cfg
from audio.silero_vad import SileroVAD
from audio.srp_phat import SRPPhatLocalizer
from audio.kalman import KalmanFilter2D
from control.servo_controller import ServoController

log = logging.getLogger(__name__)


def find_audio_device(preferred: Optional[str] = None) -> Optional[int | str]:
    """
    Resolve PortAudio / ALSA device for the 4-ch RIC card.

    Prefers a device whose name contains ``preferred`` (default from config)
    with at least NUM_MICS input channels.
    """
    import sounddevice as sd

    needle = (preferred or cfg.AUDIO_DEVICE_NAME or "").lower()
    devices = sd.query_devices()
    # Prefer name match with enough inputs
    for i, d in enumerate(devices):
        if int(d.get("max_input_channels", 0)) < cfg.NUM_MICS:
            continue
        name = str(d.get("name", "")).lower()
        if needle and needle in name:
            log.info("Audio device [%s]: %s", i, d["name"])
            return i
    # Fallback: any device with >= 4 inputs
    for i, d in enumerate(devices):
        if int(d.get("max_input_channels", 0)) >= cfg.NUM_MICS:
            log.info("Audio device fallback [%s]: %s", i, d["name"])
            return i
    log.warning("No %d-channel input device found", cfg.NUM_MICS)
    return None


class LecturerTracker:
    """Threaded audio capture + localisation + pan/tilt control."""

    def __init__(
        self,
        audio_device: Optional[str | int] = None,
        simulate_servos: Optional[bool] = None,
        vad_threshold: float = cfg.VAD_THRESHOLD,
    ) -> None:
        if audio_device is None:
            audio_device = find_audio_device()
        self.audio_device = audio_device
        self.running = False
        self._audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self.vad = SileroVAD(
            sample_rate=cfg.SAMPLE_RATE,
            threshold=vad_threshold,
            normalize=cfg.VAD_NORMALIZE,
            target_peak=cfg.VAD_TARGET_PEAK,
            noise_floor=cfg.VAD_NOISE_FLOOR,
            max_gain=cfg.VAD_MAX_GAIN,
        )
        self.localizer = SRPPhatLocalizer(
            mic_positions=cfg.MIC_POSITIONS,
            sample_rate=cfg.SAMPLE_RATE,
            speed_of_sound=cfg.SPEED_OF_SOUND,
            n_fft=cfg.SRP_FRAME_SAMPLES,
            azimuth_range=(cfg.AZIMUTH_MIN_DEG, cfg.AZIMUTH_MAX_DEG),
            elevation_range=(cfg.ELEVATION_MIN_DEG, cfg.ELEVATION_MAX_DEG),
            azimuth_step=cfg.AZIMUTH_STEP_DEG,
            elevation_step=cfg.ELEVATION_STEP_DEG,
        )
        self.kalman = KalmanFilter2D(
            dt=cfg.KALMAN_DT,
            q_pos=cfg.KALMAN_Q_POS,
            q_vel=cfg.KALMAN_Q_VEL,
            r_meas=cfg.KALMAN_R,
            max_step_deg=getattr(cfg, "KALMAN_MAX_STEP_DEG", 4.0),
            max_vel_deg_s=getattr(cfg, "KALMAN_MAX_VEL_DEG_S", 40.0),
        )
        self.servos = ServoController(simulate=simulate_servos)

        self._ring: deque[np.ndarray] = deque(maxlen=cfg.SRP_FRAME_SAMPLES)
        self._last_result: dict[str, Any] = self._empty_result()
        self._speech_on = 0
        self._speech_off = 0
        self._tracking = False
        win = int(getattr(cfg, "SRP_MEDIAN_WINDOW", 5))
        self._az_hist: deque[float] = deque(maxlen=max(1, win))
        self._el_hist: deque[float] = deque(maxlen=max(1, win))
        self._last_good_az: Optional[float] = None
        self._last_good_el: Optional[float] = None

        # GUI-controllable switches
        self._mic_enabled = True     # False → audio discarded, no VAD / DOA / servo
        self._servo_manual = False   # True → pipeline never commands the HAT
        self._servo_lock = threading.Lock()

        # VAD-gating diagnostics (config.VAD_DIAGNOSTIC_LOGGING)
        self._diag = bool(getattr(cfg, "VAD_DIAGNOSTIC_LOGGING", False))
        self._tracking_start_t: Optional[float] = None
        self._diag_last_az: Optional[float] = None
        self._diag_last_el: Optional[float] = None
        self._diag_last_conf: float = 0.0

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "vad_label": "NON_SPEECH",
            "vad_probability": 0.0,
            "vad_gain": 1.0,
            "vad_peak": 0.0,
            "speech": False,
            "azimuth_deg": 0.0,
            "elevation_deg": 0.0,
            "confidence": 0.0,
            "localisation": None,
            "pan_deg": 90.0,
            "tilt_deg": 90.0,
            "servo_simulate": True,
            "mic_rms": [0.0] * cfg.NUM_MICS,
            "mic_peak": [0.0] * cfg.NUM_MICS,
            "mic_enabled": True,
            "servo_manual": False,
            "t": time.time(),
        }

    def snapshot(self) -> dict[str, Any]:
        """Thread-safe copy of the latest pipeline state (for GUI)."""
        with self._lock:
            s = dict(self._last_result)
        s["mic_enabled"] = self._mic_enabled
        s["servo_manual"] = self._servo_manual
        s["pan_deg"] = float(self.servos.pan_deg)
        s["tilt_deg"] = float(self.servos.tilt_deg)
        return s

    # ── GUI controls ───────────────────────────────────────────────────────
    @property
    def mic_enabled(self) -> bool:
        return self._mic_enabled

    def set_mic_enabled(self, enabled: bool) -> None:
        """Mute / un-mute the array. Muting also drops any active track."""
        enabled = bool(enabled)
        if enabled == self._mic_enabled:
            return
        self._mic_enabled = enabled
        if not enabled:
            self._reset_tracking_state()
            with self._lock:
                r = dict(self._last_result)
                r.update(
                    vad_label="MUTED",
                    vad_probability=0.0,
                    speech=False,
                    confidence=0.0,
                    localisation=None,
                    mic_rms=[0.0] * cfg.NUM_MICS,
                    mic_peak=[0.0] * cfg.NUM_MICS,
                    t=time.time(),
                )
                self._last_result = r
        log.info("Mic %s", "ON" if enabled else "OFF (muted)")

    @property
    def servo_manual(self) -> bool:
        return self._servo_manual

    def set_servo_manual(self, manual: bool) -> None:
        """Manual = GUI sliders own the HAT; Auto = SRP-PHAT/Kalman own it."""
        self._servo_manual = bool(manual)
        log.info("Servo mode: %s", "MANUAL" if manual else "AUTO")

    def manual_servo(self, pan_deg: float, tilt_deg: float) -> tuple[float, float]:
        """Direct HAT command (clamped to config limits). Only used in manual mode."""
        with self._servo_lock:
            self.servos.set_angles(float(pan_deg), float(tilt_deg), force=False)
            return self.servos.pan_deg, self.servos.tilt_deg

    def servo_center(self) -> tuple[float, float]:
        """Snap to the front pose (pan 90 / tilt 145) and re-seed the Kalman."""
        with self._servo_lock:
            self.servos.go_center()
        self._reset_tracking_state()
        return self.servos.pan_deg, self.servos.tilt_deg

    def _reset_tracking_state(self) -> None:
        self._speech_on = 0
        self._speech_off = 0
        self._tracking = False
        self._az_hist.clear()
        self._el_hist.clear()
        self._last_good_az = None
        self._last_good_el = None
        self.kalman.reset(0.0, 0.0)

    @property
    def last_result(self) -> dict[str, Any]:
        return self.snapshot()

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self.running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="audio-capture", daemon=True
        )
        self._proc_thread = threading.Thread(
            target=self._process_loop, name="tracker-proc", daemon=True
        )
        self._capture_thread.start()
        self._proc_thread.start()
        log.info(
            "LecturerTracker started (device=%s, servos_simulate=%s, mount=%s)",
            self.audio_device,
            self.servos.simulate,
            cfg.ARRAY_MOUNT,
        )

    def stop(self) -> None:
        self._stop.set()
        self.running = False
        if hasattr(self, "_capture_thread"):
            self._capture_thread.join(timeout=2.0)
        if hasattr(self, "_proc_thread"):
            self._proc_thread.join(timeout=2.0)
        log.info("LecturerTracker stopped")

    def run_forever(self) -> None:
        self.start()
        try:
            while not self._stop.is_set():
                time.sleep(0.25)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    # ── audio capture ──────────────────────────────────────────────────────
    def _capture_loop(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice is required for live capture"
            ) from exc

        block = cfg.VAD_CHUNK_SAMPLES

        def callback(indata, frames, time_info, status):  # noqa: ARG001
            if status:
                log.warning("sounddevice status: %s", status)
            try:
                self._audio_q.put_nowait(indata.copy())
            except queue.Full:
                pass

        try:
            with sd.InputStream(
                samplerate=cfg.SAMPLE_RATE,
                channels=cfg.CHANNELS,
                dtype="float32",
                blocksize=block,
                device=self.audio_device,
                callback=callback,
            ):
                while not self._stop.is_set():
                    time.sleep(0.05)
        except Exception:
            log.exception("Audio capture failed (device=%s)", self.audio_device)
            self._stop.set()
            self.running = False

    # ── processing ─────────────────────────────────────────────────────────
    def _process_loop(self) -> None:
        mono_carry = np.zeros(0, dtype=np.float32)
        mic_rms = np.zeros(cfg.NUM_MICS, dtype=np.float64)
        mic_peak = np.zeros(cfg.NUM_MICS, dtype=np.float64)

        while not self._stop.is_set():
            try:
                block = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if block.ndim == 1:
                block = block.reshape(-1, 1)

            if not self._mic_enabled:
                # muted: keep the stream alive but drop everything
                mono_carry = np.zeros(0, dtype=np.float32)
                mic_rms[:] = 0.0
                mic_peak[:] = 0.0
                self._ring.clear()
                with self._lock:
                    self._last_result["t"] = time.time()
                continue

            n_ch = min(block.shape[1], cfg.NUM_MICS)
            # Per-mic levels (EMA for stable meters)
            for ch in range(n_ch):
                x = block[:, ch].astype(np.float64)
                rms = float(np.sqrt(np.mean(x * x) + 1e-18))
                peak = float(np.max(np.abs(x)))
                mic_rms[ch] = 0.7 * mic_rms[ch] + 0.3 * rms
                mic_peak[ch] = 0.5 * mic_peak[ch] + 0.5 * peak

            mono = block[:, :n_ch].mean(axis=1).astype(np.float32)
            mono_carry = np.concatenate([mono_carry, mono])

            for i in range(block.shape[0]):
                sample = np.zeros(cfg.NUM_MICS, dtype=np.float32)
                sample[:n_ch] = block[i, :n_ch]
                self._ring.append(sample)

            while len(mono_carry) >= cfg.VAD_CHUNK_SAMPLES:
                chunk = mono_carry[: cfg.VAD_CHUNK_SAMPLES]
                mono_carry = mono_carry[cfg.VAD_CHUNK_SAMPLES :]
                self._process_chunk(chunk, mic_rms, mic_peak)

    def _process_chunk(
        self,
        mono_chunk: np.ndarray,
        mic_rms: np.ndarray,
        mic_peak: np.ndarray,
    ) -> None:
        classification = self.vad.classify(mono_chunk)
        raw_speech = classification["human_speech"]
        prob = classification["probability"]

        # Debounce / hangover — ignore brief noise bursts
        on_n = int(getattr(cfg, "VAD_SPEECH_ON_CHUNKS", 3))
        off_n = int(getattr(cfg, "VAD_SPEECH_OFF_CHUNKS", 12))
        was_tracking = self._tracking
        if raw_speech:
            self._speech_on += 1
            self._speech_off = 0
            if self._speech_on >= on_n:
                self._tracking = True
        else:
            self._speech_off += 1
            self._speech_on = 0
            if self._speech_off >= off_n:
                self._tracking = False

        if self._diag:
            now = time.time()
            if self._tracking and not was_tracking:
                self._tracking_start_t = now
            elif was_tracking and not self._tracking:
                active_s = (
                    now - self._tracking_start_t
                    if self._tracking_start_t is not None
                    else 0.0
                )
                log.info(
                    "VAD-DIAG tracking OFF after %.2fs active  last az=%s el=%s conf=%.2f  "
                    "(off_chunks=%d, p=%.2f)",
                    active_s,
                    "—" if self._diag_last_az is None else f"{self._diag_last_az:.1f}",
                    "—" if self._diag_last_el is None else f"{self._diag_last_el:.1f}",
                    self._diag_last_conf,
                    self._speech_off,
                    prob,
                )
                self._tracking_start_t = None
            thr = float(classification.get("threshold", self.vad.threshold))
            if abs(prob - thr) <= 0.1:
                log.debug(
                    "VAD-DIAG borderline p=%.3f thr=%.2f gain=%.2f peak=%.3f raw=%s "
                    "on=%d off=%d tracking=%s",
                    prob,
                    thr,
                    float(classification.get("gain", 1.0)),
                    float(classification.get("peak", 0.0)),
                    raw_speech,
                    self._speech_on,
                    self._speech_off,
                    self._tracking,
                )

        speech = self._tracking
        measurement = None
        loc = None
        conf = 0.0
        move_thr = float(
            getattr(cfg, "CONFIDENCE_MOVE_THRESHOLD", cfg.CONFIDENCE_THRESHOLD)
        )
        if speech and len(self._ring) >= cfg.SRP_FRAME_SAMPLES:
            multi = np.stack(list(self._ring), axis=0)  # (N, 4)
            loc = self.localizer.localize(multi)
            conf = float(loc["confidence"])
            if conf >= cfg.CONFIDENCE_THRESHOLD:
                az_m = float(loc["azimuth_deg"])
                el_m = float(loc["elevation_deg"])
                # Reject single-frame wild jumps (clap / noise spike)
                max_jump = float(getattr(cfg, "SRP_MAX_JUMP_DEG", 25.0))
                if self._last_good_az is not None:
                    daz = (az_m - self._last_good_az + 180.0) % 360.0 - 180.0
                    del_ = el_m - self._last_good_el
                    if abs(daz) > max_jump or abs(del_) > max_jump:
                        loc = {**loc, "rejected_outlier": True}
                        conf = 0.0
                    else:
                        self._az_hist.append(az_m)
                        self._el_hist.append(el_m)
                        self._last_good_az, self._last_good_el = az_m, el_m
                else:
                    self._az_hist.append(az_m)
                    self._el_hist.append(el_m)
                    self._last_good_az, self._last_good_el = az_m, el_m

                if self._az_hist and conf >= cfg.CONFIDENCE_THRESHOLD:
                    measurement = (
                        float(np.median(self._az_hist)),
                        float(np.median(self._el_hist)),
                    )
                    if self._diag:
                        self._diag_last_az, self._diag_last_el = measurement
                        self._diag_last_conf = conf

        az, el = self.kalman.step(
            measurement=measurement,
            confidence=None if measurement is None else conf,
        )

        # Only command HAT when still in speech hangover AND peak is strong
        servo_conf = conf if (speech and conf >= move_thr and measurement) else 0.0
        if self._servo_manual:
            pan, tilt = self.servos.pan_deg, self.servos.tilt_deg
        else:
            with self._servo_lock:
                pan, tilt = self.servos.set_direction(az, el, confidence=servo_conf)

        result = {
            "vad_label": classification["label"],
            "vad_probability": prob,
            "vad_gain": float(classification.get("gain", 1.0)),
            "vad_peak": float(classification.get("peak", 0.0)),
            "speech": speech,
            "azimuth_deg": az,
            "elevation_deg": el,
            "confidence": conf,
            "localisation": loc,
            "pan_deg": pan,
            "tilt_deg": tilt,
            "servo_simulate": self.servos.simulate,
            "mic_rms": mic_rms.tolist(),
            "mic_peak": mic_peak.tolist(),
            "mic_enabled": True,
            "servo_manual": self._servo_manual,
            "t": time.time(),
        }
        with self._lock:
            self._last_result = result

        if speech:
            log.info(
                "SPEECH p=%.2f  az=%.1f el=%.1f  conf=%s  pan=%.1f tilt=%.1f",
                prob,
                az,
                el,
                f"{conf:.2f}" if loc else "—",
                pan,
                tilt,
            )
