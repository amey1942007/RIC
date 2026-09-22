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
        )
        self.servos = ServoController(simulate=simulate_servos)

        self._ring: deque[np.ndarray] = deque(maxlen=cfg.SRP_FRAME_SAMPLES)
        self._last_result: dict[str, Any] = self._empty_result()

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
            "t": time.time(),
        }

    def snapshot(self) -> dict[str, Any]:
        """Thread-safe copy of the latest pipeline state (for GUI)."""
        with self._lock:
            return dict(self._last_result)

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
        speech = classification["human_speech"]
        prob = classification["probability"]

        measurement = None
        loc = None
        conf = 0.0
        if speech and len(self._ring) >= cfg.SRP_FRAME_SAMPLES:
            multi = np.stack(list(self._ring), axis=0)  # (N, 4)
            loc = self.localizer.localize(multi)
            conf = float(loc["confidence"])
            if conf >= cfg.CONFIDENCE_THRESHOLD:
                measurement = (loc["azimuth_deg"], loc["elevation_deg"])

        az, el = self.kalman.step(
            measurement=measurement,
            confidence=None if loc is None else loc.get("confidence"),
        )

        pan, tilt = self.servos.set_direction(
            az,
            el,
            confidence=1.0 if measurement is not None else 0.0,
        )

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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    tracker = LecturerTracker()
    tracker.run_forever()


if __name__ == "__main__":
    main()
