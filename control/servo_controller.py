"""
Pan-tilt servo controller for Waveshare Pan-Tilt HAT (PCA9685 @ 0x40).

Wiring (from Waveshare wiki):
  S0 = tilt servo   → config.TILT_CHANNEL (0)
  S1 = pan servo    → config.PAN_CHANNEL  (1)
  Brown=GND, Red=5V, Orange=PWM

IMPORTANT (Waveshare): do NOT assemble servos into the bracket until both
have been commanded to 0° (see scripts/home_pantilt.py).

Docs: https://www.waveshare.com/wiki/Pan-Tilt_HAT
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional, Tuple

import config as cfg

log = logging.getLogger(__name__)


class ServoController:
    """Map localisation angles → pan/tilt PWM on the Waveshare HAT."""

    def __init__(
        self,
        position_file: Optional[Path] = None,
        simulate: Optional[bool] = None,
    ) -> None:
        self.position_file = Path(
            position_file
            or (
                cfg.POSITION_FILE
                if cfg.POSITION_FILE.parent.exists()
                else cfg.POSITION_FILE_DEV
            )
        )
        self.position_file.parent.mkdir(parents=True, exist_ok=True)

        if simulate is None:
            simulate = False  # try real HAT first
        self.simulate = bool(simulate)
        self.kit = None
        self.i2c_address = int(getattr(cfg, "PCA9685_ADDRESS", 0x40))

        if not self.simulate:
            try:
                self.kit = self._open_hat()
            except Exception as exc:
                log.warning("Waveshare Pan-Tilt HAT unavailable (%s) — simulate", exc)
                self.simulate = True
                self.kit = None

        # On first boot after home, load last pose; otherwise mid look-ahead
        self.pan_deg, self.tilt_deg = self._load_position()
        if not self.simulate:
            self._apply(self.pan_deg, self.tilt_deg, force=True)
        log.info(
            "ServoController ready (simulate=%s, addr=0x%02X) pan=%.1f tilt=%.1f",
            self.simulate,
            self.i2c_address,
            self.pan_deg,
            self.tilt_deg,
        )

    def _open_hat(self):
        """Open PCA9685 via Adafruit ServoKit (Waveshare default addr 0x40)."""
        from adafruit_servokit import ServoKit  # type: ignore

        kit = ServoKit(channels=16, address=self.i2c_address)
        # Standard analogue servos @ 50 Hz
        try:
            kit.frequency = 50
        except Exception:
            pass
        for ch in (cfg.PAN_CHANNEL, cfg.TILT_CHANNEL):
            kit.servo[ch].actuation_range = 180
            # MG90S / SG90 typical pulse window
            try:
                kit.servo[ch].set_pulse_width_range(500, 2500)
            except Exception:
                pass
        log.info(
            "Waveshare HAT OK — tilt=S%d pan=S%d @ 0x%02X",
            cfg.TILT_CHANNEL,
            cfg.PAN_CHANNEL,
            self.i2c_address,
        )
        return kit

    @staticmethod
    def hat_present(bus: int = 1, address: int = 0x40) -> bool:
        """True if PCA9685 ACK on I2C (Waveshare default 0x40)."""
        try:
            from smbus2 import SMBus  # type: ignore

            with SMBus(bus) as sm:
                sm.read_byte(address)
            return True
        except Exception:
            pass
        try:
            import subprocess

            out = subprocess.check_output(
                ["i2cdetect", "-y", str(bus)], text=True, stderr=subprocess.DEVNULL
            )
            token = f"{address:02x}"
            return token in out.lower()
        except Exception:
            return False

    def _load_position(self) -> Tuple[float, float]:
        if self.position_file.exists():
            try:
                data = json.loads(self.position_file.read_text(encoding="utf-8"))
                return float(data["pan_deg"]), float(data["tilt_deg"])
            except Exception as exc:
                log.warning("Could not load position file: %s", exc)
        mid_pan = 0.5 * (cfg.PAN_MIN_DEG + cfg.PAN_MAX_DEG)
        mid_tilt = 0.5 * (cfg.TILT_MIN_DEG + cfg.TILT_MAX_DEG)
        return mid_pan, mid_tilt

    def _save_position(self) -> None:
        payload = {"pan_deg": self.pan_deg, "tilt_deg": self.tilt_deg}
        self.position_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def map_angles(azimuth_deg: float, elevation_deg: float) -> Tuple[float, float]:
        """
        SRP azimuth/elevation → Waveshare pan/tilt degrees (0..180).

        Azimuth AZ_MIN..AZ_MAX → pan PAN_MIN..PAN_MAX (S1)
        Elevation depends on ARRAY_MOUNT for tilt (S0)
        """
        az = float(np_clip(azimuth_deg, cfg.AZIMUTH_MIN_DEG, cfg.AZIMUTH_MAX_DEG))
        el = float(np_clip(elevation_deg, cfg.ELEVATION_MIN_DEG, cfg.ELEVATION_MAX_DEG))

        az_norm = (az - cfg.AZIMUTH_MIN_DEG) / (
            cfg.AZIMUTH_MAX_DEG - cfg.AZIMUTH_MIN_DEG
        )
        pan = cfg.PAN_MIN_DEG + az_norm * (cfg.PAN_MAX_DEG - cfg.PAN_MIN_DEG)

        el_span = cfg.ELEVATION_MAX_DEG - cfg.ELEVATION_MIN_DEG
        el_norm = (el - cfg.ELEVATION_MIN_DEG) / el_span if el_span else 0.0
        if getattr(cfg, "ARRAY_MOUNT", "vertical_stand") == "ceiling":
            tilt = cfg.TILT_MAX_DEG - el_norm * (cfg.TILT_MAX_DEG - cfg.TILT_MIN_DEG)
        else:
            tilt = cfg.TILT_MIN_DEG + el_norm * (cfg.TILT_MAX_DEG - cfg.TILT_MIN_DEG)
        return pan, tilt

    def go_home(self) -> None:
        """
        Drive both servos to 0° — Waveshare required step BEFORE assembling
        the bracket (prevents binding / burnout).
        """
        self._apply(0.0, 0.0, force=True)
        log.warning(
            "Servos at 0° (Waveshare home). Power off, then assemble bracket "
            "without rotating the tilt horn."
        )

    def go_center(self) -> None:
        """Look straight ahead (mid pan / mid tilt)."""
        pan = 0.5 * (cfg.PAN_MIN_DEG + cfg.PAN_MAX_DEG)
        tilt = 0.5 * (cfg.TILT_MIN_DEG + cfg.TILT_MAX_DEG)
        self._apply(pan, tilt, force=True)

    def set_angles(self, pan_deg: float, tilt_deg: float, force: bool = False) -> None:
        """Direct pan/tilt command in servo degrees (for tests)."""
        self._apply(pan_deg, tilt_deg, force=force)

    def set_direction(
        self,
        azimuth_deg: float,
        elevation_deg: float,
        confidence: float = 1.0,
    ) -> Tuple[float, float]:
        """Pipeline entry: SRP angles → HAT PWM (gated by confidence)."""
        if confidence < cfg.CONFIDENCE_THRESHOLD:
            return self.pan_deg, self.tilt_deg

        target_pan, target_tilt = self.map_angles(azimuth_deg, elevation_deg)
        d_pan = abs(target_pan - self.pan_deg)
        d_tilt = abs(target_tilt - self.tilt_deg)
        if d_pan < cfg.DEAD_ZONE_DEG and d_tilt < cfg.DEAD_ZONE_DEG:
            return self.pan_deg, self.tilt_deg

        self._apply(target_pan, target_tilt, force=False)
        return self.pan_deg, self.tilt_deg

    def _apply(self, pan: float, tilt: float, force: bool) -> None:
        pan = float(np_clip(pan, 0.0, 180.0))
        tilt = float(np_clip(tilt, 0.0, 180.0))
        # Soft operating limits during tracking (home may still use 0°)
        if not force:
            pan = float(np_clip(pan, cfg.PAN_MIN_DEG, cfg.PAN_MAX_DEG))
            tilt = float(np_clip(tilt, cfg.TILT_MIN_DEG, cfg.TILT_MAX_DEG))

        if not force and math.isclose(pan, self.pan_deg, abs_tol=0.05) and math.isclose(
            tilt, self.tilt_deg, abs_tol=0.05
        ):
            return

        self.pan_deg, self.tilt_deg = pan, tilt
        if self.simulate:
            log.info("SIM HAT pan=%.1f tilt=%.1f", pan, tilt)
        else:
            assert self.kit is not None
            self.kit.servo[cfg.PAN_CHANNEL].angle = pan
            self.kit.servo[cfg.TILT_CHANNEL].angle = tilt
        self._save_position()


def np_clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
