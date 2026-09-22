"""
Pan-tilt servo controller (Waveshare HAT / PCA9685 via Adafruit ServoKit).

On non-Pi hosts the controller runs in *simulation* mode (logs angles only).
Position persistence: last commanded pose saved to disk (no encoders).
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
    """Map localisation angles → pan/tilt PWM with dead-zone + persistence."""

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
            simulate = False  # try real HAT first; fall back below
        self.simulate = bool(simulate)
        self.kit = None

        if not self.simulate:
            try:
                from adafruit_servokit import ServoKit  # type: ignore

                self.kit = ServoKit(channels=16)
                self.kit.servo[cfg.PAN_CHANNEL].actuation_range = 180
                self.kit.servo[cfg.TILT_CHANNEL].actuation_range = 180
            except Exception as exc:
                log.warning("Pan-tilt HAT unavailable (%s) — simulate mode", exc)
                self.simulate = True
                self.kit = None

        self.pan_deg, self.tilt_deg = self._load_position()
        self._apply(self.pan_deg, self.tilt_deg, force=True)
        log.info(
            "ServoController ready (simulate=%s) at pan=%.1f tilt=%.1f",
            self.simulate,
            self.pan_deg,
            self.tilt_deg,
        )

    @staticmethod
    def _pca9685_available() -> bool:
        try:
            from adafruit_servokit import ServoKit  # noqa: F401

            return True
        except Exception:
            return False

    def _load_position(self) -> Tuple[float, float]:
        if self.position_file.exists():
            try:
                data = json.loads(self.position_file.read_text(encoding="utf-8"))
                return float(data["pan_deg"]), float(data["tilt_deg"])
            except Exception as exc:
                log.warning("Could not load position file: %s", exc)
        # Home / mid
        mid_pan = 0.5 * (cfg.PAN_MIN_DEG + cfg.PAN_MAX_DEG)
        mid_tilt = 0.5 * (cfg.TILT_MIN_DEG + cfg.TILT_MAX_DEG)
        return mid_pan, mid_tilt

    def _save_position(self) -> None:
        payload = {"pan_deg": self.pan_deg, "tilt_deg": self.tilt_deg}
        self.position_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def map_angles(azimuth_deg: float, elevation_deg: float) -> Tuple[float, float]:
        """
        Azimuth AZ_MIN..AZ_MAX → pan PAN_MIN..PAN_MAX

        Elevation mapping depends on ARRAY_MOUNT:
          vertical_stand — higher elevation tilts camera up (el_min→TILT_MIN)
          ceiling        — inverted (el_min→TILT_MAX) for downward look
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
        """Drive both servos to 0° PWM reference BEFORE physical assembly."""
        self._apply(0.0, 0.0, force=True)
        log.warning(
            "Servos commanded to 0°. Assemble bracket only after confirming home."
        )

    def set_direction(
        self,
        azimuth_deg: float,
        elevation_deg: float,
        confidence: float = 1.0,
    ) -> Tuple[float, float]:
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
        pan = float(np_clip(pan, min(cfg.PAN_MIN_DEG, 0.0), max(cfg.PAN_MAX_DEG, 180.0)))
        tilt = float(
            np_clip(tilt, min(cfg.TILT_MIN_DEG, 0.0), max(cfg.TILT_MAX_DEG, 180.0))
        )
        if not force and math.isclose(pan, self.pan_deg, abs_tol=0.05) and math.isclose(
            tilt, self.tilt_deg, abs_tol=0.05
        ):
            return

        self.pan_deg, self.tilt_deg = pan, tilt
        if self.simulate:
            log.debug("SIM servo pan=%.1f tilt=%.1f", pan, tilt)
        else:
            assert self.kit is not None
            self.kit.servo[cfg.PAN_CHANNEL].angle = pan
            self.kit.servo[cfg.TILT_CHANNEL].angle = tilt
        self._save_position()


def np_clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
