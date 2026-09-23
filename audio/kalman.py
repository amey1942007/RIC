"""
2D Kalman filter for azimuth / elevation smoothing.

Tuned for lecture tracking: high measurement noise, low process noise,
per-step and velocity clamps so one noisy SRP peak cannot yank the HAT.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class KalmanFilter2D:
    """Constant-velocity Kalman filter on a 2-D angular track."""

    def __init__(
        self,
        dt: float = 0.032,
        q_pos: float = 0.08,
        q_vel: float = 0.3,
        r_meas: float = 80.0,
        initial_az: float = 0.0,
        initial_el: float = 10.0,
        max_step_deg: float = 4.0,
        max_vel_deg_s: float = 40.0,
    ) -> None:
        self.dt = float(dt)
        self.max_step_deg = float(max_step_deg)
        self.max_vel_deg_s = float(max_vel_deg_s)

        self.x = np.array(
            [initial_az, initial_el, 0.0, 0.0], dtype=np.float64
        )
        self.P = np.diag([30.0, 30.0, 50.0, 50.0]).astype(np.float64)

        self.F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.H = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float64)
        self.R = np.eye(2, dtype=np.float64) * float(r_meas)
        self.I = np.eye(4, dtype=np.float64)
        self._last_az = float(initial_az)
        self._last_el = float(initial_el)

    def predict(self) -> Tuple[float, float]:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self._clamp_velocity()
        return float(self.x[0]), float(self.x[1])

    def update(
        self,
        azimuth: float,
        elevation: float,
        confidence: Optional[float] = None,
    ) -> Tuple[float, float]:
        z = np.array([azimuth, elevation], dtype=np.float64)
        R = self.R.copy()
        if confidence is not None:
            # low conf → much larger R (ignore junk peaks)
            c = float(np.clip(confidence, 0.05, 1.0))
            R = R / (c * c)

        y = z - self.H @ self.x
        y[0] = (y[0] + 180.0) % 360.0 - 180.0

        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P
        self._clamp_velocity()
        return float(self.x[0]), float(self.x[1])

    def step(
        self,
        measurement: Optional[Tuple[float, float]] = None,
        confidence: Optional[float] = None,
    ) -> Tuple[float, float]:
        self.predict()
        if measurement is not None:
            self.update(measurement[0], measurement[1], confidence)

        az, el = float(self.x[0]), float(self.x[1])
        # Hard slew limit so one spike cannot jump tens of degrees
        daz = az - self._last_az
        del_ = el - self._last_el
        daz = (daz + 180.0) % 360.0 - 180.0
        max_s = self.max_step_deg
        if abs(daz) > max_s:
            az = self._last_az + max_s * (1.0 if daz > 0 else -1.0)
        if abs(del_) > max_s:
            el = self._last_el + max_s * (1.0 if del_ > 0 else -1.0)
        self.x[0], self.x[1] = az, el
        self._last_az, self._last_el = az, el
        return az, el

    def _clamp_velocity(self) -> None:
        self.x[2] = float(np.clip(self.x[2], -self.max_vel_deg_s, self.max_vel_deg_s))
        self.x[3] = float(np.clip(self.x[3], -self.max_vel_deg_s, self.max_vel_deg_s))

    @property
    def state(self) -> dict:
        return {
            "azimuth_deg": float(self.x[0]),
            "elevation_deg": float(self.x[1]),
            "az_velocity": float(self.x[2]),
            "el_velocity": float(self.x[3]),
        }

    def reset(self, azimuth: float = 0.0, elevation: float = 10.0) -> None:
        self.x[:] = [azimuth, elevation, 0.0, 0.0]
        self.P = np.diag([30.0, 30.0, 50.0, 50.0]).astype(np.float64)
        self._last_az, self._last_el = float(azimuth), float(elevation)
