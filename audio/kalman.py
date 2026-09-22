"""
2D Kalman filter for azimuth / elevation smoothing.

State: [azimuth, elevation, az_velocity, el_velocity]
- Predict every audio frame
- Update only when speech is detected (VAD gate)
Tuned smooth-over-fast for lecture recording (high R relative to Q).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class KalmanFilter2D:
    """Constant-velocity Kalman filter on a 2-D angular track."""

    def __init__(
        self,
        dt: float = 0.032,
        q_pos: float = 0.5,
        q_vel: float = 2.0,
        r_meas: float = 25.0,
        initial_az: float = 0.0,
        initial_el: float = 20.0,
    ) -> None:
        self.dt = float(dt)

        # State: [az, el, vaz, vel]
        self.x = np.array(
            [initial_az, initial_el, 0.0, 0.0], dtype=np.float64
        )

        self.P = np.diag([50.0, 50.0, 100.0, 100.0]).astype(np.float64)

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
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        q = np.array(
            [
                [q_pos, 0.0, 0.0, 0.0],
                [0.0, q_pos, 0.0, 0.0],
                [0.0, 0.0, q_vel, 0.0],
                [0.0, 0.0, 0.0, q_vel],
            ],
            dtype=np.float64,
        )
        self.Q = q

        self.R = np.eye(2, dtype=np.float64) * float(r_meas)
        self.I = np.eye(4, dtype=np.float64)

    def predict(self) -> Tuple[float, float]:
        """Advance state by one frame; return predicted (az, el)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0]), float(self.x[1])

    def update(
        self,
        azimuth: float,
        elevation: float,
        confidence: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Measurement update. Optional confidence scales measurement noise
        (lower confidence → larger R → trust prediction more).
        """
        z = np.array([azimuth, elevation], dtype=np.float64)

        R = self.R.copy()
        if confidence is not None:
            # confidence 1 → R, confidence 0 → 20×R
            c = float(np.clip(confidence, 0.05, 1.0))
            R = R / c

        y = z - self.H @ self.x
        # Wrap azimuth innovation into [-180, 180]
        y[0] = (y[0] + 180.0) % 360.0 - 180.0

        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P
        return float(self.x[0]), float(self.x[1])

    def step(
        self,
        measurement: Optional[Tuple[float, float]] = None,
        confidence: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Predict always; update only if measurement is provided (speech gate).
        """
        az, el = self.predict()
        if measurement is not None:
            az, el = self.update(measurement[0], measurement[1], confidence)
        return az, el

    @property
    def state(self) -> dict:
        return {
            "azimuth_deg": float(self.x[0]),
            "elevation_deg": float(self.x[1]),
            "az_velocity": float(self.x[2]),
            "el_velocity": float(self.x[3]),
        }

    def reset(self, azimuth: float = 0.0, elevation: float = 20.0) -> None:
        self.x[:] = [azimuth, elevation, 0.0, 0.0]
        self.P = np.diag([50.0, 50.0, 100.0, 100.0]).astype(np.float64)
