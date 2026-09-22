"""
SRP-PHAT (Steered Response Power with PHAT weighting) localiser.

Uses all C(4,2)=6 microphone pairs voting on a shared (azimuth, elevation)
grid. Designed for a planar square array with 6 cm adjacent mic spacing.

Note: the d < λ/2 spatial-aliasing rule applies to delay-and-sum beamforming,
not to SRP-PHAT voting grids. Classroom-practical spacing limit ~10 cm.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
from numpy.fft import rfft, irfft


MicPos = Mapping[int, Sequence[float]]


class SRPPhatLocalizer:
    """Precomputed-TDOA SRP-PHAT over azimuth × elevation."""

    def __init__(
        self,
        mic_positions: MicPos,
        sample_rate: int = 16_000,
        speed_of_sound: float = 343.0,
        n_fft: int = 2048,
        azimuth_range: Tuple[float, float] = (-90.0, 90.0),
        elevation_range: Tuple[float, float] = (0.0, 60.0),
        azimuth_step: float = 2.0,
        elevation_step: float = 2.0,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.speed_of_sound = float(speed_of_sound)
        self.n_fft = int(n_fft)

        self.mic_ids = sorted(mic_positions.keys())
        if len(self.mic_ids) < 2:
            raise ValueError("need at least 2 microphones")

        self.positions = {
            i: np.asarray(mic_positions[i], dtype=np.float64).reshape(3)
            for i in self.mic_ids
        }
        self.pairs = list(combinations(self.mic_ids, 2))

        self.azimuths = np.arange(
            azimuth_range[0], azimuth_range[1] + 1e-9, azimuth_step, dtype=np.float64
        )
        self.elevations = np.arange(
            elevation_range[0], elevation_range[1] + 1e-9, elevation_step, dtype=np.float64
        )
        self._precompute_tdoa_grid()

    # ── geometry / grid ────────────────────────────────────────────────────
    def _direction_unit(self, az_deg: float, el_deg: float) -> np.ndarray:
        """
        Unit vector toward a source at azimuth / elevation.

        Shared convention:
          az = 0 → +Y (front into the room), az > 0 → toward +X (right)

        Elevation sign depends on mount (config.ARRAY_MOUNT):
          vertical_stand — el > 0 → +Z (up)
          ceiling        — el > 0 → −Z (toward floor)
        """
        import config as cfg

        az = np.deg2rad(az_deg)
        el = np.deg2rad(el_deg)
        x = np.sin(az) * np.cos(el)
        y = np.cos(az) * np.cos(el)
        if getattr(cfg, "ARRAY_MOUNT", "vertical_stand") == "ceiling":
            z = -np.sin(el)
        else:
            z = np.sin(el)
        v = np.array([x, y, z], dtype=np.float64)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _precompute_tdoa_grid(self) -> None:
        """
        For each (az, el) candidate and each mic pair, store expected sample delay.
        delay[i→j] = (proj_j - proj_i) / c * fs  (positive ⇒ j further from source).
        """
        n_az = len(self.azimuths)
        n_el = len(self.elevations)
        n_pairs = len(self.pairs)
        self.tdoa_samples = np.zeros((n_az, n_el, n_pairs), dtype=np.float64)

        for ia, az in enumerate(self.azimuths):
            for ie, el in enumerate(self.elevations):
                dvec = self._direction_unit(az, el)
                for ip, (i, j) in enumerate(self.pairs):
                    # Far-field: τ = (r_i - r_j) · û / c
                    # Signal at mic i relative to j arrives earlier if i is closer.
                    tau = float(
                        np.dot(self.positions[i] - self.positions[j], dvec)
                        / self.speed_of_sound
                    )
                    self.tdoa_samples[ia, ie, ip] = tau * self.sample_rate

        # Max |delay| used when aligning GCC peaks
        self.max_delay = int(np.ceil(np.max(np.abs(self.tdoa_samples)))) + 2

    # ── PHAT GCC ───────────────────────────────────────────────────────────
    def _gcc_phat(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return circular GCC-PHAT correlation (length n_fft)."""
        n = self.n_fft
        X = rfft(x, n=n)
        Y = rfft(y, n=n)
        R = X * np.conj(Y)
        denom = np.abs(R)
        denom[denom < 1e-12] = 1e-12
        R /= denom  # PHAT weighting
        cc = irfft(R, n=n)
        return cc

    def _interp_peak(self, cc: np.ndarray, delay_samples: float) -> float:
        """
        Linear-interpolate GCC-PHAT at expected pair delay.

        For cc = irfft(Xi * conj(Xj)), if channel j is delayed by +d samples
        relative to i, the peak sits at index (-d) % n (NumPy FFT convention).
        ``delay_samples`` stores physical (τ_j - τ_i)*fs = ((r_i-r_j)·û/c)*fs,
        so we read cc at (-delay_samples) % n.
        """
        n = len(cc)
        idx = (-delay_samples) % n
        i0 = int(np.floor(idx)) % n
        i1 = (i0 + 1) % n
        frac = idx - np.floor(idx)
        return float((1.0 - frac) * cc[i0] + frac * cc[i1])

    # ── public API ─────────────────────────────────────────────────────────
    def localize(self, multichannel: np.ndarray) -> Dict[str, float]:
        """
        Estimate source direction from a short multichannel frame.

        Parameters
        ----------
        multichannel : (n_samples, n_mics) or (n_mics, n_samples) float array.
            Channel order must match mic_ids order (0..3 for square array).

        Returns
        -------
        dict with azimuth_deg, elevation_deg, confidence ∈ [0, 1]
        """
        frames = self._as_mic_dict(multichannel)

        # One GCC-PHAT per pair
        gccs = []
        for i, j in self.pairs:
            gccs.append(self._gcc_phat(frames[i], frames[j]))

        power = np.zeros((len(self.azimuths), len(self.elevations)), dtype=np.float64)
        for ia in range(len(self.azimuths)):
            for ie in range(len(self.elevations)):
                s = 0.0
                for ip, cc in enumerate(gccs):
                    s += self._interp_peak(cc, self.tdoa_samples[ia, ie, ip])
                power[ia, ie] = s

        # Softmax-style confidence from peak prominence
        flat = power.ravel()
        peak_idx = int(np.argmax(flat))
        ia, ie = np.unravel_index(peak_idx, power.shape)
        peak = float(flat[peak_idx])
        mean = float(np.mean(flat))
        std = float(np.std(flat)) + 1e-9
        # Map z-score-ish score into [0, 1]
        confidence = float(1.0 / (1.0 + np.exp(-(peak - mean) / std)))

        return {
            "azimuth_deg": float(self.azimuths[ia]),
            "elevation_deg": float(self.elevations[ie]),
            "confidence": confidence,
            "srp_peak": peak,
        }

    def _as_mic_dict(self, multichannel: np.ndarray) -> Dict[int, np.ndarray]:
        x = np.asarray(multichannel, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("multichannel must be 2-D")

        # Accept (samples, mics) or (mics, samples)
        if x.shape[1] == len(self.mic_ids):
            # (samples, mics)
            channels = {mid: x[:, k] for k, mid in enumerate(self.mic_ids)}
        elif x.shape[0] == len(self.mic_ids):
            channels = {mid: x[k, :] for k, mid in enumerate(self.mic_ids)}
        else:
            raise ValueError(
                f"expected {len(self.mic_ids)} channels, got shape {x.shape}"
            )

        # Zero-mean + light window to reduce spectral leakage
        out: Dict[int, np.ndarray] = {}
        for mid, sig in channels.items():
            sig = sig - np.mean(sig)
            if len(sig) < self.n_fft:
                pad = np.zeros(self.n_fft, dtype=np.float64)
                pad[: len(sig)] = sig
                sig = pad
            else:
                sig = sig[: self.n_fft]
            window = np.hanning(len(sig))
            out[mid] = sig * window
        return out
