"""
SRP-PHAT (Steered Response Power with PHAT weighting) localiser.

Uses all C(4,2)=6 microphone pairs voting on a shared (azimuth, elevation)
grid. Designed for a planar square array with 6 cm adjacent mic spacing.

Search strategy (config.SRP_ADAPTIVE_SEARCH):
  * adaptive (default) — coarse-to-fine: pass 1 evaluates the TDOA grid at
    ``SRP_COARSE_STEP_MULTIPLIER × fine step``, pass 2 refines only a local
    window (±1 coarse step) around the best coarse cell at the fine step.
    Reduces per-frame search cost vs full dense grid; based on
    modified/adaptive-search SRP-PHAT literature.
  * brute force — every fine cell evaluated (legacy path, kept for A/B).

"Modified SRP-PHAT" volume accumulation (config.SRP_VOLUME_NEIGHBORS): each
candidate cell's score is the unweighted mean of the interpolated GCC-PHAT
value at the cell and its immediate neighbours on the angle grid, which
stabilises the peak against fractional-delay interpolation error.

The full fine TDOA grid is precomputed up front — at the current config
(91 az × 46 el × 6 pairs ≈ 25k floats) that is far cheaper than the
bookkeeping for on-demand fine-grid generation, so the adaptive path simply
skips evaluation of cells outside the refined window.

Note: the d < λ/2 spatial-aliasing rule applies to delay-and-sum beamforming,
not to SRP-PHAT voting grids. Classroom-practical spacing limit ~10 cm.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, Mapping, Optional, Sequence, Tuple

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
        adaptive: Optional[bool] = None,
        coarse_multiplier: Optional[int] = None,
        volume_neighbors: Optional[int] = None,
    ) -> None:
        import config as cfg

        self.sample_rate = int(sample_rate)
        self.speed_of_sound = float(speed_of_sound)
        self.n_fft = int(n_fft)

        # Search-strategy knobs: explicit arg > config > safe default
        self.adaptive = bool(
            getattr(cfg, "SRP_ADAPTIVE_SEARCH", True) if adaptive is None else adaptive
        )
        self.coarse_multiplier = max(
            1,
            int(
                getattr(cfg, "SRP_COARSE_STEP_MULTIPLIER", 3)
                if coarse_multiplier is None
                else coarse_multiplier
            ),
        )
        self.volume_neighbors = max(
            0,
            int(
                getattr(cfg, "SRP_VOLUME_NEIGHBORS", 1)
                if volume_neighbors is None
                else volume_neighbors
            ),
        )

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
        For each fine (az, el) candidate and each mic pair, store expected
        sample delay.  delay[i→j] = (proj_j - proj_i) / c * fs
        (positive ⇒ j further from source).

        Also derives the coarse index sets used by pass 1 of the adaptive
        search (every ``coarse_multiplier``-th fine index, always including
        the last index so the range edges are covered).
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
                    tau = float(
                        np.dot(self.positions[i] - self.positions[j], dvec)
                        / self.speed_of_sound
                    )
                    self.tdoa_samples[ia, ie, ip] = tau * self.sample_rate

        # Max |delay| used when aligning GCC peaks
        self.max_delay = int(np.ceil(np.max(np.abs(self.tdoa_samples)))) + 2

        m = self.coarse_multiplier
        self.coarse_ia = self._coarse_indices(n_az, m)
        self.coarse_ie = self._coarse_indices(n_el, m)

    @staticmethod
    def _coarse_indices(n: int, mult: int) -> np.ndarray:
        idx = np.arange(0, n, mult, dtype=np.int64)
        if idx[-1] != n - 1:
            idx = np.append(idx, n - 1)
        return idx

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
        Linear-interpolate GCC-PHAT at expected pair delay (scalar helper).

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

    def _eval_power(
        self,
        gcc: np.ndarray,
        ia_idx: np.ndarray,
        ie_idx: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorised steered-response power for the cell subset
        ``ia_idx × ie_idx`` (fine-grid indices).

        gcc : (n_pairs, n_lags) stacked GCC-PHAT correlations.
        Returns (len(ia_idx), len(ie_idx)) power with volume accumulation
        over ±volume_neighbors fine-grid cells (unweighted mean).
        TODO(tuning): try distance-weighted neighbour accumulation.
        """
        n_lags = gcc.shape[1]
        n_pairs = gcc.shape[0]
        n_az, n_el = self.tdoa_samples.shape[:2]
        pair_ax = np.arange(n_pairs)[None, None, :]

        nb = self.volume_neighbors
        acc = np.zeros((len(ia_idx), len(ie_idx)), dtype=np.float64)
        count = 0
        for da in range(-nb, nb + 1):
            ia = np.clip(ia_idx + da, 0, n_az - 1)
            for de in range(-nb, nb + 1):
                ie = np.clip(ie_idx + de, 0, n_el - 1)
                # (na, ne, n_pairs) expected delays
                T = self.tdoa_samples[np.ix_(ia, ie)]
                idx = (-T) % n_lags
                i0 = np.floor(idx).astype(np.int64) % n_lags
                i1 = (i0 + 1) % n_lags
                frac = idx - np.floor(idx)
                vals = (1.0 - frac) * gcc[pair_ax, i0] + frac * gcc[pair_ax, i1]
                acc += vals.sum(axis=2)
                count += 1
        return acc / float(count)

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
        dict with azimuth_deg, elevation_deg, confidence ∈ [0, 1], srp_peak
        """
        frames = self._as_mic_dict(multichannel)
        gcc = np.stack(
            [self._gcc_phat(frames[i], frames[j]) for i, j in self.pairs], axis=0
        )
        return self._search(gcc)

    def _search(self, gcc: np.ndarray) -> Dict[str, float]:
        """Run adaptive or brute-force grid search over a stacked GCC matrix."""
        n_az, n_el = self.tdoa_samples.shape[:2]
        all_ia = np.arange(n_az)
        all_ie = np.arange(n_el)

        if not self.adaptive:
            power = self._eval_power(gcc, all_ia, all_ie)
            flat = power.ravel()
            k = int(np.argmax(flat))
            ia, ie = np.unravel_index(k, power.shape)
            best_ia, best_ie = int(all_ia[ia]), int(all_ie[ie])
            evaluated = flat
            peak = float(flat[k])
        else:
            # Pass 1 — coarse grid
            p_coarse = self._eval_power(gcc, self.coarse_ia, self.coarse_ie)
            k = int(np.argmax(p_coarse))
            ca, ce = np.unravel_index(k, p_coarse.shape)
            c_ia, c_ie = int(self.coarse_ia[ca]), int(self.coarse_ie[ce])

            # Pass 2 — fine window ±1 coarse step around best coarse cell
            m = self.coarse_multiplier
            win_ia = np.arange(max(0, c_ia - m), min(n_az, c_ia + m + 1))
            win_ie = np.arange(max(0, c_ie - m), min(n_el, c_ie + m + 1))
            p_fine = self._eval_power(gcc, win_ia, win_ie)
            k2 = int(np.argmax(p_fine))
            fa, fe = np.unravel_index(k2, p_fine.shape)
            best_ia, best_ie = int(win_ia[fa]), int(win_ie[fe])
            peak = float(p_fine[fa, fe])
            evaluated = np.concatenate([p_coarse.ravel(), p_fine.ravel()])

        mean = float(np.mean(evaluated))
        std = float(np.std(evaluated)) + 1e-9
        confidence = float(1.0 / (1.0 + np.exp(-(peak - mean) / std)))

        return {
            "azimuth_deg": float(self.azimuths[best_ia]),
            "elevation_deg": float(self.elevations[best_ie]),
            "confidence": confidence,
            "srp_peak": peak,
        }

    def _as_mic_dict(self, multichannel: np.ndarray) -> Dict[int, np.ndarray]:
        x = np.asarray(multichannel, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("multichannel must be 2-D")

        # Accept (samples, mics) or (mics, samples)
        if x.shape[1] == len(self.mic_ids):
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
