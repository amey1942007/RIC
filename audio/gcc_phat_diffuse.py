"""
GCC-PHAT with a coherent-to-diffuse-ratio (CDR) diffuseness mask.

STATUS: documented **fallback** for high-reverberation classrooms. It is NOT
the default localiser and is not wired into the running pipeline. To try it,
swap the localiser in ``LecturerTracker.__init__`` (pipeline/lecturer_tracker.py):

    from audio.gcc_phat_diffuse import GCCPhatDiffuseLocalizer
    self.localizer = GCCPhatDiffuseLocalizer(
        mic_positions=cfg.MIC_POSITIONS, ...same kwargs as SRPPhatLocalizer...
    )

Idea
----
Late reverberation is spatially diffuse: at a mic pair with spacing d the
coherence of an ideal diffuse field is Γ_n(f) = sinc(2π f d / c).  A direct
path adds a coherent component, so per time-frequency (T-F) bin the measured
short-time coherence Γ_x(f) lets us estimate the coherent-to-diffuse ratio
(DOA-independent estimator, Schwarz & Kellermann, IEEE/ACM TASLP 2015):

    CDR = ( Γn·Re(Γx) − |Γx|² − sqrt(Γn²·Re(Γx)² − Γn²·|Γx|² + Γn² − 2Γn·Re(Γx) + |Γx|²) )
          / ( |Γx|² − 1 )

Diffuseness D = 1 / (1 + CDR) ∈ [0, 1].  Bins with D ≥ ``GCC_DIFFUSE_THRESHOLD``
are dominated by reverberation and are zeroed (binary mask) before the PHAT
cross-spectrum is inverse-transformed, so only direct-path-dominated bins vote
in the steered-response grid search.

Confidence
----------
``confidence`` is the fraction of T-F bins retained by the mask, averaged over
all mic pairs — in [0, 1].  Note this is a *different* statistic from the
SRP-PHAT sigmoid peak-sharpness confidence; ``CONFIDENCE_THRESHOLD`` /
``CONFIDENCE_MOVE_THRESHOLD`` in config.py will need re-tuning if this
localiser is enabled.

Interface
---------
Same constructor and ``localize()`` signature / return keys as
``SRPPhatLocalizer`` (azimuth_deg, elevation_deg, confidence, srp_peak) plus an
extra ``retained_fraction`` key.  The grid search itself (adaptive or brute
force) is inherited unchanged.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from numpy.fft import rfft, irfft, rfftfreq

from .srp_phat import SRPPhatLocalizer, MicPos


class GCCPhatDiffuseLocalizer(SRPPhatLocalizer):
    """SRP-PHAT grid vote using CDR-masked GCC-PHAT per mic pair."""

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
        diffuse_threshold: Optional[float] = None,
        subframe: Optional[int] = None,
    ) -> None:
        import config as cfg

        super().__init__(
            mic_positions,
            sample_rate=sample_rate,
            speed_of_sound=speed_of_sound,
            n_fft=n_fft,
            azimuth_range=azimuth_range,
            elevation_range=elevation_range,
            azimuth_step=azimuth_step,
            elevation_step=elevation_step,
            adaptive=adaptive,
            coarse_multiplier=coarse_multiplier,
            volume_neighbors=volume_neighbors,
        )
        self.diffuse_threshold = float(
            getattr(cfg, "GCC_DIFFUSE_THRESHOLD", 0.6)
            if diffuse_threshold is None
            else diffuse_threshold
        )
        self.diffuse_threshold = float(np.clip(self.diffuse_threshold, 0.0, 1.0))
        sub = int(
            getattr(cfg, "GCC_DIFFUSE_SUBFRAME", 512) if subframe is None else subframe
        )
        # Sub-frame must fit at least twice into the analysis frame for a
        # meaningful coherence average.
        self.subframe = int(max(64, min(sub, self.n_fft // 2)))
        self.hop = self.subframe // 2
        self._sub_window = np.hanning(self.subframe)
        self._sub_freqs = rfftfreq(self.subframe, d=1.0 / self.sample_rate)
        self._full_freqs = rfftfreq(self.n_fft, d=1.0 / self.sample_rate)

        # Per-pair ideal diffuse-field coherence sinc(2π f d / c)
        self._gamma_n: Dict[Tuple[int, int], np.ndarray] = {}
        for i, j in self.pairs:
            d = float(np.linalg.norm(self.positions[i] - self.positions[j]))
            arg = 2.0 * np.pi * self._sub_freqs * d / self.speed_of_sound
            self._gamma_n[(i, j)] = np.sinc(arg / np.pi)  # np.sinc(x)=sin(πx)/(πx)

    # ── diffuseness estimation ─────────────────────────────────────────────
    def _stft(self, x: np.ndarray) -> np.ndarray:
        """(n_subframes, n_bins) short-time spectra of one channel."""
        n = len(x)
        starts = range(0, n - self.subframe + 1, self.hop)
        frames = np.stack([x[s : s + self.subframe] * self._sub_window for s in starts])
        return rfft(frames, axis=1)

    def _diffuseness(self, Xi: np.ndarray, Xj: np.ndarray, gamma_n: np.ndarray) -> np.ndarray:
        """
        Per-frequency diffuseness D ∈ [0,1] from short-time coherence,
        averaged over the sub-frames of the current analysis frame.
        """
        eps = 1e-12
        Pii = np.mean(np.abs(Xi) ** 2, axis=0) + eps
        Pjj = np.mean(np.abs(Xj) ** 2, axis=0) + eps
        Pij = np.mean(Xi * np.conj(Xj), axis=0)
        gx = Pij / np.sqrt(Pii * Pjj)  # complex coherence
        mag2 = np.clip(np.abs(gx) ** 2, 0.0, 1.0 - 1e-6)
        re = np.real(gx)
        gn = np.clip(gamma_n, -1.0 + 1e-6, 1.0 - 1e-6)

        inner = gn**2 * re**2 - gn**2 * mag2 + gn**2 - 2.0 * gn * re + mag2
        inner = np.maximum(inner, 0.0)
        cdr = (gn * re - mag2 - np.sqrt(inner)) / (mag2 - 1.0)
        cdr = np.maximum(cdr, 0.0)
        return 1.0 / (1.0 + cdr)

    def _masked_gcc_phat(
        self, x: np.ndarray, y: np.ndarray, pair: Tuple[int, int]
    ) -> Tuple[np.ndarray, float]:
        """CDR-masked GCC-PHAT for one pair → (cc, retained_fraction)."""
        D = self._diffuseness(self._stft(x), self._stft(y), self._gamma_n[pair])
        mask_sub = (D < self.diffuse_threshold).astype(np.float64)
        # Resample the sub-frame-resolution mask onto the full-frame bins
        mask_full = np.interp(self._full_freqs, self._sub_freqs, mask_sub)
        mask_full = (mask_full >= 0.5).astype(np.float64)  # keep it binary
        retained = float(np.mean(mask_full))

        n = self.n_fft
        R = rfft(x, n=n) * np.conj(rfft(y, n=n))
        denom = np.abs(R)
        denom[denom < 1e-12] = 1e-12
        R = (R / denom) * mask_full
        return irfft(R, n=n), retained

    # ── public API ─────────────────────────────────────────────────────────
    def localize(self, multichannel: np.ndarray) -> Dict[str, float]:
        """
        Same contract as ``SRPPhatLocalizer.localize`` — returns
        azimuth_deg, elevation_deg, confidence, srp_peak (+ retained_fraction).
        ``confidence`` = mean fraction of non-diffuse T-F bins kept, in [0, 1].
        """
        frames = self._as_mic_dict(multichannel)
        ccs, kept = [], []
        for i, j in self.pairs:
            cc, r = self._masked_gcc_phat(frames[i], frames[j], (i, j))
            ccs.append(cc)
            kept.append(r)
        gcc = np.stack(ccs, axis=0)
        out = self._search(gcc)
        retained = float(np.clip(np.mean(kept), 0.0, 1.0))
        out["confidence"] = retained
        out["retained_fraction"] = retained
        return out
