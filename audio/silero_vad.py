"""
Silero VAD wrapper — speech / non-speech gating for the localisation pipeline.

Requires 16 kHz mono float32 audio in 512-sample chunks.
Outputs speech probability in [0, 1]; threshold default 0.5.

Far-field note: Silero ships one main model (v6 JIT). Quiet distant speech
drops confidence; AGC (``normalize=True``) can help later. Default for
debugging is raw audio (``config.VAD_NORMALIZE = False``).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


class SileroVAD:
    """Thin wrapper around snakers4/silero-vad for chunked live inference."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        threshold: float = 0.5,
        device: Optional[str] = None,
        *,
        normalize: bool = False,
        target_peak: float = 0.6,
        noise_floor: float = 0.004,
        max_gain: float = 40.0,
    ) -> None:
        if sample_rate not in (8_000, 16_000):
            raise ValueError("Silero VAD only supports 8000 or 16000 Hz")

        self.sample_rate = sample_rate
        self.threshold = float(threshold)
        self.chunk_samples = 256 if sample_rate == 8_000 else 512
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Level-invariant preprocessing (helps when the speaker walks away)
        self.normalize = bool(normalize)
        self.target_peak = float(target_peak)
        self.noise_floor = float(noise_floor)
        self.max_gain = float(max_gain)
        self.last_peak = 0.0
        self.last_gain = 1.0

        self.model = self._load_model()
        if hasattr(self.model, "eval"):
            self.model.eval()
        self._reset_states()

    def _load_model(self):
        """
        Load the latest Silero VAD (v5/v6 JIT ~2 MB — the only full model).
        Prefer official loader, then packaged JIT, then torch.hub.
        """
        # 1) Official helper — current pip package weights (v6)
        try:
            from silero_vad import load_silero_vad

            model = load_silero_vad(onnx=False)
            try:
                model.to(self.device)
            except Exception:
                pass
            return model
        except Exception:
            pass

        # 2) Direct JIT from installed package data
        try:
            try:
                import importlib_resources as impresources
            except ImportError:
                from importlib import resources as impresources

            model_path = str(
                impresources.files("silero_vad.data").joinpath("silero_vad.jit")
            )
            model = torch.jit.load(model_path, map_location=self.device)
            model.eval()
            return model
        except Exception:
            pass

        # 3) torch.hub fallback
        model, _utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        try:
            model.to(self.device)
        except Exception:
            pass
        return model

    def _reset_states(self) -> None:
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()

    def reset(self) -> None:
        """Clear RNN state (call between recordings / streams)."""
        self._reset_states()

    @torch.inference_mode()
    def probability(self, chunk: np.ndarray) -> float:
        """
        Return speech probability for one chunk.

        Parameters
        ----------
        chunk : np.ndarray
            Mono float32 samples, length == self.chunk_samples.
            Values should be roughly in [-1, 1].
        """
        audio = self._prepare_chunk(chunk)
        tensor = torch.from_numpy(audio).to(self.device)
        out = self.model(tensor, self.sample_rate)
        if isinstance(out, (tuple, list)):
            out = out[0]
        prob = float(out.item() if hasattr(out, "item") else out)
        return prob

    def is_speech(self, chunk: np.ndarray) -> bool:
        """True if speech probability >= threshold."""
        return self.probability(chunk) >= self.threshold

    def classify(self, chunk: np.ndarray) -> dict:
        """
        Human-speech vs non-speech label for a chunk.

        Silero VAD detects *speech activity* (typically human voice) vs
        silence / noise / non-speech. It is not a full speaker-ID or
        animal-sound classifier — labels are SPEECH vs NON_SPEECH.
        """
        prob = self.probability(chunk)
        is_speech = prob >= self.threshold
        return {
            "label": "SPEECH" if is_speech else "NON_SPEECH",
            "human_speech": is_speech,
            "probability": prob,
            "threshold": self.threshold,
            "peak": self.last_peak,
            "gain": self.last_gain,
        }

    def probability_from_buffer(self, mono: np.ndarray) -> list[float]:
        """Run VAD over a longer mono buffer; return per-chunk probabilities."""
        mono = np.asarray(mono, dtype=np.float32).reshape(-1)
        n = self.chunk_samples
        probs: list[float] = []
        self.reset()
        for i in range(0, len(mono) - n + 1, n):
            probs.append(self.probability(mono[i : i + n]))
        return probs

    def _prepare_chunk(self, chunk: np.ndarray) -> np.ndarray:
        audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if audio.ndim != 1:
            raise ValueError("chunk must be mono 1-D")
        if len(audio) != self.chunk_samples:
            raise ValueError(
                f"expected {self.chunk_samples} samples, got {len(audio)}"
            )

        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        self.last_peak = peak
        self.last_gain = 1.0

        if peak <= 0.0:
            return audio

        # Clip protection only when louder than full scale
        if peak > 1.0:
            audio = audio / peak
            peak = 1.0
            self.last_gain = 1.0 / self.last_peak

        if not self.normalize:
            return audio

        # Do not boost near-silence (avoids amplifying room noise into "speech")
        if peak < self.noise_floor:
            return audio

        # Peak-normalize quiet / far speech to a stable level for the VAD
        gain = min(self.target_peak / peak, self.max_gain)
        self.last_gain = float(gain)
        return audio * gain
