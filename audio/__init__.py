"""Audio localisation package: Silero VAD, SRP-PHAT, Kalman smoother.

``SileroVAD`` is imported lazily so that ``audio.srp_phat`` / ``audio.kalman``
can be used on a dev machine without torch (e.g. scripts/validate_localization.py).
"""

from .srp_phat import SRPPhatLocalizer
from .kalman import KalmanFilter2D

__all__ = ["SileroVAD", "SRPPhatLocalizer", "KalmanFilter2D"]


def __getattr__(name: str):
    if name == "SileroVAD":
        from .silero_vad import SileroVAD

        return SileroVAD
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
