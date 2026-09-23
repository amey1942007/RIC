"""
Person-lock state machine: AUDIO → LOCKED → AUDIO.

AUDIO  — SRP-PHAT steers (existing pipeline). Lock time accumulates while
         one talker speaks from a stable bearing.
LOCKED — entered after LOCK_AFTER_SPEECH_S. The camera face track and the
         audio bearing are fused (never vision-only when audio is live).
         Left after UNLOCK_SILENCE_S of no speech, or a sustained speaker change.

Nothing here moves the servos; the tracker applies the fused bearing.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import config as cfg

log = logging.getLogger(__name__)

MODE_AUDIO = "AUDIO"
MODE_LOCKED = "LOCKED"


def wrap_deg(d: float) -> float:
    return (float(d) + 180.0) % 360.0 - 180.0


def pan_to_azimuth(pan: float) -> float:
    """Inverse of ServoController.map_angles pan branch (includes PAN_DIRECTION)."""
    f = float(cfg.PAN_FRONT_DEG)
    lo, hi = float(cfg.PAN_MIN_DEG), float(cfg.PAN_MAX_DEG)
    if pan >= f:
        az = (pan - f) / max(1e-6, hi - f) * float(cfg.AZIMUTH_MAX_DEG)
    else:
        az = (pan - f) / max(1e-6, f - lo) * abs(float(cfg.AZIMUTH_MIN_DEG))
    if int(getattr(cfg, "PAN_DIRECTION", 1)) < 0:
        az = -az
    return az


@dataclass
class LockSnapshot:
    mode: str = MODE_AUDIO
    enabled: bool = False
    fused_az: Optional[float] = None
    fused_el: Optional[float] = None
    fused_conf: float = 0.0
    source: str = "none"          # audio | visual | fused | hold
    lock_progress: float = 0.0    # 0..1 toward lock
    silence_s: float = 0.0
    face_visible: bool = False
    face_box: Optional[tuple] = None
    face_rel_az: Optional[float] = None
    reason: str = ""


class PersonLock:
    def __init__(self) -> None:
        self.enabled = bool(getattr(cfg, "PERSON_TRACK_ENABLED", True))
        self.mode = MODE_AUDIO
        self._anchor_az: Optional[float] = None
        self._hold_s = 0.0
        self._silence_s = 0.0
        self._change_s = 0.0
        self._fused_az: Optional[float] = None
        self._fused_el: Optional[float] = None
        self._last_t = time.monotonic()
        self._reason = "audio"

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self.enabled:
            return
        self.enabled = enabled
        self.reset("track-off" if not enabled else "track-on")
        log.info("Person track %s", "ON" if enabled else "OFF")

    def reset(self, reason: str = "reset") -> None:
        was = self.mode
        self.mode = MODE_AUDIO
        self._anchor_az = None
        self._hold_s = 0.0
        self._silence_s = 0.0
        self._change_s = 0.0
        self._fused_az = None
        self._fused_el = None
        self._reason = reason
        if was == MODE_LOCKED:
            log.info("Person lock → AUDIO (%s)", reason)

    def update(
        self,
        *,
        speech: bool,
        audio_az: Optional[float],
        audio_el: Optional[float],
        audio_conf: float,
        face,                       # FaceTrack | None
        pan_deg: float,
    ) -> LockSnapshot:
        now = time.monotonic()
        dt = max(1e-3, min(0.25, now - self._last_t))
        self._last_t = now

        face_vis = face is not None
        face_box = getattr(face, "box_px", None) if face_vis else None
        face_rel = float(face.rel_az_deg) if face_vis else None
        vis_az = vis_el = None
        if face_vis:
            vis_az = pan_to_azimuth(pan_deg) + float(face.rel_az_deg)
            vis_el = float(face.rel_el_deg)

        if not self.enabled:
            self.mode = MODE_AUDIO
            return LockSnapshot(
                mode=MODE_AUDIO, enabled=False, source="audio", reason="disabled"
            )

        if speech:
            self._silence_s = 0.0
        else:
            self._silence_s += dt

        need = float(getattr(cfg, "LOCK_AFTER_SPEECH_S", 6.0))
        tol = float(getattr(cfg, "LOCK_BEARING_TOL_DEG", 20.0))
        unlock_s = float(getattr(cfg, "UNLOCK_SILENCE_S", 20.0))
        change_s = float(getattr(cfg, "SPEAKER_CHANGE_S", 4.0))
        max_off = float(getattr(cfg, "LOCK_FACE_MAX_OFF_DEG", 30.0))
        w_aud = float(getattr(cfg, "FUSION_AUDIO_WEIGHT", 0.3))
        disagree = float(getattr(cfg, "FUSION_MAX_DISAGREE_DEG", 25.0))
        ema = float(getattr(cfg, "FUSION_EMA_ALPHA", 0.25))

        if self.mode == MODE_AUDIO:
            if speech and audio_az is not None:
                if self._anchor_az is None or abs(wrap_deg(audio_az - self._anchor_az)) > tol:
                    self._anchor_az = audio_az
                    self._hold_s = 0.0
                else:
                    self._hold_s += dt
            else:
                # pause: decay progress slowly so a breath doesn't reset a 5 s lock
                self._hold_s = max(0.0, self._hold_s - 0.35 * dt)

            progress = min(1.0, self._hold_s / max(1e-3, need))
            if self._hold_s >= need:
                face_ok = (not face_vis) or (abs(float(face.rel_az_deg)) <= max_off)
                if face_ok:
                    self.mode = MODE_LOCKED
                    self._fused_az = audio_az if audio_az is not None else vis_az
                    self._fused_el = audio_el if audio_el is not None else vis_el
                    self._change_s = 0.0
                    self._reason = "locked"
                    log.info("Person lock ON  az=%s  face=%s",
                             "—" if self._fused_az is None else f"{self._fused_az:.1f}",
                             "yes" if face_vis else "pending")
            return LockSnapshot(
                mode=self.mode,
                enabled=True,
                fused_az=self._fused_az,
                fused_el=self._fused_el,
                fused_conf=audio_conf if self.mode == MODE_LOCKED else 0.0,
                source="audio" if self.mode == MODE_AUDIO else "fused",
                lock_progress=1.0 if self.mode == MODE_LOCKED else progress,
                silence_s=self._silence_s,
                face_visible=face_vis,
                face_box=face_box,
                face_rel_az=face_rel,
                reason=self._reason,
            )

        # ── LOCKED ────────────────────────────────────────────────────────
        if self._silence_s >= unlock_s:
            self.reset(f"silence {self._silence_s:.1f}s")
            return LockSnapshot(
                mode=MODE_AUDIO, enabled=True, lock_progress=0.0,
                silence_s=self._silence_s, face_visible=face_vis,
                face_box=face_box, face_rel_az=face_rel, reason=self._reason,
            )

        # speaker change: sustained audio far from the locked person
        if speech and audio_az is not None and self._fused_az is not None:
            if abs(wrap_deg(audio_az - self._fused_az)) >= disagree:
                self._change_s += dt
                if self._change_s >= change_s:
                    self.reset(f"speaker-change az {audio_az:.0f}")
                    return LockSnapshot(
                        mode=MODE_AUDIO, enabled=True, lock_progress=0.0,
                        silence_s=self._silence_s, face_visible=face_vis,
                        face_box=face_box, face_rel_az=face_rel, reason=self._reason,
                    )
            else:
                self._change_s = 0.0
        else:
            self._change_s = max(0.0, self._change_s - dt)

        raw_az = raw_el = None
        source = "hold"
        conf = 0.7
        audio_ok = bool(speech and audio_az is not None and audio_conf >= float(cfg.CONFIDENCE_THRESHOLD))

        if audio_ok and vis_az is not None:
            daz = abs(wrap_deg(audio_az - vis_az))
            if daz <= disagree:
                raw_az = (1.0 - w_aud) * vis_az + w_aud * audio_az
                raw_el = (1.0 - w_aud) * (vis_el or 0.0) + w_aud * (audio_el or 0.0)
                source = "fused"
                conf = min(1.0, 0.55 * audio_conf + 0.45)
            else:
                # cues disagree: keep the face (audio is often a reflection / other talker)
                raw_az, raw_el, source, conf = vis_az, vis_el, "visual", 0.75
        elif vis_az is not None:
            raw_az, raw_el, source, conf = vis_az, vis_el, "visual", 0.72
        elif audio_ok:
            raw_az, raw_el, source, conf = audio_az, audio_el, "audio", audio_conf
        # else: hold last fused

        if raw_az is not None:
            if self._fused_az is None:
                self._fused_az, self._fused_el = raw_az, raw_el
            else:
                self._fused_az += ema * wrap_deg(raw_az - self._fused_az)
                if raw_el is not None:
                    prev_el = 0.0 if self._fused_el is None else self._fused_el
                    self._fused_el = prev_el + ema * (raw_el - prev_el)
            self._reason = source

        return LockSnapshot(
            mode=MODE_LOCKED,
            enabled=True,
            fused_az=self._fused_az,
            fused_el=self._fused_el,
            fused_conf=conf if raw_az is not None else 0.0,
            source=source,
            lock_progress=1.0,
            silence_s=self._silence_s,
            face_visible=face_vis,
            face_box=face_box,
            face_rel_az=face_rel,
            reason=self._reason,
        )
