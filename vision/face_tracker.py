"""
Background face tracker for person lock.

Runs OpenCV's YuNet face detector (``cv2.FaceDetectorYN``, ONNX model in
``config.FACE_MODEL_PATH``) on frames from a :class:`camera.CameraFeed` at
``config.VISION_FPS`` and keeps ONE target face:

* acquisition — the face closest to the image centre (the HAT has already been
  steered at the talker by audio when the lock is requested);
* association — on later frames the detection nearest the previous target
  (normalised distance) continues the track; others are ignored;
* smoothing   — EMA of the centre / size (``VISION_EMA_ALPHA``).

The track is published as a :class:`FaceTrack` whose ``rel_az_deg`` is the
horizontal angle of the face from the camera's optical axis (pinhole model,
``CAMERA_HFOV_DEG``), positive toward the image's right × ``VISION_AZ_SIGN``.
Nothing here touches the servos; :mod:`pipeline.person_lock` does the fusion.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

import config as cfg

log = logging.getLogger(__name__)


@dataclass
class FaceTrack:
    cx: float                 # normalised centre x (0..1, image coords)
    cy: float                 # normalised centre y (0..1)
    w: float                  # normalised width
    h: float                  # normalised height
    score: float
    rel_az_deg: float         # horizontal angle from optical axis (+ = image right × sign)
    rel_el_deg: float         # vertical angle (+ = up)
    t: float                  # time of last detection (monotonic)
    n_faces: int = 0          # faces seen in that frame
    hits: int = 0             # consecutive frames with the target found
    id: int = 0               # increments when a new target is acquired
    box_px: tuple[int, int, int, int] = field(default=(0, 0, 0, 0))  # x, y, w, h in preview px

    def age(self) -> float:
        return time.monotonic() - self.t


class FaceTracker:
    def __init__(self, camera, model_path: Optional[Path] = None) -> None:
        self.camera = camera
        self.model_path = Path(model_path or cfg.FACE_MODEL_PATH)
        self.available = False
        self.error: Optional[str] = None
        self.fps_measured = 0.0
        self.detect_ms = 0.0

        self._det = None
        self._det_size: tuple[int, int] = (0, 0)
        self._track: Optional[FaceTrack] = None
        self._track_id = 0
        self._want_acquire = True
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._hfov = float(getattr(cfg, "CAMERA_HFOV_DEG", 66.0))
        self._sign = 1.0 if int(getattr(cfg, "VISION_AZ_SIGN", 1)) >= 0 else -1.0

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self) -> bool:
        if self._thread is not None:
            return self.available
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.error = "OpenCV (cv2) not importable"
            log.warning("FaceTracker disabled — %s", self.error)
            return False
        if not self.model_path.exists():
            self.error = f"face model missing: {self.model_path}"
            log.warning("FaceTracker disabled — %s", self.error)
            return False
        self.available = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="face-tracker", daemon=True)
        self._thread.start()
        log.info("FaceTracker started (YuNet %s, %d fps, HFOV %.0f°)",
                 self.model_path.name, int(cfg.VISION_FPS), self._hfov)
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.available = False

    # ── public state ───────────────────────────────────────────────────────
    def latest(self) -> Optional[FaceTrack]:
        with self._lock:
            return self._track

    def visible(self, max_age_s: Optional[float] = None) -> Optional[FaceTrack]:
        """Current target if it was seen within ``max_age_s`` (config.VISION_STALE_S)."""
        t = self.latest()
        if t is None:
            return None
        lim = float(cfg.VISION_STALE_S if max_age_s is None else max_age_s)
        return t if t.age() <= lim else None

    def reacquire(self) -> None:
        """Drop the current target; next frame picks the face nearest the centre."""
        with self._lock:
            self._want_acquire = True
            self._track = None

    # ── internals ──────────────────────────────────────────────────────────
    def _ensure_detector(self, w: int, h: int):
        import cv2

        if self._det is None or self._det_size != (w, h):
            self._det = cv2.FaceDetectorYN.create(
                str(self.model_path), "", (w, h),
                float(cfg.VISION_SCORE_MIN), 0.3, 5000,
            )
            self._det_size = (w, h)
        return self._det

    def _angles(self, cx: float, cy: float, aspect: float) -> tuple[float, float]:
        """Pinhole: normalised offset → angle from the optical axis."""
        half_w = math.tan(math.radians(self._hfov / 2.0))
        half_h = half_w / max(1e-6, aspect)
        az = math.degrees(math.atan((cx - 0.5) * 2.0 * half_w)) * self._sign
        el = -math.degrees(math.atan((cy - 0.5) * 2.0 * half_h))  # image y down → el up
        return az, el

    def _loop(self) -> None:
        import cv2

        period = 1.0 / max(1, int(cfg.VISION_FPS))
        det_w = int(cfg.VISION_DETECT_WIDTH)
        alpha = float(cfg.VISION_EMA_ALPHA)
        n = 0
        t_fps = time.monotonic()
        last_frame_id = -1
        while not self._stop.is_set():
            t0 = time.monotonic()
            frame = self.camera.latest() if self.camera is not None else None
            fid = getattr(self.camera, "frame_count", 0)
            if frame is None or fid == last_frame_id:
                time.sleep(0.01)
                continue
            last_frame_id = fid
            try:
                fh, fw = frame.shape[:2]
                scale = det_w / float(fw)
                dh = max(16, int(round(fh * scale)))
                small = cv2.resize(frame, (det_w, dh), interpolation=cv2.INTER_AREA)
                bgr = np.ascontiguousarray(small[:, :, ::-1])  # detector expects BGR
                det = self._ensure_detector(det_w, dh)
                td = time.monotonic()
                _, faces = det.detect(bgr)
                self.detect_ms = 0.8 * self.detect_ms + 0.2 * (time.monotonic() - td) * 1000.0
                self._update_track(faces, det_w, dh, fw, fh, alpha)
            except Exception as exc:  # pragma: no cover - hardware dependent
                self.error = str(exc)
                log.warning("face detection failed: %s", exc)
                time.sleep(0.5)
                continue

            n += 1
            now = time.monotonic()
            if now - t_fps >= 1.0:
                self.fps_measured = n / (now - t_fps)
                n, t_fps = 0, now
            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)

    def _update_track(self, faces, dw: int, dh: int, fw: int, fh: int, alpha: float) -> None:
        now = time.monotonic()
        cands = []
        if faces is not None:
            for f in faces:
                x, y, w, h = float(f[0]), float(f[1]), float(f[2]), float(f[3])
                score = float(f[-1])
                if w <= 0 or h <= 0:
                    continue
                cands.append(((x + w / 2) / dw, (y + h / 2) / dh, w / dw, h / dh, score))

        with self._lock:
            prev = self._track
            chosen = None
            if cands:
                if prev is None or self._want_acquire or prev.age() > float(cfg.VISION_STALE_S) * 3:
                    # acquisition: nearest to the image centre
                    chosen = min(cands, key=lambda c: (c[0] - 0.5) ** 2 + (c[1] - 0.5) ** 2)
                    self._track_id += 1
                    self._want_acquire = False
                    prev = None
                else:
                    # association: nearest to the previous target, within ~1.5 face widths
                    d, chosen = min(
                        (((c[0] - prev.cx) ** 2 + (c[1] - prev.cy) ** 2) ** 0.5, c) for c in cands
                    )
                    if d > max(0.12, 1.5 * prev.w):
                        chosen = None
            if chosen is None:
                return  # keep previous track (it ages out via .age())

            cx, cy, w, h, score = chosen
            if prev is not None:
                cx = (1 - alpha) * prev.cx + alpha * cx
                cy = (1 - alpha) * prev.cy + alpha * cy
                w = (1 - alpha) * prev.w + alpha * w
                h = (1 - alpha) * prev.h + alpha * h
            az, el = self._angles(cx, cy, fw / float(fh))
            self._track = FaceTrack(
                cx=cx, cy=cy, w=w, h=h, score=score,
                rel_az_deg=az, rel_el_deg=el, t=now,
                n_faces=len(cands),
                hits=(prev.hits + 1) if prev is not None else 1,
                id=self._track_id,
                box_px=(int((cx - w / 2) * fw), int((cy - h / 2) * fh), int(w * fw), int(h * fh)),
            )
