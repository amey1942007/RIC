"""
Background person tracker for person lock.

Runs YOLOv8n (ONNX, COCO class 0 = person only) on frames from
:class:`camera.CameraFeed`. Keeps ONE target body:

* acquisition — person closest to the image centre (HAT already steered by audio);
* association — later frames take the detection nearest the previous box;
* smoothing   — EMA of centre / size.

Falls back to OpenCV HOG pedestrians if the ONNX file or cv2.dnn is missing.
``rel_az_deg`` is the horizontal angle from the optical axis (pinhole,
``CAMERA_HFOV_DEG``). Servos are driven by :mod:`pipeline.person_lock`.
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
class PersonTrack:
    cx: float
    cy: float
    w: float
    h: float
    score: float
    rel_az_deg: float
    rel_el_deg: float
    t: float
    n_faces: int = 0          # name kept: number of person boxes this frame
    hits: int = 0
    id: int = 0
    box_px: tuple[int, int, int, int] = field(default=(0, 0, 0, 0))

    def age(self) -> float:
        return time.monotonic() - self.t


# person_lock / GUI still import FaceTrack
FaceTrack = PersonTrack


class PersonTracker:
    PERSON_CLASS = 0  # COCO

    def __init__(self, camera, model_path: Optional[Path] = None) -> None:
        self.camera = camera
        self.model_path = Path(model_path or getattr(cfg, "PERSON_MODEL_PATH", cfg.FACE_MODEL_PATH))
        self.available = False
        self.error: Optional[str] = None
        self.fps_measured = 0.0
        self.detect_ms = 0.0
        self.backend = "none"

        self._net = None
        self._hog = None
        self._in_w = 0
        self._track: Optional[PersonTrack] = None
        self._track_id = 0
        self._want_acquire = True
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._hfov = float(getattr(cfg, "CAMERA_HFOV_DEG", 66.0))
        self._sign = 1.0 if int(getattr(cfg, "VISION_AZ_SIGN", 1)) >= 0 else -1.0

    def start(self) -> bool:
        if self._thread is not None:
            return self.available
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.error = "OpenCV (cv2) not importable"
            log.warning("PersonTracker disabled — %s", self.error)
            return False
        if not self._open_backend():
            return False
        self.available = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="person-tracker", daemon=True)
        self._thread.start()
        log.info("PersonTracker started via %s (%s, %d fps, HFOV %.0f°)",
                 self.backend, self.model_path.name, int(cfg.VISION_FPS), self._hfov)
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.available = False

    def latest(self) -> Optional[PersonTrack]:
        with self._lock:
            return self._track

    def visible(self, max_age_s: Optional[float] = None) -> Optional[PersonTrack]:
        t = self.latest()
        if t is None:
            return None
        lim = float(cfg.VISION_STALE_S if max_age_s is None else max_age_s)
        return t if t.age() <= lim else None

    def reacquire(self) -> None:
        with self._lock:
            self._want_acquire = True
            self._track = None

    def _open_backend(self) -> bool:
        import cv2

        if self.model_path.exists() and hasattr(cv2, "dnn"):
            try:
                self._net = cv2.dnn.readNetFromONNX(str(self.model_path))
                self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.backend = "yolov8n"
                return True
            except Exception as exc:
                log.warning("YOLO ONNX failed (%s) — trying HOG", exc)
                self._net = None
        try:
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self._hog = hog
            self.backend = "hog"
            self.error = None
            return True
        except Exception as exc:
            self.error = f"no person detector: {exc}"
            log.warning("PersonTracker disabled — %s", self.error)
            return False

    def _angles(self, cx: float, cy: float, aspect: float) -> tuple[float, float]:
        half_w = math.tan(math.radians(self._hfov / 2.0))
        half_h = half_w / max(1e-6, aspect)
        az = math.degrees(math.atan((cx - 0.5) * 2.0 * half_w)) * self._sign
        el = -math.degrees(math.atan((cy - 0.5) * 2.0 * half_h))
        return az, el

    def _letterbox(self, bgr: np.ndarray, size: int):
        h, w = bgr.shape[:2]
        scale = size / float(max(h, w))
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        import cv2
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        top = (size - nh) // 2
        left = (size - nw) // 2
        canvas[top : top + nh, left : left + nw] = resized
        return canvas, scale, left, top

    def _detect_yolo(self, bgr: np.ndarray, score_min: float):
        import cv2

        size = int(getattr(cfg, "VISION_DETECT_WIDTH", 320))
        blob_img, scale, pad_x, pad_y = self._letterbox(bgr, size)
        blob = cv2.dnn.blobFromImage(blob_img, 1 / 255.0, (size, size), swapRB=True, crop=False)
        self._net.setInput(blob)
        out = self._net.forward()
        # YOLOv8: (1, 84, N)  or (1, N, 84)
        pred = np.squeeze(out)
        if pred.ndim != 2:
            return []
        if pred.shape[0] < pred.shape[1] and pred.shape[0] <= 84:
            pred = pred.T  # (N, 84)
        boxes = []
        h0, w0 = bgr.shape[:2]
        for row in pred:
            obj = row[4:]
            cls = int(np.argmax(obj))
            if cls != self.PERSON_CLASS:
                continue
            score = float(obj[cls])
            if score < score_min:
                continue
            cx, cy, bw, bh = row[:4]
            x0 = (cx - bw / 2.0 - pad_x) / scale
            y0 = (cy - bh / 2.0 - pad_y) / scale
            x1 = (cx + bw / 2.0 - pad_x) / scale
            y1 = (cy + bh / 2.0 - pad_y) / scale
            x0 = float(np.clip(x0, 0, w0 - 1))
            y0 = float(np.clip(y0, 0, h0 - 1))
            x1 = float(np.clip(x1, 0, w0 - 1))
            y1 = float(np.clip(y1, 0, h0 - 1))
            if x1 - x0 < 4 or y1 - y0 < 8:
                continue
            boxes.append((x0, y0, x1 - x0, y1 - y0, score))
        return boxes

    def _detect_hog(self, bgr: np.ndarray, score_min: float):
        rects, weights = self._hog.detectMultiScale(
            bgr, winStride=(8, 8), padding=(8, 8), scale=1.08
        )
        boxes = []
        for (x, y, w, h), wt in zip(rects, weights.reshape(-1)):
            score = float(1.0 / (1.0 + np.exp(-float(wt))))
            if score >= min(0.35, score_min):
                boxes.append((float(x), float(y), float(w), float(h), score))
        return boxes

    def _loop(self) -> None:
        import cv2

        period = 1.0 / max(1, int(cfg.VISION_FPS))
        det_w = int(cfg.VISION_DETECT_WIDTH)
        alpha = float(cfg.VISION_EMA_ALPHA)
        score_min = float(cfg.VISION_SCORE_MIN)
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
                bgr = np.ascontiguousarray(small[:, :, ::-1])
                td = time.monotonic()
                if self._net is not None:
                    boxes = self._detect_yolo(bgr, score_min)
                else:
                    boxes = self._detect_hog(bgr, score_min)
                self.detect_ms = 0.8 * self.detect_ms + 0.2 * (time.monotonic() - td) * 1000.0
                # boxes are in `small` pixels; map to preview pixels
                sx, sy = fw / float(det_w), fh / float(dh)
                mapped = [(x * sx, y * sy, w * sx, h * sy, sc) for x, y, w, h, sc in boxes]
                self._update_track(mapped, fw, fh, alpha)
            except Exception as exc:
                self.error = str(exc)
                log.warning("person detection failed: %s", exc)
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

    def _update_track(self, boxes, fw: int, fh: int, alpha: float) -> None:
        now = time.monotonic()
        cands = []
        for x, y, w, h, score in boxes:
            if w <= 0 or h <= 0:
                continue
            cands.append(((x + w / 2) / fw, (y + h / 2) / fh, w / fw, h / fh, score, x, y, w, h))

        with self._lock:
            prev = self._track
            chosen = None
            if cands:
                if prev is None or self._want_acquire or prev.age() > float(cfg.VISION_STALE_S) * 3:
                    chosen = min(cands, key=lambda c: (c[0] - 0.5) ** 2 + (c[1] - 0.5) ** 2)
                    self._track_id += 1
                    self._want_acquire = False
                    prev = None
                else:
                    d, chosen = min(
                        (((c[0] - prev.cx) ** 2 + (c[1] - prev.cy) ** 2) ** 0.5, c) for c in cands
                    )
                    if d > max(0.18, 1.5 * prev.w):
                        chosen = None
            if chosen is None:
                return

            cx, cy, w, h, score = chosen[0], chosen[1], chosen[2], chosen[3], chosen[4]
            if prev is not None:
                cx = (1 - alpha) * prev.cx + alpha * cx
                cy = (1 - alpha) * prev.cy + alpha * cy
                w = (1 - alpha) * prev.w + alpha * w
                h = (1 - alpha) * prev.h + alpha * h
            az, el = self._angles(cx, cy, fw / float(fh))
            self._track = PersonTrack(
                cx=cx, cy=cy, w=w, h=h, score=score,
                rel_az_deg=az, rel_el_deg=el, t=now,
                n_faces=len(cands),
                hits=(prev.hits + 1) if prev is not None else 1,
                id=self._track_id,
                box_px=(int((cx - w / 2) * fw), int((cy - h / 2) * fh), int(w * fw), int(h * fh)),
            )


# GUI / older imports
FaceTracker = PersonTracker
