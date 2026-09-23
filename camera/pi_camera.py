"""
Background camera grabber for the live GUI.

Backends (first that works wins):
  1. picamera2   — Raspberry Pi Camera Module 3 on CSI (camera index
                   ``config.CAMERA_INDEX``). Needs the venv to see system
                   site-packages (install.sh handles that).
  2. rpicam-vid  — MJPEG pipe from the libcamera CLI (works even when
                   picamera2 is not importable from the venv). Decoded with
                   Pillow.
  3. OpenCV      — ``cv2.VideoCapture(CAMERA_INDEX)`` for USB webcams / dev PCs.

If no backend opens, ``CameraFeed.available`` is False and the GUI shows a
placeholder instead of crashing.  Frames are delivered as RGB uint8 arrays of
shape (H, W, 3) via ``latest()``; the GUI converts them to a Tk PhotoImage.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from typing import Optional

import numpy as np

import config as cfg

log = logging.getLogger(__name__)


class CameraFeed:
    def __init__(
        self,
        index: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
    ) -> None:
        self.index = int(cfg.CAMERA_INDEX if index is None else index)
        self.width = int(cfg.CAMERA_PREVIEW_WIDTH if width is None else width)
        self.height = int(cfg.CAMERA_PREVIEW_HEIGHT if height is None else height)
        self.fps = int(cfg.CAMERA_FPS if fps is None else fps)
        self.hflip = bool(getattr(cfg, "CAMERA_HFLIP", False))
        self.vflip = bool(getattr(cfg, "CAMERA_VFLIP", False))

        self.backend = "none"
        self.available = False
        self.error: Optional[str] = None
        self.frame_count = 0
        self.measured_fps = 0.0

        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._picam = None
        self._proc: Optional[subprocess.Popen] = None
        self._cv = None

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self) -> bool:
        if self._thread is not None:
            return self.available
        errors = []
        for opener in (self._open_picamera2, self._open_rpicam_vid, self._open_opencv):
            try:
                if opener():
                    self.available = True
                    break
            except Exception as exc:  # pragma: no cover - hardware dependent
                errors.append(f"{opener.__name__[6:]}: {exc}")
        if not self.available:
            self.error = "; ".join(errors) or "no camera backend available"
            log.warning("Camera unavailable — %s", self.error)
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        log.info("Camera ready via %s (%dx%d @ %d fps, index %d)",
                 self.backend, self.width, self.height, self.fps, self.index)
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            if self._picam is not None:
                self._picam.stop()
                self._picam.close()
            if self._proc is not None:
                self._proc.kill()
            if self._cv is not None:
                self._cv.release()
        except Exception:
            pass
        self._picam = self._proc = self._cv = None
        self.available = False

    def latest(self) -> Optional[np.ndarray]:
        """Most recent RGB frame (H, W, 3) or None."""
        with self._lock:
            return self._frame

    # ── backends ───────────────────────────────────────────────────────────
    def _open_picamera2(self) -> bool:
        try:
            from picamera2 import Picamera2  # type: ignore
        except ImportError:
            return False
        try:
            from libcamera import Transform  # type: ignore

            transform = Transform(hflip=int(self.hflip), vflip=int(self.vflip))
        except Exception:
            transform = None

        cam = Picamera2(self.index)
        kwargs = {"main": {"size": (self.width, self.height), "format": "RGB888"}}
        if transform is not None:
            kwargs["transform"] = transform
        config = cam.create_video_configuration(**kwargs)
        cam.configure(config)
        try:
            cam.set_controls({"FrameRate": float(self.fps)})
        except Exception:
            pass
        cam.start()
        self._picam = cam
        self.backend = "picamera2"
        return True

    def _open_rpicam_vid(self) -> bool:
        exe = shutil.which("rpicam-vid") or shutil.which("libcamera-vid")
        if not exe:
            return False
        try:
            from PIL import Image  # noqa: F401  (needed for MJPEG decode)
        except ImportError:
            log.warning("rpicam-vid found but Pillow missing — pip install pillow")
            return False
        cmd = [
            exe, "--camera", str(self.index), "-t", "0", "-n",
            "--codec", "mjpeg", "--width", str(self.width), "--height", str(self.height),
            "--framerate", str(self.fps), "--flush", "-o", "-",
        ]
        if self.hflip:
            cmd.append("--hflip")
        if self.vflip:
            cmd.append("--vflip")
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0
        )
        time.sleep(0.3)
        if self._proc.poll() is not None:
            self._proc = None
            return False
        self.backend = "rpicam-vid"
        return True

    def _open_opencv(self) -> bool:
        try:
            import cv2  # type: ignore
        except ImportError:
            return False
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap.release()
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cv = cap
        self.backend = "opencv"
        return True

    # ── capture loop ───────────────────────────────────────────────────────
    def _loop(self) -> None:
        t_last = time.monotonic()
        n = 0
        period = 1.0 / max(1, self.fps)
        while not self._stop.is_set():
            t0 = time.monotonic()
            frame = None
            try:
                if self._picam is not None:
                    frame = self._picam.capture_array("main")
                    if frame.ndim == 3 and frame.shape[2] == 4:
                        frame = frame[:, :, :3]
                    frame = self._colour_fix(frame)
                elif self._proc is not None:
                    frame = self._read_mjpeg_frame()
                    if frame is not None:
                        frame = self._colour_fix(frame, swap_rb=False)
                elif self._cv is not None:
                    ok, bgr = self._cv.read()
                    if ok:
                        frame = bgr[:, :, ::-1]
                        if self.hflip:
                            frame = frame[:, ::-1]
                        if self.vflip:
                            frame = frame[::-1]
                        frame = self._colour_fix(frame, swap_rb=False)
            except Exception as exc:  # pragma: no cover
                self.error = str(exc)
                log.warning("Camera read failed: %s", exc)
                time.sleep(0.5)
                continue

            if frame is not None:
                with self._lock:
                    self._frame = np.ascontiguousarray(frame)
                self.frame_count += 1
                n += 1
                now = time.monotonic()
                if now - t_last >= 1.0:
                    self.measured_fps = n / (now - t_last)
                    n, t_last = 0, now
            if self._cv is not None or self._picam is not None:
                dt = time.monotonic() - t0
                if dt < period:
                    time.sleep(period - dt)

    @staticmethod
    def _colour_fix(frame: np.ndarray, swap_rb: bool = True) -> np.ndarray:
        """Undo picamera2 BGR-in-RGB888 and/or a photographic-negative look."""
        if frame is None or frame.ndim != 3:
            return frame
        if swap_rb and bool(getattr(cfg, "CAMERA_SWAP_RB", True)):
            frame = frame[:, :, ::-1]
        if bool(getattr(cfg, "CAMERA_INVERT_COLORS", False)):
            frame = 255 - frame
        return frame

    def _read_mjpeg_frame(self) -> Optional[np.ndarray]:
        """Pull one JPEG (FFD8 … FFD9) off the rpicam-vid pipe."""
        import io

        from PIL import Image

        assert self._proc is not None and self._proc.stdout is not None
        buf = bytearray()
        start = -1
        while not self._stop.is_set():
            chunk = self._proc.stdout.read(4096)
            if not chunk:
                return None
            buf += chunk
            if start < 0:
                start = buf.find(b"\xff\xd8")
                if start < 0:
                    del buf[:-1]
                    continue
            end = buf.find(b"\xff\xd9", start + 2)
            if end >= 0:
                jpg = bytes(buf[start : end + 2])
                img = Image.open(io.BytesIO(jpg)).convert("RGB")
                return np.asarray(img)
        return None
