"""
Live control-room GUI for the lecturer-tracking pipeline.

  ┌──────────────────────────────────────────────────────────────────────────┐
  │ RIC · Lecturer Tracker      [LIVE] [SPEECH] [HAT] [CAM]   MIC  SERVO  ⌂  │
  ├───────────────────────────────────┬──────────────────────────────────────┤
  │ Camera Module 3 (live, HUD)       │ Silero VAD                           │
  │                                   │ Sound radar (az / el / confidence)   │
  ├───────────────┬───────────────────┤ Pan / tilt HAT — gauges + sliders    │
  │ Mic meters    │ Array (front view)│   AUTO ⇄ MANUAL, nudge, centre       │
  └───────────────┴───────────────────┴──────────────────────────────────────┘

Interactive:
  • MIC ON/OFF   — mute the array (drops the track, HAT holds position)
  • AUTO/MANUAL  — MANUAL hands the HAT to the sliders / arrow keys /
                   click-on-radar; AUTO gives it back to SRP-PHAT + Kalman
  • CENTRE       — front pose (pan PAN_FRONT_DEG / tilt TILT_FRONT_DEG)
  • CAM          — start / stop the camera preview
  Keys: M mute · A auto/manual · C centre · ←→↑↓ nudge (manual) · Q quit

Run from repo root:
  ./run_gui.sh [--real-servos]
  python vad_srp_phat_pipeline.py
"""
from __future__ import annotations

import logging
import math
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from pipeline.lecturer_tracker import LecturerTracker

try:  # fast frame → Tk conversion
    from PIL import Image, ImageTk  # type: ignore

    _HAVE_PIL = True
except ImportError:  # pragma: no cover
    _HAVE_PIL = False

log = logging.getLogger(__name__)

# ── palette: graphite + signal colours ────────────────────────────────────
BG = "#0f1216"
PANEL = "#181d24"
PANEL_2 = "#1f252e"
LINE = "#2b333f"
TEXT = "#eef2f7"
MUTED = "#8b95a5"
OK = "#2ee59d"
WARN = "#ffb347"
BAD = "#ff5d6c"
ACCENT = "#4cc2ff"
VIOLET = "#a78bfa"
METER_BG = "#0b0e12"

FONT = "Segoe UI"
MONO = "Consolas"

NUDGE_DEG = 2.0


class LiveDashboard:
    def __init__(
        self,
        tracker: Optional[LecturerTracker] = None,
        simulate_servos: Optional[bool] = None,
    ) -> None:
        self.tracker = tracker or LecturerTracker(simulate_servos=simulate_servos)
        self.camera = None
        self._photo = None           # keep a ref or Tk drops the image
        self._cam_img_id = None
        self._syncing_sliders = False
        self._mouse_down = False
        self._blink = 0

        self.root = tk.Tk()
        self.root.title("RIC — Lecturer Tracker")
        self.root.configure(bg=BG)
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w, h = min(1380, sw - 40), min(840, sh - 90)
        self.root.geometry(f"{w}x{h}+{(sw - w) // 2}+{max(0, (sh - h) // 2 - 20)}")
        self.root.minsize(900, 560)

        self._build()
        self._bind_keys()
        self.tracker.start()
        if bool(getattr(cfg, "CAMERA_ENABLED", True)):
            self._camera_toggle(start=True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(cfg.GUI_REFRESH_MS, self._tick)

    # ══════════════════════════════════════════════════════════════════════
    # layout
    # ══════════════════════════════════════════════════════════════════════
    def _build(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG)

        self._build_header()

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=14, pady=(0, 10))
        body.columnconfigure(0, weight=5, uniform="col")
        body.columnconfigure(1, weight=3, uniform="col")
        body.rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(7, 0))

        # left: camera (big) + mics/array row
        left.rowconfigure(0, weight=3)
        left.rowconfigure(1, weight=2)
        left.columnconfigure(0, weight=1)
        cam_card = self._card(left, "CAMERA · Pi Camera Module 3", row=0, col=0)
        self._build_camera(cam_card)
        bottom = tk.Frame(left, bg=BG)
        bottom.grid(row=1, column=0, sticky="nsew", pady=(7, 0))
        bottom.columnconfigure(0, weight=1, uniform="b")
        bottom.columnconfigure(1, weight=1, uniform="b")
        bottom.rowconfigure(0, weight=1)
        self._build_mics(self._card(bottom, "MICROPHONES · INMP441 ×4", row=0, col=0, padx=(0, 7)))
        self._build_array(self._card(bottom, "ARRAY · 6 cm vertical square (front view)", row=0, col=1))

        # right: vad / radar / servo
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=0)
        right.rowconfigure(1, weight=3)
        right.rowconfigure(2, weight=4)
        vad_title = ("VOICE · Silero VAD" if getattr(cfg, "VAD_ENABLED", True)
                     else "SOUND GATE · VAD OFF (level only)")
        self._build_vad(self._card(right, vad_title, row=0, col=0))
        self._build_radar(self._card(right, "SOUND RADAR · SRP-PHAT", row=1, col=0, pady=(7, 0)))
        self._build_servos(self._card(right, "PAN / TILT HAT · Waveshare", row=2, col=0, pady=(7, 0)))

        self._build_footer()

    def _card(self, parent, title, row, col, padx=(0, 0), pady=(0, 0)) -> tk.Frame:
        outer = tk.Frame(parent, bg=LINE)
        outer.grid(row=row, column=col, sticky="nsew", padx=padx, pady=pady)
        inner = tk.Frame(outer, bg=PANEL)
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        head = tk.Frame(inner, bg=PANEL)
        head.pack(fill="x", padx=12, pady=(8, 2))
        tk.Label(head, text=title, bg=PANEL, fg=MUTED, font=(FONT, 9, "bold"), anchor="w").pack(side="left")
        body = tk.Frame(inner, bg=PANEL)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        body.head = head  # type: ignore[attr-defined]
        return body

    # ── header ─────────────────────────────────────────────────────────────
    def _build_header(self) -> None:
        hdr = tk.Frame(self.root, bg=BG)
        hdr.pack(fill="x", padx=14, pady=(12, 8))

        tk.Label(hdr, text="RIC", bg=BG, fg=ACCENT, font=(FONT, 20, "bold")).pack(side="left")
        tk.Label(hdr, text="  Lecturer Tracker", bg=BG, fg=TEXT, font=(FONT, 16)).pack(side="left")

        pills = tk.Frame(hdr, bg=BG)
        pills.pack(side="left", padx=24)
        self.pill_live = self._pill(pills, "STARTING")
        self.pill_speech = self._pill(pills, "IDLE")
        self.pill_vad = self._pill(pills, "VAD ON" if getattr(cfg, "VAD_ENABLED", True) else "VAD OFF")
        self._set_pill(self.pill_vad, self.pill_vad.cget("text"),
                       OK if getattr(cfg, "VAD_ENABLED", True) else WARN)
        self.pill_hat = self._pill(pills, "HAT ?")
        self.pill_cam = self._pill(pills, "CAM OFF")

        ctrl = tk.Frame(hdr, bg=BG)
        ctrl.pack(side="right")
        self.btn_cam = self._button(ctrl, "CAM  ●", self._camera_toggle, color=PANEL_2)
        self.btn_center = self._button(ctrl, "⌂  CENTRE", self._center, color=PANEL_2)
        self.btn_servo = self._button(ctrl, "SERVO  AUTO", self._servo_toggle, color=ACCENT, fg=BG)
        self.btn_mic = self._button(ctrl, "MIC  ON", self._mic_toggle, color=OK, fg=BG)

    def _pill(self, parent, text) -> tk.Label:
        lbl = tk.Label(parent, text=text, bg=PANEL_2, fg=MUTED, font=(MONO, 10, "bold"), padx=10, pady=3)
        lbl.pack(side="left", padx=4)
        return lbl

    def _button(self, parent, text, cmd, color=PANEL_2, fg=TEXT) -> tk.Button:
        b = tk.Button(
            parent, text=text, command=cmd, bg=color, fg=fg, activebackground=color,
            activeforeground=fg, relief="flat", bd=0, padx=14, pady=6,
            font=(FONT, 10, "bold"), cursor="hand2", highlightthickness=0,
        )
        b.pack(side="right", padx=4)
        return b

    @staticmethod
    def _set_pill(lbl: tk.Label, text: str, color: str, on: bool = True) -> None:
        lbl.configure(text=text, bg=color if on else PANEL_2, fg=BG if on else MUTED)

    # ── camera ─────────────────────────────────────────────────────────────
    def _build_camera(self, body: tk.Frame) -> None:
        self.cam_canvas = tk.Canvas(body, bg=METER_BG, highlightthickness=0, cursor="crosshair")
        self.cam_canvas.pack(fill="both", expand=True)
        self.cam_info = tk.Label(body.head, text="", bg=PANEL, fg=MUTED, font=(MONO, 9))
        self.cam_info.pack(side="right")

    # ── mics ───────────────────────────────────────────────────────────────
    def _build_mics(self, body: tk.Frame) -> None:
        self.mic_canvas = tk.Canvas(body, bg=PANEL, highlightthickness=0)
        self.mic_canvas.pack(fill="both", expand=True)

    def _build_array(self, body: tk.Frame) -> None:
        self.array_canvas = tk.Canvas(body, bg=PANEL, highlightthickness=0)
        self.array_canvas.pack(fill="both", expand=True)

    # ── vad ────────────────────────────────────────────────────────────────
    def _build_vad(self, body: tk.Frame) -> None:
        row = tk.Frame(body, bg=PANEL)
        row.pack(fill="x")
        self.vad_label = tk.Label(row, text="NON_SPEECH" if getattr(cfg, "VAD_ENABLED", True) else "QUIET",
                                  bg=PANEL, fg=MUTED, font=(FONT, 18, "bold"))
        self.vad_label.pack(side="left")
        self.vad_prob = tk.Label(row, text="0%", bg=PANEL, fg=ACCENT, font=(MONO, 22, "bold"))
        self.vad_prob.pack(side="right")
        self.vad_canvas = tk.Canvas(body, height=22, bg=METER_BG, highlightthickness=0)
        self.vad_canvas.pack(fill="x", pady=(6, 0))
        self.vad_meta = tk.Label(body, text="", bg=PANEL, fg=MUTED, font=(MONO, 9), anchor="w")
        self.vad_meta.pack(fill="x", pady=(4, 0))

    # ── radar ──────────────────────────────────────────────────────────────
    def _build_radar(self, body: tk.Frame) -> None:
        self.radar_canvas = tk.Canvas(body, bg=PANEL, highlightthickness=0, cursor="hand2")
        self.radar_canvas.pack(fill="both", expand=True)
        self.radar_canvas.bind("<Button-1>", self._radar_click)
        self.radar_info = tk.Label(body.head, text="", bg=PANEL, fg=MUTED, font=(MONO, 9))
        self.radar_info.pack(side="right")

    # ── servos ─────────────────────────────────────────────────────────────
    def _build_servos(self, body: tk.Frame) -> None:
        self.servo_mode_lbl = tk.Label(body.head, text="AUTO", bg=PANEL, fg=ACCENT, font=(MONO, 9, "bold"))
        self.servo_mode_lbl.pack(side="right")

        self.servo_canvas = tk.Canvas(body, bg=PANEL, highlightthickness=0, height=150)
        self.servo_canvas.pack(fill="both", expand=True)

        grid = tk.Frame(body, bg=PANEL)
        grid.pack(fill="x", pady=(6, 0))
        grid.columnconfigure(1, weight=1)

        self.pan_val = tk.Label(grid, text="—", bg=PANEL, fg=TEXT, font=(MONO, 12, "bold"), width=7, anchor="e")
        self.tilt_val = tk.Label(grid, text="—", bg=PANEL, fg=TEXT, font=(MONO, 12, "bold"), width=7, anchor="e")

        tk.Label(grid, text="PAN", bg=PANEL, fg=MUTED, font=(FONT, 9, "bold")).grid(row=0, column=0, sticky="w")
        self.pan_slider = self._slider(grid, cfg.PAN_MIN_DEG, cfg.PAN_MAX_DEG, ACCENT, self._slider_moved)
        self.pan_slider.grid(row=0, column=1, sticky="ew", padx=8)
        self.pan_val.grid(row=0, column=2)

        tk.Label(grid, text="TILT", bg=PANEL, fg=MUTED, font=(FONT, 9, "bold")).grid(row=1, column=0, sticky="w")
        self.tilt_slider = self._slider(grid, cfg.TILT_MIN_DEG, cfg.TILT_MAX_DEG, OK, self._slider_moved)
        self.tilt_slider.grid(row=1, column=1, sticky="ew", padx=8)
        self.tilt_val.grid(row=1, column=2)

        pad = tk.Frame(body, bg=PANEL)
        pad.pack(fill="x", pady=(8, 0))
        for text, dp, dt in (("◀", -NUDGE_DEG, 0), ("▶", +NUDGE_DEG, 0), ("▲", 0, -NUDGE_DEG), ("▼", 0, +NUDGE_DEG)):
            b = tk.Button(
                pad, text=text, command=lambda a=dp, b_=dt: self._nudge(a, b_), bg=PANEL_2, fg=TEXT,
                activebackground=LINE, activeforeground=TEXT, relief="flat", bd=0, width=4, pady=4,
                font=(FONT, 11, "bold"), cursor="hand2", highlightthickness=0,
            )
            b.pack(side="left", padx=3)
        tk.Label(
            pad,
            text=f"pan {cfg.PAN_MIN_DEG:.0f}–{cfg.PAN_MAX_DEG:.0f}°  front {cfg.PAN_FRONT_DEG:.0f}°   "
                 f"tilt {cfg.TILT_MIN_DEG:.0f}–{cfg.TILT_MAX_DEG:.0f}°  front {cfg.TILT_FRONT_DEG:.0f}°",
            bg=PANEL, fg=MUTED, font=(MONO, 9),
        ).pack(side="right")
        self._set_manual_widgets(False)

    def _slider(self, parent, lo, hi, color, cmd) -> tk.Scale:
        # No `command=`: Tk fires it asynchronously even for programmatic .set(),
        # which races with the periodic sync. Only real mouse input drives the HAT.
        s = tk.Scale(
            parent, from_=lo, to=hi, orient="horizontal", resolution=0.5, showvalue=False,
            bg=PANEL, fg=TEXT, troughcolor=METER_BG, activebackground=color, highlightthickness=0,
            bd=0, sliderrelief="flat", sliderlength=22,
        )
        s.bind("<B1-Motion>", lambda e: self.root.after_idle(cmd), add="+")
        s.bind("<ButtonRelease-1>", lambda e: self.root.after_idle(cmd), add="+")
        return s

    def _build_footer(self) -> None:
        foot = tk.Frame(self.root, bg=BG)
        foot.pack(fill="x", padx=14, pady=(0, 10))
        self.foot_lbl = tk.Label(
            foot,
            text=(f"audio device={self.tracker.audio_device}  ·  mount={cfg.ARRAY_MOUNT}  ·  "
                  f"M0 TL  M1 TR  M2 BR  M3 BL  ·  keys: M mic · A auto/manual · C centre · ←→↑↓ nudge · Q quit"),
            bg=BG, fg=MUTED, font=(MONO, 9), anchor="w",
        )
        self.foot_lbl.pack(fill="x")

    # ══════════════════════════════════════════════════════════════════════
    # interaction
    # ══════════════════════════════════════════════════════════════════════
    def _bind_keys(self) -> None:
        r = self.root
        r.bind("<KeyPress-m>", lambda e: self._mic_toggle())
        r.bind("<KeyPress-a>", lambda e: self._servo_toggle())
        r.bind("<KeyPress-c>", lambda e: self._center())
        r.bind("<KeyPress-q>", lambda e: self._on_close())
        r.bind("<Left>", lambda e: self._nudge(-NUDGE_DEG, 0))
        r.bind("<Right>", lambda e: self._nudge(+NUDGE_DEG, 0))
        r.bind("<Up>", lambda e: self._nudge(0, -NUDGE_DEG))
        r.bind("<Down>", lambda e: self._nudge(0, +NUDGE_DEG))

    def _mic_toggle(self) -> None:
        on = not self.tracker.mic_enabled
        self.tracker.set_mic_enabled(on)
        self.btn_mic.configure(
            text="MIC  ON" if on else "MIC  OFF", bg=OK if on else BAD, activebackground=OK if on else BAD
        )

    def _servo_toggle(self) -> None:
        manual = not self.tracker.servo_manual
        self.tracker.set_servo_manual(manual)
        self._set_manual_widgets(manual)

    def _set_manual_widgets(self, manual: bool) -> None:
        color = WARN if manual else ACCENT
        self.btn_servo.configure(
            text="SERVO  MANUAL" if manual else "SERVO  AUTO", bg=color, activebackground=color
        )
        self.servo_mode_lbl.configure(text="MANUAL — sliders / arrows / click radar" if manual else "AUTO — SRP-PHAT + Kalman",
                                      fg=color)
        state = "normal" if manual else "disabled"
        self.pan_slider.configure(state=state)
        self.tilt_slider.configure(state=state)

    def _slider_moved(self, _value=None) -> None:
        if not self.tracker.servo_manual:
            return
        self.tracker.manual_servo(float(self.pan_slider.get()), float(self.tilt_slider.get()))

    def _nudge(self, d_pan: float, d_tilt: float) -> None:
        if not self.tracker.servo_manual:
            return
        s = self.tracker.servos
        self.tracker.manual_servo(s.pan_deg + d_pan, s.tilt_deg + d_tilt)

    def _center(self) -> None:
        self.tracker.servo_center()

    def _radar_click(self, event) -> None:
        """In MANUAL mode, click a bearing on the radar to aim the HAT there."""
        if not self.tracker.servo_manual:
            return
        geo = getattr(self, "_radar_geo", None)
        if not geo:
            return
        cx, cy, radius = geo
        dx, dy = event.x - cx, cy - event.y
        if dy <= 0:
            return
        az = math.degrees(math.atan2(dx, dy))
        az = max(cfg.AZIMUTH_MIN_DEG, min(cfg.AZIMUTH_MAX_DEG, az))
        pan, _ = self.tracker.servos.map_angles(az, 0.0)
        self.tracker.manual_servo(pan, self.tracker.servos.tilt_deg)

    def _camera_toggle(self, start: Optional[bool] = None) -> None:
        if start is None:
            start = self.camera is None
        if start and self.camera is None:
            try:
                from camera.pi_camera import CameraFeed

                cam = CameraFeed()
                if cam.start():
                    self.camera = cam
                else:
                    self.cam_info.configure(text=f"no camera: {cam.error}", fg=BAD)
            except Exception as exc:  # pragma: no cover
                log.warning("camera init failed: %s", exc)
                self.cam_info.configure(text=f"camera error: {exc}", fg=BAD)
        elif not start and self.camera is not None:
            self.camera.stop()
            self.camera = None
            self._photo = None
        on = self.camera is not None
        self.btn_cam.configure(bg=VIOLET if on else PANEL_2, fg=BG if on else TEXT,
                               activebackground=VIOLET if on else PANEL_2)

    # ══════════════════════════════════════════════════════════════════════
    # drawing
    # ══════════════════════════════════════════════════════════════════════
    def _draw_camera(self, s: dict) -> None:
        c = self.cam_canvas
        w = max(c.winfo_width(), 64)
        h = max(c.winfo_height(), 36)
        c.delete("hud")
        frame = self.camera.latest() if self.camera is not None else None

        if frame is None:
            c.delete("img")
            self._photo = None
            c.create_rectangle(0, 0, w, h, fill=METER_BG, outline="", tags="hud")
            msg = "camera off" if self.camera is None else "waiting for frames…"
            c.create_text(w / 2, h / 2, text=msg, fill=MUTED, font=(FONT, 14), tags="hud")
            if self.camera is None:
                self._set_pill(self.pill_cam, "CAM OFF", PANEL_2, on=False)
        else:
            fh, fw = frame.shape[:2]
            scale = min(w / fw, h / fh)
            tw, th = max(1, int(fw * scale)), max(1, int(fh * scale))
            self._photo = self._to_photo(frame, tw, th)
            if self._cam_img_id is None or not c.find_withtag("img"):
                self._cam_img_id = c.create_image(w / 2, h / 2, image=self._photo, tags="img")
            else:
                c.itemconfigure(self._cam_img_id, image=self._photo)
                c.coords(self._cam_img_id, w / 2, h / 2)
            self._set_pill(self.pill_cam, f"CAM {self.camera.backend.upper()}", VIOLET)
            self.cam_info.configure(
                text=f"{fw}×{fh}  {self.camera.measured_fps:4.1f} fps", fg=MUTED
            )

        # HUD — crosshair, bearing ticker, speech ring
        cx, cy = w / 2, h / 2
        c.create_line(cx - 22, cy, cx - 8, cy, fill=ACCENT, width=2, tags="hud")
        c.create_line(cx + 8, cy, cx + 22, cy, fill=ACCENT, width=2, tags="hud")
        c.create_line(cx, cy - 22, cx, cy - 8, fill=ACCENT, width=2, tags="hud")
        c.create_line(cx, cy + 8, cx, cy + 22, fill=ACCENT, width=2, tags="hud")
        for x0, y0, x1, y1 in ((10, 10, 40, 10), (10, 10, 10, 40), (w - 40, 10, w - 10, 10), (w - 10, 10, w - 10, 40),
                               (10, h - 10, 40, h - 10), (10, h - 40, 10, h - 10), (w - 40, h - 10, w - 10, h - 10),
                               (w - 10, h - 40, w - 10, h - 10)):
            c.create_line(x0, y0, x1, y1, fill=LINE, width=2, tags="hud")

        speech = bool(s.get("speech"))
        if speech:
            self._blink = (self._blink + 1) % 12
            r = 7 if self._blink < 6 else 5
            c.create_oval(w - 34 - r, 22 - r, w - 34 + r, 22 + r, fill=BAD, outline="", tags="hud")
            c.create_text(w - 48, 22, text="TRACK", fill=BAD, anchor="e", font=(MONO, 10, "bold"), tags="hud")

        # where the sound is relative to the camera's current pan (HFOV≈66° for CM3 wide-ish)
        az = float(s.get("azimuth_deg") or 0.0)
        pan = float(s.get("pan_deg") or cfg.PAN_FRONT_DEG)
        pan_az = self._pan_to_az(pan)
        rel = az - pan_az
        if speech and abs(rel) < 45:
            px = cx + (rel / 33.0) * (w / 2)
            px = max(14, min(w - 14, px))
            c.create_polygon(px, h - 28, px - 9, h - 12, px + 9, h - 12, fill=OK, outline="", tags="hud")
        mode = "MANUAL" if s.get("servo_manual") else "AUTO"
        hud = f"pan {pan:5.1f}°  tilt {float(s.get('tilt_deg') or 0):5.1f}°  ·  {mode}"
        c.create_rectangle(12, h - 30, 12 + 8 * len(hud) + 10, h - 10, fill=BG, outline="", tags="hud")
        c.create_text(17, h - 20, text=hud, fill=TEXT, anchor="w", font=(MONO, 10), tags="hud")
        if not s.get("mic_enabled", True):
            c.create_text(cx, 24, text="MIC MUTED", fill=BAD, font=(FONT, 12, "bold"), tags="hud")

    @staticmethod
    def _pan_to_az(pan: float) -> float:
        """Inverse of ServoController.map_angles pan branch (for HUD only)."""
        f = float(cfg.PAN_FRONT_DEG)
        if pan >= f:
            return (pan - f) / max(1e-6, cfg.PAN_MAX_DEG - f) * cfg.AZIMUTH_MAX_DEG
        return (pan - f) / max(1e-6, f - cfg.PAN_MIN_DEG) * abs(cfg.AZIMUTH_MIN_DEG)

    def _to_photo(self, frame: np.ndarray, tw: int, th: int):
        if _HAVE_PIL:
            img = Image.fromarray(frame)
            if (img.width, img.height) != (tw, th):
                img = img.resize((tw, th), Image.BILINEAR)
            return ImageTk.PhotoImage(img)
        # Pillow-less fallback: nearest-neighbour subsample + PPM
        fh, fw = frame.shape[:2]
        ys = (np.arange(th) * fh / th).astype(int)
        xs = (np.arange(tw) * fw / tw).astype(int)
        small = frame[ys][:, xs]
        header = f"P6 {tw} {th} 255 ".encode()
        return tk.PhotoImage(data=header + small.astype(np.uint8).tobytes())

    def _draw_mic_meters(self, rms: list[float], peak: list[float], muted: bool) -> None:
        c = self.mic_canvas
        c.delete("all")
        w = max(c.winfo_width(), 100)
        h = max(c.winfo_height(), 80)
        n = cfg.NUM_MICS
        gap = 14
        bar_w = (w - gap * (n + 1)) / n
        y0, y1 = 24, h - 26
        for i in range(n):
            x0 = gap + i * (bar_w + gap)
            x1 = x0 + bar_w
            c.create_rectangle(x0, y0, x1, y1, fill=METER_BG, outline=LINE)
            # segment ticks
            for k in range(1, 10):
                yy = y1 - (y1 - y0) * k / 10
                c.create_line(x0 + 2, yy, x1 - 2, yy, fill=PANEL, width=2)
            r = float(rms[i]) if i < len(rms) else 0.0
            p = float(peak[i]) if i < len(peak) else 0.0
            level = min(1.0, r / 0.12)
            fill_h = (y1 - y0) * level
            color = OK if level < 0.7 else (WARN if level < 0.9 else BAD)
            if muted:
                color = LINE
            if fill_h > 0:
                c.create_rectangle(x0 + 3, y1 - fill_h, x1 - 3, y1, fill=color, outline="")
            pk = min(1.0, p / 0.35)
            py = y1 - (y1 - y0) * pk
            c.create_line(x0 + 3, py, x1 - 3, py, fill=TEXT if not muted else MUTED, width=2)
            c.create_text((x0 + x1) / 2, 11, text=cfg.MIC_LABELS[i], fill=TEXT, font=(FONT, 10, "bold"))
            c.create_text((x0 + x1) / 2, h - 12, text=f"{r:.3f}", fill=MUTED, font=(MONO, 9))
        if muted:
            c.create_text(w / 2, (y0 + y1) / 2, text="MUTED", fill=BAD, font=(FONT, 16, "bold"))

    def _draw_array(self, rms: list[float], az: float, el: float, locked: bool) -> None:
        c = self.array_canvas
        c.delete("all")
        w = max(c.winfo_width(), 100)
        h = max(c.winfo_height(), 100)
        cx, cy = w / 2, h / 2 + 6
        side = min(w, h) * 0.5
        corners = {
            0: (cx - side / 2, cy - side / 2),
            1: (cx + side / 2, cy - side / 2),
            2: (cx + side / 2, cy + side / 2),
            3: (cx - side / 2, cy + side / 2),
        }
        c.create_rectangle(corners[0][0], corners[0][1], corners[2][0], corners[2][1], outline=LINE, width=2, dash=(4, 3))
        c.create_text(cx, 10, text="looking at the array", fill=MUTED, font=(FONT, 9))
        for i, (x, y) in corners.items():
            r = float(rms[i]) if i < len(rms) else 0.0
            rad = 10 + min(16.0, r * 90)
            glow = OK if r > 0.01 else LINE
            c.create_oval(x - rad - 4, y - rad - 4, x + rad + 4, y + rad + 4, outline=glow, width=1)
            c.create_oval(x - rad, y - rad, x + rad, y + rad, fill=glow, outline="")
            c.create_text(x, y, text=cfg.MIC_LABELS[i], fill=BG, font=(FONT, 9, "bold"))
        # Only draw when a bearing is locked — never animate from coasting Kalman noise
        if locked:
            length = side * 0.45
            dx = math.sin(math.radians(az)) * length
            dy = -math.sin(math.radians(el)) * length
            c.create_line(cx, cy, cx + dx, cy + dy, fill=ACCENT, width=3, arrow=tk.LAST, arrowshape=(12, 14, 5))
        c.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill=TEXT, outline="")

    def _draw_vad(self, prob: float, speech: bool, muted: bool) -> None:
        c = self.vad_canvas
        c.delete("all")
        w = max(c.winfo_width(), 10)
        h = max(c.winfo_height(), 10)
        c.create_rectangle(0, 0, w, h, fill=METER_BG, outline="")
        fill = max(0.0, min(1.0, prob)) * w
        color = LINE if muted else (OK if speech else ACCENT)
        c.create_rectangle(0, 0, fill, h, fill=color, outline="")
        if getattr(cfg, "VAD_ENABLED", True):
            tx = cfg.VAD_THRESHOLD * w
            c.create_line(tx, 0, tx, h, fill=WARN, width=2)
            c.create_text(tx + 4, h / 2, text=f"thr {cfg.VAD_THRESHOLD:.2f}", fill=WARN, anchor="w", font=(MONO, 8))
        else:
            # level mode: bar = peak / gate, so the gate is always at 100 %
            c.create_text(w - 4, h / 2, text="gate →", fill=WARN, anchor="e", font=(MONO, 8))

    def _draw_radar(self, az: float, el: float, conf: float, locked: bool, pan: float) -> None:
        c = self.radar_canvas
        c.delete("all")
        w = max(c.winfo_width(), 120)
        h = max(c.winfo_height(), 120)
        el_w = 34                              # elevation side gauge
        cx = (w - el_w) / 2
        radius = min((w - el_w) / 2 - 16, h - 44)
        cy = h - 14
        self._radar_geo = (cx, cy, radius)

        # sweep rings + spokes
        for k in (0.33, 0.66, 1.0):
            r = radius * k
            c.create_arc(cx - r, cy - r, cx + r, cy + r, start=0, extent=180, style=tk.ARC, outline=LINE, width=1)
        for a in range(-90, 91, 30):
            x = cx + radius * math.sin(math.radians(a))
            y = cy - radius * math.cos(math.radians(a))
            c.create_line(cx, cy, x, y, fill=LINE, width=1)
            c.create_text(cx + (radius + 10) * math.sin(math.radians(a)), cy - (radius + 10) * math.cos(math.radians(a)),
                          text=f"{a:+d}" if a else "0", fill=MUTED, font=(MONO, 8))

        # camera aim (pan mapped back to az)
        aim_az = self._pan_to_az(pan)
        ax = cx + radius * math.sin(math.radians(aim_az))
        ay = cy - radius * math.cos(math.radians(aim_az))
        c.create_line(cx, cy, ax, ay, fill=VIOLET, width=2, dash=(6, 4))
        c.create_text(ax, ay - 10, text="cam", fill=VIOLET, font=(MONO, 8))

        # source blip — only when a bearing is locked (not free-spinning noise)
        if locked:
            d = radius * (0.35 + 0.6 * max(0.0, min(1.0, conf)))
            sx = cx + d * math.sin(math.radians(az))
            sy = cy - d * math.cos(math.radians(az))
            glow = 10 + 14 * max(conf, 0.3)
            c.create_oval(sx - glow, sy - glow, sx + glow, sy + glow, outline=OK, width=1)
            c.create_oval(sx - 6, sy - 6, sx + 6, sy + 6, fill=OK, outline="")
            c.create_line(cx, cy, sx, sy, fill=OK, width=2)
            c.create_text(sx, sy - glow - 8, text=f"{az:+.0f}°", fill=OK, font=(MONO, 9, "bold"))
        c.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, fill=TEXT, outline="")

        # elevation gauge (right)
        gx0 = w - el_w + 6
        gy0, gy1 = 14, h - 14
        c.create_rectangle(gx0, gy0, gx0 + 12, gy1, fill=METER_BG, outline=LINE)
        el_n = (el - cfg.ELEVATION_MIN_DEG) / max(1e-6, cfg.ELEVATION_MAX_DEG - cfg.ELEVATION_MIN_DEG)
        ey = gy1 - max(0.0, min(1.0, el_n)) * (gy1 - gy0)
        zero_y = gy1 - (0 - cfg.ELEVATION_MIN_DEG) / max(1e-6, cfg.ELEVATION_MAX_DEG - cfg.ELEVATION_MIN_DEG) * (gy1 - gy0)
        c.create_line(gx0 - 3, zero_y, gx0 + 15, zero_y, fill=MUTED)
        c.create_polygon(gx0 + 12, ey, gx0 + 22, ey - 6, gx0 + 22, ey + 6, fill=OK if locked else MUTED, outline="")
        c.create_text(gx0 + 6, gy0 - 8, text="el", fill=MUTED, font=(MONO, 8))

        self.radar_info.configure(
            text=f"az {az:+6.1f}°  el {el:+5.1f}°  conf {conf:.2f}" + ("  LOCK" if locked else ""),
            fg=OK if locked and conf >= cfg.CONFIDENCE_THRESHOLD else MUTED,
        )

    def _draw_servos(self, pan: float, tilt: float, manual: bool, sim: bool) -> None:
        c = self.servo_canvas
        c.delete("all")
        w = max(c.winfo_width(), 200)
        h = max(c.winfo_height(), 100)
        needle = WARN if manual else ACCENT

        # pan arc gauge (left half)
        cx, cy = w * 0.3, h - 18
        r = min(w * 0.28, h - 34)
        lo, hi = cfg.PAN_MIN_DEG, cfg.PAN_MAX_DEG
        # servo 0° = left, 180° = right → canvas angle
        def pan_xy(p, rr):
            a = math.radians(180.0 - p)  # 0 → left (180° on canvas), 90 → up
            return cx + rr * math.cos(a), cy - rr * math.sin(a)

        c.create_arc(cx - r, cy - r, cx + r, cy + r, start=0, extent=180, style=tk.ARC, outline=LINE, width=8)
        c.create_arc(cx - r, cy - r, cx + r, cy + r, start=180 - hi, extent=hi - lo, style=tk.ARC, outline=PANEL_2, width=8)
        # ribbon-limit zone 150→180 in red
        if hi < 180:
            c.create_arc(cx - r, cy - r, cx + r, cy + r, start=180 - 180, extent=180 - hi, style=tk.ARC, outline=BAD, width=8)
        for p in (lo, cfg.PAN_FRONT_DEG, hi):
            x, y = pan_xy(p, r + 14)
            c.create_text(x, y, text=f"{p:.0f}", fill=MUTED, font=(MONO, 8))
        nx, ny = pan_xy(pan, r - 6)
        c.create_line(cx, cy, nx, ny, fill=needle, width=3)
        c.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, fill=TEXT, outline="")
        c.create_text(cx, cy - r * 0.45, text=f"PAN {pan:.1f}°", fill=TEXT, font=(MONO, 11, "bold"))

        # tilt vertical gauge (right half): 80 top … 180 bottom
        gx = w * 0.7
        gy0, gy1 = 16, h - 16
        tlo, thi = cfg.TILT_MIN_DEG, cfg.TILT_MAX_DEG
        c.create_rectangle(gx - 7, gy0, gx + 7, gy1, fill=METER_BG, outline=LINE)
        t_n = (tilt - tlo) / max(1e-6, thi - tlo)
        ty = gy0 + max(0.0, min(1.0, t_n)) * (gy1 - gy0)
        f_n = (cfg.TILT_FRONT_DEG - tlo) / max(1e-6, thi - tlo)
        fy = gy0 + f_n * (gy1 - gy0)
        c.create_line(gx - 14, fy, gx + 14, fy, fill=MUTED, dash=(3, 2))
        c.create_text(gx + 20, fy, text=f"{cfg.TILT_FRONT_DEG:.0f} front", fill=MUTED, anchor="w", font=(MONO, 8))
        c.create_text(gx + 20, gy0, text=f"{tlo:.0f} up", fill=MUTED, anchor="w", font=(MONO, 8))
        c.create_text(gx + 20, gy1, text=f"{thi:.0f} down", fill=MUTED, anchor="w", font=(MONO, 8))
        c.create_rectangle(gx - 6, gy0 + 1, gx + 6, ty, fill=PANEL_2, outline="")
        c.create_polygon(gx - 16, ty, gx - 6, ty - 6, gx - 6, ty + 6, fill=needle, outline="")
        c.create_line(gx - 6, ty, gx + 6, ty, fill=needle, width=3)
        c.create_text(gx - 60, (gy0 + gy1) / 2, text=f"TILT\n{tilt:.1f}°", fill=TEXT, font=(MONO, 11, "bold"), justify="center")

        c.create_text(w - 6, 8, text="SIM" if sim else "HAT LIVE", fill=WARN if sim else OK, anchor="ne", font=(MONO, 9, "bold"))
        if not bool(getattr(cfg, "TRACK_TILT", True)):
            c.create_text(gx, gy1 + 1, text="TILT LOCKED (pan-only)", fill=WARN, anchor="n", font=(MONO, 8, "bold"))

    # ══════════════════════════════════════════════════════════════════════
    # update loop
    # ══════════════════════════════════════════════════════════════════════
    def _tick(self) -> None:
        s = self.tracker.snapshot()
        rms = list(s.get("mic_rms") or [0.0] * cfg.NUM_MICS)
        peak = list(s.get("mic_peak") or [0.0] * cfg.NUM_MICS)
        speech = bool(s.get("speech"))
        prob = float(s.get("vad_probability") or 0.0)
        az = float(s.get("azimuth_deg") or 0.0)
        el = float(s.get("elevation_deg") or 0.0)
        conf = float(s.get("confidence") or 0.0)
        pan = float(s.get("pan_deg") or cfg.PAN_FRONT_DEG)
        tilt = float(s.get("tilt_deg") or cfg.TILT_FRONT_DEG)
        sim = bool(s.get("servo_simulate", True))
        muted = not bool(s.get("mic_enabled", True))
        manual = bool(s.get("servo_manual", False))
        locked = bool(s.get("bearing_locked", False)) and not muted

        # header pills
        self._set_pill(self.pill_live, "LIVE" if self.tracker.running else "STOPPED",
                       OK if self.tracker.running else BAD)
        vad_on = bool(getattr(cfg, "VAD_ENABLED", True))
        active_txt, idle_txt = ("SPEECH", "NON_SPEECH") if vad_on else ("SOUND", "QUIET")
        if muted:
            self._set_pill(self.pill_speech, "MUTED", BAD)
        else:
            self._set_pill(self.pill_speech, active_txt if speech else "IDLE", OK, on=speech)
        self._set_pill(self.pill_hat, "HAT SIM" if sim else "HAT LIVE", WARN if sim else OK)

        # gate status (Silero speech, or plain level when VAD is off)
        self.vad_label.configure(text="MUTED" if muted else (active_txt if speech else idle_txt),
                                 fg=BAD if muted else (OK if speech else MUTED))
        self.vad_prob.configure(text=f"{prob * 100:.0f}%")
        if vad_on:
            gate = (f"Silero thr {cfg.VAD_THRESHOLD:.2f}   "
                    f"AGC={'on' if cfg.VAD_NORMALIZE else 'off'}")
        else:
            gate = (f"SILERO VAD OFF · gate ≥{float(s.get('vad_gain') or 0):.4f} "
                    f"(floor {float(s.get('noise_floor') or 0):.4f})")
        drops = int(s.get("audio_drops") or 0)
        buf = f"   q{int(s.get('audio_queue') or 0)}" + (f" drop{drops}" if drops else "")
        self.vad_meta.configure(
            text=f"raw peak {float(s.get('vad_peak') or 0):.4f}   {gate}   "
                 f"on≥{cfg.VAD_SPEECH_ON_CHUNKS} off≥{cfg.VAD_SPEECH_OFF_CHUNKS}{buf}"
        )

        # servo readouts + sliders (sync without triggering command)
        self.pan_val.configure(text=f"{pan:.1f}°")
        self.tilt_val.configure(text=f"{tilt:.1f}°")
        if not manual or not self._slider_dragging():
            self._syncing_sliders = True
            try:
                self.pan_slider.set(pan)
                self.tilt_slider.set(tilt)
            finally:
                self._syncing_sliders = False

        self._draw_camera(s)
        self._draw_mic_meters(rms, peak, muted)
        self._draw_array(rms, az, el, locked)
        self._draw_vad(prob, speech, muted)
        self._draw_radar(az, el, conf, locked, pan)
        self._draw_servos(pan, tilt, manual, sim)

        self.root.after(cfg.GUI_REFRESH_MS, self._tick)

    def _slider_dragging(self) -> bool:
        """True while the mouse button is held down over either slider."""
        if not self._mouse_down:
            return False
        try:
            widget = self.root.winfo_containing(*self.root.winfo_pointerxy())
        except Exception:
            return False
        return widget in (self.pan_slider, self.tilt_slider)

    def _on_close(self) -> None:
        if self.camera is not None:
            self.camera.stop()
        self.tracker.stop()
        self.root.destroy()

    def run(self) -> None:
        self.root.bind_all("<ButtonPress-1>", lambda e: setattr(self, "_mouse_down", True), add="+")
        self.root.bind_all("<ButtonRelease-1>", lambda e: setattr(self, "_mouse_down", False), add="+")
        self.root.mainloop()


def main(simulate_servos: Optional[bool] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    LiveDashboard(simulate_servos=simulate_servos).run()


if __name__ == "__main__":
    main()
