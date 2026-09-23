"""
Live GUI for the lecturer-tracking pipeline.

Shows:
  • per-mic RMS / peak meters (M0–M3)
  • vertical 6 cm array diagram with activity
  • Silero VAD speech probability
  • SRP-PHAT azimuth / elevation + confidence
  • pan / tilt HAT commanded angles

Run from repo root:
  ./run_gui.sh
  python vad_srp_phat_pipeline.py
  python vad_srp_phat_pipeline.py --no-gui
"""
from __future__ import annotations

import math
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from pipeline.lecturer_tracker import LecturerTracker


# Palette — industrial slate / amber (not purple / cream AI defaults)
BG = "#1c2128"
PANEL = "#252b33"
LINE = "#3d4654"
TEXT = "#e8ecf1"
MUTED = "#9aa3b2"
OK = "#3ecf8e"
WARN = "#e6a23c"
BAD = "#e35d6a"
ACCENT = "#4aa3df"
METER_BG = "#14181e"


class LiveDashboard:
    def __init__(
        self,
        tracker: Optional[LecturerTracker] = None,
        simulate_servos: Optional[bool] = None,
    ) -> None:
        self.tracker = tracker or LecturerTracker(simulate_servos=simulate_servos)
        self.root = tk.Tk()
        self.root.title("RIC — Live Mic Array + Pan/Tilt")
        self.root.configure(bg=BG)
        self.root.geometry("1100x720")
        self.root.minsize(960, 640)

        self._build()
        self.tracker.start()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(cfg.GUI_REFRESH_MS, self._tick)

    # ── layout ─────────────────────────────────────────────────────────────
    def _build(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=PANEL)
        style.configure("TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 11))
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Segoe UI Semibold", 18))
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("Big.TLabel", background=PANEL, foreground=TEXT, font=("Consolas", 22, "bold"))
        style.configure("Value.TLabel", background=PANEL, foreground=ACCENT, font=("Consolas", 16))

        header = ttk.Frame(self.root, style="TFrame")
        header.pack(fill="x", padx=16, pady=(14, 8))
        ttk.Label(header, text="Lecturer Tracker", style="Title.TLabel").pack(side="left")
        self.status_var = tk.StringVar(value="starting…")
        ttk.Label(header, textvariable=self.status_var, style="Title.TLabel").pack(side="right")

        body = ttk.Frame(self.root, style="TFrame")
        body.pack(fill="both", expand=True, padx=16, pady=8)

        left = ttk.Frame(body, style="TFrame")
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        right = ttk.Frame(body, style="TFrame")
        right.pack(side="right", fill="both", expand=True, padx=(8, 0))

        self._card_mics(left)
        self._card_array(left)
        self._card_vad(right)
        self._card_doa(right)
        self._card_servos(right)

        foot = ttk.Frame(self.root, style="TFrame")
        foot.pack(fill="x", padx=16, pady=(4, 12))
        ttk.Label(
            foot,
            text=(
                f"Vertical {cfg.MIC_SIDE_M*100:.0f}×{cfg.MIC_SIDE_M*100:.0f} cm square  ·  "
                f"mount={cfg.ARRAY_MOUNT}  ·  device={self.tracker.audio_device}"
            ),
            style="Title.TLabel",
        ).pack(anchor="w")

    def _panel(self, parent: ttk.Frame, title: str) -> ttk.Frame:
        wrap = ttk.Frame(parent, style="Card.TFrame")
        wrap.pack(fill="both", expand=True, pady=6)
        inner = tk.Frame(wrap, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        tk.Label(
            inner,
            text=title,
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI Semibold", 11),
            anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 4))
        body = tk.Frame(inner, bg=PANEL)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        return body

    def _card_mics(self, parent: ttk.Frame) -> None:
        body = self._panel(parent, "Microphone levels (INMP441)")
        self.mic_canvas = tk.Canvas(body, height=220, bg=PANEL, highlightthickness=0)
        self.mic_canvas.pack(fill="both", expand=True)
        self._mic_bars: list[dict] = []
        labels = cfg.MIC_LABELS
        for i, name in enumerate(labels):
            self._mic_bars.append({"name": name, "i": i})

    def _card_array(self, parent: ttk.Frame) -> None:
        body = self._panel(parent, "Array — vertical plane (front view)")
        self.array_canvas = tk.Canvas(body, height=240, bg=PANEL, highlightthickness=0)
        self.array_canvas.pack(fill="both", expand=True)

    def _card_vad(self, parent: ttk.Frame) -> None:
        body = self._panel(parent, "Silero VAD")
        self.vad_label = tk.StringVar(value="NON_SPEECH")
        self.vad_prob = tk.StringVar(value="0%")
        row = tk.Frame(body, bg=PANEL)
        row.pack(fill="x")
        tk.Label(row, textvariable=self.vad_label, bg=PANEL, fg=TEXT, font=("Segoe UI Semibold", 20)).pack(
            side="left"
        )
        tk.Label(row, textvariable=self.vad_prob, bg=PANEL, fg=ACCENT, font=("Consolas", 20, "bold")).pack(
            side="right"
        )
        self.vad_canvas = tk.Canvas(body, height=28, bg=METER_BG, highlightthickness=0)
        self.vad_canvas.pack(fill="x", pady=(10, 0))
        self.vad_meta = tk.StringVar(value="peak=0  gain=1.0x")
        tk.Label(body, textvariable=self.vad_meta, bg=PANEL, fg=MUTED, font=("Consolas", 10)).pack(
            anchor="w", pady=(8, 0)
        )

    def _card_doa(self, parent: ttk.Frame) -> None:
        body = self._panel(parent, "SRP-PHAT direction")
        grid = tk.Frame(body, bg=PANEL)
        grid.pack(fill="x")
        self.az_var = tk.StringVar(value="0.0°")
        self.el_var = tk.StringVar(value="0.0°")
        self.conf_var = tk.StringVar(value="0.00")
        for col, (title, var) in enumerate(
            (("Azimuth", self.az_var), ("Elevation", self.el_var), ("Confidence", self.conf_var))
        ):
            cell = tk.Frame(grid, bg=PANEL)
            cell.grid(row=0, column=col, sticky="nsew", padx=6)
            tk.Label(cell, text=title, bg=PANEL, fg=MUTED, font=("Segoe UI", 10)).pack(anchor="w")
            tk.Label(cell, textvariable=var, bg=PANEL, fg=TEXT, font=("Consolas", 18, "bold")).pack(
                anchor="w"
            )
            grid.columnconfigure(col, weight=1)

        self.doa_canvas = tk.Canvas(body, height=180, bg=PANEL, highlightthickness=0)
        self.doa_canvas.pack(fill="both", expand=True, pady=(8, 0))

    def _card_servos(self, parent: ttk.Frame) -> None:
        body = self._panel(parent, "Pan / tilt HAT")
        self.pan_var = tk.StringVar(value="—")
        self.tilt_var = tk.StringVar(value="—")
        self.servo_mode = tk.StringVar(value="—")
        row = tk.Frame(body, bg=PANEL)
        row.pack(fill="x")
        for title, var in (("Pan", self.pan_var), ("Tilt", self.tilt_var), ("Mode", self.servo_mode)):
            cell = tk.Frame(row, bg=PANEL)
            cell.pack(side="left", expand=True, fill="x")
            tk.Label(cell, text=title, bg=PANEL, fg=MUTED).pack(anchor="w")
            tk.Label(cell, textvariable=var, bg=PANEL, fg=TEXT, font=("Consolas", 16, "bold")).pack(
                anchor="w"
            )
        self.servo_canvas = tk.Canvas(body, height=70, bg=PANEL, highlightthickness=0)
        self.servo_canvas.pack(fill="x", pady=(10, 0))

    # ── draw helpers ───────────────────────────────────────────────────────
    def _draw_mic_meters(self, rms: list[float], peak: list[float]) -> None:
        c = self.mic_canvas
        c.delete("all")
        w = max(c.winfo_width(), 100)
        h = max(c.winfo_height(), 100)
        n = cfg.NUM_MICS
        gap = 16
        bar_w = (w - gap * (n + 1)) / n
        for i in range(n):
            x0 = gap + i * (bar_w + gap)
            x1 = x0 + bar_w
            y0, y1 = 28, h - 28
            c.create_rectangle(x0, y0, x1, y1, fill=METER_BG, outline=LINE)
            # map rms ~0..0.3 to fill (soft clip)
            r = float(rms[i]) if i < len(rms) else 0.0
            p = float(peak[i]) if i < len(peak) else 0.0
            level = min(1.0, r / 0.12)
            fill_h = (y1 - y0) * level
            color = OK if level > 0.05 else LINE
            if level > 0.85:
                color = WARN
            c.create_rectangle(x0 + 2, y1 - fill_h, x1 - 2, y1, fill=color, outline="")
            # peak tick
            pk = min(1.0, p / 0.35)
            py = y1 - (y1 - y0) * pk
            c.create_line(x0 + 2, py, x1 - 2, py, fill=TEXT, width=2)
            c.create_text(
                (x0 + x1) / 2,
                14,
                text=cfg.MIC_LABELS[i],
                fill=TEXT,
                font=("Segoe UI Semibold", 12),
            )
            c.create_text(
                (x0 + x1) / 2,
                h - 12,
                text=f"{r:.3f}",
                fill=MUTED,
                font=("Consolas", 10),
            )

    def _draw_array(self, rms: list[float], az: float, el: float, speech: bool) -> None:
        c = self.array_canvas
        c.delete("all")
        w = max(c.winfo_width(), 100)
        h = max(c.winfo_height(), 100)
        cx, cy = w / 2, h / 2
        side = min(w, h) * 0.55
        # front view X→right, Z→up  (canvas Y grows down)
        corners = {
            0: (cx - side / 2, cy - side / 2),  # M0 TL
            1: (cx + side / 2, cy - side / 2),  # M1 TR
            2: (cx + side / 2, cy + side / 2),  # M2 BR
            3: (cx - side / 2, cy + side / 2),  # M3 BL
        }
        c.create_rectangle(
            corners[0][0],
            corners[0][1],
            corners[2][0],
            corners[2][1],
            outline=LINE,
            width=2,
        )
        c.create_text(cx, 16, text="front view · vertical plane", fill=MUTED, font=("Segoe UI", 10))
        for i, (x, y) in corners.items():
            r = float(rms[i]) if i < len(rms) else 0.0
            rad = 10 + min(18.0, r * 80)
            color = OK if r > 0.01 else LINE
            c.create_oval(x - rad, y - rad, x + rad, y + rad, fill=color, outline=TEXT)
            c.create_text(x, y, text=cfg.MIC_LABELS[i], fill=BG, font=("Segoe UI Semibold", 10))

        # DOA arrow in plane projection (az horizontal, el vertical)
        if speech:
            length = side * 0.42
            dx = math.sin(math.radians(az)) * length
            dy = -math.sin(math.radians(el)) * length
            c.create_line(cx, cy, cx + dx, cy + dy, fill=ACCENT, width=3, arrow=tk.LAST)
        c.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill=TEXT, outline="")

    def _draw_vad(self, prob: float, speech: bool) -> None:
        c = self.vad_canvas
        c.delete("all")
        w = max(c.winfo_width(), 10)
        h = max(c.winfo_height(), 10)
        c.create_rectangle(0, 0, w, h, fill=METER_BG, outline="")
        fill = max(0.0, min(1.0, prob)) * w
        c.create_rectangle(0, 0, fill, h, fill=OK if speech else ACCENT, outline="")
        # threshold marker
        tx = cfg.VAD_THRESHOLD * w
        c.create_line(tx, 0, tx, h, fill=WARN, width=2)

    def _draw_doa(self, az: float, el: float, conf: float) -> None:
        c = self.doa_canvas
        c.delete("all")
        w = max(c.winfo_width(), 100)
        h = max(c.winfo_height(), 100)
        pad = 20
        # az strip
        c.create_text(pad, 18, text="az", fill=MUTED, anchor="w")
        y = 36
        c.create_rectangle(pad, y, w - pad, y + 16, fill=METER_BG, outline=LINE)
        az_n = (az - cfg.AZIMUTH_MIN_DEG) / (cfg.AZIMUTH_MAX_DEG - cfg.AZIMUTH_MIN_DEG)
        ax = pad + az_n * (w - 2 * pad)
        c.create_oval(ax - 7, y - 2, ax + 7, y + 18, fill=ACCENT, outline="")
        c.create_text(pad, y + 28, text=f"{cfg.AZIMUTH_MIN_DEG:.0f}°", fill=MUTED, anchor="w", font=("Consolas", 9))
        c.create_text(w - pad, y + 28, text=f"{cfg.AZIMUTH_MAX_DEG:.0f}°", fill=MUTED, anchor="e", font=("Consolas", 9))

        c.create_text(pad, 80, text="el", fill=MUTED, anchor="w")
        y2 = 98
        c.create_rectangle(pad, y2, w - pad, y2 + 16, fill=METER_BG, outline=LINE)
        el_n = (el - cfg.ELEVATION_MIN_DEG) / (cfg.ELEVATION_MAX_DEG - cfg.ELEVATION_MIN_DEG)
        ex = pad + el_n * (w - 2 * pad)
        c.create_oval(ex - 7, y2 - 2, ex + 7, y2 + 18, fill=OK, outline="")
        c.create_text(
            pad, y2 + 28, text=f"{cfg.ELEVATION_MIN_DEG:.0f}°", fill=MUTED, anchor="w", font=("Consolas", 9)
        )
        c.create_text(
            w - pad, y2 + 28, text=f"{cfg.ELEVATION_MAX_DEG:.0f}°", fill=MUTED, anchor="e", font=("Consolas", 9)
        )

        # confidence bar
        c.create_text(pad, 145, text="conf", fill=MUTED, anchor="w")
        y3 = 158
        c.create_rectangle(pad, y3, w - pad, y3 + 12, fill=METER_BG, outline=LINE)
        cw = conf * (w - 2 * pad)
        color = OK if conf >= cfg.CONFIDENCE_THRESHOLD else WARN
        c.create_rectangle(pad, y3, pad + cw, y3 + 12, fill=color, outline="")

    def _draw_servos(self, pan: float, tilt: float) -> None:
        c = self.servo_canvas
        c.delete("all")
        w = max(c.winfo_width(), 100)
        h = max(c.winfo_height(), 40)
        # two horizontal gauges
        for i, (name, val, lo, hi, color) in enumerate(
            (
                ("PAN", pan, cfg.PAN_MIN_DEG, cfg.PAN_MAX_DEG, ACCENT),
                ("TILT", tilt, cfg.TILT_MIN_DEG, cfg.TILT_MAX_DEG, OK),
            )
        ):
            y = 12 + i * 32
            c.create_text(8, y + 8, text=name, fill=MUTED, anchor="w", font=("Segoe UI", 9))
            x0, x1 = 50, w - 10
            c.create_rectangle(x0, y, x1, y + 16, fill=METER_BG, outline=LINE)
            n = 0.0 if hi == lo else (val - lo) / (hi - lo)
            n = max(0.0, min(1.0, n))
            c.create_rectangle(x0, y, x0 + n * (x1 - x0), y + 16, fill=color, outline="")

    # ── update loop ────────────────────────────────────────────────────────
    def _tick(self) -> None:
        s = self.tracker.snapshot()
        rms = list(s.get("mic_rms") or [0, 0, 0, 0])
        peak = list(s.get("mic_peak") or [0, 0, 0, 0])
        speech = bool(s.get("speech"))
        prob = float(s.get("vad_probability") or 0.0)
        az = float(s.get("azimuth_deg") or 0.0)
        el = float(s.get("elevation_deg") or 0.0)
        conf = float(s.get("confidence") or 0.0)
        pan = float(s.get("pan_deg") or 0.0)
        tilt = float(s.get("tilt_deg") or 0.0)
        sim = bool(s.get("servo_simulate", True))

        self.vad_label.set("SPEECH" if speech else "NON_SPEECH")
        self.vad_prob.set(f"{prob * 100:.0f}%")
        self.vad_meta.set(
            f"peak={float(s.get('vad_peak') or 0):.3f}  gain={float(s.get('vad_gain') or 1):.1f}x"
        )
        self.az_var.set(f"{az:.1f}°")
        self.el_var.set(f"{el:.1f}°")
        self.conf_var.set(f"{conf:.2f}")
        self.pan_var.set(f"{pan:.1f}°")
        self.tilt_var.set(f"{tilt:.1f}°")
        self.servo_mode.set("SIMULATE" if sim else "HAT LIVE")
        live = "LIVE" if self.tracker.running else "STOPPED"
        self.status_var.set(f"{live}  ·  {'SPEECH' if speech else 'idle'}")

        self._draw_mic_meters(rms, peak)
        self._draw_array(rms, az, el, speech)
        self._draw_vad(prob, speech)
        self._draw_doa(az, el, conf)
        self._draw_servos(pan, tilt)

        if self.tracker.running:
            self.root.after(cfg.GUI_REFRESH_MS, self._tick)

    def _on_close(self) -> None:
        self.tracker.stop()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main(simulate_servos: Optional[bool] = None) -> None:
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    LiveDashboard(simulate_servos=simulate_servos).run()


if __name__ == "__main__":
    main()
