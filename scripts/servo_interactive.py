#!/usr/bin/env python3
"""
Interactive Waveshare pan/tilt terminal.

Type angles to move servos and discover hardware stop limits, then copy
those into config.py (PAN_MIN_DEG / PAN_MAX_DEG / TILT_MIN_DEG / TILT_MAX_DEG).

Examples
--------
  pan 90
  tilt 45
  both 90 60
  center
  home
  limits          # print current clamps
  setpanmin 25    # set runtime clamp (also prints config line)
  setpanmax 155
  settiltmin 35
  settiltmax 145
  save            # write limits into config.py
  quit

Run on Pi:
  cd ~/RIC && .venv/bin/python scripts/servo_interactive.py
  .venv/bin/python scripts/servo_interactive.py --simulate
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as cfg
from control.servo_controller import ServoController, np_clip


HELP = """
Commands
  pan <deg>              move pan (S1) only
  tilt <deg>             move tilt (S0) only
  both <pan> <tilt>      move both
  +pan / -pan <d>        nudge pan by ±d (default 5)
  +tilt / -tilt <d>      nudge tilt by ±d (default 5)
  center                 mid of current limits
  home                   0,0 (only if free to spin — Waveshare pre-assembly)
  limits                 show clamps
  setpanmin|setpanmax <deg>
  settiltmin|settiltmax <deg>
  save                   patch config.py with current limits
  help | quit
"""


def _patch_config(pan_min: float, pan_max: float, tilt_min: float, tilt_max: float) -> None:
    path = ROOT / "config.py"
    text = path.read_text(encoding="utf-8")

    def repl(name: str, value: float, src: str) -> str:
        return re.sub(
            rf"^({name}\s*=\s*)[-\d.]+",
            rf"\g<1>{value:.1f}",
            src,
            count=1,
            flags=re.M,
        )

    text = repl("PAN_MIN_DEG", pan_min, text)
    text = repl("PAN_MAX_DEG", pan_max, text)
    text = repl("TILT_MIN_DEG", tilt_min, text)
    text = repl("TILT_MAX_DEG", tilt_max, text)
    path.write_text(text, encoding="utf-8")
    print(f"Wrote limits to {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive pan/tilt terminal")
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.simulate and not ServoController.hat_present(address=cfg.PCA9685_ADDRESS):
        print("PCA9685 @ 0x40 not found. Enable I2C or use --simulate.")
        return 1

    ctl = ServoController(simulate=args.simulate)
    # Mutable runtime clamps (start from config)
    pan_min, pan_max = float(cfg.PAN_MIN_DEG), float(cfg.PAN_MAX_DEG)
    tilt_min, tilt_max = float(cfg.TILT_MIN_DEG), float(cfg.TILT_MAX_DEG)

    def clamp_pan(v: float) -> float:
        return np_clip(v, pan_min, pan_max)

    def clamp_tilt(v: float) -> float:
        return np_clip(v, tilt_min, tilt_max)

    def show() -> None:
        mode = "SIM" if ctl.simulate else "HAT"
        print(
            f"[{mode}] pan={ctl.pan_deg:.1f}°  tilt={ctl.tilt_deg:.1f}°  "
            f"clamps pan[{pan_min:.0f},{pan_max:.0f}] tilt[{tilt_min:.0f},{tilt_max:.0f}]"
        )

    print("Waveshare interactive servo terminal")
    print(f"  tilt=S{cfg.TILT_CHANNEL}  pan=S{cfg.PAN_CHANNEL}")
    print(HELP)
    show()

    while True:
        try:
            line = input("servo> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()

        try:
            if cmd in ("q", "quit", "exit"):
                break
            if cmd in ("h", "help", "?"):
                print(HELP)
                continue
            if cmd == "limits":
                show()
                continue
            if cmd == "center":
                ctl.set_angles(
                    0.5 * (pan_min + pan_max),
                    0.5 * (tilt_min + tilt_max),
                    force=True,
                )
                show()
                continue
            if cmd == "home":
                ctl.go_home()
                show()
                continue
            if cmd == "pan" and len(parts) >= 2:
                ctl.set_angles(clamp_pan(float(parts[1])), ctl.tilt_deg, force=True)
                show()
                continue
            if cmd == "tilt" and len(parts) >= 2:
                ctl.set_angles(ctl.pan_deg, clamp_tilt(float(parts[1])), force=True)
                show()
                continue
            if cmd == "both" and len(parts) >= 3:
                ctl.set_angles(
                    clamp_pan(float(parts[1])),
                    clamp_tilt(float(parts[2])),
                    force=True,
                )
                show()
                continue
            if cmd in ("+pan", "-pan"):
                d = float(parts[1]) if len(parts) > 1 else 5.0
                if cmd == "-pan":
                    d = -d
                ctl.set_angles(clamp_pan(ctl.pan_deg + d), ctl.tilt_deg, force=True)
                show()
                continue
            if cmd in ("+tilt", "-tilt"):
                d = float(parts[1]) if len(parts) > 1 else 5.0
                if cmd == "-tilt":
                    d = -d
                ctl.set_angles(ctl.pan_deg, clamp_tilt(ctl.tilt_deg + d), force=True)
                show()
                continue
            if cmd == "setpanmin" and len(parts) >= 2:
                pan_min = float(parts[1])
                print(f"PAN_MIN_DEG = {pan_min:.1f}")
                continue
            if cmd == "setpanmax" and len(parts) >= 2:
                pan_max = float(parts[1])
                print(f"PAN_MAX_DEG = {pan_max:.1f}")
                continue
            if cmd == "settiltmin" and len(parts) >= 2:
                tilt_min = float(parts[1])
                print(f"TILT_MIN_DEG = {tilt_min:.1f}")
                continue
            if cmd == "settiltmax" and len(parts) >= 2:
                tilt_max = float(parts[1])
                print(f"TILT_MAX_DEG = {tilt_max:.1f}")
                continue
            if cmd == "save":
                if pan_min >= pan_max or tilt_min >= tilt_max:
                    print("Invalid limits (min must be < max)")
                    continue
                _patch_config(pan_min, pan_max, tilt_min, tilt_max)
                cfg.PAN_MIN_DEG = pan_min
                cfg.PAN_MAX_DEG = pan_max
                cfg.TILT_MIN_DEG = tilt_min
                cfg.TILT_MAX_DEG = tilt_max
                continue
            print("Unknown command — type help")
        except Exception as exc:
            print(f"Error: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
