#!/usr/bin/env python3
"""
Exercise Waveshare Pan-Tilt HAT after home + assembly.

  .venv/bin/python scripts/test_pantilt.py
  .venv/bin/python scripts/test_pantilt.py --center
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control.servo_controller import ServoController
import config as cfg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--center", action="store_true", help="only go to mid pose")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not ServoController.hat_present(address=cfg.PCA9685_ADDRESS):
        print("ERROR: PCA9685 @ 0x40 not on I2C — enable I2C and reseat HAT")
        return 1

    ctl = ServoController(simulate=False)
    if ctl.simulate:
        print("ERROR: HAT open failed")
        return 1

    if args.center:
        ctl.go_center()
        print(f"center pan={ctl.pan_deg:.1f} tilt={ctl.tilt_deg:.1f}")
        return 0

    print("Sweep pan (S1)…")
    for pan in (cfg.PAN_MIN_DEG, 90.0, cfg.PAN_MAX_DEG, 90.0):
        ctl.set_angles(pan, 90.0, force=True)
        print(f"  pan={pan:.0f}")
        time.sleep(0.8)

    print("Sweep tilt (S0)…")
    for tilt in (cfg.TILT_MIN_DEG, 90.0, cfg.TILT_MAX_DEG, 90.0):
        ctl.set_angles(90.0, tilt, force=True)
        print(f"  tilt={tilt:.0f}")
        time.sleep(0.8)

    ctl.go_center()
    print("OK — HAT responding. Start pipeline with: ./run_gui.sh --real-servos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
