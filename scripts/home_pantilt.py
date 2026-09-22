#!/usr/bin/env python3
"""
Home Waveshare Pan-Tilt HAT servos to 0° BEFORE assembling the bracket.

Waveshare warning: assembling before home can jam/burn the tilt servo.
  https://www.waveshare.com/wiki/Pan-Tilt_HAT

Wire (loose, free to spin):
  Pan  → S1 (orange signal)
  Tilt → S0
  Brown=GND, Red=5V

Usage on Pi:
  cd ~/RIC
  .venv/bin/python scripts/home_pantilt.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control.servo_controller import ServoController
import config as cfg


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print("Waveshare Pan-Tilt HAT — home to 0°")
    print(f"  expecting PCA9685 @ 0x{cfg.PCA9685_ADDRESS:02X}")
    print(f"  tilt=S{cfg.TILT_CHANNEL}  pan=S{cfg.PAN_CHANNEL}")
    print("  Servos must be FREE to rotate (not in bracket yet).\n")

    if not ServoController.hat_present(address=cfg.PCA9685_ADDRESS):
        print("ERROR: PCA9685 not found on I2C.")
        print("  1) HAT seated on the 40-pin header")
        print("  2) sudo raspi-config → Interface Options → I2C → Yes")
        print("  3) sudo reboot && i2cdetect -y 1   (look for 40)")
        return 1

    ctl = ServoController(simulate=False)
    if ctl.simulate:
        print("ERROR: could not open HAT (still in simulate mode)")
        return 1

    ctl.go_home()
    time.sleep(1.5)
    print("OK — both servos commanded to 0°.")
    print("Next: power off → assemble bracket (do not twist tilt horn) → power on")
    print("Then: .venv/bin/python scripts/test_pantilt.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
