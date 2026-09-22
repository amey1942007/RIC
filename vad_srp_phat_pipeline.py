"""
vad_srp_phat_pipeline.py
========================
Combined entry point for the lecturer-tracking audio stack.

Usage (Raspberry Pi with display):
  ./install.sh
  ./run_gui.sh
  ./run_gui.sh --no-gui
  ./run_gui.sh --simulate-servos
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio.silero_vad import SileroVAD  # noqa: E402
from audio.srp_phat import SRPPhatLocalizer  # noqa: E402
from audio.kalman import KalmanFilter2D  # noqa: E402
from control.servo_controller import ServoController  # noqa: E402
from pipeline.lecturer_tracker import LecturerTracker  # noqa: E402

__all__ = [
    "SileroVAD",
    "SRPPhatLocalizer",
    "KalmanFilter2D",
    "ServoController",
    "LecturerTracker",
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RIC lecturer-tracking pipeline")
    parser.add_argument("--no-gui", action="store_true", help="headless (no Tk window)")
    parser.add_argument(
        "--simulate-servos",
        action="store_true",
        help="force pan/tilt simulation (ignore HAT)",
    )
    parser.add_argument(
        "--real-servos",
        action="store_true",
        help="force real pan/tilt HAT (fail → simulate)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    log = logging.getLogger(__name__)

    simulate: bool | None
    if args.simulate_servos:
        simulate = True
    elif args.real_servos:
        simulate = False
    else:
        simulate = None  # auto-detect HAT

    log.info(
        "Pipeline ready | vertical %.0f cm square | mount=%s",
        __import__("config").MIC_SIDE_M * 100,
        __import__("config").ARRAY_MOUNT,
    )

    if args.no_gui:
        tracker = LecturerTracker(simulate_servos=simulate)
        log.info("Headless mode (Ctrl+C to stop)")
        tracker.run_forever()
        return

    from gui.live_dashboard import main as gui_main

    gui_main(simulate_servos=simulate)


if __name__ == "__main__":
    main()
