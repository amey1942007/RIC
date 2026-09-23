"""
vad_srp_phat_pipeline.py — single CLI entry for the lecturer-tracking stack.

  ./run_gui.sh                 → this file (venv + DISPLAY)
  python vad_srp_phat_pipeline.py [--no-gui] [--real-servos]

Engine lives in pipeline/lecturer_tracker.py (LecturerTracker).
GUI lives in gui/live_dashboard.py (wraps LecturerTracker).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.lecturer_tracker import LecturerTracker  # noqa: E402


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

    if args.simulate_servos:
        simulate: bool | None = True
    elif args.real_servos:
        simulate = False
    else:
        simulate = None  # auto-detect HAT

    import config as cfg

    log.info(
        "Pipeline ready | vertical %.0f cm square | mount=%s",
        cfg.MIC_SIDE_M * 100,
        cfg.ARRAY_MOUNT,
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
