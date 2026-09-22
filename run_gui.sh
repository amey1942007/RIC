#!/bin/bash
# Launch live RIC pipeline GUI on the Raspberry Pi display.
cd "$(dirname "$0")"
export DISPLAY="${DISPLAY:-:0}"
exec ./.venv/bin/python vad_srp_phat_pipeline.py "$@"
