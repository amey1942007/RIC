#!/usr/bin/env python3
"""Record 2s from ric-inmp441-4ch and print per-channel RMS."""
from __future__ import annotations

import math
import struct
import subprocess
import sys
import wave
from pathlib import Path

OUT = Path("/tmp/ric_mic_check.wav")


def main() -> int:
    cmd = [
        "arecord",
        "-D",
        "hw:ricinmp4414ch,0",
        "-c",
        "4",
        "-f",
        "S32_LE",
        "-r",
        "48000",
        "-d",
        "2",
        str(OUT),
    ]
    print(" ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("arecord not found — run on the Raspberry Pi", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as e:
        print("arecord failed:", e, file=sys.stderr)
        return 1

    w = wave.open(str(OUT))
    nch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
    raw = w.readframes(n)
    w.close()
    if sw == 4:
        data = struct.unpack("<%di" % (len(raw) // 4), raw)
    else:
        data = struct.unpack("<%dh" % (len(raw) // 2), raw)

    labels = ["M0 (SD1/L)", "M1 (SD1/R)", "M2 (SD2/L)", "M3 (SD2/R)"]
    print(f"\nchannels={nch} rate={sr} frames={n}")
    ok_n = 0
    for ch in range(nch):
        samples = data[ch::nch]
        mean = sum(samples) / len(samples)
        rms = math.sqrt(sum((x - mean) ** 2 for x in samples) / len(samples))
        peak = max(abs(x) for x in samples)
        status = "OK" if rms > 50 else "DEAD — check wiring"
        if rms > 50:
            ok_n += 1
        name = labels[ch] if ch < len(labels) else f"CH{ch}"
        print(f"  {name}: peak={peak} rms={rms:.0f} -> {status}")

    print(f"\n{ok_n}/{nch} channels live")
    if ok_n < 4:
        print(
            "If M2/M3 are DEAD: move SD2 from pin 40 (GPIO21) to pin 15 (GPIO22)."
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
