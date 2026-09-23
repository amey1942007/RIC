#!/usr/bin/env python3
"""
Mic array check for the Pi.

  .venv/bin/python scripts/check_rpi_4ch_mics.py
      2 s capture, per-channel RMS, which corner each channel maps to.

  .venv/bin/python scripts/check_rpi_4ch_mics.py --side left   [--seconds 12]
      Left/right calibration. Stand ~1 m from the array, clearly on the side you
      name (the array's LEFT is the side the camera points to at pan 0), talk
      continuously. Prints the SRP-PHAT azimuth for the current
      config.MIC_CHANNEL_ORDER and for its mirror, and tells you which one to use.
"""
from __future__ import annotations

import argparse
import math
import struct
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = Path("/tmp/ric_mic_check.wav")
SLOTS = ["SD1/L", "SD1/R", "SD2/L", "SD2/R"]
CORNER = {0: "top-left", 1: "top-right", 2: "bottom-right", 3: "bottom-left"}


def record(seconds: int, rate: int) -> bool:
    cmd = ["arecord", "-D", "hw:ricinmp4414ch,0", "-c", "4", "-f", "S32_LE",
           "-r", str(rate), "-d", str(seconds), str(OUT)]
    print(" ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("arecord not found — run on the Raspberry Pi", file=sys.stderr)
        return False
    except subprocess.CalledProcessError as e:
        print("arecord failed:", e, file=sys.stderr)
        return False
    return True


def load_wav():
    w = wave.open(str(OUT))
    nch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
    raw = w.readframes(n)
    w.close()
    fmt = "<%di" % (len(raw) // 4) if sw == 4 else "<%dh" % (len(raw) // 2)
    return nch, sr, struct.unpack(fmt, raw)


def level_check() -> int:
    if not record(2, 48000):
        return 1
    nch, sr, data = load_wav()
    try:
        import config as cfg
        order = list(getattr(cfg, "MIC_CHANNEL_ORDER", range(4)))
    except Exception:
        order = [0, 1, 2, 3]
    mic_of_ch = {ch: m for m, ch in enumerate(order)}

    print(f"\nchannels={nch} rate={sr} frames={len(data)//nch}")
    ok_n = 0
    for ch in range(nch):
        s = data[ch::nch]
        mean = sum(s) / len(s)
        rms = math.sqrt(sum((x - mean) ** 2 for x in s) / len(s))
        peak = max(abs(x) for x in s)
        ok = rms > 50
        ok_n += ok
        m = mic_of_ch.get(ch)
        tag = f"M{m} {CORNER[m]}" if m is not None else "?"
        print(f"  ch{ch} {SLOTS[ch]} -> {tag}: peak={peak} rms={rms:.0f} -> {'OK' if ok else 'DEAD — check wiring'}")
    print(f"\n{ok_n}/{nch} channels live")
    if ok_n < 4:
        print("If SD2 channels are DEAD: SD2 must be on pin 15 (GPIO22), not pin 40.")
        return 2
    return 0


def side_calibration(side: str, seconds: int) -> int:
    import numpy as np
    from scipy.signal import butter, sosfilt

    import config as cfg
    from audio.srp_phat import SRPPhatLocalizer

    print(f"\nStand ~1 m to the array's {side.upper()} and talk continuously for {seconds}s…", flush=True)
    if not record(seconds, cfg.SAMPLE_RATE):
        return 1
    nch, sr, data = load_wav()
    x = np.asarray(data, dtype=np.float64).reshape(-1, nch) / 2**31
    x -= x.mean(axis=0)
    sos = butter(4, float(getattr(cfg, "AUDIO_HIGHPASS_HZ", 120.0)), btype="highpass", fs=sr, output="sos")
    x = sosfilt(sos, x, axis=0)

    loc = SRPPhatLocalizer(
        cfg.MIC_POSITIONS, sr, cfg.SPEED_OF_SOUND, cfg.SRP_FRAME_SAMPLES,
        (cfg.AZIMUTH_MIN_DEG, cfg.AZIMUTH_MAX_DEG), (cfg.ELEVATION_MIN_DEG, cfg.ELEVATION_MAX_DEG),
        cfg.AZIMUTH_STEP_DEG, cfg.ELEVATION_STEP_DEG,
    )
    N, hop = cfg.SRP_FRAME_SAMPLES, 1024
    starts = list(range(sr // 2, len(x) - N, hop))       # skip first 0.5 s (arecord start click)
    energy = np.array([np.sqrt(np.mean(x[s:s + N] ** 2)) for s in starts])
    keep = [s for s, e in zip(starts, energy) if e >= np.percentile(energy, 60)]  # louder 40 % = speech
    if len(keep) < 8:
        print("Too little sound captured — talk louder / longer.")
        return 2

    cur = tuple(getattr(cfg, "MIC_CHANNEL_ORDER", (0, 1, 2, 3)))
    mirror = (cur[1], cur[0], cur[3], cur[2])            # x → −x : swap TL↔TR and BR↔BL
    results = {}
    for name, order in (("current", cur), ("mirror", mirror)):
        az = np.array([loc.localize(x[s:s + N][:, list(order)])["azimuth_deg"] for s in keep])
        conf = np.array([loc.localize(x[s:s + N][:, list(order)])["confidence"] for s in keep])
        results[name] = (order, float(np.median(az)), float(az.std()), float(np.median(conf)))
        print(f"  {name:8s} MIC_CHANNEL_ORDER={order}: az median={np.median(az):+5.0f}°  std={az.std():4.0f}°  conf={np.median(conf):.2f}")

    want_sign = -1 if side == "left" else +1
    ok_name = None
    for name, (order, med, std, _) in results.items():
        if abs(med) >= 8 and np.sign(med) == want_sign:
            ok_name = name
    print()
    if ok_name is None:
        print("Inconclusive: |azimuth| < 8° for both orders. Stand further to the side (≥45°) and repeat.")
        return 3
    order = results[ok_name][0]
    if ok_name == "current":
        print(f"Current MIC_CHANNEL_ORDER = {cur} is CORRECT for a talker on the {side}.")
    else:
        print(f"Pan sign is MIRRORED. Set in config.py:\n\n    MIC_CHANNEL_ORDER = {order}\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", choices=("left", "right"), help="run left/right pan-sign calibration")
    ap.add_argument("--seconds", type=int, default=12)
    a = ap.parse_args()
    if a.side:
        return side_calibration(a.side, a.seconds)
    return level_check()


if __name__ == "__main__":
    raise SystemExit(main())
