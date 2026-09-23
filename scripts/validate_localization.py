#!/usr/bin/env python3
"""
Offline validation + bounded auto-tuning of the SRP-PHAT -> Kalman chain.

  python scripts/validate_localization.py            # evaluate + tune + write if better
  python scripts/validate_localization.py --no-write # never touch config.py
  python scripts/validate_localization.py --iterations 8 --quick

What it does
------------
1. Generates synthetic 4-channel frames from ``config.MIC_POSITIONS`` for known
   (az, el) directions using exact fractional delays (FFT phase shift), adds
   independent Gaussian noise at 20 / 10 / 5 dB SNR, and optionally convolves
   the source with a measured RIR if ``data/rir.wav`` or ``data/rir.npy``
   exists (never fabricated).
2. Runs ``SRPPhatLocalizer`` on >= 21 directions per SNR and the confidence /
   outlier gates from config, then feeds a step response through
   ``KalmanFilter2D`` (with the tracker's median window) to measure settling.
3. Metrics: mean abs error (az, el, combined great-circle), p95 combined error,
   missed-or-rejected rate, Kalman settling frames (to within 2deg after a step).
4. One-parameter-at-a-time ±20 % local search over a bounded tunable set, hard
   cap 25 iterations, every iteration appended to ``data/validation_runs.json``.
   Best parameters are written to config.py ONLY if the combined score beats
   the baseline (and ``--no-write`` is not given); the diff is printed.

Deliberately NOT tuned: KALMAN_MAX_STEP_DEG / KALMAN_MAX_VEL_DEG_S, any mic
geometry / mount, any servo mapping, VAD_THRESHOLD or VAD debounce counts.

This script imports only ``SRPPhatLocalizer`` and ``KalmanFilter2D``. It never
imports or instantiates ``LecturerTracker``, ``ServoController`` or
``SileroVAD`` and never opens a sounddevice stream.

LIMITATION - read before trusting numbers
-----------------------------------------
All results are SYNTHETIC. Free-field fractional-delay simulation with white
Gaussian noise (and at most one shared RIR) does not reproduce real classroom
reverberation, mic self-noise / mismatch, HAT servo noise coupling, or moving
talkers. Passing here is a necessary, not sufficient, condition. Validate on the
Pi with a real talker before considering any tuning final.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.fft import irfft, rfft, rfftfreq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows consoles default to cp1252; never let a degree sign kill a run.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover
            pass

import config as cfg  # noqa: E402
from audio.kalman import KalmanFilter2D  # noqa: E402
from audio.srp_phat import SRPPhatLocalizer  # noqa: E402

# ── acceptance targets ────────────────────────────────────────────────────────
TARGET_MEAN_ERROR_DEG = 5.0
TARGET_P95_ERROR_DEG = 12.0
TARGET_MISS_RATE = 0.1
TARGET_SETTLE_FRAMES = 10

# ── search settings ───────────────────────────────────────────────────────────
MAX_ITERATIONS = 25
STEP_FRACTION = 0.20
SNRS_DB = (20.0, 10.0, 5.0)
FRAMES_PER_DIRECTION = 3
SETTLE_TRIALS = 3
SETTLE_MAX_FRAMES = 60
SETTLE_TOL_DEG = 2.0
SETTLE_STEP_AZ_EL = (20.0, 10.0)  # start (0,0) -> this; ~22deg combined
SETTLE_SNR_DB = 10.0
RUNS_FILE = ROOT / "data" / "validation_runs.json"
CONFIG_FILE = ROOT / "config.py"
RIR_CANDIDATES = (ROOT / "data" / "rir.wav", ROOT / "data" / "rir.npy")

# name -> (lo, hi, kind)   kind ∈ {"float", "int", "pow2"}
TUNABLES: Dict[str, Tuple[float, float, str]] = {
    "SRP_COARSE_STEP_MULTIPLIER": (1, 6, "int"),
    "SRP_VOLUME_NEIGHBORS": (0, 3, "int"),
    "SRP_FRAME_SAMPLES": (1024, 4096, "pow2"),
    "KALMAN_Q_POS": (0.005, 1.0, "float"),
    "KALMAN_Q_VEL": (0.02, 2.0, "float"),
    "KALMAN_R": (10.0, 400.0, "float"),
    "CONFIDENCE_THRESHOLD": (0.40, 0.85, "float"),
    "CONFIDENCE_MOVE_THRESHOLD": (0.45, 0.90, "float"),
    "SRP_MAX_JUMP_DEG": (10.0, 60.0, "float"),
}

LIMITATION_BANNER = (
    "NOTE: SYNTHETIC-ONLY validation (free-field fractional delays + Gaussian "
    "noise{rir}). Does not capture real classroom reverberation, mic mismatch or "
    "moving talkers. Confirm on the Pi with a real talker before trusting tuning."
)


# ── geometry helpers (mirror audio/srp_phat.py conventions) ──────────────────
def direction_unit(az_deg: float, el_deg: float) -> np.ndarray:
    az, el = math.radians(az_deg), math.radians(el_deg)
    x = math.sin(az) * math.cos(el)
    y = math.cos(az) * math.cos(el)
    z = math.sin(el)
    if getattr(cfg, "ARRAY_MOUNT", "vertical_stand") == "ceiling":
        z = -z
    v = np.array([x, y, z], dtype=np.float64)
    return v / np.linalg.norm(v)


def angular_error_deg(az1: float, el1: float, az2: float, el2: float) -> float:
    d = float(np.clip(np.dot(direction_unit(az1, el1), direction_unit(az2, el2)), -1, 1))
    return math.degrees(math.acos(d))


def wrap180(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0


# ── synthetic data ───────────────────────────────────────────────────────────
def load_rir() -> Tuple[Optional[np.ndarray], str]:
    """Return (rir, description). Never fabricates one."""
    for p in RIR_CANDIDATES:
        if not p.exists():
            continue
        try:
            if p.suffix == ".npy":
                rir = np.load(p).astype(np.float64).ravel()
                sr = cfg.SAMPLE_RATE
            else:
                try:
                    import soundfile as sf

                    rir, sr = sf.read(p, dtype="float64", always_2d=False)
                except ImportError:
                    from scipy.io import wavfile

                    sr, rir = wavfile.read(p)
                    rir = rir.astype(np.float64)
                    if rir.dtype.kind in "iu":
                        rir /= np.iinfo(rir.dtype).max
                if rir.ndim > 1:
                    rir = rir[:, 0]
            if int(sr) != int(cfg.SAMPLE_RATE):
                return None, f"{p.name} skipped (sr={sr} != {cfg.SAMPLE_RATE})"
            rir = rir / (np.max(np.abs(rir)) + 1e-12)
            return rir, f"{p.name} ({len(rir)} taps)"
        except Exception as exc:  # pragma: no cover
            return None, f"{p.name} unreadable ({exc})"
    return None, "none found (data/rir.wav | data/rir.npy) - free-field only"


def make_source(n: int, rng: np.random.Generator, fs: int) -> np.ndarray:
    """Speech-like test signal: drifting-f0 harmonic stack + 200–4000 Hz noise."""
    t = np.arange(n) / fs
    f0 = 140.0 + 25.0 * np.sin(2 * np.pi * 2.3 * t + rng.uniform(0, 2 * np.pi))
    phase = 2 * np.pi * np.cumsum(f0) / fs
    harm = np.zeros(n)
    for k in range(1, 30):
        if 140.0 * k > 4000.0:
            break
        harm += (1.0 / k) * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
    noise = rng.standard_normal(n)
    N = rfft(noise)
    f = rfftfreq(n, 1.0 / fs)
    N[(f < 200.0) | (f > 4000.0)] = 0.0
    noise = irfft(N, n=n)
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 3.0 * t + rng.uniform(0, 2 * np.pi))
    s = (harm / np.max(np.abs(harm)) + 0.6 * noise / np.max(np.abs(noise))) * env
    return s / (np.max(np.abs(s)) + 1e-12)


def synth_frame(
    az: float,
    el: float,
    snr_db: float,
    n_fft: int,
    seed: int,
    rir: Optional[np.ndarray],
) -> np.ndarray:
    """(n_fft, NUM_MICS) frame of a far-field source at (az, el) + noise."""
    rng = np.random.default_rng(seed)
    fs = cfg.SAMPLE_RATE
    pad = 128
    n = n_fft + 2 * pad
    s = make_source(n + (len(rir) if rir is not None else 0), rng, fs)
    if rir is not None:
        s = np.convolve(s, rir)[:n]
    s = s[:n]
    S = rfft(s)
    f = rfftfreq(n, 1.0 / fs)
    u = direction_unit(az, el)
    mic_ids = sorted(cfg.MIC_POSITIONS.keys())
    out = np.zeros((n_fft, len(mic_ids)), dtype=np.float64)
    for k, mid in enumerate(mic_ids):
        r = np.asarray(cfg.MIC_POSITIONS[mid], dtype=np.float64)
        tau = -float(np.dot(r, u)) / cfg.SPEED_OF_SOUND  # closer mic -> earlier
        x = irfft(S * np.exp(-2j * np.pi * f * tau), n=n)
        out[:, k] = x[pad : pad + n_fft]
    sig_rms = float(np.sqrt(np.mean(out**2)) + 1e-12)
    sigma = sig_rms / (10.0 ** (snr_db / 20.0))
    out += rng.standard_normal(out.shape) * sigma
    return out


def test_directions() -> List[Tuple[float, float]]:
    azs = np.linspace(-70.0, 70.0, 7)
    els = (-15.0, 10.0, 35.0)
    dirs = [(float(a), float(e)) for e in els for a in azs]
    # stay inside the search grid
    return [
        (a, e)
        for a, e in dirs
        if cfg.AZIMUTH_MIN_DEG <= a <= cfg.AZIMUTH_MAX_DEG
        and cfg.ELEVATION_MIN_DEG <= e <= cfg.ELEVATION_MAX_DEG
    ]


class FrameCache:
    """Frames are deterministic per (az, el, snr, idx, n_fft) -> fair comparisons."""

    def __init__(self, rir: Optional[np.ndarray]) -> None:
        self.rir = rir
        self._c: Dict[tuple, np.ndarray] = {}

    def get(self, az: float, el: float, snr: float, idx: int, n_fft: int) -> np.ndarray:
        key = (az, el, snr, idx, n_fft)
        if key not in self._c:
            seed = abs(hash((round(az, 3), round(el, 3), round(snr, 3), idx))) % (2**32)
            self._c[key] = synth_frame(az, el, snr, n_fft, seed, self.rir)
        return self._c[key]


# ── evaluation ───────────────────────────────────────────────────────────────
def build_localizer(p: Dict[str, float]) -> SRPPhatLocalizer:
    return SRPPhatLocalizer(
        mic_positions=cfg.MIC_POSITIONS,
        sample_rate=cfg.SAMPLE_RATE,
        speed_of_sound=cfg.SPEED_OF_SOUND,
        n_fft=int(p["SRP_FRAME_SAMPLES"]),
        azimuth_range=(cfg.AZIMUTH_MIN_DEG, cfg.AZIMUTH_MAX_DEG),
        elevation_range=(cfg.ELEVATION_MIN_DEG, cfg.ELEVATION_MAX_DEG),
        azimuth_step=cfg.AZIMUTH_STEP_DEG,
        elevation_step=cfg.ELEVATION_STEP_DEG,
        adaptive=bool(getattr(cfg, "SRP_ADAPTIVE_SEARCH", True)),
        coarse_multiplier=int(p["SRP_COARSE_STEP_MULTIPLIER"]),
        volume_neighbors=int(p["SRP_VOLUME_NEIGHBORS"]),
    )


def build_kalman(p: Dict[str, float]) -> KalmanFilter2D:
    return KalmanFilter2D(
        dt=cfg.KALMAN_DT,
        q_pos=float(p["KALMAN_Q_POS"]),
        q_vel=float(p["KALMAN_Q_VEL"]),
        r_meas=float(p["KALMAN_R"]),
        max_step_deg=cfg.KALMAN_MAX_STEP_DEG,  # fixed, not tuned
        max_vel_deg_s=cfg.KALMAN_MAX_VEL_DEG_S,  # fixed, not tuned
    )


def evaluate(p: Dict[str, float], cache: FrameCache, dirs: List[Tuple[float, float]]) -> Dict:
    loc = build_localizer(p)
    n_fft = int(p["SRP_FRAME_SAMPLES"])
    thr = float(p["CONFIDENCE_THRESHOLD"])
    move_thr = float(p["CONFIDENCE_MOVE_THRESHOLD"])
    max_jump = float(p["SRP_MAX_JUMP_DEG"])

    az_err: List[float] = []
    el_err: List[float] = []
    comb_err: List[float] = []
    total = missed = rejected = gated = 0
    per_snr: Dict[str, Dict[str, float]] = {}

    for snr in SNRS_DB:
        snr_comb: List[float] = []
        snr_total = snr_bad = 0
        for az, el in dirs:
            last_good: Optional[Tuple[float, float]] = None
            for k in range(FRAMES_PER_DIRECTION):
                total += 1
                snr_total += 1
                res = loc.localize(cache.get(az, el, snr, k, n_fft))
                conf = float(res["confidence"])
                ea, ee = float(res["azimuth_deg"]), float(res["elevation_deg"])
                if conf < thr:
                    missed += 1
                    snr_bad += 1
                    continue
                if last_good is not None and (
                    abs(wrap180(ea - last_good[0])) > max_jump
                    or abs(ee - last_good[1]) > max_jump
                ):
                    rejected += 1
                    snr_bad += 1
                    continue
                last_good = (ea, ee)
                if conf < move_thr:
                    # accepted by the localiser gate but the HAT would not move
                    gated += 1
                    snr_bad += 1
                    continue
                e_comb = angular_error_deg(az, el, ea, ee)
                az_err.append(abs(wrap180(ea - az)))
                el_err.append(abs(ee - el))
                comb_err.append(e_comb)
                snr_comb.append(e_comb)
        per_snr[f"{snr:g}dB"] = {
            "mean_err_deg": float(np.mean(snr_comb)) if snr_comb else 180.0,
            "miss_rate": snr_bad / max(1, snr_total),
        }

    settle = settle_frames(p, cache)

    if comb_err:
        mean_az = float(np.mean(az_err))
        mean_el = float(np.mean(el_err))
        mean_c = float(np.mean(comb_err))
        p95 = float(np.percentile(comb_err, 95))
    else:  # nothing accepted -> do not let this look like a win
        mean_az = mean_el = mean_c = p95 = 180.0
    miss_rate = (missed + rejected + gated) / max(1, total)

    metrics = {
        "mean_az_err_deg": mean_az,
        "mean_el_err_deg": mean_el,
        "mean_err_deg": mean_c,
        "p95_err_deg": p95,
        "miss_rate": miss_rate,
        "missed_frames": missed,
        "rejected_outlier_frames": rejected,
        "servo_gated_frames": gated,
        "total_frames": total,
        "settle_frames": settle,
        "per_snr": per_snr,
    }
    metrics["score"] = combined_score(metrics)
    return metrics


def settle_frames(p: Dict[str, float], cache: FrameCache) -> float:
    """
    Frames until the Kalman output is within SETTLE_TOL_DEG of the localiser's
    own steady-state estimate after a step from (0,0) to SETTLE_STEP_AZ_EL.

    The reference is the median of the accepted SRP estimates for the episode
    (not the ground truth) so this measures filter dynamics only; localiser
    bias is already captured by the error metrics.
    """
    loc = build_localizer(p)
    n_fft = int(p["SRP_FRAME_SAMPLES"])
    thr = float(p["CONFIDENCE_THRESHOLD"])
    win = max(1, int(getattr(cfg, "SRP_MEDIAN_WINDOW", 5)))
    tgt_az, tgt_el = SETTLE_STEP_AZ_EL
    results: List[int] = []
    for trial in range(SETTLE_TRIALS):
        # 1) localiser estimates for the whole episode
        ests: List[Optional[Tuple[float, float, float]]] = []
        for n in range(SETTLE_MAX_FRAMES):
            frame = cache.get(tgt_az, tgt_el, SETTLE_SNR_DB, 1000 + trial * 100 + n, n_fft)
            res = loc.localize(frame)
            conf = float(res["confidence"])
            ests.append(
                (float(res["azimuth_deg"]), float(res["elevation_deg"]), conf)
                if conf >= thr
                else None
            )
        good = [e for e in ests if e is not None]
        if not good:
            results.append(SETTLE_MAX_FRAMES)
            continue
        ref_az = float(np.median([e[0] for e in good]))
        ref_el = float(np.median([e[1] for e in good]))

        # 2) Kalman + tracker-style median window
        kf = build_kalman(p)
        kf.reset(0.0, 0.0)
        hist_az: List[float] = []
        hist_el: List[float] = []
        settled = SETTLE_MAX_FRAMES
        for n, e in enumerate(ests):
            meas = None
            conf = None
            if e is not None:
                hist_az.append(e[0])
                hist_el.append(e[1])
                hist_az, hist_el = hist_az[-win:], hist_el[-win:]
                meas = (float(np.median(hist_az)), float(np.median(hist_el)))
                conf = e[2]
            az, el = kf.step(meas, conf)
            if angular_error_deg(az, el, ref_az, ref_el) <= SETTLE_TOL_DEG:
                settled = n + 1
                break
        results.append(settled)
    return float(np.mean(results))


def combined_score(m: Dict) -> float:
    """Lower is better; each term is metric / target so 1.0 = exactly on target."""
    return (
        m["mean_err_deg"] / TARGET_MEAN_ERROR_DEG
        + m["p95_err_deg"] / TARGET_P95_ERROR_DEG
        + m["miss_rate"] / TARGET_MISS_RATE
        + m["settle_frames"] / TARGET_SETTLE_FRAMES
    )


def target_report(m: Dict) -> List[str]:
    checks = [
        ("mean_err_deg", m["mean_err_deg"], TARGET_MEAN_ERROR_DEG),
        ("p95_err_deg", m["p95_err_deg"], TARGET_P95_ERROR_DEG),
        ("miss_rate", m["miss_rate"], TARGET_MISS_RATE),
        ("settle_frames", m["settle_frames"], TARGET_SETTLE_FRAMES),
    ]
    return [
        f"  {'PASS' if v <= t else 'FAIL'}  {name:<14} {v:8.3f}  (target <= {t:g})"
        for name, v, t in checks
    ]


# ── parameter handling ───────────────────────────────────────────────────────
def current_params() -> Dict[str, float]:
    p = {}
    for name, (lo, hi, kind) in TUNABLES.items():
        v = getattr(cfg, name)
        p[name] = int(v) if kind != "float" else float(v)
    return p


def perturb(value: float, direction: int, spec: Tuple[float, float, str]) -> Optional[float]:
    lo, hi, kind = spec
    if kind == "pow2":
        new = int(value) * 2 if direction > 0 else int(value) // 2
    elif kind == "int":
        delta = max(1, int(round(abs(value) * STEP_FRACTION)))
        new = int(value) + direction * delta
    else:
        new = float(value) * (1.0 + direction * STEP_FRACTION)
    new = min(hi, max(lo, new))
    if kind != "float":
        new = int(new)
    if kind == "float":
        new = round(float(new), 5)
    return None if new == value else new


def fmt_value(v: float, kind: str) -> str:
    return str(int(v)) if kind != "float" else f"{float(v):g}"


def write_config(best: Dict[str, float], base: Dict[str, float]) -> List[str]:
    text = CONFIG_FILE.read_text(encoding="utf-8")
    diff: List[str] = []
    for name, (_, _, kind) in TUNABLES.items():
        if best[name] == base[name]:
            continue
        pat = re.compile(rf"^({re.escape(name)}\s*=\s*)([^#\n]+?)(\s*#.*)?$", re.M)
        m = pat.search(text)
        if not m:
            diff.append(f"  !! {name}: not found in config.py - NOT written")
            continue
        new_val = fmt_value(best[name], kind)
        text = text[: m.start(2)] + new_val + text[m.end(2) :]
        diff.append(f"  {name}: {m.group(2).strip()}  ->  {new_val}")
    CONFIG_FILE.write_text(text, encoding="utf-8")
    return diff


def append_run(record: Dict) -> None:
    RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    runs: List[Dict] = []
    if RUNS_FILE.exists():
        try:
            runs = json.loads(RUNS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            runs = []
    runs.append(record)
    RUNS_FILE.write_text(json.dumps(runs, indent=2), encoding="utf-8")


# ── main loop ────────────────────────────────────────────────────────────────
def main() -> int:
    global FRAMES_PER_DIRECTION, SETTLE_TRIALS

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--iterations", type=int, default=MAX_ITERATIONS,
                    help=f"tuning iterations (hard cap {MAX_ITERATIONS})")
    ap.add_argument("--no-write", action="store_true", help="never modify config.py")
    ap.add_argument("--quick", action="store_true", help="1 frame/direction, 1 settle trial")
    args = ap.parse_args()

    n_iter = max(0, min(args.iterations, MAX_ITERATIONS))
    if args.quick:
        FRAMES_PER_DIRECTION = 1
        SETTLE_TRIALS = 1

    rir, rir_desc = load_rir()
    banner = LIMITATION_BANNER.format(rir=" + shared RIR" if rir is not None else "")
    print("=" * 78)
    print(banner)
    print("=" * 78)
    print(f"RIR: {rir_desc}")
    dirs = test_directions()
    print(f"Directions per SNR: {len(dirs)}   SNRs: {SNRS_DB} dB   "
          f"frames/dir: {FRAMES_PER_DIRECTION}   adaptive={getattr(cfg, 'SRP_ADAPTIVE_SEARCH', True)}")
    cache = FrameCache(rir)
    run_id = time.strftime("%Y%m%d-%H%M%S")

    base = current_params()
    t0 = time.time()
    base_m = evaluate(base, cache, dirs)
    print(f"\n[iter 0/{n_iter}] baseline  score={base_m['score']:.3f}  "
          f"mean={base_m['mean_err_deg']:.2f}deg p95={base_m['p95_err_deg']:.2f}deg "
          f"miss={base_m['miss_rate']:.3f} settle={base_m['settle_frames']:.1f}  "
          f"({time.time() - t0:.1f}s)")
    append_run({"run_id": run_id, "iteration": 0, "params": base, "metrics": base_m,
                "accepted": True, "note": "baseline", "t": time.time()})

    best, best_m = dict(base), base_m
    names = list(TUNABLES.keys())
    it = 0
    cursor = 0  # (param, direction) round-robin
    tried: set = set()
    while it < n_iter:
        name = names[(cursor // 2) % len(names)]
        direction = 1 if cursor % 2 == 0 else -1
        cursor += 1
        if cursor > 2 * len(names) * 3:  # three full sweeps without progress -> stop
            print("Local search exhausted (no further ±20 % moves improve the score).")
            break
        new_val = perturb(best[name], direction, TUNABLES[name])
        if new_val is None:
            continue
        cand = dict(best)
        cand[name] = new_val
        key = tuple(sorted(cand.items()))
        if key in tried:
            continue
        tried.add(key)
        it += 1
        t1 = time.time()
        m = evaluate(cand, cache, dirs)
        accepted = m["score"] < best_m["score"] - 1e-9
        print(f"[iter {it}/{n_iter}] {name}: {fmt_value(best[name], TUNABLES[name][2])} -> "
              f"{fmt_value(new_val, TUNABLES[name][2])}  score={m['score']:.3f}  "
              f"mean={m['mean_err_deg']:.2f}deg p95={m['p95_err_deg']:.2f}deg "
              f"miss={m['miss_rate']:.3f} settle={m['settle_frames']:.1f}  "
              f"{'ACCEPT' if accepted else 'reject'}  ({time.time() - t1:.1f}s)")
        append_run({"run_id": run_id, "iteration": it, "params": cand, "metrics": m,
                    "changed": name, "accepted": accepted, "t": time.time()})
        if accepted:
            best, best_m = cand, m
            cursor = 0  # restart sweep from the new best

    # ── report ────────────────────────────────────────────────────────────
    print("\n" + "-" * 78)
    print(f"BEST parameters (score {best_m['score']:.3f} vs baseline {base_m['score']:.3f}):")
    for name in names:
        mark = "  *" if best[name] != base[name] else ""
        print(f"  {name:<28} {fmt_value(best[name], TUNABLES[name][2])}{mark}")
    print("Metrics vs targets:")
    lines = target_report(best_m)
    print("\n".join(lines))
    print(f"  mean az err {best_m['mean_az_err_deg']:.2f}deg  mean el err {best_m['mean_el_err_deg']:.2f}deg  "
          f"missed={best_m['missed_frames']} rejected={best_m['rejected_outlier_frames']} "
          f"servo-gated={best_m['servo_gated_frames']} / {best_m['total_frames']} frames")
    for snr, d in best_m["per_snr"].items():
        print(f"  {snr:>5}: mean err {d['mean_err_deg']:.2f}deg  miss {d['miss_rate']:.3f}")
    failed = [ln for ln in lines if "FAIL" in ln]
    if failed:
        print(f"\n{len(failed)} TARGET(S) MISSED - tuning did NOT reach spec. "
              f"Do not treat this run as a pass.")
    else:
        print("\nAll targets met on synthetic data.")

    # ── write-back ────────────────────────────────────────────────────────
    improved = best_m["score"] < base_m["score"] - 1e-9 and best != base
    if not improved:
        print("\nconfig.py unchanged (no candidate beat the baseline combined score).")
    elif args.no_write:
        print("\n--no-write given: config.py unchanged. Would have applied:")
        for name in names:
            if best[name] != base[name]:
                print(f"  {name}: {fmt_value(base[name], TUNABLES[name][2])} -> "
                      f"{fmt_value(best[name], TUNABLES[name][2])}")
    else:
        diff = write_config(best, base)
        print("\nconfig.py updated:")
        print("\n".join(diff))

    print(f"\nAll {it + 1} iterations logged to {RUNS_FILE.relative_to(ROOT)}")
    print("\n" + banner)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
