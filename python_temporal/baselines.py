"""
Temporal decoders compared against the non-monotonic HSMM (manuscript Table 10).

On the SAME Model-1 per-frame probabilities and the SAME expert phase-onset
annotations, compare the phase-onset accuracy (%MAE) of:

  (1) Frame-wise argmax            - naive, RAW probabilities, no temporal model
  (2) Argmax + temporal smoothing  - naive rule on the smoothed probabilities
  (3) Unordered HMM / Viterbi      - sequence-aware, free (unordered) transitions,
                                     no minimum duration
  (4) Change-point detection (K=5) - greedy binary segmentation on the expected-phase
                                     signal
  (5) Non-monotonic HSMM (proposed) - data-driven transitions + minimum duration +
                                     switch penalty (manuscript)

All decoders consume identical inputs, so the differences reflect the temporal model
only. Every decoder reports phase onsets as {phase: global_time or None} (first-entry
semantics) and is scored against the expert onsets via %MAE. The HSMM transition
matrix is learned per fold from the training cases only (no leakage).

Outputs (in AMVIDEO_RESULTS):  baselines_comparison.csv  (and printed to stdout).
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import config
from temporal_analysis import (
    parse_phase_times, build_surgery_timeline, smooth_probabilities,
    learn_transition, hsmm_decode, first_entry_times, compute_timepoint_errors,
    build_prediction_timeline, get_video_durations, set_duration_source,
    PHASE_INDEX,
)

K = config.NUM_SCENE_CLASSES
TRANS = config.TRANSITION_PHASES


# ── baseline decoders: each returns {phase_name: global_time or None} ──────
def b_argmax(times, P):
    """First time the running frame-wise argmax reaches phase k (first-entry)."""
    am = P.argmax(axis=1)
    out = {}
    for kname in TRANS:
        w = np.where(am >= PHASE_INDEX[kname])[0]
        out[kname] = float(times[w[0]]) if len(w) else None
    return out


def b_viterbi(times, P, ordered):
    """Frame-level Viterbi over K states. Emission = log P; transitions uniform among
    allowed states (left-right if `ordered`, else free). Onset = first frame in phase k."""
    n = len(P)
    logP = np.log(P + 1e-8)
    NEG = -1e18
    dp = np.full((n, K), NEG)
    back = np.zeros((n, K), dtype=int)
    if ordered:
        dp[0, 0] = logP[0, 0]
    else:
        dp[0] = logP[0]
    for t in range(1, n):
        for s in range(K):
            prevs = ([s - 1, s] if s > 0 else [s]) if ordered else range(K)
            best = NEG
            arg = s
            for ps in prevs:
                if dp[t - 1, ps] > best:
                    best = dp[t - 1, ps]
                    arg = ps
            dp[t, s] = best + logP[t, s]
            back[t, s] = arg
    s = int(np.argmax(dp[-1]))
    path = np.zeros(n, dtype=int)
    path[-1] = s
    for t in range(n - 1, 0, -1):
        s = back[t, s]
        path[t - 1] = s
    out = {}
    for kname in TRANS:
        k = PHASE_INDEX[kname]
        w = np.where(path >= k)[0] if ordered else np.where(path == k)[0]
        out[kname] = float(times[w[0]]) if len(w) else None
    return out


def b_changepoint(times, P):
    """K-1 boundaries via greedy binary segmentation on the expected-phase signal.
    Each split maximizes the reduction in within-segment sum of squares."""
    sig = (P * np.arange(K)).sum(axis=1)   # expected phase index over time
    n = len(sig)
    bi = []
    segs = [(0, n)]
    while len(bi) < K - 1:
        best = None
        for s, e in segs:
            if e - s < 4:
                continue
            x = sig[s:e]
            cs = np.cumsum(x)
            tot = cs[-1]
            for t in range(s + 2, e - 1):
                nl = t - s
                nr = e - t
                ml = cs[t - s - 1] / nl
                mr = (tot - cs[t - s - 1]) / nr
                g = nl * ml * ml + nr * mr * mr
                if best is None or g > best[0]:
                    best = (g, s, e, t)
        if best is None:
            break
        _, s, e, t = best
        bi.append(t)
        segs = [x for x in segs if x != (s, e)] + [(s, t), (t, e)]
    bi = sorted(bi)[:K - 1]
    # Onsets are assigned to phases in operative order (change-point is order-agnostic).
    return {kname: (float(times[bi[i]]) if i < len(bi) else None)
            for i, kname in enumerate(TRANS)}


def b_hsmm(times, P, logA):
    """Proposed non-monotonic HSMM."""
    return first_entry_times(times, hsmm_decode(times, P, logA))


def pct_errors(pred, expert, total_dur):
    errs = compute_timepoint_errors(pred, expert)
    return [errs[ph]["absolute_error"] / total_dur * 100 for ph in errs]


# Praw = raw Model-1 per-frame probabilities; Psm = temporally smoothed.
METHODS = {
    "1_argmax":         lambda t, Praw, Psm, logA: b_argmax(t, Praw),
    "2_argmax_smooth":  lambda t, Praw, Psm, logA: b_argmax(t, Psm),
    "3_HMM_unordered":  lambda t, Praw, Psm, logA: b_viterbi(t, Psm, ordered=False),
    "4_changepoint":    lambda t, Praw, Psm, logA: b_changepoint(t, Psm),
    "5_HSMM_proposed":  lambda t, Praw, Psm, logA: b_hsmm(t, Psm, logA),
}


def main():
    set_duration_source(config.SCENE_PRE_DIR)
    phase_times = parse_phase_times()
    acc = {m: [] for m in METHODS}
    for fold in range(1, 6):
        pred = Path(config.SCENE_PRE_DIR) / f"predictions_fold{fold}.csv"
        if not pred.exists():
            print(f"[WARN] {pred} not found, skipping fold {fold}")
            continue
        # Per-fold transition matrix from the training cases only (no leakage).
        test_nums = {int(c.replace("Case", "")) for c in config.FOLD_TEST_CASES[fold]}
        logA = learn_transition({cn: seq for cn, seq in phase_times.items()
                                 if cn not in test_nums})
        for case_str in config.FOLD_TEST_CASES[fold]:
            cn = int(case_str.replace("Case", ""))
            if cn not in phase_times:
                continue
            offsets, expert = build_surgery_timeline(cn, phase_times[cn])
            if offsets is None:
                continue
            times, probs = build_prediction_timeline(str(pred), cn, offsets)
            if times is None or len(times) < 50:
                continue
            sm = smooth_probabilities(times, probs)
            durs = get_video_durations(cn)
            total = sum(durs[v] + 1 for v in sorted(durs)) - 1
            for m, fn in METHODS.items():
                try:
                    pred_onsets = fn(times, probs, sm, logA)
                    acc[m].extend(pct_errors(pred_onsets, expert, total))
                except Exception as e:   # noqa: BLE001 - keep other methods running
                    print(f"  [{m}] case {cn} failed: {e}")

    rows = []
    for m, vals in acc.items():
        v = np.array(vals)
        if len(v) == 0:
            continue
        rows.append({"method": m, "n": len(v),
                     "overall_pctMAE": round(float(v.mean()), 3),
                     "median_pctMAE": round(float(np.median(v)), 3),
                     "max_pctMAE": round(float(v.max()), 2)})
    if not rows:
        print("No results; check input paths in config.py (predictions / phase times).")
        return
    df = pd.DataFrame(rows).sort_values("overall_pctMAE").reset_index(drop=True)
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    out = Path(config.RESULT_DIR) / "baselines_comparison.csv"
    df.to_csv(out, index=False)
    print("=== Baseline comparison (Model-1 inputs, %MAE) ===")
    print(df.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
