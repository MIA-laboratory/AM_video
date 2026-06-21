"""
Sequence-aware baseline decoders vs. the constrained DP (paper Table 10).

On the SAME configuration-B inputs (Model 1 per-frame probabilities with the OK/NG
gate at theta=0.5) and the SAME expert phase-time annotations, compare the
boundary-estimation accuracy (%MAE) of:

  (1) Frame-wise argmax            - naive, RAW probabilities, no temporal model
  (2) Argmax + temporal smoothing  - naive rule on the smoothed probabilities
  (3) Unordered HMM / Viterbi      - sequence-aware, free transitions
                                     (no fixed order, no minimum duration)
  (4) Ordered DP, NO min-duration  - the proposed DP with the duration constraint removed
  (5) Change-point detection (K=5) - greedy binary segmentation on the expected-phase signal
  (6) Constrained DP (proposed)    - fixed order + minimum duration (manuscript)

All decoders consume identical inputs, so the differences reflect the temporal model
only. Boundaries are compared to the expert annotations via %MAE.

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
    dp_segmentation, compute_timepoint_errors, get_video_durations,
    set_duration_source,
)
from v6_temporal import load_okprobs, build_timeline

K = config.NUM_SCENE_CLASSES
THRESHOLD = 0.5   # configuration-B OK/NG gate


# ── baseline decoders: each returns K-1 = 5 boundary times (phase 1->2 ... 5->6) ──
def b_argmax(times, P):
    """First time the running argmax reaches phase k (k=1..K-1)."""
    am = P.argmax(axis=1)
    bnds = []
    for k in range(1, K):
        idx = np.where(am >= k)[0]
        bnds.append(float(times[idx[0]]) if len(idx) else float(times[-1]))
    return bnds


def b_changepoint(times, P):
    """K-1 boundaries via greedy binary segmentation on the expected-phase signal.
    Each split maximizes the reduction in within-segment sum of squares."""
    sig = (P * np.arange(K)).sum(axis=1)   # expected phase index over time
    n = len(sig)
    bnds_idx = []
    segments = [(0, n)]
    while len(bnds_idx) < K - 1:
        best = None
        for (s, e) in segments:
            if e - s < 4:
                continue
            x = sig[s:e]
            csum = np.cumsum(x); ctot = csum[-1]
            for t in range(s + 2, e - 1):
                nl = t - s; nr = e - t
                ml = csum[t - s - 1] / nl
                mr = (ctot - csum[t - s - 1]) / nr
                gain = nl * ml * ml + nr * mr * mr
                if best is None or gain > best[0]:
                    best = (gain, s, e, t)
        if best is None:
            break
        _, s, e, t = best
        bnds_idx.append(t)
        segments = [seg for seg in segments if seg != (s, e)] + [(s, t), (t, e)]
    bnds_idx = sorted(bnds_idx)[:K - 1]
    while len(bnds_idx) < K - 1:
        bnds_idx.append(n - 1)
    return [float(times[i]) for i in bnds_idx]


def b_viterbi(times, P, ordered):
    """Left-right (ordered) or free (unordered) Viterbi over K states.
    Emission = log P; transitions uniform among allowed states. Boundaries =
    first frame assigned to each phase k=1..K-1."""
    n = len(P)
    logP = np.log(P + 1e-8)
    NEG = -1e18
    dp = np.full((n, K), NEG); back = np.zeros((n, K), dtype=int)
    if ordered:
        dp[0, 0] = logP[0, 0]
    else:
        for s in range(K):
            dp[0, s] = logP[0, s]
    for t in range(1, n):
        for s in range(K):
            prevs = ([s - 1, s] if s > 0 else [s]) if ordered else range(K)
            best = NEG; arg = s
            for ps in prevs:
                if dp[t - 1, ps] > best:
                    best = dp[t - 1, ps]; arg = ps
            dp[t, s] = best + logP[t, s]; back[t, s] = arg
    s = int(np.argmax(dp[n - 1]))
    path = np.zeros(n, dtype=int); path[n - 1] = s
    for t in range(n - 1, 0, -1):
        s = back[t, s]; path[t - 1] = s
    bnds = []
    for k in range(1, K):
        idx = np.where(path >= k)[0] if ordered else np.where(path == k)[0]
        bnds.append(float(times[idx[0]]) if len(idx) else float(times[-1]))
    return bnds


def dp_with_dur(times, P, core, post):
    """Run the constrained DP with overridden minimum-duration priors (core / post)."""
    bak = config.MIN_PHASE_DURATION
    config.MIN_PHASE_DURATION = {
        "PreCraniotomy": core, "Craniotomy": core, "ArterialFeederControl": core,
        "NidusDissection": core, "CraniotomyClosure": core, "PostClosure": post}
    try:
        return dp_segmentation(times, P)
    finally:
        config.MIN_PHASE_DURATION = bak


def pct_errors(bnds, expert, total_dur):
    errs = compute_timepoint_errors(bnds, expert)
    return [errs[ph]["absolute_error"] / total_dur * 100 for ph in errs]


# Praw = raw (config-B-filtered) per-frame probabilities; Psm = temporally smoothed.
METHODS = {
    "1_argmax":         lambda t, Praw, Psm: b_argmax(t, Praw),
    "2_argmax_smooth":  lambda t, Praw, Psm: b_argmax(t, Psm),
    "3_HMM_unordered":  lambda t, Praw, Psm: b_viterbi(t, Psm, ordered=False),
    "4_DP_no_mindur":   lambda t, Praw, Psm: dp_with_dur(t, Psm, 0, 0),
    "5_changepoint":    lambda t, Praw, Psm: b_changepoint(t, Psm),
    "6_constrainedDP":  lambda t, Praw, Psm: dp_with_dur(t, Psm, 60, 30),
}


def main():
    set_duration_source(config.SCENE_PRE_DIR)
    phase_times = parse_phase_times()
    okprobs = {f: load_okprobs(f) for f in range(1, 6)}
    acc = {m: [] for m in METHODS}
    for fold in range(1, 6):
        pred = Path(config.SCENE_PRE_DIR) / f"predictions_fold{fold}.csv"
        if not pred.exists():
            print(f"[WARN] {pred} not found, skipping fold {fold}")
            continue
        for case_str in config.FOLD_TEST_CASES[fold]:
            cn = int(case_str.replace("Case", ""))
            if cn not in phase_times:
                continue
            offsets, expert = build_surgery_timeline(cn, phase_times[cn])
            if offsets is None:
                continue
            times, probs, _ = build_timeline(pred, cn, offsets, okprobs[fold], THRESHOLD)
            if times is None or len(times) < 50:
                continue
            sm = smooth_probabilities(times, probs)
            durs = get_video_durations(cn)
            total = sum(durs[v] + 1 for v in sorted(durs)) - 1
            for m, fn in METHODS.items():
                try:
                    bnds = fn(times, probs, sm)
                    acc[m].extend(pct_errors(bnds, expert, total))
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
        print("No results; check input paths in config.py (predictions / okprobs / phase times).")
        return
    df = pd.DataFrame(rows).sort_values("overall_pctMAE").reset_index(drop=True)
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    out = Path(config.RESULT_DIR) / "baselines_comparison.csv"
    df.to_csv(out, index=False)
    print("=== Baseline comparison (configuration-B inputs, %MAE) ===")
    print(df.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
