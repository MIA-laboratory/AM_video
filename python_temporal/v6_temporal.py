"""
v6 unified temporal evaluation.

Given an existing scene-prediction CSV (Model 1 or Model 3) plus the cached
OK probabilities, build a timeline, optionally drop frames with OK prob
below `threshold`, smooth, run constrained DP, and compute %MAE.

This is invoked many times by run_v6.py (once per configuration × threshold).
The DP and CSV reading are CPU-light, so this script does NOT need a GPU.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import config
from temporal_analysis import (
    parse_phase_times, build_surgery_timeline,
    smooth_probabilities, dp_segmentation, compute_timepoint_errors,
    plot_surgery_timeline, get_video_durations, set_duration_source,
)


def load_okprobs(fold: int) -> dict[str, float]:
    cache_path = Path(config.CACHE_DIR) / f"okprobs_fold{fold}.parquet"
    if not cache_path.exists():
        raise FileNotFoundError(f"OK-prob cache missing: {cache_path}. "
                                f"Run precompute_okprobs.py first.")
    df = pd.read_parquet(cache_path)
    return dict(zip(df["path"].map(os.path.basename), df["ok_prob"]))


def build_timeline(pred_csv: Path, case_num: int, video_offsets: dict,
                   okprobs: dict | None, threshold: float):
    df = pd.read_csv(pred_csv)
    prob_cols = [f"prob_{c}" for c in config.SCENE_CLASSES]

    basenames = df["path"].apply(os.path.basename)
    pattern = r"Case(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.jpg"
    parsed = basenames.str.extract(pattern)
    parsed.columns = ["case", "vid", "h", "m", "s"]
    parsed = parsed.apply(pd.to_numeric, errors="coerce")

    mask = parsed["case"] == case_num
    df_c = df[mask].copy().reset_index(drop=True)
    p_c = parsed[mask].reset_index(drop=True)
    bn_c = basenames[mask].reset_index(drop=True)
    if df_c.empty:
        return None, None, (0, 0)

    times = []
    probs_list = []
    total = 0
    kept = 0
    for i in range(len(df_c)):
        vn = int(p_c.loc[i, "vid"]) if not pd.isna(p_c.loc[i, "vid"]) else None
        if vn not in video_offsets:
            continue
        total += 1
        if okprobs is not None:
            ok_p = okprobs.get(bn_c.loc[i], 1.0)
            if np.isnan(ok_p) or ok_p < threshold:
                continue
        kept += 1
        t = video_offsets[vn] + (int(p_c.loc[i, "h"]) * 3600
                                  + int(p_c.loc[i, "m"]) * 60
                                  + int(p_c.loc[i, "s"]))
        times.append(t)
        probs_list.append(df_c.loc[i, prob_cols].values.astype(float))

    if not times:
        return None, None, (total, kept)
    order = np.argsort(times)
    return np.array(times)[order], np.array(probs_list)[order], (total, kept)


def run_one_config(name: str, pred_dir: Path, use_okng: bool,
                   threshold: float, save_dir: Path, save_plots: bool = False,
                   config_key: str | None = None) -> pd.DataFrame:
    """`name` is the per-run directory name (may include threshold suffix).
    `config_key` is the base configuration key (without threshold)."""
    save_dir.mkdir(parents=True, exist_ok=True)
    # Derive split-video durations from this run's prediction CSVs.
    set_duration_source(pred_dir)
    phase_times = parse_phase_times()
    rows = []
    config_key = config_key or name

    okprobs_by_fold = {f: load_okprobs(f) for f in range(1, 6)} if use_okng else {f: None for f in range(1, 6)}

    for fold in range(1, 6):
        csv_path = pred_dir / f"predictions_fold{fold}.csv"
        if not csv_path.exists():
            print(f"  [{name}] missing {csv_path}, skipping fold {fold}")
            continue
        for case_str in config.FOLD_TEST_CASES[fold]:
            case_num = int(case_str.replace("Case", ""))
            if case_num not in phase_times:
                continue
            offsets, expert = build_surgery_timeline(case_num, phase_times[case_num])
            if offsets is None:
                continue
            times, probs, (n_tot, n_kept) = build_timeline(
                csv_path, case_num, offsets, okprobs_by_fold[fold], threshold)
            if times is None or len(times) < 50:
                print(f"  [{name}] case {case_num}: too few frames after filter "
                      f"(total={n_tot}, kept={n_kept}), skipping")
                continue
            smoothed = smooth_probabilities(times, probs)
            boundaries = dp_segmentation(times, smoothed)

            errors = compute_timepoint_errors(boundaries, expert)
            durs = get_video_durations(case_num)
            total_dur = sum(durs[v] + 1 for v in sorted(durs.keys())) - 1
            for pname, err in errors.items():
                rows.append({
                    "config": config_key, "threshold": threshold,
                    "fold": fold, "case": case_num, "phase": pname,
                    "signed_error": err["signed_error"],
                    "absolute_error": err["absolute_error"],
                    "predicted": err["predicted"], "expert": err["expert"],
                    "surgery_duration": total_dur,
                    "pct_error": err["absolute_error"] / total_dur * 100,
                    "frames_total": n_tot, "frames_kept": n_kept,
                })
            if save_plots:
                plot_surgery_timeline(
                    times, smoothed, boundaries, expert, case_num,
                    str(save_dir / f"timeline_case{case_num:02d}_fold{fold}.png"))

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(save_dir / "timepoint_errors.csv", index=False)
        summary = df.groupby("phase").agg(
            MAE=("absolute_error", "mean"),
            pctMAE=("pct_error", "mean"),
            median_pctMAE=("pct_error", "median"),
            n=("absolute_error", "count"),
        ).round(2)
        summary.to_csv(save_dir / "timepoint_error_summary.csv")
        overall = df["pct_error"].mean()
        median = df["pct_error"].median()
        print(f"  [{name}] overall %MAE = {overall:.3f}%  (median {median:.3f}%)")
    return df


CONFIGS_BASE = {
    "A_model1":       ("Model 1 (no OK/NG)",                    "pre",  False),
    "C_model3":       ("Model 3 (training-time OK/NG)",          "post", False),
    "B_model1_okng":  ("Model 1 + OK/NG inference filter",       "pre",  True),
    "D_model3_okng":  ("Model 3 + OK/NG inference filter",       "post", True),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS_BASE.keys()))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--save_plots", action="store_true",
                    help="Save per-case timeline PNGs (slow, do only for best config).")
    args = ap.parse_args()

    label, src_key, use_okng = CONFIGS_BASE[args.config]
    src_dir = Path(config.SCENE_PRE_DIR) if src_key == "pre" else Path(config.SCENE_POST_DIR)
    if use_okng:
        name = f"{args.config}_th{args.threshold:.2f}"
    else:
        name = args.config
    save_dir = Path(config.RESULT_DIR) / name

    print(f"=== {name} ({label}, threshold={args.threshold if use_okng else 'n/a'}) ===")
    run_one_config(name, src_dir, use_okng, args.threshold, save_dir,
                   save_plots=args.save_plots, config_key=args.config)


if __name__ == "__main__":
    main()
