"""
Surgery-level temporal analysis and DP-based scene transition estimation.

Reads per-frame class-probability CSVs from a configurable directory. The CSVs
are exported by the frame classifier (in this study, the Inception-ResNet-v2
networks trained in MATLAB); this temporal stage is classifier-agnostic and only
needs columns `path` and `prob_<ClassName>` for each of the six scene classes.

Optimized: vectorized smoothing, GPU-accelerated DP via PyTorch.
"""
import os
import re
import glob
import datetime
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openpyxl

import config


# ── Phase time parsing ─────────────────────────────────────────────────

def parse_phase_times(xlsx_path=config.PHASE_TIME_XLSX):
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb[wb.sheetnames[0]]
    phase_times = {}
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, values_only=True):
        if row[0] is None or row[2] is None:
            continue
        parts = str(row[0]).split("-")
        case_num = int(parts[0])
        video_num = int(parts[1])
        t = row[1]
        if isinstance(t, datetime.time):
            seconds = t.hour * 3600 + t.minute * 60 + t.second
        elif isinstance(t, datetime.timedelta):
            seconds = int(t.total_seconds())
        else:
            seconds = int(t)
        phase_name = str(row[2]).strip()
        if case_num not in phase_times:
            phase_times[case_num] = []
        phase_times[case_num].append((video_num, seconds, phase_name))
    for case_num in phase_times:
        phase_times[case_num].sort(key=lambda x: (x[0], x[1]))
    return phase_times


# The driver sets this to the directory holding predictions_fold*.csv so that
# split-video lengths can be derived directly from the frame filenames in the
# prediction CSVs (no access to the raw frame folders / patient data required).
_DURATION_SOURCE_DIR = None


def set_duration_source(pred_dir):
    """Point video-duration discovery at a directory of predictions_fold*.csv."""
    global _DURATION_SOURCE_DIR
    _DURATION_SOURCE_DIR = str(pred_dir)


def get_video_durations(case_num):
    """Per-split-video length (seconds) for a case, from prediction-CSV frame names.

    Each frame filename is ``CaseXXYY_HHMMSS.jpg`` (case XX, split-video YY,
    in-video time HHMMSS); the duration of split-video YY is the max HHMMSS seen.
    """
    durations = {}
    case_prefix = f"Case{case_num:02d}"
    src = _DURATION_SOURCE_DIR or config.SCENE_PRE_DIR
    for csv_path in sorted(glob.glob(os.path.join(src, "predictions_fold*.csv"))):
        try:
            paths = pd.read_csv(csv_path, usecols=["path"])["path"]
        except (ValueError, FileNotFoundError):
            continue
        for p in paths:
            fname = os.path.basename(str(p))
            if not fname.startswith(case_prefix):
                continue
            m = re.match(r"Case(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.jpg", fname)
            if m:
                vid_num = int(m.group(2))
                h, mi, s = int(m.group(3)), int(m.group(4)), int(m.group(5))
                t_sec = h * 3600 + mi * 60 + s
                if vid_num not in durations or t_sec > durations[vid_num]:
                    durations[vid_num] = t_sec
    return durations


def build_surgery_timeline(case_num, phase_times_entry):
    durations = get_video_durations(case_num)
    if not durations:
        return None, None
    video_nums = sorted(durations.keys())
    video_offsets = {}
    offset = 0
    for vn in video_nums:
        video_offsets[vn] = offset
        offset += durations[vn] + 1
    expert_transitions = []
    for vid_num, time_sec, phase_name in phase_times_entry:
        if vid_num in video_offsets:
            global_time = video_offsets[vid_num] + time_sec
            expert_transitions.append((global_time, phase_name))
    return video_offsets, expert_transitions


# ── Prediction timeline reconstruction ─────────────────────────────────

_FILENAME_RE = re.compile(r"Case(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.jpg")


def parse_filename(fname):
    basename = os.path.basename(fname)
    m = _FILENAME_RE.match(basename)
    if not m:
        return None, None, None
    case_num = int(m.group(1))
    vid_num = int(m.group(2))
    h, mi, s = int(m.group(3)), int(m.group(4)), int(m.group(5))
    return case_num, vid_num, h * 3600 + mi * 60 + s


def build_prediction_timeline(pred_csv_path, target_case_num, video_offsets):
    df = pd.read_csv(pred_csv_path)
    prob_cols = [f"prob_{c}" for c in config.SCENE_CLASSES]
    basenames = df["path"].apply(os.path.basename)
    pattern = r"Case(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.jpg"
    parsed = basenames.str.extract(pattern)
    parsed.columns = ["case", "vid", "h", "m", "s"]
    parsed = parsed.apply(pd.to_numeric, errors="coerce")
    mask = parsed["case"] == target_case_num
    df_case = df[mask].copy()
    p_case = parsed[mask]
    if df_case.empty:
        return None, None
    vid_nums = p_case["vid"].values
    t_secs = p_case["h"].values * 3600 + p_case["m"].values * 60 + p_case["s"].values
    times = []
    valid = []
    for i, (vn, ts) in enumerate(zip(vid_nums, t_secs)):
        if vn in video_offsets:
            times.append(video_offsets[vn] + ts)
            valid.append(i)
    if not times:
        return None, None
    times = np.array(times)
    probs = df_case.iloc[valid][prob_cols].values
    order = np.argsort(times)
    return times[order], probs[order]


# ── Temporal smoothing ─────────────────────────────────────────────────

def smooth_probabilities(times, probs, window=config.SMOOTHING_WINDOW):
    from scipy.ndimage import uniform_filter1d
    dt = np.median(np.diff(times)) if len(times) > 1 else 1.0
    kernel = max(1, int(round(window / dt)))
    if kernel % 2 == 0:
        kernel += 1
    smoothed = uniform_filter1d(probs, size=kernel, axis=0, mode="nearest")
    row_sums = smoothed.sum(axis=1, keepdims=True)
    smoothed = smoothed / (row_sums + 1e-8)
    return smoothed


# ── Constrained DP (GPU-accelerated) ───────────────────────────────────

def dp_segmentation(times, probs):
    """
    Find K-1 = 5 boundaries b0 < b1 < ... < b4 splitting n frames into K=6 phases.
    Phase k occupies [b_{k-1}, b_k) with b_{-1}=0 and b_5=n.
    Score = sum_k sum_{t in phase_k} log p_k(t), maximized subject to min-duration.

    Uses GPU-vectorized DP on ~500 subsampled candidate positions.
    Returns: list of 5 boundary times.
    """
    n = len(times)
    K = config.NUM_SCENE_CLASSES
    min_dur = np.array([config.MIN_PHASE_DURATION[c] for c in config.SCENE_CLASSES])
    NEG_INF = -1e15
    # Minimum-duration constraint (paper Sec. 2.8.2): each phase spans at least D_k.
    # It is enforced below as ">= min_dur - 1" — a 1-second tolerance so that a segment
    # landing exactly on D_k is not spuriously rejected by the ~500-position boundary
    # subsampling / float rounding. The %MAE results are invariant to this tolerance
    # (see the minimum-duration sensitivity table in the paper).

    # Subsample
    step = max(1, n // 500)
    idx = np.arange(0, n, step)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    m = len(idx)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    # Cumulative log-probs: cum[i, k] = sum of log_probs[0:i, k]
    log_p = np.log(probs + 1e-8)
    cum = np.zeros((n + 1, K))
    cum[1:] = np.cumsum(log_p, axis=0)
    cum_t = torch.tensor(cum, dtype=torch.float32, device=device)
    idx_t = torch.tensor(idx, dtype=torch.long, device=device)
    t_sub = torch.tensor(times[idx], dtype=torch.float32, device=device)

    # seg_score(a, b, k) = cum[b, k] - cum[a, k]  (frames [a, b))

    # --- Boundary 0: phase 0 occupies [0, idx[j]) ---
    # dp[j] = score of phase 0 from frame 0 to frame idx[j]
    dp = cum_t[idx_t, 0] - cum_t[0, 0]   # (m,)
    dp = torch.where(t_sub - t_sub[0] >= min_dur[0] - 1, dp, NEG_INF)
    bp_list = []

    # --- Boundaries 1..3 (phases 1..3, intermediate) ---
    for b in range(1, K - 2):
        # Phase b occupies [idx[i], idx[j]).  i = prev boundary pos, j = new boundary pos.
        # seg[j, i] = cum[idx[j], b] - cum[idx[i], b]
        seg = cum_t[idx_t, b].unsqueeze(1) - cum_t[idx_t, b].unsqueeze(0)  # (m, m)
        cand = dp.unsqueeze(0) + seg  # (m, m): cand[j, i]

        dur = t_sub.unsqueeze(1) - t_sub.unsqueeze(0)  # (m, m): dur[j, i]
        ii = torch.arange(m, device=device).unsqueeze(0)  # (1, m) col
        ji = torch.arange(m, device=device).unsqueeze(1)  # (m, 1) row
        mask = (ii < ji) & (dur >= min_dur[b] - 1)
        cand = torch.where(mask, cand, NEG_INF)

        dp, best_i = cand.max(dim=1)  # max over i (columns)
        bp_list.append(best_i)

    # --- Boundary K-2 = 4: phase K-2 occupies [idx[i], idx[j]), phase K-1 occupies [idx[j], n) ---
    b_mid = K - 2   # phase 4 = CraniotomyClosure
    b_last = K - 1  # phase 5 = PostClosure

    seg_mid = cum_t[idx_t, b_mid].unsqueeze(1) - cum_t[idx_t, b_mid].unsqueeze(0)  # (m,m)
    seg_last = cum_t[n, b_last] - cum_t[idx_t, b_last]  # (m,): last phase score from j to n

    cand = dp.unsqueeze(0) + seg_mid + seg_last.unsqueeze(1)  # (m, m)

    dur = t_sub.unsqueeze(1) - t_sub.unsqueeze(0)
    dur_last = t_sub[-1] - t_sub  # (m,)
    ii = torch.arange(m, device=device).unsqueeze(0)
    ji = torch.arange(m, device=device).unsqueeze(1)
    mask = ((ii < ji)
            & (dur >= min_dur[b_mid] - 1)
            & (dur_last.unsqueeze(1).expand(m, m) >= min_dur[b_last] - 1))
    cand = torch.where(mask, cand, NEG_INF)

    final_scores, best_i = cand.max(dim=1)
    bp_list.append(best_i)

    # --- Backtrack ---
    best_j = final_scores.argmax().item()
    path = [best_j]  # last boundary (boundary 4)
    cur = best_j
    for bp in reversed(bp_list):
        cur = bp[cur].item()
        path.append(cur)
    path.reverse()
    # path = [boundary0_sub, boundary1_sub, boundary2_sub, boundary3_sub, boundary4_sub]
    # = K-1 = 5 entries

    return [float(times[idx[s]]) for s in path]


# ── Error analysis ─────────────────────────────────────────────────────

def compute_timepoint_errors(predicted, expert):
    phase_to_transition = {
        "Craniotomy": 0,
        "ArterialFeederControl": 1,
        "NidusDissection": 2,
        "CraniotomyClosure": 3,
        "PostClosure": 4,
    }
    errors = {}
    for global_time, phase_name in expert:
        if phase_name not in phase_to_transition:
            continue
        idx = phase_to_transition[phase_name]
        if idx < len(predicted):
            signed_err = predicted[idx] - global_time
            errors[phase_name] = {
                "signed_error": signed_err,
                "absolute_error": abs(signed_err),
                "predicted": predicted[idx],
                "expert": global_time,
            }
    return errors


# ── Visualization ──────────────────────────────────────────────────────

def plot_surgery_timeline(times, probs, boundaries, expert_transitions, case_num, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    ax = axes[0]
    for k, cls_name in enumerate(config.SCENE_CLASSES):
        ax.plot(times, probs[:, k], color=colors[k], alpha=0.7, label=cls_name)
    for i, bt in enumerate(boundaries):
        ax.axvline(bt, color="red", linestyle="--", alpha=0.5,
                   label="Predicted" if i == 0 else "")
    for i, (gt, pname) in enumerate(expert_transitions):
        ax.axvline(gt, color="black", linestyle="-", alpha=0.5,
                   label="Expert" if i == 0 else "")
    ax.set_ylabel("Probability")
    ax.set_title(f"Case {case_num} - Surgery-level Scene Probabilities")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1)

    ax2 = axes[1]
    all_bounds = [times[0]] + boundaries + [times[-1]]
    for k in range(min(len(all_bounds) - 1, len(colors))):
        ax2.axvspan(all_bounds[k], all_bounds[k + 1], color=colors[k], alpha=0.6)
    for i, (gt, pname) in enumerate(expert_transitions):
        ax2.axvline(gt, color="black", linewidth=2,
                    label="Expert" if i == 0 else "")
    ax2.set_xlabel("Surgery time (seconds)")
    ax2.set_ylabel("Phase")
    ax2.set_yticks([])
    ax2.set_title("Phase Segmentation")
    for k in range(min(len(all_bounds) - 1, len(config.SCENE_CLASSES))):
        mid = (all_bounds[k] + all_bounds[k + 1]) / 2
        ax2.text(mid, 0.5, config.SCENE_CLASSES[k], ha="center", va="center",
                 fontsize=7, rotation=30, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ── Main pipeline ──────────────────────────────────────────────────────

def run_temporal_analysis(pred_csv_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    phase_times = parse_phase_times()
    all_errors = []

    for fold in range(1, 6):
        csv_path = os.path.join(pred_csv_dir, f"predictions_fold{fold}.csv")
        if not os.path.exists(csv_path):
            print(f"[WARN] {csv_path} not found, skipping fold {fold}")
            continue

        test_cases = config.FOLD_TEST_CASES[fold]
        for case_str in test_cases:
            case_num = int(case_str.replace("Case", ""))
            if case_num not in phase_times:
                print(f"  [WARN] No phase times for Case{case_num}, skipping")
                continue

            video_offsets, expert_transitions = build_surgery_timeline(
                case_num, phase_times[case_num])
            if video_offsets is None:
                continue

            times, probs = build_prediction_timeline(csv_path, case_num, video_offsets)
            if times is None:
                print(f"  [WARN] No predictions for Case{case_num} in fold {fold}")
                continue

            smoothed = smooth_probabilities(times, probs)
            boundaries = dp_segmentation(times, smoothed)

            errors = compute_timepoint_errors(boundaries, expert_transitions)
            for phase_name, err in errors.items():
                all_errors.append({"fold": fold, "case": case_num, "phase": phase_name, **err})

            plot_surgery_timeline(
                times, smoothed, boundaries, expert_transitions, case_num,
                os.path.join(save_dir, f"timeline_case{case_num:02d}_fold{fold}.png"))
            print(f"  Fold {fold}, Case {case_num}: {len(boundaries)} boundaries estimated")

    if all_errors:
        err_df = pd.DataFrame(all_errors)
        err_df.to_csv(os.path.join(save_dir, "timepoint_errors.csv"), index=False)
        summary = err_df.groupby("phase").agg(
            mean_signed=("signed_error", "mean"),
            std_signed=("signed_error", "std"),
            mean_absolute=("absolute_error", "mean"),
            std_absolute=("absolute_error", "std"),
            median_absolute=("absolute_error", "median"),
            n=("signed_error", "count"),
        ).round(1)
        summary.to_csv(os.path.join(save_dir, "timepoint_error_summary.csv"))
        print(f"\nTime-point error summary:\n{summary}")
    return all_errors


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    args = parser.parse_args()
    run_temporal_analysis(args.pred_dir, args.save_dir)
