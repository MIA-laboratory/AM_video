"""
Generate the best-performing-case temporal-segmentation figure (manuscript Figure 7).

(The script keeps its historical filename, generate_fig5.py, but produces the figure
numbered Figure 7 in the published manuscript.)

Layout of the manuscript timeline figure:
  - Top: smoothed scene probabilities (6 classes) vs. time.
  - Middle: DP-predicted phase band.
  - Middle: Expert (ground truth) phase band with %MAE labels at each transition.
  - Title carries Total Error (s) and Mean %MAE.

We use configuration B (Model 1 + inference-time OK/NG filter at θ=0.5), which gives
the lowest overall %MAE. Best-performing case under B is Case 16 (internal numbering),
labeled as "Case IX" under the manuscript Roman-numeral scheme.

Output: <AMVIDEO_FIGURES, default ./figures>/figure7_caseIX.png
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import config
from temporal_analysis import (
    parse_phase_times, build_surgery_timeline, smooth_probabilities,
    dp_segmentation, compute_timepoint_errors, get_video_durations,
    set_duration_source,
)
from v6_temporal import load_okprobs, build_timeline

# ── case under best configuration B ────────────────────────────────────
BEST_CASE = 16            # internal Case 16 → "Case IX" in v6 Roman scheme
BEST_FOLD = 1             # fold containing case 16
THRESHOLD = 0.50

# Internal case # → v6 Roman label
CASE_ROMAN = {2: "I", 3: "II", 8: "III", 9: "IV", 10: "V",
              11: "VI", 12: "VII", 15: "VIII", 16: "IX", 17: "X"}

PHASE_COLORS = {
    "PreCraniotomy":         "#5b9bd5",   # blue
    "Craniotomy":            "#ed7d31",   # orange
    "ArterialFeederControl": "#70ad47",   # green
    "NidusDissection":       "#7030a0",   # purple
    "CraniotomyClosure":     "#c00000",   # red
    "PostClosure":           "#2e75b6",   # teal-blue
}
PHASE_LABEL_LEGEND = {
    "PreCraniotomy":         "Pre-Craniotomy",
    "Craniotomy":            "Craniotomy",
    "ArterialFeederControl": "Art. Feeder Control",
    "NidusDissection":       "Nidus Dissection",
    "CraniotomyClosure":     "Cran. Closure",
    "PostClosure":           "Post-Closure",
}
PHASE_LABEL_BAND = {
    "PreCraniotomy":         "Pre-Cr",
    "Craniotomy":            "Craniotomy",
    "ArterialFeederControl": "Art. Feeder Ctrl",
    "NidusDissection":       "Nidus Dissection",
    "CraniotomyClosure":     "Cran. Closure",
    "PostClosure":           "Post",
}

PHASE_TO_TRANSITION_NAME = [
    "Craniotomy",            # boundary 0: end of PreCraniotomy / start of Craniotomy
    "ArterialFeederControl",
    "NidusDissection",
    "CraniotomyClosure",
    "PostClosure",
]


def compute_case_data(case_num: int, fold: int, threshold: float,
                      vis_smooth_window: int = 300):
    """Reproduce the v6 pipeline for one case and return everything needed to plot.

    `vis_smooth_window` (seconds) is an *additional* smoothing window applied
    only to the probability curves rendered in the top panel. The DP runs on
    the standard config.SMOOTHING_WINDOW-smoothed sequence, so boundary
    positions remain identical to the v6 numerical results.
    """
    phase_times = parse_phase_times()
    okprobs = load_okprobs(fold)
    set_duration_source(config.SCENE_PRE_DIR)
    pred_csv = Path(config.SCENE_PRE_DIR) / f"predictions_fold{fold}.csv"
    offsets, expert = build_surgery_timeline(case_num, phase_times[case_num])
    times, probs, _ = build_timeline(pred_csv, case_num, offsets, okprobs, threshold)
    smoothed = smooth_probabilities(times, probs)            # for DP
    smoothed_vis = smooth_probabilities(times, probs, window=vis_smooth_window)  # for plot
    boundaries = dp_segmentation(times, smoothed)
    errors = compute_timepoint_errors(boundaries, expert)
    durs = get_video_durations(case_num)
    total_dur = sum(durs[v] + 1 for v in sorted(durs.keys())) - 1
    return {
        "times": times, "smoothed": smoothed, "smoothed_vis": smoothed_vis,
        "boundaries": boundaries, "expert": expert, "errors": errors,
        "total_dur": total_dur,
    }


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}:{m:02d}"


def plot_figure(data: dict, case_label: str, out_path: Path) -> None:
    times = data["times"]
    # Use the visualization-only smoothed series for the top probability panel.
    smoothed = data.get("smoothed_vis", data["smoothed"])
    boundaries = data["boundaries"]
    expert_transitions = data["expert"]
    errors = data["errors"]
    total_dur = data["total_dur"]

    # Compute total absolute error (seconds) and mean %MAE
    abs_errors = [v["absolute_error"] for v in errors.values()]
    pct_errors = [v["absolute_error"] / total_dur * 100 for v in errors.values()]
    total_abs_err = int(sum(abs_errors))
    mean_pct = float(np.mean(pct_errors))

    # Expert boundaries in PHASE_TO_TRANSITION_NAME order (ignore phases without expert mark)
    expert_by_name = {name: tsec for tsec, name in expert_transitions}

    # ── Figure layout ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(14.0, 6.0))
    # Use a 3-row gridspec: probabilities (top), DP band, Expert band + labels (bottom)
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(3, 1, height_ratios=[3.0, 0.55, 1.20], hspace=0.22,
                           left=0.08, right=0.985, top=0.86, bottom=0.10)
    ax_p = fig.add_subplot(gs[0])
    ax_dp = fig.add_subplot(gs[1], sharex=ax_p)
    ax_ex = fig.add_subplot(gs[2], sharex=ax_p)

    classes = config.SCENE_CLASSES
    for k, cls in enumerate(classes):
        ax_p.plot(times, smoothed[:, k], color=PHASE_COLORS[cls],
                  linewidth=1.6, label=PHASE_LABEL_LEGEND[cls])
    ax_p.set_ylim(0, 1.05)
    ax_p.set_ylabel("Probability", fontsize=10)
    ax_p.tick_params(axis='x', labelbottom=False)
    ax_p.set_yticks([0.0, 0.5, 1.0])
    ax_p.grid(False)
    for spine in ('top', 'right'):
        ax_p.spines[spine].set_visible(False)

    # Legend across the top
    ax_p.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16),
                ncol=6, frameon=False, fontsize=9.5, handlelength=1.8)

    # Title above legend
    title = (f"Case {case_label} — Temporal Segmentation "
             f"(Total Error: {total_abs_err:,} s, Mean %MAE: {mean_pct:.2f}%)")
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.96)

    # ── DP-predicted band ──────────────────────────────────────────────
    dp_bounds = [times[0]] + list(boundaries) + [times[-1]]
    for k, cls in enumerate(classes):
        if k >= len(dp_bounds) - 1:
            break
        ax_dp.axvspan(dp_bounds[k], dp_bounds[k + 1],
                      ymin=0.15, ymax=0.85, color=PHASE_COLORS[cls], alpha=0.9, lw=0)
        mid = 0.5 * (dp_bounds[k] + dp_bounds[k + 1])
        if (dp_bounds[k + 1] - dp_bounds[k]) > total_dur * 0.02:
            ax_dp.text(mid, 0.5, PHASE_LABEL_BAND[cls],
                       ha='center', va='center', color='white',
                       fontsize=8, fontweight='bold')
    ax_dp.set_ylim(0, 1)
    ax_dp.set_yticks([])
    ax_dp.set_xlim(times[0], times[-1])
    ax_dp.text(-0.012, 0.5, "DP Prediction:", transform=ax_dp.transAxes,
               ha='right', va='center', fontsize=10, fontweight='bold')
    for spine in ax_dp.spines.values():
        spine.set_visible(False)
    ax_dp.tick_params(axis='x', which='both', length=0, labelbottom=False)

    # ── Expert band (faded) ────────────────────────────────────────────
    ex_bounds_in_order: list[float] = [times[0]]
    for name in PHASE_TO_TRANSITION_NAME:
        if name in expert_by_name:
            ex_bounds_in_order.append(expert_by_name[name])
    ex_bounds_in_order.append(times[-1])
    for k, cls in enumerate(classes):
        if k >= len(ex_bounds_in_order) - 1:
            break
        ax_ex.axvspan(ex_bounds_in_order[k], ex_bounds_in_order[k + 1],
                      ymin=0.65, ymax=1.0, color=PHASE_COLORS[cls], alpha=0.4, lw=0)
    ax_ex.set_ylim(0, 1)
    ax_ex.set_yticks([])
    ax_ex.set_xlim(times[0], times[-1])
    ax_ex.text(-0.012, 0.83, "Expert:", transform=ax_ex.transAxes,
               ha='right', va='center', fontsize=10, fontweight='bold')

    # Dashed connectors between DP and expert boundaries; %MAE label at expert
    for i, name in enumerate(PHASE_TO_TRANSITION_NAME):
        if i >= len(boundaries):
            break
        if name not in expert_by_name:
            continue
        dp_t = boundaries[i]
        ex_t = expert_by_name[name]
        # Dashed line spanning DP band to expert band on the lower axis
        ax_dp.axvline(dp_t, color='#444', linestyle='--', linewidth=0.7, alpha=0.7)
        ax_ex.axvline(dp_t, color='#444', linestyle='--', linewidth=0.7, alpha=0.7, ymin=0.65, ymax=1.0)
        ax_ex.axvline(ex_t, color='#222', linestyle=':', linewidth=0.7, alpha=0.7, ymin=0.65, ymax=1.0)
        pct = abs(dp_t - ex_t) / total_dur * 100
        ax_ex.text(ex_t, 0.30, f"{pct:.2f}%", ha='center', va='top',
                   color='#c00000', fontsize=9, fontweight='bold')

    # Bottom annotations
    ax_ex.text(1.0, 0.05, "%MAE at each transition boundary",
               transform=ax_ex.transAxes, ha='right', va='top',
               color='#c00000', fontsize=8.5, fontstyle='italic')
    ax_ex.text(0.5, -0.55, "Time (hours:minutes)",
               transform=ax_ex.transAxes, ha='center', va='top',
               color='#444', fontsize=10)

    # ── X-axis ticks (hours:minutes) ──────────────────────────────────
    n_hours = int(np.ceil(total_dur / 3600))
    hour_ticks = list(range(0, n_hours + 1))
    tick_positions = [h * 3600 for h in hour_ticks]
    tick_labels = [f"{h}:00" for h in hour_ticks]
    ax_ex.set_xticks(tick_positions)
    ax_ex.set_xticklabels(tick_labels, fontsize=9)
    for spine in ax_ex.spines.values():
        spine.set_visible(False)
    ax_ex.tick_params(axis='x', length=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  wrote {out_path}")
    print(f"  Case {case_label}: total error {total_abs_err}s, mean %MAE {mean_pct:.2f}%")
    print(f"  per-boundary %MAE: " +
          ", ".join(f"{name}={pct:.2f}%" for name, pct in zip(PHASE_TO_TRANSITION_NAME, pct_errors)))


def main() -> None:
    data = compute_case_data(BEST_CASE, BEST_FOLD, THRESHOLD, vis_smooth_window=600)
    case_label = CASE_ROMAN[BEST_CASE]
    out_path = Path(config.PAPER_DIR) / f"figure7_case{case_label}.png"
    plot_figure(data, case_label, out_path)


if __name__ == "__main__":
    main()
