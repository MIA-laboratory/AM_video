"""
Driver: run every temporal configuration (A/B/C/D) and, for the OK/NG-gated
configurations (B, D), sweep the OK/NG threshold theta. Reproduces the
per-configuration %MAE table and the OK/NG-threshold ablation reported in the paper.

Each individual run is delegated to v6_temporal.run_one_config (the same code path
as `python v6_temporal.py --config ...`), so the numbers are identical to single runs.

Configuration A and C are run once (no OK/NG gate). Configuration B and D are run for
theta in {0.3, 0.4, 0.5, 0.6, 0.7}.

Outputs (in AMVIDEO_RESULTS):
  run_v6_summary.csv   one row per (config, threshold): overall %MAE, median %MAE, n
and prints the same table to stdout.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

import config
from v6_temporal import CONFIGS_BASE, run_one_config

# OK/NG thresholds swept for the gated configurations (B, D); paper ablation grid.
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def main() -> None:
    summary_rows = []
    for cfg_key, (label, src_key, use_okng) in CONFIGS_BASE.items():
        src_dir = Path(config.SCENE_PRE_DIR) if src_key == "pre" else Path(config.SCENE_POST_DIR)
        thetas = THRESHOLDS if use_okng else [0.5]   # theta is ignored when use_okng is False
        for theta in thetas:
            name = f"{cfg_key}_th{theta:.2f}" if use_okng else cfg_key
            save_dir = Path(config.RESULT_DIR) / name
            print(f"=== {name} ({label}, threshold={theta if use_okng else 'n/a'}) ===")
            df = run_one_config(name, src_dir, use_okng, theta, save_dir, config_key=cfg_key)
            if df is None or df.empty:
                print(f"  [{name}] no results (missing inputs?)")
                continue
            overall = float(df["pct_error"].mean())
            median = float(df["pct_error"].median())
            summary_rows.append({
                "config": cfg_key, "label": label,
                "threshold": theta if use_okng else None,
                "overall_pctMAE": round(overall, 3),
                "median_pctMAE": round(median, 3),
                "n": int(len(df)),
            })

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        os.makedirs(config.RESULT_DIR, exist_ok=True)
        out = Path(config.RESULT_DIR) / "run_v6_summary.csv"
        summary.to_csv(out, index=False)
        print("\n=== run_v6 summary (overall / median %MAE) ===")
        print(summary.to_string(index=False))
        print(f"\nwrote {out}")
    else:
        print("No configurations produced results; check the input paths in config.py.")


if __name__ == "__main__":
    main()
