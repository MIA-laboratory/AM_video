# AM_video — Constrained-DP Temporal Reconstruction for AVM Resection Videos

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.x-blue.svg)

Source code for the **temporal reconstruction** stage of an automated surgical-phase
analysis pipeline for arteriovenous malformation (AVM) resection videos.

Given per-frame surgical-scene class probabilities (produced by a frame classifier —
in our study, **Inception-ResNet-v2** networks trained in MATLAB), this code
reconstructs a temporally consistent surgical timeline using a **constrained dynamic
programming (DP)** algorithm and evaluates phase-boundary accuracy with a
duration-normalized **percentage mean absolute error (%MAE)** metric.

> **This repository implements the methodology, not a bundled model.** It contains the
> temporal-analysis code only. You bring your own frame classifier and (optionally) your
> own OK/NG model — built in **MATLAB (`.mat`)** or **Python** — and feed their outputs
> in through the documented CSV/parquet formats. No models or data are included.

---

## What this code does

1. Reads per-frame class-probability CSVs (one row per 1-fps frame).
2. Reconstructs each surgery's timeline (concatenating split-video segments).
3. Smooths the probabilities with a sliding window and renormalizes.
4. Runs a **constrained DP** that finds the 5 phase boundaries maximizing total
   log-likelihood subject to (i) fixed phase ordering and (ii) minimum-duration
   constraints. Every predicted boundary therefore coincides with a transition in the
   resulting phase map (direct interpretability). The DP is GPU-vectorized over ~500
   candidate positions (falls back to CPU).
5. Optionally applies an **inference-time OK/NG gate**: drops frames whose OK
   probability is below a threshold before smoothing (configs B and D).
6. Computes per-phase and overall %MAE against expert phase-time annotations.

```
python_temporal/
  temporal_analysis.py   constrained DP, smoothing, %MAE, timeline plotting
  v6_temporal.py         per-configuration driver / CLI (A / B / C / D)
  generate_fig5.py       best-case timeline figure
  config.py              scene classes, DP constraints, configurable I/O paths
okng/                    bring-your-own OK/NG model: export OK-probability tables
  export_okprobs_matlab.m   from a MATLAB .mat OK/NG model
  export_okprobs_python.py  from a Python OK/NG model
```

---

## Inputs

All paths are configurable via environment variables (see `config.py`).

**1. Per-frame prediction CSVs** — `predictions_fold{1..5}.csv` in the Model 1
directory (`AMVIDEO_PRED_MODEL1`) and, for Model 3 configs, the Model 3 directory
(`AMVIDEO_PRED_MODEL3`). Required columns:

| column | meaning |
|--------|---------|
| `path` | frame filename ending in `CaseXXYY_HHMMSS.jpg` (case `XX`, split-video `YY`, in-video time `HHMMSS`) |
| `prob_PreCraniotomy` … `prob_PostClosure` | softmax probability for each of the six scene classes |

Extra columns (e.g. `true_label`, `pred_label`) are ignored. Split-video durations are
derived from the frame filenames in these CSVs — **no access to the raw frames is
required**.

**2. Expert phase-time annotations** — `AMVIDEO_PHASE_TIMES` (xlsx): per case, the start
time (seconds) of each phase from Craniotomy onward.

**3. (Optional) OK probabilities** for configs B/D — `okprobs_fold{1..5}.csv` **or**
`.parquet` in `AMVIDEO_OKPROB_DIR`, with columns `path` and `ok_prob` (the probability
each frame is OK, 0..1). This is the output of your own OK/NG model — see below.

---

## Usage

```bash
cd python_temporal

export AMVIDEO_PRED_MODEL1=/path/to/model1_predictions     # predictions_fold*.csv
export AMVIDEO_PRED_MODEL3=/path/to/model3_predictions     # (for configs C/D)
export AMVIDEO_PHASE_TIMES=/path/to/phase_times.xlsx
export AMVIDEO_OKPROB_DIR=/path/to/okprobs                 # (for configs B/D)
export AMVIDEO_RESULTS=/path/to/output
export AMVIDEO_DEVICE=cpu                                  # or cuda:0

# Configuration A: Model 1 predictions, no OK/NG gate
python v6_temporal.py --config A_model1

# Configuration B: Model 1 + inference-time OK/NG gate
python v6_temporal.py --config B_model1_okng --threshold 0.5 --save_plots
```

Configs: `A_model1`, `B_model1_okng`, `C_model3`, `D_model3_okng`. Each writes
`timepoint_errors.csv` and `timepoint_error_summary.csv` (per-phase MAE and %MAE) and
prints the overall %MAE.

---

## Bring your own OK/NG model

The inference-time OK/NG gate (configs B / D) is **model-agnostic**: it only needs, per
fold, a table of per-frame OK probabilities placed in `AMVIDEO_OKPROB_DIR`:

```
okprobs_fold{1..5}.csv   (or .parquet)
   path      frame filename ending in CaseXXYY_HHMMSS.jpg
   ok_prob   probability the frame is OK (usable), in [0, 1]
```

Produce that table from **your own** OK/NG model — the templates in `okng/` handle the
file-walking and CSV writing; you fill in model loading and per-frame prediction:

- **MATLAB `.mat` model** → [`okng/export_okprobs_matlab.m`](okng/export_okprobs_matlab.m)
- **Python model** → [`okng/export_okprobs_python.py`](okng/export_okprobs_python.py)

Configs A and C (no OK/NG gate) do not need this and run with only the prediction CSVs.

### Requirements

`numpy`, `pandas`, `torch`, `matplotlib`, `scipy`, `openpyxl`, `pyarrow`.
A GPU is optional.

---

## Scope and reproducibility

This is the temporal-analysis component only and is **independent of any particular
frame classifier** — it consumes class-probability CSVs. In the associated study the
probabilities were produced by Inception-ResNet-v2; running this code on those
predictions reproduces the corresponding temporal results.

## License

Released under the [MIT License](LICENSE).

## Citation

If you use this code, please cite the associated *Applied Sciences* (2026) article on
automated scene classification and explainable time-point estimation for AVM resection
videos.

## Acknowledgements

Supported in part by AMED (Grant Number JP256f0137006) and the Cabinet Office,
Government of Japan, Regional University and Regional Industry Creation Grant Program.

## Contact

MIA-laboratory, Hokkaido University. For questions or model-access requests, please
contact the corresponding author of the associated article.
