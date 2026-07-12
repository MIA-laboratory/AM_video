# AM_video — Non-monotonic HSMM Temporal Reconstruction for AVM Resection Videos

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.x-blue.svg)

Source code for the **temporal reconstruction** stage of an automated surgical-phase
analysis pipeline for arteriovenous malformation (AVM) resection videos.

Given per-frame surgical-scene class probabilities (produced by a frame classifier —
in our study, **Inception-ResNet-v2** networks fine-tuned in **PyTorch**), this code
reconstructs a temporally consistent surgical timeline with a **non-monotonic hidden
semi-Markov model (HSMM)** and evaluates phase-boundary accuracy with a
duration-normalized **percentage mean absolute error (%MAE)** metric.

The HSMM combines (i) a **data-driven phase-to-phase transition matrix** estimated from
the annotated workflow — with **no fixed one-way order**, so the decoded timeline may
revisit an earlier phase when the imagery supports it — (ii) explicit **per-phase
minimum durations** (semi-Markov), and (iii) a constant **switch penalty** that
suppresses over-segmentation. Every predicted time point is, by construction, a
boundary of the final phase map, so a surgeon can visually verify each transition.

> **This repository implements the methodology, not a bundled model.** It contains the
> temporal-analysis code only. You bring your own frame classifier and (optionally) your
> own OK/NG model — built in **PyTorch** or **MATLAB** — and feed their outputs in through
> the documented CSV/parquet formats. No models or data are included.

---

## What this code does

1. Reads per-frame class-probability CSVs (one row per 1-fps frame).
2. Reconstructs each surgery's timeline (concatenating split-video segments).
3. Smooths the probabilities with a sliding window and renormalizes (Eq. 1).
4. Learns a phase-to-phase transition matrix from the expert phase sequences of the
   **training cases only, per fold** (no leakage from held-out cases). Self-transitions
   are removed and never-observed transitions receive a small floor ε (Eq. 3).
5. Decodes with a **non-monotonic semi-Markov segmental DP** (Eqs. 2, 4): each segment
   scores its smoothed log-emission plus the log-transition into it, minus the switch
   penalty λ, subject to per-phase minimum durations. The first segment is fixed to
   Pre-Craniotomy. The phase onset is read as the first frame decoded to each phase
   (first-entry). Decoding runs over a few hundred subsampled candidate positions and is
   CPU-light.
6. Optionally applies an **inference-time OK/NG gate**: drops frames whose OK
   probability is below a threshold before smoothing (configs B and D).
7. Computes per-phase and overall %MAE against expert phase-time annotations.

```
python_temporal/
  temporal_analysis.py   non-monotonic HSMM decoder, transition learning, smoothing, %MAE
  v6_temporal.py         single-configuration evaluator / CLI (A / B / C / D)
  run_v6.py              driver: sweeps all configs x OK/NG thresholds
  baselines.py           temporal-model baseline comparison (manuscript Table 10)
  generate_fig5.py       best-case timeline figure (manuscript Figure 7 = Case XII)
  config.py              scene classes, HSMM parameters, cross-validation folds, I/O paths
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
time (seconds) of each phase. Rows are `CC-VV` (case–video), start time, and the phase
name (matching `config.SCENE_CLASSES`). Used both to score %MAE and to learn the
per-fold transition matrix.

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

# Configuration A: Model 1 predictions, non-monotonic HSMM, no OK/NG gate (headline)
python v6_temporal.py --config A_model1 --save_plots

# Configuration B: Model 1 + inference-time OK/NG gate
python v6_temporal.py --config B_model1_okng --threshold 0.5
```

Configs: `A_model1`, `B_model1_okng`, `C_model3`, `D_model3_okng`. Each writes
`timepoint_errors.csv` and `timepoint_error_summary.csv` (per-phase MAE and %MAE) and
prints the overall %MAE.

Run the full configuration × OK/NG-threshold sweep (reproduces the per-configuration
%MAE table and the threshold ablation) with:

```bash
python run_v6.py
```

Reproduce the temporal-model baseline comparison (the proposed non-monotonic HSMM vs.
frame-wise argmax, argmax+smoothing, an unordered HMM/Viterbi decoder, and change-point
detection) on identical Model-1 inputs with:

```bash
python baselines.py
```

Regenerate the best-case timeline figure (manuscript Figure 7, Case XII) with:

```bash
python generate_fig5.py
```

---

## Results reproduced by this code

On the 20-case cohort (499,744 frames at 1 fps; mean surgery duration 7.0 h; five-fold
video-level cross-validation), decoding **Model-1 probabilities with the non-monotonic
HSMM and no OK/NG gate (configuration A)** is the headline operating point:

| configuration | overall %MAE |
|---|---|
| **A — Model 1, no gate (HSMM)** | **3.27%** (median 0.78%, n = 69) |
| B — Model 1 + OK/NG gate (θ=0.5) | 3.86% |
| C — Model 3 (training-time OK/NG) | 3.95% |
| D — Model 3 + OK/NG gate | 3.95% |

Per-phase %MAE under configuration A: Craniotomy 0.53%, Arterial-Feeder-Control 0.95%,
Nidus-Dissection 9.32% (the gradual, visually ambiguous transition), Craniotomy-Closure
1.12%, Post-Closure 0.40%. Unlike the original ten-case study, the inference-time OK/NG
gate did **not** improve boundary localization on the 20-case cohort at any threshold, so
configuration A (no gate) is used for all reported analyses.

Baseline comparison on identical Model-1 inputs (manuscript Table 10). The proposed HSMM
reports first-entry onsets only where it decodes each phase (n = 69); the baselines never
abstain, so they are scored on all GT-present boundaries (n = 91):

| decoder | overall %MAE | n |
|---|---|---|
| **Non-monotonic HSMM (proposed)** | **3.27%** | 69 |
| Change-point detection (K=5) | 15.0% | 91 |
| Unordered HMM / Viterbi | 22.1% | 91 |
| Argmax + smoothing | 32.7% | 91 |
| Frame-wise argmax | 46.2% | 91 |

The clinically structured HSMM far outperforms the unordered decoders — confirming that
the workflow structure, not the decoding machinery, is the decisive contribution — and,
unlike a strictly ordered decoder, it additionally represents the phase returns observed
in 3 of the 20 cases.

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

- **Python model** → [`okng/export_okprobs_python.py`](okng/export_okprobs_python.py)
- **MATLAB `.mat` model** → [`okng/export_okprobs_matlab.m`](okng/export_okprobs_matlab.m)

Configs A and C (no OK/NG gate) do not need this and run with only the prediction CSVs.

### Requirements

`numpy`, `pandas`, `scipy`, `matplotlib`, `openpyxl`, `pyarrow`. No GPU is required for
this temporal stage.

---

## Scope and reproducibility

This is the temporal-analysis component only and is **independent of any particular
frame classifier** — it consumes class-probability CSVs. In the associated study the
probabilities were produced by Inception-ResNet-v2 fine-tuned in PyTorch; running this
code on those predictions reproduces the corresponding temporal results.

## License

Released under the [MIT License](LICENSE).

## Citation

If you use this code, please cite the associated *Applied Sciences* (2026) article on
automated scene classification and interpretable time-point estimation for AVM resection
videos.

## Acknowledgements

Supported in part by AMED (Grant Number JP256f0137006) and the Cabinet Office,
Government of Japan, Regional University and Regional Industry Creation Grant Program.

## Contact

MIA-laboratory, Hokkaido University. For questions or model-access requests, please
contact the corresponding author of the associated article.
