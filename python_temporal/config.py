"""
Configuration for the constrained-DP temporal reconstruction stage.

This stage consumes per-frame class-probability CSVs produced by the frame
classifier (in this study, Inception-ResNet-v2 networks trained in MATLAB) and
the expert phase-time annotations. It is classifier-agnostic: any classifier
that exports the expected CSV columns can drive it.

All paths default to sub-folders next to this file and can be overridden with
environment variables (shown in brackets).
"""
import os

# ── Base / I/O paths ───────────────────────────────────────────────────
BASE_DIR = os.environ.get("AMVIDEO_BASE", os.path.dirname(os.path.abspath(__file__)))

# Per-frame prediction CSVs: predictions_fold{1..5}.csv.
#   Model 1 = all-frame 6-class predictions (source for temporal reconstruction)
#   Model 3 = 6-class predictions from the OK-filtered classifier
# Required columns: `path`, and `prob_<ClassName>` for each class in SCENE_CLASSES.
SCENE_PRE_DIR  = os.environ.get("AMVIDEO_PRED_MODEL1",
                                os.path.join(BASE_DIR, "predictions_model1"))   # Model 1
SCENE_POST_DIR = os.environ.get("AMVIDEO_PRED_MODEL3",
                                os.path.join(BASE_DIR, "predictions_model3"))   # Model 3

# Expert phase-time annotations (per case: phase start times in seconds).
PHASE_TIME_XLSX = os.environ.get("AMVIDEO_PHASE_TIMES",
                                 os.path.join(BASE_DIR, "supervised_phaseTime.xlsx"))

# Cached OK probabilities for the inference-time OK/NG gate (configs B / D):
#   okprobs_fold{1..5}.parquet with columns `path`, `ok_prob` (OK/NG model output).
CACHE_DIR = os.environ.get("AMVIDEO_OKPROB_DIR", os.path.join(BASE_DIR, "okprobs"))

# Outputs.
RESULT_DIR = os.environ.get("AMVIDEO_RESULTS", os.path.join(BASE_DIR, "results"))
PAPER_DIR  = os.environ.get("AMVIDEO_FIGURES", os.path.join(BASE_DIR, "figures"))

# ── Scene classes (fixed surgical order) ───────────────────────────────
SCENE_CLASSES = [
    "PreCraniotomy",
    "Craniotomy",
    "ArterialFeederControl",
    "NidusDissection",
    "CraniotomyClosure",
    "PostClosure",
]
NUM_SCENE_CLASSES = len(SCENE_CLASSES)

# ── Cross-validation: held-out test cases per fold ─────────────────────
FOLD_TEST_CASES = {
    1: ["Case16", "Case17"],
    2: ["Case12", "Case15"],
    3: ["Case10", "Case11"],
    4: ["Case08", "Case09"],
    5: ["Case02", "Case03"],
}

# ── Constrained-DP parameters ──────────────────────────────────────────
SMOOTHING_WINDOW = 30          # seconds; sliding-window probability smoothing
MIN_PHASE_DURATION = {         # minimum phase duration (seconds) — DP constraint
    "PreCraniotomy": 60,
    "Craniotomy": 60,
    "ArterialFeederControl": 60,
    "NidusDissection": 60,
    "CraniotomyClosure": 60,
    "PostClosure": 30,
}

# ── Compute ────────────────────────────────────────────────────────────
DEVICE = os.environ.get("AMVIDEO_DEVICE", "cuda:0")   # falls back to CPU automatically
