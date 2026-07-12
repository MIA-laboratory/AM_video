"""
Configuration for the non-monotonic HSMM temporal-reconstruction stage.

This stage consumes per-frame class-probability CSVs produced by the frame
classifier (in this study, Inception-ResNet-v2 networks fine-tuned in PyTorch)
and the expert phase-time annotations. It is classifier-agnostic: any classifier
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
                                 os.path.join(BASE_DIR, "phase_times.xlsx"))

# Cached OK probabilities for the inference-time OK/NG gate (configs B / D):
#   okprobs_fold{1..5}.parquet with columns `path`, `ok_prob` (OK/NG model output).
CACHE_DIR = os.environ.get("AMVIDEO_OKPROB_DIR", os.path.join(BASE_DIR, "okprobs"))

# Outputs.
RESULT_DIR = os.environ.get("AMVIDEO_RESULTS", os.path.join(BASE_DIR, "results"))
PAPER_DIR  = os.environ.get("AMVIDEO_FIGURES", os.path.join(BASE_DIR, "figures"))

# ── Scene classes ──────────────────────────────────────────────────────
# Listed in the natural operative order; the HSMM does NOT impose this order
# (its transition matrix is data-driven and may revisit an earlier phase).
SCENE_CLASSES = [
    "PreCraniotomy",
    "Craniotomy",
    "ArterialFeederControl",
    "NidusDissection",
    "CraniotomyClosure",
    "PostClosure",
]
NUM_SCENE_CLASSES = len(SCENE_CLASSES)

# Transitions scored by %MAE = the onset of each phase after Pre-Craniotomy.
TRANSITION_PHASES = SCENE_CLASSES[1:]

# ── Cross-validation: held-out test cases per fold (20 cases) ───────────
# Internal case numbers; the manuscript relabels them I..XX. The Roman
# equivalents per fold are given in the comments (see manuscript Table 2).
FOLD_TEST_CASES = {
    1: ["Case16", "Case17", "Case01", "Case04"],   # IX, X, XI, XII
    2: ["Case12", "Case15", "Case05", "Case06"],   # VII, VIII, XIII, XIV
    3: ["Case10", "Case11", "Case07", "Case13"],   # V, VI, XV, XVI
    4: ["Case08", "Case09", "Case14", "Case18"],   # III, IV, XVII, XVIII
    5: ["Case02", "Case03", "Case19", "Case20"],   # I, II, XIX, XX
}

# ── Non-monotonic HSMM parameters ──────────────────────────────────────
SMOOTHING_WINDOW = 90          # seconds; sliding-window probability smoothing (Eq. 1)
MIN_PHASE_DURATION = {         # per-phase minimum duration D_k in seconds (Eq. 4)
    "PreCraniotomy": 60,
    "Craniotomy": 60,
    "ArterialFeederControl": 60,
    "NidusDissection": 60,
    "CraniotomyClosure": 60,
    "PostClosure": 30,
}
SWITCH_PENALTY = 3000.0        # constant per-segment switch penalty lambda (Eq. 2)
TRANSITION_FLOOR = 1e-3        # floor epsilon for never-observed transitions (Eq. 3)
BOUNDARY_CANDIDATES = 400      # subsampled candidate boundary positions (Sec. 2.8.4)

# The HSMM decode is a lightweight NumPy segmental DP (no GPU needed). The heavy
# compute is the upstream frame classifier, which is outside this repository.
