"""
export_okprobs_python.py
-------------------------------------------------------------------------
Template: produce the OK-probability tables that the temporal pipeline's
inference-time OK/NG gate (configs B / D) consumes, using YOUR OWN OK/NG
model built in Python (PyTorch, timm, TensorFlow, scikit-learn, ...).

Output: one file per fold,  okprobs_fold{1..5}.csv  with columns:
    path     - frame filename (CaseXXYY_HHMMSS.jpg)
    ok_prob  - probability that the frame is OK (usable), in [0, 1]
Place these CSVs in the directory you pass as AMVIDEO_OKPROB_DIR.

The temporal code is model-agnostic and only needs the (path, ok_prob) table.
Fill in `load_model()` and `ok_probability()` for your model; the rest is glue.
-------------------------------------------------------------------------
"""
import os
import glob
import argparse
import pandas as pd


# === EDIT THESE TWO FUNCTIONS FOR YOUR MODEL ============================
def load_model(weights_path):
    """Load and return your OK/NG model (any framework)."""
    # Example (PyTorch / timm), adapt to your model (any 2-class backbone works):
    #   import timm, torch
    #   model = timm.create_model("inception_resnet_v2", pretrained=False, num_classes=2)
    #   model.load_state_dict(torch.load(weights_path, map_location="cpu",
    #                                    weights_only=True))
    #   model.eval()
    #   return model
    raise NotImplementedError("Implement load_model() for your OK/NG model.")


def ok_probability(model, image_path):
    """Return P(OK) in [0, 1] for one frame image. Adapt to your model."""
    # Example (PyTorch, OK = class index 1):
    #   from PIL import Image
    #   import numpy as np, torch
    #   img = Image.open(image_path).convert("RGB").resize((299, 299))
    #   x = (np.asarray(img, np.float32) / 255.0 - [0.485, 0.456, 0.406]) \
    #       / [0.229, 0.224, 0.225]
    #   x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float()
    #   with torch.no_grad():
    #       return float(torch.softmax(model(x), 1)[0, 1])
    raise NotImplementedError("Implement ok_probability() for your OK/NG model.")
# =======================================================================


def main():
    ap = argparse.ArgumentParser(description="Export OK-probability CSVs per fold.")
    ap.add_argument("--weights", required=True, help="path to your OK/NG model weights")
    ap.add_argument("--frames", required=True,
                    help="root dir; expects subfolders fold1..fold5 of Case*.jpg frames")
    ap.add_argument("--out", default="okprobs", help="output dir (= AMVIDEO_OKPROB_DIR)")
    args = ap.parse_args()

    model = load_model(args.weights)
    os.makedirs(args.out, exist_ok=True)

    for fold in range(1, 6):
        frame_dir = os.path.join(args.frames, f"fold{fold}")
        files = sorted(glob.glob(os.path.join(frame_dir, "**", "Case*.jpg"), recursive=True))
        if not files:
            print(f"[fold {fold}] no frames under {frame_dir}, skipping")
            continue
        rows = [{"path": os.path.basename(p), "ok_prob": ok_probability(model, p)}
                for p in files]
        out_csv = os.path.join(args.out, f"okprobs_fold{fold}.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"wrote {out_csv} ({len(rows)} frames)")


if __name__ == "__main__":
    main()
