"""
Run predict_cnn.py on all images in SR_Data and its subdirectories.
Saves overlay + denoised outputs to predictions_gen/SR_Data/ mirroring the input structure.
"""

import sys, os, time, torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import predict_cnn

SR_DIR = os.path.join(os.path.dirname(__file__), "SR_Data")
OUT_BASE = os.path.join(os.path.dirname(__file__), "SR_predictions", "SR_Data")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = predict_cnn.MosquitoDenoiseCNN().to(device)
model.load_state_dict(torch.load(predict_cnn.MODEL_PATH, map_location=device), strict=False)
model.eval()
print(f"Model loaded from {predict_cnn.MODEL_PATH}")

# Find all BMPs recursively
bmps = sorted(Path(SR_DIR).rglob("*.bmp"))
print(f"Found {len(bmps)} images in {SR_DIR}\n")

total_start = time.time()
for bmp_path in bmps:
    bmp_path = str(bmp_path)
    rel = os.path.relpath(bmp_path, SR_DIR)
    subdir = os.path.dirname(rel)

    # Per-subdirectory output: override predict_cnn.OUTPUT_DIR so bilateral saves go to the right place
    out_subdir = os.path.join(OUT_BASE, subdir)
    os.makedirs(out_subdir, exist_ok=True)
    predict_cnn.OUTPUT_DIR = out_subdir

    out_path = os.path.join(out_subdir, os.path.splitext(os.path.basename(bmp_path))[0] + "_cnn.png")

    t0 = time.time()
    print(f"[{rel}]")
    predict_cnn.predict_image(model, device, bmp_path, out_path)
    print(f"  [{time.time() - t0:.0f}s]\n")

print(f"All done. Total: {time.time() - total_start:.0f}s")
print(f"Outputs saved under {OUT_BASE}")
