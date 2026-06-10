import numpy as np
import cv2
from pathlib import Path

from mainv2 import directional_entropy_map

OUTPUT = Path("test_data_output_0609")
HEATMAP_DIR = OUTPUT / "dir_heatmap"
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = Path("test_data_output_0609_jet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Helpers
# =========================================================
def apply_heatmap(data):
    lo, hi = data.min(), data.max()
    if hi - lo < 1e-8:
        norm = np.zeros_like(data, dtype=np.uint8)
    else:
        norm = ((data - lo) / (hi - lo) * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET), lo, hi


def colorbar(h, lo=0.0, hi=2.0, step=0.25):
    w = 56
    bar = np.linspace(255, 0, h, dtype=np.uint8).reshape(h, 1)
    bar = cv2.applyColorMap(bar, cv2.COLORMAP_JET)
    bar = np.hstack([bar] * w)

    for val in np.arange(lo, hi + 1e-6, step):
        ratio = (val - lo) / (hi - lo) if hi > lo else 0
        y = int((1 - ratio) * (h - 1))
        label = f"{val:.2f}"
        cv2.putText(bar, label, (4, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.line(bar, (w - 10, y), (w - 2, y), (255, 255, 255), 1)

    # Title
    cv2.putText(bar, "Entropy", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.putText(bar, "H", (6, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return bar


# =========================================================
# Generate heatmaps
# =========================================================
def run_all():
    data_dir = Path("test_data")

    for sub_dir in sorted(OUTPUT.iterdir()):
        if not sub_dir.is_dir() or sub_dir.name in ("summary", "dir_heatmap"):
            continue
        stem = sub_dir.name

        bmp_path = data_dir / f"{stem}.bmp"
        if not bmp_path.exists():
            continue

        print(f"  {stem}")

        img_gray = cv2.imread(str(bmp_path), cv2.IMREAD_GRAYSCALE)
        dir_map = directional_entropy_map(img_gray)

        # Heatmap
        hm, vmin, vmax = apply_heatmap(dir_map)

        # Build output: heatmap | colorbar
        h, w = img_gray.shape
        cbar = colorbar(h, lo=vmin, hi=vmax)
        out = np.hstack([hm, cbar])

        # Annotate stats on heatmap
        vals = dir_map.ravel()
        info_lines = [
            f"Range: [{vmin:.3f}, {vmax:.3f}]",
            f"Mean: {vals.mean():.3f}",
            f"Median: {np.median(vals):.3f}",
            f"p10: {np.percentile(vals, 10):.3f}",
            f"p25: {np.percentile(vals, 25):.3f}",
            f"p75: {np.percentile(vals, 75):.3f}",
            f"p90: {np.percentile(vals, 90):.3f}",
            f"<1.2: {(dir_map < 1.2).mean()*100:.1f}%",
        ]
        for i, line in enumerate(info_lines):
            y = h - 10 - (len(info_lines) - 1 - i) * 18
            cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 0), 3)  # black outline
            cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1)

        out_path = OUTPUT_DIR / f"{stem}.png"
        cv2.imwrite(str(out_path), out)

        # Raw data
        np.save(OUTPUT_DIR / f"{stem}_dir.npy", dir_map)

        print(f"    saved -> {out_path}")
        print(f"    dir range=[{vmin:.3f}, {vmax:.3f}]  shape={dir_map.shape}")

    print(f"\nAll jet heatmaps saved under {OUTPUT_DIR}/")


if __name__ == "__main__":
    run_all()
