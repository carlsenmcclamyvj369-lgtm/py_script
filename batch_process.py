import numpy as np
import cv2
from pathlib import Path

from main import (
    compute_edge_roi,
    high_freq_map,
    multi_scale_consistency,
    patch_similarity_map,
    coherence_map,
    mosquito_score,
    mosquito_mask,
)


# =========================================================
# Normalize helper for visualization
# =========================================================
def normalize_vis(arr):
    """Min-max normalize to 0-1, handle flat/constant cases."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


# =========================================================
# Feature directory & image save
# =========================================================
OUTPUT = Path("test_data_output")


def save_feature(vis_dir, raw_dir, name, feat):
    """Save feature map as PNG (vis) and npy (raw)."""
    np.save(raw_dir / f"{name}.npy", feat)

    # Visualize: float32 → uint8
    vis = normalize_vis(feat)
    cv2.imwrite(str(vis_dir / f"{name}.png"), (vis * 255).astype(np.uint8))


def save_mask(vis_dir, raw_dir, name, mask):
    """Save binary mask as PNG (vis) and npy (raw)."""
    np.save(raw_dir / f"{name}.npy", mask)
    cv2.imwrite(str(vis_dir / f"{name}.png"), (mask * 255).astype(np.uint8))


# =========================================================
# DCT High-Frequency Feature
# =========================================================
def dct_high_freq_energy(I, block_size=16, freq_th=16):
    """Block DCT: zero out low frequencies, sum high-freq energy.

    For each block_size×block_size block:
      1. Apply DCT
      2. Zero coefficients where i+j < freq_th (low freq)
      3. Sum squared energy of remaining (high) coefficients

    Returns a map upsampled to original image size.
    """
    I = I.astype(np.float32)
    h, w = I.shape

    # Pad to multiples of block_size
    ph = (block_size - h % block_size) % block_size
    pw = (block_size - w % block_size) % block_size
    if ph or pw:
        I = np.pad(I, ((0, ph), (0, pw)), mode='edge')

    result = np.zeros_like(I)

    for y in range(0, I.shape[0], block_size):
        for x in range(0, I.shape[1], block_size):
            block = I[y:y + block_size, x:x + block_size]
            dct = cv2.dct(block)

            # High-pass mask: keep only i+j >= freq_th
            energy = 0.0
            for i in range(block_size):
                for j in range(block_size):
                    if i + j >= freq_th:
                        energy += dct[i, j] * dct[i, j]

            result[y:y + block_size, x:x + block_size] = energy

    return result[:h, :w]


# =========================================================
# Process all images
# =========================================================
def process_all():
    data_dir = Path("test_data")
    bmp_files = sorted(data_dir.glob("*.bmp"))

    if not bmp_files:
        print(f"No .bmp files found in {data_dir.resolve()}")
        return

    for img_path in bmp_files:
        stem = img_path.stem
        print(f"Processing  {stem} ...")

        # Output subdir per image
        vis_dir = OUTPUT / stem / "vis"
        raw_dir = OUTPUT / stem / "raw"
        vis_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Read image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  SKIP: cannot read {img_path}")
            continue

        # Save original
        cv2.imwrite(str(vis_dir / "input.png"), img)
        np.save(raw_dir / "input.npy", img)

        # Compute raw features
        roi = compute_edge_roi(img)
        hf = high_freq_map(img)
        ms = multi_scale_consistency(img)
        ps = patch_similarity_map(img)
        coh = coherence_map(img)
        dct_hf = dct_high_freq_energy(img)

        # Apply thresholds (same logic as main.mosquito_score)
        hf_mask = (hf > 20).astype(np.float32)
        ms_mask = ((ms > 0.1) & (ms < 0.2)).astype(np.float32)
        ps_mask = (ps > 20).astype(np.float32)
        coh_mask = (coh < 0.3).astype(np.float32)
        dct_hf_mask = (dct_hf > 2000).astype(np.float32)

        # Score & final mask
        score = mosquito_score(img)
        mask = mosquito_mask(img)

        # Save thresholded masks (edge_roi is already a mask)
        save_mask(vis_dir, raw_dir, "edge_roi", roi)
        save_mask(vis_dir, raw_dir, "hf_mask", hf_mask)
        save_mask(vis_dir, raw_dir, "ms_mask", ms_mask)
        save_mask(vis_dir, raw_dir, "ps_mask", ps_mask)
        save_mask(vis_dir, raw_dir, "coh_mask", coh_mask)
        save_feature(vis_dir, raw_dir, "dct_hf_energy", dct_hf)
        save_mask(vis_dir, raw_dir, "dct_hf_mask", dct_hf_mask)
        save_feature(vis_dir, raw_dir, "mosquito_score", score)
        save_mask(vis_dir, raw_dir, "mosquito_mask", mask)

        print(f"  Done. ({len(list(vis_dir.iterdir()))} files)")

    print(f"\nAll done. Results saved under {OUTPUT}/")


# =========================================================
# Feature grids（一张大图对比所有特征）
# =========================================================
def make_grids():
    """For each image, composite a grid of all feature maps side-by-side."""
    skip_dirs = {"summary", "dark_channel", "connected_components", "cc_classify"}

    layout = [
        ("input", "Input"),
        ("edge_roi", "Edge ROI"),
        ("hf_mask", "HF > 20"),
        ("ms_mask", "MS < 0.3"),
        ("ps_mask", "PS > 20"),
        ("coh_mask", "Coh < 0.3"),
        ("dct_hf_mask", "DCT HF"),
        ("mosquito_score", "Score"),
        ("mosquito_mask", "Mask"),
    ]

    for img_dir in sorted(OUTPUT.iterdir()):
        if not img_dir.is_dir() or img_dir.name in skip_dirs:
            continue

        vis_dir = img_dir / "vis"
        print(f"Grid  {img_dir.name} ...")

        tiles = []
        for fname, label in layout:
            path = vis_dir / f"{fname}.png"
            if not path.exists():
                continue
            tile = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if tile is None:
                continue
            h, w = tile.shape

            # Add label bar at bottom
            bar = np.full((24, w), 255, dtype=np.uint8)
            cv2.putText(bar, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1)
            tile = np.vstack([tile, bar])
            tiles.append(tile)

        if not tiles:
            continue

        # Arrange in 3 rows × 3 cols
        row1 = np.hstack(tiles[:3])
        row2 = np.hstack(tiles[3:6])
        row3 = np.hstack(tiles[6:9])

        # Match widths
        rows = [row1, row2, row3]
        max_w = max(r.shape[1] for r in rows)
        for i in range(len(rows)):
            if rows[i].shape[1] < max_w:
                rows[i] = np.pad(rows[i], ((0, 0), (0, max_w - rows[i].shape[1])), constant_values=255)

        grid = np.vstack(rows)
        grid_path = vis_dir.parent / "grid.png"
        cv2.imwrite(str(grid_path), grid)
        print(f"  grid saved -> {grid_path}")

    # Copy all grids to summary folder
    summary_dir = OUTPUT / "summary"
    summary_dir.mkdir(exist_ok=True)
    for img_dir in sorted(OUTPUT.iterdir()):
        if not img_dir.is_dir() or img_dir.name in skip_dirs:
            continue
        src = img_dir / "grid.png"
        if src.exists():
            dst = summary_dir / f"{img_dir.name}.png"
            cv2.imwrite(str(dst), cv2.imread(str(src)))
    print(f"\nSummary grids saved under {summary_dir}/")


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    process_all()
    make_grids()
