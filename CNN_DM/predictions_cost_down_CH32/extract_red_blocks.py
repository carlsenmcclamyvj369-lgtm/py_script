"""
Extract DM blocks from *_pred8x8.bmp (pred > 128, displayed as red
in Viewer V2) and save as dm.csv per image folder.

Usage:
  python extract_red_blocks.py
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

PRED_DIR = Path(__file__).parent


def get_dm_blocks(pred_bmp_path):
    """Return list of (row, col) for 8x8 blocks with pred > 128 (DM)."""
    pred = cv2.imread(str(pred_bmp_path), cv2.IMREAD_GRAYSCALE)
    if pred is None:
        print(f"  Error: cannot read {pred_bmp_path.name}")
        return []

    h, w = pred.shape
    gh, gw = h // 8, w // 8
    grid = pred[::8, ::8]  # (gh, gw)

    blocks = []
    for bi in range(gh):
        for bj in range(gw):
            if grid[bi, bj] > 128:
                blocks.append((bi, bj))
    return blocks


def save_dm_csv(pred_bmp_path, blocks, output_dir):
    """Save the given blocks as dm.csv matching the reference format."""
    img_name = pred_bmp_path.stem.removesuffix("_pred8x8")

    cols = [
        "name", "row", "col",
        "mean_var", "low_var_count", "high_var_count",
        "edge_strength", "edge_orientation_conf",
        "second_diff_max", "second_diff_min_max",
        "ringing_mean_max", "ringing_mean_min", "ringing_mean_min_max",
        "row_ringing_max", "row_ringing_mean",
        "col_ringing_max", "col_ringing_mean",
        "row_diff_max", "col_diff_max",
        "low_var_th", "high_var_th", "very_high_var_th",
        "max_strength_th", "eps",
        "dyn_score_lo", "dyn_score_hi", "d2_score_lo", "d2_score_hi",
        "sign_score_lo", "sign_score_hi",
        "dyn_ratio", "d2_ratio", "sign_ratio",
    ]

    rows = []
    for bi, bj in blocks:
        row = {"name": img_name, "row": bi, "col": bj}
        row["low_var_th"] = 60
        row["high_var_th"] = 500
        row["very_high_var_th"] = 2000
        row["max_strength_th"] = 0.0
        row["eps"] = 3.0
        row["dyn_score_lo"] = 20
        row["dyn_score_hi"] = 120
        row["d2_score_lo"] = 5
        row["d2_score_hi"] = 60
        row["sign_score_lo"] = 1
        row["sign_score_hi"] = 4
        row["dyn_ratio"] = 0.45
        row["d2_ratio"] = 0.35
        row["sign_ratio"] = 0.2
        for c in cols:
            if c not in row:
                row[c] = 0
        rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)[cols]
    df.to_csv(output_dir / "not_dm.csv", index=False)
    print(f"  → {output_dir / 'not_dm.csv'}  ({len(rows)} blocks)")


def main():
    pred_files = sorted(PRED_DIR.glob("*_pred8x8.bmp"))
    print(f"Found {len(pred_files)} pred8x8 files\n")

    for pf in pred_files:
        img_name = pf.stem.removesuffix("_pred8x8")

        blocks = get_dm_blocks(pf)
        if not blocks:
            print(f"  {pf.name}: no DM blocks, skip")
            continue

        output_dir = PRED_DIR / img_name
        save_dm_csv(pf, blocks, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
