"""
Compare DM blocks between two folders.

For each image:

old_blocks ∩ new_blocks
    -> removed

Only keep:

new - old   => added
old - new   => removed

Output CSV format is compatible with original not_dm.csv.
"""

import cv2
import pandas as pd
from pathlib import Path

# ==========================================================
# Config
# ==========================================================

OLD_DIR = Path(
    r"C:\code\py\denoise\scripts\CNN_DM\predictions_cost_down_CH32"
)

NEW_DIR = Path(
    r"C:\code\py\denoise\scripts\CNN_DM\predictions"
)

OUTPUT_DIR = Path("diff_result")

BLOCK_SIZE = 8

# ==========================================================
# Original csv columns
# ==========================================================

CSV_COLS = [
    "name", "row", "col",
    "mean_var",
    "low_var_count",
    "high_var_count",
    "edge_strength",
    "edge_orientation_conf",
    "second_diff_max",
    "second_diff_min_max",
    "ringing_mean_max",
    "ringing_mean_min",
    "ringing_mean_min_max",
    "row_ringing_max",
    "row_ringing_mean",
    "col_ringing_max",
    "col_ringing_mean",
    "row_diff_max",
    "col_diff_max",
    "low_var_th",
    "high_var_th",
    "very_high_var_th",
    "max_strength_th",
    "eps",
    "dyn_score_lo",
    "dyn_score_hi",
    "d2_score_lo",
    "d2_score_hi",
    "sign_score_lo",
    "sign_score_hi",
    "dyn_ratio",
    "d2_ratio",
    "sign_ratio"
]

# ==========================================================
# Read DM blocks
# ==========================================================

def get_dm_blocks(pred_bmp_path):

    pred = cv2.imread(
        str(pred_bmp_path),
        cv2.IMREAD_GRAYSCALE
    )

    if pred is None:
        print(f"Cannot read: {pred_bmp_path}")
        return set()

    h, w = pred.shape

    gh = h // BLOCK_SIZE
    gw = w // BLOCK_SIZE

    grid = pred[::BLOCK_SIZE, ::BLOCK_SIZE]

    blocks = set()

    for row in range(gh):
        for col in range(gw):

            if grid[row, col] > 128:
                blocks.add((row, col))

    return blocks

# ==========================================================
# Create row
# ==========================================================

def create_csv_row(img_name, row, col):

    data = {
        "name": img_name,
        "row": row,
        "col": col,

        "low_var_th": 60,
        "high_var_th": 500,
        "very_high_var_th": 2000,

        "max_strength_th": 0,
        "eps": 3,

        "dyn_score_lo": 20,
        "dyn_score_hi": 120,

        "d2_score_lo": 5,
        "d2_score_hi": 60,

        "sign_score_lo": 1,
        "sign_score_hi": 4,

        "dyn_ratio": 0.45,
        "d2_ratio": 0.35,
        "sign_ratio": 0.20
    }

    for col_name in CSV_COLS:
        if col_name not in data:
            data[col_name] = 0

    return data

# ==========================================================
# Save csv
# ==========================================================

def save_csv(blocks, image_name, output_csv):

    rows = []

    for row, col in sorted(blocks):

        rows.append(
            create_csv_row(
                image_name,
                row,
                col
            )
        )

    df = pd.DataFrame(rows)

    if len(df) == 0:
        df = pd.DataFrame(columns=CSV_COLS)

    df = df[CSV_COLS]

    df.to_csv(
        output_csv,
        index=False
    )

# ==========================================================
# Compare one image
# ==========================================================

def compare_image(old_file, new_file):

    image_name = old_file.stem.replace(
        "_pred8x8",
        ""
    )

    old_blocks = get_dm_blocks(old_file)
    new_blocks = get_dm_blocks(new_file)

    common = old_blocks & new_blocks

    added = new_blocks - old_blocks

    removed = old_blocks - new_blocks

    out_dir = OUTPUT_DIR / image_name

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    save_csv(
        added,
        image_name,
        out_dir / "dm.csv"
    )

    save_csv(
        removed,
        image_name,
        out_dir / "removed.csv"
    )

    save_csv(
        added | removed,
        image_name,
        out_dir / "not_dm.csv"
    )

    print(
        f"{image_name}\n"
        f"  old     : {len(old_blocks)}\n"
        f"  new     : {len(new_blocks)}\n"
        f"  common  : {len(common)}\n"
        f"  added   : {len(added)}\n"
        f"  removed : {len(removed)}\n"
    )

# ==========================================================
# Main
# ==========================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    old_files = {
        f.name: f
        for f in OLD_DIR.glob("*_pred8x8.bmp")
    }

    new_files = {
        f.name: f
        for f in NEW_DIR.glob("*_pred8x8.bmp")
    }

    matched_files = sorted(
        set(old_files.keys()) &
        set(new_files.keys())
    )

    print(
        f"Found {len(matched_files)} matched files\n"
    )

    for filename in matched_files:

        compare_image(
            old_files[filename],
            new_files[filename]
        )

    print("\nDone.")

if __name__ == "__main__":
    main()