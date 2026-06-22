import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

# Load gibs_hardware_patch module from dct_show/ without needing __init__.py
_module_path = Path(__file__).resolve().parent / "dct_show" / "gibs_hardware_patch.py"
_spec = importlib.util.spec_from_file_location("gibs_hardware_patch", _module_path)
_gibs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gibs)

numpy_vectorized_predict_with_text_ui_branch = _gibs.numpy_vectorized_predict_with_text_ui_branch
save_heatmap = _gibs.save_heatmap


def main():
    data_dir = Path(__file__).resolve().parent / "test_data"
    output_dir = Path(__file__).resolve().parent / "test_data_output_gibs_patch"

    bmp_files = sorted(data_dir.glob("*.bmp"))
    if not bmp_files:
        print(f"no bmp files found in {data_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for bmp_path in bmp_files:
        stem = bmp_path.stem
        try:
            score_grid, img, _ = numpy_vectorized_predict_with_text_ui_branch(str(bmp_path))

            heatmap_path = output_dir / f"{stem}.png"
            save_heatmap(img, score_grid, str(heatmap_path))

            npy_path = output_dir / f"{stem}_score.npy"
            np.save(str(npy_path), score_grid)

            print(f"OK {stem}  shape={score_grid.shape}")
        except Exception as e:
            print(f"FAIL {stem}  error: {e}")

    print(f"\ndone. processed {len(bmp_files)} images -> {output_dir}")


if __name__ == "__main__":
    main()
