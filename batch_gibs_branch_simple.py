"""
batch_gibs_branch_simple.py — 批量跑 gibs_branch_simple.py。

用法:
    python batch_gibs_branch_simple.py [data_dir] [output_dir]

默认 data_dir = test_data/，output_dir = test_data_output_gibs/
"""

import argparse
from pathlib import Path

import numpy as np

from gibs_branch_simple import detect_gibbs, save_heatmap, GibbsConfig


def main():
    parser = argparse.ArgumentParser(description="批量检测 Gibbs 伪影")
    parser.add_argument("data_dir", nargs="?", default="test_data",
                        help="输入图片目录（默认 test_data/）")
    parser.add_argument("output_dir", nargs="?", default="test_data_output_gibs",
                        help="输出目录（默认 test_data_output_gibs/）")
    parser.add_argument("--suffix", default="*.bmp",
                        help="文件后缀过滤（默认 *.bmp）")
    parser.add_argument("--scale", type=float, default=12.0,
                        help="热力图缩放系数（默认 12.0）")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob(args.suffix))
    if not files:
        print(f"未找到图片: {data_dir}/{args.suffix}")
        return

    cfg = GibbsConfig()
    ok = fail = 0

    for fp in files:
        stem = fp.stem
        try:
            score, img, mask = detect_gibbs(str(fp), cfg)

            # 保存热力图
            save_heatmap(img, score, str(output_dir / f"{stem}.png"), scale=args.scale)

            # 保存 score grid
            np.save(str(output_dir / f"{stem}_score.npy"), score)

            # 保存 mask
            np.save(str(output_dir / f"{stem}_mask.npy"), mask)

            n_det = int(np.sum(mask))
            print(f"OK  {stem}  grid={score.shape}  detected={n_det}")
            ok += 1
        except Exception as e:
            print(f"FAIL {stem}  {e}")
            fail += 1

    print(f"\n完成: {ok} OK, {fail} FAIL -> {output_dir}")


if __name__ == "__main__":
    main()
