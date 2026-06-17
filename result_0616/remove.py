"""
删除 out_data 中的 out_nr*.bmp（NR 结果）文件，并分析 mnr_input/in 和 mnr_output/out 的特征差异。
"""
import os
import re
import glob
import sys
import numpy as np
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(__file__), "out_data")

# ========== 1. 删除 out_nr*.bmp ==========
nr_files = glob.glob(os.path.join(DATA_DIR, "*#out_nr*.bmp"))
print(f"找到 {len(nr_files)} 个 NR 文件，正在删除...")
for f in nr_files:
    os.remove(f)
    print(f"  已删除: {os.path.basename(f)}")

# ========== 2. 分析 in/out 特征差异 ==========
# 按测试用例分组
pattern = re.compile(r"^(.*?)#(mnr_input|mnr_output)(\d+)\.bmp$")

cases = {}
for fpath in glob.glob(os.path.join(DATA_DIR, "*.bmp")):
    fname = os.path.basename(fpath)
    m = pattern.match(fname)
    if m:
        prefix = m.group(1)  # test_function#test_nr#MNR#{case_name}#out1
        iotype = m.group(2)  # mnr_input / mnr_output
        seq = m.group(3)     # 数字序号
        cases.setdefault(prefix, {}).setdefault(seq, {})[iotype] = fpath

print(f"\n共 {len(cases)} 个测试用例")

results = []
for prefix in sorted(cases):
    for seq in sorted(cases[prefix]):
        files = cases[prefix][seq]
        in_path = files.get("mnr_input")
        out_path = files.get("mnr_output")
        if not in_path or not out_path:
            continue

        # 提取用例简称（如 MNR#00038）
        short_name = prefix.replace("test_function#test_nr#", "")

        label = f"{short_name}#{seq}"

        try:
            im_in = np.array(Image.open(in_path).convert("RGB")).astype(np.float32)
            im_out = np.array(Image.open(out_path).convert("RGB")).astype(np.float32)
        except Exception as e:
            print(f"  [跳过] {label}: {e}")
            continue

        # —— 逐通道统计 ——
        def ch_stats(img):
            return {
                "mean": img.mean(),
                "std": img.std(),
                "min": img.min(),
                "max": img.max(),
            }

        stats_in = [ch_stats(im_in[..., c]) for c in range(3)]
        stats_out = [ch_stats(im_out[..., c]) for c in range(3)]

        diff = im_out - im_in

        # —— 差异指标 ——
        mse = (diff ** 2).mean()
        mae = np.abs(diff).mean()
        psnr = 20 * np.log10(255.0) - 10 * np.log10(mse) if mse > 1e-10 else float("inf")

        # 差异图统计
        diff_mean = diff.mean(axis=(0, 1))  # per-channel mean diff
        diff_std = diff.std(axis=(0, 1))

        results.append({
            "case": short_name,
            "seq": seq,
            "in_mean": [s["mean"] for s in stats_in],
            "out_mean": [s["mean"] for s in stats_out],
            "in_std": [s["std"] for s in stats_in],
            "out_std": [s["std"] for s in stats_out],
            "mse": mse,
            "mae": mae,
            "psnr": psnr,
            "diff_mean": diff_mean.tolist(),
            "diff_std": diff_std.tolist(),
        })

# ========== 输出汇总 ==========
print(f"\n{'='*90}")
print(f"{'TestCase':<45} {'PSNR':>8} {'MAE':>8} {'MSE':>10} | {'In_Mean':>18} {'Out_Mean':>18}")
print(f"{'-'*90}")

for r in results:
    in_m = np.mean(r["in_mean"])
    out_m = np.mean(r["out_mean"])
    lbl = f"{r['case']}#{r['seq']}"
    print(f"{lbl:<45} {r['psnr']:>8.2f} {r['mae']:>8.2f} {r['mse']:>10.2f} | {in_m:>8.1f} {out_m:>8.1f}")

print(f"{'-'*90}")

# ========== 总体统计 ==========
psnrs = [r["psnr"] for r in results if r["psnr"] != float("inf")]
maes = [r["mae"] for r in results]
mses = [r["mse"] for r in results]

print(f"\n总体统计（{len(results)} 组对比）:")
print(f"  PSNR (dB): mean={np.mean(psnrs):.2f}, min={np.min(psnrs):.2f}, max={np.max(psnrs):.2f}")
print(f"  MAE:       mean={np.mean(maes):.2f}, min={np.min(maes):.2f}, max={np.max(maes):.2f}")
print(f"  MSE:       mean={np.mean(mses):.2f}, min={np.min(mses):.2f}, max={np.max(mses):.2f}")

# 按类别分组分析
cat_psnr = {}
for r in results:
    cat = "MIS" if "MNR_MIS" in r["case"] else "MNR"
    cat_psnr.setdefault(cat, []).append(r["psnr"])

print(f"\n按类别分组:")
for cat, vals in sorted(cat_psnr.items()):
    finite_vals = [v for v in vals if v != float("inf")]
    if finite_vals:
        print(f"  {cat}: count={len(vals)}, PSNR mean={np.mean(finite_vals):.2f} dB (finite)")
    else:
        print(f"  {cat}: count={len(vals)}, PSNR = inf (all identical)")

print("\n分析完成。")
