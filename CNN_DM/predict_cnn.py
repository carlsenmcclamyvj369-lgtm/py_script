"""
Inference on a single BMP using the trained CNN model.
Computes 16 features per 8x8 block, assembles 9x9 neighborhoods,
and runs the CNN to predict dm/not_dm. Overlays results on the image.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import sys
import time
from dm_cnn import MosquitoDenoiseCNN


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mos_featue_analyze'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mos_featue_analyze', 'excel'))

import feature_compute_ref as fcr

# ─── CNN Model (same architecture as dm_cnn.py) ───


# ─── Config ───
SCRIPT_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(SCRIPT_DIR, "mosquito_denoise_cnn.pth")
TEST_DIR = r"C:\code\py\denoise\scripts\test_data"
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "predictions")

GS = 8

# 16 features used by CNN
FEATURES16 = [
    'mean_var', 'low_var_count', 'high_var_count',
    'edge_strength', 'edge_orientation_conf',
    'col_ringing_mean', 'row_ringing_mean',
    'second_diff_max', 'second_diff_min_max',
    'profile_ringing_mean', 'ringing_mean_min', 'ringing_mean_min_max',
    'row_ringing_max', 'row_diff_max', 'col_ringing_max', 'col_diff_max',
]

# Normalization divisors (same as dm_cnn.py)
NORM_DIV16 = {
    'mean_var': 1020.0, 'low_var_count': 64.0, 'high_var_count': 64.0,
    'edge_strength': 255.0, 'edge_orientation_conf': 1.0,
    'col_ringing_mean': 1.0, 'row_ringing_mean': 1.0,
    'second_diff_max': 510.0, 'second_diff_min_max': 1.0,
    'profile_ringing_mean': 1.0, 'ringing_mean_min': 1.0, 'ringing_mean_min_max': 1.0,
    'row_ringing_max': 1.0, 'col_ringing_max': 1.0,
    'row_diff_max': 255.0, 'col_diff_max': 255.0,
}

FEATURE_IDX = None  # set at runtime

# 9x9 offsets row-major
OFFSETS_9x9 = [(dr, dc) for dr in range(-4, 5) for dc in range(-4, 5)]


def get_block(map2d, bi, bj):
    y1, y2 = bi * GS, min(bi * GS + GS, map2d.shape[0])
    x1, x2 = bj * GS, min(bj * GS + GS, map2d.shape[1])
    vals = map2d[y1:y2, x1:x2].flatten()
    return vals[~np.isnan(vals)]


def compute_grid_features(y_full):
    """Compute all 49 features for every grid block. Returns (gh, gw, 49)."""
    H, W = y_full.shape
    gh, gw = H // GS, W // GS

    var_map = fcr.compute_var_map(y_full)
    k3 = np.ones((3, 3), dtype=np.float64) / 9.0
    mean3 = cv2.filter2D(y_full, -1, k3, borderType=cv2.BORDER_REFLECT)
    res_map = fcr.compute_residual_map(y_full, mean3)
    lap_map = fcr.compute_lap_map(y_full)
    grad_map = fcr.compute_grad_map(y_full)
    h_edge, v_edge = fcr.compute_edge_maps(y_full)
    pm = (var_map, res_map, lap_map, grad_map, h_edge, v_edge)

    TOTAL = 49
    grid = np.zeros((gh, gw, TOTAL), dtype=np.float32)
    for bi in range(gh):
        for bj in range(gw):
            y1, y2 = bi*GS, min(bi*GS+GS, H)
            x1, x2 = bj*GS, min(bj*GS+GS, W)
            if y2-y1 < 2 or x2-x1 < 2:
                continue
            block = y_full[y1:y2, x1:x2]
            bh, bw = block.shape
            feats = np.zeros(TOTAL, dtype=np.float32)

            # Variance
            bv = get_block(var_map, bi, bj)
            if len(bv) > 0:
                sv = np.sort(bv)
                feats[0] = float(np.mean(bv))
                feats[1] = float(np.max(bv))
                feats[2] = float(np.mean(sv[-5:]) if len(sv) >= 5 else np.mean(sv))
                feats[3] = float(np.sum(bv < 100))
                feats[4] = float(np.sum(bv > 500))
                feats[5] = float(np.sum(bv > 2000))
            br = get_block(res_map, bi, bj)
            if len(br) > 0:
                feats[6] = float(np.mean(np.abs(br)))
                feats[7] = float(np.max(np.abs(br)))
            bl = get_block(lap_map, bi, bj)
            if len(bl) > 0:
                feats[8] = float(np.mean(bl))
                feats[9] = float(np.max(bl))
            bg = get_block(grad_map, bi, bj)
            if len(bg) > 0:
                feats[10] = float(np.mean(bg))
                feats[11] = float(np.max(bg))
            bh_e = get_block(h_edge, bi, bj)
            bv_e = get_block(v_edge, bi, bj)
            if len(bh_e) > 0 and len(bv_e) > 0:
                hs = float(np.mean(bh_e)); vs = float(np.mean(bv_e)); ms = max(hs, vs)
                feats[12] = ms; feats[13] = hs
                feats[14] = float(np.max(bh_e)); feats[15] = float(np.min(bh_e))
                feats[16] = vs; feats[17] = float(np.max(bv_e))
                feats[18] = float(np.min(bv_e))
                feats[19] = abs(hs-vs)/ms if ms > 1e-6 else 0.0

            row_energies = []
            for ry in range(bh):
                v = block[ry, :]
                if len(v) >= 3:
                    d2 = v[:-2] - 2*v[1:-1] + v[2:]
                    row_energies.append(float(np.mean(np.abs(d2))))
            col_energies = []
            for rx in range(bw):
                v = block[:, rx]
                if len(v) >= 3:
                    d2 = v[:-2] - 2*v[1:-1] + v[2:]
                    col_energies.append(float(np.mean(np.abs(d2))))
            if row_energies:
                feats[20] = float(np.mean(row_energies))
                feats[21] = float(np.max(row_energies))
                feats[22] = float(np.min(row_energies))
            if col_energies:
                feats[23] = float(np.mean(col_energies))
                feats[24] = float(np.max(col_energies))
                feats[25] = float(np.min(col_energies))
            rd, cd = feats[20], feats[23]
            feats[26] = max(rd, cd)
            feats[27] = (min(rd, cd)/max(rd, cd)) if max(rd, cd) > 0 else 0.0

            row_means = [float(block[r, :].mean()) for r in range(bh)]
            row_diffs = [abs(row_means[r+1]-row_means[r]) for r in range(bh-1)] if bh >= 2 else [0.0]
            col_means = [float(block[:, c].mean()) for c in range(bw)]
            col_diffs = [abs(col_means[c+1]-col_means[c]) for c in range(bw-1)] if bw >= 2 else [0.0]
            feats[28] = float(np.mean(row_diffs)) if row_diffs else 0.0
            feats[29] = float(np.max(row_diffs)) if row_diffs else 0.0
            feats[30] = float(np.mean(col_diffs)) if col_diffs else 0.0
            feats[31] = float(np.max(col_diffs)) if col_diffs else 0.0

            try:
                ringing = fcr.ringing_stats_for_block(block, bh, bw)
                if ringing:
                    for k, idx in [('row_ringing_max',32),('row_ringing_min',33),('row_ringing_mean',34),
                        ('ringing_mean_max',35),('ringing_mean_min',36),('ringing_mean_min_max',37),
                        ('profile_ringing_max',38),('profile_ringing_mean',39),
                        ('row_ringing_dyn_score',40),('row_ringing_d2_score',41),('row_ringing_sign_score',42),
                        ('col_ringing_max',43),('col_ringing_min',44),('col_ringing_mean',45),
                        ('col_ringing_dyn_score',46),('col_ringing_d2_score',47),('col_ringing_sign_score',48)]:
                        feats[idx] = ringing[k]
            except Exception:
                pass
            grid[bi, bj] = feats
    return grid


def predict_image(model, device, bmp_path, output_path):
    """Run CNN on a BMP, save overlay."""
    bgr = cv2.imread(bmp_path, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"ERROR: cannot read {bmp_path}")
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    y_full = fcr.compute_y_from_rgb(rgb)
    H, W = y_full.shape
    gh, gw = H // GS, W // GS
    print(f"  Image: {W}x{H}, grid: {gw}x{gh}")

    print("  Computing grid features...", end=" ", flush=True)
    t0 = time.time()
    grid = compute_grid_features(y_full)
    print(f"[{time.time()-t0:.0f}s]")

    # Select 16 features
    f_idx = [FEATURE_IDX[f] for f in FEATURES16]

    # Assemble 9x9 neighborhoods (skip border blocks where 9x9 doesn't fit)
    print("  Assembling neighborhoods & predicting...", end=" ", flush=True)
    t0 = time.time()

    pred_map = np.full((gh, gw), np.nan, dtype=np.float32)
    patches = []
    coords = []

    for bi in range(4, gh - 4):
        for bj in range(4, gw - 4):
            neigh = np.zeros((81, 16), dtype=np.float32)
            for i, (dr, dc) in enumerate(OFFSETS_9x9):
                fv = grid[bi + dr, bj + dc, f_idx]
                div = np.array([NORM_DIV16[f] for f in FEATURES16], dtype=np.float32)
                neigh[i] = np.clip(fv / div, 0, 1)
            # (81, 16) -> (9, 9, 16) -> (16, 9, 9)
            patch = neigh.reshape(9, 9, 16).transpose(2, 0, 1)
            patches.append(patch)
            coords.append((bi, bj))

    if patches:
        X = np.stack(patches, axis=0)  # (N, 16, 9, 9)
        X_t = torch.tensor(X, dtype=torch.float32, device=device)

        with torch.no_grad():
            probs = model(X_t).cpu().numpy().flatten()

        for idx, (bi, bj) in enumerate(coords):
            pred_map[bi, bj] = probs[idx]

    print(f"[{time.time()-t0:.0f}s]")

    # Visualize
    dm_count = int(np.nansum(pred_map > 0.5))
    valid_count = int(np.sum(~np.isnan(pred_map)))
    print(f"  DM: {dm_count}/{valid_count} ({100*dm_count/max(valid_count,1):.1f}%)")

    y_norm = np.clip(y_full, 0, 255).astype(np.uint8)
    display = cv2.cvtColor(y_norm, cv2.COLOR_GRAY2BGR)

    for bi in range(gh):
        for bj in range(gw):
            y1, y2 = bi*GS, min(bi*GS+GS, H)
            x1, x2 = bj*GS, min(bj*GS+GS, W)
            p = pred_map[bi, bj]
            if np.isnan(p):
                continue
            if p > 0.5:
                overlay = display[y1:y2, x1:x2].astype(np.float64)
                overlay[:, :, 2] = np.clip(overlay[:, :, 2]*0.6 + 255*0.4, 0, 255)
                display[y1:y2, x1:x2] = overlay.astype(np.uint8)
            cv2.rectangle(display, (x1, y1), (x2, y2), (100, 100, 100), 1)

    cv2.putText(display, f"CNN DM: {dm_count}/{valid_count} ({100*dm_count/max(valid_count,1):.1f}%)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    cv2.imwrite(str(output_path), display)
    print(f"  Saved: {output_path}")


def main():
    # Build feature index (49 features)
    global FEATURE_IDX
    ALL49 = [
        'mean_var', 'max_var', 'top5_var', 'low_var_count', 'high_var_count', 'very_high_var_count',
        'residual_mean', 'residual_max', 'lap_mean', 'lap_max', 'grad_mean', 'grad_max',
        'edge_strength', 'h_strength', 'h_strength_max', 'h_strength_min',
        'v_strength', 'v_strength_max', 'v_strength_min', 'edge_orientation_conf',
        'row_second_diff', 'row_second_diff_max', 'row_second_diff_min',
        'col_second_diff', 'col_second_diff_max', 'col_second_diff_min',
        'second_diff_max', 'second_diff_min_max',
        'row_diff_mean', 'row_diff_max', 'col_diff_mean', 'col_diff_max',
        'row_ringing_max', 'row_ringing_min', 'row_ringing_mean',
        'ringing_mean_max', 'ringing_mean_min', 'ringing_mean_min_max',
        'profile_ringing_max', 'profile_ringing_mean',
        'row_ringing_dyn_score', 'row_ringing_d2_score', 'row_ringing_sign_score',
        'col_ringing_max', 'col_ringing_min', 'col_ringing_mean',
        'col_ringing_dyn_score', 'col_ringing_d2_score', 'col_ringing_sign_score',
    ]
    FEATURE_IDX = {n: i for i, n in enumerate(ALL49)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MosquitoDenoiseCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")

    # Process single image or all
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == '--batch':
            # Batch: process all images
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            bmps = sorted([f for f in os.listdir(TEST_DIR) if f.endswith('.bmp')])
            print(f"Processing {len(bmps)} images...\n")
            for b in bmps:
                bmp_path = os.path.join(TEST_DIR, b)
                out_path = os.path.join(OUTPUT_DIR, b.replace('.bmp', '_cnn.png'))
                t0 = time.time()
                print(f"[{b}]")
                predict_image(model, device, bmp_path, out_path)
                print(f"  [{time.time()-t0:.0f}s]\n")
        else:
            # Single image: path or filename
            bmp_path = arg if os.path.exists(arg) else os.path.join(TEST_DIR, arg)
            out_path = os.path.join(OUTPUT_DIR, os.path.basename(bmp_path).replace('.bmp', '_cnn.png'))
            print(f"\nProcessing: {bmp_path}")
            predict_image(model, device, bmp_path, out_path)
    else:
        print("Usage:")
        print("  python predict_cnn.py <bmp_path|filename>  单张图片")
        print("  python predict_cnn.py --batch              批量处理 test_data 下所有图片")


if __name__ == "__main__":
    main()
