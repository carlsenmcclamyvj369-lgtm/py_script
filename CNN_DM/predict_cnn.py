"""
Inference on a single BMP using the trained CNN model.
Computes 16 features per 8x8 block, assembles 9x9 neighborhoods,
and runs the CNN to predict dm/not_dm. Overlays results on the image.
"""

import numpy as np
import torch
import cv2
import os
import sys
import time

from dm_cnn import MosquitoDenoiseCNN, features_list, NORM_DIV
import feature_compute_reference as fcr

# ─── Config ───
SCRIPT_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(SCRIPT_DIR, "mosquito_denoise_cnn.pth")
# TEST_DIR = r"C:\code\py\denoise\scripts\CNN_DM\gen_pattern_img"
TEST_DIR = r"C:\code\py\denoise\scripts\test_data\dot25"
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "predictions")

GS = 8
OFFSETS_9x9 = [(dr, dc) for dr in range(-4, 5) for dc in range(-4, 5)]
FEATURE_IDX = None  # set at runtime
LOW_VAR_TH = 60
HIGH_VAR_TH = 500


def get_block(map2d, bi, bj):
    y1, y2 = bi * GS, min(bi * GS + GS, map2d.shape[0])
    x1, x2 = bj * GS, min(bj * GS + GS, map2d.shape[1])
    vals = map2d[y1:y2, x1:x2].flatten()
    return vals[~np.isnan(vals)]


def compute_grid_features(y_full):
    """Compute only the 16 CNN features for every grid block. Returns (gh, gw, 16)."""
    H, W = y_full.shape
    gh, gw = H // GS, W // GS

    var_map = fcr.compute_var_map(y_full)
    h_edge, v_edge = fcr.compute_edge_maps(y_full)

    total = 16
    grid = np.zeros((gh, gw, total), dtype=np.float32)
    for bi in range(gh):
        for bj in range(gw):
            y1, y2 = bi*GS, min(bi*GS+GS, H)
            x1, x2 = bj*GS, min(bj*GS+GS, W)
            if y2-y1 < 2 or x2-x1 < 2:
                continue
            block = y_full[y1:y2, x1:x2]
            bh, bw = block.shape
            feats = np.zeros(total, dtype=np.float32)

            # var features: mean_var, low_var_count, high_var_count
            bv = get_block(var_map, bi, bj)
            if len(bv) > 0:
                feats[0] = float(np.mean(bv))          # mean_var
                feats[1] = float(np.sum(bv < LOW_VAR_TH))     # low_var_count
                feats[2] = float(np.sum(bv > HIGH_VAR_TH))     # high_var_count

            # edge features: edge_strength, edge_orientation_conf
            bh_e = get_block(h_edge, bi, bj)
            bv_e = get_block(v_edge, bi, bj)
            if len(bh_e) > 0 and len(bv_e) > 0:
                hs = float(np.mean(bh_e))
                vs = float(np.mean(bv_e))
                ms = max(hs, vs)
                feats[3] = ms                           # edge_strength
                feats[4] = abs(hs - vs) / ms if ms > 1e-6 else 0.0  # edge_orientation_conf

            # second diff features: second_diff_max, second_diff_min_max
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
            row_sd = float(np.mean(row_energies)) if row_energies else 0.0
            col_sd = float(np.mean(col_energies)) if col_energies else 0.0
            feats[7] = max(row_sd, col_sd)              # second_diff_max
            feats[8] = (min(row_sd, col_sd) / max(row_sd, col_sd)) if max(row_sd, col_sd) > 0 else 0.0  # second_diff_min_max

            # row/col diff: row_diff_max, col_diff_max
            row_means = [float(block[r, :].mean()) for r in range(bh)]
            row_diffs = [abs(row_means[r+1]-row_means[r]) for r in range(bh-1)] if bh >= 2 else [0.0]
            col_means = [float(block[:, c].mean()) for c in range(bw)]
            col_diffs = [abs(col_means[c+1]-col_means[c]) for c in range(bw-1)] if bw >= 2 else [0.0]
            feats[13] = float(np.max(row_diffs)) if row_diffs else 0.0   # row_diff_max
            feats[15] = float(np.max(col_diffs)) if col_diffs else 0.0   # col_diff_max

            # ringing features
            try:
                ringing = fcr.ringing_stats_for_block(block, bh, bw)
                if ringing:
                    feats[5] = ringing['col_ringing_mean']       # col_ringing_mean
                    feats[6] = ringing['row_ringing_mean']       # row_ringing_mean
                    feats[9] = ringing['profile_ringing_mean']   # profile_ringing_mean
                    feats[10] = ringing['ringing_mean_min']      # ringing_mean_min
                    feats[11] = ringing['ringing_mean_min_max']  # ringing_mean_min_max
                    feats[12] = ringing['row_ringing_max']       # row_ringing_max
                    feats[14] = ringing['col_ringing_max']       # col_ringing_max
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

    div_arr = np.array([NORM_DIV[f] for f in features_list], dtype=np.float32)

    print("  Assembling neighborhoods & predicting...", end=" ", flush=True)
    t0 = time.time()

    pred_map = np.full((gh, gw), np.nan, dtype=np.float32)
    patches = []
    coords = []

    for bi in range(4, gh - 4):
        for bj in range(4, gw - 4):
            neigh = np.zeros((81, 16), dtype=np.float32)
            for i, (dr, dc) in enumerate(OFFSETS_9x9):
                neigh[i] = np.clip(grid[bi + dr, bj + dc] / div_arr, 0, 1)
            patch = neigh.reshape(9, 9, 16).transpose(2, 0, 1)
            patches.append(patch)
            coords.append((bi, bj))

    if patches:
        X = np.stack(patches, axis=0)
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            probs = model(X_t).cpu().numpy().flatten()
        for idx, (bi, bj) in enumerate(coords):
            pred_map[bi, bj] = probs[idx]

    print(f"[{time.time()-t0:.0f}s]")

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
    global FEATURE_IDX
    FEATURE_IDX = {f: i for i, f in enumerate(features_list)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MosquitoDenoiseCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == '--batch':
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
