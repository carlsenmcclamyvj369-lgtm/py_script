"""
Inference on a single BMP using the trained CNN model.
Computes 16 features per 8x8 block, assembles 9x9 neighborhoods,
runs the CNN to predict dm/not_dm, and overlays results on the image.

Added:
- Save/load intermediate grid features to Excel cache to avoid recomputing.
- Optional --force-recompute to ignore cache.
- Batch CNN inference to reduce peak memory.
"""

import numpy as np
import torch
import cv2
import os
import sys
import time
import re

from dm_cnn import MosquitoDenoiseCNN, features_list, NORM_DIV
import feature_compute_reference as fcr

# ─── Config ───
SCRIPT_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(SCRIPT_DIR, "mosquito_denoise_cnn.pth")
TEST_DIR = r"C:\code\py\denoise\scripts\test_data"
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "predictions")
FEATURE_CACHE_DIR = os.path.join(SCRIPT_DIR, "feature_cache")

GS = 8
OFFSETS_9x9 = [(dr, dc) for dr in range(-4, 5) for dc in range(-4, 5)]
BATCH_SIZE = 4096

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


def safe_stem(path):
    """Make a Windows-safe cache file stem."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r'[^0-9A-Za-z_.#\-]+', '_', stem)


def feature_cache_path(bmp_path):
    os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)
    return os.path.join(FEATURE_CACHE_DIR, safe_stem(bmp_path) + "_grid_features.npz")


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

    TOTAL = 49
    grid = np.zeros((gh, gw, TOTAL), dtype=np.float32)

    for bi in range(gh):
        for bj in range(gw):
            y1, y2 = bi * GS, min(bi * GS + GS, H)
            x1, x2 = bj * GS, min(bj * GS + GS, W)
            if y2 - y1 < 2 or x2 - x1 < 2:
                continue

            block = y_full[y1:y2, x1:x2]
            bh, bw = block.shape
            feats = np.zeros(TOTAL, dtype=np.float32)

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
                abs_br = np.abs(br)
                feats[6] = float(np.mean(abs_br))
                feats[7] = float(np.max(abs_br))

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
                hs = float(np.mean(bh_e))
                vs = float(np.mean(bv_e))
                ms = max(hs, vs)
                feats[12] = ms
                feats[13] = hs
                feats[14] = float(np.max(bh_e))
                feats[15] = float(np.min(bh_e))
                feats[16] = vs
                feats[17] = float(np.max(bv_e))
                feats[18] = float(np.min(bv_e))
                feats[19] = abs(hs - vs) / ms if ms > 1e-6 else 0.0

            row_energies = []
            for ry in range(bh):
                v = block[ry, :]
                if len(v) >= 3:
                    d2 = v[:-2] - 2 * v[1:-1] + v[2:]
                    row_energies.append(float(np.mean(np.abs(d2))))

            col_energies = []
            for rx in range(bw):
                v = block[:, rx]
                if len(v) >= 3:
                    d2 = v[:-2] - 2 * v[1:-1] + v[2:]
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
            feats[27] = (min(rd, cd) / max(rd, cd)) if max(rd, cd) > 0 else 0.0

            row_means = [float(block[r, :].mean()) for r in range(bh)]
            row_diffs = [abs(row_means[r + 1] - row_means[r]) for r in range(bh - 1)] if bh >= 2 else [0.0]
            col_means = [float(block[:, c].mean()) for c in range(bw)]
            col_diffs = [abs(col_means[c + 1] - col_means[c]) for c in range(bw - 1)] if bw >= 2 else [0.0]
            feats[28] = float(np.mean(row_diffs)) if row_diffs else 0.0
            feats[29] = float(np.max(row_diffs)) if row_diffs else 0.0
            feats[30] = float(np.mean(col_diffs)) if col_diffs else 0.0
            feats[31] = float(np.max(col_diffs)) if col_diffs else 0.0

            try:
                ringing = fcr.ringing_stats_for_block(block, bh, bw)
                if ringing:
                    for k, idx in [
                        ('row_ringing_max', 32), ('row_ringing_min', 33), ('row_ringing_mean', 34),
                        ('ringing_mean_max', 35), ('ringing_mean_min', 36), ('ringing_mean_min_max', 37),
                        ('profile_ringing_max', 38), ('profile_ringing_mean', 39),
                        ('row_ringing_dyn_score', 40), ('row_ringing_d2_score', 41), ('row_ringing_sign_score', 42),
                        ('col_ringing_max', 43), ('col_ringing_min', 44), ('col_ringing_mean', 45),
                        ('col_ringing_dyn_score', 46), ('col_ringing_d2_score', 47), ('col_ringing_sign_score', 48),
                    ]:
                        feats[idx] = ringing[k]
            except Exception:
                pass

            grid[bi, bj] = feats

    return grid


def save_grid_features_npz(grid, npz_path, image_name):
    """Save grid feature tensor (gh, gw, 49) to a .npz cache file."""
    np.savez_compressed(npz_path, grid=grid, image_name=image_name)


def load_grid_features_npz(npz_path):
    """Load grid feature tensor from .npz cache file. Returns (gh, gw, 49)."""
    data = np.load(npz_path)
    return data["grid"]


def compute_or_load_grid_features(y_full, bmp_path, force_recompute=False):
    """Load grid features from Excel cache if available; otherwise compute and save."""
    cache_path = feature_cache_path(bmp_path)

    if (not force_recompute) and os.path.exists(cache_path):
        print(f"  Loading cached grid features: {cache_path}", end=" ", flush=True)
        t0 = time.time()
        grid = load_grid_features_npz(cache_path)
        print(f"[{time.time() - t0:.1f}s]")
        return grid

    print("  Computing grid features...", end=" ", flush=True)
    t0 = time.time()
    grid = compute_grid_features(y_full)
    print(f"[{time.time() - t0:.1f}s]")

    print(f"  Saving grid features cache: {cache_path}", end=" ", flush=True)
    t0 = time.time()
    save_grid_features_npz(grid, cache_path, os.path.splitext(os.path.basename(bmp_path))[0])
    print(f"[{time.time() - t0:.1f}s]")

    return grid


def build_patches_from_grid(grid):
    """Assemble 9x9 neighborhoods with edge padding, covering ALL blocks.

    Grid is edge-padded by 4 blocks on each side so the 9x9 neighborhood
    is valid even for border blocks of the original image.

    Returns X:(N,16,9,9), coords list for every block (0..gh-1, 0..gw-1).
    """
    gh, gw, _ = grid.shape
    f_idx = [FEATURE_IDX[f] for f in features_list]
    div_arr = np.array([NORM_DIV[f] for f in features_list], dtype=np.float32)

    # Edge-pad grid to make 9x9 neighborhoods valid at borders
    pad = 4
    grid_pad = np.pad(grid[..., f_idx], ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    # Normalize
    grid_pad = np.clip(grid_pad / div_arr, 0, 1)

    # sliding_window_view gives (gh+2*pad-8, gw+2*pad-8, 9, 9, 16) = (gh, gw, 9, 9, 16)
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(grid_pad, (9, 9), axis=(0, 1))
    # windows shape: (gh, gw, 9, 9, 16)
    N = gh * gw
    X = np.ascontiguousarray(windows).reshape(N, 9, 9, 16).transpose(0, 3, 1, 2)

    coords = [(bi, bj) for bi in range(gh) for bj in range(gw)]
    return X, coords


def predict_patches_batch(model, device, X, batch_size=BATCH_SIZE):
    """Run CNN in batches to avoid too much memory and reduce overhead."""
    probs_list = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            batch_np = X[start:start + batch_size]
            batch_t = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)
            probs = model(batch_t).detach().cpu().numpy().reshape(-1)
            probs_list.append(probs)
    return np.concatenate(probs_list, axis=0)


def denoise_dm_blocks(y_channel, pred_map, gs=8, d=7, sigma_color=75, sigma_space=75):
    """Apply 7x7 bilateral filter only on DM-predicted blocks.

    Args:
        y_channel: (H, W) float64 Y channel from original image.
        pred_map:  (gh, gw) float32 prediction probabilities.
        gs:        grid size (8).
        d:         bilateral filter diameter (7).
        sigma_color, sigma_space: bilateral filter parameters.

    Returns:
        denoised: (H, W) float64, filtered on DM blocks, original elsewhere.
    """
    H, W = y_channel.shape

    # Compute bilateral filter once for the whole image
    y_u8 = np.clip(np.round(y_channel), 0, 255).astype(np.uint8)
    y_bf = cv2.bilateralFilter(y_u8, d, sigma_color, sigma_space).astype(np.float64)

    denoised = y_channel.copy()
    dm_mask = pred_map > 0.5

    gh, gw = pred_map.shape
    for bi in range(gh):
        for bj in range(gw):
            if not dm_mask[bi, bj]:
                continue
            y1, y2 = bi * gs, min(bi * gs + gs, H)
            x1, x2 = bj * gs, min(bj * gs + gs, W)
            denoised[y1:y2, x1:x2] = y_bf[y1:y2, x1:x2]

    return denoised


def predict_image(model, device, bmp_path, output_path, force_recompute=False, denoise=False):
    """Run CNN on a BMP, save overlay and optionally denoised output."""
    bgr = cv2.imread(bmp_path, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"ERROR: cannot read {bmp_path}")
        return

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    y_full = fcr.compute_y_from_rgb(rgb)
    H, W = y_full.shape
    gh, gw = H // GS, W // GS
    print(f"  Image: {W}x{H}, grid: {gw}x{gh}")

    grid = compute_or_load_grid_features(y_full, bmp_path, force_recompute=force_recompute)

    print("  Assembling neighborhoods & predicting...", end=" ", flush=True)
    t0 = time.time()

    X, coords = build_patches_from_grid(grid)
    pred_map = np.zeros((gh, gw), dtype=np.float32)

    if X is not None and len(coords) > 0:
        probs = predict_patches_batch(model, device, X, batch_size=BATCH_SIZE)
        for idx, (bi, bj) in enumerate(coords):
            pred_map[bi, bj] = probs[idx]

    print(f"[{time.time() - t0:.1f}s]")

    dm_count = int(np.sum(pred_map > 0.5))
    valid_count = gh * gw
    print(f"  DM: {dm_count}/{valid_count} ({100 * dm_count / max(valid_count, 1):.1f}%)")

    y_norm = np.clip(y_full, 0, 255).astype(np.uint8)
    display = cv2.cvtColor(y_norm, cv2.COLOR_GRAY2BGR)

    # Overlay (all blocks are valid now thanks to edge padding)
    dm_mask = pred_map > 0.5
    for bi in range(gh):
        for bj in range(gw):
            if not dm_mask[bi, bj]:
                continue
            y1, y2 = bi * GS, min(bi * GS + GS, H)
            x1, x2 = bj * GS, min(bj * GS + GS, W)
            block = display[y1:y2, x1:x2].astype(np.float64)
            block[:, :, 2] = np.clip(block[:, :, 2] * 0.6 + 255 * 0.4, 0, 255)
            display[y1:y2, x1:x2] = block.astype(np.uint8)

    # Grid lines via numpy slicing
    display[0::8, :] = (100, 100, 100)
    display[:, 0::8] = (100, 100, 100)

    cv2.putText(
        display,
        f"CNN DM: {dm_count}/{valid_count} ({100 * dm_count / max(valid_count, 1):.1f}%)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    cv2.imwrite(str(output_path), display)
    print(f"  Saved: {output_path}")

    # ─── Denoise output ───
    if denoise:
        t0 = time.time()
        denoised_y = denoise_dm_blocks(y_full, pred_map, gs=GS, d=7, sigma_color=75, sigma_space=75)

        # Merge Y back with original UV (YCbCr)
        ycbcr = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float64)
        ycbcr[:, :, 0] = denoised_y
        denoised_bgr = cv2.cvtColor(ycbcr.astype(np.uint8), cv2.COLOR_YCrCb2BGR)

        denoised_path = output_path.replace('_cnn.png', '_denoised.png')
        cv2.imwrite(denoised_path, denoised_bgr)
        print(f"  Denoised: {denoised_path} [{time.time() - t0:.1f}s]")


def main():
    force_recompute = "--force-recompute" in sys.argv
    do_denoise = "--denoise" in sys.argv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if do_denoise:
        print("  Mode: overlay + 7x7 bilateral denoise on DM blocks")

    model = MosquitoDenoiseCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")

    skip_flags = {"--force-recompute", "--denoise"}
    args = [a for a in sys.argv[1:] if a not in skip_flags]

    if len(args) > 0:
        arg = args[0]
        if arg == '--batch':
            batch_dir = args[1] if len(args) > 1 else TEST_DIR
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            bmps = sorted([f for f in os.listdir(batch_dir) if f.lower().endswith('.bmp')])
            print(f"Processing {len(bmps)} images from {batch_dir}...\n")
            for b in bmps:
                bmp_path = os.path.join(batch_dir, b)
                out_path = os.path.join(OUTPUT_DIR, b.replace('.bmp', '_cnn.png'))
                t0 = time.time()
                print(f"[{b}]")
                predict_image(model, device, bmp_path, out_path,
                              force_recompute=force_recompute, denoise=do_denoise)
                print(f"  [{time.time() - t0:.1f}s]\n")
        else:
            bmp_path = arg if os.path.exists(arg) else os.path.join(TEST_DIR, arg)
            out_path = os.path.join(OUTPUT_DIR, os.path.basename(bmp_path).replace('.bmp', '_cnn.png'))
            print(f"\nProcessing: {bmp_path}")
            predict_image(model, device, bmp_path, out_path,
                          force_recompute=force_recompute, denoise=do_denoise)
    else:
        print("Usage:")
        print("  python predict_cnn_with_feature_cache.py <bmp_path|filename>                单张图片")
        print("  python predict_cnn_with_feature_cache.py --batch [目录]                     批量处理（默认 test_data）")
        print("  python predict_cnn_with_feature_cache.py --batch [目录] --denoise           批量+去噪")
        print("  python predict_cnn_with_feature_cache.py --batch [目录] --force-recompute   忽略缓存重算")


if __name__ == "__main__":
    main()
