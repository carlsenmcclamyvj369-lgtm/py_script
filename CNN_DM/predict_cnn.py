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
from pathlib import Path

from dm_cnn import MosquitoDenoiseCNN, features_list, NORM_DIV
import feature_compute_reference as fcr

# ─── Config ───
SCRIPT_DIR = os.path.dirname(__file__)
# MODEL_PATH = os.path.join(SCRIPT_DIR, "mosquito_denoise_cnn.pth")
MODEL_PATH = os.path.join(SCRIPT_DIR, "mosquito_denoise_cnn_4k.pth")
TEST_DIR = r"C:\code\py\denoise\scripts\CNN_DM\gen_pattern_img"
# TEST_DIR = r"C:\code\py\denoise\scripts\test_data\dot25"
# TEST_DIR = r"C:\code\py\denoise\scripts\test_data"
# TEST_DIR = r"C:\code\py\denoise\scripts\test_data"
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "predictions_gen")
# OUTPUT_DIR = os.path.join(SCRIPT_DIR, "predictions")

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
    """Compute 16 CNN features using vectorized torch ops. Returns (gh, gw, 16) numpy."""
    H, W = y_full.shape
    gh, gw = H // GS, W // GS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pixel-level maps (numpy + cv2)
    var_map = fcr.compute_var_map(y_full)
    h_edge, v_edge = fcr.compute_edge_maps(y_full)

    # Trim and convert to torch
    sl = slice(None, gh * GS), slice(None, gw * GS)
    y_t = torch.from_numpy(y_full[sl].copy()).float().to(device)
    var_t = torch.from_numpy(var_map[sl].copy()).float().to(device)
    he_t = torch.from_numpy(h_edge[sl].copy()).float().to(device)
    ve_t = torch.from_numpy(v_edge[sl].copy()).float().to(device)

    # (gh, 8, gw, 8) -> (gh, gw, 8, 8)
    def tb(t):
        return t.reshape(gh, GS, gw, GS).permute(0, 2, 1, 3).contiguous()

    yb = tb(y_t); vb = tb(var_t); hb = tb(he_t); veb = tb(ve_t)
    vf = vb.reshape(gh, gw, -1)
    hf = hb.reshape(gh, gw, -1)
    vff = veb.reshape(gh, gw, -1)

    # NaN-safe helpers
    def sf_mean(a):
        mask = torch.isnan(a)
        return torch.where(mask, torch.tensor(0.0, device=device), a).sum(dim=-1) / (~mask).float().sum(dim=-1).clamp(min=1)

    def sf_count_lt(a, th):
        mask = torch.isnan(a)
        return ((~mask) & (a < th)).sum(dim=-1).float()

    def sf_count_gt(a, th):
        mask = torch.isnan(a)
        return ((~mask) & (a > th)).sum(dim=-1).float()

    grid = torch.zeros((gh, gw, 16), device=device)

    # 0-2: var
    grid[..., 0] = sf_mean(vf)
    grid[..., 1] = sf_count_lt(vf, LOW_VAR_TH)
    grid[..., 2] = sf_count_gt(vf, HIGH_VAR_TH)

    # 3-4: edge
    hs = sf_mean(hf); vs = sf_mean(vff); ms = torch.max(hs, vs)
    grid[..., 3] = ms
    grid[..., 4] = torch.where(ms > 1e-6, (hs - vs).abs() / ms, torch.tensor(0.0, device=device))

    # 5-6: second diff
    d2r = yb[..., :-2] - 2.0 * yb[..., 1:-1] + yb[..., 2:]
    row_sd = d2r.abs().mean(dim=-1).mean(dim=-1)
    d2c = yb[:, :, :-2, :] - 2.0 * yb[:, :, 1:-1, :] + yb[:, :, 2:, :]
    col_sd = d2c.abs().mean(dim=-2).mean(dim=-1)
    sd_mx = torch.max(row_sd, col_sd)
    grid[..., 5] = sd_mx                                     # second_diff_max
    grid[..., 6] = torch.where(sd_mx > 0, torch.min(row_sd, col_sd) / sd_mx, torch.tensor(0.0, device=device))  # second_diff_min_max

    # 14-15: row/col diff
    rm = yb.mean(dim=-1)
    rd = (rm[..., 1:] - rm[..., :-1]).abs()
    grid[..., 14] = rd.amax(dim=-1)                        # row_diff_max
    cm = yb.mean(dim=-2)
    cd = (cm[..., 1:] - cm[..., :-1]).abs()
    grid[..., 15] = cd.amax(dim=-1)                        # col_diff_max

    # ── Ringing: batched torch ──
    def _ringing_batch(v):
        dv = torch.diff(v, dim=-1)
        if dv.shape[-1] < 2:
            return torch.zeros(v.shape[:-1], device=device), \
                   torch.zeros(v.shape[:-1], device=device), \
                   torch.zeros(v.shape[:-1], device=device), \
                   torch.zeros(v.shape[:-1], device=device)
        d1_for_d2 = torch.diff(v, dim=-1)
        d2 = torch.diff(d1_for_d2, dim=-1)

        dyn = v.amax(dim=-1) - v.amin(dim=-1)
        d2_en = d2.abs().mean(dim=-1)
        # sign changes
        sgn = torch.sign(dv)
        sgn[torch.abs(dv) < 3.0] = 0
        last = torch.zeros(v.shape[:-1], dtype=torch.long, device=device)
        chg = torch.zeros(v.shape[:-1], dtype=torch.long, device=device)
        for i in range(sgn.shape[-1]):
            nz = sgn[..., i] != 0
            hp = last != 0
            df = nz & hp & (sgn[..., i] != last)
            chg = torch.where(df, chg + 1, chg)
            last = torch.where(nz, sgn[..., i].long(), last)

        def norm_(x, lo, hi):
            return torch.clamp((x - lo) / (hi - lo), 0.0, 1.0) if hi > lo else torch.zeros_like(x)

        ds_ = norm_(dyn.float(), 20.0, 120.0)
        d2s_ = norm_(d2_en.float(), 5.0, 60.0)
        ss_ = norm_(chg.float(), 1.0, 4.0)
        return 0.45 * ds_ + 0.35 * d2s_ + 0.20 * ss_, ds_, d2s_, ss_

    rt, _, _, _ = _ringing_batch(yb)           # (gh,gw,8) row totals
    ct, _, _, _ = _ringing_batch(yb.permute(0, 1, 3, 2))  # (gh,gw,8) col totals

    rm_mean = rt.mean(dim=-1)                  # row_ringing_mean
    cm_mean = ct.mean(dim=-1)                  # col_ringing_mean
    rmx = torch.max(rm_mean, cm_mean)          # ringing_mean_max
    rmn = torch.min(rm_mean, cm_mean)          # ringing_mean_min

    grid[..., 7] = rmx                                             # ringing_mean_max
    grid[..., 8] = rmn                                             # ringing_mean_min
    grid[..., 9] = torch.where(rmx > 0, rmn / rmx, torch.tensor(0.0, device=device))  # ringing_mean_min_max
    grid[..., 10] = rt.amax(dim=-1)                                 # row_ringing_max
    grid[..., 11] = rm_mean                                         # row_ringing_mean
    grid[..., 12] = ct.amax(dim=-1)                                 # col_ringing_max
    grid[..., 13] = cm_mean                                         # col_ringing_mean

    return grid.cpu().numpy()

@torch.no_grad()
def predict_image(model, device, bmp_path, output_path, save_debug=True):
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

    print("  Assembling neighborhoods & predicting...", end=" ", flush=True)
    t0 = time.time()

    div_arr = np.array([NORM_DIV[f] for f in features_list], dtype=np.float32)

    # Repeat-pad grid → sliding_window_view → fully vectorized
    from numpy.lib.stride_tricks import sliding_window_view
    grid_pad = np.pad(grid, ((4, 4), (4, 4), (0, 0)), mode='edge')
    grid_norm = np.clip(grid_pad / div_arr, 0, 1)
    # windows = sliding_window_view(grid_norm, (9, 9), axis=(0, 1))
    # windows shape: (gh, gw, n_feat, 9, 9) → (gh, gw, 9, 9, n_feat)
    # windows = windows.transpose(0, 1, 3, 4, 2)
    X = np.ascontiguousarray(grid_norm).reshape( 1, (8+gh) , (8+gw), 16).transpose(0, 3, 1, 2)
    # print("grid shape: ", grid.shape)
    # print("grid_pad shape: ", grid_pad.shape)
    # print("windows shape: ", windows.shape)
    # print("X shape: ", X.shape)

    pred_map = np.full((gh, gw), np.nan, dtype=np.float32)
    if len(X) > 0:
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            probs = model(X_t).cpu().numpy().flatten()
        pred_map.ravel()[:] = probs

    print(f"[{time.time()-t0:.0f}s]")

    dm_count = int(np.nansum(pred_map > 0.5))
    valid_count = int(np.sum(~np.isnan(pred_map)))
    print(f"  DM: {dm_count}/{valid_count} ({100*dm_count/max(valid_count,1):.1f}%)")

    y_norm = np.clip(y_full, 0, 255).astype(np.uint8)
    display = cv2.cvtColor(y_norm, cv2.COLOR_GRAY2BGR)

    for bi in range(gh):
        for bj in range(gw):
            p = pred_map[bi, bj]
            if np.isnan(p) or p <= 0.5:
                continue
            y1, y2 = bi*GS, min(bi*GS+GS, H)
            x1, x2 = bj*GS, min(bj*GS+GS, W)
            overlay = display[y1:y2, x1:x2].astype(np.float64)
            overlay[:, :, 2] = np.clip(overlay[:, :, 2]*0.6 + 255*0.4, 0, 255)
            display[y1:y2, x1:x2] = overlay.astype(np.uint8)

    display[0::8, :] = (100, 100, 100)
    display[:, 0::8] = (100, 100, 100)

    cv2.putText(display, f"CNN DM: {dm_count}/{valid_count} ({100*dm_count/max(valid_count,1):.1f}%)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if save_debug:
        cnn_path = os.path.join(OUTPUT_DIR, Path(bmp_path).stem + "_cnn.png")
        os.makedirs(os.path.dirname(cnn_path) or '.', exist_ok=True)
        cv2.imwrite(str(cnn_path), display)
        print(f"  Saved: {cnn_path}")

    # 7x7 双边滤波
    # d=7            → 邻域直径（kernel size）
    # sigmaColor    → 颜色空间标准差
    # sigmaSpace    → 坐标空间标准差
    filtered_img = cv2.bilateralFilter(bgr, d=7, sigmaColor=50, sigmaSpace=50)

    pred_map_8x8 = pred_map.repeat(8, axis=0).repeat(8, axis=1)
    # pred_map_8x8 shape: (gh*8, gw*8), 可能小于 (H, W) 如果 H/W 不是 8 的倍数
    ph = H - pred_map_8x8.shape[0]
    pw = W - pred_map_8x8.shape[1]
    if ph > 0 or pw > 0:
        pred_map_8x8 = np.pad(pred_map_8x8, ((0, ph), (0, pw)), mode='edge')
    pred_map_3c = pred_map_8x8[..., np.newaxis]  # (H, W, 1)
    pred_map_3c = np.repeat(pred_map_3c, 3, axis=2)  # (H, W, 3)

    # 保存原始输入（在 bgr 转 float32 之前）
    stem = Path(bmp_path).stem

    bgr_f32 = bgr.astype(np.float32)
    filtered_f32 = filtered_img.astype(np.float32)
    out_img = filtered_f32 * pred_map_3c + bgr_f32 * (1 - pred_map_3c)
    out_img = np.clip(out_img, 0, 255).astype(np.uint8)

    if save_debug == True:
        cv2.imwrite(os.path.join(OUTPUT_DIR, stem + "_in.bmp"), bgr)
        # cv2.imwrite(os.path.join(OUTPUT_DIR, stem + "_bilater.bmp"), filtered_img)
        cv2.imwrite(os.path.join(OUTPUT_DIR, stem + "_pred8x8.bmp"), (pred_map_3c * 255).astype(np.uint8))
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    cv2.imwrite(str(output_path), out_img)
    print(f"  Denoised: {output_path}")

def main():
    global FEATURE_IDX
    FEATURE_IDX = {f: i for i, f in enumerate(features_list)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MosquitoDenoiseCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    bmps = sorted([f for f in os.listdir(TEST_DIR) if f.endswith('.bmp')])
    print(f"Processing {len(bmps)} images...\n")
    for b in bmps:
        bmp_path = os.path.join(TEST_DIR, b)
        out_path = os.path.join(OUTPUT_DIR, b.replace('.bmp', '_out.bmp'))
        t0 = time.time()
        print(f"[{b}]")
        predict_image(model, device, bmp_path, out_path)
        print(f"  [{time.time() - t0:.0f}s]\n")


if __name__ == "__main__":
    main()
