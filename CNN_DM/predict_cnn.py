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
COST_DOWN = True  # 与训练时的 cost_down 一致
GS = 8 # 与训练时的 GS 一致
SCRIPT_DIR = os.path.dirname(__file__)
suffix = "_cost_down" if COST_DOWN else ""
MODEL_PATH = os.path.join(SCRIPT_DIR, f"./model/mosquito_denoise_cnn_cost_down_grid_{GS}.pth")
TEST_DIR = os.path.join(SCRIPT_DIR, "test_data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, f"predictions{suffix}_grid_{GS}")

# 训练时保存的最佳阈值（不存在则用 0.5）
DM_TH = 0.5

OFFSETS_9x9 = [(dr, dc) for dr in range(-4, 5) for dc in range(-4, 5)]
FEATURE_IDX = None  # set at runtime
LOW_VAR_TH = 20
HIGH_VAR_TH = 128


def get_block(map2d, bi, bj):
    y1, y2 = bi * GS, min(bi * GS + GS, map2d.shape[0])
    x1, x2 = bj * GS, min(bj * GS + GS, map2d.shape[1])
    vals = map2d[y1:y2, x1:x2].flatten()
    return vals[~np.isnan(vals)]


def compute_grid_features(y_full):
    """Compute 16 CNN features using vectorized torch ops. Returns (gh, gw, 16) numpy."""
    H, W = y_full.shape
    gh, gw = H // GS, W // GS
    # gh, gw = (H + GS - 1) // GS, (W + GS - 1) // GS
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

    yb = tb(y_t);
    vb = tb(var_t);
    hb = tb(he_t);
    veb = tb(ve_t)
    vf = vb.reshape(gh, gw, -1)
    hf = hb.reshape(gh, gw, -1)
    vff = veb.reshape(gh, gw, -1)

    # NaN-safe helpers
    def sf_mean(a, factor):
        mask = torch.isnan(a)
        return torch.where(mask, torch.tensor(0.0, device=device), a).sum(dim=-1) * factor // (
                    (~mask).float().sum(dim=-1).clamp(min=1) * 4)

    def sf_count_lt(a, th):
        mask = torch.isnan(a)
        return ((~mask) & (a < th)).sum(dim=-1).float()

    def sf_count_gt(a, th):
        mask = torch.isnan(a)
        return ((~mask) & (a > th)).sum(dim=-1).float()

    grid = torch.zeros((gh, gw, 16), device=device)

    # 0-2: var
    grid[..., 0] = torch.clamp(torch.floor(sf_mean(vf, 4)), 0, 255)
    grid[..., 1] = torch.clamp(sf_count_lt(vf, LOW_VAR_TH) * (16 // GS) * (16 // GS), 0, 255)
    grid[..., 2] = torch.clamp(sf_count_gt(vf, HIGH_VAR_TH) * (16 // GS) * (16 // GS), 0, 255)

    # 3-4: edge
    hs = sf_mean(hf, 4);
    vs = sf_mean(vff, 4);
    ms = torch.max(hs, vs)
    grid[..., 3] = torch.clamp(ms, 0, 255)
    grid[..., 4] = torch.where(ms > 0, (hs - vs).abs() * 255 // ms, torch.tensor(0.0, device=device))
    grid[..., 4] = torch.clamp(grid[..., 4], 0, 255)

    # 5-6: second diff
    d2r = yb[..., :-2] - 2.0 * yb[..., 1:-1] + yb[..., 2:]
    row_sd = torch.clamp(d2r.abs().sum(dim=-1).sum(dim=-1) * 4 // (GS * GS * 4), 0, 255)
    d2c = yb[:, :, :-2, :] - 2.0 * yb[:, :, 1:-1, :] + yb[:, :, 2:, :]
    col_sd = torch.clamp(d2c.abs().sum(dim=-2).sum(dim=-1) * 4 // (GS * GS * 4), 0, 255)
    sd_mx = torch.max(row_sd, col_sd)
    grid[..., 5] = sd_mx  # second_diff_max
    grid[..., 6] = torch.where(sd_mx > 0, torch.min(row_sd, col_sd) * 255 // sd_mx,
                               torch.tensor(0.0, device=device))  # second_diff_min_max
    grid[..., 6] = torch.clamp(grid[..., 6], 0, 255)

    # 14-15: row/col diff
    rm = torch.floor(yb.mean(dim=-1))
    rd = (rm[..., 1:] - rm[..., :-1]).abs()
    grid[..., 14] = torch.clamp(rd.amax(dim=-1), 0, 255)  # row_diff_max
    cm = torch.floor(yb.mean(dim=-2))
    cd = (cm[..., 1:] - cm[..., :-1]).abs()
    grid[..., 15] = torch.clamp(cd.amax(dim=-1), 0, 255)  # col_diff_max

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
        d2_en = torch.floor(d2.abs().mean(dim=-1))
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

        def nr_curve2_tensor(input_val, reg_xth, reg_yth):
            xth1 = reg_xth[0]
            xth2 = xth1 + (1 << reg_xth[1])
            xth3 = xth2 + (1 << reg_xth[2])
            yth1, yth2, yth3 = reg_yth
            half1 = reg_xth[1] if reg_xth[1] <= 1 else (1 << (reg_xth[1] - 1))
            half2 = reg_xth[2] if reg_xth[2] <= 1 else (1 << (reg_xth[2] - 1))
            output = torch.empty_like(input_val)
            mask1 = input_val <= xth1
            mask2 = input_val >= xth3
            mask3 = (input_val > xth1) & (input_val <= xth2)
            mask4 = (input_val > xth2) & (input_val < xth3)

            output[mask1] = yth1
            output[mask2] = yth3
            delt = input_val[mask3] - xth1
            output[mask3] = yth1 + (((yth2 - yth1) * delt + half1) >> reg_xth[1])
            delt = input_val[mask4] - xth2
            output[mask4] = yth2 + (((yth3 - yth2) * delt + half2) >> reg_xth[2])
            return output

        ds_ = nr_curve2_tensor(dyn.long(), [20, 5, 6], [0, 85, 255])
        d2s_ = nr_curve2_tensor(d2_en.long(), [5, 4, 5], [0, 85, 255])
        reg_dms_sign_change_cnt_score = torch.tensor([0, 0, 85, 171, 255, 255, 255, 255])
        ss_ = reg_dms_sign_change_cnt_score[(chg // (GS // 8))]
        score = torch.clamp((115 * ds_ + 89 * d2s_ + 52 * ss_) // 256, 0, 255)
        return score, ds_, d2s_, ss_

    rt, _, _, _ = _ringing_batch(yb)  # (gh,gw,8) row totals
    ct, _, _, _ = _ringing_batch(yb.permute(0, 1, 3, 2))  # (gh,gw,8) col totals

    rm_mean = torch.floor(rt.float().mean(dim=-1))  # row_ringing_mean
    cm_mean = torch.floor(ct.float().mean(dim=-1))  # col_ringing_mean
    rmx = torch.max(rm_mean, cm_mean)  # ringing_mean_max
    rmn = torch.min(rm_mean, cm_mean)  # ringing_mean_min

    grid[..., 7] = rmx  # ringing_mean_max
    grid[..., 8] = rmn  # ringing_mean_min
    grid[..., 9] = torch.where(rmx > 0, rmn * 255 // rmx, torch.tensor(0.0, device=device))  # ringing_mean_min_max
    grid[..., 10] = rt.amax(dim=-1)  # row_ringing_max
    grid[..., 11] = rm_mean  # row_ringing_mean
    grid[..., 12] = ct.amax(dim=-1)  # col_ringing_max
    grid[..., 13] = cm_mean  # col_ringing_mean

    return grid.cpu().numpy()


def grid_print(grid):
    grid_int = grid.astype(np.int32)
    with open("grid.txt", "w") as f:
        H, W, C = grid_int.shape

        for h in range(H):
            for w in range(W):
                f.write(f"[{h:03d},{w:03d}] ")
                f.write(" ".join("%05d" % x for x in grid_int[h, w]))
                f.write("\n")


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
    grid_print(grid)
    print(f"[{time.time() - t0:.0f}s]")

    print("  Assembling neighborhoods & predicting...", end=" ", flush=True)
    t0 = time.time()

    div_arr = np.array([NORM_DIV[f] for f in features_list], dtype=np.float32)

    # Repeat-pad grid → sliding_window_view → fully vectorized
    from numpy.lib.stride_tricks import sliding_window_view
    grid_pad = np.pad(grid, ((4, 4), (4, 4), (0, 0)), mode='edge')
    # grid_norm = np.clip(grid_pad / div_arr, 0, 1)
    if COST_DOWN:
        grid_norm = np.clip(grid_pad / 255, 0, 1)
    else:
        grid_norm = np.clip(grid_pad / 255, 0, 1)

    X = np.ascontiguousarray(grid_norm).reshape(1, (8 + gh), (8 + gw), 16).transpose(0, 3, 1, 2)

    pred_map = np.full((gh, gw), np.nan, dtype=np.float32)
    if len(X) > 0:
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            probs = model(X_t).cpu().numpy().flatten()
        pred_map.ravel()[:] = probs

    print(f"[{time.time() - t0:.0f}s]")

    dm_count = int(np.nansum(pred_map > DM_TH))
    valid_count = int(np.sum(~np.isnan(pred_map)))
    print(f"  DM: {dm_count}/{valid_count} ({100 * dm_count / max(valid_count, 1):.1f}%)")

    y_norm = np.clip(y_full, 0, 255).astype(np.uint8)
    display = cv2.cvtColor(y_norm, cv2.COLOR_GRAY2BGR)

    for bi in range(gh):
        for bj in range(gw):
            p = pred_map[bi, bj]
            if np.isnan(p) or p <= DM_TH:
                continue
            y1, y2 = bi * GS, min(bi * GS + GS, H)
            x1, x2 = bj * GS, min(bj * GS + GS, W)
            overlay = display[y1:y2, x1:x2].astype(np.float64)
            overlay[:, :, 2] = np.clip(overlay[:, :, 2] * 0.6 + 255 * 0.4, 0, 255)
            display[y1:y2, x1:x2] = overlay.astype(np.uint8)

    # display[0::GS, :] = (100, 100, 100)
    # display[:, 0::GS] = (100, 100, 100)

    cv2.putText(display, f"CNN DM: {dm_count}/{valid_count} ({100 * dm_count / max(valid_count, 1):.1f}%)",
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

    pred_map_8x8 = pred_map.repeat(GS, axis=0).repeat(GS, axis=1)
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
        cv2.imwrite(os.path.join(OUTPUT_DIR, stem + "_pred8x8.bmp"), (pred_map_3c * 255).astype(np.uint8))
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    cv2.imwrite(str(output_path), out_img)
    print(f"  Denoised: {output_path}")


def main():
    global FEATURE_IDX
    FEATURE_IDX = {f: i for i, f in enumerate(features_list)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MosquitoDenoiseCNN(cost_down=COST_DOWN).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    model.eval()

    print(f"Model loaded from {MODEL_PATH} (cost_down={COST_DOWN})")

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
