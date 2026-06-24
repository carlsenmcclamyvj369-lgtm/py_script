"""
Inference on a single BMP image using the trained LGBM model.
Computes 45 grid-level features, predicts dm/not_dm per 8x8 block,
and overlays results on the original image.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import cv2
import os
import sys

# ─── Config ───────────────────────────────────
BMP_PATH = r"C:\code\py\denoise\scripts\mos_featue_analyze\excel\00038#out1#mnr_input0010.bmp"
BMP_PATH = r"C:\code\py\denoise\scripts\mos_featue_analyze\excel\MNR_015#out1#mnr_input0006.bmp"
MODEL_PATH = r"C:\code\py\denoise\scripts\mos_featue_analyze\excel\all_data_norm_ai_ne_model.txt"
OUTPUT_PATH = r"C:\code\py\denoise\scripts\mos_featue_analyze\excel\00038_prediction.png"

GRID_SIZE = 8  # gx = gy = 8

# Normalization divisors (same as train_data_norm generation)
NORM_DIV = {
    'mean_var': 1020, 'max_var': 1020, 'top5_var': 1020,
    'low_var_count': 64, 'high_var_count': 64, 'very_high_var_count': 64,
    'residual_mean': 255, 'residual_max': 255,
    'lap_mean': 1020, 'lap_max': 1020,
    'grad_mean': 1020, 'grad_max': 1020,
    'edge_strength': 255,
    'h_strength': 255, 'h_strength_max': 255, 'h_strength_min': 255,
    'v_strength': 255, 'v_strength_max': 255, 'v_strength_min': 255,
    'edge_orientation_conf': 1.0,
    'row_second_diff': 510, 'row_second_diff_max': 510, 'row_second_diff_min': 510,
    'col_second_diff': 510, 'col_second_diff_max': 510, 'col_second_diff_min': 510,
    'second_diff_max': 510,
    'second_diff_min_max': 1.0,
    'profile_ringing_max': 1.0, 'profile_ringing_mean': 1.0,
    'ringing_mean_max': 1.0, 'ringing_mean_min': 1.0, 'ringing_mean_min_max': 1.0,
    'row_ringing_max': 1.0, 'row_ringing_min': 1.0, 'row_ringing_mean': 1.0,
    'row_ringing_dyn_score': 1.0, 'row_ringing_d2_score': 1.0, 'row_ringing_sign_score': 1.0,
    'col_ringing_max': 1.0, 'col_ringing_min': 1.0, 'col_ringing_mean': 1.0,
    'col_ringing_dyn_score': 1.0, 'col_ringing_d2_score': 1.0, 'col_ringing_sign_score': 1.0,
}

# Thresholds from markdown
LOW_VAR_TH = 100
HIGH_VAR_TH = 500
VERY_HIGH_VAR_TH = 2000
MAX_STRENGTH_TH = 0.000001
DYN_LO, DYN_HI = 20, 120
D2_LO, D2_HI = 5, 60
SIGN_LO, SIGN_HI = 1, 4
EPS = 1e-8


def _norm(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0, 1)


def load_y_from_bmp(path):
    """Load BMP, convert to BT.709 Y channel (float64)."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"ERROR: Cannot read {path}")
        sys.exit(1)
    # BT.709 Y = 0.2126R + 0.7152G + 0.0722B
    b, g, r = bgr[:, :, 0].astype(np.float64), bgr[:, :, 1].astype(np.float64), bgr[:, :, 2].astype(np.float64)
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return y


def compute_features(y):
    """
    Compute all 45 grid-level features from Y channel.
    Returns (H//8, W//8, 45) array and feature names list.
    """
    H, W = y.shape
    gh, gw = H // GRID_SIZE, W // GRID_SIZE

    # ─── Pixel-level intermediate maps ───
    # 1. 3x3 mean (border reflect)
    kernel = np.ones((3, 3), np.float64) / 9.0
    mean3 = cv2.filter2D(y, -1, kernel, borderType=cv2.BORDER_REFLECT)

    # 2. Variance: var = mean(Y²) - mean(Y)²
    y2 = y ** 2
    mean3_y2 = cv2.filter2D(y2, -1, kernel, borderType=cv2.BORDER_REFLECT)
    var_map = np.maximum(mean3_y2 - mean3 ** 2, 0)
    # Set 1-pixel border to NaN
    var_map[0, :] = var_map[-1, :] = var_map[:, 0] = var_map[:, -1] = np.nan

    # 3. Residual: |Y - 3x3_mean|
    residual_map = np.abs(y - mean3)
    residual_map[0, :] = residual_map[-1, :] = residual_map[:, 0] = residual_map[:, -1] = np.nan

    # 4. Laplacian: |lap|
    lap_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float64)
    lap_map = np.abs(cv2.filter2D(y, -1, lap_kernel, borderType=cv2.BORDER_REFLECT))
    lap_map[0, :] = lap_map[-1, :] = lap_map[:, 0] = lap_map[:, -1] = np.nan

    # 5. Gradient: 4-direction sum of absolute diffs
    grad_map = np.zeros_like(y)
    grad_map[1:-1, 1:-1] = (
        np.abs(y[1:-1, 1:-1] - y[0:-2, 1:-1]) +
        np.abs(y[1:-1, 1:-1] - y[2:, 1:-1]) +
        np.abs(y[1:-1, 1:-1] - y[1:-1, 0:-2]) +
        np.abs(y[1:-1, 1:-1] - y[1:-1, 2:])
    )
    grad_map[0, :] = grad_map[-1, :] = grad_map[:, 0] = grad_map[:, -1] = np.nan

    # 6. Edge: horizontal and vertical gradients
    h_edge = np.abs(np.diff(y, axis=1))
    h_edge = np.pad(h_edge, ((0, 0), (1, 0)), constant_values=np.nan)  # col 0 = NaN
    v_edge = np.abs(np.diff(y, axis=0))
    v_edge = np.pad(v_edge, ((1, 0), (0, 0)), constant_values=np.nan)  # row 0 = NaN

    # ─── Grid aggregation ───
    def grid_agg(map_2d, func, default=np.nan):
        """Apply func to each 8x8 grid block."""
        result = np.full((gh, gw), default, dtype=np.float64)
        valid = ~np.isnan(map_2d)
        for gy in range(gh):
            for gx in range(gw):
                block = map_2d[gy*GRID_SIZE:(gy+1)*GRID_SIZE, gx*GRID_SIZE:(gx+1)*GRID_SIZE]
                vals = block[~np.isnan(block)]
                if len(vals) == 0:
                    continue
                result[gy, gx] = func(vals)
        return result

    def grid_apply(map_2d, fn):
        """Apply fn to each block's flat valid values."""
        result = np.full((gh, gw), np.nan, dtype=np.float64)
        for gy in range(gh):
            for gx in range(gw):
                block = map_2d[gy*GRID_SIZE:(gy+1)*GRID_SIZE, gx*GRID_SIZE:(gx+1)*GRID_SIZE]
                vals = block[~np.isnan(block)]
                if len(vals) == 0:
                    continue
                result[gy, gx] = fn(vals)
        return result

    # Variance features
    mean_var = grid_apply(var_map, np.mean)
    max_var = grid_apply(var_map, np.max)
    top5_var = grid_apply(var_map, lambda v: np.mean(np.sort(v)[-5:]) if len(v) >= 5 else np.mean(v))
    low_var_count = grid_apply(var_map, lambda v: float(np.sum(v < LOW_VAR_TH)))
    high_var_count = grid_apply(var_map, lambda v: float(np.sum(v > HIGH_VAR_TH)))
    very_high_var_count = grid_apply(var_map, lambda v: float(np.sum(v > VERY_HIGH_VAR_TH)))

    # Residual features
    residual_mean = grid_apply(residual_map, np.mean)
    residual_max = grid_apply(residual_map, np.max)

    # Laplacian features
    lap_mean = grid_apply(lap_map, np.mean)
    lap_max = grid_apply(lap_map, np.max)

    # Gradient features
    grad_mean = grid_apply(grad_map, np.mean)
    grad_max = grid_apply(grad_map, np.max)

    # Edge features
    h_strength = grid_apply(h_edge, np.mean)
    h_strength_max = grid_apply(h_edge, np.max)
    h_strength_min = grid_apply(h_edge, np.min)
    v_strength = grid_apply(v_edge, np.mean)
    v_strength_max = grid_apply(v_edge, np.max)
    v_strength_min = grid_apply(v_edge, np.min)
    edge_strength = np.maximum(h_strength, v_strength)
    # orientation confidence
    orient_conf = np.zeros_like(edge_strength)
    mask = edge_strength > MAX_STRENGTH_TH
    orient_conf[mask] = np.abs(h_strength[mask] - v_strength[mask]) / np.maximum(h_strength[mask], v_strength[mask])

    # ─── Second diff features ───
    row_second_diff = np.full((gh, gw), np.nan)
    row_second_diff_max = np.full((gh, gw), np.nan)
    row_second_diff_min = np.full((gh, gw), np.nan)
    col_second_diff = np.full((gh, gw), np.nan)
    col_second_diff_max = np.full((gh, gw), np.nan)
    col_second_diff_min = np.full((gh, gw), np.nan)

    for gy in range(gh):
        for gx in range(gw):
            block = y[gy*GRID_SIZE:(gy+1)*GRID_SIZE, gx*GRID_SIZE:(gx+1)*GRID_SIZE]
            # Row second diff: for each row, mean of |v[i]-2*v[i+1]+v[i+2]|
            row_energies = []
            for r in range(GRID_SIZE):
                row = block[r, :]
                d2 = np.abs(row[:-2] - 2 * row[1:-1] + row[2:])
                if len(d2) > 0:
                    row_energies.append(np.mean(d2))
            if row_energies:
                row_second_diff[gy, gx] = np.mean(row_energies)
                row_second_diff_max[gy, gx] = np.max(row_energies)
                row_second_diff_min[gy, gx] = np.min(row_energies)

            # Column second diff
            col_energies = []
            for c in range(GRID_SIZE):
                col = block[:, c]
                d2 = np.abs(col[:-2] - 2 * col[1:-1] + col[2:])
                if len(d2) > 0:
                    col_energies.append(np.mean(d2))
            if col_energies:
                col_second_diff[gy, gx] = np.mean(col_energies)
                col_second_diff_max[gy, gx] = np.max(col_energies)
                col_second_diff_min[gy, gx] = np.min(col_energies)

    second_diff_max = np.maximum(row_second_diff, col_second_diff)
    denom = np.maximum(row_second_diff, col_second_diff)
    second_diff_min_max = np.where(denom > EPS, np.minimum(row_second_diff, col_second_diff) / denom, 0.0)

    # ─── Ringing features ───
    row_ringing_max = np.full((gh, gw), np.nan)
    row_ringing_min = np.full((gh, gw), np.nan)
    row_ringing_mean = np.full((gh, gw), np.nan)
    row_ringing_dyn_score = np.full((gh, gw), np.nan)
    row_ringing_d2_score = np.full((gh, gw), np.nan)
    row_ringing_sign_score = np.full((gh, gw), np.nan)
    col_ringing_max = np.full((gh, gw), np.nan)
    col_ringing_min = np.full((gh, gw), np.nan)
    col_ringing_mean = np.full((gh, gw), np.nan)
    col_ringing_dyn_score = np.full((gh, gw), np.nan)
    col_ringing_d2_score = np.full((gh, gw), np.nan)
    col_ringing_sign_score = np.full((gh, gw), np.nan)

    for gy in range(gh):
        for gx in range(gw):
            block = y[gy*GRID_SIZE:(gy+1)*GRID_SIZE, gx*GRID_SIZE:(gx+1)*GRID_SIZE]

            # Per-row ringing scores
            row_scores = []
            row_dyn_scores = []
            row_d2_scores = []
            row_sign_scores = []
            for r in range(GRID_SIZE):
                row = block[r, :]
                dyn = np.max(row) - np.min(row)
                ds = _norm(dyn, DYN_LO, DYN_HI)
                # d2 energy
                d2 = np.abs(row[:-2] - 2 * row[1:-1] + row[2:])
                d2s = _norm(np.mean(d2) if len(d2) > 0 else 0, D2_LO, D2_HI)
                # sign changes
                d = np.diff(row)
                signs = np.sign(d)
                signs[np.abs(d) < 3.0] = 0  # eps=3.0
                signs = signs[signs != 0]
                if len(signs) > 1:
                    n_changes = float(np.sum(signs[:-1] != signs[1:]))
                else:
                    n_changes = 0.0
                ss = _norm(n_changes, SIGN_LO, SIGN_HI)
                # weighted score
                score = 0.45 * ds + 0.35 * d2s + 0.20 * ss
                row_scores.append(score)
                row_dyn_scores.append(ds)
                row_d2_scores.append(d2s)
                row_sign_scores.append(ss)

            if row_scores:
                row_ringing_max[gy, gx] = np.max(row_scores)
                row_ringing_min[gy, gx] = np.min(row_scores)
                row_ringing_mean[gy, gx] = np.mean(row_scores)
                row_ringing_dyn_score[gy, gx] = np.mean(row_dyn_scores)
                row_ringing_d2_score[gy, gx] = np.mean(row_d2_scores)
                row_ringing_sign_score[gy, gx] = np.mean(row_sign_scores)

            # Per-column ringing scores
            col_scores = []
            col_dyn_scores = []
            col_d2_scores = []
            col_sign_scores = []
            for c in range(GRID_SIZE):
                col = block[:, c]
                dyn = np.max(col) - np.min(col)
                ds = _norm(dyn, DYN_LO, DYN_HI)
                d2 = np.abs(col[:-2] - 2 * col[1:-1] + col[2:])
                d2s = _norm(np.mean(d2) if len(d2) > 0 else 0, D2_LO, D2_HI)
                d = np.diff(col)
                signs = np.sign(d)
                signs[np.abs(d) < 3.0] = 0
                signs = signs[signs != 0]
                if len(signs) > 1:
                    n_changes = float(np.sum(signs[:-1] != signs[1:]))
                else:
                    n_changes = 0.0
                ss = _norm(n_changes, SIGN_LO, SIGN_HI)
                score = 0.45 * ds + 0.35 * d2s + 0.20 * ss
                col_scores.append(score)
                col_dyn_scores.append(ds)
                col_d2_scores.append(d2s)
                col_sign_scores.append(ss)

            if col_scores:
                col_ringing_max[gy, gx] = np.max(col_scores)
                col_ringing_min[gy, gx] = np.min(col_scores)
                col_ringing_mean[gy, gx] = np.mean(col_scores)
                col_ringing_dyn_score[gy, gx] = np.mean(col_dyn_scores)
                col_ringing_d2_score[gy, gx] = np.mean(col_d2_scores)
                col_ringing_sign_score[gy, gx] = np.mean(col_sign_scores)

    # Combined ringing features
    ringing_mean_max = np.maximum(row_ringing_mean, col_ringing_mean)
    ringing_mean_min = np.minimum(row_ringing_mean, col_ringing_mean)
    denom_rm = np.maximum(row_ringing_mean, col_ringing_mean)
    ringing_mean_min_max = np.where(denom_rm > EPS, np.minimum(row_ringing_mean, col_ringing_mean) / denom_rm, 0.0)

    all_scores_max = np.maximum(row_ringing_max, col_ringing_max)
    all_scores_mean = np.where(
        (~np.isnan(row_ringing_max)) & (~np.isnan(col_ringing_max)),
        (np.nan_to_num(row_ringing_max, 0) + np.nan_to_num(col_ringing_max, 0)) / 2,
        np.nan
    )
    profile_ringing_max = np.maximum(row_ringing_max, col_ringing_max)
    # mean of all row+col scores
    profile_ringing_mean = np.full((gh, gw), np.nan)
    for gy in range(gh):
        for gx in range(gw):
            vals = []
            if not np.isnan(row_ringing_max[gy, gx]):
                for r in range(GRID_SIZE):
                    row = block = y[gy*GRID_SIZE:(gy+1)*GRID_SIZE, gx*GRID_SIZE:(gx+1)*GRID_SIZE][r, :]
                    dyn = np.max(row) - np.min(row)
                    ds = _norm(dyn, DYN_LO, DYN_HI)
                    d2 = np.abs(row[:-2] - 2 * row[1:-1] + row[2:])
                    d2s = _norm(np.mean(d2) if len(d2) > 0 else 0, D2_LO, D2_HI)
                    d = np.diff(row)
                    signs = np.sign(d)
                    signs[np.abs(d) < 3.0] = 0
                    signs = signs[signs != 0]
                    if len(signs) > 1:
                        n_changes = float(np.sum(signs[:-1] != signs[1:]))
                    else:
                        n_changes = 0.0
                    ss = _norm(n_changes, SIGN_LO, SIGN_HI)
                    vals.append(0.45 * ds + 0.35 * d2s + 0.20 * ss)
                for c in range(GRID_SIZE):
                    col = block = y[gy*GRID_SIZE:(gy+1)*GRID_SIZE, gx*GRID_SIZE:(gx+1)*GRID_SIZE][:, c]
                    dyn = np.max(col) - np.min(col)
                    ds = _norm(dyn, DYN_LO, DYN_HI)
                    d2 = np.abs(col[:-2] - 2 * col[1:-1] + col[2:])
                    d2s = _norm(np.mean(d2) if len(d2) > 0 else 0, D2_LO, D2_HI)
                    d = np.diff(col)
                    signs = np.sign(d)
                    signs[np.abs(d) < 3.0] = 0
                    signs = signs[signs != 0]
                    if len(signs) > 1:
                        n_changes = float(np.sum(signs[:-1] != signs[1:]))
                    else:
                        n_changes = 0.0
                    ss = _norm(n_changes, SIGN_LO, SIGN_HI)
                    vals.append(0.45 * ds + 0.35 * d2s + 0.20 * ss)
            if vals:
                profile_ringing_mean[gy, gx] = np.mean(vals)

    # ─── Assemble feature stack ───
    feature_maps = {
        'mean_var': mean_var, 'max_var': max_var, 'top5_var': top5_var,
        'low_var_count': low_var_count, 'high_var_count': high_var_count, 'very_high_var_count': very_high_var_count,
        'residual_mean': residual_mean, 'residual_max': residual_max,
        'lap_mean': lap_mean, 'lap_max': lap_max,
        'grad_mean': grad_mean, 'grad_max': grad_max,
        'edge_strength': edge_strength,
        'h_strength': h_strength, 'h_strength_max': h_strength_max, 'h_strength_min': h_strength_min,
        'v_strength': v_strength, 'v_strength_max': v_strength_max, 'v_strength_min': v_strength_min,
        'edge_orientation_conf': orient_conf,
        'row_second_diff': row_second_diff, 'row_second_diff_max': row_second_diff_max, 'row_second_diff_min': row_second_diff_min,
        'col_second_diff': col_second_diff, 'col_second_diff_max': col_second_diff_max, 'col_second_diff_min': col_second_diff_min,
        'second_diff_max': second_diff_max, 'second_diff_min_max': second_diff_min_max,
        'profile_ringing_max': profile_ringing_max, 'profile_ringing_mean': profile_ringing_mean,
        'ringing_mean_max': ringing_mean_max, 'ringing_mean_min': ringing_mean_min, 'ringing_mean_min_max': ringing_mean_min_max,
        'row_ringing_max': row_ringing_max, 'row_ringing_min': row_ringing_min, 'row_ringing_mean': row_ringing_mean,
        'row_ringing_dyn_score': row_ringing_dyn_score, 'row_ringing_d2_score': row_ringing_d2_score, 'row_ringing_sign_score': row_ringing_sign_score,
        'col_ringing_max': col_ringing_max, 'col_ringing_min': col_ringing_min, 'col_ringing_mean': col_ringing_mean,
        'col_ringing_dyn_score': col_ringing_dyn_score, 'col_ringing_d2_score': col_ringing_d2_score, 'col_ringing_sign_score': col_ringing_sign_score,
    }

    feature_names = list(feature_maps.keys())
    feature_stack = np.stack([feature_maps[n] for n in feature_names], axis=-1)  # (gh, gw, 45)

    return feature_stack, feature_names


def normalize_features(feature_stack, feature_names):
    """Normalize to [0, 1] using same factors as training."""
    result = np.copy(feature_stack)
    for i, name in enumerate(feature_names):
        div = NORM_DIV.get(name, 1.0)
        if div == 1.0:
            result[:, :, i] = np.clip(result[:, :, i], 0, 1)
        else:
            result[:, :, i] = np.clip(result[:, :, i] / div, 0, 1)
    return result


def visualize_prediction(y_img, pred_mask, proba_map, output_path):
    """Overlay prediction grid on the original Y image."""
    H, W = y_img.shape
    # Normalize Y to 0-255 for display
    y_norm = np.clip(y_img, 0, 255).astype(np.uint8)
    # Convert to BGR for color overlay
    display = cv2.cvtColor(y_norm, cv2.COLOR_GRAY2BGR)

    gh, gw = pred_mask.shape
    grid_h = H // gh
    grid_w = W // gw

    for gy in range(gh):
        for gx in range(gw):
            y1, y2 = gy * grid_h, (gy + 1) * grid_h
            x1, x2 = gx * grid_w, (gx + 1) * grid_w
            pred = pred_mask[gy, gx]
            prob = proba_map[gy, gx]

            if pred == 1:
                # dm: red overlay (semi-transparent)
                overlay = display[y1:y2, x1:x2].astype(np.float64)
                overlay[:, :, 2] = np.clip(overlay[:, :, 2] * 0.6 + 255 * 0.4, 0, 255)  # Red channel
                display[y1:y2, x1:x2] = overlay.astype(np.uint8)
            # Draw grid lines
            cv2.rectangle(display, (x1, y1), (x2, y2), (100, 100, 100), 1)

    # Add legend
    cv2.putText(display, "Red = DM (mosquito noise)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    total = gh * gw
    dm_count = int(np.sum(pred_mask))
    cv2.putText(display, f"DM: {dm_count}/{total} ({100*dm_count/total:.1f}%)", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imwrite(output_path, display)
    print(f"Saved visualization to {output_path}")
    return display


def main():
    print("Loading model...")
    model = lgb.Booster(model_file=MODEL_PATH)

    print("Loading BMP...")
    y = load_y_from_bmp(BMP_PATH)
    H, W = y.shape
    print(f"  Size: {W}x{H}")

    print("Computing features...")
    feat_stack, feat_names = compute_features(y)
    gh, gw = feat_stack.shape[0], feat_stack.shape[1]
    print(f"  Grid: {gw}x{gh} = {gh*gw} blocks")

    print("Normalizing...")
    feat_norm = normalize_features(feat_stack, feat_names)

    print("Predicting...")
    # Reshape to (N, 45) for model
    n_blocks = gh * gw
    X = feat_norm.reshape(n_blocks, -1)
    # Handle NaN: fill with 0 (treat as no-detection for missing blocks)
    X = np.nan_to_num(X, 0)
    proba = model.predict(X)  # (N,) regression output
    pred = (proba.round().clip(0, 1)).astype(np.int32)

    proba_map = proba.reshape(gh, gw)
    pred_map = pred.reshape(gh, gw)

    dm_count = int(np.sum(pred))
    print(f"  DM blocks: {dm_count}/{n_blocks} ({100*dm_count/n_blocks:.1f}%)")

    print("Visualizing...")
    visualize_prediction(y, pred_map, proba_map, OUTPUT_PATH)

    print("Done!")


if __name__ == "__main__":
    main()
