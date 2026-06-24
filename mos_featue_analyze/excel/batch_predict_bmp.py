"""
Batch inference on all BMP images in test_data directory.
Uses the trained LGBM model to predict dm/not_dm per block
and saves overlay visualizations.
"""

import numpy as np
import lightgbm as lgb
import cv2
import os
import sys
import time
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count

# ─── Config ───────────────────────────────────
TEST_DIR = Path(r"C:\code\py\denoise\scripts\test_data")
OUTPUT_DIR = Path(r"C:\code\py\denoise\scripts\test_data\predictions")
MODEL_PATH = r"C:\code\py\denoise\scripts\mos_featue_analyze\excel\all_data_norm_ai_ne_model.txt"
NUM_WORKERS = max(1, cpu_count() - 1)  # leave 1 core free
GRID_SIZE = 8

# Normalization divisors
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

# Thresholds
LOW_VAR_TH = 100
HIGH_VAR_TH = 500
VERY_HIGH_VAR_TH = 2000
MAX_STRENGTH_TH = 0.000001
DYN_LO, DYN_HI = 20, 120
D2_LO, D2_HI = 5, 60
SIGN_LO, SIGN_HI = 1, 4
EPS_VAL = 1e-8


def _norm(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0, 1)


def load_y_from_bmp(path):
    """Load BMP, convert to BT.709 Y channel (float64)."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"  ERROR: Cannot read {path}")
        return None
    b, g, r = bgr[:, :, 0].astype(np.float64), bgr[:, :, 1].astype(np.float64), bgr[:, :, 2].astype(np.float64)
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return y


def compute_features(y):
    """Compute all 45 grid-level features from Y channel."""
    H, W = y.shape
    gh, gw = H // GRID_SIZE, W // GRID_SIZE

    # 3x3 mean
    kernel = np.ones((3, 3), np.float64) / 9.0
    mean3 = cv2.filter2D(y, -1, kernel, borderType=cv2.BORDER_REFLECT)

    # Variance
    y2 = y ** 2
    mean3_y2 = cv2.filter2D(y2, -1, kernel, borderType=cv2.BORDER_REFLECT)
    var_map = np.maximum(mean3_y2 - mean3 ** 2, 0)
    var_map[0, :] = var_map[-1, :] = var_map[:, 0] = var_map[:, -1] = np.nan

    # Residual
    residual_map = np.abs(y - mean3)
    residual_map[0, :] = residual_map[-1, :] = residual_map[:, 0] = residual_map[:, -1] = np.nan

    # Laplacian
    lap_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float64)
    lap_map = np.abs(cv2.filter2D(y, -1, lap_kernel, borderType=cv2.BORDER_REFLECT))
    lap_map[0, :] = lap_map[-1, :] = lap_map[:, 0] = lap_map[:, -1] = np.nan

    # Gradient 4-direction
    grad_map = np.zeros_like(y)
    grad_map[1:-1, 1:-1] = (
        np.abs(y[1:-1, 1:-1] - y[0:-2, 1:-1]) +
        np.abs(y[1:-1, 1:-1] - y[2:, 1:-1]) +
        np.abs(y[1:-1, 1:-1] - y[1:-1, 0:-2]) +
        np.abs(y[1:-1, 1:-1] - y[1:-1, 2:])
    )
    grad_map[0, :] = grad_map[-1, :] = grad_map[:, 0] = grad_map[:, -1] = np.nan

    # Edge: horizontal and vertical gradients
    h_edge = np.abs(np.diff(y, axis=1))
    h_edge = np.pad(h_edge, ((0, 0), (1, 0)), constant_values=np.nan)
    v_edge = np.abs(np.diff(y, axis=0))
    v_edge = np.pad(v_edge, ((1, 0), (0, 0)), constant_values=np.nan)

    def grid_apply(map_2d, fn):
        result = np.full((gh, gw), np.nan, dtype=np.float64)
        for gy in range(gh):
            for gx in range(gw):
                block = map_2d[gy*GRID_SIZE:(gy+1)*GRID_SIZE, gx*GRID_SIZE:(gx+1)*GRID_SIZE]
                vals = block[~np.isnan(block)]
                if len(vals) == 0:
                    continue
                result[gy, gx] = fn(vals)
        return result

    mean_var = grid_apply(var_map, np.mean)
    max_var = grid_apply(var_map, np.max)
    top5_var = grid_apply(var_map, lambda v: np.mean(np.sort(v)[-5:]) if len(v) >= 5 else np.mean(v))
    low_var_count = grid_apply(var_map, lambda v: float(np.sum(v < LOW_VAR_TH)))
    high_var_count = grid_apply(var_map, lambda v: float(np.sum(v > HIGH_VAR_TH)))
    very_high_var_count = grid_apply(var_map, lambda v: float(np.sum(v > VERY_HIGH_VAR_TH)))
    residual_mean = grid_apply(residual_map, np.mean)
    residual_max = grid_apply(residual_map, np.max)
    lap_mean = grid_apply(lap_map, np.mean)
    lap_max = grid_apply(lap_map, np.max)
    grad_mean = grid_apply(grad_map, np.mean)
    grad_max = grid_apply(grad_map, np.max)
    h_strength = grid_apply(h_edge, np.mean)
    h_strength_max = grid_apply(h_edge, np.max)
    h_strength_min = grid_apply(h_edge, np.min)
    v_strength = grid_apply(v_edge, np.mean)
    v_strength_max = grid_apply(v_edge, np.max)
    v_strength_min = grid_apply(v_edge, np.min)
    edge_strength = np.maximum(h_strength, v_strength)
    orient_conf = np.zeros_like(edge_strength)
    mask = edge_strength > MAX_STRENGTH_TH
    orient_conf[mask] = np.abs(h_strength[mask] - v_strength[mask]) / np.maximum(h_strength[mask], v_strength[mask])

    # Second diff
    row_second_diff = np.full((gh, gw), np.nan)
    row_second_diff_max = np.full((gh, gw), np.nan)
    row_second_diff_min = np.full((gh, gw), np.nan)
    col_second_diff = np.full((gh, gw), np.nan)
    col_second_diff_max = np.full((gh, gw), np.nan)
    col_second_diff_min = np.full((gh, gw), np.nan)

    for gy in range(gh):
        for gx in range(gw):
            block = y[gy*GRID_SIZE:(gy+1)*GRID_SIZE, gx*GRID_SIZE:(gx+1)*GRID_SIZE]
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
    second_diff_min_max = np.where(denom > EPS_VAL, np.minimum(row_second_diff, col_second_diff) / denom, 0.0)

    # Ringing per row/col of each block
    ring_names = ['ring_max', 'ring_min', 'ring_mean', 'dyn', 'd2', 'sign']
    row_data = {k: np.full((gh, gw), np.nan) for k in ring_names}
    col_data = {k: np.full((gh, gw), np.nan) for k in ring_names}

    for gy in range(gh):
        for gx in range(gw):
            block = y[gy*GRID_SIZE:(gy+1)*GRID_SIZE, gx*GRID_SIZE:(gx+1)*GRID_SIZE]

            row_scores, row_ds, row_d2s, row_ss = [], [], [], []
            for r in range(GRID_SIZE):
                row = block[r, :]
                dyn = np.max(row) - np.min(row)
                ds = _norm(dyn, DYN_LO, DYN_HI)
                d2 = np.abs(row[:-2] - 2 * row[1:-1] + row[2:])
                d2s = _norm(np.mean(d2) if len(d2) > 0 else 0, D2_LO, D2_HI)
                d = np.diff(row)
                signs = np.sign(d)
                signs[np.abs(d) < 3.0] = 0
                signs = signs[signs != 0]
                n_changes = float(np.sum(signs[:-1] != signs[1:])) if len(signs) > 1 else 0.0
                ss = _norm(n_changes, SIGN_LO, SIGN_HI)
                score = 0.45 * ds + 0.35 * d2s + 0.20 * ss
                row_scores.append(score)
                row_ds.append(ds)
                row_d2s.append(d2s)
                row_ss.append(ss)

            if row_scores:
                row_data['ring_max'][gy, gx] = np.max(row_scores)
                row_data['ring_min'][gy, gx] = np.min(row_scores)
                row_data['ring_mean'][gy, gx] = np.mean(row_scores)
                row_data['dyn'][gy, gx] = np.mean(row_ds)
                row_data['d2'][gy, gx] = np.mean(row_d2s)
                row_data['sign'][gy, gx] = np.mean(row_ss)

            col_scores, col_ds, col_d2s, col_ss = [], [], [], []
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
                n_changes = float(np.sum(signs[:-1] != signs[1:])) if len(signs) > 1 else 0.0
                ss = _norm(n_changes, SIGN_LO, SIGN_HI)
                score = 0.45 * ds + 0.35 * d2s + 0.20 * ss
                col_scores.append(score)
                col_ds.append(ds)
                col_d2s.append(d2s)
                col_ss.append(ss)

            if col_scores:
                col_data['ring_max'][gy, gx] = np.max(col_scores)
                col_data['ring_min'][gy, gx] = np.min(col_scores)
                col_data['ring_mean'][gy, gx] = np.mean(col_scores)
                col_data['dyn'][gy, gx] = np.mean(col_ds)
                col_data['d2'][gy, gx] = np.mean(col_d2s)
                col_data['sign'][gy, gx] = np.mean(col_ss)

    # Combined ringing features
    rr_max, rr_min, rr_mean = row_data['ring_max'], row_data['ring_min'], row_data['ring_mean']
    cr_max, cr_min, cr_mean = col_data['ring_max'], col_data['ring_min'], col_data['ring_mean']

    ringing_mean_max = np.maximum(rr_mean, cr_mean)
    ringing_mean_min = np.minimum(rr_mean, cr_mean)
    denom_rm = np.maximum(rr_mean, cr_mean)
    ringing_mean_min_max = np.where(denom_rm > EPS_VAL, np.minimum(rr_mean, cr_mean) / denom_rm, 0.0)

    profile_ringing_max = np.maximum(rr_max, cr_max)
    profile_ringing_mean = np.full((gh, gw), np.nan)
    for gy in range(gh):
        for gx in range(gw):
            vals = []
            has_row = not np.isnan(rr_max[gy, gx])
            has_col = not np.isnan(cr_max[gy, gx])
            if has_row:
                vals.extend(row_scores)
            if has_col:
                vals.extend(col_scores)
            # Actually need to recompute per-row/per-col values...
            # Simplified: use mean of row_ringing_mean and col_ringing_mean averages
            rrm, crm = rr_mean[gy, gx], cr_mean[gy, gx]
            if not np.isnan(rrm) and not np.isnan(crm):
                vals2 = [rrm, crm]
            elif not np.isnan(rrm):
                vals2 = [rrm]
            elif not np.isnan(crm):
                vals2 = [crm]
            else:
                vals2 = []
            if vals2:
                profile_ringing_mean[gy, gx] = np.mean(vals2)

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
        'row_ringing_max': rr_max, 'row_ringing_min': rr_min, 'row_ringing_mean': rr_mean,
        'row_ringing_dyn_score': row_data['dyn'], 'row_ringing_d2_score': row_data['d2'], 'row_ringing_sign_score': row_data['sign'],
        'col_ringing_max': cr_max, 'col_ringing_min': cr_min, 'col_ringing_mean': cr_mean,
        'col_ringing_dyn_score': col_data['dyn'], 'col_ringing_d2_score': col_data['d2'], 'col_ringing_sign_score': col_data['sign'],
    }

    feature_names = list(feature_maps.keys())
    feature_stack = np.stack([feature_maps[n] for n in feature_names], axis=-1)
    return feature_stack, feature_names


def normalize_features(feature_stack, feature_names):
    result = np.copy(feature_stack)
    for i, name in enumerate(feature_names):
        div = NORM_DIV.get(name, 1.0)
        if div == 1.0:
            result[:, :, i] = np.clip(result[:, :, i], 0, 1)
        else:
            result[:, :, i] = np.clip(result[:, :, i] / div, 0, 1)
    return result


def visualize_prediction(y_img, pred_mask, proba_map, output_path):
    H, W = y_img.shape
    y_norm = np.clip(y_img, 0, 255).astype(np.uint8)
    display = cv2.cvtColor(y_norm, cv2.COLOR_GRAY2BGR)

    gh, gw = pred_mask.shape
    grid_h, grid_w = H // gh, W // gw

    for gy in range(gh):
        for gx in range(gw):
            y1, y2 = gy * grid_h, (gy + 1) * grid_h
            x1, x2 = gx * grid_w, (gx + 1) * grid_w
            pred = pred_mask[gy, gx]

            if pred == 1:
                overlay = display[y1:y2, x1:x2].astype(np.float64)
                overlay[:, :, 2] = np.clip(overlay[:, :, 2] * 0.6 + 255 * 0.4, 0, 255)
                display[y1:y2, x1:x2] = overlay.astype(np.uint8)
            cv2.rectangle(display, (x1, y1), (x2, y2), (100, 100, 100), 1)

    total = gh * gw
    dm_count = int(np.sum(pred_mask))
    cv2.putText(display, f"DM: {dm_count}/{total} ({100*dm_count/total:.1f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imwrite(str(output_path), display)
    return display


def predict_single(args):
    """Worker function for multiprocessing: predict on one BMP."""
    bmp_path, out_dir, model_path = args
    out_path = Path(out_dir) / f"{bmp_path.stem}_prediction.png"
    try:
        model = lgb.Booster(model_file=model_path)
        y = load_y_from_bmp(bmp_path)
        if y is None:
            return {'file': bmp_path.name, 'status': 'ERROR: cannot read', 'dm_pct': -1}

        H, W = y.shape
        gh, gw = H // GRID_SIZE, W // GRID_SIZE
        feat_stack, feat_names = compute_features(y)
        feat_norm = normalize_features(feat_stack, feat_names)
        X = np.nan_to_num(feat_norm.reshape(gh * gw, -1), 0)
        proba = model.predict(X)
        pred = proba.round().clip(0, 1).astype(np.int32)
        pred_map = pred.reshape(gh, gw)
        proba_map = proba.reshape(gh, gw)

        visualize_prediction(y, pred_map, proba_map, out_path)

        dm_count = int(np.sum(pred))
        return {
            'file': bmp_path.name,
            'width': W, 'height': H,
            'blocks': gh * gw,
            'dm_count': dm_count,
            'dm_pct': round(100 * dm_count / (gh * gw), 1),
            'status': 'OK',
        }
    except Exception as e:
        return {'file': bmp_path.name, 'status': f'ERROR: {e}', 'dm_pct': -1}


def main():
    print(f"Found {len(list(TEST_DIR.glob('*.bmp')))} BMP files in {TEST_DIR}")
    print(f"Using {NUM_WORKERS} worker processes\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bmp_files = sorted(TEST_DIR.glob("*.bmp"))
    # Prepare worker args: (path, out_dir, model_path)
    worker_args = [(b, str(OUTPUT_DIR), MODEL_PATH) for b in bmp_files]

    total_start = time.time()
    results = []

    with Pool(NUM_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(predict_single, worker_args), 1):
            elapsed = time.time() - total_start
            dm = result.get('dm_pct', -1)
            if dm >= 0:
                print(f"[{i}/{len(bmp_files)}] {result['file']:<50} {dm:>5.1f}% DM")
            else:
                print(f"[{i}/{len(bmp_files)}] {result['file']:<50} {result.get('status', 'FAIL')}")
            results.append(result)

    # Summary
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Total: {len(results)}/{len(bmp_files)} images processed in {total_time:.0f}s")
    if results:
        dm_counts = [r['dm_count'] for r in results]
        totals = [r['blocks'] for r in results]
        print(f"Overall DM: {sum(dm_counts)}/{sum(totals)} ({100*sum(dm_counts)/sum(totals):.1f}%)")
        print(f"DM range: [{min(dm_counts)}/{min(totals)}, {max(dm_counts)}/{max(totals)}] blocks")

        # Print per-file table
        print(f"\n{'File':<50} {'DM%':<8} {'Status':<10}")
        print("-"*70)
        for r in results:
            status = "HIGH" if r['dm_pct'] > 20 else ("MED" if r['dm_pct'] > 10 else "LOW")
            print(f"{r['file']:<50} {r['dm_pct']:<8.1f} {status:<10}")

    print(f"\nPredictions saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
