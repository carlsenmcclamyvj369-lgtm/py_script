"""
Compute features for labeled blocks from 05.02.25 BMP using feature_compute_ref.py.
"""

import numpy as np
import pandas as pd
import cv2
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import feature_compute_ref as fcr

# ─── Config ───────────────────────────────────
BMP_PATH = r"C:\code\py\denoise\scripts\test_data\05.02.25#out1#mnr_input0012.bmp"
DM_CSV = r"C:\code\py\denoise\scripts\mos_featue_analyze\excel\05.02.25#out1#mnr_input0012\dm.csv"
NOT_DM_CSV = r"C:\code\py\denoise\scripts\mos_featue_analyze\excel\05.02.25#out1#mnr_input0012\not_dm.csv"
OUT_CSV = r"C:\code\py\denoise\scripts\mos_featue_analyze\05.02.25_labeled_features_raw.csv"

GS = 8  # grid size

NORM_DIV = {
    'mean_var': 1020, 'max_var': 1020, 'top5_var': 1020,
    'low_var_count': 64, 'high_var_count': 64, 'very_high_var_count': 64,
    'residual_mean': 255, 'residual_max': 255,
    'lap_mean': 1020, 'lap_max': 1020,
    'grad_mean': 1020, 'grad_max': 1020,
    'edge_strength': 255, 'h_strength': 255, 'h_strength_max': 255, 'h_strength_min': 255,
    'v_strength': 255, 'v_strength_max': 255, 'v_strength_min': 255,
    'edge_orientation_conf': 1.0,
    'row_second_diff': 510, 'row_second_diff_max': 510, 'row_second_diff_min': 510,
    'col_second_diff': 510, 'col_second_diff_max': 510, 'col_second_diff_min': 510,
    'second_diff_max': 510, 'second_diff_min_max': 1.0,
    'profile_ringing_max': 1.0, 'profile_ringing_mean': 1.0,
    'ringing_mean_max': 1.0, 'ringing_mean_min': 1.0, 'ringing_mean_min_max': 1.0,
    'row_ringing_max': 1.0, 'row_ringing_min': 1.0, 'row_ringing_mean': 1.0,
    'row_ringing_dyn_score': 1.0, 'row_ringing_d2_score': 1.0, 'row_ringing_sign_score': 1.0,
    'col_ringing_max': 1.0, 'col_ringing_min': 1.0, 'col_ringing_mean': 1.0,
    'col_ringing_dyn_score': 1.0, 'col_ringing_d2_score': 1.0, 'col_ringing_sign_score': 1.0,
}

FEATURE_NAMES = [
    'mean_var', 'max_var', 'top5_var',
    'low_var_count', 'high_var_count', 'very_high_var_count',
    'residual_mean', 'residual_max', 'lap_mean', 'lap_max',
    'grad_mean', 'grad_max',
    'edge_strength', 'h_strength', 'h_strength_max', 'h_strength_min',
    'v_strength', 'v_strength_max', 'v_strength_min', 'edge_orientation_conf',
    'row_second_diff', 'row_second_diff_max', 'row_second_diff_min',
    'col_second_diff', 'col_second_diff_max', 'col_second_diff_min',
    'second_diff_max', 'second_diff_min_max',
    'profile_ringing_max', 'profile_ringing_mean',
    'ringing_mean_max', 'ringing_mean_min', 'ringing_mean_min_max',
    'row_ringing_max', 'row_ringing_min', 'row_ringing_mean',
    'row_ringing_dyn_score', 'row_ringing_d2_score', 'row_ringing_sign_score',
    'col_ringing_max', 'col_ringing_min', 'col_ringing_mean',
    'col_ringing_dyn_score', 'col_ringing_d2_score', 'col_ringing_sign_score',
    'row_diff_mean', 'row_diff_max', 'col_diff_mean', 'col_diff_max',
]


def get_block(map2d, bi, bj):
    """Extract 8x8 block at grid index (bi, bj), return flat valid values."""
    y1, y2 = bi * GS, min(bi * GS + GS, map2d.shape[0])
    x1, x2 = bj * GS, min(bj * GS + GS, map2d.shape[1])
    vals = map2d[y1:y2, x1:x2].flatten()
    return vals[~np.isnan(vals)]


# Parameter columns that should be passed through from input CSV
PARAM_COLS = [
    'low_var_th', 'high_var_th', 'very_high_var_th', 'max_strength_th', 'eps',
    'dyn_score_lo', 'dyn_score_hi', 'd2_score_lo', 'd2_score_hi',
    'sign_score_lo', 'sign_score_hi', 'dyn_ratio', 'd2_ratio', 'sign_ratio',
]


def extract_features_for_block(y_full, bi, bj, precomputed):
    """Compute all 45 features for one grid block (bi, bj)."""
    var_map, res_map, lap_map, grad_map, h_edge, v_edge = precomputed
    y1, y2 = bi * GS, min(bi * GS + GS, y_full.shape[0])
    x1, x2 = bj * GS, min(bj * GS + GS, y_full.shape[1])
    block_y = y_full[y1:y2, x1:x2]
    bh, bw = block_y.shape

    feats = {}

    # Variance
    bv = get_block(var_map, bi, bj)
    if len(bv) > 0:
        sv = np.sort(bv)
        feats['mean_var'] = float(np.mean(bv))
        feats['max_var'] = float(np.max(bv))
        feats['top5_var'] = float(np.mean(sv[-5:]) if len(sv) >= 5 else np.mean(sv))
        feats['low_var_count'] = float(np.sum(bv < 100))
        feats['high_var_count'] = float(np.sum(bv > 500))
        feats['very_high_var_count'] = float(np.sum(bv > 2000))

    # Residual
    br = get_block(res_map, bi, bj)
    if len(br) > 0:
        feats['residual_mean'] = float(np.mean(np.abs(br)))
        feats['residual_max'] = float(np.max(np.abs(br)))

    # Laplacian
    bl = get_block(lap_map, bi, bj)
    if len(bl) > 0:
        feats['lap_mean'] = float(np.mean(bl))
        feats['lap_max'] = float(np.max(bl))

    # Gradient
    bg = get_block(grad_map, bi, bj)
    if len(bg) > 0:
        feats['grad_mean'] = float(np.mean(bg))
        feats['grad_max'] = float(np.max(bg))

    # Edge
    bh_e = get_block(h_edge, bi, bj)
    bv_e = get_block(v_edge, bi, bj)
    if len(bh_e) > 0 and len(bv_e) > 0:
        hs = float(np.mean(bh_e))
        vs = float(np.mean(bv_e))
        ms = max(hs, vs)
        feats['edge_strength'] = ms
        feats['h_strength'] = hs
        feats['h_strength_max'] = float(np.max(bh_e))
        feats['h_strength_min'] = float(np.min(bh_e))
        feats['v_strength'] = vs
        feats['v_strength_max'] = float(np.max(bv_e))
        feats['v_strength_min'] = float(np.min(bv_e))
        feats['edge_orientation_conf'] = abs(hs - vs) / ms if ms > 1e-6 else 0.0

    # Second diff (block-level, uses Y directly)
    row_energies = []
    for ry in range(bh):
        v = block_y[ry, :]
        if len(v) >= 3:
            d2 = v[:-2] - 2 * v[1:-1] + v[2:]
            row_energies.append(float(np.mean(np.abs(d2))))
    col_energies = []
    for rx in range(bw):
        v = block_y[:, rx]
        if len(v) >= 3:
            d2 = v[:-2] - 2 * v[1:-1] + v[2:]
            col_energies.append(float(np.mean(np.abs(d2))))

    if row_energies:
        feats['row_second_diff'] = float(np.mean(row_energies))
        feats['row_second_diff_max'] = float(np.max(row_energies))
        feats['row_second_diff_min'] = float(np.min(row_energies))
    if col_energies:
        feats['col_second_diff'] = float(np.mean(col_energies))
        feats['col_second_diff_max'] = float(np.max(col_energies))
        feats['col_second_diff_min'] = float(np.min(col_energies))

    rd = feats.get('row_second_diff', 0)
    cd = feats.get('col_second_diff', 0)
    feats['second_diff_max'] = max(rd, cd)
    feats['second_diff_min_max'] = (min(rd, cd) / max(rd, cd)) if max(rd, cd) > 0 else 0.0

    # Row/col mean diff
    row_means = [float(block_y[r, :].mean()) for r in range(bh)]
    row_diffs = [abs(row_means[r+1] - row_means[r]) for r in range(bh - 1)] if bh >= 2 else [0.0]
    col_means = [float(block_y[:, c].mean()) for c in range(bw)]
    col_diffs = [abs(col_means[c+1] - col_means[c]) for c in range(bw - 1)] if bw >= 2 else [0.0]
    feats['row_diff_mean'] = float(np.mean(row_diffs))
    feats['row_diff_max'] = float(np.max(row_diffs))
    feats['col_diff_mean'] = float(np.mean(col_diffs))
    feats['col_diff_max'] = float(np.max(col_diffs))

    # Ringing
    ringing = fcr.ringing_stats_for_block(block_y, bh, bw)
    if ringing:
        for k in ['row_ringing_max', 'row_ringing_min', 'row_ringing_mean',
                   'ringing_mean_max', 'ringing_mean_min', 'ringing_mean_min_max',
                   'profile_ringing_max', 'profile_ringing_mean',
                   'row_ringing_dyn_score', 'row_ringing_d2_score', 'row_ringing_sign_score',
                   'col_ringing_max', 'col_ringing_min', 'col_ringing_mean',
                   'col_ringing_dyn_score', 'col_ringing_d2_score', 'col_ringing_sign_score']:
            feats[k] = ringing[k]

    return feats


def main():
    print("Loading BMP...")
    bgr = cv2.imread(BMP_PATH, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"ERROR: cannot read {BMP_PATH}")
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    y_full = fcr.compute_y_from_rgb(rgb)
    H, W = y_full.shape
    print(f"  Image: {W}x{H}")

    # Precompute all pixel-level maps once
    print("Precomputing pixel-level maps...")
    var_map = fcr.compute_var_map(y_full)
    k3 = np.ones((3, 3), dtype=np.float64) / 9.0
    mean3 = cv2.filter2D(y_full, -1, k3, borderType=cv2.BORDER_REFLECT)
    res_map = fcr.compute_residual_map(y_full, mean3)
    lap_map = fcr.compute_lap_map(y_full)
    grad_map = fcr.compute_grad_map(y_full)
    h_edge, v_edge = fcr.compute_edge_maps(y_full)
    precomputed = (var_map, res_map, lap_map, grad_map, h_edge, v_edge)

    # Load labels
    dm_df = pd.read_csv(DM_CSV)
    not_dm_df = pd.read_csv(NOT_DM_CSV)
    dm_df['label'] = 1
    not_dm_df['label'] = 0
    all_labels = pd.concat([dm_df, not_dm_df], ignore_index=True)
    # Deduplicate by (row, col) - keep first occurrence
    all_labels = all_labels.drop_duplicates(subset=['row', 'col'])
    print(f"  DM: {len(dm_df)}, Not DM: {len(not_dm_df)}, Unique total: {len(all_labels)}")

    # Process each labeled coordinate
    print("Computing features per block...")
    rows = []
    errors = 0
    for idx, row in all_labels.iterrows():
        r, c = int(row['row']), int(row['col'])
        label = int(row['label'])
        bi, bj = r, c  # row/col are grid indices, not pixel coords
        try:
            feats = extract_features_for_block(y_full, bi, bj, precomputed)
            out = {'name': row.get('name', ''), 'row': r, 'col': c, 'label': label}
            for fn in FEATURE_NAMES:
                out[fn] = feats.get(fn, np.nan)
            # Copy parameter columns from input
            for pcol in PARAM_COLS:
                if pcol in row:
                    out[pcol] = row[pcol]
            rows.append(out)
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  Error at ({r},{c}): {e}")

        if (idx + 1) % 200 == 0:
            print(f"  {idx+1}/{len(all_labels)}")

    result = pd.DataFrame(rows)
    print(f"\nDone: {len(result)} blocks ({errors} errors)")

    # Save
    result.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")

    dm_n = (result['label'] == 1).sum()
    ndm_n = (result['label'] == 0).sum()
    print(f"DM: {dm_n}, Not DM: {ndm_n}")

    nan_feats = result[FEATURE_NAMES].isna().sum()
    nan_feats = nan_feats[nan_feats > 0]
    if len(nan_feats) > 0:
        print(f"\nFeatures with NaN:")
        for fn, cnt in nan_feats.items():
            print(f"  {fn}: {cnt} NaN")


if __name__ == "__main__":
    main()
