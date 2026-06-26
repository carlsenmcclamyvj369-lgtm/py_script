"""
Generate 3x3 and 5x5 CSV files for all excel subfolders.
Format: LONG format - each row is one neighborhood position.
For each center block: 9 rows (3x3) or 25 rows (5x5), all with center's (row,col).
"""

import numpy as np
import pandas as pd
import cv2
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import feature_compute_ref as fcr

EXCEL_DIR = r"C:\code\py\denoise\scripts\mos_featue_analyze\excel"
TEST_DIR = r"C:\code\py\denoise\scripts\test_data"
GS = 8

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
PARAM_COLS = [
    'low_var_th', 'high_var_th', 'very_high_var_th', 'max_strength_th', 'eps',
    'dyn_score_lo', 'dyn_score_hi', 'd2_score_lo', 'd2_score_hi',
    'sign_score_lo', 'sign_score_hi', 'dyn_ratio', 'd2_ratio', 'sign_ratio',
]
OUT_COLS = ['name', 'row', 'col'] + FEATURE_NAMES + PARAM_COLS


def get_block(map2d, bi, bj):
    y1, y2 = bi * GS, min(bi * GS + GS, map2d.shape[0])
    x1, x2 = bj * GS, min(bj * GS + GS, map2d.shape[1])
    vals = map2d[y1:y2, x1:x2].flatten()
    return vals[~np.isnan(vals)]


def compute_block_features(bi, bj, precomputed, y_full):
    """Compute all features for one grid block (bi, bj)."""
    var_map, res_map, lap_map, grad_map, h_edge, v_edge = precomputed
    y1, y2 = bi * GS, min(bi * GS + GS, y_full.shape[0])
    x1, x2 = bj * GS, min(bj * GS + GS, y_full.shape[1])
    block = y_full[y1:y2, x1:x2]
    bh, bw = block.shape
    feats = {}

    bv = get_block(var_map, bi, bj)
    if len(bv) > 0:
        sv = np.sort(bv)
        feats['mean_var'] = float(np.mean(bv))
        feats['max_var'] = float(np.max(bv))
        feats['top5_var'] = float(np.mean(sv[-5:]) if len(sv) >= 5 else np.mean(sv))
        feats['low_var_count'] = float(np.sum(bv < 100))
        feats['high_var_count'] = float(np.sum(bv > 500))
        feats['very_high_var_count'] = float(np.sum(bv > 2000))

    br = get_block(res_map, bi, bj)
    if len(br) > 0:
        feats['residual_mean'] = float(np.mean(np.abs(br)))
        feats['residual_max'] = float(np.max(np.abs(br)))

    bl = get_block(lap_map, bi, bj)
    if len(bl) > 0:
        feats['lap_mean'] = float(np.mean(bl))
        feats['lap_max'] = float(np.max(bl))

    bg = get_block(grad_map, bi, bj)
    if len(bg) > 0:
        feats['grad_mean'] = float(np.mean(bg))
        feats['grad_max'] = float(np.max(bg))

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

    row_means = [float(block[r, :].mean()) for r in range(bh)]
    row_diffs = [abs(row_means[r+1] - row_means[r]) for r in range(bh - 1)] if bh >= 2 else [0.0]
    col_means = [float(block[:, c].mean()) for c in range(bw)]
    col_diffs = [abs(col_means[c+1] - col_means[c]) for c in range(bw - 1)] if bw >= 2 else [0.0]
    feats['row_diff_mean'] = float(np.mean(row_diffs))
    feats['row_diff_max'] = float(np.max(row_diffs))
    feats['col_diff_mean'] = float(np.mean(col_diffs))
    feats['col_diff_max'] = float(np.max(col_diffs))

    ringing = fcr.ringing_stats_for_block(block, bh, bw)
    if ringing:
        for k in ['row_ringing_max', 'row_ringing_min', 'row_ringing_mean',
                   'ringing_mean_max', 'ringing_mean_min', 'ringing_mean_min_max',
                   'profile_ringing_max', 'profile_ringing_mean',
                   'row_ringing_dyn_score', 'row_ringing_d2_score', 'row_ringing_sign_score',
                   'col_ringing_max', 'col_ringing_min', 'col_ringing_mean',
                   'col_ringing_dyn_score', 'col_ringing_d2_score', 'col_ringing_sign_score']:
            feats[k] = ringing[k]
    return feats


def process_folder(folder_name):
    """Generate 3x3 and 5x5 CSVs for one folder. Returns (name, status)."""
    folder_path = os.path.join(EXCEL_DIR, folder_name)
    bmp_path = os.path.join(TEST_DIR, folder_name + '.bmp')

    if not os.path.exists(bmp_path):
        return folder_name, "no BMP"

    src_csvs = {}
    for fn in ['dm.csv', 'not_dm.csv']:
        fp = os.path.join(folder_path, fn)
        if os.path.exists(fp):
            src_csvs[fn] = pd.read_csv(fp)

    if not src_csvs:
        return folder_name, "no source CSVs"

    # Load BMP
    bgr = cv2.imread(bmp_path, cv2.IMREAD_COLOR)
    if bgr is None:
        return folder_name, "BMP fail"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    y_full = fcr.compute_y_from_rgb(rgb)

    # Precompute maps
    var_map = fcr.compute_var_map(y_full)
    k3 = np.ones((3, 3), dtype=np.float64) / 9.0
    mean3 = cv2.filter2D(y_full, -1, k3, borderType=cv2.BORDER_REFLECT)
    res_map = fcr.compute_residual_map(y_full, mean3)
    lap_map = fcr.compute_lap_map(y_full)
    grad_map = fcr.compute_grad_map(y_full)
    h_edge, v_edge = fcr.compute_edge_maps(y_full)
    precomputed = (var_map, res_map, lap_map, grad_map, h_edge, v_edge)

    # Offsets: 3x3 (row-major) and 5x5 (row-major)
    configs = [
        ('3x3', [(dr, dc) for dr in range(-1, 2) for dc in range(-1, 2)]),
        ('5x5', [(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3)]),
    ]

    for csv_name, src_df in src_csvs.items():
        label = 1 if 'dm' in csv_name else 0
        label_tag = csv_name.replace('.csv', '')  # dm or not_dm

        for size_tag, offsets in configs:
            out_rows = []
            # Get unique center coordinates
            centers = src_df[['row', 'col']].drop_duplicates()

            for _, center_row_src in centers.iterrows():
                cr, cc = int(center_row_src['row']), int(center_row_src['col'])

                for dr, dc in offsets:
                    gr, gc = cr + dr, cc + dc
                    feats = compute_block_features(gr, gc, precomputed, y_full)

                    row_dict = {'name': folder_name, 'row': cr, 'col': cc}
                    for fn in FEATURE_NAMES:
                        row_dict[fn] = feats.get(fn, 0.0)
                    # Copy param columns from a matching source row
                    for pc in PARAM_COLS:
                        if pc in src_df.columns:
                            match = src_df[(src_df['row']==cr) & (src_df['col']==cc)]
                            if len(match) > 0:
                                row_dict[pc] = match.iloc[0][pc]
                    out_rows.append(row_dict)

            if not out_rows:
                continue

            out_df = pd.DataFrame(out_rows, columns=OUT_COLS)
            out_path = os.path.join(folder_path, f'{size_tag}_{label_tag}.csv')
            out_df.to_csv(out_path, index=False)

    return folder_name, "OK"


def main():
    folders = sorted([f for f in os.listdir(EXCEL_DIR)
                      if os.path.isdir(os.path.join(EXCEL_DIR, f))])

    ok = 0
    skip = 0
    for fname in folders:
        start = time.time()
        name, status = process_folder(fname)
        t = time.time() - start
        print(f"[{t:3.0f}s] {name:<55} {status}")
        if status == "OK":
            ok += 1
        else:
            skip += 1

    print(f"\nDone: {ok} OK, {skip} skipped")


if __name__ == "__main__":
    main()
