"""
Generate a single CSV with 9x9 neighborhood features for all labeled blocks.
Precomputes all block features for the entire image grid, then assembles
neighborhoods by indexing. Much faster than per-block recomputation.
"""

import numpy as np
import pandas as pd
import cv2
import os
import sys
import time
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import feature_compute_ref as fcr

EXCEL_DIR = r"C:\code\py\denoise\scripts\mos_featue_analyze\excel"
TEST_DIR = r"C:\code\py\denoise\scripts\test_data"
OUT_CSV = r"C:\code\py\denoise\scripts\mos_featue_analyze\9x9_all_data.csv"
GS = 8
NUM_WORKERS = max(1, cpu_count() - 1)

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
OUT_COLS = ['name', 'row', 'col', 'label'] + [f'{fn}_{i}' for i in range(81) for fn in FEATURE_NAMES] + PARAM_COLS

TOTAL_FEATS = len(FEATURE_NAMES)


def get_block(map2d, bi, bj):
    y1, y2 = bi * GS, min(bi * GS + GS, map2d.shape[0])
    x1, x2 = bj * GS, min(bj * GS + GS, map2d.shape[1])
    vals = map2d[y1:y2, x1:x2].flatten()
    return vals[~np.isnan(vals)]


def compute_block_feats(bi, bj, pm, y_full):
    """Compute 49 features for one grid block. Returns array or zeros if invalid."""
    H, W = y_full.shape
    y1, y2 = bi * GS, min(bi * GS + GS, H)
    x1, x2 = bj * GS, min(bj * GS + GS, W)
    if y1 >= H or x1 >= W or y2 - y1 < 2 or x2 - x1 < 2 or bi < 0 or bj < 0:
        return np.zeros(TOTAL_FEATS, dtype=np.float64)

    block = y_full[y1:y2, x1:x2]
    bh, bw = block.shape
    var_map, res_map, lap_map, grad_map, h_edge, v_edge = pm
    feats = np.zeros(TOTAL_FEATS, dtype=np.float64)

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
        feats[12] = ms; feats[13] = hs; feats[14] = float(np.max(bh_e))
        feats[15] = float(np.min(bh_e)); feats[16] = vs
        feats[17] = float(np.max(bv_e)); feats[18] = float(np.min(bv_e))
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
    row_diffs = [abs(row_means[r+1] - row_means[r]) for r in range(bh - 1)] if bh >= 2 else [0.0]
    col_means = [float(block[:, c].mean()) for c in range(bw)]
    col_diffs = [abs(col_means[c+1] - col_means[c]) for c in range(bw - 1)] if bw >= 2 else [0.0]
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
    return feats


def process_one_folder(folder_name):
    """Load image, precompute ALL grid features, then assemble neighborhoods."""
    folder_path = os.path.join(EXCEL_DIR, folder_name)
    bmp_path = os.path.join(TEST_DIR, folder_name + '.bmp')
    if not os.path.exists(bmp_path):
        return None

    src_csvs = {}
    for fn in ['dm.csv', 'not_dm.csv']:
        fp = os.path.join(folder_path, fn)
        if os.path.exists(fp):
            src_csvs[fn] = pd.read_csv(fp)
    if not src_csvs:
        return None

    bgr = cv2.imread(bmp_path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    y_full = fcr.compute_y_from_rgb(rgb)
    H, W = y_full.shape
    gh, gw = H // GS, W // GS

    # Precompute pixel-level maps
    var_map = fcr.compute_var_map(y_full)
    k3 = np.ones((3, 3), dtype=np.float64) / 9.0
    mean3 = cv2.filter2D(y_full, -1, k3, borderType=cv2.BORDER_REFLECT)
    res_map = fcr.compute_residual_map(y_full, mean3)
    lap_map = fcr.compute_lap_map(y_full)
    grad_map = fcr.compute_grad_map(y_full)
    h_edge, v_edge = fcr.compute_edge_maps(y_full)
    pm = (var_map, res_map, lap_map, grad_map, h_edge, v_edge)

    # Precompute ALL block features for the entire grid
    grid_feats = np.zeros((gh, gw, TOTAL_FEATS), dtype=np.float64)
    for bi in range(gh):
        for bj in range(gw):
            grid_feats[bi, bj] = compute_block_feats(bi, bj, pm, y_full)

    # 9x9 offsets
    offsets = [(dr, dc) for dr in range(-4, 5) for dc in range(-4, 5)]

    # Assemble neighborhoods for each labeled center
    rows_list = []
    for csv_name, src_df in src_csvs.items():
        label = 1 if ('dm' in csv_name and 'not' not in csv_name) else 0
        centers = src_df[['row', 'col']].drop_duplicates()

        for _, cs in centers.iterrows():
            cr, cc = int(cs['row']), int(cs['col'])
            row_dict = {'name': folder_name, 'row': cr, 'col': cc, 'label': label}

            # Look up 81 neighborhood features from precomputed grid
            for i, (dr, dc) in enumerate(offsets):
                gr, gc = cr + dr, cc + dc
                if 0 <= gr < gh and 0 <= gc < gw:
                    fv = grid_feats[gr, gc]
                else:
                    fv = np.zeros(TOTAL_FEATS, dtype=np.float64)
                for j, fn in enumerate(FEATURE_NAMES):
                    row_dict[f'{fn}_{i}'] = float(fv[j])

            # Param cols
            for pc in PARAM_COLS:
                if pc in src_df.columns:
                    match = src_df[(src_df['row']==cr) & (src_df['col']==cc)]
                    if len(match) > 0:
                        row_dict[pc] = match.iloc[0][pc]

            rows_list.append(row_dict)

    if rows_list:
        return pd.DataFrame(rows_list)
    return None


def main():
    folders = sorted([f for f in os.listdir(EXCEL_DIR)
                      if os.path.isdir(os.path.join(EXCEL_DIR, f))])
    print(f"Processing {len(folders)} folders...", flush=True)
    t0 = time.time()

    ok = 0
    results = []
    for fname in folders:
        start = time.time()
        result = process_one_folder(fname)
        t = time.time() - start
        if result is not None:
            results.append(result)
            ok += 1
            print(f"  {fname:<55} {len(result):>5} centers [{t:.0f}s]", flush=True)
        else:
            print(f"  {fname:<55} SKIP [{t:.0f}s]", flush=True)

    if not results:
        print("No data!")
        return

    all_data = pd.concat(results, ignore_index=True)
    all_data = all_data[OUT_COLS]
    all_data.to_csv(OUT_CSV, index=False)

    dm_n = (all_data['label'] == 1).sum()
    ndm_n = (all_data['label'] == 0).sum()
    print(f"\nSaved: {OUT_CSV}")
    print(f"Centers: {len(all_data)} ({dm_n} dm, {ndm_n} not_dm)")
    print(f"Total blocks computed (neighborhoods): ~{len(all_data) * 81}")
    print(f"Time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
