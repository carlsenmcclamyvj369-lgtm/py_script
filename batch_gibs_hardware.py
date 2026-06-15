import cv2
import numpy as np
from pathlib import Path


def robust_mad_threshold(values, scale=1.0):
    values = np.asarray(values, dtype=np.float32)
    med = np.median(values)
    mad = np.median(np.abs(values - med)) + 1e-6
    return float(scale * 1.4826 * mad)


def numpy_vectorized_predict_with_decay(
        image_path, patch_size=3, stride=3, profile_radius=4, scale=1.0
):
    # ==========================================
    # 1. 基础图像计算
    # ==========================================
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None: raise ValueError("Cannot read image")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape

    edges = cv2.Canny(gray.astype(np.uint8), 100, 200)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    blur = cv2.GaussianBlur(gray, ksize=(0, 0), sigmaX=1.0)
    residual = gray - blur
    abs_residual = np.abs(residual)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge_band = cv2.dilate(edges, kernel, iterations=1) > 0
    edge_pixels = edges > 0
    edge_neighbor_f = (edge_band & (~edge_pixels)).astype(np.float32)
    non_edge_f = (~edge_band).astype(np.float32)

    # ==========================================
    # 2. 边缘局部性 (loc_ratio)
    # ==========================================
    ctx_k = patch_size + 2 * profile_radius
    sum_neighbor = cv2.boxFilter(abs_residual * edge_neighbor_f, -1, (ctx_k, ctx_k), normalize=False)
    count_neighbor = cv2.boxFilter(edge_neighbor_f, -1, (ctx_k, ctx_k), normalize=False)
    sum_non_edge = cv2.boxFilter(abs_residual * non_edge_f, -1, (ctx_k, ctx_k), normalize=False)
    count_non_edge = cv2.boxFilter(non_edge_f, -1, (ctx_k, ctx_k), normalize=False)

    mean_neighbor = sum_neighbor / np.maximum(count_neighbor, 1.0)
    mean_non_edge = sum_non_edge / np.maximum(count_non_edge, 1e-6)
    loc_ratio_full = mean_neighbor / mean_non_edge

    # ==========================================
    # 3. 终极性能：整数方向量化 + 矩阵平移获取 Profile
    # ==========================================
    ey, ex = np.where(edges > 0)
    abs_gx, abs_gy = np.abs(gx[ey, ex]), np.abs(gy[ey, ex])
    valid_mask = (abs_gx + abs_gy) > 1e-6
    ey, ex = ey[valid_mask], ex[valid_mask]
    abs_gx, abs_gy = abs_gx[valid_mask], abs_gy[valid_mask]

    N = len(ey)
    directions = np.zeros(N, dtype=np.int8)
    gx_val, gy_val = gx[ey, ex], gy[ey, ex]

    is_horiz = abs_gx > 2 * abs_gy
    is_vert  = abs_gy > 2 * abs_gx
    is_diag1 = (~is_horiz) & (~is_vert) & (np.sign(gx_val) == np.sign(gy_val))
    is_diag2 = (~is_horiz) & (~is_vert) & (np.sign(gx_val) != np.sign(gy_val))

    directions[is_horiz] = 0
    directions[is_vert]  = 1
    directions[is_diag1] = 2
    directions[is_diag2] = 3

    steps = [
        (0, 1),   # 0: 水平延伸
        (1, 0),   # 1: 垂直延伸
        (1, 1),   # 2: 主对角线延伸
        (-1, 1)   # 3: 副对角线延伸
    ]

    t = np.arange(-profile_radius, profile_radius + 1)
    L = len(t)

    offsets = np.array([[dy * t, dx * t] for dy, dx in steps])
    selected_offsets = offsets[directions]

    samp_y = ey[:, None] + selected_offsets[:, 0, :]
    samp_x = ex[:, None] + selected_offsets[:, 1, :]

    np.clip(samp_y, 0, H - 1, out=samp_y)
    np.clip(samp_x, 0, W - 1, out=samp_x)

    profiles = residual[samp_y, samp_x]
    abs_profiles = np.abs(profiles)

    # 提前算 gray profile 的阶跃高度
    profiles_gray = gray[samp_y, samp_x]
    prof_2d = profiles_gray.reshape(N, 1, L)
    smooth_2d = cv2.GaussianBlur(prof_2d, ksize=(0, 0), sigmaX=1.0)
    smooth_profiles = smooth_2d.reshape(N, L)
    edge_step = np.max(smooth_profiles, axis=1) - np.min(smooth_profiles, axis=1)

    # --- 特征 A：零交叉 (Crossings) ---
    low_th = (0.0005 * edge_step)[:, None]
    high_th = (0.2 * edge_step)[:, None]

    signs = np.zeros_like(profiles, dtype=np.int8)
    valid_a = (np.abs(profiles) > low_th) & (np.abs(profiles) < high_th)
    signs[valid_a & (profiles > 0)] = 1
    signs[valid_a & (profiles < 0)] = -1

    mask_signs = (signs != 0)
    idx = np.where(mask_signs, np.arange(signs.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    row_indices = np.arange(N)[:, None]
    filled_signs = signs[row_indices, idx]

    valid_crossings = (filled_signs[:, :-1] != filled_signs[:, 1:]) & (filled_signs[:, :-1] != 0) & (filled_signs[:, 1:] != 0)
    crossings = np.sum(valid_crossings, axis=1)

    # --- 特征 B：衰减 (Decay) ---
    d = np.abs(t)
    near_mask, mid_mask, far_mask = (d == 1), ((d >= 2) & (d <= 3)), (d >= 4)

    near_e = np.mean(abs_profiles[:, near_mask], axis=1) if np.any(near_mask) else np.zeros(N)
    mid_e = np.mean(abs_profiles[:, mid_mask], axis=1) if np.any(mid_mask) else np.zeros(N)
    far_e = np.mean(abs_profiles[:, far_mask], axis=1) if np.any(far_mask) else np.zeros(N)

    mono_score = 0.0 + 0.5 * (near_e > mid_e) + 0.5 * (mid_e > far_e)
    weighted_dist = np.sum(abs_profiles * d, axis=1) / (np.sum(abs_profiles, axis=1) + 1e-6)
    compactness = 1.0 / (1.0 + weighted_dist)
    ratio_score = np.tanh((near_e / (far_e + 1e-6)) / 3.0)

    decay_scores = 0.4 * mono_score + 0.3 * compactness + 0.3 * ratio_score

    # --- 特征 C: Gibbs-like profile score ---
    prof_residual = profiles_gray - smooth_profiles
    abs_prof_res = np.abs(prof_residual)
    residual_energy = np.mean(abs_prof_res, axis=1)

    osc_signs = np.zeros_like(prof_residual, dtype=np.int8)
    valid_osc = (abs_prof_res > low_th) & (abs_prof_res < high_th)
    osc_signs[valid_osc & (prof_residual > 0)] = 1
    osc_signs[valid_osc & (prof_residual < 0)] = -1

    mask_osc = osc_signs != 0
    idx_osc = np.where(mask_osc, np.arange(L), 0)
    np.maximum.accumulate(idx_osc, axis=1, out=idx_osc)
    filled_osc = osc_signs[row_indices, idx_osc]
    valid_adj = mask_osc[:, 1:]
    alt = (filled_osc[:, :-1] != filled_osc[:, 1:]) & valid_adj
    total_pairs = np.sum(valid_adj, axis=1).astype(np.float32)
    oscillation_score = np.divide(np.sum(alt, axis=1), total_pairs, out=np.zeros(N), where=total_pairs > 0)

    near_g = np.mean(abs_prof_res[:, near_mask], axis=1) if np.any(near_mask) else np.zeros(N)
    far_g = np.mean(abs_prof_res[:, far_mask], axis=1) if np.any(far_mask) else np.zeros(N)
    decay_from_prof = 0.0 + 0.5 * (near_e < mid_e) + 0.5 * (mid_e < far_e)

    gibbs_scores = residual_energy * (oscillation_score + decay_from_prof)

    # ==========================================
    # 4. 聚合与判决
    # ==========================================
    num_rows = (H - patch_size) // stride + 1
    num_cols = (W - patch_size) // stride + 1

    grid_y_centers = np.arange(num_rows) * stride + patch_size // 2
    grid_x_centers = np.arange(num_cols) * stride + patch_size // 2
    grid_X, grid_Y = np.meshgrid(grid_x_centers, grid_y_centers)
    grid_loc_ratio = loc_ratio_full[grid_Y, grid_X]

    patch_idx_y = np.clip(ey // stride, 0, num_rows - 1)
    patch_idx_x = np.clip(ex // stride, 0, num_cols - 1)

    grid_crossings_sum = np.zeros((num_rows, num_cols), dtype=np.float32)
    grid_decay_sum = np.zeros((num_rows, num_cols), dtype=np.float32)
    grid_valid_count = np.zeros((num_rows, num_cols), dtype=np.float32)

    np.add.at(grid_crossings_sum, (patch_idx_y, patch_idx_x), crossings)
    np.add.at(grid_decay_sum, (patch_idx_y, patch_idx_x), decay_from_prof)
    np.add.at(grid_valid_count, (patch_idx_y, patch_idx_x), 1)

    grid_mean_crossings = grid_crossings_sum / np.maximum(grid_valid_count, 1.0)
    grid_mean_decay = grid_decay_sum / np.maximum(grid_valid_count, 1.0)

    # p90 gibbs per patch
    order = np.lexsort((patch_idx_x, patch_idx_y))
    sorted_gibbs = gibbs_scores[order]
    sorted_py = patch_idx_y[order]
    sorted_px = patch_idx_x[order]
    change = (np.diff(sorted_py) != 0) | (np.diff(sorted_px) != 0)
    bounds = np.where(change)[0] + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(sorted_gibbs)]])

    grid_gibbs_p70 = np.zeros((num_rows, num_cols), dtype=np.float32)
    for s, e in zip(starts, ends):
        py, px = sorted_py[s], sorted_px[s]
        grid_gibbs_p70[py, px] = np.percentile(sorted_gibbs[s:e], 70)

    score_grid = np.zeros((num_rows, num_cols), dtype=np.float32)
    mask = (grid_loc_ratio > 1.5) & (grid_mean_crossings > 0) & (grid_mean_decay > 0)
    score_grid[mask] = grid_gibbs_p70[mask]

    return score_grid, img, grid_gibbs_p70


def save_heatmap(img, score_grid, output_path):
    H, W = img.shape[:2]
    score = score_grid.astype(np.float32)
    score_u8 = np.clip(score * 10, 0, 255).astype(np.uint8)
    heatmap_small = cv2.applyColorMap(score_u8, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap_small, (W, H), interpolation=cv2.INTER_NEAREST)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(output_path, overlay)


def main():
    data_dir = Path(__file__).resolve().parent / "test_data"
    output_dir = Path(__file__).resolve().parent / "test_data_output_gibs"

    bmp_files = sorted(data_dir.glob("*.bmp"))
    if not bmp_files:
        print(f"未在 {data_dir} 下找到 .bmp 文件")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for bmp_path in bmp_files:
        stem = bmp_path.stem
        try:
            score_grid, img, grid_gibbs = numpy_vectorized_predict_with_decay(str(bmp_path))

            heatmap_path = output_dir / f"{stem}.png"
            save_heatmap(img, score_grid, str(heatmap_path))

            npy_path = output_dir / f"{stem}_score.npy"
            np.save(str(npy_path), score_grid)

            print(f"✓ {stem}  shape={score_grid.shape}")
        except Exception as e:
            print(f"✗ {stem}  错误: {e}")

    print(f"\n完成！共处理 {len(bmp_files)} 张图片，结果保存至 {output_dir}")


if __name__ == "__main__":
    main()
