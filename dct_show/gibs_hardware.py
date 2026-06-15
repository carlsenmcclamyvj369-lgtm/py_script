import cv2
import numpy as np
from scipy.ndimage import map_coordinates


def robust_mad_threshold(values, scale=1.0):
    """用 MAD 估计鲁棒阈值。"""
    values = np.asarray(values, dtype=np.float32)
    med = np.median(values)
    mad = np.median(np.abs(values - med)) + 1e-6
    threshold = scale * 1.4826 * mad
    return float(threshold)


def numpy_vectorized_predict_with_decay(
        image_path, patch_size=3, stride=3, profile_radius=4, scale=1.0
):
    # ==========================================
    # 1. 基础图像计算 (同前)
    # ==========================================
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None: raise ValueError("Cannot read image")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape

    edges = cv2.Canny(gray.astype(np.uint8), 100, 200)
    # cv2.imshow("edges", edges)
    # cv2.waitKey(0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    # cv2.imshow("gx", gx)
    # cv2.waitKey(0)
    # cv2.imshow("gy", abs(gy/20))
    # cv2.waitKey(0)

    blur = cv2.GaussianBlur(gray, ksize=(0, 0), sigmaX=1.0)
    residual = gray - blur
    abs_residual = np.abs(residual)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge_band = cv2.dilate(edges, kernel, iterations=1) > 0
    edge_pixels = edges > 0
    edge_neighbor_f = (edge_band & (~edge_pixels)).astype(np.float32)
    non_edge_f = (~edge_band).astype(np.float32)

    # ==========================================
    # 2. 边缘局部性 (loc_ratio) (同前)
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
    # 找出边缘点并确保梯度不是 0
    ey, ex = np.where(edges > 0)
    abs_gx, abs_gy = np.abs(gx[ey, ex]), np.abs(gy[ey, ex])
    valid_mask = (abs_gx + abs_gy) > 1e-6
    ey, ex = ey[valid_mask], ex[valid_mask]
    abs_gx, abs_gy = abs_gx[valid_mask], abs_gy[valid_mask]

    N = len(ey)
    # 量化为 4 条通道 (0: 水平, 1: 垂直, 2: 主对角线, 3: 副对角线)
    directions = np.zeros(N, dtype=np.int8)
    gx_val, gy_val = gx[ey, ex], gy[ey, ex]

    # 根据梯度绝对值大小分配方向 (方向与梯度同向，即垂直于边缘)
    # 为了简化计算，我们统一规定 dx, dy。正负无所谓，因为是向两侧双向延伸。
    is_horiz = abs_gx > 2 * abs_gy
    is_vert = abs_gy > 2 * abs_gx
    is_diag1 = (~is_horiz) & (~is_vert) & (np.sign(gx_val) == np.sign(gy_val)) # 梯度方向：左上到右下 -> 法线也是
    is_diag2 = (~is_horiz) & (~is_vert) & (np.sign(gx_val) != np.sign(gy_val)) # 梯度方向：左下到右上 -> 法线也是

    directions[is_horiz] = 0
    directions[is_vert] = 1
    directions[is_diag1] = 2
    directions[is_diag2] = 3

    # 定义 4 条通道的步长向量 (dy, dx)
    steps = [
        (0, 1),   # 0: 水平延伸
        (1, 0),   # 1: 垂直延伸
        (1, 1),   # 2: 主对角线延伸
        (-1, 1)   # 3: 副对角线延伸
    ]

    # 瞬间提取几万条法线 Profile (N, 2*radius+1)
    # 我们用高级索引结合整数加法，0 浮点运算
    t = np.arange(-profile_radius, profile_radius + 1) # e.g. [-5, ..., 5]
    L = len(t)

    # 为每种方向预先算好坐标偏移，形状 (L, 2)
    offsets = np.array([[dy * t, dx * t] for dy, dx in steps]) # shape: (4, 2, 11)

    # 利用 N 个点各自的方向，取出它们应得的偏移量
    selected_offsets = offsets[directions] # shape: (N, 2, 11)

    # 加上中心点坐标，得到绝对整数坐标
    samp_y = ey[:, None] + selected_offsets[:, 0, :] # shape: (N, 11)
    samp_x = ex[:, None] + selected_offsets[:, 1, :]

    # 越界保护 (夹逼到图像边缘)
    np.clip(samp_y, 0, H - 1, out=samp_y)
    np.clip(samp_x, 0, W - 1, out=samp_x)

    # 【绝对极限极速】整数索引直接扣取像素，零插值开销！
    profiles = residual[samp_y, samp_x]
    abs_profiles = np.abs(profiles)

    # 提前算 gray profile 的阶跃高度，供两个特征使用
    profiles_gray = gray[samp_y, samp_x]
    prof_2d = profiles_gray.reshape(N, 1, L)
    smooth_2d = cv2.GaussianBlur(prof_2d, ksize=(0, 0), sigmaX=1.0)
    smooth_profiles = smooth_2d.reshape(N, L)
    edge_step = np.max(smooth_profiles, axis=1) - np.min(smooth_profiles, axis=1)  # (N,)

    # --- 特征 A：极速计算零交叉 (Crossings) ---
    # Gibbs 振铃幅值在阶跃的 2%~20% 之间
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
    crossings = np.sum(valid_crossings, axis=1) # shape: (N,)

    # --- 特征 B：纯向量化计算衰减 (Decay) ---
    d = np.abs(t)
    near_mask, mid_mask, far_mask = (d == 1), ((d >= 2) & (d <= 3)), (d >= 4)

    near_e = np.mean(abs_profiles[:, near_mask], axis=1) if np.any(near_mask) else np.zeros(N)
    mid_e = np.mean(abs_profiles[:, mid_mask], axis=1) if np.any(mid_mask) else np.zeros(N)
    far_e = np.mean(abs_profiles[:, far_mask], axis=1) if np.any(far_mask) else np.zeros(N)

    mono_score = 0.0 + 0.5 * (near_e > mid_e) + 0.5 * (mid_e > far_e)
    weighted_dist = np.sum(abs_profiles * d, axis=1) / (np.sum(abs_profiles, axis=1) + 1e-6)
    compactness = 1.0 / (1.0 + weighted_dist)
    ratio_score = np.tanh((near_e / (far_e + 1e-6)) / 3.0)

    # decay_scores = 0.4 * mono_score + 0.3 * compactness + 0.3 * ratio_score
    decay_scores = 0.4 * mono_score + 0.3 * compactness + 0.3 * ratio_score

    # --- 特征 C: Gibbs-like profile score ---
    prof_residual = profiles_gray - smooth_profiles  # profile-level residual
    abs_prof_res = np.abs(prof_residual)
    residual_energy = np.mean(abs_prof_res, axis=1)  # (N,)

    # 同样用 2%~20% 阶跃阈值判断振荡
    osc_signs = np.zeros_like(prof_residual, dtype=np.int8)
    valid_osc = (abs_prof_res > low_th) & (abs_prof_res < high_th)
    osc_signs[valid_osc & (prof_residual > 0)] = 1
    osc_signs[valid_osc & (prof_residual < 0)] = -1

    # sign_alternation_score（简洁版 oscillation）
    mask_osc = osc_signs != 0
    idx_osc = np.where(mask_osc, np.arange(L), 0)
    np.maximum.accumulate(idx_osc, axis=1, out=idx_osc)
    filled_osc = osc_signs[row_indices, idx_osc]
    valid_adj = mask_osc[:, 1:]
    alt = (filled_osc[:, :-1] != filled_osc[:, 1:]) & valid_adj
    total_pairs = np.sum(valid_adj, axis=1).astype(np.float32)
    oscillation_score = np.divide(np.sum(alt, axis=1), total_pairs, out=np.zeros(N), where=total_pairs > 0)

    # decay from profile residual（简洁版：只用近远能量比）
    near_g = np.mean(abs_prof_res[:, near_mask], axis=1) if np.any(near_mask) else np.zeros(N)
    far_g = np.mean(abs_prof_res[:, far_mask], axis=1) if np.any(far_mask) else np.zeros(N)
    # decay_from_prof = np.tanh((far_g / (near_g + 1e-6)) / 3.0)
    decay_from_prof = 0.0 + 0.5 * (near_e < mid_e ) + 0.5 * (mid_e < far_e)

    # gibbs_profile_score per edge point
    gibbs_scores = residual_energy * (oscillation_score + decay_from_prof)

    # ==========================================
    # 4. 聚合与判决 (同前)
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
    # np.add.at(grid_decay_sum, (patch_idx_y, patch_idx_x), decay_scores)
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

    grid_gibbs_p90 = np.zeros((num_rows, num_cols), dtype=np.float32)
    for s, e in zip(starts, ends):
        py, px = sorted_py[s], sorted_px[s]
        grid_gibbs_p90[py, px] = np.percentile(sorted_gibbs[s:e], 70)

    score_grid = np.zeros((num_rows, num_cols), dtype=np.float32)
    mask = (grid_loc_ratio > 1.5) & (grid_mean_crossings > 0) & (grid_mean_decay > 0)
    score_grid[mask] = grid_gibbs_p90[mask]

    return score_grid, img, grid_gibbs_p90


def save_heatmap(img, score_grid, output_path="final_vectorized_heatmap.png"):
    H, W = img.shape[:2]
    score = score_grid.astype(np.float32)
    score_norm = score
    # if score.max() > score.min():
    #     score_norm = (score - score.min()) / (score.max() - score.min())
    # else:
    #     score_norm = np.zeros_like(score)

    score_u8 = np.clip(score_norm * 10, 0, 255).astype(np.uint8)
    heatmap_small = cv2.applyColorMap(score_u8, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap_small, (W, H), interpolation=cv2.INTER_NEAREST)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(output_path, overlay)
    cv2.imshow("heat_map", overlay)
    cv2.waitKey()

if __name__ == "__main__":
    test_img = "../test_data/hisense_mnr_mis_clarity#out1#mnr_input0002.bmp"
    # test_img = "../test_data/05.02.25#out1#mnr_input0012.bmp"
    # test_img = "../test_data/001_OnlineNews#out1#mnr_input0007.bmp"
    test_img = "../test_data/Input_1080p_image_003#out1#mnr_input0007.bmp"
    # test_img = "../test_data/05.02.25#out1#mnr_input0012.bmp"
    score_grid, img, grid_gibbs_p90 = numpy_vectorized_predict_with_decay(test_img)
    save_heatmap(img, score_grid)
    print(f"矩阵化计算完成！p90_gibbs 范围: {grid_gibbs_p90[grid_gibbs_p90 > 0].min():.4f} ~ {grid_gibbs_p90.max():.4f}")