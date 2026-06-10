import cv2
import numpy as np
from scipy.ndimage import map_coordinates


def numpy_vectorized_predict_with_decay(
        image_path, patch_size=8, stride=8, noise_floor=1.5, profile_radius=5
):
    # ==========================================
    # 1. 基础图像与梯度计算
    # ==========================================
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot read image")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape

    edges = cv2.Canny(gray.astype(np.uint8), 30, 80)  # 降低阈值，捕捉微弱的马赛克边缘
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    blur = cv2.GaussianBlur(gray, ksize=(0, 0), sigmaX=1.0)
    residual = gray - blur
    abs_residual = np.abs(residual)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge_band = cv2.dilate(edges, kernel, iterations=1) > 0
    edge_pixels = edges > 0

    edge_neighbor_f = (edge_band & (~edge_pixels)).astype(np.float32)
    non_edge_f = (~edge_band).astype(np.float32)

    # ==========================================
    # 2. 边缘局部性 (loc_ratio) - 全局滑窗
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
    # 3. 法线采点与特征计算 (Crossings & Decay)
    # ==========================================
    ey, ex = np.where(edges > 0)
    valid_mask = grad_mag[ey, ex] > 1e-6
    ey, ex = ey[valid_mask], ex[valid_mask]

    N = len(ey)  # N个边缘点
    mag = grad_mag[ey, ex]
    nx = gx[ey, ex] / mag
    ny = gy[ey, ex] / mag

    t = np.arange(-profile_radius, profile_radius + 1)  # shape: (11,) 假设 radius=5
    samp_x = ex[:, None] + nx[:, None] * t[None, :]  # shape: (N, 11)
    samp_y = ey[:, None] + ny[:, None] * t[None, :]

    # 瞬间提取几万条法线 Profile (N, 11)
    profiles = map_coordinates(residual, [samp_y, samp_x], order=1, mode='constant', cval=0.0)
    abs_profiles = np.abs(profiles)

    # --- 特征 A：极速计算零交叉 (Crossings) ---
    signs = np.zeros_like(profiles, dtype=np.int8)
    signs[profiles > noise_floor] = 1
    signs[profiles < -noise_floor] = -1

    mask_signs = (signs != 0)
    idx = np.where(mask_signs, np.arange(signs.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    row_indices = np.arange(N)[:, None]
    filled_signs = signs[row_indices, idx]

    valid_crossings = (filled_signs[:, :-1] != filled_signs[:, 1:]) & (filled_signs[:, :-1] != 0) & (
                filled_signs[:, 1:] != 0)
    crossings = np.sum(valid_crossings, axis=1)  # shape: (N,)

    # --- 特征 B：纯向量化计算衰减 (Decay) ---
    # 定义距离数组 d: [5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5]
    d = np.abs(t)

    near_mask = (d == 1)
    mid_mask = (d >= 2) & (d <= 3)
    far_mask = (d >= 4)

    # 计算近、中、远三端的平均残差能量
    near_e = np.mean(abs_profiles[:, near_mask], axis=1) if np.any(near_mask) else np.zeros(N)
    mid_e = np.mean(abs_profiles[:, mid_mask], axis=1) if np.any(mid_mask) else np.zeros(N)
    far_e = np.mean(abs_profiles[:, far_mask], axis=1) if np.any(far_mask) else np.zeros(N)

    # B1: 单调衰减分 (0 到 1)
    mono_score = 0.0 + 0.5 * (near_e > mid_e) + 0.5 * (mid_e > far_e)

    # B2: 衰减紧凑度 (能量是否集中在近端)
    weighted_dist = np.sum(abs_profiles * d, axis=1) / (np.sum(abs_profiles, axis=1) + 1e-6)
    compactness = 1.0 / (1.0 + weighted_dist)

    # B3: 远近能量比
    ratio = near_e / (far_e + 1e-6)
    ratio_score = np.tanh(ratio / 3.0)

    # 综合衰减得分 (N,)
    decay_scores = 0.4 * mono_score + 0.3 * compactness + 0.3 * ratio_score

    # ==========================================
    # 4. 聚合回 8x8 Patch 网格
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

    # 散列累加所有特征
    np.add.at(grid_crossings_sum, (patch_idx_y, patch_idx_x), crossings)
    np.add.at(grid_decay_sum, (patch_idx_y, patch_idx_x), decay_scores)
    np.add.at(grid_valid_count, (patch_idx_y, patch_idx_x), 1)

    grid_mean_crossings = grid_crossings_sum / np.maximum(grid_valid_count, 1.0)
    grid_mean_decay = grid_decay_sum / np.maximum(grid_valid_count, 1.0)

    # ==========================================
    # 5. 最终矩阵判决 (综合三个维度)
    # ==========================================
    score_grid = np.zeros((num_rows, num_cols), dtype=np.float32)

    # 门槛设定：由于严重的 JPEG 块效应不会“完美衰减”，把 decay 及格线设为 0.15~0.2 左右
    mask = (grid_loc_ratio > 1.2) & (grid_mean_crossings > 0.8) & (grid_mean_decay > 0.2)

    # 得分组合
    # score_grid[mask] = grid_loc_ratio[mask] * grid_mean_crossings[mask] * grid_mean_decay[mask]
    score_grid[mask] = grid_loc_ratio[mask] + grid_mean_crossings[mask] + grid_mean_decay[mask]
    score_grid[mask] = grid_mean_crossings[mask] * grid_mean_decay[mask]
    score_grid[mask] = grid_mean_decay[mask]

    return score_grid, img


def save_heatmap(img, score_grid, output_path="final_vectorized_heatmap.png"):
    H, W = img.shape[:2]
    score = score_grid.astype(np.float32)
    score_norm = score
    # if score.max() > score.min():
    #     score_norm = (score - score.min()) / (score.max() - score.min())
    # else:
    #     score_norm = np.zeros_like(score)

    score_u8 = np.clip(score_norm * 255, 0, 255).astype(np.uint8)
    heatmap_small = cv2.applyColorMap(score_u8, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap_small, (W, H), interpolation=cv2.INTER_NEAREST)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(output_path, overlay)


if __name__ == "__main__":
    # test_img = "../test_data/hisense_mnr_mis_clarity#out1#mnr_input0002.bmp"
    # test_img = "../test_data/05.02.25#out1#mnr_input0012.bmp"
    test_img = "../test_data/001_OnlineNews#out1#mnr_input0007.bmp"
    test_img = "../test_data/05.02.25#out1#mnr_input0012.bmp"
    test_img = "../test_data/05.02.25#out1#mnr_input0012.bmp"

    save_heatmap(img, score_grid)
    print("矩阵化计算完成！包含了 decay 特征。")