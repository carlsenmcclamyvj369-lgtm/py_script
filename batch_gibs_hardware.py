import cv2
import numpy as np
from pathlib import Path
from scipy.ndimage import map_coordinates


def robust_mad_threshold(values, scale=1.0):
    """用 MAD 估计鲁棒阈值。"""
    values = np.asarray(values, dtype=np.float32)
    med = np.median(values)
    mad = np.median(np.abs(values - med)) + 1e-6
    threshold = scale * 1.4826 * mad
    return float(threshold)


def numpy_vectorized_predict_with_decay(
        image_path, patch_size=8, stride=8, profile_radius=5, scale=1.0
):
    # ==========================================
    # 1. 基础图像与梯度计算
    # ==========================================
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot read image")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape

    edges = cv2.Canny(gray.astype(np.uint8), 30, 80)
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
    # 3. 法线采点与特征计算 (Crossings & Decay)
    # ==========================================
    ey, ex = np.where(edges > 0)
    valid_mask = grad_mag[ey, ex] > 1e-6
    ey, ex = ey[valid_mask], ex[valid_mask]

    N = len(ey)
    mag = grad_mag[ey, ex]
    nx = gx[ey, ex] / mag
    ny = gy[ey, ex] / mag

    t = np.arange(-profile_radius, profile_radius + 1)
    samp_x = ex[:, None] + nx[:, None] * t[None, :]
    samp_y = ey[:, None] + ny[:, None] * t[None, :]

    profiles = map_coordinates(residual, [samp_y, samp_x], order=1, mode='constant', cval=0.0)
    abs_profiles = np.abs(profiles)

    # --- 特征 A: 零交叉 (Crossings) ---
    # 每条 profile 独立 mean 阈值
    prof_thresh = scale * np.mean(abs_profiles, axis=1, keepdims=True)  # shape (N, 1)

    signs = np.zeros_like(profiles, dtype=np.int8)
    signs[profiles > prof_thresh] = 1
    signs[profiles < -prof_thresh] = -1

    mask_signs = (signs != 0)
    idx = np.where(mask_signs, np.arange(signs.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    row_indices = np.arange(N)[:, None]
    filled_signs = signs[row_indices, idx]

    valid_crossings = (filled_signs[:, :-1] != filled_signs[:, 1:]) & (filled_signs[:, :-1] != 0) & (
                filled_signs[:, 1:] != 0)
    crossings = np.sum(valid_crossings, axis=1)

    # --- 特征 B: 衰减 (Decay) ---
    d = np.abs(t)

    near_mask = (d == 1)
    mid_mask = (d >= 2) & (d <= 3)
    far_mask = (d >= 4)

    near_e = np.mean(abs_profiles[:, near_mask], axis=1) if np.any(near_mask) else np.zeros(N)
    mid_e = np.mean(abs_profiles[:, mid_mask], axis=1) if np.any(mid_mask) else np.zeros(N)
    far_e = np.mean(abs_profiles[:, far_mask], axis=1) if np.any(far_mask) else np.zeros(N)

    mono_score = 0.0 + 0.5 * (near_e > mid_e) + 0.5 * (mid_e > far_e)

    weighted_dist = np.sum(abs_profiles * d, axis=1) / (np.sum(abs_profiles, axis=1) + 1e-6)
    compactness = 1.0 / (1.0 + weighted_dist)

    ratio = near_e / (far_e + 1e-6)
    ratio_score = np.tanh(ratio / 3.0)

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

    np.add.at(grid_crossings_sum, (patch_idx_y, patch_idx_x), crossings)
    np.add.at(grid_decay_sum, (patch_idx_y, patch_idx_x), decay_scores)
    np.add.at(grid_valid_count, (patch_idx_y, patch_idx_x), 1)

    grid_mean_crossings = grid_crossings_sum / np.maximum(grid_valid_count, 1.0)
    grid_mean_decay = grid_decay_sum / np.maximum(grid_valid_count, 1.0)

    # ==========================================
    # 5. 最终矩阵判决
    # ==========================================
    score_grid = np.zeros((num_rows, num_cols), dtype=np.float32)

    mask = (grid_loc_ratio > 1.2) & (grid_mean_crossings > 1.5) & (grid_mean_decay > 0.5)
    score_grid[mask] = grid_loc_ratio[mask] * grid_mean_crossings[mask] * grid_mean_decay[mask]

    return score_grid, img


def save_heatmap(img, score_grid, output_path):
    H, W = img.shape[:2]
    score = score_grid.astype(np.float32)
    score_norm = score

    score_u8 = np.clip(score_norm * 25, 0, 255).astype(np.uint8)
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
        stem = bmp_path.stem  # 不含后缀的文件名
        try:
            score_grid, img = numpy_vectorized_predict_with_decay(str(bmp_path))

            # 保存热力图
            heatmap_path = output_dir / f"{stem}.png"
            save_heatmap(img, score_grid, str(heatmap_path))

            # 保存原始 score_grid (npy)
            npy_path = output_dir / f"{stem}_score.npy"
            np.save(str(npy_path), score_grid)

            print(f"✓ {stem}  shape={score_grid.shape}")
        except Exception as e:
            print(f"✗ {stem}  错误: {e}")

    print(f"\n完成！共处理 {len(bmp_files)} 张图片，结果保存至 {output_dir}")


if __name__ == "__main__":
    main()
