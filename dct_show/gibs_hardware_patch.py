import cv2
import numpy as np


def gaussian_smooth_profiles_1d(profiles, sigma=1.0):
    """
    对 N 条 profile 分别沿横向做 1D Gaussian smoothing。
    profiles: shape = (N, L)

    注意：
    不要直接用 cv2.GaussianBlur(profiles)，否则 N 维方向也会被平滑，
    不同 profile 之间会互相污染。
    """
    profiles = np.asarray(profiles, dtype=np.float32)

    if sigma <= 0:
        return profiles.copy()

    ksize = int(round(sigma * 6)) + 1
    if ksize % 2 == 0:
        ksize += 1

    kernel_1d = cv2.getGaussianKernel(ksize, sigma).astype(np.float32)
    kernel_row = kernel_1d.T

    smooth = cv2.filter2D(
        profiles,
        ddepth=-1,
        kernel=kernel_row,
        borderType=cv2.BORDER_REFLECT_101,
    )

    return smooth.astype(np.float32)


def numpy_vectorized_predict_with_text_ui_branch(
    image_path,

    # patch/profile
    patch_size=4,
    stride=2,
    profile_radius=3,

    # edge detection
    canny_low=60,
    canny_high=140,
    use_sobel_edges=True,
    sobel_percentile=88,
    sobel_min_th=6.0,

    # ringing amplitude threshold, relative to local edge step
    low_ratio=0.01,
    high_ratio=0.22,
    min_abs_low_th=0.6,
    min_edge_step=4.0,

    # natural texture reject
    context_k=24,
    max_patch_edge_density=0.60,
    max_context_edge_density=0.30,
    max_non_edge_residual=5.0,
    max_hv_score=0.35,
    hv_context_edge_density=0.20,
    min_near_far_ratio=1.10,

    # natural branch gate
    natural_min_loc_ratio=1.8,
    natural_min_crossings=1.0,
    natural_min_decay=0.4,
    natural_min_valid_count=2,
    natural_min_edge_step=10.0,

    # text / UI branch gate
    text_min_loc_ratio=1.15,
    text_min_crossings=0.5,
    text_min_decay=0.15,
    text_min_valid_count=1,
    text_min_edge_step=4.0,
    text_max_flat_side_ratio=0.15,

    # hard reject
    hard_context_edge_density=0.35,
    hard_flat_side_ratio=0.20,

    # percentile aggregation
    gibbs_percentile=70,
):
    """
    Vectorized mosquito / Gibbs-like artifact detector.

    设计思路：
    1. natural_mask:
       用于自然图像、建筑、草地、树木、城市远景。
       该分支比较严格，强调降低误检。

    2. text_ui_mask:
       用于字幕、Logo、图表、UI overlay。
       该分支放宽 context/hv 限制，但要求边缘两侧至少一侧比较平坦。

    Returns:
        score_grid: final score grid
        img: BGR image
        debug: dict of intermediate feature grids
    """

    # ============================================================
    # 1. Read image
    # ============================================================
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape

    # ============================================================
    # 2. Gradients and edges
    # ============================================================
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)

    edges_canny = cv2.Canny(gray.astype(np.uint8), canny_low, canny_high)

    if use_sobel_edges:
        grad_th = np.percentile(grad_mag, sobel_percentile)
        grad_th = max(float(grad_th), sobel_min_th)
        edges_sobel = (grad_mag > grad_th).astype(np.uint8) * 255
        edges = ((edges_canny > 0) | (edges_sobel > 0)).astype(np.uint8) * 255
    else:
        edges = edges_canny

    # ============================================================
    # 3. Global residual and edge localization
    # ============================================================
    blur = cv2.GaussianBlur(gray, ksize=(0, 0), sigmaX=1.0)
    residual = gray - blur
    abs_residual = np.abs(residual)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    edge_pixels = edges > 0
    edge_band = cv2.dilate(edges, kernel, iterations=1) > 0

    edge_neighbor_f = (edge_band & (~edge_pixels)).astype(np.float32)
    non_edge_f = (~edge_band).astype(np.float32)

    ctx_k = patch_size + 2 * profile_radius

    sum_neighbor = cv2.boxFilter(
        abs_residual * edge_neighbor_f,
        -1,
        (ctx_k, ctx_k),
        normalize=False,
    )
    count_neighbor = cv2.boxFilter(
        edge_neighbor_f,
        -1,
        (ctx_k, ctx_k),
        normalize=False,
    )

    sum_non_edge = cv2.boxFilter(
        abs_residual * non_edge_f,
        -1,
        (ctx_k, ctx_k),
        normalize=False,
    )
    count_non_edge = cv2.boxFilter(
        non_edge_f,
        -1,
        (ctx_k, ctx_k),
        normalize=False,
    )

    mean_neighbor = sum_neighbor / np.maximum(count_neighbor, 1.0)
    mean_non_edge = sum_non_edge / np.maximum(count_non_edge, 1e-6)
    loc_ratio_full = mean_neighbor / (mean_non_edge + 1e-6)

    # ============================================================
    # 4. Get edge points
    # ============================================================
    ey, ex = np.where(edges > 0)

    num_rows = (H - patch_size) // stride + 1
    num_cols = (W - patch_size) // stride + 1

    if len(ey) == 0:
        score_grid = np.zeros((num_rows, num_cols), dtype=np.float32)
        return score_grid, img, {"final_mask": score_grid.copy()}

    abs_gx = np.abs(gx[ey, ex])
    abs_gy = np.abs(gy[ey, ex])

    valid_grad = (abs_gx + abs_gy) > 1e-6

    ey = ey[valid_grad]
    ex = ex[valid_grad]
    abs_gx = abs_gx[valid_grad]
    abs_gy = abs_gy[valid_grad]

    if len(ey) == 0:
        score_grid = np.zeros((num_rows, num_cols), dtype=np.float32)
        return score_grid, img, {"final_mask": score_grid.copy()}

    N = len(ey)

    # ============================================================
    # 5. Quantize normal directions into 4 directions
    # ============================================================
    gx_val = gx[ey, ex]
    gy_val = gy[ey, ex]

    directions = np.zeros(N, dtype=np.int8)

    is_horiz = abs_gx > 2.0 * abs_gy
    is_vert = abs_gy > 2.0 * abs_gx
    is_diag1 = (~is_horiz) & (~is_vert) & (np.sign(gx_val) == np.sign(gy_val))
    is_diag2 = (~is_horiz) & (~is_vert) & (np.sign(gx_val) != np.sign(gy_val))

    directions[is_horiz] = 0
    directions[is_vert] = 1
    directions[is_diag1] = 2
    directions[is_diag2] = 3

    steps = [
        (0, 1),
        (1, 0),
        (1, 1),
        (-1, 1),
    ]

    t = np.arange(-profile_radius, profile_radius + 1, dtype=np.int32)
    L = len(t)

    offsets = np.array(
        [[dy * t, dx * t] for dy, dx in steps],
        dtype=np.int32,
    )

    selected_offsets = offsets[directions]

    samp_y = ey[:, None] + selected_offsets[:, 0, :]
    samp_x = ex[:, None] + selected_offsets[:, 1, :]

    np.clip(samp_y, 0, H - 1, out=samp_y)
    np.clip(samp_x, 0, W - 1, out=samp_x)

    # ============================================================
    # 6. Extract profiles
    # ============================================================
    profiles_gray = gray[samp_y, samp_x].astype(np.float32)
    profiles_residual = residual[samp_y, samp_x].astype(np.float32)

    smooth_profiles = gaussian_smooth_profiles_1d(profiles_gray, sigma=1.0)

    prof_residual = profiles_gray - smooth_profiles
    abs_prof_res = np.abs(prof_residual)

    edge_step = (
        np.max(smooth_profiles, axis=1)
        - np.min(smooth_profiles, axis=1)
    ).astype(np.float32)

    strong_step_mask = edge_step >= min_edge_step

    low_th = np.maximum(low_ratio * edge_step, min_abs_low_th)[:, None]
    high_th = np.maximum(high_ratio * edge_step, low_th[:, 0] + 1e-6)[:, None]

    # ============================================================
    # 7. Crossings from residual profile
    # ============================================================
    signs = np.zeros_like(profiles_residual, dtype=np.int8)

    valid_amp = (
        (np.abs(profiles_residual) > low_th)
        & (np.abs(profiles_residual) < high_th)
        & strong_step_mask[:, None]
    )

    signs[valid_amp & (profiles_residual > 0)] = 1
    signs[valid_amp & (profiles_residual < 0)] = -1

    mask_signs = signs != 0

    idx = np.where(mask_signs, np.arange(L), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)

    row_indices = np.arange(N)[:, None]
    filled_signs = signs[row_indices, idx]

    valid_crossings = (
        (filled_signs[:, :-1] != filled_signs[:, 1:])
        & (filled_signs[:, :-1] != 0)
        & (filled_signs[:, 1:] != 0)
    )

    crossings = np.sum(valid_crossings, axis=1).astype(np.float32)

    # ============================================================
    # 8. Decay features
    # ============================================================
    d = np.abs(t)

    near_mask = d == 1
    mid_mask = (d >= 2) & (d <= 3)
    far_mask = d >= 4

    near_g = (
        np.mean(abs_prof_res[:, near_mask], axis=1)
        if np.any(near_mask)
        else np.zeros(N, dtype=np.float32)
    )
    mid_g = (
        np.mean(abs_prof_res[:, mid_mask], axis=1)
        if np.any(mid_mask)
        else np.zeros(N, dtype=np.float32)
    )
    far_g = (
        np.mean(abs_prof_res[:, far_mask], axis=1)
        if np.any(far_mask)
        else np.zeros(N, dtype=np.float32)
    )

    near_far_ratio = near_g / (far_g + 1e-6)

    mono_decay = 0.5 * (near_g > mid_g) + 0.5 * (mid_g > far_g)
    decay_from_prof = mono_decay * (near_far_ratio > 1.05)

    weighted_dist = (
        np.sum(abs_prof_res * d[None, :], axis=1)
        / (np.sum(abs_prof_res, axis=1) + 1e-6)
    )

    compactness = 1.0 / (1.0 + weighted_dist)

    decay_scores = (
        0.5 * decay_from_prof
        + 0.3 * compactness
        + 0.2 * np.tanh(near_far_ratio / 3.0)
    ).astype(np.float32)

    # ============================================================
    # 9. Oscillation score
    # ============================================================
    osc_signs = np.zeros_like(prof_residual, dtype=np.int8)

    valid_osc = (
        (abs_prof_res > low_th)
        & (abs_prof_res < high_th)
        & strong_step_mask[:, None]
    )

    osc_signs[valid_osc & (prof_residual > 0)] = 1
    osc_signs[valid_osc & (prof_residual < 0)] = -1

    mask_osc = osc_signs != 0

    idx_osc = np.where(mask_osc, np.arange(L), 0)
    np.maximum.accumulate(idx_osc, axis=1, out=idx_osc)

    filled_osc = osc_signs[row_indices, idx_osc]

    valid_adj = mask_osc[:, 1:]

    alt = (
        (filled_osc[:, :-1] != filled_osc[:, 1:])
        & valid_adj
        & (filled_osc[:, :-1] != 0)
        & (filled_osc[:, 1:] != 0)
    )

    total_pairs = np.sum(valid_adj, axis=1).astype(np.float32)

    oscillation_score = np.divide(
        np.sum(alt, axis=1).astype(np.float32),
        total_pairs,
        out=np.zeros(N, dtype=np.float32),
        where=total_pairs > 0,
    )

    # ============================================================
    # 10. Flat-side feature
    # ============================================================
    left_side_mask = t <= -2
    right_side_mask = t >= 2

    left_std = (
        np.std(profiles_gray[:, left_side_mask], axis=1)
        if np.any(left_side_mask)
        else np.zeros(N, dtype=np.float32)
    )

    right_std = (
        np.std(profiles_gray[:, right_side_mask], axis=1)
        if np.any(right_side_mask)
        else np.zeros(N, dtype=np.float32)
    )

    min_side_std = np.minimum(left_std, right_std)

    flat_side_ratio = min_side_std / (edge_step + 1e-6)

    # ============================================================
    # 11. Per-edge Gibbs-like score
    # ============================================================
    residual_energy = np.mean(abs_prof_res, axis=1).astype(np.float32)

    norm_residual_energy = residual_energy / (edge_step + 1e-6)

    amp_reasonable = (
        (norm_residual_energy > low_ratio)
        & (norm_residual_energy < high_ratio)
        & strong_step_mask
    )

    gibbs_scores = (
        residual_energy
        * oscillation_score
        * decay_scores
        * amp_reasonable.astype(np.float32)
    ).astype(np.float32)

    # ============================================================
    # 12. Grid coordinates
    # ============================================================
    grid_y_centers = np.arange(num_rows) * stride + patch_size // 2
    grid_x_centers = np.arange(num_cols) * stride + patch_size // 2

    grid_X, grid_Y = np.meshgrid(grid_x_centers, grid_y_centers)

    grid_loc_ratio = loc_ratio_full[grid_Y, grid_X]
    grid_non_edge_residual = mean_non_edge[grid_Y, grid_X]

    patch_idx_y = np.clip(ey // stride, 0, num_rows - 1)
    patch_idx_x = np.clip(ex // stride, 0, num_cols - 1)

    # ============================================================
    # 13. Accumulate patch-level features
    # ============================================================
    grid_crossings_sum = np.zeros((num_rows, num_cols), dtype=np.float32)
    grid_decay_sum = np.zeros((num_rows, num_cols), dtype=np.float32)
    grid_valid_count = np.zeros((num_rows, num_cols), dtype=np.float32)

    grid_dir0_count = np.zeros((num_rows, num_cols), dtype=np.float32)
    grid_dir1_count = np.zeros((num_rows, num_cols), dtype=np.float32)
    grid_dir2_count = np.zeros((num_rows, num_cols), dtype=np.float32)
    grid_dir3_count = np.zeros((num_rows, num_cols), dtype=np.float32)

    grid_near_sum = np.zeros((num_rows, num_cols), dtype=np.float32)
    grid_far_sum = np.zeros((num_rows, num_cols), dtype=np.float32)
    grid_edge_step_sum = np.zeros((num_rows, num_cols), dtype=np.float32)
    grid_flat_side_sum = np.zeros((num_rows, num_cols), dtype=np.float32)

    np.add.at(grid_crossings_sum, (patch_idx_y, patch_idx_x), crossings)
    np.add.at(grid_decay_sum, (patch_idx_y, patch_idx_x), decay_scores)
    np.add.at(grid_valid_count, (patch_idx_y, patch_idx_x), 1.0)

    np.add.at(grid_near_sum, (patch_idx_y, patch_idx_x), near_g)
    np.add.at(grid_far_sum, (patch_idx_y, patch_idx_x), far_g)
    np.add.at(grid_edge_step_sum, (patch_idx_y, patch_idx_x), edge_step)
    np.add.at(grid_flat_side_sum, (patch_idx_y, patch_idx_x), flat_side_ratio)

    np.add.at(
        grid_dir0_count,
        (patch_idx_y[directions == 0], patch_idx_x[directions == 0]),
        1.0,
    )
    np.add.at(
        grid_dir1_count,
        (patch_idx_y[directions == 1], patch_idx_x[directions == 1]),
        1.0,
    )
    np.add.at(
        grid_dir2_count,
        (patch_idx_y[directions == 2], patch_idx_x[directions == 2]),
        1.0,
    )
    np.add.at(
        grid_dir3_count,
        (patch_idx_y[directions == 3], patch_idx_x[directions == 3]),
        1.0,
    )

    grid_mean_crossings = grid_crossings_sum / np.maximum(grid_valid_count, 1.0)
    grid_mean_decay = grid_decay_sum / np.maximum(grid_valid_count, 1.0)

    grid_edge_density = grid_valid_count / float(patch_size * patch_size)

    grid_near_mean = grid_near_sum / np.maximum(grid_valid_count, 1.0)
    grid_far_mean = grid_far_sum / np.maximum(grid_valid_count, 1.0)
    grid_near_far_ratio = grid_near_mean / (grid_far_mean + 1e-6)

    grid_edge_step_mean = grid_edge_step_sum / np.maximum(grid_valid_count, 1.0)
    grid_flat_side_ratio = grid_flat_side_sum / np.maximum(grid_valid_count, 1.0)

    grid_hv_score = (
        np.minimum(grid_dir0_count, grid_dir1_count)
        / np.maximum(grid_valid_count, 1.0)
    )

    dir_stack = np.stack(
        [
            grid_dir0_count,
            grid_dir1_count,
            grid_dir2_count,
            grid_dir3_count,
        ],
        axis=0,
    )

    grid_dominant_dir_ratio = (
        np.max(dir_stack, axis=0) / np.maximum(grid_valid_count, 1.0)
    )

    # ============================================================
    # 14. Context edge density
    # ============================================================
    edge_f = (edges > 0).astype(np.float32)

    context_edge_count = cv2.boxFilter(
        edge_f,
        -1,
        (context_k, context_k),
        normalize=False,
    )

    context_edge_density_full = context_edge_count / float(context_k * context_k)
    grid_context_edge_density = context_edge_density_full[grid_Y, grid_X]

    # ============================================================
    # 15. Percentile Gibbs score per patch
    # ============================================================
    order = np.lexsort((patch_idx_x, patch_idx_y))

    sorted_gibbs = gibbs_scores[order]
    sorted_py = patch_idx_y[order]
    sorted_px = patch_idx_x[order]

    grid_gibbs_p = np.zeros((num_rows, num_cols), dtype=np.float32)

    if len(sorted_gibbs) > 0:
        change = (
            (np.diff(sorted_py) != 0)
            | (np.diff(sorted_px) != 0)
        )

        bounds = np.where(change)[0] + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [len(sorted_gibbs)]])

        for s, e in zip(starts, ends):
            py = sorted_py[s]
            px = sorted_px[s]

            vals = sorted_gibbs[s:e]
            vals = vals[vals > 0]

            if len(vals) == 0:
                continue

            grid_gibbs_p[py, px] = np.percentile(vals, gibbs_percentile)

    # ============================================================
    # 16. Natural texture reject
    # ============================================================
    dense_patch_reject = grid_edge_density > max_patch_edge_density

    dense_context_reject = (
        grid_context_edge_density > max_context_edge_density
    )

    non_edge_texture_reject = (
        grid_non_edge_residual > max_non_edge_residual
    )

    hv_grid_reject = (
        (grid_hv_score > max_hv_score)
        & (grid_context_edge_density > hv_context_edge_density)
    )

    weak_decay_reject = (
        grid_near_far_ratio < min_near_far_ratio
    )

    texture_reject = (
        dense_patch_reject
        | dense_context_reject
        | non_edge_texture_reject
        | hv_grid_reject
        | weak_decay_reject
    )

    # ============================================================
    # 17. Two-branch decision
    # ============================================================

    # Branch A:
    # 自然图像 / 建筑 / 草地等，严格控制误检。
    natural_mask = (
        (grid_loc_ratio > natural_min_loc_ratio)
        & (grid_mean_crossings >= natural_min_crossings)
        & (grid_mean_decay > natural_min_decay)
        & (grid_gibbs_p > 0)
        & (grid_valid_count >= natural_min_valid_count)
        & (grid_edge_step_mean > natural_min_edge_step)
        & (~texture_reject)
    )

    # Branch B:
    # 字幕 / UI / 图表 / Logo，提升召回。
    # 不用 context_edge_density/hv_score 强杀，因为文字图表本身就是高 edge density。
    # 核心条件是 flat_side_ratio：边缘两侧至少一侧比较平坦。
    text_ui_mask = (
        (grid_loc_ratio > text_min_loc_ratio)
        & (grid_mean_crossings >= text_min_crossings)
        & (grid_mean_decay > text_min_decay)
        & (grid_gibbs_p > 0)
        & (grid_valid_count >= text_min_valid_count)
        & (grid_edge_step_mean > text_min_edge_step)
        & (grid_flat_side_ratio < text_max_flat_side_ratio)
    )

    # 极端纹理保护：
    # 如果 context edge density 极高，并且边缘两侧都不平坦，则排除。
    # 但不会误杀正常字幕，因为字幕通常 flat_side_ratio 较低。
    hard_texture_reject = (
        (grid_context_edge_density > hard_context_edge_density)
        & (grid_flat_side_ratio > hard_flat_side_ratio)
    )

    final_mask = (natural_mask | text_ui_mask) & (~hard_texture_reject)

    score_grid = np.zeros((num_rows, num_cols), dtype=np.float32)
    score_grid[final_mask] = grid_gibbs_p[final_mask]

    debug = {
        "grid_gibbs_p": grid_gibbs_p,
        "grid_loc_ratio": grid_loc_ratio,
        "grid_mean_crossings": grid_mean_crossings,
        "grid_mean_decay": grid_mean_decay,
        "grid_valid_count": grid_valid_count,
        "grid_edge_density": grid_edge_density,
        "grid_context_edge_density": grid_context_edge_density,
        "grid_non_edge_residual": grid_non_edge_residual,
        "grid_near_far_ratio": grid_near_far_ratio,
        "grid_edge_step_mean": grid_edge_step_mean,
        "grid_flat_side_ratio": grid_flat_side_ratio,
        "grid_hv_score": grid_hv_score,
        "grid_dominant_dir_ratio": grid_dominant_dir_ratio,

        "texture_reject": texture_reject.astype(np.float32),
        "dense_patch_reject": dense_patch_reject.astype(np.float32),
        "dense_context_reject": dense_context_reject.astype(np.float32),
        "non_edge_texture_reject": non_edge_texture_reject.astype(np.float32),
        "hv_grid_reject": hv_grid_reject.astype(np.float32),
        "weak_decay_reject": weak_decay_reject.astype(np.float32),

        "natural_mask": natural_mask.astype(np.float32),
        "text_ui_mask": text_ui_mask.astype(np.float32),
        "hard_texture_reject": hard_texture_reject.astype(np.float32),
        "final_mask": final_mask.astype(np.float32),
    }

    return score_grid, img, debug


def save_heatmap(
    img,
    score_grid,
    output_path="final_heatmap.png",
    scale=10.0,
    show=False,
):
    H, W = img.shape[:2]

    score = score_grid.astype(np.float32)
    score_u8 = np.clip(score * scale, 0, 255).astype(np.uint8)

    heatmap_small = cv2.applyColorMap(score_u8, cv2.COLORMAP_JET)
    heatmap = cv2.resize(
        heatmap_small,
        (W, H),
        interpolation=cv2.INTER_NEAREST,
    )

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    cv2.imwrite(output_path, overlay)

    if show:
        cv2.imshow("heatmap", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return overlay


def save_debug_grid_as_heatmap(
    img,
    grid,
    output_path,
    normalize=True,
    show=False,
):
    H, W = img.shape[:2]

    g = grid.astype(np.float32)

    if normalize:
        g_min = float(np.min(g))
        g_max = float(np.max(g))

        if g_max > g_min:
            g = (g - g_min) / (g_max - g_min)
        else:
            g = np.zeros_like(g)

        g_u8 = np.clip(g * 255.0, 0, 255).astype(np.uint8)
    else:
        g_u8 = np.clip(g * 255.0, 0, 255).astype(np.uint8)

    heatmap_small = cv2.applyColorMap(g_u8, cv2.COLORMAP_JET)
    heatmap = cv2.resize(
        heatmap_small,
        (W, H),
        interpolation=cv2.INTER_NEAREST,
    )

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    cv2.imwrite(output_path, overlay)

    if show:
        cv2.imshow(output_path, overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return overlay


if __name__ == "__main__":
    test_img = "../test_data/Input_1080p_image_003#out1#mnr_input0007.bmp"
    # test_img = "../test_data/hisense_mnr_mis_clarity#out1#mnr_input0002.bmp"
    test_img = "../test_data/05.02.25#out1#mnr_input0012.bmp"
    test_img = "../test_data/001_OnlineNews#out1#mnr_input0007.bmp"
    # test_img = "../test_data/Input_1080p_image_003#out1#mnr_input0007.bmp"
    # test_img = "../test_data/05.02.25#out1#mnr_input0012.bmp"

    score_grid, img, debug = numpy_vectorized_predict_with_text_ui_branch(
        test_img,

        # 对新闻字幕 / 图表 / UI 边缘更友好
        patch_size=4,
        stride=2,
        profile_radius=3,

        canny_low=60,
        canny_high=140,
        use_sobel_edges=True,
        sobel_percentile=88,
        sobel_min_th=6.0,

        low_ratio=0.01,
        high_ratio=0.22,
        min_abs_low_th=0.6,
        min_edge_step=4.0,

        context_k=24,

        max_patch_edge_density=0.60,
        max_context_edge_density=0.30,
        max_non_edge_residual=5.0,
        max_hv_score=0.35,
        hv_context_edge_density=0.20,
        min_near_far_ratio=1.10,

        natural_min_loc_ratio=1.8,
        natural_min_crossings=1.0,
        natural_min_decay=0.4,
        natural_min_valid_count=2,
        natural_min_edge_step=10.0,

        text_min_loc_ratio=1.15,
        text_min_crossings=0.5,
        text_min_decay=0.15,
        text_min_valid_count=1,
        text_min_edge_step=4.0,
        text_max_flat_side_ratio=0.15,

        hard_context_edge_density=0.35,
        hard_flat_side_ratio=0.20,

        gibbs_percentile=70,
    )

    save_heatmap(
        img,
        score_grid,
        output_path="final_heatmap_text_ui_branch.png",
        scale=12.0,
        show=True,
    )

    # Debug outputs
    save_debug_grid_as_heatmap(
        img,
        debug["text_ui_mask"],
        "debug_text_ui_mask.png",
        normalize=False,
        show=False,
    )

    save_debug_grid_as_heatmap(
        img,
        debug["natural_mask"],
        "debug_natural_mask.png",
        normalize=False,
        show=False,
    )

    save_debug_grid_as_heatmap(
        img,
        debug["grid_flat_side_ratio"],
        "debug_flat_side_ratio.png",
        normalize=True,
        show=False,
    )

    save_debug_grid_as_heatmap(
        img,
        debug["grid_context_edge_density"],
        "debug_context_edge_density.png",
        normalize=True,
        show=False,
    )

    save_debug_grid_as_heatmap(
        img,
        debug["texture_reject"],
        "debug_texture_reject.png",
        normalize=False,
        show=False,
    )

    valid_scores = score_grid[score_grid > 0]

    if len(valid_scores) > 0:
        print(
            "Done. score range: "
            f"{valid_scores.min():.4f} ~ {valid_scores.max():.4f}"
        )
    else:
        print("Done. No candidates detected under current thresholds.")