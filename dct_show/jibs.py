import csv
import cv2
import numpy as np


# ============================================================
# 1. 基础函数
# ============================================================

def read_gray_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Cannot read image: {}".format(image_path))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return gray


def bilinear_sample(image, x, y):
    """
    双线性插值采样。

    Args:
        image: H x W 灰度图
        x: 浮点横坐标
        y: 浮点纵坐标

    Returns:
        float 采样值；如果坐标越界则返回 None。
    """
    h, w = image.shape

    if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
        return None

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    dx = x - x0
    dy = y - y0

    v00 = image[y0, x0]
    v01 = image[y0, x1]
    v10 = image[y1, x0]
    v11 = image[y1, x1]

    value = (
        v00 * (1.0 - dx) * (1.0 - dy)
        + v01 * dx * (1.0 - dy)
        + v10 * (1.0 - dx) * dy
        + v11 * dx * dy
    )

    return float(value)


def robust_mad_threshold(values, scale=1.0):
    """用 MAD 估计鲁棒阈值。"""
    values = np.asarray(values, dtype=np.float32)

    med = np.median(values)
    mad = np.median(np.abs(values - med)) + 1e-6

    threshold = scale * 1.4826 * mad
    return float(threshold)


# ============================================================
# 2. 边缘和梯度
# ============================================================

def compute_edges_and_gradient(gray, canny_low=80, canny_high=160):
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)

    edges = cv2.Canny(gray_u8, canny_low, canny_high)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    grad_mag = np.sqrt(gx * gx + gy * gy)

    return edges, gx, gy, grad_mag


def sample_edge_normal_profile_global(gray, x, y, nx, ny, radius=6):
    """
    在整张图上沿边缘法线采样 profile。

    patch 是 8x8，但 profile 允许跨出当前 patch，
    这样可以观察边缘附近的振铃衰减。
    """
    values = []

    for t in range(-radius, radius + 1):
        sx = x + t * nx
        sy = y + t * ny

        value = bilinear_sample(gray, sx, sy)

        if value is None:
            return None

        values.append(value)

    return np.asarray(values, dtype=np.float32)


# ============================================================
# 3. profile 残差
# ============================================================

def smooth_profile(profile, sigma=1.0):
    """对一维 profile 做高斯平滑。"""
    profile = np.asarray(profile, dtype=np.float32)

    profile_2d = profile.reshape(1, -1)

    trend_2d = cv2.GaussianBlur(
        profile_2d,
        ksize=(0, 0),
        sigmaX=sigma,
        sigmaY=0,
    )

    return trend_2d.reshape(-1).astype(np.float32)


def compute_profile_residual(profile, sigma=1.0):
    """residual = profile - smooth(profile)"""
    profile = np.asarray(profile, dtype=np.float32)

    trend = smooth_profile(profile, sigma=sigma)
    residual = profile - trend

    return trend, residual


# ============================================================
# 4. 正负交替特征
# ============================================================

def effective_zero_crossings(residual, amp_threshold):
    """统计有效零交叉次数。"""
    r = np.asarray(residual, dtype=np.float32)

    signs = np.zeros_like(r, dtype=np.int32)
    signs[r > amp_threshold] = 1
    signs[r < -amp_threshold] = -1

    signs = signs[signs != 0]

    if len(signs) < 2:
        return 0

    return int(np.sum(signs[1:] != signs[:-1]))


def sign_alternation_score(residual, amp_threshold):
    """符号交替分数，越接近 1 越像 + - + - 的振荡。"""
    r = np.asarray(residual, dtype=np.float32)

    signs = np.zeros_like(r, dtype=np.int32)
    signs[r > amp_threshold] = 1
    signs[r < -amp_threshold] = -1

    signs = signs[signs != 0]

    if len(signs) < 2:
        return 0.0

    adjacent_product = signs[:-1] * signs[1:]
    score = np.mean(adjacent_product < 0)

    return float(score)


def alternating_peak_valley_count(residual, amp_threshold):
    """统计峰谷交替次数。"""
    r = np.asarray(residual, dtype=np.float32)

    extrema_signs = []

    for i in range(1, len(r) - 1):
        is_positive_peak = (
            r[i] > r[i - 1]
            and r[i] > r[i + 1]
            and r[i] > amp_threshold
        )

        is_negative_peak = (
            r[i] < r[i - 1]
            and r[i] < r[i + 1]
            and r[i] < -amp_threshold
        )

        if is_positive_peak:
            extrema_signs.append(1)
        elif is_negative_peak:
            extrema_signs.append(-1)

    if len(extrema_signs) < 2:
        return 0

    extrema_signs = np.asarray(extrema_signs, dtype=np.int32)

    return int(np.sum(extrema_signs[1:] != extrema_signs[:-1]))


# ============================================================
# 5. 距离衰减特征
# ============================================================

def small_profile_decay_features(residual, center):
    """
    针对 8x8 patch 的小范围衰减特征。

    profile_radius=6 时，profile 长度为 13。
    距离分段：
        near: d = 1
        mid : d = 2~3
        far : d = 4~6
    """
    r = np.asarray(residual, dtype=np.float32)

    abs_r = np.abs(r)
    distances = np.abs(np.arange(len(r)) - center)

    def band_mean(lo, hi):
        vals = abs_r[(distances >= lo) & (distances <= hi)]

        if len(vals) == 0:
            return 0.0

        return float(np.mean(vals))

    near_energy = band_mean(1, 1)
    mid_energy = band_mean(2, 3)
    far_energy = band_mean(4, 6)

    eps = 1e-6

    monotonic_decay_score = 0.0

    if near_energy > mid_energy:
        monotonic_decay_score += 0.5

    if mid_energy > far_energy:
        monotonic_decay_score += 0.5

    energy_weighted_distance = float(
        np.sum(abs_r * distances) / (np.sum(abs_r) + eps)
    )

    decay_compactness = float(
        1.0 / (1.0 + energy_weighted_distance)
    )

    decay_ratio_near_far = float(
        near_energy / (far_energy + eps)
    )

    return {
        "near_energy": float(near_energy),
        "mid_energy": float(mid_energy),
        "far_energy": float(far_energy),
        "decay_ratio_near_far": decay_ratio_near_far,
        "monotonic_decay_score": float(monotonic_decay_score),
        "energy_weighted_distance": float(energy_weighted_distance),
        "decay_compactness": float(decay_compactness),
    }


# ============================================================
# 6. 单条 profile 的 Gibbs-like 特征
# ============================================================

def gibbs_profile_features_for_8x8(profile, smooth_sigma=1.0, amp_threshold=None):
    """对一条边缘法线 profile 计算 Gibbs-like ringing 特征。"""
    profile = np.asarray(profile, dtype=np.float32)

    center = len(profile) // 2

    _, residual = compute_profile_residual(
        profile,
        sigma=smooth_sigma,
    )

    if amp_threshold is None:
        amp_threshold = robust_mad_threshold(residual, scale=1.5)

    zero_crossings = effective_zero_crossings(
        residual,
        amp_threshold,
    )

    sign_alt = sign_alternation_score(
        residual,
        amp_threshold,
    )

    peak_valley_count = alternating_peak_valley_count(
        residual,
        amp_threshold,
    )

    decay = small_profile_decay_features(
        residual,
        center=center,
    )

    residual_energy = float(np.mean(np.abs(residual)))

    oscillation_score = float(
        0.4 * sign_alt
        + 0.3 * min(zero_crossings / 4.0, 1.0)
        + 0.3 * min(peak_valley_count / 4.0, 1.0)
    )

    decay_ratio_score = float(
        np.tanh(decay["decay_ratio_near_far"] / 3.0)
    )

    decay_score = float(
        0.4 * decay["monotonic_decay_score"]
        + 0.3 * decay["decay_compactness"]
        + 0.3 * decay_ratio_score
    )

    gibbs_profile_score = float(
        residual_energy * oscillation_score * decay_score
    )

    return {
        "residual_energy": float(residual_energy),
        "amp_threshold": float(amp_threshold),

        "effective_zero_crossings": float(zero_crossings),
        "sign_alternation_score": float(sign_alt),
        "alternating_peak_valley_count": float(peak_valley_count),
        "oscillation_score": float(oscillation_score),

        "near_energy": float(decay["near_energy"]),
        "mid_energy": float(decay["mid_energy"]),
        "far_energy": float(decay["far_energy"]),
        "decay_ratio_near_far": float(decay["decay_ratio_near_far"]),
        "monotonic_decay_score": float(decay["monotonic_decay_score"]),
        "energy_weighted_distance": float(decay["energy_weighted_distance"]),
        "decay_compactness": float(decay["decay_compactness"]),
        "decay_score": float(decay_score),

        "gibbs_profile_score": float(gibbs_profile_score),
    }


# ============================================================
# 7. patch 内边缘局部性
# ============================================================

def patch_edge_localization_score_8x8(
    gray,
    edges,
    patch_x,
    patch_y,
    patch_size=8,
    context_radius=6,
    residual_sigma=1.0,
):
    """
    计算当前 8x8 patch 附近的边缘局部性。

    为避免 8x8 内部统计太不稳定，这里使用 patch 周围 context。
    """
    H, W = gray.shape

    x0 = max(0, patch_x - context_radius)
    y0 = max(0, patch_y - context_radius)
    x1 = min(W, patch_x + patch_size + context_radius)
    y1 = min(H, patch_y + patch_size + context_radius)

    crop_gray = gray[y0:y1, x0:x1]
    crop_edges = edges[y0:y1, x0:x1]

    blur = cv2.GaussianBlur(
        crop_gray,
        ksize=(0, 0),
        sigmaX=residual_sigma,
        sigmaY=residual_sigma,
    )

    abs_residual = np.abs(crop_gray - blur)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    edge_band = cv2.dilate(
        crop_edges,
        kernel,
        iterations=1,
    ) > 0

    edge_pixels = crop_edges > 0

    edge_neighbor = edge_band & (~edge_pixels)
    non_edge = ~edge_band

    eps = 1e-6

    if np.any(edge_neighbor):
        edge_residual_mean = float(
            np.mean(abs_residual[edge_neighbor])
        )
    else:
        edge_residual_mean = 0.0

    if np.any(non_edge):
        non_edge_residual_mean = float(
            np.mean(abs_residual[non_edge])
        )
    else:
        non_edge_residual_mean = eps

    edge_localization_score = float(
        edge_residual_mean / (non_edge_residual_mean + eps)
    )

    return {
        "edge_residual_mean": float(edge_residual_mean),
        "non_edge_residual_mean": float(non_edge_residual_mean),
        "edge_localization_score": float(edge_localization_score),
    }


# ============================================================
# 8. patch 分类规则
# ============================================================

def classify_8x8_patch(features):
    """
    经验规则。
    阈值需要根据具体数据集调参。
    """
    if features["num_valid_profiles"] == 0:
        return "no_valid_edge_profiles"

    score = features["patch_score"]
    edge_loc = features["edge_localization_score"]
    osc = features["mean_oscillation_score"]
    decay = features["mean_decay_score"]
    residual_energy = features["mean_residual_energy"]

    if score > 1.2 and edge_loc > 1.3 and osc > 0.30 and decay > 0.25:
        return "mosquito_noise_8x8_patch"

    if score > 0.6 and edge_loc > 1.1 and osc > 0.20:
        return "possible_mosquito_noise_8x8_patch"

    if residual_energy > 4.0 and edge_loc <= 1.1:
        return "texture_or_general_compression_8x8_patch"

    return "weak_or_clean_8x8_patch"


# ============================================================
# 9. 单个 8x8 patch 预测
# ============================================================

def predict_one_8x8_patch(
    gray,
    edges,
    gx,
    gy,
    grad_mag,
    patch_x,
    patch_y,
    patch_size=8,
    profile_radius=6,
    profile_smooth_sigma=1.0,
    max_edge_points_per_patch=32,
    rng=None,
):
    """
    对单个 8x8 patch 做预测。

    边缘点只取当前 8x8 patch 内部；
    profile 在整张图上采样。
    """
    patch_edges = edges[
        patch_y:patch_y + patch_size,
        patch_x:patch_x + patch_size,
    ]

    local_ys, local_xs = np.where(patch_edges > 0)
    num_edge_pixels = len(local_xs)

    base_result = {
        "x": int(patch_x),
        "y": int(patch_y),
        "w": int(patch_size),
        "h": int(patch_size),
        "num_edge_pixels": int(num_edge_pixels),
    }

    edge_loc = patch_edge_localization_score_8x8(
        gray=gray,
        edges=edges,
        patch_x=patch_x,
        patch_y=patch_y,
        patch_size=patch_size,
        context_radius=profile_radius,
        residual_sigma=1.0,
    )

    def make_empty_result():
        result = {
            **base_result,
            **edge_loc,

            "num_valid_profiles": 0,
            "patch_score": 0.0,

            "mean_oscillation_score": 0.0,
            "mean_decay_score": 0.0,
            "mean_residual_energy": 0.0,

            "mean_sign_alternation_score": 0.0,
            "mean_decay_ratio_near_far": 0.0,
            "mean_energy_weighted_distance": 0.0,

            "mean_gibbs_profile_score": 0.0,
            "median_gibbs_profile_score": 0.0,
            "p90_gibbs_profile_score": 0.0,
        }

        result["patch_label"] = classify_8x8_patch(result)
        return result

    if num_edge_pixels == 0:
        return make_empty_result()

    if rng is None:
        rng = np.random.default_rng(123)

    if num_edge_pixels > max_edge_points_per_patch:
        idx = rng.choice(
            num_edge_pixels,
            size=max_edge_points_per_patch,
            replace=False,
        )

        local_xs = local_xs[idx]
        local_ys = local_ys[idx]

    profile_features = []

    for lx, ly in zip(local_xs, local_ys):
        x = patch_x + int(lx)
        y = patch_y + int(ly)

        grad_x = gx[y, x]
        grad_y = gy[y, x]

        norm = float(np.sqrt(grad_x * grad_x + grad_y * grad_y))

        if norm < 1e-6:
            continue

        nx = float(grad_x / norm)
        ny = float(grad_y / norm)

        profile = sample_edge_normal_profile_global(
            gray=gray,
            x=float(x),
            y=float(y),
            nx=nx,
            ny=ny,
            radius=profile_radius,
        )

        if profile is None:
            continue

        feat = gibbs_profile_features_for_8x8(
            profile,
            smooth_sigma=profile_smooth_sigma,
            amp_threshold=None,
        )

        feat["edge_gradient_magnitude"] = float(grad_mag[y, x])

        profile_features.append(feat)

    if len(profile_features) == 0:
        return make_empty_result()

    def collect(key):
        values = [f[key] for f in profile_features]
        return np.asarray(values, dtype=np.float32)

    gibbs_scores = collect("gibbs_profile_score")
    oscillation_scores = collect("oscillation_score")
    decay_scores = collect("decay_score")
    residual_energies = collect("residual_energy")

    sign_alt_scores = collect("sign_alternation_score")
    decay_ratios = collect("decay_ratio_near_far")
    weighted_distances = collect("energy_weighted_distance")

    p90_gibbs_profile_score = float(
        np.percentile(gibbs_scores, 90)
    )

    patch_score = float(
        p90_gibbs_profile_score
        * np.log1p(edge_loc["edge_localization_score"])
    )

    result = {
        **base_result,
        **edge_loc,

        "num_valid_profiles": int(len(profile_features)),
        "patch_score": float(patch_score),

        "mean_oscillation_score": float(np.mean(oscillation_scores)),
        "mean_decay_score": float(np.mean(decay_scores)),
        "mean_residual_energy": float(np.mean(residual_energies)),

        "mean_sign_alternation_score": float(np.mean(sign_alt_scores)),
        "mean_decay_ratio_near_far": float(np.mean(decay_ratios)),
        "mean_energy_weighted_distance": float(np.mean(weighted_distances)),

        "mean_gibbs_profile_score": float(np.mean(gibbs_scores)),
        "median_gibbs_profile_score": float(np.median(gibbs_scores)),
        "p90_gibbs_profile_score": float(p90_gibbs_profile_score),
    }

    result["patch_label"] = classify_8x8_patch(result)

    return result


# ============================================================
# 10. 整图 8x8 patch 预测
# ============================================================

def predict_8x8_patches(
    image_path,
    patch_size=8,
    stride=8,
    canny_low=80,
    canny_high=160,
    profile_radius=6,
    profile_smooth_sigma=1.0,
    max_edge_points_per_patch=32,
    random_seed=123,
    drop_incomplete_patch=True,
):
    gray = read_gray_image(image_path)

    H, W = gray.shape

    edges, gx, gy, grad_mag = compute_edges_and_gradient(
        gray,
        canny_low=canny_low,
        canny_high=canny_high,
    )

    rng = np.random.default_rng(random_seed)

    xs = list(range(0, W, stride))
    ys = list(range(0, H, stride))

    if drop_incomplete_patch:
        xs = [x for x in xs if x + patch_size <= W]
        ys = [y for y in ys if y + patch_size <= H]

    score_grid = np.zeros(
        (len(ys), len(xs)),
        dtype=np.float32,
    )

    label_grid = np.empty(
        (len(ys), len(xs)),
        dtype=object,
    )

    results = []

    for row_idx, y in enumerate(ys):
        for col_idx, x in enumerate(xs):
            result = predict_one_8x8_patch(
                gray=gray,
                edges=edges,
                gx=gx,
                gy=gy,
                grad_mag=grad_mag,
                patch_x=x,
                patch_y=y,
                patch_size=patch_size,
                profile_radius=profile_radius,
                profile_smooth_sigma=profile_smooth_sigma,
                max_edge_points_per_patch=max_edge_points_per_patch,
                rng=rng,
            )

            result["patch_row"] = int(row_idx)
            result["patch_col"] = int(col_idx)

            results.append(result)

            score_grid[row_idx, col_idx] = float(result["patch_score"])
            label_grid[row_idx, col_idx] = result["patch_label"]

    return results, score_grid, label_grid


# ============================================================
# 11. 保存 CSV
# ============================================================

def save_results_csv(results, output_csv):
    if len(results) == 0:
        return

    # 收集所有 key，避免某些 result 字段顺序或字段集合不一致导致写 CSV 出错。
    keys = []
    seen = set()

    for row in results:
        for key in row.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=keys,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(results)


# ============================================================
# 12. 保存 heatmap
# ============================================================

def save_8x8_patch_heatmap(
    image_path,
    score_grid,
    output_path="patch_8x8_score_heatmap.png",
):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Cannot read image: {}".format(image_path))

    H, W = img.shape[:2]

    score = score_grid.astype(np.float32)

    if score.size == 0:
        raise ValueError(
            "score_grid is empty. Please check image size, patch_size, and stride."
        )

    # if float(score.max()) > float(score.min()):
    #     score_norm = (score - score.min()) / (score.max() - score.min())
    # else:
    #     score_norm = np.zeros_like(score)
    score_norm = score
    score_u8 = np.clip(
        score_norm * 25,
        0,
        255,
    ).astype(np.uint8)

    heatmap_small = cv2.applyColorMap(
        score_u8,
        cv2.COLORMAP_JET,
    )

    heatmap = cv2.resize(
        heatmap_small,
        (W, H),
        interpolation=cv2.INTER_NEAREST,
    )

    overlay = cv2.addWeighted(
        img,
        0.55,
        heatmap,
        0.45,
        0,
    )

    cv2.imwrite(output_path, overlay)


# ============================================================
# 13. 主函数
# ============================================================

if __name__ == "__main__":
    image_path = "../test_data/001_OnlineNews#out1#mnr_input0007.bmp"
    image_path = "../test_data/hisense_mnr_mis_clarity#out1#mnr_input0002.bmp"
    image_path = "../test_data/05.02.25#out1#mnr_input0012.bmp"

    results, score_grid, label_grid = predict_8x8_patches(
        image_path=image_path,

        patch_size=8,
        stride=8,

        canny_low=80,
        canny_high=160,

        profile_radius=4,
        profile_smooth_sigma=1.0,

        max_edge_points_per_patch=32,

        random_seed=123,
        drop_incomplete_patch=True,
    )

    print("num patches:", len(results))
    print("score_grid shape:", score_grid.shape)
    print("label_grid shape:", label_grid.shape)

    print("first 5 results:")
    for row in results[:5]:
        print(row)

    save_results_csv(
        results,
        "patch_8x8_predictions.csv",
    )

    save_8x8_patch_heatmap(
        image_path,
        score_grid,
        "patch_8x8_score_heatmap.png",
    )

    print("saved: patch_8x8_predictions.csv")
    print("saved: patch_8x8_score_heatmap.png")