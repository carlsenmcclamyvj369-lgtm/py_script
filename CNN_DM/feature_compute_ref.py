import numpy as np
import cv2


# ─────────────────────────────────────────────
# 1. Y 亮度计算
# ─────────────────────────────────────────────

def compute_y_from_rgb(img_array):
    """从 RGB 图像数组计算 BT.709 Y 亮度 (float64)"""
    r = img_array[:, :, 0].astype(np.float64)
    g = img_array[:, :, 1].astype(np.float64)
    b = img_array[:, :, 2].astype(np.float64)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# ─────────────────────────────────────────────
# 2. 方差图 (var_map)
# ─────────────────────────────────────────────

def compute_var_map(y_full):
    """
    计算全图每个点的 3×3 窗口方差。
    返回 (h, w) 的 float64 数组，边界一圈为 np.nan。
    """
    k = np.ones((3, 3), dtype=np.float64) / 9.0
    mean = cv2.filter2D(y_full, -1, k, borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.filter2D(y_full ** 2, -1, k, borderType=cv2.BORDER_REFLECT)
    var_map = np.maximum(mean_sq - mean ** 2, 0)  # 防浮点误差负值
    var_map[0, :] = np.nan
    var_map[-1, :] = np.nan
    var_map[:, 0] = np.nan
    var_map[:, -1] = np.nan
    return var_map


# ─────────────────────────────────────────────
# 3. 残差图 (residual_map)
# ─────────────────────────────────────────────

def compute_residual_map(y_full, mean=None):
    """
    计算残差图：|中心像素 - 3×3 均值|。
    若不传入 mean 则内部重新计算。
    返回 (h, w) 的 float64 数组，边界一圈为 np.nan。
    """
    if mean is None:
        k = np.ones((3, 3), dtype=np.float64) / 9.0
        mean = cv2.filter2D(y_full, -1, k, borderType=cv2.BORDER_REFLECT)
    h, w = y_full.shape
    residual_map = np.full((h, w), np.nan, dtype=np.float64)
    residual_map[1:-1, 1:-1] = (y_full - mean)[1:-1, 1:-1]
    return residual_map


# ─────────────────────────────────────────────
# 4. 拉普拉斯图 (lap_map)
# ─────────────────────────────────────────────

def compute_lap_map(y_full):
    """
    计算拉普拉斯响应绝对值图。
    核: [[0,-1,0],[-1,4,-1],[0,-1,0]]
    返回 (h, w) 的 float64 数组，边界一圈为 np.nan。
    """
    lap_k = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
    lap = cv2.filter2D(y_full, -1, lap_k, borderType=cv2.BORDER_REFLECT)
    h, w = y_full.shape
    lap_map = np.full((h, w), np.nan, dtype=np.float64)
    lap_map[1:-1, 1:-1] = np.abs(lap[1:-1, 1:-1])
    return lap_map


# ─────────────────────────────────────────────
# 5. 梯度图 (grad_map)
# ─────────────────────────────────────────────

def compute_grad_map(y_full):
    """
    计算四方向梯度绝对值之和图。
    grad = |c-up| + |c-down| + |c-left| + |c-right|
    返回 (h, w) 的 float64 数组，边界一圈为 np.nan。
    """
    h, w = y_full.shape
    grad_map = np.full((h, w), np.nan, dtype=np.float64)
    grad_interior = (
        np.abs(y_full[1:-1, 1:-1] - y_full[:-2, 1:-1]) +
        np.abs(y_full[1:-1, 1:-1] - y_full[2:, 1:-1]) +
        np.abs(y_full[1:-1, 1:-1] - y_full[1:-1, :-2]) +
        np.abs(y_full[1:-1, 1:-1] - y_full[1:-1, 2:])
    )
    grad_map[1:-1, 1:-1] = grad_interior
    return grad_map


# ─────────────────────────────────────────────
# 6. 边缘方向图 (h_edge_map, v_edge_map)
# ─────────────────────────────────────────────

def compute_edge_maps(y_full):
    """
    计算水平和垂直梯度差分图。
    返回 (h_edge_map, v_edge_map)，均为 (h, w) 的 float64 数组。
    h_edge_map 第 0 列为 np.nan，v_edge_map 第 0 行为 np.nan。
    """
    h, w = y_full.shape
    h_edge_map = np.full((h, w), np.nan, dtype=np.float64)
    h_edge_map[:, 1:] = np.abs(np.diff(y_full, axis=1))
    v_edge_map = np.full((h, w), np.nan, dtype=np.float64)
    v_edge_map[1:, :] = np.abs(np.diff(y_full, axis=0))
    return h_edge_map, v_edge_map


# ─────────────────────────────────────────────
# 7. 分块统计：方差特征
# ─────────────────────────────────────────────

def compute_block_var_stats(y_full, var_map, gx, gy,
                            low_th=100, high_th=500, very_high_th=2000):
    """
    按 grid 分块统计方差特征。
    返回 dict[(bi, bj)] -> {"mean_var", "max_var", "top5_var",
                              "low_var_count", "high_var_count", "very_high_var_count"}
    """
    h, w = y_full.shape
    block_stats = {}
    for by in range(0, h, gy):
        for bx in range(0, w, gx):
            y1, y2 = by, min(by + gy, h)
            x1, x2 = bx, min(bx + gx, w)
            block_vars = var_map[y1:y2, x1:x2].flatten()
            block_vars = block_vars[~np.isnan(block_vars)]
            if len(block_vars) == 0:
                continue
            sorted_vars = np.sort(block_vars)
            bi, bj = by // gy, bx // gx
            block_stats[(bi, bj)] = {
                "mean_var": float(np.mean(block_vars)),
                "max_var": float(np.max(block_vars)),
                "top5_var": float(np.mean(sorted_vars[-5:]) if len(sorted_vars) >= 5 else np.mean(sorted_vars)),
                "low_var_count": int(np.sum(block_vars < low_th)),
                "high_var_count": int(np.sum(block_vars > high_th)),
                "very_high_var_count": int(np.sum(block_vars > very_high_th)),
            }
    return block_stats


# ─────────────────────────────────────────────
# 8. 分块统计：残差特征
# ─────────────────────────────────────────────

def compute_block_residual_stats(y_full, residual_map, gx, gy):
    """
    按 grid 分块统计残差特征。
    返回 dict[(bi, bj)] -> {"residual_mean", "residual_max"}
    """
    h, w = y_full.shape
    block_stats = {}
    for by in range(0, h, gy):
        for bx in range(0, w, gx):
            y1, y2 = by, min(by + gy, h)
            x1, x2 = bx, min(bx + gx, w)
            block_res = residual_map[y1:y2, x1:x2].flatten()
            block_res = block_res[~np.isnan(block_res)]
            if len(block_res) == 0:
                continue
            bi, bj = by // gy, bx // gx
            block_stats[(bi, bj)] = {
                "residual_mean": float(np.mean(np.abs(block_res))),
                "residual_max": float(np.max(np.abs(block_res))),
            }
    return block_stats


# ─────────────────────────────────────────────
# 9. 分块统计：拉普拉斯特征
# ─────────────────────────────────────────────

def compute_block_lap_stats(y_full, lap_map, gx, gy):
    """
    按 grid 分块统计拉普拉斯特征。
    返回 dict[(bi, bj)] -> {"lap_mean", "lap_max"}
    """
    h, w = y_full.shape
    block_stats = {}
    for by in range(0, h, gy):
        for bx in range(0, w, gx):
            y1, y2 = by, min(by + gy, h)
            x1, x2 = bx, min(bx + gx, w)
            block_lap = lap_map[y1:y2, x1:x2].flatten()
            block_lap = block_lap[~np.isnan(block_lap)]
            if len(block_lap) == 0:
                continue
            bi, bj = by // gy, bx // gx
            block_stats[(bi, bj)] = {
                "lap_mean": float(np.mean(block_lap)),
                "lap_max": float(np.max(block_lap)),
            }
    return block_stats


# ─────────────────────────────────────────────
# 10. 分块统计：梯度特征
# ─────────────────────────────────────────────

def compute_block_grad_stats(y_full, grad_map, gx, gy):
    """
    按 grid 分块统计梯度特征。
    返回 dict[(bi, bj)] -> {"grad_mean", "grad_max"}
    """
    h, w = y_full.shape
    block_stats = {}
    for by in range(0, h, gy):
        for bx in range(0, w, gx):
            y1, y2 = by, min(by + gy, h)
            x1, x2 = bx, min(bx + gx, w)
            block_grad = grad_map[y1:y2, x1:x2].flatten()
            block_grad = block_grad[~np.isnan(block_grad)]
            if len(block_grad) == 0:
                continue
            bi, bj = by // gy, bx // gx
            block_stats[(bi, bj)] = {
                "grad_mean": float(np.mean(block_grad)),
                "grad_max": float(np.max(block_grad)),
            }
    return block_stats


# ─────────────────────────────────────────────
# 11. 分块统计：边缘方向特征
# ─────────────────────────────────────────────

def compute_block_edge_stats(y_full, h_edge_map, v_edge_map, gx, gy,
                              max_strength_th=0.000001):
    """
    按 grid 分块统计边缘方向特征。
    返回 dict[(bi, bj)] -> {"edge_strength", "h_strength", "h_strength_max",
                              "h_strength_min", "v_strength", "v_strength_max",
                              "v_strength_min", "edge_orientation_conf"}
    """
    h, w = y_full.shape
    block_stats = {}
    for by in range(0, h, gy):
        for bx in range(0, w, gx):
            y1, y2 = by, min(by + gy, h)
            x1, x2 = bx, min(bx + gx, w)
            block_h = h_edge_map[y1:y2, x1:x2].flatten()
            block_v = v_edge_map[y1:y2, x1:x2].flatten()
            block_h = block_h[~np.isnan(block_h)]
            block_v = block_v[~np.isnan(block_v)]
            if len(block_h) == 0 or len(block_v) == 0:
                continue
            h_s = float(np.mean(block_h))
            v_s = float(np.mean(block_v))
            max_strength = max(h_s, v_s)
            if max_strength <= max_strength_th:
                orient_conf = 0.0
            else:
                orient_conf = abs(h_s - v_s) / max_strength
            bi, bj = by // gy, bx // gx
            block_stats[(bi, bj)] = {
                "edge_strength": max_strength,
                "h_strength": h_s,
                "h_strength_max": float(np.max(block_h)),
                "h_strength_min": float(np.min(block_h)),
                "v_strength": v_s,
                "v_strength_max": float(np.max(block_v)),
                "v_strength_min": float(np.min(block_v)),
                "edge_orientation_conf": orient_conf,
            }
    return block_stats


# ─────────────────────────────────────────────
# 12. 分块统计：振荡特征 & 行/列均值差
# ─────────────────────────────────────────────

def compute_block_osc_stats(y_full, gx, gy):
    """
    按 grid 分块统计振荡特征（二阶差分）和行/列均值差。
    返回 dict[(bi, bj)] -> {"row_second_diff", "row_second_diff_max",
                              "row_second_diff_min", "col_second_diff",
                              "col_second_diff_max", "col_second_diff_min",
                              "second_diff_max", "second_diff_min_max",
                              "row_diff_mean", "row_diff_max",
                              "col_diff_mean", "col_diff_max"}
    """
    h, w = y_full.shape
    block_stats = {}
    for by in range(0, h, gy):
        for bx in range(0, w, gx):
            y1, y2 = by, min(by + gy, h)
            x1, x2 = bx, min(bx + gx, w)
            block_y = y_full[y1:y2, x1:x2]
            bh, bw = block_y.shape

            # 二阶差分
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

            row_energy = float(np.mean(row_energies)) if row_energies else 0.0
            col_energy = float(np.mean(col_energies)) if col_energies else 0.0
            row_second_diff_max = float(np.max(row_energies)) if row_energies else 0.0
            row_second_diff_min = float(np.min(row_energies)) if row_energies else 0.0
            col_second_diff_max = float(np.max(col_energies)) if col_energies else 0.0
            col_second_diff_min = float(np.min(col_energies)) if col_energies else 0.0

            # 行/列均值差
            row_means = [float(block_y[r, :].mean()) for r in range(bh)]
            row_diffs = [abs(row_means[r+1] - row_means[r]) for r in range(bh - 1)] if bh >= 2 else [0.0]
            col_means = [float(block_y[:, c].mean()) for c in range(bw)]
            col_diffs = [abs(col_means[c+1] - col_means[c]) for c in range(bw - 1)] if bw >= 2 else [0.0]

            bi, bj = by // gy, bx // gx
            block_stats[(bi, bj)] = {
                "row_second_diff": row_energy,
                "row_second_diff_max": row_second_diff_max,
                "row_second_diff_min": row_second_diff_min,
                "col_second_diff": col_energy,
                "col_second_diff_max": col_second_diff_max,
                "col_second_diff_min": col_second_diff_min,
                "second_diff_max": max(row_energy, col_energy),
                "second_diff_min_max": (min(row_energy, col_energy) / max(row_energy, col_energy))
                                        if max(row_energy, col_energy) > 0 else 0.0,
                "row_diff_mean": float(np.mean(row_diffs)),
                "row_diff_max": float(np.max(row_diffs)),
                "col_diff_mean": float(np.mean(col_diffs)),
                "col_diff_max": float(np.max(col_diffs)),
            }
    return block_stats


# ─────────────────────────────────────────────
# 13. 振铃轮廓评分
# ─────────────────────────────────────────────

def _norm(x, lo, hi):
    """将 x 归一化到 [0, 1]，区间由 lo/hi 定义"""
    if hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def ringing_score_vec(v, eps=3.0,
                      dyn_lo=20, dyn_hi=120,
                      d2_lo=5, d2_hi=60,
                      sign_lo=1, sign_hi=4,
                      dyn_ratio=0.45, d2_ratio=0.35, sign_ratio=0.20):
    """
    对一维向量计算振铃轮廓评分。
    返回 (total_score, dyn_score, d2_score, sign_score) 均为 float。
    """
    v = np.asarray(v, dtype=np.float64)
    d = np.diff(v)
    d2 = np.diff(v, n=2)
    dyn = float(np.max(v) - np.min(v))
    d2_energy = float(np.mean(np.abs(d2))) if len(d2) > 0 else 0.0

    signs = np.sign(d)
    signs[np.abs(d) < eps] = 0
    nonzero = signs[signs != 0]
    if len(nonzero) < 2:
        sign_changes = 0
    else:
        sign_changes = int(np.sum(np.diff(nonzero) != 0))

    ds = _norm(dyn, dyn_lo, dyn_hi)
    d2s = _norm(d2_energy, d2_lo, d2_hi)
    ss = _norm(sign_changes, sign_lo, sign_hi)
    total = dyn_ratio * ds + d2_ratio * d2s + sign_ratio * ss
    return total, ds, d2s, ss


def ringing_stats_for_block(block_y, bh, bw, eps=3.0,
                             dyn_lo=20, dyn_hi=120,
                             d2_lo=5, d2_hi=60,
                             sign_lo=1, sign_hi=4,
                             dyn_ratio=0.45, d2_ratio=0.35, sign_ratio=0.20):
    """
    计算一个 block 的行/列振铃轮廓统计，返回字典或 None。
    key 列表: row_ringing_max|min|mean, col_ringing_max|min|mean,
               ringing_mean_max|min|min_max, profile_ringing_max|mean,
               row/col_ringing_dyn_score|d2_score|sign_score
    """
    row_scores, row_dyn, row_d2, row_sign = [], [], [], []
    for ry in range(bh):
        total, ds, d2s, ss = ringing_score_vec(
            block_y[ry, :], eps, dyn_lo, dyn_hi, d2_lo, d2_hi, sign_lo, sign_hi,
            dyn_ratio, d2_ratio, sign_ratio)
        row_scores.append(total)
        row_dyn.append(ds)
        row_d2.append(d2s)
        row_sign.append(ss)

    col_scores, col_dyn, col_d2, col_sign = [], [], [], []
    for rx in range(bw):
        total, ds, d2s, ss = ringing_score_vec(
            block_y[:, rx], eps, dyn_lo, dyn_hi, d2_lo, d2_hi, sign_lo, sign_hi,
            dyn_ratio, d2_ratio, sign_ratio)
        col_scores.append(total)
        col_dyn.append(ds)
        col_d2.append(d2s)
        col_sign.append(ss)

    if not (row_scores and col_scores):
        return None

    all_scores = row_scores + col_scores
    row_mean = float(np.mean(row_scores))
    col_mean = float(np.mean(col_scores))
    return {
        "row_ringing_max": float(np.max(row_scores)),
        "row_ringing_min": float(np.min(row_scores)),
        "row_ringing_mean": row_mean,
        "ringing_mean_max": max(row_mean, col_mean),
        "ringing_mean_min": min(row_mean, col_mean),
        "ringing_mean_min_max": (min(row_mean, col_mean) / max(row_mean, col_mean))
                                 if max(row_mean, col_mean) > 0 else 0.0,
        "profile_ringing_max": float(np.max(all_scores)),
        "profile_ringing_mean": float(np.mean(all_scores)),
        "row_ringing_dyn_score": float(np.mean(row_dyn)),
        "row_ringing_d2_score": float(np.mean(row_d2)),
        "row_ringing_sign_score": float(np.mean(row_sign)),
        "col_ringing_max": float(np.max(col_scores)),
        "col_ringing_min": float(np.min(col_scores)),
        "col_ringing_mean": col_mean,
        "col_ringing_dyn_score": float(np.mean(col_dyn)),
        "col_ringing_d2_score": float(np.mean(col_d2)),
        "col_ringing_sign_score": float(np.mean(col_sign)),
    }


def compute_block_ringing_stats(y_full, gx, gy,
                                 eps=3.0,
                                 dyn_lo=20, dyn_hi=120,
                                 d2_lo=5, d2_hi=60,
                                 sign_lo=1, sign_hi=4,
                                 dyn_ratio=0.45, d2_ratio=0.35, sign_ratio=0.20):
    """
    按 grid 分块计算振铃轮廓统计。
    返回 dict[(bi, bj)] -> 同上 ringing_stats_for_block 返回的字典
    """
    h, w = y_full.shape
    block_stats = {}
    for by in range(0, h, gy):
        for bx in range(0, w, gx):
            y1, y2 = by, min(by + gy, h)
            x1, x2 = bx, min(bx + gx, w)
            block_y = y_full[y1:y2, x1:x2]
            bh, bw = block_y.shape
            result = ringing_stats_for_block(
                block_y, bh, bw, eps, dyn_lo, dyn_hi, d2_lo, d2_hi, sign_lo, sign_hi,
                dyn_ratio, d2_ratio, sign_ratio)
            if result is not None:
                bi, bj = by // gy, bx // gx
                block_stats[(bi, bj)] = result
    return block_stats


# ─────────────────────────────────────────────
# 14. 滤波差值图计算
# ─────────────────────────────────────────────

def compute_filter_diff_maps(y_full):
    """
    对 Y 通道生成高斯/双边 3,5,7 滤波及差值图。
    返回 (gauss_diffs, bilat_diffs)，每个为 [d3, d5, d7] 的 list，
    其中 d3=abs(orig - 3x3), d5=abs(3x3-5x5), d7=abs(5x5-7x7)。
    """
    y_u8 = np.clip(np.round(y_full), 0, 255).astype(np.uint8)
    g3 = cv2.GaussianBlur(y_u8, (3, 3), 0).astype(np.float64)
    g5 = cv2.GaussianBlur(y_u8, (5, 5), 0).astype(np.float64)
    g7 = cv2.GaussianBlur(y_u8, (7, 7), 0).astype(np.float64)
    b3 = cv2.bilateralFilter(y_u8, 3, 75, 75).astype(np.float64)
    b5 = cv2.bilateralFilter(y_u8, 5, 75, 75).astype(np.float64)
    b7 = cv2.bilateralFilter(y_u8, 7, 75, 75).astype(np.float64)

    gauss_diffs = [
        np.abs(y_full - g3),
        np.abs(g3 - g5),
        np.abs(g5 - g7),
    ]
    bilat_diffs = [
        np.abs(y_full - b3),
        np.abs(b3 - b5),
        np.abs(b5 - b7),
    ]
    return gauss_diffs, bilat_diffs


# ─────────────────────────────────────────────
# 15. 特征数组转特征图（repeat 到原图大小）
# ─────────────────────────────────────────────

_SCALE_RULES = {
    "mean_var": 0.25, "max_var": 0.25, "top5_var": 0.25,
    "residual_mean": None, "residual_max": None,
    "lap_mean": 0.25, "lap_max": 0.25,
    "grad_mean": 0.25, "grad_max": 0.25,
    "edge_strength": None,
    "h_strength": None, "h_strength_max": None, "h_strength_min": None,
    "v_strength": None, "v_strength_max": None, "v_strength_min": None,
    "edge_orientation_conf": 255.0,
    "row_second_diff": 0.5, "col_second_diff": 0.5,
    "row_second_diff_max": 0.5, "row_second_diff_min": 0.5,
    "col_second_diff_max": 0.5, "col_second_diff_min": 0.5,
    "second_diff_max": 0.5,
    "second_diff_min_max": 255.0,
    "profile_ringing_max": 255.0, "profile_ringing_mean": 255.0,
    "ringing_mean_max": 255.0, "ringing_mean_min": 255.0,
    "ringing_mean_min_max": 255.0,
    "row_ringing_max": 255.0, "row_ringing_min": 255.0, "row_ringing_mean": 255.0,
    "row_ringing_dyn_score": 255.0, "row_ringing_d2_score": 255.0,
    "row_ringing_sign_score": 255.0,
    "col_ringing_max": 255.0, "col_ringing_min": 255.0, "col_ringing_mean": 255.0,
    "col_ringing_dyn_score": 255.0, "col_ringing_d2_score": 255.0,
    "col_ringing_sign_score": 255.0,
}


def save_feature_maps(block_data_dict, basename, tmp_dir, h, w, gx, gy):
    """
    将 block 级特征数组 repeat 到原图大小并保存为 BMP。
    block_data_dict: {feat_name: (bh_grid, bw_grid) float64 ndarray}
    返回 {feat_name: out_path} 映射。
    """
    import os
    block_size = gx * gy
    scale_rules = dict(_SCALE_RULES)
    scale_rules["low_var_count"] = 256.0 / block_size if block_size > 0 else 1.0
    scale_rules["high_var_count"] = 256.0 / block_size if block_size > 0 else 1.0
    scale_rules["very_high_var_count"] = 256.0 / block_size if block_size > 0 else 1.0

    os.makedirs(tmp_dir, exist_ok=True)
    feat_paths = {}
    for feat_name, block_map in block_data_dict.items():
        if block_map.size == 0:
            continue
        full = np.repeat(np.repeat(block_map, gy, axis=0), gx, axis=1)
        full = full[:h, :w]

        scale = scale_rules.get(feat_name)
        if scale is not None:
            img_u8 = np.clip(full * scale, 0, 255).astype(np.uint8)
        else:
            img_u8 = np.clip(full, 0, 255).astype(np.uint8)

        out_path = os.path.join(tmp_dir, f"{basename}_{feat_name}.bmp")
        cv2.imwrite(out_path, img_u8)
        feat_paths[feat_name] = out_path

    return feat_paths
