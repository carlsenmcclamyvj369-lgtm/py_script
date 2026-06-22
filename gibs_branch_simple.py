"""
gibbs_branch_textline_v2.py

基于 gibs_branch_simple.py 的增强版 Gibbs 伪影检测器。

主要优化：
  1. 增强文字/字幕/UI 边沿判别：
     - corner_density: 文字笔画、数字、中文字通常角点多
     - direction entropy: 文字方向更丰富，长直线方向更单一
     - chroma ringing: 针对青/紫/红蓝彩边，补充灰度通道漏检

  2. 抑制直线/测试卡/规则图形过处理：
     - line_like: 高边缘密度 + 高方向一致性 + 低角点密度
     - 对 line_like 采用 soft penalty，不一刀切

  3. 保持 two-branch 决策：
     - natural branch: 严格控自然纹理误检
     - text_ui branch: 加入 text-likeness，避免变成“宽松边缘分支”
"""

import cv2
import numpy as np
from dataclasses import dataclass


# ============================================================
# 配置
# ============================================================
@dataclass
class GibbsConfig:
    """Gibbs detector 配置参数。"""

    # 网格 & profile
    patch_size: int = 4
    stride: int = 2
    profile_radius: int = 3

    # 边缘检测
    canny_low: int = 60
    canny_high: int = 140
    use_sobel_edges: bool = True
    sobel_percentile: float = 88
    sobel_min_th: float = 6.0

    # ringing 幅度阈值
    low_ratio: float = 0.01
    high_ratio: float = 0.22
    min_abs_low_th: float = 0.6
    min_edge_step: float = 4.0

    # 上下文窗口
    context_k: int = 24

    # ── natural 分支 ─────────────────────
    natural_min_loc_ratio: float = 1.8
    natural_min_crossings: float = 1.0
    natural_min_decay: float = 0.4
    natural_min_valid_count: int = 2
    natural_min_edge_step: float = 10.0

    # natural texture reject
    max_patch_edge_density: float = 0.60
    max_context_edge_density: float = 0.30
    max_non_edge_residual: float = 5.0
    max_hv_score: float = 0.35
    hv_context_edge_density: float = 0.20
    min_near_far_ratio: float = 1.10

    # ── text/UI 分支基础门限 ──────────────
    text_min_loc_ratio: float = 0.75
    text_min_crossings: float = 0.2
    text_min_decay: float = 0.08
    text_min_valid_count: int = 1
    text_min_edge_step: float = 1.2
    text_max_flat_side_ratio: float = 0.28

    # ── 新增：文字结构判别 ────────────────
    text_min_corner_density: float = 0.0015
    text_min_dir_entropy: float = 0.22
    text_min_chroma_ring: float = 0.025

    # 对很小的文字，角点和方向熵可能都不稳定，因此允许 chroma 单独增强
    text_chroma_boost_min_score: float = 0.015

    # ── 新增：长直线 / 测试卡线条抑制 ─────
    line_min_edge_density: float = 0.12
    line_min_context_edge_density: float = 0.06
    line_min_max_dir_ratio: float = 0.82
    line_max_dir_entropy: float = 0.22
    line_max_corner_density: float = 0.0020
    line_penalty: float = 0.35

    # 如果 line_like 但同时非常像文字，则不强压
    line_text_rescue_corner_density: float = 0.0030
    line_text_rescue_dir_entropy: float = 0.35

    # ── hard reject ─────────────────────
    hard_context_edge_density: float = 0.38
    hard_flat_side_ratio: float = 0.25

    # ── 聚合 ─────────────────────────────
    gibbs_percentile: float = 70

    # ── text mask 后处理 ─────────────────
    enable_text_close: bool = True
    text_close_kernel_w: int = 3
    text_close_kernel_h: int = 1


# ============================================================
# 工具函数
# ============================================================
def _smooth_1d(profiles, sigma=1.0):
    """逐 profile 进行 1D Gaussian 平滑。"""
    profiles = np.asarray(profiles, dtype=np.float32)

    if sigma <= 0:
        return profiles.copy()

    ksize = int(round(sigma * 6)) | 1
    kernel = cv2.getGaussianKernel(ksize, sigma).astype(np.float32).T

    return cv2.filter2D(
        profiles,
        -1,
        kernel,
        borderType=cv2.BORDER_REFLECT_101,
    ).astype(np.float32)


def _safe_div(a, b, eps=1e-6):
    return a / (b + eps)


def _grid_zeros(n_rows, n_cols):
    return np.zeros((n_rows, n_cols), dtype=np.float32)


def _box_mean(data, weight, k):
    """带权 box mean。"""
    s = cv2.boxFilter(data * weight, -1, (k, k), normalize=False)
    c = cv2.boxFilter(weight, -1, (k, k), normalize=False)
    return s / np.maximum(c, 1e-6)


def _compute_corner_density(gray, context_k):
    """
    Harris corner density。
    用于增强文字/数字/中文字边沿，压制长直线。
    """
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)

    corner = cv2.cornerHarris(gray_u8, blockSize=2, ksize=3, k=0.04)
    corner = cv2.dilate(corner, None)

    # 使用高百分位，避免背景噪声变成 corner
    th = max(float(np.percentile(corner, 99.2)), 1e-6)
    corner_map = (corner > th).astype(np.float32)

    corner_cnt = cv2.boxFilter(
        corner_map,
        -1,
        (context_k, context_k),
        normalize=False,
    )

    return corner_cnt / float(context_k * context_k)


def _compute_chroma_residual(img):
    """
    计算色度 residual，用于捕捉文字边沿的青/紫/红蓝彩边。
    """
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]

    cr_blur = cv2.GaussianBlur(cr, (0, 0), sigmaX=1.0)
    cb_blur = cv2.GaussianBlur(cb, (0, 0), sigmaX=1.0)

    res_cr = cr - cr_blur
    res_cb = cb - cb_blur

    return np.sqrt(res_cr * res_cr + res_cb * res_cb).astype(np.float32)


# ============================================================
# 主检测函数
# ============================================================
def detect_gibbs(image_path, cfg: GibbsConfig | None = None):
    """
    检测 Gibbs 伪影，返回：
        score_grid, bgr_img, final_mask

    score_grid:
        grid 级别分数。final_mask 为 True 的区域才有非零分数。

    final_mask:
        需要处理的 Gibbs patch。
    """
    if cfg is None:
        cfg = GibbsConfig()

    # ------------------------------------------------------------
    # 1. 读图
    # ------------------------------------------------------------
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape

    n_rows = (H - cfg.patch_size) // cfg.stride + 1
    n_cols = (W - cfg.patch_size) // cfg.stride + 1

    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("图像尺寸过小，无法生成 patch grid。")

    # ------------------------------------------------------------
    # 2. 梯度 + 边缘
    # ------------------------------------------------------------
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    edges = cv2.Canny(
        np.clip(gray, 0, 255).astype(np.uint8),
        cfg.canny_low,
        cfg.canny_high,
    )

    grad_mag = np.sqrt(gx * gx + gy * gy)

    if cfg.use_sobel_edges:
        th = max(float(np.percentile(grad_mag, cfg.sobel_percentile)), cfg.sobel_min_th)
        sobel_edge = ((grad_mag > th).astype(np.uint8) * 255)
        edges = edges | sobel_edge

    # ------------------------------------------------------------
    # 3. 全局 residual + localization ratio
    # ------------------------------------------------------------
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    residual = gray - blur
    abs_residual = np.abs(residual)

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge_band = cv2.dilate(edges, kernel3, iterations=1) > 0

    edge_near = (edge_band & (edges <= 0)).astype(np.float32)
    non_edge = (~edge_band).astype(np.float32)

    ctx_k_loc = cfg.patch_size + 2 * cfg.profile_radius

    mean_near = _box_mean(abs_residual, edge_near, ctx_k_loc)
    mean_non_edge = _box_mean(abs_residual, non_edge, ctx_k_loc)

    loc_ratio = _safe_div(mean_near, mean_non_edge)

    # ------------------------------------------------------------
    # 4. 新增结构特征：corner density / chroma residual
    # ------------------------------------------------------------
    corner_density = _compute_corner_density(gray, cfg.context_k)
    chroma_residual = _compute_chroma_residual(img)

    # ------------------------------------------------------------
    # 5. 边缘点沿法线近似方向采样 profile
    # ------------------------------------------------------------
    ey, ex = np.where(edges > 0)

    if len(ey) == 0:
        empty_score = np.zeros((n_rows, n_cols), dtype=np.float32)
        empty_mask = np.zeros((n_rows, n_cols), dtype=bool)
        return empty_score, img, empty_mask

    gx_e = gx[ey, ex]
    gy_e = gy[ey, ex]

    valid_grad = (np.abs(gx_e) + np.abs(gy_e)) > 1e-6
    ey = ey[valid_grad]
    ex = ex[valid_grad]
    gx_e = gx_e[valid_grad]
    gy_e = gy_e[valid_grad]

    N = len(ey)

    if N == 0:
        empty_score = np.zeros((n_rows, n_cols), dtype=np.float32)
        empty_mask = np.zeros((n_rows, n_cols), dtype=bool)
        return empty_score, img, empty_mask

    # 方向量化为 4 个主方向
    dirs = np.zeros(N, dtype=np.int8)

    agx = np.abs(gx_e)
    agy = np.abs(gy_e)

    is_h = agx > 2.0 * agy
    is_v = agy > 2.0 * agx

    is_d1 = (~is_h) & (~is_v) & (np.sign(gx_e) == np.sign(gy_e))
    is_d2 = (~is_h) & (~is_v) & (np.sign(gx_e) != np.sign(gy_e))

    dirs[is_h] = 0
    dirs[is_v] = 1
    dirs[is_d1] = 2
    dirs[is_d2] = 3

    # 这里的 step_map 是 profile 方向近似
    step_map = [
        (0, 1),    # horizontal profile
        (1, 0),    # vertical profile
        (1, 1),    # diag
        (-1, 1),   # anti-diag
    ]

    t = np.arange(-cfg.profile_radius, cfg.profile_radius + 1, dtype=np.int32)
    L = len(t)

    offsets = np.array(
        [[dy * t, dx * t] for dy, dx in step_map],
        dtype=np.int32,
    )

    sel = offsets[dirs]  # shape: N, 2, L

    sy = np.clip(ey[:, None] + sel[:, 0, :], 0, H - 1)
    sx = np.clip(ex[:, None] + sel[:, 1, :], 0, W - 1)

    # ------------------------------------------------------------
    # 6. profile 特征
    # ------------------------------------------------------------
    p_gray = gray[sy, sx].astype(np.float32)
    p_res = residual[sy, sx].astype(np.float32)
    p_chroma = chroma_residual[sy, sx].astype(np.float32)

    p_smooth = _smooth_1d(p_gray, sigma=1.0)
    p_ring = p_gray - p_smooth
    abs_ring = np.abs(p_ring)

    edge_step = np.max(p_smooth, axis=1) - np.min(p_smooth, axis=1)
    strong = edge_step >= cfg.min_edge_step

    low_th = np.maximum(cfg.low_ratio * edge_step, cfg.min_abs_low_th)
    high_th = np.maximum(cfg.high_ratio * edge_step, low_th + 1e-6)

    # ------------------------------------------------------------
    # 6a. Crossings：使用全局 residual
    # ------------------------------------------------------------
    cross_in = (
        (np.abs(p_res) > low_th[:, None])
        & (np.abs(p_res) < high_th[:, None])
        & strong[:, None]
    )

    signs_cross = np.zeros((N, L), dtype=np.int8)
    signs_cross[cross_in & (p_res > 0)] = 1
    signs_cross[cross_in & (p_res < 0)] = -1

    has_cross = signs_cross != 0
    idx_c = np.where(has_cross, np.arange(L), 0)
    np.maximum.accumulate(idx_c, axis=1, out=idx_c)

    filled_c = signs_cross[np.arange(N)[:, None], idx_c]

    cross = (
        (filled_c[:, :-1] != filled_c[:, 1:])
        & (filled_c[:, :-1] != 0)
        & (filled_c[:, 1:] != 0)
    )

    crossings = np.sum(cross, axis=1).astype(np.float32)

    # ------------------------------------------------------------
    # 6b. Oscillation：使用 profile residual
    # ------------------------------------------------------------
    in_range = (
        (abs_ring > low_th[:, None])
        & (abs_ring < high_th[:, None])
        & strong[:, None]
    )

    signs = np.zeros((N, L), dtype=np.int8)
    signs[in_range & (p_ring > 0)] = 1
    signs[in_range & (p_ring < 0)] = -1

    has_sign = signs != 0
    idx_o = np.where(has_sign, np.arange(L), 0)
    np.maximum.accumulate(idx_o, axis=1, out=idx_o)

    filled_o = signs[np.arange(N)[:, None], idx_o]

    adj = has_sign[:, 1:]

    alt = (
        adj
        & (filled_o[:, :-1] != filled_o[:, 1:])
        & (filled_o[:, :-1] != 0)
        & (filled_o[:, 1:] != 0)
    )

    adj_cnt = np.sum(adj, axis=1).astype(np.float32)

    oscillation = np.divide(
        np.sum(alt, axis=1).astype(np.float32),
        adj_cnt,
        out=np.zeros(N, dtype=np.float32),
        where=adj_cnt > 0,
    )

    # ------------------------------------------------------------
    # 6c. Decay
    # ------------------------------------------------------------
    d = np.abs(t)

    near_mask = d == 1
    mid_mask = (d >= 2) & (d <= 3)
    far_mask = d >= 4

    near_g = np.mean(abs_ring[:, near_mask], axis=1) if np.any(near_mask) else np.zeros(N)
    mid_g = np.mean(abs_ring[:, mid_mask], axis=1) if np.any(mid_mask) else np.zeros(N)
    far_g = np.mean(abs_ring[:, far_mask], axis=1) if np.any(far_mask) else np.zeros(N)

    near_far = near_g / (far_g + 1e-6)

    mono_decay = 0.5 * (near_g > mid_g) + 0.5 * (mid_g > far_g)
    decay = mono_decay * (near_far > 1.05)

    wdist = np.sum(abs_ring * d[None, :], axis=1) / (np.sum(abs_ring, axis=1) + 1e-6)
    compactness = 1.0 / (1.0 + wdist)

    decay_scores = (
        0.5 * decay
        + 0.3 * compactness
        + 0.2 * np.tanh(near_far / 3.0)
    ).astype(np.float32)

    # ------------------------------------------------------------
    # 6d. Flat-side
    # ------------------------------------------------------------
    left_side = t <= -2
    right_side = t >= 2

    l_std = np.std(p_gray[:, left_side], axis=1) if np.any(left_side) else np.zeros(N)
    r_std = np.std(p_gray[:, right_side], axis=1) if np.any(right_side) else np.zeros(N)

    flat_side = np.minimum(l_std, r_std) / (edge_step + 1e-6)

    # ------------------------------------------------------------
    # 6e. Chroma ringing
    # ------------------------------------------------------------
    chroma_energy = np.mean(p_chroma, axis=1)
    chroma_ring_score = chroma_energy / (edge_step + 1e-6)

    # 对极弱边缘，色度比值容易虚高，做一点限制
    chroma_ring_score = chroma_ring_score * (edge_step > cfg.text_min_edge_step).astype(np.float32)
    chroma_ring_score = np.clip(chroma_ring_score, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------
    # 6f. 单边缘 Gibbs 分数
    # ------------------------------------------------------------
    res_energy = np.mean(abs_ring, axis=1)
    norm_e = res_energy / (edge_step + 1e-6)

    amp_ok = (
        (norm_e > cfg.low_ratio)
        & (norm_e < cfg.high_ratio)
        & strong
    )

    gibbs_scores = (
        res_energy
        * oscillation
        * decay_scores
        * amp_ok.astype(np.float32)
    ).astype(np.float32)

    # ------------------------------------------------------------
    # 7. 聚合到 grid
    # ------------------------------------------------------------
    pi_y = np.clip(ey // cfg.stride, 0, n_rows - 1)
    pi_x = np.clip(ex // cfg.stride, 0, n_cols - 1)

    def _g():
        return _grid_zeros(n_rows, n_cols)

    def _accum(vals):
        g = _g()
        np.add.at(g, (pi_y, pi_x), vals.astype(np.float32))
        return g

    g_valid = _g()
    np.add.at(g_valid, (pi_y, pi_x), 1.0)

    c = np.maximum(g_valid, 1.0)

    g_cross = _accum(crossings)
    g_decay = _accum(decay_scores)
    g_near = _accum(near_g.astype(np.float32))
    g_far = _accum(far_g.astype(np.float32))
    g_step = _accum(edge_step.astype(np.float32))
    g_flat = _accum(flat_side.astype(np.float32))
    g_chroma = _accum(chroma_ring_score.astype(np.float32))

    g_mean_cross = g_cross / c
    g_mean_decay = g_decay / c
    g_mean_step = g_step / c
    g_mean_flat = g_flat / c
    g_mean_chroma = g_chroma / c

    g_edge_density = g_valid / float(cfg.patch_size * cfg.patch_size)
    g_near_far_grid = (g_near / c) / (g_far / c + 1e-6)

    # ------------------------------------------------------------
    # 7a. 方向 histogram：用于文字增强和长线抑制
    # ------------------------------------------------------------
    g_dirs = []

    for k_dir in range(4):
        gd = _g()
        m = dirs == k_dir
        np.add.at(gd, (pi_y[m], pi_x[m]), 1.0)
        g_dirs.append(gd)

    g_dir_stack = np.stack(g_dirs, axis=0)
    g_dir_sum = np.maximum(np.sum(g_dir_stack, axis=0), 1.0)

    g_dir_prob = g_dir_stack / g_dir_sum[None, :, :]

    g_max_dir_ratio = np.max(g_dir_prob, axis=0)

    eps = 1e-6
    g_dir_entropy = -np.sum(
        g_dir_prob * np.log(g_dir_prob + eps),
        axis=0,
    ) / np.log(4.0)

    # 兼容原来的 hv score
    g_hv = np.minimum(g_dirs[0], g_dirs[1]) / c

    # ------------------------------------------------------------
    # 7b. grid 级 Gibbs percentile
    # ------------------------------------------------------------
    order = np.lexsort((pi_x, pi_y))

    sg = gibbs_scores[order]
    spy = pi_y[order]
    spx = pi_x[order]

    g_gibbs_p = _g()

    if len(sg) > 0:
        change = (np.diff(spy) != 0) | (np.diff(spx) != 0)
        bounds = np.where(change)[0] + 1

        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [len(sg)]])

        for s, e in zip(starts, ends):
            vals = sg[s:e]
            vals = vals[vals > 0]

            if len(vals):
                g_gibbs_p[spy[s], spx[s]] = np.percentile(vals, cfg.gibbs_percentile)

    # ------------------------------------------------------------
    # 8. 上下文特征采样到 grid
    # ------------------------------------------------------------
    edge_f = (edges > 0).astype(np.float32)

    ctx_cnt = cv2.boxFilter(
        edge_f,
        -1,
        (cfg.context_k, cfg.context_k),
        normalize=False,
    )

    ctx_density = ctx_cnt / float(cfg.context_k * cfg.context_k)

    gy_c = np.arange(n_rows) * cfg.stride + cfg.patch_size // 2
    gx_c = np.arange(n_cols) * cfg.stride + cfg.patch_size // 2

    gy_c = np.clip(gy_c, 0, H - 1)
    gx_c = np.clip(gx_c, 0, W - 1)

    gX, gY = np.meshgrid(gx_c, gy_c)

    g_loc_ratio = loc_ratio[gY, gX]
    g_ctx_density = ctx_density[gY, gX]
    g_non_edge = mean_non_edge[gY, gX]
    g_corner_density = corner_density[gY, gX]

    # ------------------------------------------------------------
    # 9. natural texture reject
    # ------------------------------------------------------------
    texture_reject = (
        (g_edge_density > cfg.max_patch_edge_density)
        | (g_ctx_density > cfg.max_context_edge_density)
        | (g_non_edge > cfg.max_non_edge_residual)
        | ((g_hv > cfg.max_hv_score) & (g_ctx_density > cfg.hv_context_edge_density))
        | (g_near_far_grid < cfg.min_near_far_ratio)
    )

    # ------------------------------------------------------------
    # 10. 新增：text-likeness 和 line-likeness
    # ------------------------------------------------------------
    has_score = g_gibbs_p > 0

    flat_ok = g_mean_flat < cfg.text_max_flat_side_ratio
    corner_ok = g_corner_density > cfg.text_min_corner_density
    dir_diverse_ok = g_dir_entropy > cfg.text_min_dir_entropy
    chroma_ok = g_mean_chroma > cfg.text_min_chroma_ring

    # 文字结构证据：
    # flat_side 是基础；corner、方向熵、色度任一满足即可加强 text branch。
    text_like = (
        flat_ok
        & (
            corner_ok
            | dir_diverse_ok
            | chroma_ok
        )
    )

    # 小字/彩边 rescue：
    # 一些小字灰度 ringing 分数很弱，但色度彩边明显。
    chroma_text_rescue = (
        flat_ok
        & (g_mean_chroma > cfg.text_min_chroma_ring)
        & (g_gibbs_p > cfg.text_chroma_boost_min_score)
    )

    text_like = text_like | chroma_text_rescue

    line_like = (
        (g_edge_density > cfg.line_min_edge_density)
        & (g_ctx_density > cfg.line_min_context_edge_density)
        & (g_max_dir_ratio > cfg.line_min_max_dir_ratio)
        & (g_dir_entropy < cfg.line_max_dir_entropy)
        & (g_corner_density < cfg.line_max_corner_density)
    )

    # 有些字幕或中文笔画本身局部也可能方向一致，允许被 corner/entropy rescue
    line_text_rescue = (
        (g_corner_density > cfg.line_text_rescue_corner_density)
        | (g_dir_entropy > cfg.line_text_rescue_dir_entropy)
    )

    effective_line_like = line_like & (~line_text_rescue)

    # ------------------------------------------------------------
    # 11. Two-branch 决策
    # ------------------------------------------------------------

    # Branch A: natural
    natural = (
        has_score
        & (g_loc_ratio > cfg.natural_min_loc_ratio)
        & (g_mean_cross >= cfg.natural_min_crossings)
        & (g_mean_decay >= cfg.natural_min_decay)
        & (g_valid >= cfg.natural_min_valid_count)
        & (g_mean_step > cfg.natural_min_edge_step)
        & (~texture_reject)
    )

    # Branch B: text / UI / subtitle
    text_ringing = (
        has_score
        & (g_loc_ratio > cfg.text_min_loc_ratio)
        & (g_mean_cross >= cfg.text_min_crossings)
        & (g_mean_decay >= cfg.text_min_decay)
        & (g_valid >= cfg.text_min_valid_count)
        & (g_mean_step > cfg.text_min_edge_step)
    )

    text_ui = text_ringing & text_like

    # text branch mask 后处理：只对文字分支做轻微横向 closing
    if cfg.enable_text_close:
        k_w = max(int(cfg.text_close_kernel_w), 1)
        k_h = max(int(cfg.text_close_kernel_h), 1)

        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))

        text_ui = cv2.morphologyEx(
            text_ui.astype(np.uint8),
            cv2.MORPH_CLOSE,
            close_kernel,
        ).astype(bool)

    # Hard reject：极端纹理兜底
    hard_reject = (
        (g_ctx_density > cfg.hard_context_edge_density)
        & (g_mean_flat > cfg.hard_flat_side_ratio)
        & (~text_like)
    )

    final = (natural | text_ui) & (~hard_reject)

    # ------------------------------------------------------------
    # 12. Long-line soft penalty
    # ------------------------------------------------------------
    line_penalty = np.ones((n_rows, n_cols), dtype=np.float32)
    line_penalty[effective_line_like] = cfg.line_penalty

    score_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
    score_grid[final] = g_gibbs_p[final] * line_penalty[final]

    return score_grid, img, final


# ============================================================
# 可视化
# ============================================================
def save_heatmap(img, score_grid, path="heatmap.png", scale=10.0):
    """保存 score_grid 热力图叠加。"""
    H, W = img.shape[:2]

    u8 = np.clip(score_grid.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    heat = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    heat = cv2.resize(
        heat,
        (W, H),
        interpolation=cv2.INTER_NEAREST,
    )

    overlay = cv2.addWeighted(img, 0.6, heat, 0.4, 0)
    cv2.imwrite(path, overlay)

    return overlay


def save_mask_overlay(img, mask, path="mask_overlay.png", alpha=0.45):
    """保存二值 mask 叠加图，方便检查过检/漏检。"""
    H, W = img.shape[:2]

    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_big = cv2.resize(mask_u8, (W, H), interpolation=cv2.INTER_NEAREST)

    color = np.zeros_like(img)
    color[:, :, 2] = mask_big

    overlay = cv2.addWeighted(img, 1.0, color, alpha, 0)
    cv2.imwrite(path, overlay)

    return overlay


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "test_data/001_OnlineNews#out1#mnr_input0007.bmp"
    path = sys.argv[1] if len(sys.argv) > 1 else "test_data/hisense_mnr_mis_clarity#out1#mnr_input0002.bmp"
    path = sys.argv[1] if len(sys.argv) > 1 else "test_data/05.02.25#out1#mnr_input0012.bmp"
    path = sys.argv[1] if len(sys.argv) > 1 else "test_data/004_pal_un_ds_hvdefinition#out1#mnr_input0007.bmp"

    cfg = GibbsConfig()

    score, img, mask = detect_gibbs(path, cfg)

    save_heatmap(
        img,
        score,
        path="gibbs_textline_v2_heatmap.png",
        scale=12.0,
    )

    save_mask_overlay(
        img,
        mask,
        path="gibbs_textline_v2_mask.png",
    )

    n_det = int(np.sum(mask))
    print(f"检测完成: {n_det} 个 patch 有 Gibbs 伪影 ({path})")