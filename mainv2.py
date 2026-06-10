import numpy as np
import cv2


# =========================================================
# 1. Edge ROI
# =========================================================
def compute_edge_roi(I, edge_th=200, dilate_size=5):
    I = I.astype(np.float32)
    # print(I[])
    # print(np.max(I))

    gx = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=3)
    # print("max:{}, min:{}, mean:{}" .format(np.max(abs(gx)), np.min(abs(gx)), np.mean(abs(gx))))
    # print("max:{}, min:{}, mean:{}" .format(np.max(abs(gy)), np.min(abs(gy)), np.mean(abs(gy))))

    grad = np.sqrt(gx * gx + gy * gy)
    # print(grad)
    edge = grad > edge_th

    kernel = np.ones((dilate_size, dilate_size), np.uint8)
    roi = cv2.dilate(edge.astype(np.uint8), kernel)

    return roi.astype(np.float32)


# =========================================================
# 2. High Frequency（I - blur）
# =========================================================
def high_freq_map(I):
    I = I.astype(np.float32)

    blur = cv2.GaussianBlur(I, (5, 5), 1)
    hf = np.abs(I - blur)

    return hf  # ✅ 不做全局归一


# =========================================================
# 3. Multi-scale Consistency
# =========================================================
def multi_scale_consistency(I):
    I = I.astype(np.float32)

    blur1 = cv2.GaussianBlur(I, (5, 5), 1)
    hf_fine = np.abs(I - blur1)

    I_coarse = cv2.GaussianBlur(I, (9, 9), 2)
    blur2 = cv2.GaussianBlur(I_coarse, (5, 5), 1)
    hf_coarse = np.abs(I_coarse - blur2)

    ratio = hf_coarse / (hf_fine + 1e-6)

    return ratio  # ✅ 保持原始比例


# =========================================================
# 4. directional_entropy_map（固定阈值版本）
# =========================================================
def directional_entropy_map(I):
    """
        计算 4 方向的局部梯度方向熵。
        人工文字/噪声：方向极其单一，相干性高（低熵值）；
        自然纹理（树枝/草地）：方向四面八方都有，极度混乱（高熵值）。
        """
    win_size = 9
    edge_gate = 10.0
    I = I.astype(np.float32)

    # 步一：一阶差分计算梯度
    dx = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=3)

    abs_dx = np.abs(dx)
    abs_dy = np.abs(dy)
    mag = abs_dx + abs_dy  # 硬件级幅值平替

    # 步二：用区域判决平替 atan2，直接分出 4 个方向区间 (Bin)
    # 只有梯度大于 edge_gate 的强结构点才参与方向统计，规避平坦背景噪点
    mask_h = (abs_dx > 2.0 * abs_dy) & (mag > edge_gate)  # 水平
    mask_v = (abs_dy > 2.0 * abs_dx) & (mag > edge_gate)  # 垂直
    mask_d45 = (dx * dy > 0) & (~mask_h) & (~mask_v) & (mag > edge_gate)  # 45度
    mask_d135 = (dx * dy < 0) & (~mask_h) & (~mask_v) & (mag > edge_gate)  # 135度

    # 步三：统计局部窗口（用盒式滤波 boxFilter 快速实现窗口内的点数统计）
    # 分别统计窗口内各个方向点的"概率分布"
    c_h = cv2.boxFilter(mask_h.astype(np.float32), -1, (win_size, win_size))
    c_v = cv2.boxFilter(mask_v.astype(np.float32), -1, (win_size, win_size))
    c_d45 = cv2.boxFilter(mask_d45.astype(np.float32), -1, (win_size, win_size))
    c_d135 = cv2.boxFilter(mask_d135.astype(np.float32), -1, (win_size, win_size))

    total = c_h + c_v + c_d45 + c_d135 + 1e-6

    p_h = c_h / total
    p_v = c_v / total
    p_d45 = c_d45 / total
    p_d135 = c_d135 / total

    # 步四：计算香农方向熵 H
    # H = - sum(p * log2(p))
    entropy = -(
            0.5 * p_h * np.log2(p_h + 1e-5) +
            0.5 * p_v * np.log2(p_v + 1e-5) +
            p_d45 * np.log2(p_d45 + 1e-5) +
            p_d135 * np.log2(p_d135 + 1e-5))

    return entropy  # ✅ 不归一化，直接用固定阈值


# =========================================================
# 5. Coherence（局部窗口）
# =========================================================
def coherence_map(I, win_size=5):
    I = I.astype(np.float32)

    Ix = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=3)

    Jxx = cv2.boxFilter(Ix * Ix, -1, (win_size, win_size))
    Jyy = cv2.boxFilter(Iy * Iy, -1, (win_size, win_size))
    Jxy = cv2.boxFilter(Ix * Iy, -1, (win_size, win_size))

    trace = Jxx + Jyy
    det = Jxx * Jyy - Jxy * Jxy

    temp = np.sqrt(np.maximum(trace * trace - 4 * det, 0))

    l1 = (trace + temp) / 2
    l2 = (trace - temp) / 2

    coh = (l1 - l2) / (l1 + l2 + 1e-6)

    return coh


# =========================================================
# 6. Mosquito Score（固定阈值逻辑）
# =========================================================
def mosquito_score(I):
    I = I.astype(np.float32)

    # --- feature ---
    roi = compute_edge_roi(I)
    hf = high_freq_map(I)
    ms = multi_scale_consistency(I)
    dir = directional_entropy_map(I)
    coh = coherence_map(I)

    # =====================================================
    # ✅ 固定阈值判断（全部局部可用）
    # =====================================================

    # 高频条件（8bit建议）
    # hf_mask = (hf > 5).astype(np.float32)
    hf_mask = (hf > 10).astype(np.float32)

    # multi-scale（小 → 噪声）
    ms_mask = (ms < 0.3).astype(np.float32)

    # patch similarity（大 → 噪声）
    # dir_mask = ((dir < 2.0) & (dir > 0.05)).astype(np.float32)
    dir_mask = ((dir < 1.9) & (dir > 0.1)).astype(np.float32)
    # dir_mask = (dir < 1.9).astype(np.float32)

    # coherence（小 → 无结构）
    coh_mask = (coh < 0.3).astype(np.float32)

    # =====================================================
    # ✅ score（简单组合）
    # =====================================================
    score = (
            0.0 * ms_mask +
            0.9 * dir_mask +
            0.1 * coh_mask
    )

    # gating
    score = score * roi * hf_mask

    return score


# =========================================================
# 7. Mosquito Mask（最终输出）
# =========================================================
def mosquito_mask(I, th=0.5):
    score = mosquito_score(I)

    mask = (score > th).astype(np.float32)

    # ✅ 空间平滑（避免碎点）
    mask = cv2.GaussianBlur(mask, (7, 7), 2)

    return mask


# =========================================================
# 8. Example
# =========================================================
if __name__ == "__main__":
    # img = cv2.imread("test_data/001_OnlineNews#out1#mnr_input0007.bmp", cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread("test_data/05.02.25#out1#mnr_input0012.bmp", cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread("test_data/05.02.25#out1#mnr_input0012.bmp", cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread("test_data/hisense_mnr_mis_clarity#out1#mnr_input0012.bmp", cv2.IMREAD_GRAYSCALE)
    img = cv2.imread("test_data/hisense_mnr_mis_clarity#out1#mnr_input0002.bmp", cv2.IMREAD_GRAYSCALE)

    mask = mosquito_mask(img)

    cv2.imshow("Input", img)
    cv2.imshow("Mosquito Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
