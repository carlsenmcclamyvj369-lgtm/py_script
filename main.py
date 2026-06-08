import numpy as np
import cv2


# =========================================================
# 1. Edge ROI
# =========================================================
def compute_edge_roi(I, edge_th=10, dilate_size=5):
    I = I.astype(np.float32)

    gx = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=3)

    grad = np.sqrt(gx * gx + gy * gy)

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
# 4. Patch Similarity（固定阈值版本）
# =========================================================
def patch_similarity_map(I):
    I = I.astype(np.float32)

    sim = np.zeros_like(I)

    shifts = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for dx, dy in shifts:
        shifted = np.roll(I, shift=(dy, dx), axis=(0, 1))
        sim += (I - shifted) ** 2

    sim /= len(shifts)

    return sim  # ✅ 不归一化，直接用固定阈值


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
    ps = patch_similarity_map(I)
    coh = coherence_map(I)

    # =====================================================
    # ✅ 固定阈值判断（全部局部可用）
    # =====================================================

    # 高频条件（8bit建议）
    hf_mask = (hf > 5).astype(np.float32)

    # multi-scale（小 → 噪声）
    ms_mask = (ms < 0.3).astype(np.float32)

    # patch similarity（大 → 噪声）
    ps_mask = (ps > 20).astype(np.float32)

    # coherence（小 → 无结构）
    coh_mask = (coh < 0.3).astype(np.float32)

    # =====================================================
    # ✅ score（简单组合）
    # =====================================================
    score = (
            0.35 * ms_mask +
            0.35 * ps_mask +
            0.30 * coh_mask
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
    img = cv2.imread("test_data/001_OnlineNews#out1#mnr_input0007.bmp", cv2.IMREAD_GRAYSCALE)

    mask = mosquito_mask(img)

    cv2.imshow("Input", img)
    cv2.imshow("Mosquito Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()