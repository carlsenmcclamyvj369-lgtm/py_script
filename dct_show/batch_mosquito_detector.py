import cv2
import numpy as np
import os
import glob
from scipy.fftpack import dctn


def dct2(block):
    """计算 2D DCT (正交归一化)"""
    return dctn(block, type=2, norm="ortho")


def detect_mosquito_candidates(img_bgr, threshold=80000):
    """
    遍历图像提取振铃/蚊子噪声候选区域掩膜 (已集成极高频验证与宏观拓扑过滤)
    """
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[..., 0].astype(np.float32) - 128.0
    H, W = Y.shape

    raw_mask = np.zeros((H, W), dtype=np.uint8)

    # 预生成 8x8 极高频掩膜 (用于剔除自然平滑衰减的光学边缘，如山脊)
    extreme_hf_mask = np.zeros((8, 8), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            if r + c >= 10:
                extreme_hf_mask[r, c] = 1.0

    # 1. 遍历所有 8x8 宏块 (微观频率特征提取)
    for by in range(0, H - 7, 8):
        for bx in range(0, W - 7, 8):
            block = Y[by:by + 8, bx:bx + 8]
            dct_coef = dct2(block)
            abs_dct = np.abs(dct_coef)

            # 剔除三大低频
            l1_sum = np.sum(abs_dct) - abs_dct[0, 0] - abs_dct[0, 1] - abs_dct[1, 0]
            l2_sum = np.sum(abs_dct ** 2) - abs_dct[0, 0] ** 2 - abs_dct[0, 1] ** 2 - abs_dct[1, 0] ** 2

            # 计算能量集中度与蚊子噪声指数
            energy_concentration = l2_sum / (l1_sum + 1e-6)
            mosquito_index = l1_sum * energy_concentration

            # 统计极小系数
            small_coef_count = np.sum(abs_dct < 0.1)
            if abs_dct[0, 0] < 0.1: small_coef_count -= 1
            if abs_dct[0, 1] < 0.1: small_coef_count -= 1
            if abs_dct[1, 0] < 0.1: small_coef_count -= 1

            # 计算极高频占比 (自然纹理极高频衰减快，人工痕迹溢出多)
            extreme_hf_energy = np.sum(abs_dct * extreme_hf_mask)
            hf_ratio = extreme_hf_energy / (l1_sum + 1e-6)

            # 综合阈值判断
            if mosquito_index > threshold and hf_ratio > 0.03 and small_coef_count >= 1:
                raw_mask[by:by + 8, bx:bx + 8] = 255

    # 2. 宏观拓扑过滤 (剔除树枝交织网)
    # 超大核强制融合
    mega_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
    macro_mask = cv2.dilate(raw_mask, mega_kernel, iterations=1)

    # 连通域计算
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(macro_mask, connectivity=8)

    valid_macro_zone = np.zeros_like(macro_mask)
    img_area = H * W

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        # 抛弃面积占比过大 (成片树枝) 或高度贯穿画面 (如整条电线/高山) 的连通域
        if area > (img_area * 0.05) or height > (H * 0.4):
            continue

            # 保留合法孤岛
        valid_macro_zone[labels == i] = 100

    # 利用合法探照灯过滤微观掩膜
    filtered_raw_mask = cv2.bitwise_and(raw_mask, valid_macro_zone)

    # 3. 局部非对称形态学处理 (精细裁切与扩张)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))

    eroded_mask = cv2.erode(filtered_raw_mask, kernel_erode, iterations=1)
    final_mask = cv2.dilate(eroded_mask, kernel_dilate, iterations=1)

    return raw_mask, final_mask


def process_dataset(input_dir, output_dir, threshold=80000):
    """
    批量处理文件夹中的图片
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    if not image_paths:
        print(f"在 '{input_dir}' 中没有找到图片。请检查路径。")
        return

    print(f"找到 {len(image_paths)} 张图片，开始处理 (阈值 > {threshold})...")

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f"正在处理: {filename}")

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        raw_mask, final_mask = detect_mosquito_candidates(img_bgr, threshold=threshold)

        overlay = img_bgr.copy()
        overlay[final_mask == 255] = [0, 0, 255]

        alpha = 0.5
        vis_result = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)

        mask_bgr = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((img_bgr, vis_result, mask_bgr))

        save_path = os.path.join(output_dir, f"detected_{filename}")
        cv2.imwrite(save_path, combined)

    print(f"处理完成！结果已保存至 '{output_dir}' 文件夹。")


if __name__ == "__main__":
    INPUT_DATASET_DIR = "test_set"
    OUTPUT_RESULT_DIR = "test_results"

    # 注意主函数中的 threshold 必须与指标量级匹配，已修改为 80000
    process_dataset(INPUT_DATASET_DIR, OUTPUT_RESULT_DIR, threshold=80000)