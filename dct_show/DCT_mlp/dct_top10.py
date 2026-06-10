import matplotlib.pyplot as plt
import numpy as np

# 1. 配置你的参数
TOP_10_INDICES = [63, 56, 1, 47, 55, 8, 45, 37, 29, 58]
WEIGHTS = np.array([0.350351, 0.135693, -0.013753, 0.108656, 0.091662,
                    -0.003755, -0.235803, -0.073702, 0.078756, 0.015822], dtype=np.float32)


def visualize_dct_weights():
    # 2. 创建 8x8 全零底板
    weight_map = np.zeros((8, 8), dtype=np.float32)

    # 3. 将 10 个权重填入对应的索引位置
    for idx, weight in zip(TOP_10_INDICES, WEIGHTS):
        r, c = divmod(idx, 8)  # 算出 0-63 对应的行列
        weight_map[r, c] = weight

    # 4. 可视化绘制
    plt.figure(figsize=(8, 6))
    # 使用 'RdBu' 颜色映射：红色代表正权重(人工噪声)，蓝色代表负权重(自然纹理)
    plt.imshow(weight_map, cmap='RdBu', interpolation='nearest', vmin=-0.3, vmax=0.3)
    plt.colorbar(label='Weight Magnitude')

    # 添加网格和数值标注
    for i in range(8):
        for j in range(8):
            val = weight_map[i, j]
            if abs(val) > 0.01:  # 只标注非零权重，防止图太乱
                plt.text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=9)

    plt.title("AI-ISP Learned DCT Weights (Top 10)")
    plt.xticks(range(8));
    plt.yticks(range(8))
    plt.grid(True, which='both', color='white', linestyle='-', linewidth=0.5)
    plt.show()


if __name__ == "__main__":
    visualize_dct_weights()