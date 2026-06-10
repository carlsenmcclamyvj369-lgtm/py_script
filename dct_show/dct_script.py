import cv2
import numpy as np
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# ================================
from scipy.fftpack import dctn
import sys


def dct2(block):
    """计算 2D DCT (正交归一化)"""
    return dctn(block, type=2, norm="ortho")


class InteractiveDCTViewer:
    def __init__(self, image_path):
        # 1. 读取图像
        self.img_bgr = cv2.imread(image_path)
        if self.img_bgr is None:
            print(f"Error: 找不到图像 '{image_path}'")
            sys.exit(1)

        # 2. 转换为 YCrCb 空间，并提取 Y 通道
        self.ycrcb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2YCrCb)

        # 遵循 JPEG 标准：将亮度值从 [0, 255] 平移到 [-128, 127]，消除直流(DC)偏置
        self.Y = self.ycrcb[..., 0].astype(np.float32) - 128.0
        self.H, self.W = self.Y.shape

        # 3. 设置 UI (Matplotlib 双子图)
        self.fig, (self.ax_img, self.ax_dct) = plt.subplots(1, 2, figsize=(14, 7))
        self.fig.canvas.manager.set_window_title('Interactive 8x8 DCT Viewer')

        # 左图：显示原图
        self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
        self.ax_img.imshow(self.img_rgb)
        self.ax_img.set_title("Click to select an 8x8 block (Snaps to grid)")
        self.ax_img.axis('off')

        # 红色高亮框，用于指示当前选中的 8x8 块
        self.rect = plt.Rectangle((0, 0), 8, 8, edgecolor='red', facecolor='none', lw=2)
        self.ax_img.add_patch(self.rect)

        # 右图：初始化 DCT 显示区
        self.ax_dct.set_title("8x8 DCT Coefficients")
        self.ax_dct.axis('off')

        # 4. 绑定鼠标点击事件
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        plt.tight_layout()
        plt.show()

    def onclick(self, event):
        # 增加打印，用来在终端确认鼠标是否被识别
        print(f"鼠标点击: x={event.xdata}, y={event.ydata}, 在图像区: {event.inaxes == self.ax_img}")

        if event.inaxes != self.ax_img:
            return

        x, y = int(event.xdata), int(event.ydata)

        # 强制对齐到 JPEG 的 8x8 宏块网格
        bx = (x // 8) * 8
        by = (y // 8) * 8

        # 边界检查
        if bx < 0 or by < 0 or bx + 8 > self.W or by + 8 > self.H:
            return

        # 更新左图中的红色高亮框位置
        self.rect.set_xy((bx, by))

        # 提取 8x8 亮度块并计算 DCT
        block = self.Y[by:by + 8, bx:bx + 8]
        dct_coef = dct2(block)

        # ==========================================
        # 新增核心逻辑：L1 范数统计与均值计算
        # ==========================================
        abs_dct = np.abs(dct_coef)

        # 1. 剔除 第一行第一个[0,0], 第一行第二个[0,1], 第二行第一个[1,0]
        l1_sum = np.sum(abs_dct) - abs_dct[0, 0] - abs_dct[0, 1] - abs_dct[1, 0]

        # 2. 计算高频系数的绝对值均值 (除以剩余的 61 个数)
        mean_ac = l1_sum / 61.0

        # 3. 计算亮度归一化比值 (除以该块的像素亮度的均值)
        # 注意：因为初始化时 Y 通道减去了 128，这里要加回来恢复真实亮度
        block_mean = np.mean(block) + 128.0
        norm_by_mean = l1_sum / (block_mean + 1e-6)  # 加 1e-6 防止纯黑块除零报错
        # ==========================================

        small_coef_count = np.sum(abs_dct < 0.05)
        # 刨除前面剔除掉的三个低频成分（如果它们碰巧也小于 0.1，需要从总数里扣除）
        if abs_dct[0, 0] < 0.1: small_coef_count -= 1
        if abs_dct[0, 1] < 0.1: small_coef_count -= 1
        if abs_dct[1, 0] < 0.1: small_coef_count -= 1

        # 3. 将 L1_sum 乘以小于 0.1 的个数
        weighted_l1 = l1_sum * small_coef_count


        # 更新右侧 DCT 可视化
        self.ax_dct.clear()

        # 【关键修改】将 L1 统计量和均值显示在图表正上方的标题中
        title_str = (f"DCT @ (x={bx}, y={by})\n"
                     f"L1 Sum(excl. DC/Low): {l1_sum:.1f} | AC Mean: {mean_ac:.2f} | L1/Brightness: {norm_by_mean:.2f} | weighted_l1: {weighted_l1:.2f}")
        self.ax_dct.set_title(title_str, fontsize=10, fontweight='bold', color='darkblue')

        # 绘制热力图 (使用对数尺度以便同时观察 DC 和微弱的 AC 高频分量)
        heatmap_data = np.log10(np.abs(dct_coef) + 1e-5)
        self.ax_dct.imshow(heatmap_data, cmap='magma')

        # 在每个网格中心打印具体的 DCT 系数数值
        for i in range(8):
            for j in range(8):
                val = dct_coef[i, j]
                text_color = 'black' if heatmap_data[i, j] > np.max(heatmap_data) * 0.8 else 'white'
                self.ax_dct.text(j, i, f"{val:.1f}", ha='center', va='center',
                                 color=text_color, fontsize=9, fontweight='bold')

        # 隐藏坐标轴刻度，但保留网格线感觉
        self.ax_dct.set_xticks(np.arange(-.5, 8, 1))
        self.ax_dct.set_yticks(np.arange(-.5, 8, 1))
        self.ax_dct.set_xticklabels([])
        self.ax_dct.set_yticklabels([])
        self.ax_dct.grid(color='white', linestyle='-', linewidth=1)

        # 刷新画布
        self.fig.canvas.draw()


if __name__ == "__main__":
    # 将这里替换为你想要分析的图像路径
    # IMAGE_PATH = "./test_data/test_image.jpg"
    IMAGE_PATH = "img2.bmp"
    # IMAGE_PATH = "img1.png"
    # IMAGE_PATH = "img3.bmp"
    # IMAGE_PATH = "img.png"
    # IMAGE_PATH = "img.png"

    try:
        viewer = InteractiveDCTViewer(IMAGE_PATH)
    except Exception as e:
        print(e)