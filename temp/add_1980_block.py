import cv2
import numpy as np


def add_block(img, x1, y1, x2, y2, block_size=8, mode="solid", alpha=0.8):
    """
    在指定区域加 block

    mode:
        solid -> 纯白
        blend -> 和原图融合（更自然）
    """
    out = img.copy()

    for y in range(y1, y2, block_size):
        for x in range(x1, x2, block_size):

            y_end = min(y + block_size, y2)
            x_end = min(x + block_size, x2)

            if mode == "solid":
                out[y:y_end, x:x_end] = 255

            elif mode == "blend":
                white = np.ones_like(out[y:y_end, x:x_end]) * 255
                out[y:y_end, x:x_end] = (
                    alpha * white + (1 - alpha) * out[y:y_end, x:x_end]
                ).astype(np.uint8)

    return out


def draw_debug_boxes(img, regions):
    """画红框用于确认区域"""
    dbg = img.copy()
    for (x1, y1, x2, y2) in regions:
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return dbg


if __name__ == "__main__":
    # >>>>>> 改成你的图片路径 <<<<<<
    img = cv2.imread("05.02.25#out1#mnr_input0012.bmp")

    if img is None:
        raise ValueError("图片读取失败")

    h, w = img.shape[:2]
    print("image size:", w, "x", h)

    # ✅ ====== 你图里的3个红框区域 ======
    # （已经帮你大致对齐“1980”，如果不准微调10~20像素即可）
    regions = [
        # 左竖框（1左边）
        (352, 752, 360, 800),

        (360, 752, 368, 760),

        (352, 800, 432, 808),  # x2: 434 → 432（最近8的倍数，偏差-2）

        # 右竖框
        (424, 752, 432, 800),
    ]
    # ✅ 可视化检查区域（建议先看这个）
    debug_img = draw_debug_boxes(img, regions)
    cv2.imshow("ROI check", debug_img)
    cv2.waitKey(0)

    # ✅ 加 block
    out = img.copy()
    for (x1, y1, x2, y2) in regions:
        out = add_block(
            out,
            x1, y1, x2, y2,
            block_size=8,
            mode="solid",   # 👉 改成 "blend" 可更自然
            alpha=0.8
        )

    # ✅ 保存结果
    cv2.imwrite("05.02.25#out1#mnr_input0012_block.bmp", out)

    cv2.imshow("block result", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("完成：output_blocks.png")