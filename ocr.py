import cv2
import numpy as np


def detect_text_extreme(image_path, output_path="output_v4.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图片！")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. 提取垂直边缘 (Sobel X)
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)

    # 2. 二值化 (Otsu)
    _, binary = cv2.threshold(abs_grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. 强力横向涂抹 (将宽度增加到 35，高度降低到 4 减少垂直乱粘连)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 4))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 4. 寻找连通区域
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. 极限宽松的过滤规则
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        rect_area = w * h
        contour_area = cv2.contourArea(contour)

        if h == 0 or rect_area == 0:
            continue

        aspect_ratio = w / float(h)
        solidity = contour_area / float(rect_area)

        # 【V4 极限版规则：只设极低门槛，不设任何上限】
        is_valid_height = h > 5  # 只要高度大于5个像素（防单像素噪点）
        is_valid_ratio = aspect_ratio > 0.2  # 允许任何极长或微扁的图形
        is_valid_area = rect_area > 30  # 只要面积大于30个像素
        is_solid = solidity > 0.08  # 饱满度降到极低，只要有一点笔画挨着就认

        if is_valid_height and is_valid_ratio and is_valid_area and is_solid:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imwrite(output_path, img)
    print(f"处理完成，结果已保存至 {output_path}")

# if __name__ == "__main__":
#     detect_text_broadcast("your_news

if __name__ == "__main__":
    detect_text_extreme("./test_data/hisense_mnr_mis_clarity#out1#mnr_input0002.bmp", "text_detected_0.jpg")# ==========================================
    detect_text_extreme("./test_data/004_pal_un_ds_hvdefinition#out1#mnr_input0007.bmp", "text_detected_1.jpg")# ==========================================
    detect_text_extreme("./test_data/001_OnlineNews#out1#mnr_input0007.bmp", "text_detected_2.jpg")# ==========================================
# 运行示例
