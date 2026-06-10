import cv2
import numpy as np
import os
import time  # 新增：用于生成时间戳


def manual_roi_annotation(image_path):
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"找不到图片: {image_path}")
        return

    # ==========================================
    # 核心修改：提取输入图像的前缀名
    # 例如: "../../test_data/img001.bmp" -> "img001"
    # ==========================================
    original_filename = os.path.basename(image_path)
    base_name, _ = os.path.splitext(original_filename)

    # 2. 调用 OpenCV 内置的 ROI 选择器
    print("-----------------------------------------")
    print(f"正在处理: {original_filename}")
    print("操作指南：")
    print("1. 用鼠标拖拽画框。")
    print("2. 按【空格键】或【回车键】确认截取。")
    print("3. 按【C】键取消重画。")
    print("-----------------------------------------")

    roi_rect = cv2.selectROI("Select ROI", img, showCrosshair=True, fromCenter=False)
    x, y, w, h = roi_rect

    # 3. 检查用户是否真的画了框
    if w > 0 and h > 0:
        roi_img = img[y:y + h, x:x + w]

        # ====== 可视化展示 ======
        cv2.imshow("Extracted: Mosquito Noise Region", roi_img)

        marked_img = img.copy()
        cv2.rectangle(marked_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        text = "Mosquito Noise (ROI)"
        cv2.putText(marked_img, text, (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Marked Image", marked_img)

        # ====== 保存数据 (修复了命名逻辑) ======
        save_dir = "dataset/mosquito_noise"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 改进命名：原图前缀 + _roi_ + 时分秒
        # 结果示例: img001_roi_143025.bmp
        timestamp = time.strftime("%H%M%S")
        new_filename = f"{base_name}_roi_{timestamp}.bmp"
        save_path = os.path.join(save_dir, new_filename)

        cv2.imwrite(save_path, roi_img)
        print(f"\n✅ 成功！已保存至: {save_path}")

        print("按任意键退出...")
        cv2.waitKey(0)
    else:
        print("\n未选择任何区域，操作取消。")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 你的图片路径
    # IMAGE_PATH = "../../test_data/001_OnlineNews#out1#mnr_input0007.bmp"
    # IMAGE_PATH = "../../test_data/05.02.25#out1#mnr_input0012.bmp"
    # IMAGE_PATH = "../../test_data/00009#out1#mnr_input0007.bmp"
    IMAGE_PATH = "../../test_data/004_pal_un_ds_hvdefinition#out1#mnr_input0007.bmp"
    # IMAGE_PATH = "../../test_data/5.2.74.KPIX#out1#mnr_input0016.bmp"
    IMAGE_PATH = "../../test_data/002_VideoResolution_1#out1#mnr_input0007.bmp"
    IMAGE_PATH = "../../test_data/tcl_480p_img#out1#mnr_input0006.bmp"
    IMAGE_PATH = "../../test_data/MNR_001#out1#mnr_input0016.bmp"
    IMAGE_PATH = "../../test_data/tcl_576p_img#out1#mnr_input0009.bmp"

    manual_roi_annotation(IMAGE_PATH)