import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.fftpack import dctn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import time

# ================= 配置与参数 =================
TOP_10_INDICES = [63, 56, 1, 47, 55, 8, 45, 37, 29, 58]
BIAS = -0.443728
WEIGHTS = np.array(
    [0.350351, 0.135693, -0.013753, 0.108656, 0.091662, -0.003755, -0.235803, -0.073702, 0.078756, 0.015822],
    dtype=np.float32)


def dct2(block): return dctn(block, type=2, norm="ortho")


class DCTClassifier(nn.Module):
    def __init__(self):
        super(DCTClassifier, self).__init__()
        self.net = nn.Sequential(nn.Linear(10, 1))

    def forward(self, x): return self.net(x)


# ================= 功能模块 =================

def manual_roi_annotation(image_path):
    img = cv2.imread(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    roi_rect = cv2.selectROI("Select ROI", img, showCrosshair=True, fromCenter=False)
    x, y, w, h = roi_rect
    if w > 0 and h > 0:
        roi_img = img[y:y + h, x:x + w]
        save_dir = "dataset/mosquito_noise"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{base_name}_roi_{time.strftime('%H%M%S')}.bmp")
        cv2.imwrite(save_path, roi_img)
        print(f"✅ 样本已保存: {save_path}")
    cv2.destroyAllWindows()


def load_dataset_features(dataset_dir="dataset"):
    X_list, Y_list = [], []
    for cat, label in {'clean': 0, 'mosquito_noise': 1}.items():
        for img_path in glob.glob(os.path.join(dataset_dir, cat, "*.*")):
            img = cv2.imread(img_path)
            if img is None: continue
            Y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32) - 128.0
            for by in range(0, Y.shape[0] - 7, 8):
                for bx in range(0, Y.shape[1] - 7, 8):
                    abs_dct = np.abs(dct2(Y[by:by + 8, bx:bx + 8]))
                    abs_dct[0, 0] = 0.0
                    feat_10 = np.log1p(abs_dct).flatten()[TOP_10_INDICES]
                    X_list.append(feat_10);
                    Y_list.append([label])
    return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32)


def train_and_eval(dataset_dir="dataset", model_save_path="dct_mlp_top10.pth"):
    X, Y = load_dataset_features(dataset_dir)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    model = DCTClassifier()
    pos_weight = torch.tensor([len(Y_train[Y_train == 0]) / (len(Y_train[Y_train == 1]) + 1e-5)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(Y_train)), batch_size=64, shuffle=True)
    for epoch in range(100):
        for bx, by in loader:
            optimizer.zero_grad();
            loss = criterion(model(bx), by);
            loss.backward();
            optimizer.step()

    torch.save(model.state_dict(), model_save_path)
    print("✅ 模型训练完成")
    return model


def infer_and_save_mask(img_path, model, output_dir="test_results"):
    img = cv2.imread(img_path)
    Y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32) - 128.0
    mask = np.zeros(Y.shape, dtype=np.uint8)

    for by in range(0, Y.shape[0] - 7, 8):
        for bx in range(0, Y.shape[1] - 7, 8):
            abs_dct = np.abs(dct2(Y[by:by + 8, bx:bx + 8]))
            if np.sum(abs_dct) - abs_dct[0, 0] < 50: continue
            abs_dct[0, 0] = 0.0
            feat_10 = np.log1p(abs_dct).flatten()[TOP_10_INDICES]

            # 使用 NumPy 加速推理 (替换 PyTorch 模型推理)
            if np.dot(feat_10, WEIGHTS) + BIAS > 0.0:
                mask[by:by + 8, bx:bx + 8] = 255

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, os.path.basename(img_path).split('.')[0] + "_mask.png")
    cv2.imwrite(save_path, mask)
    print(f"📁 Mask 已保存: {save_path}")


# ================= 主流程入口 =================
if __name__ == "__main__":
    # 模式选择: "annotate" (截取), "train" (训练), "infer" (批量推理)
    MODE = "infer"
    # 定义测试文件夹路径
    TEST_DATA_DIR = "../../test_data"

    if MODE == "annotate":
        # 如果是标注模式，依然只处理单张图片
        IMG_PATH = os.path.join(TEST_DATA_DIR, "001_OnlineNews#out1#mnr_input0007.bmp")
        manual_roi_annotation(IMG_PATH)

    elif MODE == "train":
        train_and_eval()

    elif MODE == "infer":
        # 获取该目录下所有的 bmp 图片
        image_files = glob.glob(os.path.join(TEST_DATA_DIR, "*.bmp"))

        if not image_files:
            print(f"⚠️ 在 {TEST_DATA_DIR} 下没找到 bmp 文件，请检查路径。")
        else:
            print(f"🚀 开始批量推理，共找到 {len(image_files)} 张图片...")
            for img_file in image_files:
                print(f"--- 正在处理: {os.path.basename(img_file)} ---")
                try:
                    infer_and_save_mask(img_file, None)
                except Exception as e:
                    print(f"❌ 处理失败 {img_file}: {e}")
            print("\n✨ 全部批量任务处理完毕！")