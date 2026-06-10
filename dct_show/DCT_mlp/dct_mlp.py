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


# ==========================================
# 1. 基础函数与网络定义 (极致精简版)
# ==========================================
def dct2(block):
    """计算 2D DCT (正交归一化)"""
    return dctn(block, type=2, norm="ortho")


class DCTClassifier(nn.Module):
    """单层感知机：只有纯粹的 1x64 乘加运算，不包含任何非线性激活函数"""

    def __init__(self):
        super(DCTClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 1)  # 抛弃 Sigmoid，输出 Logits
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 2. 数据集加载与特征工程
# ==========================================
def load_dataset_features(dataset_dir="dataset"):
    """
    遍历 clean 和 mosquito_noise 文件夹，提取 8x8 DCT 特征
    """
    X_list = []
    Y_list = []

    # 定义标签映射：clean 是 0 (树枝/自然)，mosquito_noise 是 1 (文字/人工噪声)
    categories = {'clean': 0, 'mosquito_noise': 1}

    for cat, label in categories.items():
        folder_path = os.path.join(dataset_dir, cat)
        if not os.path.exists(folder_path):
            print(f"警告: 找不到文件夹 {folder_path}")
            continue

        img_paths = glob.glob(os.path.join(folder_path, "*.*"))
        for img_path in img_paths:
            img = cv2.imread(img_path)
            if img is None: continue

            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            Y = ycrcb[..., 0].astype(np.float32) - 128.0
            H, W = Y.shape

            for by in range(0, H - 7, 8):
                for bx in range(0, W - 7, 8):
                    block = Y[by:by + 8, bx:bx + 8]
                    dct_coef = dct2(block)

                    # 特征工程三板斧：绝对值 -> 剔除DC -> log1p -> 展平
                    abs_dct = np.abs(dct_coef)
                    abs_dct[0, 0] = 0.0
                    feat = np.log1p(abs_dct).flatten()

                    X_list.append(feat)
                    Y_list.append([label])

    X_data = np.array(X_list, dtype=np.float32)
    Y_data = np.array(Y_list, dtype=np.float32)
    return X_data, Y_data


# ==========================================
# 3. 核心评估与训练函数
# ==========================================
def train_and_evaluate_mlp(dataset_dir="dataset", model_save_path="dct_mlp_weights.pth", epochs=150):
    print(">>> 正在从数据集中提取 DCT 特征...")
    X_data, Y_data = load_dataset_features(dataset_dir)

    if len(X_data) < 10:
        print("错误: 数据集样本过少！请先使用截图脚本收集更多样本。")
        return None

    print(f"特征提取完毕！共获得 {len(X_data)} 个 8x8 样本。")
    print(f"其中正样本 (文字噪声 1): {np.sum(Y_data == 1)} 个，负样本 (干净树枝 0): {np.sum(Y_data == 0)} 个。")

    # 【核心A】划分 80% 训练集，20% 盲测集
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_data, Y_data, test_size=0.2, random_state=42, stratify=Y_data
    )

    dataset = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = DCTClassifier()
    # 【关键修改】因为没有 Sigmoid，必须使用 BCEWithLogitsLoss
    num_pos = np.sum(Y_train == 1)
    num_neg = np.sum(Y_train == 0)
    weight_ratio = num_neg / (num_pos + 1e-5)
    pos_weight_tensor = torch.tensor([weight_ratio], dtype=torch.float32)

    print(f"\n[策略调整] 检测到类别不平衡，已启用正样本权重加持: {weight_ratio:.2f} 倍惩罚")

    # 将动态计算出的权重传给 Loss 函数
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # 保持较低的学习率，确保平稳收敛
    optimizer = optim.Adam(model.parameters(), lr=0.0005)


    print(f"\n>>> 开始训练单层模型 (训练集: {len(X_train)}，测试集: {len(X_test)})...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 30 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), model_save_path)
    print(f"✅ 训练完成！模型权重已保存至: {model_save_path}")

    # 【核心B】盲测评估报告
    print("\n" + "=" * 50)
    print(">>> 盲测评估报告 (测试集未参与训练):")
    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test, dtype=torch.float32)
        true_labels = Y_test

        logits = model(test_tensor).numpy()
        # 【关键修改】在 Logit 空间，阈值是 0.0 (大于0判定为1，小于0判定为0)
        pred_labels = (logits > 0.0).astype(float)

        print("\n[混淆矩阵] (Confusion Matrix):")
        cm = confusion_matrix(true_labels, pred_labels)
        print(f"真负类(正确放过树枝): {cm[0][0]:<5} | 假正类(误杀树枝): {cm[0][1]}")
        print(f"假负类(漏掉文字噪声): {cm[1][0]:<5} | 真正类(正确抓出噪声): {cm[1][1]}")

        print("\n[详细指标评估] (Classification Report):")
        print(classification_report(true_labels, pred_labels, target_names=['Clean(树枝)', 'Mosquito(文字)']))

    # 【核心C】提取部署参数
    print("=" * 50)
    print(">>> 硬件 C++ 部署参数提取:")
    weights = model.net[0].weight.data.numpy()[0]
    bias = model.net[0].bias.data.numpy()[0]
    print(f"float bias = {bias:.6f}f;")

    # 格式化打印为 C++ 数组格式
    np.set_printoptions(formatter={'float': lambda x: f"{x:.6f}f"})
    weights_str = np.array2string(weights, separator=', ').replace('[', '{').replace(']', '}')
    print(f"float weights[64] = {weights_str};")
    print("=" * 50 + "\n")

    return model


# ==========================================
# 4. 全图推理与 ROI 掩膜输出函数
# ==========================================
def infer_full_image(image_path, model):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"无法读取测试图片: {image_path}")
        return None, None

    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[..., 0].astype(np.float32) - 128.0
    H, W = Y.shape

    roi_mask = np.zeros((H, W), dtype=np.uint8)
    model.eval()

    for by in range(0, H - 7, 8):
        batch_features = []
        x_coords = []

        for bx in range(0, W - 7, 8):
            block = Y[by:by + 8, bx:bx + 8]
            abs_dct = np.abs(dct2(block))

            # 如果平坦区(能量极小)，直接跳过
            if np.sum(abs_dct) - abs_dct[0, 0] < 50:
                continue

            abs_dct[0, 0] = 0.0
            feat = np.log1p(abs_dct).flatten()
            batch_features.append(feat)
            x_coords.append(bx)

        if not batch_features:
            continue

        input_tensor = torch.tensor(np.array(batch_features), dtype=torch.float32)
        with torch.no_grad():
            logits = model(input_tensor).numpy()

        # 【关键修改】推理判定条件改为 Logit > 0.0
        for idx, bx in enumerate(x_coords):
            if logits[idx][0] > 0.0:
                roi_mask[by:by + 8, bx:bx + 8] = 255

    return img_bgr, roi_mask


# ==========================================
# 5. 主执行逻辑
# ==========================================
if __name__ == "__main__":
    MODEL_PATH = "dct_mlp_weights.pth"

    # 1. 训练与盲测评估
    trained_model = train_and_evaluate_mlp(dataset_dir="dataset", model_save_path=MODEL_PATH, epochs=150)

    # 2. 对外部原图进行全图推理
    if trained_model is not None:
        # 定位到你 test_data 下的图片
        test_img_path = "../../test_data/001_OnlineNews#out1#mnr_input0007.png"

        print(f">>> 正在推理图片: {test_img_path}")
        img_bgr, mask = infer_full_image(test_img_path, trained_model)

        if img_bgr is not None:
            # 轻微形态学闭运算填充内部孔洞
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            smoothed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            overlay = img_bgr.copy()
            overlay[smoothed_mask == 255] = [0, 0, 255]
            vis_result = cv2.addWeighted(overlay, 0.5, img_bgr, 0.5, 0)

            mask_bgr = cv2.cvtColor(smoothed_mask, cv2.COLOR_GRAY2BGR)
            combined = np.hstack((img_bgr, vis_result, mask_bgr))

            # 自适应窗口大小显示，防止图片太大超出屏幕
            cv2.namedWindow("MLP ROI Prediction", cv2.WINDOW_NORMAL)
            cv2.imshow("MLP ROI Prediction", combined)
            print("按任意键退出...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()