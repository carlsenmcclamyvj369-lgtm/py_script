import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# =========================
# 1. 只使用这16个特征
# =========================
features_list = [
    'mean_var',
    'low_var_count',
    'high_var_count',
    'edge_strength',
    'edge_orientation_conf',
    'col_ringing_mean',
    'row_ringing_mean',
    'second_diff_max',
    'second_diff_min_max',
    'profile_ringing_mean',
    'ringing_mean_min',
    'ringing_mean_min_max',
    'row_ringing_max',
    'row_diff_max',
    'col_ringing_max',
    'col_diff_max'
]

# =========================
# 2. 固定归一化到 [0, 1]
#    根据 compute_labeled_features.py/markdown
# =========================
NORM_DIV = {
    'mean_var': 1020.0,
    'low_var_count': 64.0,
    'high_var_count': 64.0,
    'edge_strength': 255.0,
    'edge_orientation_conf': 1.0,
    'col_ringing_mean': 1.0,
    'row_ringing_mean': 1.0,
    'second_diff_max': 510.0,
    'second_diff_min_max': 1.0,
    'profile_ringing_mean': 1.0,
    'ringing_mean_min': 1.0,
    'ringing_mean_min_max': 1.0,
    'row_ringing_max': 1.0,
    'col_ringing_max': 1.0,
    # row/col diff 是亮度差，按 255 归一化
    'row_diff_max': 255.0,
    'col_diff_max': 255.0,
}

def normalize_features(df, features_list):
    x = df[features_list].copy()
    for feat in features_list:
        div = NORM_DIV[feat]
        x[feat] = x[feat].astype(np.float32) / div
    x = np.clip(x.values.astype(np.float32), 0.0, 1.0)
    return x

# =========================
# 3. Dataset
#    每81行 -> 一个 9x9 patch
# =========================
class MosquitoPatchDataset(Dataset):
    def __init__(self, csv_path, label, patch_size=9):
        df = pd.read_csv(csv_path)
        # 兼容你之前写的列名
        rename_map = {
            'Row Diff Max': 'row_diff_max',
            'Col Diff Max': 'col_diff_max'
        }
        df = df.rename(columns=rename_map)
        missing = [c for c in features_list if c not in df.columns]
        if len(missing) > 0:
            raise ValueError(f"Missing feature columns: {missing}")
        self.patch_size = patch_size
        self.patch_area = patch_size * patch_size
        # 只取16个feature，并归一化到[0,1]
        x = normalize_features(df, features_list)
        # 要求每81行组成一个patch
        num_rows = x.shape[0]
        num_patches = num_rows // self.patch_area
        if num_patches == 0:
            raise ValueError(
                f"CSV rows={num_rows}, not enough for one {patch_size}x{patch_size} patch"
            )
        # 丢掉不能整除的尾部
        x = x[:num_patches * self.patch_area]
        # shape: (N, 81, 16)
        x = x.reshape(num_patches, self.patch_area, len(features_list))
        # shape: (N, 9, 9, 16)
        x = x.reshape(num_patches, patch_size, patch_size, len(features_list))
        # PyTorch Conv2d 需要 channel-first:
        # (N, 9, 9, 16) -> (N, 16, 9, 9)
        x = np.transpose(x, (0, 3, 1, 2))
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.full((num_patches, 1), float(label), dtype=torch.float32)
        print(f"{csv_path}: rows={num_rows}, patches={num_patches}, x={self.x.shape}")

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# =========================
# 4. 4层 CNN Model
#    padding 全部为 0
# =========================
class MosquitoDenoiseCNN(nn.Module):
    def __init__(self):
        super(MosquitoDenoiseCNN, self).__init__()
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 16, kernel_size=3, padding=0)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 1, kernel_size=3, padding=0)
        # 兼容可能的 bn4
        self.bn4 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(x)
        return x

# =========================
# 5. 加载数据
# =========================
dm_dataset = MosquitoPatchDataset("9x9_dm.csv", label=1)
not_dm_dataset = MosquitoPatchDataset("9x9_not_dm.csv", label=0)
dataset = ConcatDataset([dm_dataset, not_dm_dataset])
dataloader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0
)

# =========================
# 6. 训练
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MosquitoDenoiseCNN().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 40
best_loss = 100000

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    total_count = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    avg_loss = total_loss / total_count
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")

    # ★★★ 核心逻辑：只有比历史最佳更好时才保存 ★★★
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "mosquito_denoise_cnn.pth")

# =========================
# 7. 简单验证输出
# =========================
model.eval()

with torch.no_grad():
    x, y = next(iter(dataloader))
    x = x.to(device)
    pred = model(x)

    print("Pred:", pred[:10].squeeze().cpu().numpy())
    print("GT:  ", y[:10].squeeze().numpy())

# =========================
# 8. 保存模型
# =========================
# torch.save(model.state_dict(), "mosquito_denoise_cnn.pth")
# print("Model saved to mosquito_denoise_cnn.pth")
