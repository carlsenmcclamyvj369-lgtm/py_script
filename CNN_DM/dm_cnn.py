import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset

# =========================
# 1. 只使用这16个特征
# =========================
features_list = [
    'mean_var',              # 1
    'low_var_count',         # 2
    'high_var_count',        # 3
    'edge_strength',         # 4
    'edge_orientation_conf', # 5
    'second_diff_max',       # 6
    'second_diff_min_max',   # 7
    'ringing_mean_max',      # 8
    'ringing_mean_min',      # 9
    'ringing_mean_min_max',  # 10
    'row_ringing_max',       # 11
    'row_ringing_mean',      # 12
    'col_ringing_max',       # 13
    'col_ringing_mean',      # 14
    'row_diff_max',          # 15
    'col_diff_max',          # 16
]

# =========================
# 2. 固定归一化到 [0, 1]
#    根据 compute_labeled_features.py/markdown
# =========================
NORM_DIV = {
    'mean_var': 1020.0,             # 1
    'low_var_count': 64.0,          # 2
    'high_var_count': 64.0,         # 3
    'edge_strength': 255.0,         # 4
    'edge_orientation_conf': 1.0,   # 5
    'second_diff_max': 510.0,       # 6
    'second_diff_min_max': 1.0,     # 7
    'ringing_mean_max': 1.0,        # 8
    'ringing_mean_min': 1.0,        # 9
    'ringing_mean_min_max': 1.0,    # 10
    'row_ringing_max': 1.0,         # 11
    'row_ringing_mean': 1.0,        # 12
    'col_ringing_max': 1.0,         # 13
    'col_ringing_mean': 1.0,        # 14
    # row/col diff 是亮度差，按 255 归一化
    'row_diff_max': 255.0,          # 15
    'col_diff_max': 255.0,          # 16
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
        x = normalize_features(df, features_list)
        num_rows = x.shape[0]
        num_patches = num_rows // self.patch_area
        if num_patches == 0:
            raise ValueError(
                f"CSV rows={num_rows}, not enough for one {patch_size}x{patch_size} patch"
            )
        x = x[:num_patches * self.patch_area]
        x = x.reshape(num_patches, self.patch_area, len(features_list))
        x = x.reshape(num_patches, patch_size, patch_size, len(features_list))
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
#    用 cost_down=True 去掉 BN + sigmoid → ReLU+Clip
# =========================
class MosquitoDenoiseCNN(nn.Module):
    def __init__(self, cost_down=False):
        super(MosquitoDenoiseCNN, self).__init__()
        self.cost_down = cost_down
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(64, 16, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(16, 1, kernel_size=3, padding=0)
        if not cost_down:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(16)

    def forward(self, x):
        if self.cost_down:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = self.conv4(x)
            x = x.view(x.size(0), -1)
            x = torch.clip(torch.relu(x), 0, 1)
        else:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.conv4(x)
            x = x.view(x.size(0), -1)
            x = torch.sigmoid(x)
        return x

# =========================
# 5. 以下训练代码仅在直接运行时执行
# =========================
if __name__ == "__main__":
    # --- 开关：True=无BN+ReLU+Clip, False=BN+Sigmoid ---
    COST_DOWN = True

    dm_datasets = [
        MosquitoPatchDataset("C:\code\py\denoise\scripts\CNN_DM\9x9_dm.csv", label=1),
        MosquitoPatchDataset("C:\code\py\denoise\scripts\CNN_DM\9x9_dm_merged.csv", label=1),
        MosquitoPatchDataset("C:\code\py\denoise\scripts\CNN_DM\9x9_dm_SR_x3.csv", label=1),
        MosquitoPatchDataset("C:\code\py\denoise\scripts\CNN_DM\9x9_dm_SR_4k_0707.csv", label=1),
        MosquitoPatchDataset("C:\code\py\denoise\scripts\CNN_DM\9x9_dm_seq_0710.csv", label=1),
    ]
    not_dm_datasets = [
        MosquitoPatchDataset("C:\code\py\denoise\scripts\CNN_DM\9x9_not_dm.csv", label=0),
        MosquitoPatchDataset("C:\code\py\denoise\scripts\CNN_DM\9x9_not_dm_merged.csv", label=0),
        MosquitoPatchDataset("C:\code\py\denoise\scripts\CNN_DM\9x9_not_dm_SR_x3.csv", label=0),
        MosquitoPatchDataset("C:\code\py\denoise\scripts\CNN_DM\9x9_not_dm_SR_x2_0707.csv", label=0),
        MosquitoPatchDataset("C:\code\py\denoise\scripts\CNN_DM\9x9_not_dm_SR_4k_0707.csv", label=0),
        MosquitoPatchDataset("C:\code\py\denoise\scripts\CNN_DM\9x9_not_dm_seq_0710.csv", label=0),
    ]

    dm_dataset = ConcatDataset(dm_datasets)
    not_dm_dataset = ConcatDataset(not_dm_datasets)

    val_ratio = 0.2
    dm_size = len(dm_dataset)
    not_dm_size = len(not_dm_dataset)

    dm_val = int(dm_size * val_ratio)
    not_dm_val = int(not_dm_size * val_ratio)

    dm_indices = np.arange(dm_size)
    not_dm_indices = np.arange(not_dm_size)
    np.random.shuffle(dm_indices)
    np.random.shuffle(not_dm_indices)

    train_idx = list(dm_indices[dm_val:]) + [dm_size + i for i in not_dm_indices[not_dm_val:]]
    val_idx = list(dm_indices[:dm_val]) + [dm_size + i for i in not_dm_indices[:not_dm_val]]

    full_dataset = ConcatDataset([dm_dataset, not_dm_dataset])
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)

    print(f"Train: {len(train_dataset)} patches, Val: {len(val_dataset)} patches")

    # =========================
    # 6. 训练
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MosquitoDenoiseCNN(cost_down=COST_DOWN).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 200
    best_f1 = 0.0

    for epoch in range(epochs):
        # ─── Train ───
        model.train()
        total_loss = 0.0
        total_count = 0

        for x, y in train_loader:
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

        # ─── Validation ───
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                all_preds.append(pred.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        all_preds = np.concatenate(all_preds).flatten()
        all_labels = np.concatenate(all_labels).flatten()
        pred_binary = (all_preds > 0.5).astype(np.int64)

        tp = np.sum((pred_binary == 1) & (all_labels == 1))
        tn = np.sum((pred_binary == 0) & (all_labels == 0))
        fp = np.sum((pred_binary == 1) & (all_labels == 0))
        fn = np.sum((pred_binary == 0) & (all_labels == 1))

        acc = (tp + tn) / (tp + tn + fp + fn + 1e-10)
        prec = tp / (tp + fp + 1e-10)
        rec = tp / (tp + fn + 1e-10)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        cm = np.array([[tn, fp], [fn, tp]])

        print(f"Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.6f}  "
              f"Val Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    TN={cm[0,0]:>5d}  FP={cm[0,1]:>5d}")
        print(f"    FN={cm[1,0]:>5d}  TP={cm[1,1]:>5d}")

        if f1 > best_f1:
            best_f1 = f1
            suffix = "_cost_down" if COST_DOWN else ""
            torch.save(model.state_dict(), f"mosquito_denoise_cnn{suffix}.pth")
            print(f"  >>> Model saved (F1 improved to {f1:.4f})")
