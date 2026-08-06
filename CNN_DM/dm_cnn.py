"""
CNN mosquito noise detection — training script.

用法:
  python dm_cnn.py                          # 按 train() 默认参数跑 (GS=16, cost_down, 20 epochs)
  python -c "from dm_cnn import *; train(gs=8, cost_down=True, epochs=400)"   # GS=8 训练

导入使用:
  from dm_cnn import MosquitoDenoiseCNN, MosquitoPatchDataset, make_norm_div

  model = MosquitoDenoiseCNN(cost_down=True)
  dataset = MosquitoPatchDataset("data.csv", label=1, gs=8)

训练两个 GS 依次跑:
  train(gs=8,  cost_down=True, epochs=400)
  train(gs=16, cost_down=True, epochs=400)

数据源 (自动选择最新的 MNR_Label_GS8_* 根目录) 分源划分:
  SR_data / JPG_data / DIV2K: 图像级 8:2 划分 (同一图像的整体 Patch 进同一侧)
  test_data:                   patch 级 8:2 混洗划分
  Train = 图像源 80% + test_data 80%,  Val = 图像源 20% + test_data 20%

输出: model/mosquito_denoise_cnn_cost_down_grid_{GS}.pth
       model/best_th_cost_down_grid_{GS}.npy
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import os
import random

# =========================
# 1. 只使用这16个特征
# =========================
features_list = [
    'mean_var',  # 1
    'low_var_count',  # 2
    'high_var_count',  # 3
    'edge_strength',  # 4
    'edge_orientation_conf',  # 5
    'second_diff_max',  # 6
    'second_diff_min_max',  # 7
    'ringing_mean_max',  # 8
    'ringing_mean_min',  # 9
    'ringing_mean_min_max',  # 10
    'row_ringing_max',  # 11
    'row_ringing_mean',  # 12
    'col_ringing_max',  # 13
    'col_ringing_mean',  # 14
    'row_diff_max',  # 15
    'col_diff_max',  # 16
]

# =========================
# 2. 固定归一化到 [0, 1]
#    根据 compute_labeled_features.py/markdown
# =========================
def make_norm_div(gs):
    """根据 block size gs 构造归一化除数 dict。"""
    return {
        'mean_var': 1020.0,  # 1
        'low_var_count': 64.0 * (gs // 8) * (gs // 8),  # 2 — 每个 block 的像素数
        'high_var_count': 64.0 * (gs // 8) * (gs // 8),  # 3 — 同上
        'edge_strength': 255.0,  # 4
        'edge_orientation_conf': 1.0,  # 5
        'second_diff_max': 510.0,  # 6
        'second_diff_min_max': 1.0,  # 7
        'ringing_mean_max': 1.0,  # 8
        'ringing_mean_min': 1.0,  # 9
        'ringing_mean_min_max': 1.0,  # 10
        'row_ringing_max': 1.0,  # 11
        'row_ringing_mean': 1.0,  # 12
        'col_ringing_max': 1.0,  # 13
        'col_ringing_mean': 1.0,  # 14
        # row/col diff 是亮度差，按 255 归一化
        'row_diff_max': 255.0,  # 15
        'col_diff_max': 255.0,  # 16
    }


NORM_DIV = make_norm_div(8)


def normalize_features(df, features_list, gs=8):
    x = df[features_list].copy()
    for feat in features_list:
        x[feat] = np.clip(x[feat].astype(np.float32), 0, 255) / 255
    if x.isna().any().any():
        nan_cols = [c for c in features_list if x[c].isna().any()]
        raise ValueError(f"CSV contains NaN in columns: {nan_cols}")
    return x.values.astype(np.float32)


# =========================
# 3. Dataset
#    每81行 -> 一个 9x9 patch
# =========================
class MosquitoPatchDataset(Dataset):
    def __init__(self, csv_path, label, patch_size=9, gs=8):
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
        x = normalize_features(df, features_list, gs=gs)
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
    def __init__(self, cost_down=True):
        super(MosquitoDenoiseCNN, self).__init__()
        self.cost_down = cost_down
        self.debug = False
        if cost_down:
            self.conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=0)
            self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=0)
            self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=0)
            self.conv4 = nn.Conv2d(16, 1, kernel_size=3, padding=0)
            self._init_weights()
        else:
            self.conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=0)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
            self.conv3 = nn.Conv2d(64, 16, kernel_size=3, padding=0)
            self.conv4 = nn.Conv2d(16, 1, kernel_size=3, padding=0)
            # self.bn1 = nn.BatchNorm2d(32)
            # self.bn2 = nn.BatchNorm2d(64)
            # self.bn3 = nn.BatchNorm2d(16)
            self._init_weights()

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.cost_down:
            x = self.relu1(self.conv1(x))
            x = torch.clamp(x, 0, 3)
            if self.debug: print(f"  after conv1+relu1: min={x.min().item():.4f} max={x.max().item():.4f} mean={x.mean().item():.4f}")
            x = self.relu2(self.conv2(x))
            x = torch.clamp(x, 0, 3)
            if self.debug: print(f"  after conv2+relu2: min={x.min().item():.4f} max={x.max().item():.4f} mean={x.mean().item():.4f}")
            x = self.relu3(self.conv3(x))
            x = torch.clamp(x, 0, 3)
            if self.debug: print(f"  after conv3+relu3: min={x.min().item():.4f} max={x.max().item():.4f} mean={x.mean().item():.4f}")
            x = self.conv4(x)
            x = torch.clamp(x, -10, 10)
            if self.debug: print(f"  after conv4 (pre-sigmoid): min={x.min().item():.4f} max={x.max().item():.4f} mean={x.mean().item():.4f}")
            x = x.view(x.size(0), -1)
            x = torch.sigmoid(x)
            if self.debug: print(f"  after sigmoid: min={x.min().item():.4f} max={x.max().item():.4f} mean={x.mean().item():.4f}")

        else:
            x = self.relu1(self.conv1(x))
            x = torch.clamp(x, 0, 3)
            x = self.relu2(self.conv2(x))
            x = torch.clamp(x, 0, 3)
            x = self.relu3(self.conv3(x))
            x = torch.clamp(x, 0, 3)
            x = self.conv4(x)
            x = torch.clamp(x, -10, 10)
            x = x.view(x.size(0), -1)
            x = torch.sigmoid(x)
        return x


def load_data_dirs(txt_file):
    data_dirs = []

    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                data_dirs.append(line)

    return data_dirs


def data_root_from_txt(txt_file):
    """从 grid_8_dataset_paths.txt 中提取标注数据根目录名 (如 MNR_Label_GS8_20260806_1640)。"""
    with open(txt_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for part in line.replace("\\", "/").split("/"):
                if part.startswith("MNR_Label_GS8_"):
                    return part
    raise ValueError(f"{txt_file} 中未找到 MNR_Label_GS8_* 数据条目")


# =========================
# 5. 以下训练代码仅在直接运行时执行
# =========================
def train(gs, cost_down=True, epochs=20):
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    DATA_DIR = os.path.dirname(__file__) if '__file__' in dir() else '.'


    txt_file = f"grid_{gs}_dataset_paths.txt"
    DATA_ROOT = os.path.join(DATA_DIR, data_root_from_txt(txt_file))
    if not os.path.isdir(DATA_ROOT):
        raise FileNotFoundError(f"数据根目录不存在 (请重新生成 {txt_file}): {DATA_ROOT}")
    print(f"数据根目录: {DATA_ROOT}")
    IMAGE_LEVEL_SOURCES = ["SR_data", "JPG_data", "DIV2K"]  # 图像级划分 (8:2)
    PATCH_LEVEL_SOURCE = "test_data"                        # patch 级划分 (8:2)
    val_ratio = 0.2

    # ─── 1. 图像级划分: 同一图像(子文件夹)的所有 Patch 整体进入 Train 或 Val ───
    image_dirs = []  # (source, 图像文件夹路径)
    for src in IMAGE_LEVEL_SOURCES:
        src_root = os.path.join(DATA_ROOT, src)
        if not os.path.isdir(src_root):
            print(f"WARNING: 数据源不存在, 跳过: {src_root}")
            continue
        for name in sorted(os.listdir(src_root)):
            full = os.path.join(src_root, name)
            if os.path.isdir(full):
                image_dirs.append((src, full))
    np.random.shuffle(image_dirs)
    n_val_images = int(len(image_dirs) * val_ratio)
    val_image_dirs = image_dirs[:n_val_images]
    train_image_dirs = image_dirs[n_val_images:]

    # 保存图像级划分 (供 predict_cnn.py 推理时按 验证集/测试集 分开保存输出)
    # 注意: 训练时的 train(80%) 即推理观看时的 "test", val(20%) 即 "val"
    split_path = os.path.join(DATA_DIR, f"image_split_gs{gs}.csv")
    with open(split_path, "w", encoding="utf-8") as f:
        f.write("source,image_dir,split\n")
        for src, img_dir in train_image_dirs:
            f.write(f"{src},{os.path.relpath(img_dir, DATA_DIR)},test\n")
        for src, img_dir in val_image_dirs:
            f.write(f"{src},{os.path.relpath(img_dir, DATA_DIR)},val\n")
    print(f"图像级划分已保存: {split_path}")

    def load_image_list(dir_list):
        """加载图像文件夹列表的 dm / not_dm 数据集"""
        dm_ds, not_dm_ds = [], []
        for _, img_dir in dir_list:
            dm_csv = os.path.join(img_dir, f"grid_{gs}_dm_9x9.csv")
            if os.path.exists(dm_csv):
                dm_ds.append(MosquitoPatchDataset(dm_csv, label=1))
            not_dm_csv = os.path.join(img_dir, f"grid_{gs}_not_dm_9x9.csv")
            if os.path.exists(not_dm_csv):
                not_dm_ds.append(MosquitoPatchDataset(not_dm_csv, label=0))
        return dm_ds, not_dm_ds

    def count_patches(ds_list):
        return sum(len(d) for d in ds_list)

    train_img_dm, train_img_not_dm = [], []
    val_img_dm, val_img_not_dm = [], []
    src_stats = []  # (source, train 文件夹, train patches, val 文件夹, val patches)
    for src in IMAGE_LEVEL_SOURCES:
        tr_dirs = [e for e in train_image_dirs if e[0] == src]
        va_dirs = [e for e in val_image_dirs if e[0] == src]
        tr_dm, tr_nd = load_image_list(tr_dirs)
        va_dm, va_nd = load_image_list(va_dirs)
        train_img_dm += tr_dm
        train_img_not_dm += tr_nd
        val_img_dm += va_dm
        val_img_not_dm += va_nd
        src_stats.append((src, len(tr_dirs), count_patches(tr_dm) + count_patches(tr_nd),
                          len(va_dirs), count_patches(va_dm) + count_patches(va_nd)))

    # ─── 2. Patch 级划分: test_data 全部 Patch 混洗后 8:2 ───
    patch_datasets = []
    n_patch_dm = 0
    n_patch_not_dm = 0
    test_root = os.path.join(DATA_ROOT, PATCH_LEVEL_SOURCE)
    test_dirs_with_csv = set()
    for root, _, files in os.walk(test_root):
        for f in sorted(files):
            if f == f"grid_{gs}_dm_9x9.csv":
                ds = MosquitoPatchDataset(os.path.join(root, f), label=1)
                patch_datasets.append(ds)
                n_patch_dm += len(ds)
                test_dirs_with_csv.add(root)
            elif f == f"grid_{gs}_not_dm_9x9.csv":
                ds = MosquitoPatchDataset(os.path.join(root, f), label=0)
                patch_datasets.append(ds)
                n_patch_not_dm += len(ds)
                test_dirs_with_csv.add(root)

    test_all = ConcatDataset(patch_datasets)
    test_idx = np.arange(len(test_all))
    np.random.shuffle(test_idx)
    n_val_test = int(len(test_all) * val_ratio)
    train_patch_ds = Subset(test_all, test_idx[n_val_test:])
    val_patch_ds = Subset(test_all, test_idx[:n_val_test])

    # ─── 3. 组装 ───
    train_dataset = ConcatDataset(train_img_dm + train_img_not_dm + [train_patch_ds])
    val_dataset = ConcatDataset(val_img_dm + val_img_not_dm + [val_patch_ds])

    # ─── 4. 数据源统计 ───
    dm_size = count_patches(train_img_dm) + count_patches(val_img_dm) + n_patch_dm
    not_dm_size = count_patches(train_img_not_dm) + count_patches(val_img_not_dm) + n_patch_not_dm
    print("=" * 72)
    print(f"数据源划分 (GS={gs}, val_ratio={val_ratio:.0%}):")
    print(f"{'数据源':<14}{'Train 文件夹':>10}{'Train patches':>14}{'Val 文件夹':>10}{'Val patches':>14}")
    print("-" * 72)
    for src, tr_d, tr_p, va_d, va_p in src_stats:
        print(f"{src:<14}{tr_d:>10,}{tr_p:>14,}{va_d:>10,}{va_p:>14,}")
    print(f"{PATCH_LEVEL_SOURCE + '*':<14}{'-':>10}{len(train_patch_ds):>14,}{'-':>10}{len(val_patch_ds):>14,}")
    print("-" * 72)
    print(f"{'总计':<14}{len(train_image_dirs):>10,}{len(train_dataset):>14,}"
          f"{len(val_image_dirs):>10,}{len(val_dataset):>14,}")
    print(f"  * {PATCH_LEVEL_SOURCE}: {len(test_dirs_with_csv)} 个子文件夹, patch 级划分")
    print(f"  全量 DM patches: {dm_size:,} | Not-DM: {not_dm_size:,} (DM:Not-DM = 1:{not_dm_size / max(dm_size, 1):.2f})")
    print("=" * 72)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)

    # =========================
    # 6. 训练
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MosquitoDenoiseCNN(cost_down=cost_down).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    max_grad_norm = 1.0
    label_smoothing = 0.05
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-5
    )
    criterion = nn.BCELoss()

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
            if label_smoothing > 0:
                y = y * (1 - label_smoothing) + label_smoothing / 2
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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

        thresholds = [0.5]
        best_th = 0.5
        best_f1_epoch = 0.0
        best_cm = np.zeros((2, 2), dtype=np.int64)
        for th in thresholds:
            pb = (all_preds > th).astype(np.int64)
            tp = np.sum((pb == 1) & (all_labels == 1))
            tn = np.sum((pb == 0) & (all_labels == 0))
            fp = np.sum((pb == 1) & (all_labels == 0))
            fn = np.sum((pb == 0) & (all_labels == 1))
            prec = tp / (tp + fp + 1e-10)
            rec = tp / (tp + fn + 1e-10)
            f1_th = 2 * prec * rec / (prec + rec + 1e-10)
            if f1_th > best_f1_epoch:
                best_f1_epoch = f1_th
                best_th = th
                best_cm = np.array([[tn, fp], [fn, tp]])

        acc = (best_cm[0, 0] + best_cm[1, 1]) / best_cm.sum()
        prec = best_cm[1, 1] / (best_cm[1, 1] + best_cm[0, 1] + 1e-10)
        rec = best_cm[1, 1] / (best_cm[1, 1] + best_cm[1, 0] + 1e-10)

        print(f"Epoch [{epoch + 1}/{epochs}]  Loss: {avg_loss:.6f}  "
              f"Val Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {best_f1_epoch:.4f}  "
              f"th={best_th:.2f}")
        print(f"  Confusion Matrix:")
        print(f"    TN={best_cm[0, 0]:>5d}  FP={best_cm[0, 1]:>5d}")
        print(f"    FN={best_cm[1, 0]:>5d}  TP={best_cm[1, 1]:>5d}")

        if best_f1_epoch > best_f1:
            best_f1 = best_f1_epoch
            suffix = "_cost_down" if cost_down else ""
            model_dir = os.path.join(DATA_DIR, "model")
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_dir, f"mosquito_denoise_cnn{suffix}_grid_{gs}.pth"))
            np.save(os.path.join(model_dir, f"best_th{suffix}_grid_{gs}.npy"), np.array(best_th))
            print(f"  >>> Model saved (F1 improved to {best_f1_epoch:.4f} @ th={best_th:.2f})")

        if scheduler is not None:
            scheduler.step(best_f1_epoch)


if __name__ == "__main__":
    train(gs=8, cost_down=True, epochs=400)
    # train(gs=8, cost_down=False, epochs=20)
    # train(gs=16, cost_down=True, epochs=200)
