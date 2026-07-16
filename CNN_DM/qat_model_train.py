"""
QAT (Quantization-Aware Training) — 将 FP32 模型压缩为 INT8
使用 NNI 2.10 的 QAT_Quantizer，训练流程参考 dm_cnn.py
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset

from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer
from nni.compression.pytorch.quantization.settings import set_quant_scheme_dtype

from dm_cnn import MosquitoDenoiseCNN, MosquitoPatchDataset

# =========================
# 配置
# =========================
COST_DOWN = True
SEED = 42
DATA_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(DATA_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# =========================
# 数据（与 dm_cnn.py 一致）
# =========================
dm_datasets = [
    MosquitoPatchDataset(os.path.join(DATA_DIR, "9x9_dm.csv"), label=1),
    MosquitoPatchDataset(os.path.join(DATA_DIR, "9x9_dm_merged.csv"), label=1),
    MosquitoPatchDataset(os.path.join(DATA_DIR, "9x9_dm_SR_x3.csv"), label=1),
    MosquitoPatchDataset(os.path.join(DATA_DIR, "9x9_dm_SR_4k_0707.csv"), label=1),
    MosquitoPatchDataset(os.path.join(DATA_DIR, "9x9_dm_seq_0710.csv"), label=1),
    MosquitoPatchDataset(os.path.join(DATA_DIR, "9x9_dm_test_data_append_0715.csv"), label=1),
]
not_dm_datasets = [
    MosquitoPatchDataset(os.path.join(DATA_DIR, "9x9_not_dm.csv"), label=0),
    MosquitoPatchDataset(os.path.join(DATA_DIR, "9x9_not_dm_merged.csv"), label=0),
    MosquitoPatchDataset(os.path.join(DATA_DIR, "9x9_not_dm_SR_x3.csv"), label=0),
    MosquitoPatchDataset(os.path.join(DATA_DIR, "9x9_not_dm_SR_x2_0707.csv"), label=0),
    MosquitoPatchDataset(os.path.join(DATA_DIR, "9x9_not_dm_SR_4k_0707.csv"), label=0),
    MosquitoPatchDataset(os.path.join(DATA_DIR, "9x9_not_dm_seq_0710.csv"), label=0),
    MosquitoPatchDataset(os.path.join(DATA_DIR, "9x9_not_dm_test_data_append_0715.csv"), label=0),
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
# 设备 & 模型
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = MosquitoDenoiseCNN(cost_down=COST_DOWN).to(device)

# 可选：加载预训练 FP32 权重加速收敛（不存在则从头训）
pretrained_path = os.path.join(DATA_DIR, "mosquito_denoise_cnn_cost_down_32_16_sig.pth")
if os.path.exists(pretrained_path):
    model.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)
    print(f"Loaded pretrained weights from {pretrained_path}")
else:
    print("No pretrained weights found, training from scratch")

# =========================
# QAT 配置
# =========================
# 量化 Conv2d 的 weight (INT8, per-channel) + activation (UINT8, per-tensor)
config_list = [{
    'quant_types': ['weight', 'input'],
    'quant_bits': {'weight': 8, 'input': 8},
    'op_types': ['Conv2d'],
    'quant_scheme': {
        'weight': 'per_channel_symmetric',
        'input': 'per_tensor_affine',
    },
    'quant_dtype': {
        'weight': 'int',
        'input': 'uint',
    },
    'quant_start_step': 0,
}]

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
criterion = nn.BCELoss()

# dummy_input 用于图跟踪（检测 Conv 结构）
dummy_input = torch.randn(1, 16, 9, 9).to(device)

quantizer = QAT_Quantizer(model, config_list, optimizer, dummy_input=dummy_input)
quantizer.compress()
print("QAT_Quantizer initialized")

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=50, T_mult=2, eta_min=1e-5
)

# =========================
# 训练
# =========================
epochs = 400
best_f1 = 0.0

for epoch in range(epochs):
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

    # 扫阈值找最佳 F1
    thresholds = np.arange(0.30, 0.75, 0.05)
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

    print(f"Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.6f}  "
          f"Val Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {best_f1_epoch:.4f}  "
          f"th={best_th:.2f}")
    print(f"  TN={best_cm[0,0]:>5d}  FP={best_cm[0,1]:>5d}  "
          f"FN={best_cm[1,0]:>5d}  TP={best_cm[1,1]:>5d}")

    if best_f1_epoch > best_f1:
        best_f1 = best_f1_epoch
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "mosquito_denoise_cnn_qat.pth"))
        np.save(os.path.join(DATA_DIR, "best_th_qat.npy"), np.array(best_th))
        print(f"  >>> Model saved (F1 improved to {best_f1_epoch:.4f} @ th={best_th:.2f})")

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print(f"  LR: {current_lr:.2e}")

# =========================
# 导出量化模型 + 校准参数
# =========================
print("\nExporting quantized model...")
quantizer.export_model(
    model_path=os.path.join(MODEL_DIR, "mosquito_denoise_cnn_qat_exported.pth"),
    calibration_path=os.path.join(MODEL_DIR, "mosquito_denoise_cnn_qat_calib.json"),
)
print(f"Done — Best F1: {best_f1:.4f}")
print(f"  Model:  {MODEL_DIR}/mosquito_denoise_cnn_qat_exported.pth")
print(f"  Calib:  {MODEL_DIR}/mosquito_denoise_cnn_qat_calib.json")
