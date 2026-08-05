"""
Window Size Ablation Study

验证 9x9 Patch Window 是否必要, 以及 DM 检测对上下文信息的依赖程度。

实验原则:
  - 只改变输入窗口大小 (当前 WINDOW_SIZES=[5, 7, 9], 支持 1/3/5/7/9)
  - 其余条件完全一致: Label / 数据集 / Train-Val划分 / Seed / LR / Batch / Epoch / CostDown
  - 中心 Patch 固定不变: 从 9x9 中心裁剪出 window_size x window_size, 不补零
  - CNN 保持 4 层卷积结构, 但为匹配小窗口, 用 1x1 conv 替代部分 3x3 conv:
      window 9 -> kernels [3,3,3,3]
      window 7 -> kernels [3,3,3,1]
      window 5 -> kernels [3,3,1,1]
      window 3 -> kernels [3,1,1,1]
      window 1 -> kernels [1,1,1,1]
    每个 3x3 conv 使特征图缩小 2 像素, 从 window_size 到 1 需 (window_size-1)/2 个
    3x3 conv, 因此无需 padding, 输入尺寸随窗口大小变化。

运行:
  python window_ablation.py
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset

from dm_cnn import MosquitoDenoiseCNN, MosquitoPatchDataset, load_data_dirs

# ─── 配置 (与 dm_cnn.py train() 保持一致) ───
SEED = 42
GS = 8
TXT_FILE = "grid_8_dataset_paths.txt"
VAL_RATIO = 0.2
EPOCHS = 5
BATCH_SIZE = 128
COST_DOWN = True
# WINDOW_SIZES = [1, 3, 5, 7, 9]
WINDOW_SIZES = [5, 7, 9]

SCRIPT_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(SCRIPT_DIR, "model_window")
os.makedirs(MODEL_DIR, exist_ok=True)
CSV_PATH = os.path.join(MODEL_DIR, "window_ablation.csv")
REPORT_PATH = os.path.join(MODEL_DIR, "window_ablation_report.md")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_datasets(window_size):
    """与 dm_cnn.train() 相同的数据加载, 仅增加 window_size。"""
    DATA_DIR = SCRIPT_DIR
    data_dirs = load_data_dirs(TXT_FILE)
    dm_datasets, not_dm_datasets = [], []
    for data_dir in data_dirs:
        dm_csv = os.path.join(data_dir, f"grid_{GS}_dm_9x9.csv")
        if os.path.exists(dm_csv) and os.path.getsize(dm_csv) > 0:
            dm_datasets.append(MosquitoPatchDataset(dm_csv, label=1, gs=GS, window_size=window_size))
        not_dm_csv = os.path.join(data_dir, f"grid_{GS}_not_dm_9x9.csv")
        if os.path.exists(not_dm_csv) and os.path.getsize(not_dm_csv) > 0:
            not_dm_datasets.append(MosquitoPatchDataset(not_dm_csv, label=0, gs=GS, window_size=window_size))

    dm_dataset = ConcatDataset(dm_datasets)
    not_dm_dataset = ConcatDataset(not_dm_datasets)

    # 与 dm_cnn.train() 完全相同的划分逻辑
    dm_size = len(dm_dataset)
    not_dm_size = len(not_dm_dataset)
    dm_val = int(dm_size * VAL_RATIO)
    not_dm_val = int(not_dm_size * VAL_RATIO)

    dm_indices = np.arange(dm_size)
    not_dm_indices = np.arange(not_dm_size)
    np.random.shuffle(dm_indices)
    np.random.shuffle(not_dm_indices)

    train_idx = list(dm_indices[dm_val:]) + [dm_size + i for i in not_dm_indices[not_dm_val:]]
    val_idx = list(dm_indices[:dm_val]) + [dm_size + i for i in not_dm_indices[:not_dm_val]]

    full_dataset = ConcatDataset([dm_dataset, not_dm_dataset])
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  window={window_size}x{window_size}: train={len(train_dataset)}, val={len(val_dataset)}")
    return train_loader, val_loader


def train_window(window_size):
    """训练单个窗口模型, 返回 (best_metrics_dict)。"""
    set_seed()
    train_loader, val_loader = build_datasets(window_size)

    model = MosquitoDenoiseCNN(cost_down=COST_DOWN, window_size=window_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    max_grad_norm = 1.0
    label_smoothing = 0.05
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-5
    )
    criterion = nn.BCELoss()

    best_f1 = 0.0
    best_metrics = None
    model_path = os.path.join(MODEL_DIR, f"model_window_{window_size}x{window_size}.pth")

    for epoch in range(EPOCHS):
        # ─── Train ───
        model.train()
        total_loss = 0.0
        total_count = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            if label_smoothing > 0:
                y = y * (1 - label_smoothing) + label_smoothing / 2
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_count += x.size(0)
        avg_loss = total_loss / total_count

        # ─── Validation ───
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                all_preds.append(model(x).cpu().numpy())
                all_labels.append(y.cpu().numpy())
        all_preds = np.concatenate(all_preds).flatten()
        all_labels = np.concatenate(all_labels).flatten()

        pb = (all_preds > 0.5).astype(np.int64)
        tp = int(np.sum((pb == 1) & (all_labels == 1)))
        tn = int(np.sum((pb == 0) & (all_labels == 0)))
        fp = int(np.sum((pb == 1) & (all_labels == 0)))
        fn = int(np.sum((pb == 0) & (all_labels == 1)))
        prec = tp / (tp + fp + 1e-10)
        rec = tp / (tp + fn + 1e-10)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        acc = (tn + tp) / (tn + tp + fp + fn)

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  [{window_size}x{window_size}] epoch {epoch + 1}/{EPOCHS}  "
                  f"loss={avg_loss:.6f}  F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_metrics = dict(f1=f1, precision=prec, recall=rec, accuracy=acc,
                                tp=tp, tn=tn, fp=fp, fn=fn, epoch=epoch + 1)
            torch.save(model.state_dict(), model_path)

        if scheduler is not None:
            scheduler.step(f1)

    print(f"  >>> [{window_size}x{window_size}] best F1={best_f1:.4f} @ epoch={best_metrics['epoch']}")
    return best_metrics


def main():
    print(f"Device: {device}")
    results = {}
    for w in WINDOW_SIZES:
        print(f"\n=== Window {w}x{w} ===")
        results[w] = train_window(w)

    # ─── CSV ───
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        f.write("window_size,f1,precision,recall,accuracy,tp,tn,fp,fn\n")
        for w in WINDOW_SIZES:
            m = results[w]
            f.write(f"{w},{m['f1']:.4f},{m['precision']:.4f},{m['recall']:.4f},"
                    f"{m['accuracy']:.4f},{m['tp']},{m['tn']},{m['fp']},{m['fn']}\n")
    print(f"\nCSV saved: {CSV_PATH}")

    # ─── Report ───
    base = results[9]
    lines = []
    lines.append("# Window Size Ablation Report")
    lines.append("")
    lines.append(f"- **Date**: (run date)")
    lines.append(f"- **Data**: `{TXT_FILE}` | GS={GS} | Epochs={EPOCHS} | Seed={SEED}")
    lines.append(f"- **中心 Patch 固定不变 (从 9x9 中心裁剪, 不补零); 4 层 CNN 中部分 3x3 conv 由 1x1 conv 替代以匹配小窗口, 其余训练条件完全一致**")
    lines.append("")

    lines.append("## 1. 各窗口性能对比")
    lines.append("")
    lines.append("| Window | F1 | Precision | Recall | Accuracy | TP | TN | FP | FN |")
    lines.append("|--------|-----|-----------|--------|----------|----|----|----|----|")
    for w in WINDOW_SIZES:
        m = results[w]
        lines.append(f"| {w}x{w} | {m['f1']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
                     f"{m['accuracy']:.4f} | {m['tp']} | {m['tn']} | {m['fp']} | {m['fn']} |")
    lines.append("")

    lines.append("## 2. 相对 9x9 的性能损失 (F1 drop)")
    lines.append("")
    lines.append("| Window | F1 | F1_drop vs 9x9 |")
    lines.append("|--------|-----|----------------|")
    for w in WINDOW_SIZES:
        m = results[w]
        drop = base['f1'] - m['f1']
        lines.append(f"| {w}x{w} | {m['f1']:.4f} | {drop:+.4f} |")
    lines.append("")

    lines.append("## 3. 上下文收益分析 (Gain per window step)")
    lines.append("")
    lines.append("| Step | Gain |")
    lines.append("|------|------|")
    for i in range(1, len(WINDOW_SIZES)):
        prev, cur = WINDOW_SIZES[i - 1], WINDOW_SIZES[i]
        gain = results[cur]['f1'] - results[prev]['f1']
        lines.append(f"| {cur}x{cur} - {prev}x{prev} | {gain:+.4f} |")
    lines.append("")

    # 分析
    lines.append("## 4. 分析")
    lines.append("")
    gains = [results[WINDOW_SIZES[i]]['f1'] - results[WINDOW_SIZES[i - 1]]['f1']
             for i in range(1, len(WINDOW_SIZES))]
    # 上下文饱和: 最后几步增益趋近 0
    sat = all(abs(g) < 0.01 for g in gains[-2:])
    # 大窗口退化: 9x9 比 7x7 差
    degrade = results[9]['f1'] < results[7]['f1']
    # 最小可接受窗口: F1 drop < 0.02 的最小窗口
    min_ok = next((w for w in WINDOW_SIZES if (base['f1'] - results[w]['f1']) < 0.02), None)

    lines.append(f"- **上下文饱和 (Context Saturation)**: {'是' if sat else '否'} "
                 f"(最后两步增益: {gains[-2]:+.4f}, {gains[-1]:+.4f})")
    lines.append(f"- **大窗口性能退化**: {'是' if degrade else '否'} "
                 f"(9x9 F1={results[9]['f1']:.4f} vs 7x7 F1={results[7]['f1']:.4f})")
    lines.append(f"- **最小可接受窗口** (F1 drop < 0.02): "
                 f"{f'{min_ok}x{min_ok}' if min_ok else '无 (所有窗口均有显著损失)'}")
    lines.append("")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
