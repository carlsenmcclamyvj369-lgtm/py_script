"""
数据贡献分析: 找出 grid_8_dataset_paths.txt 中哪些数据目录起正向作用, 哪些引入污染。

实验设计:
  Phase A (baseline): 用全部数据训练一个模型, 在"固定验证集"上按目录分别评估 F1。
                      固定验证集 = 每个目录各自 20% 的 patch (按目录分层, 固定 seed),
                      保证后续所有消融实验都在同一个验证集上对比。

  Phase B (消融): 对可疑目录 (Phase A 中 F1 低 / 样本异常) 逐个做 leave-one-out:
                  用"排除该目录后剩余数据"重新训练, 在同一个固定验证集上评估 F1。

  判定:
    - 排除后验证集 F1 明显上升  → 该目录是污染 (模型在它上学到了错误映射)
    - 排除后验证集 F1 明显下降  → 该目录有正向贡献 (不可缺失)
    - 排除后验证集 F1 基本不变  → 中性 (冗余数据)

运行:
  python data_ablation.py              # Phase A + B 全跑
  python data_ablation.py baseline     # 只跑 Phase A
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset

from dm_cnn import MosquitoDenoiseCNN, MosquitoPatchDataset, load_data_dirs

# ─── 配置 ───
GS = 8
TXT_FILE = "grid_8_dataset_paths.txt"
SEED = 42
VAL_RATIO = 0.2
EPOCHS_BASELINE = 400   # Phase A 训练轮数
EPOCHS_ABLATION = 20    # Phase B 消融: warm-start 微调轮数 (对比方向即可)
MAX_ABLATE = 15         # Phase B 最多消融多少个目录 (按可疑度排序)
F1_SUSPECT_TH = 0.50    # 目录验证 F1 低于此值 → 判定可疑
BATCH_SIZE = 128
USE_AMP = True          # 混合精度加速 (CUDA 下 ~1.5-2x)
MODEL_SAVE_DIR = os.path.join(os.path.dirname(__file__), "model_ablation")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True


# ─── 数据准备 ───
def load_all_dirs():
    """返回 [(dir_name, ConcatDataset(dm+not_dm)), ...]"""
    data_dirs = load_data_dirs(TXT_FILE)
    dirs = []
    for d in data_dirs:
        name = os.path.basename(d.rstrip("/\\"))
        datasets = []
        dm_csv = os.path.join(d, f"grid_{GS}_dm_9x9.csv")
        notdm_csv = os.path.join(d, f"grid_{GS}_not_dm_9x9.csv")
        if os.path.exists(dm_csv) and os.path.getsize(dm_csv) > 0:
            datasets.append(MosquitoPatchDataset(dm_csv, label=1, gs=GS))
        if os.path.exists(notdm_csv) and os.path.getsize(notdm_csv) > 0:
            datasets.append(MosquitoPatchDataset(notdm_csv, label=0, gs=GS))
        if datasets:
            dirs.append((name, ConcatDataset(datasets)))
    return dirs


def build_split(dir_datasets):
    """按目录分层划分 train/val (固定 seed)。返回:
    - full_dataset, dir_id (每个全局索引属于哪个目录)
    - train_idx, val_idx (全局索引)
    - val_dir_map: {dir_id: val 全局索引}
    """
    rng = np.random.default_rng(SEED)
    full_datasets = [ds for _, ds in dir_datasets]
    sizes = [len(ds) for ds in full_datasets]
    offsets = np.cumsum([0] + sizes)

    dir_id = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])
    full_dataset = ConcatDataset(full_datasets)

    train_idx, val_idx = [], []
    val_dir_map = {}
    for i in range(len(dir_datasets)):
        idx = rng.permutation(sizes[i]) + offsets[i]
        n_val = int(sizes[i] * VAL_RATIO)
        val = idx[:n_val]
        tr = idx[n_val:]
        train_idx.extend(tr.tolist())
        val_idx.extend(val.tolist())
        val_dir_map[i] = val
    return full_dataset, dir_id, np.array(train_idx), np.array(val_idx), val_dir_map


# ─── 训练 / 评估 ───
def train_model(model, train_loader, epochs, tag="", lr=3e-4, warm_start=False):
    """训练/微调。warm_start=True 时用较低 lr 微调 (Phase B 消融)。"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP and device.type == "cuda"):
                pred = model(x)
            # BCELoss 不能在 autocast 里计算, 移出 autocast 块
            loss = criterion(pred, y * 0.95 + 0.025)  # label smoothing
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{tag}] epoch {epoch + 1}/{epochs}  loss={total_loss / n:.6f}", flush=True)
    return model


@torch.no_grad()
def predict_all(model, indices, full_dataset):
    preds = []
    labels = []
    loader = DataLoader(Subset(full_dataset, indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    for x, y in loader:
        x = x.to(device)
        preds.append(model(x).cpu().numpy().flatten())
        labels.append(y.numpy().flatten())
    return np.concatenate(preds), np.concatenate(labels)


def metrics(preds, labels, th=0.5):
    pb = (preds > th).astype(np.int64)
    tp = np.sum((pb == 1) & (labels == 1))
    tn = np.sum((pb == 0) & (labels == 0))
    fp = np.sum((pb == 1) & (labels == 0))
    fn = np.sum((pb == 0) & (labels == 1))
    prec = tp / (tp + fp + 1e-10)
    rec = tp / (tp + fn + 1e-10)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    acc = (tn + tp) / max(tn + tp + fp + fn, 1)
    return f1, prec, rec, acc, int(tp), int(tn), int(fp), int(fn)


# ─── Phase A: baseline + 按目录验证 F1 ───
def run_baseline(full_dataset, dir_id, train_idx, val_idx, val_dir_map, dir_names):
    print("\n" + "=" * 80)
    print(f"Phase A: 训练 baseline (全部数据, {EPOCHS_BASELINE} epochs)")
    print("=" * 80)
    model = MosquitoDenoiseCNN(cost_down=True).to(device)
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    train_model(model, train_loader, EPOCHS_BASELINE, tag="baseline")
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "baseline.pth"))

    # 全局验证集
    g_preds, g_labels = predict_all(model, val_idx, full_dataset)
    g_f1, g_prec, g_rec, g_acc, *cm = metrics(g_preds, g_labels)
    print(f"\nBaseline 全局验证集 (val={len(val_idx)}): F1={g_f1:.4f} Prec={g_prec:.4f} "
          f"Rec={g_rec:.4f} Acc={g_acc:.4f}")

    # 按目录验证 F1
    print(f"\n{'目录':<45} {'val数':>7} {'dm':>5} {'F1':>7} {'Prec':>7} {'Rec':>7} {'FP':>5} {'判据':>8}")
    print("-" * 95)
    rows = []
    for i, name in enumerate(dir_names):
        v_idx = val_dir_map[i]
        if len(v_idx) == 0:
            continue
        preds, labels = predict_all(model, v_idx, full_dataset)
        f1, prec, rec, acc, tp, tn, fp, fn = metrics(preds, labels)
        n_dm = int(np.sum(labels == 1))
        if n_dm == 0:
            # 无 DM 正样本: F1 无意义, 用 FP 率评估 (not-DM 被误判成 DM 的比例)
            fp_rate = fp / max(tn + fp, 1)
            flag = "无DM样本" if fp_rate <= 0.1 else f"无DM+FP高({fp_rate:.0%})"
            flag = flag if fp_rate <= 0.1 else f"无DM,FP={fp}/{tn+fp}"
            rows.append((i, name, len(v_idx), 0, float('nan'), float('nan'), float('nan'), fp, flag))
            print(f"{name:<45} {len(v_idx):>7} {n_dm:>5} {'-':>7} {'-':>7} {'-':>7} {fp:>5} {flag:>8}")
        else:
            flag = "可疑" if f1 < F1_SUSPECT_TH else ""
            rows.append((i, name, len(v_idx), n_dm, f1, prec, rec, fp, flag))
            print(f"{name:<45} {len(v_idx):>7} {n_dm:>5} {f1:>7.3f} {prec:>7.3f} {rec:>7.3f} {fp:>5} {flag:>8}")

    # Phase A 结果落盘, 挂了也能复用
    import csv
    with open(os.path.join(MODEL_SAVE_DIR, "phaseA_dir_f1.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dir", "val_count", "n_dm", "f1", "prec", "rec", "fp", "flag"])
        for r in rows:
            f1s = "-" if np.isnan(r[4]) else f"{r[4]:.4f}"
            precs = "-" if np.isnan(r[5]) else f"{r[5]:.4f}"
            recs = "-" if np.isnan(r[6]) else f"{r[6]:.4f}"
            w.writerow([r[1], r[2], r[3], f1s, precs, recs, r[7], r[8]])
    print(f"Phase A 结果已保存: {os.path.join(MODEL_SAVE_DIR, 'phaseA_dir_f1.csv')}")

    return model, g_f1, rows


# ─── Phase B: leave-one-out 消融 ───
def run_ablation(full_dataset, dir_id, train_idx, val_idx, val_dir_map, dir_names, rows, g_f1):
    import csv
    result_csv = os.path.join(MODEL_SAVE_DIR, "phaseB_ablation.csv")

    # 按可疑度排序: 先低 F1, 再小样本 (row: i, name, val_count, n_dm, f1, prec, rec, fp, flag)
    # 无 DM 样本的目录 F1 无意义, 不参与消融
    with_dm = [r for r in rows if r[3] > 0 and not np.isnan(r[4])]
    suspicious = sorted(with_dm, key=lambda r: (r[4], r[2]))
    ablate_list = [r for r in suspicious if r[4] < F1_SUSPECT_TH][:MAX_ABLATE]
    if not ablate_list:
        ablate_list = suspicious[:MAX_ABLATE]

    # 已完成的消融 (断点续跑)
    done = set()
    if os.path.exists(result_csv):
        with open(result_csv, newline="", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                done.add(line.split(",")[0])

    print("\n" + "=" * 80)
    print(f"Phase B: leave-one-out 消融 (warm-start {EPOCHS_ABLATION} epochs/次, 最多 {len(ablate_list)} 个目录)")
    print("=" * 80)

    baseline_state = torch.load(os.path.join(MODEL_SAVE_DIR, "baseline.pth"), map_location=device)

    results = []
    for row in ablate_list:
        i, name, _, _, f1_own, _, _, _, _ = row
        if name in done:
            print(f"\n--- 跳过已完成: [{name}]")
            continue

        exclude = np.where(dir_id == i)[0]
        train_mask = np.isin(train_idx, exclude, invert=True)
        new_train = train_idx[train_mask]
        print(f"\n--- 排除 [{name}] (目录自身 F1={f1_own:.3f}) 微调 {len(new_train)} patches ---")

        try:
            model = MosquitoDenoiseCNN(cost_down=True).to(device)
            model.load_state_dict(baseline_state)
            loader = DataLoader(Subset(full_dataset, new_train), batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=0)
            train_model(model, loader, EPOCHS_ABLATION, tag=name[:20], lr=1e-4, warm_start=True)

            preds, labels = predict_all(model, val_idx, full_dataset)
            f1, prec, rec, acc, *cm = metrics(preds, labels)
            delta = f1 - g_f1
            if delta > 0.005:
                verdict = "污染 (排除后F1↑)"
            elif delta < -0.005:
                verdict = "正向 (排除后F1↓)"
            else:
                verdict = "中性"
            results.append((name, f1_own, f1, delta, verdict))
            print(f"  val F1: {f1:.4f} (baseline {g_f1:.4f}, Δ={delta:+.4f}) → {verdict}")
        except Exception as e:
            verdict = f"失败: {e}"
            print(f"  [ERROR] {name} 消融失败: {e}")
            results.append((name, f1_own, float('nan'), float('nan'), verdict))
            f1, delta = float('nan'), float('nan')

        # 逐条落盘, 挂了可续跑
        with open(result_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if os.path.getsize(result_csv) == 0:
                w.writerow(["dir", "own_f1", "abl_f1", "delta_f1", "verdict"])
            w.writerow([name, f"{f1_own:.4f}", f"{f1:.4f}", f"{delta:+.4f}", verdict])

    print("\n" + "=" * 80)
    print("消融结果汇总 (固定验证集):")
    print("=" * 80)
    print(f"{'目录':<45} {'自身F1':>7} {'排除后F1':>9} {'ΔF1':>7}  结论")
    for name, f1_own, f1, delta, verdict in results:
        print(f"{name:<45} {f1_own:>7.3f} {f1:>9.4f} {delta:>+7.4f}  {verdict}")

    return results


def write_report(rows, results, g_f1):
    """生成 markdown 记录文档 model_ablation/ablation_report.md"""
    import datetime
    lines = []
    lines.append("# 数据消融分析报告 (Data Ablation Report)")
    lines.append("")
    lines.append(f"- **日期**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- **数据文件**: `{TXT_FILE}`")
    lines.append(f"- **GS**: {GS} | **训练轮数**: Phase A={EPOCHS_BASELINE}, Phase B={EPOCHS_ABLATION}")
    lines.append(f"- **固定验证集**: 每目录分层 20%, seed={SEED}")
    lines.append(f"- **Baseline 全局验证 F1**: {g_f1:.4f}")
    lines.append("")

    lines.append("## Phase A: 按目录验证 F1 (Baseline 模型)")
    lines.append("")
    lines.append("| 目录 | val 数 | DM 数 | F1 | Prec | Rec | FP | 判据 |")
    lines.append("|------|--------|-------|-----|------|-----|-----|------|")
    for r in sorted(rows, key=lambda x: (np.isnan(x[4]), x[4])):
        if np.isnan(r[4]):
            lines.append(f"| {r[1]} | {r[2]} | {r[3]} | - | - | - | {r[7]} | {r[8]} |")
        else:
            lines.append(f"| {r[1]} | {r[2]} | {r[3]} | {r[4]:.3f} | {r[5]:.3f} | {r[6]:.3f} | {r[7]} | {r[8]} |")
    lines.append("")

    lines.append("## Phase B: Leave-one-out 消融")
    lines.append("")
    lines.append("> 排除该目录后 warm-start 微调, 在固定验证集上对比 F1。")
    lines.append("> **ΔF1 > +0.005 → 污染**, ΔF1 < -0.005 → 正向, 其余 → 中性。")
    lines.append("")
    lines.append("| 目录 | 自身 F1 | 排除后 F1 | ΔF1 | 结论 |")
    lines.append("|------|---------|-----------|------|------|")
    for name, f1_own, f1, delta, verdict in results:
        if isinstance(f1, float) and np.isnan(f1):
            lines.append(f"| {name} | {f1_own:.3f} | - | - | {verdict} |")
        else:
            lines.append(f"| {name} | {f1_own:.3f} | {f1:.4f} | {delta:+.4f} | {verdict} |")
    lines.append("")

    # 汇总统计
    pol = [r for r in results if isinstance(r[3], float) and not np.isnan(r[3]) and r[3] > 0.005]
    pos = [r for r in results if isinstance(r[3], float) and not np.isnan(r[3]) and r[3] < -0.005]
    neu = [r for r in results if isinstance(r[3], float) and not np.isnan(r[3]) and -0.005 <= r[3] <= 0.005]
    lines.append("## 结论")
    lines.append("")
    lines.append(f"- **污染目录 (排除后验证集提升)**: {len(pol)} 个")
    for r in pol:
        lines.append(f"  - {r[0]} (Δ={r[3]:+.4f})")
    lines.append(f"- **正向贡献目录 (排除后下降)**: {len(pos)} 个")
    for r in pos:
        lines.append(f"  - {r[0]} (Δ={r[3]:+.4f})")
    lines.append(f"- **中性目录**: {len(neu)} 个")
    lines.append("")

    report_path = os.path.join(MODEL_SAVE_DIR, "ablation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n报告已保存: {report_path}")


def load_phaseA_rows():
    """从 phaseA_dir_f1.csv 恢复 rows (用于只跑 Phase B)。"""
    import csv
    csv_path = os.path.join(MODEL_SAVE_DIR, "phaseA_dir_f1.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} 不存在, 需要先跑 Phase A")
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            parts = line.rstrip("\n").split(",")
            name, val_count, n_dm, f1s, precs, recs, fp, flag = parts
            f1 = float(f1s) if f1s != "-" else float('nan')
            prec = float(precs) if precs != "-" else float('nan')
            rec = float(recs) if recs != "-" else float('nan')
            rows.append((None, name, int(val_count), int(n_dm), f1, prec, rec, int(fp), flag))
    return rows


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    only_baseline = mode == "baseline"
    only_phaseb = mode == "phaseb"

    print(f"加载数据目录: {TXT_FILE}")
    dir_datasets = load_all_dirs()
    dir_names = [n for n, _ in dir_datasets]
    total = sum(len(ds) for _, ds in dir_datasets)
    print(f"共 {len(dir_datasets)} 个目录, 总 patches: {total}")

    full_dataset, dir_id, train_idx, val_idx, val_dir_map = build_split(dir_datasets)
    print(f"train={len(train_idx)}, val={len(val_idx)}")

    if only_phaseb:
        # 只跑 Phase B: 复用已保存的 baseline 和 Phase A 结果
        baseline_path = os.path.join(MODEL_SAVE_DIR, "baseline.pth")
        if not os.path.exists(baseline_path):
            raise FileNotFoundError(f"{baseline_path} 不存在, 需要先跑 Phase A")

        model = MosquitoDenoiseCNN(cost_down=True).to(device)
        model.load_state_dict(torch.load(baseline_path, map_location=device))
        model.eval()

        # 重建 rows + 重算 baseline 全局 F1 (一次前向, 很快)
        rows = load_phaseA_rows()
        # 补上 dir 索引
        name2idx = {n: i for i, n in enumerate(dir_names)}
        rows = [(name2idx[r[1]],) + r[1:] for r in rows]

        g_preds, g_labels = predict_all(model, val_idx, full_dataset)
        g_f1, *_ = metrics(g_preds, g_labels)
        print(f"Baseline 全局验证 F1 (重算): {g_f1:.4f}")

        results = run_ablation(full_dataset, dir_id, train_idx, val_idx, val_dir_map,
                               dir_names, rows, g_f1)
        write_report(rows, results, g_f1)
        return

    model, g_f1, rows = run_baseline(full_dataset, dir_id, train_idx, val_idx, val_dir_map, dir_names)

    results = []
    if not only_baseline:
        results = run_ablation(full_dataset, dir_id, train_idx, val_idx, val_dir_map, dir_names, rows, g_f1)

    write_report(rows, results, g_f1)


if __name__ == "__main__":
    main()
