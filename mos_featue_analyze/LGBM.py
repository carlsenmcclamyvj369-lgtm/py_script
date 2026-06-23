import pandas as pd
import numpy as np
import warnings
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# 忽略所有非致命警告（如 FutureWarning, DataConversionWarning 等）
warnings.filterwarnings('ignore')

EPS = 1e-6


def build_samples_from_csv(csv_path, label):
    df = pd.read_csv(csv_path)
    assert len(df) % 9 == 0
    feature_cols = [c for c in df.columns if c not in ["row", "col"]]
    samples = []
    for i in range(0, len(df), 9):
        # 使用 .copy() 避免任何 SettingWithCopyWarning
        g = df.iloc[i:i + 9].reset_index(drop=True).copy()
        center = g.iloc[4]
        # 直接按索引 4 删除，更加干净
        neigh = g.drop(4).reset_index(drop=True)

        row = {
            "label": label,
            "center_row": int(center["row"]),
            "center_col": int(center["col"]),
        }
        for f in feature_cols:
            c = float(center[f])
            nv = neigh[f].astype(float).values
            vals = g[f].astype(float).values

            n_med = float(np.median(nv))
            n_mean = float(np.mean(nv))
            n_std = float(np.std(nv))

            row[f"c_{f}"] = c
            row[f"n_med_{f}"] = n_med
            row[f"n_mean_{f}"] = n_mean
            row[f"n_std_{f}"] = n_std
            row[f"d_med_{f}"] = c - n_med
            row[f"r_med_{f}"] = c / (n_med + EPS)
            row[f"rank_{f}"] = float(pd.Series(vals).rank(method="average").iloc[4] / 9.0)
        samples.append(row)
    return pd.DataFrame(samples)


dm = build_samples_from_csv("input0012dm.csv", label=1)
dm_1 = build_samples_from_csv("input0007_dm.csv", label=1)
not_dm_1 = build_samples_from_csv("input0012_not_dm.csv", label=0)
not_dm_2 = build_samples_from_csv("input0007_not_dm.csv", label=0)
not_dm_3 = build_samples_from_csv("input0002_not_dm.csv", label=0)
data = pd.concat([dm, dm_1, not_dm_1, not_dm_2, not_dm_3], ignore_index=True)

# 确保 y 为整型，避免 lightgbm 将其视为浮点类型产生警告
y = data["label"].astype(int)
X = data.drop(columns=["label", "center_row", "center_col"])

# 1. 修改类别权重：增加正样本(1)的权重，迫使模型重视正样本，减少漏报
# 使用 scale_pos_weight 替代 class_weight，这是 LightGBM 原生参数，更高效且无兼容性警告
model = LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=7,
    max_depth=3,
    min_child_samples=3,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=2.0,
    scale_pos_weight=2.0,  # 核心修改点：使用原生参数加大正样本权重
    random_state=42,
    verbosity=-1,  # 关闭 LightGBM 的警告和冗余日志
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
print("AUC scores:", auc_scores)
print("Mean AUC:", auc_scores.mean())

model.fit(X, y)
proba = model.predict_proba(X)[:, 1]

# 2. 动态阈值搜索：寻找满足“召回率达标且误判最少”的最佳阈值
best_th = 0.5
best_precision = 0.0
target_recall = 1.0  # 目标：正样本全部预测正确（召回率100%）。如果数据有噪声可设为 0.95 或 0.98
print("\n--- Searching for optimal threshold ---")

for th in np.arange(0.01, 0.99, 0.01):
    temp_pred = (proba >= th).astype(int)
    # 使用 labels=[0, 1] 确保始终返回 2x2 矩阵，彻底避免形状报错和警告
    cm = confusion_matrix(y, temp_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # 策略：在满足目标召回率的前提下，寻找精确率最高（即FP误判最少）的阈值
    if recall >= target_recall:
        if precision > best_precision:
            best_precision = precision
            best_th = th

print(f"Optimal threshold found: {best_th:.2f} (Recall >= {target_recall}, Precision: {best_precision:.3f})")

# 使用搜索到的最优阈值进行预测
pred = (proba >= best_th).astype(int)
print("\n--- Final Evaluation ---")
print("Confusion Matrix:")
print(confusion_matrix(y, pred, labels=[0, 1]))
print("\nClassification Report:")
print(classification_report(y, pred))
