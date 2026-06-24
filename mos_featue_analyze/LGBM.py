import pandas as pd
import numpy as np
import os
import warnings
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# 忽略所有非致命警告（如 FutureWarning, DataConversionWarning 等）
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), "excel")

# 列为参数的列，不作为特征使用
PARAM_COLS = {
    "low_var_th", "high_var_th", "very_high_var_th", "max_strength_th",
    "eps",
    "dyn_score_lo", "dyn_score_hi",
    "d2_score_lo", "d2_score_hi",
    "sign_score_lo", "sign_score_hi",
    "dyn_ratio", "d2_ratio", "sign_ratio",
}

# 不作为特征的元数据列
SKIP_COLS = {"name", "row", "col"} | PARAM_COLS


def load_excel_data(data_dir=None):
    """遍历 excel 目录，加载所有 dm.csv / not_dm.csv 作为独立样本。"""
    if data_dir is None:
        data_dir = DATA_DIR
    samples = []
    for folder in sorted(os.listdir(data_dir)):
        fpath = os.path.join(data_dir, folder)
        if not os.path.isdir(fpath):
            continue
        for fname, label in [("dm.csv", 1), ("not_dm.csv", 0)]:
            csv_path = os.path.join(fpath, fname)
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            row_data = df.drop(columns=[c for c in SKIP_COLS if c in df.columns], errors="ignore")
            row_data["label"] = label
            samples.append(row_data)

    data = pd.concat(samples, ignore_index=True)
    print(f"Loaded {len(data)} samples ({data['label'].sum()} dm, {(1 - data['label']).sum()} not_dm)")
    return data


data = load_excel_data()

# 确保 y 为整型，避免 lightgbm 将其视为浮点类型产生警告
y = data["label"].astype(int)
X = data.drop(columns=["label"])

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
target_recall = 0.95  # 目标：正样本全部预测正确（召回率100%）。如果数据有噪声可设为 0.95 或 0.98
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
