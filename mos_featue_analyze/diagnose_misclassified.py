"""
Diagnose misclassified DM samples and optimize for 33/33 detection.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix, f1_score

DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# 1. Load data (same as before)
# ============================================================

def load_samples(csv_path, label):
    df = pd.read_csv(csv_path)
    n = len(df)
    n_samples = n // 9
    df = df.iloc[:n_samples * 9]
    samples = []
    for i in range(n_samples):
        group = df.iloc[i * 9:(i + 1) * 9].copy().reset_index(drop=True)
        samples.append({
            "source": csv_path.name,
            "center_row": int(group.loc[4, "row"]),
            "center_col": int(group.loc[4, "col"]),
            "group": group,
            "label": label,
        })
    return samples

FEATURE_NAMES = []
def add_feature(feat_dict, name, value):
    FEATURE_NAMES.append(name)
    feat_dict[name] = value

def extract_features(sample):
    g = sample["group"]
    center = g.loc[4]
    f = {}
    raw_center_keys = [
        "mean_var", "max_var", "top5_var", "median_var",
        "high_var_count", "very_high_var_count",
        "residual_mean", "residual_max",
        "lap_mean", "lap_max", "grad_mean", "grad_max",
        "edge_strength", "edge_orientation_conf",
        "row_second_diff", "col_second_diff",
        "profile_ringing_max", "profile_ringing_mean",
        "row_ringing_dyn_score", "row_ringing_d2_score", "row_ringing_sign_score",
        "col_ringing_dyn_score", "col_ringing_d2_score", "col_ringing_sign_score",
    ]
    for k in raw_center_keys:
        add_feature(f, f"center_{k}", center[k])
    stat_keys = [
        "mean_var", "max_var", "lap_mean", "lap_max",
        "edge_strength", "edge_orientation_conf",
        "profile_ringing_max", "profile_ringing_mean",
        "residual_mean", "residual_max", "grad_mean", "grad_max",
        "high_var_count", "very_high_var_count",
    ]
    for k in stat_keys:
        vals = g[k].values
        add_feature(f, f"neighbor_{k}_mean", float(np.mean(vals)))
        add_feature(f, f"neighbor_{k}_std", float(np.std(vals)))
        add_feature(f, f"neighbor_{k}_max", float(np.max(vals)))
    eps = 1e-8
    for k in ["mean_var", "max_var", "lap_mean", "lap_max",
              "edge_strength", "profile_ringing_max", "residual_mean", "grad_mean"]:
        neighbor_mean = float(np.mean(g[k].values))
        add_feature(f, f"ratio_{k}", center[k] / (neighbor_mean + eps))
    for k in ["mean_var", "lap_mean", "profile_ringing_max", "edge_strength", "residual_max"]:
        add_feature(f, f"center_is_max_{k}", 1.0 if center[k] == g[k].max() else 0.0)
        add_feature(f, f"count_above_median_{k}", int(np.sum(g[k].values > g[k].median())))
    add_feature(f, "total_high_var_count", int(np.sum(g["high_var_count"].values > 0)))
    add_feature(f, "total_very_high_var_count", int(np.sum(g["very_high_var_count"].values > 0)))
    ringing_vals = g["profile_ringing_max"].values
    add_feature(f, "ringing_consistency", float(np.std(ringing_vals)))
    add_feature(f, "ringing_center_minus_mean", center["profile_ringing_max"] - float(np.mean(ringing_vals)))
    orient_vals = g["edge_orientation_conf"].values
    add_feature(f, "orient_consistency", float(np.std(orient_vals)))
    return f

# Load
all_samples = []
for csv_path in sorted(DATA_DIR.glob("*.csv")):
    name = csv_path.stem.lower()
    if "not_dm" in name:
        all_samples.extend(load_samples(csv_path, label=0))
    elif "dm" in name:
        all_samples.extend(load_samples(csv_path, label=1))

X_rows, y, meta = [], [], []
for s in all_samples:
    feat = extract_features(s)
    X_rows.append(feat)
    y.append(s["label"])
    meta.append((s["source"], s["center_row"], s["center_col"]))

X = pd.DataFrame(X_rows)
y = np.array(y)
print(f"Total: {X.shape[0]} samples, {sum(y)} DM, {len(y)-sum(y)} non-DM")

# ============================================================
# 2. LOO CV with baseline RF
# ============================================================

clf = RandomForestClassifier(
    n_estimators=200, max_depth=5, min_samples_leaf=1,
    class_weight="balanced", random_state=42,
)

loo = LeaveOneOut()
y_true_all, y_pred_all, y_prob_all = [], [], []
train_scores = []

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    clf.fit(X_train, y_train)
    y_pred_all.append(clf.predict(X_test)[0])
    y_prob_all.append(clf.predict_proba(X_test)[0, 1])
    y_true_all.append(y_test[0])

y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)
y_prob_all = np.array(y_prob_all)

cm = confusion_matrix(y_true_all, y_pred_all)
print(f"\nBaseline RF: acc={np.mean(y_true_all==y_pred_all):.3f}, F1={f1_score(y_true_all, y_pred_all):.3f}")
print(f"Confusion:\n{cm}")

# ============================================================
# 3. Diagnose misclassified DM samples
# ============================================================

mis_dm_idx = [i for i in range(len(y)) if y[i]==1 and y_pred_all[i]==0]
fp_idx = [i for i in range(len(y)) if y[i]==0 and y_pred_all[i]==1]

print(f"\n{'='*70}")
print(f"MISCLASSIFIED DM SAMPLES ({len(mis_dm_idx)} false negatives)")
print(f"{'='*70}")

# Compare feature values for misclassified DM vs correctly classified DM vs non-DM
key_feats = ["center_mean_var", "center_max_var", "center_lap_mean",
             "center_profile_ringing_max", "center_edge_strength",
             "center_residual_max", "total_high_var_count",
             "center_row_ringing_sign_score", "center_col_ringing_sign_score",
             "orient_consistency", "neighbor_residual_mean_std"]

for i in mis_dm_idx:
    print(f"\n  Sample {i}: source={meta[i][0]}, pos=({meta[i][1]},{meta[i][2]}), prob={y_prob_all[i]:.3f}")
    for feat in key_feats[:8]:
        val = X.loc[i, feat]
        dm_mean = np.mean([X.loc[j, feat] for j in range(len(y)) if y[j]==1 and y_pred_all[j]==1])
        nd_mean = np.mean([X.loc[j, feat] for j in range(len(y)) if y[j]==0])
        print(f"    {feat:45s} = {val:10.2f}  (correct DM mean: {dm_mean:10.2f}, non-DM mean: {nd_mean:10.2f})")

print(f"\n  False positive ({len(fp_idx)}):")
for i in fp_idx:
    print(f"    Sample {i}: source={meta[i][0]}, pos=({meta[i][1]},{meta[i][2]}), prob={y_prob_all[i]:.3f}")
    for feat in key_feats[:6]:
        val = X.loc[i, feat]
        dm_mean = np.mean([X.loc[j, feat] for j in range(len(y)) if y[j]==1])
        nd_mean = np.mean([X.loc[j, feat] for j in range(len(y)) if y[j]==0])
        print(f"    {feat:35s} = {val:10.2f}  (DM mean: {dm_mean:10.2f}, non-DM mean: {nd_mean:10.2f})")

# ============================================================
# 4. Strategy A: Threshold tuning
# ============================================================

print(f"\n{'='*70}")
print("Strategy A: Threshold Tuning")
print(f"{'='*70}")

thresholds = np.arange(0.05, 0.75, 0.025)
results = []
for th in thresholds:
    pred = (y_prob_all >= th).astype(int)
    cm = confusion_matrix(y_true_all, pred)
    fn = cm[1][0]  # false negatives (DM predicted as non-DM)
    fp = cm[0][1]  # false positives
    results.append({"threshold": th, "fn": fn, "fp": fp,
                    "dm_recall": cm[1][1] / (cm[1][0] + cm[1][1]),
                    "nd_specificity": cm[0][0] / (cm[0][0] + cm[0][1]),
                    "accuracy": np.mean(y_true_all == pred)})

res_df = pd.DataFrame(results)
target_rows = res_df[res_df["fn"] == 0]
print(f"\nThresholds achieving 33/33 DM (FN=0):")
if len(target_rows) > 0:
    print(target_rows.to_string(index=False))
else:
    # Find threshold with max DM recall and print nearby
    best = res_df.loc[res_df["dm_recall"].idxmax()]
    print(f"  Cannot achieve FN=0. Best: th={best['threshold']:.3f}, "
          f"FN={best['fn']}, FP={best['fp']}, acc={best['accuracy']:.3f}")

# Find best trade-off
print(f"\nThreshold sweep (showing thresholds where DM recall changes):")
for _, row in res_df[res_df["threshold"].isin(
    np.arange(0.1, 0.7, 0.05)
)].iterrows():
    print(f"  th={row['threshold']:.3f}: DM recall={row['dm_recall']:.3f} "
          f"non-DM specificity={row['nd_specificity']:.3f} "
          f"FN={int(row['fn'])} FP={int(row['fp'])} acc={row['accuracy']:.3f}")

# ============================================================
# 5. Strategy B: Increase model capacity + DM class weight
# ============================================================

print(f"\n{'='*70}")
print("Strategy B: Model Capacity & Class Weight Tuning")
print(f"{'='*70}")

configs = [
    {"name": "RF deep (max_depth=10)", "max_depth": 10, "n_estimators": 300, "class_weight": "balanced"},
    {"name": "RF deep + heavy DM weight 1:5", "max_depth": 10, "n_estimators": 300, "class_weight": {0: 1, 1: 5}},
    {"name": "RF deep + heavy DM weight 1:10", "max_depth": 10, "n_estimators": 300, "class_weight": {0: 1, 1: 10}},
    {"name": "RF deep + heavy DM weight 1:3", "max_depth": 10, "n_estimators": 300, "class_weight": {0: 1, 1: 3}},
    {"name": "RF shallow + heavy DM 1:5", "max_depth": 5, "n_estimators": 500, "class_weight": {0: 1, 1: 5}},
    {"name": "RF default + DM weight 1:5", "max_depth": None, "n_estimators": 200, "class_weight": {0: 1, 1: 5}},
]

best_cfg = None
best_fn = 99
best_results = []

for cfg in configs:
    clf = RandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        min_samples_leaf=1,
        class_weight=cfg["class_weight"],
        random_state=42,
    )
    loo = LeaveOneOut()
    yt, yp, ypr = [], [], []
    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        clf.fit(X_tr, y_tr)
        yp.append(clf.predict(X_te)[0])
        ypr.append(clf.predict_proba(X_te)[0, 1])
        yt.append(y_te[0])
    yt, yp = np.array(yt), np.array(yp)
    cm = confusion_matrix(yt, yp)
    fn = cm[1][0]
    fp = cm[0][1]
    acc = np.mean(yt == yp)
    f1 = f1_score(yt, yp)
    print(f"  {cfg['name']:40s} acc={acc:.3f} F1={f1:.3f} FN={fn} FP={fp}")
    best_results.append({"name": cfg["name"], "fn": fn, "fp": fp, "acc": acc, "f1": f1,
                         "cm": cm, "y_pred": yp, "y_prob": np.array(ypr), "yt": yt})
    if fn < best_fn:
        best_fn = fn
        best_cfg = best_results[-1]

# Also try threshold tuning on the best config
print(f"\n  Threshold tuning on best config ({best_cfg['name']}):")
for th in np.arange(0.1, 0.6, 0.05):
    pred = (best_cfg["y_prob"] >= th).astype(int)
    cm = confusion_matrix(best_cfg["yt"], pred)
    print(f"    th={th:.2f}: FN={cm[1][0]} FP={cm[0][1]} acc={np.mean(best_cfg['yt']==pred):.3f}")

# ============================================================
# 6. Strategy C: Feature Reduction (remove noise)
# ============================================================

print(f"\n{'='*70}")
print("Strategy C: Feature Selection (reduce overfitting)")
print(f"{'='*70}")

# Train on full data to get importance
rf_full = RandomForestClassifier(n_estimators=500, max_depth=8, class_weight="balanced", random_state=42)
rf_full.fit(X, y)
imp = pd.DataFrame({"feat": X.columns, "imp": rf_full.feature_importances_}).sort_values("imp", ascending=False)

# Test with different numbers of top features
for n_feat in [10, 15, 20, 30, 40]:
    top_feats = imp.head(n_feat)["feat"].values
    X_sub = X[list(top_feats)]

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=8,
        class_weight={0: 1, 1: 5},
        random_state=42,
    )
    loo = LeaveOneOut()
    yt, yp = [], []
    for train_idx, test_idx in loo.split(X_sub):
        X_tr, X_te = X_sub.iloc[train_idx], X_sub.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        clf.fit(X_tr, y_tr)
        yp.append(clf.predict(X_te)[0])
        yt.append(y_te[0])
    yt, yp = np.array(yt), np.array(yp)
    cm = confusion_matrix(yt, yp)
    fn = cm[1][0]
    print(f"  Top-{n_feat:2d} features: acc={np.mean(yt==yp):.3f} FN={fn} FP={cm[0][1]}")

# Best feature set from last run
top_feats = imp.head(20)["feat"].values
X_best = X[list(top_feats)]

clf_best = RandomForestClassifier(
    n_estimators=500, max_depth=10,
    class_weight={0: 1, 1: 5},
    min_samples_leaf=1,
    random_state=42,
)

loo = LeaveOneOut()
yt, yp, ypr = [], [], []
for train_idx, test_idx in loo.split(X_best):
    X_tr, X_te = X_best.iloc[train_idx], X_best.iloc[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    clf_best.fit(X_tr, y_tr)
    yp.append(clf_best.predict(X_te)[0])
    ypr.append(clf_best.predict_proba(X_te)[0, 1])
    yt.append(y_te[0])

yt, yp, ypr = np.array(yt), np.array(yp), np.array(ypr)
cm = confusion_matrix(yt, yp)
fn_mask = (yt == 1) & (yp == 0)
fp_mask = (yt == 0) & (yp == 1)

print(f"\n{'='*70}")
print("BEST CONFIG: Top-20 features + RF(depth=10, DM:non-DM=5:1)")
print(f"{'='*70}")
print(f"  Confusion: {cm.tolist()}")
print(f"  FN count: {cm[1][0]}, FP count: {cm[0][1]}")
print(f"  Acc: {np.mean(yt==yp):.3f}")

if cm[1][0] > 0:
    print(f"\n  Remaining FN samples:")
    for i in np.where(fn_mask)[0]:
        print(f"    {i}: ({meta[i][1]},{meta[i][2]}) prob={ypr[i]:.3f}")
    print(f"\n  Threshold to catch them:")
    min_fn_prob = min([ypr[i] for i in np.where(fn_mask)[0]])
    for th in np.arange(max(0.05, min_fn_prob-0.05), min_fn_prob+0.05, 0.01):
        pred_at_th = (ypr >= th).astype(int)
        c = confusion_matrix(yt, pred_at_th)
        print(f"    th={th:.2f}: FN={c[1][0]} FP={c[0][1]}")

# ============================================================
# 7. Summary & Recommendation
# ============================================================

print(f"\n{'='*70}")
print("SUMMARY: Path to 33/33 DM")
print(f"{'='*70}")

# Check if threshold 0.3 works
th_test = 0.30
pred_03 = (y_prob_all >= th_test).astype(int)
cm03 = confusion_matrix(y_true_all, pred_03)
print(f"\n  Option 1: Keep baseline RF, lower threshold to {th_test}")
print(f"    DM recall: {cm03[1][1]}/{cm03[1][0]+cm03[1][1]} "
      f"(FN={cm03[1][0]}), non-DM specificity: {cm03[0][0]}/{cm03[0][0]+cm03[0][1]} "
      f"(FP={cm03[0][1]})")

th_test = 0.32
pred_032 = (y_prob_all >= th_test).astype(int)
cm032 = confusion_matrix(y_true_all, pred_032)
print(f"\n  Option 1b: Keep baseline RF, lower threshold to {th_test}")
print(f"    DM recall: {cm032[1][1]}/{cm032[1][0]+cm032[1][1]} "
      f"(FN={cm032[1][0]}), non-DM specificity: {cm032[0][0]}/{cm032[0][0]+cm032[0][1]} "
      f"(FP={cm032[0][1]})")

print(f"\n  Option 2: Best config (Top-20 features, RF depth=10, DM weight=5)")
cm_best_05 = confusion_matrix(yt, yp)
print(f"    Default th=0.5: FN={cm_best_05[1][0]} FP={cm_best_05[0][1]} acc={np.mean(yt==yp):.3f}")
# find threshold for FN=0
for th in np.arange(0.01, 0.6, 0.01):
    pred_t = (ypr >= th).astype(int)
    c = confusion_matrix(yt, pred_t)
    if c[1][0] == 0:
        print(f"    th={th:.2f} achieves FN=0, FP={c[0][1]}, acc={np.mean(yt==pred_t):.3f}")
        break

print(f"\n  Option 3: Ensemble - average predictions of multiple models")
print(f"    (Train 5 RF with different random seeds, average probabilities)")
