"""
Mosquito Noise (DM) vs Non-DM Binary Classification Analysis.

Input: two CSV files containing 3x3 block neighborhood features.
- dm: label=1, not_dm: label=0
- Each sample = 9 consecutive rows (3x3 pixel block)
- Center point = 5th row of each 9-row group

Strategy:
  Use center-point features + 3x3 context features to train a Random Forest.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, f1_score
)
from sklearn.inspection import permutation_importance

# ============================================================
# 1. Load & group into samples (9 rows per sample)
# ============================================================

DATA_DIR = Path(__file__).parent

def load_samples(csv_path, label):
    """Load CSV, group every 9 rows as one sample."""
    df = pd.read_csv(csv_path)
    n = len(df)
    n_samples = n // 9
    if n % 9 != 0:
        print(f"  WARNING: {csv_path.name} has {n} rows, not divisible by 9. "
              f"Dropping last {n % 9} rows.")
        df = df.iloc[:n_samples * 9]

    samples = []
    for i in range(n_samples):
        group = df.iloc[i * 9:(i + 1) * 9].copy().reset_index(drop=True)
        samples.append({
            "center_row": int(group.loc[4, "row"]),
            "center_col": int(group.loc[4, "col"]),
            "group": group,
            "label": label,
        })
    return samples


print("Loading samples...")
csv_files = sorted(DATA_DIR.glob("*.csv"))

dm_samples = []
not_dm_samples = []
unknown_samples = []

for csv_path in csv_files:
    name = csv_path.stem.lower()
    if "not_dm" in name:
        samples = load_samples(csv_path, label=0)
        not_dm_samples.extend(samples)
        print(f"  {csv_path.name}: {len(samples)} non-DM samples")
    elif "dm" in name:
        samples = load_samples(csv_path, label=1)
        dm_samples.extend(samples)
        print(f"  {csv_path.name}: {len(samples)} DM samples")
    else:
        samples = load_samples(csv_path, label=-1)
        unknown_samples.extend(samples)
        print(f"  {csv_path.name}: {len(samples)} samples (unlabeled, skipped)")

print(f"\n  Total: {len(dm_samples)} DM, {len(not_dm_samples)} non-DM, "
      f"{len(unknown_samples)} unknown")

all_samples = dm_samples + not_dm_samples

# ============================================================
# 2. Feature Engineering
# ============================================================

FEATURE_NAMES = []

def add_feature(feat_dict, name, value):
    FEATURE_NAMES.append(name)
    feat_dict[name] = value


def extract_features(sample):
    g = sample["group"]
    center = g.loc[4]
    neighbors = g.drop(4)  # the other 8

    f = {}

    # ---- A. Center-point raw features ----
    raw_center_keys = [
        "mean_var", "max_var", "top5_var", "median_var",
        "high_var_count", "very_high_var_count",
        "residual_mean", "residual_max",
        "lap_mean", "lap_max",
        "grad_mean", "grad_max",
        "edge_strength", "edge_orientation_conf",
        "row_second_diff", "col_second_diff",
        "profile_ringing_max", "profile_ringing_mean",
        "row_ringing_dyn_score", "row_ringing_d2_score",
        "row_ringing_sign_score",
        "col_ringing_dyn_score", "col_ringing_d2_score",
        "col_ringing_sign_score",
    ]
    for k in raw_center_keys:
        add_feature(f, f"center_{k}", center[k])

    # ---- B. 3x3 neighborhood statistics ----
    stat_keys = [
        "mean_var", "max_var", "lap_mean", "lap_max",
        "edge_strength", "edge_orientation_conf",
        "profile_ringing_max", "profile_ringing_mean",
        "residual_mean", "residual_max",
        "grad_mean", "grad_max",
        "high_var_count", "very_high_var_count",
    ]
    for k in stat_keys:
        vals = g[k].values
        add_feature(f, f"neighbor_{k}_mean", float(np.mean(vals)))
        add_feature(f, f"neighbor_{k}_std", float(np.std(vals)))
        add_feature(f, f"neighbor_{k}_max", float(np.max(vals)))

    # ---- C. Ratio features: center vs neighborhood ----
    ratio_keys = [
        "mean_var", "max_var", "lap_mean", "lap_max",
        "edge_strength", "profile_ringing_max",
        "residual_mean", "grad_mean",
    ]
    eps = 1e-8
    for k in ratio_keys:
        neighbor_mean = float(np.mean(g[k].values))
        add_feature(f, f"ratio_{k}", center[k] / (neighbor_mean + eps))

    # ---- D. Spatial pattern features ----
    # Is center the maximum in the 3x3 window?
    for k in ["mean_var", "lap_mean", "profile_ringing_max",
              "edge_strength", "residual_max"]:
        center_is_max = 1.0 if center[k] == g[k].max() else 0.0
        add_feature(f, f"center_is_max_{k}", center_is_max)

        # How many neighbors also have high values?
        threshold = g[k].median()
        count_above = int(np.sum(g[k].values > threshold))
        add_feature(f, f"count_above_median_{k}", count_above)

    # ---- E. Aggregate context features ----
    # Number of high-var points in 3x3 (including center)
    add_feature(f, "total_high_var_count", int(np.sum(g["high_var_count"].values > 0)))
    add_feature(f, "total_very_high_var_count", int(np.sum(g["very_high_var_count"].values > 0)))

    # Ringing activity spread
    ringing_vals = g["profile_ringing_max"].values
    add_feature(f, "ringing_consistency", float(np.std(ringing_vals)))
    add_feature(f, "ringing_center_minus_mean",
                center["profile_ringing_max"] - float(np.mean(ringing_vals)))

    # Edge orientation consistency across 3x3
    orient_vals = g["edge_orientation_conf"].values
    add_feature(f, "orient_consistency", float(np.std(orient_vals)))

    return f


print("\nExtracting features...")
X_rows = []
y = []
meta = []
for s in all_samples:
    feat = extract_features(s)
    X_rows.append(feat)
    y.append(s["label"])
    meta.append((s["center_row"], s["center_col"]))

X = pd.DataFrame(X_rows)
print(f"  Feature matrix: {X.shape}")
print(f"  Labels: {sum(y)} DM, {len(y) - sum(y)} non-DM")

# ============================================================
# 3. Train/Evaluate Classifier (LOO CV due to small dataset)
# ============================================================

print("\n" + "=" * 60)
print("Classification with Leave-One-Out CV")
print("=" * 60)

clf = RandomForestClassifier(
    n_estimators=200, max_depth=5, min_samples_leaf=1,
    class_weight="balanced", random_state=42,
)

loo = LeaveOneOut()
y_true_all, y_pred_all, y_prob_all = [], [], []

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = np.array(y)[train_idx], np.array(y)[test_idx]

    clf.fit(X_train, y_train)

    y_pred_all.append(clf.predict(X_test)[0])
    y_prob_all.append(clf.predict_proba(X_test)[0, 1])
    y_true_all.append(y_test[0])

# Metrics
print(f"\nAccuracy:  {np.mean(np.array(y_true_all) == np.array(y_pred_all)):.3f}")
print(f"F1-score:  {f1_score(y_true_all, y_pred_all):.3f}")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_true_all, y_pred_all))
print(f"\nClassification Report:")
print(classification_report(y_true_all, y_pred_all, target_names=["non-DM", "DM"]))

# ============================================================
# 4. Feature Importance (trained on full data)
# ============================================================

print("=" * 60)
print("Feature Importance Analysis")
print("=" * 60)

clf_full = RandomForestClassifier(
    n_estimators=500, max_depth=5, min_samples_leaf=1,
    class_weight="balanced", random_state=42,
)
clf_full.fit(X, y)

imp = pd.DataFrame({
    "feature": X.columns,
    "importance": clf_full.feature_importances_,
}).sort_values("importance", ascending=False)

print("\nTop-20 features:")
for i, row in imp.head(20).iterrows():
    print(f"  {row['feature']:45s} {row['importance']:.4f}")

# Permutation importance (more robust for small data)
perm_imp = permutation_importance(
    clf_full, X, y, n_repeats=30, random_state=42,
)
perm_df = pd.DataFrame({
    "feature": X.columns,
    "importance_mean": perm_imp.importances_mean,
    "importance_std": perm_imp.importances_std,
}).sort_values("importance_mean", ascending=False)

print("\nTop-20 Permutation Importance:")
for i, row in perm_df.head(20).iterrows():
    print(f"  {row['feature']:45s} {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")

# ============================================================
# 5. Visualizations
# ============================================================

OUTPUT_DIR = DATA_DIR / "analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- 5a. ROC Curve ---
fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
roc_auc = auc(fpr, tpr)

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle("Mosquito Noise (DM) Classification Analysis", fontsize=14)

# ROC
axes[0, 0].plot(fpr, tpr, "b-", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
axes[0, 0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0, 0].set_xlabel("False Positive Rate")
axes[0, 0].set_ylabel("True Positive Rate")
axes[0, 0].set_title("ROC Curve (LOO CV)")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Confusion Matrix
cm = confusion_matrix(y_true_all, y_pred_all)
axes[0, 1].matshow(cm, cmap="Blues", alpha=0.8)
for (i, j), v in np.ndenumerate(cm):
    axes[0, 1].text(j, i, str(v), ha="center", va="center", fontsize=14)
axes[0, 1].set_xticks([0, 1])
axes[0, 1].set_yticks([0, 1])
axes[0, 1].set_xticklabels(["non-DM", "DM"])
axes[0, 1].set_yticklabels(["non-DM", "DM"])
axes[0, 1].set_xlabel("Predicted")
axes[0, 1].set_ylabel("True")
axes[0, 1].set_title("Confusion Matrix")

# Feature Importance (top 10)
top10 = imp.head(10)
axes[0, 2].barh(range(len(top10)), top10["importance"].values, color="steelblue")
axes[0, 2].set_yticks(range(len(top10)))
axes[0, 2].set_yticklabels(top10["feature"].values)
axes[0, 2].invert_yaxis()
axes[0, 2].set_xlabel("Importance")
axes[0, 2].set_title("Feature Importance (Gini)")

# --- 5b. Key feature box plots ---
key_feats = [
    "center_mean_var", "center_lap_mean", "center_profile_ringing_max",
    "center_edge_strength", "ratio_profile_ringing_max", "total_high_var_count",
]
for idx, feat in enumerate(key_feats):
    ax = axes[1, idx] if idx < 3 else axes[1, idx - 3]
    vals_dm = [X.loc[i, feat] for i in range(len(y)) if y[i] == 1]
    vals_non = [X.loc[i, feat] for i in range(len(y)) if y[i] == 0]
    bp = ax.boxplot([vals_non, vals_dm], tick_labels=["non-DM", "DM"], widths=0.5,
                    patch_artist=True)
    for patch, color in zip(bp["boxes"], ["#66c2a5", "#fc8d62"]):
        patch.set_facecolor(color)
    ax.set_title(feat.replace("_", " "))
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dm_classification_analysis.png", dpi=150)
print(f"\nSaved analysis figure to: {OUTPUT_DIR / 'dm_classification_analysis.png'}")
plt.close()

# --- 5c. Detailed pairplot of top features ---
top4 = imp.head(4)["feature"].values
X_subset = X[list(top4)].copy()
X_subset["label"] = ["DM" if v == 1 else "non-DM" for v in y]

fig2, axes2 = plt.subplots(4, 4, figsize=(12, 12))
fig2.suptitle("Top-4 Feature Pairplot", fontsize=14)

colors = {"DM": "#fc8d62", "non-DM": "#66c2a5"}
for i, fi in enumerate(top4):
    for j, fj in enumerate(top4):
        ax = axes2[i, j]
        if i == j:
            for lbl in ["non-DM", "DM"]:
                subset = X_subset[X_subset["label"] == lbl]
                ax.hist(subset[fi], alpha=0.6, bins=8, color=colors[lbl], label=lbl)
            ax.legend(fontsize=6)
        else:
            for lbl in ["non-DM", "DM"]:
                subset = X_subset[X_subset["label"] == lbl]
                ax.scatter(subset[fj], subset[fi], c=colors[lbl],
                           alpha=0.8, edgecolors="k", s=40)
        if i == 3:
            ax.set_xlabel(fj.replace("_", " "), fontsize=7)
        if j == 0:
            ax.set_ylabel(fi.replace("_", " "), fontsize=7)
        ax.tick_params(labelsize=6)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dm_top_features_pairplot.png", dpi=150)
plt.close()

# ============================================================
# 6. Summary of per-sample predictions
# ============================================================

print("\n" + "=" * 60)
print("Per-Sample Predictions (LOO CV)")
print("=" * 60)
print(f"{'Idx':>4} {'(row,col)':>12} {'True':>5} {'Pred':>5} {'Prob':>6} {'Correct?':>8}")
print("-" * 45)
for i in range(len(y)):
    mark = "OK" if y_true_all[i] == y_pred_all[i] else "NO"
    print(f"{i:>4} ({meta[i][0]:>3},{meta[i][1]:>3}) "
          f"{y_true_all[i]:>5} {y_pred_all[i]:>5} "
          f"{y_prob_all[i]:.3f} {mark:>8}")

print(f"\nResults saved to: {OUTPUT_DIR}/")
