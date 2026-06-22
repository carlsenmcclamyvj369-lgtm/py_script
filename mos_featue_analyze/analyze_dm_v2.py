"""
Improved DM classifier — targets 33/33 DM detection.

Key additions over v1:
  - Asymmetry features: center-to-max-neighbor ratio (catches "quiet center + active halo")
  - Row-split features: top-3 vs bottom-3, left-3 vs right-3
  - "Halo" detection: center is quiet but surround is active
  - Threshold tuning for perfect DM recall
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score

DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. Load & group into samples (9 rows per sample)
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


print("=" * 60)
print("IMPROVED DM CLASSIFIER v2")
print("=" * 60)

all_samples = []
for csv_path in sorted(DATA_DIR.glob("*.csv")):
    name = csv_path.stem.lower()
    if "not_dm" in name:
        all_samples.extend(load_samples(csv_path, label=0))
    elif "dm" in name:
        all_samples.extend(load_samples(csv_path, label=1))

print(f"\nLoaded {len(all_samples)} samples: "
      f"{sum(1 for s in all_samples if s['label']==1)} DM, "
      f"{sum(1 for s in all_samples if s['label']==0)} non-DM")

# ============================================================
# 2. Feature Engineering (v2 — enhanced)
# ============================================================

FEATURE_NAMES = []

def add_feat(f, name, value):
    FEATURE_NAMES.append(name)
    f[name] = value

def extract_features_v2(sample):
    g = sample["group"]
    center = g.loc[4]
    f = {}

    # ---- A. Center-point raw features (same as v1) ----
    for k in ["mean_var", "max_var", "top5_var", "median_var",
              "high_var_count", "very_high_var_count",
              "residual_mean", "residual_max",
              "lap_mean", "lap_max", "grad_mean", "grad_max",
              "edge_strength", "edge_orientation_conf",
              "row_second_diff", "col_second_diff",
              "profile_ringing_max", "profile_ringing_mean",
              "row_ringing_dyn_score", "row_ringing_d2_score", "row_ringing_sign_score",
              "col_ringing_dyn_score", "col_ringing_d2_score", "col_ringing_sign_score"]:
        add_feat(f, f"center_{k}", center[k])

    # ---- B. Neighborhood statistics ----
    for k in ["mean_var", "max_var", "lap_mean", "lap_max",
              "edge_strength", "edge_orientation_conf",
              "profile_ringing_max", "profile_ringing_mean",
              "residual_mean", "residual_max", "grad_mean", "grad_max",
              "high_var_count", "very_high_var_count"]:
        vals = g[k].values
        add_feat(f, f"n_mean_{k}", float(np.mean(vals)))
        add_feat(f, f"n_std_{k}", float(np.std(vals)))
        add_feat(f, f"n_max_{k}", float(np.max(vals)))

    # ---- C. NEW: Center-to-MAX ratio (asymmetry) ----
    # Instead of center/neighbor_mean, use center/neighbor_max.
    # This catches "quiet center but at least one neighbor is extremely active".
    eps = 1e-8
    for k in ["mean_var", "max_var", "lap_mean", "lap_max",
              "edge_strength", "profile_ringing_max", "residual_mean", "grad_mean"]:
        neighbor_max = float(np.max(g[k].values))
        add_feat(f, f"center_to_max_{k}", center[k] / (neighbor_max + eps))

    # ---- D. NEW: Row-split features (top-3 vs bottom-3, left-3 vs right-3) ----
    # The 3x3 block is indexed:
    #   0 1 2  (top row)
    #   3 4 5  (middle row, center at 4)
    #   6 7 8  (bottom row)
    top3 = g.iloc[0:3]
    bot3 = g.iloc[6:9]
    left3 = g.iloc[[0, 3, 6]]   # col 0 of each row
    right3 = g.iloc[[2, 5, 8]]  # col 2 of each row

    for k in ["mean_var", "lap_mean", "profile_ringing_max", "edge_strength"]:
        add_feat(f, f"top_bot_ratio_{k}",
                 (float(np.mean(top3[k].values)) + eps) /
                 (float(np.mean(bot3[k].values)) + eps))
        add_feat(f, f"left_right_ratio_{k}",
                 (float(np.mean(left3[k].values)) + eps) /
                 (float(np.mean(right3[k].values)) + eps))

    # ---- E. NEW: "Halo" detection features ----
    # Center is quiet but surround is active → strong mosquito signal
    for k in ["mean_var", "lap_mean", "residual_max", "profile_ringing_max"]:
        center_val = center[k]
        surround_max = float(np.max(np.concatenate([top3[k].values, bot3[k].values,
                                                     left3[k].values, right3[k].values])))
        surround_mean = float(np.mean(np.concatenate([top3[k].values, bot3[k].values,
                                                       left3[k].values, right3[k].values])))
        # "Halo strength": how much more active the surround is vs center
        add_feat(f, f"halo_strength_{k}",
                 max(0, surround_mean - center_val) / (surround_mean + eps))
        # "Max halo": strongest single neighbor vs center
        add_feat(f, f"halo_max_{k}",
                 max(0, surround_max - center_val) / (surround_max + eps))

    # ---- F. Aggregate context (same as v1) ----
    add_feat(f, "total_hvc", int(np.sum(g["high_var_count"].values > 0)))
    add_feat(f, "total_vhvc", int(np.sum(g["very_high_var_count"].values > 0)))

    ringing_vals = g["profile_ringing_max"].values
    add_feat(f, "ringing_std", float(np.std(ringing_vals)))
    add_feat(f, "ringing_center_minus_mean",
             center["profile_ringing_max"] - float(np.mean(ringing_vals)))

    orient_vals = g["edge_orientation_conf"].values
    add_feat(f, "orient_consistency", float(np.std(orient_vals)))

    return f


print("\nExtracting features (v2)...")
X_rows, y, meta = [], [], []
for s in all_samples:
    feat = extract_features_v2(s)
    X_rows.append(feat)
    y.append(s["label"])
    meta.append((s["source"], s["center_row"], s["center_col"]))

X = pd.DataFrame(X_rows)
y = np.array(y)
print(f"  Feature matrix: {X.shape}")

# ============================================================
# 3. Train & Evaluate — find best config
# ============================================================

best_result = None
best_fn = 999

configs = [
    # (name, max_depth, n_est, class_weight)
    ("RF depth=5, balanced",       5,  200, "balanced"),
    ("RF depth=8, balanced",       8,  300, "balanced"),
    ("RF depth=8, DM=3:1",         8,  300, {0: 1, 1: 3}),
    ("RF depth=10, balanced",     10,  300, "balanced"),
    ("RF depth=10, DM=3:1",       10,  300, {0: 1, 1: 3}),
    ("RF depth=10, DM=5:1",       10,  300, {0: 1, 1: 5}),
    ("RF depth=None, DM=3:1",     None, 500, {0: 1, 1: 3}),
    ("RF depth=None, balanced",   None, 500, "balanced"),
]

print(f"\n{'='*70}")
print(f"{'Config':<35s} {'Acc':>6} {'F1':>6} {'FN':>4} {'FP':>4}")
print(f"{'='*70}")

all_cv_results = []

for name, md, ne, cw in configs:
    clf = RandomForestClassifier(
        n_estimators=ne, max_depth=md, min_samples_leaf=1,
        class_weight=cw, random_state=42,
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
    ypr = np.array(ypr)
    cm = confusion_matrix(yt, yp)
    fn, fp = cm[1][0], cm[0][1]
    acc = np.mean(yt == yp)
    f1 = f1_score(yt, yp)
    print(f"{name:<35s} {acc:>6.3f} {f1:>6.3f} {fn:>4} {fp:>4}")

    all_cv_results.append((name, yt, yp, ypr, cm, fn, fp))

    if fn < best_fn or (fn == best_fn and fp < (best_result[6] if best_result else 999)):
        best_result = (name, yt, yp, ypr, cm, fn, fp)
        best_fn = fn

# ============================================================
# 4. Threshold sweep on best config
# ============================================================

print(f"\n{'='*70}")
bname, byt, byp, bypr, bcm, bfn, bfp = best_result
print(f"Best config: {bname} (FN={bfn}, FP={bfp})")
print(f"{'='*70}")

print(f"\nThreshold sweep for FN=0 on best config:")
best_th = None
for th in np.arange(0.01, 0.60, 0.01):
    pred_t = (bypr >= th).astype(int)
    cm_t = confusion_matrix(byt, pred_t)
    if cm_t[1][0] == 0:  # FN=0
        print(f"  th={th:.2f}: FN={cm_t[1][0]} FP={cm_t[0][1]} "
              f"DM={cm_t[1][1]}/33 non-DM={cm_t[0][0]}/36 acc={np.mean(byt==pred_t):.3f}")
        if best_th is None:
            best_th = th

if best_th is None:
    print("  No threshold achieves FN=0 with this config.")
else:
    print(f"\n  >>> Recommend: threshold={best_th:.2f} <<<")

# Also try combining best config with best threshold
print(f"\n{'='*70}")
print("Combined: Best config + Best Threshold")
print(f"{'='*70}")
final_pred = (bypr >= best_th).astype(int) if best_th else byp
final_cm = confusion_matrix(byt, final_pred)
print(final_cm)
print(classification_report(byt, final_pred, target_names=["non-DM", "DM"]))
print(f"\nFN={final_cm[1][0]}, FP={final_cm[0][1]}, "
      f"DM recall={final_cm[1][1]}/33, non-DM specificity={final_cm[0][0]}/36")

# ============================================================
# 5. Feature importance
# ============================================================

clf_full = RandomForestClassifier(
    n_estimators=500, max_depth=10, class_weight={0: 1, 1: 3}, random_state=42,
)
clf_full.fit(X, y)

imp = pd.DataFrame({
    "feature": X.columns,
    "importance": clf_full.feature_importances_,
}).sort_values("importance", ascending=False)

print(f"\n{'='*70}")
print("Top-20 Features (v2)")
print(f"{'='*70}")
for _, row in imp.head(20).iterrows():
    print(f"  {row['feature']:45s} {row['importance']:.4f}")

# Show which NEW features (v2 additions) rank well
v2_features = [c for c in imp["feature"] if any(kw in c for kw in
    ["center_to_max", "top_bot_", "left_right_", "halo_", "total_hvc"])]
print(f"\nNew v2 features in top-30:")
for _, row in imp[imp["feature"].isin(v2_features)].head(30).iterrows():
    print(f"  [{row['importance']:.4f}] {row['feature']}")

# ============================================================
# 6. Visualizations
# ============================================================

# Confusion matrix comparison: baseline (th=0.5) vs tuned
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle(f"Improved DM Classifier — {bname}", fontsize=12)

for idx, (th, title) in enumerate([(0.5, f"Default th=0.5 (FN={bfn})"),
                                    (best_th, f"Tuned th={best_th:.2f} (FN=0)")]):
    p = (bypr >= th).astype(int)
    cm = confusion_matrix(byt, p)
    ax = axes[idx]
    ax.matshow(cm, cmap="Blues", alpha=0.8)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=16, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["non-DM", "DM"])
    ax.set_yticklabels(["non-DM", "DM"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dm_v2_confusion.png", dpi=150)
print(f"\nSaved: {OUTPUT_DIR / 'dm_v2_confusion.png'}")

# Feature importance plot (top 15)
top15 = imp.head(15)
fig2, ax2 = plt.subplots(figsize=(8, 6))
colors = ["#e74c3c" if any(kw in f for kw in ["center_to_max", "halo", "top_bot", "left_right"])
          else "#3498db" for f in top15["feature"]]
ax2.barh(range(len(top15)), top15["importance"], color=colors)
ax2.set_yticks(range(len(top15)))
ax2.set_yticklabels(top15["feature"].values, fontsize=8)
ax2.invert_yaxis()
ax2.set_xlabel("Importance")
ax2.set_title("Feature Importance (red = v2 new features)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dm_v2_feature_importance.png", dpi=150)
print(f"Saved: {OUTPUT_DIR / 'dm_v2_feature_importance.png'}")
plt.close("all")

# ============================================================
# 7. Per-sample result with best config
# ============================================================

print(f"\n{'='*70}")
print("Per-Sample Predictions (best config + tuned threshold)")
print(f"{'='*70}")
print(f"{'Idx':>4} {'Source':>20} {'Pos':>10} {'True':>5} {'Pred':>5} {'Prob':>6}")
print("-" * 55)
for i in range(len(y)):
    final_p = 1 if bypr[i] >= best_th else 0
    print(f"{i:>4} {meta[i][0]:>20} ({meta[i][1]:>3},{meta[i][2]:>3}) "
          f"{y[i]:>5} {final_p:>5} {bypr[i]:.3f}")
