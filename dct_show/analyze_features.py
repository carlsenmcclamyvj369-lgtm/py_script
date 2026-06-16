import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent

# =========================================================
# 1. 读取 CSV (纯 numpy)
# =========================================================
FEATURE_NAMES = [
    'num_edge_pixels', 'edge_residual_mean', 'non_edge_residual_mean',
    'edge_localization_score', 'num_valid_profiles', 'patch_score',
    'mean_oscillation_score', 'mean_decay_score', 'mean_residual_energy',
    'mean_sign_alternation_score', 'mean_decay_ratio_near_far',
    'mean_energy_weighted_distance', 'mean_gibbs_profile_score',
    'median_gibbs_profile_score', 'p90_gibbs_profile_score'
]

def load_csv(path):
    """返回 (features, rows, cols), 过滤掉 no_valid_edge 的样本"""
    # 列: 0:x 1:y 2:w 3:h 4~18:features(15个) 19:patch_label(跳过) 20:patch_row 21:patch_col
    raw = np.loadtxt(path, delimiter=',', skiprows=1, dtype=np.float32,
                     usecols=list(range(4, 19)) + list(range(20, 22)))
    feats = raw[:, :15]   # 15个特征
    rows = raw[:, 15].astype(int)
    cols = raw[:, 16].astype(int)
    return feats, rows, cols

feats_02, rows_02, cols_02 = load_csv(BASE / 'patch_8x8_predictions_02.csv')
feats_12, rows_12, cols_12 = load_csv(BASE / 'patch_8x8_predictions_12.csv')

print(f"02: {feats_02.shape}, 12: {feats_12.shape}")
print(f"02 rows: {rows_02.min()}~{rows_02.max()}, cols: {cols_02.min()}~{cols_02.max()}")
print(f"12 rows: {rows_12.min()}~{rows_12.max()}, cols: {cols_12.min()}~{cols_12.max()}")

# =========================================================
# 2. 特征分布对比 (02 vs 12) — boxplot
# =========================================================
n_feats = feats_02.shape[1]
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
axes = axes.ravel()

for i in range(n_feats):
    ax = axes[i]
    data_02 = feats_02[:, i]
    data_12 = feats_12[:, i]
    # 跳过全零特征（没有边缘点的patch）
    valid_02 = data_02[data_02 != 0]
    valid_12 = data_12[data_12 != 0]
    bp = ax.boxplot([valid_02, valid_12], labels=['02', '12'], widths=0.5, patch_artist=True)
    bp['boxes'][0].set_facecolor('#4ECDC4')
    bp['boxes'][1].set_facecolor('#FF6B6B')
    ax.set_title(FEATURE_NAMES[i], fontsize=10)
    ax.tick_params(labelsize=8)
    # 标注均值
    ax.plot(1, np.mean(valid_02), 'D', color='darkblue', markersize=5)
    ax.plot(2, np.mean(valid_12), 'D', color='darkred', markersize=5)

plt.suptitle('Feature Distribution: config_02 vs config_12 (nonzero patches only)', fontsize=14)
plt.tight_layout()
plt.savefig(BASE / 'feature_distribution_comparison.png', dpi=150)
print("已保存 feature_distribution_comparison.png")

# =========================================================
# 3. 特征相关性矩阵 (12数据集)
# =========================================================
# 只取有边缘的patch
valid_mask_12 = feats_12[:, 0] > 0  # num_edge_pixels > 0
feats_valid_12 = feats_12[valid_mask_12]
corr = np.corrcoef(feats_valid_12.T)

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(n_feats))
ax.set_yticks(range(n_feats))
ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(FEATURE_NAMES, fontsize=8)
plt.colorbar(im, shrink=0.8)
plt.title('Feature Correlation Matrix (config_12, valid patches only)', fontsize=12)
plt.tight_layout()
plt.savefig(BASE / 'feature_correlation_matrix.png', dpi=150)
print("已保存 feature_correlation_matrix.png")

# =========================================================
# 4. 特征 vs patch_score 相关性 (影响力度量)
# =========================================================
patch_score_idx = 5  # patch_score 在第6列 (index 5)
scores_12 = feats_valid_12[:, patch_score_idx]
corr_with_score = []
for i in range(n_feats):
    if i == patch_score_idx:
        corr_with_score.append(0)
        continue
    c = np.corrcoef(feats_valid_12[:, i], scores_12)[0, 1]
    corr_with_score.append(abs(c))

rank_idx = np.argsort(corr_with_score)[::-1]

fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#4ECDC4' if i == patch_score_idx else '#FF6B6B' for i in range(n_feats)]
bars = ax.barh(range(n_feats), [corr_with_score[i] for i in rank_idx], color=[colors[i] for i in rank_idx])
ax.set_yticks(range(n_feats))
ax.set_yticklabels([FEATURE_NAMES[i] for i in rank_idx], fontsize=9)
ax.set_xlabel('|Correlation with patch_score|')
ax.set_title('Feature Importance: Absolute Correlation with patch_score\n(config_12, valid patches)', fontsize=12)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(BASE / 'feature_importance_correlation.png', dpi=150)
print("已保存 feature_importance_correlation.png")

# =========================================================
# 5. Top-4 特征散点图 (patch 级，按 patch_score 着色)
# =========================================================
# 取与 score 相关性最高的4个特征 (排除 patch_score 自身)
top4_idx = [i for i in rank_idx if i != patch_score_idx][:4]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

# 随机采样 2000 个有效 patch 避免图太密
rng = np.random.RandomState(42)
n_valid = feats_valid_12.shape[0]
sample_idx = rng.choice(n_valid, min(2000, n_valid), replace=False)

for idx_in_sub, (fi, fj) in enumerate([(top4_idx[0], top4_idx[1]),
                                        (top4_idx[0], top4_idx[2]),
                                        (top4_idx[0], top4_idx[3]),
                                        (top4_idx[1], top4_idx[2])]):
    ax = axes[idx_in_sub]
    sc = ax.scatter(feats_valid_12[sample_idx, fi], feats_valid_12[sample_idx, fj],
                    c=scores_12[sample_idx], cmap='viridis', alpha=0.5, s=8)
    ax.set_xlabel(FEATURE_NAMES[fi], fontsize=9)
    ax.set_ylabel(FEATURE_NAMES[fj], fontsize=9)
    plt.colorbar(sc, ax=ax, label='patch_score')

plt.suptitle('Top Features Scatter (config_12 valid patches, colored by patch_score)', fontsize=13)
plt.tight_layout()
plt.savefig(BASE / 'feature_scatter_top4.png', dpi=150)
print("已保存 feature_scatter_top4.png")

# =========================================================
# 6. 02 vs 12 特征差异最大的维度
# =========================================================
# 对每个特征，计算02和12的均值差（effect size）
valid_02 = feats_02[feats_02[:, 0] > 0]
valid_12 = feats_12[feats_12[:, 0] > 0]
mean_diff = []
for i in range(n_feats):
    m02 = np.mean(valid_02[:, i])
    m12 = np.mean(valid_12[:, i])
    pooled_std = np.sqrt((np.std(valid_02[:, i])**2 + np.std(valid_12[:, i])**2) / 2)
    effect = abs(m02 - m12) / max(pooled_std, 1e-8)
    mean_diff.append(effect)

diff_rank = np.argsort(mean_diff)[::-1]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(n_feats), [mean_diff[i] for i in diff_rank], color='#6C5CE7')
ax.set_yticks(range(n_feats))
ax.set_yticklabels([FEATURE_NAMES[i] for i in diff_rank], fontsize=9)
ax.set_xlabel('Effect Size |mean_02 - mean_12| / pooled_std')
ax.set_title('Features Most Different Between config_02 and config_12', fontsize=12)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(BASE / 'feature_diff_02_vs_12.png', dpi=150)
print("已保存 feature_diff_02_vs_12.png")

print("\n===== 特征重要性排序 (与patch_score的|相关系数|) =====")
for i in range(n_feats):
    idx = rank_idx[i]
    if idx == patch_score_idx:
        continue
    print(f"  {i+1}. {FEATURE_NAMES[idx]}: {corr_with_score[idx]:.4f}")

print("\n===== 02 vs 12 差异最大的特征 =====")
for i in range(n_feats):
    idx = diff_rank[i]
    m02 = np.mean(valid_02[:, idx])
    m12 = np.mean(valid_12[:, idx])
    print(f"  {i+1}. {FEATURE_NAMES[idx]}: effect={mean_diff[idx]:.3f}  (02={m02:.4f}, 12={m12:.4f})")
