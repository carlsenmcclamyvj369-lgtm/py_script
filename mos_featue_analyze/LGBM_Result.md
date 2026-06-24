# LGBM 训练结果

## 数据概况

- **总样本**: 20372 (dm: 9960, not_dm: 10412)
- **训练集**: 14516 行 (9 个视频文件夹)
- **测试集**: 5856 行 (4 个视频文件夹)
- **特征数**: 45
- **切分方式**: 按文件夹分层切分，避免数据泄漏

## 模型参数 (mode 2 grid search)

| 参数 | 值 |
|---|---|
| n_estimators | 130 |
| learning_rate | 0.1 |
| max_depth | 8 |
| num_leaves | 31 |
| reg_alpha | 0.1 |
| reg_lambda | 1.0 |

## 分类结果

### 训练集

```
[[6188  184]        TN  FP
 [ 230 5010]]       FN  TP
```

| 指标 | 值 |
|---|---|
| 准确率 | 96.43% |
| 精确率 | 96.46% |
| 召回率 | 95.61% |
| R² | 0.856 |

### 测试集

```
[[1464  130]        TN  FP
 [ 125 1185]]       FN  TP
```

| 指标 | 值 |
|---|---|
| 准确率 | **91.22%** |
| 精确率 | **90.11%** |
| 召回率 | **90.46%** |
| R² | 0.645 |

## 特征重要性

| 排名 | 特征 | 重要性 |
|---|---|---|
| 1 | max_var | 235 |
| 2 | h_strength_max | 183 |
| 3 | grad_max | 179 |
| 4 | mean_var | 172 |
| 5 | top5_var | 157 |
| 6 | v_strength_max | 155 |
| 7 | lap_max | 144 |
| 8 | low_var_count | 142 |
| 9 | high_var_count | 134 |
| 10 | row_second_diff_min | 134 |
| 11 | residual_max | 118 |
| 12 | col_ringing_dyn_score | 117 |
| 13 | col_ringing_sign_score | 115 |
| 14 | col_second_diff_min | 113 |
| 15 | row_second_diff_max | 106 |
| 16 | row_ringing_sign_score | 101 |
| 17 | lap_mean | 98 |
| 18 | h_strength_min | 91 |
| 19 | v_strength_min | 86 |
| 20 | col_second_diff_max | 80 |
| 21 | edge_orientation_conf | 80 |
| 22 | second_diff_min_max | 78 |
| 23 | row_ringing_dyn_score | 73 |
| 24 | v_strength | 72 |
| 25 | row_second_diff | 69 |
| 26 | edge_strength | 68 |
| 27 | very_high_var_count | 66 |
| 28 | second_diff_max | 63 |
| 29 | h_strength | 60 |
| 30 | ringing_mean_min_max | 58 |
| 31 | col_ringing_max | 53 |
| 32 | ringing_mean_max | 52 |
| 33 | residual_mean | 49 |
| 34 | grad_mean | 48 |
| 35 | col_second_diff | 48 |
| 36 | row_ringing_min | 41 |
| 37 | col_ringing_min | 39 |
| 38 | profile_ringing_max | 33 |
| 39 | col_ringing_mean | 32 |
| 40 | ringing_mean_min | 30 |
| 41 | profile_ringing_mean | 29 |
| 42 | row_ringing_max | 28 |
| 43 | col_ringing_d2_score | 21 |
| 44 | row_ringing_d2_score | 20 |
| 45 | row_ringing_mean | 19 |

### Top 10 特征分析

| 特征 | 类别 | 说明 |
|---|---|---|
| max_var | 方差 | 局部区域最大方差，纹理复杂度 |
| h_strength_max | 水平强度 | 水平方向最大强度，边缘检测 |
| grad_max | 梯度 | 最大梯度值，锐利度 |
| mean_var | 方差 | 局部区域平均方差 |
| top5_var | 方差 | 前 5 大方差均值 |
| v_strength_max | 垂直强度 | 垂直方向最大强度 |
| lap_max | 拉普拉斯 | 最大拉普拉斯值，边缘强度 |
| low_var_count | 方差 | 低方差像素计数，平坦区域 |
| high_var_count | 方差 | 高方差像素计数，纹理区域 |
| row_second_diff_min | 二阶差分 | 行方向最小二阶差分，条带检测 |

主导特征集中在 **方差类 (var)** 和 **边缘强度类 (strength/grad/lap)**，说明模型主要依赖纹理复杂度和边缘信息来区分 dm / not_dm。
