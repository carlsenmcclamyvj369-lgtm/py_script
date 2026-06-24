# 特征理论范围与归一化方式

> 本文档基于 **Single Img Tool2 v2026.06.23b** 版本的代码编写。

## 说明

- **理论最小值/最大值**：所有特征均基于 BT.709 Y 亮度（`Y = 0.2126R + 0.7152G + 0.0722B`，范围 `[0, 255]`）计算。
- **归一化方式**：使用固定缩放系数（非动态 min-max 归一化），确保不同图像间同一特征图的灰度具有可比性。
- **最终输出**：所有特征图统一 `clip(0, 255)` 后保存为 8-bit BMP。

---

## 1. 方差特征

### 计算方式

1. 对全图 Y 亮度计算 3×3 窗口均值：`mean = filter2D(Y, 3×3 均值核)`，边界反射填充。
2. 对 Y² 计算 3×3 均值：`mean_sq = filter2D(Y², 3×3 均值核)`。
3. 逐像素方差：`var = max(mean_sq - mean², 0)`（防浮点误差负值）。
4. 边界 1 像素置为 NaN（3×3 窗口不完整）。
5. 按 grid 分块（gx=gy=8），取块内有效像素的方差值进行统计。

### 特征列表

| 特征名 | 计算方式 | 理论最小值 | 理论最大值 | 归一化方式 | 归一化后范围 |
|--------|---------|-----------|-----------|-----------|-------------|
| `mean_var` | block 内所有像素方差的均值 | 0 | ~16056 | ×0.25（/4） → clip(0,255) | [0, 255] |
| `max_var` | block 内所有像素方差的最大值 | 0 | ~16056 | ×0.25（/4） → clip(0,255) | [0, 255] |
| `top5_var` | block 内方差最高的 5 个像素的均值（不足 5 个则取全部均值） | 0 | ~16056 | ×0.25（/4） → clip(0,255) | [0, 255] |
| `low_var_count` | block 内方差 `< low_var_th`（默认 100）的像素计数 | 0 | 64 | ×256 / 64 → clip(0,255) | [0, 256) |
| `high_var_count` | block 内方差 `> high_var_th`（默认 500）的像素计数 | 0 | 64 | ×256 / 64 → clip(0,255) | [0, 256) |
| `very_high_var_count` | block 内方差 `> very_high_var_th`（默认 2000）的像素计数 | 0 | 64 | ×256 / 64 → clip(0,255) | [0, 256) |

---

## 2. 残差特征

### 计算方式

1. 复用方差计算中的 3×3 均值。
2. 逐像素残差：`residual = Y - mean`（仅取内部区域，边界为 NaN）。
3. 按 grid 分块，取残差绝对值的均值和最大值。

### 特征列表

| 特征名 | 计算方式 | 理论最小值 | 理论最大值 | 归一化方式 | 归一化后范围 |
|--------|---------|-----------|-----------|-----------|-------------|
| `residual_mean` | block 内 `|Y - 3×3 均值|` 的均值 | 0 | ~255 | 不缩放，直接 clip | [0, 255] |
| `residual_max` | block 内 `|Y - 3×3 均值|` 的最大值 | 0 | ~255 | 不缩放，直接 clip | [0, 255] |

---

## 3. 拉普拉斯特征

### 计算方式

1. 对 Y 亮度用拉普拉斯核做卷积：`lap = filter2D(Y, [[0,-1,0],[-1,4,-1],[0,-1,0]])`，边界反射填充。
2. 取绝对值：`|lap|`，边界置为 NaN。
3. 按 grid 分块，取块内均值和最大值。

### 特征列表

| 特征名 | 计算方式 | 理论最小值 | 理论最大值 | 归一化方式 | 归一化后范围 |
|--------|---------|-----------|-----------|-----------|-------------|
| `lap_mean` | block 内 `|4×中心 - 上 - 下 - 左 - 右|` 的均值 | 0 | ~1020（4×255） | ×0.25（/4） | [0, ~255] |
| `lap_max` | block 内拉普拉斯响应绝对值的最大值 | 0 | ~1020 | ×0.25（/4） | [0, ~255] |

---

## 4. 梯度特征

### 计算方式

1. 对每个像素计算四方向绝对差之和：

   `grad = |Y[c] - Y[up]| + |Y[c] - Y[down]| + |Y[c] - Y[left]| + |Y[c] - Y[right]|`

2. 边界置为 NaN。
3. 按 grid 分块，取块内均值和最大值。

### 特征列表

| 特征名 | 计算方式 | 理论最小值 | 理论最大值 | 归一化方式 | 归一化后范围 |
|--------|---------|-----------|-----------|-----------|-------------|
| `grad_mean` | block 内四方向梯度之和的均值 | 0 | ~1020（4×255） | ×0.25（/4） | [0, ~255] |
| `grad_max` | block 内四方向梯度之和的最大值 | 0 | ~1020 | ×0.25（/4） | [0, ~255] |

---

## 5. 边缘方向特征

### 计算方式

1. 水平梯度（逐行差分）：`h_edge = |diff(Y, axis=1)|`，第 0 列置为 NaN。
2. 垂直梯度（逐列差分）：`v_edge = |diff(Y, axis=0)|`，第 0 行置为 NaN。
3. 按 grid 分块，取块内水平和垂直梯度的均值/最大/最小。
4. 边缘强度：`edge_strength = max(h_strength_mean, v_strength_mean)`。
5. 方向置信度：当 `edge_strength > max_strength_th`（默认 0.000001）时，`orient_conf = |H_mean - V_mean| / max(H_mean, V_mean)`；否则为 0。

### 特征列表

| 特征名 | 计算方式 | 理论最小值 | 理论最大值 | 归一化方式 | 归一化后范围 |
|--------|---------|-----------|-----------|-----------|-------------|
| `edge_strength` | `max(水平梯度均值, 垂直梯度均值)` | 0 | ~255 | 不缩放，直接 clip | [0, 255] |
| `h_strength` | block 内水平梯度 `|diff(Y, axis=1)|` 的均值 | 0 | ~255 | 不缩放，直接 clip | [0, 255] |
| `h_strength_max` | block 内水平梯度的最大值 | 0 | ~255 | 不缩放，直接 clip | [0, 255] |
| `h_strength_min` | block 内水平梯度的最小值 | 0 | ~255 | 不缩放，直接 clip | [0, 255] |
| `v_strength` | block 内垂直梯度 `|diff(Y, axis=0)|` 的均值 | 0 | ~255 | 不缩放，直接 clip | [0, 255] |
| `v_strength_max` | block 内垂直梯度的最大值 | 0 | ~255 | 不缩放，直接 clip | [0, 255] |
| `v_strength_min` | block 内垂直梯度的最小值 | 0 | ~255 | 不缩放，直接 clip | [0, 255] |
| `edge_orientation_conf` | 当 `edge_strength > th` 时：`\|H_mean - V_mean\| / max(H_mean, V_mean)`，否则 0 | 0 | 1 | ×255 | [0, 255] |

---

## 6. 振荡特征（二阶差分）

### 计算方式

1. 对 block 内每一行计算二阶差分绝对值均值：

   `row_d2[r] = mean(|Y[r, i] - 2×Y[r, i+1] + Y[r, i+2]|)`，遍历所有列。

2. 对 block 内每一列做同样的计算。
3. 各统计量：取所有行的均值/最大/最小、所有列的均值/最大/最小。
4. `second_diff_max = max(row_energy_mean, col_energy_mean)`。
5. `second_diff_min_max = min(row_energy_mean, col_energy_mean) / max(...)`，天然 [0, 1] 的比值。

### 特征列表

| 特征名 | 计算方式 | 理论最小值 | 理论最大值 | 归一化方式 | 归一化后范围 |
|--------|---------|-----------|-----------|-----------|-------------|
| `row_second_diff` | block 内每行二阶差分绝对值均值的均值 | 0 | ~510（2×255） | ×0.5（/2） | [0, ~255] |
| `row_second_diff_max` | block 内各行二阶差分均值中的最大值 | 0 | ~510 | ×0.5（/2） | [0, ~255] |
| `row_second_diff_min` | block 内各行二阶差分均值中的最小值 | 0 | ~510 | ×0.5（/2） | [0, ~255] |
| `col_second_diff` | block 内每列二阶差分绝对值均值的均值 | 0 | ~510 | ×0.5（/2） | [0, ~255] |
| `col_second_diff_max` | block 内各列二阶差分均值中的最大值 | 0 | ~510 | ×0.5（/2） | [0, ~255] |
| `col_second_diff_min` | block 内各列二阶差分均值中的最小值 | 0 | ~510 | ×0.5（/2） | [0, ~255] |
| `second_diff_max` | `max(row_second_diff, col_second_diff)` | 0 | ~510 | ×0.5（/2） | [0, ~255] |
| `second_diff_min_max` | `min(row, col) / max(row, col)`（除零保护） | 0 | 1 | ×255 | [0, 255] |

---

## 7. 振铃轮廓特征

### 计算方式

振铃评分基于对一维轮廓（block 的每一行和每一列）的 3 个子项评分：

#### 子项 1：动态范围评分 `dyn_score`

`dyn = max(v) - min(v)`，经 `_norm(x, dyn_lo=20, dyn_hi=120)` 归一化到 [0, 1]：

```
dyn_score = clip((dyn - dyn_lo) / (dyn_hi - dyn_lo), 0, 1)
```

#### 子项 2：二阶差分能量评分 `d2_score`

`d2_energy = mean(|v[i] - 2×v[i+1] + v[i+2]|)`，经 `_norm(x, d2_lo=5, d2_hi=60)` 归一化：

```
d2_score = clip((d2_energy - d2_lo) / (d2_hi - d2_lo), 0, 1)
```

#### 子项 3：梯度符号变化评分 `sign_score`

1. 计算一阶差分 `d = diff(v)`。
2. 取符号 `sign(d)`，绝对值 `< eps`（默认 3.0）的视为平坦（符号记为 0）。
3. 去除符号为零的值，统计剩余符号序列的正负变化次数 `sign_changes`。
4. 经 `_norm(x, sign_lo=1, sign_hi=4)` 归一化：

```
sign_score = clip((sign_changes - sign_lo) / (sign_hi - sign_lo), 0, 1)
```

#### 加权组合

```
ringing_score = dyn_ratio×dyn_score + d2_ratio×d2_score + sign_ratio×sign_score
```

默认权重：`dyn_ratio=0.45, d2_ratio=0.35, sign_ratio=0.20`（可在 UI 中自定义）。

#### 块级统计

对 block 的每一行和每一列分别计算 `ringing_score`，然后：

| 特征名 | 计算方式 |
|--------|---------|
| `row_ringing_max` | 各行评分的最大值 |
| `row_ringing_min` | 各行评分的最小值 |
| `row_ringing_mean` | 各行评分的均值 |
| `col_ringing_max` | 各列评分的最大值 |
| `col_ringing_min` | 各列评分的最小值 |
| `col_ringing_mean` | 各列评分的均值 |
| `ringing_mean_max` | `max(row_mean, col_mean)` |
| `ringing_mean_min` | `min(row_mean, col_mean)` |
| `ringing_mean_min_max` | `min(row_mean, col_mean) / max(row_mean, col_mean)`（除零保护） |
| `profile_ringing_max` | 所有行+列评分中的最大值 |
| `profile_ringing_mean` | 所有行+列评分的均值 |
| `row_ringing_dyn_score` | 各行 dyn_score 的均值 |
| `row_ringing_d2_score` | 各行 d2_score 的均值 |
| `row_ringing_sign_score` | 各行 sign_score 的均值 |
| `col_ringing_dyn_score` | 各列 dyn_score 的均值 |
| `col_ringing_d2_score` | 各列 d2_score 的均值 |
| `col_ringing_sign_score` | 各列 sign_score 的均值 |

### 特征列表

| 特征名 | 理论最小值 | 理论最大值 | 归一化方式 | 归一化后范围 |
|--------|-----------|-----------|-----------|-------------|
| `profile_ringing_max` | 0 | 1 | ×255 | [0, 255] |
| `profile_ringing_mean` | 0 | 1 | ×255 | [0, 255] |
| `ringing_mean_max` | 0 | 1 | ×255 | [0, 255] |
| `ringing_mean_min` | 0 | 1 | ×255 | [0, 255] |
| `ringing_mean_min_max` | 0 | 1 | ×255 | [0, 255] |
| `row_ringing_max` | 0 | 1 | ×255 | [0, 255] |
| `row_ringing_min` | 0 | 1 | ×255 | [0, 255] |
| `row_ringing_mean` | 0 | 1 | ×255 | [0, 255] |
| `row_ringing_dyn_score` | 0 | 1 | ×255 | [0, 255] |
| `row_ringing_d2_score` | 0 | 1 | ×255 | [0, 255] |
| `row_ringing_sign_score` | 0 | 1 | ×255 | [0, 255] |
| `col_ringing_max` | 0 | 1 | ×255 | [0, 255] |
| `col_ringing_min` | 0 | 1 | ×255 | [0, 255] |
| `col_ringing_mean` | 0 | 1 | ×255 | [0, 255] |
| `col_ringing_dyn_score` | 0 | 1 | ×255 | [0, 255] |
| `col_ringing_d2_score` | 0 | 1 | ×255 | [0, 255] |
| `col_ringing_sign_score` | 0 | 1 | ×255 | [0, 255] |

---

## 8. 滤波差值（仅左键单击信息面板显示，不生成特征图）

### 计算方式

1. 将 Y 亮度取整 clip 到 uint8：`Y_u8 = clip(round(Y), 0, 255).astype(uint8)`。
2. 分别用高斯滤波和双边滤波对 Y_u8 做 3×3、5×5、7×7 核滤波。
3. 将滤波结果转回 float64，计算尺度间绝对差值：

   ```
   gauss_diffs = [|Y - g3|, |g3 - g5|, |g5 - g7|]
   bilat_diffs = [|Y - b3|, |b3 - b5|, |b5 - b7|]
   ```

| 差值类型 | 含义 |
|---------|------|
| `abs(orig - gauss_3x3)` | 原图与高斯 3×3 的差异 |
| `abs(gauss_3x3 - gauss_5x5)` | 高斯 3×3 与 5×5 的差异 |
| `abs(gauss_5x5 - gauss_7x7)` | 高斯 5×5 与 7×7 的差异 |
| `abs(orig - bilat_3x3)` | 原图与双边 3×3 的差异 |
| `abs(bilat_3x3 - bilat_5x5)` | 双边 3×3 与 5×5 的差异 |
| `abs(bilat_5x5 - bilat_7x7)` | 双边 5×5 与 7×7 的差异 |

