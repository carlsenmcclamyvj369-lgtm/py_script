# AI-based Mosquito Noise Detection (CNN-DM)

## 项目概述

将蚊式噪声（Mosquito Noise / DM）检测从传统的规则阈值方法升级为基于轻量级 CNN 的 AI 检测方案。项目包含完整的训练、推理、量化部署流程。

---

## 目录结构

```
CNN_DM/
├── dm_cnn.py                          # 训练脚本：模型定义、Dataset、训练循环
├── predict_cnn.py                     # 推理脚本：特征计算、CNN 推理、结果可视化
├── predict_cnn_old.py                 # 旧版推理（float 特征，用于对比验证）
├── dm_cnn_old.py                      # 旧版模型（对应 predict_cnn_old.py）
├── feature_compute_reference.py       # 特征计算参考实现（新版 int 版用）
├── feature_compute_reference_old.py   # 特征计算参考实现（旧版 float 版用）
├── train_qat_test.py                  # QAT 量化感知训练
├── validate_features.py               # 特征对比验证脚本
├── feature_validation_report.md       # 特征验证报告
├── model/                             # 训练好的模型权重
│   ├── mosquito_denoise_cnn_cost_down_grid_8.pth
│   ├── mosquito_denoise_cnn_cost_down_grid_16.pth
│   └── mosquito_denoise_cnn_qat_grid_*.pth
├── test_data/                         # 测试图片
├── grid_8_dataset_paths.txt           # GS=8 数据集路径列表
├── MNR_Label_GS8_20260727_2007/       # GS=8 训练数据目录
└── predictions_*/                     # 推理输出目录
```

---

## 模型架构

4 层 CNN（无 BN，硬件友好）：

```
Input:  (16, 9, 9)        ← 16 个手工特征，9x9 patch
  ├─ Conv2d(16→32, k=3) + ReLU + Clip(0,1)
  ├─ Conv2d(32→16, k=3) + ReLU + Clip(0,1)
  ├─ Conv2d(16→16, k=3) + ReLU + Clip(0,1)
  ├─ Conv2d(16→1,  k=3)
  └─ Sigmoid → output (dm_probability)
```

两个变体：

| 变体 | 说明 |
|------|------|
| `cost_down=True` (默认) | 无 BN，ReLU + Clip，适合硬件部署 |
| `cost_down=False` | 有 BN，原始 Sigmoid，训练更稳定 |

---

## 16 个手工特征

| # | 名称 | 说明 | 归一化 |
|---|------|------|--------|
| 1 | mean_var | block 内方差均值 | /255 |
| 2 | low_var_count | 低方差像素计数 | clamp(0,255)/255 |
| 3 | high_var_count | 高方差像素计数 | clamp(0,255)/255 |
| 4 | edge_strength | 边缘强度 (H/V max) | clamp(0,255)/255 |
| 5 | edge_orientation_conf | 边缘方向置信度 | clamp(0,255)/255 |
| 6 | second_diff_max | 二阶差分最大值 | clamp(0,255)/255 |
| 7 | second_diff_min_max | 二阶差分比值 | clamp(0,255)/255 |
| 8-13 | ringing 特征 | 振铃检测 6 维特征 | clamp(0,255)/255 |
| 14-15 | row/col_diff_max | 行列差分最大值 | clamp(0,255)/255 |

> 新版本（predict_cnn.py）所有特征统一整数化到 [0,255] 再 /255 归一化，便于硬件部署。
> 旧版本（predict_cnn_old.py）每个特征使用独立的 float 归一化除数。

---

## 训练

### 用法

```bash
# 训练 GS=8（默认 20 epoch）
python -c "from dm_cnn import *; train(gs=8, cost_down=True, epochs=400)"

# 训练 GS=16
python -c "from dm_cnn import *; train(gs=16, cost_down=True, epochs=400)"

# 依次训练 GS8 + GS16
python dm_cnn.py
```

### 训练流程

1. 读取 `grid_*_dataset_paths.txt` 指向的 CSV 特征文件
2. `MosquitoPatchDataset` 将每 81 行 (9x9 patch) 组装为一个样本
3. 16 个特征归一化到 [0,1]
4. 4 层 CNN 训练，BCEWithLogitsLoss
5. 每 epoch 验证集计算 F1，保存最佳模型

### 输出

```
model/
├── mosquito_denoise_cnn_cost_down_grid_8.pth
├── mosquito_denoise_cnn_cost_down_grid_16.pth
├── best_th_cost_down_grid_8.npy      # 最佳推理阈值
└── best_th_cost_down_grid_16.npy
```

---

## 推理

### Preset 配置系统

```python
PRESETS = {
    "fp32_gs8":  Preset(name="fp32_gs8",  gs=8,  cost_down=True, is_qat=False),
    "fp32_gs16": Preset(name="fp32_gs16", gs=16, cost_down=True, is_qat=False),
    "qat_gs8":   Preset(name="qat_gs8",   gs=8,  cost_down=True, is_qat=True),
    "qat_gs16":  Preset(name="qat_gs16",  gs=16, cost_down=True, is_qat=True),
}
```

### 用法

```bash
# 跑所有 preset（默认 test_data）
python predict_cnn.py

# 指定数据集目录
python -c "from predict_cnn import *; main(test_dir='./SR_Data')"

# 单条跑
python -c "from predict_cnn import *; run_preset(PRESETS['fp32_gs8'], test_dir='./SR_Data')"
```

### 推理流程

1. 读取 BMP → Y 通道提取
2. `compute_grid_features()` → 16 维特征图 (gh, gw, 16)
3. 边缘 padding → 4 层 CNN → sigmoid → DM 概率图
4. 双边滤波去噪 + 概率加权融合
5. 输出：`_out.bmp`（去噪结果）、`_cnn.bmp`（DM 覆盖检测图）

### 输出目录结构

输入目录的目录结构会被保留：

```
SR_Data/video1/0001.bmp  →  predictions_fp32_gs8/video1/0001_out.bmp
SR_Data/video2/0001.bmp  →  predictions_fp32_gs8/video2/0001_out.bmp
```

每个输出目录包含：
- `xxx_out.bmp` — 双边滤波去噪结果
- `xxx_cnn.bmp` — DM 检测 overlay
- `xxx_in.bmp` — 原始输入（debug）
- `xxx_pred8x8.bmp` — 预测概率图（debug）

---

## QAT 量化

### 流程

1. 加载预训练 `mosquito_denoise_cnn_cost_down_grid_8.pth`
2. 使用 NNI `QAT_Quantizer` 插入 FakeQuantize 节点
3. QAT finetune（10 epoch，低学习率 3e-6）
4. 保存量化感知模型 `mosquito_denoise_cnn_qat_grid_8.pth`

### 配置

| 层 | 量化类型 | 位宽 |
|----|---------|------|
| conv1 | input + weight | 8+8 |
| conv2/3/4 | weight + output | 8+8 |
| relu1/2/3, sigmoid | output | 8 |

```bash
python train_qat_test.py
```

---

## 特征验证

对比新版本（int 整数化）和旧版本（float）的特征一致性。

```bash
python validate_features.py
```

输出 `feature_validation_report.md`。关键结论：

| 等级 | 数量 | 说明 |
|------|------|------|
| Safe | 8 | ringing 特征(6) + 行列差分(2) |
| Minor Loss | 3 | edge_strength, second_diff_max, ringing_mean_min_max |
| High Risk | 5 | mean_var, low_var_count, high_var_count, edge_orientation_conf, second_diff_min_max |

---

## GS8 vs GS16

| | GS=8 | GS=16 |
|--|------|-------|
| Block 大小 | 8x8 | 16x16 |
| 特征图分辨率 | W/8 x H/8 | W/16 x H/16 |
| low_var_count 除数 | 64 (8x8 像素数) | 256 (16x16 像素数) |
| 适用场景 | 精细检测 | 大块 DM / 性能优先 |

---

## 数据集格式

每个子目录包含：
- `grid_{gs}_dm_9x9.csv` — DM 样本特征（label=1）
- `grid_{gs}_not_dm_9x9.csv` — 非 DM 样本特征（label=0）

CSV 包含 16 个特征列，每 81 行构成一个 9x9 patch。

路径通过 `grid_8_dataset_paths.txt` / `grid_16_dataset_paths.txt` 管理。
