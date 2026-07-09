"""
对推理结果做硬负例挖掘：把模型预测为 DM 的 block 提取出来存为图片，
方便人工检查哪些是误检（false positive），这些可以加入 not_dm 训练集。
"""
import sys, os, cv2, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import predict_cnn, feature_compute_reference as fcr

TEST_DIR = r"C:\code\py\denoise\scripts\test_data"
OUT_DIR = os.path.join(os.path.dirname(__file__), "hard_negatives")
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = predict_cnn.MosquitoDenoiseCNN().to(device)
model.load_state_dict(torch.load(predict_cnn.MODEL_PATH, map_location=device), strict=False)
model.eval()

bmps = sorted([f for f in os.listdir(TEST_DIR) if f.endswith('.bmp')])
for b in bmps:
    bmp_path = os.path.join(TEST_DIR, b)
    bgr = cv2.imread(bmp_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    y_full = fcr.compute_y_from_rgb(rgb)
    H, W = y_full.shape
    gh, gw = H // 8, W // 8

    grid = predict_cnn.compute_grid_features(y_full)
    f_idx = [predict_cnn.FEATURE_IDX[f] for f in predict_cnn.features_list]
    div_arr = np.array([predict_cnn.NORM_DIV[f] for f in predict_cnn.features_list], dtype=np.float32)

    from numpy.lib.stride_tricks import sliding_window_view
    grid_pad = np.pad(grid, ((4, 4), (4, 4), (0, 0)), mode='edge')
    grid_norm = np.clip(grid_pad / div_arr, 0, 1)
    windows = sliding_window_view(grid_norm, (9, 9), axis=(0, 1))
    windows = windows.transpose(0, 1, 3, 4, 2)
    X = np.ascontiguousarray(windows).reshape(gh * gw, 9, 9, 16).transpose(0, 3, 1, 2)

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        probs = model(X_t).cpu().numpy().flatten()
    pred_map = probs.reshape(gh, gw)

    dm_mask = pred_map > 0.5
    dm_indices = np.where(dm_mask)
    print(f"{b}: {len(dm_indices[0])} DM blocks")

    # 保存每个 DM block 的截图
    for k in range(min(len(dm_indices[0]), 50)):
        bi, bj = dm_indices[0][k], dm_indices[1][k]
        y1, y2 = bi * 8, min(bi * 8 + 8, H)
        x1, x2 = bj * 8, min(bj * 8 + 8, W)
        block = bgr[y1:y2, x1:x2]
        score = pred_map[bi, bj]
        stem = os.path.splitext(b)[0]
        cv2.imwrite(os.path.join(OUT_DIR, f"{stem}_b{bi}_{bj}_p{score:.3f}.bmp"), block)
