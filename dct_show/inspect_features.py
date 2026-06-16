import cv2
import numpy as np
from scipy.ndimage import map_coordinates

# =========================================================
# 直接从 gibs_hardware.py 复用 (保留全部计算逻辑)
# =========================================================
def robust_mad_threshold(values, scale=1.0):
    values = np.asarray(values, dtype=np.float32)
    med = np.median(values)
    mad = np.median(np.abs(values - med)) + 1e-6
    return float(scale * 1.4826 * mad)


def compute_all_features(image_path, patch_size=3, stride=1, profile_radius=3, scale=1.0):
    """完整计算并返回所有中间特征，供交互查看。"""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None: raise ValueError("Cannot read image")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape

    edges = cv2.Canny(gray.astype(np.uint8), 90, 120)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    blur = cv2.GaussianBlur(gray, ksize=(0, 0), sigmaX=1.0)
    residual = gray - blur
    abs_residual = np.abs(residual)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge_band = cv2.dilate(edges, kernel, iterations=1) > 0
    edge_pixels = edges > 0
    edge_neighbor_f = (edge_band & (~edge_pixels)).astype(np.float32)
    non_edge_f = (~edge_band).astype(np.float32)

    # ---- loc_ratio ----
    ctx_k = patch_size + 2 * profile_radius
    sum_neighbor = cv2.boxFilter(abs_residual * edge_neighbor_f, -1, (ctx_k, ctx_k), normalize=False)
    count_neighbor = cv2.boxFilter(edge_neighbor_f, -1, (ctx_k, ctx_k), normalize=False)
    sum_non_edge = cv2.boxFilter(abs_residual * non_edge_f, -1, (ctx_k, ctx_k), normalize=False)
    count_non_edge = cv2.boxFilter(non_edge_f, -1, (ctx_k, ctx_k), normalize=False)
    mean_neighbor = sum_neighbor / np.maximum(count_neighbor, 1.0)
    mean_non_edge = sum_non_edge / np.maximum(count_non_edge, 1e-6)
    loc_ratio_full = mean_neighbor / mean_non_edge

    # ---- edge profile extraction ----
    ey, ex = np.where(edges > 0)
    abs_gx, abs_gy = np.abs(gx[ey, ex]), np.abs(gy[ey, ex])
    valid_mask = (abs_gx + abs_gy) > 1e-6
    ey, ex = ey[valid_mask], ex[valid_mask]
    abs_gx, abs_gy = abs_gx[valid_mask], abs_gy[valid_mask]

    N = len(ey)
    directions = np.zeros(N, dtype=np.int8)
    gx_val, gy_val = gx[ey, ex], gy[ey, ex]
    is_horiz = abs_gx > 2 * abs_gy
    is_vert  = abs_gy > 2 * abs_gx
    is_diag1 = (~is_horiz) & (~is_vert) & (np.sign(gx_val) == np.sign(gy_val))
    is_diag2 = (~is_horiz) & (~is_vert) & (np.sign(gx_val) != np.sign(gy_val))
    directions[is_horiz] = 0; directions[is_vert] = 1
    directions[is_diag1] = 2; directions[is_diag2] = 3
    steps = [(0, 1), (1, 0), (1, 1), (-1, 1)]
    t = np.arange(-profile_radius, profile_radius + 1)
    L = len(t)
    offsets = np.array([[dy * t, dx * t] for dy, dx in steps])
    selected_offsets = offsets[directions]
    samp_y = ey[:, None] + selected_offsets[:, 0, :]
    samp_x = ex[:, None] + selected_offsets[:, 1, :]
    np.clip(samp_y, 0, H - 1, out=samp_y)
    np.clip(samp_x, 0, W - 1, out=samp_x)

    # ---- Feature A: crossings ----
    profiles = residual[samp_y, samp_x]
    abs_profiles = np.abs(profiles)
    prof_thresh = scale * np.mean(abs_profiles, axis=1, keepdims=True)
    signs = np.zeros_like(profiles, dtype=np.int8)
    signs[profiles > prof_thresh] = 1
    signs[profiles < -prof_thresh] = -1
    mask_signs = (signs != 0)
    idx = np.where(mask_signs, np.arange(L), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    row_indices = np.arange(N)[:, None]
    filled_signs = signs[row_indices, idx]
    valid_crossings = (filled_signs[:, :-1] != filled_signs[:, 1:]) & (filled_signs[:, :-1] != 0) & (filled_signs[:, 1:] != 0)
    crossings = np.sum(valid_crossings, axis=1)

    # ---- Feature B: decay ----
    d = np.abs(t)
    near_mask, mid_mask, far_mask = (d == 1), ((d >= 2) & (d <= 3)), (d >= 4)
    near_e = np.mean(abs_profiles[:, near_mask], axis=1) if np.any(near_mask) else np.zeros(N)
    mid_e = np.mean(abs_profiles[:, mid_mask], axis=1) if np.any(mid_mask) else np.zeros(N)
    far_e = np.mean(abs_profiles[:, far_mask], axis=1) if np.any(far_mask) else np.zeros(N)
    mono_score = 0.0 + 0.5 * (near_e > mid_e) + 0.5 * (mid_e > far_e)
    weighted_dist = np.sum(abs_profiles * d, axis=1) / (np.sum(abs_profiles, axis=1) + 1e-6)
    compactness = 1.0 / (1.0 + weighted_dist)
    ratio_score = np.tanh((near_e / (far_e + 1e-6)) / 3.0)
    decay_scores = 0.4 * mono_score + 0.3 * compactness + 0.3 * ratio_score

    # ---- Feature C: gibbs ----
    profiles_gray = gray[samp_y, samp_x]
    prof_2d = profiles_gray.reshape(N, 1, L)
    smooth_2d = cv2.GaussianBlur(prof_2d, ksize=(0, 0), sigmaX=1.0)
    prof_residual = profiles_gray - smooth_2d.reshape(N, L)
    abs_prof_res = np.abs(prof_residual)
    residual_energy = np.mean(abs_prof_res, axis=1)

    grad_mag_edge = np.sqrt(gx[ey, ex]**2 + gy[ey, ex]**2)
    osc_thresh = scale * np.mean(abs_prof_res, axis=1, keepdims=True)
    osc_signs = np.zeros_like(prof_residual, dtype=np.int8)
    osc_signs[prof_residual > osc_thresh] = 1
    osc_signs[prof_residual < -osc_thresh] = -1
    mask_osc = osc_signs != 0
    idx_osc = np.where(mask_osc, np.arange(L), 0)
    np.maximum.accumulate(idx_osc, axis=1, out=idx_osc)
    filled_osc = osc_signs[row_indices, idx_osc]
    valid_adj = mask_osc[:, 1:]
    alt = (filled_osc[:, :-1] != filled_osc[:, 1:]) & valid_adj
    total_pairs = np.sum(valid_adj, axis=1).astype(np.float32)
    oscillation_score = np.divide(np.sum(alt, axis=1), total_pairs, out=np.zeros(N), where=total_pairs > 0)

    near_g = np.mean(abs_prof_res[:, near_mask], axis=1) if np.any(near_mask) else np.zeros(N)
    far_g = np.mean(abs_prof_res[:, far_mask], axis=1) if np.any(far_mask) else np.zeros(N)
    decay_from_prof = np.tanh((near_g / (far_g + 1e-6)) / 3.0)
    gibbs_scores = residual_energy * oscillation_score * decay_from_prof

    # ---- aggregation ----
    num_rows = (H - patch_size) // stride + 1
    num_cols = (W - patch_size) // stride + 1
    grid_y_centers = np.arange(num_rows) * stride + patch_size // 2
    grid_x_centers = np.arange(num_cols) * stride + patch_size // 2
    grid_X, grid_Y = np.meshgrid(grid_x_centers, grid_y_centers)
    grid_loc_ratio = loc_ratio_full[grid_Y, grid_X]

    patch_idx_y = np.clip(ey // stride, 0, num_rows - 1)
    patch_idx_x = np.clip(ex // stride, 0, num_cols - 1)

    grid_crossings_sum = np.zeros((num_rows, num_cols), dtype=np.float32)
    grid_decay_sum = np.zeros((num_rows, num_cols), dtype=np.float32)
    grid_valid_count = np.zeros((num_rows, num_cols), dtype=np.float32)
    np.add.at(grid_crossings_sum, (patch_idx_y, patch_idx_x), crossings)
    np.add.at(grid_decay_sum, (patch_idx_y, patch_idx_x), decay_scores)
    np.add.at(grid_valid_count, (patch_idx_y, patch_idx_x), 1)
    grid_mean_crossings = grid_crossings_sum / np.maximum(grid_valid_count, 1.0)
    grid_mean_decay = grid_decay_sum / np.maximum(grid_valid_count, 1.0)

    order = np.lexsort((patch_idx_x, patch_idx_y))
    sorted_gibbs = gibbs_scores[order]
    sorted_py = patch_idx_y[order]
    sorted_px = patch_idx_x[order]
    change = (np.diff(sorted_py) != 0) | (np.diff(sorted_px) != 0)
    bounds = np.where(change)[0] + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(sorted_gibbs)]])
    grid_gibbs_p90 = np.zeros((num_rows, num_cols), dtype=np.float32)
    for s, e in zip(starts, ends):
        py, px = sorted_py[s], sorted_px[s]
        grid_gibbs_p90[py, px] = np.percentile(sorted_gibbs[s:e], 90)

    score_grid = np.zeros((num_rows, num_cols), dtype=np.float32)
    mask = (grid_loc_ratio > 1.5) & (grid_mean_crossings > 1.2) & (grid_mean_decay > 0.4) & (grid_gibbs_p90 > 3)
    score_grid[mask] = grid_gibbs_p90[mask]

    return {
        "img": img, "gray": gray, "H": H, "W": W,
        "edges": edges, "gx": gx, "gy": gy,
        "residual": residual, "abs_residual": abs_residual,
        "loc_ratio_full": loc_ratio_full,
        "ey": ey, "ex": ex, "N": N,
        "directions": directions,
        "samp_y": samp_y, "samp_x": samp_x,
        "t": t, "L": L, "d": d,
        "near_mask": near_mask, "mid_mask": mid_mask, "far_mask": far_mask,
        "profiles": profiles, "abs_profiles": abs_profiles,
        "prof_thresh": prof_thresh,
        "crossings": crossings, "decay_scores": decay_scores,
        "profiles_gray": profiles_gray,
        "prof_residual": prof_residual, "abs_prof_res": abs_prof_res,
        "residual_energy": residual_energy,
        "grad_mag_edge": grad_mag_edge,
        "oscillation_score": oscillation_score,
        "near_g": near_g, "far_g": far_g,
        "decay_from_prof": decay_from_prof,
        "gibbs_scores": gibbs_scores,
        "patch_size": patch_size, "stride": stride,
        "num_rows": num_rows, "num_cols": num_cols,
        "grid_loc_ratio": grid_loc_ratio,
        "grid_mean_crossings": grid_mean_crossings,
        "grid_mean_decay": grid_mean_decay,
        "grid_gibbs_p90": grid_gibbs_p90,
        "score_grid": score_grid,
        "patch_idx_y": patch_idx_y, "patch_idx_x": patch_idx_x,
    }


def patch_label(p, row, col, num_cols):
    """单一 patch 的详细特征报告。"""
    p90 = p["grid_gibbs_p90"][row, col]
    lr = p["grid_loc_ratio"][row, col]
    cr = p["grid_mean_crossings"][row, col]
    dc = p["grid_mean_decay"][row, col]
    sc = p["score_grid"][row, col]
    lines = [
        f"===== Patch [({row},{col})  x={col*p['stride']}  y={row*p['stride']}] =====",
        f"  loc_ratio       = {lr:.3f}",
        f"  mean_crossings  = {cr:.3f}",
        f"  mean_decay      = {dc:.3f}",
        f"  gibbs_p90       = {p90:.3f}",
        f"  score           = {sc:.3f}",
    ]
    return "\n".join(lines)


def edge_profile_info(p, edge_idx):
    """单条边缘 profile 的详细特征。"""
    if edge_idx < 0 or edge_idx >= p["N"]:
        return "no edge"
    lines = [f"--- Edge ({p['ey'][edge_idx]},{p['ex'][edge_idx]}) ---"]
    lines.append(f"  grad={p['grad_mag_edge'][edge_idx]:.0f}  dir={p['directions'][edge_idx]}")
    lines.append(f"  energy={p['residual_energy'][edge_idx]:.3f}  osc={p['oscillation_score'][edge_idx]:.3f}")
    lines.append(f"  decay={p['decay_from_prof'][edge_idx]:.3f}  gibbs={p['gibbs_scores'][edge_idx]:.3f}")
    lines.append(f"  crossings={p['crossings'][edge_idx]}  decay_img={p['decay_scores'][edge_idx]:.3f}")
    return "\n".join(lines)


def draw_patch_overlay(disp, row, col, stride, patch_size):
    """在显示图上高亮当前 patch。"""
    y0 = row * stride
    x0 = col * stride
    cv2.rectangle(disp, (x0, y0), (x0 + patch_size, y0 + patch_size), (0, 255, 0), 2)
    return disp


def draw_edge_point(disp, ey, ex, color=(0, 0, 255)):
    cv2.circle(disp, (ex, ey), 2, color, -1)
    return disp


def main():
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "../test_data/05.02.25#out1#mnr_input0012.bmp"
        path = "../test_data/001_OnlineNews#out1#mnr_input0007.bmp"
        path = "../test_data/hisense_mnr_mis_clarity#out1#mnr_input0002.bmp"

    print(f"Loading {path} ...")
    p = compute_all_features(path)
    img = p["img"].copy()
    H, W = p["H"], p["W"]
    stride, ps = p["stride"], p["patch_size"]

    # 生成 score heatmap 用于叠加显示
    score = p["score_grid"].astype(np.float32)
    score_u8 = np.clip(score * 25, 0, 255).astype(np.uint8)
    heatmap_small = cv2.applyColorMap(score_u8, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap_small, (W, H), interpolation=cv2.INTER_NEAREST)

    # ===== 缩放显示 =====
    # 缩放图片窗口 (占屏幕上方区域)
    max_win_w = 1600
    scale = min(1.0, max_win_w / (W * 2))
    disp_w = int(W * 2 * scale)
    disp_h = int(H * scale)

    img_small = cv2.resize(img, None, fx=scale, fy=scale)
    heatmap_small = cv2.resize(heatmap, None, fx=scale, fy=scale)
    overlay_small = cv2.addWeighted(img_small, 0.6, heatmap_small, 0.4, 0)

    def draw_rect(d, row, col, sc):
        y0 = int(row * stride * sc)
        x0 = int(col * stride * sc)
        sz = int(ps * sc)
        cv2.rectangle(d, (x0, y0), (x0 + sz, y0 + sz), (0, 255, 0), max(1, int(2 * sc)))

    def draw_circle(d, ey, ex, sc):
        cv2.circle(d, (int(ex * sc), int(ey * sc)), max(1, int(2 * sc)), (0, 0, 255), -1)

    img_win = "Image — click to inspect  |  ESC exit"
    cv2.namedWindow(img_win)

    # 独立信息窗口 (黑底白字, 大字体)
    info_win = "Patch Info"
    cv2.namedWindow(info_win)
    info_w, info_h = 780, 380

    def make_info_display(text_lines):
        """生成信息窗口画面 (黑底白字大字体)。"""
        board = np.zeros((info_h, info_w, 3), dtype=np.uint8)
        y = 30
        for line in text_lines:
            cv2.putText(board, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (220, 220, 220), 1)
            y += 26
        return board

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ox = int(x / scale)
        oy = int(y / scale)
        col = ox // stride
        row = oy // stride
        if row >= p["num_rows"] or col >= p["num_cols"]:
            return

        patch_mask = (p["patch_idx_y"] == row) & (p["patch_idx_x"] == col)
        edge_indices = np.where(patch_mask)[0]

        # 更新图片窗口：高亮 patch + 边缘点
        left = img_small.copy()
        draw_rect(left, row, col, scale)
        for ei in edge_indices:
            draw_circle(left, p["ey"][ei], p["ex"][ei], scale)
        right = overlay_small.copy()
        draw_rect(right, row, col, scale)
        hw = disp_w // 2
        merged = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
        merged[:disp_h, :hw] = left
        merged[:disp_h, hw:] = right
        cv2.imshow(img_win, merged)

        # 更新信息窗口
        info_lines = [
            f">>> Click @ ({ox}, {oy})  →  Patch grid (row={row}, col={col})",
            f"    loc_ratio={p['grid_loc_ratio'][row,col]:.3f}   crossings={p['grid_mean_crossings'][row,col]:.3f}   decay={p['grid_mean_decay'][row,col]:.3f}",
            f"    gibbs_p90={p['grid_gibbs_p90'][row,col]:.3f}   score={p['score_grid'][row,col]:.3f}",
        ]
        if len(edge_indices) > 0:
            cy, cx = oy, ox
            dists = (p["ey"][edge_indices] - cy) ** 2 + (p["ex"][edge_indices] - cx) ** 2
            nei = edge_indices[np.argmin(dists)]
            info_lines.append("")
            info_lines.append(f">>> Edge ({p['ey'][nei]}, {p['ex'][nei]})   dir={p['directions'][nei]}   grad={p['grad_mag_edge'][nei]:.0f}")
            info_lines.append(f"    energy={p['residual_energy'][nei]:.3f}   oscillation={p['oscillation_score'][nei]:.3f}")
            info_lines.append(f"    decay={p['decay_from_prof'][nei]:.3f}   gibbs={p['gibbs_scores'][nei]:.3f}")
            info_lines.append(f"    crossings_feat={p['crossings'][nei]}   decay_feat={p['decay_scores'][nei]:.3f}")
            info_lines.append("")
            # Profile 用紧凑格式
            raw = p["profiles_gray"][nei]
            res = p["prof_residual"][nei]
            info_lines.append(f"    profile:  " + " ".join(f"{v:7.1f}" for v in raw))
            info_lines.append(f"    residual: " + " ".join(f"{v:7.2f}" for v in res))

        cv2.imshow(info_win, make_info_display(info_lines))

    cv2.setMouseCallback(img_win, on_click)

    # 初始显示
    merged = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
    hw = disp_w // 2
    merged[:disp_h, :hw] = img_small
    merged[:disp_h, hw:] = overlay_small
    cv2.imshow(img_win, merged)
    cv2.imshow(info_win, make_info_display(["Click on the image to inspect..."]))

    while True:
        key = cv2.waitKey(20)
        if key == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
