import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt


# ============================================================
# Basic utilities
# ============================================================

def safe_norm(x, lo, hi):
    if hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def rgb_to_gray(rgb):
    """
    RGB image to gray.
    Input: RGB uint8 or float image, shape HxWx3
    Output: gray float32, shape HxW
    """
    rgb = rgb.astype(np.float32)

    # BT.601 luma
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return gray.astype(np.float32)


def crop_to_multiple_of_8(img):
    """
    Crop image to multiples of 8.
    """
    h, w = img.shape[:2]
    h8 = h // 8 * 8
    w8 = w // 8 * 8
    return img[:h8, :w8]


# ============================================================
# 8x8 block feature extraction
# ============================================================

def block_3x3_var_map(block):
    block = block.astype(np.float32)
    out = np.zeros((6, 6), dtype=np.float32)

    for y in range(1, 7):
        for x in range(1, 7):
            patch = block[y - 1:y + 2, x - 1:x + 2]
            out[y - 1, x - 1] = np.var(patch)

    return out


def block_residual_map(block):
    block = block.astype(np.float32)
    out = np.zeros((6, 6), dtype=np.float32)

    for y in range(1, 7):
        for x in range(1, 7):
            patch = block[y - 1:y + 2, x - 1:x + 2]
            out[y - 1, x - 1] = block[y, x] - np.mean(patch)

    return out


def block_laplacian_map(block):
    block = block.astype(np.float32)
    out = np.zeros((6, 6), dtype=np.float32)

    for y in range(1, 7):
        for x in range(1, 7):
            c = block[y, x]
            up = block[y - 1, x]
            down = block[y + 1, x]
            left = block[y, x - 1]
            right = block[y, x + 1]
            out[y - 1, x - 1] = abs(4 * c - up - down - left - right)

    return out


def block_gradient_map(block):
    block = block.astype(np.float32)
    out = np.zeros((6, 6), dtype=np.float32)

    for y in range(1, 7):
        for x in range(1, 7):
            c = block[y, x]
            up = block[y - 1, x]
            down = block[y + 1, x]
            left = block[y, x - 1]
            right = block[y, x + 1]
            out[y - 1, x - 1] = (
                abs(c - up) + abs(c - down) + abs(c - left) + abs(c - right)
            )

    return out


def second_diff_energy_1d(v):
    v = np.asarray(v, dtype=np.float32)
    if len(v) < 3:
        return 0.0

    d2 = v[:-2] - 2 * v[1:-1] + v[2:]
    return float(np.mean(np.abs(d2)))


def row_col_oscillation_score(block):
    block = block.astype(np.float32)

    row_energy = np.mean([second_diff_energy_1d(block[y, :]) for y in range(8)])
    col_energy = np.mean([second_diff_energy_1d(block[:, x]) for x in range(8)])

    return {
        "row_second_diff": float(row_energy),
        "col_second_diff": float(col_energy),
        "max_second_diff": float(max(row_energy, col_energy)),
        "mean_second_diff": float(0.5 * (row_energy + col_energy)),
    }


def directional_edge_strength(block):
    block = block.astype(np.float32)

    dx = np.abs(np.diff(block, axis=1))
    dy = np.abs(np.diff(block, axis=0))

    h_strength = float(np.mean(dx))
    v_strength = float(np.mean(dy))
    max_strength = max(h_strength, v_strength)

    if max_strength <= 1e-6:
        orientation_conf = 0.0
    else:
        orientation_conf = abs(h_strength - v_strength) / max_strength

    return {
        "h_edge": h_strength,
        "v_edge": v_strength,
        "edge_strength": max_strength,
        "edge_orientation_conf": float(orientation_conf),
    }


def profile_ringing_score(block):
    """
    Detect high-low-high or low-high-low ringing profile.
    Scan all rows and columns.
    """
    block = block.astype(np.float32)

    def score_vec(v):
        v = np.asarray(v, dtype=np.float32)

        d = np.diff(v)
        d2 = np.diff(v, n=2)

        dyn = float(np.max(v) - np.min(v))
        d2_energy = float(np.mean(np.abs(d2))) if len(d2) > 0 else 0.0

        eps = 3.0
        signs = np.sign(d)
        signs[np.abs(d) < eps] = 0

        sign_changes = 0
        prev = 0
        for s in signs:
            if s == 0:
                continue
            if prev != 0 and s != prev:
                sign_changes += 1
            prev = s

        dyn_score = safe_norm(dyn, 20, 120)
        d2_score = safe_norm(d2_energy, 5, 60)
        sign_score = safe_norm(sign_changes, 1, 4)

        return 0.45 * dyn_score + 0.35 * d2_score + 0.20 * sign_score

    scores = []
    for y in range(8):
        scores.append(score_vec(block[y, :]))
    for x in range(8):
        scores.append(score_vec(block[:, x]))

    return float(np.max(scores)), float(np.mean(scores))


def texture_suppression_score(block, var_map):
    flat = var_map.flatten()

    median_var = float(np.median(flat))
    high_count = int(np.sum(flat > 500))
    very_high_count = int(np.sum(flat > 2000))

    edge_info = directional_edge_strength(block)
    orientation_conf = edge_info["edge_orientation_conf"]

    broad_activity = safe_norm(high_count, 8, 24)
    median_activity = safe_norm(median_var, 200, 1500)

    weak_orientation = 1.0 - orientation_conf

    texture_score = (
        0.45 * broad_activity +
        0.35 * median_activity +
        0.20 * weak_orientation
    )

    # If very strong oriented edge exists, reduce texture suppression.
    if very_high_count >= 6 and orientation_conf > 0.35:
        texture_score *= 0.65

    return float(np.clip(texture_score, 0.0, 1.0))


def clean_edge_suppression_score(block, var_map, residual_map):
    edge_info = directional_edge_strength(block)
    osc_info = row_col_oscillation_score(block)

    max_var = float(np.max(var_map))
    residual_mean = float(np.mean(np.abs(residual_map)))
    residual_max = float(np.max(np.abs(residual_map)))

    edge_score = safe_norm(edge_info["edge_strength"], 20, 100)
    var_score = safe_norm(max_var, 500, 6000)

    low_residual = 1.0 - safe_norm(residual_mean, 5, 25)
    low_osc = 1.0 - safe_norm(osc_info["max_second_diff"], 8, 60)

    orient = edge_info["edge_orientation_conf"]

    clean_score = (
        0.35 * edge_score +
        0.25 * var_score +
        0.25 * low_residual +
        0.15 * orient
    )

    if residual_max > 45:
        clean_score *= 0.65

    if osc_info["max_second_diff"] > 45:
        clean_score *= 0.60

    return float(np.clip(clean_score, 0.0, 1.0))


# ============================================================
# Stage 1: block initial mosquito score
# ============================================================

def mosquito_block_initial_score(block):
    """
    Internal function:
    Calculate initial mosquito score for one 8x8 block.
    Main function will NOT call this directly.
    """
    block = np.asarray(block, dtype=np.float32)

    var_map = block_3x3_var_map(block)
    residual_map = block_residual_map(block)
    lap_map = block_laplacian_map(block)
    grad_map = block_gradient_map(block)

    edge_info = directional_edge_strength(block)
    osc_info = row_col_oscillation_score(block)
    profile_max, profile_mean = profile_ringing_score(block)

    flat_var = var_map.flatten()

    mean_var = float(np.mean(flat_var))
    median_var = float(np.median(flat_var))
    max_var = float(np.max(flat_var))
    top5_var = float(np.mean(np.sort(flat_var)[-5:]))

    residual_mean = float(np.mean(np.abs(residual_map)))
    residual_max = float(np.max(np.abs(residual_map)))

    lap_mean = float(np.mean(lap_map))
    lap_max = float(np.max(lap_map))

    grad_mean = float(np.mean(grad_map))
    grad_max = float(np.max(grad_map))

    high_var_count = int(np.sum(flat_var > 500))
    very_high_var_count = int(np.sum(flat_var > 2000))

    activity_score = (
        0.30 * safe_norm(mean_var, 100, 2500) +
        0.25 * safe_norm(max_var, 500, 8000) +
        0.25 * safe_norm(top5_var, 500, 6000) +
        0.20 * safe_norm(high_var_count, 3, 18)
    )

    edge_score = (
        0.50 * safe_norm(edge_info["edge_strength"], 15, 100) +
        0.30 * safe_norm(grad_max, 40, 250) +
        0.20 * edge_info["edge_orientation_conf"]
    )

    residual_score = (
        0.35 * safe_norm(residual_mean, 5, 30) +
        0.25 * safe_norm(residual_max, 15, 80) +
        0.20 * safe_norm(lap_mean, 15, 100) +
        0.20 * safe_norm(lap_max, 40, 250)
    )

    ringing_score = (
        0.55 * profile_max +
        0.25 * profile_mean +
        0.20 * safe_norm(osc_info["max_second_diff"], 8, 80)
    )

    texture_score = texture_suppression_score(block, var_map)
    clean_edge_score = clean_edge_suppression_score(block, var_map, residual_map)

    raw_score = (
        0.28 * activity_score +
        0.22 * edge_score +
        0.25 * residual_score +
        0.25 * ringing_score
    )

    suppression = (
        0.45 * clean_edge_score * (1.0 - 0.5 * ringing_score) +
        0.35 * texture_score * (1.0 - 0.4 * edge_score)
    )

    score = raw_score * (1.0 - 0.55 * suppression)

    # Flat block: leave context possibility but avoid high single-block score.
    if mean_var < 30 and residual_mean < 5:
        score = min(score, 0.25)

    score = float(np.clip(score, 0.0, 1.0))

    features = {
        "score_initial": score,

        "mean_var": mean_var,
        "median_var": median_var,
        "max_var": max_var,
        "top5_var": top5_var,
        "high_var_count": high_var_count,
        "very_high_var_count": very_high_var_count,

        "residual_mean": residual_mean,
        "residual_max": residual_max,
        "lap_mean": lap_mean,
        "lap_max": lap_max,
        "grad_mean": grad_mean,
        "grad_max": grad_max,

        "edge_strength": edge_info["edge_strength"],
        "edge_orientation_conf": edge_info["edge_orientation_conf"],

        "row_second_diff": osc_info["row_second_diff"],
        "col_second_diff": osc_info["col_second_diff"],
        "max_second_diff": osc_info["max_second_diff"],

        "profile_ringing_max": profile_max,
        "profile_ringing_mean": profile_mean,

        "activity_score": activity_score,
        "edge_score": edge_score,
        "residual_score": residual_score,
        "ringing_score": ringing_score,
        "texture_suppression": texture_score,
        "clean_edge_suppression": clean_edge_score,
    }

    return score, features


# ============================================================
# Image/block processing
# ============================================================

def image_to_blocks(gray):
    gray = gray.astype(np.float32)
    h, w = gray.shape

    assert h % 8 == 0 and w % 8 == 0

    bh = h // 8
    bw = w // 8

    blocks = gray.reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3)
    return blocks


def blocks_to_score_map(blocks):
    bh, bw = blocks.shape[:2]

    score_map = np.zeros((bh, bw), dtype=np.float32)
    feature_map = [[None for _ in range(bw)] for _ in range(bh)]

    for y in range(bh):
        for x in range(bw):
            score, feat = mosquito_block_initial_score(blocks[y, x])
            score_map[y, x] = score
            feature_map[y][x] = feat

    return score_map, feature_map


def get_window(arr, cy, cx, radius):
    h, w = arr.shape

    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius + 1)

    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius + 1)

    return arr[y0:y1, x0:x1]


def refine_score_with_context(score_map, feature_map, radius=1):
    """
    radius = 1: 3x3 blocks
    radius = 2: 5x5 blocks
    """
    h, w = score_map.shape
    final = np.zeros_like(score_map, dtype=np.float32)

    edge_strength = np.zeros_like(score_map, dtype=np.float32)
    orient_conf = np.zeros_like(score_map, dtype=np.float32)
    ring_score = np.zeros_like(score_map, dtype=np.float32)
    texture_sup = np.zeros_like(score_map, dtype=np.float32)
    clean_sup = np.zeros_like(score_map, dtype=np.float32)

    for y in range(h):
        for x in range(w):
            f = feature_map[y][x]
            edge_strength[y, x] = f["edge_strength"]
            orient_conf[y, x] = f["edge_orientation_conf"]
            ring_score[y, x] = f["ringing_score"]
            texture_sup[y, x] = f["texture_suppression"]
            clean_sup[y, x] = f["clean_edge_suppression"]

    for y in range(h):
        for x in range(w):
            s0 = float(score_map[y, x])

            win_score = get_window(score_map, y, x, radius)
            win_edge = get_window(edge_strength, y, x, radius)
            win_orient = get_window(orient_conf, y, x, radius)
            win_ring = get_window(ring_score, y, x, radius)
            win_texture = get_window(texture_sup, y, x, radius)
            win_clean = get_window(clean_sup, y, x, radius)

            neighbor_mean = float(np.mean(win_score))
            neighbor_max = float(np.max(win_score))
            neighbor_pos_ratio = float(np.mean(win_score > 0.5))

            edge_context = safe_norm(float(np.mean(win_edge)), 15, 90)
            ring_context = float(np.mean(win_ring))
            orient_context = float(np.mean(win_orient))

            texture_context = float(np.mean(win_texture))
            clean_context = float(np.mean(win_clean))

            strong_neighbor = safe_norm(neighbor_max, 0.45, 0.9)

            continuity = (
                0.45 * neighbor_mean +
                0.35 * neighbor_pos_ratio +
                0.20 * strong_neighbor
            )

            context_boost = (
                0.35 * continuity +
                0.25 * edge_context +
                0.25 * ring_context +
                0.15 * orient_context
            )

            context_suppress = (
                0.45 * clean_context +
                0.35 * texture_context
            )

            s = 0.60 * s0 + 0.40 * context_boost

            # Context-dependent flat mosquito:
            # current block looks flat, but neighbors strongly indicate mosquito.
            f = feature_map[y][x]
            flat_current = f["mean_var"] < 40 and f["residual_mean"] < 6

            if flat_current and neighbor_mean > 0.45 and edge_context > 0.4:
                s = max(s, 0.45 + 0.35 * neighbor_mean)

            s = s * (1.0 - 0.35 * context_suppress)

            # Preserve strong local mosquito evidence.
            if s0 > 0.75 and f["ringing_score"] > 0.45:
                s = max(s, s0 * 0.9)

            final[y, x] = np.clip(s, 0.0, 1.0)

    return final


def mosquito_detect_rgb_image(rgb, use_5x5=True):
    """
    Final pipeline:
    input RGB image
    output initial block score, final block score, gray image
    """
    rgb = crop_to_multiple_of_8(rgb)
    gray = rgb_to_gray(rgb)

    blocks = image_to_blocks(gray)

    initial_score_map, feature_map = blocks_to_score_map(blocks)

    score_3x3 = refine_score_with_context(
        initial_score_map,
        feature_map,
        radius=1
    )

    if use_5x5:
        final_score_map = refine_score_with_context(
            score_3x3,
            feature_map,
            radius=2
        )
    else:
        final_score_map = score_3x3

    return initial_score_map, final_score_map, gray, rgb


# ============================================================
# Visualization
# ============================================================

def block_score_to_image(score_map, target_h, target_w):
    """
    Convert block-level score map to pixel-level heatmap.
    """
    score_img = np.repeat(np.repeat(score_map, 8, axis=0), 8, axis=1)
    score_img = score_img[:target_h, :target_w]
    return score_img.astype(np.float32)


def overlay_mosquito_level(rgb, score_img, alpha=0.45):
    """
    Overlay mosquito score heatmap on RGB image.

    score_img: HxW, value in [0,1]
    """
    rgb = rgb.astype(np.uint8)

    heat = np.uint8(np.clip(score_img * 255.0, 0, 255))
    heat_color_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat_color_rgb = cv2.cvtColor(heat_color_bgr, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(rgb, 1.0 - alpha, heat_color_rgb, alpha, 0)
    return overlay


def visualize_and_save(rgb, gray, initial_map, final_map, output_path=None):
    h, w = gray.shape

    initial_img = block_score_to_image(initial_map, h, w)
    final_img = block_score_to_image(final_map, h, w)

    overlay = overlay_mosquito_level(rgb, final_img, alpha=0.45)

    plt.figure(figsize=(16, 6))

    plt.subplot(1, 3, 1)
    plt.title("Input RGB")
    plt.imshow(rgb)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Final Mosquito Level")
    plt.imshow(final_img, cmap="jet", vmin=0, vmax=1)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Mosquito Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=150)
        print(f"Saved visualization to: {output_path}")

    plt.show()

    return overlay, final_img


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--input",
    #     type=str,
    #     required=True,
    #     help="Input RGB image path"
    # )
    parser.add_argument(
        "--output",
        type=str,
        default="mosquito_overlay.png",
        help="Output visualization path"
    )
    parser.add_argument(
        "--no_5x5",
        action="store_true",
        help="Disable 5x5 context refinement, only use 3x3"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Heatmap overlay alpha"
    )
    args = parser.parse_args()

    # Read RGB image
    # bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    bgr = cv2.imread(r"\\10.18.11.192\tv_alg_video\58_AI-NR\300.Dataset\05.02.25#out1#mnr_input0012.bmp", cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {args.input}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Main only calls the final RGB-image pipeline.
    initial_map, final_map, gray, rgb_crop = mosquito_detect_rgb_image(
        rgb,
        use_5x5=not args.no_5x5
    )

    h, w = gray.shape
    final_img = block_score_to_image(final_map, h, w)
    overlay = overlay_mosquito_level(rgb_crop, final_img, alpha=args.alpha)

    # Save overlay image
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, overlay_bgr)
    print(f"Saved overlay image to: {args.output}")

    # Show visualization
    visualize_and_save(
        rgb_crop,
        gray,
        initial_map,
        final_map,
        output_path=None
    )


if __name__ == "__main__":
    main()