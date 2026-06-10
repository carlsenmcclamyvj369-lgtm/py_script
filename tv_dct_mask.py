import cv2
import numpy as np
from scipy.fftpack import dct


# ============================================================
# Basic utilities
# ============================================================

def to_float01(img):
    """Convert uint8 image to float32 [0, 1]."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def to_uint8(img):
    """Convert float image to uint8."""
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8)


def dct2(block):
    """2D orthonormal DCT."""
    return dct(dct(block.T, norm="ortho").T, norm="ortho")


def normalize_for_vis(x, percentile=99.0):
    """Normalize map for visualization."""
    x = x.astype(np.float32)
    denom = np.percentile(np.abs(x), percentile) + 1e-8
    y = x / denom
    return np.clip(y, 0.0, 1.0)


def signed_texture_vis(texture, gain=8.0):
    """
    Visualize signed texture layer.
    Zero texture -> 0.5 gray.
    """
    return np.clip(texture * gain + 0.5, 0.0, 1.0)


# ============================================================
# Guided filter / smoothing
# ============================================================

def box_filter(img, r):
    ksize = 2 * r + 1
    return cv2.blur(img, (ksize, ksize))


def guided_filter_gray(I, p, r=16, eps=1e-3):
    """
    Grayscale guided filter.

    I: guide image, HxW, float32 [0,1]
    p: input image, HxW, float32
    """
    I = I.astype(np.float32)
    p = p.astype(np.float32)

    mean_I = box_filter(I, r)
    mean_p = box_filter(p, r)

    corr_I = box_filter(I * I, r)
    corr_Ip = box_filter(I * p, r)

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box_filter(a, r)
    mean_b = box_filter(b, r)

    q = mean_a * I + mean_b
    return q.astype(np.float32)


def structure_texture_decomposition(
    Y,
    method="guided",
    guided_radius=24,
    guided_eps=2e-3,
    bilateral_d=11,
    bilateral_sigma_color=0.12,
    bilateral_sigma_space=15,
):
    """
    Approximate structure-texture decomposition.

    Original paper uses TV regularization:
        min (I_S - I)^2 + lambda * |grad I_S|

    Here we provide guided/bilateral approximation for easier implementation.
    """
    Y = Y.astype(np.float32)

    if method == "guided":
        S = guided_filter_gray(
            I=Y,
            p=Y,
            r=guided_radius,
            eps=guided_eps,
        )

    elif method == "bilateral":
        S = cv2.bilateralFilter(
            Y,
            d=bilateral_d,
            sigmaColor=bilateral_sigma_color,
            sigmaSpace=bilateral_sigma_space,
        )

    elif method == "gaussian":
        S = cv2.GaussianBlur(Y, (0, 0), 2.0)

    else:
        raise ValueError(f"Unknown decomposition method: {method}")

    T = Y - S
    return S.astype(np.float32), T.astype(np.float32)


# ============================================================
# Conservative DCT scene-detail mask
# ============================================================

def dct_detail_mask_conservative(
    texture,
    block_size=8,
    energy_threshold=3e-4,
    ratio_threshold=0.5,
    percentile_keep=60,
):
    """
    Conservative scene-detail mask using 8x8 DCT on texture layer.

    Paper's likelihood:
        t = sum B[u,v]^2 - B[0,0]^2 - B[0,1]^2 - B[1,0]^2

    This implementation is more conservative:
    1. No global texture normalization.
    2. Use absolute high-frequency energy threshold.
    3. Use high-frequency ratio threshold.
    4. Optional percentile sparsity control.

    Args:
        texture:
            HxW float32 texture layer.
        block_size:
            JPEG block size, normally 8.
        energy_threshold:
            Minimum high-frequency energy.
            Increase this if mask is too large.
        ratio_threshold:
            E_high / E_total threshold.
            Increase this if mask is too large.
        percentile_keep:
            Keep only blocks above this percentile among candidates.
            Example 60 keeps top 40% candidate blocks.
            Set None to disable.

    Returns:
        mask:
            HxW float32 block-wise 0/1 detail mask.
        score_map:
            HxW float32 block-wise score map.
        energy_map:
            HxW float32 high-frequency energy map.
        ratio_map:
            HxW float32 high-frequency ratio map.
    """
    H, W = texture.shape
    texture = texture.astype(np.float32)

    pad_h = (block_size - H % block_size) % block_size
    pad_w = (block_size - W % block_size) % block_size

    tex_pad = np.pad(texture, ((0, pad_h), (0, pad_w)), mode="reflect")

    Hp, Wp = tex_pad.shape
    nb_y = Hp // block_size
    nb_x = Wp // block_size

    block_score = np.zeros((nb_y, nb_x), dtype=np.float32)
    block_energy = np.zeros((nb_y, nb_x), dtype=np.float32)
    block_ratio = np.zeros((nb_y, nb_x), dtype=np.float32)

    eps = 1e-12

    for by in range(nb_y):
        for bx in range(nb_x):
            y = by * block_size
            x = bx * block_size

            block = tex_pad[y:y + block_size, x:x + block_size]
            B = dct2(block)

            # Total non-DC energy.
            E_total = np.sum(B ** 2) - B[0, 0] ** 2
            E_total = max(E_total, 0.0)

            # Exclude two lowest AC components.
            E_low = B[0, 1] ** 2 + B[1, 0] ** 2
            E_high = max(E_total - E_low, 0.0)

            ratio = E_high / (E_total + eps)

            # Conservative score.
            score = E_high * ratio

            block_energy[by, bx] = E_high
            block_ratio[by, bx] = ratio
            block_score[by, bx] = score

    block_mask = (block_energy > energy_threshold) & (block_ratio > ratio_threshold)

    if percentile_keep is not None:
        valid_scores = block_score[block_mask]
        if valid_scores.size > 0:
            th = np.percentile(valid_scores, percentile_keep)
            block_mask = block_mask & (block_score >= th)

    mask_pad = np.zeros((Hp, Wp), dtype=np.float32)
    score_pad = np.zeros((Hp, Wp), dtype=np.float32)
    energy_pad = np.zeros((Hp, Wp), dtype=np.float32)
    ratio_pad = np.zeros((Hp, Wp), dtype=np.float32)

    for by in range(nb_y):
        for bx in range(nb_x):
            y = by * block_size
            x = bx * block_size

            if block_mask[by, bx]:
                mask_pad[y:y + block_size, x:x + block_size] = 1.0

            score_pad[y:y + block_size, x:x + block_size] = block_score[by, bx]
            energy_pad[y:y + block_size, x:x + block_size] = block_energy[by, bx]
            ratio_pad[y:y + block_size, x:x + block_size] = block_ratio[by, bx]

    return (
        mask_pad[:H, :W],
        score_pad[:H, :W],
        energy_pad[:H, :W],
        ratio_pad[:H, :W],
    )


# ============================================================
# Conservative mask refinement
# ============================================================

def refine_mask_conservative(
    mask,
    structure_gray,
    open_kernel=3,
    blur_sigma=0.8,
    edge_suppress=0.6,
    hard_threshold=0.35,
):
    """
    Conservative refinement of block-wise detail mask.

    This approximates the paper's matting refinement, but intentionally avoids
    region expansion. The paper uses matting Laplacian generated from the
    structure layer. Here we use a simpler conservative approximation.

    Steps:
    1. Morphological open removes isolated small responses.
    2. Gaussian blur makes boundary soft.
    3. Strong structure edges suppress the mask to reduce ringing retention.
    4. Hard threshold avoids huge soft masks.
    """
    mask_u8 = (mask > 0.5).astype(np.uint8) * 255

    if open_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (open_kernel, open_kernel)
        )
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

    refined = mask_u8.astype(np.float32) / 255.0

    if blur_sigma > 0:
        refined = cv2.GaussianBlur(refined, (0, 0), blur_sigma)

    gx = cv2.Sobel(structure_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(structure_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)

    grad_norm = grad / (np.percentile(grad, 99) + 1e-6)
    grad_norm = np.clip(grad_norm, 0.0, 1.0)

    # Suppress detail mask near strong edges, where ringing/mosquito noise often lives.
    refined = refined * (1.0 - edge_suppress * grad_norm)

    if hard_threshold is not None:
        refined = np.where(refined > hard_threshold, refined, 0.0)

    return np.clip(refined, 0.0, 1.0).astype(np.float32)


# ============================================================
# Artifact-like mask
# ============================================================

def build_artifact_like_mask(
    structure,
    detail_mask,
    dct_score,
    edge_percentile=85,
    edge_dilate_size=7,
    blur_sigma=1.0,
):
    """
    Build artifact-like mask.

    Idea:
        artifact-like = near strong edge
                        * not real scene detail
                        * has DCT high-frequency response

    This is useful for ringing/mosquito-like artifact localization.
    """
    gx = cv2.Sobel(structure, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(structure, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)

    edge_th = np.percentile(grad, edge_percentile)
    edge = grad > edge_th

    edge_u8 = edge.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (edge_dilate_size, edge_dilate_size)
    )
    near_edge = cv2.dilate(edge_u8, kernel).astype(np.float32) / 255.0

    score_norm = dct_score / (np.percentile(dct_score, 99) + 1e-8)
    score_norm = np.clip(score_norm, 0.0, 1.0)

    artifact = near_edge * (1.0 - detail_mask) * score_norm

    if blur_sigma > 0:
        artifact = cv2.GaussianBlur(artifact, (0, 0), blur_sigma)

    return np.clip(artifact, 0.0, 1.0).astype(np.float32)


# ============================================================
# Texture deblocking at JPEG 8x8 borders
# ============================================================

def deblock_texture_at_8x8_borders(
    texture,
    beta=0.5,
    iterations=1,
    block_size=8,
):
    """
    Approximate texture-layer deblocking at JPEG 8x8 borders.

    Paper objective:
        min sum (I_T^d - I_T)^2 + beta * sum_{i in eta} |grad I_T^d|^2

    Here we approximate by blending pixels across block boundaries.
    """
    out = texture.copy().astype(np.float32)
    H, W = out.shape

    for _ in range(iterations):
        src = out.copy()

        # Vertical block borders.
        for x in range(block_size, W, block_size):
            left = src[:, x - 1]
            right = src[:, x]
            avg = 0.5 * (left + right)

            out[:, x - 1] = (1.0 - beta) * src[:, x - 1] + beta * avg
            out[:, x] = (1.0 - beta) * src[:, x] + beta * avg

        # Horizontal block borders.
        for y in range(block_size, H, block_size):
            top = src[y - 1, :]
            bottom = src[y, :]
            avg = 0.5 * (top + bottom)

            out[y - 1, :] = (1.0 - beta) * src[y - 1, :] + beta * avg
            out[y, :] = (1.0 - beta) * src[y, :] + beta * avg

    return out.astype(np.float32)


# ============================================================
# Tone curve and derivative
# ============================================================

def tone_curve_enhancement(x, gamma=0.75):
    """
    Simple tone curve:
        f(x) = x^gamma

    gamma < 1 brightens / enhances dark regions.
    """
    x = np.clip(x, 0.0, 1.0)
    return x ** gamma


def tone_curve_derivative(x, gamma=0.75):
    """
    Derivative of f(x)=x^gamma:
        f'(x)=gamma*x^(gamma-1)

    Used as K in paper's layer recomposition approximation.
    """
    eps = 1e-4
    x = np.maximum(x, eps)
    return gamma * (x ** (gamma - 1.0))


# ============================================================
# Main pipeline
# ============================================================

def jpeg_artifact_suppression_contrast_enhance_v2(
    img_bgr,
    gamma=0.75,

    # decomposition
    decomposition_method="guided",
    guided_radius=24,
    guided_eps=2e-3,

    # DCT detail mask
    energy_threshold=3e-4,
    ratio_threshold=0.65,
    percentile_keep=60,

    # mask refinement
    open_kernel=3,
    blur_sigma=0.8,
    edge_suppress=0.6,
    hard_threshold=0.35,

    # artifact mask
    use_artifact_mask=True,
    artifact_strength=0.8,
    edge_percentile=85,
    edge_dilate_size=7,

    # deblocking
    deblock_beta=0.5,
    deblock_iterations=1,

    # texture recomposition
    texture_floor=0.25,
    max_texture_gain=3.0,
):
    """
    JPEG artifact suppression integrated with contrast enhancement.

    Input:
        img_bgr: uint8 BGR image.

    Output:
        result_bgr: uint8 BGR image.
        debug: dict of intermediate maps.

    Important:
        detail_mask means "likely real scene texture", not JPEG noise.
        artifact_like_mask means "likely ringing/mosquito artifact".
    """
    img_bgr_f = to_float01(img_bgr)

    # Work on luminance channel.
    ycrcb = cv2.cvtColor(to_uint8(img_bgr_f), cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0
    Y = ycrcb[..., 0].astype(np.float32)

    # 1. Structure-texture decomposition.
    S, T = structure_texture_decomposition(
        Y,
        method=decomposition_method,
        guided_radius=guided_radius,
        guided_eps=guided_eps,
    )

    # 2. Conservative DCT scene-detail mask.
    mask0, dct_score, dct_energy, dct_ratio = dct_detail_mask_conservative(
        T,
        block_size=8,
        energy_threshold=energy_threshold,
        ratio_threshold=ratio_threshold,
        percentile_keep=percentile_keep,
    )

    # 3. Conservative refinement.
    detail_mask = refine_mask_conservative(
        mask0,
        S,
        open_kernel=open_kernel,
        blur_sigma=blur_sigma,
        edge_suppress=edge_suppress,
        hard_threshold=hard_threshold,
    )

    # 4. Texture deblocking.
    T_deblocked = deblock_texture_at_8x8_borders(
        T,
        beta=deblock_beta,
        iterations=deblock_iterations,
        block_size=8,
    )

    # 5. Build artifact-like mask.
    if use_artifact_mask:
        artifact_mask = build_artifact_like_mask(
            structure=S,
            detail_mask=detail_mask,
            dct_score=dct_score,
            edge_percentile=edge_percentile,
            edge_dilate_size=edge_dilate_size,
            blur_sigma=1.0,
        )
    else:
        artifact_mask = np.zeros_like(Y, dtype=np.float32)

    # 6. Enhance structure layer.
    S_enhanced = tone_curve_enhancement(S, gamma=gamma)

    # 7. Texture gain K.
    K = tone_curve_derivative(S, gamma=gamma)
    K = np.clip(K, 0.0, max_texture_gain)

    # 8. Texture recomposition.
    #
    # detail_mask controls how much texture is trusted.
    # texture_floor avoids completely killing mask-outside texture.
    #
    # artifact_mask suppresses likely ringing/mosquito artifacts.
    texture_keep = texture_floor + (1.0 - texture_floor) * detail_mask
    artifact_suppress = 1.0 - artifact_strength * artifact_mask
    artifact_suppress = np.clip(artifact_suppress, 0.0, 1.0)

    T_clean = T_deblocked * texture_keep * artifact_suppress
    T_enhanced = K * T_clean

    Y_out = S_enhanced + T_enhanced
    Y_out = np.clip(Y_out, 0.0, 1.0)

    # Put enhanced Y back.
    out_ycrcb = ycrcb.copy()
    out_ycrcb[..., 0] = Y_out

    result_bgr = cv2.cvtColor(to_uint8(out_ycrcb), cv2.COLOR_YCrCb2BGR)

    debug = {
        "Y": Y,
        "structure": S,
        "texture": T,
        "mask_initial": mask0,
        "detail_mask": detail_mask,
        "artifact_mask": artifact_mask,
        "dct_score": dct_score,
        "dct_energy": dct_energy,
        "dct_ratio": dct_ratio,
        "texture_deblocked": T_deblocked,
        "texture_keep": texture_keep,
        "artifact_suppress": artifact_suppress,
        "texture_clean": T_clean,
        "structure_enhanced": S_enhanced,
        "Y_out": Y_out,
    }

    return result_bgr, debug


# ============================================================
# Debug saving
# ============================================================

def save_debug_maps(debug, prefix="debug"):
    cv2.imwrite(f"{prefix}_Y.png", to_uint8(debug["Y"]))
    cv2.imwrite(f"{prefix}_structure.png", to_uint8(debug["structure"]))
    cv2.imwrite(f"{prefix}_structure_enhanced.png", to_uint8(debug["structure_enhanced"]))

    cv2.imwrite(
        f"{prefix}_texture.png",
        to_uint8(signed_texture_vis(debug["texture"], gain=8.0))
    )

    cv2.imwrite(f"{prefix}_mask_initial.png", to_uint8(debug["mask_initial"]))
    cv2.imwrite(f"{prefix}_detail_mask.png", to_uint8(debug["detail_mask"]))
    cv2.imwrite(f"{prefix}_artifact_mask.png", to_uint8(debug["artifact_mask"]))

    cv2.imwrite(
        f"{prefix}_dct_score.png",
        to_uint8(normalize_for_vis(debug["dct_score"], percentile=99.0))
    )
    cv2.imwrite(
        f"{prefix}_dct_energy.png",
        to_uint8(normalize_for_vis(debug["dct_energy"], percentile=99.0))
    )
    cv2.imwrite(f"{prefix}_dct_ratio.png", to_uint8(debug["dct_ratio"]))

    cv2.imwrite(
        f"{prefix}_texture_deblocked.png",
        to_uint8(signed_texture_vis(debug["texture_deblocked"], gain=8.0))
    )

    cv2.imwrite(f"{prefix}_texture_keep.png", to_uint8(debug["texture_keep"]))
    cv2.imwrite(f"{prefix}_artifact_suppress.png", to_uint8(debug["artifact_suppress"]))

    cv2.imwrite(
        f"{prefix}_texture_clean.png",
        to_uint8(signed_texture_vis(debug["texture_clean"], gain=8.0))
    )

    cv2.imwrite(f"{prefix}_Y_out.png", to_uint8(debug["Y_out"]))


# ============================================================
# CLI example
# ============================================================

if __name__ == "__main__":
    # input_path = "./test_data/hisense_mnr_mis_clarity#out1#mnr_input0002.bmp"
    input_path = "./test_data/05.02.25#out1#mnr_input0012.bmp"
    # input_path = "./test_data/001_OnlineNews#out1#mnr_input0007.bmp"
    output_path = "output_v2.png"

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(input_path)

    result, debug = jpeg_artifact_suppression_contrast_enhance_v2(
        img,
        gamma=0.75,

        decomposition_method="guided",
        guided_radius=24,
        guided_eps=2e-3,

        energy_threshold=3e-4,
        ratio_threshold=0.65,
        percentile_keep=60,

        open_kernel=3,
        blur_sigma=0.8,
        edge_suppress=0.6,
        hard_threshold=0.35,

        use_artifact_mask=True,
        artifact_strength=0.8,
        edge_percentile=85,
        edge_dilate_size=7,

        deblock_beta=0.5,
        deblock_iterations=1,

        texture_floor=0.25,
        max_texture_gain=3.0,
    )

    cv2.imwrite(output_path, result)
    save_debug_maps(debug, prefix="debug_v2")

    print(f"Saved result: {output_path}")
    print("Saved debug maps with prefix: debug_v2_*")