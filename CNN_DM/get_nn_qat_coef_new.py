# get_nn_qat_coef_new.py

import torch
import numpy as np


# =====================================================
# Quantization
# =====================================================

def quantize_weight(w):

    w_min = torch.min(w).item()
    w_max = torch.max(w).item()

    scale = (w_max - w_min) / 255.0

    if scale < 1e-9:
        scale = 1e-9

    zp = round(-w_min / scale)

    wq = torch.round(w / scale) + zp

    wq = torch.clamp(wq, 0, 255)

    return wq, scale, zp


# =====================================================
# Export
# =====================================================

def save_coef_to_txt(state, save_txtPath):

    # -----------------------------------------
    # fixed activation quant
    # -----------------------------------------

    sin = 1.0 / 255.0

    so1 = 1.0 / 255.0
    so2 = 1.0 / 255.0
    so3 = 1.0 / 255.0
    so4 = 1.0 / 255.0

    # -----------------------------------------
    # get tensors
    # -----------------------------------------

    conv1_w = state["conv1.weight"]
    conv1_b = state["conv1.bias"]

    conv2_w = state["conv2.weight"]
    conv2_b = state["conv2.bias"]

    conv3_w = state["conv3.weight"]
    conv3_b = state["conv3.bias"]

    conv4_w = state["conv4.weight"]
    conv4_b = state["conv4.bias"]

    # -----------------------------------------
    # quantize weight
    # -----------------------------------------

    qw1, sw1, zw1 = quantize_weight(conv1_w)
    qw2, sw2, zw2 = quantize_weight(conv2_w)
    qw3, sw3, zw3 = quantize_weight(conv3_w)
    qw4, sw4, zw4 = quantize_weight(conv4_w)

    # -----------------------------------------
    # quantize bias
    # -----------------------------------------

    bias_bit = 2 ** 12

    q_bias1 = torch.round(
        bias_bit * conv1_b / (sw1 * sin)
    )

    q_bias2 = torch.round(
        bias_bit * conv2_b / (sw2 * so1)
    )

    q_bias3 = torch.round(
        bias_bit * conv3_b / (sw3 * so2)
    )

    q_bias4 = torch.round(
        bias_bit * conv4_b / (sw4 * so3)
    )

    # -----------------------------------------
    # multiplier
    # -----------------------------------------

    M1 = sw1 * sin / so1
    M2 = sw2 * so1 / so2
    M3 = sw3 * so2 / so3
    M4 = sw4 * so3 / so4

    M_BIT = 2 ** 20

    q_m1 = int(M1 * M_BIT + 0.5)
    q_m2 = int(M2 * M_BIT + 0.5)
    q_m3 = int(M3 * M_BIT + 0.5)
    q_m4 = int(M4 * M_BIT + 0.5)

    # ==========================================
    # network description
    # ==========================================

    names = [
        "input",
        "conv1",
        "conv2",
        "conv3",
        "conv4"
    ]

    layers = [
        "input",
        "conv2d",
        "conv2d",
        "conv2d",
        "conv2d"
    ]

    channels = [
        (16, 16),
        (16, 32),
        (32, 16),
        (16, 16),
        (16, 1)
    ]

    shapehv = [
        (9, 9),
        (7, 7),
        (5, 5),
        (3, 3),
        (1, 1)
    ]

    ksizes = [
        (0, 0),
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3)
    ]

    activation = [
        "None",
        "relu",
        "relu",
        "relu",
        "clip"
    ]

    Q_W = [
        None,
        qw1,
        qw2,
        qw3,
        qw4
    ]

    Q_BIAS = [
        None,
        q_bias1,
        q_bias2,
        q_bias3,
        q_bias4
    ]

    Q_M = [
        0,
        q_m1,
        q_m2,
        q_m3,
        q_m4
    ]

    Q_Z = [
        0,
        zw1,
        zw2,
        zw3,
        zw4
    ]

    SCALE = [
        0,
        sw1,
        sw2,
        sw3,
        sw4
    ]

    # ==========================================
    # write txt
    # ==========================================

    with open(save_txtPath, "w") as file:

        for i in range(len(layers)):

            file.write(
                "{} {}\n".format(
                    names[i],
                    layers[i]
                )
            )

            if layers[i] == "input":

                file.write(
                    "shape {} {} {} {}\n".format(
                        channels[i][0],
                        channels[i][1],
                        shapehv[i][0],
                        shapehv[i][1]
                    )
                )

                file.write("\n")
                continue

            # --------------------------------

            file.write(
                "out_ch {}\n".format(
                    channels[i][1]
                )
            )

            file.write(
                "in_ch {}\n".format(
                    channels[i][0]
                )
            )

            file.write(
                "kernel {} {}\n".format(
                    ksizes[i][0],
                    ksizes[i][1]
                )
            )

            file.write(
                "scale {}\n".format(
                    SCALE[i]
                )
            )

            file.write(
                "zero_point {}\n".format(
                    Q_Z[i]
                )
            )

            file.write(
                "multiplier {}\n".format(
                    Q_M[i]
                )
            )

            # --------------------------------
            # weight
            # --------------------------------

            weight = (
                Q_W[i]
                .cpu()
                .numpy()
                .astype(np.int32)
            )

            file.write("weights\n")

            weight_flat = weight.reshape(-1)

            for v in weight_flat:
                file.write("{} ".format(int(v)))

            file.write("\n")

            # --------------------------------
            # bias
            # --------------------------------

            bias = (
                Q_BIAS[i]
                .cpu()
                .numpy()
                .astype(np.int32)
            )

            file.write("bias\n")

            for v in bias:
                file.write("{} ".format(int(v)))

            file.write("\n")

            # --------------------------------

            file.write(
                "activation {}\n".format(
                    activation[i]
                )
            )

            file.write("\n")

    print("save txt success")
    print(save_txtPath)


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    MODEL_PATH = "./model_save/model_qat.pt"

    SAVE_TXT = "./model_save/nnl_coef.txt"

    model = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    state = model.state_dict()

    save_coef_to_txt(
        state,
        SAVE_TXT
    )