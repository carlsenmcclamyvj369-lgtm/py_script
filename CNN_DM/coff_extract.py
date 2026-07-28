import os
import copy
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def quantize_input(inputs, f_range=[0, 1], i_range=[0, 255]):
    inputs = torch.clamp(inputs, 0, 1)
    scale_factor = (f_range[1] - f_range[0]) / (i_range[1] - i_range[0])
    zp = i_range[1] - round(f_range[1] / scale_factor)
    xq = torch.round(inputs / scale_factor) + zp
    # x = (xq - zp) * scale_factor #dequantize

    return xq, scale_factor, zp


def quantize_weight(x):
    min_val, max_val = torch.min(x).item(), torch.max(x).item()
    scale_factor = (max_val - min_val) / (255 - 0)  # adla [0, 254]
    zp = 255 - round(max_val / scale_factor)
    xq = torch.round(x / scale_factor) + zp
    # x = (xq - zp) * scale_factor #dequantize

    return xq, scale_factor, zp


def quantize_output(x, f_range=[0, 1], i_range=[0, 255]):
    output = torch.clamp(x, f_range[0], f_range[1])  # for relu, clip output range to [0, 1]
    scale_factor = (f_range[1] - f_range[0]) / (i_range[1] - i_range[0])
    # scale_factor = torch.tensor(scale_factor).type_as(x)
    zp = i_range[1] - round(f_range[1] / scale_factor)
    xq = torch.round(output / scale_factor) + zp
    xq = torch.clamp(xq, i_range[0], i_range[1])
    # x = (xq - zp) * scale_factor #dequantize

    return xq, scale_factor, zp


def save_coef_to_txt(state, save_txtPath):

    sin = 1 / 255.0
    zin = 0

    so1 = 1 / 255.0
    zo1 = 0

    so2 = 1 / 255.0
    zo2 = 0

    so3 = 1 / 255.0
    zo3 = 0

    so4 = 1 / 255.0
    zo4 = 0

    f_range = [0, 1]
    i_range = [0, 255]
    scale_in = (f_range[1] - f_range[0]) / (i_range[1] - i_range[0])
    zx = i_range[1] - round(f_range[1] / scale_in)

    f_range = [0, 1]
    i_range = [0, 255]
    scale_feature = (f_range[1] - f_range[0]) / (i_range[1] - i_range[0])
    zf = i_range[1] - round(f_range[1] / scale_feature)

    conv1_weight = state['conv1.module.weight']
    conv2_weight = state['conv2.module.weight']
    conv3_weight = state['conv3.module.weight']
    conv4_weight = state['conv4.module.weight']

    conv1_bias = state['conv1.module.bias']
    conv2_bias = state['conv2.module.bias']
    conv3_bias = state['conv3.module.bias']
    conv4_bias = state['conv4.module.bias']

    qw1, sw1, zw1 = quantize_weight(conv1_weight)
    qw2, sw2, zw2 = quantize_weight(conv2_weight)
    qw3, sw3, zw3 = quantize_weight(conv3_weight)
    qw4, sw4, zw4 = quantize_weight(conv4_weight)

    bias_bit = 2**12  # adjust
    q_bias1 = torch.floor(bias_bit * conv1_bias / (sw1 * sin) + 0.5)
    q_bias2 = torch.floor(bias_bit * conv2_bias / (sw2 * so1) + 0.5)
    q_bias3 = torch.floor(bias_bit * conv3_bias / (sw3 * so2) + 0.5)
    q_bias4 = torch.floor(bias_bit * conv4_bias / (sw4 * so3) + 0.5)

    M1 = sw1 * sin / so1
    M2 = sw2 * so1 / so2
    M3 = sw3 * so2 / so3
    M4 = sw4 * so3 / so4

    M_bit = (2**20)  # adjust
    q_m1 = int(M1 * M_bit + 0.5)
    q_m2 = int(M2 * M_bit + 0.5)
    q_m3 = int(M3 * M_bit + 0.5)
    q_m4 = int(M4 * M_bit + 0.5)

    M1_dif = q_m1 / M_bit - M1
    M2_dif = q_m2 / M_bit - M2
    M3_dif = q_m3 / M_bit - M3
    M4_dif = q_m4 / M_bit - M4

    print("q_M1={}, q_M2={}, q_M3={}, q_M4={}".format(q_m1, q_m2, q_m3, q_m4))
    print("M1_dif={}, M2_dif={}, M3_dif={}, M4_dif={}".format(M1_dif, M2_dif, M3_dif, M4_dif))
    print("qw1, qw2, qw3, qw4 ", qw1.shape, qw2.shape, qw3.shape, qw3.shape)
    print("zw1={}, zw2={}, zw3={}, zw4={}".format(zw1, zw2, zw3, zw4))
    print("zxl={}, zx2={}, zx3={}, zx4={}".format(zx, zx, zx, zx))
    print("zf1={}, zf2={}, zf3={}, zf4={}".format(zf, zf, zf, zf))
    print("M1={}, M2={}, M3={}, M4={}".format(M1, M2, M3, M4))

    print(conv1_bias.shape, conv2_bias.shape, conv3_bias.shape, conv4_bias.shape)
    # print("q_bias1 = \n", q_bias1)
    # print("q_bias2 = \n", q_bias2)
    # print("q_bias3 = \n", q_bias3)
    # print("q_bias4 = \n", q_bias4)

    #           # input   conv1    conv2    conv3    conv4
    names = ['input', 'conv1', 'conv2', 'conv3', 'conv4']
    layers = ['input', 'conv2d', 'conv2d', 'conv2d', 'conv2d']
    strides = [0, 1, 1, 1, 1]

    #           # input   conv1    conv2    conv3    conv4
    shapehv = [(9, 9), (7, 7), (5, 5), (3, 3), (1, 1)]
    channels = [(16, 16), (16, 32), (32, 16), (16, 16), (16, 1)]
    ksizes = [(0, 0), (3, 3), (3, 3), (3, 3), (3, 3)]
    activation = ['None', 'relu', 'relu', 'relu', 'None']
    padding = [0, 0, 0, 0, 0]
    pad_mode = ['None', 'None', 'None', 'None', 'None']

    # list
    #           #input  conv1   conv2   conv3   conv4
    scale = [0, sw1, sw2, sw3, sw4]
    QZ = [0, zw1, zw2, zw3, zw4]
    Qi = [0, zx, zx, zx, zx]
    Qf = [0, zf, zf, zf, zf]
    Q_M = [0, q_m1, q_m2, q_m3, q_m4]
    Q_w = [0, qw1, qw2, qw3, qw4]
    Q_bia = [0, q_bias1, q_bias2, q_bias3, q_bias4]

    with open(save_txtPath, 'w') as file:
        for i in range(len(layers)):
            file.write("{} {}\n".format(names[i], layers[i]))

            if layers[i] == "input":
                print("idx = {}, layers = {}".format(i, layers[i]))
                file.write("shape {} {} {} {}\n".format(channels[i][0], channels[i][1], shapehv[i][0], shapehv[i][1]))
                file.write('\n')

            elif layers[i] == "conv2d":
                print("idx = {}, layers = {}".format(i, layers[i]))
                file.write("weights {} {} {} {} {} {} {} {}\n".format(
                    channels[i][0], channels[i][1], ksizes[i][0], ksizes[i][1],
                    strides[i], padding[i], pad_mode[i]
                ))

                # write weights
                if (ksizes[i][0] == 3):
                    conv_weights = Q_w[i].view(-1, 9).cpu().numpy()
                else:
                    conv_weights = Q_w[i].squeeze().cpu().numpy()

                print("conv_weights shape = ", conv_weights.shape)
                for r in range(conv_weights.shape[0]):
                    for c in range(conv_weights.shape[1]):
                        if c < conv_weights.shape[1] - 1:
                            file.write('{} '.format(int(conv_weights[r][c])))
                        else:
                            file.write('{}'.format(int(conv_weights[r][c])))
                    file.write('\n')

                # write bias
                file.write("{} {}\n".format("bias", channels[i][1]))
                conv_bias = Q_bia[i].cpu().numpy()
                for r in range(conv_bias.shape[0]):
                    if r < conv_bias.shape[0] - 1:
                        file.write('{} '.format(int(conv_bias[r])))
                    else:
                        file.write('{}'.format(int(conv_bias[r])))
                file.write("\n\n")

                # write activation
                file.write("{} {}\n".format(names[i], activation[i]))

                if (activation[i] != "None"):
                    file.write("weights {} {} {} {}\n".format(channels[i][1], channels[i][1], shapehv[i][0], shapehv[i][1]))
                    file.write("\n\n")

    file.write("\n\n")
    print("save txt.....")


if __name__ == "__main__":

    model_path = "./model/mosquito_denoise_cnn_qat.pth"
    save_txtPath = "./model/cnn_coef.txt"

    model = torch.load(model_path)
    state = model.state_dict()
    # print(state.keys())

    # 'conv1.module.bias', 'conv1.module.weight',
    # 'conv2.module.bias', 'conv2.module.weight',
    # 'conv3.module.bias', 'conv3.module.weight',
    # 'conv4.module.bias', 'conv4.module.weight',

    save_coef_to_txt(state, save_txtPath)