import os
import random
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset

from nni.algorithms.compression.pytorch.quantization.am_dorefa_quantizer import AmDoReFaQuantizer
from nni.algorithms.compression.pytorch.quantization.qat_quantizer import QAT_Quantizer
from nni.compression.pytorch.quantization.settings import set_quant_scheme_dtype

from dm_cnn import MosquitoDenoiseCNN, features_list, NORM_DIV, MosquitoPatchDataset, load_data_dirs
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    # Two things should be kept in mind when set this configure_list:
    # 1. When deploying model on backend, some layers will be fused into one layer. For example, the consecutive
    # conv + bn + relu layers will be fused into one big layer. If we want to execute the big layer in quantization
    # mode, we should tell the backend the quantization information of the input, output, and the weight tensor of
    # the big layer, which correspond to conv's input, conv's weight and relu's output.
    # 2. Same tensor should be quantized only once. For example, if a tensor is the output of layer A and the input
    # of the layer B, you should configure either {'quant_types': ['input'], 'op_names': ['b']} or
    # {'quant_types': ['output'], 'op_names': ['a']} in the configure_list.

    #model_pth = "./model_save/"
    # model_save_path = "./model_save/"

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    SCRIPT_DIR = os.path.dirname(__file__)
    MODEL_PATH = os.path.join(SCRIPT_DIR, f"./model/mosquito_denoise_cnn_cost_down_grid_{GS}.pth")
    model_save_path = "./model/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### change
    #-----------------------------
    configure_list = [
        {
            'quant_types': ['weight', 'input'],
            'quant_bits': {'weight': 8, 'input': 8},
            'op_names': ['conv1']
        }, {
            'quant_types': ['output'],
            'quant_bits': {'output': 8},
            'op_names': ['relu1', 'relu2', 'relu3', 'sigmoid']
        },
        {
            'quant_types': ['weight'],
            'quant_bits': {'weight': 8},
            'op_names': ['conv2', 'conv3', 'conv4']
        }
    ]

    output_range_dict = {"relu1": [0, 1], "relu2": [0, 1], "relu3": [0, 1], "sigmoid": [0, 1]}

    ### replace with your own model
    model = MosquitoDenoiseCNN(COST_DOWN).to(device)
    # state = torch.load(os.path.join(model_pth, 'model.pth')) l
    # print(model)
    # print(state)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    # model.to(device)

    if COST_DOWN:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-6, weight_decay=1e-4)
        max_grad_norm = 1.0
        label_smoothing = 0.05
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-5
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        max_grad_norm = None
        label_smoothing = 0.0
        scheduler = None

    ### don't modify
    quantizer = AmDoReFaQuantizer(model, configure_list, optimizer, output_range_dict)
    quantizer.compress()
    # print(model)

    ### change
    model_train(model, device, epochs=10)  # 200

    # torch.save(model, os.path.join(model_save_path, f"mosquito_denoise_cnn_gat_grid_{GS}.pth"))


if __name__ == '__main__':
    main()
